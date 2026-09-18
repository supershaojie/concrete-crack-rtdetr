"""Strict, bounded checkpoint and live-FP32 continuation diagnostics.

All comparisons use the declared tolerances below. Raw continuation failures are
retained even when an identical-gradient diagnostic succeeds. No training entry
point imports this module; it never changes the production optimizer or scaler.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
import gc
import hashlib
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from dcc_common import require, write_json, sha256, native_rebuild
from c19_lif_v1_probe import capture, targets
from c19_lif_v1_diagnostic import rng_state, restore_rng
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import ModelEMA

ATOL, RTOL = 2e-5, 2e-4


def attach_mode_acceptance(checkpoint_report, acceptance):
    """Annotate a newly generated report; never rewrite raw tensor findings."""
    checkpoint_report.setdefault("raw_status", checkpoint_report["status"])
    checkpoint_report.setdefault("raw_admission", checkpoint_report.get("admission"))
    checkpoint_report["contract_version"] = acceptance["contract_version"]
    checkpoint_report["status"] = acceptance["status"]
    checkpoint_report["checkpoint_correctness"] = deepcopy(acceptance["restoration"])
    checkpoint_report["trajectory_repeatability"] = deepcopy(acceptance["trajectory"])
    checkpoint_report["admission"] = "REQUIRES_FULL_ENGINEERING_GATE"
    return checkpoint_report


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def optimizer_layout(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    require(set(actual) == {id(p) for p in model.parameters() if p.requires_grad}, "Optimizer coverage differs")
    require(all(count == 1 for count in Counter(actual).values()), "Optimizer duplicates parameters")
    return [dict(parameters=[names[id(p)] for p in group["params"]],
                 hyperparameters=_cpu({key: value for key, value in group.items() if key != "params"}))
            for group in optimizer.param_groups]


def snapshot(trainer):
    """Named, independent CPU copies of every persistent continuation state."""
    names = {id(parameter): name for name, parameter in trainer.model.named_parameters()}
    return dict(
        parameters=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad)
                    for n, p in trainer.model.named_parameters()],
        model=_cpu(trainer.model.state_dict()),
        model_buffers=_cpu(dict(trainer.model.named_buffers())),
        optimizer_groups=optimizer_layout(trainer.model, trainer.optimizer),
        optimizer={names[id(p)]: _cpu(state) for p, state in trainer.optimizer.state.items()},
        scaler=_cpu(trainer.scaler.state_dict()), scaler_enabled=trainer.scaler.is_enabled(),
        ema=_cpu(trainer.ema.ema.state_dict()), ema_buffers=_cpu(dict(trainer.ema.ema.named_buffers())),
        ema_updates=trainer.ema.updates, epoch=getattr(trainer, "epoch", None),
        start_epoch=getattr(trainer, "start_epoch", None), epochs=getattr(trainer, "epochs", None),
        best_fitness=getattr(trainer, "best_fitness", None))


def checkpoint_expected_state(checkpoint, amp):
    """Construct an independent named oracle from saved bytes without mutating EMA."""
    from dcc_checkpoint import require_checkpoint_policy
    require_checkpoint_policy(checkpoint)
    model = checkpoint["ema"]

    def model_value(value):
        return value.detach().cpu().to(dtype=torch.float32).clone() if value.is_floating_point() else value.detach().cpu().clone()

    state = {name: model_value(value) for name, value in model.state_dict().items()}
    buffers = {name: model_value(value) for name, value in model.named_buffers()}
    names = checkpoint["dcc_checkpoint"]["optimizer_param_names"]
    named_optimizer, groups = {}, []
    for group, parameter_names in zip(checkpoint["optimizer"]["param_groups"], names):
        groups.append(dict(parameters=deepcopy(parameter_names),
                           hyperparameters=_cpu({key: value for key, value in group.items() if key != "params"})))
        for identifier, name in zip(group["params"], parameter_names):
            if identifier in checkpoint["optimizer"]["state"]:
                named_optimizer[name] = _cpu(checkpoint["optimizer"]["state"][identifier])
    require(bool(checkpoint["scaler"]) == amp, "Saved scaler mode differs from requested precision")
    return dict(parameters=[dict(name=name, shape=list(parameter.shape), dtype=str(torch.float32), requires_grad=True)
                            for name, parameter in model.named_parameters()],
                model=state, model_buffers=buffers, optimizer_groups=groups, optimizer=named_optimizer,
                scaler=deepcopy(checkpoint["scaler"]), scaler_enabled=amp,
                ema=_cpu(state), ema_buffers=_cpu(buffers), ema_updates=checkpoint["updates"],
                epoch=None, start_epoch=checkpoint["epoch"] + 1, epochs=checkpoint["train_args"]["epochs"],
                best_fitness=checkpoint["best_fitness"])


def audit_restoration(trainer, checkpoint, amp):
    """Strict saved-byte oracle comparison, reusable after native resume setup."""
    result = compare_states(checkpoint_expected_state(checkpoint, amp), snapshot(trainer), exact=True)
    result["compared"] = "saved EMA/model weights+buffers, optimizer names/group order/all hyperparameters/all states+steps, scaler, epoch, EMA updates"
    require(result["status"] == "PASSED", "Restored state differs from independently mapped saved checkpoint")
    return result


def tensor_metric(left, right):
    left, right = left.detach().cpu(), right.detach().cpu()
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(equal=False, allclose=False, shape_a=list(left.shape), shape_b=list(right.shape),
                    dtype_a=str(left.dtype), dtype_b=str(right.dtype), reason="shape/dtype mismatch")
    equal = bool(torch.equal(left, right))
    if not left.is_floating_point():
        return dict(equal=equal, allclose=equal, shape=list(left.shape), dtype=str(left.dtype),
                    max_abs=0.0 if equal else float((left.long() - right.long()).abs().max()))
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return dict(equal=equal, allclose=False, finite=False, shape=list(left.shape), dtype=str(left.dtype),
                    max_abs=None, relative_L2=None, exceeded_fraction=None)
    original_dtype = str(left.dtype)
    left, right = left.double(), right.double()
    difference = (left - right).abs()
    failed = difference > ATOL + RTOL * right.abs()
    return dict(equal=equal, allclose=bool(not failed.any()), finite=True, shape=list(left.shape), dtype=original_dtype,
                max_abs=float(difference.max()) if difference.numel() else 0.,
                relative_L2=float(torch.linalg.vector_norm(left-right) /
                                  torch.linalg.vector_norm(right).clamp_min(1e-12)),
                exceeded_fraction=float(failed.double().mean()) if failed.numel() else 0., atol=ATOL, rtol=RTOL)


def compare_states(left, right, exact=False):
    rows, metadata_errors = {}, []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            rows[path] = tensor_metric(a, b)
            # Adam step counters must match exactly, including raw comparisons.
            if path.endswith(".step") and not rows[path]["equal"]:
                metadata_errors.append(path + ": step counter differs")
        elif type(a) is not type(b):
            metadata_errors.append(path + ": type differs")
        elif isinstance(a, dict):
            if set(a) != set(b):
                metadata_errors.append(path + ": key set differs")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + "." + str(key))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                metadata_errors.append(path + ": sequence length differs")
            for index, (aa, bb) in enumerate(zip(a, b)):
                visit(aa, bb, path + "." + str(index))
        elif a != b:
            metadata_errors.append(path + ": value differs")

    visit(left, right, "state")
    failed = [name for name, row in rows.items() if not row["equal" if exact else "allclose"]]
    return dict(status="PASSED" if not failed and not metadata_errors else "FAILED", exact_requested=exact,
                exact=not metadata_errors and all(row["equal"] for row in rows.values()),
                allclose=not metadata_errors and all(row["allclose"] for row in rows.values()),
                metadata_errors=metadata_errors, failed_tensors=failed, tensors=rows,
                max_abs=max((row.get("max_abs") or 0. for row in rows.values()), default=0.))


def storage_inventory(trainer):
    """Inspect all reachable tensor attributes, including caches, grads and scaler internals."""
    found, visited = {}, set()

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.numel():
                storage = value.untyped_storage() if hasattr(value, "untyped_storage") else value.storage()
                found[path] = (str(value.device), int(storage.data_ptr()))
            if (value.is_leaf or value.retains_grad) and value.grad is not None:
                visit(value.grad, path + ".grad")
        elif isinstance(value, dict):
            for key, item in value.items():
                # Optimizer state dictionaries have Parameter objects as keys.
                if isinstance(key, torch.Tensor):
                    visit(key, path + ".parameter_key")
                visit(item, path + "." + str(key if not isinstance(key, torch.Tensor) else id(key)))
        elif isinstance(value, (tuple, list, set)):
            for index, item in enumerate(value):
                visit(item, path + "." + str(index))
        elif isinstance(value, (torch.nn.Module, torch.optim.Optimizer, ModelEMA, RTDETRTrainer,
                                torch.cuda.amp.GradScaler, SimpleNamespace)) or type(value).__name__ == "GradScaler":
            for key, item in vars(value).items():
                visit(item, path + "." + key)

    visit(trainer, "trainer")
    return found


def assert_no_shared_storage(first, second):
    a, b = storage_inventory(first), storage_inventory(second)
    shared = set(a.values()) & set(b.values())
    require(not shared, "Independent controls share mutable tensor storage")
    return dict(status="PASSED", checked="all reachable tensor storages: model/cache/grad/optimizer/scaler/EMA",
                first_tensors=len(a), second_tensors=len(b), shared_storage_count=0)


def _state_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _state_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to(item, device) for item in value)
    return value


def move_trainer(trainer, device):
    trainer.model.to(device)
    trainer.ema.ema.to(device)
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in trainer.optimizer.state.get(parameter, {}).items():
                # Native AdamW keeps a CPU step scalar unless capturable/fused.
                destination = device if key != "step" or group.get("capturable") or group.get("fused") else "cpu"
                trainer.optimizer.state[parameter][key] = _state_to(value, destination)
    for key in ("_scale", "_growth_tracker"):
        value = getattr(trainer.scaler, key, None)
        if isinstance(value, torch.Tensor):
            setattr(trainer.scaler, key, value.to(device))
    return trainer


def make_audit_trainer(model, amp, device, trainer_cls=None):
    if trainer_cls is None:
        from dcc_checkpoint import DCCCheckpointTrainer
        trainer_cls = DCCCheckpointTrainer
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.model = model.to(device)
    trainer.args = SimpleNamespace(lr0=.0005, warmup_bias_lr=.1, weight_decay=.0001)
    trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    optimizer_layout(trainer.model, trainer.optimizer)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=amp)
    require(trainer.scaler.is_enabled() == amp, "Requested scaler mode unavailable")
    trainer.ema = ModelEMA(trainer.model)
    return trainer


def clone_live(trainer, amp):
    require(all(p.dtype == torch.float32 for p in trainer.model.parameters()), "Live control requires actual FP32 parameters")
    require(all(value.dtype == torch.float32 for state in trainer.optimizer.state.values()
                for name, value in state.items() if name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}),
            "Live control requires original FP32 moments before any load/cast")
    clone = make_audit_trainer(deepcopy(trainer.model).cpu(), amp, "cpu", type(trainer))
    clone.optimizer.load_state_dict(_cpu(trainer.optimizer.state_dict()))
    clone.scaler.load_state_dict(deepcopy(trainer.scaler.state_dict()))
    clone.ema.ema = deepcopy(trainer.ema.ema).cpu()
    clone.ema.updates = trainer.ema.updates
    for field in ("epoch", "start_epoch", "epochs", "best_fitness"):
        if hasattr(trainer, field):
            setattr(clone, field, deepcopy(getattr(trainer, field)))
    return clone


def _finite_norm(tensor):
    return float(tensor.detach().double().norm()) if bool(torch.isfinite(tensor).all()) else None


def step_once(trainer, batch, amp, gradients=False):
    """One real original-loss update through native optimizer_step and native scaler."""
    require(trainer.scaler.is_enabled() == amp, "Scaler mode changed")
    trainer.model.train()
    trainer.model.nc = 1
    trainer.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch["img"].device.type, enabled=amp):
        predictions, taps = capture(trainer.model, batch["img"], targets(batch))
        loss = trainer.model.loss(batch, preds=predictions)[0]
    require(bool(torch.isfinite(loss).all()), "Nonfinite original detection loss")
    trainer.scaler.scale(loss).backward()
    scale = float(trainer.scaler.get_scale())
    captured = {name: parameter.grad.detach().cpu().clone() / scale
                for name, parameter in trainer.model.named_parameters() if parameter.grad is not None}
    before = sum(float(state.get("step", 0)) for state in trainer.optimizer.state.values())
    trainer.optimizer_step()
    after = sum(float(state.get("step", 0)) for state in trainer.optimizer.state.values())
    row = dict(loss=float(loss.detach()), scale_before=scale, scale_after=float(trainer.scaler.get_scale()),
               effective_update=after > before, skipped=after <= before,
               dcc_grad_norms={name: _finite_norm(value) for name, value in captured.items() if ".dcc." in name},
               gt=int(batch["bboxes"].shape[0]), dn_split=taps.get("dn_split"))
    if gradients:
        row["forward_hashes"] = {key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                                 for key, value in taps.items() if isinstance(value, torch.Tensor)}
        return row, captured, {key: value.clone() for key, value in taps.items() if isinstance(value, torch.Tensor)}
    return row


def bounded_updates(trainer, batch, amp, max_batches=16, target_updates=3):
    require(1 <= target_updates <= max_batches <= 16, "Invalid bounded update budget")
    rows, count = [], 0
    for _ in range(max_batches):
        row = step_once(trainer, batch, amp)
        rows.append(row)
        count += int(row["effective_update"])
        if count >= target_updates:
            break
    require(count >= target_updates, "Insufficient effective native updates within fixed budget")
    if hasattr(trainer.model.model[17], "dcc"):
        require(torch.count_nonzero(trainer.model.model[17].dcc.W_o.weight) > 0, "DCC W_o did not learn")
        require(all(value is not None and value > 0 for value in rows[-1]["dcc_grad_norms"].values()),
                "DCC upstream gradients not activated")
    return dict(status="PASSED", observed_batches=len(rows), effective_updates=count, steps=rows,
                scope="Fixed B2/160 real train batch, original loss; not B16/640 capacity or formal training")


def seed42():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return rng_state()


def _raw_pair(first, second, batch, amp):
    device = batch["img"].device
    initial_rng = seed42()
    move_trainer(first, device)
    restore_rng(initial_rng)
    first_step, first_grads, first_forward = step_once(first, batch, amp, gradients=True)
    move_trainer(first, "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    move_trainer(second, device)
    restore_rng(initial_rng)
    second_step, second_grads, second_forward = step_once(second, batch, amp, gradients=True)
    move_trainer(second, "cpu")
    require(first_step["effective_update"] and second_step["effective_update"], "Next-update comparison was skipped")
    post = compare_states(snapshot(first), snapshot(second))
    gradients_result = compare_states(first_grads, second_grads)
    forward = compare_states(first_forward, second_forward)
    # Anchors contain intentional +inf at invalid positions. Hash equality is
    # required for those sentinels; all other captured tensors must be finite.
    sentinels = {key for key in first_forward if "anchor" in key and
                 not bool(torch.isfinite(first_forward[key]).all())}
    for key in sentinels:
        require(not torch.isnan(first_forward[key]).any() and not torch.isnan(second_forward[key]).any(), "NaN anchor sentinel")
        require(torch.equal(first_forward[key], second_forward[key]), "Anchor sentinel differs")
        row = forward["tensors"]["state." + key]
        row.update(allclose=True, equal=True, note="Matching native invalid-anchor +inf sentinel")
    forward["failed_tensors"] = [key for key, value in forward["tensors"].items() if not value["allclose"]]
    forward["allclose"] = not forward["failed_tensors"] and not forward["metadata_errors"]
    forward["exact"] = not forward["metadata_errors"] and all(value["equal"] for value in forward["tensors"].values())
    forward["status"] = "PASSED" if forward["allclose"] else "FAILED"
    model_rows = {key: value for key, value in post["tensors"].items() if key.startswith("state.model.")}
    require(bool(model_rows), "Missing model tensors in raw continuation comparison")
    require(all(bool(torch.isfinite(value).all()) for value in list(first_grads.values()) + list(second_grads.values())),
            "Nonfinite raw continuation gradient")
    forward_exact = first_step["forward_hashes"] == second_step["forward_hashes"]
    loss_exact = first_step["loss"] == second_step["loss"]
    return dict(status="PASSED" if post["allclose"] and forward["allclose"] and gradients_result["allclose"] and
                forward_exact and loss_exact else "UNRESOLVED",
                raw_next_update_allclose=all(value["allclose"] for value in model_rows.values()),
                raw_all_states_allclose=post["allclose"], state_comparison=post,
                gradient_comparison=gradients_result, captured_forward_comparison=forward,
                captured_forward_hashes_exact=forward_exact,
                loss_exact=loss_exact, direct_step=first_step, resumed_step=second_step), first_grads


def _fixed_gradient_step(trainer, gradients, amp, device):
    move_trainer(trainer, device)
    require(trainer.scaler.is_enabled() == amp, "Replay must retain original native AMP/scaler mode")
    trainer.optimizer.zero_grad(set_to_none=True)
    # Native scale() lazily creates scale/growth tensors. Replaying a captured
    # unscaled gradient requires restoring its scale before native unscale_().
    token = torch.zeros((), device=device, requires_grad=True)
    trainer.scaler.scale(token)
    scale = float(trainer.scaler.get_scale())
    names = dict(trainer.model.named_parameters())
    require(set(gradients) <= set(names), "Replay gradient parameter name mismatch")
    for name, parameter in names.items():
        parameter.grad = gradients[name].to(device=device, dtype=parameter.dtype).clone() * scale if name in gradients else None
    before = sum(float(state.get("step", 0)) for state in trainer.optimizer.state.values())
    trainer.optimizer_step()  # inherited native unscale, clip, scaler.step/update, zero_grad, EMA
    after = sum(float(state.get("step", 0)) for state in trainer.optimizer.state.values())
    require(after > before, "Same-gradient native replay was skipped")
    move_trainer(trainer, "cpu")
    return dict(amp_scaler_path=amp, scaler_enabled=trainer.scaler.is_enabled(), scale_before=scale,
                scale_after=float(trainer.scaler.get_scale()), effective_update=True,
                path="native scale -> assigned scaled gradients -> native optimizer_step/unscale/clip/step/update/EMA")


def audit_amp_overflow_fixture(device, scaler_state):
    """Tiny CUDA-only fixture for native overflow/skip behavior, including EMA."""
    device = torch.device(device)
    require(device.type == "cuda", "Enabled native AMP overflow fixture requires CUDA")
    saved_rng = rng_state()
    try:
        seed42()
        toy = make_audit_trainer(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh()), True, device)
        toy.scaler.load_state_dict(deepcopy(scaler_state))
        gradients = {name: torch.full_like(p, .01, device="cpu") for name, p in toy.model.named_parameters()}
        _fixed_gradient_step(toy, gradients, True, device)
        first, second = clone_live(toy, True), clone_live(toy, True)
        independence = assert_no_shared_storage(first, second)
        before = snapshot(first)
        rows = []
        for current in (first, second):
            move_trainer(current, device)
            token = torch.zeros((), device=device, requires_grad=True)
            current.scaler.scale(token)
            scale = float(current.scaler.get_scale())
            for parameter in current.model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            next(current.model.parameters()).grad.reshape(-1)[0] = float("inf")
            steps = sum(float(item.get("step", 0)) for item in current.optimizer.state.values())
            ema_updates = current.ema.updates
            current.optimizer_step()
            require(sum(float(item.get("step", 0)) for item in current.optimizer.state.values()) == steps,
                    "Injected Inf failed to skip native optimizer update")
            require(float(current.scaler.get_scale()) < scale, "Native scaler did not back off on Inf")
            rows.append(dict(skipped=True, scale_before=scale, scale_after=float(current.scaler.get_scale()),
                             ema_updates_before=ema_updates, ema_updates_after=current.ema.updates))
            move_trainer(current, "cpu")
        result = compare_states(snapshot(first), snapshot(second), exact=True)
        require(result["status"] == "PASSED", "Native overflow copies diverged")
        after = snapshot(first)
        require(compare_states(before["model"], after["model"], exact=True)["status"] == "PASSED", "Skipped model update changed weights")
        require(compare_states(before["optimizer"], after["optimizer"], exact=True)["status"] == "PASSED", "Skipped optimizer changed state")
        require(first.ema.updates == before["ema_updates"] + 1, "Unexpected inherited EMA-on-skipped-step behavior")
        return dict(status="PASSED", fixture="Two independent Linear(3,2)+Tanh FP32 models; no detector forward",
                    amp_scaler_path=True, storage_independence=independence, first=rows[0], second=rows[1],
                    complete_state_comparison=result, model_optimizer_unchanged_on_skip=True,
                    native_ema_updates_on_skipped_step=True,
                    note="Inherited optimizer_step updates EMA even when scaler skips optimizer; observed and preserved")
    finally:
        restore_rng(saved_rng)


def audit_live_control(trainer, batch, amp, folder, label):
    """Two independent copies of truly updated live FP32 state, with no checkpoint quantization."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = next(trainer.model.parameters()).device
    saved_rng = rng_state()
    report = dict(status="FAILED", label=label, seed=42, atol=ATOL, rtol=RTOL,
                  quantization="NONE: copied live FP32 model/optimizer/EMA", actual_updates_required=True)
    try:
        require(bool(trainer.optimizer.state), "Live control requires actually updated optimizer state")
        move_trainer(trainer, "cpu")
        first, second = clone_live(trainer, amp), clone_live(trainer, amp)
        report["storage_independence"] = assert_no_shared_storage(first, second)
        report["original_first_independence"] = assert_no_shared_storage(trainer, first)
        report["initial_state"] = compare_states(snapshot(first), snapshot(second), exact=True)
        require(report["initial_state"]["status"] == "PASSED", "Live controls start from different state")
        raw, _ = _raw_pair(first, second, batch, amp)
        from dcc_acceptance import CONTRACT_VERSION
        report.update(raw, raw_status=raw["status"], contract_version=CONTRACT_VERSION,
                      admission="REQUIRES_MODE_ACCEPTANCE")
        del first, second
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        move_trainer(trainer, device)
        restore_rng(saved_rng)
        write_json(folder / "live_control.json", report)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def _restore_checkpoint(checkpoint, path, amp, trainer_cls, native):
    from dcc_checkpoint import require_checkpoint_policy
    require_checkpoint_policy(checkpoint)
    if native:
        trainer = trainer_cls.__new__(trainer_cls)
        trainer.args = SimpleNamespace(**deepcopy(checkpoint["train_args"]))
        trainer.model, trainer.data = str(path), dict(nc=1, channels=3)
        trainer.epochs, trainer.resume = 200, True
        loaded = trainer.setup_model()
        require_checkpoint_policy(loaded)
        # Build the optimizer using identical native grouping on the new model.
        trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
        trainer.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        trainer.ema = ModelEMA(trainer.model)
        trainer.resume_training(loaded)
        return trainer
    weights = deepcopy(checkpoint["ema"]).float()
    model = native_rebuild(weights.yaml, weights, nc=1, channels=3)
    trainer = make_audit_trainer(model, amp, "cpu", trainer_cls)
    trainer.optimizer.load_state_dict(deepcopy(checkpoint["optimizer"]))
    trainer.scaler.load_state_dict(deepcopy(checkpoint["scaler"]))
    trainer.ema.ema.load_state_dict(weights.state_dict(), strict=True)
    trainer.ema.updates = checkpoint["updates"]
    trainer.start_epoch, trainer.epochs = checkpoint["epoch"] + 1, 200
    trainer.best_fitness = checkpoint["best_fitness"]
    return trainer


def audit_checkpoint_resume(trainer, batch, amp, folder, trainer_cls=None):
    """Strict native restoration + raw step + native same-gradient AMP replay.

    The supplied trained live state is left unchanged. Only owned temporary
    checkpoint files are removed; JSON evidence is retained in ``folder``.
    """
    from dcc_checkpoint import DCCCheckpointTrainer, optimizer_param_names, require_checkpoint_policy
    from dcc_acceptance import CONTRACT_VERSION, evaluate_checkpoint
    trainer_cls = trainer_cls or DCCCheckpointTrainer
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    device = next(trainer.model.parameters()).device
    saved_rng = rng_state()
    report = dict(status="FAILED", contract_version=CONTRACT_VERSION, seed=42, atol=ATOL, rtol=RTOL, native_setup_model=True,
                  native_resume_training=True, completed_training_epochs=0, diagnostic_epoch_metadata=0,
                  formal_training="NOT_STARTED", final_test="NOT_RUN")
    try:
        move_trainer(trainer, "cpu")
        with tempfile.TemporaryDirectory(prefix="_resume_audit_", dir=str(folder)) as temporary:
            temporary = Path(temporary).resolve()
            require(temporary.parent == folder, "Unexpected temporary checkpoint location")
            proxy = copy(trainer)
            proxy.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, "model": str(temporary / "last.pt"),
                                           "epochs": 200, "close_mosaic": 10, "data": "diagnostic_only", "task": "detect"})
            proxy.epoch, proxy.best_fitness, proxy.fitness = 0, 0., 0.
            proxy.metrics, proxy.save_period = {}, -1
            proxy.wdir, proxy.last, proxy.best, proxy.csv = temporary, temporary / "last.pt", temporary / "best.pt", temporary / "absent.csv"
            trainer_cls.save_model(proxy)
            checkpoint = torch_load(proxy.last, map_location="cpu")
            report["checkpoint_source_validation"] = require_checkpoint_policy(checkpoint)
            report["checkpoint_policy"] = "optimizer_fp32_v1"
            report["saved_ema_dtypes"] = sorted({str(value.dtype) for value in checkpoint["ema"].state_dict().values()})
            report["checkpoint_sha256"] = sha256(proxy.last)
            report["serializer"] = deepcopy(checkpoint.get("dcc_checkpoint", {}))
            require(report["serializer"].get("policy") == "optimizer_fp32_v1", "Expected explicit FP32 optimizer serializer")
            require(report["serializer"]["optimizer_param_names"] == optimizer_param_names(trainer.model, trainer.optimizer),
                    "Serialized optimizer names/order differ")
            report["saved_optimizer_live_exact"] = compare_states(_cpu(trainer.optimizer.state_dict()), checkpoint["optimizer"], exact=True)
            require(report["saved_optimizer_live_exact"]["status"] == "PASSED", "Serializer changed live optimizer state")
            direct = _restore_checkpoint(checkpoint, proxy.last, amp, trainer_cls, native=False)
            resumed = _restore_checkpoint(checkpoint, proxy.last, amp, trainer_cls, native=True)
            require(optimizer_param_names(direct.model, direct.optimizer) == report["serializer"]["optimizer_param_names"] ==
                    optimizer_param_names(resumed.model, resumed.optimizer), "Restored optimizer parameter name/order differs")
            restoration = compare_states(snapshot(direct), snapshot(resumed), exact=True)
            restoration["direct_vs_saved_checkpoint"] = audit_restoration(direct, checkpoint, amp)
            restoration["native_vs_saved_checkpoint"] = audit_restoration(resumed, checkpoint, amp)
            restoration["storage_independence"] = assert_no_shared_storage(direct, resumed)
            restoration["checkpoint_epoch"] = checkpoint["epoch"]
            restoration["restored_start_epoch"] = resumed.start_epoch
            require(resumed.start_epoch == checkpoint["epoch"] + 1, "Native epoch restoration differs")
            require(restoration["status"] == "PASSED", "Complete checkpoint restoration mismatch")
            report["restoration"] = restoration
            # CPU copies of original restored states are made before either raw update.
            replay_a, replay_b = clone_live(direct, amp), clone_live(resumed, amp)
            replay_storage = assert_no_shared_storage(replay_a, replay_b)
            raw, fixed_gradients = _raw_pair(direct, resumed, batch, amp)
            report.update(raw)
            del direct, resumed
            a = _fixed_gradient_step(replay_a, fixed_gradients, amp, device)
            b = _fixed_gradient_step(replay_b, fixed_gradients, amp, device)
            replay = compare_states(snapshot(replay_a), snapshot(replay_b), exact=True)
            replay.update(amp_scaler_path=amp, native_scaler_mode_preserved=True,
                          storage_independence=replay_storage, first=a, second=b,
                          compared="parameter names/group order/hyperparameters/model/all optimizer states including step/scaler/epoch/EMA weights+updates",
                          scope="Diagnostic only; success never changes raw continuation result")
            report["same_gradient_replay"] = replay
            require(replay["status"] == "PASSED", "Native same-gradient replay state mismatch")
            if amp:
                replay["overflow_fixture"] = audit_amp_overflow_fixture(device, checkpoint["scaler"])
            report["raw_status"] = report["status"]
            report["raw_admission"] = "PASSED" if report["raw_status"] == "PASSED" else "BLOCKED"
            label = "cpu_fp32" if device.type == "cpu" else "cuda_native_amp" if amp else "cuda_fp32"
            report["checkpoint_correctness"] = evaluate_checkpoint(report, label)
            report["trajectory_repeatability"] = dict(status="PENDING", reason="Requires matching parent/DCC live controls")
            if report["checkpoint_correctness"]["status"] == "FAILED":
                report["status"] = "FAILED"
            report["admission"] = "REQUIRES_FULL_ENGINEERING_GATE"
            report["temporary_checkpoints"] = "removed after audit; hashes retained"
            del replay_a, replay_b, checkpoint, proxy
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), admission="BLOCKED")
        raise
    finally:
        move_trainer(trainer, device)
        restore_rng(saved_rng)
        write_json(folder / "resume_comparison.json", report)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report
