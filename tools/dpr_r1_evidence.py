"""Bounded factual collectors for approved R1; no production model overrides."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import pickle
import random
import shutil
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from init_dpr import require, sha256
from preflight_dpr import (snapshot, compare, storages, clone_live, move, seed42, one_step, cpu_copy, TARGET, NEW_NAMES)
from dpr_diagnostics import (GradientObserver, gradient_mapping_check, transpose_maps, capture_trace, compare_traces,
                             backend_audit, half_decomposition, state_fingerprint, metric)
from dpr_acceptance import policy, scope, UNSUPPORTED


def file_record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((value.dtype, tuple(value.shape))).encode() + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def batch_hash(batch):
    return hashlib.sha256(pickle.dumps([(k, tensor_hash(v)) for k, v in sorted(batch.items())])).hexdigest()


def starting_state(trainer, batch, amp):
    return dict(python_rng=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                numpy_rng=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                cpu_rng=tensor_hash(torch.get_rng_state()),
                cuda_rng=[tensor_hash(v) for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                batch_sha256=batch_hash(batch), amp=amp, scaler_enabled=trainer.scaler.is_enabled(),
                scaler_state=cpu_copy(trainer.scaler.state_dict()),
                model_dtypes=sorted({str(p.dtype) for p in trainer.model.parameters()}),
                autocast_dtype=str(torch.get_autocast_gpu_dtype()) if amp else "torch.float32",
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic=torch.are_deterministic_algorithms_enabled(), warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


@contextmanager
def fp32_inference_flags():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def differences(left, right):
    if isinstance(left, torch.Tensor):
        return left.double() - right.double() if left.is_floating_point() else left - right
    if isinstance(left, dict): return {k: differences(v, right[k]) for k, v in left.items()}
    if isinstance(left, (tuple, list)): return type(left)(differences(a, b) for a, b in zip(left, right))
    return (left, right)


@contextmanager
def native_update_capture(trainer, destination, records, replay_states):
    """Instrument only the call boundary. Native update order/code stays untouched."""
    require("optimizer_step" not in trainer.__dict__, "Unexpected instance update override")
    native = type(trainer).optimizer_step
    calls = []
    def capture():
        require(not calls, "Multiple update calls in one B run")
        calls.append(1)
        require(not getattr(trainer.scaler, "_per_optimizer_states", {}), "Capture occurred after native unscale")
        pre = snapshot(trainer)
        scaled = {n: cpu_copy(p.grad) for n, p in trainer.model.named_parameters() if p.grad is not None}
        # Exclude this boundary wrapper from deepcopy; observer hooks never run
        # during optimizer replay, and no observer closures are serialized.
        del trainer.optimizer_step
        try:
            replay = clone_live(trainer)
        finally:
            trainer.optimizer_step = capture
        # GradScaler.__getstate__ intentionally serializes lazy CPU scalars and
        # drops the live CUDA tensors. Restore these independent tensors before
        # replaying an already completed backward (no second scale/unscale).
        for key in ("_scale", "_growth_tracker"):
            value = getattr(trainer.scaler, key, None)
            if isinstance(value, torch.Tensor): setattr(replay.scaler, key, value.detach().clone())
        for name, parameter in replay.model.named_parameters():
            parameter.grad = scaled[name].to(parameter.device).clone() if name in scaled else None
        independent = not (storages(replay) & storages(trainer))
        pre_exact = compare(pre, snapshot(replay), exact=True)["exact"]
        grad_exact = compare(scaled, {n: cpu_copy(p.grad) for n, p in replay.model.named_parameters() if p.grad is not None}, exact=True)["exact"]
        native(trainer)
        actual = snapshot(trainer)
        native(replay)
        replayed = snapshot(replay)
        record = dict(capture_stage="after_backward_before_native_optimizer_step_scaled", native_calls=len(calls),
                      storage_independent=independent, scaled_gradient_names=sorted(scaled),
                      pre_state_exact=pre_exact, scaled_gradients_exact=grad_exact,
                      post_comparison=compare(actual, replayed, exact=True))
        # Full raw tensors stay local to the evidence host, never in light packages.
        torch.save(dict(pre=pre, original_scaled_gradients=scaled, actual_post=actual,
                        model_modes={n: m.training for n, m in trainer.model.named_modules()},
                        parameter_requires_grad={n: p.requires_grad for n, p in trainer.model.named_parameters()},
                        purpose="R1 B6 post-backward causal update replay; not a resumable training checkpoint"), destination)
        record["artifact"] = file_record(destination)
        records.append(record); replay_states.append(replayed)
        del replay, pre, actual, scaled
        gc.collect()
    trainer.optimizer_step = capture
    try:
        yield
        require(len(calls) == 1, "No native update observed")
    finally:
        del trainer.optimizer_step


def fp64_adjoint_reference(target):
    local = deepcopy(target).cpu().double()
    generator = torch.Generator().manual_seed(721)
    upstream = torch.randn(local.conv.weight.shape, dtype=torch.float64, generator=generator)
    actual = torch.autograd.grad((local.get_equivalent_kernel() * upstream).sum(),
                                 [getattr(local, n) for n in local.parameter_names])
    expected = transpose_maps(upstream)
    rows = {}
    for name, a in zip(local.parameter_names, actual):
        b = expected[name]
        delta = (a-b).abs()
        exceeded = delta > 1e-10 + 1e-9*b.abs()
        rows[name] = dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                         raw_allclose=not bool(exceeded.any()), max_abs=float(delta.max()),
                         relative_L2=float((a-b).norm()/b.norm().clamp_min(1e-30)), exceed_fraction=float(exceeded.double().mean()),
                         atol=1e-10, rtol=1e-9, dtype="torch.float64", statistics_version="allclose_right_reference_v2")
    return dict(comparisons=rows, reference="independent transpose maps versus native FP64 autograd; no FP32 metric cast")


def native_probe_arguments(trainer):
    """Match production argument container; keep all 109 values unchanged.

    The older engineering fixture used SimpleNamespace, which cannot be passed
    to native strip_optimizer's dict(model.args). Production uses this iterable
    namespace. No production serializer/model function is overridden.
    """
    from ultralytics.utils import IterableSimpleNamespace
    before = deepcopy(vars(trainer.args))
    trainer.args = IterableSimpleNamespace(**before)
    trainer.model.args = trainer.args
    trainer.ema.ema.args = deepcopy(trainer.args)
    require(vars(trainer.args) == before, "Probe argument values changed")


def live_pair_r1(trainer, batch, amp, device, output, candidate=False):
    move(trainer, "cpu")
    copies = [clone_live(trainer), clone_live(trainer)]
    independent = not (storages(copies[0]) & storages(copies[1])) and all(not (storages(trainer) & storages(c)) for c in copies)
    initial = compare(snapshot(copies[0]), snapshot(copies[1]), exact=True)
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    extra = dict(inventory=dict(parameters=list(names.values()), buffers=list(dict(trainer.model.named_buffers())),
                               groups=[[names[id(p)] for p in g["params"]] for g in trainer.optimizer.param_groups]),
                 initial_comparison=initial, starts=[], native_replays=[])
    results, gradients, forwards, replay_states, kernels, added, runs = [], [], [], [], [], [], []
    for index, current in enumerate(copies):
        move(current, device); seed42()
        extra["starts"].append(starting_state(current, batch, amp))
        with native_update_capture(current, output / ("native_replay_"+str(index)+".pt"), extra["native_replays"], replay_states):
            if candidate:
                with GradientObserver(current.model) as observation:
                    row, grad, forward = one_step(current, batch, amp)
            else:
                row, grad, forward = one_step(current, batch, amp)
        if candidate:
            mapping, kernel, groups = observation.evidence(grad, row["scale_before"], amp, device)
            runs.append(mapping); kernels.append(kernel); added.append(groups)
            torch.save(dict(module_initial=cpu_copy(observation.initial.state_dict()), captured=observation.values,
                            scale=row["scale_before"], actual_unscaled_groups=groups, actual_unscaled_W=grad[TARGET+"conv.weight"]),
                       output / ("local_gradient_"+str(index)+".pt"))
            runs[-1]["artifact"] = file_record(output / ("local_gradient_"+str(index)+".pt"))
            del observation
        results.append(row); gradients.append(grad); forwards.append(forward)
        move(current, "cpu")
        if device == "cuda": torch.cuda.empty_cache()
    a, b = snapshot(copies[0]), snapshot(copies[1])
    state, gradient, forward = compare(a, b), compare(gradients[0], gradients[1]), compare(forwards[0], forwards[1], exact=True)
    fields = ("buffers", "ema_buffers", "groups", "scaler", "scaler_enabled", "updates", "epoch", "start_epoch", "epochs", "best_fitness")
    exact = compare({k:a[k] for k in fields}, {k:b[k] for k in fields}, exact=True)
    extra["replay_delta_comparison"] = compare(differences(a, b), differences(*replay_states), exact=True)
    control = dict(initial_exact=initial["exact"], storage_independent=independent, forward_exact=forward["exact"],
                   loss_exact=results[0]["loss"] == results[1]["loss"], exact_metadata_buffers=exact["exact"],
                   raw_allclose=state["raw_allclose"] and gradient["raw_allclose"], state=state, gradients=gradient,
                   forward=forward, steps=results, atol=2e-5, rtol=2e-4, quantization="NONE; independent live FP32 copies")
    extra["metadata_comparison"] = exact
    if candidate:
        extra["mapping"] = dict(runs=runs, difference_propagation=gradient_mapping_check(kernels[0]-kernels[1],
            {n:added[0][n]-added[1][n] for n in added[0]}), fp64_reference=fp64_adjoint_reference(trainer.model.get_submodule(TARGET.rstrip("."))))
        flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        with fp32_inference_flags():
            captures, selections = [], []
            for current in copies:
                move(current, device); current.model.float().eval()
                capture, selection = capture_trace(current.model, batch["img"].float())
                captures.append(capture); selections.append(selection)
                if current is copies[0]: move(current, "cpu")
            extra["updated_function"] = compare_traces(*captures, copies[1].model, batch["img"].float())
            extra["updated_function"]["actual_selections"] = selections
        extra["updated_function"]["tf32_restored"] = flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    for current in copies: move(current, "cpu")
    del copies, a, b, replay_states, gradients, forwards
    gc.collect()
    return control, extra


def native_ema_validation(trainer, batch, output):
    """Run inherited validator __call__/preprocess/loss/postprocess once; no AP."""
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.nn.modules.dpr import DPRConvNormLayer
    from ultralytics.nn.modules.block import ConvNormLayer
    started = time.monotonic()
    original = snapshot(trainer)
    copy = clone_live(trainer); move(copy, "cuda")
    copy.amp = True; copy.world_size = 1
    copy.stopper = SimpleNamespace(possible_stop=False)
    copy.loss_items = torch.zeros(3, device="cuda")
    copy.loss_names = ("giou_loss", "cls_loss", "l1_loss")
    original_ema = cpu_copy(copy.ema.ema.state_dict())
    record = dict(actual_native_call=False, batches=0, ground_truth=int(batch["bboxes"].shape[0]), input_dtypes=[])
    class OneBatch:
        dataset = range(len(batch["img"]))
        def __len__(self): return 1
        def __iter__(self):
            data = cpu_copy(batch)
            data["img"] = (data["img"] * 255).round().to(torch.uint8)
            yield data
    class Probe(RTDETRValidator):
        # Only aggregate metric/report sinks are replaced. Native inference,
        # actual GT loss, dtype transitions, and postprocess are inherited.
        def init_metrics(self, model):
            target = model.get_submodule(TARGET.rstrip("."))
            record.update(unfused=isinstance(target, DPRConvNormLayer) and not target.deployed,
                          norm_retained=isinstance(target.norm, torch.nn.BatchNorm2d),
                          all_q_nonzero=all(bool(getattr(target, n).abs().sum()) for n in target.parameter_names),
                          parameter_dtypes=sorted({str(v.dtype) for v in model.parameters()}),
                          ema_before_native_dtype=state_fingerprint(model))
        def update_metrics(self, preds, data):
            record["batches"] += 1
            record["input_dtypes"].append(str(data["img"].dtype))
            record["finite_postprocess"] = all(bool(torch.isfinite(v).all()) for p in preds for v in p.values() if isinstance(v, torch.Tensor))
        def get_stats(self): return {}
        def gather_stats(self): pass
        def finalize_metrics(self): pass
        def print_results(self): pass
    validator = Probe(dataloader=OneBatch(), save_dir=output / "epoch_probe", args=dict(imgsz=160, batch=2, plots=False, save_json=False))
    validator(trainer=copy)
    record.update(actual_native_call=type(validator).__call__ is RTDETRValidator.__call__,
                  finite_loss=bool(torch.isfinite(validator.loss).all()), returned_float32=all(p.dtype == torch.float32 for p in copy.ema.ema.parameters()),
                  native_half_roundtrip_exact=compare({n:v.half().float() if v.is_floating_point() else v for n,v in original_ema.items()},
                                                     cpu_copy(copy.ema.ema.state_dict()), exact=True)["exact"])
    move(copy, "cpu")
    copy.wdir = output / "epoch_native_checkpoint"
    copy.last, copy.best, copy.csv = copy.wdir / "last.pt", copy.wdir / "best.pt", output / "missing.csv"
    copy.save_model()
    record["checkpoint"] = file_record(copy.last)
    record["original_trainer_unchanged"] = compare(original, snapshot(trainer), exact=True)["exact"]
    record["elapsed_seconds"] = time.monotonic() - started
    return record, copy.last


@contextmanager
def half_budget(label, output):
    def expired():
        from train_dpr import atomic_json
        atomic_json(output / "half_timeout.json", dict(status="BLOCKED", path=label, reason="H path exceeded 120 seconds", admission_eligible=False))
        print("BLOCKED: " + label + " exceeded 120 seconds", flush=True)
        os._exit(124)
    timer = threading.Timer(120, expired); timer.daemon = True; timer.start()
    try: yield
    finally: timer.cancel()


def half_paths(trainer, batch, output):
    from ultralytics.utils.torch_utils import strip_optimizer
    from train_dpr import atomic_json
    result = dict(policy=policy(), scope=scope())
    with half_budget("H0", output):
        result["H0"], checkpoint = native_ema_validation(trainer, batch, output)
    atomic_json(output / "half_progress.json", result)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(raw["model"] is None and raw["ema"] is not None, "Not an actual native EMA file")
    source = raw["ema"].to("cuda").eval()
    require(all(p.dtype == torch.float16 for p in source.parameters()), "Native EMA bytes are not half")
    image = batch["img"].to("cuda").float()
    with fp32_inference_flags():
        # H3/H4 retain all raw tolerances and failures. Same true learned EMA.
        with half_budget("H3_H4_raw", output):
            decomposition = half_decomposition(deepcopy(source).float(), image)
        comparisons = decomposition["comparisons"]
        result["H3"] = dict(capability=UNSUPPORTED, raw={k:v for k,v in comparisons.items() if k in ("U32_to_U16", "D32_to_D16", "N32_to_N16")})
        result["H4"] = dict(capability=UNSUPPORTED, raw={k:v for k,v in comparisons.items() if k in ("U16_to_D16", "D16_to_N16")})
        # Direct U16 -> N16 is also preserved, beyond the decomposition's adjacent edges.
        with half_budget("H4_U16_to_N16", output):
            u = deepcopy(source); n = deepcopy(source).float().fuse(verbose=False).half()
            uc, _ = capture_trace(u, image.half()); nc, _ = capture_trace(n, image.half())
            result["H4"]["raw"]["U16_to_N16"] = compare_traces(uc, nc, n, image.half(), atol=2e-3, rtol=2e-2)
        with half_budget("H1", output):
            started = time.monotonic()
            deployed_file = output / "same_path_half.pt"
            torch.save(n, deployed_file)
            identity = file_record(deployed_file)
            restored = torch.load(deployed_file, map_location="cuda", weights_only=False)
            restored_capture, _ = capture_trace(restored, image.half())
            target = restored.get_submodule(TARGET.rstrip("."))
            result["H1"] = dict(checkpoint=identity, state_bytes_exact=state_fingerprint(n) == state_fingerprint(restored),
                deployed=target.deployed, norm_retained=isinstance(target.norm, torch.nn.BatchNorm2d), dtype=str(next(restored.parameters()).dtype),
                checkpoint_unchanged=sha256(deployed_file) == identity["sha256"], nonzero_source=result["H0"]["all_q_nonzero"],
                comparisons={k:metric(nc[k], restored_capture[k], 2e-3, 2e-2) for k in nc if k != "candidate_indices"},
                elapsed_seconds=time.monotonic()-started)
        del u, n, restored
        atomic_json(output / "half_progress.json", result)
        result["H2"] = {}
        for label, half, memory in (("native_ema_half", True, False), ("final_eval_half", True, False), ("independent_memory_fp32", False, True)):
            with half_budget(label, output):
                started = time.monotonic()
                file = checkpoint
                if label == "final_eval_half":
                    file = output / "disposable_final_eval.pt"
                    shutil.copyfile(checkpoint, file); strip_optimizer(file)
                saved = torch.load(file, map_location="cpu", weights_only=False)
                selected = saved["ema"] if saved.get("ema") is not None else saved["model"]
                selected.to("cuda").eval()
                # Legacy comparison is only background; independently loaded same-file
                # reference and actual backend are the hard gate.
                legacy_model = deepcopy(selected).float().fuse(verbose=False)
                if half: legacy_model.half()
                inp = image.half() if half else image
                legacy, _ = capture_trace(legacy_model, inp)
                audit = backend_audit(file, selected, inp, legacy, half=half, memory_entry=memory)
                row = dict(checkpoint=file_record(file), audit=audit, elapsed_seconds=time.monotonic()-started)
                if memory: row["actual_memory_entry"] = True
                if label == "final_eval_half":
                    row.update(disposable_strip=True, terminal_fields_verified=saved["epoch"] == -1 and all(saved.get(k) is None for k in ("optimizer", "ema", "updates", "scaler")))
                result["H2"][label] = row
                atomic_json(output / "half_progress.json", result)
                del selected, saved, legacy_model
    del source, raw
    gc.collect()
    return result
