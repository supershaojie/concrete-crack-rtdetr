"""Bounded real-data DPR engineering checks; never starts formal training or test.

CPU/CUDA B2/160 lifecycle probes are distinct from native B16/640 AMP capacity.
Every invocation owns a fresh timestamp directory. Unknown checks remain PENDING.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from init_dpr import (ROOT, MAIN, VARIANTS, SOURCE_SHA256, require, sha256, controlled_models,
                      build_training_model, native_rebuild, verify_model)
from train_dpr import (CONTRACT, DPRTrainer, DPRSingleTrainer, atomic_json, code_identity, environment,
                       paths, recipe, revoke_permit, timestamp, trainer_class, strict_gate)
from dpr_checkpoint import DPRCheckpointTrainer, optimizer_param_names, require_checkpoint_policy
from dpr_data import dataset_identity
from c19_lif_v1_data import real_batch
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML, ASSETS
from ultralytics.utils.torch_utils import ModelEMA, autocast, init_seeds

ATOL, RTOL = 2e-5, 2e-4
TARGET = "model.5.blocks.1.branch2b."
NEW_NAMES = [TARGET + name for name in ("dpr_cd", "dpr_hd", "dpr_vd", "dpr_ad")]


def seed42():
    # Same initialization as native BaseTrainer.__init__; diagnostics must not
    # accidentally compare a nondeterministic CPU/CUDA configuration.
    init_seeds(42, deterministic=True)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return deepcopy(value)


def metric(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(equal=False, raw_allclose=False, finite=False, reason="shape/dtype mismatch")
    equal = bool(torch.equal(left, right))
    if not left.is_floating_point():
        return dict(equal=equal, raw_allclose=equal, finite=True, max_abs=0 if equal else float((left-right).abs().max()))
    a, b = left.detach().cpu().double(), right.detach().cpu().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(equal=equal, raw_allclose=False, finite=False, max_abs=None, relative_L2=None, exceeded_fraction=None)
    delta = (a - b).abs()
    exceeded = delta > ATOL + RTOL * b.abs()
    return dict(equal=equal, raw_allclose=not bool(exceeded.any()), finite=True,
                max_abs=float(delta.max()) if delta.numel() else 0.,
                relative_L2=float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-12)),
                exceeded_fraction=float(exceeded.double().mean()) if delta.numel() else 0.)


def compare(left, right, exact=False):
    rows, errors = {}, []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            rows[path] = metric(a, b)
        elif type(a) is not type(b):
            errors.append(path + ": type mismatch")
        elif isinstance(a, dict):
            if set(a) != set(b):
                errors.append(path + ": keys mismatch")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + "." + str(key))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                errors.append(path + ": length mismatch")
            for index, (aa, bb) in enumerate(zip(a, b)):
                visit(aa, bb, path + "." + str(index))
        elif a != b:
            errors.append(path + ": value mismatch")

    visit(left, right, "state")
    failures = [name for name, row in rows.items() if not row["equal" if exact else "raw_allclose"]]
    return dict(status="PASSED" if not errors and not failures else "FAILED", exact=not errors and all(r["equal"] for r in rows.values()),
                raw_allclose=not errors and all(r["raw_allclose"] for r in rows.values()),
                finite=all(r["finite"] for r in rows.values()), failed_tensors=failures, metadata_errors=errors, tensors=rows)


def snapshot(trainer):
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    groups = [dict(names=[names[id(p)] for p in group["params"]],
                   hyperparameters=cpu_copy({k: v for k, v in group.items() if k != "params"}))
              for group in trainer.optimizer.param_groups]
    return dict(model=cpu_copy(trainer.model.state_dict()), ema=cpu_copy(trainer.ema.ema.state_dict()),
                buffers=cpu_copy(dict(trainer.model.named_buffers())), ema_buffers=cpu_copy(dict(trainer.ema.ema.named_buffers())),
                optimizer={names[id(p)]: cpu_copy(v) for p, v in trainer.optimizer.state.items()}, groups=groups,
                scaler=cpu_copy(trainer.scaler.state_dict()), scaler_enabled=trainer.scaler.is_enabled(),
                updates=trainer.ema.updates, epoch=trainer.epoch, start_epoch=trainer.start_epoch,
                epochs=trainer.epochs, best_fitness=trainer.best_fitness)


def storages(value):
    visited, result = set(), set()

    def visit(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.numel():
                result.add((str(obj.device), obj.untyped_storage().data_ptr()))
            if obj.is_leaf and obj.grad is not None:
                visit(obj.grad)
        elif isinstance(obj, dict):
            for key, val in obj.items():
                visit(key)
                visit(val)
        elif isinstance(obj, (tuple, list, set)):
            for val in obj:
                visit(val)
        elif isinstance(obj, (torch.nn.Module, torch.optim.Optimizer, ModelEMA, RTDETRTrainer, SimpleNamespace)) or type(obj).__name__ == "GradScaler":
            visit(vars(obj))

    visit(value)
    return result


def move(trainer, device):
    trainer.model.to(device)
    trainer.ema.ema.to(device)
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in trainer.optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    destination = "cpu" if key == "step" and not group.get("capturable") and not group.get("fused") else device
                    trainer.optimizer.state[parameter][key] = value.to(destination)
    for key in ("_scale", "_growth_tracker"):
        value = getattr(trainer.scaler, key, None)
        if isinstance(value, torch.Tensor):
            setattr(trainer.scaler, key, value.to(device))
    trainer.device = torch.device(device)
    return trainer


def make_trainer(model, device, amp, variant):
    cls = trainer_class(variant)
    trainer = cls.__new__(cls)
    config, _ = recipe(variant)
    trainer.args = SimpleNamespace(**config)
    trainer.model = model.float().to(device).train()
    trainer.model.args = config
    trainer.data = dict(nc=1, channels=3, names={0: "crack"})
    trainer.set_model_attributes()
    trainer.device = torch.device(device)
    trainer.epochs, trainer.epoch, trainer.start_epoch, trainer.resume = 200, 0, 0, False
    trainer.best_fitness, trainer.fitness, trainer.metrics = 0., 0., {}
    trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    optimizer_param_names(trainer.model, trainer.optimizer)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=amp)
    trainer.ema = ModelEMA(trainer.model)
    trainer.save_period = -1
    trainer.dpr_checkpoint_purpose = "lifecycle_diagnostic"
    return trainer


def steps(trainer):
    return sum(float(item.get("step", 0)) for item in trainer.optimizer.state.values())


def one_step(trainer, batch, amp):
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    captures = {}

    def capture_decoder(module, inputs, output):
        captures["decoder"] = cpu_copy(output)

    hook = trainer.model.model[-1].register_forward_hook(capture_decoder)
    try:
        with autocast(amp):
            loss, components = trainer.model(batch)
            loss = loss.sum()
        require(bool(torch.isfinite(loss)), "Nonfinite real detection loss")
        trainer.scaler.scale(loss).backward()
    finally:
        hook.remove()
    scale = float(trainer.scaler.get_scale())
    gradients = {name: p.grad.detach().cpu().float().clone() / scale
                 for name, p in trainer.model.named_parameters() if p.grad is not None}
    before = steps(trainer)
    trainer.optimizer_step()
    effective = steps(trainer) > before
    dn = captures["decoder"][-1] if isinstance(captures["decoder"], tuple) else None
    return dict(loss=float(loss), components=cpu_copy(components).tolist(), scale_before=scale,
                scale_after=float(trainer.scaler.get_scale()), effective_update=effective,
                dn_seen=isinstance(dn, dict) and bool(dn.get("dn_num_split")),
                finite_gradients=all(bool(torch.isfinite(g).all()) for g in gradients.values())), gradients, captures


def bounded_updates(trainer, batch, amp):
    rows, effective, observed_grads = [], 0, {}
    kernel_before = trainer.model.get_submodule(TARGET.rstrip(".")).get_equivalent_kernel().detach().cpu().clone()
    before = {k: p.detach().cpu().clone() for k, p in trainer.model.named_parameters() if k in NEW_NAMES or k == TARGET + "conv.weight"}
    for index in range(16):
        row, grads, _ = one_step(trainer, batch, amp)
        rows.append(row)
        effective += int(row["effective_update"])
        if row["finite_gradients"] and row["effective_update"]:
            observed_grads = grads
        if effective >= 2:
            break
    require(effective >= 2, "No two effective optimizer updates within 16 batches")
    require(any(row["dn_seen"] for row in rows), "Real loss did not cover native denoising targets")
    changes = {k: not torch.equal(v, dict(trainer.model.named_parameters())[k].detach().cpu()) for k, v in before.items()}
    gradients = {name[len(TARGET):]: dict(finite_nonzero=bool(torch.isfinite(observed_grads[name]).all()
                                                           and observed_grads[name].abs().sum() > 0),
                                        l2=float(torch.linalg.vector_norm(observed_grads[name])))
                 for name in before}
    require(all(changes.values()), "A DPR parameter group or original W did not actually update")
    require(all(row["finite_nonzero"] for row in gradients.values()), "Missing finite effective DPR/W gradient")
    kernel_after = trainer.model.get_submodule(TARGET.rstrip(".")).get_equivalent_kernel().detach().cpu()
    require(bool(torch.isfinite(kernel_after).all()) and not torch.equal(kernel_before, kernel_after),
            "Actual effective dense kernel did not update finitely")
    return dict(status="PASSED", observed_batches=len(rows), effective_updates=effective, rows=rows,
                parameter_changes=changes, gradients=gradients, effective_kernel_changed=True,
                effective_kernel_delta_l2=float(torch.linalg.vector_norm(kernel_after-kernel_before)),
                effective_kernel_finite=True, effective_kernel_max_abs=float((kernel_after-kernel_before).abs().max()))


def clone_live(trainer):
    return deepcopy(trainer)


def live_pair(trainer, batch, amp, device):
    move(trainer, "cpu")
    first, second = clone_live(trainer), clone_live(trainer)
    require(not (storages(first) & storages(second)) and not (storages(trainer) & storages(first)), "Live copies share tensor storage")
    initial = compare(snapshot(first), snapshot(second), exact=True)
    results, gradients, forwards = [], [], []
    for current in (first, second):
        move(current, device)
        seed42()
        row, grad, forward = one_step(current, batch, amp)
        require(row["effective_update"] and row["finite_gradients"], "Independent continuation did not effectively update")
        results.append(row)
        gradients.append(grad)
        forwards.append(forward)
        move(current, "cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
    state = compare(snapshot(first), snapshot(second))
    gradient = compare(gradients[0], gradients[1])
    forward = compare(forwards[0], forwards[1], exact=True)
    first_state, second_state = snapshot(first), snapshot(second)
    exact_fields = compare({k: first_state[k] for k in ("buffers", "ema_buffers", "groups", "scaler", "updates", "epoch", "start_epoch")},
                           {k: second_state[k] for k in ("buffers", "ema_buffers", "groups", "scaler", "updates", "epoch", "start_epoch")}, exact=True)
    result = dict(initial_exact=initial["exact"], storage_independent=True, forward_exact=forward["exact"],
                  loss_exact=results[0]["loss"] == results[1]["loss"], exact_metadata_buffers=exact_fields["exact"],
                  raw_allclose=state["raw_allclose"] and gradient["raw_allclose"], state=state, gradients=gradient,
                  forward=forward, steps=results, atol=ATOL, rtol=RTOL,
                  quantization="NONE; independent actually updated live FP32 copies")
    del first, second
    gc.collect()
    return result


def saved_oracle(ckpt):
    model = ckpt["ema"]
    state = {k: (v.float() if v.is_floating_point() else v).clone() for k, v in model.state_dict().items()}
    buffers = {k: (v.float() if v.is_floating_point() else v).clone() for k, v in model.named_buffers()}
    groups, optimizer = [], {}
    for group, names in zip(ckpt["optimizer"]["param_groups"], ckpt["DPR_checkpoint"]["optimizer_param_names"]):
        groups.append(dict(names=names, hyperparameters={k: v for k, v in group.items() if k != "params"}))
        for ident, name in zip(group["params"], names):
            if ident in ckpt["optimizer"]["state"]:
                optimizer[name] = ckpt["optimizer"]["state"][ident]
    return dict(model=state, buffers=buffers, ema=deepcopy(state), ema_buffers=deepcopy(buffers), optimizer=optimizer,
                groups=groups, scaler=ckpt["scaler"], scaler_enabled=bool(ckpt["scaler"]), updates=ckpt["updates"],
                epoch=0, start_epoch=ckpt["epoch"] + 1, epochs=200, best_fitness=ckpt["best_fitness"])


def restore_native(checkpoint, amp, variant):
    cls = trainer_class(variant)
    trainer = cls.__new__(cls)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_checkpoint_policy(ckpt)
    trainer.args = SimpleNamespace(**deepcopy(ckpt["train_args"]))
    trainer.args.model = str(checkpoint)
    trainer.model, trainer.data = str(checkpoint), dict(nc=1, channels=3, names={0: "crack"})
    trainer.epochs, trainer.epoch, trainer.start_epoch, trainer.resume = 200, 0, 0, True
    trainer.device = torch.device("cpu")
    restored = trainer.setup_model()
    trainer.set_model_attributes()
    trainer.optimizer = trainer.build_optimizer(trainer.model, name="AdamW", lr=.0005, momentum=.937, decay=.0001)
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=amp)
    trainer.ema = ModelEMA(trainer.model)
    trainer.resume_training(restored)
    return trainer


def replay(trainer, gradients, device, overflow=False):
    move(trainer, device)
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scaler.scale(torch.zeros((), device=device, requires_grad=True))
    scale = trainer.scaler.get_scale()
    for name, parameter in trainer.model.named_parameters():
        parameter.grad = gradients[name].to(device).clone() * scale if name in gradients else None
    if overflow:
        next(p for p in trainer.model.parameters() if p.grad is not None).grad.reshape(-1)[0] = float("inf")
    before = steps(trainer)
    trainer.optimizer_step()
    result = dict(effective_update=steps(trainer) > before, scale_before=scale, scale_after=trainer.scaler.get_scale())
    move(trainer, "cpu")
    return result


def checkpoint_a(trainer, batch, amp, variant, device, temporary, evidence=None):
    """Saved-byte oracle precedes loading/casting; both formal and probe share saver."""
    move(trainer, "cpu")
    evidence = evidence if evidence is not None else {}
    evidence.update(status="FAILED", stage="save")
    trainer.wdir = Path(temporary) / "native_checkpoint"
    trainer.last, trainer.best = trainer.wdir / "last.pt", trainer.wdir / "best.pt"
    trainer.csv = Path(temporary) / "missing_results.csv"
    live_optimizer = cpu_copy(trainer.optimizer.state_dict())
    trainer.save_model()
    ckpt = torch.load(trainer.last, map_location="cpu", weights_only=False)
    policy = require_checkpoint_policy(ckpt)
    optimizer_exact = compare(live_optimizer, ckpt["optimizer"], exact=True)
    evidence.update(policy=policy, saved_live_optimizer_comparison=optimizer_exact)
    require(optimizer_exact["exact"], "Saved FP32 optimizer differs from live independent copy")
    require(not (storages(trainer.optimizer) & storages(ckpt["optimizer"])), "Saved optimizer aliases live tensors")
    first, second = restore_native(trainer.last, amp, variant), restore_native(trainer.last, amp, variant)
    oracle = saved_oracle(ckpt)
    restoration = compare(oracle, snapshot(first), exact=True)
    evidence.update(stage="restore", restoration=restoration)
    require(restoration["exact"] and compare(oracle, snapshot(second), exact=True)["exact"], "Saved-byte oracle restoration mismatch")
    independent = not (storages(first) & storages(second)) and not (storages(first) & storages(ckpt))
    require(independent, "Restored models share mutable storage")
    require(all(bool(dict(first.model.named_parameters())[n].abs().sum()) for n in NEW_NAMES), "Resume lost learned DPR")
    torch.save(trainer.model.state_dict(), Path(temporary) / "state.pt")
    state_exact = compare(trainer.model.state_dict(), torch.load(Path(temporary) / "state.pt", weights_only=False), exact=True)["exact"]
    torch.save(trainer.model, Path(temporary) / "full.pt")
    full = torch.load(Path(temporary) / "full.pt", map_location="cpu", weights_only=False)
    full_exact = compare(trainer.model.state_dict(), full.state_dict(), exact=True)["exact"]
    del full
    cpu_native_exact = None
    if device == "cpu":
        native_rows = []
        for current in (first, second):
            seed42()
            native_row, _, _ = one_step(current, batch, amp)
            native_rows.append(native_row)
        cpu_native_comparison = compare(snapshot(first), snapshot(second), exact=True)
        evidence.update(stage="cpu_native_continuation", cpu_native_comparison=cpu_native_comparison,
                        cpu_native_rows=native_rows)
        cpu_native_exact = (cpu_native_comparison["exact"]
                            and all(row["effective_update"] for row in native_rows)
                            and native_rows[0]["loss"] == native_rows[1]["loss"])
        require(cpu_native_exact, "CPU independent saved-state native continuation differs")
        del first, second
        first, second = restore_native(trainer.last, amp, variant), restore_native(trainer.last, amp, variant)
    # Take real unscaled gradients from a separate saved-state probe. It is discarded.
    probe = restore_native(trainer.last, amp, variant)
    move(probe, device)
    seed42()
    row, gradients, _ = one_step(probe, batch, amp)
    require(row["effective_update"] and row["finite_gradients"], "Saved-state real gradient probe did not effectively update")
    del probe
    gc.collect()
    rows = [replay(current, gradients, device) for current in (first, second)]
    replay_exact = compare(snapshot(first), snapshot(second), exact=True)
    evidence.update(stage="same_gradient_replay", replay=replay_exact, replay_steps=rows)
    require(replay_exact["exact"] and all(row["effective_update"] for row in rows), "Native identical-gradient replay differs")
    overflow_exact = None
    overflow_rows = []
    if amp:
        before_overflow = snapshot(first)
        overflow_rows = [replay(current, gradients, device, overflow=True) for current in (first, second)]
        after = snapshot(first)
        overflow_exact = (compare(snapshot(first), snapshot(second), exact=True)["exact"]
                          and compare(before_overflow["model"], after["model"], exact=True)["exact"]
                          and compare(before_overflow["optimizer"], after["optimizer"], exact=True)["exact"]
                          and all(not row["effective_update"] and row["scale_after"] < row["scale_before"] for row in overflow_rows))
        require(overflow_exact, "Native scaler overflow skip/replay failed")
    result = dict(status="PASSED", policy=policy, saved_live_optimizer_exact=True, saved_byte_restoration_exact=True,
                  saved_live_optimizer_comparison=optimizer_exact, cpu_native_continuation_exact=cpu_native_exact,
                  optimizer_names_groups_exact=True, no_shared_storage=independent, same_gradient_complete_state_exact=True,
                  nonzero_dpr_preserved=True, state_dict_reload_exact=state_exact, full_model_reload_exact=full_exact,
                  native_trainer_rebuild_exact=restoration["exact"], effective_step_exact=True,
                  overflow_skip_exact=overflow_exact, overflow_rows=overflow_rows,
                  native_ema_updates_even_on_skip="preserved", checkpoint_sha256=sha256(trainer.last),
                  restoration=restoration, replay=replay_exact, replay_steps=rows,
                  storage_precision="optimizer_fp32_v1; EMA remains native half; reference is saved quantized bytes")
    require(state_exact and full_exact, "State/full-model reload failed")
    del first, second, ckpt, oracle
    gc.collect()
    evidence.update(result)
    return result


def gradient_support(control):
    """Require observed finite gradient variation for each differing persistent tensor.

    Tiny gradient differences can be below raw allclose yet amplified by AdamW
    near-zero moments; the raw gradient and parameter metrics remain unchanged.
    """
    result = {}
    for key in control["state"]["failed_tensors"]:
        if key.startswith("state.model."):
            name = key[len("state.model."):]
        elif key.startswith("state.ema."):
            name = key[len("state.ema."):]
        elif key.startswith("state.optimizer."):
            name = key[len("state.optimizer."):].rsplit(".", 1)[0]
        else:
            result[key] = False
            continue
        gradient = control["gradients"]["tensors"].get("state." + name, {})
        result[key] = gradient.get("finite") is True and gradient.get("equal") is False
    return result


def classify_b(parent, candidate, a_status, device):
    controls = (parent, candidate)
    valid = all(c["initial_exact"] and c["storage_independent"] and c["forward_exact"] and c["loss_exact"]
                and c["exact_metadata_buffers"] and c["state"]["finite"] and c["gradients"]["finite"] for c in controls)
    raw = all(c["raw_allclose"] for c in controls)
    status, explanation = ("PASSED", "Original raw tolerance passed") if valid and raw else ("FAILED", "Independent backward difference unresolved")
    if valid and not raw and device == "cuda" and a_status == "PASSED":
        # Only established floating parameter/moment differences are candidates for a note.
        parent_failed = set(parent["state"]["failed_tensors"])
        candidate_failed = set(candidate["state"]["failed_tensors"])
        inherited = candidate_failed <= parent_failed
        added_ok = all(row["raw_allclose"] for collection in (candidate["state"], candidate["gradients"])
                       for key, row in collection["tensors"].items() if ".dpr_" in key)
        permitted = all((".model." in k or ".ema." in k or ".optimizer." in k) and not k.endswith(".step")
                        for c in controls for k in c["state"]["failed_tensors"])
        parent_support, candidate_support = gradient_support(parent), gradient_support(candidate)
        gradients_support = bool(parent_support) and bool(candidate_support) and all(parent_support.values()) and all(candidate_support.values())
        if inherited and added_ok and permitted and gradients_support:
            status, explanation = "PRECISION_NOTE", ("A passed; exact forward/loss/buffers/scaler/metadata; finite independent CUDA backward differences "
                                                      "are confined to inherited state failures also observed in parent; added DPR updates/gradients meet original tolerance")
    return dict(status=status, raw_allclose=raw, atol=ATOL, rtol=RTOL, explanation=explanation,
                parent=parent, candidate=candidate, long_term_trajectory_equivalence="NOT_CLAIMED")


def lifecycle(args, report):
    parent80, _, _ = controlled_models(args.source, args.variant)
    initialized = RTDETR(str(args.initialized)).model
    seed42()
    candidate, _ = build_training_model(initialized.yaml, initialized, dict(nc=1, channels=3), args.variant)
    seed42()
    parent = native_rebuild(parent80.yaml, parent80, nc=1, channels=3)
    del parent80, initialized
    data_config = YAML.load(args.data)
    dataset_root = Path(data_config["path"])
    batch_cpu, samples = real_batch(dataset_root, size=160, count=2)
    require(batch_cpu["bboxes"].shape[0] > 0, "Lifecycle batch has no GT")
    report["lifecycle_batch"] = dict(batch=2, imgsz=160, samples=samples, real_gt=int(batch_cpu["bboxes"].shape[0]),
                                      scope="bounded lifecycle diagnostic; does not establish B16/640 capacity")
    inference = report["checks"]["learned_inference"] = dict(status="PENDING")
    for device, amp, label in (("cpu", False, "cpu_fp32"), ("cuda", False, "cuda_fp32"), ("cuda", True, "cuda_native_amp")):
        entry = report["checks"][label] = dict(status="PENDING", device=device, amp=amp, batch=2, imgsz=160)
        if (args.device != "all" and args.device != device) or (device == "cuda" and not torch.cuda.is_available()):
            entry["reason"] = "Device excluded/unavailable"
            continue
        print("BEGIN lifecycle " + label, flush=True)
        batch = {key: value.to(device) for key, value in batch_cpu.items()}
        try:
            seed42()
            p = make_trainer(deepcopy(parent), device, amp, args.variant)
            # Parent has no DPR parameters; only genuine optimizer updates are required.
            parent_steps = []
            for _ in range(16):
                row, _, _ = one_step(p, batch, amp)
                parent_steps.append(row)
                if sum(r["effective_update"] for r in parent_steps) >= 2:
                    break
            require(sum(r["effective_update"] for r in parent_steps) >= 2, "Parent has insufficient effective updates")
            parent_b = live_pair(p, batch, amp, device)
            del p
            seed42()
            c = make_trainer(deepcopy(candidate), device, amp, args.variant)
            entry["updates"] = bounded_updates(c, batch, amp)
            entry["gradients"] = entry["updates"]["gradients"]
            candidate_b = live_pair(c, batch, amp, device)
            entry["B"] = classify_b(parent_b, candidate_b, "PENDING", device)
            entry["A"] = dict(status="FAILED", stage="not_started")
            with tempfile.TemporaryDirectory(prefix="dpr_lifecycle_", dir=ROOT / "outputs/dpr") as temporary:
                entry["A"] = checkpoint_a(c, batch, amp, args.variant, device, temporary, entry["A"])
            entry["B"] = classify_b(parent_b, candidate_b, entry["A"]["status"], device)
            entry["status"] = entry["B"]["status"]
            if not amp:
                from check_dpr import audit_learned_model
                move(c, device)
                flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    result = audit_learned_model(deepcopy(c.model).eval(), batch["img"])
                    if device == "cuda":
                        from check_dpr import audit_cuda_half
                        inference["cuda_half"] = audit_cuda_half(deepcopy(c.model).eval(), batch["img"])
                finally:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
                inference[label] = result
                report.setdefault("learned_inference_details", {})[label] = result
                # Detailed report is interpreted below, never guessed from a summary.
                del result
            move(c, "cpu")
            del c
        except Exception as error:
            entry.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
            print(entry["traceback"], flush=True)
        finally:
            atomic_json(args.output / "preflight.json", report)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class CapacityComplete(Exception):
    pass


class CapacityTrainer(DPRTrainer):
    """Native train path with observation only; callbacks enforce finite budget."""
    capacity_report = None
    effective = 0
    observed = 0
    last_gradients = None
    dpr_checkpoint_purpose = "bounded_capacity"

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        require(batch["img"].shape[0] == 16 and list(batch["img"].shape[-2:]) == [640, 640], "Capacity batch/imgsz changed")
        self.capacity_report["gt_instances"] += int(batch["bboxes"].shape[0])
        return batch

    def optimizer_step(self):
        names = dict(self.model.named_parameters())
        scale = float(self.scaler.get_scale())
        norms = {k[len(TARGET):]: float(torch.linalg.vector_norm(names[k].grad.detach().float() / scale))
                 for k in NEW_NAMES + [TARGET + "conv.weight"] if names[k].grad is not None}
        before = steps(self)
        super().optimizer_step()
        effective = steps(self) > before
        if effective:
            require(len(norms) == 5 and all(math.isfinite(value) and value > 0 for value in norms.values()),
                    "Effective capacity update has invalid DPR/W gradients")
        self.effective += int(effective)
        self.capacity_report["steps"].append(dict(effective_update=effective, scale_before=scale,
                                                  scale_after=float(self.scaler.get_scale()), gradient_norms=norms))


class CapacitySingleTrainer(CapacityTrainer):
    variant = "dpr_v1"


def amp_resources():
    """Use existing trusted local AMP probe resources; never download or install."""
    roots = [ROOT, Path(os.environ.get("DPR_MAIN", "/root/autodl-tmp/projects/Crack_RTDETR"))]
    records = []
    for name, destination, suffixes in (("yolo26n.pt", ROOT / "yolo26n.pt", ("yolo26n.pt", "weights/yolo26n.pt")),
                                       ("bus.jpg", ASSETS / "bus.jpg", ("ultralytics-main/ultralytics/assets/bus.jpg", "bus.jpg"))):
        if not destination.is_file():
            candidates = [root / suffix for root in roots for suffix in suffixes]
            source = next((p for p in candidates if p.is_file()), None)
            require(source is not None, "PENDING: existing native AMP resource missing: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as inp, destination.open("xb") as out:
                shutil.copyfileobj(inp, out)
        records.append(dict(path=str(destination), sha256=sha256(destination)))
    return records


def capacity(args):
    result = dict(status="PENDING", batch=16, imgsz=640, native_amp=True, observed_batches=0, effective_updates=0,
                  gt_instances=0, dn_seen=False, optimizer_exact_coverage=False, steps=[], losses=[],
                  formal_training="NOT_STARTED", final_test="NOT_RUN")
    if not torch.cuda.is_available():
        result["reason"] = "CUDA unavailable"
        return result
    hooks = []
    started = time.perf_counter()
    cls = CapacityTrainer if args.variant == MAIN else CapacitySingleTrainer
    cls.capacity_report, cls.effective, cls.observed = result, 0, 0
    trainer_ref = []
    atomic_json(args.output / "capacity_progress.json", result)
    try:
        result["amp_resources"] = amp_resources()
        config, diff = recipe(args.variant, args.initialized, args.data)
        with tempfile.TemporaryDirectory(prefix="dpr_native_capacity_", dir=ROOT / "outputs/dpr") as temporary:
            config.update(project=temporary, name="bounded_capacity", save_dir=str(Path(temporary) / "bounded_capacity"))
            model = RTDETR(str(args.initialized))

            def start(trainer):
                trainer_ref.append(trainer)
                require(trainer.amp and trainer.args.batch == 16 and trainer.args.imgsz == 640, "Native capacity precision/shape changed")
                result["optimizer_names"] = optimizer_param_names(trainer.model, trainer.optimizer)
                result["optimizer_exact_coverage"] = True
                result["actual_trainer_rebuild"] = trainer.dpr_rebuild_audit
                result["setup_completed"] = True
                trainer.dpr_initial_state = {name: value.detach().cpu().clone() for name, value in trainer.model.named_parameters()
                                             if name in NEW_NAMES + [TARGET + "conv.weight"]}
                torch.cuda.reset_peak_memory_stats()

                def decoder_seen(module, inputs, output):
                    metadata = output[-1] if isinstance(output, tuple) else None
                    result["dn_seen"] |= isinstance(metadata, dict) and bool(metadata.get("dn_num_split"))

                hooks.append(trainer.model.model[-1].register_forward_hook(decoder_seen))

            def batch_start(trainer):
                trainer._oom_retries = 3

            def batch_end(trainer):
                trainer.observed += 1
                result["observed_batches"], result["effective_updates"] = trainer.observed, trainer.effective
                value = float(trainer.loss.detach())
                result["losses"].append(value)
                atomic_json(args.output / "capacity_progress.json", result)
                require(math.isfinite(value), "Capacity loss nonfinite")
                if trainer.effective >= 2 or trainer.observed >= 16:
                    raise CapacityComplete()

            model.add_callback("on_train_start", start)
            model.add_callback("on_train_batch_start", batch_start)
            model.add_callback("on_train_batch_end", batch_end)
            try:
                model.train(trainer=cls, **config)
                raise RuntimeError("Native capacity returned without bounded interruption")
            except CapacityComplete:
                require(result["effective_updates"] >= 2 and result["dn_seen"] and result["gt_instances"] > 0,
                        "Insufficient effective updates/GT/DN within 16 batches")
                trained = model.trainer
                final_state = dict(trained.model.named_parameters())
                result["parameter_changes"] = {name: not torch.equal(value, final_state[name].detach().cpu())
                                               for name, value in trained.dpr_initial_state.items()}
                require(all(result["parameter_changes"].values()), "Capacity did not change DPR parameters and original W")
                result["status"] = "PASSED"
    except Exception as error:
        result.update(status="PENDING" if str(error).startswith("PENDING:") else "FAILED", error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            hook.remove()
        for trainer in trainer_ref:
            for loader_name in ("train_loader", "test_loader"):
                iterator = getattr(getattr(trainer, loader_name, None), "iterator", None)
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        result["elapsed_seconds"] = time.perf_counter() - started
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def file_record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def capacity_subprocess(args):
    """Own the capacity process tree and stop it after a finite wall-clock budget."""
    output = args.output / "capacity_worker"
    output.mkdir()
    command = [sys.executable, str(ROOT / "tools/preflight_dpr.py"), "--capacity-worker", "--variant", args.variant,
               "--source", str(args.source), "--initialized", str(args.initialized), "--data", str(args.data),
               "--output", str(output)]
    options = {"cwd": str(ROOT)}
    if os.name != "nt":
        options["start_new_session"] = True
    process = subprocess.Popen(command, **options)
    started = time.perf_counter()

    def stop_owned_process():
        if process.poll() is not None:
            return
        # The child PID/group belongs to this exact Popen invocation.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=30)

    try:
        exit_code = process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        stop_owned_process()
        progress_path = output / "capacity_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        progress.update(status="PENDING", reason="TIMEOUT: owned native capacity process exceeded 600-second budget",
                        wallclock_budget_seconds=600, batch=16, imgsz=640, native_amp=True,
                        exit_code=process.returncode, elapsed_seconds=time.perf_counter() - started)
        return progress
    finally:
        # KeyboardInterrupt and parent exceptions must not leave our DataLoader
        # children running. No process search or external job cancellation occurs.
        stop_owned_process()
    path = output / "capacity.json"
    if not path.is_file():
        return dict(status="FAILED", reason="Native capacity process produced no result", exit_code=exit_code)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["worker_exit_code"] = exit_code
    require(exit_code == (0 if result.get("status") == "PASSED" else 3), "Capacity worker exit code disagrees with report")
    return result


def run(args):
    revoke_permit(args.variant)
    args.output = (args.output or paths(args.variant)["meta"] / ("preflight_" + timestamp())).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    report = dict(contract=CONTRACT, report_kind="full_preflight_engineering", status="PENDING", variant=args.variant,
                  formal_training="NOT_STARTED", final_test="NOT_RUN", checks={}, output=str(args.output),
                  environment=environment(), tolerances=dict(atol=ATOL, rtol=RTOL))
    started = time.perf_counter()
    try:
        report["code_identity"] = code_identity()
        report["source_sha256"] = sha256(args.source)
        require(report["source_sha256"] == SOURCE_SHA256, "Wrong public source")
        report["initialization_sha256"] = sha256(args.initialized)
        report["dataset_identity"] = dataset_identity(args.data)
        report["precision"] = dict(deterministic=True, deterministic_warn_only=True, seed=42,
                                   native_seed_function="ultralytics.utils.torch_utils.init_seeds",
                                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
        report["recipe"], report["recipe_diff"] = recipe(args.variant, args.initialized, args.data)
        report["initialization"] = file_record(args.init_report)
        report["mathematics"] = file_record(args.math_report)
        math_report = json.loads(args.math_report.read_text(encoding="utf-8"))
        from check_dpr import validate_math_report
        report["mathematics_verified"] = validate_math_report(math_report, args.variant)
        require(report["mathematics_verified"], "DPR mathematics/structure report missing or failed")
        lifecycle(args, report)
        if args.capacity:
            print("BEGIN native B16/640 AMP capacity", flush=True)
            report["checks"]["capacity"] = capacity_subprocess(args)
        else:
            report["checks"]["capacity"] = dict(status="PENDING", reason="Capacity not requested")
        # Inference details are interpreted by the audited module implementation.
        normalize_inference(report)
        statuses = [item.get("status", "PENDING") for item in report["checks"].values()]
        report["status"] = "FAILED" if "FAILED" in statuses else "PENDING" if "PENDING" in statuses else "PASSED"
        if not report["environment"]["server_environment"]:
            report["server_admission"] = "PENDING; local engineering evidence never grants server start"
        atomic_json(args.output / "preflight.json", report)
        if report["status"] == "PASSED" and report["environment"]["server_environment"]:
            report["admission"] = strict_gate(args.variant, args.source, args.initialized, args.data, args.output / "preflight.json")
    except Exception as error:
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["code_identity_at_end"] = code_identity()
        report["source_stable_during_checks"] = report.get("code_identity") == report["code_identity_at_end"]
        atomic_json(args.output / "preflight.json", report)
        atomic_json(paths(args.variant)["meta"] / "latest_preflight.json", dict(report=str(args.output / "preflight.json"),
                    sha256=sha256(args.output / "preflight.json"), status=report["status"]))
    print(json.dumps(dict(status=report["status"], report=str(args.output / "preflight.json")), indent=2))
    return 0 if report["status"] == "PASSED" else 3


def normalize_inference(report):
    """Map explicitly audited learned-state leaves; missing evidence never passes."""
    inference = report["checks"].setdefault("learned_inference", {})
    mappings = dict(ema=("ema_retains_nonzero",), fold_norm_retained=("dpr_fold_only",),
                    native_fuse=("native_fuse",), fuse_idempotent=("native_fuse_again",),
                    deploy_reload=("deploy_state_reload", "deploy_full_reload_idempotent", "native_fused_saved_reload"),
                    autobackend=("autobackend_saved_best",))
    details = report.get("learned_inference_details", {})
    for name, keys in mappings.items():
        rows = {mode: {key: details.get(mode, {}).get("checks", {}).get(key, {"status": "PENDING"}) for key in keys}
                for mode in ("cpu_fp32", "cuda_fp32")}
        values = [row["status"] for modes in rows.values() for row in modes.values()]
        inference[name] = dict(status="FAILED" if "FAILED" in values else "PENDING" if "PENDING" in values else
                              "PRECISION_NOTE" if "PRECISION_NOTE" in values else "PASSED", modes=rows)
    for key in ("cpu_fp32", "cuda_fp32", "cuda_half", "ema", "fold_norm_retained", "native_fuse",
                "fuse_idempotent", "deploy_reload", "autobackend"):
        inference.setdefault(key, dict(status="PENDING", reason="Required learned inference leaf not completed"))
    statuses = [inference[key].get("status", "PENDING") for key in inference if key != "status" and isinstance(inference[key], dict)]
    inference["status"] = "FAILED" if "FAILED" in statuses else "PENDING" if "PENDING" in statuses else "PASSED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=MAIN)
    for key in ("source", "initialized", "data", "init-report", "math-report", "output"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--capacity", action="store_true", help="Run original B16/640/AMP native training path, at most 16 batches")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--capacity-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    p = paths(args.variant)
    args.source, args.initialized, args.data = args.source or p["source"], args.initialized or p["init"], args.data or p["data"]
    args.init_report = args.init_report or p["meta"] / "init.json"
    args.math_report = args.math_report or p["meta"] / "math.json"
    if args.capacity_worker:
        torch.set_num_threads(args.threads)
        result = capacity(args)
        atomic_json(args.output / "capacity.json", result)
        return 0 if result["status"] == "PASSED" else 3
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
