"""Audit C19 initialization, decoder behavior, dtypes and optional real-data smoke. No formal training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from init_rtdetr_r18_lite_cbr_controlled import (
    BASE_CFG, DEFAULT_SOURCE, DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, new_prefix, CBR_CFG,
    clean_checkpoint, read_source, require, runtime_info, sha256, tensor_rows, verify_module,
    verify_protected, verify_reloads, write_json,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import CrackBoundaryRefinement, RTDETRDecoderCBR
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import init_seeds


def state_hashes(model):
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items()}


def trainer_build(cfg, weights, nc):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.data = {"nc": nc, "channels": 3}
    init_seeds(42, deterministic=True)  # Same single-process seed + 1 + RANK as C2.
    model = trainer.get_model(cfg=deepcopy(cfg), weights=weights, verbose=False)
    return model.eval(), torch.get_rng_state().clone()


def compare_output(a, b, path="output"):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, f"{path}: shape/dtype mismatch.")
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), f"{path}: non-finite output.")
        require(torch.equal(a, b), f"{path}: bitwise CPU FP32 equivalence failed.")
        return [{"tensor": path, "shape": list(a.shape), "equal": True, "max_abs": 0.0}]
    if isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), f"{path}: container mismatch.")
        return [r for i, (x, y) in enumerate(zip(a, b)) for r in compare_output(x, y, f"{path}.{i}")]
    if isinstance(a, dict):
        require(type(a) is type(b) and a.keys() == b.keys(), f"{path}: mapping mismatch.")
        return [row for k in a for row in compare_output(a[k], b[k], f"{path}.{k}")]
    require(a == b, f"{path}: value mismatch.")
    return []


def real_optimizer(model):
    trainer = RTDETRTrainer.__new__(RTDETRTrainer)
    trainer.args = SimpleNamespace(warmup_bias_lr=0.1, lr0=0.0005, weight_decay=0.0001)
    optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.0005, momentum=0.937, decay=0.0001)
    require(vars(trainer.args) == dict(warmup_bias_lr=0.1, lr0=0.0005, weight_decay=0.0001),
            "Optimizer mutated C2 settings.")
    return optimizer


def optimizer_rows(model, optimizer):
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    require(len({id(p) for p in parameters}) == len(parameters), "Optimizer contains duplicate parameters.")
    require({id(p) for p in parameters} == {id(p) for p in model.parameters()}, "Optimizer coverage differs.")
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    return [{"name": n, "numel": p.numel(), "group": groups[id(p)]["param_group"],
             "weight_decay": groups[id(p)]["weight_decay"],
             "warmup_start_lr": 0.1 if groups[id(p)]["param_group"] == "bias" else 0.0}
            for n, p in model.named_parameters()]


def audit_optimizer(baseline, target):
    a, b = real_optimizer(baseline), real_optimizer(target)
    old = {r["name"]: r for r in optimizer_rows(baseline, a)}
    rows = optimizer_rows(target, b)
    counts = {g["param_group"]: len(g["params"]) for g in b.param_groups}
    for row in rows:
        if row["name"] in old:
            require(row == old[row["name"]], f"C2 optimizer membership changed: {row['name']}")
    added = [r for r in rows if r["name"].startswith(new_prefix(target))]
    require({r["name"] for r in added} == {new_prefix(target) + n for n, _ in target.model[-1].cbr.named_parameters()},
            "New optimizer coverage differs.")
    for row in added:
        expected = "bias" if row["name"].endswith(".bias") else "weight"
        require(row["group"] == expected, f"New parameter incorrectly grouped: {row}")
        require(row["weight_decay"] == (0.0 if expected == "bias" else 0.0001), "New weight decay differs.")
    base_counts = {g["param_group"]: len(g["params"]) for g in a.param_groups}
    source = inspect.getsource(RTDETRTrainer._do_train)
    require('x.get("param_group") == "bias"' in source, "Trainer warmup rule changed.")
    return {"counts": counts, "baseline_counts": base_counts, "new_parameters": added, "all_parameters": rows,
            "common_membership_equal": True,
            "no_missing_or_duplicate_parameters": True,
            "lr0": 0.0005, "weight_decay": 0.0001, "warmup_bias_lr": 0.1, "status": "passed"}


def audit_initialization(source, initialized):
    _, base_weights = read_source(source)
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "CBR contains training state.")
    require(checkpoint.get("cbr_provenance", {}).get("source_sha256") == SOURCE_SHA256,
            "Missing C2 provenance.")
    weights = RTDETR(str(initialized)).model
    verify_module(weights)
    rows = tensor_rows(base_weights.state_dict(), weights.state_dict())
    require(all(r["equal"] for r in rows), "C2 checkpoint mapping incomplete/inexact.")
    report = {"source": str(source.resolve()), "source_sha256": sha256(source),
              "initialized": str(initialized.resolve()), "initialized_sha256": sha256(initialized),
              "checkpoint_common_states": rows, "reloads": verify_reloads(initialized, weights.state_dict()),
              "entry_point": "unchanged RTDETRTrainer.get_model; __new__ fixture avoids dataset I/O",
              "seed": 42, "classes": {}}
    optimizer_report = None
    for nc in (1, 80):
        # Loading shared weights could hide constructor drift; inspect fresh constructors independently.
        torch.manual_seed(42)
        fresh_base = RTDETRDetectionModel(str(BASE_CFG), nc=nc, verbose=False)
        fresh_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        fresh_target = RTDETRDetectionModel(str(CBR_CFG), nc=nc, verbose=False)
        require(torch.equal(fresh_rng, torch.get_rng_state()), f"nc={nc}: fresh construction RNG differs.")
        require(all(r["equal"] for r in tensor_rows(fresh_base.state_dict(), fresh_target.state_dict())),
                f"nc={nc}: fresh common construction states differ.")
        del fresh_base, fresh_target
        baseline, expected_rng = trainer_build(base_weights.yaml, base_weights, nc)
        target, actual_rng = trainer_build(weights.yaml, weights, nc)
        rows = tensor_rows(baseline.state_dict(), target.state_dict())
        require(all(r["equal"] for r in rows), f"nc={nc}: common states differ.")
        require(torch.equal(expected_rng, actual_rng), f"nc={nc}: construction RNG differs.")
        added = set(target.state_dict()) - set(baseline.state_dict())
        require(added == {new_prefix(target) + k for k in target.model[-1].cbr.state_dict()}, "Unexpected extra states.")
        require(all(r["equal"] for r in tensor_rows({k: weights.state_dict()[k] for k in added}, target.state_dict())),
                "Trainer rebuild changed new initialization states.")
        verify_module(target)
        image = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(123))
        with torch.inference_mode():
            a, b = baseline(image), target(image)
        require(a[0].shape == (1, 300, nc + 4), "Unexpected complete forward shape.")
        classification = [r for r in rows if r["key"].startswith("model.26.") and
                          any(s in r["key"] for s in ("denoising_class_embed", "enc_score_head", "dec_score_head"))]
        require(len(classification) == 9, "Expected nine decoder classification states (five weights).")
        report["classes"][str(nc)] = {
            "common_states": rows, "classification_states": classification, "cpu_rng_equal_after_build_load": True,
            "fresh_constructor_common_states_and_rng_equal": True, "total_states": len(target.state_dict()),
            "new_states_exact": sorted(added),
            "cpu_640_outputs": compare_output(a, b), "unfused_parameters": sum(p.numel() for p in target.parameters()),
            "state_sha256": state_hashes(target), "status": "passed",
        }
        report["classes"][str(nc)]["real_train_api"] = audit_real_train_api(initialized, nc, target, actual_rng)
        if nc == 80:
            reloaded = RTDETR(str(initialized)).model.eval()
            with torch.inference_mode():
                report["reload_cpu_640_outputs"] = compare_output(b, reloaded(image))
            del reloaded
        if nc == 1:
            optimizer_report = audit_optimizer(baseline, target)
        del baseline, target, a, b
        gc.collect()
    return report, optimizer_report


def audit_real_train_api(initialized, nc, expected, expected_rng):
    """Run the actual Model.train -> RTDETRTrainer.get_model rebuild, stopping before data/steps."""
    from train_rtdetr_r18_lite_cbr import actual_args_check, build_locked_args, DEFAULT_NAME
    recipe = YAML.load(ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
    locked, _ = build_locked_args(recipe, initialized, DEFAULT_NAME)
    expected_hashes = state_hashes(expected)
    observed = {}

    class AuditStopped(Exception):
        pass

    class DatasetFreeTrainer(RTDETRTrainer):
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**{k: v for k, v in overrides.items() if k != "session"})
            self.data = {"nc": nc, "channels": 3}
            init_seeds(42, deterministic=True)

        def train(self):
            actual_args_check(vars(self.args), locked)
            require(state_hashes(self.model) == expected_hashes, "Actual model.train rebuild changed initialized states.")
            require(torch.equal(torch.get_rng_state(), expected_rng), "Actual model.train rebuild changed CPU RNG.")
            observed.update(status="passed", states_exact=len(expected_hashes), args_fields=len(locked),
                            rng_equal=True, inherited_get_model=True, dataset_io=False, optimizer_steps=0)
            raise AuditStopped()

    require(DatasetFreeTrainer.get_model is RTDETRTrainer.get_model, "Fixture must inherit the actual get_model.")
    with patch("ultralytics.engine.model.checks.check_pip_update_available"):
        try:
            RTDETR(str(initialized)).train(trainer=DatasetFreeTrainer, **locked)
        except AuditStopped:
            pass
    require(observed.get("status") == "passed", "Actual train API was not inspected.")
    return observed



def gradient_magnitudes(module):
    result = {}
    for name, p in module.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid gradient: {name}")
        result[name] = float(p.grad.float().abs().sum())
    return result


def audit_precision(initialized, device, mode):
    amp, half = mode == "amp", mode == "half"
    weights = RTDETR(str(initialized)).model
    model, _ = trainer_build(weights.yaml, weights, 1)
    model.nc = 1  # set_model_attributes does this in the real trainer
    model.to(device)
    if half:
        model.half()
    dtype = torch.float16 if half else torch.float32
    ctx = lambda: torch.autocast(device_type=device, dtype=torch.float16) if amp else nullcontext()
    with torch.inference_mode(), ctx():
        out = model(torch.rand(1, 3, 640, 640, device=device, dtype=dtype))
    require(out[0].shape == (1, 300, 5) and torch.isfinite(out[0]).all(), f"{device}/{mode}: 640 forward failed")
    report = {"mode": mode, "shape": list(out[0].shape), "output_dtype": str(out[0].dtype)}
    # Explicit-half is inference-only in the native trainer. FP32 and AMP need backward.
    if not half:
        model.train()
        optimizer = real_optimizer(model)
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
        steps = []
        for count in (0, 3, 3):
            batch = {"img": torch.rand(2, 3, 128, 128, device=device),
                     "cls": torch.zeros(count, 1, device=device),
                     "bboxes": torch.tensor([[.5,.5,.6,.08],[.5,.5,.08,.6],[.02,.95,.1,.09]], device=device)[:count],
                     "batch_idx": torch.tensor([0,1,1], device=device)[:count]}
            optimizer.zero_grad(set_to_none=True)
            with ctx():
                loss, items = model(batch)
            require(torch.isfinite(loss) and torch.isfinite(items).all(), f"{mode}: nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Empty GT uses classification loss; native RT-DETR omits box losses,
            # so a box-only refinement is correctly unused in that backward.
            grads = gradient_magnitudes(model.model[-1].cbr) if count else {
                k: None if p.grad is None else float(p.grad.abs().sum())
                for k, p in model.model[-1].cbr.named_parameters()}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    require(torch.isfinite(parameter.grad).all(), f"Nonfinite common gradient: {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer); scaler.update()
            steps.append({"gt_count": count, "loss": float(loss.detach()), "cbr_gradients": grads})
        require(all(v > 0 for k, v in grads.items() if k != "score.bias"), f"{mode}: branch remained closed")
        report["synthetic_steps"] = steps
    report["status"] = "passed"
    return report


def benchmark(initialized, source, device):
    import time
    _, base_weights = read_source(source)
    weights = RTDETR(str(initialized)).model
    result = {"device": device, "batch": 1, "imgsz": 640, "dtype": "FP32", "warmup": 3, "iterations": 10,
              "note": "Unfused complete-model forward; same random image. Not dataset validator speed."}
    x = torch.rand(1, 3, 640, 640, device=device)
    for label, w in (("C2", base_weights), ("C19", weights)):
        model, _ = trainer_build(w.yaml, w, 1)
        model.to(device)
        with torch.inference_mode():
            for _ in range(3): model(x)
            if device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(10): model(x)
            if device == "cuda": torch.cuda.synchronize()
            result[label] = {"parameters": sum(p.numel() for p in model.parameters()),
                             "ms_per_image": (time.perf_counter() - start) * 100,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
        del model
    return result


def run_regressions(report_dir):
    result = {}
    for pattern in ("test_cbr.py", "test_cbr_tools.py"):
        suite = unittest.TestLoader().discover(str(ROOT / "ultralytics-main/tests"), pattern=pattern)
        stream = io.StringIO()
        ran = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        log = report_dir / (pattern + ".log")
        log.write_text(stream.getvalue(), encoding="utf-8")
        result[pattern] = {"run": ran.testsRun, "failures": len(ran.failures), "errors": len(ran.errors), "log": str(log)}
        require(ran.wasSuccessful() and ran.testsRun > 0 and not ran.skipped, f"Regression failed: {log}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initialized", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/cbr/audit.json")
    parser.add_argument("--require-torch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke-data", type=Path, help="Real dataset YAML; use C2 path on server.")
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "outputs/cbr/smoke")
    parser.add_argument("--c2-args", type=Path, default=ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "runtime": runtime_info()}
    try:
        if args.require_torch:
            require(torch.__version__.split("+")[0] == args.require_torch, "Wrong PyTorch version.")
        if args.require_cuda: require(torch.cuda.is_available(), "CUDA is required.")
        report["protected"] = verify_protected()
        base, target = YAML.load(BASE_CFG), YAML.load(CBR_CFG)
        target["head"][-1][2] = "RTDETRDecoder"
        require(base == target, "YAML differs beyond the decoder class.")
        print("Auditing exact initialization, real train API and 640 FP32 equality...", flush=True)
        report["initialization"], report["optimizer"] = audit_initialization(args.source, args.initialized)
        report["regressions"] = run_regressions(args.report.parent)
        report["cpu"] = audit_precision(args.initialized, "cpu", "fp32")
        report["cuda"] = {"status": "unverified"}
        if torch.cuda.is_available():
            report["cuda"] = {mode: audit_precision(args.initialized, "cuda", mode) for mode in ("fp32", "amp", "half")}
            report["cuda"]["status"] = "passed"
        report["benchmark"] = benchmark(args.initialized, args.source, "cuda" if torch.cuda.is_available() else "cpu")
        report["real_data_smoke"] = {"status": "not_run"}
        if args.smoke_data:
            from smoke_rtdetr_r18_lite_cbr import smoke
            report["real_data_smoke"] = smoke(args.initialized, args.smoke_data, args.c2_args, args.smoke_dir,
                                                   report["initialization"]["classes"]["1"]["state_sha256"])
        report["status"] = "passed" if report["cuda"]["status"] == "passed" else "cpu_passed_cuda_unverified"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        write_json(args.report, report)
        print(f"Audit {report['status']}: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
