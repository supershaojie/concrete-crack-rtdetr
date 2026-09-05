"""Audit C18 with real trainer construction/optimizer and synthetic inputs only; never val/test."""
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

from init_rtdetr_r18_lite_gsdr_aifi_v3_controlled import (
    BASE_CFG, DEFAULT_SOURCE, DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, SPARSE_PREFIX, V3_CFG,
    clean_checkpoint, read_source, require, runtime_info, sha256, tensor_rows, verify_module,
    verify_protected, verify_reloads, write_json,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import GSDRAIFIV3, QueryLocalDeformableRelationV3
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
    added = [r for r in rows if r["name"].startswith(SPARSE_PREFIX)]
    require({r["name"] for r in added} == {SPARSE_PREFIX + n for n, _ in target.model[9].sparse_relation.named_parameters()},
            "New optimizer coverage differs.")
    require(len(added) == 5 and sum(r["numel"] for r in added) == 88064, "Projection weight count differs.")
    require(all(r["group"] == "weight" and r["weight_decay"] == 0.0001 and r["warmup_start_lr"] == 0
                for r in added), "Local weights incorrectly grouped.")
    require(not any("bias" in r["name"] for r in added), "Unexpected new bias.")
    base_counts = {g["param_group"]: len(g["params"]) for g in a.param_groups}
    require(counts == {k: v + (len(added) if k == "weight" else 0) for k, v in base_counts.items()},
            "Optimizer groups changed beyond the five new weights.")
    source = inspect.getsource(RTDETRTrainer._do_train)
    require('x.get("param_group") == "bias"' in source, "Trainer warmup rule changed.")
    return {"counts": counts, "baseline_counts": base_counts, "new_parameters": added, "all_parameters": rows,
            "common_membership_equal": True,
            "no_missing_or_duplicate_parameters": True,
            "lr0": 0.0005, "weight_decay": 0.0001, "warmup_bias_lr": 0.1, "status": "passed"}


def audit_initialization(source, initialized):
    _, base_weights = read_source(source)
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "V3 contains training state.")
    require(checkpoint.get("gsdr_aifi_v3_provenance", {}).get("source_sha256") == SOURCE_SHA256,
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
        fresh_target = RTDETRDetectionModel(str(V3_CFG), nc=nc, verbose=False)
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
        require(added == {SPARSE_PREFIX + k for k in target.model[9].sparse_relation.state_dict()}, "Unexpected extra states.")
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
            require(sum(p.numel() for p in target.parameters()) == 20170836, "nc=1 parameter count differs.")
            optimizer_report = audit_optimizer(baseline, target)
        del baseline, target, a, b
        gc.collect()
    return report, optimizer_report


def audit_real_train_api(initialized, nc, expected, expected_rng):
    """Run the actual Model.train -> RTDETRTrainer.get_model rebuild, stopping before data/steps."""
    from train_rtdetr_r18_lite_gsdr_aifi_v3 import actual_args_check, build_locked_args, DEFAULT_NAME
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


def audit_gradient_opening():
    torch.manual_seed(42)
    relation = QueryLocalDeformableRelationV3(32, 32, 4)
    optimizer = real_optimizer(relation)
    x, probe = torch.randn(2, 32, 8, 10), torch.randn(2, 32, 8, 10)
    history = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        (relation(x) * probe).sum().backward()
        magnitudes = gradient_magnitudes(relation)
        if step == 0:
            require(magnitudes["output_proj.weight"] > 0,
                    "Zero output projection received no first-backward gradient.")
            require(all(v == 0 for k, v in magnitudes.items() if k != "output_proj.weight"),
                    "Expected the zero output boundary to block upstream first-step gradients.")
        history.append(magnitudes)
        optimizer.step()  # Actual ordinary AdamW updates; no manually randomized boundaries.
    required = list(dict(relation.named_parameters()))
    require(all(history[-1][k] > 0 for k in required), "Upstream weights remained blocked after optimizer updates.")
    return {"optimizer_steps": 4, "gradient_abs_sum_by_step": history, "upstream_weights": required, "status": "passed"}


def audit_module(mode):
    device = "cpu" if mode == "cpu_fp32" else "cuda"
    amp, half = mode == "cuda_amp_fp16", mode == "cuda_half"
    torch.manual_seed(42)
    module = GSDRAIFIV3(256, 1024, 8).to(device)
    if half:
        module.half()
    results = []
    for opened in (False, True):
        if opened:
            with torch.no_grad():
                module.sparse_relation.output_proj.weight.normal_(std=0.02)
        module.zero_grad(set_to_none=True)
        x = torch.randn(2, 256, 8, 10, device=device, dtype=torch.float16 if half else torch.float32,
                        requires_grad=True)
        observed = {}
        def recorder(name, function):
            def record(*args, **kwargs):
                out = function(*args, **kwargs)
                observed.setdefault(name, []).append(str(out.dtype))
                return out
            return record
        from contextlib import ExitStack
        import torch.nn.functional as F
        with ExitStack() as stack:
            for name in ("softmax", "tanh", "atanh"):
                stack.enter_context(patch("torch." + name, side_effect=recorder(name, getattr(torch, name))))
            stack.enter_context(patch("torch.nn.functional.grid_sample", side_effect=recorder("grid_sample", F.grid_sample)))
            original_aggregate = module.sparse_relation.aggregate
            stack.enter_context(patch.object(module.sparse_relation, "aggregate",
                                            side_effect=recorder("aggregate", original_aggregate)))
            norm_hook = module.sparse_relation.token_norm.register_forward_hook(
                lambda m, a, out: observed.setdefault("token_norm", []).append(str(out.dtype)))
            stack.callback(norm_hook.remove)
            if amp:
                stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.float16))
            output = module(x)
        for name in ("softmax", "tanh", "atanh", "grid_sample", "aggregate", "token_norm"):
            require(observed.get(name) and set(observed[name]) == {"torch.float32"},
                    f"{mode}: {name} was not actually FP32: {observed}")
        require(torch.isfinite(output).all(), f"{mode}: non-finite output.")
        (output.float() * torch.randn_like(output.float())).mean().backward()
        require(x.grad is not None and torch.isfinite(x.grad).all(), f"{mode}: invalid input gradient.")
        grads = gradient_magnitudes(module.sparse_relation)
        require(grads["output_proj.weight"] > 0, f"{mode}: output boundary gradient is zero.")
        if opened:
            require(all(v > 0 for v in grads.values()), f"{mode}: opened projection gradient is zero: {grads}")
        results.append({"output_projection_open": opened, "sensitive_dtypes": observed,
                        "output_dtype": str(output.dtype), "gradient_abs_sum": grads})
    return {"mode": mode, "checks": results, "status": "passed"}


def audit_cuda(initialized):
    if not torch.cuda.is_available():
        return {"status": "unverified", "reason": "CUDA unavailable; rerun this audit on server"}
    result = {mode: audit_module(mode) for mode in ("cuda_fp32", "cuda_amp_fp16", "cuda_half")}
    weights = RTDETR(str(initialized)).model
    for nc in (1, 80):
        model, _ = trainer_build(weights.yaml, weights, nc)
        model.cuda().half()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.float16))
        require(output[0].shape == (1, 300, nc + 4) and torch.isfinite(output[0]).all(),
                f"CUDA nc={nc} 640 half failed.")
        result[f"nc{nc}_complete_640_half"] = {"shape": list(output[0].shape), "status": "passed"}
        del model, output
        torch.cuda.empty_cache()
    result["status"] = "passed"
    return result


def run_regressions(report_dir):
    result = {}
    for pattern, expected in (("test_gsdr_aifi.py", None), ("test_gsdr_aifi_v2.py", None),
                              ("test_gsdr_aifi_v2_tools.py", None), ("test_gsdr_aifi_v3.py", None),
                              ("test_gsdr_aifi_v3_tools.py", None)):
        suite = unittest.TestLoader().discover(str(ROOT / "ultralytics-main/tests"), pattern=pattern)
        stream = io.StringIO()
        ran = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        log = report_dir / (pattern + ".log")
        log.write_text(stream.getvalue(), encoding="utf-8")
        result[pattern] = {"run": ran.testsRun, "failures": len(ran.failures), "errors": len(ran.errors),
                           "skipped": [{"test": str(t), "reason": reason} for t, reason in ran.skipped], "log": str(log)}
        require(ran.wasSuccessful() and ran.testsRun > 0 and (expected is None or ran.testsRun == expected),
                f"Regression failure/count mismatch: {log}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--initialized", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/gsdr_aifi_v3/audit.json")
    parser.add_argument("--require-torch", help="Fail unless this exact base PyTorch version is installed (server: 2.1.2).")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA cannot be audited.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"runtime": runtime_info(), "status": "running", "dataset_io": False, "formal_training": False}
    try:
        if args.require_torch:
            require(str(torch.__version__).split("+")[0] == args.require_torch, "PyTorch version differs from --require-torch.")
        if args.require_cuda:
            require(torch.cuda.is_available(), "--require-cuda requested but CUDA is unavailable.")
        report["protected"] = verify_protected()
        base, target = YAML.load(BASE_CFG), YAML.load(V3_CFG)
        base["head"][1] = [-1, 1, "GSDRAIFIV3", [1024, 8, 128, 4, 4, 2.0]]
        require(base == target, "YAML changed beyond specified layer 9.")
        print("Auditing strict source mapping and real trainer nc=1/nc=80 construction...", flush=True)
        report["initialization"], report["optimizer"] = audit_initialization(args.source, args.initialized)
        report["gradient_opening"] = audit_gradient_opening()
        report["cpu_fp32"] = audit_module("cpu_fp32")
        print("Auditing CUDA and regression suites...", flush=True)
        report["cuda"] = audit_cuda(args.initialized)
        report["regressions"] = run_regressions(args.report.parent)
        report["status"] = "passed" if report["cuda"]["status"] == "passed" else "cpu_passed_cuda_unverified"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        write_json(args.report, report)
        print(f"Audit {report['status']}: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
