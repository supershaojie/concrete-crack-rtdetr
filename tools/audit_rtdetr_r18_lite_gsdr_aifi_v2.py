"""Audit C16 with real trainer construction/optimizer and synthetic inputs only; never val/test."""
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

from init_rtdetr_r18_lite_gsdr_aifi_v2_controlled import (
    BASE_CFG, DEFAULT_SOURCE, DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, SPARSE_PREFIX, V2_CFG,
    clean_checkpoint, read_source, require, runtime_info, sha256, tensor_rows, verify_module,
    verify_protected, verify_reloads, write_json,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import GSDRAIFIV2, SparseDeformableRelationV2
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
        require(isinstance(b, torch.Tensor) and a.shape == b.shape, f"{path}: shape mismatch.")
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), f"{path}: non-finite output.")
        require(torch.equal(a, b), f"{path}: bitwise CPU FP32 equivalence failed.")
        return [{"tensor": path, "shape": list(a.shape), "equal": True, "max_abs": 0.0}]
    if isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), f"{path}: container mismatch.")
        return [r for i, (x, y) in enumerate(zip(a, b)) for r in compare_output(x, y, f"{path}.{i}")]
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
    require(counts == {"weight": 132, "bn": 82, "bias": 143}, f"Unexpected optimizer counts: {counts}")
    for row in rows:
        if row["name"] in old:
            require(row == old[row["name"]], f"C2 optimizer membership changed: {row['name']}")
    added = [r for r in rows if r["name"].startswith(SPARSE_PREFIX)]
    positional = [r for r in added if ".position_mlp." in r["name"] and r["name"].endswith(".weight")]
    require(len(positional) == 8 and sum(r["numel"] for r in positional) == 384, "Position weight count differs.")
    require(all(r["group"] == "weight" and r["weight_decay"] == 0.0001 and r["warmup_start_lr"] == 0
                for r in positional), "Position weights incorrectly grouped.")
    require(all(r["group"] == "bias" for r in added if r["name"].endswith(".bias")), "True bias misgrouped.")
    source = inspect.getsource(RTDETRTrainer._do_train)
    require('x.get("param_group") == "bias"' in source, "Trainer warmup rule changed.")
    return {"counts": counts, "new_parameters": added, "common_membership_equal": True,
            "lr0": 0.0005, "weight_decay": 0.0001, "warmup_bias_lr": 0.1, "status": "passed"}


def audit_initialization(source, initialized):
    _, base_weights = read_source(source)
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "V2 contains training state.")
    require(checkpoint.get("gsdr_aifi_v2_provenance", {}).get("source_sha256") == SOURCE_SHA256,
            "Missing C2 provenance.")
    weights = RTDETR(str(initialized)).model
    verify_module(weights)
    require(len(weights.state_dict()) == 564, "Expected 564 V2 states.")
    rows = tensor_rows(base_weights.state_dict(), weights.state_dict())
    require(len(rows) == 533 and all(r["equal"] for r in rows), "C2 checkpoint mapping incomplete/inexact.")
    report = {"source": str(source.resolve()), "source_sha256": sha256(source),
              "initialized": str(initialized.resolve()), "initialized_sha256": sha256(initialized),
              "checkpoint_common_states": rows, "reloads": verify_reloads(initialized, weights.state_dict()),
              "entry_point": "unchanged RTDETRTrainer.get_model; __new__ fixture avoids dataset I/O",
              "seed": 42, "classes": {}}
    optimizer_report = None
    for nc in (1, 80):
        baseline, expected_rng = trainer_build(base_weights.yaml, base_weights, nc)
        target, actual_rng = trainer_build(weights.yaml, weights, nc)
        rows = tensor_rows(baseline.state_dict(), target.state_dict())
        require(len(rows) == 533 and all(r["equal"] for r in rows), f"nc={nc}: common states differ.")
        require(torch.equal(expected_rng, actual_rng), f"nc={nc}: construction RNG differs.")
        added = set(target.state_dict()) - set(baseline.state_dict())
        require(len(added) == 31 and all(k.startswith(SPARSE_PREFIX) for k in added), "Unexpected extra states.")
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
            "cpu_640_outputs": compare_output(a, b), "unfused_parameters": sum(p.numel() for p in target.parameters()),
            "state_sha256": state_hashes(target), "status": "passed",
        }
        if nc == 1:
            require(sum(p.numel() for p in target.parameters()) == 20200416, "nc=1 parameter count differs.")
            optimizer_report = audit_optimizer(baseline, target)
        del baseline, target, a, b
        gc.collect()
    return report, optimizer_report


def gradient_magnitudes(module):
    result = {}
    for name, p in module.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid gradient: {name}")
        result[name] = float(p.grad.float().abs().sum())
    return result


def audit_gradient_opening():
    torch.manual_seed(42)
    relation = SparseDeformableRelationV2(32, 32, 4, 4)
    optimizer = real_optimizer(relation)
    x, probe = torch.randn(2, 32, 8, 10), torch.randn(2, 32, 8, 10)
    history = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        (relation(x) * probe).sum().backward()
        magnitudes = gradient_magnitudes(relation)
        if step == 0:
            require(magnitudes["output_proj.weight"] > 0 and magnitudes["output_proj.bias"] > 0,
                    "Zero output projection received no first-backward gradient.")
        history.append(magnitudes)
        optimizer.step()  # Actual ordinary AdamW updates; no manually randomized boundaries.
    required = ["input_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "offset_conv.weight",
                "offset_norm.weight", "offset_out.weight", *(f"position_mlp.{g}.{fc}.weight"
                for g in range(4) for fc in ("fc1", "fc2"))]
    require(all(history[-1][k] > 0 for k in required), "Upstream weights remained blocked after optimizer updates.")
    return {"optimizer_steps": 4, "gradient_abs_sum_by_step": history, "upstream_weights": required, "status": "passed"}


def audit_module(mode):
    device = "cpu" if mode == "cpu_fp32" else "cuda"
    amp, half = mode == "cuda_amp_fp16", mode == "cuda_half"
    module = GSDRAIFIV2(256, 1024, 8).to(device)
    if half:
        module.half()
    x = torch.randn(1, 256, 8, 10, device=device, dtype=torch.float16 if half else torch.float32,
                    requires_grad=True)
    observed = []
    original = torch.matmul

    def record(*args, **kwargs):
        out = original(*args, **kwargs)
        observed.append(str(out.dtype))
        return out

    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
    with patch("torch.matmul", side_effect=record), context:
        output = module(x)
    require(observed == ["torch.float32", "torch.float32"], f"{mode}: actual sparse QK/AV dtypes: {observed}")
    require(torch.isfinite(output).all(), f"{mode}: non-finite output.")
    (output.float() * torch.randn_like(output.float())).sum().backward()
    require(x.grad is not None and torch.isfinite(x.grad).all(), f"{mode}: invalid input gradient.")
    grads = gradient_magnitudes(module)
    require(grads["sparse_relation.output_proj.weight"] > 0, f"{mode}: sparse boundary gradient is zero.")
    return {"mode": mode, "sparse_matmul_output_dtypes": observed, "output_dtype": str(output.dtype),
            "gradient_abs_sum": grads, "status": "passed"}


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
    for pattern, expected in (("test_gsdr_aifi.py", 17), ("test_gsdr_aifi_v2.py", 5),
                              ("test_gsdr_aifi_v2_tools.py", None)):
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
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/gsdr_aifi_v2/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"runtime": runtime_info(), "status": "running", "dataset_io": False, "formal_training": False}
    try:
        report["protected"] = verify_protected()
        base, target = YAML.load(BASE_CFG), YAML.load(V2_CFG)
        base["head"][1] = [-1, 1, "GSDRAIFIV2", [1024, 8, 128, 4, 4, 2, 2.0, 3]]
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
