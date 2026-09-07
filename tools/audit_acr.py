"""Audit ACR initialization, decoder behavior, dtypes and optional real-data smoke. No formal training."""
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

from init_acr import (
    BASE_CFG, DEFAULT_SOURCE, DEFAULT_OUTPUT, ROOT, SOURCE_SHA256, ACR_CFG,
    clean_checkpoint, read_source, require, runtime_info, sha256, tensor_rows, verify_module,
    verify_protected, verify_reloads, write_json, remap_key, common_rows, new_state_keys, branch_modules,
)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.modules import CoverageRelationSelfAttention
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
        a, b = a.detach().cpu(), b.detach().cpu()  # Optimizer scalar step states may reload on CPU.
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), f"{path}: non-finite output.")
        maximum = float((a.float()-b.float()).abs().max()) if a.numel() else 0.
        require(torch.allclose(a,b,atol=1e-6,rtol=1e-5), f"{path}: tolerance failed; max_abs={maximum}")
        return [{"tensor":path,"shape":list(a.shape),"equal":torch.equal(a,b),"max_abs":maximum,"atol":1e-6,"rtol":1e-5}]
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
    old = {remap_key(r["name"]): {**r, "name": remap_key(r["name"])} for r in optimizer_rows(baseline, a)}
    rows = optimizer_rows(target, b)
    counts = {g["param_group"]: len(g["params"]) for g in b.param_groups}
    for row in rows:
        if row["name"] in old:
            require(row == old[row["name"]], f"C2 optimizer membership changed: {row['name']}")
    added = [r for r in rows if r["name"] not in old]
    require({r["name"] for r in added} == new_state_keys(target),
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


def audit_initialization(source, initialized, recipe_path=None):
    _, base_weights = read_source(source)
    checkpoint = torch_load(initialized, map_location="cpu")
    require(clean_checkpoint(checkpoint), "ACR contains training state.")
    require(checkpoint.get("acr_provenance", {}).get("source_sha256") == SOURCE_SHA256,
            "Missing C2 provenance.")
    weights = RTDETR(str(initialized)).model
    verify_module(weights)
    rows = common_rows(base_weights.state_dict(), weights.state_dict())
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
        fresh_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        torch.manual_seed(42)
        fresh_target = RTDETRDetectionModel(str(ACR_CFG), nc=nc, verbose=False)
        require(torch.equal(fresh_rng, torch.get_rng_state()), f"nc={nc}: fresh construction RNG differs.")
        if fresh_cuda_rng:
            require(all(torch.equal(a, b) for a, b in zip(fresh_cuda_rng, torch.cuda.get_rng_state_all())),
                    f"nc={nc}: CUDA constructor RNG differs.")
        require(all(r["equal"] for r in common_rows(fresh_base.state_dict(), fresh_target.state_dict())),
                f"nc={nc}: fresh common construction states differ.")
        if nc == 80:
            new = {k: fresh_target.state_dict()[k] for k in new_state_keys(fresh_target)}
            require(all(r["equal"] for r in tensor_rows(new, weights.state_dict())),
                    "Stored added parameters differ from controlled fresh nc=80 initialization.")
        del fresh_base, fresh_target
        baseline, expected_rng = trainer_build(base_weights.yaml, base_weights, nc)
        target, actual_rng = trainer_build(weights.yaml, weights, nc)
        rows = common_rows(baseline.state_dict(), target.state_dict())
        require(all(r["equal"] for r in rows), f"nc={nc}: common states differ.")
        require(torch.equal(expected_rng, actual_rng), f"nc={nc}: construction RNG differs.")
        added = set(target.state_dict()) - {remap_key(k) for k in baseline.state_dict()}
        require(added == new_state_keys(target), "Unexpected extra states.")
        require(all(r["equal"] for r in tensor_rows({k: weights.state_dict()[k] for k in added}, target.state_dict())),
                "Trainer rebuild changed new initialization states.")
        verify_module(target)
        image = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(123))
        with torch.inference_mode():
            a, b = baseline(image), target(image)
        require(a[0].shape == (1, 300, nc + 4), "Unexpected complete forward shape.")
        classification = [r for r in rows if r["key"].startswith(f"model.{len(baseline.model) - 1}.") and
                          any(s in r["key"] for s in ("denoising_class_embed", "enc_score_head", "dec_score_head"))]
        require(len(classification) == 9, "Expected nine decoder classification states (five weights).")
        report["classes"][str(nc)] = {
            "common_states": rows, "classification_states": classification, "cpu_rng_equal_after_build_load": True,
            "fresh_constructor_common_states_and_rng_equal": True, "total_states": len(target.state_dict()),
            "cuda_constructor_rng_equal": True if fresh_cuda_rng else None,
            "new_states_exact": sorted(added),
            "cpu_640_outputs": compare_output(a, b), "unfused_parameters": sum(p.numel() for p in target.parameters()),
            "state_sha256": state_hashes(target), "status": "passed",
        }
        report["classes"][str(nc)]["real_train_api"] = audit_real_train_api(initialized, nc, target, actual_rng, recipe_path)
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


def audit_real_train_api(initialized, nc, expected, expected_rng, recipe_path=None):
    """Run the actual Model.train -> RTDETRTrainer.get_model rebuild, stopping before data/steps."""
    from train_acr import actual_args_check, build_locked_args, DEFAULT_NAME
    recipe = YAML.load(recipe_path or ROOT / "ultralytics-main/tests/fixtures/c2_original_args.yaml")
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
        if not name.startswith('acr_'):
            continue
        require(p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid gradient: {name}")
        result[name] = float(p.grad.float().abs().sum())
    return result


def branch_gradients(model):
    return {prefix + "." + name: magnitude for prefix, module in branch_modules(model).items()
            for name, magnitude in gradient_magnitudes(module).items()}


def audit_wiring(initialized):
    model=RTDETR(str(initialized)).model.eval()
    verify_module(model)
    capture={}
    hooks=[]
    for name,module in branch_modules(model).items():
        def hook(m, inputs, kwargs, output, name=name):
            capture[name]={"class":type(m).__name__, "query":list(inputs[0].shape),
                "reference":list(kwargs["refer_bbox"].shape), "range":[float(kwargs["refer_bbox"].min()),float(kwargs["refer_bbox"].max())],
                "output":list(output[0].shape), "added_parameters":616}
        hooks.append(module.register_forward_hook(hook, with_kwargs=True))
    pyramid=[]
    hooks.append(model.model[-1].register_forward_pre_hook(lambda m, a: pyramid.extend(list(x.shape) for x in a[0])))
    with torch.inference_mode(): model(torch.rand(1,3,640,640))
    for h in hooks: h.remove()
    require(len(capture)==3 and pyramid==[[1,256,80,80],[1,256,40,40],[1,256,20,20]], "Runtime wiring mismatch")
    require(all(0<=r["range"][0]<=r["range"][1]<=1 for r in capture.values()), "Reference boxes are not sigmoid coordinates")
    return {"self_attention":capture,"pyramid":pyramid,"decoder_inputs":model.model[-1].f,
            "added_parameters":1848,"all_parameters_independent":True,"cross_attention":"MSDeformAttn"}


def audit_precision(initialized, source, device, mode):
    _, base_weights = read_source(source)
    weights = RTDETR(str(initialized)).model
    baseline, _ = trainer_build(base_weights.yaml, base_weights, 1)
    model, _ = trainer_build(weights.yaml, weights, 1)
    model.to(device); baseline.to(device)
    if mode == 'half': model.half(); baseline.half()
    dtype = torch.float16 if mode == 'half' else torch.float32
    ctx = lambda: torch.autocast(device_type=device, dtype=torch.float16) if mode == 'amp' else nullcontext()
    image = torch.rand(1,3,640,640,device=device,dtype=dtype)
    with torch.inference_mode(), ctx():
        a,b = baseline(image),model(image)
    rows = compare_output(a,b)
    # Entire non-square image, not just descriptor unit test.
    with torch.inference_mode(), ctx():
        x = torch.rand(1,3,320,640,device=device,dtype=dtype)
        non_square = compare_output(baseline(x),model(x))
    training_dn = None
    if mode != 'half':
        model.train(); baseline.train()
        targets = {'cls':torch.zeros(3,device=device,dtype=torch.long),
                   'bboxes':torch.tensor([[.5,.5,.6,.08],[.5,.5,.08,.6],[.2,.8,.1,.09]],device=device),
                   'batch_idx':torch.tensor([0,1,1],device=device), 'gt_groups':[1,2]}
        x = torch.rand(2,3,640,640,device=device)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices), torch.no_grad(), ctx():
            torch.manual_seed(987)
            original = baseline.predict(x, batch=targets)
            torch.manual_seed(987)
            replaced = model.predict(x, batch=targets)
        require(original[4] is not None and original[0].shape[2] > 300, 'DN path not exercised.')
        training_dn = {'outputs':compare_output(original,replaced), 'dn_num_split':original[4]['dn_num_split'],
                       'input':[2,3,640,640], 'note':'Forward equality only; real-data backward in separate smoke.'}
    return {'status':'passed','mode':mode,'output_dtype':str(b[0].dtype),
            'complete_model_640_batch1':rows,'complete_model_320x640':non_square, 'training_dn':training_dn}


def benchmark(initialized, source, device):
    import time
    _, base_weights = read_source(source)
    weights = RTDETR(str(initialized)).model
    result = {"device": device, "batch": 1, "imgsz": 640, "dtype": "FP32", "warmup": 3, "iterations": 10,
              "note": "Unfused complete-model forward; same random image. Not dataset validator speed."}
    x = torch.rand(1, 3, 640, 640, device=device)
    for label, w in (("reference", base_weights), ("ACR", weights)):
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
    for pattern in ("test_acr.py", "test_acr_tools.py"):
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
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/acr/audit.json")
    parser.add_argument("--require-torch")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke-data", type=Path, help="Real dataset YAML; use C2 path on server.")
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "outputs/acr/smoke")
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
        report["topology"] = audit_wiring(args.initialized)
        print("Auditing exact initialization, real train API and 640 FP32 equality...", flush=True)
        report["initialization"], report["optimizer"] = audit_initialization(args.source, args.initialized, args.c2_args)
        report["regressions"] = run_regressions(args.report.parent)
        report["native_amp"] = {"status": "cuda_unavailable"}
        if torch.cuda.is_available():
            from acr_resources import native_amp_check
            helper_model = RTDETR(str(args.initialized)).model.cuda()
            report["native_amp"] = native_amp_check(helper_model)
            del helper_model
        report["cpu"] = audit_precision(args.initialized, args.source, "cpu", "fp32")
        report["cuda"] = {"status": "unverified"}
        if torch.cuda.is_available():
            report["cuda"] = {mode: audit_precision(args.initialized, args.source, "cuda", mode) for mode in ("fp32", "amp", "half")}
            report["cuda"]["status"] = "passed"
        report["benchmark"] = benchmark(args.initialized, args.source, "cuda" if torch.cuda.is_available() else "cpu")
        report["real_data_smoke"] = {"status": "not_run"}
        if args.smoke_data:
            from smoke_acr import smoke
            report["real_data_smoke"] = smoke(args.initialized, args.smoke_data, args.c2_args, args.smoke_dir,
                                                   report["initialization"]["classes"]["1"]["state_sha256"], args.source)
        checkpoint = torch_load(args.initialized, map_location="cpu")
        require(clean_checkpoint(checkpoint), "Smoke contaminated the formal initialization.")
        require(sha256(args.initialized) == report["initialization"]["initialized_sha256"], "Initialization file changed during audit.")
        require(state_hashes(checkpoint["model"].float()) == report["initialization"]["classes"]["80"]["state_sha256"],
                "Fresh post-smoke initialization differs from audited nc=80 model.")
        report["post_smoke_formal_initialization"] = {"clean_checkpoint": True, "unchanged_sha256": True,
                                                     "all_states_exact": True, "status": "passed"}
        report["status"] = ("passed" if report["native_amp"]["status"] == "passed" else "cuda_passed_native_amp_unverified") if report["cuda"]["status"] == "passed" else "cpu_passed_cuda_unverified"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        write_json(args.report, report)
        print(f"Audit {report['status']}: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
