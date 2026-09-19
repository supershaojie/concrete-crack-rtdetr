#!/usr/bin/env python3
"""Bounded DPR mathematics/structure/deployment checks; never starts training or test.

Random model checks are explicitly not public-source initialization evidence. Real
detection-loss, native optimizer/resume and B16 CUDA evidence belong to preflight.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
import torch.nn.functional as F
from ultralytics.nn.modules import ConvNormLayer, DPRConvNormLayer
from ultralytics.nn.modules.dpr import count_dpr_conv_norm, profile_dpr, thop_dpr_semantics
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA

TARGET = "model.5.blocks.1.branch2b"
FP32_ATOL, FP32_RTOL = 2e-5, 2e-4
FP64_ATOL, FP64_RTOL = 1e-10, 1e-9
HALF_ATOL, HALF_RTOL = 2e-3, 2e-2
VARIANTS = {
    "cbr_lif_dpr_v1": ("rtdetr-resnet18-lite-cbr-lif-down.yaml", "rtdetr-resnet18-lite-cbr-lif-dpr-v1.yaml", 20152837, 19948037, 19944965),
    "dpr_v1": ("rtdetr-resnet18-lite.yaml", "rtdetr-resnet18-lite-dpr-v1.yaml", 20085844, 19880788, 19877716),
}
CFG = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"


def source_identity():
    paths = [ROOT / "tools/check_dpr.py", ROOT / "ultralytics-main/ultralytics/nn/modules/dpr.py",
             ROOT / "ultralytics-main/ultralytics/nn/modules/__init__.py", ROOT / "ultralytics-main/ultralytics/nn/tasks.py",
             ROOT / "ultralytics-main/ultralytics/utils/torch_utils.py"]
    paths += [CFG / name for row in VARIANTS.values() for name in row[:2]]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
            for p in paths}


def validate_math_report(report, variant):
    """Revalidate actual mathematical/structural evidence, not a summary marker."""
    def need(condition, message):
        if not condition:
            raise RuntimeError("DPR mathematical evidence rejected: " + message)
    need(report.get("report_kind") == "dpr_mathematical_structural_v1", "wrong report kind")
    need(report.get("status") in {"PASSED","PRECISION_NOTE"}, "incomplete report")
    need(report.get("source_identity") == source_identity(), "mathematical source/config identity changed")
    math = report.get("checks",{}).get("cpu_fp64_math",{})
    need(math.get("status") == "PASSED" and math.get("dtype") == "float64" and math.get("device") == "cpu", "CPU FP64 absent")
    need(math.get("atol") == FP64_ATOL and math.get("rtol") == FP64_RTOL, "FP64 thresholds changed")
    need(math.get("directional_gradcheck") is True and math.get("forward_conv2d_calls") == 1, "numerical/autograd/single-convolution evidence absent")
    need(math.get("half_kernel_arithmetic") == "float32", "kernel arithmetic not FP32")
    for name in DPRConvNormLayer.parameter_names:
        values = math.get("mappings",{}).get(name,{})
        need(0 <= values.get("max_abs", float("inf")) <= FP64_ATOL and
             0 <= values.get("zero_sum_max_abs",float("inf")) <= FP64_ATOL, "mapping/zero sum: " + name)
    for name in ("input","conv.weight",*DPRConvNormLayer.parameter_names):
        value = math.get("gradients",{}).get(name,{})
        need(0 < value.get("gradient_norm",0) < float("inf") and
             0 <= value.get("max_abs_difference",float("inf")) <= FP64_ATOL, "gradient: " + name)
    checks = report.get("checks",{}).get(variant,{})
    need(checks.get("status") in {"PASSED","PRECISION_NOTE"} and checks.get("variant") == variant, "variant absent")
    need(checks.get("nc") == 1 and checks.get("target") == TARGET and checks.get("graph_nodes") == 27 and
         checks.get("changed_top_level_definition") == [5], "structural target/graph differs")
    need(checks.get("public_state_equal") is True and checks.get("new_buffers") == [] and
         set(checks.get("new_state_keys",[])) == {TARGET+"."+n for n in DPRConvNormLayer.parameter_names}, "common/new state inventory differs")
    _,_,training,intermediate,deployed = VARIANTS[variant]
    params = checks.get("parameters",{})
    need(params == {"parent_unfused":training-3072,"candidate_unfused":training,"new_trainable":3072,
                    "native_fuse_without_dpr_fold":intermediate,"standard_deploy":deployed,"parent_native_fused":deployed}, "parameter counts differ")
    need(checks.get("shapes",{}).get("target") == [1,128,80,80], "640 input evidence absent")
    zero = checks.get("zero_initialization",{})
    need(zero.get("status") == "PASSED", "zero initialization not passed")
    for key in ("target","P3","P4","P5","encoder_features","candidate_scores","output"):
        value = zero.get("details",{}).get(key,{})
        need(value.get("raw_allclose") is True and value.get("finite") is True and
             value.get("atol") == FP32_ATOL and value.get("rtol") == FP32_RTOL, "zero continuous evidence missing: "+key)
    lifecycle = checks.get("learned_lifecycle",{})
    required = {"training_state_reload","training_full_reload","ema_retains_nonzero","dpr_fold_only","deploy_state_reload",
                "deploy_full_reload_idempotent","native_fuse","native_fuse_again","autobackend_saved_best","native_fused_saved_reload"}
    need(required <= set(lifecycle.get("checks",{})), "missing lifecycle leaves")
    need(lifecycle.get("status") in {"PASSED","PRECISION_NOTE"}, "lifecycle failed")
    for key in required:
        entry = lifecycle["checks"][key]
        need(entry.get("status") in {"PASSED","PRECISION_NOTE"}, "lifecycle leaf failed: "+key)
        if "details" in entry:
            for name in ("target","P3","P4","P5","encoder_features","candidate_scores"):
                metric = entry["details"].get(name,{})
                need(metric.get("raw_allclose") is True and metric.get("finite") is True and
                     metric.get("atol") == FP32_ATOL and metric.get("rtol") == FP32_RTOL, "lifecycle continuous path: "+key+":"+name)
            metric = entry["details"].get("output",{})
            if entry["status"] == "PASSED":
                need(metric.get("raw_allclose") is True, "lifecycle output not passed: "+key)
            else:
                need(metric.get("finite") is True and entry["details"].get("candidate_index_changes",0) > 0 and
                     entry["details"].get("fixed_candidate_replay_output",{}).get("raw_allclose") is True,
                     "unexplained precision note: "+key)
    backend = lifecycle["checks"]["autobackend_saved_best"]
    need(backend.get("deployed") is True and backend.get("norm_retained") is True and
         backend.get("comparison",{}).get("raw_allclose") is True, "AutoBackend evidence incomplete")
    flops = checks.get("gflops",{})
    need(flops.get("parent_unfused",0) > 0 and flops.get("candidate_corrected") == flops.get("parent_unfused") and
         flops.get("standard_deploy") == flops.get("parent_native_fused") and
         abs(flops.get("missing_functional_conv",0)-1.8874368) < 1e-9, "functional convolution FLOPs not corrected")
    return True


def comparison(a, b, atol=FP32_ATOL, rtol=FP32_RTOL):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"raw_allclose": bool(finite and torch.allclose(a, b, atol=atol, rtol=rtol)),
            "max_abs": float(d.max()) if d.numel() else 0.,
            "relative_L2": float(d.norm() / a.norm().clamp_min(1e-30)),
            "exceed_fraction": float((d > atol + rtol * a.abs()).float().mean()),
            "finite": finite, "atol": atol, "rtol": rtol}


def _reference_kernels(module):
    t, h, v, a = (getattr(module, key) for key in module.parameter_names)
    z = t[:, 0] * 0
    cd = torch.stack([t[:, i] if i != 4 else -sum(t[:, j] for j in range(9) if j != 4)
                      for i in range(9)], 1).reshape(-1, 3, 3)
    hd = torch.stack((h[:, 0], z, -h[:, 0], h[:, 1], z, -h[:, 1], h[:, 2], z, -h[:, 2]), 1).reshape(-1, 3, 3)
    vd = torch.stack((v[:, 0], v[:, 1], v[:, 2], z, z, z, -v[:, 0], -v[:, 1], -v[:, 2]), 1).reshape(-1, 3, 3)
    ad = torch.stack((a[:, 0]-a[:, 3], a[:, 1]-a[:, 0], a[:, 2]-a[:, 1], a[:, 3]-a[:, 6],
                      a[:, 4]-a[:, 4], a[:, 5]-a[:, 2], a[:, 6]-a[:, 7], a[:, 7]-a[:, 8],
                      a[:, 8]-a[:, 5]), 1).reshape(-1, 3, 3)
    return cd, hd, vd, ad


def mathematics():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(813)
        module = DPRConvNormLayer(ConvNormLayer(128, 128, 3, 1)).double().eval()
        with torch.no_grad():
            for key in module.parameter_names:
                getattr(module, key).normal_(0, .07)
        actual, reference = module.difference_kernels(), _reference_kernels(module)
        maps = {}
        for key, a, b in zip(module.parameter_names, actual, reference):
            torch.testing.assert_close(a, b, atol=FP64_ATOL, rtol=FP64_RTOL)
            torch.testing.assert_close(a.sum((1, 2)), torch.zeros(128, dtype=torch.float64), atol=FP64_ATOL, rtol=0)
            maps[key] = {"max_abs": float((a-b).abs().max()), "zero_sum_max_abs": float(a.sum((1,2)).abs().max())}
        effective = module.get_equivalent_kernel()
        off_diagonal = ~torch.eye(128, dtype=torch.bool)
        assert torch.equal(effective[off_diagonal], module.conv.weight[off_diagonal])
        x = torch.randn(1, 128, 4, 5, dtype=torch.float64, requires_grad=True)
        y = F.conv2d(x, effective, padding=1)
        r = F.conv2d(x, module.conv.weight, padding=1) + F.conv2d(x, sum(reference).unsqueeze(1), padding=1, groups=128)
        torch.testing.assert_close(y, r, atol=FP64_ATOL, rtol=FP64_RTOL)
        upstream = torch.randn_like(y)
        variables = (x, module.conv.weight, *(getattr(module, key) for key in module.parameter_names))
        gy = torch.autograd.grad((y*upstream).sum(), variables, retain_graph=True)
        gr = torch.autograd.grad((r*upstream).sum(), variables)
        gradients = {}
        for key, a, b in zip(("input", "conv.weight", *module.parameter_names), gy, gr):
            torch.testing.assert_close(a, b, atol=FP64_ATOL, rtol=FP64_RTOL)
            assert torch.isfinite(a).all() and a.abs().max() > 0
            gradients[key] = {"max_abs_difference": float((a-b).abs().max()), "gradient_norm": float(a.norm())}
        assert torch.count_nonzero(gy[2][:,4]) == 0 and torch.count_nonzero(gy[5][:,4]) == 0
        # Fast directional numerical check retains all 128 channels and full dense W.
        from torch.func import functional_call
        keys = ("conv.weight", *module.parameter_names)
        def fn(x_value, *parameter_values):
            return functional_call(module, dict(zip(keys, parameter_values)), (x_value,))
        numeric = torch.autograd.gradcheck(fn, variables, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
        calls = []
        native_conv = F.conv2d
        def count_conv(*args, **kwargs):
            calls.append(1)
            return native_conv(*args, **kwargs)
        with patch("torch.nn.functional.conv2d", count_conv), torch.no_grad():
            module(x)
        assert len(calls) == 1
        assert module.half().get_equivalent_kernel().dtype == torch.float32
        return {"status": "PASSED", "device": "cpu", "dtype": "float64", "atol": FP64_ATOL,
                "rtol": FP64_RTOL, "mappings": maps, "gradients": gradients,
                "directional_gradcheck": numeric, "forward_conv2d_calls": len(calls),
                "half_kernel_arithmetic": "float32", "nullspace": "CD center and AD center gradients exactly zero; groups have nonzero effective gradients"}


def _capture(model, image, fixed_indices=None):
    """Capture continuous paths; remove every temporary closure hook before returning."""
    values, handles = {}, []
    layers = {"target": model.get_submodule(TARGET), "P3": model.model[5], "P4": model.model[6],
              "P5": model.model[7], "encoder_features": model.model[-1].enc_output,
              "candidate_scores": model.model[-1].enc_score_head}
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(lambda m, a, o, key=name: values.__setitem__(key, o.detach().clone())))
    native_topk = torch.topk
    def fixed_topk(input, k, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else -1)
        if fixed_indices is not None and dim == 1 and k == model.model[-1].num_queries and input.ndim == 2:
            indices = fixed_indices.to(input.device)
            return torch.return_types.topk((input.gather(1, indices), indices))
        return native_topk(input, k, *args, **kwargs)
    try:
        with torch.no_grad(), patch("torch.topk", fixed_topk):
            output = model(image)
            values["output"] = (output[0] if isinstance(output, (tuple,list)) else output).detach().clone()
        values["candidate_indices"] = native_topk(values["candidate_scores"].max(-1).values,
                                                   model.model[-1].num_queries, dim=1).indices
    finally:
        for handle in handles:
            handle.remove()
    return values


def _compare_capture(before, after, model=None, image=None, atol=FP32_ATOL, rtol=FP32_RTOL):
    details = {name: comparison(before[name], after[name],atol,rtol) for name in before if name != "candidate_indices"}
    index_changes = int((before["candidate_indices"] != after["candidate_indices"]).sum())
    details["candidate_index_changes"] = index_changes
    continuous = all(details[key]["raw_allclose"] for key in details if isinstance(details[key], dict) and key != "output")
    output_ok = details["output"]["raw_allclose"]
    status = "PASSED" if continuous and output_ok else "FAILED"
    if continuous and not output_ok and index_changes and model is not None:
        replay = _capture(model, image, fixed_indices=before["candidate_indices"])
        details["fixed_candidate_replay_output"] = comparison(before["output"], replay["output"],atol,rtol)
        if details["fixed_candidate_replay_output"]["raw_allclose"]:
            status = "PRECISION_NOTE"
    return {"status": status, "details": details}


def _native_fuse_without_dpr(model):
    # Audit-only skip of DPR fold to measure the intermediate representation.
    with patch.object(DPRConvNormLayer, "switch_to_deploy", lambda self: self):
        model.fuse(verbose=False)
    return model


def audit_learned_model(model, image):
    """Audit a learned nonzero model; no live mutation and no retained large artifacts.

    Caller must document where its nonzero updates came from. This function does
    not certify detection loss, optimizer/resume, CUDA AMP or capacity by itself.
    """
    model = deepcopy(model).eval()
    target = model.get_submodule(TARGET)
    assert not target.deployed and all(torch.count_nonzero(getattr(target, n)) > 0 for n in target.parameter_names)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    report = {"device": str(device), "dtype": str(dtype), "input_shape": list(image.shape), "checks": {}}
    checks = report["checks"]
    before = _capture(model, image)
    original_dpr = {n: getattr(target,n).detach().clone() for n in target.parameter_names}
    temporary_root = ROOT / "outputs/dpr"
    temporary_root.mkdir(parents=True, exist_ok=True)
    # Keep local checkpoints under this workspace. The upstream downloader strips
    # apostrophes in Windows user-temp paths (e.g. o'v'o), falsely treating a real
    # local checkpoint as a downloadable asset.
    with tempfile.TemporaryDirectory(prefix="dpr_lifecycle_", dir=temporary_root) as tmp:
        tmp = Path(tmp)
        # state_dict and full-module training representations both retain nonzero DPR.
        torch.save(model.state_dict(), tmp / "training_state.pt")
        state_copy = deepcopy(model)
        state_copy.load_state_dict(torch.load(tmp / "training_state.pt", map_location=device, weights_only=True), strict=True)
        checks["training_state_reload"] = _compare_capture(before, _capture(state_copy, image))
        del state_copy
        torch.save({"model": model, "ema": None, "train_args": {"task": "detect"}, "epoch": 0}, tmp / "best.pt")
        full = torch.load(tmp / "best.pt", map_location=device, weights_only=False)["model"]
        checks["training_full_reload"] = _compare_capture(before, _capture(full, image))
        for name, value in original_dpr.items():
            assert torch.equal(getattr(full.get_submodule(TARGET), name), value)
        del full
        ema = ModelEMA(model)
        ema.update(model)
        assert all(torch.count_nonzero(getattr(ema.ema.get_submodule(TARGET), n)) > 0 for n in target.parameter_names)
        checks["ema_retains_nonzero"] = {"status": "PASSED", "updates": ema.updates}
        del ema
        folded = deepcopy(model)
        norm_before = {k:v.clone() for k,v in folded.get_submodule(TARGET).norm.state_dict().items()}
        folded.get_submodule(TARGET).switch_to_deploy()
        assert all(torch.equal(v,folded.get_submodule(TARGET).norm.state_dict()[k]) for k,v in norm_before.items())
        assert all(not hasattr(folded.get_submodule(TARGET),n) for n in target.parameter_names)
        checks["dpr_fold_only"] = _compare_capture(before, _capture(folded,image), folded,image)
        torch.save(folded.state_dict(), tmp / "deploy_state.pt")
        deploy_state = deepcopy(model)
        deploy_state.get_submodule(TARGET).switch_to_deploy()
        deploy_state.load_state_dict(torch.load(tmp / "deploy_state.pt", map_location=device, weights_only=True), strict=True)
        checks["deploy_state_reload"] = _compare_capture(_capture(folded,image), _capture(deploy_state,image))
        del deploy_state
        torch.save(folded, tmp / "deploy_only.pt")
        deployed = torch.load(tmp / "deploy_only.pt", map_location=device, weights_only=False)
        deployed.get_submodule(TARGET).switch_to_deploy()
        checks["deploy_full_reload_idempotent"] = _compare_capture(_capture(folded,image), _capture(deployed,image))
        del deployed
        folded.fuse(verbose=False)
        fused_capture = _capture(folded, image)
        checks["native_fuse"] = _compare_capture(before, fused_capture, folded, image)
        folded.fuse(verbose=False)
        checks["native_fuse_again"] = _compare_capture(fused_capture, _capture(folded,image))
        # Actual saved-file backend path, including load_checkpoint and native fuse.
        backend = AutoBackend(model=str(tmp / "best.pt"), device=device, fp16=dtype == torch.float16, fuse=True, verbose=False)
        backend.eval()
        with torch.no_grad():
            backend_output = backend(image)
        backend_output = backend_output[0] if isinstance(backend_output, (tuple,list)) else backend_output
        output_comparison = comparison(fused_capture["output"], backend_output)
        assert backend.model.get_submodule(TARGET).deployed
        checks["autobackend_saved_best"] = {"status": "PASSED" if output_comparison["raw_allclose"] else "FAILED",
                                             "comparison": output_comparison, "deployed": True,
                                             "norm_retained": isinstance(backend.model.get_submodule(TARGET).norm, torch.nn.BatchNorm2d)}
        torch.save(folded, tmp / "native_fused.pt")
        native_reload = torch.load(tmp / "native_fused.pt", map_location=device, weights_only=False)
        native_reload.fuse(verbose=False)
        checks["native_fused_saved_reload"] = _compare_capture(fused_capture, _capture(native_reload,image))
    report["status"] = "FAILED" if any(v["status"] == "FAILED" for v in checks.values()) else (
        "PRECISION_NOTE" if any(v["status"] == "PRECISION_NOTE" for v in checks.values()) else "PASSED")
    return report


def audit_cuda_half(model, image):
    """Explicit CUDA half audit on learned weights; FP32 fold precedes half cast.

    Prespecified half tolerances are separate from the unchanged strict FP32
    tolerance. Sorting changes need finite continuous paths and fixed replay.
    """
    if next(model.parameters()).device.type != "cuda":
        return {"status":"PENDING", "reason":"CUDA model required"}
    model = deepcopy(model).float().eval()
    target = model.get_submodule(TARGET)
    assert all(torch.count_nonzero(getattr(target,n)) > 0 for n in target.parameter_names)
    image = image.to(device=next(model.parameters()).device, dtype=torch.float32)
    image_half = image.half()
    reference = _capture(model,image)
    native_half = deepcopy(model).half()
    half_capture = _capture(native_half,image_half)
    checks = {"native_half_vs_fp32": _compare_capture(reference,half_capture,native_half,image_half,HALF_ATOL,HALF_RTOL)}
    deployed = deepcopy(model)
    deployed.fuse(verbose=False)  # FP32 DPR kernel fold and parent-symmetric native fusion.
    deployed.half()
    deployed_capture = _capture(deployed,image_half)
    checks["fp32_fold_then_half"] = _compare_capture(half_capture,deployed_capture,deployed,image_half,HALF_ATOL,HALF_RTOL)
    temporary_root = ROOT / "outputs/dpr"
    temporary_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dpr_cuda_half_",dir=temporary_root) as tmp:
        tmp = Path(tmp)
        torch.save(deployed,tmp/"deploy_half.pt")
        restored = torch.load(tmp/"deploy_half.pt",map_location=image.device,weights_only=False)
        checks["half_deploy_reload"] = _compare_capture(deployed_capture,_capture(restored,image_half),atol=HALF_ATOL,rtol=HALF_RTOL)
        torch.save({"model":model,"ema":None,"epoch":0,"train_args":{"task":"detect"}},tmp/"best.pt")
        backend = AutoBackend(model=str(tmp/"best.pt"),device=image.device,fp16=True,fuse=True,verbose=False)
        backend.eval()
        with torch.no_grad():
            output = backend(image_half)
        output = output[0] if isinstance(output,(tuple,list)) else output
        diff = comparison(deployed_capture["output"],output,HALF_ATOL,HALF_RTOL)
        checks["autobackend_half"] = {"status":"PASSED" if diff["raw_allclose"] else "FAILED", "comparison":diff,
                                       "deployed":backend.model.get_submodule(TARGET).deployed,
                                       "dtype":str(next(backend.model.parameters()).dtype)}
    return {"status":"FAILED" if any(v["status"] == "FAILED" for v in checks.values()) else (
        "PRECISION_NOTE" if any(v["status"] == "PRECISION_NOTE" for v in checks.values()) else "PASSED"),
        "device":str(image.device),"dtype":"float16","atol":HALF_ATOL,"rtol":HALF_RTOL,"checks":checks,
        "fold_order":"independent eval FP32 copy -> DPR fold preserving norm -> parent-native fusion -> half"}


def _profile_snapshot(model):
    """Capture tensor state and module bookkeeping without changing the model."""
    hook_names = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                  "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_backward_hooks", "_backward_pre_hooks")
    return {
        "state": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "parameters": {key: id(value) for key, value in model.named_parameters()},
        "modules": {
            name: {"identity": id(module), "training": module.training,
                   "buffers": {key: (id(value), None if value is None else value.detach().clone())
                               for key, value in module._buffers.items()},
                   "non_persistent_buffers": set(module._non_persistent_buffers_set),
                   "hooks": {key: dict(getattr(module, key, {})) for key in hook_names}}
            for name, module in model.named_modules()
        },
    }


def _assert_profile_unchanged(model, before, label):
    state = model.state_dict()
    assert set(state) == set(before["state"]), f"{label}: state_dict keys changed"
    for key, value in before["state"].items():
        assert torch.equal(value, state[key]), f"{label}: state_dict tensor changed: {key}"
    assert {key: id(value) for key, value in model.named_parameters()} == before["parameters"], (
        f"{label}: parameter objects changed")
    modules = dict(model.named_modules())
    assert set(modules) == set(before["modules"]), f"{label}: module tree changed"
    for name, original in before["modules"].items():
        module = modules[name]
        assert id(module) == original["identity"] and module.training == original["training"], (
            f"{label}: module identity/training changed: {name}")
        assert set(module._buffers) == set(original["buffers"]), f"{label}: buffers leaked/removed: {name}"
        assert module._non_persistent_buffers_set == original["non_persistent_buffers"], (
            f"{label}: buffer persistence changed: {name}")
        for key, (identity, value) in original["buffers"].items():
            current = module._buffers[key]
            assert id(current) == identity, f"{label}: buffer object changed: {name}.{key}"
            assert (current is None if value is None else torch.equal(current, value)), (
                f"{label}: buffer value changed: {name}.{key}")
        for key, hooks in original["hooks"].items():
            assert dict(getattr(module, key, {})) == hooks, f"{label}: hooks leaked/removed: {name}.{key}"


def _profile_repeated(model, image, custom_ops, label):
    """Run the real THOP twice on one isolated copy and check both objects."""
    original = _profile_snapshot(model)
    isolated = deepcopy(model)
    # Exercise restoration of a pre-existing hook and mixed child training flags.
    # THOP evaluates every module during the measurement itself.
    hook_calls = []
    handle = isolated.register_forward_hook(lambda module, inputs, output: hook_calls.append(1))
    leaf = next(module for module in isolated.modules() if not list(module.children()))
    leaf.training = not isolated.training
    before = _profile_snapshot(isolated)
    input_before = image.detach().clone()
    readings = []
    try:
        for index in range(2):
            operations, parameters = profile_dpr(isolated, inputs=(image,), custom_ops=custom_ops, verbose=False)
            readings.append({"gflops": float(operations) * 2 / 1e9, "thop_parameters": float(parameters)})
            _assert_profile_unchanged(isolated, before, f"{label} isolated profile {index + 1}")
            _assert_profile_unchanged(model, original, f"{label} original profile {index + 1}")
            assert torch.equal(image, input_before), f"{label}: profiling changed the input"
        assert readings[0] == readings[1], f"{label}: repeated profile counts changed: {readings}"
        assert len(hook_calls) == 2, f"{label}: pre-existing hook did not survive both calls"
    finally:
        handle.remove()
    return readings[0]["gflops"], {"status": "PASSED", "readings": readings,
        "original_state_unchanged": True, "isolated_state_unchanged": True,
        "module_training_flags_restored": True, "mixed_training_flags_exercised": True,
        "existing_hook_preserved": True, "no_hook_or_buffer_residue": True, "input_unchanged": True}


def structure(variant, imgsz):
    parent_yaml, target_yaml, expected_training, expected_intermediate, expected_deploy = VARIANTS[variant]
    p_cfg, d_cfg = YAML.load(CFG / parent_yaml), YAML.load(CFG / target_yaml)
    p_graph, d_graph = p_cfg["backbone"] + p_cfg["head"], d_cfg["backbone"] + d_cfg["head"]
    assert len(p_graph) == len(d_graph) == 27
    assert [i for i,(a,b) in enumerate(zip(p_graph,d_graph)) if a != b] == [5]
    assert p_graph[5][:2] == d_graph[5][:2] and p_graph[5][3] == d_graph[5][3]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        parent = RTDETRDetectionModel(str(CFG / parent_yaml), nc=1, verbose=False).eval()
        torch.manual_seed(42)
        candidate = RTDETRDetectionModel(str(CFG / target_yaml), nc=1, verbose=False).eval()
    ps, ds = parent.state_dict(), candidate.state_dict()
    assert set(ds)-set(ps) == {TARGET+"."+n for n in DPRConvNormLayer.parameter_names}
    assert all(torch.equal(v,ds[k]) for k,v in ps.items())
    assert all(torch.count_nonzero(ds[k]) == 0 for k in set(ds)-set(ps))
    assert [(name,type(m).__name__) for name,m in candidate.named_modules() if isinstance(m,DPRConvNormLayer)] == [(TARGET,"DPRConvNormLayer")]
    params = lambda m: sum(p.numel() for p in m.parameters())
    assert params(candidate) == expected_training and params(candidate)-params(parent) == 3072
    if variant == "dpr_v1":
        assert all("CBR" not in type(m).__name__ and "LIF" not in type(m).__name__ for m in candidate.modules())
    else:
        head = candidate.model[-1]
        assert head.num_queries == 300 and head.hidden_dim == 256 and head.decoder.num_layers == 3
        assert head.decoder.eval_idx == 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(914)
        image = torch.rand(1,3,imgsz,imgsz)
    pc, dc = _capture(parent,image), _capture(candidate,image)
    zero_comparison = _compare_capture(pc,dc,candidate,image)
    assert zero_comparison["status"] == "PASSED"
    expected_shapes = {"target": [1,128,imgsz//8,imgsz//8], "P3": [1,128,imgsz//8,imgsz//8],
                       "P4": [1,256,imgsz//16,imgsz//16], "P5": [1,512,imgsz//32,imgsz//32]}
    assert all(list(dc[k].shape) == v for k,v in expected_shapes.items())
    intermediate = _native_fuse_without_dpr(deepcopy(candidate))
    actual_intermediate = params(intermediate)
    assert actual_intermediate == expected_intermediate
    del intermediate
    deployed = deepcopy(candidate).fuse(verbose=False)
    parent_fused = deepcopy(parent).fuse(verbose=False)
    assert params(deployed) == params(parent_fused) == expected_deploy
    assert isinstance(deployed.get_submodule(TARGET).norm,torch.nn.BatchNorm2d)
    thop_semantics = thop_dpr_semantics()
    profile_checks = {}
    def flops(model, custom_ops, label):
        value, profile_checks[label] = _profile_repeated(model, image, custom_ops, label)
        return value
    parent_flops = flops(parent, {}, "parent_unfused")
    raw_dpr_flops = flops(candidate, {}, "raw_candidate")
    corrected_flops = flops(candidate, {DPRConvNormLayer: count_dpr_conv_norm}, "candidate_corrected")
    deploy_flops = flops(deployed, {DPRConvNormLayer: count_dpr_conv_norm}, "standard_deploy")
    parent_fused_flops = flops(parent_fused, {}, "parent_native_fused")
    missed_conv_flops = 2*128*128*3*3*(imgsz//8)**2/1e9
    flops_diagnostic = {
        "variant": variant, "thop": thop_semantics,
        "parent_flops": parent_flops, "raw_dpr_flops": raw_dpr_flops,
        "corrected_flops": corrected_flops, "deploy_flops": deploy_flops,
        "parent_fused_flops": parent_fused_flops, "missed_conv_flops": missed_conv_flops,
        "differences": {"parent_minus_corrected": parent_flops-corrected_flops,
                        "parent_minus_raw_minus_missed": parent_flops-raw_dpr_flops-missed_conv_flops,
                        "parent_fused_minus_deploy": parent_fused_flops-deploy_flops},
        "absolute_tolerance": 1e-9,
    }
    diagnostic_text = "DPR FLOPs diagnostics: " + json.dumps(flops_diagnostic, sort_keys=True)
    print(diagnostic_text, flush=True)
    assert abs(parent_flops-corrected_flops) < 1e-9, diagnostic_text
    assert abs(parent_flops-raw_dpr_flops-missed_conv_flops) < 1e-9, diagnostic_text
    assert abs(parent_fused_flops-deploy_flops) < 1e-9, diagnostic_text
    del parent_fused,deployed,parent,ps,ds,pc,dc
    # Two bounded local synthetic MSE updates make all four groups nonzero. This
    # is a lifecycle fixture, explicitly not actual detector-loss preflight.
    local = candidate.get_submodule(TARGET)
    optimizer = torch.optim.AdamW(local.parameters(),lr=.0005,weight_decay=.0001)
    x,desired = torch.randn(2,128,7,7),torch.randn(2,128,7,7)
    losses,gradients = [],[]
    initial_effective = local.get_equivalent_kernel().detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(local(x),desired)
        loss.backward()
        grad = {n:float(p.grad.norm()) for n,p in local.named_parameters() if n in local.parameter_names or n == "conv.weight"}
        assert all(v > 0 for v in grad.values())
        optimizer.step()
        losses.append(float(loss.detach()))
        gradients.append(grad)
    assert not torch.equal(initial_effective,local.get_equivalent_kernel())
    lifecycle = audit_learned_model(candidate,image)
    assert lifecycle["status"] in ("PASSED","PRECISION_NOTE")
    return {"status": lifecycle["status"], "variant":variant, "nc":1, "device":"cpu", "dtype":"float32",
            "parent_yaml":parent_yaml,"target_yaml":target_yaml,"target":TARGET,
            "target_definition":{"conv":"Conv2d(128,128,3,stride=1,padding=1,bias=False)","norm":"BatchNorm2d(128)","act":"Identity"},
            "public_state_equal":True,"new_state_keys":[TARGET+"."+n for n in local.parameter_names],
            "new_buffers":[],"graph_nodes":27,"changed_top_level_definition":[5],"shapes":expected_shapes,
            "parameters":{"parent_unfused":expected_training-3072,"candidate_unfused":expected_training,
                          "new_trainable":3072,"native_fuse_without_dpr_fold":actual_intermediate,
                          "standard_deploy":expected_deploy,"parent_native_fused":expected_deploy},
            "gflops":{"parent_unfused":parent_flops,"raw_thop_candidate_incorrect":raw_dpr_flops,
                      "candidate_corrected":corrected_flops,"missing_functional_conv":missed_conv_flops,
                      "standard_deploy":deploy_flops,"parent_native_fused":parent_fused_flops,
                      "diagnostics":flops_diagnostic,"repeated_profile":profile_checks,
                      "convention":"THOP MAC*2, conv/BN same as parent; kernel construction separately documented",
                      "kernel_construction":{"difference_parameters":3072,"dense_diagonal_elements":128*128*9,
                                             "dense_kernel_additions":128*128*9,"note":"CD reduction, four maps, three delta sums, diag allocation and dense add each forward; not included in THOP convolution GFLOPs"}},
            "zero_initialization":zero_comparison,
            "nonzero_fixture":{"source":"two local synthetic MSE updates, not detector loss", "losses":losses,"gradients":gradients},
            "learned_lifecycle":lifecycle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=["both",*VARIANTS],default="both")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--math-only",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing check report")
    torch.set_num_threads(args.threads)
    report = {"report_kind":"dpr_mathematical_structural_v1","status":"RUNNING",
              "torch_version":torch.__version__,"source_root":str(ROOT),"source_identity":source_identity(),"checks":{},
              "real_detection_loss":"PENDING_SEPARATE_PREFLIGHT","cuda_lifecycle":"PENDING_SEPARATE_PREFLIGHT",
              "capacity_B16_640_AMP":"PENDING_SEPARATE_PREFLIGHT","formal_training":"NOT_STARTED","final_test":"NOT_RUN"}
    operations = [("cpu_fp64_math",mathematics)]
    if not args.math_only:
        operations.extend((v,lambda variant=v:structure(variant,args.imgsz)) for v in (VARIANTS if args.variant == "both" else [args.variant]))
    start = time.time()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name,operation in operations:
        try:
            report["checks"][name] = operation()
        except Exception:
            report["checks"][name] = {"status":"FAILED","traceback":traceback.format_exc()}
        print(name,report["checks"][name]["status"],flush=True)
        if report["checks"][name]["status"] == "FAILED":
            print(report["checks"][name]["traceback"],flush=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        gc.collect()
    report["elapsed_seconds"] = time.time()-start
    report["status"] = "FAILED" if any(v["status"] == "FAILED" for v in report["checks"].values()) else (
        "PRECISION_NOTE" if any(v["status"] == "PRECISION_NOTE" for v in report["checks"].values()) else "PASSED")
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(str(args.output.resolve()),flush=True)
    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
