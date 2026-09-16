#!/usr/bin/env python3
"""Independent BSC v1 math/precision checks and native-THOP model comparisons.

All models are diagnostic copies. This entry point never accesses a formal run,
trains the detector, or writes an initialization/checkpoint used by training.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import BilateralContextSupport, BSCRepC3, Conv, LIFDown, RepC3, RepConv
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import ModelEMA

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
MODEL_DIR = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
PAIRS = {
    "bscrep_v1": ("rtdetr-resnet18-lite.yaml", "rtdetr-resnet18-lite-bscrep-v1.yaml"),
    "cbr_lif_bscrep_v1": ("rtdetr-resnet18-lite-cbr-lif-down.yaml", "rtdetr-resnet18-lite-cbr-lif-bscrep-v1.yaml"),
}
UNCOVERED_OPERATORS = [
    "BSC replicate padding, slicing/views, additions, division/means, abs, cat, sigmoid, multiplies, dtype casts",
    "CBR functional grid_sample, sample geometry, softmax, evidence construction, and residual arithmetic",
    "LIF functional padding, Haar/align arithmetic, GELU, stacking/concatenation",
    "Decoder/AIFI functional attention/matmul, deformable grid_sample, topk/query selection and other functional ops",
    "All native THOP results are partial registered-module operation counts; no speed claim follows from GFLOPs",
]


@contextmanager
def strict_fp32():
    """Use a stated strict FP32 policy, restoring the caller's TF32 configuration."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def reference_context(z, dy, dx, sign):
    """Independent clamped-coordinate gather reference, with no replicate slicing."""
    h, w = z.shape[-2:]
    yy, xx = torch.arange(h, device=z.device), torch.arange(w, device=z.device)
    samples = [z[:, :, (yy + sign * k * dy).clamp(0, h - 1)[:, None],
                 (xx + sign * k * dx).clamp(0, w - 1)[None, :]] for k in (1, 2, 3)]
    return torch.stack(samples).mean(0)


def reference_delta(branch, t, directions=DIRECTIONS):
    """Literal reference formula; only used on small tensors."""
    z = F.conv2d(t.float(), branch.in_proj.weight.float())
    responses = []
    for dy, dx in directions:
        plus, minus = reference_context(z, dy, dx, 1), reference_context(z, dy, dx, -1)
        mean, difference = (plus + minus) / 2, (plus - minus).abs()
        gate = F.conv2d(torch.cat([z, mean, difference], 1), branch.gate.weight.float(),
                        branch.gate.bias.float()).sigmoid()
        responses.append(gate * (mean - z))
    return F.conv2d(torch.stack(responses).mean(0), branch.out_proj.weight.float()).to(t.dtype)


def assert_state_equal(left, right):
    assert left.keys() == right.keys(), (left.keys(), right.keys())
    for name in left:
        assert torch.equal(left[name], right[name]), name


def nonzero_output(branch):
    with torch.no_grad():
        branch.out_proj.weight.normal_(0, 0.01)
    return branch


def gradients(module):
    return {name: {"finite": bool(p.grad is not None and torch.isfinite(p.grad).all()),
                   "nonzero": bool(p.grad is not None and torch.count_nonzero(p.grad)),
                   "max_abs": None if p.grad is None else float(p.grad.abs().max())}
            for name, p in module.named_parameters()}


def run_math_checks(device="cpu"):
    """Raise on a failed assertion, otherwise return reproducible audit evidence."""
    checks = {}
    torch.manual_seed(42)
    branch = BilateralContextSupport(128)
    expected_keys = {"in_proj.weight", "gate.weight", "gate.bias", "out_proj.weight"}
    assert set(branch.state_dict()) == expected_keys
    assert sum(p.numel() for p in branch.parameters()) == 11296
    assert torch.count_nonzero(branch.in_proj.weight) and torch.count_nonzero(branch.gate.weight)
    assert not torch.count_nonzero(branch.gate.bias) and not torch.count_nonzero(branch.out_proj.weight)
    checks["parameters_and_initialization"] = {"status": "PASS", "new_parameters": 11296,
                                                "new_keys": sorted(expected_keys)}

    torch.manual_seed(123)
    parent = RepC3(512, 256, 3, 0.5).eval()
    rng_after_parent = torch.get_rng_state().clone()
    torch.manual_seed(123)
    candidate = BSCRepC3(512, 256, 3, 0.5).eval()
    assert torch.equal(torch.get_rng_state(), rng_after_parent)
    assert_state_equal(parent.state_dict(), {k: v for k, v in candidate.state_dict().items() if not k.startswith("bsc.")})
    assert len(candidate.m) == 3 and all(isinstance(m, RepConv) for m in candidate.m)
    x = torch.randn(2, 512, 5, 7)
    original_x = x.clone()
    with torch.no_grad():
        assert torch.equal(parent(x), candidate(x))
    assert torch.equal(x, original_x)
    checks["parent_rng_state_paths_and_zero_output"] = {"status": "PASS", "bitwise_output_equal": True}

    # A coordinate image distinguishes both signs and both diagonals even at corners.
    z = (100 * torch.arange(5)[:, None] + torch.arange(7)[None, :]).float()[None, None]
    padded = F.pad(z, (3, 3, 3, 3), mode="replicate")
    coordinate_examples = []
    for dy, dx in DIRECTIONS:
        plus, minus = branch._contexts(padded, 5, 7, dy, dx)
        torch.testing.assert_close(plus, reference_context(z, dy, dx, 1), rtol=0, atol=0)
        torch.testing.assert_close(minus, reference_context(z, dy, dx, -1), rtol=0, atol=0)
        coordinate_examples.append({"dy_dx": [dy, dx], "plus_top_left": float(plus[0, 0, 0, 0]),
                                    "minus_top_left": float(minus[0, 0, 0, 0])})
    checks["coordinate_directions_radius_replicate_edges"] = {"status": "PASS", "examples": coordinate_examples}

    branch = nonzero_output(branch).to(device)
    errors = {}
    for h, w in ((1, 1), (1, 2), (2, 1), (2, 3), (5, 9), (9, 5)):
        t = torch.randn(2, 128, h, w, device=device)
        actual, expected = branch(t), reference_delta(branch, t)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        errors[f"{h}x{w}"] = float((actual - expected).abs().max())
    constant = torch.randn(2, 128, 1, 1, device=device).expand(-1, -1, 5, 9)
    constant_error = float(branch(constant).abs().max())
    assert constant_error < 1e-6, constant_error
    t = torch.randn(2, 128, 5, 9, device=device)
    torch.testing.assert_close(reference_delta(branch, t),
                               reference_delta(branch, t, tuple((-dy, -dx) for dy, dx in DIRECTIONS)),
                               rtol=0, atol=0)
    checks["reference_rectangles_small_maps_constant_and_swap"] = {
        "status": "PASS", "max_abs_errors": errors, "constant_max_abs_including_boundary": constant_error,
        "endpoint_swap_bitwise_equal": True,
    }

    initial = BilateralContextSupport(128).to(device)
    t = torch.randn(2, 128, 5, 9, device=device, requires_grad=True)
    probe = torch.randn_like(t)
    (initial(t) * probe).mean().backward()
    first = gradients(initial)
    assert first["out_proj.weight"]["finite"] and first["out_proj.weight"]["nonzero"]
    assert all(item["finite"] and not item["nonzero"] for key, item in first.items() if key != "out_proj.weight")
    branch.zero_grad(set_to_none=True)
    (branch(t) * probe).mean().backward()
    learned = gradients(branch)
    assert all(item["finite"] and item["nonzero"] for item in learned.values()), learned
    checks["initial_and_nonzero_output_gradients"] = {"status": "PASS", "first_step": first, "diagnostic_nonzero": learned}

    # Check the actual half-parameter fallback, including gradient casts to leaves.
    half = copy.deepcopy(branch).half()
    ids = {name: id(p) for name, p in half.named_parameters()}
    half.zero_grad(set_to_none=True)
    ht = torch.randn(2, 128, 5, 9, device=device, dtype=torch.float16, requires_grad=True)
    hy = half(ht)
    assert hy.dtype == ht.dtype
    hy.float().square().mean().backward()
    half_grads = gradients(half)
    assert all(item["finite"] and item["nonzero"] for item in half_grads.values()), half_grads
    assert all(p.dtype == torch.float16 and id(p) == ids[name] for name, p in half.named_parameters())
    assert ht.grad is not None and torch.isfinite(ht.grad).all()
    torch.testing.assert_close(hy, reference_delta(half, ht), rtol=0.002, atol=0.0001)
    checks["checkpoint_half_differentiable_fp32_fallback"] = {"status": "PASS", "gradients": half_grads,
                                                             "parameter_identity_preserved": True}

    persistence = {}
    for label, source in (("zero", initial), ("nonzero", branch)):
        stream = io.BytesIO()
        torch.save({"model": copy.deepcopy(source), "optimizer_updates": 0}, stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location=device, weights_only=False)["model"]
        assert_state_equal(source.state_dict(), loaded.state_dict())
        ema = ModelEMA(source)
        ema.update(source)
        # EMA averaging can introduce a last-bit roundoff; check against the
        # actual EMA recurrence, and require preservation of nonzero learned state.
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(ema.ema.state_dict()[name], tensor, rtol=1e-6, atol=1e-8)
        assert bool(torch.count_nonzero(ema.ema.out_proj.weight)) == (label == "nonzero")
        persistence[label] = {"serialized_reload": "PASS", "ema_update": "PASS"}
    checks["branch_serialized_reload_and_ema"] = {"status": "PASS", **persistence}

    from thop import profile
    counted = copy.deepcopy(branch).float()
    calls = []
    hook = counted.gate.register_forward_hook(lambda _m, _a, _y: calls.append(1))
    macs, params = profile(counted, inputs=(torch.randn(1, 128, 80, 80, device=device),), verbose=False)
    hook.remove()
    assert len(calls) == 4 and macs == 131072000 and params == 11296, (len(calls), macs, params)
    assert not any("total_ops" in key or "total_params" in key for key in branch.state_dict())
    checks["native_thop_on_float_copy"] = {"status": "PASS", "gate_calls": len(calls), "macs": int(macs),
                                           "params": int(params), "conv_gflops_2_per_mac": 2 * macs / 1e9,
                                           "uncovered": UNCOVERED_OPERATORS[0]}
    return {"status": "PASS", "device": str(device), "checks": checks}


def check_cuda_native_amp():
    """Use real dynamic GradScaler; failures and backoff are recorded, not masked."""
    if not torch.cuda.is_available():
        return {"status": "PENDING", "reason": "CUDA unavailable; run this entry point on the server"}
    branch = nonzero_output(BilateralContextSupport(128)).cuda()
    optimizer = torch.optim.AdamW(branch.parameters(), lr=0.0005, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
    trace = []
    x, target = torch.randn(2, 128, 9, 7, device="cuda"), torch.randn(2, 128, 9, 7, device="cuda")
    hook_dtypes = []
    hook = branch.gate.register_forward_hook(lambda _m, _a, y: hook_dtypes.append(str(y.dtype)))
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        scale = scaler.get_scale()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Mimic the half activation from the enclosing RepC3 native AMP path.
            output = branch(x.half())
            loss = (output.float() - target).square().mean()
        assert output.dtype == torch.float16 and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_info = gradients(branch)
        finite = all(item["finite"] for item in grad_info.values())
        scaler.step(optimizer)
        scaler.update()
        trace.append({"scale_before": scale, "scale_after": scaler.get_scale(), "unscaled_finite": finite,
                      "loss": float(loss), "gradients": grad_info})
        if finite:
            break
    hook.remove()
    assert trace[-1]["unscaled_finite"], trace
    assert set(hook_dtypes) == {"torch.float32"}, hook_dtypes
    return {"status": "PASS", "trace": trace, "internal_gate_dtype": "torch.float32",
            "scope": "isolated BSC branch; detector original-loss and real-data capacity are separate preflight checks"}


def profile_models(device="cpu", imgsz=640):
    """Count the same nc=1/640 pairs with native leaf hooks on isolated FP32 copies."""
    from thop import profile
    report = {"status": "PASS", "device": device, "nc": 1, "batch": 1, "imgsz": imgsz,
              "policy": "FP32 copies; native THOP leaf hooks only; 2 FLOPs/MAC; TF32 disabled then restored",
              "uncovered_operators": UNCOVERED_OPERATORS, "variants": {}}
    with strict_fp32(), torch.no_grad():
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        for variant, pair in PAIRS.items():
            counts = {}
            for label, yaml in zip(("parent", "bsc"), pair):
                torch.manual_seed(42)
                model = RTDETRDetectionModel(str(MODEL_DIR / yaml), nc=1, verbose=False).eval().to(device).float()
                if label == "bsc":
                    assert isinstance(model.model[19], BSCRepC3) and len(model.model[19].m) == 3
                    assert sum(isinstance(m, BilateralContextSupport) for m in model.modules()) == 1
                    assert model.model[19].cv1.conv.in_channels == 512
                    assert model.model[19].cv1.conv.out_channels == 128
                    assert model.model[26].f == [19, 22, 25]
                    if variant == "bscrep_v1":
                        assert type(model.model[20]) is Conv
                        assert type(model.model[26]).__name__ == "RTDETRDecoder"
                    else:
                        assert isinstance(model.model[20], LIFDown)
                        assert type(model.model[26]).__name__ == "RTDETRDecoderCBR"
                counts[label] = {}
                for fused in (False, True):
                    diagnostic = copy.deepcopy(model)
                    if fused:
                        diagnostic.fuse(verbose=False)
                    params = sum(p.numel() for p in diagnostic.parameters())
                    calls = []
                    hook = diagnostic.model[19].bsc.gate.register_forward_hook(lambda _m, _a, _y: calls.append(1)) if label == "bsc" else None
                    macs, thop_params = profile(diagnostic, inputs=(x,), verbose=False)
                    if hook is not None:
                        hook.remove()
                        assert len(calls) == 4
                    counts[label]["fused" if fused else "unfused"] = {
                        "parameters": params, "thop_parameters": int(thop_params), "macs": int(macs),
                        "gflops_2_per_mac": 2 * macs / 1e9, "bsc_gate_calls": len(calls),
                    }
                    del diagnostic
                del model
            for mode in ("unfused", "fused"):
                new, old = counts["bsc"][mode], counts["parent"][mode]
                assert new["parameters"] - old["parameters"] == 11296
                assert new["macs"] - old["macs"] == 20480 * (imgsz // 8) ** 2
            report["variants"][variant] = counts
    return report


def run_checks():
    """Shared preflight API; unavailable CUDA is PENDING, never a passing skip."""
    with strict_fp32():
        return {
            "status": "PASS" if torch.cuda.is_available() else "PENDING",
            "cpu_fp32": run_math_checks("cpu"),
            "cuda_fp32": run_math_checks("cuda") if torch.cuda.is_available() else {"status": "PENDING", "reason": "CUDA unavailable"},
            "cuda_native_amp": check_cuda_native_amp(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bscrep_v1/math_checks.json")
    parser.add_argument("--profile-models", action="store_true", help="Also compare both full detector pairs at nc=1, B1/640")
    parser.add_argument("--profile-device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = {"torch_version": torch.__version__, "formal_optimizer_updates": 0, "status": "RUNNING"}
    try:
        report.update(run_checks())
        if args.profile_models:
            report["model_profiles"] = profile_models(args.profile_device)
        report["status"] = "PASS" if torch.cuda.is_available() else "PENDING"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
