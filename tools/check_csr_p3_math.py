#!/usr/bin/env python3
"""Independent CSR coordinate and gradient tests; synthetic tests are not a server capacity pass."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from ultralytics.nn.modules import Conv, CSRConv, CurvedSamplingResidual
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights


def assert_close(actual, expected, atol=3e-6, rtol=0):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual - expected).abs().max())


def oracle_sample(feature, x, y):
    """Scalar Python bilinear reference; no production grid/normalization/sample helpers."""
    h, w = feature.shape
    x, y = min(max(x, 0.), w - 1.), min(max(y, 0.), h - 1.)
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = x - x0, y - y0
    return ((1-tx)*(1-ty)*float(feature[y0, x0]) + tx*(1-ty)*float(feature[y0, x1])
            + (1-tx)*ty*float(feature[y1, x0]) + tx*ty*float(feature[y1, x1]))


def coordinate_and_feature_oracle(device):
    csr = CurvedSamplingResidual().to(device)
    # axis/side/step all differ: catches axis swaps, negative sign mistakes and endpoint omission.
    increments = torch.tensor([.11, -.07, .23, -.19, .31, .13,
                               -.17, .29, .09, .21, -.13, .27], device=device)
    h, w = 7, 11
    logits = increments.atanh().view(1, 12, 1, 1).expand(1, 12, h, w).clone().requires_grad_()
    curved, straight = csr.sampling_grids(logits)
    expected = torch.empty_like(curved)
    expected_straight = torch.empty_like(straight)
    cumulative = {}
    values = increments.cpu().tolist()
    for axis in range(2):
        for point, r in enumerate(range(-3, 4)):
            side = 0 if r < 0 else 1
            d = sum(values[axis*6 + side*3 : axis*6 + side*3 + abs(r)]) if r else 0.
            cumulative[axis, point] = d
            for y in range(h):
                for x in range(w):
                    qx, qy = ((x+r, y+d) if axis == 0 else (x+d, y+r))
                    sx, sy = ((x+r, y) if axis == 0 else (x, y+r))
                    expected[0, axis, point, y, x] = torch.tensor([2*(qx+.5)/w-1, 2*(qy+.5)/h-1], device=device)
                    expected_straight[0, axis, point, y, x] = torch.tensor([2*(sx+.5)/w-1, 2*(sy+.5)/h-1], device=device)
    grid_error = assert_close(curved, expected, atol=3e-7)
    assert_close(straight, expected_straight, atol=3e-7)
    assert_close(curved[:, :, 3], straight[:, :, 3], atol=0)
    assert csr.cumulative_offsets(logits).abs().max() < 3
    # One last endpoint must differentiate through all 3 outward increments, including negative side.
    endpoint_grad = torch.autograd.grad(csr.cumulative_offsets(logits)[0, 0, 0, 3, 5], logits)[0]
    for step in range(3):
        assert_close(endpoint_grad[0, step, 3, 5], 1-increments[step].square(), atol=1e-7)
    assert int(torch.count_nonzero(endpoint_grad)) == 3

    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    impulse = torch.zeros(h, w)
    impulse[1, 2] = 1
    features = torch.stack((xx.float(), yy.float(), 2*xx.float()-3*yy.float(), impulse))
    z = features.repeat(8, 1, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        # Distinct point weights catch negative-side point ordering (uniform weights would hide it).
        csr.theta.copy_(torch.linspace(-.8, .9, 7, device=device)[None, None, :].expand_as(csr.theta))
    weights = csr.theta.detach().double().softmax(-1).cpu()
    oracle = torch.zeros_like(z).cpu()
    for c in range(32):
        feature = features[c % 4]
        for y in range(h):
            for x in range(w):
                value = 0.
                for axis in range(2):
                    for point, r in enumerate(range(-3, 4)):
                        d = cumulative[axis, point]
                        qx, qy = ((x+r, y+d) if axis == 0 else (x+d, y+r))
                        sx, sy = ((x+r, y) if axis == 0 else (x, y+r))
                        value += .5*float(weights[axis, c, point])*(oracle_sample(feature, qx, qy)-oracle_sample(feature, sx, sy))
                oracle[0, c, y, x] = value
    difference = csr.sampling_difference(z, logits)
    sample_error = assert_close(difference, oracle.to(device), atol=3e-6)
    assert difference.abs().max() > .01
    center_weight_grad = torch.autograd.grad(difference.square().mean(), csr.theta)[0][:, :, 3]
    assert torch.isfinite(center_weight_grad).all() and center_weight_grad.abs().max() > 1e-8
    return {"shape": [1, 32, h, w], "grid_max_error": grid_error, "sample_max_error": sample_error,
            "center_softmax_logit_gradient_max": float(center_weight_grad.abs().max()),
            "covered": ["horizontal/vertical/oblique ramps", "impulse", "all borders", "four independent chains",
                        "negative-side ordering", "center fixed", "endpoint gradient", "rectangular normalization"]}


def initialization_and_updates(device):
    torch.manual_seed(765)
    parent = Conv(128, 256, 1, act=False)
    following_parent = torch.rand(23)
    torch.manual_seed(765)
    wrapper = CSRConv(128, 256, 1, act=False)
    following_new = torch.rand(23)
    assert_close(following_new, following_parent, atol=0)
    for key, value in parent.state_dict().items():
        assert_close(wrapper.state_dict()[key], value, atol=0)
    csr = wrapper.csr.to(device)
    # Parent initialization must not turn the offset activation into an inplace operation.
    initialize_weights(csr)
    other = CurvedSamplingResidual()
    for key, value in csr.state_dict().items():
        assert_close(value.cpu(), other.state_dict()[key], atol=0)
    assert sum(p.numel() for p in csr.parameters()) == 17548
    assert list(csr.named_buffers()) == []
    assert torch.count_nonzero(csr.offset_pw.weight) == 0 and torch.count_nonzero(csr.offset_pw.bias) == 0
    for layer in (csr.in_proj, csr.offset_dw, csr.out_proj):
        assert torch.count_nonzero(layer.weight) == layer.weight.numel()
    assert_close(csr.theta.softmax(-1), torch.full_like(csr.theta, 1/7), atol=0)
    torch.manual_seed(93)
    x = torch.randn(2, 256, 9, 13, device=device, requires_grad=True)
    upstream = torch.randn_like(x)
    result = csr(x)
    output_error = assert_close(result, x, atol=0)
    dx = torch.autograd.grad((result*upstream).sum(), x)[0]
    gradient_error = assert_close(dx, upstream, atol=3e-6)
    # Explicit cancellation at Z independently checks both sampling paths stay connected.
    z = torch.randn(1, 32, 9, 13, device=device, requires_grad=True)
    logits = torch.zeros(1, 12, 9, 13, device=device, requires_grad=True)
    diff = csr.sampling_difference(z, logits)
    dz, do = torch.autograd.grad((diff*torch.randn_like(diff)).sum(), (z, logits))
    assert_close(dz, torch.zeros_like(dz), atol=3e-6)
    assert torch.isfinite(do).all() and do.abs().max() > 1e-6
    # Isolated synthetic nonlinear unit updates. Detection loss and real data are checked elsewhere.
    optimizer = torch.optim.AdamW(csr.parameters(), lr=.005, weight_decay=0)
    target = torch.randn_like(x)
    updates = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (csr(x.detach())-target).square().mean()
        loss.backward()
        gradients = {}
        for name, parameter in csr.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            gradients[name] = float(parameter.grad.abs().max())
        assert gradients["offset_pw.weight"] > 1e-8
        if step == 0:
            for name in ("in_proj.weight", "offset_dw.weight", "theta", "out_proj.weight"):
                assert gradients[name] < 1e-7, (name, gradients[name])
        if step == 3:
            assert all(value > 1e-10 for value in gradients.values()), gradients
        optimizer.step()
        updates.append({"step": step+1, "loss": float(loss.detach()), "max_abs_gradients": gradients})
    diagnostics = csr.diagnostics(x.detach())
    assert diagnostics["all_finite"] and diagnostics["difference_abs_max"] > 1e-5
    assert diagnostics["residual_l2_relative_to_input"] > 1e-5
    assert all(p.dtype == torch.float32 for p in csr.parameters())
    return csr, {"parameters": 17548, "identity_output_max_error": output_error,
                 "identity_input_gradient_max_error": gradient_error, "synthetic_updates": updates,
                 "diagnostics_after_updates": diagnostics}


def half_and_fused(csr, device):
    # Use learned nonzero state, never the initial identity, to test fusion and half inference.
    module = CSRConv(128, 256, 1, act=False).to(device).eval()
    module.csr.load_state_dict(csr.state_dict())
    x = torch.randn(2, 128, 9, 13, device=device)
    fused = deepcopy(module)
    fused.conv = fuse_conv_and_bn(fused.conv, fused.bn)
    del fused.bn
    fused.forward = fused.forward_fuse
    with torch.no_grad():
        expected = module(x)
        base = module.act(module.bn(module.conv(x)))
        assert (expected-base).abs().max() > 1e-4
        fusion_error = assert_close(fused(x), expected, atol=2e-5, rtol=1e-5)
        half_module = deepcopy(module).half()
        identities = {name: id(p) for name, p in half_module.named_parameters()}
        half_result = half_module(x.half())
        assert half_result.dtype == torch.float16 and torch.isfinite(half_result).all()
        assert identities == {name: id(p) for name, p in half_module.named_parameters()}
        assert all(p.dtype == torch.float16 for p in half_module.parameters())
        half_error = assert_close(half_result.float(), expected, atol=.008, rtol=.01)
        # Forward helper also preserves float32 offset/grid/reduction under explicit half.
        z = half_module.csr.in_proj(base.half())
        logits = half_module.csr.offset_logits(z)
        grids = half_module.csr.sampling_grids(logits)
        difference = half_module.csr.sampling_difference(z, logits)
        assert logits.dtype == grids[0].dtype == grids[1].dtype == difference.dtype == torch.float32
    return {"nonzero_residual_abs_max": float((expected-base).abs().max()),
            "fused_max_error": fusion_error, "half_max_error": half_error,
            "half_parameters_preserved": True, "sampling_dtype": "float32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; does not claim B16/640 capacity")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Existing evidence is preserved; choose a new --output path: {args.output}")
    started = time.perf_counter()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    paths = [Path("tools/check_csr_p3_math.py"),
             Path("ultralytics-main/ultralytics/nn/modules/csr_p3.py"),
             Path("ultralytics-main/ultralytics/nn/modules/conv.py"),
             Path("ultralytics-main/ultralytics/nn/tasks.py")]
    paths += [Path("ultralytics-main/ultralytics/cfg/models/rt-detr") / name for name in
              ("rtdetr-resnet18-lite-cbr-lif-csr-p3-v1.yaml", "rtdetr-resnet18-lite-csr-p3-v1.yaml")]
    git_cmd = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    git = subprocess.run(git_cmd+["rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(git_cmd+["status", "--porcelain"], capture_output=True, text=True)
    provenance = {"commit": git.stdout.strip() if git.returncode == 0 else "UNAVAILABLE",
                  "git_error": git.stderr.strip() if git.returncode else None,
                  "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
                  "file_sha256": {str(path).replace("\\", "/"): hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                                  for path in paths}}
    report = {"status": "RUNNING", "scope": "synthetic mathematical/module checks only",
              "provenance": provenance,
              "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "module_file": str(Path(sys.modules[CurvedSamplingResidual.__module__].__file__).resolve()),
              "tolerances": {"grid_atol": 3e-7, "sampling_atol": 3e-6, "identity_output_atol": 0,
                             "input_gradient_atol": 3e-6, "fusion_atol": 2e-5, "fusion_rtol": 1e-5,
                             "half_atol": .008, "half_rtol": .01}}
    try:
        report["coordinate_and_feature_oracle"] = coordinate_and_feature_oracle(device)
        csr, report["initialization_and_updates"] = initialization_and_updates(device)
        report["half_and_fused"] = half_and_fused(csr, device)
        report["status"] = "PASSED"
    except Exception:
        report["status"] = "FAILED"
        report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter()-started
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n", encoding="utf-8")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
