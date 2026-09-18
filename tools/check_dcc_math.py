#!/usr/bin/env python3
"""Bounded DCC mathematical, gradient, precision and wrapper lifecycle checks.

Runs no detector training and reads no dataset. Reported module checks do not
replace the separate real-loss/Trainer/resume and B16/640 capacity preflight.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from ultralytics.nn.modules import Conv, DCC, DCCConv
from ultralytics.utils.torch_utils import ModelEMA, fuse_conv_and_bn

ATOL, RTOL = 2e-5, 2e-4  # declared before any tests; never adapted to results


def comparison(a, b, atol=ATOL, rtol=RTOL):
    delta = (a.float() - b.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "relative_L2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)),
        "exceed_fraction": float((delta > atol + rtol * b.float().abs()).float().mean()),
        "allclose": bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def require_close(a, b):
    result = comparison(a, b)
    assert result["allclose"], result
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_device(device):
    torch.manual_seed(42017)
    result = {"device": str(device), "status": "RUNNING"}
    initial = DCC().to(device)
    result["new_trainable_parameters"] = sum(p.numel() for p in initial.parameters())
    assert result["new_trainable_parameters"] == 18432
    assert set(initial.state_dict()) == {name + ".weight" for name in ("W_d", "W_q", "W_k", "W_o")}
    assert not initial.W_o.weight.count_nonzero()
    assert all(getattr(initial, name).weight.count_nonzero() for name in ("W_d", "W_q", "W_k"))
    x = torch.randn(2, 256, 7, 11, device=device)
    assert torch.equal(initial(x), x)

    # Genuine standalone optimization: first W_o starts, then upstream starts.
    trained = copy.deepcopy(initial)
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
    target = torch.randn_like(x)
    result["gradient_steps"] = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (trained(x) - target).square().mean()
        loss.backward()
        grads = {name: float(parameter.grad.norm()) for name, parameter in trained.named_parameters()}
        assert all(torch.isfinite(parameter.grad).all() for parameter in trained.parameters())
        assert grads["W_o.weight"] > 0
        if step == 0:
            assert all(grads[name + ".weight"] == 0 for name in ("W_d", "W_q", "W_k"))
        else:
            assert all(value > 0 for value in grads.values())
        optimizer.step()
        result["gradient_steps"].append({"step": step + 1, "loss": float(loss), "gradient_norms": grads,
                                         "W_o_norm_after_update": float(trained.W_o.weight.norm())})

    # A larger nonzero projection makes mathematical tests nontrivial; lifecycle
    # tests below continue to use the genuinely optimized module, not this copy.
    nonzero = copy.deepcopy(trained)
    torch.nn.init.kaiming_uniform_(nonzero.W_o.weight, a=5 ** 0.5)
    result["shapes"] = []
    with torch.no_grad():
        for shape in ((1, 256, 7, 11), (2, 256, 9, 5), (1, 256, 1, 1)):
            for kind in ("random", "zero", "constant", "low_amplitude"):
                value = torch.randn(shape, device=device)
                if kind == "zero":
                    value.zero_()
                elif kind == "constant":
                    value = torch.randn(shape[:2] + (1, 1), device=device).expand(shape).contiguous()
                elif kind == "low_amplitude":
                    value.mul_(1e-8)
                delta, attention = nonzero._delta_and_attention(value)
                output = nonzero(value)
                assert output.shape == value.shape and torch.isfinite(output).all()
                assert list(attention.shape) == [shape[0], 4, 8, 8]
                mean = float(delta.mean(dim=(-2, -1)).abs().max())
                assert mean < 2e-6, mean
                if kind in ("zero", "constant"):
                    assert float(delta.abs().max()) < 2e-6
                result["shapes"].append({"shape": list(shape), "input": kind, "attention_shape": list(attention.shape),
                                         "branch_mean_abs_max": mean, "residual_abs_max": float(delta.abs().max())})

        products = []
        original_matmul = torch.matmul

        def capture(a, b):
            output = original_matmul(a, b)
            products.append({"left": list(a.shape), "right": list(b.shape), "output": list(output.shape)})
            return output

        with patch("ultralytics.nn.modules.dcc.torch.matmul", side_effect=capture):
            nonzero(x)
        assert [p["output"] for p in products] == [[2, 4, 8, 8], [2, 4, 8, 77]]
        result["actual_matmul_shapes"] = products
        # Captured only products are channel A and channel-to-value product;
        # neither creates a 77 x 77 spatial attention tensor.
        delta, attention = nonzero._delta_and_attention(x)
        _, changed_attention = nonzero._delta_and_attention(torch.randn_like(x))
        result["attention_input_change_max_abs"] = float((attention - changed_attention).abs().max())
        result["nonzero_branch_max_abs"] = float(delta.abs().max())
        assert result["attention_input_change_max_abs"] > 1e-5
        assert result["nonzero_branch_max_abs"] > 1e-5
        permutation = torch.randperm(x.shape[-2] * x.shape[-1], device=device)
        permuted = x.flatten(2)[..., permutation].reshape_as(x)
        result["spatial_permutation_equivariance"] = require_close(
            nonzero(permuted), nonzero(x).flatten(2)[..., permutation].reshape_as(x))

        # Nonzero state_dict and complete module round trips, no reset on load.
        stream = io.BytesIO()
        torch.save(trained.state_dict(), stream)
        stream.seek(0)
        reloaded = DCC().to(device)
        reloaded.load_state_dict(torch.load(stream, map_location=device, weights_only=False))
        assert torch.equal(reloaded.W_o.weight, trained.W_o.weight)
        result["state_dict_reload"] = require_close(reloaded(x), trained(x))
        stream = io.BytesIO()
        torch.save(trained, stream)
        stream.seek(0)
        complete = torch.load(stream, map_location=device, weights_only=False)
        assert torch.equal(complete.W_o.weight, trained.W_o.weight)
        result["complete_module_reload"] = require_close(complete(x), trained(x))

        ema = ModelEMA(trained)
        prior = copy.deepcopy(ema.ema)
        trained.W_o.weight.add_(0.0001)
        ema.update(trained)
        # Use the EMA's own state and independent explicit update as reference.
        decay = ema.decay(ema.updates)
        for key, tensor in prior.state_dict().items():
            if tensor.dtype.is_floating_point:
                tensor.mul_(decay).add_(trained.state_dict()[key] * (1 - decay))
        assert ema.ema.W_o.weight.count_nonzero()
        result["ema_own_weight_reference"] = require_close(ema.ema(x), prior(x))

        wrapper = DCCConv(128, 256, 1, 1, None, 1, 1, False).to(device).eval()
        wrapper.dcc.load_state_dict(trained.state_dict())
        wrapper.bn.running_mean.copy_(torch.randn_like(wrapper.bn.running_mean) * 0.2)
        wrapper.bn.running_var.copy_(torch.rand_like(wrapper.bn.running_var) + 0.5)
        wx = torch.randn(2, 128, 9, 7, device=device)
        normal = wrapper(wx)
        before = {key: value.clone() for key, value in wrapper.dcc.state_dict().items()}
        fused = copy.deepcopy(wrapper)
        fused.conv = fuse_conv_and_bn(fused.conv, fused.bn)
        delattr(fused, "bn")
        fused.forward = fused.forward_fuse
        calls = []
        handle = fused.dcc.register_forward_hook(lambda module, args, out: calls.append(1))
        fused_output = fused(wx)
        handle.remove()
        assert len(calls) == 1
        assert all(torch.equal(before[key], value) for key, value in fused.dcc.state_dict().items())
        result["nonzero_wrapper_fusion"] = require_close(fused_output, normal)
        result["nonzero_wrapper_fusion"]["dcc_calls"] = len(calls)
        assert not torch.equal(fused_output, fused.act(fused.conv(wx)))

        if device.type == "cuda":
            amp_wrapper = copy.deepcopy(wrapper)
            with torch.cuda.amp.autocast():
                amp_output = amp_wrapper(wx)
            assert torch.isfinite(amp_output).all()
            result["native_amp"] = {"finite": True, "output_dtype": str(amp_output.dtype)}
            half_wrapper = copy.deepcopy(wrapper).half()
            half_output = half_wrapper(wx.half())
            assert torch.isfinite(half_output).all()
            half_module = copy.deepcopy(nonzero).half()
            hx = x.half()
            hd, _ = half_module._delta_and_attention(hx)
            hy = half_module(hx)
            assert hy.dtype == torch.float16 and hd.dtype == torch.float32
            result["cuda_half"] = {"finite": bool(torch.isfinite(hy).all()),
                                   "branch_dtype": str(hd.dtype),
                                   "branch_mean_abs_max": float(hd.mean((-2, -1)).abs().max()),
                                   "final_output_difference_mean_abs_max": float((hy.float() - hx.float()).mean((-2, -1)).abs().max())}

    if device.type == "cuda":
        # Older API works with the required PyTorch 2.1.2 runtime too.
        amp_model = copy.deepcopy(wrapper).train()
        amp_optimizer = torch.optim.SGD(amp_model.parameters(), lr=0.01)
        scaler = torch.cuda.amp.GradScaler()
        amp_steps = []
        for _ in range(2):
            amp_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                amp_loss = (amp_model(wx).float() - torch.randn(2, 256, 9, 7, device=device)).square().mean()
            scaler.scale(amp_loss).backward()
            scaler.unscale_(amp_optimizer)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in amp_model.parameters())
            previous_scale = scaler.get_scale()
            scaler.step(amp_optimizer)
            scaler.update()
            assert scaler.get_scale() >= previous_scale
            amp_steps.append({"loss": float(amp_loss), "scale_before": previous_scale, "scale_after": scaler.get_scale()})
        result["native_amp_optimizer_steps"] = amp_steps
    result["status"] = "PASSED"
    return result


def main():
    from dcc_acceptance import CONTRACT_VERSION
    from dcc_common import runtime
    from train_dcc import code_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dcc/module_checks.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    flags = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    files = ["ultralytics-main/ultralytics/nn/modules/dcc.py", "tools/check_dcc_math.py"]
    report = {"status": "RUNNING", "torch": torch.__version__, "python": sys.version,
              "report_kind": "dcc_math_audit", "contract_version": CONTRACT_VERSION,
              "code_identity": code_identity(), "runtime": runtime(),
              "cuda_available": torch.cuda.is_available(), "default_precision_flags": flags,
              "strict_precision_flags": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False},
              "head": subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path: sha(ROOT / path) for path in files},
              "tolerance": {"atol": ATOL, "rtol": RTOL}, "devices": []}
    try:
        state = torch.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        default_initialized = DCC()
        assert torch.equal(state, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))
        report["constructor_preserves_cpu_cuda_rng"] = True
        with torch.random.fork_rng(devices=[]):
            references = [torch.nn.Conv2d(cin, cout, 1, bias=False) for cin, cout in
                          ((256, 32), (32, 32), (32, 32), (32, 256))]
        assert all(torch.equal(getattr(default_initialized, name).weight, reference.weight)
                   for name, reference in zip(("W_d", "W_q", "W_k"), references))
        assert not default_initialized.W_o.weight.count_nonzero()
        report["upstream_matches_torch_conv2d_default_initialization_exactly"] = True
        # Same seed leaves public Conv construction and all later RNG identical.
        torch.manual_seed(42)
        parent = Conv(128, 256, 1, 1, None, 1, 1, False)
        later_parent = torch.rand(5)
        torch.manual_seed(42)
        target = DCCConv(128, 256, 1, 1, None, 1, 1, False)
        later_target = torch.rand(5)
        assert torch.equal(later_parent, later_target)
        assert all(torch.equal(value, target.state_dict()[key]) for key, value in parent.state_dict().items())
        report["original_conv_state_and_later_cpu_rng_preserved"] = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        devices = ["cpu"] if args.device == "auto" else [args.device]
        if args.device == "auto" and torch.cuda.is_available():
            devices.append("cuda")
        elif args.device == "auto":
            report["devices"].append({"device": "cuda", "status": "PENDING", "reason": "CUDA unavailable"})
        for name in devices:
            if name == "cuda" and not torch.cuda.is_available():
                report["devices"].append({"device": "cuda", "status": "PENDING", "reason": "CUDA unavailable"})
            else:
                report["devices"].append(run_device(torch.device(name)))
        report["status"] = "PASSED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
        report["restored_precision_flags"] = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        report["seconds"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(args.output), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
