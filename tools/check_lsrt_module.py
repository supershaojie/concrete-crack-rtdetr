"""Bounded LSRT mathematical checks; no formal training, evaluation, or optimizer step."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from torch.nn import functional as F

from ultralytics.nn.modules import LocalSemanticResidualTransport, LSRTConcat
from ultralytics.utils.torch_utils import ModelEMA


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=2e-5, rtol=2e-4):
    require(torch.allclose(actual, expected, atol=atol, rtol=rtol),
            f"{message}; max absolute error={(actual - expected).abs().max().item()}")
    return float((actual - expected).abs().max())


def naive_reference(module, low, high):
    """Independent literal per-fine-pixel formula, used only on a 4x6 fixture."""
    with torch.autocast(device_type=low.device.type, enabled=False):
        q = F.normalize(F.conv2d(low.float(), module.q_proj.weight.float()), dim=1, eps=1e-6)
        k = F.normalize(F.conv2d(high.float(), module.k_proj.weight.float()), dim=1, eps=1e-6)
        v = F.conv2d(high.float(), module.v_proj.weight.float())
        h, w = high.shape[-2:]
        rows = []
        for y in range(2 * h):
            columns = []
            for x in range(2 * w):
                cy, cx = y // 2, x // 2
                scores, differences = [], []
                for index in range(9):
                    yy, xx = cy + index // 3 - 1, cx + index % 3 - 1
                    if 0 <= yy < h and 0 <= xx < w:
                        scores.append(8.0 * (q[:, :, y, x] * k[:, :, yy, xx]).sum(1)
                                      + module.rel_bias[index].float())
                        differences.append(v[:, :, yy, xx] - v[:, :, cy, cx])
                attention = torch.stack(scores, dim=-1).softmax(-1)
                columns.append((torch.stack(differences, dim=-1) * attention[:, None]).sum(-1))
            rows.append(torch.stack(columns, dim=-1))
        residual = torch.stack(rows, dim=-2)
        return F.conv2d(residual, module.out_proj.weight.float())


def _run_checks(device):
    if device.type == "cuda" and not torch.cuda.is_available():
        return {"status": "PENDING", "reason": "CUDA unavailable", "device": str(device)}
    report = {"status": "FAILED", "device": str(device), "torch": torch.__version__,
              "formal_optimizer_steps": 0, "diagnostic_optimizer_steps": 0, "checks": {}}
    torch.random.default_generator.manual_seed(123)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(123)
    cpu_rng = torch.get_rng_state().clone()
    module = LocalSemanticResidualTransport().to(device)
    require(torch.equal(cpu_rng, torch.get_rng_state()), "Module construction consumed shared RNG")
    require(sum(p.numel() for p in module.parameters()) == 32777, "Wrong exact parameter delta")
    require(len(list(module.parameters())) == 5 and not list(module.buffers()), "Unexpected parameters/buffers")
    require(torch.count_nonzero(module.out_proj.weight) == 0 and torch.count_nonzero(module.rel_bias) == 0,
            "Output/bias initialization is not zero")
    require(all(torch.count_nonzero(p.weight) > 0 for p in (module.q_proj, module.k_proj, module.v_proj)),
            "Upstream projection unexpectedly zero")
    another = LocalSemanticResidualTransport().to(device)
    require(all(torch.equal(v, another.state_dict()[k]) for k, v in module.state_dict().items()),
            "LSRT variant initial values differ")
    report["checks"]["initialization_rng_parameters"] = {"status": "PASSED", "parameters": 32777}

    coordinate = torch.arange(24, device=device, dtype=torch.float32).reshape(1, 1, 4, 6)
    phases = module.to_phases(coordinate)
    for y in range(4):
        for x in range(6):
            require(phases[0, 0, 2 * (y % 2) + x % 2, (y // 2) * 3 + x // 2] == coordinate[0, 0, y, x],
                    "nearest floor center/phase mapping changed")
    require(torch.equal(module.from_phases(phases, 2, 3), coordinate), "Phase roundtrip changed coordinates")
    values = torch.arange(12, device=device, dtype=torch.float32).reshape(1, 1, 3, 4)
    neighbors, differences, valid = module.value_neighborhoods(values)
    require(torch.equal(neighbors[0, 0, :, 5], values[0, 0, :3, :3].flatten()), "3x3 row-major order changed")
    require(valid[0, :, 0].tolist() == [False, False, False, False, True, True, False, True, True],
            "Top-left mask is wrong")
    require(valid[0, :, -1].tolist() == [True, True, False, True, True, False, False, False, False],
            "Bottom-right mask is wrong")
    require(torch.count_nonzero(differences[:, :, 4]) == 0, "Center difference must be exact zero")
    require(torch.count_nonzero(differences.masked_select(~valid[:, None])) == 0,
            "Invalid differences not explicitly zero")
    one_hot = torch.zeros(1, 4, 9, 12, device=device)
    one_hot[:, :, 4] = 1
    require(torch.count_nonzero(module.aggregate(one_hot, differences)) == 0, "Center one-hot must return zero")
    one_hot.zero_()
    one_hot[:, :, 5] = 1
    require(torch.equal(module.aggregate(one_hot, differences), differences[:, :, 5:6].expand(-1, -1, 4, -1)),
            "Neighbor one-hot did not produce the exact neighbor-minus-center value")
    report["checks"]["geometry_mask_one_hot"] = {"status": "PASSED", "center_index": 4}

    activated = deepcopy(module)
    with torch.no_grad():
        activated.out_proj.weight.normal_(0, 0.05)
    shapes = [(1, 1), (1, 5), (4, 1), (2, 3)]
    for h, w in shapes:
        low = torch.randn(2, 256, 2 * h, 2 * w, device=device)
        high = torch.randn(2, 256, h, w, device=device)
        output, details = activated.forward_with_diagnostics(low, high)
        require(output.shape == low.shape and output.dtype == torch.float32, "Invalid correction shape/dtype")
        weights, mask = details["attention"], details["valid"][:, None]
        close(weights.sum(2), torch.ones_like(weights.sum(2)), "Valid weights must sum to 1", 2e-7, 0)
        require(torch.count_nonzero(weights.masked_select(~mask)) == 0, "Out-of-image attention is nonzero")
        high_constant = torch.randn(2, 256, 1, 1, device=device).expand(-1, -1, h, w).contiguous()
        require(torch.count_nonzero(activated(low, high_constant)) == 0,
                f"Constant nonzero H under nonzero Wo must yield exact zero, including edges: {(h,w)}")
    try:
        activated(torch.zeros(1, 256, 4, 5, device=device), torch.zeros(1, 256, 2, 3, device=device))
    except ValueError as error:
        require("strict nearest x2" in str(error), "Shape failure is not explicit")
    else:
        raise AssertionError("Invalid shape silently accepted")
    for value in (float("nan"), float("inf")):
        bad = torch.zeros(1, 256, 2, 2, device=device)
        bad[0, 0, 0, 0] = value
        try:
            activated(bad, torch.zeros(1, 256, 1, 1, device=device))
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite input silently accepted")
    report["checks"]["shapes_constant_edges_nonfinite"] = {"status": "PASSED", "coarse_shapes": shapes}

    low = torch.randn(1, 256, 4, 6, device=device, requires_grad=True)
    high = torch.randn(1, 256, 2, 3, device=device, requires_grad=True)
    low_before, high_before = low.detach().clone(), high.detach().clone()
    output, details = activated.forward_with_diagnostics(low, high)
    require(not torch.allclose(details["attention"][:, 0], details["attention"][:, 1]),
            "The two fine queries at one coarse center incorrectly share attention")
    shifted_query = activated(low.flip(-1), high)
    require(not torch.allclose(output, shifted_query), "Local low-level matching has no effect")
    constant_low = torch.ones_like(low)
    require(torch.count_nonzero(activated(constant_low, high)) > 0,
            "Constant L does not imply zero correction for nonuniform H")
    coefficient = torch.randn_like(output)
    first_inputs = [low, high, *activated.parameters()]
    gradients = torch.autograd.grad((output * coefficient).sum(), first_inputs)
    reference_model = deepcopy(activated)
    reference_low, reference_high = low.detach().clone().requires_grad_(), high.detach().clone().requires_grad_()
    reference = naive_reference(reference_model, reference_low, reference_high)
    reference_gradients = torch.autograd.grad((reference * coefficient).sum(),
                                             [reference_low, reference_high, *reference_model.parameters()])
    output_error = close(output, reference, "Vectorized result differs from literal formula")
    gradient_errors = {}
    for name, got, wanted in zip(["L", "H", *dict(activated.named_parameters())], gradients, reference_gradients):
        gradient_errors[name] = close(got, wanted, f"Naive gradient mismatch: {name}", 5e-5, 2e-4)
        require(torch.isfinite(got).all() and torch.count_nonzero(got) > 0, f"Activated gradient invalid: {name}")
    require(torch.equal(low, low_before) and torch.equal(high, high_before), "Forward mutated shared inputs")
    report["checks"]["naive_forward_gradients_independent_queries"] = {
        "status": "PASSED", "maximum_output_error": output_error, "maximum_gradient_errors": gradient_errors}

    fresh = deepcopy(module)
    (fresh(low, high) * coefficient).sum().backward()
    first_norms = {}
    for name, parameter in fresh.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"Missing/invalid first grad: {name}")
        first_norms[name] = float(parameter.grad.norm())
        require((torch.count_nonzero(parameter.grad) > 0) == name.startswith("out_proj."),
                f"Zero-output first-step gradient contract violated: {name}")
    with torch.no_grad():
        before = activated(low, high)
        activated.rel_bias.add_(3.0)
        after = activated(low, high)
        close(before, after, "Shared additive relative bias should cancel in softmax")
        activated.rel_bias.sub_(3.0)
    report["checks"]["zero_first_gradient_activated_gradient"] = {
        "status": "PASSED", "first_gradient_norms": first_norms,
        "activation_method": "controlled nonzero out_proj on disposable copy; optimizer steps=0"}
    zero_delta, zero_details = fresh.forward_with_diagnostics(low, high)
    summary = fresh.diagnostic_summary(F.interpolate(high, scale_factor=2, mode="nearest"), zero_delta, zero_details)
    require(len(summary["neighbor_attention_mean"]) == 9 and summary["delta_to_U_norm_ratio"] == 0,
            "Lightweight diagnostic summary is wrong")
    require(abs(sum(summary["neighbor_attention_mean"]) - 1) < 1e-6, "Mean neighborhood attention sum changed")
    json.dumps(summary, allow_nan=False)
    report["checks"]["lightweight_diagnostics"] = {"status": "PASSED", "summary": summary}

    adapter = LSRTConcat([256, 256, 256]).to(device)
    adapter.lsrt.load_state_dict(activated.state_dict(), strict=True)
    route_high = high.detach().clone().requires_grad_()
    route_low = low.detach().clone().requires_grad_()
    upsampled = F.interpolate(route_high, scale_factor=2, mode="nearest")
    upsampled_before = upsampled.detach().clone()
    concat = adapter([upsampled, route_low, route_high])
    require(torch.equal(concat[:, 256:], route_low), "L concat order/value changed")
    close(concat[:, :256], upsampled + activated(route_low, route_high), "U correction/order mismatch")
    require(torch.equal(upsampled, upsampled_before), "U mutated in place")
    g_low, g_high = torch.autograd.grad((concat * torch.randn_like(concat)).sum(), [route_low, route_high])
    require(all(torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in (g_low, g_high)),
            "Original and residual routes must preserve L/H gradients")
    report["checks"]["concat_order_input_immutability_gradient_routes"] = {"status": "PASSED"}

    for label, model in (("zero", module), ("activated", activated)):
        baseline = model(low, high).detach()
        serialized = io.BytesIO()
        torch.save({"model": deepcopy(model), "state_dict": model.state_dict()}, serialized)
        serialized.seek(0)
        restored = torch.load(serialized, map_location=device, weights_only=False)
        state_loaded = LocalSemanticResidualTransport().to(device)
        state_loaded.load_state_dict(restored["state_dict"], strict=True)
        ema = ModelEMA(model)
        ema.update(model)
        for name, candidate in (("checkpoint", restored["model"]), ("state_dict", state_loaded), ("ema", ema.ema)):
            candidate.eval()
            close(candidate(low, high), baseline, f"{label} {name} changed behavior")
            close(candidate.out_proj.weight, model.out_proj.weight, f"{label} {name} reset out_proj", 1e-8, 1e-6)
    half_module = deepcopy(activated).half()
    ids = {name: id(p) for name, p in half_module.named_parameters()}
    half_low, half_high = low.detach().half(), high.detach().half()
    half_result = half_module(half_low, half_high)
    require(half_result.dtype == torch.float32 and torch.isfinite(half_result).all(), "Half model FP32 branch failed")
    half_reference = deepcopy(half_module).float()(half_low.float(), half_high.float())
    close(half_result, half_reference, "Functional half-parameter projections disagree with FP32 reference")
    require(all(p.dtype == torch.float16 and id(p) == ids[name] for name, p in half_module.named_parameters()),
            "Forward replaced or cast registered parameters")
    half_result.square().mean().backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in half_module.parameters()),
            "Half parameter gradient lost through functional float cast")
    report["checks"]["reload_eval_ema_half_parameters"] = {"status": "PASSED", "checkpoint_and_state_dict": ["zero", "activated"]}
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            amp_result = activated(low, high)
        close(amp_result, activated(low, high), "Native AMP altered local FP32 branch", 0, 0)
        report["checks"]["native_amp"] = {"status": "PASSED", "branch_dtype": str(amp_result.dtype)}
    else:
        report["checks"]["native_amp"] = {"status": "PENDING", "reason": "CUDA-only; CPU uses FP32"}
    report["status"] = "PASSED"
    return report


def run_checks(device="cpu"):
    """Return JSONable results and restore the caller's RNG and TF32 state."""
    device = torch.device(device)
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" and torch.cuda.is_available() else [])
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            result = _run_checks(device)
        result["precision_policy"] = {"local_branch": "FP32", "TF32": False, "caller_flags_restored": True}
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved_tf32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"status": "FAILED"}
    try:
        report = run_checks(args.device)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
