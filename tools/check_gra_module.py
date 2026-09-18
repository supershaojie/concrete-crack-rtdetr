"""Independent analytic GRA checks; no dataset, epochs, validation or test split."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))
import torch
from torch.nn import functional as F
from ultralytics.nn.modules import GRAConcat
from ultralytics.utils.torch_utils import ModelEMA


def compare(actual, expected, atol=2e-5, rtol=2e-4):
    actual, expected = actual.detach().float(), expected.detach().float()
    error = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    over = error > atol + rtol * expected.abs()
    return {"status": "PASSED" if finite and not bool(over.any()) else "FAILED",
            "shape": list(actual.shape), "dtype": str(actual.dtype), "device": str(actual.device),
            "max_abs": float(error.max()), "relative_L2": float(torch.linalg.vector_norm(error) /
            torch.linalg.vector_norm(expected).clamp_min(1e-30)), "over_tolerance_fraction": float(over.float().mean()),
            "finite": finite, "atol": atol, "rtol": rtol}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def gradient_stats(value):
    if value is None:
        return {"finite": False, "max_abs": None, "nonzero": False}
    value = value.detach()
    return {"finite": bool(torch.isfinite(value).all()), "max_abs": float(value.abs().max()),
            "nonzero": bool(torch.count_nonzero(value))}


def initialization():
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    module = GRAConcat(256, 256, 256)
    after_cpu = torch.random.get_rng_state()
    after_cuda = torch.cuda.get_rng_state_all() if before_cuda else []
    cpu_equal = torch.equal(before_cpu, after_cpu)
    cuda_equal = all(torch.equal(a, b) for a, b in zip(before_cuda, after_cuda))
    stats = {k: {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max()),
                  "mean": float(v.mean()), "std": float(v.std()), "nonzero": int(torch.count_nonzero(v))}
             for k, v in module.state_dict().items()}
    require(cpu_equal and cuda_equal, "GRA construction altered caller RNG")
    require(sum(p.numel() for p in module.parameters()) == 8744, "GRA parameter budget differs")
    for key, row in stats.items():
        require((row["nonzero"] == 0) == key.startswith("offset."), f"Unexpected initialization: {key}")
    return {"status": "PASSED", "parameters": 8744, "CPU_RNG_unchanged": cpu_equal,
            "CUDA_RNG_unchanged": cuda_equal if before_cuda else "PENDING: CUDA unavailable", "state": stats}


def analytic_ramps(device):
    source_height, source_width = 7, 11
    yy, xx = torch.meshgrid(torch.arange(source_height, device=device, dtype=torch.float32),
                            torch.arange(source_width, device=device, dtype=torch.float32), indexing="ij")
    high = torch.stack([value for group in range(4) for value in (xx, yy, torch.full_like(xx, group + 2.0))])[None]
    offsets = torch.tensor([[.2, -.1], [-.15, .225], [.075, .125], [-.225, -.175]], device=device)
    records = []
    for target_height, target_width in ((14, 22), (13, 19)):
        up = F.interpolate(high, size=(target_height, target_width), mode="nearest")
        lateral = torch.sin(up * .7)
        module = GRAConcat(12, 12, 12).to(device)
        with torch.no_grad():
            zero = module([high, up, lateral])
            zero_check = compare(zero, torch.cat((up, lateral), 1), atol=0, rtol=0)
            require(zero_check["status"] == "PASSED", "Zero offset must exactly recover cat(U,L)")
            module.offset.bias.copy_(torch.atanh(offsets.flatten() / .25))
            output = module([high, up, lateral])
        ys = (torch.arange(target_height, device=device, dtype=torch.float32) + .5) * source_height / target_height - .5
        xs = (torch.arange(target_width, device=device, dtype=torch.float32) + .5) * source_width / target_width - .5
        sy, sx = torch.meshgrid(ys, xs, indexing="ij")
        expected = torch.stack([value for dx, dy in offsets for value in
                                (.5 * ((sx + dx).clamp(0, source_width - 1) - sx.clamp(0, source_width - 1)),
                                 .5 * ((sy + dy).clamp(0, source_height - 1) - sy.clamp(0, source_height - 1)),
                                 torch.zeros_like(sx))])[None]
        correction = output[:, :12] - up
        all_pixels = compare(correction, expected)
        inside = (sx > .25) & (sx < source_width - 1.25) & (sy > .25) & (sy < source_height - 1.25)
        interior = compare(correction[..., inside], expected[..., inside])
        border = compare(correction[..., ~inside], expected[..., ~inside])
        lateral_check = compare(output[:, 12:], lateral, atol=0, rtol=0)
        constant_check = compare(correction[:, [2, 5, 8, 11]], torch.zeros_like(correction[:, [2, 5, 8, 11]]))
        # A target-pixel denominator bug halves the expected standard-2x ramp.
        unit_sensitivity = float((expected[..., inside] - expected[..., inside] * .5).abs().max())
        checks = (all_pixels, interior, border, lateral_check, constant_check)
        require(all(row["status"] == "PASSED" for row in checks), "Analytic source-pixel ramp mismatch")
        mutation_checks = []
        if (target_height, target_width) == (14, 22):
            # Inject coordinate faults only at the shifted sampler boundary. The
            # independent analytic oracle must reject all three real bug classes.
            original_sample = F.grid_sample
            gy, gx = torch.meshgrid((torch.arange(target_height, device=device) + .5) * (2 / target_height) - 1,
                                    (torch.arange(target_width, device=device) + .5) * (2 / target_width) - 1,
                                    indexing="ij")
            base_grid = torch.stack((gx, gy), -1)[None]
            for mutation in ("target_pixel_denominator", "xy_swap", "group_roll"):
                calls = [0]

                def mutated_sample(source, grid, **kwargs):
                    if calls[0] == 0:
                        difference = grid - base_grid
                        if mutation == "target_pixel_denominator":
                            difference = difference * difference.new_tensor((source_width / target_width, source_height / target_height))
                        elif mutation == "xy_swap":
                            pixels = difference * difference.new_tensor((source_width / 2, source_height / 2))
                            difference = pixels[..., [1, 0]] * pixels.new_tensor((2 / source_width, 2 / source_height))
                        else:
                            difference = difference.roll(1, dims=0)
                        grid = base_grid + difference
                    calls[0] += 1
                    return original_sample(source, grid, **kwargs)

                with torch.no_grad(), patch("ultralytics.nn.modules.gra.F.grid_sample", side_effect=mutated_sample):
                    corrupted = module([high, up, lateral])[:, :12] - up
                rejected = compare(corrupted, expected)
                require(rejected["status"] == "FAILED" and rejected["finite"], f"Ramp oracle failed to reject {mutation}")
                mutation_checks.append({"mutation": mutation, "status": "PASSED_FAULT_REJECTED", "oracle_result": rejected})
        records.append({"status": "PASSED", "source_shape": list(high.shape), "target_shape": list(up.shape),
                        "dtype": str(high.dtype), "device": str(high.device), "offsets_xy": offsets.tolist(),
                        "zero_offset": zero_check, "all_pixels": all_pixels, "interior": interior, "border": border,
                        "constant_channels": constant_check, "lateral_unchanged": lateral_check,
                        "wrong_target_units_error_at_least": unit_sensitivity, "mutation_sensitivity": mutation_checks})
    return records


def gradients_and_state(device):
    generator = torch.Generator(device="cpu").manual_seed(73021)
    high = torch.randn(2, 12, 7, 11, generator=generator).to(device).requires_grad_()
    lateral = torch.randn(2, 12, 14, 22, generator=generator).to(device).requires_grad_()
    weights = torch.randn(2, 24, 14, 22, generator=generator).to(device)
    module = GRAConcat(12, 12, 12).to(device)
    optimizer = torch.optim.SGD(module.parameters(), lr=.2)

    def step(backward=True):
        up = F.interpolate(high, size=lateral.shape[-2:], mode="nearest")
        output = module([high, up, lateral])
        loss = (output * weights).mean() + .17 * output.square().mean()
        if backward:
            loss.backward()
        return output, up, loss

    zero_output, up, first_loss = step()
    first = {k: gradient_stats(v.grad) for k, v in module.named_parameters()}
    require(all(v["finite"] for v in first.values()), "Nonfinite first-step gradient")
    require(first["offset.weight"]["nonzero"] and first["offset.bias"]["nonzero"], "Offset head did not start")
    require(all(not v["nonzero"] for k, v in first.items() if not k.startswith("offset.")),
            "Zero last layer must initially block descriptor gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    high.grad = lateral.grad = None
    learned_output, up, second_loss = step()
    second = {k: gradient_stats(v.grad) for k, v in module.named_parameters()}
    second.update(H=gradient_stats(high.grad), L=gradient_stats(lateral.grad))
    require(all(v["finite"] and v["nonzero"] for v in second.values()), "Second-step descriptor/source gradient missing")
    fixed_inputs = [high.detach(), up.detach(), lateral.detach()]
    module.eval()
    with torch.no_grad():
        reference = module(fixed_inputs)
        offsets = .25 * module.offset_logits(fixed_inputs).float().tanh()
    require(0 < float(offsets.abs().max()) < .25, "Learned offsets are zero or out of bounds")
    payload = io.BytesIO()
    torch.save(module.state_dict(), payload)
    payload.seek(0)
    reloaded = GRAConcat(12, 12, 12).to(device).eval()
    reloaded.load_state_dict(torch.load(payload, map_location=device, weights_only=True))
    reload_check = compare(reloaded(fixed_inputs), reference, atol=0, rtol=0)
    require(reload_check["status"] == "PASSED", "Learned state reload changed output")
    ema = ModelEMA(module)
    with torch.no_grad():
        ema.ema.offset.bias.add_(.002)
    ema.update(module)
    ema_copy = deepcopy(ema.ema)
    ema_check = compare(ema_copy(fixed_inputs), ema.ema(fixed_inputs), atol=0, rtol=0)
    require(ema_check["status"] == "PASSED", "EMA self-reference failed")
    quantized = deepcopy(module).half().float()
    quantized_copy = deepcopy(quantized)
    quantization_check = compare(quantized_copy(fixed_inputs), quantized(fixed_inputs), atol=0, rtol=0)
    require(quantization_check["status"] == "PASSED", "Same-quantization half/float round trip failed")
    modes = []
    for mode in (["fp32", "native_amp", "explicit_half"] if torch.device(device).type == "cuda" else ["fp32"]):
        test_module = deepcopy(module)
        inputs = [v.clone().to(dtype=torch.float32 if mode == "fp32" else torch.float16).requires_grad_()
                  for v in fixed_inputs]
        if mode == "explicit_half":
            test_module.half()
        sampled = []
        original_sample = F.grid_sample

        def sample(source, grid, **kwargs):
            sampled.append({"source_dtype": str(source.dtype), "grid_dtype": str(grid.dtype),
                            "source_shape": list(source.shape), "grid_shape": list(grid.shape),
                            "device": str(source.device), **kwargs})
            return original_sample(source, grid, **kwargs)

        with patch("ultralytics.nn.modules.gra.F.grid_sample", side_effect=sample):
            with torch.autocast(device_type=torch.device(device).type, enabled=mode == "native_amp"):
                output = test_module(inputs)
                loss = output.float().square().mean()
            loss.backward()
        finite = bool(torch.isfinite(output).all()) and all(
            bool(torch.isfinite(p.grad).all()) for p in test_module.parameters() if p.grad is not None)
        require(finite and len(sampled) == 2, f"{mode} forward/backward or two-sample contract failed")
        require(all(s["source_dtype"] == s["grid_dtype"] == "torch.float32" for s in sampled), "Sampling left FP32")
        require(gradient_stats(test_module.offset.weight.grad)["nonzero"], "Learned branch lost gradient")
        modes.append({"status": "PASSED", "mode": mode, "input_dtype": str(inputs[0].dtype),
                      "output_dtype": str(output.dtype), "device": str(output.device), "shape": list(output.shape),
                      "finite": finite, "sampling": sampled, "offset_weight_gradient": gradient_stats(test_module.offset.weight.grad),
                      "residual_max_abs": float((output[:, :12].float() - inputs[1].float()).abs().max())})
    return {"status": "PASSED", "first_loss": float(first_loss.detach()), "second_loss": float(second_loss.detach()),
            "first_step_gradients": first, "second_step_gradients": second,
            "zero_residual_max_abs": float((zero_output[:, :12].detach() - up.detach()).abs().max()),
            "learned_offset_max_abs": float(offsets.abs().max()), "learned_offset_mean_abs": float(offsets.abs().mean()),
            "learned_tanh_saturation_fraction": float((offsets.abs() > .25 * .99).float().mean()),
            "learned_residual_max_abs": float((learned_output[:, :12].detach() - up.detach()).abs().max()),
            "reload": reload_check, "EMA_self_copy": ema_check, "same_quantization_half_float": quantization_check,
            "numeric_modes": modes, "CPU_half": "NOT_REQUIRED: not a substitute for CUDA explicit half"}


def run_checks(device="cpu"):
    device = str(torch.device(device))
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "PENDING", "device": device, "reason": "CUDA unavailable"}
    return {"status": "PASSED", "device": device, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "initialization": initialization(), "analytic_ramps": analytic_ramps(device),
            "gradient_and_state": gradients_and_state(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run_checks(args.device)
    except Exception as error:
        report = {"status": "FAILED", "device": args.device, "error": f"{type(error).__name__}: {error}"}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
