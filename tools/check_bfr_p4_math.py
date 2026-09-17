#!/usr/bin/env python
"""Independent BFR mathematics, gradient-start and core lifecycle checks.

This bounded diagnostic never starts a detector training run or evaluates test.
Full detection loss, real data, fusion and native Trainer resume are checked by
the separate experiment preflight. CPU half here tests only the BFR core.
"""

import argparse
import copy
import io
import json
import math
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.bfr_p4 import BFRP4, BFRRepC3, count_bfr_p4
from ultralytics.nn.modules.block import RepC3


def reference_basis(n, device):
    """Independent scalar Python double precision DCT-II definition."""
    return torch.tensor(
        [[math.sqrt((1.0 if k == 0 else 2.0) / n) * math.cos(math.pi * (i + 0.5) * k / n)
          for i in range(n)] for k in range(n)], dtype=torch.float64, device=device,
    )


def reference_dct(x):
    h, w = x.shape[-2:]
    return torch.einsum("ui,bcij,vj->bcuv", reference_basis(h, x.device), x.double(), reference_basis(w, x.device))


def reference_idct(x):
    h, w = x.shape[-2:]
    return torch.einsum("ui,bcuv,vj->bcij", reference_basis(h, x.device), x.double(), reference_basis(w, x.device))


def reference_bands(h, w, device):
    values = []
    for k in range(4):
        rows = []
        for u in range(h):
            row = []
            for v in range(w):
                rho = math.sqrt((u / max(h - 1, 1)) ** 2 + (v / max(w - 1, 1)) ** 2) / math.sqrt(2)
                q = [math.exp(-(rho - j / 3) ** 2 / (2 * .2**2)) for j in range(4)]
                row.append(q[k] / sum(q))
            rows.append(row)
        values.append(rows)
    return torch.tensor(values, dtype=torch.float64, device=device)


def close(actual, expected, atol=1e-5, rtol=1e-5):
    expected = expected.to(device=actual.device, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    error = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    return {"max_abs": error, "relative_to_peak": error / max(scale, 1e-12)}


@contextmanager
def strict_fp32():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield {"matmul_allow_tf32": matmul, "cudnn_allow_tf32": cudnn}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def run_checks(device="cpu"):
    start = time.perf_counter()
    device = torch.device(device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    metrics = {}
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), strict_fp32() as original_flags:
        torch.random.default_generator.manual_seed(17)
        if device.type == "cuda":
            torch.cuda.manual_seed(17)
        metrics["strict_diagnostics_original_tf32_flags"] = original_flags
        shapes = [(5, 7), (9, 3), (1, 7), (7, 1), (1, 1), (40, 40)]
        transform = []
        for h, w in shapes:
            ch, cw, bands, non_dc = BFRP4.constants(h, w, device)
            row = {"shape": [h, w]}
            row["basis_h"] = close(ch, reference_basis(h, device), 5e-6, 5e-6)
            row["basis_w"] = close(cw, reference_basis(w, device), 5e-6, 5e-6)
            row["orthogonality_h"] = close(ch @ ch.T, torch.eye(h, device=device), 6e-6, 6e-6)
            row["orthogonality_w"] = close(cw @ cw.T, torch.eye(w, device=device), 6e-6, 6e-6)
            assert bands.dtype == non_dc.dtype == ch.dtype == cw.dtype == torch.float32
            assert bands.min().item() >= 0 and non_dc[0, 0].item() == 0 and non_dc.sum().item() == h * w - 1
            row["bands"] = close(bands, reference_bands(h, w, device), 2e-7, 2e-6)
            # Four rounded divisions followed by a four-term sum can differ
            # from one by several FP32 ulps on CUDA; the independent mask
            # formula is separately checked above, without renormalization.
            row["partition"] = close(bands.sum(0), torch.ones(h, w, device=device), 4 * torch.finfo(torch.float32).eps, 0)
            x = torch.randn(2, 3, h, w, device=device)
            f = BFRP4.dct2(x, ch, cw)
            row["independent_forward"] = close(f, reference_dct(x), 3e-5, 1e-5)
            row["independent_inverse"] = close(BFRP4.idct2(f, ch, cw), reference_idct(f), 3e-5, 1e-5)
            row["roundtrip"] = close(BFRP4.idct2(f, ch, cw), x, 3e-5, 1e-5)
            constant = torch.full_like(x, 1.3)
            expected_dc = torch.zeros_like(x)
            expected_dc[..., 0, 0] = 1.3 * math.sqrt(h * w)
            row["constant_dc"] = close(BFRP4.dct2(constant, ch, cw), expected_dc, 5e-5, 1e-5)
            known_f = torch.zeros_like(x)
            known_f[..., min(2, h - 1), min(1, w - 1)] = 2.5
            known_signal = reference_idct(known_f).float()
            row["single_basis"] = close(BFRP4.dct2(known_signal, ch, cw), known_f, 3e-5, 1e-5)
            transform.append(row)
        metrics["independent_dct_and_bands"] = transform

        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        core = BFRP4().to(device)
        assert torch.equal(cpu_before, torch.random.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())) if cuda_before else True
        other = BFRP4().to(device)
        assert all(torch.equal(v, other.state_dict()[k]) for k, v in core.state_dict().items())
        assert len(list(core.buffers())) == 0
        assert sum(p.numel() for p in core.parameters()) == 20688
        assert core.wo.weight.count_nonzero().item() == 0
        for name in ("wd", "fc1", "fc2"):
            assert getattr(core, name).weight.count_nonzero().item() > 0
        metrics["initialization"] = {"cpu_rng_unchanged": True, "cuda_rng_unchanged_if_initialized": True,
                                     "new_state_identical": True, "parameters": 20688, "persistent_buffers": 0}
        saved_rng = torch.random.get_rng_state()
        parent = RepC3(512, 256, 3, .5)
        after_parent = torch.random.get_rng_state()
        torch.random.set_rng_state(saved_rng)
        wrapper = BFRRepC3(512, 256, 3, .5)
        assert torch.equal(after_parent, torch.random.get_rng_state())
        assert len(wrapper.m) == 3
        assert all(torch.equal(v, wrapper.state_dict()[k]) for k, v in parent.state_dict().items())
        assert set(wrapper.state_dict()) - set(parent.state_dict()) == {"bfr." + k for k in core.state_dict()}
        metrics["wrapper_parent_state_paths_and_following_rng"] = "PASSED"

        x = torch.randn(2, 256, 7, 9, device=device)
        assert torch.equal(core(x), x)
        diag = core.diagnostics(x)
        assert all(v.dtype == torch.float32 for v in diag.values())
        # Independently recompute every channel's normalized four-band energy,
        # including the epsilon and channel-major flatten ordering.
        f_ref = reference_dct(diag["z"])
        masks_ref = reference_bands(7, 9, device)
        e_ref = torch.empty(2, 32, 4, dtype=torch.float64, device=device)
        for b in range(2):
            for c in range(32):
                total = sum(f_ref[b, c, u, v].square() for u in range(7) for v in range(9) if u or v)
                for k in range(4):
                    numerator = sum(f_ref[b, c, u, v].square() * masks_ref[k, u, v]
                                    for u in range(7) for v in range(9) if u or v)
                    e_ref[b, c, k] = numerator / (total + 1e-6)
        metrics["independent_per_sample_per_channel_energy"] = close(diag["e"], e_ref, 4e-6, 4e-6)
        flattened = torch.stack([torch.stack([e_ref[b, c, k] for c in range(32) for k in range(4)]) for b in range(2)])
        hidden_ref = flattened @ core.fc1.weight.double().T + core.fc1.bias.double()
        hidden_ref = hidden_ref * torch.sigmoid(hidden_ref)
        a_ref = .5 * torch.tanh(hidden_ref @ core.fc2.weight.double().T + core.fc2.bias.double())
        metrics["channel_major_controller"] = close(diag["a"], a_ref.reshape(2, 32, 4), 2e-6, 2e-5)

        # A simple hand-checkable two-frequency signal, with different channel
        # energies, makes the denominator and absence of band-area normalization explicit.
        simple_f = torch.zeros(2, 32, 5, 7, device=device)
        simple_f[0, 0, 0, 0], simple_f[0, 0, 1, 2], simple_f[0, 0, 3, 4] = 99, 2, 3
        _, _, m, non_dc = core.constants(5, 7, device)
        p = (simple_f * non_dc).square()
        simple_e = (p.unsqueeze(2) * m[None, None]).sum((-2, -1)) / (p.sum((-2, -1))[..., None] + 1e-6)
        expected = (4 * m[:, 1, 2] + 9 * m[:, 3, 4]) / (13 + 1e-6)
        metrics["hand_energy_two_frequency_eps"] = close(simple_e[0, 0], expected, 1e-7, 1e-6)
        assert simple_e[1].count_nonzero().item() == 0

        # Diagnostic-only fixed coefficients: never injected into production forward.
        ch, cw, m, non_dc = core.constants(5, 7, device)
        coeff = torch.tensor([.2, -.1, .35, -.4], device=device)
        gain = (coeff[:, None, None] * m).sum(0) * non_dc
        single_f = torch.zeros(1, 32, 5, 7, device=device)
        single_f[..., 2, 3] = torch.arange(1, 33, device=device).float()[None] / 32
        z = reference_idct(single_f).float()
        r = core.idct2(gain * single_f, ch, cw)
        metrics["fixed_coeff_residual_spectrum"] = close(reference_dct(r).float(), gain * single_f, 3e-6, 1e-5)
        metrics["fixed_coeff_corrected_spectrum"] = close(reference_dct(z + r).float(), (1 + gain) * single_f, 3e-6, 1e-5)
        learned = copy.deepcopy(core)
        torch.nn.init.xavier_uniform_(learned.wo.weight)
        projected = F.conv2d(r, learned.wo.weight)
        projected_ref = torch.einsum("oc,bcij->boij", learned.wo.weight[..., 0, 0].double(), reference_idct(gain * single_f))
        metrics["fixed_coeff_nonzero_output_projection"] = close(projected, projected_ref, 2e-6, 1e-5)

        diag = learned.diagnostics(x)
        mean_error = diag["r"].mean((-2, -1)).abs().max().item()
        norm_ratio = (diag["r"].flatten(2).norm(dim=-1) / diag["z"].flatten(2).norm(dim=-1).clamp_min(1e-12)).max().item()
        assert mean_error < 2e-6 and norm_ratio <= .50001
        assert diag["a"].abs().max().item() <= .5 and diag["delta_g"].abs().max().item() <= .5
        assert diag["delta_g"][..., 0, 0].count_nonzero().item() == 0
        constant = learned.diagnostics(torch.randn(2, 256, 1, 1, device=device).expand(-1, -1, 7, 9))
        constant_r = constant["r"].abs().max().item()
        assert constant_r < 3e-6
        metrics["latent_bounds_and_dc"] = {"residual_max_abs_spatial_mean": mean_error, "max_latent_norm_ratio": norm_ratio,
                                           "constant_input_max_abs_r": constant_r,
                                           "projected_spatial_mean": diag["delta"].mean((-2, -1)).abs().max().item(),
                                           "bound_applies_to": "R, not W_o(R)"}
        # Two spatial frequencies with the same sample/channel scaling.
        signals_f = torch.zeros(2, 256, 7, 9, device=device)
        scales = torch.randn(256, device=device)
        signals_f[0, :, 1, 1], signals_f[1, :, 5, 7] = scales, scales
        signals = reference_idct(signals_f).float()
        adaptive = learned.diagnostics(signals)
        e_change = (adaptive["e"][0] - adaptive["e"][1]).abs().max().item()
        a_change = (adaptive["a"][0] - adaptive["a"][1]).abs().max().item()
        ch, cw, m, non_dc = learned.constants(7, 9, device)
        changed_gain = ((adaptive["a"][0] - adaptive["a"][1])[..., None, None] * m[None]).sum(1) * non_dc
        influence = F.conv2d(learned.idct2(changed_gain * adaptive["f"][:1], ch, cw), learned.wo.weight)
        assert e_change > 1e-3 and a_change > 1e-5 and influence.abs().max().item() > 1e-6
        metrics["image_adaptive_output"] = {"max_e_change": e_change, "max_a_change": a_change,
                                            "output_change_at_fixed_sample0_spectrum": influence.abs().max().item()}

        optimizer = torch.optim.SGD(core.parameters(), lr=.2)
        ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(core.parameters()))
        target = torch.randn_like(x)
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (core(x) - target).square().mean()
            loss.backward()
            gradients = {name: p.grad.norm().item() for name, p in core.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters())
            assert gradients["wo.weight"] > 0
            if step == 0:
                assert all(value == 0 for name, value in gradients.items() if name != "wo.weight")
            else:
                for name in ("wd.weight", "gn.weight", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
                    assert gradients[name] > 0, name
            before = core.wo.weight.detach().clone()
            optimizer.step()
            changed = (core.wo.weight - before).abs().max().item()
            assert changed > 0 and core.wo.weight.count_nonzero().item() > 0
            steps.append({"step": step + 1, "loss": loss.item(), "grad_norms": gradients, "wo_max_update": changed,
                          "actual_optimizer_update": True})
        metrics["two_effective_optimizer_updates"] = steps
        metrics["gradient_exception"] = "gn.bias: a per-channel spatial constant removed by DC exclusion; roundoff gradients are not learning evidence"

        # Nonzero learned state only. Same-quantization save/reload and dtype moves.
        half = copy.deepcopy(core).half()
        ids_before = [id(p) for p in half.parameters()]
        half_x = x.half()
        half_out = half(half_x)
        assert half_out.dtype == torch.float16 and torch.isfinite(half_out).all()
        assert [id(p) for p in half.parameters()] == ids_before
        assert all(p.dtype == torch.float16 for p in half.parameters())
        half_diag = half.diagnostics(half_x)
        assert all(v.dtype == torch.float32 for v in half_diag.values())
        quantized_reference = copy.deepcopy(half).float()
        expected_half = half_x + quantized_reference.diagnostics(half_x)["delta"].half()
        metrics["explicit_half_same_quantization"] = close(half_out, expected_half, 0, 0)
        payload = io.BytesIO()
        torch.save(half, payload)
        payload.seek(0)
        restored = torch.load(payload, map_location=device, weights_only=False)
        metrics["half_checkpoint_nonzero_branch_reload"] = close(restored(half_x), half_out, 0, 0)
        assert restored.wo.weight.count_nonzero().item() > 0 and not list(restored.buffers())
        half.zero_grad(set_to_none=True)
        half(half_x).float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in half.parameters())
        metrics["explicit_half_functional_autograd"] = {"parameters_preserved": True, "all_gradients_finite": True,
                                                        "core_only_cpu_half": device.type == "cpu"}
        for h, w in ((1, 1), (3, 11), (9, 1), (7, 9)):
            output = restored.float()(torch.randn(2, 256, h, w, device=device))
            assert torch.isfinite(output).all()
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                amp_output = core(x)
                amp_diag = core.diagnostics(x)
            assert torch.isfinite(amp_output).all() and all(v.dtype == torch.float32 for v in amp_diag.values())
            metrics["cuda_autocast_local_fp32"] = "PASSED"
        metrics["no_cache_shape_dtype_deepcopy_roundtrips"] = "PASSED (constants rebuilt per call)"
        dummy = BFRP4()
        dummy.total_ops = 0
        count_bfr_p4(dummy, (torch.empty(1, 256, 40, 40),), None)
        assert dummy.total_ops == 34410496
        dummy.total_ops = torch.zeros(1, dtype=torch.float64)
        count_bfr_p4(dummy, (torch.empty(1, 256, 40, 40),), None)
        assert dummy.total_ops.item() == 34410496
        metrics["major_macs"] = {"at_640_p4_40x40_b1": 34410496, "gflops_at_two_per_mac": .068820992,
                                  "int_and_tensor_thop_counters": "PASSED", "coverage": "PARTIAL: GN, elementwise, basis generation excluded",
                                  "cache_policy": "no cache; every forward rebuilds constants"}
    assert torch.backends.cuda.matmul.allow_tf32 == original_flags["matmul_allow_tf32"]
    assert torch.backends.cudnn.allow_tf32 == original_flags["cudnn_allow_tf32"]
    return {"status": "PASSED", "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
            "scope": "core mathematical/lifecycle diagnostics; no dataset and no detector training",
            "module_source": str(Path(sys.modules[BFRP4.__module__].__file__).resolve()),
            "elapsed_seconds": time.perf_counter() - start, "checks": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; no automatic GPU use")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bfr_p4/math_cpu.json")
    args = parser.parse_args()
    result = run_checks(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "device": result["device"], "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
