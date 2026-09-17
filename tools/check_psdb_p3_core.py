#!/usr/bin/env python3
"""Small CPU checks for PSDB mathematics, parser wiring, RNG and parameter accounting.

These synthetic core diagnostics complement (do not replace) the real detection
loss and server B16/640/native AMP checks in check_psdb_p3.py.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import RepC3, PSDBP3, PSDBRepC3
from ultralytics.nn.tasks import RTDETRDetectionModel, yaml_model_load

CFG = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"


def phase_checks():
    cases = []
    # Multiple channels expose pixel_unshuffle's different channel order.
    for h, w in ((4, 6), (3, 5), (3, 6), (4, 5), (1, 1)):
        p2 = torch.arange(2 * h * w).reshape(1, 2, h, w).float()
        expected = torch.empty(1, 8, (h + 1) // 2, (w + 1) // 2)
        for phase, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            for channel in range(2):
                for row in range((h + 1) // 2):
                    for col in range((w + 1) // 2):
                        expected[0, 2 * phase + channel, row, col] = p2[
                            0, channel, min(2 * row + dy, h - 1), min(2 * col + dx, w - 1)
                        ]
        actual = PSDBP3.phase_rearrange(p2)
        assert torch.equal(actual, expected), (h, w, actual, expected)
        cases.append({"p2_hw": [h, w], "exact_phase_and_padding": True})
    return cases


def core_checks():
    torch.manual_seed(309)
    rng = torch.get_rng_state().clone()
    block = PSDBP3()
    assert torch.equal(rng, torch.get_rng_state()), "PSDB construction changed caller RNG"
    torch.manual_seed(912)
    other = PSDBP3()
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in block.state_dict().items())
    assert sum(p.numel() for p in block.parameters()) == 28864
    assert not list(block.buffers()), "PSDB v1 defines no new buffers"
    assert torch.count_nonzero(block.W_o.weight) == 0
    assert block.W_phi_q.in_channels == 32 and block.W_phi_q.out_channels == 8
    assert block.W_phi_k.in_channels == 64 and block.W_phi_k.out_channels == 8
    assert sum(1 for name, _ in block.named_parameters() if name.startswith("W_phi_k.")) == 1
    for name in ("W_d", "DW3", "W_q", "W_g", "W_phi_q", "W_phi_k"):
        assert torch.count_nonzero(getattr(block, name).weight) > 0, name
    for h, w in ((8, 10), (7, 9)):
        p2 = torch.randn(2, 64, h, w)
        p3 = torch.randn(2, 256, (h + 1) // 2, (w + 1) // 2)
        for training in (True, False):
            block.train(training)
            assert torch.equal(block(p2, p3), p3)
    try:
        block(torch.zeros(1, 64, 7, 9), torch.zeros(1, 256, 3, 5))
    except ValueError as error:
        assert "ceil(P2/2)" in str(error)
    else:
        raise AssertionError("Invalid P2/P3 geometry was silently accepted")

    block.train()
    p2, p3 = torch.randn(2, 64, 8, 10), torch.randn(2, 256, 4, 5)
    target = torch.randn_like(p3)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    F.mse_loss(block(p2, p3), target).backward()
    first = {}
    for name, param in block.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        first[name] = float(param.grad.norm())
        assert (first[name] > 0) == (name == "W_o.weight"), (name, first[name])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(block(p2, p3), target).backward()
    second = {name: float(param.grad.norm()) for name, param in block.named_parameters()}
    for name in ("W_d.weight", "DW3.weight", "W_q.weight", "W_g.weight", "W_phi_q.weight", "W_phi_k.weight"):
        assert torch.isfinite(dict(block.named_parameters())[name].grad).all() and second[name] > 0, name
    # Isolate the added path: P2 has no backbone route and P3 is held fixed.
    independent_p2 = p2.detach().clone().requires_grad_()
    residual = block(independent_p2, p3.detach()) - p3.detach()
    residual.square().mean().backward()
    bypass_p2_grad = float(independent_p2.grad.norm())
    assert torch.isfinite(independent_p2.grad).all() and bypass_p2_grad > 0
    disabled = deepcopy(block)
    with torch.no_grad():
        disabled.W_o.weight.zero_()
        difference = float((block(p2, p3) - disabled(p2, p3)).abs().max())
    assert difference > 0
    stream = io.BytesIO()
    torch.save(block.state_dict(), stream)
    stream.seek(0)
    loaded = PSDBP3()
    loaded.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
    assert torch.equal(block(p2, p3), loaded(p2, p3))
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in block.state_dict().items())
    return {"parameters": 28864, "buffers": 0, "initial_identity_exact": True,
            "initial_gradient_norms": first, "after_one_update_gradient_norms": second,
            "isolated_bypass_p2_gradient_norm": bypass_p2_grad,
            "nonzero_vs_disabled_max_abs": difference, "nonzero_state_reload_exact": True,
            "fixed_seed_and_caller_rng_preserved": True}


def attention_checks():
    """Independent formula, bounded weights, content sensitivity and nonzero-Wo degeneration."""
    torch.manual_seed(440)
    block = PSDBP3().eval()
    p2, t3 = torch.randn(2, 64, 7, 9), torch.randn(2, 256, 4, 5)
    phases = block.phase_rearrange(p2).chunk(4, dim=1)
    semantic = block.GN_Q(block.W_q(t3))
    q = F.normalize(F.conv2d(semantic.float(), block.W_phi_q.weight.float()), dim=1, eps=1e-6)
    keys = [F.normalize(F.conv2d(phase.float(), block.W_phi_k.weight.float()), dim=1, eps=1e-6)
            for phase in phases]
    manual_a = torch.stack([(q * key).sum(dim=1) for key in keys], dim=1).mul(2.0).softmax(dim=1)
    actual_a, weights = block.phase_attention(phases, semantic), block.phase_weights(phases, semantic)
    assert torch.equal(manual_a, actual_a), "Attention differs from channel-normalized four-phase formula"
    assert weights.dtype == actual_a.dtype == torch.float32
    assert torch.equal(weights, 0.75 + actual_a)
    assert torch.isfinite(weights).all() and weights.min() >= .75 and weights.max() <= 1.75
    assert torch.allclose(weights.sum(1), torch.full_like(weights[:, 0], 4), atol=5e-7, rtol=0)
    assert float((actual_a - .25).abs().max()) > 1e-4, "Accidentally uniform q/k initialization"
    q_change = float((actual_a - block.phase_attention(phases, semantic.roll(1, 1))).abs().max())
    changed = list(phases)
    changed[2] = -phases[2]
    phase_change = float((actual_a - block.phase_attention(tuple(changed), semantic)).abs().max())
    assert q_change > 1e-4 and phase_change > 1e-4, "Attention insensitive to Q or a phase"
    weighted = torch.cat([weights[:, i:i+1].to(phase.dtype) * phase for i, phase in enumerate(phases)], 1)
    seen = []
    handle = block.W_d.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        block(p2, t3)
    finally:
        handle.remove()
    assert torch.equal(seen[0], weighted) and weighted.shape[1] == 256

    # Shared parameters are used directly to evaluate the historical SDB formula.
    # A nonzero projection makes equality informative rather than a zero-residual tautology.
    with torch.no_grad():
        block.W_o.weight.normal_(std=.03)
    detail = F.silu(block.GN_D(block.DW3(block.W_d(torch.cat(phases, 1)))))
    gate = block.W_g(torch.cat([detail, semantic, detail * semantic], 1)).sigmoid()
    original_delta = block.W_o(gate * detail)
    original_output = t3 + original_delta
    normal_output = block(p2, t3)
    with patch.object(block, "phase_weights", side_effect=lambda phases, semantic: torch.ones(
            semantic.shape[0], 4, *semantic.shape[-2:], device=semantic.device, dtype=torch.float32)):
        uniform_output = block(p2, t3)
    assert torch.count_nonzero(original_delta) > 0
    assert torch.equal(uniform_output, original_output), "Uniform weights do not recover the original SDB formula"
    assert torch.equal(uniform_output - t3, original_output - t3)
    normal_difference = float((normal_output - uniform_output).abs().max())
    assert normal_difference > 1e-5, "q/k do not affect the nonzero branch output"
    return dict(status="PASSED", shared_k_parameter_sets=1, attention_formula_exact=True,
                embedding_normalization_dim=1, softmax_phase_dim=1, attention_dtype=str(actual_a.dtype),
                weight_min=float(weights.min()), weight_max=float(weights.max()),
                weight_sum_max_abs_error=float((weights.sum(1) - 4).abs().max()),
                phase_major_broadcast_input_exact=True, semantic_change_attention_max_abs=q_change,
                one_phase_change_attention_max_abs=phase_change, nonzero_Wo_uniform_SDB_formula_exact=True,
                nonzero_uniform_delta_norm=float(original_delta.norm()),
                selected_vs_uniform_output_max_abs=normal_difference)


def parser_checks():
    pairs = (("rtdetr-resnet18-lite-cbr-lif-down.yaml", "rtdetr-resnet18-lite-cbr-lif-psdb-p3-v1.yaml",
              20149765, 20178629, 19944965, 19973829),
             ("rtdetr-resnet18-lite.yaml", "rtdetr-resnet18-lite-psdb-p3-v1.yaml",
              20082772, 20111636, 19877716, 19906580))
    reports = []
    new_state = None
    for original, target, parent_count, target_count, parent_fused, target_fused in pairs:
        torch.manual_seed(42)
        parent = RTDETRDetectionModel(str(CFG / original), nc=1, verbose=False)
        parent_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = RTDETRDetectionModel(str(CFG / target), nc=1, verbose=False)
        assert torch.equal(parent_rng, torch.get_rng_state()), "Later public construction RNG changed"
        assert len(model.model) == len(parent.model) == 27
        assert type(model.model[19]) is PSDBRepC3
        assert type(parent.model[19]) is RepC3
        assert model.model[19].f == [18, 4] and 4 in model.save and len(model.model[19].m) == 3
        assert model.model[19].cv1.conv.in_channels == 512
        assert model.model[19].cv1.conv.out_channels == 128
        assert model.model[19].cv3.conv.out_channels == 256
        assert model.model[26].f == [19, 22, 25]
        for index in (20, 22, 25, 26):
            assert model.model[index].f == parent.model[index].f
            assert type(model.model[index]) is type(parent.model[index])
        public = parent.state_dict()
        state = model.state_dict()
        assert all(k in state and torch.equal(v, state[k]) for k, v in public.items())
        assert all(k.startswith("model.19.psdb.") for k in state.keys() - public.keys())
        psdb_state = model.model[19].psdb.state_dict()
        if new_state is not None:
            assert all(torch.equal(v, new_state[k]) for k, v in psdb_state.items())
        new_state = {k: v.clone() for k, v in psdb_state.items()}
        assert sum(p.numel() for p in parent.parameters()) == parent_count
        assert sum(p.numel() for p in model.parameters()) == target_count
        parent.eval().fuse(verbose=False)
        model.eval().fuse(verbose=False)
        assert sum(p.numel() for p in parent.parameters()) == parent_fused
        assert sum(p.numel() for p in model.parameters()) == target_fused
        assert isinstance(model.model[19].psdb.GN_D, nn.GroupNorm)
        assert isinstance(model.model[19].psdb.GN_Q, nn.GroupNorm)
        assert all(torch.equal(v, model.model[19].psdb.state_dict()[k]) for k, v in new_state.items())
        reports.append({"variant_yaml": target, "common_state_items_exact": len(public),
                        "parent_unfused_parameters": parent_count, "target_unfused_parameters": target_count,
                        "parent_fused_parameters": parent_fused, "target_fused_parameters": target_fused,
                        "internal_repeats": 3, "p2_saved": True})
        del parent, model
    scaled = yaml_model_load(str(CFG / pairs[0][1]))
    scaled["scales"]["l"][0] = 0.5
    model = RTDETRDetectionModel(scaled, nc=1, verbose=False)
    assert type(model.model[19]) is PSDBRepC3 and len(model.model[19].m) == 2
    return {"variants": reports, "depth_scaled_internal_repeats": 2,
            "new_state_equal_between_variants": True}


def count_checks():
    from thop import profile
    from thop.vision.basic_hooks import count_convNd

    # Exercise the actual installed child-convolution hook with both historical
    # total_ops types. No parent PSDB hook is registered, so convolutions count once.
    conv = nn.Conv2d(2, 3, 1, bias=False)
    x = torch.zeros(1, 2, 2, 2)
    y = conv(x)
    results = []
    for initial in (0, torch.zeros(1, dtype=torch.float64)):
        conv.total_ops = initial
        count_convNd(conv, (x,), y)
        results.append(float(conv.total_ops))
        assert results[-1] == 24
    block = PSDBP3().eval()
    # All non-convolution arithmetic is explicitly excluded in this count.
    raw_macs, parameters = profile(block, inputs=(torch.zeros(1, 64, 160, 160),
                                             torch.zeros(1, 256, 80, 80)),
                               custom_ops={nn.GroupNorm: lambda module, inputs, output: None}, verbose=False)
    # Functional FP32 q/k convolutions do not trigger Conv2d hooks. Add exactly
    # q once and the shared k four times, with no parent composite hook.
    assert raw_macs == 178790400, raw_macs
    functional_qk_macs = (32 * 8 + 4 * 64 * 8) * 80 * 80
    macs = raw_macs + functional_qk_macs
    assert macs == 193536000, macs
    assert sum(p.numel() for p in block.parameters()) == 28864
    return {"thop_int_and_tensor_total_ops": results, "hooked_convolution_macs": raw_macs,
            "functional_qk_supplement_macs": functional_qk_macs, "conv_only_macs_640": macs,
            "conv_only_gflops_640_at_2_flops_per_mac": 2 * macs / 1e9,
            "scope": "PARTIAL: convolutions only; functional q/k included exactly once; excludes GN, L2 normalization, dot products, softmax, activations and elementwise operations.",
            "thop_reported_parameters": parameters}


def warmup_checks():
    seen = []
    def capture(x):
        assert torch.isfinite(x).all() and torch.count_nonzero(x) == 0
        seen.append(list(x.shape))
    fake = SimpleNamespace(pt=True, jit=False, onnx=False, engine=False, saved_model=False, pb=False,
                           triton=True, nn_module=False, device=torch.device("cpu"), fp16=False, forward=capture)
    AutoBackend.warmup(fake, imgsz=(1, 3, 32, 32))
    assert seen == [[1, 3, 32, 32]]
    return {"finite_zero_input_observed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a JSON result (including failures) to this path")
    args = parser.parse_args()
    torch.set_num_threads(2)
    tracked = ["tools/check_psdb_p3_core.py", "ultralytics-main/ultralytics/nn/modules/psdb_p3.py",
               "ultralytics-main/ultralytics/nn/modules/cbr.py", "ultralytics-main/ultralytics/nn/modules/lif_down.py",
               "ultralytics-main/ultralytics/nn/modules/__init__.py", "ultralytics-main/ultralytics/nn/tasks.py",
               "ultralytics-main/ultralytics/nn/autobackend.py"]
    tracked.extend(str(p.relative_to(ROOT)).replace("\\", "/") for p in CFG.glob("*psdb-p3-v1.yaml"))
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    result = {"status": "RUNNING", "device": "cpu", "precision": "FP32", "torch": torch.__version__,
              "git_head": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                                    "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "tracked_dirty": bool(subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"],
                                                             cwd=ROOT, text=True).strip()),
              "source_lf_sha256": {path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                    for path in tracked},
              "scope": "Synthetic core/parser/count checks; not the server training gate."}
    try:
        for name, check in (("phases", phase_checks), ("core", core_checks), ("attention", attention_checks), ("parser", parser_checks),
                            ("complexity", count_checks), ("warmup", warmup_checks)):
            result["stage"] = name
            result[name] = check()
        result["status"] = "PASSED"
    except Exception:
        result["status"] = "FAILED"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        text = json.dumps(result, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
