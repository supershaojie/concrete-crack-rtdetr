#!/usr/bin/env python3
"""Small CPU checks for SDB mathematics, parser wiring, RNG and parameter accounting.

These synthetic core diagnostics complement (do not replace) the real detection
loss and server B16/640/native AMP checks in check_sdb_p3.py.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import RepC3, SDBP3, SDBRepC3
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
        actual = SDBP3.phase_rearrange(p2)
        assert torch.equal(actual, expected), (h, w, actual, expected)
        cases.append({"p2_hw": [h, w], "exact_phase_and_padding": True})
    return cases


def core_checks():
    torch.manual_seed(309)
    rng = torch.get_rng_state().clone()
    block = SDBP3()
    assert torch.equal(rng, torch.get_rng_state()), "SDB construction changed caller RNG"
    torch.manual_seed(912)
    other = SDBP3()
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in block.state_dict().items())
    assert sum(p.numel() for p in block.parameters()) == 28096
    assert not list(block.buffers()), "SDB v1 defines no new buffers"
    assert torch.count_nonzero(block.W_o.weight) == 0
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
    for name in ("W_d.weight", "DW3.weight", "W_q.weight", "W_g.weight"):
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
    loaded = SDBP3()
    loaded.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
    assert torch.equal(block(p2, p3), loaded(p2, p3))
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in block.state_dict().items())
    return {"parameters": 28096, "buffers": 0, "initial_identity_exact": True,
            "initial_gradient_norms": first, "after_one_update_gradient_norms": second,
            "isolated_bypass_p2_gradient_norm": bypass_p2_grad,
            "nonzero_vs_disabled_max_abs": difference, "nonzero_state_reload_exact": True,
            "fixed_seed_and_caller_rng_preserved": True}


def parser_checks():
    pairs = (("rtdetr-resnet18-lite-cbr-lif-down.yaml", "rtdetr-resnet18-lite-cbr-lif-sdb-p3-v1.yaml",
              20149765, 20177861, 19944965, 19973061),
             ("rtdetr-resnet18-lite.yaml", "rtdetr-resnet18-lite-sdb-p3-v1.yaml",
              20082772, 20110868, 19877716, 19905812))
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
        assert type(model.model[19]) is SDBRepC3
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
        assert all(k.startswith("model.19.sdb.") for k in state.keys() - public.keys())
        sdb_state = model.model[19].sdb.state_dict()
        if new_state is not None:
            assert all(torch.equal(v, new_state[k]) for k, v in sdb_state.items())
        new_state = {k: v.clone() for k, v in sdb_state.items()}
        assert sum(p.numel() for p in parent.parameters()) == parent_count
        assert sum(p.numel() for p in model.parameters()) == target_count
        parent.eval().fuse(verbose=False)
        model.eval().fuse(verbose=False)
        assert sum(p.numel() for p in parent.parameters()) == parent_fused
        assert sum(p.numel() for p in model.parameters()) == target_fused
        assert isinstance(model.model[19].sdb.GN_D, nn.GroupNorm)
        assert isinstance(model.model[19].sdb.GN_Q, nn.GroupNorm)
        assert all(torch.equal(v, model.model[19].sdb.state_dict()[k]) for k, v in new_state.items())
        reports.append({"variant_yaml": target, "common_state_items_exact": len(public),
                        "parent_unfused_parameters": parent_count, "target_unfused_parameters": target_count,
                        "parent_fused_parameters": parent_fused, "target_fused_parameters": target_fused,
                        "internal_repeats": 3, "p2_saved": True})
        del parent, model
    scaled = yaml_model_load(str(CFG / pairs[0][1]))
    scaled["scales"]["l"][0] = 0.5
    model = RTDETRDetectionModel(scaled, nc=1, verbose=False)
    assert type(model.model[19]) is SDBRepC3 and len(model.model[19].m) == 2
    return {"variants": reports, "depth_scaled_internal_repeats": 2,
            "new_state_equal_between_variants": True}


def count_checks():
    from thop import profile
    from thop.vision.basic_hooks import count_convNd

    # Exercise the actual installed child-convolution hook with both historical
    # total_ops types. No parent SDB hook is registered, so convolutions count once.
    conv = nn.Conv2d(2, 3, 1, bias=False)
    x = torch.zeros(1, 2, 2, 2)
    y = conv(x)
    results = []
    for initial in (0, torch.zeros(1, dtype=torch.float64)):
        conv.total_ops = initial
        count_convNd(conv, (x,), y)
        results.append(float(conv.total_ops))
        assert results[-1] == 24
    block = SDBP3().eval()
    # All non-convolution arithmetic is explicitly excluded in this count.
    macs, parameters = profile(block, inputs=(torch.zeros(1, 64, 160, 160),
                                             torch.zeros(1, 256, 80, 80)),
                               custom_ops={nn.GroupNorm: lambda module, inputs, output: None}, verbose=False)
    assert macs == 178790400, macs
    assert sum(p.numel() for p in block.parameters()) == 28096
    return {"thop_int_and_tensor_total_ops": results, "conv_only_macs_640": macs,
            "conv_only_gflops_640_at_2_flops_per_mac": 2 * macs / 1e9,
            "scope": "Convolutions only; excludes GN, activation, sigmoid and elementwise operations.",
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
    tracked = ["tools/check_sdb_p3_core.py", "ultralytics-main/ultralytics/nn/modules/sdb_p3.py",
               "ultralytics-main/ultralytics/nn/modules/cbr.py", "ultralytics-main/ultralytics/nn/modules/lif_down.py",
               "ultralytics-main/ultralytics/nn/modules/__init__.py", "ultralytics-main/ultralytics/nn/tasks.py",
               "ultralytics-main/ultralytics/nn/autobackend.py"]
    tracked.extend(str(p.relative_to(ROOT)).replace("\\", "/") for p in CFG.glob("*sdb-p3-v1.yaml"))
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
        for name, check in (("phases", phase_checks), ("core", core_checks), ("parser", parser_checks),
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
