#!/usr/bin/env python
"""Bounded CPU parameter/MAC comparison, identical nc=1, B=1, 640 input.

THOP totals are explicitly PARTIAL (the unchanged detector contains functional
attention and other operations without complete hooks). The added BFR major
MAC count is explicit, and never double-counts functional Conv/Linear layers.
Only disposable deepcopies are profiled; THOP buffers cannot enter checkpoints.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
from thop import profile

from ultralytics.nn.modules.bfr_p4 import BFRP4, count_bfr_p4
from ultralytics.nn.tasks import RTDETRDetectionModel

MODELS = {
    "cbr_lif_parent": ("rtdetr-resnet18-lite-cbr-lif-down.yaml", 20149765, 19944965),
    "cbr_lif_bfr_p4_v1": ("rtdetr-resnet18-lite-cbr-lif-bfr-p4-v1.yaml", 20170453, 19965653),
    "c2_parent": ("rtdetr-resnet18-lite.yaml", 20082772, 19877716),
    "bfr_p4_v1": ("rtdetr-resnet18-lite-bfr-p4-v1.yaml", 20103460, 19898404),
}


def profile_copy(model, image):
    count = sum(p.numel() for p in model.parameters())
    clone = copy.deepcopy(model).eval()
    started = time.perf_counter()
    try:
        macs, thop_params = profile(clone, inputs=(image,), custom_ops={BFRP4: count_bfr_p4}, verbose=False)
    finally:
        # Some THOP versions leave buffers on unrecognized containers/GroupNorm.
        for module in clone.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)
    assert not any(k.endswith(("total_ops", "total_params")) for k in model.state_dict())
    return {"parameters": count, "thop_reported_parameters_not_authoritative": thop_params,
            "partial_macs": macs, "partial_gflops_two_per_mac": 2 * macs / 1e9,
            "profile_wall_seconds": time.perf_counter() - started,
            "cache": "BFR constants rebuilt each call; no BFR cold/hot cache distinction"}


def run_checks():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        image = torch.randn(1, 3, 640, 640)
        for name, (yaml_name, expected, expected_fused) in MODELS.items():
            torch.random.default_generator.manual_seed(42)
            model = RTDETRDetectionModel(str(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr" / yaml_name), nc=1, verbose=False).eval()
            assert sum(p.numel() for p in model.parameters()) == expected
            rows[name] = {"yaml": yaml_name, "unfused": profile_copy(model, image)}
            fused = copy.deepcopy(model).fuse(verbose=False)
            assert sum(p.numel() for p in fused.parameters()) == expected_fused
            rows[name]["fused"] = profile_copy(fused, image)
    comparisons = {}
    for target, parent in (("cbr_lif_bfr_p4_v1", "cbr_lif_parent"), ("bfr_p4_v1", "c2_parent")):
        for mode in ("unfused", "fused"):
            parameter_delta = rows[target][mode]["parameters"] - rows[parent][mode]["parameters"]
            mac_delta = rows[target][mode]["partial_macs"] - rows[parent][mode]["partial_macs"]
            assert parameter_delta == 20688
            assert mac_delta == 34410496, (target, mode, mac_delta)
            comparisons[target + "/" + mode] = {"parameters_delta": parameter_delta, "major_macs_delta": mac_delta}
    return {"status": "PASSED", "coverage": "PARTIAL", "device": "cpu", "nc": 1, "input": [1, 3, 640, 640],
            "new_branch_major_macs": 34410496, "new_branch_major_gflops": .068820992,
            "exclusions": "BFR GN/statistics/gain/activation/elementwise/basis generation; detector functional attention/deformable sampling and any unregistered THOP operations",
            "parameters_policy": "sum(model.parameters()), excluding all profiler/runtime buffers",
            "rows": rows, "differences": comparisons,
            "peak_cuda_memory": "PENDING CUDA capacity preflight; this tool performs CPU diagnostics only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bfr_p4/ops_cpu.json")
    args = parser.parse_args()
    report = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "coverage": report["coverage"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
