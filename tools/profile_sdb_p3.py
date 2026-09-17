#!/usr/bin/env python3
"""Measure nc=1 parent/target THOP counts at 640, with explicit incomplete operator scope."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import torch
import thop
from thop.profile import register_hooks

from init_sdb_p3 import ROOT, VARIANTS, build


def measure(model):
    # THOP may attach counting buffers to unknown composite modules. Profile an
    # expendable copy so no counter can leak into a training/checkpoint model.
    candidate = deepcopy(model).cpu().eval()
    parameters = sum(p.numel() for p in candidate.parameters())
    leaf_types = {type(module) for module in candidate.modules() if not list(module.children())}
    covered = {kind.__name__: register_hooks[kind].__name__ for kind in leaf_types if kind in register_hooks}
    unregistered = sorted(kind.__name__ for kind in leaf_types if kind not in register_hooks)
    started = time.perf_counter()
    with torch.no_grad():
        operations, thop_parameters = thop.profile(candidate, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
    elapsed = time.perf_counter() - started
    del candidate
    gc.collect()
    return dict(actual_parameters=parameters, thop_reported_parameters=int(thop_parameters),
                thop_raw_operation_count=float(operations), thop_2x_raw_gflops=float(operations) * 2 / 1e9,
                seconds=elapsed, registered_leaf_hooks=covered, unregistered_leaf_types=unregistered)


def run():
    results = {}
    for variant in VARIANTS:
        rows = results[variant] = {}
        for baseline, label in ((True, "parent"), (False, "target")):
            print(f"PROFILE {variant} {label} nc=1 CPU 640", flush=True)
            model = build(variant, nc=1, baseline=baseline).cpu().eval()
            rows[label] = {"unfused": measure(model)}
            model.fuse(verbose=False)
            rows[label]["fused"] = measure(model)
            del model
            gc.collect()
        rows["deltas"] = {}
        for state in ("unfused", "fused"):
            parent, target = rows["parent"][state], rows["target"][state]
            delta = {"parameters": target["actual_parameters"] - parent["actual_parameters"],
                     "thop_raw_operation_count": target["thop_raw_operation_count"] - parent["thop_raw_operation_count"],
                     "thop_2x_raw_gflops": target["thop_2x_raw_gflops"] - parent["thop_2x_raw_gflops"]}
            if delta["parameters"] != 28096:
                raise RuntimeError(f"Unexpected {variant}/{state} parameter increment: {delta}")
            rows["deltas"][state] = delta
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Write measured partial-scope JSON")
    args = parser.parse_args()
    torch.set_num_threads(2)
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    files = ["tools/profile_sdb_p3.py", "tools/init_sdb_p3.py", "ultralytics-main/ultralytics/nn/tasks.py",
             "ultralytics-main/ultralytics/nn/modules/sdb_p3.py", "ultralytics-main/ultralytics/nn/modules/cbr.py",
             "ultralytics-main/ultralytics/nn/modules/lif_down.py"]
    model_dir = ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr"
    files += [str((model_dir / name).relative_to(ROOT)).replace("\\", "/")
              for row in VARIANTS.values() for name in row[:2]]
    report = dict(status="RUNNING", operator_scope="PARTIAL", device="cpu", precision="FP32", batch=1, nc=1,
                  input_shape=[1, 3, 640, 640], torch=torch.__version__, thop=thop.__version__,
                  git_head=subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  tracked_dirty=bool(subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"],
                                                            cwd=ROOT, text=True).strip()),
                  source_lf_sha256={name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                   for name in files},
                  counting_convention="Default installed THOP hooks, no custom parent or child hooks. 2x raw shown for historical comparison; not complete network FLOPs.",
                  exclusions=["Functional grid_sample in original CBR and deformable attention is uncounted; no complete GFLOPs claim.",
                              "Functional transformer matrix products/softmax and some attention work are uncounted.",
                              "Functional SiLU/sigmoid, D*Q, gate*D, residual addition and spatial rearrangement are uncounted.",
                              "GN is uncounted when GroupNorm appears in unregistered_leaf_types. Registered hooks and exclusions are recorded per model."],
                  analytical_sdb=dict(parameters=28096, conv_macs_640=178790400,
                                      conv_gflops_640_at_2_flops_per_mac=0.3575808,
                                      excludes="GroupNorm, activation, sigmoid and elementwise/spatial operations"),
                  formal_training="NOT_STARTED", test="NOT_RUN")
    try:
        report["variants"] = run()
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "operator_scope": "PARTIAL", "output": str(args.output)}))


if __name__ == "__main__":
    main()
