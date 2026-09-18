"""C+D bounded continuation audit only: no math/fusion, formal training, validation or test.

Exit 0 means diagnostics completed, including UNRESOLVED numerical findings.
Admission remains BLOCKED unless every requested control passes; read the JSON.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time

import torch

from dcc_common import controlled_models, runtime, require, sha256, write_json
from dcc_resume_audit import (ATOL, RTOL, make_audit_trainer, bounded_updates, audit_live_control,
                              audit_checkpoint_resume, seed42, rng_state, restore_rng, compare_states)
from c19_lif_v1_data import dataset_inventory, real_batch
from ultralytics import RTDETR


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    prior_rng = rng_state()
    flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report = dict(status="FAILED", admission="BLOCKED", variant=args.variant, runtime=runtime(),
                  initialization_sha256=sha256(args.initialized), seed=42, tolerances=dict(atol=ATOL, rtol=RTOL),
                  formal_training="NOT_STARTED", final_test="NOT_RUN", full_validation="NOT_RUN", devices={},
                  scope="One fixed B2/160 real train batch; bounded live-FP32 parent/DCC controls + checkpoint restore/replay",
                  precision=dict(deterministic_warn_only=True, cudnn_deterministic=True, cudnn_benchmark=False,
                                 cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32))
    started = time.perf_counter()
    try:
        parent, target, _ = controlled_models(args.source, args.variant)
        initialized = RTDETR(str(args.initialized)).model
        exact = compare_states(target.state_dict(), initialized.state_dict(), exact=True)
        require(exact["status"] == "PASSED", "Initialization differs from controlled public source")
        del initialized, exact
        report["dataset_identity"] = dataset_inventory(args.real_dataset)
        batch, report["samples"] = real_batch(args.real_dataset, size=160, count=2)
        require(batch["bboxes"].shape[0] > 0, "Fixed real batch has no valid GT")
        for device, amp, label in (("cpu", False, "cpu_fp32"), ("cuda", False, "cuda_fp32"), ("cuda", True, "cuda_native_amp")):
            if args.device != "all" and args.device != device:
                continue
            if device == "cuda" and not torch.cuda.is_available():
                report["devices"][label] = dict(status="PENDING", admission="BLOCKED", reason="CUDA unavailable")
                continue
            print("BEGIN C+D " + label, flush=True)
            entry = report["devices"][label] = dict(status="RUNNING", admission="BLOCKED")
            data = {key: value.to(device) for key, value in batch.items()}
            for model_name, model in (("parent_cbr_lif", parent), ("dcc", target)):
                seed42()
                trainer = make_audit_trainer(deepcopy(model), amp, device)
                current = entry[model_name] = {}
                current["updates"] = bounded_updates(trainer, data, amp, max_batches=16, target_updates=3)
                current["live_control"] = audit_live_control(trainer, data, amp, args.output / label / model_name / "live", model_name)
                if model_name == "dcc":
                    current["checkpoint_resume"] = audit_checkpoint_resume(trainer, data, amp, args.output / label / model_name / "checkpoint")
                del trainer
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                write_json(args.output / "resume_checks.json", report)
            checks = [entry["parent_cbr_lif"]["live_control"], entry["dcc"]["live_control"], entry["dcc"]["checkpoint_resume"]]
            entry["status"] = "PASSED" if all(item["status"] == "PASSED" for item in checks) else "UNRESOLVED"
            entry["admission"] = "PASSED" if entry["status"] == "PASSED" else "BLOCKED"
            del data
            print(entry["status"] + " C+D " + label, flush=True)
        report["status"] = "PASSED" if all(item["status"] == "PASSED" for item in report["devices"].values()) else "UNRESOLVED"
        report["admission"] = "PASSED" if report["status"] == "PASSED" and set(report["devices"]) == {
            "cpu_fp32", "cuda_fp32", "cuda_native_amp"} else "BLOCKED"
    except BaseException as error:
        report.update(status="FAILED", admission="BLOCKED", error=repr(error))
        raise
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[2:]
        restore_rng(prior_rng)
        report["elapsed_seconds"] = time.perf_counter() - started
        from train_dcc import code_identity
        report["code_identity"] = code_identity()
        write_json(args.output / "resume_checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "initialized", "real-dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", choices=("cbr_lif_dcc_v1",), default="cbr_lif_dcc_v1")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "admission", "formal_training", "final_test")}, indent=2))
    return 0  # Completed diagnostics are not admission; server wrapper checks the JSON.


if __name__ == "__main__":
    raise SystemExit(main())
