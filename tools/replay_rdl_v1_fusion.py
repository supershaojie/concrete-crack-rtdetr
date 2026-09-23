"""Replay a trusted saved RDL fusion fixture without rebuilding the lifecycle."""
import argparse
import json
import os
from pathlib import Path

import torch

from init_c19_lif_v1 import require, sha256
from ultralytics.utils.patches import torch_load
from c19_lif_v1_diagnostic import atomic_json, restore_rng
from rdl_v1_fusion import check_ema_fusion, state_hash
from rdl_v1_fusion_acceptance import review_fusion


def replay(fixture, original_report, output, device):
    require(not output.exists(), "Use a new output directory; original FAIL evidence is immutable")
    old = json.loads(original_report.read_text(encoding="utf-8"))
    require(old["original_status"] in {"PASS", "FAIL"}, "Replay requires an original comparison report")
    # Pickle is loaded only from the user's explicitly selected local/server fixture.
    saved = torch_load(fixture, map_location="cpu")
    require(state_hash(saved["model"]) == old["before_predict"]["state_sha256"], "Fixture model/report mismatch")
    require(tuple(saved["image"].shape) == (2,3,160,192), "Wrong lifecycle fixture shape")
    for name, output_tensor in (("before", saved["before"]), ("after", saved["after"])):
        require(tuple(output_tensor.shape) == (2,300,5) and torch.isfinite(output_tensor).all(), "Invalid saved " + name)
    report = dict(status="FAIL", original_report_sha256=sha256(original_report), fixture_sha256=sha256(fixture),
                  original_report=str(original_report.resolve()), fixture=str(fixture.resolve()), lifecycle="NOT_RERUN")
    # Reproduce the saved forward runtime, then restore the caller even on failure.
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        torch.set_num_threads(old["runtime"]["threads"])
        torch.use_deterministic_algorithms(old["runtime"]["deterministic"], warn_only=True)
        torch.backends.cudnn.benchmark = old["runtime"]["cudnn_benchmark"]
        # v2's original producer called native init_seeds(42, deterministic=True),
        # which also set these values, although its JSON omitted them.
        torch.backends.cudnn.deterministic = old["runtime"].get("cudnn_deterministic",old["runtime"]["deterministic"])
        workspace = old["runtime"].get("cublas_workspace_config", ":4096:8" if old["runtime"]["deterministic"] else previous[5])
        if workspace is None: os.environ.pop("CUBLAS_WORKSPACE_CONFIG",None)
        else: os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        restore_rng(saved["rng"])
        image = saved["image"].to(device)
        batch = dict(img=image, bboxes=torch.tensor([[.4,.4,.2,.1],[.6,.6,.3,.2],[.2,.7,.1,.25]],device=device),
                     cls=torch.zeros(3,1,device=device),batch_idx=torch.tensor([0,1,1],device=device))
        result = check_ema_fusion(saved["model"].to(device), batch, output, old["controlled_source_sha256"], reference=saved)
        report.update(status="PASS", original_status=result["original_status"], fusion_acceptance=result["fusion_acceptance"],
                      precision_scope=result["precision_scope"], saved_fixture_replay=result["saved_fixture_replay"])
        return report
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cudnn.deterministic = previous[4]
        if previous[5] is None: os.environ.pop("CUBLAS_WORKSPACE_CONFIG",None)
        else: os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous[5]
        atomic_json(output.parent/(output.name+"_result.json"), report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.fixture is None:
        require(not args.output.exists(), "Use a new report path")
        original = json.loads(args.report.read_text(encoding="utf-8"))
        result = dict(mode="SAVED_EVIDENCE_REVIEW_ONLY", original_report_sha256=sha256(args.report),
                      original_status=original["original_status"], fusion_acceptance=review_fusion(original))
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["fusion_acceptance"]["accepted"] else 1)
    replay(args.fixture, args.report, args.output, args.device)
