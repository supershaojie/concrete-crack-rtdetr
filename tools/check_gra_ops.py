"""Finite CLI, evaluation-matching, threshold grouping and archive checks; no training/test."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from init_gra import ROOT, require, write_json, sha256, runtime
from train_gra import source_manifest
from eval_gra import postprocess, fixed_precision, EVAL, POLICY
from pack_gra_light import package
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils.metrics import DetMetrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Preserve existing ops checks")
    args.output.mkdir(parents=True)
    report = {"status": "FAILED", "formal_training": "NOT_STARTED", "final_test": "NOT_RUN", "runtime": runtime()}
    try:
        commands = [["init_gra.py", "--help"], ["check_gra.py", "--help"], ["preflight_gra.py", "--help"],
                    ["train_gra.py", "--help"], ["train_gra.py", "plan", "--help"], ["train_gra.py", "start", "--help"],
                    ["train_gra.py", "resume", "--help"], ["eval_gra.py", "evaluate", "--help"],
                    ["eval_gra.py", "fixed-p", "--help"], ["pack_gra_light.py", "--help"]]
        report["help"] = []
        for script, *argv in commands:
            result = subprocess.run([sys.executable, str(ROOT / "tools" / script), *argv], cwd=ROOT, capture_output=True, text=True, timeout=120)
            require(result.returncode == 0, f"Help failed: {script} {argv}: {result.stderr}")
            report["help"].append({"command": [script, *argv], "exit_code": result.returncode})
        result = subprocess.run([sys.executable, str(ROOT / "tools/train_gra.py"), "plan", "--initialized", str(ROOT / "weights/cbr_lif_gra_v1_controlled_init.pt")],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        require(result.returncode == 0, "Plan command failed")
        plan = json.loads(result.stdout)
        require(plan["status"] == "PLAN_ONLY" and plan["args"]["epochs"] == 200 and plan["args"]["batch"] == 16,
                "Plan changed formal recipe")
        report["plan"] = {"status": "PASSED", "fields": len(plan["args"]), "changed": [r["field"] for r in plan["differences"] if r["changed"]]}
        result = subprocess.run([sys.executable, str(ROOT / "tools/train_gra.py"), "start", "--initialized", "missing.pt",
                                 "--audit", "missing.json", "--checks", "missing.json", "--preflight", "missing.json",
                                 "--source", "missing.pt", "--expected-sha", "invalid"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        require(result.returncode != 0 and "Full lowercase delivery SHA required" in result.stderr, "Start missing identity not rejected")
        report["invalid_start_rejected_before_training"] = {"status": "PASSED", "exit_code": result.returncode}

        # Low-score row first catches old unsorted confidence mask; no model inference.
        raw = torch.tensor([[[.1, .1, .1, .1, .0001], [.5, .5, .2, .2, .9], [.8, .8, .1, .1, .8]]])
        rows, affected = postprocess(raw, 640, .001)
        require(torch.allclose(rows[0]["conf"], torch.tensor([.9, .8])) and affected == 1, "Sorted confidence mask incorrect")
        validator = object.__new__(RTDETRValidator)
        validator.device = torch.device("cpu")
        validator.iouv = torch.linspace(.5, .95, 10)
        validator.niou, validator.seen = 10, 0
        validator.args = SimpleNamespace(single_cls=False, plots=False, save_json=False, save_txt=False)
        validator.metrics = DetMetrics()
        batch = {"img": torch.zeros(1, 3, 640, 640), "cls": torch.zeros(1, 1), "bboxes": torch.tensor([[.5, .5, .2, .2]]),
                 "batch_idx": torch.zeros(1), "ori_shape": [(480, 800)], "ratio_pad": [(1., 1.)], "im_file": ["fixture.jpg"]}
        gt = validator._prepare_batch(0, batch)
        direct = validator._process_batch(validator.scale_preds(rows[0], gt), gt)["tp"]
        validator.update_metrics(rows, batch)
        native = validator.metrics.stats["tp"][0]
        require(np.array_equal(direct, native) and native[0].all() and not native[1].any(), "Native update_metrics matching differs")
        require(torch.allclose(gt["bboxes"], torch.tensor([[256., 256., 384., 384.]])), "GT should be in 640 input-space")
        scale = torch.tensor([800 / 640, 480 / 640] * 2)
        report["evaluation_fixture"] = {"status": "PASSED", "sorted_scores": rows[0]["conf"].tolist(),
                                        "GT_input_xyxy": gt["bboxes"].tolist(), "GT_original_xyxy": (gt["bboxes"] * scale).tolist(),
                                        "TP_all_10_IoUs": native.tolist(), "max_input_box_error": float((gt["bboxes"] - rows[0]["bboxes"][:1]).abs().max())}

        predictions = args.output / "fixture_predictions.jsonl.gz"
        fixture = {"image": "fixture", "predictions": [{"bbox": [0, 0, 1, 1], "class_id": 0, "score": .9, "tp50_fixed_pool": True},
                    {"bbox": [5, 5, 6, 6], "class_id": 0, "score": .9, "tp50_fixed_pool": False}],
                   "ground_truth": [{"bbox": [0, 0, 1, 1], "class_id": 0}]}
        with gzip.open(predictions, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(fixture) + "\n")
        metrics = args.output / "fixture_metrics.json"
        write_json(metrics, {"status": "COMPLETED", "split": "val", "policy": POLICY, "settings": EVAL,
                            "evidence_scope": "full_split", "export_complete": True, "ground_truth": 1,
                            "split_paths_sha256": hashlib.sha256(b"fixture").hexdigest(),
                            "images": 1, "checkpoint_sha256": "fixture_only", "data_sha256": "fixture_only", "predictions_gt_sha256": sha256(predictions)})
        common = dict(predictions=predictions, metrics=metrics)
        not_achieved = fixed_precision(SimpleNamespace(**common, output=args.output / "not_achieved.json", precision_target=.8656))
        achieved = fixed_precision(SimpleNamespace(**common, output=args.output / "achieved.json", precision_target=.5))
        require(not_achieved["status"] == "NOT_ACHIEVED" and not_achieved["working_point"] is None and
                achieved["working_point"]["TP"] == achieved["working_point"]["FP"] == 1 and achieved["working_point"]["threshold"] == .9,
                "Equal-score predictions were split or unattainable working point invented")
        report["fixed_p_tie_groups"] = {"status": "PASSED", "target_08656": not_achieved, "target_05": achieved}
        archive = package("cbr_lif_gra_v1", args.output / "fixture_LIGHT.tar.gz", run=args.output / "absent_run")
        report["packing"] = json.loads(Path(str(archive) + ".verification.json").read_text(encoding="utf-8"))
        report["status"] = "PASSED"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["source_manifest"] = source_manifest()
        write_json(args.output / "ops_checks.json", report)


if __name__ == "__main__":
    main()
