"""Explicit independent val/test of the same training-selected best.pt, in FP32."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from init_psdb_p3 import ROOT, VARIANTS, require, sha256, runtime, verify_model
from train_psdb_p3 import DEFAULT_VARIANT, paths, run_state, validate_plan, verify_data, write_json
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    """Existing corrected policy: sort complete rows, then mask sorted confidence."""
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    result = []
    for pred in preds:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        rows = rows[rows[:, 4] > conf]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result


def evaluate(split, variant=DEFAULT_VARIANT, device="0"):
    p = paths(variant)
    state = run_state(p)
    require(state["status"] in {"COMPLETED_200", "EARLY_STOPPED"} or
            (state["status"] == "FAILED" and state.get("final_validation") == "FAILED"),
            "Independent evaluation requires finished training; test is forbidden during training")
    plan = json.loads((p["launch"] / "plan.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    weights = p["run"] / "weights/best.pt"
    require(weights.is_file(), "Training-selected best.pt missing")
    digest = sha256(weights)
    data_inventory = verify_data(plan["data"])
    output = p["launch"] / ("evaluation_" + split)
    require(not output.exists(), "Existing evaluation preserved; do not silently suffix/overwrite")
    if split == "test":
        prior = json.loads((p["launch"] / "evaluation_val/metrics.json").read_text(encoding="utf-8"))
        require(prior["status"] == "PASSED" and prior["split"] == "val" and prior["policy"] == POLICY,
                "Test requires completed independent val")
        require(prior["checkpoint_sha256"] == digest and prior["commit"] == plan["commit"] and
                prior["data_sha256"] == plan["data_sha256"] and prior["dataset_inventory"] == data_inventory and
                prior["settings"] == EVAL, "Val/test checkpoint, code, data or settings changed")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", split=split, variant=variant, checkpoint=str(weights), checkpoint_sha256=digest,
                  commit=plan["commit"], runtime=runtime(), data_sha256=plan["data_sha256"], dataset_inventory=data_inventory,
                  policy=POLICY, settings=EVAL, selection="training val selected best.pt; no test model/threshold selection",
                  precision_recall_policy="model own maximum-F1 working point")
    try:
        model = RTDETR(str(weights))
        verify_model(model.model, variant)
        require(model.model.model[-1].nc == 1, "Evaluation requires nc=1")
        class CorrectedValidator(RTDETRValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                require(not self.training and not self.args.half, "Independent FP32 validation required")
                require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()),
                        "Effective evaluation settings changed")
                require(len(self.dataloader.dataset.im_files) == data_inventory[split]["images"], "Incomplete split")
                report["actual_settings"] = vars(self.args).copy()
            def postprocess(self, preds):
                return postprocess(preds, self.args.imgsz, self.args.conf)
        metrics = model.val(validator=CorrectedValidator, **dict(EVAL, data=plan["data"], split=split, device=device,
                            plots=True, save_json=False, save_txt=False, project=str(output), name="plots", exist_ok=False))
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Unexpected/nonfinite AP")
        require(sha256(weights) == digest and sha256(plan["data"]) == plan["data_sha256"], "Evaluation inputs changed")
        report.update(status="PASSED", P=float(metrics.box.mp), R=float(metrics.box.mr), AP50=float(metrics.box.map50),
                      AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)], AP_by_class=ap.tolist(),
                      speed_ms_per_image=metrics.speed, images=data_inventory[split]["images"])
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.split, args.variant, args.device)
