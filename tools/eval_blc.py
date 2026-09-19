"""Frozen-best independent val/test, corrected_sorted_conf_mask_v1; never trains."""
from __future__ import annotations
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from blc_common import *
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300,
            augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    """Same reviewed algorithm as mother tools/c19_lif_v1_results.py at a0459d6."""
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    result, affected = [], 0
    for pred in preds:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        affected += int(not torch.equal(mask, score > conf))
        rows = rows[mask]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result, affected


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def evaluate(variant, split):
    require(split in ("val", "test"), "Explicit val or test only")
    p = paths(variant)
    weights = p["run"]/"weights/best.pt"
    require(weights.is_file(), "Trained formal best.pt missing")
    digest, data, code = sha256(weights), dataset_identity(p["data"]), code_identity()
    selection = p["evidence"]/"frozen_best.json"
    if not selection.exists():
        require(split == "val", "Freeze best through independent val before test")
        launches = sorted(p["evidence"].glob("training-*.json"))
        completed = bool(launches) and json.loads(launches[-1].read_text(encoding="utf-8")).get("status") == "COMPLETED"
        csv = p["run"]/"results.csv"
        epochs = max(0, len(csv.read_text().splitlines()) - 1) if csv.exists() else 0
        require(completed or epochs >= 200, "Formal training not completed; final best selection not frozen")
        write_json(selection, dict(weights=str(weights), sha256=digest, code=code, data=data, policy=POLICY))
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    require(frozen["sha256"] == digest and frozen["code"] == code and frozen["data"] == data, "Frozen weights/source/data changed")
    if split == "test":
        vals = sorted(p["evidence"].glob("val-*/metrics.json"))
        require(vals, "Independent completed val required before test")
        val = json.loads(vals[-1].read_text(encoding="utf-8"))
        require(val["status"] == "PASSED" and val["weights_sha256"] == digest and val["code"] == code
                and val["data"] == data and val["policy"] == POLICY, "Val/test identity mismatch")
    output = p["evidence"]/(split+"-"+stamp())
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", variant=variant, split=split, policy=POLICY, settings=EVAL, code=code,
                  data=data, weights_sha256=digest, precision_recall="each model maximum-F1 working point")
    counts = dict(images=0, GT=0, predictions=0, historical_mask_affected_images=0)
    seen = set()
    class Validator(RTDETRValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            require(not self.training and not self.args.half, "Independent FP32 protocol required")
            require(all(getattr(self.args, k) == v for k, v in EVAL.items()), "Evaluation recipe changed")
            self.expected = {str(Path(p).resolve()) for p in self.dataloader.dataset.im_files}
            require(len(self.expected) == len(self.dataloader.dataset.im_files), "Duplicate images")
        def postprocess(self, preds):
            rows, affected = postprocess(preds, self.args.imgsz, self.args.conf)
            counts["historical_mask_affected_images"] += affected
            return rows
        def update_metrics(self, preds, batch):
            for i, pred in enumerate(preds):
                path = str(Path(batch["im_file"][i]).resolve())
                require(path not in seen, "Duplicate evaluated image")
                seen.add(path)
                counts["images"] += 1
                counts["GT"] += int((batch["batch_idx"] == i).sum())
                counts["predictions"] += len(pred["conf"])
            return super().update_metrics(preds, batch)
        def finalize_metrics(self):
            require(seen == self.expected, "Incomplete split evaluation")
            return super().finalize_metrics()
    try:
        model = RTDETR(str(weights))
        verify_model(model.model, variant)
        metrics = model.val(validator=Validator, **EVAL, data=str(p["data"]), split=split, device="0",
                            plots=True, save_json=False, project=str(output), name="plots", exist_ok=False)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid ten-IoU AP")
        require(sha256(weights) == digest and dataset_identity(p["data"]) == data, "Evaluation inputs changed")
        report.update(status="PASSED", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      ap_iou=[round(.5+i*.05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      metrics=plain(metrics.results_dict), speed_ms_per_image=plain(metrics.speed), **counts)
        write_json(output/"curves.json", plain(metrics.curves_results))
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc(), **counts)
        raise
    finally:
        write_json(output/"metrics.json", report)
        print(output/"metrics.json", report["status"], flush=True)
    return report
