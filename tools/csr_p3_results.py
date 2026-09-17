"""Explicit independent CSR-P3 val/test using corrected_sorted_conf_mask_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from init_csr_p3 import VARIANTS, require, runtime, sha256, verify_model, write_json
from train_csr_p3 import check_preflight, load_json, paths, require_clean, run_state
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import init_seeds

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=0.001, iou=0.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    """Apply the confidence mask after sorting, without a second NMS."""
    preds = preds[0] if isinstance(preds, (tuple, list)) else preds
    output = []
    for prediction in preds:
        boxes = ops.xywh2xyxy(prediction[:, :4] * imgsz)
        score, cls = prediction[:, 4:].max(-1)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[score.argsort(descending=True)]
        rows = rows[rows[:, 4] > conf]
        output.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return output


def evaluate(variant, split, device="0"):
    require_clean()
    p = paths(variant)
    state = run_state(p)
    require(state["status"] in {"COMPLETED_200", "EARLY_STOPPED"} or
            (state["status"] == "FAILED" and state.get("training_finished") is True and state.get("final_eval") == "FAILED"),
            "Independent evaluation requires finished training; test is forbidden during training")
    record = load_json(p["launch"] / "plan.json")
    attempt = Path(state["attempt"])
    check_preflight(record, attempt / "preflight.json")
    require(record["runtime"]["commit"] == runtime()["commit"], "Source changed since training")
    checkpoint = p["run"] / "weights/best.pt"
    require(checkpoint.is_file(), "Training-selected best.pt missing")
    checkpoint_hash = sha256(checkpoint)
    output = p["launch"] / ("evaluation_" + split)
    require(not output.exists(), "Existing evaluation protected; inspect prior report")
    if split == "test":
        val = load_json(p["launch"] / "evaluation_val/metrics.json")
        require(val.get("status") == "PASSED" and val["split"] == "val" and val["checkpoint_sha256"] == checkpoint_hash,
                "Test requires successful independent val of exactly the same best.pt SHA256")
        require(val["commit"] == runtime()["commit"] and val["policy"] == POLICY and val["settings"] == EVAL,
                "Source/protocol differs from independent val")
        require(val["data_sha256"] == sha256(record["args"]["data"]), "Data changed since independent val")
    settings = dict(EVAL, data=record["args"]["data"], split=split, device=device, plots=True,
                    project=str(output), name="plots", exist_ok=False, save_json=False, save_txt=False)
    report = dict(status="FAILED", variant=variant, split=split, policy=POLICY, checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, settings=EVAL, data_sha256=sha256(record["args"]["data"]),
                  commit=runtime()["commit"], runtime=runtime(), selection="Training val selected best; no test model/threshold selection",
                  precision_recall_policy="Per-model maximum-F1 point from native metric accumulator")
    output.mkdir(parents=True, exist_ok=False)
    try:
        init_seeds(EVAL["seed"], deterministic=True)
        class CorrectedValidator(RTDETRValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                require(not self.training and not self.args.half, "Independent FP32 evaluation required")
                require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()),
                        "Effective evaluation protocol changed")
                report["expected_images"] = len(self.dataloader.dataset.im_files)

            def postprocess(self, predictions):
                return postprocess(predictions, self.args.imgsz, self.args.conf)

            def finalize_metrics(self):
                require(self.seen == report["expected_images"], "Incomplete independent evaluation split")
                report["images"] = self.seen
                return super().finalize_metrics()

        model = RTDETR(str(checkpoint))
        verify_model(model.model, variant, zero=False)
        require(model.model.model[-1].nc == 1, "Expected nc=1 trained checkpoint")
        metrics = model.val(validator=CorrectedValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Unexpected/nonfinite full AP table")
        require(report.get("images") == report["expected_images"], "Incomplete evaluation split")
        require(sha256(checkpoint) == checkpoint_hash and sha256(record["args"]["data"]) == report["data_sha256"],
                "Checkpoint/data changed during evaluation")
        report.update(status="PASSED", P=float(metrics.box.mp), R=float(metrics.box.mr), AP50=float(metrics.box.map50),
                      AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      iou_thresholds=[round(0.5 + i * 0.05, 2) for i in range(10)], AP_by_class=ap.tolist(),
                      AP_class_index=np.asarray(metrics.box.ap_class_index).tolist(),
                      speed_ms_per_image={k: float(v) for k, v in metrics.speed.items()})
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_csr_p3_v1")
    parser.add_argument("--device", default="0")
    arguments = parser.parse_args()
    evaluate(arguments.variant, arguments.split, arguments.device)
