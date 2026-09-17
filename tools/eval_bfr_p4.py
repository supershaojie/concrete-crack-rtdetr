"""Independent frozen-best FP32 val/test using corrected_sorted_conf_mask_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bfr_p4_lifecycle import (VARIANTS, paths, code_identity, data_identity, require, sha256,
                             write_json, read_json, verify_model, runtime)
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=0.001, iou=0.7,
            max_det=300, augment=False, rect=False, seed=42)


def corrected_postprocess(preds, imgsz, conf):
    """Same protocol as successful parent: sort first, mask the sorted confidences. No NMS."""
    tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
    require(torch.isfinite(tensor).all(), "Nonfinite final Decoder predictions")
    result = []
    for prediction in tensor:
        boxes = ops.xywh2xyxy(prediction[:, :4] * imgsz)
        scores, classes = prediction[:, 4:].max(-1)
        rows = torch.cat((boxes, scores[:, None], classes[:, None]), -1)[scores.argsort(descending=True)]
        rows = rows[rows[:, 4] > conf]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result


def evaluate(args):
    p = paths(args.variant)
    state = read_json(p["metadata"] / "training_state.json")
    require(state["status"] in {"COMPLETED_200", "EARLY_STOPPED"}, "Independent evaluation requires completed training")
    require(code_identity() == state["identity"]["code"], "Evaluation code differs from training")
    data = data_identity(args.data)
    require(data == state["identity"]["data"], "Evaluation data differs from training")
    require(args.checkpoint.resolve() == (p["run"] / "weights/best.pt").resolve(), "Freeze this run's val-selected best.pt")
    weight_sha = sha256(args.checkpoint)
    settings = dict(EVAL, data=str(args.data.resolve()), split=args.split, device="0", plots=True,
                    save_json=False, save_txt=False, project=str(args.output), name="plots", exist_ok=False)
    if args.split == "test":
        require(args.val_report is not None, "Test requires the independent val report for this identical best.pt")
        prior = read_json(args.val_report)
        require(prior["status"] == "PASSED" and prior["split"] == "val" and prior["variant"] == args.variant and
                prior["checkpoint_sha256"] == weight_sha and prior["data"] == data and prior["policy"] == POLICY and
                prior["code"] == state["identity"]["code"] and
                all(prior["settings"][k] == v for k, v in EVAL.items()), "Val/test frozen identity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", split=args.split, variant=args.variant, policy=POLICY, settings=settings,
                  code=state["identity"]["code"], runtime=runtime(), data=data, checkpoint_sha256=weight_sha,
                  selection="Training val selected best.pt; independent val then identical-weight test",
                  precision_recall_policy="Each model's own maximum-F1 working point",
                  test_history="Prior experiments' test results have already been seen; this candidate uses val selection")
    try:
        model = RTDETR(str(args.checkpoint))
        verify_model(model.model, args.variant, zero=False)
        require(model.model.model[-1].nc == 1, "Expected nc1 selected checkpoint")
        class CorrectedValidator(RTDETRValidator):
            def init_metrics(self, current_model):
                super().init_metrics(current_model)
                require(not self.training and all(getattr(self.args, k) == v for k, v in EVAL.items()),
                        "Effective independent FP32 evaluation settings changed")
                report["effective_settings"] = vars(self.args)
                expected = data["inventory"][args.split]
                files = sorted(Path(x).resolve().relative_to(Path(data["root"])).as_posix()
                               for x in self.dataloader.dataset.im_files)
                import hashlib
                require(len(files) == expected["images"] and
                        hashlib.sha256("\n".join(files).encode()).hexdigest() == expected["split_paths_sha256"],
                        "Evaluated image list differs from frozen split")
                report["images"] = len(files)
            def postprocess(self, preds):
                return corrected_postprocess(preds, self.args.imgsz, self.args.conf)
        metrics = model.val(validator=CorrectedValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Unexpected/nonfinite full AP array")
        require(sha256(args.checkpoint) == weight_sha and data_identity(args.data) == data,
                "Frozen checkpoint/data changed during evaluation")
        report.update(status="PASSED", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      iou_thresholds=[round(0.5 + i * 0.05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      ap_class_index=np.asarray(metrics.box.ap_class_index).tolist(), speed_ms_per_image=metrics.speed,
                      metrics_full_precision=metrics.results_dict, plots=str(args.output / "plots"))
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        write_json(args.output / "metrics.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "metrics.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_bfr_p4_v1")
    for flag in ("checkpoint", "data", "output", "val-report"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    p = paths(args.variant)
    args.checkpoint = args.checkpoint or p["run"] / "weights/best.pt"
    args.data = args.data or p["data"]
    args.output = args.output or p["metadata"] / ("evaluation_" + args.split)
    torch.set_num_threads(4)
    evaluate(args)


if __name__ == "__main__":
    main()
