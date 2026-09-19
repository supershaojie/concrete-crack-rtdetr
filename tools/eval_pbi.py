"""Explicit independent val/test with corrected_sorted_conf_mask_v1; never trains."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import torch

from pbi_common import ROOT, VARIANTS, require, sha256, write_json, runtime, verify_model
from train_pbi import code_identity, dataset_identity
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
PARENT_BEST_SHA256 = "24185828a44b79ea0709a45d3d8cd8780065533df9103f9317cc1bbb2f9710aa"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7,
            max_det=300, augment=False, rect=False, seed=42)


def corrected_postprocess(preds, imgsz, conf, max_det=300):
    """Same sorted-row mask as inspected c19_lif_v1_results.postprocess; no NMS."""
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    require(bool(torch.isfinite(preds).all()), "Nonfinite detector inference output")
    result, affected = [], 0
    for prediction in preds:
        boxes = ops.xywh2xyxy(prediction[:, :4] * imgsz)
        score, cls = prediction[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        affected += int(not torch.equal(mask, score > conf))
        rows = rows[mask][:max_det]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result, affected


def fixed_precision(scores, tp, gt_count, target=.8656):
    """Scan only attainable score >= threshold points; equal scores enter together.

    TP flags are native one-to-one IoU=.50 matches in the fixed score-floor/max_det
    candidate pool. They are NOT recomputed independently for each threshold.
    """
    scores, tp = np.asarray(scores, dtype=np.float64), np.asarray(tp, dtype=bool)
    require(scores.ndim == tp.ndim == 1 and len(scores) == len(tp) and
            np.isfinite(scores).all() and gt_count >= 0 and 0 < target <= 1, "Invalid fixed-P inputs")
    result = dict(status="NOT_ACHIEVED", precision_target=target, match_iou=.50, GT=int(gt_count),
                  threshold=None, precision=None, recall=None, TP=None, FP=None, FN=None,
                  threshold_rule="score >= threshold; original equal scores enter as one group",
                  matching="native fixed-candidate-pool one-to-one IoU=.50; not per-threshold rematching",
                  score_floor=EVAL["conf"], max_det=EVAL["max_det"])
    if not len(scores) or not gt_count:
        return result
    order = np.argsort(-scores, kind="stable")
    scores, tp = scores[order], tp[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true = np.cumsum(tp, dtype=np.int64)[ends]
    count = ends + 1
    precision, recall = true / count, true / gt_count
    valid = np.flatnonzero(precision >= target)
    if not len(valid):
        return result
    chosen = max(valid.tolist(), key=lambda i: (recall[i], precision[i], scores[ends[i]]))
    t, n = int(true[chosen]), int(count[chosen])
    result.update(status="ACHIEVED", threshold=float(scores[ends[chosen]]),
                  precision=float(precision[chosen]), recall=float(recall[chosen]),
                  TP=t, FP=n-t, FN=int(gt_count)-t, unique_thresholds=int(len(ends)))
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def evaluate(args):
    weights, data, out = args.weights.resolve(), args.data.resolve(), args.output.resolve()
    require(weights.is_file() and data.is_file(), "Missing fixed weights/data config")
    require(weights.name == "best.pt", "Use original training-rule selected best.pt; do not cherry-pick checkpoints")
    require(not (args.split == "test" and args.fixed_precision), "Threshold search is val-only; never optimize on test")
    digest, data_id = sha256(weights), dataset_identity(data)
    if args.parent_baseline:
        require(digest == PARENT_BEST_SHA256, "Parent best checkpoint SHA differs from the specified historical parent")
    if args.split == "test":
        require(args.val_report and args.val_report.is_file(), "Test requires completed val report of the same best.pt")
        prior = json.loads(args.val_report.read_text(encoding="utf-8"))
        if "report_path" in prior:
            prior_path = Path(prior["report_path"])
            require(prior_path.is_file() and sha256(prior_path) == prior.get("report_sha256"), "Val pointer changed")
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        require(prior.get("status") == "COMPLETED" and prior.get("split") == "val" and
                prior.get("weights_sha256") == digest and prior.get("policy") == POLICY and
                prior.get("settings") == EVAL and prior.get("dataset_identity") == data_id and
                prior.get("code_identity") == code_identity(), "Val/test fixed identity or protocol changed")
    pointer = out.parent / (out.name + "_latest.json")

    def publish(report_path):
        temporary = pointer.with_name(pointer.name + ".%d.tmp" % os.getpid())
        write_json(temporary, dict(report_path=str(report_path), report_sha256=sha256(report_path)))
        os.replace(temporary, pointer)

    if out.exists():
        previous_path = out / "metrics.json"
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if (previous.get("status") == "COMPLETED" and previous.get("split") == args.split and
                    previous.get("weights_sha256") == digest and previous.get("dataset_identity") == data_id and
                    previous.get("settings") == EVAL and previous.get("policy") == POLICY and
                    previous.get("code_identity") == code_identity() and
                    previous.get("variant") == ("parent_cbr_lif" if args.parent_baseline else args.variant) and
                    bool(previous.get("fixed_precision_requested")) == args.fixed_precision):
                artifacts = previous.get("artifacts", {})
                require(artifacts and all((out / name).is_file() and sha256(out / name) == digest
                        for name, digest in artifacts.items()), "Existing evaluation artifact changed/missing")
                publish(previous_path)
                return previous
        # Preserve failed/incompatible evaluations and choose a fresh directory.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = out.with_name(out.name + "_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", split=args.split, variant="parent_cbr_lif" if args.parent_baseline else args.variant,
                  weights=str(weights), weights_sha256=digest, dataset_identity=data_id, policy=POLICY, settings=EVAL,
                  runtime=runtime(), code_identity=code_identity(), selection="original native training best.pt rule",
                  output=str(out), fixed_precision_requested=args.fixed_precision,
                  training="NOT_RUN_BY_EVALUATOR", fixed_precision_diagnostic="NOT_REQUESTED",
                  parent_fixed_precision="PENDING unless the parent's raw native predictions are separately evaluated")
    counts = {"images": 0, "GT": 0, "predictions": 0, "historical_mask_affected_images": 0}
    seen = set()
    try:
        model = RTDETR(str(weights))
        if not args.parent_baseline:
            verify_model(model.model, args.variant, zero=False)
        require(model.model.model[-1].nc == 1, "Expected nc=1 trained checkpoint")

        class CorrectedValidator(RTDETRValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                require(not self.training and not self.args.half, "Independent FP32 evaluation required")
                require(all(getattr(self.args, k) == v for k, v in EVAL.items()), "Effective evaluation protocol changed")
                self.expected_images = {str(Path(p).resolve()) for p in self.dataloader.dataset.im_files}
                require(len(self.expected_images) == len(self.dataloader.dataset.im_files), "Duplicate split images")

            def postprocess(self, preds):
                rows, affected = corrected_postprocess(preds, self.args.imgsz, self.args.conf, self.args.max_det)
                counts["historical_mask_affected_images"] += affected
                return rows

            def update_metrics(self, preds, batch):
                for i, pred in enumerate(preds):
                    path = str(Path(batch["im_file"][i]).resolve())
                    require(path not in seen, "Image evaluated more than once")
                    seen.add(path)
                    counts["images"] += 1
                    counts["predictions"] += len(pred["conf"])
                    counts["GT"] += int((batch["batch_idx"] == i).sum())
                return super().update_metrics(preds, batch)

            def get_stats(self):
                # Take native flags before super.get_stats clears them; no alternate matcher.
                if args.fixed_precision:
                    stats = self.metrics.stats
                    scores = np.concatenate(stats["conf"]) if stats["conf"] else np.zeros(0)
                    tp = np.concatenate(stats["tp"])[:, 0] if stats["tp"] else np.zeros(0, dtype=bool)
                    gt = sum(len(a) for a in stats["target_cls"])
                    diagnostic = fixed_precision(scores, tp, gt)
                    diagnostic.update(weights_sha256=digest, data_identity=data_id)
                    report["fixed_precision_diagnostic"] = diagnostic
                    raw = out / "fixed_pool_scores_tp.npz"
                    np.savez_compressed(raw, scores=scores, tp50=tp, gt=np.asarray(gt))
                    report["fixed_pool"] = dict(path=str(raw), sha256=sha256(raw), excluded_from_light_pack=True)
                return super().get_stats()

            def finalize_metrics(self):
                require(seen == self.expected_images, "Evaluation did not cover the complete split")
                return super().finalize_metrics()

        metrics = model.val(validator=CorrectedValidator, **dict(EVAL, data=str(data), split=args.split,
                              device=args.device, plots=True, save_json=False, save_txt=False,
                              project=str(out), name="plots", exist_ok=False))
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid per-IoU AP")
        require(sha256(weights) == digest and dataset_identity(data) == data_id, "Fixed weights/data changed during evaluation")
        report.update(status="COMPLETED", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      ap_iou=[round(.50 + i*.05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      metrics=_plain(metrics.results_dict), speed_ms_per_image=_plain(metrics.speed), **counts)
        write_json(out / "curves.json", _plain(metrics.curves_results))
        report["artifacts"] = {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                               if p.is_file() and p.name != "metrics.json"}
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_json(out / "metrics.json", report)
    publish(out / "metrics.json")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("val", "test"))
    p.add_argument("--variant", choices=VARIANTS, default="cbr_lif_pbi_v1")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--val-report", type=Path)
    p.add_argument("--fixed-precision", action="store_true", help="Val-only P>=0.8656 attainable threshold diagnostic")
    p.add_argument("--parent-baseline", action="store_true", help="Same protocol, pinned historical parent best.pt SHA")
    args = p.parse_args()
    report = evaluate(args)
    print(json.dumps({k: report[k] for k in ("status", "weights_sha256", "precision", "recall", "AP50", "AP75", "mAP50_95")}, indent=2))


if __name__ == "__main__":
    main()
