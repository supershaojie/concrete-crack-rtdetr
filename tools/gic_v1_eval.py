"""One inference per split; all ordinary queries and exact metric inputs exported."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from gic_v1_common import ROOT, OUT, RUN, read_json, write_json, require, sha256, digest, now, binding, COUNTS
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou, smooth

PROTOCOL = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False, seed=42)


def evaluator_identity():
    return {p: sha256(ROOT / p) for p in (
        "tools/gic_v1_eval.py", "ultralytics-main/ultralytics/engine/validator.py",
        "ultralytics-main/ultralytics/models/rtdetr/val.py", "ultralytics-main/ultralytics/utils/metrics.py")}


def postprocess(predictions, imgsz, conf):
    """Mother c19_lif_v1_results.postprocess protocol, retaining original IDs.

    Sort first, apply the mask to sorted scores. No NMS, top-k or new threshold.
    """
    raw = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    results = []
    for values in raw:
        boxes = ops.xywh2xyxy(values[:, :4] * imgsz)
        score, classes = values[:, 4:].max(-1)
        order = score.argsort(descending=True)
        kept = order[score[order] > conf]
        results.append({"bboxes": boxes[kept], "conf": score[kept], "cls": classes[kept], "query_ids": kept,
                        "all_boxes": boxes, "all_scores": score, "all_classes": classes})
    return results


def artifact_record(folder):
    return {p.relative_to(folder).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name not in ("success.json", "metrics.json")}


def verified_success(folder, identity=None, split=None):
    record = read_json(Path(folder) / "metrics.json")
    index = read_json(Path(folder) / "success.json")
    if index is not None and index != record:
        return None
    if not record or record.get("status") != "COMPLETE" or record.get("exit_code") != 0:
        return None
    if identity is not None and record.get("identity") != identity:
        return None
    if split is not None and record.get("split") != split:
        return None
    for relative, info in record.get("artifacts", {}).items():
        file = Path(folder) / relative
        if not file.is_file() or sha256(file) != info["sha256"] or file.stat().st_size != info["bytes"]:
            return None
    required = {"queries_gt.jsonl.gz", "raw_stats.npz", "curves.npz"}
    if not required <= set(record.get("artifacts", {})):
        return None
    return record


def evaluate(split):
    require(split in ("val", "test"), "Expected independent val/test split")
    current = binding()
    complete = read_json(OUT / "training_completed.json")
    require(complete and complete["binding"] == current, "This identity has no completed training; interrupted runs require resume")
    best = RUN / "weights/best.pt"
    best_hash = sha256(best)
    require(complete["weights"]["best"]["sha256"] == best_hash, "Selected best changed after completion")
    identity = {"binding": current, "best_sha256": best_hash, "protocol": PROTOCOL, "settings": EVAL,
                "evaluator": evaluator_identity()}
    lock = read_json(OUT / "selection_lock.json")
    require(not lock or lock == identity, "Best/config already locked to a different identity")
    if split == "test":
        val = read_json(OUT / "evaluation_val.json", {})
        require(val and verified_success(val["folder"], identity, "val"), "Test requires complete verified independent val; use finish to restore missing indexes")
    prior = read_json(OUT / f"evaluation_{split}.json")
    candidates = [Path(prior["folder"])] if prior and "folder" in prior else []
    candidates += sorted((OUT / "evaluations" / split).glob("*"), reverse=True)
    for folder in dict.fromkeys(candidates):
        reused = verified_success(folder, identity, split)
        if reused:
            if not lock:
                write_json(OUT / "selection_lock.json", identity)
            write_json(folder / "success.json", reused)  # reconstruct an absent index from verified metrics
            write_json(OUT / f"evaluation_{split}.json", reused)
            print(f"REUSED {split}: {folder}", flush=True)
            return reused
    for folder in candidates:
        record = read_json(folder / "success.json") or read_json(folder / "metrics.json")
        require(not record or record.get("status") != "COMPLETE",
                f"{split}: successful evaluation exists but identity/export is incomplete or altered: {folder}. "
                "Preserved; automatic full-split reinference is forbidden. Inspect/restore the missing artifacts.")
    if not lock:
        write_json(OUT / "selection_lock.json", identity)
    plan = read_json(OUT / "prepare.json")
    snapshot = read_json(OUT / "data_snapshot.json")
    folder = OUT / "evaluations" / split / now()
    folder.mkdir(parents=True, exist_ok=False)
    record = {"status": "RUNNING", "split": split, "identity": identity, "folder": str(folder), "best": str(best),
              "best_sha256": best_hash, "best_epoch": complete["best_epoch"], "selection": "complete training val mAP50-95",
              "precision_recall_policy": "native smoothed maximum-F1 operating point for this split; test point is report-only",
              "mother_PR_reference": "historical same-report reference, not the val-fixed threshold comparison"}
    seen = set()
    gt_counts = []
    snapshot_rows = {r["image"]: r for r in snapshot["records"] if r["split"] == split}
    stream_path = folder / "queries_gt.jsonl.gz"
    try:
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class ExportValidator(RTDETRValidator):
                def init_metrics(self, model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half, "Independent evaluation must be FP32")
                    require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()), "Effective eval protocol changed")
                    record["actual_settings"] = vars(self.args).copy()

                def postprocess(self, predictions):
                    return postprocess(predictions, self.args.imgsz, self.args.conf)

                def update_metrics(self, predictions, batch):
                    for i, prediction in enumerate(predictions):
                        gt = self._prepare_batch(i, batch)
                        path = Path(gt["im_file"]).resolve().relative_to(Path(snapshot["root"])).as_posix()
                        require(path not in seen, f"Duplicate evaluation image: {path}")
                        require(path in snapshot_rows and len(gt["cls"]) == snapshot_rows[path]["boxes"],
                                f"Evaluation GT count differs from prepared parser snapshot: {path}")
                        require(bool(torch.isfinite(gt["bboxes"]).all() and (gt["cls"] == 0).all()), f"Invalid evaluation GT: {path}")
                        seen.add(path)
                        gt_counts.append(len(gt["cls"]))
                        boxes = prediction.pop("all_boxes")
                        scores = prediction.pop("all_scores")
                        classes = prediction.pop("all_classes")
                        query_ids = prediction.pop("query_ids")
                        require(len(scores) == 300 and torch.isfinite(boxes).all() and torch.isfinite(scores).all(), "Incomplete/nonfinite ordinary query export")
                        # The exact AP match matrix used by the inherited metric implementation.
                        tp = self._process_batch(prediction, gt)["tp"]
                        row = {"image_id": path, "path": str(Path(gt["im_file"]).resolve()), "original_size_hw": list(gt["ori_shape"]),
                               "input_size_hw": list(gt["imgsz"]), "preprocess": "RT-DETR stretch; BGR->RGB; /255",
                               "box_space": "input_image_pixels_xyxy", "inverse_scale_xy": [gt["ori_shape"][1] / gt["imgsz"][1], gt["ori_shape"][0] / gt["imgsz"][0]],
                               "query_ids": list(range(len(scores))), "boxes": boxes.cpu().tolist(), "scores": scores.cpu().tolist(), "classes": classes.cpu().tolist(),
                               "gt_boxes": gt["bboxes"].cpu().tolist(), "gt_classes": gt["cls"].cpu().tolist(),
                               "metric_query_ids": query_ids.cpu().tolist(), "metric_tp_iou50_to95": tp.tolist()}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    return super().update_metrics(predictions, batch)

                def get_stats(self):
                    np.savez_compressed(folder / "raw_stats.npz", **{k: np.concatenate(v, 0) for k, v in self.metrics.stats.items()})
                    return super().get_stats()

            wrapper = RTDETR(str(best))
            require(wrapper.model.model[-1].nc == 1, "nc mismatch")
            settings = dict(plan["args"])
            settings.update(EVAL)
            settings.update(model=str(best), data=plan["data_yaml"], split=split, device="0", mode="val", resume=False,
                            plots=True, save_json=False, save_txt=False, project=str(folder), name="plots",
                            save_dir=str(folder / "plots"), exist_ok=False)
            record["requested_settings"] = settings
            metrics = wrapper.val(validator=ExportValidator, **settings)
        require(set(seen) == set(snapshot["splits"][split]["image_ids"]), "Incomplete/wrong evaluation image list")
        require((len(seen), sum(gt_counts)) == COUNTS[split], "Evaluation image/GT counts differ from fixed mother split")
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "AP shape/nonfinite error")
        np.savez_compressed(folder / "curves.npz", ap=ap, p=metrics.box.p_curve, r=metrics.box.r_curve,
                            f1=metrics.box.f1_curve, confidence=metrics.box.px, pr_precision=metrics.box.prec_values)
        threshold = float(metrics.box.px[smooth(metrics.box.f1_curve.mean(0), .1).argmax()])
        record.update(status="COMPLETE", exit_code=0, images=len(seen), ground_truths=sum(gt_counts), image_ids_sha256=digest(sorted(seen)),
            precision=float(metrics.box.mp), recall=float(metrics.box.mr),
            F1=float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)) if metrics.box.mp + metrics.box.mr else 0.,
            AP50=float(metrics.box.map50), AP75=float(metrics.box.map75), mAP50_95=float(metrics.box.map),
            ap_by_iou=ap.tolist(), native_max_f1_threshold=threshold,
            # Unrounded archived JSON values, not numbers reconstructed from screenshots.
            mother_delta_pp=100 * (float(metrics.box.map) - {"val": .5245427221919051, "test": .5220090191444802}[split]))
        require(sha256(best) == best_hash, "Checkpoint changed during evaluation")
        record["artifacts"] = artifact_record(folder)
        record["percent"] = {k: 100 * record[k] for k in ("precision", "recall", "F1", "AP50", "AP75", "mAP50_95")}
        record["class_names"] = {"0": "crack", "all": "crack (single class)"}
        write_json(folder / "metrics.json", record)
        write_json(folder / "success.json", record)
        write_json(OUT / f"evaluation_{split}.json", record)
        return record
    except BaseException as error:
        record.update(status="FAILED", exit_code=1, error=repr(error), completed_images=sorted(seen))
        write_json(folder / "metrics.json", record)
        raise


def threshold_counts(path, threshold):
    """Re-match freshly filtered query sets at IoU=.5; never truncate stale TP arrays."""
    matcher = SimpleNamespace(iouv=torch.tensor([.5]))
    counts = {"TP": 0, "FP": 0, "FN": 0, "images": 0}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            scores = torch.tensor(row["scores"], dtype=torch.float32)
            order = scores.argsort(descending=True)
            selected = order[scores[order] > threshold]
            boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)[selected]
            gt = torch.tensor(row["gt_boxes"], dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(row["classes"])[selected]
            gt_classes = torch.tensor(row["gt_classes"])
            matches = BaseValidator.match_predictions(matcher, classes, gt_classes, box_iou(gt, boxes))
            tp = int(matches.sum())
            counts["TP"] += tp
            counts["FP"] += len(boxes) - tp
            counts["FN"] += len(gt) - tp
            counts["images"] += 1
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    return dict(counts, confidence=threshold, iou=.5, precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0., F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def offline_analysis():
    records = {s: read_json(OUT / f"evaluation_{s}.json") for s in ("val", "test")}
    require(all(r and verified_success(r["folder"], r["identity"]) for r in records.values()), "Need complete verified val/test exports")
    require(records["val"]["identity"] == records["test"]["identity"], "Val/test identities differ")
    source = {s: sha256(Path(r["folder"]) / "queries_gt.jsonl.gz") for s, r in records.items()}
    prior = read_json(OUT / "threshold_analysis.json")
    threshold = records["val"]["native_max_f1_threshold"]
    if (prior and prior.get("status") == "COMPLETE" and prior.get("source_sha256") == source
            and prior.get("threshold") == threshold and prior.get("protocol") == PROTOCOL
            and prior.get("best_sha256") == records["val"]["best_sha256"]):
        print("REUSED offline threshold analysis", flush=True)
        return prior
    report = {"status": "COMPLETE", "source_sha256": source, "best_sha256": records["val"]["best_sha256"],
              "selection": "val-only native smoothed maximum-F1 curve threshold; test never selects threshold",
              "threshold": threshold, "protocol": PROTOCOL, "iou": .5,
              "counts": {s: threshold_counts(Path(r["folder"]) / "queries_gt.jsonl.gz", threshold) for s, r in records.items()}}
    write_json(OUT / "threshold_analysis.json", report)
    return report
