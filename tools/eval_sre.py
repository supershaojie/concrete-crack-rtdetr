"""Independent val/test and val-only fixed-Precision diagnosis; never starts training."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.utils.metrics import ap_per_class
from init_c19_lif_v1 import require, runtime, sha256, verify_model as verify_parent
# The successful parent's corrected mask implementation is the authority. No NMS is added.
from c19_lif_v1_results import EVAL, POLICY, postprocess

DIAGNOSTIC = "sre_val_fixed_validator_labels_unique_scores_v1"
MATCHING = ("RTDETRValidator._process_batch -> BaseValidator.match_predictions(use_scipy=False); "
            "class-aware IoU >= threshold, IoU-sorted greedy unique prediction then unique GT; "
            "one fixed candidate pool, no rematching at score thresholds")
ARCHITECTURES = {
    "parent_cbr_lif": "rtdetr-resnet18-lite-cbr-lif-down.yaml",
    "parent_c2": "rtdetr-resnet18-lite.yaml",
    "cbr_lif_sre_v1": "rtdetr-resnet18-lite-cbr-lif-sre-v1.yaml",
    "sre_v1": "rtdetr-resnet18-lite-sre-v1.yaml",
}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def matcher_identity():
    paths = ["ultralytics-main/ultralytics/models/yolo/detect/val.py",
             "ultralytics-main/ultralytics/engine/validator.py",
             "tools/c19_lif_v1_results.py", "tools/eval_sre.py"]
    return {p: sha256(ROOT / p) for p in paths}


def verify_architecture(model, architecture):
    """Load the checkpoint's original architecture; never transplant parent weights into SRE."""
    expected = YAML.load(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr" / ARCHITECTURES[architecture])
    require(all(model.yaml[k] == expected[k] for k in ("backbone", "head", "scales")),
            "Checkpoint architecture differs from the explicitly selected architecture")
    require(len(model.model) == 27 and model.model[-1].nc == 1, "Expected native 27-node nc=1 RT-DETR")
    require(model.model[-1].num_queries == 300, "Expected original 300 queries")
    if architecture == "parent_cbr_lif":
        verify_parent(model)
    return dict(architecture=architecture, yaml=ARCHITECTURES[architecture],
                config_sha256=sha256(ROOT / "ultralytics-main/ultralytics/cfg/models/rt-detr" / ARCHITECTURES[architecture]))


def split_identity(dataset, root):
    """Fingerprint the actual validator image list and corresponding labels, without resplitting."""
    root = Path(root).resolve()
    images = [Path(p).resolve() for p in dataset.im_files]
    labels = [Path(p).resolve() for p in img2label_paths([str(p) for p in images])]
    pairs = sorted(zip(images, labels), key=lambda pair: pair[0].relative_to(root).as_posix())
    paths, label_rows, boxes = [], [], 0
    for image, label in pairs:
        paths.append(image.relative_to(root).as_posix())
        require(label.is_file(), f"Missing label: {label}")
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(all(len(r) == 5 and float(r[0]) == 0 and np.isfinite(np.asarray(r, float)).all() for r in rows),
                "Expected finite crack-only labels")
        boxes += len(rows)
        label_rows.append(label.relative_to(root).as_posix() + ":" + sha256(label))
    require(len(paths) == len(set(paths)), "Duplicate image in split")
    return dict(images=len(paths), boxes=boxes,
                split_paths_sha256=hashlib.sha256("\n".join(paths).encode()).hexdigest(),
                label_inventory_sha256=hashlib.sha256("\n".join(label_rows).encode()).hexdigest())


def evaluate(weights, data, split, output, architecture, device="0", val_report=None):
    weights, data, output = (Path(x).resolve() for x in (weights, data, output))
    require(weights.is_file() and data.is_file(), "Missing checkpoint or data YAML")
    require(not output.exists(), f"Preserving existing evaluation: {output}")
    digest, data_digest, info = sha256(weights), sha256(data), runtime()
    settings = dict(EVAL, data=str(data), split=split, device=device, plots=True, save_json=False,
                    save_txt=False, project=str(output), name="plots", exist_ok=False)
    if split == "test":
        require(val_report and Path(val_report).is_file(), "Test requires the same frozen checkpoint's completed val report")
        prior = json.loads(Path(val_report).read_text(encoding="utf-8"))
        require(prior["status"] == "completed" and prior["split"] == "val", "Val must finish before test")
        require(prior["checkpoint_sha256"] == digest and prior["data_sha256"] == data_digest,
                "Checkpoint/data changed after val")
        require(prior["runtime"]["commit"] == info["commit"] and prior["policy"] == POLICY,
                "Code or protocol changed after val")
        require(prior["architecture"]["architecture"] == architecture, "Val/test architecture mismatch")
        require(all(prior["settings"][k] == v and type(prior["settings"][k]) is type(v) for k, v in EVAL.items()),
                "Val/test evaluation settings mismatch")
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "Only original crack nc=1 dataset is supported")
    output.mkdir(parents=True, exist_ok=False)
    counts = dict(images=0, predictions=0, metric_predictions=0, ground_truth=0, historical_mask_affected_images=0)
    report = dict(status="failed", evidence_scope="full_split", runtime=info, split=split, settings=settings,
                  checkpoint=str(weights), checkpoint_sha256=digest, data_sha256=data_digest,
                  data_config=YAML.load(data), policy=POLICY, matching=MATCHING,
                  source_sha256=matcher_identity(), export_complete=False,
                  precision_recall_policy="each model own maximum-F1 working point; original smoothed/interpolated metrics",
                  selection="Original training-val best.pt selection; independent test never selects thresholds",
                  tp_label_origin="actual validator _process_batch return, captured during this evaluation")
    stream_path = output / "predictions_gt.jsonl.gz"
    seen = set()
    try:
        model = RTDETR(str(weights))
        report["architecture"] = verify_architecture(model.model, architecture)
        report["parameters_unfused"] = sum(p.numel() for p in model.model.parameters())
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class CapturingValidator(RTDETRValidator):
                def init_metrics(self, backend):
                    super().init_metrics(backend)
                    require(not self.training and not self.args.half, "Independent FP32 evaluation only")
                    require(all(getattr(self.args, k) == v and type(getattr(self.args, k)) is type(v)
                                for k, v in EVAL.items()), "Effective evaluation protocol changed")
                    report["actual_settings"] = vars(self.args).copy()
                    report["data_identity"] = split_identity(self.dataloader.dataset, self.data["path"])
                    frozen = json.loads((ROOT / "docs/sre/parent_dataset_inventory.json").read_text(encoding="utf-8"))[split]
                    require(report["data_identity"] == frozen,
                            "Evaluation image/label identity differs from the successful parent's frozen split")
                    report["expected_images"] = report["data_identity"]["images"]
                    report["split_paths_sha256"] = report["data_identity"]["split_paths_sha256"]

                def postprocess(self, preds):
                    selected, affected = postprocess(preds, self.args.imgsz, self.args.conf)
                    counts["historical_mask_affected_images"] += affected
                    require(all(len(p["conf"]) <= 300 for p in selected), "More than original 300 queries")
                    return selected

                def _process_batch(self, pred, batch):
                    result = super()._process_batch(pred, batch)
                    image = Path(batch["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                    require(image not in seen, "Duplicate evaluation image")
                    seen.add(image)
                    boxes, scores, classes = (pred[k].detach().float().cpu() for k in ("bboxes", "conf", "cls"))
                    gt_boxes, gt_cls = (batch[k].detach().float().cpu() for k in ("bboxes", "cls"))
                    require(all(torch.isfinite(t).all() for t in (boxes, scores, gt_boxes)), "Nonfinite exported values")
                    row = dict(image=image, box_format="xyxy", coordinate_space="validator_resized_pixels",
                               original_size_hw=list(map(int, batch["ori_shape"])), validator_size_hw=[640, 640],
                               tp_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)],
                               predictions=[dict(bbox=b, score=s, class_id=int(c), used_for_metrics=True, tp_iou=t)
                                            for b, s, c, t in zip(boxes.tolist(), scores.tolist(), classes.tolist(), result["tp"].tolist())],
                               ground_truth=[dict(bbox=b, class_id=int(c)) for b, c in zip(gt_boxes.tolist(), gt_cls.tolist())])
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    counts["images"] += 1
                    counts["predictions"] += len(scores)
                    counts["metric_predictions"] += len(scores)
                    counts["ground_truth"] += len(gt_cls)
                    return result

            metrics = model.val(validator=CapturingValidator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Unexpected/nonfinite AP")
        require(counts["images"] == report["expected_images"] and counts["ground_truth"] == report["data_identity"]["boxes"],
                "Incomplete split export or labels altered by loader")
        require(sha256(weights) == digest and sha256(data) == data_digest, "Weights/data changed during evaluation")
        report.update(status="completed", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP75=float(ap[:, 5].mean()), mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      ap_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      ap_class_index=np.asarray(metrics.box.ap_class_index).tolist(), speed_ms_per_image=metrics.speed,
                      predictions_gt_sha256=sha256(stream_path), export_complete=True,
                      confusion_matrix=np.asarray(metrics.confusion_matrix.matrix).tolist(),
                      curves=[dict(x=np.asarray(c[0]).tolist(), y=np.asarray(c[1]).tolist(), x_label=c[2], y_label=c[3])
                              for c in metrics.box.curves_results], **counts)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_new(output / "metrics.json", report)
    return report


def fixed_precision_recall(scores, tp, gt_count, precision_target=.8656):
    """Scan only realizable score>=threshold points; ties enter as complete groups."""
    scores = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(tp)
    require(scores.ndim == flags.ndim == 1 and len(scores) == len(flags), "Invalid score/TP shape")
    require(np.isfinite(scores).all() and np.isin(flags, [False, True, 0, 1]).all(), "Invalid scores/TP flags")
    require(isinstance(gt_count, int) and gt_count > 0 and 0 < precision_target <= 1, "GT/Precision target invalid")
    require(int(flags.sum()) <= gt_count, "More TP than GT; one-to-one contract violated")
    result = dict(status="NOT_ACHIEVED", R_at_P=None, threshold=None, precision=None, recall=None,
                  TP=None, FP=None, FN=None, selected_predictions=None, gt_count=gt_count,
                  predictions=len(scores), precision_target=precision_target,
                  threshold_rule="score >= original unique threshold, all ties included",
                  tie_break="maximum Recall, then highest Precision, then highest threshold",
                  empty_predictions="undefined Precision; never a successful working point")
    if not len(scores):
        return result
    order = np.argsort(-scores, kind="stable")
    scores, flags = scores[order], flags[order].astype(np.int64)
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    positives = np.cumsum(flags)[ends]
    totals = ends + 1
    precisions = positives / totals
    eligible = np.flatnonzero(precisions >= precision_target)
    result["realizable_nonempty_points"] = len(ends)
    if not len(eligible):
        return result
    best = max(eligible, key=lambda k: (int(positives[k]), float(precisions[k]), float(scores[ends[k]])))
    tp_count, total = int(positives[best]), int(totals[best])
    recall = tp_count / gt_count
    result.update(status="ACHIEVED", R_at_P=recall, threshold=float(scores[ends[best]]),
                  precision=float(precisions[best]), recall=recall, TP=tp_count, FP=total-tp_count,
                  FN=gt_count-tp_count, selected_predictions=total)
    return result


def original_matcher():
    """No new matching implementation or validator initialization side effects."""
    validator = object.__new__(RTDETRValidator)
    validator.iouv = torch.linspace(.5, .95, 10)
    validator.niou = 10
    return validator


def replay_row(row, validator):
    selected = [p for p in row["predictions"] if p["used_for_metrics"]]
    require(len(selected) <= 300 and all(p["class_id"] == 0 and p["score"] > .001 for p in selected),
            "Candidate pool differs from original conf>.001 / max_det=300 / class=0")
    require(all(p["class_id"] == 0 for p in row["ground_truth"]), "Non-crack GT")
    require(all(bool(p["used_for_metrics"]) == (p["score"] > .001) for p in row["predictions"]),
            "Historical confidence-mask disagreement")
    require(all(selected[i]["score"] >= selected[i+1]["score"] for i in range(len(selected)-1)),
            "Candidate order must match sorted original validator output")
    boxes = torch.tensor([p["bbox"] for p in selected], dtype=torch.float32).reshape(-1, 4)
    gt_boxes = torch.tensor([p["bbox"] for p in row["ground_truth"]], dtype=torch.float32).reshape(-1, 4)
    if row["coordinate_space"] == "original_image_pixels":
        # Original archive inverted RT-DETR's stretch. Undo that scaling before the original matcher.
        h, w = row["original_size_hw"]
        scale = torch.tensor([w/640, h/640, w/640, h/640], dtype=torch.float32)
        boxes, gt_boxes = boxes / scale, gt_boxes / scale
    else:
        require(row["coordinate_space"] == "validator_resized_pixels", "Unknown coordinate convention")
    require(torch.isfinite(boxes).all() and torch.isfinite(gt_boxes).all(), "Nonfinite archive boxes")
    pred = dict(bboxes=boxes, cls=torch.zeros(len(selected)))
    gt = dict(bboxes=gt_boxes, cls=torch.zeros(len(gt_boxes)))
    flags = validator._process_batch(pred, gt)["tp"]
    return selected, flags


def recall_from_archive(report_path, stream_path, output, inventory=None, role="candidate", weights=None):
    report_path, stream_path, output = map(Path, (report_path, stream_path, output))
    require(not output.exists(), "Preserving existing recall report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["status"] == "completed" and report["split"] == "val" and report["export_complete"],
            "Recall diagnosis is val-only and requires a complete original evaluation")
    require(report["policy"] == POLICY and report["evidence_scope"] == "full_split", "Evaluation protocol mismatch")
    require(all(report["settings"][k] == v and type(report["settings"][k]) is type(v) for k,v in EVAL.items()),
            "Evaluation settings differ from predeclared protocol")
    require(sha256(stream_path) == report["predictions_gt_sha256"], "Prediction/GT archive hash mismatch")
    if weights:
        require(sha256(weights) == report["checkpoint_sha256"], "Checkpoint hash mismatch")
    identity = report.get("data_identity")
    if inventory:
        saved = json.loads(Path(inventory).read_text(encoding="utf-8"))["val"]
        require(identity is None or identity == saved, "Dataset inventory differs")
        identity = saved
    require(identity and identity["split_paths_sha256"] == report["split_paths_sha256"], "Missing/mismatched val identity")
    validator, scores, all_tp, seen, gt_count = original_matcher(), [], [], set(), 0
    replayed = 0
    with gzip.open(stream_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(row["image"] not in seen, "Duplicate archive image")
            seen.add(row["image"])
            gt_count += len(row["ground_truth"])
            selected, replay_flags = replay_row(row, validator)
            if "tp_iou_thresholds" in row:
                require(row["tp_iou_thresholds"] == [round(.5+i*.05,2) for i in range(10)], "Captured IoU grid changed")
                flags = np.asarray([p["tp_iou"] for p in selected], dtype=bool).reshape(-1, 10)
                require(flags.shape == (len(selected), 10) and np.array_equal(flags, replay_flags),
                        "Captured validator TP labels disagree with original matcher")
            else:
                flags = replay_flags
                replayed += 1
            scores.extend(p["score"] for p in selected)
            all_tp.append(flags)
    require(len(seen) == report["images"] == identity["images"] == 1728, "Incomplete val image set")
    paths_hash = hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest()
    require(paths_hash == identity["split_paths_sha256"], "Val image-list identity mismatch")
    require(gt_count == report["ground_truth"] == identity["boxes"], "GT denominator mismatch")
    require(len(scores) == report["metric_predictions"], "Metric candidate count mismatch")
    flags = np.concatenate(all_tp, axis=0)
    # Audit reconstruction against ALL historical AP columns and maximum-F1 P/R.
    values = ap_per_class(flags, np.asarray(scores, dtype=np.float32), np.zeros(len(scores)), np.zeros(gt_count))
    derived = dict(precision=float(values[2][0]), recall=float(values[3][0]), mAP50=float(values[5][0,0]),
                   AP75=float(values[5][0,5]), mAP50_95=float(values[5].mean()), ap_by_class=values[5].tolist())
    errors = {k: float(np.max(np.abs(np.asarray(v) - np.asarray(report[k])))) for k,v in derived.items()}
    if replayed:
        # Old exports scale float32 boxes to original pixels, so reversing the stretch
        # need not recover every bit. This diagnostic is strictly at IoU=.50: demand
        # exact historical P/R/AP50, disclose every other AP difference, and do not
        # replace the historical official metrics by these reconstructed quantities.
        require(all(errors[k] == 0 for k in ("precision", "recall", "mAP50")),
                "Historical IoU=.50 reconstruction does not exactly match recorded metrics: " + str(errors))
    else:
        require(all(v == 0 for v in errors.values()), "Captured TP labels fail exact original-metric reconstruction: " + str(errors))
    result = fixed_precision_recall(scores, flags[:, 0], gt_count)
    result.update(schema=DIAGNOSTIC, role=role, split="val", matching_IoU=.50, score_floor=.001,
                  candidate_pool="original corrected validator conf > 0.001, at most 300 queries/image, nc=1; pretruncated",
                  matching=MATCHING, policy=POLICY, data_identity=identity,
                  checkpoint_sha256=report["checkpoint_sha256"], checkpoint=str(weights or report["checkpoint"]),
                  checkpoint_file_verified=bool(weights), evaluation_report=str(report_path.resolve()),
                  evaluation_report_sha256=sha256(report_path), predictions_gt=str(stream_path.resolve()),
                  predictions_gt_sha256=sha256(stream_path), runtime=runtime(), source_sha256=matcher_identity(),
                  original_metrics={k:report[k] for k in derived}, reconstructed_metrics=derived,
                  metric_reconstruction_max_abs_error=errors,
                  metric_reconstruction_policy=("P/R/AP50 required exactly equal; higher-IoU AP differences disclosed and not used by this IoU=.50 diagnostic"
                                                if replayed else "all original AP columns and P/R exactly equal"),
                  archive_roundtrip_caveat=("Historical boxes were exported in float32 original-image pixels. Reversing scale can change high-IoU boundary matches. This is archived-prediction reconstruction, not new model inference or a new official val run."
                                             if replayed else None),
                  tp_label_origin=("original validator matcher replayed once on archived fixed candidates; exact historical P/R/AP50 audited"
                                   if replayed else "actual validator TP flags captured during evaluation; audited by matcher replay"),
                  matching_replayed_images=replayed,
                  interpretation="fixed validator matching-label PR diagnostic; no threshold rematching, smoothing or interpolation",
                  comparison="Parent and candidate use their own val threshold; historical maximum-F1 Recall is not parent R_at_P")
    write_new(output, result)
    return result


def compare(parent_path, candidate_path, output):
    parent, candidate = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (parent_path, candidate_path))
    for report, role in ((parent, "parent"), (candidate, "candidate")):
        require(report["schema"] == DIAGNOSTIC and report["split"] == "val" and report["role"] == role,
                "Expected corresponding complete parent/candidate val diagnostic")
    for key in ("policy", "matching", "matching_IoU", "score_floor", "precision_target", "data_identity"):
        require(parent[key] == candidate[key], f"Parent/candidate protocol differs: {key}")
    achieved = all(r["status"] == "ACHIEVED" for r in (parent,candidate))
    result = dict(schema=DIAGNOSTIC, status="COMPARABLE" if achieved else "NOT_ACHIEVED", split="val",
                  precision_target=.8656, parent_checkpoint_sha256=parent["checkpoint_sha256"],
                  candidate_checkpoint_sha256=candidate["checkpoint_sha256"], data_identity=parent["data_identity"],
                  parent=parent, candidate=candidate,
                  delta_R_at_P=candidate["R_at_P"]-parent["R_at_P"] if achieved else None,
                  note="Each model has its own val score threshold. Also inspect original P/R/AP75/mAP and new false positives.")
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for split in ("val", "test"):
        p = sub.add_parser(split, help=f"Explicit independent {split}; no training")
        for key in ("weights", "data", "output"):
            p.add_argument("--" + key, type=Path, required=True)
        p.add_argument("--architecture", choices=ARCHITECTURES, required=True)
        p.add_argument("--device", default="0")
        if split == "test":
            p.add_argument("--val-report", type=Path, required=True)
    p = sub.add_parser("recall", help="Val-only fixed-Precision diagnosis from recorded predictions and GT")
    for key in ("report", "stream", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--inventory", type=Path, help="Historical launch dataset_inventory.json (required for old archives)")
    p.add_argument("--weights", type=Path, help="Optionally also verify actual checkpoint file SHA256")
    p.add_argument("--role", choices=("parent", "candidate"), default="candidate")
    p = sub.add_parser("compare", help="Compare parent/candidate val reports with identical protocol/data")
    for key in ("parent", "candidate", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("val", "test"):
        result = evaluate(args.weights, args.data, args.command, args.output, args.architecture, args.device,
                          getattr(args, "val_report", None))
    elif args.command == "recall":
        result = recall_from_archive(args.report, args.stream, args.output, args.inventory, args.role, args.weights)
    else:
        result = compare(args.parent, args.candidate, args.output)
    print(json.dumps({k:result[k] for k in ("status", "R_at_P", "threshold", "precision", "recall", "TP", "FP", "FN") if k in result}, indent=2))


if __name__ == "__main__":
    main()
