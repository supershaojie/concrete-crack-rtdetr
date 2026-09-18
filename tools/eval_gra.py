"""Explicit independent val/test and optional fixed-precision val diagnosis; never trains."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from init_gra import ROOT, VARIANTS, require, runtime, sha256, write_json, verify_model
from train_gra import source_manifest, inventory, read_json
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.patches import torch_load

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300,
            augment=False, rect=False, seed=42)


def postprocess(predictions, imgsz, conf):
    """Parent c19_lif_v1_results.py policy: mask sorted scores, without NMS."""
    predictions = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    result, affected = [], 0
    for pred in predictions:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        scores, classes = pred[:, 4:].max(-1)
        order = scores.argsort(descending=True)
        rows = torch.cat((boxes, scores[:, None], classes[:, None]), -1)[order]
        mask = rows[:, 4] > conf
        affected += int(not torch.equal(mask, scores > conf))
        rows = rows[mask]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result, affected


def evaluate(args):
    output = args.output.resolve()
    require(not output.exists(), "Existing evaluation output protected")
    require(args.weights.name == "best.pt" and sha256(args.weights) == args.weights_sha, "Freeze native selected best.pt SHA before evaluation")
    ckpt = torch_load(args.weights, map_location="cpu")
    require(ckpt["train_args"]["name"] == VARIANTS[args.variant][2], "Wrong experiment checkpoint")
    state_path = ROOT / "outputs/gra" / args.variant / "training/training_state.json"
    state = read_json(state_path)
    require(state["status"] in {"COMPLETED_200", "EARLY_STOPPED"} or
            (state["completed_epochs"] == 200 and state["final_eval"] == "FAILED"),
            "Training has not finished; do not evaluate/test an interim checkpoint")
    require(args.weights.resolve() == (Path(state["run"]) / "weights/best.pt").resolve(), "Weight is not native selected best from recorded run")
    identities = {"checkpoint_sha256": args.weights_sha, "data_sha256": sha256(args.data),
                  "dataset_inventory": inventory(args.data), "source_manifest": source_manifest()}
    if args.split == "test":
        require(args.val_report is not None, "Test requires completed independent val of same frozen best")
        val = read_json(args.val_report)
        require(val.get("status") == "COMPLETED" and val.get("split") == "val" and val.get("policy") == POLICY, "Invalid independent val report")
        require(all(val.get(k) == v for k, v in identities.items()), "Val/test weights/data/source identity differs")
        require(all(val["settings"][k] == v for k, v in EVAL.items()), "Val/test settings differ")
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(EVAL, data=str(args.data.resolve()), split=args.split, device=args.device, plots=True,
                    save_json=False, save_txt=False, project=str(output), name="plots", exist_ok=False)
    report = {"status": "FAILED", "split": args.split, "variant": args.variant, "policy": POLICY, "settings": settings,
              "evidence_scope": "full_split",
              "runtime": runtime(), "checkpoint": str(args.weights.resolve()), **identities,
              "precision_recall_policy": "each model own maximum-F1 working point",
              "selection": "native training best.pt, fixed weight SHA before independent val/test"}
    stream_path, seen = output / "predictions_gt.jsonl.gz", set()
    counts = dict(images=0, ground_truth=0, predictions=0, historical_mask_affected_images=0)
    resolved = check_det_dataset(str(args.data), autodownload=False)
    try:
        model = RTDETR(str(args.weights.resolve()))
        verify_model(model.model, args.variant)
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class Validator(RTDETRValidator):
                def init_metrics(self, model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half, "Independent FP32 evaluation required")
                    require(all(getattr(self.args, k) == v for k, v in EVAL.items()), "Effective eval settings changed")
                    self.expected = {Path(p).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                                     for p in self.dataloader.dataset.im_files}
                    require(len(self.expected) == len(self.dataloader.dataset.im_files), "Duplicate data paths")
                    report["split_paths_sha256"] = hashlib.sha256("\n".join(sorted(self.expected)).encode()).hexdigest()

                def postprocess(self, predictions):
                    rows, affected = postprocess(predictions, self.args.imgsz, self.args.conf)
                    counts["historical_mask_affected_images"] += affected
                    require(all(len(p["conf"]) <= self.args.max_det for p in rows), "More than max_det in fixed candidate pool")
                    return rows

                def update_metrics(self, predictions, batch):
                    for i, pred in enumerate(predictions):
                        gt = self._prepare_batch(i, batch)
                        image = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                        require(image not in seen, "Duplicate prediction image")
                        seen.add(image)
                        # Exactly the native one-to-one matcher, on the fixed score-floor candidate pool.
                        tp = self._process_batch(self.scale_preds(pred, gt), gt)["tp"][:, 0]
                        scale = torch.tensor([gt["ori_shape"][1] / gt["imgsz"][1], gt["ori_shape"][0] / gt["imgsz"][0]] * 2)
                        boxes, gt_boxes = pred["bboxes"].detach().float().cpu() * scale, gt["bboxes"].detach().float().cpu() * scale
                        scores, classes = pred["conf"].detach().cpu(), pred["cls"].detach().cpu()
                        require(all(torch.isfinite(t).all() for t in (boxes, gt_boxes, scores)), "Nonfinite eval output")
                        row = {"image": image, "box_format": "xyxy", "coordinate_space": "original_image_pixels",
                               "predictions": [{"bbox": b, "score": float(s), "class_id": int(c), "tp50_fixed_pool": bool(t)}
                                               for b, s, c, t in zip(boxes.tolist(), scores.tolist(), classes.tolist(), tp)],
                               "ground_truth": [{"bbox": b, "class_id": int(c)} for b, c in zip(gt_boxes.tolist(), gt["cls"].tolist())]}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                        counts["images"] += 1
                        counts["ground_truth"] += len(gt["cls"])
                        counts["predictions"] += len(scores)
                    return super().update_metrics(predictions, batch)

                def finalize_metrics(self):
                    require(seen == self.expected, "Incomplete split evaluation")
                    return super().finalize_metrics()

            metrics = model.val(validator=Validator, **settings)
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Invalid full AP array")
        require(sha256(args.weights) == args.weights_sha and sha256(args.data) == identities["data_sha256"], "Weights/data changed during eval")
        report.update(status="COMPLETED", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP50=float(metrics.box.map50), AP75=float(ap[:, 5].mean()), mAP50_95=float(metrics.box.map),
                      ap_by_class=ap.tolist(), ap_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)],
                      speed_ms_per_image=metrics.speed, predictions_gt_sha256=sha256(stream_path), export_complete=True, **counts)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_json(output / "metrics.json", report)
    return report


def fixed_precision(args):
    require(not args.output.exists(), "Existing fixed-P report protected")
    if not args.predictions.is_file() or not args.metrics.is_file():
        report = {"status": "PENDING", "reason": "same-protocol val predictions/identity missing; no inference from summary metrics"}
        write_json(args.output, report)
        return report
    metrics = read_json(args.metrics)
    require(metrics.get("split") == "val" and metrics.get("policy") == POLICY and
            metrics.get("status") in {"COMPLETED", "completed"}, "Fixed-P threshold search is val-only, same corrected protocol")
    require(metrics.get("evidence_scope") == "full_split" and metrics.get("export_complete") is True,
            "Fixed-P requires complete split predictions; bounded fixture/partial eval is not a result")
    require(metrics["settings"]["conf"] == .001 and metrics["settings"]["max_det"] == 300, "Candidate score floor/max_det differs")
    require(metrics.get("predictions_gt_sha256") == sha256(args.predictions), "Predictions differ from recorded metric pass")
    groups, total_gt, images, seen = {}, 0, 0, set()
    matcher = SimpleNamespace(iouv=torch.tensor([.50]))
    with gzip.open(args.predictions, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(row["image"] not in seen, "Duplicate fixed-P prediction image")
            seen.add(row["image"])
            pred = sorted((v for v in row["predictions"] if v["score"] > .001), key=lambda v: v["score"], reverse=True)[:300]
            gt = row["ground_truth"]
            total_gt += len(gt)
            images += 1
            if pred and all("tp50_fixed_pool" in v for v in pred):
                tp = [v["tp50_fixed_pool"] for v in pred]
            elif pred:
                # Historical parent export has full original-space boxes/GT, not TP flags.
                iou = box_iou(torch.tensor([v["bbox"] for v in gt], dtype=torch.float32).reshape(-1, 4),
                              torch.tensor([v["bbox"] for v in pred], dtype=torch.float32))
                tp = BaseValidator.match_predictions(matcher, torch.tensor([v["class_id"] for v in pred]),
                      torch.tensor([v["class_id"] for v in gt]), iou)[:, 0].tolist()
            else:
                tp = []
            for value, correct in zip(pred, tp):
                score = value["score"]
                require(np.isfinite(score), "Nonfinite original score")
                group = groups.setdefault(score, [0, 0])
                group[0] += int(correct)
                group[1] += 1 - int(correct)
    require(total_gt > 0 and images == metrics.get("images") and total_gt == metrics.get("ground_truth"), "Val export image/GT identity mismatch")
    require(hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest() == metrics.get("split_paths_sha256"), "Val export paths differ")
    true_positive = false_positive = 0
    best = None
    for threshold in sorted(groups, reverse=True):
        true_positive += groups[threshold][0]
        false_positive += groups[threshold][1]
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_gt
        candidate = {"threshold": threshold, "precision": precision, "recall": recall,
                     "TP": true_positive, "FP": false_positive, "FN": total_gt - true_positive}
        if precision >= args.precision_target and (best is None or
                (recall, precision, threshold) > (best["recall"], best["precision"], best["threshold"])):
            best = candidate
    report = {"status": "ACHIEVED" if best else "NOT_ACHIEVED", "working_point": best,
              "precision_target": args.precision_target, "matching_iou": .50, "formal_eval_iou": .7,
              "score_floor": .001, "max_det": 300, "GT": total_gt, "images": images,
              "threshold_operator": ">=; complete groups of equal original scores included",
              "tie_break": "maximum recall, then maximum precision, then highest original threshold",
              "matching": "native one-to-one, fixed floor candidate pool; no per-threshold rematching/interpolation",
              "checkpoint_sha256": metrics["checkpoint_sha256"], "data_sha256": metrics["data_sha256"],
              "split_paths_sha256": metrics.get("split_paths_sha256"), "predictions_sha256": sha256(args.predictions)}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--variant", choices=tuple(VARIANTS), default="cbr_lif_gra_v1")
    ev.add_argument("--split", choices=("val", "test"), required=True)
    ev.add_argument("--weights", type=Path, required=True)
    ev.add_argument("--weights-sha", required=True)
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--device", default="0")
    ev.add_argument("--val-report", type=Path)
    diag = sub.add_parser("fixed-p")
    diag.add_argument("--predictions", type=Path, required=True)
    diag.add_argument("--metrics", type=Path, required=True)
    diag.add_argument("--output", type=Path, required=True)
    diag.add_argument("--precision-target", type=float, default=.8656)
    args = p.parse_args()
    torch.set_num_threads(4)
    (evaluate if args.command == "evaluate" else fixed_precision)(args)


if __name__ == "__main__":
    main()
