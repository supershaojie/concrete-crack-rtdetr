"""DPR independent val/test using the successful parent corrected final-box protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np
import torch
from init_dpr import ROOT, MAIN, VARIANTS, require, runtime, sha256, verify_model, write_json, SOURCE_SHA256
from dpr_data import dataset_identity
from ultralytics.utils import ops
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.data.utils import check_det_dataset

POLICY = "corrected_sorted_conf_mask_v1"
from dpr_acceptance import CONTRACT, scope as capability_scope, policy as acceptance_policy
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False, seed=42)


# Reused verbatim from C24 scca_results.py at beedcfa; no experimental-module import.
def postprocess(preds, imgsz, conf):
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


def image_record(pred, gt, dataset_root, conf):
    """One image, all 300 final FP32 predictions and all GT; no decimal rounding/clipping."""
    h, w = map(int, gt["ori_shape"])
    ih, iw = map(int, gt["imgsz"])
    scale = torch.tensor([w / iw, h / ih, w / iw, h / ih], dtype=torch.float32)
    boxes = pred["bboxes"].detach().float().cpu() * scale
    gt_boxes = gt["bboxes"].detach().float().cpu() * scale
    scores, classes = pred["conf"].detach().float().cpu(), pred["cls"].detach().cpu()
    require(all(torch.isfinite(t).all() for t in (boxes, gt_boxes, scores)), "Nonfinite prediction/GT export")
    rel = Path(gt["im_file"]).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
    return dict(image=rel, original_size_hw=[h, w], box_format="xyxy", coordinate_space="original_image_pixels",
                coordinates="continuous, no rounding or clipping; RT-DETR stretch inversion; class_id is zero-based",
                predictions=[dict(class_id=int(c), score=float(s), bbox=b, used_for_metrics=bool(s > conf))
                             for b, s, c in zip(boxes.tolist(), scores, classes)],
                ground_truth=[dict(class_id=int(c), bbox=b) for b, c in zip(gt_boxes.tolist(), gt["cls"].detach().cpu())])


def evaluate(weights, data, split, output, device="0", val_report=None, evidence_scope="full_split", variant=MAIN):
    weights, data, output = Path(weights).resolve(), Path(data).resolve(), Path(output).resolve()
    require(split in ("val", "test"), "Expected val/test")
    require(weights.is_file() and data.is_file(), "Missing checkpoint/data config")
    inventory = dataset_identity(data)
    require(evidence_scope == "full_split", "Formal independent full-split entry only")
    require(EVAL["half"] is False and capability_scope()["independent_evaluation"] == "FP32_ONLY", "R1 independent evaluation scope")
    require(not output.exists(), f"Preserve previous evaluation: {output}; use a new timestamp directory")
    settings = dict(EVAL, data=str(data), split=split, device=device, plots=True, save_json=False, save_txt=False,
                    project=str(output), name="plots", exist_ok=False)
    digest, data_digest, info = sha256(weights), sha256(data), runtime()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(), "Commit code before formal evaluation")
    if split == "test":
        require(val_report and Path(val_report).is_file(), "Test requires completed val of the selected checkpoint")
        prior = json.loads(Path(val_report).read_text(encoding="utf-8"))
        require(prior["evidence_scope"] == evidence_scope, "Val/test scope differs")
        require(prior["status"] == "completed" and prior["split"] == "val" and prior["checkpoint_sha256"] == digest,
                "Test checkpoint differs from validated selection")
        require(prior["data_sha256"] == data_digest and prior["policy"] == POLICY and prior["variant"] == variant and prior["dataset_inventory"] == inventory, "Val/test data/policy/variant changed")
        require(prior["runtime"]["commit"] == info["commit"], "Source changed after validation; validate fixed source/checkpoint again")
        require(all(type(prior["settings"][k]) is type(v) and prior["settings"][k] == v for k, v in EVAL.items()),
                "Val/test settings changed")
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "Expected crack nc=1")
    output.mkdir(parents=True, exist_ok=False)
    seen, counts = set(), dict(images=0, predictions=0, metric_predictions=0, ground_truth=0, historical_mask_affected_images=0)
    report = dict(status="failed", contract=CONTRACT, acceptance_policy=acceptance_policy(), capability_scope=capability_scope(), variant=variant, dataset_inventory=inventory, evidence_scope=evidence_scope, runtime=info, split=split, settings=settings, checkpoint=str(weights),
                  checkpoint_sha256=digest, data_sha256=data_digest, data_config=YAML.load(data), policy=POLICY,
                  boxes="original final Decoder layer; no extra NMS", precision_recall_policy="each model own maximum-F1 working point", selection="Training val selects best; freeze checkpoint/config before independent test",
                  full_server_preflight="bounded_required")
    stream_path = output / "predictions_gt.jsonl.gz"
    try:
        model = RTDETR(str(weights))
        verify_model(model.model, variant)
        require(model.model.model[-1].nc == 1, "Evaluation requires nc=1")
        report["parameters_unfused"] = sum(p.numel() for p in model.model.parameters())
        with gzip.open(stream_path, "xt", encoding="utf-8") as stream:
            class StreamingValidator(RTDETRValidator):
                def init_metrics(self, model):
                    super().init_metrics(model)
                    require(not self.training and not self.args.half, "Independent FP32 evaluation only")
                    require(all(type(getattr(self.args, k)) is type(v) and getattr(self.args, k) == v for k, v in EVAL.items()),
                            "Effective evaluation settings changed")
                    report["actual_settings"] = vars(self.args).copy()
                    files = [Path(p).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
                             for p in self.dataloader.dataset.im_files]
                    require(len(files) == len(set(files)), "Duplicate split image paths")
                    self.expected_images = set(files)
                    report["expected_images"] = len(files)
                    report["split_paths_sha256"] = hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()

                def postprocess(self, preds):
                    selected, affected = postprocess(preds, self.args.imgsz, self.args.conf)
                    full, _ = postprocess(preds, self.args.imgsz, -float("inf"))
                    counts["historical_mask_affected_images"] += affected
                    # Only the current batch travels to update_metrics, never a cross-batch activation cache.
                    for selected_image, all_queries in zip(selected, full):
                        require(len(all_queries["conf"]) == 300, "Incomplete regular-query export")
                        selected_image["_all_queries"] = all_queries
                    return selected

                def update_metrics(self, preds, batch):
                    for i, pred in enumerate(preds):
                        all_queries = pred.pop("_all_queries")
                        row = image_record(all_queries, self._prepare_batch(i, batch), self.data["path"], self.args.conf)
                        require(row["image"] not in seen, "Image exported more than once")
                        seen.add(row["image"])
                        counts["images"] += 1
                        counts["predictions"] += len(row["predictions"])
                        counts["metric_predictions"] += len(pred["conf"])
                        counts["ground_truth"] += len(row["ground_truth"])
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    return super().update_metrics(preds, batch)

                def finalize_metrics(self):
                    require(seen == self.expected_images, "Prediction/GT export does not cover the complete split")
                    return super().finalize_metrics()

            metrics = model.val(validator=StreamingValidator, **settings)
        require(counts["images"] == inventory[split]["images"] and counts["ground_truth"] == inventory[split]["boxes"], "Evaluation coverage differs from frozen parent identity")
        require(report["split_paths_sha256"] == inventory[split]["split_paths_sha256"], "Evaluation split list differs")
        ap = np.asarray(metrics.box.all_ap)
        require(ap.shape == (1, 10) and np.isfinite(ap).all(), "Nonfinite/unexpected AP array")
        require(sha256(weights) == digest and sha256(data) == data_digest, "Checkpoint/data changed during evaluation")
        report.update(status="completed", precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      AP75=float(ap[:, 5].mean()), mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      ap_iou_thresholds=[round(.5 + i * .05, 2) for i in range(10)], ap_by_class=ap.tolist(),
                      ap_class_index=np.asarray(metrics.box.ap_class_index).tolist(), speed_ms_per_image=metrics.speed,
                      predictions_gt_sha256=sha256(stream_path), export_complete=True, **counts)
    except BaseException as error:
        report.update(error=repr(error), **counts)
        raise
    finally:
        write_json(output / "metrics.json", report)
    return report



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('val', 'test'))
    parser.add_argument('--variant', choices=VARIANTS, default=MAIN)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--val-report', type=Path)
    parser.add_argument('--launch', type=Path, help='Completed training launch record; defaults to latest for variant')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output or ROOT / 'outputs/dpr' / args.variant / ('evaluation_' + args.split + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    require(args.weights.name == 'best.pt', 'Independent results require frozen native best.pt selection')
    metadata = ROOT / 'outputs/dpr' / args.variant
    launches = sorted(metadata.glob('launch_*.json'))
    launch_path = args.launch or (launches[-1] if launches else None)
    require(launch_path is not None and launch_path.is_file(), 'Missing training completion record')
    launch = json.loads(launch_path.read_text(encoding='utf-8'))
    planned = launch['recipe']
    require(launch.get('status') == 'COMPLETED' or launch.get('completed_epochs', 0) >= planned['epochs'],
            'Training is incomplete; do not use independent evaluation to disguise an unfinished run')
    require(launch.get('head') == runtime()['commit'] and planned['name'] == VARIANTS[args.variant][2],
            'Training source/run differs from evaluation')
    require(args.weights.resolve() == (Path(planned['save_dir']) / 'weights/best.pt').resolve(),
            'Weights are not the native selected best from this experiment run')
    from ultralytics.utils.patches import torch_load
    checkpoint = torch_load(args.weights, map_location='cpu')
    train_args = checkpoint.get('train_args', {})
    require(train_args.get('name') == VARIANTS[args.variant][2], 'Checkpoint run does not match requested variant')
    require(checkpoint.get('ema') is not None or checkpoint.get('model') is not None, 'Missing selected model')
    # Freeze once, before independent val. Retries use another evaluation directory
    # but cannot silently switch best weights/source/data after viewing results.
    selection = dict(variant=args.variant, checkpoint=str(args.weights.resolve()),
                     checkpoint_sha256=sha256(args.weights), head=launch['head'],
                     dataset_inventory=dataset_identity(args.data), policy=POLICY, settings=EVAL)
    selection_path = metadata / 'best_selection.json'
    if selection_path.exists():
        require(json.loads(selection_path.read_text(encoding='utf-8')) == selection, 'Frozen best selection changed')
    else:
        require(args.split == 'val', 'Independent val must freeze selection before test')
        metadata.mkdir(parents=True, exist_ok=True)
        with selection_path.open('x', encoding='utf-8') as stream:
            json.dump(selection, stream, ensure_ascii=False, indent=2)
    evaluate(args.weights, args.data, args.split, output, device=args.device,
             val_report=args.val_report, variant=args.variant)
    print(str(output / 'metrics.json'))


if __name__ == '__main__':
    main()
