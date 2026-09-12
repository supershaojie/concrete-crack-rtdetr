"""C24+LIF evaluator, selectively reused from successful LIF 0e95bba. Same corrected confidence mask; no lifecycle/package copy."""
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
from init_c24_lif_v1 import ROOT, require, runtime, sha256, verify_model, write_json, SOURCE_SHA256
from train_c24_lif_v1 import MAIN, paths, run_state, verify_delivery
from ultralytics.utils import ops
from ultralytics import RTDETR
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML
from ultralytics.data.utils import check_det_dataset

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7, max_det=300, augment=False, rect=False)


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


def evaluate(weights, data, split, output, device="0", val_report=None, evidence_scope="full_split"):
    weights, data, output = Path(weights).resolve(), Path(data).resolve(), Path(output).resolve()
    require(split in ("val", "test"), "Expected val/test")
    require(weights.is_file() and data.is_file(), "Missing checkpoint/data config")
    require(not output.exists(), f"Preserve previous evaluation: {output}")
    settings = dict(EVAL, data=str(data), split=split, device=device, plots=True, save_json=False, save_txt=False,
                    project=str(output), name="plots", exist_ok=False)
    digest, data_digest, info = sha256(weights), sha256(data), runtime()
    if split == "test":
        require(val_report and Path(val_report).is_file(), "Test requires completed val of the selected checkpoint")
        prior = json.loads(Path(val_report).read_text(encoding="utf-8"))
        require(prior["evidence_scope"] == evidence_scope, "Val/test scope differs")
        require(prior["status"] == "completed" and prior["split"] == "val" and prior["checkpoint_sha256"] == digest,
                "Test checkpoint differs from validated selection")
        require(prior["data_sha256"] == data_digest and prior["policy"] == POLICY, "Val/test data/policy changed")
        require(prior["runtime"]["commit"] == info["commit"], "Source changed after validation; validate fixed source/checkpoint again")
        require(all(type(prior["settings"][k]) is type(v) and prior["settings"][k] == v for k, v in EVAL.items()),
                "Val/test settings changed")
    resolved = check_det_dataset(str(data), autodownload=False)
    require(resolved["nc"] == 1, "Expected crack nc=1")
    output.mkdir(parents=True, exist_ok=False)
    seen, counts = set(), dict(images=0, predictions=0, metric_predictions=0, ground_truth=0, historical_mask_affected_images=0)
    report = dict(status="failed", evidence_scope=evidence_scope, runtime=info, split=split, settings=settings, checkpoint=str(weights),
                  checkpoint_sha256=digest, data_sha256=data_digest, data_config=YAML.load(data), policy=POLICY,
                  boxes="native final C2 Decoder layer", selection="Training val selects best; freeze checkpoint/config before independent test",
                  full_server_preflight="PASSED; checked in fixed launch report")
    stream_path = output / "predictions_gt.jsonl.gz"
    try:
        model = RTDETR(str(weights))
        verify_model(model.model)
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


if __name__ == '__main__':
    from c24_lif_v1_common import read_json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('val','test'))
    a=parser.parse_args();torch.set_num_threads(4);verify_delivery();p=paths()
    require(run_state()=='SUCCESS','Completed successful training required')
    plan=read_json(p['launch']/'plan.json');require(plan and plan['runtime']['commit']==runtime()['commit'],'Training source changed')
    require(sha256(p['data'])==plan['data_sha256'],'Data changed since training')
    evaluate(p['run']/'weights/best.pt',p['data'],a.mode,p['launch']/('evaluation_'+a.mode),device='0',
             val_report=p['launch']/'evaluation_val/metrics.json' if a.mode=='test' else None)
