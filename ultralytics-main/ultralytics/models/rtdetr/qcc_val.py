"""Corrected native evaluation and atomic, first-pass, all-query + GT exports."""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops, __version__
from ultralytics.utils.metrics import smooth
from .qcc_io import write_json, sha256

POLICY = "corrected_sorted_conf_mask_v1"
SCHEMA = "qcc.all_queries_gt.v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=0.001, iou=0.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    # Same corrected policy as a0459d6 tools/c19_lif_v1_results.py / ef9cb7e arg_val.
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    result = []
    for pred in raw:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        score, cls = pred[:, 4:].max(-1)
        order = score.argsort(descending=True)
        rows = torch.cat((boxes, score[:, None], cls[:, None]), -1)[order]
        rows = rows[rows[:, 4] > conf]
        result.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return result


class OneBatch:
    def __init__(self, loader):
        self.loader, self.dataset, self.batch_size = loader, loader.dataset, loader.batch_size
    def __len__(self):
        return 1
    def __iter__(self):
        yield next(iter(self.loader))


class QCCValidator(RTDETRValidator):
    """Independent FP32 evaluations only; epoch val/best selection stays native."""
    one_batch = False
    export_path = None
    export_identity = None
    expected_images = None
    expected_gt = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.mode = "val"

    def get_dataloader(self, dataset_path, batch_size):
        loader = super().get_dataloader(dataset_path, batch_size)
        return OneBatch(loader) if self.one_batch else loader

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.training:
            raise RuntimeError("QCCValidator is independent only; use native validator during training")
        self.actual_settings = vars(self.args).copy()
        self.qcc_seen, self.qcc_gt_count = set(), 0
        self.qcc_raw = self.qcc_logits = None
        if not self.one_batch and not self.export_path:
            raise RuntimeError("Formal QCC evaluation requires its first-pass export path")
        if self.export_path:
            self.export_path = Path(self.export_path)
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            self.partial = self.export_path.with_name(self.export_path.name + ".partial")
            if self.export_path.exists():
                raise FileExistsError("Completed export preserved; reuse the lock")
            self.qcc_stream = gzip.open(self.partial, "xt", encoding="utf-8")

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if raw.shape[1:] != (300, 5) or not torch.isfinite(raw).all():
            raise ValueError("Expected 300 finite ordinary single-class queries")
        if self.export_path:
            self.qcc_raw = raw.detach().float().cpu().clone()
            if isinstance(preds, (list, tuple)) and len(preds) > 1 and isinstance(preds[1], (list, tuple)):
                self.qcc_logits = preds[1][1][-1].detach().float().cpu()
                if self.qcc_logits.shape != raw[:, :, 4:].shape or not torch.isfinite(self.qcc_logits).all():
                    raise ValueError("Raw final ordinary logits do not match exported queries")
                if not torch.allclose(self.qcc_logits.sigmoid(), self.qcc_raw[:, :, 4:], atol=1e-7, rtol=1e-6):
                    raise ValueError("Export logits/sigmoid scores disagree")
        return postprocess(preds, self.args.imgsz, self.args.conf)

    def update_metrics(self, preds, batch):
        for i in range(len(preds)):
            gt = self._prepare_batch(i, batch)
            key = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
            if key in self.qcc_seen:
                raise ValueError(f"Duplicate evaluation image: {key}")
            self.qcc_seen.add(key)
            self.qcc_gt_count += len(gt["cls"])
            if self.export_path:
                h, w = map(int, gt["ori_shape"])
                ih, iw = gt["imgsz"]
                raw = self.qcc_raw[i]
                boxes = ops.xywh2xyxy(raw[:, :4]) * torch.tensor([w, h, w, h])
                scores, classes = raw[:, 4:].max(-1)
                gt_boxes = gt["bboxes"].detach().float().cpu() * torch.tensor([w / iw, h / ih, w / iw, h / ih])
                row = dict(schema=SCHEMA, identity=self.export_identity, image_id=key, image=key, split=self.args.split,
                    original_size_hw=[h, w], input_size_hw=[ih, iw], query_indices=list(range(len(raw))),
                    boxes_normalized_cxcywh=raw[:, :4].tolist(), boxes_original_xyxy=boxes.tolist(),
                    scores=scores.tolist(), classes=classes.tolist(), class_scores=raw[:, 4:].tolist(),
                    logits=self.qcc_logits[i].tolist() if self.qcc_logits is not None else None,
                    gt_boxes_original_xyxy=gt_boxes.tolist(), gt_classes=gt["cls"].detach().cpu().tolist(),
                    gt_boxes_normalized_cxcywh=batch["bboxes"][batch["batch_idx"] == i].detach().cpu().tolist())
                self.qcc_stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.qcc_raw = self.qcc_logits = None
        return super().update_metrics(preds, batch)

    def get_stats(self):
        # Preserve evaluator assignments/confidences before the native clearing step.
        # Fixed-val-threshold analysis later is pure offline work with these arrays.
        stats = self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        if self.export_path:
            np.savez_compressed(self.export_path.parent / "evaluator_stats.npz", **stats)
        self.metrics.clear_stats()
        box = self.metrics.box
        ap = np.asarray(box.all_ap)
        if ap.shape != (1, 10) or not np.isfinite(ap).all():
            raise ValueError("Expected finite nc=1 ten-IoU AP metrics")
        threshold_index = int(smooth(box.f1_curve.mean(0), 0.1).argmax())
        nt = int(self.metrics.nt_per_class[0])
        tp = int(round(float(box.r[0]) * nt))
        fp = int(round(tp / (float(box.p[0]) + 1e-16) - tp))
        self.qcc_metrics = dict(Precision=float(box.mp), Recall=float(box.mr), F1=float(np.mean(box.f1)),
            AP50=float(box.map50), AP75=float(box.map75), mAP50_95=float(box.map),
            ap_iou_thresholds=[round(0.5 + 0.05 * i, 2) for i in range(10)], ap_by_class=ap.tolist(),
            class_average="single class crack (macro mean equals that class)",
            reported_workpoint=dict(rule="native split-specific maximum smoothed mean F1 on 1000 confidence samples",
                confidence=float(box.px[threshold_index]), iou=0.5, TP=tp, FP=fp, FN=nt-tp,
                count_note="native rounded interpolated P/R-derived counts, not exact integer threshold counts"),
            postprocess=dict(policy=POLICY, conf=self.args.conf, max_det=self.args.max_det,
                iou_argument=self.args.iou, iou_argument_used=False, NMS=False,
                note="All 300 queries sorted, then conf mask on the SAME order; no extra max_det truncation in native RT-DETR"))
        if self.export_path:
            write_json(self.export_path.parent / "curves.json", {k: np.asarray(getattr(box, k)).tolist() for k in
                       ("p_curve", "r_curve", "f1_curve", "px", "prec_values")})
        return self.metrics.results_dict

    def finalize_metrics(self):
        if not self.one_batch and (len(self.qcc_seen) != len(self.dataloader.dataset)
                or (self.expected_images is not None and len(self.qcc_seen) != self.expected_images)
                or (self.expected_gt is not None and self.qcc_gt_count != self.expected_gt)):
            raise RuntimeError("Evaluation/export did not cover the complete fixed split")
        super().finalize_metrics()
        if self.export_path:
            self.qcc_metrics["confusion_matrix"] = dict(matrix=self.confusion_matrix.matrix.tolist() if self.args.plots else None,
                computed=bool(self.args.plots),
                confidence=0.25 if self.args.conf in (None, 0.001) else self.args.conf, iou=0.45,
                note="Native confusion-matrix working point; different from AP/P-R working point")

    def __call__(self, *args, **kwargs):
        success = False
        try:
            result = super().__call__(*args, **kwargs)
            if self.export_path:
                self.qcc_stream.close()
                os.replace(self.partial, self.export_path)
                write_json(self.export_path.with_suffix(self.export_path.suffix + ".complete.json"), dict(
                    status="PASS", schema=SCHEMA, identity=self.export_identity, images=len(self.qcc_seen), gt=self.qcc_gt_count,
                    predictions_sha256=sha256(self.export_path), actual_settings=self.actual_settings,
                    evaluator_version=__version__, metrics=self.qcc_metrics, scope="one_batch_diagnostic" if self.one_batch else "full_split",
                    coordinates="Raw CBR cxcywh normalized to stretch-resized image; xyxy multiplied independently by original W/H; no clipping, filtering, sorting or rounding in raw export. GT from actual val batch. JSON retains FP32 values.",
                    assignment_note="QCC GT ownership/Hungarian is distinct from AP TP matching."))
            success = True
            return result
        finally:
            if hasattr(self, "qcc_stream"):
                self.qcc_stream.close()
            if not success and self.export_path:
                write_json(Path(str(self.export_path) + ".failure.json"), dict(status="INCOMPLETE", partial=str(getattr(self, "partial", ""))))
