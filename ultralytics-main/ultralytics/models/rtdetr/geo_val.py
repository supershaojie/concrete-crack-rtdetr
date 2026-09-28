"""Corrected FP32 evaluation, streaming every ordinary query and GT in the same pass.

Sorted confidence mask and original-pixel export follow ARG ef9cb7e / mother
c19_lif_v1_results.py. Native epoch validation remains RTDETRValidator.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=.001, iou=.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
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


class GEOValidator(RTDETRValidator):
    one_batch = False
    export_path = None
    export_identity = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.mode = "val"

    def get_dataloader(self, dataset_path, batch_size):
        loader = super().get_dataloader(dataset_path, batch_size)
        return OneBatch(loader) if self.one_batch else loader

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.training:
            raise RuntimeError("GEOValidator is only for independent FP32 evaluation")
        if self.nc != 1:
            raise ValueError("GEO v1 requires nc=1")
        self.actual_settings = vars(self.args).copy()
        self.geo_seen, self.geo_raw = set(), None
        if self.export_path is None:
            self.export_path = self.save_dir / f"{self.args.split}_predictions_gt.jsonl.gz"
        self.export_path = Path(self.export_path)
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self.geo_stream = gzip.open(self.export_path, "xt", encoding="utf-8")

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if not bool(torch.isfinite(raw).all()) or raw.shape[-1] != 5 or raw.shape[1] not in (0, 300):
            raise ValueError("Expected finite nc1 final 300 ordinary queries (empty output also supported)")
        self.geo_raw = raw.detach().float().cpu()
        return postprocess(preds, self.args.imgsz, self.args.conf)

    def update_metrics(self, preds, batch):
        for i in range(len(preds)):
            gt = self._prepare_batch(i, batch)
            key = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
            if key in self.geo_seen:
                raise ValueError(f"Duplicate evaluation image: {key}")
            self.geo_seen.add(key)
            h, w = map(int, gt["ori_shape"])
            ih, iw = gt["imgsz"]
            raw = self.geo_raw[i]
            boxes = ops.xywh2xyxy(raw[:, :4]) * torch.tensor([w, h, w, h])
            scores, classes = raw[:, 4:].max(-1)
            gt_boxes = gt["bboxes"].detach().float().cpu() * torch.tensor([w/iw, h/ih, w/iw, h/ih])
            row = dict(identity=self.export_identity, image_id=f"{self.args.split}:{key}", image=key,
                split=self.args.split, original_size_hw=[h, w], coordinate_space="original_image_pixels",
                box_format="xyxy", query_indices=list(range(len(raw))), boxes=boxes.tolist(),
                scores=scores.tolist(), classes=classes.tolist(), gt_boxes=gt_boxes.tolist(),
                gt_classes=gt["cls"].detach().cpu().tolist(), export_conf_filter=None,
                official_metric_conf_strictly_greater_than=.001)
            self.geo_stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.geo_raw = None
        return super().update_metrics(preds, batch)

    def finalize_metrics(self):
        if not self.one_batch and len(self.geo_seen) != len(self.dataloader.dataset):
            raise RuntimeError("Incomplete split evaluation")
        return super().finalize_metrics()

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if hasattr(self, "geo_stream"):
                self.geo_stream.close()
            self.geo_raw = None
