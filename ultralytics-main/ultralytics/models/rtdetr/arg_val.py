"""Mother's corrected_sorted_conf_mask_v1, with optional streaming raw queries."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=0.001, iou=0.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    # Same computation/order as tools/c19_lif_v1_results.py at a0459d6.
    preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    result = []
    for pred in preds:
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


class ARGValidator(RTDETRValidator):
    """Only independent evaluations use this class; native epoch validation stays native."""
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
            raise RuntimeError("ARGValidator is for independent evaluation")
        self.actual_settings = vars(self.args).copy()
        self.arg_seen = set()
        self.arg_raw = None
        if self.export_path:
            self.arg_stream = gzip.open(self.export_path, "xt", encoding="utf-8")

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite independent evaluation predictions")
        if self.export_path:
            self.arg_raw = raw.detach().float().cpu()
        return postprocess(preds, self.args.imgsz, self.args.conf)

    def update_metrics(self, preds, batch):
        for i in range(len(preds)):
            gt = self._prepare_batch(i, batch)
            key = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
            if key in self.arg_seen:
                raise ValueError(f"Duplicate evaluation image: {key}")
            self.arg_seen.add(key)
            if self.export_path:
                h, w = map(int, gt["ori_shape"])
                ih, iw = gt["imgsz"]
                raw = self.arg_raw[i]
                boxes = ops.xywh2xyxy(raw[:, :4]) * torch.tensor([w, h, w, h])
                scores, classes = raw[:, 4:].max(-1)
                gt_boxes = gt["bboxes"].detach().float().cpu() * torch.tensor([w / iw, h / ih, w / iw, h / ih])
                row = dict(identity=self.export_identity, image=key, original_size_hw=[h, w],
                    coordinate_space="original_image_pixels", box_format="xyxy", query_indices=list(range(len(raw))),
                    boxes=boxes.tolist(), scores=scores.tolist(), classes=classes.tolist(),
                    gt_boxes=gt_boxes.tolist(), gt_classes=gt["cls"].detach().cpu().tolist())
                self.arg_stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.arg_raw = None
        return super().update_metrics(preds, batch)

    def finalize_metrics(self):
        if not self.one_batch and len(self.arg_seen) != len(self.dataloader.dataset):
            raise RuntimeError("Independent evaluation did not cover full split")
        return super().finalize_metrics()

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if hasattr(self, "arg_stream"):
                self.arg_stream.close()
