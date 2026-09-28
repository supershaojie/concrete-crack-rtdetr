"""Independent FP32 evaluation + complete query/GT stream in the same pass.

Corrected sorted-confidence mask from C19 a0459d6 / ARG ef9cb7e.
Training epoch validator and its fitness/best selection are unchanged.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops

POLICY = "corrected_sorted_conf_mask_v1"
EVAL = dict(imgsz=640, batch=16, workers=0, half=False, conf=0.001, iou=0.7,
            max_det=300, augment=False, rect=False, seed=42)


def postprocess(preds, imgsz, conf):
    raw = preds[0] if isinstance(preds, (tuple, list)) else preds
    outputs = []
    for pred in raw:
        boxes = ops.xywh2xyxy(pred[:, :4] * imgsz)
        scores, classes = pred[:, 4:].max(-1)
        order = scores.argsort(descending=True)
        rows = torch.cat((boxes, scores[:, None], classes[:, None]), -1)[order]
        rows = rows[rows[:, 4] > conf]
        outputs.append(dict(bboxes=rows[:, :4], conf=rows[:, 4], cls=rows[:, 5]))
    return outputs


class OneBatch:
    def __init__(self, loader):
        self.loader, self.dataset, self.batch_size = loader, loader.dataset, loader.batch_size

    def __len__(self):
        return 1

    def __iter__(self):
        yield next(iter(self.loader))


class DTRValidator(RTDETRValidator):
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
            raise RuntimeError("DTRValidator is only for independent evaluation")
        self.actual_settings = vars(self.args).copy()
        self.seen_ids, self.raw_queries = set(), None
        self.query_count, self.gt_count = 0, 0
        if self.export_path:
            self.stream = gzip.open(self.export_path, "xt", encoding="utf-8")

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite FP32 evaluation predictions")
        if raw.shape[-1] != 5:
            raise ValueError("DTR v1 evaluation requires nc=1")
        if self.export_path:
            self.raw_queries = raw.detach().float().cpu()
        return postprocess(preds, self.args.imgsz, self.args.conf)

    def update_metrics(self, preds, batch):
        for i in range(len(preds)):
            gt = self._prepare_batch(i, batch)
            key = Path(gt["im_file"]).resolve().relative_to(Path(self.data["path"]).resolve()).as_posix()
            if key in self.seen_ids:
                raise ValueError(f"Duplicate evaluation image: {key}")
            self.seen_ids.add(key)
            self.gt_count += len(gt["cls"])
            if self.export_path:
                h, w = map(int, gt["ori_shape"])
                ih, iw = gt["imgsz"]
                raw = self.raw_queries[i]
                self.query_count += len(raw)
                scores, classes = raw[:, 4:].max(-1)
                row = dict(identity=self.export_identity, image_id=key, image=key, split=self.args.split,
                    original_size_hw=[h, w], coordinate_space="original_image_pixels", box_format="xyxy",
                    query_indices=list(range(len(raw))),
                    boxes=(ops.xywh2xyxy(raw[:, :4]) * torch.tensor([w, h, w, h])).tolist(),
                    scores=scores.tolist(), classes=classes.tolist(),
                    gt_boxes=(gt["bboxes"].detach().float().cpu() * torch.tensor([w / iw, h / ih, w / iw, h / ih])).tolist(),
                    gt_classes=gt["cls"].detach().cpu().tolist())
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.raw_queries = None
        return super().update_metrics(preds, batch)

    def get_stats(self):
        # Preserve the evaluator's exact TP/conf arrays before native clear_stats.
        if self.export_path:
            arrays = {k: np.concatenate(v, axis=0) if v else np.array([]) for k, v in self.metrics.stats.items()}
            np.savez_compressed(Path(self.export_path).parent / "native_statistics.npz", **arrays)
        return super().get_stats()

    def finalize_metrics(self):
        if not self.one_batch and len(self.seen_ids) != len(self.dataloader.dataset):
            raise RuntimeError("Evaluation did not cover the entire split")
        return super().finalize_metrics()

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        finally:
            if hasattr(self, "stream"):
                self.stream.close()
