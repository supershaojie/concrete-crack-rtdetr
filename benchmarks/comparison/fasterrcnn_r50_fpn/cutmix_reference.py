# Ultralytics AGPL-3.0. CutMix methods from fixed commit f2d3aed634a5b0e4828024718d4a61ab2f83fb19.
# Only class name/imports/docstrings changed; old BaseMixTransform is adapted by augment_b19.py.
from __future__ import annotations
from typing import Any
import numpy as np
from augmentation_reference import BaseMixTransform
from instances import Instances
from geometry_ops import bbox_ioa

class FixedCutMix(BaseMixTransform):

    def __init__(self, dataset, pre_transform=None, p: float=0.0, beta: float=1.0, num_areas: int=3) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return (x1, y1, x2, y2)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        (h, w) = labels['img'].shape[:2]
        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels['instances'].bboxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            return labels
        labels2 = labels.pop('mix_labels')[0]
        area = cut_areas[np.random.choice(idx)]
        ioa2 = bbox_ioa(area[None], labels2['instances'].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels['instances'].segments) else 0.1))[0]
        if len(indexes2) == 0:
            return labels
        instances2 = labels2['instances'][indexes2]
        instances2.convert_bbox('xyxy')
        instances2.denormalize(w, h)
        (x1, y1, x2, y2) = area.astype(np.int32)
        labels['img'][y1:y2, x1:x2] = labels2['img'][y1:y2, x1:x2]
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls'][indexes2]], axis=0)
        labels['instances'] = Instances.concatenate([labels['instances'], instances2], axis=0)
        return labels
