"""AGPL-3.0: narrow v8.4.0 detection CutMix integration into pinned v8.3.20.

The reference class is unchanged computationally. This module owns the old
Instances interface, train-only partner selection, and worker-shared counters.
It never imports new model, loss, optimizer, or assigner code.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import numpy as np
from ultralytics.data import augment
from cutmix_reference import FixedCutMix
from support import HERE, read_json, sha256

OPERATIONS = ('mosaic', 'mixup', 'cutmix')
EVENTS = ('visits', 'triggered', 'applied', 'skipped_geometry')


def counters():
    # A spawn-context lock is also valid when Linux uses fork for data workers.
    return mp.get_context('spawn').Array('q', len(OPERATIONS)*len(EVENTS), lock=True)


def count(dataset, operation, event):
    index = OPERATIONS.index(operation)*len(EVENTS)+EVENTS.index(event)
    with dataset.augmentation_counts.get_lock():
        dataset.augmentation_counts[index] += 1


def snapshot(dataset):
    with dataset.augmentation_counts.get_lock():
        values = list(dataset.augmentation_counts[:])
    return {name:dict(zip(EVENTS, values[i*len(EVENTS):(i+1)*len(EVENTS)]))
            for i,name in enumerate(OPERATIONS)}


class CountedMosaic(augment.Mosaic):
    def __call__(self, labels):
        count(self.dataset, 'mosaic', 'visits')
        return labels if self.p <= 0 else super().__call__(labels)

    def _mix_transform(self, labels):
        count(self.dataset, 'mosaic', 'triggered')
        result = super()._mix_transform(labels)
        count(self.dataset, 'mosaic', 'applied')
        return result


class CountedMixUp(augment.MixUp):
    def __call__(self, labels):
        count(self.dataset, 'mixup', 'visits')
        return labels if self.p <= 0 else super().__call__(labels)

    def _mix_transform(self, labels):
        count(self.dataset, 'mixup', 'triggered')
        result = super()._mix_transform(labels)
        count(self.dataset, 'mixup', 'applied')
        return result


class DetectionCutMix(FixedCutMix):
    def __init__(self, dataset, pre_transform=None, p=0.0):
        if dataset.split != 'train' or dataset.use_segments or dataset.use_keypoints:
            raise ValueError('Pilot CutMix requires the isolated train detection dataset')
        super().__init__(dataset, pre_transform, p=p, beta=1.0, num_areas=3)

    def get_indexes(self):
        # The new BaseMixTransform implements this default; the old one is abstract.
        # get_image_and_label is the raw train sample, never dataset.__getitem__.
        return random.randint(0, len(self.dataset)-1)

    def __call__(self, labels):
        count(self.dataset, 'cutmix', 'visits')
        return labels if self.p <= 0 else super().__call__(labels)

    def _mix_transform(self, labels):
        count(self.dataset, 'cutmix', 'triggered')
        h,w = labels['img'].shape[:2]
        partner = labels['mix_labels'][0]
        if partner['img'].shape != labels['img'].shape:
            raise ValueError('CutMix partner must have the same post-affine square shape')
        for item in (labels, partner):
            # bbox_ioa in the fixed algorithm needs absolute xyxy BEFORE IOA.
            item['instances'].convert_bbox('xyxy')
            item['instances'].denormalize(w, h)
            if len(item['cls']) != len(item['instances']):
                raise ValueError('CutMix class/instance count differs')
        before = len(labels['cls'])
        result = super()._mix_transform(labels)
        event = 'applied' if len(result['cls']) > before else 'skipped_geometry'
        count(self.dataset, 'cutmix', event)
        return result


def training_transforms(dataset, imgsz, hyp, cutmix_probability):
    lock = read_json(HERE/'augmentation.lock.json')
    if sha256(HERE/lock['excerpt']) != lock['excerpt_sha256']:
        raise ValueError('Fixed official CutMix excerpt changed')
    # Reuse the complete old pipeline; replace only its two counted mixers and
    # insert the fixed detection CutMix at the same position as v8.4.0.
    chain = augment.v8_transforms(dataset, imgsz, hyp, stretch=True)
    pre = chain.transforms[0]
    pre.transforms[0] = CountedMosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    chain.transforms[1] = CountedMixUp(dataset, pre_transform=pre, p=hyp.mixup)
    chain.insert(2, DetectionCutMix(dataset, pre_transform=pre, p=cutmix_probability))
    dataset.transform_report = {
        'pipeline': ['Mosaic', 'CopyPaste(p=0)', 'RandomPerspective', 'MixUp', 'DetectionCutMix',
                     'NoAlbumentations', 'MotherHSV(additive hue)', 'flipud', 'fliplr', 'Format(RGB)'],
        'training_resize': 'square stretch; old BaseDataset.load_image(rect_mode=False)',
        'partners': 'only train get_image_and_label + bounded Mosaic/CopyPaste/Affine; no full-pipeline recursion',
        'probabilities': {'mosaic':hyp.mosaic, 'mixup':hyp.mixup, 'cutmix':cutmix_probability, 'copy_paste':hyp.copy_paste},
        'geometry': {k:getattr(hyp,k) for k in ('degrees','translate','scale','shear','perspective')},
        'hsv': {k:getattr(hyp,k) for k in ('hsv_h','hsv_s','hsv_v')},
        'flips': {'flipud':hyp.flipud, 'fliplr':hyp.fliplr},
        'cutmix': {'source_commit':lock['commit'], 'beta':1.0, 'candidate_rectangles':3,
                   'rule':'zero main-box intersection; donor retained IOA>=0.1; donor boxes clipped to patch',
                   'labels':'hard detection class IDs + concatenated Instances, no soft labels'},
        'counters':'worker-shared cumulative visits/triggered/applied/skipped_geometry; visits include bounded partner pre-transforms'}
    return chain
