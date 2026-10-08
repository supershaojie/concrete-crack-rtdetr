"""AGPL-3.0: reference v8 detection pipeline on the pinned YOLO26 runtime.

The reference class is unchanged computationally. This module owns the old
Instances interface, train-only partner selection, and worker-shared counters.
It never imports new model, loss, optimizer, or assigner code.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import numpy as np
import augmentation_reference as augment
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
    if sha256(HERE/lock['reference_pipeline']['local_excerpt']) != lock['reference_pipeline']['excerpt_sha256']:
        raise ValueError('Locked reference v8 detection transforms changed')
    from image_adapters import MotherHSV, NoAlbumentations
    affine = augment.RandomPerspective(degrees=hyp.degrees,translate=hyp.translate,scale=hyp.scale,
        shear=hyp.shear,perspective=hyp.perspective,pre_transform=None)
    pre = augment.Compose([CountedMosaic(dataset,imgsz=imgsz,p=hyp.mosaic),augment.CopyPaste(p=0.,mode='flip'),affine])
    chain = augment.Compose([pre,CountedMixUp(dataset,pre_transform=pre,p=hyp.mixup),
        DetectionCutMix(dataset,pre_transform=pre,p=cutmix_probability),NoAlbumentations(),
        MotherHSV(hgain=hyp.hsv_h,sgain=hyp.hsv_s,vgain=hyp.hsv_v),
        augment.RandomFlip(direction='vertical',p=hyp.flipud),augment.RandomFlip(direction='horizontal',p=hyp.fliplr)])
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


def describe_transform(transform):
    """Read the constructed objects rather than claiming application from YAML alone."""
    result = {'type':type(transform).__module__ + '.' + type(transform).__name__}
    for key in ('p','beta','num_areas','hgain','sgain','vgain','degrees','translate','scale',
                'shear','perspective','direction','bgr','normalize','bbox_format','stretch','new_shape'):
        value = getattr(transform,key,None)
        if value is not None and isinstance(value,(str,int,float,bool,tuple,list)):
            result[key] = value
    if hasattr(transform,'transforms'):
        result['children'] = [describe_transform(t) for t in transform.transforms]
    if getattr(transform,'pre_transform',None) is not None:
        result['pre_transform'] = describe_transform(transform.pre_transform)
    return result
