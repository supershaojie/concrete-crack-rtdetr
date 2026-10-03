# Adapted from Ultralytics AGPL-3.0 CutMix, v8.4.0
# f2d3aed634a5b0e4828024718d4a61ab2f83fb19, ultralytics/data/augment.py:878-1009.
# Complete license and unmodified reference are in vendor/. No Ultralytics imports.
"""Detection CutMix for v5 absolute [class,x1,y1,x2,y2] labels."""
from multiprocessing import get_context
import numpy as np

COUNTERS = ('samples', 'pre_transform', 'mosaic_applied', 'mixup_triggered', 'mixup_applied',
            'cutmix_triggered', 'cutmix_applied', 'cutmix_no_free_area', 'cutmix_no_donor', 'hsv_applied')
PIPELINE = ('train-only square stretch', 'Mosaic or single-image path', 'v5 random perspective',
            'MixUp p=.135 with pre-transform-only train partner',
            'CutMix p=.03 with pre-transform-only train partner',
            'retained additive HSV', 'vertical flip p=0', 'horizontal flip p=.5', 'v5 label formatting')


def make_counters():
    # Spawn-compatible synchronized storage is also shared by fork workers on Linux.
    return get_context('spawn').Array('q', len(COUNTERS), lock=True)


def bump(dataset, name):
    counter = dataset._comparison_counts
    with counter.get_lock():
        counter[COUNTERS.index(name)] += 1


def snapshot(dataset):
    with dataset._comparison_counts.get_lock():
        return dict(zip(COUNTERS, dataset._comparison_counts[:]))


def rand_bbox(width, height):
    lam = np.random.beta(1.0, 1.0)
    ratio = np.sqrt(1.0-lam)
    cw, ch = int(width*ratio), int(height*ratio)
    cx, cy = np.random.randint(width), np.random.randint(height)
    return (np.clip(cx-cw//2, 0, width), np.clip(cy-ch//2, 0, height),
            np.clip(cx+cw//2, 0, width), np.clip(cy+ch//2, 0, height))


def bbox_ioa(areas, boxes):
    """Intersection / donor box area; matches fixed source bbox_ioa eps=1e-7."""
    lo = np.maximum(areas[:, None, :2], boxes[None, :, :2])
    hi = np.minimum(areas[:, None, 2:], boxes[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(2)
    return intersection / ((boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])+1e-7)


def detection_cutmix(image, labels, donor, targets):
    """Three candidate regions, zero base-box overlap; donor IOA>=.1, then clip.

    Rejected triggers keep the original image and labels. No class soft weights,
    extra width/height filter, or modification of existing base boxes is added.
    Detection-only labels have no segments, so the source .1 threshold applies.
    """
    if donor.shape != image.shape:
        raise ValueError('CutMix requires identically pre-transformed train image shapes')
    h, w = image.shape[:2]
    areas = np.asarray([rand_bbox(w, h) for _ in range(3)], dtype=np.float32)
    free = np.nonzero(bbox_ioa(areas, labels[:, 1:]).sum(axis=1) <= 0)[0]
    if not len(free):
        return image, labels, 'no_free_area'
    area = areas[np.random.choice(free)]
    keep = np.nonzero(bbox_ioa(area[None], targets[:, 1:]).squeeze(0) >= .1)[0]
    if not len(keep):
        return image, labels, 'no_donor'
    x1, y1, x2, y2 = area.astype(np.int32)
    image[y1:y2, x1:x2] = donor[y1:y2, x1:x2]
    added = targets[keep].copy()
    added[:, [1, 3]] = np.clip(added[:, [1, 3]], x1, x2)
    added[:, [2, 4]] = np.clip(added[:, [2, 4]], y1, y2)
    return image, np.concatenate((labels, added), axis=0), 'applied'
