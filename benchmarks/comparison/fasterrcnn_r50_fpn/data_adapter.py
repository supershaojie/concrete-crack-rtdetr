"""Pinned YOLOv8m-scratch data operations, producing TorchVision image/target lists.

MotherHSV and isolated cache logic derive from the frozen 61c386bc adapter (AGPL-3.0).
Only data components are used; no YOLO model, predictor, trainer or loss is constructed.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random

from support import RUNTIME, canonical, configure_augmentation, digest, local_lock, read_json, write_json
configure_augmentation(Path(os.environ.get('FRCNN_DATA_RUNTIME', RUNTIME/'data-worker')))

import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import augment
from ultralytics.data.dataset import YOLODataset
from data import image_size, read_labels

AUG_KEYS = ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
            'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'erasing', 'bgr', 'auto_augment')


class MotherHSV(augment.RandomHSV):
    def __call__(self, labels):
        img = labels['img']
        if img.shape[-1] == 3 and (self.hgain or self.sgain or self.vgain):
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            x = np.arange(256, dtype=r.dtype)
            hue = ((x + r[0] * 180) % 180).astype(img.dtype)
            sat = np.clip(x * (r[1] + 1), 0, 255).astype(img.dtype)
            val = np.clip(x * (r[2] + 1), 0, 255).astype(img.dtype)
            sat[0] = 0
            h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            hsv = cv2.merge((cv2.LUT(h, hue), cv2.LUT(s, sat), cv2.LUT(v, val)))
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)
        labels['hsv_executed'] = True
        return labels


class NoAlbumentations:
    def __init__(self, *args, **kwargs):
        self.transform = None

    def __call__(self, labels):
        return labels


class TracedMosaic(augment.Mosaic):
    def _mix_transform(self, labels):
        result = super()._mix_transform(labels)
        self.dataset.observed['mosaic'] += 1
        return result


class TracedMixUp(augment.MixUp):
    def _mix_transform(self, labels):
        result = super()._mix_transform(labels)
        self.dataset.observed['mixup'] += 1
        return result


augment.RandomHSV = MotherHSV
augment.Albumentations = NoAlbumentations
augment.Mosaic = TracedMosaic
augment.MixUp = TracedMixUp


def augmentation_record(cfg):
    return {'reference_commit': '61c386bc722b11208453cab0808d8f6edb15385d',
        'upstream': 'ultralytics v8.3.20/f4d8f7765a490f3920e2d14c592a2967e347f185',
        'order': ['square stretch load', 'Mosaic(4)', 'CopyPaste(disabled)', 'RandomPerspective',
                  'MixUp(second sample uses same pre-transform)', 'Albumentations(disabled)',
                  'additive MotherHSV', 'vertical flip', 'horizontal flip', 'RGB CHW float32 /255'],
        'parameters': {k: cfg[k] for k in AUG_KEYS},
        'sampling': 'native Mosaic buffer random.choices; MixUp random.randint, Beta(32,32), pixel blend + concatenate boxes',
        'filtering': 'native RandomPerspective box_candidates thresholds; legal empty targets retained',
        'training_geometry': 'v8_transforms(stretch=True), BaseDataset.load_image(rect_mode=False)',
        'normalization': 'data RGB [0,1] only; model applies its own image_mean/image_std',
        'close_zero_based_epoch': cfg['epochs']-cfg['close_mosaic'],
        'worker_propagation': 'persistent_workers=False; exhaust and destroy each epoch iterator; fresh copies next epoch',
        'extra_operations': {'albumentations': False, 'copy_paste': False, 'cutmix': False, 'erasing': False},
        'differences_from_frozen_data_adapter': 'Final xyxy pixel boxes and foreground label=1; no YOLO batch tensors/loss'}


class TrainDataset(YOLODataset):
    def __init__(self, run, manifest, cfg):
        self.run, self.manifest, self.cfg = Path(run), manifest, deepcopy(cfg)
        self.records = [r for r in manifest['records'] if r['split'] == 'train']
        self.shared_root = Path(manifest['data_root'])
        self.cache_root = self.run/'cache'
        self.observed = {'mosaic': 0, 'mixup': 0}
        self.epoch = 0
        hyp = get_cfg(overrides={k: cfg[k] for k in AUG_KEYS})
        super().__init__(img_path=str(self.run/'train.txt'), imgsz=cfg['input_height'],
            batch_size=cfg['batch_size'], augment=True, hyp=hyp, rect=False, cache=False,
            data={'names': {0: 'crack'}}, task='detect')
        # BaseDataset otherwise reads/deletes shared *.npy even when cache=False.
        self.npy_files = [self.cache_root/'unused-image-cache'/(str(i)+'.npy') for i in range(self.ni)]

    def get_img_files(self, img_path):
        actual = super().get_img_files(img_path)
        expected = [str((self.shared_root/r['image']).resolve()) for r in self.records]
        if [str(Path(p).resolve()) for p in actual] != expected:
            raise ValueError('Loader image order differs from frozen manifest')
        return actual

    def get_labels(self):
        identity = digest(canonical({'version': 'frcnn_header_labels_v1', 'records': self.records,
                                    'root': str(self.shared_root)}))
        self.cache_path = self.cache_root/'train'/(identity[:24]+'.labels.json')
        with local_lock(self.cache_path.with_suffix('.lock')):
            if self.cache_path.exists():
                cached = read_json(self.cache_path)
                if cached['identity'] != identity:
                    raise ValueError('Wrong local cache identity')
            else:
                entries = []
                for r, filename in zip(self.records, self.im_files):
                    boxes, sha = read_labels(self.shared_root/r['label'])
                    if sha != r['label_sha256'] or boxes != r['boxes']:
                        raise ValueError('Source label changed: ' + r['label'])
                    width, height = image_size(filename)
                    entries.append({'image': r['image'], 'width': width, 'height': height, 'boxes': boxes})
                cached = {'identity': identity, 'entries': entries}
                write_json(self.cache_path, cached)
        if len(cached['entries']) != len(self.records):
            raise ValueError('Incomplete label/size cache')
        labels = []
        for r, entry, filename in zip(self.records, cached['entries'], self.im_files):
            if entry['image'] != r['image'] or entry['boxes'] != r['boxes']:
                raise ValueError('Cache differs from original labels')
            boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)
            labels.append({'im_file': filename, 'shape': (entry['height'], entry['width']),
                'cls': boxes[:, :1], 'bboxes': boxes[:, 1:], 'segments': [], 'keypoints': None,
                'normalized': True, 'bbox_format': 'xywh'})
        self.label_files = [str(self.shared_root/r['label']) for r in self.records]
        return labels

    def load_image(self, i, rect_mode=True):
        return super().load_image(i, rect_mode=False)

    def build_transforms(self, hyp=None):
        pipeline = augment.v8_transforms(self, self.imgsz, hyp, stretch=True)
        pipeline.append(augment.Format(bbox_format='xyxy', normalize=False, batch_idx=False,
            mask_ratio=hyp.mask_ratio, mask_overlap=hyp.overlap_mask, bgr=0.0))
        return pipeline

    def set_epoch(self, epoch):
        self.epoch = epoch
        hyp = get_cfg(overrides={k: self.cfg[k] for k in AUG_KEYS})
        if self.cfg['close_mosaic'] and epoch >= self.cfg['epochs'] - self.cfg['close_mosaic']:
            hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
        self.transforms = self.build_transforms(hyp)
        self.active_probabilities = {'mosaic': hyp.mosaic, 'mixup': hyp.mixup}

    def __getitem__(self, index):
        self.observed = {'mosaic': 0, 'mixup': 0}
        sample = super().__getitem__(index)
        image = sample['img'].float().div_(255)
        boxes = sample['bboxes'].to(dtype=torch.float32).reshape(-1, 4)
        original_cls = sample['cls'].reshape(-1)
        if len(boxes) != len(original_cls) or (original_cls != 0).any():
            raise ValueError('Expected YOLO class 0 before the foreground-1 mapping')
        if not torch.isfinite(boxes).all() or ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
            raise ValueError('Invalid post-augmentation boxes')
        target = {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64),
                  'image_id': torch.tensor([self.records[index]['image_id']], dtype=torch.int64),
                  'area': (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                  'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)}
        meta = {'epoch': self.epoch, 'worker_pid': os.getpid(), **self.observed,
                'hsv': bool(sample.get('hsv_executed')), 'empty_target': not len(boxes)}
        return image, target, meta


def letterbox(image, size=640):
    height, width = image.shape[:2]
    ratio = min(size/height, size/width)
    resized_w, resized_h = round(width*ratio), round(height*ratio)
    dw, dh = (size-resized_w)/2, (size-resized_h)/2
    left, right, top, bottom = round(dw-.1), round(dw+.1), round(dh-.1), round(dh+.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    result = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if result.shape != (size, size, 3):
        raise ValueError('Unexpected letterbox shape')
    return result, {'original_width': width, 'original_height': height,
        'resized_width': resized_w, 'resized_height': resized_h,
        'scale_x': resized_w/width, 'scale_y': resized_h/height,
        'left': left, 'right': right, 'top': top, 'bottom': bottom,
        'size': size, 'rounding': 'round(resize), round(half_pad +/- 0.1), auto=False, scaleup=True'}


def forward_boxes(boxes, geometry):
    result = boxes.clone().float()
    result[:, [0, 2]] = result[:, [0, 2]]*geometry['scale_x'] + geometry['left']
    result[:, [1, 3]] = result[:, [1, 3]]*geometry['scale_y'] + geometry['top']
    return result


def inverse_boxes(boxes, geometry):
    result = boxes.detach().float().cpu().clone()
    result[:, [0, 2]] = (result[:, [0, 2]]-geometry['left'])/geometry['scale_x']
    result[:, [1, 3]] = (result[:, [1, 3]]-geometry['top'])/geometry['scale_y']
    # Preserve floats and all native ROI detections; no extra clipping, NMS or rounding.
    return result


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, root, gt, size=640):
        self.root, self.images, self.size = Path(root), sorted(gt['images'], key=lambda r: r['id']), size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = self.images[index]
        pixels = cv2.imread(str(self.root/im['file_name']))
        if pixels is None or tuple(pixels.shape[:2]) != (im['height'], im['width']):
            raise ValueError('Decode/dimension mismatch: ' + im['file_name'])
        pixels, geometry = letterbox(pixels, self.size)
        image = torch.from_numpy(np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1))).float()/255
        return image, im, geometry


def collate(batch):
    return tuple(map(list, zip(*batch)))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def train_loader(dataset, cfg, generator):
    return torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['train_workers'], persistent_workers=False, pin_memory=True, drop_last=False,
        collate_fn=collate, worker_init_fn=seed_worker, generator=generator)
