"""Narrow data/trainer adapters; official YOLO11 model, loss, DFL and assigner stay intact.

HSV formula is derived from the mother AGPL-3.0 source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Checkpoint serialization follows official v8.3.20 BaseTrainer.save_model (AGPL-3.0).
"""
from __future__ import annotations

from copy import copy, deepcopy
from datetime import datetime
import io
import os
import random
from pathlib import Path
import sys

from support import (SOURCE, ROOT, LOCK, atomic_bytes, canonical, configure, digest, local_lock,
                     read_json, write_json, initialization_record, model_yaml, model_config, validate_checkpoint, sha256)

if 'ultralytics' not in sys.modules:  # spawned workers must also import the official tree
    configure(Path(os.environ.get('YOLO11M_RUNTIME', ROOT / '.runtime/yolo11m-scratch/worker')),
              Path(os.environ.get('YOLO11M_SOURCE', SOURCE)))
import ultralytics
if Path(ultralytics.__file__).resolve() != Path(os.environ['YOLO11M_SOURCE']) / 'ultralytics/__init__.py':
    raise RuntimeError('Refusing a mother/installed Ultralytics import')
import cv2
import numpy as np
import torch
from ultralytics.data import augment
from ultralytics.data.build import InfiniteDataLoader, build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import intersect_dicts
from data import image_size, read_labels


class MotherHSV(augment.RandomHSV):
    def __call__(self, labels):
        img = labels['img']
        if img.shape[-1] != 3:
            return labels
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            x = np.arange(256, dtype=r.dtype)
            hue = ((x + r[0] * 180) % 180).astype(img.dtype)
            sat = np.clip(x * (r[1] + 1), 0, 255).astype(img.dtype)
            val = np.clip(x * (r[2] + 1), 0, 255).astype(img.dtype)
            sat[0] = 0
            h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            hsv = cv2.merge((cv2.LUT(h, hue), cv2.LUT(s, sat), cv2.LUT(v, val)))
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class NoAlbumentations:
    """Explicit pre-result choice: historical activation was not directly proved."""
    def __init__(self, *args, **kwargs):
        self.transform = None

    def __call__(self, labels):
        return labels


augment.RandomHSV = MotherHSV
augment.Albumentations = NoAlbumentations


class IsolatedDataset(YOLODataset):
    def __init__(self, *args, manifest, split, cache_root, **kwargs):
        self.manifest = manifest
        self.split = split
        self.cache_root = Path(cache_root).resolve()
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.shared_root = Path(manifest['data_root']).resolve()
        if kwargs.get('cache') not in (False, None) or kwargs.get('rect', False):
            raise ValueError('Only cache=False, rect=False are part of this recipe')
        super().__init__(*args, **kwargs)
        # BaseDataset otherwise reads/deletes neighboring .npy even with cache=False.
        self.npy_files = [self.cache_root / 'unused-image-cache' / (str(i)+'.npy') for i in range(self.ni)]

    def get_img_files(self, img_path):
        actual = super().get_img_files(img_path)
        expected = [str((self.shared_root / r['image']).resolve()) for r in self.records]
        if [str(Path(p).resolve()) for p in actual] != expected:
            raise ValueError('Actual dataset loader paths/order differ from preflight: ' + self.split)
        return actual

    def get_labels(self):
        # Build real labels/sizes, not an AUDITED claim or a foreign framework cache.
        identity = digest(canonical({'version': 'yolo11m_header_labels_v1',
            'upstream': read_json(LOCK)['commit'], 'split': self.split,
            'root': str(self.shared_root), 'records': self.records}))
        self.cache_path = self.cache_root / self.split / (identity + '.labels.json')
        with local_lock(self.cache_path.with_suffix('.lock')):
            if self.cache_path.exists():
                cached = read_json(self.cache_path)
                if cached['identity'] != identity:
                    raise ValueError('Local label/size cache identity differs')
            else:
                entries = []
                for r, filename in zip(self.records, self.im_files):
                    boxes, label_hash = read_labels(self.shared_root / r['label'])
                    if label_hash != r['label_sha256'] or boxes != r['boxes']:
                        raise ValueError('Label changed while building cache: ' + r['label'])
                    width, height = image_size(filename)
                    entries.append({'image': r['image'], 'width': width, 'height': height, 'boxes': boxes})
                cached = {'identity': identity, 'format': 'yolo11m_header_labels_v1', 'entries': entries}
                write_json(self.cache_path, cached)
            if len(cached['entries']) != len(self.im_files):
                raise ValueError('Incomplete local label cache')
        labels = []
        for r, entry, filename in zip(self.records, cached['entries'], self.im_files):
            if entry['image'] != r['image'] or entry['boxes'] != r['boxes']:
                raise ValueError('Cache records differ from original YOLO labels')
            boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)
            labels.append({'im_file': filename, 'shape': (entry['height'], entry['width']),
                           'cls': boxes[:, :1], 'bboxes': boxes[:, 1:], 'segments': [],
                           'keypoints': None, 'normalized': True, 'bbox_format': 'xywh'})
        self.label_files = [str(self.shared_root / r['label']) for r in self.records]
        print('Isolated label cache: ' + str(self.cache_path), flush=True)
        return labels

    def load_image(self, i, rect_mode=True):
        # Training matches mother's square stretch; validation retains YOLO letterbox.
        return super().load_image(i, rect_mode=not self.augment)

    def build_transforms(self, hyp=None):
        if self.augment:
            transforms = augment.v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = augment.Compose([augment.LetterBox((self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(augment.Format(bbox_format='xywh', normalize=True, batch_idx=True,
                          mask_ratio=hyp.mask_ratio, mask_overlap=hyp.overlap_mask, bgr=0.0))
        return transforms


def reset_workers(loader):
    """Discard prefetched old-transform batches before creating fresh worker copies."""
    iterator = getattr(loader, 'iterator', None)
    if iterator is not None and hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    InfiniteDataLoader.reset(loader)


class SquarePredictor(DetectionPredictor):
    def pre_transform(self, images):
        # v8.3.20 ignores rect=False here and enables auto padding for same-shaped inputs.
        letterbox = augment.LetterBox(self.imgsz, auto=False, stride=self.model.stride)
        return [letterbox(image=im) for im in images]

    def postprocess(self, preds, img, orig_imgs):
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops
        # Same official NMS, without its time-budget early break (which can leave
        # later images empty under contention). No changes to IoU/ranking rules.
        predictions = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes, max_time_img=float('inf'), max_nms=30000)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for pred, original, path in zip(predictions, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], original.shape)
            results.append(Results(original, path=path, names=self.model.names, boxes=pred))
        return results


class CompleteValidator(DetectionValidator):
    def postprocess(self, preds):
        from ultralytics.utils import ops
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            labels=self.lb, multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, max_time_img=float('inf'), max_nms=30000)


def model_identity(model, nc):
    """Check actual YOLO11 M modules, not the filename or a YOLOv8 parameter constant."""
    from ultralytics.nn.modules import C3k2, C2PSA, SPPF, DWConv
    from ultralytics.nn.modules.block import C3k
    head = model.model[-1]
    cfg = model.yaml
    blocks = [m for m in model.model if type(m) is C3k2]
    depthwise_head = (type(head) is Detect and all(
        isinstance(branch[0], torch.nn.Sequential) and type(branch[0][0]) is DWConv
        and isinstance(branch[1], torch.nn.Sequential) and type(branch[1][0]) is DWConv
        for branch in head.cv3))
    expected_yaml = model_config(Path(os.environ['YOLO11M_SOURCE']))
    if (nc not in (1, 80) or type(head) is not Detect or head.nc != nc
            or cfg.get('scale') != 'm' or cfg.get('nc') != nc
            or cfg.get('scales', {}).get('m') != [.5, 1., 512]
            or cfg.get('backbone') != expected_yaml['backbone'] or cfg.get('head') != expected_yaml['head']
            or head.legacy or not depthwise_head or head.nl != 3 or head.reg_max != 16
            or model.stride.tolist() != [8., 16., 32.]
            or len(blocks) != 8 or not all(len(m.m) == 1 and type(m.m[0]) is C3k for m in blocks)
            or type(model.model[9]) is not SPPF or type(model.model[10]) is not C2PSA
            or len(model.model[10].m) != 1
            or [branch[0][0].conv.in_channels for branch in head.cv3] != [256,512,512]):
        raise ValueError('Expected pinned official YOLO11m C3k2/C2PSA and nonlegacy P3/P4/P5 Detect')
    params = sum(p.numel() for p in model.parameters())
    return {'scale': 'm', 'scaling': [.5,1.,512], 'nc': head.nc,
            'parameters_unfused': params, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'head': type(head).__module__ + '.' + type(head).__name__, 'legacy': head.legacy,
            'classification_head': 'native depthwise Conv branches',
            'C3k2_blocks': len(blocks), 'C3k2_M_branch': 'C3k in all eight blocks',
            'C2PSA_blocks': 1, 'reg_max': head.reg_max, 'strides': model.stride.tolist(), 'yaml': cfg}


def strict_amp_probe(model):
    """Use the actual model, without downloading an unrelated model or disabling AMP."""
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('Formal recipe requires CUDA device=0 and AMP')
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device]):
            probe = deepcopy(model).eval()
            with torch.no_grad():
                x = torch.linspace(0,1,3*64*64,device=device).reshape(1,3,64,64)
                with torch.autocast(device_type='cuda', enabled=False):
                    fp = probe(x)[0]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    amp = probe(x)[0]
                if (fp.shape != amp.shape or not torch.isfinite(fp).all() or not torch.isfinite(amp).all()
                        or not torch.allclose(fp, amp.float(), rtol=.1, atol=.5)):
                    raise RuntimeError('Actual-model AMP accuracy probe failed; recipe is not silently changed')
            del probe
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return True


class ComparisonTrainer(DetectionTrainer):
    def __init__(self, *args, run, manifest, identity, **kwargs):
        self.comparison_run = Path(run)
        self.comparison_manifest = manifest
        self.comparison_identity = identity
        self.comparison_best_epoch = None
        from ultralytics.engine import trainer as native_trainer
        from ultralytics.utils import callbacks
        native_trainer.check_amp = strict_amp_probe
        callbacks.add_integration_callbacks = lambda instance: None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        if not self.args.resume:
            self.resume = False
            return
        last = self.comparison_run/'train/weights/last.pt'
        if not isinstance(self.args.resume, (str, Path)) or Path(self.args.resume).resolve() != last.resolve():
            raise ValueError('Explicit same-run last.pt required; automatic resume is forbidden')
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        validate_checkpoint(ckpt, self.comparison_identity, resume=True)
        if ckpt['comparison_initialization_sha256'] != sha256(self.comparison_run/'initialization.json'):
            raise ValueError('Original initialization record changed')
        from ultralytics.cfg import get_cfg
        self.args = get_cfg(ckpt['train_args'])
        self.args.model = self.args.resume = str(last)
        self.resume = True

    def get_dataset(self):
        # Already checked, absolute data paths; no dataset download/font side effects.
        from support import load_yaml
        self.data = load_yaml(self.args.data)
        return self.data['train'], self.data['val']

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return CompleteValidator(self.test_loader, save_dir=self.save_dir,
                                 args=copy(self.args), _callbacks=self.callbacks)

    def build_dataset(self, img_path, mode='train', batch=None):
        return IsolatedDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=16,
            augment=mode == 'train', hyp=copy(self.args), rect=False, cache=False, stride=32,
            pad=0.0, task='detect', data=self.data, manifest=self.comparison_manifest, split=mode,
            cache_root=self.comparison_run / 'cache', prefix=mode + ': ')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        # Ignore base trainer's validation batch * 2. Both loaders use exactly 16.
        dataset = self.build_dataset(dataset_path, mode, 16)
        loader = build_dataloader(dataset, 16, self.args.workers if mode == 'train' else 0,
                                  shuffle=mode == 'train', rank=-1)
        loader.reset = lambda: reset_workers(loader)
        return loader

    def setup_model(self):
        if self.args.resume:
            last = self.comparison_run / 'train/weights/last.pt'
            if Path(self.args.resume).resolve() != last.resolve():
                raise ValueError('Resume accepts only this run last.pt')
            ckpt = torch.load(last, map_location='cpu', weights_only=False)
            validate_checkpoint(ckpt, self.comparison_identity, resume=True)
            # Restore the live training weights; EMA remains a separate official ModelEMA.
            self.model = self.get_model(model_config(Path(os.environ['YOLO11M_SOURCE'])),
                                        weights=ckpt['model'])
            return ckpt
        if self.comparison_identity['initialization_type']=='random' and not self.args.resume:
            source = Path(os.environ['YOLO11M_SOURCE'])
            if self.args.pretrained is not False or str(self.model) != str(model_yaml(source)):
                raise ValueError('Scratch setup requires the verified official YAML and pretrained=False')
            self.model = self.get_model(model_config(source), weights=None)
            return None
        return super().setup_model()

    def get_model(self, cfg=None, weights=None, verbose=True):
        scratch = self.comparison_identity['initialization_type']=='random'
        if not scratch:
            raise ValueError('This model branch only accepts random-initialized scratch runs')
        if scratch and not self.args.resume:
            if weights is not None or self.args.pretrained is not False:
                raise ValueError('New scratch training forbids all supplied weights')
            if cfg != model_config(Path(os.environ['YOLO11M_SOURCE'])):
                raise ValueError('Scratch model must use the pinned official M YAML')
        elif weights is None:
            raise ValueError('Explicit scratch resume requires verified live model weights')
        if weights is not None:
            model_identity(weights, 1)
        model = super().get_model(cfg, weights, verbose)
        record = model_identity(model, 1)
        if weights is None:
            record.update(initialization_record(Path(os.environ['YOLO11M_SOURCE'])))
            record.update(transferred_tensors=0, transferred_elements=0, transferred_keys=[],
                          destination_tensors=len(model.state_dict()), weights_argument=None, resume=False,
                          construction='ComparisonTrainer.setup_model -> get_model(weights=None) -> DetectionModel',
                          fixed_parameters=[n for n,p in model.named_parameters() if not p.requires_grad])
            record.update(run_identity=self.comparison_identity,
                          expanded_train_args=vars(self.args),
                          expanded_train_args_sha256=digest(canonical(vars(self.args))),
                          source_patch=read_json(LOCK))
            initial_path = self.comparison_run/'initialization.json'
            if initial_path.exists():
                raise FileExistsError('Original initialization record is immutable')
            write_json(initial_path, record)
            return model
        state = weights.float().state_dict()
        transferred = intersect_dicts(state, model.state_dict())
        if set(transferred) != set(model.state_dict()) or set(transferred) != set(state):
            raise ValueError('Resume must restore every model tensor without partial transfer')
        record.update(transferred_tensors=len(transferred), destination_tensors=len(model.state_dict()),
                      transferred_elements=sum(v.numel() for v in transferred.values()),
                      transferred_keys=sorted(transferred), missing_or_reshaped_keys=sorted(set(model.state_dict())-set(transferred)))
        record.update(initialization_record(Path(os.environ['YOLO11M_SOURCE'])))
        record['restored_checkpoint_tensors'] = len(transferred) if self.args.resume else 0
        if not self.args.resume:
            record['pretrained_tensors_loaded'] = len(transferred)
        if not all(torch.equal(model.state_dict()[k], v) for k, v in transferred.items()):
            raise ValueError('Same-run checkpoint restore did not reproduce matching tensors')
        write_json(self.comparison_run / ('resume_model_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'), record)
        return model

    def validate(self):
        metrics = self.validator(self)
        native_fitness = metrics.pop('fitness')
        fitness = float(self.validator.metrics.box.map)  # full precision, not five-decimal log rounding
        if not np.isfinite(fitness):
            raise ValueError('Nonfinite validation mAP50-95')
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.comparison_best_epoch = self.epoch + 1
        metrics['metrics/native_fitness'] = native_fitness
        write_json(self.comparison_run / 'training_progress.json', {
            'completed_epoch': self.epoch+1, 'best_epoch': self.comparison_best_epoch,
            'selection_metric': 'training_val_mAP50_95_full_precision', 'fitness': fitness,
            'best_fitness': self.best_fitness, 'native_fitness_0.1AP50_0.9mAP': native_fitness})
        return metrics, fitness

    def save_model(self):
        # Native EMA, optimizer and args, with resumable identity and atomic writes.
        buffer = io.BytesIO()
        torch.save({'epoch': self.epoch, 'best_fitness': self.best_fitness,
            'model': deepcopy(self.model).float().cpu(),
            'ema': deepcopy(self.ema.ema).float().cpu(), 'updates': self.ema.updates,
            'optimizer': deepcopy(self.optimizer.state_dict()), 'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'comparison_stopper': {'best_fitness': self.stopper.best_fitness,
                                  'best_epoch': self.stopper.best_epoch,
                                  'possible_stop': self.stopper.possible_stop,
                                  'patience': self.stopper.patience},
            'comparison_initialization_sha256': sha256(self.comparison_run/'initialization.json'),
            'comparison_expanded_args_sha256': digest(canonical(vars(self.args))),
            'checkpoint_schema': 'yolo11m_scratch_fp32_resume_v1',
            'train_args': vars(self.args), 'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_results': self.read_results_csv(), 'date': datetime.now().isoformat(),
            'version': ultralytics.__version__, 'license': 'AGPL-3.0', 'docs': 'https://docs.ultralytics.com',
            'comparison_identity': self.comparison_identity, 'comparison_best_epoch': self.comparison_best_epoch}, buffer)
        raw = buffer.getvalue()
        atomic_bytes(self.last, raw)
        if self.best_fitness == self.fitness:
            atomic_bytes(self.best, raw)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if self.resume:
            validate_checkpoint(ckpt, self.comparison_identity, resume=True)
            if ckpt['comparison_initialization_sha256'] != sha256(self.comparison_run/'initialization.json'):
                raise ValueError('Original initialization record changed')
            self.scaler.load_state_dict(ckpt['scaler'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.comparison_best_epoch = ckpt['comparison_best_epoch']
            saved = ckpt['comparison_stopper']
            for name in ('best_fitness', 'best_epoch', 'possible_stop'):
                setattr(self.stopper, name, saved[name])
            if self.start_epoch > self.epochs - self.args.close_mosaic:
                self.train_loader.reset()  # rebuild workers carrying the closed augmentation
            write_json(self.comparison_run / ('resume_state_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'), {
                'start_epoch_zero_based': self.start_epoch, 'best_epoch': self.comparison_best_epoch,
                'scaler': self.scaler.state_dict(), 'stopper': saved, 'ema_updates': self.ema.updates,
                'model_restore': 'live FP32 model', 'ema_restore': 'separate FP32 ModelEMA',
                'optimizer_restore': 'unrounded checkpoint state',
                'limit': 'epoch-boundary restart; prefetched worker RNG/partial batches are not replayed'})

    def final_eval(self):
        # Every epoch already validated. Preserve resumable best/last; the next step
        # exports the same selected best with explicit FP32, then invokes the public evaluator.
        print('Native training finished; separate FP32 public export/evaluation follows.', flush=True)
