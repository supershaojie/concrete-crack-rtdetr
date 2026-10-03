"""Narrow data/trainer adapters; official YOLOv8 model, loss, DFL and assigner stay intact.

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
                     read_json, write_json, initialization_record, recipe, validate_checkpoint)

if 'ultralytics' not in sys.modules:  # spawned workers must also import the official tree
    configure(Path(os.environ.get('YOLOV8M_RUNTIME', ROOT / '.runtime/yolov8m-coco-b19-pilot/worker')),
              Path(os.environ.get('YOLOV8M_SOURCE', SOURCE)))
import ultralytics
if Path(ultralytics.__file__).resolve() != Path(os.environ['YOLOV8M_SOURCE']) / 'ultralytics/__init__.py':
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
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, intersect_dicts
from data import image_size, read_labels
from augment_b19 import counters, snapshot, training_transforms


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
    def __init__(self, *args, manifest, split, cache_root, cutmix_probability=None, **kwargs):
        self.manifest = manifest
        self.split = split
        self.cache_root = Path(cache_root).resolve()
        self.cutmix_probability = recipe()['cutmix'] if cutmix_probability is None else cutmix_probability
        self.augmentation_closed = False
        self.augmentation_counts = counters()
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
        identity = digest(canonical({'version': 'yolov8m_header_labels_v1',
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
                    if len(entries) % 500 == 0 or len(entries) == len(self.records):
                        print(f'Isolated {self.split} label/header cache: {len(entries)}/{len(self.records)}',flush=True)
                cached = {'identity': identity, 'format': 'yolov8m_header_labels_v1', 'entries': entries}
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
            transforms = training_transforms(self, self.imgsz, hyp, self.cutmix_probability)
        else:
            transforms = augment.Compose([augment.LetterBox((self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(augment.Format(bbox_format='xywh', normalize=True, batch_idx=True,
                          mask_ratio=hyp.mask_ratio, mask_overlap=hyp.overlap_mask, bgr=0.0))
        return transforms

    def close_mosaic(self, hyp):
        # Old close_mosaic omits CutMix; this adapter owns all four switches.
        hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
        self.cutmix_probability = 0.0
        self.augmentation_closed = True
        self.transforms = self.build_transforms(hyp)


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
        self.inverse_geometry = []
        transformed = []
        for im in images:
            h,w = im.shape[:2]
            r = min(self.imgsz[0]/h, self.imgsz[1]/w)
            resized = (round(w*r), round(h*r))
            left = round((self.imgsz[1]-resized[0])/2-0.1)
            top = round((self.imgsz[0]-resized[1])/2-0.1)
            actual = letterbox(image=im)
            if actual.shape[:2] != tuple(self.imgsz):
                raise ValueError('Public letterbox output shape differs')
            self.inverse_geometry.append({'gain_x':resized[0]/w, 'gain_y':resized[1]/h,
                                          'left':left, 'top':top, 'resized':resized})
            transformed.append(actual)
        return transformed

    def postprocess(self, preds, img, orig_imgs):
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops
        if img.dtype != torch.float32:
            raise ValueError('Public inference input must be FP32')
        # Same official NMS, without its time-budget early break (which can leave
        # later images empty under contention). No changes to IoU/ranking rules.
        predictions = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes, max_time_img=float('inf'))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        if len(self.inverse_geometry) != len(predictions):
            raise ValueError('Missing actual per-image resize/padding metadata')
        for pred, original, path, geometry in zip(predictions, orig_imgs, self.batch[0], self.inverse_geometry):
            pred[:, [0,2]] = (pred[:, [0,2]]-geometry['left'])/geometry['gain_x']
            pred[:, [1,3]] = (pred[:, [1,3]]-geometry['top'])/geometry['gain_y']
            # Keep every NMS box after coordinate inversion. Clamping padding-only
            # detections collapses them to zero area (invalid public schema) and
            # removing those detections would erase false positives. The public
            # matcher accepts finite positive-area boxes outside the image.
            results.append(Results(original, path=path, names=self.model.names, boxes=pred))
        return results


class CompleteValidator(DetectionValidator):
    def preprocess(self, batch):
        result = super().preprocess(batch)
        self.actual_precision = {'input_dtype':str(result['img'].dtype),
                                 'half_argument':self.args.half, 'device':str(self.device)}
        return result

    def postprocess(self, preds):
        from ultralytics.utils import ops
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            labels=self.lb, multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, max_time_img=float('inf'))


def model_identity(model, nc):
    head = model.model[-1]
    params = sum(p.numel() for p in model.parameters())
    scale = model.yaml.get('scale')
    if scale is None and (model.yaml.get('depth_multiple'), model.yaml.get('width_multiple')) == (.67, .75):
        scale = 'm'  # legacy official COCO checkpoint embeds explicit depth/width, not 'scale'
    from ultralytics.nn.modules import Conv
    expected = {80: 25902640, 1: 25856899}[nc]
    if (type(head) is not Detect or head.nc != nc or scale != 'm' or params != expected
            or not all(type(branch[0]) is Conv and type(branch[1]) is Conv for branch in head.cv3)):
        raise ValueError('Expected official YOLOv8m Detect: ' + str((type(head), head.nc, scale, params)))
    return {'scale': scale, 'nc': head.nc, 'parameters_unfused': params,
            'head': type(head).__module__ + '.' + type(head).__name__,
            'reg_max': head.reg_max, 'strides': model.stride.tolist(), 'yaml': model.yaml}


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
        self.comparison_optimizer_steps = 0
        from ultralytics.engine import trainer as native_trainer
        from ultralytics.utils import callbacks
        native_trainer.check_amp = strict_amp_probe
        callbacks.add_integration_callbacks = lambda instance: None
        super().__init__(*args, **kwargs)

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

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise ValueError('COCO initialization or explicit resume requires verified weights')
        model = super().get_model(cfg, weights, verbose)
        record = model_identity(model, 1)
        state = weights.float().state_dict()
        transferred = intersect_dicts(state, model.state_dict())
        record.update(transferred_tensors=len(transferred), destination_tensors=len(model.state_dict()),
                      transferred_elements=sum(v.numel() for v in transferred.values()),
                      transferred_keys=sorted(transferred), missing_or_reshaped_keys=sorted(set(model.state_dict())-set(transferred)))
        record.update(initialization_record(Path(os.environ['YOLOV8M_SOURCE'])))
        record['restored_checkpoint_tensors'] = len(transferred) if self.args.resume else 0
        if not self.args.resume:
            record['pretrained_tensors_loaded'] = len(transferred)
            if len(transferred) != 469 or len(model.state_dict()) != 475:
                raise ValueError('Official nc80 -> nc1 transfer coverage changed')
            record['skip_reasons'] = {k:{'reason':'nc=80 -> nc=1 class-output shape mismatch',
                    'source_shape':list(state[k].shape), 'target_shape':list(model.state_dict()[k].shape)}
                    for k in record['missing_or_reshaped_keys']}
        if not all(torch.equal(model.state_dict()[k], v) for k, v in transferred.items()):
            raise ValueError('Official pretrained transfer did not reproduce matching tensors')
        record['matching_values_verified_before_first_update'] = True
        record['resume'] = bool(self.args.resume)
        record['construction'] = ('official DetectionTrainer.get_model -> nc1 model.load(resume EMA); exact FP32 train state restored by resume_training'
            if self.args.resume else 'official DetectionTrainer.get_model -> DetectionModel(original COCO yaml, nc=1) -> model.load(weights)')
        destination = self.comparison_run / ('resume_model.json' if self.args.resume else 'initialization.json')
        if not self.args.resume and destination.exists():
            raise FileExistsError('Original initialization record is immutable')
        write_json(destination, record)
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
        if hasattr(self.validator, 'actual_precision'):
            write_json(self.comparison_run/'native_validation.json', {
                **self.validator.actual_precision, 'imgsz':self.args.imgsz, 'batch':self.test_loader.batch_size,
                'rect':False, 'conf':self.validator.args.conf, 'iou':self.validator.args.iou,
                'max_det':self.validator.args.max_det, 'TTA':False,
                'selection':'native full precision mAP50-95; ties select later epoch'})
        return metrics, fitness

    def optimizer_step(self):
        # Leave native unscale -> clip(10) -> scaler.step/update -> zero_grad -> EMA intact.
        super().optimizer_step()
        self.comparison_optimizer_steps += 1

    def save_model(self):
        # Native EMA, optimizer and args, with resumable identity and atomic writes.
        buffer = io.BytesIO()
        torch.save({'epoch': self.epoch, 'best_fitness': self.best_fitness, 'model': None,
            'ema': deepcopy(self.ema.ema).half(), 'updates': self.ema.updates,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'train_args': vars(self.args), 'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_results': self.read_results_csv(), 'date': datetime.now().isoformat(),
            'version': ultralytics.__version__, 'license': 'AGPL-3.0', 'docs': 'https://docs.ultralytics.com',
            'comparison_identity': self.comparison_identity, 'comparison_best_epoch': self.comparison_best_epoch,
            'comparison_initialization':read_json(self.comparison_run/'initialization.json'),
            'pilot_recipe':recipe(), 'comparison_training_state':{
                'model':cpu_state(self.model.state_dict()), 'optimizer':cpu_state(self.optimizer.state_dict()),
                'ema':cpu_state(self.ema.ema.state_dict()), 'scaler':self.scaler.state_dict(),
                'scheduler':self.scheduler.state_dict(), 'rng':capture_rng(),
                'optimizer_steps':self.comparison_optimizer_steps,
                'augmentation_closed':self.train_loader.dataset.augmentation_closed}}, buffer)
        raw = buffer.getvalue()
        atomic_bytes(self.last, raw)
        if self.best_fitness == self.fitness:
            atomic_bytes(self.best, raw)

    def resume_training(self, ckpt):
        if self.resume:
            validate_checkpoint(ckpt, self.comparison_identity, resume=True)
        super().resume_training(ckpt)
        if self.resume:
            self.restore_training_state(ckpt)
            self.comparison_best_epoch = ckpt['comparison_best_epoch']
            self.stopper.best_fitness = self.best_fitness
            self.stopper.best_epoch = self.comparison_best_epoch
            self.stopper.possible_stop = self.start_epoch - self.comparison_best_epoch >= self.args.patience - 1
            if self.start_epoch > self.epochs - self.args.close_mosaic:
                self.train_loader.reset()  # base resume closes transforms after workers were created

    def restore_training_state(self, ckpt):
        state = ckpt['comparison_training_state']
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.ema.ema.load_state_dict(state['ema'], strict=True)
        self.scaler.load_state_dict(state['scaler'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.comparison_optimizer_steps = state['optimizer_steps']
        if read_json(self.comparison_run/'initialization.json') != ckpt['comparison_initialization']:
            raise ValueError('Original initialization record differs on resume')
        if state['augmentation_closed'] != self.train_loader.dataset.augmentation_closed:
            raise ValueError('Saved close_mosaic state differs from resumed dataset')
        restore_rng(state['rng'])
        write_json(self.comparison_run/'resume_state.json', {'epoch':ckpt['epoch']+1,
            'restored':['FP32 model','FP32 optimizer','FP32 EMA','scaler','scheduler','RNG','patience','augmentation state'],
            'optimizer_steps':self.comparison_optimizer_steps, 'original_initialization_preserved':True,
            'continuation':'native epoch-boundary resume; prefetched worker batches and cross-epoch pending gradients are not bitwise replayed'})

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        write_json(self.comparison_run/'augmentation_closed.json', {
            'zero_based_epoch':self.epochs-self.args.close_mosaic,
            'closed':['mosaic','mixup','cutmix','copy_paste'],
            'pipeline':self.train_loader.dataset.transform_report,
            'counts_at_rebuild':snapshot(self.train_loader.dataset),
            'worker_reset':'native epoch trigger resets iterator; resume also resets closed workers'})

    def final_eval(self):
        # Every epoch already validated. Preserve resumable best/last; the next step
        # exports the same selected best with explicit FP32, then invokes the public evaluator.
        print('Native training finished; separate FP32 public export/evaluation follows.', flush=True)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(cpu_state(v) for v in value)
    return deepcopy(value)


def capture_rng():
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
