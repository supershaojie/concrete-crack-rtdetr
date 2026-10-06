"""Narrow data/trainer adapters; official YOLOv13 model, loss, DFL and assigner stay intact.

HSV formula is derived from the mother AGPL-3.0 source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Checkpoint serialization preserves the pinned author's trainer and EMA semantics (AGPL-3.0).
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
                     read_json, write_json, initialization_record, recipe, validate_checkpoint, native_recipe, sha256,
                     model_config)

# Repeat path/version/source-content guards even if a worker pre-imported Ultralytics.
configure(Path(os.environ.get('YOLOV13L_RUNTIME', ROOT / '.runtime/yolov13l-configurable/worker')),
          Path(os.environ.get('YOLOV13L_SOURCE', SOURCE)))
import ultralytics
if Path(ultralytics.__file__).resolve() != Path(os.environ['YOLOV13L_SOURCE']) / 'ultralytics/__init__.py':
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
from augment_b19 import counters, snapshot, training_transforms, describe_transform


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
    def __init__(self, *args, manifest, split, cache_root, cutmix_probability=None, reuse_cache=None, **kwargs):
        self.manifest = manifest
        self.split = split
        self.cache_root = Path(cache_root).resolve()
        self.reuse_cache = Path(reuse_cache).resolve() if reuse_cache else None
        if self.reuse_cache is not None:
            raise ValueError('YOLOv13-L label/size caches must belong to this run; reuse only checked manifests/GT')
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
        identity = digest(canonical({'version': 'yolov13l_header_labels_v1',
            'upstream': read_json(LOCK)['commit'], 'split': self.split,
            'root': str(self.shared_root), 'records': self.records}))
        self.cache_path = self.cache_root / self.split / (identity + '.labels.json')
        reused = self.reuse_cache / self.split / self.cache_path.name if self.reuse_cache else None
        if reused and reused.is_file():
            # The active pilot cache is read-only: no locks or writes in its tree.
            cached = read_json(reused)
            if cached.get('identity') != identity or cached.get('format') != 'yolov13l_header_labels_v1':
                raise ValueError('Reused label/size cache identity differs')
            self.label_cache_provenance = {'reused_from': str(reused), 'sha256': sha256(reused)}
            return self.labels_from_cache(cached)
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
                cached = {'identity': identity, 'format': 'yolov13l_header_labels_v1', 'entries': entries}
                write_json(self.cache_path, cached)
            if len(cached['entries']) != len(self.im_files):
                raise ValueError('Incomplete local label cache')
        self.label_cache_provenance = {'local_cache': str(self.cache_path), 'identity': identity}
        return self.labels_from_cache(cached)

    def labels_from_cache(self, cached):
        if len(cached['entries']) != len(self.im_files):
            raise ValueError('Incomplete label/size cache')
        labels = []
        for r, entry, filename in zip(self.records, cached['entries'], self.im_files):
            if entry['image'] != r['image'] or entry['boxes'] != r['boxes']:
                raise ValueError('Cache records differ from original YOLO labels')
            boxes = np.asarray(entry['boxes'], dtype=np.float32).reshape(-1, 5)
            labels.append({'im_file': filename, 'shape': (entry['height'], entry['width']),
                           'cls': boxes[:, :1], 'bboxes': boxes[:, 1:], 'segments': [],
                           'keypoints': None, 'normalized': True, 'bbox_format': 'xywh'})
        self.label_files = [str(self.shared_root / r['label']) for r in self.records]
        print('Isolated label cache: ' + str(self.label_cache_provenance), flush=True)
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
        if self.augment:
            self.transform_report['actual_transform_tree'] = describe_transform(transforms)
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
            classes=self.args.classes, max_nms=30000, max_time_img=float('inf'))
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
    def __call__(self, *args, **kwargs):
        from backend import attention_scope, flash_forward_count
        before = flash_forward_count()
        with attention_scope('native','training_validation'):
            result = super().__call__(*args, **kwargs)
            if flash_forward_count() != before:
                raise RuntimeError('Training validation unexpectedly called Flash')
            return result

    def preprocess(self, batch):
        result = super().preprocess(batch)
        self.actual_precision = {'input_dtype':str(result['img'].dtype),
                                 'half_argument':self.args.half, 'device':str(self.device),
                                 'attention_backend':'native','flash_calls':0,
                                 'precision_semantics':'unmodified author validator; CUDA training val uses trainer.amp half setting'}
        return result

    def postprocess(self, preds):
        from ultralytics.utils import ops
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            labels=self.lb, multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, max_nms=30000, max_time_img=float('inf'))


def model_identity(model, nc):
    """Author's full L graph; measured COCO80/nc1 identities are locked separately."""
    from collections import Counter
    head = model.model[-1]
    cfg = model.yaml
    params = sum(p.numel() for p in model.parameters())
    expected = model_config(Path(os.environ['YOLOV13L_SOURCE']))
    graph = [(row[0], row[2].replace('nn.', '')) for row in expected['backbone']+expected['head']]
    actual = [(m.f, type(m).__name__) for m in model.model]
    counts = Counter(type(m).__name__ for m in model.modules())
    expected_counts = {'AAttn':16, 'ABlock':16, 'HyperACE':1, 'FullPAD_Tunnel':7,
                       'DSC3k2':6, 'DSConv':58, 'A2C2f':2, 'DownsampleConv':1}
    if (nc not in (1, 80) or type(head) is not Detect or head.nc != nc
            or cfg.get('scale') != 'l' or cfg.get('nc') != nc
            or cfg.get('scales', {}).get('l') != [1., 1., 512]
            or cfg.get('backbone') != expected['backbone'] or cfg.get('head') != expected['head']
            or actual != graph or head.nl != 3 or head.reg_max != 16
            or model.stride.tolist() != [8.,16.,32.]
            or any(counts[k] != v for k,v in expected_counts.items())):
        raise ValueError('Expected pinned author YOLOv13-L full graph: '+str((nc,params,counts)))
    measured = read_json(LOCK).get('measured_identity', {}).get(str(nc))
    if measured and (params != measured['parameters_unfused'] or len(model.state_dict()) != measured['state_tensors']):
        raise ValueError('YOLOv13-L measured parameter/tensor identity differs')
    return {'scale': 'l', 'scaling':[1.,1.,512], 'nc': head.nc, 'parameters_unfused': params,
            'state_tensors':len(model.state_dict()), 'legacy':head.legacy,
            'module_counts':{k:counts[k] for k in expected_counts}, 'graph':actual,
            'trainable_parameter_elements':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'fixed_parameters':[n for n,p in model.named_parameters() if not p.requires_grad],
            'head': type(head).__module__ + '.' + type(head).__name__,
            'reg_max': head.reg_max, 'strides': model.stride.tolist(), 'yaml': model.yaml}


def transfer_report(original, model, resume=False):
    source, target = original.float().state_dict(), model.state_dict()
    transferred = intersect_dicts(source, target)
    skipped = sorted(set(target) - set(transferred))
    head_index = len(model.model)-1
    permitted = {f'model.{head_index}.cv3.{i}.{len(branch)-1}.{kind}'
                 for i,branch in enumerate(model.model[-1].cv3) for kind in ('weight','bias')}
    if set(source) != set(target) or set(skipped) != (set() if resume else permitted):
        raise ValueError('Unexpected missing/reshaped tensors beyond nc80 -> nc1 class outputs')
    for key in skipped:
        if (source[key].shape[0] != 80 or target[key].shape[0] != 1
                or source[key].shape[1:] != target[key].shape[1:]):
            raise ValueError('Class-output mismatch has another cause: ' + key)
    if not all(torch.equal(target[k], v) for k,v in transferred.items()):
        raise ValueError('Matching pretrained tensors were not transferred exactly before first update')
    measured = read_json(LOCK).get('transfer', {})
    if measured and not resume and len(transferred) != measured['transferred_tensors']:
        raise ValueError('YOLOv13-L measured COCO transfer coverage differs')
    protected_names = [name for name,module in model.named_modules()
                       if type(module).__name__ in ('FullPAD_Tunnel','HyperACE','AAttn','ABlock','A2C2f')]
    protected = sorted(k for k in transferred if any(k.startswith(name+'.') for name in protected_names))
    if not protected:
        raise ValueError('No protected gate/HyperACE/attention tensors were transferred')
    from ultralytics.nn.modules.block import FullPAD_Tunnel
    gates = {name:float(module.gate.detach()) for name,module in model.named_modules()
             if isinstance(module,FullPAD_Tunnel)}
    return {'transferred_tensors':len(transferred), 'destination_tensors':len(target),
            'transferred_elements':sum(v.numel() for v in transferred.values()),
            'transferred_keys':sorted(transferred), 'missing_or_reshaped_keys':skipped,
            'source_only_keys':sorted(set(source)-set(target)),
            'pretrained_tensors_loaded':0 if resume else len(transferred),
            'restored_checkpoint_tensors':len(transferred) if resume else 0,
            'skip_reasons':{k:{'reason':'nc80 -> nc1 class-output dimension only',
                'source_shape':list(source[k].shape), 'target_shape':list(target[k].shape)} for k in skipped},
            'protected_tensor_keys':protected, 'protected_tensor_count':len(protected),
            'FullPAD_gate_values_after_transfer':gates,
            'matching_tensor_sha256':{k:tensor_digest(v) for k,v in transferred.items()},
            'protected_values_equal_to_source':True,
            'matching_values_verified_before_first_update':True}


def tensor_digest(value):
    return digest(value.detach().cpu().contiguous().numpy().tobytes())


AMP_PROBE_RECORD = None


def strict_amp_probe(model):
    """Actual-model copy; original weights/BN/optimizer/EMA/scaler and all RNGs are untouched."""
    global AMP_PROBE_RECORD
    from backend import ArithmeticAudit, native_fp32, attention_scope
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('Formal recipe requires CUDA device=0 and AMP')
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            probe = deepcopy(model).float().eval()
            x = torch.linspace(0,1,3*64*64,device=device).reshape(1,3,64,64)
            with torch.no_grad():
                with native_fp32(), ArithmeticAudit(require_fp32=True) as fp_audit:
                    fp = probe(x)[0]
                with attention_scope('native','AMP_reference'), torch.autocast('cuda', dtype=torch.float16), ArithmeticAudit() as amp_audit:
                    amp = probe(x)[0]
                if (fp.shape != amp.shape or not torch.isfinite(fp).all() or not torch.isfinite(amp).all()
                        or not torch.allclose(fp,amp.float(),rtol=.1,atol=.5)):
                    raise RuntimeError('Actual-model AMP accuracy probe failed; no silent recipe changes')
            AMP_PROBE_RECORD = {'input':[1,3,64,64], 'rtol':.1, 'atol':.5,
                'max_abs_error':float((fp-amp.float()).abs().max()),
                'fp32':fp_audit.report(), 'amp':amp_audit.report(), 'reference_backend':'native',
                'flash_parity':'NOT_VERIFIED', 'model_state_and_rng_preserved':True}
            del probe
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return True


class ComparisonTrainer(DetectionTrainer):
    @property
    def amp_probe_record(self):
        return AMP_PROBE_RECORD

    def __init__(self, *args, run, manifest, identity, config=None, reuse_cache=None, **kwargs):
        self.comparison_run = Path(run)
        self.comparison_manifest = manifest
        self.comparison_identity = identity
        self.comparison_config = deepcopy(recipe() if config is None else config)
        self.comparison_reuse_cache = reuse_cache
        self.comparison_best_epoch = None
        self.comparison_optimizer_steps = 0
        self.comparison_optimizer_updates = 0
        self.comparison_overflow_skips = 0
        self.comparison_epoch_trace = []
        from ultralytics.engine import trainer as native_trainer
        from ultralytics.utils import callbacks
        native_trainer.check_amp = strict_amp_probe
        callbacks.add_integration_callbacks = lambda instance: None
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides):
        # Validate raw arguments before native setup scales decay into optimizer groups.
        # Native check_resume otherwise permits batch/device/close_mosaic changes.
        super().check_resume(overrides)
        if self.comparison_identity.get('scope') == 'CONFIGURABLE_FORMAL':
            for key, expected in native_recipe(self.comparison_config).items():
                if key in ('model', 'resume'):
                    continue
                actual = getattr(self.args, key)
                if (str(actual) != str(expected) if key == 'device' else actual != expected):
                    raise ValueError('Actual native argument differs from frozen run: ' + key)

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
        return IsolatedDataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=self.args.batch,
            augment=mode == 'train', hyp=copy(self.args), rect=False, cache=False, stride=32,
            pad=0.0, task='detect', data=self.data, manifest=self.comparison_manifest, split=mode,
            cache_root=self.comparison_run / 'cache', prefix=mode + ': ',
            cutmix_probability=self.comparison_config['cutmix'], reuse_cache=self.comparison_reuse_cache)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        # Ignore native validation batch * 2; both use this run's physical batch.
        physical_batch = getattr(self.args, 'batch', 16)
        dataset = self.build_dataset(dataset_path, mode, physical_batch)
        loader = build_dataloader(dataset, physical_batch, self.args.workers if mode == 'train' else 0,
                                  shuffle=mode == 'train', rank=-1)
        loader.reset = lambda: reset_workers(loader)
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise ValueError('COCO initialization or explicit resume requires verified weights')
        if not self.args.resume:
            model_identity(weights, 80)
        # Native DetectionModel changes cfg['nc'] in place. Preserve the verified
        # source checkpoint YAML rather than aliasing it into the nc1 destination.
        from backend import attention_scope
        with attention_scope('native','model_construction'):
            model = super().get_model(deepcopy(cfg if cfg is not None else weights.yaml), weights, verbose)
        record = model_identity(model, 1)
        record['source_model_identity'] = model_identity(weights, 1 if self.args.resume else 80)
        state = weights.float().state_dict()
        transferred = intersect_dicts(state, model.state_dict())
        record.update(transferred_tensors=len(transferred), destination_tensors=len(model.state_dict()),
                      transferred_elements=sum(v.numel() for v in transferred.values()),
                      transferred_keys=sorted(transferred), missing_or_reshaped_keys=sorted(set(model.state_dict())-set(transferred)))
        record.update(initialization_record(Path(os.environ['YOLOV13L_SOURCE']), self.comparison_config))
        record['restored_checkpoint_tensors'] = len(transferred) if self.args.resume else 0
        record.update(transfer_report(weights, model, resume=bool(self.args.resume)))
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
            'best_fitness': self.best_fitness, 'native_fitness': native_fitness,
            'native_fitness_definition':'pinned author Metric: [P,R,AP50,AP75,mAP50-95] weights [0,0,0,0,1]'})
        if hasattr(self.validator, 'actual_precision'):
            write_json(self.comparison_run/'native_validation.json', {
                **self.validator.actual_precision, 'imgsz':self.args.imgsz, 'batch':self.test_loader.batch_size,
                'rect':False, 'conf':self.validator.args.conf, 'iou':self.validator.args.iou,
                'max_det':self.validator.args.max_det, 'TTA':False,
                'selection':'native full precision mAP50-95; ties select later epoch'})
        return metrics, fitness

    def optimizer_step(self):
        # Leave native unscale -> clip(10) -> scaler.step/update -> zero_grad -> EMA intact.
        scale_before = self.scaler.get_scale()
        super().optimizer_step()
        self.comparison_optimizer_steps += 1
        overflow = self.scaler.get_scale() < scale_before
        self.comparison_overflow_skips += int(overflow)
        self.comparison_optimizer_updates += int(not overflow)

    def save_model(self):
        # Native EMA, optimizer and args, with resumable identity and atomic writes.
        self.comparison_epoch_trace.append(epoch_trace(self))
        buffer = io.BytesIO()
        torch.save({'epoch': self.epoch, 'best_fitness': self.best_fitness, 'model': None,
            'ema': deepcopy(self.ema.ema).half(), 'updates': self.ema.updates,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'train_args': vars(self.args), 'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_results': self.read_results_csv(), 'date': datetime.now().isoformat(),
            'version': ultralytics.__version__, 'license': 'AGPL-3.0', 'docs': 'https://docs.ultralytics.com',
            'comparison_identity': self.comparison_identity, 'comparison_best_epoch': self.comparison_best_epoch,
            'comparison_epoch_trace':self.comparison_epoch_trace,
            'comparison_initialization':read_json(self.comparison_run/'initialization.json'),
            'pilot_recipe':self.comparison_config, 'comparison_training_state':{
                'model':cpu_state(self.model.state_dict()), 'optimizer':cpu_state(self.optimizer.state_dict()),
                'ema':cpu_state(self.ema.ema.state_dict()), 'scaler':self.scaler.state_dict(),
                'scheduler':self.scheduler.state_dict(), 'rng':capture_rng(),
                'stopper':{k:getattr(self.stopper,k) for k in ('best_fitness','best_epoch','possible_stop','patience')},
                'optimizer_steps':self.comparison_optimizer_steps,
                'optimizer_updates':self.comparison_optimizer_updates, 'overflow_skips':self.comparison_overflow_skips,
                'augmentation_counters':list(self.train_loader.dataset.augmentation_counts[:]),
                'augmentation_closed':self.train_loader.dataset.augmentation_closed}}, buffer)
        raw = buffer.getvalue()
        atomic_bytes(self.last, raw)
        if self.best_fitness == self.fitness:
            atomic_bytes(self.best, raw)
        write_json(self.comparison_run/'checkpoint_integrity.json', {
            'epoch':self.epoch+1, 'best_epoch':self.comparison_best_epoch,
            'last_sha256':digest(raw), 'best_sha256':digest(raw) if self.best_fitness == self.fitness else sha256(self.best)})

    def resume_training(self, ckpt):
        if self.resume:
            validate_checkpoint(ckpt, self.comparison_identity, resume=True,
                                config=getattr(self, 'comparison_config', None))
        super().resume_training(ckpt)
        if self.resume:
            self.restore_training_state(ckpt)
            self.comparison_best_epoch = ckpt['comparison_best_epoch']
            self.stopper.best_fitness = self.best_fitness
            self.stopper.best_epoch = self.comparison_best_epoch
            self.stopper.possible_stop = self.start_epoch - self.comparison_best_epoch >= self.stopper.patience - 1
            if self.args.close_mosaic and self.start_epoch > self.epochs - self.args.close_mosaic:
                self.train_loader.reset()  # base resume closes transforms after workers were created

    def restore_training_state(self, ckpt):
        state = ckpt['comparison_training_state']
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.ema.ema.load_state_dict(state['ema'], strict=True)
        self.scaler.load_state_dict(state['scaler'])
        self.scheduler.load_state_dict(state['scheduler'])
        if 'stopper' in state:
            if state['stopper']['patience'] != self.stopper.patience:
                raise ValueError('Saved early-stopping patience differs from frozen configuration')
            for key,value in state['stopper'].items():
                setattr(self.stopper,key,value)
        self.comparison_optimizer_steps = state['optimizer_steps']
        self.comparison_optimizer_updates = state['optimizer_updates']
        self.comparison_overflow_skips = state['overflow_skips']
        self.comparison_epoch_trace = ckpt['comparison_epoch_trace']
        with self.train_loader.dataset.augmentation_counts.get_lock():
            self.train_loader.dataset.augmentation_counts[:] = state['augmentation_counters']
        if read_json(self.comparison_run/'initialization.json') != ckpt['comparison_initialization']:
            raise ValueError('Original initialization record differs on resume')
        if state['augmentation_closed'] != self.train_loader.dataset.augmentation_closed:
            raise ValueError('Saved close_mosaic state differs from resumed dataset')
        restore_rng(state['rng'])
        write_json(self.comparison_run/'resume_state.json', {'epoch':ckpt['epoch']+1,
            'restored':['FP32 model','FP32 optimizer','FP32 EMA','scaler','scheduler','RNG','patience','augmentation state'],
            'optimizer_steps':self.comparison_optimizer_steps, 'original_initialization_preserved':True,
            'optimizer_updates':self.comparison_optimizer_updates, 'overflow_skips':self.comparison_overflow_skips,
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


def epoch_trace(t):
    return {'epoch':t.epoch+1, 'accumulate':t.accumulate, 'amp':bool(t.amp), 'batch':t.batch_size,
        'learning_rates':t.lr, 'best_epoch':t.comparison_best_epoch, 'fitness':t.fitness,
        'optimizer_steps_cumulative':t.comparison_optimizer_steps,
        'optimizer_updates_cumulative':t.comparison_optimizer_updates,
        'amp_overflow_skips_cumulative':t.comparison_overflow_skips,
        'augmentation':t.train_loader.dataset.transform_report['probabilities'],
        'augmentation_closed':t.train_loader.dataset.augmentation_closed,
        'augmentation_counts_cumulative':snapshot(t.train_loader.dataset),
        'losses':t.tloss.detach().cpu().tolist(),
        'train_attention_backend':t.comparison_identity.get('train_attention_backend'),
        'eval_attention_backend':'native', 'flash_deterministic_backward':t.comparison_config['deterministic'],
        'flash_evidence':__import__('backend').flash_evidence()}


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
