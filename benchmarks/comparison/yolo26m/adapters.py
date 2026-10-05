"""Narrow data/trainer adapters; official YOLO26 model, E2ELoss, L1 slot and assigners stay intact.

HSV formula is derived from the mother AGPL-3.0 source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Complete FP32 checkpoint state and native v8.4.0 epoch loss updates are retained.
"""
from __future__ import annotations

from copy import copy, deepcopy
from datetime import datetime
import io
import os
import random
from pathlib import Path
import sys
import uuid

from support import (SOURCE, ROOT, LOCK, atomic_bytes, canonical, configure, digest, local_lock,
                     read_json, write_json, initialization_record, recipe, validate_checkpoint, native_recipe, sha256,
                     model_config)

# Repeat path/version/source-content guards even if a worker pre-imported Ultralytics.
configure(Path(os.environ.get('YOLO26M_RUNTIME', ROOT / '.runtime/yolo26m-configurable/worker')),
          Path(os.environ.get('YOLO26M_SOURCE', SOURCE)))
import ultralytics
if Path(ultralytics.__file__).resolve() != Path(os.environ['YOLO26M_SOURCE']) / 'ultralytics/__init__.py':
    raise RuntimeError('Refusing a mother/installed Ultralytics import')
import cv2
import numpy as np
import torch
import augmentation_reference as augment
from ultralytics.data.build import InfiniteDataLoader, build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import intersect_dicts, unwrap_model
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
        identity = digest(canonical({'version': 'yolo26m_configurable_header_labels_v1',
            'upstream': read_json(LOCK)['commit'], 'split': self.split,
            'root': str(self.shared_root), 'records': self.records}))
        self.cache_path = self.cache_root / self.split / (identity + '.labels.json')
        reused = self.reuse_cache / self.split / self.cache_path.name if self.reuse_cache else None
        if reused and reused.is_file():
            # The active pilot cache is read-only: no locks or writes in its tree.
            cached = read_json(reused)
            if cached.get('identity') != identity or cached.get('format') != 'yolo26m_configurable_header_labels_v1':
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
                cached = {'identity': identity, 'format': 'yolo26m_configurable_header_labels_v1', 'entries': entries}
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
        # v8.4.0 ignores rect=False here and enables auto padding for same-shaped inputs.
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

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops
        if img.dtype != torch.float32 or not self.model.end2end:
            raise ValueError('Public inference requires FP32 native end2end one-to-one output')
        end2end_tensor(preds)
        predictions = nms.non_max_suppression(preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
            classes=self.args.classes, end2end=True, max_time_img=float('inf'))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        if not (len(self.inverse_geometry) == len(predictions) == len(orig_imgs) == len(self.batch[0])):
            raise ValueError('Incomplete full-batch native end2end postprocessing')
        results = []
        for pred, original, path, g in zip(predictions, orig_imgs, self.batch[0], self.inverse_geometry):
            pred[:, [0,2]] = (pred[:, [0,2]]-g['left'])/g['gain_x']
            pred[:, [1,3]] = (pred[:, [1,3]]-g['top'])/g['gain_y']
            results.append(Results(original, path=path, names=self.model.names, boxes=pred))
        return results


class CompleteValidator(DetectionValidator):
    def preprocess(self, batch):
        result = super().preprocess(batch)
        self.actual_precision = {'input_dtype':str(result['img'].dtype),
                                 'half_argument':self.args.half, 'device':str(self.device)}
        return result

    def postprocess(self, preds):
        if not self.end2end:
            raise ValueError('Training validation requires native one-to-one output')
        end2end_tensor(preds)
        return super().postprocess(preds)




def end2end_tensor(preds):
    value = preds[0] if isinstance(preds, (tuple,list)) else preds
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 6:
        raise ValueError('Expected native one-to-one postprocessed [B,N,6], not raw branch dict or BCN')
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite native one-to-one output')
    return value

def model_identity(model, nc, deployed=False):
    from ultralytics.nn.modules import C3k2, C2PSA, SPPF
    head, cfg = model.model[-1], model.yaml
    expected = model_config(Path(os.environ['YOLO26M_SOURCE']))
    if (nc not in (1,80) or type(head) is not Detect or head.nc != nc
            or cfg.get('scale') != 'm' or cfg.get('nc') != nc
            or cfg.get('scales',{}).get('m') != [.5,1.,512]
            or cfg.get('backbone') != expected['backbone'] or cfg.get('head') != expected['head']
            or head.reg_max != 1 or not head.end2end or not isinstance(head.dfl,torch.nn.Identity)
            or model.stride.tolist() != [8.,16.,32.]
            or not hasattr(head,'one2one_cv2') or not hasattr(head,'one2one_cv3')
            or type(model.model[9]) is not SPPF or type(model.model[10]) is not C2PSA
            or sum(type(m) is C3k2 for m in model.model) != 8):
        raise ValueError('Expected locked official YOLO26m m/nc/reg_max/end2end/stride/structure')
    if not deployed and (head.cv2 is None or head.cv3 is None or not any(
            isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())):
        raise ValueError('Training requires complete unfused one-to-many and one-to-one branches')
    return {'scale':'m','scaling':[.5,1.,512],'nc':head.nc,
        'parameters_deployed' if deployed else 'parameters_unfused':sum(p.numel() for p in model.parameters()),
        'state_tensors':len(model.state_dict()), 'trainable_parameter_elements':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'fixed_parameters':[n for n,p in model.named_parameters() if not p.requires_grad],
        'head':type(head).__module__+'.'+type(head).__name__,'end2end':True,'reg_max':1,
        'C3k2_blocks':8,'SPPF_index':9,'C2PSA_index':10,'strides':model.stride.tolist(),
        'one_to_many_present':head.cv2 is not None,'one_to_one_present':True,'yaml':deepcopy(cfg)}


def transfer_report(original, model, resume=False):
    source, target = original.float().state_dict(), model.state_dict()
    transferred = intersect_dicts(source, target)
    skipped = sorted(set(target)-set(transferred))
    permitted = {f'model.23.{branch}.{i}.2.{kind}' for branch in ('cv3','one2one_cv3')
                 for i in range(3) for kind in ('weight','bias')}
    if set(source) != set(target) or set(skipped) != (set() if resume else permitted):
        raise ValueError('Unexpected COCO migration mismatch beyond both nc80->nc1 class outputs')
    for k in skipped:
        if source[k].shape[0] != 80 or target[k].shape[0] != 1 or source[k].shape[1:] != target[k].shape[1:]:
            raise ValueError('Class-output shape mismatch has another cause: '+k)
    if not all(torch.equal(target[k],v) for k,v in transferred.items()):
        raise ValueError('Migrated tensors differ before the first optimizer update')
    for branch in ('model.23.cv2.','model.23.one2one_cv2.','model.23.cv3.','model.23.one2one_cv3.'):
        if not any(k.startswith(branch) for k in transferred):
            raise ValueError('Missing matching tensors for detection branch '+branch)
    return {'source_tensors':len(source),'destination_tensors':len(target),'transferred_tensors':len(transferred),
        'transferred_elements':sum(v.numel() for v in transferred.values()),'transferred_keys':sorted(transferred),
        'matched_shapes':{k:list(v.shape) for k,v in transferred.items()},'missing_or_reshaped_keys':skipped,
        'source_only_keys':sorted(set(source)-set(target)),
        'skip_reasons':{k:{'reason':'nc80 -> nc1 class-output dimension only',
            'source_shape':list(source[k].shape),'target_shape':list(target[k].shape)} for k in skipped},
        'pretrained_tensors_loaded':0 if resume else len(transferred),
        'matching_values_verified_before_first_update':True}


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
                    fp, fp_branches = probe(x)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    amp, amp_branches = probe(x)
                if fp.shape != amp.shape or not torch.isfinite(fp).all() or not torch.isfinite(amp).all():
                    raise RuntimeError('Nonfinite or malformed AMP end2end output')
                # Top-k may reorder nearly equal scores across precisions. Compare
                # the SAME anchors in each raw branch, before discontinuous top-k.
                for branch in ('one2many','one2one'):
                    for key in ('boxes','scores'):
                        left,right=fp_branches[branch][key],amp_branches[branch][key]
                        if (not torch.isfinite(left).all() or not torch.isfinite(right).all()
                                or not torch.allclose(left,right.float(),rtol=.1,atol=.5)):
                            raise RuntimeError('Actual-model AMP raw-branch accuracy probe failed: '+branch+'/'+key)
            del probe
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return True

class ComparisonTrainer(DetectionTrainer):
    def __init__(self, *args, run, manifest, identity, config=None, reuse_cache=None, **kwargs):
        self.comparison_run = Path(run)
        self.comparison_manifest = manifest
        self.comparison_identity = identity
        self.comparison_config = deepcopy(recipe() if config is None else config)
        self.comparison_reuse_cache = reuse_cache
        self.comparison_best_epoch = None
        self.comparison_optimizer_steps = self.comparison_optimizer_updates = self.comparison_overflow_skips = 0
        self.comparison_epoch_trace = []
        self.resume_attempt_id = uuid.uuid4().hex[:12]
        from ultralytics.engine import trainer as native_trainer
        from ultralytics.utils import callbacks
        native_trainer.check_amp = strict_amp_probe
        callbacks.add_integration_callbacks = lambda instance: None
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        from support import load_yaml
        data = load_yaml(self.args.data)
        data['channels'] = 3
        return data

    def check_resume(self, overrides):
        self.resume = bool(self.args.resume)
        self._resume_checkpoint = None
        if self.resume:
            path = self.comparison_run/'train/weights/last.pt'
            if Path(str(self.args.resume)).resolve() != path.resolve():
                raise ValueError('Resume only accepts this frozen run last.pt')
            ckpt = torch.load(path,map_location='cpu',weights_only=False)
            validate_checkpoint(ckpt,self.comparison_identity,resume=True,config=self.comparison_config)
            from ultralytics.cfg import get_cfg
            self.args = get_cfg(overrides=ckpt['train_args'])
            self.args.model = self.args.resume = str(path)
            self._resume_checkpoint = ckpt
        for key, expected in native_recipe(self.comparison_config).items():
            if key in ('model','resume'): continue
            actual = getattr(self.args,key)
            if (str(actual) != str(expected) if key == 'device' else actual != expected):
                raise ValueError('Native args differ from frozen run: '+key)

    def setup_model(self):
        source = Path(os.environ['YOLO26M_SOURCE'])
        if self.resume:
            from ultralytics.nn.tasks import DetectionModel
            self.model = DetectionModel(deepcopy(model_config(source)),nc=1,verbose=False)
            self.model.load_state_dict(self._resume_checkpoint['comparison_training_state']['model'],strict=True)
            write_json(self.comparison_run/('resume_model_'+self.resume_attempt_id+'.json'),
                {'identity':self.comparison_identity,'restored':'full unfused dual-branch FP32 training state',
                 'model':model_identity(self.model,1),'external_pretrained_tensors_loaded':0})
            return self._resume_checkpoint
        from run import load_initial_model
        from support import checked_weight
        self.model, record = load_initial_model(source,checked_weight(self.args.model),self.comparison_config)
        path = self.comparison_run/'initialization.json'
        if path.exists(): raise FileExistsError('Original initialization record is immutable')
        write_json(path,{**record,'resume':False,'identity':self.comparison_identity})
        return None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None: raise ValueError('Verified COCO weights required')
        model_identity(weights,80)
        original_yaml = deepcopy(weights.yaml)
        model = super().get_model(deepcopy(cfg if cfg is not None else weights.yaml),weights,verbose)
        model_identity(model,1); transfer_report(weights,model)
        if weights.yaml != original_yaml: raise ValueError('Source COCO YAML was mutated')
        return model

    def get_validator(self):
        self.loss_names = 'box_loss','cls_loss','dfl_loss'
        return CompleteValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

    def build_dataset(self,img_path,mode='train',batch=None):
        return IsolatedDataset(img_path=img_path,imgsz=self.args.imgsz,batch_size=self.args.batch,
            augment=mode=='train',hyp=copy(self.args),rect=False,cache=False,stride=32,pad=0.,task='detect',
            data=self.data,manifest=self.comparison_manifest,split=mode,cache_root=self.comparison_run/'cache',
            prefix=mode+': ',cutmix_probability=self.comparison_config['cutmix'],reuse_cache=self.comparison_reuse_cache)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        physical_batch = self.args.batch
        dataset = self.build_dataset(dataset_path,mode,physical_batch)
        loader = build_dataloader(dataset,physical_batch,self.args.workers if mode=='train' else 0,shuffle=mode=='train',rank=-1)
        loader.reset = lambda: reset_workers(loader)
        return loader

    def build_optimizer(self,model,name='MuSGD',lr=.01,momentum=.937,decay=.0005,iterations=1e5):
        from optimizer import normalize_optimizer, annotate_groups, optimizer_identity
        name = normalize_optimizer(name)  # Native use_muon must see the exact name before grouping.
        optimizer = super().build_optimizer(model,name,lr,momentum,decay,iterations)
        annotate_groups(optimizer,model,name,lr,decay)
        cfg = getattr(self,'comparison_config',vars(self.args))
        self.comparison_optimizer_identity = optimizer_identity(optimizer,model,cfg)
        return optimizer

    def preprocess_batch(self,batch):
        from optimizer import group_values
        nb = len(self.train_loader)
        ni = getattr(self,'comparison_seen_batches',self.start_epoch*nb)
        self.comparison_seen_batches = ni+1
        nw = max(round(self.args.warmup_epochs*nb),100) if self.args.warmup_epochs>0 else -1
        if ni <= nw:
            with (self.comparison_run/'warmup_trace.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(canonical({'global_batch':ni,'epoch':self.epoch+1,'warmup_iterations':nw,
                    'accumulate':self.accumulate,'groups':group_values(self.optimizer)}).decode())
        return super().preprocess_batch(batch)

    def validate(self):
        metrics = self.validator(self)
        native_fitness = metrics.pop('fitness')
        fitness = float(self.validator.metrics.box.map)
        if not np.isfinite(fitness): raise ValueError('Nonfinite validation mAP50-95')
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness; self.comparison_best_epoch = self.epoch+1
        metrics['metrics/native_fitness'] = native_fitness
        write_json(self.comparison_run/'training_progress.json',{'completed_epoch':self.epoch+1,
            'best_epoch':self.comparison_best_epoch,'selection_metric':'one_to_one_training_val_mAP50_95_full_precision',
            'fitness':fitness,'best_fitness':self.best_fitness,'native_fitness':native_fitness})
        write_json(self.comparison_run/'native_validation.json',{**self.validator.actual_precision,
            'imgsz':self.args.imgsz,'batch':self.test_loader.batch_size,'workers':0,'conf':self.validator.args.conf,
            'iou':.7,'NMS_IoU':'NOT_APPLICABLE','max_det':300,'branch':'one-to-one','extra_IoU_NMS':False,
            'selection':'unrounded native mAP50-95; ties select later epoch'})
        return metrics,fitness

    def _handle_nan_recovery(self,epoch):
        if not torch.isfinite(self.loss).all() or not np.isfinite(self.fitness):
            raise FloatingPointError('Nonfinite loss/fitness; no automatic rollback or loss-schedule reset')
        return False

    def optimizer_step(self):
        before = self.scaler.get_scale()
        super().optimizer_step()
        overflow = self.scaler.get_scale() < before
        self.comparison_optimizer_steps += 1
        self.comparison_optimizer_updates += int(not overflow)
        self.comparison_overflow_skips += int(overflow)

    def save_model(self):
        from optimizer import optimizer_identity, verify_buffers
        model = unwrap_model(self.model)
        schedule = criterion_state(model.criterion)
        if schedule['updates'] != self.epoch+1: raise ValueError('Native E2ELoss epoch update timing differs')
        optimizer_id = optimizer_identity(self.optimizer,model,self.comparison_config)
        if optimizer_id != self.comparison_optimizer_identity: raise ValueError('Optimizer identity changed')
        self.comparison_epoch_trace.append(epoch_trace(self))
        state = {'model':cpu_state(model.state_dict()),'optimizer':cpu_state(self.optimizer.state_dict()),
            'ema':cpu_state(self.ema.ema.state_dict()),'scaler':self.scaler.state_dict(),
            'scheduler':self.scheduler.state_dict(),'rng':capture_rng(),'criterion':schedule,
            'optimizer_identity':optimizer_id,'optimizer_buffers':verify_buffers(self.optimizer),
            'stopper':vars(self.stopper).copy(),'optimizer_steps':self.comparison_optimizer_steps,
            'optimizer_updates':self.comparison_optimizer_updates,'overflow_skips':self.comparison_overflow_skips,
            'augmentation_counters':list(self.train_loader.dataset.augmentation_counts[:]),
            'augmentation_closed':self.train_loader.dataset.augmentation_closed,
            'gradients':{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None},
            'last_opt_step':getattr(self,'comparison_last_opt_step',-1),'accumulate':self.accumulate,
            'loader_generators':{mode:loader.generator.get_state() for mode,loader in [('train',self.train_loader),('val',self.test_loader)]}}
        buffer = io.BytesIO()
        torch.save({'checkpoint_schema':'yolo26m_configurable_v1','epoch':self.epoch,'completed':bool(self.stop),
            'best_fitness':self.best_fitness,'model':None,'ema':clean_model_copy(self.ema.ema),'updates':self.ema.updates,
            'optimizer':state['optimizer'],'scaler':state['scaler'],'train_args':vars(self.args),
            'train_metrics':{**self.metrics,'fitness':self.fitness},'train_results':self.read_results_csv(),
            'date':datetime.now().isoformat(),'version':ultralytics.__version__,'license':'AGPL-3.0',
            'comparison_identity':self.comparison_identity,'comparison_best_epoch':self.comparison_best_epoch,
            'comparison_epoch_trace':self.comparison_epoch_trace,'comparison_initialization':read_json(self.comparison_run/'initialization.json'),
            'pilot_recipe':self.comparison_config,'comparison_training_state':state},buffer)
        raw = buffer.getvalue(); atomic_bytes(self.last,raw)
        if self.best_fitness == self.fitness: atomic_bytes(self.best,raw)
        write_json(self.comparison_run/'checkpoint_integrity.json',{'epoch':self.epoch+1,'best_epoch':self.comparison_best_epoch,
            'last_sha256':digest(raw),'best_sha256':digest(raw) if self.best_fitness==self.fitness else sha256(self.best)})
        write_json(self.comparison_run/'checkpoint_state.json',{'completed_epoch':self.epoch+1,'criterion':schedule,
            'optimizer_identity':optimizer_id,'optimizer_buffers':state['optimizer_buffers'],
            'last_opt_step':state['last_opt_step'],'accumulate':state['accumulate'],'completed':bool(self.stop)})

    def resume_training(self,ckpt):
        if self.resume:
            validate_checkpoint(ckpt,self.comparison_identity,resume=True,config=self.comparison_config)
            if self.comparison_optimizer_identity != ckpt['comparison_training_state']['optimizer_identity']:
                raise ValueError('Resume optimizer identity differs before native state loading')
        super().resume_training(ckpt)
        if self.resume:
            self.restore_training_state(ckpt)
            self.comparison_best_epoch = ckpt['comparison_best_epoch']
            def resume_start(t):
                state=ckpt['comparison_training_state']
                for mode,loader in [('train',t.train_loader),('val',t.test_loader)]:
                    loader.generator.set_state(state['loader_generators'][mode]); loader.reset()
                restore_rng(state['rng'])
                t._resume_checkpoint=None
            self.add_callback('on_train_start',resume_start)

    def restore_training_state(self,ckpt):
        from optimizer import optimizer_identity, verify_buffers
        state = ckpt['comparison_training_state']
        if self.comparison_optimizer_identity != state['optimizer_identity']:
            raise ValueError('Resume optimizer class/coefficients/groups/LR multiplier/warmup identity differs')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if optimizer_identity(self.optimizer,self.model,self.comparison_config) != state['optimizer_identity']:
            raise ValueError('Loaded optimizer identity differs')
        verify_buffers(self.optimizer)
        self.ema.ema.load_state_dict(state['ema'],strict=True)
        self.scaler.load_state_dict(state['scaler']); self.scheduler.load_state_dict(state['scheduler'])
        self.stopper.__dict__.update(state['stopper'])
        self.model.criterion = self.model.init_criterion()
        restore_criterion(self.model.criterion,state['criterion'])
        self.comparison_last_opt_step=state['last_opt_step']; self.accumulate=state['accumulate']
        for name,p in self.model.named_parameters():
            grad=state['gradients'].get(name); p.grad=None if grad is None else grad.to(device=p.device,dtype=p.dtype)
        self.comparison_restored_gradients=True
        self.comparison_optimizer_steps=state['optimizer_steps']; self.comparison_optimizer_updates=state['optimizer_updates']
        self.comparison_overflow_skips=state['overflow_skips']; self.comparison_epoch_trace=ckpt['comparison_epoch_trace']
        with self.train_loader.dataset.augmentation_counts.get_lock():
            self.train_loader.dataset.augmentation_counts[:]=state['augmentation_counters']
        if state['augmentation_closed'] and not self.train_loader.dataset.augmentation_closed:
            self._close_dataloader_mosaic()
        if state['augmentation_closed'] != self.train_loader.dataset.augmentation_closed:
            raise ValueError('Resume augmentation phase differs')
        if read_json(self.comparison_run/'initialization.json') != ckpt['comparison_initialization']:
            raise ValueError('Original initialization record differs')
        write_json(self.comparison_run/'resume_state.json',{'epoch':ckpt['epoch']+1,'criterion':criterion_state(self.model.criterion),
            'restored':['FP32 dual-branch model','native optimizer and both MuSGD buffers','FP32 EMA','AMP scaler',
                        'scheduler','E2ELoss epoch schedule','RNG','stopper','pending gradients','loader generators','augmentation phase/counters'],
            'optimizer_identity':state['optimizer_identity'],'optimizer_steps':self.comparison_optimizer_steps,
            'optimizer_updates':self.comparison_optimizer_updates,'overflow_skips':self.comparison_overflow_skips,
            'original_initialization_preserved':True,'last_opt_step':state['last_opt_step'],'accumulate':state['accumulate'],
            'limits':'Epoch-boundary recovery; worker RNG, prefetch buffers, sampler cursor and Mosaic buffer are not serialized. No bitwise continuation claim.'})

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        write_json(self.comparison_run/'augmentation_closed.json',{'zero_based_epoch':self.epochs-self.args.close_mosaic,
            'closed':['mosaic','mixup','cutmix','copy_paste'],'pipeline':self.train_loader.dataset.transform_report,
            'counts_at_rebuild':snapshot(self.train_loader.dataset),'worker_reset':'native epoch trigger or explicit resume rebuild'})

    def final_eval(self):
        print('Native training finished; independent FP32 public val follows; test requires finalize.',flush=True)


def epoch_trace(t):
    from optimizer import group_values
    return {'epoch':t.epoch+1,'accumulate':t.accumulate,'amp':bool(t.amp),'batch':t.batch_size,
        'learning_rates':t.lr,'optimizer_groups':group_values(t.optimizer),'best_epoch':t.comparison_best_epoch,'fitness':t.fitness,
        'optimizer_steps_cumulative':t.comparison_optimizer_steps,'optimizer_updates_cumulative':t.comparison_optimizer_updates,
        'amp_overflow_skips_cumulative':t.comparison_overflow_skips,'criterion':criterion_state(t.model.criterion),
        'E2ELoss_update_timing':'once at native epoch end after batches, before EMA validation and save',
        'augmentation':t.train_loader.dataset.transform_report['probabilities'],'augmentation_closed':t.train_loader.dataset.augmentation_closed,
        'augmentation_counts_cumulative':snapshot(t.train_loader.dataset),'losses':t.tloss.detach().cpu().tolist()}


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

CRITERION_FIELDS=('updates','total','o2m','o2o','o2m_copy','final_o2m')

def criterion_state(criterion):
    from ultralytics.utils.loss import E2ELoss
    if type(criterion) is not E2ELoss:
        raise ValueError('Expected native YOLO26 E2ELoss')
    return {k:getattr(criterion,k) for k in CRITERION_FIELDS}

def restore_criterion(criterion,state):
    expected=criterion.decay(state['updates'])
    if abs(state['o2m']-expected)>1e-12 or abs(state['o2m']+state['o2o']-1)>1e-12:
        raise ValueError('Loss schedule state inconsistent with frozen epochs')
    for k in CRITERION_FIELDS:
        setattr(criterion,k,state[k])

def clean_model_copy(model):
    saved=deepcopy(model).float().cpu()
    if hasattr(saved,'criterion'):
        del saved.criterion  # loss contains device tensors and lambda assigner helpers
    return saved
