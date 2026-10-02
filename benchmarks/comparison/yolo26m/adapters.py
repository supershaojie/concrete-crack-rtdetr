"""Narrow data/trainer adapters; official YOLO26 model, E2ELoss and assigners stay intact.

HSV formula is derived from the mother AGPL-3.0 source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Checkpoint serialization follows official v8.4.0 BaseTrainer.save_model (AGPL-3.0).
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
                     read_json, write_json, initialization_record, model_yaml, model_config, validate_checkpoint)

if 'ultralytics' not in sys.modules:  # spawned workers must also import the official tree
    configure(Path(os.environ.get('YOLO26M_RUNTIME', ROOT / '.runtime/yolo26m-scratch/worker')),
              Path(os.environ.get('YOLO26M_SOURCE', SOURCE)))
import ultralytics
if Path(ultralytics.__file__).resolve() != Path(os.environ['YOLO26M_SOURCE']) / 'ultralytics/__init__.py':
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
from ultralytics.utils.torch_utils import unwrap_model
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
        identity = digest(canonical({'version': 'yolo26m_header_labels_v1',
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
                cached = {'identity': identity, 'format': 'yolo26m_header_labels_v1', 'entries': entries}
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
            transforms = comparison_transforms(self, self.imgsz, hyp)
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


def comparison_transforms(dataset, imgsz, hyp):
    # Same effective detection order as the reference v8 adapter. CutMix is
    # explicitly absent (p=0) so it cannot introduce a new random-number draw.
    mosaic = augment.Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = augment.RandomPerspective(degrees=hyp.degrees, translate=hyp.translate,
        scale=hyp.scale, shear=hyp.shear, perspective=hyp.perspective, pre_transform=None)
    pre = augment.Compose([mosaic, augment.CopyPaste(p=0.0, mode='flip'), affine])
    return augment.Compose([pre, augment.MixUp(dataset, pre_transform=pre, p=hyp.mixup),
        NoAlbumentations(), augment.RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        augment.RandomFlip(direction='vertical', p=hyp.flipud),
        augment.RandomFlip(direction='horizontal', p=hyp.fliplr)])


def end2end_tensor(preds):
    value = preds[0] if isinstance(preds, (tuple,list)) else preds
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 6:
        raise ValueError('Expected native one-to-one postprocessed [B,N,6], not raw branch dict or BCN')
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite native one-to-one output')
    return value


class SquarePredictor(DetectionPredictor):
    def pre_transform(self, images):
        letterbox = augment.LetterBox(self.imgsz, auto=False, stride=self.model.stride)
        return [letterbox(image=im) for im in images]

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if not self.model.end2end:
            raise ValueError('YOLO26 predictor requires native end2end=True')
        end2end_tensor(preds)
        # Official nms.non_max_suppression returns early on end2end: only
        # confidence/class filtering and max_det; no IoU suppression runs.
        return super().postprocess(preds, img, orig_imgs, **kwargs)


class CompleteValidator(DetectionValidator):
    def postprocess(self, preds):
        if not self.end2end:
            raise ValueError('Training validation requires one-to-one output')
        end2end_tensor(preds)
        # v8.4.0 returns a list of dictionaries (bboxes/conf/cls/extra).
        return super().postprocess(preds)


def model_identity(model, nc, deployed=False):
    head = model.model[-1]
    params = sum(p.numel() for p in model.parameters())
    cfg = model.yaml
    if (type(head) is not Detect or head.nc != nc or cfg.get('scale') != 'm'
            or cfg['scales']['m'] != [0.5,1.0,512] or head.reg_max != 1
            or not head.end2end or not isinstance(head.dfl, torch.nn.Identity)
            or not hasattr(head, 'one2one_cv2') or not hasattr(head, 'one2one_cv3')):
        raise ValueError('Expected pinned official YOLO26m end2end head')
    if not deployed and (head.cv2 is None or head.cv3 is None or not any(
            isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())):
        raise ValueError('Training requires complete unfused one-to-many and one-to-one structure')
    return {'scale':'m', 'scale_values':cfg['scales']['m'], 'nc':head.nc,
            'parameters_unfused' if not deployed else 'parameters_deployed':params,
            'head':type(head).__module__+'.'+type(head).__name__, 'end2end':head.end2end,
            'reg_max':head.reg_max, 'strides':model.stride.tolist(), 'yaml':cfg,
            'one_to_many_present':head.cv2 is not None, 'one_to_one_present':True}


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
    def __init__(self, *args, run, manifest, identity, **kwargs):
        self.comparison_run = Path(run)
        self.resume_attempt_id = uuid.uuid4().hex[:12]
        self.comparison_manifest = manifest
        self.comparison_identity = identity
        self.comparison_best_epoch = None
        from ultralytics.engine import trainer as native_trainer
        from ultralytics.utils import callbacks
        native_trainer.check_amp = strict_amp_probe
        callbacks.add_integration_callbacks = lambda instance: None
        super().__init__(*args, **kwargs)

    def get_dataset(self):
        from support import load_yaml
        data=load_yaml(self.args.data)  # v8.4.0 BaseTrainer expects the data dict
        data['channels']=3
        return data

    def check_resume(self, overrides):
        self.resume = bool(self.args.resume)
        self._resume_checkpoint = None
        if not self.resume:
            return
        path = self.comparison_run / 'train/weights/last.pt'
        if Path(str(self.args.resume)).resolve() != path.resolve():
            raise ValueError('Resume only accepts this run last.pt')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        validate_checkpoint(ckpt, self.comparison_identity, resume=True)
        from ultralytics.cfg import get_cfg
        self.args = get_cfg(overrides=ckpt['train_args'])
        self.args.model = self.args.resume = str(path)
        self._resume_checkpoint = ckpt

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
        source = Path(os.environ['YOLO26M_SOURCE'])
        if not self.resume and (self.args.pretrained is not False or str(self.model) != str(model_yaml(source))):
            raise ValueError('Scratch requires verified official YAML and pretrained=False')
        self.model = self.get_model(model_config(source), weights=None)
        if self.resume:
            self.model.load_state_dict(self._resume_checkpoint['training_model_state'], strict=True)
            write_json(self.comparison_run / ('resume_model_' + self.resume_attempt_id + '.json'),
                {'identity':self.comparison_identity, 'restored':'full unfused training weights in FP32',
                 'model':model_identity(self.model,1), 'external_pretrained_tensors_loaded':0})
        return self._resume_checkpoint

    def get_model(self, cfg=None, weights=None, verbose=True):
        source = Path(os.environ['YOLO26M_SOURCE'])
        if weights is not None or self.args.pretrained is not False or cfg != model_config(source):
            raise ValueError('Only official M structure with weights=None and pretrained=False is accepted')
        model = super().get_model(cfg, weights=None, verbose=verbose)
        record = {**model_identity(model,1), **initialization_record(source),
            'identity':self.comparison_identity, 'transferred_tensors':0, 'transferred_elements':0,
            'transferred_keys':[], 'destination_tensors':len(model.state_dict()),
            'weights_argument':None, 'resume':False,
            'construction':'setup_model -> get_model(weights=None) -> official DetectionModel',
            'fixed_parameters':[n for n,p in model.named_parameters() if not p.requires_grad]}
        if not self.resume:
            path=self.comparison_run/'initialization.json'
            if path.exists():
                raise FileExistsError('Original initialization audit already exists')
            write_json(path,record)
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
            'best_fitness': self.best_fitness, 'native_fitness_full_map_rounded_in_return': native_fitness})
        return metrics, fitness

    def _handle_nan_recovery(self, epoch):
        # Native automatic recovery resets loss scheduling; explicit failure keeps
        # the frozen run reviewable and leaves any recovery to verified resume.
        if not torch.isfinite(self.loss).all() or not np.isfinite(self.fitness):
            raise FloatingPointError('Nonfinite training loss/fitness; explicit inspected resume required')
        return False

    def save_model(self):
        model=unwrap_model(self.model)
        criterion=criterion_state(model.criterion)
        if criterion['updates'] != self.epoch+1:
            raise ValueError('Native E2ELoss update timing differs')
        buffer=io.BytesIO()
        torch.save({'checkpoint_schema':'yolo26m_scratch_v1', 'epoch':self.epoch,
            'completed':bool(self.stop), 'best_fitness':self.best_fitness, 'model':None,
            'training_model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'ema':clean_model_copy(self.ema.ema), 'updates':self.ema.updates,
            'optimizer':deepcopy(self.optimizer.state_dict()), 'scaler':self.scaler.state_dict(),
            'scheduler':self.scheduler.state_dict(), 'criterion_state':criterion,
            'stopper_state':vars(self.stopper).copy(),
            'gradient_state':{k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None},
            'last_opt_step':getattr(self,'comparison_last_opt_step',-1), 'accumulate':self.accumulate,
            'rng_state':capture_rng(),
            'loader_state':{mode:getattr(loader,'generator').get_state() for mode,loader in
                            [('train',self.train_loader),('val',self.test_loader)]},
            'augmentation_state':{'phase':'closed' if self.epoch >= self.epochs-self.args.close_mosaic else 'active'},
            'train_args':vars(self.args), 'train_metrics':{**self.metrics,'fitness':self.fitness},
            'train_results':self.read_results_csv(), 'date':datetime.now().isoformat(),
            'version':ultralytics.__version__, 'license':'AGPL-3.0',
            'comparison_identity':self.comparison_identity, 'comparison_best_epoch':self.comparison_best_epoch},buffer)
        raw=buffer.getvalue()
        atomic_bytes(self.last,raw)
        if self.best_fitness == self.fitness:
            atomic_bytes(self.best,raw)
        write_json(self.comparison_run/'checkpoint_state.json',{
            'completed_epoch':self.epoch+1, 'best_epoch':self.comparison_best_epoch,
            'criterion':criterion, 'scheduler':self.scheduler.state_dict(),
            'stopper':vars(self.stopper), 'ema_updates':self.ema.updates,
            'scaler':self.scaler.state_dict(), 'completed':bool(self.stop)})

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if not self.resume:
            return
        validate_checkpoint(ckpt,self.comparison_identity,resume=True)
        self.comparison_best_epoch=ckpt['comparison_best_epoch']
        self.stopper.__dict__.update(ckpt['stopper_state'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.model.criterion=self.model.init_criterion()
        restore_criterion(self.model.criterion,ckpt['criterion_state'])
        self.comparison_last_opt_step=ckpt['last_opt_step']
        self.accumulate=ckpt['accumulate']
        for name,p in self.model.named_parameters():
            grad=ckpt['gradient_state'].get(name)
            p.grad=None if grad is None else grad.to(device=p.device,dtype=p.dtype)
        self.comparison_restored_gradients=True
        def restore_at_start(trainer):
            for mode,loader in [('train',trainer.train_loader),('val',trainer.test_loader)]:
                loader.generator.set_state(ckpt['loader_state'][mode])
                loader.reset()
            restore_rng(ckpt['rng_state'])
            write_json(self.comparison_run/('resume_state_'+self.resume_attempt_id+'.json'),{
                'completed_epoch':self.start_epoch,'next_epoch':self.start_epoch+1,
                'criterion':criterion_state(self.model.criterion), 'scheduler':self.scheduler.state_dict(),
                'scaler':self.scaler.state_dict(),'stopper':vars(self.stopper),
                'ema_updates':self.ema.updates,'last_opt_step':self.comparison_last_opt_step,
                'restored_gradient_tensors':len(ckpt['gradient_state']),
                'rng':'Python/NumPy/CPU/CUDA restored; loader generators restored with new iterators',
                'limits':'Worker RNG, prefetched batches, sampler cursor and in-memory Mosaic buffer are not serialized; no bitwise continuation claim'})
            self._resume_checkpoint=None
        self.add_callback('on_train_start',restore_at_start)

    def final_eval(self):
        # Every epoch already validated. Preserve resumable best/last; the next step
        # exports the same selected best with explicit FP32, then invokes the public evaluator.
        print('Native training finished; separate FP32 public export/evaluation follows.', flush=True)


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

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def describe_transforms(transform):
    row={'type':type(transform).__module__+'.'+type(transform).__name__}
    for key in ('p','direction','hgain','sgain','vgain','degrees','translate','scale',
                'shear','perspective','n','border','buffer_enabled','normalize','bbox_format','bgr'):
        value=getattr(transform,key,None)
        if value is not None:
            row[key]=value
    if hasattr(transform,'transforms'):
        row['children']=[describe_transforms(t) for t in transform.transforms]
    if getattr(transform,'pre_transform',None) is not None:
        row['pre_transform']=describe_transforms(transform.pre_transform)
    return row
