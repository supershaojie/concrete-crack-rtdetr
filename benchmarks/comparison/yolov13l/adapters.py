"""Narrow data/trainer adapters; official YOLOv13 model, loss, DFL and assigner stay intact.

HSV formula is derived from the mother AGPL-3.0 source at a0459d6a652cb702699087c88fa39a3e4c4087ec.
Checkpoint serialization follows official v8.3.63 BaseTrainer.save_model (AGPL-3.0).
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
                     read_json, write_json, initialization_record, model_yaml, model_config,
                     no_external_initialization, sha256, validate_checkpoint)

if 'ultralytics' not in sys.modules:  # spawned workers must also import the official tree
    configure(Path(os.environ.get('YOLOV13L_RUNTIME', ROOT / '.runtime/yolov13l-scratch/worker')),
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
        identity = digest(canonical({'version': 'yolov13l_header_labels_v1',
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
                cached = {'identity': identity, 'format': 'yolov13l_header_labels_v1', 'entries': entries}
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
        # Enforce square padding explicitly, independent of framework auto-padding heuristics.
        letterbox = augment.LetterBox(self.imgsz, auto=False, stride=self.model.stride)
        return [letterbox(image=im) for im in images]

    def postprocess(self, preds, img, orig_imgs):
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops
        # Same official NMS, without its time-budget early break (which can leave
        # later images empty under contention). No changes to IoU/ranking rules.
        predictions = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det, max_nms=30000,
            classes=self.args.classes, max_time_img=float('inf'))
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
            max_det=self.args.max_det, max_nms=30000, max_time_img=float('inf'))


def model_identity(model, nc):
    from collections import Counter
    from ultralytics.nn.modules import DWConv
    head = model.model[-1]
    params = sum(p.numel() for p in model.parameters())
    cfg = model_config(Path(os.environ['YOLOV13L_SOURCE']))
    graph = cfg['backbone']+cfg['head']
    actual = [(m.f, type(m).__name__) for m in model.model]
    expected_graph = [(row[0], row[2].replace('nn.','')) for row in graph]
    counts = Counter(type(m).__name__ for m in model.modules())
    expected_counts = {'AAttn':16, 'ABlock':16, 'HyperACE':1, 'FullPAD_Tunnel':7,
                       'DSC3k2':6, 'DSConv':58, 'A2C2f':2, 'DownsampleConv':1}
    if (nc != 1 or type(head) is not Detect or head.nc != 1 or model.yaml.get('scale') != 'l'
            or params != 27566874 or actual != expected_graph
            or any(counts[k] != v for k,v in expected_counts.items())
            or head.legacy or not all(type(branch[0][0]) is DWConv for branch in head.cv3)
            or model.stride.tolist() != [8.0,16.0,32.0]):
        raise ValueError('Expected official YOLOv13l nc=1 full topology; observed '+str((head.nc, params, counts)))
    return {'scale':'l','nc':1,'parameters_unfused':params,'head':type(head).__module__+'.'+type(head).__name__,
            'legacy':head.legacy,'reg_max':head.reg_max,'strides':model.stride.tolist(),'yaml':model.yaml,
            'module_counts':{k:counts[k] for k in expected_counts}, 'graph':actual}


def native_initialization(model):
    from ultralytics.nn.modules.block import FullPAD_Tunnel, ABlock, A2C2f
    gates = [m.gate.item() for m in model.modules() if isinstance(m,FullPAD_Tunnel)]
    if gates != [0.0]*7:
        raise ValueError('Native FullPAD gate initialization changed')
    dfl=model.model[-1].dfl.conv.weight
    if dfl.requires_grad or not torch.equal(dfl.flatten(),torch.arange(16,dtype=dfl.dtype,device=dfl.device)):
        raise ValueError('Native fixed DFL changed')
    bns=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    if not all(torch.all(m.weight==1) and torch.all(m.bias==0) and m.eps==.001 and m.momentum==.03 for m in bns):
        raise ValueError('Official BN affine/eps/momentum changed')
    # Detect bias_init happens after the dummy stride forward; no blanket reinitialization.
    import math
    for box,cls,stride in zip(model.model[-1].cv2,model.model[-1].cv3,model.stride):
        if not torch.all(box[-1].bias==1) or not torch.allclose(cls[-1].bias,torch.full_like(cls[-1].bias,math.log(5/(640/float(stride))**2))):
            raise ValueError('Native Detect bias priors changed')
    return {'FullPAD_gates':gates,'fixed_DFL_arange_16':True,'BN_affine_and_hyperparameters':True,
            'Detect_bias_priors':True,'ABlock_initialization':'official apply(_init_weights): Conv trunc_normal std=.02, bias=0',
            'ABlock_conv_std_observed':[float(m.attn.qk.conv.weight.std()) for m in model.modules() if isinstance(m,ABlock)],
            'A2C2f_gamma':[float(m.gamma.mean()) for m in model.modules() if isinstance(m,A2C2f) and m.gamma is not None],
            'HyperACE_prototypes':'official xavier_uniform_; no external transfer'}


AMP_PROBE_RECORD = None

def strict_amp_probe(model):
    """Actual-model copy; original model/BN/optimizer/EMA/scaler and every RNG are untouched."""
    global AMP_PROBE_RECORD
    from backend import ArithmeticAudit, native_fp32
    device=next(model.parameters()).device
    if device.type!='cuda':
        raise RuntimeError('Formal recipe requires CUDA device=0 and AMP')
    py_state,np_state=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            probe=deepcopy(model).float().eval()
            x=torch.linspace(0,1,3*64*64,device=device).reshape(1,3,64,64)
            with torch.no_grad():
                with native_fp32(), ArithmeticAudit(require_fp32=True) as fp_audit:
                    fp=probe(x)[0]
                with torch.autocast('cuda',dtype=torch.float16), ArithmeticAudit() as amp_audit:
                    amp=probe(x)[0]
                if (fp.shape!=amp.shape or not torch.isfinite(fp).all() or not torch.isfinite(amp).all()
                        or not torch.allclose(fp,amp.float(),rtol=.1,atol=.5)):
                    raise RuntimeError('Actual-model AMP accuracy probe failed; no silent recipe changes')
            AMP_PROBE_RECORD={'input':[1,3,64,64],'rtol':.1,'atol':.5,
                              'max_abs_error':float((fp-amp.float()).abs().max()),
                              'fp32':fp_audit.report(),'amp':amp_audit.report(),
                              'flash_parity':'NOT_VERIFIED; native backend fixed'}
            del probe
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
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

    @property
    def amp_probe_record(self):
        return AMP_PROBE_RECORD

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
        if self.comparison_identity.get('initialization_type') != 'random':
            raise ValueError('Only this model random run is supported')
        if not self.args.resume:
            source=Path(os.environ['YOLOV13L_SOURCE'])
            if self.args.pretrained is not False or str(self.model)!=str(model_yaml(source)):
                raise ValueError('New scratch requires verified official YAML and pretrained=False')
            self.model=self.get_model(model_config(source),weights=None)
            return None
        ckpt=super().setup_model()
        validate_checkpoint(ckpt,self.comparison_identity,resume=True)
        if ckpt['comparison_initialization_sha256']!=sha256(self.comparison_run/'initialization.json'):
            raise ValueError('Original initialization record changed')
        self.model.load_state_dict(ckpt['comparison_model_state'],strict=True)
        return ckpt

    def get_model(self,cfg=None,weights=None,verbose=True):
        if self.comparison_identity.get('initialization_type')!='random':
            raise ValueError('Foreign initialization is forbidden')
        source=Path(os.environ['YOLOV13L_SOURCE'])
        if not self.args.resume:
            if weights is not None or self.args.pretrained is not False or cfg!=model_config(source):
                raise ValueError('Fresh scratch requires exact official L YAML and no supplied weights')
            with no_external_initialization():
                model=super().get_model(cfg,weights=None,verbose=verbose)
            record={**model_identity(model,1),**initialization_record(source),
                    'native_initialization_checks':native_initialization(model),
                    'transferred_tensors':0,'transferred_elements':0,'transferred_keys':[],
                    'destination_tensors':len(model.state_dict()),'weights_argument':None,'resume':False,
                    'construction':'setup_model -> get_model(weights=None) -> official DetectionModel with loading interception',
                    'identity':self.comparison_identity,'expanded_train_args':vars(self.args),
                    'expanded_args_sha256':digest(canonical(vars(self.args))),
                    'fixed_parameters':[n for n,p in model.named_parameters() if not p.requires_grad]}
            path=self.comparison_run/'initialization.json'
            if path.exists():
                raise FileExistsError('Original initialization record is immutable')
            write_json(path,record)
        else:
            if weights is None:
                raise ValueError('Explicit resume needs the verified same-run checkpoint')
            model=super().get_model(cfg,weights,verbose)
            record={**model_identity(model,1),'restored_run':self.comparison_identity,
                    'original_initialization_sha256':sha256(self.comparison_run/'initialization.json')}
            import uuid
            write_json(self.comparison_run/('resume_model_'+uuid.uuid4().hex+'.json'),record)
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
        # Keep FP32 training/optimizer/EMA state and scaler, not only quantized EMA.
        # These are trusted local run checkpoints, never arbitrary pretrained assets.
        from ultralytics.utils.torch_utils import de_parallel
        model=de_parallel(self.model)
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
             'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        stopper={'best_epoch':self.stopper.best_epoch,'best_fitness':self.stopper.best_fitness,
                 'possible_stop':self.stopper.possible_stop,'patience':self.stopper.patience}
        checkpoint={'epoch':self.epoch,'best_fitness':self.best_fitness,'model':None,
            'ema':deepcopy(self.ema.ema).float().cpu(),'updates':self.ema.updates,
            'optimizer':deepcopy(self.optimizer.state_dict()),'scaler':self.scaler.state_dict(),
            'train_args':vars(self.args),'train_metrics':{**self.metrics,'fitness':self.fitness},
            'train_results':self.read_results_csv(),'date':datetime.now().isoformat(),
            'version':ultralytics.__version__,'license':'AGPL-3.0',
            'comparison_identity':self.comparison_identity,'comparison_best_epoch':self.comparison_best_epoch,
            'comparison_initialization_sha256':sha256(self.comparison_run/'initialization.json'),
            'comparison_stopper':stopper,'comparison_scheduler':self.scheduler.state_dict(),'comparison_rng':rng,
            'comparison_model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
        buffer=io.BytesIO();torch.save(checkpoint,buffer);raw=buffer.getvalue()
        atomic_bytes(self.last,raw)
        if self.best_fitness==self.fitness:
            atomic_bytes(self.best,raw)
        write_json(self.comparison_run/'checkpoint_pair.json', {
            'last_sha256':sha256(self.last),'best_sha256':sha256(self.best),
            'completed_epoch':self.epoch+1,'best_epoch':self.comparison_best_epoch})

    def resume_training(self,ckpt):
        if ckpt is not None and self.resume:
            validate_checkpoint(ckpt,self.comparison_identity,resume=True)
        super().resume_training(ckpt)
        if self.resume:
            self.scaler.load_state_dict(ckpt['scaler'])
            self.scheduler.load_state_dict(ckpt['comparison_scheduler'])
            self.comparison_best_epoch=ckpt['comparison_best_epoch']
            self.stopper.best_fitness=ckpt['comparison_stopper']['best_fitness']
            self.stopper.best_epoch=ckpt['comparison_stopper']['best_epoch']
            self.stopper.possible_stop=ckpt['comparison_stopper']['possible_stop']
            if self.start_epoch>self.epochs-self.args.close_mosaic:
                self.train_loader.reset()
            rng=ckpt['comparison_rng']
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'].cpu())
            if rng['cuda']:
                torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])

    def final_eval(self):
        # Every epoch already validated. Preserve resumable best/last; the next step
        # exports the same selected best with explicit FP32, then invokes the public evaluator.
        print('Native training finished; separate FP32 public export/evaluation follows.', flush=True)
