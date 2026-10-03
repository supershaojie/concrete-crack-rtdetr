"""Narrow hooks called by the reviewed v7.0 patch. No detector/loss replacement."""
from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy
import os
import random
import numpy as np
import torch

PIPELINE = 'v7.0 native aspect resize -> Mosaic (native MixUp within Mosaic) or LetterBox -> RandomPerspective -> multiplicative RandomHSV -> flips -> native box clipping/filtering'
OPTIMIZER_STEPS = 0
CUMULATIVE_LR_DECAY = [0., 0., 0.]

def groups(optimizer):
    return [{'lr':g['lr'], 'initial_lr':g.get('initial_lr'), 'weight_decay':g.get('weight_decay',0.),
             'momentum':g.get('momentum'), 'nesterov':g.get('nesterov'),
             'betas':list(g['betas']) if 'betas' in g else None, 'eps':g.get('eps')}
            for g in optimizer.param_groups]

def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def checkpoint_identity(save_dir):
    return json.loads((Path(save_dir).parent/'frozen.json').read_text(encoding='utf-8'))['checkpoint_identity']

def local_checkpoint(weights):
    path = Path(weights).resolve()
    if not path.is_file():
        raise FileNotFoundError('Explicit local checkpoint missing; never auto-download: '+str(path))
    return str(path)

def validate_checkpoint(ckpt, save_dir, resume=False):
    if ckpt.get('comparison_identity') != checkpoint_identity(save_dir):
        raise ValueError('Checkpoint initialization/data/code/run identity differs')
    if resume:
        root = Path(save_dir)
        required = ('scaler','scheduler','last_opt_step','accumulate','rng_python','rng_numpy',
                    'rng_torch','rng_cuda','train_generator','val_generator','initialization_sha256',
                    'optimizer_steps','cumulative_lr_times_decay')
        state_native = ckpt.get('comparison_state', {})
        if any(k not in state_native for k in required) or ckpt.get('ema') is None or ckpt.get('updates') is None:
            raise ValueError('Pilot checkpoint lacks recoverable native scaler/scheduler/RNG/EMA state')
        from support import sha256
        if state_native['initialization_sha256'] != sha256(root/'initialization.json'):
            raise ValueError('Original initialization audit changed')
        from config import snapshot
        config = snapshot(root.parent)
        if ckpt['opt'].get('optimizer') != config['optimizer'] or not ckpt.get('optimizer') or len(ckpt['optimizer']['param_groups']) != 3:
            raise ValueError('Native optimizer/group identity differs')
        if (root/'training_complete.json').exists() or ckpt.get('optimizer') is None or not 0 <= ckpt['epoch'] < config['epochs']-1:
            raise ValueError('Checkpoint completed or not resumable')
        state = json.loads((root/'epoch_state.json').read_text(encoding='utf-8'))
        if state['completed_epochs'] != ckpt['epoch']+1 or state['end_reason'] != 'running':
            raise ValueError('Checkpoint/state mismatch or stopping condition already reached')
        best = torch.load(root/'weights/best.pt', map_location='cpu', weights_only=False)
        if best.get('comparison_identity') != ckpt['comparison_identity'] or best['epoch']+1 != state['best_epoch']:
            raise ValueError('Best/last checkpoint pair inconsistent')
        if ckpt['best_fitness'] != state['best_fitness'] or state_native['scheduler']['last_epoch'] != ckpt['epoch']:
            raise ValueError('Best fitness/scheduler/checkpoint state mismatch')

def save_checkpoint(ckpt, path):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(ckpt, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def record_initialization(save_dir, model, opt):
    identity = checkpoint_identity(save_dir)
    init = identity['initialization']
    if not opt.resume:
        if init['initialization_type']=='random' and (opt.weights or not opt.cfg):
            raise ValueError('Scratch requires weights empty, real architecture YAML, resume=False')
        if init['initialization_type']=='random':
            save(Path(save_dir)/'initialization.json', {**init,
                 'construction':'official train.py non-pretrained Model(cfg) branch',
                 'weights_argument':opt.weights, 'cfg_argument':opt.cfg, 'resume':False,
                 'parameters_unfused':sum(p.numel() for p in model.parameters()),
                 'model_tensors':len(model.state_dict()), 'checkpoint_identity':identity})
        else:
            transfer = json.loads((Path(save_dir)/'pretrained_load.json').read_text(encoding='utf-8'))
            if transfer['source_sha256'] != init['coco_source_sha256'] or transfer['transferred_tensors'] != init['pretrained_tensors_loaded']:
                raise ValueError('Official COCO provenance/transfer count differs from frozen identity')
            save(Path(save_dir)/'initialization.json', {**init, **transfer,
                 'construction':'official train.py compatible full state_dict transfer; before AMP and AutoAnchor',
                 'weights_argument':opt.weights,'cfg_argument':opt.cfg,'resume':False,
                 'checkpoint_identity':identity})

def strict_amp_probe(model, requested=True):
    """A private copy, native constants intact; restore all RNG streams on success or failure."""
    if not requested:
        return False
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('Formal recipe requires CUDA and AMP; use the explicit smoke entry point for CPU')
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device]):
            probe = deepcopy(model).eval()
            with torch.no_grad():
                x = torch.linspace(0,1,3*64*64,device=device).reshape(1,3,64,64)
                with torch.autocast('cuda', enabled=False):
                    fp = probe(x)[0]
                with torch.autocast('cuda', dtype=torch.float16):
                    mixed = probe(x)[0]
                if (fp.shape != mixed.shape or not torch.isfinite(fp).all()
                        or not torch.isfinite(mixed).all()
                        or not torch.allclose(fp, mixed.float(), rtol=.1, atol=.5)):
                    raise RuntimeError('Actual-model AMP accuracy probe failed; frozen AMP was not changed')
            del probe
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return True

def transfer_audit(model, state, source, weights):
    from support import sha256
    target = model.state_dict()
    compatible = {k for k,v in source.items() if k in target and v.shape == target[k].shape}
    if set(state) != compatible:
        raise ValueError('Compatible COCO tensors were excluded from native transfer')
    mismatches = [k for k,v in state.items() if not torch.equal(target[k].detach().cpu(), v.detach().cpu())]
    if mismatches:
        raise ValueError('Actual loaded values differ from source: '+str(mismatches))
    skipped = {k:{'reason':'class-dependent tensor shape mismatch' if k in target else 'absent in target',
                  'source_shape':list(v.shape),'target_shape':list(target[k].shape) if k in target else None}
               for k,v in source.items() if k not in compatible}
    if sum(p.numel() for p in model.parameters()) != 20871318:
        raise ValueError('Unexpected unfused nc=1 YOLOv5m parameter count')
    return {'transferred_tensors':len(state),'model_tensors':len(target),
            'source_sha256':sha256(weights),'source_file':str(weights),
            'all_compatible_tensors_loaded':True,'actual_value_equality_verified':True,
            'audit_timing':'immediately after load, before AutoAnchor/AMP/optimizer/first update',
            'skipped':skipped,'missing_keys':sorted(set(target)-set(state)),
            'parameters_unfused':20871318}

def record_transfer(save_dir, model, state, source, weights, resume):
    path = Path(save_dir)/'pretrained_load.json'
    if path.exists():
        path = Path(save_dir)/('resume_load_' + str(len(list(Path(save_dir).glob('resume_load_*.json')))+1) + '.json')
    audit = transfer_audit(model, state, source, weights)
    save(path,
         {**audit, 'resume':bool(resume),
          'missing_keys':sorted(set(model.state_dict())-set(state)),
          'nc':model.model[-1].nc, 'head':type(model.model[-1]).__name__,
          'anchor_levels':model.model[-1].nl, 'anchors_per_level':model.model[-1].na,
          'depth_multiple':model.yaml['depth_multiple'], 'width_multiple':model.yaml['width_multiple']})

def prepare_anchor_audit(save_dir, dataset, imgsz, threshold):
    dataset._comparison_save_dir = str(save_dir)
    manifest = json.loads((Path(save_dir).parent/'data/manifest.json').read_text(encoding='utf-8'))
    inventory = {str((Path(manifest['root'])/r['image']).resolve()):r for r in manifest['records'] if r['split']=='train'}
    if set(map(lambda p:str(Path(p).resolve()), dataset.im_files)) != set(inventory):
        raise ValueError('AutoAnchor dataset must contain exactly train images')
    for path, shape in zip(dataset.im_files, dataset.shapes):
        row = inventory[str(Path(path).resolve())]
        if list(shape) != [row['width'], row['height']]:
            raise ValueError('Native cache shapes differ from frozen original dimensions: '+path)
    save(Path(save_dir)/'autoanchor.json', {'source_split':'train','enabled':True,
        'imgsz':imgsz, 'threshold':threshold, 'images':len(dataset.im_files),
        'shapes_checked_against_frozen_manifest':True,
        'target_scale':'imgsz * original_wh / max(original_wh); native uniform(0.9,1.1) jitter',
        'integer_resize_rounding':'pinned v7.0 load_image int truncation on resized h/w; AutoAnchor retains native continuous target approximation'})

def record_anchor_fit(dataset, wh, anchors, threshold, phase):
    root = Path(dataset._comparison_save_dir)
    record = json.loads((root/'autoanchor.json').read_text(encoding='utf-8'))
    ratio = wh.cpu()[:,None] / anchors.detach().cpu().view(-1,2)[None]
    scores = torch.minimum(ratio,1/ratio).amin(2)
    record[phase] = {'bpr':float((scores.amax(1)>1/threshold).float().mean()),
                     'anchors_above_threshold':float((scores>1/threshold).float().sum(1).mean()),
                     'anchors_pixel_units':anchors.detach().cpu().view(-1,2).tolist(),
                     'target_wh_min':wh.amin(0).tolist(),'target_wh_max':wh.amax(0).tolist(),
                     'targets':len(wh), 'native_jittered_targets':True}
    save(root/'autoanchor.json', record)

def record_anchors(save_dir, before, model):
    after = model.model[-1].anchors.detach().cpu()
    record = json.loads((Path(save_dir)/'autoanchor.json').read_text(encoding='utf-8'))
    if not all(key in record for key in ('before','after')):
        raise RuntimeError('Native AutoAnchor failed or omitted BPR audit; inspect its log')
    save(Path(save_dir)/'autoanchor.json', {**record,
        'changed':not torch.equal(before,after), 'before_grid_units':before.tolist(),
        'after_grid_units':after.tolist(), 'stride':model.stride.tolist()})

def close_augmentation(loader, epoch, epochs=200, close=10):
    """Kill stale prefetched batches and worker copies once, retaining dataset/cache/sampler/generator."""
    ds = loader.dataset
    if close == 0 or epoch < epochs-close or getattr(ds, '_comparison_closed', False):
        return False
    ds.mosaic = False
    ds.hyp = dict(ds.hyp)
    for name in ('mosaic','mixup','cutmix','copy_paste'):
        ds.hyp[name] = 0.0
    ds._comparison_closed = True
    iterator = loader.iterator
    if hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    # InfiniteDataLoader uses a private repeat sampler. Calling base DataLoader.__iter__
    # rebuilds its iterator with the same sampler/generator and newly seeded workers.
    loader.iterator = torch.utils.data.DataLoader.__iter__(loader)
    return True

def record_effective(save_dir, hyp, opt, amp, loader, lf, warmup, accumulate, optimizer, model):
    global OPTIMIZER_STEPS, CUMULATIVE_LR_DECAY
    OPTIMIZER_STEPS, CUMULATIVE_LR_DECAY = 0, [0.,0.,0.]
    root = Path(save_dir)
    target = root/'effective_training.json'
    if opt.resume:
        target = root/('effective_resume_'+str(len(list(root.glob('effective_resume_*.json')))+1)+'.json')
    from config import snapshot, native_hyp
    raw = native_hyp(snapshot(root.parent))
    save(target,
         {'hyp_raw':raw, 'hyp_after_nc_imgsz_decay_scaling':hyp, 'options':vars(opt), 'amp_requested':opt.amp, 'amp_actual':bool(amp),
          'amp_probe':'isolated actual model, 1x3x64x64; RNG streams restored' if opt.amp else 'disabled by run snapshot',
          'actual_batch':opt.batch_size, 'nbs':opt.nbs, 'post_warmup_accumulate':int(accumulate),
          'effective_batch_after_warmup':opt.batch_size*int(accumulate),
          'effective_decay':hyp['weight_decay'], 'decay_scale':opt.batch_size*int(accumulate)/opt.nbs,
          'warmup_accumulate':f'round(linear(1,{opt.nbs}/{opt.batch_size})), minimum 1', 'warmup_steps':warmup,
          'loader_workers':loader.num_workers, 'seed':opt.seed, 'deterministic':opt.deterministic,
          'optimizer_type':type(optimizer).__module__+'.'+type(optimizer).__name__,
          'all_parameters_trainable':all(p.requires_grad for p in model.parameters()),
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'augmentation_pipeline':PIPELINE,
          'copy_paste':0.0,'cutmix':0.0,'albumentations':None,
          'native_validation':'actual loader/arguments/model/input dtypes in native_validation_calls.jsonl',
          'cosine_multiplier_by_epoch':[float(lf(e)) for e in range(opt.epochs+1)],
          'optimizer_groups':groups(optimizer)})

def record_optimizer_step(optimizer, scale_before, scaler):
    global OPTIMIZER_STEPS
    if scaler.get_scale() < scale_before:  # GradScaler skipped the overflowing update
        return
    OPTIMIZER_STEPS += 1
    for index, group in enumerate(optimizer.param_groups):
        CUMULATIVE_LR_DECAY[index] += float(group['lr']*group.get('weight_decay',0.))

def native_state(save_dir, scaler, scheduler, last_opt_step, accumulate, train_loader, val_loader):
    from support import sha256
    # Native v5 intentionally zeros residual gradients at the next epoch start.
    return {'scaler':scaler.state_dict(),'scheduler':scheduler.state_dict(),
            'last_opt_step':int(last_opt_step),'accumulate':int(accumulate),
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
            'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
            'train_generator':train_loader.generator.get_state(),'val_generator':val_loader.generator.get_state(),
            'initialization_sha256':sha256(Path(save_dir)/'initialization.json'),
            'optimizer_steps':OPTIMIZER_STEPS, 'cumulative_lr_times_decay':CUMULATIVE_LR_DECAY.copy(),
            'resume_semantics':'native epoch boundary, half checkpoint model/EMA; worker prefetch streams restart'}

def restore_native_state(state, scaler, scheduler, train_loader, val_loader, device):
    global OPTIMIZER_STEPS, CUMULATIVE_LR_DECAY
    OPTIMIZER_STEPS = state['optimizer_steps']
    CUMULATIVE_LR_DECAY = list(state['cumulative_lr_times_decay'])
    scaler.load_state_dict(state['scaler'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    # Workers are independent processes: restart instead of consuming stale prefetched batches.
    for loader,key in ((train_loader,'train_generator'),(val_loader,'val_generator')):
        if hasattr(loader.iterator,'_shutdown_workers'):
            loader.iterator._shutdown_workers()
        loader.generator.set_state(state[key])
        loader.iterator = torch.utils.data.DataLoader.__iter__(loader)
    return int(state['last_opt_step']),int(state['accumulate'])

def record_update(save_dir, model, optimizer, accumulate, ni):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    save(Path(save_dir)/'first_update.json',{'batch_index_global':ni,'actual_accumulate':int(accumulate),
        'gradient_tensors':len(grads),'all_gradients_finite':all(torch.isfinite(g).all().item() for g in grads),
        'nonzero_gradient_tensors':sum(bool(torch.count_nonzero(g)) for g in grads),
        'optimizer_type':type(optimizer).__module__+'.'+type(optimizer).__name__,
        'groups':groups(optimizer),
        'order':'backward -> scaler.unscale -> native clip max_norm10 -> this audit -> scaler.step/update -> zero_grad -> EMA'})

def begin_epoch(loader, epoch):
    global ACTIVE_EPOCH
    ACTIVE_EPOCH = epoch

def record_native_metrics(save_dir, precision, recall, ap50, ap75, map95):
    raw={'precision':float(precision),'recall':float(recall),'AP50':float(ap50),
         'AP75':float(ap75),'mAP50_95':float(map95)}
    save(Path(save_dir)/f'native_val_epoch_{ACTIVE_EPOCH:03d}.json',{
        'epoch_index':ACTIVE_EPOCH,'raw':raw,'percent':{k:100*v for k,v in raw.items()},
        'class':'crack','policy':'original v5 native validation, rect=True, NMS .6; separate from public metrics'})

def record_augmentation(save_dir, loader, epoch, optimizer, accumulate):
    save(Path(save_dir)/f'augmentation_epoch_{epoch:03d}.json',{
        'epoch_index':epoch,'closed':bool(getattr(loader.dataset,'_comparison_closed',False)),
        'hyp':loader.dataset.hyp,'pipeline':PIPELINE,'workers':loader.num_workers,
        'actual_accumulate_at_epoch_end':int(accumulate),
        'optimizer_steps_total':OPTIMIZER_STEPS,
        'cumulative_lr_times_decay_by_group':CUMULATIVE_LR_DECAY,
        'decay_record_semantics':'sum(lr * group weight_decay) on successful updates; coupled SGD/Adam decay is not a separate multiplicative shrink',
        'optimizer_groups':groups(optimizer)})

def record_native_validation(save_dir, model, loader, tensor, batch, imgsz, conf, iou, max_det, tta):
    value={'requested_batch':batch,'loader_batch':loader.batch_size,'rect':loader.dataset.rect,
           'imgsz':imgsz,'first_batch_shape':list(tensor.shape),'conf':conf,'nms_iou':iou,
           'max_det':max_det,'TTA':bool(tta),'input_dtype':str(tensor.dtype),
           'parameter_dtype':str(next(model.parameters()).dtype),'workers':loader.num_workers,
           'selection':'native val mAP50_95; public evaluation is separate'}
    with (Path(save_dir)/'native_validation_calls.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value)+'\n')

def restore_stopper(stopper, save_dir, start_epoch):
    if start_epoch:
        d = json.loads((Path(save_dir)/'epoch_state.json').read_text(encoding='utf-8'))
        if d['completed_epochs'] != start_epoch:
            raise ValueError('Checkpoint/state epoch mismatch; inspect interrupted checkpoint writes')
        stopper.best_epoch = d['best_epoch']-1
        stopper.best_fitness = d['best_fitness']
        stopper.possible_stop = start_epoch-1-stopper.best_epoch >= stopper.patience-1

def record_epoch(save_dir, epoch, stopper, stop, epochs, loader):
    root = Path(save_dir)
    current=json.loads((root/f'native_val_epoch_{epoch:03d}.json').read_text())
    best=json.loads((root/f'native_val_epoch_{stopper.best_epoch:03d}.json').read_text())
    save(Path(save_dir)/'epoch_state.json',
         {'completed_epochs':epoch+1, 'best_epoch':stopper.best_epoch+1, 'best_fitness':float(stopper.best_fitness),
          'native_val_current':current,'native_val_best':best,
          'selection':'native_training_val_mAP50_95_latest_tie',
          'end_reason':'patience' if stop else 'epoch_limit' if epoch+1==epochs else 'running',
          'mosaic_closed':bool(getattr(loader.dataset,'_comparison_closed',False))})

def complete(save_dir):
    path = Path(save_dir)/'epoch_state.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    if d['end_reason'] not in ('patience','epoch_limit'):
        raise RuntimeError('Trainer returned before the declared stopping condition')
    save(Path(save_dir)/'training_complete.json', d)
