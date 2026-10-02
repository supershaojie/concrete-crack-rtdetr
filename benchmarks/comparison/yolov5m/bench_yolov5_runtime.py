"""Narrow hooks called by the reviewed v7.0 patch. No detector/loss replacement."""
from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy
import os
import random
import numpy as np
import torch

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
        if (root/'training_complete.json').exists() or ckpt.get('optimizer') is None or not 0 <= ckpt['epoch'] < 199:
            raise ValueError('Checkpoint completed or not resumable')
        state = json.loads((root/'epoch_state.json').read_text(encoding='utf-8'))
        if state['completed_epochs'] != ckpt['epoch']+1 or state['end_reason'] != 'running':
            raise ValueError('Checkpoint/state mismatch or stopping condition already reached')
        best = torch.load(root/'weights/best.pt', map_location='cpu', weights_only=False)
        if best.get('comparison_identity') != ckpt['comparison_identity'] or best['epoch']+1 != state['best_epoch']:
            raise ValueError('Best/last checkpoint pair inconsistent')

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

def strict_amp_probe(model):
    """A private copy, native constants intact; restore all RNG streams on success or failure."""
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

def record_transfer(save_dir, model, state):
    path = Path(save_dir)/'pretrained_load.json'
    if path.exists():
        path = Path(save_dir)/('resume_load_' + str(len(list(Path(save_dir).glob('resume_load_*.json')))+1) + '.json')
    save(path,
         {'transferred_tensors':len(state), 'model_tensors':len(model.state_dict()),
          'missing_keys':sorted(set(model.state_dict())-set(state)),
          'nc':model.model[-1].nc, 'head':type(model.model[-1]).__name__,
          'anchor_levels':model.model[-1].nl, 'anchors_per_level':model.model[-1].na,
          'depth_multiple':model.yaml['depth_multiple'], 'width_multiple':model.yaml['width_multiple']})

def record_anchors(save_dir, before, model):
    after = model.model[-1].anchors.detach().cpu()
    save(Path(save_dir)/'autoanchor.json', {'source_split':'train', 'enabled':True,
        'changed':not torch.equal(before,after), 'before_grid_units':before.tolist(),
        'after_grid_units':after.tolist(), 'stride':model.stride.tolist()})

def close_augmentation(loader, epoch, epochs=200, close=10):
    """Kill stale prefetched batches and worker copies once, retaining dataset/cache/sampler/generator."""
    ds = loader.dataset
    if epoch < epochs-close or getattr(ds, '_comparison_closed', False):
        return False
    ds.mosaic = False
    ds.hyp = dict(ds.hyp)
    for name in ('mosaic','mixup','copy_paste'):
        ds.hyp[name] = 0.0
    ds._comparison_closed = True
    iterator = loader.iterator
    if hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    # InfiniteDataLoader uses a private repeat sampler. Calling base DataLoader.__iter__
    # rebuilds its iterator with the same sampler/generator and newly seeded workers.
    loader.iterator = torch.utils.data.DataLoader.__iter__(loader)
    return True

def record_effective(save_dir, hyp, opt, amp, loader, lf, warmup, accumulate, optimizer):
    save(Path(save_dir)/'effective_training.json',
         {'hyp_after_nc_imgsz_decay_scaling':hyp, 'options':vars(opt), 'amp_requested':True, 'amp_actual':bool(amp),
          'amp_probe':'isolated actual model, 1x3x64x64; allclose rtol=0.1 atol=0.5; all RNG streams restored',
          'actual_batch':opt.batch_size, 'nbs':64, 'post_warmup_accumulate':int(accumulate),
          'warmup_accumulate':'round(linear(1,64/16)), minimum 1', 'warmup_steps':warmup,
          'loader_workers':loader.num_workers, 'seed':opt.seed, 'deterministic':True,
          'cosine_multiplier_by_epoch':[float(lf(e)) for e in range(opt.epochs+1)],
          'optimizer_groups':[{'lr':g['lr'],'weight_decay':g['weight_decay'],'momentum':g.get('momentum'),
                               'nesterov':g.get('nesterov')} for g in optimizer.param_groups]})

def restore_stopper(stopper, save_dir, start_epoch):
    if start_epoch:
        d = json.loads((Path(save_dir)/'epoch_state.json').read_text(encoding='utf-8'))
        if d['completed_epochs'] != start_epoch:
            raise ValueError('Checkpoint/state epoch mismatch; inspect interrupted checkpoint writes')
        stopper.best_epoch = d['best_epoch']-1
        stopper.best_fitness = d['best_fitness']
        stopper.possible_stop = start_epoch-1-stopper.best_epoch >= stopper.patience-1

def record_epoch(save_dir, epoch, stopper, stop, epochs):
    save(Path(save_dir)/'epoch_state.json',
         {'completed_epochs':epoch+1, 'best_epoch':stopper.best_epoch+1, 'best_fitness':float(stopper.best_fitness),
          'selection':'native_training_val_mAP50_95_latest_tie',
          'end_reason':'patience' if stop else 'epoch_limit' if epoch+1==epochs else 'running',
          'mosaic_closed':epoch >= epochs-10})

def complete(save_dir):
    path = Path(save_dir)/'epoch_state.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    if d['end_reason'] not in ('patience','epoch_limit'):
        raise RuntimeError('Trainer returned before the declared stopping condition')
    save(Path(save_dir)/'training_complete.json', d)
