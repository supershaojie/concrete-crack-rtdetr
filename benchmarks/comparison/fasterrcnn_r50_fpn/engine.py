"""Native detection losses and SGD; public validation drives selection and patience."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from model import build_kwargs, build_model, finite_gradients, finite_losses, seed_all
from support import atomic_bytes, canonical, read_json, sha256, status, validate_recipe, write_json


class StepSchedule:
    """LR for zero-based optimizer step s, with inclusive endpoints in each phase."""
    def __init__(self, cfg, batches):
        self.total = cfg['epochs']*batches
        self.warmup = cfg['warmup_epochs']*batches
        self.lr0, self.start, self.end = cfg['lr0'], cfg['warmup_start_factor'], cfg['lrf']
        self.steps_done = 0
        if not 1 < self.warmup < self.total-1:
            raise ValueError('Schedule needs at least two steps in warmup and cosine phases')

    def lr(self, step=None):
        step = self.steps_done if step is None else step
        if not 0 <= step < self.total:
            raise ValueError('Optimizer step outside the frozen planned schedule')
        if step < self.warmup:
            return self.lr0*(self.start + (1-self.start)*step/(self.warmup-1))
        fraction = (step-self.warmup)/(self.total-self.warmup-1)
        return self.lr0*(self.end+(1-self.end)*(1+math.cos(math.pi*fraction))/2)

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        if {k: v for k, v in state.items() if k != 'steps_done'} != {
                k: v for k, v in vars(self).items() if k != 'steps_done'}:
            raise ValueError('Saved LR schedule differs from frozen plan')
        if not 0 <= state['steps_done'] <= self.total:
            raise ValueError('Invalid completed optimizer step count')
        self.steps_done = state['steps_done']


def rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader_generator': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs on resume')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    generator.set_state(state['loader_generator'])


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    # Only own local state_dict checkpoints, never user-supplied COCO/model pickle paths.
    return torch.load(path, map_location='cpu', weights_only=False)


def validate_checkpoint(ckpt, identity, cfg, resume=False):
    if ckpt.get('format') != 'frcnn_scratch_checkpoint_v1' or ckpt.get('comparison_identity') != identity:
        raise ValueError('Checkpoint model/initialization/code/config/data/run/scope identity differs')
    if ckpt.get('model_build') != build_kwargs(cfg):
        raise ValueError('Checkpoint model construction differs')
    if not 1 <= ckpt.get('best_epoch', 0) <= ckpt.get('completed_epoch', 0) <= cfg['epochs']:
        raise ValueError('Invalid saved completed/best epoch')
    if not math.isfinite(ckpt.get('best_score', float('nan'))):
        raise ValueError('Invalid saved best validation score')
    if resume and any(ckpt.get(k) is None for k in ('optimizer', 'scaler', 'schedule', 'rng', 'history')):
        raise ValueError('Resume state is missing')
    if ckpt.get('early_stopping') != {'patience': cfg['patience'], 'best_epoch': ckpt['best_epoch'],
                                    'bad_epochs': ckpt['completed_epoch']-ckpt['best_epoch']}:
        raise ValueError('Early stopping state differs')


def checked_pair(run, identity, cfg):
    index = read_json(run/'checkpoints/index.json')
    for name in ('best', 'last'):
        if sha256(run/'checkpoints'/(name+'.pt')) != index[name+'_sha256']:
            raise ValueError('Checkpoint transaction interrupted or modified; inspect retained files before resume')
    last = load_checkpoint(run/'checkpoints/last.pt')
    validate_checkpoint(last, identity, cfg, resume=True)
    best = load_checkpoint(run/'checkpoints/best.pt')
    validate_checkpoint(best, identity, cfg)
    if best['completed_epoch'] != last['best_epoch'] or best['best_score'] != last['best_score']:
        raise ValueError('Best/last checkpoint selection state differs')
    if index['completed_epoch'] != last['completed_epoch'] or index['best_epoch'] != last['best_epoch']:
        raise ValueError('Checkpoint index epochs differ')
    return last, index


def train(run, manifest, identity, cfg, env, resume=False, interrupt_after_epoch=None):
    """interrupt_after_epoch is only used by the dedicated synthetic smoke program."""
    from data_adapter import TrainDataset, augmentation_record, train_loader
    from export import validate_epoch
    from tqdm import tqdm
    run = Path(run)
    validate_recipe(cfg)
    if interrupt_after_epoch is not None and identity['scope'] != 'SMOKE_ONLY_SYNTHETIC':
        raise ValueError('Synthetic interruption hooks are forbidden for formal runs')
    if os.environ.get('WORLD_SIZE', '1') != '1' or os.environ.get('RANK', '-1') != '-1':
        raise ValueError('Only single-process training is supported')
    if cfg['amp'] and (not cfg['device'].startswith('cuda') or not torch.cuda.is_available()):
        raise ValueError('CUDA AMP required; refusing CPU/FP32 fallback')
    previous = read_json(run/'train_status.json') if (run/'train_status.json').exists() else {}
    if previous.get('status') == 'completed':
        raise ValueError('Training already complete; export/evaluate the selected best')
    if resume:
        if read_json(run/'identity.json') != identity:
            raise ValueError('Resume requires unchanged run/code/config/data/environment identity')
        ckpt, _ = checked_pair(run, identity, cfg)
        attempt = uuid.uuid4().hex
        write_json(run/'resumes'/(attempt+'.json'), {'checkpoint_sha256': sha256(run/'checkpoints/last.pt'),
            'completed_epoch': ckpt['completed_epoch'], 'best_epoch': ckpt['best_epoch'], 'identity': identity,
            'support': 'resume from last fully saved epoch; restore RNG, shuffle generator, SGD, scaler and schedule; '
                       'mid-epoch work is replayed; native CUDA ops/worker scheduling need not be bitwise reproducible'})
    else:
        if (run/'identity.json').exists() or (run/'checkpoints').exists():
            raise FileExistsError('Existing training state protected; resume must be explicit')
        ckpt = None
        write_json(run/'identity.json', identity)
        write_json(run/'environment.json', env)
        write_json(run/'expanded_recipe.json', cfg)
        write_json(run/'augmentation.json', augmentation_record(cfg))
    status(run, 'train', 'running', resume=resume, scope=identity['scope'])
    seed_all(cfg['seed'], cfg['deterministic'])
    model, initialization = build_model(cfg)
    initialization.update(dataset_identity_sha256=manifest['dataset_identity_sha256'],
                          recipe_sha256=identity['recipe_sha256'], code_identity=identity, environment=env)
    if not resume:
        write_json(run/'initialization.json', initialization)
    model.to(cfg['device']).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr0'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'], init_scale=cfg['amp_init_scale'],
                                     growth_interval=cfg['amp_growth_interval'])
    dataset = TrainDataset(run, manifest, cfg)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = train_loader(dataset, cfg, generator)
    schedule = StepSchedule(cfg, len(loader))
    completed, best_epoch, best_score, history = 0, 0, -math.inf, []
    if resume:
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        schedule.load_state_dict(ckpt['schedule'])
        completed, best_epoch, best_score = ckpt['completed_epoch'], ckpt['best_epoch'], ckpt['best_score']
        history = ckpt['history']
        if schedule.steps_done != completed*len(loader) or len(history) != completed:
            raise ValueError('Completed epochs, optimizer steps or metric history disagree')
        restore_rng(ckpt['rng'], generator)
        if (run/'epoch_trace.jsonl').exists():
            atomic_bytes(run/'resumes'/(attempt+'_pre_resume_trace.jsonl'), (run/'epoch_trace.jsonl').read_bytes())
        atomic_bytes(run/'epoch_trace.jsonl', b''.join(canonical(r) for r in history))
        del ckpt
    else:
        write_json(run/'actual_training_setup.json', {'optimizer': 'SGD', 'parameter_groups': 1,
            'all_trainable_parameters_in_group': len(params), 'BN_and_bias_weight_decay': cfg['weight_decay'],
            'gradient_accumulation_steps': 1, 'ema': False, 'gradient_clip': None,
            'amp_requested': cfg['amp'], 'amp_enabled': scaler.is_enabled(), 'initial_scale': scaler.get_scale(),
            'train_batch': loader.batch_size, 'train_workers': loader.num_workers, 'eval_workers': cfg['eval_workers'],
            'persistent_workers': False, 'drop_last': False, 'batches_per_epoch': len(loader),
            'schedule': schedule.state_dict(), 'lr_boundary_definition': 'step0=start; warmup_steps-1=lr0; '
                'warmup_steps=lr0; total_steps-1=lr0*lrf; one LR assignment before each SGD step',
            'determinism': 'seed42; cudnn deterministic; torch deterministic algorithms warn_only=True to '
                'preserve native RoIAlign/RPN CUDA behavior; no bitwise guarantee', 'environment': env})
    if not scaler.is_enabled() and cfg['amp']:
        raise ValueError('AMP silently disabled')
    for epoch in range(completed, cfg['epochs']):
        if completed and completed-best_epoch >= cfg['patience']:
            break
        if interrupt_after_epoch is not None and completed >= interrupt_after_epoch:
            raise KeyboardInterrupt('Intentional synthetic interruption at completed epoch boundary')
        dataset.set_epoch(epoch)
        model.train()
        sums, observed, first_lr, last_lr = {}, {'mosaic': 0, 'mixup': 0, 'hsv': 0, 'empty_target': 0}, None, None
        batches = 0
        progress = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}', dynamic_ncols=True)
        for images, targets, traces in progress:
            if any(t['epoch'] != epoch for t in traces):
                raise RuntimeError('Stale data-worker epoch')
            if epoch >= cfg['epochs']-cfg['close_mosaic'] and cfg['close_mosaic'] and any(
                    t['mosaic'] or t['mixup'] for t in traces):
                raise RuntimeError('Mosaic/MixUp remained active in a worker after close boundary')
            images = [im.to(cfg['device'], non_blocking=True) for im in images]
            targets = [{k: v.to(cfg['device'], non_blocking=True) for k, v in t.items()} for t in targets]
            last_lr = schedule.lr()
            first_lr = last_lr if first_lr is None else first_lr
            for group in optimizer.param_groups:
                group['lr'] = last_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg['amp']):
                losses = model(images, targets)
                loss = finite_losses(losses)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradients(model)  # fail explicitly; never silently skip overflowing batches
            scaler.step(optimizer)
            scaler.update()
            schedule.steps_done += 1
            batches += 1
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0)+float(value.detach())
            for trace in traces:
                for name in observed:
                    observed[name] += int(trace[name])
            progress.set_postfix(loss=f'{float(loss.detach()):.4f}', lr=f'{last_lr:.7f}')
        # The DataLoader iterator is exhausted; nonpersistent worker processes have exited.
        if batches != len(loader):
            raise RuntimeError('Incomplete training epoch')
        metrics = validate_epoch(model, run, manifest, cfg, identity, epoch+1, cfg['device'])
        score = metrics['mAP50_95']
        is_best = score >= best_score  # equality refreshes best_epoch and patience, same as frozen adapter
        if is_best:
            best_score, best_epoch = score, epoch+1
        completed = epoch+1
        row = {'epoch': completed, 'optimizer_steps': schedule.steps_done, 'lr_first': first_lr, 'lr_last': last_lr,
            'losses': {k: v/batches for k, v in sums.items()}, 'val_raw_0_1': metrics['raw_0_1'],
            'best_epoch': best_epoch, 'best_mAP50_95': best_score, 'bad_epochs': completed-best_epoch,
            'augmentation_probabilities': dataset.active_probabilities, 'observed_augmentation_calls': observed,
            'amp_scale': scaler.get_scale(), 'backbone_input': model.backbone.comparison_observed_size}
        history.append(row)
        checkpoint = {'format': 'frcnn_scratch_checkpoint_v1', 'comparison_identity': identity,
            'model_build': build_kwargs(cfg), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'schedule': schedule.state_dict(), 'rng': rng_state(generator),
            'completed_epoch': completed, 'best_epoch': best_epoch, 'best_score': best_score, 'history': history,
            'early_stopping': {'patience': cfg['patience'], 'best_epoch': best_epoch, 'bad_epochs': completed-best_epoch}}
        if is_best:
            atomic_checkpoint(run/'checkpoints/best.pt', checkpoint)
        atomic_checkpoint(run/'checkpoints/last.pt', checkpoint)
        index = {'completed_epoch': completed, 'best_epoch': best_epoch,
                 'best_sha256': sha256(run/'checkpoints/best.pt'), 'last_sha256': sha256(run/'checkpoints/last.pt')}
        write_json(run/'checkpoints/index.json', index)
        atomic_bytes(run/'epoch_trace.jsonl', b''.join(canonical(r) for r in history))
        write_json(run/'training_progress.json', row)
        print(f'Epoch {completed}: val={metrics["display_percent"]}; best={best_epoch}; bad_epochs={completed-best_epoch}', flush=True)
    result = {**read_json(run/'checkpoints/index.json'), 'best_validation_mAP50_95': best_score,
        'stop_reason': 'epoch_limit' if completed == cfg['epochs'] else 'patience',
        'optimizer_steps': schedule.steps_done, 'scope': identity['scope'], 'exit_code': 0}
    status(run, 'train', 'completed', **result)
    return result
