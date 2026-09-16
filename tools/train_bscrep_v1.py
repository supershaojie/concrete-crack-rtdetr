"""Explicit plan/start/resume for BSC-Rep v1. Import and plan never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal

import torch
from init_bscrep_v1 import (ROOT, VARIANTS, require, write_json, sha256, runtime,
                            verify_model, build_training_model)
from bscrep_v1_common import (DEFAULT_VARIANT, paths, recipe, fingerprint, verified_data,
                              optimizer_coverage, git)
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import YAML
from ultralytics.utils.patches import torch_load
from train_c19_lif_v1 import ensure_amp_resources, disable_oom_retry


class AuditedTrainer(RTDETRTrainer):
    variant = DEFAULT_VARIANT
    audit_dir = None
    fresh = True
    expected_args = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None, 'Controlled initialization or explicit resume checkpoint required')
        model, report = build_training_model(cfg, weights, self.data, self.variant, fresh=self.fresh)
        self.loading_report = report
        if self.audit_dir:
            write_json(Path(self.audit_dir)/'nc1_loading.json', report)
        return model

    def build_optimizer(self, model, *args, **kwargs):
        # Runs after native nc adaptation, before any update; new tensors already registered.
        verify_model(model, self.variant, zero=self.fresh)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self.coverage_report = optimizer_coverage(model, optimizer)
        if self.audit_dir:
            write_json(Path(self.audit_dir)/'optimizer_coverage.json', self.coverage_report)
        return optimizer

    def _setup_train(self):
        super()._setup_train()
        require(bool(self.amp), 'Native check_amp disabled AMP; formal recipe may not be changed')
        require(self.batch_size == 16 and self.args.imgsz == 640, 'Formal B16/640 changed')
        verify_model(self.model, self.variant, zero=self.fresh)
        if self.expected_args:
            actual = vars(self.args)
            differences = {k: [v, actual.get(k)] for k, v in self.expected_args.items()
                           if type(v) is not type(actual.get(k)) or actual.get(k) != v}
            require(not differences, f'Actual training recipe changed: {differences}')
        if self.audit_dir:
            YAML.save(Path(self.audit_dir)/'actual_train_args.yaml', vars(self.args))
            write_json(Path(self.audit_dir)/'setup.json', dict(status='PASSED', amp=bool(self.amp),
                       nc=self.model.model[-1].nc, batch=16, imgsz=640, optimizer_steps=0))


def require_preflight(report, identity):
    require(report.get('status') == 'PASSED', 'Preflight FAILED/PENDING; complete server checks first')
    require(report.get('identity') == identity, 'Preflight does not match current source/config/init/data')
    require(report.get('server') is True and report.get('formal_initialization_unchanged') is True and
            report.get('formal_optimizer_steps') == 0, 'Server preflight must preserve the unused formal initialization')
    for key in ('cpu', 'cuda', 'dataset'):
        require(report.get(key, {}).get('status') == 'PASSED', f'Mandatory {key} preflight is incomplete')
    capacity = report.get('capacity', {})
    require(capacity.get('status') == 'PASSED' and capacity.get('batch') == 16 and
            capacity.get('imgsz') == 640 and capacity.get('AMP') is True, 'Actual B16/640/native AMP capacity evidence required')
    require(capacity.get('online_augmentation') is True and capacity.get('native_setup_train') is True and
            capacity.get('data_scope') == 'verified real train split' and capacity.get('formal_optimizer_steps') == 0,
            'Capacity must exercise the real Trainer and unchanged online augmentation')
    attempts = capacity.get('attempts', [])
    require(attempts and attempts[-1].get('unscaled_all_finite') is True and
            isinstance(attempts[-1].get('loss'), (int, float)) and math.isfinite(attempts[-1]['loss']),
            'Capacity lacks finite original loss and unscaled gradients')


def resume_checkpoint(checkpoint, variant, p, expected):
    """Validate before native __init__, which writes args.yaml into the saved run."""
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint == (p['run']/'weights/last.pt').resolve(), 'Resume only this experiment last.pt')
    require(checkpoint.is_file(), 'Resume checkpoint missing')
    ckpt = torch_load(checkpoint, map_location='cpu')
    require(type(ckpt.get('epoch')) is int and 0 <= ckpt['epoch'] < expected['epochs'] - 1 and
            ckpt.get('optimizer') is not None, 'Checkpoint is stripped/complete, cannot resume')
    saved = ckpt.get('ema') or ckpt.get('model')
    require(saved is not None, 'Resume checkpoint contains no model or EMA')
    verify_model(saved, variant)
    require(saved.model[-1].nc == 1, 'Resume requires the learned nc=1 model')
    saved_args = ckpt.get('train_args', {})
    differences = {k: [v, saved_args.get(k)] for k, v in expected.items() if k not in {'model', 'resume'}
                   and (type(v) is not type(saved_args.get(k)) or saved_args.get(k) != v)}
    require(not differences, f'Resume checkpoint recipe/path changed: {differences}')
    require(saved_args.get('model') in (str(p['init'].resolve()), str(checkpoint)), 'Resume checkpoint has another experiment model identity')
    require(saved_args.get('resume') is False or saved_args.get('resume') == str(checkpoint), 'Resume checkpoint has another resume identity')
    return ckpt


def plan(variant, main=None):
    p = paths(variant, main)
    args, rows = recipe(variant, p['init'], p['data'], p['run'].parent)
    return dict(mode='plan', variant=variant, formal_training='NOT_STARTED', test='NOT_RUN',
                runtime=runtime(), paths=p, args=args, recipe_diff=rows,
                requirements=['fixed source SHA256', 'controlled init', 'matching PASSED server preflight',
                              'unchanged successful dataset', 'unused run for start'])


def run(variant, preflight, main=None, resume=None):
    p = paths(variant, main)
    if not resume:
        require(not p['run'].exists(), f'Existing run protected: {p["run"]}')
    require(ROOT.resolve() != p['main'], 'Use the independent worktree')
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'rtdetr', 'Activate the existing rtdetr environment')
    require(torch.cuda.is_available(), 'Formal run requires CUDA')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Tracked code must be clean')
    args, rows = recipe(variant, p['init'], p['data'], p['run'].parent)
    identity = fingerprint(p['source'], p['init'], variant, p['data'])
    checked = json.loads(Path(preflight).read_text(encoding='utf-8'))
    require_preflight(checked, identity)
    _, inventory = verified_data(p['data'])
    require(checked.get('dataset_inventory') == inventory, 'Preflight dataset identity mismatch')
    require(checked.get('runtime', {}).get('torch') == str(torch.__version__), 'Runtime changed since preflight')
    p['launch'].mkdir(parents=True, exist_ok=True)
    attempt = p['launch']/('resume_' if resume else 'start_')
    attempt = attempt.with_name(attempt.name + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    lock = p['run'].with_name(p['run'].name + '.bscrep-active')
    lock.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        resume = Path(resume).resolve()
        require((p['launch']/'identity.json').is_file(), 'Missing original launch identity')
        require(json.loads((p['launch']/'identity.json').read_text()) == identity, 'Source/config/init changed since start')
        ckpt = resume_checkpoint(resume, variant, p, args)
        del ckpt
        # check_resume restores the same saved 200-epoch recipe; no new initialization.
        args.update(model=str(resume), resume=str(resume))
    else:
        require(not p['run'].exists(), f'Existing run protected: {p["run"]}')
    lock.mkdir(exist_ok=False)
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(lock/'owner.json', dict(pid=os.getpid(), variant=variant, worktree=str(ROOT), attempt=str(attempt)))
    status = dict(status='STARTING', formal_training='NOT_STARTED', test='NOT_RUN', variant=variant)
    write_json(attempt/'state.json', status)
    trainer = None
    try:
        if not resume:
            # Atomic reservation closes the gap between the existence guard and
            # native Trainer.__init__, which otherwise accepts an explicit save_dir.
            p['run'].mkdir(parents=True, exist_ok=False)
        ensure_amp_resources(p['main'], attempt/'amp_resources.json')
        write_json(p['launch']/'identity.json', identity)
        write_json(attempt/'preflight.json', checked)
        write_json(attempt/'recipe_diff.json', rows)
        write_json(attempt/'dataset_inventory.json', inventory)
        YAML.save(attempt/'resolved_args.yaml', args)

        class ConfiguredTrainer(AuditedTrainer):
            pass
        ConfiguredTrainer.variant, ConfiguredTrainer.audit_dir = variant, attempt
        ConfiguredTrainer.fresh, ConfiguredTrainer.expected_args = not bool(resume), args
        trainer = ConfiguredTrainer(overrides=args)
        require(trainer.save_dir.resolve() == p['run'].resolve(), 'Refuse automatic name2/name3')
        trainer.add_callback('on_train_batch_start', disable_oom_retry)
        def progress(t):
            write_json(attempt/'state.json', dict(status='RUNNING', epoch=int(t.epoch)+1,
                       formal_training='RUNNING', test='NOT_RUN', variant=variant))
        trainer.add_callback('on_train_start', progress)
        trainer.add_callback('on_fit_epoch_end', progress)
        trainer.train()
        completed = int(trainer.epoch) + 1
        state = 'COMPLETED_200' if completed >= 200 else ('EARLY_STOPPED' if trainer.stop else 'INTERRUPTED')
        status.update(status=state, epochs_completed=completed, formal_training=state, exit_code=0)
    except KeyboardInterrupt:
        status.update(status='INTERRUPTED', formal_training='INTERRUPTED', exit_code=130)
        raise
    except BaseException as error:
        status.update(status='FAILED', error=repr(error), formal_training='FAILED', exit_code=1)
        raise
    finally:
        status['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(attempt/'state.json', status)
        # Only our exact lock files; a killed process leaves a visible lock for inspection.
        (lock/'owner.json').unlink()
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'start', 'resume'))
    parser.add_argument('--variant', choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, help='Optional plan JSON outside the formal run')
    a = parser.parse_args()
    torch.set_num_threads(4)
    if a.mode == 'plan':
        result = plan(a.variant, a.main)
        if a.output:
            require(not a.output.exists(), 'Existing plan protected')
            require(not a.output.resolve().is_relative_to(paths(a.variant, a.main)['run']), 'Plan cannot occupy formal run')
            write_json(a.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        require(a.preflight is not None, '--preflight is required')
        require((a.checkpoint is not None) == (a.mode == 'resume'), 'Only resume requires --checkpoint')
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run(a.variant, a.preflight, a.main, a.checkpoint)


if __name__ == '__main__':
    main()
