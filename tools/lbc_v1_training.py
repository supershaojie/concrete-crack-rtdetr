"""LBC's native Trainer integration, initialization, resume and deploy conversion."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import random
import numpy as np
import torch

from init_c19_lif_v1 import (ROOT, MODEL_DIR, CONFIGS, SOURCE_SHA256, controlled_models,
                             build_training_model, verify_model, require, sha256, write_json)
from ultralytics import __version__
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.lbc import LBCDetectionModel, LBC_CONFIG, HEAD_KEYS, ramp
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import unwrap_model
from lbc_v1_reporting import append_jsonl


def copy_decoder_cache(source, target):
    device = next(target.parameters()).device
    for name in ('shapes', 'anchors', 'valid_mask'):
        value = getattr(source.model[-1], name)
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device).clone()
            if value.is_floating_point(): value = value.float()
        setattr(target.model[-1], name, deepcopy(value))


def native_copy(model):
    """Reconstruct the native model, rejecting every missing public parameter/buffer."""
    require(type(model) in (RTDETRDetectionModel, LBCDetectionModel), "Unexpected model type")
    with torch.random.fork_rng(devices=[]):
        native = RTDETRDetectionModel(deepcopy(model.yaml), nc=model.model[-1].nc, verbose=False)
    source = model.state_dict()
    public = {k: v for k, v in source.items() if not k.startswith('lbc_head.')}
    require(set(source) - set(public) == (HEAD_KEYS if isinstance(model, LBCDetectionModel) else set()),
            "Unexpected auxiliary state inventory")
    require(set(public) == set(native.state_dict()), "Missing/unexpected native deployment state")
    native.load_state_dict(public, strict=True)
    native.nc = model.model[-1].nc
    # Native decoder caches are plain attributes (not state_dict buffers). A saved
    # FP16 EMA can contain rounded anchors, so preserve these public caches too.
    copy_decoder_cache(model, native)
    for name in ('args', 'names', 'task', 'pt_path'):
        if hasattr(model, name):
            setattr(native, name, deepcopy(getattr(model, name)))
    verify_model(native)
    return native


def audit_wrapper(native, model):
    require(type(native) is RTDETRDetectionModel, "Native strict audit required")
    verify_model(native)
    require(type(model) is LBCDetectionModel and model.lbc_config == LBC_CONFIG, "Wrong LBC version/config")
    before, after = native.state_dict(), model.state_dict()
    require(set(after) - set(before) == HEAD_KEYS and set(before) <= set(after), "Unexpected state keys")
    require(all(torch.equal(v, after[k]) for k, v in before.items()), "Public states differ")
    require(sum(p.numel() for p in model.lbc_head.parameters()) == 4128, "Head count mismatch")
    return dict(public_states=len(before), public_exact=True, new_keys=sorted(HEAD_KEYS), new_buffers=[],
                original_parameters=sum(p.numel() for p in native.parameters()),
                training_parameters=sum(p.numel() for p in model.parameters()), added_parameters=4128)


def initialize(source, destination):
    destination = Path(destination)
    require(not destination.exists(), "Preserve existing initialization")
    _, native, provenance = controlled_models(source)
    verify_model(native, zero=True)
    before_rng = torch.get_rng_state().clone()
    model = LBCDetectionModel(native)
    require(torch.equal(before_rng, torch.get_rng_state()), "Head construction consumed public RNG")
    audit = audit_wrapper(native, model)
    model.eval()
    model.args = dict(DEFAULT_CFG_DICT, model=str(MODEL_DIR/CONFIGS['pair']), task='detect')
    model.task, model.pt_path = 'detect', str(destination.resolve())
    checkpoint = dict(epoch=-1, model=model, ema=None, updates=None, optimizer=None, scaler=None,
                      best_fitness=None, train_args=model.args, lbc_config=deepcopy(LBC_CONFIG),
                      lbc_provenance=provenance, date=datetime.now(timezone.utc).isoformat())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(checkpoint, stream)
    restored = torch_load(destination, map_location='cpu')['model']
    require(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()), 'Init roundtrip differs')
    return dict(status='PASSED', source_sha256=SOURCE_SHA256, output_sha256=sha256(destination),
                provenance=provenance, audit=audit, rng_restored=True)


def validate_checkpoint(ckpt):
    require(ckpt.get('lbc_config') == LBC_CONFIG, 'Missing or changed LBC checkpoint definition')
    for name in ('model', 'ema'):
        value = ckpt.get(name)
        require(type(value) is LBCDetectionModel and value.lbc_config == LBC_CONFIG,
                f'Resume requires complete {name}, never random auxiliary repair')
        require(HEAD_KEYS <= set(value.state_dict()), f'Missing auxiliary state in {name}')
    require(ckpt.get('epoch', -1) >= 0 and ckpt.get('optimizer') is not None, 'Checkpoint has no resumable optimizer/epoch')
    require(all(k in ckpt for k in ('scaler', 'scheduler', 'lbc_loop', 'lbc_rng', 'stopper')), 'Incomplete resume state')


class LBCTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        self.lbc_effective_updates = 0
        self.lbc_overflow_skips = 0
        self.lbc_consecutive_skips = 0
        self.lbc_epoch_rows = []
        self.lbc_output = None
        self.lbc_resume_state = None
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_epoch_start', epoch_start)
        self.add_callback('on_train_epoch_end', epoch_end)
        self.add_callback('on_train_batch_start', no_oom_retry)
        self.add_callback('on_train_batch_end', batch_end)

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(type(weights) is LBCDetectionModel and weights.lbc_config == LBC_CONFIG,
                'Use the controlled LBC initialization or a complete LBC resume checkpoint')
        source = native_copy(weights)
        if self.resume:
            native = RTDETRTrainer.get_model(self, cfg, source, verbose=False)
            copy_decoder_cache(source, native)
            verify_model(native)
            require(set(native.state_dict()) == set(source.state_dict()), 'Resume public keys mismatch')
            require(all(torch.equal(v, native.state_dict()[k]) for k, v in source.state_dict().items()),
                    'Resume public values mismatch')
            self.lbc_adaptation = dict(resume=True, class_adaptation=[])
        else:
            native, self.lbc_adaptation = build_training_model(cfg, source, self.data)
        model = LBCDetectionModel(native)
        model.lbc_head.load_state_dict(weights.lbc_head.float().state_dict(), strict=True)
        self.lbc_model_audit = audit_wrapper(native, model)
        return model

    def build_optimizer(self, model, *args, **kwargs):
        optimizer = super().build_optimizer(model, *args, **kwargs)
        names = {id(p): n for n, p in model.named_parameters()}
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        require(len(ids) == len(set(ids)) and set(ids) == set(names), 'Optimizer omissions/duplicates')
        self.lbc_optimizer_groups = []
        for i, group in enumerate(optimizer.param_groups):
            for p in group['params']:
                if names[id(p)] in HEAD_KEYS:
                    require(group['param_group'] == 'weight', 'Auxiliary parameter misclassified')
                    self.lbc_optimizer_groups.append(dict(name=names[id(p)], group=i,
                        param_group=group['param_group'], lr=group['lr'], weight_decay=group['weight_decay']))
        require(len(self.lbc_optimizer_groups) == 2, 'Both auxiliary weights must be optimized')
        return optimizer

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._lbc_ni = getattr(self, '_lbc_ni', self.start_epoch * len(self.train_loader) - 1) + 1
        return batch

    def optimizer_step(self):
        model = unwrap_model(self.model)
        self.scaler.unscale_(self.optimizer)
        original = [p for n, p in model.named_parameters() if n not in HEAD_KEYS]
        head = list(model.lbc_head.parameters())
        diagnostic = getattr(self, 'lbc_update_diagnostics', None)
        if diagnostic is not None:
            diagnostic.before_clip(self, model, HEAD_KEYS)
        n0 = torch.nn.utils.clip_grad_norm_(original, max_norm=10.0)
        nh = torch.nn.utils.clip_grad_norm_(head, max_norm=10.0)
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < old_scale
        self.lbc_effective_updates += int(not skipped)
        self.lbc_overflow_skips += int(skipped)
        self.lbc_consecutive_skips = self.lbc_consecutive_skips + 1 if skipped else 0
        if diagnostic is not None:
            diagnostic.after_step(self, model, n0, nh, skipped)
        # None preserves AdamW's skip semantics on an entirely unsupervised window.
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema:
            self.ema.update(self.model)  # native ordering, including native overflow behavior
        self._lbc_last_opt_step = self._lbc_ni
        self.lbc_clip = dict(original=float(n0), head=float(nh), threshold=10.0,
                             effective=not skipped, scale=self.scaler.get_scale())
        require(self.lbc_consecutive_skips < 16, '16 consecutive AMP skipped updates; inspect numerical failure')

    def save_model(self):
        """Keep raw model, FP32 optimizer, EMA and native scaler; preserve best/last."""
        raw = unwrap_model(self.model)
        ckpt = dict(epoch=self.epoch, best_fitness=self.best_fitness,
                    model=deepcopy(raw).cpu().float(), ema=deepcopy(unwrap_model(self.ema.ema)).cpu().half(),
                    updates=self.ema.updates, optimizer=deepcopy(self.optimizer.state_dict()),
                    scaler=self.scaler.state_dict(), scheduler=self.scheduler.state_dict(),
                    train_args=vars(self.args).copy(), train_metrics={**self.metrics, 'fitness': self.fitness},
                    train_results=self.read_results_csv(), version=__version__,
                    date=datetime.now(timezone.utc).isoformat(), lbc_config=deepcopy(LBC_CONFIG),
                    lbc_loop=dict(last_opt_step=self._lbc_last_opt_step, ni=self._lbc_ni,
                        accumulate=self.accumulate, batches=len(self.train_loader),
                        gradients={n: p.grad.detach().cpu().clone() for n, p in raw.named_parameters() if p.grad is not None},
                        effective_updates=self.lbc_effective_updates, overflow_skips=self.lbc_overflow_skips),
                    lbc_rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    stopper=deepcopy(vars(self.stopper)))
        self.wdir.mkdir(parents=True, exist_ok=True)
        targets = [self.last] + ([self.best] if self.best_fitness == self.fitness else [])
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            targets.append(self.wdir / f'epoch{self.epoch}.pt')
        for target in targets:
            tmp = target.with_suffix('.pt.partial')
            torch.save(ckpt, tmp)
            tmp.replace(target)

    def resume_training(self, ckpt):
        if ckpt is None or not self.resume:
            return
        validate_checkpoint(ckpt)
        require(ckpt['epoch'] + 1 < self.epochs, 'Completed run cannot resume as new training')
        super().resume_training(ckpt)
        unwrap_model(self.model).load_state_dict(ckpt['model'].float().state_dict(), strict=True)
        copy_decoder_cache(ckpt['model'], unwrap_model(self.model))
        copy_decoder_cache(ckpt['ema'], self.ema.ema)
        self.lbc_resume_state = ckpt
        self.stopper.__dict__.update(ckpt['stopper'])
        unwrap_model(self.model).lbc_epoch = self.start_epoch

    def _setup_train(self):
        super()._setup_train()
        if self.lbc_resume_state:
            # Native setup assigns last_epoch after resume_training; restore after that assignment.
            self.scheduler.load_state_dict(self.lbc_resume_state['scheduler'])

    def initial_optimizer_step_index(self):
        state = self.lbc_resume_state
        self._lbc_ni = self.start_epoch * len(self.train_loader) - 1
        self._lbc_last_opt_step = -1
        if state:
            loop = state['lbc_loop']
            require(loop['batches'] == len(self.train_loader) and loop['ni'] == self._lbc_ni,
                    'Resume supports completed epoch boundaries with identical loader length')
            self._lbc_last_opt_step = loop['last_opt_step']
            self.accumulate = loop['accumulate']
            self.lbc_effective_updates = loop['effective_updates']
            self.lbc_overflow_skips = loop['overflow_skips']
        return self._lbc_last_opt_step

    def initialize_train_gradients(self):
        self.optimizer.zero_grad(set_to_none=True)
        if self.lbc_resume_state:
            state = self.lbc_resume_state
            parameters = dict(unwrap_model(self.model).named_parameters())
            for name, value in state['lbc_loop']['gradients'].items():
                require(name in parameters and value.shape == parameters[name].shape, 'Invalid pending gradient')
                parameters[name].grad = value.to(parameters[name].device, dtype=parameters[name].dtype)
            rng = state['lbc_rng']
            torch.set_rng_state(rng['torch']); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
            self.lbc_resume_state = None

    def final_eval(self):
        # Native final_eval strips optimizer/epoch in place. This experiment retains resumable files.
        if self.best.exists():
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False
            self.metrics = self.validator(model=self.best)
            self.metrics.pop('fitness', None)
            self.run_callbacks('on_fit_epoch_end')


def no_oom_retry(trainer):
    # Set at batch start, not epoch start: native loop resets its counter afterwards.
    trainer._oom_retries = 3


def epoch_start(trainer):
    unwrap_model(trainer.model).lbc_epoch = trainer.epoch
    trainer.lbc_epoch_rows = []


def batch_end(trainer):
    require(torch.isfinite(trainer.loss).all(), 'Nonfinite total training loss')
    trainer.lbc_epoch_rows.append(deepcopy(unwrap_model(trainer.model).lbc_last))
    # The counter suppresses auto-reduction but must not trigger native epoch retry.
    trainer._oom_retries = 0


def epoch_end(trainer):
    rows = trainer.lbc_epoch_rows
    m = sum(row.get('pairs', 0) for row in rows)
    report = dict(epoch=trainer.epoch, ramp=ramp(trainer.epoch), weight=.05*ramp(trainer.epoch),
                  batches=len(rows), pairs=m, effective_updates=trainer.lbc_effective_updates,
                  overflow_skips=trainer.lbc_overflow_skips, geometry_evaluated=ramp(trainer.epoch)>0,
                  validation='L0 only; auxiliary not evaluated')
    for k in ('gt', 'valid_gt', 'invalid_box', 'empty_positive', 'insufficient_background',
              'positive_candidates', 'negative_candidates', 'positive_positions', 'negative_positions'):
        report[k] = sum(row.get(k, 0) for row in rows)
    for k in ('raw_pair', 'raw_bg', 'raw_total', 's_pos', 's_neg', 'gap'):
        report[k] = sum(row.get(k, 0)*row.get('pairs', 0) for row in rows)/m if m else None
    report['weighted_mean_per_microbatch'] = sum(row.get('weighted', 0) for row in rows)/max(len(rows), 1)
    report['clip_last'] = getattr(trainer, 'lbc_clip', None)
    if trainer.lbc_output:
        path = Path(trainer.lbc_output)/'epochs.jsonl'
        append_jsonl(path, report)


def deploy(checkpoint, destination):
    destination = Path(destination)
    require(not destination.exists(), 'Deployment must not overwrite an existing checkpoint')
    ckpt = torch_load(checkpoint, map_location='cpu')
    require(ckpt.get('lbc_config') == LBC_CONFIG, 'Wrong LBC source')
    model = (ckpt.get('ema') or ckpt['model']).float().eval().requires_grad_(False)
    native = native_copy(model).eval().requires_grad_(False)
    require(not any(k.startswith('lbc_head.') for k in native.state_dict()), 'Head leaked into deploy')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(424001)
        x = torch.rand(1, 3, 160, 160)
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            a, b = model(x)[0], native(x)[0]
        require(torch.equal(a, b) and torch.isfinite(a).all(), 'Deployment changed inference')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(dict(model=native, ema=None, epoch=-1, train_args=ckpt['train_args'],
                        lbc_deploy_source_sha256=sha256(checkpoint)), stream)
    report = dict(status='PASSED', source_sha256=sha256(checkpoint), deploy_sha256=sha256(destination),
                  inference_exact=True, parameters=sum(p.numel() for p in native.parameters()),
                  removed_keys=sorted(HEAD_KEYS), model_type=type(native).__name__,
                  public_decoder_caches=['shapes','anchors','valid_mask'])
    write_json(destination.with_suffix('.json'), report)
    return report
