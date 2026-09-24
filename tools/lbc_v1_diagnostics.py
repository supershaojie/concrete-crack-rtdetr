"""Read-only optimizer observations, enabled only by the isolated capacity check."""
from copy import deepcopy
import inspect
import math
import torch


def configure_capacity_scaler(trainer):
    if getattr(trainer, 'resume', False) or trainer.lbc_effective_updates or trainer.lbc_overflow_skips:
        raise RuntimeError('Diagnostic scaler override requires a fresh isolated trainer')
    native = trainer.scaler
    default = inspect.signature(type(native)).parameters['init_scale'].default
    trainer.scaler = type(native)(init_scale=128.0, enabled=trainer.amp)
    return dict(scope='isolated capacity diagnostic only; never formal training or resume',
                native_default_init_scale=float(default), original_scale=native.get_scale(),
                original_state=deepcopy(native.state_dict()), diagnostic_init_scale=128.0,
                active_initial_scale=trainer.scaler.get_scale(), enabled=trainer.amp,
                native_dynamic_scaling=True, reinitializations=1)


def gradient_summary(named):
    present = [(n, p.grad.detach()) for n, p in named if p.grad is not None]
    bad = [n for n, g in present if not torch.isfinite(g).all().item()]
    # FP64 observation avoids confusing a finite gradient with an FP32 norm overflow.
    norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for _, g in present])).item() if present else 0.0
    return dict(finite=not bad, tensors_with_grad=len(present),
                missing_grad_parameters=[n for n, p in named if p.grad is None],
                nonfinite_parameters=bad, l2_norm_fp64=norm)


class UpdateDiagnostics:
    def __init__(self, backbone_name='model.5.blocks.0.branch2a.conv.weight'):
        self.attempts = []
        self.backbone_name = backbone_name

    def before_clip(self, trainer, model, head_keys):
        named = list(model.named_parameters())
        params = dict(named)
        if self.backbone_name not in params:
            raise RuntimeError('Missing backbone observation parameter: ' + self.backbone_name)
        tracked = sorted(head_keys | {self.backbone_name})
        self.before = {n: params[n].detach().clone() for n in tracked}
        attempt = dict(index=len(self.attempts), completed=False, microbatch_index=trainer._lbc_ni,
                       accumulation_target=trainer.accumulate,
                       accumulated_microbatches=trainer._lbc_ni-trainer._lbc_last_opt_step,
                       scale_before=trainer.scaler.get_scale(),
                       observation='after native unscale_, before either clip',
                       original_gradients=gradient_summary([(n, p) for n, p in named if n not in head_keys]),
                       head_gradients=gradient_summary([(n, p) for n, p in named if n in head_keys]),
                       backbone_parameter=self.backbone_name)
        self.attempts.append(attempt)

    def after_step(self, trainer, model, n0, nh, skipped):
        named = dict(model.named_parameters())
        changed = [n for n, before in self.before.items() if not torch.equal(before, named[n].detach())]
        bad = [n for n, p in named.items() if not torch.isfinite(p.detach()).all().item()]
        self.attempts[-1].update(completed=True, scale_after=trainer.scaler.get_scale(),
            effective_update=not skipped, overflow_skip=skipped,
            effective_updates=trainer.lbc_effective_updates, overflow_skips=trainer.lbc_overflow_skips,
            consecutive_overflow_skips=trainer.lbc_consecutive_skips,
            clip_pre_norms=dict(original=float(n0), head=float(nh), threshold=10.0),
            parameters_finite=not bad, nonfinite_parameters=bad,
            changed_observed_parameters=changed, head_updated=any(n.startswith('lbc_head.') for n in changed),
            backbone_updated=self.backbone_name in changed, accumulated_microbatches_after=0)
        self.before = {}


def require_valid_attempts(attempts):
    """Overflow can recover dynamically; a claimed effective update must be healthy."""
    for row in attempts:
        if not row['completed']:
            raise RuntimeError('Incomplete optimizer attempt')
        if not row['parameters_finite']:
            raise RuntimeError('Nonfinite parameters after optimizer attempt')
        if row['overflow_skip']:
            if row['head_updated'] or row['backbone_updated']:
                raise RuntimeError('AMP overflow unexpectedly changed observed parameters')
        elif not (row['original_gradients']['finite'] and row['head_gradients']['finite']
                  and row['original_gradients']['tensors_with_grad'] > 0
                  and row['head_gradients']['tensors_with_grad'] > 0
                  and all(math.isfinite(v) for v in row['clip_pre_norms'].values())
                  and row['head_updated'] and row['backbone_updated']):
            raise RuntimeError('Effective update lacks finite gradients/norms or head/backbone update')
