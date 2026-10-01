"""NCR v1: detached per-axis nesting gates on final regular-query matches only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .loss import RTDETRDetectionLoss


@dataclass(frozen=True)
class NCRConfig:
    lambda_ncr: float = 0.25
    min_side_px: float = 4.0
    huber_delta: float = 1.0
    ramp_start: int = 4
    ramp_epochs: int = 15

    def __post_init__(self):
        if not math.isfinite(self.lambda_ncr) or self.lambda_ncr < 0:
            raise ValueError("lambda_ncr must be finite and nonnegative")
        if (self.min_side_px, self.huber_delta, self.ramp_start, self.ramp_epochs) != (4.0, 1.0, 4, 15):
            raise ValueError("NCR v1 fixes the scale floor, Huber transition and epoch ramp")

    def coefficient(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("NCR requires the actual zero-based trainer epoch, including on resume")
        return self.lambda_ncr * min(1.0, max(0.0, (epoch - self.ramp_start) / self.ramp_epochs))


def ncr_terms(boxes, targets, input_hw, min_side_px=4.0):
    """FP32 terms. Only predicted centers retain gradients; GT and the whole gate do not."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or boxes.shape != targets.shape:
        raise ValueError("NCR expects equally shaped [M,4] normalized cxcywh tensors")
    height, width = map(int, input_hw)
    if min(height, width) <= 0:
        raise ValueError("input_hw must describe the actual network input")
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        b, g = boxes.float(), targets.detach().float()
        delta = b[:, :2] - g[:, :2]
        margin = (b[:, 2:] - g[:, 2:]).abs() * 0.5
        gate = (torch.relu(margin - delta.abs()) / (margin + 1e-7)).detach()
        scale = torch.maximum(g[:, 2:], g.new_tensor([min_side_px / width, min_side_px / height]))
        u = delta / scale
        rho = torch.where(u.abs() <= 1, 0.5 * u.square(), u.abs() - 0.5)
        raw = (gate * rho).sum() / (2 * max(len(b), 1))
    return raw, gate, u, scale


def distribution(values):
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return dict(count=0, mean=None, p50=None, p90=None, max=None)
    return dict(count=values.numel(), mean=float(values.mean()), p50=float(values.quantile(.5)),
                p90=float(values.quantile(.9)), max=float(values.max()))


def summarize(boxes, targets, input_hw, coefficient, terms, base_center_grad=None):
    """Detached, bounded-size aggregates; no predictions or training graphs are stored."""
    raw, gate, u, scale = (t.detach() for t in terms)
    active = gate > 0
    added = coefficient * gate * u.clamp(-1, 1) / (2 * max(len(boxes), 1) * scale)
    short = (targets.detach().float()[:, 2:] * targets.new_tensor([input_hw[1], input_hw[0]])).amin(-1)
    groups = {}
    for name, mask in (("lt4px", short < 4), ("4to16px", (short >= 4) & (short < 16)),
                       ("16to32px", (short >= 16) & (short < 32)), ("ge32px", short >= 32)):
        groups[name] = dict(matches=int(mask.sum()), relative_center_error=distribution(u[mask].abs()))
    ratios = dict(status="NOT_RUN", reason="No regression gradient in this context")
    if base_center_grad is not None:
        original = base_center_grad.detach().float().abs()
        valid = active & (original > 1e-12)
        ratios = dict(status="PASSED", definition="abs(weighted NCR center gradient) / abs(5L1+2GIoU center gradient)",
                      active_axes=int(active.sum()), zero_base_axes=int((active & ~valid).sum()),
                      ratio=distribution(added.abs()[valid] / original[valid]),
                      added_abs=distribution(added.abs()[active]), base_abs=distribution(original[active]))
    return dict(matches=len(boxes), active_fraction_xy=active.float().mean(0).tolist() if len(boxes) else [0., 0.],
                h_xy=[distribution(gate[:, a]) for a in (0, 1)], short_side_groups=groups,
                raw=float(raw), lambda_eff=coefficient, weighted=float(raw * coefficient), gradient_ratio=ratios)


class _FinalRegularNCR:
    """One forward's local hook. It is neither a module buffer nor a checkpoint dependency."""
    def __init__(self, reference, coefficient, input_hw, config, collect):
        self.weighted = reference.new_zeros((), dtype=torch.float32)
        self.coefficient, self.input_hw, self.config, self.collect = coefficient, input_hw, config, collect
        self.diagnostics = None

    def __call__(self, boxes, targets, losses):
        if not len(boxes):
            self.diagnostics = dict(matches=0, raw=0., weighted=0., lambda_eff=self.coefficient,
                                    active_fraction_xy=[0., 0.])
            return
        terms = ncr_terms(boxes, targets, self.input_hw, self.config.min_side_px)
        self.weighted = self.coefficient * terms[0]
        if self.collect:
            base_grad = None
            if torch.is_grad_enabled() and boxes.requires_grad:
                base_grad = torch.autograd.grad(losses["loss_bbox"] + losses["loss_giou"], boxes,
                                                retain_graph=True)[0][:, :2]
            self.diagnostics = summarize(boxes, targets, self.input_hw, self.coefficient, terms, base_grad)


class NCRDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = NCRConfig(**(config or {}))
        self.last_diagnostics = None

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None, *, input_hw, epoch,
                collect_diagnostics=False):
        coefficient = self.config.coefficient(epoch)
        # No hook, gates, matching cache, extra graph, or RNG consumption when off.
        hook = _FinalRegularNCR(preds[0], coefficient, input_hw, self.config, collect_diagnostics) if coefficient else None
        losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta, final_regular_hook=hook)
        # Assemble AFTER native no-DN zero terms, so loss_ncr_dn never exists.
        losses["loss_ncr"] = hook.weighted if hook is not None else preds[0].new_zeros((), dtype=torch.float32)
        self.last_diagnostics = hook.diagnostics if hook is not None else dict(
            raw=0., weighted=0., lambda_eff=0., status="OFF",
            matches=sum(min(preds[0].shape[-2], int(n)) for n in batch["gt_groups"]))
        return losses


def configure_ncr(model, config=None):
    """Persist plain config on the original model class; inference/state_dict are unchanged."""
    model.ncr_config = asdict(NCRConfig(**(config or {})))
    if hasattr(model, "criterion"):
        del model.criterion
    return model
