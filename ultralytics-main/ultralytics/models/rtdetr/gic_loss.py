"""GIC v1: a detached geometry interval for the final ordinary VFL positive term.

No model parameters, matching, output changes, or additional loss dictionary keys.
The parent retains all negative modulation and its autograd behavior verbatim.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

CONFIG = {"version": "gic_v1", "shift_px": 1.0, "eta_max": 0.5,
          "ramp_start_epoch": 4, "ramp_duration": 15}


def eta_at_epoch(epoch, eta_max=0.5):
    if not isinstance(epoch, int) or epoch < 0 or not 0 <= eta_max <= 0.5:
        raise ValueError("GIC needs a real, nonnegative integer epoch and eta_max in [0,.5]")
    return eta_max * min(max((epoch - 4) / 15, 0.0), 1.0)


def check_probability(value, name):
    if not bool((torch.isfinite(value) & (value >= 0) & (value <= 1)).all()):
        raise ValueError(f"GIC {name} is nonfinite or outside [0,1]; inspect source boxes/precision")


def entropy(value):
    """Exact endpoint convention; clamp ONLY the log's input, never its multiplier."""
    tiny = torch.finfo(value.dtype).tiny
    return -(value * value.clamp_min(tiny).log() + (1 - value) * (1 - value).clamp_min(tiny).log())


@torch.no_grad()
def shifted_boxes(boxes, image_hw, shift_px=1.0):
    h, w = map(int, image_hw)
    if h <= 0 or w <= 0 or not math.isfinite(shift_px) or shift_px < 0:
        raise ValueError("Invalid GIC input dimensions/shift")
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        boxes = boxes.detach().float()
        if not bool(torch.isfinite(boxes).all() and (boxes[..., 2:] > 0).all()):
            raise ValueError("GIC requires finite cxcywh boxes with positive sizes")
        offsets = boxes.new_tensor([(dx * shift_px / w, dy * shift_px / h, 0, 0)
                                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        # No edge clipping: shifted center may legitimately leave the image.
        return boxes[:, None, :] + offsets[None, :, :]


@torch.no_grad()
def geometry_interval(boxes, gt_boxes, q, image_hw, shift_px=1.0):
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        gt_boxes, q = gt_boxes.detach().float(), q.detach().float()
        check_probability(q, "original VFL quality")
        if not bool(torch.isfinite(gt_boxes).all() and (gt_boxes[..., 2:] > 0).all()):
            raise ValueError("GIC GT boxes must be finite with positive sizes")
        shifted = shifted_boxes(boxes, image_hw, shift_px)
        # Original ordinary xywh IoU helper and its epsilon; never GIoU.
        ious = bbox_iou(shifted, gt_boxes[:, None, :], xywh=True).squeeze(-1)
        check_probability(ious, "shifted IoU")
        all_quality = torch.cat((q[:, None], ious), dim=1)
        return all_quality.min(1).values, all_quality.max(1).values


def positive_delta(z, q, lo, hi, eta):
    """Unreduced signed difference; FP32 logits retain gradient, all targets detach."""
    if not 0 <= eta <= 0.5:
        raise ValueError("GIC eta must be in [0,.5]")
    with torch.autocast(device_type=z.device.type, enabled=False):
        z, q, lo, hi = z.float(), q.detach().float(), lo.detach().float(), hi.detach().float()
        for value, name in ((q, "q"), (lo, "lo"), (hi, "hi")):
            check_probability(value, name)
        if not bool(torch.isfinite(z).all() and (lo <= q).all() and (q <= hi).all()):
            raise ValueError("Nonfinite logits or interval excluding original q")
        p = z.detach().sigmoid()
        t = torch.minimum(torch.maximum(p, lo), hi).detach()
        delta = eta * q * ((q - t) * z + entropy(q) - entropy(t))
        return delta, t


@torch.no_grad()
def aggregate(z, q, lo, hi, t, delta, gt_boxes, image_hw, eta):
    """Detached scalars only; no batch tensor/activation is retained by the criterion."""
    z, q, lo, hi, t, delta = [x.detach().float() for x in (z, q, lo, hi, t, delta)]
    p = z.sigmoid()
    old_grad = q * (p - q)
    new_grad = q * ((1 - eta) * (p - q) + eta * (p - t))
    width = hi - lo
    near_zero = old_grad.abs() <= 1e-8
    meaningful = ~near_zero
    def stats(mask):
        n = int(mask.sum())
        if not n:
            return {"matched": 0}
        valid = mask & meaningful
        ratio = new_grad[valid].abs() / old_grad[valid].abs()
        old_pos = q[mask] * F.binary_cross_entropy_with_logits(z[mask], q[mask], reduction="none")
        return {"matched": n, "q_mean": float(q[mask].mean()), "lo_mean": float(lo[mask].mean()),
                "hi_mean": float(hi[mask].mean()), "p_mean": float(p[mask].mean()),
                "abs_p_minus_q_mean": float((p[mask] - q[mask]).abs().mean()),
                "width_quantiles_0_25_50_75_95_100": torch.quantile(width[mask], width.new_tensor([0,.25,.5,.75,.95,1])).cpu().tolist(),
                "interval_nonzero": int((width[mask] > 0).sum()),
                "delta_nonzero": int((delta[mask] != 0).sum()),
                "p_inside_fraction": float(((p[mask] >= lo[mask]) & (p[mask] <= hi[mask])).float().mean()),
                "gradient_weakened": int((new_grad[mask].abs() < old_grad[mask].abs() - 1e-8).sum()),
                "near_zero_old_gradient_ignored": int((mask & near_zero).sum()),
                "gradient_ratio_mean": float(ratio.mean()) if ratio.numel() else None,
                "gradient_reduction_mean": float((1-ratio).mean()) if ratio.numel() else None,
                "old_gradient_abs_mean": float(old_grad[mask].abs().mean()),
                "new_gradient_abs_mean": float(new_grad[mask].abs().mean()),
                "delta_sum": float(delta[mask].sum()), "old_positive_sum": float(old_pos.sum()),
                "new_positive_sum": float((old_pos + delta[mask]).sum())}
    h, w = image_hw
    short = (gt_boxes.detach().float()[:, 2:] * gt_boxes.new_tensor([w, h]).float()).min(-1).values
    result = stats(torch.ones_like(q, dtype=torch.bool))
    result.update(eta_eff=eta, short_side_space="actual_model_input_pixels", short_side_groups={
        "lt4": stats(short < 4), "4to16": stats((short >= 4) & (short < 16)),
        "16to32": stats((short >= 16) & (short < 32)), "ge32": stats(short >= 32)})
    return result


class GICDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = deepcopy(CONFIG if config is None else config)
        # Zero eta_max is supported for controlled ablation; all other v1 fields fixed.
        if set(self.config) != set(CONFIG) or any(self.config[k] != v for k, v in CONFIG.items() if k != "eta_max"):
            raise ValueError("Unknown or changed GIC v1 definition")
        eta_at_epoch(0, self.config["eta_max"])
        if self.vfl is None or self.fl is None:
            raise ValueError("GIC requires the original DETR VFL branch")
        self.epoch, self.image_hw, self.collect = 0, None, False
        self.diagnostics = None

    def set_context(self, epoch, image_hw, collect=False):
        eta_at_epoch(epoch, self.config["eta_max"])
        self.epoch, self.image_hw, self.collect = epoch, tuple(image_hw), bool(collect)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        eta = eta_at_epoch(self.epoch, self.config["eta_max"])
        self.diagnostics = {"epoch": self.epoch, "eta_eff": eta, "matched": 0} if self.collect else None
        # EXACT zero path: no interval, sigmoid, entropy, diagnostic tensors or RNG.
        if eta == 0 or not len(batch["bboxes"]):
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        if self.image_hw is None:
            raise ValueError("GIC needs actual input H/W from RTDETRDetectionModel.loss")
        return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta,
                               final_normal_context={"eta": eta, "image_hw": self.image_hw})

    def _adjust_final_normal_class(self, losses, scores, boxes, gt_boxes, q, idx, cls, postfix, context):
        with torch.autocast(device_type=scores.device.type, enabled=False):
            lo, hi = geometry_interval(boxes, gt_boxes, q, context["image_hw"], self.config["shift_px"])
            z = scores[idx[0], idx[1], cls].float()
            delta, t = positive_delta(z, q, lo, hi, context["eta"])
            # Parent: mean(query).sum() / (max(M,1)/nq), then class gain ONCE.
            scaled = delta.sum() / max(len(gt_boxes), 1) * self.loss_gain["class"]
            losses[f"loss_class{postfix}"] = losses[f"loss_class{postfix}"] + scaled
            if self.collect:
                self.diagnostics = aggregate(z, q, lo, hi, t, delta, gt_boxes, context["image_hw"], context["eta"])
                self.diagnostics.update(epoch=self.epoch, delta_reduced=float(scaled.detach()),
                                        denominator=len(gt_boxes), class_gain=self.loss_gain["class"])
