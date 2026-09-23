"""RDL v1: final regular matches only; all new geometry/reductions in FP32.

No model parameters, buffers, activation caches, or second forward. Diagnostics
returned by this module are separate from the dictionary summed by Ultralytics.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .loss import RTDETRDetectionLoss

CONFIG = dict(version="rdl_v1", coefficient=0.05, rho=0.10, ramp_start=5, ramp_length=15, epsilon=1e-6)


def ramp(completed_epochs):
    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ValueError("RDL requires the real nonnegative completed-epoch count")
    return min(1.0, max(0.0, (completed_epochs - 5) / 15))


def sides(boxes):
    """xywh -> L,R,T,B (NOT xyxy); both axes use the positive image direction."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), -1)


def finite(name, value, positive_wh=False):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"RDL nonfinite {name}; inspect source predictions/targets, no replacement applied")
    if positive_wh and (value[..., 2:] <= 0).any():
        raise ValueError(f"RDL nonpositive width/height in {name}")


def decompose(b0, gt, v, image_hw):
    """Return live out/in scalars and detached per-side evidence for this call.

    The empty case remains connected to both live prediction tensors. Tiny
    denominators are protected only in vstar/eta, never in a, boxes, or S.
    """
    if b0.ndim != 2 or b0.shape != gt.shape or b0.shape != v.shape or b0.shape[-1] != 4:
        raise ValueError("RDL expects matched [M,4] tensors")
    h, w = map(int, image_hw)
    if h <= 0 or w <= 0:
        raise ValueError("RDL requires actual preprocessed image H,W")
    with torch.autocast(device_type=b0.device.type, enabled=False):
        b0, gt, v = b0.float(), gt.float(), v.float()
        finite("before", b0, True)
        finite("GT", gt, True)
        finite("tanh_offsets", v)
        if (v.abs() > 1).any():
            raise ValueError("RDL v must be the actual tanh offsets")
        x0, gs = sides(b0), sides(gt).detach()
        a = (0.10 * b0[:, [2, 2, 3, 3]]).detach()
        ebar = (gs - x0).detach()
        vstar = (ebar / a.clamp_min(1e-6)).clamp(-1, 1).detach()
        eta = (a / ebar.abs().clamp_min(1e-6)).clamp(max=1)
        eta = torch.where(ebar == 0, torch.ones_like(eta), eta).detach()
        pixel = b0.new_tensor([1 / w, 1 / w, 1 / h, 1 / h])
        scale = torch.maximum(a, pixel).detach()
        o = ((gs - x0).abs() - a).relu() / scale
        out_sides = F.smooth_l1_loss(o, torch.zeros_like(o), reduction="none", beta=1.0)
        in_sides = eta * F.smooth_l1_loss(v, vstar, reduction="none", beta=1.0)
        denom = 4 * max(len(b0), 1)
        connected_zero = (b0.sum() + v.sum()) * 0.0
        out, inside = out_sides.sum() / denom + connected_zero, in_sides.sum() / denom + connected_zero
        evidence = dict(a=a, ebar=ebar, vstar=vstar, eta=eta, S=scale, o=o.detach(),
                        out_sides=out_sides.detach(), in_sides=in_sides.detach(), abs_v=v.detach().abs(),
                        a_protected=a < 1e-6, e_protected=(ebar.abs() < 1e-6) & (ebar != 0))
        return out, inside, evidence


def matched_rdl(details, indices, gt, image_hw):
    idx, gt_idx = RTDETRDetectionLoss._get_index(indices)
    b0, v = details["before"][idx], details["tanh_offsets"][idx]
    return decompose(b0, gt[gt_idx], v, image_hw)


class RDLDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if config != CONFIG:
            raise ValueError(f"RDL v1 configuration mismatch: {config}")
        self.config = dict(config)
        self.completed_epochs = 0
        self.last_stats = {}  # Python numbers only, never match/activation tensors.

    def set_epoch(self, completed_epochs):
        ramp(completed_epochs)
        self.completed_epochs = completed_epochs

    @property
    def weight(self):
        return self.config["coefficient"] * ramp(self.completed_epochs)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None,
                *, details=None, image_hw=None, enabled=False):
        weight = self.weight if enabled else 0.0
        if weight == 0:
            self.last_stats = dict(e=self.completed_epochs, weight=0.0, active=False,
                                   validation_L0_only=not enabled)
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        if details is None or image_hw is None:
            raise RuntimeError("Active RDL training requires live details from this same forward and input H,W")
        boxes, scores = preds
        for name, value in (("regular boxes", boxes), ("regular scores", scores),
                            ("GT", batch["bboxes"]), ("before", details["before"]),
                            ("after", details["after"]), ("tanh_offsets", details["tanh_offsets"])):
            finite(name, value, name in ("regular boxes", "GT", "before", "after"))
        if details["before"].shape != boxes[-1].shape or not torch.equal(details["after"], boxes[-1]):
            raise ValueError("RDL details are not aligned to the final regular-query predictions")
        # Exactly the original final refined-box match; auxiliary matches remain independent.
        indices = self.matcher(boxes[-1], scores[-1], batch["bboxes"], batch["cls"], batch["gt_groups"])
        loss = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta, final_match_indices=indices)
        out, inside, ev = matched_rdl(details, indices, batch["bboxes"], image_hw)
        loss["loss_rdl"] = weight * (out + inside)  # after all original _dn naming
        values = torch.stack((out.detach(), inside.detach(), loss["loss_rdl"].detach(),
                              ev["a_protected"].sum(), ev["e_protected"].sum(),
                              (ev["ebar"].abs() > ev["a"]).sum(), (ev["vstar"].abs() == 1).sum())).cpu().tolist()
        self.last_stats = dict(zip(("L_out", "L_in", "weighted", "a_protected", "e_protected",
                                    "outside_sides", "saturated_targets"), values))
        self.last_stats.update(e=self.completed_epochs, weight=weight, M=sum(len(i) for i, _ in indices), active=True)
        return loss
