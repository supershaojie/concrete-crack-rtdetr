"""GEO-v1-envelope-K3. No parameters, buffers, matcher changes or inference path."""
from __future__ import annotations

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.ops import xywh2xyxy

FORMULA = dict(name="GEO-v1-envelope-K3", enabled=True, lambda_geo=0.20,
               topk=3, eps_area=1e-9, ramp_start=5, ramp_full=20)


def ramp(epoch):
    """epoch is the zero-based trainer.epoch, including after resume."""
    return min(max((int(epoch) - 5) / 15.0, 0.0), 1.0)


class GEOGeometryError(ValueError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def positive(x):
    # A touching (zero-length) intersection has zero gradient.
    return torch.where(x.detach() > 0, x, torch.zeros_like(x))


def intersection(a, b):
    lengths = torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2])
    return positive(lengths).prod(-1)


def envelope(pred, gt):
    gt = gt.detach()
    return torch.cat((torch.where(pred[..., :2].detach() < gt[..., :2], pred[..., :2], gt[..., :2]),
                      torch.where(pred[..., 2:].detach() > gt[..., 2:], pred[..., 2:], gt[..., 2:])), -1)


def pair_overflow(pred, own, neighbors):
    """[M,4], [M,4], [N,4] xyxy -> [M,N]; dtype kept for FP64 reference checks."""
    own, neighbors = own.detach(), neighbors.detach()
    h = envelope(pred, own)
    hl, ht, hr, hb = h.unbind(-1)
    gl, gt, gr, gb = own.unbind(-1)
    strips = torch.stack((torch.stack((hl, ht, gl, hb), -1),
                          torch.stack((gr, ht, hr, hb), -1),
                          torch.stack((gl, ht, gr, gt), -1),
                          torch.stack((gl, gb, gr, hb), -1)), 1)
    extra = intersection(strips[:, :, None, :], neighbors[None, None, :, :]).sum(1)
    own_area = (own[:, 2:] - own[:, :2]).prod(-1)
    neighbor_area = (neighbors[:, 2:] - neighbors[:, :2]).prod(-1)
    denominator = torch.maximum(own_area[:, None], neighbor_area[None, :]).clamp_min(FORMULA["eps_area"])
    return extra / denominator


def raw_geo_xyxy(pred, gt, groups, matches, sample=False):
    """Full augmented GT, global GT match IDs, final ordinary boxes only."""
    gt = gt.detach()
    if not bool(torch.isfinite(gt).all()) or not bool(((gt[:, 2:] - gt[:, :2]) > 0).all()):
        raise GEOGeometryError("Invalid augmented GT for GEO (positive width/height required)",
                               gt_xyxy=gt.cpu().tolist(), groups=groups)
    if not bool(torch.isfinite(pred).all()):
        raise GEOGeometryError("Nonfinite final ordinary predictions", groups=groups)
    if sum(groups) != len(gt) or len(groups) != len(pred) or len(matches) != len(pred):
        raise GEOGeometryError("GEO GT grouping mismatch", groups=groups)
    count = sum(len(src) for src, _ in matches)
    total = pred.sum() * 0.0  # finite differentiable zero, also when M=0
    stats = dict(M=count, images=len(groups), gt_count=len(gt), neighbor_count=0, selected_count=0,
                 outside_edges=0, active_matches_1e8=0, active_matches_001=0,
                 E_sum=0.0, E_count=0, E_max=0.0, gt_with_overlap=0,
                 match_mean_sum=0.0, per_image=[]) if sample else None
    offset = 0
    for image, (n, (src, dst)) in enumerate(zip(groups, matches)):
        src, own_ids = src.to(pred.device), dst.to(pred.device) - offset
        local = gt[offset:offset + n]
        offset += n
        m = min(3, max(n - 1, 0))
        if len(src) and not bool(((own_ids >= 0) & (own_ids < n)).all()):
            raise GEOGeometryError("Matched GT identity belongs to a different image", image=image)
        if sample:
            stats["neighbor_count"] += len(src) * max(n - 1, 0)
            stats["selected_count"] += len(src) * m
            stats["per_image"].append(dict(gt=n, M=len(src), neighbors=max(n-1, 0), m_i=m))
            overlap = intersection(local[:, None], local[None, :]) > 0
            overlap.fill_diagonal_(False)
            stats["gt_with_overlap"] += int(overlap.any(1).sum())
        if not len(src):
            continue
        boxes, own = pred[image, src], local[own_ids]
        if sample:
            stats["outside_edges"] += int(torch.cat((boxes[:, :2] < own[:, :2], boxes[:, 2:] > own[:, 2:]), -1).sum())
        if not m:
            continue
        values = pair_overflow(boxes, own, local)
        rank_values = values.detach().clone()
        rank_values.scatter_(1, own_ids[:, None], -torch.inf)
        # Columns start in original GT order; stable sort breaks ties by GT ID.
        indices = torch.argsort(rank_values, dim=1, descending=True, stable=True)[:, :m]
        chosen = values.gather(1, indices)
        per_match = chosen.mean(1)  # zero E entries remain in m_i
        total = total + per_match.sum()
        if sample:
            detached = per_match.detach()
            other = torch.ones_like(values, dtype=torch.bool).scatter_(1, own_ids[:, None], False)
            e = values.detach()[other]
            stats["E_sum"] += float(e.sum())
            stats["E_count"] += e.numel()
            stats["E_max"] = max(stats["E_max"], float(e.max()))
            stats["active_matches_1e8"] += int((detached > 1e-8).sum())
            stats["active_matches_001"] += int((detached > .01).sum())
    raw = total / max(count, 1)
    if not bool(torch.isfinite(raw)):
        raise GEOGeometryError("Nonfinite GEO", groups=groups)
    if sample:
        stats.update(match_mean_sum=float(total.detach()), raw_geo=float(raw.detach()), finite=True)
    return raw, stats


class GEODetectionLoss(RTDETRDetectionLoss):
    def __init__(self, nc=1, **kwargs):
        if nc != 1:
            raise ValueError("GEO v1 supports nc=1 only")
        super().__init__(nc=nc, **kwargs)
        self.epoch, self.enabled, self.sample = 0, True, False
        self._capture_final, self._geo_context = False, None
        self.last_diagnostics = None

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix="", match_indices=None):
        if self._capture_final and postfix == "":
            self._capture_final = False  # final ordinary layer is first; never aux or DN
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                             gt_groups, masks=masks, gt_mask=gt_mask)
            self._geo_context = (pred_bboxes, gt_bboxes, gt_groups, match_indices)
        return super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                masks, gt_mask, postfix, match_indices)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        weight = FORMULA["lambda_geo"] * ramp(self.epoch) if self.enabled else 0.0
        self.last_diagnostics = None
        self._capture_final = bool(weight or self.sample)
        self._geo_context = None
        try:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            if weight:
                boxes, gt, groups, matches = self._geo_context
                with torch.autocast(device_type=boxes.device.type, enabled=False):
                    raw, stats = raw_geo_xyxy(xywh2xyxy(boxes.float()), xywh2xyxy(gt.detach().float()),
                                               groups, matches, self.sample)
                losses["loss_geo"] = weight * raw  # added AFTER all original DN/zero-DN paths
            elif self.sample:
                boxes, gt, groups, matches = self._geo_context
                stats = dict(M=sum(len(s) for s, _ in matches), images=len(groups), gt_count=len(gt),
                             raw_geo=None, match_mean_sum=0.0, geometry_skipped=True)
                stats["per_image"] = [dict(gt=n, M=len(s), neighbors=max(n-1,0), m_i=min(3,max(n-1,0)))
                                      for n,(s,_) in zip(groups,matches)]
                stats["neighbor_count"] = sum(r["M"]*r["neighbors"] for r in stats["per_image"])
                stats["selected_count"] = sum(r["M"]*r["m_i"] for r in stats["per_image"])
            if self.sample:
                self.last_diagnostics = dict(stats, epoch_zero=self.epoch, epoch_display=self.epoch+1,
                    ramp=ramp(self.epoch), weighted=float(losses["loss_geo"].detach()) if weight else 0.0,
                    original_losses={k: float(v.detach()) for k, v in losses.items() if k != "loss_geo"},
                    finite=all(bool(torch.isfinite(v)) for v in losses.values()))
            return losses
        finally:
            self._capture_final, self._geo_context = False, None
