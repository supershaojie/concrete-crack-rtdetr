"""ARG v2: final regular-query Axis Repair Gain Loss (no trainable state)."""
from __future__ import annotations

import torch

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss

FORMULA = dict(version="ARG-v2-q2", lambda_arg=0.20, epsilon_w=0.05, gamma_quality=2.0, eps_num=1e-7)


def ramp(epoch):
    """epoch is the number of completed epochs, including after native resume."""
    return min(max((int(epoch) - 5) / 15.0, 0.0), 1.0)


def xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), -1)


def iou2d(b, g):
    wh = (torch.minimum(b[..., 2:], g[..., 2:]) - torch.maximum(b[..., :2], g[..., :2])).clamp_min(0)
    intersection = wh.prod(-1)
    union = (b[..., 2:] - b[..., :2]).prod(-1) + (g[..., 2:] - g[..., :2]).prod(-1) - intersection
    return intersection / union.clamp_min(FORMULA["eps_num"]), union


def interval_loss(b, g):
    u, v = b.unbind(-1)
    gu, gv = g.unbind(-1)
    intersection = (torch.minimum(v, gv) - torch.maximum(u, gu)).clamp_min(0)
    union = v - u + gv - gu - intersection
    cover = torch.maximum(v, gv) - torch.minimum(u, gu)
    eps = FORMULA["eps_num"]
    return 1 - intersection / union.clamp_min(eps) + (cover - union) / cover.clamp_min(eps), union, cover


class ARGGeometryError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"ARG invalid geometry: {details}")


def _valid(b, g, coordinate_space):
    bad = ~(torch.isfinite(b).all(-1) & torch.isfinite(g).all(-1))
    wh_b = b[:, 2:] if coordinate_space == "cxcywh" else b[:, 2:] - b[:, :2]
    wh_g = g[:, 2:] if coordinate_space == "cxcywh" else g[:, 2:] - g[:, :2]
    bad |= (wh_b <= 0).any(-1) | (wh_g <= 0).any(-1)
    if bad.any():
        idx = bad.nonzero().flatten()[:8]
        raise ARGGeometryError(dict(space=coordinate_space, matched_rows=idx.tolist(),
                                    prediction=b[idx].detach().cpu().tolist(), gt=g[idx].detach().cpu().tolist(),
                                    pred_wh=wh_b[idx].detach().cpu().tolist(), gt_wh=wh_g[idx].detach().cpu().tolist()))


def geometry(prediction, target):
    """Explicit repair boxes also define the protected tiny-area case unambiguously."""
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, g = prediction.float(), target.detach().float()
        _valid(p, g, "cxcywh")
        b, g = xyxy(p), xyxy(g)
        _valid(b, g, "xyxy")
        with torch.no_grad():
            frozen = b.detach()
            repair_x = torch.stack((g[:, 0], frozen[:, 1], g[:, 2], frozen[:, 3]), -1)
            repair_y = torch.stack((frozen[:, 0], g[:, 1], frozen[:, 2], g[:, 3]), -1)
            q, u = iou2d(frozen, g)
            qx, ux = iou2d(repair_x, g)
            qy, uy = iou2d(repair_y, g)
            dx, dy = (qx - q).clamp_min(0), (qy - q).clamp_min(0)
            epsilon = FORMULA["epsilon_w"]
            wx = 2 * (epsilon + dx) / (2 * epsilon + dx + dy)
            wy = 2 * (epsilon + dy) / (2 * epsilon + dx + dy)
            quality_gate = q.square()
        ell_x, union_x, cover_x = interval_loss(b[:, [0, 2]], g[:, [0, 2]])
        ell_y, union_y, cover_y = interval_loss(b[:, [1, 3]], g[:, [1, 3]])
        per_match = 0.5 * (wx * ell_x + wy * ell_y)
        gated = quality_gate * per_match
        raw = gated.sum() / max(len(b), 1)
        protected = torch.stack((u, ux, uy, union_x, cover_x, union_y, cover_y), -1) < FORMULA["eps_num"]
        details = dict(q=q, qx=qx, qy=qy, dx=dx, dy=dy, wx=wx, wy=wy,
                       quality_gate=quality_gate, ell_x=ell_x, ell_y=ell_y,
                       per_match=per_match, gated=gated, protected=protected.any(-1))
        if not torch.isfinite(raw):
            raise ARGGeometryError({k: v.detach().cpu().tolist()[:8] for k, v in details.items()})
        return raw, details


def sampled_statistics(details):
    """Detached sufficient statistics; each match has one vote across batches."""
    values = {k: details[k].detach().float() for k in
              ("q", "quality_gate", "wx", "wy", "ell_x", "ell_y", "per_match", "gated")}
    values["abs_wx_minus_one"] = (values["wx"] - 1).abs()
    moments = {k: dict(count=v.numel(), sum=float(v.sum()), sumsq=float(v.square().sum())) for k, v in values.items()}
    bins = []
    q = values["q"]
    for lo, hi in ((0, .5), (.5, .75), (.75, .9), (.9, 1.0)):
        mask = (q >= lo) & ((q <= hi) if hi == 1 else (q < hi))
        bins.append(dict(lower=lo, upper=hi, upper_inclusive=hi == 1, count=int(mask.sum()),
                         raw_v1_sum=float(values["per_match"][mask].sum()),
                         raw_v2_sum=float(values["gated"][mask].sum())))
    return dict(moments=moments, iou_bins=bins)


def aggregate_samples(samples):
    """Count/sum aggregation, including null means for empty IoU intervals."""
    import math
    if not samples:
        return dict(matched_count=0, moments={}, iou_bins=[])
    moments = {}
    for key in samples[0]["statistics"]["moments"]:
        rows = [s["statistics"]["moments"][key] for s in samples]
        count, total, square = (sum(r[k] for r in rows) for k in ("count", "sum", "sumsq"))
        mean = total / count if count else None
        moments[key] = dict(count=count, sum=total, sumsq=square, mean=mean,
                            std=math.sqrt(max(0.0, square / count - mean * mean)) if count else None)
    bins = []
    for index, first in enumerate(samples[0]["statistics"]["iou_bins"]):
        rows = [s["statistics"]["iou_bins"][index] for s in samples]
        count = sum(r["count"] for r in rows)
        totals = {k: sum(r[k] for r in rows) for k in ("raw_v1_sum", "raw_v2_sum")}
        bins.append(dict(lower=first["lower"], upper=first["upper"], upper_inclusive=first["upper_inclusive"],
                         count=count, **totals,
                         raw_v1_mean=totals["raw_v1_sum"] / count if count else None,
                         raw_v2_mean=totals["raw_v2_sum"] / count if count else None))
    return dict(matched_count=moments["q"]["count"], moments=moments, iou_bins=bins)


class ARGv2DetectionLoss(RTDETRDetectionLoss):
    """Reuse the native final match once; leave auxiliary and DN assignments native."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.enabled = True
        self.sample = False
        self.last_diagnostics = None
        self.last_matches = None

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_diagnostics = None
        self.last_matches = None
        factor = ramp(self.epoch) if self.enabled else 0.0
        if factor == 0:
            # No new matching, geometry, zeros or autograd graph during warmup/validation.
            return super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
        boxes, scores = preds
        self.device = boxes.device
        gt, cls, groups = batch["bboxes"], batch["cls"], batch["gt_groups"]
        matches = self.matcher(boxes[-1], scores[-1], gt, cls, groups)
        total = self._get_loss(boxes[-1], scores[-1], gt, cls, groups, match_indices=matches)
        if self.aux_loss:
            # Encoder and each earlier decoder retain their own native matching.
            total.update(self._get_loss_aux(boxes[:-1], scores[:-1], gt, cls, groups))
        if dn_meta is not None:
            dn_matches = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], groups)
            total.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=dn_matches))
        else:
            # Complete the mother's zero fill BEFORE adding ARG (no loss_arg_dn).
            total.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in list(total)})
        idx, gt_idx = self._get_index(matches)
        try:
            raw, details = geometry(boxes[-1][idx], gt[gt_idx])
        except ARGGeometryError as error:
            error.details.update(batch_indices=idx[0].tolist(), query_indices=idx[1].tolist(), gt_indices=gt_idx.tolist())
            raise
        total["loss_arg"] = FORMULA["lambda_arg"] * factor * raw
        if self.sample:
            # Scalars only; never retain an activation graph across batches/checkpoints.
            original = sum(v.detach() for k, v in total.items() if k != "loss_arg")
            localization = sum(v.detach() for k, v in total.items() if "bbox" in k or "giou" in k)
            self.last_diagnostics = dict(epoch=int(self.epoch), ramp=factor, matches=len(gt_idx),
                protected_count=int(details["protected"].sum()),
                raw=float(raw.detach()), weighted=float(total["loss_arg"].detach()), L0=float(original),
                raw_v1_reference=float(details["per_match"].detach().sum() / max(len(gt_idx), 1)),
                statistics=sampled_statistics(details),
                localization=float(localization), ratio_to_localization=float(total["loss_arg"].detach() / localization.clamp_min(1e-7)),
                **{k: float(v.detach().float().mean()) if len(v) else None for k, v in details.items()})
            self.last_matches = [(a.cpu().tolist(), b.cpu().tolist()) for a, b in matches]
        return total
