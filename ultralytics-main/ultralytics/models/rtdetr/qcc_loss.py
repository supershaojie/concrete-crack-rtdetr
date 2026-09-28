"""QCC-v1-capK3: classification-only quality-capped competition on native final matches.

No matcher, VFL, regression, auxiliary or DN formula is replaced. The short-lived
capture below observes the native VFL input, before any matched-only slicing can
hide unmatched queries. Nothing from a batch survives forward except Python stats.
"""
from __future__ import annotations

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

FORMULA = dict(id="QCC-v1-capK3", enabled=True, lambda_qcc=0.10, topk=3, ramp_start=5, ramp_full=20)


def ramp(epoch):
    return max(0.0, min(1.0, (epoch - 5) / 15))


def moments(values):
    v = values.detach().double().reshape(-1)
    return dict(count=v.numel(), sum=float(v.sum()), sumsq=float(v.square().sum()),
                min=float(v.min()) if v.numel() else None, max=float(v.max()) if v.numel() else None)


def group_kl(logits, q, a):
    """One already-selected group; FP64 supported for independent derivative tests.

    q and a are detached here as well as at selection. Equality uses a strict
    detached sigmoid comparison, never torch.minimum's shared boundary gradient.
    Callers skip q=0 and empty candidates, so neither logit(0) nor padding exists.
    """
    q, a = q.detach(), a.detach()
    if not (torch.isfinite(q).all() and 0 < q <= 1 and torch.isfinite(a).all() and (a > 0).all()):
        raise ValueError("QCC group requires finite 0<q<=1 and a>0")
    if q == 1:
        positive, entropy = logits[0], q.new_zeros(())
    else:
        cap = q.log() - torch.log1p(-q)
        positive = torch.where(logits[0].detach().sigmoid() < q, logits[0], cap)
        entropy = q * q.log() + (1 - q) * torch.log1p(-q)
    group = torch.cat((positive.reshape(1), logits[1:] + a.log(), logits.new_zeros(1)))
    return torch.logsumexp(group, 0) - q * positive + entropy, group


@torch.no_grad()
def select_groups(boxes, logits, gt, gt_groups, matches, quality):
    """Native cxcywh bbox_iou/eps, per image ownership over ALL GT, stable top K.

    Matcher GT indices are global flat indices; owner indices are per image.
    The list has only actual eligible competitors, never padded query indices.
    """
    groups, offset = [], 0
    excluded_better, no_candidates = 0, 0
    for b, ((src, dst), n_gt) in enumerate(zip(matches, gt_groups)):
        src, dst = src.to(boxes.device), dst.to(boxes.device)
        if len(src) and not ((dst >= offset) & (dst < offset + n_gt)).all():
            raise ValueError("QCC native global GT index is outside this image")
        available = torch.ones(boxes.shape[1], dtype=torch.bool, device=boxes.device)
        available[src] = False
        unmatched = available.nonzero(as_tuple=True)[0]  # increasing original query index
        if len(src) == 0 or len(unmatched) == 0 or n_gt == 0:
            no_candidates += len(src)
            offset += n_gt
            continue
        ious = bbox_iou(boxes[b, unmatched, None], gt[None, offset:offset + n_gt], xywh=True).squeeze(-1)
        if not torch.isfinite(ious).all() or (ious < 0).any() or (ious > 1).any():
            raise ValueError("QCC native candidate IoU is nonfinite or outside [0,1]")
        best, owner = ious.max(-1)
        second = ious.topk(2, dim=-1).values[:, 1] if n_gt > 1 else torch.zeros_like(best)
        a = (best - second).clamp_min(0)
        p = logits[b, :, 0].sigmoid()
        for pos, global_gt in zip(src, dst):
            q = quality[b, pos]
            owned = (owner == global_gt - offset) & (a > 0)
            excluded_better += int((owned & (best > q)).sum())
            eligible = (owned & (best <= q)).nonzero(as_tuple=True)[0]
            if q == 0 or len(eligible) == 0:
                no_candidates += 1
                continue
            scores = a[eligible] * p[unmatched[eligible]]
            chosen = eligible[torch.argsort(scores, descending=True, stable=True)[:3]]
            competitors = unmatched[chosen]
            h = (a[chosen] * p[competitors]).max()
            groups.append(dict(batch=b, positive=pos, gt=global_gt, competitors=competitors,
                               q=q, a=a[chosen], h=h, iou=best[chosen]))
        offset += n_gt
    if offset != len(gt):
        raise ValueError("QCC gt_groups does not describe the actual augmented GT")
    return groups, dict(no_candidates=no_candidates, excluded_better=excluded_better)


def competition(boxes, logits, gt, gt_groups, matches, quality, sample=False):
    """Returns raw QCC and detached diagnostics, with geometry disconnected."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = logits.float()  # retains the classification gradient
        if z.shape[-1] != 1:
            raise NotImplementedError("QCC v1 supports nc=1 only")
        b, g, q = boxes.detach().float(), gt.detach().float(), quality.detach().float()
        if not all(torch.isfinite(x).all() for x in (z, b, g, q)):
            raise ValueError("QCC received nonfinite logits/boxes/GT/native VFL quality")
        if (q < 0).any() or (q > 1).any():
            raise ValueError("QCC native VFL quality outside [0,1]; investigate its source")
        if (b[..., 2:] < 0).any() or (g[..., 2:] < 0).any():
            raise ValueError("QCC received negative box dimensions")
        M = sum(len(src) for src, _ in matches)
        # Empty-slice sum is a connected zero without overflow on a large finite sum.
        raw = z.reshape(-1)[:0].sum()
        groups, counts = select_groups(b, z.detach(), g, gt_groups, matches, q) if M else ([], dict(no_candidates=0, excluded_better=0))
        kls, weights, aa, cp, ci = [], [], [], [], []
        capped = higher = 0
        for group in groups:
            bi, pos, comp = group["batch"], group["positive"], group["competitors"]
            selected = torch.cat((z[bi, pos, 0].reshape(1), z[bi, comp, 0]))
            kl, _ = group_kl(selected, group["q"], group["a"])
            raw = raw + group["h"] * kl / max(M, 1)
            if sample:
                kls.append(kl.detach()); weights.append(group["h"]); aa.append(group["a"])
                cp.append(z[bi, comp, 0].detach().sigmoid()); ci.append(group["iou"])
                capped += int(group["q"] < 1 and z[bi, pos, 0].detach().sigmoid() >= group["q"])
                higher += int((cp[-1] > z[bi, pos, 0].detach().sigmoid()).any())
        if not torch.isfinite(raw):
            raise ValueError("QCC nonfinite group KL/raw loss")
        stats = None
        if sample:
            idx, _ = RTDETRDetectionLoss._get_index(matches)
            matched_q, matched_p = q[idx], z.detach()[idx].squeeze(-1).sigmoid()
            empty = z.detach().new_empty(0)
            cat = lambda xs: torch.cat([x.reshape(-1) for x in xs]) if xs else empty
            stats = dict(M=M, active_groups=len(groups), selected=sum(len(x["competitors"]) for x in groups),
                         cap_active_groups=capped, competitor_higher_groups=higher, **counts,
                         p_lt_q=int((matched_p < matched_q).sum()), p_ge_q=int((matched_p >= matched_q).sum()),
                         moments={"q": moments(matched_q), "p": moments(matched_p), "p_minus_q": moments(matched_p - matched_q),
                                  "a": moments(cat(aa)), "h": moments(cat(weights)), "KL": moments(cat(kls)),
                                  "competitor_p": moments(cat(cp)), "competitor_iou": moments(cat(ci))},
                         weighted_KL_sum=float(raw.detach()) * max(M, 1), raw_qcc=float(raw.detach()), finite=True)
        return raw, stats


class QCCDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, nc=1, **kwargs):
        if nc != 1:
            raise NotImplementedError("QCC-v1-capK3 supports nc=1 only")
        super().__init__(nc=nc, **kwargs)
        self.enabled, self.epoch, self.sample = True, 0, False
        self.last_diagnostics = None
        self._need_final = False
        self._capturing = False
        self._captured = None
        self._native_q = None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        if self._capturing:
            self._native_q = gt_scores.detach()  # the actual unmodified native VFL argument
        return super()._get_loss_class(pred_scores, targets, gt_scores, num_gts, postfix)

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix="", match_indices=None):
        if not self._need_final:
            return super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                     masks, gt_mask, postfix, match_indices)
        self._need_final = False  # first call only: native final ordinary layer
        if postfix:
            raise RuntimeError("Unexpected QCC capture outside final ordinary layer")
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                         masks=masks, gt_mask=gt_mask)
        self._capturing = True
        try:
            result = super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                       masks, gt_mask, postfix, match_indices)
            self._captured = (pred_bboxes, pred_scores, gt_bboxes, gt_groups, match_indices, self._native_q)
            return result
        finally:
            self._capturing = False

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_diagnostics = None
        active = self.enabled and self.training and ramp(self.epoch) > 0
        self._need_final = active
        try:
            # Complete ALL native ordinary, auxiliary, DN or DN-zero paths first.
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            if active:
                raw, stats = competition(*self._captured, sample=self.sample)
                losses["loss_qcc"] = FORMULA["lambda_qcc"] * ramp(self.epoch) * raw
                if stats is not None:
                    self.last_diagnostics = dict(stats, epoch=self.epoch, ramp=ramp(self.epoch),
                        lambda_qcc=FORMULA["lambda_qcc"], loss_qcc=float(losses["loss_qcc"].detach()),
                        original_finite=all(bool(torch.isfinite(v).all()) for k, v in losses.items() if k != "loss_qcc"),
                        original_losses={k: float(v.detach()) for k, v in losses.items() if k != "loss_qcc"})
            elif self.sample:
                self.last_diagnostics = dict(epoch=self.epoch, ramp=ramp(self.epoch), lambda_qcc=0.10,
                    M=sum(min(n, preds[1].shape[2]) for n in batch["gt_groups"]),
                    M_source="native complete Hungarian cardinality min(GT, ordinary queries); no additional matching",
                    grouping="disabled; no additional IoU or sorting", loss_qcc=0.0,
                    original_finite=all(bool(torch.isfinite(v).all()) for v in losses.values()),
                    original_losses={k: float(v.detach()) for k, v in losses.items()})
            return losses
        finally:
            self._need_final = self._capturing = False
            self._captured = self._native_q = None
