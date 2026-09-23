"""ROR v1: fixed final-match refinement reversals; no learnable parameters or tensor caches."""
from __future__ import annotations

import torch

from .loss import RTDETRDetectionLoss

CONFIG = dict(version="ror_v1", coefficient=0.10, delta=0.02, clip=1e-4,
              margin_cap=2.0, warmup_start=5, warmup_span=15)


def ramp(completed_epochs):
    if not isinstance(completed_epochs, int) or completed_epochs < 0:
        raise ValueError("ROR e must be the nonnegative integer number of completed epochs")
    return min(1.0, max(0.0, (completed_epochs - 5) / 15.0))


def finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("ROR nonfinite " + name)


def aligned_iou(boxes, targets):
    """Normalized cxcywh -> aligned ordinary IoU, detached FP32 (never M x M)."""
    with torch.autocast(device_type=boxes.device.type, enabled=False), torch.no_grad():
        boxes, targets = boxes.detach().float(), targets.detach().float()
        if boxes.shape != targets.shape or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("ROR aligned boxes must both be [M,4]")
        finite(boxes, "boxes"); finite(targets, "GT")
        if (boxes[:, 2:] <= 0).any() or (targets[:, 2:] <= 0).any():
            raise ValueError("ROR requires positive box widths/heights")
        lo0, hi0 = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
        lo1, hi1 = targets[:, :2] - targets[:, 2:] / 2, targets[:, :2] + targets[:, 2:] / 2
        intersection = (torch.minimum(hi0, hi1) - torch.maximum(lo0, lo1)).clamp(min=0).prod(-1)
        union = (hi0-lo0).prod(-1) + (hi1-lo1).prod(-1) - intersection
        if (union <= 0).any():
            raise ValueError("ROR nonpositive union")
        q = intersection / union
        finite(q, "IoU")
        return q


def image_loss(q0, q1, raw_logits, classes, detailed=False):
    """One image's matched GT-class raw logits. q inputs are always stop-gradient."""
    with torch.autocast(device_type=raw_logits.device.type, enabled=False):
        q0, q1, z = q0.detach().float(), q1.detach().float(), raw_logits.float()
        if not (q0.shape == q1.shape == z.shape == classes.shape) or z.ndim != 1:
            raise ValueError("ROR per-image vectors must have equal shape [M]")
        for value, name in ((q0, "q0"), (q1, "q1"), (z, "raw logits")):
            finite(value, name)
        if ((q0 < 0) | (q0 > 1) | (q1 < 0) | (q1 > 1)).any():
            raise ValueError("ROR quality must be ordinary IoU in [0,1]")
        same = classes[:, None] == classes[None, :]
        before = q0[None, :] - q0[:, None]  # j - i
        after = q1[:, None] - q1[None, :]   # i - j
        mask = same & (before >= 0.02) & (after >= 0.02)
        count = int(mask.sum())
        # sum(z)*0 could overflow for large, finite logits. This connected zero cannot.
        zero = z[:0].sum()
        row = dict(positives=z.numel(), comparable_pairs=int(same.triu(diagonal=1).sum()),
                   eligible_pairs=count, violating_pairs=0, raw_loss=0.0,
                   pair_huber_sum=0.0, pair_weighted_sum=0.0)
        if detailed:
            row.update(q0=q0.tolist(),q1=q1.tolist(),quality_change=(q1-q0).tolist())
        if count == 0:
            return zero, row
        ell = torch.log(q1.clamp(1e-4, 1 - 1e-4)) - torch.log1p(-q1.clamp(1e-4, 1 - 1e-4))
        margin = (ell[:, None] - ell[None, :])[mask].clamp(max=2.0)
        weight = after[mask]
        gap = (z[:, None] - z[None, :])[mask]
        violation = (margin - gap).relu()
        # Algebraic Huber avoids evaluating a huge square in the unused branch.
        quadratic = violation.clamp(max=1.0)
        huber = 0.5 * quadratic.square() + (violation - quadratic)
        weighted = weight * huber
        loss = weighted.sum() / count
        finite(loss, "loss")
        row.update(violating_pairs=int((violation > 0).sum()), raw_loss=float(loss.detach()),
                   pair_huber_sum=float(huber.detach().sum()), pair_weighted_sum=float(weighted.detach().sum()))
        if detailed:
            row.update(pairs=mask.nonzero().tolist(), pre_gap=before[mask].tolist(),
                       post_gap=weight.tolist(), margin=margin.tolist(), logit_gap=gap.detach().tolist(),
                       huber=huber.detach().tolist(), weighted=weighted.detach().tolist())
        return loss, row


def matched_loss(before, after, raw_logits, targets, indices, detailed=False):
    """Matcher GT indices are batch-flat, query indices are local to each image."""
    if before.shape != after.shape or before.shape[:2] != raw_logits.shape[:2]:
        raise ValueError("ROR before/after/logits query alignment mismatch")
    batch_size = raw_logits.shape[0]
    if batch_size == 0 or len(indices) != batch_size or len(targets["gt_groups"]) != batch_size:
        raise ValueError("ROR actual batch size/match count mismatch")
    with torch.autocast(device_type=raw_logits.device.type, enabled=False):
        total, rows, offset = raw_logits.float()[:, :0].sum(), [], 0
        for n, ((query, gt), group) in enumerate(zip(indices, targets["gt_groups"])):
            query, gt = query.to(raw_logits.device), gt.to(raw_logits.device)
            if len(query) != len(gt) or (gt < offset).any() or (gt >= offset + group).any():
                raise ValueError("ROR matcher GT indices are not aligned batch-flat indices")
            if len(query.unique()) != len(query) or len(gt.unique()) != len(gt):
                raise ValueError("ROR requires unique regular final matching")
            labels = targets["cls"][gt].long()
            q0 = aligned_iou(before[n, query], targets["bboxes"][gt])
            q1 = aligned_iou(after[n, query], targets["bboxes"][gt])
            value, row = image_loss(q0, q1, raw_logits[n, query, labels], labels, detailed)
            total = total + value
            rows.append(row)
            offset += group
        return total / batch_size, rows


class RORLoss(RTDETRDetectionLoss):
    """Only loss_ror enters the sum; diagnostics contain Python numbers, never tensors."""

    def __init__(self, nc=1):
        super().__init__(nc=nc, use_vfl=True)
        self.config = dict(CONFIG)
        self.set_epoch(0)
        self.reset_statistics()

    def set_epoch(self, completed_epochs):
        self.completed_epochs = completed_epochs
        self.ror_weight = CONFIG["coefficient"] * ramp(completed_epochs)

    def reset_statistics(self):
        self.statistics = dict(batches=0, images=0, positives=0, comparable_pairs=0,
                               eligible_pairs=0, violating_pairs=0, nonzero_batches=0,
                               raw_sum=0.0, weighted_sum=0.0)

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None,
                ror_enabled=False, ror_details=None):
        if not ror_enabled or self.ror_weight == 0:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            for key, value in losses.items(): finite(value, key)
            return losses
        if ror_details is None or not {"before", "after"} <= ror_details.keys():
            raise RuntimeError("Active ROR is missing same-forward CBR details")
        boxes, scores = preds
        if not torch.equal(ror_details["after"], boxes[-1]):
            raise RuntimeError("ROR details do not identify the final regular prediction")
        # Exactly one final Hungarian call; original auxiliary matches remain per layer.
        indices = self.matcher(boxes[-1], scores[-1], batch["bboxes"], batch["cls"], batch["gt_groups"])
        losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta, final_match_indices=indices)
        raw, rows = matched_loss(ror_details["before"], boxes[-1], scores[-1], batch, indices)
        losses["loss_ror"] = raw * self.ror_weight  # after native DN suffix expansion, once only
        for key, value in losses.items(): finite(value, key)
        stats = self.statistics
        stats["batches"] += 1; stats["images"] += len(rows)
        for key in ("positives", "comparable_pairs", "eligible_pairs", "violating_pairs"):
            stats[key] += sum(row[key] for row in rows)
        value = float(raw.detach())
        stats["nonzero_batches"] += int(value > 0)
        stats["raw_sum"] += value; stats["weighted_sum"] += value * self.ror_weight
        return losses
