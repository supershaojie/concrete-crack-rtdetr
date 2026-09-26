# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Explicit final assignment reuse without changing auxiliary/DN assignment."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.nn.modules.peq import finite


def aligned_iou(boxes, gt):
    """Ordinary continuous IoU; no prediction clipping, no additive union epsilon."""
    a0, a1 = boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2
    b0, b1 = gt[..., :2] - gt[..., 2:] / 2, gt[..., :2] + gt[..., 2:] / 2
    intersection = (torch.minimum(a1, b1) - torch.maximum(a0, b0)).clamp_min(0).prod(-1)
    union = boxes[..., 2:].prod(-1) + gt[..., 2:].prod(-1) - intersection
    if bool((union <= 0).any()):
        raise ValueError("PEQ IoU requires positive union")
    return intersection / union


def threshold_targets(ious):
    finite(ious)
    thresholds = ious.new_tensor([.50, .55, .60, .65, .70, .75, .80, .85, .90, .95])
    return (ious[..., None] >= thresholds).float().detach()


def quality_targets(boxes, gt, matches, gt_groups):
    with torch.no_grad(), torch.autocast(device_type=boxes.device.type, enabled=False):
        boxes, gt = boxes.detach().float(), gt.detach().float()
        finite(boxes, gt)
        bs, nq = boxes.shape[:2]
        if len(matches) != bs or sum(gt_groups) != len(gt):
            raise ValueError("PEQ assignment/GT inventory mismatch")
        target = boxes.new_zeros((bs, nq, 10))
        mask = torch.zeros((bs, nq), device=boxes.device, dtype=torch.bool)
        offset = 0
        for image, ((src, dst), count) in enumerate(zip(matches, gt_groups)):
            src, dst = src.to(boxes.device).long(), dst.to(boxes.device).long()
            if len(src) != len(dst) or len(src.unique()) != len(src) or len(dst.unique()) != len(dst):
                raise ValueError("PEQ assignments are not one-to-one")
            if len(src):
                if bool(((src < 0) | (src >= nq) | (dst < offset) | (dst >= offset+count)).any()):
                    raise ValueError("PEQ final assignment global GT offset mismatch")
                target[image, src] = threshold_targets(aligned_iou(boxes[image, src], gt[dst]))
                mask[image, src] = True
            offset += count
    return target, mask


def quality_loss(logits, target, matched):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, target = logits.float(), target.detach().float()
        finite(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        nm, nu = int(matched.sum()), int((~matched).sum())
        if nm + nu == 0:
            raise ValueError("PEQ cannot supervise an empty batch/query set")
        lm = bce[matched].mean() if nm else None
        lu = bce[~matched].mean() if nu else None
        loss = .5 * lm + .5 * lu if nm and nu else (lm if nm else lu)
        details = dict(matched_count=nm, unmatched_count=nu,
                       loss_matched=None if lm is None else float(lm.detach()),
                       loss_unmatched=None if lu is None else float(lu.detach()),
                       loss_peq=float(loss.detach()),
                       target_positive_count=target.sum((0, 1)).cpu().tolist(),
                       matched_positive_count=target[matched].sum(0).cpu().tolist())
    return loss, details


class PEQDetectionLoss(RTDETRDetectionLoss):
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None,
                peq_payload=None, return_details=False):
        if peq_payload is None:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            return (losses, None) if return_details else losses
        pred_bboxes, pred_scores = preds
        self.device = pred_bboxes.device
        finite(pred_bboxes, pred_scores, batch["bboxes"])
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        # One real final-main matcher call; its indices remain local to this invocation.
        final_matches = self.matcher(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups)
        total = self._get_loss(pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups,
                               match_indices=final_matches)
        if self.aux_loss:
            total.update(self._get_loss_aux(pred_bboxes[:-1], pred_scores[:-1],
                                            gt_bboxes, gt_cls, gt_groups, None, ""))
        if dn_meta is not None:
            matches = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], gt_groups)
            total.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=matches))
        else:
            total.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total.keys()})
        targets, mask = quality_targets(pred_bboxes[-1], gt_bboxes, final_matches, gt_groups)
        total["loss_peq"], details = quality_loss(peq_payload["quality_logits"], targets, mask)
        return (total, details) if return_details else total
