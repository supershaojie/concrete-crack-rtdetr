"""DTR-v1-dnmedian-excess. No model parameters, RNG, extra matching, or DN loss changes."""
from __future__ import annotations

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.ops import xywh2xyxy

FORMULA = dict(id="DTR-v1-dnmedian-excess", enabled=True, lambda_dtr=0.20,
               delta=0.05, scale_floor_pixels=1.0, ramp_start=5, ramp_full=20)


def ramp(epoch):
    """epoch is the native, zero-based trainer.epoch, including on resume."""
    return min(max((int(epoch) - 5) / 15.0, 0.0), 1.0)


def require(condition, message):
    if not condition:
        raise ValueError("DTR: " + message)


def phi(u):
    return u.square() / (torch.sqrt(u.square() + 0.05 ** 2) + 0.05)


def index_pair(pair, image, offset, count, queries, device):
    require(len(pair) == 2, "invalid index pair")
    q, g = pair
    require(q.ndim == g.ndim == 1 and q.numel() == g.numel(), "index length mismatch")
    require(q.dtype == g.dtype == torch.long, "indices must be int64")
    q, g = q.detach().to(device), g.detach().to(device)
    require(bool(((q >= 0) & (q < queries)).all()), f"query index outside image {image}")
    require(bool(((g >= offset) & (g < offset + count)).all()), f"GT identity crosses image {image}")
    require(q.unique().numel() == q.numel(), "duplicate query index")
    return q, g


def tolerance_loss(boxes, gt, matches, gt_groups, dn_boxes, dn_matches, input_hw, sample=False, gt_wh=None):
    """All boxes are normalized xyxy; mappings use batch-flat GT IDs.

    The helper intentionally accepts missing references; the criterion separately
    checks the native dn_meta contract. No DN/GT tensor is retained after return.
    """
    require(len(input_hw) == 2 and all(int(x) > 0 for x in input_hw), "invalid actual network H/W")
    require(boxes.ndim == 3 and boxes.shape[-1] == 4, "ordinary boxes must be B,Q,4")
    require(len(matches) == len(gt_groups) == len(boxes), "batch/matching length mismatch")
    require(sum(gt_groups) == len(gt) and gt.shape == (len(gt), 4), "GT count/shape mismatch")
    require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(gt).all()), "nonfinite ordinary/GT coordinates")
    if dn_boxes is not None:
        require(dn_boxes.ndim == 3 and dn_boxes.shape[0] == len(boxes) and dn_boxes.shape[-1] == 4, "DN shape mismatch")
        require(dn_matches is not None and len(dn_matches) == len(boxes), "DN mapping length mismatch")
        require(bool(torch.isfinite(dn_boxes).all()), "nonfinite DN coordinates")
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        x, g = boxes.float(), gt.detach().float()
        widths = g[:, 2:] - g[:, :2] if gt_wh is None else gt_wh.detach().to(device=x.device, dtype=torch.float32)
        require(widths.shape == (len(g), 2) and bool(torch.isfinite(widths).all()) and bool((widths >= 0).all()),
                "invalid detached GT widths/heights")
        teacher = None if dn_boxes is None else dn_boxes.detach().float()
        # An empty slice provides a differentiable zero without overflow in a sum.
        total = x.reshape(-1)[:0].sum()
        m = sum(len(pair[0]) for pair in matches)
        stats = dict(M=m, with_dn=0, without_dn=m, K={}, edges=0,
                     radius_over_scale_sum=[0.] * 4, error_over_scale_sum=[0.] * 4,
                     excess_sum=[0.] * 4, active_count=[0] * 4)
        offset = 0
        image_rows, query_rows, gt_rows, reference_rows = [], [], [], []
        h, w = map(int, input_hw)
        floor = g.new_tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])
        for image, count in enumerate(gt_groups):
            q, gids = index_pair(matches[image], image, offset, count, x.shape[1], x.device)
            if teacher is not None:
                dq, dg = index_pair(dn_matches[image], image, offset, count, teacher.shape[1], x.device)
            else:
                dq = dg = torch.empty(0, dtype=torch.long, device=x.device)
            # Build an explicit ragged map on detached integer identities. The
            # expensive float statistics below are vectorized across all matches.
            by_gt = {}
            for dqi, dgi in zip(dq.cpu().tolist(), dg.cpu().tolist()):
                by_gt.setdefault(dgi, []).append(dqi)
            for qi, gi in zip(q.cpu().tolist(), gids.cpu().tolist()):
                refs = by_gt.get(gi, [])
                if not refs:
                    continue
                image_rows.append(image)
                query_rows.append(qi)
                gt_rows.append(gi)
                reference_rows.append(refs)
            offset += count
        if reference_rows:
            sizes = [len(r) for r in reference_rows]
            ks = torch.tensor(sizes, device=x.device)
            images = torch.tensor(image_rows, device=x.device)
            queries = torch.tensor(query_rows, device=x.device)
            truths = g[torch.tensor(gt_rows, device=x.device)]
            references = torch.tensor([r + [0] * (max(sizes) - len(r)) for r in reference_rows], device=x.device)
            mask = torch.arange(max(sizes), device=x.device)[None, :] < ks[:, None]
            err = (teacher[images[:, None], references] - truths[:, None]).abs()
            # Padding here is a temporary ragged-statistics mask, never a DN query
            # selected as reference. Both selected order statistics are finite.
            ordered = err.masked_fill(~mask[..., None], float("inf")).sort(dim=1).values
            rows = torch.arange(len(sizes), device=x.device)
            radius = (ordered[rows, (ks - 1) // 2] + ordered[rows, ks // 2]) * .5
            wh = widths[torch.tensor(gt_rows, device=x.device)]
            scale = torch.maximum(wh.repeat(1, 2), floor).detach()
            ordinary_error = (x[images, queries] - truths).abs()
            u = torch.relu(ordinary_error - radius) / scale
            total = total + phi(u).sum()
            if sample:
                from collections import Counter
                stats["with_dn"] = len(sizes)
                stats["K"] = {str(k): v for k, v in Counter(sizes).items()}
                for key, value in (("radius_over_scale_sum", radius / scale),
                                   ("error_over_scale_sum", ordinary_error / scale),
                                   ("excess_sum", u), ("active_count", (u > 0).long())):
                    stats[key] = value.detach().sum(0).cpu().tolist()
        raw = total / (4 * max(m, 1))
        require(bool(torch.isfinite(raw)), "nonfinite new loss")
        if sample:
            stats["without_dn"] = m - stats["with_dn"]
            stats["edges"] = 4 * stats["with_dn"]
            stats["raw_dtr"] = float(raw.detach())
        return raw, stats if sample else None


class DTRDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, nc=1, **kwargs):
        require(nc == 1, "v1 supports only nc=1")
        super().__init__(nc=nc, **kwargs)  # keep native VFL/matcher defaults
        self.enabled, self.epoch, self.sample = True, 0, False
        self.input_hw = None
        self.last_diagnostics = None
        self._capture_final = False
        self._final_matches = None

    def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                  masks=None, gt_mask=None, postfix="", match_indices=None):
        if self._capture_final:
            require(postfix == "" and self._final_matches is None, "unexpected final-layer capture")
            self._capture_final = False  # auxiliary and DN cannot write this context
            if match_indices is None:
                match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                             gt_groups, masks=masks, gt_mask=gt_mask)
            self._final_matches = [(q.detach(), g.detach()) for q, g in match_indices]
        # Only THIS final _get_loss sees the captured match. super.forward never
        # receives it, so encoder/auxiliary/DN retain their original assignments.
        return super()._get_loss(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                                 masks, gt_mask, postfix, match_indices)

    def _dn_indices(self, preds, batch, dn_bboxes, dn_scores, meta):
        if meta is None:
            require(dn_bboxes is None and dn_scores is None, "DN tensors without metadata")
            return None
        require(all(k in meta for k in ("dn_pos_idx", "dn_num_group", "dn_num_split")), "incomplete dn_meta")
        k, pos = meta["dn_num_group"], meta["dn_pos_idx"]
        require(isinstance(k, int) and k >= 1, "invalid dn_num_group")
        require(len(pos) == len(batch["gt_groups"]), "dn_pos_idx batch length mismatch")
        require(dn_bboxes is not None and dn_scores is not None, "metadata without DN outputs")
        require(dn_bboxes.ndim == dn_scores.ndim == 4 and dn_bboxes.shape[-1] == 4
                and dn_scores.shape[-1] == 1 and dn_bboxes.shape[:3] == dn_scores.shape[:3], "DN output shape mismatch")
        require(dn_bboxes.shape[0] == preds[0].shape[0] - 1
                and dn_bboxes.shape[1] == preds[0].shape[1], "final DN layer/batch mismatch")
        require(list(meta["dn_num_split"]) == [dn_bboxes.shape[2], preds[0].shape[2]], "DN split mismatch")
        for p, n in zip(pos, batch["gt_groups"]):
            require(torch.is_tensor(p) and p.dtype == torch.long and p.ndim == 1 and p.numel() == n * k,
                    "positive DN length/dtype mismatch")
            require(bool(((p >= 0) & (p < dn_bboxes.shape[2])).all()), "positive DN index out of bounds")
        require(bool(torch.isfinite(dn_bboxes[-1]).all()), "nonfinite final DN coordinates")
        return self.get_dn_match_indices(pos, k, batch["gt_groups"])

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        self.last_diagnostics = None
        active = self.enabled and ramp(self.epoch) > 0
        self._capture_final, self._final_matches = active, None
        try:
            if active:
                require(self.input_hw is not None, "model must supply actual batch img H/W")
                dn_matches = self._dn_indices(preds, batch, dn_bboxes, dn_scores, dn_meta)
                if "batch_idx" in batch:
                    expected = torch.repeat_interleave(torch.arange(len(batch["gt_groups"]), device=batch["bboxes"].device),
                                                       torch.tensor(batch["gt_groups"], device=batch["bboxes"].device))
                    require(torch.equal(batch["batch_idx"].detach().view(-1), expected), "flattened GT/image order mismatch")
            # Complete all original losses, including native no-DN zero keys first.
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            stats = dict(skipped=True, epoch=self.epoch, ramp=ramp(self.epoch)) if self.sample else None
            if active:
                with torch.autocast(device_type=preds[0].device.type, enabled=False):
                    raw, stats = tolerance_loss(xywh2xyxy(preds[0][-1].float()),
                        xywh2xyxy(batch["bboxes"].detach().float()), self._final_matches, batch["gt_groups"],
                        None if dn_bboxes is None else xywh2xyxy(dn_bboxes[-1].detach().float()),
                        dn_matches, self.input_hw, self.sample, gt_wh=batch["bboxes"][..., 2:])
                losses["loss_dtr"] = FORMULA["lambda_dtr"] * ramp(self.epoch) * raw
                if self.sample:
                    stats.update(skipped=False, epoch=self.epoch, ramp=ramp(self.epoch),
                                 weighted_dtr=float(losses["loss_dtr"].detach()))
            if self.sample:
                stats["losses"] = {k: float(v.detach()) for k, v in losses.items()}
                stats["all_losses_finite"] = all(bool(torch.isfinite(v)) for v in losses.values())
                self.last_diagnostics = stats
            return losses
        finally:
            self._capture_final, self._final_matches = False, None
