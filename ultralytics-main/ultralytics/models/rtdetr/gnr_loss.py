"""GNR v1: detached, GT-conditioned redundancy weights for final ordinary negatives.

No model state, statistical buffers, matching changes, or inference operations.
The VFL parameters are read from the criterion which evaluates the loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ultralytics.models.utils.loss import DETRLoss, RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou, box_iou
from ultralytics.utils.ops import xywh2xyxy

FORMULA = {"name": "gnr_v1_actual_vfl", "version": 1, "beta": 0.5,
           "ramp_start": 5, "ramp_end": 20, "vfl_source": "criterion.vfl.alpha/gamma",
           "expected_vfl": {"alpha": 0.25, "gamma": 1.5}, "nc": 1}


def ramp(epoch):
    return min(1.0, max(0.0, (float(epoch) - 5.0) / 15.0))


def require(ok, message):
    if not ok:
        raise ValueError("GNR: " + message)


def negative_derivative(logits, alpha, gamma):
    """FP32 analytic reference; includes the sigmoid modulation derivative."""
    p = logits.sigmoid()
    return alpha * p.pow(gamma) * (p + gamma * (1 - p) * F.softplus(logits))


def ordered_weights(a, conflict, pair_iou, cosine, strength, beta, rate):
    """Inputs are in ascending original query order; stable sort resolves exact ties."""
    order = strength.argsort(descending=True, stable=True)
    redundancy = (pair_iou[order][:, order] * cosine[order][:, order] * a[order][None]).tril(-1).sum(1)
    weights = 1 - beta * rate * conflict[order] * redundancy / (1 + redundancy)
    inverse = order.argsort()
    return weights[inverse], redundancy[inverse], order


@torch.no_grad()
def negative_weights(boxes, logits, features, gt_boxes, gt_groups, matches, quality, *, alpha, gamma, beta, epoch):
    """boxes/GT: normalized cxcywh. All indices refer to ordinary queries only.

    matches contain flattened batch GT indices from the ORIGINAL final Hungarian
    assignment. quality is its original [B,N] VFL target, never recomputed here.
    Returned diagnostics are caller-owned detached tensors, never module caches.
    """
    require(logits.ndim == 3 and logits.shape[-1] == 1, "only nc=1 is supported")
    batch, nq = logits.shape[:2]
    require(boxes.shape == (batch, nq, 4), "final box/logit shape mismatch")
    require(features is not None and features.ndim == 3 and features.shape[:2] == (batch, nq), "missing/misaligned final Linear input")
    require(quality.shape == (batch, nq), "original VFL quality shape mismatch")
    require(len(gt_groups) == len(matches) == batch and sum(gt_groups) == len(gt_boxes), "GT batch offsets mismatch")
    require(0 <= beta <= .5, "beta outside v1 range")
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        b, z, h, g, q = [t.detach().float() for t in (boxes, logits[..., 0], features, gt_boxes, quality)]
        require(all(torch.isfinite(t).all() for t in (b, z, h, g, q)), "nonfinite box/logit/feature/quality input")
        require(bool((b[..., 2:] >= 0).all() and (g[..., 2:] >= 0).all()), "negative box width/height")
        require(bool(((q >= 0) & (q <= 1)).all()), "VFL quality out of range")
        v = torch.cat((h, torch.ones_like(h[..., :1])), -1)
        norms = v.norm(dim=-1)
        require(bool(torch.isfinite(norms).all() and (norms > 0).all()), "nonfinite feature/bias norm")
        unit = v / norms[..., None]
        strength = negative_derivative(z, alpha, gamma) * norms
        require(bool(torch.isfinite(strength).all()), "nonfinite analytic negative strength")
        weights = torch.ones_like(z)
        negative = torch.ones_like(z, dtype=torch.bool)
        ownership = torch.full_like(z, -1, dtype=torch.long)
        affinity, conflict, redundancy, pos_cos, underfit = [torch.zeros_like(z) for _ in range(5)]
        first = torch.zeros_like(negative)
        group_sizes, missing_matches, offset = [], 0, 0
        for image, ngt in enumerate(gt_groups):
            src, dst = [t.to(device=z.device, dtype=torch.long) for t in matches[image]]
            require(src.numel() == dst.numel() and src.unique().numel() == src.numel(), f"image {image}: duplicate/misaligned positive queries")
            require(bool(((src >= 0) & (src < nq)).all()), f"image {image}: ordinary query index out of bounds")
            require(bool(((dst >= offset) & (dst < offset + ngt)).all()) and dst.unique().numel() == dst.numel(), f"image {image}: flattened GT index out of bounds/duplicated")
            negative[image, src] = False
            if ngt == 0:
                continue
            positives = torch.full((ngt,), -1, device=z.device, dtype=torch.long)
            positives[dst - offset] = src
            ids = torch.where(negative[image])[0]  # ascending ordinary query ID
            if ids.numel():
                overlaps = box_iou(xywh2xyxy(b[image, ids]), xywh2xyxy(g[offset:offset + ngt]))
                largest, owner = overlaps.max(1)
                second = overlaps.topk(2, dim=1).values[:, 1] if ngt > 1 else torch.zeros_like(largest)
                margin = largest - second  # exact FP32 equality excludes ties; no epsilon gate
                ownership[image, ids] = torch.where(margin > 0, owner, -1)
                affinity[image, ids] = margin
                for j in range(ngt):
                    members = ids[(owner == j) & (margin > 0)]
                    if not members.numel():
                        continue
                    group_sizes.append(members.numel())
                    pos = positives[j]
                    if bool(pos < 0):
                        missing_matches += members.numel()
                        continue
                    u = (q[image, pos] - z[image, pos].sigmoid()).relu() / q[image, pos].clamp_min(1e-8)
                    directions = (unit[image, members] @ unit[image, pos]).clamp(-1, 1)
                    positive_cosine = directions.clamp_min(0)
                    c = affinity[image, members] * u * positive_cosine
                    cos = (unit[image, members] @ unit[image, members].T).clamp(-1, 1).clamp_min(0)
                    pair = box_iou(xywh2xyxy(b[image, members]), xywh2xyxy(b[image, members]))
                    require(bool(torch.isfinite(pair).all() and torch.isfinite(cos).all()), f"image {image}, GT {j}: nonfinite pair relation")
                    w, r, order = ordered_weights(affinity[image, members], c, pair, cos, strength[image, members], beta, ramp(epoch))
                    weights[image, members], redundancy[image, members] = w, r
                    conflict[image, members], pos_cos[image, members], underfit[image, members] = c, directions, u
                    first[image, members[order[0]]] = True
            offset += ngt
        require(bool(torch.isfinite(weights).all() and ((weights >= .5) & (weights <= 1)).all()), "nonfinite/out-of-range weights")
        require(bool((weights[first] == 1).all() and (weights[~negative] == 1).all()), "strongest/positive protection violated")
        return weights, dict(negative=negative, ownership=ownership, a=affinity, c=conflict, R=redundancy,
                             w=weights, s=strength, positive_cosine=pos_cos, underfit=underfit, first=first,
                             group_sizes=group_sizes, unmatched_gt_negative_queries=missing_matches,
                             matches=matches, quality=q, alpha=alpha, gamma=gamma, epoch=epoch, ramp=ramp(epoch))


def summarize(details):
    """Small detached JSON diagnostics, outside the automatically summed loss dict."""
    neg, grouped = details["negative"], details["negative"] & (details["ownership"] >= 0)
    w, s = details["w"], details["s"]
    all_strength = float(s[neg].sum())
    grouped_strength = float(s[grouped].sum())
    removed = float(((1 - w) * s)[neg].sum())
    quantiles = {}
    for name in ("a", "R", "w", "positive_cosine", "underfit"):
        values = details[name][grouped]
        quantiles[name] = values.quantile(values.new_tensor([0, .25, .5, .75, 1])).cpu().tolist() if values.numel() else None
    return dict(epoch=details["epoch"], ramp=details["ramp"], alpha=details["alpha"], gamma=details["gamma"],
                negatives=int(neg.sum()), grouped=int(grouped.sum()), changed=int((w[neg] < 1).sum()),
                group_sizes=details["group_sizes"], quantiles=quantiles, strongest_retained=bool((w[details["first"]] == 1).all()),
                underfit_grouped=int((details["underfit"][grouped] > 0).sum()),
                positive_cosine_grouped=int((details["positive_cosine"][grouped] > 0).sum()),
                all_negative_strength=all_strength, grouped_negative_strength=grouped_strength, removed_strength=removed,
                removed_negative_strength_fraction=removed / all_strength if all_strength else None,
                removed_grouped_strength_fraction=removed / grouped_strength if grouped_strength else None,
                unmatched_gt_negative_queries=details["unmatched_gt_negative_queries"],
                raw_negative_loss=details.get("raw_negative_loss"), weighted_negative_loss=details.get("weighted_negative_loss"),
                class_difference=details.get("class_difference"))


class GNRDetectionLoss(RTDETRDetectionLoss):
    """Original criterion off-path; on-path matches once, uses its exact IoU targets."""

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None, *, features=None, epoch=0, beta=.5, return_details=False):
        if beta == 0 or ramp(epoch) == 0:
            losses = super().forward(preds, batch, dn_bboxes, dn_scores, dn_meta)
            return (losses, None) if return_details else losses
        require(self.nc == 1 and self.fl is not None and self.vfl is not None, "expected nc=1 VFL criterion")
        require((self.vfl.alpha, self.vfl.gamma) == (.25, 1.5), f"actual criterion.vfl parameters changed: {self.vfl.alpha}, {self.vfl.gamma}")
        boxes, scores = preds
        self.device = boxes.device
        final_b, final_z = boxes[-1], scores[-1]
        require(features is not None, "active GNR requires explicit final Linear inputs")
        gt_cls, gt_boxes, groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        matches = self.matcher(final_b, final_z, gt_boxes, gt_cls, groups)
        idx, gt_idx = self._get_index(matches)
        selected_b, selected_g = final_b[idx], gt_boxes[gt_idx]
        bs, nq = final_z.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]
        quality = torch.zeros([bs, nq], device=scores.device)
        if len(selected_g):
            quality[idx] = bbox_iou(selected_b.detach(), selected_g, xywh=True).squeeze(-1)
        weights, details = negative_weights(final_b, final_z, features, gt_boxes, groups, matches, quality,
                                           alpha=self.vfl.alpha, gamma=self.vfl.gamma, beta=beta, epoch=epoch)
        if not len(selected_g):
            # Mother intentionally uses FocalLoss, not VFL, when the batch has no matches.
            classification = self._get_loss_class(final_z, targets, quality, 0)
            details.update(raw_negative_loss=None, weighted_negative_loss=None, class_difference=0.)
            # Empty-batch diagnostic denominator follows the actual mother FL
            # fallback (negative alpha factor = 1-alpha); weights are all one.
            with torch.no_grad(), torch.autocast(device_type=scores.device.type, enabled=False):
                v = torch.cat((features.detach().float(), torch.ones_like(features[..., :1], dtype=torch.float32)), -1)
                details["s"] = negative_derivative(final_z.detach().float()[..., 0], 1 - float(self.fl.alpha), self.fl.gamma) * v.norm(dim=-1)
            details["loss_branch"] = "mother_empty_batch_focal_fallback"
        else:
            one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
            one_hot.scatter_(2, targets.unsqueeze(-1), 1)
            one_hot = one_hot[..., :-1]
            gt_scores = quality.view(bs, nq, 1) * one_hot
            # Exact mother dtype/autocast/order for sigmoid modulation and BCE.
            modulation = self.vfl.alpha * final_z.sigmoid().pow(self.vfl.gamma) * (1 - one_hot) + gt_scores * one_hot
            with torch.autocast(device_type=scores.device.type, enabled=False):
                terms = F.binary_cross_entropy_with_logits(final_z.float(), gt_scores.float(), reduction="none") * modulation
                weighted = terms * weights[..., None]
                original = terms.mean(1).sum() / (max(len(selected_g), 1) / nq) * self.loss_gain["class"]
                value = weighted.mean(1).sum() / (max(len(selected_g), 1) / nq) * self.loss_gain["class"]
            classification = {"loss_class": value.squeeze()}
            negative = details["negative"]
            details.update(raw_negative_loss=float(terms.detach()[..., 0][negative].sum()),
                           weighted_negative_loss=float(weighted.detach()[..., 0][negative].sum()),
                           class_difference=float((value - original).detach()))
        losses = {**classification, **self._get_loss_bbox(selected_b, selected_g)}
        if self.aux_loss:
            losses.update(self._get_loss_aux(boxes[:-1], scores[:-1], gt_boxes, gt_cls, groups))
        if dn_meta is not None:
            dn_matches = self.get_dn_match_indices(dn_meta["dn_pos_idx"], dn_meta["dn_num_group"], groups)
            losses.update(DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=dn_matches))
        else:
            losses.update({f"{k}_dn": torch.tensor(0., device=self.device) for k in list(losses)})
        return (losses, details) if return_details else losses
