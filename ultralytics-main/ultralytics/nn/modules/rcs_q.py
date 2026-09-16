# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RCS-Q v1: region-centered value evidence with a zero-value attention slot.

The region mean is a shared-component reference, not a background label. Attention
weights are neither segmentation masks nor calibrated confidence. This branch only
refines already selected content queries; it does not recover omitted top-k objects.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .cbr import RTDETRDecoderCBR
from .head import RTDETRDecoder


def build_query_valid(embed, batch, dn_meta, num_queries=300):
    """Identify padding from the existing CDN scatter layout, never tensor contents.

    ``get_cdn_group`` scatters each image's ``range(gt_groups[b])`` into every
    length-M block, repeated 2*G times. Positive/negative identities, labels and
    clean boxes are not features of RCS-Q. Preflight independently checks these
    occupied slots against the actual generator's scatter indices.
    """
    if embed.ndim != 3:
        raise ValueError("RCS-Q expects query embeddings [B,Q,256].")
    bs, count = embed.shape[:2]
    if dn_meta is None:
        if count != num_queries:
            raise ValueError(f"Without DN, expected {num_queries} queries, got {count}.")
        return torch.ones((bs, count), dtype=torch.bool, device=embed.device)
    split = dn_meta.get("dn_num_split", ())
    groups = dn_meta.get("dn_num_group")
    if len(split) != 2 or any(not isinstance(n, Integral) or isinstance(n, bool) for n in split):
        raise ValueError("Malformed dn_num_split.")
    dn_count, normal_count = split
    if not isinstance(groups, Integral) or isinstance(groups, bool) or groups <= 0:
        raise ValueError("DN group count must be a positive integer.")
    if dn_count <= 0 or dn_count % (2 * groups) or normal_count != num_queries or count != sum(split):
        raise ValueError("DN length/group/query count does not match the original generator.")
    maximum = dn_count // (2 * groups)
    counts = None if batch is None else batch.get("gt_groups")
    if counts is None or len(counts) != bs:
        raise ValueError("DN requires one gt_groups count per image.")
    if any(not isinstance(n, Integral) or isinstance(n, bool) or not 0 <= n <= maximum for n in counts):
        raise ValueError("DN gt_groups counts are outside the scatter block range.")
    if max(counts) != maximum:
        raise ValueError("DN scatter block width differs from max(gt_groups).")
    position = torch.arange(dn_count, device=embed.device) % maximum
    dn_valid = position[None] < torch.tensor(counts, device=embed.device)[:, None]
    return torch.cat((dn_valid, torch.ones((bs, num_queries), dtype=torch.bool, device=embed.device)), dim=1)


class RegionCenteredSupportQuery(nn.Module):
    """Fixed 256 -> 64, two-head, 5x5 RCS-Q; exactly 58,370 trainable parameters."""

    def __init__(self, p3_channels=256, query_dim=256):
        super().__init__()
        if (p3_channels, query_dim) != (256, 256):
            raise ValueError("RCS-Q v1 requires P3 channels=256 and query_dim=256.")
        self.heads, self.head_dim, self.grid_size = 2, 32, 5
        # New initialization is local and identical for both variants. Neither the
        # public CPU generator nor any CUDA generator is consumed by construction.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.p3_proj = nn.Conv2d(256, 64, 1, bias=False)
            self.query_norm = nn.LayerNorm(256, elementwise_affine=True, eps=1e-5)
            self.q_proj = nn.Linear(256, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.null_logit = nn.Linear(256, 2, bias=True)
            self.out_proj = nn.Linear(64, 256, bias=False)
            for layer in (self.p3_proj, self.q_proj, self.k_proj, self.v_proj):
                nn.init.xavier_uniform_(layer.weight)
            nn.init.ones_(self.query_norm.weight)
            nn.init.zeros_(self.query_norm.bias)
            nn.init.zeros_(self.null_logit.weight)
            nn.init.constant_(self.null_logit.bias, math.log(25))
            nn.init.zeros_(self.out_proj.weight)
        # Integer storage preserves exact FP32 construction of bin centers even
        # after model.half(); rows run y first and the last coordinate is (x,y).
        yy, xx = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing="ij")
        self.register_buffer("grid_offsets", torch.stack((xx, yy), dim=-1).reshape(25, 2), persistent=False)

    @staticmethod
    def _finite(name, value):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"RCS-Q non-finite {name}.")

    def forward(self, p3, query, refer_bbox, query_valid=None, return_diagnostics=False):
        """Add support after original query detach; references are cxcywh logits.

        Only geometry is detached. Functional FP32 casts keep all registered
        parameters differentiable under native AMP and already-half inference.
        Diagnostics are explicit return values and are never cached on the module.
        """
        if p3.ndim != 4 or p3.shape[1] != 256 or query.ndim != 3 or query.shape[-1] != 256:
            raise ValueError("RCS-Q requires P3 [B,256,H,W] and queries [B,Q,256].")
        bs, count = query.shape[:2]
        if p3.shape[0] != bs or refer_bbox.shape != (bs, count, 4):
            raise ValueError("RCS-Q P3/query/reference batch or query counts differ.")
        if p3.device != query.device or refer_bbox.device != query.device:
            raise ValueError("RCS-Q inputs must share a device.")
        if query_valid is None:
            query_valid = torch.ones((bs, count), dtype=torch.bool, device=query.device)
        if query_valid.shape != (bs, count) or query_valid.dtype != torch.bool or query_valid.device != query.device:
            raise ValueError("query_valid must be a boolean [B,Q] tensor on the query device.")
        self._finite("P3 input", p3)
        self._finite("query input", query)
        # Original anchor generation legitimately uses +/-inf. Sigmoid maps those
        # to boundary coordinates; actual NaNs must raise instead of being hidden.
        if torch.isnan(refer_bbox).any():
            raise FloatingPointError("RCS-Q NaN reference logits.")

        with torch.autocast(device_type=p3.device.type, enabled=False):
            boxes = refer_bbox.detach().float().sigmoid()
            box_valid = torch.isfinite(boxes).all(dim=-1) & (boxes[..., 2:] > 0).all(dim=-1)
            offsets = self.grid_offsets.float() / 5.0
            uv = boxes[..., None, :2] + boxes[..., None, 2:] * offsets
            valid = query_valid[..., None] & box_valid[..., None] & ((uv >= 0) & (uv <= 1)).all(dim=-1)
            grid = uv * 2.0 - 1.0  # do not clamp truly out-of-image locations
            projected = F.conv2d(p3.float(), self.p3_proj.weight.float())
            self._finite("P3 projection", projected)
            sampled = F.grid_sample(projected, grid, mode="bilinear", padding_mode="border", align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1)  # [B,Q,25,64]
            self._finite("sampled features", sampled)
            valid_count = valid.sum(dim=-1)
            region_mean = torch.where(valid[..., None], sampled, 0.0).sum(dim=-2)
            region_mean = region_mean / valid_count.clamp_min(1)[..., None]
            centered = torch.where(valid[..., None], sampled - region_mean[..., None, :], 0.0)
            qn = F.layer_norm(query.float(), (256,), self.query_norm.weight.float(), self.query_norm.bias.float(), 1e-5)
            q = F.linear(qn, self.q_proj.weight.float()).reshape(bs, count, 2, 32)
            k = F.linear(centered, self.k_proj.weight.float()).reshape(bs, count, 25, 2, 32)
            v = F.linear(centered, self.v_proj.weight.float()).reshape(bs, count, 25, 2, 32)
            real_logits = (q[:, :, None] * k).sum(dim=-1).permute(0, 1, 3, 2) / math.sqrt(32)
            empty_logits = F.linear(qn, self.null_logit.weight.float(), self.null_logit.bias.float())
            self._finite("centered values", v)
            self._finite("real attention logits", real_logits)
            self._finite("null attention logits", empty_logits)
            real_logits = real_logits.masked_fill(~valid[:, :, None], float("-inf"))
            weights = torch.cat((empty_logits[..., None], real_logits), dim=-1).softmax(dim=-1)
            # Slot 0 has constant zero value. Real-slot weights retain their mass
            # below one; renormalizing them would remove null-slot competition.
            support = (weights[..., 1:].permute(0, 1, 3, 2)[..., None] * v).sum(dim=2).reshape(bs, count, 64)
            delta = F.linear(support, self.out_proj.weight.float())
            self._finite("query correction", delta)
        result = query + delta.to(query.dtype)
        self._finite("output query", result)
        if return_diagnostics:
            return result, {
                "attention_weights": weights, "null_weights": weights[..., 0],
                "valid_points": valid, "valid_count": valid_count, "query_valid": query_valid,
                "delta": delta, "centered": centered, "sampled": sampled, "region_mean": region_mean,
                "boxes": boxes, "grid": grid, "query": query,
            }
        return result


class _RCSQDecoderMixin:
    """Small forward adapter: all encoder selection, detach and decoder math are inherited."""

    def __init__(self, nc=80, ch=(256, 256, 256), hd=256, nq=300, ndp=4, nh=8, ndl=3, **kwargs):
        super().__init__(nc, ch, hd, nq, ndp, nh, ndl, **kwargs)
        if len(ch) != 3 or ch[0] != 256 or hd != 256:
            raise ValueError("RCS-Q v1 requires three decoder inputs and P3/query channels=256.")
        # Construct only after the superclass's _reset_parameters has completed.
        self.rcsq = RegionCenteredSupportQuery(ch[0], hd)

    def forward(self, x, batch=None):
        return self._forward_rcsq(x, batch, False)

    def forward_with_diagnostics(self, x, batch=None):
        """Return (original train/eval/export output, diagnostics) without state changes."""
        return self._forward_rcsq(x, batch, True)

    def _forward_rcsq(self, x, batch, diagnostics):
        from ultralytics.models.utils.ops import get_cdn_group

        p3 = x[0]
        if len(x) != 3 or any(p3.shape[-2] < f.shape[-2] or p3.shape[-1] < f.shape[-1] for f in x[1:]):
            raise ValueError("RCS-Q expects decoder inputs ordered Neck P3, P4, P5.")
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch, self.nc, self.num_queries, self.denoising_class_embed.weight,
            self.num_denoising, self.label_noise_ratio, self.box_noise_scale, self.training,
        )
        # Original method selects top-k and performs the original normal detach.
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        valid = build_query_valid(embed, batch, dn_meta, self.num_queries)
        supported = self.rcsq(p3, embed, refer_bbox, valid, return_diagnostics=diagnostics)
        embed, details = supported if diagnostics else (supported, None)
        with_cbr = isinstance(self, RTDETRDecoderCBR)
        decoded = self.decoder(
            embed, refer_bbox, feats, shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask, return_final_query=with_cbr,
        )
        dec_bboxes, dec_scores = decoded[:2]
        if with_cbr:
            # Original CBR sees exactly the Neck P3, final decoder query and final
            # boxes it previously used; its 36-point refinement is not redefined.
            refined, cbr_details = self.cbr(p3, decoded[2], dec_bboxes[-1], return_diagnostics=True)
            dec_bboxes = torch.cat((dec_bboxes[:-1], refined.unsqueeze(0)), dim=0)
            if diagnostics:
                details = {**cbr_details, "rcsq": details}
        raw = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            result = raw
        else:
            y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), dim=-1)
            result = y if self.export else (y, raw)
        return (result, details) if diagnostics else result


class RTDETRDecoderRCSQ(_RCSQDecoderMixin, RTDETRDecoder):
    """Original RT-DETR head plus RCS-Q, preserving every public state path."""


class RTDETRDecoderCBRRCSQ(_RCSQDecoderMixin, RTDETRDecoderCBR):
    """Original CBR head plus RCS-Q; original cbr.* state paths stay unchanged."""
