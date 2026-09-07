# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Axis-wise coverage relations for RT-DETR query self-attention."""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ("CoverageRelationSelfAttention",)


class CoverageRelationSelfAttention(nn.MultiheadAttention):
    """Native MHA projections/dropout with a bounded, directed regular-query relation bias.

    ``refer_bbox`` is batch-first normalized cxcywh, AFTER sigmoid. Only this
    geometric branch detaches it. DN queries must be a prefix, as in get_cdn_group.
    Public MHA state keys and construction order are deliberately preserved.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(embed_dim, num_heads, dropout=dropout, **kwargs)
        # No additional random draws escape into the construction of common layers.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            factory = {"device": self.in_proj_weight.device, "dtype": self.in_proj_weight.dtype}
            self.acr_geometry_proj = nn.Linear(10, 32, bias=True, **factory)
            self.acr_head = nn.Linear(32, num_heads, bias=True, **factory)
            nn.init.xavier_uniform_(self.acr_geometry_proj.weight)
            nn.init.zeros_(self.acr_geometry_proj.bias)
            nn.init.zeros_(self.acr_head.weight)
            nn.init.zeros_(self.acr_head.bias)
        self.acr_stats_interval = 0
        self.acr_calls = 0
        self.acr_last_stats = None

    @staticmethod
    def geometry(refer_bbox):
        """Return [B,Q,Q,10] in the fixed ACR order, computed entirely in FP32."""
        with torch.autocast(device_type=refer_bbox.device.type, enabled=False):
            boxes = refer_bbox.detach().float()
            center, size = boxes[..., :2], boxes[..., 2:].clamp_min(1e-4)
            lo, hi = center - size / 2, center + size / 2
            si, sj = size[:, :, None], size[:, None, :]
            delta = center[:, None, :] - center[:, :, None]
            left = torch.maximum(lo[:, :, None], lo[:, None, :])
            right = torch.minimum(hi[:, :, None], hi[:, None, :])
            overlap, gap = (right - left).clamp_min(0), (left - right).clamp_min(0)
            return torch.cat((
                torch.asinh(delta / (0.5 * (si + sj))).clamp(-4, 4),
                torch.log(si / sj).clamp(-4, 4),
                torch.log1p(gap / torch.minimum(si, sj)).clamp(0, 4),
                (overlap / si).clamp(0, 1), (overlap / sj).clamp(0, 1),
            ), dim=-1)

    def relation_bias(self, boxes):
        """Compute hidden geometry only for regular pairs; support AMP and actual half weights."""
        geometry = self.geometry(boxes).to(dtype=self.acr_geometry_proj.weight.dtype)
        bias = 0.5 * torch.tanh(self.acr_head(F.silu(self.acr_geometry_proj(geometry))))
        bias = bias.permute(0, 3, 1, 2)
        diagonal = torch.eye(boxes.shape[1], device=boxes.device, dtype=torch.bool)
        return bias.masked_fill(diagonal[None, None], 0)

    @staticmethod
    def additive_mask(mask, batch, heads, count, device, dtype):
        """Canonicalize 2D, batch, native batch*head, or explicit batch/head layouts."""
        if mask is None:
            return None
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise TypeError("ACR attention masks must be bool or floating point")
        mask = mask.to(device=device)
        if mask.dtype == torch.bool:
            mask = torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(mask, float("-inf"))
        else:
            mask = mask.to(dtype=dtype)
        if mask.shape == (count, count):
            mask = mask[None, None]
        elif mask.shape == (batch * heads, count, count):
            mask = mask.reshape(batch, heads, count, count)
        elif mask.shape == (batch, count, count):
            mask = mask[:, None]
        elif mask.ndim != 4 or mask.shape[-2:] != (count, count):
            raise ValueError(f"Unsupported ACR mask shape: {tuple(mask.shape)}")
        if mask.shape[0] not in (1, batch) or mask.shape[1] not in (1, heads):
            raise ValueError("Mask batch/head dimensions do not match MHA")
        return mask.expand(batch, heads, count, count)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False,
                refer_bbox=None, num_queries=None, dn_meta=None):
        if refer_bbox is None or num_queries is None:
            raise ValueError("ACR requires sigmoid reference boxes and the real num_queries")
        if query.ndim != 3 or self.add_zero_attn or self.bias_k is not None:
            raise ValueError("ACR supports batched decoder self-attention without appended MHA tokens")
        batch, count = (query.shape[0], query.shape[1]) if self.batch_first else (query.shape[1], query.shape[0])
        if refer_bbox.shape != (batch, count, 4) or key.shape != query.shape or value.shape != query.shape:
            raise ValueError("ACR query/reference layout mismatch")
        dn = count - num_queries
        if dn < 0 or num_queries <= 0:
            raise ValueError("Invalid regular query count")
        if dn_meta is None:
            if dn:
                raise ValueError("DN prefix requires dn_meta")
        elif list(dn_meta["dn_num_split"]) != [dn, num_queries]:
            raise ValueError("dn_num_split disagrees with DN prefix / regular suffix")
        bias = self.relation_bias(refer_bbox[:, dn:])
        full = F.pad(bias, (dn, 0, dn, 0)).to(dtype=query.dtype)
        original = self.additive_mask(attn_mask, batch, self.num_heads, count, query.device, query.dtype)
        if dn:
            if original is None or not torch.isneginf(original[:, :, dn:, :dn]).all():
                raise ValueError("Original regular-to-DN prohibitions must remain present")
        merged = full if original is None else full + original
        if is_causal:
            causal = torch.ones(count, count, dtype=torch.bool, device=query.device).triu(1)
            merged = merged.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, count):
                raise ValueError("Invalid query key_padding_mask")
            if key_padding_mask.dtype == torch.bool:
                padding = torch.zeros_like(key_padding_mask, dtype=query.dtype).masked_fill(key_padding_mask, float("-inf"))
            elif key_padding_mask.is_floating_point():
                padding = key_padding_mask.to(dtype=query.dtype)
            else:
                raise TypeError("key_padding_mask must be bool or floating point")
            merged = merged + padding.to(query.device)[:, None, None, :]
        # Flatten B then H, exactly as native MHA. No zero-head shortcut: gradients always flow.
        result = super().forward(query, key, value, need_weights=need_weights,
                                 attn_mask=merged.reshape(batch * self.num_heads, count, count),
                                 average_attn_weights=average_attn_weights, is_causal=False)
        self.acr_calls += 1
        if self.acr_stats_interval and self.acr_calls % self.acr_stats_interval == 0:
            self._record_stats(query, key, bias, merged, dn)
        return result

    @torch.no_grad()
    def _record_stats(self, query, key, bias, mask, dn):
        """Low-frequency scalar diagnostics; entropy is from pre-dropout softmax."""
        q, k = (query, key) if self.batch_first else (query.transpose(0, 1), key.transpose(0, 1))
        batch, count, channels = q.shape
        wq, wk, _ = self.in_proj_weight.chunk(3)
        bq, bk = (self.in_proj_bias.chunk(3)[:2] if self.in_proj_bias is not None else (None, None))
        q = F.linear(q, wq, bq).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(k, wk, bk).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        # Diagnostic softmax in FP32; never feeds back into model attention.
        logits = (q.float() * self.head_dim ** -0.5) @ k.float().transpose(-2, -1) + mask.float()
        p = logits.softmax(-1)[:, :, dn:]
        allowed_rows = torch.isfinite(logits[:, :, dn:]).any(-1)
        entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
        n = bias.shape[-1]
        off = ~torch.eye(n, dtype=torch.bool, device=bias.device)
        values = bias.detach().float()[..., off].flatten()
        self.acr_last_stats = {"calls": self.acr_calls, "regular_queries": n, "dn_queries": dn,
            "quantile_levels": [0, .01, .1, .5, .9, .99, 1],
            "bias_quantiles": torch.quantile(values, values.new_tensor([0, .01, .1, .5, .9, .99, 1])).cpu().tolist() if values.numel() else [],
            "abs_bias_gt_045": float((values.abs() > .45).float().mean()) if values.numel() else 0.,
            "allowed_regular_row_entropy": float(entropy[allowed_rows].mean()),
            "entropy_definition": "FP32 QK softmax before attention dropout, allowed keys, regular rows"}
