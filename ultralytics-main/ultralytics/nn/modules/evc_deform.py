"""Evidence verification of the existing single-pass deformable samples (EVC v1)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .head import RTDETRDecoder
from .transformer import MSDeformAttn


def sample_deformable_values(value, value_shapes, sampling_locations):
    """Original grid geometry, once per level; return [B,Q,H,S,K,D], without weighting.

    Kept separate so the original utils.multi_scale_deformable_attn_pytorch API and
    all existing models retain their behavior. No extra key sampling or validity mask.
    """
    b, _, heads, dim = value.shape
    q, points = sampling_locations.shape[1], sampling_locations.shape[4]
    values = value.split([h * w for h, w in value_shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(value_shapes):
        feature = values[level].flatten(2).transpose(1, 2).reshape(b * heads, dim, h, w)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    return torch.stack(sampled, dim=-2).view(b, heads, dim, q, len(value_shapes), points).permute(0, 3, 1, 4, 5, 2)


class EVCMSDeformAttn(MSDeformAttn):
    """Per-head 32->8 evidence, separately correcting point and scale logits.

    query already includes native positional embedding. Evidence is a learned
    consistency score, not a calibrated probability. Original 32-D values are used
    in aggregation. No persistent activations or per-batch diagnostic statistics.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4, score_dim=8):
        super().__init__(d_model, n_levels, n_heads, n_points)
        self.evc_q = nn.Parameter(torch.empty(n_heads, score_dim, d_model // n_heads))
        self.evc_v = nn.Parameter(torch.empty(n_heads, score_dim, d_model // n_heads))
        self.evc_a_p = nn.Parameter(torch.zeros(n_heads))
        self.evc_a_s = nn.Parameter(torch.zeros(n_heads))
        for projection in (self.evc_q, self.evc_v):
            for head in projection:
                nn.init.xavier_uniform_(head)

    def evidence_weights(self, query, sampled_value, logits):
        """FP32 scoring/normalization; exposed for bounded tests, never caches tensors."""
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float().reshape(*query.shape[:2], self.n_heads, self.d_model // self.n_heads)
            q_score = F.normalize(torch.einsum("bqhd,hed->bqhe", q, self.evc_q.float()), dim=-1, eps=1e-6)
            v_score = F.normalize(torch.einsum("bqhskd,hed->bqhske", sampled_value.float(), self.evc_v.float()),
                                  dim=-1, eps=1e-6)
            evidence = (q_score[:, :, :, None, None, :] * v_score).sum(-1)
            z = logits.float()
            p0 = z.softmax(-1)
            beta_p = self.evc_a_p.float().tanh()[None, None, :, None, None]
            beta_s = self.evc_a_s.float().tanh()[None, None, :, None]
            p = (z + beta_p * (evidence - evidence.mean(-1, keepdim=True))).softmax(-1)
            t = (p0 * evidence).sum(-1)  # deliberately original p0, not corrected p
            pi = (torch.logsumexp(z, dim=-1) + beta_s * t).softmax(-1)
            return pi[..., None] * p, evidence

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        b, q = query.shape[:2]
        length = value.shape[1]
        assert sum(h * w for h, w in value_shapes) == length
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], 0.)
        value = value.view(b, length, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(b, q, self.n_heads, self.n_levels, self.n_points, 2)
        logits = self.attention_weights(query).view(b, q, self.n_heads, self.n_levels, self.n_points)
        if refer_bbox.shape[-1] == 2:
            normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            locations = refer_bbox[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        elif refer_bbox.shape[-1] == 4:
            locations = refer_bbox[:, :, None, :, None, :2] + offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * .5
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {refer_bbox.shape[-1]}.")
        sampled = sample_deformable_values(value, value_shapes, locations)
        weights, _ = self.evidence_weights(query, sampled, logits)
        # Same reduction layout as the original helper, including S*K reduction order.
        samples = sampled.permute(0, 2, 5, 1, 3, 4).reshape(b * self.n_heads, -1, q, self.n_levels * self.n_points)
        weights = weights.to(sampled.dtype).transpose(1, 2).reshape(b * self.n_heads, 1, q, -1)
        output = (samples * weights).sum(-1).view(b, self.d_model, q).transpose(1, 2).contiguous()
        return self.output_proj(output)


class RTDETRDecoderEVC(RTDETRDecoder):
    """Native C2 head, replacing only the last cross_attn after all public initialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.acr or self.decoder.eval_idx != self.decoder.num_layers - 1:
            raise ValueError("EVC v1 requires native self-attention and final-layer evaluation")
        old = self.decoder.layers[-1].cross_attn
        # Preserve outer RNG sequence and all existing public module values.
        with torch.random.fork_rng(devices=[]):
            new = EVCMSDeformAttn(old.d_model, old.n_levels, old.n_heads, old.n_points)
        new.load_state_dict({**new.state_dict(), **old.state_dict()}, strict=True)
        self.decoder.layers[-1].cross_attn = new
