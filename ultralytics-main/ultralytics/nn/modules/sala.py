# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Short-Axis Level Allocation in complete multi-scale deformable cross-attention."""

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import MSDeformAttn
from .utils import multi_scale_deformable_attn_pytorch

__all__ = ("SALAMSDeformAttn",)


class SALAMSDeformAttn(MSDeformAttn):
    """Query/box-grid conditioned level logits with the original deformable sampling rule.

    Common projections retain MSDeformAttn state keys. Only the geometry descriptor
    detaches boxes; the sampling path retains its original autograd behavior.
    Residuals, normalization, FFN and query position encoding belong to the caller.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__(d_model, n_levels, n_heads, n_points)
        # Additional initialization must not shift any subsequent common parameters.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.sala_query_proj = nn.Linear(d_model, 32)
            self.sala_geometry_proj = nn.Linear(2, 32)
            self.sala_level_embedding = nn.Parameter(torch.empty(n_levels, 32))
            nn.init.normal_(self.sala_level_embedding, std=0.02)
            self.sala_level_head = nn.Linear(32, n_heads)
            nn.init.zeros_(self.sala_level_head.weight)
            nn.init.zeros_(self.sala_level_head.bias)
        # Opt-in, detached small summaries, neither persistent state nor API outputs.
        self.sala_stats_interval = 0
        self._sala_calls = 0
        self.sala_last_stats = None

    def geometry_descriptor(self, refer_bbox, value_shapes):
        """Return log2(short,long) grid sizes [B,Q,L,2], calculated in FP32."""
        if refer_bbox.ndim != 4 or refer_bbox.shape[-1] != 4:
            raise ValueError("SALA requires 4D boxes [B,Q,1 or L,4]; 2D reference points have no box size.")
        if refer_bbox.shape[2] not in (1, self.n_levels) or len(value_shapes) != self.n_levels:
            raise ValueError("Reference levels/value_shapes must match SALA n_levels.")
        wh = torch.as_tensor(value_shapes, dtype=torch.float32, device=refer_bbox.device).flip(-1)
        grid = refer_bbox.detach().float()[..., 2:] * wh[None, None]
        sides = torch.stack((grid.amin(-1), grid.amax(-1)), -1)
        return sides.clamp(0.25, 128.0).log2()

    def level_bias(self, query, refer_bbox, value_shapes):
        """Compute bounded [B,Q,L,H] biases; autocast follows the projections."""
        geometry = self.geometry_descriptor(refer_bbox, value_shapes)
        semantic = self.sala_query_proj(query)
        geometry = self.sala_geometry_proj(geometry.to(self.sala_geometry_proj.weight.dtype))
        hidden = F.silu(semantic.unsqueeze(2) + geometry + self.sala_level_embedding.to(semantic.dtype))
        return 0.5 * self.sala_level_head(hidden).tanh()

    @torch.no_grad()
    def _record_stats(self, delta, weights):
        values = delta.detach().float().flatten()
        self.sala_last_stats = {
            "call": self._sala_calls,
            "queries": weights.shape[1],
            "level_mass_by_head": weights.detach().float().sum(-1).mean((0, 1)).cpu().tolist(),
            "delta_quantiles_0_25_50_75_100": torch.quantile(
                values, values.new_tensor([0, .25, .5, .75, 1])
            ).cpu().tolist(),
            "delta_abs_mean": values.abs().mean().item(),
            "delta_abs_max": values.abs().max().item(),
            "delta_near_bound_fraction_abs_ge_0_49": (values.abs() >= .49).float().mean().item(),
        }

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """Project, allocate levels, sample and aggregate to an output of shape [B,Q,C]."""
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        logits = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        delta = self.level_bias(query, refer_bbox, value_shapes).permute(0, 1, 3, 2).unsqueeze(-1).to(logits.dtype)
        attention_weights = F.softmax((logits + delta).flatten(-2), -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )
        # This is exactly the baseline 4D sampling expression, using the ORIGINAL boxes.
        add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
        sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        if self.sala_stats_interval > 0:
            self._sala_calls += 1
            if (self._sala_calls - 1) % self.sala_stats_interval == 0:
                self._record_stats(delta, attention_weights)
        return self.output_proj(output)
