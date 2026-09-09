"""FSA-Deform v1: finite support values for C2's middle cross-attention only."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .head import RTDETRDecoder
from .transformer import MSDeformAttn


def support_geometry(raw, refer_bbox, value_shapes, n_points):
    """Return per-query/head/level radii in cells, normalized displacements and signed eta."""
    wh = torch.as_tensor(value_shapes, device=raw.device, dtype=torch.float32).flip(-1)
    base = (refer_bbox.float()[:, :, None, :, 2:] * wh[None, None, None] / (2 * n_points)).clamp(.25, 1.5)
    radius = (base * 2 * raw.float()[..., :2].sigmoid().unsqueeze(3)).clamp(.125, 2.)
    signs = raw.new_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32)
    delta = radius.unsqueeze(-2) * signs / (math.sqrt(3) * wh[None, None, None, :, None])
    eta = .5 * raw.float()[..., 2].tanh()
    return radius, delta, eta


def multi_scale_deformable_attn_fsa_pytorch(value, value_shapes, locations, weights, delta, eta):
    """One grid_sample per level, vectorizing all queries, heads and K*(center+4) reads."""
    bs, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = locations.shape
    values = value.split([h * w for h, w in value_shapes], dim=1)
    sampled = []
    eta = eta.transpose(1, 2).reshape(bs * heads, 1, queries, 1)
    for level, (height, width) in enumerate(value_shapes):
        feature = values[level].flatten(2).transpose(1, 2).reshape(bs * heads, channels, height, width)
        center = locations[:, :, :, level].float()
        support = center.unsqueeze(-2) + delta[:, :, :, level].unsqueeze(-3)
        positions = torch.cat((center.unsqueeze(-2), support), dim=-2)
        grid = (2 * positions - 1).transpose(1, 2).reshape(bs * heads, queries, points * 5, 2)
        # FP32 sampling also supports true-half inference on versions whose grid_sample lacks half kernels.
        # Casting back at the helper boundary follows the surrounding projection/autocast dtype.
        with torch.autocast(device_type=value.device.type, enabled=False):
            reads = F.grid_sample(feature.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
            reads = reads.reshape(bs * heads, channels, queries, points, 5)
            v0, area = reads[..., 0], reads[..., 1:].mean(-1)
            sampled.append(v0 + eta * (area - v0))
    weights = weights.transpose(1, 2).reshape(bs * heads, 1, queries, levels * points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1).view(bs, heads * channels, queries)
    return output.transpose(1, 2).contiguous().to(value.dtype)


class FSADeformAttn(MSDeformAttn):
    """Preserve native public Linear keys; only support_fc1/2 are new state."""

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__(d_model, n_levels, n_heads, n_points)
        self.support_fc1 = nn.Linear(d_model, 16)
        self.support_fc2 = nn.Linear(16, n_heads * 3)
        nn.init.xavier_uniform_(self.support_fc1.weight)
        nn.init.zeros_(self.support_fc1.bias)
        nn.init.zeros_(self.support_fc2.weight)
        nn.init.zeros_(self.support_fc2.bias)

    def support_parameters(self, query):
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float()
            q = q / torch.sqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)
        h = F.silu(self.support_fc1(q.to(self.support_fc1.weight.dtype)))
        return self.support_fc2(h).reshape(*query.shape[:2], self.n_heads, 3)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        if refer_bbox.shape[-1] == 2:
            return super().forward(query, refer_bbox, value, value_shapes, value_mask)
        if refer_bbox.shape[-1] != 4:
            raise ValueError("Last dim of reference_points must be 2 or 4")
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(h * w for h, w in value_shapes) == len_v
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        add = offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * .5
        locations = refer_bbox[:, :, None, :, None, :2] + add
        _, delta, eta = support_geometry(self.support_parameters(query), refer_bbox, value_shapes, self.n_points)
        return self.output_proj(multi_scale_deformable_attn_fsa_pytorch(value, value_shapes, locations, weights, delta, eta))


class RTDETRDecoderFSA(RTDETRDecoder):
    """Clone the native decoder first, then replace ONLY layers[1].cross_attn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_decoder_layers != 3:
            raise ValueError("FSA-Deform v1 requires verified C2 num_decoder_layers == 3")
        old = self.decoder.layers[1].cross_attn
        if (old.d_model, old.n_heads, old.n_points) != (256, 8, 4):
            raise ValueError("FSA-Deform v1 requires C2 d_model=256, heads=8, points=4")
        # Do not advance the public model constructor RNG stream for this local extension.
        with torch.random.fork_rng(devices=[]):
            new = FSADeformAttn(old.d_model, old.n_levels, old.n_heads, old.n_points)
        new.load_state_dict({**new.state_dict(), **old.state_dict()}, strict=True)
        self.decoder.layers[1].cross_attn = new
