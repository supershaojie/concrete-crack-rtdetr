# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""C18: unchanged global AIFI plus per-query, single-scale local deformable aggregation."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import AIFI

__all__ = ("GSDRAIFIV3", "QueryLocalDeformableRelationV3")


def _fp32_context(tensor):
    # PyTorch 2.1 CPU must not construct a CPU FP16 autocast context, even disabled.
    return torch.autocast(device_type="cuda", enabled=False) if tensor.is_cuda else nullcontext()


class QueryLocalDeformableRelationV3(nn.Module):
    """Aggregate four samples inside each query's valid radius-2 pixel-center window.

    Grid layout is [batch, H*W query, head, point, xy]. Only the point axis is
    normalized. Five bias-free projections add 88,064 parameters at C=256.
    The fixed C18 budget deliberately exposes no structural search options.
    """

    def __init__(self, c1: int, aux_dim: int = 128, aux_heads: int = 4,
                 n_points: int = 4, radius_cells: float = 2.0):
        super().__init__()
        for name, value in (("c1", c1), ("aux_dim", aux_dim), ("aux_heads", aux_heads), ("n_points", n_points)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if aux_dim % aux_heads:
            raise ValueError("aux_dim must be divisible by aux_heads")
        if n_points != 4 or isinstance(radius_cells, bool) or radius_cells != 2.0:
            raise ValueError("C18 fixes n_points=4 and radius_cells=2.0")
        self.c1, self.aux_dim, self.aux_heads = c1, aux_dim, aux_heads
        self.n_points, self.radius_cells = n_points, float(radius_cells)
        self.head_dim = aux_dim // aux_heads
        self.input_proj = nn.Conv2d(c1, aux_dim, 1, bias=False)
        self.token_norm = nn.LayerNorm(aux_dim, eps=1e-5, elementwise_affine=False)
        self.value_proj = nn.Conv2d(aux_dim, aux_dim, 1, bias=False)
        self.offset_proj = nn.Conv2d(aux_dim, aux_heads * n_points * 2, 1, bias=False)
        self.weight_proj = nn.Conv2d(aux_dim, aux_heads * n_points, 1, bias=False)
        self.output_proj = nn.Conv2d(aux_dim, c1, 1, bias=False)
        # Persist exactly representable anchors, never half-quantized atanh logits.
        self.register_buffer("anchors", torch.tensor([[-0.5, -0.5], [0.5, -0.5],
                                                     [0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier input/value, zero offsets/weights/output; no new trainable biases."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.weight_proj.weight)
        nn.init.zeros_(self.output_proj.weight)

    def sampling_geometry(self, raw, height, width):
        """Return query pixel centers, sampled pixels and align_corners=False xy grid."""
        if height < 1 or width < 1 or tuple(raw.shape[1:]) != (height * width, self.aux_heads, 4, 2):
            raise ValueError("raw must be [B, H*W, heads, 4, xy] with positive H/W")
        with _fp32_context(raw):
            yy, xx = torch.meshgrid(torch.arange(height, device=raw.device, dtype=torch.float32),
                                    torch.arange(width, device=raw.device, dtype=torch.float32), indexing="ij")
            query = torch.stack((xx, yy), -1).reshape(-1, 2)
            size = query.new_tensor((width, height))
            # These fixed window bounds depend only on query/shape/radius, never on learned locations.
            lo = torch.maximum(query - self.radius_cells, torch.zeros_like(query))
            hi = torch.minimum(query + self.radius_cells, size - 1)
            center, extent = (lo + hi) / 2, (hi - lo) / 2
            u = torch.tanh(raw.float() + torch.atanh(self.anchors.float()))
            pixels = center[None, :, None, None] + extent[None, :, None, None] * u
            grid = 2 * (pixels + 0.5) / size - 1
        return query, pixels, grid

    def aggregate(self, value, grid, weights):
        """Sample each contiguous head independently, then sum only its four points (FP32)."""
        batch, _, height, width = value.shape
        with _fp32_context(value):
            values = value.float().reshape(batch * self.aux_heads, self.head_dim, height, width)
            grids = grid.float().transpose(1, 2).reshape(batch * self.aux_heads, height * width, 4, 2)
            sampled = F.grid_sample(values, grids, mode="bilinear", padding_mode="zeros", align_corners=False)
            point_weights = weights.float().transpose(1, 2).reshape(batch * self.aux_heads, 1, height * width, 4)
            related = (sampled * point_weights).sum(-1).reshape(batch, self.aux_dim, height, width)
        return related

    def relation_features(self, x):
        """Compute local features and ephemeral diagnostics; never cache activations on the module."""
        if x.ndim != 4 or x.shape[1] != self.c1 or min(x.shape[-2:]) < 1:
            raise ValueError(f"x must be [B, {self.c1}, H, W] with positive H/W")
        batch, _, height, width = x.shape
        projected = self.input_proj(x)
        with _fp32_context(projected):
            z = self.token_norm(projected.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # Explicit weight promotion supports both outer CUDA AMP and model.half(), with gradients intact.
            value = F.conv2d(z, self.value_proj.weight.float())
            raw = F.conv2d(z, self.offset_proj.weight.float())
            raw = raw.reshape(batch, self.aux_heads, 4, 2, height * width).permute(0, 4, 1, 2, 3)
            logits = F.conv2d(z, self.weight_proj.weight.float())
            logits = logits.reshape(batch, self.aux_heads, 4, height * width).permute(0, 3, 1, 2)
            weights = torch.softmax(logits, dim=-1)
            query, pixels, grid = self.sampling_geometry(raw, height, width)
            related = self.aggregate(value, grid, weights)
        return related, {"query": query, "pixels": pixels, "grid": grid, "weights": weights}

    def forward(self, x):
        """Return the unscaled local delta; the output projection alone opens the zero boundary."""
        related, _ = self.relation_features(x)
        return self.output_proj(related.to(dtype=self.output_proj.weight.dtype))


class GSDRAIFIV3(AIFI):
    """Preserve original AIFI state paths and residual order, adding only the local branch."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, aux_dim: int = 128,
                 aux_heads: int = 4, n_points: int = 4, radius_cells: float = 2.0,
                 dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        if not isinstance(c1, int) or isinstance(c1, bool) or c1 <= 0 or c1 % 4:
            raise ValueError("c1 must be positive and divisible by 4 for AIFI position encoding")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0 or c1 % num_heads:
            raise ValueError("num_heads must be positive and divide c1")
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.c1 = c1
        # Original AIFI and all later layers consume exactly the baseline constructor RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.sparse_relation = QueryLocalDeformableRelationV3(c1, aux_dim, aux_heads, n_points, radius_cells)

    def forward(self, x):
        """Use precisely V2's pre/post-norm integration point and the original dense AIFI operations."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"x must have shape [B, {self.c1}, H, W]")
        batch, channels, height, width = x.shape
        pos = self.build_2d_sincos_position_embedding(width, height, channels).to(device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).transpose(1, 2)
        if self.normalize_before:
            normalized = self.norm1(tokens)
            query = key = self.with_pos_embed(normalized, pos)
            dense = self.ma(query, key, value=normalized)[0]
            normalized_map = normalized.transpose(1, 2).reshape(batch, channels, height, width)
            sparse = self.sparse_relation(normalized_map).flatten(2).transpose(1, 2)
            tokens = tokens + self.dropout1(dense)
            tokens = tokens + sparse
            normalized = self.norm2(tokens)
            ffn = self.fc2(self.dropout(self.act(self.fc1(normalized))))
            tokens = tokens + self.dropout2(ffn)
        else:
            query = key = self.with_pos_embed(tokens, pos)
            dense = self.ma(query, key, value=tokens)[0]
            sparse = self.sparse_relation(x).flatten(2).transpose(1, 2)
            tokens = tokens + self.dropout1(dense)
            tokens = tokens + sparse
            tokens = self.norm1(tokens)
            ffn = self.fc2(self.dropout(self.act(self.fc1(tokens))))
            tokens = self.norm2(tokens + self.dropout2(ffn))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
