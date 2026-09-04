# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Global-Sparse Deformable Relation AIFI modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import AIFI

__all__ = ("GSDRAIFI", "SparseDeformableRelation")


class _ContinuousRelativeBias(nn.Module):
    """Map continuous 2D displacements to a scalar attention bias."""

    def __init__(self, hidden_dim: int = 32):
        """Initialize the two-layer displacement MLP with a zero output layer."""
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """Return one scalar bias for every relative displacement vector."""
        return self.fc2(self.act(self.fc1(displacement)))


class SparseDeformableRelation(nn.Module):
    """Relate dense queries to content-adaptive sparse keys and values."""

    def __init__(
        self,
        c1: int,
        aux_dim: int = 128,
        aux_heads: int = 4,
        offset_groups: int = 4,
        sample_stride: int = 2,
        offset_range_factor: float = 2.0,
        offset_kernel: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the sparse deformable relation branch.

        Args:
            c1 (int): Input and output channels.
            aux_dim (int): Reduced channel width used by the sparse branch.
            aux_heads (int): Number of sparse relation heads.
            offset_groups (int): Number of independently sampled channel groups.
            sample_stride (int): Spatial stride of the sparse reference grid.
            offset_range_factor (float): Maximum offset in sparse-grid cell units.
            offset_kernel (int): Odd depthwise offset-convolution kernel size.
            dropout (float): Attention and output-feature dropout probability.
        """
        super().__init__()
        integer_values = {
            "c1": c1,
            "aux_dim": aux_dim,
            "aux_heads": aux_heads,
            "offset_groups": offset_groups,
            "sample_stride": sample_stride,
            "offset_kernel": offset_kernel,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, but received {value!r}")
        if aux_dim % aux_heads != 0:
            raise ValueError(f"aux_dim ({aux_dim}) must be divisible by aux_heads ({aux_heads})")
        if aux_dim % offset_groups != 0:
            raise ValueError(f"aux_dim ({aux_dim}) must be divisible by offset_groups ({offset_groups})")
        if aux_heads % offset_groups != 0:
            raise ValueError(
                f"aux_heads ({aux_heads}) must be divisible by offset_groups ({offset_groups}) "
                "so every attention head uses one deformable grid"
            )
        if offset_kernel % 2 == 0:
            raise ValueError(f"offset_kernel must be odd, but received {offset_kernel}")
        if not isinstance(offset_range_factor, (int, float)) or isinstance(offset_range_factor, bool):
            raise ValueError("offset_range_factor must be a finite non-negative number")
        if not math.isfinite(float(offset_range_factor)) or offset_range_factor < 0:
            raise ValueError("offset_range_factor must be a finite non-negative number")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), but received {dropout!r}")

        self.c1 = c1
        self.aux_dim = aux_dim
        self.aux_heads = aux_heads
        self.offset_groups = offset_groups
        self.sample_stride = sample_stride
        self.offset_range_factor = float(offset_range_factor)
        self.offset_kernel = offset_kernel
        self.head_dim = aux_dim // aux_heads
        self.group_dim = aux_dim // offset_groups
        self.heads_per_group = aux_heads // offset_groups
        self.attention_dropout = float(dropout)

        self.input_proj = nn.Conv2d(c1, aux_dim, kernel_size=1)
        self.q_proj = nn.Conv2d(aux_dim, aux_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(aux_dim, aux_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(aux_dim, aux_dim, kernel_size=1)
        self.offset_conv = nn.Conv2d(
            aux_dim,
            aux_dim,
            kernel_size=offset_kernel,
            stride=sample_stride,
            padding=offset_kernel // 2,
            groups=aux_dim,
            bias=False,
        )
        self.offset_norm = nn.GroupNorm(offset_groups, aux_dim)
        self.offset_act = nn.GELU()
        self.offset_out = nn.Conv2d(aux_dim, 2 * offset_groups, kernel_size=1, groups=offset_groups)
        self.relative_bias = nn.ModuleList(_ContinuousRelativeBias(32) for _ in range(offset_groups))
        self.output_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Conv2d(aux_dim, c1, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Set all branch boundary projections required for baseline equivalence to zero."""
        nn.init.zeros_(self.offset_out.weight)
        nn.init.zeros_(self.offset_out.bias)
        for mlp in self.relative_bias:
            nn.init.zeros_(mlp.fc2.weight)
            nn.init.zeros_(mlp.fc2.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _reference_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build an align_corners=False grid of cell centers in ``(x, y)`` order."""
        if height <= 0 or width <= 0:
            raise ValueError(f"grid height and width must be positive, but received {(height, width)}")
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).to(dtype=dtype).unsqueeze(0)

    def _sampling_grid(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reference positions, bounded offsets, and deformed sampling positions."""
        raw = self.offset_out(self.offset_act(self.offset_norm(self.offset_conv(features))))
        batch, _, sample_h, sample_w = raw.shape
        raw = raw.view(batch, self.offset_groups, 2, sample_h, sample_w).permute(0, 1, 3, 4, 2)

        # Coordinates use grid_sample's (x, y) order. With align_corners=False, adjacent sparse-grid cell centers
        # are 2 / size apart. Singleton axes receive no displacement rather than dividing by zero.
        x_scale = self.offset_range_factor * (2.0 / sample_w) if sample_w > 1 else 0.0
        y_scale = self.offset_range_factor * (2.0 / sample_h) if sample_h > 1 else 0.0
        scale = raw.new_tensor((x_scale, y_scale))
        offsets = torch.tanh(raw.float()).to(dtype=raw.dtype) * scale
        reference = self._reference_grid(sample_h, sample_w, features.device, features.dtype)
        deformed = (reference[:, None] + offsets).clamp(-1.0, 1.0)
        return reference, offsets, deformed

    def _sample_grouped(self, features: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Sample each contiguous channel group from its own deformable grid."""
        batch, channels, height, width = features.shape
        sample_h, sample_w = grid.shape[2:4]
        grouped = features.reshape(batch * self.offset_groups, self.group_dim, height, width)
        grouped_grid = grid.reshape(batch * self.offset_groups, sample_h, sample_w, 2)
        sampled = F.grid_sample(
            grouped,
            grouped_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.reshape(batch, channels, sample_h, sample_w)

    def _split_heads(self, features: torch.Tensor) -> torch.Tensor:
        """Convert a feature map to ``[B, heads, tokens, head_dim]``."""
        batch, _, height, width = features.shape
        return (
            features.flatten(2)
            .transpose(1, 2)
            .reshape(batch, height * width, self.aux_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _relative_displacement_bias(
        self,
        query_grid: torch.Tensor,
        sampled_grid: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute per-head bias from each query to every actual sampled position."""
        query = query_grid.flatten(1, 2)
        sampled = sampled_grid.flatten(2, 3)
        group_biases = []
        for group, mlp in enumerate(self.relative_bias):
            displacement = query[:, :, None, :] - sampled[:, group, None, :, :]
            mlp_dtype = mlp.fc1.weight.dtype
            bias = mlp(displacement.to(dtype=mlp_dtype)).squeeze(-1).float()
            group_biases.append(bias)
        grouped = torch.stack(group_biases, dim=1)
        head_groups = torch.arange(self.aux_heads, device=grouped.device) // self.heads_per_group
        return grouped.index_select(1, head_groups).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sparse relation residual with the same shape as ``x``."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"x must have shape [B, {self.c1}, H, W], but received {tuple(x.shape)}")
        if x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ValueError("x must have non-empty spatial dimensions")

        projected = self.input_proj(x)
        query = self._split_heads(self.q_proj(projected))
        _, _, sampled_grid = self._sampling_grid(projected)
        key = self._split_heads(self._sample_grouped(self.k_proj(projected), sampled_grid))
        value = self._split_heads(self._sample_grouped(self.v_proj(projected), sampled_grid))

        query_grid = self._reference_grid(x.shape[-2], x.shape[-1], x.device, x.dtype)
        relative_bias = self._relative_displacement_bias(query_grid, sampled_grid, torch.float32)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores + relative_bias.float(), dim=-1)
        attention = F.dropout(attention, p=self.attention_dropout, training=self.training)
        related = torch.matmul(attention, value.float()).to(dtype=x.dtype)
        related = related.transpose(1, 2).reshape(x.shape[0], x.shape[-2] * x.shape[-1], self.aux_dim)
        related = related.transpose(1, 2).reshape(x.shape[0], self.aux_dim, x.shape[-2], x.shape[-1])
        return self.output_proj(self.output_dropout(related))


class GSDRAIFI(AIFI):
    """Baseline AIFI augmented by a zero-initialized sparse deformable relation residual."""

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        aux_dim: int = 128,
        aux_heads: int = 4,
        offset_groups: int = 4,
        sample_stride: int = 2,
        offset_range_factor: float = 2.0,
        offset_kernel: int = 3,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """Initialize the unchanged dense AIFI path and the supplementary sparse relation branch."""
        if not isinstance(c1, int) or isinstance(c1, bool) or c1 <= 0:
            raise ValueError(f"c1 must be a positive integer, but received {c1!r}")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, but received {num_heads!r}")
        if c1 % num_heads != 0:
            raise ValueError(f"c1 ({c1}) must be divisible by num_heads ({num_heads})")
        if c1 % 4 != 0:
            raise ValueError(f"c1 ({c1}) must be divisible by 4 for the 2D sine-cosine position embedding")
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.c1 = c1
        self.sparse_relation = SparseDeformableRelation(
            c1=c1,
            aux_dim=aux_dim,
            aux_heads=aux_heads,
            offset_groups=offset_groups,
            sample_stride=sample_stride,
            offset_range_factor=offset_range_factor,
            offset_kernel=offset_kernel,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense AIFI attention plus sparse relation, then the original AIFI FFN."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"x must have shape [B, {self.c1}, H, W], but received {tuple(x.shape)}")
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
