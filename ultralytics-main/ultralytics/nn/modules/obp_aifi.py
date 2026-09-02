# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Orthogonal-Branch Perception AIFI modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import AIFI

__all__ = ("AdaptiveOrthogonalMixer", "OBPAIFI")


class AdaptiveOrthogonalMixer(nn.Module):
    """Mix horizontal, vertical, and local depthwise features with sample-adaptive weights."""

    def __init__(
        self,
        c1: int,
        cm: int = 1024,
        kernel_size: int = 7,
        dropout: float = 0.0,
        act: nn.Module | None = None,
    ):
        """Initialize the adaptive orthogonal mixer.

        Args:
            c1 (int): Input and output channels.
            cm (int): Hidden channels shared by all three branches.
            kernel_size (int): Odd strip-kernel length.
            dropout (float): Dropout probability before the output projection.
            act (nn.Module, optional): Activation module. Defaults to GELU.
        """
        super().__init__()
        if c1 <= 0 or cm <= 0:
            raise ValueError(f"c1 and cm must be positive, but received c1={c1}, cm={cm}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, but received {kernel_size}")

        padding = kernel_size // 2
        gate_channels = max(cm // 16, 16)
        self.c1 = c1
        self.cm = cm
        self.kernel_size = kernel_size
        self.input_proj = nn.Conv2d(c1, cm, kernel_size=1)
        self.horizontal = nn.Conv2d(
            cm, cm, kernel_size=(1, kernel_size), padding=(0, padding), groups=cm
        )
        self.vertical = nn.Conv2d(
            cm, cm, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=cm
        )
        self.local = nn.Conv2d(cm, cm, kernel_size=3, padding=1, groups=cm)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_reduce = nn.Conv2d(cm, gate_channels, kernel_size=1)
        self.gate_expand = nn.Conv2d(gate_channels, 3, kernel_size=1)
        self.act = act if act is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Conv2d(cm, c1, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize pointwise, depthwise, and gate convolutions."""
        for module in (self.input_proj, self.gate_reduce, self.output_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in (self.horizontal, self.vertical, self.local):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Equal logits make the initial branch weights exactly one third.
        nn.init.zeros_(self.gate_expand.weight)
        nn.init.zeros_(self.gate_expand.bias)

    def branch_weights(self, projected: torch.Tensor) -> torch.Tensor:
        """Return broadcastable sample-level branch weights for a projected feature map."""
        if projected.ndim != 4 or projected.shape[1] != self.cm:
            raise ValueError(
                f"projected must have shape [B, {self.cm}, H, W], but received {tuple(projected.shape)}"
            )
        logits = self.gate_expand(self.act(self.gate_reduce(self.gate_pool(projected))))
        weights = torch.softmax(logits.float(), dim=1).to(dtype=projected.dtype)
        return weights.unsqueeze(2)  # [B, 3, 1, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three parallel spatial branches and project the fused feature back to ``c1`` channels."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"x must have shape [B, {self.c1}, H, W], but received {tuple(x.shape)}")

        projected = self.act(self.input_proj(x))
        branches = (self.horizontal(projected), self.vertical(projected), self.local(projected))
        expected_shape = projected.shape
        if any(branch.shape != expected_shape for branch in branches):
            raise RuntimeError(
                "Orthogonal branch shapes must match the projected feature shape; "
                f"expected {tuple(expected_shape)}, received {[tuple(branch.shape) for branch in branches]}"
            )

        weights = self.branch_weights(projected)
        fused = sum(weights[:, index] * branch for index, branch in enumerate(branches))
        return self.output_proj(self.dropout(self.act(fused)))


class OBPAIFI(nn.Module):
    """AIFI with global self-attention and an adaptive orthogonal-branch spatial FFN."""

    def __init__(
        self,
        c1: int,
        cm: int = 1024,
        num_heads: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        normalize_before: bool = False,
    ):
        """Initialize OBP-AIFI.

        Args:
            c1 (int): Input and output channels.
            cm (int): Hidden channels in the orthogonal mixer.
            num_heads (int): Number of global self-attention heads.
            kernel_size (int): Odd strip-kernel length.
            dropout (float): Dropout probability.
            act (nn.Module, optional): Mixer activation module. Defaults to GELU.
            normalize_before (bool): Whether to use the AIFI-compatible pre-normalization path.
        """
        super().__init__()
        if c1 <= 0 or num_heads <= 0:
            raise ValueError(f"c1 and num_heads must be positive, but received c1={c1}, num_heads={num_heads}")
        if c1 % num_heads != 0:
            raise ValueError(f"c1 ({c1}) must be divisible by num_heads ({num_heads})")
        if c1 % 4 != 0:
            raise ValueError(f"c1 ({c1}) must be divisible by 4 for the 2D sine-cosine position embedding")

        activation = act if act is not None else nn.GELU()
        self.c1 = c1
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mixer = AdaptiveOrthogonalMixer(c1, cm, kernel_size, dropout, activation)
        self.gamma = nn.Parameter(torch.zeros(c1))
        self.normalize_before = normalize_before

    @staticmethod
    def _with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Add a position embedding to a token tensor."""
        return tensor + pos

    def _mix_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Apply the spatial mixer to validated token features."""
        batch, token_count, channels = tokens.shape
        if token_count != height * width or channels != self.c1:
            raise RuntimeError(
                "Cannot restore OBP-AIFI tokens to a feature map: "
                f"received {tuple(tokens.shape)}, expected [B, {height * width}, {self.c1}]"
            )
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        mixed_feature = self.mixer(feature)
        scale = self.gamma.to(dtype=mixed_feature.dtype).view(1, channels, 1, 1)
        mixed = scale * mixed_feature
        return mixed.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global AIFI attention followed by the orthogonal-branch perception mixer."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"x must have shape [B, {self.c1}, H, W], but received {tuple(x.shape)}")

        batch, channels, height, width = x.shape
        pos = AIFI.build_2d_sincos_position_embedding(width, height, channels).to(device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).transpose(1, 2)

        if self.normalize_before:
            normalized = self.norm1(tokens)
            query = key = self._with_pos_embed(normalized, pos)
            attended = self.ma(query, key, value=normalized)[0]
            tokens = tokens + self.dropout1(attended)
            mixed = self._mix_tokens(self.norm2(tokens), height, width)
            tokens = tokens + self.dropout2(mixed)
        else:
            query = key = self._with_pos_embed(tokens, pos)
            attended = self.ma(query, key, value=tokens)[0]
            tokens = self.norm1(tokens + self.dropout1(attended))
            mixed = self._mix_tokens(tokens, height, width)
            tokens = self.norm2(tokens + self.dropout2(mixed))

        if tokens.shape != (batch, height * width, channels):
            raise RuntimeError(
                f"Unexpected OBP-AIFI token shape {tuple(tokens.shape)} after mixing; "
                f"expected {(batch, height * width, channels)}"
            )
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
