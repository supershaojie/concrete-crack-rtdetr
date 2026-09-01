# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Cross-scale semantic-consistency edge fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CSCEF",)


class CSCEF(nn.Module):
    """Enhance a lateral feature with semantically verified Scharr edges.

    The module accepts ``[x_lateral, x_semantic]`` with shapes ``[B, C_l, H, W]`` and
    ``[B, C_s, H_s, W_s]``. The semantic feature is resized when necessary, and the
    returned tensor always has the same shape as ``x_lateral``. A zero-initialized
    residual scale makes the module an identity mapping at initialization.

    Args:
        c_lateral (int): Number of channels in the shallow lateral feature.
        c_semantic (int): Number of channels in the deep semantic feature.
        hidden_channels (int, optional): Shared projection width. Automatically selected when omitted.
        eps (float): Numerical stability constant used by edge magnitude and cosine similarity.
    """

    def __init__(
        self,
        c_lateral: int,
        c_semantic: int,
        hidden_channels: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Initialize feature projections, fixed Scharr kernels, and residual gate parameters."""
        super().__init__()
        if c_lateral <= 0 or c_semantic <= 0:
            raise ValueError("CSCEF input channel counts must be positive.")
        if eps <= 0:
            raise ValueError("CSCEF eps must be positive.")

        hidden = max(16, min(c_lateral, c_semantic) // 8) if hidden_channels is None else hidden_channels
        if hidden <= 0:
            raise ValueError("CSCEF hidden_channels must be positive.")

        self.eps = float(eps)
        self.lateral_projection = nn.Conv2d(c_lateral, hidden, kernel_size=1, bias=False)
        self.semantic_projection = nn.Conv2d(c_semantic, hidden, kernel_size=1, bias=False)
        self.edge_projection = nn.Conv2d(hidden, c_lateral, kernel_size=1, bias=False)

        # Divide by the Scharr kernel L1 norm to keep the fixed gradient response well scaled.
        scharr_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        scharr_y = scharr_x.transpose(0, 1).contiguous()
        self.register_buffer("scharr_x", scharr_x.view(1, 1, 3, 3))
        self.register_buffer("scharr_y", scharr_y.view(1, 1, 3, 3))

        self.similarity_scale = nn.Parameter(torch.tensor(1.0))
        self.similarity_bias = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _scharr_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return a stable per-channel Scharr gradient magnitude for ``x``."""
        channels = x.shape[1]
        kernel_x = self.scharr_x.to(dtype=x.dtype).expand(channels, 1, 3, 3)
        kernel_y = self.scharr_y.to(dtype=x.dtype).expand(channels, 1, 3, 3)
        gradient_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        gradient_y = F.conv2d(x, kernel_y, padding=1, groups=channels)

        # Accumulate squares in FP32 under AMP to avoid half-precision overflow or underflow.
        magnitude_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        gradient_x = gradient_x.to(magnitude_dtype)
        gradient_y = gradient_y.to(magnitude_dtype)
        return torch.sqrt(gradient_x.square() + gradient_y.square() + self.eps).to(x.dtype)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse ``[x_lateral, x_semantic]`` and return an enhanced lateral feature."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CSCEF expects [x_lateral, x_semantic].")
        x_lateral, x_semantic = inputs
        if x_lateral.ndim != 4 or x_semantic.ndim != 4:
            raise ValueError("CSCEF inputs must be 4D BCHW tensors.")

        if x_semantic.shape[-2:] != x_lateral.shape[-2:]:
            x_semantic = F.interpolate(x_semantic, size=x_lateral.shape[-2:], mode="bilinear", align_corners=False)

        lateral_aligned = self.lateral_projection(x_lateral)
        semantic_aligned = self.semantic_projection(x_semantic)
        edge = self._scharr_magnitude(lateral_aligned)

        similarity_dtype = (
            torch.float32 if lateral_aligned.dtype in {torch.float16, torch.bfloat16} else lateral_aligned.dtype
        )
        consistency = F.cosine_similarity(
            lateral_aligned.to(similarity_dtype), semantic_aligned.to(similarity_dtype), dim=1, eps=self.eps
        ).unsqueeze(1)
        gate = torch.sigmoid(self.similarity_scale * consistency + self.similarity_bias).to(edge.dtype)
        residual = self.edge_projection(edge) * gate
        return x_lateral + self.gamma * residual
