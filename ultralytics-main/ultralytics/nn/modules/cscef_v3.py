# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Bias-free discrepancy-gated cross-scale edge fusion module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CSCEFv3",)


class CSCEFv3(nn.Module):
    """Enhance a lateral feature with discrepancy-gated, RMS-matched Scharr edges.

    Both inputs must have the same channel count so one bias-free projection can embed them. The semantic input is
    resized to the lateral spatial resolution when necessary. The output always matches the lateral input's shape and
    dtype.

    Args:
        c_lateral (int): Channels in the projected shallow lateral feature.
        c_semantic (int): Channels in the upsampled deep semantic feature.
        hidden_channels (int, optional): Width of the shared embedding. Defaults to 32.
        num_groups (int): GroupNorm group count. Defaults to 8.
        alpha_init (float): Initial effective global residual scale. Defaults to 0.01.
        alpha_max (float): Strict upper bound for the effective residual scale. Defaults to 0.05.
        eps (float): Numerical stability constant. Defaults to 1e-6.
    """

    def __init__(
        self,
        c_lateral: int,
        c_semantic: int,
        hidden_channels: int | None = None,
        num_groups: int = 8,
        alpha_init: float = 0.01,
        alpha_max: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the bias-free embedding, fixed edge path, RMS matcher, and bounded global scale."""
        super().__init__()
        if not isinstance(c_lateral, int) or not isinstance(c_semantic, int) or c_lateral <= 0 or c_semantic <= 0:
            raise ValueError("CSCEFv3 input channel counts must be positive integers.")
        if c_lateral != c_semantic:
            raise ValueError(
                "CSCEFv3 requires equal lateral and semantic channel counts for its shared projection, "
                f"but received {c_lateral} and {c_semantic}."
            )
        hidden = 32 if hidden_channels is None else hidden_channels
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError("CSCEFv3 hidden_channels must be a positive integer.")
        if not isinstance(num_groups, int) or num_groups <= 0 or hidden % num_groups:
            raise ValueError("CSCEFv3 num_groups must be a positive divisor of hidden_channels.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("CSCEFv3 eps must be finite and positive.")
        if not all(math.isfinite(value) for value in (alpha_init, alpha_max)):
            raise ValueError("CSCEFv3 alpha initialization and bound must be finite.")
        if not 0 < alpha_init < alpha_max:
            raise ValueError("CSCEFv3 requires 0 < alpha_init < alpha_max.")

        self.in_channels = c_lateral
        self.hidden_channels = hidden
        self.num_groups = num_groups
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)

        self.shared_projection = nn.Conv2d(c_lateral, hidden, kernel_size=1, bias=False)
        self.shared_norm = nn.GroupNorm(num_groups, hidden, affine=False)
        self.depthwise_conv = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False
        )
        self.edge_norm = nn.GroupNorm(num_groups, hidden, affine=False)
        self.edge_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(hidden, c_lateral, kernel_size=1, bias=False)

        # Match CSCEF-v2 exactly: each fixed Scharr kernel is divided by its L1 norm (32).
        scharr_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        scharr_y = scharr_x.transpose(0, 1).contiguous()
        self.register_buffer("scharr_x", scharr_x.view(1, 1, 3, 3))
        self.register_buffer("scharr_y", scharr_y.view(1, 1, 3, 3))

        alpha_fraction = alpha_init / alpha_max
        raw_alpha_init = math.log(alpha_fraction / (1.0 - alpha_fraction))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha_init, dtype=torch.float32))

    def _project_features(
        self, x_lateral: torch.Tensor, x_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize the semantic input and apply shared bias-free projection and normalization."""
        if x_lateral.ndim != 4 or x_semantic.ndim != 4:
            raise ValueError("CSCEFv3 inputs must be 4D BCHW tensors.")
        if x_lateral.shape[1] != self.in_channels or x_semantic.shape[1] != self.in_channels:
            raise ValueError(
                f"CSCEFv3 expected both inputs to have {self.in_channels} channels, but received "
                f"{x_lateral.shape[1]} and {x_semantic.shape[1]}."
            )
        if min(*x_lateral.shape[-2:], *x_semantic.shape[-2:]) <= 0:
            raise ValueError("CSCEFv3 inputs must have non-empty spatial dimensions.")
        if x_lateral.device != x_semantic.device:
            raise ValueError("CSCEFv3 inputs must be on the same device.")
        if x_semantic.shape[-2:] != x_lateral.shape[-2:]:
            x_semantic = F.interpolate(x_semantic, size=x_lateral.shape[-2:], mode="bilinear", align_corners=False)

        lateral_embedding = self.shared_norm(self.shared_projection(x_lateral))
        semantic_embedding = self.shared_norm(self.shared_projection(x_semantic))
        return lateral_embedding, semantic_embedding

    def _compute_discrepancy_gate(
        self, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return the FP32 channel-centered cosine discrepancy gate in ``B x 1 x H x W`` form."""
        if lateral_embedding.shape != semantic_embedding.shape:
            raise ValueError("CSCEFv3 projected features must have identical shapes before discrepancy gating.")

        with torch.autocast(device_type=lateral_embedding.device.type, enabled=False):
            lateral_fp32 = lateral_embedding.float()
            semantic_fp32 = semantic_embedding.float()
            lateral_centered = lateral_fp32 - lateral_fp32.mean(dim=1, keepdim=True)
            semantic_centered = semantic_fp32 - semantic_fp32.mean(dim=1, keepdim=True)
            lateral_normalized = F.normalize(lateral_centered, p=2, dim=1, eps=self.eps)
            semantic_normalized = F.normalize(semantic_centered, p=2, dim=1, eps=self.eps)
            cosine = (lateral_normalized * semantic_normalized).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            return ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)

    def _compute_scharr_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return fixed per-channel Scharr magnitude in FP32 with safe small-spatial padding."""
        if x.ndim != 4:
            raise ValueError("CSCEFv3 Scharr input must be a 4D BCHW tensor.")
        if min(x.shape[-2:]) <= 0:
            raise ValueError("CSCEFv3 Scharr input must have non-empty spatial dimensions.")

        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            channels = x_fp32.shape[1]
            kernel_x = self.scharr_x.float().expand(channels, 1, 3, 3)
            kernel_y = self.scharr_y.float().expand(channels, 1, 3, 3)
            padding_mode = "reflect" if x_fp32.shape[-2] > 1 and x_fp32.shape[-1] > 1 else "replicate"
            padded = F.pad(x_fp32, (1, 1, 1, 1), mode=padding_mode)
            gradient_x = F.conv2d(padded, kernel_x, padding=0, groups=channels)
            gradient_y = F.conv2d(padded, kernel_y, padding=0, groups=channels)
            return torch.sqrt(gradient_x.square() + gradient_y.square() + self.eps)

    def _match_residual_rms(self, x_lateral: torch.Tensor, residual_raw: torch.Tensor) -> torch.Tensor:
        """Match ungated residual RMS to lateral RMS per sample using a detached, capped FP32 ratio."""
        if x_lateral.shape != residual_raw.shape:
            raise ValueError("CSCEFv3 lateral and raw residual shapes must match for RMS normalization.")

        with torch.autocast(device_type=x_lateral.device.type, enabled=False):
            lateral_fp32 = x_lateral.float()
            residual_fp32 = residual_raw.float()
            reduce_dims = (1, 2, 3)
            lateral_rms = torch.sqrt(lateral_fp32.square().mean(dim=reduce_dims, keepdim=True) + self.eps)
            residual_rms = torch.sqrt(residual_fp32.square().mean(dim=reduce_dims, keepdim=True) + self.eps)
            rms_scale = (lateral_rms / (residual_rms + self.eps)).clamp(max=10.0).detach()
            return residual_fp32 * rms_scale

    def _effective_alpha(self) -> torch.Tensor:
        """Return the FP32 global residual scale, strictly bounded between zero and ``alpha_max``."""
        alpha = self.alpha_max * torch.sigmoid(self.raw_alpha.float())
        zero = alpha.new_zeros(())
        upper = torch.nextafter(alpha.new_tensor(self.alpha_max), zero)
        lower = torch.nextafter(zero, alpha.new_ones(()))
        return alpha.clamp(min=lower, max=upper)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse ``[x_lateral, x_semantic]`` and return only the enhanced lateral feature."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CSCEFv3 expects [x_lateral, x_semantic].")
        x_lateral, x_semantic = inputs
        lateral_embedding, semantic_embedding = self._project_features(x_lateral, x_semantic)
        gate = self._compute_discrepancy_gate(lateral_embedding, semantic_embedding)

        edge_magnitude = self._compute_scharr_magnitude(lateral_embedding)
        edge_for_conv = edge_magnitude.to(
            device=self.depthwise_conv.weight.device,
            dtype=self.depthwise_conv.weight.dtype,
        )
        edge_feature = self.edge_activation(self.edge_norm(self.depthwise_conv(edge_for_conv)))
        residual_raw = self.output_projection(edge_feature)
        residual_normalized = self._match_residual_rms(x_lateral, residual_raw)

        gate_for_residual = gate.to(device=residual_normalized.device, dtype=residual_normalized.dtype)
        alpha = self._effective_alpha().to(device=residual_normalized.device, dtype=residual_normalized.dtype)
        delta = alpha * gate_for_residual * residual_normalized
        return x_lateral + delta.to(dtype=x_lateral.dtype)
