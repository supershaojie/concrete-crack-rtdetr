# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Constrained cross-scale semantic-consistency edge fusion module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CSCEFv2",)


def _group_count(channels: int) -> int:
    """Return the largest supported GroupNorm group count that divides ``channels``."""
    return next(groups for groups in (8, 4, 2, 1) if channels % groups == 0)


class CSCEFv2(nn.Module):
    """Enhance a lateral feature with bounded, relatively gated Scharr edges.

    Both inputs must have the same channel count so that one shared projection can embed them. The semantic input is
    resized to the lateral spatial resolution when needed, while the output always matches the lateral input shape.

    Args:
        c_lateral (int): Number of channels in the projected shallow lateral feature.
        c_semantic (int): Number of channels in the upsampled deep semantic feature.
        hidden_channels (int, optional): Width of the shared embedding. Defaults to 32.
        temperature_init (float): Initial effective gate temperature.
        temperature_min (float): Lower bound of the effective gate temperature.
        temperature_max (float): Upper bound of the effective gate temperature.
        layer_scale_init (float): Initial effective per-channel residual scale.
        layer_scale_max (float): Absolute bound of the effective residual scale.
        similarity_clip (float): Absolute clamp applied to standardized cosine similarity.
        eps (float): Numerical stability constant.
    """

    def __init__(
        self,
        c_lateral: int,
        c_semantic: int,
        hidden_channels: int | None = None,
        temperature_init: float = 1.0,
        temperature_min: float = 0.5,
        temperature_max: float = 4.0,
        layer_scale_init: float = 0.01,
        layer_scale_max: float = 0.10,
        similarity_clip: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the shared embedding, calibrated edge path, gate, and bounded LayerScale."""
        super().__init__()
        if not isinstance(c_lateral, int) or not isinstance(c_semantic, int) or c_lateral <= 0 or c_semantic <= 0:
            raise ValueError("CSCEFv2 input channel counts must be positive integers.")
        if c_lateral != c_semantic:
            raise ValueError(
                "CSCEFv2 requires equal lateral and semantic channel counts for its shared projection, "
                f"but received {c_lateral} and {c_semantic}."
            )
        hidden = 32 if hidden_channels is None else hidden_channels
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError("CSCEFv2 hidden_channels must be a positive integer.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("CSCEFv2 eps must be finite and positive.")
        if not all(math.isfinite(value) for value in (temperature_min, temperature_init, temperature_max)):
            raise ValueError("CSCEFv2 temperature bounds and initialization must be finite.")
        if not temperature_min < temperature_init < temperature_max:
            raise ValueError("CSCEFv2 requires temperature_min < temperature_init < temperature_max.")
        if not all(math.isfinite(value) for value in (layer_scale_init, layer_scale_max)):
            raise ValueError("CSCEFv2 LayerScale bounds and initialization must be finite.")
        if not 0 < layer_scale_init < layer_scale_max:
            raise ValueError("CSCEFv2 requires 0 < layer_scale_init < layer_scale_max.")
        if not math.isfinite(similarity_clip) or similarity_clip <= 0:
            raise ValueError("CSCEFv2 similarity_clip must be finite and positive.")

        self.in_channels = c_lateral
        self.hidden_channels = hidden
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.layer_scale_max = float(layer_scale_max)
        self.similarity_clip = float(similarity_clip)
        self.eps = float(eps)

        groups = _group_count(hidden)
        self.shared_projection = nn.Conv2d(c_lateral, hidden, kernel_size=1, bias=False)
        self.shared_norm = nn.GroupNorm(groups, hidden)
        self.edge_calibration = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.output_projection = nn.Conv2d(hidden, c_lateral, kernel_size=1, bias=False)

        # Divide by the Scharr kernel L1 norm to keep the fixed gradient response well scaled.
        scharr_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        scharr_y = scharr_x.transpose(0, 1).contiguous()
        self.register_buffer("scharr_x", scharr_x.view(1, 1, 3, 3))
        self.register_buffer("scharr_y", scharr_y.view(1, 1, 3, 3))

        temperature_fraction = (temperature_init - temperature_min) / (temperature_max - temperature_min)
        raw_temperature_init = math.log(temperature_fraction / (1.0 - temperature_fraction))
        self.raw_temperature = nn.Parameter(torch.tensor(raw_temperature_init, dtype=torch.float32))

        raw_layer_scale_init = math.atanh(layer_scale_init / layer_scale_max)
        self.layer_scale_raw = nn.Parameter(torch.full((1, c_lateral, 1, 1), raw_layer_scale_init))

    def _effective_temperature(self) -> torch.Tensor:
        """Return the sigmoid-constrained scalar gate temperature in FP32."""
        temperature_range = self.temperature_max - self.temperature_min
        return self.temperature_min + temperature_range * torch.sigmoid(self.raw_temperature.float())

    def _effective_layer_scale(self) -> torch.Tensor:
        """Return the bounded per-channel residual scale."""
        return self.layer_scale_max * torch.tanh(self.layer_scale_raw)

    def _project_features(
        self, x_lateral: torch.Tensor, x_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize the semantic input and embed both inputs with the same projection and normalization weights."""
        if x_lateral.ndim != 4 or x_semantic.ndim != 4:
            raise ValueError("CSCEFv2 inputs must be 4D BCHW tensors.")
        if x_lateral.shape[1] != self.in_channels or x_semantic.shape[1] != self.in_channels:
            raise ValueError(
                f"CSCEFv2 expected both inputs to have {self.in_channels} channels, but received "
                f"{x_lateral.shape[1]} and {x_semantic.shape[1]}."
            )
        if min(*x_lateral.shape[-2:], *x_semantic.shape[-2:]) <= 0:
            raise ValueError("CSCEFv2 inputs must have non-empty spatial dimensions.")
        if x_semantic.shape[-2:] != x_lateral.shape[-2:]:
            x_semantic = F.interpolate(x_semantic, size=x_lateral.shape[-2:], mode="bilinear", align_corners=False)

        lateral_embedding = self.shared_norm(self.shared_projection(x_lateral))
        semantic_embedding = self.shared_norm(self.shared_projection(x_semantic))
        return lateral_embedding, semantic_embedding

    def _scharr_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return reflect-padded per-channel Scharr magnitude, with a safe fallback for singleton dimensions."""
        if x.ndim != 4:
            raise ValueError("CSCEFv2 Scharr input must be a 4D BCHW tensor.")
        if min(x.shape[-2:]) <= 0:
            raise ValueError("CSCEFv2 Scharr input must have non-empty spatial dimensions.")

        channels = x.shape[1]
        kernel_x = self.scharr_x.to(device=x.device, dtype=x.dtype).expand(channels, 1, 3, 3)
        kernel_y = self.scharr_y.to(device=x.device, dtype=x.dtype).expand(channels, 1, 3, 3)
        padding_mode = "reflect" if x.shape[-2] > 1 and x.shape[-1] > 1 else "replicate"
        padded = F.pad(x, (1, 1, 1, 1), mode=padding_mode)
        gradient_x = F.conv2d(padded, kernel_x, padding=0, groups=channels)
        gradient_y = F.conv2d(padded, kernel_y, padding=0, groups=channels)

        # Always accumulate the squared magnitude in FP32, including under AMP.
        magnitude = torch.sqrt(gradient_x.float().square() + gradient_y.float().square() + self.eps)
        return magnitude.to(x.dtype)

    def _semantic_gate(self, lateral_embedding: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
        """Return an FP32 gate from per-sample spatially standardized cosine similarity."""
        if lateral_embedding.shape != semantic_embedding.shape:
            raise ValueError("CSCEFv2 projected features must have identical shapes before semantic gating.")

        consistency = F.cosine_similarity(
            lateral_embedding.float(), semantic_embedding.float(), dim=1, eps=self.eps
        ).unsqueeze(1)
        centered = consistency - consistency.mean(dim=(-2, -1), keepdim=True)
        variance = centered.square().mean(dim=(-2, -1), keepdim=True)
        standardized = centered * torch.rsqrt(variance + self.eps)
        standardized = standardized.clamp(min=-self.similarity_clip, max=self.similarity_clip)
        return torch.sigmoid(self._effective_temperature() * standardized)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Fuse ``[x_lateral, x_semantic]`` and return a bounded enhancement of the lateral feature."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CSCEFv2 expects [x_lateral, x_semantic].")
        x_lateral, x_semantic = inputs
        lateral_embedding, semantic_embedding = self._project_features(x_lateral, x_semantic)
        edge = self._scharr_magnitude(lateral_embedding)
        calibrated_edge = self.edge_calibration(edge)
        gate = self._semantic_gate(lateral_embedding, semantic_embedding).to(calibrated_edge.dtype)
        residual = self.output_projection(calibrated_edge * gate)
        layer_scale = self._effective_layer_scale().to(residual.dtype)
        return x_lateral + layer_scale * residual
