# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Structure-guided cross-scale content residual, with an exact identity initialization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("CSCEFv5",)


class CSCEFv5(nn.Module):
    """Fuse lateral and semantic content; detached V4 structure confidence modulates the residual.

    All convolutions use PyTorch's default Kaiming-uniform initialization, except the exactly zero output
    projection. CPU construction consumes no external RNG state, including when the trainer rebuilds at nc=1.
    Scharr is used only for confidence, never for the residual value. There is no amplitude cap or RMS matching.
    """

    def __init__(
        self, c_lateral: int, c_semantic: int, hidden_channels: int = 32, num_groups: int = 8, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if any(type(c) is not int or c <= 0 for c in (c_lateral, c_semantic, hidden_channels, num_groups)):
            raise ValueError("CSCEFv5 channel and group counts must be positive integers.")
        if hidden_channels % num_groups or hidden_channels // num_groups < 2:
            raise ValueError("CSCEFv5 groups must divide hidden_channels with at least two channels per group.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("CSCEFv5 eps must be finite and positive.")
        self.c_lateral = c_lateral
        self.c_semantic = c_semantic
        self.hidden_channels = hidden_channels
        self.eps = float(eps)

        # Keep decoder initialization identical to C2: restore the CPU RNG after ALL random constructors.
        with torch.random.fork_rng(devices=[]):
            self.lateral_projection = nn.Conv2d(c_lateral, hidden_channels, 1, bias=False)
            self.semantic_projection = nn.Conv2d(c_semantic, hidden_channels, 1, bias=False)
            self.mix_projection = nn.Conv2d(2 * hidden_channels, hidden_channels, 1, bias=False)
            self.depthwise_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                                          groups=hidden_channels, bias=False)
            self.output_projection = nn.Conv2d(hidden_channels, c_lateral, 1, bias=False)
            nn.init.zeros_(self.output_projection.weight)
        self.lateral_norm = nn.GroupNorm(num_groups, hidden_channels, eps=eps, affine=False)
        self.semantic_norm = nn.GroupNorm(num_groups, hidden_channels, eps=eps, affine=False)
        self.content_norm = nn.GroupNorm(num_groups, hidden_channels, eps=eps, affine=False)
        self.activation = nn.SiLU()
        scharr_x = torch.tensor(((3.0, 0.0, -3.0), (10.0, 0.0, -10.0), (3.0, 0.0, -3.0))) / 32.0
        self.register_buffer("scharr_x", scharr_x.view(1, 1, 3, 3))
        self.register_buffer("scharr_y", scharr_x.transpose(0, 1).contiguous().view(1, 1, 3, 3))

    def _project_features(self, lateral: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate BCHW inputs and independently project the spatially aligned content."""
        if not isinstance(lateral, torch.Tensor) or not isinstance(semantic, torch.Tensor):
            raise ValueError("CSCEFv5 inputs must be tensors.")
        if lateral.ndim != 4 or semantic.ndim != 4:
            raise ValueError("CSCEFv5 inputs must be 4D BCHW tensors.")
        if lateral.shape[0] != semantic.shape[0] or lateral.shape[0] == 0:
            raise ValueError("CSCEFv5 inputs must have equal, non-empty batches.")
        if lateral.shape[1] != self.c_lateral or semantic.shape[1] != self.c_semantic:
            raise ValueError(f"CSCEFv5 expected channels {self.c_lateral}, {self.c_semantic}.")
        if min(*lateral.shape[-2:], *semantic.shape[-2:]) <= 0:
            raise ValueError("CSCEFv5 spatial dimensions must be non-empty.")
        if lateral.device != semantic.device or lateral.device != self.lateral_projection.weight.device:
            raise ValueError("CSCEFv5 inputs and module must be on the same device.")
        if not lateral.is_floating_point() or not semantic.is_floating_point():
            raise ValueError("CSCEFv5 inputs must have floating point dtypes.")
        # The unchanged RT-DETR neck can deliver FP16 L and FP32 S under AMP.
        # Let each projection obey autocast; the final output still follows L's dtype.
        if semantic.shape[-2:] != lateral.shape[-2:]:
            semantic = F.interpolate(semantic, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        return (self.lateral_norm(self.lateral_projection(lateral)),
                self.semantic_norm(self.semantic_projection(semantic)))

    @torch.no_grad()
    def _compute_scharr_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """V4's /32 Scharr pair, shared padding and grouped convolution, always in FP32."""
        if x.ndim != 4 or min(x.shape[-2:]) <= 0:
            raise ValueError("CSCEFv5 Scharr input must be non-empty BCHW.")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            channels = x_fp32.shape[1]
            kernel_x = self.scharr_x.float().expand(channels, 1, 3, 3)
            kernel_y = self.scharr_y.float().expand(channels, 1, 3, 3)
            paired_kernels = torch.stack((kernel_x, kernel_y), dim=1).reshape(2 * channels, 1, 3, 3)
            padding_mode = "reflect" if x_fp32.shape[-2] > 1 and x_fp32.shape[-1] > 1 else "replicate"
            padded = F.pad(x_fp32, (1, 1, 1, 1), mode=padding_mode)
            paired_gradients = F.conv2d(padded, paired_kernels, padding=0, groups=channels)
            return paired_gradients.reshape(x_fp32.shape[0], channels, 2, *x_fp32.shape[-2:]).unbind(dim=2)

    @torch.no_grad()
    def _compute_structure_terms(
        self, gradient_x: torch.Tensor, gradient_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4 coherence and relative-energy reliability, including count_include_pad=True pool boundaries."""
        if gradient_x.ndim != 4 or gradient_x.shape != gradient_y.shape or min(gradient_x.shape[-2:]) <= 0:
            raise ValueError("CSCEFv5 Scharr components must be same-shaped, non-empty BCHW tensors.")
        if gradient_x.device != gradient_y.device:
            raise ValueError("CSCEFv5 Scharr components must be on the same device.")
        with torch.autocast(device_type=gradient_x.device.type, enabled=False):
            gradient_x_fp32 = gradient_x.float()
            gradient_y_fp32 = gradient_y.float()
            jxx = F.avg_pool2d(gradient_x_fp32.square().mean(dim=1, keepdim=True), 3, stride=1, padding=1)
            jyy = F.avg_pool2d(gradient_y_fp32.square().mean(dim=1, keepdim=True), 3, stride=1, padding=1)
            jxy = F.avg_pool2d((gradient_x_fp32 * gradient_y_fp32).mean(dim=1, keepdim=True),
                             3, stride=1, padding=1)
            energy = jxx + jyy
            coherence_numerator = torch.sqrt(((jxx - jyy).square() + 4.0 * jxy.square()).clamp(min=0.0))
            coherence = (coherence_numerator / (energy + self.eps)).clamp(0.0, 1.0)
            energy_reference = energy.mean(dim=(-2, -1), keepdim=True).detach()
            reliability = (energy / (energy + energy_reference + self.eps)).clamp(0.0, 1.0)
            return coherence, reliability

    @torch.no_grad()
    def _compute_structure_confidence(self, gradient_x: torch.Tensor, gradient_y: torch.Tensor) -> torch.Tensor:
        """Return detached FP32 sqrt(coherence * reliability), exactly as in V4."""
        coherence, reliability = self._compute_structure_terms(gradient_x, gradient_y)
        return torch.sqrt((coherence * reliability).clamp(0.0, 1.0)).detach()

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return L + P_out(c * SiLU(GN(DW3(P_mix([l,s])))))."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CSCEFv5 expects [lateral, semantic].")
        lateral, semantic = inputs
        l, s = self._project_features(lateral, semantic)
        h = self.activation(self.content_norm(self.depthwise_conv(self.mix_projection(torch.cat((l, s), dim=1)))))
        gx, gy = self._compute_scharr_components(l)
        c = self._compute_structure_confidence(gx, gy).to(device=h.device, dtype=h.dtype)
        delta = self.output_projection(c * h)
        return lateral + delta.to(device=lateral.device, dtype=lateral.dtype)
