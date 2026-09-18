# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Guided Residual Alignment: bounded grouped sampling difference at Y4/P3 fusion.

Background: DySample (arXiv:2308.15085), FADE (arXiv:2207.10392).
The nearest-preserving double-sampling residual below is this experiment's
specified design, not a claim of priority or a formula attributed to those papers.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GRAConcat(nn.Module):
    """Return ``cat(U + alpha * (S(H, G0 + delta) - S(H, G0)), L)``.

    Offsets are interleaved (g0_dx, g0_dy, g1_dx, ...) in source-pixel units.
    Only the offset head is zero initialized. CPU RNG isolation leaves all later
    public model layers and CUDA RNG untouched; no persistent grid is cached.
    """

    def __init__(self, c_high, c_up, c_lateral, hidden_channels=16, groups=4, max_offset=0.25, alpha=0.5):
        super().__init__()
        if c_high != c_up or groups <= 0 or c_high % groups:
            raise ValueError("GRA requires c_high == c_up and c_high divisible by positive groups")
        if min(c_high, c_lateral, hidden_channels) <= 0:
            raise ValueError("GRA channel counts must be positive")
        if (hidden_channels, groups, max_offset, alpha) != (16, 4, 0.25, 0.5):
            raise ValueError("GRA v1 fixes hidden_channels=16, groups=4, max_offset=0.25, alpha=0.5")
        self.c_high, self.c_up, self.c_lateral = c_high, c_up, c_lateral
        self.hidden_channels, self.groups = hidden_channels, groups
        self.max_offset, self.alpha = float(max_offset), float(alpha)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.high_proj = nn.Conv2d(c_high, hidden_channels, 1, bias=False)
            self.lateral_proj = nn.Conv2d(c_lateral, hidden_channels, 1, bias=False)
            self.depthwise = nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3,
                                       padding=1, groups=2 * hidden_channels, bias=False)
            self.offset = nn.Conv2d(2 * hidden_channels, 2 * groups, 1, bias=True)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)

    def offset_logits(self, inputs):
        """Predict unbounded interleaved offsets; descriptor convolutions follow autocast."""
        high, _, lateral = inputs
        high_description = F.interpolate(F.silu(self.high_proj(high)), size=lateral.shape[-2:], mode="nearest")
        lateral_description = F.silu(self.lateral_proj(lateral))
        return self.offset(F.silu(self.depthwise(torch.cat((high_description, lateral_description), dim=1))))

    def forward(self, inputs):
        high, up, lateral = inputs
        if high.ndim != 4 or up.ndim != 4 or lateral.ndim != 4:
            raise ValueError("GRA inputs must be NCHW tensors")
        if not (high.shape[0] == up.shape[0] == lateral.shape[0]):
            raise ValueError("GRA input batch sizes differ")
        if (high.shape[1], up.shape[1], lateral.shape[1]) != (self.c_high, self.c_up, self.c_lateral):
            raise ValueError("GRA input channels do not match construction")
        if up.shape[-2:] != lateral.shape[-2:]:
            raise ValueError("GRA U and L spatial shapes must agree")
        if not (high.device == up.device == lateral.device):
            raise ValueError("GRA inputs must share one device")
        raw_offsets = self.offset_logits(inputs)
        batch, channels, source_height, source_width = high.shape
        target_height, target_width = up.shape[-2:]
        # Torch 2.1-compatible native autocast. Sampling and cancellation always
        # run in FP32, including when the entire model was explicitly .half().
        with torch.autocast(device_type=high.device.type, enabled=False):
            delta = self.max_offset * raw_offsets.float().tanh()
            delta = delta.reshape(batch, self.groups, 2, target_height, target_width)
            delta = delta.permute(0, 1, 3, 4, 2).reshape(batch * self.groups, target_height, target_width, 2)
            ys = (torch.arange(target_height, device=high.device, dtype=torch.float32) + 0.5) * (2.0 / target_height) - 1
            xs = (torch.arange(target_width, device=high.device, dtype=torch.float32) + 0.5) * (2.0 / target_width) - 1
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid0 = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch * self.groups, -1, -1, -1)
            scale = delta.new_tensor((2.0 / source_width, 2.0 / source_height))
            grid = grid0 + delta * scale
            source = high.float().reshape(batch * self.groups, channels // self.groups, source_height, source_width)
            shifted = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
            reference = F.grid_sample(source, grid0, mode="bilinear", padding_mode="border", align_corners=False)
            residual = (shifted - reference).reshape(batch, channels, target_height, target_width)
            corrected = (up.float() + self.alpha * residual).to(dtype=up.dtype)
        return torch.cat((corrected, lateral), dim=1)
