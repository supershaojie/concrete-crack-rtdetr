# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Lifting Innovation Folding Downsampling v1; C2 main path plus pre-BN residual."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class LIFDown(Conv):
    """Keep Conv state names and RNG consumption; only the new output projection starts at zero."""

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        if (self.conv.kernel_size, self.conv.stride, self.conv.padding, self.conv.dilation) != (
            (3, 3), (2, 2), (1, 1), (1, 1)
        ):
            raise ValueError("LIF-Down v1 alignment requires k3/s2/p1/d1")
        r = 16
        # Local CPU RNG scope: later public layers see exactly the original Conv RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.B_proj = nn.Conv2d(c1, r, 1, bias=False)
            self.P = nn.Conv2d(r, 3 * r, 3, groups=r, bias=False)
            self.U_mix = nn.Conv2d(3 * r, r, 1, groups=r, bias=False)
            self.U_dw = nn.Conv2d(r, r, 3, groups=r, bias=False)
            self.O_proj = nn.Conv2d(4 * r, c2, 1, bias=False)
            for layer in (self.B_proj, self.P, self.U_mix, self.U_dw):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            nn.init.zeros_(self.O_proj.weight)

    @staticmethod
    def haar(z):
        """Orthonormal /2 Haar; adjacent (Dh_i,Dv_i,Dd_i) for each grouped channel."""
        h, w = z.shape[-2:]
        if h % 2 or w % 2:
            z = F.pad(z, (0, w % 2, 0, h % 2), mode="replicate")
        a, b = z[:, :, 0::2, 0::2], z[:, :, 0::2, 1::2]
        c, d = z[:, :, 1::2, 0::2], z[:, :, 1::2, 1::2]
        low = (a + b + c + d) / 2
        dh, dv, dd = (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2
        return low, torch.stack((dh, dv, dd), dim=2).flatten(1, 2)

    @staticmethod
    def align(r):
        """Separable previous/current interpolation, with replicated left/top boundaries."""
        rx = .25 * F.pad(r, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1] + .75 * r
        return .25 * F.pad(rx, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :] + .75 * rx

    def residual(self, x):
        a, d = self.haar(self.B_proj(x))
        e = d - F.gelu(self.P(F.pad(a, (1, 1, 1, 1), mode="replicate")))
        update = self.U_dw(F.pad(F.gelu(self.U_mix(e)), (1, 1, 1, 1), mode="replicate"))
        return self.align(self.O_proj(F.gelu(torch.cat((a + update, e), dim=1))))

    def forward(self, x):
        base, residual = self.conv(x), self.residual(x)
        if base.shape != residual.shape:
            raise RuntimeError(f"LIF/base shape mismatch: {residual.shape} vs {base.shape}")
        return self.act(self.bn(base + residual))

    def forward_fuse(self, x):
        # This pre-BN sum must retain its BN; BaseModel.fuse explicitly preserves this module.
        return self.forward(x)
