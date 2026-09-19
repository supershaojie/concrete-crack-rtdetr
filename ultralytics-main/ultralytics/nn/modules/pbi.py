# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""PBI-v1: fixed low-rank pointwise bilinear residual, independently implemented.

Elementwise multiplicative interaction is established prior work (e.g. StarNet,
arXiv:2403.19967). This module is not the StarNet Block; its specification is the
project's 256 -> 32 pointwise P3 adaptation. Accuracy benefit is unverified.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class PBI(nn.Module):
    """Y = X + Wo(W1(X) * W2(X)), with local FP32 product/projection/sum."""

    def __init__(self, channels=256, latent_channels=32):
        super().__init__()
        if (channels, latent_channels) != (256, 32):
            raise ValueError("PBI-v1 requires channels=256 and latent_channels=32")
        # Only new parameters consume this private CPU RNG sequence. Neither the
        # caller's CPU sequence nor any CUDA RNG is changed by construction.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.W1 = nn.Conv2d(channels, latent_channels, 1, bias=False)
            self.W2 = nn.Conv2d(channels, latent_channels, 1, bias=False)
            self.Wo = nn.Conv2d(latent_channels, channels, 1, bias=False)
            nn.init.xavier_uniform_(self.W1.weight, gain=1)
            nn.init.xavier_uniform_(self.W2.weight, gain=1)
            nn.init.zeros_(self.Wo.weight)

    def forward(self, x):
        # Input projections retain native outer autocast behavior. Final dtype
        # conversion can still overflow: this is not a finite-value guarantee.
        u, v = self.W1(x), self.W2(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = u.float() * v.float()
            delta = F.conv2d(z, self.Wo.weight.float())
            y = x.float() + delta
        return y.to(dtype=x.dtype)


class PBIConv(Conv):
    """Keep original Conv/BN/act names and semantics, then execute PBI once."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=False, latent_channels=32):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.pbi = PBI(c2, latent_channels)

    def forward(self, x):
        return self.pbi(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        # BaseModel.fuse binds this method: retain the learned residual after BN fusion.
        return self.pbi(self.act(self.conv(x)))
