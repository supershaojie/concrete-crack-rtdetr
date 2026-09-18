# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Differential Channel Calibration after the original P3 projection.

The image-dependent channel relation is shared over space, so this is not a
strictly local operator. Its FP32 differential branch has zero spatial mean in
real arithmetic; that property is not a claim about background suppression.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv

__all__ = ("DCC", "DCCConv")


class DCC(nn.Module):
    """Fixed 32-channel, four-head differential channel calibration.

    The four bias-free projections add exactly 18,432 trainable parameters for
    256 input channels. Construction preserves the caller's CPU RNG state; the
    default Conv2d initialization of W_d/W_q/W_k is nonzero, and only W_o is
    zeroed. No forward/load/fuse/EMA operation reinitializes these parameters.
    """

    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        self.latent_channels = 32
        self.heads = 4
        self.head_dim = 8
        self.temperature = 1.0
        # Isolate only CPU initialization. In particular, do not seed CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            self.W_d = nn.Conv2d(channels, 32, 1, bias=False)
            self.W_q = nn.Conv2d(32, 32, 1, bias=False)
            self.W_k = nn.Conv2d(32, 32, 1, bias=False)
            self.W_o = nn.Conv2d(32, channels, 1, bias=False)
            nn.init.zeros_(self.W_o.weight)
        self.register_buffer("identity", torch.eye(8).view(1, 1, 8, 8), persistent=False)

    def _delta_and_attention(self, x):
        """Return FP32 delta and actual B x 4 x 8 x 8 channel attention.

        This small helper also permits direct mathematical diagnostics without
        caching activations or changing the normal model forward interface.
        Weight casts remain differentiable, including after model.half().
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            b, _, height, width = x.shape
            z = F.conv2d(x.float(), self.W_d.weight.float())
            zc = z - z.mean(dim=(-2, -1), keepdim=True)
            q = F.conv2d(zc, self.W_q.weight.float()).reshape(b, 4, 8, height * width)
            k = F.conv2d(zc, self.W_k.weight.float()).reshape(b, 4, 8, height * width)
            v = zc.reshape(b, 4, 8, height * width)
            qn = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
            kn = F.normalize(k, p=2.0, dim=-1, eps=1e-6)
            attention = torch.softmax(torch.matmul(qn, kn.transpose(-2, -1)) / self.temperature, dim=-1)
            differential = torch.matmul(attention - self.identity.float(), v)
            delta = F.conv2d(differential.reshape(b, 32, height, width), self.W_o.weight.float())
        return delta, attention

    def forward(self, x):
        delta, _ = self._delta_and_attention(x)
        return x + delta.to(dtype=x.dtype)


class DCCConv(Conv):
    """Original Conv/BN/act followed once by DCC, retaining public state keys."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.dcc = DCC(c2)

    def forward(self, x):
        return self.dcc(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        # BaseModel.fuse binds this subclass method after original Conv-BN fuse.
        return self.dcc(self.act(self.conv(x)))
