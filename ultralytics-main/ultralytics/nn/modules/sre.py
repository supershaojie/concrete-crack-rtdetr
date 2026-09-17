# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Support-based Response Enrichment: one synchronous, valid-neighbor update.

The nonnegative latent values are features, not foreground probabilities. The
signed output projection does not promise monotonic scores or improved recall.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .block import RepC3


class SRE(nn.Module):
    """The fixed 256 -> 32 -> 8 descriptor / 32 -> 256 residual contract."""

    OFFSETS = tuple((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0))
    INIT_SEED = 42

    def __init__(self, channels=256, latent_channels=32, descriptor_channels=8, window_size=5):
        super().__init__()
        if (channels, latent_channels, descriptor_channels, window_size) != (256, 32, 8, 5):
            raise ValueError("SRE v1 fixes channels=256, latent_channels=32, descriptor_channels=8, window_size=5")
        # Only the CPU generator is seeded. No CUDA RNG state is modified; the
        # parent's subsequent initializations retain their exact random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.INIT_SEED)
            self.W_d = nn.Conv2d(channels, latent_channels, 1, bias=False)
            self.GN = nn.GroupNorm(4, latent_channels, eps=1e-5, affine=True)
            self.W_q = nn.Conv2d(latent_channels, descriptor_channels, 1, bias=False)
            self.W_o = nn.Conv2d(latent_channels, channels, 1, bias=False)
            nn.init.xavier_uniform_(self.W_d.weight)
            nn.init.xavier_uniform_(self.W_q.weight)
            nn.init.ones_(self.GN.weight)
            nn.init.zeros_(self.GN.bias)
            nn.init.zeros_(self.W_o.weight)

    @staticmethod
    def aggregate(q, v):
        """Compute D from original Q/V using valid overlapping slices only.

        Q is already normalized along its descriptor channel dimension. Every
        offset reads the same unmodified Q/V. Padding here pads *messages with
        zero*, never features with repeated / reflected / wrapped neighbors.
        This avoids a full unfolded neighbor tensor but autograd still saves
        per-offset intermediates; it is not a claim of constant training memory.
        """
        h, w = v.shape[-2:]
        numerator = v * 0.0
        support = v[:, :1] * 0.0
        for dy, dx in SRE.OFFSETS:
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            if y1 <= y0 or x1 <= x0:
                continue
            qp = q[..., y0:y1, x0:x1]
            qj = q[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            vp = v[..., y0:y1, x0:x1]
            vj = v[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            weight = (qp * qj).sum(dim=1, keepdim=True).clamp(-1.0, 1.0).relu().square()
            pad = (x0, w - x1, y0, h - y1)
            numerator = numerator + F.pad(weight * (vj - vp).relu(), pad)
            support = support + F.pad(weight, pad)
        return numerator / (1.0 + support)

    def forward(self, x):
        # Functional ops with differentiable casts also support explicitly half
        # parameters, without replacing Parameters or mutating module precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.conv2d(x.float(), self.W_d.weight.float())
            z = F.group_norm(z, self.GN.num_groups, self.GN.weight.float(), self.GN.bias.float(), self.GN.eps)
            q = F.normalize(F.conv2d(z, self.W_q.weight.float()), p=2, dim=1, eps=1e-6)
            v = F.softplus(z, beta=1, threshold=20)
            delta = F.conv2d(self.aggregate(q, v), self.W_o.weight.float())
        return x + delta.to(dtype=x.dtype)


class SRERepC3(RepC3):
    """Keep all parent RepC3 state keys and enrich its complete output once."""

    def __init__(self, c1, c2, n=3, e=0.5, latent_channels=32, descriptor_channels=8, window_size=5):
        super().__init__(c1, c2, n, e)
        self.sre = SRE(c2, latent_channels, descriptor_channels, window_size)

    def forward(self, x):
        return self.sre(super().forward(x))


def count_sre_projection_macs_partial(module, inputs, output):
    """Optional THOP hook: projection MACs only, explicitly PARTIAL.

    Functional convolutions bypass child Conv2d hooks; account for them once
    here. Neighborhood arithmetic, GN, normalization and activations are not
    MAC-equivalent and must be reported separately, never as full model FLOPs.
    Compatible with THOP versions storing total_ops as either int or Tensor.
    """
    b, _, h, w = inputs[0].shape
    macs = b * h * w * (256 * 32 + 32 * 8 + 32 * 256)
    previous = getattr(module, "total_ops", 0)
    if isinstance(previous, torch.Tensor):
        module.total_ops = previous + torch.as_tensor(macs, device=previous.device, dtype=previous.dtype)
    else:
        module.total_ops = previous + macs
