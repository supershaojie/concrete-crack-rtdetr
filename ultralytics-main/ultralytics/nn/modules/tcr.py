# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""TCR v1: fixed integer-lattice three-band signed contrast residual."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class TCR(nn.Module):
    tangents = ((1, 0), (1, 1), (0, 1), (-1, 1))
    normals = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
    side_steps = (1, 2)
    margin = 4

    def __init__(self, channels=256):
        super().__init__()
        if channels != 256:
            raise ValueError("TCR v1 requires the 256-channel node-17 projection")
        # All additional tensors are CPU tensors. Preserve subsequent public RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.P = nn.Conv2d(channels, 16, 1, bias=False)
            self.O = nn.Conv2d(128, channels, 1, bias=False)
            nn.init.zeros_(self.O.weight)
        self.enabled = True
        self.capture = False
        self.last_stats = None

    @staticmethod
    def shift(z, dx, dy):
        """Output at (y,x) reads input at (y+dy,x+dx); never wrap edges."""
        h, w = z.shape[-2:]
        padded = F.pad(z, (4, 4, 4, 4))
        return padded[..., 4 + dy:4 + dy + h, 4 + dx:4 + dx + w]

    @staticmethod
    def signed_agreement(a, b):
        return F.relu(torch.minimum(a, b)) - F.relu(torch.minimum(-a, -b))

    @classmethod
    def responses(cls, z):
        # Strip accumulation, differences and minimum are FP32 even under AMP.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            h, w = z.shape[-2:]
            xx = torch.arange(w, device=z.device)
            yy = torch.arange(h, device=z.device)
            valid = ((yy >= 4) & (yy < h - 4))[:, None] & ((xx >= 4) & (xx < w - 4))[None, :]
            groups = []
            for (tx, ty), (nx, ny) in zip(cls.tangents, cls.normals):
                center = sum(cls.shift(z, r * tx, r * ty) for r in (-2, -1, 0, 1, 2)) / 5.0
                for step in cls.side_steps:
                    plus = cls.shift(center, step * nx, step * ny)
                    minus = cls.shift(center, -step * nx, -step * ny)
                    # where, not multiplication: padded values never enter O.
                    groups.append(torch.where(valid, cls.signed_agreement(center - plus, center - minus), 0.0))
            return torch.cat(groups, dim=1)

    @staticmethod
    def statistics(z, r, residual, x):
        with torch.no_grad():
            interior = r[..., 4:-4, 4:-4]
            groups = interior.reshape(r.shape[0], 8, 16, *interior.shape[-2:])
            records = []
            for i in range(8):
                values = groups[:, i].float()
                count = values.numel()
                records.append(dict(direction=(0, 45, 90, 135)[i // 2], side_step=i % 2 + 1,
                                    elements=count, nonzero=int(torch.count_nonzero(values)),
                                    positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                                    sum_squares=float(values.square().sum()),
                                    rms=float(values.square().mean().sqrt()) if count else 0.0))
                for key in ("nonzero", "positive", "negative"):
                    records[-1][key + "_fraction"] = records[-1][key] / count if count else None
            return dict(groups=records, z_rms=float(z.float().square().mean().sqrt()),
                        z_elements=z.numel(), z_nonzero=int(torch.count_nonzero(z)),
                        z_positive=int((z>0).sum()), z_negative=int((z<0).sum()),
                        residual_ratio=float(residual.float().norm() / (x.float().norm() + 1e-12)),
                        input_dtype=str(x.dtype), response_dtype=str(r.dtype), shape=list(x.shape))

    def forward(self, x):
        if not self.enabled:
            return x
        z = self.P(x)
        r = self.responses(z)
        # Cast supports explicit model.half(); autocast still controls the convolution.
        residual = self.O(r.to(self.O.weight.dtype)).to(x.dtype)
        if self.capture and not torch.jit.is_tracing():
            self.last_stats = self.statistics(z, r, residual, x)
        return x + residual


class ConvTCR(Conv):
    """Original Conv parameters and BN fusion, with TCR after conv/bn/act."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.tcr = TCR(c2)

    def forward(self, x):
        return self.tcr(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.tcr(self.act(self.conv(x)))
