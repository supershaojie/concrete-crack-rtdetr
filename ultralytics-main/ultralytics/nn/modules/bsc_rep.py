# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Bilateral Context Support RepC3 v1: a shared gate on paired P3 context."""

import torch
from torch import nn
from torch.nn import functional as F

from .block import RepC3


class BilateralContextSupport(nn.Module):
    """Four bilateral, radius-3 contexts in a fixed 32-channel FP32 branch.

    The learned gate evaluates [center, bilateral mean, bilateral difference].
    It has no monotonicity constraint with respect to endpoint agreement.
    Only the output projection starts at zero; it is never reset on forward/load.
    """

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))  # (dy, dx)
    radius = 3

    def __init__(self, channels: int):
        super().__init__()
        # CPU-only construction and an isolated stream preserve all subsequent
        # parent parameters/buffers and give both experiment variants the same BSC.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.in_proj = nn.Conv2d(channels, 32, 1, bias=False, device="cpu")
            self.gate = nn.Conv2d(96, 32, 1, bias=True, device="cpu")
            self.out_proj = nn.Conv2d(32, channels, 1, bias=False, device="cpu")
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _conv_fp32(layer: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        # Keep ordinary module calls for FP32/native AMP so standard Conv2d THOP
        # hooks count each of the four calls to the shared gate. Saved half models
        # need differentiable casts, without mutating Parameter identity or dtype.
        if layer.weight.dtype == torch.float32 and (layer.bias is None or layer.bias.dtype == torch.float32):
            return layer(x)
        return F.conv2d(x, layer.weight.float(), None if layer.bias is None else layer.bias.float(),
                        layer.stride, layer.padding, layer.dilation, layer.groups)

    @classmethod
    def _contexts(cls, padded: torch.Tensor, height: int, width: int, dy: int, dx: int):
        """Return means at p +/- k*(dy,dx), k=1,2,3, with no wraparound."""
        p = cls.radius
        plus = minus = None
        for k in (1, 2, 3):
            yp, xp = p + k * dy, p + k * dx
            ym, xm = p - k * dy, p - k * dx
            zp = padded[:, :, yp:yp + height, xp:xp + width]
            zm = padded[:, :, ym:ym + height, xm:xm + width]
            plus = zp if plus is None else plus + zp
            minus = zm if minus is None else minus + zm
        return plus / 3, minus / 3

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=t.device.type, enabled=False):
            z = self._conv_fp32(self.in_proj, t.float())
            height, width = z.shape[-2:]
            padded = F.pad(z, (3, 3, 3, 3), mode="replicate")
            residual = None
            for dy, dx in self.directions:
                plus, minus = self._contexts(padded, height, width, dy, dx)
                mean = (plus + minus) / 2
                difference = (plus - minus).abs()
                gate = self._conv_fp32(self.gate, torch.cat((z, mean, difference), dim=1)).sigmoid()
                current = gate * (mean - z)
                residual = current if residual is None else residual + current
            delta = self._conv_fp32(self.out_proj, residual / 4)
        return delta.to(dtype=t.dtype)


class BSCRepC3(RepC3):
    """Original RepC3 state paths and computation plus one hidden-space branch."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        super().__init__(c1, c2, n, e)
        self.bsc = BilateralContextSupport(int(c2 * e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.m(self.cv1(x))
        return self.cv3((t + self.bsc(t)) + self.cv2(x))
