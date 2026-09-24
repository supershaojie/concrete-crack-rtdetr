# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""LCD v1: one shared spatial kernel, signed local/context difference modulation."""
import torch
from torch import nn
from torch.nn import functional as F

from .block import Blocks


class LCD(nn.Module):
    """128 -> 32 -> 128 identity-initialized residual, at unchanged feature coordinates."""

    def __init__(self, channels=128, rank=32, eps=1e-6, seed=424002):
        super().__init__()
        if (channels, rank, eps, seed) != (128, 32, 1e-6, 424002):
            raise ValueError("LCD v1 has a fixed C/r/eps/seed contract")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.P = nn.Conv2d(channels, rank, 1, bias=False)
            self.norm = nn.LayerNorm(rank, eps=eps)
            self.shared_dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
            self.G = nn.Conv2d(rank, rank, 1, bias=False)
            self.O = nn.Conv2d(rank, channels, 1, bias=False)
            nn.init.zeros_(self.O.weight)
        # Opt-in small scalar diagnostics; no activation cache or persistent state keys.
        self.collect_diagnostics = False
        self.diagnostics = {}

    def components(self, x):
        u = self.norm(self.P(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        a = self.shared_dw(u)
        b = F.conv2d(u, self.shared_dw.weight, padding=2, dilation=2, groups=32)
        t = torch.tanh(self.G(b - a))
        return self.O(F.silu(a) * t), t

    def forward(self, x):
        r, t = self.components(x)
        if self.collect_diagnostics:
            with torch.no_grad():
                self.diagnostics = {
                    "residual_rms_ratio": float(r.detach().float().square().mean().sqrt() /
                                                (x.detach().float().square().mean().sqrt() + 1e-6)),
                    "tanh_saturation": float((t.detach().abs() > .95).float().mean()),
                    "O_norm": float(self.O.weight.detach().float().norm()),
                }
        return x + r


class BlocksLCD(Blocks):
    """Preserve Blocks.blocks public keys and append LCD after the complete S3 stage."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        if ch_out * block.expansion != 128 or stage_num != 3:
            raise ValueError("BlocksLCD is restricted to the original backbone S3/8")
        super().__init__(ch_in, ch_out, block, count, stage_num, act, variant)
        self.lcd = LCD()

    def forward(self, x):
        return self.lcd(super().forward(x))
