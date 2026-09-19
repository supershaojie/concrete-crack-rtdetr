"""Region-guided Detail Modulation (RDM), a single post-stage residual adapter.

Inspired by local/region modulation in SMFANet (ECCV 2024); this is the
experiment-specific difference-conditioned gate, not a renamed SMFA block.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .block import Blocks


class RDM(nn.Module):
    """Preserve spatial shape; support rectangular H/W >= 4, including remainders."""

    def __init__(self, channels=128, r=32):
        super().__init__()
        # Only new layers consume the isolated CPU RNG. Never reset CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.Wd = nn.Conv2d(channels, r, 1, bias=False)
            self.Dl = nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False)
            self.Dc = nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False)
            self.Pc = nn.Conv2d(r, r, 1, bias=False)
            self.Wg = nn.Conv2d(3 * r, r, 1, bias=True)
            self.Wo = nn.Conv2d(r, channels, 1, bias=False)
            for layer in (self.Wd, self.Dl, self.Dc, self.Pc, self.Wg):
                nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.zeros_(self.Wg.bias)
            nn.init.zeros_(self.Wo.weight)
        self.pool = nn.AvgPool2d(4, 4, 0, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        if min(x.shape[-2:]) < 4:
            raise ValueError("RDM requires H and W >= 4 for fixed 4x4 region pooling")
        z = self.Wd(x)
        local = F.silu(self.Dl(z))
        region = F.silu(self.Pc(self.Dc(self.pool(z))))
        region = F.interpolate(region, size=z.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.Wg(torch.cat((z, region, torch.abs(z - region)), dim=1)))
        return x + self.Wo(gate * local)


class RDMBlocks(Blocks):
    """Retain original .blocks keys and apply RDM after the complete stage output."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", r=32, variant="d"):
        super().__init__(ch_in, ch_out, block, count, stage_num, act, variant)
        self.rdm = RDM(ch_out * block.expansion, r)

    def forward(self, x):
        return self.rdm(super().forward(x))
