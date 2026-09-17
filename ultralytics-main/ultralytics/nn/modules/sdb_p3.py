# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SDB-P3 v1: semantic-guided P2 detail bypass after the original P3 RepC3."""

import torch
from torch import nn
from torch.nn import functional as F

from .block import RepC3


class SDBP3(nn.Module):
    """Add gated, phase-major P2 detail to an unchanged P3 semantic feature."""

    def __init__(self, p2_channels=64, p3_channels=256, detail_channels=32):
        super().__init__()
        if detail_channels <= 0 or detail_channels % 4:
            raise ValueError("SDB-P3 detail_channels must be positive and divisible by 4")
        self.p2_channels = p2_channels
        self.p3_channels = p3_channels
        self.detail_channels = detail_channels
        # New tensors are CPU tensors. Restore the caller's RNG, including the
        # state used to construct all later original neck/decoder parameters.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.W_d = nn.Conv2d(4 * p2_channels, detail_channels, 1, bias=False)
            self.DW3 = nn.Conv2d(detail_channels, detail_channels, 3, padding=1,
                                 groups=detail_channels, bias=False)
            self.GN_D = nn.GroupNorm(4, detail_channels, eps=1e-5, affine=True)
            self.W_q = nn.Conv2d(p3_channels, detail_channels, 1, bias=False)
            self.GN_Q = nn.GroupNorm(4, detail_channels, eps=1e-5, affine=True)
            self.W_g = nn.Conv2d(3 * detail_channels, detail_channels, 1, bias=True)
            self.W_o = nn.Conv2d(detail_channels, p3_channels, 1, bias=False)
            for layer in (self.W_d, self.DW3, self.W_q, self.W_g):
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(self.W_g.bias)
            nn.init.zeros_(self.W_o.weight)
            for norm in (self.GN_D, self.GN_Q):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

    @staticmethod
    def phase_rearrange(p2):
        """Return phase-major 00,10,01,11, replicating only odd right/bottom edges."""
        if p2.ndim != 4:
            raise ValueError(f"SDB-P3 P2 must be BCHW, got {tuple(p2.shape)}")
        height, width = p2.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("SDB-P3 P2 spatial dimensions must be nonempty")
        if height % 2 or width % 2:
            p2 = F.pad(p2, (0, width % 2, 0, height % 2), mode="replicate")
        return torch.cat((p2[..., 0::2, 0::2], p2[..., 1::2, 0::2],
                          p2[..., 0::2, 1::2], p2[..., 1::2, 1::2]), dim=1)

    def forward(self, p2, original_p3):
        """Keep the original semantic path and add only the learned detail residual."""
        if p2.ndim != 4 or original_p3.ndim != 4:
            raise ValueError("SDB-P3 expects BCHW P2 and original P3 inputs")
        if p2.shape[0] != original_p3.shape[0]:
            raise ValueError("SDB-P3 P2 and P3 batch sizes must agree")
        if p2.shape[1] != self.p2_channels or original_p3.shape[1] != self.p3_channels:
            raise ValueError(f"SDB-P3 expected P2/P3 channels {self.p2_channels}/{self.p3_channels}, "
                             f"got {p2.shape[1]}/{original_p3.shape[1]}")
        expected = tuple((size + 1) // 2 for size in p2.shape[-2:])
        if tuple(original_p3.shape[-2:]) != expected:
            raise ValueError(f"SDB-P3 requires P3 spatial size ceil(P2/2)={expected}; "
                             f"P2={tuple(p2.shape[-2:])}, P3={tuple(original_p3.shape[-2:])}")
        detail = F.silu(self.GN_D(self.DW3(self.W_d(self.phase_rearrange(p2)))))
        semantic = self.GN_Q(self.W_q(original_p3))
        gate = self.W_g(torch.cat((detail, semantic, detail * semantic), dim=1)).sigmoid()
        return original_p3 + self.W_o(gate * detail)


class SDBRepC3(RepC3):
    """Preserve RepC3 cv1/cv2/m/cv3 paths and consume [fusion_input, P2]."""

    def __init__(self, c1, c2, n=3, e=0.5, p2_channels=64, detail_channels=32):
        super().__init__(c1, c2, n, e)
        self.sdb = SDBP3(p2_channels, c2, detail_channels)

    def forward(self, inputs):
        fusion_input, p2 = inputs
        original_p3 = super().forward(fusion_input)
        return self.sdb(p2, original_p3)
