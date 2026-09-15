# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SFR-D v1: spatial factors with optional latent directional calibration."""
import torch
from torch import nn

from .block import BasicBlock, Blocks


def validate_source(block, channels):
    """Reject blocks for which the fixed spatial SVD/padding identity is invalid."""
    if type(block) is not BasicBlock or not block.shortcut:
        raise ValueError("SFR requires the identity-shortcut second BasicBlock")
    conv = block.branch2b.conv
    if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, conv.bias, conv.padding_mode) != (
            channels, channels, (3, 3), (1, 1), (1, 1), (1, 1), 1, None, "zeros"):
        raise ValueError("SFR requires unfused dense bias-free k3/s1/p1/d1 zero-padded W2")
    first = block.branch2a.conv
    if first.in_channels != channels or first.out_channels != channels or first.stride != (1, 1):
        raise ValueError("SFR must preserve the complete stride-one C-to-C first convolution")
    if not isinstance(block.branch2b.act, nn.Identity) or not isinstance(block.act, nn.ReLU):
        raise ValueError("SFR requires original BN2 without activation and final ReLU")


class DirectionalCalibration(nn.Module):
    """Elementwise gain 1 + .5*tanh(dw_h(z)+dw_v(z)); zero only at construction."""

    def __init__(self, rank):
        super().__init__()
        self.dw_h = nn.Conv2d(rank, rank, (1, 5), padding=(0, 2), groups=rank, bias=False)
        self.dw_v = nn.Conv2d(rank, rank, (5, 1), padding=(2, 0), groups=rank, bias=True)
        nn.init.zeros_(self.dw_h.weight)
        nn.init.zeros_(self.dw_v.weight)
        nn.init.zeros_(self.dw_v.bias)

    def gain(self, z):
        return 1.0 + 0.5 * torch.tanh(self.dw_h(z) + self.dw_v(z))

    def forward(self, z):
        return z * self.gain(z)


class SpatialFactorConv(nn.Module):
    """Replace only W2; retain the original BN2 object and its state/metadata."""

    def __init__(self, channels, rank, norm, directional=True):
        super().__init__()
        self.A_1x3 = nn.Conv2d(channels, rank, (1, 3), padding=(0, 1), bias=False)
        self.B_3x1 = nn.Conv2d(rank, channels, (3, 1), padding=(1, 0), bias=False)
        if directional:
            self.gate = DirectionalCalibration(rank)
        self.norm = norm

    def forward(self, x):
        z = self.A_1x3(x)
        if hasattr(self, "gate"):
            z = self.gate(z)
        q = self.B_3x1(z)
        return self.norm(q) if hasattr(self, "norm") else q

    def fuse(self):
        """Fold only B + BN2 in eval; the input-dependent gate remains executable."""
        if self.training:
            raise RuntimeError("SFR fusion requires eval mode")
        if hasattr(self, "norm"):
            # Native helper constructs a FP32 zero bias even for FP64 inputs.
            # PyTorch's eval helper preserves dtype and folds exactly this pair.
            from torch.nn.utils.fusion import fuse_conv_bn_eval
            self.B_3x1 = fuse_conv_bn_eval(self.B_3x1, self.norm)
            del self.norm
        return self


class SFRStage(Blocks):
    """Keep .blocks and both BasicBlocks; replace branch2b of index 1 only."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", rank=None, directional=True):
        if block is not BasicBlock or count != 2 or (stage_num, ch_out, rank) not in (
            (4, 256, 256), (5, 512, 384)
        ):
            raise ValueError("SFR-D v1 fixes S4/r256 and S5/r384, two BasicBlocks per stage")
        # Consume exactly the original stage's RNG, including W2, before replacing it.
        super().__init__(ch_in, ch_out, block, count, stage_num, act)
        original = self.blocks[1]
        validate_source(original, ch_out)
        with torch.random.fork_rng(devices=[]):
            original.branch2b = SpatialFactorConv(ch_out, rank, original.branch2b.norm, directional)
