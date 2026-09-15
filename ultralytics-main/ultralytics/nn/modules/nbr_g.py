# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Narrow BasicBlock and input-guided grouped residual spatial calibration v1."""
import torch
from torch import nn

from .block import BasicBlock, Blocks, ConvNormLayer, _get_activation


class NarrowBasicBlock(nn.Module):
    """Two dense 3x3 convolutions, C -> hidden -> C, with identity shortcut."""

    expansion = 1

    def __init__(self, channels, hidden, act="relu"):
        super().__init__()
        if not 0 < hidden <= channels or act != "relu":
            raise ValueError("NBR v1 requires 0 < hidden <= channels and ReLU")
        self.shortcut = True
        self.branch2a = ConvNormLayer(channels, hidden, 3, 1, act=act)
        self.branch2b = ConvNormLayer(hidden, channels, 3, 1, act=None)
        self.act = _get_activation(act)

    def forward(self, x):
        return self.act(x + self.branch2b(self.branch2a(x)))


class InputGuidedGroupGate(nn.Module):
    """Eight independent spatial gates guided by input and residual magnitudes."""

    def __init__(self, channels, groups=8):
        super().__init__()
        if groups != 8 or channels % groups:
            raise ValueError("NBR-G v1 uses eight contiguous, equally sized groups")
        self.channels, self.groups = channels, groups
        # Do not change subsequent public-layer initialization, even standalone.
        with torch.random.fork_rng(devices=[]):
            self.gate_conv = nn.Conv2d(32, 8, 3, 1, 1, groups=8, bias=True)
            nn.init.zeros_(self.gate_conv.weight)
            nn.init.zeros_(self.gate_conv.bias)

    def descriptors(self, x, residual):
        if x.ndim != 4 or x.shape != residual.shape or x.shape[1] != self.channels:
            raise ValueError("Gate expects matching BCHW input/residual")
        b, c, h, w = x.shape
        xg = x.reshape(b, 8, c // 8, h, w).abs()
        fg = residual.reshape(b, 8, c // 8, h, w).abs()
        return torch.stack((xg.mean(2), xg.amax(2), fg.mean(2), fg.amax(2)), dim=2).reshape(b, 32, h, w)

    def forward(self, x, residual):
        return 1.0 + 0.5 * torch.tanh(self.gate_conv(self.descriptors(x, residual)))


class NBRGBlock(NarrowBasicBlock):
    """NBR with a bounded group gate; constructor-only zero initialization."""

    def __init__(self, channels, hidden, act="relu"):
        super().__init__(channels, hidden, act)
        self.gate = InputGuidedGroupGate(channels)

    def forward(self, x):
        residual = self.branch2b(self.branch2a(x))
        b, c, h, w = residual.shape
        gate = self.gate(x, residual)
        calibrated = (residual.reshape(b, 8, c // 8, h, w) * gate.unsqueeze(2)).reshape(b, c, h, w)
        return self.act(x + calibrated)


class NBRStage(Blocks):
    """Preserve first BasicBlock and original constructor RNG; replace second only."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", hidden=192, gated=True):
        if block is not BasicBlock or count != 2 or (stage_num, ch_out, hidden) not in ((4, 256, 192), (5, 512, 256)):
            raise ValueError("NBR v1 only supports the fixed S4/S5 second-block widths")
        if type(gated) is not bool:
            raise ValueError("gated must be an explicit boolean")
        # Consume exactly the original wide-stage constructor RNG, including block 1.
        super().__init__(ch_in, ch_out, block, count, stage_num, act)
        with torch.random.fork_rng(devices=[]):
            self.blocks[1] = (NBRGBlock if gated else NarrowBasicBlock)(ch_out, hidden, act)
