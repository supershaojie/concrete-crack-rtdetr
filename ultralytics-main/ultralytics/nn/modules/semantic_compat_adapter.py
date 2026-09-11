# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Channel-only calibration of the existing CSCEF semantic input."""
import torch
from torch import nn

__all__ = ("SemanticCompatibilityAdapter", "SCIAdapter")


class SemanticCompatibilityAdapter(nn.Module):
    """Preserve the identity gradient; learn a residual from a detached input.

    This module belongs only to the CSCEF semantic side branch. It never replaces
    the Y4 tensor consumed by the original Concat/PAN operations.
    """

    def __init__(self, channels=256, hidden=32):
        super().__init__()
        if channels != 256 or hidden != 32:
            raise ValueError("SCI uses the fixed 256 -> 32 -> 256 preset.")
        # Like the original innovation modules, preserve public constructor RNG.
        with torch.random.fork_rng(devices=[]):
            self.norm = nn.GroupNorm(8, channels, eps=1e-5, affine=True)
            self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
            self.activation = nn.SiLU()
            self.restore = nn.Conv2d(hidden, channels, 1, bias=True)
            nn.init.zeros_(self.restore.weight)
            nn.init.zeros_(self.restore.bias)

    def residual(self, y4):
        """The correction branch cannot send an additional gradient into Y4."""
        return self.restore(self.activation(self.reduce(self.norm(y4.detach())))).to(y4.dtype)

    def forward(self, y4):
        return y4 + self.residual(y4)


SCIAdapter = SemanticCompatibilityAdapter
