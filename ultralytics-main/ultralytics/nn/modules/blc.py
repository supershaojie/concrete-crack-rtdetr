"""Bilateral Line Contrast: fixed integer line contrasts, spatial nine-way selection.

The ninth response is exactly zero. No normalization/activation follows the
signed reduction or the output residual. This branch cannot be fused away.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .block import Blocks


class BLC(nn.Module):
    """The fixed 128 -> 32 -> 128 BLC contract (8,561 parameters)."""

    DIRECTIONS = (((0, 1), (1, 0)), ((1, 0), (0, 1)),
                  ((1, 1), (1, -1)), ((1, -1), (1, 1)))

    def __init__(self, channels=128, r=32):
        super().__init__()
        if (channels, r) != (128, 32):
            raise ValueError("BLC v1 requires channels=128 and r=32")
        # Only new layer construction/initialization is isolated. The fixed seed
        # gives both variants equal new values, without sharing any storage.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.Wd = nn.Conv2d(channels, r, 1, bias=False)
            self.Wg = nn.Conv2d(r + 8, 9, 1, bias=True)
            self.Wo = nn.Conv2d(r, channels, 1, bias=False)
            nn.init.xavier_uniform_(self.Wd.weight, gain=1)
            nn.init.zeros_(self.Wg.weight)
            nn.init.zeros_(self.Wg.bias)
            nn.init.zeros_(self.Wo.weight)

    @staticmethod
    def responses(z):
        """Eight signed responses; clamp each final composite coordinate."""
        h, w = z.shape[-2:]
        padded = F.pad(z, (3, 3, 3, 3), mode="replicate")

        def sample(dy, dx):
            return padded[..., 3 + dy:3 + dy + h, 3 + dx:3 + dx + w]

        result = []
        for (ty, tx), (ny, nx) in BLC.DIRECTIONS:
            center = (sample(-ty, -tx) + z + sample(ty, tx)) / 3
            for s in (1, 2):
                py, px = s * ny, s * nx
                plus = (sample(-ty + py, -tx + px) + sample(py, px)
                        + sample(ty + py, tx + px)) / 3
                minus = (sample(-ty - py, -tx - px) + sample(-py, -px)
                         + sample(ty - py, tx - px)) / 3
                dp, dm = center - plus, center - minus
                result.append(torch.minimum(F.relu(dp), F.relu(dm))
                              - torch.minimum(F.relu(-dp), F.relu(-dm)))
        return result

    def forward(self, x):
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.to(dtype=dtype)
            z = F.conv2d(work, self.Wd.weight.to(dtype=dtype))
            responses = self.responses(z)
            gate_input = torch.cat([z, *(v.abs().mean(1, keepdim=True) for v in responses)], 1)
            logits = F.conv2d(gate_input, self.Wg.weight.to(dtype=dtype), self.Wg.bias.to(dtype=dtype))
            weights = logits.softmax(dim=1)
            response = sum(weights[:, i:i + 1] * v for i, v in enumerate(responses))
            delta = F.conv2d(response, self.Wo.weight.to(dtype=dtype))
            return (work + delta).to(dtype=x.dtype)


class BLCBlocks(Blocks):
    """Original complete stage followed by exactly one BLC; preserve .blocks keys."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", r=32, variant="d"):
        super().__init__(ch_in, ch_out, block, count, stage_num, act=act, variant=variant)
        if count != 2 or stage_num != 3 or variant != "d":
            raise ValueError("BLC v1 wraps the original two-block P3 stage only")
        self.blc = BLC(ch_out * block.expansion, r)

    def forward(self, x):
        return self.blc(super().forward(x))
