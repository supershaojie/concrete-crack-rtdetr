# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""BFR-P4 v1: sample/channel dependent, DC-free real-DCT residual.

Only ``wo.weight`` starts at zero. Constants are rebuilt directly in FP32 on
every call, so checkpoints, deepcopy and dtype/device moves have no cache state.
The latent residual (not its learned output projection) has norm at most half
the latent input norm, up to floating-point roundoff.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import RepC3


class BFRP4(nn.Module):
    """Fixed four-band frequency residual, with a local FP32 computation path."""

    initialization_seed = 42

    def __init__(self, channels=256, latent_channels=32):
        super().__init__()
        if channels <= 0 or latent_channels != 32:
            raise ValueError("BFR-P4 v1 requires positive channels and latent_channels=32")
        self.channels = channels
        self.latent_channels = latent_channels
        # Preserve the parent's following-layer CPU RNG sequence. Unlike
        # torch.manual_seed, this call never modifies any CUDA RNG generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.initialization_seed)
            self.wd = nn.Conv2d(channels, latent_channels, 1, bias=False)
            self.gn = nn.GroupNorm(4, latent_channels, eps=1e-5, affine=True)
            self.fc1 = nn.Linear(128, 16, bias=True)
            self.fc2 = nn.Linear(16, 128, bias=True)
            self.wo = nn.Conv2d(latent_channels, channels, 1, bias=False)
            nn.init.xavier_uniform_(self.wd.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.ones_(self.gn.weight)
            nn.init.zeros_(self.gn.bias)
            nn.init.zeros_(self.wo.weight)

    @staticmethod
    def dct_basis(length, device):
        """Build the orthonormal DCT-II matrix directly in FP32."""
        if length <= 0:
            raise ValueError("DCT axes must have positive lengths")
        k = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        n = torch.arange(length, device=device, dtype=torch.float32)[None, :]
        alpha = torch.full((length, 1), math.sqrt(2.0 / length), device=device, dtype=torch.float32)
        alpha[0] = math.sqrt(1.0 / length)
        return alpha * torch.cos((math.pi / length) * k * (n + 0.5))

    @classmethod
    def constants(cls, height, width, device):
        """Return fresh FP32 bases, soft radial bands and a DC exclusion mask."""
        ch, cw = cls.dct_basis(height, device), cls.dct_basis(width, device)
        u = torch.arange(height, device=device, dtype=torch.float32)[:, None] / max(height - 1, 1)
        v = torch.arange(width, device=device, dtype=torch.float32)[None, :] / max(width - 1, 1)
        rho = torch.sqrt(u.square() + v.square()) / math.sqrt(2.0)
        centers = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], device=device, dtype=torch.float32)
        q = torch.exp(-(rho[None] - centers[:, None, None]).square() / (2.0 * 0.2**2))
        bands = q / q.sum(dim=0, keepdim=True)
        non_dc = torch.ones((height, width), device=device, dtype=torch.float32)
        non_dc[0, 0] = 0.0
        return ch, cw, bands, non_dc

    @staticmethod
    def dct2(z, ch, cw):
        return torch.matmul(torch.matmul(ch, z), cw.T)

    @staticmethod
    def idct2(f, ch, cw):
        return torch.matmul(torch.matmul(ch.T, f), cw)

    def diagnostics(self, x):
        """Compute the production path and expose intermediates without caching.

        Functional operators retain autograd edges to original Parameters even
        when those Parameters were explicitly converted to half precision.
        ``gn.bias`` is intentionally retained although DC exclusion makes it
        a mathematical zero-gradient direction.
        """
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected B,{self.channels},H,W input")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.conv2d(x.float(), self.wd.weight.float())
            z = F.group_norm(z, 4, self.gn.weight.float(), self.gn.bias.float(), self.gn.eps)
            ch, cw, bands, non_dc = self.constants(x.shape[-2], x.shape[-1], x.device)
            f = self.dct2(z, ch, cw)
            power = (f * non_dc).square()
            energy = (power.unsqueeze(2) * bands[None, None]).sum(dim=(-2, -1))
            energy = energy / (power.sum(dim=(-2, -1))[..., None] + 1e-6)
            hidden = F.silu(F.linear(energy.reshape(x.shape[0], 128), self.fc1.weight.float(), self.fc1.bias.float()))
            logits = F.linear(hidden, self.fc2.weight.float(), self.fc2.bias.float())
            coefficients = 0.5 * torch.tanh(logits).reshape(x.shape[0], 32, 4)
            delta_g = (coefficients[..., None, None] * bands[None, None]).sum(dim=2) * non_dc
            residual = self.idct2(delta_g * f, ch, cw)
            delta = F.conv2d(residual, self.wo.weight.float())
        return {"z": z, "f": f, "e": energy, "a": coefficients, "delta_g": delta_g, "r": residual, "delta": delta}

    def forward(self, x):
        return x + self.diagnostics(x)["delta"].to(dtype=x.dtype)


class BFRRepC3(RepC3):
    """Preserve every original RepC3 state path, then correct its complete output."""

    def __init__(self, c1, c2, n=3, e=0.5, latent_channels=32):
        super().__init__(c1, c2, n=n, e=e)
        self.bfr = BFRP4(c2, latent_channels)

    def forward(self, x):
        return self.bfr(super().forward(x))


def count_bfr_p4(module, inputs, output):
    """THOP custom hook: major MACs only (GN/elementwise/basis creation excluded).

    Register as ``custom_ops={BFRP4: count_bfr_p4}``. The child Conv/Linear
    modules are used functionally and receive no forward calls, avoiding double
    counting. Supports THOP variants whose total_ops is a Tensor or an int.
    """
    batch, channels, height, width = inputs[0].shape
    latent = module.latent_channels
    macs = batch * (2 * channels * latent * height * width + 2 * latent * (height**2 * width + height * width**2) + 4096)
    if isinstance(module.total_ops, torch.Tensor):
        module.total_ops.add_(module.total_ops.new_tensor(macs))
    else:
        module.total_ops += macs
