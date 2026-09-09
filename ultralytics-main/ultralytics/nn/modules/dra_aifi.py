# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Directional Relation-Aware AIFI: a bounded additive bias in the original MHA."""
import math

import torch
from torch import nn

from .transformer import AIFI, TransformerEncoderLayer

__all__ = ("DRAAIFI",)


class DRAAIFI(AIFI):
    """Keep AIFI public parameters/PE/residuals; condition the first four of eight heads."""

    def __init__(self, c1, cm=1024, num_heads=8, dropout=0., act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        if c1 != 256 or num_heads != 8:
            raise ValueError("DRA-AIFI v1 requires C2's 256 channels and 8 heads")
        # Extra draws must not perturb subsequent common C2 parameter initialization.
        with torch.random.fork_rng(devices=[]):
            self.dra_predictor = nn.Sequential(
                nn.Conv2d(c1, 16, 1, bias=False), nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False), nn.SiLU(),
                nn.Conv2d(16, 8, 1, bias=True),
            )
            nn.init.zeros_(self.dra_predictor[-1].weight)
            nn.init.zeros_(self.dra_predictor[-1].bias)
        self._geometry_key = None
        self.register_buffer("_axis", None, persistent=False)
        self.register_buffer("_decay", None, persistent=False)

    def _apply(self, fn, recurse=True):
        # Never reuse rounded geometry after .half().float() or a device transition.
        self._axis = self._decay = None
        self._geometry_key = None
        return super()._apply(fn, recurse)

    def geometry(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key != self._geometry_key:
            y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                  torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
            coords = torch.stack((x.flatten(), y.flatten()), -1) / max(h, w)
            delta = coords[None, :, :] - coords[:, None, :]
            dx, dy = delta.unbind(-1)
            rho2 = dx.square() + dy.square()
            offdiag = rho2 > 0
            denominator = torch.where(offdiag, rho2, torch.ones_like(rho2))
            self._axis = torch.stack(((dx.square() - dy.square()) / denominator,
                                      2 * dx * dy / denominator), -1)
            self._decay = .5 * torch.exp(-rho2 / (2 * .35 ** 2)) * offdiag
            self._geometry_key = key
        return self._axis, self._decay

    def directional_bias(self, x):
        """Return B[B,8,HW,HW] in FP32; no persistent activation/attention statistics."""
        b, _, h, w = x.shape
        z = self.dra_predictor(x).reshape(b, 4, 2, h * w).transpose(-1, -2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = z.float()
            d = z / torch.sqrt(1 + z.square().sum(-1, keepdim=True))
            axis, decay = self.geometry(h, w, x.device, x.dtype)
            a = torch.einsum("bhic,ijc->bhij", d, axis)
            # e_ij == e_ji for an unoriented double-angle axis.
            score = -.25 * (torch.logaddexp(-a / .25, -a.transpose(-1, -2) / .25) - math.log(2))
            modified = score * decay
            return torch.cat((modified, torch.zeros_like(modified)), dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        bias = self.directional_bias(x)
        # MHA logits use the autocast projection dtype, or the input dtype otherwise.
        if torch.is_autocast_enabled() and x.device.type == "cuda":
            attention_dtype = torch.get_autocast_gpu_dtype()
        elif x.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            attention_dtype = torch.get_autocast_cpu_dtype()
        else:
            attention_dtype = x.dtype
        mask = bias.to(dtype=attention_dtype).reshape(b * 8, h * w, h * w)
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        # Exactly one original MHA. Inherit both original pre/post-norm paths.
        out = TransformerEncoderLayer.forward(self, x.flatten(2).permute(0, 2, 1), src_mask=mask, pos=pos)
        return out.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
