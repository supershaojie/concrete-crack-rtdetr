# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Spatially Conditioned Channel Attention AIFI (fixed first experiment)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import AIFI

__all__ = ("SCCAAIFI",)


class SCCAAIFI(AIFI):
    """Retain AIFI spatial MHA/FFN and add a zero-output, spatially conditioned channel residual.

    Inherited AIFI.forward owns flattening, W/H position construction and unflattening.
    Only forward_pre/post are overridden. No activations or diagnostics are cached.
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.scca_width, self.scca_heads, self.scca_dim = 64, 4, 16
        self.scca_x_norm = nn.LayerNorm(c1, eps=1e-5, elementwise_affine=False)
        self.scca_s_norm = nn.LayerNorm(c1, eps=1e-5, elementwise_affine=False)
        # Explicit CPU construction also protects the RNG with non-CPU default devices.
        # Subsequent model.to()/half() moves/converts these layers normally.
        with torch.random.fork_rng(devices=[]):
            self.scca_q = nn.Linear(c1, 64, bias=False, device="cpu")
            self.scca_k = nn.Linear(c1, 64, bias=False, device="cpu")
            self.scca_v = nn.Linear(c1, 64, bias=False, device="cpu")
            self.scca_o = nn.Linear(64, c1, bias=False, device="cpu")
            for projection in (self.scca_q, self.scca_k, self.scca_v):
                nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(self.scca_o.weight)
        self.scca_temperature_raw = nn.Parameter(torch.zeros(4, 1, 1, device="cpu"))

    def scca_channel(self, x, s):
        """Return D and small Ac for optional external diagnostics; forward stores neither."""
        b, n, _ = x.shape
        xn, sn = self.scca_x_norm(x), self.scca_s_norm(s)
        q = self.scca_q(sn).reshape(b, n, 4, 16).permute(0, 2, 3, 1)
        k = self.scca_k(xn).reshape(b, n, 4, 16).permute(0, 2, 3, 1)
        v = self.scca_v(xn).reshape(b, n, 4, 16).permute(0, 2, 3, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            q = F.normalize(q - q.mean(-1, keepdim=True), dim=-1, eps=1e-6)
            k = F.normalize(k - k.mean(-1, keepdim=True), dim=-1, eps=1e-6)
            temperature = (math.log(4.0) * self.scca_temperature_raw.float().tanh()).exp()
            attention = (temperature * (q @ k.transpose(-2, -1))).softmax(dim=-1)
            u = (attention @ v).permute(0, 3, 1, 2).reshape(b, n, 64)
        # FP32 numerical core -> real projection weight dtype (also supports model.half()).
        delta = self.scca_o(u.to(dtype=self.scca_o.weight.dtype)).to(dtype=x.dtype)
        return delta, attention

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        s = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        delta, _ = self.scca_channel(src, s)
        src = self.norm1(src + self.dropout1(s) + delta)
        t = self.fc2(self.dropout(self.act(self.fc1(src))))
        return self.norm2(src + self.dropout2(t))

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        xp = self.norm1(src)
        q = k = self.with_pos_embed(xp, pos)
        s = self.ma(q, k, value=xp, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        delta, _ = self.scca_channel(xp, s)
        src = src + self.dropout1(s) + delta
        t = self.fc2(self.dropout(self.act(self.fc1(self.norm2(src)))))
        return src + self.dropout2(t)
