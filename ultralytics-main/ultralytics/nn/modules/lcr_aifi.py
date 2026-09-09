# Ultralytics AGPL-3.0
"""Local Contrast Reconstruction FFN in the original AIFI attention/norm flow."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .transformer import AIFI

__all__ = ("LCRAIFI",)


class LCRAIFI(AIFI):
    """AIFI with an identity-initialized local contrast FFN, actual H/W per call."""

    def __init__(self, c1, cm=1024, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        if cm != 1024:
            raise ValueError("LCR first version fixes cm=1024")
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.alpha, self.rms_eps = 0.5, 1e-6
        # Do not consume RNG used by the remaining C2 encoder/decoder/classification layers.
        with torch.random.fork_rng(devices=[]):
            self.dwconv = nn.Conv2d(cm, cm, 3, groups=cm, padding=0, bias=False)
            self.gate_in = nn.Conv2d(cm, 64, 1, groups=16, bias=True)
            self.gate_out = nn.Conv2d(64, cm, 1, groups=16, bias=True)
            self.gate_act = nn.GELU()
            nn.init.zeros_(self.dwconv.weight)
            with torch.no_grad():
                self.dwconv.weight[:, 0, 1, 1] = 1
            nn.init.xavier_uniform_(self.gate_in.weight)
            nn.init.zeros_(self.gate_in.bias)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)

    def _ffn(self, tokens, h, w):
        u = self.fc1(tokens).transpose(1, 2).reshape(tokens.shape[0], 1024, h, w)
        s = self.dwconv(F.pad(u, (1, 1, 1, 1), mode="replicate"))
        m = F.avg_pool2d(F.pad(s, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        # Explicit FP32 difference/RMS, per spatial location over hidden channels only.
        with torch.autocast(device_type=s.device.type, enabled=False):
            d = s.float() - m.float()
            e = d / torch.sqrt(d.square().mean(dim=1, keepdim=True) + self.rms_eps)
        # FP32 weights under AMP use native autocast; genuine half weights need half input.
        t = self.gate_out(self.gate_act(self.gate_in(e.to(self.gate_in.weight.dtype))))
        ps, pm = self.act(s), self.act(m)
        with torch.autocast(device_type=s.device.type, enabled=False):
            delta = self.alpha * torch.tanh(t.float())
            a = (ps.float() + delta * (ps.float() - pm.float())).to(ps.dtype)
        a = a.flatten(2).transpose(1, 2).contiguous()  # Original Linear token layout for identical dropout RNG
        return self.fc2(self.dropout(a))

    def forward(self, x):
        c, h, w = x.shape[1:]
        # Preserve even the original AIFI positional embedding ordering on rectangles.
        pos = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.ma(q, k, value=src2, attn_mask=None, key_padding_mask=None)[0]
            src = src + self.dropout1(src2)
            src2 = self._ffn(self.norm2(src), h, w)
            src = src + self.dropout2(src2)
        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.ma(q, k, value=src, attn_mask=None, key_padding_mask=None)[0]
            src = self.norm1(src + self.dropout1(src2))
            src2 = self._ffn(src, h, w)
            src = self.norm2(src + self.dropout2(src2))
        return src.permute(0, 2, 1).reshape(-1, c, h, w).contiguous()
