# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Token Redundancy Calibration v1: a content-conditioned additive AIFI logit bias.

Positive coefficients reduce the multiplicity effect of redundant content; negative
coefficients can enhance it. Redundancy is not a background probability or a crack
segmentation mask. The original AIFI owns every QKV/FFN/norm parameter and operation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import AIFI

__all__ = ("TokenRedundancyCalibration", "AIFI_TRC")


class TokenRedundancyCalibration(nn.Module):
    """Fixed C=256, d=32, eight-head TRC v1; exactly 10,248 trainable parameters."""

    def __init__(self, channels: int = 256, num_heads: int = 8):
        super().__init__()
        if channels != 256 or num_heads != 8:
            raise ValueError("TRC-AIFI v1 requires channels=256 and num_heads=8")
        self.channels = channels
        self.num_heads = num_heads
        self.descriptor_dim = 32
        self.tau = 8.0
        self.lambda_bound = 0.5
        self.r_clip = 2.0
        # Construct only CPU tensors and restore the caller's RNG. This also makes
        # the two variants' TRC initialization identical without perturbing later
        # public layers. Zeroing happens only here, never in loading or forward.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.descriptor = nn.Linear(channels, self.descriptor_dim, bias=False, device="cpu")
            self.coefficient = nn.Linear(channels, num_heads, bias=True, device="cpu")
            nn.init.xavier_uniform_(self.descriptor.weight)
            nn.init.zeros_(self.coefficient.weight)
            nn.init.zeros_(self.coefficient.bias)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        """Return [B,H,N,N] FP32 bias; optional per-token diagnostics are never cached.

        x is flattened content before positional embedding and before the main
        path's pre-norm. Functional FP32 casts remain differentiable even when
        stored weights are half; no parameter objects or dtypes are mutated.
        """
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[1] < 1:
            raise ValueError(f"TRC expects nonempty [B,N,256] content, got {tuple(x.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = F.layer_norm(x.float(), (self.channels,), weight=None, bias=None, eps=1e-5)
            z = F.normalize(F.linear(u, self.descriptor.weight.float()), p=2, dim=-1, eps=1e-6)
            similarity = (z @ z.transpose(-1, -2)).clamp(-1.0, 1.0)
            # Include the self term unchanged. For a zero descriptor it is exp(-8),
            # so rho >= 1 would be an incorrect invariant.
            rho = torch.exp(self.tau * (similarity - 1.0)).sum(dim=-1)
            log_rho = rho.clamp_min(1e-6).log()
            r = (log_rho - log_rho.mean(dim=-1, keepdim=True)).clamp(-self.r_clip, self.r_clip)
            coefficient = self.lambda_bound * torch.tanh(
                F.linear(u, self.coefficient.weight.float(), self.coefficient.bias.float())
            )
            coefficient = coefficient.transpose(1, 2)  # B,N,H -> B,H,N (query-conditioned)
            bias = -coefficient.unsqueeze(-1) * r[:, None, None, :]
        if return_diagnostics:
            return bias, {"rho": rho, "r": r, "lambda": coefficient}
        return bias


class AIFI_TRC(AIFI):
    """Original AIFI with an added attention mask, retaining all public state paths.

    AIFI.forward (flatten/position/reshape) and the parent's pre/post-norm attention
    implementations are inherited unchanged. In particular, need_weights retains
    its original default and no alternate QKV or attention algorithm is selected.
    """

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.trc = TokenRedundancyCalibration(c1, num_heads)

    @staticmethod
    def _attention_dtype(src: torch.Tensor, pos: torch.Tensor | None):
        """Dtype of MHA's projected QKV, including the project's older torch API."""
        if hasattr(torch, "get_autocast_dtype"):
            if torch.is_autocast_enabled(src.device.type):
                return torch.get_autocast_dtype(src.device.type)
        else:  # torch 2.1.x on the historical server
            if src.device.type == "cuda" and torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
            if src.device.type == "cpu" and torch.is_autocast_cpu_enabled():
                return torch.get_autocast_cpu_dtype()
        return src.dtype if pos is None else torch.promote_types(src.dtype, pos.dtype)

    @staticmethod
    def _float_mask(mask: torch.Tensor, dtype: torch.dtype, device: torch.device, name: str):
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, device=device, dtype=dtype).masked_fill_(mask.to(device), float("-inf"))
        if not torch.is_floating_point(mask):
            raise TypeError(f"{name} must be a bool or floating additive mask, got {mask.dtype}")
        return mask.to(device=device, dtype=dtype)

    def _attention_masks(self, src, src_mask, src_key_padding_mask, pos):
        batch, tokens, _ = src.shape
        dtype = self._attention_dtype(src, pos)
        with torch.autocast(device_type=src.device.type, enabled=False):
            # reshape preserves batch-major/head-minor order expected by MHA.
            bias = self.trc(src).reshape(batch * self.trc.num_heads, tokens, tokens)
            if src_mask is not None:
                allowed = ((tokens, tokens), (batch * self.trc.num_heads, tokens, tokens))
                if tuple(src_mask.shape) not in allowed:
                    raise ValueError(f"src_mask shape {tuple(src_mask.shape)} must be one of {allowed}")
                bias = bias + self._float_mask(src_mask, torch.float32, src.device, "src_mask")
            bias = bias.to(dtype=dtype)
            if src_key_padding_mask is not None:
                if tuple(src_key_padding_mask.shape) != (batch, tokens):
                    raise ValueError(f"src_key_padding_mask must have shape {(batch, tokens)}")
                # MHA canonicalizes bool padding masks to *input* dtype. Under AMP
                # that could promote our half mask back to FP32 before baddbmm.
                # Convert here to keep identical additive semantics and QKV dtype.
                src_key_padding_mask = self._float_mask(
                    src_key_padding_mask, dtype, src.device, "src_key_padding_mask"
                )
        return bias, src_key_padding_mask

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        bias, padding = self._attention_masks(src, src_mask, src_key_padding_mask, pos)
        return super().forward_post(src, bias, padding, pos)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # Compute TRC on original content; the parent still applies norm1 to the
        # main Q/K/V path in precisely its original order.
        bias, padding = self._attention_masks(src, src_mask, src_key_padding_mask, pos)
        return super().forward_pre(src, bias, padding, pos)
