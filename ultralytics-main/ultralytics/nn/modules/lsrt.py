# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""LSRT v1: low-level-guided residual transport from a fixed 3x3 coarse neighborhood."""

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Concat


class LocalSemanticResidualTransport(nn.Module):
    """Return an FP32 correction; leave the original nearest upsample and lateral coordinates intact.

    The four fine-grid phases have independent queries. Only 32-channel coarse K/V
    neighborhoods are unfolded; no 256-channel fine-grid neighborhood is materialized.
    The fixed inverse temperature multiplies cosine similarity (no sqrt(d) divisor).
    """

    def __init__(self, channels=256, d=32, tau=8.0):
        super().__init__()
        if channels != 256 or d != 32 or tau != 8.0:
            raise ValueError("LSRT v1 requires channels=256, d=32 and fixed inverse temperature tau=8.0")
        self.channels, self.d, self.tau = channels, d, float(tau)
        # New CPU parameters cannot perturb later RepC3/decoder/common initialization.
        # A local seed also gives the two LSRT variants identical new initial values.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.q_proj = nn.Conv2d(channels, d, 1, bias=False)
            self.k_proj = nn.Conv2d(channels, d, 1, bias=False)
            self.v_proj = nn.Conv2d(channels, d, 1, bias=False)
            self.out_proj = nn.Conv2d(d, channels, 1, bias=False)
            self.rel_bias = nn.Parameter(torch.zeros(9))
            for projection in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _finite(tensor, name):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"LSRT {name} contains NaN or Inf")

    def _validate(self, low, high):
        if low.ndim != 4 or high.ndim != 4:
            raise ValueError(f"LSRT expects BCHW tensors; L={tuple(low.shape)}, H={tuple(high.shape)}")
        if (low.shape[0] != high.shape[0] or low.shape[1] != self.channels
                or high.shape[1] != self.channels or min(high.shape[-2:]) < 1
                or low.shape[-2:] != (2 * high.shape[-2], 2 * high.shape[-1])):
            raise ValueError(f"LSRT requires matching B, C=256 and strict nearest x2 geometry; "
                             f"L={tuple(low.shape)}, H={tuple(high.shape)}")
        if low.device != high.device or not low.is_floating_point() or not high.is_floating_point():
            raise ValueError("LSRT requires floating-point L/H on the same device")
        self._finite(low, "L input")
        self._finite(high, "H input")

    @staticmethod
    def to_phases(tensor):
        """B,D,2h,2w -> B,D,4,h*w; phase index = 2*(y%2) + x%2."""
        b, d, height, width = tensor.shape
        h, w = height // 2, width // 2
        return tensor.reshape(b, d, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, d, 4, h * w)

    @staticmethod
    def from_phases(tensor, h, w):
        """Invert to_phases without moving any low-level coordinates."""
        b, d = tensor.shape[:2]
        return tensor.reshape(b, d, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, d, 2 * h, 2 * w)

    @staticmethod
    def value_neighborhoods(values):
        """Return row-major neighbors, explicit center-relative differences and validity.

        Shapes: neighbors/differences B,D,9,h*w; mask 1,9,h*w. The original
        center at index 4 is taken from the same projected V tensor.
        """
        b, d, h, w = values.shape
        neighbors = F.unfold(values, kernel_size=3, padding=1).reshape(b, d, 9, h * w)
        valid = F.unfold(values.new_ones(1, 1, h, w), kernel_size=3, padding=1).bool()
        differences = torch.where(valid[:, None], neighbors - neighbors[:, :, 4:5], 0.0)
        return neighbors, differences, valid

    @staticmethod
    def aggregate(attention, value_differences):
        """B,4,9,N weights and B,D,9,N differences -> B,D,4,N residual values."""
        return torch.einsum("bpkn,bdkn->bdpn", attention, value_differences)

    def forward(self, low, high):
        return self._forward(low, high, diagnostics=False)

    def forward_with_diagnostics(self, low, high):
        """Explicit opt-in tensors for bounded checks; never store features on the module."""
        return self._forward(low, high, diagnostics=True)

    def diagnostic_summary(self, upsampled, delta, details):
        """Convert an opt-in diagnostic result to small JSONable scalars, with no feature cache.

        Call after backward if gradient norms are needed. These logging-only detach
        operations do not affect forward tensors or the model's training graph.
        """
        weights = details["attention"].detach().float()
        delta_norm = float(delta.detach().float().norm())
        upsampled_norm = float(upsampled.detach().float().norm())
        return {
            "center_attention_mean": float(weights[:, :, 4].mean()),
            "neighbor_attention_mean": weights.mean(dim=(0, 1, 3)).cpu().tolist(),
            "attention_entropy_mean": float(-(weights * weights.clamp_min(1e-30).log()).sum(2).mean()),
            "delta_norm": delta_norm,
            "U_norm": upsampled_norm,
            "delta_to_U_norm_ratio": delta_norm / max(upsampled_norm, 1e-12),
            "out_proj_norm": float(self.out_proj.weight.detach().float().norm()),
            "gradient_norms": {name: float(parameter.grad.detach().float().norm())
                               if parameter.grad is not None else None
                               for name, parameter in self.named_parameters()},
        }

    def _forward(self, low, high, diagnostics):
        self._validate(low, high)
        b, _, h, w = high.shape
        # Functional float casts retain parameter/input autograd, support native AMP
        # and .half() inference, and never mutate the registered parameter objects.
        with torch.autocast(device_type=low.device.type, enabled=False):
            query = F.conv2d(low.float(), self.q_proj.weight.float())
            key = F.conv2d(high.float(), self.k_proj.weight.float())
            value = F.conv2d(high.float(), self.v_proj.weight.float())
            query = self.to_phases(F.normalize(query, p=2, dim=1, eps=1e-6))
            key = F.normalize(key, p=2, dim=1, eps=1e-6)
            keys = F.unfold(key, kernel_size=3, padding=1).reshape(b, self.d, 9, h * w)
            _, differences, valid = self.value_neighborhoods(value)
            logits = self.tau * torch.einsum("bdpn,bdkn->bpkn", query, keys)
            logits = logits + self.rel_bias.float()[None, None, :, None]
            attention = logits.masked_fill(~valid[:, None], -torch.inf).softmax(dim=2)
            transported = self.from_phases(self.aggregate(attention, differences), h, w)
            delta = F.conv2d(transported, self.out_proj.weight.float())
            self._finite(delta, "correction")
        if diagnostics:
            return delta, {"attention": attention, "valid": valid, "value_differences": differences,
                           "transported": transported, "center_index": 4, "phase_order": "00,01,10,11"}
        return delta


class LSRTConcat(nn.Module):
    """Keep the layer-18 concat order [U + LSRT(L,H), L] and its 512 output channels."""

    def __init__(self, channels, d=32, tau=8.0):
        super().__init__()
        if list(channels) != [256, 256, 256]:
            raise ValueError(f"LSRTConcat requires [U,L,H] channels [256,256,256], received {channels}")
        self.lsrt = LocalSemanticResidualTransport(channels[0], d, tau)
        self.concat = Concat(1)

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
            raise ValueError("LSRTConcat expects exactly [U, L, H]")
        upsampled, low, high = inputs
        if upsampled.shape != low.shape or upsampled.device != low.device:
            raise ValueError(f"LSRTConcat U/L shape/device mismatch: U={tuple(upsampled.shape)}, L={tuple(low.shape)}")
        self.lsrt._finite(upsampled, "U input")
        delta = self.lsrt(low, high)
        return self.concat((upsampled + delta.to(upsampled.dtype), low))
