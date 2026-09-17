# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""CSR-P3 v1: differentiable curved-minus-straight sampling at the P3 lateral projection.

Coordinates are pixel centers, with independent signed increments on each side of
each axis. This implementation intentionally does not reuse DSConv's detached
offset accumulation. Grid-sample cost is additional to ordinary convolution MACs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("CurvedSamplingResidual", "CSR", "CSRConv")


class CurvedSamplingResidual(nn.Module):
    """Identity at initialization, with a nonzero output projection and live offset gradients."""

    def __init__(self, channels=256, detail_channels=32, kernel_points=7):
        super().__init__()
        if kernel_points != 7:
            raise ValueError("CSR-P3 v1 fixes kernel_points=7")
        self.channels = channels
        self.detail_channels = detail_channels
        self.kernel_points = kernel_points
        # Preserve the parent's construction RNG, including all later head layers.
        # The local seed gives both variants identical CSR tensors independently of nc.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.in_proj = nn.Conv2d(channels, detail_channels, 1, bias=False)
            self.offset_dw = nn.Conv2d(detail_channels, detail_channels, 3, padding=1,
                                       groups=detail_channels, bias=True)
            self.offset_pw = nn.Conv2d(detail_channels, 12, 1, bias=True)
            self.theta = nn.Parameter(torch.zeros(2, detail_channels, 7))
            self.out_proj = nn.Conv2d(detail_channels, channels, 1, bias=False)
            for layer in (self.in_proj, self.offset_dw, self.out_proj):
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(self.offset_dw.bias)
            nn.init.zeros_(self.offset_pw.weight)
            nn.init.zeros_(self.offset_pw.bias)

    def offset_logits(self, z):
        """Only the final offset projection is forced to FP32; float views retain gradients."""
        # initialize_weights() makes ordinary nn.SiLU modules inplace globally.
        # Keep this operation explicitly non-inplace even after RTDETR construction.
        hidden = F.silu(self.offset_dw(z), inplace=False)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return F.conv2d(hidden.float(), self.offset_pw.weight.float(), self.offset_pw.bias.float())

    @staticmethod
    def cumulative_offsets(logits):
        """Return orthogonal displacement in r=[-3,-2,-1,0,1,2,3] order, FP32."""
        b, _, h, w = logits.shape
        with torch.autocast(device_type=logits.device.type, enabled=False):
            increments = logits.float().reshape(b, 2, 2, 3, h, w).tanh()
            sides = increments.cumsum(dim=3)
            center = sides.new_zeros((b, 2, 1, h, w))
            return torch.cat((sides[:, :, 0].flip(2), center, sides[:, :, 1]), dim=2)

    @staticmethod
    def _base_coordinates(logits):
        h, w = logits.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h, device=logits.device, dtype=torch.float32),
                              torch.arange(w, device=logits.device, dtype=torch.float32), indexing="ij")
        return x.unsqueeze(0), y.unsqueeze(0)

    @staticmethod
    def _grid(x, y, h, w):
        # Last coordinate dimension is always (x,y), including rectangular feature maps.
        return torch.stack((2 * (x + 0.5) / w - 1, 2 * (y + 0.5) / h - 1), dim=-1)

    def _point_grids(self, offsets, x, y, axis, point):
        b, _, _, h, w = offsets.shape
        r = point - 3
        d = offsets[:, axis, point]
        q0x, q0y = (x + r, y) if axis == 0 else (x, y + r)
        qx, qy = (q0x.expand_as(d), q0y + d) if axis == 0 else (q0x + d, q0y.expand_as(d))
        return self._grid(qx, qy, h, w), self._grid(q0x, q0y, h, w).expand(b, h, w, 2)

    def sampling_grids(self, logits):
        """Diagnostic helper: materialize grids [B,axis,point,H,W,xy]; forward streams points."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            offsets = self.cumulative_offsets(logits)
            x, y = self._base_coordinates(logits)
            curves, references = [], []
            for axis in range(2):
                pairs = [self._point_grids(offsets, x, y, axis, point) for point in range(7)]
                curves.append(torch.stack([pair[0] for pair in pairs], dim=1))
                references.append(torch.stack([pair[1] for pair in pairs], dim=1))
            return torch.stack(curves, dim=1), torch.stack(references, dim=1)

    def sampling_difference(self, z, logits):
        """Aggregate 0.5 * softmax(theta) * (curved - straight) with no detached features."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            offsets = self.cumulative_offsets(logits)
            x, y = self._base_coordinates(logits)
            weights = self.theta.float().softmax(dim=-1)
            result = torch.zeros_like(z)
            for axis in range(2):
                for point in (0, 1, 2, 4, 5, 6):
                    # The center difference is identically zero, but its softmax logit
                    # remains in the denominator and receives gradients after startup.
                    curve_grid, straight_grid = self._point_grids(offsets, x, y, axis, point)
                    curve = F.grid_sample(z, curve_grid, mode="bilinear", padding_mode="border", align_corners=False)
                    straight = F.grid_sample(z, straight_grid, mode="bilinear", padding_mode="border", align_corners=False)
                    result = result + (curve - straight) * (0.5 * weights[axis, :, point].view(1, -1, 1, 1))
            return result

    def residual(self, x):
        z = self.in_proj(x)
        difference = self.sampling_difference(z, self.offset_logits(z))
        return self.out_proj(difference.to(dtype=self.out_proj.weight.dtype)).to(dtype=x.dtype)

    def forward(self, x):
        return x + self.residual(x)

    @torch.no_grad()
    def diagnostics(self, x):
        """Finite scalar summaries only; no grids, input samples, or mutable runtime buffers."""
        z = self.in_proj(x)
        logits = self.offset_logits(z)
        increments = logits.float().tanh()
        endpoints = self.cumulative_offsets(logits)[:, :, (0, 6)].abs().flatten()
        difference = self.sampling_difference(z, logits)
        delta = self.out_proj(difference.to(self.out_proj.weight.dtype)).float()
        levels = torch.tensor([0., .5, .9, .99, 1.], device=x.device)
        return {
            "increment_abs_quantiles_0_50_90_99_100": increments.abs().flatten().quantile(levels).tolist(),
            "endpoint_abs_quantiles_0_50_90_99_100": endpoints.quantile(levels).tolist(),
            "tanh_abs_ge_095_fraction": float((increments.abs() >= .95).float().mean()),
            "difference_abs_max": float(difference.abs().max()),
            "residual_l2_relative_to_input": float(delta.norm() / x.float().norm().clamp_min(1e-12)),
            "all_finite": bool(torch.isfinite(logits).all() and torch.isfinite(delta).all()),
        }


CSR = CurvedSamplingResidual


class CSRConv(Conv):
    """Preserve original conv/bn/act keys and apply CSR after the original Conv output."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, detail_channels=32, kernel_points=7):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.csr = CurvedSamplingResidual(c2, detail_channels, kernel_points)

    def forward(self, x):
        return self.csr(super().forward(x))

    def forward_fuse(self, x):
        # BaseModel.fuse rebinds forward to this method; retaining CSR here is essential.
        return self.csr(super().forward_fuse(x))
