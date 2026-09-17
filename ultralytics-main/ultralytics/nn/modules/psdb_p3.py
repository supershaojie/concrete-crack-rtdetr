# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""PSDB-P3 v1: bounded P3-guided phase selection before P2 detail compression.

The shared detail branch and its initialization order derive from SDB-P3 commit
f976779bcdb114178f83a3d2fa959baf6d9655ec. Only the phase selector is new.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .block import RepC3


class PSDBP3(nn.Module):
    """Add selected, phase-major P2 detail to the complete original P3 feature."""

    def __init__(self, p2_channels=64, p3_channels=256, detail_channels=32, phase_channels=8):
        super().__init__()
        if detail_channels != 32 or phase_channels != 8:
            raise ValueError("PSDB-P3 v1 requires detail_channels=32 and phase_channels=8")
        if p2_channels <= 0 or p3_channels <= 0:
            raise ValueError("PSDB-P3 P2 and P3 channels must be positive")
        self.p2_channels = p2_channels
        self.p3_channels = p3_channels
        self.detail_channels = detail_channels
        self.phase_channels = phase_channels
        # Match the fixed SDB reference, including Conv2d constructors' RNG use.
        # Restore the caller's CPU RNG so later parent neck/decoder initialization
        # remains identical. This constructs CPU parameters, not CUDA tensors.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.W_d = nn.Conv2d(4 * p2_channels, detail_channels, 1, bias=False)
            self.DW3 = nn.Conv2d(detail_channels, detail_channels, 3, padding=1,
                                 groups=detail_channels, bias=False)
            self.GN_D = nn.GroupNorm(4, detail_channels, eps=1e-5, affine=True)
            self.W_q = nn.Conv2d(p3_channels, detail_channels, 1, bias=False)
            self.GN_Q = nn.GroupNorm(4, detail_channels, eps=1e-5, affine=True)
            self.W_g = nn.Conv2d(3 * detail_channels, detail_channels, 1, bias=True)
            self.W_o = nn.Conv2d(detail_channels, p3_channels, 1, bias=False)
            for layer in (self.W_d, self.DW3, self.W_q, self.W_g):
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(self.W_g.bias)
            nn.init.zeros_(self.W_o.weight)
            for norm in (self.GN_D, self.GN_Q):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)
        # Independent fixed stream: adding q/k never changes shared SDB tensors.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(43)
            self.W_phi_q = nn.Conv2d(detail_channels, phase_channels, 1, bias=False)
            self.W_phi_k = nn.Conv2d(p2_channels, phase_channels, 1, bias=False)
            nn.init.xavier_uniform_(self.W_phi_q.weight)
            nn.init.xavier_uniform_(self.W_phi_k.weight)

    @staticmethod
    def phase_rearrange(p2):
        """Return phase-major 00,10,01,11; replicate only odd right/bottom edges."""
        if p2.ndim != 4:
            raise ValueError(f"PSDB-P3 P2 must be BCHW, got {tuple(p2.shape)}")
        height, width = p2.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError("PSDB-P3 P2 spatial dimensions must be nonempty")
        if height % 2 or width % 2:
            p2 = F.pad(p2, (0, width % 2, 0, height % 2), mode="replicate")
        return torch.cat((p2[..., 0::2, 0::2], p2[..., 1::2, 0::2],
                          p2[..., 0::2, 1::2], p2[..., 1::2, 1::2]), dim=1)

    def phase_attention(self, phases, semantic):
        """Compute one query and four shared-key cosine scores in local FP32.

        Functional convolutions retain differentiable casts of half parameters;
        no parameter replacement, detach, or module dtype mutation is performed.
        MAC accounting must include these five functional projection calls.
        """
        with torch.autocast(device_type=semantic.device.type, enabled=False):
            query = F.conv2d(semantic.float(), self.W_phi_q.weight.float())
            query = F.normalize(query, p=2, dim=1, eps=1e-6)
            keys = [F.normalize(F.conv2d(phase.float(), self.W_phi_k.weight.float()),
                                p=2, dim=1, eps=1e-6) for phase in phases]
            scores = torch.stack([(query * key).sum(dim=1) for key in keys], dim=1)
            return torch.softmax(2.0 * scores, dim=1)

    def phase_weights(self, phases, semantic):
        """Return FP32 weights in [0.75, 1.75], summing to four at each site."""
        with torch.autocast(device_type=semantic.device.type, enabled=False):
            return 0.75 + self.phase_attention(phases, semantic)

    def forward(self, p2, original_p3):
        """Preserve the original semantic path and add only the learned residual."""
        if p2.ndim != 4 or original_p3.ndim != 4:
            raise ValueError("PSDB-P3 expects BCHW P2 and original P3 inputs")
        if p2.shape[0] != original_p3.shape[0]:
            raise ValueError("PSDB-P3 P2 and P3 batch sizes must agree")
        if p2.shape[1] != self.p2_channels or original_p3.shape[1] != self.p3_channels:
            raise ValueError(f"PSDB-P3 expected P2/P3 channels {self.p2_channels}/{self.p3_channels}, "
                             f"got {p2.shape[1]}/{original_p3.shape[1]}")
        expected = tuple((size + 1) // 2 for size in p2.shape[-2:])
        if tuple(original_p3.shape[-2:]) != expected:
            raise ValueError(f"PSDB-P3 requires P3 spatial size ceil(P2/2)={expected}; "
                             f"P2={tuple(p2.shape[-2:])}, P3={tuple(original_p3.shape[-2:])}")
        phases = self.phase_rearrange(p2).chunk(4, dim=1)
        # Keep Q, D, gate, and output projection on the original SDB dtype/AMP
        # boundaries. Only selection has an explicit FP32 numerical region.
        semantic = self.GN_Q(self.W_q(original_p3))
        weights = self.phase_weights(phases, semantic)
        selected = torch.cat([(weights[:, i:i + 1] * phase).to(phase.dtype)
                              for i, phase in enumerate(phases)], dim=1)
        detail = F.silu(self.GN_D(self.DW3(self.W_d(selected))))
        gate = self.W_g(torch.cat((detail, semantic, detail * semantic), dim=1)).sigmoid()
        delta = self.W_o(gate * detail)
        return original_p3 + delta.to(original_p3.dtype)


class PSDBRepC3(RepC3):
    """Preserve RepC3 cv1/cv2/m/cv3 state keys and consume [fusion_input, P2]."""

    def __init__(self, c1, c2, n=3, e=0.5, p2_channels=64, detail_channels=32, phase_channels=8):
        super().__init__(c1, c2, n, e)
        self.psdb = PSDBP3(p2_channels, c2, detail_channels, phase_channels)

    def forward(self, inputs):
        fusion_input, p2 = inputs
        original_p3 = super().forward(fusion_input)
        return self.psdb(p2, original_p3)
