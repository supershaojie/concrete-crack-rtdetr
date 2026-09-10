# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Late decoder-P3 residual from read-only projected backbone P3/P4 references."""
import torch
from .cscef_v51 import CSCEFv51

__all__ = ("DRCSCEFv6",)


class DRCSCEFv6(CSCEFv51):
    """Preserve the C17 content/confidence kernel; decouple its source and destination."""

    def __init__(self, c_base, c_lateral, c_semantic, hidden_channels=32, num_groups=8, eps=1e-6):
        if c_base != c_lateral:
            raise ValueError("DR-CSCEF requires equal base and projected lateral channels.")
        super().__init__(c_lateral, c_semantic, hidden_channels, num_groups, eps)

    def side_residual(self, lateral_ref, semantic_ref):
        """Only the side inputs detach; projections and content remain trainable."""
        l, s = self._project_features(lateral_ref.detach(), semantic_ref.detach())
        h = self.activation(self.content_norm(self.depthwise_conv(self.mix_projection(torch.cat((l, s), dim=1)))))
        gx, gy = self._compute_scharr_components(l)
        confidence = self._compute_structure_confidence(gx, gy).to(dtype=h.dtype)
        return self.output_projection(confidence * h)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("DRCSCEFv6 expects [P3_base, P3_ref, P4_ref].")
        base, lateral_ref, semantic_ref = inputs
        if base.shape != lateral_ref.shape or base.device != lateral_ref.device:
            raise ValueError("P3 base/reference must have identical BCHW shape and device.")
        return base + self.side_residual(lateral_ref, semantic_ref).to(dtype=base.dtype)
