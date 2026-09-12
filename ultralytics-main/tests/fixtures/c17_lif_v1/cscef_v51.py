# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""CSCEF-v5.1: per-image spatial mean of the complete V5 structure confidence."""

import torch

from .cscef_v5 import CSCEFv5

__all__ = ("CSCEFv51",)


class CSCEFv51(CSCEFv5):
    """Keep V5 content, initialization and state keys; only spatially average its confidence."""

    @torch.no_grad()
    def _compute_structure_confidence(self, gradient_x: torch.Tensor, gradient_y: torch.Tensor) -> torch.Tensor:
        """Average detached FP32 nonlinear confidence over H/W separately for each image.

        The inherited forward casts to content dtype only after this reduction, then applies
        output_projection(c_mean * h) at V5's original residual addition point.
        """
        confidence = super()._compute_structure_confidence(gradient_x, gradient_y)
        return confidence.mean(dim=(-2, -1), keepdim=True).expand_as(confidence)
