# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""C17 forward preserved; attenuate only the additional semantic input gradient."""

from .cscef_v51 import CSCEFv51

__all__ = ("CSCEFv52Compat",)


class CSCEFv52Compat(CSCEFv51):
    """Keep C17 state/parameters and residual; semantic backward has fixed gain 0.25."""

    def __init__(self, c_lateral, c_semantic, hidden_channels=32, num_groups=8, eps=1e-6,
                 semantic_grad_scale=0.25):
        if float(semantic_grad_scale) != 0.25:
            raise ValueError("CSCEF-v5.2-Compat fixes semantic_grad_scale=0.25; no gamma sweep.")
        super().__init__(c_lateral, c_semantic, hidden_channels, num_groups, eps)
        self.semantic_grad_scale = float(semantic_grad_scale)  # configuration, not state or optimizer parameter

    def _semantic_grad_view(self, semantic):
        detached = semantic.detach()
        return detached + self.semantic_grad_scale * (semantic - detached)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CSCEFv52Compat expects [lateral, semantic].")
        lateral, semantic = inputs
        return super().forward((lateral, self._semantic_grad_view(semantic)))
