# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""C24 forward with a firewall on the channel branch's two feature sources."""
from .scca_aifi import SCCAAIFI

__all__ = ("GISCCAAIFI",)


class GISCCAAIFI(SCCAAIFI):
    """Keep MHA/FFN, injection point, pre/post norm and all learned state paths."""

    def scca_channel(self, x, s):
        return super().scca_channel(x.detach(), s.detach())
