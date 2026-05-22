"""Optional accelerated kernels for NAIME hot paths.

Kernels in this package must always keep a PyTorch fallback. They are
performance backends, not architecture semantics.
"""

from .cross_entropy import cross_entropy_loss
from .fused_lm_ce import fused_lm_cross_entropy_loss

__all__ = ["cross_entropy_loss", "fused_lm_cross_entropy_loss"]
