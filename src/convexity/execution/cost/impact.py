"""ImpactModel — square-root market impact in bps, degrading to ~0 at retail size (§11 / Invariant I).

    impact_bps   = η · σ_horizon_bps · sqrt(participation)
    participation = |qty| · mid / volume_proxy

DEGRADES CONTINUOUSLY to ~0 as participation -> 0, so at a $1k clip the term vanishes and spread /
fees / adverse selection dominate (P8 / Invariant I). η is a RED/class constant. The linear
small-size limit is NOT used (``sqrt(x) -> x`` near 0 would charge a spurious linear amount) — the
model returns ~0, not a spurious linear charge. Retained because the SAME code runs on thin Dukascopy
proxies where participation is non-trivial at larger size.
"""

from __future__ import annotations

import math

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import MicroState

__all__ = ["ImpactModel"]


class ImpactModel:
    """Square-root impact that vanishes at retail participation (§11 / Invariant I). Independent (Inv B)."""

    def participation(self, qty: float, micro: MicroState) -> float:
        """``|qty|·mid / volume_proxy`` — the order as a fraction of expected volume (§11)."""
        vol = micro.volume_proxy if micro.volume_proxy > 0.0 else 1.0
        return abs(qty) * micro.mid / vol

    def cost_bps(self, qty: float, micro: MicroState, cfg: ResolvedConfig) -> float:
        """``η · σ_h · sqrt(participation)`` in bps; ~0 as participation -> 0 (Invariant I)."""
        part = self.participation(qty, micro)
        if part <= 0.0:
            return 0.0
        return float(cfg.defaults.impact_eta * micro.sigma_horizon_bps * math.sqrt(part))
