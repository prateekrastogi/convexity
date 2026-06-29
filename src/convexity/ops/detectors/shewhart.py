"""Shewhart individuals chart (Shewhart 1931) — the blunt fast detector for single large excursions.

Spec: OPS §6.4, §6.7c. Alarms on one point past the limit:

    ALARM when  |z_i| > L_s

Shewhart catches *big* shifts in one sample (which CUSUM/EWMA are slower on) but is blind to *small
persistent* drifts (which CUSUM/EWMA catch) — the standard SPC pairing. Crucially, on the OPS streams
``L_s`` is the **EMPIRICAL / fitted-t tail quantile** of the in-control distribution, NOT the Gaussian
3σ (§6.7c): under a fat-tailed residual the Gaussian ``1/(2Φ(-L_s))`` *over-states* the ARL0 and the
chart floods. The Gaussian ``L_s=3 ⇒ 370`` is retained only as the thin-tailed reference; the
empirical quantile (set in the RED config from the in-control ``tail_quantile``) is authoritative.
O(1): stateless (depends only on the current standardized point), but still warms up so a transient
during the settling window does not page (§6.8).
"""

from __future__ import annotations

from convexity.ops.detectors.base import WarmupMixin, make_state
from convexity.ops.types import DetectorName, DetectorState


class Shewhart(WarmupMixin):
    """Shewhart individuals chart (§6.4). ``L_s`` is the §6.7c empirical-tail quantile. ``name`` is
    ``"shewhart"``."""

    name: DetectorName = "shewhart"

    def __init__(self, L_s: float, *, n_warm: int = 0):
        self.L_s = float(L_s)
        self._init_warmup(n_warm)

    def reset(self) -> None:
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """One §6.4 test on the standardized observation ``z`` (stateless apart from warm-up)."""
        warming = self._tick_warmup()
        stat = abs(z)
        breach = stat > self.L_s
        return make_state(
            detector=self.name,
            statistic=stat,
            limit=self.L_s,
            breach=breach,
            sample_idx=self.n_seen,
            warming_up=warming,
            diagnostics={"z": z},
        )
