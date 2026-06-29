"""EWMA control chart (Roberts 1959; Lucas & Saccucci 1990 ARL tables).

Spec: OPS §6.3. Detects a *sustained small shift with memory* — the right tool for autocorrelated
streams (vol error, fill IS) where a single-point Shewhart chart is insensitive:

    S_i = λ·z_i + (1-λ)·S_{i-1}          S_0 = 0  (= in-control mean of the standardized stream)
    control limit (full):  ± L·sqrt( λ/(2-λ) · (1-(1-λ)^{2i}) )
    control limit (asym):  ± L·sqrt( λ/(2-λ) )
    ALARM when |S_i| > the limit

The transient term ``(1-(1-λ)^{2i})`` matters only for the first few samples (one of the reasons a
detector spends a governed warm-up ``N_warm`` before it is trusted, §6.8); the asymptotic limit
governs steady state. ``(λ=0.10, L=2.70)`` gives in-control ARL0 ≈ 370, ARL1@1σ ≈ 10. O(1): one
scalar state; the limit is a closed form in ``(λ, L)`` — no live fit (Invariant LIM).
"""

from __future__ import annotations

import math

from convexity.ops.detectors.base import WarmupMixin, make_state
from convexity.ops.types import DetectorName, DetectorState


class Ewma(WarmupMixin):
    """EWMA individuals control chart (§6.3). ``name`` is ``"ewma"``.

    ``use_transient_limit`` widens the early limit by the ``(1-(1-λ)^{2i})`` factor per the full
    closed form; the default uses the asymptotic limit (the steady-state limit warm-up settles into).
    """

    name: DetectorName = "ewma"

    def __init__(self, lam: float, L: float, *, n_warm: int = 0, use_transient_limit: bool = True):
        self.lam = float(lam)
        self.L = float(L)
        self.use_transient_limit = bool(use_transient_limit)
        self._init_warmup(n_warm)
        self.s = 0.0
        self._asym = self.L * math.sqrt(self.lam / (2.0 - self.lam))

    def reset(self) -> None:
        self.s = 0.0
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """One §6.3 recursion step on the standardized observation ``z``."""
        warming = self._tick_warmup()
        self.s = self.lam * z + (1.0 - self.lam) * self.s
        i = self.n_seen
        if self.use_transient_limit:
            transient = 1.0 - (1.0 - self.lam) ** (2 * i)
            limit = self.L * math.sqrt(self.lam / (2.0 - self.lam) * transient)
        else:
            limit = self._asym
        breach = abs(self.s) > limit
        return make_state(
            detector=self.name,
            statistic=abs(self.s),
            limit=limit,
            breach=breach,
            sample_idx=i,
            warming_up=warming,
            diagnostics={"s": self.s, "asymptotic_limit": self._asym},
        )

    @property
    def asymptotic_limit(self) -> float:
        """The steady-state control limit ``L·sqrt(λ/(2-λ))`` (§6.3)."""
        return self._asym
