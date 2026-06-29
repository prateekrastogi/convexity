"""Page-Hinkley mean-shift test (Hinkley 1971) — the independent confirmation on the cost stream.

Spec: OPS §6.2. A second mean-shift detector, used as the independent arm of the 2-of-2 cost
confirmation (§6.6a/§7.3) so a single chart's quirk does not fire a kill alone. It tracks the
cumulative deviation from the running mean minus a tolerance ``δ`` and alarms when the gap from the
running minimum exceeds ``λ``:

    m_T  = Σ_{i≤T} (z_i - δ)         cumulative (drift-tolerant) sum
    M_T  = min_{i≤T} m_i             running minimum
    PH_T = m_T - M_T
    ALARM (upward shift) when  PH_T > λ

The symmetric form tracks the running *maximum* for a downward shift. PH and CUSUM are
mathematically close (PH is a CUSUM variant against a running mean), which is exactly why their
alarms are *dependent* and the 2-of-2 joint ARL0 is a dependence-aware MC number, not the
independence product (§6.6a). O(1), causal: three scalar states.
"""

from __future__ import annotations

from convexity.ops.detectors.base import WarmupMixin, make_state
from convexity.ops.types import DetectorName, DetectorState


class PageHinkley(WarmupMixin):
    """Two-sided Page-Hinkley test (§6.2). ``name`` is ``"page_hinkley"``."""

    name: DetectorName = "page_hinkley"

    def __init__(self, delta: float, lam: float, *, n_warm: int = 0):
        self.delta = float(delta)
        self.lam = float(lam)
        self._init_warmup(n_warm)
        self._m_up = 0.0     # Σ (z - δ)   for the upward arm
        self._min_up = 0.0   # running min of m_up
        self._m_dn = 0.0     # Σ (z + δ)   for the downward arm
        self._max_dn = 0.0   # running max of m_dn

    def reset(self) -> None:
        self._m_up = 0.0
        self._min_up = 0.0
        self._m_dn = 0.0
        self._max_dn = 0.0
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """One §6.2 recursion step on the standardized observation ``z``."""
        warming = self._tick_warmup()
        self._m_up += z - self.delta
        self._min_up = min(self._min_up, self._m_up)
        ph_up = self._m_up - self._min_up

        self._m_dn += z + self.delta
        self._max_dn = max(self._max_dn, self._m_dn)
        ph_dn = self._max_dn - self._m_dn

        stat = max(ph_up, ph_dn)
        breach = stat > self.lam
        return make_state(
            detector=self.name,
            statistic=stat,
            limit=self.lam,
            breach=breach,
            sample_idx=self.n_seen,
            warming_up=warming,
            diagnostics={"ph_up": ph_up, "ph_dn": ph_dn},
        )

    @property
    def breached(self) -> bool:
        """Whether PH is currently above ``λ`` (used by the windowed AND-rule, §6.6a)."""
        return max(self._m_up - self._min_up, self._max_dn - self._m_dn) > self.lam
