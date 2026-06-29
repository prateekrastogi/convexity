"""Two-sided tabular CUSUM (Page 1954) + the Siegmund one-sided ARL seed (§6.6 cross-check).

Spec: OPS §6.1 (the recursion), §6.6 (the one-sided Siegmund closed form, retained as the
seed/cross-check — the MC two-sided value is authoritative). The chart detects a *persistent small
shift* in the mean by accumulating the signed deviation beyond a slack ``k`` (half the shift to
detect, in σ units), resetting at zero so it does not drift on noise:

    C+_i = max(0, C+_{i-1} + z_i - k)        upper (UPWARD shift)
    C-_i = max(0, C-_{i-1} - z_i - k)        lower (DOWNWARD shift)
    ALARM when  C+_i > h  OR  C-_i > h

A two-arm OR-chart: the in-control ARL0 the panel experiences is the two-arm combination
``1/ARL0_2sided ≈ 2/ARL0_oneside`` (§6.6) — ``(k=0.5,h=4.77)`` → one-sided ≈743 → two-sided ≈370.
O(1): two scalar states, one update per observation; ``(k,h)`` frozen from the validated ``(μ0,σ0)``
+ ARL target (Invariant LIM). ``escalation_h`` (``h2``) is the §7.3/§10 upper limit on ``C+`` that
escalates DEGRADE→HALT; the detector reports both crossings in ``diagnostics``.
"""

from __future__ import annotations

import math

from convexity.ops.detectors.base import WarmupMixin, make_state
from convexity.ops.types import DetectorName, DetectorState


class Cusum(WarmupMixin):
    """Two-sided tabular CUSUM (§6.1). ``name`` is ``"cusum"`` for the forensic chain."""

    name: DetectorName = "cusum"

    def __init__(self, k: float, h: float, *, n_warm: int = 0, escalation_h: float | None = None):
        self.k = float(k)
        self.h = float(h)
        self.escalation_h = float(escalation_h) if escalation_h is not None else None
        self._init_warmup(n_warm)
        self.c_plus = 0.0
        self.c_minus = 0.0

    def reset(self) -> None:
        self.c_plus = 0.0
        self.c_minus = 0.0
        self._reset_warmup()

    def update(self, z: float) -> DetectorState:
        """One §6.1 recursion step on the standardized observation ``z``."""
        warming = self._tick_warmup()
        self.c_plus = max(0.0, self.c_plus + z - self.k)
        self.c_minus = max(0.0, self.c_minus - z - self.k)
        stat = max(self.c_plus, self.c_minus)
        breach = stat > self.h
        diagnostics = {"c_plus": self.c_plus, "c_minus": self.c_minus}
        if self.escalation_h is not None:
            diagnostics["escalated"] = float(self.c_plus >= self.escalation_h
                                             or self.c_minus >= self.escalation_h)
        return make_state(
            detector=self.name,
            statistic=stat,
            limit=self.h,
            breach=breach,
            sample_idx=self.n_seen,
            warming_up=warming,
            diagnostics=diagnostics,
        )

    @property
    def escalated(self) -> bool:
        """Whether either arm has crossed ``escalation_h`` (the §7.3/§10 HALT escalation)."""
        if self.escalation_h is None:
            return False
        return self.c_plus >= self.escalation_h or self.c_minus >= self.escalation_h


def siegmund_arl_oneside(k: float, h: float, shift: float) -> float:
    """Siegmund's (1985) one-sided CUSUM ARL approximation (§6.6) — the SEED / cross-check, NOT
    authoritative.

    ``ARL_oneside ≈ (exp(-2·Δ·(h+1.166)) + 2·Δ·(h+1.166) - 1) / (2·Δ²)`` with ``Δ = shift - k`` in σ
    units. ``shift = 0`` in control gives the per-arm ARL0 (``Δ = -k``); ``shift = δ`` out of control
    gives the per-arm ARL1. The two-sided number the chart experiences is the §6.6 combination
    :func:`two_sided_arl0_from_oneside` (``≈ ARL0_oneside / 2`` for a symmetric chart). The MC solver
    (``validate/arl.py``) is authoritative wherever it disagrees (§6.6, Invariant ARL).
    """
    delta = shift - k
    if abs(delta) < 1e-9:
        # Δ→0 limit: the ratio ARL → (h + 1.166)² (l'Hôpital on the Siegmund form).
        return (h + 1.166) ** 2
    b = h + 1.166
    return (math.exp(-2.0 * delta * b) + 2.0 * delta * b - 1.0) / (2.0 * delta * delta)


def two_sided_arl0_from_oneside(arl0_oneside: float) -> float:
    """Combine the two near-independent arms (§6.6): ``1/ARL0_2sided ≈ 2/ARL0_oneside``.

    For a symmetric chart this is ``ARL0_oneside / 2`` — the load-bearing ``÷2`` (§6.1/§6.6). It is a
    mildly CONSERVATIVE seed (the arms are weakly *negatively* associated in control, §6.6(a)); the
    pinned ``(k,h)`` is solved against the dependence-aware MC two-sided ARL0 (``validate/arl.py``),
    not this shortcut.
    """
    return arl0_oneside / 2.0
