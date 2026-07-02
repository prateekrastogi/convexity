"""Drawdown governor ladder ``m_t`` (LEVERAGE.md §6.4).

Kept from the pre-pivot spec and re-derived on the NEW risk unit (book equity peak-to-trough,
computed from the SAME weekly ``net`` series everything else in this file uses — never a per-trade
P&L series, §6.4). ``m_t`` is a monotone NON-INCREASING step ladder in the running drawdown
``dd_t``, anchored so every breakpoint is a stated multiple (``0.5×/1.0×/1.5×/1.6×``) of the cited
~25% max-DD-at-100%-gross figure (§6.4), reaching ``m_t → 0`` at the RED terminal threshold.

The ladder is deliberately anchored AT/BEYOND the observed drawdown scale, not inside it, to avoid
reproducing the EMPIRICAL BASELINE's carry-sleeve DD-stop failure mode (a too-tight stop collapsed
carry's Sharpe from ~0.57 to ~0.10, §8 cautionary edge case).
"""

from __future__ import annotations

import numpy as np

from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.types import GovernorState


def _drawdown_from_equity(net_weekly: np.ndarray) -> float:
    """Running peak-to-trough book-equity drawdown ``dd_t = 1 − E_t / peak(E_{≤t})`` (LEVERAGE.md §6.4).

    Book equity ``E_t = E_0 · ∏(1 + R_s)`` compounded from the weekly ``net`` series (E_0 = 1), the
    SAME series the Kelly/vol terms read (§6.4). Returns the drawdown at the most recent week ``t``.
    """
    r = np.asarray(net_weekly, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity[-1] / peak[-1]
    return float(max(0.0, dd))


def ladder(dd_t: float, defaults: LeverageDefaults) -> float:
    """The §6.4 monotone-non-increasing step ladder ``m_t = ladder(dd_t)`` (LEVERAGE.md §6.4).

    Steps through ``defaults.governor_ladder`` rows ``(dd_upper_exclusive, m_t)`` in ascending
    order; the first row whose upper bound the drawdown is BELOW gives ``m_t`` (§6.4). At/above the
    terminal breakpoint ``m_t = 0`` (full flat — governor kill). ``m_t ∈ [0, 1]`` always, and is
    non-increasing in ``dd_t`` by construction of the ladder table (Invariant checked in tests).
    """
    dd = max(0.0, float(dd_t))
    for upper, m in defaults.governor_ladder:
        if dd < upper:
            return float(m)
    return 0.0  # past the last (terminal) breakpoint -> full flat


def governor_state(net_weekly: np.ndarray, defaults: LeverageDefaults) -> GovernorState:
    """Governor state ``(dd_t, m_t, terminal)`` from the weekly book-equity path (LEVERAGE.md §6.4).

    ``terminal`` is True once ``dd_t ≥ dd_terminal`` — the RED terminal threshold where the ladder
    kills the book (``m_t = 0``). Reads BOOK equity from the weekly ``net`` series (§6.4), never a
    per-trade series.
    """
    dd = _drawdown_from_equity(net_weekly)
    m = ladder(dd, defaults)
    return GovernorState(dd_t=dd, m_t=m, terminal=dd >= defaults.dd_terminal)
