"""CORE — the shared cross-pillar contracts (types / constants / the premium-at-risk sizing base).

This is the ONE definition, one unit per crossing value that the pillar specs reference across
boundaries (`docs/specs/{SIGNAL,SYSTEM,PORTFOLIO,LEVERAGE,OPS}.md`). It holds only what >= 2 pillars
share; anything a single pillar owns stays in that pillar (UNIVERSE's Symbol/Expiry/ExpiryCalendar
deliberately do NOT live here yet -- UNIVERSE.md §8: promote only when a second consumer appears).

What lives here and why (each named by a spec as a shared contract):

- ``PERIODS_PER_YEAR`` (= 252) and ``DDOF`` (= 0): the annualization constant (today ``ANN`` in
  ``signal/orb.py``) and the population-std convention that ``metrics`` uses and SYSTEM/PORTFOLIO/
  REGIME reuse VERBATIM. SYSTEM.md §7 pins these as core-owned so no consumer re-declares a second,
  drifting value (the convexity "one definition per crossing value" rule). ``signal/orb.py`` keeps
  its ``ANN`` name unchanged (the 92 tests pin it); these are the shared re-export point for the
  Phase-B pillars, equal by construction (asserted in tests).

- ``TradeRecord``: the atomic unit SIGNAL emits, PORTFOLIO pools, SYSTEM splits/bootstraps, and
  LEVERAGE sizes -- ``(exit_date: date, net_return: float)`` where ``net_return`` is a fractional
  return on option **premium-at-risk, already net of the flat round-trip ``spread``** (SIGNAL.md §1).
  It is a ``NamedTuple`` (a real ``tuple`` subclass), so it is a TRANSPARENT drop-in for the plain
  2-tuples ``orb.run`` already emits: it unpacks as ``for _, r in trades`` and indexes as ``t[0]``
  exactly like ``metrics`` (``signal/orb.py``) expects -- no existing behavior changes. The named
  fields are for the Phase-B pillars that read the stream by contract; the 2-tuple shape is fixed
  here and honored unchanged across every boundary.

- ``premium_at_risk_inr(qty, entry_premium)``: the premium-at-risk sizing BASE agreed by LEVERAGE /
  EXECUTION (and consumed by the OPS INR-P&L hand-off, OPS.md §10; PAPER scales returns through it).
  A long option's max loss IS its premium (the gap-proof bound, LEVERAGE.md §2), so premium-at-risk
  in rupees is ``qty * entry_premium`` -- and ``qty = lots * lot_size`` is ALREADY total units
  (COST §1 / EXECUTION §2), so ``lot_size`` enters EXACTLY ONCE, via ``qty``, never twice. This is
  precisely COST's rupee turnover ``premium * qty``. It is the single-owner definition so the
  fraction->rupee scaling can never silently disappear or double-count ``lot_size`` (OPS.md §10).

Pure Python + stdlib only (no pandas/numpy import here) -- constants and small helpers, no I/O, no
pricing model, no greeks (moneyness only; CLAUDE.md). Map: ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import datetime as dt
from typing import NamedTuple

__all__ = [
    "PERIODS_PER_YEAR",
    "DDOF",
    "TradeRecord",
    "ANN",
    "metrics",
]

# --- Shared annualization / dispersion conventions (SYSTEM.md §7 -- ONE definition) -----------------
# Trading days per year: the annualizer denominator `sqrt(n / (n_days / PERIODS_PER_YEAR))` used by
# `signal/orb.py::metrics` (as `ANN`) and reused unchanged by SYSTEM/PORTFOLIO/REGIME.
import math

import numpy as np

PERIODS_PER_YEAR = 252

# Population standard deviation (numpy's default) -- the per-trade Sharpe/std convention pinned across
# SIGNAL/SYSTEM/PORTFOLIO/REGIME so a display Sharpe and any recovered per-observation Sharpe share a
# `ddof`. `ddof=0` throughout (SYSTEM.md §4, PORTFOLIO.md §5, SIGNAL.md §11).
DDOF = 0


# --- The atomic trade record (SIGNAL.md §1 -- the unit every downstream pillar consumes) -----------
class TradeRecord(NamedTuple):
    """One closed trade: ``(exit_date, net_return)`` -- SIGNAL's atomic unit (SIGNAL.md §1).

    ``net_return`` is a per-trade fractional return on the option **premium-at-risk**, already NET of
    the flat round-trip ``spread`` (SIGNAL.md §6); a clean stop touch is exactly ``-stop`` and a
    total loss floors at ``-1`` (premium is the max loss). ``exit_date`` is SIGNAL's *scheduled-
    horizon* exit date (``days[end]``), NOT the physical stop day -- the documented SIGNAL.md §5
    simplification that PORTFOLIO/SYSTEM/REGIME calendar-bucketing inherits verbatim.

    Being a ``NamedTuple`` it IS a plain 2-tuple: it unpacks as ``for _, r in trades`` and indexes as
    ``t[0]`` / ``t[1]`` exactly like the bare tuples ``orb.run`` emits and ``orb.metrics`` unpacks --
    so it is a transparent, backward-compatible drop-in, never a new contract shape. Downstream
    pillars may build/read it by field name (``.exit_date`` / ``.net_return``) without changing the
    wire shape.
    """

    exit_date: dt.date
    net_return: float


# --- The premium-at-risk sizing base (LEVERAGE/EXECUTION own it; OPS.md §10 hand-off) --------------


# --- carried VERBATIM from intraday signal/orb.py::metrics (per-trade stats;
# instrument-agnostic: n/win/ev/sharpe/final_eq/max_dd + drop-top-N tells). The moved
# validation tests pin their identities against THIS function.
ANN = 252


def metrics(trades, n_days, size=0.20):
    """Per-trade stats + compounded equity (fixed `size` of capital per trade) + max drawdown."""
    a = np.array([r for _, r in trades])
    if len(a) == 0:
        return None
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in a:
        eq *= 1 + size * r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    srt = np.sort(a)[::-1]
    asc = srt[::-1]
    tot = a.sum() if a.sum() != 0 else 1e-9
    return {
        "n": len(a), "win": 100 * (a > 0).mean(), "ev": a.mean() * 100,
        "sharpe": (a.mean() / a.std() * math.sqrt(len(a) / (n_days / ANN))) if a.std() > 0 else 0.0,
        "final_eq": eq, "max_dd": mdd * 100,
        # fat-tail / winner-concentration: top-N share of total PnL, and EV/trade after dropping the top-N winners
        "top1_pnl": 100 * srt[0] / tot, "top3_pnl": 100 * srt[:3].sum() / tot,
        "ev_drop1": asc[:-1].mean() * 100 if len(a) > 1 else 0.0,
        "ev_drop3": asc[:-3].mean() * 100 if len(a) > 3 else 0.0,
    }
