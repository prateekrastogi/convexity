"""Forecast baselines (COST §12). Firewalled — never imported by live layers (Invariant V).

The four required baselines, all under identical splits/marks/embargo:
  * flat-bps cost — a constant round-trip bps with no funding (the naive stub COST replaces);
  * no-funding — fees+spread+impact only;
  * last-funding-only — forecast funding = last realized rate held flat (the bar the
    EWMA+basis blend must beat);
  * perfect-foresight bound — uses REALIZED FUTURE funding (future-only, firewalled) as an
    UPPER bound on how good any causal forecast could be (never a live path).
"""

from __future__ import annotations

import numpy as np


def flat_bps_cost(notional: float, flat_bps: float) -> float:
    """A constant round-trip bps cost, no funding (§12 flat-bps baseline)."""
    return (flat_bps / 1e4) * notional


def no_funding_cost(fees: float, half_spread: float, impact: float) -> float:
    """Fees+spread+impact only — isolates the funding contribution (§12 no-funding)."""
    return fees + half_spread + impact


def last_funding_only_forecast(side: int, notional: float, last_rate: float, m: int) -> float:
    """Forecast funding = last realized rate held flat over ``m`` stamps (§12 last-funding-only)."""
    return side * notional * last_rate * m


def perfect_foresight_funding(side: int, notionals: np.ndarray, future_rates: np.ndarray) -> float:
    """The realized-future funding sum — the UPPER bound on any causal forecast (§12).

    FUTURE-ONLY (firewalled): consumes realized future rates. Never a live path; only
    the gap from a causal forecast to this bound is reportable.
    """
    notionals = np.asarray(notionals, dtype=float)
    future_rates = np.asarray(future_rates, dtype=float)
    return float(side * np.sum(notionals * future_rates))
