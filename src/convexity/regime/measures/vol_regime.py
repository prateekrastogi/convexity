"""BTC realized volatility ``V_t`` (REGIME.md §2.3).

A single-asset weekly scalar: the annualized realized vol of BTC's own daily returns over the
rebalance week, computed directly from the panel close (no separate data source). ``V_t`` proxies
market-wide stress INDEPENDENT of the cross-section's breadth, so it stays informative in the sparse
early-history weeks where ``D_t`` (§2.2) is estimated over very few names (§2.3). Like ``D_t`` it is a
univariate scalar (§2.1), so it feeds its own single-statistic CUSUM without any covariance fit.
"""

from __future__ import annotations

import numpy as np

# Annualization factor for DAILY realized vol (REGIME.md §2.3: sqrt(365.25)). Distinct from
# src/convexity/system/metrics.PERIODS_PER_YEAR (52.18), which annualizes WEEKLY returns — V_t is built from
# the week's DAILY BTC returns, so the daily-to-annual factor sqrt(365.25) is the correct one (§2.3).
_DAILY_ANNUALIZE = float(np.sqrt(365.25))


def weekly_btc_vol(daily_returns: np.ndarray, btc_col: int, day_indices: np.ndarray) -> float:
    """Annualized weekly realized vol of BTC daily returns over rebalance week ``t`` (REGIME.md §2.3).

    ``V_t = std_{tau in week t}(r_{BTC,tau}) * sqrt(365.25)``. Uses the finite daily BTC returns within
    the week; returns ``NaN`` when fewer than two finite BTC returns are available (the week's RV is
    then undefined and the caller treats it as stale/insufficient -> WATCH, never HEALTHY, §0 P6)."""
    r = daily_returns[day_indices, btc_col]
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    return float(np.std(r, ddof=1) * _DAILY_ANNUALIZE)
