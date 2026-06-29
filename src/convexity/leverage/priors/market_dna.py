"""MarketDNA — slow monthly/quarterly priors: long-run vol + gap baseline (LEVERAGE §3 priors).

SLOW, causal (trailing-only, Inv CAUSAL-L) priors that anchor the TEAL fits: a long-run
volatility level and a baseline gap magnitude per instrument, refreshed monthly/quarterly
rather than per-bar. These are PRIORS that the fast forecast-vol blend and the gap model
shrink toward when their fast samples are thin — never a hand knob, never P&L-fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketDNA:
    """Slow per-instrument priors (§3 priors). All trailing-derived, refreshed slowly."""

    long_run_vol: float  # long-run realized volatility level (per-period, unscaled)
    gap_baseline: float  # baseline adverse-gap magnitude (fraction of price)
    n_obs: int


def fit_market_dna(returns: np.ndarray, gaps: np.ndarray, *, min_obs: int = 60) -> MarketDNA:
    """Fit the slow priors from a long trailing window of returns + observed gaps.

    ``long_run_vol`` is the long-window realized std; ``gap_baseline`` is the trailing
    median absolute gap. Below ``min_obs`` the fit is flagged via ``n_obs`` so the caller
    keeps a conservative prior rather than trusting a thin estimate (Inv CAUSAL-L)."""
    r = np.asarray(returns, dtype=float)
    g = np.abs(np.asarray(gaps, dtype=float))
    lr_vol = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    gap_base = float(np.median(g)) if g.size else 0.0
    return MarketDNA(long_run_vol=lr_vol, gap_baseline=gap_base, n_obs=int(min(r.size, g.size)))
