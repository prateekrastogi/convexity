"""Synthetic weekly return series for LEVERAGE tests (LEVERAGE.md §9 layout).

Deterministic (seeded) generators of the three canonical regimes the pillar's tests and validation
harnesses exercise: a CALM series (steady low-vol drift), a DRAWDOWN series (a sustained
peak-to-trough decline that walks the drawdown governor down its ladder, §6.4), and a REGIME-BREAK
series (a mid-series vol/mean shift the short-half-life EWMA must track, §4.1/§6.3). These are
TEST FIXTURES only — a ``synth/`` module, never imported by a live layer — used so the governor,
vol-floor, and CUSUM tests run against explicit, reproducible series rather than ad-hoc inline
arrays (mirrors ``src/convexity``'s synthetic-panel test discipline).

NOTE: the CUSUM ARL Monte-Carlo (``validate/arl.py``) draws its own ``N(0,1)`` surprises directly
(the stated in-control model, §6.3); these generators produce book-RETURN series for the vol and
governor paths, not standardized surprises.
"""

from __future__ import annotations

import numpy as np


def calm_series(n: int = 260, *, mu: float = 0.02, sigma: float = 0.02, seed: int = 0) -> np.ndarray:
    """A steady low-vol weekly return series (LEVERAGE.md §4.1 — the EWMA tracks it cleanly).

    ``n`` i.i.d. draws ``N(mu, sigma)`` (weekly). At the cited ~2% weekly scale the EWMA ``σ̂_t``
    converges to ``sigma`` and no drawdown breakpoint is crossed, so the governor sits at ``m_t=1``.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, int(n))


def drawdown_series(
    n: int = 260,
    *,
    calm_mu: float = 0.02,
    calm_sigma: float = 0.02,
    dd_depth: float = 0.30,
    dd_weeks: int = 20,
    seed: int = 1,
) -> np.ndarray:
    """A calm run then a sustained decline of ~``dd_depth`` (LEVERAGE.md §6.4 governor ladder).

    Builds ``n − dd_weeks`` calm weeks, then ``dd_weeks`` negative weeks whose compounded effect is
    approximately a ``dd_depth`` peak-to-trough drawdown, so ``governor_state`` walks down the
    ladder toward its terminal breakpoint (§6.4). Used by the governor regression fixture that
    replays a drawdown path and checks the governed Sharpe does not collapse (the carry-sleeve-DD
    cautionary edge case, §8). Deterministic given ``seed``.
    """
    n = int(n)
    dd_weeks = int(dd_weeks)
    if not 0 < dd_weeks < n:
        raise ValueError("0 < dd_weeks < n required (LEVERAGE §6.4 synthetic drawdown)")
    if not 0.0 < dd_depth < 1.0:
        raise ValueError("dd_depth must be in (0, 1) (LEVERAGE §6.4)")
    rng = np.random.default_rng(seed)
    calm = rng.normal(calm_mu, calm_sigma, n - dd_weeks)
    # per-week decline that compounds to ~dd_depth over dd_weeks: (1+r)^dd_weeks = 1 - dd_depth.
    per_week = (1.0 - dd_depth) ** (1.0 / dd_weeks) - 1.0
    decline = np.full(dd_weeks, per_week) + rng.normal(0.0, calm_sigma * 0.25, dd_weeks)
    return np.concatenate([calm, decline])


def regime_break_series(
    n: int = 260,
    *,
    mu1: float = 0.02,
    sigma1: float = 0.02,
    mu2: float = 0.02,
    sigma2: float = 0.06,
    break_at: float = 0.5,
    seed: int = 2,
) -> np.ndarray:
    """A mid-series vol/mean shift the short-half-life EWMA must track (LEVERAGE.md §4.1, §6.3).

    First ``break_at·n`` weeks are ``N(mu1, sigma1)``; the rest ``N(mu2, sigma2)`` — a step change
    in the vol (and optionally mean) regime. ``σ̂_t``'s ``h_vol=8``-week half-life should follow the
    new regime within ~two half-lives, which is exactly why §6.3 keeps ``SE(SR_week,k)`` on a short
    half-life so the CUSUM standardizer is not stale (§6.3). Deterministic given ``seed``.
    """
    n = int(n)
    if not 0.0 < break_at < 1.0:
        raise ValueError("break_at must be in (0, 1) (LEVERAGE §4.1 regime break)")
    rng = np.random.default_rng(seed)
    k = int(round(break_at * n))
    first = rng.normal(mu1, sigma1, k)
    second = rng.normal(mu2, sigma2, n - k)
    return np.concatenate([first, second])
