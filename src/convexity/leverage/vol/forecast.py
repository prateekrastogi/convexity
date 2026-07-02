"""Weekly EWMA book-volatility estimator ``σ̂_t`` (LEVERAGE.md §4).

``σ̂_t`` is a *causal* EWMA of the book's own realized weekly 100%-gross reference return, on the
WEEKLY clock (§4.1) — the same closed-form RiskMetrics recursion COST §5.5a uses on the
funding-stamp clock, re-derived here on the weekly-return clock (Invariant NO-DUP-EWMA: same FORM,
separately parameterized, no shared mutable half-life). It is a UNIVARIATE EWMA on ONE scalar
series — never an N×N cross-sectional covariance (§4.3, Invariant NO-MV).

Below the event-count floor ``n_vol_min`` the estimate falls back CONSERVATIVELY (higher assumed
vol, never lower) to the ~28%-annualized class prior, then a wider cold-start multiple (§4.2,
Invariant VOL-FLOOR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.leverage.config.defaults import PERIODS_PER_YEAR, LeverageDefaults
from convexity.leverage.types import annualized_to_weekly_vol


@dataclass(frozen=True)
class VolForecast:
    """OUTPUT of :func:`sigma_hat_t` (LEVERAGE.md §4.1).

    ``sigma_hat`` is the per-week (per-observation) vol ``σ̂_t``; ``ess`` the EWMA effective sample
    size ``(2−α)/α``; ``n_obs`` the count of realized weekly returns seen; ``on_floor`` True when
    the estimate came from the §4.2 conservative fallback (history shorter than ``n_vol_min``).
    """

    sigma_hat: float
    ess: float
    alpha: float
    n_obs: int
    on_floor: bool


def _ewma_vol(returns: np.ndarray, alpha: float) -> float:
    """Causal EWMA realized-vol recursion (LEVERAGE.md §4.1).

    ``m_k = α·R_k + (1−α)·m_{k−1}``; ``v_k = α·(R_k − m_{k−1})² + (1−α)·v_{k−1}``; ``σ̂ = √v_k`` at
    the most recent completed week. The squared-deviation uses the PRIOR mean ``m_{k−1}`` (causal —
    the RiskMetrics convention, no lookahead within the update). Seeded from the first observation.
    """
    r = np.asarray(returns, dtype=float)
    m = r[0]
    v = 0.0
    for k in range(1, r.size):
        dev = r[k] - m
        v = alpha * dev * dev + (1.0 - alpha) * v
        m = alpha * r[k] + (1.0 - alpha) * m
    return float(np.sqrt(v))


def sigma_hat_t(
    net_weekly: np.ndarray,
    defaults: LeverageDefaults,
    *,
    periods_per_year: float | None = None,
) -> VolForecast:
    """Causal EWMA weekly book-vol ``σ̂_t`` with the §4.2 conservative floor (LEVERAGE.md §4).

    ``net_weekly`` is the book's realized 100%-gross reference weekly ``net`` series
    (``research/xsec/engine.XSecResult.net``), most-recent-last. ``α = 1 − 2^(−1/h_vol)`` with
    half-life ``h_vol`` in WEEKS (§4.1). Below ``n_vol_min`` observations the estimate falls back,
    in priority order (§4.2):

    1. the class-level ~28%-annualized prior (``vol_prior_annual``), converted to weekly;
    2. on a cold start (zero history) a ``vol_cold_start_multiple`` × that prior, so a cold book
       starts UNDER-levered, never over-levered.

    The fallback is always ≥ the prior (Invariant VOL-FLOOR): thin history WIDENS, never narrows,
    the assumed vol. The returned ``sigma_hat`` is guaranteed ``> 0`` and finite, so the downstream
    Kelly term ``SR_LCB_t / σ̂_t`` never divides by zero/NaN (§8 edge case).
    """
    ppy = float(periods_per_year) if periods_per_year is not None else PERIODS_PER_YEAR
    r = np.asarray(net_weekly, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    alpha = 1.0 - 2.0 ** (-1.0 / defaults.h_vol)
    ess = (2.0 - alpha) / alpha
    prior_weekly = annualized_to_weekly_vol(defaults.vol_prior_annual, ppy)

    if n == 0:
        # cold start: no history at all -> a deliberately wide multiple of the prior (§4.2 step 2).
        return VolForecast(
            sigma_hat=prior_weekly * defaults.vol_cold_start_multiple,
            ess=ess, alpha=alpha, n_obs=0, on_floor=True,
        )
    if n < defaults.n_vol_min:
        # thin (but non-empty) history: the class prior itself (§4.2 step 1). Conservative floor —
        # never the (noisier, typically smaller) EWMA of too-few points.
        return VolForecast(sigma_hat=prior_weekly, ess=ess, alpha=alpha, n_obs=n, on_floor=True)

    est = _ewma_vol(r, alpha)
    if not np.isfinite(est) or est <= 0.0:
        # a degenerate (flat/corrupted) history still floors to the prior, never zero (§8, VOL-FLOOR).
        return VolForecast(sigma_hat=prior_weekly, ess=ess, alpha=alpha, n_obs=n, on_floor=True)
    return VolForecast(sigma_hat=est, ess=ess, alpha=alpha, n_obs=n, on_floor=False)
