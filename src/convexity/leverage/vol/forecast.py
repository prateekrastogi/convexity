"""Forecast-volatility blend — Yang-Zhang + EWMA + HAR-RV, noise-robust, horizon-scaled.

LEVERAGE §15 (vol forecast), §8.1/§8.2, References §17. The single trailing-only
volatility estimate every ceiling consumes (``RiskInputs.forecast_vol``). Three
estimators blended:

  * Yang-Zhang (2000) OHLC volatility — drift-independent, open-jump-robust per-bar σ
    from O/H/L/C, the best low-frequency realized-vol estimator.
  * EWMA of squared returns — fast-reacting, captures the current vol level.
  * HAR-RV (Corsi 2009) — daily/weekly/monthly realized-vol cascade for persistence.

NOISE-ROBUST (Inv): at 1-min sampling, microstructure noise inflates plain RV, so the
realized-variance leg uses BIPOWER variation (Barndorff-Nielsen & Shephard) — a
jump/noise-robust estimator — guarding the HAR-RV inputs.

HORIZON-SCALED ONCE (Inv S): the blend returns a per-PERIOD σ; ``forecast_vol`` then
multiplies by √T exactly once for the holding horizon. The ceilings NEVER re-scale.

CAUSAL (Inv CAUSAL-L): every window is trailing-only (``[t-w, t]``), no centred
window, no full-sample standardization, no look-ahead. ``forecast_vol(bars, idx, ...)``
reads only bars at index ``<= idx``; shifting any future bar leaves the output at
``idx`` byte-identical.
"""

from __future__ import annotations

import numpy as np

# Default blend weights (BLUE-searchable at the class level, §4; fixed here as the
# validated starting blend). Sum to 1; HAR persistence + EWMA reactivity + YZ level.
_W_YZ: float = 0.4
_W_EWMA: float = 0.3
_W_HAR: float = 0.3
_EWMA_LAMBDA: float = 0.94  # RiskMetrics-style decay for the EWMA leg
_EPS: float = 1e-12


def yang_zhang_variance(o, h, l, c, *, k: float = 0.34) -> float:  # noqa: E741 (l = low, OHLC)
    """Yang-Zhang per-period variance over the trailing window (Yang-Zhang 2000).

    σ²_YZ = σ²_overnight + k·σ²_open-close + (1-k)·σ²_RS, with σ²_RS the
    Rogers-Satchell drift-independent estimator. All logs; trailing-only.
    """
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)  # noqa: E741
    c = np.asarray(c, dtype=float)
    n = o.size
    if n < 2:
        return 0.0
    # overnight (close-to-open) and open-to-close log returns
    log_oc = np.log(o[1:] / c[:-1])  # close_{t-1} -> open_t
    log_co = np.log(c[1:] / o[1:])  # open_t -> close_t
    var_o = float(np.var(log_oc, ddof=1)) if log_oc.size > 1 else 0.0
    var_c = float(np.var(log_co, ddof=1)) if log_co.size > 1 else 0.0
    # Rogers-Satchell on the same bars
    hh = np.log(h[1:] / c[1:])
    hl = np.log(h[1:] / o[1:])
    ll = np.log(l[1:] / c[1:])
    lo = np.log(l[1:] / o[1:])
    rs = float(np.mean(hh * hl + ll * lo)) if hh.size > 0 else 0.0
    rs = max(rs, 0.0)
    var_yz = var_o + k * var_c + (1.0 - k) * rs
    return max(var_yz, 0.0)


def ewma_variance(returns, *, lam: float = _EWMA_LAMBDA) -> float:
    """Trailing EWMA of squared log-returns (RiskMetrics). Causal: oldest gets least weight."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    # weights newest-heaviest: w_i ∝ lam^(age). Normalised so it is a proper average.
    ages = np.arange(r.size - 1, -1, -1, dtype=float)
    w = lam**ages
    w /= w.sum()
    return float(np.sum(w * r * r))


def bipower_variation(returns) -> float:
    """Bipower variation (Barndorff-Nielsen & Shephard) — jump/noise-robust realized
    variance: (π/2)·Σ|r_i||r_{i-1}|. Used at 1-min to guard the HAR-RV legs against
    microstructure noise + jumps inflating plain RV (LEVERAGE noise-robust requirement)."""
    r = np.abs(np.asarray(returns, dtype=float))
    if r.size < 2:
        return float(np.sum(r * r))  # degenerate: fall back to RV on a 1-element series
    mu1 = np.sqrt(2.0 / np.pi)
    return float((1.0 / (mu1 * mu1)) * np.sum(r[1:] * r[:-1]))


def har_rv_variance(returns, *, daily: int, weekly: int, monthly: int) -> float:
    """HAR-RV (Corsi 2009) realized-variance cascade using BIPOWER variation on three
    trailing horizons (daily/weekly/monthly), averaged. Noise-robust by construction.
    All windows trailing-only; shorter windows clamp to the available history."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0

    def bv_last(w: int) -> float:
        seg = r[-w:] if w <= r.size else r
        return bipower_variation(seg)

    rv_d = bv_last(daily)
    rv_w = bv_last(weekly)
    rv_m = bv_last(monthly)
    # equal-weighted cascade average (the HAR regression's stationary forecast proxy)
    return max((rv_d + rv_w + rv_m) / 3.0, 0.0)


def blend_period_vol(
    o, h, l, c,  # noqa: E741
    *,
    har_daily: int = 1,
    har_weekly: int = 5,
    har_monthly: int = 22,
    ewma_lambda: float = _EWMA_LAMBDA,
    w_yz: float = _W_YZ,
    w_ewma: float = _W_EWMA,
    w_har: float = _W_HAR,
) -> float:
    """Per-PERIOD forecast σ = √(blend of YZ, EWMA, HAR-RV variances). Trailing-only.

    Returns a single-period volatility (NOT horizon-scaled). ``har_*`` are window
    lengths in periods. Robust to short history (legs clamp). No NaN/inf on
    adversarial data — variances are floored at 0 and the result at ``_EPS``.
    """
    o = np.asarray(o, dtype=float)
    c = np.asarray(c, dtype=float)
    returns = np.log(c[1:] / c[:-1]) if c.size >= 2 else np.zeros(0)

    var_yz = yang_zhang_variance(o, h, l, c)
    var_ewma = ewma_variance(returns, lam=ewma_lambda)
    var_har = har_rv_variance(returns, daily=har_daily, weekly=har_weekly, monthly=har_monthly)

    total_w = w_yz + w_ewma + w_har
    var_blend = (w_yz * var_yz + w_ewma * var_ewma + w_har * var_har) / max(total_w, _EPS)
    return float(np.sqrt(max(var_blend, _EPS)))


def forecast_vol(
    o, h, l, c,  # noqa: E741
    *,
    horizon_periods: float,
    lookback: int | None = None,
    **blend_kwargs,
) -> float:
    """Horizon-scaled trailing forecast volatility σ·√T (Inv S, scaled ONCE here).

    Reads only the trailing ``lookback`` bars (causal, Inv CAUSAL-L; default: all
    supplied bars, which the caller has already sliced to ``<= t``). The √T scaling is
    applied exactly once — every downstream ceiling consumes this horizon σ as-is and
    NEVER re-scales (a daily-σ stop on a multiday hold is the bug Inv S forbids).
    """
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)  # noqa: E741
    c = np.asarray(c, dtype=float)
    if lookback is not None and c.size > lookback:
        o, h, l, c = o[-lookback:], h[-lookback:], l[-lookback:], c[-lookback:]  # noqa: E741
    sigma_period = blend_period_vol(o, h, l, c, **blend_kwargs)
    return float(sigma_period * np.sqrt(max(horizon_periods, 0.0)))
