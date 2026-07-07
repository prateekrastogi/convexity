"""
Axis B -- Entry/exit as optimal control under transaction costs.

Replaces an arbitrary fixed hysteresis band with a DERIVED no-trade band (Davis & Norman 1990 /
Harrison & Taksar 1983 singular stochastic control, closed form for a quadratic-holding-cost /
proportional-trading-cost Brownian-noise reduction) combined with a DERIVED partial-adjustment
speed toward the target once outside the band (Garleanu & Pedersen 2013 continuous-time
aim-portfolio control). See the accompanying markdown section ("Axis B: Entry/Exit as Optimal
Control Under Transaction Costs") for the full derivation, literature citations, and the
explicit connection between the two frameworks.

INTERFACE CONTRACT (coordination point with Axis A / the Sweep integration agent)
----------------------------------------------------------------------------------
This module consumes:
  target : pd.Series
      A continuous "aim portfolio" / trend-strength signal in [-1, 1] (Axis A's output),
      indexed by calendar day, ALREADY decide-at-close/act-next-day lagged per bug pattern
      7(c) of the program-wide instructions -- i.e. target.iloc[t] is the value legitimately
      knowable and usable for the trade DECIDED at t (using information through t-1's close
      only). This module does NOT re-lag `target`; it treats it as a causal input and layers
      an *additional* transaction-cost control on top of it. It performs its own internal
      lagging only for the volatility estimates it computes itself (see realized_vol /
      realized_return_vol below) -- those are proven causal by injection test in
      validate_entry_exit_control.py.
  price : pd.Series
      Daily closes (e.g. i1_pipeline.daily_close_series_duka(load_stitched_duka(symbol))),
      same index as `target`, used only to estimate the underlying's realized return
      volatility (the k = gamma * sigma_r^2 holding-cost curvature term below).

It produces:
  exposure : pd.Series
      The controlled, cost-aware position x_t actually held from t to t+1.

Everything else (the actual P&L/cost accounting, e.g. i1_pipeline.apply_costs) is the Sweep
phase's job -- this module only decides WHAT exposure to hold, not how P&L is computed from it.

COST CONVENTION (matches research/indices/i1_pipeline.py exactly -- RT_PRIMARY_BP=13.0 applied
directly to |delta_exposure| per day; a full round trip 0->1->0 sums |delta_exposure| to 2.0 and
therefore costs 2*13bp=26bp total. The "13bp" IS the per-unit-|delta_exposure| rate, not half of
it.) `cost_bps` below is this same rate, in basis points; theta (the proportional cost rate in
the derivation) = cost_bps / 1e4 directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ======================================================================================
# Causal volatility estimators (bug pattern 7c: decide-at-close t-1, act at t)
# ======================================================================================
def realized_vol(series: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series:
    """Causal trailing realized volatility of the FIRST DIFFERENCE of `series` (appropriate for
    an already-bounded signal such as a trend-strength score in [-1,1]).

    value at row t = std(diff(series)) over the `window` observations ending at t-1, i.e. computed
    using information through t-1's close only, then reported at index t (shift(1)). No value at
    row t ever uses series[t] or later. See validate_entry_exit_control.py for the injection-test
    proof (including a naive/unlagged control that DOES leak, showing the test has teeth).
    """
    d = series.diff()
    rv = d.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
    return rv.shift(1)


def realized_vol_naive_UNLAGGED(series: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series:
    """NAIVE CONTROL ONLY -- not for production use. Same as realized_vol() but WITHOUT the
    causal shift(1), i.e. the value reported at row t uses series[t] itself (same-day leak).
    Exists solely to prove the injection test used to validate realized_vol() actually has
    discriminating power (a test that can't tell this apart from the real function is not a real
    causality test)."""
    d = series.diff()
    return d.rolling(window=window, min_periods=min_periods).std(ddof=ddof)


def realized_return_vol(price: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series:
    """Causal trailing realized volatility of log returns of `price`. See realized_vol() for the
    exact causality convention (shift(1) applied identically here)."""
    logret = np.log(price / price.shift(1))
    rv = logret.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
    return rv.shift(1)


# ======================================================================================
# Derived (not guessed) control parameters
# ======================================================================================
def no_trade_band_width(theta: "float | pd.Series", k_curvature: "float | pd.Series",
                         sigma_z: "float | pd.Series", floor: float = 1e-10) -> "float | pd.Series":
    """Davis-Norman / Harrison-Taksar closed-form no-trade half-width:

        Delta* = ( 3 * theta * sigma_z^2 / (2 * k) )^(1/3)

    Derived via smooth-pasting (value-matching + second-order smooth-fit) on the long-run-average
    HJB variational inequality for a singular-control problem with instantaneous quadratic holding
    cost (k/2)*z^2 (z = tracking error x - x*) and proportional trading cost theta per unit |dz|
    traded, z's uncontrolled dynamics Brownian with instantaneous variance sigma_z^2. Full
    derivation in the markdown section. Exact for this idealized LQ/proportional-cost reduction;
    matches the well-known theta^(1/3) scaling law of Constantinides (1986) / Davis & Norman
    (1990), made rigorous via viscosity solutions in Shreve & Soner (1994).

    theta       : proportional (one-unit-of-|delta_exposure|) cost rate, e.g. cost_bps/1e4.
    k_curvature : gamma * sigma_r^2, the quadratic risk/tracking-cost curvature
                  (gamma = risk-aversion scale, sigma_r = per-period realized return vol).
    sigma_z     : per-period realized vol of the *signal* x* (driver of tracking-error noise).
    """
    k = np.maximum(k_curvature, floor)
    num = 3.0 * theta * np.square(sigma_z)
    return np.cbrt(num / (2.0 * k))


def gp_adjustment_speed(k_curvature: "float | pd.Series", theta: "float | pd.Series",
                         band_width: "float | pd.Series", floor: float = 1e-12) -> "float | pd.Series":
    """Garleanu & Pedersen (2013) continuous-time partial-adjustment ("aim portfolio") speed:

        kappa = sqrt( k / lambda_eff )

    derived from the HJB first-order condition u* = V_x / lambda (see markdown section for the
    full value-function guess-and-verify solving the quadratic; this is the beta->0, no-discount
    limit, appropriate at the daily-decision horizons used here since beta << the daily dynamics).

    lambda_eff (the *quadratic* trading-cost coefficient GP's framework requires) is not directly
    observable -- the venue's real cost is PROPORTIONAL (theta per unit |delta_exposure|), not
    quadratic. We calibrate lambda_eff by matching MARGINAL cost continuity at the no-trade band's
    free boundary (band_width, derived above): the Davis-Norman branch's marginal cost there is
    exactly theta (by construction of the smooth-pasting condition); a quadratic-cost model's
    marginal cost at trade size = band_width is lambda_eff * band_width. Setting these equal:

        lambda_eff = theta / band_width

    This is the same "two views of one control problem" connection made explicit: the SAME
    holding-cost curvature k = gamma*sigma_r^2 feeds both branches, and lambda_eff ties the
    quadratic branch's aggressiveness to the proportional-cost branch's own derived band, so no
    free/arbitrary quadratic-cost parameter is introduced.

    Returns kappa as a PER-PERIOD RATE (not yet a [0,1] update fraction -- see
    kappa_to_daily_fraction()).
    """
    lam_eff = theta / np.maximum(band_width, floor)
    return np.sqrt(np.maximum(k_curvature, floor) / np.maximum(lam_eff, floor))


def kappa_to_daily_fraction(kappa: "float | np.ndarray", cap: float = 1.0) -> "float | np.ndarray":
    """Exact discretization of the continuous-time OU convergence dx = kappa*(x*-x)dt over one
    day (holding x* fixed over the interval): fraction = 1 - exp(-kappa). Capped at `cap`
    (default 1.0: never overshoot past the target in a single day)."""
    frac = 1.0 - np.exp(-np.asarray(kappa, dtype=float))
    return np.minimum(frac, cap)


# ======================================================================================
# Declared Stage-A sweep grid (cost-sensitivity x band-scale)
# ======================================================================================
# `cost_sensitivity` (zeta): multiplies theta ONLY inside the control's own calibration of
#   band_width/kappa (never the real P&L cost charged downstream by apply_costs) -- a robustness/
#   aggressiveness knob. zeta > 1 acts as if trading were more expensive than it really is
#   (wider band, slower speed, conservative); zeta < 1 acts as if cheaper (narrower band, faster
#   speed, aggressive).
# `band_scale` (eta): direct multiplier on the theoretically-derived Delta*, testing over/under-
#   trading relative to the closed-form optimum.
DEFAULT_GRID = [
    {"name": "conservative", "cost_sensitivity": 2.0, "band_scale": 1.5},
    {"name": "derived_baseline", "cost_sensitivity": 1.0, "band_scale": 1.0},
    {"name": "moderate_aggressive", "cost_sensitivity": 0.5, "band_scale": 0.75},
    {"name": "aggressive", "cost_sensitivity": 0.25, "band_scale": 0.5},
]


# ======================================================================================
# Main entry point
# ======================================================================================
def partial_adjustment_control(
    target: pd.Series,
    price: pd.Series,
    cost_bps: float = 13.0,
    gamma: float = 1.0,
    cost_sensitivity: float = 1.0,
    band_scale: float = 1.0,
    vol_window: int = 20,
    min_periods: int = 20,
    band_floor: float = 0.0,
    band_cap: float = 0.5,
    kappa_cap: float = 1.0,
) -> pd.DataFrame:
    """
    Turn a frictionless aim-portfolio trend-strength signal `target` (in [-1,1]) into an actual,
    cost-aware exposure path via the hybrid Davis-Norman no-trade-band + Garleanu-Pedersen
    partial-adjustment rule derived in the markdown section.

    Decision rule at each t (all of band_t, kappa_t computed from data through t-1 ONLY --
    causal by construction, see realized_vol/realized_return_vol):
        z_t = target_t - exposure_{t-1}
        if |z_t| <= band_t:              exposure_t = exposure_{t-1}                (no trade)
        else:                            exposure_t = exposure_{t-1} + kappa_t * z_t (partial move)

    Parameters
    ----------
    target : pd.Series   aim portfolio in [-1,1], ALREADY causal (see module docstring contract).
    price  : pd.Series   daily closes, same index as target, for realized return vol only.
    cost_bps : float     the real venue round-trip-primary rate applied to |delta_exposure|
                         (13.0 baseline; sweep scenarios use 6.5 / 26.0 per program convention).
    gamma  : float       risk-aversion scale in k = gamma * sigma_r^2 (normalized to 1.0; the
                         absolute level is not independently pinned down -- see markdown "tradeoffs"
                         discussion -- cost_sensitivity/band_scale are the meaningful sweep knobs).
    cost_sensitivity, band_scale : float   see DEFAULT_GRID doc above.
    vol_window, min_periods : int   rolling window for both realized-vol estimators.
    band_floor, band_cap : float   clip the derived band width to a sane range.
    kappa_cap : float    cap on the daily partial-adjustment fraction (default 1.0).

    Returns
    -------
    pd.DataFrame indexed like `target`/`price`, columns:
        target       passthrough of the input aim portfolio x*_t
        exposure     the controlled position x_t actually held from t to t+1
        band         no-trade half-width Delta_t in effect for the decision at t (causal)
        kappa_daily  partial-adjustment fraction used when trading (causal)
        z            target_t - exposure_{t-1} (the deviation tested against the band)
        traded       bool, whether a trade was made at t
    """
    theta = (cost_sensitivity * cost_bps) / 1e4

    sigma_r = realized_return_vol(price, window=vol_window, min_periods=min_periods)
    sigma_z = realized_vol(target, window=vol_window, min_periods=min_periods)
    k_curv = gamma * np.square(sigma_r)

    band = band_scale * no_trade_band_width(theta, k_curv, sigma_z)
    band = band.clip(lower=band_floor, upper=band_cap)
    kappa = gp_adjustment_speed(k_curv, theta, band)
    kappa_daily = pd.Series(kappa_to_daily_fraction(kappa.values, cap=kappa_cap), index=kappa.index)

    n = len(target)
    tgt = target.to_numpy(dtype=float)
    band_v = band.to_numpy(dtype=float)
    kappa_v = kappa_daily.to_numpy(dtype=float)

    exposure = np.zeros(n)
    z_arr = np.full(n, np.nan)
    traded = np.zeros(n, dtype=bool)

    x_prev = 0.0
    for t in range(n):
        if np.isnan(tgt[t]):
            exposure[t] = x_prev
            continue
        z = tgt[t] - x_prev
        z_arr[t] = z
        b = band_v[t]
        k = kappa_v[t]
        if np.isnan(b) or np.isnan(k):
            # not enough history yet to calibrate the control -- hold prior exposure (no trade)
            exposure[t] = x_prev
        elif abs(z) <= b:
            exposure[t] = x_prev
        else:
            traded[t] = True
            exposure[t] = x_prev + k * z
        x_prev = exposure[t]

    return pd.DataFrame(
        {
            "target": tgt,
            "exposure": exposure,
            "band": band_v,
            "kappa_daily": kappa_v,
            "z": z_arr,
            "traded": traded,
        },
        index=target.index,
    )
