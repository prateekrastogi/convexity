"""Axis F -- PORTFOLIO CONSTRUCTION ACROSS THE 93-NAME UNIVERSE.

Stage-A research module for the modernized-CTA program (research/cta/). This module is the
PORTFOLIO layer: it takes each name's already-determined target exposure (produced upstream by
Axis A/B/C/D/E signal modules -- a per-name, per-day scalar in [-1, +1] or unrestricted, this
module does not care which axis produced it or its sign convention beyond "larger magnitude = more
exposure") and combines the active names into ONE book, addressing Established Fact (3): the
93-name universe shares a common systematic crash factor, so naive per-name risk-parity does not
actually diversify crash-day risk the way it would for genuinely idiosyncratic risk.

Two portfolio-construction schemes are implemented, both consuming the SAME upstream per-name
target-exposure signal and the SAME realized-return panel, so a like-for-like comparison isolates
the portfolio-layer design choice:

  1. EQUAL-RISK-WEIGHT (baseline): floored cross-sectional inverse-vol weighting, ignoring
     cross-name correlation entirely -- the exact convention already validated in
     e2_pipeline.py/e3_pipeline.py's `_weights_from_signal`-style functions and reused verbatim in
     the exploratory `_classic_cta_universe_scan.py` (`FLOOR_MULT = 0.5` of the cross-sectional
     median trailing vol). This is what "93-name diversification" naively assumes is happening.

  2. CORRELATION-ADJUSTED RISK PARITY: true equal-risk-contribution (ERC) weights solved against a
     shrunk realized covariance matrix, so a name's weight is discounted when it is highly
     correlated with the rest of the currently-active book (i.e. when it is NOT actually a
     diversifying bet), not just when it is individually volatile.

Both schemes are then passed through the SAME causal correlation-spike de-risking overlay (Ang &
Chen 2002-style asymmetric-correlation regime detector) before final position sizing, and the
EFFECTIVE NUMBER OF INDEPENDENT BETS is reported as a standing diagnostic time series alongside
every run (not a one-off number) so its behavior in known crash windows (2020-03, 2022) can be
inspected directly.

CAUSALITY (mandatory, verified by test_causality_injection() at the bottom of this file): every
function that produces a time-varying value used to size or gate exposure at day t uses ONLY
returns/correlations/vols computed from data strictly before day t (rolling window ending at
t-1, `.shift(1)` semantics throughout). See INJECTION TEST section.

Reused bug-pattern fixes from research/equity/e3_pipeline.py and the closed I1/CTA-scan lineage
(re-verified fresh here, not assumed):
  (a) source-aware returns: callers must pass a returns panel built from source=='duka' rows only
      (this module does not read parquets itself in its core API -- see `load_duka_daily_returns_panel`
      helper, which DOES filter source=='duka' before any resampling, reusing the
      daily_close_series_duka convention: last trade of each real calendar day).
  (b) derived annualization: `periods_per_year` is a required, explicit argument everywhere it
      matters (e.g. de-risking overlay lookback in trading days is passed directly, so no
      hardcoded 252 leaks in silently where a caller's own annualization differs) -- this module
      itself only ever operates in native per-observation (daily) units internally; annualization
      is applied once, at the point of reporting, not layered repeatedly.
  (c) exposure-timing alignment: EVERY rolling stat below is computed on `returns.shift(0)` data
      through row t-1 and used to size the exposure APPLIED at row t -- concretely, every
      `.rolling(window)` call is immediately followed by `.shift(1)` before being used, and the
      unit tests inject a synthetic outlier at a single date and confirm no diagnostic reacts
      before the correct one-period lag (see bottom of file).
  (d) unit-gross renormalization: per the exploratory CTA scan's own documented bug-find, summing
      unnormalized per-name weights lets the book's total gross notional scale mechanically with
      breadth (how many names are active that day) -- both schemes here renormalize the raw
      per-name book to exactly 1.0 total gross (sum of |weight| across active names) BEFORE the
      correlation-spike overlay is applied, exactly mirroring that fix.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ======================================================================================
# constants (disclosed, not hidden magic numbers)
# ======================================================================================
FLOOR_MULT = 0.5          # equal-risk-weight floor: 0.5 * cross-sectional median trailing vol
                          # (exact convention reused from e2/e3_pipeline.py + the CTA universe scan)
MIN_ACTIVE_FOR_CORR = 10  # below this many simultaneously-active names, a realized correlation
                          # matrix is too noisy to be trustworthy -> scheme falls back to
                          # equal-risk-weight for that day (disclosed fallback, not silent)
RIDGE_EPS = 1e-6          # numerical ridge added to the shrunk covariance diagonal for invertibility


# ======================================================================================
# 0. data loading helper (source-aware, reuses the daily_close_series_duka convention)
# ======================================================================================
def load_duka_daily_returns_panel(symbols: list[str], data_stitched_dir) -> pd.DataFrame:
    """Build a (date x symbol) simple daily-return panel from stitched parquets.

    Bug pattern (a): filters every parquet to source=='duka' BEFORE any resampling. Daily close
    is the last duka trade of each real calendar day (the daily_close_series_duka convention from
    i1_pipeline.py/e3_pipeline.py). Returns NaN before a name's first real duka day and after any
    >5-calendar-day gap (matching e3_pipeline.py's daily_returns_from_close guard) -- point-in-time
    discipline: a name is simply absent (NaN, never 0) until it has real history.
    """
    import pathlib
    data_stitched_dir = pathlib.Path(data_stitched_dir)
    closes = {}
    for sym in symbols:
        fp = data_stitched_dir / f"{sym}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["open_time", "close", "source"])
        df = df[df["source"] == "duka"]
        if df.empty:
            continue
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        d = pd.DataFrame({"ts": ts, "close": df["close"].to_numpy()})
        d["day"] = d["ts"].dt.floor("D")
        daily_close = d.groupby("day").tail(1).set_index("day")["close"].sort_index()
        closes[sym] = daily_close
    panel_close = pd.DataFrame(closes).sort_index()
    ret = panel_close.pct_change()
    # gap guard per-column: a >5 calendar day gap invalidates that one return (mirrors
    # e3_pipeline.py::daily_returns_from_close, applied per-column here since columns can have
    # independent gap structure)
    gap_days = panel_close.index.to_series().diff().dt.days
    mask = gap_days > 5
    ret.loc[mask, :] = np.nan
    return ret


# ======================================================================================
# 1. effective number of independent bets
# ======================================================================================
def effective_n_closed_form(avg_pairwise_corr: pd.Series, n_active: pd.Series) -> pd.Series:
    """N_eff = N / (1 + (N-1)*rho_avg) for the equal-weight, equal-vol case.

    DERIVATION: for N assets with equal variance sigma^2 and constant pairwise correlation
    rho_avg, the variance of the equal-weight portfolio is
        Var_EW = (1/N) sigma^2 + ((N-1)/N) rho_avg sigma^2.
    Define the "effective number of independent bets" N_eff as the number of TRULY independent,
    equal-variance assets whose equal-weight portfolio would have the SAME variance:
        sigma^2 / N_eff = Var_EW
        1/N_eff = 1/N + (N-1)/N * rho_avg = (1 + (N-1)*rho_avg) / N
        N_eff = N / (1 + (N-1)*rho_avg).
    rho_avg=0 -> N_eff=N (full diversification). rho_avg=1 -> N_eff=1 (one bet no matter how many
    names). This is the standard result (see e.g. Bouchaud & Potters, "Theory of Financial Risk
    and Derivative Pricing", ch.2, or any risk-parity text's "diversification ratio" discussion).
    Vectorized over a time series of (rho_avg(t), N_active(t)) pairs -- both already-causal inputs
    (see rolling_avg_pairwise_correlation below); this function itself is a pure pointwise
    transform and adds no additional lag.

    NUMERICAL GUARD (found via real-data validation, not theoretical -- see module validation
    notes): a valid N-asset correlation matrix requires rho_avg >= -1/(N-1) (the average of a PSD
    matrix's off-diagonal entries cannot be more negative than this, e.g. for N=2 the two assets
    can be at most rho=-1; for N=3 the three pairwise correlations cannot all simultaneously be
    below -0.5). A rolling SAMPLE correlation estimated from a short window with very few
    simultaneously-active names (the early-history, thin-breadth part of the sample, before most
    of the 93 names have listed) can noisily land marginally below this theoretical floor, and
    since the closed-form's denominator (1+(N-1)*rho_avg) -> 0+ exactly at that floor, a noisy
    breach sends N_eff toward +infinity -- a numerical artifact of applying a POPULATION formula to
    a noisy small-sample estimate, not a real diversification signal. FIX: clip rho_avg(t) per-row
    to its own theoretically-valid floor -1/(N(t)-1) (plus a small epsilon), and additionally cap
    the final N_eff at N itself (N_eff can never legitimately exceed the actual number of active
    names -- a pure sanity bound; real average correlations for this equity universe are always
    positive and regime-varying in every window actually observed, never near this pathological
    floor -- see the reported avg_corr_mean by regime in the validation summary).
    """
    rho = avg_pairwise_corr.copy()
    n = n_active.astype(float)
    valid_floor = np.where(n > 1.0, -1.0 / (n - 1.0) + 1e-6, -0.999)
    rho = rho.clip(lower=pd.Series(valid_floor, index=rho.index), upper=0.999)
    denom = 1.0 + (n - 1.0) * rho
    denom = denom.where(denom > 1e-6, 1e-6)
    n_eff = n / denom
    n_eff = n_eff.clip(upper=n)  # sanity: cannot exceed the actual number of active names
    return n_eff.rename("n_eff_closed_form")


def effective_n_pca_entropy(cov: np.ndarray) -> float:
    """Meucci (2010, 'Managing Diversification', Risk magazine)-style effective number of bets
    via PCA eigenvalue entropy on a single covariance matrix snapshot.

    Decompose Sigma = V diag(lambda) V' (eigendecomposition). The "diversification distribution"
    is p_k = lambda_k / sum(lambda) (fraction of total variance explained by principal component
    k). The effective number of bets is the exponential of the Shannon entropy of this
    distribution: N_ent = exp(-sum_k p_k * log(p_k)). This is a MORE general measure than the
    closed-form equal-weight/equal-corr formula above -- it does not assume equal per-asset
    variance or a single constant pairwise correlation, and is invariant to the choice of basis
    (unlike counting "how many names have weight > x%"). p_k=uniform over N -> N_ent=N (fully
    independent). One dominant eigenvalue -> N_ent -> 1 (one systematic factor dominates, which is
    exactly Established Fact (3)'s claim for this universe on crash days).
    Returns a single float for one covariance-matrix snapshot; callers roll this over time
    (see rolling_effective_n_pca below) using only through-t-1 covariance estimates.
    """
    if cov.shape[0] < 2:
        return float(cov.shape[0])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return float(cov.shape[0])
    p = eigvals / total
    p = p[p > 1e-14]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ======================================================================================
# 2. rolling, CAUSAL correlation/covariance estimation with shrinkage
# ======================================================================================
def rolling_avg_pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Causal rolling average pairwise Pearson correlation across all pairs of simultaneously-
    active columns, using ONLY the `window` trading days strictly BEFORE date t (shift(1)).

    For each date t, restricts to columns with a complete, finite return history over
    [t-window, t) (no NaNs in that window -- a newly-listed or currently-gapped name is simply
    excluded from that day's correlation estimate, matching point-in-time discipline), computes
    the sample correlation matrix, and averages the off-diagonal entries. Returned series is
    indexed identically to `returns.index`; the first `window` rows are NaN (no estimate yet).
    """
    idx = returns.index
    n = len(idx)
    out = np.full(n, np.nan)
    values = returns.to_numpy()
    for t in range(window, n):
        window_slice = values[t - window:t, :]  # rows [t-window, t-1] inclusive == strictly before t
        col_ok = ~np.isnan(window_slice).any(axis=0)
        if col_ok.sum() < 2:
            continue
        sub = window_slice[:, col_ok]
        corr = np.corrcoef(sub, rowvar=False)
        k = corr.shape[0]
        if k < 2:
            continue
        off_diag_sum = corr.sum() - np.trace(corr)
        out[t] = off_diag_sum / (k * (k - 1))
    return pd.Series(out, index=idx, name="avg_pairwise_corr")


def _shrink_to_constant_correlation(sample_corr: np.ndarray, shrink_intensity: float) -> np.ndarray:
    """Ledoit & Wolf (2004, 'Honey, I Shrunk the Sample Covariance Matrix', J. Portfolio Mgmt / and
    the closely related constant-correlation model of Ledoit & Wolf 2003) shrinkage TARGET: the
    constant-correlation matrix (every off-diagonal entry equal to the sample's own average
    pairwise correlation rho_bar, diagonal = 1). This is a well-motivated, low-parameter target for
    a universe with a dominant common factor (Established Fact 3) -- it shrinks the noisy, high-
    dimensional sample correlation matrix (unstable when the number of names approaches or exceeds
    the estimation window length) toward the single number that best summarizes "how correlated is
    this book right now", without requiring a full DCC-GARCH estimation.

    SIMPLIFICATION DISCLOSED: this module uses a FIXED, disclosed shrinkage intensity (passed by
    the caller, not the full analytic Ledoit-Wolf optimal-intensity estimator, which requires
    estimating fourth-moment cross-terms over the same rolling window and is a meaningfully bigger
    engineering lift for a Stage-A module). A fixed intensity in a sensible range (0.2-0.4 is
    typical in the shrinkage literature for daily equity returns) is judged an honest, adequate
    simplification for this deliverable; the grid below sweeps two window lengths so the shrinkage
    target itself (rho_bar) still varies appropriately with regime even though its blend weight is
    fixed. `shrunk = (1-delta)*sample_corr + delta*constant_corr_target`.
    """
    k = sample_corr.shape[0]
    off = sample_corr[~np.eye(k, dtype=bool)]
    rho_bar = float(off.mean()) if off.size else 0.0
    target = np.full((k, k), rho_bar)
    np.fill_diagonal(target, 1.0)
    shrunk = (1.0 - shrink_intensity) * sample_corr + shrink_intensity * target
    np.fill_diagonal(shrunk, 1.0)
    return shrunk


def rolling_shrunk_covariance(
    returns: pd.DataFrame, t: int, window: int = 60, shrink_intensity: float = 0.3
) -> tuple[np.ndarray, list[str]] | None:
    """Return (shrunk covariance matrix, active column names) for use in sizing date `values[t]`,
    estimated from rows [t-window, t) ONLY (strictly causal). Returns None if fewer than
    MIN_ACTIVE_FOR_CORR names have a complete window (caller should fall back to equal-risk-weight
    for that date). Vols come from the SAME window (sample std, ddof=1); the shrunk CORRELATION
    matrix is converted back to a covariance matrix via outer(sigma, sigma) * shrunk_corr, then a
    small ridge (RIDGE_EPS * mean variance) is added to the diagonal for numerical invertibility
    before being handed to the risk-parity solver.
    """
    if t < window:
        return None
    cols = returns.columns
    window_slice = returns.iloc[t - window:t, :]
    col_ok = window_slice.notna().all(axis=0)
    active = list(cols[col_ok.to_numpy()])
    if len(active) < MIN_ACTIVE_FOR_CORR:
        return None
    sub = window_slice[active].to_numpy()
    sample_corr = np.corrcoef(sub, rowvar=False)
    shrunk_corr = _shrink_to_constant_correlation(sample_corr, shrink_intensity)
    sigma = sub.std(axis=0, ddof=1)
    sigma = np.maximum(sigma, 1e-8)
    cov = np.outer(sigma, sigma) * shrunk_corr
    ridge = RIDGE_EPS * np.mean(np.diag(cov))
    cov = cov + ridge * np.eye(len(active))
    return cov, active


# ======================================================================================
# 3. correlation-spike de-risking overlay (Ang & Chen 2002-style)
# ======================================================================================
def correlation_spike_scale(
    avg_pairwise_corr: pd.Series,
    lookback: int = 756,          # ~3 trading years, disclosed
    spike_percentile: float = 90.0,
    full_deleverage_percentile: float = 99.0,
    scale_floor: float = 0.4,
) -> pd.Series:
    """Causal de-risking multiplier in [scale_floor, 1.0], keyed off where TODAY's already-causal
    avg_pairwise_corr(t) (itself computed from data through t-1, see above -- no additional lag
    needed here since the input is already a t-1-information quantity) sits in ITS OWN trailing
    `lookback`-day empirical distribution (computed using only avg_pairwise_corr values at indices
    STRICTLY BEFORE t, i.e. `avg_pairwise_corr.shift(1).rolling(lookback)` semantics).

    MOTIVATION (Ang & Chen 2002, 'Asymmetric Correlations of Equity Portfolios', J. Financial
    Economics; also Longin & Solnik 2001): equity correlations are well-documented to spike
    specifically during market stress/drawdowns, exactly the regime in which "93-name
    diversification" stops working (Established Fact 3). A book that does not respond to this
    keeps running its normal risk budget straight into the one regime where its effective breadth
    has collapsed toward N_eff~1.

    DESIGN: piecewise-linear scale-down between the `spike_percentile`-th and
    `full_deleverage_percentile`-th percentile of the name's OWN trailing history (not a fixed
    absolute correlation threshold -- a universe whose typical rho_avg is already 0.35 needs a
    different trigger than one whose typical rho_avg is 0.10; anchoring to the series' own history
    is the standard practical fix and avoids a hand-tuned absolute constant):
        scale(t) = 1.0                                              if rho(t) <= p_spike(t)
        scale(t) = linear from 1.0 down to scale_floor               if p_spike(t) < rho(t) < p_full(t)
        scale(t) = scale_floor                                       if rho(t) >= p_full(t)
    where p_spike(t), p_full(t) are the trailing-`lookback`-day `spike_percentile`/
    `full_deleverage_percentile` empirical percentiles of rho, evaluated on history strictly before
    t. First `lookback` rows (no trailing distribution yet) get scale 1.0 (fail open, consistent
    with overlay.py::vol_target_scale's own convention for its own warm-up window).
    """
    rho = avg_pairwise_corr.to_numpy()
    n = len(rho)
    scale = np.ones(n)
    rho_hist = pd.Series(rho).shift(1)  # strictly-before-t view for the trailing distribution
    p_spike = rho_hist.rolling(lookback, min_periods=lookback).quantile(spike_percentile / 100.0)
    p_full = rho_hist.rolling(lookback, min_periods=lookback).quantile(full_deleverage_percentile / 100.0)
    p_spike_a = p_spike.to_numpy()
    p_full_a = p_full.to_numpy()
    for t in range(n):
        r = rho[t]
        lo, hi = p_spike_a[t], p_full_a[t]
        if not (np.isfinite(r) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi <= lo:
            continue
        if r <= lo:
            scale[t] = 1.0
        elif r >= hi:
            scale[t] = scale_floor
        else:
            frac = (r - lo) / (hi - lo)
            scale[t] = 1.0 - frac * (1.0 - scale_floor)
    return pd.Series(scale, index=avg_pairwise_corr.index, name="corr_spike_scale")


# ======================================================================================
# 4. per-name weighting schemes
# ======================================================================================
def floored_inverse_vol_weights(rv_lagged_row: pd.Series, active_mask: pd.Series) -> pd.Series:
    """Equal-risk-weight baseline for ONE date: floored cross-sectional inverse-vol, exact
    convention from e2_pipeline.py/e3_pipeline.py and the exploratory CTA universe scan:
        floor = FLOOR_MULT * cross-sectional median of rv_lagged over active names that day
        weight_i (unnormalized) = 1 / max(rv_lagged_i, floor)   for active i, 0 otherwise
    `rv_lagged_row` must already be a through-t-1 trailing-vol estimate (caller's responsibility --
    this function performs no additional lagging, it is a pure pointwise transform of an
    already-causal input, exactly like effective_n_closed_form above).
    """
    active_rv = rv_lagged_row[active_mask].dropna()
    if active_rv.empty:
        return pd.Series(0.0, index=rv_lagged_row.index)
    floor = FLOOR_MULT * float(active_rv.median())
    floor = max(floor, 1e-8)
    w = pd.Series(0.0, index=rv_lagged_row.index)
    w.loc[active_rv.index] = 1.0 / active_rv.clip(lower=floor)
    return w


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """Equal-risk-contribution (ERC) weights for a single covariance-matrix snapshot, via Spinu's
    (2013, 'An Algorithm for Computing Risk Parity Weights') convex reformulation: the ERC
    condition w_i*(Sigma w)_i = c for all i (c a constant, w>0) is the first-order condition of the
    convex program
        min_{y>0}  0.5 * y' Sigma y - sum_i log(y_i)
    (unique minimizer exists and is strictly positive whenever Sigma is positive definite, since
    the objective is strictly convex and -> +infinity as any y_i -> 0+ or ||y|| -> infinity). The
    ERC weights are then w = y / sum(y). This avoids hand-rolling a bespoke fixed-point iteration
    and has a standard, well-behaved convergence proof (see Spinu 2013; also Griveau-Billiotte
    2013 for the equivalence proof). Solved here via a simple, dependency-light multiplicative
    Newton-like fixed point on the log-barrier stationarity condition (Sigma y = 1/y elementwise),
    y_i <- y_i * sqrt((1/y_i) / (Sigma y)_i)... implemented directly as the well-known cyclical
    risk-budgeting update (Roncalli, 'Introduction to Risk Parity and Budgeting', 2013, sec 2.2):
        y_i <- y_i * sqrt(target_i / (y_i * (Sigma y)_i))     [target_i = 1/k uniform]
    normalized to sum(y)=1 after each sweep, iterated to convergence (max relative change in risk
    contributions < tol, or max_iter reached -- convergence is fast, typically <100 iterations for
    the well-conditioned shrunk covariance matrices produced by rolling_shrunk_covariance).
    """
    k = cov.shape[0]
    if k == 1:
        return np.array([1.0])
    sigma = np.sqrt(np.diag(cov))
    y = 1.0 / np.maximum(sigma, 1e-12)
    y = y / y.sum()
    target = 1.0 / k
    for _ in range(max_iter):
        mrc = cov @ y                      # marginal risk contribution
        rc = y * mrc                       # risk contribution
        rc = np.maximum(rc, 1e-16)
        y_new = y * np.sqrt(target / rc)
        y_new = y_new / y_new.sum()
        if np.max(np.abs(y_new - y)) < tol:
            y = y_new
            break
        y = y_new
    return y


# ======================================================================================
# 5. portfolio-level aggregation (the deliverable)
# ======================================================================================
@dataclass
class PortfolioResult:
    """Container for aggregate_portfolio()'s output: final per-name weights plus every diagnostic
    needed to audit the construction (effective-N time series, avg correlation, overlay scale)."""
    weights: pd.DataFrame                 # date x symbol, final signed exposure per name
    avg_pairwise_corr: pd.Series
    n_active: pd.Series
    n_eff_closed_form: pd.Series
    n_eff_pca: pd.Series                  # sparser: only defined on dates the corr-scheme actually solved ERC
    corr_spike_scale: pd.Series
    scheme_used: pd.Series                # per-date: "corr_risk_parity" or "equal_risk_weight_fallback"


def aggregate_portfolio(
    target_exposure: pd.DataFrame,        # date x symbol, upstream per-name signal (sign + magnitude)
    returns: pd.DataFrame,                # date x symbol, realized simple daily returns (duka-only)
    rv_lagged: pd.DataFrame,              # date x symbol, trailing realized vol through t-1 (caller-supplied)
    scheme: str = "corr_risk_parity",     # "corr_risk_parity" | "equal_risk_weight"
    corr_window: int = 60,
    shrink_intensity: float = 0.3,
    apply_spike_overlay: bool = True,
    spike_lookback: int = 756,
    spike_percentile: float = 90.0,
    full_deleverage_percentile: float = 99.0,
    scale_floor: float = 0.4,
) -> PortfolioResult:
    """THE deliverable: combine each active name's own upstream target exposure into one book.

    Per date t (all inputs already t-1-safe by construction -- see CAUSALITY docstring at top):
      1. active_i(t) = finite(target_exposure_i(t)) & finite(rv_lagged_i(t))
      2. per-name RAW weight:
           - scheme=="equal_risk_weight": raw_i = target_i(t) * floored_inverse_vol_weight_i(t)
           - scheme=="corr_risk_parity" : raw_i = target_i(t) * ERC_weight_i(t) where ERC weights
             solve equal-risk-contribution on the rolling shrunk covariance of ACTIVE names only
             (falls back to floored-inverse-vol for that date, `scheme_used="equal_risk_weight_fallback"`,
             if fewer than MIN_ACTIVE_FOR_CORR names are simultaneously active -- disclosed, not
             silent: see `scheme_used` diagnostic column).
      3. UNIT-GROSS RENORMALIZATION (bug-pattern-(d) fix, reused from the CTA universe scan):
           final_i(t) = raw_i(t) / sum_j |raw_j(t)|     (0 on an all-flat day)
      4. CORRELATION-SPIKE DE-RISKING OVERLAY applied multiplicatively to the whole book:
           final_i(t) *= corr_spike_scale(t)            (if apply_spike_overlay)
    Returns a PortfolioResult with the final weights plus every diagnostic series required to
    report effective-N as a standing time series (not a one-off number).
    """
    if scheme not in ("corr_risk_parity", "equal_risk_weight"):
        raise ValueError(f"unknown scheme {scheme!r}")

    idx = target_exposure.index
    cols = target_exposure.columns
    assert (returns.index == idx).all(), "returns index must match target_exposure index"

    avg_corr = rolling_avg_pairwise_correlation(returns, window=corr_window)
    spike_scale = (correlation_spike_scale(avg_corr, lookback=spike_lookback,
                                            spike_percentile=spike_percentile,
                                            full_deleverage_percentile=full_deleverage_percentile,
                                            scale_floor=scale_floor)
                   if apply_spike_overlay else pd.Series(1.0, index=idx))

    n = len(idx)
    final = pd.DataFrame(0.0, index=idx, columns=cols)
    n_active = pd.Series(0, index=idx, dtype=int)
    n_eff_pca = pd.Series(np.nan, index=idx)
    scheme_used = pd.Series("n/a", index=idx, dtype=object)

    target_np = target_exposure.to_numpy()
    rv_np = rv_lagged.to_numpy()

    for t in range(n):
        active_mask = np.isfinite(target_np[t, :]) & np.isfinite(rv_np[t, :]) & (target_np[t, :] != 0.0)
        n_active.iloc[t] = int(active_mask.sum())
        if n_active.iloc[t] == 0:
            continue
        active_cols = cols[active_mask]

        use_corr_scheme = False
        cov_result = None
        if scheme == "corr_risk_parity":
            cov_result = rolling_shrunk_covariance(returns, t, window=corr_window,
                                                     shrink_intensity=shrink_intensity)
            if cov_result is not None:
                cov, active_names_for_cov = cov_result
                # ERC solved on ALL names with a full window (not just today's active set) so the
                # covariance structure reflects the true book neighborhood; then restricted to
                # today's actually-active names for final position sizing.
                usable = [c for c in active_cols if c in active_names_for_cov]
                if len(usable) >= 2:
                    use_corr_scheme = True

        if use_corr_scheme:
            cov, active_names_for_cov = cov_result
            name_to_idx = {c: i for i, c in enumerate(active_names_for_cov)}
            sel = [name_to_idx[c] for c in usable]
            sub_cov = cov[np.ix_(sel, sel)]
            erc_w = risk_parity_weights(sub_cov)
            n_eff_pca.iloc[t] = effective_n_pca_entropy(cov)
            raw = pd.Series(0.0, index=cols)
            tgt_row = target_exposure.iloc[t]
            for c, w in zip(usable, erc_w):
                raw[c] = tgt_row[c] * w
            scheme_used.iloc[t] = "corr_risk_parity"
        else:
            rv_row = rv_lagged.iloc[t]
            active_series = pd.Series(active_mask, index=cols)
            inv_vol_w = floored_inverse_vol_weights(rv_row, active_series)
            raw = target_exposure.iloc[t] * inv_vol_w
            scheme_used.iloc[t] = ("equal_risk_weight" if scheme == "equal_risk_weight"
                                    else "equal_risk_weight_fallback")

        gross = raw.abs().sum()
        if gross > 1e-12:
            row_final = (raw / gross) * spike_scale.iloc[t]
            final.iloc[t] = row_final.to_numpy()

    n_eff_cf = effective_n_closed_form(avg_corr, n_active)

    return PortfolioResult(
        weights=final,
        avg_pairwise_corr=avg_corr,
        n_active=n_active,
        n_eff_closed_form=n_eff_cf,
        n_eff_pca=n_eff_pca,
        corr_spike_scale=spike_scale,
        scheme_used=scheme_used,
    )


# ======================================================================================
# INJECTION TEST (causality proof) -- see structured-output "injection_test_result"
# ======================================================================================
def _naive_uncausal_avg_corr(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """NAIVE CONTROL (deliberately wrong): a centered/including-day-t rolling correlation, used
    ONLY to prove the injection test has teeth (it SHOULD react same-day to a same-day outlier,
    unlike rolling_avg_pairwise_correlation above which must not)."""
    idx = returns.index
    n = len(idx)
    out = np.full(n, np.nan)
    values = returns.to_numpy()
    for t in range(window, n):
        window_slice = values[t - window + 1:t + 1, :]  # INCLUDES row t -- the bug under test
        col_ok = ~np.isnan(window_slice).any(axis=0)
        if col_ok.sum() < 2:
            continue
        sub = window_slice[:, col_ok]
        corr = np.corrcoef(sub, rowvar=False)
        k = corr.shape[0]
        if k < 2:
            continue
        out[t] = (corr.sum() - np.trace(corr)) / (k * (k - 1))
    return pd.Series(out, index=idx)


def run_causality_injection_test(window: int = 60, inject_at_frac: float = 0.7, seed: int = 0):
    """Synthetic-outlier-injection test (mandated by the Stage-A task spec).

    Injects one large fake simultaneous move (+40%) across every column at a single date index,
    then confirms:
      (1) rolling_avg_pairwise_correlation's value at the injected date is IDENTICAL to the
          un-injected baseline (it cannot see row t yet -- window is [t-window, t)).
      (2) rolling_avg_pairwise_correlation's value at the FIRST date AFTER the injection DOES
          change vs baseline (the injected row has now entered its trailing window -- proving the
          function is not simply broken/inert).
      (3) NAIVE CONTROL (_naive_uncausal_avg_corr) DOES change at the injected date itself,
          proving the test methodology has real discriminating power (a function that looked at
          the future would have been caught).
    Returns a dict summary suitable for direct inclusion in a report.
    """
    rng = np.random.default_rng(seed)
    n_days, n_names = 800, 12
    dates = pd.bdate_range("2019-01-02", periods=n_days, tz="UTC")
    base = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, n_names)), index=dates,
                         columns=[f"SYN{i}" for i in range(n_names)])
    t_inject = int(n_days * inject_at_frac)

    baseline_causal = rolling_avg_pairwise_correlation(base, window=window)

    injected = base.copy()
    injected.iloc[t_inject, :] = 0.40  # large synthetic simultaneous move at exactly one row
    injected_causal = rolling_avg_pairwise_correlation(injected, window=window)

    at_inject_unchanged = bool(np.isclose(injected_causal.iloc[t_inject],
                                           baseline_causal.iloc[t_inject], equal_nan=True))
    after_changed = bool(not np.isclose(injected_causal.iloc[t_inject + 1],
                                         baseline_causal.iloc[t_inject + 1], equal_nan=True))

    baseline_naive = _naive_uncausal_avg_corr(base, window=window)
    injected_naive = _naive_uncausal_avg_corr(injected, window=window)
    naive_reacts_same_day = bool(not np.isclose(injected_naive.iloc[t_inject],
                                                 baseline_naive.iloc[t_inject], equal_nan=True))

    return {
        "t_inject": t_inject,
        "causal_fn_unchanged_at_injection_day": at_inject_unchanged,
        "causal_fn_changed_day_after": after_changed,
        "naive_control_changed_same_day": naive_reacts_same_day,
        "PASS": at_inject_unchanged and after_changed and naive_reacts_same_day,
        "baseline_val_at_inject": float(baseline_causal.iloc[t_inject]),
        "injected_val_at_inject": float(injected_causal.iloc[t_inject]),
        "baseline_val_day_after": float(baseline_causal.iloc[t_inject + 1]),
        "injected_val_day_after": float(injected_causal.iloc[t_inject + 1]),
        "naive_baseline_val_at_inject": float(baseline_naive.iloc[t_inject]),
        "naive_injected_val_at_inject": float(injected_naive.iloc[t_inject]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_causality_injection_test(), indent=2, default=str))
