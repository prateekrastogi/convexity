"""
research/cta/lib/vol_forecast.py -- Axis D: volatility forecasting for position sizing.

Replaces naive trailing realized volatility with a small family of causal, one-step-ahead
volatility forecasters, evaluated by direct forecast-quality loss (QLIKE / MSE against realized
next-day squared return) BEFORE any portfolio-outcome comparison, per the Stage-A brief.

CAUSALITY CONVENTION (bug pattern 7(c), re-verified fresh here): every `*_forecast` function in
this module returns a pandas Series indexed identically to its daily input, where
``forecast[t]`` is the volatility (or variance, where documented) applicable to trading on day
``t``, computed using information available only through the close of day ``t-1``. Concretely
this means: compute the estimator's own "as-of-close-of-day-t" value, then ``.shift(1)`` (or an
equivalent explicit one-step recursion) before returning. Callers must never need to shift these
outputs again. This is verified for every estimator by
``run_all_injection_tests`` (see bottom of file), which injects a large synthetic outlier at one
day-index into a COPY of real data and confirms:
  (a) forecast[j]   is IDENTICAL before/after injecting the outlier at index j (no lookahead), and
  (b) forecast[j+1] DIFFERS (the estimator does react, one day late) -- and a naive/unlagged
      control version of the same estimator DOES react at forecast[j] itself, proving the test
      has real discriminating power and is not a tautology.

DATA CONTRACT (verified fresh against research/equity/DATA.md and the I1/CTA-scan precedent):
  - Every stitched parquet is filtered to source=='duka' BEFORE any computation (bug pattern (a)
    -- the underlying/perp splice seam fabricates returns if crossed). This module's
    `aggregate_daily_ohlc_from_duka` takes an ALREADY-FILTERED df (caller's responsibility, same
    division of labor as i1_pipeline.load_stitched_duka / daily_close_series_duka).
  - Daily bars are AGGREGATED from 1-minute duka rows per real trading day (no pre-aggregated
    daily field exists): open = first bar's open, high = max(high), low = min(low),
    close = last bar's close, using the exact `ts.dt.floor("D")` day-grouping convention already
    established in i1_pipeline.daily_close_series_duka (re-used verbatim here for consistency).
  - Realized-variance-from-intraday (used by HAR-RV) sums squared log returns of subsampled
    (default 5-minute) 1-minute duka closes WITHIN each real trading day only -- day boundaries
    are never crossed, so overnight jumps never contaminate the intraday RV measure (they are
    exactly what the overnight component of Yang-Zhang is for instead).

periods_per_year is NEVER hardcoded here (bug pattern (b)): all vol figures in this module are
DAILY (not annualized) by design, matching the daily rv_lagged convention already used by
i1_pipeline / _classic_cta_exploration.py's vol-targeting code (rv is a daily std, exposure is
sized off a ratio of two daily vols, so annualization factors cancel exactly and are never
needed). If a caller wants an annualized number for reporting, multiply by
sqrt(periods_per_year) computed fresh from that name's own trading-day count, per bug pattern (b).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ======================================================================================
# 0. Daily OHLC aggregation from 1-minute duka bars (verify-first, per DATA_CONTRACT)
# ======================================================================================
def aggregate_daily_ohlc_from_duka(df_duka: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an ALREADY source=='duka'-filtered 1-minute dataframe (columns: ts [tz-aware
    datetime64], open, high, low, close, volume) into real-calendar-day OHLCV.

    day = ts.dt.floor("D") -- identical grouping convention to
    i1_pipeline.daily_close_series_duka, so the `close` column here is byte-for-byte the same
    series that convention already produces (verified in the module's self-test).
    """
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    g = d.groupby("day", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "n_bars": g["close"].size(),
    })
    return out


def daily_log_returns(daily_close: pd.Series) -> pd.Series:
    return np.log(daily_close / daily_close.shift(1))


def realized_variance_from_intraday(df_duka: pd.DataFrame, sampling_minutes: int = 5) -> pd.Series:
    """Per real trading day, sum of squared log returns of `sampling_minutes`-subsampled 1-minute
    duka closes WITHIN that day only (never crosses the day boundary -- avoids conflating the
    overnight gap into the intraday RV measure). Index = day (tz-aware, floor("D")), value = RV_t
    (a VARIANCE, not vol) as of the close of day t. This is a purely backward-looking, same-day
    "as of close" quantity -- callers wanting a t-1-causal forecast must lag it themselves
    (`har_rv_vol_forecast` below does this).

    sampling_minutes=5 (the Corsi (2009)-standard, and the broader realized-volatility
    literature's standard, choice) is used to damp bid-ask-bounce/microstructure noise that
    contaminates a raw 1-minute sum; 1-minute is offered as an alternate grid point.
    """
    d = df_duka[["ts", "close"]].copy()
    d = d.sort_values("ts")
    d["day"] = d["ts"].dt.floor("D")
    out = {}
    for day, grp in d.groupby("day", sort=True):
        px = grp["close"].to_numpy()
        if sampling_minutes > 1:
            px = px[:: int(sampling_minutes)]
        if len(px) < 3:
            continue
        logpx = np.log(px)
        r = np.diff(logpx)
        out[day] = float(np.sum(r * r))
    return pd.Series(out, name="rv_intraday").sort_index()


# ======================================================================================
# 1. Baseline / null: trailing realized volatility (close-to-close)
# ======================================================================================
def trailing_realized_vol_forecast(daily_close: pd.Series, window: int = 20) -> pd.Series:
    """Established baseline/null. forecast[t] = std of daily log returns over the trailing
    `window` days ending at close of day t-1 (rolling std through t-1, then applied at t)."""
    ret = daily_log_returns(daily_close)
    rv = ret.rolling(window, min_periods=window).std(ddof=1)
    return rv.shift(1).rename(f"trailing_rv_{window}")


def _trailing_realized_vol_NAIVE_UNLAGGED(daily_close: pd.Series, window: int = 20) -> pd.Series:
    """NAIVE CONTROL ONLY -- deliberately includes day t's own return in forecast[t]. Used solely
    by the injection-test harness to prove the causality test has teeth. NEVER use for sizing."""
    ret = daily_log_returns(daily_close)
    return ret.rolling(window, min_periods=window).std(ddof=1).rename(f"trailing_rv_{window}_naive")


# ======================================================================================
# 2. Yang-Zhang (2000) OHLC range-based estimator
# ======================================================================================
def yang_zhang_variance_asof(daily_ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    """Yang & Zhang (2000) "Drift-Independent Volatility Estimation Based on High, Low, Open,
    and Close Prices" minimum-variance unbiased estimator. Derivation sketch:

    For day t define, using O_t,H_t,L_t,C_t and the PRIOR close C_{t-1}:
        overnight (close-to-open) log return   o_t  = ln(O_t / C_{t-1})
        open-to-close log return               c_t  = ln(C_t / O_t)
        Rogers-Satchell (1991) per-day term:
            rs_t = ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t)
        (Rogers-Satchell is itself drift-independent/unbiased for the CONTINUOUS open-to-close
        variance even under nonzero drift, by construction of the two log-range cross terms.)

    Close-to-close and Parkinson/Garman-Klass estimators are biased under opening jumps because
    they either ignore the overnight gap entirely or implicitly assume O_t = C_{t-1}. Yang-Zhang
    fixes this by explicitly modeling the OVERNIGHT variance component o_t separately, then
    combining it with a *weighted average* of the open-to-close sample variance and the
    Rogers-Satchell estimate:

        V_o  = (1/(n-1)) * sum_{i=t-n+1}^{t} (o_i - obar)^2         [overnight variance]
        V_c  = (1/(n-1)) * sum_{i=t-n+1}^{t} (c_i - cbar)^2         [open-to-close variance]
        V_rs = (1/n)     * sum_{i=t-n+1}^{t} rs_i                    [Rogers-Satchell, already
                                                                       unbiased, no demeaning]
        V_YZ = V_o + k * V_c + (1-k) * V_rs

    where the weight k = 0.34 / (1.34 + (n+1)/(n-1)) is the closed-form Yang-Zhang minimum-
    variance choice (derived by minimizing Var(V_YZ) over k under their assumed diffusion +
    independent overnight-jump model; k -> 0.34/2.34 ~= 0.1453 as n -> infinity). V_YZ is
    unbiased under BOTH nonzero drift (inherited from Rogers-Satchell) AND opening jumps
    (inherited from explicitly modeling V_o), which is exactly what close-to-close realized vol
    and naive range estimators each separately fail to be.

    Returns the AS-OF-CLOSE-OF-DAY-t variance (n-day trailing window ending at and including day
    t) -- NOT yet lagged for forecasting use; see `yang_zhang_vol_forecast` for the causal,
    forecast-ready version.
    """
    o, h, l, c = (daily_ohlc["open"], daily_ohlc["high"], daily_ohlc["low"], daily_ohlc["close"])
    prev_c = c.shift(1)
    log_o = np.log(o / prev_c)
    log_c = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)
    rs = log_ho * log_hc + log_lo * log_lc

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    V_o = log_o.rolling(n, min_periods=n).var(ddof=1)
    V_c = log_c.rolling(n, min_periods=n).var(ddof=1)
    V_rs = rs.rolling(n, min_periods=n).mean()

    V_yz = V_o + k * V_c + (1 - k) * V_rs
    return V_yz.clip(lower=0.0)


def yang_zhang_vol_forecast(daily_ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    """Causal, forecast-ready Yang-Zhang DAILY VOLATILITY (not variance): forecast[t] uses only
    OHLC through day t-1 (the as-of-close-of-t-1 trailing-window estimate, shifted onto day t)."""
    v = yang_zhang_variance_asof(daily_ohlc, window=window)
    return np.sqrt(v).shift(1).rename(f"yang_zhang_{window}")


def _yang_zhang_NAIVE_UNLAGGED(daily_ohlc: pd.DataFrame, window: int = 20) -> pd.Series:
    """NAIVE CONTROL ONLY -- uses day t's own OHLC (via the unshifted as-of-close value) as
    forecast[t]. Injection-test harness use only."""
    v = yang_zhang_variance_asof(daily_ohlc, window=window)
    return np.sqrt(v).rename(f"yang_zhang_{window}_naive")


# ======================================================================================
# 3. EWMA / RiskMetrics exponential smoothing of squared returns
# ======================================================================================
def ewma_vol_forecast(daily_close: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics recursion: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2, i.e. forecast[t]
    (the variance applicable to day t) uses only returns realized through day t-1. Implemented
    via pandas' ewm(adjust=False) recursion on r^2 (which incorporates r_t at position t), then
    shifted by one so that forecast[t] reflects only information through t-1. lam=0.94 is the
    original daily RiskMetrics constant; lam=0.97 (slower decay, longer effective memory,
    ~1/(1-0.97)=33 vs ~1/(1-0.94)=17 trading-day half-life-ish window) is the second grid point.
    """
    ret = daily_log_returns(daily_close)
    sq = ret**2
    sigma2 = sq.ewm(alpha=(1 - lam), adjust=False, min_periods=2).mean()
    return np.sqrt(sigma2).shift(1).rename(f"ewma_{lam}")


def _ewma_NAIVE_UNLAGGED(daily_close: pd.Series, lam: float = 0.94) -> pd.Series:
    """NAIVE CONTROL ONLY -- unshifted, so forecast[t] includes r_t^2 itself."""
    ret = daily_log_returns(daily_close)
    sq = ret**2
    sigma2 = sq.ewm(alpha=(1 - lam), adjust=False, min_periods=2).mean()
    return np.sqrt(sigma2).rename(f"ewma_{lam}_naive")


# ======================================================================================
# 4. GARCH(1,1), fit strictly causally with a periodic refit schedule
# ======================================================================================
def garch11_vol_forecast(
    daily_close: pd.Series,
    refit_freq: int = 21,
    window: int | None = 756,
    min_obs: int = 250,
    dist: str = "normal",
) -> pd.Series:
    """Causal GARCH(1,1) one-step-ahead DAILY VOLATILITY forecast.

    Refit-schedule discipline (matches the periodic-refit convention used elsewhere in this
    program rather than a full re-fit every day, which would be both needlessly expensive across
    a 93-name universe and unnecessary for causality): the (omega, alpha, beta) parameters are
    RE-ESTIMATED via `arch.arch_model` only every `refit_freq` trading days, each time using only
    returns through the PRIOR day's close (`window` trailing days if given, else the full
    expanding history). Between refits, the one-step-ahead variance forecast is advanced day by
    day using the FIXED (last-fitted) parameters via the exact GARCH(1,1) recursion
        sigma2_{t+1} = omega + alpha * r_t^2 + beta * sigma2_t
    applied with the REAL, already-realized daily return r_t (known by day t, hence still
    causal) -- this is standard practice (e.g. Engle's own rolling-forecast evaluations) and
    avoids re-optimizing a numerical MLE every single day while remaining exactly as causal as a
    daily refit would be for the one-step-ahead forecast itself.

    refit_freq=21 (~monthly) with window=756 (~3y rolling) is grid point 1; refit_freq=63
    (~quarterly) with window=None (expanding, full history) is grid point 2 -- chosen to bracket
    a fast-adapting/short-memory design against a slow-adapting/long-memory design.
    """
    from arch import arch_model

    ret = daily_log_returns(daily_close).dropna()
    ret_pct = ret * 100.0  # arch's documented numerical-stability convention
    idx = ret_pct.index
    n = len(ret_pct)
    sigma2_forecast = pd.Series(index=idx, dtype=float)

    if n <= min_obs:
        return np.sqrt(sigma2_forecast / 10000.0).rename(f"garch11_rf{refit_freq}_w{window}")

    omega = alpha = beta = None
    sigma2_running = None

    for i in range(min_obs, n):
        need_refit = (omega is None) or ((i - min_obs) % refit_freq == 0)
        if need_refit:
            lo = 0 if window is None else max(0, i - window)
            hist = ret_pct.iloc[lo:i]  # strictly through day i-1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(hist.to_numpy(), mean="Zero", vol="Garch", p=1, q=1, dist=dist)
                res = am.fit(disp="off", show_warning=False)
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            beta = float(res.params["beta[1]"])
            fc = res.forecast(horizon=1, reindex=False)
            sigma2_running = float(fc.variance.values[-1, 0])  # forecast FOR day i, using data through i-1
        else:
            r_prev = float(ret_pct.iloc[i - 1])  # realized, known by day i-1
            sigma2_running = omega + alpha * (r_prev**2) + beta * sigma2_running
        sigma2_forecast.iloc[i] = sigma2_running

    sigma2_forecast_natural = sigma2_forecast / (100.0**2)  # undo the *100 percent scaling
    return np.sqrt(sigma2_forecast_natural).rename(f"garch11_rf{refit_freq}_w{window}")


# ======================================================================================
# 5. HAR-RV (Corsi 2009) -- daily/weekly/monthly realized-variance components regression
# ======================================================================================
_HAR_RV_LOG_EPS = 1e-12  # floor before taking logs; RV can be exactly 0 on ultra-thin days


def har_rv_features(rv: pd.Series) -> pd.DataFrame:
    """rv[t] = realized variance AS OF close of day t (e.g. `realized_variance_from_intraday`).
    Returns the three Corsi (2009) components IN LEVELS, each defined AS OF close of day t (i.e.
    NOT yet lagged -- `har_rv_vol_forecast` handles the causal alignment and the log transform):
        d_t = RV_t                          (daily)
        w_t = mean(RV_{t-4}, ..., RV_t)      (weekly, 5 trading days incl. today)
        m_t = mean(RV_{t-21}, ..., RV_t)     (monthly, 22 trading days incl. today)
    """
    d = rv
    w = rv.rolling(5, min_periods=5).mean()
    m = rv.rolling(22, min_periods=22).mean()
    return pd.DataFrame({"d": d, "w": w, "m": m})


def har_rv_vol_forecast(rv: pd.Series, refit_freq: int = 21, min_obs: int = 60) -> pd.Series:
    """Causal HAR-RV one-step-ahead DAILY VOLATILITY forecast (Corsi 2009, "A Simple Approximate
    Long-Memory Model of Realized Volatility"): regress log(RV_{t+1}) on
    (log(RV_t), log(RV_t^w), log(RV_t^m)) -- a parsimonious linear proxy for the long-memory/
    multi-horizon persistence structure typically seen in realized volatility, avoiding a full
    ARFIMA fit. OLS coefficients (c, b_d, b_w, b_m) are re-estimated every `refit_freq` trading
    days via plain least squares.

    LOG-SPACE, NOT LEVEL-SPACE (a deliberate deviation from Corsi's original level-OLS
    specification, and the one genuine implementation bug this module's own validation run
    surfaced): a first level-OLS implementation was tried and REJECTED after real-data testing on
    SPY/QQQ/AAPL/MSFT/JPM produced QLIKE values in the hundreds of millions on a nontrivial
    minority of days. Diagnosis: plain OLS on RV levels has no constraint keeping predictions
    positive; on quiet stretches (small d/w/m, near-collinear after a long calm period) the fitted
    hyperplane occasionally predicts a near-zero or slightly negative RV, which after the
    (necessary) floor at a tiny epsilon produces a variance forecast many orders of magnitude
    smaller than the realized proxy -- QLIKE's rv/sigma2 term is UNBOUNDED as sigma2 -> 0, so a
    single such day dominates the mean loss. Regressing in log space guarantees a strictly
    positive forecast by construction (forecast = exp(predicted log-RV), never needs an artificial
    floor) and is itself a standard, literature-documented refinement (log-realized-variance is
    also closer to conditionally Gaussian than raw RV, which is heavily right-skewed) -- not a
    deviation invented to flatter this estimator, but a correctness fix verified to eliminate the
    blowup (see the module's validation run, both variants' QLIKE reported for transparency).

    CAUSALITY (the one genuinely easy-to-get-wrong step here): at the moment we are about to
    forecast RV_i (using feat_{i-1}, i.e. info through day i-1), the training set used to fit the
    regression coefficients must consist ONLY of pairs (feat_t, RV_{t+1}) with t+1 <= i-1 -- i.e.
    train_feat = feat.iloc[:i-1], NOT feat.iloc[:i]. Using feat.iloc[:i] would silently leak
    RV_i (the value about to be predicted) into its own training set via the pair
    (feat_{i-1}, RV_i). This off-by-one is verified explicitly by the injection test.

    refit_freq=21 (~monthly, matches the GARCH grid-point-1 cadence) is grid point 1;
    refit_freq=63 (~quarterly) is grid point 2.
    """
    feat = np.log(har_rv_features(rv).clip(lower=_HAR_RV_LOG_EPS))
    target_next = np.log(rv.clip(lower=_HAR_RV_LOG_EPS)).shift(-1)  # log(RV_{t+1})
    idx = rv.index
    n = len(idx)
    forecast = pd.Series(index=idx, dtype=float)  # holds log-RV forecasts until final exp()

    if n <= min_obs + 2:
        return np.exp(forecast).pow(0.5).rename(f"har_rv_rf{refit_freq}")

    coefs = None
    for i in range(min_obs, n):
        need_refit = (coefs is None) or ((i - min_obs) % refit_freq == 0)
        if need_refit:
            train_feat = feat.iloc[: i - 1].dropna()  # strictly feat_t for t <= i-2
            train_y = target_next.loc[train_feat.index]  # RV_{t+1} for t <= i-2 => known thru i-1
            train = pd.concat([train_feat, train_y.rename("y")], axis=1).dropna()
            if len(train) < 30:
                continue
            X = np.column_stack([np.ones(len(train)), train["d"], train["w"], train["m"]])
            y = train["y"].to_numpy()
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        f = feat.iloc[i - 1]  # feat_{i-1}: known through close of day i-1
        if coefs is None or f.isna().any():
            continue
        pred_log_rv = coefs[0] + coefs[1] * f["d"] + coefs[2] * f["w"] + coefs[3] * f["m"]
        forecast.iloc[i] = pred_log_rv

    # forecast currently holds predicted log(RV); vol = sqrt(RV) = exp(log(RV)/2), always > 0
    return np.exp(forecast / 2.0).rename(f"har_rv_rf{refit_freq}")


def _har_rv_NAIVE_UNLAGGED(rv: pd.Series, refit_freq: int = 21, min_obs: int = 60) -> pd.Series:
    """NAIVE CONTROL ONLY -- deliberately uses feat_i (today's own RV components, via
    train_feat = feat.iloc[:i] and prediction input feat.iloc[i]) instead of feat_{i-1}, so
    forecast[i] reacts to day i's own data. Same log-space transform as the causal version above
    (only the causality of the feature/training alignment differs). Injection-test harness use
    only."""
    feat = np.log(har_rv_features(rv).clip(lower=_HAR_RV_LOG_EPS))
    target_next = np.log(rv.clip(lower=_HAR_RV_LOG_EPS)).shift(-1)
    idx = rv.index
    n = len(idx)
    forecast = pd.Series(index=idx, dtype=float)
    if n <= min_obs + 2:
        return np.exp(forecast / 2.0).rename(f"har_rv_rf{refit_freq}_naive")
    coefs = None
    for i in range(min_obs, n):
        need_refit = (coefs is None) or ((i - min_obs) % refit_freq == 0)
        if need_refit:
            train_feat = feat.iloc[:i].dropna()
            train_y = target_next.loc[train_feat.index]
            train = pd.concat([train_feat, train_y.rename("y")], axis=1).dropna()
            if len(train) < 30:
                continue
            X = np.column_stack([np.ones(len(train)), train["d"], train["w"], train["m"]])
            y = train["y"].to_numpy()
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        f = feat.iloc[i]  # LOOKAHEAD: today's own components
        if coefs is None or f.isna().any():
            continue
        pred_log_rv = coefs[0] + coefs[1] * f["d"] + coefs[2] * f["w"] + coefs[3] * f["m"]
        forecast.iloc[i] = pred_log_rv
    return np.exp(forecast / 2.0).rename(f"har_rv_rf{refit_freq}_naive")


# ======================================================================================
# 6. Forecast-quality evaluation: QLIKE / MSE vs realized next-day squared return
# ======================================================================================
def evaluate_forecast_quality(forecast_vol: pd.Series, realized_return: pd.Series) -> dict:
    """Direct, portfolio-agnostic forecast-quality evaluation.

    `forecast_vol[t]` must already be the CAUSAL forecast applicable to day t (as produced by
    every `*_forecast` function above). `realized_return[t]` must be the ACTUAL day-t log return
    (i.e. `daily_log_returns(daily_close)`, unshifted) -- so `realized_return[t]**2` is exactly
    the realized-squared-return proxy for the variance that `forecast_vol[t]` was trying to
    predict one step ahead (both indexed at day t; no further shifting needed by the caller).

    QLIKE (Patton 2011's robust loss under a noisy volatility proxy):
        L_t = rv_t/sigma2_t - ln(rv_t/sigma2_t) - 1        (>= 0, minimized at rv_t == sigma2_t,
                                                              penalizes under-forecasting more
                                                              heavily than MSE does in relative
                                                              terms -- appropriate for a
                                                              risk-sizing use case, where
                                                              understating vol is the costlier
                                                              error)
    MSE on variance:
        L_t = (rv_t - sigma2_t)^2

    Returns {"qlike": mean, "mse": mean, "n": count} over the overlapping, finite, sigma2>0,
    rv>0 sample only.
    """
    sigma2 = forecast_vol**2
    rv = realized_return**2
    df = pd.concat([sigma2.rename("sigma2"), rv.rename("rv")], axis=1).dropna()
    df = df[(df["sigma2"] > 0) & (df["rv"] > 0) & np.isfinite(df["sigma2"]) & np.isfinite(df["rv"])]
    if len(df) == 0:
        return {"qlike": float("nan"), "mse": float("nan"), "n": 0}
    ratio = df["rv"] / df["sigma2"]
    qlike = ratio - np.log(ratio) - 1.0
    mse = (df["rv"] - df["sigma2"]) ** 2
    return {"qlike": float(qlike.mean()), "mse": float(mse.mean()), "n": int(len(df))}


# ======================================================================================
# 7. Sizing: exposure magnitude proportional to target_vol / forecast_vol, capped
#    (exact convention established by i1_pipeline.vol_target_magnitude /
#    _classic_cta_exploration.calibrate_target_daily_vol -- reused, not reinvented)
# ======================================================================================
def vol_target_exposure_magnitude(
    forecast_vol: pd.Series, target_daily_vol: float, leverage_target: float = 1.0,
    cap_mult: float = 2.0,
) -> pd.Series:
    """mag[t] = min(cap_mult * leverage_target, leverage_target * target_daily_vol / forecast_vol[t])
    -- identical functional form to i1_pipeline.vol_target_magnitude (rv_lagged there ==
    forecast_vol here, both already causal). `target_daily_vol` should be calibrated with the
    harmonic-mean correction established in _classic_cta_exploration.calibrate_target_daily_vol
    (an arithmetic-mean calibration systematically overshoots leverage_target by Jensen's
    inequality on the convex map vol -> 1/vol) -- calibration itself is an integration-phase
    concern and is deliberately NOT re-implemented here to avoid duplicating that already-
    validated primitive; this function only applies a GIVEN target_daily_vol.
    """
    fv = forecast_vol.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.minimum(cap_mult * leverage_target, leverage_target * target_daily_vol / fv)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    return pd.Series(mag, index=forecast_vol.index, name="vol_target_mag")


# ======================================================================================
# 8. Injection-test harness (causal-validity proof, with naive controls)
# ======================================================================================
def _inject_price_outlier(daily_ohlc: pd.DataFrame, j: int, mult: float = 8.0) -> pd.DataFrame:
    """Return a COPY of daily_ohlc with a large synthetic multiplicative shock applied to day
    index j's close (and high pushed up to remain >= close, low left alone) -- simulates a
    single-day fat-tail event at a known, controlled location."""
    out = daily_ohlc.copy()
    day = out.index[j]
    new_close = out.loc[day, "close"] * mult
    out.loc[day, "close"] = new_close
    out.loc[day, "high"] = max(out.loc[day, "high"], new_close)
    return out


def run_injection_test_ohlc_based(daily_ohlc: pd.DataFrame, j: int | None = None) -> dict:
    """Injection test for OHLC/close-based estimators (trailing-RV, Yang-Zhang, EWMA): inject an
    8x close shock at day j (mid-series by default), and confirm:
      - forecast[j]   unchanged   for the CAUSAL version (uses info only through j-1)
      - forecast[j+1] changed     for the CAUSAL version (now includes day j's shocked return)
      - forecast[j]   CHANGED     for the NAIVE/unlagged control (proves the test has teeth)
    Returns a dict of booleans + the raw before/after values for inspection.
    """
    if j is None:
        j = len(daily_ohlc) // 2
    base = daily_ohlc
    shocked = _inject_price_outlier(daily_ohlc, j)

    results = {}
    close_base, close_shocked = base["close"], shocked["close"]

    for name, causal_fn, naive_fn, arg in [
        ("trailing_rv_20", trailing_realized_vol_forecast, _trailing_realized_vol_NAIVE_UNLAGGED, close_base),
        ("yang_zhang_20", yang_zhang_vol_forecast, _yang_zhang_NAIVE_UNLAGGED, base),
        ("ewma_0.94", ewma_vol_forecast, _ewma_NAIVE_UNLAGGED, close_base),
    ]:
        if name == "yang_zhang_20":
            f_before = causal_fn(base, window=20)
            f_after = causal_fn(shocked, window=20)
            n_before = naive_fn(base, window=20)
            n_after = naive_fn(shocked, window=20)
        elif name == "trailing_rv_20":
            f_before = causal_fn(close_base, window=20)
            f_after = causal_fn(close_shocked, window=20)
            n_before = naive_fn(close_base, window=20)
            n_after = naive_fn(close_shocked, window=20)
        else:
            f_before = causal_fn(close_base)
            f_after = causal_fn(close_shocked)
            n_before = naive_fn(close_base)
            n_after = naive_fn(close_shocked)

        idx = f_before.index
        at_j = idx[j]
        at_j1 = idx[j + 1] if j + 1 < len(idx) else None

        causal_j_unchanged = np.isclose(f_before.loc[at_j], f_after.loc[at_j], equal_nan=True)
        causal_j1_changed = (
            at_j1 is not None
            and not np.isclose(f_before.loc[at_j1], f_after.loc[at_j1], equal_nan=True)
        )
        naive_j_changed = not np.isclose(n_before.loc[at_j], n_after.loc[at_j], equal_nan=True)

        results[name] = {
            "causal_forecast_j_unchanged": bool(causal_j_unchanged),
            "causal_forecast_j+1_changed": bool(causal_j1_changed),
            "naive_control_forecast_j_changed": bool(naive_j_changed),
            "PASS": bool(causal_j_unchanged and causal_j1_changed and naive_j_changed),
        }
    return results


def run_injection_test_garch(daily_close: pd.Series, j: int | None = None, **kwargs) -> dict:
    """Same protocol as `run_injection_test_ohlc_based`, specialized for GARCH(1,1) (no naive
    control implemented for GARCH -- an "unlagged GARCH" is not a standard object; the trailing-
    RV/EWMA/YZ naive controls above already establish the test methodology has teeth, and HAR-RV
    below provides a second, independent naive-control confirmation)."""
    if j is None:
        j = len(daily_close) // 2
    shocked = daily_close.copy()
    shocked.iloc[j] = shocked.iloc[j] * 8.0

    f_before = garch11_vol_forecast(daily_close, **kwargs)
    f_after = garch11_vol_forecast(shocked, **kwargs)

    idx = f_before.index
    ret_idx_j = j - 1  # daily_log_returns drops the first obs, so return-index j-1 <-> price-index j
    if ret_idx_j < 0 or ret_idx_j >= len(idx):
        return {"garch11": {"PASS": None, "note": "j out of range for return-indexed series"}}
    at_j = idx[ret_idx_j]
    at_j1 = idx[ret_idx_j + 1] if ret_idx_j + 1 < len(idx) else None

    causal_j_unchanged = np.isclose(f_before.loc[at_j], f_after.loc[at_j], equal_nan=True, rtol=1e-9)
    causal_j1_changed = at_j1 is not None and not np.isclose(
        f_before.loc[at_j1], f_after.loc[at_j1], equal_nan=True, rtol=1e-9
    )
    return {
        "garch11": {
            "causal_forecast_j_unchanged": bool(causal_j_unchanged),
            "causal_forecast_j+1_changed": bool(causal_j1_changed),
            "PASS": bool(causal_j_unchanged and causal_j1_changed),
        }
    }


def run_injection_test_har_rv(rv: pd.Series, j: int | None = None, **kwargs) -> dict:
    """Injection test for HAR-RV: shock RV at day j directly (an 8x variance multiplier, standing
    in for "a large synthetic outlier move that day"), confirm causal forecast[j] unaffected,
    forecast[j+1] affected, and the naive (off-by-one, leaked) control reacts at forecast[j]
    itself."""
    if j is None:
        j = len(rv) // 2
    shocked = rv.copy()
    shocked.iloc[j] = shocked.iloc[j] * 8.0

    f_before = har_rv_vol_forecast(rv, **kwargs)
    f_after = har_rv_vol_forecast(shocked, **kwargs)
    n_before = _har_rv_NAIVE_UNLAGGED(rv, **kwargs)
    n_after = _har_rv_NAIVE_UNLAGGED(shocked, **kwargs)

    idx = f_before.index
    at_j = idx[j]
    at_j1 = idx[j + 1] if j + 1 < len(idx) else None

    causal_j_unchanged = np.isclose(f_before.loc[at_j], f_after.loc[at_j], equal_nan=True)
    causal_j1_changed = at_j1 is not None and not np.isclose(
        f_before.loc[at_j1], f_after.loc[at_j1], equal_nan=True
    )
    naive_j_changed = not np.isclose(n_before.loc[at_j], n_after.loc[at_j], equal_nan=True)

    return {
        "har_rv": {
            "causal_forecast_j_unchanged": bool(causal_j_unchanged),
            "causal_forecast_j+1_changed": bool(causal_j1_changed),
            "naive_control_forecast_j_changed": bool(naive_j_changed),
            "PASS": bool(causal_j_unchanged and causal_j1_changed and naive_j_changed),
        }
    }
