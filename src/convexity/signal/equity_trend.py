"""Equity trend book signal + policy (PAPER.md §12.1) — the D1v2 deployable.

An EXACT production port of the D1-validated construction (reference scripts pinned at
``tests/fixtures/equity_book/``; validated: CAGR 28.26%, maxDD 30.02%, Sharpe 1.1676, realized
avg leverage 1.1103). Owner decision of record: C1's optimized construction was research-rejected;
this SIMPLE ROBUST construction is deployed deliberately — timed beta, small allocation.

Construction (verbatim per I1_PREREG.md S1 + the D1 drawdown-overlay waterfall result):
  SIGNAL per name, per day — equal-weight ensemble of three 0/1 rules → state ∈ {0,1/3,2/3,1}:
    (a) 200-day SMA ±2% hysteresis band (daily; hold prior state inside the band)
    (b) 12-month (252-trading-day) total-return sign, evaluated MONTHLY at each month's last
        real trading day, broadcast to every day of the FOLLOWING month
    (c) 10-month SMA of month-end closes (Meb Faber), same monthly evaluate/broadcast
  LONG ONLY. A name is eligible only once all three rules are computable (~260 trading days —
  the point-in-time warmup; no backfilling before real listing).
  SIZING: w_unit = state × inv(rvol20 floored at 0.5× cross-sectional median among eligible),
  normalized to unit gross across active names.
  BOOK GROSS: L_HALF × breadth_frac, breadth_frac = Σ(states over eligible)/n_eligible — the
  breadth-proportional design (NO renormalization to unit gross; zero parameters).
  BRAKE: m_t = clip(1 − DD_{t−1}/0.50, 0, 1) on the book's OWN equity running peak (causal).
  Decide Friday close, act next session; weights constant intra-week; P&L marked daily.

L_HALF is a FROZEN constant (half-Kelly derived once from the validated construction's own
empirical return distribution). Deliberately NOT re-derived at boot — a boot-time re-derivation
would drift with new data; ``tests/paper/test_equity_parity.py`` re-derives it from the pinned
fixtures and asserts equality, keeping provenance verifiable without runtime drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from convexity.paper.fleetconfig import (
    EQUITY_BRAKE_CEILING,
    EQUITY_ETF_EXCLUSIONS,
    EQUITY_L_HALF,
    EQUITY_WARMUP_DAYS,
)

# ======================================================================================
# Per-name signal rules (exact ports of the validated reference implementation)
# ======================================================================================


def compute_state_a(close: pd.Series) -> pd.Series:
    """(a) 200-day SMA ±2% hysteresis band: long above SMA×1.02, flat below SMA×0.98, HOLD prior
    state inside the band; first eligible day inside the band defaults flat. NaN until the SMA is
    valid. Computed through the day's OWN close — the caller applies it from the NEXT session."""
    sma200 = close.rolling(200, min_periods=200).mean()
    upper = (sma200 * 1.02).to_numpy()
    lower = (sma200 * 0.98).to_numpy()
    c = close.to_numpy()
    n = len(c)
    out = np.full(n, np.nan)
    prev = np.nan
    for i in range(n):
        if np.isnan(upper[i]):
            continue
        if c[i] > upper[i]:
            prev = 1.0
        elif c[i] < lower[i]:
            prev = 0.0
        elif np.isnan(prev):
            prev = 0.0
        out[i] = prev
    return pd.Series(out, index=close.index)


def month_end_trading_days(idx: pd.DatetimeIndex) -> pd.Series:
    """Last REAL trading day per calendar month (period → date)."""
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    periods = idx_naive.to_period("M")
    df = pd.DataFrame({"date": idx, "period": periods})
    return df.groupby("period")["date"].max()


def compute_state_b_daily(close: pd.Series, med: pd.Series) -> pd.Series:
    """(b) trailing-252-trading-day total-return sign at each month's last trading day, broadcast
    to every day of the FOLLOWING month (never its own)."""
    ret_252 = close.pct_change(252)
    b_decision = {}
    for p, d in med.items():
        r = ret_252.get(d, np.nan)
        b_decision[p] = np.nan if pd.isna(r) else (1.0 if r > 0 else 0.0)
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    prev_periods = idx_naive.to_period("M") - 1
    out = np.array([b_decision.get(p, np.nan) for p in prev_periods])
    return pd.Series(out, index=close.index)


def compute_state_c_daily(close: pd.Series, med: pd.Series) -> pd.Series:
    """(c) month-end close vs its trailing 10-month-end SMA (Faber), broadcast to the FOLLOWING
    month."""
    month_end_close = close.loc[med.to_numpy()]
    month_end_close.index = med.index
    sma10 = month_end_close.rolling(10, min_periods=10).mean()
    c_decision = {}
    for p in med.index:
        mec, s10 = month_end_close.loc[p], sma10.loc[p]
        c_decision[p] = np.nan if pd.isna(s10) else (1.0 if mec > s10 else 0.0)
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    prev_periods = idx_naive.to_period("M") - 1
    out = np.array([c_decision.get(p, np.nan) for p in prev_periods])
    return pd.Series(out, index=close.index)


def name_signal(close: pd.Series) -> dict | None:
    """Full per-name signal state from a daily close series. None if under the warmup."""
    close = close.dropna()
    if len(close) < EQUITY_WARMUP_DAYS:
        return None
    simple_ret = close.pct_change()
    state_a = compute_state_a(close)
    med = month_end_trading_days(close.index)
    state_b = compute_state_b_daily(close, med)
    state_c = compute_state_c_daily(close, med)
    ens = pd.concat([state_a, state_b, state_c], axis=1)
    ens.columns = ["a", "b", "c"]
    eligible = ens.notna().all(axis=1)
    state = ens.mean(axis=1)
    state[~eligible] = np.nan
    rvol20 = simple_ret.rolling(20, min_periods=20).std()
    return {"close": close, "simple_ret": simple_ret, "state": state, "rvol20": rvol20}


# ======================================================================================
# Book construction (exact ports)
# ======================================================================================


def unitgross_weight_and_breadth(
    state_snap: np.ndarray, rvol_snap: np.ndarray
) -> tuple[np.ndarray, int, int, float]:
    """One decision day's unit-gross weights + breadth (the validated formula, verbatim).

    Returns (w_unit, n_active, n_eligible, breadth_frac); breadth_frac = Σ(states over
    ELIGIBLE)/n_eligible — the state SUM, not the active count (a 1/3-state name contributes
    1/3 of a fully-trending name to breadth)."""
    n_names = len(state_snap)
    eligible = ~np.isnan(state_snap) & ~np.isnan(rvol_snap) & (rvol_snap > 0)
    active = eligible & (state_snap > 0)
    w = np.zeros(n_names)
    if active.any():
        vfloor = 0.5 * np.nanmedian(rvol_snap[eligible]) if eligible.any() else 1e-6
        vfloor = vfloor if (np.isfinite(vfloor) and vfloor > 0) else 1e-6
        inv_vol = 1.0 / np.maximum(rvol_snap[active], vfloor)
        raw = state_snap[active] * inv_vol
        raw_sum = raw.sum()
        if raw_sum > 0:
            w[active] = raw / raw_sum
    n_eligible = int(eligible.sum())
    breadth_frac = (float(np.nansum(state_snap[eligible])) / n_eligible) if n_eligible > 0 else 0.0
    return w, int(active.sum()), n_eligible, breadth_frac


def brake_multiplier(equity_chain: np.ndarray, ceiling: float = EQUITY_BRAKE_CEILING) -> float:
    """GZ-style linear drawdown brake off the book's OWN equity chain THROUGH YESTERDAY.

    m = clip(1 − DD_prev/ceiling, 0, 1) where DD_prev is the drawdown of the LAST equity value
    from the chain's running peak. Causal by construction — the caller must pass equity marks
    strictly through the prior close (journal ``equity_mark`` chain + latest recon unrealized).
    An empty/one-point chain (fresh book) has no drawdown: m = 1."""
    eq = np.asarray([e for e in equity_chain if np.isfinite(e) and e > 0], dtype=float)
    if len(eq) == 0:
        return 1.0
    peak = float(np.max(eq))
    dd_prev = 1.0 - eq[-1] / peak if peak > 0 else 0.0
    return float(min(1.0, max(0.0, 1.0 - dd_prev / ceiling)))


# ======================================================================================
# Cost/metrics helpers (ported verbatim from the validated reference — used by the parity
# backtest driver and by tests; the LIVE settle path uses the paper package's own accounting)
# ======================================================================================


def apply_costs(weights_df: pd.DataFrame, returns_df: pd.DataFrame, cost_bp: float,
                funding_bp_week: float) -> np.ndarray:
    idx = weights_df.index
    w = weights_df.fillna(0.0).to_numpy()
    r = returns_df.reindex(columns=weights_df.columns).fillna(0.0).to_numpy()
    delta_w = np.diff(w, axis=0, prepend=w[[0]])
    delta_w[0, :] = w[0, :]
    trading_cost = (cost_bp / 1e4) * np.abs(delta_w)
    cal_days = pd.Series(idx).diff().dt.days.fillna(1.0).to_numpy()
    cal_days = np.where(cal_days <= 0, 1.0, cal_days)
    funding_cost = (funding_bp_week / 1e4 / 7.0) * cal_days[:, None] * np.abs(w)
    gross = w * r
    net_per_name = gross - trading_cost - funding_cost
    return np.nansum(net_per_name, axis=1)


def cagr_maxdd(net_ret: np.ndarray, dates: pd.DatetimeIndex) -> tuple[float, float]:
    n = len(net_ret)
    if n < 2:
        return np.nan, np.nan
    span_years = (dates[-1] - dates[0]).days / 365.25
    if span_years <= 0:
        return np.nan, np.nan
    equity = np.cumprod(1.0 + net_ret)
    final_eq = equity[-1]
    cagr = (final_eq ** (1.0 / span_years) - 1.0) if final_eq > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, np.nan)
    max_dd = float(np.nanmax(dd)) if len(dd) else np.nan
    return float(cagr), max_dd


def derive_periods_per_yr(dates: pd.DatetimeIndex) -> float:
    n_obs = len(dates)
    if n_obs < 2:
        return 252.0
    span_years = (dates[-1] - dates[0]).days / 365.25
    return n_obs / span_years if span_years > 0 else 252.0


# ======================================================================================
# Backtest driver (the PARITY code path — same signal/sizing/brake functions production uses,
# driven over a historical daily-close panel; asserts nothing itself, the parity TEST does)
# ======================================================================================


def run_brake_on_prescaled(r_scaled: np.ndarray, lev_scaled: np.ndarray,
                           dd_ceiling: float = EQUITY_BRAKE_CEILING) -> dict:
    """The validated post-hoc brake recursion (notional scaling, no incremental trading cost —
    the reference precedent), verbatim."""
    n_ = len(r_scaled)
    wealth = np.empty(n_ + 1)
    peak = np.empty(n_ + 1)
    m_arr = np.empty(n_)
    net = np.empty(n_)
    eff_lev = np.empty(n_)
    wealth[0] = 1.0
    peak[0] = 1.0
    for t in range(n_):
        dd_prev = 1.0 - wealth[t] / peak[t] if peak[t] > 0 else 0.0
        m_t = min(1.0, max(0.0, 1.0 - dd_prev / dd_ceiling))
        m_arr[t] = m_t
        rt = r_scaled[t] if np.isfinite(r_scaled[t]) else 0.0
        net[t] = m_t * rt
        eff_lev[t] = m_t * lev_scaled[t]
        wealth[t + 1] = wealth[t] * (1.0 + net[t])
        peak[t + 1] = max(peak[t], wealth[t + 1])
    return {"m": m_arr, "wealth": wealth, "peak": peak, "net": net, "eff_lev": eff_lev}


def backtest_book(closes: dict[str, pd.Series], *, cost_bp: float = 13.0,
                  l_half: float = EQUITY_L_HALF,
                  dd_ceiling: float = EQUITY_BRAKE_CEILING) -> dict:
    """Run the FULL production policy chain over a historical daily-close panel.

    ``closes``: symbol → daily close Series (the parity test feeds duka-only closes rebuilt from
    the stitched lake; the same functions serve the live path via ``cli._equity_targets``).
    Returns net-return series + headline metrics for the braked, breadth-scaled, half-Kelly book.
    """
    cache: dict[str, dict] = {}
    for sym, close in closes.items():
        sig = name_signal(close)
        if sig is not None:
            cache[sym] = sig
    if not cache:
        raise ValueError("no name passed the signal warmup")
    active_names = sorted(cache.keys())
    idx = None
    for sym in active_names:
        idx = cache[sym]["close"].index if idx is None else idx.union(cache[sym]["close"].index)
    idx = idx.sort_values()

    RET = pd.DataFrame({s: cache[s]["simple_ret"].reindex(idx) for s in active_names})
    STATE = pd.DataFrame({s: cache[s]["state"].reindex(idx) for s in active_names}).ffill(limit=5)
    RVOL = pd.DataFrame({s: cache[s]["rvol20"].reindex(idx) for s in active_names}).ffill(limit=5)

    dow = idx.dayofweek.to_numpy()
    fri_positions = np.where(dow == 4)[0]
    if len(fri_positions) < 2:
        raise ValueError("panel too short: fewer than 2 Fridays")
    n_days, n_names = len(idx), len(active_names)
    STATE_np, RVOL_np = STATE.to_numpy(), RVOL.to_numpy()

    weights_daily = np.zeros((n_days, n_names))
    weekly_breadth, weekly_active, weekly_eligible = [], [], []
    for k in range(len(fri_positions) - 1):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        w_unit, n_active, n_eligible, breadth = unitgross_weight_and_breadth(
            STATE_np[f], RVOL_np[f])
        weights_daily[f + 1:f_next + 1, :] = w_unit * (l_half * breadth)
        weekly_breadth.append(breadth)
        weekly_active.append(n_active)
        weekly_eligible.append(n_eligible)

    lo, hi = fri_positions[0] + 1, fri_positions[-1] + 1
    dates_scored = idx[lo:hi]
    W = pd.DataFrame(weights_daily[lo:hi], index=dates_scored, columns=active_names)
    R = RET.loc[dates_scored]

    r_pre_brake = apply_costs(W, R, cost_bp=cost_bp, funding_bp_week=0.0)
    lev_daily = W.sum(axis=1).to_numpy()
    braked = run_brake_on_prescaled(r_pre_brake, lev_daily, dd_ceiling)

    ppy = derive_periods_per_yr(dates_scored)
    cagr, maxdd = cagr_maxdd(braked["net"], dates_scored)
    mu, sd = np.nanmean(braked["net"]), np.nanstd(braked["net"], ddof=1)
    sharpe = mu / sd * np.sqrt(ppy) if sd > 0 else np.nan
    return {
        "dates": dates_scored, "net": braked["net"], "m": braked["m"],
        "eff_lev": braked["eff_lev"], "pre_brake_net": r_pre_brake,
        "cagr": cagr, "maxdd": maxdd, "sharpe": float(sharpe),
        "realized_avg_leverage": float(braked["eff_lev"].mean()),
        "periods_per_yr": float(ppy),
        "weekly_breadth": np.array(weekly_breadth),
        "weekly_active": np.array(weekly_active),
        "weekly_eligible": np.array(weekly_eligible),
        "active_names": active_names,
    }


# ======================================================================================
# Live decision surface (consumed by paper/cli._equity_targets)
# ======================================================================================


def live_decision(closes: dict[str, pd.Series], equity_chain: np.ndarray,
                  flagged: frozenset[str] = frozenset()) -> dict:
    """One live rebalance decision from per-name daily closes + the book's own equity history.

    Applies the ETF exclusions and split-watch ``flagged`` names BEFORE signal computation.
    Returns w_unit (aligned to ``symbols``), g_t = L_HALF × breadth × m_brake, and diagnostics.
    """
    usable = {s: c for s, c in closes.items()
              if s not in EQUITY_ETF_EXCLUSIONS and s not in flagged}
    cache: dict[str, dict] = {}
    pending: list[str] = []
    for sym, close in usable.items():
        sig = name_signal(close)
        if sig is None:
            pending.append(sym)
        else:
            cache[sym] = sig
    symbols = sorted(cache.keys())
    if not symbols:
        return {"symbols": [], "w_unit": np.zeros(0), "g_t": 0.0, "m_brake": 1.0,
                "breadth": 0.0, "n_eligible": 0, "n_active": 0, "pending": sorted(pending),
                "excluded": sorted(set(closes) - set(usable))}
    state_snap = np.array([cache[s]["state"].iloc[-1] for s in symbols])
    rvol_snap = np.array([cache[s]["rvol20"].iloc[-1] for s in symbols])
    w_unit, n_active, n_eligible, breadth = unitgross_weight_and_breadth(state_snap, rvol_snap)
    m_brake = brake_multiplier(equity_chain)
    g_t = EQUITY_L_HALF * breadth * m_brake
    return {
        "symbols": symbols, "w_unit": w_unit, "g_t": float(g_t), "m_brake": float(m_brake),
        "breadth": float(breadth), "n_eligible": int(n_eligible), "n_active": int(n_active),
        "pending": sorted(pending), "excluded": sorted(set(closes) - set(usable)),
    }
