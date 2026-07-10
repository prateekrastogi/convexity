"""
research/cta/_s1_ensemble_93name.py -- I1's S1 trend-ensemble construction (verbatim per
research/indices/I1_PREREG.md's S1 section), applied for the FIRST TIME to the FULL 93-name
reconciled equity universe (previously only tested on SPY/QQQ in the closed I1 program).

This is a scratch/analysis script (leading underscore, not committed) for a verification/
completion task, NOT a new certified research program. It reuses established, already-validated
machinery rather than reimplementing it:
  - ctaB_pipeline.reconcile_universe()            -- 93-name universe reconciliation
  - ctaB_pipeline.vf.aggregate_daily_ohlc_from_duka() -- duka 1-min -> daily OHLC aggregation
  - ctaB_pipeline.apply_costs() / gross_book_return() -- turnover-scaled cost machinery
  - ctaB_pipeline.regime_window_returns() / REGIME_WINDOWS -- regime windows
  - ctaB_pipeline.derive_periods_per_yr() / _cagr_from_returns()
  - e2_pipeline.py's floored-inverse-vol convention (0.5x cross-sectional median floor)

AAPL split patch: AAPLUSDT's stitched parquet has NOT been fixed for AAPL's real 4-for-1 split
effective 2020-08-31 (fake -74.12% day on 2020-08-31, confirmed below). This script patches
around it IN MEMORY ONLY, using the identical divisor convention as
scripts/stitch_tradfi_1m.py's KNOWN_SPLITS / _split_adjust pattern (back-adjust bars BEFORE the
split by dividing by the split factor). Does NOT touch the stitched parquet on disk -- the
permanent fix still needs to land there (per the D1 deployment prompt's own prerequisite item).

Construction (verbatim per task spec / I1_PREREG.md S1 section):
  SIGNAL (per name, per day), ensemble = equal-weight mean of three 0/1 rules -> {0,1/3,2/3,1}:
    (a) 200-day SMA +-2% hysteresis band (daily, decide-at-close/act-next-session)
    (b) 12-month (252-trading-day) total-return sign, evaluated MONTHLY, held constant intra-month
    (c) 10-month SMA of month-end closes (Meb Faber), evaluated MONTHLY, held constant intra-month
  Long-only, never short. Point-in-time entry: a name is "eligible" only once all three rules are
  computable (no backfilling before real listing).

  SIZING: per-name weight = ensemble_state[name] * (1 / max(rvol20[name], vfloor)), where rvol20 is
  the trailing-20-trading-day realized vol of simple returns (computed through the decision day's
  own close, i.e. "through t-1" relative to the NEXT session/week), vfloor = 0.5x the cross-
  sectional median rvol20 among ELIGIBLE names that day (e2_pipeline.py's floor convention).
  Combined into one book via simple equal-risk-weight aggregation (no correlation adjustment),
  normalized so weights across all active (state>0) names sum to 1.0 (unit gross, long-only).

  REBALANCE CADENCE (explicit convention chosen and stated, per task's own instruction to state
  it explicitly): weights are DECIDED at each Friday's close using signal/vol state as of that
  Friday's own close (decide-at-close), and EXECUTED starting the very next trading session
  (Monday, "act-next-day") -- i.e. the weight vector decided at Friday_k applies to every daily
  return from Friday_k+1 (exclusive) through Friday_{k+1} (inclusive), held CONSTANT all week.
  This is the standard Friday-to-Friday weekly block convention already used elsewhere in this
  codebase (e2_pipeline.py's weekly_forward_name_returns). P&L is marked DAILY throughout (weights
  are constant intra-week, but daily returns still compound day by day) -- periods_per_yr is
  DERIVED from the resulting daily-return array's actual observation spacing (bug pattern 2),
  not assumed to be ~52 just because decisions are weekly.

  COSTS: 13bp round-trip primary on realized turnover (|delta-weight| per name-week), {6.5,26}bp
  scenarios also reported. Funding N/A (unlevered 1x book, per task).

Run: nohup /home/speculation/convexity/.venv/bin/python research/cta/_s1_ensemble_93name.py \
       > research/cta/_s1_ensemble_93name.out 2>&1 &
Writes: research/cta/_s1_ensemble_93name_netret.csv (daily net-return series, NOT committed)
        research/cta/_s1_ensemble_93name_results.json (summary numbers, NOT committed)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO_ROOT}/research/cta")
sys.path.insert(0, f"{REPO_ROOT}/research/cta/lib")
sys.path.insert(0, REPO_ROOT)

import ctaB_pipeline as ctaB   # reuse reconcile_universe, apply_costs, regime windows, etc.

OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
NETRET_CSV = OUT_DIR / "_s1_ensemble_93name_netret.csv"
RESULTS_JSON = OUT_DIR / "_s1_ensemble_93name_results.json"
STITCHED_DIR = ctaB.STITCHED_DIR

# ======================================================================================
# AAPL known unadjusted-split patch (task-specified workaround, in-memory only).
# ======================================================================================
AAPL_SYMBOL = "AAPLUSDT"
AAPL_SPLIT_DATE = "2020-08-31"   # AAPL 4-for-1 split, effective 2020-08-31 (well known)
AAPL_SPLIT_FACTOR = 4


def load_name_frames_patched(symbol: str) -> dict | None:
    fp = STITCHED_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(fp)
    df = df[df["source"] == "duka"].copy()          # bug pattern (1)
    if len(df) == 0:
        return None
    if symbol == AAPL_SYMBOL:
        day_str = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        pre_mask = (day_str < AAPL_SPLIT_DATE).to_numpy()
        divisor = np.where(pre_mask, float(AAPL_SPLIT_FACTOR), 1.0)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c] / divisor
        df["volume"] = df["volume"] * divisor
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = ctaB.vf.aggregate_daily_ohlc_from_duka(df)
    if len(ohlc) < 60:
        return None
    daily_close = ohlc["close"]
    return {
        "symbol": symbol, "close": daily_close,
        "n_obs": len(ohlc),
        "span_years": (ohlc.index[-1] - ohlc.index[0]).days / 365.25,
    }


def aapl_fix_evidence() -> dict:
    """Before/after for the one known bad date, proving the AAPL patch is real."""
    raw = load_name_frames_patched.__wrapped__ if False else None
    # BEFORE: literally reuse ctaB.load_name_frames (unpatched, reads straight from parquet)
    before = ctaB.load_name_frames(AAPL_SYMBOL)
    before_close = before["close"]
    before_ret = before_close.pct_change()
    before_window = before_close.loc["2020-08-25":"2020-09-03"]
    before_worst = float(before_ret.min())
    before_worst_date = str(before_ret.idxmin().date())

    after = load_name_frames_patched(AAPL_SYMBOL)
    after_close = after["close"]
    after_ret = after_close.pct_change()
    after_window = after_close.loc["2020-08-25":"2020-09-03"]
    after_ret_on_date = float(after_ret.loc["2020-08-31"])

    return {
        "before_close_window": {str(k.date()): float(v) for k, v in before_window.items()},
        "before_worst_single_day_return": before_worst,
        "before_worst_single_day_date": before_worst_date,
        "after_close_window": {str(k.date()): float(v) for k, v in after_window.items()},
        "after_return_on_2020_08_31": after_ret_on_date,
    }


# ======================================================================================
# S1 signal construction (verbatim per I1_PREREG.md S1 section)
# ======================================================================================
def compute_state_a(close: pd.Series) -> pd.Series:
    """200-day SMA +-2% hysteresis band. Long when price > SMA*1.02, flat when price < SMA*0.98,
    HOLD prior state inside the band. NaN until SMA_200 first becomes valid (not yet eligible).
    Computed through the day's OWN close (the "decide at close" reference point); the caller is
    responsible for applying this to the NEXT session's return, never the same day's."""
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
            prev = 0.0   # first eligible day landing inside the band with no prior state: default flat
        out[i] = prev
    return pd.Series(out, index=close.index)


def month_end_trading_days(idx: pd.DatetimeIndex) -> pd.Series:
    """Actual last REAL trading day per calendar month (not the calendar month-end label)."""
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx   # UTC-midnight timestamps; safe to drop tz for month grouping
    periods = idx_naive.to_period("M")
    df = pd.DataFrame({"date": idx, "period": periods})
    return df.groupby("period")["date"].max()


def compute_state_b_daily(close: pd.Series, med: pd.Series) -> pd.Series:
    """(b) 12-month (252-trading-day) total-return sign, evaluated monthly. Decision made at each
    month's own last trading day using the trailing-252-trading-day return through THAT day; the
    decision is then broadcast to every day of the FOLLOWING month (never the same month)."""
    ret_252 = close.pct_change(252)
    b_decision = {}
    for p, d in med.items():
        r = ret_252.get(d, np.nan)
        b_decision[p] = np.nan if pd.isna(r) else (1.0 if r > 0 else 0.0)
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    day_periods = idx_naive.to_period("M")
    prev_periods = day_periods - 1
    out = np.array([b_decision.get(p, np.nan) for p in prev_periods])
    return pd.Series(out, index=close.index)


def compute_state_c_daily(close: pd.Series, med: pd.Series) -> pd.Series:
    """(c) 10-month SMA (Meb Faber), evaluated monthly on MONTH-END closes. Decision made at each
    month's own last trading day (month-end price vs trailing 10-month-end SMA including this
    month), broadcast to every day of the FOLLOWING month."""
    month_end_close = close.loc[med.to_numpy()]
    month_end_close.index = med.index
    sma10 = month_end_close.rolling(10, min_periods=10).mean()
    c_decision = {}
    for p in med.index:
        mec, s10 = month_end_close.loc[p], sma10.loc[p]
        c_decision[p] = np.nan if pd.isna(s10) else (1.0 if mec > s10 else 0.0)
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    day_periods = idx_naive.to_period("M")
    prev_periods = day_periods - 1
    out = np.array([c_decision.get(p, np.nan) for p in prev_periods])
    return pd.Series(out, index=close.index)


def build_name_signal(symbol: str):
    try:
        frames = load_name_frames_patched(symbol)
        if frames is None:
            return symbol, None
        close = frames["close"]
        if len(close) < 260:
            return symbol, None
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

        return symbol, {
            "close": close, "simple_ret": simple_ret, "state": state, "rvol20": rvol20,
            "state_a": state_a, "state_b": state_b, "state_c": state_c,
            "n_obs": frames["n_obs"], "span_years": frames["span_years"],
        }
    except Exception as e:
        print(f"[signal] {symbol} FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return symbol, None


# ======================================================================================
# Synthetic-outlier injection test (bug pattern 3: exposure-timing alignment)
# ======================================================================================
def synthetic_injection_test(close: pd.Series) -> dict:
    """Inject a large fake price spike at a single interior day; confirm the daily state_a array
    (and its inputs) BEFORE the injection day is byte-identical to the un-injected baseline (no
    lookahead), and that the state DOES demonstrably react at/after the injection day (proving the
    test has real discriminating power, not a vacuous pass because the state was already
    saturated). Direction of the injected shock is chosen OPPOSITE the baseline state at that day
    (if already long, inject a crash; if flat/inside-band, inject a spike) specifically so a
    ceiling/floor effect can never produce a false-negative "no reaction" reading."""
    n = len(close)
    inject_pos = n // 2
    base_a = compute_state_a(close)
    baseline_state_at_pos = base_a.iloc[inject_pos]
    shock_mult = 0.2 if baseline_state_at_pos == 1.0 else 5.0   # opposite-direction shock

    close_bad = close.copy()
    close_bad.iloc[inject_pos] = close_bad.iloc[inject_pos] * shock_mult

    inj_a = compute_state_a(close_bad)

    pre = base_a.iloc[:inject_pos].to_numpy()
    pre_inj = inj_a.iloc[:inject_pos].to_numpy()
    unaffected = bool(np.allclose(np.nan_to_num(pre, nan=-999), np.nan_to_num(pre_inj, nan=-999)))

    post_differs = not np.allclose(
        np.nan_to_num(base_a.iloc[inject_pos:inject_pos + 5].to_numpy(), nan=-999),
        np.nan_to_num(inj_a.iloc[inject_pos:inject_pos + 5].to_numpy(), nan=-999),
    )
    return {
        "inject_pos": int(inject_pos), "inject_date": str(close.index[inject_pos].date()),
        "baseline_state_at_inject_pos": float(baseline_state_at_pos),
        "shock_multiplier_used": shock_mult,
        "state_before_inject_unaffected": unaffected,
        "state_reacts_at_or_after_inject": bool(post_differs),
    }


# ======================================================================================
# Portfolio construction: weekly Friday-close decisions, daily marks
# ======================================================================================
def main():
    t_start = time.time()
    print(f"=== _s1_ensemble_93name.py starting {pd.Timestamp.now()} ===", flush=True)

    print("\n--- AAPL split-bug fix evidence ---", flush=True)
    aapl_ev = aapl_fix_evidence()
    print(json.dumps(aapl_ev, indent=2), flush=True)

    names = ctaB.reconcile_universe()
    print(f"\n[universe] {len(names)} names", flush=True)

    cache, failed = {}, []
    t0 = time.time()
    for sym in names:
        s, d = build_name_signal(sym)
        if d is None:
            failed.append(sym)
        else:
            cache[s] = d
    print(f"[signal] built {len(cache)}/{len(names)} usable in {time.time()-t0:.1f}s "
          f"(failed/excluded: {failed})", flush=True)

    active_names = sorted(cache.keys())
    idx = None
    for sym in active_names:
        idx = cache[sym]["close"].index if idx is None else idx.union(cache[sym]["close"].index)
    idx = idx.sort_values()
    print(f"[panel] master calendar: {len(idx)} days, {idx[0].date()} .. {idx[-1].date()}", flush=True)

    RET = pd.DataFrame({sym: cache[sym]["simple_ret"].reindex(idx) for sym in active_names})
    STATE = pd.DataFrame({sym: cache[sym]["state"].reindex(idx) for sym in active_names}).ffill(limit=5)
    RVOL = pd.DataFrame({sym: cache[sym]["rvol20"].reindex(idx) for sym in active_names}).ffill(limit=5)

    # ------------------------------------------------------------------------------
    # Synthetic outlier injection test -- run on the longest-history active name.
    # ------------------------------------------------------------------------------
    longest_sym = max(active_names, key=lambda s: cache[s]["n_obs"])
    inj_result = synthetic_injection_test(cache[longest_sym]["close"])
    print(f"\n[injection test] name={longest_sym} -> {inj_result}", flush=True)

    # ------------------------------------------------------------------------------
    # Weekly Friday-close decisions -> daily-broadcast weights
    # ------------------------------------------------------------------------------
    dow = idx.dayofweek.to_numpy()
    fri_positions = np.where(dow == 4)[0]
    print(f"[calendar] {len(fri_positions)} Fridays in master calendar", flush=True)

    n_days, n_names = len(idx), len(active_names)
    weights_daily = np.zeros((n_days, n_names))
    weekly_weight_vecs = []
    weekly_active_counts = []

    STATE_np = STATE.to_numpy()
    RVOL_np = RVOL.to_numpy()

    for k in range(len(fri_positions) - 1):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        state_snap = STATE_np[f]
        rvol_snap = RVOL_np[f]
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
        weights_daily[f + 1:f_next + 1, :] = w
        weekly_weight_vecs.append(w)
        weekly_active_counts.append(int(active.sum()))

    weekly_weight_vecs = np.array(weekly_weight_vecs)   # (n_weeks, n_names)
    weekly_turnover_series = np.abs(np.diff(weekly_weight_vecs, axis=0, prepend=weekly_weight_vecs[[0]])).sum(axis=1)
    weekly_turnover_series[0] = np.abs(weekly_weight_vecs[0]).sum()   # entering from flat

    # Scored sample: first decided week's first trading day through the last decided Friday.
    lo, hi = fri_positions[0] + 1, fri_positions[-1] + 1   # hi exclusive upper bound for slicing
    dates_scored = idx[lo:hi]
    W = pd.DataFrame(weights_daily[lo:hi], index=dates_scored, columns=active_names)
    R = RET.loc[dates_scored]

    print(f"[scored sample] {len(dates_scored)} days, {dates_scored[0].date()} .. {dates_scored[-1].date()}", flush=True)

    net_13 = ctaB.apply_costs(W, R, cost_bp=13.0, funding_bp_week=0.0)
    net_65 = ctaB.apply_costs(W, R, cost_bp=6.5, funding_bp_week=0.0)
    net_26 = ctaB.apply_costs(W, R, cost_bp=26.0, funding_bp_week=0.0)
    gross_1x = ctaB.gross_book_return(W, R)

    periods_per_yr = ctaB.derive_periods_per_yr(dates_scored)
    span_years = (dates_scored[-1] - dates_scored[0]).days / 365.25
    print(f"[annualization] n_obs={len(dates_scored)} / span_years={span_years:.4f} "
          f"-> periods_per_yr={periods_per_yr:.4f} (DERIVED, marked DAILY despite weekly rebalance)",
          flush=True)

    cagr_13, maxdd_13 = ctaB._cagr_from_returns(net_13, dates_scored)
    cagr_65, _ = ctaB._cagr_from_returns(net_65, dates_scored)
    cagr_26, _ = ctaB._cagr_from_returns(net_26, dates_scored)
    mu, sd = np.nanmean(net_13), np.nanstd(net_13, ddof=1)
    sharpe_13 = mu / sd * np.sqrt(periods_per_yr) if sd > 0 else np.nan
    worst_day_13 = float(np.nanmin(net_13))

    # ------------------------------------------------------------------------------
    # Weekly-compounded return series (for EV/week, win rate, worst week)
    # ------------------------------------------------------------------------------
    offset = fri_positions[0] + 1
    weekly_rets, weekly_dates = [], []
    for k in range(len(fri_positions) - 1):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        block_lo, block_hi = f + 1 - offset, f_next - offset + 1
        block = net_13[block_lo:block_hi]
        weekly_rets.append(float(np.prod(1.0 + block) - 1.0))
        weekly_dates.append(idx[f_next])
    weekly_rets = np.array(weekly_rets)
    weekly_dates = pd.DatetimeIndex(weekly_dates)

    ev_per_week_bp = float(np.mean(weekly_rets) * 1e4)
    win_rate_weekly = float(np.mean(weekly_rets > 0))
    worst_week = float(np.min(weekly_rets))
    weekly_turnover_mean = float(np.mean(weekly_turnover_series))
    avg_active_names = float(np.mean(weekly_active_counts))
    daily_win_rate = float(np.mean(net_13[net_13 != 0] > 0)) if np.any(net_13 != 0) else np.nan

    print(f"\n[headline @13bp] CAGR={cagr_13:.4f} maxDD={maxdd_13:.4f} Sharpe={sharpe_13:.4f} "
          f"EV/wk(bp)={ev_per_week_bp:.3f} win_rate(wk)={win_rate_weekly:.4f} "
          f"weekly_turnover={weekly_turnover_mean:.4f} worst_week={worst_week:.4f} "
          f"avg_active_names={avg_active_names:.2f}", flush=True)
    print(f"[cost scenarios] CAGR 6.5bp={cagr_65:.4f} 13bp={cagr_13:.4f} 26bp={cagr_26:.4f}", flush=True)

    # ------------------------------------------------------------------------------
    # Regime breakdown (reuse ctaB.regime_window_returns + REGIME_WINDOWS verbatim, plus the
    # 2025-04 squeeze week which is NOT in that dict -- computed identically by hand).
    # ------------------------------------------------------------------------------
    regime_ret = ctaB.regime_window_returns(net_13, dates_scored)
    ser = pd.Series(np.nan_to_num(net_13), index=dates_scored)
    squeeze_window = ser.loc[(dates_scored >= pd.Timestamp("2025-04-04", tz="UTC")) &
                              (dates_scored <= pd.Timestamp("2025-04-11", tz="UTC"))]
    regime_ret["2025-04_squeeze_week"] = (float(np.prod(1.0 + squeeze_window.to_numpy()) - 1.0)
                                           if len(squeeze_window) else np.nan)
    print(f"[regime returns @13bp] {json.dumps(regime_ret, indent=2)}", flush=True)

    # active-name counts / warmup honesty check per named regime window (mirrors I1's own
    # 2018-Q4 "confirmed data-coverage artifact" honesty check)
    active_count_daily = (W > 0).sum(axis=1)
    regime_active_names = {}
    all_windows = dict(ctaB.REGIME_WINDOWS)
    all_windows["2025-04_squeeze_week"] = ("2025-04-04", "2025-04-11")
    for label, (start, end) in all_windows.items():
        mask = (dates_scored >= pd.Timestamp(start, tz="UTC")) & (dates_scored <= pd.Timestamp(end, tz="UTC"))
        seg = active_count_daily.loc[dates_scored[mask]] if mask.any() else pd.Series(dtype=float)
        regime_active_names[label] = {
            "n_days_in_window": int(mask.sum()),
            "mean_active_names": float(seg.mean()) if len(seg) else None,
            "all_zero": bool((seg == 0).all()) if len(seg) else None,
        }
    print(f"[regime active-name honesty check] {json.dumps(regime_active_names, indent=2)}", flush=True)

    # ------------------------------------------------------------------------------
    # Fresh 93-name equal-weight buy & hold benchmark, on the SAME AAPL-patched RET panel.
    # (a) over the SAME scored date range as the S1 book (apples-to-apples vs S1)
    # (b) over the FULL raw universe date range (apples-to-apples vs the historical ~46.8% figure)
    # ------------------------------------------------------------------------------
    bh_scored = RET.loc[dates_scored].mean(axis=1, skipna=True).fillna(0.0).to_numpy()
    bh_cagr_scored, bh_maxdd_scored = ctaB._cagr_from_returns(bh_scored, dates_scored)
    bh_mu, bh_sd = np.nanmean(bh_scored), np.nanstd(bh_scored, ddof=1)
    bh_sharpe_scored = bh_mu / bh_sd * np.sqrt(periods_per_yr) if bh_sd > 0 else np.nan

    bh_full = RET.mean(axis=1, skipna=True).fillna(0.0).to_numpy()
    ppy_full = ctaB.derive_periods_per_yr(idx)
    bh_cagr_full, bh_maxdd_full = ctaB._cagr_from_returns(bh_full, idx)
    bh_mu_f, bh_sd_f = np.nanmean(bh_full), np.nanstd(bh_full, ddof=1)
    bh_sharpe_full = bh_mu_f / bh_sd_f * np.sqrt(ppy_full) if bh_sd_f > 0 else np.nan

    print(f"\n[benchmark, SAME scored range {dates_scored[0].date()}..{dates_scored[-1].date()}] "
          f"CAGR={bh_cagr_scored:.4f} maxDD={bh_maxdd_scored:.4f} Sharpe={bh_sharpe_scored:.4f}", flush=True)
    print(f"[benchmark, FULL range {idx[0].date()}..{idx[-1].date()}] "
          f"CAGR={bh_cagr_full:.4f} maxDD={bh_maxdd_full:.4f} Sharpe={bh_sharpe_full:.4f} "
          f"(vs. established Fact-6 comparator CAGR~0.468, Sharpe~1.353, maxDD~0.410)", flush=True)

    # ------------------------------------------------------------------------------
    # Write daily net-return series to CSV (Phase 2 needs this directly for Kelly derivation)
    # ------------------------------------------------------------------------------
    out_df = pd.DataFrame({
        "date": dates_scored.tz_convert(None) if dates_scored.tz is not None else dates_scored,
        "net_ret_13bp": net_13,
        "net_ret_6p5bp": net_65,
        "net_ret_26bp": net_26,
        "gross_ret": gross_1x,
        "n_active_names": active_count_daily.to_numpy(),
    })
    out_df.to_csv(NETRET_CSV, index=False)
    print(f"\n[output] wrote {NETRET_CSV} ({len(out_df)} rows, "
          f"{dates_scored[0].date()} .. {dates_scored[-1].date()})", flush=True)

    results = {
        "meta": {
            "universe_n": len(names), "names_usable": len(active_names), "names_failed": failed,
            "master_calendar_span": [str(idx[0].date()), str(idx[-1].date())],
            "scored_span": [str(dates_scored[0].date()), str(dates_scored[-1].date())],
            "n_obs_daily": len(dates_scored), "span_years": span_years,
            "periods_per_yr_derived": periods_per_yr, "n_weeks": len(weekly_rets),
        },
        "aapl_fix_evidence": aapl_ev,
        "injection_test": inj_result,
        "headline_13bp": {
            "cagr": cagr_13, "maxdd": maxdd_13, "sharpe": sharpe_13,
            "ev_per_week_bp": ev_per_week_bp, "win_rate_weekly": win_rate_weekly,
            "win_rate_daily": daily_win_rate, "weekly_turnover_mean": weekly_turnover_mean,
            "worst_week": worst_week, "worst_day": worst_day_13,
            "avg_active_names": avg_active_names,
        },
        "cost_scenarios_cagr": {"6.5bp": cagr_65, "13bp": cagr_13, "26bp": cagr_26},
        "regime_returns_13bp": regime_ret,
        "regime_active_name_honesty_check": regime_active_names,
        "benchmark_fresh_recompute": {
            "same_scored_range": {"cagr": bh_cagr_scored, "maxdd": bh_maxdd_scored, "sharpe": bh_sharpe_scored},
            "full_range": {"cagr": bh_cagr_full, "maxdd": bh_maxdd_full, "sharpe": bh_sharpe_full},
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[output] wrote {RESULTS_JSON}", flush=True)
    print(f"\n=== done in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
