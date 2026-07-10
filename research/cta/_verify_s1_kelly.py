"""
INDEPENDENT VERIFICATION of the I1 S1 93-name ensemble build + Kelly leverage numbers.

Read-only verification task. Reuses ONLY ctaB_pipeline.reconcile_universe() and
ctaB_pipeline.load_name_frames() (already-established/validated data-loading machinery, per the
task's explicit permission). ALL signal construction, sizing, aggregation, cost accounting, Kelly
optimization, and drawdown-brake code below is written fresh/independently -- it does NOT import
or call the build agent's _s1_ensemble_93name.py or _s1_kelly_leverage_analysis.py.

Five checks:
  1. Hand-derive 1x unlevered CAGR/maxDD/Sharpe from raw parquet, compare to reported
     cagr=0.2205 maxdd=0.3873 sharpe=1.06.
  2. Synthetic-outlier injection on the FULL chain (combined 3-rule state + inverse-vol sizing +
     weekly decide/act-next-week weight construction), with a genuine naive/unlagged control
     (decide-and-act-same-day) to prove the test has real discriminating power.
  3. Confirm AAPLUSDT split-bug patch: check 2020-08-31 directly in the series actually used here.
  4. Independent full-Kelly L* via bisection-on-derivative (root of mean[r/(1+L*r)]=0), NOT
     scipy's bounded minimizer -- a different numerical method on the same empirical series.
  5. Drawdown-brake spot check: recompute the brake myself, pull actual multiplier values for
     2020-02-15..2020-05-31 and 2022-01-01..2022-12-31.

Run: nohup /home/speculation/convexity/.venv/bin/python research/cta/_verify_s1_kelly.py \
       > research/cta/_verify_s1_kelly.out 2>&1 &
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time
import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO_ROOT}/research/cta")
sys.path.insert(0, f"{REPO_ROOT}/research/cta/lib")
sys.path.insert(0, REPO_ROOT)

import ctaB_pipeline as ctaB   # ONLY reconcile_universe() and load_name_frames() are reused

OUT_JSON = f"{REPO_ROOT}/research/cta/_verify_s1_kelly_results.json"
BUILD_NETRET_CSV = f"{REPO_ROOT}/research/cta/_s1_ensemble_93name_netret.csv"

t_start = time.time()
print(f"=== _verify_s1_kelly.py starting {pd.Timestamp.now()} ===", flush=True)

# ==========================================================================================
# CHECK 3 prep + fix: independent AAPL split patch (own code, own date check)
# ==========================================================================================
AAPL_SYMBOL = "AAPLUSDT"
AAPL_SPLIT_DATE = pd.Timestamp("2020-08-31", tz="UTC")
AAPL_SPLIT_FACTOR = 4.0


def my_load_close_patched(symbol: str) -> pd.Series | None:
    """Independent loader: reuse ctaB.load_name_frames() for the raw daily-OHLC aggregation
    (permitted reuse), then apply MY OWN AAPL patch in memory on the returned daily close."""
    frames = ctaB.load_name_frames(symbol)
    if frames is None:
        return None
    close = frames["close"].copy()
    if symbol == AAPL_SYMBOL:
        pre = close.index < AAPL_SPLIT_DATE
        close = close.copy()
        close.loc[pre] = close.loc[pre] / AAPL_SPLIT_FACTOR
    if len(close) < 260:
        return None
    return close


# --- Check 3: confirm the split bug and the fix, on MY OWN loaded series ---
print("\n--- CHECK 3: AAPLUSDT split-bug confirmation (independent) ---", flush=True)
raw_frames = ctaB.load_name_frames(AAPL_SYMBOL)
raw_close = raw_frames["close"]
raw_ret = raw_close.pct_change()
check3 = {}
if AAPL_SPLIT_DATE in raw_ret.index:
    check3["unpatched_return_on_2020_08_31"] = float(raw_ret.loc[AAPL_SPLIT_DATE])
else:
    # nearest available index
    idx_near = raw_ret.index[raw_ret.index.get_indexer([AAPL_SPLIT_DATE], method="nearest")[0]]
    check3["unpatched_return_on_nearest_to_2020_08_31"] = {str(idx_near.date()): float(raw_ret.loc[idx_near])}
worst_i = raw_ret.idxmin()
check3["unpatched_worst_single_day_return"] = float(raw_ret.min())
check3["unpatched_worst_single_day_date"] = str(worst_i.date())
window_before = raw_close.loc["2020-08-25":"2020-09-03"]
check3["unpatched_close_window"] = {str(k.date()): float(v) for k, v in window_before.items()}

patched_close = my_load_close_patched(AAPL_SYMBOL)
patched_ret = patched_close.pct_change()
check3["patched_return_on_2020_08_31"] = float(patched_ret.loc[AAPL_SPLIT_DATE])
window_after = patched_close.loc["2020-08-25":"2020-09-03"]
check3["patched_close_window"] = {str(k.date()): float(v) for k, v in window_after.items()}
check3["fake_74pct_crash_gone"] = bool(patched_ret.loc[AAPL_SPLIT_DATE] > -0.20)
check3["patched_worst_single_day_return_overall"] = float(patched_ret.min())
print(json.dumps(check3, indent=2), flush=True)

# ==========================================================================================
# CHECK 1: hand-derived signal / sizing / aggregation -- INDEPENDENT implementation
# ==========================================================================================
print("\n--- CHECK 1: independent signal/sizing/aggregation build ---", flush=True)


def my_state_a(close: pd.Series) -> pd.Series:
    """200d SMA +-2% hysteresis, vectorized (different algorithm than a day-by-day loop):
    mark +1 above upper band, 0 below lower band, NaN (hold) inside band or before SMA valid;
    forward-fill the holds; the first eligible day landing inside the band defaults to 0."""
    sma = close.rolling(200, min_periods=200).mean()
    eligible = sma.notna()
    upper, lower = sma * 1.02, sma * 0.98
    raw = pd.Series(np.nan, index=close.index)
    raw[eligible & (close > upper)] = 1.0
    raw[eligible & (close < lower)] = 0.0
    elig_idx = eligible[eligible].index
    if len(elig_idx):
        first = elig_idx[0]
        if pd.isna(raw.loc[first]):
            raw.loc[first] = 0.0
    state = raw.where(eligible).ffill()
    state[~eligible] = np.nan
    return state


def my_month_end_dates(idx: pd.DatetimeIndex) -> pd.Series:
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    per = idx_naive.to_period("M")
    return pd.Series(idx, index=per).groupby(level=0).last()   # period -> last real trading day


def my_state_b(close: pd.Series, month_end: pd.Series) -> pd.Series:
    """12m (252td) total-return sign, decided at each month's own last trading day, broadcast to
    the FOLLOWING month. Independent construction via period-indexed mapping (no loop)."""
    ret252 = close.pct_change(252)
    dec_at_me = ret252.reindex(month_end.to_numpy())
    dec_at_me.index = month_end.index                      # index by period
    decision_by_period = (dec_at_me > 0).astype(float)
    decision_by_period[dec_at_me.isna()] = np.nan
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    day_period = idx_naive.to_period("M")
    lookup_period = day_period - 1
    out = decision_by_period.reindex(lookup_period).to_numpy()
    return pd.Series(out, index=close.index)


def my_state_c(close: pd.Series, month_end: pd.Series) -> pd.Series:
    """10-month SMA of month-end closes (Meb Faber), decided at month-end, broadcast to next
    month."""
    me_close = close.reindex(month_end.to_numpy())
    me_close.index = month_end.index
    sma10 = me_close.rolling(10, min_periods=10).mean()
    decision_by_period = (me_close > sma10).astype(float)
    decision_by_period[sma10.isna()] = np.nan
    idx_naive = close.index.tz_localize(None) if close.index.tz is not None else close.index
    day_period = idx_naive.to_period("M")
    lookup_period = day_period - 1
    out = decision_by_period.reindex(lookup_period).to_numpy()
    return pd.Series(out, index=close.index)


def build_signal(symbol: str):
    close = my_load_close_patched(symbol)
    if close is None or len(close) < 260:
        return symbol, None
    ret = close.pct_change()
    a = my_state_a(close)
    me = my_month_end_dates(close.index)
    b = my_state_b(close, me)
    c = my_state_c(close, me)
    ens = pd.concat([a, b, c], axis=1)
    ens.columns = ["a", "b", "c"]
    eligible = ens.notna().all(axis=1)
    state = ens.mean(axis=1)
    state[~eligible] = np.nan
    rvol20 = ret.rolling(20, min_periods=20).std()
    return symbol, {"close": close, "ret": ret, "state": state, "rvol20": rvol20,
                     "a": a, "b": b, "c": c, "n_obs": len(close)}


names = ctaB.reconcile_universe()
print(f"[universe] {len(names)} names", flush=True)

cache = {}
t0 = time.time()
for sym in names:
    s, d = build_signal(sym)
    if d is not None:
        cache[s] = d
print(f"[signal] built {len(cache)}/{len(names)} in {time.time()-t0:.1f}s", flush=True)

active_names = sorted(cache.keys())
idx = None
for sym in active_names:
    idx = cache[sym]["close"].index if idx is None else idx.union(cache[sym]["close"].index)
idx = idx.sort_values()
print(f"[panel] {len(idx)} days {idx[0].date()}..{idx[-1].date()}", flush=True)

RET = pd.DataFrame({s: cache[s]["ret"].reindex(idx) for s in active_names})
STATE = pd.DataFrame({s: cache[s]["state"].reindex(idx) for s in active_names}).ffill(limit=5)
RVOL = pd.DataFrame({s: cache[s]["rvol20"].reindex(idx) for s in active_names}).ffill(limit=5)


def decide_weekly_weights(state_df: pd.DataFrame, rvol_df: pd.DataFrame, idx: pd.DatetimeIndex):
    """Friday-close decision -> weight vector per week. Independent re-implementation (numpy
    array loop over Fridays only, same normalization convention as spec: state*inv_vol,
    vfloor = 0.5x cross-sectional median rvol among ELIGIBLE names, normalized to sum 1 across
    active names)."""
    dow = idx.dayofweek.to_numpy()
    fridays = np.where(dow == 4)[0]
    S = state_df.to_numpy()
    V = rvol_df.to_numpy()
    n_names = S.shape[1]
    weekly_w = []
    for f in fridays:
        s_row, v_row = S[f], V[f]
        eligible = ~np.isnan(s_row) & ~np.isnan(v_row) & (v_row > 0)
        active = eligible & (s_row > 0)
        w = np.zeros(n_names)
        if active.any():
            vfloor = 0.5 * np.nanmedian(v_row[eligible]) if eligible.any() else 1e-6
            if not (np.isfinite(vfloor) and vfloor > 0):
                vfloor = 1e-6
            inv_vol = 1.0 / np.maximum(v_row[active], vfloor)
            raw = s_row[active] * inv_vol
            rs = raw.sum()
            if rs > 0:
                w[active] = raw / rs
        weekly_w.append(w)
    return fridays, np.array(weekly_w)


fridays, weekly_w = decide_weekly_weights(STATE, RVOL, idx)
print(f"[calendar] {len(fridays)} Fridays", flush=True)

n_days, n_names = len(idx), len(active_names)
W_daily = np.zeros((n_days, n_names))
for k in range(len(fridays) - 1):
    f, f_next = fridays[k], fridays[k + 1]
    W_daily[f + 1:f_next + 1, :] = weekly_w[k]

lo, hi = fridays[0] + 1, fridays[-1] + 1
dates_scored = idx[lo:hi]
W = W_daily[lo:hi]
R = RET.loc[dates_scored].fillna(0.0).to_numpy()

# --- cost / net-return: my own implementation, 13bp round-trip on realized weekly turnover ---
delta_w = np.diff(W, axis=0, prepend=W[[0]])
delta_w[0] = W[0]
cost = (13.0 / 1e4) * np.abs(delta_w)
gross_per_name = W * R
net_ret = np.nansum(gross_per_name, axis=1) - np.nansum(cost, axis=1)
gross_ret = np.nansum(gross_per_name, axis=1)


def my_cagr_maxdd(net: np.ndarray, dates: pd.DatetimeIndex):
    span_years = (dates[-1] - dates[0]).days / 365.25
    eq = np.cumprod(1.0 + net)
    cagr = eq[-1] ** (1.0 / span_years) - 1.0 if eq[-1] > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return float(cagr), float(dd.max()), eq


ppy = len(dates_scored) / ((dates_scored[-1] - dates_scored[0]).days / 365.25)
cagr1, maxdd1, eq1 = my_cagr_maxdd(net_ret, dates_scored)
mu, sd = net_ret.mean(), net_ret.std(ddof=1)
sharpe1 = mu / sd * np.sqrt(ppy) if sd > 0 else np.nan

check1 = {
    "n_obs": len(dates_scored), "date_range": [str(dates_scored[0].date()), str(dates_scored[-1].date())],
    "periods_per_yr": ppy,
    "hand_derived_cagr": cagr1, "hand_derived_maxdd": maxdd1, "hand_derived_sharpe": float(sharpe1),
    "reported_cagr": 0.2205, "reported_maxdd": 0.3873, "reported_sharpe": 1.06,
    "cagr_diff": cagr1 - 0.2205, "maxdd_diff": maxdd1 - 0.3873, "sharpe_diff": float(sharpe1) - 1.06,
}
print(json.dumps(check1, indent=2), flush=True)

# ==========================================================================================
# CHECK 2: synthetic outlier injection on the FULL chain, with naive/unlagged control
# ==========================================================================================
print("\n--- CHECK 2: full-chain injection test + naive/unlagged control ---", flush=True)
longest_sym = max(active_names, key=lambda s: cache[s]["n_obs"])
sym_col = active_names.index(longest_sym)
print(f"[injection] using name={longest_sym}", flush=True)

# pick an injection index that lands exactly on a decision Friday, well inside the sample,
# with an already-eligible state (not in warmup) so a reaction is meaningful
base_close = cache[longest_sym]["close"]
name_idx = base_close.index
name_dow = name_idx.dayofweek.to_numpy()
name_fridays = np.where(name_dow == 4)[0]
# choose a Friday roughly 60% through the eligible history
base_state_series = cache[longest_sym]["state"]
eligible_positions = np.where(base_state_series.notna().to_numpy())[0]
target_pos_guess = eligible_positions[int(len(eligible_positions) * 0.6)]
inject_pos = name_fridays[np.argmin(np.abs(name_fridays - target_pos_guess))]
inject_date = name_idx[inject_pos]
baseline_state_at_inject = float(base_state_series.iloc[inject_pos])
shock_mult = 0.2 if baseline_state_at_inject >= 0.5 else 5.0

print(f"[injection] date={inject_date.date()} pos={inject_pos} baseline_state={baseline_state_at_inject} "
      f"shock_mult={shock_mult}", flush=True)

shocked_close = base_close.copy()
shocked_close.iloc[inject_pos] = shocked_close.iloc[inject_pos] * shock_mult


def rebuild_one_name_signal(close: pd.Series):
    ret = close.pct_change()
    a = my_state_a(close)
    me = my_month_end_dates(close.index)
    b = my_state_b(close, me)
    c = my_state_c(close, me)
    ens = pd.concat([a, b, c], axis=1)
    ens.columns = ["a", "b", "c"]
    eligible = ens.notna().all(axis=1)
    state = ens.mean(axis=1)
    state[~eligible] = np.nan
    rvol20 = ret.rolling(20, min_periods=20).std()
    return ret, state, rvol20


ret_base, state_base, rvol_base = rebuild_one_name_signal(base_close)
ret_shock, state_shock, rvol_shock = rebuild_one_name_signal(shocked_close)

# (2a) causality of the per-name state/vol arrays themselves: identical strictly BEFORE inject_pos
pre_state_equal = bool(np.allclose(np.nan_to_num(state_base.iloc[:inject_pos].to_numpy(), nan=-999),
                                    np.nan_to_num(state_shock.iloc[:inject_pos].to_numpy(), nan=-999)))
pre_rvol_equal = bool(np.allclose(np.nan_to_num(rvol_base.iloc[:inject_pos].to_numpy(), nan=-999),
                                   np.nan_to_num(rvol_shock.iloc[:inject_pos].to_numpy(), nan=-999)))
state_reacts_at_inject = bool(not np.isclose(
    np.nan_to_num(state_base.iloc[inject_pos], nan=-999),
    np.nan_to_num(state_shock.iloc[inject_pos], nan=-999)))

# (2b) FULL-CHAIN weekly weight vectors, correct (decide@Fri_k, act next week) vs naive/unlagged
#      (decide@Fri_k, act SAME week i.e. Fri_{k-1}+1 .. Fri_k inclusive: no shift). Use the
#      single-name RVOL/STATE panels already built for the full 93-name book (which include this
#      name), but rebuild the FULL BOOK under the shocked close for this one name to see the
#      effect on the actual portfolio-level weight for this symbol and on the book's net return
#      on the injection day itself.


def build_full_state_rvol_with_override(symbol_override: str, close_override: pd.Series):
    state_cols, rvol_cols = {}, {}
    for s in active_names:
        if s == symbol_override:
            ret_o = close_override.pct_change()
            a = my_state_a(close_override)
            me = my_month_end_dates(close_override.index)
            b = my_state_b(close_override, me)
            c = my_state_c(close_override, me)
            ens = pd.concat([a, b, c], axis=1); ens.columns = ["a", "b", "c"]
            elig = ens.notna().all(axis=1)
            st = ens.mean(axis=1); st[~elig] = np.nan
            rv = ret_o.rolling(20, min_periods=20).std()
            state_cols[s] = st.reindex(idx)
            rvol_cols[s] = rv.reindex(idx)
        else:
            state_cols[s] = cache[s]["state"].reindex(idx)
            rvol_cols[s] = cache[s]["rvol20"].reindex(idx)
    STATE_o = pd.DataFrame(state_cols).ffill(limit=5)
    RVOL_o = pd.DataFrame(rvol_cols).ffill(limit=5)
    return STATE_o, RVOL_o


STATE_shocked, RVOL_shocked = build_full_state_rvol_with_override(longest_sym, shocked_close)

fridays2, weekly_w_base = decide_weekly_weights(STATE, RVOL, idx)
fridays2b, weekly_w_shock = decide_weekly_weights(STATE_shocked, RVOL_shocked, idx)
assert np.array_equal(fridays2, fridays2b)

# find which weekly-decision index corresponds to inject_pos's Friday in the MASTER calendar idx
master_pos_of_inject = idx.get_indexer([inject_date])[0]
assert master_pos_of_inject >= 0, "injection date not found in master calendar (unexpected)"
week_k_of_inject = int(np.where(fridays2 == master_pos_of_inject)[0][0]) if master_pos_of_inject in fridays2 else None
print(f"[injection] master_calendar_pos={master_pos_of_inject}, is_decision_friday={week_k_of_inject is not None}", flush=True)

# weekly weight vectors for weeks strictly BEFORE the injection decision: must be identical
weeks_before_equal = bool(np.allclose(weekly_w_base[:week_k_of_inject], weekly_w_shock[:week_k_of_inject]))
# the weight vector DECIDED at the injection Friday itself (uses that day's own close -> allowed to react)
weight_at_inject_week_differs = bool(not np.allclose(weekly_w_base[week_k_of_inject], weekly_w_shock[week_k_of_inject]))

# --- CORRECT construction: broadcast decided-at-Fri_k weight to (Fri_k, Fri_{k+1}] (next week) ---
def broadcast_correct(weekly_w, fridays, n_days, n_names):
    Wd = np.zeros((n_days, n_names))
    for k in range(len(fridays) - 1):
        f, f_next = fridays[k], fridays[k + 1]
        Wd[f + 1:f_next + 1, :] = weekly_w[k]
    return Wd

# --- NAIVE/UNLAGGED control: broadcast decided-at-Fri_k weight to (Fri_{k-1}, Fri_k] (that SAME
#     week, i.e. no execution lag -- the position is applied retroactively to the week whose
#     return the decision-maker could not have known when the week started). This is a genuine,
#     realistic bug class (off-by-one in the broadcast window), not a strawman.
def broadcast_naive_unlagged(weekly_w, fridays, n_days, n_names):
    Wd = np.zeros((n_days, n_names))
    for k in range(len(fridays) - 1):
        f, f_next = fridays[k], fridays[k + 1]
        Wd[f + 1:f_next + 1, :] = weekly_w[k + 1]   # <-- bug: use NEXT week's decision, one step early
    return Wd

Wd_correct_base = broadcast_correct(weekly_w_base, fridays2, n_days, n_names)
Wd_correct_shock = broadcast_correct(weekly_w_shock, fridays2, n_days, n_names)
Wd_naive_base = broadcast_naive_unlagged(weekly_w_base, fridays2, n_days, n_names)
Wd_naive_shock = broadcast_naive_unlagged(weekly_w_shock, fridays2, n_days, n_names)

# Look at the injection day's own row (master_pos_of_inject) across all names' weight vectors:
# CORRECT: that day's weight came from the week decided at fridays2[week_k_of_inject - 1]
#          (BEFORE injection) -> must be IDENTICAL baseline vs shocked.
# NAIVE:   that day's weight came from the week decided at fridays2[week_k_of_inject]
#          (the injection Friday itself, i.e. from the future relative to the week that started
#          right after the PRIOR Friday) -> must DIFFER baseline vs shocked, proving real leakage
#          the test can catch.
row = master_pos_of_inject
correct_injectday_weight_equal = bool(np.allclose(Wd_correct_base[row], Wd_correct_shock[row]))
naive_injectday_weight_differs = bool(not np.allclose(Wd_naive_base[row], Wd_naive_shock[row]))

# Also confirm ALL rows strictly BEFORE the injection master position are identical under the
# CORRECT construction (comprehensive causality check across the full weight matrix, not just one
# name/one day), and that the NAIVE construction genuinely differs somewhere before/at that row
# (proving discriminating power at the full-matrix level too).
correct_all_pre_rows_equal = bool(np.allclose(Wd_correct_base[:row], Wd_correct_shock[:row]))
naive_any_pre_or_at_row_differs = bool(not np.allclose(Wd_naive_base[:row + 1], Wd_naive_shock[:row + 1]))

check2 = {
    "injection_symbol": longest_sym, "injection_date": str(inject_date.date()),
    "injection_master_calendar_pos": int(row),
    "baseline_state_at_injection": baseline_state_at_inject, "shock_multiplier": shock_mult,
    "per_name_state_rvol_causality": {
        "state_unaffected_strictly_before_injection": pre_state_equal,
        "rvol_unaffected_strictly_before_injection": pre_rvol_equal,
        "state_reacts_at_injection_day_itself": state_reacts_at_inject,
    },
    "weekly_decision_causality": {
        "weeks_decided_before_injection_friday_unchanged": weeks_before_equal,
        "week_decided_ON_injection_friday_reacts": weight_at_inject_week_differs,
    },
    "full_chain_correct_construction (decide@Fri_k, act next week)": {
        "injection_day_own_return_weight_unaffected": correct_injectday_weight_equal,
        "all_rows_strictly_before_injection_unaffected": correct_all_pre_rows_equal,
    },
    "naive_unlagged_control (decide@Fri_k, act retroactively on SAME week)": {
        "injection_day_own_return_weight_DIFFERS": naive_injectday_weight_differs,
        "some_row_at_or_before_injection_differs": naive_any_pre_or_at_row_differs,
    },
    "test_has_discriminating_power": bool(correct_injectday_weight_equal and naive_injectday_weight_differs),
}
print(json.dumps(check2, indent=2), flush=True)

# ==========================================================================================
# CHECK 4: independent full-Kelly L* via bisection on the derivative root (NOT scipy minimize)
# ==========================================================================================
print("\n--- CHECK 4: independent Kelly L* (bisection-on-derivative, own return series + build CSV) ---", flush=True)


def kelly_L_star_bisection(r: np.ndarray, l_max: float = 25.0, tol: float = 1e-10, max_iter: int = 200):
    """g(L) = mean(r / (1+L*r)) is the derivative of mean(log(1+L*r)) w.r.t. L; it is strictly
    decreasing in L wherever defined (objective concave), so bisection on g(L)=0 is a totally
    different numerical method than a bounded scalar minimizer, on the identical objective."""
    r = np.asarray(r, dtype="float64")
    r = r[np.isfinite(r)]
    # domain ceiling: 1+L*r>0 for all r -> L < 1/(-min(r)) when min(r)<0
    rmin = r.min()
    hard_ceiling = (1.0 / (-rmin)) * 0.999999 if rmin < 0 else l_max
    hi = min(l_max, hard_ceiling)
    lo = 0.0

    def g(L):
        denom = 1.0 + L * r
        return np.mean(r / denom)

    g_lo, g_hi = g(lo), g(hi)
    if g_lo <= 0:
        return 0.0
    if g_hi >= 0:
        # objective still increasing at the domain ceiling -- optimum is at (or beyond) ceiling
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if abs(gm) < tol or (hi - lo) < 1e-12:
            return mid
        if gm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# (a) on the BUILD agent's own persisted CSV series (isolates the optimization step itself)
build_df = pd.read_csv(BUILD_NETRET_CSV, parse_dates=["date"]).set_index("date")
r13_build = build_df["net_ret_13bp"].to_numpy()
L_star_build_series = kelly_L_star_bisection(r13_build)

# (b) on MY OWN hand-derived net_ret series from Check 1 (fully independent end-to-end)
L_star_my_series = kelly_L_star_bisection(net_ret)

check4 = {
    "reported_full_kelly_L": 4.651257067218902,
    "my_kelly_L_on_build_csv_series": L_star_build_series,
    "my_kelly_L_on_own_handderived_series": L_star_my_series,
    "diff_vs_reported_on_build_series": L_star_build_series - 4.651257067218902,
    "method": "bisection on root of mean[r/(1+L*r)]=0 (derivative of E[log(1+Lr)]); "
              "reported figure used scipy bounded scalar minimize -- different numerical method, same objective.",
}
print(json.dumps(check4, indent=2), flush=True)

# ==========================================================================================
# CHECK 5: drawdown-brake spot check -- own re-implementation, pull actual dates
# ==========================================================================================
print("\n--- CHECK 5: drawdown-brake spot check (own implementation, on build CSV series) ---", flush=True)
L_half_reported = 2.325629  # = L*/2 per reported half-Kelly target used for the brake


def run_brake_vectorized(r: np.ndarray, L_target: float, ceiling: float = 0.35):
    """Independent re-implementation using cumulative running-peak arithmetic done slightly
    differently (compute day-by-day since the brake is inherently path-dependent/causal -- an
    O(n) loop is unavoidable for a correct causal implementation, but written fresh here)."""
    n = len(r)
    wealth = 1.0
    peak = 1.0
    mult = np.empty(n)
    exposure = np.empty(n)
    net = np.empty(n)
    for t in range(n):
        dd_prev = 0.0 if peak <= 0 else max(0.0, 1.0 - wealth / peak)
        m_t = min(1.0, max(0.0, 1.0 - dd_prev / ceiling))
        mult[t] = m_t
        L_t = L_target * m_t
        exposure[t] = L_t
        rt = r[t] if np.isfinite(r[t]) else 0.0
        net[t] = L_t * rt
        wealth = wealth * (1.0 + net[t])
        peak = max(peak, wealth)
    return mult, exposure, net


mult, exposure, net_brake = run_brake_vectorized(r13_build, L_half_reported, 0.35)
dates_build = build_df.index

covid_mask = (dates_build >= "2020-02-15") & (dates_build <= "2020-05-31")
y2022_mask = (dates_build >= "2022-01-01") & (dates_build <= "2022-12-31")

covid_engaged_days = int((exposure[covid_mask] < L_half_reported - 1e-9).sum())
covid_n = int(covid_mask.sum())
y2022_engaged_days = int((exposure[y2022_mask] < L_half_reported - 1e-9).sum())
y2022_n = int(y2022_mask.sum())

# pull a handful of actual dated multiplier values in each window as literal spot-check evidence
covid_sample = [{"date": str(d.date()), "multiplier": float(m), "exposure": float(e)}
                 for d, m, e in zip(dates_build[covid_mask], mult[covid_mask], exposure[covid_mask])][:15]
y2022_sample = [{"date": str(d.date()), "multiplier": float(m), "exposure": float(e)}
                 for d, m, e in zip(dates_build[y2022_mask], mult[y2022_mask], exposure[y2022_mask])
                 if d.day in (1, 15) or d.month in (1, 3)][:20]

overall_engaged = int((exposure < L_half_reported - 1e-9).sum())

check5 = {
    "L_target_half_kelly": L_half_reported,
    "overall_engagement": {"n_days_engaged": overall_engaged, "n_total": len(r13_build),
                            "pct": overall_engaged / len(r13_build)},
    "reported_overall_engagement": {"n_days_engaged": 1656, "n_total": 2046, "pct": 0.8094},
    "covid_2020_02_15_to_2020_05_31": {
        "n_days_in_window": covid_n, "n_days_engaged": covid_engaged_days,
        "reported_n_days_engaged": 71, "reported_n_total": 72,
        "sample_dates": covid_sample,
        "mean_exposure_in_window": float(exposure[covid_mask].mean()) if covid_n else None,
        "reported_mean_exposure": 0.609,
    },
    "y2022_full_year": {
        "n_days_in_window": y2022_n, "n_days_engaged": y2022_engaged_days,
        "reported_n_days_engaged": 251, "reported_n_total": 251,
        "sample_dates": y2022_sample,
        "mean_exposure_in_window": float(exposure[y2022_mask].mean()) if y2022_n else None,
        "reported_mean_exposure": 0.552,
    },
    "brake_metrics": {
        "cagr": my_cagr_maxdd(net_brake, dates_build)[0],
        "maxdd": my_cagr_maxdd(net_brake, dates_build)[1],
        "reported_brake_cagr": 0.21101489056798628, "reported_brake_maxdd": 0.3337938081225025,
    },
    "realized_avg_leverage": float(exposure.mean()),
    "reported_realized_avg_leverage": 1.378781807290258,
}
print(json.dumps(check5, indent=2), flush=True)

# ==========================================================================================
# Write everything out
# ==========================================================================================
results = {
    "check1_hand_derived_1x": check1,
    "check2_injection_test": check2,
    "check3_aapl_fix": check3,
    "check4_kelly": check4,
    "check5_brake": check5,
}


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


with open(OUT_JSON, "w") as f:
    json.dump(_sanitize(results), f, indent=2, default=str)
print(f"\n[output] wrote {OUT_JSON}", flush=True)
print(f"=== done in {time.time()-t_start:.1f}s ===", flush=True)
