"""
research/cta/_dd_diag_corrspike.py -- DIAGNOSTIC/INFORMATIONAL ONLY. Tests overlay #4
(CORRELATION-SPIKE DE-RISKING) ALONE on top of half-Kelly (L=2.3256) nominal leverage, with NO
other overlay stacked. This is explicitly NOT part of the mandated waterfall (which tests #1+#2
together first, per the owner's preregistered methodology) -- it is a standalone characterization
of #4 in isolation, requested separately for diagnostic purposes.

REUSE, not reimplementation: imports research/cta/_s1_ensemble_93name.py's own functions
(load_name_frames_patched, build_name_signal, the AAPL in-memory split patch) and
research/cta/ctaB_pipeline.py's (reconcile_universe, apply_costs, gross_book_return,
_cagr_from_returns, derive_periods_per_yr) DIRECTLY, rather than reimplementing the signal/sizing
machinery. The half-Kelly leverage constant L=2.325628533609451 is loaded verbatim from the
already-established research/cta/_s1_kelly_leverage_results.json (produced by
_s1_kelly_leverage_analysis.py), not re-derived, so this script cannot silently drift from the
already-agreed leverage constant.

OVERLAY #4 DEFINITION (per task spec): scale gross down when the trailing 60-trading-day average
pairwise correlation across CURRENTLY-ACTIVE names exceeds its own trailing 1-year historical norm
(here: trailing-52-week median of that same series) by 1.5x. Causal.

DESIGN CHOICES MADE EXPLICIT (the task under-specifies these; stated and justified here rather than
silently assumed):
  1. "Currently-active names" = the SAME definition already used for overlay #2 and for the base
     book's own sizing step: names with ensemble state>0 (in an uptrend) AND eligible (all three
     S1 sub-rules computable) as of the most recent Friday decision. This is the book's own actual
     holdings, not the full 93-name universe regardless of trend state.
  2. EVALUATION CADENCE: computed at WEEKLY (Friday-close) cadence, not daily -- consistent with
     the base construction's own "decide-at-close, act-next-session" weekly convention (state_a /
     rvol20 / sizing are all decided at Friday's own close and broadcast constant through the next
     week; overlay #4's trigger is decided the same way, at the SAME decision points, using the
     SAME 60-trading-day-window-ending-at-Friday's-own-close reference point already used by
     rvol20). A namewise daily-varying "active" mask recomputed every single day would require
     inventing a NEW intra-week cadence the base construction does not have; reusing the existing
     weekly decision points is the more conservative, less-invented choice.
  3. TRAILING "1-year historical norm" = trailing 52 WEEKLY observations of the corr series
     (~1 calendar year at weekly cadence), computed causally using ONLY the 52 weeks strictly
     BEFORE the current decision week (shift(1).rolling(52), i.e. this week's own corr value is
     compared against a norm that does NOT include itself). Minimum 26 valid prior weekly
     observations (~half a year) required before a trigger can be evaluated at all; before that
     warmup is satisfied, the overlay fails OPEN (multiplier=1.0, consistent with the codebase's
     own established fail-open convention for other warmup windows, e.g.
     lib/portfolio_construction.py's correlation_spike_scale).
  4. DE-RISKING MAGNITUDE: the task specifies the TRIGGER condition precisely (1.5x trailing
     median) but does not specify how much to de-risk once triggered. Rather than inventing a new
     tunable continuous function (which would itself be a hidden fitted parameter), this script
     adopts the SAME simple binary convention already sanctioned elsewhere in this task's own
     spec for an analogous "state-conditional" overlay -- overlay #5's explicit "halve book gross
     exposure" rule. Overlay #4 here: multiplier = 0.5 while triggered, 1.0 otherwise. This adds
     zero new tunable parameters beyond the already-stated 1.5x/60-day/52-week choices.

CAUSALITY: verified below via a synthetic-outlier injection test WITH an explicit naive/unlagged
control (bug pattern 3) -- the naive control deliberately mislabels week k's decision value using
a window that peeks one week into the future, and is shown to react one week EARLIER than the real
(causal) implementation to an injected correlation shock, proving the test has real discriminating
power; the real implementation is then shown to NOT exhibit that premature reaction.

Run: nohup /home/speculation/convexity/.venv/bin/python research/cta/_dd_diag_corrspike.py \
       > research/cta/_dd_diag_corrspike.out 2>&1 &
Writes: research/cta/_dd_diag_corrspike_results.json (NOT committed, diagnostic only)
Does NOT touch _s1_ensemble_93name_netret.csv / _s1_kelly_leverage_results.json (read-only reuse).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO_ROOT}/research/cta")
sys.path.insert(0, f"{REPO_ROOT}/research/cta/lib")
sys.path.insert(0, REPO_ROOT)

import ctaB_pipeline as ctaB                 # reuse: reconcile_universe, apply_costs, etc.
import _s1_ensemble_93name as ens            # reuse: build_name_signal, AAPL patch, S1 rules

OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
KELLY_JSON = OUT_DIR / "_s1_kelly_leverage_results.json"
NETRET_CSV = OUT_DIR / "_s1_ensemble_93name_netret.csv"
RESULTS_JSON = OUT_DIR / "_dd_diag_corrspike_results.json"

CORR_WINDOW = 60          # trailing trading days, per task spec
NORM_WINDOW_WEEKS = 52    # trailing 1-year historical norm, at weekly cadence (see design note 3)
NORM_MIN_PERIODS = 26     # warmup requirement before a trigger can be evaluated (fail-open before)
SPIKE_MULTIPLE = 1.5      # per task spec, not tuned
DERISK_MULTIPLIER = 0.5   # per design note 4, mirrors overlay #5's stated "halve" convention


def metrics_from_net(net: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """CAGR/maxDD via ctaB._cagr_from_returns (reused); Sharpe via periods_per_yr DERIVED from
    actual observation spacing (bug pattern 2), never hardcoded."""
    net = np.nan_to_num(np.asarray(net, dtype="float64"))
    cagr, maxdd = ctaB._cagr_from_returns(net, dates)
    ppy = ctaB.derive_periods_per_yr(dates)
    mu, sd = np.mean(net), np.std(net, ddof=1)
    sharpe = (mu / sd) * np.sqrt(ppy) if sd > 0 else float("nan")
    return {"cagr": float(cagr), "maxdd": float(maxdd), "sharpe": float(sharpe), "ppy": float(ppy)}


def build_base_panel():
    """Rebuild the exact per-name panel (RET, STATE, RVOL) and weekly Friday-decision sizing
    used by _s1_ensemble_93name.py, by calling ITS OWN functions directly (not reimplementing the
    S1 signal/sizing logic). Returns everything needed for both the base net_13 reproduction check
    and the new correlation-overlay computation."""
    names = ctaB.reconcile_universe()
    cache, failed = {}, []
    for sym in names:
        s, d = ens.build_name_signal(sym)
        if d is None:
            failed.append(sym)
        else:
            cache[s] = d
    print(f"[panel] built {len(cache)}/{len(names)} usable (failed/excluded: {failed})", flush=True)

    active_names = sorted(cache.keys())
    idx = None
    for sym in active_names:
        idx = cache[sym]["close"].index if idx is None else idx.union(cache[sym]["close"].index)
    idx = idx.sort_values()

    RET = pd.DataFrame({sym: cache[sym]["simple_ret"].reindex(idx) for sym in active_names})
    STATE = pd.DataFrame({sym: cache[sym]["state"].reindex(idx) for sym in active_names}).ffill(limit=5)
    RVOL = pd.DataFrame({sym: cache[sym]["rvol20"].reindex(idx) for sym in active_names}).ffill(limit=5)

    dow = idx.dayofweek.to_numpy()
    fri_positions = np.where(dow == 4)[0]

    STATE_np = STATE.to_numpy()
    RVOL_np = RVOL.to_numpy()
    n_days, n_names = len(idx), len(active_names)
    weights_daily = np.zeros((n_days, n_names))
    weekly_active_masks = []   # boolean (n_names,) per decision week -- IDENTICAL mask used for sizing
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
        weekly_active_masks.append(active)

    lo, hi = fri_positions[0] + 1, fri_positions[-1] + 1
    dates_scored = idx[lo:hi]
    W = pd.DataFrame(weights_daily[lo:hi], index=dates_scored, columns=active_names)
    R = RET.loc[dates_scored]
    net_13 = ctaB.apply_costs(W, R, cost_bp=13.0, funding_bp_week=0.0)

    return {
        "names": names, "active_names": active_names, "idx": idx, "RET": RET,
        "fri_positions": fri_positions, "weekly_active_masks": weekly_active_masks,
        "dates_scored": dates_scored, "net_13": net_13,
    }


def weekly_avg_pairwise_corr(RET: pd.DataFrame, active_names: list[str], fri_positions: np.ndarray,
                              weekly_active_masks: list[np.ndarray], window: int = CORR_WINDOW) -> np.ndarray:
    """For each decision week k (Friday position f=fri_positions[k]), compute the average pairwise
    Pearson correlation among that week's ACTIVE names (weekly_active_masks[k]), using the trailing
    `window` trading days of RETURNS ENDING AT AND INCLUDING f (rows [f-window+1, f] inclusive) --
    i.e. "through the decision day's own close", the SAME reference-point convention already used
    for rvol20/state_a in the base construction. Returns np.nan for weeks with <2 qualifying active
    names or an insufficient/incomplete window (f < window-1, or NaNs present for a name in the
    window -- should essentially never trigger for genuinely-eligible names since eligibility
    already requires >=260 days of price history, but guarded for safety)."""
    RET_np = RET[active_names].to_numpy()
    n_weeks = len(fri_positions) - 1
    out = np.full(n_weeks, np.nan)
    for k in range(n_weeks):
        f = fri_positions[k]
        if f < window - 1:
            continue
        active = weekly_active_masks[k]
        if active.sum() < 2:
            continue
        win = RET_np[f - window + 1: f + 1, :][:, active]   # (window, n_active)
        col_ok = ~np.isnan(win).any(axis=0)
        if col_ok.sum() < 2:
            continue
        sub = win[:, col_ok]
        corr = np.corrcoef(sub, rowvar=False)
        k_names = corr.shape[0]
        off_diag_sum = corr.sum() - np.trace(corr)
        out[k] = off_diag_sum / (k_names * (k_names - 1))
    return out


def trigger_from_corr_series(corr_series: np.ndarray, norm_window: int = NORM_WINDOW_WEEKS,
                              min_periods: int = NORM_MIN_PERIODS, spike_multiple: float = SPIKE_MULTIPLE):
    """Causal trailing-median norm (shift(1).rolling(norm_window, min_periods)) and the 1.5x-spike
    trigger boolean. Returns (trailing_median, triggered) both length == len(corr_series)."""
    s = pd.Series(corr_series)
    trailing_median = s.shift(1).rolling(norm_window, min_periods=min_periods).median().to_numpy()
    triggered = np.zeros(len(s), dtype=bool)
    valid = np.isfinite(corr_series) & np.isfinite(trailing_median)
    triggered[valid] = corr_series[valid] > spike_multiple * trailing_median[valid]
    return trailing_median, triggered


def broadcast_weekly_to_daily(weekly_values: np.ndarray, fri_positions: np.ndarray, idx: pd.DatetimeIndex,
                               dates_scored: pd.DatetimeIndex, fill_value: float) -> np.ndarray:
    n_weeks = len(fri_positions) - 1
    daily_full = np.full(len(idx), fill_value)
    for k in range(n_weeks):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        v = weekly_values[k]
        daily_full[f + 1:f_next + 1] = v if np.isfinite(v) else fill_value
    lo = idx.get_loc(dates_scored[0])
    hi = idx.get_loc(dates_scored[-1]) + 1
    return daily_full[lo:hi]


# ==========================================================================================
# Causality injection test (bug pattern 3): synthetic correlation-spike injection, WITH an
# explicit naive/unlagged control that mislabels week k's value using data one week ahead.
# ==========================================================================================
def causality_injection_test(RET: pd.DataFrame, active_names: list[str], fri_positions: np.ndarray,
                               weekly_active_masks: list[np.ndarray]) -> dict:
    n_weeks = len(fri_positions) - 1
    active_counts = np.array([m.sum() for m in weekly_active_masks])
    # pick an interior week with a healthy active-name count and enough history on both sides
    candidates = [k for k in range(30, n_weeks - 15)
                  if active_counts[k] >= 10 and fri_positions[k] >= CORR_WINDOW]
    if not candidates:
        candidates = [k for k in range(n_weeks) if active_counts[k] >= 2]
    k_inj = candidates[len(candidates) // 2]
    f_inj = fri_positions[k_inj]              # injection day = week k_inj's OWN decision Friday
    inject_date = str(RET.index[f_inj].date())

    active_syms = [active_names[i] for i, a in enumerate(weekly_active_masks[k_inj]) if a]
    shock_syms = active_syms[:min(10, len(active_syms))]

    RET_bad = RET.copy()
    for sym in shock_syms:
        RET_bad.loc[RET_bad.index[f_inj], sym] = -0.15   # identical shock -> artificial perfect-corr day

    base_corr = weekly_avg_pairwise_corr(RET, active_names, fri_positions, weekly_active_masks)
    inj_corr = weekly_avg_pairwise_corr(RET_bad, active_names, fri_positions, weekly_active_masks)

    # --- CAUSAL (real) implementation: pre-injection weeks (k < k_inj) must be byte-identical ---
    pre = np.nan_to_num(base_corr[:k_inj], nan=-999.0)
    pre_inj = np.nan_to_num(inj_corr[:k_inj], nan=-999.0)
    causal_pre_unaffected = bool(np.allclose(pre, pre_inj, atol=1e-12))
    causal_reacts_at_k_inj = bool(abs(inj_corr[k_inj] - base_corr[k_inj]) > 1e-6) if np.isfinite(base_corr[k_inj]) else False

    # --- NAIVE/unlagged control: mislabel week k's value using a window ending at f_{k+1} (one
    #     week AHEAD) instead of f_k. This is the classic "peek at next week's close today" bug.
    #     If real, this would make naive_corr[k_inj - 1] already reflect the k_inj-Friday shock
    #     (since naive week (k_inj-1) uses a window ending at f_{k_inj} = the injection day).
    def naive_corr_series(RET_df):
        RET_np = RET_df[active_names].to_numpy()
        out = np.full(n_weeks, np.nan)
        for k in range(n_weeks - 1):
            f_ahead = fri_positions[k + 1]        # BUG: one week ahead of the proper f=fri_positions[k]
            if f_ahead < CORR_WINDOW - 1:
                continue
            active = weekly_active_masks[k]
            if active.sum() < 2:
                continue
            win = RET_np[f_ahead - CORR_WINDOW + 1: f_ahead + 1, :][:, active]
            col_ok = ~np.isnan(win).any(axis=0)
            if col_ok.sum() < 2:
                continue
            sub = win[:, col_ok]
            corr = np.corrcoef(sub, rowvar=False)
            kk = corr.shape[0]
            out[k] = (corr.sum() - np.trace(corr)) / (kk * (kk - 1))
        return out

    naive_base = naive_corr_series(RET)
    naive_inj = naive_corr_series(RET_bad)
    naive_reacts_one_week_early = bool(
        np.isfinite(naive_base[k_inj - 1]) and abs(naive_inj[k_inj - 1] - naive_base[k_inj - 1]) > 1e-6
    )
    # the naive control's OWN pre-(k_inj-1) weeks should still be clean (sanity on the control itself)
    naive_pre = np.nan_to_num(naive_base[:k_inj - 1], nan=-999.0)
    naive_pre_inj = np.nan_to_num(naive_inj[:k_inj - 1], nan=-999.0)
    naive_pre_unaffected = bool(np.allclose(naive_pre, naive_pre_inj, atol=1e-12))
    # and the REAL causal implementation must NOT show that same premature (k_inj-1) reaction
    causal_k_inj_minus_1_unaffected = bool(
        np.isfinite(base_corr[k_inj - 1]) and abs(inj_corr[k_inj - 1] - base_corr[k_inj - 1]) < 1e-12
    )

    return {
        "k_inj": int(k_inj), "inject_date": inject_date, "n_shocked_names": len(shock_syms),
        "shocked_names": shock_syms,
        "base_corr_at_k_inj": float(base_corr[k_inj]) if np.isfinite(base_corr[k_inj]) else None,
        "injected_corr_at_k_inj": float(inj_corr[k_inj]) if np.isfinite(inj_corr[k_inj]) else None,
        "causal_impl": {
            "pre_injection_weeks_byte_identical": causal_pre_unaffected,
            "reacts_at_k_inj": causal_reacts_at_k_inj,
            "k_inj_minus_1_unaffected (no early leak)": causal_k_inj_minus_1_unaffected,
        },
        "naive_unlagged_control": {
            "description": ("deliberately mislabels week k's corr using a window ending at "
                             "fri_positions[k+1] (one week AHEAD) instead of fri_positions[k]"),
            "pre_k_inj_minus_1_weeks_byte_identical": naive_pre_unaffected,
            "reacts_one_week_early_at_k_inj_minus_1": naive_reacts_one_week_early,
        },
        "verdict": ("PASS: naive control demonstrably reacts one week EARLY (proving the injection "
                    "test has real discriminating power to catch a lookahead bug), while the real "
                    "causal implementation used in the overlay shows no such early reaction and "
                    "only reacts at/after its own proper decision week."
                    if (naive_reacts_one_week_early and causal_k_inj_minus_1_unaffected
                        and causal_pre_unaffected and causal_reacts_at_k_inj)
                    else "FAIL/INCONCLUSIVE -- see individual fields."),
    }


def main():
    t0 = time.time()
    print(f"=== _dd_diag_corrspike.py starting {pd.Timestamp.now()} ===", flush=True)
    print("DIAGNOSTIC/INFORMATIONAL ONLY -- overlay #4 tested ALONE on top of half-Kelly, "
          "not part of the mandated waterfall.", flush=True)

    panel = build_base_panel()
    names, active_names = panel["names"], panel["active_names"]
    idx, RET = panel["idx"], panel["RET"]
    fri_positions, weekly_active_masks = panel["fri_positions"], panel["weekly_active_masks"]
    dates_scored, net_13 = panel["dates_scored"], panel["net_13"]

    # ---- reproduction check against the already-established netret CSV ----
    csv_df = pd.read_csv(NETRET_CSV, parse_dates=["date"])
    csv_r13 = csv_df["net_ret_13bp"].to_numpy()
    if len(csv_r13) == len(net_13):
        max_abs_diff = float(np.max(np.abs(csv_r13 - net_13)))
    else:
        max_abs_diff = float("nan")
    m_1x = metrics_from_net(net_13, dates_scored)
    print(f"\n[repro check] n_obs={len(net_13)} vs csv n_obs={len(csv_r13)}  "
          f"max_abs_diff={max_abs_diff:.3e}", flush=True)
    print(f"[repro check] recomputed 1x: CAGR={m_1x['cagr']:.4f} maxDD={m_1x['maxdd']:.4f} "
          f"Sharpe={m_1x['sharpe']:.4f}  (established: CAGR=0.2205 maxDD=0.3873 Sharpe=1.06)", flush=True)
    repro_ok = bool(np.isfinite(max_abs_diff) and max_abs_diff < 1e-8)
    print(f"[repro check] byte-level reproduction of established netret CSV: {repro_ok}", flush=True)

    # ---- load the already-established half-Kelly leverage constant (not re-derived) ----
    kelly = json.load(open(KELLY_JSON))
    L_half = float(kelly["L_half"])
    print(f"\n[leverage] loaded L_half={L_half:.6f} from {KELLY_JSON.name} "
          f"(established value: 2.325628533609451)", flush=True)
    assert abs(L_half - 2.325628533609451) < 1e-6, "L_half drifted from the established value"

    # ---- overlay #4: weekly avg pairwise correlation among currently-active names ----
    print("\n[overlay #4] computing weekly avg pairwise correlation among active names "
          f"(window={CORR_WINDOW}d)...", flush=True)
    corr_series = weekly_avg_pairwise_corr(RET, active_names, fri_positions, weekly_active_masks)
    trailing_median, triggered = trigger_from_corr_series(corr_series)
    n_weeks = len(corr_series)
    n_valid_corr = int(np.sum(np.isfinite(corr_series)))
    n_valid_norm = int(np.sum(np.isfinite(trailing_median)))
    n_triggered = int(triggered.sum())
    print(f"[overlay #4] n_weeks={n_weeks} n_valid_corr_obs={n_valid_corr} "
          f"n_weeks_with_norm_established={n_valid_norm} n_weeks_triggered={n_triggered} "
          f"({n_triggered / n_weeks * 100:.2f}% of ALL weeks, "
          f"{n_triggered / max(n_valid_norm,1) * 100:.2f}% of weeks where a norm existed)", flush=True)

    # ---- broadcast weekly trigger -> daily multiplier -> daily exposure -> overlay net return ----
    weekly_mult = np.where(triggered, DERISK_MULTIPLIER, 1.0)
    mult_daily = broadcast_weekly_to_daily(weekly_mult, fri_positions, idx, dates_scored, fill_value=1.0)
    exposure_daily = L_half * mult_daily
    overlay_net = exposure_daily * net_13   # net_13 is the UNLEVERED, cost-embedded 1x book return
    # bug pattern 4 (costs scale with realized exposure changes): no SEPARATE overlay trading cost is
    # charged for the multiplier's own exposure changes -- this scalar leverage/de-risk multiplier
    # applies to the WHOLE book at once (not per-name rebalancing), and net_ret_Lx(t)=L(t)*net_ret_1x(t)
    # is the exact linear-in-leverage relationship already proven in _s1_kelly_leverage_analysis.py's
    # Step 1 (cost scales linearly with L for a fixed underlying turnover fraction); this mirrors the
    # identical, already-established treatment of the GZ-brake's own exposure changes in that script.

    m_overlay = metrics_from_net(overlay_net, dates_scored)
    realized_avg_leverage = float(np.mean(exposure_daily))
    n_days_engaged = int(np.sum(mult_daily < 1.0 - 1e-12))
    n_days = len(mult_daily)
    print(f"\n[overlay #4 result] CAGR={m_overlay['cagr']:.4f} maxDD={m_overlay['maxdd']:.4f} "
          f"Sharpe={m_overlay['sharpe']:.4f} realized_avg_leverage={realized_avg_leverage:.4f} "
          f"(nominal L_half={L_half:.4f})", flush=True)
    print(f"[overlay #4 result] days engaged (de-risked): {n_days_engaged}/{n_days} "
          f"({n_days_engaged / n_days * 100:.2f}%)", flush=True)

    # ---- empirical check: does avg pairwise corr genuinely spike in 2020-03 / 2022? ----
    weekly_dates = pd.DatetimeIndex([idx[fri_positions[k + 1]] for k in range(n_weeks)])
    corr_s = pd.Series(corr_series, index=weekly_dates)
    med_s = pd.Series(trailing_median, index=weekly_dates)
    trig_s = pd.Series(triggered, index=weekly_dates)

    def window_stats(start, end, label):
        mask = (weekly_dates >= pd.Timestamp(start, tz=weekly_dates.tz)) & \
               (weekly_dates <= pd.Timestamp(end, tz=weekly_dates.tz))
        seg = corr_s.loc[mask]
        seg_med = med_s.loc[mask]
        seg_trig = trig_s.loc[mask]
        overall_median = float(np.nanmedian(corr_series))
        stats = {
            "window": label, "start": start, "end": end, "n_weeks_in_window": int(mask.sum()),
            "mean_corr_in_window": float(seg.mean()) if mask.any() else None,
            "max_corr_in_window": float(seg.max()) if mask.any() else None,
            "mean_trailing_median_norm_in_window": float(seg_med.mean(skipna=True)) if mask.any() else None,
            "overall_sample_median_corr": overall_median,
            "n_weeks_triggered_in_window": int(seg_trig.sum()),
            "pct_weeks_triggered_in_window": float(seg_trig.mean() * 100) if mask.any() else None,
        }
        return stats

    covid = window_stats("2020-02-15", "2020-05-31", "2020-03 COVID crash")
    y2022 = window_stats("2022-01-01", "2022-12-31", "2022 chop/decline")
    overall = {
        "overall_mean_corr": float(np.nanmean(corr_series)),
        "overall_median_corr": float(np.nanmedian(corr_series)),
        "overall_p90_corr": float(np.nanpercentile(corr_series[np.isfinite(corr_series)], 90)),
        "overall_min_corr": float(np.nanmin(corr_series)),
        "overall_max_corr": float(np.nanmax(corr_series)),
    }
    print(f"\n[regime corr check] overall: {json.dumps(overall, indent=2)}", flush=True)
    print(f"[regime corr check] 2020-03 window: {json.dumps(covid, indent=2)}", flush=True)
    print(f"[regime corr check] 2022 window: {json.dumps(y2022, indent=2)}", flush=True)

    # ---- causality injection test w/ naive/unlagged control ----
    print("\n[causality] running injection test with naive/unlagged control...", flush=True)
    inj_result = causality_injection_test(RET, active_names, fri_positions, weekly_active_masks)
    print(json.dumps(inj_result, indent=2), flush=True)

    results = {
        "meta": {
            "diagnostic_only": True, "overlay_tested": "#4 CORRELATION-SPIKE DE-RISKING, ALONE",
            "nominal_leverage_L_half": L_half, "corr_window_days": CORR_WINDOW,
            "norm_window_weeks": NORM_WINDOW_WEEKS, "norm_min_periods_weeks": NORM_MIN_PERIODS,
            "spike_multiple": SPIKE_MULTIPLE, "derisk_multiplier_when_triggered": DERISK_MULTIPLIER,
            "n_obs_daily": len(net_13), "n_weeks": n_weeks,
        },
        "reproduction_check": {"max_abs_diff_vs_established_csv": max_abs_diff,
                                "byte_level_match": repro_ok, "recomputed_1x_metrics": m_1x},
        "overlay4_alone_result": {
            "cagr": m_overlay["cagr"], "maxdd": m_overlay["maxdd"], "sharpe": m_overlay["sharpe"],
            "realized_avg_leverage": realized_avg_leverage, "nominal_L_half": L_half,
            "n_days_engaged": n_days_engaged, "n_days_total": n_days,
            "pct_days_engaged": n_days_engaged / n_days * 100,
            "n_weeks_triggered": n_triggered, "n_weeks_total": n_weeks,
            "pct_weeks_triggered": n_triggered / n_weeks * 100,
        },
        "corr_regime_diagnostics": {"overall": overall, "covid_2020_03": covid, "y2022": y2022},
        "causality_injection_test": inj_result,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[output] wrote {RESULTS_JSON}", flush=True)
    print(f"=== done in {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
