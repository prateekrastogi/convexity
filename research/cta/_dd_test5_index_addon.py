"""
research/cta/_dd_test5_index_addon.py -- ONE-SHOT, PRE-DECLARED TEST: does layering overlay #5
(index-level trend gate) as one more multiplicative layer on top of the EXISTING, already-adopted
#1+#2 drawdown-overlay policy (research/cta/_dd_combined_waterfall_netret.csv) improve Sharpe
above that policy's own 1.1676 while keeping maxDD at/below its own 30.023% -- i.e. a genuine
additional improvement, not a regression toward the original looser 50% brake-ceiling budget.

This is NOT a new waterfall step and NOT a search: the combination method, the pass/fail bar, and
the overlay's own construction were all fixed BEFORE this script was run. Report the honest
result, whichever it is.

REUSED, NOT REIMPLEMENTED:
  - research/cta/_dd_combined_waterfall_netret.csv -- the ALREADY-BUILT, ALREADY-ADOPTED #1+#2
    policy's own daily net return (net_ret_final_policy), effective leverage
    (effective_leverage), and the inherited 1x base return (r13_base). Not rebuilt from the
    per-name panel; this task multiplies a NEW scalar overlay onto this existing series exactly
    as _dd_combined_waterfall.py's own STEP A multiplied the GZ-brake's m_t onto the pre-brake
    #2-alone book, and exactly as this same repo's _dd_diag_indexgate.py multiplied overlay #5's
    gate onto plain half-Kelly.
  - research/cta/_s1_ensemble_93name.py's compute_state_a() -- Rule 1 (200d-SMA +-2% hysteresis
    band), verbatim, applied to SPY's own close, IDENTICAL construction to the already-tested
    standalone overlay-#5 diagnostic (_dd_diag_indexgate.py) and to the index-gate branch already
    coded (but never reached, because Step A stopped first) inside _dd_combined_waterfall.py's
    own STEP D.
  - research/cta/ctaB_pipeline.py: load_name_frames() (SPY OHLC, source=='duka' filtered, bug
    pattern 1), derive_periods_per_yr(), _cagr_from_returns(), REGIME_WINDOWS, cpcv_paths(),
    per_obs_sharpe(), dsr(), CPCV_N_GROUPS/CPCV_K/EMBARGO_TRADING_DAYS (all imported as `ctaB`,
    none reimplemented).

COMBINATION METHOD (pre-declared, stated here verbatim before any number below was computed):
  Let net_12(t) = the existing #1+#2 policy's own daily net return (net_ret_final_policy column).
  Let eff_lev_12(t) = that same policy's own daily realized effective leverage
      (effective_leverage column, = m1(t) * L_half * breadth_frac(t) already).
  Build overlay #5's gate multiplier m5(t) EXACTLY as the standalone diagnostic did: SPY's own
  close vs its own 200-day SMA +-2% hysteresis band (compute_state_a, decide-at-close(t), causal
  .shift(1), applied starting session t+1), reindexed onto the #1+#2 policy's own calendar via
  ffill ONLY (never bfill), leading warmup days (no SPY regime evidence yet) DEFAULTED to
  gate-OFF (m5=1.0, full exposure) -- identical convention to _dd_diag_indexgate.py. m5(t) = 0.5
  when the gate is engaged (SPY state == 0, i.e. below its own lower band or holding a prior
  below-band state), else 1.0.
  Apply m5(t) via notional/margin scaling (NO incremental trading cost for the overlay's own
  multiplier changes beyond the explicit flip-cost line item below) -- this is the SAME
  "no incremental cost for the overlay's own day-to-day multiplier changes, applied via
  notional/margin scaling on top of an already fully-costed book" precedent already established
  and used for overlay #1 (the GZ-brake's own m_t, in _dd_combined_waterfall.py's STEP A) and for
  overlay #3 (vol-targeting's own scale(t), in that same script's STEP C). Stated explicitly here,
  not silently reinvented:
      net_125_pregate_cost(t) = m5(t) * net_12(t)
      eff_lev_125(t)          = m5(t) * eff_lev_12(t)
  COST (bug pattern 4), per the task's explicit instruction: overlay #5 introduces one genuinely
  NEW, previously-uncosted source of turnover -- whenever m5 flips (1.0 <-> 0.5), the CURRENT
  notional of the #1+#2 book (eff_lev_12(t), which is itself time-varying, unlike the standalone
  diagnostic's constant L_half book) must be traded up/down by that fraction. This is the SAME
  generalization move _dd_combined_waterfall.py's own STEP A already made explicit in its own
  docstring when it generalized the standalone GZ-brake's run_brake() (built for a constant-L
  book) into run_brake_on_prescaled() (built to sit on top of an already time-varying-exposure
  book) -- reusing the identical 13bp round-trip rate, no new parameter:
      extra_cost(t) = 13bp * |m5(t) - m5(t-1)| * eff_lev_12(t)      [m5(-1) := 1.0, "enters ungated"]
      net_125(t) = m5(t) * net_12(t) - extra_cost(t)
  charged on the day the new multiplier first takes effect, as its OWN visible line item (never
  silently folded into net_12).

PASS CONDITION (decided BEFORE running): Sharpe(net_125) > 1.1676 AND maxDD(net_125) <= 0.30023
  (rounded tolerance <= 0.3003 for floating point). FAIL if either metric misses. This script
  reports whichever actually happens -- it does not iterate to a different overlay,
  parameterization, or combination order if the test fails.

Read-only outside research/cta/; writes only this script's own results JSON + netret CSV. Nothing
committed. Scratch script, leading underscore.
"""
from __future__ import annotations

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
sys.path.insert(0, f"{REPO_ROOT}/src")

import ctaB_pipeline as ctaB           # noqa: E402  -- reuse verbatim, no reimplementation
import _s1_ensemble_93name as s1ens    # noqa: E402  -- reuse compute_state_a (Rule 1), verbatim

OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
WATERFALL_NETRET_CSV = OUT_DIR / "_dd_combined_waterfall_netret.csv"
RESULTS_JSON = OUT_DIR / "_dd_test5_index_addon_results.json"
NETRET_CSV = OUT_DIR / "_dd_test5_index_addon_netret.csv"

COST_BP = 13.0  # already-established primary cost scenario, reused verbatim, no new parameter

# The existing #1+#2 policy's own previously-reported headline numbers (mandatory reproduction
# target, step 1 of the task).
EXISTING_12_REPORTED = {
    "cagr": 0.28264240237223026, "maxdd": 0.30023443778215164, "sharpe": 1.1675672380874091,
    "realized_avg_leverage": 1.1103454095396768, "n_obs": 2046,
}
# Pre-declared pass condition.
PASS_SHARPE_MIN = 1.1676
PASS_MAXDD_MAX = 0.3003  # 30.023% rounded up for float tolerance, per task spec


def metrics(net: np.ndarray, dates: pd.DatetimeIndex, ppy: float) -> dict:
    """Same convention as _dd_combined_waterfall.py's own local metrics(): cagr/maxdd via
    ctaB._cagr_from_returns (imported verbatim), Sharpe = mean/std(ddof=1) * sqrt(ppy)."""
    net = np.nan_to_num(np.asarray(net, dtype="float64"))
    cagr, maxdd = ctaB._cagr_from_returns(net, dates)
    mu, sd = np.mean(net), np.std(net, ddof=1)
    sharpe = (mu / sd) * np.sqrt(ppy) if sd > 0 else float("nan")
    return {"cagr": float(cagr), "maxdd": float(maxdd), "sharpe": float(sharpe),
            "worst_day": float(np.min(net)), "best_day": float(np.max(net))}


# ==========================================================================================
# The actual composition function under test -- factored out so it can be run BOTH on the real
# data and on synthetic injection-test data (task step 5: confirm no lookahead introduced by
# THIS specific combination step, not just by the already-separately-verified components).
# ==========================================================================================
def build_gate_m5(spy_close: pd.Series, dates_scored: pd.DatetimeIndex, lag: int = 1) -> np.ndarray:
    """SPY Rule-1 state -> causal gate multiplier, reindexed onto dates_scored via ffill only.
    lag=1 is the real (causal) construction; lag=0 is the naive/unlagged control used ONLY in the
    injection test below, never in the actual P&L."""
    state_native = s1ens.compute_state_a(spy_close)
    gate_native = state_native.shift(lag) if lag else state_native
    gate_book = gate_native.reindex(dates_scored).ffill()
    gate_book = gate_book.fillna(1.0)  # no regime evidence yet -> default gate OFF (m=1.0)
    return np.where(gate_book.to_numpy() == 0.0, 0.5, 1.0)


def combine_step5(net_12: np.ndarray, eff_lev_12: np.ndarray, m5: np.ndarray,
                   cost_bp: float = COST_BP) -> dict:
    """The pre-declared combination step: net_125(t) = m5(t)*net_12(t) - extra_cost(t),
    extra_cost(t) = cost_bp * |m5(t)-m5(t-1)| * eff_lev_12(t), m5(-1) := 1.0."""
    m5_prev = np.empty_like(m5)
    m5_prev[0] = 1.0
    m5_prev[1:] = m5[:-1]
    delta_m5 = np.abs(m5 - m5_prev)
    extra_cost = (cost_bp / 1e4) * delta_m5 * eff_lev_12
    net_125 = m5 * net_12 - extra_cost
    eff_lev_125 = m5 * eff_lev_12
    return {"net_125": net_125, "eff_lev_125": eff_lev_125, "extra_cost": extra_cost,
            "delta_m5": delta_m5, "n_flip_days": int(np.sum(delta_m5 > 0))}


def main():
    t_start = time.time()
    print(f"=== _dd_test5_index_addon.py starting {pd.Timestamp.now()} ===", flush=True)

    # ======================================================================================
    # TASK STEP 1: mandatory reproduction gate -- the existing #1+#2 policy's own numbers,
    # EXACTLY, before trusting anything downstream.
    # ======================================================================================
    print("\n--- STEP 1: reproduce existing #1+#2 policy headline from its own netret CSV ---",
          flush=True)
    df12 = pd.read_csv(WATERFALL_NETRET_CSV, parse_dates=["date"]).set_index("date")
    dates_scored = df12.index
    net_12 = df12["net_ret_final_policy"].to_numpy(dtype="float64")
    eff_lev_12 = df12["effective_leverage"].to_numpy(dtype="float64")
    r13_base = df12["r13_base"].to_numpy(dtype="float64")
    n = len(net_12)
    ppy = ctaB.derive_periods_per_yr(dates_scored)
    span_years = (dates_scored[-1] - dates_scored[0]).days / 365.25
    print(f"[load] {WATERFALL_NETRET_CSV.name}: n_obs={n} "
          f"{dates_scored[0].date()}..{dates_scored[-1].date()} span_years={span_years:.6f} "
          f"ppy={ppy:.6f} dates_tz={dates_scored.tz}", flush=True)

    repro_12 = metrics(net_12, dates_scored, ppy)
    repro_12["realized_avg_leverage"] = float(np.mean(eff_lev_12))
    print(f"[verify] recomputed #1+#2: {json.dumps(repro_12, indent=2)}", flush=True)
    print(f"[verify] previously reported #1+#2: {json.dumps(EXISTING_12_REPORTED, indent=2)}",
          flush=True)
    repro_ok = (
        n == EXISTING_12_REPORTED["n_obs"]
        and abs(repro_12["cagr"] - EXISTING_12_REPORTED["cagr"]) < 1e-6
        and abs(repro_12["maxdd"] - EXISTING_12_REPORTED["maxdd"]) < 1e-6
        and abs(repro_12["sharpe"] - EXISTING_12_REPORTED["sharpe"]) < 1e-6
        and abs(repro_12["realized_avg_leverage"] - EXISTING_12_REPORTED["realized_avg_leverage"]) < 1e-6
    )
    print(f"\n*** STEP 1 REPRODUCTION CHECK PASSED: {repro_ok} ***", flush=True)
    if not repro_ok:
        print("*** WARNING: reproduction mismatch -- stopping before trusting anything "
              "downstream ***", flush=True)
        sys.exit(1)

    # ======================================================================================
    # TASK STEP 2: build overlay #5's gate and the #1+#2+#5 combination.
    # ======================================================================================
    print("\n--- STEP 2: build overlay #5 (index trend gate) and combine on top of #1+#2 ---",
          flush=True)
    spy = ctaB.load_name_frames("SPYUSDT")
    spy_close = spy["close"]
    print(f"[load] SPYUSDT close: n_obs={spy['n_obs']} "
          f"{spy_close.index[0].date()}..{spy_close.index[-1].date()} tz={spy_close.index.tz}",
          flush=True)

    data_gap_days = (dates_scored[-1] - spy_close.index[-1]).days
    print(f"[data caveat] book's own scored range ends {dates_scored[-1].date()}; SPY's own duka "
          f"feed ends {spy_close.index[-1].date()} -- tail gap = {data_gap_days} calendar days "
          f"with no fresh SPY price, held via unlimited causal ffill of the last known gate "
          f"state (never bfill). Identical caveat to the standalone _dd_diag_indexgate.py test.",
          flush=True)

    m5 = build_gate_m5(spy_close, dates_scored, lag=1)
    n_leading_default = int(np.sum(
        s1ens.compute_state_a(spy_close).shift(1).reindex(dates_scored).isna()))
    print(f"[warmup] {n_leading_default} leading day(s) defaulted to gate-OFF (m5=1.0, no SPY "
          f"regime evidence yet)", flush=True)

    combo = combine_step5(net_12, eff_lev_12, m5, cost_bp=COST_BP)
    net_125 = combo["net_125"]
    eff_lev_125 = combo["eff_lev_125"]
    n_flip_days = combo["n_flip_days"]
    total_extra_cost_bp = float(combo["extra_cost"].sum() * 1e4)
    mean_extra_cost_per_flip_bp = float(
        combo["extra_cost"][combo["delta_m5"] > 0].mean() * 1e4) if n_flip_days else 0.0
    gate_active_frac = float(np.mean(m5 == 0.5))
    print(f"[cost] {n_flip_days} gate-flip day(s); total extra cost over full sample = "
          f"{total_extra_cost_bp:.4f}bp; mean extra cost per flip day = "
          f"{mean_extra_cost_per_flip_bp:.4f}bp", flush=True)
    print(f"[gate] active (m5=0.5) fraction of days = {gate_active_frac*100:.2f}% "
          f"({int(np.sum(m5==0.5))}/{n} days)", flush=True)

    m_125 = metrics(net_125, dates_scored, ppy)
    realized_avg_lev_125 = float(np.mean(eff_lev_125))
    print(f"\n[RESULT] #1+#2+#5 combination: {json.dumps(m_125, indent=2)}", flush=True)
    print(f"[RESULT] realized average leverage = {realized_avg_lev_125:.6f} "
          f"(#1+#2-only was {repro_12['realized_avg_leverage']:.6f})", flush=True)

    # weekly-compounded EV/week, on the scored calendar's own Friday boundaries (dates_scored is
    # already the scored/active calendar, so Fridays are located directly on it -- no offset
    # translation needed, unlike _dd_combined_waterfall.py which located Fridays on a larger
    # pre-scoring master calendar).
    dow = dates_scored.dayofweek.to_numpy()
    fri_pos = np.where(dow == 4)[0]
    weekly_rets = []
    for k in range(len(fri_pos) - 1):
        lo, hi = fri_pos[k] + 1, fri_pos[k + 1] + 1
        block = net_125[lo:hi]
        if len(block):
            weekly_rets.append(float(np.prod(1.0 + block) - 1.0))
    weekly_rets = np.array(weekly_rets)
    ev_per_week_bp = float(np.mean(weekly_rets) * 1e4)
    win_rate_weekly = float(np.mean(weekly_rets > 0))
    worst_week = float(np.min(weekly_rets))
    print(f"[RESULT] EV/week={ev_per_week_bp:.3f}bp win_rate_wk={win_rate_weekly:.4f} "
          f"worst_week={worst_week:.4f} (n_weeks={len(weekly_rets)})", flush=True)

    # turnover: overlay #5's own flip-cost line item (above) PLUS mean daily |effective-leverage
    # change|, for direct comparison against the #1+#2-only policy's own already-reported
    # mean_daily_effective_leverage_turnover = 0.032870689681123534.
    lev_turnover_125 = float(np.mean(np.abs(np.diff(eff_lev_125, prepend=eff_lev_125[0]))))
    lev_turnover_12 = float(np.mean(np.abs(np.diff(eff_lev_12, prepend=eff_lev_12[0]))))
    print(f"[turnover] mean daily |eff-leverage change|: #1+#2 only = {lev_turnover_12:.6f}, "
          f"#1+#2+#5 = {lev_turnover_125:.6f}", flush=True)

    # ======================================================================================
    # TASK STEP 3: PASS/FAIL verdict against the pre-declared condition.
    # ======================================================================================
    sharpe_pass = m_125["sharpe"] > PASS_SHARPE_MIN
    maxdd_pass = m_125["maxdd"] <= PASS_MAXDD_MAX
    overall_pass = sharpe_pass and maxdd_pass
    verdict = "PASS" if overall_pass else "FAIL"
    print("\n" + "=" * 90)
    print(f"PRE-DECLARED VERDICT: {verdict}")
    print(f"  Sharpe(#1+#2+#5) = {m_125['sharpe']:.6f}  vs required > {PASS_SHARPE_MIN}  "
          f"-> {'PASS' if sharpe_pass else 'FAIL'}")
    print(f"  maxDD(#1+#2+#5)  = {m_125['maxdd']:.6f}  vs required <= {PASS_MAXDD_MAX}  "
          f"-> {'PASS' if maxdd_pass else 'FAIL'}")
    print("=" * 90, flush=True)

    # ======================================================================================
    # TASK STEP 4: regime breakdown, reusing ctaB.REGIME_WINDOWS + the 2025-04 squeeze week
    # convention already used in _dd_combined_waterfall.py.
    # ======================================================================================
    print("\n--- STEP 4: regime breakdown ---", flush=True)
    dd_wealth = np.cumprod(1.0 + np.nan_to_num(net_125))
    dd_peak = np.maximum.accumulate(dd_wealth)
    dd_ser = pd.Series(1.0 - dd_wealth / dd_peak, index=dates_scored)
    net_ser = pd.Series(net_125, index=dates_scored)
    lev_ser = pd.Series(eff_lev_125, index=dates_scored)
    m5_ser = pd.Series(m5, index=dates_scored)
    net12_ser = pd.Series(net_12, index=dates_scored)

    def window_report(label, start, end):
        s_ts = pd.Timestamp(start, tz="UTC")
        e_ts = pd.Timestamp(end, tz="UTC")
        mask = (dates_scored >= s_ts) & (dates_scored <= e_ts)
        if not mask.any():
            return {"window": [start, end], "n_days": 0, "note": "no overlap with scored sample"}
        d = dates_scored[mask]
        rep = {
            "window": [start, end], "n_days": int(mask.sum()),
            "cum_return_125": float(np.prod(1.0 + net_ser.loc[d]) - 1.0),
            "cum_return_12_only": float(np.prod(1.0 + net12_ser.loc[d]) - 1.0),
            "maxdd_in_window_125": float(dd_ser.loc[d].max()),
            "mean_effective_leverage_125": float(lev_ser.loc[d].mean()),
            "frac_days_gate_engaged": float((m5_ser.loc[d] == 0.5).mean()),
        }
        return rep

    regime_2020_03 = window_report("2020-03_COVID", *ctaB.REGIME_WINDOWS["2020-03_COVID"])
    regime_2022 = window_report("2022_full_year", *ctaB.REGIME_WINDOWS["2022_full_year"])
    regime_2018Q4 = window_report("2018Q4_chop", *ctaB.REGIME_WINDOWS["2018Q4_chop"])
    regime_meltup = window_report("2023_2024_meltup", *ctaB.REGIME_WINDOWS["2023_2024_meltup"])
    regime_squeeze = window_report("2025-04_squeeze_week", "2025-04-04", "2025-04-11")

    q4_n_days = regime_2018Q4.get("n_days", 0)
    q4_cum = regime_2018Q4.get("cum_return_12_only")
    if q4_n_days and q4_cum == 0.0:
        warmup_caveat_2018q4 = (
            "Inherited warmup artifact (not a market signal), same as the underlying #1+#2 and "
            "1x base construction: the 2018Q4_chop window predates the point where enough of the "
            "93-name universe has sufficient own trailing history for all three S1 rules to be "
            "simultaneously defined, so the #1+#2 book itself already shows exactly 0.0% return "
            "for this quarter (established fact, not introduced by overlay #5) -- read as "
            "'strategy not yet live', not 'de-risked out of a bear market'.")
    else:
        warmup_caveat_2018q4 = f"2018Q4_chop: cum_return_12_only={q4_cum} -- no 0.0% warmup flag."
    print(f"[2018Q4 warmup caveat] {warmup_caveat_2018q4}", flush=True)

    for lbl, rep in [("2020-03 COVID", regime_2020_03), ("2022 full year", regime_2022),
                     ("2018Q4 chop", regime_2018Q4), ("2023-24 meltup", regime_meltup),
                     ("2025-04 squeeze week", regime_squeeze)]:
        print(f"[{lbl}] {json.dumps(rep, indent=2, default=str)}", flush=True)

    # ======================================================================================
    # TASK STEP 5: synthetic-outlier-injection test on the NEW combination step itself (not just
    # on the already-separately-verified compute_state_a component) -- with a naive/unlagged
    # control proving the test has real discriminating power.
    # ======================================================================================
    print("\n--- STEP 5: synthetic-outlier-injection test on the #1+#2 -x- #5 combination step ---",
          flush=True)
    rng = np.random.default_rng(20260708)
    n_syn = 700
    syn_dates = pd.date_range("2015-01-02", periods=n_syn, freq="B", tz="UTC")
    # synthetic SPY close: random-walk in log space, enough length for 200d-SMA to be valid well
    # before the injection point.
    syn_close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_syn))),
                           index=syn_dates)
    # synthetic #1+#2 book: independent small daily returns + independent time-varying leverage,
    # standing in for net_12(t)/eff_lev_12(t) -- the combination step must be blind to whatever is
    # inside these arrays; only m5(t)'s OWN causal timing matters for the injection test.
    syn_net12 = rng.normal(0.0003, 0.01, n_syn)
    syn_eff_lev12 = 1.0 + 0.5 * np.abs(rng.normal(0, 1, n_syn))

    inject_pos = n_syn // 2
    baseline_state_at_pos = s1ens.compute_state_a(syn_close).iloc[inject_pos]
    shock_mult = 0.2 if baseline_state_at_pos == 1.0 else 5.0  # opposite-direction shock
    syn_close_shocked = syn_close.copy()
    syn_close_shocked.iloc[inject_pos] = syn_close_shocked.iloc[inject_pos] * shock_mult

    # --- CAUSAL (lag=1) version: run the ACTUAL build_gate_m5 + combine_step5 functions ---
    m5_causal_base = build_gate_m5(syn_close, syn_dates, lag=1)
    m5_causal_shocked = build_gate_m5(syn_close_shocked, syn_dates, lag=1)
    combo_causal_base = combine_step5(syn_net12, syn_eff_lev12, m5_causal_base)
    combo_causal_shocked = combine_step5(syn_net12, syn_eff_lev12, m5_causal_shocked)
    net125_causal_base = combo_causal_base["net_125"]
    net125_causal_shocked = combo_causal_shocked["net_125"]

    causal_pre_unaffected = bool(np.allclose(
        net125_causal_base[:inject_pos], net125_causal_shocked[:inject_pos]))
    causal_unaffected_on_inject_day = bool(np.isclose(
        net125_causal_base[inject_pos], net125_causal_shocked[inject_pos]))
    causal_reacts_next_day = bool(not np.isclose(
        net125_causal_base[inject_pos + 1], net125_causal_shocked[inject_pos + 1]))

    # --- NAIVE (lag=0, unlagged) control run through the SAME combine_step5 combination step:
    # should react ON the injection day itself, proving the methodology can detect a violation.
    m5_naive_base = build_gate_m5(syn_close, syn_dates, lag=0)
    m5_naive_shocked = build_gate_m5(syn_close_shocked, syn_dates, lag=0)
    combo_naive_base = combine_step5(syn_net12, syn_eff_lev12, m5_naive_base)
    combo_naive_shocked = combine_step5(syn_net12, syn_eff_lev12, m5_naive_shocked)
    naive_reacts_on_inject_day = bool(not np.isclose(
        combo_naive_base["net_125"][inject_pos], combo_naive_shocked["net_125"][inject_pos]))

    injection_result = {
        "inject_pos": int(inject_pos), "inject_date": str(syn_dates[inject_pos].date()),
        "baseline_state_at_inject_pos": float(baseline_state_at_pos),
        "shock_multiplier_used": shock_mult,
        "causal_version_of_full_combination_step": {
            "pre_injection_days_unaffected": causal_pre_unaffected,
            "unaffected_on_injection_day_itself": causal_unaffected_on_inject_day,
            "reacts_starting_next_day": causal_reacts_next_day,
        },
        "naive_unlagged_control_same_combination_step": {
            "reacts_on_injection_day_itself": naive_reacts_on_inject_day,
        },
        "test_has_discriminating_power": bool(
            causal_unaffected_on_inject_day and naive_reacts_on_inject_day),
    }
    print(json.dumps(injection_result, indent=2), flush=True)
    print(f"[injection test] interpretation: running the ACTUAL build_gate_m5()+combine_step5() "
          f"composition end-to-end on synthetic data shows the causal (lag=1) version is "
          f"byte-identical before AND on the injection day itself, and only diverges starting the "
          f"next day; the naive/unlagged (lag=0) control run through the IDENTICAL combine_step5 "
          f"combination step DOES react same-day -- confirming the combination step itself "
          f"introduces no new lookahead and that this test can actually detect one if it existed.",
          flush=True)
    assert injection_result["test_has_discriminating_power"], \
        "injection test failed to discriminate -- STOP, do not trust downstream results"
    assert causal_pre_unaffected and causal_unaffected_on_inject_day and causal_reacts_next_day, \
        "causal combination step shows lookahead -- STOP"

    # ======================================================================================
    # TASK STEP 6: CPCV (8 groups, k=2, embargo=252), reusing ctaB.cpcv_paths() verbatim, plus
    # drop-top-5-days robustness, matching _dd_combined_cpcv.py's own recipe.
    # ======================================================================================
    print("\n--- STEP 6: CPCV (reused ctaB.cpcv_paths verbatim) ---", flush=True)
    from itertools import combinations as _combos
    folds = ctaB.cpcv_paths(net_125, ctaB.CPCV_N_GROUPS, ctaB.CPCV_K, ctaB.EMBARGO_TRADING_DAYS, ppy)
    n_theoretical = len(list(_combos(range(ctaB.CPCV_N_GROUPS), ctaB.CPCV_K)))
    path_sharpes = [f["sharpe_ann"] for f in folds]
    cpcv_summary = {
        "n_groups": ctaB.CPCV_N_GROUPS, "k": ctaB.CPCV_K, "embargo": ctaB.EMBARGO_TRADING_DAYS,
        "n_theoretical_combinations": n_theoretical,
        "n_valid_combinations_used": len(folds),
        "dropped_combinations": n_theoretical - len(folds),
        "sharpe_mean": float(np.mean(path_sharpes)) if path_sharpes else None,
        "sharpe_std": float(np.std(path_sharpes, ddof=0)) if path_sharpes else None,
        "sharpe_median": float(np.median(path_sharpes)) if path_sharpes else None,
        "sharpe_min": float(np.min(path_sharpes)) if path_sharpes else None,
        "sharpe_max": float(np.max(path_sharpes)) if path_sharpes else None,
        "frac_positive_sharpe": float(np.mean(np.array(path_sharpes) > 0)) if path_sharpes else None,
        "folds": folds,
    }
    print(f"[CPCV] n_theoretical={n_theoretical} n_valid={len(folds)} "
          f"sharpe_mean={cpcv_summary['sharpe_mean']} sharpe_std={cpcv_summary['sharpe_std']} "
          f"frac_pos={cpcv_summary['frac_positive_sharpe']}", flush=True)

    order = np.argsort(net_125)[::-1]
    top5_idx = order[:5]
    drop5_ret = net_125.copy()
    drop5_ret[top5_idx] = 0.0
    cagr_drop5, maxdd_drop5 = ctaB._cagr_from_returns(drop5_ret, dates_scored)
    sharpe_drop5 = float(ctaB.per_obs_sharpe(drop5_ret[np.isfinite(drop5_ret)]) * np.sqrt(ppy))
    sharpe_full_pobs = float(ctaB.per_obs_sharpe(net_125[np.isfinite(net_125)]) * np.sqrt(ppy))
    drop_top5 = {
        "top5_dates": [str(dates_scored[i].date()) for i in top5_idx],
        "top5_returns": [float(net_125[i]) for i in top5_idx],
        "full_sample_sharpe_ann_per_obs_convention": sharpe_full_pobs,
        "cagr_after_drop5": cagr_drop5, "maxdd_after_drop5": maxdd_drop5,
        "sharpe_ann_after_drop5": sharpe_drop5,
    }
    print(f"[drop-top-5] {json.dumps(drop_top5, indent=2)}", flush=True)

    # ======================================================================================
    # TASK STEP 7: DSR, informational only, both N_trials framings reported explicitly.
    # ======================================================================================
    print("\n--- STEP 7: DSR (informational only, both N_trials framings) ---", flush=True)
    dsr_n2 = ctaB.dsr(net_125, n_trials=2, periods_per_yr=ppy)
    dsr_n5 = ctaB.dsr(net_125, n_trials=5, periods_per_yr=ppy)
    print(f"[DSR] n_trials=2 (nearest well-defined lower bound, N=1 degenerate for the PPF "
          f"formula): {dsr_n2}", flush=True)
    print(f"[DSR] n_trials=5 (this being the 5th configuration ever evaluated in the full "
          f"waterfall-plus-this-followup lineage: Step A stop-after-A/-C/-D4/-D5 [4] + this "
          f"one-shot #1+#2+#5 followup [+1]): {dsr_n5}", flush=True)

    # ======================================================================================
    # Write outputs.
    # ======================================================================================
    netret_df = pd.DataFrame({
        "date": dates_scored, "net_ret_125": net_125, "effective_leverage_125": eff_lev_125,
        "m5_gate": m5, "net_ret_12_only": net_12, "effective_leverage_12_only": eff_lev_12,
    })
    netret_df.to_csv(NETRET_CSV, index=False)
    print(f"\n[output] wrote {NETRET_CSV} ({len(netret_df)} rows)", flush=True)

    results = {
        "step1_reproduction": {"recomputed": repro_12, "previously_reported": EXISTING_12_REPORTED,
                                "passed": repro_ok},
        "combination_method_precedent_note": (
            "m5(t) applied via notional/margin scaling with NO incremental cost for the overlay's "
            "own day-to-day multiplier changes beyond the explicit flip-cost line item -- the "
            "same precedent already established for overlay #1 (GZ-brake m_t) and overlay #3 "
            "(vol-targeting scale(t)) in _dd_combined_waterfall.py."
        ),
        "overlay5_gate": {
            "n_flip_days": n_flip_days, "gate_active_fraction_of_days": gate_active_frac,
            "n_leading_days_defaulted_gate_off": n_leading_default,
            "data_gap_tail_calendar_days_spy_vs_book": data_gap_days,
            "total_extra_flip_cost_bp": total_extra_cost_bp,
            "mean_extra_flip_cost_bp_per_flip_day": mean_extra_cost_per_flip_bp,
        },
        "step2_combination_result": {
            "metrics": m_125, "realized_avg_leverage": realized_avg_lev_125,
            "ev_per_week_bp": ev_per_week_bp, "win_rate_weekly": win_rate_weekly,
            "worst_week": worst_week, "n_weeks": len(weekly_rets),
            "mean_daily_effective_leverage_turnover_125": lev_turnover_125,
            "mean_daily_effective_leverage_turnover_12_only": lev_turnover_12,
        },
        "step3_verdict": {
            "sharpe_125": m_125["sharpe"], "required_sharpe_gt": PASS_SHARPE_MIN,
            "sharpe_pass": sharpe_pass,
            "maxdd_125": m_125["maxdd"], "required_maxdd_lte": PASS_MAXDD_MAX,
            "maxdd_pass": maxdd_pass,
            "overall_verdict": verdict,
        },
        "step4_regime_breakdown": {
            "2020-03_COVID": regime_2020_03, "2022_full_year": regime_2022,
            "2018Q4_chop": regime_2018Q4, "2018Q4_warmup_caveat": warmup_caveat_2018q4,
            "2023_2024_meltup": regime_meltup, "2025-04_squeeze_week": regime_squeeze,
        },
        "step5_injection_test": injection_result,
        "step6_cpcv": cpcv_summary,
        "step6_drop_top5": drop_top5,
        "step7_dsr_informational": {
            "dsr_n_trials_2_nearest_lower_bound": dsr_n2,
            "dsr_n_trials_5_full_lineage_count": dsr_n5,
            "note": "DSR/PBO reported informationally only per the frozen adoption-gate "
                    "convention (ctaB_pipeline.py docstring) -- never a pass/fail gate here.",
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[output] wrote {RESULTS_JSON}", flush=True)
    print(f"\n=== FINAL VERDICT: {verdict} ===", flush=True)
    print(f"=== done in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
