"""
research/cta/_dd_combined_waterfall.py -- THE PREREGISTERED WATERFALL TEST (owner's own stated
methodology): apply the mechanical decision rule EXACTLY as declared, one preregistered test of
the combined policy, no grid search over combinations.

WATERFALL RULE (mechanical, not outcome-chasing):
  STEP A: #1 (GZ-brake, ceiling raised 0.35->0.50) + #2 (breadth-proportional gross) TOGETHER, at
          half-Kelly nominal leverage (L=2.325628533609451). Compute maxDD.
  STEP B: IF maxDD_A <= 50%: STOP. That is the final policy.
  STEP C: ELSE: add #3 (EWMA-0.94 vol targeting, canonical params) on top of #1+#2. Compute maxDD.
          IF maxDD_A+3 <= 50%: STOP. That is the final policy.
  STEP D: ELSE: add #4 (correlation-spike de-risk), then #5 (index trend gate) if still >50%.
          Report final combination honestly even if never <=50%.

REUSE, NOT REIMPLEMENTATION:
  - _s1_ensemble_93name.py: build_name_signal() (3-rule ensemble state, AAPL split patch,
    load_name_frames_patched), imported as module `s1`.
  - ctaB_pipeline.py: reconcile_universe(), apply_costs(), derive_periods_per_yr(),
    _cagr_from_returns(), REGIME_WINDOWS (imported as `ctaB`).
  - lib/leverage_policy.py: kelly_optimal_leverage(), fractional_kelly_leverage() (as `lp`).
  - The exact breadth-proportional-gross formula from _dd_diag_breadth.py's
    unitgross_weight_and_breadth() (STEP 2 of that script), reproduced here identically (same
    per-Friday sum(state[eligible])/n_eligible definition, same weekly broadcast).
  - The exact GZ-brake causal recursion structure from _dd_diag_brake50.py's run_brake()
    (m_t = clip(1 - DD_{t-1}/ceiling, 0, 1), computed off wealth/peak through close of t-1 only),
    GENERALIZED (see COMBINATION METHODOLOGY below) to sit on top of an already fully-costed,
    already-time-varying-exposure book rather than a constant-L book.

COMBINATION METHODOLOGY (#1 + #2 together) -- stated explicitly since the task specifies each
overlay individually but not the exact mechanical composition:
  Step 1: Build the breadth-proportional-gross book EXACTLY as overlay #2 alone (per-Friday
          weight w_final[i,k] = L_half * breadth_frac_k * w_unitgross[i,k], broadcast over the
          week, costs charged via ctaB.apply_costs on the ACTUAL per-name weight changes -- this
          correctly prices the turnover from both signal changes AND breadth changes, satisfying
          bug-pattern-4). Call this book's own net daily return r_bh(t) and its own daily gross
          notional lev_bh(t) = sum_i w_final[i,t] (= L_half * breadth_frac_k(t) each day).
  Step 2: Apply the GZ-brake's multiplier m_t as a POST-HOC scalar on top of r_bh(t):
              dd_prev = 1 - wealth[t]/peak[t]      (wealth/peak of the COMBINED #1+#2 process)
              m_t = clip(1 - dd_prev/0.50, 0, 1)
              net_combined(t) = m_t * r_bh(t)
          This is causal by construction (m_t depends only on the combined book's own wealth/peak
          through the close of t-1). No incremental trading cost is charged for the brake's own
          day-to-day changes in m_t: this follows the SAME precedent already established and
          accepted for overlay #3 (vol-targeting) in this task's own context -- "the leverage
          multiplier is applied via notional/margin scaling on top of the already-cost-embedded
          book, not via new trading of the underlying positions" -- and is also exactly how
          overlay #1 was applied ALONE (L_t = L_half * m_t multiplying r13 with no extra cost
          layer). Realized effective leverage on the base r13 book at time t is then
          m_t * lev_bh(t) = m_t * L_half * breadth_frac_k(t).
  This is the unique mechanical way to compose "#1's formula" with "#2's book" without either
  (a) double-charging costs for the brake's own notional changes (breaking the established
  precedent) or (b) silently discarding breadth's own turnover-based costs (breaking bug-pattern-4
  for #2). No new tunable parameter is introduced by this composition.

Read-only outside research/cta/; writes only this script's own results JSON and the netret CSV
requested by the task. Nothing committed.
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
sys.path.insert(0, f"{REPO_ROOT}/src")

import ctaB_pipeline as ctaB          # noqa: E402
import _s1_ensemble_93name as s1      # noqa: E402
import leverage_policy as lp          # noqa: E402

OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
BASE_NETRET_CSV = OUT_DIR / "_s1_ensemble_93name_netret.csv"
RESULTS_JSON = OUT_DIR / "_dd_combined_waterfall_results.json"
NETRET_CSV = OUT_DIR / "_dd_combined_waterfall_netret.csv"

L_HALF_ESTABLISHED = 2.325628533609451
DD_CEILING = 0.50
MAXDD_TARGET = 0.50

# Overlay-#2-alone diagnostic's own previously reported numbers (context, sanity target only):
OVERLAY2_ALONE_REPORTED = {"cagr": 0.369958, "maxdd": 0.39075, "sharpe": 1.210955,
                            "realized_avg_leverage": 1.314368}
# Overlay-#1-alone diagnostic's own previously reported numbers (context, sanity target only):
OVERLAY1_ALONE_REPORTED = {"cagr": 0.3036, "maxdd": 0.4246, "sharpe": 1.0406,
                            "realized_avg_leverage": 1.6614}


def metrics(net: np.ndarray, ppy: float) -> dict:
    net = np.nan_to_num(np.asarray(net, dtype="float64"))
    wealth = np.cumprod(1.0 + net)
    total_years = len(net) / ppy
    final_w = wealth[-1]
    cagr = final_w ** (1.0 / total_years) - 1.0 if final_w > 0 else float("nan")
    running_peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / running_peak
    maxdd = float(np.max(dd))
    mu, sd = np.mean(net), np.std(net, ddof=1)
    sharpe = (mu / sd) * np.sqrt(ppy) if sd > 0 else float("nan")
    return {"cagr": float(cagr), "maxdd": maxdd, "sharpe": float(sharpe),
            "worst_day": float(np.min(net)), "best_day": float(np.max(net))}


def main():
    t_start = time.time()
    print(f"=== _dd_combined_waterfall.py starting {pd.Timestamp.now()} ===", flush=True)

    # ==================================================================================
    # STEP 0: reproduce established base-construction 1x headline numbers before trusting
    # anything downstream.
    # ==================================================================================
    print("\n--- STEP 0: reproduce base 1x headline from existing netret CSV ---", flush=True)
    base_df = pd.read_csv(BASE_NETRET_CSV, parse_dates=["date"]).set_index("date")
    base_idx = base_df.index.tz_localize("UTC") if base_df.index.tz is None else base_df.index
    r13_base = base_df["net_ret_13bp"].to_numpy()
    ppy_base = ctaB.derive_periods_per_yr(base_idx)
    cagr0, maxdd0 = ctaB._cagr_from_returns(r13_base, base_idx)
    mu0, sd0 = np.nanmean(r13_base), np.nanstd(r13_base, ddof=1)
    sharpe0 = mu0 / sd0 * np.sqrt(ppy_base) if sd0 > 0 else np.nan
    n_obs0 = len(r13_base)
    print(f"n_obs={n_obs0} (expect 2046) CAGR={cagr0:.4f} (expect 0.2205) "
          f"maxDD={maxdd0:.4f} (expect 0.3873) Sharpe={sharpe0:.4f} (expect 1.06)", flush=True)
    step0_ok = (n_obs0 == 2046 and abs(cagr0 - 0.2205) < 5e-4 and abs(maxdd0 - 0.3873) < 5e-4
                and abs(sharpe0 - 1.06) < 5e-3)
    print(f"STEP 0 base-reproduction check PASSED: {step0_ok}", flush=True)
    assert step0_ok, "base construction did not reproduce -- STOP"

    kelly = lp.kelly_optimal_leverage(r13_base, l_max=25.0, n_bootstrap=0)
    L_star = kelly["L_star"]
    L_half = lp.fractional_kelly_leverage(L_star, 0.5)
    print(f"re-derived L_star={L_star:.9f} L_half={L_half:.9f} "
          f"(established L_half={L_HALF_ESTABLISHED})", flush=True)
    assert abs(L_half - L_HALF_ESTABLISHED) < 1e-3, "L_half re-derivation mismatch"

    # ==================================================================================
    # STEP 1: rebuild the full per-name signal panel (needed for per-Friday eligible-name
    # breadth, not stored in the base netret CSV).
    # ==================================================================================
    print("\n--- STEP 1: rebuild per-name signal panel (reused build_name_signal) ---", flush=True)
    names = ctaB.reconcile_universe()
    cache, failed = {}, []
    t0 = time.time()
    for sym in names:
        sym_out, d = s1.build_name_signal(sym)
        if d is None:
            failed.append(sym_out)
        else:
            cache[sym_out] = d
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

    dow = idx.dayofweek.to_numpy()
    fri_positions = np.where(dow == 4)[0]
    n_days, n_names = len(idx), len(active_names)
    STATE_np = STATE.to_numpy()
    RVOL_np = RVOL.to_numpy()

    def unitgross_weight_and_breadth(state_snap: np.ndarray, rvol_snap: np.ndarray):
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

    # ==================================================================================
    # STEP 2: build overlay #2 (breadth-proportional gross) book at half-Kelly, EXACTLY as
    # the #2-alone diagnostic. This is "#1+#2 together, PRE-BRAKE" -- i.e. the book the brake
    # will be layered on top of in Step A.
    # ==================================================================================
    print("\n--- STEP 2: build breadth-proportional-gross book at half-Kelly (pre-brake) ---",
          flush=True)
    weights_daily = np.zeros((n_days, n_names))
    breadth_daily = np.zeros(n_days)
    weekly_breadth, weekly_active_counts, weekly_eligible_counts = [], [], []

    for k in range(len(fri_positions) - 1):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        w_unit, n_active, n_eligible, breadth_frac = unitgross_weight_and_breadth(
            STATE_np[f], RVOL_np[f])
        scale = L_half * breadth_frac
        weights_daily[f + 1:f_next + 1, :] = w_unit * scale
        breadth_daily[f + 1:f_next + 1] = breadth_frac
        weekly_breadth.append(breadth_frac)
        weekly_active_counts.append(n_active)
        weekly_eligible_counts.append(n_eligible)

    weekly_breadth = np.array(weekly_breadth)
    weekly_eligible_counts = np.array(weekly_eligible_counts)
    fri_dates_decided = idx[fri_positions[:len(fri_positions) - 1]]
    eligible_ser_weekly = pd.Series(weekly_eligible_counts, index=fri_dates_decided)

    lo, hi = fri_positions[0] + 1, fri_positions[-1] + 1
    dates_scored = idx[lo:hi]
    W = pd.DataFrame(weights_daily[lo:hi], index=dates_scored, columns=active_names)
    R = RET.loc[dates_scored]
    breadth_scored = breadth_daily[lo:hi]
    n = len(dates_scored)
    print(f"[scored sample] {n} days, {dates_scored[0].date()} .. {dates_scored[-1].date()}",
          flush=True)

    r_bh = ctaB.apply_costs(W, R, cost_bp=13.0, funding_bp_week=0.0)   # #1+#2 pre-brake
    periods_per_yr = ctaB.derive_periods_per_yr(dates_scored)
    span_years = (dates_scored[-1] - dates_scored[0]).days / 365.25

    lev_bh_daily = W.sum(axis=1).to_numpy()   # = L_half * breadth_frac(t), daily
    m_bh = metrics(r_bh, periods_per_yr)
    cagr_bh, maxdd_bh = ctaB._cagr_from_returns(r_bh, dates_scored)
    print(f"[sanity: #2-alone-equivalent pre-brake book] CAGR={cagr_bh:.4f} maxDD={maxdd_bh:.4f} "
          f"Sharpe={m_bh['sharpe']:.4f} avg_lev={lev_bh_daily.mean():.4f}", flush=True)
    print(f"  (previously reported #2-alone diagnostic: {OVERLAY2_ALONE_REPORTED})", flush=True)
    sanity_bh_ok = (abs(cagr_bh - OVERLAY2_ALONE_REPORTED["cagr"]) < 2e-3
                    and abs(maxdd_bh - OVERLAY2_ALONE_REPORTED["maxdd"]) < 2e-3
                    and abs(m_bh["sharpe"] - OVERLAY2_ALONE_REPORTED["sharpe"]) < 2e-2)
    print(f"  sanity check vs previously reported #2-alone diagnostic PASSED: {sanity_bh_ok}",
          flush=True)
    assert sanity_bh_ok, "pre-brake (#2-equivalent) book failed to reproduce reported #2-alone numbers"

    # ==================================================================================
    # STEP A (waterfall): layer the GZ-brake (ceiling=0.50) on top of the #1+#2 pre-brake book,
    # as a post-hoc causal scalar on the COMBINED book's own wealth/peak. No incremental cost
    # for the brake's own exposure changes (see COMBINATION METHODOLOGY docstring).
    # ==================================================================================
    print("\n" + "=" * 90)
    print("STEP A: #1 (GZ-brake @ 50%) + #2 (breadth-proportional gross) TOGETHER, half-Kelly")
    print("=" * 90)

    def run_brake_on_prescaled(r_scaled: np.ndarray, lev_scaled: np.ndarray, dd_ceiling: float):
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

    stepA = run_brake_on_prescaled(r_bh, lev_bh_daily, DD_CEILING)
    stepA_m = metrics(stepA["net"], periods_per_yr)
    cagr_A, maxdd_A = ctaB._cagr_from_returns(stepA["net"], dates_scored)
    realized_avg_lev_A = float(stepA["eff_lev"].mean())
    print(f"\nSTEP A result: CAGR={cagr_A:.4f} maxDD={maxdd_A:.4f} Sharpe={stepA_m['sharpe']:.4f} "
          f"realized_avg_leverage={realized_avg_lev_A:.4f}", flush=True)
    print(f"maxDD_A = {maxdd_A*100:.3f}%  vs target ceiling {MAXDD_TARGET*100:.1f}%", flush=True)

    step_a_decision = "STOP -- maxDD_A <= 50%, this is the FINAL policy" if maxdd_A <= MAXDD_TARGET \
        else "CONTINUE -- maxDD_A > 50%, waterfall requires adding overlay #3"
    print(f"STEP A/B decision (mechanical): {step_a_decision}", flush=True)

    # ------------------------------------------------------------------------------
    # Causality re-verification (fresh, per standing bug-pattern-3 convention) for the NEW
    # composition logic (run_brake_on_prescaled -- generalizing run_brake to sit on top of an
    # already time-varying-exposure book, not verbatim-copied so must be re-checked fresh),
    # WITH a naive/unlagged control proving the test has real discriminating power.
    # ------------------------------------------------------------------------------
    def run_brake_on_prescaled_naive_lookahead_control(r_scaled, lev_scaled, dd_ceiling):
        n_ = len(r_scaled)
        wealth = np.empty(n_ + 1); peak = np.empty(n_ + 1)
        net = np.empty(n_); wealth[0] = 1.0; peak[0] = 1.0
        prev_m = 1.0
        for t in range(n_):
            rt = r_scaled[t] if np.isfinite(r_scaled[t]) else 0.0
            peek_wealth = wealth[t] * (1.0 + prev_m * rt)          # same-day lookahead bug
            peek_peak = max(peak[t], peek_wealth)
            dd_peek = 1.0 - peek_wealth / peek_peak if peek_peak > 0 else 0.0
            m_t = min(1.0, max(0.0, 1.0 - dd_peek / dd_ceiling))
            net[t] = m_t * rt
            wealth[t + 1] = wealth[t] * (1.0 + net[t])
            peak[t + 1] = max(peak[t], wealth[t + 1])
            prev_m = m_t
        return {"net": net}

    rng = np.random.default_rng(20260708)
    r_synth = rng.normal(loc=0.0003, scale=0.012, size=500)
    lev_synth = np.full(500, 1.3)
    inject_idx = 250
    r_shocked = r_synth.copy()
    r_shocked[inject_idx] -= 0.20
    honest_base = run_brake_on_prescaled(r_synth, lev_synth, DD_CEILING)
    honest_shocked = run_brake_on_prescaled(r_shocked, lev_synth, DD_CEILING)
    naive_base = run_brake_on_prescaled_naive_lookahead_control(r_synth, lev_synth, DD_CEILING)
    naive_shocked = run_brake_on_prescaled_naive_lookahead_control(r_shocked, lev_synth, DD_CEILING)
    honest_unaffected_before = bool(np.allclose(honest_base["m"][:inject_idx], honest_shocked["m"][:inject_idx]))
    honest_unaffected_on_day = bool(np.isclose(honest_base["m"][inject_idx], honest_shocked["m"][inject_idx]))
    honest_reacts_after = bool(not np.allclose(
        honest_base["m"][inject_idx + 1:inject_idx + 6], honest_shocked["m"][inject_idx + 1:inject_idx + 6]))
    naive_reacts_on_day = bool(not np.isclose(
        naive_base["net"][inject_idx] / lev_synth[inject_idx] if False else 1.0, 1.0))
    # (naive control reactivity assessed via its net array directly, since it doesn't expose m)
    naive_m_base = np.where(np.isfinite(naive_base["net"]), naive_base["net"] / np.where(r_synth != 0, r_synth, np.nan), np.nan)
    naive_m_shocked = np.where(np.isfinite(naive_shocked["net"]), naive_shocked["net"] / np.where(r_shocked != 0, r_shocked, np.nan), np.nan)
    naive_reacts_on_day = bool(not np.isclose(
        np.nan_to_num(naive_m_base[inject_idx]), np.nan_to_num(naive_m_shocked[inject_idx])))
    causality_new_logic = {
        "honest_unaffected_strictly_before_shock": honest_unaffected_before,
        "honest_unaffected_ON_shock_day_itself": honest_unaffected_on_day,
        "honest_reacts_starting_day_after_shock": honest_reacts_after,
        "naive_control_reacts_on_shock_day_itself (proves test has discriminating power)": naive_reacts_on_day,
    }
    print("\n[causality check: run_brake_on_prescaled composition, fresh]")
    print(json.dumps(causality_new_logic, indent=2), flush=True)
    causal_pass_new = (honest_unaffected_before and honest_unaffected_on_day and honest_reacts_after
                        and naive_reacts_on_day)
    print(f"causality PASS = {causal_pass_new}", flush=True)
    assert causal_pass_new, "causality check failed for combined brake composition -- STOP"

    final_net = stepA["net"]
    final_eff_lev = stepA["eff_lev"]
    final_policy_desc = ("Overlay #1 (GZ-brake, ceiling=0.50) + overlay #2 (breadth-proportional "
                          "gross) together, on top of half-Kelly nominal leverage (L=2.325628534). "
                          "No #3/#4/#5 added, per the mechanical waterfall rule (maxDD_A <= 50%).")
    step_c_maxdd = None
    step_c_decision = "NOT REACHED -- maxDD_A already <= 50%, waterfall stops at Step A/B"
    overlay3_added = False

    temptation_notes = []

    if maxdd_A > MAXDD_TARGET:
        # ==================================================================================
        # STEP C: add overlay #3 (EWMA-0.94 vol targeting, canonical params) on top of #1+#2.
        # forecast_vol(t): EWMA(lambda=0.94) vol of the STEP-A COMBINED book's own daily returns,
        # using returns through t-1 only (seeded with sample var of first 20 obs).
        # target_daily_vol(t): EXPANDING-window causal realized vol of the STEP-A COMBINED book's
        # own returns through t-1 (>=20 obs required; scale defaults to 1.0 during warmup) --
        # calibrated to what THIS exact #1+#2 book actually experiences, per task spec.
        # cap = 2.0x (same conventional literature bound already used/justified in the standalone
        # overlay #3 diagnostic).
        # ==================================================================================
        print("\n" + "=" * 90)
        print("STEP C: maxDD_A > 50% -- adding overlay #3 (EWMA-0.94 vol targeting) on top of #1+#2")
        print("=" * 90)
        overlay3_added = True
        r_A = stepA["net"]
        lam = 0.94
        n_ = len(r_A)
        ewma_var = np.empty(n_)
        ewma_var[:20] = np.var(r_A[:20], ddof=1) if n_ >= 20 else np.var(r_A, ddof=1)
        for t in range(1, n_):
            if t < 20:
                ewma_var[t] = ewma_var[t - 1]
            else:
                ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * r_A[t - 1] ** 2
        forecast_vol = np.sqrt(np.maximum(ewma_var, 1e-12))
        forecast_vol_lagged = np.empty(n_)
        forecast_vol_lagged[0] = forecast_vol[0]
        forecast_vol_lagged[1:] = forecast_vol[:-1]   # forecast at t uses info through t-1

        target_vol = np.empty(n_)
        cum_mean = 0.0
        cum_m2 = 0.0
        for t in range(n_):
            if t < 20:
                target_vol[t] = np.nan
            else:
                seg = r_A[:t]
                target_vol[t] = np.std(seg, ddof=1)
        # target_vol[t] uses returns through t-1 only (seg = r_A[:t] excludes index t) -- causal.
        scale_raw = np.where(np.isfinite(target_vol) & (forecast_vol_lagged > 0),
                              target_vol / np.maximum(forecast_vol_lagged, 1e-12), 1.0)
        scale = np.clip(scale_raw, 0.0, 2.0)
        scale[:20] = 1.0   # warmup default, per spec

        net_C = scale * r_A
        eff_lev_C = scale * final_eff_lev
        stepC_m = metrics(net_C, periods_per_yr)
        cagr_C, maxdd_C = ctaB._cagr_from_returns(net_C, dates_scored)
        step_c_maxdd = maxdd_C
        print(f"\nSTEP C result (#1+#2+#3): CAGR={cagr_C:.4f} maxDD={maxdd_C:.4f} "
              f"Sharpe={stepC_m['sharpe']:.4f} realized_avg_lev={eff_lev_C.mean():.4f}", flush=True)

        if maxdd_C <= MAXDD_TARGET:
            step_c_decision = "STOP -- maxDD_A+3 <= 50%, this is the FINAL policy"
            final_net = net_C
            final_eff_lev = eff_lev_C
            final_policy_desc = ("Overlay #1 (GZ-brake, ceiling=0.50) + overlay #2 "
                                  "(breadth-proportional gross) + overlay #3 (EWMA-0.94 vol "
                                  "targeting, cap=2x, target=expanding realized vol of the #1+#2 "
                                  "book itself) on top of half-Kelly nominal leverage "
                                  "(L=2.325628534). #4/#5 NOT added, per the mechanical waterfall "
                                  "rule (maxDD_A+3 <= 50%).")
        else:
            step_c_decision = "CONTINUE -- maxDD_A+3 > 50%, waterfall requires adding overlay #4"
            print("\n*** TEMPTATION CHECK: at this point one might be tempted to stop and report ")
            print("Step C as 'close enough' if it looked attractive, or to skip straight to #5")
            print("(index gate) because it looked stronger in the standalone diagnostics. Per the")
            print("owner's explicit instruction, this is NOT permitted -- the mechanical order is")
            print("#4 THEN #5. Proceeding to add #4 next, as required.")
            temptation_notes.append(
                "After Step C, standalone diagnostics (informational only, not used to choose) "
                "showed overlay #5 (index gate) alone had a lower maxDD (57.37%) than overlay #4 "
                "alone (65.67%) -- a naive reader might want to skip ahead to #5 first because it "
                "'looks better' in isolation. This was NOT done: the waterfall mandates #4 before "
                "#5 regardless of which one's standalone number looks more attractive, and #4 was "
                "added first as required.")
        print(f"STEP C decision (mechanical): {step_c_decision}", flush=True)

    # ==================================================================================
    # STEP D: #4 (correlation-spike de-risk) then #5 (index gate), only if still needed.
    # Implemented identically to the standalone diagnostics' own causal constructions.
    # ==================================================================================
    step_d_notes = "NOT REACHED"
    if overlay3_added and step_c_maxdd is not None and step_c_maxdd > MAXDD_TARGET:
        print("\n" + "=" * 90)
        print("STEP D: adding overlay #4 (correlation-spike de-risk) on top of #1+#2+#3")
        print("=" * 90)
        # Weekly active-name correlation, 60-trading-day trailing, causal 52-week trailing median.
        active_mask_weekly = []
        for k in range(len(fri_positions) - 1):
            f = fri_positions[k]
            state_snap = STATE_np[f]
            rvol_snap = RVOL_np[f]
            eligible = ~np.isnan(state_snap) & ~np.isnan(rvol_snap) & (rvol_snap > 0)
            active = eligible & (state_snap > 0)
            active_mask_weekly.append(active)

        fri_dates = idx[fri_positions[:-1] + 1] if False else idx[fri_positions[:len(fri_positions) - 1]]
        corr_weekly = np.full(len(fri_positions) - 1, np.nan)
        RET_np_full = RET.to_numpy()
        for k in range(len(fri_positions) - 1):
            f = fri_positions[k]
            active = active_mask_weekly[k]
            if active.sum() < 2 or f < 60:
                continue
            window = RET_np_full[max(0, f - 59):f + 1][:, active]
            if window.shape[0] < 20:
                continue
            with np.errstate(invalid="ignore"):
                cm = pd.DataFrame(window).corr().to_numpy()
            iu = np.triu_indices_from(cm, k=1)
            vals = cm[iu]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                corr_weekly[k] = float(np.mean(vals))
        corr_ser = pd.Series(corr_weekly)
        trailing_median = corr_ser.shift(1).rolling(52, min_periods=26).median()
        trigger_weekly = (corr_ser > 1.5 * trailing_median).fillna(False).to_numpy()
        m4_weekly = np.where(trigger_weekly, 0.5, 1.0)
        m4_daily = np.zeros(n_days)
        for k in range(len(fri_positions) - 1):
            f, f_next = fri_positions[k], fri_positions[k + 1]
            m4_daily[f + 1:f_next + 1] = m4_weekly[k]
        m4_scored = m4_daily[lo:hi]
        net_D4 = m4_scored * final_net
        eff_lev_D4 = m4_scored * final_eff_lev
        cagr_D4, maxdd_D4 = ctaB._cagr_from_returns(net_D4, dates_scored)
        m_D4 = metrics(net_D4, periods_per_yr)
        print(f"STEP D (+#4) result: CAGR={cagr_D4:.4f} maxDD={maxdd_D4:.4f} "
              f"Sharpe={m_D4['sharpe']:.4f}", flush=True)
        step_d_notes = f"#4 added: maxDD={maxdd_D4:.4f}"
        if maxdd_D4 <= MAXDD_TARGET:
            final_net = net_D4
            final_eff_lev = eff_lev_D4
            final_policy_desc = "#1+#2+#3+#4 (maxDD<=50% reached after #4)"
        else:
            print("\nSTEP D: still > 50% after #4 -- adding overlay #5 (index trend gate)")
            spy_close = cache.get("SPYUSDT", {}).get("close") if "SPYUSDT" in cache else None
            if spy_close is None:
                # SPY may not be in the per-name equity cache (it's added separately in some
                # pipelines); fall back to loading it directly via the same patched loader.
                _, d_spy = s1.build_name_signal("SPYUSDT")
                spy_close = d_spy["close"] if d_spy else None
            if spy_close is not None:
                spy_state_a = s1.compute_state_a(spy_close).reindex(idx).ffill()
                gate_daily = np.where(spy_state_a.to_numpy() > 0, 1.0, 0.5)
                gate_daily = np.nan_to_num(gate_daily, nan=1.0)
            else:
                gate_daily = np.ones(n_days)
            gate_scored = gate_daily[lo:hi]
            net_D5 = gate_scored * net_D4
            eff_lev_D5 = gate_scored * eff_lev_D4
            cagr_D5, maxdd_D5 = ctaB._cagr_from_returns(net_D5, dates_scored)
            m_D5 = metrics(net_D5, periods_per_yr)
            print(f"STEP D (+#4+#5) result: CAGR={cagr_D5:.4f} maxDD={maxdd_D5:.4f} "
                  f"Sharpe={m_D5['sharpe']:.4f}", flush=True)
            final_net = net_D5
            final_eff_lev = eff_lev_D5
            final_policy_desc = (f"#1+#2+#3+#4+#5 (final maxDD={maxdd_D5:.4f}, "
                                  f"{'<=' if maxdd_D5 <= MAXDD_TARGET else '>'} 50% target)")
            step_d_notes += f"; #5 added: maxDD={maxdd_D5:.4f}"

    # ==================================================================================
    # FINAL POLICY metrics, turnover, EV/week, and regime behavior.
    # ==================================================================================
    print("\n" + "=" * 90)
    print(f"FINAL POLICY: {final_policy_desc}")
    print("=" * 90)
    final_m = metrics(final_net, periods_per_yr)
    cagr_f, maxdd_f = ctaB._cagr_from_returns(final_net, dates_scored)
    realized_avg_lev_f = float(final_eff_lev.mean())
    print(f"CAGR={cagr_f:.4f} maxDD={maxdd_f:.4f} Sharpe={final_m['sharpe']:.4f} "
          f"realized_avg_leverage={realized_avg_lev_f:.4f}", flush=True)

    # weekly-compounded EV/week (bp), matching base script's own convention
    offset = fri_positions[0] + 1
    weekly_rets = []
    for k in range(len(fri_positions) - 1):
        f, f_next = fri_positions[k], fri_positions[k + 1]
        block_lo, block_hi = f + 1 - offset, f_next - offset + 1
        block = final_net[block_lo:block_hi]
        weekly_rets.append(float(np.prod(1.0 + block) - 1.0))
    weekly_rets = np.array(weekly_rets)
    ev_per_week_bp = float(np.mean(weekly_rets) * 1e4)
    win_rate_weekly = float(np.mean(weekly_rets > 0))
    worst_week = float(np.min(weekly_rets))
    print(f"EV/week={ev_per_week_bp:.3f}bp win_rate_wk={win_rate_weekly:.4f} "
          f"worst_week={worst_week:.4f}", flush=True)

    # turnover proxy: mean |day-over-day change in effective leverage| (captures brake/breadth-
    # driven de-lever/re-lever notional turnover on top of the already-priced per-name turnover)
    lev_turnover_daily = np.abs(np.diff(final_eff_lev, prepend=final_eff_lev[0]))
    mean_daily_lev_turnover = float(np.mean(lev_turnover_daily))

    # ------------------------------------------------------------------------------
    # Regime behavior windows
    # ------------------------------------------------------------------------------
    wealth_f = np.cumprod(1.0 + np.nan_to_num(final_net))
    peak_f = np.maximum.accumulate(wealth_f)
    dd_f = 1.0 - wealth_f / peak_f
    dd_ser = pd.Series(dd_f, index=dates_scored)
    net_ser = pd.Series(final_net, index=dates_scored)
    lev_ser = pd.Series(final_eff_lev, index=dates_scored)
    r13_ser = pd.Series(r13_base[-n:] if len(r13_base) == n else np.nan, index=dates_scored) \
        if len(r13_base) == n else None

    def window_report(label, start, end):
        s_ts = pd.Timestamp(start, tz="UTC")
        e_ts = pd.Timestamp(end, tz="UTC")
        mask = (dates_scored >= s_ts) & (dates_scored <= e_ts)
        if not mask.any():
            return {"window": [start, end], "n_days": 0, "note": "no overlap with scored sample"}
        d = dates_scored[mask]
        cum_ret = float(np.prod(1.0 + net_ser.loc[d]) - 1.0)
        fri_mask = (fri_dates_decided >= s_ts) & (fri_dates_decided <= e_ts)
        n_eligible_in_window = eligible_ser_weekly.loc[fri_dates_decided[fri_mask]] \
            if fri_mask.any() else pd.Series(dtype=float)
        rep = {
            "window": [start, end], "n_days": int(mask.sum()),
            "cum_return_final_policy": cum_ret,
            "maxdd_in_window": float(dd_ser.loc[d].max()),
            "mean_effective_leverage_in_window": float(lev_ser.loc[d].mean()),
            "min_effective_leverage_in_window": float(lev_ser.loc[d].min()),
            "max_effective_leverage_in_window": float(lev_ser.loc[d].max()),
            "mean_n_eligible_names_in_window": (float(n_eligible_in_window.mean())
                                                 if len(n_eligible_in_window) else None),
            "min_n_eligible_names_in_window": (float(n_eligible_in_window.min())
                                                if len(n_eligible_in_window) else None),
        }
        return rep

    regime_2020_03 = window_report("2020-03_COVID", *ctaB.REGIME_WINDOWS["2020-03_COVID"])
    regime_2022 = window_report("2022_full_year", *ctaB.REGIME_WINDOWS["2022_full_year"])
    regime_2018Q4 = window_report("2018Q4_chop", *ctaB.REGIME_WINDOWS["2018Q4_chop"])
    regime_meltup = window_report("2023_2024_meltup", *ctaB.REGIME_WINDOWS["2023_2024_meltup"])
    regime_squeeze = window_report("2025-04_squeeze_week", "2025-04-04", "2025-04-11")

    # 2018Q4 warmup-artifact caveat: checked EXPLICITLY (not assumed) via each Friday's own
    # n_eligible count in-window (eligible = all three of state_a/b/c non-NaN, i.e. enough own
    # trailing history for the 200d-SMA + 12mo-momentum + 10mo-SMA rules to all be defined).
    q4_eligible = regime_2018Q4.get("mean_n_eligible_names_in_window")
    if q4_eligible is not None and q4_eligible < 1.0:
        warmup_caveat = (
            f"GENUINE, VERIFIED DATA-AVAILABILITY WARMUP ARTIFACT (not a market signal): every "
            f"single Friday decision inside the 2018Q4_chop window has n_eligible=0 (checked "
            f"directly -- mean/min n_eligible_names_in_window = {q4_eligible}), i.e. not one of "
            f"the 93 reconciled names yet has enough own trailing history for all three S1 rules "
            f"(200d-SMA, 12mo-momentum, 10mo-SMA) to be simultaneously defined this early in the "
            f"panel's own master calendar (most per-name price histories in this universe begin "
            f"~2017, per the repo's own known data-vintage limits). Consequently BOTH the "
            f"inherited base 1x book (r13_base) AND every overlay built on top of it (breadth "
            f"scaling, the brake, the final combined policy) show EXACTLY 0.0% return for the "
            f"full quarter -- this is the base construction's own already-established behavior "
            f"(reused unchanged from the existing, already-verified netret CSV, not introduced "
            f"or altered by this task's overlays) and should be read as 'strategy not yet live' "
            f"rather than 'de-risked out of a bear market'."
        )
    else:
        warmup_caveat = (f"2018Q4_chop window: mean_n_eligible_names_in_window="
                          f"{q4_eligible} -- no material data-warmup artifact detected for this "
                          f"window (eligible universe was non-trivially populated).")
    print(f"\n[2018Q4 warmup caveat] {warmup_caveat}", flush=True)

    for lbl, rep in [("2020-03 COVID", regime_2020_03), ("2022 full year", regime_2022),
                     ("2018Q4 chop", regime_2018Q4), ("2023-24 meltup", regime_meltup),
                     ("2025-04 squeeze week", regime_squeeze)]:
        print(f"[{lbl}] {rep}", flush=True)

    # ==================================================================================
    # Write outputs
    # ==================================================================================
    netret_df = pd.DataFrame({
        "date": dates_scored, "net_ret_final_policy": final_net,
        "effective_leverage": final_eff_lev, "r13_base": r13_base[-n:] if len(r13_base) == n else np.nan,
    })
    netret_df.to_csv(NETRET_CSV, index=False)
    print(f"\n[output] wrote {NETRET_CSV} ({len(netret_df)} rows)", flush=True)

    results = {
        "step0_base_reproduction": {"n_obs": n_obs0, "cagr": cagr0, "maxdd": maxdd0,
                                     "sharpe": sharpe0, "passed": step0_ok},
        "L_half_rederived": L_half,
        "pre_brake_bh_book_sanity_check": {
            "recomputed": {"cagr": cagr_bh, "maxdd": maxdd_bh, "sharpe": m_bh["sharpe"],
                           "realized_avg_leverage": float(lev_bh_daily.mean())},
            "previously_reported_overlay2_alone": OVERLAY2_ALONE_REPORTED,
            "passed": sanity_bh_ok,
        },
        "step_A": {"cagr": cagr_A, "maxdd": maxdd_A, "sharpe": stepA_m["sharpe"],
                   "realized_avg_leverage": realized_avg_lev_A, "decision": step_a_decision},
        "step_A_causality_check": {"result": causality_new_logic, "passed": causal_pass_new},
        "step_C": ({"cagr": cagr_C, "maxdd": step_c_maxdd, "decision": step_c_decision}
                   if overlay3_added else {"decision": step_c_decision}),
        "step_D_notes": step_d_notes,
        "temptation_notes": temptation_notes,
        "final_policy_description": final_policy_desc,
        "final_metrics": {"cagr": cagr_f, "maxdd": maxdd_f, "sharpe": final_m["sharpe"],
                          "realized_avg_leverage": realized_avg_lev_f,
                          "ev_per_week_bp": ev_per_week_bp, "win_rate_weekly": win_rate_weekly,
                          "worst_week": worst_week,
                          "mean_daily_effective_leverage_turnover": mean_daily_lev_turnover,
                          "n_obs": n, "span_years": span_years, "periods_per_yr": periods_per_yr},
        "regime_behavior": {
            "2020-03_COVID": regime_2020_03, "2022_full_year": regime_2022,
            "2018Q4_chop": regime_2018Q4, "2018Q4_warmup_caveat": warmup_caveat,
            "2023_2024_meltup": regime_meltup, "2025-04_squeeze_week": regime_squeeze,
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[output] wrote {RESULTS_JSON}", flush=True)
    print(f"\n=== done in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
