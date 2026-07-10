#!/usr/bin/env python
"""research/cta/_dd_diag_voltarget.py -- DIAGNOSTIC ONLY (not part of the waterfall unless the
pre-declared rule calls for it): overlay #3 (Moreira-Muir style volatility targeting, canonical
EWMA lambda=0.94) tested ALONE on top of half-Kelly (L=2.3256) nominal leverage on the S1 93-name
ensemble base -- no #1 brake, no #2 breadth-scaling, no other overlay stacked in.

Reuses, does not reimplement:
  - research/cta/_s1_ensemble_93name_netret.csv   -- the already-verified base 1x daily net-return
    series (net_ret_13bp column) produced by _s1_ensemble_93name.py (bug pattern 1 -- source=='duka'
    filtering, universe reconciliation, AAPL split patch -- is already baked into that CSV; nothing
    here re-touches raw data).
  - research/cta/lib/leverage_policy.py's kelly_optimal_leverage / fractional_kelly_leverage --
    used to freshly RE-DERIVE L_star and L_half from the real sample (not hardcoded), verifying
    against the task's stated L_half=2.3256 before trusting anything downstream.
  - the same metrics()/CAGR/maxDD/Sharpe/ppy-derivation convention as
    research/cta/_s1_kelly_leverage_analysis.py (STEP 0/2/3), and the same
    causal-injection-test-with-naive-control pattern as that script's STEP 5 / brake causality
    check, adapted here to the EWMA vol-forecast + expanding target-vol construction.

CONSTRUCTION (overlay #3, canonical parameters only, per task spec):
  1. r13        = base unlevered (1x) daily net-return series (net_ret_13bp column).
  2. L_star     = fresh empirical full-Kelly optimum on r13 (maximize E[log(1+L*r)]).
  3. L_half     = 0.5 * L_star  (fractional_kelly_leverage) -- verified ~= 2.3256.
  4. r_half(t)  = L_half * r13(t)   -- the CONSTANT half-Kelly book being overlaid (no other
     overlay). This is "the book" whose own experienced volatility overlay #3 is calibrated to,
     per the task's instruction to calibrate the vol target to "what this exact book actually
     experiences" -- since this diagnostic tests #3 ALONE (not #1+#2+#3), "the book" here is
     the half-Kelly book, not a #1+#2-overlaid book that does not exist in this run.
  5. forecast_vol(t) = sqrt(EWMA variance of r_half), lambda=0.94 (RiskMetrics convention, NOT
     tuned), computed CAUSALLY: var_hat[t] is only a function of r_half[0..t-1], used to scale
     the position applied AT day t. Recursion: var_hat[t] = lambda*var_hat[t-1] + (1-lambda)*
     r_half[t-1]**2, seeded with the sample variance of the first WARMUP=20 observations of
     r_half (a short, standard EWMA-seeding convention; NOT a new tunable exposure parameter --
     it only affects the first few weeks of the ~8-year sample and decays out geometrically at
     rate 0.94^k within ~60 obs).
  6. target_daily_vol(t) = EXPANDING-window (not a fixed trailing window -- deliberately chosen
     to avoid smuggling in an extra undeclared window-length parameter beyond the one already-
     fixed EWMA lambda) realized daily std of r_half, computed CAUSALLY using r_half[0..t-1] only.
     This converges toward the book's true full-sample unconditional vol as history accumulates,
     which is the "long-window, state which" option offered by the task, made causal (a literal
     full-sample target as in the original Moreira-Muir paper would leak future information; the
     expanding-causal version is the honest analogue). Requires >= WARMUP obs before it is
     considered valid.
  7. raw_scale(t) = target_daily_vol(t) / forecast_vol(t), scale(t) = clip(raw_scale(t), 0, CAP)
     with CAP = 2.0 (a conventional, literature-typical vol-target leverage cap -- e.g. AQR/
     Moreira-Muir-style implementations commonly cap the vol-managed leverage multiplier at 2x
     the base weight to avoid extreme over-levering into abrupt post-crash vol collapses; stated
     explicitly here as a sensible bound, not fitted to this data). No floor is imposed beyond the
     natural floor of 0 (letting the overlay de-risk fully to flat during genuine vol spikes is
     exactly the intended protective behavior, not a bug).
  8. Before both forecast_vol and target_daily_vol are valid (first WARMUP obs), scale(t) = 1.0
     (i.e. behaves as plain half-Kelly with no overlay yet -- there is no other sensible default
     during the unavoidable estimation warmup).
  9. L_t = L_half * scale(t);  r_voltarget(t) = L_t * r13(t).
  10. COSTS: no incremental transaction cost is charged for the vol-target overlay's OWN exposure
      changes, for the identical reason already established and stated explicitly in
      _s1_kelly_leverage_analysis.py STEP 5 for the drawdown brake: a leverage multiplier (whether
      constant, brake-driven, or vol-target-driven) is applied via notional/margin scaling on TOP
      of the already-cost-embedded 1x book, not via incremental trading of the underlying 93-name
      positions -- so no NEW turnover is introduced by scaling the whole book's notional up or
      down. (Bug pattern 4 -- "costs scale with realized exposure changes" -- is therefore
      satisfied trivially/by-construction here: the ONLY costs present are the already-embedded
      13bp per-name costs in r13, which do scale correctly and linearly with applied leverage,
      as proven in _s1_kelly_leverage_analysis.py STEP 1; there is no second cost layer to get
      wrong.)

CAUSALITY VERIFICATION: synthetic-outlier injection test with a naive/unlagged control, exactly
per the house convention -- inject a large fake shock into r13 at an interior day, confirm (a) the
HONEST (causal) forecast_vol/target_vol/scale/L_t arrays are byte-identical to the un-injected
baseline at every day STRICTLY BEFORE the injection day, (b) the HONEST arrays DO demonstrably
react starting at (or after) the injection day (proving the test has power, not a vacuous pass),
and (c) a deliberately-buggy NAIVE control that uses r_half[0..t] (through t, not t-1) DOES react
on the injection day itself, proving the test could catch a lookahead bug if one were present.

Run: nohup /home/speculation/convexity/.venv/bin/python research/cta/_dd_diag_voltarget.py \
       > research/cta/_dd_diag_voltarget.out 2>&1 &
Writes: research/cta/_dd_diag_voltarget_results.json (NOT committed, scratch diagnostic only)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

REPO = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/research/cta/lib")
import leverage_policy as lp  # noqa: E402

CSV = f"{REPO}/research/cta/_s1_ensemble_93name_netret.csv"
OUT_JSON = f"{REPO}/research/cta/_dd_diag_voltarget_results.json"

EWMA_LAMBDA = 0.94      # RiskMetrics convention, NOT tuned
WARMUP = 20             # obs to seed EWMA var + minimum obs before expanding target-vol is "valid"
SCALE_CAP = 2.0         # conventional vol-target leverage-multiplier cap (stated, not fitted)


# ======================================================================================
# Shared metrics helper (identical convention to _s1_kelly_leverage_analysis.py::metrics)
# ======================================================================================
def metrics(net: np.ndarray, ppy: float) -> dict:
    net = np.asarray(net, dtype="float64")
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
            "final_wealth": float(final_w), "worst_day": float(np.min(net)),
            "best_day": float(np.max(net))}


# ======================================================================================
# Overlay #3 core construction -- CAUSAL version
# ======================================================================================
def compute_voltarget(r_half: np.ndarray, lam: float = EWMA_LAMBDA, warmup: int = WARMUP,
                       cap: float = SCALE_CAP) -> dict:
    """Returns forecast_vol, target_vol, raw_scale, scale (post-cap), L_t, all CAUSAL: value at
    index t is a function of r_half[0..t-1] ONLY (never r_half[t] itself)."""
    n = len(r_half)
    r = np.nan_to_num(r_half, nan=0.0)

    forecast_var = np.full(n, np.nan)
    target_var = np.full(n, np.nan)   # expanding causal variance through t-1

    # seed EWMA variance with sample variance of the first `warmup` obs (obs 0..warmup-1)
    seed_var = float(np.var(r[:warmup], ddof=1)) if warmup >= 2 else float(r[0] ** 2)
    seed_var = seed_var if (np.isfinite(seed_var) and seed_var > 0) else 1e-8

    # running sums for expanding (through t-1) mean/var of r_half
    run_sum = 0.0
    run_sumsq = 0.0

    ewma_var_prev = seed_var
    for t in range(n):
        # --- forecast_vol(t): EWMA variance using returns through t-1 ---
        if t == 0:
            forecast_var[t] = seed_var
        else:
            forecast_var[t] = lam * ewma_var_prev + (1.0 - lam) * (r[t - 1] ** 2)
        ewma_var_prev = forecast_var[t]

        # --- target_daily_vol(t): expanding realized var of r[0..t-1], causal ---
        if t >= warmup:
            m = run_sum / t
            v = run_sumsq / t - m ** 2
            target_var[t] = v if v > 0 else np.nan
        # update running sums to include r[t] (available AFTER computing day t's target, for use at t+1)
        run_sum += r[t]
        run_sumsq += r[t] ** 2

    forecast_vol = np.sqrt(np.maximum(forecast_var, 1e-300))
    target_vol = np.sqrt(target_var)  # NaN before warmup, by construction

    raw_scale = np.where(np.isfinite(target_vol), target_vol / forecast_vol, 1.0)
    scale = np.clip(raw_scale, 0.0, cap)
    scale = np.where(np.isfinite(target_vol), scale, 1.0)  # pre-warmup default: scale=1 (plain half-Kelly)

    return {"forecast_vol": forecast_vol, "target_vol": target_vol,
            "raw_scale": raw_scale, "scale": scale}


def compute_voltarget_naive_lookahead_control(r_half: np.ndarray, lam: float = EWMA_LAMBDA,
                                               warmup: int = WARMUP, cap: float = SCALE_CAP) -> dict:
    """Deliberately-buggy control: forecast_vol(t) uses r_half[0..t] (THROUGH t, including the
    same-day return) instead of through t-1. Exists ONLY to prove the injection test has real
    discriminating power (it must react on the injection day itself, unlike the honest version)."""
    n = len(r_half)
    r = np.nan_to_num(r_half, nan=0.0)
    forecast_var = np.full(n, np.nan)
    seed_var = float(np.var(r[:warmup], ddof=1)) if warmup >= 2 else float(r[0] ** 2)
    seed_var = seed_var if (np.isfinite(seed_var) and seed_var > 0) else 1e-8
    ewma_var_prev = seed_var
    for t in range(n):
        # BUG: uses r[t] (same day) instead of r[t-1]
        forecast_var[t] = lam * ewma_var_prev + (1.0 - lam) * (r[t] ** 2) if t > 0 else seed_var
        ewma_var_prev = forecast_var[t]
    forecast_vol = np.sqrt(np.maximum(forecast_var, 1e-300))
    # target_vol reuses the same (already-causal) expanding estimator for isolation of the bug to forecast_vol only
    target_var = np.full(n, np.nan)
    run_sum, run_sumsq = 0.0, 0.0
    for t in range(n):
        if t >= warmup:
            m = run_sum / t
            v = run_sumsq / t - m ** 2
            target_var[t] = v if v > 0 else np.nan
        run_sum += r[t]
        run_sumsq += r[t] ** 2
    target_vol = np.sqrt(target_var)
    raw_scale = np.where(np.isfinite(target_vol), target_vol / forecast_vol, 1.0)
    scale = np.clip(raw_scale, 0.0, cap)
    scale = np.where(np.isfinite(target_vol), scale, 1.0)
    return {"forecast_vol": forecast_vol, "target_vol": target_vol, "raw_scale": raw_scale, "scale": scale}


def main():
    t_start = time.time()
    print(f"=== _dd_diag_voltarget.py starting {pd.Timestamp.now()} ===", flush=True)
    print("DIAGNOSTIC ONLY: overlay #3 (EWMA-0.94 vol targeting) tested ALONE on top of "
          "half-Kelly. No #1/#2/#4/#5. Not part of the pre-registered waterfall unless the "
          "waterfall's own rule calls for it.\n", flush=True)

    df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
    r13 = df["net_ret_13bp"].to_numpy()
    n = len(r13)
    dates = df.index
    span_years = (dates[-1] - dates[0]).days / 365.25
    ppy = n / span_years   # bug pattern 2: derived from actual observation spacing, not hardcoded
    print(f"[data] n_obs={n} span_years={span_years:.6f} ppy_derived={ppy:.6f}", flush=True)

    # ------------------------------------------------------------------------------
    # STEP 0: reproduce the base 1x and half-Kelly numbers before trusting anything downstream
    # ------------------------------------------------------------------------------
    base_m = metrics(r13, ppy)
    print(f"\n[STEP 0] base 1x reproduction: {json.dumps(base_m, indent=2)}", flush=True)
    print("  (expect CAGR~0.2205 maxDD~0.3873 Sharpe~1.06, n_obs=2046)", flush=True)

    kelly = lp.kelly_optimal_leverage(r13, l_max=25.0, n_bootstrap=0)
    L_star = kelly["L_star"]
    L_half = lp.fractional_kelly_leverage(L_star, 0.5)
    print(f"\n[STEP 0] fresh L_star={L_star:.6f} (expect ~4.651)  L_half={L_half:.6f} "
          f"(expect ~2.3256)", flush=True)

    r_half = L_half * r13
    half_m = metrics(r_half, ppy)
    print(f"[STEP 0] half-Kelly (no overlay) reproduction: {json.dumps(half_m, indent=2)}", flush=True)
    print("  (expect CAGR~0.4839 maxDD~0.7052 Sharpe~1.06)", flush=True)

    # ------------------------------------------------------------------------------
    # STEP 1: causal injection test (with naive/unlagged control) BEFORE trusting the overlay
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 92, flush=True)
    print("STEP 1: synthetic-outlier-injection causality test (honest vs naive-lookahead control)", flush=True)
    print("=" * 92, flush=True)
    inject_pos = n // 2
    r13_bad = r13.copy()
    FAKE_MOVE = -0.20 if r13[inject_pos] >= 0 else 0.20  # shock opposite the day's own sign, so it's unambiguous
    r13_bad[inject_pos] = FAKE_MOVE
    r_half_bad = L_half * r13_bad

    honest_orig = compute_voltarget(r_half)
    honest_inj = compute_voltarget(r_half_bad)
    naive_orig = compute_voltarget_naive_lookahead_control(r_half)
    naive_inj = compute_voltarget_naive_lookahead_control(r_half_bad)

    def _diff_mask(a, b):
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.nan_to_num(a, nan=-999.0) - np.nan_to_num(b, nan=-999.0))
        return d

    honest_scale_diff = _diff_mask(honest_orig["scale"], honest_inj["scale"])
    naive_scale_diff = _diff_mask(naive_orig["scale"], naive_inj["scale"])

    honest_before_unaffected = bool(np.allclose(honest_scale_diff[:inject_pos], 0.0, atol=1e-12))
    honest_reacts_at_inject_day = not np.isclose(honest_scale_diff[inject_pos], 0.0, atol=1e-12)
    honest_reacts_next_day = not np.isclose(honest_scale_diff[inject_pos + 1], 0.0, atol=1e-12)
    naive_reacts_at_inject_day = not np.isclose(naive_scale_diff[inject_pos], 0.0, atol=1e-12)

    causal_pass = honest_before_unaffected and (not honest_reacts_at_inject_day) and honest_reacts_next_day
    control_has_teeth = naive_reacts_at_inject_day

    print(f"  inject_pos={inject_pos} ({dates[inject_pos].date()}), fake move={FAKE_MOVE:+.0%}", flush=True)
    print(f"  HONEST (causal): unaffected before inject day = {honest_before_unaffected} (want True)", flush=True)
    print(f"  HONEST (causal): reacts AT inject day t        = {honest_reacts_at_inject_day} (want False)", flush=True)
    print(f"  HONEST (causal): reacts AT t+1                 = {honest_reacts_next_day} (want True)", flush=True)
    print(f"  NAIVE (buggy control): reacts AT inject day t  = {naive_reacts_at_inject_day} "
          f"(want True -- proves test has teeth)", flush=True)
    print(f"  ==> CAUSALITY TEST PASS: {causal_pass}", flush=True)
    print(f"  ==> naive control has real discriminating power: {control_has_teeth}", flush=True)

    if not (causal_pass and control_has_teeth):
        print("  *** CAUSALITY TEST FAILED -- ABORTING, downstream numbers would be untrustworthy ***",
              flush=True)
        sys.exit(1)

    # ------------------------------------------------------------------------------
    # STEP 2: apply overlay #3 to the REAL half-Kelly book
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 92, flush=True)
    print("STEP 2: apply overlay #3 (EWMA-0.94 vol targeting, cap=2.0x) to the real half-Kelly book", flush=True)
    print("=" * 92, flush=True)
    vt = compute_voltarget(r_half)
    L_t = L_half * vt["scale"]
    r_vt = L_t * r13   # equivalent to vt["scale"] * r_half

    vt_m = metrics(r_vt, ppy)
    realized_avg_L = float(np.mean(L_t))
    print(f"  overlay #3 metrics: {json.dumps(vt_m, indent=2)}", flush=True)
    print(f"  realized avg leverage = {realized_avg_L:.6f}  (nominal half-Kelly target = {L_half:.6f})", flush=True)

    # ------------------------------------------------------------------------------
    # STEP 3: scale-cap diagnostics
    # ------------------------------------------------------------------------------
    valid = np.isfinite(vt["target_vol"])  # post-warmup only
    n_valid = int(valid.sum())
    n_cap_bind = int(np.sum(valid & (vt["raw_scale"] > SCALE_CAP - 1e-9)))
    pct_cap_bind = n_cap_bind / n_valid * 100 if n_valid else float("nan")
    raw_scale_valid = vt["raw_scale"][valid]
    scale_valid = vt["scale"][valid]
    print(f"\n[STEP 3] scale-cap diagnostics (post-warmup n={n_valid}):", flush=True)
    print(f"  cap binds (raw_scale > {SCALE_CAP}) on {n_cap_bind}/{n_valid} days ({pct_cap_bind:.2f}%)", flush=True)
    print(f"  raw_scale distribution: min={raw_scale_valid.min():.4f} p5={np.percentile(raw_scale_valid,5):.4f} "
          f"median={np.median(raw_scale_valid):.4f} p95={np.percentile(raw_scale_valid,95):.4f} "
          f"max={raw_scale_valid.max():.4f}", flush=True)
    print(f"  applied-scale distribution (post-cap): min={scale_valid.min():.4f} "
          f"median={np.median(scale_valid):.4f} max={scale_valid.max():.4f}", flush=True)
    over_cap_amounts = raw_scale_valid[raw_scale_valid > SCALE_CAP] - SCALE_CAP
    mean_excess_over_cap = float(over_cap_amounts.mean()) if len(over_cap_amounts) else 0.0
    max_excess_over_cap = float(over_cap_amounts.max()) if len(over_cap_amounts) else 0.0
    print(f"  when cap binds: mean excess over cap={mean_excess_over_cap:.4f}x  "
          f"max excess over cap={max_excess_over_cap:.4f}x", flush=True)

    # ------------------------------------------------------------------------------
    # STEP 4: behavior during 2020-03 (COVID) and 2022
    # ------------------------------------------------------------------------------
    def window_report(start, end, label):
        mask = ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
        mask = np.asarray(mask)
        if mask.sum() == 0:
            print(f"\n[{label}] no days in window [{start}..{end}]", flush=True)
            return None
        L_win = L_t[mask]
        scale_win = vt["scale"][mask]
        fvol_win = vt["forecast_vol"][mask]
        tvol_win = vt["target_vol"][mask]
        r_vt_win = r_vt[mask]
        cum_ret = float(np.prod(1.0 + r_vt_win) - 1.0)
        cum_ret_half_only = float(np.prod(1.0 + r_half[mask]) - 1.0)
        print(f"\n[{label}] [{start}..{end}]: n_days={mask.sum()}", flush=True)
        print(f"  L_t: min={L_win.min():.4f} mean={L_win.mean():.4f} max={L_win.max():.4f} "
              f"(nominal target={L_half:.4f})", flush=True)
        print(f"  scale: min={scale_win.min():.4f} mean={scale_win.mean():.4f} max={scale_win.max():.4f}", flush=True)
        print(f"  forecast_vol: min={np.nanmin(fvol_win):.5f} mean={np.nanmean(fvol_win):.5f} "
              f"max={np.nanmax(fvol_win):.5f}", flush=True)
        print(f"  target_vol (expanding, causal): mean={np.nanmean(tvol_win):.5f}", flush=True)
        print(f"  cumulative return in window: overlay#3={cum_ret:+.4%}  half-Kelly-only={cum_ret_half_only:+.4%}",
              flush=True)
        return {"n_days": int(mask.sum()), "L_t_min": float(L_win.min()), "L_t_mean": float(L_win.mean()),
                "L_t_max": float(L_win.max()), "scale_min": float(scale_win.min()),
                "scale_mean": float(scale_win.mean()), "scale_max": float(scale_win.max()),
                "cum_ret_overlay3": cum_ret, "cum_ret_half_kelly_only": cum_ret_half_only}

    print("\n" + "=" * 92, flush=True)
    print("STEP 4: regime behavior -- 2020-03 COVID and 2022 chop/decline", flush=True)
    print("=" * 92, flush=True)
    covid_report = window_report("2020-02-15", "2020-04-15", "2020-03 COVID crash window")
    y2022_report = window_report("2022-01-01", "2022-12-31", "2022 chop/decline window")

    # ------------------------------------------------------------------------------
    # write results
    # ------------------------------------------------------------------------------
    results = {
        "meta": {"n_obs": n, "span_years": span_years, "ppy_derived": ppy,
                 "ewma_lambda": EWMA_LAMBDA, "warmup_obs": WARMUP, "scale_cap": SCALE_CAP},
        "base_1x_reproduction": base_m,
        "L_star_fresh": L_star, "L_half_fresh": L_half,
        "half_kelly_no_overlay_reproduction": half_m,
        "causality_test": {
            "inject_pos": int(inject_pos), "inject_date": str(dates[inject_pos].date()),
            "fake_move": FAKE_MOVE,
            "honest_before_unaffected": honest_before_unaffected,
            "honest_reacts_at_inject_day": honest_reacts_at_inject_day,
            "honest_reacts_at_t_plus_1": honest_reacts_next_day,
            "naive_control_reacts_at_inject_day": naive_reacts_at_inject_day,
            "causal_pass": causal_pass, "control_has_teeth": control_has_teeth,
        },
        "overlay3_metrics": vt_m,
        "realized_avg_leverage": realized_avg_L,
        "nominal_half_kelly_target": L_half,
        "scale_cap_diagnostics": {
            "n_valid_postwarmup": n_valid, "n_cap_bind": n_cap_bind, "pct_cap_bind": pct_cap_bind,
            "raw_scale_min": float(raw_scale_valid.min()), "raw_scale_p5": float(np.percentile(raw_scale_valid, 5)),
            "raw_scale_median": float(np.median(raw_scale_valid)),
            "raw_scale_p95": float(np.percentile(raw_scale_valid, 95)),
            "raw_scale_max": float(raw_scale_valid.max()),
            "applied_scale_min": float(scale_valid.min()), "applied_scale_max": float(scale_valid.max()),
            "mean_excess_over_cap": mean_excess_over_cap, "max_excess_over_cap": max_excess_over_cap,
        },
        "regime_2020_03_covid": covid_report,
        "regime_2022": y2022_report,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[output] wrote {OUT_JSON}", flush=True)
    print(f"\n=== done in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
