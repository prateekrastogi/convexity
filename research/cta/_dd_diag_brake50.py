#!/usr/bin/env python
"""
Overlay #1 ALONE, diagnostic: GZ-style linear drawdown brake at a 50% ceiling on top of
half-Kelly (L=2.325628533609451) nominal leverage applied to the S1 93-name three-rule ensemble's
own already-established, already-verified daily net-return series
(research/cta/_s1_ensemble_93name_netret.csv, net_ret_13bp column). NO other overlay is added
here -- this is diagnostic/informational only, per the owner's pre-declared waterfall (Step A of
the waterfall tests #1+#2 together; this script is a separate, standalone probe of #1 in
isolation, requested explicitly ahead of that).

REUSE, not reimplementation:
  - The base construction (universe reconciliation, 3-rule signal, inverse-vol sizing, AAPL
    split patch, weekly Friday-close decisions, 13bp costs) lives entirely upstream in
    research/cta/_s1_ensemble_93name.py and is NOT touched here -- we read its own output CSV
    directly (the actual base-construction return series), per the task instructions.
  - Kelly leverage (L_star via numerical maximization of E[log(1+L*r)] over the REAL sample,
    L_half = 0.5*L_star) is recomputed here via the shared research/cta/lib/leverage_policy.py
    functions kelly_optimal_leverage()/fractional_kelly_leverage() -- the exact same functions
    and exact same call signature (l_max=25.0, n_bootstrap=500, bootstrap_seed=42) already used
    in research/cta/_s1_kelly_leverage_analysis.py -- rather than hand-typing the constant, so
    the L=2.3256 value used below is independently re-derived, not just pasted in.
  - The brake formula itself (m_t = clip(1 - DD_{t-1}/ceiling, 0, 1), causal, on the LEVERED
    book's own running peak/drawdown) and the metrics() helper are copied VERBATIM (unchanged
    algorithm, unchanged code) from the STEP 5 `run_brake()` function of
    research/cta/_s1_kelly_leverage_analysis.py -- the one and only change made anywhere in this
    script relative to that already-established, already-verified formula is the ceiling constant
    passed in (0.50 here vs 0.35 there). This is a deliberate, minimal-diff re-run, not a
    reimplementation, so no NEW causality bug could have been introduced by rewriting the loop.
    A causality injection test (Step 6 below) is still run fresh against a naive/unlagged control
    to confirm this specific ceiling value doesn't somehow change that property (it can't,
    algebraically, but the standing convention in this repo is to verify fresh regardless of
    inheritance -- see the five carried-forward bug patterns in the task brief).
  - Regime windows ("2020-03 COVID" and "2022") are the base construction's OWN canonical
    REGIME_WINDOWS dict from research/cta/ctaB_pipeline.py (imported, not re-typed), specifically
    the "2020-03_COVID" (2020-02-15..2020-04-15) and "2022_chop_decline"/"2022_full_year"
    (2022-01-01..2022-12-31) entries.

Read-only. Writes only into research/cta/ as a new underscore-prefixed scratch file (this
script's own results JSON); does not modify or commit any existing file.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

REPO = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/research/cta")
sys.path.insert(0, f"{REPO}/research/cta/lib")
import leverage_policy as lp     # noqa: E402
import ctaB_pipeline as ctaB     # noqa: E402  (for the canonical REGIME_WINDOWS dict only)

CSV = f"{REPO}/research/cta/_s1_ensemble_93name_netret.csv"

# ======================================================================================
# STEP 0: load the base construction's OWN output series (reused, not rebuilt) and verify it
# reproduces the already-established headline 1x numbers before trusting anything downstream.
# ======================================================================================
df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
print(f"loaded {len(df)} rows from {CSV}, cols={list(df.columns)}", flush=True)

r13 = df["net_ret_13bp"].to_numpy()
n = len(r13)

# bug pattern 2: periods_per_year derived from actual observation spacing, never hardcoded
span_years = (df.index[-1] - df.index[0]).days / 365.25
ppy = n / span_years
print(f"n_obs={n} span_years={span_years:.6f} ppy_derived={ppy:.6f}", flush=True)


def metrics(net: np.ndarray, ppy: float) -> dict:
    """Copied verbatim from research/cta/_s1_kelly_leverage_analysis.py -- unchanged."""
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


base_m = metrics(r13, ppy)
print("\n" + "=" * 90)
print("STEP 0 sanity check vs already-established base-construction numbers")
print("=" * 90)
print(json.dumps(base_m, indent=2))
established = {"cagr": 0.2205, "maxdd": 0.3873, "sharpe": 1.06, "n_obs": 2046}
checks = {
    "cagr_matches": abs(base_m["cagr"] - established["cagr"]) < 5e-4,
    "maxdd_matches": abs(base_m["maxdd"] - established["maxdd"]) < 5e-4,
    "sharpe_matches": abs(base_m["sharpe"] - established["sharpe"]) < 5e-3,
    "n_obs_matches": n == established["n_obs"],
}
print(f"reproduction checks: {checks}", flush=True)
assert all(checks.values()), f"FAILED to reproduce established base numbers: {checks}"
print("==> base construction numbers reproduced exactly. Proceeding.", flush=True)

# ======================================================================================
# STEP 1: re-derive half-Kelly leverage via the shared lib (not hand-typed), same call
# signature as the already-established analysis, and confirm it lands on L=2.325628533609451.
# ======================================================================================
print("\n" + "=" * 90)
print("STEP 1: re-derive L_star / L_half via research/cta/lib/leverage_policy.py (reused, not")
print("reimplemented) -- confirm it reproduces the already-established L_half=2.325628533609451")
print("=" * 90)
kelly = lp.kelly_optimal_leverage(r13, l_max=25.0, n_bootstrap=500, bootstrap_seed=42)
L_star = kelly["L_star"]
L_half = lp.fractional_kelly_leverage(L_star, 0.5)
print(f"L_star={L_star:.9f}  L_half={L_half:.9f}  "
      f"(established: L_star~4.651216..., L_half=2.325628533609451)", flush=True)
established_L_half = 2.325628533609451
assert abs(L_half - established_L_half) < 1e-6, (
    f"L_half re-derivation mismatch: got {L_half}, expected {established_L_half}")
print("==> L_half reproduced exactly. Using this re-derived value going forward.", flush=True)

# ======================================================================================
# STEP 2: the brake itself -- copied VERBATIM from _s1_kelly_leverage_analysis.py STEP 5,
# only the dd_ceiling argument changes (0.35 there -> 0.50 here, per this task's assignment).
# ======================================================================================
print("\n" + "=" * 90)
print("STEP 2: overlay #1 ALONE -- GZ-style linear drawdown brake at a 50% ceiling on top of")
print("half-Kelly. multiplier m_t = clip(1 - DD_{t-1}/0.50, 0, 1), DD measured causally on the")
print("LEVERED book's own running peak through the close of t-1 only. No overlay-specific extra")
print("trading cost is charged (not specified in the task; the underlying 13bp cost is already")
print("embedded in r13 and scales linearly with applied exposure, per the already-established")
print("cost-linearity proof in _s1_kelly_leverage_analysis.py Step 1).")
print("=" * 90)


def run_brake(r: np.ndarray, L_target: float, dd_ceiling: float):
    """Copied verbatim from research/cta/_s1_kelly_leverage_analysis.py STEP 5 -- unchanged
    algorithm. m_t is computed from wealth[t]/peak[t], i.e. state accumulated strictly through
    the close of day t-1, then applied to r[t] -- this is what makes it causal by construction."""
    n = len(r)
    wealth = np.empty(n + 1)
    peak = np.empty(n + 1)
    exposure = np.empty(n)
    net = np.empty(n)
    wealth[0] = 1.0
    peak[0] = 1.0
    for t in range(n):
        dd_prev = 1.0 - wealth[t] / peak[t] if peak[t] > 0 else 0.0
        m_t = min(1.0, max(0.0, 1.0 - dd_prev / dd_ceiling))
        L_t = L_target * m_t
        exposure[t] = L_t
        rt = r[t] if np.isfinite(r[t]) else 0.0
        net[t] = L_t * rt
        wealth[t + 1] = wealth[t] * (1.0 + net[t])
        peak[t + 1] = max(peak[t], wealth[t + 1])
    return {"exposure": exposure, "wealth": wealth, "peak": peak, "net": net}


DD_CEILING = 0.50
brk = run_brake(r13, L_half, DD_CEILING)
brk_m = metrics(brk["net"], ppy)
realized_avg_L = float(brk["exposure"].mean())
print(f"\noverlay #1 alone (brake@50%) on half-Kelly (L_target={L_half:.6f}):")
print(json.dumps(brk_m, indent=2), flush=True)
print(f"realized average leverage (mean L_t across all {n} days) = {realized_avg_L:.6f}")
print(f"nominal half-Kelly target                                = {L_half:.6f}")
print(f"realized/nominal ratio                                   = {realized_avg_L/L_half:.4f}")

# ======================================================================================
# STEP 3: engagement frequency / episodes
# ======================================================================================
eps = 1e-9
engaged = brk["exposure"] < (L_half - eps)
n_engaged = int(engaged.sum())
print(f"\ndays with brake engaged (multiplier<1.0, any partial de-risking): {n_engaged} / {n} "
      f"({n_engaged/n*100:.2f}%)", flush=True)

episodes = []
i = 0
dates = df.index
while i < n:
    if engaged[i]:
        j = i
        while j < n and engaged[j]:
            j += 1
        episodes.append({
            "start": str(dates[i].date()), "end": str(dates[j - 1].date()), "n_days": j - i,
            "min_exposure": float(np.min(brk["exposure"][i:j])),
            "min_exposure_frac_of_target": float(np.min(brk["exposure"][i:j]) / L_half),
            "max_dd_in_episode": float(np.max(1.0 - brk["wealth"][i+1:j+1] / brk["peak"][i+1:j+1])),
        })
        i = j
    else:
        i += 1
print(f"number of distinct engagement episodes: {len(episodes)}", flush=True)
episodes_sorted = sorted(episodes, key=lambda e: -e["n_days"])
print("top 15 episodes by duration:")
for e in episodes_sorted[:15]:
    print(f"  {e['start']} -> {e['end']}  n_days={e['n_days']:4d}  "
          f"min_exposure={e['min_exposure']:.4f} ({e['min_exposure_frac_of_target']*100:.1f}% of "
          f"target)  max_dd_in_episode={e['max_dd_in_episode']*100:.2f}%")

# ======================================================================================
# STEP 4: regime-specific behavior -- 2020-03 COVID and 2022, using the base construction's OWN
# canonical REGIME_WINDOWS (imported from ctaB_pipeline, not re-typed).
# ======================================================================================
print("\n" + "=" * 90)
print("STEP 4: regime-window behavior (base construction's own canonical REGIME_WINDOWS)")
print("=" * 90)
print(f"canonical windows in use: 2020-03_COVID={ctaB.REGIME_WINDOWS['2020-03_COVID']}  "
      f"2022_chop_decline={ctaB.REGIME_WINDOWS['2022_chop_decline']}", flush=True)


def window_report(label: str, start: str, end: str) -> dict:
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    mask = np.asarray(mask)
    exp_w = brk["exposure"][mask]
    eng_w = engaged[mask]
    net_w = brk["net"][mask]
    b1x_w = r13[mask]
    unlevered_ret = float(np.prod(1.0 + b1x_w) - 1.0) if mask.sum() else float("nan")
    brake_ret = float(np.prod(1.0 + net_w) - 1.0) if mask.sum() else float("nan")
    naive_half_kelly_ret = float(np.prod(1.0 + L_half * b1x_w) - 1.0) if mask.sum() else float("nan")
    rep = {
        "window": [start, end], "n_days_in_window": int(mask.sum()),
        "days_engaged": int(eng_w.sum()),
        "pct_days_engaged": float(eng_w.sum() / mask.sum()) if mask.sum() else float("nan"),
        "min_exposure": float(exp_w.min()) if mask.sum() else float("nan"),
        "mean_exposure": float(exp_w.mean()) if mask.sum() else float("nan"),
        "target_L_half": L_half,
        "window_return_brake50": brake_ret,
        "window_return_naive_half_kelly_no_brake": naive_half_kelly_ret,
        "window_return_unlevered_1x": unlevered_ret,
        "any_engaged": bool(eng_w.any()) if mask.sum() else False,
    }
    print(f"\n{label} [{start}..{end}]: n_days={rep['n_days_in_window']}  "
          f"days_engaged={rep['days_engaged']} ({rep['pct_days_engaged']*100:.1f}%)  "
          f"min_exp={rep['min_exposure']:.4f}  mean_exp={rep['mean_exposure']:.4f} "
          f"(target={L_half:.4f})", flush=True)
    print(f"  window return: brake@50%={brake_ret*100:7.2f}%   "
          f"naive half-Kelly (no brake)={naive_half_kelly_ret*100:7.2f}%   "
          f"unlevered 1x={unlevered_ret*100:7.2f}%", flush=True)
    return rep


covid_start, covid_end = ctaB.REGIME_WINDOWS["2020-03_COVID"]
y2022_start, y2022_end = ctaB.REGIME_WINDOWS["2022_chop_decline"]
covid_report = window_report("2020-03 COVID window", covid_start, covid_end)
y2022_report = window_report("2022 window", y2022_start, y2022_end)

# ======================================================================================
# STEP 5: peak drawdown attribution -- does the 50% ceiling itself ever bind (multiplier hits
# exactly 0), or is de-risking always partial? And where does the book's own realized maxDD
# (brk_m["maxdd"]) fall relative to the 50% budget?
# ======================================================================================
print("\n" + "=" * 90)
print("STEP 5: does the 50% ceiling ever fully bind (multiplier -> 0)?")
print("=" * 90)
fully_binds = bool(np.any(brk["exposure"] <= 1e-9))
worst_dd_idx = int(np.argmax(1.0 - brk["wealth"][1:] / brk["peak"][1:]))
worst_dd_val = float(np.max(1.0 - brk["wealth"][1:] / brk["peak"][1:]))
print(f"multiplier ever fully hits 0 (complete de-risking): {fully_binds}")
print(f"book's own realized maxDD under brake@50% = {worst_dd_val*100:.3f}% "
      f"(on {dates[worst_dd_idx].date()}), vs the 50% budget -- "
      f"budget headroom used = {worst_dd_val/0.50*100:.1f}%", flush=True)

# ======================================================================================
# STEP 6: causality re-verification (fresh, per standing convention) -- naive/unlagged control
# proves the injection test has real discriminating power, then confirm the honest run_brake()
# used above does NOT exhibit the same-day leak at the 50% ceiling either.
# ======================================================================================
print("\n" + "=" * 90)
print("STEP 6: causality injection test (fresh, ceiling=0.50) with naive/unlagged control")
print("=" * 90)


def run_brake_naive_lookahead_control(r: np.ndarray, L_target: float, dd_ceiling: float):
    """DELIBERATELY WRONG control, injection-test-only: sizes day t's multiplier using
    wealth/peak AFTER applying day t's own return (same-day lookahead), mirroring the pattern
    of leverage_policy.grossman_zhou_exposure_naive_lookahead_control. Used solely to prove the
    injection test below has real discriminating power."""
    n = len(r)
    wealth = np.empty(n + 1)
    peak = np.empty(n + 1)
    exposure = np.empty(n)
    net = np.empty(n)
    wealth[0] = 1.0
    peak[0] = 1.0
    prev_L = 0.0
    for t in range(n):
        rt = r[t] if np.isfinite(r[t]) else 0.0
        peek_wealth = wealth[t] * (1.0 + prev_L * rt)
        peek_peak = max(peak[t], peek_wealth)
        dd_peek = 1.0 - peek_wealth / peek_peak if peek_peak > 0 else 0.0
        m_t = min(1.0, max(0.0, 1.0 - dd_peek / dd_ceiling))
        L_t = L_target * m_t
        exposure[t] = L_t
        net[t] = L_t * rt
        wealth[t + 1] = wealth[t] * (1.0 + net[t])
        peak[t + 1] = max(peak[t], wealth[t + 1])
        prev_L = L_t
    return {"exposure": exposure, "wealth": wealth, "peak": peak, "net": net}


rng = np.random.default_rng(20260708)
r_synth = rng.normal(loc=0.0003, scale=0.01, size=500)
inject_idx = 250
r_shocked = r_synth.copy()
r_shocked[inject_idx] -= 0.20   # inject a large fake -20% shock on day `inject_idx`

honest_base = run_brake(r_synth, 2.0, DD_CEILING)
honest_shocked = run_brake(r_shocked, 2.0, DD_CEILING)
naive_base = run_brake_naive_lookahead_control(r_synth, 2.0, DD_CEILING)
naive_shocked = run_brake_naive_lookahead_control(r_shocked, 2.0, DD_CEILING)

honest_exposure_unaffected_before_shock = bool(np.allclose(
    honest_base["exposure"][:inject_idx], honest_shocked["exposure"][:inject_idx]))
honest_exposure_on_shock_day_unaffected = bool(np.isclose(
    honest_base["exposure"][inject_idx], honest_shocked["exposure"][inject_idx]))
honest_exposure_after_shock_diverges = bool(not np.allclose(
    honest_base["exposure"][inject_idx+1:inject_idx+6],
    honest_shocked["exposure"][inject_idx+1:inject_idx+6]))

naive_exposure_on_shock_day_reacts = bool(not np.isclose(
    naive_base["exposure"][inject_idx], naive_shocked["exposure"][inject_idx]))

causality_result = {
    "honest_run_brake_unaffected_strictly_before_shock_day": honest_exposure_unaffected_before_shock,
    "honest_run_brake_unaffected_ON_shock_day_itself": honest_exposure_on_shock_day_unaffected,
    "honest_run_brake_reacts_starting_day_AFTER_shock": honest_exposure_after_shock_diverges,
    "naive_control_reacts_on_shock_day_itself (proves test has discriminating power)":
        naive_exposure_on_shock_day_reacts,
}
print(json.dumps(causality_result, indent=2), flush=True)
causal_pass = (honest_exposure_unaffected_before_shock and honest_exposure_on_shock_day_unaffected
               and honest_exposure_after_shock_diverges and naive_exposure_on_shock_day_reacts)
print(f"\n==> causality PASS (honest brake is strictly t-1-only AND the naive control "
      f"demonstrably would have leaked if it were used instead): {causal_pass}", flush=True)
assert causal_pass, "causality check failed -- do not trust results above"

# ======================================================================================
# WRITE RESULTS
# ======================================================================================
results = {
    "overlay": "1_alone_gz_brake_50pct_ceiling_on_half_kelly",
    "n_obs": n, "span_years": span_years, "ppy": ppy,
    "base_1x_reproduction_check": {"metrics": base_m, "checks": checks},
    "L_star_rederived": L_star, "L_half_rederived": L_half,
    "L_half_matches_established": bool(abs(L_half - established_L_half) < 1e-6),
    "dd_ceiling": DD_CEILING,
    "overlay_metrics": brk_m,
    "realized_avg_leverage": realized_avg_L,
    "nominal_L_half": L_half,
    "realized_over_nominal_ratio": realized_avg_L / L_half,
    "engagement": {
        "n_days_engaged": n_engaged, "pct_days_engaged": n_engaged / n,
        "n_episodes": len(episodes), "top_episodes": episodes_sorted[:15],
        "ceiling_ever_fully_binds_to_zero": fully_binds,
        "book_own_realized_maxdd": worst_dd_val,
        "book_own_realized_maxdd_date": str(dates[worst_dd_idx].date()),
        "budget_headroom_used_pct": worst_dd_val / DD_CEILING,
    },
    "regime_2020_03_covid": covid_report,
    "regime_2022": y2022_report,
    "causality_check": causality_result,
    "causality_pass": causal_pass,
}
outpath = f"{REPO}/research/cta/_dd_diag_brake50_results.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nwrote {outpath}")
print("\nDONE.")
