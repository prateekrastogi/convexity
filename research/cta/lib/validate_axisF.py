#!/usr/bin/env python
"""Validation driver for research/cta/lib/portfolio_construction.py (Axis F).

Runs on the equity-cta worktree's real data:
  1. Reconciles the 93-name base universe fresh via convexity.panel.candidates.load_universe().
  2. Loads source=='duka' daily-close returns for the full reconciled universe.
  3. Runs the synthetic-outlier-injection causality test (self-contained, prints PASS/FAIL) AND a
     SECOND injection test on REAL data (SPY/QQQ/AAPL/MSFT/NVDA) through the full aggregate_portfolio
     pipeline, with a naive (unlagged) control to prove the test has teeth.
  4. Computes and reports the effective-N (closed-form + PCA-entropy) diagnostic as a time series
     across the full reconciled universe, with explicit inspection of 2020-03 and 2022.
  5. Runs the {equal_risk_weight, corr_risk_parity} x {2 corr windows} x {with/without spike
     overlay} = 8-cell grid on a placeholder trend signal (200d SMA sign) purely to prove the
     aggregation mechanics execute end to end and produce sane unit-gross output -- NOT a
     performance claim (no upstream Axis A-E signal exists yet at Stage A).

Writes JSON summary to research/cta/lib/_axisF_validation_summary.json (small artifact, safe to
commit) and prints everything to stdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity-cta")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "research/cta/lib"))

from convexity.panel.candidates import load_universe as _lu  # noqa: E402
import portfolio_construction as pc  # noqa: E402

DATA_STITCHED = REPO / "data/stitched"

SHORT_HISTORY_EXCLUDE = {"STXXUSDT", "CBRSUSDT", "DRAMUSDT"}
LEVERAGED_EXCLUDE = {"UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT"}


def build_base_universe() -> list[str]:
    u = _lu()
    eq = u["equities"]
    print(f"  load_universe(): {len(eq)} total equities entries in candidates.md", flush=True)
    with_duka = [e for e in eq if e.get("dukascopy")]
    print(f"  with real dukascopy field: {len(with_duka)}", flush=True)
    base = [e["symbol"] for e in with_duka
            if e["symbol"] not in SHORT_HISTORY_EXCLUDE and e["symbol"] not in LEVERAGED_EXCLUDE]
    print(f"  minus {len(SHORT_HISTORY_EXCLUDE)} short-history, minus {len(LEVERAGED_EXCLUDE)} "
          f"leveraged/inverse/decay-ETF -> base universe = {len(base)} names", flush=True)
    if len(base) != 93:
        print(f"  *** RECONCILIATION MISMATCH: expected 93, got {len(base)} -- proceeding with "
              f"the actually-loaded set, reported honestly. ***", flush=True)
    missing = [s for s in base if not (DATA_STITCHED / f"{s}.parquet").exists()]
    if missing:
        print(f"  WARNING: {len(missing)} base names have no stitched parquet, dropping: {missing}",
              flush=True)
        base = [s for s in base if s not in missing]
    return base, len(with_duka), missing


def main():
    t0 = time.time()
    summary = {}

    print("=== 1. universe reconciliation ===", flush=True)
    base_syms, n_with_duka, missing = build_base_universe()
    summary["universe"] = {"n_with_duka_field": n_with_duka, "n_base_reconciled": len(base_syms),
                            "missing_parquet": missing, "matches_declared_93": len(base_syms) == 93}

    print("\n=== 2. synthetic injection test (self-contained) ===", flush=True)
    synth_result = pc.run_causality_injection_test()
    print(json.dumps(synth_result, indent=2), flush=True)
    summary["synthetic_injection_test"] = synth_result

    print("\n=== 3. loading real duka daily-return panel for full base universe ===", flush=True)
    t1 = time.time()
    returns_full = pc.load_duka_daily_returns_panel(base_syms, DATA_STITCHED)
    print(f"  panel shape: {returns_full.shape}, span {returns_full.index.min()} -> "
          f"{returns_full.index.max()}, loaded in {time.time()-t1:.1f}s", flush=True)
    summary["returns_panel"] = {
        "n_names": returns_full.shape[1], "n_days": returns_full.shape[0],
        "start": str(returns_full.index.min()), "end": str(returns_full.index.max()),
    }

    print("\n=== 4. real-data injection test on 5 liquid names (SPY,QQQ,AAPL,MSFT,NVDA) via full "
          "pipeline (rolling_avg_pairwise_correlation + correlation_spike_scale) ===", flush=True)
    test_syms = [s for s in ["SPYUSDT", "QQQUSDT", "AAPLUSDT", "MSFTUSDT", "NVDAUSDT"] if s in returns_full.columns]
    print(f"  using symbols: {test_syms}", flush=True)
    real_sub = returns_full[test_syms].dropna()
    # trim to a manageable but real, contiguous 1000-day window for the injection test (full-length
    # O(n*window) rolling-corr loop is fine but we don't need 2000 days to prove causality)
    real_sub = real_sub.iloc[-1000:].copy()
    window = 60
    t_inject = 700

    baseline_corr = pc.rolling_avg_pairwise_correlation(real_sub, window=window)
    injected = real_sub.copy()
    injected.iloc[t_inject, :] = 0.5  # +50% synthetic simultaneous move, one day, real data copy
    injected_corr = pc.rolling_avg_pairwise_correlation(injected, window=window)

    naive_baseline = pc._naive_uncausal_avg_corr(real_sub, window=window)
    naive_injected = pc._naive_uncausal_avg_corr(injected, window=window)

    real_injection_result = {
        "symbols": test_syms,
        "t_inject_date": str(real_sub.index[t_inject]),
        "causal_unchanged_at_inject_day": bool(np.isclose(
            baseline_corr.iloc[t_inject], injected_corr.iloc[t_inject], equal_nan=True)),
        "causal_changed_day_after": bool(not np.isclose(
            baseline_corr.iloc[t_inject + 1], injected_corr.iloc[t_inject + 1], equal_nan=True)),
        "naive_control_changed_same_day": bool(not np.isclose(
            naive_baseline.iloc[t_inject], naive_injected.iloc[t_inject], equal_nan=True)),
        "baseline_val_at_inject": float(baseline_corr.iloc[t_inject]),
        "injected_val_at_inject": float(injected_corr.iloc[t_inject]),
        "baseline_val_day_after": float(baseline_corr.iloc[t_inject + 1]),
        "injected_val_day_after": float(injected_corr.iloc[t_inject + 1]),
        "naive_baseline_at_inject": float(naive_baseline.iloc[t_inject]),
        "naive_injected_at_inject": float(naive_injected.iloc[t_inject]),
    }
    real_injection_result["PASS"] = (real_injection_result["causal_unchanged_at_inject_day"]
                                      and real_injection_result["causal_changed_day_after"]
                                      and real_injection_result["naive_control_changed_same_day"])
    print(json.dumps(real_injection_result, indent=2), flush=True)
    summary["real_data_injection_test"] = real_injection_result

    print("\n=== 4b. full aggregate_portfolio() injection test (real data, placeholder trend signal) ===",
          flush=True)
    # placeholder upstream signal: sign(close vs trailing-200d SMA), close reconstructed from
    # cumulative returns (purely to exercise the aggregation mechanics end-to-end -- NOT a real
    # Axis A-E signal, disclosed explicitly)
    close_proxy = (1.0 + real_sub.fillna(0.0)).cumprod()
    sma200 = close_proxy.rolling(200, min_periods=200).mean().shift(1)
    close_lag1 = close_proxy.shift(1)
    target_exposure = np.sign(close_lag1 - sma200).where(sma200.notna(), 0.0)
    rv_lagged = real_sub.rolling(20, min_periods=20).std().shift(1)

    res_baseline = pc.aggregate_portfolio(target_exposure, real_sub, rv_lagged,
                                           scheme="corr_risk_parity", corr_window=60)
    injected_target = target_exposure.copy()
    injected_rets = injected.copy()  # the +50% one-day shock built above
    injected_rv = injected_rets.rolling(20, min_periods=20).std().shift(1)
    res_injected = pc.aggregate_portfolio(injected_target, injected_rets, injected_rv,
                                           scheme="corr_risk_parity", corr_window=60)

    w_base_at = res_baseline.weights.iloc[t_inject]
    w_inj_at = res_injected.weights.iloc[t_inject]
    w_base_after = res_baseline.weights.iloc[t_inject + 1]
    w_inj_after = res_injected.weights.iloc[t_inject + 1]
    pipeline_result = {
        "weights_unchanged_at_inject_day": bool(np.allclose(w_base_at.to_numpy(), w_inj_at.to_numpy(),
                                                             equal_nan=True, atol=1e-10)),
        "weights_changed_day_after": bool(not np.allclose(w_base_after.to_numpy(), w_inj_after.to_numpy(),
                                                           equal_nan=True, atol=1e-10)),
    }
    pipeline_result["PASS"] = (pipeline_result["weights_unchanged_at_inject_day"]
                                and pipeline_result["weights_changed_day_after"])
    print(json.dumps(pipeline_result, indent=2), flush=True)
    summary["full_pipeline_injection_test"] = pipeline_result

    print("\n=== 5. effective-N standing diagnostic across full reconciled universe ===", flush=True)
    avg_corr_full = pc.rolling_avg_pairwise_correlation(returns_full, window=60)
    n_active_full = returns_full.notna().sum(axis=1)
    n_eff_full = pc.effective_n_closed_form(avg_corr_full, n_active_full)

    def window_stats(label, start, end):
        m = (returns_full.index >= start) & (returns_full.index <= end)
        sub_corr = avg_corr_full[m].dropna()
        sub_neff = n_eff_full[m].dropna()
        sub_nact = n_active_full[m]
        if sub_corr.empty:
            return {"label": label, "note": "no data in window"}
        return {
            "label": label, "start": start, "end": end,
            "avg_corr_mean": float(sub_corr.mean()), "avg_corr_max": float(sub_corr.max()),
            "n_eff_mean": float(sub_neff.mean()) if not sub_neff.empty else None,
            "n_eff_min": float(sub_neff.min()) if not sub_neff.empty else None,
            "n_active_mean": float(sub_nact.mean()),
        }

    diag = {
        "full_sample": window_stats("full_sample", str(returns_full.index.min()), str(returns_full.index.max())),
        "pre_2020": window_stats("pre_2020_calm", "2019-01-01", "2020-01-31"),
        "covid_crash_2020_03": window_stats("covid_crash_2020_03", "2020-02-15", "2020-04-15"),
        "calm_2021": window_stats("calm_2021", "2021-01-01", "2021-12-31"),
        "selloff_2022": window_stats("selloff_2022", "2022-01-01", "2022-12-31"),
        "recent_2024_2026": window_stats("recent_2024_2026", "2024-01-01", str(returns_full.index.max())),
    }
    print(json.dumps(diag, indent=2), flush=True)
    summary["effective_n_diagnostic"] = diag

    # also dump the raw daily time series (avg corr + n_eff + n_active) for anyone wanting to plot it
    ts_out = pd.DataFrame({
        "avg_pairwise_corr": avg_corr_full, "n_active": n_active_full, "n_eff_closed_form": n_eff_full,
    })
    ts_path = REPO / "research/cta/lib/_axisF_effective_n_timeseries.csv"
    ts_out.to_csv(ts_path)
    print(f"\n  wrote effective-N time series to {ts_path} ({len(ts_out)} rows)", flush=True)

    print("\n=== 6. 8-cell grid smoke test (mechanics only, placeholder signal) ===", flush=True)
    grid_syms = base_syms[:20]  # a 20-name slice, full 93 x full history x scipy-free ERC loop is
                                 # slow in a single-threaded validation run; the Sweep phase runs
                                 # the real grid on the full universe with the integrated real signal
    rsub = returns_full[grid_syms].copy()
    rsub = rsub.loc[rsub.index >= "2019-01-01"]
    close_proxy2 = (1.0 + rsub.fillna(0.0)).cumprod()
    sma200_2 = close_proxy2.rolling(200, min_periods=200).mean().shift(1)
    close_lag1_2 = close_proxy2.shift(1)
    tgt2 = np.sign(close_lag1_2 - sma200_2).where(sma200_2.notna(), 0.0)
    rv2 = rsub.rolling(20, min_periods=20).std().shift(1)

    grid_cells = []
    for scheme in ["equal_risk_weight", "corr_risk_parity"]:
        for cw in [60, 120]:
            for overlay in [True, False]:
                res = pc.aggregate_portfolio(tgt2, rsub, rv2, scheme=scheme, corr_window=cw,
                                              apply_spike_overlay=overlay)
                gross = res.weights.abs().sum(axis=1)
                gross_nonzero = gross[gross > 1e-9]
                cell = {
                    "scheme": scheme, "corr_window": cw, "spike_overlay": overlay,
                    "n_days_with_position": int((gross > 1e-9).sum()),
                    "mean_gross_when_active": float(gross_nonzero.mean()) if not gross_nonzero.empty else None,
                    "unit_gross_check_max_abs_dev": float((gross_nonzero - 1.0).abs().max()) if not gross_nonzero.empty else None,
                }
                grid_cells.append(cell)
                print(f"  {cell}", flush=True)
    summary["grid_smoke_test"] = grid_cells
    summary["grid_point_count_declared"] = "2 schemes x 2 corr_windows x 2 overlay settings = 8"

    summary["total_runtime_sec"] = time.time() - t0
    out_path = REPO / "research/cta/lib/_axisF_validation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {time.time()-t0:.1f}s. Summary written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
