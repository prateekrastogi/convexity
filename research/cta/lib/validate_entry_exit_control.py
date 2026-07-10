"""
Validation script for research/cta/lib/entry_exit_control.py (Axis B).

Run with:
  cd /home/speculation/convexity-cta && \
  PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
  /home/speculation/convexity/.venv/bin/python research/cta/lib/validate_entry_exit_control.py

Does three things:
  1. Builds a synthetic-but-realistic causal trend-strength proxy signal (a stand-in for Axis A's
     actual output, clearly NOT the real Axis A module -- used only so this module's control logic
     can be exercised end-to-end on real price data before Axis A's real signal exists) on real
     daily closes for SPY, QQQ, AAPL, MSFT, NVDA.
  2. Runs partial_adjustment_control() on each and reports summary diagnostics (band width,
     kappa, turnover) so the derived numbers can be sanity-checked against the venue's real 13bp
     cost.
  3. THE CAUSALITY PROOF: injects a large fake one-day price/signal spike at a single index into a
     copy of the real data, re-runs the pipeline, and confirms:
       (a) realized_vol()/realized_return_vol() output at the injected index is UNCHANGED
           (causal -- does not react same-day);
       (b) output at index+1 onward IS changed (the lag is exactly one period, not more);
       (c) the naive/unlagged control realized_vol_naive_UNLAGGED() DOES change at the injected
           index itself -- proving the test has real discriminating power, not just a tautology;
       (d) the full partial_adjustment_control() exposure path inherits the same one-period-lag
           property for its band/kappa columns.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity-cta")
sys.path.insert(0, str(REPO / "research" / "indices"))
sys.path.insert(0, str(REPO / "research" / "cta" / "lib"))

import i1_pipeline as i1  # noqa: E402
import entry_exit_control as eec  # noqa: E402

# i1_pipeline.py hardcodes REPO to the convexity-indices worktree it was originally written in;
# this program's data lives under the convexity-cta worktree (symlinked to the same underlying
# data/stitched lake) -- repoint before calling any of its data-loading helpers. Verified
# equivalent: convexity-cta/data/stitched is the same symlinked lake per the host-safety
# instructions' `ls -la data/` check.
i1.REPO = REPO
i1.DATA_STITCHED = REPO / "data" / "stitched"


# ======================================================================================
# 0. Synthetic-but-realistic causal trend-strength proxy (TEST-ONLY stand-in for Axis A)
# ======================================================================================
def proxy_trend_signal(close: pd.Series, fast: int = 50, slow: int = 200, vol_window: int = 60) -> pd.Series:
    """NOT Axis A's real signal -- a simple, clearly-labeled, causal MA-crossover-distance proxy
    used only to exercise this module's control logic end-to-end on real data. Decide-at-close/
    act-next-day: value at t uses close through t-1 only (both MAs and the vol-normalizer are
    shifted by one period after being computed on data through t-1)."""
    ma_fast = close.rolling(fast, min_periods=fast).mean()
    ma_slow = close.rolling(slow, min_periods=slow).mean()
    ret = np.log(close / close.shift(1))
    rv = ret.rolling(vol_window, min_periods=vol_window).std()
    raw = (ma_fast - ma_slow) / (close * rv.replace(0, np.nan) * np.sqrt(vol_window))
    sig = raw.shift(1)  # causal: value used at t reflects data through t-1 only
    return sig.clip(-1.0, 1.0)


def load_daily_close(symbol: str) -> pd.Series:
    df = i1.load_stitched_duka(symbol)
    return i1.daily_close_series_duka(df)


# ======================================================================================
# 1 & 2. Real-data run + diagnostics
# ======================================================================================
def run_on_symbol(symbol: str, verbose: bool = True) -> dict:
    close = load_daily_close(symbol)
    sig = proxy_trend_signal(close)
    out = eec.partial_adjustment_control(sig, close, cost_bps=13.0)
    n_valid = out["band"].notna().sum()
    n_trades = int(out["traded"].sum())
    turnover_per_yr = None
    span_years = (close.index[-1] - close.index[0]).days / 365.25
    sum_abs_delta = out["exposure"].diff().abs().sum()
    turnover_events_per_yr = sum_abs_delta / 2.0 / span_years
    result = {
        "symbol": symbol,
        "n_obs": len(close),
        "span_years": round(span_years, 3),
        "n_valid_control_days": int(n_valid),
        "n_trades": n_trades,
        "pct_days_traded": round(100.0 * n_trades / max(n_valid, 1), 3),
        "median_band": float(out["band"].median(skipna=True)),
        "median_kappa_daily": float(out["kappa_daily"].median(skipna=True)),
        "mean_abs_exposure": float(out["exposure"].abs().mean()),
        "turnover_events_per_yr_roundtrip_equiv": round(float(turnover_events_per_yr), 4),
    }
    if verbose:
        print(f"  [{symbol}] n_obs={result['n_obs']} span_yrs={result['span_years']} "
              f"n_trades={result['n_trades']}/{result['n_valid_control_days']} "
              f"({result['pct_days_traded']}% of valid days) "
              f"median_band={result['median_band']:.5f} median_kappa_daily={result['median_kappa_daily']:.5f} "
              f"turnover/yr(rt-equiv)={result['turnover_events_per_yr_roundtrip_equiv']}", flush=True)
    return result


# ======================================================================================
# 3. Injection test (with naive control)
# ======================================================================================
def injection_test(symbol: str) -> dict:
    close = load_daily_close(symbol)
    n = len(close)
    inject_idx = n // 2  # somewhere safely in the middle, well past any warmup window

    close_injected = close.copy()
    # a large, obviously-fake one-day spike (+50%) at a single index
    close_injected.iloc[inject_idx] = close_injected.iloc[inject_idx] * 1.5

    sig_orig = proxy_trend_signal(close)
    sig_inj = proxy_trend_signal(close_injected)

    # --- (a)/(b): realized_return_vol causal lag check ---------------------------------
    rv_orig = eec.realized_return_vol(close)
    rv_inj = eec.realized_return_vol(close_injected)
    rv_diff = (rv_orig - rv_inj).abs()

    at_injection_unchanged = bool(rv_diff.iloc[inject_idx] < 1e-12 or np.isnan(rv_diff.iloc[inject_idx]))
    after_injection_changed = bool(rv_diff.iloc[inject_idx + 1] > 1e-12)

    # --- (c): naive/unlagged control DOES react same-day (proves the test has teeth) ---
    logret_orig = np.log(close / close.shift(1))
    logret_inj = np.log(close_injected / close_injected.shift(1))
    rv_naive_orig2 = logret_orig.rolling(20, min_periods=20).std()
    rv_naive_inj2 = logret_inj.rolling(20, min_periods=20).std()
    naive_diff = (rv_naive_orig2 - rv_naive_inj2).abs()
    naive_reacts_same_day = bool(naive_diff.iloc[inject_idx] > 1e-9)

    # --- (c'): same check but on realized_vol() applied DIRECTLY to a signal-shaped series
    # (sigma_z path) -- realized_return_vol above only exercises the price-vol path via
    # log-returns; this exercises the diff()-based estimator used for sigma_z. Inject the spike
    # directly into a signal series here (rather than reusing sig_orig/sig_inj, which are
    # themselves ALREADY one-period-lagged by proxy_trend_signal's own shift(1) -- injecting into
    # `close` and reading off `sig` at the SAME index would spuriously look "unchanged" for a
    # reason unrelated to realized_vol's own causality, since the proxy signal doesn't reflect a
    # `close` move until the following index. Testing directly on a signal-shaped series isolates
    # exactly what realized_vol()/realized_vol_naive_UNLAGGED() do.)
    sig_direct_orig = sig_orig.fillna(0.0)
    sig_direct_inj = sig_direct_orig.copy()
    sig_direct_inj.iloc[inject_idx] = sig_direct_inj.iloc[inject_idx] + 0.9  # large fake spike

    sigz_causal_orig = eec.realized_vol(sig_direct_orig)
    sigz_causal_inj = eec.realized_vol(sig_direct_inj)
    sigz_causal_diff = (sigz_causal_orig - sigz_causal_inj).abs()
    sigz_causal_unchanged_at_injection = bool(
        sigz_causal_diff.iloc[inject_idx] < 1e-12 or np.isnan(sigz_causal_diff.iloc[inject_idx])
    )
    sigz_causal_changed_after = bool(sigz_causal_diff.iloc[inject_idx + 1] > 1e-12)

    sigz_naive_orig = eec.realized_vol_naive_UNLAGGED(sig_direct_orig)
    sigz_naive_inj = eec.realized_vol_naive_UNLAGGED(sig_direct_inj)
    sigz_naive_diff = (sigz_naive_orig - sigz_naive_inj).abs()
    sigz_naive_reacts_same_day = bool(sigz_naive_diff.iloc[inject_idx] > 1e-9)

    # --- (d): full pipeline (target + price both injected) causal check ----------------
    out_orig = eec.partial_adjustment_control(sig_orig, close, cost_bps=13.0)
    out_inj = eec.partial_adjustment_control(sig_inj, close_injected, cost_bps=13.0)
    band_diff = (out_orig["band"] - out_inj["band"]).abs()
    kappa_diff = (out_orig["kappa_daily"] - out_inj["kappa_daily"]).abs()

    pipeline_band_unchanged_at_injection = bool(
        band_diff.iloc[inject_idx] < 1e-12 or np.isnan(band_diff.iloc[inject_idx])
    )
    pipeline_band_changed_after = bool(band_diff.iloc[inject_idx + 1] > 1e-12)
    pipeline_kappa_unchanged_at_injection = bool(
        kappa_diff.iloc[inject_idx] < 1e-12 or np.isnan(kappa_diff.iloc[inject_idx])
    )

    result = {
        "symbol": symbol,
        "inject_idx": int(inject_idx),
        "realized_return_vol_unchanged_at_injection_day": at_injection_unchanged,
        "realized_return_vol_changed_day_after": after_injection_changed,
        "naive_unlagged_control_reacts_same_day": naive_reacts_same_day,
        "sigz_causal_unchanged_at_injection": sigz_causal_unchanged_at_injection,
        "sigz_causal_changed_after": sigz_causal_changed_after,
        "sigz_naive_reacts_same_day": sigz_naive_reacts_same_day,
        "pipeline_band_unchanged_at_injection_day": pipeline_band_unchanged_at_injection,
        "pipeline_band_changed_day_after": pipeline_band_changed_after,
        "pipeline_kappa_unchanged_at_injection_day": pipeline_kappa_unchanged_at_injection,
        "PASS": bool(
            at_injection_unchanged
            and after_injection_changed
            and naive_reacts_same_day
            and sigz_causal_unchanged_at_injection
            and sigz_causal_changed_after
            and sigz_naive_reacts_same_day
            and pipeline_band_unchanged_at_injection
            and pipeline_band_changed_after
            and pipeline_kappa_unchanged_at_injection
        ),
    }
    print(f"  [{symbol}] injection test @ idx {inject_idx}: {result}", flush=True)
    return result


if __name__ == "__main__":
    symbols = ["SPYUSDT", "QQQUSDT", "AAPLUSDT", "MSFTUSDT", "NVDAUSDT"]

    print("=== 1/2: real-data run + diagnostics ===", flush=True)
    diag_results = [run_on_symbol(s) for s in symbols]

    print("\n=== 3: causal injection test (incl. naive control) ===", flush=True)
    inj_results = [injection_test(s) for s in symbols]

    all_pass = all(r["PASS"] for r in inj_results)
    print(f"\n=== ALL INJECTION TESTS PASS: {all_pass} ===", flush=True)

    print("\n=== 4: Stage-A grid sanity (run derived_baseline vs conservative vs aggressive on SPY) ===",
          flush=True)
    close_spy = load_daily_close("SPYUSDT")
    sig_spy = proxy_trend_signal(close_spy)
    for g in eec.DEFAULT_GRID:
        out = eec.partial_adjustment_control(
            sig_spy, close_spy, cost_bps=13.0,
            cost_sensitivity=g["cost_sensitivity"], band_scale=g["band_scale"],
        )
        n_valid = out["band"].notna().sum()
        n_trades = int(out["traded"].sum())
        span_years = (close_spy.index[-1] - close_spy.index[0]).days / 365.25
        turnover = out["exposure"].diff().abs().sum() / 2.0 / span_years
        print(f"  grid[{g['name']}] zeta={g['cost_sensitivity']} eta={g['band_scale']}: "
              f"median_band={out['band'].median(skipna=True):.5f} "
              f"median_kappa={out['kappa_daily'].median(skipna=True):.5f} "
              f"n_trades={n_trades}/{n_valid} turnover/yr={turnover:.3f}", flush=True)

    if not all_pass:
        sys.exit(1)
