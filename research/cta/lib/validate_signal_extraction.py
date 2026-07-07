"""Ad-hoc validation driver for research/cta/lib/signal_extraction.py -- NOT the Stage-A sweep
(no __main__ sweep loop is implemented here per the task instructions; this is purely the
validation/injection-test harness the task asked the axis owner to run before reporting the
module as validated). Run with:

  PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
    /home/speculation/convexity/.venv/bin/python research/cta/lib/validate_signal_extraction.py
"""
import sys
import time
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research/cta/lib"))

import numpy as np
import pandas as pd
import signal_extraction as se

# NOTE: research/indices/i1_pipeline.py hardcodes REPO = Path("/home/speculation/convexity-indices")
# (a different worktree that does not exist on this host -- only convexity/ and convexity-cta/ do),
# so its load_stitched_duka/DATA_STITCHED cannot be imported directly from this worktree. The two
# functions below reproduce i1_pipeline.py's exact documented convention (source=='duka' filter
# BEFORE anything else, per bug pattern (a); last real trade of each calendar day, per
# daily_close_series_duka) pointed at THIS worktree's own (symlinked) data/stitched -- verified
# byte-for-byte equivalent in logic to i1_pipeline.py's functions of the same name.
DATA_STITCHED = REPO / "data/stitched"

SYMBOLS = ["SPYUSDT", "QQQUSDT", "MSFTUSDT", "JPMUSDT", "COSTUSDT"]


def load_stitched_duka(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df = df[df["source"] == "duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    last_idx = d.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


def build_close(symbol: str) -> pd.Series:
    df = load_stitched_duka(symbol)
    assert bool((df["source"] == "duka").all()), f"{symbol}: non-duka rows leaked"
    close = daily_close_series_duka(df)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()


def main():
    print("=" * 90)
    print("PART 1: real-data smoke test + component sanity on 5 names (SPY, QQQ, + 3 equities)")
    print("=" * 90)
    for sym in SYMBOLS:
        t0 = time.time()
        close = build_close(sym)
        log_price = np.log(close)
        print(f"\n--- {sym}: {len(close)} daily obs, {close.index.min().date()} -> {close.index.max().date()}")

        # exercise every component individually
        cross = se.ewma_price_crossover(log_price, 8, 32)
        cross_identity = se.crossover_as_return_ewma_difference(log_price, 8, 32)
        max_abs_diff = float((cross - cross_identity).abs().max())
        print(f"  identity check |crossover - return_ewma_diff|.max() = {max_abs_diff:.3e} "
              f"(should be ~0 to float precision)")
        assert max_abs_diff < 1e-8, "crossover<->return-EWMA identity failed on real data"

        kf = se.kalman_local_linear_trend(log_price, **se.DEFAULT_KALMAN_PARAMS)
        ss = se.steady_state_kalman_gain(**se.DEFAULT_KALMAN_PARAMS)
        holt = se.holt_parameters_from_kalman_gain(ss["K_level"], ss["K_slope"])
        last_gain = (kf["K_level"].iloc[-1], kf["K_slope"].iloc[-1])
        print(f"  Kalman steady-state gain: K_level={ss['K_level']:.6f} K_slope={ss['K_slope']:.6f} "
              f"(converged={ss['converged']} in {ss['n_iter']} iters); "
              f"filter's own last-day gain={last_gain[0]:.6f},{last_gain[1]:.6f} (should match closely)")
        print(f"  <-> Holt linear smoothing: alpha={holt['alpha']:.6f} beta={holt['beta']:.6f}")
        assert abs(last_gain[0] - ss["K_level"]) < 1e-4

        l1_level = se.l1_trend_filter_rolling(log_price, window=120, lam=3.0)
        n_valid = int(l1_level.notna().sum())
        print(f"  l1 trend filter: {n_valid}/{len(l1_level)} valid days, "
              f"last level={l1_level.iloc[-1]:.4f} vs last log_price={log_price.iloc[-1]:.4f}")

        ensemble = se.multi_horizon_trend_signal(log_price)
        n_valid_combined = int(ensemble["combined"].notna().sum())
        print(f"  ensemble 'combined': {n_valid_combined}/{len(ensemble)} valid days, "
              f"last 5 values = {ensemble['combined'].tail(5).round(4).tolist()}")
        print(f"  elapsed: {time.time()-t0:.1f}s")

    print("\n" + "=" * 90)
    print("PART 2: synthetic-outlier-injection causality test (SPY, real data)")
    print("=" * 90)
    close = build_close("SPYUSDT")
    log_price = np.log(close)
    result = se.synthetic_outlier_injection_test(log_price, shock_size=0.5)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 90)
    print("PART 3: same injection test on a 2nd name (QQQ) for robustness")
    print("=" * 90)
    close2 = build_close("QQQUSDT")
    log_price2 = np.log(close2)
    result2 = se.synthetic_outlier_injection_test(log_price2, shock_size=-0.4)
    for k, v in result2.items():
        print(f"  {k}: {v}")

    print("\nOVERALL:", "PASS" if (result["overall_pass"] and result2["overall_pass"]) else "FAIL")


if __name__ == "__main__":
    main()
