#!/usr/bin/env python
"""One-off validation script for research/cta/lib/regime_hmm.py (Axis C). NOT part of the
permanent pytest suite (EM fits are too slow to run on every commit across many names/grid
cells) -- this is the manual real-data validation required by the Axis-C task brief:
  1. Causality proof (injection test + naive control) on real SPY returns.
  2. Fit + causal posterior on SPY, QQQ, AAPL, MSFT, NVDA (5 real names, >=3 required).
  3. Sanity-check comparison: smooth HMM p_bull vs binary 200d-SMA gate around the known
     2020-03 COVID crash and 2022 drawdown windows.

Run with:
  PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
    /home/speculation/convexity/.venv/bin/python research/cta/lib/_validate_regime_hmm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_hmm import (  # noqa: E402
    causal_hmm_regime_posterior, sma_binary_gate, injection_test,
)

DATA_STITCHED = Path("/home/speculation/convexity-cta/data/stitched")


def load_stitched_duka(symbol: str) -> pd.DataFrame:
    """Same convention as research/indices/i1_pipeline.py's load_stitched_duka: filter to
    source=='duka' ONLY before anything else touches the frame (bug pattern 1)."""
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df = df[df["source"] == "duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    last_idx = d.groupby("day").tail(1).set_index("day")["close"]
    return last_idx.sort_index()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def main() -> None:
    results: dict = {}

    # ---------------------------------------------------------------------------------------
    # 1. Causality proof on real SPY log returns
    # ---------------------------------------------------------------------------------------
    spy_close = daily_close_series_duka(load_stitched_duka("SPYUSDT"))
    spy_ret = log_returns(spy_close)
    print(f"SPY: {len(spy_close)} daily closes ({spy_close.index.min().date()} -> "
          f"{spy_close.index.max().date()}), {len(spy_ret)} returns", flush=True)

    t0 = time.time()
    inj_2state = injection_test(spy_ret, n_states=2, refit_every=63, min_train=252, n_restarts=3, random_state=0)
    print(f"[{time.time()-t0:.1f}s] injection_test (2-state, refit=63): {inj_2state}", flush=True)
    results["injection_test_2state"] = inj_2state

    t0 = time.time()
    inj_3state = injection_test(spy_ret, n_states=3, refit_every=252, min_train=504, n_restarts=3, random_state=0)
    print(f"[{time.time()-t0:.1f}s] injection_test (3-state, refit=252): {inj_3state}", flush=True)
    results["injection_test_3state"] = inj_3state

    assert inj_2state["PASS"], "2-state causality proof FAILED"
    assert inj_3state["PASS"], "3-state causality proof FAILED"
    print("CAUSALITY PROOF: PASS for both 2-state and 3-state configurations.", flush=True)

    # ---------------------------------------------------------------------------------------
    # 2. Fit + causal posterior on 5 real names
    # ---------------------------------------------------------------------------------------
    names = ["SPYUSDT", "QQQUSDT", "AAPLUSDT", "MSFTUSDT", "NVDAUSDT"]
    per_name = {}
    for sym in names:
        close = daily_close_series_duka(load_stitched_duka(sym))
        ret = log_returns(close)
        t0 = time.time()
        post2 = causal_hmm_regime_posterior(ret, n_states=2, refit_every=63, min_train=252, n_restarts=3, random_state=0)
        dt = time.time() - t0
        n_valid = post2["p_bull"].notna().sum()
        print(f"{sym}: {len(ret)} returns, 2-state HMM fit in {dt:.1f}s, "
              f"{n_valid} valid p_bull rows, mean p_bull={post2['p_bull'].mean():.3f}", flush=True)
        per_name[sym] = {
            "n_returns": int(len(ret)), "fit_seconds": round(dt, 2),
            "n_valid_pbull": int(n_valid), "mean_pbull": float(post2["p_bull"].mean()),
        }
    results["per_name_2state"] = per_name

    # ---------------------------------------------------------------------------------------
    # 3. Sanity check: HMM p_bull vs 200d-SMA gate around 2020-03 crash and 2022 drawdown, SPY
    # ---------------------------------------------------------------------------------------
    post_spy = causal_hmm_regime_posterior(spy_ret, n_states=2, refit_every=63, min_train=252, n_restarts=5, random_state=0)
    sma_gate = sma_binary_gate(spy_close, window=200, band=0.02)
    # align sma_gate (indexed by close date) to returns index (returns index = close date of the
    # return's END day, i.e. same calendar day convention) -- both indexed by the same daily date.
    sma_gate_aligned = sma_gate.reindex(post_spy.index)

    windows = {
        "2020_covid_crash": ("2020-02-15", "2020-04-15"),
        "2022_drawdown": ("2022-01-01", "2022-10-15"),
    }
    window_summary = {}
    for label, (start, end) in windows.items():
        mask = (post_spy.index >= pd.Timestamp(start, tz="UTC")) & (post_spy.index <= pd.Timestamp(end, tz="UTC"))
        hmm_mean = float(post_spy.loc[mask, "p_bull"].mean())
        hmm_min = float(post_spy.loc[mask, "p_bull"].min())
        sma_mean = float(sma_gate_aligned.loc[mask].mean())
        sma_flips = int((sma_gate_aligned.loc[mask].diff().abs() > 0).sum())
        window_summary[label] = {
            "n_days": int(mask.sum()), "hmm_p_bull_mean": round(hmm_mean, 3),
            "hmm_p_bull_min": round(hmm_min, 3), "sma_gate_mean": round(sma_mean, 3),
            "sma_gate_flips": sma_flips,
        }
        print(f"{label}: n_days={mask.sum()}, HMM p_bull mean={hmm_mean:.3f} min={hmm_min:.3f}, "
              f"SMA gate mean={sma_mean:.3f} flips={sma_flips}", flush=True)
    results["regime_window_sanity_check"] = window_summary

    out_path = Path(__file__).resolve().parent / "validate_regime_hmm_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
