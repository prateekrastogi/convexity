#!/usr/bin/env python
"""Supplementary check: how did the smooth HMM p_bull and the binary SMA gate behave right around
2025-04-09 (the specific tariff-pause-squeeze episode named in Established Fact (4) as the case
trailing realized vol missed)? Not part of the required 2020-03/2022 sanity check, but SPY's
sample span happens to cover this exact date, and it is the direct motivating example for this
whole axis, so worth checking."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_hmm import causal_hmm_regime_posterior, sma_binary_gate  # noqa: E402

DATA_STITCHED = Path("/home/speculation/convexity-cta/data/stitched")


def load_stitched_duka(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_STITCHED / f"{symbol}.parquet")
    df = df[df["source"] == "duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def daily_close_series_duka(df_duka: pd.DataFrame) -> pd.Series:
    d = df_duka.copy()
    d["day"] = d["ts"].dt.floor("D")
    return d.groupby("day").tail(1).set_index("day")["close"].sort_index()


def main() -> None:
    close = daily_close_series_duka(load_stitched_duka("SPYUSDT"))
    ret = np.log(close / close.shift(1)).dropna()
    post = causal_hmm_regime_posterior(ret, n_states=2, refit_every=63, min_train=252, n_restarts=5, random_state=0)
    sma = sma_binary_gate(close, window=200, band=0.02).reindex(post.index)

    center = pd.Timestamp("2025-04-09", tz="UTC")
    window = post.loc[(post.index >= center - pd.Timedelta(days=15)) & (post.index <= center + pd.Timedelta(days=10))]
    ret_window = ret.reindex(window.index)
    print(f"{'date':<12} {'ret%':>7} {'p_bull':>8} {'sma_gate':>9}")
    for d in window.index:
        print(f"{d.date()!s:<12} {ret_window.loc[d]*100:7.2f} {window.loc[d,'p_bull']:8.3f} {sma.loc[d]:9.1f}")


if __name__ == "__main__":
    main()
