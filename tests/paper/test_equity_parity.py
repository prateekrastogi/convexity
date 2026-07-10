"""PARITY TEST (PAPER.md §12.1) — THE blocking anti-bug gate for the equity book.

The production signal/sizing/brake chain (convexity.signal.equity_trend — the SAME functions the
live rebalance calls), run in backtest mode over the validated historical window, must reproduce
the D1-validated reference (tests/fixtures/equity_book/dd_combined_waterfall_netret.csv:
CAGR 28.26%, maxDD 30.02%, Sharpe 1.1676). A deployment whose backtest-mode output diverges from
the validated artifact is broken by definition.

Data note: the reference was produced from duka-only closes with AAPL's 4:1 split patched
IN MEMORY (divisor 4 before 2020-08-31). The stitched lake has since had the identical fix
applied AT SOURCE (KNOWN_SPLITS, same divisor convention) — so rebuilding duka-only closes from
today's lake reproduces the reference's inputs exactly.

Marked slow (reads ~100 stitched parquets, ~2min): included in the full suite; skipped
automatically when the stitched lake is absent (CI environments without the data volume).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from convexity.signal.equity_trend import EQUITY_ETF_EXCLUSIONS, backtest_book

STITCHED = pathlib.Path("data/stitched")
FIXTURE = pathlib.Path("tests/fixtures/equity_book/dd_combined_waterfall_netret.csv")

# The D1-validated reference numbers (independently verified bit-for-bit in the D1 program).
REF_CAGR = 0.28264240237223026
REF_MAXDD = 0.30023443778215164
REF_SHARPE = 1.1675672380874091
REF_AVG_LEV = 1.1103454095396768

# Tolerance: float-path/library-version differences only. Expected agreement ~1e-6; anything
# past 1e-3 absolute means the construction diverged and the book is NOT the validated one.
TOL = 1e-3

pytestmark = pytest.mark.skipif(
    not STITCHED.exists() or not any(STITCHED.glob("*.parquet")),
    reason="stitched lake not present (data volume required for the parity test)",
)


def _backtest_universe() -> list[str]:
    """The validated 93-name backtest universe, reconciled exactly as the D1 reference did:
    equities sleeve, duka-history present, span >= 100d, minus the 6 leveraged/inverse ETFs."""
    from convexity.panel.candidates import load_universe

    out = []
    for x in load_universe()["equities"]:
        sym = x["symbol"]
        if sym in EQUITY_ETF_EXCLUSIONS:
            continue
        fp = STITCHED / f"{sym}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["open_time", "source"])
        duka = df[df["source"] == "duka"]
        if len(duka) == 0:
            continue
        ts = pd.to_datetime(duka["open_time"], unit="ms", utc=True)
        span_days = (ts.iloc[-1] - ts.iloc[0]).days
        if span_days < 100:
            continue
        out.append(sym)
    return sorted(out)


def _duka_daily_closes(symbols: list[str]) -> dict[str, pd.Series]:
    """Duka-only daily closes, the reference's exact aggregation: UTC-day floor, last close."""
    closes = {}
    for sym in symbols:
        df = pd.read_parquet(STITCHED / f"{sym}.parquet", columns=["open_time", "close", "source"])
        df = df[df["source"] == "duka"]
        if len(df) == 0:
            continue
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        day = ts.dt.floor("D")
        closes[sym] = df.groupby(day.to_numpy())["close"].last().rename(sym)
        closes[sym].index = pd.DatetimeIndex(closes[sym].index)
    return closes


@pytest.mark.slow
def test_equity_book_parity_against_validated_reference():
    symbols = _backtest_universe()
    assert len(symbols) >= 90, f"universe reconciliation drifted: {len(symbols)} names"

    closes = _duka_daily_closes(symbols)

    # Guard: the AAPL split fix must be present at source (the parity inputs depend on it).
    aapl = closes.get("AAPLUSDT")
    if aapl is not None:
        r = aapl.pct_change()
        r_split_day = r.loc["2020-08-31"]
        r_val = float(r_split_day.iloc[0] if hasattr(r_split_day, "iloc") else r_split_day)
        assert abs(r_val) < 0.20, (
            f"AAPL 2020-08-31 return {r_val:+.2%}: the KNOWN_SPLITS source fix is missing — "
            f"re-run the stitch for AAPLUSDT before trusting any equity-book output")

    res = backtest_book(closes, cost_bp=13.0)

    assert abs(res["cagr"] - REF_CAGR) < TOL, (
        f"PARITY FAIL CAGR: production {res['cagr']:.6f} vs validated {REF_CAGR:.6f}")
    assert abs(res["maxdd"] - REF_MAXDD) < TOL, (
        f"PARITY FAIL maxDD: production {res['maxdd']:.6f} vs validated {REF_MAXDD:.6f}")
    assert abs(res["sharpe"] - REF_SHARPE) < 1e-2, (
        f"PARITY FAIL Sharpe: production {res['sharpe']:.6f} vs validated {REF_SHARPE:.6f}")
    assert abs(res["realized_avg_leverage"] - REF_AVG_LEV) < TOL, (
        f"PARITY FAIL avg leverage: {res['realized_avg_leverage']:.6f} vs {REF_AVG_LEV:.6f}")

    # Cross-check the day-level series against the pinned reference CSV where dates overlap.
    ref = pd.read_csv(FIXTURE)
    ref_net = pd.Series(
        ref["net_ret_final_policy"].to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(ref["date"], utc=True)),
    )
    prod_net = pd.Series(res["net"], index=res["dates"])
    joined = pd.concat([ref_net.rename("ref"), prod_net.rename("prod")], axis=1).dropna()
    assert len(joined) > 1500, f"insufficient overlap with reference series: {len(joined)}"
    max_daily_diff = float((joined["ref"] - joined["prod"]).abs().max())
    assert max_daily_diff < 5e-4, (
        f"PARITY FAIL day-level: max |ref−prod| daily-return diff {max_daily_diff:.2e}")


@pytest.mark.slow
def test_l_half_provenance():
    """The frozen L_HALF constant must re-derive from the pinned fixture inputs — provenance
    stays verifiable without runtime drift (PAPER.md §12.1)."""
    import importlib.util

    from convexity.paper.fleetconfig import EQUITY_L_HALF

    ref_lp = pathlib.Path("tests/fixtures/equity_book/reference_leverage_policy.py")
    spec = importlib.util.spec_from_file_location("ref_leverage_policy", ref_lp)
    lp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lp)

    base = pd.read_csv("tests/fixtures/equity_book/s1_ensemble_93name_netret.csv")
    net = base["net_ret_13bp"].to_numpy(dtype=float)
    # Point estimate only (the bootstrap CI does not move L_star; skipping it keeps the test fast).
    l_star = lp.kelly_optimal_leverage(net, l_max=25.0)["L_star"]
    l_half = lp.fractional_kelly_leverage(float(l_star), 0.5)
    assert abs(float(l_half) - EQUITY_L_HALF) < 1e-6, (
        f"L_HALF provenance broken: re-derived {l_half} vs frozen {EQUITY_L_HALF}")
