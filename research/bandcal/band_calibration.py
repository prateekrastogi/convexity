"""OFFLINE CALIBRATION of the no-trade band constant ``band_c`` (EXECUTION.md §4.2).

WHY THIS EXISTS
---------------
``ExecutionDefaults`` makes ``band_c`` a REQUIRED field with no default — the spec says c is
"fitted once offline to sigma_w" and "refusing to invent a number is the contract". But
``policy.build_plan`` supplies ``band_c: float = 0.5`` as a Python default, inventing exactly the
number the config layer refused to invent. At c=0.5 the band half-width is

    b = c * (theta/rho)^(1/3),  theta = 2*(taker + half_spread)
      = 0.5 * (0.0016)^(1/3) = 5.85% of NAV   (equity/commodity)
      = 0.5 * (0.0012)^(1/3) = 5.31% of NAV   (crypto)

against a median |w| of 1.68% (prod equity) and 1.41% (crypto) — i.e. the band exceeds the ENTIRE
position of 100% of names in both books. Only ``REASON_FLIP_FULL`` trades (open-from-flat,
close-to-flat, sign change) get through; every resize of a held position is frozen forever. This
is the mechanism behind the stable ~640 USDT demo-crypto position_drift.

THIS SCRIPT
-----------
Reproduces ``equity_trend.backtest_book`` exactly, but inserts the live band rule at each weekly
decision so the carried book ratchets the way production does. Sweeps c and reports the
cost/tracking frontier.

FAITHFULNESS GATE: at c=0 the band is inert, so the harness MUST reproduce the D1-validated
reference (CAGR 28.2642%, maxDD 30.0234%, Sharpe 1.16757). If it does not, every other row is
meaningless and the script says so and exits non-zero.

Read-only: touches no journal, places no orders, writes only its own results file.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

from convexity.panel.candidates import load_universe
from convexity.signal.equity_trend import (
    EQUITY_BRAKE_CEILING,
    EQUITY_ETF_EXCLUSIONS,
    EQUITY_L_HALF,
    apply_costs,
    cagr_maxdd,
    derive_periods_per_yr,
    name_signal,
    run_brake_on_prescaled,
    unitgross_weight_and_breadth,
)

STITCHED = pathlib.Path("data/stitched")
OUT = pathlib.Path("research/bandcal/results_bandcal.json")

# D1-validated reference (tests/paper/test_equity_parity.py)
REF = {"cagr": 0.28264240237223026, "maxdd": 0.30023443778215164, "sharpe": 1.1675672380874091}

# Live production constants for the equity book.
TAKER = 0.0005            # policy.build_plan default
HALF_SPREAD = 3.0e-4      # cli.PLAN_HALF_SPREAD["equity"]
RHO = 1.0                 # ExecutionDefaults.rho as passed by build_plan
THETA = 2.0 * (TAKER + HALF_SPREAD)
COST_BP = 13.0            # the validated settle charge

C_GRID = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075,
          0.10, 0.15, 0.20, 0.30, 0.50]


def band_half_width(c: float) -> float:
    """execution.band_width(theta, rho, c) — reproduced locally so the sweep is self-contained."""
    return c * (THETA / RHO) ** (1.0 / 3.0)


def backtest_universe() -> list[str]:
    out = []
    for x in load_universe(include_discovered=False)["equities"]:
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
        if (ts.iloc[-1] - ts.iloc[0]).days < 100:
            continue
        out.append(sym)
    return sorted(out)


def duka_daily_closes(symbols: list[str]) -> dict[str, pd.Series]:
    closes = {}
    for sym in symbols:
        df = pd.read_parquet(STITCHED / f"{sym}.parquet", columns=["open_time", "close", "source"])
        df = df[df["source"] == "duka"]
        if len(df) == 0:
            continue
        ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        closes[sym] = df.groupby(ts.dt.floor("D").to_numpy())["close"].last().rename(sym)
        closes[sym].index = pd.DatetimeIndex(closes[sym].index)
    return closes


def apply_band(w_target: np.ndarray, w_carried: np.ndarray, band: float) -> np.ndarray:
    """The LIVE rule, execution.py:870-881, vectorised.

    flip (open-from-flat / close-to-flat / sign change) trades in FULL, band ignored (§13);
    otherwise |x| <= band skips, and a surviving trade goes only to the band EDGE (§4.3).
    """
    x = w_target - w_carried
    if band <= 0.0:
        return w_carried + x
    flip = ((w_target == 0.0) != (w_carried == 0.0)) | (w_target * w_carried < 0.0)
    trade = np.zeros_like(x)
    trade[flip] = x[flip]
    normal = (~flip) & (np.abs(x) > band)
    trade[normal] = x[normal] - np.copysign(band, x[normal])
    return w_carried + trade


def run(c: float, ctx: dict) -> dict:
    band = band_half_width(c)
    idx, fri = ctx["idx"], ctx["fri"]
    STATE_np, RVOL_np, RET = ctx["STATE"], ctx["RVOL"], ctx["RET"]
    names = ctx["names"]
    n_days, n_names = len(idx), len(names)

    weights_daily = np.zeros((n_days, n_names))
    w_carried = np.zeros(n_names)
    n_skipped = n_flipped = n_decisions = 0
    for k in range(len(fri) - 1):
        f, f_next = fri[k], fri[k + 1]
        w_unit, _na, _ne, breadth = unitgross_weight_and_breadth(STATE_np[f], RVOL_np[f])
        w_tgt = w_unit * (EQUITY_L_HALF * breadth)
        w_ach = apply_band(w_tgt, w_carried, band)

        x = w_tgt - w_carried
        flip = ((w_tgt == 0.0) != (w_carried == 0.0)) | (w_tgt * w_carried < 0.0)
        live = x != 0.0
        n_decisions += int(live.sum())
        n_flipped += int((live & flip).sum())
        n_skipped += int((live & ~flip & (np.abs(x) <= band)).sum())

        weights_daily[f + 1:f_next + 1, :] = w_ach
        w_carried = w_ach

    lo, hi = fri[0] + 1, fri[-1] + 1
    dates = idx[lo:hi]
    W = pd.DataFrame(weights_daily[lo:hi], index=dates, columns=names)
    R = RET.loc[dates]

    r_pre = apply_costs(W, R, cost_bp=COST_BP, funding_bp_week=0.0)
    braked = run_brake_on_prescaled(r_pre, W.sum(axis=1).to_numpy(), EQUITY_BRAKE_CEILING)
    ppy = derive_periods_per_yr(dates)
    cagr, maxdd = cagr_maxdd(braked["net"], dates)
    mu, sd = np.nanmean(braked["net"]), np.nanstd(braked["net"], ddof=1)

    wn = W.to_numpy()
    dw = np.diff(wn, axis=0, prepend=wn[[0]])
    dw[0, :] = wn[0, :]
    turnover_yr = float(np.abs(dw).sum() / ((dates[-1] - dates[0]).days / 365.25))

    return {
        "c": c,
        "band_pct_nav": band * 100.0,
        "cagr": float(cagr),
        "maxdd": float(maxdd),
        "sharpe": float(mu / sd * math.sqrt(ppy)) if sd > 0 else float("nan"),
        "avg_lev": float(braked["eff_lev"].mean()),
        "turnover_per_yr": turnover_yr,
        "pct_decisions_band_skipped": 100.0 * n_skipped / max(n_decisions, 1),
        "pct_decisions_flip_full": 100.0 * n_flipped / max(n_decisions, 1),
        "net": braked["net"],
    }


def main() -> int:
    print("loading validated inputs (duka-only closes, committed universe)...", flush=True)
    syms = backtest_universe()
    closes = duka_daily_closes(syms)
    print(f"  {len(syms)} names", flush=True)

    cache = {s: sig for s, c_ in closes.items() if (sig := name_signal(c_)) is not None}
    names = sorted(cache)
    idx = None
    for s in names:
        idx = cache[s]["close"].index if idx is None else idx.union(cache[s]["close"].index)
    idx = idx.sort_values()
    ctx = {
        "idx": idx, "names": names,
        "RET": pd.DataFrame({s: cache[s]["simple_ret"].reindex(idx) for s in names}),
        "STATE": pd.DataFrame({s: cache[s]["state"].reindex(idx)
                               for s in names}).ffill(limit=5).to_numpy(),
        "RVOL": pd.DataFrame({s: cache[s]["rvol20"].reindex(idx)
                              for s in names}).ffill(limit=5).to_numpy(),
        "fri": np.where(idx.dayofweek.to_numpy() == 4)[0],
    }
    print(f"  {len(names)} signal-active names, {len(ctx['fri'])} Fridays", flush=True)

    rows = []
    base_net = None
    for c in C_GRID:
        r = run(c, ctx)
        if c == 0.0:
            base_net = r["net"]
            d = max(abs(r["cagr"] - REF["cagr"]), abs(r["maxdd"] - REF["maxdd"]))
            print(f"\nFAITHFULNESS GATE (c=0 must equal the validated reference):", flush=True)
            print(f"  CAGR   {r['cagr']:.6f} vs {REF['cagr']:.6f}", flush=True)
            print(f"  maxDD  {r['maxdd']:.6f} vs {REF['maxdd']:.6f}", flush=True)
            print(f"  Sharpe {r['sharpe']:.6f} vs {REF['sharpe']:.6f}", flush=True)
            if d > 1e-3:
                print(f"  *** GATE FAILED (max diff {d:.2e}) — harness is not faithful, "
                      f"every row below would be meaningless. ***", flush=True)
                return 1
            print(f"  GATE PASSED (max diff {d:.2e})\n", flush=True)
        te = (float(np.nanstd(r["net"] - base_net, ddof=1) * math.sqrt(252.0) * 100.0)
              if base_net is not None else 0.0)
        r["te_vs_validated_pct_yr"] = te
        rows.append(r)
        print(f"  c={c:<6.3f} band={r['band_pct_nav']:5.3f}%  CAGR {r['cagr']:7.2%}  "
              f"Sharpe {r['sharpe']:5.3f}  maxDD {r['maxdd']:6.2%}  "
              f"turn {r['turnover_per_yr']:5.2f}x  skip {r['pct_decisions_band_skipped']:5.1f}%  "
              f"TE {te:5.2f}%", flush=True)

    for r in rows:
        r.pop("net", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"theta": THETA, "rho": RHO, "cost_bp": COST_BP,
                               "reference": REF, "rows": rows}, indent=2))
    print(f"\nwrote {OUT}", flush=True)

    best = max(rows, key=lambda r: r["sharpe"])
    print(f"\nBEST net Sharpe: c={best['c']} (band {best['band_pct_nav']:.3f}% of NAV), "
          f"Sharpe {best['sharpe']:.4f} vs validated {REF['sharpe']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
