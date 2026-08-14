"""ROBUSTNESS check on the band_c sweep — is the plateau stable, or an in-sample artifact?

The full-history sweep put the net-Sharpe argmax at c=0.03. Picking an argmax on one pass is the
overfit trap this repo charges to DSR everywhere else, so before shipping a number we check:

  1. SUB-PERIOD: split the history in half; does the same region win in BOTH halves?
  2. PER-YEAR: is c=0.03 better than the shipped c=0.5 in most individual years, or is the whole
     edge one lucky stretch?
  3. The comparison that actually matters for the decision is c_new vs c=0.5 (what we run today)
     and vs c=0 (the validated construction) — not the fine ranking inside the plateau.

Reuses band_calibration's harness verbatim. Read-only.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/bandcal")

from band_calibration import (  # noqa: E402
    COST_BP,
    EQUITY_BRAKE_CEILING,
    EQUITY_L_HALF,
    REF,
    apply_band,
    apply_costs,
    backtest_universe,
    band_half_width,
    cagr_maxdd,
    derive_periods_per_yr,
    duka_daily_closes,
    name_signal,
    run_brake_on_prescaled,
    unitgross_weight_and_breadth,
)

C_TEST = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.50]


def weights_for(c: float, ctx: dict) -> pd.DataFrame:
    band = band_half_width(c)
    idx, fri, names = ctx["idx"], ctx["fri"], ctx["names"]
    wd = np.zeros((len(idx), len(names)))
    carried = np.zeros(len(names))
    for k in range(len(fri) - 1):
        f, f_next = fri[k], fri[k + 1]
        w_unit, _a, _e, breadth = unitgross_weight_and_breadth(ctx["STATE"][f], ctx["RVOL"][f])
        ach = apply_band(w_unit * (EQUITY_L_HALF * breadth), carried, band)
        wd[f + 1:f_next + 1, :] = ach
        carried = ach
    lo, hi = fri[0] + 1, fri[-1] + 1
    return pd.DataFrame(wd[lo:hi], index=idx[lo:hi], columns=names)


def metrics(W: pd.DataFrame, RET: pd.DataFrame) -> dict:
    dates = W.index
    r_pre = apply_costs(W, RET.loc[dates], cost_bp=COST_BP, funding_bp_week=0.0)
    braked = run_brake_on_prescaled(r_pre, W.sum(axis=1).to_numpy(), EQUITY_BRAKE_CEILING)
    net = braked["net"]
    ppy = derive_periods_per_yr(dates)
    cagr, maxdd = cagr_maxdd(net, dates)
    mu, sd = np.nanmean(net), np.nanstd(net, ddof=1)
    return {"cagr": float(cagr), "maxdd": float(maxdd),
            "sharpe": float(mu / sd * math.sqrt(ppy)) if sd > 0 else float("nan"),
            "net": pd.Series(net, index=dates)}


def main() -> int:
    syms = backtest_universe()
    closes = duka_daily_closes(syms)
    cache = {s: sig for s, c_ in closes.items() if (sig := name_signal(c_)) is not None}
    names = sorted(cache)
    idx = None
    for s in names:
        idx = cache[s]["close"].index if idx is None else idx.union(cache[s]["close"].index)
    idx = idx.sort_values()
    RET = pd.DataFrame({s: cache[s]["simple_ret"].reindex(idx) for s in names})
    ctx = {
        "idx": idx, "names": names, "RET": RET,
        "STATE": pd.DataFrame({s: cache[s]["state"].reindex(idx)
                               for s in names}).ffill(limit=5).to_numpy(),
        "RVOL": pd.DataFrame({s: cache[s]["rvol20"].reindex(idx)
                              for s in names}).ffill(limit=5).to_numpy(),
        "fri": np.where(idx.dayofweek.to_numpy() == 4)[0],
    }

    W = {c: weights_for(c, ctx) for c in C_TEST}
    full = {c: metrics(W[c], RET) for c in C_TEST}
    dates = W[0.0].index
    mid = dates[len(dates) // 2]
    print(f"history {dates[0].date()} -> {dates[-1].date()}, split at {mid.date()}\n")

    print("=== SUB-PERIOD net Sharpe (does the same region win in BOTH halves?) ===")
    print(f"{'c':>7s} {'band%':>7s} {'FULL':>8s} {'H1':>8s} {'H2':>8s}")
    h1, h2 = {}, {}
    for c in C_TEST:
        a = metrics(W[c].loc[:mid], RET)
        b = metrics(W[c].loc[mid:], RET)
        h1[c], h2[c] = a["sharpe"], b["sharpe"]
        print(f"{c:7.3f} {band_half_width(c) * 100:6.3f}% {full[c]['sharpe']:8.3f} "
              f"{a['sharpe']:8.3f} {b['sharpe']:8.3f}")
    print(f"\n  argmax FULL c={max(full, key=lambda k: full[k]['sharpe'])}  "
          f"H1 c={max(h1, key=h1.get)}  H2 c={max(h2, key=h2.get)}")

    print("\n=== PER-YEAR net return: candidate vs shipped c=0.5 vs validated c=0 ===")
    yrs = sorted({d.year for d in dates})
    print(f"{'year':>6s} {'c=0':>9s} {'c=0.03':>9s} {'c=0.50':>9s}   {'0.03 beats 0.50':>16s}")
    wins = 0
    for y in yrs:
        m = [d.year == y for d in dates]
        r0 = float((1 + full[0.0]["net"][m]).prod() - 1)
        r3 = float((1 + full[0.03]["net"][m]).prod() - 1)
        r5 = float((1 + full[0.50]["net"][m]).prod() - 1)
        win = r3 > r5
        wins += win
        print(f"{y:6d} {r0:8.1%} {r3:8.1%} {r5:8.1%}   {'yes' if win else 'no':>16s}")
    print(f"\n  c=0.03 beat the shipped c=0.50 in {wins}/{len(yrs)} years")

    print("\n=== headline: what changing the constant is worth ===")
    for c in (0.0, 0.02, 0.03, 0.05):
        d = full[c]
        s = full[0.50]
        print(f"  c={c:<5.2f} CAGR {d['cagr']:6.2%} Sharpe {d['sharpe']:5.3f} "
              f"maxDD {d['maxdd']:6.2%}   vs shipped: CAGR {d['cagr'] - s['cagr']:+6.2%} "
              f"Sharpe {d['sharpe'] - s['sharpe']:+5.3f} maxDD {d['maxdd'] - s['maxdd']:+6.2%}")
    print(f"\n  validated reference Sharpe {REF['sharpe']:.4f}, CAGR {REF['cagr']:.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
