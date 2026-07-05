#!/usr/bin/env python
"""POST-HOC robustness for the RV pair study (NOT pre-registered; labeled as such in the report).
Triggered by the prereg's own above-ceiling scrutiny clause (0.73 > stated 0.3-0.5 ceiling):
(1) pair jackknife on the pre-named book, (2) first/second-half split, (3) turnover + gate stats.
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import swing_pipeline as sp  # noqa: E402
import durable_pipeline as dp  # noqa: E402
import rv_pipeline as rv  # noqa: E402

WK = 52.0


def main() -> None:
    curve = sp.load_curve_raw()
    cot = pd.read_parquet(Path("/home/speculation/convexity/data/cot/cot_disagg.parquet"))
    P = sp.build_panel(curve, cot)
    fridays, wret = P["fridays"], P["wret"]
    n, m = wret.shape
    g = np.log1p(np.nan_to_num(wret, nan=0.0))
    lp = np.zeros((n, m))
    lp[1:] = np.cumsum(g, axis=0)[:-1]

    books = rv.pair_books(lp, rv.ZWINS, use_beta=False, hl_gate=True)  # the pre-named construction
    out = {"jackknife": {}, "halves": {}, "gate": {}, "turnover": {}}

    def sh(net):
        return round(float(sp.per_obs_sharpe(net) * sqrt(WK)), 3)

    # full book reference
    comb = rv.pair_risk_parity(books, wret)
    net, turn, valid = dp.net_of(comb, wret)
    out["full"] = {"sharpe": sh(net), "turn_mean": round(float(np.nanmean(turn)), 3)}
    half = len(net) // 2
    out["halves"]["full_book"] = {"H1": sh(net[:half]), "H2": sh(net[half:])}

    # (1) jackknife: drop each pair
    for drop in list(books):
        sub = {k: v for k, v in books.items() if k != drop}
        c = rv.pair_risk_parity(sub, wret)
        nn, tt, _ = dp.net_of(c, wret)
        h = len(nn) // 2
        out["jackknife"][f"drop_{drop}"] = {"sharpe": sh(nn), "H1": sh(nn[:h]), "H2": sh(nn[h:])}

    # (2) per-pair gate open fraction + per-pair net (HL-gated, solo)
    for name, Wp in books.items():
        active = float((np.abs(Wp).sum(axis=1) > 0).mean())
        nn, tt, _ = dp.net_of(Wp, wret)
        out["gate"][name] = {"active_frac": round(active, 2), "solo_sharpe": sh(nn)}

    # (3) cost realism: EV vs turnover at the book level
    out["turnover"] = {"mean_gross_turn_wk": round(float(np.nanmean(turn)), 3),
                       "cost_bp_wk_at12": round(float(np.nanmean(turn)) * 12.0, 2)}

    (HERE / "results_rv_posthoc.json").write_text(json.dumps(out, indent=1))
    print("POSTHOC DONE")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
