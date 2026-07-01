"""Full cross-sectional sweep: momentum(L x skip) / carry / combined x quantile x weighting.

Builds the point-in-time 50-crypto daily panel once, runs the full config grid, then for the
whole trial family computes: net & gross annualized Sharpe, CAGR, maxDD, win-rate, turnover/yr,
cost-drag/yr, funding-income/yr; purged walk-forward OOS Sharpe (>=2 folds) + a final OOS holdout;
per-regime Sharpe (bull/bear/crisis buckets by UTC year windows); and the Deflated Sharpe over the
FULL grid trial count (the honest multiplicity). Writes a JSON blob + a human table to stdout / log.

Run on the VM:
  PYTHONPATH=. .venv/bin/python scripts/xsec_full.py --tag xsec_full
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from research.universe import all_symbols

from research.xsec import metrics
from research.xsec.engine import XSecConfig, XSecResult, run
from research.xsec.panel import Panel, build_panel, universe_size_by_year
from research.xsec.validate import deflated_sharpe, purged_walkforward

PERIODS_PER_YEAR = metrics.PERIODS_PER_YEAR

# regime windows (UTC calendar): the crypto macro regimes over the sample.
REGIMES: dict[str, tuple[str, str]] = {
    "2021_bull": ("2021-01-01", "2021-11-10"),
    "2022_bear": ("2021-11-11", "2022-12-31"),  # LUNA/FTX crisis inside
    "2023_recovery": ("2023-01-01", "2023-12-31"),
    "2024_bull": ("2024-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-12-31"),
}


def _gross_sharpe(res: XSecResult) -> float:
    """Annualized Sharpe of the GROSS (pre-cost) weekly series = price P&L + funding, no friction."""
    g = res.gross_price + res.funding
    if g.size < 2:
        return 0.0
    sd = np.std(g, ddof=1)
    return float(np.mean(g) / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0


def _regime_sharpe(res: XSecResult) -> dict[str, dict]:
    out: dict[str, dict] = {}
    dates = res.dates
    for name, (lo, hi) in REGIMES.items():
        mask = np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))
        seg = res.net[mask]
        if seg.size >= 2 and np.std(seg, ddof=1) > 0:
            sh = float(np.mean(seg) / np.std(seg, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        else:
            sh = 0.0
        out[name] = {"sharpe": round(sh, 3), "n_weeks": int(seg.size),
                     "mean_wk_pct": round(float(np.mean(seg) * 100), 3) if seg.size else 0.0}
    return out


def _final_holdout(res: XSecResult, frac: float = 0.25) -> dict:
    """A single held-out final slice (last `frac` of weeks) never used in tuning."""
    n = len(res)
    cut = int(n * (1 - frac))
    seg = res.net[cut:]
    if seg.size >= 2 and np.std(seg, ddof=1) > 0:
        sh = float(np.mean(seg) / np.std(seg, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    else:
        sh = 0.0
    return {"n_weeks": int(seg.size), "sharpe": round(sh, 3),
            "mean_wk_pct": round(float(np.mean(seg) * 100), 3) if seg.size else 0.0}


def build_grid() -> list[tuple[str, XSecConfig]]:
    grid: list[tuple[str, XSecConfig]] = []
    quantiles = ["quintile", "tercile"]
    weightings = ["equal", "volscaled"]
    # momentum: L x skip x quantile x weighting
    for L, skip, q, wt in product([7, 14, 28, 56], [0, 3, 7], quantiles, weightings):
        cfg = XSecConfig(signal="momentum", lookback=L, skip=skip, quantile=q, weighting=wt)
        grid.append((cfg.label, cfg))
    # carry: carry_lookback x quantile x weighting
    for cl, q, wt in product([7, 14, 28], quantiles, weightings):
        cfg = XSecConfig(signal="carry", carry_lookback=cl, quantile=q, weighting=wt)
        grid.append((cfg.label, cfg))
    # combined: a focused set (L=28/56, skip 0/7, carry 7/14) x quantile x weighting
    for L, skip, cl, q, wt in product([28, 56], [0, 7], [7, 14], quantiles, weightings):
        cfg = XSecConfig(signal="combined", lookback=L, skip=skip, carry_lookback=cl,
                         quantile=q, weighting=wt)
        grid.append((cfg.label, cfg))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xsec_full")
    ap.add_argument("--ckpt-dir", default="_ckpt")
    args = ap.parse_args()

    t0 = time.time()
    ckpt = Path(args.ckpt_dir) / args.tag
    ckpt.mkdir(parents=True, exist_ok=True)

    syms = tuple(all_symbols(sleeves=("crypto",)))
    print(f"[{args.tag}] building panel for {len(syms)} crypto perps ...", flush=True)
    panel: Panel = build_panel(syms)
    print(f"[{args.tag}] panel: {panel.n_days}d x {panel.n_symbols}sym "
          f"[{panel.dates[0].date()}..{panel.dates[-1].date()}]", flush=True)
    usize = {int(k): int(v) for k, v in universe_size_by_year(panel).items()}
    print(f"[{args.tag}] universe/yr: {usize}", flush=True)

    grid = build_grid()
    print(f"[{args.tag}] grid: {len(grid)} configs", flush=True)

    rows: list[dict] = []
    net_by_label: dict[str, np.ndarray] = {}
    results: dict[str, XSecResult] = {}
    for i, (label, cfg) in enumerate(grid):
        res = run(panel, cfg)
        m = metrics.compute(res.net, turnover=res.turnover, cost=res.cost, funding=res.funding)
        rows.append({
            "label": label, "signal": cfg.signal, "lookback": cfg.lookback, "skip": cfg.skip,
            "carry_lookback": cfg.carry_lookback, "quantile": cfg.quantile, "weighting": cfg.weighting,
            "n_weeks": m.n_weeks,
            "sharpe_net": round(m.sharpe_ann, 3), "sharpe_gross": round(_gross_sharpe(res), 3),
            "cagr_pct": round(m.cagr_pct, 2), "maxdd_pct": round(m.maxdd_pct, 2),
            "win_rate": round(m.win_rate_weeks, 4), "vol_ann_pct": round(m.vol_ann_pct, 2),
            "turnover_yr": round(m.turnover_per_year, 2),
            "cost_drag_yr_pct": round(m.cost_drag_per_year * 100, 3),
            "funding_income_yr_pct": round(m.funding_income_per_year * 100, 3),
            "univ_min": int(res.universe_size.min()), "univ_med": int(np.median(res.universe_size)),
            "univ_max": int(res.universe_size.max()),
        })
        net_by_label[label] = res.net
        results[label] = res
        if (i + 1) % 10 == 0:
            print(f"[{args.tag}] {i+1}/{len(grid)} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False).reset_index(drop=True)

    # trial family: ALL grid net Sharpes (honest multiplicity for DSR).
    trial_sharpes = df["sharpe_net"].to_numpy(dtype=float)
    n_trials = len(grid)

    # best net per signal family + overall
    best_overall = df.iloc[0]["label"]
    best_by_signal = {
        s: df[df["signal"] == s].iloc[0]["label"] for s in ("momentum", "carry", "combined")
        if (df["signal"] == s).any()
    }

    # discipline + regime + holdout on the notable configs
    notable = sorted(set([best_overall, *best_by_signal.values()]))
    detail: dict[str, dict] = {}
    for label in notable:
        res = results[label]
        wf = purged_walkforward(res.net, n_folds=3, embargo_weeks=1)
        dsr = deflated_sharpe(res.net, trial_sharpes, n_trials_effective=n_trials, k_distinct=n_trials)
        detail[label] = {
            "wf_oos_sharpe": round(wf.oos_sharpe_ann, 3), "wf_oos_n_weeks": wf.oos_n_weeks,
            "wf_fold_sharpes": [round(x, 3) for x in wf.fold_sharpes],
            "dsr": round(dsr.dsr, 4), "dsr_n_obs": dsr.n_obs, "min_trl": round(dsr.min_trl, 1),
            "dsr_sufficient": bool(dsr.n_obs_sufficient), "dsr_reasons": list(dsr.reason_codes),
            "holdout": _final_holdout(res),
            "regime": _regime_sharpe(res),
        }

    out = {
        "tag": args.tag, "runtime_s": round(time.time() - t0, 1), "n_configs": n_trials,
        "panel": {"n_days": panel.n_days, "n_symbols": panel.n_symbols,
                  "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date())},
        "universe_by_year": usize,
        "best_overall": best_overall, "best_by_signal": best_by_signal,
        "detail": detail,
        "table": df.to_dict(orient="records"),
    }
    outp = ckpt / "results.json"
    outp.write_text(json.dumps(out, indent=2))

    # human table
    print(f"\n[{args.tag}] TOP 20 by net Sharpe:", flush=True)
    cols = ["label", "n_weeks", "sharpe_net", "sharpe_gross", "cagr_pct", "maxdd_pct",
            "win_rate", "turnover_yr", "cost_drag_yr_pct", "funding_income_yr_pct"]
    print(df[cols].head(20).to_string(index=False), flush=True)
    print(f"\n[{args.tag}] BEST per signal:", flush=True)
    for s, lab in best_by_signal.items():
        r = df[df["label"] == lab].iloc[0]
        print(f"  {s:<9} {lab}", flush=True)
        print(f"            net_Sh={r.sharpe_net:+.2f} gross_Sh={r.sharpe_gross:+.2f} "
              f"CAGR={r.cagr_pct:+.1f}% maxDD={r.maxdd_pct:.1f}% win={r.win_rate:.2%} "
              f"turn/yr={r.turnover_yr:.1f} cost/yr={r.cost_drag_yr_pct:+.2f}% "
              f"fund/yr={r.funding_income_yr_pct:+.2f}%", flush=True)
    print(f"\n[{args.tag}] DISCIPLINE (best configs):", flush=True)
    for label, d in detail.items():
        print(f"  {label}", flush=True)
        print(f"    WF-OOS Sh={d['wf_oos_sharpe']:+.2f} (n={d['wf_oos_n_weeks']}, folds={d['wf_fold_sharpes']})",
              flush=True)
        print(f"    DSR={d['dsr']:.3f} n_obs={d['dsr_n_obs']} MinTRL={d['min_trl']} "
              f"suff={d['dsr_sufficient']} reasons={d['dsr_reasons']}", flush=True)
        print(f"    holdout(last25%)={d['holdout']}", flush=True)
        print(f"    regime={d['regime']}", flush=True)
    print(f"\n[{args.tag}] wrote {outp} ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
