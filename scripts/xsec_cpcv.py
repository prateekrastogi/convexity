"""CPCV on the xsec momentum LOOKBACK FAMILY (quintile / vol-scaled / weekly / 50-crypto / REAL cost).

The one robustness pillar the xsec edge had NOT yet faced. Purged-WF + DSR + block-bootstrap + 25%
holdout already ran; this adds Combinatorial Purged CV — a DISTRIBUTION of OOS path Sharpes for the
selected lookback plus PBO (probability of backtest overfitting) over the whole lookback family (is
picking the best L overfitting?). REUSES ``convexity.system.validate.cpcv.SystemCPCV`` and the xsec
engine; the CPCV glue lives in ``convexity.system.cpcv``.

Family: L in {10,12,14,16,18,21,28} d, all momentum / quintile / vol-scaled / weekly dollar-neutral,
50-crypto point-in-time, REAL cost (dollar-volume spread proxy + Amihud impact baked into the engine's
round-trip friction defaults). N=8 groups, k=2 test groups => C(8,2)=28 backtest paths, embargo=1 week.

Run on the VM:
  PYTHONPATH=. .venv/bin/python scripts/xsec_cpcv.py --tag xsec_cpcv
  PYTHONPATH=. .venv/bin/python scripts/xsec_cpcv.py --tag xsec_cpcv_l14 --only-l14   # sanity
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from convexity.panel.candidates import all_symbols

from convexity.system import metrics
from convexity.system.cpcv import run_cpcv
from convexity.portfolio.engine import XSecConfig, run
from convexity.panel.panel import Panel, build_panel, universe_size_by_year

LOOKBACKS: tuple[int, ...] = (10, 12, 14, 16, 18, 21, 28)
SELECTED_L: int = 14  # the fitted peak — the config whose OOS path distribution we headline


def _cfg(lookback: int) -> XSecConfig:
    """The DEPLOYABLE family member: momentum / quintile / vol-scaled / weekly / engine REAL cost."""
    return XSecConfig(
        signal="momentum",
        lookback=lookback,
        skip=0,
        quantile="quintile",
        weighting="volscaled",
        rebalance_days=7,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xsec_cpcv")
    ap.add_argument("--ckpt-dir", default="_ckpt")
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("--k-test", type=int, default=2)
    # default 3 = ceil(L_max/7), the official config since research/ROBUST.md R3 (LdP:
    # embargo >= feature horizon); pass --embargo 1 to reproduce the historical record.
    ap.add_argument("--embargo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-l14", action="store_true",
                    help="sanity: run CPCV path distribution for L=14 alone (still needs the family for PBO)")
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

    # Build the weekly net-return series for the WHOLE lookback family, keyed by rebalance DATE so we
    # can align on the shared trailing grid. A longer L needs more warmup, so its first Friday can be
    # one rebalance later (e.g. L=28 => 258 weeks vs 259); every series shares the SAME terminal grid,
    # so we align on the intersection of rebalance dates (a trailing window) — no lookahead, just the
    # weeks all configs cover in common.
    raw_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # label -> (dates, net)
    label_by_l: dict[int, str] = {}
    for L in LOOKBACKS:
        cfg = _cfg(L)
        res = run(panel, cfg)
        raw_series[cfg.label] = (res.dates.to_numpy(), res.net)
        label_by_l[L] = cfg.label
        m = metrics.compute(res.net, turnover=res.turnover, cost=res.cost, funding=res.funding)
        print(f"[{args.tag}] L={L:>2}d  {cfg.label}  n_weeks={m.n_weeks}  "
              f"net_Sh={m.sharpe_ann:+.3f}  CAGR={m.cagr_pct:+.1f}%  maxDD={m.maxdd_pct:.1f}%", flush=True)

    common = set.intersection(*(set(d.tolist()) for d, _ in raw_series.values()))
    family_series: dict[str, np.ndarray] = {}
    for label, (dates, net) in raw_series.items():
        keep = np.array([dt in common for dt in dates.tolist()], dtype=bool)
        family_series[label] = net[keep]
    n_weeks = len(next(iter(family_series.values())))
    print(f"[{args.tag}] aligned family on {n_weeks} common weeks "
          f"(dropped {max(len(n) for _, n in raw_series.values()) - n_weeks} leading week(s))", flush=True)
    selected_label = label_by_l[SELECTED_L]

    report = run_cpcv(
        family_series,
        selected_label,
        n_groups=args.n_groups,
        k_test=args.k_test,
        embargo=args.embargo,
        seed=args.seed,
    )
    ps = report.path_summary
    pbo = report.pbo

    out = {
        "tag": args.tag,
        "runtime_s": round(time.time() - t0, 1),
        "panel": {"n_days": panel.n_days, "n_symbols": panel.n_symbols,
                  "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date())},
        "universe_by_year": usize,
        "cpcv": {"n_groups": report.n_groups, "k_test": report.k_test,
                 "embargo_weeks": report.embargo_weeks, "n_weeks": n_weeks, "n_paths": report.n_paths},
        "family_labels": list(report.family_labels),
        "selected_label": selected_label,
        "path_sharpe_distribution_selected": {
            "n_paths": ps.n_paths, "mean": round(ps.mean, 4), "median": round(ps.median, 4),
            "p05": round(ps.p05, 4), "min": round(ps.minimum, 4),
            "frac_pos": round(ps.frac_pos, 4), "frac_gt_0.5": round(ps.frac_gt_half, 4),
            "naive_full_sharpe": round(ps.naive_full_sharpe, 4), "haircut": round(ps.haircut, 4),
        },
        "pbo": {
            "pbo": round(pbo.pbo, 4), "n_splits": pbo.n_splits, "n_configs": pbo.n_configs,
            "median_oos_rank_of_best_IS": round(float(np.median(pbo.oos_ranks)), 4) if pbo.oos_ranks else None,
            "best_IS_label_counts": {lab: pbo.best_is_labels.count(lab) for lab in report.family_labels},
        },
    }
    if args.only_l14:
        out["sanity_only_l14"] = True

    outp = ckpt / "results.json"
    outp.write_text(json.dumps(out, indent=2))

    print(f"\n[{args.tag}] CPCV N={report.n_groups} k={report.k_test} "
          f"embargo={report.embargo_weeks}w  => {report.n_paths} paths on {n_weeks} weeks", flush=True)
    print(f"[{args.tag}] SELECTED {selected_label}", flush=True)
    print(f"    path Sharpe: mean={ps.mean:+.3f} median={ps.median:+.3f} p05={ps.p05:+.3f} "
          f"min={ps.minimum:+.3f}", flush=True)
    print(f"    %paths>0={ps.frac_pos:.1%}  %paths>0.5={ps.frac_gt_half:.1%}", flush=True)
    print(f"    naive full-sample Sh={ps.naive_full_sharpe:+.3f}  haircut={ps.haircut:+.3f}", flush=True)
    print(f"[{args.tag}] PBO over the lookback family = {pbo.pbo:.3f} "
          f"(n_splits={pbo.n_splits}, n_configs={pbo.n_configs})", flush=True)
    print(f"    best-IS pick counts: {out['pbo']['best_IS_label_counts']}", flush=True)
    print(f"\n[{args.tag}] wrote {outp} ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
