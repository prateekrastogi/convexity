# convexity — crypto cross-sectional momentum research

> The directional convex-breakout thesis this repo started from is **dead** (proven no gross
> directional edge — see `docs/EDGE.md`). What survived the investigation is **crypto
> cross-sectional momentum**: weekly, dollar-neutral, quintile, vol-scaled. This repo is now
> stripped to the minimum needed to continue that one edge.

## The edge (one line)

Weekly dollar-neutral quintile momentum on the point-in-time crypto perp panel (~60–75 names,
12–16d lookback band, vol-scaled): honest **net Sharpe ~0.8–1.0**, CAGR ~20–25% @100% gross,
maxDD ~25%, ~60% of weeks positive. Passed purged-WF / DSR / bootstrap / holdout / per-regime /
breadth / CPCV-PBO. Full memo: **[`docs/EDGE.md`](docs/EDGE.md)**.

## Layout

```
research/xsec/          THE STRATEGY-RESEARCH HOME — the cross-sectional engine
  panel.py                point-in-time universe + daily panel (no survivorship lookahead)
  signals.py              no-lookahead momentum / carry signals
  portfolio.py            quantile construction, vol-scaled weighting, realized vol
  accounting.py           honest turnover cost (Binance taker) + signed funding cash flow
  engine.py               run(config) -> weekly return series (XSecBook / XSecResult)
  overlay.py              regime / breadth overlays
  metrics.py              weekly-return performance metrics (annualized Sharpe, CAGR, DD, ...)
  validate.py             purged walk-forward + Deflated Sharpe wiring
  cpcv.py                 combinatorial purged CV / PBO
  universe.py             point-in-time membership helpers
research/data.py         load_bars / load_funding from the lake
research/assetclass.py   sleeve / asset-class detection, native-funding flag
research/README.md        research-layer notes

src/convexity/           the reused pillars the engine imports (nothing else survives)
  core/                    types (Bars, InstrumentSpec, AssetClass), params, protocols
  cost/                    the ONE fee truth (Binance USDT-M) + cost model
  selection/               universe loader (UNIVERSE.md) + dynamic optimisation
  system/config/           SystemDefaults, resolve, tuning
  system/validate/         DeflatedSharpe, PurgedWalkForward, CPCV, + the validation battery
  system/types.py          the validation records (TrialRecord / DSRResult / SystemReport)

scripts/
  xsec_full.py             CANONICAL full sweep: momentum/carry grid x DSR x purged-WF x regime
  xsec_cpcv.py             CPCV / PBO entry
  backfill_*_1m.py         self-contained lake backfill (crypto klines, funding, dukascopy, tradfi)
  stitch_tradfi_1m.py      tradfi-perp proxy stitching

tests/                   coverage for the surviving pillars + the xsec engine
  core/  cost/  research/xsec/  test_dynamic_optimization.py

data/                    137-instrument 1-min lake (gitignored, ~15G — untouched)
docs/EDGE.md             the investigation synthesis + the deploy decision
docs/specs/              COST.md, SYSTEM.md (surviving pillar specs; directional specs deleted)
docs/ARCHITECTURE.md     architecture reference (directional sections now stale)
```

## Run

```bash
# full cross-sectional sweep (writes JSON + a human table)
PYTHONPATH=. .venv/bin/python scripts/xsec_full.py --tag xsec_full

# CPCV / PBO
PYTHONPATH=. .venv/bin/python scripts/xsec_cpcv.py

# tests
.venv/bin/python -m pytest tests/ -q
```

## The live strategy slot is intentionally EMPTY

There is **no live/paper rebalancer yet** — this repo is research + validation only. When papering
begins, build the weekly rebalancer against NautilusTrader (the reuse rules in `CLAUDE.md` still
apply) as `research/xsec/live/` (or a `strategy/` slot). Do NOT pre-build it. The next concrete
step is in `docs/EDGE.md` §(e).

## Discipline (non-negotiable)

Net-of-cost always (fees + signed funding). No-lookahead by construction (point-in-time universe,
signals strictly before the rebalance stamp). Report winner-robustness (DSR, PBO, drop-top-N) —
the cross-sectional backtest is trivially easy to fool yourself on. Deploy the lookback **band**,
not one fitted L. Size quarter-Kelly; scale up only on live confirmation.
