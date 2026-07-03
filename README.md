# convexity — crypto cross-sectional momentum research

> The directional convex-breakout thesis this repo started from is **dead** (proven no gross
> directional edge — see `docs/EDGE.md`). What survived the investigation is **crypto
> cross-sectional momentum**: weekly, dollar-neutral, quintile, vol-scaled. This repo is now
> stripped to the minimum needed to continue that one edge.

## The edge (one line)

Weekly dollar-neutral quintile momentum on the point-in-time crypto perp panel (~60–75 names,
12–16d lookback band, vol-scaled): honest **net Sharpe ~0.8–1.0**, CAGR ~20–25% @100% gross,
maxDD ~25%, ~60% of weeks positive. Passed purged-WF / bootstrap / holdout / per-regime / breadth /
CPCV-PBO (official record, extended 6.5yr panel @ 3wk embargo: CPCV mean-path **1.13**,
frac_pos **1.00**, PBO **0.11**; 5yr historical: 0.95/0.93/0.40); **DSR now PASSES**
(0.9505, n_obs 334 vs MinTRL 331 — marginal, see `docs/EDGE.md` §(d)); Friday phase premium
persists with all phases positive on the extended panel;
**DSR fails on track length only** (PSR 0.988 — the ~5yr sample is short of MinTRL; the single
biggest honesty flag). Full memo: **[`docs/EDGE.md`](docs/EDGE.md)**.

**Status (2026-07):** all 9 pillar specs in `docs/specs/` are IMPLEMENTED and test-covered
(613 tests), independently math-audited (2 critical + 8 major defects found and fixed — including
a turnover double-drift that overstated cost in every earlier backtest; baselines re-measured),
real per-symbol Binance lot/tick filters wired, and the standard equity-momentum enhancement menu
was pre-registered, tested, and REJECTED (`docs/EDGE.md` §(d)). The one remaining step is the
live/paper rebalancer below.

## Layout

```
src/convexity/           THE DEPLOYABLE PACKAGE — nine pillars + a shared panel foundation,
                          self-contained (no dependency on research/, which no longer exists)
  panel/                    shared foundation: point-in-time daily panel, lake access
                            (data.py), asset-class/sleeve detection, candidate universe
  universe/                 UNIVERSE   — eligibility, liquidity ranking, shortability, PIT checks
  signal/                   SIGNAL     — no-lookahead momentum / carry scores
  portfolio/                PORTFOLIO  — quantile construction, vol-scaled weights, engine
  leverage/                 LEVERAGE   — Kelly/vol-target sizing, CUSUM ratchet, capacity ceilings
  cost/                     COST       — the ONE fee truth (Binance USDT-M taker), impact/spread/
                            funding accrual, ablation + CPCV cost validation
  execution/                EXECUTION  — Almgren-Chriss rebalance trajectory, no-trade band
  regime/                   REGIME     — CUSUM market-state detector, dispersion overlay
  system/                   SYSTEM     — Deflated Sharpe, purged walk-forward, CPCV/PBO,
                            promotion gate
  ops/                      OPS        — kill-switch, reconciliation, ARL-calibrated brakes
  core/                     shared types (Bars, InstrumentSpec, AssetClass), params, protocols

docs/
  EDGE.md                   the investigation synthesis + the deploy decision — read this first
  ARCHITECTURE.md           9-pillar data-flow + module map (historical directional architecture
                            preserved in Appendix A, not live)
  specs/                    UNIVERSE.md SIGNAL.md PORTFOLIO.md LEVERAGE.md COST.md EXECUTION.md
                            REGIME.md SYSTEM.md OPS.md — the authoritative build contract (exact
                            formulas, invariants, function signatures) for every pillar above

scripts/
  xsec_full.py              CANONICAL full sweep: momentum/carry grid x DSR x purged-WF x regime
  xsec_cpcv.py               CPCV / PBO entry
  xsec_pipeline.py           e2e entry point exercising all 9 pillars together on real data
  backfill_*_1m.py           self-contained lake backfill (crypto klines, funding, dukascopy, tradfi)
  stitch_tradfi_1m.py        tradfi-perp proxy stitching

tests/                    one directory per pillar (core/ cost/ execution/ leverage/ ops/ regime/
                          system/ universe/) plus integration/ for the full e2e pipeline
data/                     137-instrument 1-min lake (gitignored, ~15G — untouched)
```

## Run

```bash
# full cross-sectional sweep (writes JSON + a human table)
PYTHONPATH=. .venv/bin/python scripts/xsec_full.py --tag xsec_full

# CPCV / PBO
PYTHONPATH=. .venv/bin/python scripts/xsec_cpcv.py

# e2e pipeline across all 9 pillars
PYTHONPATH=. .venv/bin/python scripts/xsec_pipeline.py

# tests
.venv/bin/python -m pytest tests/ -q
```

## The live strategy slot: paper rebalancer

**BUILT (2026-07)** — `src/convexity/paper/` (spec: `docs/specs/PAPER.md`; units + runbook:
`deploy/`): scheduled weekly sessions (settle→target→size→plan→orders→journal), targets
parity-pinned bit-for-bit to the validated engine, quarter-Kelly gross, USDT→USDC collateral
waterfall (BTC never — user mandate), maker-first venue-exact orders, TWR flow stripping, daily
reconciliation. Dry-run works with no keys (`python -m convexity.paper rebalance`); the demo
gateway is EXPERIMENTAL until the first SUPERVISED `--live` session — run that before enabling
the timers (`deploy/README.md`).

## Discipline (non-negotiable)

Net-of-cost always (fees + signed funding). No-lookahead by construction (point-in-time universe,
signals strictly before the rebalance stamp). Report winner-robustness (DSR, PBO, drop-top-N) —
the cross-sectional backtest is trivially easy to fool yourself on. Deploy the lookback **band**,
not one fitted L. Size quarter-Kelly; scale up only on live confirmation.
