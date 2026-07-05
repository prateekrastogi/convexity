# convexity — cross-sectional momentum research + multi-book paper trading

> The directional convex-breakout thesis this repo started from is **dead** (proven no gross
> directional edge — see `docs/EDGE.md`). What survived the investigation is **crypto
> cross-sectional momentum**: weekly, dollar-neutral, quintile, vol-scaled — now LIVE as the
> crypto paper book on the Binance demo. A second, independent **commodity RV pair book** (the
> one surviving commodity candidate, `docs/EDGE.md` §(f)) is built beside it as a separate paper
> strategy — own journal, symbol domain, sizing and timers (`docs/specs/PAPER.md` §11).

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
was pre-registered, tested, and REJECTED (`docs/EDGE.md` §(d)). The crypto paper rebalancer is
LIVE on the demo (weekly timers since 2026-07-03); the commodity RV book is built and awaiting
its supervised first session (`docs/specs/PAPER.md` §11.5).

## Layout

```
src/convexity/           THE DEPLOYABLE PACKAGE — nine pillars + a shared panel foundation
                          (research/ holds study records + reusable study machinery, never
                          imported by the deployable package)
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
  paper/                    PAPER      — the multi-book live/paper rebalancer (PAPER.md):
                            books.py (disjoint book registry), cli.py (--book sessions),
                            policy/gateway/journal/capital/topup
  core/                     shared types (Bars, InstrumentSpec, AssetClass), params, protocols

docs/
  EDGE.md                   the investigation synthesis + the deploy decision — read this first
  ARCHITECTURE.md           9-pillar data-flow + module map (historical directional architecture
                            preserved in Appendix A, not live)
  specs/                    UNIVERSE.md SIGNAL.md PORTFOLIO.md LEVERAGE.md COST.md EXECUTION.md
                            REGIME.md SYSTEM.md OPS.md PAPER.md — the authoritative build contract
                            (exact formulas, invariants, function signatures) for every pillar
                            above; PAPER.md §11 = the multi-book design

research/
  entry_exit/               formal entry/exit model package (frozen preregistration discipline)
  entry_exit/study/         the commodity study record: frozen preregs + reports + results for
                            the swing-xsec (DEAD), durable-TSMOM/carry (DEAD) and RV-pair
                            (SURVIVING candidate -> the commodity paper book) programs, plus the
                            reusable pipelines (swing_pipeline.py = shared data layer)

scripts/
  xsec_full.py              CANONICAL full sweep: momentum/carry grid x DSR x purged-WF x regime
  xsec_cpcv.py               CPCV / PBO entry
  xsec_pipeline.py           e2e entry point exercising all 9 pillars together on real data
  backfill_*_1m.py           self-contained lake backfill (crypto klines, funding, dukascopy, tradfi)
  stitch_tradfi_1m.py        tradfi-perp proxy stitching
  discover_tradfi_perps.py   weekly commodity-perp discovery -> universe overlay + backfill
                             (data/universe only — NEVER auto-traded)

deploy/                   systemd units + runbook: crypto book timers (live) and commodity book
                          timers (installed, enable only after PAPER.md §11.5 runbook)

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

## The paper books

**Crypto (LIVE on demo since 2026-07-03)** — `src/convexity/paper/` (spec: `docs/specs/PAPER.md`;
units + runbook: `deploy/`): scheduled weekly sessions (settle→target→size→plan→orders→journal),
targets parity-pinned bit-for-bit to the validated engine, quarter-Kelly gross, USDT→USDC
collateral waterfall (BTC never — user mandate), maker-first venue-exact orders, TWR flow
stripping, daily reconciliation + the domain-scoped orphan sweep.

**Commodity RV pairs (LIVE on demo since 2026-07-06)** — the surviving Program-4
candidate (`research/entry_exit/study/RV_REPORT.md`: backtest net 0.73, plan-around 0.5–0.6
live): 5 pre-named pairs, band-z mean reversion, half-life gate, pair risk parity, 10% vol
target (`src/convexity/signal/rv_pairs.py`). Runs as a SEPARATE book (PAPER.md §11): own journal,
own virtual capital (`allocate`), own timers/logs; the shared account is never used to infer a
book's equity. Every command is book-scoped: `python -m convexity.paper rebalance --book
commodity` (dry-run needs no keys). Live bootstrap notes: the account must sign the venue's
TradFi-Perps agreement once (`gateway.sign_tradfi_agreement()`, else every order draws -4411),
and pairs are gated on what the EXECUTION venue actually lists (the demo carries the metals but
not CL/NATGAS yet, so the energy pair activates automatically when Binance adds them).

## Discipline (non-negotiable)

Net-of-cost always (fees + signed funding). No-lookahead by construction (point-in-time universe,
signals strictly before the rebalance stamp). Report winner-robustness (DSR, PBO, drop-top-N) —
the cross-sectional backtest is trivially easy to fool yourself on. Deploy the lookback **band**,
not one fitted L. Size quarter-Kelly; scale up only on live confirmation.
