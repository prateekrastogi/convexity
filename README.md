# convexity — quantitative research + a three-book live trading fleet

> The directional convex-breakout thesis this repo started from is **dead** (proven no gross
> directional edge — `docs/EDGE.md`). What survived is three independent books, each with its own
> journal, symbol domain, capital allocation and schedule: **crypto cross-sectional momentum**,
> a **commodity RV pair book**, and an **equity trend book**. They run together as one fleet
> (`docs/specs/PAPER.md` §11–§12).

**Status (2026-08): LIVE ON REAL MONEY.** The fleet has traded the production Binance USDT-M
account since 2026-07-11, with the demo account kept running as a parity shadow. Capital is split
40/30/30 across crypto/commodity/equity. 736 tests. The whole deployment is one Docker container
(`docker compose up -d`) — see [`ops/README.md`](ops/README.md).

## The three books

| book | edge | status |
|---|---|---|
| **crypto** | weekly dollar-neutral quintile momentum, vol-scaled, ~60–75 perp names, 12–16d lookback band | the surviving research verdict — net Sharpe ~0.8–1.0 |
| **commodity** | band-z mean reversion on 5 pre-named RV pairs, half-life gate, pair risk parity, 10% vol target | surviving Program-4 candidate — backtest net 0.73, plan around 0.5–0.6 live |
| **equity** | S1 three-rule trend ensemble, inverse-vol, breadth-scaled, half-Kelly, drawdown brake | **owner-decision timed beta over a negative research verdict** — read the caveat below |

**Crypto** is the edge the investigation actually produced: weekly, dollar-neutral, quintile,
vol-scaled on the point-in-time perp panel. CAGR ~20–25% @100% gross, maxDD ~25%, ~60% of weeks
positive. Passed purged-WF / bootstrap / holdout / per-regime / breadth / CPCV-PBO (extended
6.5yr panel @ 3wk embargo: CPCV mean-path 1.13, frac_pos 1.00, PBO 0.11). DSR passes at 0.9505
(n_obs 334 vs MinTRL 331) — **marginal, on track length**, and the single biggest honesty flag.

**Commodity** is the one surviving commodity candidate (`docs/EDGE.md` §(f)). Fifteen separate
attempts to improve its drawdown profile across programs P5–P8 all failed; the book is deployed as
validated, not as improved.

**Equity** ships against a negative research verdict and the distinction matters. What IS
validated (D1, and pinned bit-for-bit by `tests/paper/test_equity_parity.py`) is the
*construction*: CAGR 28.26%, maxDD 30.02%, Sharpe 1.1676. What is **not** certified is the
underlying edge — the universe is survivorship-curated, so the book is a deliberate,
time-boxed owner bet, not a research conclusion. `docs/EDGE.md` §(g).

## Layout

```
src/convexity/           THE DEPLOYABLE PACKAGE — nine pillars + a shared panel foundation
                          (research/ holds study records + reusable study machinery, never
                          imported by the deployable package)
  panel/                    shared foundation: point-in-time daily panel, lake access
                            (data.py), asset-class/sleeve detection, candidate universe
  universe/                 UNIVERSE   — eligibility, liquidity ranking, shortability, PIT checks
  signal/                   SIGNAL     — no-lookahead momentum / carry / RV-pair / equity-trend
  portfolio/                PORTFOLIO  — quantile construction, vol-scaled weights, engine
  leverage/                 LEVERAGE   — Kelly/vol-target sizing, CUSUM ratchet, capacity ceilings
  cost/                     COST       — the ONE fee truth (Binance USDT-M taker), impact/spread/
                            funding accrual, ablation + CPCV cost validation
  execution/                EXECUTION  — Almgren-Chriss trajectory, calibrated no-trade band
  regime/                   REGIME     — CUSUM market-state detector, dispersion overlay
  system/                   SYSTEM     — Deflated Sharpe, purged walk-forward, CPCV/PBO,
                            promotion gate
  ops/                      OPS        — kill-switch, reconciliation, ARL-calibrated brakes
  paper/                    PAPER      — the three-book live rebalancer (PAPER.md):
                            books.py (disjoint book registry), cli.py (--book sessions),
                            policy/gateway/journal/capital/topup/venue
  core/                     shared types (Bars, InstrumentSpec, AssetClass), params, protocols

ops/                     THE DEPLOYMENT (see ops/README.md)
  docker/                   what runs INSIDE the container: crontab, job wrapper, entrypoint,
                            missed-job catch-up
  deploy/                   what deploys the container: autodeploy.sh (poll origin -> build ->
                            test -> promote)

docs/
  EDGE.md                   the investigation synthesis + the deploy decision — read this first
  ARCHITECTURE.md           9-pillar data-flow + module map (historical directional architecture
                            preserved in Appendix A, not live)
  specs/                    UNIVERSE.md SIGNAL.md PORTFOLIO.md LEVERAGE.md COST.md EXECUTION.md
                            REGIME.md SYSTEM.md OPS.md PAPER.md — the authoritative build contract
                            (exact formulas, invariants, function signatures) for every pillar
                            above; PAPER.md §11–§12 = the multi-book fleet design
  programs/                 in-flight program state records

research/
  entry_exit/               formal entry/exit model package (frozen preregistration discipline)
  entry_exit/study/         study records: frozen preregs + reports for the swing-xsec (DEAD),
                            durable-TSMOM/carry (DEAD) and RV-pair (SURVIVING -> the commodity
                            book) programs, plus reusable pipelines
  bandcal/                  the offline no-trade-band calibration (band_c 0.5 -> 0.03)

scripts/                  research entry points + lake backfill/maintenance
tests/                    one directory per pillar, plus integration/ and paper/
data/                     ~24 GB 1-min lake + the fleet's journals (gitignored)
```

## Run

Everything runs in the container — there is no host `.venv`.

```bash
docker compose up -d                 # start the fleet (scheduler included)
docker compose logs -f fleet         # watch it
docker compose ps                    # is it alive
```

One-off commands use the image's default entrypoint (the interpreter):

```bash
docker compose run --rm --entrypoint /app/.venv/bin/python fleet -m convexity.paper status
docker compose run --rm --entrypoint /app/.venv/bin/python fleet scripts/fleet_digest.py
```

Tests run as part of the image build and **gate it** — a red suite produces no image:

```bash
docker build -t convexity:latest .                    # build == test
docker build --target test -t convexity:test .        # tests only
```

The equity parity gate needs the 24 GB lake, which is deliberately not in the build context, so it
self-skips in-image and is run against the mounted data by `ops/deploy/autodeploy.sh` before any
image is promoted.

## Discipline (non-negotiable)

Net-of-cost always (fees + signed funding). No-lookahead by construction (point-in-time universe,
signals strictly before the rebalance stamp). Report winner-robustness (DSR, PBO, drop-top-N) —
a cross-sectional backtest is trivially easy to fool yourself on. Deploy the lookback **band**,
not one fitted L. Size quarter-Kelly; scale up only on live confirmation. A negative result is a
result: the dead programs above are kept in `research/` precisely so they are not re-run.
