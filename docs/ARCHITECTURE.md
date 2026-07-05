# Convexity — Implementation Architecture

**This document describes the live system: a 9-pillar crypto CROSS-SECTIONAL MOMENTUM book**
(weekly, dollar-neutral, quintile, vol-scaled) on Binance USDT-M perpetuals. The **specs in
`docs/specs/{UNIVERSE,SIGNAL,PORTFOLIO,LEVERAGE,COST,EXECUTION,REGIME,SYSTEM,OPS}.md` are the
authoritative source** for every formula, invariant, and parameter; this document is the map
between those nine specs, the real `src/convexity/{panel,universe,signal,portfolio,leverage,cost,
execution,regime,system,ops}/` code, and the weekly data-flow that ties them together. The fuller
evidentiary narrative — what was
tried, what died, what survived, and why — lives in **`docs/EDGE.md`**; read it before this
document if you have not already.

The repo was pivoted from a single-instrument, single-trade **directional convex-breakout**
thesis. That thesis is **PROVEN DEAD** (`docs/EDGE.md` §a: gross directional R ≈ 0 across every
timeframe, exit rule, entry filter, leverage tier, and three asset classes tested, even inverted —
a structural finding, not a tuning miss). Its architecture is preserved for the historical record
in **Appendix A** below, but it is not live, not being built, and not the subject of this
document's main sections.

---

## 0. First-class principle: NO-ML, closed-form-only

**Every estimator in this system is closed-form or classical statistics — never a fitted,
trained, or black-box model.** This is not an implementation detail buried in one pillar; it is a
repo-wide constraint that every one of the nine specs independently upholds, and it is why the
system is auditable end-to-end from the formulas in `docs/specs/*.md` to the code in
`src/convexity/{panel,universe,signal,portfolio,leverage,cost,execution,regime,system,ops}/`.

**Forbidden, unconditionally:** neural networks, gradient-boosted trees, any learned/fitted
black-box signal combiner, large hyperparameter grid search or walk-forward retraining loops, and
continuously-refit hidden-Markov/EM regime models. `pyproject.toml`'s own dependency doctrine
states this plainly: `scipy: closed-form (NOT ML)`.

**What is used instead, pillar by pillar:**

| Tool | Where it appears | Spec anchor |
|---|---|---|
| Cross-sectional percentile rank (order statistics) | SIGNAL's momentum/carry/combined score | `SIGNAL.md` §3 |
| EWMA volatility estimator | LEVERAGE's `σ̂_t`, REGIME's dispersion `D_t` | `LEVERAGE.md` §4, `REGIME.md` §2.2 |
| Ledoit-Wolf shrinkage (Ledoit & Wolf 2004) | cited as the disciplined alternative wherever a covariance estimate is tempting | `LEVERAGE.md` §4.3, `PORTFOLIO.md` §4.2 |
| Inverse-realized-vol weighting (closed-form, no optimizer) | PORTFOLIO's within-leg weights | `PORTFOLIO.md` §4.2 |
| CUSUM sequential change-detection + average-run-length calibration (Page 1954) | REGIME's market-state detector, OPS's kill-switch | `REGIME.md` §3, `OPS.md` §4 |
| Moving-block bootstrap (Künsch 1989), block length per Politis & White (2004) | SYSTEM's confidence intervals on weekly Sharpe | `SYSTEM.md` §7 |
| Deflated / Probabilistic Sharpe Ratio, closed-form (Bailey & López de Prado) | SYSTEM's promotion gate | `SYSTEM.md` §5 |
| Combinatorial Purged CV + PBO, closed-form combinatorics (López de Prado 2014) | SYSTEM/`src/convexity/system/cpcv.py`'s lookback-band robustness check | `SYSTEM.md` §4 |
| Cube-root law / Almgren-Chriss optimal execution (closed-form) | EXECUTION's rebalance-trajectory sizing | `EXECUTION.md` §2.4, §4.2 |

**Why full mean-variance/Markowitz optimization is refused, specifically (not just "we prefer
simple methods"):** `PORTFOLIO.md` §4.2 and `LEVERAGE.md` §4.3 independently invoke the
**Marchenko–Pastur / random-matrix-theory eigenvalue-spreading argument** at the system's actual
ratio `N/n_obs ≈ 50/258 ≈ 0.19` — with only ~5 years of weekly observations (`n_obs≈258`) against
an `N≈50`-name universe, a sample cross-sectional covariance matrix is materially ill-conditioned;
its smallest eigenvalues are dominated by estimation noise, not real structure, and a fitted
mean-variance optimizer would lever up on exactly that noise. This is a closed-form,
falsifiable reason, not a style preference — LEVERAGE's own §4.3 ("Invariant NO-MV") confirms
the system never estimates the cross-sectional covariance matrix.

---

## 1. The nine pillars, the weekly rebalance, and the code they map to

The system runs on a **Friday-close weekly clock**. At each rebalance timestamp `t`, information
is admissible only if it is measurable with respect to the filtration `F_t` (the point-in-time
information set — formalized once in `UNIVERSE.md` §1.1 and reused without redefinition by every
downstream spec). One pass through the nine pillars turns `F_t` into next week's portfolio return.

| # | Pillar | Spec | Answers | Implemented in |
|---|---|---|---|---|
| 1 | **UNIVERSE** | `docs/specs/UNIVERSE.md` | Which names is it legitimate to rank/trade this week, using only what was knowable strictly before `t`? | `src/convexity/panel/panel.py` (point-in-time daily close/funding panel + listed mask), `src/convexity/panel/sleeve.py` (candidate-table loader), `src/convexity/panel/candidates.py` (UNIVERSE.md JSON parser) |
| 2 | **SIGNAL** | `docs/specs/SIGNAL.md` | Given the eligible universe, what is each name's cross-sectional rank (momentum, carry, combined)? | `src/convexity/signal/signals.py` |
| 3 | **PORTFOLIO** | `docs/specs/PORTFOLIO.md` | Given ranks, what dollar-neutral quantile weight vector realizes them? | `src/convexity/portfolio/portfolio.py` (`quantile_weights`, `liquidity_weights`, `realized_vol`) |
| 4 | **LEVERAGE** | `docs/specs/LEVERAGE.md` | What gross exposure `G_t` should scale the unit-gross book this week (quarter-Kelly, confidence-adjusted)? | Sizing math specified; consumes `src/convexity/system/validate/deflated.py` (`SR_LCB_t`) and `robustness.py` (`block_bootstrap_ci`) |
| 5 | **COST** | `docs/specs/COST.md` | What is the honest round-trip friction and signed funding cash flow for this week's turnover? | `src/convexity/cost/accounting.py` (`round_trip_friction`, `settle_week`), reusing `src/convexity/cost/config/fee_schedule.py` (`BINANCE_USDM_FEE_SCHEDULE`, the ONE fee truth) |
| 6 | **EXECUTION** | `docs/specs/EXECUTION.md` | How does the target book computed at the Friday close actually get filled — child-order trajectory, shortfall? | Specified only; **not yet built** — `src/convexity/cost/accounting.py::settle_week` still assumes a frictionless instantaneous Friday-close fill (the decision-price benchmark, not a fill price) |
| 7 | **REGIME** | `docs/specs/REGIME.md` | Should the book's gross be cut (or a defensive overlay applied) given the current cross-sectional market state? | `src/convexity/regime/overlay.py` (`beta_hedge_returns`, `vol_target_scale`, `drawdown_stop_scale`) |
| 8 | **SYSTEM** | `docs/specs/SYSTEM.md` | Does the resulting weekly return series survive the promotion gate (DSR, CPCV/PBO, purged walk-forward)? | `src/convexity/system/xsec_validate.py` (purged-WF + `DeflatedSharpe` wiring), `src/convexity/system/cpcv.py` (CPCV/PBO over the lookback family), reusing `src/convexity/system/validate/{deflated,cpcv,walkforward}.py` verbatim |
| 9 | **OPS** | `docs/specs/OPS.md` | How is the live/paper book monitored operationally — reconciliation, kill-switch, SLA, forced-flatten? | `src/convexity/ops/` (flows/TWR, recon, kill-switch) consumed by the LIVE paper rebalancer `src/convexity/paper/` (`docs/specs/PAPER.md`; multi-book §11: crypto + commodity, per-book journals/recon) |

Two supporting, non-pillar modules complete the picture: `src/convexity/portfolio/engine.py` is the driver
that composes panel → signal → portfolio → accounting into one weekly series per config, and
`src/convexity/system/metrics.py` computes the raw (non-deflated) weekly Sharpe/CAGR/maxDD/turnover/cost
summary that `validate.py` then deflates.

Of the eight `src/convexity/<pillar>/` packages the pre-pivot architecture (Appendix A §2) laid
out, two survive largely as-is and are reused verbatim by the new system; the rest were promoted
or newly built into the ten per-pillar packages the table above lists
(`{panel,universe,signal,portfolio,leverage,cost,execution,regime,system,ops}/`):

* **`src/convexity/core/`** — shared types (`Bars`, `InstrumentSpec`, `AssetClass`), the
  RED/AMBER/BLUE/TEAL governance buckets (`params.py`), and the cross-pillar composition
  Protocols (`protocols.py`). Still load-bearing: the xsec engine's data layer builds on
  `core.types.Bars`/`InstrumentSpec`, and every spec's parameter-governance section still uses the
  four-bucket taxonomy defined here.
* **`src/convexity/cost/`** — the fee schedule and friction structure. `BINANCE_USDM_FEE_SCHEDULE`
  in `cost/config/fee_schedule.py` is the ONE fee truth `src/convexity/cost/accounting.py` imports
  directly (`BINANCE_VIP0_TAKER = BINANCE_USDM_FEE_SCHEDULE.tier_for_volume(0.0).taker_rate`,
  5bp) — never a second, drifted fee table.
* **`src/convexity/system/validate/{deflated,cpcv,walkforward}.py`** — the DSR/CPCV/purged-WF
  machinery. Bailey–López de Prado's mathematics does not care whether the return series is
  1-minute trade R-multiples or weekly portfolio returns, so `src/convexity/system/xsec_validate.py` and
  `src/convexity/system/cpcv.py` reuse these modules directly rather than reimplementing the math for a
  weekly cadence — they instead build a **weekly-tuned `SystemDefaults`** (`weekly_defaults`,
  overriding `ppy` and the observation-count floors) so only the *cadence*, never the *math*,
  changes (`SYSTEM.md` §5.2).

The remaining `breakout/`, `trend/` package trees the old architecture (Appendix A §2) planned for
the directional system were never fully built and are **not being built** — that thesis is dead
(§0 above). The cross-sectional UNIVERSE/SIGNAL/PORTFOLIO/LEVERAGE/EXECUTION/REGIME/OPS pillars
are specified in `docs/specs/*.md` and are now implemented as the `src/convexity/{panel,universe,
signal,portfolio,leverage,execution,regime,ops}/` packages the table above cites, superseding the
old `selection/` (folded into `panel/` + `universe/`) naming.

---

## 2. Weekly data-flow (the composition graph)

Unlike the old directional system's per-instrument, per-trade composition graph (Appendix A §4),
every arrow below carries a **whole-book weekly vector**, not a single instrument's signal — this
is the risk-unit shift every pillar spec states explicitly: *"the risk unit is next week's
portfolio return, not one trade's R-multiple"* (`PORTFOLIO.md` §0/§2; `SYSTEM.md` §1).

```
 UNIVERSE.panel                                    SIGNAL.rank_t
 (panel.py: point-in-time      ──U_t──▶            (signals.py: momentum/carry/
  daily close+funding panel,                        combined percentile rank
  listed mask, F_t filtration)                       over U_t, [0,1], NaN if
                                                      ineligible)
                                                            │
                                                            │ rank_t
                                                            ▼
                                            PORTFOLIO.quantile_weights
                                            (portfolio.py §4-5 of PORTFOLIO.md:
                                             top/bottom quintile, inverse-realized-
                                             vol within leg, dollar-neutral,
                                             gross=1.0 reference — PORTFOLIO.md §2.3/§4.3)
                                                            │
                                                            │ w_t  (unit gross)
                                                            ▼
                                            LEVERAGE.size_book
                                            (LEVERAGE.md §6.1: G_t = c·SR_LCB_t/σ̂_t,
                                             quarter-Kelly on SYSTEM's DeflatedSharpe /
                                             block_bootstrap_ci — PORTFOLIO.md §14 names
                                             this hand-off; LEVERAGE.md's top-of-file
                                             note pins the shared symbol G)
                                                            │
                                                            │ G_t
                                                            ▼
                                            REGIME.overlay
                                            (overlay.py: beta-hedge / drawdown-stop /
                                             g_t∈[0,1] gross multiplier from CUSUM
                                             market-state detection — REGIME.md §0/§3/§4;
                                             gross_live = g_t · G_t)
                                                            │
                                                            │ w_t^live = gross_live · w_t
                                                            ▼
                                            EXECUTION (spec only — not yet built)
                                            (EXECUTION.md: child-order trajectory /
                                             cube-root no-trade band / implementation
                                             shortfall for turning w_t^live into fills;
                                             today accounting.settle_week stands in with
                                             a frictionless instantaneous Friday-close fill)
                                                            │
                                                            ▼
                                            accounting.settle_week
                                            (turnover, signed per-leg funding, price P&L)
                                                            │
                                                            │ turnover
                                                            ▼
                                            COST.round_trip_friction
                                            (accounting.py, reusing cost/config/fee_schedule.py:
                                             2·(taker+half_spread+impact) both legs)
                                                            │
                                                            ▼
                                            WeekResult.net  ──▶  metrics.py
                                                                  (raw weekly Sharpe/CAGR/maxDD)
                                                            │
                                                            ▼
                                SYSTEM.{validate,cpcv}  ──▶  DSR / CPCV / PBO / frac_pos
                                (validate.py + cpcv.py, reusing                │
                                 system/validate/{deflated,cpcv,walkforward}.py verbatim)
                                                                               │
                                                                               ▼
                                                                  promotion gate (SYSTEM.md §6)
                                                                               │
                                                                               ▼
                                            OPS -> PAPER (the live multi-book rebalancer,
                                            docs/specs/PAPER.md: weekly oneshot sessions,
                                            per-book journal/recon/orphan-sweep; crypto book
                                            live, commodity RV book built)
```

**The single-call-site gross invariant (never a double cut).** The gross-exposure scalar is set at
exactly ONE call site, never independently by two pillars: `PORTFOLIO.quantile_weights` /
`liquidity_weights` accept a single `gross=` parameter; LEVERAGE computes `G_t`, and REGIME
composes its own `g_t ∈ [0,1]` multiplicatively on top — `gross_live = g_t · G_t` — and that ONE
product is what is passed into `gross=` at the same call site, never two separate `gross=` calls.
This is the same "never a double cut" principle the old directional system's Appendix A §5 states
for its own exposure brakes (REGIME/LEVERAGE-governor/OPS-kill composing on disjoint axes rather
than multiplying independently); it carries over unchanged to this graph.

**Where the pipeline is genuinely incomplete today.** The live/paper rebalancer IS built and
papering (`src/convexity/paper/`, `docs/specs/PAPER.md` — thin composition over the pillars;
multi-book per §11). The remaining honest gap is accounting-level: `accounting.settle_week`
marks the whole week's P&L as if the book re-weighted atomically at the Friday closing print,
which is the correct *decision-price benchmark* but not a real fill price (`EXECUTION.md` §0
states this gap explicitly) — closing it is precisely the paper track's TCA mission.

---

## 3. Governance and parameter discipline (unchanged doctrine, carried over)

Every pillar spec classifies its parameters into the same four governance buckets defined once in
`src/convexity/core/params.py` and reused without redefinition by all nine specs:

* **RED** — shared economic/modelling constants, set once globally, reachable only through a
  `resolve_config(defaults=...)` single-merge-point (e.g. the Kelly fraction `φ`, CUSUM `(k,h)`).
* **AMBER** — per-instrument market metadata never optimised by any pillar (tick/lot size, venue,
  fee schedule, funding interval).
* **BLUE** — finite, pre-declared candidate grids searched offline (e.g. the `{12,14,16}`-day
  lookback family) and promoted only through SYSTEM's deflated-Sharpe/CPCV gate; every BLUE grid's
  cardinality is counted into SYSTEM's trial ledger so multiple-testing deflation is honest.
  `SYSTEM.md`'s real code-level floor for the minimum group count in a CPCV split is
  `k_min=3` (`src/convexity/system/xsec_validate.py::weekly_defaults`) — cite this, not any larger invented
  figure.
* **TEAL** — fitted per-name from strictly causal (`F_{t-1}`) past data at runtime: rolling
  realized vol, EWMA dispersion, Ledoit-Wolf shrinkage intensity. Never a hand knob, never
  full-sample, never P&L-fit.

This is the same governance discipline the old architecture's §7 (Appendix A) established; nothing
about the bucket taxonomy changed in the pivot, only which parameters live in which pillar.

---

## 4. Current validated performance envelope (the evidentiary baseline)

This is the system's live evidentiary baseline as of the most recent validated run. Every number
below is cited from `docs/EDGE.md`; do not restate a headline figure without its honest
counterpart. **`docs/EDGE.md` is the fuller narrative — read it for the full validation battery,
caveats, and next steps.**

| Dimension | Honest / deployable number | Caveat |
|---|---|---|
| Net Sharpe | **~0.8–1.0** (real Amihud-impact cost) | The light-cost/single-peak headline of Sharpe ~1.6–1.9 / CAGR ~65% is an **optimistic upper bound, not the deployable estimate** — cite it only as such |
| CAGR @ 100% gross | **~20–25%** | Scales with chosen gross; Sharpe is gross-invariant, CAGR/maxDD are not (`PORTFOLIO.md` §2.3, Lemma 1) |
| Max drawdown @ 100% gross | **~25%** (~28% annualized vol) | Scales ~linearly with gross/leverage |
| Weekly win rate | **~60%** | — |
| Universe size | **~50–75** point-in-time names; **N≈50 is the defensible simple choice** | Peaks empirically at N≈60–75, but the gain over N=50 is inside the block-bootstrap CI — not a real improvement; N=100 declines from illiquid-tail cost + ~15/50 added names effectively unshortable |
| Signal lookback | **L ∈ {12, 14, 16} trading days**, a sparse 3-point ensemble (m=3), deployed as a band | L=14 is a sharp fitted peak (PBO logit-relative-rank 0.31–0.47 across partitions); single-L selection is partly overfit — deploy the band, not one L. The ensemble is anchored on the three lookbacks with an explicit, cited, tested Sharpe figure; L=13 and L=15 have no tested backtest configuration anywhere in the evidence base and are deliberately excluded, not interpolated (`SYSTEM.md` §5.1) |
| CPCV (N=8 groups, k=2 test groups, 28 paths, 1-week embargo) | Mean OOS path Sharpe **~0.91** vs naive single-path **0.98** (haircut ~0.07) | No OOS collapse — categorically unlike the dead directional book, whose CPCV paths straddled zero |
| `frac_pos` (fraction of CPCV paths with positive Sharpe) | **0.85–1.00** — the **robust** statistic | The 5th-percentile path Sharpe is partition-fragile (swings +0.42 to −0.59 with N/k choice); promotion criteria weight `frac_pos` over `p05` |
| PBO (probability of backtest overfitting) | **≈0.36** | Moderate — real edge, not a slam-dunk; reinforces the band-not-a-point deployment rule |
| DSR status | **Fails on track-record length only** (PSR≈0.994, high; ~5yr = 258 weekly observations is short relative to the multi-testing-inflated MinTRL) | A track-length problem, not a deflation rejection |
| Funding carry (satellite only, never core sizing) | Real income **+12.9%/yr** gross | Structural ~70–75% max drawdown, regime-fragile (dies 2023–24 low-funding stretch), fails DSR (0.10–0.45 depending on overlay); no overlay fixes it — beta-hedge only lifts Sharpe to ~0.57 at ~70% DD, a causal DD-stop caps DD at ~24% but collapses Sharpe to ~0.10 and income to ~0.6%/yr |
| Sizing rule | **Quarter-Kelly**, scale up only on live confirmation | Conservative by construction given the PBO/DSR caveats above |

**Directional breakout/trend background, for contrast (dead, not deployed):** gross directional R
≈ 0 across every timeframe/exit/filter/leverage tier/asset class tested, even inverted
(`docs/EDGE.md` §a). Cited here only so the magnitude of "dead" (R≈0) is visible against the
magnitude of "alive" (Sharpe 0.8–1.0) in the same document.

---

## 5. Testing philosophy (carried over, re-scoped to the weekly risk unit)

The old architecture's four-layer test strategy (Appendix A §8: invariant tests, worked-example
regression fixtures, property tests, integration tests) is the still-correct shape of testing
discipline; what changed is the unit each layer operates on.

* **Invariant tests** — now keyed to the weekly risk unit's invariants: no-lookahead (perturbing
  any panel row after rebalance day `d` must not change the signal/weights at `d`), the
  single-call-site gross invariant (§2 above), dollar-neutrality (long leg sums to +0.5, short leg
  to −0.5), and the point-in-time listed mask (a delisted name is closed at its last live bar, a
  not-yet-listed name never enters a cross-section). These live under `tests/{cost,portfolio,regime,signal,system}/`.
* **Worked-example regression fixtures** — each spec's numbered worked example (e.g. COST.md's
  §9 end-to-end friction reconciliation, SYSTEM.md's §5.3 DSR worked numbers, REGIME.md's §3.5
  CUSUM-ARL calibration) is reproduced bit-for-bit from its tabulated inputs, exactly as Appendix
  A §8.2 already required for the directional pillars.
  `tests/cost/{test_accounting,test_breadth}.py`, `tests/system/{test_cpcv,
  test_metrics_validate}.py`, `tests/portfolio/{test_engine_panel,test_multiasset,
  test_portfolio}.py`, `tests/regime/test_overlay.py`, and `tests/signal/test_signals.py`
  are the live instances of this layer.
- **Property tests** — the monotonicities each spec asserts (e.g. `frac_pos` robustness to N/k
  choice, Sharpe's gross-invariance from PORTFOLIO.md §2.3's Lemma 1) are swept and asserted
  explicitly, no `hypothesis` dependency required — same discipline as Appendix A §8.3.
* **Integration tests** — the paper layer is covered by `tests/paper/` (parity pin, venue
  formatting, waterfall, book scoping/orphan-sweep isolation, settle windows) + supervised
  first `--live` sessions per book (`deploy/README.md`); deeper Nautilus integration tests
  activate with the TCA build.

Every pillar's tests need **no external data** — the cross-sectional engine reuses the same
seeded-synthetic-generator discipline the old architecture's §8 closing note required
(`tests/conftest.py` fixtures), so invariant and worked-example tests never touch
the 137-instrument production data lake.

---

## 6. Environment, repo, and data-layer conventions (unchanged by the pivot)

These conventions predate the pivot and remain accurate; see `CLAUDE.md` for the full, current
statement of each.

* **Dependency management:** `uv`, pinned via `uv.lock` (committed); `.python-version = 3.12`
  (the cp312 manylinux aarch64 wheel is what makes `uv sync` resolve a binary NautilusTrader
  wheel when the live rebalancer is eventually built). Run research code with `.venv/bin/python`
  or `uv run --no-sync`.
* **Repo hygiene:** hatchling build backend, ruff line-length 100, pytest. Commit/push only when
  explicitly asked; never commit secrets (`.env` is gitignored).
* **Data lake:** `data/` is the 137-instrument 1-min lake (gitignored, ~15G) — do not touch or
  delete it. `src/convexity/panel/data.py` (`load_bars`/`load_funding`) is the sole loader the xsec panel
  builds on. Symbology: the USDT-M perpetual is `BTCUSDT-PERP` in Nautilus terms; bare `BTCUSDT`
  is spot. Funding only accrues if explicitly charged — `src/convexity/cost/accounting.py` charges
  signed funding by side; there is no silent-zero-funding trap in the xsec path.
* **Open-source-first, minimal custom surface** (the repo's core development principle, stated in
  full in `CLAUDE.md`): NautilusTrader for the eventual paper/live engine, its native Binance
  USDT-M adapter for connectivity, pandas/numpy/pyarrow for the research dataframe layer, httpx +
  `data.binance.vision` for bulk historical backfill. The only code written from scratch is the
  cross-sectional engine (`src/convexity/{panel,universe,signal,portfolio,leverage,cost,execution,
  regime,system,ops}/`), the data-layer backfill scripts
  (`scripts/backfill_*_1m.py`), and the live/paper weekly rebalancer (`src/convexity/paper/`,
  reusing Nautilus's own Binance HTTP plumbing — never a hand-rolled signer). Do not
  reimplement an engine, a broker connector, a dataframe layer, or a backtest framework; do not
  add competing engines (freqtrade/hummingbot/vectorbt/zipline) or raw `python-binance`.
* **Refuted/forbidden — do not reintroduce** (full list in `CLAUDE.md`): directional
  breakout/trend in any form; funding carry as a standalone book (income-overlay at most); long-vol
  straddle/strangle, HAR-RV/Lee-Mykland vol forecasting, backspreads, OU mean-reversion exits
  (refuted upstream in the `vol_research`/`arbitrage` lineage); options greeks/moneyness/expiry-roll
  concepts (perps are linear instruments, no analog); presenting un-robust, lookahead-tainted, or
  gross-of-cost results as edge (cross-sectional backtests lie easily — survivorship, delist gaps,
  size-blind cost — results that don't survive DSR/PBO/purged-WF/drop-top-N are not edge).

---

## Appendix A — the directional convex-breakout architecture (historical record, not live)

**Status of this appendix.** Everything below describes the architecture for the directional
convex-breakout system, which is **PROVEN DEAD** (`docs/EDGE.md` §a: gross directional R ≈ 0
across every timeframe/exit/filter/leverage/asset-class cell tested, even inverted). It is
retained verbatim as a historical record of that build — including the cross-spec interface
mismatches it resolved and the test strategy it established, both of which informed the
conventions §§3/5 above carry forward — not as a live build plan. Do not resurrect any of the
`breakout/`, `trend/`, `leverage/`, `execution/`, `ops/`, `regime/` package trees below; they were
never fully built out and are not being built now.

### A.1 Architecture principle

Every pillar was split into two layers:

* **Pillar POLICY (pure-Python, NautilusTrader-FREE).** All the modelling, detection, sizing, cost,
  regime, monitoring and orchestration logic. Depends only on `numpy`/`pandas`/`scipy` and the
  shared `convexity.core`. Unit-testable without the engine — the test venv (`.venv-test`) has
  no `nautilus_trader` installed, and every pillar test runs `PYTHONPATH=src .venv-test/bin/python
  -m pytest`.
* **Thin Nautilus ADAPTER.** A small number of clearly-isolated modules — `strategy/`,
  `<pillar>/nautilus/`, `cost/facts/nautilus_adapter.py` — that translate between Nautilus
  events/state and the pure-Python pillar types. These are the *only* files that `import
  nautilus_trader`. They were to be covered by integration tests on the VM (where nautilus is
  installed), not by the pillar unit suite.

The single live `strategy/convexity_strategy.py` Nautilus `Strategy` subclass was to **compose**
the pillars: build the pure-Python inputs from Nautilus bars/events, call each pillar's outward
method, and route the returned commands/intents back through the Nautilus adapters.

This split is *why* the foundation (`convexity.core`) is the thing that makes pillars interoperate:
the boundary objects that cross pillar lines must be one shared type, not redefined per pillar.
**This principle is the one part of Appendix A that is still load-bearing today** — `core/types.py`
and `core/params.py` are reused as-is by the cross-sectional system (§1 above).

### A.2 Package layout (as planned; never fully built)

```
src/convexity/
  core/                      # SHARED FOUNDATION (implemented; still reused — see §1)
    params.py                #   RED/AMBER/BLUE/TEAL governance + typed param registry
    types.py                 #   cross-pillar boundary dataclasses (Bars, Side, EntrySignal, …)
    protocols.py             #   the composition-seam Protocols each pillar implements

  selection/                 # ALREADY IMPLEMENTED (Carver Ch.25 dynamic optimisation + universe)
    dynamic_optimization.py  #   greedy tracking-error optimiser (19 passing tests — DO NOT regress)
    universe.py  UNIVERSE.md

  breakout/                  # BREAKOUT pillar  (entry detector) — never fully built, not being built
  trend/                     # TREND pillar  (convex trend-life exit) — never fully built
  leverage/                  # LEVERAGE pillar  (min() of seven ceilings) — never fully built
  execution/                 # EXECUTION pillar  (tactic selection + protective stop) — never fully built
  cost/                      # COST pillar (fee schedule) — IMPLEMENTED, reused verbatim — see §1
  system/                    # SYSTEM pillar (orchestrator + integrated validation)
    validate/{deflated,cpcv,walkforward}.py   # IMPLEMENTED, reused verbatim — see §1
    # remaining SYSTEM orchestration modules (compose/, modules/) — never fully built
  ops/                       # OPS pillar  (SPC monitoring + kill switch) — never fully built
  regime/                    # REGIME pillar  (global gross multiplier g_t) — never fully built
  strategy/                  # THE Nautilus composition layer — existing stubs, never filled
  costs/binance_fees.py      # legacy stub — absorbed into cost/config/fee_schedule.py
```

### A.3 Cross-spec interface mismatches found in the blueprints — and their resolution

These were the conflicts each directional-era spec's `risks` section raised, resolved so that
build (had it continued) would have been unblocked. Retained for historical record:

1. **Package placement (all eight specs).** Resolution: everything under `src/convexity/<pillar>/`.
2. **REGIME `g_t` channel: REGIME.md vs LEVERAGE.md.** Resolution: single channel through
   `L_portfolio`, never a separate min() term.
3. **`ExecTarget` cross-pillar contract**, undefined in any sibling spec. Resolution: defined in
   `core/types.ExecTarget` with exactly 8 fields, no edge field (Inv A).
4. **`HoldHorizon` producer/`m_max` redundancy.** Resolution: `HoldHorizon.from_quantile`
   classmethod computes `m_max` so the field can never disagree.
5. **COST carry sign / LEVERAGE clamp location.** Resolution: `CarryTerm.over_horizon` returns
   signed (credit negative); the `max(carry,0)` clamp lives in `leverage/ceilings/stop.py`.
6. **`HoldHorizonEstimate` → COST `HoldHorizon` mapping.** Resolution: COST computes `m_max` via
   `HoldHorizon.from_quantile`; TREND never computes funding-stamp counts.
7. **CLAUDE.md "Lee-Mykland forbidden" vs TREND/BREAKOUT jump detectors.** Resolution: jump
   detectors default to the robust standardized-return exceedance branch for crypto; the
   bipower/Lee-Mykland branch is config-gated off by default.
8. **`scipy` dependency.** Resolution: scipy allowed in validation/research code, kept out of the
   live-engine runtime closure.
9. **`BookState` undefined in LEVERAGE.** Resolution: `system/types.BookState` is canonical.
10. **SYSTEM SelectionAdapter contract mismatch.** Resolution: `system/modules/selection.py` maps
    `DynOptResult→ProposedPortfolio`.
11. **Nautilus version (1.229) vs spec prose.** Resolution: adapters target nautilus 1.229/py312
    exactly.
12. **OPS DD-kill (45%) vs LEVERAGE governor flat (43%) shared reference.** Resolution: both read
    the same equity series + running-max peak.
13. **OPS `§17 rollout` filename.** Resolution: `ops/brake/rollout.py`.

### A.4 Test strategy (the shape carried forward into §5 above)

Four layers, mirroring the spec invariants + worked examples, all under `PYTHONPATH=src
.venv-test/bin/python -m pytest` with no nautilus:

1. **Invariant tests** — firewall/governance AST checks, causality/no-lookahead perturbation
   tests, batch/live parity + snapshot-restore, score/edge firewalls, min()/precedence uniqueness,
   diagnostics-are-log-only checks.
2. **Worked-example regression fixtures** — each spec's numbered worked example reproduced
   bit-for-bit (e.g. BREAKOUT's strict-pre-bar barrier identity, TREND's `z_drift=0.9531018`,
   LEVERAGE's governor ladder, COST's §5.7 centerpiece `C_impact=0.0282843`, SYSTEM's DSR
   FAIL/PASS fixtures, OPS's CUSUM cost-drift fixture, REGIME's §6.1 corr-spike fixture).
3. **Property tests** — monotonicities (`p_fill` ↓ / `A` ↑ / `D` ↑ in urgency; rounding only
   reduces risk; ARL₀ ↑ with `D_min`).
4. **Integration tests** (VM-only, nautilus installed) — batch↔live parity through the real
   engine, restart reconciliation, stop placement.

### A.5 Legacy stub reconciliation

* `strategy/breakout.py` → logic was to move to the `breakout/` package; never completed.
* `strategy/config.py`, `strategy/universe.py` → were to fold into per-pillar `config/`.
* `strategy/convexity_strategy.py` → the planned live composer; never filled.
* `costs/binance_fees.py` → absorbed into `cost/config/fee_schedule.py`; this part DID happen and
  is the fee truth §1 above cites.

### A.6 Governance discipline

Every pillar was to implement `resolve_config(defaults=...)` as the single merge point where RED
enters; per-class overrides through a separate `AssetClassPreset`; per-instrument `InstrumentSpec`
carrying AMBER only. This is the same RED/AMBER/BLUE/TEAL discipline §3 above restates for the live
system — it did not change in the pivot, only which parameters live in which pillar.
