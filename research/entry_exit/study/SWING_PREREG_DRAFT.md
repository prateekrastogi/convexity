# SWING XSEC COMMODITY PRE-REGISTRATION — DRAFT v0.1 (2026-07-04, NOT YET FROZEN)

**Freeze conditions (all three, then this file is dated-frozen and only then may anything run):**
(1) the Dukascopy backward-extension has landed and re-stitched; (2) the COT lake is verified;
(3) the OWNER decides the curve-source question (§F1/F2: acquire CME term-structure data — e.g.
Norgate ~$40/mo or a one-off databento daily-settle pull — or defer carry/basis-momentum to a later
amendment). Nothing here may be tuned against strategy P&L before freezing; data-quality checks are
allowed, backtests are not.

**Honesty preamble (owner-supplied research, accepted as the prior):** published commodity xsec
factors are 0.5–0.9 at breadth 20–30; our breadth is 5→7 (≈0.53× scaling), post-publication decay
applies, and anything backtesting >1.5 at this horizon is treated as a bug/overfit alarm, not a
win. Target after vol-managed ensemble: **0.5–0.7 lake-only; 0.85–1.15 only if the curve factors
are acquired.** Directional constructions are already measured dead on this universe (this study's
own stage-2/2b); everything below is CROSS-SECTIONAL and dollar-neutral.

## Universe, data, calendar

7 Binance TradFi perps: XAUUSDT, XAGUSDT, CLUSDT, NATGASUSDT, COPPERUSDT, XPTUSDT, XPDUSDT.
History = stitched 1-min (Dukascopy proxy + perp tail), extended: XAU 2009+, XAG 2008-06+,
CL/NATGAS 2013+, COPPER 2012+, XPT/XPD 2021+ (no earlier duka history — probed). Panel = weekly,
**Friday-close UTC** rebalance (the crypto book's convention; machinery reuse). A name enters the
cross-section when listed (PIT); the book trades only weeks with ≥5 listed names (⇒ effective
start 2013; ~690 weekly obs to 2026-06). Daily features from UTC-day closes of the 1-min lake.
COT joins on `usable_from` (report Tue, released Fri; a rebalance may only read reports with
usable_from ≤ rebalance timestamp — one-week-stale by construction, never lookahead).

## Factors (constructions frozen at freeze-time; grids ARE the multiplicity ledger)

- **F3 Liquidity-provision reversal (COT-conditioned).** Signal = −(1-week return), traded ONLY in
  names where |Δmm_net|/OI over the same week is in the top half of the cross-section (the KRT
  flow-accommodation condition; mm_net = mm_long − mm_short, lagged per usable_from). Hold 1 week.
  Grid: condition {on, off} (2).
- **F4 Momentum.** 8-week lookback band {6,8,10} (the swing-relevant short window, accepting the
  documented decay), skip-week none. Long top-2 / short bottom-2. Hold 1 week. Grid: 3 (band
  members, deployed as the BAND ensemble like the crypto book — not argmax).
- **F5 Skewness.** 60-day realized daily-return skew; long most-negative-2 / short most-positive-2;
  hold 1 week. Grid: window {40,60} (2).
- **F6 Hedging-pressure conditioner.** prod_net/OI percentile as a double-sort filter ON F4
  (momentum taken only where it agrees with hedging pressure). Grid: {on, off} (2).
- **F1 Carry + F2 Basis-momentum — CONDITIONAL on curve data.** If acquired before freeze: F1 =
  annualized front-slope, long most-backwardated-2 / short most-contangoed-2, hold 1-4wk (grid 2:
  1wk/4wk); F2 = 4-week Δslope xsec, hold 1wk. Gold's degenerate carry (infinite inventory) is
  accepted as-is — the rank construction handles it; no special-casing. If NOT acquired: F1/F2
  enter a later dated amendment and are NOT part of this family.
- **F7 Vol-managed overlay (not an edge).** Inverse-vol leg weights (20-day σ) inside every factor;
  book-level vol targeting to a fixed σ* with scaling capped at [0.5, 2.0]; quarter-Kelly sizing
  downstream. No grid (house doctrine).

**Ensemble.** Equal-RISK combination of the available factors' weekly weight vectors (each factor
dollar-neutral, inverse-vol legs, gross 1.0), then the F7 overlay. Correlation between factor
weight vectors reported; no optimization of combination weights (equal-risk is frozen).

## Costs

12bp RT primary per name-week traded (Binance TradFi taker+slippage; measured spreads ≤~9bp
single-side, mostly 1-5bp); scenarios {6,12,24}bp. Funding: actuals where the perp era has them;
±5bp/week-held scenario for the proxy era (declared approximation). Turnover reported per factor;
F3 is the turnover hog and must survive 24bp to be believed.

## Validation & gates (the crypto book's bar, verbatim discipline)

- CPCV on the weekly grid: 8 groups, k=2 (28 splits), **embargo 6 weeks** (covers 4-week holds +
  feature windows). Factor params are a-priori (no fitting) except F1 hold — the full grids run and
  are ALL charged.
- **Multiplicity ledger N_eff** = every grid point above (lake-only: 2+3+2+2=9 factor variants + 3
  ensemble subsets {F3,F4,F5}/{+F6}/{all}; with curves: +4) ≈ **12–16 registered trials**, plus
  correlation-deflation reported. DSR (per-observation Sharpe, n_obs = weekly obs on the traded
  span) must pass 0.95; **MinTRL table reported against the actual 690-week span** — this is why
  the history extension exists.
- PBO (CSCV) over the family ≤ 0.25; drop-top-N weeks (top-5 weekly returns removed) must stay
  positive; per-regime (pre/post 2020, rate regimes) sign stability; paired bootstrap of the
  ensemble vs its best single factor.
- **Adoption**: the DEPLOYED book is the pre-named ensemble {F3+F4+F5 (+F6 conditioner, +F1/F2 if
  acquired)} — not the best backtest cell. It ships only if the ENSEMBLE passes DSR+PBO+drop-top-N;
  otherwise the negative is recorded in docs/EDGE.md and nothing ships.

## Reporting

Per factor and ensemble: net Sharpe (weekly, annualized), CAGR@σ*, maxDD, hit rate, EV/name-week,
turnover, per-year Sharpes, CPCV mean-path + frac_pos, DSR/PSR/MinTRL, PBO, drop-top-5-weeks,
quarter/full-Kelly and DD-vs-leverage curve. Expectation restated: 0.5–0.7 (lake-only) / 0.85–1.15
(with curves); >1.5 = alarm, investigate as a bug.
