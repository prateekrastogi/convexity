# EDGE.md — what died, what survived, and what to do next

A decision memo, not a lab notebook. It records the whole convexity investigation so we never
re-run a dead experiment and so the one surviving edge is stated honestly, with its caveats. The
repo has been stripped to the minimum needed to continue that one edge (see `README.md`).

---

## TL;DR

- **Directional breakout / trend is DEAD.** Tested exhaustively across timeframes, exits, entry
  filters, leverage, three asset classes, and even inverted. Root cause is structural, not a
  tuning miss: **gross directional edge ≈ 0** — the signal carries no information about the next
  move, so net P&L is just friction bleeding out in R-units. Not deployable in any form.
- **Funding carry is real income but not deployable standalone.** ~+12.9%/yr gross carry, but it
  rides a ~70% structural drawdown (you are short the names that rip in a squeeze). Income overlay
  at most, never the book.
- **The one surviving edge is crypto CROSS-SECTIONAL MOMENTUM.** Weekly, dollar-neutral, quintile,
  vol-scaled, on the point-in-time crypto perp panel. Honest **net Sharpe ~0.8–1.0**, CAGR
  ~20–25% at 100% gross, maxDD ~25%, ~60% of weeks positive. Passed the full validation battery
  (purged-WF, block-bootstrap CI, holdout, per-regime, breadth, weighting, CPCV/PBO); DSR fails
  on TRACK LENGTH only (PSR high — see §(d), the biggest honesty flag).
- **Verdict: paper it, quarter-Kelly, deploy the 12–16d lookback BAND, not a single fitted L.**

---

## (a) What was tried — and DIED

### Directional breakout / trend (the original convexity thesis)

The founding thesis was "be long convexity: trade the breakout, let the stop be the edge" — cut
losers fast, ride the fat right tail. It was tested to exhaustion:

- **Timeframes**: 1m, 1h, 4h, daily.
- **Exits**: fixed-% take-profit, vol-based stop (ATR-scaled), no-stop / trend-death exit,
  fixed risk-reward brackets (RR ∈ {1.0, 2.0, 2.5, 3.0}).
- **Entry filters**: raw breakout, breakout-strength tercile, conviction/quality gating,
  trend-alignment (breakout only when it matches a Kalman local-trend), the 9-cell
  breakout-strength × trend-strength grid.
- **Leverage**: {3, 5, 10, 20, 50}× with liquidation modeled (position wiped when max adverse
  excursion hits 1/L − maint-margin, before the vol-stop can fire).
- **Asset classes**: crypto (50 USDT-M perps), US equities (30 Dukascopy underlyings), commodities
  (7 Dukascopy underlyings) — sleeve-aware (session gaps, assumed perp funding for non-crypto).
- **Inverted / fade**: the whole bracket rebuilt on the flipped side to fade the signal.

**Root cause of death — structural, not tunable.** Measured per-trade in R-units and net of
honest cost (fees + half-spread + impact + signed funding):

- **gross directional R ≈ 0** across every cell, timeframe, and asset class. The breakout signal
  simply does not predict the next move. There is no directional information to harvest.
- The realized loss is therefore **pure friction expressed in R** — the ~13bp round-trip cost the
  1-min harness pays thousands of times a year.
- **The "anti-predictive strong-breakout" anomaly was a cost artifact, not a fadeable edge.**
  Friction takes no side, so it is bit-identical FOLLOW-vs-INVERT; a friction-dominated FOLLOW
  loser only flips to an INVERT winner if a *real negative* directional edge exceeds the unchanged
  round-trip cost. It did not. Inverting a no-edge signal just pays the friction twice.

### Funding carry

- Short high-funding perps, long low/negative-funding perps, harvest the 8h credit.
- **~+12.9%/yr gross carry income — real, structural, not a prediction.**
- But it carries a **~70% structural drawdown**: you are systematically short exactly the names
  that squeeze hardest. Not deployable as a standalone book. Income-only overlay at most.

### Also refuted upstream (do not reintroduce)

Long-vol straddle/strangle, realized-vs-implied vol forecasting (HAR-RV, Lee-Mykland),
backspreads, OU mean-reversion exit rules. All tested in the `vol_research` / `arbitrage`
lineage and bled net-of-cost.

---

## (b) What WORKED — crypto cross-sectional momentum

A **relative ranking**, not a single-asset prediction. We do not forecast any coin's absolute
return; we bet recent relative winners keep out-ranking recent relative losers, dollar-neutral.
Market-wide beta cancels; what remains is the cross-sectional dispersion premium. Weekly rebalance
means the 13bp friction is paid ~52×/yr, not thousands of times — so it barely bites.

**Configuration**

- Weekly rebalance, dollar-neutral, **quintile** (long top 20% / short bottom 20%).
- **Vol-scaled** weighting (inverse-realized-vol within each leg) — beats liquidity-capped and
  plain equal-weight net of cost.
- Universe: **~50–75 point-in-time crypto names** (see breadth below).
- Signal: momentum lookback in the **12–16 day BAND** — NOT the single fitted 14d peak.

**Honest performance (the numbers that survive real cost)**

| Metric | Honest (real Amihud cost) | Headline (light-cost, peak-L) |
|---|---|---|
| Net Sharpe | **~0.8–1.0** | 1.6 |
| CAGR @ 100% gross | **~20–25%** | 65% |
| maxDD @ 100% gross | **~25%** (scales with leverage) | — |
| Win-rate | **~60% of weeks** | — |

The headline 1.6 Sharpe / 65% CAGR was light-cost and pinned to the single peak lookback. Under
realistic Amihud (size-aware) cost it reconciles to ~0.8–1.0 Sharpe / ~20–25% CAGR at 100% gross.
**Cite the honest column.**

---

## (c) Validation — the full battery it passed

This is the whole reason cross-sectional backtests lie, so the discipline was heavy:

- **Purged / embargoed walk-forward** on the weekly series (reuses SYSTEM `PurgedWalkForward`):
  no OOS collapse. Contrast: breakout *did* collapse OOS.
- **Deflated Sharpe** over the full config-trial count (the honest multiplicity).
- **Block-bootstrap CI** on the weekly returns.
- **Holdout** — a final untouched OOS slice.
- **Per-regime** — positive across all 5 crypto macro windows (2021 bull, 2022 bear incl.
  LUNA/FTX, 2023 recovery, 2024 bull, 2025–26).
- **Breadth / universe size** — net edge peaks around **60–75 names**; 100 does not honestly beat
  50, and 500 is *worse* (dilution + illiquid-tail cost). Bigger is not better here.
- **Weighting** — plain vol-scaled beats liquidity-capped net of cost.
- **CPCV / PBO** — survives combinatorial purged cross-validation; **PBO ≈ 0.39** (moderate;
  re-measured after the engine turnover-drift fix — pre-fix records read 0.36/0.979/0.906,
  post-fix 0.39/0.987/0.914; frac_pos 1.00 and haircut 0.07 unchanged). The
  takeaway drives the deployment rule: **deploy the lookback BAND, not one L**, because the single
  best L is partition-fragile. **`frac_pos` (fraction of positive OOS paths) is the robust signal;
  the p05 tail is partition-fragile** — trust breadth of positivity, not the worst-partition tail.

---

## (d) Honest caveats

- **The equity-momentum enhancement menu was tested and REJECTED (2026-07, pre-registered
  9-variant family: vol-target/crash-brake, absolute-momentum gate, residual momentum, FIP blend,
  skip, tranches, rank hysteresis).** Nothing cleared the pre-registered adoption bar
  (ΔCPCV ≥ +0.05): best was vol-targeting at +0.005 — the inverse-vol dollar-neutral book already
  self-stabilizes and the 25% DD brake never fired in 5yr; the TSMOM gate re-imports the dead §(a)
  directional signal (−0.25, maxDD 31.5%); skip=2 was the worst single change (−0.55 — no
  short-term reversal to dodge at weekly cadence). Family PBO 0.57–0.70: picking the "best"
  enhancement would itself be overfitting. Deployed config unchanged; the study artifacts were
  removed from the repo — this note is the record (raw per-variant results retained off-git in
  the VM's `_ckpt/enhance/results.json`).

- **Young track record (~5yr).** DSR *fails on length* — there simply are not enough weekly obs to
  clear MinTRL — even though PSR is high. This is the single biggest honesty flag.
- **Moderate PBO (0.39).** Real edge, but not a slam-dunk; overfit risk is non-trivial → the
  band-not-a-point rule is mandatory, not optional.
- **Crypto-specific.** The edge does NOT extend to equities or commodities (tested; those sleeves
  were directional-grade only). This is a crypto cross-sectional dispersion premium, full stop.
- **Size QUARTER-KELLY.** Conservative by construction given the above. Pair it with a
  **scale-up-on-live-confirmation** rule: only raise gross after paper/live re-confirms the edge
  out of sample.

---

## (e) Next

- **Status (2026-07): everything upstream of papering is DONE.** The 9 pillar specs are
  implemented and test-covered (613 tests), independently math-audited (a turnover double-drift
  fix re-measured the records: naive 0.987 / CPCV mean 0.914 / PBO 0.39, frac_pos and haircut
  unchanged), real Binance lot/tick/min-notional filters are wired, and the enhancement menu was
  tested and rejected (§(d)). The paper rebalancer is the ONLY remaining build item.
- **Paper the 12–16d lookback ensemble** (the band, vol-scaled quintile, ~60–75 names, weekly),
  dollar-neutral, at quarter-Kelly gross.
- **Open items / not yet done:**
  - Exact real-cost CAGR reconciliation (nail down the Amihud size-cost → the ~20–25% figure).
  - The **live rebalancer is NOT built** — this repo is research + validation only. The live/paper
    strategy slot is intentionally empty (see `src/convexity/` and the strategy note in
    `README.md`). Build it against NautilusTrader when papering starts; do not pre-build it now.

---

## Key numbers to remember

- Directional gross R ≈ **0** (dead everywhere).
- Funding carry **+12.9%/yr** gross / **~70%** structural DD (income-only).
- Xsec honest net Sharpe **~0.8–1.0**, CAGR **~20–25%** @100% gross, maxDD **~25%**, **~60%** weeks +.
- Universe sweet spot **~60–75** names; lookback band **12–16d**; **PBO 0.39**; ~5yr sample.
- Sizing: **quarter-Kelly**, scale up only on live confirmation.
