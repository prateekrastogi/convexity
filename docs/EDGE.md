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

- **PHASE PREMIUM at Friday (2026-07, measured; SOFTENED by the panel extension).** On the
  5yr panel the non-Friday phases looked dead ([−0.01 … +1.12]); on the extended 6.5yr panel
  every phase is POSITIVE — [Mon 0.52, Tue 0.53, Wed 0.45, Thu 0.82, **Fri 1.29**, Sat 0.75,
  Sun 0.51] — so "other phases dead" was substantially small-sample, while the Friday premium
  itself persists and still looks structural (full-week information at the Friday close). Friday
  was the a-priori convention, never fitted. Deploy at Friday as designed; paper confirms.
  Studies R1 + E1 (harness `research/xsec_robust.py`; raw results in the VM's `_ckpt/`;
  preregs + full write-ups in git history — the 7-phase ensemble was NOT adopted).
- **Real per-name cost holds the record** (study R2): dollar-volume spread +
  Amihud impact at $3M NAV shaves only ~0.02 Sharpe vs the flat 13bp (cost 3.8→4.1%/yr). A
  confirmation, recorded for completeness.
- **Validation now runs at embargo = 3 weeks** (study R3, LdP's embargo ≥ feature
  horizon): CPCV mean 0.95, frac_pos 0.93, PBO ≈ 0.40–0.43 — not an embargo-leak artifact; PBO
  stays moderate (band rule unchanged).
- **Track record: DSR now PASSES — but MARGINALLY (2026-07 panel extension).** Backfilling to
  the venue archive's origin (2020-01, study E1) gives n_obs = 334 weeks vs
  MinTRL ≈ 331: DSR 0.9505 ≥ 0.95, PSR 0.9995 — the long-standing #1 flag clears by ~2 weeks.
  Honest framing: a hair above the gate, not comfortably above it; every paper/live week widens
  the margin. The choices were all frozen before the 2020 data was downloaded, so the extension
  is backward-out-of-sample evidence, not fitting.
- **PBO drops to clean-pass territory on the extended panel: 0.107 at (8,2), 0.075 at (10,3)**
  (both < 0.2), with (6,2) at 0.40 still moderate — partition sensitivity remains, so the
  band-not-a-point rule STAYS mandatory. (5yr historical record: ≈0.39–0.43.)
- **Crypto-specific.** The edge does NOT extend to equities or commodities (tested; those sleeves
  were directional-grade only). This is a crypto cross-sectional dispersion premium, full stop.
  **Including INTRADAY (tested 2026-07, negative):** US-equity cross-sectional winners/losers held
  open→NY-close is DEAD net-of-cost in BOTH directions (25 mega-caps, 1,311 days, 3 signal windows
  × 3 cost levels): gross momentum ≈ 0 and sign-unstable; the one real gross pattern is midday
  REVERSAL (+4.6%/yr, Sharpe +0.84 gross) which even a 2bp round-trip fully consumes (−0.08 net;
  −1 to −5 at realistic 4–10bp — daily cadence pays ~250 RT/yr against sub-1bp/day gross effects).
  GHLZ market-intraday-momentum is also dead in-sample (QQQ −1.76 net). And the roster is
  survivorship-flattered, so these numbers are the OPTIMISTIC case. Re-runnable harness:
  `research/us_intraday.py`. **ORB (opening-range breakout + tight 1-min stop + ride-to-close —
  the unconfirmed NIFTY ITM-candidate construction) also tested 2026-07, negative:** on the
  mega-cap cross-section gross is NEGATIVE (−0.2 to −0.5bp/trade, 91% stop rate, net −1.8 to −2.6
  Sharpe); QQQ ≈ 0 net at even futures-grade 1bp costs; the single alive-looking cell (IWM OR15,
  +0.6 net Sharpe @1bp) fails every fragility diagnostic — parameter-fragile (OR30 on the same
  instrument is negative), year-unstable, DSR ≈ 0.5 over the 6-cell family, and the top 10 trades
  are ~91% of ALL gross (the same fat-tail luck-shape the NIFTY candidate is flagged with). The
  construction is a laddered lottery: ~90% of trades pay ~−10bp for a ~10% shot at ~+100bp — fair
  to negative net of costs on US instruments. Harness: `research/us_orb.py`.
- **Size QUARTER-KELLY.** Conservative by construction given the above. Pair it with a
  **scale-up-on-live-confirmation** rule: only raise gross after paper/live re-confirms the edge
  out of sample.

---

## (e) Next

- **Status (2026-07): the crypto book is PAPERING.** The 9 pillar specs are implemented and
  test-covered, independently math-audited (a turnover double-drift fix re-measured the records:
  naive 0.987 / CPCV mean 0.914 / PBO 0.39, frac_pos and haircut unchanged), real Binance
  lot/tick/min-notional filters are wired, and the enhancement menu was tested and rejected
  (§(d)). The **live/paper rebalancer is LIVE on the demo** (first supervised session 2026-07-03,
  weekly timers armed): `src/convexity/paper/`, spec `docs/specs/PAPER.md`, units `deploy/` —
  12–16d band ensemble, dollar-neutral, quarter-Kelly gross, USDT→USDC waterfall (BTC never),
  maker-first orders + finalize crossing + the domain-scoped orphan sweep, TWR flows, daily recon.
- **Paper's mission:** measure maker fill rates (the +1.75%/yr E2 upside), the Friday phase
  premium OOS, and accumulate DSR track weeks.
- **Open items:** exact real-cost CAGR reconciliation (nail down the Amihud size-cost → the
  ~20–25% figure).

---

## (f) The commodity book — three programs, one survivor (2026-07-05)

The 7 Binance commodity/metal TradFi perps (XAU/XAG/XPT/XPD/COPPER/CL/NATGAS) were attacked with
three pre-registered programs on the databento-futures + stitched-proxy lakes
(`research/entry_exit/study/`):

1. **Swing cross-sectional factors — DEAD** (`REPORT.md`): carry, basis-momentum, COT-reversal,
   momentum, skew, hedging-pressure and ensembles ALL net-negative; breadth 7 (effective ~4,
   4/7 degenerate-carry metals) cannot feed a quantile book.
2. **Durable slow TSMOM + carry-on-storables — DEAD net** (`DURABLE_REPORT.md`): thesis direction
   confirmed (slowest trend L52 +0.15 and carry +0.14 are the only positive cells) but magnitude
   sub-noise — combined book 0.09, DSR 0.08, PBO 0.54, drop-top-5 negative. Same breadth wall.
3. **RV pair mean reversion — the SURVIVOR** (`RV_REPORT.md`): 5 pre-named pairs, band z
   {26,52,104}wk, half-life gate, pair RP, 10% vol target → **net 0.73 @12bp (0.54 @24bp),
   maxDD 18.5%, drop-top-5 +0.57 (first-ever pass on this universe), CPCV 28/28 paths positive**;
   DSR 0.856 and PBO 0.386 miss the frozen full-pass bar → **no real-money deploy; graduated to a
   PAPER book instead** (plan-around 0.5–0.6 live; nearly uncorrelated diversifier to the crypto
   book, not a co-equal).

The RV book is implemented as a second, fully separated paper strategy (`docs/specs/PAPER.md`
§11: disjoint symbol domains, own journal + virtual capital via `allocate`, domain-scoped orphan
sweep, own timers) with weekly commodity-perp discovery + data refresh. Sizing v1 = the study's
own 10% vol target — **quarter-Kelly is deferred until the paper track exists**; adopt only if
paper tracks the backtest. This CLOSES commodity-directional research on this universe; the only
path to more commodity Sharpe is breadth (a 25–40-market futures venue), not another signal.

---

## Key numbers to remember

- Directional gross R ≈ **0** (dead everywhere).
- Funding carry **+12.9%/yr** gross / **~70%** structural DD (income-only).
- Xsec honest net Sharpe **~0.9–1.1** on the extended 6.5yr panel (band ensemble 1.29 naive at
  Friday; plan conservatively), CAGR **~20–25%** @100% gross, maxDD **~25%**, **~60%** weeks +.
- Universe sweet spot **~60–75** names; lookback band **12–16d**; **PBO 0.11** [(8,2), extended;
  5yr historical 0.39]; **DSR passes 0.95 at 334wk vs MinTRL 331** (marginal); Friday phase
  premium persists (all phases positive on 6.5yr).
- Maker-first execution upside ~+1.75%/yr cost recovery (E2 scenario, contingent on paper fill
  rates); 85/15 carry sleeve tested and NOT adopted (+0.033 < +0.05 bar).
- Sizing: **quarter-Kelly**, scale up only on live confirmation.

---

## (g) The equity arc (2026-07): five certified negatives, one owner-decision timed-beta book

The TradFi-equity-perp sleeve (110 Binance perps, ~93 usable with real 2018+ underlying history)
got the full certification treatment — five independently pre-registered, independently
verified, adversarially reviewed programs. Every one closed negative. Records:
`research/equity/` (E1/E2/E3 + `STATE.md`'s consolidated bug list), `research/indices/` (I1),
`research/cta/` (C1 + the D1 drawdown-overlay record).

- **E1 perp-native** (funding carry, basis MR, overnight structure): deployable −1.7 net,
  DSR=0. Overnight>intraday is REAL gross (SPY 3.44 vs 1.11) but dies at 13 bp × daily
  round-trips. Funding/weekend cells were underpowered (~4 months of perp history), not
  disproven — **revisit trigger ~mid-2027** with 12+ months of perp era.
- **E2 sector-residual reversal** (also the equity RV study — named pairs deliberately scoped
  out as a documented anti-edge): −0.95 net, DSR=0. Found strong 3–10-day residual reversal —
  which is why any equity momentum needs a skip-month.
- **E3 momentum + BAB + PEAD** (monthly, earnings dates acquired + validated): combined book
  fails all gates. S1/S2 look positive standalone but are FAKE BREADTH — S1 is 40% GameStop;
  S2's whole edge is LLY+NVO (drop 2 of 93 names → negative). The defining equity-sleeve
  lesson: apparent edges here concentrate into 2–4 idiosyncratic stories.
- **I1 SPY/QQQ timed-beta overlay** (trend + vol-target + calendar tilts; benchmark-relative
  gate): 0.69 net Sharpe vs buy-and-hold's 0.77 — the overlay's drawdown protection cost more
  than it saved in a bull-heavy sample. Factor decomposition: the TREND filter alone beat the
  benchmark (0.85); vol-targeting and tilts subtracted.
- **C1 mathematically-optimal CTA** (Kalman/l1-trend/HMM/GARCH/Kelly/Grossman-Zhou/CDaR, a
  29,760-cell exploratory grid + frozen confirmatory stage): fails all numeric gates; the
  winner sat at the 99.96th percentile of its own search grid (DSR 0.13–0.19) — sophistication
  found the CEILING, not an edge. Its lasting product: the leverage mathematics. 2–5× average
  leverage is jointly INFEASIBLE with any sane drawdown budget on this universe (in-band best:
  ~46% CAGR at 62% maxDD); Sharpe is leverage-invariant under linear costs, so CAGR@DD-budget
  is the only honest leverage objective.
- **D1 drawdown overlays** (the constructive coda): on the simple 3-rule trend ensemble at
  frozen half-Kelly, a pre-declared 2-step waterfall (NO grid search; DSR@N=4 0.987) found
  **GZ-brake@50% + breadth-proportional gross**: CAGR 28.26% / maxDD 30.02% / **Sharpe 1.17 >
  the unlevered base's 1.06** at ~1.11× realized leverage, 21/21 CPCV paths positive,
  independently reproduced bit-for-bit. The #5 index-gate addon FAILED its one-shot
  pre-declared bar and was excluded. Regime honesty: protection is real where trend gets time
  (2022: −21% vs benchmark −35%) and absent in fast crashes (2020-03); the 93-name equal-weight
  benchmark (~41% CAGR, 41% maxDD, survivorship-flattered) still wins on raw Sharpe in this
  bull-heavy sample.

**Deployment (PAPER.md §12) is an OWNER DECISION, not a certification**: the research verdict
on this universe stands negative; the owner deploys the D1-validated construction as timed beta
— survivorship-curated universe accepted as a feature, bull-market lag accepted as the premium —
at 250/1000, parity-pinned (Invariant EQUITY-PARITY) so the live book provably IS the validated
one. The five programs also produced the fleet's standing engineering discipline: the five
carried-forward bug patterns (source-aware returns, derived annualization, exposure-timing
alignment, exposure-scaled costs, leveraged-ETF exclusion — `research/equity/STATE.md`), the
synthetic-outlier injection test with naive control, and the parity-gate pattern.
