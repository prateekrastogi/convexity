# E3 CROSS-SECTIONAL EQUITY FACTOR SUITE — PRE-REGISTRATION (Program 7; FROZEN 2026-07-07, before
any pipeline code)

Owner-directed follow-up to E1 (perp-native suite, closed — clean negative) and E2 (stress-
conditioned residual reversal, closed — clean negative). Isolation: runs in the `equity` worktree
(/home/speculation/convexity-equity); reads the shared on-disk lake at `data/stitched/` and
`data/earnings/` (both symlinked into the same lake as `main` and `commodity` — reads only);
NEVER touches the live paper system (main's checkout, its branch, its systemd timers, or its
`data/paper/` journals) or `src/convexity`'s live-trading paths. Executed by `e3_pipeline.py` (new,
self-contained file under `research/equity/`); outcome to be recorded in `E3_REPORT.md`.

## Thesis (each INCLUDED sleeve, 1-2 sentences)

**S1 slow sector-residual momentum.** Sector-demeaned returns that have trended over a 3-12 month
formation window, with a mandatory skip month to avoid the short-horizon residual reversal E2 just
documented in this same data, should continue to trend over the next month — classic
Jegadeesh-Titman cross-sectional momentum, implemented sector-residually (not raw) so the book is
not just relevering a market-beta or sector-beta trade.

**S2 betting-against-beta (low-volatility).** Frazzini-Pedersen leverage-aversion logic: capital-
constrained investors overpay for high-beta names to get leveraged-like exposure without using
leverage, compressing the low-beta names' expected-return-per-unit-of-beta below and the high-beta
names' above what CAPM predicts, so a beta-neutral book long-low-beta/short-high-beta (each leg
leverage-adjusted to true zero net beta, not just dollar-neutral) should earn a positive risk-
adjusted return.

**S3 price-based post-earnings-announcement drift (PEAD), INCLUDED.** S3 is in scope because the
Data Acquisition phase's coverage_fraction = 0.8172 (76/93 base-universe names have >=8 usable
historical announcement-date events in [2018-01-01, 2026-07-04]) clears the declared 0.60 program-
level floor by a wide margin — this determination was made and fixed BEFORE this document was
frozen, per the task's own discipline, and is not revisited here. Thesis: the market
under-reacts to the information in an earnings surprise, so an abnormal announcement-window return
predicts same-sign drift over the following weeks (Bernard-Thomas), and this is tested here as a
pure PRICE-based signal (the announcement-window abnormal return itself, not an analyst-consensus
EPS-surprise number, since no EPS-estimate feed is part of this data contract).

**Program is a THREE-sleeve program** (S1 + S2 + S3), not a two-sleeve fallback — the S3 go/no-go
decision already resolved to "proceed" and is treated here as a fixed input.

## Base universe (frozen, verified fresh via a real `load_universe()` call — landmine 5)

Starting point: 110 equities entries in `src/convexity/panel/candidates.md`, 102 with a non-null
`dukascopy` field (real databento underlying history; the other 8 are perp-only: ANTHROPICUSDT,
BBXUSDT, BSPUSDT, HYUNDAIUSDT, OPENAIUSDT, QNTXUSDT, SAMSUNGUSDT, SKHYNIXUSDT, excluded from all
three sleeves entirely — S1/S2/S3 all require real underlying-history point-in-time daily closes).

**Step 1 — short-history exclusion** (re-verified fresh against `DATA.md`'s manifest, matches
E2's independently-confirmed 99): STXXUSDT (0.19yr), CBRSUSDT (0.14yr), DRAMUSDT (0.25yr) excluded
— too short to clear even S1's shortest formation window (3 months) plus its skip month plus a
vol-window buffer. Gives **99 usable names**.

**Step 2 — leveraged/inverse/decay ETF exclusion** (landmine 5, mandatory, ALL sleeves; fresh
`load_universe()` call run in this session, sector tags enumerated directly, not from memory —
confirmed live on 2026-07-07):

| Symbol | Sector tag | Product |
|---|---|---|
| UVXYUSDT | `vol-etf` | 2x-notional long-vol-decay; two real reverse splits (2018-09-18 -5x, 2023-06-23 -10x) |
| SQQQUSDT | `inverse-etf-nasdaq` | -3x inverse Nasdaq; reverse split 2019-05-24 -4x |
| TQQQUSDT | `leveraged-etf-nasdaq` | 3x long Nasdaq; splits 2018-05-24 3:1, 2021-01-21 2:1 |
| SOXLUSDT | `leveraged-etf-semis` | 3x long semis |
| KORUUSDT | `thematic` (mislabeled) | Direxion Daily South Korea Bull 3X Shares — real 3x-leveraged single-country ETF, 2025-02-10 -10x reverse split confirmed via Direxion's own press release |
| MVLLUSDT | `thematic` (mislabeled) | GraniteShares 2x Long MRVL Daily ETF — single-stock 2x daily-reset leveraged product on Marvell Technology |

Two of the six (KORUUSDT, MVLLUSDT) carry the generic `thematic` tag, not a `leveraged-etf-*`/
`inverse-etf-*`/`vol-etf` tag, and were found only by judgment/external verification against the
real underlying security — a naive tag-substring filter would miss both. Borderline cases
explicitly checked and NOT excluded: STRCUSDT (Strategy Inc.'s Series A Perpetual Stretch
Preferred Stock, a real preferred-equity security of an operating company — its 2023-07-06 -5x
reverse split is a genuine one-time corporate action, not mechanical daily decay); KSTRUSDT
(KraneShares SSE STAR Market 50 Index ETF, a plain unlevered index tracker); BBXUSDT/BSPUSDT
(perp-only synthetic thematics, already excluded on data-availability grounds in Step 0, not
double-counted here). STXXUSDT is also a genuine leveraged product (Tradr 2X Long STX Daily ETF)
but is already excluded on short-history grounds in Step 1, not double-subtracted. MSTRUSDT and
other bitcoin/thematic-proxy operating companies are real companies and are explicitly NOT
excluded by this rule.

**Resulting base universe for S1 and S2: 99 - 6 = 93 names.** Verified live in this session:
`load_universe()`'s 102-with-`dukascopy` set, minus the 3 short-history names, minus the 6
leveraged/inverse names, counts to exactly 93.

**S3's own universe is a subset of the 93**: of the 93, 76 clear the >=8-usable-announcement-event
floor (`research/equity/e3_pead_data_coverage.json`, `usable_ticker_count=76`,
`coverage_fraction=0.8172`); the excluded 17 = 11 ETFs/index/thematic products that file no
quarterly earnings at all (EWJ, EWT, EWY, EWZ, IWM, KSTR, QQQ, SPCX, SPY, URNM, XLE — all
correctly zero-event, not a coverage failure) + 6 genuine 2025/2026-vintage IPOs with 1-5 real
events (BMNR, CRCL, CRWV, SNDK, USAR, STRC) whose full trading history does not yet contain 8
quarterly events regardless of source.

## Demo-gate reporting (mandatory, landmine 5 / lesson 7 — full-universe AND on_demo, recomputed
fresh, never reused from E1/E2)

`on_demo=true` count verified fresh via `load_universe()` in this session: **19** names (AMZN,
COIN, CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR, MVLL, NVDA, OPENAI, PAYP, PLTR, QNTX, QQQ, SPCX,
SPY, TSLA). After this program's own exclusions (OPENAI/QNTX perp-only, MVLL now excluded as a
leveraged single-stock ETF under landmine 5 — note this is a CHANGE from E2's on_demo-17, since E2
did not apply the leveraged-ETF exclusion to MVLL), the **on_demo-and-real-history-and-not-
leveraged subset for S1/S2 is 16 names**: AMZN, COIN, CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR,
NVDA, PAYP, PLTR, QQQ, SPCX, SPY, TSLA. For **S3**, intersecting this 16 with the 76 PEAD-usable
names (EWJ/QQQ/SPCX/SPY are non-earners, CRCL is below the 8-event floor) gives **11 names**: AMZN,
COIN, GOOGL, HOOD, INTC, META, MSTR, NVDA, PAYP, PLTR, TSLA. Every headline number in `E3_REPORT.md`
is reported against BOTH the full base universe (93 for S1/S2, 76 for S3) AND its respective
on_demo subset (16 for S1/S2, 11 for S3) — the 93/76/16/11 figures are the authoritative
denominators for this program, not E1's or E2's own prior on_demo-17/19 figures, which do not
apply here because this program's own leveraged-ETF exclusion (MVLL) changes the on_demo
intersection.

## Survivorship (mandatory, headline number in every section, NOT one blanket caveat — landmine 6)

- **S1 (momentum) — MOST SEVERE.** The 93-name roster is today's Binance winners. For momentum
  specifically this is close to worst-case: survivors are mechanically enriched with past winners
  (a name that trended up persistently is more likely to still be listed, popular, and liquid
  enough to have made it onto this venue's roster today), which can inflate a momentum backtest's
  apparent Sharpe more than almost any other cross-sectional anomaly — a name that trended into
  irrelevance or delisted is simply absent from the universe entirely, so the momentum book never
  gets to record its own worst-case losing tail. Stated with the 93 headline number (and 16 for
  on_demo) in every S1 results section, not once in an appendix.
- **S2 (BAB) — MODERATE, explicitly NOT nil.** Beta-sorting does not mechanically enrich for past
  winners the way momentum does (a low-beta name and a high-beta name are equally likely to have
  survived to today's roster on beta grounds alone), so this is less severe than S1. It is still
  nonzero: the 93-name roster is still today's popularity-selected list, not a point-in-time index
  membership, so names that would have ranked in the extreme beta deciles but later delisted or
  fell out of relevance are absent from the historical cross-section entirely. Stated with the 93/
  16 headline numbers in every S2 results section.
- **S3 (PEAD) — MODERATE, doubly caveated.** Same base-roster survivorship as S2 (93-name universe
  is today's winners), PLUS a second, independent layer specific to S3: the announcement-DATE
  history itself is collected only for today's 76-name roster — a name that delisted or was
  acquired before 2026-07-04 would contribute zero announcement events to this dataset even if it
  had a real, tradeable PEAD history while it existed. Both layers stated explicitly with the 76/
  11 headline numbers in every S3 results section.

## AVOIDING E1'S AND E2'S FIVE NAMED BUG PATTERNS (mandatory, declared before any code is written
— DATA_CONTRACT landmines, verified fresh in this document, not assumed inherited)

**(1) Source-aware returns.** Every `data/stitched/{SYMBOL}.parquet` read for S1/S2's core signal
and S3's abnormal-return construction filters to `df[df["source"] == "duka"]` BEFORE any
resampling, ranking, or return computation — reusing E2's fix verbatim (E2's own
`verify_no_perp_rows_in_signal_panel()` check, independently confirmed correct in E2's own
verification, is reused as-is: it asserts every row consumed by the daily-close builder has
`source=='duka'` and spot-checks a known splice-seam name (PAYPUSDT) never contributes a
`perp`-sourced close). `e3_pipeline.py` writes its own small source-aware daily-close builder
(same construction as E2's, not `convexity.panel.panel.build_panel()`/`load_bars()`, since
`load_bars()` does not read the `source` column at all — verified in E2's own prereg drafting and
re-confirmed as still true, not re-verified independently here since no `src/` file has changed
since E2 froze). This applies identically to all three sleeves: S1/S2's daily-close panel and S3's
announcement-window return are both built from the same source-filtered daily-close series.

**(2) Derived annualization.** `periods_per_yr = n_obs / span_years` computed fresh from each net-
return array's own actual date index, asserted near the expected cadence constant, printed as an
auditable diagnostic, and passed (never a hardcoded literal) into every `metrics()`/`dsr()`/
`cpcv_paths()`/`per_year_sharpe()` call — reusing `derive_periods_per_yr()`'s exact pattern from
E2. S1 and S2 are MONTHLY-rebalanced and monthly-marked (one net-return observation per calendar
month), so `periods_per_yr` is expected to derive to roughly 12 — verified, not assumed, exactly
as E2 verified its own ~52 expectation rather than trusting it. **S3 needs its OWN derivation,
separately stated**: S3's P&L is an OVERLAPPING event-portfolio (multiple names' 2/4/6-week
holding windows are live simultaneously and roll on their own announcement-driven calendars, not
a single shared monthly/weekly grid), so the marked P&L array's actual observation spacing is
whatever the realized rebalance-and-mark calendar turns out to be once the event-portfolio is
actually constructed — `e3_pipeline.py` derives `periods_per_yr` from THAT array's real date
index exactly like every other cell, with no assumption in advance that it lands near any
particular constant (52, 12, or otherwise); the derived value and its diagnostic (`n_obs`,
`span_years`) are printed and shown in `E3_REPORT.md` for S3 specifically, not inherited from
S1/S2's monthly convention.

**(3) Exposure-timing alignment.** Any time-varying gross-exposure array (S2's beta-leverage
adjustment, the combined book's vol-target overlay) is index-aligned to each return observation's
START date, not its END date — before trusting any such construction, `e3_pipeline.py` prints a
diagnostic tracing which calendar date each array index corresponds to for BOTH the return series
and the exposure-scale series, and this is inspected by hand before any headline number is
trusted. Concretely: S2's beta estimate used to size month-`t`'s position is estimated from data
strictly BEFORE month `t`'s rebalance date (rolling OLS beta computed through the PRIOR week's
close, never including the week the position is about to earn), and the combined book's 10%-vol-
target overlay scale for month `t` is computed from trailing volatility measured strictly before
month `t` begins (reusing `regime/overlay.py::vol_target_scale`'s `net[lo:t]` half-open-interval,
excludes-`t`-itself convention, and `metrics()`'s own internal `vol_target()` helper's `.shift(1)`
pattern — both already-correct templates identified during Setup). This is the exact bug class
that survived E2's own dedicated pre-run smoke-testing and a first-pass independent verification,
and was caught only by a third, dedicated adversarial lookahead pass — `e3_pipeline.py` treats
that as a standing instruction to re-derive, not re-trust-by-analogy, every exposure/return
date-alignment claim from scratch.

**(4) Costs scale with realized exposure.** Every cost component — the trading-fee term (already
correctly turnover-scaled, i.e. `turn_scaled = turn_raw * es`) AND the funding/borrow term — scales
with the book's REALIZED notional/leverage for that period, not a flat per-period constant,
whenever the combined book's vol-target overlay varies gross exposure over time. This is a direct,
explicit fix of E2's documented bug (E2's funding term was a flat `funding_bp_per_wk/1e4`
regardless of the stress-exposure scale `es`, while its fee term correctly scaled) — `e3_pipeline.
py`'s `apply_costs()` multiplies BOTH the fee term and the funding term by the same realized
exposure-scale array `es` for the combined book's cost/funding grid, and this is verified by an
explicit unit check (funding-scenario deltas must themselves scale with `es`, not appear as an
`es`-independent flat shift, before any cost/funding number is trusted).

**(5) Leveraged/inverse/decay ETF exclusion.** Declared and applied in full above (Base universe
section) — UVXYUSDT, SQQQUSDT, TQQQUSDT, SOXLUSDT, KORUUSDT, MVLLUSDT excluded from the base
universe for ALL THREE sleeves, resulting base universe = 93 names (76 for S3 after the coverage
floor). This directly closes E2's own documented dependence finding (UVXYUSDT alone = 20.2% of
E2's book's total gross P&L, top 4 names >50%) — none of the six excluded names can re-enter
through S1's momentum ranking, S2's beta-decile sort, or S3's earnings-surprise ranking in this
program.

## S1 — slow sector-residual momentum (frozen)

- **Universe**: the 93-name base universe, point-in-time from each name's own first real
  `source=='duka'` day (no backfill before real listing).
- **Panel**: monthly close per name (last real `duka`-sourced daily close of each calendar month),
  point-in-time masked to each name's own first/last real `duka` day.
- **Sector residualization**: within each month's point-in-time universe, group by `sector`
  (`candidates.md` field); groups with <4 point-in-time members that month fold into a single
  "other" bucket for that month, recomputed fresh every month as the universe grows (reusing E2's
  per-day group-membership convention, adapted to monthly). `resid[i,t] = ret[i,t] -
  mean(ret[j,t] for j in group(i,t))`.
- **Formation grid (frozen, this IS the multiplicity ledger)**: lookback L in {3, 6, 12} months.
  Each lookback uses formation window `[t-L-1, t-1]` months with a MANDATORY skip of >=1 month
  before the rebalance date (skip month = `[t-1, t]`) — i.e. the signal at rebalance date `t` is
  the cumulative sector-residual return from `t-L-1` months ago through `t-1` months ago, never
  including the most recent completed month. **Reasoning, stated explicitly**: E2 found strong
  sector-residual REVERSAL at 3-10 trading-day horizons in this exact data — a skip-less momentum
  signal computed straight through to `t-1`'s close would still contain that short-horizon reversal
  contamination working AGAINST momentum's own sign in its most recent weeks, understating the
  true momentum signal. The skip month removes exactly that contaminated window.
- **Ensemble**: equal-weight band ensemble of the three lookbacks — NEVER argmax.
- **Sign check, stated explicitly** (the single easiest catastrophic error given the construction's
  similarity to E2's dead reversal sleeve): this is MOMENTUM, the OPPOSITE sign from E2. Ranking &
  book: **LONG the top quintile (biggest sector-residual winners) / SHORT the bottom quintile
  (biggest sector-residual losers)** — verified explicitly against E2's `_weights_from_signal`
  convention (which longs the BOTTOM quintile for reversal); S1's construction must rank-sort the
  SAME way and then assign the OPPOSITE quintiles to long/short, or equivalently negate the signal
  before calling the same weight function. This sign is checked by an explicit unit assertion in
  `e3_pipeline.py` before any P&L is computed, not left to visual inspection of a table.
- **Position sizing**: inverse trailing-12-month realized vol within each leg, floored at 0.5x the
  cross-sectional median vol that period (reusing `_weights_from_signal`'s floor convention).
  Dollar-neutral, unit gross before overlay.
- **Rebalance**: MONTHLY, last trading day of the calendar month (exact day convention, reusing
  each name's own last real `duka`-sourced trading day of the month).

**S1 grid cells (multiplicity ledger)**: 1 ensemble + 3 per-lookback (L=3, L=6, L=12) = **4 cells**.

**S1 regime sanity (own signature, MANDATORY, DIFFERENT from S2/S3)**: momentum's own documented
failure mode is a MOMENTUM CRASH during a sharp reversal off a bottom — 2009 and 2020-04 (the
COVID V-shaped snapback) are the classic episodes, and 2020-04 is within this data's span.
**Declared check**: does the honest S1 pipeline show a visible, identifiable drawdown/crash episode
around 2020-04, consistent with the documented phenomenon? `E3_REPORT.md` reports the ACTUAL
monthly return and drawdown numbers for the S1 ensemble in and around 2020-04 (not just a yes/no
verdict) — a momentum book showing NO stress episode there is itself flagged as suspicious
(candidate causes: a construction bug, an insufficiently-severe formation/skip window, or the
sector-residualization removing too much of the market-wide snapback for the crash to show up in
the RESIDUAL momentum signal specifically — this last possibility is explicitly not assumed away,
since S1 trades sector-residual returns, not raw returns, and the well-documented 2020 momentum
crash is usually described in raw/market-relative terms, not sector-residual terms; if the
residual construction genuinely dampens or eliminates the crash signature, this is reported as a
finding about this specific construction, not silently treated as a pass).

## S2 — betting-against-beta (low-volatility) (frozen)

- **Universe**: the 93-name base universe, same point-in-time discipline as S1.
- **Market proxy**: EQUAL-WEIGHT of the 93-name base universe's own weekly log-returns (point-in-
  time — a name only enters the proxy average from its own first real `duka` day), NOT SPY/QQQ
  alone. Reasoning, stated explicitly: SPYUSDT and QQQUSDT are themselves constituents of this
  universe, and using either alone as the market proxy would make the beta estimate dominated by
  two mega-cap-index-ETF-like names rather than reflecting true systematic risk across the panel —
  the equal-weight-of-base-universe proxy is the more faithful "the market this book actually
  trades against" construction.
- **Beta estimation**: rolling OLS beta of each name's weekly log-return against the equal-weight
  market proxy. **Beta window grid (frozen, this IS the multiplicity ledger)**: W_beta in {26, 52,
  104} weeks — 3 variants, each independently estimated and reported; no ensemble declared for S2
  (the task's own S2 spec does not call for a band-ensemble the way S1/S3 do — each beta window is
  its own cell). **Primary window, declared now**: W_beta=52 (one year) is the primary/headline
  cell — long enough to average out weekly noise in the OLS estimate without spanning multiple
  business cycles' worth of regime-varying beta the way W_beta=104 would, and matches the
  conventional "trailing-year beta" convention most commonly cited in the BAB literature itself
  (Frazzini-Pedersen's own paper uses 1- and 5-year rolling betas; 52 weeks is this study's
  practical middle-ground analog given the ~8-year span). W_beta=26 and W_beta=104 are reported as
  robustness variants, not the pick.
- **Rebalance**: MONTHLY, same last-trading-day-of-month convention as S1. At each rebalance, rank
  all point-in-time names by estimated beta (as of the OLS window ending strictly before the
  rebalance date — see AVOIDING BUGS (3) above), form low-beta (long) and high-beta (short) decile
  portfolios.
- **TRUE Frazzini-Pedersen construction, NOT naive dollar-neutral (declared exact order of
  operations)**: (a) each leg is FIRST leverage-adjusted so its own beta contribution is
  normalized — long-leg weight scaled by `1/beta_long_leg` (the long leg's own realized average
  decile beta), short-leg weight scaled by `1/beta_short_leg` — this is applied BEFORE the
  inverse-vol sizing step; (b) inverse trailing-12-month realized-vol sizing is then applied WITHIN
  each already-beta-leverage-adjusted leg, floored at 0.5x the cross-sectional median vol that
  period (same floor convention as S1); (c) the two legs' dollar exposures are then balanced so
  the COMBINED book's own ex-ante beta (computed from each leg's leverage-adjusted weight times its
  own beta) is target-zero. **Order-of-operations justification, stated now**: beta-leverage
  adjustment is applied first because it is the economically load-bearing step (this is what makes
  the construction "BAB" rather than a naive long-low/short-high book); vol-scaling is applied
  second, within each already beta-neutralized leg, purely to prevent a handful of low-vol or
  high-vol outlier names from dominating a leg's own internal weight distribution — applying
  vol-scaling first and beta-adjustment second would let a name's raw volatility distort the very
  beta-neutrality the construction is supposed to guarantee.
- **MANDATORY adversarial check (frozen, must be run and reported as an actual number)**: the
  resulting book's REALIZED ex-post beta — regression of the book's own monthly return series
  against the same equal-weight market proxy, over the full sample — is computed and reported
  explicitly in `E3_REPORT.md`. If this is not close to zero, this is flagged PROMINENTLY as a
  construction defect (a disguised leveraged short-market bet), not silently called "BAB." This
  check applies to the PRIMARY (W_beta=52) cell as the headline, with the other two windows'
  realized ex-post betas also reported for comparison.

**S2 grid cells (multiplicity ledger)**: 3 beta-window variants (W_beta=26, 52, 104) = **3 cells**.

**S2 regime sanity (own signature, MANDATORY, DIFFERENT from S1/S3)**: BAB's documented behavior
is a fairly steady, comparatively low-drawdown return stream that UNDERPERFORMS during strong
momentum/melt-up markets and HOLDS UP OR OUTPERFORMS during risk-off/deleveraging episodes.
**Declared check**: across the same four named stress windows used in E1/E2 — 2018 Q4, 2020-03
COVID, 2022 drawdown, 2024-08 vol spike — does S2's (W_beta=52 primary) performance show resilience
(not necessarily outperformance, but not a matching-or-worse crash than the broad 93-name equal-
weight book) in each window? `E3_REPORT.md` states the ACTUAL per-window S2 return/drawdown number
AND the broad-book comparator's own number for the same window side by side (not just a yes/no
verdict per window).

## S3 — price-based post-earnings-announcement drift (PEAD) (frozen)

- **Universe**: the 76-name PEAD-usable subset of the 93-name base universe (per the Data
  Acquisition coverage outcome above); a name with fewer than 8 usable announcement events over
  the sample is excluded from S3 specifically (already applied in deriving the 76) — such a name
  may still appear in S1/S2.
- **Announcement dates**: `data/earnings/{SYMBOL}.csv` (82 cached files, 76 of which meet the
  coverage floor; ticker-mapping edge cases PAYPUSDT->PYPL and BRKBUSDT->BRK-B already resolved in
  the cached data per the Data Acquisition phase's own notes), with BMO/AMC timing derived from
  each event's Eastern-local timestamp where available (`>=16:00 ET` -> AMC, `<09:30 ET` -> BMO,
  else UNKNOWN — per the cached data's own documented convention).
- **"Surprise" proxy**: the announcement-window ABNORMAL return — the name's own return minus its
  sector-group's mean return (same per-period sector grouping/fold-to-"other" convention as S1/S2,
  computed at daily granularity for this window) over the window from close of `t-1` to close of
  `t+1`, where `t` is the announcement date; for confirmed-AMC announcements the window is close of
  `t` to close of `t+1` (the AMC report happens after `t`'s close, so `t-1`-to-`t+1` would include
  a day of pre-announcement return that is not part of the surprise window); if BMO/AMC timing is
  UNKNOWN for a given event, the conservative `t-1`-to-`t+1` window is used and this is a declared
  convention, not a silent default.
- **Signal**: the sign and magnitude of the announcement-window abnormal return.
- **Entry**: close of the FIRST tradeable day AFTER the full announcement window has closed — no
  same-window entry, since the window's own return is not knowable until it closes. This is itself
  a lookahead-avoidance discipline, stated explicitly: a position entered ON the window's last day
  (rather than the day after) would require knowing that day's own close before the window is
  scored, which is circular.
- **Position**: LONG names in the top tercile of announcement abnormal return that period, SHORT
  the bottom tercile — drift direction = same sign as the surprise, per the PEAD literature (this
  is a momentum-sense construction in the surprise dimension, analogous to S1's sign but on a
  completely different, event-driven time axis — not to be confused with or reused from S1's
  monthly formation logic).
- **Holding-period grid (frozen, this IS the multiplicity ledger)**: {2, 4, 6} weeks, deployed as
  an EQUAL-WEIGHT BAND ENSEMBLE (exit at horizon, no stop-loss) — never argmax a single holding
  period.
- **Portfolio construction**: overlapping events across names form a rolling event-portfolio.
  Equal-risk (inverse trailing-vol) sizing PER LIVE EVENT POSITION, book gross normalized to unit.
  Dollar-neutral only to the extent the long/short event counts allow at any given time — the
  realized average net exposure is reported HONESTLY in `E3_REPORT.md` (event books are rarely
  perfectly balanced at every moment; neutrality is NOT forced by leveraging up the thinner side —
  the imbalance is reported as a real, disclosed characteristic of this construction, not hidden).
- **Costs**: same 13bp primary / {6.5bp, 26bp} scenarios + funding scenarios (0, +5bp/wk, -5bp/wk)
  as S1/S2. PEAD's turnover is event-driven (~4 events/name/year x 2 trades each per name, roughly
  ~8 trades/name/year expected) — `E3_REPORT.md` reports the realized `turn_mean` against this
  expectation and flags any large deviation as a potential construction issue worth investigating
  before trusting the cost-adjusted number.

**S3 grid cells (multiplicity ledger)**: 1 ensemble + 3 per-holding-period (2wk, 4wk, 6wk) =
**4 cells**.

**S3 regime sanity (own signature, MANDATORY, DIFFERENT from S1/S2)**: the PEAD literature's own
documented profile is that the drift is strongest in smaller/less-covered names and has DECAYED in
large caps since roughly 2010 in the US market. **Declared check, two parts**: (a) split the
76-name S3 universe by trailing dollar volume (top-half vs bottom-half, ranked at each event date
from the prior 20 trading days' `close*volume`) and confirm the drift is concentrated in the
less-liquid half — `E3_REPORT.md` reports the ACTUAL Sharpe/EV-bp for both halves side by side; if
the drift is NOT concentrated in the less-liquid half, this is stated plainly, not explained away.
(b) per-year Sharpes are checked for a pure decay-to-zero pattern (edge existing only in the
earliest years of the sample, 2018-2019, and vanishing thereafter) — if found, this is disclosed
explicitly in `E3_REPORT.md`, not silently omitted from the regime-sanity section.

**S3 lookahead canary (mandatory, in addition to the shared canary discipline below)**: a version
where the entry is backdated INTO the announcement window itself (peeking at the announcement
return before it has actually completed, i.e. entering at close of `t-1`/`t` instead of waiting for
the window to close) — must show a hugely inflated, obviously-fake Sharpe. Excluded from real
results/family/PBO/DSR per the shared convention below.

## Combined pre-named deployable (decided now, before any run — never adjusted after seeing any
sleeve's P&L)

**Equal-risk-contribution blend of all three sleeves** (S1 ensemble + S2 W_beta=52 primary + S3
ensemble): each sleeve's own monthly-marked net-return series (S3's event-portfolio series
resampled/aligned onto the same monthly grid as S1/S2 for blending purposes — the blend operates
at monthly cadence even though S3's own native event-time series may have different internal
spacing, per bug-pattern (2) above; the resampling step itself is declared and its own diagnostic
printed, not silently assumed lossless) is independently vol-scaled to contribute equal risk to
the combined book, reusing the cluster-risk-parity + vol-target overlay pattern from
`research/entry_exit/study/DURABLE_REPORT.md`'s S4 construction (equal risk contribution across
sub-books, THEN a single 10% vol-target overlay applied on top, reusing
`regime/overlay.py::vol_target_scale`/`apply_scale`'s convention). Primary cost 13bp. **This is THE
deployable** — sleeves are reported individually for transparency, but the adoption gate applies to
the COMBINED book, never to whichever sleeve looks best in isolation.

This combined overlay is exactly the kind of time-varying-gross-exposure construction where E2's
exposure-timing and cost-scaling bugs (landmines 3 and 4) could recur if not explicitly guarded
against from the start — both are declared and guarded above (AVOIDING BUGS (3) and (4)) and are
re-verified specifically at the combined-book level (not just at each individual sleeve's own
internal overlay, if any), since the combined book's overlay is a NEW construction layered on top
of three already-vol-scaled sleeve books, not a reused sleeve-level overlay.

## N_trials ledger (declared in full before any run — every cell below is a real cell that COULD
be run and IS run; compute the EXACT final integer)

- **S1**: 1 ensemble + 3 per-lookback (L=3, 6, 12) = **4 cells**.
- **S2**: 3 beta-window variants (W_beta=26, 52, 104) = **3 cells**.
- **S3**: 1 ensemble + 3 holding-period variants (2, 4, 6 weeks) = **4 cells**.
- **Combined deployable**: **1 cell**.
- **Cost/funding scenario grid**: applied to the four HEADLINE cells only (S1 ensemble, S2 primary
  window [W_beta=52], S3 ensemble, combined deployable) — not to every grid cell above, which
  would multiply the ledger for no new information about which underlying SIGNAL works. **4
  headline cells x 3 cost scenarios (13bp primary, {6.5bp, 26bp}) x 3 funding scenarios (0bp/wk,
  +5bp/wk, -5bp/wk) = 36 cells**.
- **Lookahead canaries**: one per sleeve (S1, S2, S3) = 3 canaries, EXCLUDED from N_trials per the
  mandatory convention (bug-detectors, not candidate edges), reported separately in
  `E3_REPORT.md`.

**N_trials = 4 (S1) + 3 (S2) + 4 (S3) + 1 (combined) + 36 (cost/funding) = 48.** This is the full,
honestly-enumerated ledger charged to DSR's N_trials — no hand-picked subset. The three canaries
are excluded per the mandatory convention and reported separately.

## Lookahead canary — shared discipline (mandatory per sleeve, excluded from real
results/family/PBO/DSR)

Each sleeve gets its own deliberately-wrong version peeking at the period's own forward
information, per the sleeve-specific descriptions above (S1/S2: the ranking signal at rebalance
date `t` is computed using period `t`'s OWN forward return instead of only trailing information
through `t`'s formation/estimation window; S3: entry backdated into the announcement window
itself, as declared in the S3 section). Each must show an inflated, obviously-too-good Sharpe that
clearly dominates its honest counterpart — if any canary does NOT clearly dominate, the
corresponding honest pipeline is treated as suspect for a construction bug and re-audited before
its result is reported, exactly per E1's and E2's own precedent (E2's first canary attempt failed
this exact check and revealed a real sign/construction bug before the reported run). All three
canaries are excluded from every family/PBO/DSR calculation and from the N_trials ledger above.

## Costs & validation (shared gate family, reused per METHODOLOGY)

- CPCV(8 groups, k=2, embargo >=2 months given monthly/event cadence — reusing `cpcv_paths()`'s
  signature from `e2_pipeline.py`, parameterized by each sleeve's own DERIVED `periods_per_yr`, not
  a hardcoded one; S3's event-portfolio series gets its own embargo consistent with its own derived
  cadence, not assumed identical to S1/S2's monthly embargo).
- CSCV-style PBO over the FULL 48-cell family.
- Drop-top-5-periods robustness (periods = whatever each cell's own native marking cadence is —
  monthly for S1/S2, the combined book's own monthly grid, and S3's own native event-marking
  cadence for its own standalone drop-top-5 check).
- Per-year Sharpes for every headline cell.
- Full metrics table per cell: net Sharpe, EV bp/name-period, win%, maxDD, CAGR raw + at-10%-vol,
  turnover via `turn=` wired and populated (reusing E2's fix — E1's `turn=` parameter was accepted
  but silently never passed at any call site; this is verified NOT repeated by an explicit
  `turn_mean is not None` assertion in `e3_pipeline.py` for every cell before that cell's results
  are trusted), and cost scenarios — GROSS shown alongside NET everywhere, not just at headline
  cells.
- Full test suite (>=654 passed/1 skipped baseline, confirmed unchanged through E1 and E2) must
  pass before any commit:
  `cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/python -m pytest tests -q`
  — the pass count is reported in `E3_REPORT.md`.

## Adoption gate (frozen, pre-outcome, applied to the COMBINED deployable)

Recommend PAPER-FIRST deployment of the combined deployable ONLY IF ALL of the following hold
simultaneously:

1. Net Sharpe positive on the full base-universe (93 for S1/S2 legs, 76 for S3's leg, blended)
   AND on the on_demo-and-real-history-and-not-leveraged subset (16 for S1/S2, 11 for S3, blended)
   — both reported, on_demo subset used for the paper-deployability judgment per E1/E2 precedent.
2. DSR pass at the full N_trials = 48.
3. PBO <= 0.25 over the full 48-cell family.
4. Drop-top-5-periods Sharpe still positive.
5. EVERY INCLUDED sleeve's own regime-sanity signature holds per its declared check above:
   - S1: a visible, identifiable momentum-crash-consistent stress episode near 2020-04 (actual
     numbers shown, not just yes/no).
   - S2: ex-post book beta close to zero (the mandatory adversarial check, an actual computed
     number) AND a resilience pattern (not necessarily outperformance, but not a matching-or-worse
     crash) across the four named stress windows, with actual per-window numbers shown against the
     broad-book comparator.
   - S3: drift concentrated in the less-liquid half (actual Sharpe/EV-bp shown for both halves) AND
     no undisclosed pure-decay-to-zero pattern across per-year Sharpes.

If ANY gate fails, the negative is recorded plainly in `E3_REPORT.md` as the program's conclusion,
and deployment is recommended AGAINST — a correctly-identified dead end is a complete, valuable
result, exactly per E1's and E2's own discipline. Any individual sleeve or cell that independently
clears its OWN gates (net Sharpe positive, and its own regime-sanity signature holds) even if the
combined deployable does not, is reported as a SECONDARY finding, not a silent redefinition of the
deployable — the combined book named above remains "the deployable" unless this document is
revised and re-frozen first.

**Honest target (stated before any run)**: 0.3-0.6 NET Sharpe for the combined deployable is the
realistic target — two-to-three moderate ~0.3-0.5-ish sleeves combined with a diversification
benefit, not one heroic single-sleeve number. Above ~1.0 net on any SINGLE sleeve triggers a
mandatory bug hunt against all five carried-forward lessons (DATA_CONTRACT landmines above) before
being reported as a finding, exactly as E1's and E2's reports both required for their own near-
miss/interesting cells.

## Full test suite

Before any commit, the command in "Costs & validation" above must pass with no fewer than the
654-passed/1-skipped baseline recorded at the E1/E2 commits — the pass count is reported in
`E3_REPORT.md`.
