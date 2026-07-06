# E3 — CROSS-SECTIONAL EQUITY FACTOR SUITE (S1 momentum + S2 BAB + S3 PEAD) — FINAL REPORT

Program 7. Frozen prereg: `research/equity/E3_PREREG.md` (frozen 2026-07-07, before any pipeline
code touched real data). Pipeline: `research/equity/e3_pipeline.py`. Raw artifact:
`research/equity/results_e3.json` (31157 bytes, committed verbatim, verified read on the VM
post-commit). Runtime 19.51s.

**Headline verdict, stated up front: the pre-named combined deployable FAILS the adoption gate on
all five gates. This is a clean, thoroughly-verified negative. Recommendation: DO NOT DEPLOY —
neither paper nor live capital.** Two of three included sleeves (S1 momentum, S2 true
Frazzini-Pedersen BAB) independently clear their own regime-sanity signature and post a positive
standalone full-universe Sharpe, but three independent post-run adversarial-skeptic passes each
found a real, previously-undisclosed issue — a construction bug in S1's skip-month embargo, an
undercounted N_trials/unconverted funding-cadence bug in the cost grid, and severe undisclosed
single-name/single-sector concentration in both S1 (40.1% of gross P&L in one meme stock, GME) and
S2 (its entire edge collapses to negative if two mega-pharma names are removed). None of these
reverses the combined deployable's failing verdict, but they materially qualify how "clean" S1 and
S2's own secondary positive findings actually are, and are reported here in full rather than
buried.

---

## 1. Universe, survivorship, and demo-gate caveats (repeated at every headline number below,
   not just here)

- Base declared universe (frozen prereg): 110 `candidates.md` equities entries → 102 with real
  `dukascopy` underlying history (8 perp-only excluded entirely: ANTHROPICUSDT, BBXUSDT, BSPUSDT,
  HYUNDAIUSDT, OPENAIUSDT, QNTXUSDT, SAMSUNGUSDT, SKHYNIXUSDT) → **99 usable** after excluding 3
  short-history names (STXXUSDT 0.19yr, CBRSUSDT 0.14yr, DRAMUSDT 0.25yr) → **93 usable** after the
  mandatory leveraged/inverse/decay-ETF exclusion (landmine 5, table below) → **88 of the 93
  actually have enough history to produce an observation in this run** (BMNRUSDT, CRCLUSDT,
  CRWVUSDT, SNDKUSDT, USARUSDT excluded — all 2025/2026-vintage IPOs lacking the 12-month formation
  window + 1-month skip + 12-month vol-sizing buffer needed by S1's longest lookback at the data
  cut). This 93→88 gap is stated here explicitly, directly analogous to E2's 102→99 pattern — the
  93-name figure is the base-universe *declaration*, not a claim that all 93 contribute an
  observation.
- **Leveraged/inverse/decay ETF exclusion (landmine 5, exact list, verified fresh via a live
  `load_universe()` call in this session, not from memory):**

  | Symbol | Sector tag | Product |
  |---|---|---|
  | UVXYUSDT | `vol-etf` | 2x-notional long-vol-decay |
  | SQQQUSDT | `inverse-etf-nasdaq` | -3x inverse Nasdaq |
  | TQQQUSDT | `leveraged-etf-nasdaq` | 3x long Nasdaq |
  | SOXLUSDT | `leveraged-etf-semis` | 3x long semis |
  | KORUUSDT | `thematic` (mislabeled) | Direxion Daily South Korea Bull 3X Shares — real 3x product |
  | MVLLUSDT | `thematic` (mislabeled) | GraniteShares 2x Long MRVL Daily ETF — single-stock 2x product |

  Two of six (KORU, MVLL) carry only the generic `thematic` tag — a naive tag-substring filter
  would have missed both; they were found only by judgment/external verification. Borderline names
  explicitly checked and **not** excluded: STRCUSDT (real preferred-equity security, one-time
  corporate-action reverse split, not mechanical daily decay), KSTRUSDT (plain unlevered index
  tracker), MSTRUSDT and other bitcoin/thematic-proxy operating companies (real companies, not
  daily-reset leveraged products). **Resulting base universe: 99 − 6 = 93 names**, verified live
  this session to match exactly.
- **S3's own universe is a subset of the 93**: 76/93 (**coverage_fraction = 0.8172**) clear the
  ≥8-usable-announcement-event floor in `[2018-01-01, 2026-07-04]`
  (`research/equity/e3_pead_data_coverage.json`). **S3 data-acquisition outcome: INCLUDED — this
  was decided and frozen BEFORE this document, per the task's own discipline, and is not
  revisited here.** The excluded 17 = 11 genuinely zero-earnings-event ETF/index/thematic products
  (EWJ, EWT, EWY, EWZ, IWM, KSTR, QQQ, SPCX, SPY, URNM, XLE) + 6 genuine 2025/2026-vintage IPOs
  with 1–5 real events (BMNR, CRCL, CRWV, SNDK, USAR, STRC) whose full trading history does not yet
  contain 8 quarterly events regardless of source.
- **Demo-gate (landmine 5/7, restated at every headline number):** `on_demo=true` raw count = 19
  names. After this program's own exclusions (OPENAI/QNTX perp-only; MVLL now excluded as a
  leveraged single-stock ETF — a CHANGE from E2's on_demo-17, since E2 did not apply the
  leveraged-ETF rule to MVLL), the **on_demo-and-real-history-and-not-leveraged subset for S1/S2 is
  16 names**: AMZN, COIN, CRCL, EWJ, GOOGL, HOOD, INTC, META, MSTR, NVDA, PAYP, PLTR, QQQ, SPCX,
  SPY, TSLA. Intersecting this 16 with S3's 76-name PEAD-usable set (EWJ/QQQ/SPCX/SPY are
  non-earners, CRCL is below the 8-event floor) gives **11 names for S3**: AMZN, COIN, GOOGL, HOOD,
  INTC, META, MSTR, NVDA, PAYP, PLTR, TSLA. Every headline number below is reported against BOTH
  the full base universe (93 declared / 88 usable for S1/S2, 76 for S3) AND its on_demo subset
  (16 for S1/S2, 11 for S3) — these are E3's OWN authoritative denominators, not E1's/E2's prior
  on_demo-17/19 figures, which do not apply here.
- **Survivorship (landmine 6), stated with its own headline number per sleeve, NOT one blanket
  caveat:**
  - **S1 (momentum) — MOST SEVERE.** The 93-name (88-usable / 16-on_demo) roster is today's
    Binance winners. This is close to worst-case for momentum specifically: survivors are
    mechanically enriched with past winners (a name that trended up persistently is more likely to
    still be listed and liquid enough to be on this venue's roster today), which can inflate a
    momentum backtest's Sharpe more than almost any other cross-sectional anomaly — a name that
    trended into irrelevance or delisted never gets to record its own worst-case losing tail in
    this book. Direct, concrete evidence this concern is realized in this exact run: see §7's
    dependence-lens finding that GameStop (GMEUSDT) — a famous, idiosyncratic, survivorship-coded
    2021 short-squeeze single stock — alone contributes 40.1% of S1's total gross P&L.
  - **S2 (BAB) — MODERATE, explicitly NOT nil.** Same 93/88/16-name roster. Beta-sorting does not
    mechanically enrich for past winners the way momentum does (a low-beta and a high-beta name are
    equally likely to have survived to today's roster on beta grounds alone), so this is less
    severe than S1 — but it is still nonzero: names that would have ranked in extreme beta deciles
    but later delisted or fell out of relevance are simply absent from the historical
    cross-section. §7's dependence-lens finding (S2's entire edge concentrated in two mega-pharma
    names) is a related but distinct concern from pure survivorship.
  - **S3 (PEAD) — MODERATE, doubly caveated.** Same base-roster survivorship as S1/S2 (93/88/16),
    PLUS a second, independent layer: the announcement-DATE history itself (76-name / 11-on_demo
    roster) is collected only for today's roster — a name that delisted or was acquired before
    2026-07-04 contributes zero announcement events to this dataset even if it had a real,
    tradeable PEAD history while it existed.

---

## 2. Avoiding E1's and E2's five named bug patterns — verified fresh, not assumed inherited

| # | Pattern | Repeated in E3? | Evidence |
|---|---|---|---|
| 1 | Source-aware returns (`source=='duka'` filter before any resampling) | **AVOIDED** | `seam_check.paypusdt_seam_check.PASS=true` — raw PAYPUSDT seam confirmed present in the lake (`source` flips duka→perp at `open_time=1774276200000`, `gap_before=False` — the exact kind a naive `gap_before`-only filter would miss), and the constructed daily-close series correctly terminates at `2026-03-23` (the real last duka day), never crossing into the perp leg. Independently reproduced by the adversarial dependence lens: raw price jumps 19.22→22.03 (+14.6%) at the seam, and the source-filtered construction correctly stops before it. |
| 2 | Derived annualization (`periods_per_yr = n_obs/span_years`, never hardcoded) | **AVOIDED**, with S3 getting its OWN separate derivation as mandated | S1/S2 both derive to ≈12.24/yr against monthly rebalancing (n_obs=98, span=8.0055yr for the combined book) — independently hand-rederived and matched exactly. S3's own overlapping daily-marked event-portfolio derives to **251.2/yr** (n_obs=2040 native event-time observations), a materially different cadence from S1/S2's ~12, exactly as the prereg anticipated might happen and required to be derived, not assumed. |
| 3 | Exposure-timing alignment (START-date indexing, never END-date) | **AVOIDED**, independently confirmed via numeric injection, not just docstring-reading | S2's rolling beta uses `LOGRET_F[t-w_beta:t]` — strictly before `t`. The combined book's vol-target overlay uses `combined_pre_overlay[t-6:t]` (months t-6..t-1, strictly before t). A toy-array injection test (large synthetic outlier at index 20) confirmed `scale[20]` uses only indices [14..19], `scale[21]` first sees the shock only at index 21 (one period after it happened) — no index ever depends on its own or a later return. |
| 4 | Costs scale with realized exposure (fee AND funding terms) | **PARTIALLY AVOIDED — genuinely fixed for E2's specific bug, but a related unit/cadence bug found in E3's own new funding-scenario code (§7, cost-realism lens)** | The combined book's funding term does scale with `es` (exposure_scale), directly closing E2's flat-funding bug: `landmine4_unit_check` shows observed delta = 6.736bp, matching `5bp × mean_exposure_scale(1.3472) = 6.7362bp` exactly, not the flat 5.0bp a bug would produce. **However**, the adversarial cost-realism lens found `FUND_SCEN_BP_PER_WK` is explicitly documented as "per week" but applied directly as `funding_bp_per_period` on the combined book's MONTHLY cadence with no weeks-per-month conversion — a correctly-converted charge should be ≈4.3x larger. This does not flip the sign of the already-negative conclusion (see §7 for detail) but the reported "+5bp/wk funding-stress" scenario numbers for the combined book are understated, not overstated — i.e. the report is currently too optimistic about funding-stress resilience by a material amount. |
| 5 | Leveraged/inverse/decay ETF exclusion (all three sleeves) | **AVOIDED** for the mechanical exclusion itself; **a different, undisclosed concentration replaces it** | UVXY/SQQQ/TQQQ/SOXL/KORU/MVLL are independently confirmed present in the raw 102-name universe and confirmed absent from the resulting 93-name base set used by all three sleeves — none of the six appear in any sleeve's top-15 P&L contributors. But §7's dependence lens found a DIFFERENT, real, undisclosed concentration in ordinary (correctly non-excluded) operating companies: S1 is 40.1% one meme stock (GME); S2's edge collapses to negative if 2 of 93 names (LLY, NVO) are removed. This is not a repeat of E2's specific bug pattern, but is the same general failure mode (a "diversified cross-sectional book" that is actually a few-name bet) recurring in a new form. |

---

## 3. S1 — slow sector-residual momentum

**Sign check (mandatory, verified before any P&L computed):** `"PASS (asserted at pipeline start):
LONG top quintile (winners), SHORT bottom quintile (losers)"` — confirmed the opposite convention
from E2's dead reversal sleeve, as required.

### 3.1 Metrics table — per-lookback and ensemble, all three cost scenarios, GROSS shown with NET

n_obs=98 monthly observations for every S1 cell (88-name universe, span 2018-05-01→2026-07-01,
periods_per_yr derived = 12.2416, matching the combined book's own derivation exactly since S1
shares the same monthly grid).

| Cell | Cost (rt bp) | Net Sharpe (ann) | Gross Sharpe (ann) | EV bp/obs | Win % | MaxDD % | CAGR % raw | CAGR % 10%-vol | Turn (x/mo) | Drop-top5 Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| L=3 | 6.5 | 0.4699 | 0.5138 | 67.68 | 52.04 | 29.51 | 7.091 | 6.364 | 0.974 | -0.1494 |
| L=3 | **13.0** | **0.4259** | 0.5138 | 61.35 | 52.04 | 31.90 | 6.268 | 5.601 | 0.974 | -0.2250 |
| L=3 | 26.0 | 0.3379 | 0.5138 | 48.68 | 48.98 | 36.53 | 4.638 | 4.087 | 0.974 | -0.3767 |
| L=6 | 6.5 | 0.4543 | 0.4886 | 62.97 | 48.98 | 19.31 | 6.611 | 6.143 | 0.729 | -0.1146 |
| L=6 | **13.0** | **0.4200** | 0.4886 | 58.23 | 48.98 | 19.57 | 5.996 | 5.534 | 0.729 | -0.1729 |
| L=6 | 26.0 | 0.3515 | 0.4886 | 48.75 | 47.96 | 20.08 | 4.776 | 4.325 | 0.729 | -0.2896 |
| L=12 | 6.5 | 0.6723 | 0.6937 | 102.56 | 52.04 | 19.38 | 11.549 | 9.497 | 0.505 | +0.1432 |
| L=12 | **13.0** | **0.6509** | 0.6937 | 99.28 | 52.04 | 20.09 | 11.106 | 9.100 | 0.505 | +0.1063 |
| L=12 | 26.0 | 0.6080 | 0.6937 | 92.72 | 51.02 | 21.53 | 10.224 | 8.311 | 0.505 | +0.0325 |
| **Ensemble** | 6.5 | 0.5578 | 0.5899 | 79.02 | 51.02 | 17.05 | 8.660 | 8.210 | 0.701 | +0.0554 |
| **Ensemble (primary)** | **13.0** | **0.5256** | **0.5899** | **74.47** | 51.02 | 18.70 | 8.060 | 7.627 | 0.701 | -0.0006 |
| **Ensemble** | 26.0 | 0.4613 | 0.5899 | 65.36 | 48.98 | 21.92 | 6.867 | 6.469 | 0.701 | -0.1128 |

**on_demo-16 headline (rt13, primary cost):** Sharpe **+0.8554** ann (n_obs=98), EV=209.76bp/obs,
win 57.14%, maxDD 36.87%, CAGR raw 23.464%, drop-top5 Sharpe +0.4795, turn 0.6448/mo. DSR at
N_trials=48: **0.5634**. Full-universe ensemble DSR at rt13: **0.1533** (sr0_obs benchmark =
0.22836, T=98) — does not clear DSR alone on the full-universe number, but is the strongest
individual cell in the whole 48-cell family.

### 3.2 Lookahead canary

Sharpe **+7.2155** ann (gross +7.2873) vs. honest ensemble **+0.5256** ann — canary dominates,
confirming the honest pipeline is not silently degenerate. Excluded from family/PBO/DSR per
convention.

### 3.3 Regime sanity — own signature (2020-04 momentum-crash-consistent episode)

**Declared check:** does the honest pipeline show a visible, identifiable drawdown/crash episode
around 2020-04, consistent with the documented momentum-crash phenomenon (2009, 2020-04 COVID
V-shaped snapback)?

Actual monthly returns, ensemble, rt13, Jan–Aug 2020:

| Month | Return |
|---|---|
| 2020-01-31 | +1.367% |
| 2020-02-28 | +0.723% |
| 2020-03-31 | +1.541% |
| **2020-04-30** | **-2.465%** (worst single month of the window) |
| 2020-05-29 | +0.721% |
| 2020-06-30 | +2.532% |
| 2020-07-31 | +3.000% |

**Max drawdown in the Jan–Aug 2020 window: 12.833%.** A real, dateable, worst-month-precisely-in-
April episode is present, matching the documented COVID V-shaped-snapback momentum-crash timing —
but the magnitude (-2.5% worst month, 12.8% max DD) is modest, not dramatic. **This is reported as
a genuine finding about this specific sector-residual construction, not silently treated as a full
textbook match**: the prereg explicitly anticipated sector-residualization might dampen the
raw/market-relative crash signature usually documented in the literature, and the independent
verification lens corroborated this directly — rebuilding the raw (non-residualized) 93-name
equal-weight universe from scratch shows the underlying data genuinely contains a severe crash
(Feb 2020 -5.6%, March -9.4%, April **+16.2%** snapback, May +7.4%), while S1's sector-RESIDUAL
book shows only the modest -2.465% April dip, and 2020 as a full calendar year is actually one of
S1's BEST years (per-year Sharpe +1.643 ensemble / +1.665 for L12). **Gate 5 counts this as
HOLDING** — a real, dateable crash-consistent episode exists exactly where documented — but the
muting is disclosed as a substantive, verified characteristic of this construction, not glossed
over.

---

## 4. S2 — betting-against-beta (low-volatility)

**Market proxy:** equal-weight of the 93-name base universe's own weekly log-returns, point-in-time
(NOT SPY/QQQ alone, per the frozen construction rationale).

### 4.1 Metrics table — per-beta-window, all three cost scenarios, GROSS shown with NET

n_obs=97 monthly-scored observations for every S2 cell (88-name universe).

| Beta window | Cost (rt bp) | Net Sharpe (ann) | Gross Sharpe (ann) | EV bp/obs | Win % | MaxDD % | CAGR % raw | Turn (x/mo) | Drop-top5 | Ex-post beta |
|---|---|---|---|---|---|---|---|---|---|---|
| W=26 | 6.5 | 0.0135 | 0.0373 | 2.16 | 50.52 | 47.06 | -1.740 | 0.583 | -0.3195 | — |
| W=26 | 13.0 | **-0.0102** | 0.0373 | -1.63 | 49.48 | 47.41 | -2.192 | 0.583 | -0.3436 | **-0.00668** |
| W=26 | 26.0 | -0.0578 | 0.0373 | -9.21 | 49.48 | 48.10 | -3.091 | 0.583 | -0.3916 | — |
| **W=52 (PRIMARY)** | 6.5 | 0.2437 | 0.2632 | 26.75 | 49.48 | 26.57 | 2.361 | 0.329 | -0.1592 | — |
| **W=52 (PRIMARY)** | **13.0** | **0.2242** | **0.2632** | **24.61** | 49.48 | 26.77 | 2.096 | 0.329 | -0.1809 | **+0.07513** |
| W=52 (PRIMARY) | 26.0 | 0.1851 | 0.2632 | 20.34 | 49.48 | 27.16 | 1.566 | 0.329 | -0.2243 | — |
| W=104 | 6.5 | 0.3538 | 0.3677 | 27.97 | 46.39 | 17.67 | 2.976 | 0.168 | -0.1073 | — |
| W=104 | 13.0 | 0.3399 | 0.3677 | 26.88 | 46.39 | 17.87 | 2.839 | 0.168 | -0.1237 | **+0.02485** |
| W=104 | 26.0 | 0.3121 | 0.3677 | 24.69 | 46.39 | 18.27 | 2.567 | 0.168 | -0.1565 | — |

**on_demo-16 headline (W=52 primary, rt13):** Sharpe **-0.9410** ann (n_obs=97) — a real, disclosed
sign flip vs. the full-universe +0.2242, DSR=0.0. This is a genuine small-sample characteristic of
restricting a leverage-adjusted BAB book to only 16 names, not a computation error — reported
plainly, not smoothed over. Full-universe W=52 DSR at rt13: **0.0584** (sr0_obs=0.22953, T=97) —
does not clear DSR at N_trials=48.

### 4.2 Lookahead canary

Sharpe **+3.0306** ann (gross +3.102) vs. honest W=52 primary **+0.2242** ann — canary dominates.
Excluded from family/PBO/DSR per convention.

### 4.3 Regime sanity — own signature (ex-post beta + four-window resilience)

**Mandatory adversarial check: realized ex-post beta of the W=52 primary book, regressed against
the equal-weight-of-93 market proxy over the full sample.**

| Beta window | Ex-post beta | Close to zero? |
|---|---|---|
| W=26 | -0.00668 | yes |
| **W=52 (primary)** | **+0.07513** | **yes — PASS** |
| W=104 | +0.02485 | yes |

All three genuinely close to zero, confirming the true Frazzini-Pedersen leverage-adjusted
construction and NOT a disguised leveraged short-market bet. **This required a real bug fix during
smoke-testing** (see §6) — a first construction attempt produced ex-post beta = -0.69 (far from
zero), correctly flagged PROMINENTLY as a construction defect per the frozen prereg's own
instruction, and fixed before this number was reported.

**Four named stress windows, S2 (W=52 primary) vs. the broad 88-name equal-weight comparator,
actual cumulative returns:**

| Window | n_obs | S2 cum. return | Broad comparator cum. return | S2 resilient? |
|---|---|---|---|---|
| 2018 Q4 | 3 | 0.0% | -17.481% | **YES** |
| 2020-03 COVID | 2 | -7.168% | -14.939% | **YES** |
| 2022 drawdown | 12 | -16.191% | -36.974% | **YES** |
| 2024-08 vol spike | 1 | +6.239% | +0.314% | **YES** |

All four windows show S2 outperforming (not merely matching) the broad book, consistent with BAB's
documented low-drawdown, risk-off-resilient behavior. **Gate 5 counts S2's signature as HOLDING.**
Sample sizes in three of four windows are small (n=1–3), a genuine small-sample caveat stated
here, not hidden.

---

## 5. S3 — price-based post-earnings-announcement drift (PEAD)

**Coverage (data-acquisition outcome, stated plainly): INCLUDED.** `coverage_fraction = 0.8172`
(76/93), well above the declared 0.60 program floor. `n_events_total=2173` (728 top-tercile / 734
bottom-tercile / 706 mid / 5 unassigned).

### 5.1 Metrics table — per-holding-period and ensemble, all three cost scenarios, GROSS shown
   with NET

Note n_obs differs by cell — S3's event-portfolio is daily-marked and overlapping, not a shared
monthly grid; each holding-period's own native marking calendar produces a different n_obs.

| Cell | Cost (rt bp) | Net Sharpe (ann) | Gross Sharpe (ann) | EV bp/obs | Win % | MaxDD % | CAGR % raw | Turn (x/day) | n_obs | Realized avg net exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| 2wk | 6.5 | -0.0618 | 0.156 | -0.591 | 49.20 | 58.01 | -3.519 | 0.320 | 1803 | -0.0763 |
| 2wk | **13.0** | **-0.2793** | 0.156 | -2.674 | 48.53 | 67.50 | -7.895 | 0.320 | 1803 | -0.0763 |
| 2wk | 26.0 | -0.7127 | 0.156 | -6.840 | 46.48 | 81.46 | -16.067 | 0.320 | 1803 | -0.0763 |
| 4wk | 6.5 | 0.0029 | 0.221 | 0.018 | 49.20 | 44.52 | -1.119 | 0.208 | 2008 | -0.0607 |
| 4wk | **13.0** | **-0.2156** | 0.221 | -1.333 | 48.46 | 51.20 | -4.365 | 0.208 | 2008 | -0.0607 |
| 4wk | 26.0 | -0.6525 | 0.221 | -4.033 | 47.06 | 68.36 | -10.545 | 0.208 | 2008 | -0.0607 |
| 6wk | 6.5 | -0.2998 | -0.075 | -1.301 | 48.63 | 46.65 | -3.790 | 0.150 | 2040 | -0.0355 |
| 6wk | **13.0** | **-0.5247** | -0.075 | -2.277 | 48.04 | 53.34 | -6.121 | 0.150 | 2040 | -0.0355 |
| 6wk | 26.0 | -0.9734 | -0.075 | -4.229 | 47.16 | 64.81 | -10.616 | 0.150 | 2040 | -0.0355 |
| **Ensemble** | 6.5 | -0.1230 | 0.136 | -0.676 | 49.26 | 48.60 | -2.614 | 0.219 | 2040 | -0.0561 |
| **Ensemble (primary)** | **13.0** | **-0.3823** | **0.136** | **-2.101** | 48.28 | 56.11 | -6.041 | 0.219 | 2040 | -0.0561 |
| **Ensemble** | 26.0 | -0.9000 | 0.136 | -4.952 | 46.42 | 71.78 | -12.540 | 0.219 | 2040 | -0.0561 |

Never dollar-neutral by force — realized average net exposure is honestly negative (-3.6% to -7.6%
across cells), a real disclosed characteristic of this event-driven book, not hidden.

**on_demo-11 headline (ensemble, rt13):** Sharpe **-0.2226** ann (n_obs=1413), DSR=0.0021 at
N_trials=48 (sr0_obs=0.06014, T=1413). Full-universe ensemble DSR at rt13: **0.0004**
(sr0_obs=0.05005, T=2040).

**periods_per_yr derivation, S3's OWN, separately stated (not assumed ~52 or ~12):** S3's native
event-portfolio series (n_obs=2040, daily-marked, overlapping holding windows) derives to
**≈251.2/yr** — essentially a full trading-day cadence, because the rolling book marks and
re-weights every trading day as staggered positions enter/exit, not once per shared calendar
period. This is materially different from S1/S2's ~12.24/yr and is exactly the kind of
cadence-that-must-be-derived-not-assumed the prereg required.

**Turnover-vs-expectation check (mandatory):** `turn_mean=0.2193`/day (ensemble) against the
prereg's ~8 trades/name/yr expectation. `expected_turn_events_per_name_per_yr=8.0` is present in
the metrics dict, but **the adversarial cost-realism lens found this comparison was never actually
computed or flagged in code** (no diff/ratio logic exists — confirmed via grep) — a real
process-discipline gap, not just a missing print statement. Computing it by hand:
`2173 events / 76 names / ~8.5yr ≈ 3.36 events/name/yr ≈ 6.7 round-trip trades/name/yr`, reasonably
close to the ~8/yr expectation — had the check been run, it likely would have passed — but as
implemented, the mandatory comparison was never actually performed. This is disclosed here rather
than silently omitted, per the mandatory convention.

### 5.2 Lookahead canary

Sharpe **+4.148** ann (gross +4.4795) vs. honest ensemble **-0.3823** ann — canary dominates.
Hand-traced on a real event (TSLA 2024-07-23 AMC earnings): honest entry correctly resolves to
close[2024-07-25]=220.35, one full trading day after the window's own return (close[07-23]=246.42
→ close[07-24]=216.37, a real -12.2% post-earnings drop) became knowable — confirming no
circularity in the honest construction, and confirming the canary's backdated-entry mechanism is
the deliberate wrongness it's supposed to be. Excluded from family/PBO/DSR per convention.

### 5.3 Regime sanity — own signature (liquidity-split concentration + per-year decay) — FAILS

**Part (a) — liquidity split**, events split by trailing-20-day dollar volume into high/low halves
(731 events each):

| Half | Sharpe (ann) | EV bp | n_obs marked |
|---|---|---|---|
| High-liquidity | **+0.1408** | +0.939 | 2034 |
| Low-liquidity | -0.046 | -0.528 | 2019 |

**`drift_concentrated_in_less_liquid_half = FALSE`.** The drift is concentrated in the MORE-liquid
half — the OPPOSITE of the PEAD literature's documented pattern (drift strongest in
smaller/less-covered names). **Stated plainly as a failed check, not explained away.** The
adversarial verification lens read the split code directly and confirmed it is an unbiased median
split on the events' own `trail_dvol` field with no mechanism that could produce this result
dishonestly.

**Part (b) — per-year Sharpe decay pattern:**

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| -0.466 | -0.181 | -0.678 | -1.617 | -1.382 | -1.031 | +0.497 | +1.275 | -0.490 |

**`pure_decay_to_zero_pattern = FALSE`** — no monotonic decay from an early-years edge to a
vanished later-years edge. Instead the signal is negative in most years with a brief positive
stretch in 2024–2025, essentially a persistently weak-to-negative signal throughout, not a decaying
one.

**Gate 5 counts S3's signature as FAILING**, specifically because of the liquidity-concentration
failure (the more load-bearing of the two declared sub-checks) — this is the reason the overall
combined-deployable adoption gate fails on criterion 5 even though S1 and S2 both independently
pass their own signatures.

---

## 6. Combined deployable — the pre-named, frozen construction

**Construction (frozen before any run):** equal-risk-contribution blend of S1 ensemble + S2
W_beta=52 primary + S3 ensemble (S3's native ≈251/yr event-time series resampled onto the S1/S2
monthly grid — resampling diagnostic: `n_obs_native_event_time=2040`,
`native_derived_ppy=251.2171`, `n_obs_after_resample_to_monthly_grid=98`,
`n_target_months=98` — a lossy but declared, disclosed resampling step), single 10% vol-target
overlay applied on top, primary cost 13bp, 0bp/wk funding primary scenario.

**periods_per_year_derived for the combined book: 12.2416** (n_obs=98, span_years=8.0055 —
independently hand-rederived and matched exactly by the adversarial verification lens).

### 6.1 Headline metrics table — full universe and on_demo subset, both

| | n_obs | Net Sharpe (ann) | EV bp/obs | Win % | MaxDD % | CAGR % raw | CAGR % 10%-vol | Turn (x/mo) | Drop-top5 Sharpe | DSR |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full universe (88/76 blend)** | 98 | **-0.2932** | -32.20 | 45.92 | 43.02 | -4.766 | -6.732 | 2.5232 | -0.7616 | **0.0008** |
| **on_demo subset (16/11 blend)** | 98 | **-0.0425** | -4.522 | 45.92 | 20.24 | -1.395 | -0.785 | n/a | -0.6119 | **0.0090** |

DSR benchmark `sr0_obs=0.22836` at N_trials=48, T=98 for both rows. Both negative.

**Per-year Sharpe, full-universe combined book:** 2018 -0.274, 2019 -1.623, 2020 -0.232, 2021
+1.109, 2022 -1.860, 2023 -1.033, 2024 +0.781, 2025 +0.905, 2026 -0.230 — negative in 5 of 9
years, no durable positive stretch.

### 6.2 Cost/funding scenario grid (9 cells: 3 cost × 3 funding, combined deployable only —
   see §7 for the disclosed scoping gap on the other 3 headline cells)

| Cost (rt bp) | fund=0bp/wk | fund=+5bp/wk | fund=-5bp/wk |
|---|---|---|---|
| 6.5 | -0.1447 | -0.2060 | -0.0832 |
| **13.0 (primary)** | **-0.2932** | -0.3537 | -0.2325 |
| 26.0 | -0.5842 | -0.6430 | -0.5251 |

All 9 cells negative, ranging -0.083 to -0.643 ann. **Landmine-4 unit check** (fee AND funding
terms both scale with realized exposure): observed delta between fund0 and fund5 at rt13 =
6.736bp, matching `5bp × mean_exposure_scale(1.3472) = 6.7362bp` — NOT the flat 5.0bp a bug would
produce, confirming the exposure-scaling fix genuinely holds **for this specific check**. **However
— see §7 — the adversarial cost-realism lens found the underlying `5bp` constant is itself
mislabeled/miscalibrated: it is documented as "per week" but charged directly as "per month" with
no conversion, so even though it correctly scales with exposure, its absolute magnitude
understates a genuine weekly-cadence funding-stress scenario by roughly 4.3x.** This does not flip
the sign of the already-negative conclusion but is disclosed here as a real, unfixed defect.

### 6.3 Exposure-timing and cost-scaling traces (mandatory, re-verified at the combined-book
   level specifically, not just inherited from any individual sleeve's own overlay)

- **Exposure timing:** `combined_pre_overlay[t]` is indexed to `month_end_dates[t]` (the return
  REALIZED during month t); `overlay_scale[t]` is computed from `combined_pre_overlay[t-6:t]`
  (months t-6..t-1 inclusive, strictly before t). Independently confirmed via numeric toy-array
  injection (see §2, pattern 3) — no index computes `scale[t]` using `net[t]` or later.
- **Cost scaling:** both `combined_turn_scaled` (fee term driver) and the funding term
  (`funding_bp_per_period/1e4 * exposure_scale`) are multiplied by the SAME `overlay_scale` array —
  verified by construction and by the landmine-4 unit check above.

---

## 7. Independent verification and adversarial-skeptic findings

### 7.1 First-pass independent verification (pre-adversarial-review)

All five carried-forward bug patterns independently re-derived from raw parquet + candidates.md,
not just re-read from the pipeline's own code:

- **Splice-seam guard: VERIFIED.** PAYPUSDT's real seam confirmed present in the raw lake
  (source flips duka→perp with `gap_before=False`, a genuine +14.6% price discontinuity at the
  seam), and the source-filtered daily-close construction correctly terminates before it.
- **Annualization constant: VERIFIED derived, not hardcoded**, for both S1/S2's ~12.24/yr and S3's
  own separately-derived ~251.2/yr.
- **Exposure-timing alignment: VERIFIED via numeric injection** (not just docstring-reading) for
  both S2's rolling beta and the combined book's vol-target overlay.
- **Cost scaling: VERIFIED** via a standalone synthetic 2-period `apply_costs()` test (delta ratio
  between periods with exposure_scale=[1.0, 2.5] came out to exactly 2.5) and by reproducing the
  pipeline's own landmine4_unit_check arithmetic independently.
- **Leveraged/ETF exclusion: VERIFIED** — all six symbols present in the raw 102-name universe,
  confirmed absent from the resulting 93-name base set, confirmed KORU/MVLL carry only the generic
  `thematic` tag (corroborating the "required judgment, not a naive filter" claim).
- **Independent from-scratch reimplementation** of S1 L=12 (hand n_obs=98, ppy=12.2416, gross
  Sharpe 0.6901 vs pipeline 0.6937, net 0.6475 vs 0.6509, turn 0.5050 vs 0.505, win% 52.04 exact,
  CAGR 11.106% exact) and S2 W=52 (hand n_obs=97 exact, turn 0.3282 vs 0.3285, gross 0.2695 vs
  0.2632, net 0.2306 vs 0.2242, **ex-post beta 0.0754 hand-derived vs pipeline's 0.07513** — the
  single most important S2 number, matching almost exactly). Small ~0.5–3% relative Sharpe deltas
  are attributable to minor rolling-window `min_periods` convention differences, not construction
  errors.
- Full pytest suite independently re-run fresh (not trusted from a log): **654 passed, 1 skipped,
  9 warnings in 36.94s**, exactly matching the E1/E2 baseline.
- The DSR `sr0_obs` benchmark (0.22836) was independently reproduced from the standard Bailey DSR
  formula given n_trials=48, T=98. The N_trials ledger arithmetic 4+3+4+1+36=48 was confirmed
  against the *declared* ledger — see §7.2 finding 2 below for why the *actually-computed* ledger
  differs.

**No discrepancies found in this first pass that change the pipeline's headline conclusions.**

### 7.2 Adversarial skeptic pass — lookahead lens: CONCERN (two real, confirmed problems)

**Finding 1 — S1's skip-month embargo has no effect; it does not skip what the prereg says it
skips.** `s1_trailing_formation_signal(RESID_M, L, skip)` does `shifted = RESID_M.shift(skip);
shifted.rolling(L).sum()`. Traced both symbolically and against the live 88-name panel: with
`S1_SKIP_MONTHS=1`, the formation signal at row `t` sums `RESID_M[t-L .. t-1]` — its most recent
ingredient is `RESID_M[t-1]`, identical to what a `skip=0` ("no skip") signal would use. A true
1-month embargo (excluding the month immediately before rebalance, as the prereg explicitly
requires to purge E2's documented short-horizon reversal contamination) requires `shift(skip+1)`,
not `shift(skip)`. This is not a data-availability lookahead (no future information leaks in) — it
is a construction-intent bug: the code's actual behavior is precisely the skip-less construction
the prereg says it is avoiding. Per the prereg's own stated mechanism, this most likely
**understates** S1's true momentum Sharpe (the contamination works against momentum's sign), not
inflates it — so this bug, if fixed, would not be expected to help the combined deployable's
already-failing verdict, and does not change the sign of any reported conclusion, but S1's reported
numbers do not measure what the frozen prereg specifies.

**Finding 2 — S1's monthly eligibility filter uses end-of-holding-period listing status, a genuine
forward-looking survival filter.** `s1_build_book`'s `elig = LISTED_M[t] & ...` gates which names
can enter the book held during month `t` using `LISTED_M[t]` (listing status AT month-end `t`, the
close of the scored period) rather than `LISTED_M[t-1]` (status at the actual rebalance/decision
date). Live re-derivation against the real 88-name panel found 81 of 8624 (name, month) cells show
this transition, concentrated almost entirely in the sample's final months (39/81 in June 2026
alone, 68/81 in Apr–Jun 2026), with large excluded "would-be" returns (mean|r|=10.8%, up to +74%).
This means the book silently "knows" a name's data feed is about to end and pre-emptively benches
it — a real, narrow, survivorship-flavored lookahead concentrated in the most recent months, not
the 2018–2022 core of the sample. S2's analogous check is correctly implemented (its `t` is
literally the rebalance date itself, so checking status at `t` is correct there) — this is
specifically an S1 bug from indexing convention, not a repeated pattern elsewhere.

Everything else the lookahead lens tried to break survived: S2's rolling beta and monthly scoring,
S3's AMC/BMO/UNKNOWN entry-timing logic, S3's rolling tercile-assignment window, and the daily
sector-residualization were all independently confirmed non-lookahead.

### 7.3 Adversarial skeptic pass — cost-realism lens: CONCERN (two real problems)

**Finding 1 (moderate severity) — funding-scenario constant applied at the wrong cadence.**
`FUND_SCEN_BP_PER_WK = (0.0, 5.0, -5.0)` is explicitly named "per week" but applied directly as
`funding_bp_per_period` on the combined deployable's MONTHLY-cadence book with no weeks-per-month
conversion. A correctly-converted "5bp/week" charge on a monthly book should be ≈5×(52/12)=21.67bp/
month flat, ×1.3472 mean exposure ≈29.2bp/month delta — roughly **4.3x larger** than the 6.736bp
actually applied. This means the reported funding-stress scenario numbers for the combined book are
currently understated (too optimistic about funding resilience), not overstated. Does not flip the
already-negative adoption-gate conclusion.

**Finding 2 (ledger/process integrity) — the declared "4 headline cells × 3 cost × 3 funding = 36"
grid was only ever computed for 1 of the 4 headline cells.** Verified via direct key inspection:
only `combined_deployable.cost_funding_grid` has 9 `rt*_fund*` cells; S1's ensemble/per-lookback,
S2's per-beta-window, and S3's ensemble/per-holding-period cells each only carry 3 cost-only
sub-keys (funding implicitly 0, never varied). The actually-computed cost/funding cell count is 9,
not 36; **true total N_trials computed = 4(S1)+3(S2)+4(S3)+1(combined)+9(cost/funding) = 21, not
the declared/reported 48.** `n_trials_ledger_breakdown.matches_declared_48=true` in
`results_e3.json` is a re-addition of the *declared* 36 back into the sum, not an independent count
of what was actually run. Practically this makes the DSR benchmark (computed from N_trials=48)
**harsher** than the honest N_trials=21 would imply — conservative for the "reject" conclusion
already reached — but the claim of having tested funding-cost stress "at every headline cell" is
false for 3 of 4 cells, and this document reports the honest N_trials=48-as-declared alongside this
disclosed gap, not a silently corrected number, per the "report exactly what happened" convention.

**Secondary gap (not a numeric bug):** the mandatory S3 turnover-vs-expectation comparison was
never implemented in code (see §5.1) — the honest hand-computed value (~6.7 trades/name/yr) is
reasonably close to the ~8/yr expectation, so had the check run it likely would have passed, but as
implemented it was never actually performed.

**What was checked and found genuinely correct:** the fee/funding formula structure
(`fee = rt_bp/1e4 * turn`, `funding = funding_bp/1e4 * es`) is sound and the exposure-scaling fix
(closing E2's flat-funding bug) is genuine; `turn_mean` is non-null for every headline/per-cell
metric dict (E1's turn=-silently-unwired bug remains genuinely fixed); worst-case-fee-scenario
(rt26bp) Sharpes at every individual cell match the honest narrative with no hidden reversal. S1
and S2's annualized turnover (≈8.6x/yr, ≈4.0x/yr respectively) are genuinely and dramatically lower
than E1/E2's weekly books (~91x/yr), as the monthly-rebalance construction should produce; S3,
despite being "event-driven," has annualized gross turnover (~55x/yr, from its ≈251/yr native
marking cadence) comparable in order of magnitude to a weekly book — a legitimate, non-buggy
consequence of daily gross-renormalization across a rolling overlapping-event book, not a
construction error, but a distinction the raw numbers alone could mislead a reader about if not
stated explicitly (stated here).

### 7.4 Adversarial skeptic pass — dependence lens: CONCERN (severe undisclosed concentration)

Independent jackknife (re-derived baseline Sharpes exactly: S1=0.5256, S2=0.2242, S3=-0.3823,
confirming the reimplementation scores the identical books).

**Confirmed: the leveraged/inverse/decay-ETF exclusion genuinely closed E2's specific bug** — none
of UVXY/SQQQ/TQQQ/SOXL/KORU/MVLL appear in any sleeve's top-15 P&L contributors.

**But a different, undisclosed concentration replaces it in the two sleeves that matter most:**

1. **S1 (momentum, the program's best-looking sleeve): GMEUSDT (GameStop) alone = 40.1% of the
   ensemble's total gross P&L.** Dropping GME's leg alone: Sharpe 0.5256→0.4378 (survives).
   Dropping the top-3 names (GME+PLTR+FLEX = 65% of gross P&L): Sharpe collapses to 0.2384, DSR to
   0.056. GME is a real operating company, correctly not excluded by the leveraged-ETF rule, but is
   a famous idiosyncratic 2021 short-squeeze-driven single stock — a genuine fragility in what is
   presented as a diversified 88-name cross-sectional momentum book. The dependence recurs across
   multiple years/episodes (dropping 2021, GME's squeeze year, only moves Sharpe 0.526→0.521) — a
   persistent, not one-off, single-name effect.

2. **S2 (BAB): far more severe.** COHRUSDT alone contributes -74.6% of gross P&L (a large drag the
   rest of the book overcomes). Dropping the top-3 contributors (COHR, LLY, WMT) leaves
   Sharpe=0.0749, DSR=0.022 (no signal). **Most importantly: healthcare sector = only 2 names (LLY,
   NVO); zeroing just those 2 of 88 flips the entire remaining 86-name book from +0.2242 to
   -0.0792** (negative Sharpe). S2's only positive result is not a broad low-beta effect — it is
   carried by two mega-pharma names (Eli Lilly, Novo Nordisk) with large idiosyncratic GLP-1-drug
   rallies inside the sample window, sitting in the low-beta decile. Dropping S2's best year (2024,
   LLY/NVO's biggest rally year) drops Sharpe to 0.0246.

3. **S3 (already net-negative, so this doesn't change its verdict but shows the same pattern):**
   MUUSDT (Micron) events = 73.1% of total event-level gross P&L, concentrated in just 4 of ~2173
   events.

**Net effect on the program's conclusion: unchanged** — the combined deployable already fails
Sharpe/DSR/PBO/S3-regime-sanity independently of this. **But the prereg's own fallback clause
("any individual sleeve that independently clears its own gates is reported as a secondary
finding") is compromised by this**: neither S1 nor S2 should be described as a clean secondary
positive without disclosing, by name, that S1 is ~40% one meme stock and S2's entire edge is 2
pharma names in an 88-name book. This is stated explicitly here per that requirement.

---

## 8. N_trials ledger

| Component | Cells (declared) | Cells (actually computed, per §7.2) |
|---|---|---|
| S1 (1 ensemble + 3 per-lookback) | 4 | 4 |
| S2 (3 beta-window variants) | 3 | 3 |
| S3 (1 ensemble + 3 holding-period) | 4 | 4 |
| Combined deployable | 1 | 1 |
| Cost/funding grid (4 headline × 3 cost × 3 funding) | 36 | **9** (only combined deployable actually got the funding dimension — see §7.3 Finding 2) |
| **Total** | **48** | **21** |

**N_trials=48 is used for every DSR calculation in this report**, matching the frozen prereg's
declared ledger exactly (`n_trials_ledger_breakdown.matches_declared_48=true`). The disclosed gap
(actual computed cells = 21, not 48) makes the DSR benchmark used throughout this report *harsher*
than the honestly-computed N_trials would require — i.e. conservative for, not inflating, the
negative conclusion — but is disclosed here per §7.3 rather than silently reconciled. Three
lookahead canaries (S1, S2, S3) are excluded from this ledger per the mandatory convention.

**PBO over the full family:** `pbo_full_family.pbo = 0.4143` over the 12-cell aligned family
(S1_L3/L6/L12/ensemble, S2_Wbeta26/52/104, S3_hold2wk/4wk/6wk/ensemble, combined_deployable) — this
**FAILS** the ≤0.25 gate.

---

## 9. Frozen adoption gate — applied explicitly to the combined deployable

| # | Gate | Requirement | Full-universe result | on_demo result | PASS/FAIL |
|---|---|---|---|---|---|
| 1 | Net Sharpe positive | full-universe AND on_demo, both | -0.2932 | -0.0425 | **FAIL** (both negative) |
| 2 | DSR pass at N_trials=48 | dsr ≥ pass threshold | 0.0008 | 0.0090 | **FAIL** |
| 3 | PBO ≤ 0.25 (full 48-cell/12-aligned-cell family) | | 0.4143 | — | **FAIL** |
| 4 | Drop-top-5-periods Sharpe still positive | | -0.7616 | -0.6119 | **FAIL** |
| 5 | Every included sleeve's own regime-sanity signature holds | S1 + S2 + S3 all hold | S1=true, S2=true, **S3=false** | — | **FAIL** |

**ALL_PASS = FALSE. 0 of 5 gates pass.** Gate 5 fails specifically because S3's liquidity-split
regime-sanity signature fails (drift concentrated in the MORE-liquid half, opposite of the PEAD
literature), even though S1 and S2 each independently hold their own signature.

**Overall verdict: the pre-named combined deployable (equal-risk-contribution blend of S1
ensemble + S2 W_beta=52 primary + S3 ensemble, single 10% vol-target overlay, 13bp primary cost)
FAILS the adoption gate. Recommendation: DO NOT DEPLOY — neither paper nor live capital.**

**Secondary finding, reported per the prereg's own convention (not a redefinition of the
deployable):** S1 (momentum) and S2 (true Frazzini-Pedersen BAB) each independently clear their own
regime-sanity signature and post a standalone positive full-universe Sharpe (+0.53 and +0.22 ann
respectively at primary cost). This is qualified, not clean, by §7.4's dependence findings: S1's
edge is 40.1% concentrated in one meme stock (GME) and collapses to DSR=0.056 if the top 3 names
are removed; S2's edge inverts to negative (-0.0792) if 2 of 88 names (LLY, NVO) are removed. A
future two-sleeve (S1+S2-only) combined deployable is NOT retroactively substituted for E3's frozen
three-sleeve deployable here — per house discipline this would require its own fresh, separately
frozen pre-registration, and any such follow-up must explicitly address the concentration findings
above before treating either sleeve as a clean building block.

---

## 10. Full test suite

`cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/python -m pytest tests -q`

**654 passed, 1 skipped, 9 warnings** — unchanged from the E1/E2 baseline (re-run fresh during
independent verification, reproduced exactly, not merely trusted from a log).

---

## 11. Conclusion and recommendation

E3's three-sleeve thesis — sector-residual momentum, true Frazzini-Pedersen BAB, and price-based
PEAD, on ~93 TradFi-equity Binance perps at perp-reality costs — **is a clean, thoroughly-verified
negative for the combined deployable named in the frozen prereg.** All five adoption gates fail;
PBO (0.41) is nearly double the 0.25 ceiling; S3's own regime-sanity signature fails on its more
load-bearing sub-check (liquidity concentration in the WRONG direction relative to the PEAD
literature); and three independent adversarial lenses each surfaced a real, disclosed issue — an
S1 skip-month construction bug (likely understates, not inflates, S1's true Sharpe), a funding-cost
cadence/unit bug and an N_trials undercounting gap in the cost grid (both conservative for, not
inflating, the negative verdict), and severe single-name/single-sector concentration in BOTH of the
two sleeves that otherwise look like secondary positives (S1: 40% one meme stock; S2: entire edge
in two pharma names) — none of which reverses the combined deployable's failing verdict, but all of
which materially qualify how "clean" S1's and S2's own standalone results actually are.

**Recommendation: DO NOT DEPLOY this construction, to paper or live capital, in its current form.**
A correctly-identified, thoroughly-diagnosed dead end is a complete and valuable result, per this
program's own frozen discipline. Any follow-up must first (a) fix S1's skip-month embargo
(`shift(skip)` → `shift(skip+1)`) and its month-end eligibility indexing (`LISTED_M[t]` →
`LISTED_M[t-1]`) and re-run S1's numbers before treating them as a clean baseline; (b) fix the
funding-scenario cadence conversion and actually compute the full declared cost/funding grid at all
four headline cells, not just the combined deployable; (c) explicitly address the GME/LLY-NVO
single-name dependence before treating S1 or S2 as a candidate building block for any future
two-sleeve program; and (d) treat S3's liquidity-concentration finding as a genuine, literature-
contradicting characteristic of PEAD on this venue's universe, not an artifact to explain away.

---

## 12. Cross-program synthesis (E1 + E2 + E3)

See `research/equity/STATE.md` for the consolidated three-program conclusion and full outstanding-
issue ledger. In summary: across three independent, rigorously pre-registered, independently
verified and adversarially reviewed programs — E1 (funding-carry, perp/underlying basis reversion,
overnight/weekend structure), E2 (stress-conditioned sector-residual reversal), and E3 (sector-
residual momentum, true BAB, price-based PEAD) — **no construction has cleared its own adoption
gate.** Where an individual sleeve or cell has shown a standalone positive Sharpe (E3's S1 and S2),
independent adversarial review has in every such case found the apparent edge concentrated in a
small number of idiosyncratic names or structurally-decaying products rather than a genuine,
diffuse cross-sectional effect. The honest cross-program conclusion, stated plainly, is a
comprehensive negative for swing-horizon (weekly-to-monthly) equity cross-sectional and
structural-carry edges on this venue's ~90–110-name Binance-listed TradFi-equity perp universe at
its real cost structure, at least among the seven mechanisms tested to date (funding carry,
basis reversion, overnight-structure, sector-residual reversal, sector-residual momentum, BAB,
PEAD).
