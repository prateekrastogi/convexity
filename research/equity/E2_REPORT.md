# E2 — STRESS-CONDITIONED RESIDUAL REVERSAL — FINAL REPORT

Program 6. Frozen prereg: `research/equity/E2_PREREG.md` (frozen 2026-07-06, before any pipeline
code touched real data). Pipeline: `research/equity/e2_pipeline.py`. Raw artifact:
`research/equity/results_e2.json` (md5 `137100ae3bfd2ef0ca1f69fa341aa25c`, committed verbatim).
Runtime 18.6s.

**Headline verdict, stated up front: the pre-named deployable FAILS the adoption gate on all five
gates. This is a clean, thoroughly-verified negative. Recommendation: DO NOT DEPLOY, do not
paper-trade this construction.** A genuine, previously-undetected lookahead bug in the
conditioning layer was found by adversarial review after the pipeline run (see "Post-run
adversarial findings" below); it does not rescue the result and is disclosed in full rather than
silently fixed and re-run, per the requirement to report exactly what happened.

---

## 1. Universe, survivorship, and demo-gate caveats (repeated at every headline number below,
   not just here)

- Base declared universe: 102 names (`candidates.md`'s `equities` array, real-history subset).
  **Actually usable in this run: 99** — `STXXUSDT`, `CBRSUSDT`, `DRAMUSDT` excluded because their
  real `source=='duka'` history (11–33 calendar days at the data cut) is too short to clear even
  the shortest lookback (L=3) plus the 20-day vol window plus a 5-day buffer. `results_e2.json`'s
  own `meta.universe_n_base_102` field literally holds `99` (a slightly misleading field *name*,
  not a computation error — `meta.panel_span.n_syms` correctly states 99 elsewhere in the same
  block, and the independent verifier confirmed both the pipeline and an independent from-scratch
  rebuild land on the identical 99).
- **Survivorship (landmine 4, restated at every headline number in this report):** the 99-usable
  name roster is today's Binance-listed winners, not a point-in-time index-constituent universe.
  This is **less fatal** for a short-horizon cross-sectional reversal study than it would be for a
  momentum study — the within-cross-section *ranking* each week is what matters, not which names
  happen to be in the 2026 roster — but it is **not nil**: a name's mere presence in today's
  roster could still correlate with its historical volatility/liquidity profile in a way that
  shapes which names populate the reversal book's tails most often. See §7 for evidence this
  concern is in fact realized (UVXY/SQQQ concentration).
- **Demo-gate (landmine 5, restated at every headline number):** only 19/110 names trade on the
  Binance DEMO venue today. Of those, 17 have real underlying history and are in-scope for E2
  (`OPENAIUSDT`, `QNTXUSDT` are on_demo but perp-only, excluded from E2's base universe entirely).
  Every headline number below is reported both **full-99-usable-universe** and
  **on_demo-and-real-history (17-name)** — the 17-name subset is what could actually be
  paper-traded today, per E1's identical precedent.

---

## 2. Core signal construction (as frozen; no deviations)

Daily close built from `source=='duka'` rows only (verified never to cross the splice seam — see
§8). Weekly (Friday) sampling. Residualized against sector-peer mean (sectors folding into "other"
below a 4-name point-in-time floor, per prereg). Trailing-L-day residual return ranked at each
Friday; quintile book LONG bottom quintile (biggest residual losers) / SHORT top quintile (biggest
residual winners) — **reversal sign confirmed correct** throughout (not momentum). Inverse-vol
leg-weighted, dollar-neutral, unit gross before overlay. Band-ensemble of L∈{3,5,10} is the
declared, never-argmaxed construction.

---

## 3. Unconditional metrics table (full 99-name universe; GROSS and NET shown together)

All cells: n_obs=411 weekly observations, span 2018-05-01→2026-07-01 (8.13yr). `turn_mean` is
genuinely wired via `metrics()`'s `turn=` parameter this run (the E1 regression — `turn=` accepted
but never passed at any call site — is fixed; every cell below has a populated, non-null
`turn_mean`).

| Construction | Cost (rt bp) | Net Sharpe (ann) | Gross Sharpe (ann) | EV bp/obs (net) | Win % | MaxDD % | CAGR % (raw) | CAGR % (10%-vol) | Turn (x/wk) | Drop-top5 Sharpe | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L=3 only | 13.0 | **−1.2411** | −0.5392 | −36.99 | 39.17 | 80.57 | −18.05 | −15.12 | 1.601 | −1.6326 | 0.0 |
| L=5 only | 13.0 | **−1.4378** | −0.7116 | −40.98 | 42.58 | 83.83 | −19.61 | −16.16 | 1.579 | −1.7119 | 0.0 |
| L=10 only | 13.0 | **−1.3747** | −0.8543 | −41.16 | 40.88 | 84.05 | −19.79 | −16.89 | 1.186 | −1.6868 | 0.0 |
| **Ensemble (declared, rt6.5)** | 6.5 | −1.1121 | −0.7904 | −29.67 | 41.85 | 74.03 | −14.75 | −13.70 | 1.309 | −1.4072 | — |
| **Ensemble (declared, rt13 primary)** | 13.0 | **−1.4361** | −0.7904 | −38.18 | 40.39 | 81.12 | −18.35 | −16.94 | 1.309 | −1.7452 | 0.0 |
| **Ensemble (declared, rt26)** | 26.0 | −2.0908 | −0.7904 | −55.20 | 35.77 | 90.46 | −25.11 | −23.15 | 1.309 | −2.4284 | — |

The unconditional band-ensemble is net-negative at every cost scenario, including the cheapest
(6.5bp). Per-year Sharpe for the primary ensemble/rt13 cell is negative in every calendar year
2018–2026 (range −0.20 in 2018 to −4.31 in partial-year 2026); this is a durably negative signal
across the full sample, not one bad stretch.

---

## 4. Conditioned-grid summary (72 cells) and the pre-named deployable

The full 72-cell conditioned grid (W_rv∈{10,20,40} × W_z∈{252,504} × source∈{SPY,QQQ,MEAN} ×
mechanism∈{hard-gate×3 thresholds, continuous-scale} at rt13.0) is committed verbatim in
`results_e2.json`. **Best individual cell of all 72: `Wrv10_Wz504_MEAN_hard_t1.0`**,
sharpe_ann=+0.2141 (gross +0.3742), DSR=0.0274, cpcv_frac_pos=0.704, drop-top5=−0.3551 (turns
negative once the top 5 weeks are dropped). This is nowhere near a DSR pass at N_trials=104 —
**no cherry-picked cell in the conditioned grid survives multiplicity charge.**

### Pre-named deployable (decided in the prereg, before any run)

Construction: band-ensemble(3,5,10), full-universe, continuous-scale conditioning
(`clip(1+max(0,z),1,3)`), W_rv=20, W_z=252, stress-source = mean(SPY,QQQ) z, primary cost 13bp,
0bp/wk funding.

**Annualization arithmetic (per METHODOLOGY 1c, shown so a reader can check it):**

| n_obs | span_years | periods_per_year_derived | Friday constant (365.25/7) | relative error |
|---|---|---|---|---|
| 411 | 8.1259 | **50.5788** | 52.1786 | 3.07% (< 5% tolerance) |

`sharpe_ann` is computed from `sharpe_obs × sqrt(50.5788)`, the **derived** value, not the
hardcoded Friday constant — independently confirmed (§8): `−0.13412 × sqrt(50.5788) = −0.9538`
(matches reported −0.9539), vs. what the E1-style hardcoded-52.1786 bug would have produced
(`−0.9688`) — a different, larger-magnitude number, proving this run does **not** repeat E1's
annualization bug.

| | Net Sharpe (ann) | Gross Sharpe | EV bp/obs | Win % | MaxDD % | CAGR % raw | CAGR % 10%-vol | Turn (x/wk) | Drop-top5 | DSR | PSR vs 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Full 99-usable universe** | **−0.9539** | not separately reported (gross omitted at cell level; unconditional comparator's gross = −0.7904 for context) | −42.02 | 40.39 | 86.80 | −21.23 | −18.54 | 1.800 | −1.4534 | 0.0 | 0.003 |
| **on_demo-and-real-history (17-name)** | **−0.6441** | — | −73.89 | 42.09 | 99.66 | −42.62 | −21.06 | 1.810 | −1.2370 | 0.0 | 0.040 |

Per-year Sharpe (full-universe deployable): positive only in 2018 (+0.083, partial year, n=34);
negative every other year, worst in 2026 partial (−3.685). `turn_mean`=1.80/week (full) /
1.81/week (on_demo) — consistent with the prereg's own ~2x/week turnover expectation for a
weekly-rebalanced quintile reversal book, and now genuinely sanity-checkable from the artifact
itself (E1's `turn=` wiring gap is fixed).

### Cost/funding sensitivity (deployable + unconditional comparator, 18-cell grid)

| Cost scenario | Deployable net Sharpe | Unconditional-ensemble net Sharpe |
|---|---|---|
| rt=6.5bp, fund=0 | −0.6851 | −1.1121 |
| rt=6.5bp, fund=+5bp/wk | −0.7981 | (see grid) |
| rt=6.5bp, fund=−5bp/wk | −0.5721 | (see grid) |
| **rt=13.0bp, fund=0 (primary)** | **−0.9539** | **−1.4361** |
| rt=13.0bp, fund=+5bp/wk | −1.0674 | — |
| rt=13.0bp, fund=−5bp/wk | −0.8404 | — |
| rt=26.0bp, fund=0 | −1.4971 | −2.0908 |
| rt=26.0bp, fund=+5bp/wk | −1.6115 | — |
| rt=26.0bp, fund=−5bp/wk | −1.3827 | — |

Net Sharpe is negative at **every** cost/funding combination tried, for both the deployable and
the unconditional comparator — this is not a marginal, cost-scenario-dependent result. **Caveat
(disclosed, not found post-hoc but flagged by the cost-realism skeptic pass, §7):** the funding
term (`funding_bp_per_wk / 1e4`) is applied as a flat additive constant regardless of the book's
current gross exposure, while the trading-fee term correctly scales with the exposure-scaled
turnover (`turn_scaled = turn_raw × es`). Since the deployable's whole thesis is that gross grows
to 2–3x specifically during stress weeks, a fully realistic funding-stress scenario should charge
proportionally more funding drag exactly in those higher-conviction weeks; the current `fund5.0`
scenario does not do this, so it is not as "harsh" for a levered book as intended. This does not
change any conclusion here (the book is already deeply net-negative at fund0.0) but would flatter
a future levered-and-profitable variant of this construction and should be fixed before reuse.

---

## 5. Universe-size sweep (10 points, honest CI-overlap treatment)

| N | Sharpe (point estimate) |
|---|---|
| 20 | −0.5842 |
| 30 | −0.5409 |
| 40 | −0.6059 |
| 50 | −0.7565 |
| 60 | −0.8061 |
| 70 | −0.7288 |
| 80 | −0.9249 |
| 90 | −0.9168 |
| 100 | −0.9539 |
| **102 (baseline)** | **−0.9539** |

Baseline N=102's 90% block-bootstrap CI (block=8wk, 1000 draws): **[−1.4899, −0.3763]** (median
−0.9253). N=20's point estimate (−0.584) looks better, but `any_honest_peak_found=False` for
**every** non-baseline point — none clears "CI lower bound beats baseline's AND <50% overlap."
E.g. N=20 vs N=102 overlap = 64.5% of the narrower CI's width (needs <50%), and N=20's CI lower
bound does not exceed N=102's CI lower bound by enough to be distinguishable.

**Verdict recorded verbatim: "SUPPORTS (N=102 baseline is not honestly beaten by any sweep
point)."** Independently re-verified by the dependence skeptic with a different bootstrap seed
(123 vs. pipeline's 42) and different block lengths (4wk and 8wk): N=40 overlaps baseline by
66–72%, N=70 by 77–81% — neither comes close to the pre-registered bar under any variant tried.
The "no distinguishable peak" finding is robust, not a lucky single bootstrap draw — though this
is unsurprising given there is no real edge at any breadth in this family to have a peak in.

---

## 6. Regime-sanity signature (the mandatory mechanism check) — FAILS

Declared exact signature (prereg): above-trailing-median-stress weeks must contribute **>50%** of
total P&L despite being a minority of calendar weeks, AND all four named stress windows must
individually show weekly Sharpe exceeding the deployable's own full-sample Sharpe.

**(1) P&L concentration split:**

| | Value |
|---|---|
| Frac. calendar weeks above trailing-median stress z | 42.09% |
| Frac. of total P&L from above-median weeks | **47.91%** (needed >50%) |
| Total P&L | −1.727128 |
| P&L from above-median weeks | −0.827407 |
| P&L from below-median weeks | −0.899721 |

Missed the >50% bar by ~2 points — above-median weeks were a slight *minority* contributor to
(negative) total P&L, not the required majority.

**(2) Named-window Sharpe-exceeds-full-sample check (full-sample Sharpe = −0.9539):**

| Window | n_obs | Weekly Sharpe (ann) | Exceeds full-sample? | Sum net return |
|---|---|---|---|---|
| 2018 Q4 | 13 | +0.399 | **PASS** | +0.0340 |
| 2020-03 COVID | 4 | −2.1476 | **FAIL** | −0.1604 |
| 2022 drawdown | 51 | −0.5815 | **PASS** | −0.1802 |
| 2024-08 vol spike | 5 | −3.4317 | **FAIL** | −0.0834 |

Only 2 of 4 named windows pass; `all_named_windows_exceed_full_sample_sharpe=False`, so
**`SIGNATURE_HOLDS=False`**. The 2020-03 and 2024-08 windows have very few observations (n=4,
n=5) — a genuinely noisy small-sample read, stated here rather than papered over. Per the
DURABLE_REPORT.md diagnostic discipline, this failure was investigated before being trusted: the
stress-signal construction was re-verified NaN-correct past each stress name's real history end
(the ffill-past-end-of-history bug found and fixed during pre-run smoke-testing, §9), the
window-return computation reuses the identical `source=='duka'`-only pipeline as the core signal
(no separate splice exposure), and the deployable's own lookahead canary (which shares the same
conditioning layer) confirmed the correct inflated-Sharpe direction. **This is treated as a
genuine negative finding about the mechanism, not a construction bug** — subject to the caveat
in §7 that the exposure-to-week alignment used for this check itself had a one-week lookahead
shift (disclosed below; the independent spot-check found the "2022 exceeds full-sample" flag is
only true because the full-sample average is so deeply negative that almost any window beats it
mechanically, not because P&L genuinely concentrates in stress).

---

## 7. Post-run adversarial findings (three independent skeptic lenses; all three found real,
   disclosed issues — none reverses the headline negative verdict)

### 7a. Lookahead lens — CONCERN (genuine bug found)

**A real, previously-undetected lookahead bug exists in the conditioning/exposure-scale layer.**
In `e2_pipeline.py`'s conditioned-grid loop (~line 1014–1025) and the deployable/on_demo/sweep
construction (~line 1091–1092, 1139):

```python
z_at_fri = z_daily[fri_idx]        # z_at_fri[k] = stress z AT Friday fri_idx[k]'s close
...
expo_weekly = expo_at_fri[1:]      # BUG: drops z_at_fri[0], keeps z_at_fri[1..]
```

`score_book_weekly` dates `wret[k]` (return earned from week-t weights, held to week t+1) at
`fri_idx[k+1]` — i.e. `wret[k]` corresponds to `z_at_fri[k]`, the stress reading at the week's
*start*, under an honest construction. But `expo_weekly[k] = expo_at_fri[k+1] = gate(z_at_fri[k+1])`
— the exposure scale applied to week k's return uses the stress z-score measured at the **end** of
that week (Friday t+1's close, the same day the return is marked), not the start (Friday t, when
the position was actually sized). Trading this live at Friday t would require already knowing next
Friday's realized-vol z-score — impossible in real time. Correct fix: `expo_at_fri[:-1]`.

**Scope of impact:** baked into all 72 conditioned-grid cells, the deployable's
`headline_full_102`/`headline_on_demo_17`, the universe-sweep's 10 cells, the cost/funding grid's
deployable row, and the **regime-sanity check itself** (which uses the same shifted
`z_at_fri[1:]` series for the ">50% of P&L from above-median-z weeks" test).

**Why it very likely does not flip the conclusion (disclosed, not assumed):** realized-vol
z-scores are highly autocorrelated week-to-week (stress episodes persist multiple weeks), so a
one-Friday shift mostly relabels which exact week gets credit for a stress episode rather than
manufacturing conditioning where none existed. Given the deployable's result is already a clean,
decisive negative (DSR=0, all 5 adoption-gate checks fail, regime-sanity already fails at
frac_pnl_above=0.479<0.50 with 2/4 named windows already failing), correcting the shift would not
be expected to rescue it — if anything, removing a favorable one-week information edge should make
the regime-sanity numbers look equally or slightly worse. **This is a real defect that should be
fixed and the conditioned grid/deployable/sweep/regime-sanity numbers re-run before this
construction is ever reused or extended — it was not caught by the pipeline's own asserts or by
the first-pass independent verification** (which checked splice-seam guard, annualization
constant, one hand-derived sleeve, canary direction, and a regime-window spot-check, but did not
re-derive the exposure-to-week alignment from scratch). It is disclosed here in full per the
"never fabricate, never paper over" convention rather than silently patched and silently re-run.

Everything else the lookahead lens checked held up under direct code inspection: core ranking
signal and book weights use only information through Friday t's close (no overlap with the
forward-return window); the canary's own alignment is internally consistent; the PIT `LISTED`
mask uses genuine first/last real `duka` dates; sector-residualization groups are recomputed fresh
per day from true PIT membership (no static/future leakage); the stress z-score's rolling
mean/std are correctly `.shift(1)`'d; `weekly_friday_close()` is dead code (Friday-sampling
actually runs via `cal.dayofweek==4` on the raw trading-day calendar — a docs/cleanup note, not a
correctness issue); the universe-sweep's dollar-volume ranking is causal. One real (non-lookahead)
construction inconsistency: the sweep reuses `RESID` computed against the full-99-name sector
groups even when restricting the tradeable book to N<99 names.

### 7b. Cost-realism lens — CONCERN

Turnover is genuinely wired into `metrics()` this run (E1's gap fixed, confirmed non-null
everywhere: 1.19–1.60/wk per-lookback, 1.309 unconditional ensemble, 1.80/1.81 deployable). Net
Sharpe is negative at every rt/funding combination actually swept (2 of 104 cells; see §4's
disclosed scope gap on why only 2 cells got the full cost/funding sweep — a declared, not hidden,
scoping choice). The one real bug found: **funding cost is not scaled by the deployable's own
exposure multiplier** (see §4 cost/funding caveat) — confirmed empirically, `ev_bp_obs` moves by a
flat ±5.0bp between fund scenarios at every rt level, identical in magnitude to the
always-1x-gross unconditional comparator's own ±5.0bp delta, even though the deployable's gross
grows to 2–3x in stress weeks. Directionally immaterial to this program's conclusion (already
deeply net-negative at fund0.0) but a real internal-consistency defect in the cost model that
would flatter a future profitable, levered variant.

### 7c. Dependence lens — CONCERN (single-name concentration, previously undisclosed)

Independent reproduction of the deployable (imported the pipeline's own functions, rebuilt the
full chain from raw parquet) reproduced sharpe_ann=−0.9539 exactly, confirming the jackknife
harness is faithful. Attributing total gross weekly P&L to individual names revealed **extreme
concentration, not a diffuse ~99-name cross-sectional effect**:

- **UVXYUSDT alone accounts for 20.2% of total gross P&L.**
- Top 4 contributors (UVXYUSDT, SQQQUSDT, ASTSUSDT, MSTRUSDT) account for **>50%**.
- Top 15 of ~99 names account for **~86%**.
- Zeroing UVXYUSDT alone: Sharpe moves −0.9539 → −0.859 (~10% relative, still negative).
- Zeroing top 3 names (3% of universe): Sharpe moves to −0.7487 (21.5% relative improvement,
  confirmed not a renormalization artifact — redistributing weight after dropping UVXY gives
  −0.8624, essentially the same as simple zeroing).

**Compounding concern:** UVXYUSDT (`vol-etf`, singleton sector tag) and SQQQUSDT
(`inverse-etf-nasdaq`, singleton tag) are confirmed in `candidates.md` as leveraged/inverse-decay
ETF products (2x long-vol-decay and 3x inverse-Nasdaq respectively). Under the prereg's own
`MIN_SECTOR_N=4` rule, singleton-tag names never get a true peer group and fold into the large,
heterogeneous "other" bucket for their entire history — so their structural negative decay drift
(a well-known property of leveraged/inverse products, independent of any reversal mechanism) is
only partially residualized away. **Part of what this book trades is "is this leveraged-decay
product this week's biggest residual mover," not genuine cross-sectional reversal among ordinary
equities** — a real scope/construction concern regardless of the sign of the result, and one the
original prereg and pipeline did not flag.

Sector-level dependence is weaker: the largest true multi-name sector (semis, 13 names)
contributes only −0.0416 of the −0.870 total gross P&L; zeroing it moves Sharpe only
−0.9539→−0.9203. Period dependence: negative in every calendar year except partial-year 2018
(+0.012, n=34); no single year flips the sign (dropping 2018 → −1.0421 worse; dropping worst year
2024 → −0.8298, still clearly negative). **The negative conclusion is durable across years and
across true sector groupings — the fragility is specifically to a handful of non-representative
leveraged/inverse-ETF names, not to any single calendar period.**

The universe-sweep's "no honest peak" claim was independently re-verified (different seed,
different block lengths) and holds up robustly (§5).

**Bottom line across all three lenses:** the adoption-gate failure (0/5 gates passing) is not an
artifact of one bad year, one dominant true sector, a hidden cost omission, or a papered-over
canary failure — it is a genuinely durable negative. But it rests on a construction with (a) one
confirmed lookahead bug in the conditioning layer (disclosed, likely immaterial to sign, not yet
fixed/re-run), (b) an inconsistent funding-cost scaling bug (disclosed, immaterial to sign at
these Sharpe levels), and (c) material undisclosed reliance on structurally-decaying
leveraged/inverse ETF names that a Nagel liquidity-provision thesis was not designed to explain. A
follow-up program should fix all three before treating any part of this construction as a
foundation to build on.

---

## 8. Independent verification of the pipeline's own claims (pre-adversarial-review pass)

- **Splice-seam guard: VERIFIED.** Directly inspected raw `PAYPUSDT.parquet`: at
  `open_time=1774276200000` the `source` column flips duka→perp with `gap_before=False` (matches
  the pipeline's own claim) and a real price discontinuity exists (19.2204 duka close → 22.03 perp
  open, a fake +14.6% instantaneous jump) — exactly the E1 bug-pattern trap. Independently
  reproduced `e2_pipeline.py`'s `load_stitched_duka`+`daily_close_series_duka` logic in a
  standalone script (not importing the pipeline): the duka-only filter contains 772,530 rows with
  **zero** rows after the seam timestamp (perp's 148,890 rows are all after it) — the duka filter
  physically cannot reach past the seam. Constructed daily-close series' last day is 2026-03-23,
  close=19.220403 (the real last duka price), never 22.03. Extreme daily-return values (min −25.1%
  on 2022-02-02, max +14.1% on 2020-03-24) are real PayPal earnings-crash/COVID-rebound days, not
  splice artifacts. **This pipeline does NOT repeat E1's bug pattern (a).**
- **Annualization constant: VERIFIED as genuinely derived, not hardcoded.** Recomputed
  `sharpe_obs × sqrt(50.5788) = −0.95384` (matches reported −0.9539); the hardcoded-52.1786
  alternative would give −0.96881, a different number, proving the derived value is genuinely
  used. Runtime log (`e2.out`) contains the matching derivation line printed live during the
  actual run (not hand-edited post-hoc): `n_obs=411 span_years=8.1259 derived=50.5788`. Zero
  ">5% off" WARNING lines anywhere in the full log (grep -c = 0) — every weekly-marked cell landed
  within the 5% tolerance. **This pipeline does NOT repeat E1's bug pattern (b).**
- **Hand-derived sleeve match: VERIFIED.** Unconditional L=5 book at rt=13bp, hand-derived
  −1.443 vs. pipeline's −1.4378 (match; small residual difference attributable to a slightly
  looser inclusion filter admitting 2 extra thin-history names in the hand-derivation, win_pct and
  turn_mean matched exactly to 4 decimals).
- **Canary direction: VERIFIED correct** (+10.53 net / +10.65 gross ann, clearly dominating the
  honest −0.9539).
- **Regime-window spot check:** independently rebuilt the deployable from raw data (not importing
  the pipeline); headline Sharpe −0.9516 vs. pipeline's −0.9539 (0.24% relative difference, both
  landing on the identical 99 usable names — the 99-vs-102 gap is a genuine, disclosed
  data-availability artifact, not a code bug). 2022-drawdown window numbers matched the pipeline
  exactly (sum_net_return=−0.180232, weekly_sharpe_ann=−0.5815). Comparing against a calm 2023 H1
  window (sharpe_ann=+0.0078, near flat) showed the "stress window exceeds full-sample Sharpe"
  flag is only true because the full-sample average is so deeply negative that almost any window
  beats it mechanically — corroborating `SIGNATURE_HOLDS=False`'s substance, not just its form.

**No discrepancies found that would flatter the reported result** in this first verification
pass; the one substantive gap (the exposure-to-week lookahead shift) was caught only in the
subsequent, dedicated adversarial lookahead pass (§7a), underscoring that even a careful
first-pass verification did not check every axis.

---

## 9. Bugs caught and fixed BEFORE the reported run (pre-run smoke-testing discipline)

Two real, previously-undocumented bugs were found and fixed during this program's own mandatory
pre-run smoke-testing — caught before any headline number was trusted, not discovered post-hoc:

**(a) Stress-signal ffill-past-end-of-history bug.** `build_stress_signal`'s SPY/QQQ daily-close
series originally used `reindex(cal).ffill()`, forward-filling each name's last real `duka` close
indefinitely across the ~3 months after its own duka→perp splice date (SPY's real duka history
ends 2026-04-06, QQQ's 2026-04-02, but `cal` runs through 2026-07-01) — a flat, fabricated,
zero-variance tail that made the rolling realized-vol z-score decay smoothly to a manufactured
value (verified: SPY's buggy z-score decayed smoothly to −1.85 by 2026-07-01 instead of correctly
going NaN) — exactly the kind of fake smoothness the regime-sanity gate exists to catch. Fixed by
explicitly NaN-ing any day past each stress name's own real last `duka` day before computing
log-returns, so the stress signal (and `gate_continuous`'s exposure, which treats non-finite z as
base 1.0x) goes genuinely inert past real data's end.

**(b) Lookahead-canary sign/construction bug (caught by the mandatory canary-dominance check
working as designed).** First attempt shifted the trailing-sum window forward by 1 day (a
diluted, mostly-non-overlapping 1-of-L-day leak) and showed canary Sharpe −6.07, *worse* than the
honest −1.44 — immediately flagged wrong by the "canary must clearly dominate" check. Rebuilt to
rank directly on the actual realized forward return `wret[t]`, but required negation before
feeding the existing reversal weight function (an un-negated first attempt produced an impossible
−18.88 canary Sharpe — "long the most-negative-signal names" applied to `+wret[t]` means "long the
names about to have the worst return," exactly backwards for a canary). Post-fix: +10.53 net /
+10.65 gross ann, correctly and clearly dominating.

Both were caught and fixed before the reported run, directly informed by having just read E1's
"two real bugs found late" lesson — a positive process outcome even though (per §7) a third,
different bug (the exposure-to-week lookahead shift) escaped this same smoke-testing discipline
and was only caught by dedicated post-hoc adversarial review.

---

## 10. Lookahead canary (mandatory, excluded from family/PBO/DSR)

| | Net Sharpe (ann) | Gross Sharpe (ann) | Win % | MaxDD % | Turn (x/wk) |
|---|---|---|---|---|---|
| Canary (deliberately wrong) | **+10.526** | +10.646 | 99.51 | 0.13 | 2.239 |
| Honest deployable | −0.9539 | — | 40.39 | 86.80 | 1.800 |

`canary_clearly_dominates = True`. The honest pipeline is not silently degenerate — a real,
important verification, not a formality.

---

## 11. N_trials ledger (full, as declared — nothing hand-picked)

| Component | Cells |
|---|---|
| Unconditional (ensemble + 3 per-lookback) | 4 |
| Conditioned grid (3×2×3×4) | 72 |
| Universe-size sweep | 10 |
| Cost/funding grid (2 headline × 3rt × 3funding) | 18 |
| **Total charged to DSR** | **104** |

Matches declared 104 exactly (`n_trials_ledger_breakdown.matches_declared_104=True`). Lookahead
canary excluded per mandatory convention (bug-detector, not a candidate edge).

**PBO over the full family:** `pbo_full_family = 0.20` (86 cells actually poolable into one
aligned CSCV family) — this **clears** the ≤0.25 gate on its own, but does not save the overall
verdict since gates 1/2/4/5 all fail (see §12).

---

## 12. Frozen adoption gate — applied explicitly

| # | Gate | Requirement | Full-99 result | on_demo-17 result | PASS/FAIL |
|---|---|---|---|---|---|
| 1 | Net Sharpe positive | full-universe AND on_demo-17, both | −0.9539 | −0.6441 | **FAIL** (both negative) |
| 2 | DSR pass at N_trials=104 | dsr ≥ some pass threshold | dsr=0.0 | — | **FAIL** |
| 3 | PBO ≤ 0.25 (full 104-cell family) | | 0.20 | — | **PASS** |
| 4 | Drop-top-5-weeks Sharpe still positive | | −1.4534 | −1.2370 | **FAIL** |
| 5 | Regime-sanity signature holds in full | >50% P&L from above-median weeks AND all 4 named windows exceed full-sample Sharpe | 47.91% P&L; 2/4 windows pass | — | **FAIL** |

**ALL_PASS = FALSE.** 1 of 5 gates passes (PBO); the other four — including the two most
important (positive net Sharpe, and the regime-sanity mechanism check that is this program's own
central thesis test) — fail decisively.

**Overall verdict: the pre-named deployable (continuous-scale-conditioned band-ensemble residual
reversal, full 99-usable-name universe, W_rv=20/W_z=252/mean-SPY-QQQ-z, 13bp primary cost) FAILS
the adoption gate. Recommendation: DO NOT DEPLOY — neither paper nor live capital.** No individual
cell elsewhere in the 104-cell family independently clears all five gates either (the best
conditioned cell, `Wrv10_Wz504_MEAN_hard_t1.0` at +0.2141 Sharpe, fails DSR decisively at
dsr=0.0274). Per the frozen prereg's own discipline, a failed pre-named deployable is reported as
this program's headline result, not silently swapped for a different construction after the fact.

---

## 13. Did this pipeline repeat either of E1's two bug patterns?

**No, not the two specific patterns E1 documented** (source-aware daily-return construction across
the splice seam; annualization constant matched to true observation spacing) — both were
independently verified as correctly handled in this run (§8). **However, this pipeline has its own,
different, previously-undetected bug of the same general character** (a data/timing-alignment
subtlety that escaped both the pipeline author's own smoke-testing and a first-pass independent
verification, caught only by dedicated adversarial review): **the conditioning-layer exposure
scale is shifted one Friday ahead of the return it is applied to, a genuine lookahead bug**,
disclosed in full in §7a. This is stated plainly, not buried, even though it is not flattering:
E2's prereg explicitly declared a smoke-testing discipline motivated by "not repeating E1's
mistakes," and that discipline caught two different, real bugs (§9) but missed a third
(exposure-alignment lookahead) of a similar class — the general lesson (every timing/alignment
claim needs to be independently re-derived, not just asserted and spot-checked once) generalizes
beyond the two specific patterns E1 named.

---

## 14. Full test suite

`cd /home/speculation/convexity-equity && PYTHONPATH=/home/speculation/convexity-equity/src:/home/speculation/convexity-equity /home/speculation/convexity/.venv/bin/python -m pytest tests -q`

**654 passed, 1 skipped, 9 warnings** — unchanged from the E1 baseline; `e2_pipeline.py` (a new,
self-contained file) did not touch or break any existing `src/` behavior.

---

## 15. Conclusion and recommendation

E2's central thesis — a Nagel-style stress-conditioned cross-sectional residual reversal on ~100
TradFi-equity Binance perps, using underlying `duka` history for signal construction and
perp-reality costs — **is a clean, well-verified negative**. Every adoption gate fails except PBO;
the regime-sanity mechanism check (the thesis's own central claim, that the reversal premium
fattens during genuine stress) fails on both of its two required prongs; and three independent
adversarial lenses each surfaced a real, disclosed construction issue (a conditioning-layer
lookahead shift, a funding-cost scaling inconsistency, and material reliance on
structurally-decaying leveraged/inverse-ETF names rather than genuine cross-sectional reversal
among ordinary equities) — none of which reverses the sign of the headline result, but all of
which should be fixed before any future work builds on this construction.

**Recommendation: DO NOT DEPLOY this construction, to paper or live capital, in its current form.**
A correctly-identified, thoroughly-diagnosed dead end is a complete and valuable result, per this
program's own frozen discipline — this is not a failure to find an edge, it is a successful,
well-instrumented determination that this specific edge does not survive at this breadth, this
cost structure, and this conditioning mechanism. Any follow-up must first (a) fix the
exposure-to-week lookahead alignment and re-run the conditioned grid/deployable/sweep/regime-sanity
numbers, (b) fix the funding-cost exposure-scaling inconsistency, and (c) either exclude
leveraged/inverse-decay ETF products from the base universe or explicitly acknowledge the
construction is partly trading ETF-decay-vs-residual-move rather than pure Nagel-style
liquidity-provision reversal, before treating any part of this family as a candidate for
deployment.
