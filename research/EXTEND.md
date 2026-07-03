# Extension study E1-E3 — PRE-REGISTRATION (panel history / maker execution / carry sleeve)

**Pre-registered before any result is seen** (commit timestamp = the record). Follow-on to
research/ROBUST.md; same discipline: fixed constants, fixed decision rules, frozen accounting,
official validation config (CPCV N=8/k=2, embargo=3wk official + embargo-1 historical).

## E1 — extend the panel to the start of Binance USDT-M perp history (data, not a variant)

Backfill data.binance.vision UM dumps back to the venue's launch (HISTORY_YEARS 5 -> 7; dumps
begin ~2019-12/2020-01; months before a symbol's listing 404 and are skipped — point-in-time by
construction). Consolidate via the existing stitch (perp-only path), rebuild the panel, and re-run
UNCHANGED: the 7-lookback family, CPCV/PBO at embargo 3 (official) and 1 (historical), DSR/MinTRL,
and the R1 7-phase table. NOTHING about the strategy changes — this is more out-of-sample-ish
history for the SAME pre-committed configuration (every choice was frozen before this data was
downloaded, so the added ~2020 weeks cannot have influenced any choice; the panel's early breadth
will be thin — the engine's existing >= 2-names-per-leg floor and PIT machinery handle it).
Decision rule: none needed (a measurement, not an adoption). Record old-vs-new side by side;
flag loudly if the extended-panel CPCV mean drops below 0.5 (the EDGE floor) or the phase
concentration INVERTS (Friday no longer max — would reclassify R1 toward luck).

## E2 — maker-first execution tactic + fee-scenario restatement (deployment layer)

(a) TACTIC (code, EXECUTION pillar): at weekly cadence with no urgency, child orders default
POST_ONLY_MAKER with a taker fallback for any unfilled remainder at the window end — a pure
function of (urgency, time_remaining_frac) over the existing core Tactic ladder; RED constants:
maker window = the whole delta window minus one final slice (no new fitted numbers).
(b) SCENARIO (research, NOT the official record): restate the band-ensemble record at the
CONSERVATIVE maker assumption — fee 5bp -> 2bp/leg (Binance USDT-M VIP0 maker), half-spread STILL
PAID in full (adverse selection assumed to eat the earned spread — conservative), impact
unchanged: round-trip 13bp -> 7bp. The official record STAYS at taker until paper measures real
maker fill rates (the honest gate); the scenario quantifies the deployment upside contingent on
fills. No adoption decision — a documented scenario.

## E3 — carry as a separate book-level sleeve (the one untested portfolio idea)

Combination: net_combined = 0.85 * net_momentum_ensemble + 0.15 * net_carry_book, both books as
already validated (momentum = the deployed band ensemble; carry = carry-7d/quintile/volscaled/
weekly, the recorded income-sleeve config), each with its own friction (book-level combination —
NOT a signal blend, which was already tested and rejected). w_carry = 0.15 FIXED ex ante (single
point, never searched): the anchor is the maxDD budget — carry's recorded ~70% structural DD
diluted at 0.15 contributes <= ~10.5pp, keeping the combined book inside the 25% maxDD anchor.
Report: realized momentum-carry weekly correlation, the SIGNAL.md §6.1 additivity-identity
PREDICTED combined Sharpe vs the measured one, CPCV at the official config, maxDD, funding split.
ADOPTION RULE (fixed ex ante): adopt the 85/15 book iff (i) CPCV(8,2, embargo=3) mean >=
momentum-only + 0.05 AND frac_pos >= momentum-only - 0.02; (ii) improvement sign holds at (6,2)
and (10,3); (iii) combined maxDD <= 25%. Otherwise record as a negative finding and keep the
momentum-only book.

## Protocol
Main-thread, no agents; every stage checkpointed to _ckpt/extend/ (results.json + PROGRESS.md);
prod docs/src updated ONLY after all three complete; research artifacts stay in research/.

## RESULTS (2026-07-03; raw detail _ckpt/extend/results.json)

### E1 — the extension pays off on every axis (334 weeks, 2020-01 → 2026-07; +29% track)

Klines backfilled to the Binance UM dump archive's origin (2020-01; 770 new month-files, 24.4M
rows, 0 errors); funding archive begins 2020-06 (earlier weeks use the documented 0-accrual
convention — momentum needs only closes). Every strategy choice was frozen BEFORE this data was
downloaded, so the extension is backward-out-of-sample evidence, not fitting. On the extended
panel (universe 2020: 34 names — above every breadth floor):

- **DSR PASSES: 0.9505 ≥ 0.95, PSR 0.9995, MinTRL 331.1 < n_obs 333** — the repo's #1 honesty
  flag (track length) clears, MARGINALLY (by 2 weeks; every live week widens it). Honest framing:
  a hair above the gate, not comfortably above it.
- **PBO collapses**: (8,2) 0.393 → **0.107**, (10,3) 0.425 → **0.075** — both under the 0.2
  clean-pass line; (6,2) 0.400 (partition sensitivity remains → the BAND rule stays mandatory).
- CPCV (official embargo-3, L=14 selected over the 7-family): mean **+1.134**, frac_pos **1.00**,
  naive 1.070. Per-L naive Sharpe (334wk): 10: 1.02, 12: 1.48, 14: 1.07, 16: 1.08, 18: 0.92,
  21: 1.06, 28: 0.84 — every member improved; 2020 (COVID crash + recovery) was GOOD for the book.
- **Phase table softens exactly as more data should adjudicate** (deployable Friday band ensemble
  per phase): [Mon 0.52, Tue 0.53, Wed 0.45, Thu 0.82, **Fri 1.29**, Sat 0.75, Sun 0.51] — every
  phase now POSITIVE (5yr worst was −0.03); the Friday premium persists and still looks
  structural, but "other phases are dead" was substantially a small-sample artifact. Neither R1
  flag condition fired (Friday still max; CPCV well above 0.5).
- The official record MOVES to the extended panel (prod docs restated); the 5yr numbers remain
  the historical record in git.

### E2 — maker-first execution: tactic implemented, upside quantified, record unchanged

(a) `execution.tactic_for` (test-pinned): POST_ONLY_MAKER on every slice except the last;
final slice crosses LIMIT_IOC; URGENT/IMMEDIATE escalate per the core ladder. (b) Conservative
scenario (fees 5→2bp/leg, half-spread STILL paid): cost 3.80 → 2.05 %/yr, naive Sharpe 1.124 →
1.206, CPCV(e=3) 1.336 → 1.423 (5yr grid, same-grid comparison). ~+1.75%/yr of recoverable cost,
CONTINGENT on maker fill rates — the official record stays at taker until paper measures them.

### E3 — 85/15 carry sleeve: NEAR-MISS, NOT ADOPTED (pre-registered bar holds)

Book-level 0.85·momentum + 0.15·carry (5yr grid): ΔCPCV(8,2,e=3) = **+0.033 vs the +0.05 bar** —
fails rule (i); sign-consistent at all three partitions (+0.026/+0.040/+0.033), maxDD 21.2% ≤ 25%,
frac_pos equal. The SIGNAL.md §6.1 additivity identity predicted the combined Sharpe EXACTLY
(1.174 predicted = 1.174 measured); realized momentum–carry corr 0.32 is too high for a 15%
sleeve to clear a +0.05 materiality bar. Momentum-only book retained; negative finding recorded.
(E2/E3 ran on the pre-extension 5yr grid — both are same-grid relative comparisons; their
conclusions are grid-internal and unaffected by the extension.)
