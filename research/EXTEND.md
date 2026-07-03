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
