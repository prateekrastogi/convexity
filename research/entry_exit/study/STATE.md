# STUDY STATE — research/entry_exit/study (resume brain; no loop currently armed)

## Program 1 — directional entry/exit model study: CLOSED (SUPERSEDED 2026-07-04)

Superseded by the owner's pivot to the SWING XSEC program below before stage 3 completed.
Record: prereg + ADDENDUM1/2 frozen and committed; stage 2 BASELINE = DEAD (pooled gross ≈ +2bp ≈ 0,
net@12bp −9.8bp; re-confirms the convexity EDGE gross≈0 verdict on these 7 perps); ADDENDUM 1
anchor-reproduction = CL negative under EVERY session anchor incl. the exact MCX construction and
its original window (the MCX crude positive was instrument-specific cold-open microstructure, not
transferable to USD perps); stage 3 candidate family was killed by owner at 288/~3,360 records —
partial checkpoint kept at `results_candidates.jsonl` (resume-safe via done-keys) but NOT planned
to resume. Directional constructions on this universe: measured dead twice over.

## Program 2 — SWING XSEC COMMODITY (active)

Owner-supplied factor menu (carry, basis-momentum, liquidity-provision reversal, momentum, skew,
hedging-pressure, vol-managed overlay) accepted with constructibility analysis: carry/basis-momentum
BLOCKED without external CME curve data (perps have no term structure; funding ≠ convenience
yield); the rest constructible from lake + COT. Honest targets: 0.5–0.7 lake-only, 0.85–1.15 with
curves; >1.5 = alarm.

| Step | Status |
|---|---|
| Prereg DRAFT (`SWING_PREREG_DRAFT.md` v0.1 — freeze blocked on: duka extension landed + COT verified + OWNER curve-source decision) | ✅ drafted, committed |
| Data 1: Dukascopy backward-extension (XAU 2009+, XAG 2008-06+, CL/NATGAS 2013+, COPPER 2012+; XPT/XPD have no pre-2021 duka history — probed) via `scripts/backfill_dukascopy_hist.py` | 🔄 RUNNING (bg; 18,867 code-days; ~19h politeness-paced; log `data/backfill_duka_hist.out`; resume = re-run same command) |
| Data 2: CFTC Disaggregated COT 2006-06→now via `scripts/backfill_cot.py` → `data/cot/cot_disagg.parquet` | ✅ 7 symbols × 1,046 reports; NYMEX/COMEX-anchored mapping (Feb-2022 renames handled: NAT GAS NYME, COPPER- #1; WTI ICE-Europe look-alike excluded; loud double-mapping guard) |
| Data 3: term-structure source for F1 carry + F2 basis-momentum | 🔎 FREE COMPOSITE FOUND (2026-07-04): EIA API = WTI+HenryHub contracts 1-4 daily (RCLC1-4/RNGC1-4, decades, free key) → F1/F2 complete for CL+NG; metals = rates-carry (gold/silver carry degenerate per the owner research — no curve needed); copper = LME cash-3M via Westmetall as a DECLARED venue proxy (verify feed at wiring); CHRIS deprecated/stale (ruled out). Optional exact-everything: databento $125 signup credit covers all-months daily settles 2010+ for all 7 roots (~few $ of credit; needs owner signup). NEXT: wire EIA (+Westmetall) fetchers; databento top-up if owner signs up |
| Re-stitch after duka lands: extend `scripts/stitch_tradfi_1m.py` output back to the new floors; verify seams/splits; rebuild frames | ⬜ after Data 1 |
| FREEZE prereg → build factors on the crypto-book machinery (panel/run_book/CPCV/DSR/PBO) → validate → report | ⬜ |

## Invariants (binding, unchanged)

Prereg-before-run; net-of-cost {6,12,24}bp + funding scenarios; PIT panel entry; COT read only at
`usable_from` ≤ rebalance; N_eff charges the full declared grid; deployed book = the pre-named
ensemble, never the best cell; negatives recorded in docs/EDGE.md; write-local-then-scp; VM
background compute; never touch src/convexity or the live paper system.

## NEXT ACTION

Wait for the duka hist job ("HIST DONE" in `data/backfill_duka_hist.out`; if dead earlier, re-run —
resume-safe). Then: re-stitch the 5 extended codes into `data/stitched/` (ratio-adjust at the
duka→perp seam exactly as the original stitch; validate largest non-gap moves are real events),
rebuild study frames, verify weekly-obs count (~690 at breadth ≥5 from 2013). In parallel the OWNER
decides Data 3 (curve source). Then freeze `SWING_PREREG` (dated) and begin the factor build.
