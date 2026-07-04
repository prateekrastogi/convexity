# COMMODITY STUDY — STATE (the /loop resume brain)

Loop: cron `e63947b0`, every 5h at :07, session-only. Every firing reads THIS file first and
continues from NEXT ACTION; a running VM background job makes the firing a heartbeat (check
progress via its log, update this file, never duplicate). Checkpoint + commit (`commodity` branch)
after every stage. All work under `research/entry_exit/study/`; never touch `src/convexity` or the
live paper system.

## Stages

| # | Stage | Status |
|---|---|---|
| 0 | Pre-registration frozen + committed (COMMODITY_PREREG.md) | ✅ `efecd2b` |
| 1 | Data prep: 1-min → per-symbol DAILY decision frames (`prep_frames.py` → `frames/{SYM}.parquet`, gitignored) | ✅ 7 frames, ~1330 valid days each |
| 2 | Baseline harness (`run_baseline.py`): 6 declared cells × 7 symbols, full-sample + CPCV(8,2,embargo 10) net-of-cost → `results_baseline.json` | ✅ `54c06d1` — **BASELINE DEAD as prior predicted**: pooled gross ≈ +2bp/trade ≈ 0; net@12bp −9.8bp (primary cell); best symbols XAU +5.7bp/XAG +10.6bp net (annSh 0.28, CPCV frac+ 0.68-0.75) but drop-top-3 NEGATIVE everywhere; CL/COPPER/XPT/XPD negative outright. No baseline cell alive. Re-confirms convexity EDGE gross≈0 verdict |
| 3 | Candidate harness (`run_candidates.py`): the frozen 32-combo ledger, train-fold fit + FROZEN params, same CPCV/cost harness → `results_candidates.json` (checkpoint per symbol×entry) | ⬜ |
| 4 | Selection + gates: PBO/DSR at ledger N_eff, drop-top-N, paired bootstrap vs baseline, parsimony penalty (§6 of the prereg) → `results_selection.json` | ⬜ |
| 5 | REPORT.md: baseline vs optimal, per-commodity ranking, Kelly/DD-vs-leverage, win/EV/Sharpe/DD tables, multiplicity verdicts; report to user in chat; cleanup per research conventions | ⬜ |
| 6 | DONE → cancel loop (CronList → CronDelete e63947b0) | ⬜ |

## Invariants (binding)

- Prereg is FROZEN: only §3-ledger configs may run/be claimed; N_eff per §3; gates per §6.
- Net-of-cost always (12bp RT primary; 6/24bp + ±5bp/day funding scenarios reported).
- Purity: train-fold fit, frozen for test; filtered/causal only; day-close decision convention
  identical for baseline and candidates (matched comparisons).
- Honesty: convexity EDGE prior says directional breakout is dead — expect a weak/dead baseline;
  report negatives plainly; never fabricate; every number from actual runs on the stitched lake.
- Compute = VM background python (nohup, `.venv/bin/python`, log under study/), scripts + small
  JSON committed; parquet/frames gitignored.

## NEXT ACTION

**Stage 3 — the candidate harness (`run_candidates.py`), the study's real question** (baseline is
dead; do the formal models beat it?). Per COMMODITY_PREREG.md §3, frozen ledger: entries E1 BOCPD
(hazard 1/{50,100,200} × thresh {0.5,0.7}), E2 SPRT/GLR ((α,β)∈{(.05,.2),(.01,.1)}, m∈{20,60}),
E3 HMM filtered P>{0.6,0.75}, E4 LM-jumps (K∈{16,32}); exits X0 fixed-5d, X1 optimal-stopping,
X2 OU/Leung-Li, X3 hazard (thresh {0.5,0.7}); HAR toggle. Implementation: import the entry_exit
package modules (sys.path research/) — the models operate on the DAILY log-return series from
frames/{SYM}.parquet; for each CPCV split (cpcv_daily.splits, same 8/2/10 config): fit each
entry/exit model's params on TRAIN days (grid per ledger, pick by train objective, FREEZE), emit
entry signals on TEST days, simulate with the SAME engine as run_baseline.simulate (day-close
entry, ATR20 s=0.5 stop floor, 1-min fills — refactor simulate() to accept an external entry/exit
signal stream), net 12bp RT; accumulate per-combo OOS daily series + trades; checkpoint JSON per
symbol (resume-safe). Output results_candidates.json. THEN stage 4 (gates: PBO over the family via
convexity compute_pbo, DSR at ledger N_eff via entry_exit.validation.deflated_sharpe on
per-observation Sharpe, drop-top-3, paired bootstrap vs the baseline primary cell, parsimony
penalty) → stage 5 REPORT.md + chat report (incl. Kelly/leverage + best-commodity ranking) →
stage 6 cancel loop `e63947b0`. Compute note: BOCPD/HMM per split×symbol×grid on ~1000-day series
— budget ~1-3s per fit, ~28 splits × 7 syms × grid ≈ manageable; run nohup background, poll.
