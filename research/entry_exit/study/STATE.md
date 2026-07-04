# COMMODITY STUDY — STATE (the /loop resume brain)

Loop: cron `e63947b0`, every 5h at :07, session-only. Every firing reads THIS file first and
continues from NEXT ACTION; a running VM background job makes the firing a heartbeat (check
progress via its log, update this file, never duplicate). Checkpoint + commit (`commodity` branch)
after every stage. All work under `research/entry_exit/study/`; never touch `src/convexity` or the
live paper system.

## Stages

| # | Stage | Status |
|---|---|---|
| 0 | Pre-registration frozen + committed (COMMODITY_PREREG.md) | ✅ this commit |
| 1 | Data prep: 1-min → per-symbol DAILY decision frames (`prep_frames.py` → `frames/{SYM}.parquet`, gitignored) | ✅ launched this commit (bg job, ~min) — verify next firing |
| 2 | Baseline harness (`run_baseline.py`): 6 declared cells × 7 symbols, full-sample + CPCV(8,2,embargo 10) net-of-cost → `results_baseline.json` | ⬜ next |
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

Verify stage-1 frames completed (`study/prep_frames.log` should end with "ALL FRAMES DONE";
`frames/*.parquet` = 7 files). Then author + launch stage 2 (`run_baseline.py`): for each symbol ×
{W∈{30,120}} × {s∈{0.25,0.5,1.0}}: simulate per §2 of COMMODITY_PREREG.md (day-close entry, ATR20
stop with 1-min touch + gap-through-at-open fills, 5-day hold, non-overlap), net 12bp RT (+6/24
scenarios); emit per-trade lists + daily net series; full-sample metrics + CPCV(8 groups, k=2,
embargo 10d) using convexity.system.cpcv (adapt over the daily grid); write results_baseline.json;
commit. PRIMARY cell = W=120,s=0.5 (pre-named). Then stage 3.
