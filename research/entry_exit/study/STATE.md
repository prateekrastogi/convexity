# COMMODITY STUDY — STATE (the /loop resume brain)

Loop: cron `50486e00`, every 5h at :07, session-only. Every firing reads THIS file first and
continues from NEXT ACTION; a running VM background job makes the firing a heartbeat (check
progress via its log, update this file, never duplicate). Checkpoint + commit (`commodity` branch)
after every stage. All work under `research/entry_exit/study/`; never touch `src/convexity` or the
live paper system.

## Stages

| # | Stage | Status |
|---|---|---|
| 0 | Pre-registration frozen + committed (COMMODITY_PREREG.md + ADDENDUM1 + ADDENDUM2) | ✅ `efecd2b`+ |
| 1 | Data prep: 1-min → per-symbol DAILY decision frames (`prep_frames.py` → `frames/{SYM}.parquet`, gitignored) | ✅ 7 frames, ~1330 valid days each |
| 2 | Baseline (`run_baseline.py`): 6 declared cells × 7 symbols, full-sample + CPCV(8,2,10) net-of-cost → `results_baseline.json` | ✅ `54c06d1` — **BASELINE DEAD as prior predicted** (pooled gross ≈ +2bp ≈ 0; net@12bp −9.8bp primary; XAU/XAG mildly + but drop-top-3 negative everywhere) |
| 2b | ADDENDUM 1 anchor-reproduction diagnostic → `results_anchor_repro.json` | ✅ — CL negative under EVERY anchor incl. exact MCX construction & its original window (which was CL-perp's WORST); the MCX crude positive = instrument-specific cold-open microstructure, does not transfer to USD perps |
| 3 | Candidates (`run_candidates.py`): ADDENDUM2 family (12 entry-points × 5 exit-variants × HAR = ~120 configs/sym), expanding WF 5-blocks/4-scored, embargo 10d, JSONL checkpoints | 🔄 RUNNING (bg; log `run_candidates.log`; ~3-5h; resume-safe) |
| 4 | Gates: aggregate jsonl → per config×symbol OOS metrics; DSR at ledger N_eff (~168 raw) via `entry_exit.validation.deflated_sharpe` (per-observation), PBO via `convexity.system.cpcv.compute_pbo` over the family matrix, drop-top-3, paired bootstrap vs baseline PRIMARY on the same OOS days, parsimony penalty (models expose `.k`) → `results_selection.json` | ⬜ |
| 5 | REPORT.md: baseline vs optimal, per-commodity ranking, quarter/full-Kelly + DD-vs-leverage, win/EV/Sharpe/DD tables, multiplicity verdicts; report in chat; cleanup per research conventions | ⬜ |
| 6 | DONE → cancel loop (CronList → CronDelete, current job `50486e00`) | ⬜ |

## Invariants (binding)

- Prereg FROZEN (main + addenda): only ledger configs may run/be claimed; N_eff charges the full
  family (~120 candidate configs + 6 baseline + 42 addendum-1 cells).
- Net-of-cost always (12bp RT primary; 6/24bp + ±5bp/day funding scenarios reported).
- Purity: train-fold fit, FROZEN for test; filtered/causal only (F_{t-1} via ScanCursor); entry at
  decision-day close for baseline AND candidates (matched; candidates carry no intraday info).
- Honesty: report negatives plainly; adopt only what passes ALL five §6 gates; never fabricate.
- Compute = VM background python (nohup, `.venv/bin/python`); write-local-then-scp
  (COPYFILE_DISABLE=1, never heredoc-edit over ssh); scripts + small JSON/JSONL committed;
  parquet/frames gitignored.

## NEXT ACTION

Stage 3 is RUNNING in background (`pgrep -f run_candidates.py`; log
`research/entry_exit/study/run_candidates.log`; checkpoints `results_candidates.jsonl`, expected
≈ 7 syms × 4 folds × 12 entries × ≤10 exit×HAR combos ≈ up to ~3,360 records; BOCPD entries are
the slow ones). On firing:
- ALIVE → heartbeat: report `wc -l results_candidates.jsonl` progress, update this file, end.
- DEAD + incomplete → relaunch the same command (it resumes from the JSONL done-keys).
- COMPLETE ("CANDIDATES DONE" in the log) → stage 4 then 5 per the table above. For stage 4:
  pool the 4 folds per (config, symbol) by concatenating trades; OOS day-count for annualization =
  the 4 scored blocks' spans; paired comparison uses the baseline PRIMARY cell (W120_s0.5)
  restricted to the same OOS-day union; then stage 5 REPORT.md + chat report; then stage 6 cancel
  the loop (`50486e00`).

## DONE (definition)

REPORT.md committed + results reported in chat → cancel loop `50486e00`, confirm, stop. On any
firing after DONE: confirm and do nothing.
