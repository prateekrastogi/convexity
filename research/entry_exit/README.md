# entry_exit — formal entry/exit candidate models + selection harness

Provenance: built 2026-07-04 in the `intraday` repo's Phase-C (pre-registered, independently
audited, synthetic-tested), moved here the same day when `intraday` was reset to its
pre-build state. **Next target: commodity & metal futures** on convexity's stitched 1-min
lake (XAU/XAG/CL/BZ/NATGAS/COPPER/XPT/XPD perps).

## Modules

| Module | What it is |
|---|---|
| `_core.py` | shared types (TradeRecord, DDOF, PERIODS_PER_YEAR) + `metrics()` (per-trade stats incl. drop-top-N tells) |
| `base.py` | the candidate interface: causal EntryModel/ExitModel protocols, ScanCursor (F_{t-1} purity), TrainFold freeze, CandidateResult (+parsimony k) |
| `changepoint.py` | BOCPD (Adams–MacKay) entry — run-length posterior, t+1 execution offset |
| `sequential.py` | Wald SPRT / window-limited GLR drift-sign entry (ARL-controlled boundaries) |
| `hmm.py` | 2-state Gaussian HMM entry on the FILTERED posterior (smoothed = lookahead, pinned) |
| `jumps.py` | Lee–Mykland jump test, directional jump-continuation entry (trailing window) |
| `stopping.py` | optimal-stopping free-boundary exit (momentum/regime-switching latent) |
| `ou_exit.py` | OU / Leung–Li mean-reversion optimal exit (MLE fit + free-boundary b*) |
| `hazard.py` | discrete-time survival/hazard reversal exit (dynamic hold = hazard threshold) |
| `har.py` | HAR-RV (Corsi) one-step-ahead vol input (scaling / adaptive threshold / covariate) |
| `validation.py` | splits+embargo, bootstrap CIs, PSR/DSR/MinTRL (per-observation Sharpe, BLdP) |
| `selection.py` | the adoption conjunction: OOS-vs-incumbent + DSR/PBO + drop-top-N + paired-CI + cross-variant |

Purity contract everywhere: causal FILTERED / one-step-ahead F_{t-1}-measurable estimates
only; params fit on train and FROZEN for OOS; every model ships a known-generating-process
recovery test and a no-lookahead pinning test (`tests/`).

## Run the tests

```
cd /home/speculation/convexity
.venv/bin/python -m pytest research/entry_exit/tests -q
```

## Deliberately NOT carried from intraday

The NSE-options pillars (paper/ops/execution/universe/portfolio/regime/panel), the options
lake runner (`select/run.py`, BLOCKED-ON-LAKE), and the options specs. A futures study needs
a NEW thin runner over `data/stitched/` and — per PREREGISTRATION.md — a NEW dated
pre-registration (fresh candidate ledger + adoption threshold) BEFORE any empirical run.
