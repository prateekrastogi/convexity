"""Integrated validation — PART OF THE MODEL, never a report bolted on (SYSTEM §6).

Validates the COMPOSED pipeline (selection → breakout → leverage → trend → execution-stub) as ONE
object, under the strictest causal discipline in the repo. Nothing here is imported by ``compose/``,
``modules/``, or ``nautilus/`` (Invariant FW): the system triple-barrier labels use FUTURE bars and
are structurally unreachable from live composition.

  * labels.py      — system triple-barrier labels; vertical = H_sys (Invariants FW, HC)
  * walkforward.py — purged walk-forward; union embargo keyed to H_sys (Invariants SC, HC)
  * cpcv.py        — sampled combinatorial purged CV (the path-Sharpe distribution)
  * deflated.py    — PSR/DSR over EFFECTIVE N_sys + the singularity guards — THE CENTERPIECE
  * robustness.py  — drop-top-N + block-bootstrap CIs + regime/fold attribution
  * attribution.py — Shapley (16 coalitions) OR sequential (5 knock-ins) (Invariant ATTR)
  * baselines.py   — buy&hold, random-matched, ablations, 16 coalition off-semantics (Invariant BL)
  * trials.py      — the PERSISTENT trial ledger → effective N_sys (Invariants SG, RP)
  * promotion.py   — re-promotion cadence + held-out PAIRED block-bootstrap forward gate (Inv RP)
  * budget.py      — overnight per-stage cost model + K_max/T_night cap (Invariant CB-BUDGET)
  * cache.py       — the EvidenceCache (Invariant CA)
  * reports.py     — SystemReport assembly (headline = DSR | n_obs vs MinTRL)
"""
