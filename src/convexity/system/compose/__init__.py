"""The composition algebra (SYSTEM §2-§4).

Deterministic precedence + a bounded iterate-to-consistency between selection and leverage. No
discretion, no learned combiner, no multiplicative score blob (P1/P2). Everything here is an
algebra of ``min()``, set operations, and a damped projection + monotone greedy fill — all
auditable, all monotone (Invariant CB).

  * cadence.py     — decision-bar / deadband / event-recompose logic (Invariant CAD)
  * incumbent.py   — the hold-incumbent firewall + rotation test (Section 4, Invariant HI)
  * budget.py      — residual-budget priority by ceiling-slack/MCTR-efficiency (Invariant A)
  * iterate.py     — the damped fixed point Φ + monotone greedy lot-fill (Section 3, Invariant IT)
  * precedence.py  — the deterministic A..G pipeline (Section 2)
  * orchestrator.py — ``SystemOrchestrator.decide``: one public method (Invariant P1/CB)
"""
