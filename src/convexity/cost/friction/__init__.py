"""COST pillar validation layer for the xsec-momentum book (docs/specs/COST.md Section 3).

The accounting CORE (``research/xsec/accounting.py``: ``settle_week`` / ``round_trip_friction`` /
``WeekResult``) is the parity-critical settlement truth and lives OUTSIDE this package — nothing
here re-derives its turnover or settlement arithmetic (COST.md Section 1, Invariant N). This
package holds the spec's NEW validation pieces built AROUND that core:

  * ``friction_stress`` — the 1.5x-friction promotion-gate scenario (COST.md Section 8, SYSTEM
    Gate 6's dependency) and the funding-ablation ``net_sharpe_ex_funding`` (COST.md Section 6.4).
  * ``validate.corwin_schultz_reject`` — the documented Corwin-Schultz NEGATIVE result kept as a
    standing regression guard (COST.md Section 5.4.3, Invariant CS-REJECT).

Import submodules directly (``from convexity.cost.friction.friction_stress import ...``); this
``__init__`` deliberately re-exports nothing so partially-built sibling modules never break the
package import.
"""
