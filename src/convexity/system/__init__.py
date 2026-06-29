"""Cross-pillar orchestrator + integrated validation (pure-Python pillar).

Spec: SYSTEM. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
The orchestrator (``compose/``) composes the four pillar adapters (``modules/``) into one target
book per bar; the validation layer (``validate/``) proves the composed chain has an edge net of
cost and net of the joint multiple-testing tax (the deflated-Sharpe centerpiece). See
docs/ARCHITECTURE.md §4/§5 and docs/specs/SYSTEM.md.
"""
