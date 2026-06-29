"""Glue to NautilusTrader — no engine reimplementation (SYSTEM §10).

Import-isolated: this is the ONLY SYSTEM sub-package that may touch Nautilus, and the rest of the
pillar (compose/, validate/, modules/) imports nothing from here, so the pure-Python policy logic
stays NautilusTrader-free and unit-testable without the engine.

  * adapter.py   — read book/account/position into ``BookState``; emit ``SystemDecision`` deltas
  * exec_stub.py — the CONSERVATIVE cost+fill UPPER bound used in validation only (Invariant XC)
"""
