"""Thin pillar adapters (SYSTEM §1/§10, Invariant P1).

Each module here wraps one already-built pillar behind its public contract and reshapes its output
into a SYSTEM boundary object. NO sizing, exit, entry, or selection math lives here — only the
import + the boundary reshape (Invariant P1: "compose, never reimplement"). Every adapter is
MOCKABLE: the orchestrator is unit-tested against scripted adapter outputs (the §15 composition
gate), so each adapter exposes a small protocol the tests can satisfy without the real pillar.

  * selection.py        — maps the Carver Ch.25 ``DynOptResult`` → ``ProposedPortfolio`` (ARCH §5.10)
  * breakout_adapter.py — wraps ``BreakoutModule.propose`` → ``EntrySignal | None``
  * leverage_adapter.py — wraps ``LeverageEngine.size_position`` → ``LeverageDecision``
  * trend_adapter.py    — wraps ``TrendEngine.update``; injects ``tau_cap = H_sys`` (Invariant HC)
"""
