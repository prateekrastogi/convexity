"""NautilusTrader glue — read book/account/position, emit SystemDecision deltas (SYSTEM §10).

SYSTEM reads the portfolio/position/account snapshot from Nautilus into the canonical
:class:`BookState` (the shared reference, ARCHITECTURE §5.9) and emits the orchestrator's
:class:`SystemDecision` deltas for the engine to execute — it NEVER places raw orders itself
(SYSTEM §2 step G). Batch replay and one-bar streaming produce tolerance-equal SystemDecisions
because the composition is a pure function of (BookState, module outputs) and the engine owns parity
(Invariant PAR).

This module is import-isolated (the rest of SYSTEM imports nothing from ``nautilus/``), so the
pure-Python policy stays engine-free. The reads below are framed as pure mappers from a generic
snapshot object so they are testable without the live engine; a live deployment passes the real
Nautilus ``Portfolio``/``Cache`` objects, which expose the same attributes.
"""

from __future__ import annotations

import numpy as np

from convexity.system.types import BookState, OpenPosition, SystemDecision


def read_book_state(snapshot: object, *, cov: np.ndarray, governor_multiplier: float) -> BookState:
    """Map a Nautilus portfolio/account snapshot into the canonical :class:`BookState` (ARCH §5.9).

    ``snapshot`` exposes ``equity``, ``peak_equity``, ``free_margin`` and a ``positions`` mapping of
    instrument → (side, units, entry_price, held_risk_weight, trend_state). ``cov`` is the
    LEVERAGE RISK covariance; ``governor_multiplier`` is the LEVERAGE drawdown-governor m ∈ [0,1].
    SYSTEM does not recompute equity/peak/governor — it reads the LEVERAGE-owned reference (§5.12).
    """
    positions = {
        inst: OpenPosition(
            instrument=inst,
            side=p["side"],
            units=int(p["units"]),
            entry_price=float(p["entry_price"]),
            held_risk_weight=float(p["held_risk_weight"]),
            trend_state=str(p.get("trend_state", "alive")),
        )
        for inst, p in getattr(snapshot, "positions", {}).items()
    }
    return BookState(
        equity=float(getattr(snapshot, "equity")),
        peak_equity=float(getattr(snapshot, "peak_equity")),
        free_margin=float(getattr(snapshot, "free_margin")),
        positions=positions,
        cov=np.asarray(cov, dtype=float),
        governor_multiplier=float(governor_multiplier),
    )


def emit_deltas(decision: SystemDecision) -> tuple[dict, ...]:
    """Lower a :class:`SystemDecision` into engine-agnostic order DELTAS (never raw orders; §2 G).

    Each delta is a plain dict the Nautilus adapter turns into the engine's order command; SYSTEM
    only describes the target book change (open/close/reduce/hold/flat_all) + its reason chain.
    """
    return tuple(
        {
            "instrument": a.instrument,
            "kind": a.kind,
            "target_units": a.target_units,
            "side": a.side,
            "driven_by": a.driven_by,
            "binding": a.binding,
            "reason_chain": a.reason_chain,
        }
        for a in decision.actions
        if a.kind != "hold"  # holds are no-ops at the engine
    )
