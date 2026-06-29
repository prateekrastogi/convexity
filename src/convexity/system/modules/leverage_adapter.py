"""LeverageAdapter — wraps ``LeverageEngine.size_position`` (SYSTEM §2/§3, Invariant P1).

SYSTEM's only multi-input combination over ceilings is delegated to the LEVERAGE engine: the
adapter hands it a :class:`SignalInput` + the SYSTEM-owned :class:`BookState` and reads back a
:class:`LeverageDecision` (the ``min()`` of seven ceilings × governor; ``binding_ceiling`` names
which bound). No sizing math lives here (Invariant P1); the contraction map in ``compose/iterate``
calls this adapter once per position per iteration to get each ceiling under the *current* book.
"""

from __future__ import annotations

from convexity.core.protocols import LeverageEngine
from convexity.core.types import LeverageDecision, SignalInput
from convexity.system.types import BookState


class LeverageAdapter:
    """Wrap a concrete (or mock) :class:`~convexity.core.protocols.LeverageEngine`.

    ``size`` is a pure pass-through to ``size_position(signal, risk, cfg, book)``; the SYSTEM
    BookState is the shared book (incumbents + reserved heat + candidate weights + existing margin)
    LEVERAGE consumes as ``object`` at its seam (ARCHITECTURE §5.9). The returned ``LeverageDecision``
    carries ``leverage``/``size``/``rounded_size``, the ``binding_ceiling``, and the governor
    multiplier echo — all SYSTEM reads to enforce the ceiling-collapse precedence (Section 2 step B).
    """

    def __init__(self, engine: LeverageEngine) -> None:
        self._engine = engine

    def size(self, signal: SignalInput, risk: object, cfg: object, book: BookState) -> LeverageDecision:
        """Delegate sizing to the LEVERAGE engine under the current book (no reimplementation)."""
        return self._engine.size_position(signal, risk, cfg, book)
