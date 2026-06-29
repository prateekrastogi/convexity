"""BreakoutAdapter — wraps ``BreakoutModule.propose`` (SYSTEM §2, Invariant P1).

Thin pass-through: for each selection-eligible, not-open instrument, ask the breakout module for an
``EntrySignal | None``. SYSTEM reads ONLY ``direction``/``status=='entry'``/``urgency`` and the
price levels — NEVER ``quality_score`` magnitude (Invariant CB/K: the composed decision depends on
the boolean qualification + side, never the score). No detection math lives here (Invariant P1).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from convexity.core.protocols import BreakoutModule
from convexity.core.types import EntrySignal, InstrumentSpec


class BreakoutAdapter:
    """Wrap a concrete (or mock) :class:`~convexity.core.protocols.BreakoutModule`.

    ``propose_for`` returns the qualified entry candidates for the given instrument specs, keyed by
    symbol, dropping ``None`` (no-trade) results. SYSTEM uses ``status == 'entry'`` + ``direction``
    only; the ``quality_score`` is carried for audit but is forbidden as a composition input.
    """

    def __init__(self, module: BreakoutModule) -> None:
        self._module = module

    def propose_for(self, specs: Sequence[InstrumentSpec]) -> Mapping[str, EntrySignal]:
        """Qualified breakout entries keyed by symbol (no-trade ⇒ omitted)."""
        out: dict[str, EntrySignal] = {}
        for spec in specs:
            sig = self._module.propose(spec)
            if sig is not None and sig.status == "entry" and sig.direction != 0:
                out[spec.symbol] = sig
        return out
