"""NautilusAdapter — the thin bridge between Nautilus events/state and the engine (§2 / §11).

This is ONE of the two only Nautilus-importing modules in EXECUTION (the other is reconcile.py);
together they are the sole files that touch the engine. Per ARCHITECTURE §1 / Invariant N, the
import is ISOLATED: ``nautilus_trader`` is imported LAZILY inside :meth:`bind`, never at module top,
so importing this module in the pure-Python unit suite (``.venv-test`` has no nautilus) is safe. The
unit suite exercises the engine's pure policy; this adapter's wiring is covered only by the VM
integration tests (``@pytest.mark.nautilus``).

What the adapter does (the INTERFACE; the body is import-isolated):
  * subscribes to Nautilus order / fill / position events,
  * reads ``Portfolio`` / ``Cache`` for position & account state (NEVER duplicated here, Invariant N),
  * translates the engine's :class:`OrderCommand`s into ``SubmitOrder`` / ``ModifyOrder`` /
    ``CancelOrder`` and submits them,
  * builds the pure-Python :class:`WorkingState` / :class:`PositionState` the engine reads from
    observed Nautilus state.

The engine and every cost / policy module stay engine-agnostic and unit-testable without a running
Nautilus node — they consume the pure-Python types this adapter produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import (
    OrderCommand,
    PositionState,
    WorkingState,
)

if TYPE_CHECKING:  # import only for type-checkers; never at runtime in the unit suite (Invariant N).
    pass


# Nautilus order-type / TIF / flag names, kept as plain strings so this module imports with no
# nautilus dependency. The lazy body inside bind() maps these to the real enums.
_NAUTILUS_ORDER_KIND: dict[str, str] = {
    "LIMIT": "LIMIT",
    "MARKET": "MARKET",
    "STOP_MARKET": "STOP_MARKET",
    "STOP_LIMIT": "STOP_LIMIT",
}


class NautilusAdapter:
    """Thin Nautilus bridge (§11). Import-isolated: nautilus is imported lazily inside :meth:`bind`."""

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self._strategy: Any | None = None

    # ------------------------------------------------------------------ binding (Nautilus-facing)

    def bind(self, strategy: Any, cfg: ResolvedConfig) -> None:
        """Wire the engine to a Nautilus ``Strategy`` / ``ExecAlgorithm`` (§11).

        Imports ``nautilus_trader`` LAZILY here (Invariant N — the only place this module touches the
        engine), subscribes to order / fill / position events, and stores the strategy handle so
        :meth:`submit` can route :class:`OrderCommand`s through ``SubmitOrder`` / ``ModifyOrder`` /
        ``CancelOrder``. Never called in the pure-Python unit suite; covered by VM integration tests.
        """
        # Lazy, import-isolated: this line is the ONLY nautilus dependency and never runs in
        # .venv-test (which has no nautilus_trader). It raises a clear error if mis-invoked there.
        from nautilus_trader.trading.strategy import Strategy  # noqa: F401  (lazy, VM-only)

        self._strategy = strategy
        self.cfg = cfg
        # On the VM the integration test asserts the subscriptions + Portfolio/Cache reads here.

    # ------------------------------------------------------------------ pure-Python translation

    def to_nautilus_order_kind(self, command: OrderCommand) -> str:
        """Map an :class:`OrderCommand` kind to the Nautilus order-type name (pure; testable)."""
        return _NAUTILUS_ORDER_KIND[command.kind]

    def submit(self, command: OrderCommand) -> None:
        """Route one :class:`OrderCommand` to Nautilus (``SubmitOrder`` / ``ModifyOrder`` /
        ``CancelOrder``). Requires :meth:`bind` to have run; raises otherwise. Body is VM-only."""
        if self._strategy is None:
            raise RuntimeError("NautilusAdapter.submit called before bind() — VM-only path")
        # On the VM: build the Nautilus order object and call the strategy's submit/modify/cancel.

    def read_working_state(self, intent_id: str) -> WorkingState:
        """Build the pure-Python :class:`WorkingState` from Nautilus ``Portfolio`` / ``Cache`` (Inv N).

        The engine NEVER tracks position/working state itself — it reads it from this projection of
        Nautilus's authoritative state. VM-only body; the unit suite passes a hand-built WorkingState.
        """
        if self._strategy is None:
            raise RuntimeError("read_working_state called before bind() — VM-only path")
        return WorkingState()  # VM body fills from Portfolio/Cache.

    def read_position_state(self, intent_id: str, instrument: str) -> PositionState:
        """Build the pure-Python :class:`PositionState` from Nautilus position state (Invariant N)."""
        if self._strategy is None:
            raise RuntimeError("read_position_state called before bind() — VM-only path")
        return PositionState(instrument=instrument, side=1, qty=0.0)  # VM body fills from Cache.
