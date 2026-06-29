"""ParityReplay — batch/live parity harness (TREND §11 online/replay, Inv I).

Drives the engine once via ``run_batch`` and once bar-by-bar via ``update``, then asserts the two
decision streams agree within deterministic numerical tolerance (Inv I). This is the parity gate the
release uses; it is also imported by the parity test.
"""

from __future__ import annotations

from convexity.core.types import Bars, InstrumentSpec, Position
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.engine import TrendLifeEngine
from convexity.trend.types import ActionDecision


def _key(d: ActionDecision) -> tuple:
    """The decision identity used for parity comparison (action + cause + level + size)."""
    return (
        d.idx,
        d.action,
        d.stop_cause,
        round(d.exit_state.trail_level, 9),
        round(d.size_fraction, 9),
        d.action_idx,
    )


class ParityReplay:
    """Batch vs one-bar online parity (§11, Inv I)."""

    def check(
        self, bars: Bars, position: Position, spec: InstrumentSpec, cfg: ResolvedConfig
    ) -> bool:
        """Return True iff ``run_batch`` and sequential ``update`` produce identical decisions."""
        batch_engine = TrendLifeEngine(cfg)
        batch = batch_engine.run_batch(bars, position, spec, cfg)

        live_engine = TrendLifeEngine(cfg)
        live: list[ActionDecision] = []
        for idx in range(position.entry_idx, len(bars)):
            d = live_engine.update(bars, position, spec, cfg, idx)
            if d is None:
                break
            live.append(d)
            if d.action == "exit":
                break

        if len(batch) != len(live):
            return False
        return all(_key(a) == _key(b) for a, b in zip(batch, live))
