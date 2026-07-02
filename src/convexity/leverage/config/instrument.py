"""Per-instrument AMBER metadata LEVERAGE reads (LEVERAGE.md §9 layout, §3 AMBER bucket).

``InstrumentSpec`` names the AMBER (supplied market-metadata) fields the capacity ceiling reads per
instrument: ``shortable`` (a name that cannot be shorted contributes ``ADV_i = 0`` to the short
leg, §5.1), and the ``lot``/``multiplier`` contract-size metadata (§3 AMBER). These are SUPPLIED
market facts, never RED constants and never fitted (Invariant C-cfg): they enter the sizing chain
only through the per-call ``CapacityInput`` list, never through ``resolve_config`` (§3).

``CapacityInput`` (``types.py``) already carries ``shortable`` inline for the capacity call — this
spec-named type is the fuller AMBER record a caller (UNIVERSE/PORTFOLIO) hydrates from venue
metadata and from which the ``CapacityInput.shortable`` flag is projected, keeping the AMBER field
set explicit and traceable to §3 rather than only implicit on the capacity input.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    """AMBER per-instrument metadata the capacity ceiling reads (LEVERAGE.md §3 AMBER, §5.1).

    ``symbol`` is the lake symbol; ``shortable`` the venue borrow/short flag binding the capacity
    ceiling per instrument (an unshortable name contributes ``ADV_i = 0`` on the short side, §5.1);
    ``lot`` the minimum tradeable lot and ``multiplier`` the contract multiplier (§3 AMBER — read,
    never chosen, by this pillar). All are SUPPLIED metadata (AMBER), so they arrive as inputs, not
    from the RED defaults table (Invariant C-cfg). ``lot``/``multiplier`` default to ``1.0`` for the
    spot/perp linear case where they are unit; a caller overrides them for genuine contract sizing.
    """

    symbol: str
    shortable: bool = True
    lot: float = 1.0
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.lot > 0.0:
            raise ValueError(f"lot {self.lot} must be > 0 (LEVERAGE §3 AMBER, {self.symbol})")
        if not self.multiplier > 0.0:
            raise ValueError(
                f"multiplier {self.multiplier} must be > 0 (LEVERAGE §3 AMBER, {self.symbol})"
            )
