"""AMBER per-instrument metadata for SYSTEM (SYSTEM §5).

Universe membership + the per-instrument market metadata SYSTEM needs to compose cross-asset:
tick/lot/multiplier, min-notional, asset class, venue/calendar. This is AMBER ONLY — drives
cross-asset behavior via metadata, never forked logic (Invariant C). No RED/BLUE field
(precedence, cadence, embargo, deflation constant) may appear here; the resolve-time firewall
(:func:`convexity.core.params.assert_no_red_in_amber`) proves it.

SYSTEM reuses the shared :class:`convexity.core.types.InstrumentSpec` as the canonical AMBER record;
``SystemUniverse`` is the thin membership wrapper the orchestrator/lot-sizer reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from convexity.core.types import InstrumentSpec

# The AMBER field names on the per-instrument record (used by the Invariant-C firewall).
INSTRUMENT_META_FIELDS = frozenset(
    {
        "symbol",
        "asset_class",
        "venue",
        "timezone",
        "tick_size",
        "lot_size",
        "price_multiplier",
        "calendar_id",
        "min_notional",
        "maker_fee_bps",
        "taker_fee_bps",
        "maker_rebate_bps",
        "funding_interval_ms",
        "funding_cap",
        "data_policy",
        "optional_sources",
    }
)


@dataclass(frozen=True)
class SystemUniverse:
    """The universe SYSTEM composes over (SYSTEM §1). ``specs`` maps symbol → AMBER InstrumentSpec;
    ``symbols`` is the ordered membership (the deterministic tie-break total order, Section 3.3)."""

    specs: Mapping[str, InstrumentSpec]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Ordered symbols — the fixed total order for deterministic tie-breaks (Section 3.3)."""
        return tuple(sorted(self.specs))

    def spec(self, symbol: str) -> InstrumentSpec:
        return self.specs[symbol]

    def lot_size(self, symbol: str) -> float:
        return self.specs[symbol].lot_size


def load_universe_meta(specs: Sequence[InstrumentSpec]) -> SystemUniverse:
    """Build a :class:`SystemUniverse` from a sequence of AMBER instrument specs."""
    return SystemUniverse(specs={s.symbol: s for s in specs})
