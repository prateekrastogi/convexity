"""LOTS↔UNITS round-trip is exact for a NON-unit contract multiplier (ARCHITECTURE §5.3).

The single seam where SYSTEM's integer LOT count meets the venue's raw UNIT denomination is
:func:`convexity.strategy.convexity_strategy._lots_to_units` (lots → units, multiply by ``lot_size``)
and its inverse inside :func:`convexity.strategy.projection.book_state_from_snapshot` (units → lots,
divide by ``lot_size`` and round). Every other fixture in the suite uses ``lot_size == 1.0`` where
lots and units COINCIDE, so a silently-dropped multiplier would pass them all. These tests feed a
genuinely non-unit multiplier (0.001-BTC perp, 25-unit lot, fractional) and assert the two are exact
inverses — a 0.001 multiplier can never turn N lots into N units.

Pure-Python (no nautilus): both functions are NautilusTrader-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.core.types import InstrumentSpec
from convexity.strategy.convexity_strategy import _lots_to_units
from convexity.strategy.projection import PositionSnapshot, book_state_from_snapshot


def _spec(symbol: str, lot_size: float) -> InstrumentSpec:
    """A minimal AMBER spec carrying ONLY the contract multiplier under test."""
    return InstrumentSpec(
        symbol=symbol, asset_class="crypto_perp", venue="BINANCE", timezone="UTC",
        tick_size=0.1, lot_size=lot_size,
    )


# A 0.001-BTC perp, a 25-unit-per-lot contract, a fractional multiplier, and the degenerate 1.0.
NON_UNIT_LOT_SIZES = [0.001, 25.0, 0.5, 2.0, 100.0]


@pytest.mark.parametrize("mult", NON_UNIT_LOT_SIZES)
@pytest.mark.parametrize("lots", [1, 3, 7, 40, 137])
def test_lots_units_round_trip_is_exact_for_non_unit_multiplier(mult: float, lots: int) -> None:
    """lots → units → lots recovers the SAME integer lot count for a non-unit ``lot_size``."""
    spec = _spec("BTCUSDT", mult)

    # Forward: SYSTEM's integer LOT count → raw venue UNITS.
    units = _lots_to_units(lots, spec)
    assert units == pytest.approx(lots * mult), "forward conversion must scale by the multiplier"
    # For a non-unit multiplier the units must NOT equal the lot count (the bug these fixtures miss).
    if mult != 1.0:
        assert units != float(lots), "non-unit multiplier must make units != lots"

    # Inverse: the strategy reads raw UNITS off Nautilus; book_state_from_snapshot divides by the
    # multiplier to recover the lot count SYSTEM reasons in. Round-trip must be the identity.
    snap = PositionSnapshot(
        instrument="BTCUSDT", side=1, qty=units, entry_price=30_000.0,
        entry_idx=5, invalidation=29_800.0, generation=0,
    )
    book = book_state_from_snapshot(
        equity=1000.0, peak_equity=1000.0, free_margin=800.0,
        snapshots=[snap], specs={"BTCUSDT": spec},
    )
    recovered_lots = book.positions["BTCUSDT"].units
    assert recovered_lots == lots, (
        f"units→lots inverse must recover {lots} lots, got {recovered_lots} "
        f"(mult={mult}, units={units})"
    )


def test_unit_multiplier_is_the_degenerate_coincidence() -> None:
    """With ``lot_size == 1.0`` lots == units — the coincidence every other fixture relies on."""
    spec = _spec("UNITPERP", 1.0)
    assert _lots_to_units(13, spec) == 13.0
    snap = PositionSnapshot(
        instrument="UNITPERP", side=-1, qty=13.0, entry_price=100.0,
        entry_idx=1, invalidation=110.0, generation=0,
    )
    book = book_state_from_snapshot(
        equity=1000.0, peak_equity=1000.0, free_margin=800.0,
        snapshots=[snap], specs={"UNITPERP": spec}, cov=np.eye(1),
    )
    assert book.positions["UNITPERP"].units == 13


def test_lots_to_units_rejects_a_misconfigured_multiplier() -> None:
    """A non-positive / NaN multiplier must blow up at the seam, never silently scale (the guard)."""
    with pytest.raises(AssertionError):
        _lots_to_units(5, _spec("BAD", 0.0))
    with pytest.raises(AssertionError):
        _lots_to_units(5, _spec("BAD", -1.0))
    with pytest.raises(AssertionError):
        _lots_to_units(5, _spec("BAD", float("nan")))
