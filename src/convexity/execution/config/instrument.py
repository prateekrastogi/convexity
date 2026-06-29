"""The per-instrument AMBER field set EXECUTION reads (§4, Invariant C).

EXECUTION reuses the shared :class:`convexity.core.types.InstrumentSpec` (AMBER-only by construction
— ARCHITECTURE §3). This module just NAMES the field set so the Invariant-C test can assert no
RED/BLUE EXECUTION param (χ, a timeout, the maker floor, ``N_bucket``, …) leaks into it via
:func:`convexity.core.params.assert_no_red_in_amber`.

There is intentionally no EXECUTION-specific instrument dataclass: tick, lot, multiplier, venue, fee
schedule, min-notional, funding interval, and the session calendar all already live on the shared
AMBER ``InstrumentSpec`` (§4 AMBER list). A RED value appearing on it is the bug Invariant C catches.
"""

from __future__ import annotations

from convexity.core.types import InstrumentSpec

# The AMBER field names the Invariant-C check scans (every field of the shared spec is AMBER by
# construction; the test asserts no EXECUTION RED/BLUE name appears among them).
INSTRUMENT_SPEC_FIELDS: frozenset[str] = frozenset(InstrumentSpec.__dataclass_fields__)

__all__ = ["InstrumentSpec", "INSTRUMENT_SPEC_FIELDS"]
