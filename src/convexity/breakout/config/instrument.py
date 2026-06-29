"""Per-instrument AMBER metadata for BREAKOUT (BREAKOUT §2 AMBER, Invariant D).

BREAKOUT reuses the shared :class:`convexity.core.types.InstrumentSpec` (the single cross-pillar
AMBER object) rather than redefining one — the foundation directive forbids re-declaring boundary
types. This module only EXPOSES the spec's AMBER field-name set so ``resolve_config`` /
the Inv-D test can run ``assert_no_red_in_amber`` against it: a RED/BLUE name appearing on the
instrument spec is the overfit bug Invariant D forbids.
"""

from __future__ import annotations

from dataclasses import fields

from convexity.core.types import InstrumentSpec

# The AMBER field names carried per-instrument. Frozen set so the Inv-D firewall can intersect it
# with the RED/BLUE param names and prove the intersection is empty.
INSTRUMENT_SPEC_FIELDS: frozenset[str] = frozenset(f.name for f in fields(InstrumentSpec))

__all__ = ["InstrumentSpec", "INSTRUMENT_SPEC_FIELDS"]
