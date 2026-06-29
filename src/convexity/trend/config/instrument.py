"""InstrumentSpec field-name set for the Inv-C AMBER-only guard (TREND §4, Inv C).

``InstrumentSpec`` (shared core) is AMBER metadata only. No trail multiplier, ARL, alpha/beta,
Kalman ratio, horizon, or policy threshold may be stored there (Inv C). The Inv-C test feeds these
field names + :data:`convexity.trend.config.defaults.TREND_PARAM_REGISTRY` to
``assert_no_red_in_amber`` so any RED/BLUE name leaking onto the instrument spec is caught.
"""

from __future__ import annotations

from dataclasses import fields

from convexity.core.types import InstrumentSpec

# The AMBER field names on the shared InstrumentSpec — the only per-symbol surface TREND reads.
INSTRUMENT_SPEC_FIELDS: frozenset[str] = frozenset(f.name for f in fields(InstrumentSpec))
