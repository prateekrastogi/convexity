"""State -> g_t band table (REGIME.md §4.1).

Four states ordered by severity, each mapping to a BAND whose FLOOR is the conservative live ``g_t``
(§4.1: the cross-sectional book's exposure is continuous, so REGIME emits the band's floor as the
live value, per §0 P6, while the band itself is recorded for research/BLUE-grid purposes). This is
an explicit correction of the old draft's discrete {1,.75,.5,.25} ladder (tuned for a
single-instrument book) — HALT here floors at EXACTLY 0 (full flat), not the old .25 (§4.1).
"""

from __future__ import annotations

from convexity.core.types import RegimeStateName
from convexity.regime.config.resolve import ResolvedRegimeConfig

# States ordered by severity (REGIME.md §4.1). Index = severity rank; used by the machine to compare
# and to enforce one-level-at-a-time re-arm (§4.2).
SEVERITY_ORDER: tuple[RegimeStateName, ...] = ("HEALTHY", "WATCH", "STRESSED", "HALT")


def severity(state: RegimeStateName) -> int:
    """Severity rank of a state (0=HEALTHY .. 3=HALT), for escalate/deescalate comparisons (§4.1)."""
    return SEVERITY_ORDER.index(state)


def g_t_for_state(state: RegimeStateName, cfg: ResolvedRegimeConfig) -> float:
    """The conservative live ``g_t`` band FLOOR for a state (REGIME.md §4.1).

    HEALTHY=1.00, WATCH=0.75, STRESSED=0.40, HALT=0.00 (RED, §4.1 / config defaults). g_t is floored
    at the state's value, never interpolated below it and never exceeding it while the state holds
    (§4.1); HALT floors at exactly 0 (§4.1 update over the old .25 floor)."""
    return cfg.g_bands[state]
