"""TrendDeathHazard — optional event-count hazard ladder (TREND §9 hazard).

Disabled by default (§9, §20): hazard stays off until there are enough labeled trend-death events,
keeping v1 cheap and avoiding fake sophistication. The event-count ladder (§9):

  * disabled : default if events are insufficient or validation fails.
  * constant : fewer than ~15 events -> a single constant hazard rate.
  * single   : ~15-50 events -> one covariate only.
  * state    : 50+ events -> 3-5 interpretable covariates.

A richer rung must BEAT constant hazard out of sample; if not, fall back to the simpler rung
(promotion is decided in ``validate/``/``tune/``, not here). ``update`` returns ``(hazard, rung)``;
hazard can REDUCE, not hard-exit, unless a validated class-level hard-exit rule is configured (§9).
"""

from __future__ import annotations

from typing import Mapping

from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.types import HazardRung


class EventTable:
    """A minimal labeled-death-event container the ladder fits on (validation-fed)."""

    def __init__(self, n_events: int = 0, constant_rate: float = 0.0) -> None:
        self.n_events = int(n_events)
        self.constant_rate = float(constant_rate)


class TrendDeathHazard:
    """Event-count hazard ladder, disabled by default (§9)."""

    def __init__(self) -> None:
        self._rung: HazardRung = "disabled"
        self._constant_rate = 0.0

    def fit(self, events: EventTable, cfg: ResolvedConfig) -> None:
        """Select the rung from the event count (§9). Honors the RED default ``hazard_mode`` cap."""
        mode = cfg.defaults.hazard_mode
        if mode == "disabled":
            self._rung = "disabled"
            return
        n = events.n_events
        if n < 15:
            self._rung = "constant" if mode in ("constant", "single", "state") and n > 0 else "disabled"
        elif n < 50:
            self._rung = "single" if mode in ("single", "state") else "constant"
        else:
            self._rung = "state" if mode == "state" else ("single" if mode == "single" else "constant")
        self._constant_rate = events.constant_rate

    def update(self, state: Mapping[str, float]) -> tuple[float, HazardRung]:
        """Return ``(hazard, rung)`` (§9). Disabled => 0 hazard. Hazard may reduce, not hard-exit."""
        if self._rung == "disabled":
            return 0.0, "disabled"
        if self._rung == "constant":
            return self._constant_rate, "constant"
        # single/state: a thin interpretable model would live here; v1 returns the base rate scaled
        # by an interpretable covariate (giveback fraction) when supplied. Still reduce-only.
        cov = float(state.get("giveback_fraction", 0.0))
        return self._constant_rate * (1.0 + cov), self._rung
