"""Realized vol vs LEVERAGE-assumed vol → the ``vol`` stream (§5.3).

Spec: OPS §5.3. Per bar/decision, LEVERAGE's vol forecast assumed for sizing vs the realized short-
horizon vol Nautilus's bars deliver. The observation is the log-ratio (symmetric + additive):

    e_vol(t) = log( realized_vol(t) / assumed_vol(t) )         (~0 in control)

A realized vol 2× the assumed is +0.69, half is -0.69. A sustained positive ``e_vol`` means the book
is sized for calmer markets than it is getting — the leverage ceilings were derived against a vol the
world has left. Detectors: EWMA (vol error is autocorrelated — EWMA is the right smoother) + CUSUM.
The §4 TEAL diagnostic ``σ_live/σ0`` gap surfaces here as telemetry, never into a limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from convexity.ops.types import StreamObs


@dataclass(frozen=True)
class VolFact:
    """Realized vs assumed vol for one bar/decision (the §5.3 inputs). A frozen FACT view."""

    realized_vol: float
    assumed_vol: float
    sigma_live: float | None = None   # diagnostic-only rolling live σ (§4 TEAL; never a limit)


class VolStream:
    """Causal adapter LEVERAGE/Nautilus vol → ``vol`` :class:`StreamObs` (§5.3)."""

    name = "vol"

    def __init__(self) -> None:
        self._idx = 0

    def observe(self, fact: VolFact, ts: int = 0) -> StreamObs:
        """The ``e_vol`` log-ratio observation for the EWMA + CUSUM charts (§5.3)."""
        if fact.assumed_vol <= 0.0 or fact.realized_vol <= 0.0:
            # Degenerate vol — fail toward SAFE (P6): treat as a large positive vol error.
            e_vol = math.log(max(fact.realized_vol, 1e-9) / max(fact.assumed_vol, 1e-9))
        else:
            e_vol = math.log(fact.realized_vol / fact.assumed_vol)
        self._idx += 1
        meta = {}
        if fact.sigma_live is not None:
            meta["sigma_live"] = fact.sigma_live  # §4 TEAL telemetry only
        return StreamObs(stream="vol", sample_idx=self._idx, value=e_vol, ts=ts, meta=meta)


__all__ = ["VolFact", "VolStream"]
