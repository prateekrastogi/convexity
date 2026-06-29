"""EXECUTION realized IS / fill-rate / adverse-selection → the ``fills`` stream (§5.1).

Spec: OPS §5.1. EXECUTION emits, per fill, realized implementation shortfall ``IS_realized`` (bps vs
arrival mid), realized fill rate, and realized adverse selection. The monitored observation is the
**residual** vs the model:

    e_fill(i)     = IS_realized(i) - IS_expected(i)            (bps, the model error)
    e_fillrate(i) = fillrate_realized(i) - fillrate_expected(i)

In control ``e_fill`` is NEGATIVE-or-zero-mean (SYSTEM's exec-stub is a conservative upper bound,
Invariant XC — realized cost ≤ modeled). A positive drift means realized fills are WORSE than the
conservative model — a real break. This adapter is pure data shaping (no modelling, no decision —
Invariant STREAM): one :class:`StreamObs` per fill.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.types import StreamObs


@dataclass(frozen=True)
class FillFact:
    """One realized fill from EXECUTION (the §5.1 inputs). A frozen FACT view, never re-modelled."""

    is_realized_bps: float
    is_expected_bps: float
    fillrate_realized: float
    fillrate_expected: float
    adverse_selection_bps: float = 0.0


class FillsStream:
    """Causal adapter EXECUTION fill → ``fills`` :class:`StreamObs` (§5.1). Stateless apart from idx."""

    name = "fills"

    def __init__(self) -> None:
        self._idx = 0
        self._fr_idx = 0

    def observe(self, fact: FillFact, ts: int = 0) -> StreamObs:
        """The ``e_fill`` residual observation for the CUSUM/EWMA/Shewhart charts (§5.1)."""
        e_fill = fact.is_realized_bps - fact.is_expected_bps
        self._idx += 1
        return StreamObs(
            stream="fills",
            sample_idx=self._idx,
            value=e_fill,
            ts=ts,
            meta={
                "e_fillrate": fact.fillrate_realized - fact.fillrate_expected,
                "fillrate_realized": fact.fillrate_realized,
                "adverse_selection_bps": fact.adverse_selection_bps,
            },
        )

    def observe_fillrate(self, fact: FillFact, ts: int = 0) -> StreamObs:
        """The ``e_fillrate`` residual observation for the §7.4 fill-rate-collapse trigger."""
        e_fr = fact.fillrate_realized - fact.fillrate_expected
        self._fr_idx += 1
        return StreamObs(
            stream="fills",
            sample_idx=self._fr_idx,
            value=e_fr,
            ts=ts,
            meta={"fillrate_realized": fact.fillrate_realized},
        )


__all__ = ["FillFact", "FillsStream"]
