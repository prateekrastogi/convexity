"""COST realized-vs-forecast cost & funding residual → the ``cost_tca`` stream (§5.2).

Spec: OPS §5.2. COST's post-trade TCA produces realized round-trip cost and realized accrued funding
vs the forecast it made. The monitored observation is the **TCA residual**:

    e_cost(i)    = cost_realized(i)    - cost_forecast(i)      (round-trip, bps)
    e_funding(i) = funding_realized(i) - funding_forecast(i)   (signed, bps)

A persistent positive ``e_cost`` (realized cost exceeds even COST's conservative forecast) is the
canonical "cost model has gone stale" break — the §10 worked example. Detectors: CUSUM on ``e_cost``
(the centerpiece) confirmed by Page-Hinkley (the 2-of-2 AND, §6.6a); CUSUM on ``e_funding``. This
adapter is pure data shaping (Invariant STREAM); it does NOT re-price anything (OPS vs COST, §15).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.types import StreamObs


@dataclass(frozen=True)
class TcaFact:
    """One realized round-trip TCA from COST (the §5.2 inputs). A frozen FACT view, never re-priced."""

    cost_realized_bps: float
    cost_forecast_bps: float
    funding_realized_bps: float = 0.0
    funding_forecast_bps: float = 0.0
    funding_rate: float = 0.0          # current realized funding rate (for the §7.8 cap check)
    funding_cap: float | None = None   # venue funding cap (AMBER, from COST)
    paying_side: bool = False          # whether the book is on the paying side (§7.8)


class CostTcaStream:
    """Causal adapter COST TCA → ``cost_tca`` :class:`StreamObs` (§5.2)."""

    name = "cost_tca"

    def __init__(self) -> None:
        self._cost_idx = 0
        self._fund_idx = 0

    def observe_cost(self, fact: TcaFact, ts: int = 0) -> StreamObs:
        """The ``e_cost`` residual observation for the CUSUM + Page-Hinkley 2-of-2 (§5.2/§6.6a)."""
        e_cost = fact.cost_realized_bps - fact.cost_forecast_bps
        self._cost_idx += 1
        return StreamObs(
            stream="cost_tca",
            sample_idx=self._cost_idx,
            value=e_cost,
            ts=ts,
            meta={"kind": 0.0},  # 0 == cost arm
        )

    def observe_funding(self, fact: TcaFact, ts: int = 0) -> StreamObs:
        """The ``e_funding`` residual observation for the §7.8 funding-blowout CUSUM (§5.2)."""
        e_funding = fact.funding_realized_bps - fact.funding_forecast_bps
        self._fund_idx += 1
        near_cap = 0.0
        if fact.funding_cap is not None and fact.paying_side and fact.funding_cap > 0:
            near_cap = float(abs(fact.funding_rate) >= 0.9 * fact.funding_cap)
        return StreamObs(
            stream="cost_tca",
            sample_idx=self._fund_idx,
            value=e_funding,
            ts=ts,
            meta={"kind": 1.0, "near_cap": near_cap, "paying_side": float(fact.paying_side)},
        )


__all__ = ["TcaFact", "CostTcaStream"]
