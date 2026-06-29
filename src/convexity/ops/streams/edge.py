"""Realized hit-rate / edge-per-trade / trade-frequency → the ``edge`` stream (§5.4).

Spec: OPS §5.4. The alpha-decay streams — the most important and hardest, because edge is the
lowest-frequency, highest-variance stream (one observation per closed trade). Per closed trade:

    e_edge(j)    = edge_realized(j)        (net-of-cost P&L per trade, R-multiples or bps)
    hit(j)       = 1{trade j profitable}   (Bernoulli stream)
    freq(window) = trades per unit time vs validated frequency

Detectors: the CUSUM-SPRT change-point form on the hit-rate LLR (§6.5a, v1 default) + CUSUM on
``e_edge`` (the continuous complement) + Shewhart on the trade-frequency count per window. This adapter
emits THREE observation kinds; the reconcile map binds each to its detector. The ``hit`` observation
is 0/1 (consumed by SPRT/CUSUM-SPRT, which treat ``value`` as the Bernoulli outcome); ``e_edge`` is a
standardized residual; ``freq`` is the windowed count.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.types import StreamObs


@dataclass(frozen=True)
class TradeFact:
    """One closed trade from SYSTEM (the §5.4 inputs). A frozen FACT view, never re-validated."""

    edge_realized: float       # net-of-cost P&L per trade (R-multiples or bps)
    profitable: bool           # the Bernoulli hit


class EdgeStream:
    """Causal adapter closed-trade → ``edge`` :class:`StreamObs` (§5.4).

    Three observation kinds, each with its own sample index so its detector's ARL is counted on the
    right base (per-trade for hit/edge, per-window for frequency).
    """

    name = "edge"

    def __init__(self) -> None:
        self._hit_idx = 0
        self._edge_idx = 0
        self._freq_idx = 0

    def observe_hit(self, fact: TradeFact, ts: int = 0) -> StreamObs:
        """The Bernoulli ``hit`` observation for the CUSUM-SPRT / SPRT (§5.4/§6.5a)."""
        self._hit_idx += 1
        return StreamObs(
            stream="edge",
            sample_idx=self._hit_idx,
            value=1.0 if fact.profitable else 0.0,
            ts=ts,
            meta={"kind": 0.0},  # hit arm
        )

    def observe_edge(self, fact: TradeFact, ts: int = 0) -> StreamObs:
        """The ``e_edge`` per-trade observation for the continuous CUSUM (§5.4)."""
        self._edge_idx += 1
        return StreamObs(
            stream="edge",
            sample_idx=self._edge_idx,
            value=fact.edge_realized,
            ts=ts,
            meta={"kind": 1.0},  # edge arm
        )

    def observe_frequency(self, trades_in_window: float, validated_freq: float,
                          ts: int = 0) -> StreamObs:
        """The trade-frequency observation for the Shewhart count chart (§5.4).

        ``value`` is the realized count minus the validated frequency (a frequency *collapse* — the
        system has stopped finding trades — and an *explosion* — over-trading noise — are both breaks).
        """
        self._freq_idx += 1
        return StreamObs(
            stream="edge",
            sample_idx=self._freq_idx,
            value=trades_in_window - validated_freq,
            ts=ts,
            meta={"kind": 2.0, "trades_in_window": trades_in_window},  # frequency arm
        )


__all__ = ["TradeFact", "EdgeStream"]
