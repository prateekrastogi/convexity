"""FundingAccrual — realized funding over [entry, now], a FACT (COST §5.5, Invariant A/N/P).

    F_accrued = Σ_{k : t_entry <= τ_k <= t_now}  side * Notional(τ_k) * r_k

summed over the funding timestamps that ACTUALLY occurred (read from Nautilus's
ingested funding events — NOT modelled, Invariant N). A perp LONG paying positive
funding accrues a COST (``F_accrued > 0`` is money out); a SHORT in the same regime
accrues a CREDIT (Invariant P). ``Notional(τ_k)`` is the UNSIGNED |position notional|
AT ``τ_k`` (it changes with price and partial reduces) read from Nautilus state, never
assumed to be entry notional — the trade's ``side`` factor supplies the sign (audit
C-1: a SIGNED notional here would double-sign, flipping a short's funding credit into
a cost; the tests pin the unsigned+side convention). This is a FACT (``is_fact=True``), kept strictly
separate from the forecast (Invariant A).
"""

from __future__ import annotations

from convexity.cost.facts.nautilus_adapter import FundingEvent
from convexity.cost.types import CostComponent, Side


class FundingAccrual:
    """Realized funding accrual (§5.5, FACT)."""

    def accrued(
        self,
        side: Side,
        funding_events: tuple[FundingEvent, ...],
        t_entry: int,
        t_now: int,
    ) -> float:
        """``Σ side * Notional(τ_k) * r_k`` over stamps in [t_entry, t_now] (§5.5, FACT).

        ``FundingEvent.notional_at_stamp`` is the UNSIGNED |position notional| at each
        stamp (Invariant N: read, not recomputed); ``side`` supplies the sign. The result is signed (Invariant P):
        positive = money out for a long paying positive funding.
        """
        total = 0.0
        for ev in funding_events:
            if t_entry <= ev.ts <= t_now:
                total += side * ev.notional_at_stamp * ev.rate
        return total

    def component(
        self,
        side: Side,
        funding_events: tuple[FundingEvent, ...],
        t_entry: int,
        t_now: int,
        notional: float,
    ) -> CostComponent:
        """The accrued-funding line as a FACT ``CostComponent`` (``is_fact=True``, Invariant A)."""
        quote = self.accrued(side, funding_events, t_entry, t_now)
        bps = 1e4 * quote / notional if notional else 0.0
        return CostComponent(
            name="funding_accrued",
            quote_ccy=quote,
            bps_of_notional=bps,
            is_fact=True,
            reason="accrued_from_nautilus_events",
        )
