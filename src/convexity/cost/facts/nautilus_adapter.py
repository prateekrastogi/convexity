"""Facts layer — adapt Nautilus commission/funding/fill objects into COST views (COST §1, Inv N).

COST builds the cost/funding POLICY *on top of* what Nautilus already owns: raw
``Commission`` per fill, ingested funding events, position/account state, restart
reconciliation. It does NOT rebuild any of that (Invariant N). This module defines
the FROZEN fact dataclasses (``FundingEvent``, ``Fill``, ``NautilusFacts``,
``ClosedTrade``, ``ReferenceMarks``) the rest of COST consumes — these are pure
Python and ARE exercised by the unit suite.

The single function that *touches* a live ``nautilus_trader`` object,
``nautilus_facts_from_engine``, is IMPORT-ISOLATED: the import lives inside the
function body, so importing this module (and running the unit suite) never imports
NautilusTrader (ARCHITECTURE §1: the test venv has no nautilus). The unit suite
builds these dataclasses directly from synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.cost.types import Side


@dataclass(frozen=True)
class FundingEvent:
    """One realized funding stamp Nautilus ingested (FACT, §5.5). Never modelled here."""

    ts: int  # epoch-ms of the funding stamp (Binance USDT-M: 00:00/08:00/16:00 UTC)
    rate: float  # realized funding rate r_k at this stamp
    notional_at_stamp: float  # signed position notional at the instant (read from Nautilus state)


@dataclass(frozen=True)
class Fill:
    """One fill, with its raw maker/taker commission (FACT, §1). ``commission`` is the
    Nautilus ``Commission`` re-keyed, not re-modelled (Invariant N)."""

    ts: int
    qty: float  # signed
    price: float
    commission: float  # quote ccy, FACT from Nautilus
    liquidity: str  # 'maker' | 'taker'


@dataclass(frozen=True)
class NautilusFacts:
    """The frozen view of one position's Nautilus-owned facts COST aggregates over (§1).

    ``accrued_fees`` is the sum of the per-fill ``Commission`` (Invariant N: a
    re-key/aggregate, not a re-model). ``funding_events`` are the realized stamps the
    accrual sums over (§5.5, FACT).
    """

    symbol: str
    fills: tuple[Fill, ...]
    funding_events: tuple[FundingEvent, ...]
    rolling_30d_volume: float  # FACT from Nautilus fills, drives the fee tier (§7)

    @property
    def accrued_fees(self) -> float:
        """Σ Nautilus Commission for the position (Invariant N)."""
        return float(sum(f.commission for f in self.fills))


@dataclass(frozen=True)
class ReferenceMarks:
    """Implementation-shortfall reference prices for TCA (§9). All FACT, read from Nautilus."""

    decision_price: float  # the signal/decision mark (Perold's "paper" price)
    arrival_price: float  # the actionable/arrival mark (decision->arrival = delay)
    avg_entry_price: float  # realized average fill on entry
    avg_exit_price: float  # realized average fill on exit
    final_mid: float  # mark used to value any unfilled/missed portion (opportunity cost)


@dataclass(frozen=True)
class ClosedTrade:
    """A closed round trip for post-trade TCA (§9). All FACT.

    ``gross_pnl`` is the decision-price-to-final paper P&L (no costs); ``net_pnl`` the
    realized P&L after every cost. ``intended_qty``/``filled_qty`` capture the
    unfilled/missed portion that becomes opportunity cost. ``carry_cost`` is the
    realized roll/MTF/storage carry over the hold (FACT). ``side`` signs funding.
    """

    symbol: str
    side: Side
    intended_qty: float
    filled_qty: float
    gross_pnl: float
    net_pnl: float
    realized_impact_cost: float  # FACT from EXECUTION (not re-modelled, §5.4)
    realized_carry_cost: float  # FACT realized roll/MTF/storage carry over the hold
    expected_at_entry: dict[str, float] = field(default_factory=dict)  # forecast per line (Inv E)


def nautilus_facts_from_engine(position: object, account: object) -> NautilusFacts:  # pragma: no cover
    """Build :class:`NautilusFacts` from a live ``nautilus_trader`` position/account.

    IMPORT-ISOLATED (ARCHITECTURE §1): the ``nautilus_trader`` import lives inside the
    body so importing this module never pulls in the engine, and the unit suite — which
    has no nautilus installed — never exercises this path. The aggregation is a
    re-key, not a re-model (Invariant N): commissions and funding events are READ.
    """
    import nautilus_trader  # noqa: F401  (presence check; real adapter wired on the VM)

    raise NotImplementedError(
        "nautilus_facts_from_engine is wired by the VM integration layer "
        "(strategy/ + */nautilus/); the unit suite builds NautilusFacts directly."
    )
