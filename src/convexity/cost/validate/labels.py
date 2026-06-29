"""Validation labels — realized round-trip cost / funding / IS (COST §12, FUTURE-ONLY).

These labels use FUTURE bars (the realized round-trip cost, the realized funding
accrual, the realized implementation shortfall). They are structurally unreachable
from the live layers (Invariant V): nothing in ``facts/``, ``components/``,
``assemble/`` or ``consume/`` imports this module. The AST firewall test asserts that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealizedLabels:
    """The realized (future-only) cost labels for one closed round trip (§12)."""

    realized_round_trip_cost: float  # realized total cost, quote ccy
    realized_funding_accrual: float  # F_accrued (FACT), quote ccy
    realized_implementation_shortfall: float  # gross - net, quote ccy


def realized_funding(side: int, notionals: np.ndarray, rates: np.ndarray) -> float:
    """``Σ side·Notional(τ_k)·r_k`` over the REALIZED funding path (future-only, §5.5)."""
    notionals = np.asarray(notionals, dtype=float)
    rates = np.asarray(rates, dtype=float)
    return float(side * np.sum(notionals * rates))


def realized_round_trip_cost(
    fees: float, half_spread: float, impact: float, funding: float, carry: float
) -> float:
    """The realized additive round-trip cost (future-only label, §12)."""
    return fees + half_spread + impact + funding + carry


def build_labels(
    *,
    fees: float,
    half_spread: float,
    impact: float,
    realized_funding_value: float,
    carry: float,
    gross_pnl: float,
    net_pnl: float,
) -> RealizedLabels:
    """Assemble the realized labels for one round trip (§12, future-only)."""
    return RealizedLabels(
        realized_round_trip_cost=realized_round_trip_cost(
            fees, half_spread, impact, realized_funding_value, carry
        ),
        realized_funding_accrual=realized_funding_value,
        realized_implementation_shortfall=gross_pnl - net_pnl,
    )
