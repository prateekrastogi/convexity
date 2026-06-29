"""CarryTerm — the SIGNED carry term handed to LEVERAGE's stop ceiling (COST §8.3, P2).

Returns ``(carry_quote_ccy, diagnostics)`` — the forecast funding+carry over the
holding horizon, SIGNED (a credit is negative). ONE signed truth (P1) with TWO
reader conventions:
  * the net-EV gate (§8.1) and the funding-aware hold gate (§8.2) consume it SIGNED
    (a credit is real expected income there);
  * the survival stop-ceiling consumes only its COST side — LEVERAGE's StopCeiling
    subtracts the CLAMPED carry, ``L_stop = (c - max(carry,0))/d``. The
    ``max(carry,0)`` clamp lives in LEVERAGE (ARCHITECTURE §5.5), NOT here: a credit
    does NOT relax the hard survival cap (its EV is already in the gate).

Same forecast as §8.1/§8.2 (Invariant T) — bit-for-bit, because it calls the SAME
``CarryModel`` with the SAME inputs.
"""

from __future__ import annotations

from typing import Mapping

from convexity.core.types import HoldHorizon

from convexity.cost.components.carry import CarryModel
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostMarketState, Side


class CarryTerm:
    """LEVERAGE stop-ceiling carry term (§8.3). Thin transform over the SAME carry model."""

    def __init__(self, carry: CarryModel | None = None) -> None:
        self.carry = carry if carry is not None else CarryModel()

    def over_horizon(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> tuple[float, Mapping[str, float]]:
        """Return ``(carry_quote_ccy, diagnostics)`` SIGNED (credit negative) (§8.3).

        Uses the side-selected conservative band (the same truth the gate/hold input
        use, Invariant T). The ``max(carry,0)`` clamp is LEVERAGE's, not COST's
        (ARCHITECTURE §5.5).
        """
        carry_value, diag = self.carry.carry(
            side, notional, horizon, market, cfg, use_band=True
        )
        # Return diagnostics as a float mapping (the seam types it Mapping[str,float]).
        float_diag: dict[str, float] = {
            k: float(v) for k, v in diag.items() if isinstance(v, (int, float))
        }
        return carry_value, float_diag
