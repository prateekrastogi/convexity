"""The execution stub — a CONSERVATIVE cost UPPER bound (SYSTEM §6.1b, Invariant XC).

The net-of-cost labels (and therefore the entire DSR gate) are only as honest as this stub. A stub
that under-estimates fills/slippage/funding biases every Sharpe UP and lets a no-edge system pass.
The contract is therefore NOT "model cost accurately" but "OVER-estimate cost", so the DSR is biased
LOW and a config that passes the gate passes DESPITE a pessimistic cost assumption:

  spread   : charge the FULL quoted spread (cross the book), never mid or half-spread, entry AND exit.
  slippage : assume the WORST plausible fill within the bar (buys at/above the high region, sells
             at/below the low region), parameterized by a conservative impact model.
  impact   : square-root market impact at the modeled participation rate, scaled UP by a RED safety
             multiple.
  funding  : full funding/financing/carry over the realized hold, with NO netting of favorable
             funding.
  fees     : full taker fees on every fill (never assume maker rebates).

All cost components are charged at their pessimistic end; favorable realizations are NOT credited.
The stub's net P&L is a LOWER bound on achievable net P&L (Invariant XC). This is NautilusTrader-free
pure math (the live engine reads its own fills); the stub exists for validation parity only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StubCost:
    """The conservative per-fill cost decomposition (Section 6.1b). ``total`` is in fraction-of-
    notional terms and is a LOWER bound on net P&L when subtracted (Invariant XC)."""

    spread: float
    slippage: float
    impact: float
    funding: float
    fees: float

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.impact + self.funding + self.fees


class ExecStub:
    """The conservative cost + fill UPPER bound (SYSTEM §6.1b, Invariant XC)."""

    def __init__(self, defaults) -> None:
        self._d = defaults

    def fill_cost(
        self,
        *,
        spread_frac: float,
        bar_range_frac: float,
        participation: float,
        taker_fee_frac: float,
        funding_frac: float,
        n_funding_stamps: int,
        side: str = "entry",
    ) -> StubCost:
        """The conservative cost charged on ONE fill (Section 6.1b).

        ``spread_frac`` = the full quoted spread as a fraction of price (charged FULL, both sides);
        ``bar_range_frac`` = (high − low)/price (the adverse-extreme slippage budget, scaled by the
        RED ``slippage_extreme_fraction``); ``participation`` ∈ [0,1] feeds the sqrt impact model,
        scaled UP by the RED ``impact_safety_multiple``; ``funding_frac`` per stamp × stamps, NO
        favorable netting; ``taker_fee_frac`` the full taker fee (never a maker rebate).
        """
        d = self._d
        spread = abs(spread_frac)  # full spread, never half
        slippage = abs(bar_range_frac) * d.slippage_extreme_fraction  # worst-in-bar
        impact = d.impact_safety_multiple * math.sqrt(max(participation, 0.0)) * abs(spread_frac)
        funding = abs(funding_frac) * max(n_funding_stamps, 0)  # full, no netting
        fees = abs(taker_fee_frac)
        return StubCost(spread=spread, slippage=slippage, impact=impact, funding=funding, fees=fees)

    def round_trip_cost(
        self,
        *,
        spread_frac: float,
        bar_range_frac: float,
        participation: float,
        taker_fee_frac: float,
        funding_frac: float,
        n_funding_stamps: int,
    ) -> float:
        """The full round-trip (entry + exit) conservative cost (Section 6.1b). Used as the
        ``entry_cost + exit_cost`` of the triple-barrier label."""
        entry = self.fill_cost(
            spread_frac=spread_frac, bar_range_frac=bar_range_frac, participation=participation,
            taker_fee_frac=taker_fee_frac, funding_frac=funding_frac, n_funding_stamps=n_funding_stamps,
            side="entry",
        )
        exit_ = self.fill_cost(
            spread_frac=spread_frac, bar_range_frac=bar_range_frac, participation=participation,
            taker_fee_frac=taker_fee_frac, funding_frac=0.0, n_funding_stamps=0, side="exit",
        )
        return entry.total + exit_.total

    @staticmethod
    def mid_reference_cost(spread_frac: float, taker_fee_frac: float) -> float:
        """A realistic mid/half-spread reference cost for ONE fill (the Invariant-XC calibration
        baseline). The stub's charged cost MUST be ≥ this on every fill, else the stub under-charges
        and the conservatism test FAILS (Section 6.1b)."""
        return 0.5 * abs(spread_frac) + abs(taker_fee_frac)
