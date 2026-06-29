"""MarginModel — broker/exchange margin model, portfolio-margin aware (LEVERAGE §8.4).

BROKER-REAL, NOT THEORETICAL (§8.4): the margin requirement comes from the actual
broker/exchange margin model or an API snapshot — NOT a fixed leverage constant.
Futures margin is SCENARIO-BASED (SPAN-style: evaluate portfolio loss across a grid
of price/vol scenarios), so the model exposes BOTH the current requirement and a
STRESSED (correlation-break / hiked) requirement; the margin ceiling sizes against
the stressed one (P4 — netted portfolio margin SPIKES when correlations break).

A hardcoded notional/margin ratio is allowed ONLY as a clearly-labeled offline
placeholder before a broker feed is wired (``FlatRatioMarginModel`` below); the
SPAN-style ``ScenarioMarginModel`` is the production shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class MarginModel(Protocol):
    """The margin surface the ceiling consumes (§8.4). Both methods return a margin
    requirement in account currency for a position of ``notional`` (signed by side)."""

    def requirement(self, notional: float, price: float) -> float:
        """Current (calm) margin requirement."""
        ...

    def stressed_requirement(self, notional: float, price: float, stress_multiplier: float) -> float:
        """Requirement under a correlation-break / margin-hike scenario (P4)."""
        ...


@dataclass(frozen=True)
class ScenarioMarginModel:
    """SPAN-style scenario margin (§8.4). The requirement is the WORST portfolio loss
    across a grid of price-shift / vol-shift scenarios — the production margin shape.

    ``price_scenarios`` are signed fractional price shifts (e.g. ±0.03, ±0.05); the
    margin is the maximum adverse loss over the grid. The stressed branch widens the
    grid by ``stress_multiplier`` (and the broker's event/expiry hike), modelling the
    netted-margin spike when diversification vanishes (P4).
    """

    price_scenarios: tuple[float, ...] = (-0.03, -0.05, -0.08, 0.03, 0.05, 0.08)
    vol_multiplier_scenarios: tuple[float, ...] = (1.0, 1.5)
    # broker minimum margin floor as a fraction of notional (never below this).
    min_margin_fraction: float = 0.05

    def _worst_loss(self, notional: float, scenarios: Sequence[float], vol_mults: Sequence[float]) -> float:
        """Worst adverse |loss| over the price×vol scenario grid, as account-ccy margin."""
        n = abs(float(notional))
        worst = 0.0
        for shift in scenarios:
            for vm in vol_mults:
                loss = n * abs(shift) * vm  # adverse move on the position notional
                worst = max(worst, loss)
        return max(worst, self.min_margin_fraction * n)

    def requirement(self, notional: float, price: float) -> float:
        return self._worst_loss(notional, self.price_scenarios, self.vol_multiplier_scenarios)

    def stressed_requirement(self, notional: float, price: float, stress_multiplier: float) -> float:
        stressed_scen = tuple(s * stress_multiplier for s in self.price_scenarios)
        stressed_vol = tuple(v * stress_multiplier for v in self.vol_multiplier_scenarios)
        return self._worst_loss(notional, stressed_scen, stressed_vol)


@dataclass(frozen=True)
class FlatRatioMarginModel:
    """OFFLINE PLACEHOLDER ONLY (§8.4): a fixed initial-margin fraction of notional,
    used before the broker/exchange feed is wired. Clearly labelled — a hardcoded
    notional/margin ratio is forbidden in production (§8.4)."""

    initial_margin_fraction: float = 0.115  # ~8.7x permitted leverage (NIFTY-like, §13)

    def requirement(self, notional: float, price: float) -> float:
        return self.initial_margin_fraction * abs(float(notional))

    def stressed_requirement(self, notional: float, price: float, stress_multiplier: float) -> float:
        return self.initial_margin_fraction * stress_multiplier * abs(float(notional))


def financing_over_horizon(notional: float, annual_rate: float, horizon_years: float) -> float:
    """Financing/carry COST over the holding horizon, account ccy (§8.4): notional ×
    annual_rate × horizon_years. POSITIVE is a cost (reduces effective leverage and,
    via the §8.1 carry term, tightens L_stop). Simple Actual/365-style accrual."""
    return abs(float(notional)) * float(annual_rate) * max(float(horizon_years), 0.0)


__all__ = [
    "MarginModel",
    "ScenarioMarginModel",
    "FlatRatioMarginModel",
    "financing_over_horizon",
]
