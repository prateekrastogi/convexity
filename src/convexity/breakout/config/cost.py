"""Validation-friction cost defaults for BREAKOUT (BREAKOUT §2 AMBER / §10).

BREAKOUT reuses the shared :class:`convexity.core.types.CostModel` (the validation-friction STUB);
the authoritative net-of-cost truth is the COST pillar (COST P1). This module only provides a small
factory of per-asset-class default ``CostModel`` instances for isolated unit tests + the baseline
suite, all under the IDENTICAL label contract (Invariant F). It is NEVER a promotion gate.
"""

from __future__ import annotations

from convexity.core.types import AssetClass, CostModel

# Conservative validation frictions by class (bps). Crypto-perp taker-heavy; equities tighter.
_DEFAULTS: dict[AssetClass, CostModel] = {
    "crypto_perp": CostModel(spread_bps=1.0, slippage_bps=1.5, fees_bps=4.0, min_tick_slippage=1.0),
    "crypto_spot": CostModel(spread_bps=1.5, slippage_bps=1.5, fees_bps=10.0, min_tick_slippage=1.0),
    "us_equity": CostModel(spread_bps=0.5, slippage_bps=0.5, fees_bps=0.2, min_tick_slippage=1.0),
    "commodity_future": CostModel(spread_bps=0.8, slippage_bps=1.0, fees_bps=0.5, min_tick_slippage=1.0),
    "precious_metal": CostModel(spread_bps=0.8, slippage_bps=1.0, fees_bps=0.5, min_tick_slippage=1.0),
    "industrial_metal": CostModel(spread_bps=1.2, slippage_bps=1.5, fees_bps=0.5, min_tick_slippage=1.0),
}


def default_cost_model(asset_class: AssetClass) -> CostModel:
    """Conservative per-class validation-friction CostModel (validation/baselines only, Inv F)."""
    return _DEFAULTS.get(asset_class, CostModel(2.0, 2.0, 5.0, 1.0))
