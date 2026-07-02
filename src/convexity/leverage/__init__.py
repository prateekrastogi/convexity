"""LEVERAGE pillar — portfolio-level gross-exposure sizing (LEVERAGE.md).

Sizes the ONE scalar ``G_t`` (gross exposure as a multiple of equity) for the weekly, dollar-neutral
cross-sectional-momentum book: a fractional-Kelly / vol-target term off a DSR-gated bootstrap-CB
Sharpe, a capacity ceiling, and a drawdown governor, composed by ``min()`` (§6.1). Public entry:
``convexity.leverage.engine_api.LeverageEngine.size_book``.
"""
