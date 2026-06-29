"""Shared TREND test fixtures. H_sys is SYSTEM-owned, injected here as a fixture (Inv HC)."""

import pytest

from convexity.core.types import HoldHorizon, InstrumentSpec, Position
from convexity.trend import resolve_config


@pytest.fixture
def spec():
    return InstrumentSpec(
        symbol="BTCUSDT",
        asset_class="crypto_perp",
        venue="BINANCE",
        timezone="UTC",
        tick_size=0.1,
        lot_size=0.001,
        funding_interval_ms=28_800_000,
    )


# H_sys is the SYSTEM-injected operational cap (Inv HC). 7 calendar days is the typical value.
H_SYS_DAYS = 7.0


@pytest.fixture
def cfg(spec):
    return resolve_config(spec, h_sys=H_SYS_DAYS)


def make_position(bars, entry_idx=0, side=1, stop_frac=0.02):
    """A position (long or short) seeded with a BREAKOUT-style initial invalidation + risk."""
    p0 = float(bars.c[entry_idx])
    inval = p0 * (1 - stop_frac) if side > 0 else p0 * (1 + stop_frac)
    return Position(
        side=side,
        entry_idx=entry_idx,
        entry_price=p0,
        initial_invalidation=inval,
        initial_risk=abs(p0 - inval),
    )


def make_trend_cost_source(*, market_kind="calm", notional=1000.0):
    """Build the §6.1 / Inv M COST-sourced friction adapter from the REAL COST pillar (cost_api),
    NOT the local stub. ``market_kind`` selects a COST synth market: 'calm' (light friction) or
    'spike' (a funding shock whose carry flips a gross-positive regime net-negative)."""
    from convexity.cost.config.instrument import InstrumentCostSpec
    from convexity.cost.config.resolve import resolve_config as cost_resolve
    from convexity.cost.cost_api import CostAPI
    from convexity.cost.synth.generate import calm_market, funding_spike_market
    from convexity.trend.validate.cost_source import TrendCostSource

    cost_spec = InstrumentCostSpec(
        symbol="BTCUSDT",
        carry_class="crypto_perp",
        fee_schedule_venue="BINANCE",
        funding_interval_ms=28_800_000,
        funding_cap=0.0075,
        interest_rate_unit="per_interval",
    )
    market = funding_spike_market() if market_kind == "spike" else calm_market()
    return TrendCostSource(
        api=CostAPI(), market=market, cfg=cost_resolve(cost_spec), notional=notional
    )


def cost_hold(interval_ms=28_800_000, stamps=3):
    """A COST ``HoldHorizon`` over ``stamps`` funding intervals (for the cost source)."""
    return HoldHorizon.from_quantile(stamps * interval_ms, interval_ms)
