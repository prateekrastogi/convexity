"""COST §5.7 worked example reproduced bit-for-bit (Invariant WE).

The single anchor pinning the dimensional + band conventions: every line incl.
`C_impact=0.0282843` (NOT 2.83e-6, Inv U-dim), `F_band_conservative=0.69798`,
`total_BAND=1.9545485`, `passes=False`, `margin=-0.1045485`; EWMA
`alpha=1-2^(-1/9)=0.074125`.
"""

import pytest

from convexity.core.types import HoldHorizon

from convexity.cost.config.defaults import CostDefaults
from convexity.cost.config.instrument import InstrumentCostSpec
from convexity.cost.config.resolve import resolve_config
from convexity.cost.cost_api import CostAPI
from convexity.cost.synth.generate import worked_example_market

TOL = 1e-6


@pytest.fixture
def cfg():
    spec = InstrumentCostSpec(
        symbol="BTCUSDT",
        carry_class="crypto_perp",
        fee_schedule_venue="BINANCE",
        funding_interval_ms=28_800_000,
        funding_cap=0.0075,
        interest_rate_unit="per_interval",
    )
    return resolve_config(spec)


@pytest.fixture
def horizon():
    return HoldHorizon.from_quantile(3 * 28_800_000, 28_800_000)


def test_ewma_alpha_and_ess():
    d = CostDefaults()
    assert d.funding_ewma_alpha == pytest.approx(0.074125, abs=1e-6)
    assert d.funding_ewma_ess == pytest.approx(25.98, abs=0.05)


def test_worked_example_every_line(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    d = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m, cfg)
    rt = d.round_trip
    by = {c.name: c for c in rt.components}

    assert by["fee_in"].quote_ccy == pytest.approx(0.5, abs=TOL)
    assert by["fee_out"].quote_ccy == pytest.approx(0.5, abs=TOL)
    assert by["half_spread_in"].quote_ccy == pytest.approx(0.1, abs=TOL)
    assert by["half_spread_out"].quote_ccy == pytest.approx(0.1, abs=TOL)

    # Inv U-dim: impact is a fraction × notional, NOT 1e4× smaller.
    assert by["impact_in"].quote_ccy == pytest.approx(0.0282843, abs=TOL)
    assert by["impact_out"].quote_ccy == pytest.approx(0.0282843, abs=TOL)
    assert by["impact_in"].quote_ccy != pytest.approx(2.83e-6, abs=1e-7)

    # funding: point is telemetry, the conservative band is the binding summand.
    assert rt.funding_forecast == pytest.approx(0.45798, abs=TOL)
    assert rt.funding_band_conservative == pytest.approx(0.69798, abs=TOL)
    assert d.binding_funding == pytest.approx(rt.funding_band_conservative, abs=TOL)

    assert rt.total_quote_ccy == pytest.approx(1.9545485, abs=TOL)
    assert d.margin == pytest.approx(-0.1045485, abs=TOL)
    assert d.passes is False
    assert d.funding_sensitivity == pytest.approx(0.15399, abs=TOL)
    assert d.funding_m_sensitive is True


def test_point_total_telemetry(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    d = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m, cfg)
    rt = d.round_trip
    point_total = (
        rt.total_quote_ccy - rt.funding_band_conservative + rt.funding_forecast
    )
    assert point_total == pytest.approx(1.7145485, abs=TOL)
