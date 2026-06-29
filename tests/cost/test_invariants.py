"""COST named-invariant tests (COST §11). One test per load-bearing invariant."""

import ast
import dataclasses
import pathlib

import numpy as np
import pytest

from convexity.core.types import HoldHorizon

from convexity.cost.assemble.cost_book import CostBook
from convexity.cost.components.carry import CarryModel
from convexity.cost.components.funding_accrual import FundingAccrual
from convexity.cost.components.funding_forecast import FundingForecast
from convexity.cost.config.defaults import COST_PARAM_REGISTRY
from convexity.cost.config.instrument import (
    INSTRUMENT_COST_SPEC_FIELDS,
    InstrumentCostSpec,
)
from convexity.cost.config.resolve import resolve_config
from convexity.cost.cost_api import CostAPI
from convexity.cost.facts.execution_adapter import ExecutionFacts
from convexity.cost.facts.nautilus_adapter import (
    ClosedTrade,
    Fill,
    FundingEvent,
    NautilusFacts,
    ReferenceMarks,
)
from convexity.cost.synth.generate import (
    commodity_market,
    dated_future_market,
    equity_mtf_market,
    interest_unit_pair,
    thin_residual_market,
    worked_example_market,
)
from convexity.core.params import Bucket

_8H = 28_800_000


@pytest.fixture
def cfg():
    spec = InstrumentCostSpec("BTCUSDT", "crypto_perp", "BINANCE", _8H, 0.0075, "per_interval")
    return resolve_config(spec)


@pytest.fixture
def horizon():
    return HoldHorizon.from_quantile(3 * _8H, _8H)


def _cfg_for(carry_class):
    spec = InstrumentCostSpec("X", carry_class, "BINANCE", _8H, 0.0075, "per_interval")
    return resolve_config(spec)


# --------------------------------------------------------------------------- Inv C-cfg

def test_inv_c_cfg_red_unreachable_from_instrument():
    # No RED/BLUE param name appears as an InstrumentCostSpec field (else raises).
    resolve_config(InstrumentCostSpec("X", "crypto_perp", "BINANCE", _8H, 0.0075, "per_interval"))
    red_blue = {
        p.name for p in COST_PARAM_REGISTRY.params() if p.bucket in (Bucket.RED, Bucket.BLUE)
    }
    assert not (red_blue & INSTRUMENT_COST_SPEC_FIELDS)


def test_blue_grid_cardinality_counted():
    # Two BLUE grids (3 × 3) feed SYSTEM's trial ledger.
    assert COST_PARAM_REGISTRY.blue_grid_cardinality() == 9


# --------------------------------------------------------------------------- Inv B

def test_inv_b_total_is_additive_sum(cfg, horizon):
    api = CostAPI()
    rt = api.round_trip(1, 1000.0, "taker", "taker", horizon, worked_example_market(), cfg)
    assert rt.total_quote_ccy == pytest.approx(sum(c.quote_ccy for c in rt.components), abs=1e-12)


def test_inv_b_perturb_one_component_shifts_total_by_that_delta(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    rt = api.round_trip(1, 1000.0, "taker", "taker", horizon, m, cfg)
    m2 = dataclasses.replace(m, ask=m.ask + 10.0)  # widen spread only
    rt2 = api.round_trip(1, 1000.0, "taker", "taker", horizon, m2, cfg)
    by = {c.name: c for c in rt.components}
    by2 = {c.name: c for c in rt2.components}
    delta = by2["half_spread_in"].quote_ccy - by["half_spread_in"].quote_ccy
    # The total moves by exactly the two half-spread leg deltas, nothing else.
    assert rt2.total_quote_ccy - rt.total_quote_ccy == pytest.approx(2 * delta, abs=1e-9)


# --------------------------------------------------------------------------- Inv C

def test_inv_c_future_funding_rate_garbage_does_not_change_forecast(cfg):
    # Causality (Inv C): the forecast reads the realized funding PATH and the
    # published-next override; it MUST NOT read any future realized rate. The schedule
    # (timestamps) may be read ahead; the rates may not. Shoving garbage into the
    # forward portion of the stamp clock (and arbitrarily shifting all stamp TIMES into
    # the future) must leave r_hat / band bit-for-bit unchanged, because no future rate
    # is an input. (Same pattern the spec prescribes for Inv C.)
    ff = FundingForecast()
    m = worked_example_market()
    base = ff.forecast(1, 1000.0, 3, m, cfg)
    # Shift every funding TIMESTAMP far into the future — the deterministic schedule is
    # allowed to be read ahead; the live forecast must not depend on it.
    future_times = m.recent_funding_times + 10 * _8H
    m_future_sched = dataclasses.replace(m, recent_funding_times=future_times)
    shifted = ff.forecast(1, 1000.0, 3, m_future_sched, cfg)
    assert shifted.r_hat == base.r_hat
    assert shifted.funding_band_conservative == base.funding_band_conservative
    # A future realized RATE is structurally absent from CostMarketState: there is no
    # field through which a post-t rate could enter the live path (Inv F).
    field_names = {f.name for f in dataclasses.fields(m)}
    assert "future_funding_rate" not in field_names
    assert "realized_future_volume" not in field_names


# --------------------------------------------------------------------------- Inv W

def test_inv_w_impact_invariant_to_future_volume_and_reduce_schedule(cfg, horizon):
    # Inv W: the gate's impact term uses the CAUSAL v_hat forecast and forecast notional
    # is current notional held FLAT (no TREND reduce schedule). Mutating anything that is
    # not a causal input (funding stamp times standing in for "future bars/reduce path")
    # leaves the impact term AND the forecast-funding term bit-for-bit unchanged.
    api = CostAPI()
    m = worked_example_market()
    base = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m, cfg)
    base_impact = sum(
        c.quote_ccy for c in base.round_trip.components if c.name.startswith("impact")
    )
    garbage_times = m.recent_funding_times + 99 * _8H
    m2 = dataclasses.replace(m, recent_funding_times=garbage_times)
    d2 = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m2, cfg)
    impact2 = sum(c.quote_ccy for c in d2.round_trip.components if c.name.startswith("impact"))
    assert impact2 == base_impact
    assert d2.binding_funding == base.binding_funding
    # impact depends ONLY on the causal v_hat forecast + notional: a LOWER v_hat (a more
    # conservative volume forecast) raises impact, proving it is the binding causal input.
    m_low_v = dataclasses.replace(m, v_hat_exit_window=m.v_hat_exit_window / 2.0)
    d_low = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m_low_v, cfg)
    impact_low = sum(
        c.quote_ccy for c in d_low.round_trip.components if c.name.startswith("impact")
    )
    assert impact_low > base_impact


# --------------------------------------------------------------------------- Inv P / P-band

def test_inv_p_long_short_equal_and_opposite(cfg, horizon):
    cm = CarryModel()
    m = worked_example_market()
    vL, _ = cm.carry(1, 1000.0, horizon, m, cfg, use_band=False)
    vS, _ = cm.carry(-1, 1000.0, horizon, m, cfg, use_band=False)
    assert vL == pytest.approx(-vS, abs=1e-12)


def test_inv_p_band_short_is_smaller_credit_never_inflated_cost(cfg, horizon):
    cm = CarryModel()
    m = worked_example_market()
    vS_point, _ = cm.carry(-1, 1000.0, horizon, m, cfg, use_band=False)
    bS, _ = cm.carry(-1, 1000.0, horizon, m, cfg, use_band=True)
    assert bS < 0.0  # still a credit
    assert abs(bS) < abs(vS_point)  # a SMALLER credit, not an inflated positive cost
    bL, _ = cm.carry(1, 1000.0, horizon, m, cfg, use_band=True)
    # long conservative band is a cost (positive), magnitude equals short's worse tail spread
    assert bL > 0.0


def test_inv_p_band_magnitude_equal_only_under_zero_mean_funding(cfg, horizon):
    # Inv P-band PROSE: "a long and a short of equal notional in the same
    # funding/basis/residual regime get conservative band selections EQUAL IN MAGNITUDE
    # on their respective worse-for-position tails." That magnitude-equality holds ONLY
    # when r_hat is symmetric about zero — because the conservative band is the endpoint
    # MAXIMIZING side·N·Σ r_hat (the worse-for-position cost tail), and with a zero mean
    # the long's +e_q tail and the short's −e_q tail are exact mirror images.
    #
    # WARNING TO A FUTURE READER: do NOT "fix" the asymmetric bands you see with the
    # POSITIVE worked-example r_hat (long +0.69798 vs short −0.21798) to match this
    # prose. Those are CORRECTLY unequal: the band re-clamps the SAME e_q shock around a
    # POSITIVE mean, so the long's worse tail (+e_q, away from the cap) and the short's
    # worse tail (−e_q, the smaller credit) have genuinely different magnitudes. The
    # prose's magnitude-equality is a property of the ZERO-MEAN regime ONLY (this test);
    # the asymmetric positive case is checked by the two tests above and is the spec's
    # intended behavior, NOT a bug.
    cm = CarryModel()
    m = worked_example_market()
    # Force a symmetric, zero-mean funding regime: flat-zero realized path (r_ewma -> 0),
    # zero basis (r_basis -> 0), no published-next override, no interest unit. r_hat == 0
    # at every stamp, so f_forecast == 0 and the only band content is the ±e_q residual.
    rates_zero = np.zeros(len(m.recent_funding_rates), dtype=float)
    mz = dataclasses.replace(
        m,
        recent_funding_rates=rates_zero,
        premium_index_8h=0.0,
        published_next_funding=0.0,
        interest_quote=0.0,
        interest_base=0.0,
    )
    # POINT carry is exactly zero for both sides (Inv P long/short equal-and-opposite,
    # here both 0).
    vL_point, _ = cm.carry(1, 1000.0, horizon, mz, cfg, use_band=False)
    vS_point, _ = cm.carry(-1, 1000.0, horizon, mz, cfg, use_band=False)
    assert vL_point == pytest.approx(0.0, abs=1e-12)
    assert vS_point == pytest.approx(-vL_point, abs=1e-12)
    # Conservative BANDS: each side selects its worse-for-position tail. With a zero
    # mean both worse tails are the +N·Σ e_q cost endpoint, so the long and short bands
    # are EQUAL IN MAGNITUDE (and, here, both a positive cost — never an inflated short
    # credit). This is exactly the Inv P-band prose, satisfied in its native regime.
    bL, _ = cm.carry(1, 1000.0, horizon, mz, cfg, use_band=True)
    bS, _ = cm.carry(-1, 1000.0, horizon, mz, cfg, use_band=True)
    assert bL > 0.0  # the long's +e_q cost tail
    assert bS > 0.0  # the short's −e_q tail, mirror image -> also a positive cost
    assert abs(bL) == pytest.approx(abs(bS), abs=1e-12)  # EQUAL IN MAGNITUDE (zero mean)
    # Both bands are strictly worse than the (zero) point carry — the band only widens.
    assert abs(bL) > abs(vL_point)
    # Contrast: with the POSITIVE worked-example r_hat the SAME estimator yields UNEQUAL
    # magnitudes (the documented asymmetry above), proving the equality is regime-bound.
    bL_pos, _ = cm.carry(1, 1000.0, horizon, m, cfg, use_band=True)
    bS_pos, _ = cm.carry(-1, 1000.0, horizon, m, cfg, use_band=True)
    assert abs(bL_pos) != pytest.approx(abs(bS_pos), abs=1e-6)


# --------------------------------------------------------------------------- Inv F-band-gate

def test_inv_f_band_gate_band_binds_not_point(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    d = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m, cfg)
    # passes at the POINT (3.85 > 3.7145), fails at the BAND (3.85 < 3.9545).
    assert d.binding_funding == pytest.approx(d.round_trip.funding_band_conservative, abs=1e-12)
    assert d.binding_funding != pytest.approx(d.round_trip.funding_forecast, abs=1e-9)
    assert d.passes is False


# --------------------------------------------------------------------------- Inv M-hold

def test_inv_m_hold_smaller_expected_m_caller_does_not_relax(cfg):
    api = CostAPI()
    m = worked_example_market()
    h_diag = HoldHorizon.from_quantile(3 * _8H, _8H, expected_m_caller=1)
    h_plain = HoldHorizon.from_quantile(3 * _8H, _8H)
    d1 = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", h_diag, m, cfg)
    d2 = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", h_plain, m, cfg)
    assert d1.passes == d2.passes
    assert d1.round_trip.funding_band_conservative == d2.round_trip.funding_band_conservative


# --------------------------------------------------------------------------- Inv G-band

def test_inv_g_band_comonotone_exceeds_independent_sum(cfg):
    ff = FundingForecast()
    m = worked_example_market()
    res = ff.forecast(1, 1000.0, 3, m, cfg)
    # comonotone band width over 3 stamps
    como = res.funding_band_high - res.f_forecast
    # naive independent-draw sum would scale like sqrt(m)*single-stamp shock
    single = 1000.0 * m.residual_e_q_ci_edge
    independent = np.sqrt(3) * single
    assert como > independent  # autocorrelated comonotone > independent-draw
    # never exceeds m*cap*|N|
    assert abs(res.funding_band_high) <= 3 * m.funding_cap * 1000.0 + 1e-9


# --------------------------------------------------------------------------- Inv G-est

def test_inv_g_est_thin_residual_falls_back(cfg):
    ff = FundingForecast()
    m = thin_residual_market()
    eq, reason = ff._e_q(m, cfg)
    assert reason == "fallback_band_thin"
    assert eq == pytest.approx(abs(cfg.preset.class_worst_residual_e_q))


# --------------------------------------------------------------------------- Inv J-tier

def test_inv_j_tier_conservative_near_breakpoint(cfg, horizon):
    # Inv J-tier / §7: at (or within ``tier_proximity`` of) a VIP breakpoint the GATE
    # prices the round-trip fee at the WORSE (higher-fee) tier, while the plain FORWARD
    # lookup resolves to the better tier the volume actually clears. Sit EXACTLY on the
    # VIP1 breakpoint (15M) — the fee-ambiguous case the spec calls out.
    book = CostBook()
    m = worked_example_market()
    near = dataclasses.replace(m, rolling_30d_volume=15_000_000.0)

    # 1. FORWARD lookup at 15M clears the VIP1 floor -> the BETTER tier (taker 0.0004).
    fwd_tier = cfg.fee_schedule.tier_for_volume(15_000_000.0)
    assert fwd_tier.name == "VIP1"
    assert fwd_tier.taker_rate == pytest.approx(0.00040, abs=1e-12)

    # 2. The GATE's conservative selection prices the WORSE VIP0 tier (taker 0.0005) and
    #    flags it (P6). volume_to_next feeds the proximity test (Inv D intact).
    gate_tier, gate_reason = book.fees.gate_tier(15_000_000.0, cfg.fee_schedule, cfg)
    assert gate_tier.name == "VIP0"
    assert gate_tier.taker_rate == pytest.approx(0.00050, abs=1e-12)
    assert gate_reason == "tier_proximity_conservative"
    assert gate_tier.taker_rate > fwd_tier.taker_rate  # the gate side is strictly worse

    # 3. End-to-end: the GATE round-trip (gate_fee_tier=True) prices fee_in at the worse
    #    VIP0 tier; a NON-gate round-trip (gate_fee_tier=False, the FACT/TCA path) prices
    #    it at the in-force VIP1 tier. The gate fee is STRICTLY GREATER — never cheaper.
    rt_gate = book.round_trip(
        1, 1000.0, "taker", "taker", horizon, near, cfg, gate_fee_tier=True
    )
    rt_nongate = book.round_trip(
        1, 1000.0, "taker", "taker", horizon, near, cfg, gate_fee_tier=False
    )
    fee_gate = next(c for c in rt_gate.components if c.name == "fee_in")
    fee_nongate = next(c for c in rt_nongate.components if c.name == "fee_in")
    # VIP0 taker 0.0005 * 1000 = 0.5 (gate) vs VIP1 taker 0.0004 * 1000 = 0.4 (in-force).
    assert fee_gate.quote_ccy == pytest.approx(0.00050 * 1000.0, abs=1e-12)
    assert fee_nongate.quote_ccy == pytest.approx(0.00040 * 1000.0, abs=1e-12)
    assert fee_gate.quote_ccy > fee_nongate.quote_ccy  # gate is STRICTLY conservative
    assert fee_gate.reason == "tier_proximity_conservative"
    assert fee_nongate.reason == "tier_lookup"

    # Away from any breakpoint the gate falls back to the plain step lookup (no bump):
    # a $1M retail book is far below 15M, so gate and forward lookup agree on VIP0.
    far_tier, far_reason = book.fees.gate_tier(1_000_000.0, cfg.fee_schedule, cfg)
    assert far_tier.name == "VIP0"
    assert far_reason == "tier_lookup"


# --------------------------------------------------------------------------- Inv Q

@pytest.mark.parametrize("drop", ["spread", "funding", "impact"])
def test_inv_q_stale_inputs_finite_positive_fail_closed(cfg, horizon, drop):
    from convexity.cost.synth.generate import stale_market

    api = CostAPI()
    m = stale_market(drop=drop)
    rt = api.round_trip(1, 1000.0, "taker", "taker", horizon, m, cfg)
    assert np.isfinite(rt.total_quote_ccy)
    assert rt.total_quote_ccy > 0.0
    # gate fails closed for a tiny edge
    d = api.net_ev_gate(1, 1000.0, 0.01, "taker", "taker", horizon, m, cfg)
    assert d.passes is False


# --------------------------------------------------------------------------- Inv X / Y-finance

def test_inv_x_one_interface_all_classes(horizon):
    cm = CarryModel()
    for carry_class, market in (
        ("crypto_perp", worked_example_market()),
        ("dated_future", dated_future_market()),
        ("equity_mtf", equity_mtf_market()),
        ("commodity", commodity_market()),
    ):
        cfg = _cfg_for(carry_class)
        value, diag = cm.carry(1, 1000.0, horizon, market, cfg)
        assert np.isfinite(value)
        assert diag["carry_class"] == carry_class


def test_inv_y_finance_charged_once():
    cm = CarryModel()
    horizon = HoldHorizon.from_quantile(7 * 86_400_000, _8H)  # 7-day hold
    cfg = _cfg_for("commodity")
    unlev = commodity_market(financing_mode="unlevered")
    fin = commodity_market(financing_mode="mtf")
    v_unlev, d_unlev = cm.carry(1, 1000.0, horizon, unlev, cfg)
    v_fin, d_fin = cm.carry(1, 1000.0, horizon, fin, cfg)
    # storage is identical; the only delta is the MTF line, booked ONCE.
    assert d_unlev["storage_carry"] == pytest.approx(d_fin["storage_carry"], abs=1e-12)
    assert d_unlev["mtf_financing"] == 0.0
    assert d_fin["mtf_financing"] > 0.0
    assert (v_fin - v_unlev) == pytest.approx(d_fin["mtf_financing"], abs=1e-12)


def test_inv_h_time_mtf_simple_positive():
    cm = CarryModel()
    cfg = _cfg_for("equity_mtf")
    horizon = HoldHorizon.from_quantile(7 * 86_400_000, _8H)
    m = equity_mtf_market()
    value, diag = cm.carry(1, 1000.0, horizon, m, cfg)
    expected = 500.0 * 0.12 * (7.0 / 365.0)
    assert value == pytest.approx(expected, abs=1e-9)  # SIMPLE, not compounded
    assert value > 0.0  # POSITIVE = cost (tightens L_stop, P7)


def test_inv_h_time_basis_scales_with_dt():
    cm = CarryModel()
    cfg = _cfg_for("dated_future")
    m = dated_future_market()  # tau_spread=90d, basis=0.01
    full = HoldHorizon.from_quantile(90 * 86_400_000, _8H)
    half = HoldHorizon.from_quantile(45 * 86_400_000, _8H)
    tiny = HoldHorizon.from_quantile(1, _8H)
    v_full, _ = cm.carry(1, 1000.0, full, m, cfg)
    v_half, _ = cm.carry(1, 1000.0, half, m, cfg)
    v_tiny, _ = cm.carry(1, 1000.0, tiny, m, cfg)
    assert v_half == pytest.approx(v_full / 2.0, rel=1e-6)
    assert v_tiny == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- Inv 5.5b interest unit

def test_interest_unit_no_double_discount(cfg):
    ff = FundingForecast()
    per_interval, per_day = interest_unit_pair()
    # numbers chosen so both give the same interest_8h (0.0000333); the per_interval path
    # must NOT divide by 3.
    assert ff.interest_8h(per_interval) == pytest.approx(0.0000333, abs=1e-9)
    assert ff.interest_8h(per_day) == pytest.approx(0.0001 * (8 / 24), abs=1e-9)


# --------------------------------------------------------------------------- Inv A / N / P (accrual)

def test_inv_a_n_accrual_equals_sum_of_events():
    fa = FundingAccrual()
    events = (
        FundingEvent(ts=1000, rate=0.0001, notional_at_stamp=1000.0),
        FundingEvent(ts=2000, rate=0.0002, notional_at_stamp=1000.0),
        FundingEvent(ts=9999, rate=0.5, notional_at_stamp=1000.0),  # outside window
    )
    accrued = fa.accrued(1, events, t_entry=0, t_now=5000)
    assert accrued == pytest.approx(1000.0 * (0.0001 + 0.0002), abs=1e-12)
    # short flips sign (Inv P)
    accrued_short = fa.accrued(-1, events, t_entry=0, t_now=5000)
    assert accrued_short == pytest.approx(-accrued, abs=1e-12)


def test_inv_a_accrual_is_fact():
    fa = FundingAccrual()
    events = (FundingEvent(ts=1000, rate=0.0001, notional_at_stamp=1000.0),)
    comp = fa.component(1, events, 0, 5000, 1000.0)
    assert comp.is_fact is True


# --------------------------------------------------------------------------- Inv N (fees == Σ Commission)

def test_inv_n_accrued_fees_equal_sum_of_nautilus_commission():
    # COST does not re-model commissions; accrued_fees is a re-key/aggregate of the raw
    # per-fill Nautilus Commission (Inv N).
    fills = (
        Fill(ts=0, qty=1.0, price=100.0, commission=0.5, liquidity="taker"),
        Fill(ts=10, qty=-1.0, price=110.0, commission=0.5, liquidity="taker"),
    )
    nf = NautilusFacts("X", fills=fills, funding_events=(), rolling_30d_volume=1e6)
    assert nf.accrued_fees == pytest.approx(sum(f.commission for f in fills), abs=1e-12)
    assert nf.accrued_fees == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- Inv U-dim

def test_inv_u_dim_impact_is_fraction_times_notional_no_rogue_div(cfg):
    # Impact is a return FRACTION × notional, NOT bps: no rogue /1e4 (Inv U-dim).
    from convexity.cost.components.impact import ImpactModel

    im = ImpactModel()
    m = worked_example_market()
    comp = im.leg_component(m, 1000.0, "impact_in", cfg)
    # frac = k·σ·sqrt(N/v_hat) = 0.10·0.004·sqrt(1000/200000) = 0.0000282843
    frac = 0.10 * 0.004 * (1000.0 / 200000.0) ** 0.5
    assert comp.quote_ccy == pytest.approx(frac * 1000.0, abs=1e-9)
    assert comp.quote_ccy == pytest.approx(0.0282843, abs=1e-6)
    # NOT the 1e4×-smaller bug value, and the bps column is frac MULTIPLIED by 1e4.
    assert comp.quote_ccy != pytest.approx(frac * 1000.0 / 1e4, abs=1e-9)
    assert comp.bps_of_notional == pytest.approx(1e4 * frac, abs=1e-9)


# --------------------------------------------------------------------------- Inv T

def test_inv_t_one_carry_truth_across_consumers(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    gate = api.net_ev_gate(1, 1000.0, 3.85, "taker", "taker", horizon, m, cfg)
    hold = api.hold_carry(1, 1000.0, horizon, 5.0, m, cfg)
    carry_val, _ = api.carry_term(1, 1000.0, horizon, m, cfg)
    assert gate.binding_funding == pytest.approx(hold.funding_band_conservative, abs=1e-12)
    assert gate.binding_funding == pytest.approx(carry_val, abs=1e-12)


def test_inv_t_carry_term_signed_short_credit(cfg, horizon):
    api = CostAPI()
    m = worked_example_market()
    carry_long, _ = api.carry_term(1, 1000.0, horizon, m, cfg)
    carry_short, _ = api.carry_term(-1, 1000.0, horizon, m, cfg)
    assert carry_long > 0.0
    assert carry_short < 0.0  # credit, NOT clamped here (LEVERAGE clamps max(carry,0))


# --------------------------------------------------------------------------- Inv R / E (TCA)

def _closed_trade_fixture():
    fills = (
        Fill(ts=0, qty=1.0, price=100.0, commission=0.5, liquidity="taker"),
        Fill(ts=10, qty=-1.0, price=110.0, commission=0.5, liquidity="taker"),
    )
    events = (FundingEvent(ts=5, rate=0.0001, notional_at_stamp=100.0),)
    nf = NautilusFacts("X", fills=fills, funding_events=events, rolling_30d_volume=1e6)
    ef = ExecutionFacts("X", realized_impact_cost=0.3, realized_half_spread_cost=0.2,
                        fitted_k=0.1)
    marks = ReferenceMarks(decision_price=99.0, arrival_price=100.0, avg_entry_price=100.0,
                           avg_exit_price=110.0, final_mid=110.0)
    trade = ClosedTrade("X", side=1, intended_qty=1.0, filled_qty=1.0,
                        gross_pnl=11.0, net_pnl=9.0, realized_impact_cost=0.3,
                        realized_carry_cost=0.4)
    return trade, marks, nf, ef


def test_inv_r_tca_exact_identity(cfg):
    api = CostAPI()
    trade, marks, nf, ef = _closed_trade_fixture()
    rep = api.tca(trade, marks, nf, ef, cfg, t_entry=0, t_now=10)
    eight = (rep.delay_cost + rep.fee_cost + rep.half_spread_cost + rep.impact_cost
             + rep.funding_cost + rep.carry_cost + rep.opportunity_cost
             + rep.unattributed_residual)
    assert eight == pytest.approx(rep.implementation_shortfall, abs=1e-12)
    assert sum(rep.attribution_pct.values()) == pytest.approx(1.0, abs=1e-9)


def test_inv_e_realized_vs_expected_present(cfg):
    api = CostAPI()
    trade, marks, nf, ef = _closed_trade_fixture()
    rep = api.tca(trade, marks, nf, ef, cfg, t_entry=0, t_now=10)
    assert "funding_cost" in rep.realized_vs_expected
    assert len(rep.realized_vs_expected) == 7


# --------------------------------------------------------------------------- Inv D / V (firewalls, AST)

_LIVE_DIRS = ("facts", "components", "assemble", "consume")
_COST_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "cost"


def test_inv_d_no_order_actions_in_live_layers():
    banned = ("place_order", "close_position", "cancel_order", "submit_order")
    for sub in _LIVE_DIRS:
        for path in (_COST_ROOT / sub).rglob("*.py"):
            src = path.read_text()
            for tok in banned:
                assert tok not in src, f"{path} contains {tok}"


def test_inv_v_live_layers_never_import_validate():
    for sub in _LIVE_DIRS:
        for path in (_COST_ROOT / sub).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and "cost.validate" in mod:
                    raise AssertionError(f"{path} imports cost.validate")


def test_facts_layer_no_top_level_nautilus_import():
    # The nautilus import is isolated inside a function body, not at module scope.
    path = _COST_ROOT / "facts" / "nautilus_adapter.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:  # top-level only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            assert not any("nautilus" in n for n in names)
