"""LEVERAGE worked-example regression fixtures (LEVERAGE §15, BUILD_PLAN Phase 3).

Each pinned worked example reproduced to the spec's stated tolerance.
"""

import numpy as np
import pytest

from convexity.leverage.ceilings.kelly import KellyCeiling
from convexity.leverage.ceilings.stop import StopCeiling
from convexity.leverage.ceilings.vol_target import VolTargetCeiling
from convexity.leverage.combine.governor import DrawdownGovernor
from convexity.leverage.combine.kelly_fraction import AdaptiveKellyFraction
from convexity.leverage.config.instrument import InstrumentLeverageSpec
from convexity.leverage.config.margin import ScenarioMarginModel
from convexity.leverage.config.resolve import resolve_config
from convexity.leverage.combine.lot_sizer import Quantizer
from convexity.leverage.synth import generate as g
from convexity.leverage.types import ConditionalMAE
from convexity.leverage.vol.forecast import forecast_vol


def _cfg():
    return g.calm().cfg


# --------------------------------------------------------------------------- L_stop carry-sign clamp

def test_l_stop_carry_sign_clamp():
    cfg = _cfg()
    st = StopCeiling()
    mae = ConditionalMAE({"low_vol": 0.0, "global": 0.0}, {"low_vol": 100, "global": 100})
    # d = max(2*0.06, 0) = 0.12, c=0.13
    l_none, diag = st.ceiling(1, 100.0, 0.13, 0.0, 0.06, "low_vol", mae, cfg)
    assert diag["d"] == pytest.approx(0.12)
    assert l_none == pytest.approx(0.13 / 0.12)
    # positive carry 0.03 -> (0.13-0.03)/0.12
    l_cost, _ = st.ceiling(1, 100.0, 0.13, 0.03, 0.06, "low_vol", mae, cfg)
    assert l_cost == pytest.approx((0.13 - 0.03) / 0.12)
    # credit -0.03 clamped to 0 -> equals no-carry
    l_credit, _ = st.ceiling(1, 100.0, 0.13, -0.03, 0.06, "low_vol", mae, cfg)
    assert l_credit == pytest.approx(l_none)


# --------------------------------------------------------------------------- L_vol calm/volatile

def test_l_vol_calm_and_volatile():
    vc = VolTargetCeiling()
    l_calm, _ = vc.ceiling(0.05, 0.175)
    l_vol, _ = vc.ceiling(0.35, 0.175)
    assert l_calm == pytest.approx(0.175 / 0.05)
    assert l_vol == pytest.approx(0.175 / 0.35)
    assert l_calm > l_vol  # rises in calm, falls in volatile


# --------------------------------------------------------------------------- d(2T)/d(T) = sqrt(2)

def test_stop_distance_scales_sqrt2():
    o = np.array([100.0, 100.5, 101.0, 100.7, 101.2, 101.5])
    h, low, c = o + 0.4, o - 0.4, o + 0.05
    v1 = forecast_vol(o, h, low, c, horizon_periods=1.0)
    v4 = forecast_vol(o, h, low, c, horizon_periods=4.0)
    assert v4 == pytest.approx(2.0 * v1, rel=1e-9)  # sqrt(4)=2


# --------------------------------------------------------------------------- L_kelly notional-invariance

def test_l_kelly_notional_invariant():
    kc = KellyCeiling()
    cfg = _cfg()
    # doubling the contract multiplier must NOT change L_kelly (returns-based, §5.1)
    l1, _ = kc.ceiling(0.05, 0.1, 0.12, 0.5, cfg)
    spec2 = InstrumentLeverageSpec(
        symbol="BTCUSDT", asset_class="crypto_perp", lot_step=0.001,
        contract_multiplier=2.0, tick_size=0.01,
    )
    cfg2 = resolve_config(spec2)
    l2, _ = kc.ceiling(0.05, 0.1, 0.12, 0.5, cfg2)
    assert l1 == pytest.approx(l2)
    # explicit value: 0.5 * 0.05/0.1^2 = 0.5*5 = 2.5
    assert l1 == pytest.approx(0.5 * 0.05 / 0.01)


def test_l_kelly_slack_when_exceeds_other_ceilings():
    # a large mu makes L_kelly huge so it does NOT bind (calm scenario binds on stop).
    sc = g.calm()
    import dataclasses
    risk = dataclasses.replace(sc.risk, mu_hat=5.0)
    from convexity.leverage.engine_api import size_position
    d = size_position(sc.signal, risk, sc.cfg, sc.book)
    assert d.binding_ceiling != "kelly"


# --------------------------------------------------------------------------- governor ladder + ramp

@pytest.mark.parametrize("dd,expected", [
    (0.05, 1.00), (0.15, 0.75), (0.25, 0.50), (0.35, 0.25),
    (0.415, 0.125), (0.42, 0.25 * (0.43 - 0.42) / 0.03), (0.45, 0.0),
])
def test_governor_ladder_and_ramp(dd, expected):
    cfg = _cfg()
    gov = DrawdownGovernor()
    m, _ = gov.multiplier(10_000.0, 10_000.0 * (1.0 - dd), cfg)
    assert m == pytest.approx(expected, rel=1e-9)


def test_governor_dd042_is_00833():
    cfg = _cfg()
    gov = DrawdownGovernor()
    m, _ = gov.multiplier(10_000.0, 5_800.0, cfg)  # dd = 0.42
    assert m == pytest.approx(0.0833333, abs=1e-6)


# ----------------------------------------------------------- §15 boundary-EXACT: steps EXACTLY at bp

@pytest.mark.parametrize("equity,expected,band", [
    (9_000.0, 0.75, "dd==0.10"),   # 1 - 9000/10000 = 0.0999999...8 (float-fragile) -> 0.75
    (8_000.0, 0.50, "dd==0.20"),   # inclusive-of-breakpoint: steps DOWN at the breakpoint
    (7_000.0, 0.25, "dd==0.30"),
    (6_000.0, 0.25, "dd==0.40"),   # exactly at ramp lo -> ramp top = 0.25
    (5_700.0, 0.00, "dd==0.43"),   # exactly at flat -> 0.00
])
def test_governor_exact_breakpoints_step_down(equity, expected, band):
    # The 'steps EXACTLY at 10/20/30/40%' gate (§15): dd computed as 1 - equity/peak is
    # float-fragile at the literal breakpoints; the epsilon-robust, inclusive-of-breakpoint
    # convention (§6) makes a breakpoint hit take the LOWER (more conservative) multiplier.
    cfg = _cfg()
    gov = DrawdownGovernor()
    m, reason = gov.multiplier(10_000.0, equity, cfg)
    assert m == pytest.approx(expected, rel=1e-9), f"{band}: got {m} reason={reason}"


def test_governor_breakpoint_is_inclusive_not_held_high():
    # Pin the convention DIRECTLY: dd exactly 0.10 must NOT hold the higher 1.00 band.
    cfg = _cfg()
    gov = DrawdownGovernor()
    m, _ = gov.multiplier(10_000.0, 9_000.0, cfg)  # dd ~ 0.10
    assert m == pytest.approx(0.75)
    assert m != pytest.approx(1.00)


# --------------------------------------------------------------------------- adaptive phi {0.5, 1/3}

def test_adaptive_phi_switches_and_hysteresis():
    cfg = _cfg()
    af = AdaptiveKellyFraction()
    # calm -> 0.5
    phi_calm, _ = af.fraction(0.3, 0.1, 0.5, cfg)
    assert phi_calm == pytest.approx(0.5)
    # high vol -> 1/3
    phi_hi, _ = af.fraction(0.95, 0.1, 0.5, cfg)
    assert phi_hi == pytest.approx(1.0 / 3.0)
    # hysteresis: once stressed, an in-between vol pct (0.85, between lo 0.80 and hi 0.90)
    # stays stressed.
    phi_hold, _ = af.fraction(0.85, 0.1, 0.5, cfg, prev_phi=1.0 / 3.0)
    assert phi_hold == pytest.approx(1.0 / 3.0)
    # below lo -> relax back to 0.5
    phi_relax, _ = af.fraction(0.5, 0.1, 0.5, cfg, prev_phi=1.0 / 3.0)
    assert phi_relax == pytest.approx(0.5)


# --------------------------------------------------------------------------- MCTR sum identity

def test_mctr_sum_identity():
    from convexity.leverage.allocate.risk_budget import mctr, portfolio_vol
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 4))
    cov = a @ a.T / 4.0
    w = np.array([0.3, 0.2, 0.4, 0.1])
    assert float(np.sum(mctr(w, cov))) == pytest.approx(portfolio_vol(w, cov), rel=1e-12)


# --------------------------------------------------------------------------- g_t=1.0 no-overlay

def test_g1_equals_no_overlay():
    import dataclasses
    from convexity.leverage.engine_api import size_position
    cut = g.regime_cut_full_equity()
    d_cut = size_position(cut.signal, dataclasses.replace(cut.risk, g_t=1.0), cut.cfg, cut.book)
    dual = g.no_regime_overlay()
    d_dual = size_position(dual.signal, dual.risk, dual.cfg, dual.book)
    assert d_cut.leverage == pytest.approx(d_dual.leverage, rel=1e-9)


# --------------------------------------------------------------------------- g_t=0.25 cut at full equity

def test_g25_cut_at_full_equity():
    from convexity.leverage.engine_api import size_position
    sc = g.regime_cut_full_equity()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.binding_ceiling == "portfolio"
    assert "regime_heat_cut" in d.reason_codes
    assert d.governor_multiplier == pytest.approx(1.0)
    assert d.g_t == pytest.approx(0.25)


# --------------------------------------------------------------------------- round-down + divergence

def test_quantize_round_down_and_divergence():
    # divergence = (raw_steps - units)/raw_steps. Construct raw_steps = 37, lot_step 1,
    # then cap at 30 lots -> divergence = (37-30)/37 = 0.189189...
    spec = InstrumentLeverageSpec(
        symbol="BANKNIFTY", asset_class="us_equity", lot_step=1.0,
        contract_multiplier=1.0, tick_size=0.05, exchange_max_lots=30,
    )
    cfg2 = resolve_config(spec)
    q = Quantizer()
    mm = ScenarioMarginModel(min_margin_fraction=0.001)
    # price 1.0, mult 1.0 -> unit_notional = 1.0; raw_notional 37.4 -> raw_steps 37.4
    res = q.quantize(
        raw_leverage=1.0, raw_notional=37.4, instrument=spec,
        position_margin=1000.0, account_equity=1e6, entry_price=1.0, stop_distance=0.01,
        capital_cap=0.13, margin_model=mm, free_margin=1e6, existing_margin_used=0.0, cfg=cfg2,
    )
    assert res.units == 30  # exchange_max_capped
    assert "exchange_max_capped" in res.flags
    assert res.divergence == pytest.approx((37.4 - 30) / 37.4, rel=1e-6)


def test_quantize_divergence_0189189():
    # exact pinned divergence 0.189189 = (37-30)/37
    spec = InstrumentLeverageSpec(
        symbol="X", asset_class="us_equity", lot_step=1.0, contract_multiplier=1.0,
        tick_size=0.01, exchange_max_lots=30,
    )
    cfg2 = resolve_config(spec)
    q = Quantizer()
    mm = ScenarioMarginModel(min_margin_fraction=0.0001)
    res = q.quantize(
        raw_leverage=1.0, raw_notional=37.0, instrument=spec,
        position_margin=1000.0, account_equity=1e7, entry_price=1.0, stop_distance=0.001,
        capital_cap=0.13, margin_model=mm, free_margin=1e7, existing_margin_used=0.0, cfg=cfg2,
    )
    assert res.units == 30
    assert res.divergence == pytest.approx(0.189189, abs=1e-6)
