"""SpreadImpactModel: derived per-name half-spread + Amihud impact (COST.md Sections 5.2-5.5, 7).

Invariants exercised, each a failing-if-violated test: CONV (one half-spread convention, median
anchor reproduces accounting.py's flat defaults bit-for-bit), U-dim (impact is a fraction, no rogue
/1e4), N50 (inf, never a finite number, for an unshortable short-leg name), the Amihud-undefined
fallback to CRYPTO_FRICTION.impact (Section 10), and PARTICIPATION (the calibrated regime).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.xsec.accounting import (
    HALF_SPREAD_FRAC,
    IMPACT_FRAC,
    round_trip_friction,
    settle_week,
)
from research.xsec.cost.spread_impact import (
    ANCHOR_NOTIONAL,
    SpreadImpactModel,
    amihud_ratio,
)
from research.xsec.universe import CRYPTO_FRICTION

# A representative median-liquidity anchor: $40M/day ADV, a small Amihud ratio.
ADV_MEDIAN = 40_000_000.0
AMIHUD_MEDIAN = 1.0e-10


def _model() -> SpreadImpactModel:
    return SpreadImpactModel.anchor(ADV_MEDIAN, AMIHUD_MEDIAN)


def test_conv_median_name_reproduces_flat_research_defaults_bit_for_bit():
    # Invariant CONV / COST.md Section 5.5: at the calibration anchor the DERIVED per-name rate
    # equals accounting.py's flat HALF_SPREAD_FRAC / IMPACT_FRAC defaults exactly — the per-name
    # model is a drop-in generalization, no extra /2 composed (the double-discount bug CONV catches).
    m = _model()
    assert m.half_spread(ADV_MEDIAN) == pytest.approx(HALF_SPREAD_FRAC, rel=1e-12)
    assert m.impact_frac(AMIHUD_MEDIAN, ANCHOR_NOTIONAL) == pytest.approx(IMPACT_FRAC, rel=1e-12)
    rt = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL)
    assert rt == pytest.approx(round_trip_friction(), rel=1e-12)  # the 13bp anchor
    assert rt == pytest.approx(0.0013, rel=1e-12)


def test_half_spread_sqrt_law_scaling():
    # COST.md Section 5.2: a name with 1/16 the median liquidity pays sqrt(16)=4x the median spread;
    # 4x the liquidity pays half. One anchor + the sqrt form fixes every other name.
    m = _model()
    thin = m.half_spread(ADV_MEDIAN / 16.0)
    thick = m.half_spread(ADV_MEDIAN * 4.0)
    assert thin == pytest.approx(4.0 * HALF_SPREAD_FRAC, rel=1e-12)
    assert thick == pytest.approx(0.5 * HALF_SPREAD_FRAC, rel=1e-12)


def test_u_dim_impact_is_a_fraction_no_rogue_div_by_1e4():
    # Invariant U-dim / COST.md Section 5.4: impact_frac is a return FRACTION multiplying notional
    # directly; C_impact = impact_frac * notional in quote ccy with NO further /1e4. The bps column
    # is frac * 1e4 (the reverse direction). Catch the historical 1e4-too-small bug.
    m = _model()
    frac = m.impact_frac(AMIHUD_MEDIAN, ANCHOR_NOTIONAL)
    assert frac == pytest.approx(IMPACT_FRAC, rel=1e-12)  # ~0.5bp as a fraction (5e-5)
    impact_bps = 1e4 * frac
    assert impact_bps == pytest.approx(0.5, rel=1e-9)  # 0.5 bp reporting column
    # the rogue-div bug would make the fraction 1e4x too small:
    assert frac != pytest.approx(IMPACT_FRAC / 1e4, rel=1e-6)


def test_impact_grows_sublinearly_with_notional():
    # COST.md Section 5.3: impact_frac = k * sqrt(Amihud * Notional) — 4x the trade size raises the
    # impact fraction by sqrt(4) = 2x, not 4x (the square-root concavity, Section 5.2).
    m = _model()
    base = m.impact_frac(AMIHUD_MEDIAN, ANCHOR_NOTIONAL)
    quad = m.impact_frac(AMIHUD_MEDIAN, 4.0 * ANCHOR_NOTIONAL)
    assert quad == pytest.approx(2.0 * base, rel=1e-12)


def test_tail_name_costs_more_worked_example_section_5_6():
    # COST.md Section 5.6 worked example: name B at ADV_median/16 pays ~4bp half-spread and ~2bp
    # impact -> ~22bp round-trip vs the median's 13bp, ~1.7x more per unit turnover. This is the
    # illiquid-tail cost mechanism (no separate universe-size cost model needed).
    m = _model()
    # tune the impact anchor so the median impact is 0.5bp; a 16x-thinner name has 16x the Amihud
    # ratio (return-per-dollar scales inversely with volume), so impact_frac scales sqrt(16)=4x.
    rt_median = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL)
    rt_tail = m.round_trip_friction_i(
        ADV_MEDIAN / 16.0, AMIHUD_MEDIAN * 16.0, notional_leg=ANCHOR_NOTIONAL
    )
    assert rt_median == pytest.approx(0.0013, rel=1e-9)
    # half_spread 4bp, impact 4*0.5=2bp, fee 5bp -> 2*(5+4+2)=22bp
    assert rt_tail == pytest.approx(0.0022, rel=1e-9)
    assert rt_tail / rt_median == pytest.approx(0.0022 / 0.0013, rel=1e-9)


def test_n50_unshortable_short_leg_returns_inf_never_finite():
    # Invariant N50 / COST.md Section 7: a name placed in the SHORT leg that cannot be borrowed gets
    # round_trip_friction_i = math.inf (never NaN, never a large-but-finite placeholder) so
    # settle_week rejects the infeasible short arithmetically instead of pricing it.
    m = _model()
    rt = m.round_trip_friction_i(
        ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL, is_shortable=False,
        in_short_leg=True,
    )
    assert rt == math.inf
    assert not math.isnan(rt)


def test_n50_gap_state_none_prices_normally():
    # COST.md Section 7 item 1: until UNIVERSE exposes is_shortable, callers pass None and the model
    # prices the name at its normal finite liquidity cost (a documented KNOWN GAP, not a silent
    # success). A shortable name, or a name NOT in the short leg, is also priced normally.
    m = _model()
    normal = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL)
    assert (
        m.round_trip_friction_i(
            ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL, is_shortable=None
        )
        == normal
    )
    assert (
        m.round_trip_friction_i(
            ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL, is_shortable=True,
            in_short_leg=True,
        )
        == normal
    )
    # unshortable but NOT in the short leg (it is a long) -> priced normally, not inf.
    assert (
        m.round_trip_friction_i(
            ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL, is_shortable=False,
            in_short_leg=False,
        )
        == normal
    )


def test_amihud_undefined_falls_back_to_class_constant_never_zero():
    # COST.md Section 10 / P5: an undefined Amihud (NaN / non-positive: thin sample) falls back to
    # CRYPTO_FRICTION.impact — never zero, never an extrapolated thin-sample fit.
    m = _model()
    assert m.impact_frac(float("nan"), ANCHOR_NOTIONAL) == CRYPTO_FRICTION.impact
    assert m.impact_frac(0.0, ANCHOR_NOTIONAL) == CRYPTO_FRICTION.impact
    assert m.impact_frac(-1e-9, ANCHOR_NOTIONAL) == CRYPTO_FRICTION.impact
    assert CRYPTO_FRICTION.impact > 0.0


def test_zero_adv_is_inf_not_silent_zero():
    # COST.md Section 10: half_spread = c/sqrt(ADV) diverges as ADV -> 0; a non-positive ADV (should
    # be pre-excluded upstream) yields an inf rate — an untradeable name, never a silent 0.
    m = _model()
    assert m.half_spread(0.0) == math.inf
    assert m.round_trip_friction_i(0.0, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL) == math.inf


def test_round_trip_friction_i_requires_explicit_notional_leg():
    # Footgun closed (review gap 3): the per-rebalance surface must state the true leg notional, never
    # silently price at the fixed calibration ANCHOR_NOTIONAL. Omitting notional_leg is a TypeError,
    # not a silent misprice at 40k regardless of |dw_i|*NAV.
    m = _model()
    with pytest.raises(TypeError):
        m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        m.friction_vector(np.full(2, ADV_MEDIAN), np.full(2, AMIHUD_MEDIAN))  # type: ignore[call-arg]
    # a larger real leg prices STRICTLY higher impact than the calibration size (sqrt-scaling), so the
    # value genuinely depends on the passed notional — not frozen at the anchor.
    at_anchor = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL)
    at_4x = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=4.0 * ANCHOR_NOTIONAL)
    assert at_4x > at_anchor


def test_anchor_rejects_degenerate_universe():
    with pytest.raises(ValueError):
        SpreadImpactModel.anchor(0.0, AMIHUD_MEDIAN)
    with pytest.raises(ValueError):
        SpreadImpactModel.anchor(ADV_MEDIAN, 0.0)


def test_friction_vector_reduces_to_scalar_at_the_anchor():
    # COST.md Section 5.6: at the median anchor the per-symbol friction vector is constant, so the
    # per-symbol cost path reduces EXACTLY to the scalar path (the derived model is a strict
    # generalization). Fed into the frozen settle_week (Invariant N) it agrees bit-for-bit.
    m = _model()
    n = 4
    adv = np.full(n, ADV_MEDIAN)
    amihud = np.full(n, AMIHUD_MEDIAN)
    vec = m.friction_vector(adv, amihud, notional_leg=ANCHOR_NOTIONAL)
    assert np.allclose(vec, 0.0013, rtol=1e-12)

    close = np.full((30, n), 100.0)
    funding = np.zeros((30, n))
    listed = np.ones((30, n), dtype=bool)
    w = np.array([0.25, 0.25, -0.25, -0.25])
    res_vec, _ = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w, w_prev=np.zeros(n), friction=vec
    )
    res_scalar, _ = settle_week(
        close, funding, listed, d0=7, d1=14, w_target=w, w_prev=np.zeros(n), friction=0.0013
    )
    assert res_vec.cost == pytest.approx(res_scalar.cost, rel=1e-12)


def test_friction_vector_gates_unshortable_short_names_to_inf():
    # The vectorized N50 gate: an unshortable name flagged in the short leg gets inf in the array,
    # forcing settle_week to reject that allocation.
    m = _model()
    n = 3
    adv = np.full(n, ADV_MEDIAN)
    amihud = np.full(n, AMIHUD_MEDIAN)
    is_shortable = np.array([True, False, True])
    in_short_leg = np.array([False, True, False])
    vec = m.friction_vector(
        adv, amihud, notional_leg=ANCHOR_NOTIONAL, is_shortable=is_shortable,
        in_short_leg=in_short_leg,
    )
    assert math.isinf(vec[1])
    assert np.isfinite(vec[0]) and np.isfinite(vec[2])


def test_amihud_ratio_matches_hand_computation():
    # COST.md Section 5.3: Amihud_i = mean(|r|/$Vol) over valid days; non-positive/non-finite $Vol
    # dropped; a name with no valid day -> NaN (which impact_frac maps to the class fallback).
    returns = np.array([[0.02, 0.01], [-0.04, 0.02], [0.0, np.nan]])
    dvol = np.array([[1e6, 2e6], [1e6, 0.0], [5e5, 1e6]])  # a zero and a NaN-return cell dropped
    a = amihud_ratio(returns, dvol)
    # name 0: mean(0.02/1e6, 0.04/1e6, 0.0/5e5) = (2e-8 + 4e-8 + 0)/3
    assert a[0] == pytest.approx((0.02 / 1e6 + 0.04 / 1e6 + 0.0 / 5e5) / 3.0, rel=1e-12)
    # name 1: only day 0 valid (day1 $Vol=0 dropped, day2 return NaN dropped) -> 0.01/2e6
    assert a[1] == pytest.approx(0.01 / 2e6, rel=1e-12)


def test_amihud_ratio_all_invalid_is_nan():
    returns = np.array([[0.02], [0.03]])
    dvol = np.array([[0.0], [np.nan]])
    a = amihud_ratio(returns, dvol)
    assert np.isnan(a[0])
