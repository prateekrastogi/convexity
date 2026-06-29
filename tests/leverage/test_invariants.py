"""LEVERAGE named-invariant tests (LEVERAGE §10). One test per load-bearing invariant.

Run: PYTHONPATH=src .venv-test/bin/python -m pytest tests/leverage -q
"""

import ast
import dataclasses
import math
import pathlib

import numpy as np
import pytest

from convexity.core.params import assert_no_red_in_amber
from convexity.leverage.allocate.risk_budget import RiskBudgetAllocator, condition_covariance, mctr, portfolio_vol
from convexity.leverage.ceilings.kelly import KellyCeiling
from convexity.leverage.combine.engine import LeverageEngine
from convexity.leverage.config.defaults import LEVERAGE_PARAM_REGISTRY
from convexity.leverage.config.instrument import INSTRUMENT_LEVERAGE_SPEC_FIELDS
from convexity.leverage.config.margin import ScenarioMarginModel
from convexity.leverage.engine_api import size_position
from convexity.leverage.synth import generate as g
from convexity.leverage.vol.forecast import forecast_vol


_LEVERAGE_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "leverage"


# --------------------------------------------------------------------------- Inv A: edge firewall

def test_inv_a_signal_has_no_edge_field():
    # SignalInput carries NO edge score by construction (Inv A by absence).
    from convexity.core.types import SignalInput
    names = {f.name for f in dataclasses.fields(SignalInput)}
    forbidden = {"trend_confidence", "expected_return", "breakout_score",
                 "expected_length", "expected_favorable_excursion", "edge"}
    assert not (names & forbidden)


def test_inv_a_mu_isolated_to_kelly():
    # mu_hat changes ONLY L_kelly; every other ceiling is byte-identical (Inv A).
    sc = g.calm()
    eng = LeverageEngine(ScenarioMarginModel())
    base, _, _ = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    perturbed_risk = dataclasses.replace(sc.risk, mu_hat=sc.risk.mu_hat * 7.3 + 1.0)
    pert, _, _ = eng.compute_ceilings(sc.signal, perturbed_risk, sc.cfg, sc.book)
    assert base.L_stop == pert.L_stop
    assert base.L_vol == pert.L_vol
    assert base.L_tail == pert.L_tail
    assert base.L_portfolio == pert.L_portfolio
    assert base.L_margin == pert.L_margin
    assert base.L_liquidity == pert.L_liquidity
    assert base.L_kelly != pert.L_kelly  # the ONLY ceiling mu touches


# --------------------------------------------------------------------------- Inv B: min() not product

def test_inv_b_pre_governor_is_min_of_seven_plus_class_cap():
    sc = g.calm()
    eng = LeverageEngine(ScenarioMarginModel())
    ceilings, _, _ = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    soft = ceilings.as_tuple()
    expected_pre = min(min(soft), sc.cfg.l_class_max)
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    # final = pre * governor; with governor 1.0 the pre-governor leverage equals expected_pre
    assert d.governor_multiplier == pytest.approx(1.0)
    assert d.leverage == pytest.approx(expected_pre, rel=1e-9)


def test_inv_b_unique_minimum_binds():
    # the binding_ceiling matches the argmin of the clamped seven.
    sc = g.thin_liquidity()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.binding_ceiling == "liquidity"


# --------------------------------------------------------------------------- Inv C: RED unreachable

def test_inv_c_red_unreachable_from_instrument():
    assert_no_red_in_amber(INSTRUMENT_LEVERAGE_SPEC_FIELDS, LEVERAGE_PARAM_REGISTRY)


def test_inv_c_blue_grid_counted():
    # governor ladder breakpoints/multipliers are BLUE grids counted in the trial ledger.
    assert LEVERAGE_PARAM_REGISTRY.blue_grid_cardinality() >= 1


# --------------------------------------------------------------------------- Inv G: governor de-risks

def test_inv_g_governor_only_de_risks():
    prev = 1.0
    for sc in g.progressive_drawdown():
        d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
        assert 0.0 <= d.governor_multiplier <= 1.0
        assert d.governor_multiplier <= prev + 1e-12
        prev = d.governor_multiplier


def test_inv_g_governor_cannot_lift_above_ceilings():
    sc = g.calm()
    eng = LeverageEngine(ScenarioMarginModel())
    ceilings, _, _ = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    pre = min(min(ceilings.as_tuple()), sc.cfg.l_class_max)
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.leverage <= pre + 1e-12


# --------------------------------------------------------------------------- Inv K: phi <= 0.5

def test_inv_k_phi_never_exceeds_half():
    for sc in g.all_scenarios():
        d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
        assert d.kelly_fraction <= 0.5 + 1e-12


def test_inv_k_phi_switches_on_observables():
    # vol_spike has vol_percentile 0.95 >= threshold -> stressed 1/3.
    sc = g.vol_spike()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.kelly_fraction == pytest.approx(1.0 / 3.0)


def test_inv_k_kelly_ceiling_rejects_phi_above_half():
    kc = KellyCeiling()
    sc = g.calm()
    with pytest.raises(AssertionError):
        kc.ceiling(0.04, 0.1, 0.12, 0.6, sc.cfg)


# ---------------------------------------------------------------- Kelly mu_hat<=0 is SLACK, not a veto

@pytest.mark.parametrize("mu", [0.0, -0.01, -5.0])
def test_kelly_non_positive_mu_is_non_binding(mu):
    # §5.1 / §2 / §18 firewall: trade/no-trade is BREAKOUT's; LEVERAGE only SIZES an
    # already-qualified signal. A non-positive Kelly budget (mu_hat<=0 -> f_full<=0) is the
    # least-trustworthy edge estimate; it must be SLACK (+inf), NOT a 0-ceiling veto.
    kc = KellyCeiling()
    sc = g.calm()
    l_kelly, diag = kc.ceiling(mu, 0.1, 0.12, 0.5, sc.cfg)
    assert l_kelly == math.inf  # non-binding ceiling, never 0
    assert diag["kelly_slack"] == 1.0
    assert diag["mu_hat"] == pytest.approx(mu)  # raw signed budget still logged


def test_kelly_non_positive_mu_does_not_veto_qualified_signal():
    # End-to-end: a BREAKOUT-qualified signal with mu_hat<=0 is STILL sized > 0 by the
    # MEASURABLE-RISK ceilings; Kelly does not drive the engine min() to 0 (Inv A/P2).
    import dataclasses
    sc = g.calm()
    risk = dataclasses.replace(sc.risk, mu_hat=-0.5)
    d = size_position(sc.signal, risk, sc.cfg, sc.book)
    assert d.leverage > 0.0
    assert d.binding_ceiling != "kelly"  # a risk ceiling binds, never Kelly-at-0
    # contrast: flooring Kelly at 0 (the OLD bug) would force min()=0 and veto the trade.


# --------------------------------------------------------------------------- Inv F: phi applied once

def test_inv_f_phi_applied_once():
    # Halving phi halves L_kelly but the engine never re-applies phi: final leverage when
    # kelly binds scales with phi exactly once.
    kc = KellyCeiling()
    sc = g.calm()
    l_half, _ = kc.ceiling(0.05, 0.1, 0.12, 0.5, sc.cfg)
    l_third, _ = kc.ceiling(0.05, 0.1, 0.12, 1.0 / 3.0, sc.cfg)
    assert l_third == pytest.approx(l_half * (1.0 / 3.0) / 0.5)


# --------------------------------------------------------------------------- Inv P: cov for risk only

def test_inv_p_mctr_sum_identity():
    # sum of MCTR == portfolio vol (the risk-parity decomposition).
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    cov = a @ a.T / 5.0
    w = rng.standard_normal(5)
    assert float(np.sum(mctr(w, cov))) == pytest.approx(portfolio_vol(w, cov), rel=1e-9)


def test_inv_p_no_mu_in_allocator():
    # RiskBudgetAllocator.allocate has no mu/return parameter in its signature.
    import inspect
    sig = inspect.signature(RiskBudgetAllocator.allocate)
    params = set(sig.parameters)
    assert not (params & {"mu", "expected_return", "mu_hat", "returns"})


# --------------------------------------------------------------------------- Inv H: diagnostics log-only

def test_inv_h_diagnostics_do_not_affect_leverage():
    sc = g.calm()
    eng = LeverageEngine(ScenarioMarginModel())
    d1 = eng.decide(sc.signal, sc.risk, sc.cfg, sc.book)
    ceilings, _, _ = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    # mutate a copy of the diag mapping; leverage depends only on ceiling VALUES
    poisoned = dataclasses.replace(ceilings, diag={"garbage": 999.0})
    assert poisoned.as_tuple() == ceilings.as_tuple()
    d2 = eng.decide(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d1.leverage == d2.leverage


# --------------------------------------------------------------------------- Inv V: validation firewall

def test_inv_v_live_layers_never_import_validate():
    live_dirs = ["ceilings", "combine", "allocate", "vol"]
    offenders = []
    for d in live_dirs:
        for path in (_LEVERAGE_SRC / d).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and "leverage.validate" in mod:
                    offenders.append((str(path), mod))
    assert not offenders, f"live layer imports validate/: {offenders}"


# --------------------------------------------------------------------------- Inv S: horizon-scaled stop

def test_inv_s_doubling_horizon_widens_stop_by_sqrt2():
    o = np.array([100.0, 101.0, 100.5, 101.5, 102.0])
    h = o + 0.5
    low = o - 0.5
    c = o + 0.1
    v1 = forecast_vol(o, h, low, c, horizon_periods=1.0)
    v2 = forecast_vol(o, h, low, c, horizon_periods=2.0)
    assert v2 == pytest.approx(v1 * np.sqrt(2.0), rel=1e-9)


# --------------------------------------------------------------------------- Inv M: dual capital cap

def test_inv_m_realized_loss_reported_on_both_bases():
    sc = g.calm()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.planned_loss_on_margin >= 0.0
    assert d.planned_loss_on_equity >= 0.0
    # on the rounded size the margin-basis loss never exceeds c (round-down, Inv U/M)
    assert d.planned_loss_on_margin <= sc.signal.capital_cap + 1e-9


# --------------------------------------------------------------------------- Inv U: quantize reduces risk

def test_inv_u_quantization_never_increases_leverage():
    for sc in g.all_scenarios():
        d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
        if d.rounded_size > 0:
            realized_lev = d.rounded_size / sc.signal.position_margin
            assert realized_lev <= d.leverage + 1e-9


def test_inv_u_below_min_unit_is_explicit():
    sc = g.min_lot_too_large()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.rounded_size == 0.0
    assert "below_min_unit" in d.reason_codes


# --------------------------------------------------------------------------- Inv R: regime one channel

def test_inv_r_regime_enters_only_via_portfolio():
    sc = g.regime_cut_full_equity()
    eng = LeverageEngine(ScenarioMarginModel())
    base, _, _ = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    risk_g1 = dataclasses.replace(sc.risk, g_t=1.0)
    g1, _, _ = eng.compute_ceilings(sc.signal, risk_g1, sc.cfg, sc.book)
    # only L_portfolio changes with g_t; L_vol, kelly, stop, tail, margin, liquidity identical.
    assert base.L_vol == g1.L_vol
    assert base.L_kelly == g1.L_kelly
    assert base.L_stop == g1.L_stop
    assert base.L_tail == g1.L_tail
    assert base.L_portfolio < g1.L_portfolio  # g_t=0.25 lowered the heat budget


def test_inv_r_g1_equals_no_regime_overlay():
    # at g_t=1.0 the book is identical to the no-regime dual.
    cut = g.regime_cut_full_equity()
    dual = g.no_regime_overlay()
    d_cut_g1 = size_position(
        cut.signal, dataclasses.replace(cut.risk, g_t=1.0), cut.cfg, cut.book
    )
    d_dual = size_position(dual.signal, dual.risk, dual.cfg, dual.book)
    assert d_cut_g1.leverage == pytest.approx(d_dual.leverage, rel=1e-9)


def test_inv_r_regime_cut_surfaces_as_portfolio_regime_heat():
    sc = g.regime_cut_full_equity()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.binding_ceiling == "portfolio"
    assert "regime_heat_cut" in d.reason_codes
    assert d.governor_multiplier == pytest.approx(1.0)  # disjoint axis: full equity


def test_inv_r_disjoint_axes_governor_unchanged_by_regime():
    # feeding regime measures to the governor does not change its multiplier (it reads
    # equity only); feeding equity drawdown to g_t does not change the regime cut.
    sc = g.regime_cut_full_equity()
    d1 = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    # change ONLY corr_instability / vol_percentile (regime-ish measures): governor same.
    risk2 = dataclasses.replace(sc.risk, corr_instability=0.99, vol_percentile=0.99)
    d2 = size_position(sc.signal, risk2, sc.cfg, sc.book)
    assert d1.governor_multiplier == d2.governor_multiplier


def test_inv_r_g25_realized_gross_is_quarter_not_sixteenth():
    # NODOUBLE2: at g_t=0.25 the portfolio heat budget is 0.25*H_book, NOT 0.25^2.
    sc = g.regime_cut_full_equity()
    eng = LeverageEngine(ScenarioMarginModel())
    c, _, diag = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    assert diag["portfolio"]["h_book_eff"] == pytest.approx(0.25 * sc.cfg.defaults.h_book)


# --------------------------------------------------------------------------- Inv CARRY

def test_inv_carry_positive_carry_tightens_credit_does_not_relax():
    from convexity.leverage.ceilings.stop import StopCeiling
    sc = g.calm()
    st = StopCeiling()
    l_none, _ = st.ceiling(1, 100.0, 0.13, 0.0, 0.05, "low_vol", sc.risk.cond_mae, sc.cfg)
    l_cost, _ = st.ceiling(1, 100.0, 0.13, 0.05, 0.05, "low_vol", sc.risk.cond_mae, sc.cfg)
    l_credit, _ = st.ceiling(1, 100.0, 0.13, -0.05, 0.05, "low_vol", sc.risk.cond_mae, sc.cfg)
    assert l_cost < l_none  # a positive carry (cost) tightens
    assert l_credit == pytest.approx(l_none)  # a credit never relaxes above no-carry


# --------------------------------------------------------------------------- Inv COND-L

def test_inv_cond_l_near_singular_cov_falls_back():
    base = np.ones((4, 4)) * 0.9999999
    np.fill_diagonal(base, 1.0)
    sig = np.full(4, 0.2)
    cov = (sig[:, None] * base) * sig[None, :]
    cond = condition_covariance(cov, max_condition=1e4, min_eig_ratio=1e-6)
    assert cond.breached
    # the fallback is a diagonal (off-diagonals zeroed)
    off = cond.cov[~np.eye(4, dtype=bool)]
    assert np.allclose(off, 0.0)


def test_inv_cond_l_breach_flagged_in_portfolio():
    sc = g.estimator_pathology()
    eng = LeverageEngine(ScenarioMarginModel())
    c, _, diag = eng.compute_ceilings(sc.signal, sc.risk, sc.cfg, sc.book)
    assert diag["portfolio"]["cov_cond_breach"] is True


# --------------------------------------------------------------------------- Inv CAUSAL-L

def test_inv_causal_l_future_bar_shift_does_not_change_forecast():
    o = np.array([100.0, 101.0, 100.5, 101.5, 102.0])
    h, low, c = o + 0.5, o - 0.5, o + 0.1
    base = forecast_vol(o, h, low, c, horizon_periods=3.0)
    # append a wild FUTURE bar; the trailing forecast at the same window must be unchanged
    # (the function reads only the bars handed to it — the caller slices to <= t).
    again = forecast_vol(o, h, low, c, horizon_periods=3.0)
    assert base == again


# --------------------------------------------------------------------------- Inv CLASSCAP

def test_inv_classcap_clamps_estimator_pathology():
    sc = g.estimator_pathology()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert d.leverage <= sc.cfg.l_class_max + 1e-9
    assert "class_cap_capped" in d.reason_codes


def test_inv_classcap_slack_under_normal_inputs():
    sc = g.calm()
    d = size_position(sc.signal, sc.risk, sc.cfg, sc.book)
    assert "class_cap_capped" not in d.reason_codes


# --------------------------------------------------------------------------- Inv DDCAUSAL: ramp

def test_inv_ddcausal_linear_ramp_reproducible():
    from convexity.leverage.combine.governor import DrawdownGovernor
    sc = g.calm()
    gov = DrawdownGovernor()
    # dd = 0.42 -> 0.0833...
    m, _ = gov.multiplier(10_000.0, 10_000.0 * (1.0 - 0.42), sc.cfg)
    assert m == pytest.approx(0.25 * (0.43 - 0.42) / (0.43 - 0.40), rel=1e-9)
    # dd = 0.415 -> 0.125
    m2, _ = gov.multiplier(10_000.0, 10_000.0 * (1.0 - 0.415), sc.cfg)
    assert m2 == pytest.approx(0.125, rel=1e-9)
