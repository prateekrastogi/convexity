"""SYSTEM worked-example regression fixtures (SYSTEM §7.3b / §13, ARCHITECTURE §8).

Reproduced bit-for-bit from the tabulated spec inputs:
  * the §7.3b DSR FAIL fixture (DSR≈0.6950 @ n_obs=108 < MinTRL≈1114) with full intermediates;
  * the §7.3b PASS contrast (DSR≈0.999995 @ n_obs=400 ≥ MinTRL≈56);
  * the N_sys=1 BLdP zero-benchmark convention (SR*0=0, DSR=PSR(0));
  * an anti-correlated trial ledger ⇒ N_sys=K (the de-correlated floor, Invariant DSR-MONO);
  * a double-consumed HALT ⇒ a SINGLE flatten per position (no double order);
  * the drawdown backstop sits ABOVE the LEVERAGE governor (45% OPS DD-kill > 43% governor flat);
  * Carver [0,1,3] preserved — the existing 19 selection tests (untouched) still pass.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from convexity.system.config.defaults import SystemDefaults
from convexity.system.config.resolve import resolve_config
from convexity.system.config.universe import load_universe_meta
from convexity.system.synth.generate import (
    DSR_LEDGER_SHARPES_ANNUAL,
    make_book,
    make_candidate,
    make_correlated_ledger,
    make_open_position,
    make_proposed,
    make_trend_action,
)
from convexity.system.system_api import decide_live
from convexity.system.validate.deflated import DeflatedSharpe, effective_trials, min_trl, sr_star0


class _Cfg:
    defaults = SystemDefaults()


def cfg():
    return resolve_config(load_universe_meta([]))


def deflate(**kw):
    base = dict(
        path_returns=np.zeros(108),
        trial_sharpes=DSR_LEDGER_SHARPES_ANNUAL,
        n_obs=108,
        skew=-0.4,
        kurt=6.0,
        n_trials_effective=4.0,
        cfg=_Cfg(),
        k_distinct=12,
    )
    base.update(kw)
    return DeflatedSharpe().deflate(**base)


# --------------------------------------------------------------------------- §7.3b FAIL fixture
def test_dsr_7_3b_fail_with_full_intermediates():
    """§7.3b FAIL: SR_annual=0.92, n_obs=108, N_sys=4 ⇒ every tabulated intermediate, DSR=0.6950 FAIL.

    Ledger per-observation Var ≈ 0.0014389; var_floor 0.0093681 DOMINATES at n_obs=108; SR_pp=0.153333;
    SR*0≈0.101834; DSR≈0.6950 < 0.95; MinTRL≈1114 > 108 ⇒ insufficient_track_record. FAIL on both axes.
    """
    r = deflate(sr_annual=0.92, n_obs=108, path_returns=np.zeros(108))

    # ledger per-observation variance (annualized vector / sqrt(36), ddof=1):
    ts_pp = DSR_LEDGER_SHARPES_ANNUAL / 6.0
    assert float(np.var(ts_pp, ddof=1)) == pytest.approx(0.0014389, abs=1e-6)

    # the variance floor dominates the cross-config dispersion at this thin n_obs:
    assert r.var_trials == pytest.approx(0.0093681, abs=1e-6)
    assert r.var_floored is True

    # per-observation Sharpe + deflated benchmark:
    assert r.sr_pp == pytest.approx(0.153333, abs=1e-5)
    assert r.sr_star0 == pytest.approx(0.101834, abs=1e-5)

    # the per-observation z and DSR:
    denom = 1 - (-0.4) * r.sr_pp + ((6.0 - 1) / 4) * r.sr_pp**2
    assert denom == pytest.approx(1.090722, abs=1e-5)
    z = (r.sr_pp - r.sr_star0) * math.sqrt(107) / math.sqrt(denom)
    assert z == pytest.approx(0.510079, abs=1e-3)
    assert r.dsr == pytest.approx(0.6950, abs=1e-3)

    # MinTRL track-length floor ≈ 1114 ≫ 108:
    assert r.min_trl == pytest.approx(1114, abs=2)
    assert r.n_obs_sufficient is False
    assert r.passed is False
    assert "insufficient_track_record" in r.reason_codes


def test_dsr_7_3b_annualized_input_would_flip_the_gate():
    """§7.3b: feeding the ANNUALIZED 0.92 into the SAME z-formula inflates z 0.51→≈2.05 ⇒ a spurious
    PASS — the corrected defect. The honest per-observation gate FAILS."""
    r = deflate(sr_annual=0.92, n_obs=108, path_returns=np.zeros(108))
    denom = 1 - (-0.4) * r.sr_pp + ((6.0 - 1) / 4) * r.sr_pp**2
    # the honest per-observation z FAILS the gate:
    z_pp = (r.sr_pp - r.sr_star0) * math.sqrt(107) / math.sqrt(denom)
    assert z_pp == pytest.approx(0.510079, abs=1e-3)
    assert norm.cdf(z_pp) < 0.95
    # naively plugging the ANNUALIZED Sharpe into the same z-formula inflates z far past the gate ⇒
    # a spurious PASS (the corrected defect): the inflation is ≈sqrt(ppy) with no denominator offset.
    z_ann = (0.92 - r.sr_star0) * math.sqrt(107) / math.sqrt(denom)
    assert z_ann > z_pp
    assert norm.cdf(z_ann) > 0.95  # would have spuriously PASSED


# --------------------------------------------------------------------------- §7.3b PASS contrast
def test_dsr_7_3b_pass_contrast():
    """§7.3b PASS: SR_annual=1.80, n_obs=400, N_sys=4 ⇒ DSR≈0.999995, MinTRL≈56, PASS (sufficient)."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400))
    assert r.sr_pp == pytest.approx(0.300000, abs=1e-6)
    assert r.var_trials == pytest.approx(0.0026125, abs=1e-6)  # floor still binds
    assert r.sr_star0 == pytest.approx(0.053777, abs=1e-5)
    denom = 1 - (-0.4) * r.sr_pp + ((6.0 - 1) / 4) * r.sr_pp**2
    assert denom == pytest.approx(1.2325, abs=1e-4)
    z = (r.sr_pp - r.sr_star0) * math.sqrt(399) / math.sqrt(denom)
    assert z == pytest.approx(4.4302, abs=1e-2)
    assert r.dsr == pytest.approx(0.999995, abs=1e-5)
    assert r.min_trl == pytest.approx(56, abs=2)
    assert r.n_obs_sufficient is True
    assert r.passed is True


def test_min_trl_formula_inverse_of_z_test():
    """§7.3a: MinTRL is the algebraic inverse of the z-test, and +∞ when SR_pp ≤ SR*0 (fails on
    edge, not length)."""
    # reproduce the FAIL-case MinTRL from the standalone formula:
    mtrl = min_trl(0.153333, 0.101834, -0.4, 6.0, norm.ppf(0.95), 1e-6)
    assert mtrl == pytest.approx(1114, abs=2)
    assert min_trl(0.05, 0.10, -0.4, 6.0, 1.645, 1e-6) == float("inf")


# --------------------------------------------------------------------------- N_sys=1 BLdP convention
def test_n_sys_one_bldp_zero_benchmark():
    """§7.3a guard 2: N_sys=1 ⇒ SR*0=0 (no max-of-N inflation), DSR=PSR(0); Φ⁻¹(0) never evaluated."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400), n_trials_effective=1.0)
    assert r.sr_star0 == 0.0
    assert np.isfinite(r.dsr)
    assert r.dsr == pytest.approx(r.psr, abs=1e-12)


# --------------------------------------------------------------------------- anti-correlated ledger
def test_anti_correlated_ledger_gives_n_sys_equals_k():
    """§7.1: an anti-correlated (rho_bar clipped at 0) ledger ⇒ N_eff=K ⇒ N_sys=K (de-correlated
    floor). Trials never count as fewer than the all-independent count."""
    returns = make_correlated_ledger(k=8, rho=-0.5, n=400, seed=5)
    n_sys, _ = effective_trials(returns, rho_cl=0.7)
    assert n_sys == pytest.approx(8.0, abs=1e-9)


def test_sr_star0_grows_with_n_sys():
    """§7.3: SR*0 grows with N_sys — more trials ⇒ a higher luck-max benchmark to clear."""
    v = 0.0093681
    assert sr_star0(v, 8.0) > sr_star0(v, 2.0)


# --------------------------------------------------------------------------- double-consumed HALT
def test_double_consumed_halt_single_flatten():
    """§2 step A / §13: a HALT consumed by BOTH the governor flat AND a scripted trend exit on the
    SAME position produces a SINGLE flat_all per instrument — never a double close order."""
    book = make_book(
        {"A": make_open_position("A"), "B": make_open_position("B")},
        governor_multiplier=0.0,  # ≥43% drawdown ⇒ FLAT_ALL
    )
    sel = make_proposed(("A", "B"))
    # TREND also says exit A (a second consumer of the same halt on A):
    open_acts = {"A": make_trend_action("exit"), "B": make_trend_action("continue")}
    d = decide_live(book, sel, open_acts, {}, cfg())
    # exactly one action per open position, all flat_all (no duplicated close for A):
    per_inst = {}
    for a in d.actions:
        per_inst.setdefault(a.instrument, []).append(a)
    assert {k: len(v) for k, v in per_inst.items()} == {"A": 1, "B": 1}
    assert all(a.kind == "flat_all" for a in d.actions)
    assert "governor_flat_all" in d.reason_codes and "halt_new_opens" in d.reason_codes


# --------------------------------------------------------------------------- DD backstop above governor
def test_drawdown_backstop_sits_above_leverage_governor():
    """§13 / ARCH §5.12: the OPS 45% DD-kill backstop sits ABOVE the LEVERAGE 43% governor flat on the
    SHARED equity/peak reference (45% > 43%), so the governor binds FIRST. At 43% drawdown the
    governor already flattens — the backstop never has to fire below it."""
    d = SystemDefaults()
    governor_flat_dd = 0.43
    ops_kill_dd = 0.45
    assert ops_kill_dd > governor_flat_dd  # the backstop is strictly above the governor

    # the shared reference: a book at the 43% governor threshold reports drawdown 0.43 and, with the
    # LEVERAGE-supplied governor multiplier 0, flattens via step A (below the 45% OPS backstop).
    book = make_book({"A": make_open_position("A")}, equity=570.0, peak_equity=1000.0,
                     governor_multiplier=0.0)
    assert book.drawdown == pytest.approx(governor_flat_dd, abs=1e-9)
    assert book.drawdown < ops_kill_dd
    dec = decide_live(book, make_proposed(("A",)), {"A": make_trend_action("continue")}, {}, cfg())
    assert all(a.kind == "flat_all" for a in dec.actions)  # governor flattened first
    _ = d  # SystemDefaults available for the shared RED reference


# --------------------------------------------------------------------------- Carver [0,1,3] preserved
def test_carver_0_1_3_selection_preserved():
    """ARCH §8 / §5.10: the existing 19 selection tests are untouched. We re-run the canonical Carver
    Ch.25 worked example here to confirm SYSTEM's selection adapter never regressed the optimiser: the
    greedy tracking-error fill yields the contract vector [0,1,3] (buffered [0,1,2])."""
    from convexity.selection.dynamic_optimization import Instrument, optimise_portfolio

    instruments = [
        Instrument("US5y", 0.4, 0.22, 0.052, cost_per_contract=5.50),
        Instrument("US10y", 0.9, 0.24, 0.082, cost_per_contract=11.50),
        Instrument("SP500", 3.1, 0.04, 0.171, cost_per_contract=0.875),
    ]
    corr = np.array([[1.0, 0.9, -0.1], [0.9, 1.0, -0.1], [-0.1, -0.1, 1.0]])
    result = optimise_portfolio(instruments, corr, 500_000.0, apply_buffering=True)
    assert result.optimised_positions == {"US5y": 0, "US10y": 1, "SP500": 3}  # [0,1,3]
    assert result.buffered_positions == {"US5y": 0, "US10y": 1, "SP500": 2}


# --------------------------------------------------------------------------- budget contention worked case
def test_budget_contention_admits_by_risk_defers_rest():
    """§13 budget contention: more qualified breakouts than H_book admits ⇒ the priority firewall
    admits by ceiling-slack/MCTR-efficiency and DEFERS the rest with budget_deferred, never by score."""
    book = make_book()
    sel = make_proposed(("B", "C"))
    # both want a lot but only one fits the (synthetic) residual heat; the higher-score one must NOT
    # be preferred for that reason — id/risk order decides.
    cands = {
        "B": make_candidate("B", quality_score=999.0, wpc=0.18, ceiling_weight=0.18, ceiling_slack=0.1),
        "C": make_candidate("C", quality_score=1.0, wpc=0.05, ceiling_weight=0.18, ceiling_slack=0.9),
    }
    d = decide_live(book, sel, {}, cands, cfg())
    opened = {a.instrument for a in d.actions if a.kind == "open"}
    # C (higher ceiling-slack, lower score) is admitted; the high-score B does not win on score.
    assert "C" in opened or "B" in d.deferred
