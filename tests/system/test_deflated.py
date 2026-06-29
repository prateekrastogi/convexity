"""Deflated-Sharpe centerpiece tests (SYSTEM §7, Invariants DSR / DSR-GUARD / DSR-MINTRL /
DSR-MONO / DSR-ANN / SR-SRC).

The §7.3b worked examples are reproduced bit-for-bit: the realistic case FAILS (DSR≈0.6950,
n_obs=108 vs MinTRL≈1114) and the PASS contrast (DSR≈0.999995, n_obs=400 vs MinTRL≈56).
"""

import numpy as np
import pytest

from convexity.system.config.defaults import SystemDefaults
from convexity.system.synth.generate import (
    DSR_LEDGER_SHARPES_ANNUAL,
    make_correlated_ledger,
)
from convexity.system.validate.deflated import (
    DeflatedSharpe,
    effective_trials,
    min_trl,
    sr_star0,
)


class _Cfg:
    defaults = SystemDefaults()


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


# --------------------------------------------------------------------------- §7.3b worked examples


def test_worked_example_fail_case_7_3b():
    """§7.3b FAIL: SR_annual=0.92, n_obs=108 ⇒ DSR≈0.6950, SR*0≈0.101834, MinTRL≈1114, FAIL."""
    r = deflate(sr_annual=0.92, n_obs=108, path_returns=np.zeros(108))
    assert r.var_trials == pytest.approx(0.0093681, abs=1e-6)
    assert r.var_floored is True
    assert r.sr_pp == pytest.approx(0.153333, abs=1e-5)
    assert r.sr_star0 == pytest.approx(0.101834, abs=1e-5)
    assert r.dsr == pytest.approx(0.6950, abs=1e-3)
    assert r.min_trl == pytest.approx(1114, abs=2)
    assert r.n_obs_sufficient is False
    assert r.passed is False
    assert "insufficient_track_record" in r.reason_codes


def test_worked_example_pass_contrast_7_3b():
    """§7.3b PASS contrast: SR_annual=1.80, n_obs=400 ⇒ DSR≈0.999995, MinTRL≈56, PASS."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400))
    assert r.var_trials == pytest.approx(0.0026125, abs=1e-6)
    assert r.sr_star0 == pytest.approx(0.053777, abs=1e-5)
    assert r.dsr == pytest.approx(0.999995, abs=1e-5)
    assert r.min_trl == pytest.approx(56, abs=2)
    assert r.n_obs_sufficient is True
    assert r.passed is True


# --------------------------------------------------------------------------- singularity guards


def test_n_sys_one_uses_bldp_zero_benchmark():
    """Invariant DSR / guard 2: N_sys=1 ⇒ SR*0=0, DSR=PSR(0), never Φ⁻¹(0)."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400), n_trials_effective=1.0)
    assert r.sr_star0 == 0.0
    assert np.isfinite(r.dsr)
    assert r.dsr == pytest.approx(r.psr, abs=1e-12)  # DSR = PSR(0) when N_sys=1


def test_insufficient_trials_guard():
    """Invariant DSR-GUARD: < K_min distinct trials ⇒ insufficient_trials, not promotable."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400), k_distinct=3)
    assert r.passed is False
    assert "insufficient_trials" in r.reason_codes


def test_degenerate_variance_floored_still_bites():
    """Invariant DSR-GUARD: a near-identical ledger (Var→0) is floored so SR*0 > 0 still bites."""
    identical = np.full(12, 0.92)
    r = deflate(trial_sharpes=identical, sr_annual=0.92, n_obs=108, path_returns=np.zeros(108))
    assert r.var_floored is True
    assert r.sr_star0 > 0.0


def test_n_obs_floor_blocks_high_point_dsr():
    """Invariant DSR-MINTRL: a high point-DSR off a thin n_obs (< n_obs_floor) does NOT pass."""
    # tiny n_obs ⇒ even a strong SR fails the absolute floor.
    r = deflate(sr_annual=3.0, n_obs=12, path_returns=np.zeros(12))
    assert r.n_obs_sufficient is False
    assert r.passed is False
    assert "insufficient_track_record" in r.reason_codes


# --------------------------------------------------------------------------- annualization (DSR-ANN)


def test_annualized_input_rejected():
    """Invariant DSR-ANN: feeding the ANNUALIZED Sharpe into the z-test is rejected."""
    r = deflate(sr_annual=0.92, n_obs=108, assume_annualized_input=True)
    assert r.passed is False
    assert "annualization_mismatch" in r.reason_codes


def test_annualized_input_would_inflate_z():
    """Invariant DSR-ANN: had we fed the annualized SR, z inflates by ≈sqrt(ppy) — a spurious PASS.

    We confirm the per-observation gate FAILS (z≈0.51) while naively plugging the annualized SR into
    the same z-formula would have given ≈2.05 ⇒ DSR≈0.98 (the corrected defect).
    """
    r = deflate(sr_annual=0.92, n_obs=108, path_returns=np.zeros(108))
    # the honest per-observation z:
    import math

    from scipy.stats import norm

    denom = 1 - (-0.4) * r.sr_pp + ((6.0 - 1) / 4) * r.sr_pp**2
    z_pp = (r.sr_pp - r.sr_star0) * math.sqrt(107) / math.sqrt(denom)
    assert z_pp == pytest.approx(0.5101, abs=1e-3)
    # the spurious annualized z (defect): SR_annual=0.92 with the SAME denom.
    z_ann = (0.92 - r.sr_star0) * math.sqrt(107) / math.sqrt(denom)
    assert norm.cdf(z_ann) > 0.95  # would have spuriously PASSED


# --------------------------------------------------------------------------- N_sys monotonicity / floor


def test_n_sys_monotone_non_decreasing_in_rho_cl():
    """Invariant DSR-MONO: sweeping rho_cl loose→tight gives a non-decreasing N_sys."""
    returns = make_correlated_ledger(k=12, rho=0.5, n=300, seed=3)
    prev = 0.0
    for rho_cl in (0.3, 0.5, 0.7, 0.9):
        n_sys, _ = effective_trials(returns, rho_cl)
        assert n_sys >= prev - 1e-9
        prev = n_sys


def test_anti_correlated_ledger_gives_n_sys_equals_k():
    """Invariant DSR-MONO: an anti-correlated (rho_bar clipped at 0) ledger ⇒ N_sys = K."""
    returns = make_correlated_ledger(k=8, rho=-0.5, n=400, seed=5)
    n_sys, _ = effective_trials(returns, rho_cl=0.7)
    assert n_sys == pytest.approx(8.0, abs=1e-9)


def test_n_sys_bounded_one_to_k():
    """N_sys ∈ [1, K] always (Section 7.1)."""
    returns = make_correlated_ledger(k=10, rho=0.9, n=300, seed=9)
    n_sys, _ = effective_trials(returns, rho_cl=0.7)
    assert 1.0 <= n_sys <= 10.0


# --------------------------------------------------------------------------- formula spot-checks


def test_min_trl_infinite_when_edge_nonpositive():
    """MinTRL = +∞ when SR_pp ≤ SR*0 (fails on edge, not length)."""
    assert min_trl(0.05, 0.10, -0.4, 6.0, 1.645, 1e-6) == float("inf")


def test_sr_star0_grows_with_n_sys():
    """SR*0 grows with N_sys (more trials ⇒ higher luck-max benchmark)."""
    v = 0.0093681
    assert sr_star0(v, 8.0) > sr_star0(v, 2.0)
