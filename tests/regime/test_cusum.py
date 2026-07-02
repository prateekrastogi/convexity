"""CUSUM recursion + ARL closed-form tests (REGIME.md §3).

Reproduces the spec's OWN worked h* arithmetic bit-for-bit (§3.5 Steps 1-4) and pins the §3.2
derivation corrections (the in-control ARL is the theta->0 limit b^2 at mu_0+k, NOT a substitution
of theta=-k/sigma which gives ~3.9e6). Also asserts the Page-1954 recursion and the alarm-and-reset
rule, and Invariant ARL (empirical, not closed-form-asserted, ARL_0).
"""

import numpy as np
import pytest

from convexity.regime import cusum


# --------------------------------------------------------------------------- §3.1 recursion

def test_cusum_upper_matches_page_1954_recursion():
    """S_0=0; S_t = max(0, S_{t-1} + (x_t - mu_0 - k)) (REGIME.md §3.1)."""
    x = np.array([1.0, 1.0, 1.0, 1.0])
    s = cusum.cusum_upper(x, mu_0=0.0, k=0.5)
    # each step adds (1 - 0 - 0.5) = 0.5, never clamped: 0.5, 1.0, 1.5, 2.0
    np.testing.assert_allclose(s, [0.5, 1.0, 1.5, 2.0])


def test_cusum_upper_floors_at_zero():
    """The max(0, ...) floor discards evidence against a shift (REGIME.md §3.1 SPRT-restart)."""
    x = np.array([-5.0, -5.0, 3.0])
    s = cusum.cusum_upper(x, mu_0=0.0, k=0.0)
    # first two drive the sum negative -> clamped to 0; third lifts to 3.0
    np.testing.assert_allclose(s, [0.0, 0.0, 3.0])


def test_cusum_lower_detects_downward_shift():
    """S_t^- = max(0, S_{t-1}^- + (mu_0 - k - x_t)) catches dispersion COLLAPSE (REGIME.md §3.3/§8.2)."""
    x = np.array([-1.0, -1.0, -1.0])
    s = cusum.cusum_lower(x, mu_0=0.0, k=0.0)
    np.testing.assert_allclose(s, [1.0, 2.0, 3.0])


def test_upper_and_lower_are_anti_correlated_on_shared_input():
    """A high draw pushes S^+ up AND S^- toward its floor (REGIME.md §3.3 shared-input caveat)."""
    x = np.array([2.0, 2.0, 2.0])
    su = cusum.cusum_upper(x, mu_0=0.0, k=0.0)
    sl = cusum.cusum_lower(x, mu_0=0.0, k=0.0)
    assert su[-1] > 0 and sl[-1] == 0.0  # cannot both be elevated from the same statistic's noise


# --------------------------------------------------------------------------- §3.5 worked arithmetic

def test_worked_example_reference_value_k():
    """§3.5 Step 1: k = Delta_target * sigma / 2 = 1.5 * 0.0060 / 2 = 0.0045."""
    k = cusum.reference_value_k(delta_target=1.5, sigma=0.0060)
    assert k == pytest.approx(0.0045, abs=1e-12)


def test_worked_example_solve_h_from_arl0():
    """§3.5 Step 2: ARL_0=104, sigma=0.0060 -> h/sigma = 9.032 -> h = 0.05419."""
    h = cusum.solve_h_from_arl0(arl_0_target=104.0, sigma=0.0060)
    # sqrt(104) = 10.198..., minus 1.166 = 9.032..., times 0.0060
    assert h / 0.0060 == pytest.approx(9.032, abs=1e-3)
    assert h == pytest.approx(0.05419, abs=1e-5)


def test_worked_example_in_control_arl_is_b_squared_at_mu0_plus_k():
    """§3.2/§3.5: ARL(mu_0+k) = b^2 = (h/sigma+1.166)^2 = 10.198^2 = 103.99 ~ 104 (the theta->0 limit)."""
    h = 0.05419
    arl0 = cusum.arl_in_control(h, sigma=0.0060)
    assert arl0 == pytest.approx(103.99, abs=0.05)


def test_in_control_limit_is_not_the_theta_minus_k_over_sigma_substitution():
    """§3.2 correction: evaluating the general formula at theta=-k/sigma=-0.75 (i.e. at mu_0) gives
    ~3.9e6, NOT 104 — the two calculations answer different questions and are not interchangeable."""
    h, sigma, k = 0.05419, 0.0060, 0.0045
    wrong = cusum.arl_wald(theta=-k / sigma, h=h, sigma=sigma)  # at the ACTUAL in-control mean mu_0
    assert wrong > 1e6  # astronomically longer than 104 (large negative drift toward the 0 floor)
    right = cusum.arl_in_control(h, sigma)  # the b^2 zero-drift limit at mu_0+k
    assert right == pytest.approx(103.99, abs=0.05)
    assert wrong / right > 10_000  # they are wildly different — the conflation is a derivation bug


def test_worked_example_out_of_control_arl():
    """§3.5 Step 4: at mu_0+2k the drift theta = k/sigma = 0.75, giving ARL_1 ~ 12.71 weeks."""
    arl1 = cusum.arl_out_of_control(h=0.05419, sigma=0.0060, k=0.0045)
    assert arl1 == pytest.approx(12.71, abs=0.05)


def test_arl_wald_zero_drift_limit_matches_b_squared():
    """The theta->0 removable singularity resolves to b^2 (REGIME.md §3.2 L'Hopital)."""
    h, sigma = 0.05419, 0.0060
    b = h / sigma + cusum.SIEGMUND_CORRECTION
    near_zero = cusum.arl_wald(theta=1e-7, h=h, sigma=sigma)
    assert near_zero == pytest.approx(b * b, rel=1e-4)


# --------------------------------------------------------------------------- §3.1 alarm rule

def test_run_length_to_alarm_first_crossing():
    """First S_t >= h is the alarm week (REGIME.md §3.1); right-censored if never crossed."""
    s = np.array([0.1, 0.2, 0.6, 0.3])
    assert cusum.run_length_to_alarm(s, h=0.5) == 3
    assert cusum.run_length_to_alarm(np.array([0.1, 0.2]), h=0.5) == 2  # censored -> path length


# --------------------------------------------------------------------------- Invariant ARL (§3.5 Step 5)

def test_empirical_arl0_is_reported_not_closed_form():
    """Invariant ARL: the delivered ARL_0 is the EMPIRICAL block-bootstrap value on the in-control
    series, not the closed-form b^2. Perturbing the in-control window's autocorrelation changes the
    empirical estimate while the closed form (a pure function of h/sigma) does not move."""
    rng = np.random.default_rng(0)
    iid = rng.normal(0.0420, 0.0060, size=260)
    cal_iid = cusum.calibrate(
        iid, delta_target=1.5, arl_0_target=104.0,
        n_boot=300, block_length=6, seed=1,
    )
    # AR(1)-injected variant (dispersion clustering) — same marginal-ish scale, different dependence.
    ar = np.empty(260)
    ar[0] = 0.0420
    e = rng.normal(0, 0.0060 * np.sqrt(1 - 0.6**2), size=260)
    for i in range(1, 260):
        ar[i] = 0.0420 + 0.6 * (ar[i - 1] - 0.0420) + e[i]
    cal_ar = cusum.calibrate(
        ar, delta_target=1.5, arl_0_target=104.0,
        n_boot=300, block_length=6, seed=1,
    )
    assert cal_iid.arl0_empirical is not None and cal_ar.arl0_empirical is not None
    # empirical values differ under different autocorrelation (Invariant ARL test contract) ...
    assert cal_iid.arl0_empirical != cal_ar.arl0_empirical
    # ... and the empirical number is NOT just the closed-form b^2 echoed back.
    assert cal_iid.arl0_empirical != pytest.approx(cal_iid.arl0_closed_form, rel=1e-6)


def test_calibrate_fits_moments_and_derives_k_h():
    """calibrate fits TEAL mu_0/sigma on the in-control series and derives k,h (REGIME.md §3.4)."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.05, 0.008, size=200)
    cal = cusum.calibrate(x, delta_target=1.5, arl_0_target=104.0, n_boot=100, block_length=6, seed=3)
    assert cal.mu_0 == pytest.approx(float(np.mean(x)), rel=1e-9)
    assert cal.sigma == pytest.approx(float(np.std(x, ddof=1)), rel=1e-9)
    assert cal.k == pytest.approx(1.5 * cal.sigma / 2, rel=1e-9)


# ------------------------------------------------ Invariant ARL — the TOLERANCE-BAND gate (§3.5 Step 5)
# The empirical-ARL check has TWO clauses: (i) the delivered number is empirical (tested above) AND
# (ii) it must LAND WITHIN the RED tolerance band to ship. These tests bind clause (ii): they would
# all PASS if the tolerance gate were absent (an ungated calibration would happily report a
# way-out-of-band ARL_0), so their assertions on .within_tolerance/.ships/require_ships can only hold
# because the gate exists.

def _cal(arl0_emp, *, arl0_target=104.0, tol=0.20):
    """A hand-built ARLCalibration with a fixed empirical ARL_0 — isolates the gate arithmetic from
    bootstrap noise (the empirical estimator itself is tested separately)."""
    return cusum.ARLCalibration(
        mu_0=0.042, sigma=0.006, k=0.0045, h=0.05419,
        arl0_closed_form=103.99, arl0_empirical=arl0_emp, arl1=12.71,
        arl0_target=arl0_target, arl_tol_frac=tol,
    )


def test_within_tolerance_true_inside_band_false_outside():
    """Clause (ii): empirical ARL_0 inside arl0_target +/- tol*arl0_target ships; outside does not.
    Band for target=104, tol=0.20 is [83.2, 124.8]."""
    assert _cal(104.0).within_tolerance is True     # dead on target
    assert _cal(90.0).within_tolerance is True       # inside band
    assert _cal(124.8).within_tolerance is True      # on the upper edge (<=)
    assert _cal(83.2).within_tolerance is True        # on the lower edge (<=)
    assert _cal(60.0).within_tolerance is False       # too FEW weeks between false alarms -> reject
    assert _cal(300.0).within_tolerance is False       # too sluggish -> also rejected (two-sided band)


def test_ships_requires_both_empirical_present_and_in_band():
    """Invariant ARL gate: .ships is True only when an EMPIRICAL estimate exists AND it is in band."""
    assert _cal(104.0).ships is True
    assert _cal(60.0).ships is False                              # in-band clause fails
    # empirical estimate missing -> cannot ship (clause i) even though it is "not out of band"
    no_emp = cusum.ARLCalibration(
        mu_0=0.042, sigma=0.006, k=0.0045, h=0.05419, arl0_closed_form=103.99,
        arl0_empirical=None, arl0_target=104.0, arl_tol_frac=0.20,
    )
    assert no_emp.ships is False and no_emp.within_tolerance is False
    # tolerance metadata missing -> ungated -> conservatively NOT shippable (§0 P6)
    ungated = cusum.ARLCalibration(
        mu_0=0.042, sigma=0.006, k=0.0045, h=0.05419, arl0_closed_form=103.99,
        arl0_empirical=104.0,
    )
    assert ungated.ships is False


def test_calibrate_require_ships_raises_when_empirical_arl0_out_of_band():
    """Invariant ARL (the 'ships only within tolerance' half): with require_ships=True a calibration
    whose EMPIRICAL ARL_0 misses the band raises — no (k, h) goes live out of tolerance. Here an iid
    N(0.042, 0.006) window with target=104 but an absurdly TIGHT tolerance (0.1%) cannot land in band,
    so the strict gate must reject it. Without the gate this call would silently return a
    non-shippable calibration."""
    rng = np.random.default_rng(7)
    iid = rng.normal(0.042, 0.006, size=260)
    with pytest.raises(ValueError, match="Invariant ARL"):
        cusum.calibrate(
            iid, delta_target=1.5, arl_0_target=104.0,
            n_boot=200, block_length=6, seed=1,
            arl_tol_frac=0.001, require_ships=True,
        )


def test_calibrate_carries_tolerance_metadata_for_the_gate():
    """calibrate threads arl_0_target/arl_tol_frac onto the record so .ships is decidable (§3.5)."""
    rng = np.random.default_rng(8)
    iid = rng.normal(0.042, 0.006, size=260)
    cal = cusum.calibrate(
        iid, delta_target=1.5, arl_0_target=104.0,
        n_boot=200, block_length=6, seed=1, arl_tol_frac=0.20,
    )
    assert cal.arl0_target == 104.0 and cal.arl_tol_frac == 0.20
    # .ships is now a decidable function of the delivered empirical estimate (not always True/False).
    assert cal.ships == (abs(cal.arl0_empirical - 104.0) <= 0.20 * 104.0)
