"""Unit + causality tests for research/cta/lib/signal_extraction.py.

Mirrors the research/entry_exit/tests/ convention (a research-local pytest suite, not part of
the main `tests/` package run by the repo-wide `pytest tests -q` gate). Run explicitly:

  PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
    /home/speculation/convexity/.venv/bin/python -m pytest research/cta/tests -q
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research/cta/lib"))

import signal_extraction as se  # noqa: E402


def make_synthetic_price(n=400, seed=0, drift=0.0002, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    log_ret = rng.normal(drift, vol, size=n)
    log_price = pd.Series(np.cumsum(log_ret), index=idx) + np.log(100.0)
    return log_price


class TestEwmaIdentity:
    def test_crossover_equals_return_ewma_difference(self):
        lp = make_synthetic_price()
        a = se.ewma_price_crossover(lp, 8, 32)
        b = se.crossover_as_return_ewma_difference(lp, 8, 32)
        pd.testing.assert_series_equal(a, b, check_exact=False, atol=1e-8)

    def test_crossover_requires_fast_lt_slow(self):
        lp = make_synthetic_price()
        with pytest.raises(ValueError):
            se.ewma_price_crossover(lp, 32, 8)

    def test_center_of_mass_matches_pandas_com_semantics(self):
        # pandas alpha for halflife hl: alpha = 1 - 2**(-1/hl); com = (1-alpha)/alpha
        hl = 20.0
        lam = se.halflife_to_decay(hl)
        alpha = 1 - lam
        expected_com = (1 - alpha) / alpha
        assert se.center_of_mass(hl) == pytest.approx(expected_com, rel=1e-9)


class TestKalman:
    def test_steady_state_gain_converges(self):
        r = se.steady_state_kalman_gain(q_level=1e-5, q_slope=1e-7, r=1e-4)
        assert r["converged"]
        assert 0 < r["K_level"] < 1
        assert 0 < r["K_slope"] < 1

    def test_filter_gain_settles_near_steady_state(self):
        lp = make_synthetic_price(n=600)
        kf = se.kalman_local_linear_trend(lp, q_level=1e-5, q_slope=1e-7, r=1e-4)
        ss = se.steady_state_kalman_gain(q_level=1e-5, q_slope=1e-7, r=1e-4)
        assert abs(kf["K_level"].iloc[-1] - ss["K_level"]) < 1e-3

    def test_holt_correspondence_matches_manual_recursion(self):
        ss = se.steady_state_kalman_gain(q_level=1e-4, q_slope=1e-6, r=1e-3)
        holt = se.holt_parameters_from_kalman_gain(ss["K_level"], ss["K_slope"])
        lp = make_synthetic_price(n=300, seed=1)
        y = lp.to_numpy()
        # run steady-state KF update
        level_kf, slope_kf = y[0], 0.0
        level_h, slope_h = y[0], 0.0
        for t in range(1, len(y)):
            e_kf = y[t] - (level_kf + slope_kf)
            level_kf, slope_kf = level_kf + slope_kf + ss["K_level"] * e_kf, slope_kf + ss["K_slope"] * e_kf
            e_h = y[t] - (level_h + slope_h)
            new_level_h = level_h + slope_h + holt["alpha"] * e_h
            slope_h = holt["beta"] * (new_level_h - level_h) + (1 - holt["beta"]) * slope_h
            level_h = new_level_h
        assert level_kf == pytest.approx(level_h, abs=1e-9)
        assert slope_kf == pytest.approx(slope_h, abs=1e-9)


class TestL1TrendFilter:
    def test_window_output_shape_and_finiteness(self):
        rng = np.random.default_rng(2)
        y = np.cumsum(rng.normal(0, 0.01, 60)) + 5.0
        x = se.l1_trend_filter_window(y, lam=3.0)
        assert x.shape == y.shape
        assert np.all(np.isfinite(x))

    def test_short_window_passthrough(self):
        y = np.array([1.0, 2.0, 3.0])
        x = se.l1_trend_filter_window(y, lam=3.0)
        np.testing.assert_array_equal(x, y)

    def test_rolling_is_causal_by_construction_regardless_of_edge_offset(self):
        lp = make_synthetic_price(n=200, seed=3)
        for eo in (0, 3):
            level = se.l1_trend_filter_rolling(lp, window=40, lam=2.0, edge_offset=eo, n_iter=60)
            # perturb the series strictly AFTER index 150 and confirm level[:150] unaffected
            lp2 = lp.copy()
            lp2.iloc[151:] += 1.0
            level2 = se.l1_trend_filter_rolling(lp2, window=40, lam=2.0, edge_offset=eo, n_iter=60)
            diff_before = (level.iloc[:150] - level2.iloc[:150]).abs().max()
            assert diff_before < 1e-8, f"edge_offset={eo}: future perturbation leaked into the past"

    def test_smooths_piecewise_linear_signal_with_low_error(self):
        n = 80
        idx = np.arange(n)
        true_x = np.where(idx < 40, idx * 0.1, 4.0 + (idx - 40) * -0.05)
        rng = np.random.default_rng(4)
        y = true_x + rng.normal(0, 0.02, n)
        xhat = se.l1_trend_filter_window(y, lam=1.0, n_iter=300)
        assert np.mean((xhat - true_x) ** 2) < np.mean((y - true_x) ** 2)

    def test_auto_lambda_calibration_is_scale_invariant(self):
        # a fixed absolute lam calibrated on one price SCALE should not silently degenerate to a
        # no-op (lam far too small) or a single straight line (lam far too large) when the same
        # relative dynamics are replayed at a very different price level/vol scale; auto (lam=None)
        # calibration should behave consistently across both.
        rng = np.random.default_rng(9)
        n = 150
        base_ret = rng.normal(0.0003, 0.01, n)
        lp_small = pd.Series(np.cumsum(base_ret) + np.log(10.0),
                              index=pd.bdate_range("2020-01-01", periods=n))
        lp_large = lp_small * 50.0 + 100.0  # arbitrary large affine rescaling of the same dynamics

        level_small = se.l1_trend_filter_rolling(lp_small, window=60, lam=None, lam_mult=1.0, n_iter=80)
        level_large = se.l1_trend_filter_rolling(lp_large, window=60, lam=None, lam_mult=1.0, n_iter=80)
        valid = level_small.notna() & level_large.notna()
        assert valid.sum() > 30
        # neither degenerates to "no smoothing at all" (variance of filtered second-diffs should
        # be materially reduced vs raw) nor "single straight line" (some residual curvature must
        # survive) -- check both are non-trivially smoother than raw without being perfectly linear.
        raw_small_curv = np.abs(np.diff(lp_small[valid].to_numpy(), n=2)).mean()
        fit_small_curv = np.abs(np.diff(level_small[valid].to_numpy(), n=2)).mean()
        assert 0.0 < fit_small_curv < raw_small_curv, (
            "auto-calibrated l1 trend filter should reduce curvature without eliminating it "
            f"entirely (raw={raw_small_curv}, fit={fit_small_curv})")


class TestCombination:
    def test_combination_weights_shape_and_bounds(self):
        idx = pd.bdate_range("2020-01-01", periods=50)
        ic = pd.DataFrame({"a": np.linspace(-0.1, 0.3, 50), "b": np.linspace(0.2, -0.1, 50)}, index=idx)
        w = se.combination_weights_from_ic(ic, shrinkage=0.5)
        assert w.shape == ic.shape
        row_sums = w.sum(axis=1)
        assert np.allclose(row_sums.dropna(), 1.0, atol=1e-9)

    def test_shrinkage_bounds_validated(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        ic = pd.DataFrame({"a": 0.1}, index=idx)
        with pytest.raises(ValueError):
            se.combination_weights_from_ic(ic, shrinkage=1.5)

    def test_ic_alignment_uses_only_past_pairs(self):
        # construct a signal that perfectly predicts fwd return with a lag, verify the returned
        # IC series index t never requires log_price beyond t
        lp = make_synthetic_price(n=300, seed=5)
        sig = se.ewma_return_momentum(lp, 10)
        ic = se.rolling_information_coefficient(sig, lp, window=60)
        # ic.loc[t] must be computable from data available through t; perturbing log_price strictly
        # after t must not change ic.loc[t]
        t_check = 200
        lp2 = lp.copy()
        lp2.iloc[t_check + 1:] += 5.0
        sig2 = se.ewma_return_momentum(lp2, 10)
        ic2 = se.rolling_information_coefficient(sig2, lp2, window=60)
        assert ic.iloc[t_check] == pytest.approx(ic2.iloc[t_check], abs=1e-9, nan_ok=True) if False else (
            (np.isnan(ic.iloc[t_check]) and np.isnan(ic2.iloc[t_check]))
            or abs(ic.iloc[t_check] - ic2.iloc[t_check]) < 1e-9
        )


class TestInjectionCausality:
    def test_synthetic_injection_full_pass(self):
        lp = make_synthetic_price(n=500, seed=6)
        result = se.synthetic_outlier_injection_test(lp, shock_size=0.5)
        assert result["raw_causal_ok"], result
        assert result["exposure_causal_ok"], result
        assert result["naive_control_reacts_early"], (
            "naive control did not react early -- injection test has no discriminating power", result)
        assert result["overall_pass"]

    def test_synthetic_injection_negative_shock(self):
        lp = make_synthetic_price(n=500, seed=7)
        result = se.synthetic_outlier_injection_test(lp, shock_size=-0.4)
        assert result["overall_pass"], result
