"""Tests for the Carver dynamic optimiser (AFTS Strategy 25).

Anchored to the book's worked example (Tables 114–123, capital $500,000, US 5y / US 10y / S&P 500).
Where the book's PRINTED numbers are internally inconsistent with its own stated formula, we test
the formula-faithful value and document the errata.
"""

import numpy as np
import pytest

from convexity.selection.dynamic_optimization import (
    Instrument,
    apply_buffer,
    covariance_matrix,
    greedy_optimise,
    optimise_portfolio,
    select_candidates,
    tracking_error_std,
)
from convexity.selection.dynamic_optimization import _cost_in_weight_terms, _round_half_up

# --- Book example inputs (Table 114) --------------------------------------------------------
CAPITAL = 500_000.0
N_OPT = np.array([0.4, 0.9, 3.1])            # optimal unrounded positions (contracts)
WPC = np.array([0.22, 0.24, 0.04])           # weight per contract (b / capital)
SIGMA = np.array([0.052, 0.082, 0.171])      # annualised stdev
CORR = np.array([[1.0, 0.9, -0.1], [0.9, 1.0, -0.1], [-0.1, -0.1, 1.0]])
TARGET_W = N_OPT * WPC                        # optimal weight = a × c = [0.088, 0.216, 0.124]
COST_PER_CONTRACT = np.array([5.50, 11.50, 0.875])  # US 5y, US 10y, S&P 500 micro


def test_covariance_matches_table_115_exactly():
    cov = covariance_matrix(SIGMA, CORR)
    expected = np.array(
        [
            [0.002704, 0.003838, -0.000889],
            [0.003838, 0.006724, -0.001402],
            [-0.000889, -0.001402, 0.029241],
        ]
    )
    # Table 115 is printed to 6 dp; rounding our exact σᵀρσ to 6 dp must match cell-for-cell.
    assert np.allclose(np.round(cov, 6), expected, atol=0.0)


def test_target_weights():
    assert np.allclose(TARGET_W, [0.088, 0.216, 0.124])


def test_cost_in_weight_terms_match_book():
    # Book image 257: (5.50/500000)/0.22 = 0.0050%; (11.50/...)/0.24 ≈ 0.010%; (0.875/...)/0.04 ≈ 0.0044%
    cw = _cost_in_weight_terms(COST_PER_CONTRACT, WPC, CAPITAL)
    assert cw[0] == pytest.approx(0.00005, rel=1e-6)      # 0.0050%
    assert cw[1] == pytest.approx(0.00009583, rel=1e-4)   # ~0.010%
    assert cw[2] == pytest.approx(0.00004375, rel=1e-4)   # ~0.0044%


def test_tracking_error_of_zero_portfolio():
    # Formula-faithful √(eᵀΣe) for e = [-0.088,-0.216,-0.124] is 2.89% (the book PRINTS 2.67% — errata).
    cov = covariance_matrix(SIGMA, CORR)
    te = tracking_error_std(np.zeros(3), TARGET_W, cov)
    assert te == pytest.approx(0.028908, abs=1e-5)


def test_greedy_is_formula_faithful_not_book_errata():
    # Canonical greedy on the example yields [0, 1, 3] (≈ rounding of [0.4,0.9,3.1]); NOT the book's [0,1,2].
    cov = covariance_matrix(SIGMA, CORR)
    pos, _w = greedy_optimise(TARGET_W, WPC, cov)
    assert list(pos) == [0, 1, 3]
    # and [0,1,3] genuinely tracks better than the book's [0,1,2]:
    te_013 = tracking_error_std(np.array([0, 1, 3]) * WPC, TARGET_W, cov)
    te_012 = tracking_error_std(np.array([0, 1, 2]) * WPC, TARGET_W, cov)
    assert te_013 < te_012
    assert te_013 == pytest.approx(0.002952, abs=1e-5)


def test_positions_are_integers_and_never_against_forecast_sign():
    # One short (negative optimal) + one long; optimiser must never flip sign vs the forecast.
    target = np.array([-0.30, 0.20])
    wpc = np.array([0.05, 0.04])
    sigma = np.array([0.15, 0.20])
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    cov = covariance_matrix(sigma, corr)
    pos, _w = greedy_optimise(target, wpc, cov)
    assert pos.dtype.kind == "i"
    assert pos[0] <= 0  # short stays short (or zero)
    assert pos[1] >= 0  # long stays long (or zero)


def test_zero_target_is_never_traded():
    target = np.array([0.0, 0.20])
    wpc = np.array([0.05, 0.04])
    cov = covariance_matrix(np.array([0.15, 0.20]), np.eye(2))
    pos, _w = greedy_optimise(target, wpc, cov)
    assert pos[0] == 0


def test_tradeable_mask_excludes_holding():
    # Instrument 0 has the strongest target but is non-tradeable → must end at 0 contracts.
    target = np.array([0.50, 0.20])
    wpc = np.array([0.05, 0.04])
    cov = covariance_matrix(np.array([0.15, 0.20]), np.array([[1.0, 0.5], [0.5, 1.0]]))
    pos, _w = greedy_optimise(target, wpc, cov, tradeable=np.array([False, True]))
    assert pos[0] == 0


def test_position_limits_are_respected():
    target = np.array([0.50])  # would want many contracts
    wpc = np.array([0.05])
    cov = covariance_matrix(np.array([0.15]), np.array([[1.0]]))
    pos, _w = greedy_optimise(target, wpc, cov, position_limits=[(0, 2)])
    assert pos[0] <= 2


def test_buffer_adjustment_factor_and_trades():
    # From flat, buffer the optimised [0,1,3] back toward the edge of a 1% tracking-error buffer.
    cov = covariance_matrix(SIGMA, CORR)
    opt = np.array([0, 1, 3])
    prev = np.array([0, 0, 0])
    # T (current vs optimised) and α
    ep = opt * WPC - prev * WPC
    T = float(np.sqrt(ep @ cov @ ep))
    alpha = max((T - 0.05 * 0.20) / T, 0.0)
    assert alpha == pytest.approx(0.6293, abs=1e-3)
    new_pos = apply_buffer(opt, prev, WPC, cov, risk_target=0.20)
    assert list(new_pos) == [0, 1, 2]  # round(0.629×[0,1,3]) = [0,1,2]


def test_buffer_no_trade_inside_band():
    cov = covariance_matrix(SIGMA, CORR)
    opt = np.array([0, 1, 3])
    # already holding the optimised positions → T = 0 < buffer → no trade
    new_pos = apply_buffer(opt, opt, WPC, cov, risk_target=0.20)
    assert list(new_pos) == [0, 1, 3]


def test_high_level_optimise_and_candidates():
    instruments = [
        Instrument("US5y", 0.4, 0.22, 0.052, cost_per_contract=5.50),
        Instrument("US10y", 0.9, 0.24, 0.082, cost_per_contract=11.50),
        Instrument("SP500", 3.1, 0.04, 0.171, cost_per_contract=0.875),
    ]
    res = optimise_portfolio(instruments, CORR, CAPITAL, apply_buffering=True)
    # tiny costs in this example don't change the greedy result -> optimised [0,1,3], buffered [0,1,2]
    assert res.optimised_positions == {"US5y": 0, "US10y": 1, "SP500": 3}
    assert res.buffered_positions == {"US5y": 0, "US10y": 1, "SP500": 2}
    assert res.candidates == ["US10y", "SP500"]
    assert res.optimal_weights["SP500"] == pytest.approx(0.124)

    cands = select_candidates(instruments, CORR, CAPITAL)
    assert cands == ["US10y", "SP500"]


def test_small_account_zeros_out_expensive_high_weight_contract():
    # A contract whose single-contract weight massively overshoots a tiny target should be dropped,
    # with risk transferred to a cheaper correlated instrument (the chapter's headline behaviour).
    instruments = [
        Instrument("BIG", 0.05, 0.40, 0.20),   # one contract = 0.40 weight, target only 0.02
        Instrument("SMALL", 0.05, 0.02, 0.20),  # one contract = 0.02 weight, target 0.001
    ]
    corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    res = optimise_portfolio(instruments, corr, CAPITAL, apply_buffering=False)
    assert res.optimised_positions["BIG"] == 0  # too lumpy to hold


# --- live-trading code paths (flagged by the verification audit) ----------------------------


def test_round_half_up_is_deterministic():
    assert int(_round_half_up(2.5)) == 3  # np.round(2.5) would give 2 (banker's)
    assert int(_round_half_up(-2.5)) == -3
    assert int(_round_half_up(1.4)) == 1
    assert int(_round_half_up(2.6)) == 3


def test_cost_penalty_reduces_trading():
    # Without costs the greedy buys 2 contracts; a large cost penalty makes the first trade not worth it.
    target = np.array([0.10])
    wpc = np.array([0.05])
    cov = covariance_matrix(np.array([0.20]), np.array([[1.0]]))
    no_cost, _ = greedy_optimise(target, wpc, cov)
    with_cost, _ = greedy_optimise(
        target, wpc, cov, cost_per_contract=np.array([200.0]), capital=CAPITAL, cost_multiplier=50.0
    )
    assert list(no_cost) == [2]
    assert list(with_cost) == [0]
    assert with_cost[0] < no_cost[0]


def test_buffer_on_a_sell():
    # prev over the optimised long -> buffering must SELL part way back, never below the optimum.
    cov = covariance_matrix(np.array([0.15, 0.20]), np.eye(2))
    wpc = np.array([0.05, 0.04])
    new = apply_buffer(np.array([0, 1]), np.array([0, 3]), wpc, cov, risk_target=0.20)
    assert new[1] == 2  # sold one (3 -> 2), between optimum 1 and prev 3


def test_buffer_on_sign_flip():
    # prev long, optimised short -> buffering moves toward the short side.
    cov = covariance_matrix(np.array([0.20]), np.array([[1.0]]))
    new = apply_buffer(np.array([-2]), np.array([1]), np.array([0.05]), cov, risk_target=0.20)
    assert new[0] < 0  # now short (moved from +1 toward the optimised -2)


def test_buffer_clamps_to_position_limits():
    # A tightened max can never leave us holding outside it (audit: opt=2, prev=5 would return 3).
    cov = covariance_matrix(np.array([0.20]), np.array([[1.0]]))
    unclamped = apply_buffer(np.array([2]), np.array([5]), np.array([0.05]), cov, risk_target=0.20)
    assert unclamped[0] == 3  # documents the raw buffered value...
    clamped = apply_buffer(
        np.array([2]), np.array([5]), np.array([0.05]), cov, risk_target=0.20, position_limits=[(0, 2)]
    )
    assert clamped[0] == 2  # ...which the limit clamps back to max


def test_position_limit_plumbed_through_instrument():
    instruments = [Instrument("X", optimal_position=10.0, weight_per_contract=0.05, sigma=0.20, max_contracts=2)]
    res = optimise_portfolio(instruments, np.array([[1.0]]), CAPITAL, apply_buffering=True)
    assert res.optimised_positions["X"] <= 2
    assert res.buffered_positions["X"] <= 2
