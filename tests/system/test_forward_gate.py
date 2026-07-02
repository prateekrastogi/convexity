"""The §7.5 held-out PAIRED block-bootstrap forward/cadence gate (SYSTEM §7.5, Invariant RP/BL/HC).

Covers ForwardGate — the swap-cadence gate SYSTEM §11 assigns to ``promotion.py`` — which had no
test in ``tests/`` before this file (the adjacent §6 conjunction lives in ``promotion_gate.py`` and
is covered by ``test_promotion_gate.py``). Every named invariant of the gate gets a test that FAILS
if the invariant is violated: the paired-difference construction (Inv BL — identical-length series,
a length mismatch RAISES), the CI-excludes-zero AND clears-cost swap rule, the short-forward-window
DEFERRAL (``insufficient_forward_track`` when ``n_obs < max(MinTRL, n_obs_floor)``), and the
determinism of the block-bootstrap given a seed (iid bootstrap FORBIDDEN — §8.1 block length is fed
straight from the RED defaults).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from convexity.system.config.defaults import SystemDefaults
from convexity.system.validate.promotion import ForwardGate, ForwardGateResult


def _cfg() -> SimpleNamespace:
    """ForwardGate's only documented dependency is ``cfg.defaults`` (a :class:`SystemDefaults`),
    from which it reads ``forward_block_length``/``forward_n_boot``/``forward_ci_level`` (§8.1 RED
    block length, NOT an iid bootstrap) and ``n_obs_floor`` (§7.5 forward-track floor)."""
    return SimpleNamespace(defaults=SystemDefaults())


def test_length_mismatch_raises_paired_invariant_BL():
    """Invariant BL: the forward gate is PAIRED — two independent-length samples are illegal."""
    cand = np.zeros(50)
    inc = np.zeros(49)
    with pytest.raises(ValueError, match="identical-length"):
        ForwardGate().evaluate(cand, inc, _cfg())


def test_short_forward_window_defers_with_insufficient_forward_track():
    """§7.5: a forward block shorter than max(MinTRL, n_obs_floor) DEFERS the swap, never swaps."""
    rng = np.random.default_rng(0)
    n = 20  # < n_obs_floor (24) => insufficient regardless of how good the difference looks
    cand = 0.05 + 0.001 * rng.standard_normal(n)  # hugely candidate-favorable
    inc = np.zeros(n)
    res = ForwardGate().evaluate(cand, inc, _cfg())
    assert isinstance(res, ForwardGateResult)
    assert not res.n_obs_sufficient
    assert not res.swap
    assert res.reason_codes == ("insufficient_forward_track",)
    assert res.forward_n_obs == n


def test_min_trl_raises_the_floor_above_n_obs_floor():
    """§7.5: the floor is max(MinTRL, n_obs_floor); a MinTRL above n_obs_floor can defer a
    window that would otherwise clear the raw 24-obs floor."""
    rng = np.random.default_rng(1)
    n = 30  # >= n_obs_floor(24) but < min_trl(60): must still defer
    cand = 0.05 + 0.001 * rng.standard_normal(n)
    inc = np.zeros(n)
    res = ForwardGate().evaluate(cand, inc, _cfg(), min_trl=60.0)
    assert not res.n_obs_sufficient
    assert not res.swap
    assert res.reason_codes == ("insufficient_forward_track",)


def test_strong_positive_difference_swaps_ci_excludes_zero_and_clears_cost():
    """A strong, persistent candidate-minus-incumbent edge over a long window: the difference-Sharpe
    CI excludes zero and its lower bound clears a small cost hurdle => swap=True."""
    rng = np.random.default_rng(6)
    n = 200
    d = 0.03 + 0.01 * rng.standard_normal(n)  # high, stable positive edge (Sharpe ~3 per bucket)
    cand = d
    inc = np.zeros(n)
    res = ForwardGate().evaluate(cand, inc, _cfg(), round_trip_cost=0.0013, seed=6)
    assert res.n_obs_sufficient
    assert res.ci_low > 0.0
    assert res.ci_low > 0.0013
    assert res.swap
    assert res.reason_codes == ("forward_gate_passed",)


def test_ci_including_zero_does_not_swap():
    """A noisy zero-mean difference: the paired CI straddles zero => no swap (Inv RP)."""
    rng = np.random.default_rng(2)
    n = 200
    d = 0.02 * rng.standard_normal(n)  # zero-mean difference
    cand = d
    inc = np.zeros(n)
    res = ForwardGate().evaluate(cand, inc, _cfg())
    assert res.n_obs_sufficient
    assert res.ci_low <= 0.0
    assert not res.swap
    assert "paired_ci_includes_zero" in res.reason_codes


def test_positive_ci_below_cost_does_not_swap():
    """§7.5: the CI (on the difference SHARPE) can exclude zero yet fail the cost hurdle. The gate
    compares ``ci_low`` against ``round_trip_cost``; a cost hurdle set just ABOVE the measured
    lower bound must DEFER even though the CI is strictly positive."""
    rng = np.random.default_rng(4)
    n = 200
    # modest positive-Sharpe difference (positive mean, real variance => bounded finite Sharpe CI)
    d = 0.01 + 0.02 * rng.standard_normal(n)
    cand = d
    inc = np.zeros(n)
    # read the gate's own lower bound with a zero hurdle, then set a hurdle just above it
    probe = ForwardGate().evaluate(cand, inc, _cfg(), round_trip_cost=0.0, seed=5)
    assert probe.ci_low > 0.0 and probe.swap  # positive CI clears a zero hurdle
    hurdle = probe.ci_low + 1e-6
    res = ForwardGate().evaluate(cand, inc, _cfg(), round_trip_cost=hurdle, seed=5)
    assert res.n_obs_sufficient
    assert res.ci_low > 0.0
    assert res.ci_low <= hurdle
    assert not res.swap
    assert "paired_ci_below_cost" in res.reason_codes


def test_bootstrap_is_deterministic_given_seed():
    """The §8.1 block bootstrap is seeded and reproducible (not an unseeded iid draw)."""
    rng = np.random.default_rng(3)
    n = 200
    cand = 0.005 + 0.02 * rng.standard_normal(n)
    inc = 0.02 * rng.standard_normal(n)
    r1 = ForwardGate().evaluate(cand, inc, _cfg(), seed=11)
    r2 = ForwardGate().evaluate(cand, inc, _cfg(), seed=11)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)
    assert r1.swap == r2.swap


def test_block_length_reported_is_the_red_default():
    """The gate reports the §8.1 RED block length it used (not an ad-hoc value)."""
    cand = np.full(200, 0.01)
    inc = np.zeros(200)
    res = ForwardGate().evaluate(cand, inc, _cfg())
    assert res.block_length == SystemDefaults().forward_block_length == 4
