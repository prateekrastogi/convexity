"""SYSTEM validation units that shipped with no coverage at all: triple-barrier labelling and
the evidence cache.

Both carry named invariants in their own docstrings — HC (no realized hold exceeds H_sys) and CA
(a combiner-only config change re-scores entirely from cache). Those invariants are what is
asserted here; the modules were at 0%, so nothing was holding them.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.system.validate.cache import EvidenceCache
from convexity.system.validate.labels import triple_barrier_label


def ramp(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------- triple barrier
def test_long_hits_the_upper_barrier_first():
    prices = ramp(100, 101, 105, 90)
    lab = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.02,
                               b_up=1.0, b_dn=1.0, h_sys=5)
    assert lab.outcome == "upper"
    assert lab.touch_idx == 2          # first touch, not the best touch
    assert lab.ret_net > 0


def test_long_hits_the_lower_barrier_first():
    prices = ramp(100, 99, 97, 120)
    lab = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.02,
                               b_up=1.0, b_dn=1.0, h_sys=5)
    assert lab.outcome == "lower"
    assert lab.ret_net < 0


def test_short_side_inverts_which_move_is_favourable():
    """A price FALL is the short's upper (favourable) barrier."""
    prices = ramp(100, 97)
    lab = triple_barrier_label(prices, 0, side=-1, sigma_horizon=0.02,
                               b_up=1.0, b_dn=1.0, h_sys=5)
    assert lab.outcome == "upper"
    assert lab.ret_net > 0


def test_untouched_barriers_force_a_time_cap():
    prices = ramp(100, 100.05, 100.02, 100.01, 99.99)
    lab = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.05,
                               b_up=1.0, b_dn=1.0, h_sys=3)
    assert lab.outcome == "time_cap"


def test_invariant_HC_no_realized_hold_exceeds_h_sys():
    """The contractual cap. A label that outlived H_sys would silently extend the horizon the
    whole validation was sized for."""
    prices = ramp(*[100.0 + 0.001 * i for i in range(50)])
    for h in (1, 3, 7, 20):
        lab = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.9,
                                   b_up=1.0, b_dn=1.0, h_sys=h)
        assert lab.horizon_bars <= h
        assert lab.touch_idx <= h


def test_time_cap_never_scans_past_the_end_of_the_series():
    prices = ramp(100, 100.1)
    lab = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.5,
                               b_up=1.0, b_dn=1.0, h_sys=99)
    assert lab.touch_idx == len(prices) - 1


def test_costs_bias_the_label_low():
    """Section 6.1b: labels are deliberately conservative — cost is subtracted, never added."""
    prices = ramp(100, 105)
    free = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.02,
                                b_up=1.0, b_dn=1.0, h_sys=3)
    charged = triple_barrier_label(prices, 0, side=1, sigma_horizon=0.02,
                                   b_up=1.0, b_dn=1.0, h_sys=3,
                                   entry_cost=0.001, exit_cost=0.002)
    assert charged.ret_net == pytest.approx(free.ret_net - 0.003)
    assert charged.ret_net < free.ret_net


# ---------------------------------------------------------------- evidence cache
def test_cache_computes_once_then_serves_from_memory():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "evidence"

    c = EvidenceCache()
    args = ("mod", "v1", "fold0", "cfgA", (0, 100))
    assert c.get_or_compute(*args, compute) == "evidence"
    assert c.get_or_compute(*args, compute) == "evidence"
    assert calls["n"] == 1
    assert (c.hits, c.misses) == (1, 1)


def test_invariant_CA_a_combiner_only_change_is_a_pure_cache_hit():
    """THE point of the module: module evidence does not depend on the combiner coordinate, so
    re-scoring a new combiner config must cost nothing. The key deliberately omits config_hash."""
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return object()

    c = EvidenceCache()
    c.get_or_compute("mod", "v1", "fold0", "combiner-A", (0, 100), compute)
    for cfg in ("combiner-B", "combiner-C", "combiner-D"):
        c.get_or_compute("mod", "v1", "fold0", cfg, (0, 100), compute)
    assert calls["n"] == 1
    assert c.hits == 3
    assert c.hit_rate == pytest.approx(0.75)


def test_a_new_module_version_must_miss():
    """Cache correctness in the other direction: a changed module cannot serve stale evidence."""
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    c = EvidenceCache()
    c.get_or_compute("mod", "v1", "fold0", "cfg", (0, 100), compute)
    c.get_or_compute("mod", "v2", "fold0", "cfg", (0, 100), compute)
    assert calls["n"] == 2


def test_a_different_fold_or_bar_range_misses():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    c = EvidenceCache()
    c.get_or_compute("mod", "v1", "fold0", "cfg", (0, 100), compute)
    c.get_or_compute("mod", "v1", "fold1", "cfg", (0, 100), compute)
    c.get_or_compute("mod", "v1", "fold0", "cfg", (100, 200), compute)
    assert calls["n"] == 3
    assert c.misses == 3


def test_hit_rate_of_an_untouched_cache_is_zero_not_a_zero_division():
    assert EvidenceCache().hit_rate == 0.0
