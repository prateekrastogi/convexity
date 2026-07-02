"""Kelly / vol-target term — c·SR_LCB/σ̂ + c(2−c) growth capture (LEVERAGE.md §2.3–§2.5, §10)."""

from __future__ import annotations

import dataclasses

import numpy as np

from convexity.leverage.ceilings.kelly_vol_target import growth_capture, kelly_vol_target_term
from convexity.leverage.config.defaults import LeverageDefaults


def test_growth_capture_identity_bit_for_bit():
    # §2.3 / §10 gate: c(2−c) = 0.4375 at c=0.25 (quarter-Kelly), 0.75 at c=0.5 (half-Kelly).
    assert growth_capture(0.25) == 0.4375
    assert growth_capture(0.5) == 0.75


def test_kelly_term_matches_worked_example_step1():
    # §7 worked example: f*_LCB = 0.091/0.0389 = 2.339...; c·f* = 0.25·2.339 = 0.585 (3 s.f.).
    gt = kelly_vol_target_term(0.25, 0.091, 0.0389)
    assert np.isclose(gt.kelly_term, 0.25 * 0.091 / 0.0389)
    assert round(gt.kelly_term, 3) == 0.585


def test_size_one_diagnostic_consistency():
    # Invariant SIZE-ONE: kelly_term · σ̂_t == c·SR_LCB_t == σ_target (one dial, no second σ_target).
    gt = kelly_vol_target_term(0.5, 0.12, 0.04)
    assert np.isclose(gt.kelly_term * gt.sigma_hat, gt.c_t * gt.sr_lcb)
    assert np.isclose(gt.sigma_target, gt.c_t * gt.sr_lcb)


def test_no_second_sigma_target_field_in_config():
    # Invariant SIZE-ONE static check: LeverageDefaults exposes no independent sigma_target dial.
    fields = {f.name for f in dataclasses.fields(LeverageDefaults)}
    assert not any("sigma_target" in f or "vol_target" in f for f in fields)


def test_zero_or_nan_sigma_hat_yields_zero_term_not_crash():
    # §8 edge case: a non-positive/NaN σ̂_t (should never happen post-floor) floors the term to 0.
    assert kelly_vol_target_term(0.25, 0.09, 0.0).kelly_term == 0.0
    assert kelly_vol_target_term(0.25, 0.09, float("nan")).kelly_term == 0.0


def test_term_never_negative_on_negative_edge():
    # §8: a non-positive SR_LCB (should already be floored upstream) never produces an inverted book.
    assert kelly_vol_target_term(0.25, -0.5, 0.04).kelly_term == 0.0
