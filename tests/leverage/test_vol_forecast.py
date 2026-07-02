"""Volatility forecast σ̂_t — closed-form EWMA + conservative floor (LEVERAGE.md §4, §10)."""

from __future__ import annotations

import numpy as np

from convexity.leverage.config.defaults import PERIODS_PER_YEAR, LeverageDefaults
from convexity.leverage.ceilings.kelly_vol_target import kelly_vol_target_term
from convexity.leverage.types import annualized_to_weekly_vol
from convexity.leverage.vol.forecast import sigma_hat_t


def test_alpha_matches_half_life_formula():
    d = LeverageDefaults()
    vf = sigma_hat_t(np.random.default_rng(0).normal(0, 0.04, 100), d)
    assert np.isclose(vf.alpha, 1.0 - 2.0 ** (-1.0 / d.h_vol))
    assert np.isclose(vf.ess, (2.0 - vf.alpha) / vf.alpha)


def test_ewma_tracks_a_known_constant_vol_series():
    # a long calm series of ~2% weekly vol -> σ̂_t near 0.02, well below the 3.9% prior.
    d = LeverageDefaults()
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.02, 400)
    vf = sigma_hat_t(r, d)
    assert not vf.on_floor
    assert 0.015 < vf.sigma_hat < 0.025


def test_vol_floor_thin_history_uses_28pct_annualized_prior():
    # Invariant VOL-FLOOR: history shorter than n_vol_min falls back to the class prior, never lower.
    d = LeverageDefaults()
    prior_weekly = annualized_to_weekly_vol(d.vol_prior_annual, PERIODS_PER_YEAR)
    # a thin, deceptively-CALM history (tiny realized vol) must NOT narrow σ̂_t below the prior.
    thin = np.full(d.n_vol_min - 1, 0.001)
    vf = sigma_hat_t(thin, d)
    assert vf.on_floor
    assert np.isclose(vf.sigma_hat, prior_weekly)
    assert vf.sigma_hat >= prior_weekly  # thin history WIDENS, never narrows (VOL-FLOOR)


def test_vol_floor_cold_start_is_wider_than_prior():
    # zero history -> a deliberately wide MULTIPLE of the prior (§4.2 step 2): under-levered, never over.
    d = LeverageDefaults()
    prior_weekly = annualized_to_weekly_vol(d.vol_prior_annual, PERIODS_PER_YEAR)
    vf = sigma_hat_t(np.array([]), d)
    assert vf.on_floor
    assert np.isclose(vf.sigma_hat, prior_weekly * d.vol_cold_start_multiple)
    assert vf.sigma_hat > prior_weekly


def test_vol_floor_makes_g_smaller_than_calm_full_history_at_same_edge():
    # Invariant VOL-FLOOR: at the fallback, G_t < G_t computed from a full calm history at same SR_LCB.
    d = LeverageDefaults()
    sr_lcb, c = 0.09, 0.25
    thin = np.full(d.n_vol_min - 1, 0.001)
    calm = np.random.default_rng(2).normal(0.0, 0.02, 400)
    g_thin = kelly_vol_target_term(c, sr_lcb, sigma_hat_t(thin, d).sigma_hat).kelly_term
    g_calm = kelly_vol_target_term(c, sr_lcb, sigma_hat_t(calm, d).sigma_hat).kelly_term
    assert g_thin < g_calm


def test_sigma_hat_never_zero_or_nan_on_degenerate_history():
    # §8 edge case: a flat/degenerate long history still floors to the prior, never 0/NaN.
    d = LeverageDefaults()
    vf = sigma_hat_t(np.zeros(400), d)
    assert vf.sigma_hat > 0.0 and np.isfinite(vf.sigma_hat)


def test_n_vol_min_derived_from_h_vol_invariant():
    # §3: n_vol_min >= 2*h_vol enforced in __post_init__.
    import pytest

    with pytest.raises(ValueError):
        LeverageDefaults(h_vol=8, n_vol_min=10)


def test_no_dup_ewma_sigma_hat_depends_only_on_h_vol_not_cost():
    # Invariant NO-DUP-EWMA: σ̂_t is parameterized by LEVERAGE's own h_vol, wholly independent of
    # COST's funding EWMA. The forecast module imports no COST config; changing a COST half-life
    # cannot alter σ̂_t. Assert σ̂_t is a pure function of (series, h_vol) via the defaults object.
    import ast
    import pathlib

    from convexity.leverage.vol import forecast as fc

    tree = ast.parse(pathlib.Path(fc.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not any("cost" in m for m in imported), (
        f"vol/forecast.py must not import COST config (NO-DUP-EWMA); imports={imported}"
    )
    r = np.random.default_rng(5).normal(0.0, 0.03, 300)
    a = sigma_hat_t(r, LeverageDefaults(h_vol=8, n_vol_min=16)).sigma_hat
    b = sigma_hat_t(r, LeverageDefaults(h_vol=4, n_vol_min=16)).sigma_hat
    assert a != b  # different LEVERAGE half-life -> different estimate (own dial, not shared)
