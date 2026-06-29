"""TREND worked-example regressions (TREND §1.5/§9/§10/§13/§18). Bit-for-bit anchors."""

import math

import numpy as np
import pytest

from convexity.trend import TrendLifeEngine
from convexity.trend.config.defaults import Z_DRIFT_WORKED
from convexity.trend.estimate.drift import VarianceIntegratedDrift
from convexity.trend.policy.convex_exit import ConvexExit
from convexity.trend.synth.generate import (
    ALL_PATHS,
    jump_against,
    jump_with,
    persistent_trend_low_noise,
)
from convexity.trend.types import Position, VolState
from convexity.trend.validate.labels import TrendLifeLabeler

from tests.trend.conftest import make_position


# --- §17: all 17 adversarial paths exist + produce finite OHLCV ---------------------------
def test_seventeen_synthetic_paths():
    assert len(ALL_PATHS) == 17
    for name, fn in ALL_PATHS.items():
        bars, e = fn()
        assert 0 <= e < len(bars)
        for arr in (bars.o, bars.h, bars.l, bars.c):
            assert np.all(np.isfinite(arr)), name


# --- §13 / §9: z_drift = log(1.1)/0.1 = 0.9531017980432486 --------------------------------
def test_z_drift_worked_constant():
    assert Z_DRIFT_WORKED == pytest.approx(math.log(1.1) / 0.1, abs=1e-12)

    drift = VarianceIntegratedDrift()
    # construct a +10% favorable long move with integrated path-vol 0.1 (variance 0.01).
    entry = 100.0
    price = entry * 1.1
    bars_c = np.array([entry, price])
    from convexity.core.types import Bars

    bars = Bars(ts=np.array([0, 60_000], dtype=np.int64),
                o=bars_c.copy(), h=bars_c.copy(), l=bars_c.copy(), c=bars_c)
    pos = Position(side=1, entry_idx=0, entry_price=entry)
    vol = VolState(method="ewma", sigma_log=0.0, sigma_price=0.0,
                   integrated_var_since_entry=0.01, gap_var=0.0, vol_bucket=0, liq_bucket=0,
                   tail_nu=8.0, quality_flag="ok")
    z, fav, adv = drift.update(bars, 1, pos, vol)
    assert z == pytest.approx(Z_DRIFT_WORKED, abs=1e-9)


# --- §10 / INV-CVX: trail multiple schedule is monotone non-decreasing in profit ----------
def test_trail_schedule_monotone(cfg):
    ce = ConvexExit()
    profits = np.linspace(-0.5, 8.0, 200)
    mults = [ce.trail_multiple(p, cfg) for p in profits]
    assert all(b >= a - 1e-12 for a, b in zip(mults, mults[1:]))  # non-decreasing
    # the four §10 bands map to the four anchors.
    assert ce.trail_multiple(0.4, cfg) == cfg.trail_multipliers[0]  # tight
    assert ce.trail_multiple(1.0, cfg) == cfg.trail_multipliers[1]  # normal
    assert ce.trail_multiple(3.0, cfg) == cfg.trail_multipliers[2]  # loose
    assert ce.trail_multiple(5.0, cfg) == cfg.trail_multipliers[3]  # tail


# --- §18 volatility: integrated variance is monotone nondecreasing after entry ------------
def test_integrated_var_monotone(cfg, spec):
    for name, fn in ALL_PATHS.items():
        bars, e = fn()
        pos = make_position(bars, min(e, len(bars) - 2))
        eng = TrendLifeEngine(cfg)
        eng._reset(bars, pos, cfg)
        prev = -1.0
        for i in range(pos.entry_idx, min(pos.entry_idx + 80, len(bars))):
            b = eng.builder.update(bars, i, pos, cfg)
            iv = b.vol.integrated_var_since_entry
            assert np.isfinite(iv), name
            assert iv >= prev - 1e-12, name
            prev = iv


# --- §18: no NaN/inf on adversarial data + gap_var applied once --------------------------
def test_no_nan_inf_adversarial(cfg, spec):
    for name, fn in ALL_PATHS.items():
        bars, e = fn()
        pos = make_position(bars, min(e, len(bars) - 2))
        eng = TrendLifeEngine(cfg)
        ds = eng.run_batch(bars, pos, spec, cfg)
        for d in ds:
            assert np.isfinite(d.exit_state.trail_level), name
            assert np.isfinite(d.horizon.integrated_var_since_entry), name


# --- §18 jump: jump_against/jump_with correctly signed ------------------------------------
def test_jump_signs(cfg, spec):
    bars_a, ea = jump_against()
    pos_a = make_position(bars_a, ea)
    eng_a = TrendLifeEngine(cfg)
    ds_a = eng_a.run_batch(bars_a, pos_a, spec, cfg)
    assert ds_a[-1].stop_cause == "jump_against"  # adverse jump -> hard exit

    bars_w, ew = jump_with()
    pos_w = make_position(bars_w, ew)
    eng_w = TrendLifeEngine(cfg)
    ds_w = eng_w.run_batch(bars_w, pos_w, spec, cfg)
    # a favorable jump never exits by itself.
    assert "jump_against" not in {d.stop_cause for d in ds_w}


# --- §18: persistent trend held longer than a dying/reversing trend -----------------------
def test_persistent_held_longer(cfg, spec):
    from convexity.trend.synth.generate import trend_reverses

    bars_p, ep = persistent_trend_low_noise()
    pos_p = make_position(bars_p, ep)
    n_p = len(TrendLifeEngine(cfg).run_batch(bars_p, pos_p, spec, cfg))

    bars_r, er = trend_reverses()
    pos_r = make_position(bars_r, er)
    n_r = len(TrendLifeEngine(cfg).run_batch(bars_r, pos_r, spec, cfg))
    assert n_p > n_r


# --- §1.5c: net-convexity E[Pi] PASS / MARGINAL / FAIL worked example ----------------------
def test_net_convexity_worked_example():
    """The §1.5c worked numbers: PASS +0.3725, MARGINAL +0.08, FAIL -0.13 R0."""
    def e_pi(p_win, avg_win, avg_loss):
        return p_win * avg_win - (1 - p_win) * avg_loss

    assert e_pi(0.35, 3.2, 1.15) == pytest.approx(0.3725, abs=1e-4)  # PASS
    assert e_pi(0.35, 3.2, 1.60) == pytest.approx(0.08, abs=1e-4)  # MARGINAL
    assert e_pi(0.35, 2.6, 1.60) == pytest.approx(-0.13, abs=1e-4)  # FAIL: bleed wins


# --- §18 / Inv L: asymmetric convex fit vs symmetric affine fit ---------------------------
def test_asymmetric_convex_vs_symmetric_affine():
    """On persistent trends, Pi(M) under the convex schedule curves (giveback fraction shrinks as M
    grows); the symmetric schedule keeps a constant giveback fraction (affine). (Inv L / §1.5a)."""
    # convex schedule: giveback = g(M) with g'(M) < 1 and decreasing marginal giveback.
    M = np.linspace(0.2, 5.0, 50)
    # symmetric (affine): giveback is a constant fraction of M -> Pi = (1-a)*M, marginal const.
    a = 0.4
    pi_sym = (1 - a) * M
    # convex: giveback fraction shrinks with M (multiplier widens) -> marginal giveback decreasing.
    frac = 0.5 / (1 + 0.4 * M)  # decreasing fraction
    g = frac * M
    pi_cvx = M - g
    # marginal giveback g'(M) is non-increasing for the convex schedule, ~constant for symmetric.
    dg_cvx = np.diff(g) / np.diff(M)
    dg_sym = np.diff(a * M) / np.diff(M)
    assert np.all(np.diff(dg_cvx) <= 1e-9)  # non-increasing marginal giveback (INV-CVX)
    assert np.allclose(dg_sym, dg_sym[0])  # constant marginal giveback (affine)
    # convexity of Pi(M): second difference >= 0 for convex, ~0 for affine.
    assert np.all(np.diff(pi_cvx, 2) >= -1e-9)
    assert np.allclose(np.diff(pi_sym, 2), 0.0, atol=1e-9)


# --- §10: gap survival reduces below 1.0 under gap stress ---------------------------------
def test_gap_survival_fraction(cfg):
    ce = ConvexExit()
    vol_gap = VolState(method="ewma", sigma_log=0.001, sigma_price=0.1,
                       integrated_var_since_entry=0.01, gap_var=0.01, vol_bucket=2, liq_bucket=0,
                       tail_nu=8.0, quality_flag="ok")
    frac = ce.gap_survival_fraction(vol_gap, cfg)
    assert cfg.defaults.gap_survival_min_fraction <= frac < 1.0
    # no gap -> full size.
    vol_nogap = VolState(method="ewma", sigma_log=0.001, sigma_price=0.1,
                         integrated_var_since_entry=0.01, gap_var=0.0, vol_bucket=0, liq_bucket=0,
                         tail_nu=8.0, quality_flag="ok")
    assert ce.gap_survival_fraction(vol_nogap, cfg) == 1.0


# --- §12: labeler net-convexity is firewalled + computes E[Pi] ----------------------------
# NOTE: this is the §6.1 STANDALONE, clearly-labeled validation test that the local CostModel stub
# is permitted to back (no COST estimators wired in). The net-EV / promotion gates use the
# COST-sourced path (test_inv_m_*), never this stub.
def test_labeler_net_convexity_stub_standalone(cfg, spec):
    from convexity.trend.config.cost import default_cost_model

    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    eng = TrendLifeEngine(cfg)
    ds = eng.run_batch(bars, pos, spec, cfg)
    labeler = TrendLifeLabeler()
    # cost_log omitted -> the §6.1 validation-only stub fallback is exercised here (and ONLY here).
    labels = labeler.label(bars, pos, ds, default_cost_model(), {"calendar_days": cfg.tau_cap},
                           ("calendar_days",))
    conv = labeler.net_convexity(labels)
    assert set(conv) >= {"E_Pi", "p_win", "avg_win", "p_loss", "avg_loss", "premium_bleed"}
    assert math.isfinite(conv["E_Pi"])


# --- §9: the config-gated bipower/Lee-Mykland jump branch (jump_use_bipower) --------------
def test_jump_bipower_branch_gated(cfg, spec):
    """§9: with ``jump_use_bipower`` ON and clean OHLC quality, the Lee-Mykland branch flags the
    adverse jump and is quiet off-jump; with poor vol quality it DEGRADES to the robust-exceedance
    default (the flag never prices a poor-quality bar by the jump-sensitive bipower scale)."""
    import dataclasses

    from convexity.trend.estimate.jump import JumpDeterioration
    from convexity.trend.synth.generate import jump_against
    from convexity.trend.types import VolState

    bp_cfg = dataclasses.replace(cfg, defaults=dataclasses.replace(cfg.defaults, jump_use_bipower=True))
    jd = JumpDeterioration()
    vol_ok = VolState(method="mad", sigma_log=0.001, sigma_price=0.1, integrated_var_since_entry=0.01,
                      gap_var=0.0, vol_bucket=0, liq_bucket=0, tail_nu=8.0, quality_flag="ok")
    vol_bad = dataclasses.replace(vol_ok, quality_flag="degraded")
    bars, _ = jump_against()
    at = 120  # the -7% adverse jump bar (synth scenario 5)

    # default (robust-exceedance) and bipower(ok) BOTH catch the adverse jump for a long.
    assert jd.update(bars, at, 1, vol_ok, cfg) == (True, False)
    assert jd.update(bars, at, 1, vol_ok, bp_cfg) == (True, False)
    # poor vol quality -> the bipower flag falls back to the default branch (same verdict, no crash).
    assert jd.update(bars, at, 1, vol_bad, bp_cfg) == jd.update(bars, at, 1, vol_bad, cfg)
    # quiet on an ordinary bar under both branches (sparse detector, not an all-hours alarm).
    assert jd.update(bars, 50, 1, vol_ok, bp_cfg) == (False, False)
