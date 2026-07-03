"""Cost validation companions: stats, baselines, amihud_calibration (COST.md Sections 5.3, 6.4, 8.4).

Invariants exercised: MINOR/HONEST-SR (block-bootstrap CI machinery delegates to SYSTEM's ONE
bootstrap, win-rate-by-funding-sign diagnostic present), the flat-bps / no-impact / perfect-foresight
baselines are apples-to-apples comparators, and the walk-forward calibration-drift + PARTICIPATION
diagnostics are causal and closed-form (P7, no fit).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from convexity.cost.accounting import round_trip_friction
from convexity.cost.friction.validate.amihud_calibration import (
    calibrate_window,
    participation_rate,
    walk_forward_drift,
)
from convexity.cost.friction.validate.baselines import (
    flat_bps_friction,
    no_impact_friction,
    perfect_foresight_friction,
)
from convexity.cost.friction.validate.stats import (
    cost_drag_ci,
    net_sharpe_ci,
    win_rate_by_funding_sign,
)
from convexity.system.metrics import PERIODS_PER_YEAR, compute
from convexity.cost.friction.spread_impact import ANCHOR_NOTIONAL, SpreadImpactModel
from convexity.panel.sleeve import CRYPTO_FRICTION

ADV_MEDIAN = 40_000_000.0
AMIHUD_MEDIAN = 1.0e-10

# Real-lake e2e gate: the walk-forward-drift real-universe check only runs where data/stitched exists
# (it does on arb-vm). Unit-test convention keeps everything else lake-free (conftest synthetic panels);
# CPCV parity is anchored in scripts/xsec_cpcv.py, not the unit suite (2-core VM constraint).
_LAKE_DIR = Path(__file__).resolve().parents[4] / "data" / "stitched"
_HAS_LAKE = _LAKE_DIR.is_dir()


# ---------------------------------------------------------------- stats.py (COST.md Section 6.4)

def test_net_sharpe_ci_delegates_to_block_bootstrap_and_brackets_point():
    # The CI uses the SAME annualization as metrics.compute; the point Sharpe should sit inside the
    # bootstrap CI. Deterministic given seed. Uses SYSTEM's ONE block_bootstrap_ci, no own bootstrap.
    rng = np.random.default_rng(5)
    net = rng.normal(0.003, 0.01, 260)
    lo, hi = net_sharpe_ci(net, seed=0)
    point = compute(net).sharpe_ann
    assert lo < hi
    assert lo <= point <= hi
    # deterministic
    assert (lo, hi) == net_sharpe_ci(net, seed=0)


def test_cost_drag_ci_positive_and_ordered():
    rng = np.random.default_rng(6)
    cost = np.abs(rng.normal(0.0013, 0.0003, 260))  # weekly cost drag, positive
    lo, hi = cost_drag_ci(cost, seed=0)
    assert 0.0 <= lo <= hi
    point = float(np.mean(cost) * PERIODS_PER_YEAR)
    assert lo <= point <= hi


def test_win_rate_by_funding_sign_diagnostic():
    # COST.md Section 6.4: unconditional matches metrics.compute; conditional rates split by funding
    # sign so a future divergence is visible. Zero-funding weeks excluded from both conditionals.
    net = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0])
    funding = np.array([0.001, -0.001, 0.001, -0.001, 0.0, 0.002])
    wr = win_rate_by_funding_sign(net, funding)
    assert wr.unconditional == pytest.approx(compute(net).win_rate_weeks)
    # positive-funding weeks: indices 0 (+0.01 win), 2 (+0.03 win), 5 (0.0 -> not a win) -> 2/3
    assert wr.n_positive == 3
    assert wr.when_funding_positive == pytest.approx(2.0 / 3.0)
    # negative-funding weeks: indices 1 (-0.02 loss), 3 (-0.01 loss) -> 0/2
    assert wr.n_negative == 2
    assert wr.when_funding_negative == pytest.approx(0.0)


def test_win_rate_by_funding_sign_rejects_misaligned():
    with pytest.raises(ValueError):
        win_rate_by_funding_sign(np.zeros(5), np.zeros(4))


# ------------------------------------------------------------- baselines.py (COST.md Sections 1, 8.4)

def test_flat_bps_baseline_is_crypto_friction_13bp():
    # The flat CRYPTO_FRICTION research comparator, assembled via accounting.round_trip_friction.
    assert flat_bps_friction() == round_trip_friction(
        taker=CRYPTO_FRICTION.taker,
        half_spread=CRYPTO_FRICTION.half_spread,
        impact=CRYPTO_FRICTION.impact,
    )
    assert flat_bps_friction() == pytest.approx(0.0013, rel=1e-12)


def test_no_impact_baseline_isolates_the_impact_line():
    # no_impact = derived model with impact zeroed; the delta to the full model is exactly the
    # impact contribution (COST.md Section 5.3: the illiquid-tail cost is only visible if impact
    # varies per name).
    m = SpreadImpactModel.anchor(ADV_MEDIAN, AMIHUD_MEDIAN)
    full = m.round_trip_friction_i(ADV_MEDIAN, AMIHUD_MEDIAN, notional_leg=ANCHOR_NOTIONAL)
    no_imp = no_impact_friction(m, ADV_MEDIAN)
    assert no_imp < full
    # the gap is 2x the per-leg impact fraction (round-trip doubles it).
    imp_frac = m.impact_frac(AMIHUD_MEDIAN, ANCHOR_NOTIONAL)
    assert full - no_imp == pytest.approx(2.0 * imp_frac, rel=1e-9)


def test_perfect_foresight_baseline_is_zero_cost():
    # COST.md Section 8.4 / Invariant HONEST-SR: the optimistic upper-bound comparator is ZERO cost;
    # its net Sharpe must never be reported as deployable — kept only to quantify the honest gap.
    z = perfect_foresight_friction(5)
    assert z.shape == (5,)
    assert np.all(z == 0.0)


# ----------------------------------------------------- amihud_calibration.py (COST.md Section 5.3)

def _synth_returns_dv(n_days: int, n_sym: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, (n_days, n_sym))
    dvol = np.full((n_days, n_sym), ADV_MEDIAN) * rng.uniform(0.8, 1.2, (n_days, n_sym))
    return returns, dvol


def test_calibrate_window_is_causal_and_reproduces_anchor():
    # calibrate_window uses only rows <= right_edge (F_t-measurable, COST.md Section 0). The anchored
    # model's median name reproduces the flat defaults by construction.
    returns, dvol = _synth_returns_dv(60, 8)
    cal = calibrate_window(returns, dvol, right_edge=40, window=30)
    assert cal.right_edge == 40
    assert cal.adv_median > 0.0
    assert cal.amihud_median > 0.0
    # a name at the window's median ADV/Amihud reproduces the flat anchors under this calibration.
    m = SpreadImpactModel(c=cal.c, k_amihud=cal.k_amihud)
    from convexity.cost.accounting import HALF_SPREAD_FRAC, IMPACT_FRAC

    assert m.half_spread(cal.adv_median) == pytest.approx(HALF_SPREAD_FRAC, rel=1e-9)
    assert m.impact_frac(cal.amihud_median, ANCHOR_NOTIONAL) == pytest.approx(
        IMPACT_FRAC, rel=1e-9
    )


def test_calibrate_window_does_not_read_future_rows():
    # Corrupting rows strictly AFTER right_edge must not change the calibration (causality).
    returns, dvol = _synth_returns_dv(60, 8)
    cal_base = calibrate_window(returns, dvol, right_edge=40, window=30)
    returns2 = returns.copy()
    dvol2 = dvol.copy()
    returns2[41:] = 999.0  # garbage in the future
    dvol2[41:] = 1.0
    cal_corrupt = calibrate_window(returns2, dvol2, right_edge=40, window=30)
    assert cal_corrupt.c == cal_base.c
    assert cal_corrupt.k_amihud == cal_base.k_amihud


def test_walk_forward_drift_small_on_a_STATIONARY_input():
    # PROPERTY check on a deliberately STATIONARY synthetic universe (constant liquidity
    # distribution): when the cross-sectional liquidity profile does NOT move, the estimator reports
    # small anchor drift. This proves the estimator RESPONDS to (in)stability — it is NOT a claim that
    # the REAL universe stays inside these bounds. The real 100-crypto lake spans 2020->2024, over
    # which crypto liquidity expanded by more than an order of magnitude, so the real drift is LARGE
    # (c_rel_range~1.4, k_amihud_rel_range~1.1 >> these synthetic bounds) — that large real drift is
    # the intended governance recalibration signal, exercised in
    # test_walk_forward_drift_on_real_lake_reports_large_governance_drift below. The 0.5 / 1.0 numbers
    # here are stationary-input sanity ceilings, NOT thresholds calibrated to real universe turnover.
    returns, dvol = _synth_returns_dv(120, 12)
    edges = np.array([40, 60, 80, 100])
    drift = walk_forward_drift(returns, dvol, edges, window=30)
    assert len(drift.windows) == 4
    assert drift.c_rel_range < 0.5  # stationary input -> modest anchor drift (sanity ceiling)
    assert drift.k_amihud_rel_range < 1.0


def test_walk_forward_drift_moves_when_liquidity_regime_shifts():
    # The complement of the stationary check: a synthetic universe whose median dollar-volume RAMPS
    # over the sample (a liquidity-regime shift) must produce MATERIALLY larger anchor drift than the
    # stationary case — proving the estimator is a live detector of the exact instability COST.md
    # Section 5.3 warns about, not a constant. This is the mechanism the real-lake test then confirms
    # actually fires on the true 2020->2024 crypto liquidity expansion.
    rng = np.random.default_rng(7)
    n_days, n_sym = 120, 12
    returns = rng.normal(0.0, 0.02, (n_days, n_sym))
    ramp = np.linspace(1.0, 20.0, n_days)[:, None]  # 20x liquidity growth over the sample
    dvol = np.full((n_days, n_sym), ADV_MEDIAN) * ramp * rng.uniform(0.8, 1.2, (n_days, n_sym))
    edges = np.array([40, 60, 80, 100])
    ramped = walk_forward_drift(returns, dvol, edges, window=30)
    stationary_returns, stationary_dvol = _synth_returns_dv(120, 12)
    stationary = walk_forward_drift(stationary_returns, stationary_dvol, edges, window=30)
    assert ramped.c_rel_range > stationary.c_rel_range
    assert ramped.c_rel_range > 0.5  # a real regime shift blows past the stationary sanity ceiling


def test_participation_rate_flags_large_book():
    # Invariant PARTICIPATION / COST.md Section 5.3: participation = Notional / ADV per name. At the
    # calibrated ~small-single-digit-% regime it is small; scaling the book 20x pushes it out of the
    # validated range — the signal to re-run this check before reusing k_amihud/c unchanged.
    adv = np.array([ADV_MEDIAN, ADV_MEDIAN / 10.0])
    small = participation_rate(adv, ANCHOR_NOTIONAL)
    assert small[0] == pytest.approx(ANCHOR_NOTIONAL / ADV_MEDIAN, rel=1e-12)
    assert small[0] < 0.01  # ~0.1% — inside the calibrated regime
    big = participation_rate(adv, 20.0 * ANCHOR_NOTIONAL)
    assert big[1] > small[1]  # thinner name + bigger book -> higher participation
    # a zero-ADV name is untradeable at any size.
    assert math.isinf(participation_rate(np.array([0.0]), ANCHOR_NOTIONAL)[0])


# ------------- amihud_calibration real-lake e2e (gap 6: no longer synthetic-only) -----------------


@pytest.mark.skipif(not _HAS_LAKE, reason="requires the data/stitched crypto lake (arb-vm only)")
def test_walk_forward_drift_on_real_lake_reports_large_governance_drift():
    # GAP 6 fix: walk_forward_drift was previously validated ONLY on synthetic stationary data. This
    # e2e runs it CAUSALLY against the real panel.build_dollar_volume on the true 100-crypto
    # point-in-time lake (READ-only), confirming (a) it composes with build_panel/build_dollar_volume
    # and (b) the empirically-observed anchor drift over the 2020->2024 sample is LARGE — the
    # governance recalibration signal COST.md Section 5.3 describes, not the small synthetic-stationary
    # drift. This is what turns the drift thresholds from "synthetic sanity bounds" into
    # empirically-grounded reference numbers: on the real universe the liquidity distribution moved
    # by more than an order of magnitude, so c_rel_range and k_amihud_rel_range are BOTH ~1+, well
    # above the stationary-input sanity ceilings of test_walk_forward_drift_small_on_a_STATIONARY_input.
    from convexity.panel.candidates import all_symbols
    from convexity.panel.panel import build_dollar_volume, build_panel

    syms = tuple(all_symbols(sleeves=("crypto",)))
    panel = build_panel(syms)
    dvol = build_dollar_volume(panel)
    close = panel.close
    rets = np.full_like(close, np.nan)
    rets[1:] = close[1:] / close[:-1] - 1.0
    rets = np.where(panel.listed & np.isfinite(rets), rets, np.nan)

    edges = np.arange(60, panel.n_days, 30)  # ~monthly right-edges, each with a full 30d window
    drift = walk_forward_drift(rets, dvol, edges, window=30)

    # (a) it composes and produces a finite anchor at (nearly) every window — causal, no crash.
    finite = [w for w in drift.windows if np.isfinite(w.c) and np.isfinite(w.k_amihud)]
    assert len(finite) >= int(0.9 * len(edges))
    assert len(drift.windows) == len(edges)
    # (b) the real cross-sectional liquidity distribution shifted materially over the sample, so the
    # anchor drift is large — the recalibration governance signal, NOT inside the synthetic ceilings.
    assert drift.c_rel_range > 0.5
    assert drift.k_amihud_rel_range > 0.75
    # per-window anchors are finite positives (no NaN/inf leaking into a deployed anchor).
    for w in finite:
        assert w.c > 0.0 and math.isfinite(w.c)
        assert w.k_amihud > 0.0 and math.isfinite(w.k_amihud)
