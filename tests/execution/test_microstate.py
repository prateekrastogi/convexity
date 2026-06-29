"""Microstate-layer tests: causality, √horizon vol, fill-prob 3-rung fallback + monotonicity (§5/§11)."""

import numpy as np
import pytest

from convexity.core.types import Bars
from convexity.execution.config.resolve import resolve_config
from convexity.execution.microstate.fill_prob import (
    FillProbability,
    PassiveFillRecord,
    logistic_fill_curve,
)
from convexity.execution.microstate.spread import SpreadEstimator
from convexity.execution.microstate.vol import HorizonVol
from convexity.execution.microstate.volume_proxy import VolumeProxy
from convexity.execution.synth.generate import perp_instrument


def _bars(n=100, seed=3):
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return Bars(
        ts=np.arange(n, dtype=np.int64) * 60_000, o=c, h=c * 1.001, l=c * 0.999, c=c,
        v=np.full(n, 1000.0),
    )


def test_vol_sqrt_horizon_scaling():
    """Vol scales by sqrt(horizon): doubling horizon multiplies the estimate by sqrt(2) (§11)."""
    cfg = resolve_config(perp_instrument())
    bars = _bars()
    hv = HorizonVol(window=30)
    v30 = hv.estimate(bars, 80, 30.0, cfg)
    v60 = hv.estimate(bars, 80, 60.0, cfg)
    assert v60 == pytest.approx(v30 * np.sqrt(2.0), rel=1e-9)


def test_vol_causal_no_lookahead():
    """The vol at idx is unchanged when bars AFTER idx are perturbed (causality, Invariant L-style)."""
    cfg = resolve_config(perp_instrument())
    bars = _bars()
    hv = HorizonVol(window=30)
    before = hv.estimate(bars, 50, 30.0, cfg)
    c2 = bars.c.copy()
    c2[51:] += 50.0  # perturb the future only
    bars2 = Bars(ts=bars.ts, o=bars.o, h=bars.h, l=bars.l, c=c2, v=bars.v)
    after = hv.estimate(bars2, 50, 30.0, cfg)
    assert before == pytest.approx(after)


def test_vol_nan_inf_free_on_flat_series():
    """A flat series yields a finite positive vol floor, never NaN/0 (§15 microstate gate)."""
    cfg = resolve_config(perp_instrument())
    n = 50
    c = np.full(n, 100.0)
    bars = Bars(ts=np.arange(n, dtype=np.int64) * 60_000, o=c, h=c, l=c, c=c, v=np.full(n, 1.0))
    v = HorizonVol().estimate(bars, n - 1, 30.0, cfg)
    assert np.isfinite(v) and v > 0.0


def test_volume_proxy_positive_floor_no_volume():
    """No-volume feed -> a positive floor (impact participation never divides by zero) (§11)."""
    cfg = resolve_config(perp_instrument())
    n = 30
    c = np.full(n, 100.0)
    bars = Bars(ts=np.arange(n, dtype=np.int64) * 60_000, o=c, h=c, l=c, c=c, v=None)
    assert VolumeProxy().estimate(bars, n - 1, 60.0, cfg) > 0.0


def test_fill_prob_monotone_in_offset():
    """p_fill is monotone DECREASING in offset (§3 / fill-prob gate)."""
    curve = logistic_fill_curve(base_rate=0.9, sigma_bps=8.0, tick_bps=0.5)
    vals = [curve(k, 30.0, 0.0) for k in (0.0, 1.0, 2.0, 3.0, 5.0)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_fill_prob_decreasing_in_g_increasing_in_sigma():
    """p_fill DECREASES in g and INCREASES in sigma (§3 / Invariant J)."""
    c_lowsig = logistic_fill_curve(base_rate=0.9, sigma_bps=3.0, tick_bps=0.5)
    c_hisig = logistic_fill_curve(base_rate=0.9, sigma_bps=12.0, tick_bps=0.5)
    assert c_lowsig(2.0, 30.0, 3.0) <= c_lowsig(2.0, 30.0, 0.0)  # higher g -> lower fill
    assert c_hisig(2.0, 30.0, 0.0) >= c_lowsig(2.0, 30.0, 0.0)  # higher sigma -> higher fill


def test_fill_prob_bounded_unit_interval():
    """p_fill is bounded in [0,1] for any input (§3 / fill-prob gate)."""
    curve = logistic_fill_curve(base_rate=1.0, sigma_bps=8.0, tick_bps=0.5)
    for k in (-5.0, 0.0, 100.0):
        for g in (-5.0, 0.0, 50.0):
            p = curve(k, 30.0, g)
            assert 0.0 <= p <= 1.0


def test_fill_prob_three_rung_fallback():
    """Cold-start (< N_red own fills) -> class_prior rung with maker GATED OFF (§5.5 #2)."""
    cfg = resolve_config(perp_instrument())
    fp = FillProbability()
    # No own fills at all -> class_prior, cold_start True.
    r = fp.curve(None, "b", cfg, as_of_ts=10, sigma_bps=8.0, tick_bps=0.03)
    assert r.rung == "class_prior" and r.cold_start

    # Seed enough resolved bucket fills -> bucket rung, not cold.
    for _ in range(cfg.defaults.n_bucket + 5):
        fp.admit(PassiveFillRecord("b", 1.0, 0.5, filled=True, outcome_close_ts=0))
    r2 = fp.curve(None, "b", cfg, as_of_ts=10, sigma_bps=8.0, tick_bps=0.03)
    assert r2.rung == "bucket" and not r2.cold_start


def test_fill_prob_global_rung():
    """Below N_bucket but >= N_global pooled fills -> the GLOBAL rung (§5.5 #2)."""
    cfg = resolve_config(perp_instrument())
    fp = FillProbability()
    # Many fills in OTHER buckets (so the target bucket is thin but global is rich).
    for i in range(cfg.defaults.n_global + 5):
        fp.admit(PassiveFillRecord(f"other{i % 7}", 1.0, 0.5, filled=True, outcome_close_ts=0))
    r = fp.curve(None, "target", cfg, as_of_ts=10, sigma_bps=8.0, tick_bps=0.03)
    assert r.rung == "global"


def test_spread_class_proxy_floor_when_no_data():
    """With no usable bars the spread ladder returns the class-proxy floor, finite positive (§11)."""
    cfg = resolve_config(perp_instrument())
    c = np.array([100.0, 100.0])
    bars = Bars(ts=np.array([0, 60_000], dtype=np.int64), o=c, h=c, l=c, c=c, v=None)
    out = SpreadEstimator(window=30).estimate(bars, 1, cfg)
    assert out.half_spread_bps == pytest.approx(cfg.class_proxy_half_spread_bps)
    assert out.rung == "class_proxy"
