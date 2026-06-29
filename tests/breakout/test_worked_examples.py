"""BREAKOUT worked-example regression fixtures (BUILD_PLAN Phase 2 worked examples).

Load-bearing numeric examples, each driven from the seeded synthetic generator:
  * strict-pre-bar barrier identity + inclusive-max-never-fires (the gate assertion strict>0, incl==0),
  * same-k 1-tick-vs-kσ on a REALISTIC liquid fixture (1 tick << k*sigma) for crypto AND equity,
  * entry-timestamp entry_idx == signal_idx + 1,
  * score-scale invariance (c = 7),
  * future-needle,
  * cross-sectional FDR on a noise universe,
  * convexity_ratio formula = median MFE_R / max(|median MAE_R|, eps).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from convexity.core.types import InstrumentSpec

from convexity.breakout.config.resolve import resolve_config
from convexity.breakout.detect.candidates import CandidateGenerator
from convexity.breakout.features.range_geometry import (
    RangeGeometry,
    rolling_max_strict_prebar,
    rolling_min_strict_prebar,
)
from convexity.breakout.pipeline import PipelineOrchestrator
from convexity.breakout.standardize.seasonal import SeasonalProfile
from convexity.breakout.standardize.volatility import VolatilityStandardizer
from convexity.breakout.synth.generate import make_regime, noise_universe
from convexity.breakout.types import EntryLabel


def crypto_spec() -> InstrumentSpec:
    return InstrumentSpec("BTCUSDT", "crypto_perp", "BINANCE", "UTC", tick_size=0.01, lot_size=0.001)


def equity_spec() -> InstrumentSpec:
    return InstrumentSpec(
        "AAPL", "us_equity", "NASDAQ", "America/New_York", tick_size=0.01, lot_size=1.0,
        calendar_id="us_equity_rth",
    )


def _std(bars, spec, cfg):
    des = SeasonalProfile().fit_transform(bars, 20)
    return VolatilityStandardizer(
        n_vol_buckets=cfg.defaults.n_vol_buckets, n_liq_buckets=cfg.defaults.n_liq_buckets
    ).standardize(bars, des, cfg.defaults.vol_window)


# --------------------------------------------------------------------------- strict-pre-bar barrier


def test_strict_prebar_barrier_identity_and_gate():
    """upper_barrier[t] == rolling_max(h[t-W .. t-1]); strict barrier fires > 0, inclusive == 0."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    W = min(cfg.windows)

    upper = rolling_max_strict_prebar(bars.h, W)
    lower = rolling_min_strict_prebar(bars.l, W)

    # identity: out[t] equals max over the window ending at t-1.
    for t in range(W + 1, len(bars)):
        assert np.isclose(upper[t], np.max(bars.h[t - W : t]))
        assert np.isclose(lower[t], np.min(bars.l[t - W : t]))

    strict_fires = int(np.sum(np.isfinite(upper) & (bars.c > upper)))
    incl = np.array([np.max(bars.h[max(0, t - W + 1) : t + 1]) for t in range(len(bars))])
    incl_fires = int(np.sum(bars.c > incl))
    assert strict_fires > 0  # GATE
    assert incl_fires == 0  # GATE


# --------------------------------------------------------------------------- same-k 1-tick vs kσ


def _one_tick_vs_ksigma(spec: InstrumentSpec):
    cfg = resolve_config(spec)
    # a realistic liquid path: per-bar sigma ~ a few bps of price, so 1 tick (0.01 on a 100 price) is
    # a tiny fraction of k*sigma_price. Pick a bar with a finite barrier and NO pre-existing active
    # long zone (so the same-zone suppressor does not mask the thrust).
    bars = make_regime("null_random_walk", 400, 1)
    s = _std(bars, spec, cfg)
    geom = RangeGeometry(family="donchian").compute(bars, s, cfg.windows)
    gen = CandidateGenerator(cross_k=cfg.cross_k)

    # a bar whose close sits inside the barrier band (no active zone) with a finite upper barrier.
    t = None
    for cand_t in range(150, 350):
        u, low = geom.upper_barrier[cand_t], geom.lower_barrier[cand_t]
        if np.isfinite(u) and np.isfinite(low) and low < bars.c[cand_t] < u:
            t = cand_t
            break
    assert t is not None
    upper_t = geom.upper_barrier[t]
    sigma_price = s.sigma[t] * bars.c[t]
    assert spec.tick_size < cfg.cross_k * sigma_price, "fixture must be liquid (1 tick << k*sigma)"

    c1 = bars.c.copy()
    c1[t] = upper_t + spec.tick_size  # 1-tick poke
    assert gen.candidates(replace(bars, c=c1), s, geom)[t] == 0

    c2 = bars.c.copy()
    c2[t] = upper_t + 3.0 * cfg.cross_k * sigma_price
    assert gen.candidates(replace(bars, c=c2), s, geom)[t] == 1
    return cfg.cross_k


def test_same_k_crypto_and_equity():
    """The SAME k governs the 1-tick-vs-kσ behaviour on a crypto AND an equity fixture (no per-symbol k)."""
    k_crypto = _one_tick_vs_ksigma(crypto_spec())
    k_equity = _one_tick_vs_ksigma(equity_spec())
    assert k_crypto == k_equity


# --------------------------------------------------------------------------- entry timestamp


def test_entry_idx_equals_signal_idx_plus_one():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    sigs = PipelineOrchestrator().run_batch(make_regime("drift_breakout", 400, 1), spec, cfg)
    assert sigs
    assert all(s.entry_idx == s.signal_idx + 1 for s in sigs)


# --------------------------------------------------------------------------- score-scale invariance


def test_score_scale_invariance_c7():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    po = PipelineOrchestrator()
    base = po.run_batch(bars, spec, cfg)
    c = 7.0
    cfg2 = replace(
        cfg,
        weights={k: c * v for k, v in cfg.weights.items()},
        defaults=replace(cfg.defaults, quality_threshold=c * cfg.defaults.quality_threshold),
    )
    scaled = po.run_batch(bars, spec, cfg2)
    assert [(s.signal_idx, s.direction) for s in base] == [(s.signal_idx, s.direction) for s in scaled]


# --------------------------------------------------------------------------- future-needle


def test_future_needle():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("drift_breakout", 500, 1)
    po = PipelineOrchestrator()
    base = po.run_batch(bars, spec, cfg)
    tamper = 380
    c2 = bars.c.copy()
    c2[tamper:] *= 1.2
    after = po.run_batch(replace(bars, c=c2), spec, cfg)
    b = {(s.signal_idx, round(s.quality_score, 9)) for s in base if s.signal_idx < tamper}
    a = {(s.signal_idx, round(s.quality_score, 9)) for s in after if s.signal_idx < tamper}
    assert a == b


# --------------------------------------------------------------------------- cross-sectional FDR


def test_cross_sectional_fdr_noise_universe():
    """On a pure-noise universe, the FDR gate approves far fewer symbols than a naive uncorrected
    pass (Invariant L). Uses real noise instruments + a uniform-null p-value proxy."""
    from convexity.breakout.tune.promotion import promote_universe

    universe = noise_universe(40, n=300, seed=7)
    assert len(universe) == 40
    rng = np.random.default_rng(7)
    provisional = [(f"S{i}", "approved", float(rng.uniform())) for i in range(40)]
    out = promote_universe(provisional, fdr_alpha=0.10)
    n_app = sum(v == "approved" for v in out.values())
    naive = sum(1 for _, _, p in provisional if p <= 0.10)
    assert n_app <= naive  # BH never approves MORE than the naive count
    assert n_app <= 8  # and far fewer than the 40 'approved' provisionals


# --------------------------------------------------------------------------- convexity_ratio formula


def test_convexity_ratio_formula():
    from convexity.breakout.validate.stats import convexity_ratio

    labels = [
        EntryLabel(0, 1, 30, mfe_r=m, mae_r=a, hit_plus_1r_before_stop=True,
                   hit_plus_2r_before_stop=False, hit_stop_first=False, time_to_plus_1r=5,
                   false_breakout=False, net_return_to_horizon_bps=0.0)
        for m, a in [(2.0, 0.5), (3.0, 1.0), (1.0, 0.5), (4.0, 2.0)]
    ]
    # median MFE = median(2,3,1,4) = 2.5 ; median MAE = median(0.5,1,0.5,2) = 0.75
    # ratio = 2.5 / max(0.75, eps) = 3.3333...
    assert np.isclose(convexity_ratio(labels), 2.5 / 0.75)
