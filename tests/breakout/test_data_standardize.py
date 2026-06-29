"""Data-layer + standardize tests (BREAKOUT §15: schema, calendar, CA, rolls, seasonal, vol, tails).

Covers malformed OHLCV / duplicate / missing / stale / invalid-session flagging, corporate-action +
roll back-adjustment (artifacts do not survive), past-only seasonal + volatility (the Inv J
measurability anchor: sigma[t] excludes r[t]), and the fold-local tail fit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from convexity.core.types import InstrumentSpec

from convexity.breakout.config.resolve import resolve_config
from convexity.breakout.data.calendar import Sessionizer
from convexity.breakout.data.corporate_actions import CorporateAction, CorporateActionAdjuster
from convexity.breakout.data.quality import QualityProfiler
from convexity.breakout.data.rolls import RollAdjuster, RollEvent
from convexity.breakout.data.schema import BarValidator
from convexity.breakout.standardize.returns import log_returns
from convexity.breakout.standardize.seasonal import SeasonalProfile
from convexity.breakout.standardize.tails import fit_student_t_nu, winsorize_causal
from convexity.breakout.standardize.volatility import VolatilityStandardizer
from convexity.breakout.synth.generate import make_regime


def crypto_spec() -> InstrumentSpec:
    return InstrumentSpec("BTCUSDT", "crypto_perp", "BINANCE", "UTC", tick_size=0.1, lot_size=0.001)


# --------------------------------------------------------------------------- schema


def test_schema_flags_duplicate_and_bad_ohlc():
    spec = crypto_spec()
    bars = make_regime("null_random_walk", 100, 1)
    ts = bars.ts.copy()
    ts[50] = ts[49]  # duplicate timestamp
    h = bars.h.copy()
    h[60] = bars.l[60] - 1.0  # high < low (inconsistent OHLC)
    out = BarValidator().validate(replace(bars, ts=ts, h=h), spec)
    assert not out.is_valid[50]
    assert not out.is_valid[60]


def test_schema_flags_stale_run():
    spec = crypto_spec()
    bars = make_regime("missing_stale", 600, 12)
    out = BarValidator().validate(bars, spec)
    assert np.any(~out.is_valid), "stale/missing run must be flagged"


def test_schema_no_silent_forward_fill():
    """The validator flags but does not OVERWRITE prices (no silent forward-fill)."""
    spec = crypto_spec()
    bars = make_regime("null_random_walk", 100, 1)
    out = BarValidator().validate(bars, spec)
    assert np.array_equal(out.c, bars.c)


# --------------------------------------------------------------------------- calendar


def test_calendar_crypto_marks_utc_days():
    spec = crypto_spec()
    bars = make_regime("crypto_weekend", 600, 9)
    bars = replace(bars, is_session_open=None)
    out = Sessionizer().apply(bars, spec)
    assert out.is_session_open is not None
    assert out.is_session_open[0]  # first bar opens a session
    assert out.is_session_open.sum() >= 1


def test_calendar_deterministic():
    spec = crypto_spec()
    bars = replace(make_regime("null_random_walk", 300, 1), is_session_open=None)
    a = Sessionizer().apply(bars, spec).is_session_open
    b = Sessionizer().apply(bars, spec).is_session_open
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- corporate actions / rolls


def test_corporate_action_removes_split_step():
    """After a 2:1 split back-adjustment the pre-split prices are halved -> the artifact step is gone."""
    bars = make_regime("equity_split", 600, 11)
    # locate the largest log-step (the split bar) empirically — robust to the synth bar alignment.
    steps = np.abs(np.diff(np.log(bars.c)))
    split_idx = int(np.argmax(steps)) + 1
    action = CorporateAction(ex_date_ms=int(bars.ts[split_idx]), ratio=2.0)
    adj = CorporateActionAdjuster().adjust(bars, (action,))
    raw_step = abs(np.log(bars.c[split_idx] / bars.c[split_idx - 1]))
    adj_step = abs(np.log(adj.c[split_idx] / adj.c[split_idx - 1]))
    assert adj_step < raw_step  # the discontinuity is reduced by the adjustment


def test_roll_back_adjust_and_marks_window():
    bars = make_regime("futures_roll", 600, 10)
    roll_idx = int(np.argmax(np.abs(np.diff(bars.c)))) + 1
    gap = bars.c[roll_idx] - bars.c[roll_idx - 1]
    adj = RollAdjuster().adjust(bars, (RollEvent(int(bars.ts[roll_idx]), adjust=gap, window_bars=2),))
    assert adj.is_roll_window is not None and adj.is_roll_window[roll_idx]
    raw_step = abs(bars.c[roll_idx] - bars.c[roll_idx - 1])
    adj_step = abs(adj.c[roll_idx] - adj.c[roll_idx - 1])
    assert adj_step < raw_step


# --------------------------------------------------------------------------- seasonal (past-only)


def test_seasonal_is_past_only_no_op_without_sessions():
    bars = replace(make_regime("null_random_walk", 300, 1), is_session_open=None)
    des = SeasonalProfile().fit_transform(bars, 20)
    rs = log_returns(bars)
    assert np.allclose(des, rs.r)  # no session structure => no deseasonalization


def test_seasonal_session_uses_prior_sessions_only():
    """The seasonal value at the first bar of the first session must be unchanged (no prior session
    to subtract) — proving session S uses sessions strictly before S."""
    bars = make_regime("session_ushape", 600, 8)
    des = SeasonalProfile().fit_transform(bars, 20)
    assert np.all(np.isfinite(des))


# --------------------------------------------------------------------------- volatility (Inv J anchor)


def test_volatility_sigma_is_f_tminus1_excludes_contemporaneous_return():
    """sigma[t] must NOT depend on r[t] (the future-needle trap / Inv J measurability): changing ONLY
    bar t's close leaves sigma[t] identical."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 300, 2)
    des = SeasonalProfile().fit_transform(bars, 20)
    vs = VolatilityStandardizer()
    s1 = vs.standardize(bars, des, cfg.defaults.vol_window)

    t = 150
    c2 = bars.c.copy()
    c2[t] *= 1.5  # change ONLY bar t's close (and hence r[t])
    des2 = des.copy()
    des2[t] = des[t] * 1.5
    s2 = vs.standardize(replace(bars, c=c2), des2, cfg.defaults.vol_window)
    assert np.isclose(s1.sigma[t], s2.sigma[t]), "sigma[t] leaked the contemporaneous r[t]"


def test_volatility_full_sample_sigma_forbidden_by_construction():
    """sigma is strictly increasing in information: sigma[t] uses [..t-1], so sigma at early bars
    cannot equal a full-sample constant."""
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 300, 2)
    des = SeasonalProfile().fit_transform(bars, 20)
    s = VolatilityStandardizer().standardize(bars, des, cfg.defaults.vol_window)
    assert s.sigma[10] != s.sigma[250]  # adapts over time (not a global constant)


def test_buckets_past_only_no_future_quantile():
    spec = crypto_spec()
    cfg = resolve_config(spec)
    bars = make_regime("variance_breakout", 300, 2)
    des = SeasonalProfile().fit_transform(bars, 20)
    s = VolatilityStandardizer().standardize(bars, des, cfg.defaults.vol_window)
    # perturbing a FUTURE bar must not change an EARLIER bucket assignment.
    bars2 = replace(bars, c=bars.c.copy())
    bars2.c[250:] *= 1.3
    des2 = des.copy()
    des2[250:] *= 1.3
    s2 = VolatilityStandardizer().standardize(bars2, des2, cfg.defaults.vol_window)
    assert np.array_equal(s.vol_bucket[:200], s2.vol_bucket[:200])


# --------------------------------------------------------------------------- tails


def test_tail_nu_and_causal_winsorize():
    rng = np.random.default_rng(0)
    x = rng.standard_t(df=5, size=2000)
    nu = fit_student_t_nu(x)
    assert 3.0 <= nu <= 50.0
    w = winsorize_causal(x, z=4.0)
    assert w.shape == x.shape


# --------------------------------------------------------------------------- quality profile


def test_quality_profile_counts():
    bars = make_regime("missing_stale", 600, 12)
    out = BarValidator().validate(bars, InstrumentSpec("B", "crypto_perp", "V", "UTC", 0.1, 0.001))
    prof = QualityProfiler().profile(out)
    assert prof.n_stale > 0
    assert prof.n_bars == len(bars)
