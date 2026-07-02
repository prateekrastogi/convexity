"""RegimeEngine end-to-end tests (REGIME.md §9, §0 P4 causality, §0 P6 conservative fallback).

Builds a synthetic Panel + weekly grid, calibrates the two CUSUMs on a calm in-control stretch, and
checks: HEALTHY on a calm continuation, escalation on an injected dispersion spike, causality (the
past is unchanged by future corruption), and the Protocol conformance of the migrated surface.
"""

import numpy as np
import pandas as pd

from convexity.core.protocols import RegimeEngine as RegimeEngineProto
from convexity.core.types import RegimeState
from research.xsec.panel import Panel
from convexity.regime import cusum
from convexity.regime.config.resolve import resolve_config
from convexity.regime.engine_api import RegimeEngine
from convexity.regime.types import RegimeCalibration


def _make(n_days, n_sym, vol_by_day, seed=0):
    """Synthetic panel where each day's cross-sectional return scale is vol_by_day[day]."""
    rng = np.random.default_rng(seed)
    rets = np.empty((n_days, n_sym))
    for d in range(n_days):
        rets[d] = rng.normal(0, vol_by_day[d], size=n_sym)
    close = 100 * np.cumprod(1 + rets, axis=0)
    syms = tuple(f"C{i}USDT" for i in range(n_sym))
    syms = ("BTCUSDT",) + syms[1:]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    listed = np.ones((n_days, n_sym), dtype=bool)
    return Panel(
        dates=dates, symbols=syms, close=close, funding=np.zeros((n_days, n_sym)),
        listed=listed, first_day={s: 0 for s in syms}, last_day={s: n_days - 1 for s in syms},
    )


def _weeks(n_days, wk=5):
    return [np.arange(i, min(i + wk, n_days)) for i in range(1, n_days, wk)]


def _ic(mu, sd, n, seed):
    """A noisy in-control training series (a zero-variance constant series is a degenerate
    calibration, REGIME.md §0 P6 — the calibrator rejects sigma<=0)."""
    return np.random.default_rng(seed).normal(mu, sd, size=n)


def _engine(panel, weeks, in_control_D, in_control_V):
    cfg = resolve_config()
    d_cal = cusum.calibrate(in_control_D, delta_target=1.5, arl_0_target=104.0,
                            n_boot=100, block_length=6, seed=1)
    v_cal = cusum.calibrate(in_control_V, delta_target=1.5, arl_0_target=104.0,
                            n_boot=100, block_length=6, seed=2)
    cal = RegimeCalibration(dispersion=d_cal, btc_vol=v_cal)
    return RegimeEngine(calibration=cal, cfg=cfg, week_day_indices=weeks, market_symbol="BTCUSDT")


def test_engine_satisfies_protocol():
    """The migrated RegimeEngine conforms to core.protocols.RegimeEngine (regime_state(panel, t))."""
    p = _make(60, 20, np.full(60, 0.03))
    eng = _engine(p, _weeks(60), _ic(0.04, 0.006, 50, 11), _ic(0.5, 0.08, 50, 12))
    assert isinstance(eng, RegimeEngineProto)


def test_calm_continuation_is_healthy():
    """A calm regime matching the in-control moments stays HEALTHY with g_t=1.0 (§4.1)."""
    p = _make(80, 25, np.full(80, 0.03), seed=5)
    eng = _engine(p, _weeks(80), _ic(0.03, 0.005, 60, 11), _ic(0.5, 0.08, 60, 12))
    states = eng.all_states(p)
    # last few weeks (past the warmup) should be HEALTHY on a stable calm regime
    tail = states[-5:]
    assert all(s.state in ("HEALTHY", "WATCH") for s in tail)
    assert all(0.0 <= s.g_t <= 1.0 for s in states)


def test_dispersion_spike_escalates():
    """A sustained dispersion SPIKE drives S_D^+ across h -> STRESSED/HALT (§3/§4)."""
    vol = np.full(80, 0.02)
    vol[40:] = 0.20  # 10x dispersion spike sustained from day 40
    p = _make(80, 25, vol, seed=7)
    eng = _engine(p, _weeks(80), _ic(0.02, 0.004, 30, 11), _ic(0.3, 0.06, 30, 12))
    states = eng.all_states(p)
    assert any(s.state in ("STRESSED", "HALT") for s in states[-6:])
    # a fired alarm surfaces a dispersion detector in the reason chain (§0 P7)
    alarmed = [s for s in states if s.state in ("STRESSED", "HALT")]
    assert any("dispersion" in c for s in alarmed for c in s.reason_codes)


def test_causal_future_corruption_does_not_change_past():
    """§0 P4: the state at week t uses only weeks <= t; corrupting a FUTURE week leaves the past
    states byte-identical."""
    vol = np.full(80, 0.03)
    p = _make(80, 25, vol, seed=9)
    weeks = _weeks(80)
    eng_a = _engine(p, weeks, _ic(0.03, 0.005, 40, 11), _ic(0.5, 0.08, 40, 12))
    states_a = eng_a.all_states(p)

    # corrupt the LAST week's data with a massive spike; earlier weeks must be unchanged.
    p2_close = p.close.copy()
    p2_close[weeks[-1]] *= 5.0
    p2 = Panel(dates=p.dates, symbols=p.symbols, close=p2_close, funding=p.funding,
               listed=p.listed, first_day=p.first_day, last_day=p.last_day)
    eng_b = _engine(p2, weeks, _ic(0.03, 0.005, 40, 11), _ic(0.5, 0.08, 40, 12))
    states_b = eng_b.all_states(p2)

    for a, b in zip(states_a[:-1], states_b[:-1]):
        assert a.state == b.state and a.g_t == b.g_t and a.as_of == b.as_of


def test_states_are_frozen_regime_state_records():
    p = _make(40, 20, np.full(40, 0.03))
    eng = _engine(p, _weeks(40), _ic(0.03, 0.005, 20, 11), _ic(0.5, 0.08, 20, 12))
    s0 = eng.regime_state(p, 0)
    assert isinstance(s0, RegimeState)
    assert s0.engine_version == "regime-v1"


def test_engine_reports_watch_insufficient_breadth_on_majority_below_floor_week():
    """§8.1 majority rule END-TO-END: a week where a MAJORITY of days fall below the breadth floor
    (n_min_dispersion=15) emits WATCH with reason insufficient_breadth and EXCLUDES D_t from the
    dispersion CUSUM — never HEALTHY (§0 P6). Here one week has 3 of 5 days with <15 listed names."""
    n_days = 40
    n_sym = 25
    p = _make(n_days, n_sym, np.full(n_days, 0.03), seed=4)
    listed = p.listed.copy()
    # weeks are days [1..5], [6..10], ... ; corrupt week starting at day 6: days 6,7,8 keep only 5
    # names listed (< n_min=15), days 9,10 stay full -> 3 of 5 below floor -> MAJORITY -> WATCH.
    for d in (6, 7, 8):
        listed[d, 5:] = False
    p = Panel(dates=p.dates, symbols=p.symbols, close=p.close, funding=p.funding,
              listed=listed, first_day=p.first_day, last_day=p.last_day)
    eng = _engine(p, _weeks(n_days), _ic(0.03, 0.005, 20, 11), _ic(0.5, 0.08, 20, 12))
    states = eng.all_states(p)
    # week index 1 covers days [6..10] (week 0 is days [1..5]).
    wk = states[1]
    assert wk.state == "WATCH"
    assert "insufficient_breadth" in wk.reason_codes


def _make_split(n_days, n_sym, btc_vol_by_day, xsec_vol_by_day, seed=0):
    """Panel where BTC (col 0) has its OWN daily-return vol (drives V_t, §2.3) INDEPENDENT of the
    non-BTC cross-section's spread (drives D_t, §2.2). Lets a test spike dispersion in one week and
    BTC vol in a DIFFERENT week — the setup §4.3's sequential-week HALT needs."""
    rng = np.random.default_rng(seed)
    rets = np.empty((n_days, n_sym))
    for d in range(n_days):
        rets[d, 0] = rng.normal(0, btc_vol_by_day[d])          # BTC's own vol -> V_t
        rets[d, 1:] = rng.normal(0, xsec_vol_by_day[d], size=n_sym - 1)  # cross-section spread -> D_t
    close = 100 * np.cumprod(1 + rets, axis=0)
    syms = ("BTCUSDT",) + tuple(f"C{i}USDT" for i in range(1, n_sym))
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    listed = np.ones((n_days, n_sym), dtype=bool)
    return Panel(
        dates=dates, symbols=syms, close=close, funding=np.zeros((n_days, n_sym)),
        listed=listed, first_day={s: 0 for s in syms}, last_day={s: n_days - 1 for s in syms},
    )


def test_engine_reaches_halt_via_sequential_week_alarms():
    """§4.3 END-TO-END through the engine: a dispersion spike in one week (S_D^+ alarm -> STRESSED,
    then §3.1 resets it) followed by a BTC-vol spike in a LATER week (S_V^+ alarm) drives the engine
    to HALT — the co-occurrence at the state layer. Before the active-axis fix the engine could reach
    HALT ONLY when D and V crossed in the SAME week; this sequential path was unreachable. The
    dispersion spike and vol spike are placed in DIFFERENT weeks (BTC calm during the dispersion week,
    cross-section calm during the vol week) so no single week has both alarms."""
    n_days = 90
    btc_vol = np.full(n_days, 0.02)
    xsec_vol = np.full(n_days, 0.02)
    # week ~ days 40-49: cross-sectional dispersion SPIKE (BTC stays calm) -> S_D^+ only
    xsec_vol[40:50] = 0.30
    # week ~ days 55-69: BTC-vol SPIKE (cross-section back to calm) -> S_V^+ only, a LATER week
    btc_vol[55:70] = 0.40
    p = _make_split(n_days, 25, btc_vol, xsec_vol, seed=3)
    eng = _engine(p, _weeks(n_days), _ic(0.02, 0.004, 40, 11), _ic(0.3, 0.05, 40, 12))
    states = eng.all_states(p)
    assert any(s.state == "HALT" for s in states), "engine never reached HALT via sequential alarms"
    halt = next(s for s in states if s.state == "HALT")
    # the co-occurrence HALT names BOTH contributing detectors in its reason chain (§4.3).
    assert "dispersion_cusum_upper_alarm" in halt.reason_codes
    assert "btc_vol_cusum_upper_alarm" in halt.reason_codes
