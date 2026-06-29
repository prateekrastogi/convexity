"""CROSS-MODE GAP MARKER parity (FRICTION #2): a bar gap raises the SAME per-bar flag in BOTH paths.

The data-consuming pillars gate on ONE per-bar gap signal — ``Bars.is_session_open`` (True on the
first bar after a session gap / discontinuity). BREAKOUT suppresses the cross on that bar (a
session-open jump is not a continuous breakout, ``breakout/detect/candidates._suppress_artifacts``)
and TREND applies its once-per-boundary ``gap_var`` there (``trend/estimate/volatility``). For the
pillars to behave IDENTICALLY regardless of mode, the flag must be raised the same way whether the
bars came from the stitched lake (backtest) or the live venue stream (paper/live).

This module proves the two paths agree, at the ONE shared seam — ``strategy.projection.RollingBars``,
which both modes feed through ``on_bar``:

  * live / paper projection path: a bar-TIMESTAMP DISCONTINUITY (a reconnect / missing bar) is
    auto-detected by ``RollingBars`` (``push(gap_before=None)``) and raises ``is_session_open``.
  * backtest lake-load path: ``run/data_loader.load_bars(return_gaps=True)`` returns the lake's
    ``gap_before`` bar-close timestamps, which the strategy threads via
    ``StrategyDeps.gap_close_ms_by_instrument`` so ``push(gap_before=True)`` raises the SAME flag —
    even for a stitch boundary that leaves NO timestamp hole (a 24/7 crypto source boundary), which
    pure ts-discontinuity would miss.

Both assertions land on the same ``Bars.is_session_open`` array the pillars read, so the gap handling
is demonstrably cross-mode-consistent. Pure-Python (no nautilus) except the lake-load case, which is
guarded so the file still imports in the pillar venv.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.strategy.projection import RollingBars

_MIN_MS = 60_000


# --------------------------------------------------------------------------- live-projection path


def test_live_projection_ts_discontinuity_raises_gap_flag():
    """A bar-timestamp gap (missing bar / reconnect) auto-raises is_session_open (the LIVE path)."""
    roll = RollingBars(bar_interval_ms=_MIN_MS)
    # 10 contiguous 1-min bars, then a 5-minute hole (a reconnect), then 5 more contiguous bars.
    ts = 1_700_000_000_000
    for _ in range(10):
        roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0)  # gap_before=None ⇒ auto
        ts += _MIN_MS
    ts += 5 * _MIN_MS  # the discontinuity: a 5-bar hole
    gap_idx = len(roll)
    for _ in range(5):
        roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0)
        ts += _MIN_MS

    bars = roll.as_bars()
    assert bars.is_session_open is not None
    # Exactly the bar right after the hole is flagged; every contiguous bar is not.
    assert bool(bars.is_session_open[gap_idx]) is True
    assert int(bars.is_session_open.sum()) == 1
    assert not bool(bars.is_session_open[0])  # the very first bar is never a gap bar


def test_live_projection_contiguous_feed_has_no_gap():
    """A clean on-cadence feed raises NO gap flag (the live path does not invent session boundaries)."""
    roll = RollingBars(bar_interval_ms=_MIN_MS)
    ts = 1_700_000_000_000
    for _ in range(30):
        roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0)
        ts += _MIN_MS
    bars = roll.as_bars()
    assert bars.is_session_open is not None
    assert int(bars.is_session_open.sum()) == 0


def test_explicit_lake_gap_without_ts_hole_raises_same_flag():
    """The backtest's explicit gap_before raises is_session_open even with NO ts hole (24/7 case).

    A 24/7 (crypto) source boundary can be flagged by the lake stitcher WITHOUT leaving a timestamp
    hole — the live ts-discontinuity test would miss it, so the explicit lake flag must still raise
    the SAME ``Bars.is_session_open`` the pillars read."""
    roll = RollingBars(bar_interval_ms=_MIN_MS)
    ts = 1_700_000_000_000
    for i in range(10):
        # bar 6 is flagged by the lake despite being perfectly contiguous (no ts hole).
        roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0,
                  gap_before=(i == 6))
        ts += _MIN_MS
    bars = roll.as_bars()
    assert bars.is_session_open is not None
    assert int(bars.is_session_open.sum()) == 1
    assert bool(bars.is_session_open[6]) is True


def test_explicit_flag_or_with_ts_hole_agree():
    """An explicit gap_before=False on a bar that ALSO has a real ts hole still raises the flag.

    The two signals are OR-ed: a real timestamp hole is never silently dropped because the upstream
    column happened not to mark it (the lake-load and ts-discontinuity paths agree)."""
    roll = RollingBars(bar_interval_ms=_MIN_MS)
    ts = 1_700_000_000_000
    roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0, gap_before=False)
    ts += 10 * _MIN_MS  # a real hole, but the upstream column said False
    roll.push(ts=ts, o=100.0, h=101.0, low=99.0, c=100.0, v=10.0, gap_before=False)
    bars = roll.as_bars()
    assert bool(bars.is_session_open[1]) is True


# --------------------------------------------------------------------------- pillar-consumption proof


def test_pillar_consumes_gap_flag_breakout_suppresses_cross():
    """BREAKOUT's candidate detector zeroes a cross on a gap bar — proving the seam is consumed.

    This is the load-bearing check: the flag ``RollingBars`` produces (``Bars.is_session_open``) is
    exactly the one BREAKOUT's ``_suppress_artifacts`` gates on (``detect/candidates.py``), so
    threading it cross-mode actually changes pillar behavior — it is not a dead field. We drive a raw
    candidate array straight through the suppression seam so the test is independent of the heavy
    StdSeries/geometry build."""
    from convexity.breakout.detect.candidates import CandidateGenerator
    from convexity.core.types import Bars

    n = 20
    c = np.full(n, 100.0)
    bars_with_gap = Bars(
        ts=(np.arange(n, dtype=np.int64) * _MIN_MS) + 1_700_000_000_000,
        o=c.copy(), h=c + 0.5, l=c - 0.5, c=c, v=np.full(n, 10.0),
        is_session_open=_one_hot(n, 10), is_valid=np.ones(n, dtype=bool),
    )
    bars_no_gap = Bars(
        ts=bars_with_gap.ts, o=bars_with_gap.o, h=bars_with_gap.h, l=bars_with_gap.l,
        c=bars_with_gap.c, v=bars_with_gap.v,
        is_session_open=np.zeros(n, dtype=bool), is_valid=np.ones(n, dtype=bool),
    )

    gen = CandidateGenerator()
    raw = np.ones(n, dtype=np.int8)  # pretend every bar produced a long cross
    suppressed_gap = gen._suppress_artifacts(bars_with_gap, raw.copy())
    suppressed_clean = gen._suppress_artifacts(bars_no_gap, raw.copy())
    # The cross on the gap bar (10) is zeroed; with no gap flag it survives — the flag is consumed.
    assert suppressed_gap[10] == 0
    assert suppressed_clean[10] == 1


def _one_hot(n: int, idx: int) -> np.ndarray:
    a = np.zeros(n, dtype=bool)
    a[idx] = True
    return a


# --------------------------------------------------------------------------- lake-load path (VM-only)


@pytest.mark.nautilus
def test_lake_load_threads_gap_into_same_flag(tmp_path):
    """The backtest lake-load path raises the SAME Bars.is_session_open flag as the live path.

    Builds a tiny stitched lake with a single ``gap_before=True`` row that leaves NO timestamp hole
    (the 24/7 case), runs it through ``run/data_loader.load_bars(return_gaps=True)``, and feeds the
    resulting bars + gap-close timestamps into ``RollingBars`` exactly as ``run/backtest.run_session``
    + the strategy's ``on_bar`` do — then asserts the flag lands on the flagged bar. This closes the
    loop: the lake column → loader → deps map → RollingBars → ``Bars.is_session_open`` (the pillar
    input), the SAME array the live ts-discontinuity path produces."""
    pytest.importorskip("nautilus_trader")
    import pandas as pd

    from convexity.run.data_loader import build_instrument, load_bars
    from convexity.run.session import instrument_id_str, resolve_universe

    sym = "BTCUSDT"
    n = 12
    open_time = (np.arange(n, dtype=np.int64) * _MIN_MS) + 1_700_000_000_000  # contiguous, no holes
    gap = np.zeros(n, dtype=bool)
    gap[7] = True  # a stitch boundary with NO timestamp hole (24/7 crypto source switch)
    df = pd.DataFrame({
        "open_time": open_time,
        "open": np.full(n, 100.0), "high": np.full(n, 100.5),
        "low": np.full(n, 99.5), "close": np.full(n, 100.0),
        "volume": np.full(n, 10.0), "source": ["a"] * n, "gap_before": gap,
    })
    lake = tmp_path / "stitched"
    lake.mkdir()
    df.to_parquet(lake / f"{sym}.parquet")

    spec = resolve_universe(symbols=(sym,))[sym]
    inst = build_instrument(spec)
    bars, gap_close_ms = load_bars(sym, inst, lake=lake, return_gaps=True)
    assert len(bars) == n
    # The lake gap_before row's bar-CLOSE ts (open_time + 1 min) is surfaced (no ts hole present).
    assert int(open_time[7]) + _MIN_MS in gap_close_ms
    assert len(gap_close_ms) == 1

    # Replay the strategy's on_bar gap-threading: push each loaded bar with gap_before keyed by the
    # deps map (gap_close_ms), exactly as _ConvexityStrategy.on_bar does in the backtest.
    iid = instrument_id_str(sym)
    assert iid == str(inst.id)
    roll = RollingBars(bar_interval_ms=_MIN_MS)
    for b in bars:
        ts_ms = int(b.ts_event // 1_000_000)
        roll.push(
            ts=ts_ms, o=float(b.open), h=float(b.high), low=float(b.low),
            c=float(b.close), v=float(b.volume),
            gap_before=(ts_ms in gap_close_ms),
        )
    out = roll.as_bars()
    assert out.is_session_open is not None
    # The flag lands on bar 7 (the lake-flagged bar) — and ONLY there (no spurious ts-hole gaps).
    assert int(out.is_session_open.sum()) == 1
    assert bool(out.is_session_open[7]) is True
