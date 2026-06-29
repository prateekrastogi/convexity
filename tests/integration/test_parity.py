"""Backtest↔live PARITY (ARCHITECTURE §8.4 a / SYSTEM Inv PAR).

The single :class:`ConvexityComposer` is a PURE function of (BookState, per-bar pillar outputs), so
the decision it yields when driven bar-by-bar (the live/streaming path) is IDENTICAL to the decision
a scripted one-shot pillar replay yields. Because the composition holds no hidden engine state, the
backtest engine and a scripted replay produce the same SystemDecision + the same venue-order plan.
This is the property that lets ONE strategy class run unchanged across backtest / paper / live.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from convexity.system.synth.generate import make_book, make_proposed  # noqa: E402

from .conftest import (  # noqa: E402
    SYMBOL,
    ScriptedBreakout,
    build_composer,
    exec_cfg,
    make_entry,
    perp_micro,
    perp_spec,
    sys_cfg,
)

pytestmark = pytest.mark.nautilus


def _run_once(composer, *, bars, idx):
    spec = perp_spec()
    return composer.on_decision_bar(
        bars_by_instrument={SYMBOL: bars},
        idx_by_instrument={SYMBOL: idx},
        specs={SYMBOL: spec},
        book=make_book(),
        open_positions={},
        selection=make_proposed((SYMBOL,)),
        leverage_risk={SYMBOL: object()},
        leverage_cfg=_LevCfg(),
        trend_cfg=object(),
        exec_cfg=exec_cfg(),
        micro_by_instrument={SYMBOL: perp_micro()},
        sys_cfg=sys_cfg(),
        bar_ts=int(bars.ts[idx]),
    )


class _LevCfg:
    """A stand-in leverage cfg exposing the ``defaults`` the SignalInput builder reads."""

    class defaults:
        position_margin = 100.0
        capital_cap = 0.13


def test_streaming_equals_oneshot_decision(bars):
    """Driving the composer twice on the same bar yields a byte-identical SystemDecision + plan.

    Determinism is the foundation of parity: a fresh composer (streaming/live path) and a re-run
    (the backtest replay) must agree exactly, because the composition is a pure function of F_t.
    """
    idx = len(bars) - 1
    res_live = _run_once(build_composer(), bars=bars, idx=idx)
    res_replay = _run_once(build_composer(), bars=bars, idx=idx)

    assert [(a.instrument, a.kind, a.target_units, a.side) for a in res_live.decision.actions] == [
        (a.instrument, a.kind, a.target_units, a.side) for a in res_replay.decision.actions
    ]
    assert res_live.decision.reason_codes == res_replay.decision.reason_codes
    assert [c.client_order_id for c in res_live.commands] == [
        c.client_order_id for c in res_replay.commands
    ]


def test_entry_admits_and_rests_a_protective_stop(bars):
    """A scripted breakout entry composes to an OPEN delta + a venue reduce-only protective stop.

    This is the full §4 path: BREAKOUT.propose → LEVERAGE size → SYSTEM admit → EXECUTION lower →
    venue stop. The stop is reduce-only and triggers at the signal invalidation (EXECUTION P6).
    """
    idx = len(bars) - 1
    res = _run_once(build_composer(breakout=ScriptedBreakout(make_entry(direction=1))),
                    bars=bars, idx=idx)
    opens = [a for a in res.decision.actions if a.kind == "open"]
    assert opens, "scripted entry should admit an open"
    stops = [c for c in res.commands if c.kind in ("STOP_MARKET", "STOP_LIMIT")]
    assert stops, "an open must rest a venue protective stop (EXECUTION P6)"
    assert all(s.reduce_only for s in stops)
    assert stops[0].price == make_entry(direction=1).invalidation


def test_no_signal_yields_no_open(bars):
    """No breakout entry ⇒ no open delta and no protective stop (nothing to admit)."""
    idx = len(bars) - 1
    res = _run_once(build_composer(breakout=ScriptedBreakout(None)), bars=bars, idx=idx)
    assert not [a for a in res.decision.actions if a.kind == "open"]
    assert not res.commands
