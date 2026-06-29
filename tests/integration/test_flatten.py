"""The OPS flatten / kill path (ARCHITECTURE §8.4 d / OPS §9.1).

When the OPS kill machine reaches HALT, the composer overrides the normal book delta with a
reduce-only flatten of every open position (the cancel-entries-then-reduce-only-flatten sequence,
OPS §9.1b). The venue-resident reduce-only stop is the standing floor; the flatten is the active
exit. A RUN directive leaves the normal composition untouched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from convexity.ops.nautilus.adapter import OpsAdapter  # noqa: E402
from convexity.ops.ops_api import build_engine as build_ops_engine  # noqa: E402
from convexity.ops.types import EquityRef  # noqa: E402
from convexity.system.synth.generate import make_book, make_proposed  # noqa: E402

from .conftest import (  # noqa: E402
    SYMBOL,
    ScriptedBreakout,
    ScriptedTrend,
    build_composer,
    exec_cfg,
    make_entry,
    open_position_view,
    perp_micro,
    perp_spec,
    sys_cfg,
)
from .test_parity import _LevCfg  # noqa: E402

pytestmark = pytest.mark.nautilus


def _compose(composer, *, open_positions, equity_ref=None, bars, heartbeat_code=0.0):
    idx = len(bars) - 1
    return composer.on_decision_bar(
        bars_by_instrument={SYMBOL: bars},
        idx_by_instrument={SYMBOL: idx},
        specs={SYMBOL: perp_spec()},
        book=make_book(),
        open_positions=open_positions,
        selection=make_proposed((SYMBOL,)),
        leverage_risk={SYMBOL: object()},
        leverage_cfg=_LevCfg(),
        trend_cfg=object(),
        exec_cfg=exec_cfg(),
        micro_by_instrument={SYMBOL: perp_micro()},
        sys_cfg=sys_cfg(),
        equity_ref=equity_ref,
        heartbeat_code=heartbeat_code,
        bar_ts=int(bars.ts[idx]),
    )


def test_run_directive_leaves_composition_untouched(bars):
    """A healthy book (RUN) keeps the normal plan and a 'run' directive."""
    composer = build_composer(with_ops=True)
    res = _compose(composer, open_positions={}, equity_ref=EquityRef(equity=1000.0, peak=1000.0),
                   bars=bars)
    assert res.directive.state == "run"
    assert res.flatten is None


def test_halt_flattens_every_open_position_reduce_only(bars):
    """A catastrophic drawdown (>45%) drives OPS to HALT ⇒ reduce-only flatten of all open positions.

    The composer replaces the normal plan with one reduce-only MARKET exit per open position; the
    venue reduce-only stop remains the standing floor (OPS §9.1b).
    """
    composer = build_composer(
        breakout=ScriptedBreakout(make_entry()),
        trend=ScriptedTrend("continue"),
        with_ops=True,
    )
    view = open_position_view(side=1, qty=2.0)
    # equity 50% below peak ⇒ above the OPS 45% DD-kill threshold (OPS §7.1a).
    res = _compose(composer, open_positions={view.instrument: view},
                   equity_ref=EquityRef(equity=500.0, peak=1000.0), bars=bars)
    assert res.directive.state == "halt"
    assert res.flatten is not None and res.flatten.reduce_only
    assert res.commands, "HALT must emit a flatten command per open position"
    assert all(c.reduce_only for c in res.commands)
    assert all(c.reason_codes == ("ops_halt_flatten",) for c in res.commands)
    assert "ops_halt" in res.reason_codes


def test_degrade_suppresses_new_entries_but_keeps_managing_incumbents(bars):
    """A DEGRADE directive (stale feed) cancels NEW-signal entry orders, not incumbent management.

    OPS §7 DEGRADE: stop NEW entries, hold/manage existing. A stale heartbeat (HB_STALE=2.0) drives
    OPS to DEGRADE with ``cancel_entries`` set but no halt; the composer must drop the new-admission
    orders (entry slices + the fresh protective stop) while leaving any incumbent-driven commands.
    Regression guard: the entry orders carry the ``new_signal_admission`` tag the filter keys on, so
    this path is no longer a silent no-op.
    """
    composer = build_composer(
        breakout=ScriptedBreakout(make_entry()),
        trend=ScriptedTrend("continue"),
        with_ops=True,
    )
    # Healthy equity (no DD-kill), but a stale feed ⇒ DEGRADE not HALT.
    res = _compose(composer, open_positions={}, equity_ref=EquityRef(equity=1000.0, peak=1000.0),
                   bars=bars, heartbeat_code=2.0)
    assert res.directive.state == "degrade"
    assert res.flatten is None  # degrade never flattens
    assert "ops_cancel_entries" in res.reason_codes
    # the scripted breakout admitted an open, but every command for it must be suppressed.
    assert res.commands == (), "DEGRADE must cancel the new-signal entry orders + their fresh stop"


def test_degrade_with_new_entry_off_suppresses_nothing(bars):
    """DEGRADE with NO new signal leaves the (empty) plan untouched — the filter is admission-scoped."""
    composer = build_composer(breakout=ScriptedBreakout(None), trend=ScriptedTrend("continue"),
                              with_ops=True)
    res = _compose(composer, open_positions={}, equity_ref=EquityRef(equity=1000.0, peak=1000.0),
                   bars=bars, heartbeat_code=2.0)
    assert res.directive.state == "degrade"
    assert res.commands == ()


def test_flatten_command_is_cancel_entries_first():
    """The OPS flatten directive cancels working ENTRY orders before the reduce-only flatten (§9.1b)."""
    adapter = OpsAdapter(build_ops_engine())
    directive, flatten = adapter.on_bar(equity=EquityRef(equity=500.0, peak=1000.0))
    assert directive.state == "halt"
    assert flatten is not None
    assert flatten.cancel_entries_first is True
    assert flatten.reduce_only is True
