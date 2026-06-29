"""END-TO-END OPS safety-net wiring, driven through the live STRATEGY (OPS §7/§9, ARCHITECTURE §4).

Unlike ``test_backtest_end_to_end.py`` (which proves the OPS overlay is *reachable* by driving the
composer directly) this suite drives the live ``_ConvexityStrategy``'s OWN ``on_bar``/``_decide_for``
path — the production seam — and asserts the OPS engine actually escalates RUN→DEGRADE→HALT off the
reads the strategy feeds it (a live ``EquityRef`` from the Portfolio, the heartbeat/staleness code,
the reconciliation flag). These tests FAIL against the pre-fix wiring, where ``_decide_for`` called
``composer.on_decision_bar`` WITHOUT ``equity_ref``/``recon_mismatch``/``heartbeat_code`` — so OPS saw
equity=None / recon=False / heartbeat=0.0 and 7.1/7.5/7.6 could never fire in production.

  * DD-kill (§7.1): a peak-to-trough drawdown past ``DD_kill`` (45%) ⇒ HALT ⇒ the node submits
    reduce-only FLATTEN orders against the open position.
  * Stale heartbeat (§7.6): a stale-feed code ⇒ DEGRADE ⇒ new-signal entries are suppressed (the
    incumbent + its venue reduce-only stop are untouched).
  * §9.1b: HALT cancels WORKING ENTRY orders before flattening (a resting entry can't fill mid-flatten).
  * N-pair de-dup: a re-proposing breakout submits the same deterministic entry coid at most once
    within one bar-clock (Nautilus rejects duplicate client_order_ids).
"""

from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

from decimal import Decimal  # noqa: E402

from convexity.run.backtest import run_backtest  # noqa: E402


def _open_count(engine):
    return len(list(engine.cache.positions_open()))


def _resting_reduce_only(engine):
    out = []
    for inst in engine.cache.instruments():
        for o in engine.cache.orders_open(instrument_id=inst.id):
            if getattr(o, "is_reduce_only", False):
                out.append(o)
    return out


def test_dd_kill_halt_submits_reduce_only_flatten_through_the_node():
    """A drawdown past DD_kill (45%), fed through the strategy's live equity read, HALTs the node and
    submits reduce-only FLATTEN orders against the open position (OPS §7.1 → §9.1b).

    The equity trajectory is driven by an on_bar hook WHILE the engine is connected: equity holds at
    the peak until the entry has opened (after ``fire_after_idx``), then crashes 60% to trip §7.1, so
    the HALT flatten orders actually reach the live venue (not a disconnected post-run exec client).
    Pre-fix (``equity_ref`` never passed) the DD-kill could never fire and nothing flattens."""
    crash_after = 40  # well after the entry (fire_after_idx=20) has opened the position

    def _hook(strategy, _bar):
        if strategy.bars_seen < crash_after:
            strategy.set_equity_override(1000.0)   # establish + hold the running-max peak
        else:
            strategy.set_equity_override(400.0)    # 60% drawdown > DD_kill (45%)

    result = run_backtest(n_bars=72, n_pairs=1, warmup_bars=16, fire_after_idx=20,
                          on_bar_hook=_hook)
    strat, engine = result.strategy, result.engine
    try:
        # HALT submitted reduce-only flatten orders through the node (the §9.1b reduce-only exits).
        assert strat.reduces_submitted > 0, (
            "DD-kill HALT must submit reduce-only flatten orders through the node"
        )
        assert strat.flatten_outcomes_reported > 0, "the flatten outcome must be relayed (§9.1c)"
        # The flatten actually reached the venue: a reduce-only EXIT order (no trigger price, i.e. the
        # MARKET flatten — NOT the resting STOP) was submitted, and the book is flat by run end.
        reduce_exits = [
            o for o in engine.cache.orders()
            if getattr(o, "is_reduce_only", False) and getattr(o, "trigger_price", None) is None
        ]
        assert reduce_exits, "a reduce-only MARKET flatten order must have been submitted to the venue"
        assert _open_count(engine) == 0, "the HALT flatten must leave the book flat"
    finally:
        engine.dispose()


def test_stale_heartbeat_degrade_suppresses_new_entries_through_the_node():
    """A stale-feed heartbeat code, read by the strategy, DEGRADEs the node: fresh breakout entries are
    suppressed (OPS §7.6 → DEGRADE). A re-proposing breakout offers a NEW entry every bar; once the
    hook latches the feed stale, the node must stop admitting them — pre-fix (heartbeat never passed)
    the directive stays RUN and entries keep flowing."""
    from convexity.ops.streams.heartbeat import HB_OK, HB_STALE
    from convexity.run.backtest import build_synthetic_composer

    def _factory(*, mids, fire_after_idx):
        composer, breakout = build_synthetic_composer(mids=mids, fire_after_idx=0)
        breakout._fired = _NeverFired()  # never mark fired ⇒ a fresh entry proposed every bar
        return composer, breakout

    stale_from = 30  # latch the feed stale from this bar on (after warmup + first entries)
    state = {"entries_at_stale": None}

    def _hook(strategy, _bar):
        strategy.set_equity_override(1_000_000.0)  # no drawdown — isolate the §7.6 trigger
        if strategy.bars_seen >= stale_from:
            strategy.set_heartbeat_code(HB_STALE)
            if state["entries_at_stale"] is None:
                state["entries_at_stale"] = strategy.entries_submitted
        else:
            strategy.set_heartbeat_code(HB_OK)

    result = run_backtest(n_bars=60, n_pairs=1, warmup_bars=16, fire_after_idx=0,
                          composer_factory=_factory, on_bar_hook=_hook)
    strat, engine = result.strategy, result.engine
    try:
        assert state["entries_at_stale"] is not None, "the stale-feed hook must have latched"
        # The OPS kill machine ACTUALLY escalated to DEGRADE off the heartbeat code the strategy fed
        # it (pre-fix the code never reached OPS, so the directive stayed RUN — degrade_bars==0).
        assert strat.degrade_bars > 0, (
            "the stale feed must drive the OPS directive to DEGRADE in the live path"
        )
        assert strat.halt_bars == 0, "a stale feed is DEGRADE, never HALT (no flatten)"
        # Entries flowed BEFORE the feed went stale (the breakout path is live), then STOPPED once
        # DEGRADE latched — no new entry admitted across the whole stale window.
        assert state["entries_at_stale"] > 0, "entries must flow while the feed is healthy (RUN)"
        assert strat.entries_submitted == state["entries_at_stale"], (
            "DEGRADE must suppress every new-signal entry once the feed is stale"
        )
        # The incumbent's venue reduce-only protective stop is untouched by DEGRADE.
        assert _resting_reduce_only(engine), "the incumbent's venue reduce-only stop must remain"
    finally:
        engine.dispose()


def test_halt_cancels_working_entries_before_flatten():
    """OPS §9.1b: a working (resting, NON-reduce-only) entry is CANCELED before the reduce-only flatten
    so it cannot fill mid-flatten. A hook rests a deep passive limit entry early (it rests, never
    fills), then crashes equity to HALT the node — all while the engine is connected."""
    rest_at = 30        # rest the working entry once warmed up + the position is open
    crash_after = 45    # HALT after the entry is resting
    state = {"working_coid": None}

    def _hook(strategy, bar):
        iid_str = str(bar.bar_type.instrument_id)
        if strategy.bars_seen < crash_after:
            strategy.set_equity_override(1000.0)
        else:
            strategy.set_equity_override(400.0)  # 60% drawdown > DD_kill
        if strategy.bars_seen == rest_at and state["working_coid"] is None:
            from nautilus_trader.model.enums import OrderSide
            instrument = strategy._instruments[iid_str]
            working = strategy.order_factory.limit(
                instrument_id=instrument.id, order_side=OrderSide.BUY,
                quantity=instrument.make_qty(Decimal("0.001")),
                price=instrument.make_price(Decimal("100.0")),  # deep below ⇒ rests, never fills
                post_only=True, reduce_only=False,
            )
            strategy.submit_order(working)
            state["working_coid"] = working.client_order_id

    result = run_backtest(n_bars=72, n_pairs=1, warmup_bars=16, fire_after_idx=20,
                          on_bar_hook=_hook)
    strat, engine = result.strategy, result.engine
    try:
        assert state["working_coid"] is not None, "the working entry must have been rested"
        assert strat.entries_canceled_on_halt > 0, (
            "HALT must cancel the working entry order (§9.1b: cancel entries first)"
        )
        # The working entry is no longer a live resting order (canceled by the §9.1b sequence).
        order = strat.cache.order(state["working_coid"])
        assert order is not None and order.is_closed, (
            "the working entry must be canceled before the flatten (not left resting)"
        )
        assert strat.reduces_submitted > 0, "the reduce-only flatten must follow the cancels"
    finally:
        engine.dispose()


def test_repeated_proposal_within_one_bar_clock_submits_entry_at_most_once():
    """N-pair de-dup: the SAME deterministic entry client_order_id, proposed twice within one
    bar-clock before the first fill, is submitted at most once (Nautilus rejects duplicate coids).

    A hook fires a SECOND ``_decide_for`` on the same bar-clock at one chosen bar (the per-instrument
    re-run the N-pair ``on_bar`` loop does) while the engine is connected. With a breakout that
    re-proposes every bar, pre-fix this submits the deterministic entry coid twice → a duplicate-coid
    DENY; the guard makes it submit at most once with no deny."""
    from convexity.run.backtest import build_synthetic_composer

    def _factory(*, mids, fire_after_idx):
        composer, breakout = build_synthetic_composer(mids=mids, fire_after_idx=0)
        breakout._fired = _NeverFired()  # re-proposes every call
        return composer, breakout

    double_at = 18  # just past warmup; the breakout is offering a fresh entry
    state = {"before": None, "after": None}

    def _hook(strategy, bar):
        strategy.set_equity_override(1_000_000.0)  # no drawdown — isolate the de-dup
        if strategy.bars_seen == double_at and state["before"] is None:
            iid_str = str(bar.bar_type.instrument_id)
            state["before"] = strategy.entries_submitted
            # The FIRST _decide_for for this bar-clock (mirrors on_bar's call); on_bar will then make
            # the SECOND against the SAME bar_ts — exactly the N-pair within-bar re-proposal.
            strategy._decide_for(iid_str, bar)
            state["after"] = strategy.entries_submitted

    result = run_backtest(n_bars=24, n_pairs=1, warmup_bars=16, fire_after_idx=0,
                          composer_factory=_factory, on_bar_hook=_hook)
    strat, engine = result.strategy, result.engine
    try:
        assert state["before"] is not None, "the double-decide hook must have run"
        # Across the FIRST decide (hook) + the SECOND decide (on_bar) on the same bar-clock, at most
        # one entry was submitted.
        first_pass = state["after"] - state["before"]
        assert first_pass <= 1, f"first decide must submit at most one entry, got {first_pass}"
        # No order was DENIED for a duplicate client_order_id (the failure the guard prevents).
        denied = [
            o for o in engine.cache.orders()
            if "DENIED" in str(getattr(o, "status", ""))
        ]
        assert not denied, f"no entry order may be denied for a duplicate coid, got {len(denied)}"
        # Sanity: entries did open (the breakout path is live, not dead).
        assert strat.entries_submitted >= 1, "the re-proposing breakout must submit at least one entry"
    finally:
        engine.dispose()


def test_latched_halt_keeps_suppressing_new_entries_across_bars():
    """OPS §7 HALT = "stop ALL": once HALTed, a re-proposing breakout must NOT leak a new entry on the
    LATCHED HALT bars that follow the single §9.1a flatten wave. The flatten latch suppresses
    RE-FLATTEN, never RE-ENTRY — a regression guard for the bug where a latched HALT bar (flatten
    already issued ⇒ ``flatten is None``) fell through to the normally-composed plan and re-admitted
    entries. The book must end flat and entries must stop the moment HALT latches."""
    from convexity.run.backtest import build_synthetic_composer

    def _factory(*, mids, fire_after_idx):
        composer, breakout = build_synthetic_composer(mids=mids, fire_after_idx=0)
        breakout._fired = _NeverFired()  # re-proposes a fresh entry EVERY bar (the adversarial case)
        return composer, breakout

    crash_after = 30  # HALT well before the run ends so many latched HALT bars follow the flatten
    state = {"entries_at_halt": None}

    def _hook(strategy, _bar):
        if strategy.bars_seen < crash_after:
            strategy.set_equity_override(1000.0)
        else:
            strategy.set_equity_override(400.0)  # 60% drawdown > DD_kill (45%) ⇒ HALT, stays halted
            if state["entries_at_halt"] is None and strategy.halt_bars > 0:
                state["entries_at_halt"] = strategy.entries_submitted

    result = run_backtest(n_bars=80, n_pairs=1, warmup_bars=16, fire_after_idx=0,
                          composer_factory=_factory, on_bar_hook=_hook)
    strat, engine = result.strategy, result.engine
    try:
        assert strat.halt_bars > 1, "the test must exercise MULTIPLE latched HALT bars"
        assert state["entries_at_halt"] is not None, "the HALT must have latched mid-run"
        # No NEW entry may be admitted on any bar at/after the HALT latched — the re-proposing breakout
        # is offering one every bar, so a leak would bump entries_submitted past the latch snapshot.
        assert strat.entries_submitted == state["entries_at_halt"], (
            "a latched HALT must keep suppressing new entries (the flatten latch suppresses re-flatten, "
            "not re-entry)"
        )
        assert _open_count(engine) == 0, "the HALT flatten must leave the book flat"
    finally:
        engine.dispose()


class _NeverFired:
    """A set-like that never records membership ⇒ the synthetic breakout re-proposes every bar."""

    def add(self, _item) -> None:  # noqa: D102
        return None

    def __contains__(self, _item) -> bool:  # noqa: D102
        return False
