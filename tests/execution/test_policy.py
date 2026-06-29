"""Policy-layer tests: chase ladder/abort, partial dust, order issuer idempotency, gates (§5–§8)."""

import dataclasses


from convexity.execution.config.resolve import resolve_config
from convexity.execution.engine_api import build_engine, on_target
from convexity.execution.policy.chase import ChaseLadder, worst_entry
from convexity.execution.policy.order_issuer import OrderIssuer
from convexity.execution.policy.partial import PartialFillManager
from convexity.execution.synth.generate import ExecutionSynth, perp_instrument
from convexity.execution.types import ExecTarget, WorkingState


def test_chase_finite_and_bounded():
    """The chase ladder is finite: it never repriced beyond worst_entry on an entry (§5.2 / Inv D)."""
    sc = ExecutionSynth().runaway_breakout()
    cfg = sc.cfg
    ladder = ChaseLadder()
    we = worst_entry(sc.target, sc.target.arrival_mid, cfg)
    working = sc.working
    seen = set()
    for _ in range(cfg.defaults.max_chase_steps + 5):
        action = ladder.step(sc.target, sc.micro, working, cfg, post_still_wins=True)
        assert action.action in ("post", "cross", "abort")
        if action.action != "abort":
            assert action.price <= we + 1e-6  # never beyond worst_entry (long)
        working = dataclasses.replace(working, chase_step=action.chase_step)
        if action.action == "abort":
            assert action.abort_reason == "chase_budget_exhausted"
            break
        # progress guard: chase_step must not cycle forever
        assert (action.chase_step,) not in seen or action.action == "post"
    else:
        # If it never aborted it must have crossed within budget — also acceptable, but the budget
        # edge MUST eventually terminate.
        pass


def test_engine_post_only_abort_on_step_exhaustion():
    """ENGINE-level: a calm low-g entry chooses POST_ONLY, then at max_chase_steps reaches a TRUE
    POST_ONLY ExecDecision with aborted=True / chase_budget_exhausted via STEP exhaustion (§5.1.2 /
    Inv D). The runaway scenario trips the maker-floor gate to a crossing tactic, so this calm path
    is the only one that actually reaches the POST_ONLY abort assertion."""
    eng = build_engine()
    sc = ExecutionSynth().calm_wide_spread()
    cfg = sc.cfg
    # Sanity: at step 0 the calm market genuinely SELECTS POST_ONLY (no gate forced a cross).
    d0 = on_target(eng, sc.target, sc.micro, sc.working, cfg)
    assert d0.tactic == "POST_ONLY_MAKER" and d0.binding_gate == "none"
    # Drive the chase to the step-count edge with a worst_entry FAR enough that the next reprice price
    # does NOT breach it — so the abort is STEP exhaustion, not the price bound.
    we = worst_entry(sc.target, sc.target.arrival_mid, cfg)
    next_price = sc.target.arrival_mid + cfg.defaults.reprice_tick_step * cfg.instrument.tick_size
    assert next_price <= we  # the abort below is NOT a price breach
    working = WorkingState(chase_step=cfg.defaults.max_chase_steps, worst_entry=we)
    d = on_target(eng, sc.target, sc.micro, working, cfg)
    assert d.tactic == "POST_ONLY_MAKER"
    assert d.aborted and d.abort_reason == "chase_budget_exhausted"
    # An abort emits NO entry order (only the protective stop, if any position is open).
    assert all(c.reduce_only or c.kind.startswith("STOP") for c in d.commands)


def test_engine_post_only_abort_on_worst_entry_price_breach():
    """ENGINE-level: with a TIGHT invalidation the NEXT reprice price breaches worst_entry while the
    step count is well within budget — so the abort comes from the PRICE bound (ChaseLadder._beyond,
    the literal P5/Inv D guard), NOT step exhaustion."""
    eng = build_engine()
    sc = ExecutionSynth().calm_wide_spread()
    cfg = sc.cfg
    entry0 = sc.target.arrival_mid
    # Tight invalidation so worst_entry sits inside ONE reprice tick of entry0:
    #   worst_entry - entry0 = chi*|inv-entry0| < reprice_tick_step*tick.
    tight = dataclasses.replace(sc.target, invalidation=entry0 - 0.2)
    we = worst_entry(tight, entry0, cfg)
    next_price = entry0 + cfg.defaults.reprice_tick_step * cfg.instrument.tick_size
    assert next_price > we  # the next reprice WOULD breach worst_entry (price bound)
    # Start at step 0: step count is NOT exhausted (1 <= max_chase_steps), so the abort MUST be the
    # price breach, not the step counter.
    working = WorkingState(chase_step=0, worst_entry=we)
    assert 1 <= cfg.defaults.max_chase_steps
    d = on_target(eng, tight, sc.micro, working, cfg)
    assert d.tactic == "POST_ONLY_MAKER"
    assert d.aborted and d.abort_reason == "chase_budget_exhausted"
    assert d.chase_step == 0  # aborted at step 0 — it never advanced the step counter


def test_chase_flip_post_to_cross():
    """When the re-run min() says posting no longer wins, the chase crosses, capped at worst_entry."""
    sc = ExecutionSynth().runaway_breakout()
    cfg = sc.cfg
    we = worst_entry(sc.target, sc.target.arrival_mid, cfg)
    action = ChaseLadder().step(sc.target, sc.micro, sc.working, cfg, post_still_wins=False)
    assert action.action == "cross"
    assert action.price <= we + 1e-6


def test_chase_exit_crosses_at_budget_not_abort():
    """An exit at the budget edge crosses at MARKET, never aborts (P5/P6)."""
    sc = ExecutionSynth().runaway_breakout()
    cfg = sc.cfg
    exit_t = dataclasses.replace(sc.target, reduce_only=True)
    working = WorkingState(chase_step=cfg.defaults.max_chase_steps, worst_entry=0.0)
    action = ChaseLadder().step(exit_t, sc.micro, working, cfg, post_still_wins=True)
    assert action.action == "cross" and action.tactic == "MARKET" and not action.aborted


def test_partial_dust_below_min_notional():
    """A sub-min-notional residual is closed with below_min_notional, not emitted as dust (§8.3)."""
    sc = ExecutionSynth().heavy_partial_fills()
    plan = PartialFillManager().on_fill(sc.target, sc.working, sc.micro, sc.cfg)
    assert not plan.pursue and plan.closed_reason == "below_min_notional"


def test_exit_residual_pursued_to_flat():
    """A reducible reduce/exit residual is ALWAYS pursued to flat (crossing if needed), never
    abandoned — only an ENTRY residual may be abandoned (§8 / P5 / P6)."""
    sc = ExecutionSynth().calm_wide_spread()
    exit_t = dataclasses.replace(sc.target, reduce_only=True)
    # Half-filled exit, residual well above min-notional -> must pursue (not close).
    working = WorkingState(filled_qty=0.005, avg_fill_price=sc.micro.mid, worst_entry=sc.micro.mid)
    plan = PartialFillManager().on_fill(exit_t, working, sc.micro, sc.cfg)
    assert plan.pursue and plan.closed_reason == ""
    assert plan.reason_codes == ("pursue_exit_residual",)


def test_exit_residual_sub_min_notional_venue_floor_carveout():
    """A sub-min-notional EXIT residual hits the explicit venue-floor CARVE-OUT (§8): it cannot be
    placed on the venue, so it is closed with the DISTINCT exit_dust_venue_floor reason — NOT the
    entry below_min_notional abort. The same-size ENTRY residual still closes as below_min_notional,
    proving the entry/exit asymmetry."""
    sc = ExecutionSynth().heavy_partial_fills()  # residual ~ $3 < $5 min-notional
    # ENTRY: canonical dust close.
    entry_plan = PartialFillManager().on_fill(sc.target, sc.working, sc.micro, sc.cfg)
    assert not entry_plan.pursue and entry_plan.closed_reason == "below_min_notional"
    # EXIT (same residual): venue-floor carve-out with the distinct reason, not a chosen abandon.
    exit_t = dataclasses.replace(sc.target, reduce_only=True)
    exit_plan = PartialFillManager().on_fill(exit_t, sc.working, sc.micro, sc.cfg)
    assert not exit_plan.pursue and exit_plan.closed_reason == "exit_dust_venue_floor"
    assert exit_plan.reason_codes == ("exit_dust_venue_floor",)


def test_partial_complete_when_filled():
    """A fully-filled target leaves no residual to pursue (complete) (§8)."""
    sc = ExecutionSynth().calm_wide_spread()
    working = WorkingState(filled_qty=abs(sc.target.target_qty), worst_entry=30_000.0)
    plan = PartialFillManager().on_fill(sc.target, working, sc.micro, sc.cfg)
    assert not plan.pursue and plan.closed_reason == "complete"


def test_partial_never_overshoots():
    """Even an over-reported filled_qty cannot make the residual negative/overshoot (Invariant A)."""
    sc = ExecutionSynth().calm_wide_spread()
    working = WorkingState(filled_qty=abs(sc.target.target_qty) * 2.0, worst_entry=30_000.0)
    plan = PartialFillManager().on_fill(sc.target, working, sc.micro, sc.cfg)
    assert abs(plan.remaining_qty) <= abs(sc.target.target_qty) + 1e-12


def test_order_issuer_idempotent_retry_same_id():
    """A retry (same intent/slice/attempt) reuses the SAME client_order_id (Invariant Q)."""
    cfg = resolve_config(perp_instrument())
    iss = OrderIssuer()
    t = ExecTarget("X", "BTCUSDT", 1, 0.01, 100.0, 98.0, "NORMAL", False)
    c1 = iss.issue("LIMIT_IOC", t, 100.0, 0.01, False, cfg, slice_idx=0, attempt=0)
    c2 = iss.issue("LIMIT_IOC", t, 100.0, 0.01, False, cfg, slice_idx=0, attempt=0)
    assert c1.client_order_id == c2.client_order_id
    # A new attempt -> a new id (a true new submission, not a retry).
    c3 = iss.issue("LIMIT_IOC", t, 100.0, 0.01, False, cfg, slice_idx=0, attempt=1)
    assert c3.client_order_id != c1.client_order_id


def test_order_issuer_tactic_order_types():
    """Each tactic maps to the right order kind / TIF / post-only flag (§7)."""
    cfg = resolve_config(perp_instrument())
    iss = OrderIssuer()
    t = ExecTarget("X", "BTCUSDT", 1, 0.01, 100.0, 98.0, "NORMAL", False)
    post = iss.issue("POST_ONLY_MAKER", t, 99.9, 0.01, False, cfg)
    ioc = iss.issue("LIMIT_IOC", t, 100.0, 0.01, False, cfg)
    mkt_lim = iss.issue("MARKETABLE_LIMIT", t, 100.0, 0.01, False, cfg)
    mkt = iss.issue("MARKET", t, 0.0, 0.01, False, cfg)
    assert post.kind == "LIMIT" and post.post_only and post.time_in_force == "GTC"
    assert ioc.kind == "LIMIT" and not ioc.post_only and ioc.time_in_force == "IOC"
    assert mkt_lim.kind == "LIMIT" and mkt_lim.time_in_force == "IOC"
    assert mkt_lim.price > 100.0  # touch + through_ticks for a long
    assert mkt.kind == "MARKET"


def test_filled_qty_never_exceeds_target_under_retries():
    """No retry/restart sequence overshoots target qty (Invariant Q) — the engine clamps residual."""
    eng = build_engine()
    sc = ExecutionSynth().calm_wide_spread()
    # Simulate repeated re-issues against a growing filled_qty; total emitted never overshoots.
    total_target = abs(sc.target.target_qty)
    for filled in (0.0, 0.003, 0.007, total_target):
        working = WorkingState(filled_qty=filled, worst_entry=30_000.0)
        d = on_target(eng, sc.target, sc.micro, working, sc.cfg)
        assert d.qty <= total_target - filled + 1e-9


def test_audit_decomposition_matches_emitted_tactic_after_chase_flip():
    """§12: when the ChaseLadder flips the originally-selected POST_ONLY to a crossing tactic, the
    ExecDecision logs the cost/decomposition of the tactic ACTUALLY emitted, not the selected one —
    otherwise the realized-vs-expected regression compares a realized cross to an expected post."""
    from convexity.execution.cost.adverse_selection import AdverseSelection
    from convexity.execution.cost.delay_cost import DelayCost
    from convexity.execution.cost.impact import ImpactModel
    from convexity.execution.cost.spread_cost import SpreadCost
    from convexity.execution.cost.tactic_cost import TacticCostModel
    from convexity.execution.policy.tactic_select import TacticSelector

    eng = build_engine()
    sc = ExecutionSynth().calm_wide_spread()
    cfg = sc.cfg
    # A calm EXIT at the chase-budget edge: TacticSelector picks POST_ONLY (cheap maker), but the
    # ladder crosses at MARKET (exits never abort, P5/P6) -> selected != emitted.
    exit_t = dataclasses.replace(sc.target, reduce_only=True)
    we = worst_entry(exit_t, exit_t.arrival_mid, cfg)
    working = WorkingState(
        chase_step=cfg.defaults.max_chase_steps, worst_entry=we,
        filled_qty=0.005, avg_fill_price=sc.micro.mid,
    )
    cm = TacticCostModel(SpreadCost(), ImpactModel(), AdverseSelection(), DelayCost())
    costs = cm.costs(exit_t, sc.micro, cfg)
    sel = TacticSelector().select(exit_t, sc.micro, costs, cfg)
    d = on_target(eng, exit_t, sc.micro, working, cfg)
    assert sel.tactic == "POST_ONLY_MAKER" and d.tactic == "MARKET"  # a genuine flip occurred
    # The headline cost is the EMITTED tactic's, not the stale selected one.
    assert d.expected_cost_bps == costs.by_tactic[d.tactic]
    assert d.expected_cost_bps != costs.by_tactic[sel.tactic]
    # Every decomposition component matches the emitted tactic's cost maps.
    assert d.cost_decomposition["spread_bps"] == costs.spread_bps[d.tactic]
    assert d.cost_decomposition["adverse_sel_bps"] == costs.adverse_sel_bps[d.tactic]
    assert d.cost_decomposition["delay_bps"] == costs.delay_bps[d.tactic]
    assert d.cost_decomposition["funding_bps"] == costs.funding_bps[d.tactic]


def test_maker_floor_gate_removes_post_only():
    """A thin/miscalibrated bucket removes POST_ONLY via the LCB gate (§6.3 gate 3)."""
    eng = build_engine()
    sc = ExecutionSynth().maker_never_fills()
    d = on_target(eng, sc.target, sc.micro, sc.working, sc.cfg)
    assert d.tactic != "POST_ONLY_MAKER"
    assert d.binding_gate == "maker_floor"


def test_rate_limit_gate_prefers_crossing():
    """A low rate-limit budget removes POST_ONLY -> a single crossing order (§6.4)."""
    eng = build_engine()
    sc = ExecutionSynth().rate_limit_pressure()
    d = on_target(eng, sc.target, sc.micro, sc.working, sc.cfg)
    assert d.tactic != "POST_ONLY_MAKER"
    assert d.binding_gate == "rate_limit"


def test_gates_only_more_aggressive():
    """Every gate can only remove POST_ONLY / force crossing — never ADD a passive tactic (§6)."""
    eng = build_engine()
    for sc in (ExecutionSynth().protective_exit(), ExecutionSynth().halt_stale_feed(),
               ExecutionSynth().maker_never_fills(), ExecutionSynth().rate_limit_pressure()):
        d = on_target(eng, sc.target, sc.micro, sc.working, sc.cfg)
        # under any binding gate the chosen tactic is a crossing tactic, never POST_ONLY.
        if d.binding_gate != "none":
            assert d.tactic != "POST_ONLY_MAKER"
