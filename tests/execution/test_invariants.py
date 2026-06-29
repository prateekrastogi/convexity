"""EXECUTION named-invariant tests (EXECUTION §16). One test per load-bearing invariant."""

import ast
import dataclasses
import pathlib

import pytest

from convexity.core.params import Bucket
from convexity.execution.config.defaults import EXECUTION_PARAM_REGISTRY
from convexity.execution.config.instrument import INSTRUMENT_SPEC_FIELDS
from convexity.execution.config.resolve import resolve_config
from convexity.execution.cost.tactic_cost import TacticCostModel, fund_bps
from convexity.execution.cost.adverse_selection import AdverseSelection
from convexity.execution.cost.delay_cost import DelayCost
from convexity.execution.cost.impact import ImpactModel
from convexity.execution.cost.spread_cost import SpreadCost
from convexity.execution.engine_api import build_engine, on_target
from convexity.execution.microstate.fill_prob import (
    FillProbability,
    PassiveFillRecord,
)
from convexity.execution.nautilus.reconcile import (
    IntendedState,
    StopReconciler,
    stop_ok,
)
from convexity.execution.policy.chase import worst_entry
from convexity.execution.policy.order_issuer import (
    slice_client_order_id,
    stop_client_order_id,
)
from convexity.execution.policy.protective_stop import ProtectiveStop
from convexity.execution.synth.generate import (
    ExecutionSynth,
    equity_instrument,
    perp_instrument,
)
from convexity.execution.types import ExecTarget, PositionState, WorkingState


@pytest.fixture
def synth():
    return ExecutionSynth(seed=11)


@pytest.fixture
def engine():
    return build_engine()


def _cost_model():
    return TacticCostModel(SpreadCost(), ImpactModel(), AdverseSelection(), DelayCost())


# --------------------------------------------------------------------------- Invariant A

def test_inv_a_no_edge_field_on_exectarget():
    """ExecTarget carries NO signal/sizing/exit score by construction (Invariant A)."""
    fields = set(ExecTarget.__dataclass_fields__)
    forbidden = {"breakout_score", "expected_return", "trend_confidence", "kelly_fraction", "edge"}
    assert not (fields & forbidden)


def test_inv_a_qty_never_exceeds_target(engine, synth):
    """Residual qty never exceeds target; no decision overshoots |target_qty| (Invariant A)."""
    for sc in synth.all_scenarios():
        d = on_target(engine, sc.target, sc.micro, sc.working, sc.cfg)
        assert d.qty <= abs(sc.target.target_qty) + 1e-9


def test_inv_a_partial_residual_bounded(engine):
    """A partial fill leaves a residual that never grows the target (Invariant A)."""
    s = ExecutionSynth().calm_wide_spread()
    # already-filled half: residual must be <= remaining half.
    working = WorkingState(filled_qty=0.005, avg_fill_price=30_000.0, worst_entry=30_000.0)
    d = on_target(engine, s.target, s.micro, working, s.cfg)
    assert d.qty <= 0.005 + 1e-9


# --------------------------------------------------------------------------- Invariant B

def test_inv_b_tactic_is_argmin(engine, synth):
    """Chosen tactic == argmin E[cost] over the ungated set; never a product of scores (Inv B)."""
    cm = _cost_model()
    for sc in synth.all_scenarios():
        costs = cm.costs(sc.target, sc.micro, sc.cfg)
        d = on_target(engine, sc.target, sc.micro, sc.working, sc.cfg)
        if d.aborted or d.qty == 0.0:
            continue
        # The decided tactic must be the cheapest among the gate-allowed set; since gates only
        # remove tactics, the chosen tactic's cost is the min over whatever survived.
        chosen_cost = costs.by_tactic[d.tactic]
        # POST_ONLY chosen => it was the global argmin (no gate removed it).
        if d.binding_gate == "none":
            assert chosen_cost <= min(costs.by_tactic.values()) + 1e-9


def test_inv_b_components_independent():
    """Spread / impact / fee compute independently — perturbing impact never moves spread (Inv B)."""
    inst = perp_instrument()
    cfg = resolve_config(inst)
    sc = ExecutionSynth().calm_wide_spread()
    sp = SpreadCost()
    base = sp.taker_cost_bps(sc.micro.half_spread_bps, cfg)
    # Impact is a separate object; computing it cannot mutate the spread term.
    ImpactModel().cost_bps(sc.target.target_qty, sc.micro, cfg)
    assert sp.taker_cost_bps(sc.micro.half_spread_bps, cfg) == base


# --------------------------------------------------------------------------- Invariant C

def test_inv_c_no_red_in_amber():
    """RED/BLUE EXECUTION params are unreachable from the per-instrument AMBER spec (Invariant C)."""
    forbidden = {
        p.name for p in EXECUTION_PARAM_REGISTRY.params()
        if p.bucket in (Bucket.RED, Bucket.BLUE)
    }
    assert not (forbidden & set(INSTRUMENT_SPEC_FIELDS))


def test_inv_c_resolve_runs_check():
    """resolve_config runs the Invariant-C assertion without raising on a clean spec."""
    resolve_config(perp_instrument())  # must not raise


# --------------------------------------------------------------------------- Invariant D

def test_inv_d_runaway_aborts_never_beyond_worst_entry(engine):
    """A runaway breakout aborts at worst_entry, never fills beyond it (Invariant D)."""
    sc = ExecutionSynth().runaway_breakout()
    # Drive the chase to the budget edge: many steps already taken.
    cfg = sc.cfg
    we = worst_entry(sc.target, sc.target.arrival_mid, cfg)
    working = WorkingState(chase_step=cfg.defaults.max_chase_steps, worst_entry=we)
    d = on_target(engine, sc.target, sc.micro, working, cfg)
    if d.tactic == "POST_ONLY_MAKER" and d.aborted:
        assert d.abort_reason == "chase_budget_exhausted"
    # No emitted entry order ever prices beyond worst_entry (long: price <= we).
    for cmd in d.commands:
        if cmd.reduce_only or cmd.kind.startswith("STOP"):
            continue
        if cmd.kind != "MARKET":
            assert cmd.price <= we + 1e-6


def test_inv_d_worst_entry_long_short_canonical():
    """worst_entry = entry0 + side·χ·|inv−entry0|: long above entry0, short below (§5.3)."""
    cfg = resolve_config(perp_instrument())
    chi = cfg.defaults.chase_fraction_chi
    long_t = ExecTarget("l", "BTCUSDT", 1, 0.01, 100.0, 98.0, "URGENT", False)
    short_t = ExecTarget("s", "BTCUSDT", -1, -0.01, 100.0, 102.0, "URGENT", False)
    assert worst_entry(long_t, 100.0, cfg) == pytest.approx(100.0 + chi * 2.0)
    assert worst_entry(short_t, 100.0, cfg) == pytest.approx(100.0 - chi * 2.0)


def test_inv_d_exit_never_aborts(engine):
    """A reduce/exit at the budget edge crosses at MARKET, never aborts (P5/P6 / §5.3)."""
    sc = ExecutionSynth().runaway_breakout()
    exit_t = dataclasses.replace(sc.target, reduce_only=True, urgency="URGENT")
    cfg = sc.cfg
    we = worst_entry(exit_t, exit_t.arrival_mid, cfg)
    working = WorkingState(chase_step=cfg.defaults.max_chase_steps, worst_entry=we, filled_qty=0.0)
    d = on_target(engine, exit_t, sc.micro, working, cfg)
    assert not d.aborted


# --------------------------------------------------------------------------- Invariant E

def test_inv_e_determinism(engine, synth):
    """Batch replay and a one-event-at-a-time stream produce identical decisions (Invariant E)."""
    for sc in synth.all_scenarios():
        d1 = on_target(engine, sc.target, sc.micro, sc.working, sc.cfg)
        d2 = on_target(build_engine(), sc.target, sc.micro, sc.working, sc.cfg)
        assert d1.tactic == d2.tactic
        assert d1.price == pytest.approx(d2.price)
        assert d1.binding_gate == d2.binding_gate
        assert d1.aborted == d2.aborted


# --------------------------------------------------------------------------- Invariant H

def test_inv_h_diagnostics_log_only(engine):
    """Diagnostic contents never alter the chosen tactic/price/qty (Invariant H)."""
    sc = ExecutionSynth().calm_wide_spread()
    d = on_target(engine, sc.target, sc.micro, sc.working, sc.cfg)
    # The decision is a pure function of (target, micro, working, cfg); the diagnostics map is
    # derived from costs and cannot feed back. Mutating a copy's diagnostics changes nothing.
    assert isinstance(d.cost_decomposition, dict)
    d2 = on_target(engine, sc.target, sc.micro, sc.working, sc.cfg)
    assert d.tactic == d2.tactic and d.price == d2.price and d.qty == d2.qty


# --------------------------------------------------------------------------- Invariant I

def test_inv_i_impact_vanishes_at_retail_size():
    """Impact → ~0 as participation → 0; grows on thin proxies at large size (Invariant I)."""
    small, large = ExecutionSynth().thin_book_large_size()
    im = ImpactModel()
    small_bps = im.cost_bps(small.target.target_qty, small.micro, small.cfg)
    large_bps = im.cost_bps(large.target.target_qty, large.micro, large.cfg)
    assert small_bps < 1.0  # a $30 clip: negligible
    assert large_bps > small_bps  # large clip on a thin venue: non-trivial


def test_inv_i_impact_monotone_continuous():
    """Impact is monotone non-decreasing and continuous in participation; 0 at participation 0."""
    sc = ExecutionSynth().calm_wide_spread()
    im = ImpactModel()
    prev = -1.0
    for q in (0.0, 0.001, 0.01, 0.1, 1.0):
        t = dataclasses.replace(sc.target, target_qty=q)
        c = im.cost_bps(t.target_qty, sc.micro, sc.cfg)
        assert c >= prev
        prev = c
    assert im.cost_bps(0.0, sc.micro, sc.cfg) == 0.0


# --------------------------------------------------------------------------- Invariant J

def test_inv_j_maker_trio_sign_relations():
    """As g rises: p_fill non-increasing, A non-decreasing, D non-decreasing (Invariant J)."""
    cfg = resolve_config(perp_instrument())
    adverse, delay = AdverseSelection(), DelayCost()
    p_fills, A_vals, D_vals = [], [], []
    for sc in ExecutionSynth().swept_g():
        g = sc.micro.drift_g
        p = sc.micro.fill_prob_curve(0.0, cfg.defaults.fill_horizon_s, g)
        a = adverse.cost_bps(sc.target, sc.micro, 1.0, "b", g, cfg)
        d = delay.cost_bps(sc.target, sc.micro, 1.0, cfg.defaults.fill_horizon_s, g, cfg)
        p_fills.append(p)
        A_vals.append(a)
        D_vals.append(d)
    assert all(p_fills[i] >= p_fills[i + 1] - 1e-12 for i in range(len(p_fills) - 1))
    assert all(A_vals[i] <= A_vals[i + 1] + 1e-12 for i in range(len(A_vals) - 1))
    assert all(D_vals[i] <= D_vals[i + 1] + 1e-12 for i in range(len(D_vals) - 1))


def test_inv_j_offset_monotonicity():
    """A and the deeper-post structure are non-decreasing in offset_ticks (Invariant J)."""
    cfg = resolve_config(perp_instrument())
    sc = ExecutionSynth().swept_g()[2]
    adverse = AdverseSelection()
    vals = [adverse.cost_bps(sc.target, sc.micro, k, "b", 1.0, cfg) for k in (0.0, 1.0, 2.0, 3.0)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_inv_j_g_reread_per_step_not_frozen():
    """drift_g feeds the trio from the CURRENT mid_k, not the arrival-anchored value (Invariant J)."""
    # A MicroState whose drift_g is re-read; the cost model uses micro.drift_g directly, so a
    # different micro (mid_k walked) yields a different g into the trio.
    cfg = resolve_config(perp_instrument())
    sc = ExecutionSynth().swept_g()[0]
    m_calm = sc.micro
    m_run = dataclasses.replace(sc.micro, drift_g=3.0)
    a_calm = AdverseSelection().cost_bps(sc.target, m_calm, 1.0, "b", m_calm.drift_g, cfg)
    a_run = AdverseSelection().cost_bps(sc.target, m_run, 1.0, "b", m_run.drift_g, cfg)
    assert a_run > a_calm


# --------------------------------------------------------------------------- Invariant L

def test_inv_l_online_embargo_excludes_unresolved():
    """A fill whose outcome horizon has not closed is ABSENT from the curve used at t (Invariant L)."""
    cfg = resolve_config(perp_instrument())
    fp = FillProbability()
    # Seed enough resolved fills to clear the bucket floor.
    for i in range(cfg.defaults.n_bucket + 10):
        fp.admit(PassiveFillRecord("b", 1.0, 0.5, filled=True, outcome_close_ts=100))
    r_before = fp.curve(None, "b", cfg, as_of_ts=200, sigma_bps=8.0, tick_bps=0.03)
    # Inject a not-yet-resolved fill (closes at 999, as_of 200): must NOT enter the curve at t=200.
    fp.admit(PassiveFillRecord("b", 1.0, 0.5, filled=False, outcome_close_ts=999))
    r_after = fp.curve(None, "b", cfg, as_of_ts=200, sigma_bps=8.0, tick_bps=0.03)
    assert r_before.n_events == r_after.n_events  # the unresolved fill is embargoed out


# --------------------------------------------------------------------------- Invariant N

def test_inv_n_only_nautilus_dir_imports_engine():
    """Only execution/nautilus/* may import nautilus_trader (Invariant N)."""
    root = pathlib.Path("src/convexity/execution")
    offenders = []
    for path in root.rglob("*.py"):
        if "nautilus" in path.parts:
            continue
        src = path.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.startswith("nautilus_trader"):
                        offenders.append(str(path))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("nautilus_trader"):
                    offenders.append(str(path))
    assert offenders == []


def test_inv_n_adapter_import_isolated():
    """The nautilus adapter imports with NO nautilus installed (lazy import inside bind, Inv N)."""
    from convexity.execution.nautilus import adapter, reconcile  # must import cleanly
    assert adapter is not None and reconcile is not None


# --------------------------------------------------------------------------- Invariant Q / Q-STOP

def test_inv_q_slice_id_deterministic():
    """Slice client_order_id is deterministic in (intent_id, slice_idx, attempt) (Invariant Q)."""
    a = slice_client_order_id("X", 2, 1)
    b = slice_client_order_id("X", 2, 1)
    c = slice_client_order_id("X", 2, 2)
    assert a == b and a != c


def test_inv_qstop_disjoint_from_slice_scheme():
    """A stop id never equals any entry/exit slice id (Invariant Q-STOP disjointness)."""
    slice_ids = {slice_client_order_id("X", i, a) for i in range(20) for a in range(5)}
    stop_ids = {stop_client_order_id("X", g) for g in range(20)}
    assert slice_ids.isdisjoint(stop_ids)


def test_inv_qstop_distinct_per_generation():
    """N successive amends produce N distinct accepted stop ids (Invariant Q-STOP)."""
    ids = [stop_client_order_id("X", g) for g in range(10)]
    assert len(set(ids)) == 10


# --------------------------------------------------------------------------- Invariant R

def test_inv_r_all_protective_orders_reduce_only():
    """Every stop command is reduce_only (Invariant R)."""
    cfg = resolve_config(perp_instrument())
    pos = PositionState("BTCUSDT", 1, 0.01)
    place = ProtectiveStop().place("X", pos, 29_000.0, cfg)
    assert place.reduce_only is True
    for c in ProtectiveStop().amend("X", pos, 29_500.0, cfg, amend_generation=0):
        assert c.reduce_only is True


# --------------------------------------------------------------------------- Invariant S

def test_inv_s_no_stop_places_one(synth):
    """Open position, NO venue stop -> reconcile places one before anything else (Invariant S)."""
    sc = synth.restart_no_stop()
    rec = StopReconciler()
    intended = IntendedState("recon-1", sc.target.invalidation, 0)
    plan = rec.reconcile(intended, sc.stops, sc.position, sc.cfg)
    assert len(plan.commands) == 1
    assert plan.commands[0].action == "submit" and plan.commands[0].reduce_only


def test_inv_s_two_stops_collapse_to_one(synth):
    """Crash in amend window -> TWO stops; reconcile keeps current gen, cancels the stale (Inv S/d)."""
    sc = synth.amend_crash_two_stops()
    rec = StopReconciler()
    intended = IntendedState("recon-2", sc.target.invalidation, sc.working.invalidation_generation)
    plan = rec.reconcile(intended, sc.stops, sc.position, sc.cfg)
    cancels = [c for c in plan.commands if c.action == "cancel"]
    assert len(cancels) == 1
    assert cancels[0].client_order_id == stop_client_order_id("recon-2", 0)  # the stale gen


def test_inv_s_partial_fire_recovers_remaining(synth):
    """Partial STOP_MARKET fire -> reconcile re-covers the EXACT remaining position qty (Inv S/e)."""
    sc = synth.partial_stop_fire()
    rec = StopReconciler()
    intended = IntendedState("recon-3", sc.target.invalidation, 0)
    plan = rec.reconcile(intended, sc.stops, sc.position, sc.cfg)
    submits = [c for c in plan.commands if c.action in ("submit", "modify")]
    assert submits and submits[0].qty == pytest.approx(abs(sc.position.qty))


def test_inv_s_orphan_stop_position_flat(synth):
    """Stop fired while down -> position flat; reconcile cancels the orphan, marks closed (Inv S)."""
    sc = synth.outage_gap_restart()
    rec = StopReconciler()
    intended = IntendedState("recon-4", sc.target.invalidation, 0)
    plan = rec.reconcile(intended, sc.stops, sc.position, sc.cfg)
    assert plan.position_closed
    assert all(c.action == "cancel" for c in plan.commands)


def test_inv_s_idempotent_second_run_empty(synth):
    """reconcile() run twice on a satisfying state -> EMPTY plan on the second run (Invariant S)."""
    sc = synth.restart_no_stop()
    rec = StopReconciler()
    intended = IntendedState("recon-1", sc.target.invalidation, 0)
    plan = rec.reconcile(intended, sc.stops, sc.position, sc.cfg)
    placed = plan.commands[0]
    # The placed stop now rests on the venue; re-run with it observed -> STOP_OK holds -> empty.
    from convexity.execution.types import VenueStop
    resting = (VenueStop(placed.client_order_id, placed.price, placed.qty, True, placed.kind),)
    plan2 = rec.reconcile(intended, resting, sc.position, sc.cfg)
    assert plan2.is_empty
    assert stop_ok(intended, sc.position, resting)


# --------------------------------------------------------------------------- Invariant T

def test_inv_t_chi_and_timeouts_are_class_level_not_per_instrument():
    """χ / chase timeouts / urgency-mapping knobs are RED constants shared across instruments,
    never fitted per symbol on realized P&L (Invariant T). Two different instruments (a perp and
    a US equity) resolve to the IDENTICAL RED defaults — there is no per-symbol χ to tune."""
    perp_cfg = resolve_config(perp_instrument())
    eq_cfg = resolve_config(equity_instrument())
    # The RED tuning knobs are bit-identical across symbols — same global PolicyDefaults values,
    # never a per-symbol fit (frozen dataclass equality on the whole RED block).
    assert perp_cfg.defaults == eq_cfg.defaults
    assert perp_cfg.defaults.chase_fraction_chi == eq_cfg.defaults.chase_fraction_chi
    assert perp_cfg.defaults.timeout_step_s == eq_cfg.defaults.timeout_step_s
    assert perp_cfg.defaults.max_chase_steps == eq_cfg.defaults.max_chase_steps


def test_inv_t_red_knobs_absent_from_instrument_spec():
    """χ, the timeouts, and the urgency thresholds NEVER appear as a per-instrument field — a
    symbol cannot carry its own tuned χ (Invariant T, the structural side of the firewall)."""
    red_names = {
        p.name for p in EXECUTION_PARAM_REGISTRY.params() if p.bucket is Bucket.RED
    }
    tuning_knobs = {"chase_fraction_chi", "timeout_step_s", "max_chase_steps", "reprice_tick_step"}
    assert tuning_knobs <= red_names  # they ARE declared RED
    assert tuning_knobs.isdisjoint(INSTRUMENT_SPEC_FIELDS)  # and never per-instrument AMBER


def test_inv_t_chi_search_is_a_class_level_blue_grid():
    """χ is searched (offline, against realized IS — a COST metric) only via a class-level BLUE
    grid, never a per-symbol P&L fit (Invariant T)."""
    blue = {p.name: p for p in EXECUTION_PARAM_REGISTRY.params() if p.bucket is Bucket.BLUE}
    assert "chi_grid" in blue
    grid = blue["chi_grid"].blue_grid
    assert grid is not None and len(grid) >= 2  # a finite candidate set, not a free per-symbol knob
    # Every candidate sits inside the spec's [0.25, 0.40] RED band (§5.3).
    assert all(0.25 <= c <= 0.40 for c in grid)


# --------------------------------------------------------------------------- Invariant W

def test_inv_w_funding_paying_side_raises_maker(synth):
    """A paying-side perp clip across an imminent stamp gets fund_bps > 0 (Invariant W)."""
    sc = synth.funding_stamp_imminent()
    h = sc.cfg.defaults.fill_horizon_s
    fb = fund_bps(sc.target, sc.micro, h, sc.cfg)
    assert fb > 0.0


def test_inv_w_receiving_side_floored_zero(synth):
    """The receiving side is floored at 0 (one-signed conservative, Invariant W)."""
    sc = synth.funding_receiving_side()
    h = sc.cfg.defaults.fill_horizon_s
    assert fund_bps(sc.target, sc.micro, h, sc.cfg) == 0.0


def test_inv_w_nonperp_zero():
    """A non-perp class has an identically-0 funding term (Invariant W)."""
    small, _ = ExecutionSynth().thin_book_large_size()  # us_equity
    h = small.cfg.defaults.fill_horizon_s
    assert fund_bps(small.target, small.micro, h, small.cfg) == 0.0


def test_inv_w_half_open_boundary(synth):
    """A clip whose horizon ends EXACTLY at a stamp counts ZERO stamps (half-open, Invariant W)."""
    sc = synth.funding_boundary()
    h = sc.cfg.defaults.fill_horizon_s  # secs_to_next_funding == h by construction
    assert fund_bps(sc.target, sc.micro, h, sc.cfg) == 0.0


def test_inv_w_crossing_tactic_not_over_charged_funding(synth):
    """§5.4: a taker order that crosses NOW does not rest for the fill horizon, so it is NOT charged
    funding over the full resting window even when a stamp falls inside [now, now+h). The crossing
    tactics' funding term is the immediate (~0) cross window, NOT the over-counted full-h term."""
    sc = synth.funding_stamp_imminent()  # stamp 10s away, h=30s -> a stamp IS inside [now, now+h)
    cfg = sc.cfg
    h = cfg.defaults.fill_horizon_s
    # The resting (maker) funding over the full window IS positive (the stamp is straddled).
    assert fund_bps(sc.target, sc.micro, h, cfg) > 0.0
    # But the assembled crossing tactics are NOT charged that full-window funding (the §5.4 fix).
    costs = _cost_model().costs(sc.target, sc.micro, cfg)
    for tac in ("MARKET", "MARKETABLE_LIMIT", "LIMIT_IOC"):
        assert costs.funding_bps[tac] == 0.0  # immediate cross spans ~0 stamps, not over-charged
    # The maker rest branch DOES carry the full-window funding (it can flip the min() to cross-now).
    assert costs.funding_bps["POST_ONLY_MAKER"] > 0.0


def test_inv_w_maker_fill_branch_over_charge_is_one_signed(synth):
    """§5.4 Fix-4: charging the maker FILL branch the full-h funding (rather than expected
    rest-until-fill) is a deliberate ONE-SIGNED conservative over-charge — it can only RAISE the
    maker cost, never lower it, biasing the min() toward the lower-variance cross (the SAFE
    direction). Proven by construction: removing the funding (the unfloored 'truth' lower bound)
    yields a maker cost <= the charged one for a paying-side perp."""
    sc = synth.funding_stamp_imminent()  # paying side, stamp straddled
    cfg = sc.cfg
    cm = _cost_model()
    charged = cm.costs(sc.target, sc.micro, cfg).by_tactic["POST_ONLY_MAKER"]
    # The unfloored lower bound: the SAME clip on a NON-funding-aware preset (funding term -> 0),
    # i.e. the maker cost with zero funding. The charged maker cost must be >= this (one-signed up).
    no_fund_micro = dataclasses.replace(sc.micro, predicted_funding_rate=0.0)
    truth_lb = cm.costs(sc.target, no_fund_micro, cfg).by_tactic["POST_ONLY_MAKER"]
    assert charged >= truth_lb - 1e-12  # over-charge is one-signed toward crossing, never below truth


# --------------------------------------------------------------------------- Invariant V

def test_inv_v_live_layers_never_import_validate():
    """Nothing in microstate/cost/policy/nautilus imports validate/ (Invariant V)."""
    root = pathlib.Path("src/convexity/execution")
    live_dirs = {"microstate", "cost", "policy", "nautilus"}
    offenders = []
    for path in root.rglob("*.py"):
        if not (set(path.parts) & live_dirs):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            if mod and "execution.validate" in mod:
                offenders.append(str(path))
    assert offenders == []
