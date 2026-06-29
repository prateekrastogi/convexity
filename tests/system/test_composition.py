"""Composition / precedence tests (SYSTEM §2-§4, Invariants CB, HI, IT, CAD, A, P1).

The orchestrator runs against SCRIPTED module outputs (mock adapters) — no real pillars (§15 gate).
"""

import numpy as np

from convexity.system.compose.budget import Candidate, priority_order
from convexity.system.compose.cadence import decision_bar, exceeds_deadband
from convexity.system.compose.incumbent import reconcile, resolve_incumbent
from convexity.system.compose.iterate import FixedPointComposer, Position, greedy_lot_fill
from convexity.system.config.resolve import resolve_config
from convexity.system.config.universe import load_universe_meta
from convexity.system.synth.generate import (
    make_book,
    make_candidate,
    make_open_position,
    make_proposed,
    make_trend_action,
)
from convexity.system.system_api import decide_live


def cfg():
    return resolve_config(load_universe_meta([]))


# --------------------------------------------------------------------------- determinism + CB


def test_decision_is_deterministic():
    book = make_book({"A": make_open_position("A")})
    sel = make_proposed(("A", "B"))
    cands = {"B": make_candidate("B")}
    open_acts = {"A": make_trend_action("continue")}
    d1 = decide_live(book, sel, open_acts, cands, cfg())
    d2 = decide_live(book, sel, open_acts, cands, cfg())
    assert d1.actions == d2.actions
    assert d1.reason_codes == d2.reason_codes


def test_breakout_score_scaling_leaves_decision_unchanged():
    """Invariant CB: scaling every breakout score by a positive constant changes no decision."""
    book = make_book()
    sel = make_proposed(("B", "C"))
    open_acts: dict = {}
    base = {"B": make_candidate("B", quality_score=1.0), "C": make_candidate("C", quality_score=1.0)}
    scaled = {"B": make_candidate("B", quality_score=1000.0), "C": make_candidate("C", quality_score=5.0)}
    d_base = decide_live(book, sel, open_acts, base, cfg())
    d_scaled = decide_live(book, sel, open_acts, scaled, cfg())
    assert [(a.instrument, a.kind, a.target_units) for a in d_base.actions] == [
        (a.instrument, a.kind, a.target_units) for a in d_scaled.actions
    ]


# --------------------------------------------------------------------------- governor cascade (A)


def test_governor_flat_all_at_zero_multiplier():
    """Step A: governor multiplier 0 ⇒ FLAT_ALL, halt new opens."""
    book = make_book({"A": make_open_position("A")}, governor_multiplier=0.0)
    sel = make_proposed(("A", "B"))
    d = decide_live(book, sel, {"A": make_trend_action("continue")}, {"B": make_candidate("B")}, cfg())
    assert all(a.kind == "flat_all" for a in d.actions)
    assert "governor_flat_all" in d.reason_codes
    assert "halt_new_opens" in d.reason_codes


def test_governor_scales_new_lot_heat():
    """Step A: a partial governor (m<1) scales every size; fewer lots admitted than at m=1."""
    book = make_book()
    sel = make_proposed(("B",))
    cand = {"B": make_candidate("B", ceiling_weight=0.12, wpc=0.03)}
    full = decide_live(book, sel, {}, cand, cfg(), bar_ts=0)
    book_low = make_book(governor_multiplier=0.25)
    throttled = decide_live(book_low, sel, {}, cand, cfg(), bar_ts=0)
    full_lots = sum(a.target_units for a in full.actions if a.kind == "open")
    low_lots = sum(a.target_units for a in throttled.actions if a.kind == "open")
    assert low_lots <= full_lots


# --------------------------------------------------------------------------- hold-incumbent (HI)


def test_hold_incumbent_trend_alive_no_replacement():
    """Invariant HI: a trend-alive, ceiling-slack incumbent demoted by selection with NO qualifying
    replacement stays OPEN and emits hold_incumbent."""
    pos = make_open_position("A")
    sel = make_proposed(keep=("B",), eligible=("B",))  # A not in keep ⇒ selection wants it out
    dec = resolve_incumbent(
        pos, make_trend_action("continue"), sel,
        ceiling_collapsed=False, open_instruments=frozenset({"A"}),
        qualified=frozenset(), delta_rank=2.0, risk_improvement=0.0, round_trip_cost=1.0,
    )
    assert dec.verdict == "hold"
    assert dec.reason == "hold_incumbent"


def test_trend_exit_overrides_selection_keep():
    """Step B / §4: TREND exit closes even an in-keep incumbent."""
    book = make_book({"A": make_open_position("A")})
    sel = make_proposed(("A",))
    d = decide_live(book, sel, {"A": make_trend_action("exit")}, {}, cfg())
    a = next(x for x in d.actions if x.instrument == "A")
    assert a.kind == "close"
    assert "trend_exit" in a.reason_chain


def test_rotation_requires_rank_margin_and_cost_clearance():
    """Invariant HI: rotation only when rank margin > Δ_rank AND risk improvement clears cost."""
    pos = make_open_position("A")
    sel = make_proposed(keep=("B",), eligible=("B",), rank={"A": 0.0, "B": 5.0})
    # qualifying replacement B, big rank margin, risk improvement clears cost ⇒ rotate.
    dec = resolve_incumbent(
        pos, make_trend_action("continue"), sel,
        ceiling_collapsed=False, open_instruments=frozenset({"A"}),
        qualified=frozenset({"B"}), delta_rank=2.0, risk_improvement=10.0, round_trip_cost=1.0,
    )
    assert dec.verdict == "sel_rotate"
    assert dec.replacement == "B"
    # same setup but cost not cleared ⇒ hold.
    dec2 = resolve_incumbent(
        pos, make_trend_action("continue"), sel,
        ceiling_collapsed=False, open_instruments=frozenset({"A"}),
        qualified=frozenset({"B"}), delta_rank=2.0, risk_improvement=0.5, round_trip_cost=1.0,
    )
    assert dec2.verdict == "hold"


def test_ceiling_collapse_reduces_incumbent():
    """Step B / §4: a collapsed ceiling reduces an incumbent ahead of selection."""
    book = make_book({"A": make_open_position("A")})
    sel = make_proposed(("A",))
    d = decide_live(book, sel, {"A": make_trend_action("continue")}, {}, cfg(),
                    collapsed=frozenset({"A"}))
    a = next(x for x in d.actions if x.instrument == "A")
    assert a.kind == "reduce"
    assert a.binding == "ceiling_collapse"


# --------------------------------------------------------------------------- budget priority (A)


def test_budget_priority_is_risk_not_edge():
    """Invariant A: admission order is ceiling-slack → MCTR-efficiency → id; no edge input exists."""
    cands = [
        Candidate("Z", ceiling_slack=0.1, risk_weight_delivered=0.05, heat_consumed=0.05),
        Candidate("A", ceiling_slack=0.9, risk_weight_delivered=0.05, heat_consumed=0.05),
        Candidate("M", ceiling_slack=0.9, risk_weight_delivered=0.10, heat_consumed=0.05),
    ]
    order = priority_order(cands, budget_priority="slack_then_mctr")
    # highest slack first; among equal slack, higher MCTR-efficiency; tie-break id.
    assert order[0].instrument == "M"  # slack 0.9, eff 2.0
    assert order[1].instrument == "A"  # slack 0.9, eff 1.0
    assert order[2].instrument == "Z"  # slack 0.1


def test_candidate_has_no_edge_field():
    """Invariant A: the firewall is the ABSENCE of any edge field on Candidate."""
    fields = Candidate.__dataclass_fields__
    for forbidden in ("breakout_score", "expected_return", "trend_confidence", "edge"):
        assert forbidden not in fields


# --------------------------------------------------------------------------- cadence / deadband (CAD)


def test_deadband_suppresses_one_lot_jitter():
    """Invariant CAD: a sub-deadband target change generates no order."""
    assert not exceeds_deadband(2, 3, 0.05, 0.051, deadband_lots=1, deadband_weight=0.02)
    assert exceeds_deadband(2, 5, 0.05, 0.12, deadband_lots=1, deadband_weight=0.02)


def test_event_recompose_is_affected_only():
    trigger = decision_bar(scheduled_due=False, trend_event=True)
    assert trigger.should_recompose
    assert trigger.affected_only
    assert "event_trend" in trigger.reason_codes
    none = decision_bar(scheduled_due=False)
    assert not none.should_recompose


# --------------------------------------------------------------------------- fixed point (IT)


def test_greedy_fill_respects_book_heat():
    """Invariant IT: the greedy fill never exceeds H_book; the emitted book is feasible."""
    positions = [
        Position("A", wpc=0.05, ceiling_weight=0.20, incumbent_lots=0, target_weight=0.20,
                 priority=(0.0, 0.0, "A")),
        Position("B", wpc=0.05, ceiling_weight=0.20, incumbent_lots=0, target_weight=0.20,
                 priority=(1.0, 0.0, "B")),
    ]
    book, _w, heat, _deferred = greedy_lot_fill(positions, h_book=0.10, equity_ladder_cap=1e9)
    assert heat <= 0.10 + 1e-9
    assert sum(book.values()) == 2  # 0.10 / 0.05 = 2 lots total


def test_incumbents_reserved_first():
    """P4: incumbents are seated before any new lot and never dropped to make room."""
    positions = [
        Position("INC", wpc=0.05, ceiling_weight=0.05, incumbent_lots=1, target_weight=0.05,
                 priority=(0.0, 0.0, "INC")),
        Position("NEW", wpc=0.05, ceiling_weight=0.20, incumbent_lots=0, target_weight=0.20,
                 priority=(1.0, 0.0, "NEW")),
    ]
    book, _w, heat, _deferred = greedy_lot_fill(positions, h_book=0.10, equity_ladder_cap=1e9)
    assert book["INC"] == 1  # incumbent kept
    assert heat <= 0.10 + 1e-9


def test_oscillation_free_single_fill_terminates():
    """Invariant IT: the adversarial A-drops/B-fits setup terminates in one deterministic fill."""
    # Two equal-heat candidates that a naive integer loop would oscillate between: the monotone fill
    # admits in priority order and never revokes, so it terminates with ONE feasible book.
    positions = [
        Position("A", wpc=0.06, ceiling_weight=0.06, incumbent_lots=0, target_weight=0.06,
                 priority=(0.0, 0.0, "A")),
        Position("B", wpc=0.06, ceiling_weight=0.06, incumbent_lots=0, target_weight=0.06,
                 priority=(1.0, 0.0, "B")),
    ]
    book, _w, heat, deferred = greedy_lot_fill(positions, h_book=0.06, equity_ladder_cap=1e9)
    assert heat <= 0.06 + 1e-9
    assert sum(book.values()) == 1  # only one fits; the other defers (never oscillates)
    assert "B" in deferred


def test_fixed_point_converges_on_benign_inputs():
    """§15 fixed-point gate: Φ converges within r_max on benign inputs; feasible emitted book."""
    positions = [
        Position("A", wpc=0.05, ceiling_weight=0.10, incumbent_lots=0, target_weight=0.08,
                 priority=(0.0, 0.0, "A")),
    ]
    d = cfg().defaults
    fp = FixedPointComposer().solve(
        positions, h_book=0.20, equity_ladder_cap=1e9, eta=d.eta, r_max=d.r_max, tol=d.iterate_tol
    )
    assert fp.converged
    assert fp.iterations <= 5
    assert fp.heat_used <= 0.20 + 1e-9


def test_non_convergence_emits_feasible_fallback():
    """Invariant IT: an unsettled continuous relaxation emits the greedy-filled FEASIBLE book with
    converged=False and iterate_fallback."""
    # A ceiling_fn that flip-flops the ceilings so ‖Δw‖ never settles within r_max.
    positions = [
        Position("A", wpc=0.05, ceiling_weight=0.10, incumbent_lots=0, target_weight=0.10,
                 priority=(0.0, 0.0, "A")),
    ]
    state = {"i": 0}

    def ceiling_fn(_w):
        state["i"] += 1
        return np.array([0.10]) if state["i"] % 2 else np.array([0.02])

    fp = FixedPointComposer().solve(
        positions, h_book=0.20, equity_ladder_cap=1e9, eta=1.0, r_max=4, tol=1e-9,
        ceiling_fn=ceiling_fn,
    )
    assert fp.converged is False
    assert "iterate_fallback" in fp.reason_codes
    assert fp.heat_used <= 0.20 + 1e-9  # still feasible


# --------------------------------------------------------------------------- reconcile integration


def test_reconcile_demotes_dead_trend():
    positions = {"A": make_open_position("A", trend_state="death")}
    sel = make_proposed(keep=("B",), eligible=("B",))
    decisions = reconcile(
        positions, {"A": make_trend_action("reduce")}, sel,
        collapsed=frozenset(), qualified=frozenset(), delta_rank=2.0,
    )
    assert decisions["A"].verdict == "sel_demote_dead"
