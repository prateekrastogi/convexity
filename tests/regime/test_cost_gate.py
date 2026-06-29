"""Cost-gate acceptance gate (REGIME §13 Cost gate, Invariant COSTGATE)."""

import pytest

from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.state.cost_gate import cost_gate
from convexity.regime.types import BookState


def test_escalation_never_blocked_by_cost():
    """Cutting gross DOWN for safety is never cost-blocked (REGIME §5.2 asymmetry)."""
    book = BookState(book_gross_at_full=1000.0, current_gross=750.0)
    # Escalation NORMAL->CRISIS, with an absurd cost estimator that would block a de-escalation.
    res = cost_gate(
        g_prev=0.75, g_new=0.25, transition="escalate", book=book, cfg=DEFAULTS,
        round_trip_estimate=lambda dg: 1e9,
    )
    assert res.passed is True
    assert res.g_emit == 0.25
    assert res.delta_gross == pytest.approx(500.0)


def test_deescalation_deadband_defers_drifted_gross_6_3():
    """§6.3: current_gross drifted to 120 == g_new*full -> delta_gross 0 < band -> DEFER (clause b)."""
    book = BookState(book_gross_at_full=160.0, current_gross=120.0)
    res = cost_gate(
        g_prev=0.50, g_new=0.75, transition="deescalate", book=book, cfg=DEFAULTS,
    )
    # target_gross_new = 0.75*160 = 120 ; delta = |120-120| = 0 < 0.10*120 = 12 -> deferred.
    assert res.delta_gross == pytest.approx(0.0)
    assert res.passed is False
    assert res.reason == "regime_change_cost_gated"
    assert res.g_emit == 0.50  # held at g_prev


def test_deescalation_cost_cap_blocks_expensive_regross():
    """Re-grossing UP must clear the cost cap (REGIME §5.2 clause c)."""
    book = BookState(book_gross_at_full=1000.0, current_gross=500.0)
    # delta_gross = |0.75*1000 - 500| = 250, deadband 0.10*500=50 cleared; cost > 0.5*250=125 blocks.
    res = cost_gate(
        g_prev=0.50, g_new=0.75, transition="deescalate", book=book, cfg=DEFAULTS,
        round_trip_estimate=lambda dg: 200.0,
    )
    assert res.passed is False
    assert res.reason == "regime_change_cost_gated"


def test_deescalation_passes_when_cheap():
    book = BookState(book_gross_at_full=1000.0, current_gross=500.0)
    res = cost_gate(
        g_prev=0.50, g_new=0.75, transition="deescalate", book=book, cfg=DEFAULTS,
        round_trip_estimate=lambda dg: 1.0,
    )
    assert res.passed is True
    assert res.g_emit == 0.75
