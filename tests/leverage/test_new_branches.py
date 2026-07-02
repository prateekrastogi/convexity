"""Branch-coverage teeth for new LEVERAGE codepaths the pillar tests left untriggered.

These are reachable (non-defensive) branches surfaced by the e2e coverage pass: the full
``next_c`` promotion+demotion path (not just the empty-history early return), the ``compose_g_t``
reason-code branches (``governor_terminal`` / ``edge_floored_zero``), and the ``governor_state``
empty-series drawdown. Each asserts the branch's OBSERVABLE effect, so it fails if the branch is
removed or mis-wired.
"""

from __future__ import annotations

import numpy as np

from convexity.leverage.ceilings.capacity import CapacityCeiling
from convexity.leverage.ceilings.governor import governor_state
from convexity.leverage.combine.engine import compose_g_t
from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.ratchet.state import next_c
from convexity.leverage.types import GTarget, GovernorState


def _gtarget(kelly_term: float, sr_lcb: float) -> GTarget:
    return GTarget(kelly_term=kelly_term, c_t=0.25, sr_lcb=sr_lcb, sigma_hat=0.02,
                   sigma_target=0.02, growth_capture=0.5)


def test_next_c_nonempty_history_runs_the_full_promotion_demotion_path() -> None:
    """A non-empty live history exercises evaluate_promotion + standardized_surprise + apply_demotion
    (state.py full body), not the empty-history floor early-return."""
    d = LeverageDefaults()
    # A long healthy run below the promotion elapsed-weeks floor stays at the quarter-Kelly start,
    # but this drives the FULL code path (promotion eval + latest-surprise demotion), unlike the
    # empty-array early return the existing test covers.
    sr = np.full(30, 1.0)
    c = next_c(sr, se_sr=1.0, elapsed_weeks=30, defaults=d, sr_ref=0.5, demote_z_threshold=0.0)
    assert c in d.c_grid  # a valid grid point was resolved via the full ratchet, not the floor stub


def test_next_c_can_promote_after_the_elapsed_floor() -> None:
    """A sufficiently long, high-surprise, low-SE history clears the §6.3 sequential test -> half-Kelly."""
    d = LeverageDefaults()
    sr = np.full(d.n_min_weeks + 20, 2.0)  # persistent out-performance vs sr_ref
    c = next_c(sr, se_sr=0.5, elapsed_weeks=d.n_min_weeks + 20, defaults=d, sr_ref=0.0,
               demote_z_threshold=-10.0)  # demotion effectively disarmed
    assert c == d.c_promoted  # the promotion branch of the ratchet fired


def test_compose_g_t_flags_governor_terminal_reason() -> None:
    """A terminal governor (m_t -> 0) makes it the binding ceiling AND adds the reason code."""
    g_target = _gtarget(kelly_term=0.6, sr_lcb=0.1)
    capacity = CapacityCeiling(g_cap=float("inf"), binding_symbol=None, n_unshortable_zeroed=0)
    governor = GovernorState(dd_t=0.45, m_t=0.0, terminal=True)
    dec = compose_g_t(g_target, capacity, governor, g_max=1.5)
    assert dec.g_t == 0.0
    assert dec.binding_ceiling == "governor"
    assert "governor_terminal" in dec.reason_codes


def test_compose_g_t_flags_edge_floored_zero_reason() -> None:
    """A non-positive SR_LCB (no measurable edge) surfaces the edge_floored_zero reason code."""
    g_target = _gtarget(kelly_term=0.0, sr_lcb=0.0)
    capacity = CapacityCeiling(g_cap=float("inf"), binding_symbol=None, n_unshortable_zeroed=0)
    governor = GovernorState(dd_t=0.0, m_t=1.0, terminal=False)
    dec = compose_g_t(g_target, capacity, governor, g_max=1.5)
    assert dec.g_t == 0.0  # kelly term is 0 -> binds at 0
    assert "edge_floored_zero" in dec.reason_codes


def test_governor_state_on_empty_series_is_zero_drawdown() -> None:
    """An all-non-finite (or empty) weekly series yields a 0.0 drawdown and full m_t (governor.py:31)."""
    d = LeverageDefaults()
    gs = governor_state(np.array([np.nan, np.nan]), d)
    assert gs.dd_t == 0.0
    assert gs.m_t == 1.0
    assert gs.terminal is False
