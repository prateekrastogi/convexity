"""Staged rollout gates + OOS promotion + baselines (Invariant ROLLOUT, OOS, §11.3, §17)."""

from __future__ import annotations

import numpy as np

from convexity.ops.brake.rollout import (
    StageGateInput,
    demote_stage,
    evaluate_gate,
    next_stage,
)
from convexity.ops.validate.baselines import (
    cusum_delay,
    no_monitor_delay,
    shewhart_only_delay,
)
from convexity.ops.validate.oos import OosObjective, promote

_NS = dict(n_paper=30, n_canary=50, n_scaled=100)


def _ok(n_trades: int) -> StageGateInput:
    return StageGateInput(
        n_trades=n_trades, edge_in_ci=True, fill_in_ci=True, tca_in_ci=True, vol_in_ci=True,
        sprt_accepts_h0=True, recon_clean=True, no_kill_fired=True, all_gates_warmed=True,
        e_cost_cusum_breached=False,
    )


def test_advance_requires_min_sample_not_calendar():
    """Invariant ROLLOUT: a below-min sample DEFERS regardless of elapsed time."""
    assert evaluate_gate("paper", _ok(10), **_NS).decision == "defer"
    assert evaluate_gate("paper", _ok(40), **_NS).decision == "advance"


def test_warming_up_defers_advance():
    """§17.1/§6.8: a stream still in warm-up DEFERS (a freshly promoted config does not page/advance)."""
    ev = _ok(40)
    from dataclasses import replace
    ev = replace(ev, all_gates_warmed=False)
    assert evaluate_gate("paper", ev, **_NS).decision == "defer"


def test_e_cost_cusum_breach_blocks_and_demotes():
    """§17.1 (Invariant XC): an e_cost CUSUM breach blocks advance and demotes."""
    from dataclasses import replace
    ev = replace(_ok(40), e_cost_cusum_breached=True)
    assert evaluate_gate("canary", ev, **_NS).decision == "demote"


def test_kill_demotes_immediately():
    from dataclasses import replace
    ev = replace(_ok(200), no_kill_fired=False)
    assert evaluate_gate("scaled", ev, **_NS).decision == "demote"


def test_below_ci_holds_not_advance():
    from dataclasses import replace
    ev = replace(_ok(40), edge_in_ci=False)
    assert evaluate_gate("paper", ev, **_NS).decision == "hold"


def test_stage_transitions():
    assert next_stage("paper") == "canary"
    assert next_stage("full") == "full"
    assert demote_stage("scaled") == "canary"
    assert demote_stage("paper") == "paper"


def test_oos_objective_has_no_pnl_term():
    """Invariant OOS: the promotion objective contains NO P&L term; swapping P&L for noise leaves the
    promoted limit unchanged (there is no P&L input to change)."""
    obj = OosObjective(realized_far=0.003, far_target=0.003, realized_delay=10.0, delay_target=10.0)
    assert obj.contains_pnl_term() is False
    cands = {
        "tight": OosObjective(0.01, 0.003, 8.0, 10.0),     # floods (far off budget)
        "good": OosObjective(0.0031, 0.003, 11.0, 10.0),   # near budget, slightly slow
        "blind": OosObjective(0.0001, 0.003, 40.0, 10.0),  # too slow
    }
    assert promote(cands) == "good"


def test_panel_beats_shewhart_only_on_small_drift():
    """§11.3: the sequential CUSUM beats the NAIVE-3σ-only baseline on a small persistent drift (which
    Shewhart-only is blind to) and both beat NO-MONITOR."""
    # a small SUSTAINED step (≈0.8σ, sub-3σ per-point) the sequential CUSUM integrates but the
    # single-point Shewhart-3σ is blind to (§11.3).
    import numpy as np
    rng = np.random.default_rng(1)
    n, change_at = 120, 20
    stream = rng.standard_normal(n)
    stream[change_at:] += 0.8
    cd = cusum_delay(stream, change_at)
    sd = shewhart_only_delay(stream, change_at, L_s=3.0)
    nm = no_monitor_delay(stream)
    assert cd < sd                # CUSUM detects the small persistent drift; Shewhart-only is blind
    assert cd < nm                # and far faster than no monitor (which never detects)


def test_ar1_student_t_synth_generates_dependent_fat_tail():
    """§6.7d synth: the AR(1)-Student-t in-control stream is autocorrelated AND fat-tailed."""
    from convexity.ops.synth.generate import in_control_ar1_student_t
    x = in_control_ar1_student_t(50000, rho=0.5, dof=5.0, seed=0)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag1 > 0.3                                  # autocorrelated
    assert (np.abs(x) > 3.0).mean() > 2 * 0.0027       # fatter tail than Gaussian
