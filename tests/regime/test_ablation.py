"""Acceptance reports: binding ablation + the realized ARL audit (REGIME §9, §3.6d, Invariant V).

These exercise ``validate/ablation.py`` end-to-end against a real engine run on synth panels:
  * the realized per-bar false-escalation rate on a CALM-NULL stretch stays within the
    ``alpha_eff``/``ARL0`` budget (a too-tight threshold would over-fire on noise);
  * a TRUE correlation shift is detected with ``ARL1 ≈ 1`` (escalation is immediate, §3.2).
"""

from collections import Counter

import numpy as np

from convexity.regime import RegimeEngine
from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.synth import generate as g
from convexity.regime.types import CostState, ExecState
from convexity.regime.validate.ablation import arl_audit, binding_ablation, detection_delay


def _calm_liq(live):
    return (
        ExecState(spread_pct={s: 0.4 for s in live}),
        CostState(funding_pct={s: 0.4 for s in live}),
    )


def _panel(matrix, asof=None):
    from convexity.regime.types import Panel

    w, n = matrix.shape
    return Panel(returns=matrix, instruments=tuple(f"S{i}" for i in range(n)), asof_bar=asof or w)


def _book():
    from convexity.regime.types import BookState

    return BookState(book_gross_at_full=1000.0, current_gross=1000.0)


def test_arl_audit_calm_null_realized_rate_within_budget():
    """On a CALM-NULL stretch the realized per-state escalation rate stays within the ARL0 budget.

    The calm null is a STATIONARY calm regime (the market sits at its calm reference across many
    decision bars), so the percentile measures sit at their median and no rung over-fires — the
    realized rate is within ``alpha_eff`` for every rung (REGIME §3.6d: a rate materially ABOVE
    budget would mean the threshold is too tight). The companion ``test_detection_delay_*`` proves
    the brake is NOT merely dead — it fires immediately on a TRUE shift — so the pair is
    discriminating (a frozen brake would pass this but fail that)."""
    eng = RegimeEngine()
    live = tuple(f"S{i}" for i in range(40))
    ex, co = _calm_liq(live)
    n_bars = 200
    calm = g.calm_panel(w=300, n=40, seed=4242)  # one stationary calm regime, observed many bars

    esc: dict[str, Counter] = {m: Counter() for m in ("vol", "turb", "liq")}
    states = []
    prev = None
    for _ in range(n_bars):
        prev = eng.regime_state(_panel(calm), ex, co, _book(), prev)
        states.append(prev)
        if prev.transition == "escalate" and prev.binding_sub in esc:
            esc[prev.binding_sub][prev.state] += 1

    # Binding ablation is computable over the run (firewalled report, REGIME §9).
    counts = binding_ablation(states)
    assert sum(counts.values()) == n_bars

    # ARL audit: realized per-bar escalation rate per state must NOT exceed its alpha_eff budget.
    for measure in ("vol", "turb", "liq"):
        realized = dict(esc[measure])
        rows = arl_audit(measure, realized, n_bars, DEFAULTS)
        assert rows  # one row per escalation rung
        for row in rows:
            assert 0.0 <= row.realized_rate <= row.budget_alpha_eff + 1e-9, (
                f"{measure}/{row.state} over-fired on the calm null: "
                f"{row.realized_rate} > budget {row.budget_alpha_eff}"
            )
            assert row.realized_count == realized.get(row.state, 0)
            assert row.n_bars == n_bars


def test_arl_audit_arithmetic_matches_budget_rows():
    """arl_audit threads the arl0_budget rows through: realized_rate = count/n_bars exactly."""
    rows = arl_audit("vol", {"NORMAL": 3, "STRESSED": 1}, n_bars=100, cfg=DEFAULTS)
    by_state = {r.state: r for r in rows}
    assert by_state["NORMAL"].realized_rate == 0.03
    assert by_state["STRESSED"].realized_rate == 0.01
    assert by_state["CRISIS"].realized_rate == 0.0  # no count -> zero
    assert all(r.budget_arl0 > 0 for r in rows)


def test_detection_delay_true_shift_is_immediate():
    """A genuine correlation-to-1 shift cuts g_t with ARL1 ≈ 1 (escalation is immediate, §3.2/§9)."""
    eng = RegimeEngine()
    live = tuple(f"S{i}" for i in range(40))
    ex, co = _calm_liq(live)

    g_path = []
    prev = None
    # Stationary calm warm-up so the absorption stratum has a valid delta_AR baseline and g_t=1.0.
    shift_bar = 60
    calm = g.calm_panel(w=300, n=40, seed=7000)
    for _ in range(shift_bar):
        prev = eng.regime_state(_panel(calm), ex, co, _book(), prev)
        g_path.append(prev.g_t)
    assert g_path[-1] == 1.0  # calm => no cut yet

    # A true, sustained correlation-to-1 spike from shift_bar onward (same n_t=40 band as warm-up).
    spike = g.correlation_spike_panel(w=300, n=40, seed=7777, rho=0.9)
    for _ in range(40):
        prev = eng.regime_state(_panel(spike), ex, co, _book(), prev)
        g_path.append(prev.g_t)

    delay = detection_delay(g_path, shift_bar)
    assert delay >= 1
    assert delay <= 3  # detected within a couple of bars of the genuine shift (ARL1 ~ 1)
    assert min(g_path[shift_bar:]) < 1.0  # the book is actually cut after the shift
    assert np.isfinite(prev.g_t)
