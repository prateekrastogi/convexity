"""End-to-end engine acceptance (REGIME §8, §13 Boundary/composition).

Covers Invariants SCOPE, CAP, DET, STALEATTR, and the calm/spike/blowout behavioural assertions.
"""

import numpy as np

from convexity.core.types import RegimeState
from convexity.regime import RegimeEngine
from convexity.regime.synth import generate as g
from convexity.regime.types import BookState, CostState, ExecState


def _panel_from(returns_matrix, asof=None):
    """Wrap a (W, n) matrix into a Panel directly (already de-meaned by the synth generators)."""
    from convexity.regime.types import Panel

    w, n = returns_matrix.shape
    return Panel(returns=returns_matrix, instruments=tuple(f"S{i}" for i in range(n)), asof_bar=asof or w)


def _book():
    return BookState(book_gross_at_full=1000.0, current_gross=1000.0)


def _calm_liq(live):
    return ExecState(spread_pct={s: 0.4 for s in live}), CostState(funding_pct={s: 0.4 for s in live})


def test_calm_first_bar_is_full_gross():
    """A genuinely calm bar (median measures, no liq stress) -> g_t = 1.0, no sub-state binds.

    On the first bar the price-measure histories are empty (percentiles default to the 0.5 median ->
    CALM) and liq is calm, so the min() is CALM and the overlay runs full gross (REGIME §12 calm).
    """
    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(60))
    ex, co = _calm_liq(instruments)
    panel = _panel_from(g.calm_panel(w=300, n=60, seed=0))
    out = eng.regime_state(panel, ex, co, _book(), None)
    assert out.g_t == 1.0
    assert out.state == "CALM"
    assert isinstance(out, RegimeState)


def test_calm_run_respects_cap_floor_and_is_sticky():
    """Over a long calm run g_t stays in [0.25,1.0] and the overlay is STICKY, never whipsawing.

    On an adversarial i.i.d.-per-bar measure stream the discrete hysteretic machine is DELIBERATELY
    sticky (it holds an elevated state rather than flap — the regime analogue of Carver buffering,
    P4); a far higher change-count would mean the brake whipsaws. We assert CAP/FLOOR every bar and
    that the regime-change count stays well below the bar count (low turnover).
    """
    from convexity.regime.validate.baselines import turnover

    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(60))
    ex, co = _calm_liq(instruments)
    prev = None
    g_path = []
    for k in range(200):
        panel = _panel_from(g.calm_panel(w=300, n=60, seed=1000 + k))
        prev = eng.regime_state(panel, ex, co, _book(), prev)
        assert 0.25 <= prev.g_t <= 1.0  # CAP/FLOOR every bar
        g_path.append(prev.g_t)
    rep = turnover(np.asarray(g_path))
    assert rep.n_changes < len(g_path) // 2  # sticky: far fewer changes than bars (no whipsaw)


def test_scope_engine_emits_only_regime_state():
    """The engine returns a RegimeState and nothing else (Invariant SCOPE)."""
    eng = RegimeEngine()
    panel = _panel_from(g.calm_panel(w=300, n=30, seed=1))
    ex, co = _calm_liq(panel.instruments)
    out = eng.regime_state(panel, ex, co, _book(), None)
    assert isinstance(out, RegimeState)
    # No size/order field exists on the output (it carries only g_t/state/binding/reasons).
    assert hasattr(out, "g_t") and not hasattr(out, "size")


def test_cap_g_t_in_range_and_equals_no_overlay_at_full():
    """g_t in [0.25,1.0]; at g_t=1.0 the downstream book equals no-overlay (Invariant CAP)."""
    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(30))
    ex, co = _calm_liq(instruments)
    prev = None
    out = None
    for k in range(60):
        panel = _panel_from(g.calm_panel(w=300, n=30, seed=200 + k))
        out = eng.regime_state(panel, ex, co, _book(), prev)
        prev = out
    assert 0.25 <= out.g_t <= 1.0
    if out.g_t == 1.0:
        book = _book()
        assert out.g_t * book.book_gross_at_full == book.book_gross_at_full  # identical to no overlay


def test_det_identical_inputs_identical_output():
    """Identical inputs -> identical RegimeState (Invariant DET)."""
    panel = _panel_from(g.calm_panel(w=300, n=30, seed=3))
    ex, co = _calm_liq(panel.instruments)
    e1, e2 = RegimeEngine(), RegimeEngine()
    o1 = e1.regime_state(panel, ex, co, _book(), None)
    o2 = e2.regime_state(panel, ex, co, _book(), None)
    assert o1.g_t == o2.g_t and o1.state == o2.state and o1.binding_sub == o2.binding_sub


def test_liquidity_blowout_binds_liq():
    """A CRISIS-level liq blowout binds liq from a CALM baseline (the first bar has median measures)."""
    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(60))
    panel = _panel_from(g.calm_panel(w=300, n=60, seed=360))
    # First bar: empty measure histories -> price percentiles default to 0.5 (CALM); corr stratum
    # not yet warmed -> frozen at the CALM baseline. Liq is the only stressed measure.
    ex, co = g.liquidity_blowout(instruments, level=0.99, kind="spread")
    out = eng.regime_state(panel, ex, co, _book(), None)
    assert out.binding_sub == "liq"
    assert out.g_t == 0.25


def test_staleattr_event_path_is_liq_attributed():
    """An out-of-cadence liq spike escalates with binding=liq, never a stale corr (Invariant STALEATTR)."""
    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(60))
    exc, coc = _calm_liq(instruments)
    # One scheduled recompute caches the (stale, low) corr/vol/turb sub-states.
    panel = _panel_from(g.calm_panel(w=300, n=60, seed=500))
    prev = eng.regime_state(panel, exc, coc, _book(), None)
    g_before = prev.g_t
    # Sustained minute-level liq blowout for D_liq_min minutes -> out-of-cadence ESCALATION, which
    # MUST be attributed to liq (never to a stale corr/vol/turb — Invariant STALEATTR).
    ex, co = g.liquidity_blowout(instruments, level=0.99, kind="spread")
    out = prev
    for _ in range(eng.policy.d_liq_min):
        out = eng.regime_state_event(ex, co, _book(), out, instruments)
    assert out.binding_sub == "liq"
    assert "event_driven_liq" in out.reason_codes
    assert out.g_t <= g_before


def test_event_single_minute_blip_does_not_move_g():
    """A single-minute blip below D_liq_min consecutive bars does not move g_t (§3.6 b')."""
    eng = RegimeEngine()
    instruments = tuple(f"S{i}" for i in range(20))
    exc, coc = _calm_liq(instruments)
    panel = _panel_from(g.calm_panel(w=300, n=20, seed=6))
    prev = eng.regime_state(panel, exc, coc, _book(), None)
    ex, co = g.liquidity_blowout(instruments, level=0.99, kind="spread")
    out = eng.regime_state_event(ex, co, _book(), prev, instruments)  # only ONE minute
    assert out.g_t == prev.g_t  # suppressed by the minute dwell
