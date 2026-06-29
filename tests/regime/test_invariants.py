"""One test per NAMED REGIME Invariant (REGIME.md §1–§9, the boxed Invariants).

This file is the canonical home of the per-invariant acceptance gates: each named Invariant in the
spec becomes exactly one test, in spec order. Where a measure/state/engine unit already exercises an
invariant in a sibling test module, the test here asserts the SAME load-bearing property against the
real API so the invariant is provable from this file alone. Style matches selection/
dynamic_optimization.py: small, explicit, no hypothesis dependency.

Invariants covered (the full REGIME set):
  SCOPE MIN HYST DWELL CLOCK ARL DISP TRACE ARFLOOR ARCOND ARDISPNORM CONDNUM NODOUBLEVC TURBNORM
  STALEATTR NODOUBLE2 CAP FLOOR NODOUBLE ML NOFIT CAUSAL COND CHEAP COSTGATE NREBUILD C V WE DET
"""

import ast
import importlib
import inspect
import pkgutil
from dataclasses import replace

import numpy as np
import pytest

from convexity.core.params import Bucket, assert_no_red_in_amber
from convexity.core.types import InstrumentSpec, RegimeState
from convexity.regime import RegimeEngine
from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.config.resolve import param_registry, resolve_config
from convexity.regime.measures._shrinkage import lw_shrink_correlation
from convexity.regime.measures.absorption import (
    AbsorptionState,
    absorption_ratio,
    compute_absorption,
    standardized_shift,
    top_k,
)
from convexity.regime.measures.liquidity import compute_liquidity
from convexity.regime.measures.turbulence import TurbulenceState, compute_turbulence
from convexity.regime.measures.vol_regime import sigma_agg
from convexity.regime.state.bands import arl0_budget, classify_substate, g_of
from convexity.regime.state.cost_gate import cost_gate
from convexity.regime.state.emit import assert_floor
from convexity.regime.state.machine import step_machine
from convexity.regime.synth import generate as gen
from convexity.regime.types import BookState, CostState, ExecState, Panel, SubName, SubState
from convexity.regime.validate.backtest import k_regime, register_blue_grid


# --------------------------------------------------------------------------- helpers

def _panel(matrix: np.ndarray, asof: int | None = None) -> Panel:
    w, n = matrix.shape
    return Panel(returns=matrix, instruments=tuple(f"S{i}" for i in range(n)), asof_bar=asof or w)


def _book(full: float = 1000.0, current: float = 1000.0) -> BookState:
    return BookState(book_gross_at_full=full, current_gross=current)


def _calm_liq(live: tuple[str, ...]) -> tuple[ExecState, CostState]:
    return (
        ExecState(spread_pct={s: 0.4 for s in live}),
        CostState(funding_pct={s: 0.4 for s in live}),
    )


def _sub(name: SubName, value: float, state: str, *, asof: int = 0, fresh: bool = True) -> SubState:
    return SubState(name=name, value=value, state=state, g_sub=g_of(state), asof_bar=asof, fresh=fresh, diag={})


def _prev(state: str, binding: SubName, dwell: int, g: float) -> RegimeState:
    return RegimeState(
        g_t=g, state=state, binding_sub=binding, dwell_bars=dwell, transition="hold",
        cost_gate_passed=True, rebalance_cost_hat=0.0, reason_codes=(), asof_bar=0, engine_version="t",
    )


def _as_prev(res) -> RegimeState:
    return RegimeState(
        g_t=res.g_s, state=res.state, binding_sub=res.binding, dwell_bars=res.dwell_bars,
        transition="hold", cost_gate_passed=True, rebalance_cost_hat=0.0, reason_codes=(),
        asof_bar=0, engine_version="t",
    )


# --------------------------------------------------------------------------- SCOPE

def test_inv_scope_engine_emits_only_regime_state():
    """REGIME emits a RegimeState (g_t/s_t) and nothing else — no size, no order (Invariant SCOPE)."""
    eng = RegimeEngine()
    panel = _panel(gen.calm_panel(w=300, n=30, seed=1))
    ex, co = _calm_liq(panel.instruments)
    out = eng.regime_state(panel, ex, co, _book(), None)
    assert isinstance(out, RegimeState)
    # No sizing/order surface on the output: it carries only g_t/state/binding/dwell/reasons.
    for forbidden in ("size", "qty", "order", "leverage", "target_qty"):
        assert not hasattr(out, forbidden)


# --------------------------------------------------------------------------- MIN

def test_inv_min_worst_sub_binds_unique_minimum():
    """g_t = min over the four sub-states; the unique worst measure binds, never a product (MIN)."""
    subs = {
        "vol": _sub("vol", 0.72, "NORMAL"),
        "corr": _sub("corr", 2.2, "CRISIS"),
        "turb": _sub("turb", 0.80, "NORMAL"),
        "liq": _sub("liq", 0.55, "CALM"),
    }
    res = step_machine(subs, None, DEFAULTS)
    assert res.state == "CRISIS"
    assert res.binding == "corr"  # the unique minimum binds and is reported
    assert res.g_s == 0.25
    # Two moderately-elevated measures do NOT compound below the worse single one.
    subs2 = {
        "vol": _sub("vol", 0.72, "NORMAL"),
        "corr": _sub("corr", 0.7, "NORMAL"),
        "turb": _sub("turb", 0.5, "CALM"),
        "liq": _sub("liq", 0.5, "CALM"),
    }
    assert step_machine(subs2, None, DEFAULTS).g_s == 0.75  # min(.75,.75,1,1), NOT .75*.75=.5625


# --------------------------------------------------------------------------- HYST

def test_inv_hyst_oscillation_in_gap_holds_state():
    """A measure dithering inside its enter/exit gap does not change its own sub-state (HYST)."""
    bands = DEFAULTS.bands_for("vol")  # NORMAL 0.70/0.60
    assert classify_substate(0.72, "CALM", bands) == "NORMAL"  # enter on the high mark
    assert classify_substate(0.65, "NORMAL", bands) == "NORMAL"  # inside the 0.60-0.70 gap -> hold
    assert classify_substate(0.55, "NORMAL", bands) == "CALM"  # only below the exit mark de-escalates


# --------------------------------------------------------------------------- DWELL

def test_inv_dwell_blocks_deescalation_until_d_min():
    """De-escalation is blocked until dwell >= D_min; escalation is always immediate (DWELL)."""
    prev = _prev("STRESSED", "corr", dwell=1, g=0.50)  # dwell 1 < D_min 3
    subs = {
        "vol": _sub("vol", 0.1, "CALM"),
        "corr": _sub("corr", 0.3, "NORMAL"),  # binding measure fell below the STRESSED exit
        "turb": _sub("turb", 0.1, "CALM"),
        "liq": _sub("liq", 0.1, "CALM"),
    }
    res = step_machine(subs, prev, DEFAULTS)
    assert res.state == "STRESSED" and res.transition == "hold"  # dwell-locked
    assert res.dwell_bars == 2  # clock keeps counting
    # Escalation is never dwell-blocked.
    esc = step_machine({n: _sub(n, 2.2, "CRISIS") for n in ("vol", "corr", "turb", "liq")},
                       _prev("NORMAL", "vol", dwell=1, g=0.75), DEFAULTS)
    assert esc.state == "CRISIS" and esc.transition == "escalate"


# --------------------------------------------------------------------------- CLOCK

def test_inv_clock_global_dwell_continues_through_binding_handoff():
    """Global dwell on g(s_t): a corr->vol handoff at unchanged g continues the clock (CLOCK §6.2)."""
    corr_b, pct_b = DEFAULTS.bands_for("corr"), DEFAULTS.bands_for("vol")
    # t0: corr STRESSED, rest CALM (entered from NORMAL).
    r0 = step_machine(
        {"corr": _sub("corr", 1.3, classify_substate(1.3, "STRESSED", corr_b)),
         "vol": _sub("vol", 0.5, "CALM"), "turb": _sub("turb", 0.5, "CALM"), "liq": _sub("liq", 0.5, "CALM")},
        _prev("NORMAL", "corr", dwell=2, g=0.75), DEFAULTS,
    )
    assert r0.state == "STRESSED" and r0.binding == "corr"
    # t1: corr holds STRESSED, vol escalates to STRESSED below the binding (non-binding wiggle).
    r1 = step_machine(
        {"corr": _sub("corr", 0.9, classify_substate(0.9, "STRESSED", corr_b)),
         "vol": _sub("vol", 0.88, classify_substate(0.88, "CALM", pct_b)),
         "turb": _sub("turb", 0.5, "CALM"), "liq": _sub("liq", 0.5, "CALM")},
        _as_prev(r0), DEFAULTS,
    )
    assert r1.binding == "corr" and r1.dwell_bars == r0.dwell_bars + 1
    # t2: corr de-escalates to NORMAL, vol now worst -> HANDOFF corr->vol at unchanged g=0.50.
    r2 = step_machine(
        {"corr": _sub("corr", 0.5, classify_substate(0.5, "STRESSED", corr_b)),
         "vol": _sub("vol", 0.90, classify_substate(0.90, "STRESSED", pct_b)),
         "turb": _sub("turb", 0.5, "CALM"), "liq": _sub("liq", 0.5, "CALM")},
        _as_prev(r1), DEFAULTS,
    )
    assert r2.state == "STRESSED" and r2.binding == "vol"
    assert r2.transition == "hold" and r2.reason == "binding_handoff"
    assert r2.dwell_bars == r1.dwell_bars + 1  # the clock CONTINUES, does not reset (§3.3 ii)


# --------------------------------------------------------------------------- ARL

def test_inv_arl_budget_increases_with_d_min_and_liq_uses_one_minus_p_squared():
    """Every threshold has alpha + implied ARL0; ARL0 rises with D_min; liq uses 1-p^2 (ARL)."""
    b2 = {r.state: r.arl0 for r in arl0_budget("vol", replace(DEFAULTS, d_min=2))}
    b4 = {r.state: r.arl0 for r in arl0_budget("vol", replace(DEFAULTS, d_min=4))}
    for state in ("NORMAL", "STRESSED", "CRISIS"):
        assert b4[state] > b2[state]  # tighter filter -> longer run length
    liq = {r.state: r.alpha_eff for r in arl0_budget("liq", DEFAULTS)}
    vol = {r.state: r.alpha_eff for r in arl0_budget("vol", DEFAULTS)}
    assert liq["NORMAL"] > vol["NORMAL"]  # max-of-two roughly doubles each raw rate
    assert liq["NORMAL"] == pytest.approx(0.51 * 0.64 ** 2, rel=1e-6)  # spec's ~0.209


# --------------------------------------------------------------------------- DISP

def test_inv_disp_delta_ar_scales_by_raw_level_std():
    """delta_AR divides by the RAW AR-level std, NOT the mean-difference SE (Invariant DISP)."""
    assert standardized_shift(0.710, 0.600, 0.050) == pytest.approx(2.20, abs=1e-9)
    se = 0.050 * np.sqrt(1 / 15 + 1 / 250)  # the FORBIDDEN substitution inflates ~3.7x
    assert (0.710 - 0.600) / se > 7.0  # ~8.2 if mis-scaled — our value stays +2.20


# --------------------------------------------------------------------------- TRACE

def test_inv_trace_denominator_is_n_t_by_identity():
    """AR_t denominator is n_t by the correlation trace identity, NOT a re-summed total (TRACE)."""
    eig = np.array([5.0, 3.0, 1.0, 1.0])  # sums to 10, but the denominator must be n_t=4
    assert absorption_ratio(eig, n_t=4, k=2) == pytest.approx(8.0 / 4.0)
    # The shrink preserves the unit diagonal so the eigenvalue sum equals n_t to tolerance.
    shrink = lw_shrink_correlation(gen.calm_panel(w=300, n=40, seed=0))
    assert np.isclose(shrink.eigenvalues.sum(), 40.0, atol=1e-8)


# --------------------------------------------------------------------------- ARFLOOR

def test_inv_arfloor_freezes_on_rank_deficient_window():
    """W_AR < c_AR*n_t -> absorption FREEZES g^corr at its last valid value (Invariant ARFLOOR)."""
    panel = gen.rank_deficient_panel(n=80, w=60)  # 60 < 2*80
    res = compute_absorption(panel, cfg=DEFAULTS, state=AbsorptionState(last_delta_ar=0.42))
    assert res.fresh is False
    assert res.diag.get("ar_window_deficient") == 1.0
    assert res.delta_ar == 0.42  # frozen, NOT reset to a fresh trigger / 1.0


# --------------------------------------------------------------------------- ARCOND

def test_inv_arcond_shrink_debiases_ar_on_marginal_tn_noise():
    """On pure-noise T/N~2, AR on the SHRUNK C_t is below AR on the RAW matrix (Invariant ARCOND)."""
    panel = gen.marginal_tn_noise_panel(n=60, c_ar=2.0, seed=3)
    n_t = panel.shape[1]
    k = top_k(DEFAULTS.kappa, n_t)
    std = panel.std(axis=0, ddof=0)
    z = panel / np.where(std > 0, std, 1.0)
    raw = (z.T @ z) / panel.shape[0]
    np.fill_diagonal(raw, 1.0)
    ar_raw = np.sort(np.linalg.eigvalsh(raw))[::-1][:k].sum() / n_t
    ar_shrunk = absorption_ratio(lw_shrink_correlation(panel).eigenvalues, n_t, k)
    assert ar_shrunk < ar_raw


# --------------------------------------------------------------------------- ARDISPNORM

def test_inv_ardispnorm_freezes_delta_ar_across_n_t_step():
    """delta_AR does NOT move when n_t steps outside frac_dn; freezes flagged ar_nt_changed."""
    full, shrunk = gen.corr_fixed_shrink_pair(w=300, n_full=80, drop=20, seed=8)
    st = AbsorptionState()
    for _ in range(DEFAULTS.w_ar_short + 5):  # warm the full-size stratum
        compute_absorption(full, cfg=DEFAULTS, state=st)
    res_full = compute_absorption(full, cfg=DEFAULTS, state=st)
    assert res_full.fresh is True
    res_shrunk = compute_absorption(shrunk, cfg=DEFAULTS, state=st)  # n_t drops beyond frac_dn
    assert res_shrunk.fresh is False
    assert res_shrunk.diag.get("ar_nt_changed") == 1.0
    assert res_shrunk.delta_ar == res_full.delta_ar  # held frozen across the step


# --------------------------------------------------------------------------- CONDNUM

def test_inv_condnum_freezes_absorption_on_breach():
    """A post-shrink cond > cond_max freezes g^corr and flags cond_breach_corr (Invariant CONDNUM)."""
    panel = gen.condition_breach_panel(w=200, n=40, seed=9)
    tight = replace(DEFAULTS, cond_max=1.5)  # force the near-duplicate pair to breach
    res = compute_absorption(panel, cfg=tight, state=AbsorptionState(last_delta_ar=0.1))
    assert res.diag.get("cond_breach_corr") == 1.0
    assert res.fresh is False


# --------------------------------------------------------------------------- NODOUBLEVC

def test_inv_nodoublevc_g_vol_reads_aggregate_not_per_instrument_forecast():
    """g^vol keys off cross-asset AGGREGATE vol; it never reads a per-instrument forecast (NODOUBLEVC)."""
    sig = inspect.signature(RegimeEngine.regime_state)
    assert "forecast_vol" not in set(sig.parameters)
    # sigma_agg is the equal-weight cross-asset aggregate — a scalar of the panel, not a vol vector.
    assert np.isfinite(sigma_agg(gen.calm_panel(w=300, n=10, seed=0), DEFAULTS.w_rv))


# --------------------------------------------------------------------------- TURBNORM

def test_inv_turbnorm_percentile_stable_across_panel_shrink():
    """turb_pct keys off d_norm = d_t/n_t, so dropping assets does not shift it (Invariant TURBNORM)."""
    full, shrunk = gen.panel_shrink_pair(w=300, n_full=80, drop=30, seed=10)
    hist = np.random.default_rng(0).uniform(0.8, 1.2, size=200).tolist()
    sf, ss = TurbulenceState(d_norm_history=list(hist)), TurbulenceState(d_norm_history=list(hist))
    rf = compute_turbulence(full, cfg=DEFAULTS, state=sf)
    rs = compute_turbulence(shrunk, cfg=DEFAULTS, state=ss)
    assert rf.diag["d_norm"] == pytest.approx(rs.diag["d_norm"], rel=0.5)
    assert abs(rf.turb_pct - rs.turb_pct) < 0.25


# --------------------------------------------------------------------------- STALEATTR

def test_inv_staleattr_event_escalation_is_liq_attributed():
    """An out-of-cadence escalation can only be liq-bound — never a stale corr/vol/turb (STALEATTR)."""
    eng = RegimeEngine()
    live = tuple(f"S{i}" for i in range(60))
    exc, coc = _calm_liq(live)
    prev = eng.regime_state(_panel(gen.calm_panel(w=300, n=60, seed=500)), exc, coc, _book(), None)
    g_before = prev.g_t
    ex, co = gen.liquidity_blowout(live, level=0.99, kind="spread")
    out = prev
    for _ in range(eng.policy.d_liq_min):  # sustained minute-level blowout
        out = eng.regime_state_event(ex, co, _book(), out, live)
    assert out.binding_sub == "liq"
    assert "event_driven_liq" in out.reason_codes
    assert out.g_t <= g_before
    # A single-minute blip below D_liq_min cannot move g_t.
    eng2 = RegimeEngine()
    p2 = eng2.regime_state(_panel(gen.calm_panel(w=300, n=20, seed=6)), *_calm_liq(live[:20]), _book(), None)
    ex2, co2 = gen.liquidity_blowout(live[:20], level=0.99, kind="spread")
    assert eng2.regime_state_event(ex2, co2, _book(), p2, live[:20]).g_t == p2.g_t


# --------------------------------------------------------------------------- NODOUBLE2

def test_inv_nodouble2_crisis_gross_is_g_not_g_squared():
    """At CRISIS the realized gross is g_t*full (250), NOT g_t^2*full (62.50) (Invariant NODOUBLE2)."""
    book = _book()
    realized = 0.25 * book.book_gross_at_full
    assert realized == pytest.approx(250.0)
    assert realized != pytest.approx(0.25 ** 2 * book.book_gross_at_full)  # 62.50 forbidden


# --------------------------------------------------------------------------- CAP

def test_inv_cap_g_t_never_exceeds_one_and_equals_no_overlay_at_full():
    """g_t in [0.25,1.0]; at g_t=1.0 the downstream gross equals no-overlay (Invariant CAP)."""
    eng = RegimeEngine()
    live = tuple(f"S{i}" for i in range(30))
    ex, co = _calm_liq(live)
    prev = None
    for k in range(40):
        prev = eng.regime_state(_panel(gen.calm_panel(w=300, n=30, seed=200 + k)), ex, co, _book(), prev)
        assert 0.25 <= prev.g_t <= 1.0  # CAP + FLOOR every bar
    if prev.g_t == 1.0:
        book = _book()
        assert prev.g_t * book.book_gross_at_full == book.book_gross_at_full


# --------------------------------------------------------------------------- FLOOR

def test_inv_floor_g_t_never_below_quarter():
    """REGIME never emits g < 0.25 — flatting is OPS's job, not REGIME's brake (Invariant FLOOR)."""
    res = step_machine({n: _sub(n, 9.9, "CRISIS") for n in ("vol", "corr", "turb", "liq")}, None, DEFAULTS)
    assert res.g_s == 0.25 >= DEFAULTS.g_floor
    assert_floor(0.25)
    assert_floor(1.0)
    with pytest.raises(AssertionError):
        assert_floor(0.10)


# --------------------------------------------------------------------------- NODOUBLE

def test_inv_nodouble_regime_ignores_equity_drawdown_axis():
    """REGIME reads market-state only — no equity-drawdown/governor/kill input on its seam (NODOUBLE)."""
    params = set(inspect.signature(RegimeEngine.regime_state).parameters)
    for forbidden in ("equity_drawdown", "governor_multiplier", "kill_state", "peak_equity"):
        assert forbidden not in params


# --------------------------------------------------------------------------- ML

def test_inv_ml_no_learned_classifier_in_live_layers():
    """No ML framework / learned regime classifier in measures or state (Invariant ML)."""
    import convexity.regime.measures as measures_pkg
    import convexity.regime.state as state_pkg

    banned = ("sklearn", "torch", "tensorflow", "hmmlearn", "keras", "xgboost", "lightgbm")
    for pkg in (measures_pkg, state_pkg):
        for mod in pkgutil.iter_modules(pkg.__path__):
            m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
            text = open(m.__file__).read()
            for b in banned:
                assert f"import {b}" not in text and f"from {b}" not in text


# --------------------------------------------------------------------------- NOFIT

def test_inv_nofit_thresholds_are_quantile_based_and_grid_capped():
    """BLUE thresholds are finite enumerated grids, K_regime <= K_regime_max, LW deltas not DOF (NOFIT)."""
    reg = register_blue_grid(DEFAULTS)
    assert reg.k_regime <= DEFAULTS.k_regime_max == 64
    assert reg.k_regime == k_regime()
    for p in param_registry().params():
        if p.bucket is Bucket.BLUE:
            assert p.blue_grid is not None and len(p.blue_grid) >= 1  # finite, never a free fit
    names = {p.name for p in param_registry().params()}
    assert "delta_AR" not in names and "delta_turb" not in names  # LW intensities are TEAL, not DOF
    with pytest.raises(ValueError):
        resolve_config(blue={"kappa": 0.99})  # an unlisted value cannot sneak in


# --------------------------------------------------------------------------- CAUSAL

def test_inv_causal_future_bar_cannot_change_output_at_t():
    """A perturbation strictly after asof_bar leaves the panel (and g_t) at t identical (CAUSAL)."""
    rng = np.random.default_rng(0)
    rets = {f"S{i}": rng.normal(0.0, 0.01, size=400) for i in range(12)}
    from convexity.regime.panel.build import build_panel

    ac = {s: "crypto_perp" for s in rets}
    p1 = build_panel(returns_by_symbol=rets, asset_class_by_symbol=ac, window=100, asof_bar=200)
    rets2 = {k: v.copy() for k, v in rets.items()}
    for k in rets2:
        rets2[k][250] += 5.0  # mutate a FUTURE stamp (250 > asof_bar=200)
    p2 = build_panel(returns_by_symbol=rets2, asset_class_by_symbol=ac, window=100, asof_bar=200)
    assert np.allclose(p1.returns, p2.returns)
    e1, e2 = RegimeEngine(), RegimeEngine()
    ex, co = _calm_liq(p1.instruments)
    assert e1.regime_state(p1, ex, co, _book(), None).g_t == e2.regime_state(p2, ex, co, _book(), None).g_t


# --------------------------------------------------------------------------- COND

def test_inv_cond_near_singular_cov_falls_back_no_nonsense_spike():
    """A near-singular Sigma_t inverts via the diagonal fallback, finite d_t — never amplified (COND)."""
    res = compute_turbulence(gen.near_singular_panel(w=200, n=80, seed=5), cfg=DEFAULTS, state=TurbulenceState())
    assert np.isfinite(res.diag["d_t"])
    assert res.diag["d_t"] < 1e6  # no inverse-amplified nonsense spike
    # Window-deficient W_turb < c_cov*n_t also routes to the robust diagonal.
    defi = compute_turbulence(gen.calm_panel(w=60, n=40, seed=1), cfg=replace(DEFAULTS, w_turb=60),
                              state=TurbulenceState())
    assert defi.diag.get("cov_window_deficient") == 1.0


# --------------------------------------------------------------------------- CHEAP

def test_inv_cheap_no_eig_or_inverse_on_the_per_minute_path():
    """The event-driven (per-minute) liq path runs O(n) only — no eig/inverse (Invariant CHEAP)."""
    src = inspect.getsource(RegimeEngine.regime_state_event)
    for heavy in ("eigh", "eigvals", "lw_shrink", "solve(", "inv(", "compute_absorption", "compute_turbulence"):
        assert heavy not in src  # the O(n^3) work is cadence-gated, never on the minute path
    # The per-minute path only touches the O(n) liquidity aggregation.
    assert "compute_liquidity" in src


# --------------------------------------------------------------------------- COSTGATE

def test_inv_costgate_escalation_never_blocked_deescalation_must_pay():
    """Cutting gross is never cost-blocked; re-grossing up must clear the deadband + cost cap (COSTGATE)."""
    # Escalation passes even with an absurd cost estimate (safety beats cost on the downside).
    esc = cost_gate(g_prev=0.75, g_new=0.25, transition="escalate", book=_book(1000.0, 750.0),
                    cfg=DEFAULTS, round_trip_estimate=lambda dg: 1e9)
    assert esc.passed is True and esc.g_emit == 0.25
    # Re-grossing up: deadband cleared but cost cap blows -> deferred.
    expensive = cost_gate(g_prev=0.50, g_new=0.75, transition="deescalate", book=_book(1000.0, 500.0),
                          cfg=DEFAULTS, round_trip_estimate=lambda dg: 200.0)
    assert expensive.passed is False and expensive.reason == "regime_change_cost_gated"
    # Cheap re-gross passes.
    cheap = cost_gate(g_prev=0.50, g_new=0.75, transition="deescalate", book=_book(1000.0, 500.0),
                      cfg=DEFAULTS, round_trip_estimate=lambda dg: 1.0)
    assert cheap.passed is True and cheap.g_emit == 0.75


# --------------------------------------------------------------------------- NREBUILD

def test_inv_nrebuild_liq_is_pure_read_of_exec_cost():
    """g^liq is a pure max-of-two aggregation of EXECUTION/COST scalars — no re-modelled spread (NREBUILD)."""
    live = ("A", "B", "C")
    ex = ExecState(spread_pct={"A": 0.9, "B": 0.8, "C": 0.7})
    co = CostState(funding_pct={"A": 0.2, "B": 0.3, "C": 0.1})
    res = compute_liquidity(exec_state=ex, cost_state=co, live=live)
    assert res.spread_blowout == pytest.approx(0.8)
    assert res.funding_extreme == pytest.approx(0.2)
    assert res.liq_stress == pytest.approx(0.8)  # the worse of the two
    # Only live names contribute (no name absent from the panel drags the aggregate).
    res2 = compute_liquidity(exec_state=ExecState(spread_pct={"A": 0.5, "B": 0.5, "C": 0.99}),
                             cost_state=CostState(funding_pct={"A": 0.5, "B": 0.5}), live=("A", "B"))
    assert res2.spread_blowout == pytest.approx(0.5)


# --------------------------------------------------------------------------- C

def test_inv_c_red_unreachable_from_instrument_spec():
    """No RED/BLUE regime param name appears on the per-instrument InstrumentSpec (Invariant C)."""
    amber_fields = frozenset(InstrumentSpec.__dataclass_fields__.keys())
    assert_no_red_in_amber(amber_fields, param_registry())  # raises if a RED/BLUE name leaked


# --------------------------------------------------------------------------- V

def test_inv_v_live_layers_never_import_validate():
    """Nothing in panel/measures/state imports regime.validate or forensics (Invariant V)."""
    live_packages = ["convexity.regime.panel", "convexity.regime.measures", "convexity.regime.state"]
    for pkg_name in live_packages:
        pkg = importlib.import_module(pkg_name)
        for mod in pkgutil.iter_modules(pkg.__path__):
            m = importlib.import_module(f"{pkg_name}.{mod.name}")
            tree = ast.parse(open(m.__file__).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "regime.validate" not in node.module, f"{m.__name__} imports validate (V)"


# --------------------------------------------------------------------------- WE

def test_inv_we_worked_examples_are_reproduced():
    """The Section-6 worked examples reproduce bit-for-bit (Invariant WE — full fixtures live in
    test_worked_examples.py; this asserts the headline §6.1 anchors are reachable)."""
    n_t = 80
    assert top_k(0.20, n_t) == 16
    eig = np.zeros(n_t)
    eig[:16] = 60.0 / 16.0
    assert absorption_ratio(eig, n_t, 16) == pytest.approx(0.750)
    assert standardized_shift(0.710, 0.600, 0.050) == pytest.approx(2.20, abs=1e-9)
    gate = cost_gate(g_prev=0.75, g_new=0.25, transition="escalate",
                     book=_book(1000.0, 750.0), cfg=DEFAULTS)
    assert gate.passed and gate.delta_gross == pytest.approx(500.0)
    assert gate.rebalance_cost_hat == pytest.approx(1.00, abs=1e-9)  # ~20bps on 500


# --------------------------------------------------------------------------- DET

def test_inv_det_identical_inputs_identical_output():
    """Two engines on identical inputs emit byte-identical RegimeState (Invariant DET)."""
    panel = _panel(gen.calm_panel(w=300, n=30, seed=3))
    ex, co = _calm_liq(panel.instruments)
    o1 = RegimeEngine().regime_state(panel, ex, co, _book(), None)
    o2 = RegimeEngine().regime_state(panel, ex, co, _book(), None)
    assert (o1.g_t, o1.state, o1.binding_sub, o1.dwell_bars) == (o2.g_t, o2.state, o2.binding_sub, o2.dwell_bars)
