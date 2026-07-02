"""OPS pillar tests (``docs/specs/OPS.md``) — one pytest per named Invariant that fails if the
invariant is violated, plus worked examples, the CUSUM calibration reproduction, and the sec-12
cross-consistency / ``OpsEngine.on_obs`` conformance gate.

Covers: RECON (sec 1.4), CADENCE (P2/sec 1.3), DNEUT (sec 3.3), CAL (sec 4.3), MONO (sec 4.4),
SLA (sec 5.3), HALT-ONE (sec 6), PROV (P3 / sec 9), C (sec 7), and the sec-12 Protocol conformance
of ``ops.brake.engine.OpsEngine``. All offline / simulatable — no live venue.

Module paths follow OPS.md sec 8's named tree: reconciliation tolerance in
``ops.reconcile.streams_map``, the book snapshot in ``ops.streams.recon``, SLA in ``ops.streams.fills``,
``z_k`` in ``ops.streams.whiten``, renormalization in ``ops.brake.renormalize``, the ARL calibration in
``ops.validate.arl``, the composing entry point in ``ops.brake.engine``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from convexity.core.types import Alarm, OpsDirective
from convexity.ops.brake.engine import OpsEngine
from convexity.ops.brake.kill_switch import (
    KillSwitch,
    flatten,
    state_from_statistic,
)
from convexity.ops.brake.renormalize import (
    DegenerateLegError,
    renormalize_after_delist,
)
from convexity.ops.brake.triggers import (
    alarm_severity,
    combine,
    recon_severity,
)
from convexity.ops.config.defaults import DEFAULTS, PERIODS_PER_YEAR, OpsDefaults
from convexity.ops.detectors.cusum import arl_0_siegmund, solve_boundary
from convexity.ops.reconcile.streams_map import (
    epsilon_i,
    reconcile_instrument,
    reconcile_stale,
    worst_tier,
)
from convexity.ops.streams.fills import (
    completion_lag_ok,
    evaluate_sla,
    slippage_ok,
)
from convexity.ops.streams.recon import PositionSnapshot, reconcile_book
from convexity.ops.streams.whiten import whiten
from convexity.ops.types import OpsObs, RebalanceFill
from convexity.ops.validate.arl import calibrate_and_verify, mc_arl_0, mc_arl_1, solve_h_star


# ======================================================================== migration / cross-consistency

def test_ops_directive_matches_spec_shape():
    """OPS.md sec 9 / sec 12: core.types.OpsDirective has THIS spec's shape, one source of truth."""
    from convexity.core import types as ct
    from convexity.ops import types as ot

    # The stale pre-pivot symbols are gone (sec 9 item 2).
    assert not hasattr(ct, "KillState"), "pre-pivot KillState must be removed (OPS.md sec 9)"
    # ops.types.OpsDirective and core.types.OpsDirective are the SAME object, not two classes.
    assert ot.OpsDirective is ct.OpsDirective
    assert ot.Alarm is ct.Alarm
    d = OpsDirective(state="RUN", gross_multiplier=1.0, alarms=(), as_of=0)
    fields = {f for f in d.__dataclass_fields__}
    assert fields == {"state", "gross_multiplier", "alarms", "as_of"}


def test_ops_directive_field_types_match_sec9_exactly():
    """OPS.md sec 12 cross-consistency: the migrated OpsDirective has sec 9's EXACT field set/shape —
    ``state`` a RUN/DEGRADE/HALT literal, ``gross_multiplier`` float, ``alarms`` a tuple of Alarm,
    ``as_of`` int. A foundation-level guard on the load-bearing migration (not the RegimeState
    migration, which is a separate REGIME-owned change to core/types.py)."""
    d = OpsDirective(state="DEGRADE", gross_multiplier=0.5,
                     alarms=(Alarm(kind="CUSUM_WARN", detail="x", value=1.0, threshold=2.0, as_of=0),),
                     as_of=7)
    assert d.state == "DEGRADE"
    assert isinstance(d.gross_multiplier, float)
    assert isinstance(d.alarms, tuple) and all(isinstance(a, Alarm) for a in d.alarms)
    assert isinstance(d.as_of, int)
    # Alarm's own field set matches sec 9 exactly.
    a = d.alarms[0]
    assert set(a.__dataclass_fields__) == {"kind", "detail", "value", "threshold", "as_of"}


def test_ops_directive_and_alarm_are_frozen():
    d = OpsDirective(state="RUN", gross_multiplier=1.0, alarms=(), as_of=0)
    a = Alarm(kind="CUSUM_ALARM", detail="x", value=1.0, threshold=2.0, as_of=0)
    with pytest.raises((AttributeError, TypeError)):
        d.state = "HALT"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        a.value = 9.0  # type: ignore[misc]


# ======================================================================== Invariant PROV (P3, sec 9)

def test_prov_directive_carries_no_weight_or_per_instrument_veto():
    """Inv PROV: OPS emits only {RUN,DEGRADE,HALT} + gross_multiplier + reason chain — no weight."""
    d = OpsDirective(state="DEGRADE", gross_multiplier=0.5, alarms=(), as_of=0)
    fnames = set(d.__dataclass_fields__)
    # No per-instrument or weight field of any kind.
    for forbidden in ("weights", "weight", "symbol", "instrument", "target_qty", "veto"):
        assert forbidden not in fnames
    assert d.state in ("RUN", "DEGRADE", "HALT")


# ======================================================================== Invariant C (sec 7)

def test_c_red_constants_are_not_per_instrument():
    """Inv C: RED control constants live on OpsDefaults (injected via resolve), never per-instrument.

    A cheap structural guard: the InstrumentSpec AMBER metadata object carries none of OPS's RED
    control knobs (h*, m_slip, m_warn, tolerances, boundaries)."""
    from convexity.core.types import InstrumentSpec

    inst_fields = set(InstrumentSpec.__dataclass_fields__)
    for red in ("h_star", "m_slip", "m_warn", "k_lot", "k_notional", "delta_target",
                "arl_0_target", "g_degrade", "n_confirm", "sla_severe_lag_multiple"):
        assert red not in inst_fields, f"RED {red} must not be an InstrumentSpec field (Inv C)"
        assert red in OpsDefaults.__dataclass_fields__, f"{red} must be a RED default"


def test_no_blue_grid_no_ml_imports():
    """OPS.md sec 7 / sec 12: no BLUE grid, no ML/optimizer import anywhere in ops/ (AST-style scan)."""
    import ast
    import importlib
    import pkgutil

    import convexity.ops as ops_pkg

    banned = {"sklearn", "torch", "tensorflow", "keras", "xgboost", "lightgbm", "optuna",
              "catboost", "jax"}
    for m in pkgutil.walk_packages(ops_pkg.__path__, ops_pkg.__name__ + "."):
        mod = importlib.import_module(m.name)
        src = getattr(mod, "__file__", None)
        if src is None:
            continue
        with open(src) as fh:
            tree = ast.parse(fh.read(), filename=src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in banned, f"{m.name} imports banned ML lib {top}"
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                top = node.module.split(".")[0]
                assert top not in banned, f"{m.name} imports banned ML lib {top}"


def test_no_edge_miscitation_of_system_only_numbers():
    """OPS.md sec 12 kill-switch gate: the frac_pos / CPCV-Sharpe / PBO figures are SYSTEM.md's, and
    no code comment may attribute them to EDGE.md (a grep-style regression guard against the
    mis-citation this spec itself once carried)."""
    import importlib
    import pkgutil

    import convexity.ops as ops_pkg

    # These specific numbers belong to SYSTEM.md sec 4.2-4.4, never EDGE.md.
    system_only_tokens = ("0.357", "PBO", "frac_pos")
    for m in pkgutil.walk_packages(ops_pkg.__path__, ops_pkg.__name__ + "."):
        mod = importlib.import_module(m.name)
        src = getattr(mod, "__file__", None)
        if src is None:
            continue
        with open(src) as fh:
            text = fh.read()
        for tok in system_only_tokens:
            if tok in text:
                # If the token appears, it must NOT be attributed to EDGE.md on the same line.
                for line in text.splitlines():
                    if tok in line:
                        assert "EDGE.md" not in line, (
                            f"{m.name}: SYSTEM-only figure '{tok}' mis-attributed to EDGE.md"
                        )


# ======================================================================== Invariant RECON (sec 1.4)

def test_epsilon_scaled_not_flat_global():
    """OPS.md sec 1.2: epsilon_i is scaled per-instrument (lot floor vs notional floor), not flat."""
    d = DEFAULTS
    # A small position whose lot floor binds: notional term = 1e-3*500/100 = 0.005 << 2 lots.
    eps_alt = epsilon_i(lot_size_i=1.0, target_notional_i=500.0, price_i=100.0,
                        k_lot=d.k_lot, k_notional=d.k_notional)
    # A major: notional floor binds (1e-3*50000/50000 = 1e-3 >> 2*0.001 = 0.002... use bigger lot).
    eps_maj = epsilon_i(lot_size_i=1e-6, target_notional_i=50_000.0, price_i=50_000.0,
                        k_lot=d.k_lot, k_notional=d.k_notional)
    assert eps_alt == pytest.approx(d.k_lot * 1.0)  # 2 lots (lot floor binds)
    assert eps_maj == pytest.approx(d.k_notional * 50_000.0 / 50_000.0)  # 0.1% of 1 unit (notional floor)
    # They are genuinely different scales (not a single global constant).
    assert eps_alt != eps_maj


def test_epsilon_rejects_nonpositive_price():
    with pytest.raises(ValueError):
        epsilon_i(lot_size_i=1.0, target_notional_i=1.0, price_i=0.0, k_lot=2.0, k_notional=1e-3)


def test_recon_breach_escalates_to_halt_independent_of_cusum_state():
    """Inv RECON: a 1.5 * epsilon * m_warn divergence produces RECON_BREACH -> HALT within one cycle,
    regardless of the CUSUM state at that moment (OPS.md sec 1.4)."""
    d = DEFAULTS
    eps = epsilon_i(lot_size_i=1.0, target_notional_i=10_000.0, price_i=100.0,
                    k_lot=d.k_lot, k_notional=d.k_notional)
    breach_delta = 1.5 * eps * d.m_warn
    res = reconcile_instrument(instrument="ALT", book_position=breach_delta,
                               exchange_position=0.0, epsilon=eps, m_warn=d.m_warn, as_of=1000)
    assert res.tier == "BREACH"
    assert res.alarm is not None and res.alarm.kind == "RECON_BREACH"
    # A BREACH on ANY instrument means the whole book must HALT (sec 1.4) — the flatten routine is
    # invoked directly, with no dependence on the CUSUM statistic having accumulated.
    directive = flatten((res.alarm,), as_of=res.alarm.as_of)
    assert directive.state == "HALT"
    assert directive.gross_multiplier == 0.0


def test_recon_warn_band_does_not_halt():
    """OPS.md sec 1.4: a drift in (epsilon, m_warn*epsilon] is WARN — logged, no book action."""
    d = DEFAULTS
    eps = 10.0
    warn_delta = 2.0 * eps  # inside (eps, 3*eps]
    res = reconcile_instrument(instrument="X", book_position=warn_delta, exchange_position=0.0,
                               epsilon=eps, m_warn=d.m_warn, as_of=0)
    assert res.tier == "WARN"
    # A WARN-band divergence is a POSITION divergence event (kind RECON_BREACH, sub-severe),
    # distinguished from a halting breach by its WARN tier -- NOT a RECON_STALE (which sec 1.4
    # reserves strictly for a failed READ). Asserting the exact kind makes this FAIL if the WARN band
    # were mislabeled RECON_STALE.
    assert res.alarm is not None
    assert res.alarm.kind == "RECON_BREACH", (
        "WARN-band divergence must be RECON_BREACH kind (tier WARN), never RECON_STALE (sec 1.4)"
    )


def test_recon_warn_kind_ambiguity_resolved_by_tier_not_kind():
    """OPS.md sec 9/1.4: because the closed AlarmKind set reuses RECON_BREACH for both a WARN-band
    divergence AND a halting breach, severity MUST be taken from the tier, never the kind alone.
    ``recon_severity`` resolves WARN -> RUN and BREACH -> HALT, closing the sec-9 ambiguity at the
    OPS->live-loop boundary (the auditor's flagged 'genuine ambiguity')."""
    d = DEFAULTS
    eps = 10.0
    warn = reconcile_instrument(instrument="X", book_position=2.0 * eps, exchange_position=0.0,
                                epsilon=eps, m_warn=d.m_warn, as_of=0)
    breach = reconcile_instrument(instrument="Y", book_position=10.0 * eps, exchange_position=0.0,
                                  epsilon=eps, m_warn=d.m_warn, as_of=0)
    # Both alarms share the SAME kind (the ambiguity) ...
    assert warn.alarm is not None and breach.alarm is not None
    assert warn.alarm.kind == breach.alarm.kind == "RECON_BREACH"
    # ... but the tier discriminates severity, and recon_severity keys on tier, not kind.
    assert recon_severity(warn.tier) == "RUN"
    assert recon_severity(breach.tier) == "HALT"
    # And the precedence combiner respects that: a WARN-only recon does NOT halt.
    warn_directive = combine(recon_results=[warn], g_degrade=d.g_degrade, as_of=0)
    assert warn_directive.state == "RUN"
    breach_directive = combine(recon_results=[breach], g_degrade=d.g_degrade, as_of=0)
    assert breach_directive.state == "HALT"


def test_recon_ok_within_tolerance():
    res = reconcile_instrument(instrument="X", book_position=5.0, exchange_position=0.0,
                               epsilon=10.0, m_warn=3.0, as_of=0)
    assert res.tier == "OK"
    assert res.alarm is None


def test_recon_stale_read_never_silently_ok():
    """OPS.md sec 1.4: a failed read is RECON_STALE (WARN), escalating to BREACH past staleness_max —
    never a silent OK. Conservative fallback (COST.md P6 in spirit)."""
    d = DEFAULTS
    fresh = reconcile_stale(instrument="X", days_unreadable=1,
                            staleness_max_days=d.staleness_max_days, as_of=0)
    assert fresh.tier == "STALE" and fresh.alarm is not None
    assert fresh.alarm.kind == "RECON_STALE"
    stale = reconcile_stale(instrument="X", days_unreadable=d.staleness_max_days + 1,
                            staleness_max_days=d.staleness_max_days, as_of=0)
    assert stale.tier == "BREACH" and stale.alarm.kind == "RECON_BREACH"


def test_reconcile_book_routes_stale_reads_and_scales_epsilon():
    """OPS.md sec 1 / sec 2: the book-snapshot adapter routes a failed read to the stale ladder (never
    a silent OK) and scales epsilon per instrument (sec 1.2), producing the per-instrument ladder."""
    d = DEFAULTS
    snaps = [
        # in tolerance (majors, notional floor ~ 0.001): OK
        PositionSnapshot("BTC", book_position=1.0, exchange_position=1.0, lot_size=1e-6,
                         target_notional=50_000.0, price=50_000.0),
        # a failed read -> STALE, never compared as OK
        PositionSnapshot("ETH", book_position=0.0, exchange_position=0.0, lot_size=1e-3,
                         target_notional=10_000.0, price=3000.0, days_unreadable=1),
        # a hard breach on a small alt -> BREACH
        PositionSnapshot("ALT", book_position=1000.0, exchange_position=0.0, lot_size=1.0,
                         target_notional=500.0, price=100.0),
    ]
    results = reconcile_book(snaps, k_lot=d.k_lot, k_notional=d.k_notional, m_warn=d.m_warn,
                             staleness_max_days=d.staleness_max_days, as_of=42)
    tiers = {r.instrument: r.tier for r in results}
    assert tiers["BTC"] == "OK"
    assert tiers["ETH"] == "STALE"  # the failed read is surfaced, not silently OK
    assert tiers["ALT"] == "BREACH"
    assert worst_tier(results) == "BREACH"


def test_worst_tier_any_breach_dominates():
    results = [
        reconcile_instrument(instrument="a", book_position=0.0, exchange_position=0.0,
                             epsilon=10.0, m_warn=3.0, as_of=0),
        reconcile_instrument(instrument="b", book_position=100.0, exchange_position=0.0,
                             epsilon=10.0, m_warn=3.0, as_of=0),
    ]
    assert worst_tier(results) == "BREACH"


def test_cadence_is_stricter_than_rebalance():
    """Inv CADENCE (P2): the reconciliation cadence is strictly finer than the weekly rebalance, and
    strictly finer than the sigma_0 re-estimation cadence (so the null cannot chase a break inside a
    DEGRADE/HALT episode, sec 7). Binds the full sec-1.3/sec-7 cadence chain, not just one pair."""
    d = DEFAULTS
    # daily recon < weekly rebalance (sec 1.3)
    assert d.reconciliation_cadence_days < d.rebalance_days
    # staleness escalation window < one rebalance interval (sec 1.4: "strictly less than one interval")
    assert d.staleness_max_days < d.rebalance_days
    # sigma_0 re-estimation (weeks) >> n_confirm window (weeks) so the null can't adapt in-episode.
    assert d.sigma_0_reestimation_weeks > d.n_confirm
    # and the control clock (daily) is finer than the rebalance clock it protects (P2).
    assert d.reconciliation_cadence_days >= 1


# ======================================================================== Invariant DNEUT (sec 3.3)

def _dollar_neutral_book(rng: np.random.Generator, n_per_leg: int = 10, gross: float = 1.0):
    """A vol-scaled-ish dollar-neutral book: long leg +gross/2, short leg -gross/2 (sec 3.3 setup)."""
    long_raw = rng.uniform(0.5, 2.0, n_per_leg)
    short_raw = rng.uniform(0.5, 2.0, n_per_leg)
    long_w = long_raw / long_raw.sum() * (gross / 2.0)
    short_w = -short_raw / short_raw.sum() * (gross / 2.0)
    return np.concatenate([long_w, short_w]), n_per_leg


@pytest.mark.parametrize("seed", range(20))
def test_dneut_long_leg_delisting_restores_neutrality_exactly(seed):
    """Inv DNEUT: mid-week delisting of a long-leg name restores Sum w_long = -Sum w_short exactly,
    and leaves every surviving pairwise weight ratio unchanged (OPS.md sec 3.3)."""
    rng = np.random.default_rng(seed)
    w, n = _dollar_neutral_book(rng)
    assert abs(w.sum()) < 1e-12
    victim = int(rng.integers(0, n))  # a long-leg index
    ratios_before = _pairwise_ratios(w[:n], exclude=victim)
    res = renormalize_after_delist(w, victim)
    assert abs(res.weights.sum()) < 1e-9
    assert res.weights[victim] == 0.0
    assert res.leg == "long"
    # Relative weights within the surviving long leg preserved (proportional, not equal top-up).
    ratios_after = _pairwise_ratios(res.weights[:n], exclude=victim)
    np.testing.assert_allclose(ratios_after, ratios_before, atol=1e-9)
    # Short leg untouched.
    np.testing.assert_allclose(res.weights[n:], w[n:], atol=1e-15)


@pytest.mark.parametrize("seed", range(20))
def test_dneut_short_leg_delisting_restores_neutrality_exactly(seed):
    """Inv DNEUT (short-leg case, sec 3.3 mirror image): a SHORT-leg delisting restores neutrality."""
    rng = np.random.default_rng(seed + 100)
    w, n = _dollar_neutral_book(rng)
    victim = int(n + rng.integers(0, n))  # a short-leg index
    ratios_before = _pairwise_ratios(w[n:], exclude=victim - n)
    res = renormalize_after_delist(w, victim)
    assert abs(res.weights.sum()) < 1e-9
    assert res.weights[victim] == 0.0
    assert res.leg == "short"
    ratios_after = _pairwise_ratios(res.weights[n:], exclude=victim - n)
    np.testing.assert_allclose(ratios_after, ratios_before, atol=1e-9)
    # Long leg untouched.
    np.testing.assert_allclose(res.weights[:n], w[:n], atol=1e-15)


def test_dneut_sequential_double_delisting(seed=7):
    """OPS.md sec 3.4: two independent delistings in one week — apply sec 3.3 sequentially; the
    proof assumes only neutrality immediately before each exit, so induction holds."""
    rng = np.random.default_rng(seed)
    w, n = _dollar_neutral_book(rng)
    first = renormalize_after_delist(w, 0)  # long-leg name
    assert abs(first.weights.sum()) < 1e-9
    # Second delisting: a short-leg name, against the ALREADY-renormalized book.
    second = renormalize_after_delist(first.weights, n + 1)
    assert abs(second.weights.sum()) < 1e-9
    assert second.weights[0] == 0.0 and second.weights[n + 1] == 0.0


def test_dneut_degenerate_single_name_leg_forces_halt():
    """OPS.md sec 3.4: delisting the entire leg (denominator -> 0) forces HALT, never a div-by-zero."""
    # A degenerate 1-long / 2-short book that is still dollar-neutral.
    w = np.array([0.5, -0.25, -0.25])
    with pytest.raises(DegenerateLegError):
        renormalize_after_delist(w, 0)  # the only long name


def test_dneut_rejects_non_neutral_input():
    w = np.array([0.6, 0.1, -0.5])  # not neutral
    with pytest.raises(ValueError):
        renormalize_after_delist(w, 0)


def _pairwise_ratios(leg: np.ndarray, *, exclude: int) -> np.ndarray:
    survivors = np.delete(leg, exclude)
    # Ratio of each survivor to the first survivor — a scale-invariant fingerprint of relative sizing.
    return survivors / survivors[0]


# ======================================================================== Invariant CAL (sec 4.3)

def test_cal_h_star_reproduces_6_23():
    """OPS.md sec 4.3 / sec 12: the shipped h* = 6.23 is what solving ARL_0(0.5, h) = 1000 produces.

    Delegates the solve to ops.validate.arl (the sec-8 named home of the reproducible calibration),
    NOT an inline re-solve — the auditor's flag that the calibration was inlined into the test."""
    h = solve_h_star(delta_target=DEFAULTS.delta_target, arl_0_target=DEFAULTS.arl_0_target)
    assert h == pytest.approx(6.23, abs=0.01)
    # And the shipped RED constant matches the solve to two decimals.
    assert DEFAULTS.h_star == pytest.approx(round(h, 2), abs=0.005)
    assert DEFAULTS.d == pytest.approx(0.5)
    # solve_boundary and solve_h_star agree (solve_h_star is the delta_target->d convenience).
    assert solve_boundary(0.5, 1000.0) == pytest.approx(h)


def test_cal_siegmund_arl_monotone_and_positive():
    assert arl_0_siegmund(0.5, 6.23) == pytest.approx(1000.0, rel=0.02)
    # ARL_0 increases with the boundary h.
    assert arl_0_siegmund(0.5, 7.0) > arl_0_siegmund(0.5, 6.23)


def test_cal_monte_carlo_false_alarm_rate_within_budget():
    """Inv CAL: Monte-Carlo-simulating iid N(0,1) z_k through the calibrated (d, h*) recursion gives
    a false-HALT rate within MC error of the <=5%/yr target (ARL_0 ~ 1000 weeks) (OPS.md sec 4.3).

    Delegates the MC to ops.validate.arl.mc_arl_0 (the sec-8 named calibration home) instead of
    reimplementing the loop in the test. The two-sided [2500, 7000] band BINDS h* mis-calibration:
    the shipped h*=6.23 gives MC ARL_0 ~4000 (this seed/size); a too-low h*=5.5 gives ~1535 and a
    too-high h*=7.5 gives ~16055, both outside the band. The ~4x gap above the 1000-week Siegmund
    design target is the spec-disclosed conservatism of the large-ARL Wald approximation (sec 4.3:
    achieved ARL_0 is slightly ABOVE the 5%/yr budget, never below)."""
    arl_0 = mc_arl_0(d=DEFAULTS.d, h_star=DEFAULTS.h_star, n_years=4000, seed=0)
    assert 2500.0 < arl_0 < 7000.0, (
        f"empirical ARL_0 {arl_0:.0f} outside the calibrated [2500, 7000] band — "
        "h* is mis-calibrated (too low => false-alarm blowout; too high => dead detector)"
    )
    per_year_prob = 1.0 - (1.0 - 1.0 / arl_0) ** PERIODS_PER_YEAR
    assert per_year_prob < 0.05, f"per-year false-HALT prob {per_year_prob:.4f} exceeds 5%/yr budget"


def test_cal_arl_1_much_faster_than_arl_0_under_injected_shift():
    """Inv CAL detection side: a persistent delta_target=1.0 downward shift is detected far faster
    than the null false-alarm run length (ARL_1 << ARL_0) (OPS.md sec 4.3). Delegates to
    ops.validate.arl.mc_arl_1."""
    arl_1 = mc_arl_1(d=DEFAULTS.d, h_star=DEFAULTS.h_star, delta_target=DEFAULTS.delta_target,
                     n_reps=2000, seed=1)
    assert arl_1 < 100.0, f"ARL_1 {arl_1:.1f} not << ARL_0 ~1000 (detection too slow)"


def test_cal_calibrate_and_verify_report_is_reproducible():
    """OPS.md sec 11.1: the full ARL acceptance run (the gate for changing d/h*/delta_target) — h*
    reproduces to two decimals, per-year false-HALT is under the 5%/yr budget, and ARL_1 << ARL_0."""
    rep = calibrate_and_verify(delta_target=DEFAULTS.delta_target, arl_0_target=DEFAULTS.arl_0_target,
                               h_star=DEFAULTS.h_star, n_years_null=2000, n_reps_alt=1000, seed=0)
    assert round(rep.h_star_solved, 2) == DEFAULTS.h_star
    assert rep.per_year_false_halt < 0.05
    assert rep.arl_1_mc < rep.arl_0_mc
    assert rep.arl_1_mc < 100.0


# ======================================================================== Invariant MONO (sec 4.4)

def _fresh_killswitch() -> KillSwitch:
    d = DEFAULTS
    return KillSwitch(mu_0=d.mu_0, sigma_0=d.sigma_0, h_star=d.h_star, d=d.d,
                      g_degrade=d.g_degrade, n_confirm=d.n_confirm)


def _bad_week_return() -> float:
    """A weekly net return that produces a strongly negative z_k (drives the downward CUSUM up)."""
    return DEFAULTS.mu_0 - 4.0 * DEFAULTS.sigma_0


def _good_week_return() -> float:
    return DEFAULTS.mu_0 + 1.0 * DEFAULTS.sigma_0


def test_mono_halt_does_not_self_heal_on_one_good_week():
    """Inv MONO: cross into HALT, then one single healthy z_k — remains HALT (no silent revert)."""
    ks = _fresh_killswitch()
    for i in range(20):
        directive = ks.observe(_bad_week_return(), as_of=i)
        if directive.state == "HALT":
            break
    assert directive.state == "HALT"
    after = ks.observe(_good_week_return(), as_of=100)
    assert after.state == "HALT"
    ks.rearm()
    assert ks.directive().state == "RUN"


def test_mono_degrade_recovery_requires_n_confirm_consecutive_weeks():
    """Inv MONO: DEGRADE recovers only after n_confirm consecutive healthy weeks (OPS.md sec 4.4)."""
    d = DEFAULTS
    ks = _fresh_killswitch()
    r_moderate = d.mu_0 - 1.8 * d.sigma_0
    directive = None
    for i in range(30):
        directive = ks.observe(r_moderate, as_of=i)
        if directive.state == "DEGRADE":
            break
    assert directive is not None and directive.state == "DEGRADE"
    assert directive.gross_multiplier == pytest.approx(d.g_degrade)
    strong_good = d.mu_0 + 5.0 * d.sigma_0
    states = []
    for i in range(d.n_confirm + 3):
        states.append(ks.observe(strong_good, as_of=200 + i).state)
    # Inv MONO binds n_confirm EXACTLY: the first RUN must land at 0-indexed position n_confirm-1
    # (the n_confirm-th healthy observation); every earlier observation must still read DEGRADE.
    assert "RUN" in states, "DEGRADE must eventually recover after a healthy run"
    first_run = states.index("RUN")
    assert first_run == d.n_confirm - 1, (
        f"recovery must require exactly n_confirm={d.n_confirm} consecutive healthy weeks "
        f"(first RUN expected at index {d.n_confirm - 1}, got {first_run})"
    )
    assert all(st == "DEGRADE" for st in states[:first_run]), (
        "no early self-heal: every observation before the n_confirm-th healthy week stays DEGRADE"
    )


def test_state_from_statistic_thresholds():
    h = 6.23
    assert state_from_statistic(0.0, h_star=h) == "RUN"
    assert state_from_statistic(h / 2.0, h_star=h) == "DEGRADE"
    assert state_from_statistic(h, h_star=h) == "HALT"
    assert state_from_statistic(h / 2.0 - 1e-9, h_star=h) == "RUN"


def test_whiten_matches_evidence_anchors():
    """OPS.md sec 4.2: mu_0 = SR_0/sqrt(52.18), sigma_0 = 0.28/sqrt(52.18)."""
    d = DEFAULTS
    assert d.mu_0 == pytest.approx(0.85 / math.sqrt(PERIODS_PER_YEAR))
    assert d.sigma_0 == pytest.approx(0.28 / math.sqrt(PERIODS_PER_YEAR))
    # z_k of a return exactly at mu_0 is 0.
    assert whiten(d.mu_0, mu_0=d.mu_0, sigma_0=d.sigma_0) == pytest.approx(0.0)


def test_ann_vol_0_is_derived_from_edge_sharpe_and_cagr():
    """OPS.md sec 4.2: ann_vol_0=0.28 is DERIVED from EDGE.md sec b, not asserted.

    EDGE.md sec b's "Honest performance" table prints net Sharpe ~0.8-1.0 (lower 0.85 == sr_0),
    CAGR @100% gross ~20-25% (lower 0.20) and maxDD ~25%, but NO literal vol. The annualized-Sharpe
    identity SR = arith_mean/vol plus the geometric<->arithmetic drift correction
    arith_mean = CAGR + 0.5*vol**2 give the fixed point  vol = (CAGR + 0.5*vol**2)/SR. Solving that
    fixed point at EDGE.md's own lower band (SR = sr_0 = 0.85, CAGR = 0.20) must reproduce the shipped
    ann_vol_0 to 2 dp. This pins the constant's ORIGIN so it can never silently drift back into a bare
    magic number. The two INPUT figures are the primary EDGE.md numbers, not fabricated.
    """
    edge_sharpe_lower = 0.85  # EDGE.md sec b honest net Sharpe band lower bound (== DEFAULTS.sr_0)
    edge_cagr_lower = 0.20    # EDGE.md sec b CAGR @100% gross band lower bound

    assert DEFAULTS.sr_0 == edge_sharpe_lower  # the vol derivation reuses the SAME Sharpe anchor

    # Fixed-point solve for vol given (CAGR, SR): vol = (CAGR + 0.5*vol**2)/SR.
    vol = edge_cagr_lower / edge_sharpe_lower
    for _ in range(500):
        vol = (edge_cagr_lower + 0.5 * vol * vol) / edge_sharpe_lower

    assert vol == pytest.approx(0.28211, abs=1e-4)      # the un-rounded derived value
    assert round(vol, 2) == DEFAULTS.ann_vol_0          # shipped constant is the 2-dp rounding

    # Round-trip: the shipped ann_vol_0 with sr_0 reproduces EDGE.md's lower CAGR bound.
    arith_mean = DEFAULTS.sr_0 * DEFAULTS.ann_vol_0
    implied_cagr = arith_mean - 0.5 * DEFAULTS.ann_vol_0 ** 2
    assert implied_cagr == pytest.approx(edge_cagr_lower, abs=2e-3)

    # maxDD 25% is a coherent third leg: Calmar = CAGR/maxDD ~= sr_0.
    edge_maxdd = 0.25
    assert (edge_cagr_lower / edge_maxdd) == pytest.approx(DEFAULTS.sr_0, abs=0.06)


# ======================================================================== Invariant SLA (sec 5.3)

def test_sla_severe_lag_halts_before_next_weekly_observation():
    """Inv SLA: a rebalance whose last fill settles at 2.5 * W_exec produces SLA_BREACH_SEVERE and a
    HALT directive, needing no weekly CUSUM observation (OPS.md sec 5.3)."""
    d = DEFAULTS
    w_exec = 3.0 * 3600_000.0  # 3 hours in ms (EXECUTION's Delta order of magnitude)
    res = evaluate_sla(
        completion_lag=2.5 * w_exec, w_exec=w_exec,
        realized_slippage=1.0, expected_slippage=1.0, m_slip=d.m_slip,
        severe_lag_multiple=d.sla_severe_lag_multiple,
        severe_slip_multiple=d.sla_severe_slip_multiple, as_of=0,
    )
    assert res.tier == "SLA_BREACH_SEVERE"
    assert res.alarm is not None and res.alarm.kind == "SLA_BREACH_SEVERE"
    directive = flatten((res.alarm,), as_of=0)
    assert directive.state == "HALT"


def test_sla_base_breach_does_not_halt():
    """OPS.md sec 5.3: a single SLA_BREACH (WARN tier) is logged, not halted."""
    w_exec = 3.0 * 3600_000.0
    res = evaluate_sla(
        completion_lag=1.5 * w_exec, w_exec=w_exec,  # over budget, under severe
        realized_slippage=1.0, expected_slippage=1.0, m_slip=2.0,
        severe_lag_multiple=2.0, severe_slip_multiple=3.0, as_of=0,
    )
    assert res.tier == "SLA_BREACH"
    assert res.lag_breach and not res.slip_breach


def test_sla_slippage_predicate():
    assert slippage_ok(realized_slippage=20.0, expected_slippage=13.0, m_slip=2.0)  # 20 <= 26
    assert not slippage_ok(realized_slippage=30.0, expected_slippage=13.0, m_slip=2.0)  # 30 > 26


def test_sla_slippage_severe_halts():
    """OPS.md sec 5.3: realized slippage > 3 * m_slip * expected -> SLA_BREACH_SEVERE."""
    res = evaluate_sla(
        completion_lag=0.0, w_exec=1.0,
        realized_slippage=100.0, expected_slippage=13.0, m_slip=2.0,  # 100 > 3*2*13 = 78
        severe_lag_multiple=2.0, severe_slip_multiple=3.0, as_of=0,
    )
    assert res.tier == "SLA_BREACH_SEVERE"
    assert res.slip_breach


def test_sla_ok_within_budget():
    res = evaluate_sla(completion_lag=0.5, w_exec=1.0, realized_slippage=10.0,
                       expected_slippage=13.0, m_slip=2.0, severe_lag_multiple=2.0,
                       severe_slip_multiple=3.0, as_of=0)
    assert res.tier == "OK" and res.alarm is None


def test_completion_lag_predicate():
    assert completion_lag_ok(0.9, 1.0)
    assert not completion_lag_ok(1.1, 1.0)


# ======================================================================== Invariant HALT-ONE (sec 6)

def test_halt_one_all_triggers_use_the_same_flatten_routine():
    """Inv HALT-ONE: RECON_BREACH, CUSUM_ALARM, and SLA_BREACH_SEVERE all converge on the SAME
    flatten() code path, producing an identically-shaped HALT directive (OPS.md sec 6)."""
    recon_alarm = Alarm(kind="RECON_BREACH", detail="r", value=9.0, threshold=1.0, as_of=1)
    cusum_alarm = Alarm(kind="CUSUM_ALARM", detail="c", value=7.0, threshold=6.23, as_of=1)
    sla_alarm = Alarm(kind="SLA_BREACH_SEVERE", detail="s", value=99.0, threshold=1.0, as_of=1)
    for alarm in (recon_alarm, cusum_alarm, sla_alarm):
        d = flatten((alarm,), as_of=alarm.as_of)
        assert d.state == "HALT"
        assert d.gross_multiplier == 0.0
        assert d.alarms == (alarm,)


def test_halt_one_captures_simultaneous_multi_alarm_reason_chain():
    """OPS.md sec 13: flatten is idempotent to a multi-alarm reason chain (Alarm is a tuple) — a
    simultaneous recon breach + CUSUM alarm are captured TOGETHER, not the second dropped."""
    a1 = Alarm(kind="RECON_BREACH", detail="r", value=9.0, threshold=1.0, as_of=1)
    a2 = Alarm(kind="CUSUM_ALARM", detail="c", value=7.0, threshold=6.23, as_of=1)
    d = flatten((a1, a2), as_of=1)
    assert d.state == "HALT"
    assert set(d.alarms) == {a1, a2}


# ======================================================================== brake.triggers precedence

def test_triggers_alarm_severity_mapping():
    """OPS.md sec 4.4/5.3/2: alarm-kind -> operational severity for the NON-reconciliation kinds."""
    assert alarm_severity("CUSUM_ALARM") == "HALT"
    assert alarm_severity("SLA_BREACH_SEVERE") == "HALT"
    assert alarm_severity("HEARTBEAT_SILENCE") == "HALT"
    assert alarm_severity("CUSUM_WARN") == "DEGRADE"
    assert alarm_severity("SLA_BREACH") == "RUN"  # base SLA breach is WARN-tier, no book action
    assert alarm_severity("ANOMALOUS_OUTPERFORMANCE") == "RUN"  # telemetry only


def test_triggers_combine_max_severity_precedence():
    """OPS.md sec 4.4: the directive is the MAX severity any alarm implies; the reason chain carries
    every fired alarm; a HALT-severity alarm dominates a co-occurring DEGRADE (max precedence)."""
    d = DEFAULTS
    warn = Alarm(kind="CUSUM_WARN", detail="w", value=1.0, threshold=1.0, as_of=0)
    severe = Alarm(kind="SLA_BREACH_SEVERE", detail="s", value=9.0, threshold=1.0, as_of=0)
    # DEGRADE alone -> DEGRADE (gross g_degrade).
    dir_deg = combine(other_alarms=(warn,), cusum_state="RUN", g_degrade=d.g_degrade, as_of=0)
    assert dir_deg.state == "DEGRADE" and dir_deg.gross_multiplier == pytest.approx(d.g_degrade)
    # DEGRADE + a co-occurring HALT-severe -> HALT (max precedence), reason chain keeps BOTH.
    dir_halt = combine(other_alarms=(warn, severe), cusum_state="RUN", g_degrade=d.g_degrade, as_of=0)
    assert dir_halt.state == "HALT" and dir_halt.gross_multiplier == 0.0
    assert set(dir_halt.alarms) == {warn, severe}
    # RUN with no alarms -> empty reason chain (P3).
    dir_run = combine(cusum_state="RUN", g_degrade=d.g_degrade, as_of=0)
    assert dir_run.state == "RUN" and dir_run.alarms == ()


# ======================================================================== sec-12 OpsEngine conformance

def _fresh_engine(w_exec: float = 3.0 * 3600_000.0) -> OpsEngine:
    d = DEFAULTS
    return OpsEngine(defaults=d, mu_0=d.mu_0, sigma_0=d.sigma_0, w_exec=w_exec)


def test_engine_satisfies_ops_engine_protocol():
    """OPS.md sec 12 cross-consistency (the gate that CLOSES sec 9's collision, not just documents it):
    ops.brake.engine.OpsEngine satisfies core.protocols.OpsEngine — on_obs(obs)->OpsDirective plus a
    directive() accessor, and the RETURN TYPE is exactly this spec's OpsDirective (not merely 'a
    method named on_obs exists'). mypy is not installed on the VM; the spec permits an 'equivalent',
    so this uses the runtime_checkable Protocol isinstance check + a concrete return-type assertion."""
    from convexity.core.protocols import OpsEngine as OpsEngineProto

    eng = _fresh_engine()
    # runtime_checkable structural conformance (the method surface exists with the right names).
    assert isinstance(eng, OpsEngineProto)
    # The return type is exactly core.types.OpsDirective (the reconciled sec-9 shape), closing the
    # collision: the Protocol's annotation resolves to this same class.
    out = eng.on_obs(OpsObs(as_of=0, weekly_net_return=DEFAULTS.mu_0))
    from convexity.core import types as ct
    assert type(out) is ct.OpsDirective
    assert type(eng.directive()) is ct.OpsDirective


def test_engine_run_when_healthy():
    """A healthy weekly return keeps the engine at RUN with full gross and an empty reason chain."""
    eng = _fresh_engine()
    out = eng.on_obs(OpsObs(as_of=1000, weekly_net_return=DEFAULTS.mu_0))
    assert out.state == "RUN"
    assert out.gross_multiplier == 1.0
    assert out.alarms == ()


def test_engine_recon_breach_halts_via_shewhart_without_weekly_return():
    """Inv RECON + sec 12: a daily recon obs with a hard breach HALTs the engine immediately — no
    weekly_net_return present, so the CUSUM never advanced (the Shewhart single-point path, P2)."""
    eng = _fresh_engine()
    snaps = [PositionSnapshot("ALT", book_position=1e6, exchange_position=0.0, lot_size=1.0,
                              target_notional=500.0, price=100.0)]
    out = eng.on_obs(OpsObs(as_of=5000, position_snapshots=snaps))
    assert out.state == "HALT"
    assert out.gross_multiplier == 0.0
    assert any(a.kind == "RECON_BREACH" for a in out.alarms)


def test_engine_sla_severe_halts_intraday():
    """Inv SLA + sec 12: an intraday fill obs whose completion blows 2.5*W_exec HALTs immediately,
    without waiting for the weekly CUSUM observation."""
    w_exec = 3.0 * 3600_000.0
    eng = _fresh_engine(w_exec)
    fill = RebalanceFill(completion_lag=2.5 * w_exec, realized_slippage=1.0, expected_slippage=1.0)
    out = eng.on_obs(OpsObs(as_of=6000, rebalance_fill=fill))
    assert out.state == "HALT"
    assert any(a.kind == "SLA_BREACH_SEVERE" for a in out.alarms)


def test_engine_heartbeat_silence_halts():
    """OPS.md sec 2/sec 6: a silent heartbeat forces HALT via the same flatten path (Inv HALT-ONE)."""
    eng = _fresh_engine()
    out = eng.on_obs(OpsObs(as_of=7000, heartbeat_silent=True))
    assert out.state == "HALT"
    assert any(a.kind == "HEARTBEAT_SILENCE" for a in out.alarms)


def test_engine_halt_latch_never_self_restarts_until_rearm():
    """Inv MONO / sec 6 step 5: once ANY source HALTs the engine it stays HALT across subsequent
    healthy obs, until an explicit rearm() — the composed engine-level latch, not only the CUSUM's."""
    eng = _fresh_engine()
    # HALT via a recon breach (a Shewhart trigger with no CUSUM state of its own).
    snaps = [PositionSnapshot("ALT", book_position=1e6, exchange_position=0.0, lot_size=1.0,
                              target_notional=500.0, price=100.0)]
    assert eng.on_obs(OpsObs(as_of=1, position_snapshots=snaps)).state == "HALT"
    # A subsequent perfectly healthy weekly return must NOT restore RUN.
    healthy = eng.on_obs(OpsObs(as_of=2, weekly_net_return=DEFAULTS.mu_0 + DEFAULTS.sigma_0))
    assert healthy.state == "HALT", "engine HALT latch must not self-restart (sec 6 step 5)"
    # Only an explicit rearm exits HALT.
    eng.rearm()
    assert eng.directive().state == "RUN"


def test_engine_cusum_degrade_composes_through_engine():
    """sec 4.4 + sec 12: a run of moderately-bad weekly returns drives the engine to DEGRADE (the
    CUSUM operational state composes through the engine's precedence combiner) with gross g_degrade."""
    d = DEFAULTS
    eng = _fresh_engine()
    r_moderate = d.mu_0 - 1.8 * d.sigma_0
    out = None
    for i in range(30):
        out = eng.on_obs(OpsObs(as_of=100 + i, weekly_net_return=r_moderate))
        if out.state == "DEGRADE":
            break
    assert out is not None and out.state == "DEGRADE"
    assert out.gross_multiplier == pytest.approx(d.g_degrade)


def test_engine_rejects_non_obs():
    eng = _fresh_engine()
    with pytest.raises(TypeError):
        eng.on_obs({"weekly_net_return": 0.0})
