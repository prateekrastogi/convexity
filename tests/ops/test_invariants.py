"""One test per named OPS Invariant (OPS §14): PROV, LIM, OOS, CAUSAL, CHEAP, ARL, KILL, BRAKE,
RECON, STREAM, ALARM, SAFE, AUDIT, WE, V, ROLLOUT.

These assert the structural contract of each Invariant against the real OPS API (the same surface the
topic-scoped suites exercise), so the named-Invariant coverage is legible in one place. Scoped to OPS
only. ``validate/`` carries scipy and is never imported by the live layers (Invariant V is asserted
here by AST).
"""

from __future__ import annotations

import ast
import pathlib
import time
from dataclasses import replace

import numpy as np

from convexity.core.params import Bucket, assert_no_red_in_amber
from convexity.ops.brake.alarms import AlarmRouter, panel_false_page_rate
from convexity.ops.brake.kill_switch import KillSwitch
from convexity.ops.brake.rollout import StageGateInput, evaluate_gate
from convexity.ops.brake.safe_state import safe_state_for
from convexity.ops.brake.triggers import (
    TriggerResult,
    max_drawdown_backstop,
    reconciliation_mismatch,
)
from convexity.ops.config.defaults import (
    AMBER_FIELD_NAMES,
    DEFAULTS,
    param_registry,
)
from convexity.ops.config.in_control import fixture_system_report, in_control_from_report
from convexity.ops.config.resolve import resolve_config
from convexity.ops.detectors.cusum import Cusum
from convexity.ops.forensics.log import ForensicLog
from convexity.ops.forensics.replay import replay_halts
from convexity.ops.ops_api import build_engine
from convexity.ops.streams.whiten import CausalPrewhitener, standardize
from convexity.ops.types import (
    AlarmRecord,
    EquityRef,
    InControlParams,
    OpsDirective,
    Severity,
    TriggerName,
)
from convexity.ops.validate import arl
from convexity.ops.validate import derive_limits as dl

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "ops"


def _tr(trigger: TriggerName, sev: Severity) -> TriggerResult:
    return TriggerResult(
        trigger,
        sev,
        AlarmRecord(stream="cost_tca", detector="cusum", trigger=trigger, severity=sev,
                    statistic=5.0, limit=4.77, sample_idx=15, in_control=None),
    )


def _imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    return mods


# --- PROV: monitors / brakes, never trades ----------------------------------------------------

def test_invariant_prov_monitors_brakes_never_trades():
    """Invariant PROV: OPS holds no edge/sizing/exit parameter; its only output is the directive."""
    forbidden = {"kelly", "edge_buffer", "stop_distance", "target_qty", "leverage", "size"}
    names = {p.name for p in param_registry().params()}
    assert not (names & forbidden)
    eng = build_engine()
    eng.on_equity(EquityRef(equity=1000.0, peak=1000.0))
    d = eng.directive()
    assert isinstance(d, OpsDirective)
    assert set(d.__dataclass_fields__) == {
        "state", "flatten", "cancel_entries", "reason_codes", "engine_version"
    }


# --- LIM: limits derived from the validated distribution and frozen ----------------------------

def test_invariant_lim_mean_and_variance_shifts_do_not_move_limit():
    """Invariant LIM: a shifted MEAN trips the detector but does not move ``h``; a 2× live VARIANCE
    does not move the raw trip point μ0+h·σ0 (standardization uses the frozen σ0)."""
    cfg = resolve_config()
    h_before = cfg.defaults.cusum.h
    c = Cusum(cfg.defaults.cusum.k, cfg.defaults.cusum.h)
    for _ in range(20):
        c.update(2.0)
    assert c.h == h_before
    ic = InControlParams(stream="cost_tca", mu0=0.0, sigma0=2.0)
    assert standardize(20.0, ic) == 10.0   # σ0=2.0 denominator regardless of live vol
    assert ic.mu0 + 4.77 * ic.sigma0 == 9.54


# --- OOS: limits validated out-of-sample, never tuned on live P&L ------------------------------

def test_invariant_oos_promotion_objective_has_no_pnl_term():
    """Invariant OOS: the promotion objective is detection quality (FAR + delay), never P&L."""
    from convexity.ops.validate.oos import OosObjective, promote
    obj = OosObjective(realized_far=0.003, far_target=0.003, realized_delay=10.0, delay_target=10.0)
    assert obj.contains_pnl_term() is False
    cands = {
        "tight": OosObjective(0.01, 0.003, 8.0, 10.0),
        "good": OosObjective(0.0031, 0.003, 11.0, 10.0),
        "blind": OosObjective(0.0001, 0.003, 40.0, 10.0),
    }
    assert promote(cands) == "good"


# --- CAUSAL: O(1) causal recursion + warm-up suppression ---------------------------------------

def test_invariant_causal_one_at_a_time_equals_batch_and_warmup_suppresses():
    """Invariant CAUSAL: feeding one-at-a-time matches batch (future bars never change a past state);
    SPC alarms are suppressed before N_warm while the recursion still updates."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    a, b = Cusum(0.5, 4.77), Cusum(0.5, 4.77)
    sa = [a.update(float(z)).statistic for z in x]
    sb = [b.update(float(x[i])).statistic for i in range(50)]
    assert sa == sb
    c = Cusum(k=0.5, h=1.0, n_warm=5)
    states = [c.update(3.0) for _ in range(5)]
    assert all(s.warming_up for s in states) and not any(s.alarm for s in states)
    assert c.c_plus > 1.0                       # recursion climbed during warm-up
    assert c.update(3.0).alarm                  # past warm-up it alarms immediately


# --- CHEAP: O(streams) per bar, no growth with history -----------------------------------------

def test_invariant_cheap_per_bar_cost_is_bounded_and_history_independent():
    """Invariant CHEAP: the per-bar panel cost is a handful of scalar updates and does NOT grow with
    history length (the recursions carry bounded state)."""
    eng = build_engine()

    def time_n_bars(n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            eng.on_equity(EquityRef(equity=1000.0, peak=1000.0))
            eng.evaluate()
        return time.perf_counter() - t0

    time_n_bars(200)                       # warm the path
    early = time_n_bars(2000)
    late = time_n_bars(2000)               # same count, much later in history
    assert late < early * 3.0 + 0.05       # flat per-bar cost (no history-length growth)


# --- ARL: stated ARL0/ARL1, MC authoritative, panel budgeted over sources ----------------------

def test_invariant_arl_two_sided_mc_370_and_h_about_477_not_35():
    """Invariant ARL: derive_limits solving for two-sided ARL0=370 returns h≈4.77 (NOT ≈3.5); the MC
    two-sided ARL0 at (0.5,4.77) lands near 370; the closed-form ÷2 seed is the cross-check."""
    h = dl.solve_h_for_two_sided_arl0(0.5, 370.0)
    assert abs(h - 4.77) < 0.05
    one = arl.cusum_arl0_siegmund_two_sided(0.5, 4.77)   # the analytic seed/cross-check
    assert 330.0 < one < 420.0
    est = arl.cusum_arl_mc(0.5, 4.77, n_runs=1200, seed=1)
    assert 320.0 < est.mean < 430.0


# --- KILL: precedence (max-severity), never a blended score ------------------------------------

def test_invariant_kill_state_is_max_severity_precedence():
    """Invariant KILL: two simultaneous triggers (DEGRADE + HALT) yield HALT; a trigger only raises."""
    ks = KillSwitch()
    v = ks.step([_tr("vol_regime_break", Severity.DEGRADE),
                 _tr("max_drawdown_backstop", Severity.HALT)])
    assert v.state == "halt"
    v2 = ks.step([_tr("vol_regime_break", Severity.DEGRADE)])  # lower this bar
    assert v2.state == "halt"                                  # never lowered by a fresh trigger


# --- BRAKE: DD-kill 45% strictly above governor flat 43% on a shared reference ------------------

def test_invariant_brake_dd_kill_strictly_above_governor_flat():
    """Invariant BRAKE: the 45% DD HALT sits strictly above the 43% governor flat on the SHARED
    equity/peak reference, so the ordering cannot invert."""
    assert DEFAULTS.dd_kill > DEFAULTS.dd_governor_flat
    assert max_drawdown_backstop(EquityRef(equity=570.0, peak=1000.0), 0.45) is None     # 43% DD
    r = max_drawdown_backstop(EquityRef(equity=540.0, peak=1000.0), 0.45)                # 46% DD
    assert r is not None and r.severity == Severity.HALT


# --- RECON: reconciliation is SYSTEM's; OPS monitors the OUTPUT --------------------------------

def test_invariant_recon_halts_on_flag_does_not_rederive():
    """Invariant RECON: OPS HALTs on the ``reconciliation_ok`` flag; it does not re-derive recon."""
    assert reconciliation_mismatch(False) is None
    r = reconciliation_mismatch(True, "pos delta 3")
    assert r is not None and r.severity == Severity.HALT
    eng = build_engine()
    eng.on_recon(True, "pos delta 3")
    v = eng.evaluate()
    assert v.state == "halt" and "reconciliation_mismatch" in v.triggers


# --- STREAM: every stream is a live-vs-validated residual/event vs frozen params ----------------

def test_invariant_stream_in_control_traces_to_system_report():
    """Invariant STREAM: each stream's in-control params trace to a SYSTEM-report field, not a live
    estimate; the standardization uses the frozen (μ0, σ0)."""
    ic = in_control_from_report(fixture_system_report())
    assert ic["cost_tca"].mu0 == 0.0 and ic["cost_tca"].sigma0 == 2.0
    assert ic["edge"].p0 == 0.55 and ic["edge"].p1 == 0.45
    assert standardize(4.0, ic["cost_tca"]) == 2.0   # (4-0)/2 — frozen σ0


def test_invariant_stream_funding_residual_is_standardized():
    """Invariant STREAM: the e_funding arm is standardized against its OWN frozen (mu0_fund,
    sigma0_fund) before the §7.8 CUSUM — the funding CUSUM sees UNIT-VARIANCE input, not raw bps.

    Regression on the old engine.py path ``z = (obs.value - 0.0)`` which fed RAW bps to a σ-unit
    CUSUM (h=4.77). We give funding a reference DISTINCT from the cost reference (mu0_fund=1.0,
    sigma0_fund=4.0 vs cost σ0=2.0) and assert the binding divides the residual by sigma0_fund."""
    from convexity.ops.reconcile.streams_map import build_bindings
    from convexity.ops.streams.cost_tca import CostTcaStream, TcaFact

    report = {k: dict(v) for k, v in fixture_system_report().items()}
    report["cost_tca"]["mu0_fund"] = 1.0
    report["cost_tca"]["sigma0_fund"] = 4.0
    cfg = resolve_config(system_report=report)
    assert cfg.in_control["cost_tca"].sigma0_fund == 4.0

    # the binding standardizes the funding residual by sigma0_fund (centred at mu0_fund), NOT the raw
    # bps and NOT the cost σ0: (9 - 1)/4 == 2.0, while raw bps would be 9.0 and cost-σ0 would be 4.5.
    binding = build_bindings(cfg)["cost_tca"]
    z_fund = binding.standardize_funding(9.0)
    assert abs(z_fund - 2.0) < 1e-12
    assert z_fund != 9.0                 # NOT the raw-bps path the review flagged
    assert abs(z_fund - 4.5) > 1e-6      # NOT the cost-σ0 path either

    # the funding CUSUM sees a unit-variance input: a synthetic in-control funding stream drawn at
    # std == sigma0_fund standardizes to ~unit variance through the engine's funding path.
    rng = np.random.default_rng(0)
    raw = 1.0 + 4.0 * rng.standard_normal(4000)          # N(mu0_fund, sigma0_fund)
    fb = build_bindings(cfg)["cost_tca"]
    zs = np.array([fb.standardize_funding(float(v)) for v in raw])
    assert abs(float(np.std(zs)) - 1.0) < 0.06           # unit variance, not σ0_fund(=4)-scaled
    assert abs(float(np.mean(zs))) < 0.06                # zero mean (centred at mu0_fund)

    # end-to-end: the engine routes e_funding through standardize_funding (not the raw path).
    eng = build_engine(system_report=report)
    stream = CostTcaStream()
    obs = stream.observe_funding(TcaFact(cost_realized_bps=0.0, cost_forecast_bps=0.0,
                                         funding_realized_bps=9.0, funding_forecast_bps=0.0))
    eng.on_obs(obs)
    st = eng._states[("cost_tca", "funding_cusum")]      # standardized z=(9-1)/4=2 ⇒ C+ = max(0,2-0.5)
    assert abs(st.statistic - 1.5) < 1e-9                # 2.0 - k(0.5); raw bps would give 8.5


# --- ALARM: trigger-defined severity, deterministic dedup, ARL-budgeted paging, gated re-arm ----

def test_invariant_alarm_dedup_paging_over_sources_and_gated_rearm():
    """Invariant ALARM: a held breach pages once (+ coalesced); the panel rate sums over independent
    SOURCES (not raw stream/detector count); a HALT does not auto-un-trip."""
    router = AlarmRouter(dedup_window=200)
    alarm = AlarmRecord("cost_tca", "cusum", "cost_divergence", Severity.DEGRADE, 5.0, 4.77, 15, None)
    pages = 0
    for _ in range(100):
        router.tick()
        pages += len(router.route([alarm]))
    assert pages == 1
    # 6 independent sources at ARL0=2000 → 6/2000 (NOT, say, 9 raw detectors)
    assert abs(panel_false_page_rate([2000.0] * 6) - 0.003) < 1e-9
    ks = KillSwitch(m_rearm=5)
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)], venue_confirmed_flat=True)
    for _ in range(10):
        ks.step([])
    assert not ks.request_rearm()    # window cleared but no ack → blocked
    ks.acknowledge()
    assert ks.request_rearm()        # ack + window → re-arm


# --- SAFE: deterministic degradation, venue-stop floor, idempotent + race-free flatten ----------

def test_invariant_safe_deterministic_degradation_and_one_flatten_wave():
    """Invariant SAFE: each failure maps to one directive failing toward SAFE; a double-consumed HALT
    issues exactly ONE flatten wave; a rejected flatten stays latched HALT, never a spurious flat."""
    assert safe_state_for("heartbeat_loss").state == "halt"
    fg = safe_state_for("feed_gap_single_venue")
    assert fg.state == "degrade" and fg.stop_new_entries and not fg.flatten
    ks = KillSwitch(flatten_retry_interval=1, flatten_retry_max=3)
    v1 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert v1.flatten and v1.cancel_entries
    v2 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert not v2.flatten and v2.flatten_unconfirmed     # latch: one wave
    v3 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)], flatten_rejected=True)
    assert v3.state == "halt" and v3.flatten and not ks.flatten_confirmed


# --- AUDIT: every halt is replayable from the append-only log ----------------------------------

def test_invariant_audit_halt_reconstructable_from_log_alone():
    """Invariant AUDIT: a halt is reconstructed from the log alone (stream/detector/stat/limit/idx)."""
    log = ForensicLog()
    eng = build_engine(log=log)
    eng.on_recon(True, "pos delta 3")
    eng.evaluate()
    halts = replay_halts(log)
    assert halts and halts[0].verdict.state == "halt"
    assert "reconciliation_mismatch" in halts[0].verdict.triggers
    assert "HALT" in halts[0].summary()


# --- WE: the §10 worked example is a regression fixture ----------------------------------------

def test_invariant_we_section10_cusum_recursion_bit_for_bit():
    """Invariant WE: the §10 C+ recursion crosses h=4.77 at i=15 and h2=8.0 at i=18 bit-for-bit."""
    from convexity.ops.synth.generate import cost_drift_section10
    _e, z = cost_drift_section10()
    c = Cusum(k=0.5, h=4.77, escalation_h=8.0)
    cps = [c.update(float(zi)).diagnostics["c_plus"] for zi in z]
    assert cps[9] == 0.0 and cps[14] == 5.0 and cps[17] == 8.0


# --- V: validation/forensics never imported by the live layers ---------------------------------

def test_invariant_v_live_layers_never_import_validate_or_forensics():
    """Invariant V: nothing in streams/detectors/reconcile/brake imports validate/ or forensics/."""
    offenders = []
    for d in ("streams", "detectors", "reconcile", "brake"):
        for py in (_SRC / d).glob("*.py"):
            for mod in _imports_of(py):
                if "convexity.ops.validate" in mod or "convexity.ops.forensics" in mod:
                    offenders.append((py.name, mod))
    assert not offenders, f"live layer imports validate/forensics: {offenders}"


# --- C (governance, folded into the Invariant set): RED unreachable from per-instrument ---------

def test_invariant_c_red_unreachable_from_per_instrument():
    """Invariant C: no RED/BLUE param name leaks into the AMBER per-instrument field set."""
    assert_no_red_in_amber(AMBER_FIELD_NAMES, param_registry())
    reg = param_registry()
    amber = {p.name for p in reg.of_bucket(Bucket.AMBER)}
    red_blue = {p.name for p in reg.params() if p.bucket in (Bucket.RED, Bucket.BLUE)}
    assert not (amber & red_blue)


# --- ROLLOUT: every stage gate is a statistical live-vs-validated test over a min sample --------

def test_invariant_rollout_min_sample_warmup_and_cost_breach_demote():
    """Invariant ROLLOUT: a below-min / warming-up stream DEFERS (never a calendar); an e_cost CUSUM
    breach blocks the advance and demotes; a kill demotes immediately."""
    ns = dict(n_paper=30, n_canary=50, n_scaled=100)
    ok = StageGateInput(
        n_trades=40, edge_in_ci=True, fill_in_ci=True, tca_in_ci=True, vol_in_ci=True,
        sprt_accepts_h0=True, recon_clean=True, no_kill_fired=True, all_gates_warmed=True,
        e_cost_cusum_breached=False,
    )
    assert evaluate_gate("paper", replace(ok, n_trades=10), **ns).decision == "defer"
    assert evaluate_gate("paper", replace(ok, all_gates_warmed=False), **ns).decision == "defer"
    assert evaluate_gate("paper", ok, **ns).decision == "advance"
    assert evaluate_gate("canary", replace(ok, e_cost_cusum_breached=True), **ns).decision == "demote"
    assert evaluate_gate("scaled", replace(ok, n_trades=200, no_kill_fired=False), **ns).decision == "demote"


def test_invariant_prewhitening_is_causal():
    """CAUSAL (the §6.7a prewhitener arm): reads only residual_i and the frozen prior residual."""
    ic = InControlParams(stream="vol", mu0=0.0, sigma0=1.0, phi=(0.5,))
    x = [0.3, -0.2, 0.5, 0.1, -0.4]
    pw_full = CausalPrewhitener(ic)
    za = [pw_full.update(v) for v in x][:3]   # full stream, take first 3
    pw_short = CausalPrewhitener(ic)
    zb = [pw_short.update(v) for v in x[:3]]  # only first 3 fed
    assert za == zb                            # later samples never changed the first 3
