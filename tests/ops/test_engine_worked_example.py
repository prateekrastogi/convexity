"""End-to-end engine + the §10 worked-example fixture (Invariant WE, KILL, AUDIT, PROV)."""

from __future__ import annotations

from convexity.ops.brake.engine import OpsEngine
from convexity.ops.config.resolve import resolve_config
from convexity.ops.forensics.log import ForensicLog
from convexity.ops.forensics.replay import replay_halts
from convexity.ops.ops_api import build_engine
from convexity.ops.streams.cost_tca import CostTcaStream, TcaFact
from convexity.ops.synth.generate import cost_drift_section10
from convexity.ops.types import EquityRef, OpsDirective


def _engine_no_warm() -> OpsEngine:
    """Engine with CUSUM/PH warm-up disabled and cost_tca prewhitening OFF (φ=()) so the §10 plain
    standardization ``z=(e_cost-0)/2.0`` reproduces bit-for-bit (the §10 table assumes no AR)."""
    from dataclasses import replace

    from convexity.ops.config.in_control import fixture_system_report
    report = {k: dict(v) for k, v in fixture_system_report().items()}
    report["cost_tca"]["phi"] = ()       # §10 uses plain standardization, not the AR(1) prewhitener
    report["cost_tca"]["rho"] = 0.0
    cfg = resolve_config(system_report=report)
    cfg = replace(cfg, defaults=replace(cfg.defaults, n_warm_cusum=0, n_warm_page_hinkley=0))
    return OpsEngine(cfg)


def test_section10_cusum_recursion_fixture():
    """Invariant WE: the §10 z-sequence + C+ recursion reproduce bit-for-bit (DEGRADE@15, HALT@18)."""
    from convexity.ops.detectors.cusum import Cusum
    e_cost, z = cost_drift_section10()
    assert list(z[:11]) == [0.2, -0.5, 0.4, -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.1, 1.5]
    c = Cusum(k=0.5, h=4.77, escalation_h=8.0)
    cps = [c.update(float(zi)).diagnostics["c_plus"] for zi in z]
    assert cps[9] == 0.0       # in control
    assert cps[14] == 5.0      # crosses h at sample 15
    assert cps[17] == 8.0      # crosses h2 at sample 18


def test_section10_through_engine_degrade_then_halt():
    """The §10 cost drift through the full engine: DEGRADE at sample 15, HALT at sample 18 (the 2-of-2
    AND confirmed by Page-Hinkley within M_conf, §6.6a)."""
    eng = _engine_no_warm()
    stream = CostTcaStream()
    e_cost, _z = cost_drift_section10()
    states = []
    for v in e_cost:
        eng.on_obs(stream.observe_cost(TcaFact(cost_realized_bps=float(v), cost_forecast_bps=0.0)))
        states.append(eng.evaluate())
    # find first DEGRADE and first HALT
    first_degrade = next(i for i, s in enumerate(states) if s.state == "degrade")
    first_halt = next(i for i, s in enumerate(states) if s.state == "halt")
    assert first_degrade + 1 == 15      # sample index (1-based) where DEGRADE fires
    assert first_halt + 1 == 18         # HALT escalation
    assert "cost_divergence" in states[first_degrade].triggers
    assert "cost_divergence_escalated" in states[first_halt].triggers


def test_engine_forensic_log_replayable():
    """Invariant AUDIT: a halt is reconstructed from the log alone (stream/detector/stat/limit/idx)."""
    log = ForensicLog()
    eng = build_engine(log=log)
    # drive a recon mismatch → immediate HALT
    eng.on_recon(True, "pos delta 3")
    eng.evaluate()
    halts = replay_halts(log)
    assert halts and halts[0].verdict.state == "halt"
    assert "reconciliation_mismatch" in halts[0].verdict.triggers
    assert "HALT" in halts[0].summary()


def test_engine_recon_halt_single_flatten_across_bars():
    """Invariant KILL/SAFE: a HALT held across multiple bars issues exactly ONE flatten wave."""
    eng = build_engine()
    eng.on_recon(True, "")
    v1 = eng.evaluate()
    eng.on_recon(True, "")
    v2 = eng.evaluate()
    assert v1.flatten and not v2.flatten     # latch: one wave


def test_engine_emits_only_directive_no_trade_params():
    """Invariant PROV: the engine's only output is an OpsDirective (no size/edge/exit)."""
    eng = build_engine()
    eng.on_equity(EquityRef(equity=1000.0, peak=1000.0))
    d = eng.directive()
    assert isinstance(d, OpsDirective)
    assert d.state == "run"
    # OpsDirective carries only state + flatten/cancel + reasons + version (no size/edge field)
    fields = set(d.__dataclass_fields__)
    assert fields == {"state", "flatten", "cancel_entries", "reason_codes", "engine_version"}


def test_dd_backstop_halts_through_engine():
    """The §12 drawdown ramp: sitting at ≥45% → OPS HALT (binary backstop above the governor)."""
    eng = build_engine()
    eng.on_equity(EquityRef(equity=540.0, peak=1000.0))   # 46% DD
    v = eng.evaluate()
    assert v.state == "halt"
    assert "max_drawdown_backstop" in v.triggers
