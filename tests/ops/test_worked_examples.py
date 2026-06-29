"""OPS worked-example regression fixtures (OPS §10, §6.5, §6.6, §6.7, §9.1, Invariant WE/ARL/SAFE).

Reproduces, bit-for-bit where exact:
- the §10 CUSUM cost-drift fixture (full C+ recursion, DEGRADE@15 / HALT@18) end-to-end through the
  engine;
- the SPRT boundaries (+2.890, -2.251) and the boundary-weighted ASN (99, 118), explicitly NOT 40-50;
- the two-sided CUSUM ARL0 743 -> 370 with h≈4.77 (NOT ≈3.5);
- the AR(1)-Student-t realized ARL0 within CI of 370 ONLY AFTER prewhitening (raw floods);
- the double-consumed HALT -> single flatten wave; the flatten-rejected -> stays latched;
- the DD backstop above the governor.

Scoped to OPS only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from convexity.ops.brake.engine import OpsEngine
from convexity.ops.brake.kill_switch import KillSwitch
from convexity.ops.config.in_control import fixture_system_report
from convexity.ops.config.resolve import resolve_config
from convexity.ops.detectors.cusum import Cusum, siegmund_arl_oneside, two_sided_arl0_from_oneside
from convexity.ops.detectors.sprt import asn, wald_boundaries
from convexity.ops.ops_api import build_engine
from convexity.ops.streams.cost_tca import CostTcaStream, TcaFact
from convexity.ops.streams.whiten import CausalPrewhitener
from convexity.ops.synth.generate import cost_drift_section10
from convexity.ops.types import AlarmRecord, EquityRef, InControlParams, Severity, TriggerName
from convexity.ops.validate import arl
from convexity.ops.validate import derive_limits as dl


def _tr(trigger: TriggerName, sev: Severity):
    from convexity.ops.brake.triggers import TriggerResult
    return TriggerResult(
        trigger, sev,
        AlarmRecord(stream="cost_tca", detector="cusum", trigger=trigger, severity=sev,
                    statistic=8.0, limit=8.0, sample_idx=18, in_control=None),
    )


def _engine_no_warm() -> OpsEngine:
    """Engine with CUSUM/PH warm-up off and cost_tca prewhitening off so the §10 plain
    standardization z=(e_cost-0)/2.0 reproduces bit-for-bit (the §10 table assumes no AR)."""
    report = {k: dict(v) for k, v in fixture_system_report().items()}
    report["cost_tca"]["phi"] = ()
    report["cost_tca"]["rho"] = 0.0
    cfg = resolve_config(system_report=report)
    cfg = replace(cfg, defaults=replace(cfg.defaults, n_warm_cusum=0, n_warm_page_hinkley=0))
    return OpsEngine(cfg)


# --- §10 CUSUM cost-drift: the full C+ recursion, exact ----------------------------------------

def test_section10_z_sequence_and_full_cplus_recursion_exact():
    """§10: the standardized z-sequence and the full C+ recursion match the spec table exactly."""
    e_cost, z = cost_drift_section10()
    assert list(z[:18]) == [0.2, -0.5, 0.4, -0.3, 0.6, -0.2, 0.1, -0.4, 0.3, -0.1,
                            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    assert list(e_cost[10:18]) == [3.0] * 8
    c = Cusum(k=0.5, h=4.77, escalation_h=8.0)
    cps = [round(c.update(float(zi)).diagnostics["c_plus"], 10) for zi in z]
    # the exact §10 C+ column
    assert cps[:18] == [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert cps[14] == 5.0    # crosses h=4.77 at sample 15
    assert cps[17] == 8.0    # crosses h2=8.0 at sample 18


def test_section10_through_engine_degrade_at_15_halt_at_18():
    """§10 end-to-end: DEGRADE at sample 15 (CUSUM h confirmed by PH within M_conf), HALT at 18."""
    eng = _engine_no_warm()
    stream = CostTcaStream()
    e_cost, _z = cost_drift_section10()
    states = []
    for v in e_cost:
        eng.on_obs(stream.observe_cost(TcaFact(cost_realized_bps=float(v), cost_forecast_bps=0.0)))
        states.append(eng.evaluate())
    first_degrade = next(i for i, s in enumerate(states) if s.state == "degrade")
    first_halt = next(i for i, s in enumerate(states) if s.state == "halt")
    assert first_degrade + 1 == 15
    assert first_halt + 1 == 18
    assert "cost_divergence" in states[first_degrade].triggers
    assert "cost_divergence_escalated" in states[first_halt].triggers


# --- SPRT boundaries + boundary-weighted ASN ---------------------------------------------------

def test_sprt_boundaries_plus_2890_minus_2251():
    """§6.5: A=+2.890, B=-2.251 for (α=0.05, β=0.10)."""
    a, b = wald_boundaries(0.05, 0.10)
    assert abs(a - 2.890) < 0.01
    assert abs(b - (-2.251)) < 0.01


def test_sprt_asn_99_and_118_not_40_50():
    """§6.5: the boundary-weighted ASN is ASN_H0≈99, ASN_H1≈118 — NOT the dropped-weighting 40-50."""
    asn_h0 = asn(0.55, 0.55, 0.45, 0.05, 0.10)
    asn_h1 = asn(0.45, 0.55, 0.45, 0.05, 0.10)
    assert abs(asn_h0 - 99.0) < 2.0
    assert abs(asn_h1 - 118.0) < 2.0
    assert asn_h0 > 60.0 and asn_h1 > 60.0          # explicitly not the wrong 40-50 figure
    table = arl.sprt_asn_table(0.55, 0.45, 0.05, 0.10)
    assert abs(table.asn_h0 - 99.0) < 2.0 and abs(table.asn_h1 - 118.0) < 2.0


# --- two-sided ARL0 743 -> 370 with h≈4.77 (NOT ≈3.5) ------------------------------------------

def test_two_sided_arl0_743_to_370_h_about_477_not_35():
    """§6.6: the one-sided Siegmund 743 combines ÷2 to two-sided 370; solving for two-sided ARL0=370
    returns h≈4.77, NOT the ≈3.5 the bare one-sided solve gives (the load-bearing factor-of-2 bug)."""
    one = siegmund_arl_oneside(0.5, 4.77, 0.0)
    assert abs(one - 743.0) < 5.0
    assert abs(two_sided_arl0_from_oneside(one) - 371.5) < 2.0
    h = dl.solve_h_for_two_sided_arl0(0.5, 370.0)
    assert abs(h - 4.77) < 0.05
    # the bare one-sided solve for ARL0=370 lands on a distinctly smaller (wrong) h ≈ 3.5
    lo, hi = 1.0, 8.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if siegmund_arl_oneside(0.5, m, 0.0) < 370.0:
            lo = m
        else:
            hi = m
    h_bug = 0.5 * (lo + hi)
    assert h_bug < h - 0.4
    assert abs(h_bug - 3.5) < 0.6                    # the bug lands near 3.5


# --- AR(1)-Student-t realized ARL0 holds 370 ONLY after prewhitening (raw floods) ---------------

def test_ar1_student_t_raw_floods_prewhitened_holds_370():
    """§6.7d / Invariant ARL (the panel-does-not-flood proof, MC-AUTHORITATIVE h):

    The RAW autocorrelated + fat-tailed residual at the IID limit (h=4.77) FLOODS (realized ARL0 <<
    370); prewhitening (§6.7a) lifts it sharply but the t(5) tail still keeps it below 370 at the
    IID-derived h. The Invariant's literal claim — realized ARL0 HOLDS at 370 after the correction —
    is proven by the MC-authoritative solve (§6.6): solve h on the AR(1)-Student-t (prewhitened)
    synthetic so its REALIZED two-sided ARL0 = 370, then assert the realized ARL0 at that solved h
    CONTAINS 370. The IID h=4.77 is the floods-at-the-wrong-limit contrast."""
    ic = InControlParams(stream="cost_tca", mu0=0.0, sigma0=2.0, phi=(0.4,), rho=0.4, tail_dof=5.0)

    def raw(n, rng):
        return arl.ar1_student_t_stream(n, rho=0.4, dof=5.0, rng=rng)   # standardized raw (unit σ)

    def prewhitened(n, rng):
        r = arl.ar1_student_t_stream(n, rho=0.4, dof=5.0, rng=rng) * 2.0
        pw = CausalPrewhitener(ic)
        return np.array([pw.update(x) for x in r])

    raw_arl0 = arl.cusum_arl_mc(0.5, 4.77, n_runs=400, max_len=4000, seed=2, stream_fn=raw).mean
    pw_at_iid = arl.cusum_arl_mc(0.5, 4.77, n_runs=400, max_len=8000, seed=3,
                                 stream_fn=prewhitened).mean
    assert raw_arl0 < 150.0
    assert pw_at_iid > 1.8 * raw_arl0
    assert pw_at_iid < 320.0                 # the t(5) tail keeps it below 370 at the IID h

    h_star = dl.solve_h_for_two_sided_arl0_mc(
        0.5, 370.0, n_runs=350, seed=3, stream_fn=prewhitened,
        h_lo=4.0, h_hi=8.0, iters=12, max_len=10000,
    )
    assert h_star > 4.77
    realized = arl.cusum_arl_mc(0.5, h_star, n_runs=600, max_len=22000, seed=23,
                                stream_fn=prewhitened)
    assert realized.contains(370.0)          # the literal Invariant claim: realized ARL0 HOLDS at 370


# --- double-consumed HALT -> single flatten wave; rejected flatten stays latched ----------------

def test_double_consumed_halt_single_flatten_then_rejected_stays_latched():
    """§9.1a/§9.1c: a multi-bar HALT issues exactly ONE flatten wave; a rejected flatten stays latched
    HALT with retry + unconfirmed, NEVER a spurious 'flat'."""
    ks = KillSwitch(flatten_retry_interval=1, flatten_retry_max=3)
    v1 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert v1.flatten and v1.cancel_entries          # cancel entries first, then flatten (§9.1b)
    v2 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert not v2.flatten and v2.flatten_unconfirmed  # latch → no duplicate wave
    v3 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)], flatten_rejected=True)
    assert v3.state == "halt" and v3.flatten          # retried on the RED cadence
    assert "flatten_unconfirmed" in v3.reason_codes
    assert not ks.flatten_confirmed                   # never spuriously 'flat'


def test_double_consumed_halt_single_flatten_through_engine():
    """The §10/§12 double-consumed HALT through the full engine path (recon mismatch held two bars)."""
    eng = build_engine()
    eng.on_recon(True, "")
    v1 = eng.evaluate()
    eng.on_recon(True, "")
    v2 = eng.evaluate()
    assert v1.flatten and not v2.flatten              # exactly one flatten wave


# --- DD backstop above the governor ------------------------------------------------------------

def test_dd_backstop_above_governor_through_engine():
    """§7.1a: sitting at ≥45% DD HALTs (the binary backstop strictly above the 43% governor flat)."""
    eng = build_engine()
    eng.on_equity(EquityRef(equity=570.0, peak=1000.0))   # 43% DD — governor territory, no kill
    assert eng.evaluate().state != "halt"
    eng2 = build_engine()
    eng2.on_equity(EquityRef(equity=540.0, peak=1000.0))  # 46% DD — past the 45% backstop
    v = eng2.evaluate()
    assert v.state == "halt" and "max_drawdown_backstop" in v.triggers
