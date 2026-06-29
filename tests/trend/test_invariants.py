"""TREND invariant tests (TREND §7). One test per named structural rule.

Every named Invariant of TREND §7 maps to one test here:
  A beliefs-vs-actions   B exactly-one-detector   C no-RED-in-spec   D no-claimed-posterior
  E Kalman-class-preset  F diagnostics-log-only    G validation-firewall (AST)
  H no-lookahead         I batch/live parity       J action-timing-next-bar
  K causal-stopping-time HC H_sys-cap              L/CVX convex-payoff   M net-of-cost-gate
  N intent-not-placement O entry-from-BREAKOUT      P hold-horizon/residual-move
plus the §18 structural gates: integrated-variance-monotone, survival-no-pool,
survival-validation-only.

H_sys (SYSTEM-owned, Inv HC) is injected as ``tau_cap`` via the ``cfg`` fixture (conftest).
"""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from convexity.core.params import assert_no_red_in_amber
from convexity.core.types import ExecutionIntent, Position
from convexity.trend import TrendLifeEngine, resolve_config
from convexity.trend.config.defaults import TREND_PARAM_REGISTRY
from convexity.trend.config.instrument import INSTRUMENT_SPEC_FIELDS
from convexity.trend.estimate.kalman import FixedPresetLocalTrend
from convexity.trend.estimate.sequential import SequentialDeterioration
from convexity.trend.online.replay import ParityReplay
from convexity.trend.online.snapshot import TrendStateSnapshot
from convexity.trend.policy.convex_exit import ConvexExit
from convexity.trend.synth.generate import (
    ALL_PATHS,
    jump_against,
    persistent_trend_low_noise,
    trend_reverses,
)
from convexity.trend.types import ActionDecision
from convexity.trend.validate.survival import assert_no_pooling

from tests.trend.conftest import make_position

SRC = Path(__file__).resolve().parents[2] / "src" / "convexity" / "trend"


# --- Invariant A: estimation emits beliefs, only decision.py emits ActionDecision ---------
def test_inv_a_only_decision_emits_actiondecision():
    """No estimate/ module constructs ActionDecision; only policy/decision.py does (Inv A)."""
    offenders = []
    for path in (SRC / "estimate").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "ActionDecision":
                    offenders.append(path.name)
    assert not offenders, f"estimate/ constructs ActionDecision: {offenders}"

    decision_src = (SRC / "policy" / "decision.py").read_text()
    assert "ActionDecision(" in decision_src


# --- Invariant B: exactly one deterioration detector --------------------------------------
def test_inv_b_exactly_one_detector(cfg):
    det = SequentialDeterioration(cfg.detector_kind, cfg)
    assert det.kind in ("cusum", "sprt")
    with pytest.raises(ValueError):
        SequentialDeterioration("both", cfg)  # type: ignore[arg-type]


# --- Invariant C: no RED/BLUE in InstrumentSpec -------------------------------------------
def test_inv_c_no_red_blue_in_amber():
    assert_no_red_in_amber(INSTRUMENT_SPEC_FIELDS, TREND_PARAM_REGISTRY)


# --- Invariant D: no claimed drift posterior ----------------------------------------------
def test_inv_d_no_drift_posterior():
    """BeliefState has no P(trend_alive) / P(mu>0) field (Inv D)."""
    from convexity.trend.types import BeliefState

    names = {f.name for f in dataclasses.fields(BeliefState)}
    for forbidden in ("p_alive", "p_trend_alive", "prob_alive", "posterior", "p_mu_pos"):
        assert forbidden not in names


# --- Invariant E: Kalman noise ratio is a class preset, fixed during run ------------------
def test_inv_e_kalman_fixed_class_preset(cfg, spec):
    eng = TrendLifeEngine(cfg)
    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    ratio_before = eng.builder.kalman.noise_ratio
    eng.run_batch(bars, pos, spec, cfg)
    assert eng.builder.kalman.noise_ratio == ratio_before  # never adapts online
    with pytest.raises(ValueError):
        FixedPresetLocalTrend(noise_ratio=0.0)


# --- Invariant F: diagnostics are log-only ------------------------------------------------
def test_inv_f_diagnostics_dont_change_decisions(cfg, spec):
    """Perturbing BeliefDiagnostics must not change the decision (Inv F)."""
    from convexity.trend.estimate.state import BeliefBuilder
    from convexity.trend.policy.decision import DecisionPolicy
    from convexity.trend.types import BeliefDiagnostics

    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    builder = BeliefBuilder(cfg)
    ce = ConvexExit()
    pol = DecisionPolicy(convex_exit=ce)
    vol0 = builder.vol.update(bars, e, pos, cfg)
    exit_state = ce.initial(bars, pos, vol0, cfg)
    idx = e + 30
    builder.reset()
    for i in range(e, idx + 1):
        belief = builder.update(bars, i, pos, cfg)
        if i > e:
            exit_state = ce.update(bars, i, pos, belief, exit_state, cfg)
    d1 = pol.decide(bars, idx, pos, belief, exit_state, cfg)
    junk = BeliefDiagnostics(
        innovation_whiteness=999.0, kalman_consistency=-1.0, detector_margin=42.0,
        jump_flag_rate=1.0, hazard_rung="state", effective_sample_size=0, tail_nu=1.0,
        vol_method="mad", vol_quality_flag="garbage",
    )
    belief2 = dataclasses.replace(belief, diag=junk)
    d2 = pol.decide(bars, idx, pos, belief2, exit_state, cfg)
    assert d1.action == d2.action and d1.stop_cause == d2.stop_cause


# --- Invariant G: validation labels are firewalled ----------------------------------------
def test_inv_g_label_firewall():
    """No estimate/ or policy/ module imports validate/ (Inv G), via AST import-graph scan."""
    offenders = []
    for sub in ("estimate", "policy"):
        for path in (SRC / sub).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and ("trend.validate" in mod or mod.startswith("convexity.trend.validate")):
                    offenders.append(path.name)
    assert not offenders, f"live modules import validate/: {offenders}"


# --- Invariant H: no lookahead (perturb future bars, output <= t identical) ---------------
def test_inv_h_no_lookahead_future_needle(cfg, spec):
    """Changing every bar AFTER t cannot change any decision at or before t (Inv H)."""
    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    eng = TrendLifeEngine(cfg)
    base = eng.run_batch(bars, pos, spec, cfg)
    t = min(len(base) - 1, 50)

    # perturb every bar AFTER t (close + high + open) with a large multiplicative needle.
    c2 = bars.c.copy()
    h2 = bars.h.copy()
    o2 = bars.o.copy()
    c2[t + 1 :] *= 1.5
    h2[t + 1 :] *= 1.5
    o2[t + 1 :] *= 1.5
    bars2 = dataclasses.replace(bars, c=c2, h=h2, o=o2)
    pert = TrendLifeEngine(cfg).run_batch(bars2, pos, spec, cfg)

    for i in range(min(t + 1, len(base), len(pert))):
        assert base[i].action == pert[i].action, i
        assert base[i].stop_cause == pert[i].stop_cause, i
        assert base[i].exit_state.trail_level == pytest.approx(pert[i].exit_state.trail_level)


# --- Invariant I: batch/live parity -------------------------------------------------------
@pytest.mark.parametrize("fn", [persistent_trend_low_noise, jump_against, trend_reverses])
def test_inv_i_batch_live_parity(cfg, spec, fn):
    bars, e = fn()
    pos = make_position(bars, e)
    assert ParityReplay().check(bars, pos, spec, cfg)


def test_inv_i_snapshot_restore(cfg, spec):
    """Snapshot/restore reproduces the same next decisions (Inv I)."""
    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    eng = TrendLifeEngine(cfg)
    mid = e + 40
    for i in range(e, mid + 1):
        eng.update(bars, pos, spec, cfg, i)
    snap = TrendStateSnapshot.of(eng).snapshot()

    cont = []
    for i in range(mid + 1, mid + 21):
        d = eng.update(bars, pos, spec, cfg, i)
        if d:
            cont.append((d.action, round(d.exit_state.trail_level, 9)))

    eng2 = TrendLifeEngine(cfg)
    TrendStateSnapshot.restore_bytes(snap).into(eng2)
    cont2 = []
    for i in range(mid + 1, mid + 21):
        d = eng2.update(bars, pos, spec, cfg, i)
        if d:
            cont2.append((d.action, round(d.exit_state.trail_level, 9)))
    assert cont == cont2


# --- Invariant J: action timing is explicit (next-bar) ------------------------------------
def test_inv_j_action_timing_next_bar(cfg, spec):
    bars, e = jump_against()
    pos = make_position(bars, e)
    eng = TrendLifeEngine(cfg)
    ds = eng.run_batch(bars, pos, spec, cfg)
    # every decision carries decision_idx, action_idx, action_price_kind (Inv J).
    for d in ds:
        assert d.action_idx >= d.decision_idx
        assert d.action_price_kind in ("next_open", "next_mid", "close_if_live_executable")
    exit_d = ds[-1]
    assert exit_d.action == "exit"
    # the exit acts at the NEXT bar, never implicitly the same bar (Inv J).
    assert exit_d.action_idx >= exit_d.decision_idx + 1 or exit_d.action_idx == len(bars) - 1


# --- Invariant K: dynamic horizon is a causal stopping time -------------------------------
def test_inv_k_causal_stopping_time(cfg, spec):
    """{tau_live <= t} is F_t-measurable: future bars cannot flip whether the stop fired by t.

    Distinct from Inv H (which pins the whole decision): Inv K pins the stopping EVENT.
    """
    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    base = TrendLifeEngine(cfg).run_batch(bars, pos, spec, cfg)
    t = min(len(base) - 1, 40)

    # increasing future MFE after t (a stronger right tail) must not change stopping_time_fired<=t.
    c2 = bars.c.copy()
    c2[t + 1 :] *= 2.0  # much more favorable future
    bars2 = dataclasses.replace(bars, c=c2, h=bars.h.copy() * 2.0)
    pert = TrendLifeEngine(cfg).run_batch(bars2, pos, spec, cfg)
    for i in range(min(t + 1, len(base), len(pert))):
        assert base[i].stopping_time_fired == pert[i].stopping_time_fired, i
        assert base[i].stop_cause == pert[i].stop_cause, i


# --- Invariant HC: operational cap is the SYSTEM-injected bound, forces time_cap ----------
def test_inv_hc_time_cap_forced(spec):
    """A still-alive trend is force-marked time_cap when the active clock crosses H_sys (Inv HC);
    no realized hold exceeds the contractual bound."""
    bars, e = persistent_trend_low_noise(n=600)
    small_cfg = resolve_config(spec, h_sys=0.05)  # 0.05 day ~ 72 minutes -> clock crosses early
    pos = make_position(bars, e)
    ds = TrendLifeEngine(small_cfg).run_batch(bars, pos, spec, small_cfg)
    assert ds[-1].action == "exit"
    assert ds[-1].stop_cause == "time_cap"
    cap = small_cfg.tau_cap
    assert all(d.horizon.elapsed_value <= cap + 1e-9 for d in ds[:-1])


def test_inv_hc_resolve_requires_h_sys(spec):
    with pytest.raises(TypeError):
        resolve_config(spec)  # h_sys is required (Inv HC) — never a TREND-internal constant
    with pytest.raises(ValueError):
        resolve_config(spec, h_sys=-1.0)


# --- Invariant L / INV-CVX: realized payoff is option-like (convex, bounded-below) --------
def test_inv_l_convex_payoff(cfg, spec):
    """Drive the ACTUAL ``ConvexExit.trail_multiple``/``update`` schedule over synthetic persistent
    trends and FIT realized Pi vs realized favorable move M (Inv L's TEST text): the asymmetric
    schedule yields a CONVEX (option-like) fit — positive Pi(M) curvature, a giveback FRACTION that
    DECREASES in M, and a left tail bounded below by -R0 — while a SYMMETRIC constant-fraction
    baseline is AFFINE. A bug in ``trail_multiple``/``update`` (e.g. a non-monotone or flat schedule)
    changes the realized trail level and breaks the convex fit, so this catches it (unlike a
    hand-written array)."""
    from convexity.trend.policy.convex_exit import ConvexExit

    # ---- (A) realized (M, Pi) from the REAL trail schedule over persistent-trend paths ----
    ce = ConvexExit()
    builder = TrendLifeEngine(cfg).builder
    m_obs: list[float] = []
    pi_obs: list[float] = []
    for seed in range(1, 13):
        bars, e = persistent_trend_low_noise(n=400, seed=seed)
        pos = make_position(bars, e)
        builder.reset()
        vol0 = builder.vol.update(bars, e, pos, cfg)
        st = ce.initial(bars, pos, vol0, cfg)
        r0 = st.initial_risk
        entry = pos.entry_price
        for i in range(e, len(bars)):
            belief = builder.update(bars, i, pos, cfg)
            if i > e:
                st = ce.update(bars, i, pos, belief, st, cfg)  # ACTUAL schedule under test
            m_r = (st.high_watermark - entry) / r0           # realized favorable move, R0 units
            pi_r = (st.trail_level - entry) / r0             # realized round-trip Pi if stopped now
            if m_r > 1.0:  # the loose/tail region where INV-CVX claims convexity (past the R0 floor)
                m_obs.append(m_r)
                pi_obs.append(pi_r)
    m_arr = np.array(m_obs)
    pi_arr = np.array(pi_obs)
    assert m_arr.size > 500 and m_arr.max() > 5.0  # a real tail was ridden

    # CONVEX fit: positive quadratic curvature in Pi(M), and a giveback FRACTION decreasing in M
    # (the marginal giveback g'(M) falls as the trail multiple widens, §10 / INV-CVX).
    curvature = float(np.polyfit(m_arr, pi_arr, 2)[0])
    giveback_frac = (m_arr - pi_arr) / m_arr
    frac_slope = float(np.polyfit(m_arr, giveback_frac, 1)[0])
    assert curvature > 0.0                 # Pi(M) convex (option-like), from the real schedule
    assert frac_slope < 0.0                # giveback fraction shrinks as M grows (convexity)
    # bounded-below by -R0: the trail floors the stop at the initial risk, so the left-tail cut is
    # never worse than -R0 (Pi >= -1 in R0 units), by construction.
    assert pi_arr.min() >= -1.0 - 1e-9

    # the schedule itself is monotone non-decreasing in profit (a bug here would flatten the fit).
    sched = [ce.trail_multiple(p, cfg) for p in np.linspace(0.0, 8.0, 50)]
    assert all(b >= a - 1e-12 for a, b in zip(sched, sched[1:]))

    # ---- (B) SYMMETRIC constant-fraction baseline is AFFINE (Inv L contrast) ----
    a = 0.4
    m_sym = np.linspace(1.0, 8.0, 60)
    pi_sym = (1 - a) * m_sym  # constant giveback fraction -> Pi affine in M
    assert np.allclose(np.diff(pi_sym, 2), 0.0, atol=1e-9)               # affine: zero curvature
    assert np.allclose(np.diff(a * m_sym) / np.diff(m_sym), a)           # constant marginal giveback


# --- Invariant M: convexity is gated NET of cost, not gross -------------------------------
def test_inv_m_net_of_cost_gate(cfg, spec):
    """A regime that is convex GROSS (positive skew/MFE capture) but whose premium bleed exceeds the
    right-tail capture must be REJECTED by the net-EV gate (Inv M / §1.5c FAIL case). Costs are
    sourced from the REAL COST pillar (cost_api), NOT the local CostModel stub (§6.1)."""
    from convexity.trend.validate.labels import TrendLifeLabeler

    from tests.trend.conftest import cost_hold, make_trend_cost_source

    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    ds = TrendLifeEngine(cfg).run_batch(bars, pos, spec, cfg)

    # §6.1 / Inv M: friction comes from COST.md (cost_api net_ev_gate + hold_carry), NOT the stub.
    src = make_trend_cost_source(market_kind="calm")
    cost_log = src.round_trip_log(pos.side, float(pos.initial_risk), cost_hold())
    assert cost_log > 0.0  # a real COST round-trip charge is being subtracted

    labeler = TrendLifeLabeler()
    # cost_log is supplied (real COST wiring); the local stub is NOT consulted on this path.
    labels = labeler.label(bars, pos, ds, None, {"calendar_days": cfg.tau_cap},  # type: ignore[arg-type]
                           ("calendar_days",), cost_log=cost_log)
    gross = labeler.net_convexity(labels, cost_per_loss_r=0.0)
    assert "E_Pi" in gross and "premium_bleed" in gross  # bleed reported alongside capture (§13)

    # the §1.5c arithmetic gate, exercised directly so the FAIL branch is unambiguous.
    def e_pi(p_win, avg_win, avg_loss):
        return p_win * avg_win - (1 - p_win) * avg_loss

    assert e_pi(0.35, 3.2, 1.15) > 0.0    # PASS gross+net
    assert e_pi(0.35, 2.6, 1.60) < 0.0    # FAIL: premium bleed wins -> gate rejects
    # the labeler's gate flag must reject the net-negative case.
    assert labeler.net_convexity(
        [{"net_pi": -0.13}], cost_per_loss_r=0.0
    )["net_convex_pass"] is False


def test_inv_m_promotion_rejects_gross_positive_net_negative(cfg, spec):
    """GENUINE Inv-M reject: a regime that is convex GROSS (positive skew, right-tail capture, beats
    baselines) but whose COST-sourced premium bleed exceeds the right-tail capture (gross-positive,
    NET-negative) must be REJECTED by ``promotion_gate`` with the Inv-M reason (§13 / §1.5c FAIL,
    third worked line). The gross labels come from the REAL engine over a population of clean choppy
    cuts plus a few clipped persistent winners; the bleed comes from the REAL COST pillar (cost_api
    funding-spike market), NOT the local stub."""
    import math

    from convexity.trend.synth.generate import choppy_mean_reverting
    from convexity.trend.tune.promotion import promotion_gate
    from convexity.trend.validate.labels import TrendLifeLabeler

    from tests.trend.conftest import cost_hold, make_position, make_trend_cost_source

    labeler = TrendLifeLabeler()
    # short operational cap so the persistent winners are CAP-CLIPPED (§13a.6) to a few R0 rather
    # than running to a huge tail — the §1.5c "avg_win erodes" lever that makes the gross edge
    # marginal enough for the premium bleed to flip it.
    clip_cfg = resolve_config(spec, h_sys=0.02)

    def r0_log(pos):  # the initial-risk cut R0 in side-adjusted log-PnL units.
        return math.log(pos.entry_price / (pos.entry_price - pos.initial_risk))

    # build a multi-trade POPULATION (one terminal R0-unit label per trade) from the real engine.
    terminals: list[dict] = []
    r0_ref = 0.02
    for seed in range(1, 7):  # many clean choppy cuts: ~50% small losers/winners (positive skew)
        b, _ = choppy_mean_reverting(seed=seed)
        for entry in (0, 80, 160, 240, 320):
            p = make_position(b, entry)
            d = TrendLifeEngine(clip_cfg).run_batch(b, p, spec, clip_cfg)
            lbl = labeler.label(b, p, d, None, {"calendar_days": clip_cfg.tau_cap},  # type: ignore[arg-type]
                                ("calendar_days",), cost_log=0.0)[-1]
            terminals.append({**lbl, "net_pi": lbl["net_pi"] / r0_log(p)})  # R0 units (§1.5c)
    for seed in (1, 2):  # a few clipped persistent winners -> tips the GROSS edge positive
        b, _ = persistent_trend_low_noise(seed=seed)
        for entry in (0, 150):
            p = make_position(b, entry)
            r0_ref = r0_log(p)
            d = TrendLifeEngine(clip_cfg).run_batch(b, p, spec, clip_cfg)
            lbl = labeler.label(b, p, d, None, {"calendar_days": clip_cfg.tau_cap},  # type: ignore[arg-type]
                                ("calendar_days",), cost_log=0.0)[-1]
            terminals.append({**lbl, "net_pi": lbl["net_pi"] / r0_log(p)})

    # GROSS: positive net expectancy + right-tail capture (the convex gross diagnostics pass).
    gross = labeler.net_convexity_population(terminals, cost_per_loss_r=0.0)
    gross_tail_capture = gross["p_win"] * gross["avg_win"]
    assert gross["E_Pi"] > 0.0          # convex GROSS (positive skew)
    assert gross_tail_capture > 0.0     # a real captured right tail

    # NET: charge the COST-sourced funding-spike round-trip (in R0 units) on every cut.
    src = make_trend_cost_source(market_kind="spike")
    cost_per_loss_r = src.cost_per_loss_r(1, r0_ref, cost_hold())
    assert cost_per_loss_r > 0.0  # a real COST charge from cost_api, not the stub
    net = labeler.net_convexity_population(terminals, cost_per_loss_r=cost_per_loss_r)
    e_pi_net = net["E_Pi"]
    premium_bleed = net["premium_bleed"]
    net_tail_capture = net["p_win"] * net["avg_win"]
    assert e_pi_net <= 0.0                      # gross-positive, NET-negative (§1.5c FAIL)
    assert premium_bleed > net_tail_capture     # bleed exceeds right-tail capture (the Inv-M FAIL)

    # drive promotion_gate with the gross-positive baselines/convex flags but the net-negative E[Pi];
    # Inv M must REJECT despite beating baselines and a convex gross fit.
    decision = promotion_gate(
        asset_class="crypto_perp",
        symbol="BTCUSDT",
        engine_version=cfg.defaults.engine_version,
        resolved_config_hash="deadbeef",
        fit_window=(0, 200),
        validation_window=(200, 400),
        e_pi_net=e_pi_net,
        premium_bleed=premium_bleed,
        tail_capture=net_tail_capture,
        beats_baselines=True,        # GROSS diagnostics pass
        convex_fit_ok=True,          # convex gross fit passes
        parity_ok=True,
        leakage_ok=True,
        n_events=len(terminals),
        min_events=len(terminals),   # ensure the only failing gate is Inv-M, not n_events
    )
    assert decision.status == "rejected"
    assert "Inv M" in decision.reason and "net-negative" in decision.reason
    assert decision.report_summary["e_pi_net"] == e_pi_net


# --- Invariant N: TREND owns exit level/intent; EXECUTION owns placement ------------------
def test_inv_n_no_order_placement_in_policy_engine():
    """No policy/ or engine.py code calls an order-placement API (Inv N)."""
    forbidden = ("place_order", "submit_order", "amend_order", "cancel_order", "OrderCommand")
    files = list((SRC / "policy").glob("*.py")) + [SRC / "engine.py"]
    for path in files:
        txt = path.read_text()
        for f in forbidden:
            assert f not in txt, f"{path.name} references order API {f!r}"


def test_inv_n_intent_not_placement(cfg, spec):
    """Hard exit -> reduce_only IMMEDIATE intent; continue/trail move -> amend_stop intent (Inv N)."""
    bars, e = jump_against()
    pos = make_position(bars, e)
    ds = TrendLifeEngine(cfg).run_batch(bars, pos, spec, cfg)
    assert isinstance(ds[-1].execution_intent, ExecutionIntent)
    assert ds[-1].execution_intent.kind == "exit_reduce_only"
    assert ds[-1].execution_intent.urgency == "immediate"
    assert ds[-1].execution_intent.reduce_only
    cont = next(d for d in ds if d.action == "continue")
    assert cont.execution_intent.kind == "amend_stop"


# --- Invariant O: entry/invalidation from BREAKOUT, never recomputed ----------------------
def test_inv_o_entry_from_breakout(cfg, spec):
    """ConvexExit.initial() returns BREAKOUT's initial_invalidation UNCHANGED when supplied; the
    vol/swing fallback fires only when it is absent (Inv O / §6c)."""
    bars, e = persistent_trend_low_noise()
    ce = ConvexExit()
    inval = float(bars.c[e]) * 0.97
    pos = Position(side=1, entry_idx=e, entry_price=float(bars.c[e]),
                   initial_invalidation=inval, initial_risk=float(bars.c[e]) - inval)
    vol0 = TrendLifeEngine(cfg).builder.vol.update(bars, e, pos, cfg)
    es = ce.initial(bars, pos, vol0, cfg)
    assert es.invalidation == pytest.approx(inval)  # used unchanged, no fallback

    pos2 = Position(side=1, entry_idx=e, entry_price=float(bars.c[e]))  # no invalidation
    es2 = ce.initial(bars, pos2, vol0, cfg)
    assert es2.invalidation != inval  # fallback path fired


# --- Invariant P: TREND publishes hold-horizon + residual-move outputs --------------------
def test_inv_p_hold_horizon_and_residual_move(cfg, spec):
    bars, e = persistent_trend_low_noise()
    pos = make_position(bars, e)
    ds = TrendLifeEngine(cfg).run_batch(bars, pos, spec, cfg)
    d = ds[len(ds) // 2]
    hh = d.hold_horizon
    assert hh.expected_remaining_lo_ms <= hh.expected_remaining_hi_ms  # a RANGE, not a point
    assert hh.expected_remaining_lo_ms <= hh.max_hold_quantile_ms
    # op_cap_remaining = H_sys - elapsed (Inv P / §1.6a), within one-bar tolerance of the clock.
    elapsed_ms = int(d.horizon.elapsed_value * 86_400_000)  # calendar_days clock -> ms
    cap_ms = int(cfg.tau_cap * 86_400_000)
    assert abs(hh.op_cap_remaining_ms - (cap_ms - elapsed_ms)) <= 86_400_000
    rm = d.residual_move
    assert rm.expected_residual_move_lo <= rm.expected_residual_move_hi  # a band, not a point
    assert isinstance(d, ActionDecision)


# --- §18 volatility: integrated variance is monotone non-decreasing after entry -----------
def test_integrated_var_monotone(cfg, spec):
    for name, fn in ALL_PATHS.items():
        bars, e = fn()
        pos = make_position(bars, min(e, len(bars) - 2))
        eng = TrendLifeEngine(cfg)
        eng._reset(bars, pos, cfg)
        prev = -1.0
        for i in range(pos.entry_idx, min(pos.entry_idx + 80, len(bars))):
            b = eng.builder.update(bars, i, pos, cfg)
            iv = b.vol.integrated_var_since_entry
            assert np.isfinite(iv), name
            assert iv >= prev - 1e-12, name
            prev = iv


# --- §13a.8 / §18: survival never pools across clocks or asset classes ---------------------
def test_survival_no_pool():
    with pytest.raises(ValueError):
        assert_no_pooling(("calendar_days", "trading_sessions"), ("crypto_perp",))
    with pytest.raises(ValueError):
        assert_no_pooling(("calendar_days",), ("crypto_perp", "us_equity"))
    assert_no_pooling(("calendar_days",), ("crypto_perp",))  # single clock + class is fine


# --- §13a.3 D3: the reduce slope-flip is SIDE-adjusted (short side fully exercised) -------
def test_reduce_slope_flip_side_adjusted(cfg, spec):
    """§13a.3 D3: the slope-flip reduce triggers when ``filtered_slope`` crosses zero AGAINST the
    position (``side*slope < 0``), NOT merely when it is negative. For a SHORT (side=-1) a favorable
    FALLING price (negative slope) must NOT reduce; only a RISING price (positive slope, against the
    short) reduces. Builds a real BeliefState from the engine and flips only filtered_slope."""
    from convexity.trend.policy.reduce import ReducePolicy
    from convexity.trend.synth.generate import persistent_downtrend_low_noise

    bars, e = persistent_downtrend_low_noise()
    pos = make_position(bars, e, side=-1)
    builder = TrendLifeEngine(cfg).builder
    builder.reset()
    ce = ConvexExit()
    vol0 = builder.vol.update(bars, e, pos, cfg)
    st = ce.initial(bars, pos, vol0, cfg)
    idx = e + 20
    for i in range(e, idx + 1):
        belief = builder.update(bars, i, pos, cfg)
        if i > e:
            st = ce.update(bars, i, pos, belief, st, cfg)
    rp = ReducePolicy(ce)

    # falling price (negative slope) is FAVORABLE for a short -> NO slope_flip reduce.
    fav = dataclasses.replace(belief, filtered_slope=-0.001, hazard=0.0)
    _, reasons_fav_short = rp.fraction(fav, st, -1, cfg)
    assert "slope_flip" not in reasons_fav_short
    # rising price (positive slope) is AGAINST a short -> slope_flip reduce fires.
    adv = dataclasses.replace(belief, filtered_slope=+0.001, hazard=0.0)
    _, reasons_adv_short = rp.fraction(adv, st, -1, cfg)
    assert "slope_flip" in reasons_adv_short
    # mirror for a LONG: a negative slope is AGAINST the long -> fires; a positive slope does not.
    long_adverse = dataclasses.replace(belief, filtered_slope=-0.001, hazard=0.0)
    long_favorable = dataclasses.replace(belief, filtered_slope=+0.001, hazard=0.0)
    assert "slope_flip" in rp.fraction(long_adverse, st, +1, cfg)[1]
    assert "slope_flip" not in rp.fraction(long_favorable, st, +1, cfg)[1]


# --- Inv I (short side): batch/live parity + a full short decision stream ----------------
@pytest.mark.parametrize("fn", ["persistent_downtrend_low_noise", "jump_with_short"])
def test_short_side_parity_and_decisions(cfg, spec, fn):
    """The entire SHORT side (side=-1): the trail anchors ABOVE price and ratchets down, a favorable
    DOWN jump is a jump_with (never a spurious jump_against), and batch/live parity holds (Inv I)."""
    from convexity.trend.synth import generate as gen

    bars, e = getattr(gen, fn)()
    pos = make_position(bars, e, side=-1)
    ds = TrendLifeEngine(cfg).run_batch(bars, pos, spec, cfg)
    assert ParityReplay().check(bars, pos, spec, cfg)  # short-side batch == live (Inv I)
    # a favorable down move never trips jump_against for a short.
    if fn == "jump_with_short":
        assert "jump_against" not in {d.stop_cause for d in ds}
    # the short trail sits ABOVE the entry side: trail_level - low_watermark > 0 (anchored on lwm).
    mid = ds[len(ds) // 2]
    assert mid.exit_state.trail_level >= mid.exit_state.low_watermark


# --- §18: survival module is validation-only (not imported by estimate/ or policy/) -------
def test_survival_validation_only():
    for sub in ("estimate", "policy"):
        for path in (SRC / sub).glob("*.py"):
            txt = path.read_text()
            assert "validate.survival" not in txt
            assert "import survival" not in txt
