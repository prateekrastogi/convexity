"""One test per named SYSTEM Invariant (SYSTEM §14).

The orchestrator invariants run against SCRIPTED module outputs (mock pillar adapters / the synth
builders) per the §15 composition gate — no real pillars. The validation invariants exercise the
deflation/ledger/firewall/budget machinery directly. Each test names its Invariant.

Invariants covered: CB, HI, IT, SC, FW, DSR, DSR-GUARD, DSR-MINTRL, DSR-MONO, DSR-ANN, SG, BL, CA,
CAD, PAR, P1, C, A, HC, XC, SR-SRC, RP, ATTR, CB-BUDGET.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from convexity.core.params import Bucket
from convexity.system.compose.budget import Candidate, priority_order
from convexity.system.compose.cadence import exceeds_deadband
from convexity.system.compose.incumbent import resolve_incumbent
from convexity.system.compose.iterate import Position, greedy_lot_fill
from convexity.system.config.defaults import SYSTEM_PARAM_REGISTRY, SystemDefaults
from convexity.system.config.resolve import resolve_config
from convexity.system.config.universe import INSTRUMENT_META_FIELDS, load_universe_meta
from convexity.system.nautilus.exec_stub import ExecStub
from convexity.system.synth.generate import (
    DSR_LEDGER_SHARPES_ANNUAL,
    make_book,
    make_candidate,
    make_correlated_ledger,
    make_open_position,
    make_proposed,
    make_trend_action,
    make_trial_records,
    persistent_trend_prices,
)
from convexity.system.system_api import decide_live
from convexity.system.validate.attribution import PillarAttribution
from convexity.system.validate.baselines import (
    OFF_SEMANTICS,
    SINGLE_PILLAR_ABLATIONS,
    coalition_subsets,
    random_matched_frequency,
)
from convexity.system.validate.budget import OvernightBudget
from convexity.system.validate.cache import EvidenceCache
from convexity.system.validate.deflated import DeflatedSharpe, effective_trials, sr_star0
from convexity.system.validate.labels import triple_barrier_label
from convexity.system.validate.promotion import ForwardGate
from convexity.system.validate.trials import TrialLedger
from convexity.system.validate.walkforward import PurgedWalkForward, system_embargo


def cfg():
    return resolve_config(load_universe_meta([]))


class _Cfg:
    defaults = SystemDefaults()


def deflate(**kw):
    base = dict(
        path_returns=np.zeros(108),
        trial_sharpes=DSR_LEDGER_SHARPES_ANNUAL,
        n_obs=108,
        skew=-0.4,
        kurt=6.0,
        n_trials_effective=4.0,
        cfg=_Cfg(),
        k_distinct=12,
    )
    base.update(kw)
    return DeflatedSharpe().deflate(**base)


# --------------------------------------------------------------------------- CB
def test_cb_composition_is_not_a_score_blob():
    """Invariant CB: scaling every breakout score by a positive constant changes no decision."""
    book = make_book()
    sel = make_proposed(("B", "C"))
    base = {"B": make_candidate("B", quality_score=1.0), "C": make_candidate("C", quality_score=1.0)}
    scaled = {"B": make_candidate("B", quality_score=1e6), "C": make_candidate("C", quality_score=5.0)}
    d_base = decide_live(book, sel, {}, base, cfg())
    d_scaled = decide_live(book, sel, {}, scaled, cfg())
    sig = lambda d: [(a.instrument, a.kind, a.target_units) for a in d.actions]  # noqa: E731
    assert sig(d_base) == sig(d_scaled)


# --------------------------------------------------------------------------- HI
def test_hi_incumbent_protected():
    """Invariant HI: a trend-alive, ceiling-slack incumbent demoted with no replacement HOLDs."""
    dec = resolve_incumbent(
        make_open_position("A"), make_trend_action("continue"),
        make_proposed(keep=("B",), eligible=("B",)),
        ceiling_collapsed=False, open_instruments=frozenset({"A"}),
        qualified=frozenset(), delta_rank=2.0, risk_improvement=0.0, round_trip_cost=1.0,
    )
    assert dec.verdict == "hold" and dec.reason == "hold_incumbent"


# --------------------------------------------------------------------------- IT
def test_it_bounded_feasible_oscillation_free():
    """Invariant IT: the A-drops/B-fits adversarial set terminates in ONE feasible, non-oscillating
    greedy fill; the emitted book never exceeds H_book."""
    positions = [
        Position("A", wpc=0.06, ceiling_weight=0.06, incumbent_lots=0, target_weight=0.06,
                 priority=(0.0, 0.0, "A")),
        Position("B", wpc=0.06, ceiling_weight=0.06, incumbent_lots=0, target_weight=0.06,
                 priority=(1.0, 0.0, "B")),
    ]
    book, _w, heat, deferred = greedy_lot_fill(positions, h_book=0.06, equity_ladder_cap=1e9)
    assert heat <= 0.06 + 1e-9
    assert sum(book.values()) == 1 and "B" in deferred  # one fits, the other defers (no oscillation)


# --------------------------------------------------------------------------- SC
def test_sc_no_cross_module_fold_leakage():
    """Invariant SC: purge+embargo leaves a gap before each test fold so no module's test label
    overlaps another module's training fit."""
    wf = PurgedWalkForward(SystemDefaults())
    for f in wf.folds(n_bars=400_000, n_folds=3):
        assert f.train_hi <= f.test_lo - wf.embargo_bars + 1
        assert f.test_lo < f.test_hi


# --------------------------------------------------------------------------- FW
def test_fw_live_packages_do_not_import_validate():
    """Invariant FW: nothing in modules/, compose/, nautilus/ imports from validate/ (AST scan)."""
    root = Path(__file__).resolve().parents[2] / "src" / "convexity" / "system"
    offenders = []
    for d in ("modules", "compose", "nautilus"):
        for py in (root / d).rglob("*.py"):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "system.validate" in node.module:
                    offenders.append((str(py), node.module))
                if isinstance(node, ast.Import):
                    offenders += [(str(py), n.name) for n in node.names if "system.validate" in n.name]
    assert offenders == []


# --------------------------------------------------------------------------- DSR
def test_dsr_nominal_product_never_feeds_sr_star0():
    """Invariant DSR: SR*0 is computed from the EFFECTIVE N_sys (1≤N_sys≤K), never the Cartesian
    product, and N_sys + Var[SR_trials] come from the same ledger."""
    returns = make_correlated_ledger(k=12, rho=0.4, n=300, seed=4)
    ledger = TrialLedger()
    ledger.extend(make_trial_records(DSR_LEDGER_SHARPES_ANNUAL, returns))
    n_sys, _ = ledger.n_sys(0.7)
    assert 1.0 <= n_sys <= ledger.k
    assert len(ledger.trial_sharpes()) == ledger.k  # same K configs feed Var[SR_trials]


# --------------------------------------------------------------------------- DSR-GUARD
def test_dsr_guard_insufficient_trials_and_var_floor():
    """Invariant DSR-GUARD: < K_min distinct trials cannot PASS; a degenerate ledger is floored so
    SR*0 > 0 still bites."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400), k_distinct=3)
    assert r.passed is False and "insufficient_trials" in r.reason_codes
    ident = deflate(trial_sharpes=np.full(12, 0.92), sr_annual=0.92, n_obs=108)
    assert ident.var_floored is True and ident.sr_star0 > 0.0


# --------------------------------------------------------------------------- DSR-GUARD (N_sys=1)
def test_dsr_n_sys_one_uses_zero_benchmark():
    """Invariant DSR / guard 2: N_sys=1 ⇒ SR*0=0, DSR=PSR(0), never Φ⁻¹(0)."""
    r = deflate(sr_annual=1.80, n_obs=400, path_returns=np.zeros(400), n_trials_effective=1.0)
    assert r.sr_star0 == 0.0
    assert np.isfinite(r.dsr) and r.dsr == pytest.approx(r.psr, abs=1e-12)


# --------------------------------------------------------------------------- DSR-MINTRL
def test_dsr_mintrl_gates_thin_samples():
    """Invariant DSR-MINTRL: a high point-DSR off a thin n_obs (< floor) does NOT pass."""
    r = deflate(sr_annual=3.0, n_obs=12, path_returns=np.zeros(12))
    assert r.n_obs_sufficient is False and r.passed is False
    assert "insufficient_track_record" in r.reason_codes


# --------------------------------------------------------------------------- DSR-MONO
def test_dsr_mono_non_decreasing_in_rho_cl():
    """Invariant DSR-MONO: N_sys is monotone non-decreasing as rho_cl tightens."""
    returns = make_correlated_ledger(k=12, rho=0.5, n=300, seed=3)
    prev = 0.0
    for rho_cl in (0.3, 0.5, 0.7, 0.9):
        n_sys, _ = effective_trials(returns, rho_cl)
        assert n_sys >= prev - 1e-9
        prev = n_sys


# --------------------------------------------------------------------------- DSR-ANN
def test_dsr_ann_rejects_annualized_input():
    """Invariant DSR-ANN: feeding the ANNUALIZED Sharpe into the per-observation z-test is rejected."""
    r = deflate(sr_annual=0.92, n_obs=108, assume_annualized_input=True)
    assert r.passed is False and "annualization_mismatch" in r.reason_codes


# --------------------------------------------------------------------------- SG
def test_sg_ledger_length_equals_distinct_config_hashes():
    """Invariant SG: ledger length == number of distinct config_hash scored (idempotent on hash)."""
    returns = make_correlated_ledger(k=12, rho=0.3, n=200, seed=1)
    records = make_trial_records(DSR_LEDGER_SHARPES_ANNUAL, returns)
    ledger = TrialLedger()
    ledger.extend(records)
    ledger.extend(records)
    assert ledger.k == 12


# --------------------------------------------------------------------------- BL
def test_bl_off_semantics_share_the_contract():
    """Invariant BL: the 4 pillar off-semantics + 5 single-pillar ablations are defined under the
    identical evaluation contract; the random-matched null has zero expected edge."""
    assert set(OFF_SEMANTICS) == {"selection", "breakout", "leverage", "trend"}
    assert OFF_SEMANTICS["breakout"].changes_module_evidence
    assert not OFF_SEMANTICS["selection"].changes_module_evidence
    assert len(SINGLE_PILLAR_ABLATIONS) == 5
    r = random_matched_frequency(n_obs=1000, n_trades=50, holding_dist=np.array([10]), seed=0, ret_scale=0.01)
    assert abs(float(np.mean(r))) < 0.01


# --------------------------------------------------------------------------- CA
def test_ca_combiner_only_change_hits_cache():
    """Invariant CA: a combiner-only config change re-scores module evidence from cache (100% hit)."""
    cache = EvidenceCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "evidence"

    cache.get_or_compute("breakout", "v1", "fold0", "hashA", (0, 100), compute)
    cache.get_or_compute("breakout", "v1", "fold0", "hashB", (0, 100), compute)
    assert calls["n"] == 1 and cache.hits == 1 and cache.misses == 1


# --------------------------------------------------------------------------- CAD
def test_cad_deadband_suppresses_jitter():
    """Invariant CAD: a sub-deadband target change generates no order; a real change does."""
    assert not exceeds_deadband(2, 3, 0.05, 0.051, deadband_lots=1, deadband_weight=0.02)
    assert exceeds_deadband(2, 5, 0.05, 0.12, deadband_lots=1, deadband_weight=0.02)


# --------------------------------------------------------------------------- PAR
def test_par_batch_live_parity():
    """Invariant PAR: the composed pipeline produces identical SystemDecisions in batch replay and
    one-bar streaming over the same bars (the composition is a pure function of (BookState, outputs))."""
    book = make_book({"A": make_open_position("A")})
    sel = make_proposed(("A", "B"))
    open_acts = {"A": make_trend_action("continue")}
    cands = {"B": make_candidate("B")}
    # "batch": one orchestrator instance replayed; "live": a fresh instance, one bar at a time.
    from convexity.system.compose.orchestrator import SystemOrchestrator

    batch = [decide_live(book, sel, open_acts, cands, cfg(), bar_ts=t,
                         orchestrator=SystemOrchestrator()) for t in range(3)]
    shared = SystemOrchestrator()
    live = [decide_live(book, sel, open_acts, cands, cfg(), bar_ts=t, orchestrator=shared)
            for t in range(3)]
    for b, ln in zip(batch, live):
        assert b.actions == ln.actions and b.reason_codes == ln.reason_codes


# --------------------------------------------------------------------------- P1
def test_p1_compose_holds_no_sizing_exit_entry_math():
    """Invariant P1: compose/ + modules/ contain no sizing/exit/entry estimators — AST scan for the
    forbidden estimator call names that would mean a pillar was reimplemented."""
    root = Path(__file__).resolve().parents[2] / "src" / "convexity" / "system"
    forbidden_names = {"imply_vol", "ewma_vol", "kelly_fraction", "garch", "weibull", "kalman"}
    offenders = []
    for d in ("compose", "modules"):
        for py in (root / d).rglob("*.py"):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in forbidden_names:
                        offenders.append((str(py), node.func.id))
    assert offenders == []


# --------------------------------------------------------------------------- C
def test_c_red_unreachable_from_per_instrument_config():
    """Invariant C: no RED/BLUE param name appears on the AMBER per-instrument metadata."""
    forbidden = {p.name for p in SYSTEM_PARAM_REGISTRY.params() if p.bucket in (Bucket.RED, Bucket.BLUE)}
    assert not (forbidden & INSTRUMENT_META_FIELDS)


# --------------------------------------------------------------------------- A
def test_a_budget_priority_is_risk_not_edge():
    """Invariant A: admission order is ceiling-slack → MCTR-efficiency → id; Candidate has no edge
    field (the firewall is the ABSENCE of the field)."""
    cands = [
        Candidate("Z", ceiling_slack=0.1, risk_weight_delivered=0.05, heat_consumed=0.05),
        Candidate("A", ceiling_slack=0.9, risk_weight_delivered=0.05, heat_consumed=0.05),
        Candidate("M", ceiling_slack=0.9, risk_weight_delivered=0.10, heat_consumed=0.05),
    ]
    order = [c.instrument for c in priority_order(cands, budget_priority="slack_then_mctr")]
    assert order == ["M", "A", "Z"]
    for forbidden in ("breakout_score", "expected_return", "trend_confidence", "edge"):
        assert forbidden not in Candidate.__dataclass_fields__


# --------------------------------------------------------------------------- HC
def test_hc_horizon_cap_bounds_every_label():
    """Invariant HC: the triple-barrier vertical = H_sys; across a persistent trend every label
    force-closes at or before H_sys (time_cap); the embargo equals H_sys + max warmup."""
    prices = persistent_trend_prices(n=30_000, seed=2)
    d = SystemDefaults()
    label = triple_barrier_label(prices, entry_idx=0, side=1, sigma_horizon=10.0,
                                 b_up=d.b_up, b_dn=d.b_dn, h_sys=d.h_sys)
    assert label.horizon_bars <= d.h_sys and label.outcome == "time_cap"
    assert system_embargo(d) == max(d.h_sys, max(d.module_label_horizons)) + d.max_warmup_bars
    assert system_embargo(d) >= d.h_sys  # keyed to the cap, never a mean/typical horizon


# --------------------------------------------------------------------------- XC
def test_xc_exec_stub_is_conservative_upper_bound():
    """Invariant XC: stub cost ≥ a mid/half-spread reference on every fill; full spread, not half."""
    stub = ExecStub(SystemDefaults())
    for spread in (0.0005, 0.001, 0.002):
        cost = stub.fill_cost(spread_frac=spread, bar_range_frac=0.003, participation=0.1,
                              taker_fee_frac=0.0004, funding_frac=0.0001, n_funding_stamps=3)
        assert cost.total >= ExecStub.mid_reference_cost(spread, 0.0004)
    full = stub.fill_cost(spread_frac=0.001, bar_range_frac=0.0, participation=0.0,
                          taker_fee_frac=0.0, funding_frac=0.0, n_funding_stamps=0)
    assert full.spread == pytest.approx(0.001)


# --------------------------------------------------------------------------- SR-SRC
def test_sr_src_cross_config_variance_beats_cross_path():
    """Invariant SR-SRC: Var[SR_trials] is the cross-CONFIG ledger variance, NOT the cross-CPCV-path
    spread of one config — using the (smaller) cross-path variance collapses SR*0 and defeats the
    gate. We confirm the ledger variance is the dispersion across DISTINCT configs."""
    returns = make_correlated_ledger(k=12, rho=0.3, n=300, seed=4)
    ledger = TrialLedger()
    ledger.extend(make_trial_records(DSR_LEDGER_SHARPES_ANNUAL, returns))
    # cross-config variance over the K distinct configs' Sharpes:
    cross_config = float(np.var(ledger.trial_sharpes(), ddof=1))
    # the cross-path spread of a SINGLE config (one config replayed over CPCV paths) is much smaller:
    one_config_paths = np.full(8, float(DSR_LEDGER_SHARPES_ANNUAL[0]))  # one config ⇒ ~0 dispersion
    cross_path = float(np.var(one_config_paths, ddof=1))
    assert cross_config > cross_path
    # SR*0 built on the (correct) larger cross-config variance is a stronger benchmark.
    assert sr_star0(cross_config, 4.0) > sr_star0(cross_path, 4.0)


# --------------------------------------------------------------------------- RP
def test_rp_accumulation_and_paired_forward_gate():
    """Invariant RP: accumulating more nightly trials grows N_sys and raises SR*0; a positive POINT
    margin whose paired-difference CI includes zero does NOT swap."""
    small = make_correlated_ledger(k=4, rho=-0.5, n=400, seed=1)
    big = make_correlated_ledger(k=16, rho=-0.5, n=400, seed=1)
    n_small, _ = effective_trials(small, 0.7)
    n_big, _ = effective_trials(big, 0.7)
    assert n_big > n_small and sr_star0(0.005, n_big) > sr_star0(0.005, n_small)
    rng = np.random.default_rng(0)
    inc = rng.normal(0.0, 0.05, size=80)
    cand = inc + rng.normal(0.001, 0.05, size=80)  # tiny positive point margin, noisy
    res = ForwardGate().evaluate(cand, inc, cfg(), round_trip_cost=0.0, min_trl=24, seed=0)
    assert res.swap is False
    assert "paired_ci_includes_zero" in res.reason_codes or res.ci_low <= 0.0


# --------------------------------------------------------------------------- ATTR
def test_attr_shapley_requires_16_coalitions():
    """Invariant ATTR: mode='shapley' with < 16 coalitions is an error; with all 16 it computes the
    exact Shapley value and labels the report 'shapley'."""
    attr = PillarAttribution()
    with pytest.raises(ValueError):
        attr.attribute({frozenset(): 0.0}, "shapley", ("selection",))
    subsets = coalition_subsets()
    assert len(subsets) == 16
    val = {"selection": 1.0, "breakout": 2.0, "leverage": 3.0, "trend": 4.0}
    pnl = {s: sum(val[p] for p in s) for s in subsets}
    report = attr.attribute(pnl, "shapley", ("selection", "breakout", "leverage", "trend"))
    assert report.attribution_mode == "shapley"
    for p, v in val.items():
        assert report.pillar_contribution[p] == pytest.approx(v)


# --------------------------------------------------------------------------- CB-BUDGET
def test_cb_budget_pre_declared_hard_cap():
    """Invariant CB-BUDGET: a ledger of K_max+1 configs is rejected; the RED defaults fit T_night."""
    d = SystemDefaults()
    budget = OvernightBudget(d)
    assert budget.admits(k=d.k_max, distinct_module_configs=d.max_distinct_module_configs)
    assert not budget.admits(k=d.k_max + 1, distinct_module_configs=d.max_distinct_module_configs)
    est = budget.estimate(k=24, distinct_module_configs=8, attribution_mode="sequential")
    assert est.fits and est.total_s <= d.t_night_s
    assert est.cost_evidence_s >= est.cost_compose_s  # t_mod dominates (the binding term)
