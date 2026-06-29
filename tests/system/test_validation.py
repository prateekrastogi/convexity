"""Validation-layer invariant tests (SYSTEM §6-§9): SC/HC, FW, XC, SG/RP, CA, ATTR, BL, CB-BUDGET."""

import ast
from pathlib import Path

import numpy as np
import pytest

from convexity.system.config.defaults import SystemDefaults, system_embargo_bars
from convexity.system.config.resolve import resolve_config
from convexity.system.config.universe import load_universe_meta
from convexity.system.nautilus.exec_stub import ExecStub
from convexity.system.synth.generate import (
    DSR_LEDGER_SHARPES_ANNUAL,
    make_correlated_ledger,
    make_trial_records,
    persistent_trend_prices,
)
from convexity.system.validate.attribution import PillarAttribution
from convexity.system.validate.baselines import (
    OFF_SEMANTICS,
    SINGLE_PILLAR_ABLATIONS,
    coalition_subsets,
    random_matched_frequency,
)
from convexity.system.validate.budget import OvernightBudget
from convexity.system.validate.cache import EvidenceCache
from convexity.system.validate.labels import triple_barrier_label
from convexity.system.validate.promotion import ForwardGate
from convexity.system.validate.trials import TrialLedger
from convexity.system.validate.walkforward import PurgedWalkForward, system_embargo


def cfg():
    return resolve_config(load_universe_meta([]))


# --------------------------------------------------------------------------- causality / embargo (SC/HC)


def test_embargo_is_union_keyed_to_h_sys():
    """Invariant SC/HC: embargo = max(H_sys, max module label horizon) + max warmup."""
    d = SystemDefaults()
    expected = max(d.h_sys, max(d.module_label_horizons)) + d.max_warmup_bars
    assert system_embargo_bars(d) == expected
    assert system_embargo(d) == expected
    # keyed to the contractual cap, never a mean/typical horizon.
    assert system_embargo_bars(d) >= d.h_sys


def test_walkforward_folds_are_time_ordered_with_embargo_gap():
    wf = PurgedWalkForward(SystemDefaults())
    folds = wf.folds(n_bars=200_000, n_folds=3)
    for f in folds:
        assert f.train_hi <= f.test_lo - wf.embargo_bars + 1  # embargo gap before test
        assert f.test_lo < f.test_hi


def test_h_sys_bounds_every_label_horizon():
    """Invariant HC: across a persistent trend, every label closes at or before H_sys (time_cap)."""
    prices = persistent_trend_prices(n=30_000, seed=2)
    d = SystemDefaults()
    label = triple_barrier_label(
        prices, entry_idx=0, side=1, sigma_horizon=10.0,  # huge σ ⇒ horizontal barriers never hit
        b_up=d.b_up, b_dn=d.b_dn, h_sys=d.h_sys,
    )
    assert label.horizon_bars <= d.h_sys
    assert label.outcome == "time_cap"  # force-closed at the cap, not death


# --------------------------------------------------------------------------- firewall (FW)


def test_live_packages_do_not_import_validate():
    """Invariant FW: nothing in modules/, compose/, nautilus/ imports from validate/."""
    root = Path(__file__).resolve().parents[2] / "src" / "convexity" / "system"
    live_dirs = ["modules", "compose", "nautilus"]
    offenders = []
    for d in live_dirs:
        for py in (root / d).rglob("*.py"):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and "system.validate" in node.module:
                    offenders.append((str(py), node.module))
                if isinstance(node, ast.Import):
                    for n in node.names:
                        if "system.validate" in n.name:
                            offenders.append((str(py), n.name))
    assert offenders == [], f"live package imports validate/: {offenders}"


# --------------------------------------------------------------------------- exec stub (XC)


def test_exec_stub_is_conservative_upper_bound():
    """Invariant XC: stub cost ≥ a mid/half-spread reference on every fill."""
    stub = ExecStub(SystemDefaults())
    for spread in (0.0005, 0.001, 0.002):
        cost = stub.fill_cost(
            spread_frac=spread, bar_range_frac=0.003, participation=0.1,
            taker_fee_frac=0.0004, funding_frac=0.0001, n_funding_stamps=3,
        )
        ref = ExecStub.mid_reference_cost(spread, 0.0004)
        assert cost.total >= ref


def test_exec_stub_charges_full_spread_not_half():
    stub = ExecStub(SystemDefaults())
    cost = stub.fill_cost(
        spread_frac=0.001, bar_range_frac=0.0, participation=0.0,
        taker_fee_frac=0.0, funding_frac=0.0, n_funding_stamps=0,
    )
    assert cost.spread == pytest.approx(0.001)  # full, not 0.0005


# --------------------------------------------------------------------------- trial ledger (SG/RP)


def test_ledger_length_equals_distinct_config_hashes():
    """Invariant SG: ledger length == number of distinct config_hash scored."""
    returns = make_correlated_ledger(k=12, rho=0.3, n=200, seed=1)
    records = make_trial_records(DSR_LEDGER_SHARPES_ANNUAL, returns)
    ledger = TrialLedger()
    ledger.extend(records)
    ledger.extend(records)  # appending the same configs is idempotent on config_hash
    assert ledger.k == 12


def test_n_sys_and_var_from_same_ledger():
    """Invariant DSR/SG: N_sys and Var[SR_trials] reference the SAME ledger config set."""
    returns = make_correlated_ledger(k=12, rho=0.4, n=300, seed=4)
    records = make_trial_records(DSR_LEDGER_SHARPES_ANNUAL, returns)
    ledger = TrialLedger()
    ledger.extend(records)
    n_sys, k_distinct = ledger.n_sys(0.7)
    assert 1.0 <= n_sys <= ledger.k
    assert len(ledger.trial_sharpes()) == ledger.k  # same K configs feed the variance


def test_repeated_searches_accumulate_and_raise_sr_star0():
    """Invariant RP: accumulating each night's configs grows the ledger and (via more trials) the
    deflation benchmark. More distinct trials ⇒ larger N_sys ⇒ larger SR*0."""
    from convexity.system.validate.deflated import sr_star0

    small = make_correlated_ledger(k=4, rho=-0.5, n=400, seed=1)
    big = make_correlated_ledger(k=16, rho=-0.5, n=400, seed=1)
    n_small, _ = __import__("convexity.system.validate.deflated", fromlist=["effective_trials"]).effective_trials(small, 0.7)
    n_big, _ = __import__("convexity.system.validate.deflated", fromlist=["effective_trials"]).effective_trials(big, 0.7)
    assert n_big > n_small
    v = 0.005
    assert sr_star0(v, n_big) > sr_star0(v, n_small)


# --------------------------------------------------------------------------- forward gate (RP)


def test_paired_forward_gate_point_margin_with_ci_through_zero_does_not_swap():
    """Invariant RP: a positive POINT margin whose paired-difference CI includes zero does NOT swap."""
    rng = np.random.default_rng(0)
    # candidate beats incumbent on the mean but with huge variance ⇒ CI spans zero.
    inc = rng.normal(0.0, 0.05, size=80)
    cand = inc + rng.normal(0.001, 0.05, size=80)  # tiny positive point margin, noisy
    res = ForwardGate().evaluate(cand, inc, cfg(), round_trip_cost=0.0, min_trl=24, seed=0)
    assert res.diff_sharpe != 0.0
    assert res.swap is False
    assert "paired_ci_includes_zero" in res.reason_codes or res.ci_low <= 0.0


def test_forward_gate_defers_on_thin_block():
    """Invariant RP: a forward block too thin for n_obs ≥ max(MinTRL, n_obs_floor) defers."""
    cand = np.array([0.01, 0.02, 0.01])
    inc = np.array([0.0, 0.0, 0.0])
    res = ForwardGate().evaluate(cand, inc, cfg(), round_trip_cost=0.0, min_trl=100, seed=0)
    assert res.swap is False
    assert "insufficient_forward_track" in res.reason_codes


def test_forward_gate_requires_paired_lengths():
    with pytest.raises(ValueError):
        ForwardGate().evaluate(np.zeros(5), np.zeros(6), cfg())


# --------------------------------------------------------------------------- attribution (ATTR)


def test_shapley_requires_16_coalitions():
    """Invariant ATTR: mode='shapley' with < 16 coalitions is an error."""
    attr = PillarAttribution()
    with pytest.raises(ValueError):
        attr.attribute({frozenset(): 0.0}, "shapley", ("selection",))


def test_shapley_over_16_coalitions():
    subsets = coalition_subsets()
    assert len(subsets) == 16
    # additive game v(S) = sum of per-pillar constants ⇒ Shapley = those constants.
    val = {"selection": 1.0, "breakout": 2.0, "leverage": 3.0, "trend": 4.0}
    pnl = {s: sum(val[p] for p in s) for s in subsets}
    report = PillarAttribution().attribute(pnl, "shapley", ("selection", "breakout", "leverage", "trend"))
    assert report.attribution_mode == "shapley"
    for p, v in val.items():
        assert report.pillar_contribution[p] == pytest.approx(v)


def test_sequential_is_order_dependent_and_labeled():
    subsets = coalition_subsets()
    val = {"selection": 1.0, "breakout": 2.0, "leverage": 3.0, "trend": 4.0}
    pnl = {s: sum(val[p] for p in s) for s in subsets}
    order = ("leverage", "trend", "selection", "breakout")
    report = PillarAttribution().attribute(pnl, "sequential", order)
    assert report.attribution_mode == "sequential_fixed_order"
    assert report.fixed_order == order


# --------------------------------------------------------------------------- baselines (BL)


def test_random_matched_frequency_has_zero_edge():
    """Invariant BL / §8.3: the random matched-frequency null has zero expected edge."""
    r = random_matched_frequency(n_obs=1000, n_trades=50, holding_dist=np.array([10]), seed=0, ret_scale=0.01)
    assert abs(float(np.mean(r))) < 0.01  # zero-mean by construction


def test_off_semantics_and_ablations_defined():
    """§8.2/§8.3: the 4 pillar off-semantics + 5 single-pillar ablations are defined."""
    assert set(OFF_SEMANTICS) == {"selection", "breakout", "leverage", "trend"}
    assert OFF_SEMANTICS["breakout"].changes_module_evidence
    assert not OFF_SEMANTICS["selection"].changes_module_evidence
    assert len(SINGLE_PILLAR_ABLATIONS) == 5


# --------------------------------------------------------------------------- cache (CA)


def test_combiner_only_change_hits_module_evidence_cache():
    """Invariant CA: a combiner-only config change re-scores module evidence from cache (100% hit)."""
    cache = EvidenceCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "evidence"

    # first config_hash: miss
    cache.get_or_compute("breakout", "v1", "fold0", "hashA", (0, 100), compute)
    # different config_hash (combiner-only change) but same module/version/fold/bar_range: HIT
    cache.get_or_compute("breakout", "v1", "fold0", "hashB", (0, 100), compute)
    assert calls["n"] == 1  # module evidence computed once
    assert cache.hits == 1 and cache.misses == 1


# --------------------------------------------------------------------------- overnight budget (CB-BUDGET)


def test_overnight_budget_rejects_over_k_max():
    """Invariant CB-BUDGET: a ledger of K_max + 1 configs is rejected."""
    d = SystemDefaults()
    budget = OvernightBudget(d)
    assert budget.admits(k=d.k_max, distinct_module_configs=d.max_distinct_module_configs)
    assert not budget.admits(k=d.k_max + 1, distinct_module_configs=d.max_distinct_module_configs)


def test_overnight_budget_cost_model_fits_t_night():
    """§9.2: the RED defaults fit T_night on the target VM (the illustrative ≈2.9 h estimate)."""
    d = SystemDefaults()
    est = OvernightBudget(d).estimate(k=24, distinct_module_configs=8, attribution_mode="sequential")
    assert est.fits
    assert est.total_s <= d.t_night_s
    # cost_evidence is the binding term (t_mod dominates).
    assert est.cost_evidence_s >= est.cost_compose_s
