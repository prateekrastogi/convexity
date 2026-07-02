"""The §6 PROMOTABLE conjunction: each gate individually flips the verdict (SYSTEM §6.1–6.2).

Covers: the strict-AND semantics (every gate is an independent necessary condition), the historical
n=2-cell spurious-DSR regression (Gate 1 checked FIRST — §6's documented near-miss), both Gate 2
branches (2a clean DSR pass / 2b track-length), Invariant CPCV-STAT (p05 is structurally not a
gating input), Invariant PBO-RANGE + §12 edge cases 8–9 (pre-registered partitions, snapshot
cross-consistency), the Gate 4 three-zone deployment-mode consequence, and Gate 6's fail-closed
behavior while COST's friction-stress module is unavailable.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np

from convexity.system.config.tuning import (
    COMMITTED_CPCV_PARTITION,
    PREREGISTERED_PBO_PARTITIONS,
    PromotionGateDefaults,
)
from convexity.system.types import DSRResult
from convexity.system.validate.promotion_gate import (
    CostStressEvidence,
    PBOEvidence,
    PromotionCandidate,
    PromotionGate,
    PromotionReport,
)

from convexity.system.xsec_validate import deflated_sharpe, weekly_defaults

REGIMES_ALL_POSITIVE = {
    "2021_bull": 1.2,
    "2022_bear": 0.4,
    "2023_recovery": 0.8,
    "2024_bull": 1.0,
    "2025_26": 0.6,
}


def _dsr_result(
    *,
    dsr: float = 0.96,
    psr: float = 0.994,
    n_obs: int = 258,
    min_trl: float = 120.0,
    n_obs_sufficient: bool | None = None,
) -> DSRResult:
    """A synthetic DSRResult with only the Gate-2-relevant fields varied (SYSTEM §5.3 shapes).

    ``n_obs_sufficient`` defaults to the ENGINE's own formula ``n_obs >= max(min_trl, n_obs_floor)``
    with ``n_obs_floor=24`` (deflated.py:255-256, research/xsec/validate.py::weekly_defaults), so
    the synthetic result matches what ``DeflatedSharpe.deflate`` would produce; the override exists
    only to pin the ``MinTRL < n_obs_floor`` divergence seam that the raw ``n_obs < MinTRL`` gloss
    would get WRONG (see test_gate2_branch_2b_reads_engine_sufficiency_not_raw_min_trl).
    """
    if n_obs_sufficient is None:
        n_obs_sufficient = n_obs >= max(min_trl, weekly_defaults().n_obs_floor)
    return DSRResult(
        psr=psr,
        dsr=dsr,
        sr_star0=0.053,
        sr_pp=0.1246,
        var_trials=0.0039,
        var_floored=True,
        n_sys=3.0,
        n_obs=n_obs,
        ppy=52.18,
        min_trl=min_trl,
        n_obs_sufficient=n_obs_sufficient,
        k_distinct=5,
        psr_denom_floored=False,
        passed=dsr >= 0.95,
        reason_codes=(),
    )


def _pbo(*, pbos: tuple[float, float, float] = (0.36, 0.31, 0.47), as_of: int = 10) -> PBOEvidence:
    """Committed (8,2) + the two pre-registered alternatives (SYSTEM §4.4 / §12 edge case 9)."""
    (cn, ck) = COMMITTED_CPCV_PARTITION
    (a1, a2) = PREREGISTERED_PBO_PARTITIONS
    return PBOEvidence(
        partition_pbos=(
            (cn, ck, pbos[0]),
            (a1[0], a1[1], pbos[1]),
            (a2[0], a2[1], pbos[2]),
        ),
        as_of=as_of,
    )


def _stress(
    *,
    stressed_sharpe: float = 0.55,
    baseline_lower: float = 0.8,
    stressed_frac_pos: float = 0.82,
    multiplier: float = 1.5,
) -> CostStressEvidence:
    return CostStressEvidence(
        stressed_sharpe_ann=stressed_sharpe,
        baseline_sharpe_ann_lower=baseline_lower,
        stressed_frac_pos=stressed_frac_pos,
        stress_multiplier=multiplier,
    )


def _candidate(**overrides) -> PromotionCandidate:
    """The all-gates-pass baseline candidate (validated-band-like numbers, SYSTEM §6.2 table)."""
    base = dict(
        dsr=_dsr_result(),
        n_true_distinct=5,
        ledger_as_of=10,
        frac_pos=0.9,
        pbo=_pbo(),
        regime_sharpes=dict(REGIMES_ALL_POSITIVE),
        cost_stress=_stress(),
    )
    base.update(overrides)
    return PromotionCandidate(**base)


def _run(candidate: PromotionCandidate) -> PromotionReport:
    return PromotionGate().evaluate(candidate)


# --------------------------------------------------------------------------- the conjunction


def test_all_gates_pass_is_promotable():
    report = _run(_candidate())
    assert report.promotable
    assert len(report.gates) == 6
    assert all(g.passed for g in report.gates)
    assert tuple(g.gate for g in report.gates) == (1, 2, 3, 4, 5, 6)
    # PBO upper 0.47 sits in [0.2, 0.5): pass, but band deployment is MANDATORY (§4.4 three-zone).
    assert report.deployment_mode == "band"
    assert "mandatory_band_deployment" in report.reason_codes


def test_each_gate_individually_flips_the_conjunction():
    """Strict AND (§6.1): breaking any single gate makes the candidate non-promotable."""
    breakers = {
        1: dict(n_true_distinct=2),
        2: dict(dsr=_dsr_result(dsr=0.80, psr=0.90)),
        3: dict(frac_pos=0.7),
        4: dict(pbo=_pbo(pbos=(0.36, 0.31, 0.55))),
        5: dict(regime_sharpes={**REGIMES_ALL_POSITIVE, "2022_bear": -0.4, "2025_26": -0.1}),
        6: dict(cost_stress=_stress(stressed_sharpe=0.2)),
    }
    for gate_no, overrides in breakers.items():
        report = _run(_candidate(**overrides))
        assert not report.promotable, f"gate {gate_no} breaker did not flip the conjunction"
        assert report.failed_gates() == (gate_no,)
        assert report.deployment_mode is None


# --------------------------------------------------------------------------- Gate 1


def test_gate1_two_config_ledger_fails_regardless_of_stellar_dsr():
    """The documented n=2-cell spurious-DSR near-miss (§6): a 2-config ledger MUST fail Gate 1."""
    report = _run(_candidate(n_true_distinct=2, dsr=_dsr_result(dsr=0.99, psr=0.999)))
    assert not report.promotable
    gate1 = report.gates[0]
    assert gate1.gate == 1  # checked FIRST, before any DSR ranking (§6.1)
    assert not gate1.passed
    assert gate1.reason == "trial_floor_below_k_min"
    # Gate 2 itself looked stellar — that must not rescue the candidate.
    assert report.gates[1].passed


def test_gate1_floor_is_the_weekly_defaults_k_min_not_the_retracted_30():
    """k_min=3 is research/xsec/validate.py::weekly_defaults' floor (§6 constant reconciliation)."""
    g = PromotionGateDefaults()
    assert g.k_min == 3 == weekly_defaults().k_min
    assert g.k_min != 30  # the retracted, never-grounded draft constant
    report = _run(_candidate(n_true_distinct=3))
    assert report.gates[0].passed  # exact-boundary pass (3 >= 3, §6.2's tightened reading)


# --------------------------------------------------------------------------- Gate 2


def test_gate2_branch_2a_clean_dsr_pass():
    report = _run(_candidate(dsr=_dsr_result(dsr=0.96)))
    assert report.gates[1].passed
    assert report.gates[1].reason == "dsr_clean_pass_2a"


def test_gate2_branch_2b_track_length_conditional_pass():
    """DSR short of 0.95 but PSR(0) high and n_obs < MinTRL: the §5.5 honest-reading branch."""
    d = _dsr_result(dsr=0.807, psr=0.994, n_obs=258, min_trl=562.0)
    report = _run(_candidate(dsr=d))
    assert report.gates[1].passed
    assert report.gates[1].reason == "track_length_branch_2b"
    assert "gate2_conditional_pass_insufficient_track_record" in report.reason_codes
    assert report.promotable


def test_gate2_fails_when_track_is_long_enough_to_have_refuted_the_edge():
    """n_obs >= MinTRL with DSR still short: more data would not have saved it (§6.1)."""
    d = _dsr_result(dsr=0.90, psr=0.994, n_obs=600, min_trl=562.0)
    report = _run(_candidate(dsr=d))
    assert not report.gates[1].passed
    assert report.gates[1].reason == "dsr_shortfall_not_track_length"


def test_gate2_fails_on_low_psr_even_with_short_track():
    d = _dsr_result(dsr=0.80, psr=0.90, n_obs=100, min_trl=562.0)
    report = _run(_candidate(dsr=d))
    assert not report.gates[1].passed
    assert report.gates[1].reason == "psr_zero_below_threshold"


def test_gate2_infinite_min_trl_is_a_deflation_failure_not_a_length_one():
    """MinTRL = inf means SR_pp <= SR*0 — no track length can save it (deflated.py's convention)."""
    d = _dsr_result(dsr=0.60, psr=0.96, n_obs=100, min_trl=math.inf)
    report = _run(_candidate(dsr=d))
    assert not report.gates[1].passed
    assert report.gates[1].reason == "dsr_shortfall_not_track_length"


# --------------------------------------------------------------------------- Gate 3 (Inv CPCV-STAT)


def test_gate3_frac_pos_threshold():
    assert _run(_candidate(frac_pos=0.8)).gates[2].passed  # boundary: >= 0.8
    assert not _run(_candidate(frac_pos=0.79)).gates[2].passed


def test_p05_is_structurally_not_a_gating_input():
    """Invariant CPCV-STAT: no promotion input carries p05; the gate signature never reads it."""
    input_fields = {
        f.name
        for dc in (PromotionCandidate, PBOEvidence, CostStressEvidence)
        for f in dataclasses.fields(dc)
    }
    assert "p05" not in input_fields
    assert "p05" not in inspect.signature(PromotionGate.evaluate).parameters
    # Sweeping p05 across its observed [-0.59, +0.42] range cannot change the verdict: p05 has no
    # channel into the candidate at all, so two identical candidates give identical reports.
    assert _run(_candidate()).promotable == _run(_candidate()).promotable


# --------------------------------------------------------------------------- Gate 4 (Inv PBO-RANGE)


def test_gate4_reads_the_upper_bound_of_the_preregistered_range():
    # Upper bound 0.47 < 0.5 passes even though one alternative partition is close to the line.
    report = _run(_candidate(pbo=_pbo(pbos=(0.36, 0.31, 0.47))))
    g4 = report.gates[3]
    assert g4.passed and g4.observed == 0.47
    # A single partition at/above 0.5 fails on the UPPER bound (conservative direction).
    report = _run(_candidate(pbo=_pbo(pbos=(0.36, 0.31, 0.50))))
    assert not report.gates[3].passed
    assert report.gates[3].reason == "pbo_upper_bound_at_or_above_threshold"


def test_gate4_three_zone_deployment_mode():
    assert _run(_candidate(pbo=_pbo(pbos=(0.36, 0.31, 0.47)))).deployment_mode == "band"
    assert _run(_candidate(pbo=_pbo(pbos=(0.10, 0.05, 0.15)))).deployment_mode == "point"


def test_gate4_rejects_non_preregistered_partitions():
    """§12 edge case 9: cherry-picked partitions defeat PBO-RANGE's spirit — schema-rejected."""
    cherry = PBOEvidence(partition_pbos=((8, 2, 0.10), (7, 2, 0.11), (9, 2, 0.12)), as_of=10)
    report = _run(_candidate(pbo=cherry))
    assert not report.gates[3].passed
    assert report.gates[3].reason == "pbo_report_schema_rejected"


def test_gate4_rejects_missing_sensitivity_range():
    """Invariant PBO-RANGE: a bare point estimate (no alternatives) is rejected by the schema."""
    bare = PBOEvidence(partition_pbos=((8, 2, 0.36),), as_of=10)
    report = _run(_candidate(pbo=bare))
    assert not report.gates[3].passed
    assert report.gates[3].reason == "pbo_report_schema_rejected"


def test_gate4_rejects_stale_ledger_pbo_snapshot_pair():
    """§12 edge cases 8+9: n_true_distinct and PBO must share a ledger snapshot (±1 cycle)."""
    report = _run(_candidate(pbo=_pbo(as_of=8), ledger_as_of=10))
    assert not report.gates[3].passed
    assert report.gates[3].reason == "ledger_pbo_snapshot_inconsistent"
    # One scoring cycle of skew is tolerated.
    assert _run(_candidate(pbo=_pbo(as_of=9), ledger_as_of=10)).gates[3].passed


# --------------------------------------------------------------------------- Gate 5


def test_gate5_regime_breadth_four_of_five():
    four_of_five = {**REGIMES_ALL_POSITIVE, "2022_bear": -0.4}
    assert _run(_candidate(regime_sharpes=four_of_five)).gates[4].passed
    three_of_five = {**four_of_five, "2025_26": -0.1}
    report = _run(_candidate(regime_sharpes=three_of_five))
    assert not report.gates[4].passed
    assert report.gates[4].reason == "regime_breadth_below_required"


def test_gate5_rejects_wrong_regime_count():
    report = _run(_candidate(regime_sharpes={"2021_bull": 1.0, "2022_bear": 0.5}))
    assert not report.gates[4].passed
    assert report.gates[4].reason == "regime_count_mismatch"


# --------------------------------------------------------------------------- Gate 6


def test_gate6_fails_closed_without_cost_stress_evidence():
    """§6.2: Gate 6's dependency code may not exist yet — the conjunction must not pass without it."""
    report = _run(_candidate(cost_stress=None))
    assert not report.promotable
    assert report.gates[5].reason == "cost_stress_unavailable"


def test_gate6_retention_and_frac_pos_floors():
    # 0.5 * baseline_lower(0.8) = 0.4: stressed Sharpe 0.4 passes at the boundary, 0.39 fails.
    assert _run(_candidate(cost_stress=_stress(stressed_sharpe=0.4))).gates[5].passed
    report = _run(_candidate(cost_stress=_stress(stressed_sharpe=0.39)))
    assert report.gates[5].reason == "stressed_sharpe_below_retention"
    report = _run(_candidate(cost_stress=_stress(stressed_frac_pos=0.74)))
    assert report.gates[5].reason == "stressed_frac_pos_below_floor"
    assert _run(_candidate(cost_stress=_stress(stressed_frac_pos=0.75))).gates[5].passed


def test_gate6_rejects_softened_stress_multiplier():
    """A stress run at anything but the shared RED 1.5x must not clear Gate 6 (COST.md §8.1)."""
    report = _run(_candidate(cost_stress=_stress(multiplier=1.2)))
    assert not report.gates[5].passed
    assert report.gates[5].reason == "stress_multiplier_mismatch"


# --------------------------------------------------------------------------- integration


def test_gate2_consumes_a_real_dsr_result_from_the_reused_deflation_stack():
    """End-to-end Gate 2 against DeflatedSharpe's actual output (zero reimplementation, §6.2)."""
    rng = np.random.default_rng(7)
    net = 0.0025 + 0.012 * rng.standard_normal(258)  # ~5y of weekly obs, positive drift
    sr_ann = float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(52.18))
    trials = np.array([sr_ann, 0.3, 0.1, -0.2, 0.5])
    d = deflated_sharpe(net, trials, n_trials_effective=3.0, k_distinct=5)
    report = _run(_candidate(dsr=d))
    gate2 = report.gates[1]
    # The verdict must be exactly what the DSRResult's own fields dictate — no re-derivation.
    if d.dsr >= 0.95:
        assert gate2.passed and gate2.reason == "dsr_clean_pass_2a"
    elif d.psr >= 0.95 and math.isfinite(d.min_trl) and d.n_obs < d.min_trl:
        assert gate2.passed and gate2.reason == "track_length_branch_2b"
    else:
        assert not gate2.passed


def test_report_timestamps_the_ledger_snapshot():
    """§12 edge case 8: a promotion report must carry n_true_distinct's snapshot stamp."""
    report = _run(_candidate(ledger_as_of=42, pbo=_pbo(as_of=42)))
    assert report.ledger_as_of == 42
    assert report.n_true_distinct == 5


def test_gate2_branch_2b_reads_engine_sufficiency_not_raw_min_trl():
    """Gate 2 branch-2b delegates to the ENGINE's ``n_obs_sufficient`` field, not a raw
    ``n_obs < MinTRL`` re-derivation (§6.1 gloss vs deflated.py:256's authoritative diagnosis).

    Construct the divergence window ``MinTRL < n_obs < n_obs_floor``: with MinTRL=10 and
    n_obs_floor=24, an n_obs=15 candidate has ``n_obs >= MinTRL`` (raw gloss => "track long
    enough", which would FAIL 2b as dsr_shortfall_not_track_length) yet the engine computes
    ``n_obs_sufficient = 15 >= max(10, 24) = False`` and emits ``insufficient_track_record``.
    The gate MUST follow the engine: PSR high + not-sufficient + finite MinTRL => branch-2b PASS.
    """
    n_obs_floor = weekly_defaults().n_obs_floor
    min_trl = 10.0
    n_obs = 15  # MinTRL(10) <= n_obs(15) < n_obs_floor(24): the disagreement window
    assert min_trl < n_obs < n_obs_floor
    assert n_obs >= min_trl  # the RAW gloss says "long enough" — the wrong answer
    d = _dsr_result(dsr=0.80, psr=0.994, n_obs=n_obs, min_trl=min_trl)
    assert not d.n_obs_sufficient  # engine's authoritative diagnosis: track too short
    report = _run(_candidate(dsr=d))
    gate2 = report.gates[1]
    assert gate2.passed, "branch-2b must honor engine n_obs_sufficient, not raw n_obs < MinTRL"
    assert gate2.reason == "track_length_branch_2b"
    assert "gate2_conditional_pass_insufficient_track_record" in report.reason_codes
    # And the contrapositive: engine says sufficient (n_obs >= floor) with DSR still short => FAIL,
    # even though for a small MinTRL the raw ``n_obs < MinTRL`` gloss is also False here.
    d_suff = _dsr_result(dsr=0.80, psr=0.994, n_obs=30, min_trl=min_trl)
    assert d_suff.n_obs_sufficient
    report2 = _run(_candidate(dsr=d_suff))
    assert not report2.gates[1].passed
    assert report2.gates[1].reason == "dsr_shortfall_not_track_length"


def test_gate4_rejects_subset_of_preregistered_partitions_missing_one():
    """Invariant PBO-RANGE / §12 edge case 9: a report that provides a strict SUBSET of the
    pre-registered set — correct pairs, but FEWER than the pre-registered count (here the committed
    (8,2) + one alternative (6,2), MISSING (10,3)) — is schema-rejected by the set-inequality check
    (promotion.py:_gate4), not silently accepted just because the pairs it does report are valid.
    """
    (cn, ck) = COMMITTED_CPCV_PARTITION
    (a1, _a2) = PREREGISTERED_PBO_PARTITIONS
    subset = PBOEvidence(
        partition_pbos=((cn, ck, 0.30), (a1[0], a1[1], 0.31)),  # missing (10, 3)
        as_of=10,
    )
    provided = {(n, k) for n, k, _ in subset.partition_pbos}
    required = {COMMITTED_CPCV_PARTITION, *PREREGISTERED_PBO_PARTITIONS}
    assert provided < required  # STRICT subset: correct pairs, one short of the required set
    report = _run(_candidate(pbo=subset))
    assert not report.gates[3].passed
    assert report.gates[3].reason == "pbo_report_schema_rejected"
