"""The §6 PROMOTABLE six-gate conjunction (SYSTEM §6.1–6.2) — the single promotion boolean.

SYSTEM §6.2 formalized the promotion rule on paper but recorded that the codebase did not yet
implement it ("This section (§6) is therefore a FORMALIZATION of the promotion rule that the
codebase should implement, not a description of an already-implemented and merely unrun check")
and suggested a NEW module, ``src/convexity/system/validate/promotion_gate.py`` ("name TBD,
deliberately not claimed as existing"), to compose the already-existing pieces into that boolean.
This is that module.

It is kept SEPARATE from ``promotion.py`` on purpose: SYSTEM §11's directory layout assigns
``promotion.py`` the role "re-promotion cadence + forward gate (Section 8.2, Invariant RP)" — i.e.
the §7.5 :class:`~convexity.system.validate.promotion.ForwardGate`. The §6 conjunction is a
DIFFERENT thing (a promotion-time gate battery, not a swap-cadence forward test), so it lives in the
§6.2-named file rather than being appended below the §11-mandated ForwardGate.

Everything here COMPOSES already-existing statistics (``DeflatedSharpe`` → ``DSRResult``,
``TrialLedger.n_sys`` → ``n_true_distinct``, ``src/convexity/system/cpcv.py``'s
``PathSharpeSummary.frac_pos`` / ``compute_pbo``, COST's ``friction_stress`` outputs) into ONE typed
verdict; zero statistical machinery is reimplemented here (SYSTEM §6.2 reuse note).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

from convexity.system.config.tuning import PromotionGateDefaults
from convexity.system.types import DSRResult


@dataclass(frozen=True)
class PBOEvidence:
    """The Gate 4 input: PBO reported as a RANGE over pre-registered partitions, never a bare point.

    ``partition_pbos`` holds one ``(n_groups, k_test, pbo)`` triple per CPCV partition scored — the
    committed partition (SYSTEM §4.2's (8, 2)) plus BOTH pre-registered alternatives ((6, 2) and
    (10, 3), SYSTEM §4.4 / §12 edge case 9). The gate reads the UPPER bound of the range (Invariant
    PBO-RANGE, the conservative direction) and REJECTS a report whose partitions differ from the
    pre-registered list — partitions chosen after seeing which pairs are favorable defeat the
    range's purpose (§12 edge case 9's adversarial scenario). ``as_of`` is the ledger-snapshot
    scoring-cycle stamp, cross-checked against Gate 1's ledger timestamp (§12 edge cases 8+9).
    """

    partition_pbos: tuple[tuple[int, int, float], ...]
    as_of: int

    @property
    def upper(self) -> float:
        """The upper bound of the reported PBO range — the number Gate 4 reads (Inv PBO-RANGE)."""
        return max(p for _, _, p in self.partition_pbos)


@dataclass(frozen=True)
class CostStressEvidence:
    """The Gate 6 input — the COST pillar's friction-stress outputs (COST.md §8.1–8.2, SYSTEM §6.1).

    Produced by ``src/convexity/cost/friction/friction_stress.py`` (COST.md §3): the 1.5×-round-trip-friction
    re-run through the identical ``settle_week`` machinery, re-scored on the SAME CPCV partition as
    the baseline. ``baseline_sharpe_ann_lower`` is the honest baseline's lower bound (the ~0.8 end
    of the 0.8–1.0 net band, COST.md §8.2). ``net_sharpe_ex_funding`` is REPORTED, never gated
    (COST.md §6.4 Invariant MINOR — funding must never be the unstated reason a config passes).
    ``stress_multiplier`` records the scenario actually run; the gate rejects any value other than
    the shared RED 1.5× constant (a stress run at a softer multiplier must not clear Gate 6).
    """

    stressed_sharpe_ann: float
    baseline_sharpe_ann_lower: float
    stressed_frac_pos: float
    stress_multiplier: float = 1.5
    net_sharpe_ex_funding: float | None = None


@dataclass(frozen=True)
class PromotionCandidate:
    """Everything the §6.1 conjunction reads — one typed bundle of ALREADY-COMPUTED evidence.

    ``dsr`` is the :class:`~convexity.system.types.DSRResult` from ``DeflatedSharpe.deflate`` (via
    ``src/convexity/system/xsec_validate.py::deflated_sharpe``); ``n_true_distinct`` is the de-correlated
    cluster count — the INT (``k_distinct``) component of ``TrialLedger.n_sys(rho_cl)``'s
    ``tuple[float, int]`` return (trials.py:57), NOT the effective-float component — measured AT
    PROMOTION TIME on the ledger
    snapshot stamped ``ledger_as_of`` (§12 edge case 8: a stale report must never read as current);
    ``frac_pos`` is ``PathSharpeSummary.frac_pos`` from ``src/convexity/system/cpcv.py`` — ``p05`` is
    deliberately NOT a field of this contract (Invariant CPCV-STAT); ``regime_sharpes`` maps each
    of the five tested regimes (SYSTEM §6.1 Gate 5) to its annualized Sharpe; ``cost_stress`` is
    ``None`` while COST's friction-stress run is unavailable — Gate 6 then FAILS CLOSED (§6.2
    records Gate 6 as un-runnable without that code, and a conjunction must not pass on a gate it
    could not evaluate).
    """

    dsr: DSRResult
    n_true_distinct: int
    ledger_as_of: int
    frac_pos: float
    pbo: PBOEvidence
    regime_sharpes: Mapping[str, float]
    cost_stress: CostStressEvidence | None = None


@dataclass(frozen=True)
class GateResult:
    """One row of the §6.1 conjunction: which gate, pass/fail, WHY, and the number it read."""

    gate: int
    name: str
    passed: bool
    reason: str
    observed: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class PromotionReport:
    """The single PROMOTABLE boolean (SYSTEM §6.1) plus the full per-gate audit trail.

    ``promotable = GATE1 AND … AND GATE6`` — a strict conjunction, never a weighted score. Gate 1
    is always ``gates[0]`` (checked FIRST, before any DSR ranking is trusted — the n=2-cell
    spurious-DSR near-miss §6 documents). ``deployment_mode`` carries Gate 4's three-zone
    consequence: ``"band"`` (mandatory ensemble deployment, PBO upper in [0.2, 0.5)) or ``"point"``
    (clean pass, PBO upper < 0.2); ``None`` when not promotable or Gate 4 failed.
    ``ledger_as_of`` re-stamps ``n_true_distinct``'s snapshot (§12 edge case 8).
    """

    promotable: bool
    gates: tuple[GateResult, ...]
    deployment_mode: Literal["point", "band"] | None
    n_true_distinct: int
    ledger_as_of: int
    reason_codes: tuple[str, ...] = ()

    def failed_gates(self) -> tuple[int, ...]:
        return tuple(g.gate for g in self.gates if not g.passed)


class PromotionGate:
    """The §6.1 six-gate PROMOTABLE conjunction, as ONE function instead of scattered checks.

    Composes existing statistics only (SYSTEM §6.2's implementation-gap note): DSR/PSR/MinTRL from
    ``deflated.py``, ``n_true_distinct`` from ``trials.py``, ``frac_pos``/PBO from
    ``src/convexity/system/cpcv.py``, the stressed run from COST's ``friction_stress``. Nothing here
    recomputes a statistic; it only compares them against the RED thresholds
    (:class:`~convexity.system.config.tuning.PromotionGateDefaults`, SYSTEM §10).
    """

    def evaluate(
        self, candidate: PromotionCandidate, defaults: PromotionGateDefaults | None = None
    ) -> PromotionReport:
        """PROMOTABLE := GATE1 AND GATE2 AND GATE3 AND GATE4 AND GATE5 AND GATE6 (SYSTEM §6.1).

        A strict conjunction — a candidate failing even one gate is NOT promotable regardless of
        how strongly it clears the others. Every gate is evaluated (for the audit trail) but Gate 1
        is decided FIRST: a ledger below the trial floor is rejected regardless of how favorable
        its DSR looks, because ``Var[SR_trials]`` from a near-degenerate ledger makes ``SR*₀``
        itself untrustworthy (§6's documented n=2 near-miss).
        """
        g = defaults or PromotionGateDefaults()
        d = candidate.dsr
        reasons: list[str] = []

        # ---- GATE 1 — trial floor, checked FIRST, BEFORE any DSR ranking (§6.1) ----
        gate1 = GateResult(
            gate=1,
            name="trial_floor",
            passed=candidate.n_true_distinct >= g.k_min,
            reason="trial_floor_met" if candidate.n_true_distinct >= g.k_min
            else "trial_floor_below_k_min",
            observed=float(candidate.n_true_distinct),
            threshold=float(g.k_min),
        )
        if not gate1.passed:
            reasons.append("trial_floor_below_k_min")

        # ---- GATE 2 — edge, either branch (§6.1) ----
        if d.dsr >= g.dsr_threshold:
            gate2 = GateResult(
                gate=2, name="edge", passed=True, reason="dsr_clean_pass_2a",
                observed=d.dsr, threshold=g.dsr_threshold,
            )
        else:
            # Branch 2b: the shortfall must diagnose as insufficient_track_record — the SAME
            # diagnosis the deflation engine itself emits. deflated.py sets
            # ``n_obs_sufficient = n_obs >= max(MinTRL, n_obs_floor)`` and appends reason code
            # ``insufficient_track_record`` exactly when ``not n_obs_sufficient`` (deflated.py:256,
            # :263-265). The spec's §6.1 gloss "n_obs < MinTRL" is the INTENT; the engine's
            # ``n_obs_sufficient`` field is its authoritative implementation of that intent, so we
            # READ IT rather than re-derive the comparison — this also closes the divergence when
            # MinTRL < n_obs_floor, where a raw ``n_obs < MinTRL`` would disagree with the engine.
            # An infinite MinTRL (SR_pp <= SR*0: the edge cannot clear the benchmark at ANY length)
            # forces n_obs_sufficient=False and would spuriously read as "track too short"; we
            # explicitly EXCLUDE it here as a deflation failure, not a length one (§6.1 exclusion).
            track_short = math.isfinite(d.min_trl) and not d.n_obs_sufficient
            psr_high = d.psr >= g.psr_zero_threshold
            if psr_high and track_short:
                gate2 = GateResult(
                    gate=2, name="edge", passed=True, reason="track_length_branch_2b",
                    observed=d.psr, threshold=g.psr_zero_threshold,
                )
                reasons.append("gate2_conditional_pass_insufficient_track_record")
            elif not psr_high:
                gate2 = GateResult(
                    gate=2, name="edge", passed=False, reason="psr_zero_below_threshold",
                    observed=d.psr, threshold=g.psr_zero_threshold,
                )
            else:
                # n_obs >= MinTRL (or MinTRL infinite) and DSR still short: more data would not
                # have saved it — fails Gate 2 outright (§6.1's branch-2b exclusion).
                gate2 = GateResult(
                    gate=2, name="edge", passed=False, reason="dsr_shortfall_not_track_length",
                    observed=d.dsr, threshold=g.dsr_threshold,
                )
            if not gate2.passed:
                reasons.append(gate2.reason)

        # ---- GATE 3 — CPCV robustness: frac_pos, NEVER p05 (§6.1 / Invariant CPCV-STAT) ----
        gate3 = GateResult(
            gate=3,
            name="cpcv_robustness",
            passed=candidate.frac_pos >= g.frac_pos_threshold,
            reason="frac_pos_met" if candidate.frac_pos >= g.frac_pos_threshold
            else "frac_pos_below_threshold",
            observed=candidate.frac_pos,
            threshold=g.frac_pos_threshold,
        )
        if not gate3.passed:
            reasons.append("frac_pos_below_threshold")

        # ---- GATE 4 — overfitting: PBO range upper bound (§6.1 / Inv PBO-RANGE / §12 cases 8+9) --
        gate4, deployment_mode = self._gate4(candidate, g)
        if not gate4.passed:
            reasons.append(gate4.reason)
        elif deployment_mode == "band":
            reasons.append("mandatory_band_deployment")

        # ---- GATE 5 — regime breadth: positive Sharpe in >= 4 of 5 tested regimes (§6.1) ----
        n_regimes = len(candidate.regime_sharpes)
        n_pos = sum(1 for v in candidate.regime_sharpes.values() if v > 0.0)
        if n_regimes != g.regime_total:
            gate5 = GateResult(
                gate=5, name="regime_breadth", passed=False, reason="regime_count_mismatch",
                observed=float(n_regimes), threshold=float(g.regime_total),
            )
        else:
            gate5 = GateResult(
                gate=5,
                name="regime_breadth",
                passed=n_pos >= g.regime_pos_required,
                reason="regime_breadth_met" if n_pos >= g.regime_pos_required
                else "regime_breadth_below_required",
                observed=float(n_pos),
                threshold=float(g.regime_pos_required),
            )
        if not gate5.passed:
            reasons.append(gate5.reason)

        # ---- GATE 6 — cost stress: the COST.md §8 shared RED constants (§6.1) ----
        gate6 = self._gate6(candidate.cost_stress, g)
        if not gate6.passed:
            reasons.append(gate6.reason)

        gates = (gate1, gate2, gate3, gate4, gate5, gate6)
        promotable = all(x.passed for x in gates)
        return PromotionReport(
            promotable=promotable,
            gates=gates,
            deployment_mode=deployment_mode if (promotable and gate4.passed) else None,
            n_true_distinct=candidate.n_true_distinct,
            ledger_as_of=candidate.ledger_as_of,
            reason_codes=tuple(reasons),
        )

    @staticmethod
    def _gate4(
        candidate: PromotionCandidate, g: PromotionGateDefaults
    ) -> tuple[GateResult, Literal["point", "band"] | None]:
        """Gate 4 with the PBO-RANGE schema validation and the §12 case-9 cross-checks.

        Rejections (all FAIL the gate, fail-closed): (a) partitions differing from the
        pre-registered ``{committed} ∪ {alternatives}`` set — a cherry-picked partition list
        satisfies PBO-RANGE's letter while defeating its spirit; (b) a PBO snapshot stamped more
        than one scoring cycle away from Gate 1's ledger snapshot — internally inconsistent
        regardless of what either number says (§12 edge case 9's audit cross-check).
        """
        pbo = candidate.pbo
        provided = {(n, k) for n, k, _ in pbo.partition_pbos}
        required = {g.committed_partition, *g.preregistered_alternatives}
        if provided != required:
            return (
                GateResult(gate=4, name="overfitting", passed=False,
                           reason="pbo_report_schema_rejected"),
                None,
            )
        if abs(pbo.as_of - candidate.ledger_as_of) > g.max_snapshot_skew:
            return (
                GateResult(gate=4, name="overfitting", passed=False,
                           reason="ledger_pbo_snapshot_inconsistent"),
                None,
            )
        upper = pbo.upper
        if upper >= g.pbo_threshold:
            return (
                GateResult(gate=4, name="overfitting", passed=False,
                           reason="pbo_upper_bound_at_or_above_threshold",
                           observed=upper, threshold=g.pbo_threshold),
                None,
            )
        mode: Literal["point", "band"] = "band" if upper >= g.pbo_band_floor else "point"
        return (
            GateResult(gate=4, name="overfitting", passed=True,
                       reason="pbo_band_zone" if mode == "band" else "pbo_clean_pass",
                       observed=upper, threshold=g.pbo_threshold),
            mode,
        )

    @staticmethod
    def _gate6(evidence: CostStressEvidence | None, g: PromotionGateDefaults) -> GateResult:
        """Gate 6: stressed Sharpe >= 0.5× honest baseline lower bound AND stressed frac_pos >= 0.75.

        Fails CLOSED when the COST friction-stress run is unavailable (SYSTEM §6.2 records Gate 6
        as blocked on ``src/convexity/cost/friction/friction_stress.py``), and rejects a stress run at any
        multiplier other than the shared RED 1.5× (COST.md §8.1 / SYSTEM §6.1). The multiplier
        match uses :func:`math.isclose` rather than ``==``: the canonical 1.5 literal is exactly
        representable so the RED-constant intent (reject anything but a true 1.5× run) is preserved,
        but a caller who arrives at the multiplier via non-exact arithmetic (e.g. ``3.0 / 2.0`` in
        a different unit) is not spuriously rejected by a last-bit float discrepancy.
        """
        if evidence is None:
            return GateResult(gate=6, name="cost_stress", passed=False,
                              reason="cost_stress_unavailable")
        if not math.isclose(evidence.stress_multiplier, g.stress_multiplier,
                            rel_tol=1e-9, abs_tol=1e-12):
            return GateResult(gate=6, name="cost_stress", passed=False,
                              reason="stress_multiplier_mismatch",
                              observed=evidence.stress_multiplier,
                              threshold=g.stress_multiplier)
        retention_floor = g.stress_retention * evidence.baseline_sharpe_ann_lower
        if evidence.stressed_sharpe_ann < retention_floor:
            return GateResult(gate=6, name="cost_stress", passed=False,
                              reason="stressed_sharpe_below_retention",
                              observed=evidence.stressed_sharpe_ann, threshold=retention_floor)
        if evidence.stressed_frac_pos < g.stressed_frac_pos_floor:
            return GateResult(gate=6, name="cost_stress", passed=False,
                              reason="stressed_frac_pos_below_floor",
                              observed=evidence.stressed_frac_pos,
                              threshold=g.stressed_frac_pos_floor)
        return GateResult(gate=6, name="cost_stress", passed=True, reason="cost_stress_survived",
                          observed=evidence.stressed_sharpe_ann, threshold=retention_floor)
