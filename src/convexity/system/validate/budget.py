"""Overnight per-stage cost model + the K_max / T_night hard cap (SYSTEM §9.2, Inv CB-BUDGET).

The overnight job is NOT open-ended. On a 2-core / 11 GB ARM VM a naive plan (full Cartesian grid ×
full CPCV × 16 coalitions × multi-year 1-min bars) is days, not one night. We bound it with a
per-stage cost model, choose the CPCV knobs to fit, and cap the number of joint configs scored.

PER-STAGE COST (with the EvidenceCache, P7):
  cost_evidence = (#distinct module configs) · P · t_mod   — a combiner-only config adds ZERO here.
  cost_compose  = K · P · t_comp                            — paid per (config × path), always.
  cost_attrib   = {Mode B: 5, Mode A: 16} · P · (t_mod_off + t_comp)
  TOTAL ≈ cost_evidence + cost_compose + cost_attrib  ≤  T_night.

Because t_mod dominates and scales with B·Y·U, the binding term is almost always cost_evidence — the
cap is driven by DISTINCT MODULE configs × paths, NOT by K (combiner-only configs are nearly free).
The job refuses to enqueue more than ``K_max`` configs and records the measured wall-clock; if a run
exceeds ``T_night`` it aborts cleanly to the last promoted config (Invariant CB-BUDGET).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetEstimate:
    """The per-stage overnight cost estimate (Section 9.2). ``fits`` is True iff ``total_s ≤
    T_night``; ``within_cap`` iff ``k ≤ K_max`` and distinct module configs ≤ the module cap."""

    cost_evidence_s: float
    cost_compose_s: float
    cost_attrib_s: float
    total_s: float
    paths: int
    fits: bool
    within_cap: bool
    reason_codes: tuple[str, ...]


def _paths(n_groups: int, k_test: int, max_paths: int | None) -> int:
    """P = min(C(n_groups, k_test), max_paths) — the number of CPCV paths actually scored."""
    from math import comb

    full = comb(n_groups, k_test)
    return full if max_paths is None else min(full, max_paths)


class OvernightBudget:
    """The overnight per-stage cost model + hard cap (SYSTEM §9.2/§12, Invariant CB-BUDGET)."""

    def __init__(self, defaults) -> None:
        self._d = defaults

    def estimate(
        self,
        *,
        k: int,
        distinct_module_configs: int,
        attribution_mode: str = "sequential",
        t_mod_off_s: float | None = None,
    ) -> BudgetEstimate:
        """Estimate the overnight wall-clock for ``k`` joint configs (Section 9.2). Raises nothing;
        the caller checks ``within_cap``/``fits`` before enqueuing (the job refuses > K_max)."""
        d = self._d
        p = _paths(d.n_groups, d.k_test, d.max_paths)
        t_mod_off = t_mod_off_s if t_mod_off_s is not None else d.t_mod_s + d.t_comp_s

        cost_evidence = distinct_module_configs * p * d.t_mod_s
        cost_compose = k * p * d.t_comp_s
        attrib_coalitions = 16 if attribution_mode == "shapley" else 5
        cost_attrib = attrib_coalitions * p * t_mod_off

        total = cost_evidence + cost_compose + cost_attrib
        fits = total <= d.t_night_s
        within_cap = k <= d.k_max and distinct_module_configs <= d.max_distinct_module_configs

        reasons: list[str] = []
        if not within_cap:
            reasons.append("exceeds_k_max" if k > d.k_max else "exceeds_module_config_cap")
        if not fits:
            reasons.append("exceeds_t_night")
        if within_cap and fits:
            reasons.append("budget_ok")
        return BudgetEstimate(
            cost_evidence_s=cost_evidence,
            cost_compose_s=cost_compose,
            cost_attrib_s=cost_attrib,
            total_s=total,
            paths=p,
            fits=fits,
            within_cap=within_cap,
            reason_codes=tuple(reasons),
        )

    def admits(self, *, k: int, distinct_module_configs: int, attribution_mode: str = "sequential") -> bool:
        """True iff the job may enqueue ``k`` configs (within K_max AND fits T_night). A synthetic
        ledger of K_max + 1 configs is rejected (the Invariant-CB-BUDGET test)."""
        est = self.estimate(
            k=k, distinct_module_configs=distinct_module_configs, attribution_mode=attribution_mode
        )
        return est.within_cap and est.fits
