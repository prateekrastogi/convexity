"""SystemReport assembly — headline = (DSR | n_obs vs MinTRL) (SYSTEM §11/§7).

Pull the deflation result, the CPCV path distribution, the robustness suite, the attribution, the
baselines, and the forward gate into one :class:`SystemReport`. The HEADLINE is always
``(DSR | n_obs vs MinTRL)``, never the vanilla Sharpe and never the DSR alone (Invariant
DSR/DSR-MINTRL). ``passed`` is the conjunction of the DSR gate AND beating random-matched AND
K_distinct ≥ K_min AND n_obs ≥ max(MinTRL, n_obs_floor) (Invariant DSR-MINTRL).
"""

from __future__ import annotations

from typing import Mapping

from convexity.system.types import DSRResult, SystemReport


def build_report(
    dsr: DSRResult,
    *,
    n_sys_nominal: int,
    k_ledger: int,
    rho_cl: float,
    walkforward: Mapping[str, float] | None = None,
    cpcv_path_sharpes: tuple[float, ...] = (),
    drop_top_n: Mapping[str, float] | None = None,
    bootstrap_ci: Mapping[str, tuple[float, float]] | None = None,
    attribution_mode: str = "sequential_fixed_order",
    pillar_contribution: Mapping[str, float] | None = None,
    baselines: Mapping[str, float] | None = None,
    regime_attribution: Mapping[str, float] | None = None,
    forward_gate: Mapping[str, float] | None = None,
    overnight_wall_clock_s: float = 0.0,
    beats_random_matched: bool = True,
) -> SystemReport:
    """Assemble the :class:`SystemReport` (Section 11). ``passed`` requires the DSR gate AND beating
    random-matched AND the K_min / MinTRL guards (all already folded into ``dsr.passed`` except the
    random-matched comparison, which is conjoined here, Invariant DSR/BL)."""
    passed = dsr.passed and beats_random_matched
    reasons = list(dsr.reason_codes)
    if not beats_random_matched:
        reasons.append("does_not_beat_random_matched")
    return SystemReport(
        psr=dsr.psr,
        dsr=dsr.dsr,
        sr_star0=dsr.sr_star0,
        n_sys=dsr.n_sys,
        n_sys_nominal=n_sys_nominal,
        k_ledger=k_ledger,
        k_distinct=dsr.k_distinct,
        var_trials=dsr.var_trials,
        var_floored=dsr.var_floored,
        n_obs=dsr.n_obs,
        ppy=dsr.ppy,
        min_trl=dsr.min_trl,
        n_obs_sufficient=dsr.n_obs_sufficient,
        rho_cl=rho_cl,
        walkforward=dict(walkforward or {}),
        cpcv_path_sharpes=cpcv_path_sharpes,
        drop_top_n=dict(drop_top_n or {}),
        bootstrap_ci=dict(bootstrap_ci or {}),
        attribution_mode=attribution_mode,  # type: ignore[arg-type]
        pillar_contribution=dict(pillar_contribution or {}),  # type: ignore[arg-type]
        baselines=dict(baselines or {}),
        regime_attribution=dict(regime_attribution or {}),
        forward_gate=dict(forward_gate) if forward_gate is not None else None,
        overnight_wall_clock_s=overnight_wall_clock_s,
        passed=passed,
        reason_codes=tuple(reasons),
    )
