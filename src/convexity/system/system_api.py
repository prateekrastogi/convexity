"""SYSTEM public entry points — one decide() for live, one for batch (SYSTEM §10/§12).

  * :func:`decide_live`     — one composed decision per bar for the live loop (the §2 pipeline + §3
                              fixed point), reading the canonical :class:`BookState`. Pure function of
                              F_t and BookState; emits deltas, not raw orders (Invariant P1).
  * :func:`validate_batch`  — the overnight integrated-validation harness: score the composed chain
                              on the CPCV path distribution, deflate over the EFFECTIVE accumulated
                              ledger N_sys, run robustness + attribution + the forward gate, assemble
                              the :class:`SystemReport` (headline = DSR | n_obs vs MinTRL).

These are thin orchestration wrappers over ``compose/`` and ``validate/``; the heavy lifting lives in
those modules. ``validate_batch`` honors the overnight budget cap (Invariant CB-BUDGET) before
scoring.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from convexity.system.compose.orchestrator import CandidateSignal, SystemOrchestrator
from convexity.system.config.resolve import ResolvedConfig
from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import BookState, ProposedPortfolio, SystemDecision, SystemReport
from convexity.system.validate.budget import OvernightBudget
from convexity.system.validate.deflated import DeflatedSharpe
from convexity.system.validate.reports import build_report
from convexity.system.validate.trials import TrialLedger


def decide_live(
    book: BookState,
    sel: ProposedPortfolio,
    open_actions: Mapping[str, TrendAction],
    candidates: Mapping[str, CandidateSignal],
    cfg: ResolvedConfig,
    *,
    collapsed: frozenset[str] = frozenset(),
    risk_improvements: Mapping[str, float] | None = None,
    round_trip_costs: Mapping[str, float] | None = None,
    bar_ts: int = 0,
    orchestrator: SystemOrchestrator | None = None,
) -> SystemDecision:
    """Produce ONE target book delta for the live loop (SYSTEM §12). Deterministic; deltas not orders."""
    orch = orchestrator or SystemOrchestrator()
    return orch.decide(
        book,
        sel,
        open_actions,
        candidates,
        cfg,
        collapsed=collapsed,
        risk_improvements=risk_improvements,
        round_trip_costs=round_trip_costs,
        bar_ts=bar_ts,
    )


def validate_batch(
    ledger: TrialLedger,
    promoted_returns: np.ndarray,
    *,
    skew: float,
    kurt: float,
    cfg: ResolvedConfig,
    n_sys_nominal: int,
    promoted_sharpe_annual: float | None = None,
    distinct_module_configs: int | None = None,
    beats_random_matched: bool = True,
    cpcv_path_sharpes: tuple[float, ...] = (),
    overnight_wall_clock_s: float = 0.0,
) -> SystemReport:
    """Score + deflate the composed chain over the ACCUMULATED ledger (SYSTEM §7/§10).

    Refuses to score more than ``K_max`` configs (Invariant CB-BUDGET). Computes the EFFECTIVE
    ``N_sys`` and the cross-config ``Var[SR_trials]`` from the SAME ledger (the consistency law),
    runs the DSR gate (per-observation units; all singularity + MinTRL guards), and assembles the
    report with the (DSR | n_obs vs MinTRL) headline.
    """
    defaults = cfg.defaults
    budget = OvernightBudget(defaults)
    distinct = distinct_module_configs if distinct_module_configs is not None else ledger.k
    if not budget.admits(k=ledger.k, distinct_module_configs=distinct):
        raise ValueError(
            f"overnight ledger {ledger.k} configs exceeds the K_max/T_night cap (Invariant CB-BUDGET)"
        )

    n_sys, k_distinct = ledger.n_sys(defaults.rho_cl)
    dsr = DeflatedSharpe().deflate(
        path_returns=np.asarray(promoted_returns, dtype=float),
        trial_sharpes=ledger.trial_sharpes(),
        n_obs=int(len(promoted_returns)),
        skew=skew,
        kurt=kurt,
        n_trials_effective=n_sys,
        cfg=cfg,
        sr_annual=promoted_sharpe_annual,
        k_distinct=k_distinct,
    )
    return build_report(
        dsr,
        n_sys_nominal=n_sys_nominal,
        k_ledger=ledger.k,
        rho_cl=defaults.rho_cl,
        cpcv_path_sharpes=cpcv_path_sharpes,
        overnight_wall_clock_s=overnight_wall_clock_s,
        beats_random_matched=beats_random_matched,
    )
