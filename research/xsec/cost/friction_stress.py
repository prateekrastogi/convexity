"""Friction-stress promotion-gate scenario + funding-ablation Sharpe (COST.md Sections 6.4, 8).

COST hands SYSTEM's promotion gate (docs/specs/SYSTEM.md Section 6, Gate 6) a STRESSED friction
scenario: the identical weekly settlement re-run at 1.5x the baseline round-trip friction rate
(COST.md Section 8.1) — never a re-fit of weights, ranks, or the lookback band under stress (a
gate that lets the signal "adapt" to higher costs would test a different, adapted strategy, not
the robustness of the one actually promoted; COST.md Section 10).

WHY SCALING THE RATE IS THE IDENTICAL MACHINERY. ``research/xsec/accounting.py::settle_week``'s
cost line is LINEAR in the friction rate (``cost = turnover * friction`` on the scalar path,
``cost = 0.5 * sum|dw_i| * friction_i`` on the per-symbol path), and
``round_trip_friction = 2 * (taker + half_spread + impact)`` is linear in its per-leg components.
Scaling each component by the stress multiplier therefore scales the assembled rate — and hence
the whole cost series — by exactly that multiplier, while ``w_target`` / ``gross_price`` /
``funding`` / ``turnover`` are untouched (Invariant STRESS, COST.md Section 8.2; the friction rate
never feeds weight formation, COST.md P2). ``friction_stress_run`` exploits this by re-running the
UNCHANGED engine (``research/xsec/engine.py::run`` -> ``settle_week``) with only the three per-leg
friction components of the config scaled.

Also home to ``net_sharpe_ex_funding`` (COST.md Section 6.4, Invariant MINOR): the funding
ablation SYSTEM's report consumes so a config that clears promotion only via funding income —
versus one that is friction-robust on price P&L alone — is visible, never silently absorbed.
``research/xsec/validate.py`` deliberately does NOT carry this helper (COST.md Section 3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from research.xsec.cpcv import path_distribution
from research.xsec.engine import XSecConfig, XSecResult, run
from research.xsec.metrics import compute

# ---------------------------------------------------------------------------------------------
# RED shared economic constants (COST.md Section 4, Section 8.1-8.2). Shared verbatim with
# SYSTEM's Gate 6 (docs/specs/SYSTEM.md Section 6.1) — one config surface, never a per-symbol
# override (Invariant C-cfg).
# ---------------------------------------------------------------------------------------------

FRICTION_STRESS_MULTIPLIER: float = 1.5  # COST.md Section 8.1: 1.5x baseline round-trip friction
STRESS_SHARPE_RETENTION: float = 0.5  # COST.md Section 8.2: stressed Sharpe >= 0.5x honest lower bound
STRESS_FRAC_POS_MIN: float = 0.75  # COST.md Section 8.2: stressed CPCV frac_pos >= 0.75


def net_sharpe_ex_funding(net_weekly: np.ndarray, funding: np.ndarray) -> float:
    """Annualized Sharpe of the weekly series with funding ZEROED OUT (COST.md Section 6.4).

    Recomputes Sharpe on ``net_weekly - funding`` (price P&L minus cost only) using the IDENTICAL
    annualization as ``research/xsec/metrics.py::compute`` (delegated to it, sqrt(52.18)/yr).
    Invariant MINOR: SYSTEM's promotion gate consumes this ALONGSIDE the funding-inclusive net
    Sharpe so funding income is reported, never the unstated reason a config passes — matching the
    finding that carry alone is DSR-failing and must not sneak in through the momentum book.
    """
    net = np.asarray(net_weekly, dtype=float)
    fund = np.asarray(funding, dtype=float)
    if net.shape != fund.shape:
        raise ValueError(f"net_weekly {net.shape} and funding {fund.shape} must be aligned")
    return compute(net - fund).sharpe_ann


def stressed_net(
    gross_price: np.ndarray,
    funding: np.ndarray,
    cost: np.ndarray,
    *,
    multiplier: float = FRICTION_STRESS_MULTIPLIER,
) -> np.ndarray:
    """Re-net the weekly series at ``multiplier`` x friction (COST.md Section 8.1).

    ``net_t^stress = gross_price_t + funding_t - multiplier * cost_t`` — exact, not an
    approximation, because ``settle_week``'s cost is linear in the friction rate (module
    docstring): re-netting the baseline decomposition IS the stressed settlement. Only the cost
    line moves; price P&L and funding are untouched by a friction stress (Invariant STRESS,
    COST.md P4's separation of concerns).
    """
    gp = np.asarray(gross_price, dtype=float)
    fund = np.asarray(funding, dtype=float)
    cst = np.asarray(cost, dtype=float)
    if not (gp.shape == fund.shape == cst.shape):
        raise ValueError(
            f"gross_price {gp.shape}, funding {fund.shape}, cost {cst.shape} must be aligned"
        )
    return gp + fund - multiplier * cst


@dataclass(frozen=True)
class FrictionStressResult:
    """Baseline vs stressed weekly series for one config (COST.md Section 8.1).

    ``baseline`` and ``stressed`` share byte-identical ``dates`` / ``gross_price`` / ``funding`` /
    ``turnover`` (Invariant STRESS); only ``cost`` (scaled by exactly ``multiplier``) and ``net``
    differ. The four Sharpes are the annualized net Sharpe and its funding-ablated companion
    (COST.md Section 6.4) for each run.
    """

    multiplier: float
    baseline: XSecResult
    stressed: XSecResult
    baseline_sharpe: float
    stressed_sharpe: float
    baseline_sharpe_ex_funding: float
    stressed_sharpe_ex_funding: float


def friction_stress_run(
    panel, cfg: XSecConfig, *, multiplier: float = FRICTION_STRESS_MULTIPLIER
) -> FrictionStressResult:
    """Run the baseline AND the ``multiplier``x-friction scenario through the IDENTICAL engine.

    The stressed config differs from ``cfg`` ONLY in the three per-leg friction components
    (``taker``/``half_spread``/``impact``, each scaled by ``multiplier``), which scales
    ``round_trip_friction = 2*(taker + half_spread + impact)`` by exactly ``multiplier`` (module
    docstring). Signal, quantile, weighting, lookback, and rebalance grid are NOT re-fit under
    stress (COST.md Section 8.1) — both runs consume the same ``panel`` object, so the weight path
    is identical by construction and the STRESS invariant is testable on the outputs.
    """
    if multiplier <= 0.0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    baseline = run(panel, cfg)
    stress_cfg = dataclasses.replace(
        cfg,
        taker=cfg.taker * multiplier,
        half_spread=cfg.half_spread * multiplier,
        impact=cfg.impact * multiplier,
    )
    stressed = run(panel, stress_cfg)
    return FrictionStressResult(
        multiplier=multiplier,
        baseline=baseline,
        stressed=stressed,
        baseline_sharpe=compute(baseline.net).sharpe_ann,
        stressed_sharpe=compute(stressed.net).sharpe_ann,
        baseline_sharpe_ex_funding=net_sharpe_ex_funding(baseline.net, baseline.funding),
        stressed_sharpe_ex_funding=net_sharpe_ex_funding(stressed.net, stressed.funding),
    )


def stressed_frac_pos(
    net_weekly: np.ndarray,
    *,
    n_groups: int = 8,
    k_test: int = 2,
    embargo: int = 1,
    seed: int = 0,
) -> float:
    """CPCV ``frac_pos`` of a (stressed) weekly net series on the SAME partition as the baseline.

    Delegates to ``research/xsec/cpcv.py::path_distribution`` with the recorded-baseline defaults
    (N_groups=8, k_test=2 -> 28 paths, 1-week embargo — COST.md Section 8.3). ``frac_pos`` is the
    gating statistic; the 5th-percentile path Sharpe is partition-fragile and is reported only,
    never a threshold (COST.md Section 8.3, SYSTEM Invariant CPCV-STAT).
    """
    _, summary = path_distribution(
        np.asarray(net_weekly, dtype=float),
        n_groups=n_groups,
        k_test=k_test,
        embargo=embargo,
        seed=seed,
    )
    return summary.frac_pos


@dataclass(frozen=True)
class StressGateResult:
    """The friction-stress sub-gate verdict SYSTEM's promotion conjunction consumes (Gate 6)."""

    passed: bool
    stressed_sharpe: float
    stressed_frac_pos: float
    sharpe_floor: float  # STRESS_SHARPE_RETENTION * baseline honest lower bound
    frac_pos_floor: float  # STRESS_FRAC_POS_MIN


def friction_stress_gate(
    *,
    baseline_sharpe_lower_bound: float,
    stressed_sharpe: float,
    stressed_frac_pos: float,
) -> StressGateResult:
    """PASS iff the stressed run stays economically meaningful (COST.md Section 8.2).

    Two sub-criteria, BOTH required: (i) stressed net Sharpe >= ``STRESS_SHARPE_RETENTION`` x the
    honest baseline's LOWER bound (the ~0.8-1.0 Amihud-cost lookback-band number, Invariant
    HONEST-SR — never the light-cost headline), and (ii) stressed ``frac_pos`` >=
    ``STRESS_FRAC_POS_MIN`` on the same CPCV partition as the baseline (Section 8.3). A config
    failing either is rejected at the friction-stress stage regardless of its baseline DSR/CPCV
    result.
    """
    sharpe_floor = STRESS_SHARPE_RETENTION * baseline_sharpe_lower_bound
    passed = (stressed_sharpe >= sharpe_floor) and (stressed_frac_pos >= STRESS_FRAC_POS_MIN)
    return StressGateResult(
        passed=passed,
        stressed_sharpe=stressed_sharpe,
        stressed_frac_pos=stressed_frac_pos,
        sharpe_floor=sharpe_floor,
        frac_pos_floor=STRESS_FRAC_POS_MIN,
    )
