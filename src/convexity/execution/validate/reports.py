"""build_reports — assemble the execution validation report (§12, firewalled, Invariant V).

Collates the §12 products — the IS decomposition, the fill-prob calibration curve, the baselines, and
the tactic ablation — into one report object the OPS / SYSTEM layers consume read-only. Validation-
only; never imported by the live layers (Invariant V).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.validate.ablation import AblationResult
from convexity.execution.validate.baselines import BaselineCosts
from convexity.execution.validate.calibration import CalibrationReport
from convexity.execution.validate.shortfall import ShortfallReport

__all__ = ["ExecutionValidationReport", "build_report"]


@dataclass(frozen=True)
class ExecutionValidationReport:
    """The assembled §12 validation report (Invariant V)."""

    shortfall: ShortfallReport
    calibration: CalibrationReport
    baselines: BaselineCosts
    ablation: AblationResult
    beats_market_on_calm: bool
    loses_to_market_on_urgent: bool


def build_report(
    shortfall: ShortfallReport,
    calibration: CalibrationReport,
    baselines: BaselineCosts,
    ablation: AblationResult,
    *,
    is_calm: bool,
) -> ExecutionValidationReport:
    """Assemble the §12 report and the two acceptance flags (beats-market-on-calm / not-lose-urgent)."""
    beats_calm = is_calm and baselines.engine_argmin <= baselines.always_market
    # On urgent flow the engine must not LOSE to always-market (its argmin <= market by construction).
    loses_urgent = (not is_calm) and baselines.engine_argmin > baselines.always_market + 1e-9
    return ExecutionValidationReport(
        shortfall=shortfall,
        calibration=calibration,
        baselines=baselines,
        ablation=ablation,
        beats_market_on_calm=beats_calm,
        loses_to_market_on_urgent=loses_urgent,
    )
