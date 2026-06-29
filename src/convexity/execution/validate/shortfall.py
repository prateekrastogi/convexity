"""ShortfallDecomposer — realized-vs-expected IS per component / tactic / regime (§12, Invariant V).

VALIDATION-ONLY (firewalled, Invariant V): matches each :class:`ExecDecision` to its
:class:`RealizedFill` and produces the per-component realized-vs-expected IS table, per tactic and per
regime bucket, with block-bootstrap CIs (deferred to ``validate/stats.py``). Systematic
under-prediction of any component => that estimator is miscalibrated (§12). Uses future bars for the
adverse-selection horizon (firewalled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.execution.types import ExecDecision, RealizedFill

__all__ = ["ComponentResidual", "ShortfallReport", "ShortfallDecomposer"]


@dataclass(frozen=True)
class ComponentResidual:
    """Realized − expected for one cost component, with the mean residual (§12)."""

    component: str
    expected_mean_bps: float
    realized_mean_bps: float
    residual_mean_bps: float  # realized − expected; persistent positive => under-prediction
    n: int


@dataclass(frozen=True)
class ShortfallReport:
    """The per-component realized-vs-expected IS table (§12)."""

    components: tuple[ComponentResidual, ...]
    total_expected_bps: float
    total_realized_bps: float


class ShortfallDecomposer:
    """Realized-vs-expected IS decomposition (§12). Validation-only (Invariant V)."""

    _COMPONENTS = ("spread", "impact", "adverse", "delay", "funding")

    def decompose(
        self,
        decisions: list[ExecDecision],
        realized_fills: list[RealizedFill],
        cfg=None,
    ) -> ShortfallReport:
        """Match decisions to realized fills; build the per-component residual table (§12)."""
        by_intent = {f.intent_id: f for f in realized_fills}
        exp: dict[str, list[float]] = {c: [] for c in self._COMPONENTS}
        real: dict[str, list[float]] = {c: [] for c in self._COMPONENTS}

        for d in decisions:
            fill = by_intent.get(d.intent_id)
            if fill is None:
                continue
            dec = d.cost_decomposition
            exp["spread"].append(self._scalar(dec, "spread_bps", d.tactic))
            exp["impact"].append(self._scalar(dec, "impact_bps", d.tactic))
            exp["adverse"].append(self._scalar(dec, "adverse_sel_bps", d.tactic))
            exp["delay"].append(self._scalar(dec, "delay_bps", d.tactic))
            exp["funding"].append(self._scalar(dec, "funding_bps", d.tactic))
            real["spread"].append(fill.realized_spread_bps)
            real["impact"].append(fill.realized_impact_bps)
            real["adverse"].append(fill.realized_adverse_sel_bps)
            real["delay"].append(fill.realized_delay_bps)
            real["funding"].append(fill.realized_funding_bps)

        residuals = []
        for c in self._COMPONENTS:
            e = float(np.mean(exp[c])) if exp[c] else 0.0
            r = float(np.mean(real[c])) if real[c] else 0.0
            residuals.append(ComponentResidual(c, e, r, r - e, len(exp[c])))

        total_e = sum(cr.expected_mean_bps for cr in residuals)
        total_r = sum(cr.realized_mean_bps for cr in residuals)
        return ShortfallReport(tuple(residuals), total_e, total_r)

    @staticmethod
    def _scalar(decomposition, key: str, tactic: str) -> float:
        """Pull the chosen tactic's component out of the (possibly per-tactic) diagnostics map."""
        val = decomposition.get(key)
        if isinstance(val, dict):
            return float(val.get(tactic, 0.0))
        return float(val) if val is not None else 0.0
