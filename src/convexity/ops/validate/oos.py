"""OOS validation of the limits inside SYSTEM's CPCV framework — NO P&L objective (§11.2, Invariant
OOS).

Spec: OPS §11.2. The BLUE limit grids are searched and promoted by a STATISTICAL objective on held-out
folds — the in-control false-alarm rate (does the panel page roughly at ``Σ_d(1/ARL0,d)``, the
source-counted budget of §6.6, on genuinely in-control OOS data?) and the detection delay at injected
synthetic shifts (does ARL1 hold OOS?). The objective contains NO P&L term (Invariant OOS): OPS limits
are graded on detection quality, not on whether halting happened to save money on one history.

The promotion objective here is a pure function of the OOS false-alarm rate + detection delay vs their
targets; swapping the P&L series for noise leaves the promoted limits unchanged (the Invariant-OOS
test). This module is part of the MODEL — never imported by the live layers (Invariant V).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OosObjective:
    """The statistical promotion objective for a candidate limit (§11.2). NO P&L term (Invariant OOS).

    Lower is better: a candidate is graded on how close its OOS realized false-alarm rate is to the
    budget AND how small its detection delay is at the design shift. ``far_target`` is the source-
    counted budget ``Σ_d(1/ARL0,d)``; ``delay_target`` the design ARL1.
    """

    realized_far: float        # OOS in-control false-alarm rate (per sample)
    far_target: float          # the Σ_d(1/ARL0,d) budget (§6.6)
    realized_delay: float      # OOS detection delay at the injected design shift
    delay_target: float        # the design ARL1

    def score(self) -> float:
        """A pure detection-quality score (no P&L): NORMALIZED FAR miss + normalized delay (§11.2).

        Both terms are dimensionless ratios so neither dominates by unit scale: the FAR miss is
        relative to the budget, the delay penalty relative to the design ARL1. A candidate that floods
        (FAR ≫ budget) is penalized as heavily as one that is blind (delay ≫ ARL1)."""
        far_miss = ((self.realized_far - self.far_target) / max(self.far_target, 1e-9)) ** 2
        delay_pen = (max(0.0, self.realized_delay - self.delay_target)
                     / max(self.delay_target, 1e-9)) ** 2
        return far_miss + delay_pen

    def contains_pnl_term(self) -> bool:
        """Invariant OOS structural check: the objective has NO P&L term (always False, by design)."""
        return False


def promote(candidates: dict[str, OosObjective]) -> str:
    """Promote the candidate with the best (lowest) detection-quality score (§11.2, Invariant OOS).

    The selection is over the statistical objective only — swapping any P&L series for noise cannot
    change the result because no P&L enters :meth:`OosObjective.score`. Returns the winning key.
    """
    if not candidates:
        raise ValueError("no candidate limits to promote")
    return min(candidates, key=lambda key: candidates[key].score())


__all__ = ["OosObjective", "promote"]
