"""Ablation — turn each tactic off, measure the IS penalty (§12, firewalled, Invariant V).

A tactic that never wins is redundant or miscalibrated; a tactic that always wins means the others
are mis-priced (§12). Reports which tactic and which gate drives realized cost. Validation-only —
never imported by the live layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.types import TacticCosts, Tactic

__all__ = ["AblationResult", "ablate_tactics"]

_ALL: tuple[Tactic, ...] = ("POST_ONLY_MAKER", "LIMIT_IOC", "MARKETABLE_LIMIT", "MARKET")


@dataclass(frozen=True)
class AblationResult:
    """Per-tactic ablation: the IS with the tactic removed vs the full argmin (§12)."""

    full_argmin_bps: float
    penalty_with_tactic_off: dict[str, float]  # tactic -> (argmin without it) − full argmin (>= 0)
    winning_tactic: str


def ablate_tactics(costs: TacticCosts) -> AblationResult:
    """Measure the IS penalty of removing each tactic from the argmin (§12). Validation-only (Inv V)."""
    full = min(costs.by_tactic.values())
    winner = min(_ALL, key=lambda t: costs.by_tactic[t])
    penalty: dict[str, float] = {}
    for t in _ALL:
        remaining = [costs.by_tactic[x] for x in _ALL if x != t]
        without = min(remaining) if remaining else float("inf")
        penalty[t] = without - full  # >= 0; 0 means the tactic never bound the argmin
    return AblationResult(full, penalty, winner)
