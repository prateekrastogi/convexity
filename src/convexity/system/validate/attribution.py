"""Pillar P&L attribution — Shapley over 16 coalitions OR sequential 5 knock-ins (SYSTEM §8.2, Inv ATTR).

A Shapley value over 4 pillars is the average marginal contribution over all 4! orderings, computed
from the OOS P&L of all 2⁴ = 16 coalition subsets:

    ϕ_p = Σ_{S ⊆ N\\{p}}  (|S|!·(|N|−|S|−1)!/|N|!)·(v(S ∪ {p}) − v(S))

Order-independence is a theorem ONLY if every v(S) is provisioned — it is NOT recoverable from 5
single-pillar ablations. So either we provision all 16 (Mode A), or we do not claim "Shapley" and
report the order-dependent sequential ablation (Mode B, the DEFAULT). Claiming Shapley from < 16
coalitions is an error (Invariant ATTR). A pillar with ~zero contribution is not earning its
complexity and its inclusion must be re-justified (mirror of TREND.md's Final Rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

PillarName = Literal["selection", "breakout", "leverage", "trend"]
ALL_PILLARS: tuple[PillarName, ...] = ("selection", "breakout", "leverage", "trend")


@dataclass(frozen=True)
class AttributionReport:
    """Output of :meth:`PillarAttribution.attribute` (Section 8.2). ``mode`` is honest: ``shapley``
    only when all 16 coalitions were provisioned, else ``sequential_fixed_order`` with the order
    stated. ``zero_contribution`` flags any pillar not earning its complexity."""

    attribution_mode: Literal["shapley", "sequential_fixed_order"]
    pillar_contribution: Mapping[PillarName, float]
    fixed_order: tuple[PillarName, ...] | None
    zero_contribution: tuple[PillarName, ...]


def _shapley(coalition_pnl: Mapping[frozenset, float]) -> dict[PillarName, float]:
    """Exact Shapley value over the 16 coalition baselines (Section 8.2)."""
    n = len(ALL_PILLARS)
    phi: dict[PillarName, float] = {}
    for p in ALL_PILLARS:
        others = [q for q in ALL_PILLARS if q != p]
        total = 0.0
        for size in range(len(others) + 1):
            from itertools import combinations

            for combo in combinations(others, size):
                s = frozenset(combo)
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                total += weight * (coalition_pnl[s | {p}] - coalition_pnl[s])
        phi[p] = total
    return phi


def _sequential(coalition_pnl: Mapping[frozenset, float], order: Sequence[PillarName]) -> dict[PillarName, float]:
    """Order-dependent marginal contributions along a fixed knock-in order (Section 8.2, Mode B)."""
    phi: dict[PillarName, float] = {}
    active: set = set()
    prev = coalition_pnl[frozenset()]
    for p in order:
        active.add(p)
        cur = coalition_pnl[frozenset(active)]
        phi[p] = cur - prev
        prev = cur
    return phi


class PillarAttribution:
    """Shapley (16 coalitions) OR sequential (5 knock-ins) pillar attribution (SYSTEM §8.2/§12)."""

    def attribute(
        self,
        coalition_pnl: Mapping[frozenset, float],
        mode: str,
        fixed_order: tuple[PillarName, ...],
        cfg=None,
        *,
        zero_eps: float = 1e-9,
    ) -> AttributionReport:
        """``mode='shapley'`` REQUIRES all 2⁴=16 coalition baselines (keys = frozensets of ON
        pillars); computes the exact Shapley ϕ_p and labels ``shapley``. ``mode='sequential'``
        requires only the 5 knock-in runs along ``fixed_order`` and labels ``sequential_fixed_order``.
        It is an ERROR to pass ``mode='shapley'`` with fewer than 16 coalitions (Invariant ATTR)."""
        if mode == "shapley":
            if len(coalition_pnl) < 16:
                raise ValueError(
                    "mode='shapley' requires all 2^4=16 coalition baselines (Invariant ATTR); "
                    f"got {len(coalition_pnl)}"
                )
            phi = _shapley(coalition_pnl)
            report_mode: Literal["shapley", "sequential_fixed_order"] = "shapley"
            order = None
        elif mode == "sequential":
            phi = _sequential(coalition_pnl, fixed_order)
            report_mode = "sequential_fixed_order"
            order = tuple(fixed_order)
        else:
            raise ValueError(f"unknown attribution mode {mode!r}")
        zero = tuple(p for p, v in phi.items() if abs(v) <= zero_eps)
        return AttributionReport(
            attribution_mode=report_mode,
            pillar_contribution=phi,
            fixed_order=order,
            zero_contribution=zero,
        )
