"""Finite candidate grids for the governed BLUE search (BREAKOUT §7 / §2 BLUE).

Enumerates the finite, predeclared BLUE candidate configs (BREAKOUT §2: "finite candidate grids
only"). Each candidate is a ``(windows, cross_k, arl0, preset)`` tuple; the cardinality is the
within-symbol trial count fed to the deflated-Sharpe gate (§10) and SYSTEM's ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from convexity.breakout.config.tuning import TuningPolicy


@dataclass(frozen=True)
class Candidate:
    """One BLUE search point (BREAKOUT §2)."""

    windows: tuple[int, ...]
    cross_k: float
    arl0: int
    preset: str


def enumerate_candidates(policy: TuningPolicy) -> list[Candidate]:
    """Return the full finite BLUE grid (cardinality == policy.blue_cardinality())."""
    return [
        Candidate(windows=w, cross_k=k, arl0=a, preset=p)
        for w, k, a, p in product(
            policy.windows_grid, policy.cross_k_grid, policy.arl0_grid, policy.preset_grid
        )
    ]
