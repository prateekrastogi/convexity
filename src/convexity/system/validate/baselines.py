"""Required baselines + the 16 coalition off-semantics (SYSTEM §8.3, Invariant BL).

Every baseline runs under the IDENTICAL label contract, costs, folds, embargo, and execution stub
as the full system (Invariant BL). The system must beat these on DSR, not raw Sharpe:

  Trivial baselines:  buy&hold per instrument + equal-weight book; RANDOM matched-frequency entries
                      (same trade count + holding distribution, random side/timing) — the null a
                      real edge must clear.
  Single-module ablations: breakout-only, no-selection, no-trend, governor-off, no-leverage-ceilings.

The pillar-OFF semantics (Section 8.2) make each coalition runnable under the identical contract:

  selection OFF : full eligible universe; equal-risk weights; constant rank (no rotation/demotion).
  breakout  OFF : RANDOM matched-frequency admission (same trade count + holding-time distribution).
  leverage  OFF : FIXED per-position notional (RED constant fraction), lot-quantized DOWN, no
                  min()-of-ceilings, no governor.
  trend     OFF : FIXED vertical barrier at H_sys + the fixed ±b·σ horizontal barriers, no trail.

The 5 single-pillar ablations are a SUBSET of the 16 coalition subsets; they suffice ONLY for
sequential (Mode B) attribution, NOT for a Shapley value (Invariant ATTR).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

PillarName = Literal["selection", "breakout", "leverage", "trend"]
ALL_PILLARS: tuple[PillarName, ...] = ("selection", "breakout", "leverage", "trend")


@dataclass(frozen=True)
class OffPolicy:
    """The off-semantics for one pillar (Section 8.2). A null policy that keeps the pipeline runnable
    under the identical contract (Invariant BL). ``changes_module_evidence`` is True for breakout-off
    (random entry) and leverage-off (fixed size) — those need their own cached evidence keyed by the
    OFF policy; selection-off and trend-off are pure combiner toggles (cache-free)."""

    pillar: PillarName
    description: str
    changes_module_evidence: bool


OFF_SEMANTICS: dict[PillarName, OffPolicy] = {
    "selection": OffPolicy("selection", "full universe + equal-risk + constant rank", False),
    "breakout": OffPolicy("breakout", "random matched-frequency entry", True),
    "leverage": OffPolicy("leverage", "fixed-fraction size, lot-quantized DOWN, no governor", True),
    "trend": OffPolicy("trend", "fixed H_sys vertical + fixed ±b·σ horizontal barriers", False),
}


def coalition_subsets() -> tuple[frozenset[PillarName], ...]:
    """The 2⁴ = 16 coalition subsets S ⊆ {selection, breakout, leverage, trend} (Section 8.2).

    Ordered by size then lexicographically (∅, 4 singletons, 6 pairs, 4 triples, the full system).
    """
    subsets: list[frozenset[PillarName]] = []
    for size in range(len(ALL_PILLARS) + 1):
        for combo in combinations(ALL_PILLARS, size):
            subsets.append(frozenset(combo))
    return tuple(subsets)


def random_matched_frequency(
    n_obs: int,
    n_trades: int,
    holding_dist: np.ndarray,
    *,
    seed: int,
    ret_scale: float = 0.0,
) -> np.ndarray:
    """The RANDOM matched-frequency entry null (Section 8.3): same trade count + holding-time
    distribution as the full system, random side/timing. Returns a per-observation return series with
    ZERO expected edge (the null a real edge must clear on DSR). Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n_obs, dtype=float)
    if n_obs <= 0 or n_trades <= 0:
        return out
    entries = rng.choice(n_obs, size=min(n_trades, n_obs), replace=False)
    sides = rng.choice((-1.0, 1.0), size=len(entries))
    for e, s in zip(entries, sides):
        out[e] += s * rng.normal(0.0, ret_scale)  # zero-mean ⇒ no edge
    return out


def buy_and_hold(prices: np.ndarray) -> float:
    """Buy&hold total return for one instrument (a trivial baseline, Section 8.3)."""
    p = np.asarray(prices, dtype=float)
    if p.size < 2 or p[0] == 0.0:
        return 0.0
    return float(p[-1] / p[0] - 1.0)


SINGLE_PILLAR_ABLATIONS: tuple[str, ...] = (
    "breakout_only",          # no selection, no trend exit, governor-off
    "no_selection",           # trade every breakout-qualified instrument
    "no_trend",               # fixed-horizon / fixed-barrier exit
    "governor_off",           # drawdown governor disabled (must show WORSE max drawdown)
    "no_leverage_ceilings",   # fixed leverage (must show fatter tails / worse ES)
)
