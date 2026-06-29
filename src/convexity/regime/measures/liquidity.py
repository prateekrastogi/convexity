"""Liquidity / funding-stress regime — a PURE aggregation of EXECUTION + COST (REGIME §2.4).

The fourth measure is NOT computed from the price panel; it is aggregated from the neighbour
pillars (Invariant NREBUILD — REGIME never re-models spread or funding):

  spread_blowout_t = aggregate over the live book of EXECUTION's causal spread percentiles
  funding_extreme_t= aggregate over the live book of COST's causal funding percentiles
  liq_stress_t     = max( spread_blowout_t , funding_extreme_t )      (the worse of the two)

Aggregation is the equal-weight mean of the per-instrument causal percentiles over the live book
(a single book-wide number in ``[0, 1]``). The MAX-of-two means a stress in EITHER channel binds.

Null note (pins the §3.6 alpha): because ``liq_stress`` is the MAX of two ~uniform percentiles it is
NOT uniform under the calm null — ``P[liq_stress > p] = 1 − p²`` (independent), so its band-entry
false-alarm rate uses the ``1 − p²`` form, NOT ``1 − p`` (the single-percentile measures' form). The
band layer carries this; this module only produces the measure. (BLUE alternative: a single
percentile of a combined stress signal restores the ``1 − p`` null — §2.4; RED keeps the max-of-two.)

This is cheap (``O(n)`` over neighbour scalars) and is the ONLY measure recomputed out-of-cadence
on the 1-minute event path (Invariant CHEAP / STALEATTR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from convexity.regime.types import CostState, ExecState


def _aggregate(pct_by_symbol: Mapping[str, float], live: tuple[str, ...]) -> float:
    """Equal-weight mean of the live book's per-instrument causal percentiles (REGIME §2.4).

    Only live (panel-surviving) names contribute; a name absent from the neighbour map is skipped
    (it does not drag the aggregate). Returns 0.0 on an empty intersection (no stress signal).
    """
    vals = [float(pct_by_symbol[s]) for s in live if s in pct_by_symbol]
    if not vals:
        return 0.0
    return float(np.mean(vals))


@dataclass(frozen=True)
class LiquidityResult:
    """``liq_stress`` (the trigger) + its two components, for the §8 diagnostics (REGIME §2.4)."""

    liq_stress: float
    spread_blowout: float
    funding_extreme: float
    diag: dict[str, float]


def compute_liquidity(
    *,
    exec_state: ExecState,
    cost_state: CostState,
    live: tuple[str, ...],
) -> LiquidityResult:
    """Aggregate EXECUTION spread + COST funding into ``liq_stress = max(...)`` (REGIME §2.4).

    Pure read of neighbour outputs (Invariant NREBUILD): no spread or funding is modelled here.
    """
    spread_blowout = _aggregate(exec_state.spread_pct, live)
    funding_extreme = _aggregate(cost_state.funding_pct, live)
    liq_stress = max(spread_blowout, funding_extreme)  # min()-style: the worse of the two
    diag = {
        "liq_stress": liq_stress,
        "spread_blowout": spread_blowout,
        "funding_extreme": funding_extreme,
    }
    return LiquidityResult(liq_stress, spread_blowout, funding_extreme, diag)
