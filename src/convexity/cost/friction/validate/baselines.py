"""Documented cost baselines / comparators for the derived per-name model (COST.md Section 3).

Firewalled validation companion (nothing here feeds a live rebalance). These are the reference
friction rates a validation run holds the derived ``SpreadImpactModel`` (Sections 5.2-5.3) up
against, each expressed as a per-unit-turnover round-trip rate consumable by ``settle_week(...,
friction=<rate>)`` (Invariant N: the rate is an INPUT, the settlement stays in accounting.py):

  * ``flat_bps_friction`` — ``src/convexity/panel/sleeve.py::CRYPTO_FRICTION``'s flat conservative
    per-class rate (the research comparator the derived model supersedes for the crypto sleeve,
    COST.md Section 1 table). A scalar; feeds the scalar cost path.
  * ``no_impact_friction`` — the derived model with the impact term ZEROED (fee + half-spread only),
    isolating how much of the cost drag the Amihud impact line contributes.
  * ``perfect_foresight_friction`` — the degenerate ZERO-cost baseline (the light-cost/optimistic
    upper bound, COST.md Section 8.4 / Invariant HONEST-SR): its net Sharpe is the number that must
    NEVER be reported as deployable, kept here ONLY so a report can show the honest-vs-optimistic gap
    with the optimistic side explicitly labeled.

Each baseline is a closed-form transform of the SAME per-leg components as the deployed model, so a
comparison is apples-to-apples (only the priced-cost line changes, never the weights or the signal).
"""

from __future__ import annotations

import numpy as np

from convexity.cost.accounting import round_trip_friction
from convexity.cost.friction.spread_impact import SpreadImpactModel
from convexity.panel.sleeve import CRYPTO_FRICTION


def flat_bps_friction() -> float:
    """The flat ``CRYPTO_FRICTION`` round-trip rate (COST.md Section 1 research comparator).

    ``2 * (taker + half_spread + impact)`` from ``universe.CRYPTO_FRICTION`` (5 + 1 + 0.5 bp per leg
    -> 13bp round-trip) via ``accounting.round_trip_friction`` (Invariant N: the assembly is
    accounting.py's, never re-derived). The flat conservative research default the derived per-name
    model supersedes for the crypto production sleeve.
    """
    return round_trip_friction(
        taker=CRYPTO_FRICTION.taker,
        half_spread=CRYPTO_FRICTION.half_spread,
        impact=CRYPTO_FRICTION.impact,
    )


def no_impact_friction(model: SpreadImpactModel, adv_i: float) -> float:
    """Derived per-name friction with the Amihud impact term zeroed (fee + half-spread only).

    Isolates the impact line's contribution to cost drag (COST.md Section 5.3: the illiquid-tail
    cost is only visible if impact varies per name). Uses the SAME half-spread the deployed model
    computes for ``adv_i`` (Sections 5.2), so the delta against ``model.round_trip_friction_i`` is
    exactly the impact contribution.
    """
    return round_trip_friction(
        taker=model.taker,
        half_spread=model.half_spread(adv_i),
        impact=0.0,
    )


def perfect_foresight_friction(n_names: int = 1) -> np.ndarray:
    """The degenerate ZERO-cost baseline (COST.md Section 8.4, Invariant HONEST-SR).

    A zero per-name friction vector: its net Sharpe is the light-cost/optimistic upper bound
    (~1.6-1.9, NOT deployable). Kept ONLY so a report can quantify the honest-vs-optimistic gap with
    the optimistic side explicitly labeled — never a promotion-gate input, a headline, or a sizing
    input (Invariant HONEST-SR). Returns a length-``n_names`` zero array for the per-symbol cost path.
    """
    return np.zeros(int(n_names), dtype=float)
