"""Frozen cross-boundary dataclasses for the UNIVERSE pillar (UNIVERSE.md §11.1).

One ``EligibilitySnapshot`` per rebalance ``t`` (the long/base gate, §1-3); one ``ShortabilitySnapshot``
per rebalance ``t`` (the stricter short subset, §4). Both immutable: a snapshot is the frozen record
of what was knowable in ``F_t`` at ``t``, and must never be mutated after construction (§9 edge case 9:
``U_t`` history is never retroactively recomputed). Field shapes/types match §11.1's build contract
byte-for-byte; array conventions mirror ``src/convexity/panel/panel.py`` (bool masks and float ranks over the
panel's fixed ``(n_symbols,)`` column order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EligibilitySnapshot:
    """One rebalance's eligibility state (UNIVERSE.md §1-3). Immutable, one per ``t``.

    Field contract is §11.1 verbatim. All masks/ranks are over the panel's fixed ``(n_symbols,)``
    column order (``src/convexity/panel/panel.py::Panel.symbols``).
    """

    t: int  # rebalance index (Friday-close ordinal, matching src/convexity/panel/panel.py's day index)
    base_eligible: np.ndarray  # shape (n_symbols,), bool — U_t^base (§1.4)
    hysteresis_state: np.ndarray  # shape (n_symbols,), bool — U_t^base-liquid IN/OUT (§2.2)
    # rank_i(t) (§2.1). Three-valued over the panel column order:
    #   (0,1] — percentile rank of a genuinely-rankable name (1 = most liquid);
    #   0.0   — §2.4 conservative-fallback SENTINEL for a base-eligible name whose DV is
    #           missing/stale (fewer than ceil(W·min_valid_frac) valid days). NOT a percentile;
    #           deliberately below the (0,1] band so the hysteresis gate can only EXIT it, never
    #           admit it (ranking.py::liquidity_rank docstring, §2.4);
    #   NaN   — name outside U_t^base (§1.4), i.e. not in the ranking pool at all.
    # The (0,1] contract in §11.1 is the range for the ranking POOL; 0.0/NaN are the two
    # documented out-of-pool markers, handled uniformly as least-liquid/OUT downstream.
    rank: np.ndarray  # shape (n_symbols,), float
    universe: np.ndarray  # shape (n_symbols,), bool — final U_t (§3.2), universe <= hysteresis_state
    n_hysteresis_entries: int  # count of OUT->IN flips this rebalance (§9 edge case 8 diagnostic)
    n_hysteresis_exits: int  # count of IN->OUT flips this rebalance (§9 edge case 8 diagnostic)


@dataclass(frozen=True)
class ShortabilitySnapshot:
    """One rebalance's stricter short-eligibility state (UNIVERSE.md §4). Immutable, one per ``t``.

    ``short_eligible`` is the ``U_t^short`` mask, always a SUBSET of the companion ``EligibilitySnapshot``'s
    ``universe`` (§4.1: ``U_t^short ⊆ U_t``, Invariant SHORT). UNIVERSE is a provider (P2): when the
    bottom-ranked tail is short-starved (§4.4) it EXPOSES the constraint via ``n_short_eligible_in_tail``
    / ``short_starved_flag`` and never resolves it — it does NOT silently widen the short set or shrink
    the long leg (that policy belongs to PORTFOLIO/SIGNAL).
    """

    t: int  # rebalance index, matching the companion EligibilitySnapshot.t
    short_eligible: np.ndarray  # shape (n_symbols,), bool — U_t^short (§4.2), short_eligible <= universe
    short_permitted: np.ndarray  # shape (n_symbols,), bool — AMBER venue-status flag (§4.2)
    short_liquid: np.ndarray  # shape (n_symbols,), bool — strict short-liquidity gate (§4.2)
    n_short_eligible_in_tail: int  # count of short-eligible names in the bottom-k tail (§4.4 diagnostic)
    short_starved_flag: bool  # True iff n_short_eligible_in_tail < k (§4.4; a report, never a resolution)
