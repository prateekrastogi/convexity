"""Cross-sectional liquidity rank ``rank_i(t)`` and the size-N cutoff (UNIVERSE.md §2.1, §3.2).

Because the target universe is a fixed COUNT (``N ≈ 50``), the liquidity floor is not an absolute
dollar threshold but a cross-sectional ORDER STATISTIC (§2.1): at each rebalance, rank all
base-eligible names by ``DV_i(d)`` and admit by rank. ``rank_i(d) ∈ (0, 1]`` is ``i``'s percentile rank
of ``DV_i(d)`` among ``{j : j ∈ U_d^base}``; rank 1 = MOST liquid.

Two uses of the ONE ranking (§3.2): first as the pass/fail hysteresis gate (§2.2, via
``hysteresis.hysteresis_step``), then as the cardinality cutoff (``U_t = top-N(U_t^base-liquid)``) —
never two divergent liquidity notions (Invariant SIZE). If fewer than ``N`` names clear the gate,
``U_t`` is the whole hysteresis-eligible set, never backfilled with ineligible names (§3.2, §9 edge 3).

§2.4 conservative fallback (P6): a name whose ``DV_i(d)`` is missing/stale (fewer than ``ceil(W/2)``
valid days, or NaN) is assigned rank 0 (least liquid) — never a neutral/average rank, so it can only
be excluded by this rule, never admitted.
"""

from __future__ import annotations

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, UniverseConfig
from convexity.universe.liquidity.dollar_volume import min_valid_days


def liquidity_rank(
    dv_t: np.ndarray,
    base_eligible: np.ndarray,
    *,
    valid_days: np.ndarray | None = None,
    cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG,
) -> np.ndarray:
    """Cross-sectional percentile liquidity rank ``rank_i(t)`` in (0,1], 1 = most liquid (UNIVERSE.md §2.1).

    ``dv_t`` -- shape (n_symbols,), this rebalance's ``DV_i(d)`` (§2.1); NaN for names not live at ``d``.
    ``base_eligible`` -- shape (n_symbols,), bool ``U_t^base`` (§1.4). Only base-eligible names are ranked;
                         all others get NaN (they are not in the ranking pool at all).
    ``valid_days`` -- optional shape (n_symbols,) int from ``dollar_volume.valid_day_count``. When given,
                      a base-eligible name with fewer than ``ceil(W * min_valid_frac)`` valid days is
                      forced to rank 0 (§2.4 conservative fallback, P6), NOT dropped from the pool — so
                      it counts as a ranked-but-least-liquid member the hysteresis gate can exit.

    Returns shape (n_symbols,) float in (0,1] for ranked names, 0.0 for §2.4-stale ranked names, NaN
    for names outside ``U_t^base``. Ties are average-ranked (mirrors ``research/xsec/signals.py::_pct_rank``);
    the highest ``DV`` maps to rank ``m/m = 1.0`` (1 = most liquid, §2.1's convention).
    """
    base_eligible = np.asarray(base_eligible, dtype=bool)
    dv = np.asarray(dv_t, dtype=float)
    out = np.full(dv.shape, np.nan)

    # §2.4: a base-eligible name whose DV is NaN or below the valid-day floor is forced to rank 0.
    stale = np.zeros(dv.shape, dtype=bool)
    if valid_days is not None:
        stale = np.asarray(valid_days) < min_valid_days(cfg)
    stale = stale | ~np.isfinite(dv)
    stale = stale & base_eligible  # only names actually in the ranking pool

    rankable = base_eligible & ~stale & np.isfinite(dv)
    idx = np.flatnonzero(rankable)
    m = idx.size
    if m == 0:
        out[base_eligible & (stale)] = 0.0
        return out
    if m == 1:
        # A single rankable name is the most liquid of its (size-1) pool -> rank 1.0 (§2.1: 1=most liquid).
        out[idx] = 1.0
    else:
        v = dv[idx]
        order = np.argsort(v, kind="mergesort")  # ascending: least liquid first
        ranks = np.empty(m)
        sv = v[order]
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[j + 1] == sv[i]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank (ties averaged)
            ranks[order[i : j + 1]] = avg
            i = j + 1
        out[idx] = ranks / m  # scale to (0,1]; highest DV -> m/m = 1.0 (most liquid)
    # §2.4-stale, base-eligible names sit at rank 0 (least liquid), never NaN and never imputed neutral.
    out[stale] = 0.0
    return out


def top_n_universe(
    rank_t: np.ndarray,
    hysteresis_state: np.ndarray,
    *,
    N: int,
) -> np.ndarray:
    """Size-``N`` universe ``U_t = top-N(U_t^base-liquid, key=rank_i(t))`` (UNIVERSE.md §3.2).

    ``rank_t`` -- shape (n_symbols,), ``rank_i(t)`` from ``liquidity_rank`` (1 = most liquid).
    ``hysteresis_state`` -- shape (n_symbols,), bool ``U_t^base-liquid`` (§2.2, the IN set after the FSM).
    ``N`` -- RED cardinality cap (§3.2).

    Keeps the ``N`` MOST liquid names that already cleared the hysteresis gate. If fewer than ``N`` names
    are IN, returns the whole IN set — never backfilled with ineligible names (§3.2, §9 edge 3).
    Invariant SIZE (§3.2): ``N`` only changes HOW MANY of the ``rank``-ordered IN names survive, never
    WHICH statistic ranks them, so ``top_n_universe(N1) ⊆ top_n_universe(N2)`` for ``N1 < N2`` on the
    same ``rank_t`` and hysteresis set (monotone nesting). Ranks that are NaN among IN names (should not
    occur -- an IN name is base-eligible and thus ranked) sort last, never displacing a valid-rank name.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1 (UNIVERSE.md §3.2); got {N}")
    state = np.asarray(hysteresis_state, dtype=bool)
    rank = np.asarray(rank_t, dtype=float)
    universe = np.zeros(state.shape, dtype=bool)
    in_idx = np.flatnonzero(state)
    if in_idx.size == 0:
        return universe
    if in_idx.size <= N:
        universe[in_idx] = True
        return universe
    # Rank the IN names by rank descending (1 = most liquid first); NaN ranks sort to the end.
    r = rank[in_idx]
    r_filled = np.where(np.isfinite(r), r, -np.inf)
    # Stable sort so ties keep panel column order deterministically (mirrors _pct_rank's mergesort).
    order = np.argsort(-r_filled, kind="mergesort")
    keep = in_idx[order[:N]]
    universe[keep] = True
    return universe
