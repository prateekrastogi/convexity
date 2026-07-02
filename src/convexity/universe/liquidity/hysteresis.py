"""Per-symbol hysteresis state machine for liquidity-floor membership (UNIVERSE.md §2.2).

A bare threshold ``rank_i(t) >= q`` whipsaws at the boundary: a name oscillating around the ``q``-th
percentile enters and exits ``U_t`` every few rebalances on liquidity-rank noise, paying full
round-trip friction for zero signal content (§2.2, index "reconstitution churn"). The fix is a
path-dependent band ``q_in > q_out`` (§2.2):

    OUT and rank_t >= q_in   -> ENTER (IN)
    IN  and rank_t <  q_out  -> EXIT  (OUT)
    otherwise                -> UNCHANGED from prior state

A name must climb ABOVE ``q_in`` to earn entry but only falls out once it drops BELOW the strictly
lower ``q_out``; inside the band ``(q_out, q_in)`` a name holds whatever state it already had. This is
a deterministic per-symbol finite-state machine (state ∈ {IN, OUT}), so it is ``F_t``-measurable — the
state at ``t`` is a function of the state at ``t^-`` (itself ``F_{t^-}``-measurable) and
``rank_i(t) ∈ F_t`` — and introduces no lookahead; it only adds a one-step memory to §1.1's indicator.

The band width (``q_in - q_out``) is derived from a CUSUM average-run-length calibration (Page 1954,
§2.3), not fitted to maximize Sharpe; the deployed 0.80/0.65 (width 0.15) is a BLUE-searchable
operating point, deliberately narrower than the fully-conservative ``h_min`` bound (§2.3).

Invariant HYST (§2.2): entry requires crossing the HIGHER threshold, exit the LOWER; membership inside
the band holds the prior state. Two DISTINCT mechanisms force OUT, and this docstring keeps them apart:
  (1) a NaN rank -- a name OUTSIDE ``U_t^base`` at ``t`` (§1.4), i.e. not in the ranking pool at all --
      hits the explicit NaN->OUT branch (``~valid``) below, regardless of prior state (§11.1 contract, P6);
  (2) a §2.4-stale base-eligible name carries a FINITE rank 0.0 (the least-liquid sentinel from
      ``ranking.liquidity_rank``, NOT NaN), so it exits through the ordinary ``rank_t < q_out`` band
      branch (0.0 < q_out), not the NaN branch.
Both paths force OUT, so behavior is identical; they are named separately only because a stale name is
still a ranked pool member (rank 0.0) while an out-of-base name is not ranked at all (NaN).
"""

from __future__ import annotations

import numpy as np


def hysteresis_step(
    rank_t: np.ndarray,
    state_prior: np.ndarray,
    *,
    q_in: float,
    q_out: float,
) -> np.ndarray:
    """One rebalance's per-symbol hysteresis FSM transition (UNIVERSE.md §2.2).

    ``rank_t`` -- shape (n_symbols,), this rebalance's ``rank_i(t)`` in (0,1] for a rankable name, 0.0
                  for a §2.4-stale base-eligible name (finite least-liquid sentinel from
                  ``ranking.liquidity_rank``), NaN for a symbol not in ``U_t^base`` (§1.4). A NaN input
                  MUST transition to OUT regardless of ``state_prior`` (P6, §2.4 conservative fallback
                  extended to the FSM boundary: a name that cannot be ranked cannot be a liquidity
                  member); a stale rank 0.0 is finite and instead exits via the ordinary ``< q_out``
                  band branch (0.0 < q_out) -- both force OUT, by two distinct branches (see module
                  docstring, Invariant HYST).
    ``state_prior`` -- shape (n_symbols,), bool, this symbol's IN/OUT state at ``t^-`` (the FSM's own
                        memory -- §2.2's ``F_t``-measurability argument requires this be exactly the prior
                        call's returned state, never recomputed from scratch).
    ``q_in``, ``q_out`` -- RED thresholds, ``q_in > q_out`` (§2.2-2.3).

    Returns shape (n_symbols,) bool, the state AFTER this rebalance's transition (§2.2's three cases:
    ENTER if ``~state_prior & (rank_t >= q_in)``; EXIT if ``state_prior & (rank_t < q_out)``; else
    UNCHANGED). NaN ranks force OUT before the band logic. Invariant HYST (§2.2).
    """
    rank_t = np.asarray(rank_t, dtype=float)
    prior = np.asarray(state_prior, dtype=bool)
    if rank_t.shape != prior.shape:
        raise ValueError(f"rank_t {rank_t.shape} and state_prior {prior.shape} must match (UNIVERSE.md §2.2)")
    if not (q_in > q_out):
        raise ValueError(f"hysteresis requires q_in > q_out (UNIVERSE.md §2.2); got {q_in} <= {q_out}")

    valid = np.isfinite(rank_t)
    # Band transitions (only meaningful where rank is valid); NaN handled by the mask below.
    enter = (~prior) & valid & (rank_t >= q_in)
    exit_ = prior & (~valid | (rank_t < q_out))  # NaN -> exit (P6, §2.4); or a genuine drop below q_out

    state_next = prior.copy()
    state_next[enter] = True
    state_next[exit_] = False
    # A NaN rank can never be IN (P6): even from OUT it cannot enter (enter requires valid & >= q_in),
    # and from IN it exits via exit_ above. So every NaN column ends OUT, as required.
    return state_next
