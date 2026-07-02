"""Fractional-Kelly dial state machine ``c_t`` (LEVERAGE.md §6.2–§6.3).

``c_t ∈ {0.25, 0.50}`` starts at quarter-Kelly on cold start and every newly-promoted config (P7,
§6.2). It steps UP to half-Kelly ONLY when BOTH conditions hold (Invariant RATCHET, §6.3):

  1. the CUSUM statistic ``S_k`` (accumulated over the ENTIRE track record) crosses ``h_promote``;
  2. the minimum elapsed live-time floor ``elapsed_weeks ≥ N_min_weeks`` is satisfied.

Demotion back to quarter-Kelly is IMMEDIATE and ASYMMETRIC (fast to demote, slow to promote, §6.2)
on a stated live-Sharpe breach — deliberately NOT gated by a symmetric CUSUM accumulation (§8 edge
case: a reviewer must not "fix" this into a symmetric test). The demotion trip level
``demote_z_threshold = -k_ref = -0.25`` is now anchored in §3/§6.2: it is the degenerate
IMMEDIATE (``h_demote=0``) lower-CUSUM mirror of the promotion CUSUM's reference offset (§6.3,
Hawkins & Olwell 1998), so it defaults from ``LeverageDefaults`` yet stays overridable at the call
site (the injection path is preserved for OPS governance overrides).
"""

from __future__ import annotations

import numpy as np

from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.ratchet.cusum import run_cusum, standardized_surprise
from convexity.leverage.types import RatchetState


def initial_state(sr_ref: float, defaults: LeverageDefaults) -> RatchetState:
    """The cold-start ratchet state (LEVERAGE.md §6.2, P7).

    Quarter-Kelly (``c_t = c_start = 0.25``), zero accumulated CUSUM, zero elapsed weeks — so early
    promotion is impossible STRUCTURALLY (the minimum-elapsed-time floor is unmet), not merely by
    "starting there" (§8 cold-start edge case). ``sr_ref`` is the deflated ``SR_LCB_t`` frozen at
    eligibility time and held as the CUSUM reference (§6.3).
    """
    return RatchetState(
        c_t=defaults.c_start, s_k=0.0, elapsed_weeks=0, promoted=False, sr_ref=float(sr_ref)
    )


def evaluate_promotion(
    sr_week: np.ndarray,
    se_sr: float,
    elapsed_weeks: int,
    defaults: LeverageDefaults,
    *,
    sr_ref: float,
) -> RatchetState:
    """Evaluate the promotion ratchet over the live track record (LEVERAGE.md §6.3, Invariant RATCHET).

    ``sr_week`` = the LIVE per-week Sharpe series since eligibility began (never backtest, P7);
    ``se_sr`` = the (assumed constant/known) standardizing denominator (§6.3); ``sr_ref`` = the
    frozen deflated ``SR_LCB_t`` reference. Promotion to ``c=0.50`` requires BOTH the CUSUM crossing
    ``h_promote`` AND ``elapsed_weeks ≥ N_min_weeks`` (§6.3): a single lucky week embedded in an
    otherwise flat/negative record does NOT promote, because the accumulated ``S_k`` from the
    surrounding weeks pulls it back below ``h_promote`` (Invariant RATCHET).
    """
    z = standardized_surprise(sr_week, sr_ref, se_sr)
    cusum = run_cusum(z, k_ref=defaults.k_ref, h_promote=defaults.h_promote)
    floor_met = elapsed_weeks >= defaults.n_min_weeks
    promote = cusum.crossed and floor_met
    return RatchetState(
        c_t=defaults.c_promoted if promote else defaults.c_start,
        s_k=cusum.s_k,
        elapsed_weeks=int(elapsed_weeks),
        promoted=promote,
        sr_ref=float(sr_ref),
    )


def apply_demotion(
    state: RatchetState,
    latest_z: float,
    defaults: LeverageDefaults,
    *,
    demote_z_threshold: float | None = None,
) -> RatchetState:
    """Immediate asymmetric demotion on a live-Sharpe breach (LEVERAGE.md §6.2, §8).

    If currently promoted (``c_t == c_promoted``) and the latest standardized weekly surprise falls
    below ``demote_z_threshold``, step DOWN to ``c_start`` immediately — NOT gated by a symmetric
    CUSUM accumulation (§6.2, the deliberate slow-to-promote / fast-to-demote asymmetry). A
    quarter-Kelly book is unaffected (already at the floor).

    ``demote_z_threshold`` defaults to the §3/§6.2-anchored ``defaults.demote_z_threshold = -k_ref =
    -0.25`` when ``None``: the symmetric, degenerate-immediate (``h_demote=0``) lower-CUSUM mirror of
    the promotion reference offset (§6.3, Hawkins & Olwell 1998). It is no longer a fabricated
    constant — it equals ``-k_ref`` by derivation from the SAME framework that pins ``h_promote``.
    The injection path is retained: an OPS governance layer MAY pass an explicit value (e.g. ``0.0``
    to demote on ANY negative surprise, a stricter user preference).
    """
    thr = defaults.demote_z_threshold if demote_z_threshold is None else demote_z_threshold
    if state.promoted and latest_z < thr:
        return RatchetState(
            c_t=defaults.c_start,
            s_k=state.s_k,
            elapsed_weeks=state.elapsed_weeks,
            promoted=False,
            sr_ref=state.sr_ref,
        )
    return state


def next_c(
    sr_week: np.ndarray,
    se_sr: float,
    elapsed_weeks: int,
    defaults: LeverageDefaults,
    *,
    sr_ref: float,
    demote_z_threshold: float | None = None,
) -> float:
    """Resolve the active fractional-Kelly dial ``c_t`` for the next rebalance (LEVERAGE.md §6.2–§6.3).

    Convenience wrapper: runs the promotion evaluation and applies immediate demotion on the latest
    surprise, returning the single scalar ``c_t ∈ {0.25, 0.50}`` the composition rule multiplies by
    (§6.1). Empty/short live history stays at the quarter-Kelly floor (Invariant RATCHET).

    ``demote_z_threshold`` defaults to the §3/§6.2-anchored ``defaults.demote_z_threshold`` when
    ``None`` (see :func:`apply_demotion`); an explicit value is still honored for OPS overrides.
    """
    if np.asarray(sr_week, dtype=float).size == 0:
        return defaults.c_start
    promoted = evaluate_promotion(
        sr_week, se_sr, elapsed_weeks, defaults, sr_ref=sr_ref
    )
    latest_z = float(standardized_surprise(sr_week, sr_ref, se_sr)[-1])
    return apply_demotion(
        promoted, latest_z, defaults, demote_z_threshold=demote_z_threshold
    ).c_t
