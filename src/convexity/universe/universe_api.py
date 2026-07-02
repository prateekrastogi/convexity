"""Public UNIVERSE entry point — composes ``U_t`` and ``U_t^short`` (UNIVERSE.md §11).

Wires the contract layers of §1-4 into the frozen ``EligibilitySnapshot`` (§11.1) and
``ShortabilitySnapshot`` without changing any proven ``research/xsec/panel.py`` behavior:

    base_eligible (§1.4 history gate, per-symbol)
      -> rank_i(t)           (§2.1 liquidity order statistic, REUSES panel.trailing_dollar_volume)
      -> hysteresis_step     (§2.2 per-symbol FSM, one-step memory of prior IN/OUT state)
      -> top-N cutoff        (§3.2, U_t = top-N(IN set by rank))

and, layered strictly on top for the short leg (§4.2's three-term AND):

    U_t^short = U_t & short_permitted_i(t) & short_liquid_i(t)

The hysteresis FSM has memory, so the eligibility of ``t`` depends on the returned state of ``t-1``
(§2.2 ``F_t``-measurability requires exactly the prior call's state, never a recompute). ``build_book``
walks the rebalance calendar carrying that state forward. A single-rebalance ``build_eligibility`` is
also exposed for the PIT structural test (§7), which needs eligibility evaluated at one ``t`` against a
future-corrupted panel — that call reconstructs the prior state causally from rows ``<= t`` only, so it
reads NO future data (Invariant PIT).
"""

from __future__ import annotations

import numpy as np

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, UniverseConfig
from convexity.universe.eligibility.history_gate import base_eligible
from convexity.universe.liquidity.dollar_volume import dollar_volume_at, min_valid_days, valid_day_count
from convexity.universe.liquidity.hysteresis import hysteresis_step
from convexity.universe.liquidity.ranking import liquidity_rank, top_n_universe
from convexity.universe.shortability.liquidity_gate import ShortLiquidityThreshold, short_liquid
from convexity.universe.shortability.venue_gate import short_permitted
from convexity.universe.types import EligibilitySnapshot, ShortabilitySnapshot
from research.xsec.panel import Panel


def _snapshot_at(
    panel: Panel,
    dvol: np.ndarray,
    t: int,
    state_prior: np.ndarray,
    cfg: UniverseConfig,
) -> EligibilitySnapshot:
    """One rebalance's ``EligibilitySnapshot`` given the FSM state at ``t^-`` (UNIVERSE.md §1-3).

    ``state_prior`` is the hysteresis IN/OUT vector returned by the previous rebalance (§2.2 memory).
    Reads only panel rows ``<= t`` (via the history gate, ``trailing_dollar_volume``, and ``base_eligible``),
    so the snapshot is ``F_t``-measurable (Invariant PIT).
    """
    be = base_eligible(panel, t, cfg)
    dv_t = dollar_volume_at(panel, dvol, t, cfg)
    vdays = valid_day_count(dvol, t, cfg)
    rank_t = liquidity_rank(dv_t, be, valid_days=vdays, cfg=cfg)
    state_next = hysteresis_step(rank_t, state_prior, q_in=cfg.q_in, q_out=cfg.q_out)
    universe = top_n_universe(rank_t, state_next, N=cfg.N)
    n_entries = int(((~state_prior) & state_next).sum())
    n_exits = int((state_prior & (~state_next)).sum())
    return EligibilitySnapshot(
        t=t,
        base_eligible=be,
        hysteresis_state=state_next,
        rank=rank_t,
        universe=universe,
        n_hysteresis_entries=n_entries,
        n_hysteresis_exits=n_exits,
    )


def build_book(
    panel: Panel,
    dvol: np.ndarray,
    rebalances: list[int] | np.ndarray,
    cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG,
) -> list[EligibilitySnapshot]:
    """Walk the rebalance calendar, carrying the hysteresis FSM state forward (UNIVERSE.md §2.2).

    ``rebalances`` -- ascending panel row indices at which to evaluate eligibility (the Friday-close
                      ordinals, §0.1). The FSM starts fully OUT (no name is grandfathered IN — entry
                      requires an explicit ``q_in`` crossing, §2.2). Each snapshot's ``hysteresis_state``
                      is fed as the next rebalance's ``state_prior`` (the one-step memory §2.2 mandates,
                      never recomputed from scratch).

    Returns one ``EligibilitySnapshot`` per rebalance, in order. §9 edge case 9: this carries state
    forward unchanged across a hypothetical RED-constant change (the FSM is not reset) — callers change
    ``cfg`` prospectively only.
    """
    reb = list(int(r) for r in rebalances)
    state = np.zeros(panel.n_symbols, dtype=bool)  # everyone starts OUT (§2.2 — no grandfathering)
    out: list[EligibilitySnapshot] = []
    for t in reb:
        snap = _snapshot_at(panel, dvol, t, state, cfg)
        out.append(snap)
        state = snap.hysteresis_state
    return out


def build_eligibility(
    panel: Panel,
    t: int,
    *,
    dvol: np.ndarray | None = None,
    rebalances: list[int] | np.ndarray | None = None,
    cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG,
) -> EligibilitySnapshot:
    """Eligibility ``EligibilitySnapshot`` at a single rebalance ``t`` (UNIVERSE.md §1-3, §7).

    Because the hysteresis FSM has memory (§2.2), ``U_t`` depends on the entire prior rebalance history;
    this reconstructs it CAUSALLY by replaying every rebalance ``<= t`` in ``rebalances`` from an all-OUT
    start, reading only panel rows ``<= t``. That causal replay is exactly what makes ``U_t`` invariant to
    future-row corruption (Invariant PIT, §7) — nothing here touches a row ``> t``.

    ``dvol`` -- the daily dollar-volume array (``research.xsec.panel.build_dollar_volume``); computed once
                by the caller and passed in (it is not stored on the frozen ``Panel``). If None it is left
                to the caller to build (this function does not read the lake).
    ``rebalances`` -- the rebalance calendar up to and including ``t`` (the weekly Friday-close ordinals,
                      §0.1) — REQUIRED. The hysteresis FSM has memory, so the replay must use the SAME
                      calendar the book was walked on; there is no safe implicit default. In particular
                      NOT defaulting to the daily ``[0, t]`` calendar: on the real ~1000-day weekly panel
                      that would (a) replay ~1000 daily FSM steps per weekly rebalance (an O(t) footgun,
                      §11 build-contract note) and (b) evaluate the FSM on a DIFFERENT calendar than the
                      weekly book, yielding a different (still causal, but off-contract) ``U_t``. Pass the
                      weekly calendar explicitly. MUST NOT contain any index ``> t`` (that would be
                      lookahead); indices ``> t`` are dropped defensively.
    """
    if dvol is None:
        raise ValueError(
            "build_eligibility needs the daily dollar-volume array (research.xsec.panel.build_dollar_volume); "
            "pass dvol= explicitly (UNIVERSE.md §2.1 — this layer never reads the lake)"
        )
    if rebalances is None:
        raise ValueError(
            "build_eligibility needs the rebalance calendar (the weekly Friday-close ordinals, §0.1); "
            "pass rebalances= explicitly. There is no implicit default: the hysteresis FSM (§2.2) has "
            "memory, so U_t must be replayed on the SAME calendar the book was walked on, and defaulting "
            "to a daily [0, t] replay is both an O(t) footgun on the ~1000-day panel and off-contract "
            "(weekly book vs daily FSM). See UNIVERSE.md §11 build-contract note."
        )
    reb = [int(r) for r in rebalances if int(r) <= t]  # drop any future index (no lookahead, §1.1)
    if not reb or reb[-1] != t:
        reb = [r for r in reb if r < t] + [t]
    book = build_book(panel, dvol, reb, cfg)
    return book[-1]


def build_shortability(
    panel: Panel,
    dvol: np.ndarray,
    eligibility: EligibilitySnapshot,
    *,
    threshold_fn: ShortLiquidityThreshold,
    permitted_by_symbol: dict[str, bool] | None = None,
    k: int,
    cfg: UniverseConfig = DEFAULT_UNIVERSE_CONFIG,
) -> ShortabilitySnapshot:
    """Stricter short subset ``U_t^short`` for one rebalance (UNIVERSE.md §4).

    ``U_t^short = U_t & short_permitted_i(t) & short_liquid_i(t)`` (§4.2's three-term AND). Composed
    STRICTLY on top of ``eligibility.universe`` (``U_t``), so ``U_t^short ⊆ U_t`` always (Invariant SHORT):
    a name cannot be short-eligible without being base/long-eligible first (§4.1).

    ``threshold_fn`` -- the INJECTED short-liquidity callable (§4.2); no default, no fabricated threshold.
    ``k`` -- PORTFOLIO's quantile leg count (``portfolio.py::_leg_count``; restated, not owned here, §4.4).
             Used only to compute the short-starvation DIAGNOSTIC: if fewer than ``k`` of the bottom-ranked
             tail names are short-eligible, the flag is raised. UNIVERSE only REPORTS this; it never widens
             ``U_t^short`` or shrinks the long leg to compensate (§4.4, P2 provider-not-decider).

    §4.4 short-starvation diagnostic: the "bottom-ranked tail" is the ``k`` LEAST-liquid names within
    ``U_t`` (the short-leg candidates a bottom-quantile selection would draw from). The flag is a report;
    ``eligibility.universe`` is UNCHANGED by it.
    """
    dv_t = dollar_volume_at(panel, dvol, eligibility.t, cfg)
    # §2.4 consistency: NaN-out DV for names the long liquidity gate declared stale (fewer than
    # ``ceil(W * min_valid_frac)`` valid days) so ``short_liquid``'s ``isfinite(dv_t)`` gate (P6) treats a
    # §2.4-stale name as DV-invalid here EXACTLY as ``liquidity_rank`` did when it forced the same name to
    # rank 0 upstream. Without this, the short-liquidity absolute-floor component of ``threshold_fn`` would
    # see an un-floored DV for a name the long gate already declared unrankable (the auditor's gap 2).
    vdays = valid_day_count(dvol, eligibility.t, cfg)
    stale = np.asarray(vdays) < min_valid_days(cfg)
    dv_t = np.where(stale, np.nan, dv_t)
    permitted = short_permitted(panel, eligibility.t, permitted_by_symbol=permitted_by_symbol)
    liq = short_liquid(eligibility.rank, dv_t, threshold_fn=threshold_fn)
    short_eligible = eligibility.universe & permitted & liq  # strict subset of U_t (Invariant SHORT)

    # §4.4 diagnostic: the k least-liquid names in U_t are the intended short-leg candidates.
    in_univ = np.flatnonzero(eligibility.universe)
    if in_univ.size == 0:
        n_in_tail = 0
    else:
        r = eligibility.rank[in_univ]
        r_filled = np.where(np.isfinite(r), r, np.inf)  # NaN sorts last (most "illiquid") defensively
        order = np.argsort(r_filled, kind="mergesort")  # ascending: least liquid first
        tail = in_univ[order[: min(k, in_univ.size)]]
        n_in_tail = int(short_eligible[tail].sum())
    starved = n_in_tail < k

    return ShortabilitySnapshot(
        t=eligibility.t,
        short_eligible=short_eligible,
        short_permitted=permitted,
        short_liquid=liq,
        n_short_eligible_in_tail=n_in_tail,
        short_starved_flag=starved,
    )


__all__ = ["build_book", "build_eligibility", "build_shortability"]
