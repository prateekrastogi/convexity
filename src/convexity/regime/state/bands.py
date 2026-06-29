"""Per-sub-state hysteresis: map one measure value -> an ordered sub-state (REGIME §3.2).

Each of the four measures has its OWN entry/exit band memory (Invariant CLOCK two-clocks (1)):

  * **Escalation** to a more-stressed state uses the high ENTER mark and MAY jump multiple states
    in one bar (a true crisis prints ``delta_AR > +2.0`` straight to CRISIS).
  * **De-escalation** uses the looser EXIT mark and is ONE step at a time, gated by THIS measure's
    own exit threshold ALONE — NOT the global dwell clock (that is a separate second gate on the
    GLOBAL state, applied in the machine, §3.3). So a sub-state may step down on its own exit
    threshold while the global state is still dwell-locked (the §6.2 ``t2`` handoff).

The asymmetry (enter fast, leave slow) plus a non-zero enter/exit gap is Invariant HYST: a measure
oscillating between its enter and exit marks does NOT change its own sub-state.

This module also carries the SPC ``alpha_k``/``ARL0`` budget metadata (§3.6), including the LIQ
channel's ``1 − p²`` max-of-two-percentiles null and the per-minute ``alpha^liq_min``/``D_liq_min``
budget for the event-driven path — telemetry for the Section-9 ARL audit, never a gate on ``g_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.regime.config.defaults import HysteresisBands, PolicyDefaults
from convexity.regime.types import (
    G_OF_STATE,
    STATE_ORDER,
    RegimeStateName,
    state_rank,
)


def _entry_target(value: float, bands: HysteresisBands) -> RegimeStateName:
    """Highest (most-stressed) state whose ENTER mark ``value`` clears (REGIME §3.2 escalation)."""
    target: RegimeStateName = "CALM"
    for state, (enter, _exit) in bands.pairs():
        if value >= enter:
            target = state
    return target


def classify_substate(
    value: float,
    prev_state: RegimeStateName,
    bands: HysteresisBands,
) -> RegimeStateName:
    """Per-sub-state hysteretic classification of one measure (REGIME §3.2, Invariants HYST/CLOCK).

    ``value`` is the measure's current value (percentile for vol/turb/liq; ``delta_AR`` for corr);
    ``prev_state`` is THIS measure's own remembered band. Escalation (entry mark, multi-jump) is
    immediate; de-escalation (exit mark) is one step at a time. Returns the new sub-state.
    """
    by_entry = _entry_target(value, bands)
    prev_rank = state_rank(prev_state)

    # ESCALATION — value clears a more-stressed ENTER mark: jump straight there.
    if state_rank(by_entry) > prev_rank:
        return by_entry

    # HOLD — still at/above the current state's own ENTER mark (or its entry target equals prev).
    if state_rank(by_entry) == prev_rank:
        return prev_state

    # Possible DE-ESCALATION — value fell below the entry of the current state. Step down ONE level
    # iff it is also below the PREV state's looser EXIT mark; else hysteresis HOLDS (Invariant HYST).
    if prev_rank == 0:
        return prev_state
    _, prev_exit = _band_pair_for(prev_state, bands)
    if value < prev_exit:
        return STATE_ORDER[prev_rank - 1]  # one step down only (§3.2)
    return prev_state  # inside the hysteresis gap -> hold


def _band_pair_for(state: RegimeStateName, bands: HysteresisBands) -> tuple[float, float]:
    """The (enter, exit) pair whose TARGET is ``state`` (the boundary entering ``state``)."""
    for s, pair in bands.pairs():
        if s == state:
            return pair
    return (float("inf"), float("-inf"))  # CALM has no entering boundary


def g_of(state: RegimeStateName) -> float:
    """The sub-multiplier ``g(state)`` for one classified sub-state (REGIME §3.1)."""
    return G_OF_STATE[state]


# --------------------------------------------------------------------------- SPC / ARL budget


@dataclass(frozen=True)
class ARLBudget:
    """A per-state ``alpha_k`` raw false-entry rate + the dwell/hysteresis-implied ``ARL0`` (§3.6).

    ``alpha_eff = P[cross enter] · P[stay > exit]^(D_min−1)`` and ``ARL0 ≈ 1/alpha_eff`` (in
    decision bars). For single-percentile measures (vol/turb) the per-bar rate is ``1 − p``; for the
    LIQ max-of-two-percentiles it is ``1 − p²`` (§3.6a). Telemetry only — never gates ``g_t``.
    """

    state: RegimeStateName
    alpha_eff: float
    arl0: float


def _p_cross(p_entry: float, is_liq: bool) -> float:
    """Per-bar false-cross probability at a percentile band ``p_entry`` (``1−p`` or ``1−p²``)."""
    return (1.0 - p_entry**2) if is_liq else (1.0 - p_entry)


def arl0_budget(name: str, cfg: PolicyDefaults) -> tuple[ARLBudget, ...]:
    """The per-state ``ARL0`` budget for one PERCENTILE measure (REGIME §3.6b; liq uses ``1−p²``).

    Returns one :class:`ARLBudget` per escalation rung (NORMAL/STRESSED/CRISIS) computed from the
    canonical enter/exit percentile marks and ``D_min``. Not defined for the corr displacement
    measure (its null is the standardized-displacement normal tail, §3.6a) — callers pass a
    percentile measure name.
    """
    bands = cfg.bands_for(name)
    is_liq = name == "liq"
    out: list[ARLBudget] = []
    for state, (enter, exit_) in bands.pairs():
        p_cross = _p_cross(enter, is_liq)
        p_stay = _p_cross(exit_, is_liq)
        alpha_eff = p_cross * (p_stay ** max(cfg.d_min - 1, 0))
        arl0 = (1.0 / alpha_eff) if alpha_eff > 0 else float("inf")
        out.append(ARLBudget(state=state, alpha_eff=alpha_eff, arl0=arl0))
    return tuple(out)
