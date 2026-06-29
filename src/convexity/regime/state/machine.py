"""The regime state machine — worst-sub-state ``min()`` + ONE global dwell clock (REGIME §3.3).

Two clocks, pinned (Invariant CLOCK):
  1. **Hysteresis is PER-SUB-STATE** — done upstream in :mod:`bands` (four independent band
     classifications). This module receives the four already-classified sub-states.
  2. **Dwell is GLOBAL, on ``s_t = min over sub-states``** — exactly ONE clock, on the global
     state's multiplier level ``g(s_t)``, NOT four per-measure clocks.

The global state ``s_t`` is the WORST (lowest-``g``) sub-state — the discrete ``min()`` (P3,
Invariant MIN). The dwell clock:
  * **Escalation** of the global ``min`` to a more-stressed state is ALWAYS allowed immediately
    (brake harder now), may jump multiple states, and resets the clock (Invariant DWELL).
  * **De-escalation** is permitted only if ``dwell_bars ≥ D_min`` AND the binding measure's own exit
    threshold is satisfied — ONE step at a time — and resets the clock.
  * A **binding-identity handoff** at UNCHANGED ``g(s_t)`` (e.g. ``corr→vol`` both STRESSED) is a
    ``hold`` with reason ``binding_handoff``; ``dwell_bars`` CONTINUES, does NOT reset (§3.3 ii) —
    the book's gross target did not move, so there is nothing for the dwell to protect against.
  * A **non-binding sub-state's oscillation** below the min is invisible to ``g(s_t)`` and never
    resets the clock (§3.3 iii) — the clock sees only the scalar ``g(s_t)``, never the four-way
    identity, which is what makes a multi-measure panel deterministic (Invariant DET).

Pure function of ``(sub_states, prev, cfg)``: no hidden state, deterministic, no future data (P6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.core.types import RegimeState
from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.state.bands import _band_pair_for, g_of
from convexity.regime.types import (
    STATE_ORDER,
    RegimeStateName,
    SubName,
    SubState,
    state_rank,
)

# Deterministic tie-break order when several sub-states share the worst rank (Invariant DET). The
# §6.1 worked example has corr the unique worst; this order only resolves exact ties.
_SUB_ORDER: tuple[SubName, ...] = ("vol", "corr", "turb", "liq")


@dataclass(frozen=True)
class MachineResult:
    """The post-machine global state before the cost gate (REGIME §3.3)."""

    state: RegimeStateName
    g_s: float
    binding: SubName
    dwell_bars: int
    transition: str  # "escalate" | "deescalate" | "hold"
    reason: str


def _worst_sub(sub_states: Mapping[SubName, SubState]) -> tuple[RegimeStateName, SubName]:
    """The worst (most-stressed) sub-state and the binding measure (the discrete ``min()``, P3)."""
    worst_state: RegimeStateName = "CALM"
    binding: SubName = "vol"
    worst_rank = -1
    for name in _SUB_ORDER:
        if name not in sub_states:
            continue
        s = sub_states[name].state
        if state_rank(s) > worst_rank:
            worst_rank = state_rank(s)
            worst_state = s
            binding = name
    return worst_state, binding


def step_machine(
    sub_states: Mapping[SubName, SubState],
    prev: RegimeState | None,
    cfg: PolicyDefaults,
) -> MachineResult:
    """One global state-machine step (REGIME §3.3). Worst-sub ``min()`` + the one global dwell clock.

    ``sub_states`` are the four PER-SUB-STATE hysteretic classifications (from :mod:`bands`); each
    carries its current measure ``value`` for the binding-measure exit-threshold check. ``prev`` is
    the last emitted :class:`RegimeState` (``None`` bootstraps at CALM with a fresh clock).
    """
    proposed_state, proposed_binding = _worst_sub(sub_states)
    proposed_g = g_of(proposed_state)

    if prev is None:
        # Bootstrap: adopt the proposed state with a fresh clock (entering bar counts as 1).
        return MachineResult(
            state=proposed_state, g_s=proposed_g, binding=proposed_binding,
            dwell_bars=1, transition="escalate" if proposed_state != "CALM" else "hold",
            reason=_reason("escalate" if proposed_state != "CALM" else "hold", proposed_state, proposed_binding),
        )

    prev_rank = state_rank(prev.state)
    prop_rank = state_rank(proposed_state)

    # ESCALATION — proposed global min is more stressed: always immediate, may jump, resets clock.
    if prop_rank > prev_rank:
        return MachineResult(
            state=proposed_state, g_s=proposed_g, binding=proposed_binding,
            dwell_bars=1, transition="escalate",
            reason=_reason("escalate", proposed_state, proposed_binding),
        )

    # UNCHANGED g(s_t) — either the same binding measure holds, or a binding-identity HANDOFF.
    if prop_rank == prev_rank:
        # Prefer the INCUMBENT binding when it is STILL at the worst rank — a tie must not
        # manufacture a spurious handoff (§6.2 t1: corr+vol both STRESSED keeps binding=corr).
        if (
            prev.binding_sub in sub_states
            and state_rank(sub_states[prev.binding_sub].state) == prop_rank
        ):
            proposed_binding = prev.binding_sub
        if proposed_binding != prev.binding_sub:
            # §3.3(ii): handoff at constant g(s_t) -> hold, dwell CONTINUES (does NOT reset).
            return MachineResult(
                state=proposed_state, g_s=proposed_g, binding=proposed_binding,
                dwell_bars=prev.dwell_bars + 1, transition="hold",
                reason="binding_handoff",
            )
        # Same binding, same level: a plain hold; dwell continues. (§3.3(iii) non-binding wiggles
        # never reach here because they don't change the worst rank / binding identity.)
        return MachineResult(
            state=proposed_state, g_s=proposed_g, binding=proposed_binding,
            dwell_bars=prev.dwell_bars + 1, transition="hold",
            reason=_reason("hold", proposed_state, proposed_binding),
        )

    # PROPOSED DE-ESCALATION (prop_rank < prev_rank). Gated by dwell AND the binding exit threshold,
    # ONE step at a time (§3.3). The binding measure here is the PREV binding's measure, whose exit
    # must be cleared; we step the GLOBAL state down by exactly one rung.
    if prev.dwell_bars < cfg.d_min:
        # Dwell-locked: hold the prev state, dwell keeps counting.
        return MachineResult(
            state=prev.state, g_s=g_of(prev.state), binding=prev.binding_sub,
            dwell_bars=prev.dwell_bars + 1, transition="hold",
            reason="dwell_locked",
        )

    # Dwell satisfied: confirm the binding measure cleared its OWN exit threshold for prev.state.
    if not _binding_exit_cleared(sub_states, prev, cfg):
        return MachineResult(
            state=prev.state, g_s=g_of(prev.state), binding=prev.binding_sub,
            dwell_bars=prev.dwell_bars + 1, transition="hold",
            reason="hold_exit_not_cleared",
        )

    one_step_down: RegimeStateName = STATE_ORDER[prev_rank - 1]  # de-escalate ONE rung only (§3.2)
    return MachineResult(
        state=one_step_down, g_s=g_of(one_step_down), binding=proposed_binding,
        dwell_bars=1, transition="deescalate",
        reason=_reason("deescalate", one_step_down, proposed_binding),
    )


def _binding_exit_cleared(
    sub_states: Mapping[SubName, SubState],
    prev: RegimeState,
    cfg: PolicyDefaults,
) -> bool:
    """True if the measure that bound ``prev.state`` has fallen below ``prev.state``'s exit mark.

    De-escalation out of ``prev.state`` requires the binding measure's value below the EXIT mark of
    the boundary that ENTERED ``prev.state`` (REGIME §3.3 (b)). The proposed sub-classification
    already encodes this per-measure; we additionally verify the prev binding measure no longer
    holds ``prev.state``.
    """
    binding = prev.binding_sub
    if binding not in sub_states:
        return True
    sub = sub_states[binding]
    bands = cfg.bands_for(binding)
    _enter, exit_mark = _band_pair_for(prev.state, bands)
    return float(sub.value) < float(exit_mark)


def _reason(transition: str, state: RegimeStateName, binding: SubName) -> str:
    """Reason code, e.g. ``regime_escalate_corr_crisis`` (REGIME §6.1)."""
    if transition == "hold":
        return f"regime_hold_{binding}_{state.lower()}"
    return f"regime_{transition}_{binding}_{state.lower()}"
