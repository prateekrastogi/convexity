"""The HEALTHY/WATCH/STRESSED/HALT state machine with causal re-arm hysteresis (REGIME.md §4).

Consumes the per-week alarm status of the three g_t-affecting CUSUMs (``S_D^+``, ``S_D^-``, ``S_V^+``
— §3.3: ``S_V^-`` is diagnostic-only and never gates g_t) plus an optional independent drawdown-stop
trigger (§5), and drives the four-state machine of §4.1 with the one-severity-level-at-a-time re-arm
rule of §4.2.

The machine is a CAUSAL, DETERMINISTIC hysteresis (§4.2) — NOT a fitted transition matrix (the
Hamilton-1989 HMM this spec rejects, §0 P3). ``n_confirm`` is a fixed RED integer, not a probability
re-estimated from data. Escalation is immediate on the alarm condition; de-escalation requires
``n_confirm`` consecutive calm weeks (all g_t-affecting CUSUMs below ``h_warn`` with no re-alarm).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from convexity.core.types import RegimeStateName
from convexity.regime.state.bands import severity


@dataclass(frozen=True)
class WeekAlarms:
    """Per-week alarm status of the three g_t-affecting statistics (REGIME.md §3.3/§4.1).

    ``d_up``/``d_down``/``v_up`` are True iff that CUSUM is at/above its FULL ``h`` (a fired alarm,
    §4.1 STRESSED trigger). ``d_warn`` is True iff either dispersion CUSUM has crossed the WATCH
    sub-threshold ``h_warn < h`` (§4.1 WATCH trigger). ``stale`` marks a data outage / insufficient
    breadth (§8.1, §0 P6 -> WATCH). ``dd_stop`` is the independent §5 drawdown-stop trigger (-> HALT).
    ``all_calm`` is True iff every g_t-affecting CUSUM is BELOW ``h_warn`` (the §4.2 de-escalation
    condition)."""

    d_up: bool
    d_down: bool
    v_up: bool
    d_warn: bool
    stale: bool
    dd_stop: bool
    all_calm: bool


@dataclass(frozen=True)
class MachineState:
    """Internal state carried across weeks (REGIME.md §4.2). ``state`` is the current label,
    ``dwell_weeks`` counts consecutive weeks in it, ``calm_streak`` counts consecutive de-escalation-
    eligible weeks toward the ``n_confirm`` re-arm (§4.2).

    ``active_axes`` is the set of distinct CUSUM axes ({``d_up``, ``d_down``, ``v_up``}) whose alarm
    is STILL UNRESOLVED — it has fired and the machine has not yet re-armed past it (§4.2). This is
    how §4.3's SEQUENTIAL-week HALT is realized under §3.1's immediate-reset rule: §3.1 zeroes an
    alarmed CUSUM's ACCUMULATOR the same week it fires, but the STATE-level stress it caused persists
    (the machine is STRESSED) until hysteresis re-arms. So when a SECOND, distinct axis alarms in a
    LATER week while the first axis's stress is still active, ``active_axes`` holds two members and the
    machine escalates to HALT — the "two statistics simultaneously in alarm" of §4.3 read at the STATE
    layer, not requiring both accumulators to cross in the identical week. A single axis re-firing
    week after week keeps ``active_axes`` a singleton -> STRESSED, never HALT on duration (§8.3)."""

    state: RegimeStateName
    dwell_weeks: int
    calm_streak: int
    active_axes: frozenset[str] = frozenset()


def initial_state() -> MachineState:
    """Fresh machine: HEALTHY, no dwell, no calm streak, no active alarm axes (REGIME.md §4.1)."""
    return MachineState(state="HEALTHY", dwell_weeks=0, calm_streak=0, active_axes=frozenset())


def _fired_axes(a: WeekAlarms) -> frozenset[str]:
    """The distinct full-``h`` CUSUM axes in alarm THIS week (REGIME.md §3.3/§4.1)."""
    axes = set()
    if a.d_up:
        axes.add("d_up")
    if a.d_down:
        axes.add("d_down")
    if a.v_up:
        axes.add("v_up")
    return frozenset(axes)


def _target_state(a: WeekAlarms, active_axes: frozenset[str]) -> RegimeStateName:
    """The severity the alarm conditions imply, accounting for still-active prior alarms (§4.1/§4.3).

    HALT if the §5 drawdown-stop fired OR TWO-OR-MORE distinct axes of {S_D^+, S_D^-, S_V^+} are in
    alarm — where "in alarm" spans BOTH this week's fresh crossings AND axes still unresolved from a
    prior week (``active_axes``), so a dispersion alarm in week t co-occurring with a vol alarm in
    week t+1 escalates to HALT (§4.3's worked example) even though §3.1 already reset S_D^+ (§3.1 vs
    §4.3 tension resolved toward §3.1's reset for the accumulator, §4.3's co-occurrence for the state).
    STRESSED if exactly one axis is (freshly or still) in alarm. WATCH on a warn-crossing or stale
    data (§4.1, §8.1, §0 P6). Else HEALTHY."""
    combined = active_axes | _fired_axes(a)
    if a.dd_stop or len(combined) >= 2:
        return "HALT"
    if len(combined) == 1:
        return "STRESSED"
    if a.d_warn or a.stale:
        return "WATCH"
    return "HEALTHY"


def step(prev: MachineState, alarms: WeekAlarms, *, n_confirm: int) -> tuple[MachineState, str]:
    """Advance the machine one week (REGIME.md §4.1 escalation + §4.2 re-arm hysteresis).

    Returns ``(new_state, transition)`` where transition is one of ``escalate``/``deescalate``/
    ``hold``. Escalation to a higher severity is IMMEDIATE (§4.1). De-escalation is ONE SEVERITY LEVEL
    AT A TIME and only after ``n_confirm`` consecutive calm weeks (§4.2: STRESSED->WATCH and
    WATCH->HEALTHY each require ``all_calm`` for ``n_confirm`` weeks). A single good week never clears
    a STRESSED/HALT state (§4.2, mirroring overlay.py's drawdown re-arm discipline). Severity is NOT
    escalated purely on how long a single alarm persists (§8.3: only CO-OCCURRENCE across distinct
    axes, tracked via ``active_axes``, not duration, escalates — this is what makes §4.3's
    sequential-week HALT reachable while §3.1's immediate accumulator reset still holds)."""
    # Axes still unresolved from prior weeks, plus any that fire this week (§4.3 co-occurrence).
    combined_axes = prev.active_axes | _fired_axes(alarms)
    target = _target_state(alarms, prev.active_axes)
    cur_sev = severity(prev.state)
    tgt_sev = severity(target)

    if tgt_sev > cur_sev:
        # Immediate escalation to the alarm-implied severity (§4.1). Reset the calm streak. Carry the
        # accumulated active axes so a still-later third-axis alarm cannot be "forgotten".
        return (
            MachineState(state=target, dwell_weeks=1, calm_streak=0, active_axes=combined_axes),
            "escalate",
        )

    # Not escalating: accumulate a calm streak only when this week is de-escalation-eligible (§4.2).
    calm = alarms.all_calm and not alarms.stale and not alarms.dd_stop
    calm_streak = prev.calm_streak + 1 if calm else 0

    if cur_sev > 0 and calm_streak >= n_confirm:
        # De-escalate exactly ONE severity level (§4.2), consuming the streak. Re-arm clears the
        # active-axis set: the stress that drove the state is now considered resolved (§4.2).
        lower = _one_level_down(prev.state)
        return (
            MachineState(state=lower, dwell_weeks=1, calm_streak=0, active_axes=frozenset()),
            "deescalate",
        )

    # Hold the current state; grow dwell and (possibly) the calm streak. Active axes PERSIST until a
    # full re-arm — a single calm week does not clear an unresolved alarm (§4.2 hysteresis).
    return (
        replace(prev, dwell_weeks=prev.dwell_weeks + 1, calm_streak=calm_streak,
                active_axes=combined_axes),
        "hold",
    )


def _one_level_down(state: RegimeStateName) -> RegimeStateName:
    """The next-less-severe state (REGIME.md §4.2 one-level-at-a-time re-arm). HALT->STRESSED->
    WATCH->HEALTHY; HEALTHY is a fixed point."""
    order: tuple[RegimeStateName, ...] = ("HEALTHY", "WATCH", "STRESSED", "HALT")
    i = order.index(state)
    return order[max(0, i - 1)]
