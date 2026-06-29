"""Decision-bar / deadband / event-recompose logic (SYSTEM §3.5, Invariant CAD).

Recomposing every 1-minute bar churns the book and pays spread/impact on noise. The decision bar is
governed by three rules:

  * Event-driven recompose (always): a TREND exit/reduce, a ceiling collapse, a governor step
    change, or a new breakout qualification triggers an immediate recompose.
  * Scheduled selection recompose on cadence ``τ_sel`` (daily / session): the full
    selection↔leverage fixed point runs coarse, NOT per minute.
  * No-trade band / hysteresis: a recompose only changes a position if the new target differs from
    the held size by more than a deadband ``δ`` (in lots or risk-weight), so floating-point and
    one-lot jitter do not generate orders.

No sizing/exit math (Invariant P1) — pure scheduling + a scalar deadband filter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecomposeTrigger:
    """Why this bar is (or is not) a decision bar (Section 3.5). ``should_recompose`` is the OR of
    the event flags and the scheduled-cadence flag; ``affected_only`` is True for an event recompose
    of the affected slice only (incumbents not re-quantized unless their ceiling moved)."""

    should_recompose: bool
    scheduled: bool
    trend_event: bool
    ceiling_collapse: bool
    governor_step: bool
    new_qualification: bool
    affected_only: bool
    reason_codes: tuple[str, ...]


def decision_bar(
    *,
    scheduled_due: bool,
    trend_event: bool = False,
    ceiling_collapse: bool = False,
    governor_step: bool = False,
    new_qualification: bool = False,
) -> RecomposeTrigger:
    """Decide whether bar ``t`` triggers a recompose, and whether it is event-scoped (Section 3.5).

    ``scheduled_due`` is True when the ``τ_sel`` cadence elapses (the full fixed point runs); an
    event flag forces an immediate recompose of the affected slice only (``affected_only=True``).
    """
    events = (trend_event, ceiling_collapse, governor_step, new_qualification)
    any_event = any(events)
    should = scheduled_due or any_event
    affected_only = any_event and not scheduled_due
    reasons: list[str] = []
    if scheduled_due:
        reasons.append("scheduled_recompose")
    if trend_event:
        reasons.append("event_trend")
    if ceiling_collapse:
        reasons.append("event_ceiling_collapse")
    if governor_step:
        reasons.append("event_governor_step")
    if new_qualification:
        reasons.append("event_new_qualification")
    if not should:
        reasons.append("no_recompose")
    return RecomposeTrigger(
        should_recompose=should,
        scheduled=scheduled_due,
        trend_event=trend_event,
        ceiling_collapse=ceiling_collapse,
        governor_step=governor_step,
        new_qualification=new_qualification,
        affected_only=affected_only,
        reason_codes=tuple(reasons),
    )


def exceeds_deadband(
    current_units: int,
    target_units: int,
    current_weight: float,
    target_weight: float,
    *,
    deadband_lots: int,
    deadband_weight: float,
) -> bool:
    """True iff the target differs from the held size by MORE than the deadband δ (Invariant CAD).

    A change must clear BOTH the lot deadband AND the risk-weight deadband to generate an order, so
    one-lot jitter and floating-point wobble are both suppressed. Returns False (no order) when the
    change is within either band.
    """
    lot_delta = abs(int(target_units) - int(current_units))
    weight_delta = abs(float(target_weight) - float(current_weight))
    if lot_delta <= deadband_lots:
        return False
    if weight_delta <= deadband_weight:
        return False
    return True
