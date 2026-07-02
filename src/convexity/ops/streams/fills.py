"""Realized fill / completion-time / slippage adapter + SLA predicates (``docs/specs/OPS.md`` sec 5).

sec 8's directory layout names ``ops/streams/fills.py`` as the realized fill/completion-time adapter
(sec 5) and ``ops/streams/cost_tca.py`` as the realized-slippage adapter (sec 5.2); both feed the
SLA escalation. This module holds the SLA determination (the tier + attributed alarm) as pure
predicates; mapping an ``SLA_BREACH_SEVERE`` alarm onto an ``OpsDirective(HALT)`` is the brake's job
(:mod:`convexity.ops.brake.triggers`), so a severe breach uses the SAME flatten path as every other
HALT trigger (Inv HALT-ONE).

The weekly rebalance must fill inside a bounded execution window ``W_exec`` (== EXECUTION.md sec 2.1's
``Delta``, AMBER, ~1-4h; OPS reads it, never re-derives it, sec 5.1) and its realized implementation
shortfall must stay inside a multiple of the modeled friction (sec 5.2)::

    SLA:  completion_lag       <= W_exec
    SLA:  realized_slippage    <= m_slip * expected_slippage

Escalation (sec 5.3)::

    both inside                                              -> OK
    lag > W_exec           OR slip > m_slip*expected         -> SLA_BREACH (WARN, logged, no halt)
    lag > severe*W_exec    OR slip > severe_slip*m_slip*exp  -> SLA_BREACH_SEVERE -> HALT

A severe breach escalates directly to HALT WITHOUT waiting for the next weekly CUSUM observation
(Inv SLA) — a dramatically slow or expensive rebalance is itself evidence something is structurally
wrong right now (a venue outage, an API failure mid-rebalance), not something a statistical filter
should average away before acting (sec 5.3).

``m_slip = 2x`` is decomposition-anchored (sec 5.2): of the ~13bp round-trip friction, ~10bp is the
EXACT Binance VIP0 fee (no deviation possible) and only ~3bp is the stochastic half-spread+impact
sub-component, so a 2x-on-the-total bar is looser than even a severe 3-4x miss confined to that
sub-component. All pure predicates — offline / simulatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from convexity.core.types import Alarm

SlaTier = Literal["OK", "SLA_BREACH", "SLA_BREACH_SEVERE"]


@dataclass(frozen=True)
class SlaResult:
    """One rebalance's SLA outcome (OPS.md sec 5.3).

    ``tier`` is the escalation tier; ``alarm`` is populated for BREACH / BREACH_SEVERE (the severe
    alarm is a HALT trigger, Inv SLA). ``lag_breach``/``slip_breach`` record which predicate fired
    (either alone is sufficient — the escalation is an OR, sec 5.3).
    """

    tier: SlaTier
    lag_breach: bool
    slip_breach: bool
    alarm: Alarm | None


def completion_lag_ok(completion_lag: float, w_exec: float) -> bool:
    """Predicate: the rebalance finished inside its execution-window budget (OPS.md sec 5.1).

    ``completion_lag = t_complete - t_start`` (both realized wall-clock, from Nautilus fill events).
    ``w_exec`` is EXECUTION.md's ``Delta`` (AMBER), read here, never re-derived (sec 5.1).
    """
    return completion_lag <= w_exec


def slippage_ok(realized_slippage: float, expected_slippage: float, m_slip: float) -> bool:
    """Predicate: realized implementation shortfall stayed inside ``m_slip * expected`` (OPS.md sec 5.2).

    ``realized_slippage`` is EXECUTION.md sec 7.1's ``IS_total`` (once EXECUTION exists; COST's
    realized-fill TCA is the interim proxy, sec 2 / sec 5.2). ``expected_slippage`` is the modeled
    friction the strategy was sized against (``accounting.py::round_trip_friction`` = 13bp at the
    crypto-majors default, sec 5.2). ``m_slip`` is RED.
    """
    return realized_slippage <= m_slip * expected_slippage


def evaluate_sla(
    *,
    completion_lag: float,
    w_exec: float,
    realized_slippage: float,
    expected_slippage: float,
    m_slip: float,
    severe_lag_multiple: float,
    severe_slip_multiple: float,
    as_of: int,
) -> SlaResult:
    """Full sec 5.3 SLA escalation as a pure predicate (OPS.md sec 5.3).

    Returns ``OK`` if both predicates hold; ``SLA_BREACH`` (WARN, no halt) if either the lag or the
    slippage exceeds its base budget; ``SLA_BREACH_SEVERE`` (-> HALT, Inv SLA) if either exceeds its
    severe multiple (``lag > severe_lag_multiple * W_exec`` OR
    ``slip > severe_slip_multiple * m_slip * expected``). Severe is checked FIRST so a rebalance that
    is both a base and a severe breach reports severe (max-severity precedence, sec 5.3).
    """
    lag_severe = completion_lag > severe_lag_multiple * w_exec
    slip_severe = realized_slippage > severe_slip_multiple * m_slip * expected_slippage
    if lag_severe or slip_severe:
        detail = _detail("SEVERE", completion_lag, w_exec, realized_slippage, expected_slippage,
                         m_slip, lag_severe, slip_severe)
        alarm = Alarm(
            kind="SLA_BREACH_SEVERE",
            detail=detail,
            value=completion_lag if lag_severe else realized_slippage,
            threshold=(severe_lag_multiple * w_exec) if lag_severe
            else (severe_slip_multiple * m_slip * expected_slippage),
            as_of=as_of,
        )
        return SlaResult("SLA_BREACH_SEVERE", lag_severe, slip_severe, alarm)

    lag_breach = not completion_lag_ok(completion_lag, w_exec)
    slip_breach = not slippage_ok(realized_slippage, expected_slippage, m_slip)
    if lag_breach or slip_breach:
        detail = _detail("WARN", completion_lag, w_exec, realized_slippage, expected_slippage,
                         m_slip, lag_breach, slip_breach)
        alarm = Alarm(
            kind="SLA_BREACH",
            detail=detail,
            value=completion_lag if lag_breach else realized_slippage,
            threshold=w_exec if lag_breach else (m_slip * expected_slippage),
            as_of=as_of,
        )
        return SlaResult("SLA_BREACH", lag_breach, slip_breach, alarm)

    return SlaResult("OK", False, False, None)


def _detail(sev: str, lag: float, w_exec: float, slip: float, exp: float, m_slip: float,
            lag_flag: bool, slip_flag: bool) -> str:
    parts = []
    if lag_flag:
        parts.append(f"lag={lag:.6g} vs W_exec={w_exec:.6g}")
    if slip_flag:
        parts.append(f"slip={slip:.6g} vs {m_slip:.6g}*expected={m_slip * exp:.6g}")
    return f"SLA {sev}: " + "; ".join(parts)
