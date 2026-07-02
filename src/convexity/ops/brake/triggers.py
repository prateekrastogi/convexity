"""Alarm -> OpsDirective mapping by max-severity precedence (``docs/specs/OPS.md`` sec 8 ``brake/triggers.py``).

sec 8 names ``ops/brake/triggers.py`` as the module that maps the fired alarms (``RECON_BREACH`` sec
1.4, ``CUSUM_ALARM``/``CUSUM_WARN`` sec 4.4, ``SLA_BREACH``/``SLA_BREACH_SEVERE`` sec 5.3,
``HEARTBEAT_SILENCE`` sec 2) onto the single totally-ordered ``OpsDirective``. This is the seam
EXECUTION.md sec 7.2 feeds realized ``IS_total`` through (as an SLA alarm) and the seam that makes
Inv HALT-ONE concrete: every HALT-severity alarm, whatever its source, converges on the ONE
``brake.kill_switch.flatten()`` routine.

Precedence (P3, sec 4.4, sec 5.3, sec 1.4): the directive is the MAX-severity state any alarm
implies, with ``HALT > DEGRADE > RUN``. The reason chain (``alarms``) carries EVERY fired alarm, not
only the binding one, so a simultaneous recon breach + SLA severe are both logged (sec 13
idempotency). The severity a given alarm implies is a pure function of its ``kind`` — EXCEPT the
reconciliation family, whose ``RECON_BREACH`` kind spans both a WARN-tier divergence and a
HALT-tier breach (sec 9's closed ``AlarmKind`` set has no dedicated WARN kind): the reconciliation
severity is therefore taken from the :class:`ReconResult.tier`, never from the alarm kind alone
(see :func:`recon_severity`), closing the WARN/kind ambiguity at this boundary.
"""

from __future__ import annotations

from convexity.core.types import Alarm, AlarmKind, OpsDirective, OpsState
from convexity.ops.brake.kill_switch import flatten
from convexity.ops.reconcile.streams_map import ReconResult, ReconTier

__all__ = ["alarm_severity", "recon_severity", "combine", "SEVERITY_RANK"]

# Total order on the operational states (P3, sec 9): RUN < DEGRADE < HALT.
SEVERITY_RANK: dict[OpsState, int] = {"RUN": 0, "DEGRADE": 1, "HALT": 2}

# The HALT-severity alarm kinds (sec 1.4, sec 4.4, sec 5.3, sec 2) — each an immediate HALT trigger.
# NOTE: RECON_BREACH is DELIBERATELY absent: its severity is tier-dependent (WARN vs BREACH), so it
# is resolved via recon_severity(result.tier), never by kind alone (sec 9 AlarmKind ambiguity).
_HALT_KINDS: frozenset[AlarmKind] = frozenset(
    {"CUSUM_ALARM", "SLA_BREACH_SEVERE", "HEARTBEAT_SILENCE"}
)
# The DEGRADE-severity alarm kinds (sec 4.4): a CUSUM half-boundary crossing reduces gross, no halt.
_DEGRADE_KINDS: frozenset[AlarmKind] = frozenset({"CUSUM_WARN"})


def recon_severity(tier: ReconTier) -> OpsState:
    """The operational severity a reconciliation tier implies (OPS.md sec 1.4).

    BREACH (a live divergence past ``m_warn * epsilon``, or a stale read past ``staleness_max``) is an
    immediate HALT (Shewhart single-point rule, sec 1.4). WARN / STALE are surfaced but take no book
    action (RUN). This keys on the TIER, not the ``RECON_BREACH`` alarm kind, because the closed
    ``AlarmKind`` set (sec 9) reuses ``RECON_BREACH`` for the sub-severe WARN-band divergence too —
    the tier is the load-bearing severity discriminator (streams_map.ReconResult docstring).
    """
    return "HALT" if tier == "BREACH" else "RUN"


def alarm_severity(kind: AlarmKind) -> OpsState:
    """The operational severity a NON-reconciliation alarm kind implies (OPS.md sec 4.4 / 5.3 / 2).

    ``CUSUM_ALARM``/``SLA_BREACH_SEVERE``/``HEARTBEAT_SILENCE`` -> HALT; ``CUSUM_WARN`` -> DEGRADE;
    everything else (telemetry ``ANOMALOUS_OUTPERFORMANCE``, the WARN-tier ``SLA_BREACH``, and the
    reconciliation family — which must instead go through :func:`recon_severity`) -> RUN. Callers
    handle reconciliation alarms via :func:`recon_severity`; passing a ``RECON_BREACH`` here returns
    RUN by design (its severity is tier-dependent, not kind-dependent).
    """
    if kind in _HALT_KINDS:
        return "HALT"
    if kind in _DEGRADE_KINDS:
        return "DEGRADE"
    return "RUN"


def combine(
    *,
    recon_results: list[ReconResult] | None = None,
    other_alarms: tuple[Alarm, ...] = (),
    cusum_state: OpsState = "RUN",
    g_degrade: float,
    as_of: int,
) -> OpsDirective:
    """Combine every fired alarm into ONE directive by max-severity precedence (OPS.md sec 4.4/5.3/1.4).

    Inputs:
      * ``recon_results`` — the daily reconciliation ladder (sec 1.4); each BREACH tier implies HALT.
      * ``other_alarms`` — non-reconciliation alarms already fired this cycle (SLA severe/base,
        CUSUM alarm/warn, heartbeat silence, anomalous-outperformance telemetry).
      * ``cusum_state`` — the kill-switch's confirmation-gated operational state (sec 4.4), which the
        pure ``alarm_severity`` of a single CUSUM alarm cannot express (it carries the MONO recovery
        window). Passed explicitly so the engine's stateful CUSUM state binds here.

    The directive's state is the MAX over: every recon result's tier severity, every other alarm's
    kind severity, and ``cusum_state``. Its ``alarms`` tuple carries EVERY non-OK alarm (full reason
    chain, sec 13). At HALT it is produced by the single :func:`~convexity.ops.brake.kill_switch.flatten`
    routine (Inv HALT-ONE); at DEGRADE ``gross_multiplier = g_degrade``; at RUN ``1.0`` and empty
    reason chain (P3: RUN carries no alarms).
    """
    alarms: list[Alarm] = []
    worst: OpsState = "RUN"

    for res in recon_results or []:
        sev = recon_severity(res.tier)
        if res.alarm is not None:
            alarms.append(res.alarm)
        if SEVERITY_RANK[sev] > SEVERITY_RANK[worst]:
            worst = sev

    for a in other_alarms:
        alarms.append(a)
        sev = alarm_severity(a.kind)
        if SEVERITY_RANK[sev] > SEVERITY_RANK[worst]:
            worst = sev

    if SEVERITY_RANK[cusum_state] > SEVERITY_RANK[worst]:
        worst = cusum_state

    reason_chain = tuple(alarms)
    if worst == "HALT":
        return flatten(reason_chain, as_of=as_of)
    if worst == "DEGRADE":
        return OpsDirective(state="DEGRADE", gross_multiplier=g_degrade, alarms=reason_chain,
                            as_of=as_of)
    # RUN: P3 — a healthy directive carries no reason chain. Sub-severe WARN/telemetry alarms are
    # surfaced/logged by the streams + forensics layer, not attached to a RUN directive.
    return OpsDirective(state="RUN", gross_multiplier=1.0, alarms=(), as_of=as_of)
