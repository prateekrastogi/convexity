"""The circuit-breaker triggers (§7) + the 2-of-2 windowed-AND cost confirmation (§6.6a).

Spec: OPS §7 (the eight triggers, each with a frozen threshold + named detector/justification +
deterministic response), §6.6a (the windowed AND-rule). Each trigger is a pure function of the
relevant detector state(s) / binary event(s) + the frozen config — no P&L (Invariant KILL). A trigger
returns a :class:`TriggerResult` (fired? + severity + the AlarmRecord substance) or ``None``.

The 2-of-2 cost AND-rule (§6.6a) is its OWN combined detector with windowed semantics: it fires only
when BOTH the CUSUM and the Page-Hinkley arm are in their breached state within a confirmation window
of ``M_conf`` consecutive samples. The escalation arm (CUSUM past ``h2``) is a SEPARATE trigger
(``cost_divergence_escalated``) with its own one-sided ARL, not part of the AND-rule's joint number.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.config.severity import severity_of
from convexity.ops.streams.heartbeat import HB_CLOCK_SKEW, HB_MISS, HB_STALE
from convexity.ops.types import (
    AlarmRecord,
    DetectorState,
    EquityRef,
    InControlParams,
    Severity,
    StreamName,
    TriggerName,
)


@dataclass(frozen=True)
class TriggerResult:
    """The outcome of evaluating one trigger (§7). ``alarm`` is the decomposable record (Invariant
    AUDIT); ``severity`` the trigger-defined response (§8.1)."""

    trigger: TriggerName
    severity: Severity
    alarm: AlarmRecord


def _alarm(stream: StreamName, detector, trigger: TriggerName, st: DetectorState | None,
           ic: InControlParams | None, msg: str) -> AlarmRecord:
    sev = severity_of(trigger)
    return AlarmRecord(
        stream=stream,
        detector=detector,
        trigger=trigger,
        severity=sev,
        statistic=(st.statistic if st is not None else 0.0),
        limit=(st.limit if st is not None else 0.0),
        sample_idx=(st.sample_idx if st is not None else 0),
        in_control=ic,
        message=msg,
    )


class WindowedAndRule:
    """The 2-of-2 windowed-AND confirmation (§6.6a): both arms breached within ``M_conf`` samples.

    Tracks, per arm, the sample index of the most recent breach; fires when the two most-recent
    breaches are within ``M_conf`` of each other (and both have occurred). CUSUM and PH integrate at
    slightly different rates, so requiring an identical breach sample would miss real breaks — a short
    window keeps the false-halt reduction without blinding (§6.6a). ``M_conf=1`` is the stricter
    same-sample setting.
    """

    def __init__(self, m_conf: int):
        self.m_conf = int(m_conf)
        self._last_a: int | None = None  # last breach sample of arm A (CUSUM)
        self._last_b: int | None = None  # last breach sample of arm B (Page-Hinkley)

    def reset(self) -> None:
        self._last_a = None
        self._last_b = None

    def update(self, sample_idx: int, arm_a_breached: bool, arm_b_breached: bool) -> bool:
        """Advance the window; return True iff both arms are breached within ``M_conf`` samples."""
        if arm_a_breached:
            self._last_a = sample_idx
        if arm_b_breached:
            self._last_b = sample_idx
        if self._last_a is None or self._last_b is None:
            return False
        return abs(self._last_a - self._last_b) < self.m_conf


# --------------------------------------------------------------------------- statistical triggers

def cost_divergence(
    cusum_st: DetectorState,
    ph_st: DetectorState,
    and_rule: WindowedAndRule,
    cusum_escalated: bool,
    ic: InControlParams,
) -> list[TriggerResult]:
    """7.3 cost/funding divergence — the 2-of-2 AND (DEGRADE) + the escalation (HALT) (§6.6a/§10).

    The DEGRADE arm fires only when the windowed AND-rule is satisfied (CUSUM confirmed by PH within
    ``M_conf``). The escalation arm fires independently when the CUSUM crosses ``h2`` (its own ARL).
    Both can be active at once; the kill machine takes the max-severity (HALT wins, §7/Invariant KILL).
    """
    out: list[TriggerResult] = []
    confirmed = and_rule.update(cusum_st.sample_idx, cusum_st.alarm, ph_st.alarm)
    if confirmed:
        out.append(TriggerResult(
            "cost_divergence", Severity.DEGRADE,
            _alarm("cost_tca", "cusum", "cost_divergence", cusum_st, ic,
                   "e_cost CUSUM confirmed by Page-Hinkley (2-of-2 AND, §6.6a)"),
        ))
    if cusum_escalated:
        out.append(TriggerResult(
            "cost_divergence_escalated", Severity.HALT,
            _alarm("cost_tca", "cusum", "cost_divergence_escalated", cusum_st, ic,
                   "e_cost CUSUM crossed h2 — escalate to HALT (§7.3/§10)"),
        ))
    return out


def vol_regime_break(ewma_st: DetectorState, cusum_st: DetectorState,
                     ic: InControlParams) -> TriggerResult | None:
    """7.2 vol-regime break — EWMA/CUSUM breach on e_vol → DEGRADE + page (§5.3)."""
    if ewma_st.alarm or cusum_st.alarm:
        st = ewma_st if ewma_st.alarm else cusum_st
        det = "ewma" if ewma_st.alarm else "cusum"
        return TriggerResult(
            "vol_regime_break", Severity.DEGRADE,
            _alarm("vol", det, "vol_regime_break", st, ic,
                   "realized vol left the band sizing assumed (§7.2)"),
        )
    return None


def fill_rate_collapse(shewhart_st: DetectorState, fillrate_realized: float, n_orders: int,
                       fillrate_floor: float, n_fr: int, exit_cannot_fill: bool,
                       ic: InControlParams) -> TriggerResult | None:
    """7.4 fill-rate collapse — fill rate < floor over N_fr orders OR e_fillrate Shewhart breach.

    DEGRADE normally; HALT when a *protective exit* cannot fill (the venue stop is then the backstop,
    P6 EXEC). The fill-rate floor is checked only once the window of ``n_fr`` orders has accrued.
    """
    floor_breached = n_orders >= n_fr and fillrate_realized < fillrate_floor
    if exit_cannot_fill:
        return TriggerResult(
            "fill_rate_collapse", Severity.HALT,
            _alarm("fills", "shewhart", "fill_rate_collapse", shewhart_st, ic,
                   "protective exit cannot fill — HALT and flatten to venue stop (§7.4)"),
        )
    if floor_breached or shewhart_st.alarm:
        det = "shewhart" if shewhart_st.alarm else None
        return TriggerResult(
            "fill_rate_collapse", Severity.DEGRADE,
            _alarm("fills", det, "fill_rate_collapse", shewhart_st, ic,
                   "realized fill rate collapsed below floor (§7.4)"),
        )
    return None


def edge_decay(cusum_sprt_st: DetectorState, edge_cusum_st: DetectorState,
               ic: InControlParams) -> TriggerResult | None:
    """7.7 edge decay — CUSUM-SPRT on hit-rate LLR > h_edge OR CUSUM on e_edge breach → DEGRADE + page.

    Stop new entries, let TREND manage open winners out, page for re-validation. Detected sample-
    efficiently by the LLR change-point detector at a pinned (lifetime, not per-epoch) false-halt
    rate (§6.5a)."""
    if cusum_sprt_st.alarm or edge_cusum_st.alarm:
        st = cusum_sprt_st if cusum_sprt_st.alarm else edge_cusum_st
        det = "cusum_sprt" if cusum_sprt_st.alarm else "cusum"
        return TriggerResult(
            "edge_decay", Severity.DEGRADE,
            _alarm("edge", det, "edge_decay", st, ic,
                   "validated edge no longer holds in production (§7.7)"),
        )
    return None


def funding_blowout(funding_cusum_st: DetectorState, near_cap: bool,
                    ic: InControlParams) -> TriggerResult | None:
    """7.8 funding blowout — CUSUM on e_funding breach OR realized funding near the venue cap on a
    paying-side book → DEGRADE (§5.2)."""
    if funding_cusum_st.alarm or near_cap:
        return TriggerResult(
            "funding_blowout", Severity.DEGRADE,
            _alarm("cost_tca", "cusum", "funding_blowout", funding_cusum_st, ic,
                   "funding regime bleeds the priced carry (§7.8)"),
        )
    return None


# --------------------------------------------------------------------------- binary triggers (no SPC)

def max_drawdown_backstop(equity: EquityRef, dd_kill: float) -> TriggerResult | None:
    """7.1 max-drawdown backstop — equity peak-to-trough ≥ DD_kill → HALT (§7.1a).

    A hard boundary (not SPC, no warm-up): it FLATTENS (not just stops sizing) and backstops a mis-
    firing governor. Read on the SAME equity/peak the governor uses so 45% > 43% holds by construction
    (Invariant BRAKE). It does NOT prevent gap-through ruin — only the venue stop + L_tail do (§7.1a).
    """
    if equity.drawdown >= dd_kill:
        return TriggerResult(
            "max_drawdown_backstop", Severity.HALT,
            AlarmRecord(
                stream="equity", detector=None, trigger="max_drawdown_backstop",
                severity=Severity.HALT, statistic=equity.drawdown, limit=dd_kill,
                sample_idx=0, in_control=None,
                message=f"drawdown {equity.drawdown:.3f} ≥ DD_kill {dd_kill:.3f} — HALT (§7.1)",
            ),
        )
    return None


def reconciliation_mismatch(mismatch: bool, detail_msg: str = "") -> TriggerResult | None:
    """7.5 reconciliation mismatch — ``reconciliation_ok == False`` → HALT (§5.5/P5, Invariant RECON).

    No "small acceptable mismatch": OPS monitors the flag (never re-derives recon) and halts."""
    if mismatch:
        return TriggerResult(
            "reconciliation_mismatch", Severity.HALT,
            AlarmRecord(
                stream="recon", detector=None, trigger="reconciliation_mismatch",
                severity=Severity.HALT, statistic=1.0, limit=1.0, sample_idx=0, in_control=None,
                message=f"book disagrees with venue — HALT and flatten to venue state. {detail_msg}",
            ),
        )
    return None


def data_staleness(code: float, hb_miss_count: int, n_hb_miss: int) -> TriggerResult | None:
    """7.6 data staleness / heartbeat / clock-skew (§5.6/P6).

    HALT on clock-skew or on ``n_hb_miss`` consecutive missed heartbeats; DEGRADE on a stale feed.
    Fail toward SAFE — silence is treated as a break. The binary heartbeat stream is unaffected by
    warm-up (§6.8(a)): a dark feed in the first bar still fires.
    """
    if code == HB_CLOCK_SKEW:
        sev, msg = Severity.HALT, "clock-skew beyond tolerance — HALT (§7.6)"
    elif code == HB_MISS:
        if hb_miss_count >= n_hb_miss:
            sev, msg = Severity.HALT, f"{hb_miss_count} consecutive missed heartbeats — HALT (§7.6)"
        else:
            return None  # not yet at the miss count
    elif code == HB_STALE:
        sev, msg = Severity.DEGRADE, "feed age beyond staleness window — DEGRADE (§7.6)"
    else:
        return None
    return TriggerResult(
        "data_staleness", sev,
        AlarmRecord(
            stream="heartbeat", detector=None, trigger="data_staleness", severity=sev,
            statistic=code, limit=0.0, sample_idx=0, in_control=None, message=msg,
        ),
    )


__all__ = [
    "TriggerResult",
    "WindowedAndRule",
    "cost_divergence",
    "vol_regime_break",
    "fill_rate_collapse",
    "edge_decay",
    "funding_blowout",
    "max_drawdown_backstop",
    "reconciliation_mismatch",
    "data_staleness",
]
