"""Kill state machine + triggers + alarms + safe-state (Invariant KILL, BRAKE, ALARM, SAFE, RECON)."""

from __future__ import annotations

from convexity.ops.brake.alarms import (
    AlarmRouter,
    expected_samples_between_pages,
    panel_false_page_rate,
)
from convexity.ops.brake.kill_switch import KillSwitch
from convexity.ops.brake.safe_state import safe_state_for
from convexity.ops.brake.triggers import (
    TriggerResult,
    WindowedAndRule,
    data_staleness,
    max_drawdown_backstop,
    reconciliation_mismatch,
)
from convexity.ops.streams.heartbeat import HB_CLOCK_SKEW, HB_MISS, HB_STALE
from convexity.ops.types import AlarmRecord, EquityRef, Severity, TriggerName


def _tr(trigger: TriggerName, sev: Severity) -> TriggerResult:
    return TriggerResult(
        trigger, sev,
        AlarmRecord(stream="cost_tca", detector="cusum", trigger=trigger, severity=sev,
                    statistic=5.0, limit=4.77, sample_idx=15, in_control=None),
    )


# --- KILL: precedence / max-severity ----------------------------------------------------------

def test_kill_state_is_max_severity_precedence():
    """Invariant KILL: two simultaneous triggers (DEGRADE + HALT) yield HALT (never a blended score)."""
    ks = KillSwitch()
    v = ks.step([_tr("vol_regime_break", Severity.DEGRADE),
                 _tr("max_drawdown_backstop", Severity.HALT)])
    assert v.state == "halt"


def test_trigger_only_raises_never_lowers():
    ks = KillSwitch()
    ks.step([_tr("max_drawdown_backstop", Severity.HALT)])
    v = ks.step([_tr("vol_regime_break", Severity.DEGRADE)])   # lower severity this bar
    assert v.state == "halt"                                   # stays latched (only re-arm lowers)


def test_degrade_stops_new_entries_holds_existing():
    ks = KillSwitch()
    v = ks.step([_tr("vol_regime_break", Severity.DEGRADE)])
    assert v.state == "degrade" and v.cancel_entries and not v.flatten


# --- SAFE / §9.1: flatten idempotency, retry, race-freedom ------------------------------------

def test_double_consumed_halt_issues_one_flatten_wave():
    """Invariant SAFE §9.1a: a multi-bar HALT issues exactly ONE flatten (the flatten_issued latch)."""
    ks = KillSwitch()
    v1 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert v1.flatten and v1.cancel_entries        # first HALT flattens (cancel entries first §9.1b)
    v2 = ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    assert not v2.flatten                           # latched — no duplicate wave
    assert v2.flatten_unconfirmed                   # still unconfirmed until venue reconciles flat


def test_rejected_flatten_stays_latched_with_retry_and_unconfirmed():
    """Invariant SAFE §9.1c: a rejected flatten stays latched HALT with retry + HALT-unconfirmed,
    NEVER a spurious 'flat'."""
    ks = KillSwitch(flatten_retry_interval=1, flatten_retry_max=3)
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)])     # issues flatten
    v = ks.step([_tr("reconciliation_mismatch", Severity.HALT)],
                flatten_rejected=True)
    assert v.state == "halt" and v.flatten_unconfirmed and v.flatten   # retried
    assert "flatten_unconfirmed" in v.reason_codes
    assert not ks.flatten_confirmed                                    # never spuriously 'flat'


def test_flatten_confirmed_only_on_venue_reconciled_flat():
    ks = KillSwitch()
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    v = ks.step([_tr("reconciliation_mismatch", Severity.HALT)], venue_confirmed_flat=True)
    assert ks.flatten_confirmed and not v.flatten_unconfirmed


# --- §8.4 gated re-arm ------------------------------------------------------------------------

def test_halt_does_not_auto_untrip():
    """Invariant ALARM §8.4: a HALT does not return to RUN without the window AND the acknowledgement."""
    ks = KillSwitch(m_rearm=5)
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)])
    ks.step([_tr("reconciliation_mismatch", Severity.HALT)], venue_confirmed_flat=True)
    # clean bars accrue, but no acknowledgement yet
    for _ in range(10):
        ks.step([])
    assert not ks.request_rearm()      # window cleared but no ack → blocked
    ks.acknowledge()
    assert ks.request_rearm()          # now HALT → DEGRADE
    assert ks.state == "degrade"
    assert not ks.flatten_issued       # latch cleared so a fresh HALT flattens again


def test_flapping_does_not_rearm_mid_break():
    """The §12 flapping stream: a clean-then-broken-again stream must not silently re-enter a broken
    regime — re-arm requires the full clean window."""
    ks = KillSwitch(m_rearm=5)
    ks.step([_tr("vol_regime_break", Severity.DEGRADE)])
    ks.step([])                        # one clean bar
    ks.step([_tr("vol_regime_break", Severity.DEGRADE)])   # broke again before the window
    assert not ks.request_rearm()      # cannot re-arm — clean run reset


# --- BRAKE: DD backstop above governor on a SHARED reference ----------------------------------

def test_dd_backstop_fires_at_45_not_43():
    """Invariant BRAKE §7.1a: the DD_kill (45%) HALTs strictly above the governor flat (43%) on the
    SHARED equity/peak reference."""
    assert max_drawdown_backstop(EquityRef(equity=570.0, peak=1000.0), 0.45) is None   # 43% DD
    r = max_drawdown_backstop(EquityRef(equity=540.0, peak=1000.0), 0.45)              # 46% DD
    assert r is not None and r.severity == Severity.HALT


def test_gap_through_both_thresholds_fires_after_loss():
    """§7.1a honest concession: a one-bar gap from 38% past 45% fires the kill AFTER the loss — it
    does NOT prevent gap ruin."""
    pre = max_drawdown_backstop(EquityRef(equity=620.0, peak=1000.0), 0.45)   # 38% DD, no fire
    post = max_drawdown_backstop(EquityRef(equity=500.0, peak=1000.0), 0.45)  # gapped to 50% DD
    assert pre is None and post is not None        # reactive, fires after the gap


# --- RECON / staleness ------------------------------------------------------------------------

def test_recon_mismatch_halts():
    """Invariant RECON: OPS HALTs on the mismatch flag (monitors, never re-derives)."""
    assert reconciliation_mismatch(False) is None
    r = reconciliation_mismatch(True, "pos delta")
    assert r is not None and r.severity == Severity.HALT


def test_staleness_clock_skew_halts_stale_degrades():
    assert data_staleness(HB_CLOCK_SKEW, 0, 3).severity == Severity.HALT
    assert data_staleness(HB_STALE, 0, 3).severity == Severity.DEGRADE
    assert data_staleness(HB_MISS, 1, 3) is None         # not yet at the miss count
    assert data_staleness(HB_MISS, 3, 3).severity == Severity.HALT   # reaches it


# --- ALARM: dedup + paging cadence ------------------------------------------------------------

def test_dedup_coalesces_held_breach_to_one_page():
    """Invariant ALARM: a breach held 100 bars pages ONCE (+ coalesced count), not 100 times."""
    router = AlarmRouter(dedup_window=200)
    alarm = AlarmRecord(stream="cost_tca", detector="cusum", trigger="cost_divergence",
                        severity=Severity.DEGRADE, statistic=5.0, limit=4.77, sample_idx=15,
                        in_control=None)
    pages = 0
    for _ in range(100):
        router.tick()
        pages += len(router.route([alarm]))
    assert pages == 1


def test_escalation_always_pages():
    router = AlarmRouter(dedup_window=200)
    degr = AlarmRecord("cost_tca", "cusum", "cost_divergence", Severity.DEGRADE, 5.0, 4.77, 15, None)
    halt = AlarmRecord("cost_tca", "cusum", "cost_divergence_escalated", Severity.HALT, 8.0, 8.0,
                       18, None)
    router.tick()
    assert len(router.route([degr])) == 1
    router.tick()
    assert len(router.route([halt])) == 1     # escalation to higher severity always pages


def test_panel_budget_summed_over_independent_sources():
    """Invariant ALARM §6.6: panel rate = Σ_d(1/ARL0,d) over independent sources, NOT raw stream/
    detector count."""
    # 6 independent sources at ARL0 = 2000 → rate = 6/2000 = 0.003 → ~333 samples between pages
    arl0s = [2000.0] * 6
    assert abs(panel_false_page_rate(arl0s) - 0.003) < 1e-9
    assert abs(expected_samples_between_pages(arl0s) - 1000.0 / 3.0) < 1e-6


# --- safe-state policy ------------------------------------------------------------------------

def test_safe_state_fails_toward_safe():
    assert safe_state_for("heartbeat_loss").state == "halt"
    assert safe_state_for("feed_gap_single_venue").state == "degrade"
    assert safe_state_for("feed_gap_single_venue").stop_new_entries
    assert not safe_state_for("feed_gap_single_venue").flatten   # HOLD existing, do not flatten blind
    assert safe_state_for("flatten_rejected").flatten and safe_state_for("flatten_rejected").state == "halt"


def test_windowed_and_rule_requires_both_within_window():
    rule = WindowedAndRule(m_conf=3)
    assert not rule.update(10, arm_a_breached=True, arm_b_breached=False)   # only A
    assert rule.update(12, arm_a_breached=False, arm_b_breached=True)       # B within window of A
    rule2 = WindowedAndRule(m_conf=3)
    rule2.update(10, True, False)
    assert not rule2.update(20, False, True)    # B too far from A → not confirmed
