"""The binary {run, degrade, halt} kill state machine — flatten latch + retry + gated re-arm.

Spec: OPS §7 (the state machine: state = max-severity over active triggers, precedence not sum), §9.1
(idempotent + race-free flatten), §8.4 (gated re-arm). The state is the most restrictive response
among active triggers; a trigger only RAISES the state (Invariant KILL). The §9.1 degenerate paths the
asynchronous flatten creates are pinned here:

  (a) idempotency — a ``flatten_issued`` latch so a multi-trigger / multi-bar HALT issues the flatten
      EXACTLY ONCE; re-entering HALT does not re-flatten; the latch clears only on a gated re-arm.
  (b) race-freedom — on HALT, cancel working ENTRY orders FIRST, then issue the reduce-only flatten;
      a fill in the race window is reconciled into the flatten next bar (the venue stop covers it).
  (c) flatten-order failure — a REJECTED/PARTIAL flatten stays LATCHED HALT, re-attempts on the RED
      retry cadence, pages ``HALT-unconfirmed``, and NEVER reports a spurious "flat" (only a reconciled
      venue-vs-book zero-position is "confirmed flat").

Re-arm (§8.4) is deliberate + gated: HALT→DEGRADE→RUN requires the triggering detector back in control
for ``M_rearm`` consecutive samples AND (for HALT) an explicit operator acknowledgement. An automatic
re-arm on the next clean bar is forbidden (P6).
"""

from __future__ import annotations

from convexity.core.types import KillState
from convexity.ops.brake.triggers import TriggerResult
from convexity.ops.types import (
    AlarmRecord,
    KillVerdict,
    Severity,
    TriggerName,
    kill_rank,
    severity_to_kill,
)


class KillSwitch:
    """The §7 kill state machine + the §9.1 flatten discipline + §8.4 gated re-arm.

    One instance per book. ``step(triggers)`` advances one bar: takes the max-severity state over the
    active triggers, applies the flatten latch / retry, and returns a :class:`KillVerdict`. Re-arm is
    via :meth:`request_rearm` (after the gated conditions hold), never automatic.
    """

    def __init__(self, *, flatten_retry_interval: int = 1, flatten_retry_max: int = 5,
                 m_rearm: int = 10, engine_version: str = "ops-1.0.0"):
        self.flatten_retry_interval = int(flatten_retry_interval)
        self.flatten_retry_max = int(flatten_retry_max)
        self.m_rearm = int(m_rearm)
        self.engine_version = engine_version
        self.state: KillState = "run"
        self.flatten_issued = False          # §9.1a latch
        self.flatten_confirmed = False       # only a reconciled venue-vs-book zero is "flat" (§9.1c)
        self._retry_attempts = 0
        self._bars_since_retry = 0
        self._clean_run = 0                  # consecutive in-control bars for the re-arm window (§8.4)
        self._ack = False                    # operator acknowledgement for a HALT re-arm (§8.4)

    # --- the per-bar advance ---------------------------------------------------------------------

    def step(
        self,
        triggers: list[TriggerResult],
        *,
        flatten_rejected: bool = False,
        flatten_partial: bool = False,
        venue_confirmed_flat: bool = False,
    ) -> KillVerdict:
        """Advance one bar. ``triggers`` are the active §7 triggers this bar.

        ``flatten_rejected``/``flatten_partial`` report the prior bar's flatten outcome (§9.1c);
        ``venue_confirmed_flat`` is the reconciled venue-vs-book zero-position (the ONLY thing that
        marks a flatten "confirmed flat", §9.1c).
        """
        # 1) precedence: state = max-severity over active triggers; a trigger only raises (§7).
        target = self.state
        active: list[TriggerName] = []
        alarms: list[AlarmRecord] = []
        for tr in triggers:
            active.append(tr.trigger)
            alarms.append(tr.alarm)
            cand = severity_to_kill(tr.severity)
            if kill_rank(cand) > kill_rank(target):
                target = cand
        # a trigger never LOWERS the state outside a gated re-arm (§7/§8.4)
        if kill_rank(target) > kill_rank(self.state):
            self.state = target

        # 2) clean-bar bookkeeping for the re-arm window (§8.4)
        if not triggers:
            self._clean_run += 1
        else:
            self._clean_run = 0

        # 3) the §9.1 flatten discipline (only in HALT)
        flatten = False
        cancel_entries = False
        flatten_unconfirmed = False
        reasons: list[str] = [t for t in active]

        if self.state == "halt":
            if venue_confirmed_flat:
                self.flatten_confirmed = True
            if not self.flatten_issued:
                # (b) race-free ordering is enforced by the directive flags + the live loop: cancel
                # entries first, then flatten. We emit both; the adapter sequences them (§9.1b).
                cancel_entries = True
                flatten = True
                self.flatten_issued = True          # (a) latch — issue exactly once (§9.1a)
                self._retry_attempts = 0
                self._bars_since_retry = 0
            elif (flatten_rejected or flatten_partial) and not self.flatten_confirmed:
                # (c) rejected/partial: stay latched HALT, retry on the RED cadence, page unconfirmed
                self._bars_since_retry += 1
                flatten_unconfirmed = True
                if (self._bars_since_retry >= self.flatten_retry_interval
                        and self._retry_attempts < self.flatten_retry_max):
                    flatten = True
                    cancel_entries = True
                    self._retry_attempts += 1
                    self._bars_since_retry = 0
                reasons.append("flatten_unconfirmed")
                alarms.append(
                    AlarmRecord(
                        stream="recon", detector=None, trigger="reconciliation_mismatch",
                        severity=Severity.HALT, statistic=float(self._retry_attempts),
                        limit=float(self.flatten_retry_max), sample_idx=0, in_control=None,
                        message="HALT-unconfirmed: flatten rejected/partial, venue stop resting, "
                                "retrying — NEVER spurious 'flat' (§9.1c).",
                    )
                )
            elif not self.flatten_confirmed:
                # issued, not yet confirmed flat, no reject signal — still unconfirmed (§9.1c)
                flatten_unconfirmed = True
                reasons.append("flatten_unconfirmed")
        elif self.state == "degrade":
            cancel_entries = True  # stop NEW entries; hold/manage existing (§7 DEGRADE)

        return KillVerdict(
            state=self.state,
            flatten=flatten,
            cancel_entries=cancel_entries,
            triggers=tuple(active),
            reason_codes=tuple(reasons),
            alarms=tuple(alarms),
            flatten_unconfirmed=flatten_unconfirmed,
            engine_version=self.engine_version,
        )

    # --- gated re-arm (§8.4) ---------------------------------------------------------------------

    def acknowledge(self) -> None:
        """Record the operator acknowledgement required for a HALT re-arm (§8.4)."""
        self._ack = True

    def can_rearm(self) -> bool:
        """Whether the gated re-arm conditions hold (§8.4): the re-arm window cleared AND, for HALT,
        an acknowledgement. A flatten that is unconfirmed blocks re-arm (the brake is still impaired)."""
        if self.state == "halt":
            if self.flatten_issued and not self.flatten_confirmed:
                return False
            return self._clean_run >= self.m_rearm and self._ack
        if self.state == "degrade":
            return self._clean_run >= self.m_rearm
        return True

    def request_rearm(self) -> bool:
        """Step the state DOWN one level if the gated conditions hold (§8.4). Never auto on a clean bar.

        Returns True iff a transition happened. Clears the flatten latch on a HALT→DEGRADE step so a
        fresh HALT would flatten again (§9.1a). Resets the clean-run + acknowledgement so the next step
        down re-requires the full window."""
        if not self.can_rearm():
            return False
        if self.state == "halt":
            self.state = "degrade"
            self.flatten_issued = False
            self.flatten_confirmed = False
            self._ack = False
            self._clean_run = 0
            return True
        if self.state == "degrade":
            self.state = "run"
            self._clean_run = 0
            return True
        return False


__all__ = ["KillSwitch"]
