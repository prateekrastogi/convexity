"""The CUSUM kill-switch state machine + the single shared flatten routine (OPS.md sec 4.4, sec 6).

sec 4.2 standardizes the realized weekly net return into ``z_k = (r_k - mu_0)/sigma_0``; sec 4.3
folds ``z_k`` into the downward CUSUM ``C_k``; sec 4.4 maps ``C_k`` against ``h*`` to a RUN/DEGRADE/
HALT operational state with a monotone, confirmation-gated recovery (Inv MONO, P7); sec 6 defines
the ONE shared flatten routine every HALT trigger converges on (Inv HALT-ONE).

The CUSUM statistic's raw recursion decays toward 0 on good weeks (intrinsic to CUSUM, fine), but the
OPERATIONAL state does not flip back on one good observation — RUN is re-entered only after
``C_k`` stays below ``h*/2`` for ``n_confirm`` CONSECUTIVE weeks (sec 4.4). HALT never auto-recovers
(sec 6 step 5); only an explicit operator re-arm exits HALT.

Pure / offline — no live venue dependency. The actual venue-level order submission is deferred to a
Nautilus adapter (out of v1 scope, OPS.md sec 8 ``nautilus/``); :func:`flatten` here is the single
policy-level routine every trigger invokes (Inv HALT-ONE), returning the HALT directive it produces.
"""

from __future__ import annotations

from convexity.core.types import Alarm, OpsDirective, OpsState
from convexity.ops.detectors.cusum import CUSUM
from convexity.ops.streams.whiten import whiten

__all__ = ["whiten", "state_from_statistic", "flatten", "KillSwitch"]


def state_from_statistic(c_down: float, *, h_star: float) -> OpsState:
    """Map the downward CUSUM statistic ``C_k`` to the raw escalation state (OPS.md sec 4.4).

    ``C_k < h*/2`` -> RUN; ``h*/2 <= C_k < h*`` -> DEGRADE; ``C_k >= h*`` -> HALT. This is the RAW
    (statistic-only) state; the monotone confirmation-gated OPERATIONAL state is layered on by
    :class:`KillSwitch` (Inv MONO — a single good week does not flip DEGRADE/HALT back to RUN).
    """
    half = h_star / 2.0
    if c_down >= h_star:
        return "HALT"
    if c_down >= half:
        return "DEGRADE"
    return "RUN"


def flatten(alarms: tuple[Alarm, ...], *, as_of: int) -> OpsDirective:
    """The ONE shared flatten routine (OPS.md sec 6, Inv HALT-ONE).

    Every HALT trigger — ``RECON_BREACH`` (sec 1.4), ``CUSUM_ALARM`` (sec 4.4), ``SLA_BREACH_SEVERE``
    (sec 5.3), or heartbeat silence (sec 2) — converges on THIS single code path, never a per-trigger
    bespoke close routine. It produces the HALT directive (``gross_multiplier = 0.0``, the full
    reason chain), which the live loop / operator acts on by closing the whole book via the venue
    (P3). ``alarms`` is a TUPLE so simultaneous triggers are captured together (sec 13 idempotency),
    not dropped because the state machine "already" transitioned to HALT.
    """
    return OpsDirective(state="HALT", gross_multiplier=0.0, alarms=tuple(alarms), as_of=as_of)


class KillSwitch:
    """The stateful CUSUM kill-switch with monotone, confirmation-gated recovery (OPS.md sec 4.4).

    Holds the CUSUM recursion (sec 4.3) and the operational state machine (sec 4.4). Each weekly
    observation:
      1. standardizes ``r_k`` to ``z_k`` (:func:`whiten`),
      2. folds it into the downward CUSUM,
      3. reads the raw state from ``C_k`` vs ``h*``,
      4. applies the MONO rule: escalation is immediate and monotone within an episode; de-escalation
         from DEGRADE requires ``n_confirm`` consecutive weeks with raw state == RUN; HALT never
         auto-recovers (only :meth:`rearm` exits HALT).

    ``directive()`` returns the current :class:`OpsDirective`. ``gross_multiplier`` is 1.0 at RUN,
    ``g_degrade`` at DEGRADE, 0.0 at HALT (sec 9). Two-sided telemetry (``C_k^+`` -> ANOMALOUS
    OUTPERFORMANCE, sec 4.3) is surfaced as a diagnostic alarm, never a HALT.
    """

    def __init__(self, *, mu_0: float, sigma_0: float, h_star: float, d: float,
                 g_degrade: float, n_confirm: int) -> None:
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.h_star = h_star
        self.g_degrade = g_degrade
        self.n_confirm = n_confirm
        self._cusum = CUSUM(d=d)
        self._state: OpsState = "RUN"
        self._consec_healthy = 0
        self._as_of = 0
        self._alarms: tuple[Alarm, ...] = ()

    @staticmethod
    def _rank(s: OpsState) -> int:
        return {"RUN": 0, "DEGRADE": 1, "HALT": 2}[s]

    def observe(self, r_k: float, *, as_of: int) -> OpsDirective:
        """Fold one weekly net return and return the resulting directive (OPS.md sec 4.2-4.4)."""
        self._as_of = as_of
        if self._state == "HALT":
            # HALT never auto-recovers (sec 6 step 5); the statistic keeps updating for telemetry but
            # the operational state stays HALT until an explicit rearm().
            self._cusum.update(whiten(r_k, mu_0=self.mu_0, sigma_0=self.sigma_0))
            return self.directive()

        z_k = whiten(r_k, mu_0=self.mu_0, sigma_0=self.sigma_0)
        st = self._cusum.update(z_k)
        raw = state_from_statistic(st.c_down, h_star=self.h_star)

        alarms: list[Alarm] = []
        half = self.h_star / 2.0
        # Anomalous-outperformance telemetry (sec 4.3): logged, never a HALT.
        if st.c_up >= self.h_star:
            alarms.append(Alarm(
                kind="ANOMALOUS_OUTPERFORMANCE",
                detail=f"upper CUSUM C_k^+={st.c_up:.4g} >= h*={self.h_star:.4g} (diagnostic only)",
                value=st.c_up, threshold=self.h_star, as_of=as_of,
            ))

        if self._rank(raw) > self._rank(self._state):
            # Escalation: immediate and monotone (sec 4.4).
            self._state = raw
            self._consec_healthy = 0
        elif raw == "RUN":
            # Statistic is healthy this week; count toward the confirmation window (sec 4.4).
            self._consec_healthy += 1
            if self._state == "DEGRADE" and self._consec_healthy >= self.n_confirm:
                self._state = "RUN"
                self._consec_healthy = 0
        else:
            # raw is DEGRADE while state is DEGRADE (still elevated): reset the confirmation counter.
            self._consec_healthy = 0

        if self._state == "DEGRADE" and raw == "DEGRADE":
            alarms.append(Alarm(
                kind="CUSUM_WARN",
                detail=f"C_k={st.c_down:.4g} in [h*/2={half:.4g}, h*={self.h_star:.4g}) DEGRADE",
                value=st.c_down, threshold=half, as_of=as_of,
            ))
        if self._state == "HALT":
            alarms.append(Alarm(
                kind="CUSUM_ALARM",
                detail=f"C_k={st.c_down:.4g} >= h*={self.h_star:.4g} HALT",
                value=st.c_down, threshold=self.h_star, as_of=as_of,
            ))
        self._alarms = tuple(alarms)

        if self._state == "HALT":
            return flatten(self._alarms, as_of=as_of)
        return self.directive()

    def rearm(self) -> None:
        """Explicit operator re-arm — the ONLY exit from HALT (OPS.md sec 6 step 5, P7)."""
        self._state = "RUN"
        self._consec_healthy = 0
        self._alarms = ()
        self._cusum.reset()

    def directive(self) -> OpsDirective:
        """The current directive for the operational state (OPS.md sec 9)."""
        if self._state == "HALT":
            return flatten(self._alarms, as_of=self._as_of)
        gross = self.g_degrade if self._state == "DEGRADE" else 1.0
        return OpsDirective(state=self._state, gross_multiplier=gross, alarms=self._alarms,
                            as_of=self._as_of)
