"""``OpsEngine.on_obs`` — the single Protocol entry point composing every OPS trigger (OPS.md sec 8/12).

sec 8 names ``ops/brake/engine.py`` as the module holding ``OpsEngine.on_obs()``, "the single Protocol
entry point"; sec 12's cross-consistency gate requires a concrete ``OpsEngine.on_obs`` that satisfies
``core/protocols.py::OpsEngine`` (``on_obs(self, obs: object) -> OpsDirective`` plus a ``directive()``
accessor). This is the seam EXECUTION.md sec 7.2 feeds realized ``IS_total`` through and the seam that
closes sec 9's disclosed ``OpsDirective`` collision — not just documenting the reconciled dataclass,
but showing the Protocol is actually satisfiable by an implementation of THIS spec's shape.

Composition (P3, sec 4.4/5.3/1.4): one causal :class:`~convexity.ops.types.OpsObs` per call may carry
any subset of {a settled weekly return, a daily reconciliation snapshot, a realized rebalance
fill/slippage, a heartbeat}. Each is turned into its alarm(s) by the proven sub-modules
(:class:`~convexity.ops.brake.kill_switch.KillSwitch` for the Sharpe-CUSUM, ``streams.recon`` /
``streams.fills`` for the Shewhart-style immediate triggers) and combined by max-severity precedence
in :func:`~convexity.ops.brake.triggers.combine`. The engine reimplements NO detector logic — it is a
thin composition + latch layer (doctrine REUSE-FIRST).

HALT latch (Inv MONO / sec 6 step 5): once ANY source drives the engine to HALT, it stays HALT until
an explicit :meth:`rearm` — the kill-switch's own HALT latch (sec 4.4) plus an engine-level latch for
the Shewhart triggers (recon breach / SLA severe / heartbeat silence), which have no CUSUM state of
their own to latch. DEGRADE's confirmation-window auto-recovery (sec 4.4) applies only inside the
kill-switch, never to a HALT.
"""

from __future__ import annotations

from convexity.core.types import Alarm, OpsDirective, OpsState
from convexity.ops.brake.kill_switch import KillSwitch, flatten
from convexity.ops.brake.triggers import combine
from convexity.ops.config.defaults import OpsDefaults
from convexity.ops.streams.fills import evaluate_sla
from convexity.ops.streams.recon import reconcile_book
from convexity.ops.types import OpsObs


class OpsEngine:
    """Concrete ``core.protocols.OpsEngine`` — the composing kill-switch/recon/SLA entry point.

    Satisfies the ``on_obs(self, obs: object) -> OpsDirective`` + ``directive() -> OpsDirective``
    Protocol (sec 12 cross-consistency gate). Construct from an :class:`OpsDefaults` (RED constants,
    Inv C — injected, never per-instrument) plus the AMBER null anchors ``mu_0``/``sigma_0`` and the
    execution-window ``w_exec`` (EXECUTION.md's ``Delta``, sec 5.1, read never re-derived).

    Stateful: holds the CUSUM :class:`KillSwitch` (sec 4.4 MONO recovery) and an engine-level HALT
    latch for the Shewhart triggers. Each :meth:`on_obs` folds the obs through the relevant detectors
    and returns the composed :class:`OpsDirective`; :meth:`directive` re-returns the last one.
    """

    def __init__(
        self,
        *,
        defaults: OpsDefaults,
        mu_0: float,
        sigma_0: float,
        w_exec: float,
    ) -> None:
        self._d = defaults
        self._w_exec = w_exec
        self._kill = KillSwitch(
            mu_0=mu_0,
            sigma_0=sigma_0,
            h_star=defaults.h_star,
            d=defaults.d,
            g_degrade=defaults.g_degrade,
            n_confirm=defaults.n_confirm,
        )
        # Engine-level HALT latch for the non-CUSUM (Shewhart) triggers (sec 6 step 5).
        self._halted = False
        self._halt_alarms: tuple[Alarm, ...] = ()
        self._last: OpsDirective = OpsDirective(state="RUN", gross_multiplier=1.0, alarms=(),
                                                as_of=0)

    def on_obs(self, obs: object) -> OpsDirective:
        """Fold one causal observation and emit the composed directive (OPS.md sec 3, sec 12).

        ``obs`` is typed ``object`` to match the Protocol (pillar-agnostic seam); it is expected to be
        an :class:`~convexity.ops.types.OpsObs`. Precedence and reason-chaining are delegated to
        :func:`~convexity.ops.brake.triggers.combine`; the CUSUM state comes from the stateful
        :class:`KillSwitch` (so its MONO recovery window binds), the recon/SLA alarms from the proven
        stream adapters. Once HALTED the engine stays HALTED (sec 6 step 5) — it keeps folding the
        weekly return into the CUSUM for telemetry but never de-escalates without :meth:`rearm`.
        """
        if not isinstance(obs, OpsObs):
            raise TypeError(f"OpsEngine.on_obs expects an OpsObs, got {type(obs).__name__}")

        # Advance the CUSUM kill-switch with the settled weekly return (sec 4.2-4.4), if present. The
        # KillSwitch owns its own HALT latch + MONO recovery; we read its operational state below.
        if obs.weekly_net_return is not None:
            self._kill.observe(obs.weekly_net_return, as_of=obs.as_of)
        cusum_state: OpsState = self._kill_state()

        # Shewhart-style immediate triggers (sec 1.4 recon, sec 5.3 SLA, sec 2 heartbeat) — each is a
        # certainty right now, not a statistic to accumulate; they do not wait for the weekly CUSUM.
        recon_results = None
        if obs.position_snapshots is not None:
            recon_results = reconcile_book(
                obs.position_snapshots,
                k_lot=self._d.k_lot,
                k_notional=self._d.k_notional,
                m_warn=self._d.m_warn,
                staleness_max_days=self._d.staleness_max_days,
                as_of=obs.as_of,
            )

        other: list[Alarm] = []
        if obs.rebalance_fill is not None:
            f = obs.rebalance_fill
            sla = evaluate_sla(
                completion_lag=f.completion_lag,
                w_exec=self._w_exec,
                realized_slippage=f.realized_slippage,
                expected_slippage=f.expected_slippage,
                m_slip=self._d.m_slip,
                severe_lag_multiple=self._d.sla_severe_lag_multiple,
                severe_slip_multiple=self._d.sla_severe_slip_multiple,
                as_of=obs.as_of,
            )
            if sla.alarm is not None:
                other.append(sla.alarm)
        if obs.heartbeat_silent:
            other.append(Alarm(
                kind="HEARTBEAT_SILENCE",
                detail="liveness heartbeat silent past its budget (sec 2) -> forced HALT",
                value=1.0, threshold=0.0, as_of=obs.as_of,
            ))
        # Surface the CUSUM's own alarm(s) (CUSUM_ALARM / CUSUM_WARN / ANOMALOUS_OUTPERFORMANCE) into
        # the reason chain — cusum_state already carries the CUSUM severity, so these are logged, not
        # double-counted for severity (combine() maxes over cusum_state separately).
        for a in self._kill.directive().alarms:
            other.append(a)

        directive = combine(
            recon_results=recon_results,
            other_alarms=tuple(other),
            cusum_state=cusum_state,
            g_degrade=self._d.g_degrade,
            as_of=obs.as_of,
        )

        # Engine-level HALT latch: once HALT (from ANY source), stay HALT until rearm (sec 6 step 5).
        if directive.state == "HALT":
            self._halted = True
            self._halt_alarms = directive.alarms
        if self._halted:
            directive = flatten(self._halt_alarms, as_of=obs.as_of)

        self._last = directive
        return directive

    def _kill_state(self) -> OpsState:
        """The kill-switch's current operational state (sec 4.4), read from its directive."""
        return self._kill.directive().state

    def directive(self) -> OpsDirective:
        """The most recent directive (OPS.md sec 9 accessor; part of the Protocol surface)."""
        return self._last

    def rearm(self) -> None:
        """Explicit operator re-arm — the ONLY exit from HALT (OPS.md sec 6 step 5, P7).

        Clears BOTH the engine-level Shewhart latch and the kill-switch's CUSUM latch, and resets the
        last directive to RUN. No auto-restart path reaches this (Inv MONO / HALT-ONE)."""
        self._halted = False
        self._halt_alarms = ()
        self._kill.rearm()
        self._last = OpsDirective(state="RUN", gross_multiplier=1.0, alarms=(), as_of=0)
