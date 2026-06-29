"""OpsEngine — streams → detectors → triggers → directive (the only place a directive is emitted).

Spec: OPS §2 (the engine wraps the layers), §3 (``OpsEngine.on_obs(obs) -> OpsDirective``), §7 (kill).
The engine subscribes to the monitored-stream observations + the binary events + the equity read,
feeds each through its bound detectors (with causal prewhitening, §6.7a), evaluates the §7 triggers,
and advances the kill state machine — emitting one :class:`KillVerdict` / :class:`OpsDirective` per bar.

It tracks NO position/account state itself (it reads the Nautilus ``Portfolio``/``Cache`` via the
adapter, and the shared equity reference) — Invariant PROV/RECON. It holds no edge/sizing/exit
parameter; its only outputs are the directive + alarms + the flatten directive consumed by the live
loop (P1). The forensic log is appended on every alarm/kill (Invariant AUDIT).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from convexity.core.types import OpsDirective
from convexity.ops.brake import triggers as trig
from convexity.ops.brake.alarms import AlarmRouter
from convexity.ops.brake.kill_switch import KillSwitch
from convexity.ops.brake.triggers import TriggerResult, WindowedAndRule
from convexity.ops.config.resolve import ResolvedOpsConfig
from convexity.ops.reconcile.streams_map import build_bindings
from convexity.ops.streams.heartbeat import HB_MISS, HB_OK
from convexity.ops.types import AlarmRecord, DetectorState, EquityRef, KillVerdict, StreamObs


@runtime_checkable
class AuditSink(Protocol):
    """The append-only audit seam the engine writes to (Invariant AUDIT/V).

    The brake layer MUST NOT import ``forensics/`` (Invariant V / §13 provenance): it depends on this
    Protocol, and the concrete ``forensics.log.ForensicLog`` (which satisfies it) is injected by
    ``ops_api``/the adapter. A no-op default sink is used when none is supplied.
    """

    def append_alarm(self, alarm: AlarmRecord) -> None: ...

    def append_kill(self, verdict: KillVerdict) -> None: ...


class _NullAuditSink:
    """A no-op :class:`AuditSink` (the engine never requires a concrete forensic log to run)."""

    def append_alarm(self, alarm: AlarmRecord) -> None:  # noqa: D102
        return None

    def append_kill(self, verdict: KillVerdict) -> None:  # noqa: D102
        return None


class OpsEngine:
    """The OPS panel engine (§2/§3). One instance per book.

    Construct from a frozen :class:`ResolvedOpsConfig`; feed observations via :meth:`on_obs` /
    :meth:`on_equity` / :meth:`on_recon` / :meth:`on_heartbeat`; call :meth:`evaluate` once per bar to
    get the directive. Detectors live in the reconcile bindings; triggers + the kill machine + the
    alarm router are owned here.
    """

    def __init__(self, cfg: ResolvedOpsConfig, *, log: AuditSink | None = None):
        self.cfg = cfg
        self.bindings = build_bindings(cfg)
        d = cfg.defaults
        self.kill = KillSwitch(
            flatten_retry_interval=d.flatten_retry_interval,
            flatten_retry_max=d.flatten_retry_max,
            m_rearm=d.m_rearm,
            engine_version=cfg.engine_version,
        )
        self.router = AlarmRouter(dedup_window=cfg.severity.dedup_window)
        self.and_rule = WindowedAndRule(d.m_conf)
        self.log: AuditSink = log if log is not None else _NullAuditSink()

        # latest detector states per (stream, role)
        self._states: dict[tuple[str, str], DetectorState] = {}
        # binary-event bookkeeping
        self._equity: EquityRef | None = None
        self._recon_mismatch = False
        self._recon_detail = ""
        self._hb_code = HB_OK
        self._hb_miss_count = 0
        self._fillrate_realized = 1.0
        self._n_orders = 0
        self._exit_cannot_fill = False
        self._funding_near_cap = False
        # flatten outcome reported back from the live loop (§9.1c)
        self._flatten_rejected = False
        self._flatten_partial = False
        self._venue_confirmed_flat = False

    # --- observation intake ----------------------------------------------------------------------

    def on_obs(self, obs: StreamObs) -> None:
        """Feed one statistical-stream observation through its bound detectors (§5/§6).

        Routes the raw residual through the stream's causal prewhitener (§6.7a) then each detector.
        The ``meta['kind']`` distinguishes a stream's sub-arms (cost vs funding; hit vs edge vs freq).
        """
        binding = self.bindings.get(obs.stream)
        if binding is None:
            return
        kind = obs.meta.get("kind", 0.0)

        if obs.stream == "cost_tca":
            if kind == 0.0:  # e_cost: CUSUM + Page-Hinkley (2-of-2)
                z = binding.standardize(obs.value)
                self._states[("cost_tca", "cusum")] = binding.detectors["cusum"].update(z)
                self._states[("cost_tca", "page_hinkley")] = \
                    binding.detectors["page_hinkley"].update(z)
            else:            # e_funding: standardized against its OWN frozen (mu0_fund, sigma0_fund)
                # so the §7.8 CUSUM sees unit-variance input like every other stream (Invariant STREAM)
                # — NOT the raw bps that ran against the σ-unit limit h before.
                z = binding.standardize_funding(obs.value)
                self._states[("cost_tca", "funding_cusum")] = \
                    binding.detectors["funding_cusum"].update(z)
                self._funding_near_cap = bool(obs.meta.get("near_cap", 0.0))
            return

        if obs.stream == "vol":
            z = binding.standardize(obs.value)
            self._states[("vol", "ewma")] = binding.detectors["ewma"].update(z)
            self._states[("vol", "cusum")] = binding.detectors["cusum"].update(z)
            return

        if obs.stream == "fills":
            z = binding.standardize(obs.value)
            self._states[("fills", "cusum")] = binding.detectors["cusum"].update(z)
            self._states[("fills", "ewma")] = binding.detectors["ewma"].update(z)
            self._states[("fills", "shewhart")] = binding.detectors["shewhart"].update(z)
            return

        if obs.stream == "edge":
            if kind == 0.0:   # hit: CUSUM-SPRT (Bernoulli outcome, not standardized)
                self._states[("edge", "cusum_sprt")] = \
                    binding.detectors["cusum_sprt"].update(obs.value)
            elif kind == 1.0:  # e_edge: continuous CUSUM
                z = binding.standardize(obs.value)
                self._states[("edge", "edge_cusum")] = binding.detectors["edge_cusum"].update(z)
            else:              # frequency: Shewhart on the count residual
                z = obs.value / max(self.cfg.in_control["edge"].sigma0, 1e-9)
                self._states[("edge", "freq_shewhart")] = \
                    binding.detectors["freq_shewhart"].update(z)
            return

    def on_equity(self, equity: EquityRef) -> None:
        """Read the SHARED equity/peak reference for the DD backstop (§7.1a). OPS never owns its own."""
        self._equity = equity

    def on_recon(self, mismatch: bool, detail: str = "") -> None:
        """Read SYSTEM/Nautilus's reconciliation flag (§5.5/P5). OPS never re-derives it."""
        self._recon_mismatch = mismatch
        self._recon_detail = detail

    def on_heartbeat(self, code: float) -> None:
        """Read the heartbeat/feed/clock classification (§5.6/P6). Accumulates the miss count."""
        self._hb_code = code
        if code == HB_MISS:
            self._hb_miss_count += 1
        elif code == HB_OK:
            self._hb_miss_count = 0

    def on_fill_health(self, fillrate_realized: float, n_orders: int,
                       exit_cannot_fill: bool = False) -> None:
        """Read fill-rate health for the §7.4 trigger."""
        self._fillrate_realized = fillrate_realized
        self._n_orders = n_orders
        self._exit_cannot_fill = exit_cannot_fill

    def on_flatten_outcome(self, *, rejected: bool = False, partial: bool = False,
                           venue_confirmed_flat: bool = False) -> None:
        """Report the prior bar's flatten outcome back to the kill machine (§9.1c)."""
        self._flatten_rejected = rejected
        self._flatten_partial = partial
        self._venue_confirmed_flat = venue_confirmed_flat

    # --- per-bar evaluation ----------------------------------------------------------------------

    def evaluate(self) -> KillVerdict:
        """Evaluate the §7 triggers from the current detector/binary state and advance the kill machine.

        Returns the :class:`KillVerdict`; the pages are routed through the dedup router and every alarm
        + the kill verdict are appended to the forensic log (Invariant AUDIT).
        """
        d = self.cfg.defaults
        results: list[TriggerResult] = []

        # --- binary triggers (no SPC, no warm-up; fire in the first bar, §6.8(a)) ---
        if self._equity is not None:
            r = trig.max_drawdown_backstop(self._equity, d.dd_kill)
            if r:
                results.append(r)
        r = trig.reconciliation_mismatch(self._recon_mismatch, self._recon_detail)
        if r:
            results.append(r)
        r = trig.data_staleness(self._hb_code, self._hb_miss_count, d.n_hb_miss)
        if r:
            results.append(r)

        # --- statistical triggers ---
        ic_cost = self.cfg.in_control["cost_tca"]
        cusum_st = self._states.get(("cost_tca", "cusum"))
        ph_st = self._states.get(("cost_tca", "page_hinkley"))
        if cusum_st is not None and ph_st is not None:
            cusum_det = self.bindings["cost_tca"].detectors["cusum"]
            results.extend(
                trig.cost_divergence(cusum_st, ph_st, self.and_rule,
                                     getattr(cusum_det, "escalated", False), ic_cost)
            )
        fund_st = self._states.get(("cost_tca", "funding_cusum"))
        if fund_st is not None:
            r = trig.funding_blowout(fund_st, self._funding_near_cap, ic_cost)
            if r:
                results.append(r)

        ewma_v = self._states.get(("vol", "ewma"))
        cusum_v = self._states.get(("vol", "cusum"))
        if ewma_v is not None and cusum_v is not None:
            r = trig.vol_regime_break(ewma_v, cusum_v, self.cfg.in_control["vol"])
            if r:
                results.append(r)

        she_f = self._states.get(("fills", "shewhart"))
        if she_f is not None:
            r = trig.fill_rate_collapse(she_f, self._fillrate_realized, self._n_orders,
                                        d.fillrate_floor, d.n_fr, self._exit_cannot_fill,
                                        self.cfg.in_control["fills"])
            if r:
                results.append(r)

        cs_e = self._states.get(("edge", "cusum_sprt"))
        ce_e = self._states.get(("edge", "edge_cusum"))
        if cs_e is not None and ce_e is not None:
            r = trig.edge_decay(cs_e, ce_e, self.cfg.in_control["edge"])
            if r:
                results.append(r)

        # --- advance the kill machine (§7 precedence + §9.1 flatten discipline) ---
        verdict = self.kill.step(
            results,
            flatten_rejected=self._flatten_rejected,
            flatten_partial=self._flatten_partial,
            venue_confirmed_flat=self._venue_confirmed_flat,
        )

        # --- alarm routing (dedup, §8.2) + forensic log (Invariant AUDIT) ---
        self.router.tick()
        for a in self.router.route(verdict.alarms):
            self.log.append_alarm(a)
        if verdict.state != "run":
            self.log.append_kill(verdict)

        # one-shot binary flags consumed; reset the flatten-outcome report
        self._flatten_rejected = False
        self._flatten_partial = False
        self._venue_confirmed_flat = False
        return verdict

    def directive(self) -> OpsDirective:
        """The current cross-pillar directive (P1) — projects the latest verdict for the live loop."""
        return self.evaluate().to_directive()


__all__ = ["OpsEngine"]
