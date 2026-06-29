"""Frozen OPS types — the shapes the SPC layer, the brake, and the forensics pass around.

Spec: OPS §2 (architectural split), §5 (monitored streams), §6 (detectors), §7 (kill state),
§8 (alarms). Pure-Python, NautilusTrader-free (composed by ``strategy/convexity_strategy.py``).
The cross-pillar boundary object ``OpsDirective`` + the ``KillState`` literal live in
``convexity.core.types``; everything *internal* to OPS (the per-stream observation, the per-detector
state, the alarm record, the in-control reference, the kill verdict) lives here.

Design rules (mirrors ``core/types.py``): frozen dataclasses, minimal explicit fields, no engine
import. Every observation is ``F_t``-measurable (Invariant CAUSAL); every limit traces to a frozen
in-control reference (Invariant STREAM/LIM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, Mapping

# Re-export the boundary types so OPS code can ``from convexity.ops.types import OpsDirective``.
from convexity.core.types import KillState, OpsDirective  # noqa: F401

# --------------------------------------------------------------------------- enums / literals

# The eight monitored streams (§5). Four statistical residual families + two binary-event families,
# plus the equity/drawdown read the DD backstop (§7.1) consumes. ``edge`` is split into its two
# sub-streams (hit-rate Bernoulli + edge-per-trade) inside ``streams/edge.py``.
StreamName = Literal[
    "fills",
    "cost_tca",
    "vol",
    "edge",
    "recon",
    "heartbeat",
    "equity",
]

# The detector families (§6). ``cusum_sprt`` is the §6.5a change-point form (v1 default for the
# edge hit-rate arm); ``sprt`` is the bare Wald test (option 1, lifetime-budgeted).
DetectorName = Literal[
    "cusum",
    "page_hinkley",
    "ewma",
    "shewhart",
    "sprt",
    "cusum_sprt",
]

# The eight circuit-breaker triggers (§7), named so the forensic chain is legible (Invariant AUDIT).
TriggerName = Literal[
    "max_drawdown_backstop",       # 7.1
    "vol_regime_break",            # 7.2
    "cost_divergence",             # 7.3
    "cost_divergence_escalated",   # 7.3 -> h2
    "fill_rate_collapse",          # 7.4
    "reconciliation_mismatch",     # 7.5
    "data_staleness",              # 7.6
    "edge_decay",                  # 7.7
    "funding_blowout",             # 7.8
]


class Severity(IntEnum):
    """The §8.1 response ladder, ordered so ``max()`` is "most restrictive" (Invariant KILL/ALARM).

    LOG < PAGE < DEGRADE < HALT. ``IntEnum`` so precedence is a plain ``max`` over active triggers
    (never a blended score — Invariant KILL).
    """

    LOG = 0
    PAGE = 1
    DEGRADE = 2
    HALT = 3


# Map the kill-machine state to/from severity ordering (RUN < DEGRADE < HALT; PAGE/LOG do not move
# the state). The ``KillState`` literal is the cross-pillar one in ``core.types``.
_KILL_RANK: Mapping[KillState, int] = {"run": 0, "degrade": 1, "halt": 2}


def kill_rank(state: KillState) -> int:
    """Ordinal of a kill state for precedence (``halt`` > ``degrade`` > ``run``)."""
    return _KILL_RANK[state]


def severity_to_kill(sev: Severity) -> KillState:
    """The kill state a trigger of this severity drives (§8.1). LOG/PAGE do not move the state."""
    if sev is Severity.HALT:
        return "halt"
    if sev is Severity.DEGRADE:
        return "degrade"
    return "run"


# --------------------------------------------------------------------------- in-control reference

@dataclass(frozen=True)
class InControlParams:
    """The FROZEN in-control reference for one stream (§5/§6, Invariant LIM/STREAM).

    ``mu0``/``sigma0`` are the validated mean/std that standardize the stream (``z=(x-mu0)/sigma0``);
    the frozen ``sigma0`` is the denominator (a live ``sigma`` is diagnostic-only, §4 TEAL). ``phi``
    is the frozen AR(p) prewhitening coefficient(s) (§6.7a); ``rho`` the measured in-control lag-1
    correlation, ``vif=(1+rho)/(1-rho)`` (§6.7b). ``tail_dof``/``tail_quantile`` capture the fat tail
    so Shewhart's ``L_s`` is the *empirical* quantile, not Gaussian 3σ (§6.7c). Every field traces to
    a SYSTEM validation-report field (loaded by ``config/in_control.py``), never a live estimate.
    """

    stream: StreamName
    mu0: float
    sigma0: float
    phi: tuple[float, ...] = ()          # AR(p) prewhitening coeffs (empty == no prewhiten)
    rho: float = 0.0                     # measured in-control lag-1 autocorrelation
    tail_dof: float | None = None        # Student-t dof of the in-control residual (None == Gaussian)
    tail_quantile: float | None = None   # empirical |z| quantile s.t. exceedance == 1/ARL0 (§6.7c)
    # Bernoulli streams (edge hit-rate) carry the validated hit-rate instead of mu0/sigma0.
    p0: float | None = None              # validated hit-rate (edge intact)
    p1: float | None = None              # decayed hit-rate worth halting for (p1 < p0)
    # The cost_tca stream carries a SECOND in-control reference for its e_funding arm (§5.2): the
    # funding residual is its own signed-bps distribution, distinct from the round-trip cost residual,
    # so it MUST be standardized against its own frozen (mu0_fund, sigma0_fund) before the §7.8 CUSUM —
    # otherwise the funding CUSUM runs a RAW-bps statistic against a σ-unit limit (h=4.77) and its ARL0
    # is not the pinned one (Invariant STREAM). None on every other stream.
    mu0_fund: float | None = None        # frozen in-control mean of e_funding (signed bps)
    sigma0_fund: float | None = None     # frozen in-control std of e_funding (>0); the funding denominator

    @property
    def vif(self) -> float:
        """AR(1) variance-inflation factor ``(1+rho)/(1-rho)`` (§6.7b). 1.0 when ``rho==0``."""
        return (1.0 + self.rho) / (1.0 - self.rho)


# --------------------------------------------------------------------------- stream observation

@dataclass(frozen=True)
class StreamObs:
    """One causal observation on a monitored stream (§5). ``F_t``-measurable (Invariant CAUSAL).

    ``value`` is the raw live-vs-validated residual (bps / log-ratio / R-multiple) for a statistical
    stream, or 0/1 for a binary-event stream (``recon``/``heartbeat``). ``sample_idx`` is the stream-
    local index (monotone). ``binary`` marks the binary-event streams (no SPC, immediate trigger).
    ``meta`` carries the §5 detail a trigger/forensic record needs (e.g. ``reconciliation_ok`` detail,
    feed age, fill-rate) without bloating the typed surface.
    """

    stream: StreamName
    sample_idx: int
    value: float
    ts: int = 0                          # int64 epoch-ms (0 when synthetic / not time-stamped)
    binary: bool = False
    meta: Mapping[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- detector state

@dataclass(frozen=True)
class DetectorState:
    """The output of one ``Detector.update(z)`` (§6, Invariant CAUSAL/ARL).

    ``in_control`` is the verdict; ``statistic`` the current chart value; ``limit`` the frozen limit
    it is tested against; ``alarm`` True iff the statistic crossed the limit *and* the detector is
    past warm-up (§6.8). ``severity`` is the trigger-defined response this detector contributes (set
    by ``reconcile/streams_map.py`` when it binds the detector to a stream). ``warming_up`` True while
    inside ``N_warm`` — the recursion still updates but the alarm is suppressed to LOG (§6.8).
    """

    detector: DetectorName
    in_control: bool
    statistic: float
    limit: float
    alarm: bool
    sample_idx: int
    warming_up: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- alarm record

@dataclass(frozen=True)
class AlarmRecord:
    """One alarm, fully decomposable + replayable (§7.1 of P7, Invariant AUDIT/ALARM).

    Carries the stream, detector, statistic value, the limit it crossed, the sample index, the
    severity, the trigger it belongs to, and the in-control parameters it was tested against — so a
    halt is reconstructable from the log alone (``forensics/replay.py``). ``coalesced_count`` records
    how many subsequent breaches the dedup window folded into this one page (§8.2).
    """

    stream: StreamName
    detector: DetectorName | None
    trigger: TriggerName
    severity: Severity
    statistic: float
    limit: float
    sample_idx: int
    in_control: InControlParams | None
    message: str = ""
    coalesced_count: int = 0


# --------------------------------------------------------------------------- equity / drawdown read

@dataclass(frozen=True)
class EquityRef:
    """The SHARED equity/peak reference the DD backstop reads (§7.1a, ARCHITECTURE §5.12).

    Both LEVERAGE's governor flat (43%) and OPS's DD_kill (45%) are computed on this SAME ``equity``
    series + running-max ``peak`` (the LEVERAGE/SYSTEM-owned ``BookState`` reference), so
    ``DD_kill > DD_governor_flat`` holds by construction (Invariant BRAKE). OPS reads it; it never
    maintains its own peak. ``drawdown`` is ``(peak-equity)/peak`` (0 at a new high).
    """

    equity: float
    peak: float

    @property
    def drawdown(self) -> float:
        if self.peak <= 0.0:
            return 0.0
        return max(0.0, (self.peak - self.equity) / self.peak)


# --------------------------------------------------------------------------- kill verdict

@dataclass(frozen=True)
class KillVerdict:
    """The kill state machine's output for one bar (§7), the substance behind ``OpsDirective``.

    ``state`` is the precedence (max-severity) state over active triggers (never a blended score —
    Invariant KILL). ``flatten``/``cancel_entries`` are the binary directives the live loop consumes
    asynchronously (P1). ``flatten_unconfirmed`` is True while a flatten has been issued but the venue
    has not reconciled to flat (§9.1c). ``triggers`` is the full active-trigger chain (Invariant
    AUDIT). ``alarms`` are the new alarms to page/log this bar.
    """

    state: KillState
    flatten: bool
    cancel_entries: bool
    triggers: tuple[TriggerName, ...]
    reason_codes: tuple[str, ...]
    alarms: tuple[AlarmRecord, ...] = ()
    flatten_unconfirmed: bool = False
    engine_version: str = "ops-1.0.0"

    def to_directive(self) -> OpsDirective:
        """Project to the cross-pillar ``OpsDirective`` the live loop consumes (P1, Invariant PROV)."""
        return OpsDirective(
            state=self.state,
            flatten=self.flatten,
            cancel_entries=self.cancel_entries,
            reason_codes=self.reason_codes,
            engine_version=self.engine_version,
        )
