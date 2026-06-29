"""The public REGIME entry point (REGIME §8, Invariant SCOPE/DET).

``RegimeEngine.regime_state`` is the one outward method (satisfies
:class:`convexity.core.protocols.RegimeEngine`). It composes the four layers:

    panel -> four measures -> per-sub-state hysteresis bands -> global min()/dwell machine
          -> cost gate -> emit RegimeState

It emits ``g_t``/``s_t`` ONLY — no order, no size, no per-position drawdown taper, no instrument
rank, no halt (Invariant SCOPE). It is causal (P6), deterministic (Invariant DET), and ``min()``-
not-product (Invariant MIN).

The engine holds the STATEFUL trailing accumulators the measures need across decision bars (the
absorption fixed-``n_t`` stratum history, the turbulence ``d_norm`` history, the ``sigma_agg`` history, and
the four per-sub-state hysteresis memories). These are causal F_t state, never future data; the
public ``regime_state`` is otherwise a pure function of ``(panel, exec_state, cost_state, book,
prev, cfg)`` given the engine's accumulated history.

The event-driven (out-of-cadence) liq path (``regime_state_event``) recomputes ONLY ``g^{liq}``
fresh and carries the FROZEN last ``g^{vol/corr/turb}``; any escalation it produces is LIQ-only and
attributed to ``liq`` (Invariant STALEATTR), behind the minute-level ``D_liq_min`` dwell (§3.6 b').
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.core.types import RegimeState
from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.config.resolve import ResolvedConfig, resolve_config
from convexity.regime.measures.absorption import AbsorptionState, compute_absorption
from convexity.regime.measures.liquidity import compute_liquidity
from convexity.regime.measures.turbulence import TurbulenceState, compute_turbulence
from convexity.regime.measures.vol_regime import VolState, compute_vol_regime
from convexity.regime.state.bands import classify_substate, g_of
from convexity.regime.state.cost_gate import RoundTripEstimate, cost_gate
from convexity.regime.state.emit import emit_state
from convexity.regime.state.machine import step_machine
from convexity.regime.types import (
    BookState,
    CostState,
    ExecState,
    Panel,
    RegimeStateName,
    SubName,
    SubState,
)


@dataclass
class RegimeEngine:
    """The composed regime overlay (REGIME §8). Holds the causal trailing accumulators.

    ``cfg`` is the resolved RED/BLUE policy (defaults to the canonical RED set). The per-sub-state
    band memories (``_sub_band``) and the cached fresh sub-states (``_last_sub_states``) carry the
    hysteresis / staleness state across calls; everything is F_t-measurable (P6).
    """

    cfg: ResolvedConfig = field(default_factory=resolve_config)
    _absorption: AbsorptionState = field(default_factory=AbsorptionState)
    _turbulence: TurbulenceState = field(default_factory=TurbulenceState)
    _vol: VolState = field(default_factory=VolState)
    _sub_band: dict[SubName, RegimeStateName] = field(
        default_factory=lambda: {"vol": "CALM", "corr": "CALM", "turb": "CALM", "liq": "CALM"}
    )
    _last_sub_states: dict[SubName, SubState] = field(default_factory=dict)
    _liq_run: int = 0  # consecutive minute bars liq has held above entry (event path dwell)

    @property
    def policy(self) -> PolicyDefaults:
        return self.cfg.policy

    # ---------------------------------------------------------------- scheduled (coarse) recompute

    def regime_state(
        self,
        panel: Panel,
        exec_state: ExecState,
        cost_state: CostState,
        book: BookState,
        prev: RegimeState | None,
        cfg: ResolvedConfig | None = None,
        *,
        round_trip_estimate: RoundTripEstimate | None = None,
    ) -> RegimeState:
        """Full four-measure + state-machine + cost-gate recompute on the regime cadence (REGIME §8).

        Pure function of the inputs given the engine's accumulated causal history; emits a
        ``RegimeState`` and nothing else (Invariant SCOPE). All four measures are FRESH here.
        """
        policy = (cfg or self.cfg).policy

        vol = compute_vol_regime(panel.returns, cfg=policy, state=self._vol)
        absorp = compute_absorption(panel.returns, cfg=policy, state=self._absorption)
        turb = compute_turbulence(panel.returns, cfg=policy, state=self._turbulence)
        liq = compute_liquidity(exec_state=exec_state, cost_state=cost_state, live=panel.instruments)

        sub_states = {
            "vol": self._classify("vol", vol.vol_pct, _tighten_vol(vol, policy), panel.asof_bar, vol.fresh, vol.diag),
            "corr": self._classify("corr", absorp.delta_ar, None, panel.asof_bar, absorp.fresh, absorp.diag),
            "turb": self._classify("turb", turb.turb_pct, None, panel.asof_bar, turb.fresh, turb.diag),
            "liq": self._classify("liq", liq.liq_stress, None, panel.asof_bar, True, liq.diag),
        }
        self._last_sub_states = dict(sub_states)

        return self._run_state_layer(
            sub_states, prev, book, policy, panel.asof_bar, round_trip_estimate, binding_override=None
        )

    # ---------------------------------------------------------------- event-driven (per-minute) path

    def regime_state_event(
        self,
        exec_state: ExecState,
        cost_state: CostState,
        book: BookState,
        prev: RegimeState,
        live: tuple[str, ...],
        cfg: ResolvedConfig | None = None,
        *,
        round_trip_estimate: RoundTripEstimate | None = None,
    ) -> RegimeState:
        """Out-of-cadence ESCALATION-only liq path (REGIME §3.5, §3.6 b', Invariant STALEATTR).

        Recomputes ONLY ``g^{liq}`` fresh; ``g^{vol/corr/turb}`` are the FROZEN last cadence values.
        An escalation can only be LIQ-bound and is attributed to ``liq`` — never to a stale measure.
        Gated by the minute-level run-length ``D_liq_min`` (a single-minute blip cannot move g_t).
        """
        policy = (cfg or self.cfg).policy
        liq = compute_liquidity(exec_state=exec_state, cost_state=cost_state, live=live)

        # Minute-level dwell: count consecutive minutes liq holds above its NORMAL entry mark.
        normal_entry = policy.bands_for("liq").normal[0]
        self._liq_run = self._liq_run + 1 if liq.liq_stress >= normal_entry else 0
        if self._liq_run < policy.d_liq_min:
            return prev  # not yet a sustained flash -> no out-of-cadence move (§3.6 b')

        liq_sub = self._classify("liq", liq.liq_stress, None, prev.asof_bar, True, liq.diag)
        frozen = {
            name: self._last_sub_states[name]
            for name in ("vol", "corr", "turb")
            if name in self._last_sub_states
        }
        frozen["liq"] = liq_sub
        sub_states = frozen

        machine = step_machine(sub_states, prev, policy)
        # STALEATTR: only an escalation is permitted out of cadence, and it MUST be liq-attributed.
        if machine.transition != "escalate":
            return prev
        gate = cost_gate(
            g_prev=prev.g_t, g_new=machine.g_s, transition="escalate", book=book, cfg=policy,
            round_trip_estimate=round_trip_estimate,
        )
        return emit_state(
            machine=machine, gate=gate, g_prev=prev.g_t, binding_override="liq", cfg=policy,
            asof_bar=prev.asof_bar, engine_version=self.cfg.engine_version,
            extra_reasons=("event_driven_liq",),
        )

    # ---------------------------------------------------------------- internals

    def _classify(
        self,
        name: SubName,
        value: float,
        forced_state: RegimeStateName | None,
        asof_bar: int,
        fresh: bool,
        diag: dict,
    ) -> SubState:
        """Per-sub-state hysteretic classification + memory update (REGIME §3.2)."""
        prev_band = self._sub_band[name]
        new_band = classify_substate(value, prev_band, self.policy.bands_for(name))
        if forced_state is not None:  # confirming inputs may only TIGHTEN (§2.1)
            from convexity.regime.types import worst_state

            new_band = worst_state(new_band, forced_state)
        self._sub_band[name] = new_band
        return SubState(
            name=name, value=value, state=new_band, g_sub=g_of(new_band),
            asof_bar=asof_bar, fresh=fresh, diag=diag,
        )

    def _run_state_layer(
        self,
        sub_states: dict[SubName, SubState],
        prev: RegimeState | None,
        book: BookState,
        policy: PolicyDefaults,
        asof_bar: int,
        round_trip_estimate: RoundTripEstimate | None,
        *,
        binding_override: SubName | None,
    ) -> RegimeState:
        """Machine (worst-sub min + dwell) -> cost gate -> emit (REGIME §3.3, §5.2, §8)."""
        machine = step_machine(sub_states, prev, policy)
        g_prev = prev.g_t if prev is not None else 1.0
        transition = machine.transition
        gate = cost_gate(
            g_prev=g_prev, g_new=machine.g_s, transition=transition, book=book, cfg=policy,
            round_trip_estimate=round_trip_estimate,
        )
        return emit_state(
            machine=machine, gate=gate, g_prev=g_prev, binding_override=binding_override,
            cfg=policy, asof_bar=asof_bar, engine_version=self.cfg.engine_version,
        )


def _tighten_vol(vol, policy: PolicyDefaults) -> RegimeStateName | None:
    """vol-of-vol + term-structure are CONFIRMING inputs that can only TIGHTEN ``g^{vol}`` (§2.1).

    When BOTH the vol-of-vol and the term-structure slope are elevated, the vol sub-state is nudged
    up one rung from CALM toward NORMAL — an elevated percentile WITH rising vov and a steep term is
    a stronger signal than the percentile alone. It never LOOSENS the percentile-driven state.
    """
    if vol.vov >= policy.vov_confirm and vol.term >= policy.term_confirm:
        return "NORMAL"
    return None
