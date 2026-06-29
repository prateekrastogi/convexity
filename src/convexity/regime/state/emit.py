"""Assemble the emitted ``RegimeState`` after the machine + cost gate (REGIME §3.4, §8).

REGIME's entire public output is a :class:`convexity.core.types.RegimeState` — ``g_t``/``s_t``, the
binding sub-state, the dwell count, the transition kind, the cost-gate result, and the reason chain
(Invariant SCOPE: no order, no size). ``g_t`` is in ``{1.00, 0.75, 0.50, 0.25}`` floored at 0.25
(Invariants FLOOR/CAP). When the cost gate DEFERS a move, the emitted ``g_t`` stays at ``g_prev``,
the transition is reported as ``cost_gated``, and the reason chain records it (§5.2, §6.3).
"""

from __future__ import annotations

from convexity.core.types import RegimeState, RegimeStateName
from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.state.cost_gate import CostGateResult
from convexity.regime.state.machine import MachineResult
from convexity.regime.types import G_OF_STATE, SubName


def _state_for_g(g: float) -> RegimeStateName:
    """The discrete state whose ``g(s)`` equals ``g`` (REGIME §3.1)."""
    for name, gv in G_OF_STATE.items():
        if abs(gv - g) < 1e-12:
            return name  # type: ignore[return-value]
    return "CRISIS"


def emit_state(
    *,
    machine: MachineResult,
    gate: CostGateResult,
    g_prev: float,
    binding_override: SubName | None,
    cfg: PolicyDefaults,
    asof_bar: int,
    engine_version: str,
    extra_reasons: tuple[str, ...] = (),
) -> RegimeState:
    """Build the final :class:`RegimeState` from the machine + cost-gate outputs (REGIME §3.4/§8).

    ``binding_override`` forces ``binding_sub`` to ``liq`` on an out-of-cadence (event-driven)
    escalation (Invariant STALEATTR); otherwise the machine's binding stands.
    """
    if gate.passed:
        g_t = gate.g_emit
        state = machine.state
        transition = machine.transition
        dwell = machine.dwell_bars
        reasons: tuple[str, ...] = (machine.reason,)
    else:
        # Cost gate DEFERRED the move: hold g_prev, report cost_gated (REGIME §5.2, §6.3).
        g_t = g_prev
        state = _state_for_g(g_prev)
        transition = "cost_gated"
        dwell = machine.dwell_bars
        reasons = (gate.reason,)

    g_t = max(g_t, cfg.g_floor)  # Invariants FLOOR/CAP: g_t in [0.25, 1.0]
    g_t = min(g_t, 1.0)

    binding = binding_override if binding_override is not None else machine.binding
    reasons = reasons + extra_reasons

    transition_lit = transition if transition in ("escalate", "deescalate", "hold", "cost_gated") else "hold"

    return RegimeState(
        g_t=g_t,
        state=state,
        binding_sub=binding,
        dwell_bars=dwell,
        transition=transition_lit,  # type: ignore[arg-type]
        cost_gate_passed=gate.passed,
        rebalance_cost_hat=gate.rebalance_cost_hat,
        reason_codes=reasons,
        asof_bar=asof_bar,
        engine_version=engine_version,
    )


def assert_floor(g_t: float) -> None:
    """Guard that ``g_t`` honours the floor and cap (Invariants FLOOR/CAP)."""
    if not (G_OF_STATE["CRISIS"] - 1e-12 <= g_t <= 1.0 + 1e-12):
        raise AssertionError(f"g_t={g_t} violates the [0.25, 1.0] floor/cap (Invariant FLOOR/CAP)")
