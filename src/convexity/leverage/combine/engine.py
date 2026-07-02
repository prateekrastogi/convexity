"""The ``min()`` composition of the four ceilings (LEVERAGE.md §6.1).

    G_t = min( c_t · (SR_LCB_t / σ̂_t) ,   # Kelly / vol-target sizing (§2.5)
               G_t^{cap} ,                 # capacity ceiling (§5)
               m_t · G_max ,               # drawdown governor × hard cap (§6.4)
               G_max )                     # absolute circuit breaker (§3, RED)

``min()`` only — never a weighted blend (§6.1, mirroring COST P3 "ceiling, never a score"). The
binding term is recorded on every decision (``binding_ceiling ∈ {'kelly','capacity','governor',
'hard_cap'}``, §6.1). ``G_t ≥ 0`` always (every term is ≥ 0; the Kelly term is floored at 0 by
``edge/lcb.py``'s ``max(0, ·)`` and the governor multiplier is ``∈ [0, 1]``) — never a negative or
inverted book (§8 edge cases, Invariant PROV).
"""

from __future__ import annotations

from convexity.leverage.ceilings.capacity import CapacityCeiling
from convexity.leverage.types import BindingCeiling, GovernorState, GTarget, LeverageDecision


def compose_g_t(
    g_target: GTarget,
    capacity: CapacityCeiling,
    governor: GovernorState,
    g_max: float,
) -> LeverageDecision:
    """Compose ``G_t`` as the ``min()`` of the four §6.1 ceilings (LEVERAGE.md §6.1).

    Records every ceiling term and names the numerically-smallest as ``binding_ceiling`` (always
    populated, §6.1). Ties resolve in the §6.1 declaration order (kelly, capacity, governor,
    hard_cap) so the reason code is deterministic. ``g_t`` is clamped to ``≥ 0`` defensively even
    though every input term is already non-negative (§8, Invariant PROV).
    """
    kelly_term = max(0.0, g_target.kelly_term)
    capacity_term = capacity.g_cap  # +inf when no capacity constraint (empty book)
    governor_term = governor.m_t * g_max
    hard_cap_term = g_max

    terms: list[tuple[BindingCeiling, float]] = [
        ("kelly", kelly_term),
        ("capacity", capacity_term),
        ("governor", governor_term),
        ("hard_cap", hard_cap_term),
    ]
    binding, g_t = min(terms, key=lambda kv: kv[1])
    g_t = max(0.0, float(g_t))

    reasons: list[str] = [f"binding:{binding}"]
    if governor.terminal:
        reasons.append("governor_terminal")
    if g_target.sr_lcb <= 0.0:
        reasons.append("edge_floored_zero")

    return LeverageDecision(
        g_t=g_t,
        binding_ceiling=binding,
        kelly_term=float(kelly_term),
        capacity_term=float(capacity_term),
        governor_term=float(governor_term),
        hard_cap_term=float(hard_cap_term),
        g_target=g_target,
        governor=governor,
        reason_codes=tuple(reasons),
    )
