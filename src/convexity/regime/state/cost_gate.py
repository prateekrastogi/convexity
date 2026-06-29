"""The rebalance-threshold cost gate (REGIME §5.2, Invariant COSTGATE).

A change in ``g_t`` moves the book's gross target, forcing trades that cost money (spread + impact +
funding, COST.md). Beyond the hysteresis + dwell of §3, the gate ensures REGIME does not rebalance
the whole book on a small EFFECTIVE move:

  target_gross_new   = g_new · book_gross_at_full           # the gross the new state implies
  delta_gross        = | target_gross_new − current_gross |
  rebalance_cost_hat = COST.round_trip_estimate(delta_gross)

  Apply the move ONLY if:
    (a) the state change passed hysteresis + dwell (§3), AND
    (b) delta_gross ≥ rebal_min_frac · current_gross         # rebalance deadband (RED)
    (c) for a DE-escalation (re-grossing UP): rebalance_cost_hat ≤ rebal_cost_cap · delta_gross.
        Escalation (cutting gross DOWN for safety) is NEVER blocked by cost — safety beats cost.
  Else: hold g_t at g_prev, log ``regime_change_cost_gated``.

``book_gross_at_full`` is the time-varying ``g=1`` gross READ from LEVERAGE (Invariant NREBUILD —
REGIME does not re-derive sizing). Clause (b) is LIVE when ``current_gross`` has DRIFTED away from
``g_prev·book_gross_at_full`` (positions closed, ceilings tightened) so a hysteresis+dwell-PASSED
step implies a sub-``rebal_min_frac`` ``delta_gross`` (the §6.3 deadband fixture).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from convexity.regime.config.defaults import PolicyDefaults
from convexity.regime.types import BookState

# A round-trip cost estimator: delta_gross (notional) -> cost (same currency). COST.md owns the real
# one (the EXECUTION/COST round_trip_estimate); the default is the §6.1 ~20bps one-way de-risk cost
# so the worked example reproduces without a COST instance (Invariant NREBUILD — injectable).
RoundTripEstimate = Callable[[float], float]
_DEFAULT_BPS: float = 20.0


def _default_round_trip(delta_gross: float) -> float:
    """~20 bps one-way de-risk cost on the moved notional (REGIME §6.1 worked example default)."""
    return (_DEFAULT_BPS / 1e4) * abs(delta_gross)


@dataclass(frozen=True)
class CostGateResult:
    """Whether the proposed ``g`` move clears the gate, plus the priced delta (REGIME §5.2)."""

    passed: bool
    g_emit: float  # g_new if passed, else g_prev (the move is deferred)
    delta_gross: float
    rebalance_cost_hat: float
    reason: str  # "" if passed; "regime_change_cost_gated" if deferred


def cost_gate(
    *,
    g_prev: float,
    g_new: float,
    transition: str,
    book: BookState,
    cfg: PolicyDefaults,
    round_trip_estimate: RoundTripEstimate | None = None,
) -> CostGateResult:
    """Apply the rebalance-threshold band to a proposed ``g_prev -> g_new`` move (REGIME §5.2).

    ``transition`` is the machine's transition (``escalate``/``deescalate``/``hold``). A ``hold`` is
    a no-op and passes trivially. Escalation is never cost-blocked (clause c does not apply);
    de-escalation must clear both the deadband (b) and the re-gross-up cost cap (c).
    """
    estimate = round_trip_estimate or _default_round_trip

    target_gross_new = g_new * book.book_gross_at_full
    delta_gross = abs(target_gross_new - book.current_gross)
    cost_hat = estimate(delta_gross)

    # A hold (no g change) never rebalances on REGIME's account.
    if transition == "hold" or g_new == g_prev:
        return CostGateResult(True, g_new, delta_gross, cost_hat, reason="")

    # Clause (b): deadband — defer a sub-threshold EFFECTIVE gross move (the §6.3 drifted case).
    if delta_gross < cfg.rebal_min_frac * book.current_gross:
        return CostGateResult(False, g_prev, delta_gross, cost_hat, reason="regime_change_cost_gated")

    # Clause (c): re-grossing UP (de-escalation) must pay its way; cutting DOWN (escalation) never
    # blocked by cost — safety beats cost on the downside (REGIME §5.2 asymmetry).
    if transition == "deescalate":
        if delta_gross > 0 and cost_hat > cfg.rebal_cost_cap * delta_gross:
            return CostGateResult(False, g_prev, delta_gross, cost_hat, reason="regime_change_cost_gated")

    return CostGateResult(True, g_new, delta_gross, cost_hat, reason="")
