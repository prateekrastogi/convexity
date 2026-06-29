"""PromotionDecision + the convex-exit promotion gate (TREND §13/§15 promotion, Inv M).

The gate requires (§13):
  * beats plain-vol/Chandelier/symmetric/time-stop baselines on median MFE capture + giveback;
  * POSITIVE NET-of-cost expectancy ``E[Pi] > 0`` (§1.5c), costs sourced from COST.md (§6.1), with
    ``premium_bleed`` reported alongside tail capture — a finalist that beats baselines on gross
    capture but is net-negative once premium bleed is charged is REJECTED (Inv M);
  * does not worsen CVaR-95 beyond tolerance;
  * the asymmetric trail improves skew AND its realized payoff is convex/option-like (INV-CVX);
  * batch/live parity, leakage, and label-firewall pass.

New symbols inherit approved class presets; insufficient data -> ``shadow``; failed leakage/parity ->
``rejected``; symbol-specific success with non-class params stays ``shadow`` (§15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class PromotionDecision:
    """The §15 promotion record."""

    asset_class: str
    symbol: str
    status: Literal["approved", "shadow", "rejected"]
    engine_version: str
    resolved_config_hash: str
    fit_window: tuple[int, int]
    validation_window: tuple[int, int]
    reason: str
    report_summary: Mapping[str, float]


def promotion_gate(
    *,
    asset_class: str,
    symbol: str,
    engine_version: str,
    resolved_config_hash: str,
    fit_window: tuple[int, int],
    validation_window: tuple[int, int],
    e_pi_net: float,
    premium_bleed: float,
    tail_capture: float,
    beats_baselines: bool,
    convex_fit_ok: bool,
    parity_ok: bool,
    leakage_ok: bool,
    n_events: int,
    min_events: int = 50,
    cvar95_within_tol: bool = True,
) -> PromotionDecision:
    """Apply the §13/§15 gate (Inv M: NET-of-cost E[Pi] > 0, not gross skew)."""
    summary = {
        "e_pi_net": e_pi_net,
        "premium_bleed": premium_bleed,
        "tail_capture": tail_capture,
    }

    def decide() -> tuple[str, str]:
        if not (parity_ok and leakage_ok):
            return "rejected", "failed batch/live parity or leakage/label-firewall"
        if not convex_fit_ok:
            return "rejected", "realized payoff not convex/option-like (INV-CVX failed)"
        # Inv M: gross capture is not sufficient — require NET-of-cost E[Pi] > 0.
        if e_pi_net <= 0.0:
            return "rejected", (
                f"net-negative E[Pi]={e_pi_net:.4f} once premium_bleed={premium_bleed:.4f} "
                "charged (Inv M)"
            )
        if not beats_baselines or not cvar95_within_tol:
            return "shadow", "positive net E[Pi] but does not beat baselines / CVaR tolerance"
        if n_events < min_events:
            return "shadow", f"insufficient events ({n_events} < {min_events})"
        return "approved", "beats baselines, net-positive E[Pi], convex payoff, parity+leakage pass"

    status, reason = decide()
    return PromotionDecision(
        asset_class=asset_class,
        symbol=symbol,
        status=status,  # type: ignore[arg-type]
        engine_version=engine_version,
        resolved_config_hash=resolved_config_hash,
        fit_window=fit_window,
        validation_window=validation_window,
        reason=reason,
        report_summary=summary,
    )
