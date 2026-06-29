"""KellyCeiling — Half-Kelly RISK BUDGET converted to a leverage ceiling (LEVERAGE §5, Inv K/F/A).

Kelly outputs a RISK BUDGET, not leverage (P3). Full-Kelly leverage ``f* = μ/σ²`` is NOT
used directly: production sizing uses fractional (Half) Kelly because edge-estimation
uncertainty dominates at intraday-to-weekly horizons (P3). The pinned, reproducible
budget→leverage map (§5.1), all RETURNS-based and HORIZON-scaled to the SAME holding
horizon ``T`` as the stop (Inv S):

    f_full  = μ_hat / σ_hat²            # full-Kelly leverage (the classic μ/σ²)
    L_kelly = φ · f_full                # Half-Kelly budget AS a leverage (φ ≤ 0.5)
    σ_hat   = forecast_vol · √T         # scaled consistently with the stop's σ·√T

Because μ_hat, σ_hat are RETURNS-based, scaling the instrument's notional / contract
size / point value leaves L_kelly UNCHANGED (leverage-invariance, §5.1). φ is applied
EXACTLY ONCE, here (Inv F) — the combine engine never multiplies by φ again. μ_hat is
the ONLY edge quantity in the whole engine and is ISOLATED to this ceiling (Inv A): it
must not leak to L_stop, L_vol, L_tail, L_portfolio, L_margin, or L_liquidity.

If L_kelly exceeds the stop/tail/portfolio ceilings it does NOT bind — the intended
behaviour, because the Kelly estimate is the least trustworthy input (P3, §5.1).

NON-POSITIVE μ_hat → Kelly is SLACK, never a veto (the firewall, §2/§18). Trade /
no-trade is BREAKOUT's decision; LEVERAGE only SIZES an ALREADY-QUALIFIED signal
(§2 layer 1, §18 non-goals). A non-positive Kelly budget (``μ_hat ≤ 0`` ⇒
``f_full ≤ 0``) is the LEAST-TRUSTWORTHY edge estimate disagreeing with an entry the
firewall already cleared; per §5.1 the Kelly ceiling "does NOT bind" whenever it is
not the tightest — so a non-positive budget is returned as ``+inf`` (a NON-BINDING
ceiling), letting the MEASURABLE-RISK ceilings (stop / vol / tail / portfolio /
margin / liquidity) bind. Flooring it at 0 instead would let the engine's ``min()``
drive leverage to 0 and VETO a BREAKOUT-qualified signal on the strength of an
edge sign — exactly the edge→sizing leak P2/Inv A forbid. (The raw signed budget is
still recorded in the diagnostics for telemetry.)
"""

from __future__ import annotations

import math
from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig

_EPS: float = 1e-12


class KellyCeiling:
    """L_kelly = φ·(μ_hat/σ_hat²) (§5.1)."""

    def ceiling(
        self,
        mu_hat: float,
        sigma_hat: float,
        stop_distance: float,
        phi: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, float]]:
        """Return (L_kelly, diagnostics).

        ``mu_hat`` and ``sigma_hat`` are BOTH horizon-scaled to the holding horizon T
        (Inv S): ``sigma_hat = forecast_vol·√T``. ``phi`` is the adaptive fraction
        (§5.2), φ ≤ 0.5 ALWAYS (Inv K — asserted here). All inputs are returns-based, so
        L_kelly is invariant to the instrument's notional/contract scale (§5.1).
        ``stop_distance`` is recorded so the budget-at-risk ``L·d`` is auditable.
        """
        if phi > cfg.defaults.kelly_phi_base + 1e-9:
            raise AssertionError(
                f"Invariant K: Kelly fraction phi={phi} exceeds 0.5"
            )
        sig2 = max(float(sigma_hat) ** 2, _EPS)
        f_full = float(mu_hat) / sig2
        l_kelly = float(phi) * f_full
        # A non-positive μ_hat ⇒ a non-positive Kelly budget. Per §5.1 ("if L_kelly
        # exceeds the other ceilings it does NOT bind") and the §2/§18 firewall
        # (trade/no-trade is BREAKOUT's; LEVERAGE only SIZES an already-qualified
        # signal), the Kelly ceiling must be SLACK here — NOT a veto. Returning +inf
        # makes it non-binding so the MEASURABLE-RISK ceilings (stop/vol/tail/
        # portfolio/margin/liquidity) bind; the engine's min() therefore never sizes a
        # BREAKOUT-qualified signal to 0 on the strength of an edge SIGN (which would be
        # the edge→sizing leak P2/Inv A forbid). The raw signed budget is logged.
        l_kelly_ceiling = l_kelly if l_kelly > 0.0 else math.inf
        diag: dict[str, float] = {
            "mu_hat": float(mu_hat),
            "sigma_hat": float(sigma_hat),
            "f_full": f_full,
            "phi": float(phi),
            "L_kelly_raw": l_kelly,
            "kelly_slack": float(l_kelly <= 0.0),  # 1.0 ⇒ non-positive budget, ceiling=+inf
            "budget_at_risk": (l_kelly if l_kelly > 0.0 else 0.0) * float(stop_distance),
            "stop_distance": float(stop_distance),
        }
        return l_kelly_ceiling, diag
