"""StopCeiling — L_stop = (c − max(carry,0)) / d (LEVERAGE §8.1, Inv CARRY/S).

The NORMAL-layer single-trade loss ceiling. Inverts the leverage-stop identity
``capital_loss% ≈ L·d + carry`` for the maximum leverage that keeps the loss at the
stop within the per-position cap ``c`` (§0 P1):

    L_stop = (c − max(carry, 0)) / d

CARRY CLAMP (Inv CARRY): the carry consumed here is COST's SIGNED carry; a POSITIVE
carry (a financing/funding COST) tightens L_stop exactly as ``(c−carry)/d``; a
NEGATIVE carry (a CREDIT, e.g. a funding-earning short) is clamped to 0 and does NOT
add survival headroom. A credit is an unrealized forecast that can flip; letting it
lift L_stop above the pure capital-loss cap ``c`` would make the loss-at-stop exceed
``c·position_margin`` on the strength of a forecast (an optimistic/lookahead leak into
a HARD survival constraint, contradicting P1). The earned-carry benefit is booked only
in the net-EV gate (BREAKOUT/COST), never as leverage room here. The ``max(carry,0)``
clamp lives HERE, consuming COST's signed value (ARCHITECTURE §5.5).

STOP DISTANCE (Inv S): ``d = max(k·forecast_vol_HORIZON, conditional_MAE_quantile)``.
``forecast_vol`` is ALREADY horizon-scaled (σ·√T) by ``vol/forecast.py`` — the √T is
non-negotiable; a daily-σ stop on a multiday hold is a tight stop masquerading as a
wide one. The conditional MAE uses the regime bucket's winner-MAE quantile with an
EVENT-COUNT FLOOR (falls back to global when the bucket is thin, §8.1).
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import ConditionalMAE


class StopCeiling:
    """L_stop = (c − max(carry,0))/d (§8.1)."""

    def ceiling(
        self,
        side: int,
        entry_price: float,
        c: float,
        carry: float,
        forecast_vol: float,
        regime_bucket: str,
        cond_mae: ConditionalMAE,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (L_stop, diagnostics).

        ``forecast_vol`` is the horizon-scaled σ·√T (Inv S). ``carry`` is COST's SIGNED
        carry over the holding horizon, expressed as a FRACTION of position margin
        (credit negative); the ``max(carry,0)`` clamp is applied here (Inv CARRY).
        ``c`` is the per-position committed-margin cap in [0.12,0.15].
        """
        k = cfg.defaults.stop_vol_multiple
        vol_stop = k * float(forecast_vol)
        mae_q, bucket_used = cond_mae.quantile(regime_bucket, cfg.defaults.cond_mae_min_events)
        d = max(vol_stop, float(mae_q))

        carry_clamped = max(float(carry), 0.0)  # Inv CARRY: a credit never relaxes the stop
        numerator = float(c) - carry_clamped
        l_stop = max(numerator, 0.0) / d if d > 0.0 else 0.0

        diag: dict[str, object] = {
            "d": d,
            "vol_stop": vol_stop,
            "mae_quantile": float(mae_q),
            "mae_bucket_used": bucket_used,
            "carry_signed": float(carry),
            "carry_clamped": carry_clamped,
            "c": float(c),
            "stop_binding_component": "vol" if vol_stop >= mae_q else "cond_mae",
        }
        return l_stop, diag
