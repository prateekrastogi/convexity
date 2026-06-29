"""MarginCeiling — broker margin + free-margin buffer, stressed-branch sized (LEVERAGE §8.4).

OPERATIONAL/BOTH-layer ceiling guarding the margin call. L_margin is the largest
leverage whose STRESSED margin requirement (P4 — netted portfolio margin SPIKES when
correlations break) still leaves a free-margin BUFFER (keep >= ~20-30% free beyond the
immediate requirement so a margin call never forces liquidation before the stop does).

Financing/carry over the horizon is SUBTRACTED from the usable free margin (MTF
interest for stocks; basis carry for futures) — the same carry that tightens L_stop
(§8.1). The margin requirement is BROKER-REAL (the SPAN-style ``MarginModel``,
not a fixed leverage constant): the ceiling sizes against ``stressed_requirement``,
never the calm one.
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.margin import MarginModel, financing_over_horizon
from convexity.leverage.config.resolve import ResolvedConfig

_EPS: float = 1e-12


class MarginCeiling:
    """L_margin from the STRESSED broker margin against a free-margin buffer (§8.4)."""

    def ceiling(
        self,
        instrument: str,
        margin_model: MarginModel,
        free_margin: float,
        financing_rate: float,
        horizon: float,
        position_margin: float,
        entry_price: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, float]]:
        """Return (L_margin, diagnostics).

        ``horizon`` is the holding horizon in YEARS for the financing accrual.
        ``position_margin`` is the committed margin for one unit of leverage (notional =
        L · position_margin). The leverage is solved so the STRESSED requirement on the
        implied notional plus financing fits inside the buffered free margin (P4).
        """
        buffer = cfg.defaults.free_margin_buffer
        stress = cfg.margin_stress_multiplier

        # Usable margin after holding back the buffer fraction (a margin call never forces
        # liquidation before the stop, §8.4). Financing over the horizon is reserved first.
        usable = float(free_margin) * (1.0 - buffer)

        # Stressed margin per ONE unit of leverage (notional == L·position_margin, so the
        # stressed requirement scales with L). Probe at L=1 to get the per-unit-leverage rate.
        probe_notional = float(position_margin)  # notional at L=1
        stressed_unit = margin_model.stressed_requirement(probe_notional, entry_price, stress)
        financing_unit = financing_over_horizon(probe_notional, float(financing_rate), float(horizon))
        per_unit_cost = stressed_unit + financing_unit

        l_margin = usable / per_unit_cost if per_unit_cost > _EPS else 0.0

        diag: dict[str, float] = {
            "free_margin": float(free_margin),
            "free_margin_buffer": float(buffer),
            "usable_margin": usable,
            "stress_multiplier": float(stress),
            "stressed_margin_at_L1": float(stressed_unit),
            "financing_at_L1": float(financing_unit),
            "horizon_years": float(horizon),
        }
        return l_margin, diag
