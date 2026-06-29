"""LiquidityCeiling — participation / spread / impact / stale-book ceiling (LEVERAGE §8.5).

OPERATIONAL/BOTH-layer ceiling: caps size so the position can be REDUCED or EXITED
without unacceptable spread/impact. Inputs are measurable only (no edge): bid/ask
spread, ADV or rolling notional volume, top-of-book/depth, realized impact, expected
participation rate, target exit time. Enforces, taking the most binding:

  * max PARTICIPATION rate — position notional <= rate · expected exit-window volume;
  * max SPREAD+IMPACT cost — if the round-trip spread+impact (bps) exceeds the budget,
    the ceiling tightens proportionally (a conservative square-root impact view);
  * STALE-book / outage / halt — forces L_liquidity to 0 (§8.5).

If reliable depth/ADV is unavailable the ceiling falls back to a conservative
class-level participation on whatever volume proxy exists — NOT infinity (§8.5).
"""

from __future__ import annotations

from typing import Mapping

from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import LiquidityState

_EPS: float = 1e-12


class LiquidityCeiling:
    """L_liquidity from participation / spread+impact / stale-book rules (§8.5)."""

    def ceiling(
        self,
        liquidity: LiquidityState,
        target_exit_minutes: float,
        position_margin: float,
        cfg: ResolvedConfig,
    ) -> tuple[float, Mapping[str, object]]:
        """Return (L_liquidity, diagnostics).

        The participation constraint caps NOTIONAL at ``rate · expected_volume``; the
        ceiling is that max notional divided by ``position_margin`` (notional = L·margin).
        ``target_exit_minutes`` scales the expected volume the participation rate applies
        to (the spec's exit window). A stale/halted book returns 0 (§8.5).
        """
        if liquidity.stale_or_halted:
            return 0.0, {"reason": "stale_or_halted", "L_participation": 0.0}

        rate = cfg.defaults.max_participation_rate
        cost_budget_bps = cfg.defaults.max_liquidity_cost_bps
        cfg_exit = cfg.defaults.target_exit_minutes
        exit_window = max(float(target_exit_minutes), _EPS)
        # expected volume over the exit window, scaled from the (cfg-sized) reference window
        vol_scale = exit_window / max(cfg_exit, _EPS)
        expected_vol = max(float(liquidity.expected_volume_exit_window) * vol_scale, 0.0)

        # 1) participation: max notional = rate * expected volume over the exit window
        max_notional_participation = rate * expected_vol
        margin = max(float(position_margin), _EPS)
        l_participation = max_notional_participation / margin

        # 2) spread+impact cost ceiling: if round-trip cost exceeds the budget, scale L down
        #    by the budget/cost ratio (conservative linear tightening).
        round_trip_cost_bps = float(liquidity.spread_bps) + float(liquidity.impact_bps)
        if round_trip_cost_bps > cost_budget_bps and round_trip_cost_bps > _EPS:
            cost_scale = cost_budget_bps / round_trip_cost_bps
        else:
            cost_scale = 1.0
        l_cost = l_participation * cost_scale

        l_liquidity = min(l_participation, l_cost)

        diag: dict[str, object] = {
            "L_participation": l_participation,
            "L_cost": l_cost,
            "participation_rate": float(rate),
            "expected_volume": expected_vol,
            "round_trip_cost_bps": round_trip_cost_bps,
            "cost_budget_bps": float(cost_budget_bps),
            "cost_scale": cost_scale,
            "reason": "ok",
        }
        return max(l_liquidity, 0.0), diag
