"""DynamicHorizon — causal stopping-time path/cap state (TREND §1, §9 horizon).

Computes the path state the stopping-time contract (§1) needs, and NOTHING else: it does not decide
continue/reduce/exit (Inv A). All outputs are ``F_t``-measurable (Inv H / Inv K):

  * active clock + elapsed value under that clock (§1.2);
  * cap remaining and cap fraction used against ``tau_cap = H_sys`` (§1.6a);
  * integrated variance since entry V_t (from vol);
  * running favorable excursion M_t = max_u X_u (NOT future MFE) and giveback G_t = M_t - X_t (§1.3);
  * a cap-binding warning when ``cap_fraction_used`` exceeds the configured fraction.

``M_t`` is threaded by the engine (the running max is a causal recursion); this module composes the
clock + the supplied running max/X into the frozen :class:`HorizonState`.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Position
from convexity.trend.types import ClockState, HorizonState, VolState
from convexity.trend.config.resolve import ResolvedConfig


class DynamicHorizon:
    """Causal dynamic-horizon path/cap state (§1, §9)."""

    def elapsed(self, clock: ClockState, active: str) -> float:
        """Elapsed age under the active clock (§1.2)."""
        if active == "calendar_days":
            return clock.elapsed_calendar_ms / 86_400_000.0
        if active == "trading_sessions":
            return float(clock.elapsed_sessions)
        if active == "active_minutes":
            return float(clock.elapsed_active_minutes)
        return float(clock.bars_since_entry)

    def update(
        self,
        bars: Bars,
        idx: int,
        position: Position,
        clock: ClockState,
        vol: VolState,
        z_drift: float,
        favorable_excursion_r: float,
        adverse_excursion_r: float,
        running_x: float,
        running_max_x: float,
        cfg: ResolvedConfig,
    ) -> HorizonState:
        """Return causal horizon state at ``idx`` (§1, §9).

        ``running_x`` = X_t (side-adjusted log PnL); ``running_max_x`` = M_t (the causal favorable
        high-watermark threaded by the engine). The cap is ``cfg.tau_cap == H_sys`` under
        ``cfg.active_clock`` (§1.6a). No future MFE/MAE enters here (Inv K).
        """
        active = cfg.active_clock
        elapsed = self.elapsed(clock, active)
        cap = float(cfg.tau_cap)
        frac = float(elapsed / cap) if cap > 0 else 0.0
        remaining = max(cap - elapsed, 0.0)

        m_t = max(running_max_x, 0.0)  # M_t >= 0 in side-adjusted log units (§1.5a)
        g_t = max(m_t - running_x, 0.0)  # giveback G_t = M_t - X_t >= 0

        warning = frac >= cfg.defaults.cap_binding_frac

        return HorizonState(
            active_clock=active,  # type: ignore[arg-type]
            horizon_cap=cap,
            elapsed_value=float(elapsed),
            cap_fraction_used=float(np.clip(frac, 0.0, 1.0)),
            cap_remaining=float(remaining),
            integrated_var_since_entry=vol.integrated_var_since_entry,
            running_favorable_excursion=float(m_t),
            running_giveback=float(g_t),
            cap_binding_warning=bool(warning),
        )
