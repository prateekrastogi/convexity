"""REGIME's public entry point: ``RegimeEngine.regime_state(panel, t) -> RegimeState`` (REGIME.md §9).

Composes the pieces of §2-§4 into the one outward method the ``convexity.core.protocols.RegimeEngine``
Protocol prescribes. The engine is CAUSAL (§0 P4): the state at week ``t`` uses only weekly
statistics up to and including week ``t``'s Friday close. The CUSUM parameters ``(mu_0, k, h)`` are a
FROZEN offline calibration (§0 P4) supplied at construction — the engine never re-estimates them from
the live stream.

Design: the engine is constructed with a frozen :class:`RegimeCalibration`, the resolved config, and
a weekly rebalance grid (``week_day_indices``: for each rebalance week ``t``, the panel row indices of
its constituent days — the SAME Friday-close grid the xsec engine runs on, §0). On first use it makes
one forward causal pass computing ``D_t``/``V_t``, stepping the four CUSUMs and the state machine, and
caches the emitted ``RegimeState`` per week. ``regime_state(panel, t)`` returns the cached record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convexity.core.types import RegimeState
from convexity.regime.measures.absorption import weekly_dispersion
from convexity.regime.measures.vol_regime import weekly_btc_vol
from convexity.regime.panel.build import RegimeInputs, build_regime_inputs
from convexity.regime.state.emit import emit
from convexity.regime.state.machine import MachineState, WeekAlarms, initial_state, step
from convexity.regime.types import RegimeCalibration, WeekStatistics


@dataclass
class RegimeEngine:
    """REGIME's outward composition surface (REGIME.md §9). Satisfies ``core.protocols.RegimeEngine``.

    ``calibration`` and ``cfg`` are frozen offline artifacts (§0 P4). ``week_day_indices`` maps each
    rebalance week ``t`` to its panel day-row indices (the daily grid within the week, §2.2). The
    ``market_symbol`` selects the column for ``V_t`` (§2.3). Results are computed lazily on first
    ``regime_state`` call in one forward causal pass and cached."""

    calibration: RegimeCalibration
    cfg: object  # ResolvedRegimeConfig (typed object to keep the Protocol import-light)
    week_day_indices: list[np.ndarray]
    market_symbol: str = "BTCUSDT"
    _states: list[RegimeState] = field(default_factory=list)
    _stats: list[WeekStatistics] = field(default_factory=list)
    _computed: bool = False

    def _run(self, panel: object) -> None:
        """One forward causal pass: weekly stats -> four CUSUMs -> state machine -> emit (§2-§4)."""
        inp: RegimeInputs = build_regime_inputs(panel, market_symbol=self.market_symbol)  # type: ignore[arg-type]
        cfg = self.cfg
        d_cal = self.calibration.dispersion
        v_cal = self.calibration.btc_vol
        n_min = cfg.defaults.n_min_dispersion  # type: ignore[attr-defined]
        h_warn_d = cfg.defaults.h_warn_frac * d_cal.h  # type: ignore[attr-defined]
        n_confirm = cfg.defaults.n_confirm  # type: ignore[attr-defined]

        machine: MachineState = initial_state()
        # Live CUSUM accumulators (§3.1/§3.3) — advanced incrementally to stay causal.
        s_d_up = s_d_down = s_v_up = 0.0
        for t, days in enumerate(self.week_day_indices):
            d_t, n_used = weekly_dispersion(inp.daily_returns, inp.listed, days, n_min=n_min)
            v_t = weekly_btc_vol(inp.daily_returns, inp.btc_col, days)

            stale = (not np.isfinite(d_t)) and (not np.isfinite(v_t))
            # Dispersion CUSUMs update ONLY on a finite D_t (§8.1: a stale week is EXCLUDED, never
            # fed as a degenerate value that would suppress the accumulator).
            if np.isfinite(d_t):
                s_d_up = max(0.0, s_d_up + (d_t - d_cal.mu_0 - d_cal.k))
                s_d_down = max(0.0, s_d_down + (d_cal.mu_0 - d_cal.k - d_t))
            if np.isfinite(v_t):
                s_v_up = max(0.0, s_v_up + (v_t - v_cal.mu_0 - v_cal.k))

            d_up = s_d_up >= d_cal.h
            d_down = s_d_down >= d_cal.h
            v_up = s_v_up >= v_cal.h
            d_warn = (s_d_up >= h_warn_d) or (s_d_down >= h_warn_d)
            # Insufficient breadth for D_t (but V_t may still be fine) -> WATCH per §8.1/§0 P6.
            insufficient_breadth = not np.isfinite(d_t)
            all_calm = (s_d_up < h_warn_d) and (s_d_down < h_warn_d) and (s_v_up < v_cal.h)

            alarms = WeekAlarms(
                d_up=d_up, d_down=d_down, v_up=v_up, d_warn=d_warn,
                stale=stale or insufficient_breadth, dd_stop=False, all_calm=all_calm,
            )
            machine, transition = step(machine, alarms, n_confirm=n_confirm)

            # Reset an alarmed CUSUM to 0 after it fires (§3.1 alarm-and-reset).
            s_stat, h_thr = _binding_stat(alarms, s_d_up, s_d_down, s_v_up, d_cal.h, v_cal.h)
            if d_up:
                s_d_up = 0.0
            if d_down:
                s_d_down = 0.0
            if v_up:
                s_v_up = 0.0

            self._stats.append(
                WeekStatistics(d_t=d_t, v_t=v_t, s_d_up=s_d_up, s_d_down=s_d_down,
                               s_v_up=s_v_up, n_days_used=n_used)
            )
            self._states.append(
                emit(machine, alarms, transition, cfg=cfg, s_stat=s_stat, h_threshold=h_thr, t=t)  # type: ignore[arg-type]
            )
        self._computed = True

    def regime_state(self, panel: object, t: int) -> RegimeState:
        """The state for rebalance week ``t`` (REGIME.md §9). Causal: uses only weeks <= ``t``."""
        if not self._computed:
            self._run(panel)
        if t < 0 or t >= len(self._states):
            raise IndexError(f"week index {t} out of range [0, {len(self._states)})")
        return self._states[t]

    def all_states(self, panel: object) -> list[RegimeState]:
        """The full causal state path (all weeks) — for validation/backtest (REGIME.md §7)."""
        if not self._computed:
            self._run(panel)
        return list(self._states)


def _binding_stat(
    a: WeekAlarms, s_d_up: float, s_d_down: float, s_v_up: float, h_d: float, h_v: float
) -> tuple[float, float]:
    """The triggering statistic value and its threshold for the reason chain (REGIME.md §4.3)."""
    if a.d_up:
        return (s_d_up, h_d)
    if a.d_down:
        return (s_d_down, h_d)
    if a.v_up:
        return (s_v_up, h_v)
    if a.d_warn:
        return (max(s_d_up, s_d_down), h_d)
    return (0.0, 0.0)
