"""BeliefBuilder — assembles the full causal BeliefState for one bar (TREND §9 state, Inv A/F).

Composes the estimator stack (vol -> drift -> kalman -> sequential -> jump -> hazard -> horizon) into
one :class:`BeliefState`. It emits BELIEFS ONLY — no continue/reduce/exit action (Inv A) and no drift
posterior (Inv D). It is deterministic given prior state and bars up to ``idx`` (no future labels,
Inv H). It threads the streaming estimator state (integrated variance, Kalman, CUSUM/SPRT, running
M_t/X_t) so a one-bar online update reproduces the batch result (Inv I).

This is the SINGLE place the per-bar belief is built; the engine owns the warm-up / streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convexity.core.types import Bars, Position
from convexity.trend.config.asset_class import noise_ratio_for
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.estimate.drift import VarianceIntegratedDrift
from convexity.trend.estimate.hazard import TrendDeathHazard
from convexity.trend.estimate.horizon import DynamicHorizon
from convexity.trend.estimate.jump import JumpDeterioration
from convexity.trend.estimate.kalman import FixedPresetLocalTrend
from convexity.trend.estimate.sequential import SequentialDeterioration
from convexity.trend.estimate.volatility import RobustVolatility
from convexity.trend.types import (
    BeliefDiagnostics,
    BeliefState,
    ClockState,
    VolState,
)


@dataclass
class _Stream:
    """Mutable streaming state threaded across bars (Inv I parity / Inv H causality)."""

    integrated_var: float = 0.0
    running_x: float = 0.0
    running_max_x: float = 0.0
    gap_applied_at: set = field(default_factory=set)
    jump_count: int = 0
    bar_count: int = 0


class BeliefBuilder:
    """Build the full causal BeliefState (§9). Stateful across bars; reset on a new position."""

    def __init__(self, cfg: ResolvedConfig) -> None:
        self.cfg = cfg
        self.vol = RobustVolatility()
        self.drift = VarianceIntegratedDrift(eps=cfg.defaults.eps)
        self.horizon = DynamicHorizon()
        self.jump = JumpDeterioration()
        ratio = noise_ratio_for(cfg.preset.kalman_class_preset, cfg.defaults.kalman_noise_ratio)
        self.kalman = FixedPresetLocalTrend(noise_ratio=ratio)
        self.seq = SequentialDeterioration(cfg.detector_kind, cfg)  # type: ignore[arg-type]
        self.hazard = TrendDeathHazard()
        self._stream = _Stream()

    def reset(self) -> None:
        """Reset all streaming state (new position / engine warm-up)."""
        self.kalman.reset()
        self.seq.reset()
        self._stream = _Stream()

    def _clock(self, bars: Bars, idx: int, position: Position) -> ClockState:
        """Build the four-clock elapsed state (§1.2). All past-only."""
        e = position.entry_idx
        elapsed_ms = int(bars.ts[idx] - bars.ts[e]) if idx >= e else 0
        bars_since = max(idx - e, 0)
        # sessions/active-minutes counted causally from the session-open flags.
        sessions = 0
        active_minutes = bars_since  # 1-min bars: each tradable bar is one active minute
        if bars.is_session_open is not None and idx >= e:
            sessions = int(np.sum(bars.is_session_open[e + 1 : idx + 1]))
        if bars.is_valid is not None and idx >= e:
            active_minutes = int(np.sum(bars.is_valid[e + 1 : idx + 1]))
        return ClockState(
            elapsed_calendar_ms=elapsed_ms,
            elapsed_sessions=sessions,
            elapsed_active_minutes=active_minutes,
            bars_since_entry=bars_since,
        )

    def update(self, bars: Bars, idx: int, position: Position, cfg: ResolvedConfig) -> BeliefState:
        """Build the BeliefState at ``idx`` (§9). Threads streaming state for parity (Inv I)."""
        st = self._stream
        # --- volatility (threads integrated variance for monotonicity) ---
        gap_applied = idx in st.gap_applied_at
        vol = self.vol.update(
            bars, idx, position, cfg,
            prev_integrated_var=st.integrated_var,
            gap_applied=gap_applied,
        )
        st.integrated_var = vol.integrated_var_since_entry
        if vol.gap_var > 0:
            st.gap_applied_at.add(idx)

        # --- drift z + excursions ---
        z_drift, fav_r, adv_r = self.drift.update(bars, idx, position, vol)

        # --- running X_t / M_t (causal favorable high-watermark) ---
        side = float(position.side)
        pt, p0 = bars.c[idx], position.entry_price
        x_t = side * float(np.log(pt / p0)) if (pt > 0 and p0 > 0) else 0.0
        st.running_x = x_t
        st.running_max_x = max(st.running_max_x, x_t)

        # --- kalman filtered slope (FIXED class preset, Inv E) ---
        price_log = float(np.log(pt)) if pt > 0 else 0.0
        filtered_slope, slope_var = self.kalman.update(price_log, vol.sigma_log)

        # --- sequential deterioration (EXACTLY one detector, Inv B) ---
        # increment = side-adjusted standardized one-bar return (favorable positive).
        std_ret = self._side_std_return(bars, idx, position, vol)
        seq_stat, seq_alarm = self.seq.update(std_ret, vol)

        # --- jump-against / jump-with ---
        jump_against, jump_with = self.jump.update(bars, idx, position.side, vol, cfg)
        st.bar_count += 1
        if jump_against or jump_with:
            st.jump_count += 1

        # --- horizon path/cap state ---
        clock = self._clock(bars, idx, position)
        horizon = self.horizon.update(
            bars, idx, position, clock, vol, z_drift, fav_r, adv_r,
            running_x=st.running_x, running_max_x=st.running_max_x, cfg=cfg,
        )
        giveback_frac = (
            horizon.running_giveback / horizon.running_favorable_excursion
            if horizon.running_favorable_excursion > 0
            else 0.0
        )
        hazard, hazard_model = self.hazard.update({"giveback_fraction": giveback_frac})

        # --- diagnostics (log-only, Inv F) ---
        jump_rate = st.jump_count / max(st.bar_count, 1)
        diag = BeliefDiagnostics(
            innovation_whiteness=0.0,
            kalman_consistency=float(slope_var),
            detector_margin=float(cfg.defaults.cusum_h - seq_stat)
            if cfg.detector_kind == "cusum"
            else float(seq_stat),
            jump_flag_rate=float(jump_rate),
            hazard_rung=hazard_model,
            effective_sample_size=int(st.bar_count),
            tail_nu=vol.tail_nu,
            vol_method=vol.method,
            vol_quality_flag=vol.quality_flag,
        )

        return BeliefState(
            idx=idx,
            clock=clock,
            horizon=horizon,
            vol=vol,
            filtered_slope=float(filtered_slope),
            slope_var=float(slope_var),
            z_drift=float(z_drift),
            adverse_excursion_r=float(adv_r),
            favorable_excursion_r=float(fav_r),
            seq_stat=float(seq_stat),
            seq_alarm=bool(seq_alarm),
            jump_against=bool(jump_against),
            jump_with=bool(jump_with),
            hazard=float(hazard),
            hazard_model=hazard_model,
            reason_signals={
                "z_drift": float(z_drift),
                "filtered_slope": float(filtered_slope),
                "seq_stat": float(seq_stat),
                "favorable_r": float(fav_r),
                "adverse_r": float(adv_r),
            },
            diag=diag,
        )

    @staticmethod
    def _side_std_return(bars: Bars, idx: int, position: Position, vol: VolState) -> float:
        """Side-adjusted standardized one-bar log return (favorable positive). Causal."""
        if idx < 1:
            return 0.0
        # Do not trust standardized returns before the vol stack is warm or when sigma is degenerate
        # (a single early return makes the MAD scale ~0 -> exploding z). Causal warmup gate: feed the
        # detector a zero increment so it neither false-alarms nor counts warmup as deterioration.
        if vol.quality_flag in ("warmup", "nonfinite_fallback") or vol.sigma_log <= 1e-7:
            return 0.0
        c = bars.c
        if c[idx] <= 0 or c[idx - 1] <= 0:
            return 0.0
        r = float(np.log(c[idx] / c[idx - 1]))
        scale = max(vol.sigma_log, 1e-8)
        z = float(position.side) * r / scale
        # Clamp to a sane band so a genuine but extreme single bar cannot pump the CUSUM past h in
        # one step (a true jump is handled by the jump detector, not the deterioration detector).
        return float(np.clip(z, -8.0, 8.0))
