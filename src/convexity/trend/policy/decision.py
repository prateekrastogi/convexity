"""DecisionPolicy — the ONLY module that emits ActionDecision (TREND §10 decision, Inv A).

Fixed priority order (§10), applied EXACTLY:

  1. invalid data / halted / untradeable -> continue or reduce (never a pretend executable exit).
  2. jump_against                        -> exit at next actionable bar (D5).
  3. sequential deterioration alarm      -> exit (D2).
  4. trail/invalidation breach           -> exit (D1).
  5. operational time cap reached        -> exit, reason-coded `time_cap` (Inv HC, NOT trend death).
  6. hazard/slope/gap survival           -> reduce.
  7. otherwise                            -> continue.

This method sets ``stopping_time_fired`` and ``stop_cause``. Diagnostics must not alter the decision
(Inv F). It also builds the TREND-owned outputs LEVERAGE/COST consume read-only (Inv P) and the
``ExecutionIntent`` EXECUTION executes (Inv N) — TREND never places orders.
"""

from __future__ import annotations

import math

from convexity.core.types import (
    Bars,
    ExecutionIntent,
    HoldHorizonEstimate,
    Position,
    ResidualMoveEstimate,
)
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.data.quality import QualityProfiler
from convexity.trend.data.timing import ActionTimer
from convexity.trend.policy.convex_exit import ConvexExit
from convexity.trend.policy.reduce import ReducePolicy
from convexity.trend.types import (
    Action,
    ActionDecision,
    BeliefState,
    ExitState,
    StopCause,
)

# clock -> milliseconds-per-unit, for op_cap_remaining_ms (§6b). Sessions/active-minutes/bars are
# converted to a calendar-ms estimate so COST's funding gate gets a millisecond bound.
_MS_PER_DAY = 86_400_000
_MS_PER_MIN = 60_000


class DecisionPolicy:
    """The single live-action emitter (§10, Inv A)."""

    def __init__(
        self,
        convex_exit: ConvexExit | None = None,
        reduce_policy: ReducePolicy | None = None,
        timer: ActionTimer | None = None,
        quality: QualityProfiler | None = None,
    ) -> None:
        self.convex_exit = convex_exit if convex_exit is not None else ConvexExit()
        self.reduce_policy = reduce_policy if reduce_policy is not None else ReducePolicy(self.convex_exit)
        self.timer = timer if timer is not None else ActionTimer()
        self.quality = quality if quality is not None else QualityProfiler()

    def decide(
        self,
        bars: Bars,
        idx: int,
        position: Position,
        belief: BeliefState,
        exit_state: ExitState,
        cfg: ResolvedConfig,
    ) -> ActionDecision:
        """Emit the only live action at ``idx`` by the fixed priority order (§10)."""
        action: Action = "continue"
        stop_cause: StopCause | None = None
        stopping_time_fired = False
        reasons: list[str] = []
        size_fraction = exit_state.size_fraction

        tradeable = self.quality.is_tradeable(bars, idx)
        horizon = belief.horizon
        cap_reached = horizon.elapsed_value >= horizon.horizon_cap > 0

        # ----- fixed priority (§10) -----
        if not tradeable:
            # 1. untradeable: continue or reduce, never a pretend exit.
            frac, reduce_reasons = self.reduce_policy.fraction(belief, exit_state, position.side, cfg)
            if frac < 1.0:
                action, size_fraction = "reduce", frac
                reasons.extend(["untradeable", *reduce_reasons])
            else:
                reasons.append("untradeable_continue")
        elif belief.jump_against:
            # 2. jump_against -> hard exit (D5).
            action, stop_cause, stopping_time_fired = "exit", "jump_against", True
            reasons.append("jump_against")
        elif belief.seq_alarm:
            # 3. sequential deterioration alarm -> hard exit (D2).
            action, stop_cause, stopping_time_fired = "exit", "seq_alarm", True
            reasons.append("seq_alarm")
        elif self.convex_exit.breached(bars, idx, position, exit_state):
            # 4. trail/invalidation breach -> hard exit (D1).
            action, stop_cause, stopping_time_fired = "exit", "trail_breach", True
            reasons.append("trail_breach")
        elif cap_reached:
            # 5. operational time cap -> exit, reason-coded time_cap (Inv HC; NOT trend death).
            action, stop_cause, stopping_time_fired = "exit", "time_cap", True
            reasons.append("time_cap")
        else:
            # 6. hazard/slope/gap survival -> reduce.
            frac, reduce_reasons = self.reduce_policy.fraction(belief, exit_state, position.side, cfg)
            if frac < 1.0:
                action, size_fraction = "reduce", frac
                reasons.extend(reduce_reasons)
            else:
                # 7. otherwise -> continue.
                reasons.append("alive")

        decision_idx = idx
        if action == "exit":
            action_idx, action_price_kind = self.timer.action_index(idx, bars, "next_open")
        else:
            action_idx, action_price_kind = idx, "next_mid"

        execution_intent = self._intent(action, stop_cause, exit_state)
        hold_horizon = self._hold_horizon(belief, cfg)
        residual_move = self._residual_move(belief, cfg)

        return ActionDecision(
            idx=idx,
            decision_idx=decision_idx,
            action_idx=action_idx,
            action=action,
            horizon=horizon,
            stopping_time_fired=stopping_time_fired,
            stop_cause=stop_cause,
            reason_codes=tuple(reasons),
            exit_state=exit_state,
            size_fraction=float(size_fraction),
            hold_horizon=hold_horizon,
            residual_move=residual_move,
            action_price_kind=action_price_kind,  # type: ignore[arg-type]
            execution_intent=execution_intent,
            engine_version=cfg.defaults.engine_version,
        )

    # ---- TREND-owned outputs (Inv N / Inv P) ----------------------------------------------
    @staticmethod
    def _intent(action: Action, stop_cause: StopCause | None, state: ExitState) -> ExecutionIntent:
        """Build the ExecutionIntent (Inv N). Hard exit -> reduce_only IMMEDIATE; continue/reduce
        with a moved trail -> amend_stop; pure continue -> none."""
        if action == "exit":
            return ExecutionIntent(
                invalidation_level=state.invalidation,
                reduce_only=True,
                urgency="immediate",
                kind="exit_reduce_only",
            )
        # continue/reduce: propagate the (possibly moved) protective level as an amend.
        return ExecutionIntent(
            invalidation_level=state.invalidation,
            reduce_only=True,
            urgency="passive",
            kind="amend_stop",
        )

    def _ms_per_clock_unit(self, clock: str) -> int:
        if clock == "calendar_days":
            return _MS_PER_DAY
        if clock == "trading_sessions":
            return _MS_PER_DAY  # one session ~ one calendar day for the COST ms bound
        if clock == "active_minutes":
            return _MS_PER_MIN
        return _MS_PER_MIN  # bars_since_entry: 1-min bars

    def _hold_horizon(self, belief: BeliefState, cfg: ResolvedConfig) -> HoldHorizonEstimate:
        """Conservative hold-horizon RANGE (Inv P / §6b). Never a point scalar.

        Backed by cap-remaining conditioned on cap_fraction_used: the remaining op cap is the hard
        bound; the lo/hi range is a conservative fraction of it (drift is unmeasurable, §0).
        """
        h = belief.horizon
        unit_ms = self._ms_per_clock_unit(h.active_clock)
        op_cap_remaining_ms = int(max(h.cap_remaining, 0.0) * unit_ms)
        # conservative band: lo = small near-term remainder, hi = full op-cap remainder.
        lo_ms = int(0.25 * op_cap_remaining_ms)
        hi_ms = op_cap_remaining_ms
        # max-hold quantile backing COST's funding gate: the configured survival quantile of remaining.
        q = cfg.defaults.hold_quantile
        max_hold_quantile_ms = int(q * op_cap_remaining_ms)
        return HoldHorizonEstimate(
            active_clock=h.active_clock,
            expected_remaining_lo_ms=lo_ms,
            expected_remaining_hi_ms=hi_ms,
            max_hold_quantile_ms=max(max_hold_quantile_ms, lo_ms),
            quantile_used=float(q),
            op_cap_remaining_ms=op_cap_remaining_ms,
        )

    def _residual_move(self, belief: BeliefState, cfg: ResolvedConfig) -> ResidualMoveEstimate:
        """Conservative residual FAVORABLE move band (Inv P / §6b). A band with one conservative
        point COST consumes; never a confident point forecast (drift unmeasurable, §0).

        Edge proxy: the favorable normalized drift z above the floor, scaled to R units by the
        current open favorable excursion. The conservative point is the LOW band edge.
        """
        z = belief.z_drift if math.isfinite(belief.z_drift) else 0.0
        edge = max(z - cfg.defaults.drift_floor_z, 0.0)
        # express in R units: a unit of remaining favorable move proportional to the realized edge.
        hi = float(edge)
        lo = float(cfg.defaults.residual_move_floor_r)
        point = lo  # conservative: COST consumes the LOW edge of the band
        return ResidualMoveEstimate(
            expected_residual_move_lo=lo,
            expected_residual_move_hi=max(hi, lo),
            point_used_by_cost=point,
            units="initial_risk",
        )
