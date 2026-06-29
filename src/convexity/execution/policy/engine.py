"""ExecutionEngine — the only entry point that turns a target into the next order command (§11).

Pipeline (§11): build TacticCosts -> apply gates -> TacticSelector.select -> if POST_ONLY chosen,
hand to ChaseLadder; on a partial, PartialFillManager; always ensure ProtectiveStop is placed /
reconciled. Emits exactly one :class:`ExecDecision` with the chosen tactic, price, qty, deterministic
client_order_id, expected-cost decomposition, binding gate, chase step, abort flag, and reason codes.

Deterministic given (target, micro, working, cfg). Reads position / account state from Nautilus,
never tracks it itself (Invariant N). Diagnostics never alter the decision (Invariant H). Batch
replay and one-event-at-a-time streaming yield identical decisions (Invariant E — pure function).
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.cost.adverse_selection import AdverseSelection
from convexity.execution.cost.delay_cost import DelayCost
from convexity.execution.cost.impact import ImpactModel
from convexity.execution.cost.spread_cost import SpreadCost
from convexity.execution.cost.tactic_cost import TacticCostModel
from convexity.execution.policy.chase import ChaseLadder, worst_entry
from convexity.execution.policy.order_issuer import OrderIssuer
from convexity.execution.policy.partial import PartialFillManager
from convexity.execution.policy.protective_stop import ProtectiveStop
from convexity.execution.policy.tactic_select import TacticSelector
from convexity.execution.types import (
    ExecDecision,
    ExecTarget,
    MicroState,
    OrderCommand,
    PositionState,
    TacticCosts,
    Tactic,
    WorkingState,
)

__all__ = ["ExecutionEngine"]

ENGINE_VERSION = "execution-1.0.0"


class ExecutionEngine:
    """The pure-Python policy engine (§11). One ExecDecision per call; Nautilus owns plumbing (Inv N)."""

    def __init__(
        self,
        cost_model: TacticCostModel | None = None,
        selector: TacticSelector | None = None,
        chase: ChaseLadder | None = None,
        partial: PartialFillManager | None = None,
        issuer: OrderIssuer | None = None,
        stop: ProtectiveStop | None = None,
    ) -> None:
        self.cost_model = cost_model or TacticCostModel(
            SpreadCost(), ImpactModel(), AdverseSelection(), DelayCost()
        )
        self.selector = selector or TacticSelector()
        self.chase = chase or ChaseLadder()
        self.partial = partial or PartialFillManager()
        self.issuer = issuer or OrderIssuer()
        self.stop = stop or ProtectiveStop()

    def on_target(
        self,
        target: ExecTarget,
        micro: MicroState,
        working: WorkingState,
        cfg: ResolvedConfig,
        *,
        as_of_ts: int = 0,
    ) -> ExecDecision:
        """Turn a target into the next order command (§11). Deterministic; reads state, never owns it."""
        # --- residual discipline first (§8): a partial may have already filled toward target (Inv A).
        residual = self.partial.on_fill(target, working, micro, cfg)
        if not residual.pursue:
            # Residual closed (complete or sub-min-notional dust). Still ensure the stop covers the
            # FILLED qty (§8) — the engine re-reconciles the stop on every fill event.
            stop_cmds = self._ensure_stop(target, working, cfg)
            # Both the entry dust close and the exit venue-floor carve-out (§8) surface as the
            # below_min_notional abort reason; the residual's own reason_codes preserve which it was.
            min_notional_close = residual.closed_reason in ("below_min_notional", "exit_dust_venue_floor")
            abort_reason = "below_min_notional" if min_notional_close else ""
            return ExecDecision(
                intent_id=target.intent_id, instrument=target.instrument, tactic="MARKET",
                price=0.0, qty=0.0, reduce_only=target.reduce_only,
                client_order_id="", expected_cost_bps=0.0, cost_decomposition={},
                binding_gate="min_notional" if abort_reason else "none",
                chase_step=working.chase_step, aborted=bool(abort_reason),
                abort_reason=abort_reason,  # type: ignore[arg-type]
                reason_codes=residual.reason_codes, engine_version=ENGINE_VERSION,
                commands=stop_cmds,
            )

        remaining = residual.remaining_qty

        # --- cost decomposition (§5.1) + the argmin-with-gates (§6).
        costs = self.cost_model.costs(target, micro, cfg, as_of_ts=as_of_ts)
        sel = self.selector.select(target, micro, costs, cfg)

        # --- a disabled entry (halt/stale on a NEW entry): emit a no-op abort, never post (§6.2).
        if sel.disable_entry and not target.reduce_only:
            return ExecDecision(
                intent_id=target.intent_id, instrument=target.instrument, tactic=sel.tactic,
                price=0.0, qty=0.0, reduce_only=False, client_order_id="",
                expected_cost_bps=costs.by_tactic[sel.tactic],
                cost_decomposition=self._decomposition_for(sel.tactic, costs),
                binding_gate=sel.binding_gate, chase_step=working.chase_step, aborted=True,
                abort_reason="halt", reason_codes=("entry_disabled_halt",),
                engine_version=ENGINE_VERSION, commands=(),
            )

        # --- worst_entry bound + chase ladder (§5.2/§5.3, Invariant D). On a POST_ONLY choice that is
        # mid-chase, advance the ladder; otherwise issue the chosen tactic directly.
        we = worst_entry(target, target.arrival_mid, cfg)
        commands: tuple[OrderCommand, ...] = ()
        aborted = False
        abort_reason = ""
        reason_codes: tuple[str, ...] = ()
        chase_step = working.chase_step
        tactic = sel.tactic
        price = micro.mid

        if tactic == "POST_ONLY_MAKER":
            post_wins = True  # the min() at this state already chose POST_ONLY
            action = self.chase.step(target, micro, working, cfg, post_still_wins=post_wins)
            chase_step = action.chase_step
            we = action.worst_entry
            if action.action == "abort":
                aborted, abort_reason = True, action.abort_reason
                tactic = "POST_ONLY_MAKER"
                price = action.price
                reason_codes = action.reason_codes
            else:
                tactic = action.tactic
                price = action.price
                reason_codes = action.reason_codes
                cmd = self.issuer.issue(
                    tactic, target, price, remaining, target.reduce_only, cfg,
                    slice_idx=working.chase_step, attempt=0,
                )
                commands = (cmd,)
        else:
            # A crossing tactic chosen directly. Cap the price at worst_entry for an entry (Inv D).
            price = self._cap_entry(target, micro.mid, we)
            cmd = self.issuer.issue(
                tactic, target, price, remaining, target.reduce_only, cfg,
                slice_idx=working.chase_step, attempt=0,
            )
            commands = (cmd,)
            reason_codes = (f"cross:{tactic}",)

        # --- always ensure the protective stop is placed / covers the filled qty (§9, Invariant S).
        commands = commands + self._ensure_stop(target, working, cfg)

        return ExecDecision(
            intent_id=target.intent_id,
            instrument=target.instrument,
            tactic=tactic,
            price=price,
            qty=abs(remaining),
            reduce_only=target.reduce_only,
            client_order_id=commands[0].client_order_id if commands else "",
            # §12: log the cost of the tactic ACTUALLY emitted, not the originally-selected one — the
            # ChaseLadder may have flipped POST_ONLY -> a crossing tactic (or aborted) since select().
            expected_cost_bps=costs.by_tactic[tactic],
            cost_decomposition=self._decomposition_for(tactic, costs),
            binding_gate=sel.binding_gate,
            chase_step=chase_step,
            aborted=aborted,
            abort_reason=abort_reason,  # type: ignore[arg-type]
            reason_codes=tuple(reason_codes),
            engine_version=ENGINE_VERSION,
            commands=commands,
        )

    # ------------------------------------------------------------------ helpers

    def _ensure_stop(
        self, target: ExecTarget, working: WorkingState, cfg: ResolvedConfig
    ) -> tuple[OrderCommand, ...]:
        """Emit a venue reduce-only stop covering the FILLED qty at the current invalidation (§9).

        Only when a position is (partially) open. The stop's amend_generation derives from the
        observed invalidation generation in WorkingState (Invariant Q-STOP), so it survives restart.
        """
        if abs(working.filled_qty) <= 0.0:
            return ()
        pos = PositionState(
            instrument=target.instrument, side=target.side, qty=abs(working.filled_qty)
        )
        place = self.stop.place(
            target.intent_id, pos, target.invalidation, cfg,
            amend_generation=working.invalidation_generation,
        )
        return (place,)

    @staticmethod
    def _decomposition_for(tactic: Tactic, costs: TacticCosts) -> dict[str, float]:
        """The cost decomposition of the tactic ACTUALLY emitted (§12 realized-vs-expected).

        The headline ``expected_cost_bps`` / ``cost_decomposition`` on the ExecDecision must describe
        the tactic the engine EMITTED, not the originally-selected one — after the ChaseLadder flips
        POST_ONLY -> a crossing tactic (or aborts) the emitted tactic differs from ``sel.tactic``, and
        logging the stale decomposition would make the realized-vs-expected regression (§12) compare a
        realized cross against an expected post. This pulls the per-component bps for ``tactic`` from
        the already-computed cost maps (Invariant H: log-only, never feeds the choice)."""
        return {
            "spread_bps": costs.spread_bps[tactic],
            "impact_bps": costs.impact_bps[tactic],
            "adverse_sel_bps": costs.adverse_sel_bps[tactic],
            "delay_bps": costs.delay_bps[tactic],
            "funding_bps": costs.funding_bps[tactic],
        }

    @staticmethod
    def _cap_entry(target: ExecTarget, price: float, we: float) -> float:
        """Cap a crossing entry price at worst_entry (never beyond, Invariant D); exits uncapped."""
        if target.reduce_only:
            return price
        if target.side == 1:
            return min(price, we)
        return max(price, we)
