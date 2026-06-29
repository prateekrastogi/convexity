"""TacticSelector — argmin E[cost] over ALLOWED tactics, AFTER the five hard gates (§1 / §6).

    tactic* = argmin_{t in ALLOWED} E[cost_t]      ALLOWED = {POST_ONLY_MAKER, LIMIT_IOC,
                                                              MARKETABLE_LIMIT, MARKET} \\ GATED_OUT

A pure ``min()`` over independent expected costs (Invariant B); never a multiplicative score. The
gates can only REMOVE tactics or FORCE the most aggressive one — they make execution MORE aggressive
(cross sooner), never less. Diagnostics (the per-tactic cost decomposition) never alter the choice
(Invariant H).

The five hard gates, evaluated BEFORE the min() (§6):
  1. Protective-exit gate — reduce_only + IMMEDIATE -> force MARKETABLE_LIMIT/MARKET (P5/P6).
  2. Halt / outage / stale-book gate — force the most aggressive ALLOWED for any exit; DISABLE entries.
  3. Maker-floor gate — remove POST_ONLY unless p_fill_LCB(δ*) >= floor_eff (keyed to the LOWER bound,
     not the point estimate; §6.3). Cold-start treats the symbol as below floor by default.
  4. Rate-limit gate — prefer a single marketable order over a long maker chase (§6.4).
  5. Min-notional / tick gate — handled in partial.py / the engine (§8); named here for completeness.
"""

from __future__ import annotations

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.microstate.fill_prob import FillProbability
from convexity.execution.types import (
    BindingGate,
    ExecTarget,
    MicroState,
    TacticCosts,
    Tactic,
)

__all__ = ["TacticSelector", "TacticSelection"]

_ALL: tuple[Tactic, ...] = ("POST_ONLY_MAKER", "LIMIT_IOC", "MARKETABLE_LIMIT", "MARKET")
# Crossing tactics in increasing aggressiveness; the most aggressive ALLOWED is the last present.
_CROSSING: tuple[Tactic, ...] = ("LIMIT_IOC", "MARKETABLE_LIMIT", "MARKET")


class TacticSelection:
    """The chosen tactic + offset + the binding gate + the full diagnostics (log-only, Inv H)."""

    __slots__ = ("tactic", "offset_ticks", "binding_gate", "allowed", "diagnostics", "disable_entry")

    def __init__(
        self,
        tactic: Tactic,
        offset_ticks: float,
        binding_gate: BindingGate,
        allowed: tuple[Tactic, ...],
        diagnostics: dict,
        disable_entry: bool = False,
    ) -> None:
        self.tactic = tactic
        self.offset_ticks = offset_ticks
        self.binding_gate = binding_gate
        self.allowed = allowed
        self.diagnostics = diagnostics
        self.disable_entry = disable_entry


class TacticSelector:
    """The argmin-with-gates tactic selector (§6). Gates only remove / force-aggressive (Inv B/H)."""

    def select(
        self,
        target: ExecTarget,
        micro: MicroState,
        costs: TacticCosts,
        cfg: ResolvedConfig,
    ) -> TacticSelection:
        """Return the chosen (tactic, offset, binding_gate, ...) (§6). Pure min() after the gates."""
        allowed = set(_ALL)
        binding: BindingGate = "none"
        disable_entry = False

        # --- Gate 1: protective-exit. reduce_only + IMMEDIATE -> never post; cross now (P5/P6).
        if target.reduce_only and target.urgency == "IMMEDIATE":
            allowed.discard("POST_ONLY_MAKER")
            allowed.discard("LIMIT_IOC")  # force a marketable / market crossing
            binding = "protective_exit"

        # --- Gate 2: halt / outage / stale. Force the most aggressive ALLOWED for any exit; DISABLE
        # new entries. (A halt is a crossing-only state for protective action.)
        if micro.halted or micro.stale:
            if target.reduce_only:
                forced = self._most_aggressive(allowed)
                return TacticSelection(
                    forced, costs.chosen_offset_ticks, "halt", tuple(sorted(allowed)),
                    self._diag(costs), disable_entry=True,
                )
            # an entry under halt/stale is disabled.
            allowed.discard("POST_ONLY_MAKER")
            disable_entry = True
            if binding == "none":
                binding = "halt"

        # --- Gate 3: maker-floor (keyed to the p_fill LOWER-confidence bound, not the point est).
        floor_eff = FillProbability.floor_eff(micro.fill_prob_calib_error, cfg)
        below_floor = micro.cold_start or costs.p_fill_lcb_at_offset < floor_eff
        if below_floor:
            allowed.discard("POST_ONLY_MAKER")
            if binding == "none":
                binding = "maker_floor"

        # --- Gate 4: rate-limit. Suppress cancel/replace churn -> prefer a single marketable order
        # over a long maker chase (§6.4).
        if not micro.rate_limit_ok:
            allowed.discard("POST_ONLY_MAKER")
            if binding == "none":
                binding = "rate_limit"

        # --- the pure min() over the surviving ALLOWED set (Invariant B).
        allowed_t = tuple(t for t in _ALL if t in allowed)
        chosen = min(allowed_t, key=lambda t: costs.by_tactic[t])
        offset = costs.chosen_offset_ticks if chosen == "POST_ONLY_MAKER" else 0.0
        return TacticSelection(
            chosen, offset, binding, allowed_t, self._diag(costs), disable_entry=disable_entry
        )

    @staticmethod
    def _most_aggressive(allowed: set[Tactic]) -> Tactic:
        """The most aggressive ALLOWED crossing tactic (MARKET if present, else marketable, …)."""
        for t in reversed(_CROSSING):
            if t in allowed:
                return t
        return "MARKET"

    @staticmethod
    def _diag(costs: TacticCosts) -> dict:
        """Log-only diagnostics: every tactic's E[cost] + decomposition (Invariant H)."""
        return {
            "by_tactic": dict(costs.by_tactic),
            "spread_bps": dict(costs.spread_bps),
            "impact_bps": dict(costs.impact_bps),
            "adverse_sel_bps": dict(costs.adverse_sel_bps),
            "delay_bps": dict(costs.delay_bps),
            "funding_bps": dict(costs.funding_bps),
            "p_fill": costs.p_fill_at_offset,
            "p_fill_lcb": costs.p_fill_lcb_at_offset,
            "p_abort_if_nofill": costs.p_abort_if_nofill,
            "drift_g": costs.drift_g,
        }
