"""ChaseLadder — the breakout cancel/replace chase, hard-bounded by the risk budget (§5.2 / §5.3).

If POST_ONLY_MAKER is chosen and does not fill within the per-step timeout, the engine reprices
toward the market by the RED reprice step and re-evaluates. A deterministic, FINITE ladder (not a
loop chasing to infinity). Each step re-runs the §5.1 cost comparison with the CURRENT microstate, so
a spread blowout or vol spike mid-chase can flip the decision to cross.

The chase is hard-bounded by the trade's own invalidation (§5.3, Invariant D):

    worst_entry = entry0 + side · χ · |inv − entry0|          (χ in [0.25, 0.40], RED)

The engine MUST NOT place, reprice, or cross at a price beyond ``worst_entry``. If filling the
residual would require a price beyond it:
  * ENTRY  -> ABORT the residual, cancel working orders, emit ``chase_budget_exhausted`` (§5.3).
  * REDUCE/EXIT -> do NOT abort; cross immediately at MARKET (exits never abort, P5/P6).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import ExecTarget, MicroState, Tactic, WorkingState

__all__ = ["worst_entry", "ChaseAction", "ChaseLadder"]


def worst_entry(target: ExecTarget, entry0: float, cfg: ResolvedConfig) -> float:
    """``worst_entry = entry0 + side·χ·|inv − entry0|`` — the hard chase bound (§5.3 / Invariant D).

    The SAME canonical form as P5: explicit ``side``, absolute value, and ``entry0`` (not bare
    ``entry``). For a long (side=+1) a below-invalidation gives a worst_entry ABOVE entry0 (the entry
    drifts up, away from the stop); for a short (side=-1) it is below. χ is RED, never per-instrument.
    """
    chi = cfg.defaults.chase_fraction_chi
    return float(entry0 + target.side * chi * abs(target.invalidation - entry0))


@dataclass(frozen=True)
class ChaseAction:
    """One deterministic chase step's decision (§5.2 / §5.3)."""

    action: str  # "post" | "cross" | "abort"
    tactic: Tactic
    price: float
    worst_entry: float
    chase_step: int
    aborted: bool
    abort_reason: str
    reason_codes: tuple[str, ...]


class ChaseLadder:
    """The bounded cancel/replace chase with the worst_entry abort (§5.2 / §5.3 / Invariant D)."""

    def step(
        self,
        target: ExecTarget,
        micro: MicroState,
        working: WorkingState,
        cfg: ResolvedConfig,
        *,
        post_still_wins: bool,
        cross_tactic: Tactic = "MARKETABLE_LIMIT",
    ) -> ChaseAction:
        """Advance the cancel/replace chase by one deterministic step (§5.2 / §5.3).

        ``post_still_wins`` is the re-run §5.1 min() outcome at the CURRENT microstate (the engine
        recomputes the cost min() each step and passes whether POST_ONLY still wins). Behaviour:
          * within budget and posting still wins -> reprice toward market by ``reprice_tick``, keep
            the post-only order working;
          * posting no longer wins (drift/spread flipped the min()) -> cross with ``cross_tactic``,
            capped at worst_entry;
          * the next required price would breach worst_entry -> ENTRY aborts the residual
            (``chase_budget_exhausted``); REDUCE/EXIT crosses at MARKET regardless.

        ``worst_entry`` is recomputed every step and is a HARD bound: no order is ever placed beyond
        it on an entry. The ladder is finite (``max_chase_steps`` RED).
        """
        entry0 = target.arrival_mid
        we = worst_entry(target, entry0, cfg)
        step = working.chase_step
        tick = cfg.instrument.tick_size
        side = target.side

        # The price the NEXT reprice would sit at (toward the market): entry0 drifts by step ticks.
        next_step = step + 1
        next_price = entry0 + side * next_step * cfg.defaults.reprice_tick_step * tick

        # --- budget / step exhaustion: would the next price breach worst_entry, or steps run out?
        beyond = self._beyond(side, next_price, we)
        exhausted = beyond or next_step > cfg.defaults.max_chase_steps

        if exhausted:
            if target.reduce_only:
                # EXITS NEVER ABORT (P5/P6): cross at MARKET, getting flat dominates price.
                return ChaseAction(
                    "cross", "MARKET", micro.mid, we, step, False, "",
                    ("chase_exit_cross_market",),
                )
            # ENTRY: abort the unfilled residual (Invariant D).
            return ChaseAction(
                "abort", cross_tactic, we, we, step, True, "chase_budget_exhausted",
                ("chase_budget_exhausted",),
            )

        if not post_still_wins:
            # Mid-chase flip post->cross; cap the crossing price at worst_entry (never beyond).
            cross_px = self._cap(side, micro.mid, we)
            return ChaseAction(
                "cross", cross_tactic, cross_px, we, step, False, "",
                ("chase_flip_to_cross",),
            )

        # Posting still wins and we are within budget: reprice one step toward the market.
        repriced = self._cap(side, next_price, we)
        return ChaseAction(
            "post", "POST_ONLY_MAKER", repriced, we, next_step, False, "",
            ("chase_reprice",),
        )

    @staticmethod
    def _beyond(side: int, price: float, we: float) -> bool:
        """True if ``price`` is beyond ``worst_entry`` in the trade's adverse direction (§5.3)."""
        return (side == 1 and price > we) or (side == -1 and price < we)

    @staticmethod
    def _cap(side: int, price: float, we: float) -> float:
        """Cap a price at ``worst_entry`` in the adverse direction (never beyond, Invariant D)."""
        if side == 1:
            return min(price, we)
        return max(price, we)
