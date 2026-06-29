"""PartialFillManager — the residual ladder after a partial fill, never overshooting target (§8 / Inv A).

Deterministic ladder, re-evaluating the tactic on the residual with the CURRENT microstate (a partial
that filled because the market moved through you is information — the residual may now warrant
crossing):

  1. recompute ``remaining_qty = target_qty − filled_qty`` (signed; never overshoots target, Inv A);
  2. re-run TacticSelector on ``remaining_qty`` with the chase budget reduced by progress already
     made toward worst_entry (the engine passes the advanced WorkingState);
  3. if ``|remaining_qty|·mid < min_notional``, do NOT emit a dust order: mark the residual closed
     with ``below_min_notional`` (a few $ on a $1k book is not worth a taker fee + a second adverse
     event);
  4. respect the SAME worst_entry bound + abort rule as §5.3.

A reduce/exit residual is ALWAYS pursued to flat (crossing if needed); only an ENTRY residual may be
abandoned (budget abort or min-notional). Never increases qty beyond target; never re-opens an
aborted entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import ExecTarget, MicroState, WorkingState

__all__ = ["ResidualPlan", "PartialFillManager"]


@dataclass(frozen=True)
class ResidualPlan:
    """What to do with the unfilled residual after a partial fill (§8)."""

    remaining_qty: float  # signed; |remaining| <= |target_qty| − |filled| always (Invariant A)
    pursue: bool  # False -> residual closed (dust or aborted), no further order
    closed_reason: str  # "" | "below_min_notional" | "exit_dust_venue_floor" | "complete"
    reason_codes: tuple[str, ...]


class PartialFillManager:
    """Handles the residual after a partial fill (§8). Qty discipline = Invariant A."""

    def on_fill(
        self,
        target: ExecTarget,
        working: WorkingState,
        micro: MicroState,
        cfg: ResolvedConfig,
    ) -> ResidualPlan:
        """Decide the residual plan after a partial fill (§8). Never overshoots target (Invariant A)."""
        # Signed remaining; clamp magnitude so a stale/over-reported fill can never overshoot (Inv A).
        filled_mag = min(abs(working.filled_qty), abs(target.target_qty))
        remaining_mag = abs(target.target_qty) - filled_mag
        remaining = target.side * remaining_mag

        if remaining_mag <= 0.0:
            return ResidualPlan(0.0, False, "complete", ("fill_complete",))

        residual_notional = remaining_mag * micro.mid
        below_min_notional = residual_notional < cfg.instrument.min_notional

        # §8: a reduce/exit residual is ALWAYS pursued to flat (crossing if needed); only an ENTRY
        # residual may be abandoned. So the min-notional dust close applies to ENTRIES only.
        if target.reduce_only:
            if below_min_notional:
                # SPEC-JUSTIFIED CARVE-OUT (the venue sub-min-notional reality). §8/P5/P6 require an
                # exit to be pursued to flat, but a residual below the venue min-notional CANNOT be
                # placed on the venue at all — no reduce-only order of that size is acceptable, so it
                # is a venue IMPOSSIBILITY, not a choice to abandon. We mark it closed with a DISTINCT
                # reason (``exit_dust_venue_floor``, never the entry ``below_min_notional`` abort) so
                # the audit log shows it was a venue floor on an exit, not an exit that chose to stop
                # pursuing. The bot keeps the position's stop covering this dust (§8/§9). This is the
                # one venue-reality exception to "exits never stop pursuing"; everything reducible IS
                # pursued below.
                return ResidualPlan(
                    remaining, False, "exit_dust_venue_floor", ("exit_dust_venue_floor",)
                )
            # A reducible exit residual: pursue to flat (the engine crosses if needed; exits never
            # abort, P5/P6). The crossing/min() decision is made downstream on remaining_qty.
            return ResidualPlan(remaining, True, "", ("pursue_exit_residual",))

        # ENTRY residual. §8.3 min-notional dust: a sub-min-notional entry residual is closed (not
        # emitted as dust — a few $ on a $1k book is not worth a taker fee + a second adverse event).
        if below_min_notional:
            return ResidualPlan(remaining, False, "below_min_notional", ("below_min_notional",))

        # Otherwise pursue the entry residual (the engine re-runs TacticSelector on remaining_qty with
        # the advanced WorkingState so the chase budget is reduced by progress toward worst_entry, §8).
        return ResidualPlan(remaining, True, "", ("pursue_residual",))
