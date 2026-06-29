"""Thin bridge: Nautilus / pillar events ↔ OPS streams + directive (import-isolated).

Spec: OPS §2 (the engine wraps a Nautilus actor/strategy), §nautilus/adapter. The engine SUBSCRIBES to
order/fill/position events + the heartbeat + the pillar streams and EMITS an ``OpsDirective``. It never
tracks position/account state itself — it reads Nautilus's ``Portfolio``/``Cache`` (Invariant
PROV/RECON). This adapter is the seam: it maps the (Nautilus-typed, on-VM) events into the frozen OPS
fact/observation types and pulls the directive back out.

PURE-PYTHON, NautilusTrader-free: this file does NOT import nautilus (import-isolated per the build
directive). The on-VM ``strategy/convexity_strategy.py`` is the only code that imports nautilus and
calls into this adapter with the engine-native objects projected onto the OPS fact types.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.types import OpsDirective
from convexity.ops.brake.engine import OpsEngine
from convexity.ops.nautilus.flatten import FlattenCommand, flatten_command_for
from convexity.ops.types import EquityRef, KillVerdict, StreamObs


@dataclass
class OpsAdapter:
    """The Nautilus-facing bridge around an :class:`OpsEngine` (§2/§3).

    The VM strategy feeds it the projected facts each bar and reads back the :class:`OpsDirective` +
    the :class:`FlattenCommand`. The engine owns all OPS state; the adapter is stateless glue (it holds
    only the engine reference) so it can be tested without nautilus.
    """

    engine: OpsEngine

    def on_bar(self, equity: EquityRef | None = None, *, recon_mismatch: bool = False,
               recon_detail: str = "", heartbeat_code: float = 0.0) -> tuple[OpsDirective,
                                                                            FlattenCommand | None]:
        """Push the per-bar binary reads, evaluate, and return the directive + any flatten command.

        Statistical-stream observations are pushed separately via ``engine.on_obs`` as fills/TCA/vol/
        edge events arrive (event-driven, not per-bar). This call closes the bar: it reads the shared
        equity/peak reference, the reconciliation flag (monitored not re-derived, P5), and the
        heartbeat classification, then advances the kill machine.
        """
        if equity is not None:
            self.engine.on_equity(equity)
        self.engine.on_recon(recon_mismatch, recon_detail)
        self.engine.on_heartbeat(heartbeat_code)
        verdict: KillVerdict = self.engine.evaluate()
        flatten = flatten_command_for(verdict.flatten, reason="ops_halt_flatten")
        return verdict.to_directive(), flatten

    def on_obs(self, obs: StreamObs) -> None:
        """Push one event-driven statistical-stream observation (fills/cost/vol/edge/funding) into the
        kill machine (§5/§6). The VM strategy projects an order/position/fill event into the relevant
        OPS stream's :class:`StreamObs` and routes it here so the SPC detectors (§7.2/§7.3/§7.4/§7.7/
        §7.8) actually accumulate against their frozen limits in production — not just per-bar binaries."""
        self.engine.on_obs(obs)

    def on_fill_health(self, fillrate_realized: float, n_orders: int,
                       exit_cannot_fill: bool = False) -> None:
        """Read realized fill-rate health for the §7.4 fill-rate-collapse trigger (event-driven)."""
        self.engine.on_fill_health(fillrate_realized, n_orders, exit_cannot_fill=exit_cannot_fill)

    def report_flatten_outcome(self, *, rejected: bool = False, partial: bool = False,
                               venue_confirmed_flat: bool = False) -> None:
        """Relay the venue's flatten outcome back into the kill machine (§9.1c)."""
        self.engine.on_flatten_outcome(
            rejected=rejected, partial=partial, venue_confirmed_flat=venue_confirmed_flat
        )


__all__ = ["OpsAdapter"]
