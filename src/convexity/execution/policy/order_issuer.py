"""OrderIssuer — order-type construction + idempotent client_order_id + retry/timeout supervisor (§7).

Builds the Nautilus :class:`OrderCommand` for the chosen tactic with the right TIF / flags and a
DETERMINISTIC ``client_order_id``:

  * POST_ONLY_MAKER  -> LIMIT, post-only, GTC within the step timeout, then cancel/replace.
  * LIMIT_IOC        -> LIMIT, IOC, price = touch (cap).
  * MARKETABLE_LIMIT -> LIMIT, IOC, price = touch ± through_ticks (RED) so slippage is bounded.
  * MARKET           -> MARKET (only on protective exit / abort-to-flat / halt gate).

``client_order_id = slice_id(intent_id, slice_idx, attempt)`` so a RETRY after an ambiguous timeout
reuses the SAME id and the venue de-duplicates — a retry can NEVER double-fill (Invariant Q). The
slice id scheme is DISJOINT from the stop id scheme (Invariant Q-STOP): a stop id carries the "STOP"
tag and a generation counter, never the slice_idx/attempt, so an entry-slice retry can never collide
with a stop id. Submission, acks, fills, and cancels are owned by Nautilus (Invariant N); this builds
the command and supervises timeouts/retries only.
"""

from __future__ import annotations

import hashlib

from convexity.execution.config.resolve import ResolvedConfig
from convexity.execution.types import ExecTarget, OrderCommand, Side, Tactic

__all__ = ["slice_client_order_id", "stop_client_order_id", "OrderIssuer"]


def _hash(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]  # noqa: S324 (id, not security)


def slice_client_order_id(intent_id: str, slice_idx: int, attempt: int) -> str:
    """Deterministic entry/exit SLICE id = hash(intent_id, slice_idx, attempt) (Invariant Q).

    Human-legible prefix kept so logs and the disjointness check are obvious; the trailing hash
    guarantees uniqueness. Disjoint from the stop scheme by the absence of the "STOP" tag.
    """
    return f"{intent_id}|S{slice_idx}|A{attempt}|{_hash(intent_id, 'SLICE', slice_idx, attempt)}"


def stop_client_order_id(intent_id: str, amend_generation: int) -> str:
    """Deterministic STOP id = hash(intent_id, "STOP", amend_generation) (Invariant Q-STOP).

    The "STOP" tag + a generation counter INDEPENDENT of slice_idx/attempt guarantee a stop id can
    never collide with an entry/exit slice id. ``amend_generation`` is derivable from observed state
    (the current invalidation generation in WorkingState), so it survives restart.
    """
    return f"{intent_id}|STOP|G{amend_generation}|{_hash(intent_id, 'STOP', amend_generation)}"


class OrderIssuer:
    """Builds venue-valid order commands with idempotent ids (§7). Supervises retries, owns no book."""

    def issue(
        self,
        tactic: Tactic,
        target: ExecTarget,
        price: float,
        qty: float,
        reduce_only: bool,
        cfg: ResolvedConfig,
        *,
        slice_idx: int = 0,
        attempt: int = 0,
        action: str = "submit",
    ) -> OrderCommand:
        """Construct the Nautilus order command for ``tactic`` with the right TIF / flags + a
        deterministic ``client_order_id`` (§7 / Invariant Q)."""
        coid = slice_client_order_id(target.intent_id, slice_idx, attempt)
        side: Side = target.side
        through = cfg.defaults.marketable_through_ticks * cfg.instrument.tick_size

        if tactic == "POST_ONLY_MAKER":
            kind, tif, post_only = "LIMIT", "GTC", True
            px = price
        elif tactic == "LIMIT_IOC":
            kind, tif, post_only = "LIMIT", "IOC", False
            px = price
        elif tactic == "MARKETABLE_LIMIT":
            kind, tif, post_only = "LIMIT", "IOC", False
            px = price + side * through  # touch ± through_ticks so slippage is bounded by build (§7)
        else:  # MARKET
            kind, tif, post_only = "MARKET", "IOC", False
            px = 0.0

        return OrderCommand(
            intent_id=target.intent_id,
            instrument=target.instrument,
            side=side,
            kind=kind,  # type: ignore[arg-type]
            qty=abs(qty),
            price=px,
            time_in_force=tif,  # type: ignore[arg-type]
            post_only=post_only,
            reduce_only=reduce_only,
            client_order_id=coid,
            action=action,  # type: ignore[arg-type]
            reason_codes=(f"tactic:{tactic}",),
        )

    def backoff_seconds(self, attempt: int, cfg: ResolvedConfig) -> float:
        """Exponential backoff for a rate-limit reject: base · multiplier^attempt (§7)."""
        d = cfg.defaults
        return float(d.backoff_base_s * (d.backoff_multiplier ** max(attempt, 0)))
