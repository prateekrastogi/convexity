"""SYSTEM/Nautilus reconciliation OK/mismatch → the ``recon`` binary stream (§5.5, Invariant RECON).

Spec: OPS §5.5, P5. SYSTEM/Nautilus emit ``reconciliation_ok: bool`` + the mismatch detail (venue-vs-
book position delta, working-order mismatch, account-balance mismatch). OPS does **NOT** re-derive any
of this; it MONITORS the boolean. A mismatch is a binary circuit-breaker trigger (§7.5), not a
statistical drift — there is no "small acceptable mismatch": either the book matches the venue or it
does not (Invariant RECON). Re-implementing the reconciliation logic here would be the bug; consuming
the flag + detail and firing a breaker on it is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from convexity.ops.types import StreamObs


@dataclass(frozen=True)
class ReconFact:
    """The reconciliation RESULT from SYSTEM/Nautilus (§5.5). OPS reads it; it never re-derives it."""

    reconciliation_ok: bool
    detail: Mapping[str, float] = field(default_factory=dict)  # position/order/balance deltas


class ReconStream:
    """Causal adapter reconciliation flag → ``recon`` binary :class:`StreamObs` (§5.5)."""

    name = "recon"

    def __init__(self) -> None:
        self._idx = 0

    def observe(self, fact: ReconFact, ts: int = 0) -> StreamObs:
        """A binary observation: ``value=1.0`` on a MISMATCH (the trigger condition), else 0.0.

        ``binary=True`` so the brake treats it as an immediate trigger with no SPC/warm-up (§6.8(a))
        — a recon mismatch on the first bar is still an immediate HALT (§7.5).
        """
        self._idx += 1
        mismatch = not fact.reconciliation_ok
        return StreamObs(
            stream="recon",
            sample_idx=self._idx,
            value=1.0 if mismatch else 0.0,
            ts=ts,
            binary=True,
            meta=dict(fact.detail),
        )


__all__ = ["ReconFact", "ReconStream"]
