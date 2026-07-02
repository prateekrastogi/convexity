"""Book-vs-exchange position-snapshot adapter (``docs/specs/OPS.md`` sec 8 ``ops/streams/recon.py``).

sec 2's monitored-streams table names ``position_delta`` (book vs exchange position snapshot, daily
cadence) as the input to sec 1 reconciliation. This module adapts a snapshot of (book, exchange)
positions per instrument into the per-instrument :class:`ReconResult` ladder, delegating the pure
tolerance/tier logic to :mod:`convexity.ops.reconcile.streams_map` (sec 1.2 / sec 1.4) — the split
mirrors sec 8's directory layout (``streams/`` adapts realized facts into the monitored series;
``reconcile/streams_map.py`` holds the ``epsilon_i`` map + escalation ladder).

Pure / offline: a ``PositionSnapshot`` is whatever the live loop or a replay produces; no Nautilus
import here (that is ``ops/nautilus/adapter.py``, out of v1 scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.reconcile.streams_map import (
    ReconResult,
    ReconTier,
    epsilon_i,
    reconcile_instrument,
    reconcile_stale,
    worst_tier,
)

__all__ = ["PositionSnapshot", "reconcile_book", "worst_tier", "ReconResult", "ReconTier"]


@dataclass(frozen=True)
class PositionSnapshot:
    """One instrument's reconciliation input at a daily check (OPS.md sec 1 / sec 2).

    ``book_position``/``exchange_position`` are the two quantities that must agree to ``epsilon_i``
    (sec 1.1). ``lot_size``/``target_notional``/``price`` are the AMBER metadata the tolerance is
    scaled to (sec 1.2). ``days_unreadable`` > 0 marks a snapshot whose READ itself failed (sec 1.4):
    such an instrument is routed to :func:`reconcile_stale`, never silently compared as ``OK``.
    """

    instrument: str
    book_position: float
    exchange_position: float
    lot_size: float
    target_notional: float
    price: float
    days_unreadable: int = 0


def reconcile_book(
    snapshots: list[PositionSnapshot],
    *,
    k_lot: float,
    k_notional: float,
    m_warn: float,
    staleness_max_days: int,
    as_of: int,
) -> list[ReconResult]:
    """Reconcile a whole book's daily snapshot into per-instrument :class:`ReconResult`s (sec 1.4).

    Each instrument is either compared against its scaled ``epsilon_i`` (sec 1.2) or, if its read
    failed (``days_unreadable > 0``), routed to the stale ladder (sec 1.4) — a failed read is never
    silently treated as ``OK``. The caller reduces the list to a book-level tier via
    :func:`~convexity.ops.reconcile.streams_map.worst_tier` (a BREACH on ANY instrument escalates the
    whole book, P3) and feeds the surviving alarms to the brake (:mod:`convexity.ops.brake.triggers`).
    """
    results: list[ReconResult] = []
    for s in snapshots:
        if s.days_unreadable > 0:
            results.append(
                reconcile_stale(
                    instrument=s.instrument,
                    days_unreadable=s.days_unreadable,
                    staleness_max_days=staleness_max_days,
                    as_of=as_of,
                )
            )
            continue
        eps = epsilon_i(
            lot_size_i=s.lot_size,
            target_notional_i=s.target_notional,
            price_i=s.price,
            k_lot=k_lot,
            k_notional=k_notional,
        )
        results.append(
            reconcile_instrument(
                instrument=s.instrument,
                book_position=s.book_position,
                exchange_position=s.exchange_position,
                epsilon=eps,
                m_warn=m_warn,
                as_of=as_of,
            )
        )
    return results
