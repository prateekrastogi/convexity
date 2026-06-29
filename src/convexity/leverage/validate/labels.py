"""Realized-vs-planned loss labels + gap-through events (LEVERAGE §12, firewalled).

VALIDATION-ONLY (Invariant V): never imported by ``ceilings/``, ``combine/``,
``allocate/``, ``vol/``. Scores LEVERAGE on REALIZED vs PLANNED loss, net of costs.
Future-looking quantities (the realized adverse excursion after entry) live ONLY here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossLabel:
    """One trade's planned vs realized loss on BOTH bases (Inv M) + gap-through flag (§12)."""

    instrument: str
    planned_loss_pct_margin: float
    realized_loss_pct_margin: float
    planned_loss_pct_equity: float
    realized_loss_pct_equity: float
    gapped_through_stop: bool
    binding_ceiling: str


def label_trade(
    *,
    instrument: str,
    planned_loss_pct_margin: float,
    realized_adverse_move: float,
    stop_distance: float,
    leverage: float,
    position_margin: float,
    account_equity: float,
    binding_ceiling: str,
) -> LossLabel:
    """Build a realized-vs-planned loss label. ``realized_adverse_move`` is the worst
    adverse move actually observed (a FUTURE quantity — validation-only). A gap-through
    is flagged when the realized adverse move exceeds the stop distance (the hard stop
    did not contain it — §8.3 only the tail ceiling survives gaps)."""
    realized_pct_margin = leverage * realized_adverse_move  # L·move, fraction of margin
    realized_loss_quote = realized_pct_margin * position_margin
    realized_pct_equity = realized_loss_quote / max(account_equity, 1e-12)
    planned_pct_equity = planned_loss_pct_margin * position_margin / max(account_equity, 1e-12)
    return LossLabel(
        instrument=instrument,
        planned_loss_pct_margin=float(planned_loss_pct_margin),
        realized_loss_pct_margin=float(realized_pct_margin),
        planned_loss_pct_equity=float(planned_pct_equity),
        realized_loss_pct_equity=float(realized_pct_equity),
        gapped_through_stop=bool(realized_adverse_move > stop_distance + 1e-12),
        binding_ceiling=binding_ceiling,
    )
