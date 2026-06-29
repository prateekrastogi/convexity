"""Facts layer — ingest EXECUTION's REALIZED slippage/impact (COST §5.4, §1).

COST does NOT model order-book execution (that is EXECUTION.md). It INGESTS
EXECUTION's realized per-fill slippage/impact for TCA, and consumes the TEAL-fitted
square-root coefficient ``k_i`` (with its units flag) that EXECUTION's fitter
produced from this instrument's own realized series, for the gate's forecast (§5.4).
This module defines the frozen ``ExecutionFacts`` view; it is pure Python and
exercised by the unit suite. No NautilusTrader import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutionFacts:
    """The realized execution facts COST consumes (§5.4). All FACT.

    ``realized_impact_cost`` (quote ccy) replaces the forecast in TCA (§9).
    ``fitted_k`` is the instrument's square-root coefficient (TEAL, fitted by
    EXECUTION); ``k_units`` flags whether it is a return FRACTION or bps so COST
    converts to a fraction once at ingestion (Invariant U-dim). ``fitted_k is None``
    means the conservative linear fallback governs the gate (P6).
    """

    symbol: str
    realized_impact_cost: float  # quote ccy, realized (for TCA)
    realized_half_spread_cost: float  # quote ccy, realized touch-vs-mid (for TCA)
    fitted_k: float | None  # square-root coefficient (TEAL); None -> fallback
    k_units: Literal["fraction", "bps"] = "fraction"

    def k_as_fraction(self) -> float | None:
        """Return ``fitted_k`` as a return FRACTION, converting bps once at ingestion (§5.4)."""
        if self.fitted_k is None:
            return None
        if self.k_units == "bps":
            return self.fitted_k / 1e4
        return self.fitted_k
