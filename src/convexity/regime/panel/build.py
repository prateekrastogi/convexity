"""Thin adapter over ``convexity.panel.panel.Panel`` (REGIME.md §1 Invariant N).

REGIME does NOT rebuild the panel, the signal, the portfolio construction, or the overlay mechanics
(§1). This module is a STRICT RE-KEY of the already-built ``Panel`` fields into the daily arrays the
REGIME statistics (§2.2 ``D_t`` cross-sectional dispersion, §2.3 ``V_t`` BTC realized vol) consume —
it does NOT recompute daily closes, funding sums, or listed masks (Invariant N: "a test asserts
``panel/build.py``'s outputs are a strict re-key ..., not an independent recomputation").

The one derived quantity here is daily SIMPLE RETURNS from the panel close, computed exactly as
``convexity.regime.overlay._daily_returns`` already does (same masking, same NaN convention) so there is
ONE return convention across REGIME and the overlay it gates — not a second, subtly-different
recomputation. We import that helper directly rather than re-deriving it (Invariant N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.regime.overlay import _daily_returns
from convexity.panel.panel import Panel


@dataclass(frozen=True)
class RegimeInputs:
    """The re-keyed daily arrays REGIME's measures read (REGIME.md §1, §2). A strict view of the
    ``Panel``: ``daily_returns`` is the panel-close simple-return array (via the shared
    ``overlay._daily_returns``), ``listed`` and ``symbols`` are passed through unchanged, and
    ``btc_col`` is the column index of the sleeve's market symbol (§2.3)."""

    daily_returns: np.ndarray  # (n_days, n_sym) daily simple returns, NaN off-span
    listed: np.ndarray         # (n_days, n_sym) point-in-time mask (passed through, §1 Inv N)
    symbols: tuple[str, ...]
    btc_col: int
    n_days: int
    n_sym: int


def build_regime_inputs(panel: Panel, *, market_symbol: str = "BTCUSDT") -> RegimeInputs:
    """Re-key a ``Panel`` into :class:`RegimeInputs` (REGIME.md §1 Invariant N — re-key, not recompute).

    ``daily_returns`` reuses ``convexity.regime.overlay._daily_returns`` verbatim (the ONE return
    convention). ``listed``/``symbols`` are passed through by reference — no recomputation. Raises
    ``ValueError`` if the market symbol is absent (conservative under uncertainty, §0 P6: a missing
    market column is a hard error, never a silent HEALTHY default)."""
    if market_symbol not in panel.symbols:
        raise ValueError(
            f"market_symbol {market_symbol!r} not in panel.symbols; REGIME's V_t (REGIME.md §2.3) "
            "cannot be computed without it (conservative-under-uncertainty, §0 P6)."
        )
    return RegimeInputs(
        daily_returns=_daily_returns(panel),
        listed=panel.listed,
        symbols=panel.symbols,
        btc_col=panel.symbols.index(market_symbol),
        n_days=panel.n_days,
        n_sym=panel.n_symbols,
    )
