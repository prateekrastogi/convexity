"""AMBER venue-contract-status short-permission flag ``short_permitted_i(t)`` (UNIVERSE.md §4.2).

``short_permitted_i(t)`` is an ``F_t``-measurable AMBER METADATA flag: the venue contract supports
shorting at ``t`` with no active restriction (no isolated-margin lockout, no reduce-only, no
minimum-notional/leverage-tier gate that effectively excludes a small book — §4.1). This is metadata,
not a statistic: it changes rarely and is sourced from venue contract state, NOT fitted (§4.2, §8 AMBER).

This repo has no persisted per-symbol short-permission time series (§4.2 disclosed limitation), so this
module does NOT fabricate one. It provides the CONTRACT: an injected/observed per-symbol boolean, with
a conservative default (all-permitted) that a caller supplying real venue metadata overrides. The
conservative default here differs in kind from the short-LIQUIDITY threshold (``liquidity_gate.py``):
permission defaulting to True is safe only because the stricter ``short_liquid_i(t)`` gate and the
base gate ``U_t`` are ANDed on top (§4.2's three-term product), so a missing-metadata name still cannot
be short-eligible unless it also clears the liquidity and base gates.
"""

from __future__ import annotations

import numpy as np

from convexity.panel.panel import Panel


def short_permitted(
    panel: Panel,
    t: int,
    *,
    permitted_by_symbol: dict[str, bool] | None = None,
) -> np.ndarray:
    """AMBER ``short_permitted_i(t)`` flag over the panel's symbol order (UNIVERSE.md §4.2).

    ``permitted_by_symbol`` -- observed venue metadata (AMBER), a per-symbol boolean; when None every
                               symbol defaults to permitted (the base + strict-liquidity gates still
                               bind on top, §4.2). ``F_t``-measurable: venue contract status is knowable
                               at ``t`` (metadata that changes rarely, not a forecast).

    Returns shape (n_symbols,) bool. A symbol absent from ``permitted_by_symbol`` defaults to True (the
    ANDed liquidity/base gates still constrain it — §4.2's three-term product).
    """
    n_sym = panel.n_symbols
    if permitted_by_symbol is None:
        return np.ones(n_sym, dtype=bool)
    return np.array([bool(permitted_by_symbol.get(s, True)) for s in panel.symbols], dtype=bool)
