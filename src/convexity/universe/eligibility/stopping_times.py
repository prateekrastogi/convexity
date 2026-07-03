"""Listing / delisting stopping times read off observed data (UNIVERSE.md §1.2).

For instrument ``i`` (UNIVERSE.md §1.2):

    tau_list_i   = inf{ t : i has a valid, tradeable quote at t }
    tau_delist_i = inf{ t : i permanently ceases to have a valid, tradeable quote at t }  (= +inf if never)

Both are stopping times w.r.t. the filtration ``F_t``: the event ``{tau_list_i <= t}`` is
``F_t``-measurable because "did this instrument have a valid quote by ``t``" is answerable from data
available at ``t`` — no future lookahead (§1.2, Karatzas & Shreve 1991 Ch. 1).

This module does NOT recompute the listing span: it READS it off ``src/convexity/panel/panel.py``'s
already-validated per-symbol ``listed`` mask (``listed = ~np.isnan(close)``, ``first_day``/``last_day``
derived per-symbol from its OWN observed span — "no union backfill", the panel docstring's words).
``panel.first_day[sym]`` IS ``tau_list_i`` and ``panel.last_day[sym]`` is the last live index (so
``tau_delist_i = last_day + 1``, the first index at which the name is no longer tradeable). This is the
stopping-time rule §1.2/§1.3 prove is survivorship-bias-free, NOT the ``R``-style backward roster.
"""

from __future__ import annotations

import numpy as np

from convexity.panel.panel import Panel

DELIST_NEVER: int = 2**62  # sentinel for tau_delist = +inf (never observed to delist), §1.2


def tau_list(panel: Panel) -> np.ndarray:
    """Per-symbol listing stopping time ``tau_list_i`` as a panel row index (UNIVERSE.md §1.2).

    Returns shape ``(n_symbols,)`` int: the first row index at which each symbol was live, read from
    ``panel.first_day`` (the panel's own per-symbol first-valid-day, NEVER the panel-global earliest
    date). This is ``inf{ t : listed_i(t) }`` restricted to observed data.
    """
    return np.array([panel.first_day[s] for s in panel.symbols], dtype=np.int64)


def tau_delist(panel: Panel) -> np.ndarray:
    """Per-symbol delisting stopping time ``tau_delist_i`` as a panel row index (UNIVERSE.md §1.2).

    Returns shape ``(n_symbols,)`` int: the first row index at which the symbol is NO LONGER tradeable
    = ``last_day + 1``. A symbol whose last live day is the panel's final row is treated as never
    observed to delist within the sample (``tau_delist_i = DELIST_NEVER``), matching §1.2's ``= +inf``
    convention — its eligibility is not curtailed by the panel simply ending.
    """
    last_row = panel.n_days - 1
    out = np.empty(panel.n_symbols, dtype=np.int64)
    for j, s in enumerate(panel.symbols):
        ld = panel.last_day[s]
        out[j] = DELIST_NEVER if ld >= last_row else ld + 1
    return out


def in_tradeable_life(panel: Panel, t: int) -> np.ndarray:
    """Base eligibility window ``1{ tau_list_i <= t < tau_delist_i }`` (UNIVERSE.md §1.2).

    Returns shape ``(n_symbols,)`` bool: whether each symbol is in its tradeable life at rebalance
    index ``t``. This is EXACTLY ``panel.listed[t]`` (the panel already encodes the per-symbol live
    span); it is expressed here in stopping-time terms so §1.2's proof and the history gate (§1.4)
    read against one vocabulary. ``F_t``-measurable by construction (reads only row ``t``).
    """
    if t < 0 or t >= panel.n_days:
        return np.zeros(panel.n_symbols, dtype=bool)
    return panel.listed[t].astype(bool)
