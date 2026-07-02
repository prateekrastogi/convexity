"""AMBER candidate symbol tables — the maximum CANDIDATE pool, a superset of ever-eligible tickers.

UNIVERSE.md §6 / Invariant CANDIDATE-VS-ELIGIBLE: ``research/UNIVERSE.md``'s per-sleeve symbol tables
are the maximum candidate pool, curated ONCE for sector/factor diversity and NEVER re-tuned on
realized P&L (§8 AMBER, §9 edge case 6). ``U_t`` is always a ``t``-dependent SUBSET of this pool
computed from §1-4's point-in-time machinery — never the pool itself. A candidate symbol whose
``tau_list_i > t`` (not yet listed) or ``tau_delist_i <= t`` (already delisted) is excluded from
``U_t`` at that ``t`` regardless of its permanent presence in the candidate list.

This module is a thin AMBER loader; it REUSES ``research/universe.py::all_symbols`` (the single reader
of ``research/UNIVERSE.md``'s ```json block) and never parses the tables itself. The crypto sleeve is
the only sleeve with a live cross-sectional-momentum edge (UNIVERSE.md §0 Scope, EDGE.md).
"""

from __future__ import annotations

from convexity.panel.candidates import all_symbols


def crypto_candidates() -> tuple[str, ...]:
    """The crypto candidate pool (UNIVERSE.md §6 crypto 50-name curated list; AMBER).

    Curated once for sector/market-cap diversity (see ``research/UNIVERSE.md``'s "Crypto (50) — by
    sector" table), explicitly excluding low-mcap pump coins by sector representativeness, NOT by
    realized return (§6) — so it is NOT the survivorship-flattered ``R``-style backward roster §1.3
    proves is biased. REUSES ``convexity.panel.candidates.all_symbols`` (the one reader of the ```json block).
    """
    return tuple(all_symbols(sleeves=("crypto",)))


def candidate_symbols(sleeves: tuple[str, ...] = ("crypto",)) -> tuple[str, ...]:
    """Flat candidate pool across the requested sleeves (AMBER; UNIVERSE.md §6, §8).

    Defaults to crypto only — the sole sleeve UNIVERSE.md §0 scopes an edge to. Thin pass-through to
    ``convexity.panel.candidates.all_symbols``; the point-in-time refinement to ``U_t`` happens downstream.
    """
    return tuple(all_symbols(sleeves=sleeves))
