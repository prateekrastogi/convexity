"""Multi-book registry (PAPER.md §11): one Binance demo account, N DISJOINT virtual books.

One venue account holds every strategy's positions, so raw account state MISLEADS per-strategy
book-keeping (the user mandate that created this module, 2026-07-05). Each book therefore owns:

  * a SYMBOL DOMAIN — the candidates.md sleeves it may ever touch. Domains are pairwise disjoint
    (test-pinned) and scope EVERYTHING venue-facing: order intents, the orphan sweep, carried-book
    reads, recon. A book can neither trade nor close another book's positions by construction.
  * a JOURNAL — its own append-only JSONL; equity marks, settles, targets, flows never mix.
  * VIRTUAL CAPITAL — an ``allocate`` journal record splits the shared account equity into per-book
    NAVs. In allocation mode a book's equity evolves by its OWN frozen-accounting settles (+ its
    journaled flows); the live session journals a venue-truth ``book_recon`` (domain-scoped
    positions + unrealized P&L + account totals) alongside, so live-vs-model drift is MEASURED,
    never silently blended across books. The crypto book keeps its legacy whole-account-equity
    settle until an ``allocate`` record exists (single-book back-compat; zero behavior change at
    merge) — once a second book trades the same account, the operator MUST allocate both (runbook).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.panel.candidates import ExchangeFilters, all_symbols, sleeve_filters


@dataclass(frozen=True)
class BookSpec:
    """One strategy book: identity, symbol domain, journal, default dry-run capital."""

    name: str                     # CLI/journal identity ("crypto" | "commodity")
    sleeves: tuple[str, ...]      # candidates.md sleeves = the book's symbol DOMAIN
    journal_path: str             # cwd-relative, matching the lake-dir convention
    default_nav: float            # dry-run starting NAV when no allocate/equity_mark exists yet


CRYPTO = BookSpec(
    name="crypto",
    sleeves=("crypto",),
    journal_path="data/paper/journal.jsonl",   # the original single-book journal, unchanged
    default_nav=5_000.0,
)

COMMODITY = BookSpec(
    name="commodity",
    sleeves=("commodities",),
    journal_path="data/paper/commodity/journal.jsonl",
    default_nav=2_000.0,
)

EQUITY = BookSpec(
    name="equity",
    sleeves=("equities",),
    journal_path="data/paper/equity/journal.jsonl",
    default_nav=250.0,  # ALLOC_UNITS convention: 250/1000 (PAPER.md §12.1)
)

BOOKS: dict[str, BookSpec] = {b.name: b for b in (CRYPTO, COMMODITY, EQUITY)}


def book_domain(spec: BookSpec) -> frozenset[str]:
    """Every symbol the book may ever touch (incl. discovered-overlay names for the commodity
    book, so a discovered-then-manually-traded-then-dropped name can still be swept)."""
    return frozenset(all_symbols(sleeves=spec.sleeves))


def book_filters(spec: BookSpec) -> dict[str, ExchangeFilters]:
    """The book's dry-run quantization surface (committed snapshot); live overrides with venue."""
    return sleeve_filters(spec.sleeves)


def assert_disjoint_domains() -> None:
    """No symbol may belong to two books — the invariant every venue-facing scope relies on."""
    seen: dict[str, str] = {}
    for spec in BOOKS.values():
        for sym in book_domain(spec):
            if sym in seen:
                raise AssertionError(
                    f"book domains overlap: {sym} in both {seen[sym]!r} and {spec.name!r}")
            seen[sym] = spec.name
