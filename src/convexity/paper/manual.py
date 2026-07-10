"""Manual-position carve-out (PAPER.md §12.5) — the operator's own positions are UNTOUCHABLE.

The three books share one venue account. Domain scoping already protects each book from the
OTHERS — but a position the OPERATOR opens by hand in an in-domain symbol (e.g. a discretionary
NVDAUSDT long inside the equity book's domain) would be absorbed into that book's carried state,
traded toward its target, orphan-swept, or kill-flattened. This module carves it out:

  * ``data/paper/manual_reserve.json`` — ONE operator-maintained ledger (shared lake path, all
    books/worktrees see the same file): symbol -> signed reserved qty (base units). Maintained
    ONLY via the CLI (``python -m convexity.paper manual --reserve SYM QTY | --release SYM |
    --list``), never auto-populated: EXPLICIT ownership beats silent adoption (a book must never
    guess which venue quantity is the operator's).
  * Every venue-position read a book performs (carried book, orphan sweep, kill flatten, recon
    counts, finalize remainders, book_recon unrealized) first NETS the reserve out:
    ``book_qty = venue_qty − reserved_qty``. The book sizes, sweeps and closes ONLY its share.
  * SAFETY CLAMP: a book share may never flip sign against the venue position (if the reserve
    exceeds what actually exists — the operator closed part of their position without releasing
    the reserve — the book share clamps to 0 and the discrepancy is journaled LOUDLY rather than
    letting the book "correct" a phantom short/long).
  * Margin note: in cross mode manual positions share the account's collateral. The guardrails
    deliberately read WHOLE-ACCOUNT margin (conservative — a big manual position tightens the
    books' margin budget), but DERISK/kill flatten only ever closes BOOK shares, never the
    reserve.
"""

from __future__ import annotations

import json
import pathlib

MANUAL_RESERVE_PATH = "data/paper/manual_reserve.json"
_EPS = 1e-12


def load_reserve(path: str | pathlib.Path = MANUAL_RESERVE_PATH) -> dict[str, float]:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {str(s): float(q) for s, q in raw.items() if abs(float(q)) > _EPS}
    except Exception:
        # a corrupt ledger must FAIL SAFE for the operator: treat every reserve as absent would
        # let a book close manual positions — so do the opposite: raise loudly, the session
        # aborts, the operator fixes the file. Never trade through an unreadable ledger.
        raise RuntimeError(
            f"manual reserve ledger unreadable ({p}) — fix or remove it before any live session; "
            f"trading through a corrupt ledger could close operator-owned positions")


def save_reserve(reserve: dict[str, float],
                 path: str | pathlib.Path = MANUAL_RESERVE_PATH) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {s: q for s, q in sorted(reserve.items()) if abs(q) > _EPS}, indent=1))
    tmp.rename(p)


def net_book_positions(
    positions: dict[str, float],
    unrealized: dict[str, float] | None = None,
    reserve: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Net the operator's reserve out of a (domain-scoped) venue-position view.

    Returns (book_positions, book_unrealized, notes). Unrealized is attributed proportionally by
    quantity share (venue reports it per whole position). Notes carry every carve-out and every
    discrepancy — the caller journals them."""
    reserve = load_reserve() if reserve is None else reserve
    unrealized = unrealized or {}
    out_pos: dict[str, float] = {}
    out_upnl: dict[str, float] = {}
    notes: list[str] = []
    for sym, venue_qty in positions.items():
        res = reserve.get(sym, 0.0)
        if abs(res) <= _EPS:
            out_pos[sym] = venue_qty
            if sym in unrealized:
                out_upnl[sym] = unrealized[sym]
            continue
        raw = venue_qty - res
        if raw * venue_qty < 0:
            # reserve exceeds the venue position (same sign) — book share would flip sign:
            # clamp to zero, never invert. The operator likely closed part of their manual
            # position without releasing the reserve.
            notes.append(
                f"{sym}: reserve {res:+g} exceeds venue {venue_qty:+g} — book share clamped "
                f"to 0 (release/adjust the reserve via `manual --release`)")
            continue
        book_qty = raw
        notes.append(f"{sym}: venue {venue_qty:+g} − reserved {res:+g} = book {book_qty:+g}")
        if abs(book_qty) <= _EPS:
            continue
        out_pos[sym] = book_qty
        if sym in unrealized and abs(venue_qty) > _EPS:
            out_upnl[sym] = unrealized[sym] * (book_qty / venue_qty)
    # reserves for symbols with NO venue position at all: the operator's position is gone —
    # surface it (stale reserve), harmless to the book but worth a loud nudge.
    for sym, res in reserve.items():
        if sym not in positions and abs(res) > _EPS:
            notes.append(
                f"{sym}: reserve {res:+g} but NO venue position — stale reserve? "
                f"(`manual --release {sym}` if you closed it)")
    return out_pos, out_upnl, notes


def book_share_qty(symbol: str, venue_qty: float,
                   reserve: dict[str, float] | None = None) -> float:
    """Single-symbol convenience for the finalize remainder check."""
    pos, _, _ = net_book_positions({symbol: venue_qty}, None, reserve)
    return pos.get(symbol, 0.0)
