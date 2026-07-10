"""Manual-position carve-out (PAPER.md §12.5) — the operator's own positions are UNTOUCHABLE.

The three books share one venue account. Domain scoping already protects each book from the
OTHERS — but a position the OPERATOR opens by hand (Binance app/web) in an in-domain symbol
would be absorbed into that book's carried state, traded toward its target, orphan-swept, or
kill-flattened. This module carves it out, AUTOMATICALLY:

  * **Attribution**: every order the gateway submits is stamped ``clientOrderId = cvx-…``. Any
    other fill on the account is the operator's by construction (app/web orders carry platform
    prefixes like ``ios_``/``android_``/``web_``, but the rule is positive identification of OUR
    orders — anything not ours is manual, whatever placed it).
  * **Auto-sync** (``auto_sync``): at every venue-facing entry point (recon, rebalance, finalize,
    kill/DERISK flatten — inside the account lock, BEFORE any position read) the fleet scans
    recent TERMINAL orders per symbol and signed-accumulates non-``cvx-`` executed quantity into
    ``data/paper/manual_reserve.json``. Manual sells decrement; a fully-closed manual position
    auto-releases. Adoption is journaled (``manual_autoreserve``).
  * **Netting**: every venue-position read nets ``book_qty = venue_qty − reserved_qty``. The book
    sizes, sweeps and closes ONLY its share. Manual fills move venue qty and reserve by the same
    amount, so the book's share is invariant under operator activity — even a manual sell INTO a
    book's symbol nets out exactly (reserve goes negative, book share unchanged).
  * **SAFETY CLAMP**: a book share may never flip sign against the venue position — if ledger
    and venue disagree beyond what the book owns, the share clamps to 0 and the discrepancy is
    journaled LOUDLY rather than letting the book "correct" a phantom position.
  * **First-run anchor**: sync state (``data/paper/manual_sync.json``) is created with
    ``deploy_ms = now`` — orders predating the feature (all unstamped, ours included) are never
    scanned, so nothing historical is misclassified as manual.
  * **Known edge**: a manual LIMIT order resting non-terminal for over the 14-day lookback before
    filling can escape adoption; app orders are effectively immediate, and the clamp + notes
    surface any residue. Wrong adoptions can be corrected by hand-editing the ledger JSON.
  * **Margin**: in cross mode manual positions share the account's collateral. The guardrails
    deliberately read WHOLE-ACCOUNT margin (conservative — a big manual position tightens the
    books' margin budget), but DERISK/kill flatten only ever closes BOOK shares, never the
    reserve.
"""

from __future__ import annotations

import json
import pathlib

MANUAL_RESERVE_PATH = "data/paper/manual_reserve.json"
SYNC_STATE_PATH = "data/paper/manual_sync.json"
OUR_ORDER_PREFIX = "cvx-"
TERMINAL_STATUSES = {"FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED",
                     "EXPIRED_IN_MATCH"}
LOOKBACK_MS = 14 * 86_400_000  # re-scan window for orders that went terminal late
_EPS = 1e-12


def load_reserve(path: str | pathlib.Path = MANUAL_RESERVE_PATH) -> dict[str, float]:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {str(s): float(q) for s, q in raw.items() if abs(float(q)) > _EPS}
    except Exception:
        # a corrupt ledger must FAIL SAFE for the operator: treating every reserve as absent
        # would let a book close manual positions — so do the opposite: raise loudly, the
        # session aborts, the operator fixes the file. Never trade through an unreadable ledger.
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
            # clamp to zero, never invert. Ledger and venue disagree; surface it, don't trade it.
            notes.append(
                f"{sym}: reserve {res:+g} exceeds venue {venue_qty:+g} — book share clamped "
                f"to 0 (correct data/paper/manual_reserve.json if the adoption was wrong)")
            continue
        book_qty = raw
        notes.append(f"{sym}: venue {venue_qty:+g} − reserved {res:+g} = book {book_qty:+g}")
        if abs(book_qty) <= _EPS:
            continue
        out_pos[sym] = book_qty
        if sym in unrealized and abs(venue_qty) > _EPS:
            out_upnl[sym] = unrealized[sym] * (book_qty / venue_qty)
    # reserves for symbols with NO venue position at all: normally auto-released by the sell
    # fill's adoption — a survivor here means the ledger and venue disagree; surface it.
    for sym, res in reserve.items():
        if sym not in positions and abs(res) > _EPS:
            notes.append(
                f"{sym}: reserve {res:+g} but NO venue position — stale reserve? (auto-sync "
                f"should have released it; check data/paper/manual_reserve.json)")
    return out_pos, out_upnl, notes


def book_share_qty(symbol: str, venue_qty: float,
                   reserve: dict[str, float] | None = None) -> float:
    """Single-symbol convenience for the finalize remainder check."""
    pos, _, _ = net_book_positions({symbol: venue_qty}, None, reserve)
    return pos.get(symbol, 0.0)


# ------------------------------------------------------------------ automatic fill attribution


def load_sync_state(now_ms: int,
                    path: str | pathlib.Path = SYNC_STATE_PATH) -> dict:
    """{"deploy_ms": int, "symbols": {sym: {"watermark": ms, "done": {orderId: update_ms}}}}.

    Created on first call with ``deploy_ms = now_ms`` — the anchor that keeps every pre-feature
    order (none of which carries the ``cvx-`` stamp) out of attribution forever."""
    p = pathlib.Path(path)
    if not p.exists():
        state = {"deploy_ms": int(now_ms), "symbols": {}}
        _save_sync_state(state, p)
        return state
    try:
        return json.loads(p.read_text())
    except Exception:
        raise RuntimeError(
            f"manual sync state unreadable ({p}) — restoring it wrong could double-adopt fills; "
            f"fix or delete it (deletion re-anchors deploy_ms to now: fills between the last "
            f"good sync and now would then need a hand edit of the reserve ledger)")


def _save_sync_state(state: dict, path: str | pathlib.Path = SYNC_STATE_PATH) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.rename(p)


def auto_sync(
    fetch_orders,
    venue_positions: dict[str, float],
    now_ms: int,
    ledger_path: str | pathlib.Path = MANUAL_RESERVE_PATH,
    state_path: str | pathlib.Path = SYNC_STATE_PATH,
) -> tuple[list[dict], list[str]]:
    """Adopt the operator's manual fills into the reserve ledger.

    ``fetch_orders(symbol, start_ms) -> list[dict]`` (Binance /fapi/v1/allOrders shape) is
    injected for testability. Scans symbols with a venue position or an existing reserve; only
    TERMINAL orders are processed (watermark + per-order dedup make re-runs idempotent; a
    14-day lookback catches orders that rested before going terminal). Returns
    (adopted_records, human_notes). Raises on any venue failure — the caller must abort rather
    than trade unsynced (an unadopted manual fill would be absorbed by the book)."""
    state = load_sync_state(now_ms, state_path)
    reserve = load_reserve(ledger_path)
    symbols = sorted({s for s, q in venue_positions.items() if abs(q) > _EPS} | set(reserve))
    adopted: list[dict] = []
    notes: list[str] = []
    changed = False
    for sym in symbols:
        ss = state["symbols"].setdefault(
            sym, {"watermark": int(state["deploy_ms"]), "done": {}})
        start = max(int(state["deploy_ms"]), int(ss["watermark"]) - LOOKBACK_MS)
        for o in sorted(fetch_orders(sym, start), key=lambda o: int(o.get("updateTime", 0))):
            oid = str(o.get("orderId"))
            if oid in ss["done"]:
                continue
            if str(o.get("status", "")) not in TERMINAL_STATUSES:
                continue  # still live — no watermark advance for it; caught once terminal
            ut = int(o.get("updateTime", 0))
            ss["done"][oid] = ut
            ss["watermark"] = max(int(ss["watermark"]), ut)
            if str(o.get("clientOrderId", "")).startswith(OUR_ORDER_PREFIX):
                continue  # ours by stamp
            ex = float(o.get("executedQty", 0.0) or 0.0)
            if ex <= _EPS:
                continue  # cancelled/expired with nothing done
            delta = ex if str(o.get("side", "")).upper() == "BUY" else -ex
            new = reserve.get(sym, 0.0) + delta
            if abs(new) <= 1e-9:
                reserve.pop(sym, None)  # manual position fully closed -> auto-release
            else:
                reserve[sym] = new
            changed = True
            rec = {"symbol": sym, "orderId": oid, "qty": delta,
                   "price": float(o.get("avgPrice", 0.0) or 0.0),
                   "client_id": str(o.get("clientOrderId", ""))[:24],
                   "reserve_after": round(reserve.get(sym, 0.0), 10)}
            adopted.append(rec)
            notes.append(f"{sym}: adopted manual fill {delta:+g} (order {oid}, "
                         f"{rec['client_id']!r}) → reserve {reserve.get(sym, 0.0):+g}")
        cut = int(ss["watermark"]) - LOOKBACK_MS
        ss["done"] = {k: v for k, v in ss["done"].items() if int(v) >= cut}
    if changed:
        save_reserve(reserve, ledger_path)
    _save_sync_state(state, state_path)
    return adopted, notes
