#!/usr/bin/env python
"""Weekly fleet performance digest -> Telegram (PAPER.md §12.6 alerting family).

Runs Sunday 12:30 UTC (after both watchdogs). Computes per-book live P&L on BOTH venues from
journal truth + venue snapshot, and sends one monospace table. Flow-fold correctness: a
journaled flow counts against the baseline only once a settle with asof >= flow asof exists
(settles fold flows weekly; an unfolded flow is not yet reflected in the equity marks)."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, "src")

from convexity.paper.books import BOOKS, book_domain, book_journal_path  # noqa: E402
from convexity.paper.gateway import account_snapshot  # noqa: E402
from convexity.paper.journal import Journal  # noqa: E402


def book_line(name, spec, snap):
    j = Journal(book_journal_path(spec))
    dom = book_domain(spec)
    upnl = sum(v for s, v in snap.unrealized.items() if s in dom)
    alloc = j.last("allocate")
    if not alloc:
        return f"{name:>9}: no allocation", None
    a_amt, a_date = float(alloc["amount"]), alloc["asof"]
    marks = j.read("equity_mark")
    eq = float(marks[-1]["equity"]) if marks else a_amt
    settles = [r for r in j.read("settle") if r["asof"] >= a_date]
    last_settle_asof = settles[-1]["asof"] if settles else ""
    folded = sum(float(r["amount"]) for r in j.read("flow")
                 if a_date <= r["asof"] <= last_settle_asof)
    pnl = eq + upnl - a_amt - folded
    base = a_amt + folded
    wk = float(settles[-1]["r_twr"]) if settles else None
    wk_s = f"{wk * 100:+.2f}%" if wk is not None else "n/a"
    short = {"crypto": "cry", "commodity": "com", "equity": "eqt"}.get(name, name[:3])
    return (f" {short}  {eq + upnl:9.2f}  {pnl:+7.1f}  {pnl / base * 100:+6.2f}%  {wk_s:>6}"), pnl


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    lines = [f"convexity — {now:%Y-%m-%d}"]
    for venue in ("demo", "prod"):
        os.environ["CONVEXITY_VENUE"] = venue
        try:
            snap = account_snapshot()
        except Exception as e:  # digest must never crash the alert chain silently
            lines.append(f"-- {venue.upper()}: snapshot failed: {str(e)[:80]}")
            continue
        tot = 0.0
        lines.append(f"{venue.upper()}  {snap.equity:.2f} | "
                     f"margin {snap.maint_margin / max(snap.equity, 1e-9) * 100:.1f}% | "
                     f"{len(snap.positions)} pos")
        lines.append(" bk         eq     P&L        %    week")
        for name, spec in BOOKS.items():
            line, pnl = book_line(name, spec, snap)
            lines.append(line)
            tot += pnl or 0.0
        lines.append(f" TOT             {tot:+7.1f}")
    text = "\n".join(lines)
    print(text)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat:
        body = urllib.parse.urlencode(
            {"chat_id": chat, "text": f"<pre>{text}</pre>", "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = json.loads(r.read()).get("ok")
        print(f"telegram sent: {ok}")
    else:
        print("telegram creds absent — digest printed only")


if __name__ == "__main__":
    main()
