#!/usr/bin/env python
"""Weekly fleet digest -> Telegram. VENUE-TRUTH accounting (rewritten 2026-08-06).

Per book we now report BOTH, never blended (the book_recon convention):
  nav  = model equity — the frozen-accounting chain that drives SIZING and the brake
  vP&L = REAL money: venue realized + commission + funding attributed by symbol->book domain,
         plus that book's current unrealized. This is what actually happened to the cash.
The previous version reported model_equity + unrealized, which double-counted the pre-settle
portion of unrealized (owner caught it 2026-08-06: book column summed 1028.96 vs account
1013.74). The model-vs-venue drift is now shown explicitly — it is a diagnostic, not noise:
prod holds only the min-notional-clearing subset of each target book and fills at live
Saturday prices, so divergence from the model is expected and worth watching."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, "src")

from convexity.paper.books import BOOKS, book_domain, book_journal_path  # noqa: E402
from convexity.paper.gateway import _build_api, account_snapshot  # noqa: E402
from convexity.paper.journal import Journal  # noqa: E402

TYPES = ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")


def venue_income(start_ms: int) -> dict:
    """symbol -> summed income across P&L/fees/funding, paged weekly (1000-row API cap)."""
    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client, _api, clock = _build_api()
    now = clock.timestamp_ms()
    out: dict[str, float] = {}

    async def page(itype, t0, t1):
        resp = await client.sign_request(
            http_method=HttpMethod.GET, url_path="/fapi/v1/income",
            payload={"incomeType": itype, "startTime": str(int(t0)), "endTime": str(int(t1)),
                     "limit": "1000", "timestamp": str(clock.timestamp_ms()),
                     "recvWindow": "5000"})
        return json.loads(resp.body if hasattr(resp, "body") else resp)

    week = 7 * 86_400_000
    for itype in TYPES:
        t = start_ms
        while t < now:
            for e in asyncio.run(page(itype, t, min(t + week, now))):
                s = e.get("symbol") or ""
                if s:
                    out[s] = out.get(s, 0.0) + float(e["income"])
            t += week
    return out


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    lines = [f"convexity — {now:%Y-%m-%d}"]
    for venue in ("demo", "prod"):
        os.environ["CONVEXITY_VENUE"] = venue
        try:
            snap = account_snapshot()
            starts = []
            for spec in BOOKS.values():
                a = Journal(book_journal_path(spec)).last("allocate")
                if a:
                    starts.append(dt.datetime.fromisoformat(a["asof"]).replace(
                        tzinfo=dt.timezone.utc))
            start_ms = int(min(starts).timestamp() * 1000) if starts else int(
                (now - dt.timedelta(days=30)).timestamp() * 1000)
            inc = venue_income(start_ms)
        except Exception as e:
            lines.append(f"{venue.upper()}: unavailable ({str(e)[:40]})")
            continue
        lines.append(f"{venue.upper()} {snap.equity:.2f} | margin "
                     f"{snap.maint_margin / max(snap.equity, 1e-9) * 100:.1f}% | "
                     f"{len(snap.positions)} pos")
        lines.append(" bk       nav     vP&L    week")
        t_nav = t_pnl = 0.0
        for name, spec in BOOKS.items():
            j = Journal(book_journal_path(spec))
            dom = book_domain(spec)
            mark = j.last("equity_mark")
            nav = float(mark["equity"]) if mark else 0.0
            realized = sum(v for s, v in inc.items() if s in dom)
            upnl = sum(v for s, v in snap.unrealized.items() if s in dom)
            vpnl = realized + upnl
            settles = j.read("settle")
            wk = f"{float(settles[-1]['r_twr']) * 100:+.2f}%" if settles else "n/a"
            short = {"crypto": "cry", "commodity": "com", "equity": "eqt"}[name]
            lines.append(f" {short} {nav:9.2f} {vpnl:+8.2f} {wk:>7}")
            t_nav += nav
            t_pnl += vpnl
        lines.append(f" TOT {t_nav:9.2f} {t_pnl:+8.2f}")
        # execution quality of the last cycle (PAPER.md §12.7): coverage is what drives
        # model-vs-venue drift; shortfall/maker show whether execution is behaving.
        tcas = [(n, Journal(book_journal_path(s)).last("tca")) for n, s in BOOKS.items()]
        tcas = [(n, r) for n, r in tcas if r]
        if tcas:
            cov = [r.get("coverage_pct") for _, r in tcas if r.get("coverage_pct") is not None]
            # demo is Binance's SIMULATOR: it reported 0% maker on 1476 USDT of fills while
            # prod reported 97-100% through the identical code path (2026-08-06). Its fill
            # classification and prices are synthetic, so tag them — coverage is still real
            # (it comes from our own positions vs targets, not from venue fill reporting).
            sim = " (sim)" if venue == "demo" else ""
            # per-book values after the range, so a single bad book is visible rather than
            # hidden inside a min..max span (owner request 2026-08-10)
            by = {n: r for n, r in tcas}
            def _cell(key, fmt):
                out = []
                for short, n in (("cry", "crypto"), ("com", "commodity"), ("eqt", "equity")):
                    v = by.get(n, {}).get(key)
                    if v is None:
                        v = 0.0   # no trades that cycle -> no cost incurred (not "unknown")
                    out.append(f"{short} {v:{fmt}}")
                return " ".join(out)
            avg_cov = sum(cov) / len(cov)
            lines.append(f" cov {avg_cov:.0f}% | " + _cell("coverage_pct", ".0f"))
            # NOTIONAL-WEIGHTED average: a book that traded 410 USDT and one that traded 52
            # must not count equally toward "what did execution cost us".
            wn = sum((by.get(n, {}).get("traded_notional") or 0.0)
                     for n in ("crypto", "commodity", "equity")
                     if by.get(n, {}).get("shortfall_bp") is not None)
            wsf = sum((by.get(n, {}).get("shortfall_bp") or 0.0)
                      * (by.get(n, {}).get("traded_notional") or 0.0)
                      for n in ("crypto", "commodity", "equity")
                      if by.get(n, {}).get("shortfall_bp") is not None)
            avg_sf = (wsf / wn) if wn > 0 else 0.0
            lines.append(f" sf {avg_sf:+.0f}bp{sim} | " + _cell("shortfall_bp", "+.0f"))
        # PENDING flows (journaled, not yet folded by a settle) already sit in the venue
        # balance while absent from every nav mark — without this the drift line reads as a
        # huge loss for a week after any deposit (caught 2026-08-06 after the +479).
        pending = 0.0
        for s in BOOKS.values():
            jj = Journal(book_journal_path(s))
            st = jj.last("settle")
            cut = st["asof"] if st else "0000-00-00"
            pending += sum(float(f["amount"]) for f in jj.read("flow") if f["asof"] > cut)
        drift = t_nav + pending - snap.equity
        lines.append(f" drift {drift:+.1f}"
                     + (f" ({pending:+.0f} pending fold)" if abs(pending) > 0.5 else ""))
    text = "\n".join(lines)
    print(text)

    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat:
        body = urllib.parse.urlencode(
            {"chat_id": chat, "text": f"<pre>{text}</pre>", "parse_mode": "HTML"}).encode()
        with urllib.request.urlopen(urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage", data=body), timeout=20) as r:
            print("telegram sent:", json.loads(r.read()).get("ok"))
    else:
        print("telegram creds absent — printed only")


if __name__ == "__main__":
    main()
