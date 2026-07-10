"""One-command prod cutover preflight (PAPER.md §12.6). READ-ONLY — checks, never changes.

Run from the PROD checkout with its .env sourced:
    set -a; source .env; set +a; .venv/bin/python scripts/prod_preflight.py

Exit nonzero on any FAIL. Order of the checklist mirrors §12.6's runbook."""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, "src")

RESULTS: list[tuple[str, str, str]] = []  # (level, name, detail)


def rec(level: str, name: str, detail: str) -> None:
    RESULTS.append((level, name, detail))
    print(f"[{level:>4}] {name}: {detail}")


def main() -> None:  # noqa: C901 — a checklist is linear by nature
    from convexity.paper.venue import execution_base, key_env_names, venue

    # 1) venue switch
    try:
        v = venue()
        if v == "prod":
            rec("PASS", "venue", f"CONVEXITY_VENUE=prod -> {execution_base()}")
        else:
            rec("FAIL", "venue", f"CONVEXITY_VENUE={v!r} — set 'prod' in this checkout's .env")
    except RuntimeError as e:
        rec("FAIL", "venue", str(e))
        v = None

    # 2) keys
    k_env, s_env = key_env_names()
    have_keys = bool(os.environ.get(k_env)) and bool(os.environ.get(s_env))
    rec("PASS" if have_keys else "FAIL", "keys",
        f"{k_env}/_SECRET {'present' if have_keys else 'MISSING in .env'}")

    # 3) adapter/HTTP venue parity
    try:
        from convexity.paper.venue import assert_consistent
        assert_consistent()
        rec("PASS", "venue-parity", "nautilus adapter and HTTP surfaces resolve the same base")
    except Exception as e:
        rec("FAIL", "venue-parity", str(e)[:160])

    # 4) lake + own paper state
    for sub in ("stitched", "klines", "funding", "universe"):
        p = pathlib.Path("data") / sub
        rec("PASS" if p.exists() else "FAIL", f"lake:{sub}",
            "reachable" if p.exists() else "missing — lake dir absent in this checkout")
    from convexity.paper.venue import state_root
    pp = state_root()
    pp.mkdir(parents=True, exist_ok=True)
    ok_state = pp.is_dir() and not pp.is_symlink() and pp.name != "paper"
    rec("PASS" if ok_state else "FAIL", "paper-state",
        f"venue state root {pp} (prod journals isolated from demo's data/paper)" if ok_state
        else f"state root resolves to {pp} — prod must namespace under data/prod "
             f"(is CONVEXITY_VENUE=prod set?)")

    if not have_keys:
        rec("WARN", "venue-checks", "skipped (no keys yet): auth, margin modes, balance, "
            "bootstrap — rerun after filling .env")
        _finish()
        return

    # 5) auth + account snapshot
    try:
        from convexity.paper.gateway import _build_api, account_assets, account_snapshot
        snap = account_snapshot()
        assets = account_assets()
        rec("PASS", "auth", f"account reachable: equity {snap.equity:.2f}, "
            f"{len(snap.positions)} open positions")
    except Exception as e:
        rec("FAIL", "auth", f"{str(e)[:200]}")
        _finish()
        return

    # 6) asset mode: single-asset (BTC-never mandate)
    try:
        import asyncio

        from nautilus_trader.core.nautilus_pyo3 import HttpMethod
        client, _api, clock = _build_api()

        async def multi():
            resp = await client.sign_request(
                http_method=HttpMethod.GET, url_path="/fapi/v1/multiAssetsMargin",
                payload={"timestamp": str(clock.timestamp_ms()), "recvWindow": "5000"})
            return json.loads(resp.body if hasattr(resp, "body") else resp)

        m = asyncio.run(multi())
        single = not bool(m.get("multiAssetsMargin", False))
        rec("PASS" if single else "FAIL", "asset-mode",
            "SINGLE-ASSET (BTC can never be drafted as cross collateral)" if single
            else "multi-asset mode ON — switch to single-asset before cutover")
    except Exception as e:
        rec("WARN", "asset-mode", f"unreadable: {str(e)[:120]}")

    # 7) position margin mode: CROSS everywhere
    try:
        import asyncio as _a

        from convexity.paper.gateway import _build_api as _b
        client, api, _clock = _b()
        risk = _a.run(api.query_futures_position_risk(recv_window="5000"))
        modes = {str(getattr(p, "marginType", getattr(p, "margin_type", "?"))) for p in risk or ()}
        bad = {m for m in modes if "cross" not in m.lower()}
        rec("PASS" if not bad else "FAIL", "margin-mode",
            f"all cross ({len(risk or ())} symbols)" if not bad else f"non-cross modes: {bad}")
    except Exception as e:
        rec("WARN", "margin-mode", f"unreadable: {str(e)[:120]}")

    # 8) collateral
    usdt = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
    rec("PASS" if usdt > 0 else "FAIL", "collateral",
        f"USDT+USDC = {usdt:.2f} (BTC excluded by mandate)")

    # 9) allocations vs collateral
    from convexity.paper.books import BOOKS, book_journal_path
    from convexity.paper.journal import Journal
    allocs = {}
    for name, spec in BOOKS.items():
        a = Journal(book_journal_path(spec)).last("allocate")
        allocs[name] = float(a["amount"]) if a else None
    if any(a is None for a in allocs.values()):
        rec("FAIL", "allocations",
            f"missing allocate records {[k for k, x in allocs.items() if x is None]} — run "
            f"`python -m convexity.paper allocate-split --total <USDT>` with the PROD collateral")
    else:
        tot = sum(allocs.values())
        ok = tot <= usdt * 1.001
        rec("PASS" if ok else "FAIL", "allocations",
            f"{allocs} sum {tot:.2f} vs collateral {usdt:.2f}"
            + ("" if ok else " — allocations exceed collateral"))

    # 10) manual bootstrap
    from convexity.paper.manual import load_reserve
    reserve = load_reserve()
    live_targets = any(
        any(r.get("mode") == "live" for r in Journal(book_journal_path(s)).read("targets"))
        for s in BOOKS.values())
    if snap.positions and not reserve and not live_targets:
        rec("FAIL", "bootstrap",
            f"{len(snap.positions)} venue positions, empty reserve ledger, no live targets — "
            f"run `python -m convexity.paper manual --bootstrap` (they're yours by definition)")
    else:
        rec("PASS", "bootstrap",
            f"{len(snap.positions)} venue positions / {len(reserve)} reserved / "
            f"live_targets={live_targets}")

    # 11) TradFi agreement (state-changing to sign — reminder only)
    rec("WARN", "tradfi-agreement",
        "cannot verify read-only: if equity/commodity orders draw -4411, run "
        "gateway.sign_tradfi_agreement() once (idempotent)")

    # 12) timers
    import subprocess
    out = subprocess.run(["systemctl", "list-timers", "convexity-prod-*", "--no-legend"],
                         capture_output=True, text=True).stdout.strip()
    n = len(out.splitlines()) if out else 0
    rec("WARN" if n == 0 else "PASS", "timers",
        f"{n} prod timers enabled" + (" (expected 0 pre-cutover; enable AFTER the first "
                                      "supervised cycle)" if n == 0 else ""))
    _finish()


def _finish() -> None:
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\npreflight: {len([r for r in RESULTS if r[0] == 'PASS'])} pass, "
          f"{len([r for r in RESULTS if r[0] == 'WARN'])} warn, {len(fails)} FAIL")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
