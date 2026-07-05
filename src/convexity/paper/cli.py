"""Paper-rebalancer CLI (PAPER.md §6, §11): settle -> target -> size -> plan -> intents -> journal.

Commands (systemd timers drive them; every run is a full pure-recompute session; EVERY command
takes ``--book {crypto,commodity}``, default crypto — each book has its own journal, symbol
domain, targets and virtual capital, PAPER.md §11):
  rebalance [--live] [--asof YYYY-MM-DD] [--nav X]   weekly session (settles the prior week first)
  finalize  [--live]                                  window-end crossing of unfilled remainders
  recon     [--live]                                  daily position/equity check + journal marks
  flow --amount X [--equity-before E]                 journal an external deposit/withdrawal
  allocate --amount X                                 set the book's virtual capital (PAPER.md §11)
  status                                              journal tail summary

Dry-run (default, no keys needed): carried book and equity come from the journal (equity evolves
by the frozen-accounting settled returns from a --nav starting point); orders are journaled, not
sent. --live swaps in the gateway for equity/positions/submission.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from convexity.cost.accounting import round_trip_friction, settle_week
from convexity.ops.config.defaults import OpsDefaults
from convexity.ops.streams.flows import FlowEvent, flow_risk_check, twr_weekly_return
from convexity.panel.candidates import all_symbols
from convexity.panel.panel import Panel, build_dollar_volume, build_panel
from convexity.paper.books import (
    BOOKS,
    BookSpec,
    assert_disjoint_domains,
    book_domain,
    book_filters,
)
from convexity.paper.capital import sizing_nav
from convexity.paper.journal import Journal
from convexity.paper.policy import (
    OrderIntent,
    build_plan,
    ensemble_target_weights,
    order_intents,
    orphan_close_intents,
    size_gross,
)
from convexity.paper.topup import extend_panel, fetch_tails
from convexity.portfolio.portfolio import realized_vol

STEP = 7

# Per-book round-trip friction for the frozen-accounting settle (fraction of notional).
# crypto: the recorded 13 bp parity anchor. commodity: Binance TradFi-perp VIP0 taker (same 5 bp
# fee schedule) + the commodity sleeve's wider 3 bp half-spread + 0.5 bp impact = 17 bp RT —
# inside the RV study's {12, 24} bp scenario band (RV_REPORT.md).
SETTLE_FRICTION: dict[str, float] = {
    "crypto": round_trip_friction(),
    "commodity": round_trip_friction(taker=5.0e-4, half_spread=3.0e-4, impact=0.5e-4),
}
# Plan (no-trade band) half-spread per book; other CostFacts inputs keep policy defaults.
PLAN_HALF_SPREAD: dict[str, float] = {"crypto": 1.0e-4, "commodity": 3.0e-4}


def _book(args: argparse.Namespace) -> BookSpec:
    return BOOKS[args.book]


def _journal(spec: BookSpec) -> Journal:
    return Journal(spec.journal_path)


def _overlay_native_funding(panel: Panel) -> Panel:
    """Real perp funding for the commodity book (PAPER.md §11.3, review finding 2026-07-05).

    The sleeve machinery treats commodities as synthetic-funding UNDERLYINGS (the research
    convention), so ``build_panel`` gives them funding=0 — but this book trades the REAL perp
    and ``data/funding`` holds its real venue stamps (backfilled weekly). Overlay them so the
    settle charges venue-true funding over the perp era; the proxy era has no stamps and stays
    at 0 (there was no perp to be charged on).
    """
    import pathlib

    import pandas as pd

    from convexity.panel.panel import _daily_funding

    funding = panel.funding.copy()
    changed = False
    for jx, sym in enumerate(panel.symbols):
        path = pathlib.Path("data/funding") / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("calc_time")
        if df.empty:
            continue
        s = _daily_funding(df["calc_time"].to_numpy(np.int64),
                           df["last_funding_rate"].to_numpy(float))
        vals = s.reindex(panel.dates).to_numpy(float)
        funding[:, jx] = np.where(panel.listed[:, jx], np.nan_to_num(vals, nan=0.0), np.nan)
        changed = True
    if not changed:
        return panel
    return Panel(dates=panel.dates, symbols=panel.symbols, close=panel.close, funding=funding,
                 listed=panel.listed, first_day=panel.first_day, last_day=panel.last_day)


def _load_panel(spec: BookSpec, topup: bool) -> Panel:
    syms = tuple(all_symbols(sleeves=spec.sleeves))
    panel = build_panel(syms)
    if panel.n_days == 0:
        raise SystemExit(f"[{spec.name}] empty panel — is the lake present under data/?")
    if spec.name == "commodity":
        panel = _overlay_native_funding(panel)
    if topup:
        last_ms = int(panel.dates[-1].value // 1_000_000)
        tails = fetch_tails(panel.symbols, last_ms)  # only rows AFTER the lake tail are appended
        panel = extend_panel(panel, tails)
    return panel


def _last_friday_row(panel: Panel, asof: str | None) -> int:
    wd = panel.dates.weekday.to_numpy()
    hi = panel.n_days if asof is None else int(
        np.searchsorted(panel.dates.to_numpy(), np.datetime64(asof), side="right"))
    rows = [i for i in range(hi) if wd[i] == 4]
    if not rows:
        raise SystemExit("no Friday row in panel window")
    return rows[-1]


def _aligned_weights(rec: dict, panel: Panel) -> np.ndarray:
    """Journaled target weights re-aligned to the CURRENT panel symbol order.

    Records carry their own ``symbols`` list (PAPER.md §11) because a book's panel can GROW
    between weeks (discovery overlay); positional alignment would silently mis-settle. A symbol
    that left the panel contributes 0 (it was closed at its last live day by the accounting).
    Legacy records without ``symbols`` are assumed positionally aligned (the frozen crypto
    single-book behavior).
    """
    w = np.asarray(rec["weights"], dtype=float)
    syms = rec.get("symbols")
    if syms is None:
        if w.shape[0] != panel.n_symbols:
            raise SystemExit(
                f"legacy targets record has {w.shape[0]} weights but panel has "
                f"{panel.n_symbols} symbols — cannot align safely; investigate the journal")
        return w
    by = dict(zip(syms, w.tolist(), strict=True))
    return np.array([by.get(s, 0.0) for s in panel.symbols], dtype=float)


def _allocation_mode(j: Journal) -> bool:
    """True once the operator has split the shared account into virtual books (PAPER.md §11)."""
    return j.last("allocate") is not None


def _book_equity(j: Journal, spec: BookSpec, fallback: float | None = None) -> float:
    mark = j.last("equity_mark")
    if mark is not None:
        return float(mark["equity"])
    alloc = j.last("allocate")
    if alloc is not None:
        return float(alloc["amount"])
    if fallback is not None:
        return float(fallback)
    return spec.default_nav


def _venue_book_state(domain: frozenset[str]):
    """(positions, unrealized_sum, account_equity, assets, unowned) — positions scoped to the
    book's domain; ``unowned`` = venue positions outside EVERY book's domain (nobody's sweep can
    ever close those, so the caller surfaces them loudly)."""
    from convexity.paper.gateway import account_assets, account_snapshot

    assets = account_assets()
    snap = account_snapshot()
    pos = {s: q for s, q in snap.positions.items() if s in domain}
    upnl = sum(v for s, v in snap.unrealized.items() if s in domain)
    all_domains: frozenset[str] = frozenset().union(*(book_domain(b) for b in BOOKS.values()))
    unowned = {s for s, q in snap.positions.items() if abs(q) > 1e-12 and s not in all_domains}
    return pos, float(upnl), float(snap.equity), assets, unowned


def _settle_prior_week(j: Journal, spec: BookSpec, panel: Panel, d_now: int, live: bool) -> None:
    """Settle the prior journaled targets through the frozen accounting; fold TWR + kill-switch.

    Allocation mode (PAPER.md §11): the book's equity evolves by its OWN settled model return +
    its journaled flows — the shared account's equity is polluted by the OTHER book's P&L and is
    only ever journaled as recon context, never folded in. Legacy mode (crypto before the first
    ``allocate``): whole-account equity, the original single-book behavior.
    """
    recs = j.read("targets")
    prev = recs[-1] if recs else None
    if prev is None:
        return
    dates = panel.dates.to_numpy()
    d_prev = int(np.searchsorted(dates, np.datetime64(prev["asof"])))
    # Settle window end by CALENDAR DATE, never by row count: the commodity panel has no
    # weekend rows in its proxy era (24x5 sessions), so "+7 rows" overshoots the week there —
    # while on crypto's 24/7 panel this resolves to exactly the old d_prev+7 row (byte-compat).
    target_end = np.datetime64(prev["asof"]) + np.timedelta64(STEP, "D")
    d1 = int(np.searchsorted(dates, target_end, side="right")) - 1
    week_complete = (d1 > d_prev and d_prev < d_now
                     and dates[d1] >= np.datetime64(prev["asof"]) + np.timedelta64(6, "D"))
    if not week_complete:
        print(f"[settle:{spec.name}] prior week (targets {prev['asof']}) not complete in the "
              f"panel yet — settle deferred, equity mark unchanged")
        return
    w_prev = _aligned_weights(prev, panel)
    scale_rec = float(prev.get("g_t", 1.0)) if spec.name == "commodity" else 1.0
    # the commodity book's recorded deployable IS the vol-targeted book (RV_REPORT.md), so
    # its equity evolves on the SCALED weights; crypto keeps its frozen legacy semantics
    # (unit-target settle, G_t applied at the venue) to leave the recorded track unchanged.
    w_prev = w_prev * scale_rec
    alloc_mode = _allocation_mode(j)
    # Model-evolved books must pay their REAL weekly rebalance turnover: carry the WEEK-BEFORE
    # targets in as w_prev so settle_week's turnover is |target - carried|, not drift-only.
    # (Legacy crypto keeps w_prev=w_target: its live equity comes from the venue, where the
    # cost is already real — changing it would rewrite the recorded track's semantics.)
    w_carry_in = w_prev
    if (alloc_mode or spec.name == "commodity") and len(recs) >= 2:
        p2 = recs[-2]
        w_carry_in = _aligned_weights(p2, panel)
        if spec.name == "commodity":
            w_carry_in = w_carry_in * float(p2.get("g_t", 1.0))
    res, _ = settle_week(panel.close, panel.funding, panel.listed, d0=d_prev, d1=d1,
                         w_target=w_prev, w_prev=w_carry_in,
                         friction=SETTLE_FRICTION[spec.name])
    eq_start = _book_equity(j, spec)
    flows = tuple(
        FlowEvent(ts=int(r["ts"]), amount=float(r["amount"]),
                  equity_before=float(r["equity_before"]))
        for r in j.read("flow") if prev["asof"] < r["asof"] <= str(panel.dates[d_now].date()))
    if live and not alloc_mode:
        # legacy single-book path: the account IS the book (crypto back-compat).
        from convexity.paper.gateway import account_assets
        assets = account_assets()
        eq_end = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
    else:
        eq_end = eq_start * (1.0 + res.net) + sum(f.amount for f in flows)
    r_twr = twr_weekly_return(eq_start, eq_end, flows)
    week_end = str(panel.dates[d1].date())
    j.append("settle", week_end, r_twr=r_twr, r_model=res.net, equity_start=eq_start,
             equity_end=eq_end, n_flows=len(flows),
             mode=("live" if live else "dry") + ("" if not alloc_mode else "/alloc"))
    j.append("equity_mark", week_end, equity=eq_end)
    print(f"[settle:{spec.name}] week->{week_end}: r_twr={r_twr:+.4%} (model {res.net:+.4%}) "
          f"equity {eq_start:.2f}->{eq_end:.2f} flows={len(flows)}")


def _crypto_targets(panel: Panel, d: int, j: Journal, *, live: bool = False):  # noqa: ARG001
    """The validated crypto band-ensemble targets + quarter-Kelly gross (PAPER.md §3)."""
    from convexity.portfolio.engine import XSecConfig, band_configs, run, run_ensemble
    from convexity.system import metrics

    w_target = ensemble_target_weights(panel, d)
    hist = run_ensemble(panel, band_configs())
    trials = np.array([metrics.compute(
        run(panel, XSecConfig(signal="momentum", lookback=lb, quantile="quintile",
                              weighting="volscaled")).net).sharpe_ann for lb in (12, 14, 16)])
    dvol = build_dollar_volume(panel)
    dvol_row = dvol[min(d, dvol.shape[0] - 1)]
    dec = size_gross(hist.net, trials, dvol_row=dvol_row, w_target=w_target,
                     symbols=panel.symbols)
    return w_target, float(dec.g_t), dec.binding_ceiling, {}


def _commodity_targets(panel: Panel, d: int, _j: Journal, *, live: bool = False):
    """The RV pair book (RV_PREREG.md via signal.rv_pairs): weights arrive ALREADY vol-targeted
    (the study's own 10% S4 overlay is the book's sizing; quarter-Kelly is deferred until the
    paper track exists — PAPER.md §11 deferral). Live sessions gate pairs on what the EXECUTION
    venue actually lists (PAPER.md §11.3 — demo TradFi listings lag prod; an unlisted leg would
    otherwise settle model P&L on positions that can never exist)."""
    from convexity.signal.rv_pairs import book_weights

    tradeable = None
    if live:
        from convexity.paper.topup import fetch_execution_venue_tradeable
        tradeable = fetch_execution_venue_tradeable()
    rvb = book_weights(panel, d, tradeable=tradeable)
    return rvb.w_unit[-1], rvb.vt_scale, "rv_vol_target", rvb.diag


TARGETS_BY_BOOK = {"crypto": _crypto_targets, "commodity": _commodity_targets}


def cmd_rebalance(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    # self-enforcing invariant (PAPER.md §11.1): the overlay grows the commodity domain at
    # RUNTIME, so the disjointness the sweep relies on is re-checked here, not just in CI.
    assert_disjoint_domains()
    panel = _load_panel(spec, topup=not args.no_topup)
    d = _last_friday_row(panel, args.asof)
    d_iso = str(panel.dates[d].date())
    print(f"[rebalance:{spec.name}] panel [{panel.dates[0].date()}..{panel.dates[-1].date()}] "
          f"rebalance row {d} = {d_iso} (Friday close)")

    _settle_prior_week(j, spec, panel, d, args.live)

    w_target, g_t, binding, diag = TARGETS_BY_BOOK[spec.name](panel, d, j, live=args.live)
    if diag:
        print(f"[signal:{spec.name}] {json.dumps(diag, default=str)[:400]}")

    alloc_mode = _allocation_mode(j)
    domain = book_domain(spec)
    if args.live:
        if alloc_mode:
            positions, upnl, acct_eq, assets, unowned = _venue_book_state(domain)
            if unowned:
                j.append("unowned_positions", d_iso, symbols=sorted(unowned))
                print(f"[live:{spec.name}] WARNING: positions outside every book's domain "
                      f"(never swept): {sorted(unowned)}")
            nav = _book_equity(j, spec)
            j.append("book_recon", d_iso, book_equity=round(nav, 2),
                     account_equity=round(acct_eq, 2), unrealized_book=round(upnl, 4),
                     n_positions=len(positions))
            nav_note = f"allocation-mode NAV (account={acct_eq:.2f} is shared context only)"
            used_assets: tuple[str, ...] = ("ALLOC",)
        else:
            if spec.name != "crypto":
                raise SystemExit(
                    f"[{spec.name}] live requires an allocation: run "
                    f"`python -m convexity.paper allocate --book {spec.name} --amount <USDT>` "
                    f"first — the shared account's equity belongs to multiple books "
                    f"(PAPER.md §11)")
            from convexity.paper.gateway import account_assets, account_snapshot
            assets = account_assets()
            snap = account_snapshot()
            positions = {s: q for s, q in snap.positions.items() if s in domain}
            est_margin = g_t * sum(assets.get(a, 0.0) for a in ("USDT",)) / 10.0
            nav_dec = sizing_nav(assets, est_initial_margin=est_margin)
            nav = nav_dec.nav
            nav_note = nav_dec.note
            used_assets = nav_dec.used_assets
    else:
        positions = {}
        nav = _book_equity(j, spec, fallback=args.nav)
        nav_note = "dry-run journal equity"
        used_assets = ("DRY",)
    print(f"[size:{spec.name}] G_t={g_t:.4f} binding={binding} | NAV={nav:.2f} "
          f"({'+'.join(used_assets)}) | {nav_note}")

    w_scaled = w_target * g_t
    prices = panel.close[d]
    if args.live:
        # LIVE: the carried book is ALWAYS the venue's actual positions (an empty account is a
        # flat book — never the journal heuristic, which a crashed run's record would poison;
        # learned in the first supervised session 2026-07-03). Scoped to THIS book's panel.
        w_carried = np.array([
            positions.get(sym, 0.0) * (prices[i] if np.isfinite(prices[i]) else 0.0) / nav
            for i, sym in enumerate(panel.symbols)])
    else:
        prev = j.last("targets")
        if prev is not None:
            # dry-run carried book = the prior week's SCALED targets (the band's carried input;
            # drift is second-order for the plan and exact in the settle, which re-drifts itself).
            w_carried = _aligned_weights(prev, panel) * float(prev["g_t"])
        else:
            w_carried = np.zeros(panel.n_symbols)
    vols = realized_vol(panel, d, window=28)
    plan = build_plan(w_scaled, w_carried, vols,
                      half_spread=PLAN_HALF_SPREAD[spec.name])
    intents = order_intents(plan, panel.symbols, prices, nav, w_target=w_scaled,
                            filters=book_filters(spec))
    turn = plan.turnover_live
    print(f"[plan:{spec.name}] turnover_live={turn:.4f} (backtest {plan.turnover_backtest:.4f}); "
          f"{len(intents)} order intents, {int((plan.trade == 0).sum())} names band-skipped/flat")

    j.append("targets", d_iso, weights=[round(float(x), 10) for x in w_target],
             symbols=list(panel.symbols),
             g_t=round(g_t, 6), nav=round(nav, 2), mode="live" if args.live else "dry",
             book=spec.name, nav_assets=list(used_assets))
    j.append("orders", d_iso, intents=[i.to_dict() for i in intents],
             mode="live" if args.live else "dry")

    if args.live:
        _submit_live(j, spec, d_iso, intents, positions, w_scaled, panel)
    else:
        for it in intents[:12]:
            sgn = "+" if it.qty > 0 else "-"
            print(f"[dry:{spec.name}] {it.symbol:>14} qty={sgn}{it.qty_str} {it.tactic} "
                  f"@ {it.price_str}")
        if len(intents) > 12:
            print(f"[dry:{spec.name}] ... +{len(intents) - 12} more (journaled)")


def _submit_live(j: Journal, spec: BookSpec, d_iso: str, intents: list[OrderIntent],
                 positions: dict[str, float], w_scaled: np.ndarray, panel: Panel) -> None:
    from convexity.execution.execution import quantize_price, quantize_qty
    from convexity.paper.gateway import GatewayError, cancel_open_orders, submit_intent
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import fetch_book_tickers, fetch_live_filters

    # Idempotent safety: a rerun (crash recovery) must never double-submit — clear any
    # resting orders on the intent symbols first.
    for sym in sorted({it.symbol for it in intents}):
        try:
            cancel_open_orders(sym)
        except GatewayError:
            pass
    # LIVE quantization surfaces (both single public calls, learned 2026-07-03):
    # (a) maker peg = the PASSIVE touch (buy joins best bid, sell best ask) — last-price pegs
    #     draw -5022 post-only rejections; (b) tick/lot from the venue's CURRENT exchangeInfo
    #     — the candidates.md snapshot drifts and draws -4014.
    book = fetch_book_tickers()
    live_f = fetch_live_filters()
    ok = failed = 0
    import httpx as _httpx

    for it in intents:
        live_it = it
        # refresh THIS symbol's quote at submission time — the batch snapshot goes stale
        # over a ~60s sequential loop and stale pegs draw -5022 (learned 2026-07-03).
        try:
            _bt = _httpx.get("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                             params={"symbol": it.symbol}, timeout=5).json()
            bidask = (float(_bt["bidPrice"]), float(_bt["askPrice"]))
        except Exception:
            bidask = book.get(it.symbol)
        lf = live_f.get(it.symbol)
        if bidask and lf and it.tactic == "POST_ONLY_MAKER":
            tick, lot, min_qty, min_notional = lf
            side = 1 if it.qty > 0 else -1
            bid, ask = bidask
            # degenerate book (halted/delisting on the venue: empty side or crossed/locked
            # quotes -- e.g. NFPUSDT on demo showed bid=0.0) => untradeable this week; skip
            # loudly rather than fire orders the venue rejects with generic -4014/-5022s.
            if bid <= 0.0 or ask <= 0.0 or ask <= bid:
                j.append("order_skip", d_iso, symbol=it.symbol,
                         reason=f"degenerate book bid={bid} ask={ask} (halted/delisted?)")
                print(f"[live:{spec.name}] {it.symbol} SKIPPED: degenerate book "
                      f"bid={bid} ask={ask}")
                continue
            # one tick INSIDE the passive touch: joining the exact best bid/ask still
            # crosses when the market ticks through during submit latency (-5022 cluster,
            # learned 2026-07-03); one tick of queue-priority sacrifice buys post-only safety.
            peg = (bid - tick) if side == 1 else (ask + tick)
            ref = quantize_price(peg, tick, side=side)
            qty = quantize_qty(it.qty, lot_size=lot, min_qty=min_qty)
            if qty == 0.0 or abs(qty) * ref < min_notional:
                j.append("order_skip", d_iso, symbol=it.symbol,
                         reason="below live min_qty/min_notional after re-quantize")
                continue
            live_it = OrderIntent(symbol=it.symbol, qty=qty, tactic=it.tactic,
                                  ref_price=ref, slice_idx=it.slice_idx,
                                  n_slices=it.n_slices, reduce_only=it.reduce_only,
                                  qty_str=fmt_step(abs(qty), lot),
                                  price_str=fmt_step(ref, tick))
        try:
            resp = submit_intent(live_it)
            ok += 1
            print(f"[live:{spec.name}] {live_it.symbol} {live_it.qty:+} {live_it.tactic} "
                  f"@ {live_it.price_str} -> {resp['status']}")
        except GatewayError as e:
            failed += 1
            print(f"[live:{spec.name}] {live_it.symbol} FAILED (isolated, book continues): {e}")
            j.append("order_error", d_iso, symbol=live_it.symbol, error=str(e)[:300])
    j.append("submitted", d_iso, n_ok=ok, n_failed=failed)
    print(f"[live:{spec.name}] submitted {ok}/{len(intents)} (failed {failed}, journaled)")
    # ORPHAN SWEEP (PAPER.md §4.1 + §11): close every venue position IN THIS BOOK'S DOMAIN that
    # no intent covers and no target wants — names rotated OUT of the panel between weeks or
    # closes the band/min-notional dropped. Reduce-only MARKET (reduceOnly exempts MIN_NOTIONAL);
    # sub-lot dust that cannot be closed by quantity is journaled loudly, never silently retried.
    # Positions OUTSIDE the domain belong to the other book and are untouchable by construction.
    domain = book_domain(spec)
    covered = {it.symbol for it in intents}
    targets_by_sym = {sym: float(w_scaled[i]) for i, sym in enumerate(panel.symbols)}
    live_px = {s: (b[0] + b[1]) / 2 for s, b in book.items()
               if b and b[0] > 0 and b[1] > 0}
    orphans = orphan_close_intents(positions, covered, targets_by_sym, live_f, live_px,
                                   domain=domain)
    o_ok = o_failed = 0
    for oc in orphans:
        try:
            submit_intent(oc)
            o_ok += 1
            print(f"[live:{spec.name}] ORPHAN CLOSE {oc.symbol} {oc.qty:+} reduce-only MARKET")
        except GatewayError as e:
            o_failed += 1
            j.append("order_error", d_iso, symbol=oc.symbol,
                     error=f"orphan close: {str(e)[:280]}")
            print(f"[live:{spec.name}] ORPHAN CLOSE {oc.symbol} FAILED (isolated): {e}")
    dust = sorted({s for s, a in positions.items()
                   if abs(a) > 1e-12 and s in domain and s not in covered
                   and abs(targets_by_sym.get(s, 0.0)) < 1e-12}
                  - {oc.symbol for oc in orphans})
    if orphans or dust:
        j.append("orphan_sweep", d_iso, n_closed=o_ok, n_failed=o_failed, dust=dust)
        print(f"[live:{spec.name}] orphan sweep: {o_ok} closed, {o_failed} failed"
              + (f"; sub-lot dust left: {dust}" if dust else ""))


def cmd_finalize(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    orders = j.last("orders")
    if orders is None:
        print(f"[finalize:{spec.name}] no orders journaled")
        return
    if not args.live:
        print(f"[finalize:{spec.name}] dry-run: would cancel+cross remainders for "
              f"{len(orders['intents'])} intents")
        return
    if orders.get("mode") != "live":
        # a dry rehearsal must never arm the live finalize: crossing dry intents as MARKET
        # orders sized off a dry NAV is exactly the footgun the mode field exists to prevent.
        print(f"[finalize:{spec.name}] last orders record ({orders['asof']}) is mode="
              f"{orders.get('mode')!r}, not live — nothing to cross")
        return
    from convexity.execution.execution import quantize_qty
    from convexity.paper.gateway import (
        GatewayError,
        account_snapshot,
        cancel_open_orders,
        submit_intent,
    )
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import fetch_live_filters

    syms = {it["symbol"] for it in orders["intents"]}
    for s in sorted(syms):
        cancel_open_orders(s)
    snap = account_snapshot()
    crossed = 0
    live_f = fetch_live_filters()
    failed = 0
    for it in orders["intents"]:
        want = float(it["qty"])
        have = snap.positions.get(it["symbol"], 0.0)  # net position vs intended delta: v1 heuristic
        remainder = want if abs(have) < 1e-12 else 0.0
        if remainder != 0.0:
            lf = live_f.get(it["symbol"])
            if lf:
                _tick, lot, min_qty, _mn = lf
                remainder = quantize_qty(remainder, lot_size=lot, min_qty=min_qty)
                if remainder == 0.0:
                    continue
            try:
                submit_intent(OrderIntent(symbol=it["symbol"], qty=remainder, tactic="MARKET",
                                          ref_price=it["ref_price"], slice_idx=0, n_slices=1,
                                          reduce_only=bool(it.get("reduce_only", False)),
                                          qty_str=fmt_step(abs(remainder), lf[1]) if lf else ""))
                crossed += 1
            except GatewayError as e:
                failed += 1
                j.append("order_error", orders["asof"], symbol=it["symbol"],
                         error=f"finalize: {str(e)[:280]}")
                print(f"[finalize:{spec.name}] {it['symbol']} FAILED (isolated): {e}")
    j.append("finalized", orders["asof"], crossed=crossed, failed=failed)
    print(f"[finalize:{spec.name}] crossed {crossed} remainders ({failed} failed, journaled)")


def cmd_recon(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    if not args.live:
        t = j.last("targets")
        print(f"[recon:{spec.name}] dry-run: last targets {t['asof'] if t else None}; journal ok")
        return
    alloc_mode = _allocation_mode(j)
    if not alloc_mode and spec.name != "crypto":
        # same guard as rebalance: without an allocation the whole-account equity would be
        # journaled INTO this book's chain and poison every later sizing/settle (PAPER.md §11.2)
        raise SystemExit(
            f"[{spec.name}] live recon requires an allocation: run "
            f"`python -m convexity.paper allocate --book {spec.name} --amount <USDT>` first")
    domain = book_domain(spec)
    positions, upnl, acct_eq, assets, unowned = _venue_book_state(domain)
    if unowned:
        # hygiene backstop: a position OUTSIDE EVERY book's domain belongs to nobody — the
        # domain-scoped sweeps will never close it, so it must at least be visible (journal+log)
        j.append("unowned_positions", "recon", symbols=sorted(unowned))
        print(f"[recon:{spec.name}] WARNING: venue positions outside every book's domain "
              f"(no sweep will ever close these): {sorted(unowned)}")
    if alloc_mode:
        eq = _book_equity(j, spec)
        # allocation mode: the daily mark is the BOOK's model equity; account equity + the
        # book's venue unrealized are journaled as recon context (drift is measured, PAPER.md §11)
        j.append("book_recon", "recon", book_equity=round(eq, 2),
                 account_equity=round(acct_eq, 2), unrealized_book=round(upnl, 4),
                 n_positions=len(positions))
    else:
        eq = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
        j.append("equity_mark", "recon", equity=eq)
    t = j.last("targets")
    n_tgt = int(np.count_nonzero(np.asarray(t["weights"]))) if t else 0
    j.append("recon", t["asof"] if t else "none", n_positions=len(positions), n_targets=n_tgt,
             equity=round(eq, 2), account_equity=round(acct_eq, 2),
             excluded_btc=float(assets.get("BTC", 0.0)))
    print(f"[recon:{spec.name}] book_equity={eq:.2f} account={acct_eq:.2f} "
          f"(BTC excluded: {assets.get('BTC', 0.0)}) positions={len(positions)} targets={n_tgt}")


def cmd_flow(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    mark = j.last("equity_mark")
    eq_before = args.equity_before if args.equity_before is not None else (
        float(mark["equity"]) if mark else spec.default_nav)
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    ev = FlowEvent(ts=int(now64.astype("datetime64[ms]").astype(np.int64)),
                   amount=args.amount, equity_before=eq_before)
    directive = flow_risk_check(ev, OpsDefaults())
    j.append("flow", asof, amount=ev.amount, equity_before=ev.equity_before, ts_ms=ev.ts)
    j.append("directive", asof, action=directive.action, reason=directive.reason,
             resize_factor=directive.resize_factor)
    print(f"[flow:{spec.name}] {ev.amount:+.2f} on equity {eq_before:.2f} -> {directive.action}: "
          f"{directive.reason}")


def cmd_allocate(args: argparse.Namespace) -> None:
    """Set the book's virtual capital (PAPER.md §11). The operator splits the ONE demo account
    into per-book NAVs; each book's equity then evolves by its own settles. Re-allocating is a
    new record (append-only) — treat a mid-life re-allocation as a flow event for the TWR."""
    spec = _book(args)
    j = _journal(spec)
    if not (args.amount > 0):
        raise SystemExit(f"[allocate:{spec.name}] amount must be positive, got {args.amount}")
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    j.append("allocate", asof, amount=float(args.amount), book=spec.name)
    j.append("equity_mark", asof, equity=float(args.amount))
    other = [b for b in BOOKS.values() if b.name != spec.name]
    others = {b.name: Journal(b.journal_path).last("allocate") for b in other}
    total = float(args.amount) + sum(float(a["amount"]) for a in others.values() if a)
    print(f"[allocate:{spec.name}] {args.amount:.2f} journaled; all-book allocations sum to "
          f"{total:.2f} — verify this stays within the account's USDT+USDC (runbook)")


def cmd_status(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    print(f"[status] book={spec.name} journal={spec.journal_path}")
    for kind in ("allocate", "targets", "orders", "settle", "equity_mark", "book_recon",
                 "flow", "directive", "recon"):
        rec = j.last(kind)
        if rec is None:
            continue
        brief = {k: v for k, v in rec.items() if k not in ("weights", "intents", "symbols")}
        print(f"[{kind}] {json.dumps(brief)[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="convexity.paper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_book(p):
        p.add_argument("--book", choices=sorted(BOOKS), default="crypto")

    r = sub.add_parser("rebalance")
    add_book(r)
    r.add_argument("--live", action="store_true")
    r.add_argument("--asof", default=None)
    r.add_argument("--nav", type=float, default=None)
    r.add_argument("--no-topup", action="store_true")
    r.set_defaults(fn=cmd_rebalance)
    f = sub.add_parser("finalize")
    add_book(f)
    f.add_argument("--live", action="store_true")
    f.set_defaults(fn=cmd_finalize)
    c = sub.add_parser("recon")
    add_book(c)
    c.add_argument("--live", action="store_true")
    c.set_defaults(fn=cmd_recon)
    fl = sub.add_parser("flow")
    add_book(fl)
    fl.add_argument("--amount", type=float, required=True)
    fl.add_argument("--equity-before", type=float, default=None)
    fl.set_defaults(fn=cmd_flow)
    al = sub.add_parser("allocate")
    add_book(al)
    al.add_argument("--amount", type=float, required=True)
    al.set_defaults(fn=cmd_allocate)
    s = sub.add_parser("status")
    add_book(s)
    s.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
