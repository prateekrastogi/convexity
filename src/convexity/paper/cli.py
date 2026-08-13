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
import contextlib
import json
import pathlib

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
    book_journal_path,
)
from convexity.paper.capital import sizing_nav
from convexity.paper.journal import Journal
from convexity.paper.policy import (
    OrderIntent,
    build_plan,
    ensemble_target_weights,
    order_intents,
    orphan_close_intents,
    renormalize_achievable,
    size_gross,
)
from convexity.paper.lock import account_lock
from convexity.paper.topup import extend_panel, fetch_tails
from convexity.portfolio.portfolio import realized_vol

STEP = 7

# Per-book round-trip friction for the frozen-accounting settle (fraction of notional).
# crypto: the recorded 13 bp parity anchor. commodity: Binance TradFi-perp VIP0 taker (same 5 bp
# fee schedule) + the commodity sleeve's wider 3 bp half-spread + 0.5 bp impact = 17 bp RT —
# inside the RV study's {12, 24} bp scenario band (RV_REPORT.md).
# equity: 13 bp RT — the D1-validated backtest's own primary cost (parity anchor, PAPER.md §12.1).
SETTLE_FRICTION: dict[str, float] = {
    "crypto": round_trip_friction(),
    "commodity": round_trip_friction(taker=5.0e-4, half_spread=3.0e-4, impact=0.5e-4),
    "equity": round_trip_friction(),
}
# Plan (no-trade band) half-spread per book; other CostFacts inputs keep policy defaults.
PLAN_HALF_SPREAD: dict[str, float] = {"crypto": 1.0e-4, "commodity": 3.0e-4, "equity": 3.0e-4}


def _book(args: argparse.Namespace) -> BookSpec:
    return BOOKS[args.book]


def _journal(spec: BookSpec) -> Journal:
    return Journal(book_journal_path(spec))


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
    if spec.name in ("commodity", "equity"):
        # equity added 2026-07-15 (live catch): venue funding on the equity book runs ~5bp/wk
        # of notional (~2%/yr at current leverage) — real cost the synthetic-underlying panel
        # (funding=0) was silently omitting from the model settle (PAPER.md §12.1 note).
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


def _journal_venue(j: Journal, label: str) -> None:
    """Journal + assert the session's venue resolution (PAPER.md §12.6): every live session
    records which venue it executed on, and aborts if the adapter and the HTTP surfaces would
    resolve differently."""
    from convexity.paper.venue import assert_consistent, describe

    assert_consistent()
    info = describe()
    j.append("venue", label, **info)
    print(f"[venue] {info['venue']} exec={info['execution_base']}")


def _already_settled(j: Journal, week_end: str) -> bool:
    """Settle idempotency (audit P0-2): a crash between the settle append and the targets
    append must not double-fold the same week's return into the equity chain on rerun."""
    s = j.last("settle")
    return s is not None and s.get("asof") == week_end


def _carry_in_targets(recs: list[dict]) -> dict | None:
    """The week-before targets record for settle turnover, SKIPPING same-asof duplicates
    (audit P0-2): a crash-rerun appends a second targets record for the same Friday; naive
    recs[-2] would then be this week again and the settle's turnover cost would read zero."""
    prev = recs[-1]
    for r in reversed(recs[:-1]):
        if r.get("asof") != prev.get("asof"):
            return r
    return None


def _position_drift(targets_rec: dict, positions: dict[str, float],
                    prices: dict[str, float]) -> tuple[float, list[list]]:
    """Per-symbol venue-vs-target notional drift in quote units (audit P2-14 observability).

    Target notional = w_unit x g_t x nav — the venue holds SCALED weights in every book's
    semantics. Only symbols with a usable price are compared; sub-1-USDT diffs are noise."""
    tw = dict(zip(targets_rec.get("symbols", ()), targets_rec.get("weights", ())))
    g = float(targets_rec.get("g_t", 1.0))
    nav = float(targets_rec.get("nav", 0.0))
    drift: list[list] = []
    for sym in sorted(set(positions) | {s for s, w in tw.items() if abs(float(w)) > 1e-9}):
        px = prices.get(sym)
        if not px or px <= 0:
            continue
        d = positions.get(sym, 0.0) * px - float(tw.get(sym, 0.0)) * g * nav
        if abs(d) > 1.0:
            drift.append([sym, round(d, 2)])
    drift.sort(key=lambda x: -abs(x[1]))
    return round(sum(abs(d) for _, d in drift), 2), drift


def _manual_autosync(j: Journal, spec: BookSpec, label: str) -> None:
    """Adopt the operator's app/manual fills into the reserve ledger (PAPER.md §12.5) BEFORE
    any venue-position read — inside the account lock at every venue-facing entry point.
    Attribution is by clientOrderId: the gateway stamps everything it submits ``cvx-``; any
    other terminal fill is the operator's and moves the reserve by its signed quantity.
    Venue failure propagates: trading unsynced could absorb an unadopted manual fill."""
    import time as _time

    from convexity.paper.gateway import account_snapshot, fetch_recent_orders
    from convexity.paper.manual import auto_sync

    snap = account_snapshot()
    adopted, notes = auto_sync(fetch_recent_orders, snap.positions,
                               now_ms=int(_time.time() * 1000))
    if adopted:
        j.append("manual_autoreserve", label, adopted=adopted)
    for line in notes:
        print(f"[manual:{spec.name}] {line}")


def _venue_book_state(domain: frozenset[str]):
    """(positions, unrealized_sum, account_equity, assets, unowned, manual_notes) — positions
    scoped to the book's domain AND net of the operator's manual reserve (PAPER.md §12.5): the
    book sizes/sweeps/closes ONLY its own share of any symbol the operator also holds by hand.
    ``unowned`` = venue positions outside EVERY book's domain that are NOT operator-reserved
    (nobody's sweep can ever close those, so the caller surfaces them loudly)."""
    from convexity.paper.gateway import account_assets, account_snapshot
    from convexity.paper.manual import load_reserve, net_book_positions

    assets = account_assets()
    snap = account_snapshot()
    reserve = load_reserve()
    pos, upnl_map, notes = net_book_positions(
        {s: q for s, q in snap.positions.items() if s in domain},
        {s: v for s, v in snap.unrealized.items() if s in domain},
        {s: q for s, q in reserve.items() if s in domain})
    upnl = sum(upnl_map.values())
    all_domains: frozenset[str] = frozenset().union(*(book_domain(b) for b in BOOKS.values()))
    unowned = {s for s, q in snap.positions.items()
               if abs(q) > 1e-12 and s not in all_domains and s not in reserve}
    return pos, float(upnl), float(snap.equity), assets, unowned, notes


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
    week_end = str(panel.dates[d1].date())
    if _already_settled(j, week_end):
        print(f"[settle:{spec.name}] week->{week_end} already settled (crash-rerun?) — "
              f"skipping; the equity chain folds each week exactly once (PAPER.md §12.6)")
        return
    w_prev = _aligned_weights(prev, panel)
    scale_rec = float(prev.get("g_t", 1.0)) if spec.name in ("commodity", "equity") else 1.0
    # the commodity book's recorded deployable IS the vol-targeted book (RV_REPORT.md), and the
    # equity book's deployable IS the leveraged/braked book (its g_t carries L_half x breadth x
    # brake — the model equity MUST evolve on scaled weights or the brake would read drawdowns
    # off an unlevered curve, PAPER.md §12.1); crypto keeps its frozen legacy
    # semantics (unit-target settle, G_t applied at the venue) to leave the recorded track
    # unchanged.
    w_prev = w_prev * scale_rec
    alloc_mode = _allocation_mode(j)
    # Model-evolved books must pay their REAL weekly rebalance turnover: carry the WEEK-BEFORE
    # targets in as w_prev so settle_week's turnover is |target - carried|, not drift-only.
    # (Legacy crypto keeps w_prev=w_target: its live equity comes from the venue, where the
    # cost is already real — changing it would rewrite the recorded track's semantics.)
    w_carry_in = w_prev
    p2 = _carry_in_targets(recs) if len(recs) >= 2 else None
    if (alloc_mode or spec.name in ("commodity", "equity")) and p2 is not None:
        w_carry_in = _aligned_weights(p2, panel)
        if spec.name in ("commodity", "equity"):
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


def _equity_flags() -> frozenset[str]:
    """Split-watch flagged names (data maintenance writes; operator clears). Fail-open: a missing
    or unreadable flags file means no exclusions — trading proceeds on last-good knowledge and
    the maintenance journal carries the staleness (PAPER.md §12.3)."""
    import pathlib

    from convexity.paper.fleetconfig import EQUITY_FLAGS_PATH

    p = pathlib.Path(EQUITY_FLAGS_PATH)
    if not p.exists():
        return frozenset()
    try:
        data = json.loads(p.read_text())
        return frozenset(s for s, meta in data.items()
                         if isinstance(meta, dict) and not meta.get("cleared", False))
    except Exception:
        return frozenset()


def _equity_targets(panel: Panel, d: int, j: Journal, *, live: bool = False):
    """The D1v2 equity trend book (PAPER.md §12.1): the validated S1 ensemble ×
    inverse-vol sizing × breadth-proportional gross × frozen half-Kelly × GZ brake@50% — the
    parity-pinned construction (tests/paper/test_equity_parity.py).

    g_t = L_HALF × breadth_frac × m_brake; the brake reads the BOOK's own journal equity chain
    through the prior mark (weekly settles + latest recon — the disclosed fidelity gap vs the
    backtest's daily marks, conservative direction). Live sessions gate on what the demo venue
    actually lists (the commodity book's pattern) and journal the model-vs-tradeable gap."""
    import pandas as pd

    from convexity.signal.equity_trend import live_decision

    dates = pd.DatetimeIndex(panel.dates[:d + 1])
    closes: dict[str, pd.Series] = {}
    for i, sym in enumerate(panel.symbols):
        c = pd.Series(panel.close[:d + 1, i], index=dates).dropna()
        if len(c):
            closes[sym] = c

    tradeable_gap: list[str] = []
    if live:
        from convexity.paper.topup import fetch_execution_venue_tradeable
        tradeable = fetch_execution_venue_tradeable()
        gated = {s: c for s, c in closes.items() if s in tradeable}
        tradeable_gap = sorted(set(closes) - set(gated))
        closes = gated

    # The book's own equity chain for the brake: settled marks in journal order, then the latest
    # live recon context (model equity + venue unrealized) as the freshest observation.
    # FLOW-NEUTRAL by construction (live catch 2026-08-01): the brake must measure PERFORMANCE
    # drawdown, never capital movements. Raw equity_mark values include deposits/withdrawals, so
    # the demo book's journaled -51% withdrawal (2026-07-11) read as a 51.4% drawdown, pinned
    # m_brake to 0 and flattened the book for two cycles while true performance DD was 0.0%.
    # Compounding the settles' TWR returns is immune to flows and matches the backtest, whose
    # equity curve is a pure return series with no flows at all (Invariant EQUITY-PARITY).
    chain = [1.0]
    for r in j.read("settle"):
        chain.append(chain[-1] * (1.0 + float(r["r_twr"])))
    br = j.last("book_recon")
    if br is not None:
        be = float(br["book_equity"])
        if be > 0:  # freshest intra-week observation, as a RETURN on the index
            chain.append(chain[-1] * (1.0 + float(br.get("unrealized_book", 0.0)) / be))
    dec = live_decision(closes, np.asarray(chain, dtype=float), flagged=_equity_flags())

    by_sym = dict(zip(dec["symbols"], dec["w_unit"], strict=True))
    w_unit = np.array([by_sym.get(s, 0.0) for s in panel.symbols], dtype=float)
    diag = {
        "breadth": round(dec["breadth"], 4), "m_brake": round(dec["m_brake"], 4),
        "n_eligible": dec["n_eligible"], "n_active": dec["n_active"],
        "n_pending_warmup": len(dec["pending"]), "excluded": dec["excluded"],
        "tradeable_gap_n": len(tradeable_gap),
    }
    if live:
        j.append("tradeable_gap", str(panel.dates[d].date()), model_only=tradeable_gap,
                 n_model_only=len(tradeable_gap))
    return w_unit, dec["g_t"], "breadth_x_halfkelly_x_brake", diag


TARGETS_BY_BOOK = {"crypto": _crypto_targets, "commodity": _commodity_targets,
                   "equity": _equity_targets}


def _our_order_ids(j: Journal, asof: str) -> dict[str, list[dict]]:
    """Every order WE placed for this cycle, across ALL journaled batches: the rebalance's
    submission_ids, each reprice pass's submission_ids, and finalize_ids crossings. Finalize
    MUST aggregate all of them — reading only the last batch would under-count fills and
    re-cross positions we already hold (PAPER.md §12.7)."""
    by_sym: dict[str, list[dict]] = {}
    for kind in ("submission_ids", "finalize_ids"):
        for rec in j.read(kind):
            if rec.get("asof") != asof:
                continue
            for o in rec.get("orders", ()):
                if o.get("orderId") is not None:
                    by_sym.setdefault(o["symbol"], []).append(o)
    return by_sym


def _executed_signed(by_sym: dict[str, list[dict]], symbol: str) -> float:
    """Signed executed quantity for ``symbol`` across our orders (venue truth per orderId)."""
    from convexity.paper.gateway import fetch_order

    tot = 0.0
    for o in by_sym.get(symbol, ()):
        od = fetch_order(symbol, int(o["orderId"]))
        ex = float(od.get("executedQty", 0.0) or 0.0)
        tot += ex if float(o["qty"]) > 0 else -ex
    return tot


def _maker_peg(bid: float, ask: float, side: int) -> float:
    """Post-only peg AT the passive touch — buy joins the best bid, sell the best ask.

    Previously one tick INSIDE the touch (2026-07-03, to dodge -5022 during submit latency).
    That forfeits queue priority and on thin TradFi books simply never fills: the fill-weighted
    maker share measured 2026-08-06 was 7% (equity) and 0% (commodity), i.e. the books were
    effectively taker-only and paid the full spread at finalize while the model assumes a
    1-3bp half-spread. At the touch we sit IN the queue; GTX still refuses to cross, and a
    -5022 rejection places nothing (the caller retries one tick inside)."""
    return bid if side == 1 else ask


def _submit_maker(j: Journal, spec: BookSpec, d_iso: str, it: OrderIntent, tick: float):
    """Submit a post-only maker order.

    PRIMARY: Binance BBO — LIMIT + ``priceMatch=QUEUE`` + GTX, i.e. the venue prices us AT the
    touch from its own book at placement. No local quote is involved, so the fetch->submit
    latency race (the whole reason the peg used to sit a tick behind) cannot happen.
    FALLBACK: the locally-priced peg, then one tick inside on -5022 — kept so a venue that
    rejects priceMatch still trades exactly as before."""
    from convexity.execution.execution import quantize_price
    from convexity.paper.gateway import GatewayError, submit_bbo, submit_intent
    from convexity.paper.policy import fmt_step

    if it.tactic == "POST_ONLY_MAKER":
        try:
            r = submit_bbo(it.symbol, it.qty, it.qty_str, "QUEUE", tif="GTX",
                           reduce_only=it.reduce_only)
            if r.get("orderId") is not None:
                return r
        except GatewayError as e:
            j.append("bbo_fallback", d_iso, symbol=it.symbol, error=str(e)[:180])
    try:
        return submit_intent(it)
    except GatewayError as e:
        if "5022" not in str(e) or it.tactic != "POST_ONLY_MAKER":
            raise
        side = 1 if it.qty > 0 else -1
        back = it.ref_price - tick if side == 1 else it.ref_price + tick
        ref = quantize_price(back, tick, side=side)
        j.append("order_repeg", d_iso, symbol=it.symbol,
                 reason=f"-5022 at touch {it.ref_price} -> retry one tick inside {ref}")
        return submit_intent(OrderIntent(
            symbol=it.symbol, qty=it.qty, tactic=it.tactic, ref_price=ref,
            slice_idx=it.slice_idx, n_slices=it.n_slices, reduce_only=it.reduce_only,
            qty_str=it.qty_str, price_str=fmt_step(ref, tick)))


def _kill_path(spec: BookSpec) -> pathlib.Path:
    from convexity.paper.venue import state_root

    return state_root() / spec.name / "KILL"


def _flatten_book_live(spec: BookSpec, j: Journal, reason: str) -> None:
    """Flatten every venue position in THIS book's domain, reduce-only (kill-switch / de-risk
    action, PAPER.md §12.2). Maker-then-taker is the finalize pass's job in
    ordinary flow; a KILL is an explicit 'get flat NOW' — reduce-only MARKET, the orphan-sweep
    convention. Domain scoping guarantees the other books' positions are untouchable."""
    from convexity.paper.gateway import GatewayError, cancel_open_orders, submit_intent
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import fetch_live_filters

    domain = book_domain(spec)
    try:
        _manual_autosync(j, spec, f"kill:{reason[:24]}")
    except Exception as e:  # a KILL must flatten even if attribution is unreachable —
        # the LAST synced reserve is still netted below, so operator positions stay protected
        print(f"[kill:{spec.name}] manual auto-sync failed ({e}) — flattening with last "
              f"synced reserve")
    positions, _upnl, _eq, _assets, _unowned, manual_notes = _venue_book_state(domain)
    for note in manual_notes:
        print(f"[kill:{spec.name}] manual carve-out (reserve NOT flattened): {note}")
    for sym in sorted(positions):
        try:
            cancel_open_orders(sym)
        except GatewayError:
            pass
    live_f = fetch_live_filters()
    from convexity.execution.execution import quantize_qty
    n_ok = n_failed = 0
    dust = []
    for sym, amt in sorted(positions.items()):
        lf = live_f.get(sym)
        if lf is None:
            dust.append(sym)
            continue
        _tick, lot, min_qty, _mn = lf
        qty = quantize_qty(-amt, lot_size=lot, min_qty=min_qty)
        if qty == 0.0:
            dust.append(sym)
            continue
        try:
            submit_intent(OrderIntent(symbol=sym, qty=qty, tactic="MARKET", ref_price=0.0,
                                      slice_idx=0, n_slices=1, reduce_only=True,
                                      qty_str=fmt_step(abs(qty), lot), price_str=""))
            n_ok += 1
        except GatewayError as e:
            n_failed += 1
            j.append("order_error", "kill", symbol=sym, error=f"flatten: {str(e)[:280]}")
    j.append("killed", "kill", reason=reason, n_closed=n_ok, n_failed=n_failed, dust=dust)
    print(f"[kill:{spec.name}] flattened {n_ok} positions ({n_failed} failed"
          + (f"; sub-lot dust: {dust}" if dust else "") + f") — {reason}")


def _check_kill(spec: BookSpec, j: Journal, live: bool) -> bool:
    """True if the book is killed (control file present). Live: flatten first. Checked INSIDE
    the account lock at the top of every venue-facing session (PAPER.md §12.2)."""
    if not _kill_path(spec).exists():
        return False
    print(f"[kill:{spec.name}] KILL file present ({_kill_path(spec)}) — book is halted; "
          f"`python -m convexity.paper kill --book {spec.name} --resume` to resume")
    if live:
        _flatten_book_live(spec, j, reason="KILL file present at session start")
    else:
        j.append("killed", "kill", reason="KILL file present (dry-run: no venue action)")
    return True


def _margin_guardrails(spec: BookSpec, j: Journal, d_iso: str, nav: float, g_t: float,
                       live: bool, fail_closed: bool = False) -> float:
    """Cross-margin software guardrails (PAPER.md §12.2). Returns the possibly
    scaled-down g_t. Dry-run: budget check only (no venue reads). ``fail_closed`` (audit P1-6):
    a SIZING session must abort when the monitor is unreadable — trading blind on margin during
    exactly the stress that makes margin matter is the failure mode; the daily recon stays
    fail-open (observability beats a dead monitor there)."""
    from convexity.paper.fleetconfig import (
        LEVERAGE_DIAL,
        LIQ_DISTANCE_MIN_MULTIPLE,
        MARGIN_ALERT_RATIO,
        MARGIN_DERISK_RATIO,
        WORST_HISTORICAL_DAY,
    )

    # (1) per-book budget: projected initial margin = gross notional / dial must fit in NAV.
    dial = LEVERAGE_DIAL.get(spec.name, 20)
    own_im = g_t * nav / dial
    if own_im > nav:
        g_scaled = dial * 0.98  # cap gross so IM fits with a hair of headroom
        j.append("margin_budget", d_iso, action="scaled_down", g_t_before=round(g_t, 6),
                 g_t_after=round(g_scaled, 6), dial=dial)
        print(f"[margin:{spec.name}] projected IM {own_im:.2f} > NAV {nav:.2f} — "
              f"g_t {g_t:.4f} -> {g_scaled:.4f}")
        g_t = g_scaled
    if not live:
        return g_t
    # (2)+(3) account monitor + liquidation distance (live only; fresh fetch INSIDE the lock).
    from convexity.paper.gateway import GatewayError, account_snapshot
    try:
        snap = account_snapshot()
    except GatewayError as e:
        j.append("margin_monitor_error", d_iso, error=str(e)[:200])
        if fail_closed:
            raise
        return g_t
    ratio = (snap.maint_margin / snap.equity) if snap.equity > 0 else 0.0
    j.append("margin_ratio", d_iso, maint_margin=round(snap.maint_margin, 4),
             initial_margin=round(snap.initial_margin, 4), account_equity=round(snap.equity, 2),
             ratio=round(ratio, 6))
    if ratio >= MARGIN_DERISK_RATIO:
        j.append("directive", d_iso, action="DERISK",
                 reason=f"account maint-margin ratio {ratio:.1%} >= {MARGIN_DERISK_RATIO:.0%} "
                        f"(priority: equity flattens first — runbook)")
        print(f"[margin:{spec.name}] DERISK: ratio {ratio:.1%} >= {MARGIN_DERISK_RATIO:.0%}")
        if spec.name == "equity":
            _flatten_book_live(spec, j, reason=f"margin DERISK at ratio {ratio:.1%}")
            return 0.0
    elif ratio >= MARGIN_ALERT_RATIO:
        j.append("directive", d_iso, action="MARGIN_ALERT",
                 reason=f"account maint-margin ratio {ratio:.1%} >= {MARGIN_ALERT_RATIO:.0%}")
        print(f"[margin:{spec.name}] ALERT: ratio {ratio:.1%}")
    # liquidation distance: equity buffer over maintenance, as a fraction of account equity,
    # must exceed LIQ_DISTANCE_MIN_MULTIPLE x the worst historical one-day move.
    buffer_frac = (snap.equity - snap.maint_margin) / snap.equity if snap.equity > 0 else 0.0
    floor = LIQ_DISTANCE_MIN_MULTIPLE * WORST_HISTORICAL_DAY
    j.append("liq_distance", d_iso, buffer_frac=round(buffer_frac, 6),
             floor=round(floor, 4), ok=bool(buffer_frac > floor))
    if buffer_frac <= floor:
        j.append("directive", d_iso, action="LIQ_DISTANCE_BREACH",
                 reason=f"buffer {buffer_frac:.1%} <= {floor:.1%} floor")
        print(f"[margin:{spec.name}] LIQ-DISTANCE BREACH: buffer {buffer_frac:.1%}")
    return g_t


def _boot_leverage_dial(spec: BookSpec, j: Journal, d_iso: str,
                        intents: list[OrderIntent]) -> None:
    """Integer exchange-leverage boot (PAPER.md §12.2): set the configured dial
    once per session for this book's traded symbols. Idempotent on the venue; errors (e.g.
    'cannot change with open position') are journaled, never fatal — the dial is a cap, not the
    sizing mechanism."""
    from convexity.paper.fleetconfig import LEVERAGE_DIAL

    dial = LEVERAGE_DIAL.get(spec.name)
    if dial is None:
        return
    from convexity.paper.gateway import GatewayError, set_leverage
    ok, failed = 0, []
    for sym in sorted({it.symbol for it in intents}):
        try:
            set_leverage(sym, dial)
            ok += 1
        except GatewayError as e:
            failed.append(sym)
            j.append("leverage_set_error", d_iso, symbol=sym, error=str(e)[:200])
    j.append("leverage_set", d_iso, dial=dial, n_ok=ok, n_failed=len(failed), failed=failed)
    print(f"[leverage:{spec.name}] dial={dial}x set on {ok} symbols"
          + (f" ({len(failed)} failed, journaled)" if failed else ""))


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
    # THE ACCOUNT LOCK (PAPER.md §12.2): everything venue-facing — state read, margin checks,
    # submissions, orphan sweep — runs in ONE critical section so no other book can interleave.
    # Signal computation stayed outside. Dry-run touches only this book's journal: no lock.
    lock_cm = account_lock(spec.name) if args.live else contextlib.nullcontext()
    with lock_cm:
        if _check_kill(spec, j, live=args.live):
            return
        if args.live:
            _journal_venue(j, d_iso)
            _manual_autosync(j, spec, d_iso)
            if alloc_mode:
                positions, upnl, acct_eq, assets, unowned, manual_notes = \
                    _venue_book_state(domain)
                if manual_notes:
                    j.append("manual_reserve", d_iso, notes=manual_notes)
                    for note in manual_notes:
                        print(f"[live:{spec.name}] manual carve-out: {note}")
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
                        f"`python -m convexity.paper allocate --book {spec.name} --amount "
                        f"<USDT>` first — the shared account's equity belongs to multiple books "
                        f"(PAPER.md §11)")
                from convexity.paper.gateway import account_assets, account_snapshot
                assets = account_assets()
                snap = account_snapshot()
                from convexity.paper.manual import load_reserve, net_book_positions
                _res = {s: q for s, q in load_reserve().items() if s in domain}
                positions, _um, manual_notes = net_book_positions(
                    {s: q for s, q in snap.positions.items() if s in domain}, None, _res)
                if manual_notes:
                    j.append("manual_reserve", d_iso, notes=manual_notes)
                    for note in manual_notes:
                        print(f"[live:{spec.name}] manual carve-out: {note}")
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
        # Cross-margin guardrails (PAPER.md §12.2): budget check both modes; monitor + liq-distance
        # live only. May scale g_t down (or flatten equity on DERISK, returning g_t=0).
        g_t = _margin_guardrails(spec, j, d_iso, nav, g_t, live=args.live,
                                 fail_closed=args.live)
        print(f"[size:{spec.name}] G_t={g_t:.4f} binding={binding} | NAV={nav:.2f} "
              f"({'+'.join(used_assets)}) | {nav_note}")

        w_scaled = w_target * g_t
        prices = panel.close[d]
        # PARTIAL-BOOK RENORMALIZATION (PAPER.md §12.8, default OFF): restore the target gross
        # across the names the venue grid actually allows this NAV to hold. Off => byte-identical
        # behaviour to before.
        from convexity.paper.fleetconfig import RENORM_PARTIAL_BOOK
        if getattr(args, "renorm", False) or RENORM_PARTIAL_BOOK.get(spec.name, False):
            w_scaled, rdiag = renormalize_achievable(
                w_scaled, panel.symbols, prices, nav, book_filters(spec))
            j.append("renorm", d_iso, **rdiag)
            if rdiag.get("applied"):
                print(f"[renorm:{spec.name}] scaled {rdiag['n_kept']}/{rdiag['n_total']} "
                      f"achievable names by {rdiag['k']}x to restore gross "
                      f"{rdiag['gross_after']}")
        if args.live:
            # LIVE: the carried book is ALWAYS the venue's actual positions (an empty account is
            # a flat book — never the journal heuristic, which a crashed run's record would
            # poison; learned in the first supervised session 2026-07-03). Scoped to THIS book's
            # panel.
            w_carried = np.array([
                positions.get(sym, 0.0) * (prices[i] if np.isfinite(prices[i]) else 0.0) / nav
                for i, sym in enumerate(panel.symbols)])
        else:
            prev = j.last("targets")
            if prev is not None:
                # dry-run carried book = the prior week's SCALED targets (the band's carried
                # input; drift is second-order for the plan and exact in the settle, which
                # re-drifts itself).
                w_carried = _aligned_weights(prev, panel) * float(prev["g_t"])
            else:
                w_carried = np.zeros(panel.n_symbols)
        vols = realized_vol(panel, d, window=28)
        plan = build_plan(w_scaled, w_carried, vols,
                          half_spread=PLAN_HALF_SPREAD[spec.name])
        intents = order_intents(plan, panel.symbols, prices, nav, w_target=w_scaled,
                                filters=book_filters(spec))
        turn = plan.turnover_live
        print(f"[plan:{spec.name}] turnover_live={turn:.4f} "
              f"(backtest {plan.turnover_backtest:.4f}); "
              f"{len(intents)} order intents, {int((plan.trade == 0).sum())} names "
              f"band-skipped/flat")

        j.append("targets", d_iso, weights=[round(float(x), 10) for x in w_target],
                 symbols=list(panel.symbols),
                 g_t=round(g_t, 6), nav=round(nav, 2), mode="live" if args.live else "dry",
                 book=spec.name, nav_assets=list(used_assets))
        j.append("orders", d_iso, intents=[i.to_dict() for i in intents],
                 mode="live" if args.live else "dry")

        if args.live:
            _boot_leverage_dial(spec, j, d_iso, intents)
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
    # venue status gate (live catch 2026-08-01: NFPUSDT went SETTLING -> -4140 every cycle).
    # Applies to EVERY book at the single execution choke point. reduce_only intents pass
    # through: a delisting symbol can still be CLOSED, and stranding a position would be worse.
    from convexity.paper.topup import fetch_execution_venue_tradeable
    tradeable = fetch_execution_venue_tradeable()
    ok = failed = 0
    submitted: list[dict] = []
    import httpx as _httpx

    from convexity.paper.venue import execution_base

    for it in intents:
        if it.symbol not in tradeable and not it.reduce_only:
            j.append("order_skip", d_iso, symbol=it.symbol,
                     reason="symbol not TRADING on the execution venue (delisting/settlement) "
                            "— opens blocked, closes still allowed")
            print(f"[live:{spec.name}] {it.symbol} SKIPPED: not TRADING on venue")
            continue
        live_it = it
        # refresh THIS symbol's quote at submission time — the batch snapshot goes stale
        # over a ~60s sequential loop and stale pegs draw -5022 (learned 2026-07-03).
        try:
            _bt = _httpx.get(f"{execution_base()}/fapi/v1/ticker/bookTicker",
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
            ref = quantize_price(_maker_peg(bid, ask, side), tick, side=side)
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
            resp = _submit_maker(j, spec, d_iso, live_it,
                                 live_f[live_it.symbol][0] if live_it.symbol in live_f else 0.0)
            ok += 1
            submitted.append({"symbol": live_it.symbol, "orderId": resp.get("orderId"),
                              "clientOrderId": resp.get("clientOrderId"),
                              "qty": live_it.qty})
            print(f"[live:{spec.name}] {live_it.symbol} {live_it.qty:+} {live_it.tactic} "
                  f"@ {live_it.price_str} -> {resp['status']}")
        except GatewayError as e:
            failed += 1
            print(f"[live:{spec.name}] {live_it.symbol} FAILED (isolated, book continues): {e}")
            j.append("order_error", d_iso, symbol=live_it.symbol, error=str(e)[:300])
    j.append("submitted", d_iso, n_ok=ok, n_failed=failed)
    # finalize v2 surface (PAPER.md §12.6): our own orderIds -> exact fill accounting
    j.append("submission_ids", d_iso, orders=submitted)
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
        submit_bbo,
        submit_intent,
    )
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import fetch_live_filters

    with account_lock(spec.name):
        if _check_kill(spec, j, live=True):
            return
        _journal_venue(j, orders["asof"])
        _manual_autosync(j, spec, orders["asof"])
        syms = {it["symbol"] for it in orders["intents"]}
        for s in sorted(syms):
            cancel_open_orders(s)
        snap = account_snapshot()
        crossed = 0
        live_f = fetch_live_filters()
        failed = 0
        skipped_blind = 0
        crossings: list[dict] = []
        # v2 remainder accounting (audit P1-4, PAPER.md §12.6): exact fills via our own
        # journaled orderIds — an unfilled RESIZE maker is crossed too, not only the
        # opens-from-flat the v1 zero-share heuristic could see. v1 remains the fallback for
        # journals predating submission_ids.
        from convexity.paper.gateway import fetch_order
        from convexity.paper.manual import book_share_qty, load_reserve
        _reserve = load_reserve()
        # ALL our batches for this cycle: rebalance submissions + every reprice pass +
        # prior finalize crossings (rerun idempotency, prod catch 2026-07-11).
        by_sym = _our_order_ids(j, orders["asof"])
        use_ids = bool(by_sym)
        from convexity.paper.topup import fetch_execution_venue_tradeable
        _tradeable = fetch_execution_venue_tradeable()
        for it in orders["intents"]:
            want = float(it["qty"])
            if it["symbol"] not in _tradeable and not bool(it.get("reduce_only", False)):
                j.append("order_skip", orders["asof"], symbol=it["symbol"],
                         reason="not TRADING on venue — finalize open skipped (not a failure)")
                print(f"[finalize:{spec.name}] {it['symbol']} SKIPPED: not TRADING on venue")
                continue
            if use_ids:
                try:
                    remainder = want - _executed_signed(by_sym, it["symbol"])
                except GatewayError as e:
                    # NEVER cross blind (audit P1-4): unreadable fill state could double-trade;
                    # skip loudly — the next rebalance recomputes from venue truth anyway
                    skipped_blind += 1
                    j.append("order_error", orders["asof"], symbol=it["symbol"],
                             error=f"finalize fill-state unreadable, skipped: {str(e)[:200]}")
                    print(f"[finalize:{spec.name}] {it['symbol']} SKIPPED "
                          f"(fill state unreadable): {e}")
                    continue
            else:
                # v1 heuristic on the BOOK's share only (§12.5): an operator's manual position
                # must never make an unfilled maker order look filled
                have = book_share_qty(it["symbol"],
                                      snap.positions.get(it["symbol"], 0.0), _reserve)
                remainder = want if abs(have) < 1e-12 else 0.0
            if remainder != 0.0:
                lf = live_f.get(it["symbol"])
                if lf:
                    _tick, lot, min_qty, _mn = lf
                    remainder = quantize_qty(remainder, lot_size=lot, min_qty=min_qty)
                    if remainder == 0.0:
                        continue
                try:
                    # BOUNDED CROSS FIRST (BBO, verified 2026-08-06): LIMIT priceMatch=
                    # OPPONENT_5 + IOC takes at most 5 levels of the opposing book, so a thin
                    # TradFi perp with an air pocket cannot fill us at an arbitrary price the
                    # way a plain MARKET order can. Any residue then crosses at MARKET, so the
                    # book still completes — we simply take the cheap liquidity first.
                    ro = bool(it.get("reduce_only", False))
                    qstr = fmt_step(abs(remainder), lf[1]) if lf else ""
                    cross_resp = None
                    try:
                        b = submit_bbo(it["symbol"], remainder, qstr, "OPPONENT_5",
                                       tif="IOC", reduce_only=ro)
                        if b.get("orderId") is not None:
                            filled = _executed_signed({it["symbol"]: [
                                {"orderId": b["orderId"], "qty": remainder}]}, it["symbol"])
                            j.append("bounded_cross", orders["asof"], symbol=it["symbol"],
                                     sought=remainder, filled=filled, price=b.get("price"))
                            cross_resp = b
                            residue = remainder - filled
                            if abs(residue) > 1e-12 and lf:
                                rq = quantize_qty(residue, lot_size=lf[1], min_qty=lf[2])
                                if rq != 0.0:
                                    cross_resp = submit_intent(OrderIntent(
                                        symbol=it["symbol"], qty=rq, tactic="MARKET",
                                        ref_price=it["ref_price"], slice_idx=0, n_slices=1,
                                        reduce_only=ro, qty_str=fmt_step(abs(rq), lf[1])))
                    except GatewayError as e:
                        j.append("bbo_fallback", orders["asof"], symbol=it["symbol"],
                                 error=f"bounded cross: {str(e)[:160]}")
                    if cross_resp is None:
                        cross_resp = submit_intent(OrderIntent(
                            symbol=it["symbol"], qty=remainder, tactic="MARKET",
                            ref_price=it["ref_price"], slice_idx=0, n_slices=1,
                            reduce_only=ro, qty_str=qstr))
                    crossings.append({"symbol": it["symbol"],
                                      "orderId": cross_resp.get("orderId"),
                                      "qty": remainder})
                    crossed += 1
                except GatewayError as e:
                    failed += 1
                    j.append("order_error", orders["asof"], symbol=it["symbol"],
                             error=f"finalize: {str(e)[:280]}")
                    print(f"[finalize:{spec.name}] {it['symbol']} FAILED (isolated): {e}")
        if crossings:
            j.append("finalize_ids", orders["asof"], orders=crossings)
        j.append("finalized", orders["asof"], crossed=crossed, failed=failed,
                 skipped_blind=skipped_blind, mode="ids" if use_ids else "v1")
        print(f"[finalize:{spec.name}] crossed {crossed} remainders ({failed} failed, "
              f"{skipped_blind} skipped-blind; mode={'ids' if use_ids else 'v1'})")
        if failed or skipped_blind:
            # fail LOUD so OnFailure alert+retry fire — reruns are idempotent now (finalize_ids)
            raise SystemExit(
                f"[finalize:{spec.name}] {failed} failed / {skipped_blind} blind — rerun is "
                f"safe and crosses only true remainders")


def cmd_catchup(args: argparse.Namespace) -> None:
    """Deploy a CAPITAL EVENT that the weekly rebalance cannot reach (PAPER.md §12.7).

    A deposit raises target notional instantly, but the book closes the gap only through trades
    that must clear BOTH the venue's min-notional and the no-trade band. Measured 2026-08-10
    after the +479: equity coverage FELL 69%->53% because 45 of 68 names needed an increment
    under $5 and the band suppressed most of the rest. The equity trend book is worst affected
    precisely because it is long-horizon and barely trades.

    This is the deposit mirror of the withdrawal INTERIM_RESIZE: move each held position toward
    its CURRENT target, bypassing the no-trade band — the trigger is a known capital event, not
    signal noise, so band suppression (which exists to avoid churning on noise) does not apply.
    Relative weights, and therefore the validated construction, are untouched.

    SAFETY: trades only toward the journaled target (never beyond it, never a new direction);
    caps total added notional at --max-frac of NAV; maker-first via the same BBO path; skips
    non-tradeable symbols; refuses if the book is already at/above --target-cov coverage."""
    spec = _book(args)
    j = _journal(spec)
    t = j.last("targets")
    if t is None or t.get("mode") != "live":
        print(f"[catchup:{spec.name}] no live targets — nothing to deploy")
        return
    from convexity.execution.execution import quantize_qty
    from convexity.paper.gateway import GatewayError, account_snapshot
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import (
        fetch_execution_venue_tradeable,
        fetch_live_filters,
        fetch_ticker_prices,
    )

    if getattr(args, "if_flow", False):
        # FLOW-TRIGGERED ONLY (the automatic path). catchup bypasses the no-trade band, which is
        # part of the validated construction's turnover control — so it must fire on CAPITAL
        # EVENTS, never on a schedule. Equity coverage cannot reach 95% at this account size, so
        # a weekly catchup would trade the band-suppressed deltas every cycle and push turnover
        # above what the backtest validated.
        last_cu = j.last("catchup")
        since = int(last_cu["ts"]) if last_cu else 0
        new_flows = [f for f in j.read("flow") if int(f["ts"]) > since]
        if not new_flows:
            print(f"[catchup:{spec.name}] --if-flow: no capital event since the last catchup "
                  f"— skipping (band preserved)")
            return
        print(f"[catchup:{spec.name}] --if-flow: {len(new_flows)} new flow(s) totalling "
              f"{sum(float(f['amount']) for f in new_flows):+.2f} — deploying")
    g, nav = float(t["g_t"]), float(t["nav"])
    from convexity.paper.fleetconfig import RENORM_PARTIAL_BOOK
    use_renorm = getattr(args, "renorm", False) or RENORM_PARTIAL_BOOK.get(spec.name, False)
    lock_cm = account_lock(spec.name) if args.live else contextlib.nullcontext()
    with lock_cm:
        if args.live and _check_kill(spec, j, live=True):
            return
        if args.live:
            _journal_venue(j, t["asof"])
        # CANCEL FIRST (fix 2026-08-10): without this, a re-run while orders rest posts a
        # SECOND set for the same gap and both filling overshoots the target. Cancelling makes
        # catchup idempotent, so re-running it also acts as its own chaser at the new touch.
        if args.live:
            from convexity.paper.gateway import cancel_open_orders, fetch_open_orders
            try:
                resting = {o["symbol"] for o in fetch_open_orders()
                           if o["symbol"] in book_domain(spec)
                           and str(o.get("clientOrderId", "")).startswith("cvx-")}
                for s in sorted(resting):
                    try:
                        cancel_open_orders(s)
                    except GatewayError:
                        pass
                if resting:
                    print(f"[catchup:{spec.name}] cancelled resting orders on "
                          f"{len(resting)} symbols before re-planning")
            except GatewayError as e:
                print(f"[catchup:{spec.name}] could not list open orders ({e}) — ABORT "
                      f"rather than risk double-posting")
                return
        px = fetch_ticker_prices()
        lf = fetch_live_filters()
        tradeable = fetch_execution_venue_tradeable()
        snap = account_snapshot()
        dom = book_domain(spec)
        held_map = {s: q for s, q in snap.positions.items() if s in dom}
        weights = [float(x) * g for x in t["weights"]]
        if use_renorm:
            import numpy as _np
            wv, rdiag = renormalize_achievable(
                _np.asarray(weights, dtype=float), tuple(t["symbols"]),
                _np.asarray([float(px.get(s, 0.0)) for s in t["symbols"]]), nav,
                book_filters(spec))
            weights = list(wv)
            print(f"[renorm:{spec.name}] {rdiag}")
        plan = []
        tgt_tot = held_tot = 0.0
        for w, sym in zip([x / g if g else 0.0 for x in weights], t["symbols"]):
            w = float(w)
            p = float(px.get(sym, 0.0))
            if p <= 0:
                continue
            tgt_n = w * g * nav
            held_n = held_map.get(sym, 0.0) * p
            tgt_tot += abs(tgt_n)
            held_tot += abs(held_n)
            if abs(w) < 1e-9 or sym not in tradeable:
                continue
            gap_n = tgt_n - held_n
            # only CLOSE the gap toward target, and only in the target's own direction
            if gap_n * tgt_n <= 0:
                continue
            f = lf.get(sym)
            if f is None:
                continue
            tick, lot, min_qty, min_not = f
            qty = quantize_qty(gap_n / p, lot_size=lot, min_qty=min_qty)
            if qty == 0.0 or abs(qty) * p < max(min_not, 5.0):
                continue
            plan.append((sym, qty, p, lot, tick))
        cov = held_tot / tgt_tot * 100 if tgt_tot > 0 else 0.0
        add_n = sum(abs(q) * p for _s, q, p, _l, _t in plan)
        cap = args.max_frac * nav
        print(f"[catchup:{spec.name}] coverage {cov:.1f}% (held {held_tot:.0f} / target "
              f"{tgt_tot:.0f}) | {len(plan)} names deployable for {add_n:.2f} USDT "
              f"(cap {cap:.2f})")
        if cov >= args.target_cov:
            print(f"[catchup:{spec.name}] already at/above target coverage "
                  f"{args.target_cov}% — nothing to do")
            return
        if add_n > cap:
            plan.sort(key=lambda r: -abs(r[1] * r[2]))
            keep, run = [], 0.0
            for r in plan:
                n = abs(r[1] * r[2])
                if run + n > cap:
                    continue
                keep.append(r)
                run += n
            print(f"[catchup:{spec.name}] capped at {args.max_frac:.0%} of NAV: "
                  f"{len(keep)}/{len(plan)} names, {run:.2f} USDT")
            plan = keep
        if not args.live:
            for sym, qty, p, _l, _t in plan[:10]:
                print(f"[dry:{spec.name}] {sym:>14} {qty:+g} @ ~{p} = {abs(qty * p):.2f}")
            if len(plan) > 10:
                print(f"[dry:{spec.name}] ... +{len(plan) - 10} more")
            return
        ok = failed = 0
        posted = []
        for sym, qty, _p, lot, tick in plan:
            it = OrderIntent(symbol=sym, qty=qty, tactic="POST_ONLY_MAKER", ref_price=0.0,
                             slice_idx=0, n_slices=1, reduce_only=False,
                             qty_str=fmt_step(abs(qty), lot), price_str="")
            try:
                resp = _submit_maker(j, spec, t["asof"], it, tick)
                posted.append({"symbol": sym, "orderId": resp.get("orderId"),
                               "clientOrderId": resp.get("clientOrderId"), "qty": qty})
                ok += 1
            except GatewayError as e:
                failed += 1
                j.append("order_error", t["asof"], symbol=sym,
                         error=f"catchup: {str(e)[:200]}")
        if posted:
            j.append("submission_ids", t["asof"], orders=posted)
            # Fold catchup's orders INTO the cycle's intent list so the scheduled reprice and
            # finalize passes chase/cross them (they key off j.last("orders")). Merged PER
            # SYMBOL — duplicate entries would break finalize's want-vs-executed arithmetic,
            # since executions are summed per symbol. Fix 2026-08-10: catchup orders were
            # placed for band-suppressed names absent from the rebalance's intents, so nothing
            # in the pipeline was chasing them and they rested indefinitely.
            prev = j.last("orders")
            merged: dict[str, dict] = {}
            if prev is not None and prev.get("asof") == t["asof"]:
                for it0 in prev["intents"]:
                    merged[it0["symbol"]] = dict(it0)
            for sym, qty, p, lot, tick in plan:
                if sym in merged:
                    merged[sym]["qty"] = float(merged[sym]["qty"]) + qty
                    merged[sym]["qty_str"] = fmt_step(abs(float(merged[sym]["qty"])), lot)
                else:
                    merged[sym] = {"symbol": sym, "qty": qty, "tactic": "POST_ONLY_MAKER",
                                   "ref_price": p, "slice_idx": 0, "n_slices": 1,
                                   "reduce_only": False,
                                   "qty_str": fmt_step(abs(qty), lot), "price_str": ""}
            j.append("orders", t["asof"], intents=list(merged.values()), mode="live",
                     source="catchup_merged")
        j.append("catchup", t["asof"], coverage_before=round(cov, 1), n_posted=ok,
                 n_failed=failed, notional=round(add_n, 2), max_frac=args.max_frac)
        print(f"[catchup:{spec.name}] posted {ok} maker orders ({failed} failed); "
              f"reprice/finalize will complete them")


def cmd_reprice(args: argparse.Namespace) -> None:
    """Mid-window maker RE-PEG (PAPER.md §12.7). Cancels this cycle's resting post-only orders
    and re-posts the UNFILLED remainder at the CURRENT touch, so a stale peg cannot sit through
    the whole window while the market walks away. Live catch 2026-08-06: with a single static
    peg the fill-weighted maker share was 7% (equity) / 0% (commodity) — the books paid the
    full spread at finalize. Idempotent: the remainder is computed from OUR orderIds across all
    journaled batches, so repeated passes converge and never double-trade."""
    spec = _book(args)
    j = _journal(spec)
    orders = j.last("orders")
    if orders is None or orders.get("mode") != "live":
        print(f"[reprice:{spec.name}] no live orders record — nothing to reprice")
        return
    if not args.live:
        print(f"[reprice:{spec.name}] dry-run: would re-peg "
              f"{len(orders['intents'])} intents for {orders['asof']}")
        return
    from convexity.execution.execution import quantize_price, quantize_qty
    from convexity.paper.gateway import GatewayError, cancel_open_orders
    from convexity.paper.policy import fmt_step
    from convexity.paper.topup import (
        fetch_book_tickers,
        fetch_execution_venue_tradeable,
        fetch_live_filters,
    )

    with account_lock(spec.name):
        if _check_kill(spec, j, live=True):
            return
        _journal_venue(j, orders["asof"])
        by_sym = _our_order_ids(j, orders["asof"])
        if not by_sym:
            print(f"[reprice:{spec.name}] no journaled orderIds — skipping (v1 journal)")
            return
        book = fetch_book_tickers()
        live_f = fetch_live_filters()
        tradeable = fetch_execution_venue_tradeable()
        for s in sorted({it["symbol"] for it in orders["intents"]}):
            try:
                cancel_open_orders(s)
            except GatewayError:
                pass
        posted: list[dict] = []
        n_done = n_skip = n_fail = 0
        for it in orders["intents"]:
            sym = it["symbol"]
            want = float(it["qty"])
            ro = bool(it.get("reduce_only", False))
            if sym not in tradeable and not ro:
                n_skip += 1
                continue
            try:
                remainder = want - _executed_signed(by_sym, sym)
            except GatewayError as e:
                n_fail += 1
                j.append("order_error", orders["asof"], symbol=sym,
                         error=f"reprice fill-state unreadable: {str(e)[:180]}")
                continue
            lf, ba = live_f.get(sym), book.get(sym)
            if lf is None or ba is None:
                n_skip += 1
                continue
            tick, lot, min_qty, min_notional = lf
            bid, ask = ba
            if bid <= 0.0 or ask <= 0.0 or ask <= bid:
                n_skip += 1
                continue
            side = 1 if remainder > 0 else -1
            qty = quantize_qty(remainder, lot_size=lot, min_qty=min_qty)
            ref = quantize_price(_maker_peg(bid, ask, side), tick, side=side)
            if qty == 0.0 or (abs(qty) * ref < min_notional and not ro):
                n_done += 1  # filled, or the residue is below the venue floor: nothing to do
                continue
            live_it = OrderIntent(symbol=sym, qty=qty, tactic="POST_ONLY_MAKER", ref_price=ref,
                                  slice_idx=0, n_slices=1, reduce_only=ro,
                                  qty_str=fmt_step(abs(qty), lot),
                                  price_str=fmt_step(ref, tick))
            try:
                resp = _submit_maker(j, spec, orders["asof"], live_it, tick)
                posted.append({"symbol": sym, "orderId": resp.get("orderId"),
                               "clientOrderId": resp.get("clientOrderId"), "qty": qty})
            except GatewayError as e:
                n_fail += 1
                j.append("order_error", orders["asof"], symbol=sym,
                         error=f"reprice: {str(e)[:200]}")
        if posted:
            j.append("submission_ids", orders["asof"], orders=posted)
        j.append("repriced", orders["asof"], n_posted=len(posted), n_complete=n_done,
                 n_skipped=n_skip, n_failed=n_fail)
        print(f"[reprice:{spec.name}] re-pegged {len(posted)} at touch "
              f"({n_done} already complete, {n_skip} skipped, {n_fail} failed)")


def cmd_tca(args: argparse.Namespace) -> None:
    """Post-cycle execution audit (PAPER.md §12.7). Journals, for the last live cycle:

      shortfall_bp — notional-weighted implementation shortfall of REAL fills vs the MODEL's own
                     reference price (the panel close in orders[].ref_price, pre-peg). Positive
                     = we did worse than the model assumed. Measured 2026-08-06 over 12 cycles:
                     +0.0 bp, i.e. two-sided noise, NOT a systematic basis.
      maker_pct    — fill-weighted maker share (the honest one; order counts flatter it).
      coverage_pct — held notional / target notional. THIS is what actually drives model-vs-venue
                     drift: prod holds only the min-notional-clearing subset of each book.

    Read-only against the venue; one journal record. Run after the finalize window."""
    spec = _book(args)
    j = _journal(spec)
    orders = j.last("orders")
    if orders is None or orders.get("mode") != "live":
        print(f"[tca:{spec.name}] no live cycle to audit")
        return
    if not args.live:
        print(f"[tca:{spec.name}] dry-run: would audit {orders['asof']}")
        return
    from convexity.paper.gateway import GatewayError, account_snapshot, fetch_user_trades
    from convexity.paper.topup import fetch_ticker_prices

    t0 = int(orders["ts"]) - 60_000
    t1 = t0 + 8 * 3600_000
    refs = {it["symbol"]: (float(it["ref_price"]), float(it["qty"]))
            for it in orders["intents"] if float(it["ref_price"]) > 0}
    notl = cost = fills = maker = 0.0
    worst: list[list] = []
    for sym, (rp, _q) in refs.items():
        try:
            tr = fetch_user_trades(sym, t0, t1)
        except GatewayError:
            continue
        s_notl = s_cost = 0.0
        for t in tr:
            px, q = float(t["price"]), abs(float(t["qty"]))
            sgn = 1.0 if t["side"] == "BUY" else -1.0
            s_cost += sgn * (px - rp) * q
            s_notl += px * q
            fills += 1
            maker += 1 if t.get("maker") else 0
        notl += s_notl
        cost += s_cost
        if s_notl > 0:
            worst.append([sym, round(s_cost / s_notl * 1e4, 1)])
    # coverage: held vs target notional at current prices
    try:
        px_now = fetch_ticker_prices()
        snap = account_snapshot()
        dom = book_domain(spec)
        # TARGET notional is the whole book (weights x g_t x nav), NOT just this week's
        # trades — comparing against traded notional gave a nonsense >100% reading.
        trec = j.last("targets")
        tgt = 0.0
        if trec is not None:
            g, nav = float(trec.get("g_t", 1.0)), float(trec.get("nav", 0.0))
            tgt = sum(abs(float(w)) for w in trec["weights"]) * g * nav
        held = sum(abs(q) * float(px_now.get(s, 0.0))
                   for s, q in snap.positions.items() if s in dom)
        cov = held / tgt * 100 if tgt > 0 else 0.0
    except GatewayError:
        cov = float("nan")
    worst.sort(key=lambda x: -abs(x[1]))
    rec = {"shortfall_bp": round(cost / notl * 1e4, 2) if notl > 0 else None,
           "shortfall_usdt": round(cost, 4), "traded_notional": round(notl, 2),
           "n_fills": int(fills), "maker_pct": round(maker / fills * 100, 1) if fills else None,
           "coverage_pct": round(cov, 1), "worst_symbols": worst[:5]}
    j.append("tca", orders["asof"], **rec)
    print(f"[tca:{spec.name}] {orders['asof']}: shortfall {rec['shortfall_bp']} bp on "
          f"{rec['traded_notional']} traded | maker {rec['maker_pct']}% | "
          f"coverage {rec['coverage_pct']}% | worst {rec['worst_symbols'][:3]}")


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
    with account_lock(spec.name):
        if _check_kill(spec, j, live=True):
            return
        _journal_venue(j, "recon")
        # Recon is READ-ONLY observability: a transient venue outage is not actionable at
        # 00:25 UTC and must not page (live catch 2026-07-27: demo-fapi 502 -> 3 alerts for
        # one Binance-side blip). Reads already retry (gateway._run_read); if they still fail
        # we journal and exit 0 — UNLESS the previous recon failed the same way, which means
        # persistent unreachability and DOES page. Trading sessions stay fail-closed.
        from convexity.paper.gateway import GatewayError as _GwErr
        try:
            _manual_autosync(j, spec, "recon")
            positions, upnl, acct_eq, assets, unowned, manual_notes = _venue_book_state(domain)
        except _GwErr as e:
            import time as _t
            prev = j.last("venue_unreachable")
            j.append("venue_unreachable", "recon", error=str(e)[:200])
            recent = prev is not None and (_t.time() * 1000 - int(prev["ts"])) < 36 * 3600 * 1000
            msg = f"[recon:{spec.name}] venue unreachable: {str(e)[:160]}"
            if recent:
                raise SystemExit(msg + " — SECOND consecutive failure, escalating (alert fires)")
            print(msg + " — transient, skipping this recon (no page; next run retries)")
            return
        if manual_notes:
            j.append("manual_reserve", "recon", notes=manual_notes)
            for note in manual_notes:
                print(f"[recon:{spec.name}] manual carve-out: {note}")
        if unowned:
            # hygiene backstop: a position OUTSIDE EVERY book's domain belongs to nobody — the
            # domain-scoped sweeps will never close it, so it must at least be visible
            j.append("unowned_positions", "recon", symbols=sorted(unowned))
            print(f"[recon:{spec.name}] WARNING: venue positions outside every book's domain "
                  f"(no sweep will ever close these): {sorted(unowned)}")
        if alloc_mode:
            eq = _book_equity(j, spec)
            # allocation mode: the daily mark is the BOOK's model equity; account equity + the
            # book's venue unrealized are journaled as recon context (drift measured, PAPER.md
            # §11)
            j.append("book_recon", "recon", book_equity=round(eq, 2),
                     account_equity=round(acct_eq, 2), unrealized_book=round(upnl, 4),
                     n_positions=len(positions))
            t_live = j.last("targets")
            if t_live is not None and t_live.get("mode") == "live":
                from convexity.paper.topup import fetch_ticker_prices
                try:
                    tot, top = _position_drift(t_live, positions, fetch_ticker_prices())
                    j.append("position_drift", "recon", total_abs_usdt=tot, top=top[:10],
                             n_offtarget=len(top))
                    print(f"[recon:{spec.name}] drift vs {t_live['asof']} targets: "
                          f"{tot:.2f} USDT abs across {len(top)} names")
                except Exception as e:  # observability only — never blocks the recon
                    print(f"[recon:{spec.name}] drift metric skipped: {e}")
        else:
            eq = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
            j.append("equity_mark", "recon", equity=eq)
        # daily margin monitor (PAPER.md §12.2): the recon is the fleet's fastest guardrail cadence.
        _margin_guardrails(spec, j, "recon", nav=eq, g_t=0.0, live=True)
        t = j.last("targets")
        n_tgt = int(np.count_nonzero(np.asarray(t["weights"]))) if t else 0
        j.append("recon", t["asof"] if t else "none", n_positions=len(positions),
                 n_targets=n_tgt, equity=round(eq, 2), account_equity=round(acct_eq, 2),
                 excluded_btc=float(assets.get("BTC", 0.0)))
        print(f"[recon:{spec.name}] book_equity={eq:.2f} account={acct_eq:.2f} "
              f"(BTC excluded: {assets.get('BTC', 0.0)}) positions={len(positions)} "
              f"targets={n_tgt}")


def _flow_one(spec: BookSpec, amount: float, equity_before: float | None = None) -> None:
    j = _journal(spec)
    mark = j.last("equity_mark")
    eq_before = equity_before if equity_before is not None else (
        float(mark["equity"]) if mark else spec.default_nav)
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    ev = FlowEvent(ts=int(now64.astype("datetime64[ms]").astype(np.int64)),
                   amount=amount, equity_before=eq_before)
    directive = flow_risk_check(ev, OpsDefaults())
    j.append("flow", asof, amount=ev.amount, equity_before=ev.equity_before, ts_ms=ev.ts)
    j.append("directive", asof, action=directive.action, reason=directive.reason,
             resize_factor=directive.resize_factor)
    print(f"[flow:{spec.name}] {ev.amount:+.2f} on equity {eq_before:.2f} -> {directive.action}: "
          f"{directive.reason}")


def cmd_flow(args: argparse.Namespace) -> None:
    spec = _book(args)
    # serialized: a flow record landing mid-rebalance must not change NAV under a sizing session
    # (PAPER.md §12.2 — the rebalance reads NAV once inside its own lock hold).
    with account_lock(f"flow:{spec.name}"):
        _flow_one(spec, args.amount, args.equity_before)


def _allocate_one(spec: BookSpec, amount: float) -> None:
    j = _journal(spec)
    if not (amount > 0):
        raise SystemExit(f"[allocate:{spec.name}] amount must be positive, got {amount}")
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    j.append("allocate", asof, amount=float(amount), book=spec.name)
    j.append("equity_mark", asof, equity=float(amount))
    other = [b for b in BOOKS.values() if b.name != spec.name]
    others = {b.name: Journal(book_journal_path(b)).last("allocate") for b in other}
    total = float(amount) + sum(float(a["amount"]) for a in others.values() if a)
    print(f"[allocate:{spec.name}] {amount:.2f} journaled; all-book allocations sum to "
          f"{total:.2f} — verify this stays within the account's USDT+USDC (runbook)")


def cmd_allocate(args: argparse.Namespace) -> None:
    """Set the book's virtual capital (PAPER.md §11). The operator splits the ONE demo account
    into per-book NAVs; each book's equity then evolves by its own settles. Re-allocating is a
    new record (append-only) — treat a mid-life re-allocation as a flow event for the TWR."""
    spec = _book(args)
    with account_lock(f"allocate:{spec.name}"):
        _allocate_one(spec, args.amount)


def cmd_allocate_split(args: argparse.Namespace) -> None:
    """Split a TOTAL account allocation across all three books per the fleet ratios
    (PAPER.md §12.1: 400/300/300 per 1000), atomically under the account lock —
    requirement 4's 'proportional re-allocation exactly like the other two'."""
    from convexity.paper.fleetconfig import ALLOC_UNITS, ALLOC_TOTAL

    if not (args.total > 0):
        raise SystemExit(f"[allocate-split] total must be positive, got {args.total}")
    with account_lock("allocate-split"):
        for name, units in ALLOC_UNITS.items():
            _allocate_one(BOOKS[name], args.total * units / ALLOC_TOTAL)
    print(f"[allocate-split] {args.total:.2f} split "
          f"{'/'.join(f'{n}:{u}' for n, u in ALLOC_UNITS.items())} per {ALLOC_TOTAL}")


def cmd_flow_split(args: argparse.Namespace) -> None:
    """Journal an account-level deposit/withdrawal proportionally into all three books, under
    the account lock (requirement 4)."""
    from convexity.paper.fleetconfig import ALLOC_UNITS, ALLOC_TOTAL

    with account_lock("flow-split"):
        for name, units in ALLOC_UNITS.items():
            _flow_one(BOOKS[name], args.amount * units / ALLOC_TOTAL)
    print(f"[flow-split] {args.amount:+.2f} split "
          f"{'/'.join(f'{n}:{u}' for n, u in ALLOC_UNITS.items())} per {ALLOC_TOTAL}")


def cmd_kill(args: argparse.Namespace) -> None:
    """Per-book kill switch (PAPER.md §12.2). ``kill``: creates the control file —
    the NEXT venue-facing session (or this command with --now under the lock, live) flattens the
    book and halts. ``--resume``: removes the file. Independent per book by construction."""
    spec = _book(args)
    j = _journal(spec)
    kp = _kill_path(spec)
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    if args.resume:
        if kp.exists():
            kp.unlink()
            j.append("directive", asof, action="RESUME", reason="operator resumed the book")
            print(f"[kill:{spec.name}] KILL file removed — book resumes at its next session")
        else:
            print(f"[kill:{spec.name}] no KILL file present — nothing to resume")
        return
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps({"ts": str(now64), "reason": args.reason or "operator kill"}))
    j.append("directive", asof, action="KILL", reason=args.reason or "operator kill")
    print(f"[kill:{spec.name}] KILL file created ({kp})")
    if args.now:
        with account_lock(f"kill:{spec.name}"):
            _flatten_book_live(spec, j, reason=args.reason or "operator kill --now")


def cmd_status(args: argparse.Namespace) -> None:
    spec = _book(args)
    j = _journal(spec)
    from convexity.paper.venue import venue
    print(f"[status] book={spec.name} venue={venue()} journal={book_journal_path(spec)}")
    for kind in ("allocate", "targets", "orders", "settle", "equity_mark", "book_recon",
                 "flow", "directive", "recon"):
        rec = j.last(kind)
        if rec is None:
            continue
        brief = {k: v for k, v in rec.items() if k not in ("weights", "intents", "symbols")}
        print(f"[{kind}] {json.dumps(brief)[:160]}")


def cmd_manual(args: argparse.Namespace) -> None:
    """Read-only view of the operator carve-out (PAPER.md §12.5). The ledger is maintained
    AUTOMATICALLY: fills without the gateway's cvx- stamp are adopted at every recon/rebalance/
    finalize/kill sync. To correct a wrong adoption, hand-edit data/paper/manual_reserve.json
    (books read it fresh each session)."""
    import time as _time

    from convexity.paper.manual import load_reserve, load_sync_state, reserve_path

    now_ms = int(_time.time() * 1000)
    if getattr(args, "bootstrap", False):
        from convexity.paper.gateway import account_snapshot
        from convexity.paper.manual import bootstrap_reserve, save_reserve

        with account_lock("manual"):
            live_books = [
                b.name for b in BOOKS.values()
                if any(r.get("mode") == "live" for r in _journal(b).read("targets"))]
            try:
                seeded = bootstrap_reserve(account_snapshot().positions,
                                           load_reserve(), live_books)
            except RuntimeError as e:
                raise SystemExit(f"[manual] {e}")
            load_sync_state(now_ms)  # anchor deploy_ms NOW: history stays out of attribution
            save_reserve(seeded)
        if not seeded:
            print("[manual] bootstrap: no venue positions — ledger empty, attribution anchored")
        for sym, qty in sorted(seeded.items()):
            print(f"[manual] bootstrap: {sym:>14} seeded {qty:+g} (operator-owned)")
        if seeded:
            print(f"[manual] bootstrap: {len(seeded)} positions seeded — the books will never "
                  f"touch them; app fills from now on adjust reserves automatically")
        return
    res = load_reserve()
    if not res:
        print(f"[manual] no operator reserves ({reserve_path()}) — auto-adopted from "
              f"non-cvx fills at every sync")
    for sym, qty in sorted(res.items()):
        print(f"[manual] {sym:>14} reserved {qty:+g}")
    state = load_sync_state(now_ms)
    marks = [int(s.get("watermark", 0)) for s in state.get("symbols", {}).values()]
    if marks:
        age_h = (now_ms - max(marks)) / 3.6e6
        print(f"[manual] attribution watermark: {age_h:.1f}h ago "
              f"({len(marks)} symbols tracked)")
    else:
        print("[manual] no syncs recorded yet (first live session will anchor deploy_ms)")


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
    r.add_argument("--renorm", action="store_true",
                   help="scale achievable names up to the target gross (PAPER.md §12.8)")
    r.set_defaults(fn=cmd_rebalance)
    f = sub.add_parser("finalize")
    add_book(f)
    f.add_argument("--live", action="store_true")
    f.set_defaults(fn=cmd_finalize)
    cu = sub.add_parser("catchup")
    add_book(cu)
    cu.add_argument("--live", action="store_true")
    cu.add_argument("--max-frac", type=float, default=0.5,
                    help="cap added notional at this fraction of NAV (default 0.5)")
    cu.add_argument("--target-cov", type=float, default=95.0,
                    help="skip if coverage already >= this pct (default 95)")
    cu.add_argument("--renorm", action="store_true",
                    help="scale achievable names up to the target gross (PAPER.md §12.8)")
    cu.add_argument("--if-flow", action="store_true",
                    help="only act when a flow was journaled since the last catchup (the "
                         "automatic, band-preserving trigger)")
    cu.set_defaults(fn=cmd_catchup)
    tc = sub.add_parser("tca")
    add_book(tc)
    tc.add_argument("--live", action="store_true")
    tc.set_defaults(fn=cmd_tca)
    rp = sub.add_parser("reprice")
    add_book(rp)
    rp.add_argument("--live", action="store_true")
    rp.set_defaults(fn=cmd_reprice)
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
    als = sub.add_parser("allocate-split")
    als.add_argument("--total", type=float, required=True)
    als.set_defaults(fn=cmd_allocate_split)
    fls = sub.add_parser("flow-split")
    fls.add_argument("--amount", type=float, required=True)
    fls.set_defaults(fn=cmd_flow_split)
    k = sub.add_parser("kill")
    add_book(k)
    k.add_argument("--resume", action="store_true")
    k.add_argument("--now", action="store_true",
                   help="flatten immediately (live) instead of at the next session")
    k.add_argument("--reason", default=None)
    k.set_defaults(fn=cmd_kill)
    mn = sub.add_parser("manual")  # ledger is auto-maintained (§12.5); no-args = read-only
    mn.add_argument("--bootstrap", action="store_true",
                    help="virgin-account cutover: seed the reserve with ALL venue positions "
                         "(refuses if any book has live targets or the ledger is non-empty)")
    mn.set_defaults(fn=cmd_manual)
    s = sub.add_parser("status")
    add_book(s)
    s.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
