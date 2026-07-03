"""Paper-rebalancer CLI (PAPER.md §6): settle -> target -> size -> plan -> intents -> journal.

Commands (systemd timers drive them; every run is a full pure-recompute session):
  rebalance [--live] [--asof YYYY-MM-DD] [--nav X]   weekly session (settles the prior week first)
  finalize  [--live]                                  window-end crossing of unfilled remainders
  recon     [--live]                                  daily position/equity check + journal marks
  flow --amount X [--equity-before E]                 journal an external deposit/withdrawal
  status                                              journal tail summary

Dry-run (default, no keys needed): carried book and equity come from the journal (equity evolves
by the frozen-accounting settled returns from a --nav starting point); orders are journaled, not
sent. --live swaps in the gateway for equity/positions/submission — EXPERIMENTAL until the first
supervised session (PAPER.md §8).
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
from convexity.paper.capital import sizing_nav
from convexity.paper.journal import Journal
from convexity.paper.policy import build_plan, ensemble_target_weights, order_intents, size_gross
from convexity.paper.topup import extend_panel, fetch_tails
from convexity.portfolio.engine import XSecConfig, band_configs, run, run_ensemble
from convexity.portfolio.portfolio import realized_vol
from convexity.system import metrics

DEFAULT_NAV = 5_000.0  # demo USDT balance (USER MANDATE: USDT first, USDC fallback, BTC never)
STEP = 7


def _load_panel(topup: bool) -> Panel:
    syms = tuple(all_symbols(sleeves=("crypto",)))
    panel = build_panel(syms)
    if topup:
        last_ms = int(panel.dates[-1].value // 1_000_000)
        tails = fetch_tails(syms, last_ms)  # only rows AFTER the lake tail are appended
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


def _settle_prior_week(j: Journal, panel: Panel, d_now: int, live: bool) -> None:
    """Settle the prior journaled targets through the frozen accounting; fold TWR + kill-switch."""
    prev = j.last("targets")
    if prev is None:
        return
    d_prev = int(np.searchsorted(panel.dates.to_numpy(), np.datetime64(prev["asof"])))
    if d_prev + STEP > panel.n_days - 1 or d_prev >= d_now:
        return  # week not complete yet
    w_prev = np.asarray(prev["weights"], dtype=float)
    res, _ = settle_week(panel.close, panel.funding, panel.listed, d0=d_prev, d1=d_prev + STEP,
                         w_target=w_prev, w_prev=w_prev, friction=round_trip_friction())
    last_mark = j.last("equity_mark")
    eq_start = float(last_mark["equity"]) if last_mark else DEFAULT_NAV
    flows = tuple(
        FlowEvent(ts=int(r["ts"]), amount=float(r["amount"]),
                  equity_before=float(r["equity_before"]))
        for r in j.read("flow") if prev["asof"] < r["asof"] <= str(panel.dates[d_now].date()))
    if live:
        from convexity.paper.gateway import account_assets
        assets = account_assets()
        eq_end = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
    else:
        eq_end = eq_start * (1.0 + res.net) + sum(f.amount for f in flows)
    r_twr = twr_weekly_return(eq_start, eq_end, flows)
    week_end = str(panel.dates[min(d_prev + STEP, panel.n_days - 1)].date())
    j.append("settle", week_end, r_twr=r_twr, r_model=res.net, equity_start=eq_start,
             equity_end=eq_end, n_flows=len(flows), mode="live" if live else "dry")
    j.append("equity_mark", week_end, equity=eq_end)
    print(f"[settle] week->{week_end}: r_twr={r_twr:+.4%} (model {res.net:+.4%}) "
          f"equity {eq_start:.2f}->{eq_end:.2f} flows={len(flows)}")


def cmd_rebalance(args: argparse.Namespace) -> None:
    j = Journal()
    panel = _load_panel(topup=not args.no_topup)
    d = _last_friday_row(panel, args.asof)
    d_iso = str(panel.dates[d].date())
    print(f"[rebalance] panel [{panel.dates[0].date()}..{panel.dates[-1].date()}] "
          f"rebalance row {d} = {d_iso} (Friday close)")

    _settle_prior_week(j, panel, d, args.live)

    w_target = ensemble_target_weights(panel, d)

    hist = run_ensemble(panel, band_configs())
    trials = np.array([metrics.compute(
        run(panel, XSecConfig(signal="momentum", lookback=lb, quantile="quintile",
                              weighting="volscaled")).net).sharpe_ann for lb in (12, 14, 16)])
    dvol = build_dollar_volume(panel)
    dvol_row = dvol[min(d, dvol.shape[0] - 1)]
    dec = size_gross(hist.net, trials, dvol_row=dvol_row, w_target=w_target,
                     symbols=panel.symbols)
    g_t = float(dec.g_t)

    if args.live:
        from convexity.paper.gateway import account_assets, account_snapshot
        assets = account_assets()
        snap = account_snapshot()
        positions = snap.positions
    else:
        mark = j.last("equity_mark")
        assets = {"USDT": float(mark["equity"]) if mark else args.nav}
        positions = {}
    est_margin = g_t * sum(assets.get(a, 0.0) for a in ("USDT",)) / 10.0  # ~10x maint headroom
    nav_dec = sizing_nav(assets, est_initial_margin=est_margin)
    nav = nav_dec.nav
    print(f"[size] G_t={g_t:.4f} binding={dec.binding_ceiling} | NAV={nav:.2f} "
          f"({'+'.join(nav_dec.used_assets)}; excluded={list(nav_dec.excluded)}) | {nav_dec.note}")

    w_scaled = w_target * g_t
    prices = panel.close[d]
    if positions:
        w_carried = np.array([
            positions.get(sym, 0.0) * (prices[i] if np.isfinite(prices[i]) else 0.0) / nav
            for i, sym in enumerate(panel.symbols)])
    else:
        prev = j.last("targets")
        if prev is not None:
            # dry-run carried book = the prior week's SCALED targets (the band's carried input;
            # drift is second-order for the plan and exact in the settle, which re-drifts itself).
            w_carried = np.asarray(prev["weights"], dtype=float) * float(prev["g_t"])
        else:
            w_carried = np.zeros(panel.n_symbols)
    vols = realized_vol(panel, d, window=28)
    plan = build_plan(w_scaled, w_carried, vols)
    intents = order_intents(plan, panel.symbols, prices, nav)
    turn = plan.turnover_live
    print(f"[plan] turnover_live={turn:.4f} (backtest {plan.turnover_backtest:.4f}); "
          f"{len(intents)} order intents, {int((plan.trade == 0).sum())} names band-skipped/flat")

    j.append("targets", d_iso, weights=[round(float(x), 10) for x in w_target],
             g_t=round(g_t, 6), nav=round(nav, 2), mode="live" if args.live else "dry",
             nav_assets=list(nav_dec.used_assets))
    j.append("orders", d_iso, intents=[i.to_dict() for i in intents],
             mode="live" if args.live else "dry")

    if args.live:
        from convexity.paper.gateway import submit_intent
        for it in intents:
            resp = submit_intent(it)
            print(f"[live] {it.symbol} {it.qty:+} {it.tactic} -> {resp}")
        j.append("submitted", d_iso, n=len(intents))
    else:
        for it in intents[:12]:
            sgn = "+" if it.qty > 0 else "-"
            print(f"[dry] {it.symbol:>14} qty={sgn}{it.qty_str} {it.tactic} @ {it.price_str}")
        if len(intents) > 12:
            print(f"[dry] ... +{len(intents) - 12} more (journaled)")


def cmd_finalize(args: argparse.Namespace) -> None:
    j = Journal()
    orders = j.last("orders")
    if orders is None:
        print("[finalize] no orders journaled")
        return
    if not args.live:
        print(f"[finalize] dry-run: would cancel+cross remainders for "
              f"{len(orders['intents'])} intents")
        return
    from convexity.paper.gateway import account_snapshot, cancel_open_orders, submit_intent
    from convexity.paper.policy import OrderIntent

    syms = {it["symbol"] for it in orders["intents"]}
    for s in sorted(syms):
        cancel_open_orders(s)
    snap = account_snapshot()
    crossed = 0
    for it in orders["intents"]:
        want = float(it["qty"])
        have = snap.positions.get(it["symbol"], 0.0)  # net position vs intended delta: v1 heuristic
        remainder = want if abs(have) < 1e-12 else 0.0
        if remainder != 0.0:
            submit_intent(OrderIntent(symbol=it["symbol"], qty=remainder, tactic="MARKET",
                                      ref_price=it["ref_price"], slice_idx=0, n_slices=1,
                                      reduce_only=False))
            crossed += 1
    j.append("finalized", orders["asof"], crossed=crossed)
    print(f"[finalize] crossed {crossed} remainders")


def cmd_recon(args: argparse.Namespace) -> None:
    j = Journal()
    if not args.live:
        t = j.last("targets")
        print(f"[recon] dry-run: last targets {t['asof'] if t else None}; journal ok")
        return
    from convexity.paper.gateway import account_assets, account_snapshot
    assets = account_assets()
    snap = account_snapshot()
    eq = float(assets.get("USDT", 0.0)) + float(assets.get("USDC", 0.0))
    j.append("equity_mark", "recon", equity=eq)
    t = j.last("targets")
    n_pos = len(snap.positions)
    n_tgt = int(np.count_nonzero(np.asarray(t["weights"]))) if t else 0
    j.append("recon", t["asof"] if t else "none", n_positions=n_pos, n_targets=n_tgt,
             equity=eq, excluded_btc=float(assets.get("BTC", 0.0)))
    print(f"[recon] equity={eq:.2f} (BTC excluded: {assets.get('BTC', 0.0)}) "
          f"positions={n_pos} targets={n_tgt}")


def cmd_flow(args: argparse.Namespace) -> None:
    j = Journal()
    mark = j.last("equity_mark")
    eq_before = args.equity_before if args.equity_before is not None else (
        float(mark["equity"]) if mark else DEFAULT_NAV)
    now64 = np.datetime64("now")
    asof = str(now64.astype("datetime64[D]"))
    ev = FlowEvent(ts=int(now64.astype("datetime64[ms]").astype(np.int64)),
                   amount=args.amount, equity_before=eq_before)
    directive = flow_risk_check(ev, OpsDefaults())
    j.append("flow", asof, amount=ev.amount, equity_before=ev.equity_before, ts_ms=ev.ts)
    j.append("directive", asof, action=directive.action, reason=directive.reason,
             resize_factor=directive.resize_factor)
    print(f"[flow] {ev.amount:+.2f} on equity {eq_before:.2f} -> {directive.action}: "
          f"{directive.reason}")


def cmd_status(_args: argparse.Namespace) -> None:
    j = Journal()
    for kind in ("targets", "orders", "settle", "equity_mark", "flow", "directive", "recon"):
        rec = j.last(kind)
        if rec is None:
            continue
        brief = {k: v for k, v in rec.items() if k not in ("weights", "intents")}
        print(f"[{kind}] {json.dumps(brief)[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="convexity.paper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebalance")
    r.add_argument("--live", action="store_true")
    r.add_argument("--asof", default=None)
    r.add_argument("--nav", type=float, default=DEFAULT_NAV)
    r.add_argument("--no-topup", action="store_true")
    r.set_defaults(fn=cmd_rebalance)
    f = sub.add_parser("finalize")
    f.add_argument("--live", action="store_true")
    f.set_defaults(fn=cmd_finalize)
    c = sub.add_parser("recon")
    c.add_argument("--live", action="store_true")
    c.set_defaults(fn=cmd_recon)
    fl = sub.add_parser("flow")
    fl.add_argument("--amount", type=float, required=True)
    fl.add_argument("--equity-before", type=float, default=None)
    fl.set_defaults(fn=cmd_flow)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
