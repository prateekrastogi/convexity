"""US opening-range-breakout (ORB) directional test — the NIFTY ITM-candidate construction
transplanted to the US lake (research-only scratchpad; never touches src/).

Question (user, 2026-07-03): the unconfirmed NIFTY candidate = intraday ORB directional entry +
tight 1-min stop + ride winners to the close. Does that construction have edge on QQQ / IWM / the
mega-cap lake? Vehicle-agnostic: P&L measured in underlying-return terms (a deep-ITM 0DTE option,
an ES/MES future, or shares are all ~delta-1 expressions of the SAME signal; the vehicle cannot
create edge the signal lacks).

Construction per symbol-day (1-min bars, NY session 09:30-15:59):
  OR = [min(low), max(high)] over the first K minutes, K in {15, 30}
  after minute K: LONG on the first 1-min CLOSE > OR_high; SHORT on the first CLOSE < OR_low
  (first breakout wins; one trade per symbol-day; entry at the breakout bar's close)
  tight stop: any subsequent 1-min CLOSE back through the breakout level -> exit at that close
  ride winners: otherwise exit at the session's last bar close (flat overnight by construction)
  net = gross - RT cost per trade; scenarios {1, 2, 4} bp round-trip (futures-grade to
  equity-grade; a stopped trade pays the same round trip)

Books reported: QQQ-only, IWM-only, and the equal-weight all-stocks book (daily mean of that
day's trade returns; days with no trades = 0). Diagnostics: win rate, avg win/loss, trades/day,
and the fat-tail share (top-10 trades' contribution) — the exact luck-flag the NIFTY candidate
carries. Priors: the convexity sweep measured breakout/trend directional at gross R ~ 0 on these
exact instruments (even inverted); this is the session-anchored member of that family.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

LAKE = pathlib.Path("data/stitched")
OUT = pathlib.Path("research/us_orb_results.json")
STOCKS = ["MSFT", "ORCL", "CRM", "ADBE", "GOOGL", "META", "NFLX", "DIS", "NVDA", "TSM", "AVGO",
          "AAPL", "AMZN", "TSLA", "HD", "WMT", "COST", "JPM", "V", "BRKB", "LLY", "UBER", "PLTR",
          "BABA", "IBM"]
ETFS = ["QQQ", "IWM"]
K_WINDOWS = [15, 30]
RT_BPS = [1.0, 2.0, 4.0]


def session_days(sym: str):
    f = LAKE / f"{sym}USDT.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=["open_time", "high", "low", "close"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(d=ts.dt.date, hm=ts.dt.strftime("%H:%M"))
    df = df[(df["hm"] >= "09:30") & (df["hm"] <= "15:59")].sort_values(["d", "hm"])
    return df.groupby("d")


def orb_trades(sym: str, k: int) -> pd.DataFrame:
    g = session_days(sym)
    if g is None:
        return pd.DataFrame()
    rows = []
    for d, day in g:
        if len(day) < 300 or day["hm"].iloc[0] > "09:35" or day["hm"].iloc[-1] < "15:55":
            continue
        hi = day["high"].to_numpy()
        lo = day["low"].to_numpy()
        cl = day["close"].to_numpy()
        if not (np.isfinite(cl).all() and cl.min() > 0):
            continue
        or_hi, or_lo = float(hi[:k].max()), float(lo[:k].min())
        post = cl[k:]
        up = post > or_hi
        dn = post < or_lo
        first_up = int(np.argmax(up)) if up.any() else 10**9
        first_dn = int(np.argmax(dn)) if dn.any() else 10**9
        if first_up == 10**9 and first_dn == 10**9:
            continue  # no breakout day
        if first_up <= first_dn:
            side, e_idx, level = 1, first_up, or_hi
        else:
            side, e_idx, level = -1, first_dn, or_lo
        entry = float(post[e_idx])
        after = post[e_idx + 1:]
        stopped = (after < level) if side == 1 else (after > level)
        if stopped.any():
            x_idx = int(np.argmax(stopped))
            exit_px = float(after[x_idx])
            kind = "stop"
        else:
            exit_px = float(post[-1])
            kind = "close"
        rows.append({"d": str(d), "side": side, "ret": side * (exit_px / entry - 1.0),
                     "exit": kind})
    return pd.DataFrame(rows)


def book_stats(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n_trades": 0}
    daily = trades.groupby("d")["ret"].mean()
    out = {"label": label, "n_trades": int(len(trades)),
           "trades_per_day": round(len(trades) / trades["d"].nunique(), 2),
           "win_rate_pct": round(float((trades["ret"] > 0).mean()) * 100, 1),
           "stop_rate_pct": round(float((trades["exit"] == "stop").mean()) * 100, 1),
           "avg_win_bp": round(float(trades.loc[trades.ret > 0, "ret"].mean()) * 1e4, 1),
           "avg_loss_bp": round(float(trades.loc[trades.ret <= 0, "ret"].mean()) * 1e4, 1),
           "gross_bp_per_trade": round(float(trades["ret"].mean()) * 1e4, 2),
           "top10_share_of_gross": round(float(
               trades["ret"].nlargest(10).sum() / trades["ret"].sum()), 2)
           if trades["ret"].sum() != 0 else None}
    ann = np.sqrt(252.0)
    g = daily.to_numpy()
    out["gross_daily_sharpe"] = round(float(g.mean() / g.std(ddof=1)) * ann, 3)
    for rt in RT_BPS:
        net_tr = trades["ret"] - rt / 1e4
        nd = net_tr.groupby(trades["d"]).mean().to_numpy()
        out[f"net_rt{int(rt)}bp"] = {
            "bp_per_trade": round(float(net_tr.mean()) * 1e4, 2),
            "sharpe": round(float(nd.mean() / nd.std(ddof=1)) * ann, 3),
            "ann_ret_pct": round(float(nd.mean()) * 252 * 100, 2)}
    yr = daily.groupby(lambda i: i[:4]).apply(
        lambda x: round(float(x.mean() / x.std(ddof=1)) * ann, 2) if x.std(ddof=1) > 0 else 0.0)
    out["gross_sharpe_by_year"] = yr.to_dict()
    return out


def main() -> None:
    results: dict = {}
    for k in K_WINDOWS:
        block: dict = {}
        for etf in ETFS:
            t = orb_trades(etf, k)
            block[etf] = book_stats(t, f"{etf}-OR{k}")
            s = block[etf]
            print(f"[orb] OR{k} {etf}: n={s.get('n_trades', 0)} gross={s.get('gross_bp_per_trade')}bp/tr "
                  f"Sh={s.get('gross_daily_sharpe')} | net@2bp Sh={s.get('net_rt2bp', {}).get('sharpe')}",
                  flush=True)
        all_tr = pd.concat([orb_trades(s, k) for s in STOCKS], ignore_index=True)
        block["stocks_ew"] = book_stats(all_tr, f"stocks-OR{k}")
        s = block["stocks_ew"]
        print(f"[orb] OR{k} stocks: n={s['n_trades']} gross={s['gross_bp_per_trade']}bp/tr "
              f"win={s['win_rate_pct']}% stop={s['stop_rate_pct']}% "
              f"Sh={s['gross_daily_sharpe']} | net@2bp Sh={s['net_rt2bp']['sharpe']} "
              f"({s['net_rt2bp']['bp_per_trade']}bp/tr)", flush=True)
        results[f"OR{k}"] = block
    OUT.write_text(json.dumps(results, indent=1))
    print(f"[orb] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
