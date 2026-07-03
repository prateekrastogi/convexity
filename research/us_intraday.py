"""US-equity INTRADAY cross-sectional study (research-only scratchpad; never touches src/).

Question (user, 2026-07-03): buy intraday top winners / short intraday losers, flat by the NY
close — does any edge exist on our US-equity lake?

Priors stated before running: (a) weekly xsec momentum on this exact roster is already measured
DEAD, and the roster is survivorship-flattered mega-caps (chosen ex-post), so results are biased
UP — a negative here is extra credible, a positive is suspect; (b) Lou-Polk-Skouras: equity
momentum accrues OVERNIGHT — closing at the NY close forfeits exactly that component; (c) the
intraday cross-section in the literature is REVERSAL-tilted; (d) ~250 round trips/yr at a few bp
each is a 5-15%/yr cost hurdle. BOTH directions (momentum AND reversal) are run and BOTH count as
trials; three signal windows; three cost scenarios.

Construction per day d, universe = the lake's 27 stocks (ETFs QQQ/IWM/XLE excluded):
  signal_i = P_i(t_sig)/P_i(09:30 open) - 1,  t_sig in {10:00, 10:30, 12:00} NY
  rank cross-sectionally among names with valid bars (>= 15 required, else skip day)
  MOM book: long top-k, short bottom-k (k = n//5, min 3), equal weight, gross 1.0 (0.5/leg)
  entry at the t_sig bar close, exit at the 15:59 bar close (flat overnight by construction)
  REV book = the sign-flipped gross of the same construction
  net = gross - RT_cost, RT scenarios {2, 4, 10}bp on gross (round-trip per name-day; mega-cap
  half-spread ~0.5-2bp + fees; 2bp RT is AGGRESSIVELY cheap, 10bp conservative)

Bonus (one extra trial, the one DOCUMENTED intraday effect — Gao-Han-Li-Zhou 2018 market intraday
momentum): sign of QQQ's first-30-min return trades QQQ's last 30 min (15:30->close), net 2bp RT.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

LAKE = pathlib.Path("data/stitched")
OUT = pathlib.Path("research/us_intraday_results.json")
STOCKS = ["MSFT", "ORCL", "CRM", "ADBE", "GOOGL", "META", "NFLX", "DIS", "NVDA", "TSM", "AVGO",
          "AAPL", "SONY", "AMZN", "TSLA", "HD", "WMT", "COST", "JPM", "V", "BRKB", "LLY", "NVO",
          "UBER", "PLTR", "BABA", "IBM"]
SIG_TIMES = ["10:00", "10:30", "12:00"]
RT_BPS = [2.0, 4.0, 10.0]
MIN_NAMES = 15


def day_frame(sym: str) -> pd.DataFrame:
    f = LAKE / f"{sym}USDT.parquet"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f, columns=["open_time", "open", "close"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(d=ts.dt.date, hm=ts.dt.strftime("%H:%M"))
    df = df[(df["hm"] >= "09:30") & (df["hm"] <= "15:59")]
    g = df.groupby("d")
    out = pd.DataFrame({"open": g.apply(lambda x: x.loc[x["hm"].idxmin(), "open"],
                                        include_groups=False),
                        "close_eod": g.apply(lambda x: x.loc[x["hm"].idxmax(), "close"],
                                             include_groups=False),
                        "last_hm": g["hm"].max(), "first_hm": g["hm"].min(),
                        "n_bars": g.size()})
    for t in SIG_TIMES + ["15:30"]:
        s = df[df["hm"] == t].set_index("d")["close"]
        out[f"p_{t}"] = s.reindex(out.index)
    # a valid session day: opens by 09:35, closes at/after 15:55, enough bars
    ok = (out["first_hm"] <= "09:35") & (out["last_hm"] >= "15:55") & (out["n_bars"] >= 300)
    return out[ok]


def main() -> None:
    frames = {s: day_frame(s) for s in STOCKS}
    frames = {s: f for s, f in frames.items() if len(f) > 200}
    days = sorted(set.union(*(set(f.index) for f in frames.values())))
    print(f"[us-intraday] {len(frames)} names, {len(days)} candidate days "
          f"[{days[0]}..{days[-1]}]", flush=True)

    results: dict = {"n_names": len(frames), "n_days": len(days),
                     "span": [str(days[0]), str(days[-1])], "variants": {}}
    for t_sig in SIG_TIMES:
        gross_series = []
        used_days = []
        for d in days:
            sig, hold = [], []
            for s, f in frames.items():
                if d not in f.index:
                    continue
                row = f.loc[d]
                p0, ps, pe = row["open"], row[f"p_{t_sig}"], row["close_eod"]
                if not (np.isfinite(p0) and np.isfinite(ps) and np.isfinite(pe)
                        and p0 > 0 and ps > 0):
                    continue
                sig.append(ps / p0 - 1.0)
                hold.append(pe / ps - 1.0)
            n = len(sig)
            if n < MIN_NAMES:
                continue
            sig_a, hold_a = np.asarray(sig), np.asarray(hold)
            k = max(3, n // 5)
            order = np.argsort(sig_a)
            gross = 0.5 * (hold_a[order[-k:]].mean() - hold_a[order[:k]].mean())
            gross_series.append(gross)
            used_days.append(str(d))
        g = np.asarray(gross_series)
        ann = np.sqrt(252.0)
        block = {"n_days_used": int(g.size),
                 "gross_mom": {"ann_ret_pct": round(float(g.mean()) * 252 * 100, 2),
                               "sharpe": round(float(g.mean() / g.std(ddof=1)) * ann, 3)}}
        for rt in RT_BPS:
            cost = rt / 1e4
            for name, series in (("mom", g - cost), ("rev", -g - cost)):
                m = float(series.mean())
                sd = float(series.std(ddof=1))
                eq = np.cumprod(1.0 + series)
                dd = float((1.0 - eq / np.maximum.accumulate(eq)).max()) * 100
                block[f"net_{name}_rt{int(rt)}bp"] = {
                    "ann_ret_pct": round(m * 252 * 100, 2),
                    "sharpe": round(m / sd * ann, 3), "maxdd_pct": round(dd, 1)}
        # per-year gross (direction stability)
        yr = pd.Series(g, index=pd.to_datetime(used_days)).groupby(
            lambda i: i.year).apply(lambda x: float(x.mean() / x.std(ddof=1)) * ann
                                    if x.std(ddof=1) > 0 else 0.0)
        block["gross_mom_sharpe_by_year"] = {int(k): round(v, 2) for k, v in yr.items()}
        results["variants"][f"sig_{t_sig}"] = block
        print(f"[us-intraday] {t_sig}: gross mom Sharpe={block['gross_mom']['sharpe']:+.2f} "
              f"({block['gross_mom']['ann_ret_pct']:+.1f}%/yr) | net@4bp: "
              f"mom {block['net_mom_rt4bp']['sharpe']:+.2f}, "
              f"rev {block['net_rev_rt4bp']['sharpe']:+.2f}", flush=True)

    # bonus: market intraday momentum (GHLZ) on QQQ
    q = day_frame("QQQ")
    ok = q[np.isfinite(q["p_10:00"]) & np.isfinite(q["p_15:30"])]
    r_open = ok["p_10:00"] / ok["open"] - 1.0
    r_last = ok["close_eod"] / ok["p_15:30"] - 1.0
    strat = np.sign(r_open.to_numpy()) * r_last.to_numpy() - 2.0 / 1e4
    m, sd = float(np.mean(strat)), float(np.std(strat, ddof=1))
    results["bonus_qqq_ghlz"] = {"n_days": int(strat.size),
                                 "net_sharpe_rt2bp": round(m / sd * np.sqrt(252), 3),
                                 "ann_ret_pct": round(m * 252 * 100, 2)}
    print(f"[us-intraday] GHLZ QQQ first30->last30: net Sharpe "
          f"{results['bonus_qqq_ghlz']['net_sharpe_rt2bp']:+.2f}", flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print(f"[us-intraday] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
