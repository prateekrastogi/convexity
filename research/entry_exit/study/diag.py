import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/home/speculation/convexity")
sys.path.insert(0, str(REPO / "research/entry_exit/study"))
import swing_pipeline as sp  # noqa: E402

curve = sp.load_curve_raw()
cot = pd.read_parquet(REPO / "data/cot/cot_disagg.parquet")
P = sp.build_panel(curve, cot)
roots = P["roots"]
wret, carry = P["wret"], P["carry"]

print("=== per-name roll-clean price-index total return + ann (sanity) ===")
for j, r in enumerate(roots):
    d = sp.root_daily(curve, r)
    adj = d["ret"].fillna(0).add(1).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    tot = adj.iloc[-1]
    print(f"  {r}({sp.ROOT2SYM[r]:11s}) x{tot:6.2f} over {yrs:.1f}y  ann={tot**(1/yrs)-1:+.1%}  "
          f"last-close={d['c0'].iloc[-1]:.2f}")

print("\n=== April-2020 crude sanity (negative-price contamination?) ===")
dcl = sp.root_daily(curve, "CL")
apr = dcl.loc["2020-04-15":"2020-04-28", ["c0", "c1", "ret"]]
print(apr.to_string())

print("\n=== information coefficient: corr(signal_t, wret_t) across names, mean over weeks ===")
def ic(sig):
    vals = []
    for t in range(len(sig)):
        s, y = sig[t], wret[t]
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() >= 4:
            rho, _ = spearmanr(s[ok], y[ok])
            if np.isfinite(rho):
                vals.append(rho)
    a = np.array(vals)
    return a.mean(), a.mean() / a.std() * np.sqrt(len(a))  # mean IC, IC t-stat

prev1 = np.full_like(wret, np.nan)
prev1[1:] = wret[:-1]
for nm, sig in [("carry", carry), ("mom8", None), ("reversal(-prev1)", -prev1)]:
    if sig is None:
        # 8wk momentum signal
        sig = np.full_like(wret, np.nan)
        lg = np.log1p(np.where(np.isfinite(wret), wret, np.nan))
        for t in range(8, len(wret)):
            sig[t] = np.nansum(lg[t - 8:t], axis=0)
    m, ts = ic(sig)
    print(f"  {nm:18s} mean IC={m:+.4f}  t={ts:+.2f}")

print("\n=== carry decile sanity: mean next-wk ret of most-backwardated vs most-contango name ===")
top_ret, bot_ret = [], []
for t in range(len(carry)):
    s, y = carry[t], wret[t]
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() >= 4:
        idx = np.where(ok)[0]
        order = idx[np.argsort(s[idx])]
        bot_ret.append(y[order[0]])   # most contango
        top_ret.append(y[order[-1]])  # most backwardated
print(f"  most-backwardated next-wk mean = {np.mean(top_ret)*1e4:+.1f}bp")
print(f"  most-contango    next-wk mean = {np.mean(bot_ret)*1e4:+.1f}bp")
print(f"  spread (long-back/short-contango) = {(np.mean(top_ret)-np.mean(bot_ret))*1e4:+.1f}bp/wk")
