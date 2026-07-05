import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity")
sys.path.insert(0, str(REPO / "research/entry_exit/study"))
import swing_pipeline as sp  # noqa: E402

curve = sp.load_curve_raw()
cot = pd.read_parquet(REPO / "data/cot/cot_disagg.parquet")
P = sp.build_panel(curve, cot)
carry, wret, listed = P["carry"], P["wret"], P["listed"]
n, m = wret.shape

# (A) decile top1-bot1 EQUAL weight, GROSS
d = []
for t in range(n):
    s, y = carry[t], wret[t]
    ok = np.isfinite(s) & np.isfinite(y) & listed[t]
    if ok.sum() >= 4:
        idx = np.where(ok)[0]
        o = idx[np.argsort(s[idx])]
        d.append(y[o[-1]] - y[o[0]])          # long most-back, short most-contango
d = np.array(d)
print(f"(A) decile top1-bot1 equal-wt GROSS: mean={d.mean()*1e4:+.2f}bp  Sharpe={d.mean()/d.std()*np.sqrt(52):+.2f}  n={len(d)}")

# (B) pipeline k=2 inverse-vol factor, GROSS (rt=0)
W = sp.factor_weights(P)["F1_carry"]
net, turn, valid = sp.net_series(W, wret, 0.0)
print(f"(B) pipeline F1_carry k=2 invvol GROSS: mean={net.mean()*1e4:+.2f}bp  Sharpe={net.mean()/net.std()*np.sqrt(52):+.2f}  n={len(net)}  turn={turn.mean():.2f}")

# (C) pipeline-style but k=1 equal-weight GROSS (bridge between A and B)
Wc = np.zeros((n, m))
for t in range(n):
    s = carry[t]
    ok = np.isfinite(s) & np.isfinite(P["rvol"][t]) & listed[t]
    idx = np.where(ok)[0]
    if len(idx) < 2:
        continue
    o = idx[np.argsort(s[idx])]
    Wc[t, o[-1]] = 0.5
    Wc[t, o[0]] = -0.5
netC, _, _ = sp.net_series(Wc, wret, 0.0)
print(f"(C) pipeline k=1 equal-wt GROSS:      mean={netC.mean()*1e4:+.2f}bp  Sharpe={netC.mean()/netC.std()*np.sqrt(52):+.2f}  n={len(netC)}")

# reconcile A vs C: same construction, should match closely (A gross-2, C gross-1 => C~A/2)
print(f"\nExpect C ~ A/2 (gross 1 vs 2). A/2={d.mean()*1e4/2:+.2f}bp  C={netC.mean()*1e4:+.2f}bp")

# sample 3 weeks: show carry ranks, picks, next returns
print("\nsample weeks (carry / next-wk-ret per root):")
for t in (200, 400, 600):
    s, y = carry[t], wret[t]
    row = " ".join(f"{sp.ROOTS[j]}:c={s[j]:+.2f},r={y[j]*100:+.1f}%" for j in range(m) if np.isfinite(s[j]))
    print(f"  wk{t}: {row}")
