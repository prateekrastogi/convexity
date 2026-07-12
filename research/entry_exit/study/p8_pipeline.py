#!/usr/bin/env python
"""P8 — CROSS-ASSET RV SPREADS (P8_PREREG.md, frozen c64d4fe).

Fresh data path: weekly Friday closes straight from the stitched lake (NOT the mined commodity
panel). Constructions frozen in the prereg: XLE-CL = P4 verbatim (beta=1); MSTR-BTC = Kalman
beta (statistical ratio), z band {13,26,52}. Causality identical to P4: lp[t] uses closes
through t; wret[t] is the FORWARD t->t+1 return; a deliberate lookahead canary is reported."""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import swing_pipeline as sp  # noqa: E402  (metrics/CPCV/DSR/net_series/vol_target)
import durable_pipeline as dp  # noqa: E402  (gnorm, pbo_cscv)
import rv_pipeline as rv  # noqa: E402  (roll_z, half_life)

OUT = HERE
WK = 52.0
RT_BP, RT_SCEN = 12.0, (6.0, 12.0, 24.0)
N_EFF = 6
KAL_DELTA, KAL_LAM = 1e-4, 0.98
LAKE = Path("/home/speculation/convexity/data/stitched")
SYMS = ("XLEUSDT", "CLUSDT", "MSTRUSDT", "BTCUSDT")
START = {"XLE-CL": "2018-05-01", "MSTR-BTC": "2020-09-01"}


def weekly_close(sym: str) -> pd.Series:
    d = pd.read_parquet(LAKE / f"{sym}.parquet", columns=["open_time", "close"])
    ts = pd.to_datetime(d["open_time"], unit="ms")
    s = pd.Series(d["close"].to_numpy(), index=ts).resample("1D").last().dropna()
    return s.resample("W-FRI").last().dropna()


CLO = pd.concat({s: weekly_close(s) for s in SYMS}, axis=1)
CLO = CLO.dropna(how="all")
FRIDAYS = CLO.index
N = len(CLO)
M = len(SYMS)
IDX = {s: i for i, s in enumerate(SYMS)}
close_mat = CLO.to_numpy()
# forward weekly returns (wret[t] = close[t+1]/close[t] - 1), NaN where either side missing
WRET = np.full((N, M), np.nan)
WRET[:-1] = close_mat[1:] / close_mat[:-1] - 1.0
LOGP = np.log(close_mat)  # levels through t — causal for signals scoring wret[t]


def kalman_spread(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    x = np.array([1.0, 0.0])
    Pm = np.eye(2) * 1e-2
    Q = np.eye(2) * (KAL_DELTA / (1 - KAL_DELTA))
    R = 1e-3
    s = np.full(n, np.nan)
    for t in range(n):
        if not (np.isfinite(a[t]) and np.isfinite(b[t])):
            continue
        H = np.array([b[t], 1.0])
        Pm = Pm + Q
        e = a[t] - H @ x
        s[t] = e
        S = H @ Pm @ H + R
        K = (Pm @ H) / S
        x = x + K * e
        Pm = Pm - np.outer(K, H) @ Pm
        R = KAL_LAM * R + (1 - KAL_LAM) * (e * e)
    return s


def build_pair(name: str, lp: np.ndarray, hl_gate=True):
    if name == "XLE-CL":
        A, B, zw = "XLEUSDT", "CLUSDT", (26, 52, 104)
        s = lp[:, IDX[A]] - lp[:, IDX[B]]
    else:
        A, B, zw = "MSTRUSDT", "BTCUSDT", (13, 26, 52)
        s = kalman_spread(lp[:, IDX[A]], lp[:, IDX[B]])
    start = pd.Timestamp(START[name])
    live = FRIDAYS >= start
    s = np.where(live, s, np.nan)
    with np.errstate(all="ignore"):
        z = np.nanmean(np.column_stack([rv.roll_z(s, w) for w in zw]), axis=1)
    hl = rv.half_life(s)
    gate = np.isfinite(hl) & (hl >= rv.HL_LO) & (hl <= rv.HL_HI)
    pw = -np.clip(z, -rv.ZCLIP, rv.ZCLIP) / rv.ZCLIP
    if hl_gate:
        pw = np.where(gate, pw, 0.0)
    pw = np.nan_to_num(pw)
    Wp = np.zeros((N, M))
    Wp[:, IDX[A]] = 0.5 * pw
    Wp[:, IDX[B]] = -0.5 * pw
    return Wp


def pair_rp(books: dict) -> np.ndarray:
    out = None
    for Wp in books.values():
        r = np.nansum(Wp * np.nan_to_num(WRET), axis=1)
        tv = pd.Series(r).rolling(26).std().shift(1).to_numpy()
        sc = np.where(np.isfinite(tv) & (tv > 0), 1.0 / tv, 0.0)
        med = np.nanmedian(sc[sc > 0]) if (sc > 0).any() else 1.0
        sc = np.clip(sc / max(med, 1e-12), 0.25, 4.0)
        out = Wp * sc[:, None] if out is None else out + Wp * sc[:, None]
    return out


def summarize(tag, W, full=True):
    net, turn, valid = sp.net_series(dp.gnorm(W), WRET, RT_BP)
    m = sp.metrics(net, turn)
    m["dsr"] = sp.dsr(net, N_EFF)
    if len(net) > 60:
        paths = sp.cpcv_paths(net, valid, embargo=8)
        m["cpcv_mean_path"] = round(float(np.mean(paths)), 3)
        m["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 2)
    if full:
        for rt in RT_SCEN:
            nn, _, _ = sp.net_series(dp.gnorm(W), WRET, rt)
            m[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    return m, net


def main():
    res = {"prereg": "P8_PREREG.md",
           "panel": {"n_weeks": int(N), "span": f"{FRIDAYS[0].date()}..{FRIDAYS[-1].date()}"}}
    fam = {}

    books = {}
    for name in ("XLE-CL", "MSTR-BTC"):
        Wp = build_pair(name, LOGP)
        books[name] = Wp
        m, net = summarize(f"pair_{name}", Wp)
        fam[f"pair_{name}"] = net
        res[f"pair_{name}"] = m
        mu, _ = summarize(f"pair_{name}_ungated", build_pair(name, LOGP, hl_gate=False),
                          full=False)
        res[f"pair_{name}_ungated"] = mu
        print(f"{name}: Sh={m.get('sharpe_ann')} DD={m.get('maxdd_pct')} "
              f"CAGR={m.get('cagr_pct')} dr5={m.get('drop5_sharpe')} cpcv={m.get('cpcv_frac_pos')} "
              f"| ungated Sh={mu.get('sharpe_ann')}", flush=True)

    Wb = pair_rp(books)
    mb, netb = summarize("book", Wb)
    fam["book"] = netb
    res["book"] = mb
    print("BOOK:", {k: mb.get(k) for k in ("sharpe_ann", "cagr_pct", "maxdd_pct",
          "drop5_sharpe", "cpcv_frac_pos", "sharpe_rt24")}, "DSR=", mb["dsr"].get("dsr"),
          flush=True)

    # lookahead canary: UNSHIFTED levels include the traded return -> must fake a huge edge
    lp_look = np.log(close_mat)
    WRET_l = WRET.copy()
    lp_shifted_wrong = np.vstack([lp_look[1:], lp_look[-1:]])  # uses t+1 close at t
    can_books = {n: build_pair(n, lp_shifted_wrong) for n in ("XLE-CL", "MSTR-BTC")}
    mc, _ = summarize("canary", pair_rp(can_books), full=False)
    res["canary_lookahead"] = {k: mc.get(k) for k in ("sharpe_ann", "cagr_pct")}
    print("CANARY (must be hugely positive):", res["canary_lookahead"], flush=True)

    keys = sorted(fam)
    L = min(len(v) for v in fam.values())
    Mx = np.column_stack([fam[k][:L] for k in keys])
    res["pbo"] = {"configs": keys, "pbo_cscv": dp.pbo_cscv(Mx)}

    solo_dd_ok = all((res[f"pair_{n}"].get("maxdd_pct") or 100) <= 60.0
                     for n in ("XLE-CL", "MSTR-BTC"))
    gates = {
        "book_sharpe_ge_0.60": (mb.get("sharpe_ann") or 0) >= 0.60,
        "cpcv_ge_0.75": (mb.get("cpcv_frac_pos") or 0) >= 0.75,
        "drop5_pos": (mb.get("drop5_sharpe") or -1) > 0,
        "rt24_ge_0.40": (mb.get("sharpe_rt24") or 0) >= 0.40,
        "dsr_ge_0.70": (mb["dsr"].get("dsr") or 0) >= 0.70,
        "pbo_le_0.35": (res["pbo"]["pbo_cscv"] or 1.0) <= 0.35,
        "solo_dd_le_60": solo_dd_ok,
        "canary_huge": (mc.get("sharpe_ann") or 0) > 2.0,
    }
    res["gates"] = {k: bool(v) for k, v in gates.items()}
    res["all_gates_pass"] = bool(all(gates.values()))

    (OUT / "results_p8.json").write_text(json.dumps(res, indent=1, default=str))
    print("GATES:", res["gates"], flush=True)
    print("PBO:", res["pbo"]["pbo_cscv"], flush=True)
    print("P8 DONE ->", OUT / "results_p8.json", flush=True)


if __name__ == "__main__":
    main()
