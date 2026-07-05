#!/usr/bin/env python
"""RV PAIR STUDY — mean reversion on coherent commodity pairs (RV_PREREG.md, frozen 2026-07-05).

Reuses swing_pipeline (data layer, metrics/CPCV/DSR) and durable_pipeline (gnorm, net_of,
summarize, per_year, pbo_cscv) VERBATIM. Causality: wret[t] is the FORWARD t->t+1 return
(net_series scores W[t] on wret[t]), so log price levels are built SHIFTED — lp[t] = sum_{k<t}
log1p(wret[k]) — and a deliberate UNSHIFTED lookahead canary is reported (excluded from family).
"""
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
import swing_pipeline as sp  # noqa: E402
import durable_pipeline as dp  # noqa: E402  (gnorm/net_of/summarize/per_year/pbo_cscv reuse)

OUT = HERE
WK = 52.0
RT_BP = 12.0
RT_SCEN = (6.0, 12.0, 24.0)
FUND_BP = 5.0
ZWINS = (26, 52, 104)
ZCLIP = 2.0
BETA_WIN = 104
HL_WIN = 104
HL_LO, HL_HI = 1.0, 26.0
N_TRIALS = 20
IDX = {r: i for i, r in enumerate(sp.ROOTS)}
PAIRS = [("GC", "SI"), ("GC", "PL"), ("SI", "PL"), ("PL", "PA"), ("CL", "NG")]


def roll_z(s: np.ndarray, W: int) -> np.ndarray:
    ser = pd.Series(s)
    mu = ser.rolling(W, min_periods=W).mean()
    sd = ser.rolling(W, min_periods=W).std()
    return ((ser - mu) / sd).to_numpy()


def roll_beta(a: np.ndarray, b: np.ndarray, W: int = BETA_WIN) -> np.ndarray:
    sa, sb = pd.Series(a), pd.Series(b)
    cov = sa.rolling(W, min_periods=W).cov(sb)
    var = sb.rolling(W, min_periods=W).var()
    return (cov / var).clip(0.2, 5.0).to_numpy()


def half_life(s: np.ndarray, W: int = HL_WIN) -> np.ndarray:
    """Trailing AR(1) half-life of the spread: ds_t = a + phi*s_{t-1}; hl = -ln2/ln(1+phi)."""
    n = len(s)
    hl = np.full(n, np.nan)
    for t in range(W, n):
        seg = s[t - W:t + 1]
        x, y = seg[:-1], np.diff(seg)
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() < int(0.8 * W):
            continue
        xv, yv = x[fin], y[fin]
        xd = xv - xv.mean()
        den = float((xd ** 2).sum())
        if den <= 0:
            continue
        phi = float((xd * (yv - yv.mean())).sum() / den)
        if -1.0 < phi < 0.0:
            hl[t] = -np.log(2.0) / np.log(1.0 + phi)
    return hl


def pair_books(lp: np.ndarray, zwin, use_beta: bool, hl_gate: bool) -> dict[str, np.ndarray]:
    """One (n,m) weight matrix per pair: continuous -clip(z,2)/2, 50/50-notional dollar-neutral."""
    n, m = lp.shape
    books = {}
    for A, B in PAIRS:
        a, b = IDX[A], IDX[B]
        beta = roll_beta(lp[:, a], lp[:, b]) if use_beta else np.ones(n)
        s = lp[:, a] - beta * lp[:, b]
        if isinstance(zwin, tuple):
            with np.errstate(all="ignore"):
                z = np.nanmean(np.column_stack([roll_z(s, w) for w in zwin]), axis=1)
        else:
            z = roll_z(s, zwin)
        pw = -np.clip(z, -ZCLIP, ZCLIP) / ZCLIP
        if hl_gate:
            hl = half_life(s)
            pw = np.where(np.isfinite(hl) & (hl >= HL_LO) & (hl <= HL_HI), pw, 0.0)
        pw = np.nan_to_num(pw)
        Wp = np.zeros((n, m))
        Wp[:, a] = 0.5 * pw
        Wp[:, b] = -0.5 * pw
        books[f"{A}-{B}"] = Wp
    return books


def pair_risk_parity(books: dict[str, np.ndarray], wret: np.ndarray) -> np.ndarray:
    """Inverse trailing 26wk pair-P&L vol (causal, shift 1), relative + bounded [0.25,4], summed."""
    out = None
    for Wp in books.values():
        r = np.nansum(Wp * np.nan_to_num(wret), axis=1)
        tv = pd.Series(r).rolling(26).std().shift(1).to_numpy()
        sc = np.where(np.isfinite(tv) & (tv > 0), 1.0 / tv, 0.0)
        med = np.nanmedian(sc[sc > 0]) if (sc > 0).any() else 1.0
        sc = np.clip(sc / max(med, 1e-12), 0.25, 4.0)
        scaled = Wp * sc[:, None]
        out = scaled if out is None else out + scaled
    return out


def main() -> None:
    curve = sp.load_curve_raw()
    cot = pd.read_parquet(Path("/home/speculation/convexity/data/cot/cot_disagg.parquet"))
    P = sp.build_panel(curve, cot)
    fridays, wret = P["fridays"], P["wret"]
    n, m = wret.shape

    # CAUSAL log price level at Friday-close t: wret[t] is the FORWARD t->t+1 return, so
    # lp[t] = sum of log1p(wret[k]) for k < t. (Unshifted = lookahead; kept only as the canary.)
    g = np.log1p(np.nan_to_num(wret, nan=0.0))
    lp = np.zeros((n, m))
    lp[1:] = np.cumsum(g, axis=0)[:-1]
    lp_look = np.cumsum(g, axis=0)          # CANARY ONLY: includes the return being traded

    results = {"panel": {"n_weeks": int(n), "span": f"{fridays[0].date()}..{fridays[-1].date()}"},
               "pairs": {}, "books": {}}
    fam = {}

    def run(tag: str, books: dict[str, np.ndarray], to_fam: bool = True, full: bool = True):
        comb = pair_risk_parity(books, wret)
        net, turn, valid = dp.net_of(comb, wret)
        if to_fam:
            fam[tag] = net
        r = dp.summarize(net, turn, valid, fridays, N_TRIALS, tag, want_year=full)
        if full:
            net_vt = sp.vol_target(net)
            r["voltarget"] = sp.metrics(net_vt)
            r["voltarget"]["per_year"] = dp.per_year(net_vt, valid, fridays)
            for rt in RT_SCEN:
                nn, _, _ = dp.net_of(comb, wret, rt)
                r[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
            for fb in (-FUND_BP, FUND_BP):
                nn, _, _ = dp.net_of(comb, wret, RT_BP, fb)
                r[f"sharpe_fund{int(fb):+d}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
        return r

    # ---- per-pair (band ensemble, ratio, ungated) ----
    band_books = pair_books(lp, ZWINS, use_beta=False, hl_gate=False)
    for name, Wp in band_books.items():
        net, turn, valid = dp.net_of(Wp, wret)
        fam[f"pair_{name}"] = net
        results["pairs"][name] = dp.summarize(net, turn, valid, fridays, N_TRIALS, name)

    # ---- combined books: per-W, band, OLS-beta band, HL-gated band (pre-named deployable) ----
    for W in ZWINS:
        results["books"][f"comb_z{W}"] = run(f"comb_z{W}", pair_books(lp, W, False, False),
                                             full=False)
    results["books"]["comb_band"] = run("comb_band", band_books)
    results["books"]["comb_band_olsbeta"] = run("comb_band_olsbeta",
                                                pair_books(lp, ZWINS, True, False))
    results["books"]["comb_band_hlgate"] = run("comb_band_hlgate",
                                               pair_books(lp, ZWINS, False, True))

    # ---- lookahead canary (EXCLUDED from family/PBO): unshifted levels must fake a huge edge ----
    canary = run("canary", pair_books(lp_look, ZWINS, False, False), to_fam=False, full=False)
    results["canary_lookahead"] = {k: canary.get(k) for k in ("sharpe_ann", "ev_bp_nmwk", "dsr")}

    # ---- PBO over the config family ----
    keys = sorted(fam)
    M = np.column_stack([fam[k] for k in keys])
    results["pbo"] = {"configs": keys, "pbo_cscv": dp.pbo_cscv(M)}

    (OUT / "results_rv.json").write_text(json.dumps(results, indent=1, default=str))
    print("RV DONE ->", OUT / "results_rv.json", flush=True)
    bk = results["books"]["comb_band_hlgate"]
    print("PRE-NAMED (band+HLgate):", {k: bk.get(k) for k in
          ("sharpe_ann", "ev_bp_nmwk", "win_pct", "maxdd_pct", "cagr_pct",
           "drop5_sharpe", "top5_share", "cpcv_frac_pos")}, "DSR=", bk["dsr"].get("dsr"))
    print("  voltarget:", {k: bk["voltarget"].get(k) for k in ("sharpe_ann", "maxdd_pct", "cagr_pct")})
    print("  per-year:", bk.get("per_year"))
    print("  scen:", {k: bk.get(k) for k in bk if k.startswith("sharpe_rt") or k.startswith("sharpe_fund")})
    for t in ("comb_band", "comb_band_olsbeta"):
        v = results["books"][t]
        print(f"{t}: Sh={v.get('sharpe_ann')} DD={v.get('maxdd_pct')} dr5={v.get('drop5_sharpe')} "
              f"DSR={v['dsr'].get('dsr')}")
    for W in ZWINS:
        v = results["books"][f"comb_z{W}"]
        print(f"comb_z{W}: Sh={v.get('sharpe_ann')} DSR={v['dsr'].get('dsr')}")
    print("PAIRS:", {k: v.get("sharpe_ann") for k, v in results["pairs"].items()})
    print("CANARY (lookahead, must be huge):", results["canary_lookahead"])
    print("PBO:", results["pbo"]["pbo_cscv"])


if __name__ == "__main__":
    main()
