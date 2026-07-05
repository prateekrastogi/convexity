#!/usr/bin/env python
"""DURABLE COMMODITY BOOK — slow TSMOM + carry-on-storables (DURABLE_PREREG.md, frozen 2026-07-05).

Reuses swing_pipeline VERBATIM for the data layer (roll-aware returns via instrument_id,
negative-price + winsor hygiene, weekly Friday panel, metrics/CPCV/per-obs-DSR). Adds the durable
sleeves as per-name TIME-SERIES books (not cross-sectional). Runs in the `commodity` worktree;
reads the shared lake at /home/speculation/convexity/data via swing_pipeline.REPO.
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
import swing_pipeline as sp  # noqa: E402  (data layer + metrics/cpcv/dsr; reads main-checkout data)

OUT = HERE
WK = 52.0
LBANDS = (13, 26, 39, 52)
RT_BP = 12.0
RT_SCEN = (6.0, 12.0, 24.0)
FUND_BP = 5.0
# root order from swing_pipeline.ROOTS == [CL, NG, GC, SI, HG, PL, PA]
IDX = {r: i for i, r in enumerate(sp.ROOTS)}
STOR = [IDX["CL"], IDX["NG"], IDX["HG"]]
CLUSTERS = {"precious": [IDX["GC"], IDX["SI"], IDX["PL"], IDX["PA"]],
            "copper": [IDX["HG"]], "crude": [IDX["CL"]], "natgas": [IDX["NG"]]}


def invvol(rvol: np.ndarray) -> np.ndarray:
    """1/vol with a per-week floor at 0.5x the cross-sectional median (blow-up guard, swing lesson)."""
    n, m = rvol.shape
    out = np.zeros((n, m))
    for t in range(n):
        v = rvol[t]
        fin = np.isfinite(v) & (v > 0)
        if fin.sum() == 0:
            continue
        floor = 0.5 * np.median(v[fin])
        out[t] = np.where(fin, 1.0 / np.maximum(v, floor), 0.0)
    return out


def weekly_logret(wret: np.ndarray) -> np.ndarray:
    return np.log1p(np.where(np.isfinite(wret), wret, np.nan))


def trend_sign(lr: np.ndarray, listed: np.ndarray) -> dict[int, np.ndarray]:
    """sign of trailing L-week logret, known at t (uses lr[t-L..t-1]); 0 if insufficient history."""
    n, m = lr.shape
    signs = {}
    for L in LBANDS:
        s = np.zeros((n, m))
        for t in range(L + 1, n):
            seg = lr[t - L:t]                       # returns realized through week t
            tot = np.nansum(seg, axis=0)
            cnt = np.isfinite(seg).sum(axis=0)
            ok = (cnt >= int(0.8 * L)) & listed[t]
            s[t] = np.where(ok, np.sign(tot), 0.0)
        signs[L] = s
    return signs


def carry_sign(carry: np.ndarray, listed: np.ndarray) -> np.ndarray:
    s = np.where(np.isfinite(carry) & listed, np.sign(carry), 0.0)
    return s


def net_of(W: np.ndarray, wret: np.ndarray, rt=RT_BP, fund=0.0):
    return sp.net_series(W, wret, rt, fund)


def per_year(net: np.ndarray, valid: np.ndarray, fridays: pd.DatetimeIndex) -> dict:
    idx = pd.DatetimeIndex(fridays[valid])
    s = pd.Series(net, index=idx.year)
    out = {}
    for y, g in s.groupby(level=0):
        out[int(y)] = round(float(g.mean() / g.std() * sqrt(WK)), 2) if (len(g) > 8 and g.std() > 0) else 0.0
    return out


def cluster_risk_parity(sleeve_W: np.ndarray, wret: np.ndarray) -> np.ndarray:
    """Scale each cluster's columns by inverse trailing (26wk) realized vol of its sub-book, so the
    4 clusters contribute equal risk. Causal (shift 1). Returns a combined weight matrix."""
    n, m = sleeve_W.shape
    out = np.zeros((n, m))
    for cols in CLUSTERS.values():
        sub = sleeve_W[:, cols]
        subret = np.nansum(sub * np.nan_to_num(wret[:, cols]), axis=1)
        tv = pd.Series(subret).rolling(26).std().shift(1).to_numpy()
        scale = np.where(np.isfinite(tv) & (tv > 0), 1.0 / tv, 0.0)
        scale = np.clip(scale / np.nanmedian(scale[scale > 0]), 0.25, 4.0)  # relative, bounded
        out[:, cols] = sub * scale[:, None]
    return out


def summarize(net, turn, valid, fridays, n_eff, label, want_year=True):
    m = sp.metrics(net, turn)
    m["dsr"] = sp.dsr(net, n_eff)
    if len(net) > 60:
        paths = sp.cpcv_paths(net, valid, embargo=8)
        m["cpcv_mean_path"] = round(float(np.mean(paths)), 3)
        m["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in paths])), 2)
    if want_year:
        m["per_year"] = per_year(net, valid, fridays)
    return m


def main() -> None:
    curve = sp.load_curve_raw()
    cot = pd.read_parquet(Path("/home/speculation/convexity/data/cot/cot_disagg.parquet"))
    P = sp.build_panel(curve, cot)
    fridays = P["fridays"]
    wret, carry, rvol, listed = P["wret"], P["carry"], P["rvol"], P["listed"]
    n, m = wret.shape
    iv = invvol(rvol)
    lr = weekly_logret(wret)
    tsigns = trend_sign(lr, listed)
    csign = carry_sign(carry, listed)

    results = {"panel": {"n_weeks": int(n), "span": f"{fridays[0].date()}..{fridays[-1].date()}",
                         "breadth_med": int(np.median(listed.sum(1)))}, "sleeves": {}, "combined": {}}

    # ---- S1 TSMOM per-L + band ensemble (all 7, vol-scaled directional) ----
    s1_L = {}
    for L in LBANDS:
        W = tsigns[L] * iv
        s1_L[L] = W
        net, turn, valid = net_of(W, wret)
        results["sleeves"][f"S1_tsmom_L{L}"] = summarize(net, turn, valid, fridays, 15, f"S1_L{L}")
    s1_band = np.mean([tsigns[L] for L in LBANDS], axis=0) * iv   # band ensemble
    net, turn, valid = net_of(s1_band, wret)
    results["sleeves"]["S1_tsmom_band"] = summarize(net, turn, valid, fridays, 15, "S1_band")

    # ---- S2 carry-on-storables (TS, {CL,NG,HG}) ----
    s2 = np.zeros((n, m))
    for i in STOR:
        s2[:, i] = csign[:, i] * iv[:, i]
    net, turn, valid = net_of(s2, wret)
    results["sleeves"]["S2_carry_storable"] = summarize(net, turn, valid, fridays, 15, "S2")

    # ---- S3 interaction: carry-gated / veto trend on storables ----
    band_sign = np.sign(np.mean([tsigns[L] for L in LBANDS], axis=0))
    s3_gate = np.zeros((n, m))
    s3_veto = np.zeros((n, m))
    for i in STOR:
        agree = band_sign[:, i] == csign[:, i]
        disagree = (band_sign[:, i] != 0) & (csign[:, i] != 0) & (band_sign[:, i] != csign[:, i])
        s3_gate[:, i] = np.where(agree, band_sign[:, i], 0.0) * iv[:, i]
        s3_veto[:, i] = np.where(disagree, 0.0, band_sign[:, i]) * iv[:, i]
    for nm, W in (("S3_gate_storable", s3_gate), ("S3_veto_storable", s3_veto)):
        net, turn, valid = net_of(W, wret)
        results["sleeves"][nm] = summarize(net, turn, valid, fridays, 15, nm)

    # ---- S5 KRT overlay (COT flow-shock reversal, storables+all where COT exists) ----
    prev1 = np.full((n, m), np.nan)
    prev1[1:] = wret[:-1]
    s5 = np.zeros((n, m))
    for t in range(n):
        row = P["cot_dmm"][t]
        fin = np.isfinite(row) & listed[t] & np.isfinite(prev1[t])
        if fin.sum() >= 3:
            med = np.nanmedian(row[fin])
            hot = fin & (row >= med)
            s5[t, hot] = -np.sign(prev1[t, hot]) * iv[t, hot]
    net, turn, valid = net_of(s5, wret)
    results["sleeves"]["S5_krt"] = summarize(net, turn, valid, fridays, 15, "S5")

    # ---- COMBINED (pre-named): preciouses S1-band trend + storables (S2 + S3-gate) ----
    comb = np.zeros((n, m))
    for i in CLUSTERS["precious"]:
        comb[:, i] = s1_band[:, i]
    for i in STOR:
        comb[:, i] = s2[:, i] + s3_gate[:, i]
    comb += 0.2 * s5  # S5 <=20% risk share (declared)
    comb = cluster_risk_parity(comb, wret)
    net_raw, turn, valid = net_of(comb, wret)
    net_vt = sp.vol_target(net_raw)
    cm = summarize(net_raw, turn, valid, fridays, 15, "combined")
    cm["voltarget"] = sp.metrics(net_vt)
    cm["voltarget"]["per_year"] = per_year(net_vt, valid, fridays)
    for rt in RT_SCEN:
        nn, _, _ = net_of(comb, wret, rt)
        cm[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    for fb in (-FUND_BP, FUND_BP):
        nn, _, _ = net_of(comb, wret, RT_BP, fb)
        cm[f"sharpe_fund{int(fb):+d}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    if net_raw.mean() > 0 and net_raw.var() > 0:
        kelly = float(net_raw.mean() / net_raw.var())
        cm["kelly_full"] = round(kelly, 2)
        cm["kelly_quarter_lev"] = round(0.25 * kelly, 2)
    results["combined"]["book"] = cm

    (OUT / "results_durable.json").write_text(json.dumps(results, indent=1, default=str))
    print("DURABLE DONE ->", OUT / "results_durable.json", flush=True)
    print("combined:", {k: cm.get(k) for k in ("sharpe_ann", "sharpe_rt6", "maxdd_pct", "cagr_pct")},
          "DSR=", cm["dsr"].get("dsr"))
    print("combined per-year:", cm.get("per_year"))
    print("S1_band:", {k: results["sleeves"]["S1_tsmom_band"].get(k)
                       for k in ("sharpe_ann", "cpcv_frac_pos")},
          "per-year:", results["sleeves"]["S1_tsmom_band"].get("per_year"))


if __name__ == "__main__":
    main()
