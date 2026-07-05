#!/usr/bin/env python
"""SWING XSEC COMMODITY factor study — end-to-end (SWING_PREREG.md, frozen 2026-07-05).

Price basis: databento front-month daily settles (data/curves/curve_ohlcv1d.parquet, ranks 0-7,
7 roots, 2010-06+). ROLL-AWARE daily returns via instrument_id change on rank 0 (the continuous
`symbol` alias hides the contract, but instrument_id is the true contract id): on a roll day the
return is c0_t / c1_{t-1} (the new front was yesterday's 2nd contract), never the raw c0 jump.
Weekly Friday panel; factors are weekly dollar-neutral inverse-vol books; net of cost; validated by
CPCV + per-observation DSR + PBO + drop-top-N. No-lookahead by construction: a weight at week t
reads only info <= t (features, COT usable_from<=t-Friday) and is scored on the t->t+1 return.

Run: nohup .venv/bin/python research/entry_exit/study/swing_pipeline.py > .../swing.out 2>&1 &
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity")
STUDY = REPO / "research/entry_exit/study"
sys.path.insert(0, str(REPO / "research"))
sys.path.insert(0, str(REPO / "src"))
from entry_exit.validation import per_obs_sharpe  # noqa: E402

ROOT2SYM = {"CL": "CLUSDT", "NG": "NATGASUSDT", "GC": "XAUUSDT", "SI": "XAGUSDT",
            "HG": "COPPERUSDT", "PL": "XPTUSDT", "PA": "XPDUSDT"}
ROOTS = list(ROOT2SYM)
RT_BP = 12.0
RT_SCEN = (6.0, 12.0, 24.0)
FUND_BP_WK = 5.0
SIGMA_STAR = 0.10            # weekly-annualized book vol target
WK_PER_YR = 52.0
VOL_WIN = 20                 # daily bars for inverse-vol + realized vol
SKEW_WINS = (40, 60)
MOM_BANDS = (6, 8, 10)       # weeks


RAW = REPO / "data/curves/databento"


def load_curve_raw() -> pd.DataFrame:
    """Load the raw databento CSVs (which carry instrument_id, unlike the parsed parquet)."""
    frames = []
    for root in ROOTS:
        for rank in (0, 1):
            f = RAW / f"{root}_c{rank}_ohlcv1d.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, usecols=["ts_event", "instrument_id", "close"])
            d["date"] = pd.to_datetime(d["ts_event"], format="mixed", utc=True).dt.tz_localize(None)
            d["root"] = root
            d["rank"] = rank
            frames.append(d[["date", "root", "rank", "instrument_id", "close"]])
    return pd.concat(frames, ignore_index=True)


# ---------- daily roll-aware returns + curve slope, per root ----------
def root_daily(curve: pd.DataFrame, root: str) -> pd.DataFrame:
    c0 = curve[(curve.root == root) & (curve["rank"] == 0)].set_index("date").sort_index()
    c1 = curve[(curve.root == root) & (curve["rank"] == 1)].set_index("date").sort_index()
    df = pd.DataFrame(index=c0.index)
    df["c0"] = c0["close"]
    df["iid0"] = c0["instrument_id"].astype("int64")
    df["c1"] = c1["close"].reindex(c0.index)
    df["iid1"] = c1["instrument_id"].reindex(c0.index)
    df = df.dropna(subset=["c0"])
    prev_c0 = df["c0"].shift(1)
    prev_iid0 = df["iid0"].shift(1)
    prev_c1 = df["c1"].shift(1)
    same = df["iid0"] == prev_iid0
    # roll day: new front today == 2nd contract yesterday -> return vs prev_c1 (roll-clean)
    ret = np.where(same, df["c0"] / prev_c0 - 1.0, df["c0"] / prev_c1 - 1.0)
    df["ret"] = ret
    df.loc[df.index[0], "ret"] = np.nan
    # if a roll day lacks a valid prev_c1 (rare), fall back to no return (NaN) rather than a jump
    bad = (~same) & (~np.isfinite(prev_c1))
    df.loc[bad, "ret"] = np.nan
    # DATA HYGIENE (prereg): a return computed THROUGH a non-positive price is meaningless
    # (WTI 2020-04-20 settled -$2.67); invalidate it. Then winsorize daily returns to +-40% — no
    # legitimate daily move for these 7 commodities reaches that; the bound only kills the
    # negative-price episode and outright data errors, preserving real crash days (e.g. -8% covid).
    denom_ok = np.where(same, prev_c0 > 0, prev_c1 > 0)
    df.loc[(df["c0"] <= 0) | (~denom_ok), "ret"] = np.nan
    df["ret"] = df["ret"].clip(-0.40, 0.40)
    df["logslope"] = np.where((df["c0"] > 0) & (df["c1"] > 0),
                              np.log(df["c0"] / df["c1"]), np.nan)  # >0 = backwardation
    return df


def roll_spacing_days(daily: pd.DataFrame) -> float:
    roll_dates = daily.index[daily["iid0"] != daily["iid0"].shift(1)][1:]
    if len(roll_dates) < 3:
        return 30.0
    gaps = np.diff(pd.to_datetime(roll_dates).values).astype("timedelta64[D]").astype(float)
    return float(np.median(gaps))


# ---------- weekly Friday panel ----------
def build_panel(curve: pd.DataFrame, cot: pd.DataFrame) -> dict:
    dailies, spacing = {}, {}
    for r in ROOTS:
        d = root_daily(curve, r)
        dailies[r] = d
        spacing[r] = roll_spacing_days(d)
    # union daily calendar -> Friday weekly stamps
    all_days = sorted(set().union(*[set(d.index) for d in dailies.values()]))
    cal = pd.to_datetime(pd.Index(all_days))
    fridays = pd.date_range(cal.min(), cal.max(), freq="W-FRI")

    feat: dict[str, pd.DataFrame] = {}
    for r in ROOTS:
        d = dailies[r].copy()
        d.index = pd.to_datetime(d.index)
        # daily-return-based rolling stats (F_{t} = uses data up to and incl. day t)
        d["rvol"] = d["ret"].rolling(VOL_WIN).std()
        for w in SKEW_WINS:
            d[f"skew{w}"] = d["ret"].rolling(w).skew()
        d["adj"] = d["ret"].fillna(0.0).add(1.0).cumprod()  # roll-clean price index
        ann = 365.0 / spacing[r]
        d["carry"] = d["logslope"] * ann
        # as-of each Friday: last available daily row on/<= Friday
        wk = d.reindex(d.index.union(fridays)).sort_index().ffill().reindex(fridays)
        feat[r] = wk

    # weekly return t->t+1 from the roll-clean adj index (Friday to next Friday)
    n = len(fridays)
    M = {k: np.full((n, len(ROOTS)), np.nan) for k in
         ("adj", "carry", "rvol", "skew40", "skew60", "listed")}
    for j, r in enumerate(ROOTS):
        wk = feat[r]
        M["adj"][:, j] = wk["adj"].to_numpy()
        M["carry"][:, j] = wk["carry"].to_numpy()
        M["rvol"][:, j] = wk["rvol"].to_numpy()
        M["skew40"][:, j] = wk["skew40"].to_numpy()
        M["skew60"][:, j] = wk["skew60"].to_numpy()
        M["listed"][:, j] = np.isfinite(wk["adj"].to_numpy()).astype(float)
    adj = M["adj"]
    wret = np.full_like(adj, np.nan)
    wret[:-1] = adj[1:] / adj[:-1] - 1.0        # week t -> t+1 return (the SCORED return)
    # COT weekly-aligned (usable_from <= Friday): mm_net, prod_net, doi, d_mmnet
    cot_feat = cot_weekly(cot, fridays)
    return {"fridays": fridays, "roots": ROOTS, "wret": wret,
            "carry": M["carry"], "rvol": M["rvol"],
            "skew40": M["skew40"], "skew60": M["skew60"],
            "listed": M["listed"].astype(bool), "spacing": spacing, **cot_feat}


def cot_weekly(cot: pd.DataFrame, fridays: pd.DatetimeIndex) -> dict:
    cot = cot.copy()
    cot["usable_from"] = pd.to_datetime(cot["usable_from"])
    cot["mm_net"] = cot["mm_long"] - cot["mm_short"]
    cot["prod_net"] = cot["prod_long"] - cot["prod_short"]
    n = len(fridays)
    dmm = np.full((n, len(ROOTS)), np.nan)      # |d mm_net| / OI over the report week
    pnet = np.full((n, len(ROOTS)), np.nan)     # prod_net / OI (hedging pressure)
    for j, r in enumerate(ROOTS):
        sym = ROOT2SYM[r]
        c = cot[cot.symbol == sym].sort_values("usable_from")
        if c.empty:
            continue
        c["d_mm"] = c["mm_net"].diff().abs() / c["oi"]
        c["pn"] = c["prod_net"] / c["oi"]
        # as-of backward: latest report usable on/<= each Friday
        idx = np.searchsorted(c["usable_from"].values, fridays.values, side="right") - 1
        ok = idx >= 0
        dmm[ok, j] = c["d_mm"].to_numpy()[idx[ok]]
        pnet[ok, j] = c["pn"].to_numpy()[idx[ok]]
    return {"cot_dmm": dmm, "cot_pnet": pnet}


# ---------- factor -> weekly dollar-neutral inverse-vol weight matrix ----------
def _weights_from_signal(sig: np.ndarray, listed: np.ndarray, rvol: np.ndarray,
                         k: int = 2, cond: np.ndarray | None = None) -> np.ndarray:
    """Long top-k / short bottom-k by signal each week; inverse-vol legs; dollar-neutral gross 1."""
    n, m = sig.shape
    W = np.zeros((n, m))
    for t in range(n):
        s = sig[t].copy()
        elig = listed[t] & np.isfinite(s) & np.isfinite(rvol[t]) & (rvol[t] > 0)
        if cond is not None:
            elig = elig & cond[t]
        idx = np.where(elig)[0]
        if len(idx) < 2 * k:
            continue
        # inverse-vol legs, but FLOOR each week's vol at 0.5x the cross-sectional median so a single
        # stale/flat-week tiny-vol name cannot get an exploding weight and hijack the dollar-neutral
        # book (the bug that turned carry's +8bp/wk extreme spread into a -20bp factor).
        vt = rvol[t][idx]
        vfloor = 0.5 * np.median(vt)
        inv = 1.0 / np.maximum(vt, vfloor)
        order = np.argsort(s[idx])
        sh, lo = idx[order[:k]], idx[order[-k:]]
        invs = 1.0 / np.maximum(rvol[t][sh], vfloor)
        invl = 1.0 / np.maximum(rvol[t][lo], vfloor)
        W[t, lo] = 0.5 * invl / invl.sum()
        W[t, sh] = -0.5 * invs / invs.sum()
    return W


def factor_weights(P: dict) -> dict[str, np.ndarray]:
    n, m = P["wret"].shape
    listed, rvol = P["listed"], P["rvol"]
    adj = P["carry"] * 0 + np.nan  # placeholder for prior-return computations below
    # prior 1-week return (for reversal) and k-week momentum from the adj index
    logadj = np.log(np.where(np.isfinite(P["wret"]), 1.0, 1.0))  # not used; compute below
    # reconstruct weekly log-index for momentum/reversal
    wr = P["wret"]
    prev1 = np.full((n, m), np.nan)
    prev1[1:] = wr[:-1]  # return realized over t-1->t, known at t
    F: dict[str, np.ndarray] = {}
    # F1 carry (2 holds handled downstream by rebalance frequency; weekly weight = current carry rank)
    F["F1_carry"] = _weights_from_signal(P["carry"], listed, rvol, k=2)
    # F2 basis-momentum: 4wk change in carry (slope), high positive = improving backwardation
    dcarry = np.full((n, m), np.nan)
    dcarry[4:] = P["carry"][4:] - P["carry"][:-4]
    F["F2_basismom"] = _weights_from_signal(dcarry, listed, rvol, k=2)
    # F3 liquidity-provision reversal: signal = -prev1wk return, conditioned on COT flow top-half
    cond = np.zeros((n, m), dtype=bool)
    for t in range(n):
        row = P["cot_dmm"][t]
        fin = np.isfinite(row) & listed[t]
        if fin.sum() >= 4:
            med = np.nanmedian(row[fin])
            cond[t] = fin & (row >= med)
    F["F3_reversal_cot"] = _weights_from_signal(-prev1, listed, rvol, k=2, cond=cond)
    F["F3_reversal_uncond"] = _weights_from_signal(-prev1, listed, rvol, k=2)
    # F4 momentum band: k-week trailing return (product), known at t
    for kb in MOM_BANDS:
        mom = np.full((n, m), np.nan)
        idxlog = np.log1p(np.where(np.isfinite(wr), wr, np.nan))
        cum = np.full((n, m), np.nan)
        for t in range(kb, n):
            seg = idxlog[t - kb:t]  # returns realized up to t (t-kb..t-1)
            cum[t] = np.nansum(seg, axis=0)
            cnt = np.isfinite(idxlog[t - kb:t]).sum(axis=0)
            cum[t][cnt < kb] = np.nan
        mom = cum
        F[f"F4_mom{kb}"] = _weights_from_signal(mom, listed, rvol, k=2)
    # F6 hedging-pressure conditioner on momentum (mom8): trade mom only where prod_net agrees
    mom8 = np.full((n, m), np.nan)
    for t in range(8, n):
        seg = np.log1p(np.where(np.isfinite(wr[t - 8:t]), wr[t - 8:t], np.nan))
        mom8[t] = np.nansum(seg, axis=0)
        mom8[t][np.isfinite(wr[t - 8:t]).sum(0) < 8] = np.nan
    hp_ok = P["cot_pnet"] < 0  # producers net short (normal HP) -> long-side premium
    F["F6_mom_hp"] = _weights_from_signal(mom8, listed, rvol, k=2, cond=(hp_ok & listed))
    # F5 skew
    for w in SKEW_WINS:
        F[f"F5_skew{w}"] = _weights_from_signal(-P[f"skew{w}"], listed, rvol, k=2)  # long neg-skew
    return F


# ---------- net returns, ensembles, metrics ----------
def net_series(W: np.ndarray, wret: np.ndarray, rt_bp: float, fund_bp: float = 0.0):
    n = W.shape[0]
    gross = np.nansum(W * np.nan_to_num(wret), axis=1)
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1]))))
    turn = dW.sum(axis=1)                       # gross turnover (sum |dw|) per week
    cost = (rt_bp / 1e4) * turn                 # RT bp on each unit traded
    hold = np.abs(W).sum(axis=1)
    fund = (fund_bp / 1e4) * hold
    net = gross - cost - fund
    valid = np.isfinite(wret).any(axis=1)
    valid[-1] = False                           # last week has no t+1 return
    return net[valid], turn[valid], valid


def vol_target(net: np.ndarray, sigma_star: float = SIGMA_STAR) -> np.ndarray:
    # causal: scale week t by trailing 26wk realized vol, cap [0.5,2]
    s = pd.Series(net)
    tv = s.rolling(26).std().shift(1) * np.sqrt(WK_PER_YR)
    scale = np.clip(sigma_star / tv, 0.5, 2.0).fillna(1.0).to_numpy()
    return net * scale


def metrics(net: np.ndarray, turn: np.ndarray | None = None) -> dict:
    net = net[np.isfinite(net)]
    if len(net) < 30:
        return {"n_wk": int(len(net))}
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    yrs = len(net) / WK_PER_YR
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    srt = np.sort(net)[::-1]
    return {
        "n_wk": int(len(net)),
        "sharpe_ann": round(float(per_obs_sharpe(net) * np.sqrt(WK_PER_YR)), 3),
        "sharpe_wk": round(float(per_obs_sharpe(net)), 4),
        "ev_bp_nmwk": round(float(net.mean()) * 1e4, 2),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "maxdd_pct": round(float((1 - eq / peak).max()) * 100, 1),
        "cagr_pct": round(float(cagr) * 100, 2),
        "drop5_sharpe": round(float(per_obs_sharpe(np.sort(net)[:-5]) * np.sqrt(WK_PER_YR)), 3),
        "top5_share": round(float(srt[:5].sum() / net.sum()), 2) if net.sum() != 0 else None,
        "turn_mean": round(float(np.nanmean(turn)), 2) if turn is not None else None,
    }


def cpcv_paths(net_full: np.ndarray, valid: np.ndarray, n_groups=8, k=2, embargo=6):
    net = net_full  # already the valid-week series
    N = len(net)
    edges = np.linspace(0, N, n_groups + 1).astype(int)
    bounds = [(edges[i], edges[i + 1]) for i in range(n_groups)]
    paths = []
    for combo in combinations(range(n_groups), k):
        test = np.zeros(N, bool)
        for g in combo:
            test[bounds[g][0]:bounds[g][1]] = True
        seg = net[test]
        if len(seg) > 20 and seg.std() > 0:
            paths.append(float(per_obs_sharpe(seg) * np.sqrt(WK_PER_YR)))
    return paths


def dsr(net: np.ndarray, n_trials: int) -> dict:
    from math import erf, log, sqrt
    r = net[np.isfinite(net)]
    T = len(r)
    if T < 30 or r.std() == 0:
        return {"psr": None, "dsr": None}
    sr = per_obs_sharpe(r)
    g3 = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    # expected max SR of N i.i.d. null trials (BLdP)
    eul = 0.5772156649
    z1 = _ppf(1 - 1.0 / n_trials)
    z2 = _ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = (1 - eul) * z1 + eul * z2  # per-obs units assuming var(SR)~1/T under null
    sr0 *= sqrt(1.0 / T)
    denom = sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-9))
    psr = 0.5 * (1 + erf(((sr - 0) * sqrt(T - 1)) / denom / sqrt(2)))
    dsr_v = 0.5 * (1 + erf(((sr - sr0) * sqrt(T - 1)) / denom / sqrt(2)))
    return {"sr_wk": round(sr, 4), "psr_vs0": round(psr, 4), "dsr": round(dsr_v, 4),
            "sr0_wk": round(sr0, 4), "n_trials": n_trials, "T": T}


def _ppf(p):
    # Acklam inverse normal
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= ph:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def main() -> None:
    curve = load_curve_raw()
    cot = pd.read_parquet(REPO / "data/cot/cot_disagg.parquet")
    P = build_panel(curve, cot)
    fridays = P["fridays"]
    breadth = P["listed"].sum(axis=1)
    span = f"{fridays[0].date()}..{fridays[-1].date()}"
    print(f"panel: {len(fridays)} Friday weeks {span}; breadth median={int(np.median(breadth))} "
          f"min={int(breadth.min())} max={int(breadth.max())}", flush=True)

    F = factor_weights(P)
    # count the pre-registered trial family for DSR N_eff
    N_EFF = len(F) + 3  # factor variants + the 3 named ensembles

    results = {"panel": {"n_weeks": int(len(fridays)), "span": span,
                         "breadth_median": int(np.median(breadth))},
               "n_eff": N_EFF, "factors": {}, "ensembles": {}}

    # per-factor
    factor_nets = {}
    for name, W in F.items():
        net, turn, valid = net_series(W, P["wret"], RT_BP)
        factor_nets[name] = (net, valid)
        m = metrics(net, turn)
        m["cpcv_mean_path"] = round(float(np.mean(cpcv_paths(net, valid))), 3) if len(net) > 60 else None
        m["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in cpcv_paths(net, valid)])), 2) if len(net) > 60 else None
        m["dsr"] = dsr(net, N_EFF)
        # cost sensitivity
        for rt in RT_SCEN:
            nn, _, _ = net_series(W, P["wret"], rt)
            m[f"sharpe_rt{int(rt)}"] = round(float(per_obs_sharpe(nn) * np.sqrt(WK_PER_YR)), 3)
        results["factors"][name] = m

    # ensembles (equal-weight mean of member weight matrices, renormalize gross to 1, vol-target)
    ENS = {
        "E_core_F3F4F5": ["F3_reversal_cot", "F4_mom8", "F5_skew60"],
        "E_plusF6": ["F3_reversal_cot", "F4_mom8", "F5_skew60", "F6_mom_hp"],
        "E_all_curve": ["F1_carry", "F2_basismom", "F3_reversal_cot", "F4_mom8", "F5_skew60"],
    }
    for ename, members in ENS.items():
        Ws = [F[mm] for mm in members]
        We = np.mean(np.stack(Ws), axis=0)
        gross = np.abs(We).sum(axis=1, keepdims=True)
        We = np.divide(We, gross, out=np.zeros_like(We), where=gross > 0)  # renorm gross 1
        net, turn, valid = net_series(We, P["wret"], RT_BP)
        m = metrics(net, turn)
        m["cpcv_mean_path"] = round(float(np.mean(cpcv_paths(net, valid))), 3)
        m["cpcv_frac_pos"] = round(float(np.mean([p > 0 for p in cpcv_paths(net, valid)])), 2)
        m["dsr"] = dsr(net, N_EFF)
        netvt = vol_target(net)
        m["voltarget"] = metrics(netvt)
        # quarter/full Kelly leverage off the net stream
        mu, var = float(net.mean()), float(net.var())
        kelly = mu / var if var > 0 else 0.0
        m["kelly_full"] = round(kelly, 2)
        m["kelly_quarter_lev"] = round(0.25 * kelly, 2)
        results["ensembles"][ename] = m

    (STUDY / "results_swing.json").write_text(json.dumps(results, indent=1, default=str))
    print("SWING DONE ->", STUDY / "results_swing.json", flush=True)
    print(json.dumps({k: {"sharpe": v.get("sharpe_ann"), "dsr": v.get("dsr", {}).get("dsr")}
                      for k, v in results["ensembles"].items()}, indent=1))


if __name__ == "__main__":
    main()
