#!/usr/bin/env python
"""P6 — LITERATURE DD-SHAPERS FOR THE RV BOOK (P6_PREREG.md, frozen 61ae544).

Candidates transform each pair's signal pw (state machines, all causal at the same
time-t-close convention the base construction uses) or scale book gross (L5). Composition and
netting follow p5_pipeline's equity-book conventions (gnorm once, scaled weights, costs on the
actually-traded book)."""
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
import durable_pipeline as dp  # noqa: E402
import rv_pipeline as rv  # noqa: E402

OUT = HERE
WK = 52.0
RT_BP = rv.RT_BP
N_EFF = 40
Z_STOP, Z_REARM = 3.0, 0.5
TS_MULT, TS_LO, TS_HI = 2.0, 4, 52
AL_IN, AL_OUT = 1.25, 0.5
HURST_WIN, HURST_THR = 104, 0.5
CROWD_WIN, CROWD_PCT, CROWD_SCALE, CROWD_BURN = 13, 80, 0.5, 104
PAIRS = list(rv.PAIRS)

curve = sp.load_curve_raw()
cot = pd.read_parquet("/home/speculation/convexity/data/cot/cot_disagg.parquet")
P = sp.build_panel(curve, cot)
FRIDAYS, WRET = P["fridays"], P["wret"]
N, M = WRET.shape
g = np.log1p(np.nan_to_num(WRET, nan=0.0))
LP = np.zeros((N, M))
LP[1:] = np.cumsum(g, axis=0)[:-1]


def hurst_av(s: np.ndarray, W: int = HURST_WIN) -> np.ndarray:
    """Trailing aggregated-variance Hurst: Var of k-step diffs ~ k^{2H}; regress over k=2..16."""
    ks = np.array([2, 4, 8, 16])
    out = np.full(len(s), np.nan)
    lk = np.log(ks)
    for t in range(W, len(s)):
        seg = s[t - W:t + 1]
        if not np.isfinite(seg).all():
            continue
        lv = []
        for k in ks:
            d = seg[k:] - seg[:-k]
            v = np.var(d)
            lv.append(np.log(v) if v > 0 else np.nan)
        lv = np.array(lv)
        if not np.isfinite(lv).all():
            continue
        slope = np.polyfit(lk, lv, 1)[0]
        out[t] = slope / 2.0
    return out


def base_pairs():
    """{name: dict(a,b,z,hl,gate,pw)} — the FROZEN Program-4 construction."""
    out = {}
    for A, B in PAIRS:
        a, b = rv.IDX[A], rv.IDX[B]
        s = LP[:, a] - LP[:, b]
        with np.errstate(all="ignore"):
            z = np.nanmean(np.column_stack([rv.roll_z(s, w) for w in rv.ZWINS]), axis=1)
        hl = rv.half_life(s)
        gate = np.isfinite(hl) & (hl >= rv.HL_LO) & (hl <= rv.HL_HI)
        pw = np.nan_to_num(np.where(gate, -np.clip(z, -rv.ZCLIP, rv.ZCLIP) / rv.ZCLIP, 0.0))
        out[f"{A}-{B}"] = {"a": a, "b": b, "s": s, "z": np.nan_to_num(z, nan=np.nan),
                           "hl": hl, "gate": gate, "pw": pw}
    return out


# ---------------- candidate transforms (each returns a new pw per pair; all causal) ----------
def l1_zstop(pr):
    z, pw = pr["z"], pr["pw"].copy()
    armed = True
    for t in range(N):
        zt = z[t]
        if not np.isfinite(zt):
            continue
        if armed and abs(zt) >= Z_STOP:
            armed = False
        elif not armed and abs(zt) <= Z_REARM:
            armed = True
        if not armed:
            pw[t] = 0.0
    return pw


def l2_timestop(pr):
    z, hl, pw = pr["z"], pr["hl"], pr["pw"].copy()
    age = 0
    sgn = 0.0
    armed = True
    for t in range(N):
        s_now = np.sign(pw[t])
        if s_now != 0 and s_now == sgn:
            age += 1
        else:
            age = 1 if s_now != 0 else 0
            sgn = s_now
            if s_now != 0:
                armed = True
        lim = TS_MULT * (hl[t] if np.isfinite(hl[t]) else TS_HI)
        lim = min(max(lim, TS_LO), TS_HI)
        if not armed:
            if np.isfinite(z[t]) and abs(z[t]) <= Z_REARM:
                armed = True
            else:
                pw[t] = 0.0
                continue
        if age > lim:
            armed = False
            pw[t] = 0.0
    return pw


def l3_albands(pr):
    z, gate = pr["z"], pr["gate"]
    pw = np.zeros(N)
    pos = 0.0
    for t in range(N):
        zt = z[t]
        if not np.isfinite(zt):
            pw[t] = pos if gate[t] else 0.0
            continue
        if pos == 0.0 and abs(zt) >= AL_IN:
            pos = -np.sign(zt)
        elif pos != 0.0 and abs(zt) <= AL_OUT:
            pos = 0.0
        pw[t] = pos if gate[t] else 0.0
    return pw


def l4_hurst(pr):
    h = hurst_av(pr["s"])
    return np.where(np.isfinite(h) & (h < HURST_THR), pr["pw"], 0.0)


def l5_crowd_scale(pairs_dict):
    """Book-gross scale: 0.5 while trailing mean |pairwise corr| of pair returns exceeds its
    expanding 80th percentile (causal: both the corr and its percentile use data through t-1)."""
    wret0 = np.nan_to_num(WRET)
    R = np.column_stack([(_pair_W(pr)[0] * wret0).sum(axis=1)
                         for pr in pairs_dict.values()])
    dfR = pd.DataFrame(R)
    mean_abs_corr = np.full(N, np.nan)
    for t in range(CROWD_WIN, N):
        c = dfR.iloc[t - CROWD_WIN:t].corr().to_numpy()
        iu = np.triu_indices_from(c, 1)
        mean_abs_corr[t] = np.nanmean(np.abs(c[iu]))
    scale = np.ones(N)
    for t in range(CROWD_BURN, N):
        hist = mean_abs_corr[CROWD_WIN:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 26 or not np.isfinite(mean_abs_corr[t]):
            continue
        thr = np.percentile(hist, CROWD_PCT)
        if mean_abs_corr[t] > thr:
            scale[t] = CROWD_SCALE
    return scale


def _pair_W(pr, pw=None):
    pw = pr["pw"] if pw is None else pw
    Wp = np.zeros((N, M))
    Wp[:, pr["a"]] = 0.5 * pw
    Wp[:, pr["b"]] = -0.5 * pw
    return Wp, pw


def compose(flags):
    """flags = set of accepted candidate names + optionally the trial one; deterministic."""
    prs = base_pairs()
    for name, pr in prs.items():
        pw = pr["pw"]
        if "l3" in flags:
            pr2 = dict(pr)
            pw = l3_albands(pr2)
            pr["pw"] = pw
        if "l1" in flags:
            pr["pw"] = l1_zstop(pr)
        if "l2" in flags:
            pr["pw"] = l2_timestop(pr)
        if "l4" in flags:
            pr["pw"] = l4_hurst(pr)
    books = {k: _pair_W(pr)[0] for k, pr in prs.items()}
    comb = rv.pair_risk_parity(books, WRET)
    Wu = dp.gnorm(comb)
    scale = np.ones(N)
    if "l5" in flags:
        scale = scale * l5_crowd_scale(prs)
    return Wu, scale


def net_scaled(Wu, scale, rt=RT_BP, fund=0.0):
    return sp.net_series(Wu * scale[:, None], WRET, rt, fund)


def evaluate(tag, Wu, scale, full=True):
    net, turn, valid = net_scaled(Wu, scale)
    r = dp.summarize(net, turn, valid, FRIDAYS, N_EFF, tag, want_year=full)
    if full:
        for rt in rv.RT_SCEN:
            nn, _, _ = net_scaled(Wu, scale, rt)
            r[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    return r, net


def main():
    res = {"prereg": "P6_PREREG.md"}
    fam = {}
    Wu0, sc0 = compose(set())
    s0, net0 = evaluate("S0", Wu0, sc0)
    fam["none"] = net0
    res["S0"] = s0
    print(f"S0 repro: {s0['sharpe_ann']} dd {s0['maxdd_pct']} (expect 0.732/18.5)", flush=True)

    accepted: set = set()
    cur = s0
    res["waterfall"] = {}
    for cand in ("l1", "l2", "l3", "l4", "l5"):
        Wu, sc = compose(accepted | {cand})
        mtr, nett = evaluate(f"WF_{cand}", Wu, sc)
        fam[f"{cand}_solo" if not accepted else f"{cand}_on_{'+'.join(sorted(accepted))}"] = nett
        dd0, dd1 = cur["maxdd_pct"], mtr["maxdd_pct"]
        c0, c1 = cur["cagr_pct"], mtr["cagr_pct"]
        rel_dd = (dd0 - dd1) / dd0 if dd0 > 0 else 0.0
        rel_cagr = (c0 - c1) / abs(c0) if abs(c0) > 1e-9 else 0.0
        cp0, cp1 = cur.get("cpcv_frac_pos", 0.0), mtr.get("cpcv_frac_pos", 0.0)
        ok = (rel_dd >= 0.15) and (rel_cagr <= 0.5 * rel_dd) and (cp1 >= cp0 - 0.05)
        res["waterfall"][cand] = {"metrics": mtr, "rel_dd_red": round(rel_dd, 3),
                                  "rel_cagr_red": round(rel_cagr, 3), "accepted": bool(ok)}
        print(f"WF {cand}: Sh={mtr['sharpe_ann']} dd={mtr['maxdd_pct']} cagr={mtr['cagr_pct']} "
              f"turn={mtr.get('turn_gross_wk')} relDD={rel_dd:.2f} relCAGR={rel_cagr:.2f} "
              f"cpcv={cp1} -> accepted={ok}", flush=True)
        if ok:
            accepted.add(cand)
            cur = mtr
    res["accepted"] = sorted(accepted)

    Wu_f, sc_f = compose(accepted)
    fin, net_f = evaluate("FINAL", Wu_f, sc_f)
    fam["final"] = net_f
    res["final"] = fin
    print("FINAL:", {k: fin.get(k) for k in ("sharpe_ann", "cagr_pct", "maxdd_pct",
          "drop5_sharpe", "cpcv_frac_pos")}, "DSR=", fin["dsr"].get("dsr"), flush=True)

    frontier = []
    for k in np.arange(0.5, 3.01, 0.25):
        nk, _, _ = net_scaled(Wu_f, sc_f * k)
        mk = sp.metrics(nk)
        frontier.append({"k": round(float(k), 2), "cagr_pct": mk.get("cagr_pct"),
                         "maxdd_pct": mk.get("maxdd_pct"), "sharpe_ann": mk.get("sharpe_ann")})
    res["frontier"] = frontier
    at25 = min(frontier, key=lambda x: abs((x["maxdd_pct"] or 99) - 25.0))
    ratio25 = (at25["cagr_pct"] or 0) / max(at25["maxdd_pct"] or 99, 1e-9)
    res["at_dd25"] = {"point": at25, "ratio": round(ratio25, 3)}

    keys = sorted(fam)
    L = min(len(v) for v in fam.values())
    Mx = np.column_stack([fam[k][:L] for k in keys])
    res["pbo"] = {"configs": keys, "pbo_cscv": dp.pbo_cscv(Mx)}

    gates = {
        "sharpe_ge_0.70": fin["sharpe_ann"] >= 0.70,
        "ratio_at_dd25_ge_0.60": ratio25 >= 0.60,
        "cpcv_ge_0.90": fin.get("cpcv_frac_pos", 0) >= 0.90,
        "drop5_pos": fin.get("drop5_sharpe", -1) > 0,
        "rt24_ge_0.45": fin.get("sharpe_rt24", 0) >= 0.45,
        "dsr_ge_0.80": (fin["dsr"].get("dsr") or 0) >= 0.80,
        "pbo_le_0.35": (res["pbo"]["pbo_cscv"] or 1.0) <= 0.35,
    }
    res["gates"] = {k: bool(v) for k, v in gates.items()}
    res["all_gates_pass"] = bool(all(gates.values()))

    (OUT / "results_p6.json").write_text(json.dumps(res, indent=1, default=str))
    print("GATES:", res["gates"], flush=True)
    print("at DD~25 sizing:", res["at_dd25"], flush=True)
    print("PBO:", res["pbo"]["pbo_cscv"], flush=True)
    print("P6 DONE ->", OUT / "results_p6.json", flush=True)


if __name__ == "__main__":
    main()
