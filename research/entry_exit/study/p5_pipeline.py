#!/usr/bin/env python
"""P5 — RV BREADTH (HG pairs) + DD WATERFALL + SIZING (P5_PREREG.md, frozen 2026-07-12).

Reuses rv_pipeline / swing_pipeline / durable_pipeline VERBATIM for data, costs, metrics,
CPCV/DSR/PBO. Overlay semantics follow the EQUITY book's convention: the base book is
unit-gross-normalized ONCE (gnorm), then causal per-week scale factors multiply it WITHOUT
re-normalization, and costs are charged on the SCALED (actually-traded) weights.

Causality notes (each stated where implemented): brake m_t and stop-reset decisions use equity
through t-1; active-fraction and HL gates are part of the time-t target construction (same
convention as the deployed book).
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
import durable_pipeline as dp  # noqa: E402
import rv_pipeline as rv  # noqa: E402

OUT = HERE
WK = 52.0
RT_BP = rv.RT_BP
N_EFF = 30          # cumulative honesty: Program 4's 20 + P5's <=10 cells
BRAKE_CEIL = 0.25
STOP_THR = 0.25
PAIRS5 = list(rv.PAIRS)
PAIRS_NEW = [("HG", "CL"), ("HG", "PL"), ("GC", "HG")]
PAIRS8 = PAIRS5 + PAIRS_NEW

# ---------------------------------------------------------------- panel (built once, global)
curve = sp.load_curve_raw()
cot = pd.read_parquet("/home/speculation/convexity/data/cot/cot_disagg.parquet")
P = sp.build_panel(curve, cot)
FRIDAYS, WRET = P["fridays"], P["wret"]
N, M = WRET.shape
g = np.log1p(np.nan_to_num(WRET, nan=0.0))
LP = np.zeros((N, M))
LP[1:] = np.cumsum(g, axis=0)[:-1]          # causal shifted levels (rv convention)


def pair_books(pairs, hl_gate=True):
    """{name: (Wp, gate_open, pw)} — band z {26,52,104}, beta=1, the FROZEN deployed construction."""
    books = {}
    for A, B in pairs:
        a, b = rv.IDX[A], rv.IDX[B]
        s = LP[:, a] - LP[:, b]
        with np.errstate(all="ignore"):
            z = np.nanmean(np.column_stack([rv.roll_z(s, w) for w in rv.ZWINS]), axis=1)
        pw = -np.clip(z, -rv.ZCLIP, rv.ZCLIP) / rv.ZCLIP
        hl = rv.half_life(s)
        gate = np.isfinite(hl) & (hl >= rv.HL_LO) & (hl <= rv.HL_HI)
        if hl_gate:
            pw = np.where(gate, pw, 0.0)
        pw = np.nan_to_num(pw)
        Wp = np.zeros((N, M))
        Wp[:, a] = 0.5 * pw
        Wp[:, b] = -0.5 * pw
        books[f"{A}-{B}"] = (Wp, gate, pw)
    return books


def combine_rp(books):
    return rv.pair_risk_parity({k: v[0] for k, v in books.items()}, WRET)


def net_scaled(Wu, scale, rt=RT_BP, fund=0.0):
    """Unit-gross base x causal scale, NO renormalization; costs on scaled weights."""
    W = Wu * scale[:, None]
    return sp.net_series(W, WRET, rt, fund)


def brake_scale(Wu, base_scale, ceiling=BRAKE_CEIL):
    """GZ brake: m_t = clip(1 - DD_{t-1}/ceiling, 0, 1) off the braked book's OWN equity.
    Sequential and causal: equity/peak include returns through t-1 only; costs (12bp on turnover
    of the actually-held weights) are inside the equity evolution, matching net_series."""
    n = Wu.shape[0]
    scale = np.ones(n)
    eq, peak = 1.0, 1.0
    prevW = np.zeros(Wu.shape[1])
    wret0 = np.nan_to_num(WRET)
    for t in range(n):
        dd = 1.0 - eq / peak
        m = max(0.0, min(1.0, 1.0 - dd / ceiling))
        scale[t] = m
        W = Wu[t] * base_scale[t] * m
        r = float((W * wret0[t]).sum()) - (RT_BP / 1e4) * float(np.abs(W - prevW).sum())
        eq *= (1.0 + r)
        peak = max(peak, eq)
        prevW = W
    return scale


def active_fraction(books):
    """Gross ∝ fraction of pairs whose HL gate is open at t (time-t target construction, the
    equity book's breadth analog; no renormalization)."""
    gates = np.column_stack([v[1] for v in books.values()])
    return gates.mean(axis=1)


def stop_reset(books, thr=STOP_THR):
    """Per-pair kill at pair running-DD > thr (equity from current activation), re-arm when the
    pair's z re-crosses zero (pw hits 0 or flips sign). DD decision uses pair returns through
    t-1; week t's pw is zeroed on/after the kill decision."""
    out = {}
    wret0 = np.nan_to_num(WRET)
    for name, (Wp, gate, pw) in books.items():
        r = (Wp * wret0).sum(axis=1)
        pw2 = pw.copy()
        eq, peak = 1.0, 1.0
        killed = False
        kill_sign = 0.0
        for t in range(N):
            if killed:
                if pw[t] == 0.0 or (kill_sign != 0.0 and np.sign(pw[t]) != kill_sign):
                    killed, eq, peak = False, 1.0, 1.0
                else:
                    pw2[t] = 0.0
                    continue
            dd = 1.0 - eq / peak
            if dd > thr:
                killed = True
                kill_sign = np.sign(pw[t]) if pw[t] != 0.0 else 0.0
                pw2[t] = 0.0
                continue
            eq *= (1.0 + r[t])
            peak = max(peak, eq)
        a = np.flatnonzero(np.abs(Wp).sum(axis=0))
        Wn = np.zeros_like(Wp)
        ia, ib = a[0], a[1]
        sgn_a = np.sign(Wp[:, ia][np.abs(Wp[:, ia]) > 0])
        # rebuild from pw2 with the pair's fixed leg structure (+0.5/-0.5 x pw)
        Wn[:, ia] = 0.5 * pw2
        Wn[:, ib] = -0.5 * pw2
        out[name] = (Wn, gate, pw2)
    return out


def compose(pairs, b1, b2, b3):
    """Deterministic composition per prereg order: B3 rebuilds books; B2 scales by active
    fraction; B1 brakes the (B2-scaled) book."""
    books = pair_books(pairs)
    if b3:
        books = stop_reset(books)
    Wu = dp.gnorm(combine_rp(books))
    scale = np.ones(N)
    if b2:
        scale = scale * active_fraction(books)
    if b1:
        scale = scale * brake_scale(Wu, scale)
    return Wu, scale, books


def evaluate(tag, Wu, scale, full=True):
    net, turn, valid = net_scaled(Wu, scale)
    r = dp.summarize(net, turn, valid, FRIDAYS, N_EFF, tag, want_year=full)
    if full:
        for rt in rv.RT_SCEN:
            nn, _, _ = net_scaled(Wu, scale, rt)
            r[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
        for fb in (-rv.FUND_BP, rv.FUND_BP):
            nn, _, _ = net_scaled(Wu, scale, RT_BP, fb)
            r[f"sharpe_fund{int(fb):+d}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    return r, net


def main():
    res = {"prereg": "P5_PREREG.md", "panel": {"n_weeks": int(N),
           "span": f"{FRIDAYS[0].date()}..{FRIDAYS[-1].date()}"}}
    fam = {}

    # ---- S0 repro (deployed 5-pair, no overlay) ----
    Wu5, sc5, _ = compose(PAIRS5, False, False, False)
    s0, net0 = evaluate("S0_repro_5pair", Wu5, sc5)
    fam["5p_none"] = net0
    res["S0"] = s0
    print("S0 repro:", s0["sharpe_ann"], "dd", s0["maxdd_pct"], "(RV report: 0.732 / 18.5)", flush=True)

    # ---- new-pair solos (diagnostics, band ungated per RV per-pair convention) ----
    solo_books = pair_books(PAIRS_NEW, hl_gate=False)
    res["new_pair_solos"] = {}
    solo_dd_ok = True
    for name, (Wp, _g, _p) in solo_books.items():
        netp, turnp, validp = dp.net_of(Wp, WRET)
        mp = dp.summarize(netp, turnp, validp, FRIDAYS, N_EFF, name, want_year=False)
        res["new_pair_solos"][name] = mp
        if mp.get("maxdd_pct", 100.0) > 70.0:
            solo_dd_ok = False
        print(f"solo {name}: Sh={mp.get('sharpe_ann')} DD={mp.get('maxdd_pct')}", flush=True)

    # ---- U1 (8-pair) ----
    Wu8, sc8, _ = compose(PAIRS8, False, False, False)
    u1, net8 = evaluate("U1_8pair", Wu8, sc8)
    fam["8p_none"] = net8
    res["U1"] = u1
    accept_u1 = (u1["sharpe_ann"] > s0["sharpe_ann"] and u1.get("drop5_sharpe", -1) > 0
                 and solo_dd_ok)
    res["U1_accepted"] = bool(accept_u1)
    pairs_star = PAIRS8 if accept_u1 else PAIRS5
    base_star = u1 if accept_u1 else s0
    print(f"U1: Sh={u1['sharpe_ann']} dd={u1['maxdd_pct']} dr5={u1.get('drop5_sharpe')} "
          f"-> accepted={accept_u1}", flush=True)

    # ---- Step B waterfall (frozen order, one shot each, composed) ----
    accepted = {"b1": False, "b2": False, "b3": False}
    cur_metrics = base_star
    res["waterfall"] = {}
    for cand in ("b1", "b2", "b3"):
        trial = dict(accepted)
        trial[cand] = True
        Wu, sc, _ = compose(pairs_star, trial["b1"], trial["b2"], trial["b3"])
        mtr, nett = evaluate(f"WF_{cand}", Wu, sc)
        dd0, dd1 = cur_metrics["maxdd_pct"], mtr["maxdd_pct"]
        c0, c1 = cur_metrics["cagr_pct"], mtr["cagr_pct"]
        rel_dd = (dd0 - dd1) / dd0 if dd0 > 0 else 0.0
        rel_cagr = (c0 - c1) / abs(c0) if abs(c0) > 1e-9 else 0.0
        cp0 = cur_metrics.get("cpcv_frac_pos", 0.0)
        cp1 = mtr.get("cpcv_frac_pos", 0.0)
        ok = (rel_dd >= 0.15) and (rel_cagr <= 0.5 * rel_dd) and (cp1 >= cp0 - 0.05)
        res["waterfall"][cand] = {"metrics": mtr, "rel_dd_red": round(rel_dd, 3),
                                  "rel_cagr_red": round(rel_cagr, 3), "accepted": bool(ok)}
        print(f"WF {cand}: Sh={mtr['sharpe_ann']} dd={mtr['maxdd_pct']} cagr={mtr['cagr_pct']} "
              f"relDD={rel_dd:.2f} relCAGR={rel_cagr:.2f} cpcv={cp1} -> accepted={ok}", flush=True)
        if ok:
            accepted[cand] = True
            cur_metrics = mtr
    res["accepted"] = accepted

    # ---- final construction ----
    Wu_f, sc_f, _ = compose(pairs_star, accepted["b1"], accepted["b2"], accepted["b3"])
    fin, net_f = evaluate("FINAL", Wu_f, sc_f)
    res["final"] = fin
    fam["final"] = net_f
    print("FINAL:", {k: fin.get(k) for k in ("sharpe_ann", "cagr_pct", "maxdd_pct",
          "drop5_sharpe", "cpcv_frac_pos")}, "DSR=", fin["dsr"].get("dsr"), flush=True)

    # ---- Step C: sizing frontier (report-only) ----
    frontier = []
    for k in np.arange(0.5, 3.01, 0.25):
        nk, _, _ = net_scaled(Wu_f, sc_f * k)
        mk = sp.metrics(nk)
        frontier.append({"k": round(float(k), 2), "cagr_pct": mk.get("cagr_pct"),
                         "maxdd_pct": mk.get("maxdd_pct"),
                         "sharpe_ann": mk.get("sharpe_ann")})
    res["frontier"] = frontier
    ratio_pt = min(frontier, key=lambda x: abs((x["cagr_pct"] or 0) - (x["maxdd_pct"] or 100)))
    res["ratio_1to1_point"] = ratio_pt
    mu, var = float(np.nanmean(net_f)), float(np.nanvar(net_f))
    kelly = mu / var if var > 0 else 0.0
    res["kelly"] = {"full": round(kelly, 2), "half": round(kelly / 2, 2),
                    "quarter": round(kelly / 4, 2)}

    # ---- PBO family: {5p,8p} x {none,b1,b2,b3,acc,all3} ----
    for pl, pairs in (("5p", PAIRS5), ("8p", PAIRS8)):
        for cfg, flags in (("b1", (1, 0, 0)), ("b2", (0, 1, 0)), ("b3", (0, 0, 1)),
                           ("acc", tuple(int(accepted[x]) for x in ("b1", "b2", "b3"))),
                           ("all3", (1, 1, 1))):
            key = f"{pl}_{cfg}"
            if key in fam:
                continue
            Wu, sc, _ = compose(pairs, *[bool(x) for x in flags])
            nn, _, _ = net_scaled(Wu, sc)
            fam[key] = nn
    keys = sorted(fam)
    L = min(len(v) for v in fam.values())
    Mx = np.column_stack([fam[k][:L] for k in keys])
    res["pbo"] = {"configs": keys, "pbo_cscv": dp.pbo_cscv(Mx)}

    # ---- gates ----
    gates = {
        "sharpe_ge_0.75": fin["sharpe_ann"] >= 0.75,
        "cpcv": fin.get("cpcv_mean_path", 0) > 0 and fin.get("cpcv_frac_pos", 0) >= 0.90,
        "dsr_ge_0.9": (fin["dsr"].get("dsr") or 0) >= 0.90,
        "pbo_le_0.25": (res["pbo"]["pbo_cscv"] or 1.0) <= 0.25,
        "drop5_pos": fin.get("drop5_sharpe", -1) > 0,
        "rt24_ge_0.5": fin.get("sharpe_rt24", 0) >= 0.50,
        "ratio_ge_0.8": (ratio_pt["cagr_pct"] or 0) / max(ratio_pt["maxdd_pct"] or 100, 1e-9) >= 0.8
                        and (ratio_pt["maxdd_pct"] or 100) <= 25.0 * 1.2,
    }
    res["gates"] = {k: bool(v) for k, v in gates.items()}
    res["all_gates_pass"] = bool(all(gates.values()))

    (OUT / "results_p5.json").write_text(json.dumps(res, indent=1, default=str))
    print("GATES:", res["gates"], flush=True)
    print("FRONTIER 1:1 point:", ratio_pt, "kelly:", res["kelly"], flush=True)
    print("PBO:", res["pbo"]["pbo_cscv"], flush=True)
    print("P5 DONE ->", OUT / "results_p5.json", flush=True)


if __name__ == "__main__":
    main()
