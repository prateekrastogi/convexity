#!/usr/bin/env python
"""P7 — INSTITUTIONAL RISK STACK FOR THE RV BOOK (P7_PREREG.md, frozen 0792dd9).

Composition order (deterministic, independent of acceptance sequence):
  K1 Kalman spread -> pair pw (frozen P4 z/HL machinery) -> K7 per-pair liquidity de-gross ->
  K2 innovation-CUSUM de-arm -> combine (pair RP) -> K6 PCA-1 neutralize -> gnorm ->
  sizing scales in order K4 (ES), K5 (ensemble vol), K3 (HMM), each computed causally on the
  book-so-far. Costs always on the actually-traded scaled weights (equity-book convention)."""
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
N_EFF = 50
KAL_DELTA, KAL_LAM = 1e-4, 0.98
CUSUM_K, CUSUM_H, CUSUM_COOL = 0.5, 5.0, 8
HMM_P, HMM_SCALE, HMM_REFIT = 0.7, 0.5, 52
ES_Q, ES_CLIP = 0.10, (0.5, 2.0)
VT_STAR = 0.10
PCA_WIN = 104
AMI_WIN_D, AMI_PCT, AMI_SCALE, AMI_BURN = 65, 80, 0.5, 104
PAIRS = list(rv.PAIRS)

curve = sp.load_curve_raw()
cot = pd.read_parquet("/home/speculation/convexity/data/cot/cot_disagg.parquet")
P = sp.build_panel(curve, cot)
FRIDAYS, WRET = P["fridays"], P["wret"]
N, M = WRET.shape
g = np.log1p(np.nan_to_num(WRET, nan=0.0))
LP = np.zeros((N, M))
LP[1:] = np.cumsum(g, axis=0)[:-1]
WRET0 = np.nan_to_num(WRET)


# ------------------------------------------------ K1: Kalman spread (else beta=1 level spread)
def kalman_spread(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One-step-ahead innovation of a_t = beta_t*b_t + alpha_t (random-walk state, delta=1e-4,
    obs noise EW lambda=0.98). The innovation IS the spread (prior state at t = causal)."""
    n = len(a)
    x = np.array([1.0, 0.0])                    # [beta, alpha]
    Pm = np.eye(2) * 1e-2
    Q = np.eye(2) * (KAL_DELTA / (1 - KAL_DELTA))
    R = 1e-3
    s = np.full(n, np.nan)
    for t in range(n):
        if not (np.isfinite(a[t]) and np.isfinite(b[t])):
            continue
        H = np.array([b[t], 1.0])
        Pm = Pm + Q
        e = a[t] - H @ x                        # innovation with PRIOR state -> causal spread
        s[t] = e
        S = H @ Pm @ H + R
        K = (Pm @ H) / S
        x = x + K * e
        Pm = Pm - np.outer(K, H) @ Pm
        R = KAL_LAM * R + (1 - KAL_LAM) * (e * e)
    return s


def spread_of(A, B, kalman: bool):
    a, b = rv.IDX[A], rv.IDX[B]
    if kalman:
        return kalman_spread(LP[:, a], LP[:, b]), a, b
    return LP[:, a] - LP[:, b], a, b


def base_pairs(kalman: bool):
    out = {}
    for A, B in PAIRS:
        s, a, b = spread_of(A, B, kalman)
        with np.errstate(all="ignore"):
            z = np.nanmean(np.column_stack([rv.roll_z(s, w) for w in rv.ZWINS]), axis=1)
        hl = rv.half_life(s)
        gate = np.isfinite(hl) & (hl >= rv.HL_LO) & (hl <= rv.HL_HI)
        pw = np.nan_to_num(np.where(gate, -np.clip(z, -rv.ZCLIP, rv.ZCLIP) / rv.ZCLIP, 0.0))
        out[f"{A}-{B}"] = {"a": a, "b": b, "s": s, "z": z, "hl": hl, "gate": gate, "pw": pw}
    return out


# ------------------------------------------------ K2: CUSUM on AR(1) innovations
def roll_ar1(s: np.ndarray, W: int = 104):
    """Trailing AR(1) ds = alpha + phi*s_lag via rolling cov/var (causal shift 1)."""
    ds = pd.Series(np.diff(s, prepend=np.nan))
    lag = pd.Series(np.roll(s, 1))
    lag.iloc[0] = np.nan
    cov = ds.rolling(W, min_periods=W).cov(lag)
    var = lag.rolling(W, min_periods=W).var()
    phi = (cov / var).shift(1)
    alpha = (ds.rolling(W, min_periods=W).mean() - phi * lag.rolling(W, min_periods=W).mean()).shift(1)
    return alpha.to_numpy(), phi.to_numpy(), ds.to_numpy(), lag.to_numpy()


def k2_cusum(pr):
    s, pw = pr["s"], pr["pw"].copy()
    alpha, phi, ds, lag = roll_ar1(s)
    e = ds - (alpha + phi * lag)
    sig = pd.Series(e).rolling(26, min_periods=26).std().shift(1).to_numpy()
    gp = gm = 0.0
    cool = 0
    for t in range(N):
        if cool > 0:
            pw[t] = 0.0
            cool -= 1
            continue
        if np.isfinite(e[t]) and np.isfinite(sig[t]) and sig[t] > 0:
            u = e[t] / sig[t]
            gp = max(0.0, gp + u - CUSUM_K)
            gm = max(0.0, gm - u - CUSUM_K)
            if max(gp, gm) > CUSUM_H:
                gp = gm = 0.0
                cool = CUSUM_COOL
                pw[t] = 0.0
    return pw


# ------------------------------------------------ K3: 2-state Gaussian HMM gate (own EM)
def hmm_gate(net_full: np.ndarray):
    """Filtered P(high-vol) with expanding refits every 52wk; scale week t uses net through
    t-1 only (net[t] is the t->t+1 outcome)."""
    r = np.nan_to_num(net_full)
    scale = np.ones(N)

    def fit(seg):
        med = np.median(np.abs(seg))
        st = seg[np.abs(seg) <= 2 * med + 1e-12]
        mu = np.array([seg.mean(), seg.mean()])
        sd = np.array([max(st.std(), 1e-5), max(seg.std() * 2.5, 2e-5)])
        A = np.array([[0.95, 0.05], [0.10, 0.90]])
        pi = np.array([0.8, 0.2])
        for _ in range(40):
            B = np.column_stack([_norm(seg, mu[k], sd[k]) for k in (0, 1)])
            al = np.zeros((len(seg), 2))
            al[0] = pi * B[0]
            al[0] /= al[0].sum() + 1e-300
            for t in range(1, len(seg)):
                al[t] = (al[t - 1] @ A) * B[t]
                al[t] /= al[t].sum() + 1e-300
            be = np.ones((len(seg), 2))
            for t in range(len(seg) - 2, -1, -1):
                be[t] = A @ (B[t + 1] * be[t + 1])
                be[t] /= be[t].sum() + 1e-300
            gam = al * be
            gam /= gam.sum(1, keepdims=True) + 1e-300
            xi_num = np.zeros((2, 2))
            for t in range(len(seg) - 1):
                x = np.outer(al[t], B[t + 1] * be[t + 1]) * A
                xi_num += x / (x.sum() + 1e-300)
            A = xi_num / (xi_num.sum(1, keepdims=True) + 1e-300)
            for k in (0, 1):
                w = gam[:, k]
                mu[k] = (w * seg).sum() / (w.sum() + 1e-300)
                sd[k] = max(np.sqrt((w * (seg - mu[k]) ** 2).sum() / (w.sum() + 1e-300)), 1e-6)
            pi = gam[0]
        return mu, sd, A, pi

    params = None
    hi = 1
    p_filt = 0.0
    for t in range(N):
        if t >= 156 and (t - 156) % HMM_REFIT == 0:
            seg = r[:t]
            params = fit(seg)
            hi = int(np.argmax(params[1]))
            # re-run filter up to t-1 with new params
            p = params[3].copy()
            for u in range(t):
                B = np.array([_norm1(r[u], params[0][k], params[1][k]) for k in (0, 1)])
                p = (p @ params[2]) * B
                p /= p.sum() + 1e-300
            p_state = p
        if params is None:
            continue
        p_filt = p_state[hi]
        if p_filt > HMM_P:
            scale[t] = HMM_SCALE
        B = np.array([_norm1(r[t], params[0][k], params[1][k]) for k in (0, 1)])
        p_state = (p_state @ params[2]) * B
        p_state /= p_state.sum() + 1e-300
    return scale


def _norm(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + 1e-300


def _norm1(x, mu, sd):
    return float(_norm(np.array([x]), mu, sd)[0])


# ------------------------------------------------ K4 / K5: sizing scales on book net
def es_scale(net_full: np.ndarray):
    s = pd.Series(np.nan_to_num(net_full))
    es = s.rolling(52).apply(lambda w: -np.mean(np.sort(w)[: max(1, int(len(w) * ES_Q))]),
                             raw=True).shift(1)
    scale = np.ones(N)
    hist = []
    for t in range(N):
        if np.isfinite(es.iloc[t]):
            hist.append(es.iloc[t])
            if len(hist) >= AMI_BURN:
                star = float(np.median(hist))
                if es.iloc[t] > 0:
                    scale[t] = float(np.clip(star / es.iloc[t], *ES_CLIP))
    return scale


def ens_vol_scale(net_full: np.ndarray):
    s = pd.Series(np.nan_to_num(net_full))
    sd26 = s.rolling(26).std()
    ewm = s.ewm(alpha=1 - 0.94).std()
    sig = ((sd26 + ewm) / 2).shift(1) * np.sqrt(WK)
    scale = np.clip(VT_STAR / sig, 0.5, 2.0).fillna(1.0).to_numpy()
    return scale


# ------------------------------------------------ K6: PCA-1 neutralization
def pca_neutralize(Wc: np.ndarray):
    out = Wc.copy()
    R = pd.DataFrame(WRET0)
    for t in range(PCA_WIN, N):
        seg = R.iloc[t - PCA_WIN:t].to_numpy()
        C = np.cov(seg.T)
        vals, vecs = np.linalg.eigh(C)
        v = vecs[:, -1]
        out[t] = out[t] - (out[t] @ v) * v
    return out


# ------------------------------------------------ K7: Amihud liquidity de-gross (daily volume)
def amihud_flags():
    RAW = Path("/home/speculation/convexity/data/curves/databento")
    flags = {}
    for root in sp.ROOTS:
        f = RAW / f"{root}_c0_ohlcv1d.csv"
        d = pd.read_csv(f, usecols=["ts_event", "close", "volume"])
        d["date"] = pd.to_datetime(d["ts_event"], format="mixed", utc=True).dt.tz_localize(None)
        d = d.set_index("date").sort_index()
        dv = d["close"] * d["volume"]
        illiq = (d["close"].pct_change().abs() / dv.replace(0, np.nan)).rolling(
            AMI_WIN_D, min_periods=30).mean()
        wk = illiq.resample("W-FRI").last().reindex(pd.DatetimeIndex(FRIDAYS)).shift(1)
        flag = np.zeros(N, bool)
        hist = []
        for t in range(N):
            v = wk.iloc[t]
            if np.isfinite(v):
                hist.append(v)
                if len(hist) >= AMI_BURN and v > np.percentile(hist, AMI_PCT):
                    flag[t] = True
        flags[root] = flag
    return flags


AMI = None


def k7_scale_pair(pr, name):
    global AMI
    if AMI is None:
        AMI = amihud_flags()
    A, B = name.split("-")
    bad = AMI[A] | AMI[B]
    return np.where(bad, AMI_SCALE, 1.0)


# ------------------------------------------------ compose / net / evaluate
def net_scaled(Wu, scale, rt=RT_BP, fund=0.0):
    return sp.net_series(Wu * scale[:, None], WRET, rt, fund)


def full_net(Wu, scale):
    """Full-length (N) net for scale estimators (zeros where invalid)."""
    W = Wu * scale[:, None]
    grossr = (W * WRET0).sum(axis=1)
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, M))))
    return grossr - (RT_BP / 1e4) * dW.sum(axis=1)


def compose(flags):
    prs = base_pairs(kalman=("k1" in flags))
    for name, pr in prs.items():
        if "k7" in flags:
            pr["pw"] = pr["pw"] * k7_scale_pair(pr, name)
        if "k2" in flags:
            pr["pw"] = k2_cusum(pr)
    books = {}
    for name, pr in prs.items():
        Wp = np.zeros((N, M))
        Wp[:, pr["a"]] = 0.5 * pr["pw"]
        Wp[:, pr["b"]] = -0.5 * pr["pw"]
        books[name] = Wp
    comb = rv.pair_risk_parity(books, WRET)
    if "k6" in flags:
        comb = pca_neutralize(comb)
    Wu = dp.gnorm(comb)
    scale = np.ones(N)
    for cand, fn in (("k4", es_scale), ("k5", ens_vol_scale), ("k3", hmm_gate)):
        if cand in flags:
            scale = scale * fn(full_net(Wu, scale))
    return Wu, scale


def evaluate(tag, Wu, scale, full=True):
    net, turn, valid = net_scaled(Wu, scale)
    r = dp.summarize(net, turn, valid, FRIDAYS, N_EFF, tag, want_year=False)
    if full:
        for rt in rv.RT_SCEN:
            nn, _, _ = net_scaled(Wu, scale, rt)
            r[f"sharpe_rt{int(rt)}"] = round(float(sp.per_obs_sharpe(nn) * sqrt(WK)), 3)
    return r, net


def main():
    res = {"prereg": "P7_PREREG.md"}
    fam = {}
    Wu0, sc0 = compose(set())
    s0, net0 = evaluate("S0", Wu0, sc0)
    fam["none"] = net0
    res["S0"] = s0
    print(f"S0 repro: {s0['sharpe_ann']} dd {s0['maxdd_pct']} (expect 0.732/18.5)", flush=True)

    accepted: set = set()
    cur = s0
    res["waterfall"] = {}
    for cand in ("k1", "k2", "k3", "k4", "k5", "k6", "k7"):
        Wu, sc = compose(accepted | {cand})
        mtr, nett = evaluate(f"WF_{cand}", Wu, sc)
        fam[cand] = nett
        dd0, dd1 = cur["maxdd_pct"], mtr["maxdd_pct"]
        c0, c1 = cur["cagr_pct"], mtr["cagr_pct"]
        sh0, sh1 = cur["sharpe_ann"], mtr["sharpe_ann"]
        rel_dd = (dd0 - dd1) / dd0 if dd0 > 0 else 0.0
        rel_cagr = (c0 - c1) / abs(c0) if abs(c0) > 1e-9 else 0.0
        cp0, cp1 = cur.get("cpcv_frac_pos", 0.0), mtr.get("cpcv_frac_pos", 0.0)
        rule_dd = (rel_dd >= 0.15) and (rel_cagr <= 0.5 * rel_dd) and (cp1 >= cp0 - 0.05)
        rule_sh = (sh1 - sh0 >= 0.05) and (dd1 <= dd0)
        ok = rule_dd or rule_sh
        res["waterfall"][cand] = {"metrics": mtr, "rel_dd_red": round(rel_dd, 3),
                                  "rel_cagr_red": round(rel_cagr, 3),
                                  "via": "dd" if rule_dd else ("sharpe" if rule_sh else None),
                                  "accepted": bool(ok)}
        print(f"WF {cand}: Sh={sh1} dd={dd1} cagr={c1} relDD={rel_dd:.2f} "
              f"relCAGR={rel_cagr:.2f} cpcv={cp1} -> accepted={ok}"
              + (f" (via {'dd' if rule_dd else 'sharpe'})" if ok else ""), flush=True)
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

    (OUT / "results_p7.json").write_text(json.dumps(res, indent=1, default=str))
    print("GATES:", res["gates"], flush=True)
    print("at DD~25 sizing:", res["at_dd25"], flush=True)
    print("PBO:", res["pbo"]["pbo_cscv"], flush=True)
    print("P7 DONE ->", OUT / "results_p7.json", flush=True)


if __name__ == "__main__":
    main()
