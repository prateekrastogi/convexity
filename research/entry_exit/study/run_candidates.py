#!/usr/bin/env python3
"""Stage 3 — the CANDIDATE family, per COMMODITY_PREREG.md §3 as amended by ADDENDUM2.md.

Expanding walk-forward (5 folds, 10-day embargo): each entry-model grid point is FIT on the train
window, FROZEN, its F_{t-1} signal computed once per OOS day (signal-vector cache), then every
exit-variant x HAR toggle is simulated over those signals with the SAME 1-min stop engine and
costs as the baseline (ATR/HAR stop floor s=0.5, net 12bp RT). No train-side selection: every
declared grid point is a reported family member (N_eff charges all). JSONL checkpoints per
(symbol, fold, entry-config) -> resume-safe. Multiprocessing over symbols (2 workers).
-> results_candidates.jsonl (one record per symbol x fold x entry x exit x har)
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/speculation/convexity")
STUDY = REPO / "research/entry_exit/study"
sys.path.insert(0, str(REPO / "research"))

from entry_exit.base import ScanCursor, Side, TrainFold  # noqa: E402
from entry_exit.changepoint import BOCPDEntry  # noqa: E402
from entry_exit.hazard import HazardReversalExit  # noqa: E402
from entry_exit.har import HARRV  # noqa: E402
from entry_exit.hmm import GaussianHMMEntry  # noqa: E402
from entry_exit.jumps import LeeMyklandJumpEntry  # noqa: E402
from entry_exit.ou_exit import OUExit  # noqa: E402
from entry_exit.sequential import SprtGlrEntry  # noqa: E402
from entry_exit.stopping import OptimalStoppingExit  # noqa: E402

SYMS = ["XAUUSDT", "XAGUSDT", "CLUSDT", "NATGASUSDT", "COPPERUSDT", "XPTUSDT", "XPDUSDT"]
N_FOLDS = 5
EMBARGO = 10
STOP_S = 0.5
HAR_TR_CONST = 1.6
HOLD_FIXED = 5
HOLD_CAP = 10
RT_BP = 12.0
OUT = STUDY / "results_candidates.jsonl"


def make(cls, /, **want):
    """Instantiate cls with only the kwargs its signature accepts (name-drift tolerant)."""
    import inspect
    params = inspect.signature(cls).parameters
    kw = {k: v for k, v in want.items() if k in params}
    return cls(**kw), sorted(set(want) - set(kw))


ENTRY_GRID: list[tuple[str, object, dict]] = []
for lam in (100.0, 200.0):
    for pb in (0.5, 0.7):
        ENTRY_GRID.append((f"bocpd_l{int(lam)}_p{pb}", BOCPDEntry,
                           dict(hazard_lambda=lam, lam=lam, p_break=pb, threshold=pb)))
for (a, b) in ((0.05, 0.2), (0.01, 0.1)):
    for m in (20, 60):
        ENTRY_GRID.append((f"sprt_a{a}_m{m}", SprtGlrEntry,
                           dict(alpha=a, beta=b, m=m, window_m=m, window=m)))
for th in (0.6, 0.75):
    ENTRY_GRID.append((f"hmm_t{th}", GaussianHMMEntry, dict(threshold=th)))
for k in (16, 32):
    ENTRY_GRID.append((f"lm_k{k}", LeeMyklandJumpEntry,
                       dict(window=k, k_window=k, trailing_window=k, K=k, window_days=k)))

EXITS = [("fix5", None, {}), ("stopping", OptimalStoppingExit, dict(rho=0.02)),
         ("ou", OUExit, dict(r=0.05)),
         ("hz0.5", HazardReversalExit, dict(exit_hazard_quantile=0.5)),
         ("hz0.7", HazardReversalExit, dict(exit_hazard_quantile=0.7))]


def minute_index(sym: str) -> dict:
    df = pd.read_parquet(REPO / f"data/stitched/{sym}.parquet",
                         columns=["open_time", "open", "high", "low"])
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    d = ts.dt.date.to_numpy()
    change = np.nonzero(d[1:] != d[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(d)]])
    o, h, lo = (df[c].to_numpy() for c in ("open", "high", "low"))
    return {d[s]: (o[s:e], h[s:e], lo[s:e]) for s, e in zip(starts, ends)}


def stop_hit(bars, side, stop):
    o, h, lo = bars
    if side == 1:
        if o[0] <= stop:
            return o[0]
        if lo.min() <= stop:
            return stop
    else:
        if o[0] >= stop:
            return o[0]
        if h.max() >= stop:
            return stop
    return None


def run_symbol(sym: str) -> int:
    t_start = time.time()
    frame = pd.read_parquet(STUDY / f"frames/{sym}.parquet")
    frame = frame[np.isfinite(frame["ret_1d"])]
    days = frame.index
    dates = [d.date() for d in days]
    rets = tuple(float(x) for x in frame["ret_1d"].to_numpy())
    rv = tuple(float(x * x) for x in frame["ret_1d"].to_numpy())
    close = frame["day_close"].to_numpy()
    atr = frame["atr20"].to_numpy()
    minutes = minute_index(sym)
    n = len(frame)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["sym"], r["fold"], r["entry"], r["exit"], r["har"]))
            except Exception:
                pass

    n_written = 0
    for fold_i, test_lo in enumerate(np.linspace(0, n, N_FOLDS + 1).astype(int)[1:-1]):
        test_hi = np.linspace(0, n, N_FOLDS + 1).astype(int)[fold_i + 2]
        train_hi = max(test_lo - EMBARGO, 50)
        tf_ret = TrainFold(days=tuple(dates[:train_hi]), series=tuple(rets[:train_hi]))
        tf_rv = TrainFold(days=tuple(dates[:train_hi]), series=tuple(rv[:train_hi]))

        har = None
        try:
            har = HARRV().fit(tf_rv)
        except Exception as e:
            print(f"{sym} f{fold_i}: HAR fit failed {e!r}", flush=True)

        exits_fitted = {}
        for xname, xcls, xkw in EXITS:
            if xcls is None:
                exits_fitted[xname] = None
                continue
            try:
                x, _ = make(xcls, **xkw)
                exits_fitted[xname] = x.fit(tf_ret)
            except Exception as e:  # OUStationarityError et al.: record + skip this exit
                exits_fitted[xname] = ("FIT_FAIL", repr(e))

        for ename, ecls, ekw in ENTRY_GRID:
            if all((sym, fold_i, ename, xname, hb) in done
                   for xname, _, _ in EXITS for hb in (0, 1)):
                continue  # this (symbol, fold, entry) block fully checkpointed already
            try:
                ent, _dropped = make(ecls, **ekw)
                ent = ent.fit(tf_ret)
            except Exception as e:
                print(f"{sym} f{fold_i} {ename}: entry fit failed {e!r}", flush=True)
                continue
            # signal-vector cache: one causal signal per OOS day
            sig = np.zeros(test_hi - test_lo, dtype=int)
            for k, t in enumerate(range(test_lo, test_hi)):
                try:
                    s = ent.signal_at(ScanCursor(series=rets, t=t))
                    sig[k] = int(s)
                except Exception:
                    sig[k] = 0
            for xname, _, _ in EXITS:
                xm = exits_fitted[xname]
                if isinstance(xm, tuple):  # fit failed
                    continue
                for har_on in (0, 1):
                    key = (sym, fold_i, ename, xname, har_on)
                    if key in done:
                        continue
                    if har_on and har is None:
                        continue
                    trades = []
                    i = test_lo
                    while i < test_hi - 1:
                        s = int(sig[i - test_lo])
                        if s == 0:
                            i += 1
                            continue
                        entry = close[i]
                        if har_on:
                            try:
                                vf = float(har.forecast_vol_at(ScanCursor(series=rv, t=i)))
                                dist = STOP_S * HAR_TR_CONST * vf * entry
                            except Exception:
                                dist = STOP_S * atr[i]
                        else:
                            dist = STOP_S * atr[i]
                        if not np.isfinite(dist) or dist <= 0:
                            i += 1
                            continue
                        stop = entry - s * dist
                        cap = min(i + (HOLD_FIXED if xm is None else HOLD_CAP), test_hi - 1)
                        exit_ret, exit_j = None, cap
                        j = i + 1
                        while j <= cap:
                            b = minutes.get(dates[j])
                            if b is not None:
                                px = stop_hit(b, s, stop)
                                if px is not None:
                                    exit_ret, exit_j = s * (px / entry - 1.0), j
                                    break
                            held_ret = s * (close[j] / entry - 1.0)
                            if xm is not None:
                                try:
                                    if xm.exit_now(ScanCursor(series=rets, t=j), j - i, held_ret):
                                        exit_ret, exit_j = held_ret, j
                                        break
                                except Exception:
                                    pass
                            j += 1
                        if exit_ret is None:
                            exit_ret = s * (close[cap] / entry - 1.0)
                        trades.append(float(exit_ret))
                        i = exit_j + 1
                    net = np.array(trades) - RT_BP / 1e4 if trades else np.array([])
                    rec = {"sym": sym, "fold": fold_i, "entry": ename, "exit": xname,
                           "har": har_on, "n": int(net.size),
                           "ev_bp": round(float(net.mean()) * 1e4, 1) if net.size else None,
                           "sum_ret": round(float(net.sum()), 6) if net.size else 0.0,
                           "wins": int((net > 0).sum()) if net.size else 0,
                           "trades": [round(float(x), 6) for x in net]}
                    with OUT.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    n_written += 1
            print(f"{sym} f{fold_i} {ename} done ({time.time()-t_start:.0f}s elapsed)", flush=True)
    return n_written


def main() -> None:
    with Pool(2) as p:
        counts = p.map(run_symbol, SYMS)
    print(f"CANDIDATES DONE: {sum(counts)} records", flush=True)


if __name__ == "__main__":
    main()
