"""Robustness study R1-R3 (docs/ROBUST.md, pre-registered): rebalance-phase ensemble, per-name
real cost, embargo-3 re-baseline. Checkpointed per stage to _ckpt/robust/results.json.

Run:  PYTHONPATH=. .venv/bin/python scripts/xsec_robust.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from convexity.cost.friction.spread_impact import SpreadImpactModel, amihud_ratio
from convexity.panel.candidates import all_symbols
from convexity.panel.panel import Panel, build_dollar_volume, build_panel
from convexity.portfolio.engine import XSecConfig, band_configs, run, run_book, run_ensemble
from convexity.cost.accounting import WeekResult, settle_week
from convexity.system import metrics
from convexity.system.cpcv import compute_pbo, run_cpcv

CKPT = Path("_ckpt/robust/results.json")
NAV = 3_000_000.0  # prereg R2: LEVERAGE's capacity-dial anchor, the repo's one stated book size
WINDOW = 30  # prereg R2: panel.py's own causal ADV/Amihud window (one window, no second length)
LOOKBACKS7 = (10, 12, 14, 16, 18, 21, 28)  # the committed CPCV family (R3)


class _Path:
    """Averaged band-ensemble TARGET weight path on its common grid (mirrors run_ensemble's
    E5/E8 alignment; settling this path with settle_week reproduces run_ensemble exactly —
    verified inline below before R2 uses it)."""

    def __init__(self, panel, cfgs):
        members = [run_book(panel, c) for c in cfgs]
        d_start = max(int(b.d_index[0]) for _r, b in members)
        offs = [int(np.searchsorted(b.d_index, d_start)) for _r, b in members]
        ref = members[0][1].d_index[offs[0]:]
        for (off, (_r, b)) in zip(offs, members, strict=True):
            assert np.array_equal(b.d_index[off:], ref), "phase members misaligned (E8)"
        self.w_rows = np.mean(np.stack(
            [b.weights[off:] for off, (_r, b) in zip(offs, members, strict=True)], axis=0),
            axis=0)
        self.ref_rows = np.asarray(ref, dtype=int)
        self.step = cfgs[0].rebalance_days


def _phase_cfgs(weekday: int) -> list[XSecConfig]:
    return [XSecConfig(signal="momentum", lookback=lb, quantile="quintile",
                       weighting="volscaled", phase_weekday=weekday) for lb in (12, 14, 16)]


def _mets(net, turnover=None, cost=None, funding=None) -> dict:
    m = metrics.compute(np.asarray(net), turnover=turnover, cost=cost, funding=funding)
    return {"n_weeks": int(m.n_weeks), "sharpe_ann": round(m.sharpe_ann, 4),
            "cagr_pct": round(m.cagr_pct, 2), "maxdd_pct": round(m.maxdd_pct, 2)}


def _load() -> dict:
    return json.loads(CKPT.read_text()) if CKPT.exists() else {}


def _save(state: dict) -> None:
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps(state, indent=1))


def main() -> None:
    t0 = time.time()
    state = _load()
    print(f"[robust] checkpoint stages present: {sorted(state)}", flush=True)

    syms = tuple(all_symbols(sleeves=("crypto",)))
    panel: Panel = build_panel(syms)
    print(f"[robust] panel {panel.n_days}d x {panel.n_symbols}sym ({time.time()-t0:.0f}s)",
          flush=True)

    # ================= R1: rebalance-phase ensemble =================
    if "R1" not in state:
        phase_nets: dict[int, list[float]] = {}
        phase_stats: dict[str, dict] = {}
        for wd in range(7):
            res = run_ensemble(panel, _phase_cfgs(wd))
            phase_nets[wd] = [round(float(x), 10) for x in res.net]
            phase_stats[str(wd)] = _mets(res.net, res.turnover, res.cost, res.funding)
            print(f"[robust] R1 phase {wd}: Sh={phase_stats[str(wd)]['sharpe_ann']:+.3f} "
                  f"n={phase_stats[str(wd)]['n_weeks']} ({time.time()-t0:.0f}s)", flush=True)
        n_min = min(len(v) for v in phase_nets.values())
        aligned = np.stack([np.asarray(v[-n_min:]) for v in phase_nets.values()])  # (7, n_min)
        ens_net = aligned.mean(axis=0)
        sharpes = [phase_stats[str(w)]["sharpe_ann"] for w in range(7)]
        state["R1"] = {
            "per_phase": phase_stats,
            "rtl_sharpe_std": round(float(np.std(sharpes, ddof=1)), 4),
            "rtl_sharpe_range": [min(sharpes), max(sharpes)],
            "friday_sharpe": phase_stats["4"]["sharpe_ann"],
            "ensemble": _mets(ens_net),
            "ensemble_net": [round(float(x), 10) for x in ens_net],
            "friday_net": phase_nets[4],
        }
        _save(state)
        r1 = state["R1"]
        print(f"[robust] R1 RTL: phase Sharpes {sharpes} std={r1['rtl_sharpe_std']} | "
              f"ensemble Sh={r1['ensemble']['sharpe_ann']:+.3f}", flush=True)

    # R1 CPCV: phase-mean vs Friday book, committed (8,2) machinery
    if "R1_cpcv" not in state:
        r1 = state["R1"]
        n_min = len(r1["ensemble_net"])
        fam = {"friday": np.asarray(r1["friday_net"][-n_min:]),
               "phase_ensemble": np.asarray(r1["ensemble_net"])}
        out = {}
        for sel in fam:
            rep = run_cpcv(fam, sel, n_groups=8, k_test=2, embargo=1, seed=0)
            ps = rep.path_summary
            out[sel] = {"cpcv_mean": round(ps.mean, 4), "frac_pos": round(ps.frac_pos, 4),
                        "p05": round(ps.p05, 4), "naive": round(ps.naive_full_sharpe, 4)}
            print(f"[robust] R1 CPCV {sel}: mean={ps.mean:+.3f} frac_pos={ps.frac_pos:.2f}",
                  flush=True)
        state["R1_cpcv"] = out
        _save(state)

    # ================= R2: per-name real cost on the Friday band ensemble =================
    if "R2" not in state:
        dv = build_dollar_volume(panel)  # (days, syms), NaN off-span
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = panel.close[1:] / panel.close[:-1] - 1.0
        rets = np.vstack([np.full((1, panel.n_symbols), np.nan), rets])  # align rows to panel days
        path = _Path(panel, band_configs())
        # harness parity: settling the untransformed path with the FLAT scalar must reproduce
        # run_ensemble bit-for-bit before any per-name cost is trusted.
        from convexity.cost.accounting import round_trip_friction
        _flat_fric = round_trip_friction()
        _chk_net, _wp = [], np.zeros(panel.n_symbols)
        for _j in range(path.w_rows.shape[0]):
            _d = int(path.ref_rows[_j])
            _r, _ = settle_week(panel.close, panel.funding, panel.listed, d0=_d,
                                d1=_d + path.step, w_target=path.w_rows[_j], w_prev=_wp,
                                friction=_flat_fric)
            _chk_net.append(_r.net)
            _wp = path.w_rows[_j]
        _ens_chk = run_ensemble(panel, band_configs())
        assert np.array_equal(np.asarray(_chk_net), _ens_chk.net), "R2 harness parity violated"
        print("[robust] R2 harness parity OK", flush=True)

        # anchor at the FULL-SAMPLE median ADV/Amihud (the model's committed one-point anchoring)
        adv_full = np.nanmean(np.where(dv > 0, dv, np.nan), axis=0)
        ami_full = amihud_ratio(np.where(np.isfinite(rets), rets, np.nan), dv)
        model = SpreadImpactModel.anchor(
            float(np.nanmedian(adv_full[adv_full > 0])),
            float(np.nanmedian(ami_full[ami_full > 0])),
        )
        print(f"[robust] R2 model anchored: c={model.c:.4g} k={model.k_amihud:.4g}", flush=True)

        net, trn, cst, fnd = [], [], [], []
        w_prev = np.zeros(panel.n_symbols)
        for j in range(path.w_rows.shape[0]):
            d = int(path.ref_rows[j])
            lo = max(0, d - WINDOW + 1)
            adv_t = np.nanmean(np.where(dv[lo:d + 1] > 0, dv[lo:d + 1], np.nan), axis=0)
            ami_t = amihud_ratio(rets[lo:d + 1], dv[lo:d + 1])
            w_t = path.w_rows[j]
            # per-name one-way trade notional (prereg: target-to-target diff x NAV; drift term
            # omitted from the impact SIZE argument only -- the friction is still CHARGED on the
            # exact drift-aware dw by settle_week itself)
            dw = np.abs(w_t - w_prev)
            # friction only where a trade occurs (dw>0): elsewhere 0 (cost contribution is 0
            # anyway; avoids inf*0=NaN from the model's conservative ADV<=0 -> inf half-spread).
            # A TRADED name with no window ADV falls back to the week's median ADV (the model's
            # own median-anchor philosophy); missing Amihud already falls back to 2x anchor
            # inside impact_frac.
            fric = np.zeros(panel.n_symbols)
            med_adv = float(np.nanmedian(adv_t[np.isfinite(adv_t) & (adv_t > 0)])) if np.any(
                np.isfinite(adv_t) & (adv_t > 0)) else model.anchor_notional
            for i in np.flatnonzero(dw > 0):
                adv_i = float(adv_t[i]) if np.isfinite(adv_t[i]) and adv_t[i] > 0 else med_adv
                notional = max(float(dw[i]) * NAV, 1.0)
                fric[i] = model.round_trip_friction_i(
                    adv_i,
                    float(ami_t[i]) if np.isfinite(ami_t[i]) else float("nan"),
                    notional_leg=notional,
                )
            res_j: WeekResult
            res_j, _ = settle_week(panel.close, panel.funding, panel.listed,
                                   d0=d, d1=d + path.step, w_target=w_t, w_prev=w_prev,
                                   friction=fric)
            net.append(res_j.net)
            trn.append(res_j.turnover)
            cst.append(res_j.cost)
            fnd.append(res_j.funding)
            w_prev = w_t
        flat = run_ensemble(panel, band_configs())
        state["R2"] = {
            "flat_cost": _mets(flat.net, flat.turnover, flat.cost, flat.funding),
            "real_cost": _mets(np.asarray(net), np.asarray(trn), np.asarray(cst),
                               np.asarray(fnd)),
            "real_net": [round(float(x), 10) for x in net],
            "cost_ann_flat_pct": round(float(np.mean(flat.cost)) * 52.18 * 100, 3),
            "cost_ann_real_pct": round(float(np.mean(cst)) * 52.18 * 100, 3),
            "nav_prereg": NAV,
        }
        _save(state)
        r2 = state["R2"]
        print(f"[robust] R2: flat Sh={r2['flat_cost']['sharpe_ann']:+.3f} "
              f"(cost {r2['cost_ann_flat_pct']}%/yr) -> real Sh="
              f"{r2['real_cost']['sharpe_ann']:+.3f} (cost {r2['cost_ann_real_pct']}%/yr)",
              flush=True)

    if "R2_cpcv" not in state:
        r2 = state["R2"]
        fri = np.asarray(state["R1"]["friday_net"])
        real = np.asarray(r2["real_net"])
        n_min = min(len(fri), len(real))
        fam = {"flat_cost": fri[-n_min:], "real_cost": real[-n_min:]}
        rep = run_cpcv(fam, "real_cost", n_groups=8, k_test=2, embargo=1, seed=0)
        ps = rep.path_summary
        state["R2_cpcv"] = {"cpcv_mean": round(ps.mean, 4), "frac_pos": round(ps.frac_pos, 4),
                            "naive": round(ps.naive_full_sharpe, 4)}
        _save(state)
        print(f"[robust] R2 CPCV real-cost: mean={ps.mean:+.3f} frac_pos={ps.frac_pos:.2f}",
              flush=True)

    # ================= R3: embargo-3 re-baseline of the committed 7-family =================
    if "R3" not in state:
        raw: dict[str, tuple] = {}
        for lb in LOOKBACKS7:
            cfg = XSecConfig(signal="momentum", lookback=lb, quantile="quintile",
                             weighting="volscaled")
            res = run(panel, cfg)
            raw[cfg.label] = (res.dates.to_numpy(), res.net)
        common = set.intersection(*(set(d.tolist()) for d, _ in raw.values()))
        fam = {}
        for lab, (dates, n) in raw.items():
            keep = np.array([dt in common for dt in dates.tolist()], dtype=bool)
            fam[lab] = n[keep]
        sel = "mom-14d-skip0/quintile/volscaled"
        out: dict = {}
        for emb in (1, 3):
            rep = run_cpcv(fam, sel, n_groups=8, k_test=2, embargo=emb, seed=0)
            ps = rep.path_summary
            pbos = {}
            for n_g, k_t in ((8, 2), (6, 2), (10, 3)):
                pbos[f"({n_g},{k_t})"] = round(
                    compute_pbo(fam, n_groups=n_g, k_test=k_t, embargo=emb).pbo, 4)
            out[f"embargo_{emb}"] = {
                "cpcv_mean": round(ps.mean, 4), "frac_pos": round(ps.frac_pos, 4),
                "naive": round(ps.naive_full_sharpe, 4), "haircut": round(ps.haircut, 4),
                "pbo": pbos}
            print(f"[robust] R3 embargo={emb}: mean={ps.mean:+.3f} frac_pos={ps.frac_pos:.2f} "
                  f"PBO={pbos}", flush=True)
        state["R3"] = out
        _save(state)

    state["runtime_s"] = round(time.time() - t0, 1)
    _save(state)
    print(f"[robust] DONE in {state['runtime_s']}s -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
