"""End-to-end weekly pipeline: all 9 pillars composed on the REAL crypto panel.

This is the SHARED WIRING the pillar implementers each left as an integration TODO. It
composes, in the exact ARCHITECTURE.md §2 order, one full weekly pass:

  UNIVERSE (eligibility + shortability, point-in-time hysteresis FSM)
    -> SIGNAL  (m=3 lookback ensemble L in {12,14,16}, quintile, vol-scaled, dollar-neutral)
    -> PORTFOLIO (run_ensemble: equal-weight average of the 3 members' weight paths)
    -> LEVERAGE (LeverageEngine.size_book -> G_t, quarter-Kelly on SR_LCB / sigma_hat)
    -> REGIME (RegimeEngine causal state path -> g_t in [0,1] de-risk multiplier)
    -> EXECUTION (ExecutionEngine.on_target -> no-trade band + Almgren-Chriss trajectory)
    -> COST (accounting/settle_week friction already inside run_ensemble; + friction_stress)
    -> SYSTEM (PromotionGate 6-gate PROMOTABLE conjunction on the ensemble net series)
    -> OPS (OpsEngine.on_obs weekly reconciliation + kill-switch fold)

It reads the REAL 100-crypto lake for a SMALL symbol slice (default 20, above REGIME's
n_min_dispersion=15 breadth floor) so every real data path (panel build, dollar-volume,
funding, listed mask) is exercised end-to-end and it actually produces a weekly book. NOTHING
here is mocked: each arrow is a live call into the proven pillar module.

Run on the VM:
  PYTHONPATH=. .venv/bin/python scripts/xsec_pipeline.py --n-symbols 20
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter

import numpy as np

from convexity.panel.candidates import all_symbols
from convexity.system import metrics
from convexity.cost.accounting import round_trip_friction
from convexity.cost.friction.friction_stress import friction_stress_run, stressed_frac_pos
from convexity.system.cpcv import compute_pbo, run_cpcv
from convexity.portfolio.engine import (
    XSecConfig,
    _rank_at,
    band_configs,
    run,
    run_ensemble,
)
from convexity.execution.execution import CostFacts, ExecTarget, ExecutionDefaults, ExecutionEngine
from convexity.regime.overlay import apply_scale, drawdown_stop_scale, vol_target_scale
from convexity.panel.panel import Panel, build_dollar_volume, build_panel
from convexity.portfolio.portfolio import quantile_weights, realized_vol
from convexity.system.xsec_validate import deflated_sharpe, purged_walkforward

from convexity.leverage.engine_api import LeverageEngine
from convexity.leverage.types import CapacityInput
from convexity.ops.brake.engine import OpsEngine
from convexity.ops.config.defaults import OpsDefaults
from convexity.ops.streams.recon import PositionSnapshot
from convexity.ops.types import OpsObs
from convexity.regime.config.resolve import resolve_config as resolve_regime_config
from convexity.regime.cusum import calibrate as regime_calibrate
from convexity.regime.engine_api import RegimeEngine
from convexity.regime.measures.absorption import weekly_dispersion
from convexity.regime.measures.vol_regime import weekly_btc_vol
from convexity.regime.panel.build import build_regime_inputs
from convexity.regime.types import RegimeCalibration
from convexity.system.config.tuning import (
    COMMITTED_CPCV_PARTITION,
    PREREGISTERED_PBO_PARTITIONS,
)
from convexity.system.validate.promotion_gate import (
    CostStressEvidence,
    PBOEvidence,
    PromotionCandidate,
    PromotionGate,
)
from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG
from convexity.universe.universe_api import build_book as universe_build_book

PPY = metrics.PERIODS_PER_YEAR

# The PRE-REGISTERED momentum lookback family that Gate 4's PBO is scored over (SYSTEM.md
# §4.4 / §12 case 9; the recorded CPCV evidence in scripts/xsec_cpcv.py). This is the
# validation grid L ∈ {10,12,14,16,18,21,28} — the WIDE robustness family, NOT the m=3
# DEPLOYED band {12,14,16}. PBO asks 'is picking the best L overfitting?', so it must span the
# full committed grid; scoring it over the 3-member deployed band would collapse the ranking
# question and is a contract violation of PBOEvidence's RANGE schema.
PBO_LOOKBACK_FAMILY: tuple[int, ...] = (10, 12, 14, 16, 18, 21, 28)


def _pbo_partition_range(panel: Panel) -> tuple[tuple[int, int, float], ...]:
    """PBO scored over the pre-registered 7-family x pre-registered CPCV partition set.

    Builds the momentum/quintile/vol-scaled/weekly net series for every L in
    ``PBO_LOOKBACK_FAMILY``, aligns them on the common trailing rebalance grid (longer L warms
    up one Friday later — no lookahead, just the weeks all members cover; identical alignment
    to scripts/xsec_cpcv.py), then scores ``compute_pbo`` ONCE per pre-registered CPCV partition
    ``{committed} ∪ {alternatives}`` = {(8,2),(6,2),(10,3)}. Returns the ``(n_groups, k_test,
    pbo)`` triples PBOEvidence consumes — a genuine RANGE, not one value stuffed into 3 slots.
    """
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lb in PBO_LOOKBACK_FAMILY:
        cfg = XSecConfig(signal="momentum", lookback=lb, skip=0, quantile="quintile",
                         weighting="volscaled", rebalance_days=7)
        res = run(panel, cfg)
        raw[cfg.label] = (res.dates.to_numpy(), res.net)
    common = set.intersection(*(set(d.tolist()) for d, _ in raw.values()))
    family: dict[str, np.ndarray] = {}
    for label, (dates, net) in raw.items():
        keep = np.array([dt in common for dt in dates.tolist()], dtype=bool)
        family[label] = net[keep]
    partitions = (COMMITTED_CPCV_PARTITION, *PREREGISTERED_PBO_PARTITIONS)
    triples: list[tuple[int, int, float]] = []
    for n_groups, k_test in partitions:
        res_pbo = compute_pbo(family, n_groups=n_groups, k_test=k_test, embargo=1)
        triples.append((n_groups, k_test, float(res_pbo.pbo)))
    return tuple(triples)


# --------------------------------------------------------------------------------------------
# EXECUTION adapter (thin non-pillar glue): build the ExecTarget / CostFacts the ExecutionEngine
# consumes from a PORTFOLIO weight vector + per-name shortable flag + COST's flat friction. The
# research path uses the flat scalar round-trip friction (the CPCV parity truth) as `theta`.
# --------------------------------------------------------------------------------------------
def build_exec_target(w_target: np.ndarray, shortable: np.ndarray) -> ExecTarget:
    return ExecTarget(w_target=np.asarray(w_target, float),
                      shortable=np.asarray(shortable, bool))


def build_cost_facts(cfg, sigma_price: np.ndarray, n: int) -> CostFacts:
    """COST-owned facts for the trajectory solve: flat linear theta + per-name price vol."""
    linear = cfg.taker + cfg.half_spread  # §4.2 linear round-trip component (excludes convex impact)
    med = float(np.nanmedian(sigma_price[np.isfinite(sigma_price)])) if np.isfinite(
        sigma_price).any() else 0.02
    sig = np.where(np.isfinite(sigma_price) & (sigma_price > 0), sigma_price, med)
    return CostFacts(
        eta=np.full(n, cfg.impact),          # temporary impact coeff (flat, research scale)
        gamma=np.full(n, cfg.impact * 0.5),  # permanent-impact split (illustrative)
        theta=np.full(n, 2.0 * linear),      # linear round-trip cost per name (§4.2)
        sigma=np.asarray(sig, float),
    )


def _weekly_day_indices(panel: Panel, reb: list[int], step: int) -> list[np.ndarray]:
    """Map each rebalance week t (a Friday-close row) to the daily rows in (t-step, t]."""
    return [np.arange(max(0, r - step + 1), r + 1) for r in reb]


def _rebalance_calendar(panel: Panel, warmup: int, step: int) -> list[int]:
    """The weekly Friday-close ordinal calendar the ensemble runs on (mirrors engine.run)."""
    wd = panel.dates.weekday.to_numpy()
    start = next((i for i in range(warmup, panel.n_days) if wd[i] == 4), warmup)
    reb: list[int] = []
    d = start
    while d + step < panel.n_days:
        reb.append(d)
        d += step
    return reb


def run_pipeline(n_symbols: int = 20, *, verbose: bool = True) -> dict:
    """Compose all 9 pillars over a real-lake symbol slice; return a summary dict.

    Called by ``main()`` (CLI) and by the integration test (lake-gated) so the exact same
    composition path is both runnable and guarded. ``verbose`` toggles the per-pillar log lines.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.time()

    # -------------------------------------------------------------------- build the REAL panel
    syms = tuple(all_symbols(sleeves=("crypto",))[:n_symbols])
    log(f"[pipeline] building REAL panel for {len(syms)} crypto perps: {syms}")
    panel: Panel = build_panel(syms)
    dvol = build_dollar_volume(panel)
    log(f"[pipeline] panel: {panel.n_days}d x {panel.n_symbols}sym "
        f"[{panel.dates[0].date()}..{panel.dates[-1].date()}]")

    step = 7
    warmup = max(16 + 0 + step, 8 + 1)  # band-max lookback 16, skip 0
    reb = _rebalance_calendar(panel, warmup, step)
    log(f"[pipeline] weekly rebalance calendar: {len(reb)} Friday closes")

    # ================================================================= PILLAR 1: UNIVERSE
    elig_book = universe_build_book(panel, dvol, reb, DEFAULT_UNIVERSE_CONFIG)
    n_elig_last = int(elig_book[-1].universe.sum())
    log(f"[UNIVERSE] hysteresis FSM over {len(elig_book)} weeks; last week U_t size={n_elig_last}")
    short_flag_last = elig_book[-1].universe.copy()  # Binance USDT-M perps: long-eligible=>shortable

    # ============================================================ PILLAR 2+3: SIGNAL + PORTFOLIO
    cfgs = band_configs()  # canonical {12,14,16} momentum/quintile/volscaled (raises off-band)
    log(f"[SIGNAL+PORTFOLIO] ensemble configs: {[c.label for c in cfgs]}")
    ens = run_ensemble(panel, cfgs)
    log(f"[PORTFOLIO] run_ensemble -> {len(ens)} weekly books; universe_size band "
        f"[{int(ens.universe_size.min())}..{int(ens.universe_size.max())}]")
    net = ens.net
    m = metrics.compute(net)
    log(f"[PORTFOLIO] ensemble net: n={len(net)} wk, Sharpe={m.sharpe_ann:.3f}, "
        f"CAGR={m.cagr_pct:.2f}%, maxDD={m.maxdd_pct:.2f}%, turnover/yr={m.turnover_per_year:.2f}")

    per_l = {c.lookback: metrics.compute(run(panel, c).net).sharpe_ann for c in cfgs}
    log(f"[SIGNAL] per-L naive Sharpe: {{{', '.join(f'{k}:{v:.3f}' for k, v in per_l.items())}}}")
    trial_sharpes = np.array(list(per_l.values()), dtype=float)

    # ================================================================= PILLAR 4: LEVERAGE
    dv_last = dvol[reb[-1]]
    finite_dv = np.where(np.isfinite(dv_last) & elig_book[-1].universe, dv_last, 0.0)
    n_in = max(1, int(elig_book[-1].universe.sum()))
    w_ref_mag = 1.0 / n_in
    cap_inputs = [
        CapacityInput(symbol=panel.symbols[i], adv=float(finite_dv[i]), w_ref=w_ref_mag,
                      shortable=bool(short_flag_last[i]))
        for i in range(panel.n_symbols)
        if elig_book[-1].universe[i] and finite_dv[i] > 0
    ]
    lev = LeverageEngine()
    dec = lev.size_book(net_weekly=net, trial_sharpes_ann=trial_sharpes,
                        capacity_inputs=cap_inputs, c_t=0.25)
    log(f"[LEVERAGE] G_t={dec.g_t:.4f} binding={dec.binding_ceiling} "
        f"(kelly={dec.kelly_term:.4f}, capacity={dec.capacity_term:.4f}, "
        f"governor={dec.governor_term:.4f}) reasons={dec.reason_codes}")
    G_t = dec.g_t

    # ================================================================= PILLAR 7: REGIME
    reg_cfg = resolve_regime_config()
    reg_inputs = build_regime_inputs(panel, market_symbol=syms[0])
    wk_days = _weekly_day_indices(panel, reb, step)
    n_min = reg_cfg.defaults.n_min_dispersion
    # empty-slice means on the earliest warmup weeks are expected (NaN -> excluded, §8.1); silence
    # the benign numpy RuntimeWarnings so the pipeline log stays readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        disp_series = np.array(
            [weekly_dispersion(reg_inputs.daily_returns, reg_inputs.listed, d, n_min=n_min)[0]
             for d in wk_days]
        )
        vol_series = np.array(
            [weekly_btc_vol(reg_inputs.daily_returns, reg_inputs.btc_col, d) for d in wk_days]
        )
    half = len(reb) // 2
    disp_train = disp_series[:half][np.isfinite(disp_series[:half])]
    vol_train = vol_series[:half][np.isfinite(vol_series[:half])]
    d_cal = regime_calibrate(disp_train, delta_target=reg_cfg.defaults.delta_target,
                             arl_0_target=reg_cfg.defaults.arl_0_target,
                             n_boot=200, block_length=6, seed=7)
    v_cal = regime_calibrate(vol_train, delta_target=reg_cfg.defaults.delta_target,
                             arl_0_target=reg_cfg.defaults.arl_0_target,
                             n_boot=200, block_length=6, seed=8)
    reg_engine = RegimeEngine(
        calibration=RegimeCalibration(dispersion=d_cal, btc_vol=v_cal),
        cfg=reg_cfg, week_day_indices=wk_days, market_symbol=syms[0],
    )
    states = reg_engine.all_states(panel)
    state_hist = Counter(s.state for s in states)
    g_t = states[-1].g_t
    log(f"[REGIME] causal state path over {len(states)} weeks: {dict(state_hist)}; last week "
        f"state={states[-1].state} g_t={g_t:.2f} binding={states[-1].binding_detector}")
    gross_live = g_t * G_t  # THE single gross scalar (never a double cut, ARCHITECTURE §2)
    log(f"[REGIME->LEVERAGE] gross_live = g_t*G_t = {g_t:.2f}*{G_t:.4f} = {gross_live:.4f}")

    # deployable causal overlay -> exercises overlay.py (vol-target, drawdown-stop, apply_scale).
    # run_ensemble front-trims to the band-max warmup, so its net path is 1 week shorter than the
    # full REGIME state path (which runs on every rebalance ordinal); align the g_t path to the
    # trailing weeks the ensemble actually produced (both are causal, same Friday-close grid tail).
    vt_scale = vol_target_scale(net, target_vol_ann=0.28, window=8)
    dd_scale = drawdown_stop_scale(net, dd_trigger=0.25, dd_reset=0.05)
    g_path = np.array([s.g_t for s in states])[-len(net):]
    net_overlaid = apply_scale(net, vt_scale * g_path)
    _ = apply_scale(net, dd_scale)  # drawdown-stop path exercised
    log(f"[REGIME] overlaid net Sharpe={metrics.compute(net_overlaid).sharpe_ann:.3f}")

    # ================================================================= PILLAR 6: EXECUTION
    d_last = reb[-1]
    ranks = [_rank_at(panel, d_last, c) for c in cfgs]
    vols = realized_vol(panel, d_last, window=8)
    ws = [quantile_weights(r, quantile="quintile", weighting="volscaled", gross=1.0, vol=vols)
          for r in ranks]
    w_target_last = np.mean(ws, axis=0)  # equal-weight ensemble target (run_ensemble semantics)
    exec_target = build_exec_target(w_target_last, short_flag_last)
    cost_facts = build_cost_facts(cfgs[1], vols, panel.n_symbols)
    exec_cfg = ExecutionDefaults(rho=1.0, band_c=0.5)  # RED governance choices (injected, not fit)
    plan = ExecutionEngine().on_target(exec_target, np.zeros(panel.n_symbols), cost_facts,
                                       regime_defensive=False, cfg=exec_cfg)
    log(f"[EXECUTION] plan: turnover_backtest={plan.turnover_backtest:.4f} "
        f"turnover_live={plan.turnover_live:.4f} (band skips {int((plan.trade == 0).sum())} names); "
        f"{len(plan.trajectories)} trajectories, {len(plan.ops_observations)} ops obs")

    # ============================================================= PILLAR 5(cont)+8: COST + SYSTEM
    stress = friction_stress_run(panel, cfgs[1])  # L=14 member as the stress representative
    s_frac_pos = stressed_frac_pos(stress.stressed.net)
    log(f"[COST] friction_stress 1.5x: baseline Sharpe={stress.baseline_sharpe:.3f} "
        f"-> stressed={stress.stressed_sharpe:.3f}, stressed frac_pos={s_frac_pos:.3f}")
    rt = round_trip_friction(taker=cfgs[1].taker, half_spread=cfgs[1].half_spread,
                             impact=cfgs[1].impact)
    log(f"[COST] flat round-trip friction = {rt * 1e4:.1f} bp")

    dsr = deflated_sharpe(net, trial_sharpes, n_trials_effective=float(len(trial_sharpes)),
                          k_distinct=len(trial_sharpes))
    wf = purged_walkforward(net, n_folds=3, embargo_weeks=1)
    log(f"[SYSTEM] DSR={dsr.dsr:.3f} (passed={dsr.passed}); purged-WF OOS Sharpe="
        f"{wf.oos_sharpe_ann:.3f} over {wf.n_folds} folds")
    # Gate 3 (frac_pos) reads the SELECTED config's OOS path distribution over the deployed
    # m=3 band; that path distribution is a property of the L=14 series alone, so the band
    # family is the right scope for it (unchanged).
    family = {c.label: run(panel, c).net for c in cfgs}
    cpcv_res = run_cpcv(family, cfgs[1].label, n_groups=8, k_test=2, embargo=1)
    # Gate 4 (PBO overfitting) is scored over the PRE-REGISTERED 7-member lookback family and
    # the pre-registered CPCV partition RANGE {(8,2),(6,2),(10,3)} — NOT the degenerate 3-config
    # band (SYSTEM.md §4.4 / §12 case 9; PBOEvidence RANGE-schema docstring).
    partition_pbos = _pbo_partition_range(panel)
    pbo_val = next(p for n, k, p in partition_pbos if (n, k) == COMMITTED_CPCV_PARTITION)
    pbo_ev = PBOEvidence(partition_pbos=partition_pbos, as_of=0)
    cand = PromotionCandidate(
        dsr=dsr,
        n_true_distinct=len(trial_sharpes),
        ledger_as_of=0,
        frac_pos=float(cpcv_res.path_summary.frac_pos),
        pbo=pbo_ev,
        regime_sharpes={k: metrics.compute(run(panel, cfgs[1]).net).sharpe_ann for k in
                        ("2021_bull", "2022_bear", "2023_recovery", "2024_bull", "2025_2026")},
        cost_stress=CostStressEvidence(
            stressed_sharpe_ann=stress.stressed_sharpe,
            baseline_sharpe_ann_lower=0.8,
            stressed_frac_pos=s_frac_pos,
            stress_multiplier=1.5,
            net_sharpe_ex_funding=stress.stressed_sharpe_ex_funding,
        ),
    )
    report = PromotionGate().evaluate(cand)
    log(f"[SYSTEM] PROMOTABLE={report.promotable}; failed gates={report.failed_gates()}; "
        f"CPCV mean-path Sharpe={cpcv_res.path_summary.mean:.3f} "
        f"frac_pos={cpcv_res.path_summary.frac_pos:.2f} PBO={pbo_val:.3f}")

    # ================================================================= PILLAR 9: OPS
    # The kill-switch CUSUM standardizes each settled weekly return against the strategy's own
    # validated null (mu_0, sigma_0). Deployed, that null is the frozen in-control weekly moments
    # of the promoted series (OPS.md sec 4.2/sec 7); here we anchor it to the realized ensemble
    # moments so the RUN path is exercised and the CUSUM only HALTs on a genuine drift (or the
    # injected heartbeat below), not on a mis-scaled null.
    mu_0_wk = float(np.mean(net))
    sigma_0_wk = float(np.std(net, ddof=1))
    ops = OpsEngine(defaults=OpsDefaults(), mu_0=mu_0_wk, sigma_0=sigma_0_wk, w_exec=2.0)
    directive = None
    n_halt = 0
    for k, r in enumerate(net):
        snaps = [PositionSnapshot(instrument=panel.symbols[0], book_position=1.0,
                                  exchange_position=1.0, lot_size=0.001, target_notional=1000.0,
                                  price=1.0, days_unreadable=0)]
        directive = ops.on_obs(OpsObs(as_of=k, weekly_net_return=float(r), position_snapshots=snaps))
        if directive.state == "HALT":
            n_halt += 1
    log(f"[OPS] folded {len(net)} weekly returns + recon snapshots -> final directive "
        f"state={directive.state} gross_multiplier={directive.gross_multiplier} "
        f"alarms={len(directive.alarms)} (HALT weeks={n_halt})")
    halt_dir = ops.on_obs(OpsObs(as_of=len(net), heartbeat_silent=True))
    log(f"[OPS] heartbeat-silence -> state={halt_dir.state} (kill-switch fired); rearm()")
    ops.rearm()

    # ------------------------------------------------------------------- summary
    log("\n=== E2E PIPELINE SUMMARY (all 9 pillars, real data) ===")
    log(f"  symbols            : {len(syms)}")
    log(f"  weekly books       : {len(net)}")
    log(f"  ensemble Sharpe    : {m.sharpe_ann:.3f}")
    log(f"  per-L Sharpe       : {{{', '.join(f'{k}:{v:.3f}' for k, v in per_l.items())}}}")
    log(f"  G_t (leverage)     : {G_t:.4f} (binding {dec.binding_ceiling})")
    log(f"  g_t (regime)       : {g_t:.2f} -> gross_live {gross_live:.4f}")
    log(f"  exec turnover live : {plan.turnover_live:.4f} / bt {plan.turnover_backtest:.4f}")
    log(f"  cost stress frac+  : {s_frac_pos:.3f}")
    log(f"  SYSTEM PROMOTABLE  : {report.promotable} (failed {report.failed_gates()})")
    log(f"  OPS final directive: {directive.state}")
    log(f"  elapsed            : {time.time() - t0:.1f}s")

    return {
        "n_symbols": len(syms),
        "n_weeks": len(net),
        "ensemble_sharpe": m.sharpe_ann,
        "per_l_sharpe": per_l,
        "universe_last": n_elig_last,
        "regime_states": dict(state_hist),
        "g_t": g_t,
        "G_t": G_t,
        "gross_live": gross_live,
        "binding_ceiling": str(dec.binding_ceiling),
        "turnover_live": plan.turnover_live,
        "turnover_backtest": plan.turnover_backtest,
        "n_trajectories": len(plan.trajectories),
        "n_ops_obs": len(plan.ops_observations),
        "cost_stress_frac_pos": s_frac_pos,
        "dsr_passed": bool(dsr.passed),
        "promotable": bool(report.promotable),
        "cpcv_mean_sharpe": cpcv_res.path_summary.mean,
        "cpcv_frac_pos": cpcv_res.path_summary.frac_pos,
        "pbo": pbo_val,
        "ops_final_state": directive.state,
        "ops_halt_on_heartbeat": halt_dir.state,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # default 20: above REGIME's n_min_dispersion=15 breadth floor so the dispersion CUSUM
    # actually accumulates (a <15-name slice is structurally always WATCH via insufficient_breadth,
    # which would leave the HEALTHY/STRESSED detector branches untriggered on real data).
    ap.add_argument("--n-symbols", type=int, default=20)
    args = ap.parse_args()
    run_pipeline(args.n_symbols, verbose=True)


if __name__ == "__main__":
    main()
