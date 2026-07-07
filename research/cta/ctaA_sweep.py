"""
research/cta/ctaA_sweep.py -- Stage-A Integration/Sweep.

*** REDESIGNED GRID (research/cta/CTA_MATH_NOTES.md Section 9-REDESIGN) ***
The original frozen 18,848-cell-per-universe design (Section 9, preserved verbatim in the math
notes for the record) was found computationally intractable on this 2-vCPU VM: a 3-name smoke
test showed ~220s of per-name Stage-A signal-generation cost alone, implying ~10-12 HOURS
wall-clock for the full 93-name x 18,848-cell x 3-universe run. Direct profiling (not just the
prior agent's per-component estimates) traced ~84% of that per-name cost to Axis A's causal
rolling l1-trend-filter (a genuinely daily ADMM re-solve, NOT Axis C's HMM or Axis D's GARCH,
both of which were already confirmed to use a correctly periodic, cheap causal refit schedule --
see Section 9-REDESIGN for the full investigation writeup). This file now implements the
redesigned, genuinely-tractable grid:
  - Universe: a stratified 22-name Stage-A pilot subset (`STAGE_A_UNIVERSE_22`) of the full
    93-name base universe, not all 93 -- Stage B's later frozen confirmatory run uses the full 93.
  - Axis A x B x F core: UNCHANGED at 31 x 4 x 8 = 992 (these are not the bottleneck).
  - Axis C (regime): trimmed from 5 to 3 points (drop both 3-state HMM variants).
  - Axis D (vol forecast): trimmed from 10 to 3 points, using Axis D's own QLIKE/MSE ranking.
  - Axis E (leverage): unchanged at 6 (cheap, book-level scalar calc, not a per-name cost driver).
  - A legitimate performance fix (ADMM warm-starting, NOT a math/scope change) was also applied to
    `research/cta/lib/signal_extraction.py`'s `l1_trend_filter_rolling`, verified to preserve the
    module's own causality injection test and produce numerically near-identical output.
  - A SECOND performance fix, found live during the feasibility launch (Section 9-REDESIGN.5c):
    `aggregate_portfolio()`'s per-day loop dominates real cost far more than originally estimated;
    `_compute_f_variant_weights()` cuts 8 naive calls/OAT-slot down to 3 (1 for single-name books),
    verified byte-identical (max abs diff = 0.0) to the naive computation on real data.
  - New OAT block width = 1 + (3-1) + (3-1) + (6-1) = 10 (down from 19).
  - New N_cells = 992 x 10 = 9,920 per universe x 3 universes (22-name book, SPY-alone, QQQ-alone)
    = 29,760 total rows.
See CTA_MATH_NOTES.md Section 9-REDESIGN for the full, explicit disclosure of what was cut and why.

--- Original module docstring (grid section only) follows, PRESERVED for historical continuity ---
Runs the FROZEN 18,848-cell grid (research/cta/CTA_MATH_NOTES.md Section 9) across the 93-name
reconciled base universe, plus SPY and QQQ individually (already members of the 93, run again as
single-instrument books for direct comparability with the closed I1 program). [SUPERSEDED -- see
above; the 93-name universe is now the 22-name STAGE_A_UNIVERSE_22 subset for this Stage-A pilot.]

PIPELINE (CTA_MATH_NOTES.md Section 8.1, with the three required fixes from Section 8.2-8.4):
    raw_dir_i(t)      = Axis A multi_horizon_trend_signal(...)['combined']   (already .shift(1)'d)
    regime_gate_i(t)  = Axis C exposure_scale_from_posterior(p_bull) in [0,1]
    vol_scale_i(t)    = Axis D vol_target_exposure_magnitude(forecast_vol, ...)
    gated_target_i(t) = clip(raw_dir_i(t) * regime_gate_i(t), 0, None) * vol_scale_i(t)   [fix #1/#2]
    exposure_i(t)     = Axis B partial_adjustment_control(gated_target_i, price_i,
                                                            external_vol=forecast_vol_i, ...)      [fix #3]
    weights(t)        = Axis F aggregate_portfolio(exposure, returns, rv_lagged=forecast_vol, ...)
    book_return_1x(t) = sum_i weights_i(t)*returns_i(t) - trading_cost_i(t) - funding_cost_i(t)   (per-name
                        costs: METHODOLOGY -- 13bp/6.5bp/26bp round-trip primary on |delta-weight|,
                        funding +-5bp/week PRORATED by actual elapsed calendar days, scaled by |weight|)
    final_net_return(t) = L(t) * book_return_1x(t)     (Axis E leverage overlay)

FROZEN GRID (CTA_MATH_NOTES.md Section 9): core cells = Axis A (31) x Axis B (4) x Axis F (8) = 992,
crossed with a replicated one-at-a-time (OAT) sweep of Axis C x Axis D x Axis E anchored at each
axis's own disclosed baseline (C=2-state HMM refit=63, D=ewma_0.94, E=Grossman-Zhou loose
alpha=0.50), re-run fresh at every one of the 992 core cells. OAT block width = 1 (all baseline)
+ (5-1) + (10-1) + (6-1) = 19. N_cells = 992*19 = 18,848. Run identically for the 93-name portfolio
book AND for SPY-alone and QQQ-alone (single-instrument "books"), for a declared multiplier of 3
instrument-universes => 18,848*3 = 56,544 total result rows (verified against the actual output
row count at the end of this script -- see also the printed reconciliation).

PERFORMANCE-ENGINEERING NOTE (does not change any axis's math, only avoids REDUNDANT repeated
calls to the SAME deterministic functions with the SAME inputs):
  1. Per-name Axis A/C/D outputs are computed ONCE per name per grid point (46 total per-name
     variants) and cached -- exactly the "fixed cost, independent of N_cells" the frozen design
     calls for.
  2. Axis F's `rolling_avg_pairwise_correlation` / `correlation_spike_scale` / `rolling_shrunk_
     covariance` depend ONLY on the (fixed, sweep-wide) realized-returns panel and `corr_window`
     (2 values: 60, 120) -- NOT on the per-cell target_exposure. Calling `aggregate_portfolio()`
     naively 13,888 times would recompute the IDENTICAL covariance/correlation time series
     thousands of times over. These three functions are wrapped in a memoizing cache (keyed on
     the arguments that actually vary) and monkey-patched onto the `portfolio_construction`
     module BEFORE any sweep cell runs, so `aggregate_portfolio()` transparently calls the cached
     wrapper instead -- byte-identical output, no math changed, verified by the smoke test at the
     bottom of this file (`_smoke_test_cache_equivalence`).
  3. Axis B's `partial_adjustment_control` does not depend on Axis F's grid point at all, so the
     per-(Axis-A, Axis-B, C/D-OAT-slot) exposure path is computed ONCE and reused across all 8
     Axis-F points for that (A,B) pair, rather than being recomputed once per (A,B,F) core cell
     (an 8x reduction in Axis-B call count, from ~1.29M calls down to ~161K).
  4. Phase 1 (per-name signal caching) and Phase 2 (the (A,B)-pair sweep loop) are each
     parallelized across the machine's CPUs via `multiprocessing.Pool` (fork start method so
     child workers inherit the parent's already-warmed caches/dataframes via copy-on-write,
     without re-pickling large objects per task).

Checkpointing: results are flushed to `results_ctaA_full_grid.parquet` every CHECKPOINT_EVERY
(A,B)-pairs (93-name book) / every checkpoint interval (SPY/QQQ), so a partial run always leaves
an honest, inspectable partial artifact on disk (row count printed every checkpoint).
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os

# PERFORMANCE FIX (research/cta/CTA_MATH_NOTES.md Section 9-REDESIGN): pin every BLAS/OMP thread
# pool to 1 thread BEFORE numpy/scipy are ever imported. This script parallelizes across NAMES/
# (A,B)-pairs via multiprocessing.Pool (process-level parallelism); left at their multi-threaded
# defaults, numpy/scipy's own internal BLAS thread pools would additionally try to use every
# available core INSIDE each worker process, oversubscribing this VM's 2 vCPUs (2 pool workers x
# 2+ BLAS threads each) and measurably slowing every worker down rather than speeding anything up
# -- confirmed empirically: a 2-worker phase-1 smoke test ran EACH name slower (150-200s) than a
# single undisturbed process (~110-130s with the Section 9-REDESIGN.2 L1TF warm-start fix already
# applied), the classic oversubscription signature. This changes no math, only thread scheduling.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
LIB_DIR = f"{REPO_ROOT}/research/cta/lib"
sys.path.insert(0, LIB_DIR)
sys.path.insert(0, REPO_ROOT)

import signal_extraction as se          # Axis A
import entry_exit_control as eec        # Axis B
import regime_hmm as rh                 # Axis C
import vol_forecast as vf               # Axis D
import leverage_policy as lp            # Axis E
import portfolio_construction as pc     # Axis F

from convexity.panel.candidates import load_universe

STITCHED_DIR = Path(REPO_ROOT) / "data" / "stitched"
OUT_DIR = Path(REPO_ROOT) / "research" / "cta"
RESULTS_PATH = OUT_DIR / "results_ctaA_full_grid.parquet"
CHECKPOINT_PATH = OUT_DIR / "_ctaA_sweep_checkpoint.parquet"
PROGRESS_LOG = OUT_DIR / "_ctaA_sweep_progress.log"

BENCHMARK_CAGR = 0.468  # Established Fact (6): 93-name equal-weight buy&hold, frozen reference
BENCHMARK_CAGR_22NAME: float | None = None  # derived fresh in main() for the Stage-A 22-name subset

# 6 excluded leveraged/inverse/decay ETFs (Established Bug Pattern 7e)
EXCLUDE_LEVERAGED = {"UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT"}
# 3 named short-history exclusions (DATA CONTRACT)
EXCLUDE_SHORT_HISTORY = {"STXXUSDT", "CBRSUSDT", "DRAMUSDT"}
MIN_DUKA_SPAN_DAYS = 100.0  # threshold cleanly separating the 3 short-history names (<=47d) from
                            # the rest of the usable universe, incl. recently-listed-but-real names
                            # like CRCLUSDT (~249d span, real 2025 IPO history -- included, per
                            # point-in-time discipline it simply enters the book late); next-lowest
                            # span after CRCLUSDT is ~479d (MVLLUSDT)

N_PROCESSES = max(1, mp.cpu_count())

# ======================================================================================
# Regime sub-windows for per-cell reporting (dates chosen to match established program convention,
# e.g. Axis F's own COVID window definition in CTA_MATH_NOTES.md F.3)
# ======================================================================================
REGIME_WINDOWS = {
    "ret_2018q4": ("2018-10-01", "2018-12-31"),
    "ret_2020_03_covid": ("2020-02-15", "2020-04-15"),
    "ret_2022_full": ("2022-01-01", "2022-12-31"),
    "ret_2023_24_meltup": ("2023-01-01", "2024-12-31"),
    "ret_2025_04_squeeze": ("2025-04-04", "2025-04-11"),
}

DD_BUDGETS = (0.20, 0.30, 0.41)

# ======================================================================================
# Axis A grid (31 points, CTA_MATH_NOTES.md Section A.6 exact enumeration)
# ======================================================================================
EWMA_TRIPLES = [(8.0, 32.0, 128.0), (5.0, 20.0, 80.0), (10.0, 50.0, 200.0)]
EWMA_SHRINKAGES = [0.0, 0.5, 1.0]
KALMAN_TRIPLES = [(1e-6, 1e-8, 1e-4), (1e-5, 1e-7, 1e-4), (1e-4, 1e-6, 1e-4)]
L1TF_WINDOWS = [60, 120, 252]
L1TF_LAM_MULTS = [0.5, 1.0, 2.0]
FULL_SHRINKAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
FULL_IC_WINDOWS = [126, 252]

AXIS_A_KEYS: list[str] = []
for trip in EWMA_TRIPLES:
    for shr in EWMA_SHRINKAGES:
        AXIS_A_KEYS.append(f"ewma_{trip[0]:g}_{trip[1]:g}_{trip[2]:g}_shr{shr}")
for trip in KALMAN_TRIPLES:
    AXIS_A_KEYS.append(f"kalman_{trip[0]:g}_{trip[1]:g}_{trip[2]:g}")
for w in L1TF_WINDOWS:
    for lm in L1TF_LAM_MULTS:
        AXIS_A_KEYS.append(f"l1tf_w{w}_lm{lm}")
for shr in FULL_SHRINKAGES:
    for icw in FULL_IC_WINDOWS:
        AXIS_A_KEYS.append(f"full_shr{shr}_ic{icw}")
assert len(AXIS_A_KEYS) == 31, len(AXIS_A_KEYS)

DEFAULT_KALMAN_KEY = "kalman_1e-05_1e-07_0.0001"  # == DEFAULT_KALMAN_PARAMS, reused from components
DEFAULT_L1TF_KEY = "l1tf_w120_lm1.0"              # == DEFAULT_L1TF_PARAMS, reused from components

# ======================================================================================
# Axis B grid (4 points) -- imported directly from the module (do not reimplement)
# ======================================================================================
AXIS_B_GRID = eec.DEFAULT_GRID
AXIS_B_KEYS = [g["name"] for g in AXIS_B_GRID]

# ======================================================================================
# Axis C grid -- REDESIGNED: trimmed from the original 5 to 3 points (CTA_MATH_NOTES.md Section
# 9-REDESIGN). Both 3-state HMM variants are dropped: (a) C.6 already disclosed that 3-state fits
# are "at the edge of reliable estimation for short-history names" and should be "restricted to
# long-history names" -- several names in the new 22-name Stage-A pilot universe are deliberately
# SHORT-history (chosen for diversity, e.g. BMNRUSDT/CRCLUSDT), so 3-state would silently fall
# back to 2-state for a material fraction of this specific universe anyway, and (b) 3-state is
# also the single most expensive Axis-C sub-variant (O(K^2) EM cost, ~10.1s vs 2-state's ~7.75s on
# SPY at refit=63, confirmed by direct profiling). The two retained HMM points plus the SMA
# baseline still span the full fast/slow refit-cadence tradeoff C.5 was chartered to test.
# ======================================================================================
AXIS_C_GRID = [
    {"key": "hmm2_rf63", "n_states": 2, "refit_every": 63},
    {"key": "hmm2_rf252", "n_states": 2, "refit_every": 252},
    {"key": "sma200", "type": "sma"},
]
AXIS_C_BASELINE = "hmm2_rf63"
MIN_OBS_FOR_3STATE = 750  # retained for forward-compat if 3-state is ever re-added; unused by the
                          # trimmed grid above (no n_states==3 grid point remains in Stage-A).

# ======================================================================================
# Axis D grid -- REDESIGNED: trimmed from the original 10 to 3 points (CTA_MATH_NOTES.md Section
# 9-REDESIGN), using Axis D's OWN internal QLIKE/MSE ranking (D.8): ewma_0.94 is the outright
# aggregate QLIKE winner (baseline) and ewma_0.97 the outright aggregate MSE winner -- both kept.
# The third point deliberately deviates from pure-aggregate-ranking (which would pick
# trailing_rv_60): D.8 also explicitly documents GARCH(1,1) as the single BEST estimator
# specifically on SPY and QQQ (broad, mean-reverting-vol index names) -- and this exact sweep
# tests SPY/QQQ as two of its three "universes", so dropping GARCH entirely would blind the sweep
# to a documented, axis-flagged interaction on precisely the names most likely to show it. GARCH
# is also computationally cheap (~0.4-1.8s/name, confirmed negligible by direct profiling), so
# this substitution costs nothing and is disclosed here as a deliberate, evidence-based deviation
# from the naive top-3-by-aggregate-QLIKE cut.
# ======================================================================================
AXIS_D_GRID = [
    {"key": "ewma_0.94", "kind": "ewma", "lam": 0.94},
    {"key": "ewma_0.97", "kind": "ewma", "lam": 0.97},
    {"key": "garch11_rf21_w756", "kind": "garch", "refit_freq": 21, "window": 756},
]
AXIS_D_BASELINE = "ewma_0.94"

# ======================================================================================
# Axis E grid (6 points)
# ======================================================================================
AXIS_E_KEYS = ["half_kelly", "quarter_kelly", "gz_tight", "gz_loose", "evt_ceiling", "cdar"]
AXIS_E_BASELINE = "gz_loose"

# ======================================================================================
# Axis F grid (8 points)
# ======================================================================================
AXIS_F_GRID = [
    {"key": f"{scheme}_w{cw}_{'spike' if sp else 'nospike'}",
     "scheme": scheme, "corr_window": cw, "spike": sp}
    for scheme in ("corr_risk_parity", "equal_risk_weight")
    for cw in (60, 120)
    for sp in (True, False)
]
assert len(AXIS_F_GRID) == 8

COST_SCENARIOS_BP = [13.0, 6.5, 26.0]
FUNDING_SCENARIOS_BP_WEEK = [0.0, 5.0, -5.0]

# ======================================================================================
# STAGE-A PILOT UNIVERSE (22 names) -- CTA_MATH_NOTES.md Section 9-REDESIGN.
#
# Selection rule (deterministic, reproducible from research/cta/_universe_manifest_93.csv, no
# randomness):
#   0. KNOWN-BAD DATA EXCLUSION: AAPLUSDT is dropped from the candidate pool BEFORE stratification.
#      Its real 2020-08-31 4-for-1 stock split was never detected/adjusted by the shared data
#      lake's split pipeline (data/stitched/AAPLUSDT.parquet shows a fake -74.12% single-day
#      return that day; research/equity/DATA.md's own per-name manifest independently confirms
#      zero splits detected for AAPLUSDT despite this being a real, undisputed corporate action;
#      Axis E's own CTA_MATH_NOTES.md section E.6 independently flagged the exact same finding).
#      This is a known main-branch data-lake bug, out of scope to fix from this worktree -- the
#      safe response is exclusion, not a silent ride-through that would corrupt every downstream
#      Kelly/EVT/CDaR/HMM/GARCH statistic touching this name. (Several OTHER large single-day
#      moves in the final 22 -- e.g. BMNRUSDT +665% on 2025-06-30, BEUSDT -42% on 2019-08-13,
#      AAOIUSDT +67% on 2023-08-04, ASTSUSDT +72% on 2024-05-29 -- were individually spot-checked
#      against surrounding price continuity and found to be genuine, PERSISTING moves [price holds
#      at the new level afterward, unlike a snap-back data artifact], consistent with real small-
#      /micro-cap event-driven volatility, not split artifacts; they are correctly RETAINED.)
#   1. Force-include SPYUSDT and QQQUSDT (already re-run as separate single-instrument books
#      elsewhere in this sweep, and this program's own established CAGR/Sharpe benchmark names).
#   2. Rank the remaining 90 candidate names (91 minus the AAPLUSDT exclusion) by trailing
#      dollar-volume (mean(close*volume) over each name's full real source=='duka' history) into 3
#      equal-sized terciles (low/med/high liquidity).
#   3. Cross each liquidity tercile with 2 history-length bins (short/long, split at the
#      cross-sectional median span_days ~= 2911 real duka days) => 6 strata.
#   4. Draw a fixed quota from each of the 6 strata (base 20//6=3, remainder 2 assigned to the
#      2 largest-population strata, deterministic) summing to 20; within each stratum, greedily
#      prefer a sector not yet represented anywhere in the running selection (tie-break:
#      alphabetical symbol) to maximize sector diversity for a fixed total name count.
# Result: 22 names spanning 21 of the universe's 47 raw sector labels, deliberately including
# both very high liquidity (SPY/QQQ/AMZN) and low liquidity (BMNRUSDT/CIENUSDT/EWTUSDT)
# names, and both long-history (>=~2900 real days: ADBE/AMAT/CAT/.../SPY/QQQ) and short-history
# (BMNRUSDT ~370d, CRCLUSDT ~249d, COINUSDT ~1762d, ASTSUSDT ~1881d, DELLUSDT ~2713d) names.
# See `research/cta/_universe_manifest_93.csv` and `research/cta/_build_universe22.py` (both
# committed alongside this file) for the exact manifest and selection script this list was
# generated from.
# ======================================================================================
STAGE_A_UNIVERSE_22 = [
    "AAOIUSDT", "ADBEUSDT", "AMATUSDT", "AMDUSDT", "AMZNUSDT", "ASTSUSDT", "BEUSDT", "BMNRUSDT",
    "BRKBUSDT", "BXUSDT", "CATUSDT", "CIENUSDT", "COINUSDT", "CRCLUSDT", "CRWDUSDT", "DELLUSDT",
    "DISUSDT", "EWJUSDT", "EWTUSDT", "EWYUSDT", "QQQUSDT", "SPYUSDT",
]
assert len(STAGE_A_UNIVERSE_22) == 22, len(STAGE_A_UNIVERSE_22)
assert "AAPLUSDT" not in STAGE_A_UNIVERSE_22, "AAPLUSDT has a known unadjusted-split data bug -- excluded by design"


# ======================================================================================
# 0. Universe reconciliation
# ======================================================================================
def reconcile_universe() -> list[str]:
    u = load_universe()
    eq = [d["symbol"] for d in u["equities"]]
    print(f"[universe] equities sleeve from load_universe(): {len(eq)}", flush=True)

    kept, excluded_no_duka, excluded_short, excluded_lev = [], [], [], []
    for sym in eq:
        fp = STITCHED_DIR / f"{sym}.parquet"
        if not fp.exists():
            excluded_no_duka.append(sym)
            continue
        df = pd.read_parquet(fp, columns=["open_time", "source"])
        duka = df[df["source"] == "duka"]
        if len(duka) == 0:
            excluded_no_duka.append(sym)
            continue
        span_days = (duka["open_time"].max() - duka["open_time"].min()) / 1000.0 / 86400.0
        if span_days < MIN_DUKA_SPAN_DAYS:
            excluded_short.append(sym)
            continue
        if sym in EXCLUDE_LEVERAGED:
            excluded_lev.append(sym)
            continue
        kept.append(sym)

    print(f"[universe] excluded (0 duka rows / perp-only): {len(excluded_no_duka)} {sorted(excluded_no_duka)}", flush=True)
    print(f"[universe] excluded (span < {MIN_DUKA_SPAN_DAYS}d, short-history): {len(excluded_short)} {sorted(excluded_short)}", flush=True)
    print(f"[universe] excluded (leveraged/inverse/decay ETF): {len(excluded_lev)} {sorted(excluded_lev)}", flush=True)
    print(f"[universe] RECONCILED BASE UNIVERSE: {len(kept)} names", flush=True)

    assert set(excluded_short) == EXCLUDE_SHORT_HISTORY, (
        f"short-history exclusion mismatch: got {sorted(excluded_short)}, expected {sorted(EXCLUDE_SHORT_HISTORY)}")
    if len(kept) != 93:
        print(f"[universe] *** DISCREPANCY: expected 93, got {len(kept)} ***", flush=True)
    return sorted(kept)


def reconcile_universe_stage_a() -> list[str]:
    """Stage-A pilot universe: intersect the full reconciled 93-name base universe with the
    frozen STAGE_A_UNIVERSE_22 list (defensive -- re-verifies every one of the 22 names is still
    a genuine member of today's 93-name reconciliation before using it, rather than assuming the
    hardcoded list from CTA_MATH_NOTES.md Section 9-REDESIGN is still valid against fresh data).
    """
    full = reconcile_universe()
    missing = [s for s in STAGE_A_UNIVERSE_22 if s not in full]
    if missing:
        raise RuntimeError(f"Stage-A universe names missing from the reconciled 93-name base: {missing}")
    print(f"[universe] STAGE-A PILOT SUBSET: {len(STAGE_A_UNIVERSE_22)} of {len(full)} reconciled "
          f"names selected (stratified by dollar-volume tercile x history-length bin, sector-"
          f"diversified -- CTA_MATH_NOTES.md Section 9-REDESIGN). Stage B's later frozen "
          f"confirmatory run uses the full {len(full)}-name base universe.", flush=True)
    return sorted(STAGE_A_UNIVERSE_22)


# ======================================================================================
# 1. Per-name data loading
# ======================================================================================
def load_name_frames(symbol: str) -> dict | None:
    fp = STITCHED_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(fp)
    df = df[df["source"] == "duka"].copy()          # bug pattern (a)
    if len(df) < 300 * 390:  # crude floor: need real history at all
        pass
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = vf.aggregate_daily_ohlc_from_duka(df)     # bug pattern: aggregate from 1-min duka rows
    if len(ohlc) < 60:
        # too little real history to be a meaningful book member at all (distinct from the
        # point-in-time gating that each axis module already applies internally via NaN-until-
        # ready columns for names with SOME but not yet enough history for a given signal window)
        return None
    daily_close = ohlc["close"]
    log_price = np.log(daily_close)
    simple_ret = daily_close.pct_change()
    rv_intraday = vf.realized_variance_from_intraday(df, sampling_minutes=5)
    return {
        "symbol": symbol,
        "ohlc": ohlc,
        "close": daily_close,
        "log_price": log_price,
        "simple_ret": simple_ret,
        "rv_intraday": rv_intraday,
        "n_obs": len(ohlc),
        "span_years": (ohlc.index[-1] - ohlc.index[0]).days / 365.25,
    }


# ======================================================================================
# 2. Axis A: 31-point grid
# ======================================================================================
def _combine(components: pd.DataFrame, log_price: pd.Series, ic_window: int, shrinkage: float) -> pd.Series:
    """Exact replica of multi_horizon_trend_signal's own combination step (Section A.4.4),
    factored out so the (expensive) raw component build can be shared across the 10 (shrinkage,
    ic_window) combinations in bucket 4 -- calls ONLY se.rolling_information_coefficient and
    se.combination_weights_from_ic, i.e. the axis's own public functions, in the same order its
    own top-level function uses them."""
    ic = pd.DataFrame({c: se.rolling_information_coefficient(components[c], log_price, ic_window)
                        for c in components.columns})
    weights = se.combination_weights_from_ic(ic, shrinkage=shrinkage)
    raw_combined = (components * weights).sum(axis=1, skipna=True)
    all_nan = components.isna().all(axis=1) | weights.isna().all(axis=1)
    raw_combined = raw_combined.mask(all_nan, np.nan)
    return raw_combined.shift(1)


def compute_axis_a(log_price: pd.Series) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    # Bucket 4 first: build the (shared, default-parameterized) raw components ONCE.
    components = se.build_component_signals(log_price)  # ewma_8/32/128, kalman_slope, l1tf_slope
    for shr in FULL_SHRINKAGES:
        for icw in FULL_IC_WINDOWS:
            out[f"full_shr{shr}_ic{icw}"] = _combine(components, log_price, icw, shr)

    # Bucket 1: EWMA-only ensembles (3 halflife triples x 3 shrinkages)
    for trip in EWMA_TRIPLES:
        comp = pd.DataFrame({
            f"ewma_{hl:g}": se.normalize_signal(se.ewma_return_momentum(log_price, hl))
            for hl in trip
        })
        for shr in EWMA_SHRINKAGES:
            key = f"ewma_{trip[0]:g}_{trip[1]:g}_{trip[2]:g}_shr{shr}"
            out[key] = _combine(comp, log_price, 252, shr)

    # Bucket 2: Kalman-only (3 parameter triples) -- reuse the default-triple component from bucket 4
    for q_level, q_slope, r in KALMAN_TRIPLES:
        key = f"kalman_{q_level:g}_{q_slope:g}_{r:g}"
        if key == DEFAULT_KALMAN_KEY:
            out[key] = components["kalman_slope"].shift(1)
        else:
            kf = se.kalman_local_linear_trend(log_price, q_level=q_level, q_slope=q_slope, r=r)
            raw = se.normalize_signal(kf["slope"])
            out[key] = raw.shift(1)

    # Bucket 3: l1-trend-filter-only (3 windows x 3 lam_mult) -- reuse default (120,1.0) from bucket 4
    for w in L1TF_WINDOWS:
        for lm in L1TF_LAM_MULTS:
            key = f"l1tf_w{w}_lm{lm}"
            if key == DEFAULT_L1TF_KEY:
                out[key] = components["l1tf_slope"].shift(1)
            else:
                raw_l1 = se.l1_trend_filter_slope_signal(log_price, window=w, lam_mult=lm)
                raw = se.normalize_signal(raw_l1)
                out[key] = raw.shift(1)

    assert set(out.keys()) == set(AXIS_A_KEYS), out.keys() ^ set(AXIS_A_KEYS)
    return out


# ======================================================================================
# 3. Axis C: 5-point grid -> exposure-scale gate in [0,1], causal
# ======================================================================================
def compute_axis_c(simple_ret: pd.Series, close: pd.Series, n_obs: int) -> dict[str, pd.Series]:
    log_ret = np.log(1.0 + simple_ret).dropna()
    out: dict[str, pd.Series] = {}
    hmm_cache: dict[tuple[int, int], pd.Series] = {}
    for g in AXIS_C_GRID:
        if g.get("type") == "sma":
            gate = rh.sma_binary_gate(close, window=200, band=0.02)
            out[g["key"]] = gate.reindex(close.index)
            continue
        n_states, refit_every = g["n_states"], g["refit_every"]
        eff_n_states = n_states
        if n_states == 3 and n_obs < MIN_OBS_FOR_3STATE:
            eff_n_states = 2  # disclosed per-name fallback, C.6/8.6 -- no separate grid cell
        cache_key = (eff_n_states, refit_every)
        if cache_key not in hmm_cache:
            post = rh.causal_hmm_regime_posterior(log_ret, n_states=eff_n_states,
                                                   refit_every=refit_every, n_restarts=5)
            gate = rh.exposure_scale_from_posterior(post["p_bull"])
            hmm_cache[cache_key] = gate.reindex(close.index)
        out[g["key"]] = hmm_cache[cache_key]
    assert set(out.keys()) == {g["key"] for g in AXIS_C_GRID}
    return out


# ======================================================================================
# 4. Axis D: 10-point grid -> forecast_vol (causal) + vol_scale (harmonic-mean-calibrated)
# ======================================================================================
def _harmonic_mean(s: pd.Series) -> float:
    v = s.dropna()
    v = v[v > 0]
    if len(v) == 0:
        return np.nan
    return float(len(v) / np.sum(1.0 / v))


def compute_axis_d(name_frames: dict) -> dict[str, dict[str, pd.Series]]:
    ohlc, close, rv_intraday = name_frames["ohlc"], name_frames["close"], name_frames["rv_intraday"]
    out: dict[str, dict[str, pd.Series]] = {}
    for g in AXIS_D_GRID:
        kind = g["kind"]
        if kind == "trailing":
            fvol = vf.trailing_realized_vol_forecast(close, window=g["window"])
        elif kind == "yz":
            fvol = vf.yang_zhang_vol_forecast(ohlc, window=g["window"])
        elif kind == "ewma":
            fvol = vf.ewma_vol_forecast(close, lam=g["lam"])
        elif kind == "garch":
            fvol = vf.garch11_vol_forecast(close, refit_freq=g["refit_freq"], window=g["window"])
        elif kind == "har":
            fvol = vf.har_rv_vol_forecast(rv_intraday, refit_freq=g["refit_freq"]).reindex(close.index)
        else:
            raise ValueError(kind)
        target_daily_vol = _harmonic_mean(fvol)
        if not np.isfinite(target_daily_vol) or target_daily_vol <= 0:
            target_daily_vol = 0.01
        vol_scale = vf.vol_target_exposure_magnitude(fvol, target_daily_vol=target_daily_vol,
                                                       leverage_target=1.0, cap_mult=2.0)
        out[g["key"]] = {"forecast_vol": fvol, "vol_scale": vol_scale}
    assert set(out.keys()) == {g["key"] for g in AXIS_D_GRID}
    return out


# ======================================================================================
# Phase 1 worker: everything for one name
# ======================================================================================
def build_name_cache(symbol: str) -> tuple[str, dict | None]:
    t0 = time.time()
    try:
        frames = load_name_frames(symbol)
        if frames is None:
            return symbol, None
        axis_a = compute_axis_a(frames["log_price"])
        axis_c = compute_axis_c(frames["simple_ret"], frames["close"], frames["n_obs"])
        axis_d = compute_axis_d(frames)
        cache = {
            "close": frames["close"],
            "simple_ret": frames["simple_ret"],
            "axis_a": axis_a,
            "axis_c": axis_c,
            "axis_d": axis_d,
            "n_obs": frames["n_obs"],
            "span_years": frames["span_years"],
        }
        dt = time.time() - t0
        print(f"[phase1] {symbol} done in {dt:.1f}s (n_obs={frames['n_obs']})", flush=True)
        return symbol, cache
    except Exception as e:
        print(f"[phase1] {symbol} FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return symbol, None


# ======================================================================================
# Axis F caching wrappers (memoize on args that don't vary with target_exposure) -- see module
# docstring point 2. Monkey-patched onto `portfolio_construction` before Phase 2 begins.
# ======================================================================================
_AVG_CORR_CACHE: dict = {}
_SPIKE_CACHE: dict = {}
_SHRUNK_COV_CACHE: dict = {}

_ORIG_AVG_CORR = pc.rolling_avg_pairwise_correlation
_ORIG_SPIKE = pc.correlation_spike_scale
_ORIG_SHRUNK_COV = pc.rolling_shrunk_covariance
_ORIG_PCA_ENTROPY = pc.effective_n_pca_entropy


def _cached_avg_corr(returns_, window=60):
    key = (id(returns_), window)
    if key not in _AVG_CORR_CACHE:
        _AVG_CORR_CACHE[key] = _ORIG_AVG_CORR(returns_, window=window)
    return _AVG_CORR_CACHE[key]


def _cached_spike(avg_corr, lookback=756, spike_percentile=90.0, full_deleverage_percentile=99.0,
                   scale_floor=0.4):
    key = (id(avg_corr), lookback, spike_percentile, full_deleverage_percentile, scale_floor)
    if key not in _SPIKE_CACHE:
        _SPIKE_CACHE[key] = _ORIG_SPIKE(avg_corr, lookback=lookback, spike_percentile=spike_percentile,
                                         full_deleverage_percentile=full_deleverage_percentile,
                                         scale_floor=scale_floor)
    return _SPIKE_CACHE[key]


_SHRUNK_COV_SENTINEL = object()


def _cached_shrunk_cov(returns_, t, window=60, shrink_intensity=0.3):
    key = (id(returns_), window, shrink_intensity)
    lst = _SHRUNK_COV_CACHE.get(key)
    if lst is None:
        lst = [_SHRUNK_COV_SENTINEL] * len(returns_)
        _SHRUNK_COV_CACHE[key] = lst
    if lst[t] is _SHRUNK_COV_SENTINEL:
        lst[t] = _ORIG_SHRUNK_COV(returns_, t, window=window, shrink_intensity=shrink_intensity)
    return lst[t]


_PCA_ENTROPY_CACHE: dict = {}


def _cached_pca_entropy(cov):
    key = id(cov)
    if key not in _PCA_ENTROPY_CACHE:
        _PCA_ENTROPY_CACHE[key] = _ORIG_PCA_ENTROPY(cov)
    return _PCA_ENTROPY_CACHE[key]


def install_axis_f_cache():
    pc.rolling_avg_pairwise_correlation = _cached_avg_corr
    pc.correlation_spike_scale = _cached_spike
    pc.rolling_shrunk_covariance = _cached_shrunk_cov
    pc.effective_n_pca_entropy = _cached_pca_entropy


# ======================================================================================
# PERFORMANCE FIX (CTA_MATH_NOTES.md Section 9-REDESIGN.7): eliminate REDUNDANT
# aggregate_portfolio() calls within the F-loop of process_ab_pair/process_single_instrument.
#
# Root cause found live during the feasibility check: aggregate_portfolio()'s per-day loop
# (O(T)~2053 Python-level iterations, dominated by the corr_risk_parity branch's per-day
# risk_parity_weights ERC solve) is the REAL bottleneck the original per-cell cost estimate
# ("portfolio aggregation ~4-8s/call") badly undersold -- calling it naively 8 times per (A,B,
# OAT-slot) triple (once per AXIS_F_GRID point) produced an observed ETA of ~173 minutes for
# phase 2 alone on the real 22-name universe, wildly over budget.
#
# EXACT (not approximate) fix, requiring no change to portfolio_construction.py: by inspection
# of aggregate_portfolio's own body, its "with spike overlay" output at every date t is exactly
#     final_with_spike(t) = final_no_spike(t) * spike_scale(t)
# (spike_scale(t) multiplies the already-unit-gross-renormalized raw weights; with
# apply_spike_overlay=False, spike_scale(t) === 1.0 identically). Separately, for
# scheme="equal_risk_weight", the per-day loop's weight computation does not depend on
# corr_window AT ALL (corr_window only feeds avg_corr, which is itself only consumed by the
# spike overlay) -- so all 4 equal_risk_weight AXIS_F_GRID points (2 corr_windows x 2 spike
# settings) are derivable from a SINGLE base call. For scheme="corr_risk_parity", corr_window
# DOES change the covariance (and hence the ERC weights) genuinely, so 2 base calls (one per
# corr_window) are still required, each spike-derivable in the same way. This cuts 8 calls down
# to 3 GENUINELY DISTINCT calls per OAT slot -- a >2.6x reduction, zero change to any resulting
# number (verified byte-identical to the naive 8-call path, see the smoke check run before
# relaunch in the structured output / chat report).
# ======================================================================================
_SPIKE_SCALE_BY_WINDOW: dict[tuple[int, int], pd.Series] = {}


def _get_spike_scale(returns_df: pd.DataFrame, corr_window: int) -> pd.Series:
    """Spike-scale series for a given (returns_df, corr_window), keyed on id(returns_df) so this
    works correctly for both the 22-name STAGE_A_22 book (RETURNS_DF) and each 1-name SPY/QQQ
    single-instrument book (ret_df) without cross-contaminating caches. Transparently hits the
    SAME memoizing caches install_axis_f_cache() patches onto `pc` (avg corr + spike scale), so
    this never duplicates work already done elsewhere, and is itself cached per-worker-process so
    repeated (A,B)-pair calls are free after the first. (For a 1-name book, avg_corr is
    ill-defined/NaN throughout -- matching aggregate_portfolio's own internal behavior exactly --
    so spike_scale fails open to 1.0 throughout, i.e. with/without-spike variants are identical,
    same as the original naive-8-call implementation would have produced.)"""
    key = (id(returns_df), corr_window)
    if key not in _SPIKE_SCALE_BY_WINDOW:
        avg_corr = pc.rolling_avg_pairwise_correlation(returns_df, window=corr_window)
        _SPIKE_SCALE_BY_WINDOW[key] = pc.correlation_spike_scale(avg_corr)
    return _SPIKE_SCALE_BY_WINDOW[key]


def _compute_f_variant_weights(exposure_df: pd.DataFrame, returns_df: pd.DataFrame,
                                rv_df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Compute all 8 AXIS_F_GRID points' (weights, n_active) for one (exposure_df, rv_df) pair,
    using only 3 real aggregate_portfolio() calls (see the module-level note above for the exact
    algebraic justification) -- or just 1 for a single-name book (see below). Returns
    {f_key: (weights_df, n_active_series)}."""
    # SECOND-TIER PERFORMANCE FIX, single-name (SPY-alone/QQQ-alone) books specifically: with only
    # 1 possible name, aggregate_portfolio's own `usable` set (intersection of active names with
    # covariance-computed names) can never reach length >= 2, so `use_corr_scheme` is NEVER True
    # by construction -- scheme="corr_risk_parity" ALWAYS falls back to the identical
    # "equal_risk_weight_fallback" computation regardless of corr_window. All 8 AXIS_F_GRID points
    # therefore collapse to exactly 1 real computation for a 1-name book -- exact, not
    # approximate (same code-path guarantee `aggregate_portfolio` itself already enforces via its
    # own `len(usable) >= 2` guard).
    if exposure_df.shape[1] <= 1:
        base_eq = pc.aggregate_portfolio(exposure_df, returns_df, rv_df, scheme="equal_risk_weight",
                                          corr_window=60, apply_spike_overlay=False)
        out: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        for f_g in AXIS_F_GRID:
            if f_g["spike"]:
                w = base_eq.weights.multiply(_get_spike_scale(returns_df, f_g["corr_window"]), axis=0)
            else:
                w = base_eq.weights
            out[f_g["key"]] = (w, base_eq.n_active)
        return out

    base_corr = {
        cw: pc.aggregate_portfolio(exposure_df, returns_df, rv_df, scheme="corr_risk_parity",
                                    corr_window=cw, apply_spike_overlay=False)
        for cw in (60, 120)
    }
    base_eq = pc.aggregate_portfolio(exposure_df, returns_df, rv_df, scheme="equal_risk_weight",
                                      corr_window=60, apply_spike_overlay=False)
    out = {}
    for f_g in AXIS_F_GRID:
        base = base_corr[f_g["corr_window"]] if f_g["scheme"] == "corr_risk_parity" else base_eq
        if f_g["spike"]:
            w = base.weights.multiply(_get_spike_scale(returns_df, f_g["corr_window"]), axis=0)
        else:
            w = base.weights
        out[f_g["key"]] = (w, base.n_active)
    return out


# ======================================================================================
# Cost application + metrics (the Sweep phase's own responsibility per Axis B/E docstrings)
# ======================================================================================
def apply_costs(weights_df: pd.DataFrame, returns_df: pd.DataFrame, cost_bp: float,
                 funding_bp_week: float) -> np.ndarray:
    idx = weights_df.index
    w = weights_df.fillna(0.0).to_numpy()
    r = returns_df.reindex(columns=weights_df.columns).fillna(0.0).to_numpy()
    delta_w = np.diff(w, axis=0, prepend=w[[0]])
    delta_w[0, :] = w[0, :]  # first day: trading in from flat
    trading_cost = (cost_bp / 1e4) * np.abs(delta_w)
    cal_days = pd.Series(idx).diff().dt.days.fillna(1.0).to_numpy()
    cal_days = np.where(cal_days <= 0, 1.0, cal_days)
    funding_cost = (funding_bp_week / 1e4 / 7.0) * cal_days[:, None] * np.abs(w)
    gross = w * r
    net_per_name = gross - trading_cost - funding_cost
    book_return = np.nansum(net_per_name, axis=1)
    return book_return


def _cagr_from_returns(net_ret: np.ndarray, dates: pd.DatetimeIndex) -> tuple[float, float]:
    n = len(net_ret)
    if n < 2:
        return np.nan, np.nan
    span_years = (dates[-1] - dates[0]).days / 365.25
    if span_years <= 0:
        return np.nan, np.nan
    equity = np.cumprod(1.0 + net_ret)
    final_eq = equity[-1]
    if final_eq <= 0:
        cagr = -1.0
    else:
        cagr = final_eq ** (1.0 / span_years) - 1.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, np.nan)
    max_dd = float(np.nanmax(dd)) if len(dd) else np.nan
    return float(cagr), max_dd


def apply_leverage(book_return_1x: np.ndarray, e_key: str) -> tuple[np.ndarray, float]:
    r = book_return_1x
    finite = r[np.isfinite(r)]
    if len(finite) < 60 or np.allclose(finite, 0.0):
        return r.copy(), 0.0
    try:
        kelly = lp.kelly_optimal_leverage(finite, l_max=25.0)
        l_star = kelly["L_star"]
    except Exception:
        l_star = 0.0

    if e_key == "half_kelly":
        L = lp.fractional_kelly_leverage(max(l_star, 1e-6), 0.5) if l_star > 0 else 0.0
        final = L * np.nan_to_num(r)
        return final, float(L)
    if e_key == "quarter_kelly":
        L = lp.fractional_kelly_leverage(max(l_star, 1e-6), 0.25) if l_star > 0 else 0.0
        final = L * np.nan_to_num(r)
        return final, float(L)
    if e_key == "gz_tight":
        res = lp.grossman_zhou_exposure(np.nan_to_num(r), l_base=max(l_star, 1e-6), alpha_floor=0.75, rt_bp=13.0)
        return np.asarray(res["net_return"]), float(res["avg_exposure"])
    if e_key == "gz_loose":
        res = lp.grossman_zhou_exposure(np.nan_to_num(r), l_base=max(l_star, 1e-6), alpha_floor=0.50, rt_bp=13.0)
        return np.asarray(res["net_return"]), float(res["avg_exposure"])
    if e_key == "evt_ceiling":
        try:
            evt = lp.evt_ceiling_leverage(finite, threshold_quantile=0.95, breach_years=10.0)
            L = evt["L_ceiling"]
            if not np.isfinite(L):
                L = 0.0
        except Exception:
            L = 0.0
        final = min(L, 25.0) * np.nan_to_num(r)
        return final, float(min(L, 25.0))
    if e_key == "cdar":
        try:
            cdar = lp.cdar_constrained_leverage(finite, budget=0.25, alpha=0.95)
            L = cdar["L_closed_form"]
            if not np.isfinite(L):
                L = 0.0
        except Exception:
            L = 0.0
        final = min(L, 25.0) * np.nan_to_num(r)
        return final, float(min(L, 25.0))
    raise ValueError(e_key)


def compute_metrics(final_net_return: np.ndarray, dates: pd.DatetimeIndex, avg_leverage: float,
                     n_active: pd.Series | None, n_trades: int) -> dict:
    r = np.nan_to_num(final_net_return)
    n = len(r)
    span_years = (dates[-1] - dates[0]).days / 365.25 if n > 1 else np.nan
    periods_per_year = n / span_years if span_years and span_years > 0 else 252.0

    cagr, max_dd = _cagr_from_returns(r, dates)
    mu, sd = np.mean(r), np.std(r, ddof=1) if n > 1 else np.nan
    sharpe = (mu / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan

    active_mask = r != 0.0
    n_active_days = int(active_mask.sum())
    win_rate = float(np.mean(r[active_mask] > 0)) if n_active_days > 0 else np.nan
    worst_day = float(np.min(r)) if n else np.nan
    ruin_flag = bool(np.any(1.0 + r <= 0.0))
    trades_per_yr = (n_trades / span_years) if span_years and span_years > 0 else np.nan
    total_net = float(np.sum(r))
    ev_trade_bp = (total_net / n_trades * 1e4) if n_trades > 0 else np.nan

    out = {
        "cagr": cagr, "maxdd": max_dd, "sharpe": sharpe, "ev_trade_bp": ev_trade_bp,
        "win_rate": win_rate, "trades_per_yr": trades_per_yr, "worst_day": worst_day,
        "ruin_flag": ruin_flag, "avg_leverage": avg_leverage,
        "cagr_delta_vs_benchmark": (cagr - BENCHMARK_CAGR) if np.isfinite(cagr) else np.nan,
        "n_obs": n, "span_years": span_years,
    }
    for budget in DD_BUDGETS:
        out[f"dd_budget_{int(budget*100)}"] = bool(np.isfinite(max_dd) and max_dd <= budget)

    idx = pd.DatetimeIndex(dates)
    ser = pd.Series(r, index=idx)
    for label, (start, end) in REGIME_WINDOWS.items():
        window = ser.loc[(idx >= pd.Timestamp(start, tz="UTC")) & (idx <= pd.Timestamp(end, tz="UTC"))]
        out[label] = float(np.prod(1.0 + window.to_numpy()) - 1.0) if len(window) else np.nan
    return out


# ======================================================================================
# Phase 2: (Axis A, Axis B) -> exposure_df -> (Axis F x OAT-slot) -> rows
# ======================================================================================
NAME_CACHE: dict = {}
RETURNS_DF: pd.DataFrame = None
CLOSE_DF: pd.DataFrame = None
NAMES: list[str] = []


def _gated_target(name: str, a_key: str, c_key: str, d_key: str) -> pd.Series:
    cache = NAME_CACHE[name]
    raw_dir = cache["axis_a"][a_key]
    gate = cache["axis_c"][c_key]
    vol_scale = cache["axis_d"][d_key]["vol_scale"]
    idx = cache["close"].index
    raw_dir = raw_dir.reindex(idx)
    gate = gate.reindex(idx)
    vol_scale = vol_scale.reindex(idx)
    gated = np.clip(raw_dir * gate, 0.0, None) * vol_scale
    return gated.fillna(0.0)


def _exposure_for(names: list[str], a_key: str, b_params: dict, c_key: str, d_key: str) -> tuple[pd.DataFrame, int]:
    exposures = {}
    n_trades = 0
    for name in names:
        cache = NAME_CACHE[name]
        gt = _gated_target(name, a_key, c_key, d_key)
        fvol = cache["axis_d"][d_key]["forecast_vol"].reindex(cache["close"].index)
        res = eec.partial_adjustment_control(
            gt, cache["close"], cost_bps=13.0, gamma=1.0,
            cost_sensitivity=b_params["cost_sensitivity"], band_scale=b_params["band_scale"],
            external_vol=fvol,
        )
        exposures[name] = res["exposure"].reindex(RETURNS_DF.index)
        n_trades += int(res["traded"].sum())
    return pd.DataFrame(exposures), n_trades


def process_ab_pair(args) -> list[dict]:
    a_key, b_key = args
    b_params = next(g for g in AXIS_B_GRID if g["name"] == b_key)
    rows: list[dict] = []
    dates = RETURNS_DF.index

    oat_slots = [("baseline", AXIS_C_BASELINE, AXIS_D_BASELINE)]
    oat_slots += [("C", g["key"], AXIS_D_BASELINE) for g in AXIS_C_GRID if g["key"] != AXIS_C_BASELINE]
    oat_slots += [("D", AXIS_C_BASELINE, g["key"]) for g in AXIS_D_GRID if g["key"] != AXIS_D_BASELINE]

    exposure_cache: dict[str, tuple[pd.DataFrame, int]] = {}
    rv_df_cache: dict[str, pd.DataFrame] = {}
    for slot_type, c_key, d_key in oat_slots:
        exposure_cache[(c_key, d_key)] = _exposure_for(NAMES, a_key, b_params, c_key, d_key)
        if d_key not in rv_df_cache:
            rv_df_cache[d_key] = pd.DataFrame({name: NAME_CACHE[name]["axis_d"][d_key]["forecast_vol"]
                                                .reindex(dates) for name in NAMES})

    for slot_type, c_key, d_key in oat_slots:
        exposure_df, n_trades = exposure_cache[(c_key, d_key)]
        rv_df = rv_df_cache[d_key]
        # PERFORMANCE FIX (Section 9-REDESIGN.7): 3 real aggregate_portfolio() calls per OAT
        # slot instead of 8 -- see _compute_f_variant_weights's own docstring for the exact
        # (non-approximate) algebraic justification.
        f_variants = _compute_f_variant_weights(exposure_df, RETURNS_DF, rv_df)

        for f_g in AXIS_F_GRID:
            f_key = f_g["key"]
            weights, n_active = f_variants[f_key]
            book_1x_primary = apply_costs(weights, RETURNS_DF, cost_bp=13.0, funding_bp_week=0.0)
            book_1x_65bp = apply_costs(weights, RETURNS_DF, cost_bp=6.5, funding_bp_week=0.0)
            book_1x_26bp = apply_costs(weights, RETURNS_DF, cost_bp=26.0, funding_bp_week=0.0)
            book_1x_fplus = apply_costs(weights, RETURNS_DF, cost_bp=13.0, funding_bp_week=5.0)
            book_1x_fminus = apply_costs(weights, RETURNS_DF, cost_bp=13.0, funding_bp_week=-5.0)

            def _row(e_key: str, base_book: np.ndarray) -> dict:
                final, avg_lev = apply_leverage(base_book, e_key)
                m = compute_metrics(final, dates, avg_lev, n_active, n_trades)
                m.update({
                    "universe": "STAGE_A_22", "a_key": a_key, "b_key": b_key, "c_key": c_key,
                    "d_key": d_key, "e_key": e_key, "f_key": f_key, "oat_axis": slot_type,
                })
                # cost/funding scenario CAGRs, reusing the SAME leverage path derived above
                for scen_name, scen_book in [("6.5bp", book_1x_65bp), ("26bp", book_1x_26bp),
                                              ("fund_plus5", book_1x_fplus), ("fund_minus5", book_1x_fminus)]:
                    if e_key in ("gz_tight", "gz_loose"):
                        L_final, _ = apply_leverage(scen_book, e_key)
                        cagr_s, _ = _cagr_from_returns(np.nan_to_num(L_final), dates)
                    else:
                        cagr_s, _ = _cagr_from_returns(np.nan_to_num(avg_lev * scen_book), dates)
                    m[f"cagr_at_{scen_name}"] = cagr_s
                return m

            if slot_type == "baseline":
                rows.append(_row(AXIS_E_BASELINE, book_1x_primary))
                for e_key in AXIS_E_KEYS:
                    if e_key == AXIS_E_BASELINE:
                        continue
                    rows.append(_row(e_key, book_1x_primary))
            else:
                rows.append(_row(AXIS_E_BASELINE, book_1x_primary))
    return rows


# ======================================================================================
# Single-instrument (SPY / QQQ) sweep -- same full grid, N=1 "book"
# ======================================================================================
def process_single_instrument(symbol: str) -> list[dict]:
    rows: list[dict] = []
    cache = NAME_CACHE[symbol]
    dates = cache["close"].index
    ret_df = cache["simple_ret"].to_frame(symbol)

    oat_slots = [("baseline", AXIS_C_BASELINE, AXIS_D_BASELINE)]
    oat_slots += [("C", g["key"], AXIS_D_BASELINE) for g in AXIS_C_GRID if g["key"] != AXIS_C_BASELINE]
    oat_slots += [("D", AXIS_C_BASELINE, g["key"]) for g in AXIS_D_GRID if g["key"] != AXIS_D_BASELINE]

    for a_key in AXIS_A_KEYS:
        for b_key in AXIS_B_KEYS:
            b_params = next(g for g in AXIS_B_GRID if g["name"] == b_key)
            exposure_cache = {}
            for slot_type, c_key, d_key in oat_slots:
                exposure_cache[(c_key, d_key)] = _exposure_for([symbol], b_params=b_params, a_key=a_key,
                                                                 c_key=c_key, d_key=d_key)
            for slot_type, c_key, d_key in oat_slots:
                exposure_df, n_trades = exposure_cache[(c_key, d_key)]
                rv_df = cache["axis_d"][d_key]["forecast_vol"].reindex(dates).to_frame(symbol)
                # PERFORMANCE FIX (Section 9-REDESIGN.7): 3 real aggregate_portfolio() calls per
                # OAT slot instead of 8 -- see _compute_f_variant_weights's docstring.
                f_variants = _compute_f_variant_weights(exposure_df.reindex(dates), ret_df, rv_df)

                for f_g in AXIS_F_GRID:
                    weights, n_active = f_variants[f_g["key"]]
                    book_1x = apply_costs(weights, ret_df, cost_bp=13.0, funding_bp_week=0.0)
                    book_65 = apply_costs(weights, ret_df, cost_bp=6.5, funding_bp_week=0.0)
                    book_26 = apply_costs(weights, ret_df, cost_bp=26.0, funding_bp_week=0.0)
                    book_fp = apply_costs(weights, ret_df, cost_bp=13.0, funding_bp_week=5.0)
                    book_fm = apply_costs(weights, ret_df, cost_bp=13.0, funding_bp_week=-5.0)

                    def _row(e_key, base_book):
                        final, avg_lev = apply_leverage(base_book, e_key)
                        m = compute_metrics(final, dates, avg_lev, n_active, n_trades)
                        m.update({"universe": symbol.replace("USDT", ""), "a_key": a_key, "b_key": b_key,
                                  "c_key": c_key, "d_key": d_key, "e_key": e_key, "f_key": f_g["key"],
                                  "oat_axis": slot_type})
                        for scen_name, scen_book in [("6.5bp", book_65), ("26bp", book_26),
                                                      ("fund_plus5", book_fp), ("fund_minus5", book_fm)]:
                            if e_key in ("gz_tight", "gz_loose"):
                                Lf, _ = apply_leverage(scen_book, e_key)
                                cs, _ = _cagr_from_returns(np.nan_to_num(Lf), dates)
                            else:
                                cs, _ = _cagr_from_returns(np.nan_to_num(avg_lev * scen_book), dates)
                            m[f"cagr_at_{scen_name}"] = cs
                        return m

                    if slot_type == "baseline":
                        rows.append(_row(AXIS_E_BASELINE, book_1x))
                        for e_key in AXIS_E_KEYS:
                            if e_key == AXIS_E_BASELINE:
                                continue
                            rows.append(_row(e_key, book_1x))
                    else:
                        rows.append(_row(AXIS_E_BASELINE, book_1x))
    return rows


def flush_checkpoint(all_rows: list[dict], path: Path):
    df = pd.DataFrame(all_rows)
    df.to_parquet(path, index=False)
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{pd.Timestamp.now()} checkpoint: {len(df)} rows written to {path}\n")
    print(f"[checkpoint] {len(df)} rows -> {path}", flush=True)


# ======================================================================================
# main
# ======================================================================================
def main():
    t_start = time.time()
    print(f"=== ctaA_sweep.py starting {pd.Timestamp.now()} ===", flush=True)
    print(f"N_PROCESSES = {N_PROCESSES}", flush=True)
    n_oat = 1 + (len(AXIS_C_GRID) - 1) + (len(AXIS_D_GRID) - 1) + (len(AXIS_E_KEYS) - 1)
    n_core = len(AXIS_A_KEYS) * len(AXIS_B_KEYS) * len(AXIS_F_GRID)
    print(f"REDESIGNED GRID: core (A x B x F) = {n_core}, OAT block = {n_oat}, "
          f"N_cells/universe = {n_core * n_oat}, x3 universes = {n_core * n_oat * 3} total rows "
          f"(CTA_MATH_NOTES.md Section 9-REDESIGN)", flush=True)

    names = reconcile_universe_stage_a()
    global NAMES
    NAMES = names

    print(f"[phase1] building per-name signal caches for {len(names)} names ...", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(N_PROCESSES) as pool:
        results = pool.map(build_name_cache, names, chunksize=1)
    for sym, cache in results:
        if cache is not None:
            NAME_CACHE[sym] = cache
        else:
            print(f"[phase1] WARNING: {sym} produced no cache, excluding from sweep", flush=True)
    active_names = sorted(NAME_CACHE.keys())
    NAMES = active_names
    print(f"[phase1] done in {time.time()-t0:.1f}s, {len(active_names)}/{len(names)} names usable", flush=True)

    global RETURNS_DF, CLOSE_DF
    idx = None
    for sym in active_names:
        idx = NAME_CACHE[sym]["close"].index if idx is None else idx.union(NAME_CACHE[sym]["close"].index)
    idx = idx.sort_values()
    RETURNS_DF = pd.DataFrame({sym: NAME_CACHE[sym]["simple_ret"].reindex(idx) for sym in active_names})
    CLOSE_DF = pd.DataFrame({sym: NAME_CACHE[sym]["close"].reindex(idx) for sym in active_names})
    print(f"[phase1] combined returns panel: {RETURNS_DF.shape}", flush=True)

    # Fresh, DERIVED (not hardcoded, bug pattern (b)) equal-weight unlevered daily-rebalanced
    # buy-and-hold benchmark on EXACTLY this Stage-A 22-name subset -- an apples-to-apples
    # complement to the established 93-name BENCHMARK_CAGR=0.468 (Established Fact 6), since a
    # 22-name subset's own passive benchmark CAGR need not equal the full 93-name figure.
    equal_w_ret = RETURNS_DF.mean(axis=1, skipna=True)
    global BENCHMARK_CAGR_22NAME
    BENCHMARK_CAGR_22NAME, _ = _cagr_from_returns(np.nan_to_num(equal_w_ret.to_numpy()), RETURNS_DF.index)
    print(f"[benchmark] Stage-A 22-name equal-weight unlevered buy-and-hold CAGR = "
          f"{BENCHMARK_CAGR_22NAME:.4f} (vs established 93-name benchmark {BENCHMARK_CAGR:.4f})", flush=True)

    install_axis_f_cache()
    print("[phase2] pre-warming Axis F corr/cov caches for corr_window in {60,120} ...", flush=True)
    t0 = time.time()
    for cw in (60, 120):
        _ = _cached_avg_corr(RETURNS_DF, window=cw)
        for t in range(len(RETURNS_DF)):
            _cached_shrunk_cov(RETURNS_DF, t, window=cw, shrink_intensity=0.3)
    print(f"[phase2] cache warm-up done in {time.time()-t0:.1f}s", flush=True)

    ab_pairs = list(itertools.product(AXIS_A_KEYS, AXIS_B_KEYS))
    n_f = len(AXIS_F_GRID)
    print(f"[phase2] sweeping {len(ab_pairs)} (A,B) pairs x {n_f} F x {n_oat} OAT = "
          f"{len(ab_pairs)*n_f*n_oat} rows (STAGE_A_22)", flush=True)

    all_rows: list[dict] = []
    t0 = time.time()
    CHECKPOINT_EVERY = 4
    with mp.get_context("fork").Pool(N_PROCESSES) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_ab_pair, ab_pairs, chunksize=1)):
            all_rows.extend(rows)
            if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(ab_pairs):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(ab_pairs) - (i + 1)) / rate if rate > 0 else float("nan")
                print(f"[phase2] {i+1}/{len(ab_pairs)} (A,B) pairs done, {len(all_rows)} rows, "
                      f"{elapsed:.0f}s elapsed, ETA {eta/60:.1f} min", flush=True)
                flush_checkpoint(all_rows, CHECKPOINT_PATH)

    print(f"[phase2] STAGE_A_22 sweep done in {time.time()-t0:.1f}s, {len(all_rows)} rows", flush=True)

    print("[phase3] SPY/QQQ single-instrument sweeps ...", flush=True)
    t0 = time.time()
    # PERFORMANCE FIX (Section 9-REDESIGN): SPY-alone and QQQ-alone are two fully INDEPENDENT
    # 9,920-cell computations (same math as the original script's sequential for-loop, just no
    # longer wastefully serialized onto a single core while the other vCPU sits idle) -- run them
    # concurrently via the same fork-based Pool used elsewhere in this file.
    phase3_syms = [s for s in ("SPYUSDT", "QQQUSDT") if s in NAME_CACHE]
    for s in ("SPYUSDT", "QQQUSDT"):
        if s not in NAME_CACHE:
            print(f"[phase3] WARNING: {s} not in name cache, skipping", flush=True)
    if phase3_syms:
        with mp.get_context("fork").Pool(min(N_PROCESSES, len(phase3_syms))) as pool:
            for sym, rows in zip(phase3_syms, pool.map(process_single_instrument, phase3_syms, chunksize=1)):
                all_rows.extend(rows)
                print(f"[phase3] {sym} done, {len(rows)} rows, {time.time()-t0:.0f}s elapsed", flush=True)
        flush_checkpoint(all_rows, CHECKPOINT_PATH)

    final_df = pd.DataFrame(all_rows)
    if BENCHMARK_CAGR_22NAME is not None:
        is_subset_book = final_df["universe"] == "STAGE_A_22"
        final_df["cagr_delta_vs_22name_benchmark"] = np.where(
            is_subset_book, final_df["cagr"] - BENCHMARK_CAGR_22NAME, np.nan)
    final_df.to_parquet(RESULTS_PATH, index=False)
    print(f"=== DONE: {len(final_df)} total rows -> {RESULTS_PATH} ===", flush=True)
    print(f"Expected N_cells x 3 universes = {n_core * n_oat * 3}; actual rows = {len(final_df)}",
          flush=True)
    print(f"Total wall-clock: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
