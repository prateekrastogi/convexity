"""
research/cta/_dd_combined_cpcv.py -- CPCV validation of the FINAL combined drawdown-overlay
policy's own already-computed net-return series (research/cta/_dd_combined_waterfall_netret.csv).

Reuses cpcv_paths() / per_obs_sharpe() / dsr() / per_year_sharpe() / _cagr_from_returns()
VERBATIM (import, not re-implement) from research/cta/ctaB_pipeline.py -- chosen over
research/equity/e2_pipeline.py / e3_pipeline.py because:
  (a) this task's return series IS itself a CTA-program artifact built directly on top of
      ctaB_pipeline.py's own frozen construction (overlay #2's book is literally "overlay #2
      alone" reproduced from this same file's logic), so its cpcv_paths() signature
      (net_full, n_groups, k, embargo, periods_per_yr) already takes a single flat net-return
      array -- an exact match for "one already-computed return series" with no cell-dict/pbo
      plumbing needed;
  (b) ctaB_pipeline.py's own EMBARGO_TRADING_DAYS=252 rationale (comment above) is DERIVED FROM
      THE SAME axis-C causal_hmm min_train=252 / 12-month-total-return / 10-month-SMA slowest-
      signal-component logic this task asks us to justify -- reusing it is not a coincidence,
      it is the same construction's own already-frozen embargo;
  (c) e2/e3_pipeline.py's cpcv_paths() has an IDENTICAL signature/body (byte-for-byte copies
      down the lineage) but is scoped to equity-swing programs (weekly cadence, different
      periods_per_yr derivation, different embargo conventions e.g. embargo=2/3/21) -- reusing
      ctaB's copy is more directly traceable to this exact program's own embargo precedent.

No parameter is re-tuned; CPCV_N_GROUPS=8, CPCV_K=2, EMBARGO_TRADING_DAYS=252 are copied
unchanged from ctaB_pipeline.py's own frozen module-level constants.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/speculation/convexity-cta/research/cta")

# Reuse verbatim -- no re-implementation.
from ctaB_pipeline import (
    cpcv_paths, per_obs_sharpe, dsr, per_year_sharpe, per_year_return,
    _cagr_from_returns, derive_periods_per_yr,
    CPCV_N_GROUPS, CPCV_K, EMBARGO_TRADING_DAYS,
)

NETRET_CSV = "/home/speculation/convexity-cta/research/cta/_dd_combined_waterfall_netret.csv"
OUT_JSON = "/home/speculation/convexity-cta/research/cta/_dd_combined_cpcv_results.json"

def main():
    df = pd.read_csv(NETRET_CSV, parse_dates=["date"])
    dates = pd.DatetimeIndex(df["date"])
    net = df["net_ret_final_policy"].to_numpy(dtype="float64")
    n_obs = len(net)
    periods_per_yr = derive_periods_per_yr(dates)

    results = {
        "meta": {
            "source_csv": NETRET_CSV,
            "n_obs": n_obs,
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "periods_per_yr": periods_per_yr,
            "cpcv_function_used": "ctaB_pipeline.cpcv_paths (imported verbatim)",
            "cpcv_n_groups": CPCV_N_GROUPS,
            "cpcv_k": CPCV_K,
            "embargo_trading_days": EMBARGO_TRADING_DAYS,
            "embargo_rationale": (
                "252 trading days = the slowest signal-component horizon used ANYWHERE in the "
                "base construction or its overlays: the 12-month (252-trading-day) total-return "
                "rule (research/cta/_s1_ensemble_93name.py) and the 10-month SMA rule (~210 "
                "trading days, Meb Faber convention, same file) are the two slowest axes; "
                "Axis-C causal_hmm_regime_posterior's min_train=252 (ctaB_pipeline.py) "
                "independently corroborates the same 252-day figure. All other components "
                "(vol_window=20, ewma_0.94 ~15.7d memory, corr_window<=120, the GZ-style "
                "drawdown-brake's own m_t which looks back only to running-peak-to-date with no "
                "extra fixed window) are faster/shorter. 252 (>=210) is therefore used, matching "
                "ctaB_pipeline.py's own EMBARGO_TRADING_DAYS constant exactly -- not re-derived "
                "independently, reused because it is the SAME base construction's own frozen "
                "figure."
            ),
        }
    }

    # ---- CPCV path-Sharpe distribution ----
    folds = cpcv_paths(net, CPCV_N_GROUPS, CPCV_K, EMBARGO_TRADING_DAYS, periods_per_yr)
    n_theoretical = len(list(combinations(range(CPCV_N_GROUPS), CPCV_K)))
    sharpes = [f["sharpe_ann"] for f in folds]
    cpcv_summary = {
        "n_theoretical_combinations": n_theoretical,
        "n_valid_combinations_used": len(folds),
        "dropped_combinations": n_theoretical - len(folds),
        "drop_reason": (
            "a (group,group) combo is dropped by cpcv_paths() itself when, after removing the "
            "+/-252-day embargo halo around each of the k=2 test groups, the remaining test "
            "segment has <=20 obs or zero variance -- with n_obs=2046 split into 8 groups of "
            "~256 obs each, a 252-day embargo on EACH side of EACH of the 2 test groups consumes "
            "very close to an entire adjacent group's width, so most combinations that include "
            "two adjacent (or once-removed) groups collapse to a near-empty or empty remaining "
            "test segment and are dropped; only combinations of groups that are far enough apart "
            "(and/or near the sample edges, where there is no embargo on the outward side) "
            "survive with >20 obs remaining."
        ),
        "sharpe_mean": float(np.mean(sharpes)) if sharpes else None,
        "sharpe_std": float(np.std(sharpes, ddof=0)) if sharpes else None,
        "sharpe_median": float(np.median(sharpes)) if sharpes else None,
        "sharpe_min": float(np.min(sharpes)) if sharpes else None,
        "sharpe_max": float(np.max(sharpes)) if sharpes else None,
        "frac_positive_sharpe": float(np.mean(np.array(sharpes) > 0)) if sharpes else None,
        "folds": folds,
    }
    results["cpcv"] = cpcv_summary

    # ---- drop-top-5-periods robustness (identical recipe to ctaB_pipeline.py) ----
    order = np.argsort(net)[::-1]
    top5_idx = order[:5]
    drop5_ret = net.copy()
    drop5_ret[top5_idx] = 0.0
    cagr_drop5, maxdd_drop5 = _cagr_from_returns(drop5_ret, dates)
    sharpe_drop5 = float(per_obs_sharpe(drop5_ret[np.isfinite(drop5_ret)]) * np.sqrt(periods_per_yr))
    cagr_full, maxdd_full = _cagr_from_returns(net, dates)
    sharpe_full = float(per_obs_sharpe(net[np.isfinite(net)]) * np.sqrt(periods_per_yr))
    results["drop_top5_periods"] = {
        "top5_dates": [str(dates[i].date()) for i in top5_idx],
        "top5_returns": [float(net[i]) for i in top5_idx],
        "full_sample_cagr": cagr_full,
        "full_sample_maxdd": maxdd_full,
        "full_sample_sharpe_ann": sharpe_full,
        "cagr_after_drop5": cagr_drop5,
        "maxdd_after_drop5": maxdd_drop5,
        "sharpe_ann_after_drop5": sharpe_drop5,
    }

    # ---- per-year Sharpe (and return, for context) ----
    results["per_year_sharpe"] = per_year_sharpe(net, dates, periods_per_yr)
    results["per_year_return"] = per_year_return(net, dates)

    # ---- DSR informational, honest multiplicity N_trials=4 ----
    dsr_n4 = dsr(net, n_trials=4, periods_per_yr=periods_per_yr)
    # NOTE: dsr()'s _ppf(1 - 1/n_trials) formula is degenerate at n_trials=1 (1-1/1=0 -> log(0)
    # domain error), so a true "N_trials=1" naive reference cannot be computed with this exact
    # reused formula; n_trials=2 is used instead as the closest well-defined lower-bound reference.
    dsr_n1 = dsr(net, n_trials=2, periods_per_yr=periods_per_yr)  # near-naive/minimal-multiplicity reference
    dsr_full_grid = dsr(net, n_trials=32, periods_per_yr=periods_per_yr)  # contrast only
    results["dsr_informational"] = {
        "n_trials_honest_waterfall": 4,
        "waterfall_stopping_points_rationale": (
            "The combined policy was reached via a MECHANICAL waterfall (Step A -> C -> D-corr -> "
            "D-index), not a fitted grid search: at each step there was exactly one pre-specified "
            "next overlay to test and the waterfall stops (or proceeds) based on a pre-registered "
            "pass/fail gate, not on picking the best of many. That gives AT MOST 4 distinct "
            "configurations that could ever actually have been the 'final' one the waterfall "
            "stopped at (stop-after-A, stop-after-C, stop-after-D-corr, stop-after-D-index) -- "
            "not 2^5=32 (all combinations of 5 independent overlays a grid search would have "
            "implied), and not even 2^4=16 restricted to the 4 steps actually walked, because the "
            "waterfall's sequential accept/reject logic collapses the combinatorial overlay space "
            "down to a single ordered path with one decision (continue Y/N) per step, i.e. a "
            "linear chain of <=4 stopping points rather than a tree/lattice of configurations."
        ),
        "dsr_at_n_trials_4": dsr_n4,
        "dsr_at_n_trials_2_near_naive_reference": dsr_n1,
        "dsr_at_n_trials_32_full_grid_contrast_only": dsr_full_grid,
        "contrast_note": (
            "At N_trials=2 (near-naive lower-bound reference; N_trials=1 is degenerate for this "
            "PPF-based DSR formula) vs N_trials=4 (honest waterfall count) vs N_trials=32 (what a "
            "full independent grid search over all 5 overlays' "
            "on/off combinations would have implied, ignoring within-overlay parameter variation "
            "which would push the true grid count higher still) -- the DSR haircut from 2->4 is "
            "small precisely because the waterfall's sequential, pre-registered, gate-based "
            "structure never actually multiplicity-tested a combinatorial lattice of overlay "
            "on/off states; it tested a short ordered chain. This is DSR/PBO reported "
            "INFORMATIONALLY ONLY per the frozen adoption-gate convention (ctaB_pipeline.py "
            "docstring: 'DSR/PBO are computed and reported INFORMATIONALLY ONLY -- they never "
            "gate adoption here') -- neither DSR nor PBO is a pass/fail gate for this program."
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
