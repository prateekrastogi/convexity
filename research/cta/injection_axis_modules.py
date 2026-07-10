"""
Re-run each axis module's OWN already-coded injection test on REAL full-93-universe-representative
data (SPYUSDT/QQQUSDT/MSFTUSDT), rather than assuming pass from Stage A. Also probes Axis E (CDaR)
leverage's full-sample-calibration sensitivity to a late-sample outlier (since CDaR L is NOT a
per-day causal signal but a single constant fit on the whole realized path by design).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = "/home/speculation/convexity-cta"
sys.path.insert(0, f"{REPO_ROOT}/research/cta/lib")
sys.path.insert(0, REPO_ROOT)
import regime_hmm as rh
import vol_forecast as vf
import portfolio_construction as pc
import leverage_policy as lp

STITCHED = Path(REPO_ROOT)/"data"/"stitched"

def load_close_ohlc(symbol):
    df = pd.read_parquet(STITCHED/f"{symbol}.parquet")
    df = df[df["source"]=="duka"].copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    ohlc = vf.aggregate_daily_ohlc_from_duka(df)
    return ohlc

out = {}

# --- Axis C: HMM injection_test on real SPY log returns ---
ohlc_spy = load_close_ohlc("SPYUSDT")
close_spy = ohlc_spy["close"]
simple_ret = close_spy.pct_change()
log_ret = np.log(1.0+simple_ret).dropna()
hmm_res = rh.injection_test(log_ret, n_states=2, refit_every=63, n_restarts=3)
out["axisC_hmm_injection_SPY"] = hmm_res
print("AXIS C (HMM):", json.dumps(hmm_res, indent=2), flush=True)

# --- Axis D: vol_forecast injection test on real SPY OHLC ---
vd_res = vf.run_injection_test_ohlc_based(ohlc_spy)
out["axisD_vol_injection_SPY"] = vd_res
print("AXIS D (vol forecast):", json.dumps(vd_res, indent=2), flush=True)

# --- Axis F: correlation causality injection test (already synthetic, module's own) ---
pc_res = pc.run_causality_injection_test()
out["axisF_corr_injection_synthetic"] = pc_res
print("AXIS F (corr causality):", json.dumps(pc_res, indent=2), flush=True)

# --- Axis E: CDaR full-sample-calibration sensitivity (NOT a per-day causal test -- CDaR L is a
#     single constant fit on the WHOLE realized path by design/disclosure). Demonstrate explicitly
#     that L is NOT invariant to a late-sample (last 5%) outlier -- i.e. confirm the pipeline's own
#     "not a time-varying exposure guarantee ... computed on the FULL realized history" disclosure
#     is real: injecting a large NEGATIVE shock late in the sample changes L retroactively for
#     EVERY prior day too, which is the correct, disclosed, in-sample-calibration behavior, not a
#     per-day causality bug (there is no notion of "per-day causality" for a single scalar fit on
#     the whole path) -- but is flagged here as a real look-ahead-flavored property worth restating
#     plainly in the report.
close_msft = load_close_ohlc("MSFTUSDT")["close"]
ret_msft = close_msft.pct_change().dropna().to_numpy()
base_cdar = lp.cdar_constrained_leverage(ret_msft, budget=0.25, alpha=0.95)
shocked = ret_msft.copy()
late_idx = int(len(shocked)*0.95)
shocked[late_idx] = shocked[late_idx] - 0.15   # inject a large fake -15% shock near the END
shocked_cdar = lp.cdar_constrained_leverage(shocked, budget=0.25, alpha=0.95)
# Does the shock change L for observations BEFORE late_idx too? Since L is one constant for the
# WHOLE array, yes by construction -- demonstrate this by computing L using only data through
# late_idx-1 in both base and shocked (must be identical, since neither has seen the shock yet)
# vs the full-sample L (which differs once the shock is included).
prefix_base = lp.cdar_constrained_leverage(ret_msft[:late_idx], budget=0.25, alpha=0.95)
prefix_shocked = lp.cdar_constrained_leverage(shocked[:late_idx], budget=0.25, alpha=0.95)
out["axisE_cdar_full_sample_calibration_check"] = {
    "L_full_sample_base": base_cdar["L_closed_form"],
    "L_full_sample_after_late_shock": shocked_cdar["L_closed_form"],
    "L_changes_with_future_shock": abs(base_cdar["L_closed_form"] - shocked_cdar["L_closed_form"]) > 1e-9,
    "L_prefix_through_late_idx_minus_1_base": prefix_base["L_closed_form"],
    "L_prefix_through_late_idx_minus_1_shocked": prefix_shocked["L_closed_form"],
    "prefix_L_identical_as_expected": abs(prefix_base["L_closed_form"] - prefix_shocked["L_closed_form"]) < 1e-9,
    "interpretation": ("CDaR L is a SINGLE constant fit on the entire realized sample by explicit "
                        "design (documented in ctaB_pipeline.py's apply_leverage_cdar docstring); "
                        "a shock placed near the END of the sample changes the constant applied "
                        "to EVERY day including the very first day of the backtest. This is not a "
                        "per-day causality bug (there is no such notion for a single full-sample "
                        "scalar) but it IS a real in-sample/retrospective-calibration property: "
                        "the realized 0.9718x leverage was chosen WITH HINDSIGHT of the entire "
                        "8.17-year realized path, including 2025-2026 data. A live/paper deployment "
                        "could not have known this constant in Jan 2018; it would have had to "
                        "re-estimate CDaR on a trailing/expanding window, which was NOT done here "
                        "or in Stage A. This is disclosed in the docstring but not flagged as a "
                        "limitation in CTA_PREREG.md or the pipeline's own adoption-gate reporting."),
}
print("AXIS E (CDaR calibration):", json.dumps(out["axisE_cdar_full_sample_calibration_check"], indent=2), flush=True)

with open(f"{REPO_ROOT}/research/cta/_injection_axis_modules_results.json","w") as f:
    json.dump(out, f, indent=2, default=str)
print("DONE_AXIS_MODULES", flush=True)
