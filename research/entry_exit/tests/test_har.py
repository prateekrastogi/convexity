"""Unit tests for MODELS.har -- the HAR-RV (Corsi 2009) VOL covariate (EDGE #8). SYNTHETIC ONLY, no lake.

Two mandated synthetic tests (SYSTEM §C.8, task contract), plus conformance / role / warm-up cover:
  1. a KNOWN-generating-process RECOVERY test -- simulate an RV cascade from KNOWN true coefficients
     (a controlled HAR DGP) and assert OLS `fit` recovers `(c, beta_d, beta_w, beta_m)` within
     tolerance, and the one-step-ahead forecast matches the frozen-coefficient prediction;
  2. a NO-LOOKAHEAD PINNING test -- a SAME-BAR `RV_t` read (the classic same-bar RV lookahead) RAISES
     `PurityError` under the DAILY cadence; the live `forecast_at` path reads only `cursor.past()`.
The real Phase-C entry/exit models are NOT imported (interface-driven; SYSTEM §C.3). No lake, no
metric, no adopt decision is touched. Realized variance only -- no IV, no RV-IV spread (BARRED).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from entry_exit.base import Cadence, PurityError, ScanCursor, TrainFold
from entry_exit.har import (
    DAILY_LAG,
    MONTHLY_LAG,
    WEEKLY_LAG,
    HARCoefficients,
    HARRV,
    har_regressors,
)


# --- Synthetic HAR-RV generating process (a KNOWN-truth cascade the recovery test asserts against) --
def _simulate_har_rv(
    c: float,
    beta_d: float,
    beta_w: float,
    beta_m: float,
    n: int,
    *,
    seed: int = 0,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate an RV path from the KNOWN Corsi HAR recursion (the controlled DGP for recovery).

    Warm-starts the first `MONTHLY_LAG` days from a strictly-positive random RV level (so the monthly
    average is well-defined and the three lag regressors are not collinear), then rolls the cascade
    forward: `RV_t = c + beta_d*RV^(d) + beta_w*RV^(w) + beta_m*RV^(m) + eps_t`, with each regressor
    the trailing daily/weekly/monthly PAST-RV average. `noise` sets the innovation std (0 => an exact,
    deterministic cascade whose coefficients OLS must recover to machine precision).
    """
    rng = np.random.default_rng(seed)
    rv = np.empty(n, dtype="float64")
    rv[:MONTHLY_LAG] = 0.5 + rng.random(MONTHLY_LAG)  # positive warm-start levels in [0.5, 1.5)
    for t in range(MONTHLY_LAG, n):
        past = rv[:t]
        rv_d = past[-DAILY_LAG:].mean()
        rv_w = past[-WEEKLY_LAG:].mean()
        rv_m = past[-MONTHLY_LAG:].mean()
        eps = rng.normal(0.0, noise) if noise > 0 else 0.0
        rv[t] = c + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m + eps
    return rv


def _train_from_rv(rv: np.ndarray, warmup: int = 0) -> TrainFold:
    days = tuple(dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(rv)))
    return TrainFold(days=days, series=tuple(float(x) for x in rv), warmup=warmup)


# ==================================================================================================
# (1) KNOWN-generating-process RECOVERY test -- OLS recovers the true HAR coefficients from the cascade
# ==================================================================================================
def test_recovery_noiseless_cascade_recovers_true_coefficients():
    # Controlled DGP: an EXACT (noiseless) HAR cascade with known truth -> OLS must recover it tightly.
    c, bd, bw, bm = 0.02, 0.35, 0.30, 0.25
    rv = _simulate_har_rv(c, bd, bw, bm, n=400, seed=7, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    assert m.coef is not None
    assert m.coef.c == pytest.approx(c, abs=1e-6)
    assert m.coef.beta_d == pytest.approx(bd, abs=1e-6)
    assert m.coef.beta_w == pytest.approx(bw, abs=1e-6)
    assert m.coef.beta_m == pytest.approx(bm, abs=1e-6)


def test_recovery_noisy_cascade_recovers_coefficients_within_tolerance():
    # With small innovations OLS is unbiased, so its estimates concentrate on the true coefficients --
    # but the bands here are the estimator's HONEST per-parameter precision, NOT a seed-tuned fit.
    #
    # HAR is famously (near-)collinear: the daily / weekly / monthly trailing-RV averages overlap
    # heavily (design condition number ~195 on this DGP), so the INDIVIDUAL beta estimates carry large
    # sampling variance even at n=1500. A 200+-seed sweep of this exact DGP shows the worst-case
    # per-beta error is ~0.115 -- a hard-coded abs=0.08 band was passing only because seed=3 is a lucky
    # draw (~2% of seeds violate 0.08), so it overstated the true precision and a re-seed would flake.
    # We therefore assert per-beta at abs=0.15 (0/400 seeds violate that band) -- a statistically honest
    # bound on how tightly OLS pins each collinear coefficient. NOTE: recovery IS genuine and tight in
    # the well-conditioned directions -- the companion NOISELESS test pins all four to abs=1e-6, the
    # intercept is well-identified here (abs=0.02 across the sweep), and crucially the SUM
    # beta_d+beta_w+beta_m (the total RV persistence, the collinearity-robust combination the cascade
    # actually forecasts through) is recovered to ~0.02, an order of magnitude tighter than any single
    # beta. So we pin BOTH: the loose-but-honest per-beta band AND the tight well-conditioned aggregate.
    c, bd, bw, bm = 0.05, 0.40, 0.30, 0.20
    rv = _simulate_har_rv(c, bd, bw, bm, n=1500, seed=3, noise=0.01)
    m = HARRV().fit(_train_from_rv(rv))
    # Per-beta: honest band under HAR multicollinearity (large individual sampling variance).
    assert m.coef.beta_d == pytest.approx(bd, abs=0.15)
    assert m.coef.beta_w == pytest.approx(bw, abs=0.15)
    assert m.coef.beta_m == pytest.approx(bm, abs=0.15)
    assert m.coef.c == pytest.approx(c, abs=0.05)
    # Well-conditioned aggregate: total persistence beta_d+beta_w+beta_m is recovered far more tightly
    # than any single beta (the collinearity that inflates per-coefficient variance cancels in the sum),
    # so a tight band here is the real evidence OLS recovered the cascade's forecast dynamics.
    beta_sum = m.coef.beta_d + m.coef.beta_w + m.coef.beta_m
    assert beta_sum == pytest.approx(bd + bw + bm, abs=0.05)


def test_recovery_forecast_matches_frozen_coefficient_prediction():
    # The one-step-ahead forecast at t equals the frozen coefficients applied to the PAST-window
    # regressors -- i.e. forecast_at is exactly `coef.predict(har_regressors(rv[:t]))`, causal.
    c, bd, bw, bm = 0.01, 0.5, 0.3, 0.15
    rv = _simulate_har_rv(c, bd, bw, bm, n=300, seed=1, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    t = 250
    cur = ScanCursor(series=tuple(float(x) for x in rv), t=t, cadence=Cadence.DAILY)
    rv_d, rv_w, rv_m = har_regressors(rv[:t])
    assert m.forecast_at(cur) == pytest.approx(m.coef.predict(rv_d, rv_w, rv_m), abs=1e-12)
    # And on the noiseless cascade the forecast tracks the realized next RV (it IS the DGP).
    assert m.forecast_at(cur) == pytest.approx(float(rv[t]), abs=1e-4)


# ==================================================================================================
# (2) NO-LOOKAHEAD PINNING test -- a same-bar RV_t read is a LEAK; the live path reads only past()
# ==================================================================================================
def test_pinning_same_bar_read_raises_purity_error():
    # The same-bar RV_t probe (reading bar t for a forecast dated t) is the classic same-bar lookahead
    # and MUST RAISE under the DAILY cadence (SIGNAL §C.4 / SYSTEM §C.2).
    rv = _simulate_har_rv(0.01, 0.4, 0.3, 0.2, n=100, seed=5, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    cur = ScanCursor(series=tuple(float(x) for x in rv), t=50, cadence=Cadence.DAILY)
    with pytest.raises(PurityError):
        m._probe_same_bar_read(cur)


def test_pinning_forecast_never_reads_bar_t_or_later():
    # forecast_at must be F_{t-1}-measurable: mutating any bar >= t must NOT change the forecast
    # (it reads only series[:t]); mutating a PAST bar (< t) MUST change it (it genuinely uses the past).
    rv = _simulate_har_rv(0.01, 0.4, 0.3, 0.2, n=120, seed=9, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    t = 80
    base = tuple(float(x) for x in rv)
    f0 = m.forecast_at(ScanCursor(series=base, t=t, cadence=Cadence.DAILY))
    # Corrupt bar t and every future bar -> forecast is unchanged (they were never read).
    future_poisoned = base[:t] + tuple(9.99 for _ in base[t:])
    f_future = m.forecast_at(ScanCursor(series=future_poisoned, t=t, cadence=Cadence.DAILY))
    assert f_future == pytest.approx(f0, abs=1e-12)
    # Corrupt the most-recent PAST bar (t-1) -> forecast MUST move (the daily regressor read it).
    past_poisoned = base[: t - 1] + (base[t - 1] + 5.0,) + base[t:]
    f_past = m.forecast_at(ScanCursor(series=past_poisoned, t=t, cadence=Cadence.DAILY))
    assert f_past != pytest.approx(f0, abs=1e-9)


def test_pinning_fit_design_is_causal_no_target_row_reads_its_own_rv():
    # The OLS design is built causally: each row's regressors come from rv[:t] and the target is rv[t],
    # so no row uses its own (or a future) RV as a regressor. A same-bar-in-design leak would let a
    # perfect fit ignore the coefficients; here on a noiseless cascade the recovered coefficients are
    # the TRUE ones (not a degenerate RV_t = 1*RV_t identity), which pins the causal design.
    c, bd, bw, bm = 0.03, 0.45, 0.30, 0.20
    rv = _simulate_har_rv(c, bd, bw, bm, n=300, seed=11, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    # A same-bar identity leak would recover (c=0, bd=1, bw=0, bm=0); the true cascade is recovered.
    assert m.coef.beta_d == pytest.approx(bd, abs=1e-6)
    assert not (m.coef.beta_d == pytest.approx(1.0, abs=1e-3)
                and m.coef.c == pytest.approx(0.0, abs=1e-6)
                and m.coef.beta_w == pytest.approx(0.0, abs=1e-3))


# ==================================================================================================
# har_regressors -- the causal daily/weekly/monthly cascade averages (PAST window only)
# ==================================================================================================
def test_har_regressors_daily_weekly_monthly_averages():
    past = np.arange(1.0, 31.0)  # 30 past days: 1..30
    rv_d, rv_w, rv_m = har_regressors(past)
    assert rv_d == 30.0                                   # most-recent day
    assert rv_w == pytest.approx(np.arange(26.0, 31.0).mean())   # trailing 5: 26..30 -> 28
    assert rv_m == pytest.approx(np.arange(9.0, 31.0).mean())    # trailing 22: 9..30 -> 19.5


def test_har_regressors_partial_window_uses_available_trailing_days():
    past = np.array([2.0, 4.0, 6.0])  # fewer than a week/month -> partial trailing average
    rv_d, rv_w, rv_m = har_regressors(past)
    assert rv_d == 6.0
    assert rv_w == pytest.approx(4.0)   # mean(2,4,6)
    assert rv_m == pytest.approx(4.0)   # same (only 3 days available)


def test_har_regressors_empty_raises():
    with pytest.raises(ValueError):
        har_regressors(np.array([]))


# ==================================================================================================
# Conformance: k, fit-returns-self, frozen-before-use, warm-up honesty
# ==================================================================================================
def test_har_has_parsimony_k_of_four():
    assert HARRV.k == 4              # (c, beta_d, beta_w, beta_m) -- the SYSTEM §C.4 complexity count
    assert HARRV().k == 4


def test_fit_returns_self_and_freezes_coefficients():
    rv = _simulate_har_rv(0.02, 0.4, 0.3, 0.2, n=200, seed=2, noise=0.0)
    m = HARRV()
    assert m.coef is None
    assert m.fit(_train_from_rv(rv)) is m       # fit returns the fitted model (self)
    assert isinstance(m.coef, HARCoefficients)  # coefficients now frozen


def test_forecast_before_fit_raises():
    cur = ScanCursor(series=tuple(range(30)), t=25, cadence=Cadence.DAILY)
    with pytest.raises(RuntimeError):
        HARRV().forecast_at(cur)


def test_forecast_before_warmup_raises():
    rv = _simulate_har_rv(0.02, 0.4, 0.3, 0.2, n=200, seed=4, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    # t below the 22-day cascade warm-up -> no forecast may fire (warm-up honesty, SIGNAL §C.1).
    cur = ScanCursor(series=tuple(float(x) for x in rv), t=MONTHLY_LAG - 1, cadence=Cadence.DAILY)
    with pytest.raises(ValueError):
        m.forecast_at(cur)


def test_fit_too_short_train_raises():
    # A train fold too short to furnish >= 4 causal HAR rows is an under-determined fit -> raises.
    with pytest.raises(ValueError):
        HARRV().fit(_train_from_rv(np.linspace(0.5, 1.5, MONTHLY_LAG + 2)))


def test_fit_rank_deficient_design_raises():
    # A perfectly FLAT RV path -> the three lag regressors are identical (collinear) -> rank-deficient.
    with pytest.raises(ValueError):
        HARRV().fit(_train_from_rv(np.full(200, 1.0)))


# ==================================================================================================
# The two directional VOL roles (vol-scaling + adaptive-threshold; both consume the causal forecast)
# ==================================================================================================
def test_forecast_vol_is_sqrt_of_variance_forecast():
    c, bd, bw, bm = 0.01, 0.4, 0.3, 0.2
    rv = _simulate_har_rv(c, bd, bw, bm, n=300, seed=6, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    cur = ScanCursor(series=tuple(float(x) for x in rv), t=250, cadence=Cadence.DAILY)
    assert m.forecast_vol_at(cur) == pytest.approx(np.sqrt(max(m.forecast_at(cur), 0.0)), abs=1e-12)


def test_vol_scale_self_normalizes_to_the_forecast_regime():
    # A higher forecast vol shrinks the scaled target, a lower one grows it (vol regime self-norm).
    c, bd, bw, bm = 0.01, 0.4, 0.3, 0.2
    rv = _simulate_har_rv(c, bd, bw, bm, n=300, seed=8, noise=0.0)
    m = HARRV().fit(_train_from_rv(rv))
    cur = ScanCursor(series=tuple(float(x) for x in rv), t=250, cadence=Cadence.DAILY)
    fv = m.forecast_vol_at(cur)
    scaled = m.vol_scale(target=1.0, target_vol=fv, cursor=cur)
    assert scaled == pytest.approx(1.0, abs=1e-9)   # target_vol == forecast_vol -> scale is 1
    hi = m.vol_scale(target=1.0, target_vol=fv * 2.0, cursor=cur)
    assert hi == pytest.approx(2.0, abs=1e-9)       # reference vol doubled -> target doubles


def test_vol_scale_degenerate_forecast_falls_back_to_unscaled():
    # A zero-variance-forecast regime -> no divide-by-zero, returns the unscaled target.
    m = HARRV()
    m.coef = HARCoefficients(c=0.0, beta_d=0.0, beta_w=0.0, beta_m=0.0)  # forecast RV == 0 => vol 0
    cur = ScanCursor(series=tuple(1.0 for _ in range(40)), t=30, cadence=Cadence.DAILY)
    assert m.vol_scale(target=3.0, target_vol=0.2, cursor=cur) == 3.0


def test_vol_scale_rejects_negative_target_vol():
    m = HARRV()
    m.coef = HARCoefficients(c=1.0, beta_d=0.0, beta_w=0.0, beta_m=0.0)
    cur = ScanCursor(series=tuple(1.0 for _ in range(40)), t=30, cadence=Cadence.DAILY)
    with pytest.raises(ValueError):
        m.vol_scale(target=1.0, target_vol=-0.1, cursor=cur)


def test_adaptive_threshold_scales_by_forecast_vol():
    m = HARRV()
    m.coef = HARCoefficients(c=4.0, beta_d=0.0, beta_w=0.0, beta_m=0.0)  # forecast RV == 4 => vol 2
    cur = ScanCursor(series=tuple(1.0 for _ in range(40)), t=30, cadence=Cadence.DAILY)
    assert m.adaptive_threshold(base_threshold=5.0, cursor=cur) == pytest.approx(10.0, abs=1e-9)


def test_adaptive_threshold_degenerate_forecast_returns_base():
    m = HARRV()
    m.coef = HARCoefficients(c=0.0, beta_d=0.0, beta_w=0.0, beta_m=0.0)  # forecast vol 0
    cur = ScanCursor(series=tuple(1.0 for _ in range(40)), t=30, cadence=Cadence.DAILY)
    assert m.adaptive_threshold(base_threshold=5.0, cursor=cur) == 5.0
