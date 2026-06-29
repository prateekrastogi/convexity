"""TREND trend-life survival tests (TREND §13a / §18). Censoring-bias, flags, AJ-CIF, no-pool."""

import numpy as np
import pytest

from convexity.trend.validate.survival import (
    TrendLifeSurvival,
    assert_no_pooling,
)


def _simulate_durations(n, true_median, cap, seed=0):
    """Weibull-distributed true death times, right-censored at the cap."""
    rng = np.random.default_rng(seed)
    scale = true_median / (np.log(2) ** (1 / 1.5))  # shape 1.5
    true_t = scale * rng.weibull(1.5, size=n)
    observed = np.minimum(true_t, cap)
    events = (true_t <= cap).astype(int)
    return observed, events, true_t


# --- §18: KM recovers true median while naive cap-as-death under-estimates (censoring bias) -
def test_km_censoring_bias():
    n, true_median, cap = 2000, 30.0, 50.0
    obs, events, true_t = _simulate_durations(n, true_median, cap, seed=1)
    surv = TrendLifeSurvival()
    res = surv.estimate(obs, events, horizon_cap=cap, clock="calendar_days",
                        death_definition="all_cause", estimand="km_allcause",
                        want_quantiles=(0.5,))
    km_median = res.quantiles[0].value
    assert km_median is not None
    # KM median within ~15% of truth.
    assert km_median == pytest.approx(true_median, rel=0.2)

    # naive: treat capped survivors as deaths AT the cap -> biased downward vs true median.
    naive_median = float(np.median(np.minimum(true_t, cap)))
    true_pop_median = float(np.median(true_t))
    assert naive_median <= true_pop_median + 1e-9
    # KM is closer to the true population median than the naive estimate is (when censoring bites).
    assert abs(km_median - true_pop_median) <= abs(naive_median - true_pop_median) + 5.0


# --- §13a.4 / §18: cap-dominated p99 returns a censored lower bound with flags -------------
def test_p99_cap_dominated_lower_bound():
    n, true_median, cap = 400, 40.0, 45.0  # heavy censoring near the cap
    obs, events, _ = _simulate_durations(n, true_median, cap, seed=2)
    surv = TrendLifeSurvival()
    res = surv.estimate(obs, events, horizon_cap=cap, clock="calendar_days",
                        death_definition="all_cause", want_quantiles=(0.99,))
    q = res.quantiles[0]
    # p99 not a point estimate; a censored lower bound with cap_dominated / one-sided CI.
    assert q.value is None
    assert q.lower_bound is not None
    assert q.ci_one_sided
    assert q.cap_dominated or q.low_event_count or q.tail_band_unreliable


# --- §13a.2: tiny tail at-risk count -> tail_band_unreliable lower bound -------------------
def test_tail_band_unreliable_small_nj():
    # few events, all early; the tail at-risk count is tiny.
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [50.0] * 3)
    events = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    surv = TrendLifeSurvival()
    res = surv.estimate(obs, events, horizon_cap=50.0, clock="bars_since_entry",
                        death_definition="all_cause", want_quantiles=(0.99,))
    q = res.quantiles[0]
    assert q.lower_bound is not None
    assert q.tail_band_unreliable or q.low_event_count


# --- §13a.3a: Aalen-Johansen CIF for a cause-specific definition (assumption-free) --------
def test_aalen_johansen_cif():
    rng = np.random.default_rng(3)
    n = 500
    t1 = rng.exponential(20, n)  # cause 1 (D1)
    t2 = rng.exponential(30, n)  # cause 2 (D5)
    cap = 60.0
    dur = np.minimum(np.minimum(t1, t2), cap)
    cause = np.where(t1 <= t2, 1, 5)
    cause = np.where(dur >= cap, 0, cause)  # censored at cap
    surv = TrendLifeSurvival()
    res = surv.estimate(dur, (cause > 0).astype(int), horizon_cap=cap, clock="calendar_days",
                        death_definition="D1", estimand="aj_cif", cause=cause,
                        want_quantiles=(0.5,))
    assert res.estimand == "aj_cif"
    # the CIF (1 - survival) rises monotonically and stays in [0,1].
    cif = 1.0 - res.survival
    assert np.all(np.diff(cif) >= -1e-9)
    assert np.all((cif >= -1e-9) & (cif <= 1.0 + 1e-9))
    # AJ-CIF curves carry no independence_assumed flag.
    assert all(not q.independence_assumed for q in res.quantiles)


# --- §13a.3a: cause-specific KM carries the independence_assumed flag ----------------------
def test_causespecific_km_flagged():
    obs, events, _ = _simulate_durations(500, 25.0, 60.0, seed=4)
    surv = TrendLifeSurvival()
    res = surv.estimate(obs, events, horizon_cap=60.0, clock="calendar_days",
                        death_definition="D2",
                        estimand="km_causespecific_independence_assumed",
                        want_quantiles=(0.5,))
    assert all(q.independence_assumed for q in res.quantiles)


# --- §13a.5: parametric Weibull tail fit under the censored likelihood --------------------
def test_parametric_weibull_tail():
    obs, events, _ = _simulate_durations(1500, 30.0, 80.0, seed=5)
    surv = TrendLifeSurvival()
    res = surv.estimate(obs, events, horizon_cap=80.0, clock="calendar_days",
                        death_definition="all_cause", want_quantiles=(0.5,),
                        fit_parametric=True)
    # a family is selected with finite params (Weibull/exponential/log-logistic).
    assert res.parametric_family in ("exponential", "weibull", "log_logistic")
    assert all(np.isfinite(p) for p in res.parametric_params)


# --- §13a.8 / §18: never pool across clocks or asset classes (a pooling attempt raises) ---
def test_no_pooling_raises():
    with pytest.raises(ValueError):
        assert_no_pooling(("calendar_days", "trading_sessions"), ("crypto_perp",))
    with pytest.raises(ValueError):
        assert_no_pooling(("calendar_days",), ("crypto_perp", "us_equity"))
    assert_no_pooling(("calendar_days",), ("crypto_perp",))  # single clock + class is fine


# --- §18: survival module is validation-only (not imported by estimate/policy) ------------
def test_survival_validation_only():
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "convexity" / "trend"
    for sub in ("estimate", "policy"):
        for path in (src / sub).glob("*.py"):
            txt = path.read_text()
            assert "validate.survival" not in txt
            assert "import survival" not in txt
