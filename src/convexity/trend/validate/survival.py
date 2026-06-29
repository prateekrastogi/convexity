"""TrendLifeSurvival — censoring-correct trend-life duration + tail survival (TREND §13a).

Validation-only (Inv G): this module is NEVER imported by ``estimate/`` or ``policy/`` — it is a
measured OUTPUT of the backtest. It upgrades the bare "median hold duration" (§13) into a
censoring-correct survival curve, and answers the binding question for the cap: *does the cap
amputate the right tail that produces the convexity?* (§13a.6).

Estimators:
  * Kaplan-Meier ``S_hat(t)`` with Greenwood variance and complementary-log-log (log-minus-log) CI
    bands (§13a.2). Quantiles are read off ``S_hat`` with inverted-band CIs; reported one-sided / as
    a censored lower bound when the band is unbounded or the tail at-risk count is tiny (§13a.2).
  * Aalen-Johansen cumulative-incidence function (CIF) for cause-specific (D1..D5) estimands — the
    assumption-FREE alternative to cause-specific KM under informative competing risks (§13a.3a).
  * Optional parametric tail (exponential/Weibull/log-logistic) under the censored likelihood,
    AIC-selected and goodness-of-fit gated, every beyond-cap value tagged ``model_extrapolated``
    (§13a.5).

Honesty gates (§13a.4): event-count + censoring-fraction + band-unbounded guards downgrade a quantile
to a censored lower bound with the relevant flags. NEVER pools across horizon clocks or asset classes
(§13a.8) — a pooling attempt raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:  # scipy is available in .venv-test; it lives in validation/research only (ARCHITECTURE #8).
    from scipy.optimize import minimize
    from scipy.stats import norm

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

HorizonClock = str
DeathDefinition = Literal["D1", "D2", "D3", "D4", "D5", "all_cause"]
Estimand = Literal["km_allcause", "km_causespecific_independence_assumed", "aj_cif"]

# §13a.4 event-count honesty thresholds (observed deaths) per quantile.
_MIN_EVENTS = {0.5: 50, 0.9: 100, 0.95: 200, 0.99: 300}
_CENSOR_FRAC_CAP = 0.25  # > ~20-30% censored near tail -> cap_dominated (§13a.4)
_MIN_NJ_TAIL = 10  # at-risk count below this in the tail -> tail_band_unreliable (§13a.2)


@dataclass(frozen=True)
class QuantileResult:
    """One requested quantile: a point with CI, OR a censored lower bound, plus all §13a.7 flags."""

    q: float
    value: float | None  # point estimate (None when only a lower bound is resolvable)
    lower_bound: float | None  # censored lower bound (>= H) when the quantile is not resolvable
    ci_low: float | None
    ci_high: float | None  # None / +inf-marked when the inverted band is one-sided (§13a.2)
    ci_one_sided: bool
    death_definition: DeathDefinition
    estimand: Estimand
    clock: HorizonClock
    n_deaths: int
    n_at_risk_at_q: int
    tail_censor_fraction: float
    low_event_count: bool
    cap_dominated: bool
    model_extrapolated: bool
    independence_assumed: bool
    tail_band_unreliable: bool


@dataclass(frozen=True)
class SurvivalResult:
    """The §13a.7 contract output: the curve + the per-quantile results + the parametric-tail tag."""

    clock: HorizonClock
    death_definition: DeathDefinition
    estimand: Estimand
    times: np.ndarray
    survival: np.ndarray  # S_hat(t) for KM; 1 - CIF(t) for aj_cif
    ci_low: np.ndarray
    ci_high: np.ndarray
    quantiles: tuple[QuantileResult, ...]
    parametric_family: str | None = None
    parametric_params: tuple[float, ...] = field(default_factory=tuple)


class TrendLifeSurvival:
    """Censoring-correct trend-life survival (§13a). Validation-only (Inv G)."""

    def estimate(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        horizon_cap: float,
        clock: HorizonClock,
        death_definition: DeathDefinition,
        estimand: Estimand = "km_allcause",
        want_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
        cause: np.ndarray | None = None,
        fit_parametric: bool = False,
    ) -> SurvivalResult:
        """Estimate the survival curve + quantiles with their bands/guards (§13a.2/4/5)."""
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)
        if durations.shape != events.shape:
            raise ValueError("durations and events must align")

        if estimand == "aj_cif":
            return self._aalen_johansen(
                durations, events, cause, horizon_cap, clock, death_definition, want_quantiles
            )

        times, surv, var_s, n_at_risk, d_counts = self._kaplan_meier(durations, events)
        ci_low, ci_high = self._loglog_band(surv, var_s)

        independence_assumed = estimand == "km_causespecific_independence_assumed"
        n_deaths = int(events.sum())
        # censoring fraction near the tail (within the last 10% of observed times or at the cap).
        tail_cut = horizon_cap * 0.9
        near_tail = durations >= tail_cut
        tail_censor_fraction = (
            float(np.mean(events[near_tail] == 0)) if near_tail.any() else 0.0
        )

        quantiles = tuple(
            self._quantile(
                q, times, surv, ci_low, ci_high, n_at_risk, horizon_cap, n_deaths,
                tail_censor_fraction, death_definition, estimand, clock, independence_assumed,
            )
            for q in want_quantiles
        )

        family, params = (None, ())
        if fit_parametric:
            family, params = self._fit_parametric_tail(durations, events)

        return SurvivalResult(
            clock=clock,
            death_definition=death_definition,
            estimand=estimand,
            times=times,
            survival=surv,
            ci_low=ci_low,
            ci_high=ci_high,
            quantiles=quantiles,
            parametric_family=family,
            parametric_params=params,
        )

    # ---- Kaplan-Meier ---------------------------------------------------------------------
    @staticmethod
    def _kaplan_meier(durations, events):
        """KM ``S_hat`` + Greenwood variance over the distinct observed death times (§13a.2)."""
        order = np.argsort(durations, kind="mergesort")
        t = durations[order]
        e = events[order]
        unique_t = np.unique(t[e == 1])
        n = t.shape[0]

        times = [0.0]
        surv = [1.0]
        var_terms = 0.0
        var_s = [0.0]
        n_at_risk_list = [n]
        d_counts = [0]
        s = 1.0
        for tj in unique_t:
            n_j = int(np.sum(t >= tj))  # at risk just before tj
            d_j = int(np.sum((t == tj) & (e == 1)))
            if n_j == 0:
                continue
            s *= 1.0 - d_j / n_j
            if n_j > d_j:
                var_terms += d_j / (n_j * (n_j - d_j))
            times.append(float(tj))
            surv.append(s)
            var_s.append(s * s * var_terms)  # Greenwood
            n_at_risk_list.append(n_j)
            d_counts.append(d_j)
        return (
            np.array(times),
            np.array(surv),
            np.array(var_s),
            np.array(n_at_risk_list),
            np.array(d_counts),
        )

    @staticmethod
    def _loglog_band(surv, var_s, alpha=0.05):
        """Complementary-log-log (log-minus-log) CI band on ``S_hat`` (§13a.2). Stays in (0,1)."""
        z = 1.959963984540054 if not _HAVE_SCIPY else float(norm.ppf(1 - alpha / 2))
        ci_low = np.full_like(surv, np.nan)
        ci_high = np.full_like(surv, np.nan)
        for i, (s, vs) in enumerate(zip(surv, var_s)):
            if s <= 0 or s >= 1 or vs <= 0:
                ci_low[i] = s
                ci_high[i] = s
                continue
            log_s = np.log(s)
            # Var[log(-log S)] = Var[S] / (S log S)^2
            se = np.sqrt(vs) / abs(s * log_s)
            ll = np.log(-log_s)
            lo = np.exp(-np.exp(ll + z * se))
            hi = np.exp(-np.exp(ll - z * se))
            ci_low[i] = min(lo, hi)
            ci_high[i] = max(lo, hi)
        return ci_low, ci_high

    def _quantile(
        self, q, times, surv, ci_low, ci_high, n_at_risk, horizon_cap, n_deaths,
        tail_censor_fraction, death_definition, estimand, clock, independence_assumed,
    ) -> QuantileResult:
        """Read the p-quantile off ``S_hat`` with its inverted-band CI + honesty guards (§13a.2/4)."""
        target = 1.0 - q  # quantile time is min{t : S_hat(t) <= 1 - q}
        low_event_count = n_deaths < _MIN_EVENTS.get(q, 50)
        cap_dominated = tail_censor_fraction > _CENSOR_FRAC_CAP

        below = np.where(surv <= target)[0]
        resolvable = below.size > 0
        if resolvable:
            j = below[0]
            value = float(times[j])
            n_at_risk_at_q = int(n_at_risk[j]) if j < n_at_risk.size else 0
            tail_band_unreliable = n_at_risk_at_q < _MIN_NJ_TAIL
        else:
            value = None
            n_at_risk_at_q = int(n_at_risk[-1])
            tail_band_unreliable = True

        # Inverted-band CI: the set { t : S_hat band at t contains 1 - q }.
        band_low_reaches = np.where(ci_low <= target)[0]
        band_high_reaches = np.where(ci_high <= target)[0]
        ci_low_t = float(times[band_high_reaches[0]]) if band_high_reaches.size else None
        if band_low_reaches.size:
            ci_high_t = float(times[band_low_reaches[0]])
            ci_one_sided = False
        else:
            # the band's lower edge never drops to 1-q -> upper endpoint is +inf (§13a.2 point 1).
            ci_high_t = None
            ci_one_sided = True

        # Guards -> downgrade to a censored lower bound (§13a.4 / §13a.2 point 2).
        force_lower_bound = (
            not resolvable
            or (q >= 0.95 and (low_event_count or cap_dominated or tail_band_unreliable))
            or ci_one_sided
        )
        if force_lower_bound:
            lower_bound = float(value) if value is not None else float(horizon_cap)
            return QuantileResult(
                q=q, value=None, lower_bound=lower_bound,
                ci_low=ci_low_t, ci_high=None, ci_one_sided=True,
                death_definition=death_definition, estimand=estimand, clock=clock,
                n_deaths=n_deaths, n_at_risk_at_q=n_at_risk_at_q,
                tail_censor_fraction=tail_censor_fraction,
                low_event_count=low_event_count, cap_dominated=cap_dominated,
                model_extrapolated=False, independence_assumed=independence_assumed,
                tail_band_unreliable=tail_band_unreliable,
            )

        return QuantileResult(
            q=q, value=value, lower_bound=None,
            ci_low=ci_low_t, ci_high=ci_high_t, ci_one_sided=ci_one_sided,
            death_definition=death_definition, estimand=estimand, clock=clock,
            n_deaths=n_deaths, n_at_risk_at_q=n_at_risk_at_q,
            tail_censor_fraction=tail_censor_fraction,
            low_event_count=low_event_count, cap_dominated=cap_dominated,
            model_extrapolated=False, independence_assumed=independence_assumed,
            tail_band_unreliable=tail_band_unreliable,
        )

    # ---- Aalen-Johansen CIF ---------------------------------------------------------------
    def _aalen_johansen(
        self, durations, events, cause, horizon_cap, clock, death_definition, want_quantiles
    ) -> SurvivalResult:
        """Cause-specific cumulative-incidence via Aalen-Johansen (§13a.3a). Assumption-free.

        ``cause`` encodes the cause label per observation (0 = censored). ``death_definition`` names
        the cause of interest. CIF_c(t) = sum_{t_j<=t} S_hat(t_{j-1}) * (d_cj / n_j), where S_hat is
        the ALL-CAUSE KM (any event is a 'death' for the overall risk set).
        """
        if cause is None:
            raise ValueError("aj_cif requires a per-observation cause array (§13a.3a)")
        cause = np.asarray(cause, dtype=int)
        target_cause = {"D1": 1, "D2": 2, "D3": 3, "D4": 4, "D5": 5}.get(death_definition)
        if target_cause is None:
            raise ValueError("aj_cif estimand requires a cause-specific death definition (D1..D5)")

        any_event = (cause > 0).astype(int)
        order = np.argsort(durations, kind="mergesort")
        t = durations[order]
        c = cause[order]
        ev = any_event[order]
        unique_t = np.unique(t[ev == 1])

        times = [0.0]
        cif = [0.0]
        s_prev = 1.0
        cif_val = 0.0
        for tj in unique_t:
            n_j = int(np.sum(t >= tj))
            d_all = int(np.sum((t == tj) & (ev == 1)))
            d_c = int(np.sum((t == tj) & (c == target_cause)))
            if n_j == 0:
                continue
            cif_val += s_prev * (d_c / n_j)
            s_prev *= 1.0 - d_all / n_j
            times.append(float(tj))
            cif.append(cif_val)

        times_arr = np.array(times)
        surv = 1.0 - np.array(cif)  # report as a survival-like curve (1 - CIF)
        n_deaths = int(np.sum(c == target_cause))
        zeros = np.zeros_like(surv)
        quantiles = tuple(
            QuantileResult(
                q=q, value=None, lower_bound=float(horizon_cap),
                ci_low=None, ci_high=None, ci_one_sided=True,
                death_definition=death_definition, estimand="aj_cif", clock=clock,
                n_deaths=n_deaths, n_at_risk_at_q=0, tail_censor_fraction=0.0,
                low_event_count=n_deaths < _MIN_EVENTS.get(q, 50),
                cap_dominated=False, model_extrapolated=False,
                independence_assumed=False, tail_band_unreliable=True,
            )
            if (np.where(np.array(cif) >= q)[0].size == 0)
            else self._cif_quantile(q, times_arr, np.array(cif), n_deaths, horizon_cap,
                                     death_definition, clock)
            for q in want_quantiles
        )
        return SurvivalResult(
            clock=clock, death_definition=death_definition, estimand="aj_cif",
            times=times_arr, survival=surv, ci_low=zeros, ci_high=zeros, quantiles=quantiles,
        )

    @staticmethod
    def _cif_quantile(q, times, cif, n_deaths, horizon_cap, death_definition, clock) -> QuantileResult:
        """The time at which the CIF first reaches q (a cause-specific incidence quantile)."""
        reach = np.where(cif >= q)[0]
        value = float(times[reach[0]]) if reach.size else None
        return QuantileResult(
            q=q, value=value, lower_bound=None if value is not None else float(horizon_cap),
            ci_low=None, ci_high=None, ci_one_sided=True,
            death_definition=death_definition, estimand="aj_cif", clock=clock,
            n_deaths=n_deaths, n_at_risk_at_q=0, tail_censor_fraction=0.0,
            low_event_count=n_deaths < _MIN_EVENTS.get(q, 50),
            cap_dominated=False, model_extrapolated=False,
            independence_assumed=False, tail_band_unreliable=value is None,
        )

    # ---- parametric tail ------------------------------------------------------------------
    def _fit_parametric_tail(self, durations, events) -> tuple[str | None, tuple[float, ...]]:
        """Fit exponential/Weibull/log-logistic under the censored likelihood, AIC-selected (§13a.5).

        Returns ``(family, params)`` of the best AIC fit, or ``(None, ())`` if scipy is absent.
        """
        if not _HAVE_SCIPY:
            return None, ()
        t = np.maximum(durations, 1e-9)
        d = events.astype(float)

        def nll_exp(theta):
            lam = np.exp(theta[0])
            log_f = np.log(lam) - lam * t
            log_s = -lam * t
            return -np.sum(d * log_f + (1 - d) * log_s)

        def nll_weibull(theta):
            k = np.exp(theta[0])
            scale = np.exp(theta[1])
            log_s = -((t / scale) ** k)
            log_f = np.log(k / scale) + (k - 1) * np.log(t / scale) + log_s
            return -np.sum(d * log_f + (1 - d) * log_s)

        def nll_loglogistic(theta):
            shape = np.exp(theta[0])
            scale = np.exp(theta[1])
            z = (t / scale) ** shape
            log_s = -np.log1p(z)
            log_f = np.log(shape / scale) + (shape - 1) * np.log(t / scale) - 2 * np.log1p(z)
            return -np.sum(d * log_f + (1 - d) * log_s)

        candidates = {
            "exponential": (nll_exp, np.array([np.log(1.0 / max(t.mean(), 1e-6))]), 1),
            "weibull": (nll_weibull, np.array([0.0, np.log(max(t.mean(), 1e-6))]), 2),
            "log_logistic": (nll_loglogistic, np.array([0.0, np.log(max(np.median(t), 1e-6))]), 2),
        }
        best = (None, (), np.inf)
        for name, (nll, x0, kparams) in candidates.items():
            try:
                res = minimize(nll, x0, method="Nelder-Mead")
                aic = 2 * kparams + 2 * res.fun
                if np.isfinite(aic) and aic < best[2]:
                    best = (name, tuple(np.exp(res.x)), aic)
            except Exception:  # pragma: no cover
                continue
        return best[0], best[1]


def assert_no_pooling(clocks: tuple, asset_classes: tuple) -> None:
    """Raise if a caller tries to pool across horizon clocks or asset classes (§13a.8 / §1)."""
    if len(set(clocks)) > 1:
        raise ValueError(f"survival curves must not pool across horizon clocks: {set(clocks)}")
    if len(set(asset_classes)) > 1:
        raise ValueError(
            f"survival curves must not pool across asset classes: {set(asset_classes)}"
        )
