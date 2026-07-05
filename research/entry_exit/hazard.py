"""MODELS.hazard -- Phase-C candidate #7: discrete-time survival/hazard reversal-exit (an ExitModel).

The EXIT / dynamic-HOLD candidate #7 of the frozen ledger (docs/EDGE.md "The pre-registered candidate
set (frozen)" #7, SIGNAL §C.3(c), SYSTEM §C.9 #7): a **discrete-time survival / hazard duration model
of the REVERSAL event** whose dynamic hold is a **train-frozen reversal-hazard threshold crossing** --
"the hold found by the data." It REPLACES the incumbent's fixed 5-day horizon while the true -12%
1-minute premium stop is RETAINED as the loss floor (``base.STOP_FLOOR``, never loosened -- "the stop
IS the edge", SIGNAL §C.3): this model only decides WHEN a still-alive winner is voluntarily closed.

THE FORMAL MODEL (stated assumptions / estimator / error-control -- a formal survival model, NOT a
technical indicator):

  * *Event & duration.* The trade's time-to-adverse-reversal is a duration ``T`` measured in elapsed
    lake session days ``u = 0, 1, 2, ...`` (``u = 0`` = entry day; ``u`` is EXACTLY the ExitModel
    interface covariate ``held_days``, so the incumbent's ``hold == 5`` is the degenerate constant
    hazard that fires at ``u == 5``). Discrete time is the correct formalism: the hold is a handful of
    discrete daily decisions, not a continuous clock, so a discrete-time hazard (Cox 1972 discrete
    proportional-hazards / a discrete Weibull-AFT time basis) -- NOT a continuous Cox partial
    likelihood -- is the exact representative.
  * *Hazard.* The discrete-time (conditional) hazard is
        ``h(u | x) = P(reversal at duration u | survived to u, x) = logistic(η(u, x))``
    with a **complementary-log link's logistic cousin** -- the canonical grouped-survival logit link
    (Cox 1972; Allison discrete-time survival) --
        ``η(u, x) = β0 + β_dur * g(u) + β · x``,   ``g(u) = log1p(u)``   (log-time duration basis),
    where ``x`` are ``F_{t-1}``-measurable covariates (the filtered trend strength on the CAUSAL past
    window, the running ``held_return``; a HAR-RV level may be appended, SIGNAL §C.4). ``g(u) =
    log1p(u)`` is the discrete-Weibull/AFT log-duration basis: ``β_dur > 0`` ⇒ hazard MONOTONE
    INCREASING in ``u`` (positive duration dependence -- a winner's reversal becomes more likely the
    longer it has run), which yields a well-defined threshold crossing (the monotonic-hold sanity).
  * *Estimator (fit on TRAIN, then FROZEN).* Person-period (discrete-survival) maximum likelihood:
    each observed duration is expanded into ``u = 0..T`` Bernoulli person-period rows (survive,
    survive, ..., reversal), the (duration-constant, per-trade) covariates ``x`` are broadcast across
    that trade's rows, and the pooled Bernoulli log-likelihood over the FULL design
    ``[1, log1p(u), x...]`` is maximized by **Newton-Raphson / IRLS** (pure numpy, scipy-free --
    matching ``system/__init__.py``). The FULL ``(β0, β_dur, β_1..β_p)`` -- duration AND covariate
    coefficients -- are estimated jointly and frozen for OOS; NO re-fit or threshold re-tune on TEST
    (SIGNAL §C.1, SYSTEM §C.2). Ridge-damped Newton keeps the fit well-posed on thin / separable folds.
    The covariate design is supplied to ``fit`` as per-trade rows (one row per observed duration) in
    the SAME covariate order the model reads at decision time; when no covariates are enabled the
    design collapses to the duration-only ``[1, log1p(u)]`` basis.
  * *Signal (the dynamic hold).* At each still-alive holding day, compute the **one-step-ahead**
    reversal hazard ``h(u | x_{t-1})`` from ``F_{t-1}`` information ONLY and EXIT when it exceeds the
    **train-frozen exit-hazard threshold** ``h*`` (the hold is no longer favored). This is a
    *predictive* hazard evaluated at the current bar, NEVER a full-sample fitted survival curve
    (which conditions on the whole path = SMOOTHED = lookahead, barred SIGNAL §C.1). The threshold
    ``h*`` is itself train-fit (a frozen quantile of the per-trade train reversal hazards ``h(T_i|x_i)``
    -- ONE hazard per observed trade, so a single long duration cannot survival-time-weight the
    threshold) -- part of the model, not a TEST-tuned knob.
  * *Monotone-hold sanity (enforced, not merely assumed).* The dynamic-hold contract (EDGE #7) rests
    on ``β_dur >= 0`` (hazard monotone increasing in ``u`` ⇒ a single HOLD→EXIT crossing). A fold with
    NEGATIVE duration dependence would fit ``β_dur < 0`` and silently INVERT the exit logic (exit
    early, hold longer as the trade ages). ``fit`` therefore GUARDS the sign: with
    ``require_monotone_hold=True`` (default) a fitted ``β_dur < 0`` RAISES ``ValueError`` naming the
    violation, so a decreasing-hazard fit is never shipped as a reversal-exit.

PURITY (binding, SIGNAL §C.1 / SYSTEM §C.2). ``exit_now`` reads ONLY ``held_days`` (the elapsed
duration ``u``, a count of already-elapsed bars, ``F_{t-1}``-measurable by construction),
``held_return`` (the running return known at the decision), and ``cursor.past()`` (the strictly-causal
``series[:t]`` window -- ``F_{t-1}``-measurable). The live decision routes its causal touch through
``cursor.read(cursor.t - 1)`` -- the SAME ``require_causal`` guard the other exit candidates use
(``ou_exit`` / ``stopping``) -- so any leaky refactor that reads bar ``>= t`` RAISES ``PurityError``
and breaks the pinning test. The filtered trend strength is a backward EW mean of the causal past --
a filtered ``E[·|F_{t-1}]`` estimate, never a smoothed ``E[·|F_T]`` one. ``_probe_same_bar_read`` is
the explicit pinning hook the no-lookahead test drives (a same-bar read RAISES).

DATA GATE (SYSTEM §C.8). This module is the candidate CODE, synthetic-tested via known-generating-
process recovery ONLY; it touches NO lake, emits NO metric, and adopts NOTHING. The empirical
selection RUN is BLOCKED-ON-LAKE and lives in the harness (which owns ``BlockedOnLakeError``); nothing
here raises it (this module has no data path). ``k`` is the parsimony count fed to SYSTEM §C.4.

Imports ONLY ``intraday.models.base`` (the shared interface + causal helpers) -- NEVER a sibling
Phase-C model module (SYSTEM §C.3 interface-driven-harness / off-ledger rule). Pure numpy + stdlib;
moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .base import STOP_FLOOR, ScanCursor, TrainFold, require_causal

__all__ = [
    "DiscreteHazardFit",
    "HazardReversalExit",
    "hazard_prob",
    "simulate_reversal_durations",
]

# A holding day survives with hazard `h`; below this the hazard is numerically indistinguishable from
# 0 (no reversal ever) and above `1 - _H_EPS` from a certain reversal -- clamp so log-likelihood and
# the threshold crossing stay finite (mirrors `system`'s `_STD_EPS` numerical-floor pattern).
_H_EPS = 1e-12
# Ridge damping added to the Newton Hessian diagonal: keeps the IRLS solve well-posed on a thin or
# quasi-separable train fold (a pure-MLE Hessian can be singular when a covariate perfectly splits
# survive/reversal). Small enough not to bias a well-identified fit (a weak Gaussian coefficient prior).
_RIDGE = 1e-6
_MAX_NEWTON = 100
_NEWTON_TOL = 1e-10


def _logistic(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically-stable standard logistic ``1 / (1 + e^-z)`` (the discrete-hazard link inverse)."""
    return np.where(np.asarray(z) >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def hazard_prob(coef: Sequence[float], u: int, covariates: Sequence[float] = ()) -> float:
    """The discrete-time reversal hazard ``h(u | x) = logistic(β0 + β_dur*log1p(u) + β·x)``.

    ``coef = (β0, β_dur, β_1, ..., β_p)`` (intercept, log-duration coefficient, then one coefficient
    per covariate); ``u`` is the elapsed holding time (session days since entry, ``u >= 0``);
    ``covariates`` are the ``p`` ``F_{t-1}``-measurable covariate values (``len == len(coef) - 2``).
    Returned in the open interval ``(0, 1)`` (clamped by ``_H_EPS``) so it is always a valid
    probability. This is the SINGLE definition of the hazard, shared by the fitter, the recovery
    simulator, and ``exit_now`` -- so the model that decides is exactly the model that was fit.
    """
    if u < 0:
        raise ValueError("u (elapsed holding time) must be non-negative")
    if len(coef) != 2 + len(covariates):
        raise ValueError("coef must be (β0, β_dur) followed by one coefficient per covariate")
    eta = coef[0] + coef[1] * math.log1p(u) + sum(b * x for b, x in zip(coef[2:], covariates))
    h = 1.0 / (1.0 + math.exp(-eta)) if eta >= 0.0 else math.exp(eta) / (1.0 + math.exp(eta))
    return min(max(h, _H_EPS), 1.0 - _H_EPS)


class DiscreteHazardFit:
    """The frozen result of the person-period discrete-hazard MLE (train-fit, then immutable).

    Holds the estimated ``coef`` (``β0, β_dur, β...`` -- duration AND covariate coefficients, jointly
    estimated) and the train-frozen exit-hazard threshold ``h_star``. Kept as a small value object so
    ``HazardReversalExit`` can hold ONE frozen fit and a conformance/recovery test can assert on the
    recovered coefficients directly.
    """

    __slots__ = ("coef", "h_star", "n_covariates")

    def __init__(self, coef: Sequence[float], h_star: float, n_covariates: int) -> None:
        self.coef = tuple(float(c) for c in coef)
        self.h_star = float(h_star)
        self.n_covariates = int(n_covariates)


def _expand_person_period(durations: Sequence[int], covariates: Sequence[Sequence[float]] | None):
    """Expand observed reversal durations into person-period Bernoulli rows (discrete survival).

    A trade observed to reverse at duration ``T`` contributes rows ``u = 0..T``: the outcome is 0
    (survived this day) for ``u = 0..T-1`` and 1 (reversal) at ``u = T``. Each row's design vector is
    ``[1, log1p(u), x...]`` -- the intercept, the log-duration basis, and the (duration-CONSTANT
    per-trade) covariates broadcast across that trade's rows. Returns ``(X, y, row_trade)`` where
    ``row_trade[j]`` is the index of the trade that generated design row ``j`` (so the fitter can
    recover ONE per-trade hazard for the threshold, un-weighted by survival time). This is the
    standard Allison discrete-time-survival person-period expansion; it is the exact likelihood a
    continuous Cox reduces to when ties are grouped into daily intervals.
    """
    rows: list[list[float]] = []
    ys: list[float] = []
    row_trade: list[int] = []
    cov = covariates if covariates is not None else [() for _ in durations]
    if len(cov) != len(durations):
        raise ValueError("covariates must have one row per observed duration")
    for i, (t, xs) in enumerate(zip(durations, cov)):
        if t < 0:
            raise ValueError("each observed duration must be non-negative")
        for u in range(t + 1):
            rows.append([1.0, math.log1p(u), *[float(x) for x in xs]])
            ys.append(1.0 if u == t else 0.0)
            row_trade.append(i)
    return (np.asarray(rows, dtype="float64"), np.asarray(ys, dtype="float64"),
            np.asarray(row_trade, dtype="int64"))


def _fit_logistic_newton(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge-damped Newton-Raphson / IRLS logistic MLE (pure numpy, scipy-free).

    Maximizes the pooled Bernoulli log-likelihood ``Σ y·log p + (1-y)·log(1-p)`` with
    ``p = logistic(Xβ)``. Newton step: ``β <- β + (XᵀWX + ridge·I)^{-1} Xᵀ(y - p)`` with
    ``W = diag(p(1-p))``. The tiny ridge (a weak Gaussian coefficient prior) keeps the Hessian
    invertible on a thin / quasi-separable train fold; it does not bias a well-identified fit. Iterates
    to a gradient/step tolerance or ``_MAX_NEWTON``. Scipy-free by design (the repo has no scipy;
    ``system/__init__.py`` implements its own numerics for the same reason).
    """
    n_features = x.shape[1]
    beta = np.zeros(n_features, dtype="float64")
    ridge_i = _RIDGE * np.eye(n_features)
    for _ in range(_MAX_NEWTON):
        p = _logistic(x @ beta)
        p = np.clip(p, _H_EPS, 1.0 - _H_EPS)
        grad = x.T @ (y - p)
        w = p * (1.0 - p)
        hess = x.T @ (x * w[:, None]) + ridge_i
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
        if float(np.max(np.abs(step))) < _NEWTON_TOL:
            break
    return beta


def simulate_reversal_durations(
    coef: Sequence[float],
    n: int,
    *,
    seed: int = 0,
    max_u: int = 200,
    covariates: Sequence[Sequence[float]] | None = None,
) -> list[int]:
    """Simulate reversal durations from a KNOWN discrete-hazard DGP (for the recovery test).

    Draws ``n`` durations from the discrete-time hazard ``h(u | x)`` of ``coef``: at each ``u`` the
    trade reverses with probability ``h(u|x)`` (a Bernoulli sequential-trials survival draw), so the
    simulated durations are, by construction, generated by exactly the model the fitter estimates --
    a controlled truth the recovery test asserts the fit recovers. When ``covariates`` is given (one
    row per simulated trade, ``len == n``), each trade's duration is drawn from ITS OWN covariate-
    conditioned hazard, so a covariate-recovery test can inject a true nonzero covariate effect and
    assert the fit recovers it. ``max_u`` caps a pathological never-reversing draw. This is a
    SYNTHETIC generator: no lake, unmistakably a test fixture (SYSTEM §C.8). Deterministic given
    ``seed``.
    """
    rng = np.random.default_rng(seed)
    cov = covariates if covariates is not None else [() for _ in range(n)]
    if len(cov) != n:
        raise ValueError("covariates, when given, must have exactly n rows (one per simulated trade)")
    out: list[int] = []
    for i in range(n):
        xs = cov[i]
        u = 0
        while u < max_u:
            if rng.random() < hazard_prob(coef, u, xs):
                break
            u += 1
        out.append(u)
    return out


class HazardReversalExit:
    """Phase-C candidate #7 -- discrete-time survival/hazard REVERSAL-exit ExitModel (SIGNAL §C.3(c)).

    Conforms to ``models.base.ExitModel`` (``fit -> frozen``; causal ``exit_now``). The dynamic hold is
    a train-frozen reversal-hazard threshold crossing: HOLD while the one-step-ahead reversal hazard
    ``h(u | x_{t-1})`` is at/below the frozen ``h*``, EXIT the instant it exceeds ``h*``.

    ``k`` (parsimony, SYSTEM §C.4) = the number of FREE FITTED PARAMETERS = ``2 + n_covariates`` (the
    intercept ``β0``, the log-duration coefficient ``β_dur``, and one JOINTLY-ESTIMATED coefficient
    per covariate) PLUS ``1`` for the fitted exit-hazard threshold ``h*`` -- so ``k = 3 +
    n_covariates``. Because the covariate coefficients are genuinely fitted (not frozen at 0), ``k``
    faithfully counts the model's free flexibility. The incumbent's fixed 5-day hold is the
    ``k``-minimal parsimony floor; this richer model must beat it OOS net of the ``k``-penalty.

    THE -12% STOP IS RETAINED (``base.STOP_FLOOR``): ``exit_now`` returns ``True`` only to voluntarily
    CLOSE a still-alive winner earlier than the fixed horizon; it never touches / loosens the loss cap
    (a conformance test asserts the model does not reference or raise the floor). The concrete numerics
    live here and are synthetic-tested via known-DGP recovery (including covariate-coefficient
    recovery), no lake touched.

    Parameters
    ----------
    exit_hazard_quantile : the train quantile of the PER-TRADE fitted reversal hazards used to freeze
        ``h*`` (a train-fit level, NOT a TEST-tuned knob) -- the default ``0.5`` (median per-trade
        hazard) means "exit once this bar's reversal risk exceeds the typical train reversal risk."
        ONE hazard per observed trade (``h(T_i | x_i)``) feeds the quantile, so a single long duration
        does not survival-time-weight the threshold. Frozen at fit.
    trend_halflife : EW half-life (in causal past bars) of the FILTERED trend-strength covariate built
        from ``cursor.past()`` -- a filtered ``E[·|F_{t-1}]`` estimate (a backward EW mean), never a
        smoothed one. ``0`` disables the trend covariate (pure duration + running-return model).
    use_return_covariate : include ``held_return`` as a covariate (a value-function / hazard covariate,
        SIGNAL §C.3(c)); default on.
    require_monotone_hold : if ``True`` (default), a fitted ``β_dur < 0`` (a DECREASING hazard, which
        would invert the reversal-exit logic) RAISES ``ValueError`` at fit time -- the monotone-hold
        sanity is enforced, not merely assumed (EDGE #7). Set ``False`` only in a diagnostic that
        explicitly wants to inspect a non-monotone fit.
    """

    def __init__(
        self,
        *,
        exit_hazard_quantile: float = 0.5,
        trend_halflife: float = 0.0,
        use_return_covariate: bool = True,
        require_monotone_hold: bool = True,
    ) -> None:
        if not 0.0 < exit_hazard_quantile < 1.0:
            raise ValueError("exit_hazard_quantile must lie strictly in (0, 1)")
        if trend_halflife < 0.0:
            raise ValueError("trend_halflife must be non-negative (a causal EW half-life in bars)")
        self.exit_hazard_quantile = float(exit_hazard_quantile)
        self.trend_halflife = float(trend_halflife)
        self.use_return_covariate = bool(use_return_covariate)
        self.require_monotone_hold = bool(require_monotone_hold)
        self._n_cov = int(use_return_covariate) + int(trend_halflife > 0.0)
        # k = fitted coefficients (2 + n_cov, covariate coefs GENUINELY estimated) + the fitted exit
        # threshold h* (SYSTEM §C.4 parsimony) -- each counted param is a real degree of freedom.
        self.k = 3 + self._n_cov
        self._fit: DiscreteHazardFit | None = None

    # -- causal covariate builder (F_{t-1}-measurable only) ----------------------------------------
    def _trend(self, cursor: ScanCursor) -> float:
        """Filtered trend strength = backward EW mean of the CAUSAL past window ``cursor.past()``.

        A filtered ``E[·|F_{t-1}]`` estimate: it reads ONLY ``series[:t]`` (indices ``0..t-1``), never
        bar ``t`` or later, so it cannot leak (ScanCursor.past is ``F_{t-1}``-measurable by
        construction). The EW weighting (recent past weighted up by ``trend_halflife``) is the causal
        one-sided filter; a two-sided/full-sample smooth would be lookahead and is not used.
        """
        past = cursor.past()
        if not past or self.trend_halflife <= 0.0:
            return 0.0
        decay = 0.5 ** (1.0 / self.trend_halflife)
        # Most recent past bar (index t-1) gets weight 1, older bars decay -- a backward EW mean.
        w = np.array([decay ** i for i in range(len(past))][::-1], dtype="float64")
        vals = np.asarray(past, dtype="float64")
        return float((w * vals).sum() / w.sum())

    def _covariates(self, cursor: ScanCursor, held_return: float) -> tuple[float, ...]:
        """Assemble the ``F_{t-1}``-measurable covariate vector for the current decision bar.

        Order MUST match the per-trade covariate rows passed to ``fit`` (and thus
        ``_expand_person_period``): ``[held_return?, trend?]``. Every read is causal -- ``held_return``
        is a decision-time scalar and ``_trend`` reads ``cursor.past()`` only; no bar ``>= t`` is read,
        so the pinning test cannot be tripped by this builder.
        """
        xs: list[float] = []
        if self.use_return_covariate:
            xs.append(float(held_return))
        if self.trend_halflife > 0.0:
            xs.append(self._trend(cursor))
        return tuple(xs)

    def fit(
        self,
        train: TrainFold,
        *,
        covariates: Sequence[Sequence[float]] | None = None,
    ) -> HazardReversalExit:
        """Fit the FULL discrete-hazard coefficients + freeze ``h*`` on ``train`` ONLY (SIGNAL §C.1).

        ``train.series`` carries the observed per-trade REVERSAL DURATIONS (elapsed session days to
        the adverse reversal, one per train trade) -- the survival model's response. ``covariates``
        (optional) carries the per-trade covariate ROWS, one row per observed duration, in the SAME
        covariate order the model reads at decision time (``[held_return?, trend?]``, i.e. exactly the
        ``self._n_cov`` covariates enabled by ``use_return_covariate`` / ``trend_halflife``). The
        durations are expanded into person-period Bernoulli rows, the covariates are broadcast across
        each trade's rows, and the FULL design ``[1, log1p(u), x...]`` is fit by ridge-damped Newton
        (``_fit_logistic_newton``) -- so the duration coefficient ``β_dur`` AND every covariate
        coefficient ``β_1..β_p`` are estimated JOINTLY (the covariates are NOT frozen at 0).

        When ``covariates is None`` and covariates ARE enabled, the model has no train covariate
        response to estimate against, so it RAISES -- the caller must supply the per-trade covariate
        history (the empirical study builds it causally from each train trade's ``F_{t-1}`` info;
        SIGNAL §C.3(c)). When no covariates are enabled the design collapses to the duration-only
        ``[1, log1p(u)]`` basis and ``covariates`` must be ``None``.

        ``h*`` is frozen as the requested quantile of the PER-TRADE fitted reversal hazards
        ``h(T_i | x_i)`` (ONE hazard per observed trade -- so a single long duration does not
        survival-time-weight the threshold). With ``require_monotone_hold`` (default) a fitted
        ``β_dur < 0`` (a DECREASING hazard, which would invert the reversal-exit) RAISES. All
        parameters are then IMMUTABLE for OOS (no TEST re-fit, SIGNAL §C.1 / SYSTEM §C.2).
        """
        durations = [int(round(v)) for v in train.series[train.warmup:]]
        if not durations:
            raise ValueError("HazardReversalExit.fit needs >= 1 observed reversal duration on train")
        if any(d < 0 for d in durations):
            raise ValueError("observed reversal durations must be non-negative session-day counts")
        cov_rows = self._validate_covariate_design(covariates, len(durations))
        x, y, row_trade = _expand_person_period(durations, covariates=cov_rows)
        beta = _fit_logistic_newton(x, y)  # -> (β0, β_dur, β_1..β_p) jointly estimated
        if self.require_monotone_hold and beta[1] < 0.0:
            raise ValueError(
                f"fitted β_dur={float(beta[1]):.4g} < 0: the reversal hazard would DECREASE in u "
                "(exit-early / hold-longer -- the INVERSE of a reversal-exit). This fold has negative "
                "duration dependence; the monotone-hold sanity (EDGE #7 / SIGNAL §C.3(c)) is violated. "
                "Pass require_monotone_hold=False only for an explicit diagnostic inspection.")
        coef = tuple(float(b) for b in beta)
        # Per-trade hazard at each trade's OBSERVED reversal duration h(T_i | x_i): one value per
        # trade (the last person-period row of each trade), never survival-time-weighted by the row
        # count -- so a single long duration cannot dominate the threshold quantile.
        row_p = np.clip(_logistic(x @ beta), _H_EPS, 1.0 - _H_EPS)
        last_row = np.array([np.max(np.flatnonzero(row_trade == i)) for i in range(len(durations))])
        per_trade_hazards = row_p[last_row]
        h_star = float(np.quantile(per_trade_hazards, self.exit_hazard_quantile))
        self._fit = DiscreteHazardFit(coef=coef, h_star=h_star, n_covariates=self._n_cov)
        return self

    def _validate_covariate_design(
        self, covariates: Sequence[Sequence[float]] | None, n_trades: int
    ) -> Sequence[Sequence[float]] | None:
        """Validate the per-trade covariate design against the model's enabled covariate count.

        Returns the covariate rows to feed the person-period expansion (``None`` for the duration-only
        design). RAISES if the enabled/​supplied covariate shape is inconsistent -- so the fit never
        silently drops a covariate the model claims to use (the D1 facade: a covariate that never
        affects the decision).
        """
        if self._n_cov == 0:
            if covariates is not None:
                raise ValueError(
                    "no covariates are enabled (use_return_covariate=False, trend_halflife=0) but a "
                    "covariate design was passed to fit -- enable a covariate or pass covariates=None")
            return None
        if covariates is None:
            raise ValueError(
                f"this model enables {self._n_cov} covariate(s) (use_return_covariate / trend_halflife) "
                "so fit REQUIRES per-trade covariate rows -- pass covariates=[[...], ...], one row per "
                "observed duration in order [held_return?, trend?]. Covariate coefficients are fitted, "
                "never frozen at 0 (SIGNAL §C.3(c)).")
        if len(covariates) != n_trades:
            raise ValueError("covariates must have one row per observed reversal duration")
        for row in covariates:
            if len(row) != self._n_cov:
                raise ValueError(
                    f"each covariate row must have exactly {self._n_cov} value(s) in the enabled "
                    "order [held_return?, trend?]")
        return covariates

    def hazard_at(self, cursor: ScanCursor, held_days: int, held_return: float) -> float:
        """One-step-ahead reversal hazard ``h(u | x_{t-1})`` at the current bar (``F_{t-1}``-measurable).

        The predictive hazard the exit rule thresholds: duration ``u = held_days`` (elapsed session
        days), covariates built causally from ``cursor.past()`` + ``held_return``. NEVER a full-sample
        fitted survival curve (that conditions on the whole path = smoothed = lookahead). The causal
        touch routes through the ``require_causal`` guard (``cursor.read(cursor.t - 1)``) so a leaky
        ``>= t`` read RAISES ``PurityError`` (the pinning hook). Available for a diagnostic monotonic-
        hold plot; the DECISION uses it via ``exit_now``.
        """
        if self._fit is None:
            raise RuntimeError("hazard_at before fit -- parameters not frozen yet")
        if held_days < 0:
            raise ValueError("held_days (elapsed holding time u) must be non-negative")
        # Causal touch through the guard: a DAILY decision executing on cursor.t may read strictly-past
        # bars only. We read the last available past bar (t-1); a leaky >= t read RAISES PurityError,
        # which is exactly what the no-lookahead pinning test asserts on any leaky covariate wiring.
        if cursor.t > 0:
            cursor.read(cursor.t - 1)
        xs = self._covariates(cursor, held_return)
        return hazard_prob(self._fit.coef, held_days, xs)

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        """Causal dynamic hold: EXIT (``True``) once the one-step-ahead reversal hazard exceeds ``h*``.

        HOLD (``False``) while ``h(u | x_{t-1}) <= h_star``; CLOSE the still-alive winner the instant
        the reversal hazard crosses the train-frozen threshold. Reads ONLY ``held_days`` (the duration
        ``u``), ``held_return``, and ``cursor.past()`` -- all ``F_{t-1}``-measurable; never bar ``t`` or
        later (the causal touch routes through ``cursor.read``; a pinning test FAILS any leak). The -12%
        ``STOP_FLOOR`` is untouched: this returns ``True`` only to bank a winner early, never to loosen
        the loss cap (SIGNAL §C.3).
        """
        if self._fit is None:
            raise RuntimeError("exit_now before fit -- parameters not frozen yet")
        return self.hazard_at(cursor, held_days, held_return) > self._fit.h_star

    # -- the same-bar-lookahead pinning HOOK (asserted by the pinning test) ------------------------
    def _probe_same_bar_read(self, cursor: ScanCursor) -> float:
        """PINNING HOOK: attempt to read bar ``cursor.t`` (the SAME-BAR leak) through the guard.

        Exists ONLY so the no-lookahead pinning test can assert that a covariate/decision wiring which
        read the SAME bar the decision is *for* RAISES ``PurityError`` under the DAILY cadence (SIGNAL
        §C.1 / SYSTEM §C.2). The live ``hazard_at`` / ``exit_now`` path NEVER calls this -- it reads
        only ``cursor.past()`` and touches ``cursor.read(cursor.t - 1)``; this probe merely pins that a
        same-bar read would break, exactly the mutation the pinning test forbids.
        """
        require_causal(cursor.t, cursor.t, cursor.cadence)  # DAILY: read_index == t -> leak -> RAISES
        return cursor.series[cursor.t]  # unreachable under DAILY (guard raised above)

    # -- introspection (frozen params; a test asserts the stop floor is never referenced/loosened) --
    @property
    def coef(self) -> tuple[float, ...]:
        """The frozen fitted coefficients ``(β0, β_dur, β...)`` (raises before fit)."""
        if self._fit is None:
            raise RuntimeError("coef before fit -- parameters not frozen yet")
        return self._fit.coef

    @property
    def h_star(self) -> float:
        """The train-frozen exit-hazard threshold ``h*`` (raises before fit)."""
        if self._fit is None:
            raise RuntimeError("h_star before fit -- parameters not frozen yet")
        return self._fit.h_star

    @property
    def stop_floor(self) -> float:
        """The RETAINED -12% loss cap this exit model may never loosen (== base.STOP_FLOOR)."""
        return STOP_FLOOR
