"""SYSTEM -- validation-as-code: split/embargo, paired trade-level bootstrap CIs, multiplicity gates.

The judge layer (docs/specs/SYSTEM.md): it does NOT generate signals (SIGNAL), pool instruments
(PORTFOLIO), size trades (LEVERAGE), or model fees (COST). It sits on top and decides whether a
pre-registered variant -- expressed as its trade-return stream -- has an edge that survives cost, the
held-out split, its own fat right tail, and the multiple-testing tax. Every function is PURE: it
returns a typed result, never prints (the printed three-row report already ships in
``run/backtest.py`` and is frozen -- extend, never mutate; SYSTEM.md §2).

Three library calls, the smallest correct subset of convexity's validation machinery for a
per-trade, single-family POC (SYSTEM.md §3):

  * ``split(trades, days, boundary, embargo_days)`` -- generalizes the frozen single-date ``SPLIT``
    (``run/backtest.py``) into a reusable held-out partition by trade ``exit_date`` and adds the one
    piece the inline split lacks: an **embargo** that keeps train/test information-disjoint. Rolling
    / anchored walk-forward and CPCV are DEFERRED (SYSTEM.md §6.1) -- the 3yr sample is too thin.
  * ``bootstrap_ci(returns, stat, ...)`` -- the reusable trade-level bootstrap CI, with a paired mode
    that resamples the shared trade INDEX once and applies it to both legs (the futures-vs-options
    ``P(option Sharpe > future)`` study, docs/EDGE.md). Trades are non-overlapping and independent by
    construction, so whole-trade resampling is correct; a stationary/circular-block bootstrap is
    DEFERRED (SYSTEM.md §6.4) -- the single-book stream needs none, so it is not built.
  * ``deflated_sharpe(returns, n_trials_eff)`` -- the Bailey-Lopez de Prado (2014) PSR / DSR / MinTRL
    multiplicity gate, with the ``N == 1`` base case (``SR*_0 := 0`` so ``DSR == PSR(0)`` -- the
    shipped default) and the degeneracy guards (``var_sr >= 0``; ``V <= 0``; ``SR_pp <= SR*_0``). PBO
    / CSCV is DEFERRED to a next step (SYSTEM.md §6.2) -- built only when a variant grid large enough
    for CSCV combinations is scored.

THE annualization-inflation invariant (SYSTEM.md §4, the load-bearing sign/scale rule):
``orb.metrics(...)["sharpe"]`` is an ANNUALIZED display figure and MUST NEVER be fed into any
DSR / PSR / bootstrap-of-Sharpe / MinTRL computation. Every statistic here consumes the
**per-observation** Sharpe ``SR_pp = mean/std`` (``ddof=0``) together with ``n_obs = len(returns)``
and lets ``n_obs`` -- never a pre-multiplied sqrt-factor -- carry the sample-size weight. The helper
``per_obs_sharpe`` is the ONE place that ratio is defined; ``recover_per_obs_sharpe`` inverts the
display figure when only it is in hand (same ``ddof``, or the recovery is itself biased).

``Var_SR`` (the dispersion of the N trials' Sharpe estimates under the null, SYSTEM.md §3.3) is what
scales ``SR*_0``; it is the per-OBSERVATION Sharpe sampling variance, NOT a unit-Sharpe dispersion.
When a caller does not supply it, ``deflated_sharpe`` estimates it from the track itself with the
Lo (2002) / BLdP asymptotic form ``Var_SR ~= (1 + SR_pp^2 / 2) / (n_obs - 1)`` (typically O(1e-3),
so ``SR*_0`` stays a small fraction of 1 as SYSTEM.md §3.3 requires) -- a hard-coded ``1.0`` would be
the dispersion of a UNIT-Sharpe distribution and would inflate ``SR*_0`` by an order of magnitude.

Pure numpy + stdlib ``math`` -- scipy is NOT a dependency, so the normal CDF ``Phi`` and its inverse
``Phi^-1`` are implemented here (``_norm_cdf`` via ``math.erf``; ``_norm_ppf`` via the
Acklam/Beasley-Springer rational approximation). Moneyness, not greeks (CLAUDE.md). Imports only
``intraday.core`` shared contracts (``PERIODS_PER_YEAR``, ``DDOF``, ``TradeRecord``); no sibling
Phase-B pillar. Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import numpy as np

from ._core import DDOF, PERIODS_PER_YEAR, TradeRecord

__all__ = [
    "PERIODS_PER_YEAR",
    "DDOF",
    "EULER_MASCHERONI",
    "Z_GATE",
    "DSR_PASS",
    "N_OBS_FLOOR",
    "SplitResult",
    "BootstrapCI",
    "DeflatedSharpeResult",
    "per_obs_sharpe",
    "recover_per_obs_sharpe",
    "split",
    "bootstrap_ci",
    "deflated_sharpe",
]

# --- Pre-registered gate constants (SYSTEM.md §8 -- causal, NOT grid-searched) --------------------
EULER_MASCHERONI = 0.5772156649015329  # g in the BLdP expected-max-Sharpe closed form (§3.3)
Z_GATE = 1.6448536269514722  # Phi^-1(0.95): the MinTRL confidence z (SYSTEM.md §8)
DSR_PASS = 0.95  # DSR pass threshold (SYSTEM.md §8)
N_OBS_FLOOR = 24  # sanity minimum below which no track is scored (SYSTEM.md §8)

# Below this the return dispersion is numerically indistinguishable from a constant track (float
# round-off leaves ~1e-16 rather than an exact 0 std); such a track has no defined Sharpe.
_STD_EPS = 1e-12


# ==================================================================================================
# Normal CDF / inverse-CDF (scipy-free: Phi via erf, Phi^-1 via a rational approximation)
# ==================================================================================================
def _norm_cdf(x: float) -> float:
    """Phi(x): standard-normal CDF via the stdlib error function (exact to double precision)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF (|abs err| < 1.15e-9), scipy-free.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_PLOW = 0.02425


def _norm_ppf(p: float) -> float:
    """Phi^-1(p): standard-normal quantile for 0 < p < 1 (Acklam). +-inf at the p in {0,1} edges.

    Phi^-1(0) = -inf and Phi^-1(1) = +inf are the true limits; returning them (rather than raising)
    lets the BLdP closed form's N==1 degeneracy (Phi^-1(1 - 1/1) = Phi^-1(0) = -inf) be DETECTED and
    special-cased in `deflated_sharpe`, exactly as SYSTEM.md §3.3 requires.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must lie in [0, 1] for the normal quantile")
    if p < _PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > 1.0 - _PLOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


# ==================================================================================================
# Per-observation Sharpe (SYSTEM.md §4 -- the un-annualized ratio EVERY gate below consumes)
# ==================================================================================================
def _as_returns(returns: Sequence[float]) -> np.ndarray:
    """Coerce a trade stream to a 1-D float return array.

    Accepts a raw float sequence OR an iterable of ``(exit_date, net_return)`` tuples / ``TradeRecord``
    (the SIGNAL unit) -- in the latter case the second field is taken, so a caller can hand SYSTEM the
    stream unchanged. Empty is allowed (callers guard on ``len``).
    """
    a = list(returns)
    if a and isinstance(a[0], (tuple, list)):
        a = [t[1] for t in a]
    return np.asarray(a, dtype="float64")


def per_obs_sharpe(returns: Sequence[float]) -> float:
    """PER-OBSERVATION Sharpe ``mean / std`` (``ddof=0``) -- the un-annualized ratio (SYSTEM.md §4).

    This is the ratio DSR / PSR / MinTRL / a bootstrap-of-Sharpe consume; it is NOT annualized. It
    lets ``n_obs`` (never a pre-multiplied sqrt-factor) carry the sample-size weight, so a thin sample
    cannot be inflated past a gate (THE annualization-inflation lesson, SYSTEM.md §4). ``std`` uses
    ``ddof = DDOF = 0`` to match ``orb.metrics`` exactly. Returns 0.0 for a zero-variance array (a
    degenerate track has no Sharpe), matching ``orb.metrics``' ``a.std() > 0`` guard.
    """
    a = _as_returns(returns)
    if len(a) == 0:
        return 0.0
    sd = a.std(ddof=DDOF)
    # A constant array has std that is exactly-or-numerically zero (float round-off can leave ~1e-18,
    # not a true 0); treat any std at that scale as no dispersion so a degenerate track has no Sharpe.
    return float(a.mean() / sd) if sd > _STD_EPS else 0.0


def recover_per_obs_sharpe(annualized_sharpe: float, n_obs: int, n_days: float) -> float:
    """Invert ``orb.metrics``' display Sharpe back to the per-observation ratio (SYSTEM.md §4).

    ``orb.metrics`` computes ``sharpe = SR_pp * sqrt(n_obs / (n_days / PERIODS_PER_YEAR))``; this
    divides that sqrt(trades_per_year) factor out. Use ONLY when the raw returns are unavailable --
    computing ``per_obs_sharpe`` from the returns directly is the correct path. ``ddof`` must match
    (both ``ddof=0``) or the recovery is itself biased.
    """
    if n_obs <= 0 or n_days <= 0:
        raise ValueError("n_obs and n_days must be positive to recover the per-observation Sharpe")
    ann_factor = math.sqrt(n_obs / (n_days / PERIODS_PER_YEAR))
    return annualized_sharpe / ann_factor


# ==================================================================================================
# 3.1  Splitter + embargo
# ==================================================================================================
class SplitResult(NamedTuple):
    """A held-out partition of a trade stream by ``exit_date`` (SYSTEM.md §3.1)."""

    train: list[TradeRecord]  # exit_date < boundary AND >= embargo_days session-days before it
    test: list[TradeRecord]   # exit_date >= boundary (out-of-sample, untouched by tuning)


def _session_days_before(boundary: dt.date, exit_date: dt.date, days: Sequence[dt.date]) -> int:
    """Count of LAKE SESSION days in ``[exit_date, boundary)`` -- the embargo width unit.

    Measured in lake session days (``B["days"]``), NEVER calendar days: weekends/holidays are not
    tradable and must not count toward the embargo (SYSTEM.md §3.1). Counts session days ``d`` with
    ``exit_date <= d < boundary`` -- so a trade exiting exactly ``embargo_days`` session days before
    the boundary is kept (the ``>= embargo_days`` boundary in `split`).
    """
    return sum(1 for d in days if exit_date <= d < boundary)


def split(
    trades: Sequence[TradeRecord],
    days: Sequence[dt.date],
    boundary: dt.date,
    embargo_days: int = 0,
) -> SplitResult:
    """Held-out split by trade ``exit_date`` with a session-day embargo (SYSTEM.md §3.1, §5.1-2).

    Generalizes the frozen single-date ``SPLIT`` (``run/backtest.py``): ``test`` is every trade with
    ``exit_date >= boundary`` (out-of-sample, never used to tune); ``train`` is every trade with
    ``exit_date < boundary`` that ALSO clears the embargo -- i.e. sits at least ``embargo_days`` lake
    session days before the boundary. The embargo keeps train/test information-disjoint: a trade
    entered just before ``boundary`` can still be open across it (SIGNAL trades hold <= ``cfg.hold``
    trading days), so any train trade whose scheduled ``exit_date`` lands within ``embargo_days``
    session days *before* the boundary is dropped from train (NEVER moved to test -- test is
    unchanged by the embargo, §5.2). Default ``embargo_days = cfg.hold`` (one full hold horizon) is
    the caller's to pass; 0 reproduces the plain inline split exactly.

    Invariants (SYSTEM.md §5.1-2): with ``embargo_days == 0`` the partition is disjoint AND exhaustive
    (``train ∪ test == trades``); with ``embargo_days > 0`` train only SHRINKS
    (``len(train) <= len(train_base)``) and every dropped trade has
    ``0 < (session-days before boundary) < embargo_days``; test is identical either way. Buckets by
    SIGNAL's scheduled-horizon ``exit_date`` (SIGNAL.md §5, inherited verbatim).
    """
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative (lake session days)")
    train: list[TradeRecord] = []
    test: list[TradeRecord] = []
    for t in trades:
        rec = t if isinstance(t, TradeRecord) else TradeRecord(t[0], t[1])
        exit_date = rec[0]
        if exit_date >= boundary:
            test.append(rec)
        elif embargo_days == 0 or _session_days_before(boundary, exit_date, days) >= embargo_days:
            train.append(rec)
        # else: within the embargo window -> dropped from train, NOT added to test.
    return SplitResult(train=train, test=test)


# ==================================================================================================
# 3.2  Paired trade-level bootstrap CI
# ==================================================================================================
class BootstrapCI(NamedTuple):
    """A percentile bootstrap confidence interval and its point estimate (SYSTEM.md §3.2)."""

    lo: float     # lower percentile bound
    hi: float     # upper percentile bound
    point: float  # the statistic on the ORIGINAL (un-resampled) sample


def _resample_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one ordinary trade-level bootstrap resample of length ``n``.

    Draws ``n`` whole trades with replacement -- the single-book case, since SIGNAL trades are
    non-overlapping and independent by construction (SYSTEM.md §3.2). A stationary/circular-block
    bootstrap is DEFERRED (SYSTEM.md §6.4) and deliberately NOT built here: the non-overlapping
    single-book stream needs no time-block, so shipping one would be untested dead code for a feature
    the spec says not to build. Add one against this same call if PORTFOLIO overlap or a REGIME daily
    series ever needs it.
    """
    return rng.integers(0, n, size=n)


def bootstrap_ci(
    returns: Sequence[float],
    stat: Callable[[np.ndarray], float] = per_obs_sharpe,
    *,
    B: int = 20000,
    alpha: float = 0.10,
    seed: int = 0,
    paired_with: Sequence[float] | None = None,
) -> BootstrapCI:
    """Percentile bootstrap CI for ``stat`` on a trade-return stream (SYSTEM.md §3.2, §5.8-9).

    Resamples WHOLE trades with replacement (trades are non-overlapping, so no time-block is needed
    for the single-book case), recomputes ``stat`` on each of ``B`` resamples, and returns the
    empirical ``(lo, hi)`` percentile interval at ``alpha`` (default 90% CI, ``alpha=0.10``) plus the
    point estimate on the ORIGINAL sample. The default ``B=20000`` matches the docs/EDGE.md
    futures-vs-options study.

    **Sign/scale contract (SYSTEM.md §3.2/§4):** ``stat`` operates on the PER-OBSERVATION return array;
    when it is a Sharpe it MUST be ``per_obs_sharpe`` (the un-annualized ``mean/std``, ddof=0), never
    the annualized display Sharpe -- the default ``stat`` is exactly ``per_obs_sharpe`` so the correct
    thing happens without a caller decision.

    **Paired mode (SYSTEM.md §3.2):** pass ``paired_with`` (a second stream on the SAME ORB trade
    dates, e.g. the futures leg) to bootstrap the DIFFERENCE ``stat(leg_a) - stat(leg_b)``. The shared
    trade INDEX is resampled ONCE per iteration and applied to both legs, so the difference is
    evaluated on matched pairs -- what makes ``P(option Sharpe > future)`` meaningful. The paired CI is
    sign-consistent with the point difference (§5.8).

    Deterministic given ``seed`` (§5.9): same seed => identical CI. Empty input raises (a CI is
    undefined with no trades).
    """
    a = _as_returns(returns)
    if len(a) == 0:
        raise ValueError("bootstrap_ci needs a non-empty return stream")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    rng = np.random.default_rng(seed)
    n = len(a)

    if paired_with is not None:
        b = _as_returns(paired_with)
        if len(b) != n:
            raise ValueError("paired_with must be the same length as returns (matched trade pairs)")
        point = float(stat(a) - stat(b))
        dist = np.empty(B, dtype="float64")
        for i in range(B):
            idx = _resample_indices(n, rng)
            dist[i] = stat(a[idx]) - stat(b[idx])
    else:
        point = float(stat(a))
        dist = np.empty(B, dtype="float64")
        for i in range(B):
            idx = _resample_indices(n, rng)
            dist[i] = stat(a[idx])

    lo = float(np.quantile(dist, alpha / 2.0))
    hi = float(np.quantile(dist, 1.0 - alpha / 2.0))
    return BootstrapCI(lo=lo, hi=hi, point=point)


# ==================================================================================================
# 3.3  Multiplicity gate -- DSR / PSR / MinTRL (Bailey-Lopez de Prado 2014)
# ==================================================================================================
@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Typed result of the DSR / PSR / MinTRL multiplicity gate (SYSTEM.md §3.3).

    ``psr``     : PSR against a zero benchmark, ``PSR(0)`` -- probability the true SR_pp exceeds 0.
    ``dsr``     : PSR deflated by the luck benchmark ``SR*_0``; at ``n_trials_eff == 1`` (``SR*_0=0``)
                  it collapses to ``PSR(0) == psr`` (the shipped default, no deflation).
    ``sr_pp``   : the PER-OBSERVATION Sharpe (``mean/std``, ddof=0) the gate ran on (NEVER annualized).
    ``sr_star`` : the BLdP expected-max Sharpe under the null over ``n_trials_eff`` trials (0 at N==1).
    ``n_obs``   : number of NON-OVERLAPPING trade returns = ``len(returns)`` (never a bar/day count).
    ``n_trials_eff`` : the count of GENUINELY DISTINCT configs scored (1 for the pre-registered family).
    ``min_trl`` : Minimum Track Record Length; ``inf`` when ``sr_pp <= sr_star`` (unbounded -- no length
                  clears the luck bar) or ``V <= 0`` (unscorable).
    ``passed``  : True iff scorable, ``n_obs >= N_OBS_FLOOR``, and ``dsr >= DSR_PASS``.
    ``reason``  : human-readable status -- the failing guard (V<=0, SR<=SR*, thin sample) or "ok".
    """

    psr: float
    dsr: float
    sr_pp: float
    sr_star: float
    n_obs: int
    n_trials_eff: int
    min_trl: float
    passed: bool
    reason: str


def _expected_max_sharpe(var_sr: float, n_trials_eff: int) -> float:
    """BLdP (2014) expected-max Sharpe under the null over N independent trials (SYSTEM.md §3.3).

    ``SR*_0 = sqrt(Var_SR) * [ (1-g)*Phi^-1(1 - 1/N) + g*Phi^-1(1 - 1/(N e)) ]`` for ``N >= 2``;
    ``:= 0`` for ``N == 1`` (the closed form is DEGENERATE there: ``Phi^-1(1 - 1/1) = Phi^-1(0) =
    -inf``, so the correct limit -- the single trial's own Sharpe contributes NO deflation -- is
    ``SR*_0 = 0`` and DSR collapses to ``PSR(0)``). ``g`` = Euler-Mascheroni. ``var_sr`` is the
    per-observation Sharpe sampling variance (SYSTEM.md §3.3, O(1e-3) in practice), NOT a unit-Sharpe
    dispersion -- ``deflated_sharpe`` supplies it (validated ``>= 0`` there); this helper only
    consumes it, and the ``N == 1`` base case never touches it.
    """
    if n_trials_eff == 1:
        return 0.0  # N==1 base case: no deflation (SYSTEM.md §3.3), var_sr unused.
    g = EULER_MASCHERONI
    term = (1.0 - g) * _norm_ppf(1.0 - 1.0 / n_trials_eff) + g * _norm_ppf(
        1.0 - 1.0 / (n_trials_eff * math.e))
    return math.sqrt(var_sr) * term


def deflated_sharpe(
    returns: Sequence[float],
    n_trials_eff: int = 1,
    *,
    var_sr: float | None = None,
    sr_benchmark: float = 0.0,
    z_gate: float = Z_GATE,
    dsr_pass: float = DSR_PASS,
    n_obs_floor: int = N_OBS_FLOOR,
) -> DeflatedSharpeResult:
    """The DSR / PSR / MinTRL multiplicity gate on a trade-return stream (SYSTEM.md §3.3, §5.5-6).

    Runs on the PER-OBSERVATION Sharpe ``SR_pp = mean/std`` (ddof=0) and ``n_obs = len(returns)`` --
    NEVER the annualized display Sharpe (THE annualization-inflation lesson, §4). Formulas
    (Bailey-Lopez de Prado 2014)::

        gamma3 = skew(returns);  gamma4 = raw (non-excess) kurtosis(returns)
        V      = 1 - gamma3*SR_pp + ((gamma4 - 1)/4)*SR_pp^2
        PSR(b) = Phi( (SR_pp - b) * sqrt(n_obs - 1) / sqrt(V) )
        SR*_0  = expected-max Sharpe over n_trials_eff trials (0 at N==1 -- the base case)
        DSR    = PSR(SR*_0)
        MinTRL = 1 + V * (z_gate / (SR_pp - SR*_0))^2         [ for SR_pp > SR*_0 ]

    ``n_trials_eff`` is the count of GENUINELY DISTINCT configs scored; it MUST be ``>= 1`` (a count of
    configs -- 0 or negative is a caller bug and is REJECTED). It is ``1`` for the shipped
    pre-registered family, which sets ``SR*_0 = 0`` so ``DSR == PSR(0)`` (the non-degenerate default).
    DSR is monotone NON-INCREASING in ``n_trials_eff`` (more trials => higher luck bar => lower DSR),
    and ``DSR(k) <= DSR(1) = PSR(0)`` for all ``k >= 1`` (§5.5).

    ``var_sr`` is ``Var_SR`` -- the cross-trial dispersion of the per-OBSERVATION Sharpe estimates
    under the null (SYSTEM.md §3.3), which scales ``SR*_0``. It is used ONLY for ``n_trials_eff >= 2``.
    When omitted (``None``) it is estimated from the track itself with the Lo (2002) / BLdP asymptotic
    per-observation Sharpe sampling variance ``(1 + SR_pp^2 / 2) / (n_obs - 1)`` -- typically O(1e-3),
    so ``SR*_0`` stays a small fraction of 1. A hard-coded ``1.0`` would be the dispersion of a
    UNIT-Sharpe distribution and would over-deflate ``SR*_0`` by an order of magnitude, collapsing DSR
    on a genuinely fine track the moment ``n_trials_eff >= 2``. A supplied ``var_sr`` must be ``>= 0``
    (a variance) or it is REJECTED (a negative would make ``sqrt`` a bare math-domain error).

    **Degeneracy guards (all real NaN/inf hazards -- handled explicitly, never a NaN emitted):**
      * ``n_trials_eff >= 1`` required (a count of configs) -- raises ``ValueError`` otherwise.
      * ``var_sr >= 0`` required when supplied (a variance) -- raises ``ValueError`` otherwise.
      * ``V > 0`` required: extreme skew/kurtosis can drive ``V`` non-positive; then ``sqrt(V)`` is
        undefined and the gate is UNSCORABLE -- returns ``passed=False`` with a "variance term
        non-positive" reason, ``psr=dsr=nan``, ``min_trl=inf`` (§3.3 guard 1).
      * ``SR_pp > SR*_0`` required for MinTRL: MinTRL diverges as ``SR_pp -> SR*_0`` and is meaningless
        for ``SR_pp <= SR*_0`` (no length clears the luck bar) -- reported as ``min_trl=inf`` with a
        reason, never divided (§3.3 guard 2). With ``SR*_0=0`` at N==1 this reduces to: a losing /
        zero-Sharpe track has unbounded MinTRL, correctly.

    A thin sample (``n_obs < n_obs_floor``, default 24) never PASSES (SYSTEM.md §8), though its PSR/DSR
    are still reported for inspection. Empty / single-observation input is unscorable (``n_obs - 1``
    and the std vanish).
    """
    if n_trials_eff < 1:
        raise ValueError("n_trials_eff must be >= 1 (a count of genuinely distinct configs scored)")
    if var_sr is not None and var_sr < 0:
        raise ValueError("var_sr must be >= 0 (it is the cross-trial Sharpe variance, Var_SR)")
    a = _as_returns(returns)
    n_obs = int(len(a))

    if n_obs < 2:
        return DeflatedSharpeResult(
            psr=math.nan, dsr=math.nan, sr_pp=0.0, sr_star=0.0, n_obs=n_obs,
            n_trials_eff=n_trials_eff, min_trl=math.inf, passed=False,
            reason="n_obs < 2 -- unscorable (need >= 2 non-overlapping trades)")

    sd = a.std(ddof=DDOF)
    if sd <= _STD_EPS:  # constant / numerically-flat track: Sharpe undefined
        return DeflatedSharpeResult(
            psr=math.nan, dsr=math.nan, sr_pp=0.0, sr_star=0.0, n_obs=n_obs,
            n_trials_eff=n_trials_eff, min_trl=math.inf, passed=False,
            reason="zero-variance track -- Sharpe undefined")

    sr_pp = float(a.mean() / sd)
    # Var_SR: the per-observation Sharpe sampling variance (SYSTEM.md §3.3). Supplied by the caller for
    # a real grid, else estimated from the track itself (Lo 2002 / BLdP asymptotic form, O(1e-3)) --
    # NEVER a hard 1.0 (that is a unit-Sharpe dispersion and over-deflates SR*_0 by ~an order of mag).
    v_sr = (1.0 + sr_pp ** 2 / 2.0) / (n_obs - 1) if var_sr is None else var_sr
    sr_star = _expected_max_sharpe(v_sr, n_trials_eff)

    mu = a.mean()
    m2 = (((a - mu) ** 2).mean())
    gamma3 = float(((a - mu) ** 3).mean() / m2 ** 1.5)  # skew
    gamma4 = float(((a - mu) ** 4).mean() / m2 ** 2)    # RAW (non-excess) kurtosis
    v = 1.0 - gamma3 * sr_pp + ((gamma4 - 1.0) / 4.0) * sr_pp ** 2

    # Guard 1: V must be positive or PSR/DSR/MinTRL are unscorable (SYSTEM.md §3.3).
    if v <= 0:
        return DeflatedSharpeResult(
            psr=math.nan, dsr=math.nan, sr_pp=sr_pp, sr_star=sr_star, n_obs=n_obs,
            n_trials_eff=n_trials_eff, min_trl=math.inf, passed=False,
            reason="psr variance term non-positive (extreme skew/kurtosis) -- track unscorable")

    root = math.sqrt(n_obs - 1) / math.sqrt(v)
    psr = _norm_cdf((sr_pp - sr_benchmark) * root)
    dsr = _norm_cdf((sr_pp - sr_star) * root)

    # Guard 2: MinTRL defined only for SR_pp > SR*_0 (else unbounded -- no length clears the luck bar).
    if sr_pp <= sr_star:
        return DeflatedSharpeResult(
            psr=psr, dsr=dsr, sr_pp=sr_pp, sr_star=sr_star, n_obs=n_obs,
            n_trials_eff=n_trials_eff, min_trl=math.inf, passed=False,
            reason="SR_pp <= SR*_0 -- required track length unbounded")

    min_trl = 1.0 + v * (z_gate / (sr_pp - sr_star)) ** 2

    if n_obs < n_obs_floor:
        return DeflatedSharpeResult(
            psr=psr, dsr=dsr, sr_pp=sr_pp, sr_star=sr_star, n_obs=n_obs,
            n_trials_eff=n_trials_eff, min_trl=min_trl, passed=False,
            reason=f"n_obs {n_obs} < floor {n_obs_floor} -- too thin to score")

    passed = dsr >= dsr_pass
    reason = "ok" if passed else f"dsr {dsr:.4f} < pass threshold {dsr_pass}"
    return DeflatedSharpeResult(
        psr=psr, dsr=dsr, sr_pp=sr_pp, sr_star=sr_star, n_obs=n_obs,
        n_trials_eff=n_trials_eff, min_trl=min_trl, passed=passed, reason=reason)
