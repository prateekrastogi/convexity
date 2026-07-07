"""
Axis A -- SIGNAL EXTRACTION: causal-filtering treatments of trend estimation for the Stage-A
CTA sweep (equity-cta / research/cta program).

Replaces the crude fixed-SMA-crossover / single-timeframe-argmax construction (shown fragile by
Established Fact (5): MA(50,200) wins 3 of 4 scheme/direction cuts, MA(10,50) wins the 4th -- an
honest inconsistency across sub-samples, not something to explain away by picking one "best"
window) with:

  1. an EWMA-crossover continuum, presented via the EXACT algebraic identity connecting a
     dual-EWMA-of-PRICE crossover to a weighted difference of two single-speed EWMA-of-RETURN
     momentum signals (the rigorous version of the Levine & Pedersen (2016), "Which Trend Is
     Your Friend?", intuition -- see `ewma_return_momentum`, `ewma_price_crossover`,
     `crossover_as_return_ewma_difference`);
  2. a local-linear-trend (level+slope) Kalman filter, with its steady-state gain derived from
     the discrete algebraic Riccati recursion and shown algebraically equivalent to Holt's linear
     (double) exponential smoothing -- see `kalman_local_linear_trend`, `steady_state_kalman_gain`,
     and `holt_parameters_from_kalman_gain`;
  3. a causal ROLLING adaptation of Kim-Koh-Boyd (2009) l1 trend filtering, solved per window via
     ADMM (ADMM form and banded solve after Ramdas & Tibshirani (2016), "Fast and Flexible ADMM
     Algorithms for Trend Filtering") -- see `l1_trend_filter_window`, `l1_trend_filter_rolling`;
  4. a multi-horizon COMBINATION of the above, weighted by a Grinold-Kahn Fundamental-Law-of-
     Active-Management-motivated, IC-based, shrinkage-regularized rule -- see
     `rolling_information_coefficient`, `combination_weights_from_ic`, `multi_horizon_trend_signal`.

Causal wavelet multi-resolution decomposition was investigated and explicitly REJECTED as a
comparison method -- see the module docstring section "WAVELETS: WHY NOT" below -- and is not
implemented.

--------------------------------------------------------------------------------------------------
CAUSALITY CONVENTION (read before calling anything below)
--------------------------------------------------------------------------------------------------
Every "raw filter" function (`ewma_level`, `ewma_return_momentum`, `ewma_price_crossover`,
`kalman_local_linear_trend`, `l1_trend_filter_rolling`) is a REAL-TIME filter: its value at index
t is computed using price/return data through and including t ONLY -- no future information ever
enters the estimate at t. This is the standard definition of a causal filter (what a live system
observing the market tick-by-tick could compute at the close of day t) and is what makes these
functions legitimate drop-in replacements for the fixed SMA/EWMA machinery already used elsewhere
in this codebase (e.g. `i1_pipeline.py`'s `sma200.shift(1)` pattern).

Per Established Bug Pattern (c) (decide-at-close(t-1), act-at(t)): a value computed AS OF the
close of day t may only be ACTED ON starting day t+1. Accordingly:
  - Raw / component signal functions return the t-inclusive real-time value and are NOT
    exposure-ready on their own.
  - Every function whose docstring says "EXPOSURE-READY" applies exactly one further `.shift(1)`
    internally before returning, so its output at label t depends only on information available
    through t-1's close and is safe to multiply directly against day-t's return.
  - `multi_horizon_trend_signal`, the top-level deliverable, is EXPOSURE-READY.

This is verified, not just asserted: `synthetic_outlier_injection_test` (bottom of this module)
injects one large fake price move at a single index into a copy of real data and confirms (a) no
component's raw output reacts before the injected index, (b) no EXPOSURE-READY output reacts
before injected_index + 1, and (c) a deliberately-NAIVE, unlagged control version of the same
signal DOES react same-day -- proving the test has real discriminating power.

--------------------------------------------------------------------------------------------------
WAVELETS: WHY NOT
--------------------------------------------------------------------------------------------------
A causal (online) wavelet multi-resolution decomposition was investigated as a comparison method,
per the task's requirement to be honest about practicality rather than forcing it in. Conclusion:
NOT implemented, for two compounding reasons specific to daily-close CTA signals.

First, the standard multi-resolution transforms (DWT via Mallat's pyramid algorithm, or the
redundant a-trous/MODWT used for most "trading" applications because it is shift-invariant) are
defined using a two-sided filter kernel: a coefficient at scale j, position t is a weighted
combination of samples in a NEIGHBORHOOD of t that includes samples strictly after t (e.g. a
Daubechies-4 low-pass kernel has 4 taps that, in the standard centered convention, span
[t-1, t+2]). Making the transform causal requires either (i) a fully one-sided (minimum-phase)
wavelet basis, which discards the near-perfect reconstruction / orthogonality properties that
motivate using wavelets in the first place, or (ii) shifting the kernel to be causal and eating a
DELAY equal to the kernel support at every scale -- and that delay DOUBLES at every dyadic scale
in a standard pyramid (2^j - 1 samples of pure latency at scale j), so a "slow" scale-5 component
(nominally a ~32-day feature) arrives roughly a month stale by construction. This defeats the
entire purpose of a slow trend estimate, which is precisely to describe the CURRENT state at low
latency, not a month-old one.

Second, and more decisively for this specific application: the causal EWMA and l1-trend-filter
machinery already implemented in this module perform the SAME job a wavelet multi-resolution
decomposition would be recruited for here -- extracting several simultaneous, differently-scaled
smooth components of one price series -- with a well-understood, exactly-causal, real-time
recursive (EWMA) or convex-optimization (l1TF) formulation, and with a documented, standard
recipe for combining across scales (Grinold-Kahn / IC-weighted, Section 4 below). A wavelet
decomposition would not reveal a materially different multi-horizon structure on daily-close data
that these methods do not already reveal (both are, in the frequency domain, banks of
low/band-pass filters at dyadic-ish scales), while adding the causality/latency problem above and
edge-effect distortion at the boundary of every rolling window (wavelet boundary handling --
periodic, symmetric, or zero-padding extension -- is a well-known source of spurious energy at
the most recent, and for trading the MOST IMPORTANT, edge of the series). Net: a real, honestly-
reasoned "not practical here" rather than a forced inclusion. Cited for the interested reader:
Percival & Walden (2000), "Wavelet Methods for Time Series Analysis", Ch. 5 on the MODWT's
boundary and alignment properties, which is precisely the mechanism described above.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import diags as _sparse_diags


# ======================================================================================
# 1. EWMA CONTINUUM: dual-EWMA-of-price crossover <=> weighted difference of two single-speed
#    EWMA-of-return momentum signals (exact algebraic identity, derived in the markdown section;
#    reproduced here as the two building blocks the identity connects).
# ======================================================================================

def halflife_to_decay(halflife: float) -> float:
    """lambda such that lambda**halflife == 0.5, i.e. lambda = 2**(-1/halflife).

    This is pandas' own `.ewm(halflife=...)` convention (`adjust=False` recursion
    E_t = lambda*E_{t-1} + (1-lambda)*x_t), used throughout this module so every function's
    `halflife` argument means the same thing.
    """
    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    return 2.0 ** (-1.0 / halflife)


def center_of_mass(halflife: float) -> float:
    """Mean age (center of mass) of an EWMA kernel with the given halflife: com = lambda/(1-lambda).

    This is exactly pandas' `com` parameter (`alpha = 1/(1+com)`), and it is exactly the
    coefficient `c_i` that appears in `crossover_as_return_ewma_difference` below -- see the
    markdown derivation for why the two are the same object.
    """
    lam = halflife_to_decay(halflife)
    return lam / (1.0 - lam)


def ewma_level(log_price: pd.Series, halflife: float) -> pd.Series:
    """E_lambda[log_price]_t -- causal (real-time) EWMA level. Uses data through t only."""
    return log_price.ewm(halflife=halflife, adjust=False).mean()


def ewma_return_momentum(log_price: pd.Series, halflife: float) -> pd.Series:
    """EWMA_lambda[r]_t := (1-lambda) * sum_j lambda**j * r_{t-j}, r = diff(log_price), seeded at
    r_0 := 0 (the one NaN produced by `.diff()` at the series' first index is filled with 0, i.e.
    "zero return before the first observation") rather than pandas' default `ignore_na` seeding
    (which would instead seed the recursion at the first REAL return, m_1 = r_1).

    This is the single-speed "smoothed trailing return" momentum signal (Moskowitz-Ooi-Pedersen /
    Baltas-Kosowski style time-series momentum), and per `crossover_as_return_ewma_difference` it
    is the elementary building block that a price-EWMA crossover reduces to. Causal: r_t is known
    at close t, so EWMA_lambda[r]_t uses data through t only.

    WHY THE r_0:=0 SEEDING MATTERS (not cosmetic): `ewma_level`'s recursion is seeded
    E_0 = log_price_0 (pandas' `adjust=False` convention for a series with no NaNs: the first
    value is its own unweighted seed). Writing e_t := E_t - log_price_t, that seeding gives
    e_0 = 0. Matching this exactly requires m_0 = 0 too (shown by induction in the module's
    derivation section): with m_0 = 0, e_t = -center_of_mass(halflife) * m_t HOLDS EXACTLY for
    every t, not just asymptotically. Using pandas' default `ignore_na` seeding instead (m_1=r_1)
    introduces a boundary term that decays geometrically at rate `lambda**t` but is NOT exactly
    zero for finite t -- confirmed empirically on real SPY/QQQ data (max discrepancy up to ~0.25
    in early-sample log-price units before this fix). The fillna(0.0) below is what makes
    `crossover_as_return_ewma_difference` equal `ewma_price_crossover` to float precision at every
    t, not merely in the long-history limit.
    """
    r = log_price.diff().fillna(0.0)
    return r.ewm(halflife=halflife, adjust=False).mean()


def ewma_price_crossover(log_price: pd.Series, fast_halflife: float, slow_halflife: float) -> pd.Series:
    """D_t = E_fast[log_price]_t - E_slow[log_price]_t -- the classic dual-EWMA crossover, in its
    ordinary price-level parameterization. Causal. See `crossover_as_return_ewma_difference` for
    the exactly-equivalent return-EWMA representation this module actually builds signals from.
    """
    if fast_halflife >= slow_halflife:
        raise ValueError("fast_halflife must be < slow_halflife for a crossover")
    return ewma_level(log_price, fast_halflife) - ewma_level(log_price, slow_halflife)


def crossover_as_return_ewma_difference(log_price: pd.Series, fast_halflife: float,
                                         slow_halflife: float) -> pd.Series:
    """The SAME quantity as `ewma_price_crossover`, computed via the algebraic identity derived in
    the markdown section:

        E_lambda[p]_t - p_t = -lambda/(1-lambda) * EWMA_lambda[r]_t
        =>  D_t = E_f[p]_t - E_s[p]_t
                = c_s * EWMA_s[r]_t - c_f * EWMA_f[r]_t,     c_i := center_of_mass(halflife_i)

    Returned purely as a numerical identity check (see `test_crossover_identity` in the paired
    test module -- confirms this equals `ewma_price_crossover` to float precision on real data);
    the practical reason to prefer the return-EWMA representation is that its two terms are
    directly interpretable, unit-comparable momentum scores that slot into the same combination
    machinery as every other component signal in Section 4, whereas the raw price-level crossover
    does not (it is priced in raw log-price units and its scale depends on both speeds jointly).
    """
    c_f = center_of_mass(fast_halflife)
    c_s = center_of_mass(slow_halflife)
    m_f = ewma_return_momentum(log_price, fast_halflife)
    m_s = ewma_return_momentum(log_price, slow_halflife)
    return c_s * m_s - c_f * m_f


def normalize_signal(raw: pd.Series, vol_window: int = 20, min_periods: int | None = None,
                      response: str = "tanh_gaussian") -> pd.Series:
    """Causal cross-time normalization of a raw trend/momentum signal to an approximately unit-
    scale score, following the Baltas & Kosowski (2017), "Demystifying Time-Series Momentum
    Strategies", response-function convention: normalize by trailing realized vol of the raw
    signal itself, then squash through a bounded response function so a small number of extreme
    days cannot dominate the combination step.

    `response`:
      - "tanh_gaussian": u = z / sqrt(0.5*(1+z**2)) -- the Baltas-Kosowski / Man AHL construction,
        which is the identity map near 0 and saturates smoothly to +-sqrt(2) in the tails (bounded,
        unlike a raw z-score or a hard tanh, and derived from assuming z ~ N(0,1) under the null of
        no signal, so u has, by construction, the same first two moments as z near the origin).
      - "zscore": no squashing, plain z = raw / trailing_std(raw).
      - "sign": +-1 (used only for diagnostics / the naive control in the injection test).

    Uses `.shift(1)` on the trailing-std denominator so the SCALE estimate at t never uses raw_t
    itself (avoids a subtle look-ahead: raw_t / std(raw_{t-w+1..t}) would let an extreme raw_t
    shrink its own normalizer).
    """
    std = raw.rolling(vol_window, min_periods=min_periods or vol_window).std().shift(1)
    z = raw / std.replace(0.0, np.nan)
    if response == "zscore":
        return z
    if response == "sign":
        return np.sign(z)
    if response == "tanh_gaussian":
        return z / np.sqrt(0.5 * (1.0 + z ** 2))
    raise ValueError(f"unknown response function {response!r}")


# ======================================================================================
# 2. LOCAL-LINEAR-TREND KALMAN FILTER: derivation-backed steady-state gain, connected to
#    Holt's linear (double) exponential smoothing.
# ======================================================================================

def steady_state_kalman_gain(q_level: float, q_slope: float, r: float,
                              max_iter: int = 2000, tol: float = 1e-14) -> dict:
    """Steady-state Kalman gain for the local-linear-trend state-space model:

        level_t = level_{t-1} + slope_{t-1} + w1_t,   w1 ~ N(0, q_level)
        slope_t = slope_{t-1}               + w2_t,   w2 ~ N(0, q_slope)
        y_t     = level_t                   + v_t,    v  ~ N(0, r)

    i.e. F = [[1,1],[0,1]], Q = diag(q_level, q_slope), H = [1, 0].

    The steady-state predicted-error covariance P (2x2, symmetric) is the fixed point of the
    discrete algebraic Riccati recursion
        P <- F (P - P H^T (H P H^T + r)^-1 H P) F^T + Q
    which for this 2-state system reduces to 3 coupled scalar equations in (P11, P12, P22) with
    no simple closed form in the general (q_level, q_slope both > 0) case (it is a genuine
    quartic once fully eliminated) -- so, as is standard practice (this is exactly how e.g.
    statsmodels' UnobservedComponents / classical ETS-Kalman equivalence is established
    numerically; see Harvey (1989) Sec. 3.3, Durbin & Koopman (2012) Sec. 4.3), this function
    finds P by iterating the recursion to numerical convergence from P0 = Q, which is guaranteed
    to converge to the unique stabilizing solution for this (detectable, stabilizable) system.

    Returns a dict with keys: K_level, K_slope (steady-state Kalman gain), P (2x2 steady-state
    predicted covariance), n_iter (iterations used), converged (bool).

    See `holt_parameters_from_kalman_gain` for the closed-form algebra connecting (K_level,
    K_slope) to Holt's linear-trend smoothing parameters (alpha, beta) -- a real, well-known
    result, derived from scratch in the markdown section by directly matching the steady-state
    KF update equations to Holt's recursion term by term.
    """
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_slope]])
    H = np.array([[1.0, 0.0]])
    P = Q.copy()
    converged = False
    for it in range(1, max_iter + 1):
        S = (H @ P @ H.T).item() + r
        K = (P @ H.T) / S                      # 2x1 Kalman gain at this iterate
        P_upd = P - K @ H @ P                  # filtered covariance
        P_next = F @ P_upd @ F.T + Q           # predicted covariance for next step
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            converged = True
            it_final = it
            break
        P = P_next
    else:
        it_final = max_iter
    S = (H @ P @ H.T).item() + r
    K = (P @ H.T) / S
    return {
        "K_level": float(K[0, 0]), "K_slope": float(K[1, 0]),
        "P": P, "n_iter": it_final, "converged": converged,
    }


def holt_parameters_from_kalman_gain(K_level: float, K_slope: float) -> dict:
    """alpha = K_level, beta = K_slope / K_level -- the exact correspondence between the
    steady-state local-linear-trend Kalman filter and Holt's linear exponential smoothing.

    Derivation (matching the KF steady-state update to Holt's recursion term by term):
        KF:   level_t = level_{t-1}+slope_{t-1} + K_level*e_t
              slope_t = slope_{t-1}             + K_slope*e_t         , e_t = y_t-level_{t-1}-slope_{t-1}
        Holt: level_t = level_{t-1}+slope_{t-1} + alpha*e_t
              slope_t = slope_{t-1}             + alpha*beta*e_t
    matching the level equations gives alpha = K_level directly; matching the slope equations
    gives alpha*beta = K_slope, hence beta = K_slope/alpha = K_slope/K_level. See Harvey (1989)
    Sec. 2.5.4 and Hyndman, Koehler, Ord & Snyder (2008), "Forecasting with Exponential
    Smoothing: The State Space Approach", Ch. 2, for the standard statement of this equivalence
    class (ETS(A,A,N) <-> local-linear-trend state space); the term-by-term match above is the
    derivation, not merely a citation.
    """
    if K_level <= 0:
        raise ValueError("K_level must be > 0")
    return {"alpha": K_level, "beta": K_slope / K_level}


def kalman_local_linear_trend(log_price: pd.Series, q_level: float, q_slope: float,
                               r: float) -> pd.DataFrame:
    """Causal (FILTER, not smoother -- the filtered estimate at t uses observations through t
    only, never t+1 or later) local-linear-trend Kalman filter. Runs the Riccati recursion
    forward each step (rather than assuming steady state immediately) so the first ~20-50 points
    show the genuine transient before settling near the steady-state gain from
    `steady_state_kalman_gain`.

    Returns a DataFrame indexed like `log_price` with columns:
      level  -- filtered level estimate x_hat[t|t][0]
      slope  -- filtered slope estimate x_hat[t|t][1]
      K_level, K_slope -- the gain actually used at each t (converges to steady state)
    """
    y = log_price.to_numpy(dtype="float64")
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_slope]])
    H = np.array([[1.0, 0.0]])

    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    k_level_arr = np.full(n, np.nan)
    k_slope_arr = np.full(n, np.nan)

    first = np.argmax(~np.isnan(y)) if np.any(~np.isnan(y)) else n
    if first >= n:
        return pd.DataFrame({"level": level, "slope": slope,
                              "K_level": k_level_arr, "K_slope": k_slope_arr}, index=log_price.index)

    x = np.array([y[first], 0.0])
    P = np.array([[r, 0.0], [0.0, q_slope if q_slope > 0 else 1.0]])
    level[first], slope[first] = x
    k_level_arr[first], k_slope_arr[first] = 0.0, 0.0

    for t in range(first + 1, n):
        if np.isnan(y[t]):
            level[t], slope[t] = level[t - 1], slope[t - 1]
            k_level_arr[t], k_slope_arr[t] = 0.0, 0.0
            continue
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        S = (H @ P_pred @ H.T).item() + r
        K = (P_pred @ H.T) / S
        e = y[t] - (H @ x_pred).item()
        x = x_pred + (K.flatten() * e)
        P = P_pred - K @ H @ P_pred
        level[t], slope[t] = x
        k_level_arr[t], k_slope_arr[t] = K[0, 0], K[1, 0]

    return pd.DataFrame({"level": level, "slope": slope,
                          "K_level": k_level_arr, "K_slope": k_slope_arr}, index=log_price.index)


# ======================================================================================
# 3. CAUSAL ROLLING l1 TREND FILTER (Kim-Koh-Boyd 2009, adapted from a global/offline QP to a
#    strictly-causal rolling-window estimator; ADMM form after Ramdas & Tibshirani (2016)).
# ======================================================================================

def _second_diff_operator(n: int):
    """Sparse (n-2) x n second-order difference operator D: (Dx)_i = x_i - 2 x_{i+1} + x_{i+2}.
    Penalizing ||Dx||_1 (rather than ||Dx||_2^2) is exactly Kim-Koh-Boyd's "l1 trend filter": it
    yields a PIECEWISE-LINEAR fit (the l1 penalty drives most consecutive second differences --
    i.e. changes in slope -- to exactly zero, so the fit is a small number of straight segments
    joined at a sparse set of "kink" points), unlike an l2 (Hodrick-Prescott) penalty which shrinks
    every second difference a little instead of zeroing most of them out.
    """
    return _sparse_diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))


@lru_cache(maxsize=128)
def _l1tf_cholesky_factor(n: int, rho: float):
    """Cholesky factor of (I + rho * D^T D), cached by (window length, rho) only -- this matrix
    depends on neither the data nor the day, so for a fixed (window, rho) pair used across an
    entire rolling scan (potentially thousands of days x dozens of names) the O(n^3) factorization
    cost is paid exactly ONCE, and every ADMM x-update thereafter is a single O(n^2) triangular
    solve via `cho_solve`. This caching is what makes the rolling adaptation computationally
    tractable at all -- a naive re-factorization per window would be the dominant cost.
    """
    D = _second_diff_operator(n)
    DtD = (D.T @ D).toarray()
    M = np.eye(n) + rho * DtD
    c, low = cho_factor(M, lower=True)
    return c, low, D


def l1_trend_filter_window(y: np.ndarray, lam: float, rho: float = 1.0, n_iter: int = 150,
                            tol: float = 1e-6, z_init: np.ndarray | None = None,
                            u_init: np.ndarray | None = None,
                            return_dual: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ADMM solve of Kim-Koh-Boyd l1 trend filtering on a single (short) window `y`:

        minimize_x  (1/2)||y-x||_2^2 + lam * ||Dx||_1

    via the standard scaled-ADMM splitting (Boyd et al. (2011), "Distributed Optimization and
    Statistical Learning via ADMM", Sec 3.1; Ramdas & Tibshirani (2016) apply exactly this
    splitting to trend filtering specifically, which is what is implemented here):
        x-update: x <- (I + rho D^T D)^-1 (y + rho D^T(z-u))     [banded/Cholesky solve]
        z-update: z <- soft_threshold(Dx+u, lam/rho)
        u-update: u <- u + Dx - z

    This single-window solve is itself the OFFLINE (non-causal within the window) Kim-Koh-Boyd
    estimator -- see `l1_trend_filter_rolling` for how it is adapted into a causal rolling
    estimator. Returns the fitted x (same length as y). Windows shorter than 5 points are returned
    unfiltered (too short for a second-difference penalty to mean anything).

    `z_init`/`u_init` (optional, length n-2): WARM-START the ADMM dual/auxiliary variables from a
    caller-supplied starting point instead of the zero default. This objective is STRICTLY CONVEX
    in x (the (1/2)||y-x||^2 term is positive-definite everywhere), so ADMM converges to the SAME
    unique global optimum regardless of (z,u) initialization -- warm-starting changes only how
    fast the iteration reaches the `tol` stopping criterion, never the converged answer itself
    (see `research/cta/CTA_MATH_NOTES.md` Section 9-redesign for the performance rationale and the
    empirical before/after equivalence check). `return_dual=True` additionally returns the
    converged (z, u) for the caller to feed into the NEXT window's warm start
    (`l1_trend_filter_rolling` is the only caller that does this). Default behavior
    (`z_init=None, u_init=None, return_dual=False`) is BYTE-IDENTICAL to the original
    zero-seeded, x-only-return implementation -- every existing caller/test is unaffected.
    """
    n = len(y)
    if n < 5:
        x0 = y.astype("float64").copy()
        if return_dual:
            return x0, np.zeros(max(n - 2, 0)), np.zeros(max(n - 2, 0))
        return x0
    c, low, D = _l1tf_cholesky_factor(n, float(rho))
    y = y.astype("float64")
    # z, u seeded at ZERO (not z = D @ y) -- seeding z = D @ x0 with x0 = y is a false-convergence
    # trap: the very first x-update then reproduces x_new = y EXACTLY (algebraically:
    # rhs = y + rho*D^T@D@y = (I + rho*D^T@D)@y, so x_new = (I+rho*D^T@D)^-1 @ rhs = y), which
    # makes the ||x_new - x||-based stopping check fire after a single iteration -- before z/u
    # have moved at all and before the l1 penalty has had any effect on x. Confirmed empirically:
    # with the z=D@y seeding, l1_trend_filter_window(y, lam) returned y UNCHANGED for every lam
    # tested from 0.5 to 20. Zero-seeding avoids this degenerate fixed point. A caller-supplied
    # z_init/u_init (warm start from an ADJACENT, already-solved window) does not reintroduce this
    # trap because it is never algebraically equal to D@y for the CURRENT window.
    x = y.copy()
    z = np.zeros(n - 2) if z_init is None else np.asarray(z_init, dtype="float64").copy()
    u = np.zeros(n - 2) if u_init is None else np.asarray(u_init, dtype="float64").copy()
    Dt = D.T
    for _ in range(n_iter):
        rhs = y + rho * (Dt @ (z - u))
        x_new = cho_solve((c, low), rhs)
        Dx = D @ x_new
        v = Dx + u
        z_new = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0.0)
        u = u + Dx - z_new
        converged = np.max(np.abs(x_new - x)) < tol
        x, z = x_new, z_new
        if converged:
            break
    if return_dual:
        return x, z, u
    return x


def _robust_noise_scale(y: np.ndarray) -> float:
    """MAD-based robust scale estimate of `diff(y)`, used to calibrate `lam` per-window in
    "auto" mode (see `l1_trend_filter_rolling`). `lam` is an ABSOLUTE threshold (units of `y`, cf.
    the (1/2)||y-x||^2 + lam*||Dx||_1 objective) -- a single fixed lam calibrated for one asset's
    price scale/volatility is not portable to another (empirically confirmed: lam=3.0, a
    reasonable-looking round number, turned out to force a near-maximum-regularization single
    straight-line fit -- zero interior kinks -- on real daily log-price data with ~1-2%/day
    volatility, defeating the entire point of PIECEWISE-linear trend filtering; see the paired
    calibration note in `l1_trend_filter_rolling`). A MAD-based (median absolute deviation, scaled
    by the usual 1.4826 constant for consistency with the Gaussian std under the null of no
    outliers) scale is the standard robust-statistics device for exactly this kind of
    data-adaptive threshold calibration (the same logic behind the Donoho-Johnstone universal
    threshold in wavelet shrinkage, lambda ~ sigma*sqrt(2 log n)); MAD (rather than plain std) is
    used because the window can itself contain a real, large trend move which would otherwise
    inflate a plain std-based noise estimate.
    """
    d = np.diff(y)
    if len(d) == 0:
        return 1e-8
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    scale = 1.4826 * mad
    return scale if scale > 1e-12 else (float(np.std(d)) or 1e-8)


def l1_trend_filter_rolling(log_price: pd.Series, window: int = 120, lam: float | None = None,
                             lam_mult: float = 1.0, rho: float = 1.0, edge_offset: int = 0,
                             n_iter: int = 150) -> pd.Series:
    """Causal rolling adaptation of Kim-Koh-Boyd l1 trend filtering.

    LAMBDA CALIBRATION: `lam` is in the same units as `log_price` (an absolute threshold on
    second differences), which does not generalize across names/regimes of differing volatility.
    The default (`lam=None`) instead computes, INSIDE each causal window (so still exactly
    causal), `lam := lam_mult * _robust_noise_scale(window)` -- i.e. `lam_mult` is the actual
    portable, dimensionless free parameter exposed to the sweep (see the markdown section's grid),
    playing the same role a z-score threshold multiplier plays in `normalize_signal`. Pass an
    explicit numeric `lam` to override auto-calibration (used only by the low-level unit tests
    against small synthetic windows of known, hand-picked scale).

    THE APPROXIMATION THIS INTRODUCES (be explicit, as required): the original method is a
    single GLOBAL convex program over the entire (fixed, fully-observed) series -- every fitted
    point, including the first, uses the ENTIRE sample, so it is fundamentally non-causal and
    cannot be used online at all. The adaptation here re-solves the SAME QP independently on each
    causal window `log_price[t-window+1 : t+1]` (using only data through t) and reports one point
    of that window's fit as the trend estimate assigned to time t. Two consequences of this
    windowing that would not exist in the offline estimator:
      (a) INFORMATION LOSS: each window "forgets" everything before its start, so the roughly
          global-optimal placement of kink points that the true Kim-Koh-Boyd objective would find
          over the full history is replaced by a sequence of locally-optimal placements that can
          disagree window-to-window on where a trend actually changed.
      (b) BOUNDARY-EFFECT INSTABILITY: the l1-trend-filter penalty constrains D x (second
          differences); the LAST point of any finite window is asymmetric -- it does not have a
          "future" neighbor to help two-sidedly regularize its slope the way an INTERIOR point
          does, so the raw edge estimate (edge_offset=0) is measurably noisier/more reactive to
          the single newest observation than an interior fit would be. `edge_offset` trades this
          off explicitly and REMAINS FULLY CAUSAL either way (both choices only ever read from a
          window that ends at t; the offset only changes which position WITHIN that
          already-causal window is reported as "the trend value at t"): edge_offset=0 uses the
          literal newest fitted point (most current, noisiest); edge_offset=k>0 reports the fit
          k points inside the window's right edge (better-anchored by neighbors on both sides
          within the window, at the cost of k days of extra effective smoothing/lag). Both are
          exposed; the default (edge_offset=0) prioritizes currency and lets the multi-horizon
          combination step (Section 4) average away the resulting extra noise against the other,
          less edge-sensitive component signals.

    Returns the causal trend LEVEL series (same index as log_price, NaN before `window` real
    observations are available).

    PERFORMANCE NOTE (does not change the math -- see `research/cta/CTA_MATH_NOTES.md` Section
    9-redesign): consecutive windows overlap in n-1 of their n points, so each day's ADMM solve is
    WARM-STARTED from the immediately preceding day's converged (z, u), shifted by one position to
    re-align local second-difference indices with the new window (the oldest interior
    second-difference is dropped, the newest is seeded at the previous edge value -- a standard,
    disclosed sliding-window warm-start heuristic). Because the per-window objective is strictly
    convex in x, ADMM converges to the SAME unique global optimum regardless of (z, u)
    initialization (see `l1_trend_filter_window`'s docstring) -- warm-starting only reduces the
    number of iterations needed to reach `tol`, it never changes the converged trend estimate.
    State is reset to a cold (zero) start after any NaN gap (stale state from a non-adjacent
    window would not correspond to the new window's local indexing). This is the single largest
    real-world compute cost identified in this program's Stage-A per-name signal-generation
    profiling (~84% of one name's total signal-build time before this fix) -- see the redesign
    section for the measured before/after speedup.
    """
    vals = log_price.to_numpy(dtype="float64")
    n = len(vals)
    out = np.full(n, np.nan)
    if edge_offset < 0 or edge_offset >= window:
        raise ValueError("edge_offset must be in [0, window)")
    pos = window - 1 - edge_offset
    z_prev: np.ndarray | None = None
    u_prev: np.ndarray | None = None
    for t in range(window - 1, n):
        w = vals[t - window + 1: t + 1]
        if np.any(np.isnan(w)):
            z_prev = u_prev = None
            continue
        local_lam = lam if lam is not None else lam_mult * _robust_noise_scale(w)
        if z_prev is not None and len(z_prev) == window - 2:
            # shift by one: local index i (old window) <-> local index i-1 (new window); the
            # newest interior second-difference has no predecessor, seed it with the previous
            # edge value (better than zero, still just a warm start, never part of the objective).
            z0 = np.empty_like(z_prev)
            z0[:-1] = z_prev[1:]
            z0[-1] = z_prev[-1]
            u0 = np.empty_like(u_prev)
            u0[:-1] = u_prev[1:]
            u0[-1] = u_prev[-1]
        else:
            z0 = u0 = None
        xhat, z_new, u_new = l1_trend_filter_window(
            w, lam=local_lam, rho=rho, n_iter=n_iter, z_init=z0, u_init=u0, return_dual=True,
        )
        out[t] = xhat[pos]
        z_prev, u_prev = z_new, u_new
    return pd.Series(out, index=log_price.index)


def l1_trend_filter_slope_signal(log_price: pd.Series, window: int = 120, lam: float | None = None,
                                  lam_mult: float = 1.0, rho: float = 1.0, edge_offset: int = 0,
                                  slope_lookback: int = 5) -> pd.Series:
    """Raw (not yet normalized) trend-direction signal from the causal l1 trend filter: the
    finite-difference slope of the filtered level over `slope_lookback` days. Causal by
    construction (built entirely from `l1_trend_filter_rolling`'s causal output).
    """
    level = l1_trend_filter_rolling(log_price, window=window, lam=lam, lam_mult=lam_mult, rho=rho,
                                     edge_offset=edge_offset)
    return (level - level.shift(slope_lookback)) / slope_lookback


# ======================================================================================
# 4. MULTI-HORIZON COMBINATION: Grinold-Kahn (Fundamental Law of Active Management) motivated,
#    IC/breadth-based combination weights, with shrinkage toward equal weight.
# ======================================================================================

def rolling_information_coefficient(signal: pd.Series, log_price: pd.Series,
                                     window: int) -> pd.Series:
    """Causal, trailing (expanding-then-rolling) estimate of the information coefficient (Pearson
    correlation, per Grinold & Kahn's own convention -- "IC" throughout their Fundamental Law) of
    a real-time signal against the FORWARD one-period return it is meant to predict.

    Alignment (must be gotten right to remain causal -- see module docstring):
      fwd_return.loc[t] := log_price_{t+1} - log_price_t   (the return signal_t is scored against)
      raw_ic.loc[t]     := corr( signal[t-window+1:t+1], fwd_return[t-window+1:t+1] )
                            -- note raw_ic.loc[t] itself needs fwd_return.loc[t], i.e. log_price
                            at t+1, so raw_ic.loc[t] is not yet knowable at close t.
      returned.loc[t]   := raw_ic.loc[t-1]
                            -- knowable at close t (its newest ingredient is fwd_return.loc[t-1],
                            i.e. log_price at t, known at close t). This one extra `.shift(1)` is
                            exactly bug-pattern-(c)'s decide-at-close(t-1)/act-at(t) discipline,
                            applied to a *diagnostic about a signal* rather than to the signal
                            itself -- the same discipline, one level up.

    Returns NaN until `window` complete (signal, fwd_return) pairs are available.
    """
    fwd_return = log_price.shift(-1) - log_price
    raw_ic = signal.rolling(window, min_periods=window).corr(fwd_return)
    return raw_ic.shift(1)


def combination_weights_from_ic(trailing_ic: pd.DataFrame, shrinkage: float = 0.5) -> pd.DataFrame:
    """Combination weight for each of K component trend signals, motivated by Grinold & Kahn's
    Fundamental Law of Active Management (IR ~= IC * sqrt(BR)) and derived in full in the markdown
    section via the classical GLS/regression argument (optimal linear combination of unit-variance
    forecasts weights each by its inverse-covariance-adjusted IC; with the covariance among
    highly-overlapping trend speeds too noisily estimable from the handful of independent
    "breadth" opportunities available here, this reduces -- as derived -- to weighting each
    component in proportion to its own trailing IC, floored at zero: since a stable, exploitable
    NEGATIVE information coefficient is intrinsically much less plausible than estimation noise
    for a momentum-family signal at this sample size, a negative trailing IC is treated as "no
    information" rather than "trade the opposite sign").

    `shrinkage` (in [0,1]) convexly blends the raw IC-proportional weight with the equal weight
    (1/K each): `shrinkage=0` is pure IC-weighting (fragile: Established Fact (5) is precisely a
    demonstration of single-timeframe argmax-style over-commitment failing out-of-sample), and
    `shrinkage=1` is the naive equal-weight ensemble. `shrinkage=0.5` (this module's chosen
    default -- Section-end summary) is a standard Ledoit-Wolf-style compromise: enough IC-tilt to
    reward genuinely more-informative horizons, without the raw-IC estimator's full sampling
    variance controlling the whole ensemble the way a single argmax-selected timeframe would.

    Returns a DataFrame with the same index/columns as `trailing_ic`.
    """
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")
    k = trailing_ic.shape[1]
    pos_ic = trailing_ic.clip(lower=0.0)
    row_sum = pos_ic.sum(axis=1)
    raw_w = pos_ic.div(row_sum.replace(0.0, np.nan), axis=0)
    raw_w = raw_w.fillna(1.0 / k)
    equal_w = pd.DataFrame(1.0 / k, index=trailing_ic.index, columns=trailing_ic.columns)
    return shrinkage * equal_w + (1.0 - shrinkage) * raw_w


DEFAULT_EWMA_HALFLIVES: tuple[float, ...] = (8.0, 32.0, 128.0)
DEFAULT_KALMAN_PARAMS: dict = {"q_level": 1e-5, "q_slope": 1e-7, "r": 1e-4}
DEFAULT_L1TF_PARAMS: dict = {"window": 120, "lam": None, "lam_mult": 1.0, "rho": 1.0,
                             "edge_offset": 0, "slope_lookback": 5}


def build_component_signals(log_price: pd.Series, ewma_halflives: Iterable[float] = DEFAULT_EWMA_HALFLIVES,
                             kalman_params: dict = DEFAULT_KALMAN_PARAMS,
                             l1tf_params: dict = DEFAULT_L1TF_PARAMS,
                             vol_window: int = 20) -> pd.DataFrame:
    """Build the raw (real-time, NOT exposure-shifted) normalized component signals that
    `multi_horizon_trend_signal` combines. Exposed separately so an integration agent, or the
    injection test, can inspect/validate each component in isolation.

    Columns: `ewma_{hl}` for each halflife in `ewma_halflives`, `kalman_slope`, `l1tf_slope`.
    """
    cols = {}
    for hl in ewma_halflives:
        raw = ewma_return_momentum(log_price, hl)
        cols[f"ewma_{hl:g}"] = normalize_signal(raw, vol_window=vol_window)
    kf = kalman_local_linear_trend(log_price, **kalman_params)
    cols["kalman_slope"] = normalize_signal(kf["slope"], vol_window=vol_window)
    l1_slope = l1_trend_filter_slope_signal(log_price, **l1tf_params)
    cols["l1tf_slope"] = normalize_signal(l1_slope, vol_window=vol_window)
    return pd.DataFrame(cols, index=log_price.index)


def multi_horizon_trend_signal(log_price: pd.Series, ewma_halflives: Iterable[float] = DEFAULT_EWMA_HALFLIVES,
                                kalman_params: dict = DEFAULT_KALMAN_PARAMS,
                                l1tf_params: dict = DEFAULT_L1TF_PARAMS,
                                vol_window: int = 20, ic_window: int = 252,
                                shrinkage: float = 0.5) -> pd.DataFrame:
    """TOP-LEVEL DELIVERABLE for Axis A. Multi-horizon trend-signal combination: replaces the
    fragile fixed-SMA single-timeframe argmax (Established Fact (5)) with a Grinold-Kahn-motivated
    IC-weighted ensemble across several causal trend-estimation methods and speeds.

    EXPOSURE-READY: the returned `combined` column is `.shift(1)`'d relative to the underlying
    real-time components, i.e. `combined.loc[t]` depends only on log_price through t-1's close and
    is safe to use directly as (or scale into) an exposure applied to day t's return. This holds
    for BOTH of the two lags actually stacked here: (i) `rolling_information_coefficient`'s own
    internal shift (so the WEIGHTS at t only use fully-realized (signal, fwd_return) pairs through
    t-1), and (ii) the additional explicit `.shift(1)` applied to the raw combined score itself
    below (so the SCORE at t is also computed from data through t-1, not t). Verified by
    `synthetic_outlier_injection_test`.

    Returns a DataFrame with: one column per raw component (see `build_component_signals`, NOT
    exposure-shifted -- included for diagnostics only), one `weight_*` column per component (the
    IC-based weight actually applied, already causal per (i) above), and `combined` (the final
    exposure-ready ensemble score, causal per (i)+(ii) above; NOT yet direction/vol-target scaled
    into a position -- that is an integration-layer decision downstream of this module).
    """
    components = build_component_signals(log_price, ewma_halflives=ewma_halflives,
                                          kalman_params=kalman_params, l1tf_params=l1tf_params,
                                          vol_window=vol_window)
    ic = pd.DataFrame({c: rolling_information_coefficient(components[c], log_price, ic_window)
                        for c in components.columns})
    weights = combination_weights_from_ic(ic, shrinkage=shrinkage)
    raw_combined = (components * weights).sum(axis=1, skipna=True)
    # zero out days where every component/weight pair is NaN (no real ensemble available yet)
    all_nan = components.isna().all(axis=1) | weights.isna().all(axis=1)
    raw_combined = raw_combined.mask(all_nan, np.nan)

    out = components.copy()
    for c in components.columns:
        out[f"weight_{c}"] = weights[c]
    out["combined"] = raw_combined.shift(1)
    return out


# ======================================================================================
# 5. VALIDATION: synthetic-outlier-injection causality test (+ naive unlagged control).
# ======================================================================================

def _naive_unlagged_ewma_signal(log_price: pd.Series, halflife: float) -> pd.Series:
    """Deliberately-WRONG control used only to prove the injection test has teeth: a same-day
    z-scored EWMA momentum signal that normalizes by a rolling std WITHOUT the causal `.shift(1)`
    (i.e. it lets day t's own shock inflate/inform its own normalizer), and — the actual causality
    violation being tested for — uses `center()`-style two-sided smoothing on the return series
    instead of the real (one-sided, recursive) EWMA, so a shock at index i leaks into the control
    signal's value at i-1 as well as i. This function must NEVER be used for anything but the
    injection-test control below.
    """
    r = log_price.diff()
    # two-sided (non-causal) rolling mean -- centered window leaks future info into t
    win = max(3, int(round(halflife)))
    raw = r.rolling(win, center=True, min_periods=1).mean()
    std = raw.rolling(20, min_periods=1).std()  # NOT shifted -- t's own value informs its own scale
    return raw / std.replace(0.0, np.nan)


def synthetic_outlier_injection_test(log_price: pd.Series, shock_index: int | None = None,
                                      shock_size: float = 0.5) -> dict:
    """Injects one large fake log-price move (`shock_size`, e.g. 0.5 = +50% log move) at
    `shock_index` (default: the midpoint of the series) into a COPY of real `log_price`, re-runs
    every causal component plus the full `multi_horizon_trend_signal` ensemble on the shocked
    series, and confirms:
      (a) every RAW component's first reacting index is >= shock_index (never earlier -- a raw,
          t-inclusive real-time filter is allowed to react AT the shock, just never before it);
      (b) the EXPOSURE-READY `combined` column's first reacting index is >= shock_index + 1 (the
          one extra day of lag required by the decide-at-close/act-at-t+1 convention);
      (c) the NAIVE, deliberately non-causal control (`_naive_unlagged_ewma_signal`) DOES react
          strictly before `shock_index` (proves the test methodology actually discriminates
          causal from non-causal code -- a test that both real and broken implementations always
          pass would be a useless tautology).

    Returns a dict of per-component first-reacting-index diagnostics and the three pass/fail
    booleans above (`raw_causal_ok`, `exposure_causal_ok`, `naive_control_reacts_early`).
    """
    n = len(log_price)
    if shock_index is None:
        shock_index = n // 2
    base = log_price.copy()
    shocked = base.copy()
    shocked.iloc[shock_index:] = shocked.iloc[shock_index:] + shock_size  # permanent level shock

    def first_diff_index(a: pd.Series, b: pd.Series, atol: float = 1e-9) -> int | None:
        diff = (a - b).abs()
        reacted = diff[diff > atol]
        return int(base.index.get_loc(reacted.index[0])) if len(reacted) else None

    results = {}

    # (a) raw components
    comp_base = build_component_signals(base)
    comp_shock = build_component_signals(shocked)
    raw_ok = True
    for c in comp_base.columns:
        idx = first_diff_index(comp_base[c], comp_shock[c])
        results[f"raw_first_react::{c}"] = idx
        if idx is not None and idx < shock_index:
            raw_ok = False

    # (b) exposure-ready ensemble
    ens_base = multi_horizon_trend_signal(base)
    ens_shock = multi_horizon_trend_signal(shocked)
    idx_combined = first_diff_index(ens_base["combined"], ens_shock["combined"])
    results["exposure_first_react::combined"] = idx_combined
    exposure_ok = (idx_combined is None) or (idx_combined >= shock_index + 1)

    # (c) naive unlagged control -- must react STRICTLY BEFORE shock_index to prove test has teeth
    naive_base = _naive_unlagged_ewma_signal(base, halflife=32.0)
    naive_shock = _naive_unlagged_ewma_signal(shocked, halflife=32.0)
    idx_naive = first_diff_index(naive_base, naive_shock)
    results["naive_control_first_react"] = idx_naive
    naive_reacts_early = (idx_naive is not None) and (idx_naive < shock_index)

    results["shock_index"] = shock_index
    results["raw_causal_ok"] = raw_ok
    results["exposure_causal_ok"] = exposure_ok
    results["naive_control_reacts_early"] = naive_reacts_early
    results["overall_pass"] = bool(raw_ok and exposure_ok and naive_reacts_early)
    return results
