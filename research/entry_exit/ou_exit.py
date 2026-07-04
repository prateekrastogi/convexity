"""MODELS.ou_exit -- OU / Leung-Li mean-reversion optimal-exit candidate (EDGE #6, SIGNAL §C.3(b)).

The pre-registered EXIT / dynamic-HOLD candidate #6 (re-admitted 2026-07-04, directional exit;
docs/EDGE.md "The pre-registered candidate set (frozen)", SIGNAL §C.3(b), SYSTEM §C.9 #6). It
REPLACES the incumbent's fixed 5-day horizon with a mean-reversion take-profit boundary while the
true -12% 1-minute premium stop stays the RETAINED loss floor (``base.STOP_FLOOR``; this model
only decides WHEN a still-alive winner is voluntarily banked, never loosens the loss cap -- §C.3).

FORMAL model (stated assumptions / estimator / error-control -- a PhD-level model, NOT indicator):

  * GENERATING PROCESS. The exit-relevant quantity ``X_t`` (the running position return / a
    mean-reverting value component) is assumed **Ornstein-Uhlenbeck**
    ``dX = theta (m - X) dt + sigma dW``  (SIGNAL §C.3(b) notation: speed ``theta``, long-run mean
    ``m``, vol ``sigma``). Exact discrete transition: for spacing ``dt`` the law of ``X_{i+1}|X_i``
    is Gaussian, so OU is the AR(1) ``X_{i+1} = a + b X_i + eps`` with ``b = exp(-theta dt)``,
    ``a = m (1 - b)``, ``Var(eps) = sigma^2 (1 - b^2) / (2 theta)``. This exactness is why the MLE
    below is closed-form (the conditional Gaussian likelihood is maximized by the AR(1) OLS fit).

  * ESTIMATOR (train-fit, then FROZEN -- SYSTEM §C.1). ``fit`` runs the exact conditional-Gaussian
    MLE for ``(theta, m, sigma)`` on the TRAIN fold's per-day series (the AR(1) closed form above),
    with a STATIONARITY CHECK: the OU is stationary iff ``theta > 0`` (equivalently ``0 < b < 1``);
    a non-stationary / degenerate train fit (``b <= 0``, ``b >= 1``, flat series) is REFUSED
    (``OUStationarityError``) rather than silently returning a boundary from a mis-specified fit.
    The discount rate ``r`` and net sale cost ``c`` are pre-registered constructor args (NOT fit on
    data -- an error-control / no-snoop choice; SIGNAL §C.3(b) "the discount rate ... frozen").

  * OPTIMAL BOUNDARY (Leung-Li 2015, "Optimal Mean Reversion Trading", the F/G value-function ODE).
    Under the OU law the optimal single-liquidation problem ``sup_tau E[e^{-r tau}(X_tau - c)]`` has
    value ``V(x) = (x - c)`` for ``x >= b*`` (stop, take the reward) and ``V(x) = A F(x)`` for
    ``x < b*`` (continue), where ``F`` is the increasing fundamental solution of the OU
    infinitesimal generator at discount ``r``:
        ``F(x) = int_0^inf u^{r/theta - 1} exp( sqrt(2 theta)/sigma (x - m) u  -  u^2/2 ) du``.
    Smooth-fit at the free boundary ``b*`` gives the paper's OPTIMALITY CONDITION
        ``F(b*) = (b* - c) F'(b*)``  (value-matching + smooth-pasting collapsed to one equation),
    with ``F'`` the derivative in ``x``. ``b*`` is the SMALLEST root of ``h(b) = F(b) - (b-c)F'(b)``
    above ``m`` (the take-profit level a reverting winner is banked at). Solved ONCE on the frozen
    train ``(theta, m, sigma)`` and FROZEN for OOS -- a deterministic function of the fitted params.

  * SIGNAL (causal, one-step-ahead -- SIGNAL §C.1 / SYSTEM §C.2 PURITY, binding). ``exit_now``
    liquidates a still-alive winner on the FIRST ``F_{t-1}``-measurable crossing of ``b*`` by the
    FILTERED state (here ``held_return`` = the running return observed up to the prior bar, the
    ExitModel decision covariate on ``base.ExitModel``). It reads ``held_return`` + ``cursor.past``
    ONLY; a same-bar / future read routes through ``cursor.read`` and RAISES ``PurityError`` (the
    pinning hook). The boundary is a function of the FROZEN train fit, never re-tuned on TEST, never
    a smoothed / full-sample estimate.

PARSIMONY (SYSTEM §C.4). ``k = 3`` free parameters fit on train (``theta, m, sigma``); the discount
``r`` and cost ``c`` are pre-registered (not fit) so they are NOT counted, and there is no latent
state dimension (the OU state is the observed running return). The incumbent (fixed hold + stop,
minimal ``k``) is the parsimony floor; this candidate must beat it OOS by more than its ``k`` cost.

Imports ONLY ``intraday.models.base`` (the interface + shared causal helpers) -- NEVER a sibling
Phase-C model module (SYSTEM §C.3 interface-driven-harness / off-ledger rule). Pure numpy + stdlib
``math`` (scipy is NOT a dependency -- the F/G quadrature and the root solve are rolled here, like
SYSTEM's scipy-free ``deflated_sharpe``). Moneyness, not greeks (CLAUDE.md). Synthetic-tested by a
KNOWN-OU-path recovery test + a no-lookahead pinning test (tests/models/test_ou_exit.py); the
empirical selection RUN is BLOCKED-ON-LAKE (``base.BlockedOnLakeError``). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import STOP_FLOOR, Cadence, ScanCursor

__all__ = [
    "STOP_FLOOR",
    "OUParams",
    "OUStationarityError",
    "ou_mle",
    "fg_F",
    "fg_F_prime",
    "leung_li_boundary",
    "OUExit",
]

# --- Quadrature / root-solve controls (fixed error-control, NOT data-fit, NOT grid-searched) ------
# The F/G integrand `u^{r/theta-1} exp(z u - u^2/2)` decays like a Gaussian in `u`; a fixed-node
# composite Simpson rule on a generous upper limit resolves it to ~1e-10 for the parameter ranges a
# bounded running return produces. These are numerical tolerances (like SYSTEM's Acklam constants),
# not model parameters: they do not enter `k` and are never tuned on data.
_QUAD_NODES = 4000        # Simpson panels over the u-integral (even count)
_QUAD_UPPER = 40.0        # upper limit; integrand is < 1e-300 well before this for the ranges used
_ROOT_TOL = 1e-10         # bisection tolerance on the boundary root
_ROOT_MAXIT = 200         # bisection iterations (2^-200 << _ROOT_TOL; a hard safety cap)
_STAT_EPS = 1e-9          # below this |b| deviation the AR(1) slope is treated as a degenerate edge
# Unit-root (stationarity) screen: the Dickey-Fuller t-statistic `(b - 1) / se(b)` on the AR(1) fit.
# H0 = unit root (random walk, non-stationary); the OU is stationary only if we REJECT it, i.e. the
# t-stat is more negative than the DF critical value. -2.86 is the standard DF 5% critical value for
# the constant-no-trend case (a fixed error-control threshold from the DF distribution, NOT tuned on
# data): a random walk gives t ~ -1.5 to -2 (fails to reject -> refused); a genuine OU gives t of
# order -10 or more negative (rejects -> stationary). This is the textbook near-unit-root screen.
_DF_CRIT = -2.86


class OUStationarityError(ValueError):
    """The train fit is NOT a stationary OU -- refuse a boundary rather than fit a mis-specified OU.

    Raised by ``ou_mle`` / ``OUExit.fit`` when the AR(1) slope ``b = exp(-theta dt)`` does not lie
    in ``(0, 1)`` (i.e. ``theta <= 0``: no mean reversion, a random walk or explosive path) or the
    train series is constant / too short for a transition-pair regression. The Leung-Li boundary is
    only defined for a mean-reverting (stationary) OU, so a non-stationary train fold is a REFUSE,
    not a silently-returned boundary from a mis-specified model (SIGNAL §C.3(b) stationarity check).
    A ``ValueError`` subclass so a caller may catch it as the malformed-fit signal it is.
    """


@dataclass(frozen=True)
class OUParams:
    """The frozen OU maximum-likelihood fit ``dX = theta (m - X) dt + sigma dW`` (SIGNAL §C.3(b)).

    ``theta`` mean-reversion speed (``> 0`` for a stationary OU -- the stationarity check enforces),
    ``m`` long-run mean, ``sigma`` diffusion vol (``> 0``), ``dt`` the sampling spacing the MLE used
    (in the same time unit ``r``/``theta`` are quoted in; default 1.0 = one lake session day). These
    are the ONLY data-fit quantities (``k = 3``: ``theta, m, sigma``); ``dt`` is a fixed cadence,
    not a fitted parameter.
    """

    theta: float
    m: float
    sigma: float
    dt: float = 1.0

    def __post_init__(self) -> None:
        if not (self.theta > 0.0):
            raise OUStationarityError(f"OU not stationary: theta={self.theta} must be > 0")
        if not (self.sigma > 0.0):
            raise ValueError(f"OU sigma must be > 0 (got {self.sigma})")
        if not (self.dt > 0.0):
            raise ValueError(f"OU dt must be > 0 (got {self.dt})")


# ==================================================================================================
# Exact conditional-Gaussian MLE for the OU parameters (closed form via the AR(1) representation)
# ==================================================================================================
def ou_mle(series: tuple[float, ...] | np.ndarray, dt: float = 1.0) -> OUParams:
    """Exact discrete MLE of ``(theta, m, sigma)`` for ``dX = theta(m-X)dt + sigma dW`` (train-fit).

    The OU transition ``X_{i+1} | X_i`` is exactly Gaussian, so OU IS the AR(1)
    ``X_{i+1} = a + b X_i + eps``, ``b = exp(-theta dt)``, ``a = m (1 - b)``,
    ``Var(eps) = sigma^2 (1 - b^2) / (2 theta)``. Maximizing the conditional Gaussian likelihood is
    therefore exactly the OLS regression of ``X_{i+1}`` on ``X_i`` (slope ``b``, intercept ``a``)
    plus the residual-variance estimate -- a closed form, no iteration. Recovers:
        ``theta = -ln(b) / dt``,   ``m = a / (1 - b)``,
        ``sigma = sqrt( Var(eps) * 2 theta / (1 - b^2) )``.

    STATIONARITY CHECK (SIGNAL §C.3(b)): a stationary mean-reverting OU needs ``0 < b < 1`` (i.e.
    ``theta > 0``). But a finite random-walk / near-unit-root sample yields ``b`` barely below 1
    (a spurious finite-sample reversion toward the sample mean, ``theta ~ 0``) which passes a naive
    ``b < 1`` test yet is NOT a usable mean-reverting fit -- the estimated mean ``m`` diverges far
    outside the data range. The check therefore ALSO runs the standard **Dickey-Fuller unit-root
    test** on the AR(1) slope: the t-statistic ``t_DF = (b - 1) / se(b)`` must be MORE NEGATIVE than
    the DF 5% critical value ``_DF_CRIT = -2.86`` (constant-no-trend case) to REJECT the unit-root
    null and admit the OU as stationary. A random walk gives ``t_DF ~ -1.5..-2`` (fails to reject ->
    refused); a genuine OU gives ``t_DF`` of order ``-10`` or more (rejects -> stationary). A
    degenerate / non-stationary fit (``b <= 0`` alternating/negative; ``b >= 1`` explosive; a
    near-unit-root ``b`` that fails the DF test; a constant series with no ``Var(X_i)``) RAISES
    ``OUStationarityError`` -- the boundary is only defined for a reverting OU, so a mis-specified
    train fold is refused, never silently used. ``dt`` is the (fixed) sampling spacing; the estimate
    is on-fold only (no pre-sample borrow; SIGNAL §C.1 warm-up honesty is the caller's `warmup`).
    """
    x = np.asarray(series, dtype="float64")
    if x.size < 3:
        raise OUStationarityError("OU MLE needs >= 3 samples (>= 2 transition pairs)")
    x0, x1 = x[:-1], x[1:]  # (X_i, X_{i+1}) transition pairs
    n = x0.size
    var_x0 = x0.var(ddof=0)
    if var_x0 <= _STAT_EPS:
        raise OUStationarityError("OU MLE degenerate: constant/near-constant train series (no Var)")
    # Closed-form OLS: b = Cov(X_i, X_{i+1}) / Var(X_i); a = mean(X_{i+1}) - b mean(X_i).
    mx0, mx1 = x0.mean(), x1.mean()
    s_xx = float(((x0 - mx0) ** 2).sum())
    b = float(((x0 - mx0) * (x1 - mx1)).sum() / s_xx)
    a = float(mx1 - b * mx0)
    # Stationarity: mean-reverting OU <=> 0 < b < 1 (theta > 0). Refuse anything else.
    if not (_STAT_EPS < b < 1.0 - _STAT_EPS):
        raise OUStationarityError(
            f"OU not stationary: AR(1) slope b={b:.6g} must lie in (0, 1) "
            f"(theta = -ln(b)/dt > 0); b<=0 or b>=1 is a random-walk/explosive/degenerate fit")
    theta = -math.log(b) / dt
    resid = x1 - (a + b * x0)
    # Dickey-Fuller unit-root screen: t_DF = (b-1)/se(b) must reject the random-walk null (< crit).
    # se(b) = sqrt( s2 / Sxx ), s2 the OLS residual variance (ddof=2: 2 fitted params a, b).
    if n > 2:
        s2_ols = float((resid ** 2).sum() / (n - 2))
        se_b = math.sqrt(s2_ols / s_xx) if s2_ols > 0.0 else 0.0
        t_df = (b - 1.0) / se_b if se_b > 0.0 else -math.inf  # exactly-linear fit -> reject
        if t_df > _DF_CRIT:
            raise OUStationarityError(
                f"OU near unit root: Dickey-Fuller t={t_df:.3g} > crit {_DF_CRIT} (b={b:.6g}); "
                f"fails to reject the random-walk null -- a non-stationary fit, refused")
    m = a / (1.0 - b)
    # Residual variance of the AR(1) fit -> sigma via Var(eps) = sigma^2 (1 - b^2)/(2 theta).
    var_eps = float((resid ** 2).sum() / n)  # MLE (ddof=0) conditional-Gaussian residual variance
    sigma_sq = var_eps * (2.0 * theta) / (1.0 - b * b)
    sigma = math.sqrt(sigma_sq) if sigma_sq > 0.0 else 0.0
    if not (sigma > 0.0):
        raise OUStationarityError("OU MLE degenerate: zero residual diffusion (sigma == 0)")
    return OUParams(theta=theta, m=m, sigma=sigma, dt=dt)


# ==================================================================================================
# Leung-Li (2015) F/G value functions and the optimal-liquidation free boundary b*
# ==================================================================================================
def _fg_core(
    theta: float, sigma: float, m: float, r: float, x: float, sign: float
) -> tuple[float, float, float]:
    """Simpson quadrature of the F/G kernel and its x-derivative at ``x`` (``sign=+1`` F, ``-1`` G).

    Returns the PEAK-CENTERED (log-sum-exp) triple ``(log_peak, value_norm, deriv_norm)`` so that
        ``F(x)  = exp(log_peak) * value_norm``,   ``F'(x) = exp(log_peak) * deriv_norm``.
    ``power = r/theta - 1`` is the ``u`` exponent; ``z = c0 (x-m)`` with ``c0 = sqrt(2 theta)/sigma``;
    ``dz/dx = c0``. Integrand ``g(u) = u^{power} exp( sign*z*u - u^2/2 )``; its x-derivative is
    ``sign*c0*u*g(u)``. Both share the node grid, so ONE Simpson pass yields value and derivative
    (the smooth-fit condition needs both).

    ERROR CONTROL -- overflow-safe by construction (defect fix). The log-integrand
    ``log_g(u) = power*ln(u) + sign*z*u - u^2/2`` peaks near ``u ~= z`` with height ``~ z^2/2``; for a
    moderate discount ``r`` (``power`` grows) or a larger state ``x`` (``z`` grows) that height
    exceeds ~709 and a NAIVE ``exp(log_g)`` overflows float64 -- silently fabricating a boundary from
    inf/nan quadrature (the reported high-severity defect). We therefore FACTOR OUT the log-domain
    peak ``M = max(log_g)`` and integrate the bounded, O(1) kernel ``exp(log_g - M)`` (log-sum-exp).
    ``M`` is returned SEPARATELY as ``log_peak`` (never re-exponentiated inside quadrature), so the
    Simpson sums ``value_norm``/``deriv_norm`` stay finite for EVERY valid ``(r, c, x)`` in the OU's
    parameter band, not just the narrow default ``r`` window. Callers that need the true magnitude
    (``fg_F`` / ``fg_F_prime``) multiply by ``exp(log_peak)`` once; the smooth-fit ROOT solve consumes
    only the shared-``exp(log_peak)`` ratio (below), whose SIGN is that of the normalized bracket, so
    the boundary solve never touches an overflowing magnitude at all.

    Node ``u = 0`` contributes 0 for ``power > -1`` (``r > 0`` guarantees it) and is handled by
    starting the grid at a tiny positive epsilon to keep ``u^{power}`` finite when ``power < 0``.
    """
    c0 = math.sqrt(2.0 * theta) / sigma
    z = c0 * (x - m)
    power = r / theta - 1.0
    # Simpson grid on (0, _QUAD_UPPER]; start just above 0 so u^power stays finite for power < 0.
    n = _QUAD_NODES
    u = np.linspace(0.0, _QUAD_UPPER, n + 1)
    u[0] = min(1e-12, _QUAD_UPPER / (n * 1e3))  # tiny positive lower node (avoids 0^negative)
    log_g = power * np.log(u) + (sign * z) * u - 0.5 * u * u
    # Peak-centering (log-sum-exp): subtract the log-domain peak so exp() stays in [~0, 1] and NEVER
    # overflows; the omitted factor exp(log_peak) is returned separately and re-applied by callers.
    log_peak = float(np.max(log_g))
    g = np.exp(log_g - log_peak)               # bounded kernel; <= 1 everywhere, no overflow
    dg = (sign * c0) * u * g  # d/dx of g (differentiate the exp's sign*z*u term, dz/dx = c0)
    # Composite Simpson weights: 1,4,2,4,...,4,1 over an even number of panels.
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    step = _QUAD_UPPER / n
    factor = step / 3.0
    value_norm = float(factor * np.dot(w, g))
    deriv_norm = float(factor * np.dot(w, dg))
    return log_peak, value_norm, deriv_norm


def fg_F(p: OUParams, r: float, x: float) -> float:
    """The Leung-Li increasing fundamental solution ``F(x)`` at discount ``r`` (Leung-Li 2015).

    ``F(x) = int_0^inf u^{r/theta - 1} exp( c0(x-m)u - u^2/2 ) du``, ``c0 = sqrt(2 theta)/sigma`` --
    the increasing solution of ``(1/2) sigma^2 F'' + theta (m - x) F' - r F = 0`` on the OU state.
    ``r > 0`` (a positive discount) is required so the ``u``-exponent ``r/theta - 1 > -1`` keeps the
    integral convergent at ``u = 0``. Value only (see ``fg_F_prime`` for the derivative). The
    magnitude is reconstructed overflow-safely from the peak-centered core (``exp(log_peak)*norm``);
    for an extreme state ``x`` the true value may legitimately be ``+inf`` (that is the honest
    magnitude, not a fabrication), but the boundary solve reads only the finite normalized ratio.
    """
    if not (r > 0.0):
        raise ValueError("Leung-Li discount r must be > 0 (else the F/G integral diverges at u=0)")
    log_peak, value_norm, _ = _fg_core(p.theta, p.sigma, p.m, r, x, sign=1.0)
    return float(math.exp(log_peak) * value_norm) if log_peak < 709.0 else math.inf


def fg_F_prime(p: OUParams, r: float, x: float) -> float:
    """Derivative ``F'(x)`` of the Leung-Li increasing solution (differentiated under the integral).

    ``F'(x) = int_0^inf c0 u * u^{r/theta-1} exp( c0(x-m)u - u^2/2 ) du`` (``d/dx`` of ``fg_F``);
    used by the smooth-fit optimality condition ``F(b*) = (b* - c) F'(b*)``. Reconstructed
    overflow-safely from the shared peak-centered core (``exp(log_peak)*deriv_norm``).
    """
    if not (r > 0.0):
        raise ValueError("Leung-Li discount r must be > 0 (else the F/G integral diverges at u=0)")
    log_peak, _, deriv_norm = _fg_core(p.theta, p.sigma, p.m, r, x, sign=1.0)
    return float(math.exp(log_peak) * deriv_norm) if log_peak < 709.0 else math.copysign(
        math.inf, deriv_norm)


def leung_li_boundary(p: OUParams, r: float, c: float = 0.0) -> float:
    """Solve the Leung-Li optimal-liquidation level ``b*`` from the smooth-fit optimality condition.

    Leung-Li 2015 ("Optimal Mean Reversion Trading"): the optimal single-liquidation problem
    ``sup_tau E[e^{-r tau}(X_tau - c)]`` under the OU law has value ``V(x) = A F(x)`` in the
    continuation region ``x < b*`` and ``V(x) = x - c`` at/above the sell boundary. Value-matching
    ``A F(b*) = b* - c`` and smooth-pasting ``A F'(b*) = 1`` collapse to the single OPTIMALITY
    CONDITION (dividing the two):
        ``h(b) := F(b) - (b - c) F'(b) = 0``.
    ``b*`` is the (unique, above ``m``) root of ``h``. ``h`` is continuous and, for the increasing
    convex ``F``, changes sign once above the mean, so a bracketed BISECTION converges (scipy-free).
    We bracket on ``[m + tiny, m + K sigma/sqrt(2 theta)]`` widened until ``h`` changes sign, then
    bisect to ``_ROOT_TOL``. ``c`` is the pre-registered net sale cost (default 0). Returns the
    frozen ``b*`` (a deterministic function of the frozen train ``(theta, m, sigma)``, r, c).

    The take-profit is naturally ABOVE the mean ``m`` (bank a reverting winner as it overshoots): at
    ``x`` far above ``m``, ``F``/``F'`` are large and ``h < 0`` (the ``(b-c)F'`` term dominates);
    just above ``m``, ``h > 0`` (``F`` dominates), so a sign change -- hence a root -- exists.
    """
    if not (r > 0.0):
        raise ValueError("Leung-Li discount r must be > 0")

    def h(b: float) -> float:
        # PEAK-NORMALIZED residual: h(b) = exp(log_peak) * (value_norm - (b-c) deriv_norm), and
        # exp(log_peak) > 0, so the SIGN (all bisection needs) is that of the bounded normalized
        # bracket -- overflow-safe for any (r, c, b) band (the naive exp(log_g) overflow is gone).
        _, value_norm, deriv_norm = _fg_core(p.theta, p.sigma, p.m, r, b, sign=1.0)
        return value_norm - (b - c) * deriv_norm

    # Bracket above the mean: h(m+) > 0; push the upper end out until h turns negative.
    scale = p.sigma / math.sqrt(2.0 * p.theta)  # the OU stationary-std order of magnitude
    lo = p.m + 1e-6 * max(1.0, abs(p.m), scale)
    h_lo = h(lo)
    # ECONOMIC no-boundary regime (defect fix -- refuse, never fabricate). The take-profit exists
    # ABOVE the mean only when h(m+) > 0 (F dominates near m; the (b-c)F' reward term dominates far
    # above, giving the sign change). For a HIGH discount r relative to theta (a large `power`) the
    # reward already dominates AT m, so h(m+) <= 0: the Leung-Li optimum is IMMEDIATE liquidation and
    # there is NO above-mean smooth-fit boundary. This is a valid, in-range (r, c) input that the OLD
    # code silently mis-resolved to the fabricated bracket-growth cap once the naive exp() overflowed
    # (the reported high-severity defect). It is NOT a stationarity failure (the OU is stationary) --
    # it is an out-of-band discount for this fit, so we RAISE a plain ValueError with an accurate
    # diagnosis rather than return a wrong frozen b*.
    if not (h_lo > 0.0):
        raise ValueError(
            "Leung-Li: no take-profit boundary above the mean for this discount -- h(m+) = "
            f"{h_lo:.4g} <= 0 means immediate-liquidation is optimal at r/theta = {r / p.theta:.4g} "
            f"(theta={p.theta:.4g}, m={p.m:.4g}, sigma={p.sigma:.4g}, r={r:.4g}, c={c:.4g}); "
            "the quadrature is overflow-safe, this is an economic out-of-band r, refused not faked")
    hi = p.m + max(scale, abs(p.m) + 1.0)
    h_hi = h(hi)
    grow = 0
    while h_lo * h_hi > 0.0 and grow < 80:
        hi += max(scale, 1.0)
        h_hi = h(hi)
        grow += 1
    if h_lo * h_hi > 0.0:
        # h stayed positive across a very wide bracket: no sign change locatable. Refuse rather than
        # return the fabricated bracket-growth cap (the old overflow artifact). A ValueError (not
        # OUStationarityError): the params are a stationary OU, the (r, c) band is what has no root.
        raise ValueError(
            "Leung-Li boundary not bracketed: no smooth-fit root above the mean over a wide bracket "
            f"(theta={p.theta:.4g}, m={p.m:.4g}, sigma={p.sigma:.4g}, r={r:.4g}, c={c:.4g}) -- "
            "refused rather than fabricated")
    # Bisection on the located sign-change bracket -- terminate on interval WIDTH (the peak-normalized
    # residual is not scale-invariant across b, so width is the robust stop; scipy-free root solve).
    for _ in range(_ROOT_MAXIT):
        mid = 0.5 * (lo + hi)
        if (hi - lo) < _ROOT_TOL:
            return float(mid)
        h_mid = h(mid)
        if h_mid == 0.0:
            return float(mid)
        if h_lo * h_mid <= 0.0:
            hi, h_hi = mid, h_mid
        else:
            lo, h_lo = mid, h_mid
    return float(0.5 * (lo + hi))


# ==================================================================================================
# The ExitModel candidate: fit OU on train, freeze b*, exit on the first causal crossing of b*
# ==================================================================================================
class OUExit:
    """OU / Leung-Li mean-reversion optimal EXIT (EDGE #6, SIGNAL §C.3(b)) -- an ``ExitModel``.

    ``fit(train)`` runs the exact OU MLE (with stationarity check) on the train fold's per-day
    series, then solves the Leung-Li ``b*`` ONCE and FREEZES it. ``exit_now`` voluntarily closes a
    still-alive winner on the first ``F_{t-1}``-measurable crossing of the frozen ``b*`` by the
    filtered running return -- while the -12% premium stop (``STOP_FLOOR``) remains the untouched
    loss floor (this model only banks WINNERS earlier; it never loosens the loss cap -- §C.3).
    ``k = 3`` (``theta, m, sigma``; ``r``/``c`` pre-registered, not fit, so not counted -- §C.4).

    Structurally conforms to ``base.ExitModel`` (``fit`` + ``exit_now`` + ``k``); the harness uses
    it via ``isinstance(obj, ExitModel)`` without importing this module (SYSTEM §C.3 interface).
    """

    def __init__(self, r: float = 0.05, c: float = 0.0, dt: float = 1.0) -> None:
        """Pre-registered (NOT data-fit) constants: discount ``r`` (>0), cost ``c``, spacing ``dt``.

        ``r`` and ``c`` are frozen error-control / no-snoop choices (SIGNAL §C.3(b): "the discount
        rate ... frozen"), NOT tuned on data, so they do not enter the parsimony count ``k = 3``.
        ``dt`` is the sampling spacing the OU MLE assumes (default 1.0 = one lake session day).
        """
        if not (r > 0.0):
            raise ValueError("OUExit discount r must be > 0")
        if not (dt > 0.0):
            raise ValueError("OUExit dt must be > 0")
        self.k = 3  # theta, m, sigma fit on train; r, c pre-registered (not counted)
        self.r = float(r)
        self.c = float(c)
        self.dt = float(dt)
        self.params: OUParams | None = None  # set on fit, then frozen
        self.b_star: float | None = None     # the frozen Leung-Li boundary

    def fit(self, train) -> "OUExit":
        """Fit the OU by MLE on ``train.series`` (past warm-up), solve+freeze ``b*``; return self.

        The parameters and boundary are estimated on the TRAIN fold ONLY and FROZEN for OOS (no
        re-fit / re-tune on TEST -- SYSTEM §C.1). A non-stationary train fold RAISES
        ``OUStationarityError`` (the stationarity check), never silently yields a mis-specified fit.
        """
        body = train.series[train.warmup:]
        self.params = ou_mle(body, dt=self.dt)
        self.b_star = leung_li_boundary(self.params, r=self.r, c=self.c)
        return self

    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        """Causal hold/exit at ``cursor.t``: close the winner on first crossing of frozen ``b*``.

        Reads ``held_return`` (the running position return known up to the prior bar -- the
        ExitModel filtered-state covariate; ``base.ExitModel``) and ``cursor.past()`` ONLY, so it is
        ``F_{t-1}``-measurable (DAILY cadence). A same-bar / future read would route through
        ``cursor.read`` and RAISE ``PurityError`` (the pinning hook). ``True`` iff the filtered
        running return reached the mean-reversion take-profit ``held_return >= b*``. Never loosens
        the -12% stop: this only banks a still-alive winner earlier (SIGNAL §C.3).
        """
        if self.b_star is None:
            raise RuntimeError("exit_now before fit -- OU boundary not frozen yet")
        # Touch the causal window through the guard so a leak (a >= t read) breaks the pinning test:
        # a DAILY decision executing on cursor.t may read strictly-past bars only. We read the last
        # available past bar (t-1) when present; past() itself never exposes bar t (ScanCursor).
        if cursor.t > 0:
            cursor.read(cursor.t - 1)  # F_{t-1} read via the guard; RAISES on any misuse >= t
        return bool(held_return >= self.b_star)


# The DAILY cadence is this candidate's measurability standard (an end-of-day hold/exit decision;
# SYSTEM §C.2) -- surfaced so the pinning test and any harness wiring name it explicitly.
CADENCE = Cadence.DAILY
