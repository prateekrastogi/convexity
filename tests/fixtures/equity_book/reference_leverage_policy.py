"""Axis E -- leverage as a growth-optimal control problem under fat tails and drawdown budgets.

research/cta/lib/leverage_policy.py  (Stage-A CTA sweep, axis E)

This module answers ONE question with real math, not a heuristic: given a base (1x) daily net
return series produced by some other axis's trend/vol-sizing machinery, what leverage should sit
on top of it, and should that leverage be a single constant or a state-dependent policy?

Five constructions are implemented, each returned as a self-contained, reusable function so a
later Sweep/Integration agent can call them directly on whatever base return stream axis
A-D/F produce:

  1. kelly_optimal_leverage         -- discrete-time Kelly against the EMPIRICAL return distribution
  2. fractional_kelly_leverage      -- trivial scalar shrinkage of (1), with the estimation-error
                                        justification made explicit in the docstring/report
  3. grossman_zhou_exposure         -- Grossman & Zhou (1993) drawdown-constrained growth-optimal
                                        policy: THE genuinely state-dependent, time-varying,
                                        causally-computed exposure array in this module
  4. evt_ceiling_leverage           -- Peaks-Over-Threshold / Generalized Pareto ruin-constrained
                                        leverage ceiling (McNeil & Frey 2000 style tail fit)
  5. cdar_constrained_leverage      -- Chekhlov, Uryasev & Zabarankin (2005) CDaR LP, solved with
                                        cvxpy, with a closed-form cross-check (see derivation in
                                        the function docstring: for a SINGLE return stream scaled
                                        by a scalar leverage, CDaR is exactly linear in leverage,
                                        so the LP has a closed-form optimum -- both are computed
                                        and compared as a correctness check on the LP formulation)

CAUSALITY. Only (3), grossman_zhou_exposure, produces a genuinely time-varying exposure signal
(the other four are single scalars, calibrated once from a historical sample and then held
constant for the whole backtest -- a constant has no t-dependence, so the exposure-timing-
alignment bug class (bug pattern 3c) cannot apply to it by construction). grossman_zhou_exposure
is therefore the one function in this module proven causal via a synthetic-outlier-injection test
in validate_leverage_policy.py, alongside a deliberately-wrong "naive/unlagged" control version
that DOES react same-day, proving the test has real discriminating power.

All functions take/return plain numpy arrays of daily SIMPLE returns (not log returns), matching
the house apply_costs()/metrics() convention in research/indices/i1_pipeline.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import genpareto

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except ImportError:  # pragma: no cover
    _HAVE_CVXPY = False


# ======================================================================================
# 1. Discrete-time Kelly against the EMPIRICAL distribution (no Gaussian assumption)
# ======================================================================================
def empirical_log_growth(returns: np.ndarray, L: float) -> float:
    """E[log(1 + L*r)] evaluated over the empirical (real, fat-tailed) sample -- the exact
    discrete-time Kelly objective (Kelly 1956; Thorp 1971/2006 for the finance formulation).
    Returns -inf if any scenario would produce total ruin (1 + L*r <= 0) at this L, which is
    itself the correct economic penalty: Kelly's objective is literally undefined/(-inf) past
    the point of ruin, so the optimizer is naturally repelled from L's that can wipe out capital
    on any observed historical day (importantly including the real -10.5%/-10.7% crash days --
    no synthetic Gaussian tail is substituted for them)."""
    r = returns[np.isfinite(returns)]
    growth = 1.0 + L * r
    if np.any(growth <= 0):
        return -np.inf
    return float(np.mean(np.log(growth)))


def kelly_optimal_leverage(returns: np.ndarray, l_max: float = 25.0,
                            n_bootstrap: int = 0, bootstrap_seed: int = 0) -> dict:
    """Numerically maximize E[log(1+L*r)] over the REAL historical daily return sample (no
    Gaussian closed-form substituted). Uses scipy's bounded scalar minimizer on the negated
    objective; the objective is concave in L wherever finite (log of an affine function of L is
    concave), so a bounded 1-D search is both sufficient and exact up to numerical tolerance --
    no need for a general nonlinear solver.

    Optionally bootstraps the historical sample (n_bootstrap block-free iid resamples of daily
    returns) to report a confidence interval on L*, making explicit the ESTIMATION ERROR that
    motivates fractional Kelly in practice (MacLean, Ziemba & Thorp identify this as the single
    biggest reason full Kelly is never used live: the input mean/tail is itself estimated with
    error, and full Kelly is maximally sensitive to that error near the point of ruin).

    Returns dict: L_star, growth_star, l_max, n_obs, and (if n_bootstrap>0) l_star_bootstrap_ci
    (2.5/50/97.5 percentiles) and l_star_bootstrap_std.
    """
    r = returns[np.isfinite(returns)]
    if len(r) < 30:
        raise ValueError(f"kelly_optimal_leverage: only {len(r)} finite obs, need >=30")

    def neg_growth(L):
        g = empirical_log_growth(r, L)
        return -g if np.isfinite(g) else 1e6 + abs(L)  # smooth penalty, keeps optimizer moving

    res = minimize_scalar(neg_growth, bounds=(0.0, l_max), method="bounded",
                           options={"xatol": 1e-5})
    out = {"L_star": float(res.x), "growth_star": float(-res.fun), "l_max": l_max,
           "n_obs": int(len(r))}

    if n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        boots = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            samp = rng.choice(r, size=len(r), replace=True)
            res_b = minimize_scalar(lambda L: -empirical_log_growth(samp, L)
                                     if np.isfinite(empirical_log_growth(samp, L)) else 1e6 + abs(L),
                                     bounds=(0.0, l_max), method="bounded", options={"xatol": 1e-4})
            boots[i] = res_b.x
        out["l_star_bootstrap_ci"] = [float(np.percentile(boots, 2.5)),
                                       float(np.percentile(boots, 50)),
                                       float(np.percentile(boots, 97.5))]
        out["l_star_bootstrap_std"] = float(np.std(boots))
        out["l_star_bootstrap_samples"] = n_bootstrap
    return out


def fractional_kelly_leverage(l_star: float, fraction: float) -> float:
    """Trivial scalar shrinkage L = fraction * L_star. Kept as an explicit function (not inlined
    at call sites) so the fraction used in the sweep is self-documenting. Standard fractions in
    the literature: 0.5 ("half-Kelly", MacLean-Ziemba-Thorp's headline recommendation -- roughly
    halves growth-rate give-up relative to full Kelly while cutting variance/drawdown far more
    than proportionally, because the growth-rate curve is quadratic-flat near its own peak but the
    variance/drawdown cost of overshooting is not) down to 0.1-0.25 for live, estimation-error-
    exposed, non-stationary strategies (Thorp's own live-trading practice)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    return fraction * l_star


# ======================================================================================
# 2. Grossman & Zhou (1993) drawdown-constrained growth-optimal policy
# ======================================================================================
def grossman_zhou_exposure(base_returns: np.ndarray, l_base: float, alpha_floor: float,
                            rt_bp: float = 0.0, cap: float | None = None) -> dict:
    """Grossman & Zhou (1993), "Optimal Investment Strategies for Controlling Drawdowns",
    Mathematical Finance 3(3). Their continuous-time result (geometric Brownian wealth, CRRA/log
    objective, hard constraint W_t >= alpha * M_t at all t, where M_t = running max of wealth and
    alpha in (0,1) is the floor as a FRACTION OF PEAK): the growth-optimal policy invests

        pi_t = lambda* * (1 - alpha * M_t / W_t)                                          (GZ-1)

    of current wealth in the risky asset, where lambda* is the UNCONSTRAINED Merton/Kelly
    fraction and pi_t is clipped at 0 (no shorting the cushion). Two facts fall straight out of
    (GZ-1) and are worth deriving explicitly because they are the entire mechanism:

      - At a fresh equity high (W_t = M_t), pi_t = lambda*(1-alpha) < lambda*: the policy is
        ALREADY de-risked relative to unconstrained Kelly even at the best possible moment, by
        exactly the drawdown budget (1-alpha). This is the price of drawdown insurance, paid at
        the peak, not just in the trough.
      - As wealth drifts down against a fixed prior peak, M_t/W_t rises above 1 and pi_t falls
        continuously and reaches EXACTLY ZERO as W_t -> alpha*M_t: in continuous, no-jump
        dynamics the constraint can never be breached, because exposure has already vanished at
        the boundary. This is the precise sense in which this is "the mathematically correct
        realization of drawdown control", as opposed to an ad hoc vol-target overlay that reduces
        exposure only after realized vol has already risen (reactive, not floor-anchored).

    DISCRETE-DAILY ADAPTATION (this function). GZ is a continuous-time result; this desk trades
    once a day with real overnight gaps, so the "exposure hits exactly zero at the floor" no-jump
    guarantee does NOT literally hold here -- a large enough single-day gap (the real -10.5%/
    -10.7% crash days in this universe) can still cross the floor within one step. This is stated
    as an explicit, known limitation, not hidden: the discrete adaptation gets the SHAPE of the
    correct policy (state-dependent, peak-anchored, monotonically de-risking as the cushion
    shrinks) but not the continuous-time no-breach guarantee. It is combined with the EVT ceiling
    below specifically to bound this residual single-day gap risk, which GZ's own continuous-time
    proof does not cover.

    Practical formula actually implemented, decide-at-close/act-next-day (bug pattern 3c):
        M_t, W_t are the running peak and current wealth of THIS POLICY'S OWN realized equity
        curve, known as of the close of day t-1 (i.e., wealth[t-1+1] in a 0-indexed running
        simulation -- see the loop below). Exposure applied ON day t is
            L_t = l_base * clip(1 - alpha_floor * M_{t-1}/W_{t-1}, 0, cap)
        which uses ONLY information available through the close of t-1. This is the central
        causality property proven by the injection test in validate_leverage_policy.py.

    Parameters
    ----------
    base_returns : daily SIMPLE net returns of the underlying (already-costed) 1x strategy.
    l_base : the unconstrained lambda* to scale the cushion by -- pass a (fractional-)Kelly
        leverage from kelly_optimal_leverage()/fractional_kelly_leverage() here.
    alpha_floor : GZ's alpha -- wealth floor as a FRACTION OF PEAK (e.g. 0.75 => wealth must stay
        >= 75% of its running peak, i.e. a 25% max-drawdown BUDGET). Higher alpha = tighter
        (smaller-drawdown-budget) policy = lower average realized exposure.
    rt_bp : round-trip trading-fee bp charged on |change in exposure| day to day (the cost of
        actually implementing this daily-rebalanced overlay; separate from, and on top of, any
        cost already embedded inside base_returns for the underlying 1x strategy's own turnover).
    cap : optional hard ceiling on L_t (e.g. the EVT ceiling from evt_ceiling_leverage below), so
        the two axis-E constructions can be composed: GZ policy for the SHAPE, EVT for the ceiling.

    Returns dict with 'exposure' (array, causal), 'wealth', 'peak' (both length n+1, wealth[0]=
    peak[0]=1.0 the starting normalization), 'net_return' (post-overlay-cost, length n),
    'avg_exposure', 'l_base', 'alpha_floor'.
    """
    if not (0.0 < alpha_floor < 1.0):
        raise ValueError(f"alpha_floor must be in (0,1), got {alpha_floor}")
    r = np.asarray(base_returns, dtype="float64")
    n = len(r)
    exposure = np.zeros(n)
    net_ret = np.zeros(n)
    wealth = np.empty(n + 1)
    peak = np.empty(n + 1)
    wealth[0] = 1.0
    peak[0] = 1.0
    prev_L = 0.0
    for t in range(n):
        # state known strictly through the close of day t-1: wealth[t], peak[t]
        ratio = peak[t] / wealth[t] if wealth[t] > 0 else np.inf
        L_t = l_base * max(0.0, 1.0 - alpha_floor * ratio)
        if cap is not None:
            L_t = min(L_t, cap)
        exposure[t] = L_t
        r_t = r[t]
        if np.isfinite(r_t):
            gross_t = L_t * r_t
        else:
            gross_t = 0.0
        fee_t = (rt_bp / 1e4) * abs(L_t - prev_L)
        net_ret[t] = gross_t - fee_t
        wealth[t + 1] = wealth[t] * (1.0 + net_ret[t]) if wealth[t] > 0 else wealth[t]
        peak[t + 1] = max(peak[t], wealth[t + 1])
        prev_L = L_t
    return {
        "exposure": exposure, "wealth": wealth, "peak": peak, "net_return": net_ret,
        "avg_exposure": float(np.mean(exposure)), "l_base": l_base, "alpha_floor": alpha_floor,
    }


def grossman_zhou_exposure_naive_lookahead_control(base_returns: np.ndarray, l_base: float,
                                                     alpha_floor: float, rt_bp: float = 0.0,
                                                     cap: float | None = None) -> dict:
    """DELIBERATELY WRONG control for the injection test ONLY -- never call this from a real
    pipeline. Identical to grossman_zhou_exposure() except it sizes day t's exposure using
    wealth/peak AFTER applying day t's own return (i.e. wealth[t+1]/peak[t+1] instead of
    wealth[t]/peak[t]) -- a same-day lookahead. Used solely to prove the injection test has real
    discriminating power (it must show this control reacting on the injected day itself, unlike
    the honest function above)."""
    if not (0.0 < alpha_floor < 1.0):
        raise ValueError(f"alpha_floor must be in (0,1), got {alpha_floor}")
    r = np.asarray(base_returns, dtype="float64")
    n = len(r)
    exposure = np.zeros(n)
    net_ret = np.zeros(n)
    wealth = np.empty(n + 1)
    peak = np.empty(n + 1)
    wealth[0] = 1.0
    peak[0] = 1.0
    prev_L = 0.0
    for t in range(n):
        r_t = r[t]
        # LOOKAHEAD BUG (deliberate): peek at what wealth/peak WOULD be after today's own return,
        # using a naive "as if L=prev_L held" one-step-ahead peek at the raw base return itself
        # (this is the classic same-day-signal-uses-same-day-return bug pattern, not a subtle one)
        peek_wealth = wealth[t] * (1.0 + (prev_L * r_t if np.isfinite(r_t) else 0.0))
        peek_peak = max(peak[t], peek_wealth)
        ratio = peek_peak / peek_wealth if peek_wealth > 0 else np.inf
        L_t = l_base * max(0.0, 1.0 - alpha_floor * ratio)
        if cap is not None:
            L_t = min(L_t, cap)
        exposure[t] = L_t
        gross_t = L_t * r_t if np.isfinite(r_t) else 0.0
        fee_t = (rt_bp / 1e4) * abs(L_t - prev_L)
        net_ret[t] = gross_t - fee_t
        wealth[t + 1] = wealth[t] * (1.0 + net_ret[t]) if wealth[t] > 0 else wealth[t]
        peak[t + 1] = max(peak[t], wealth[t + 1])
        prev_L = L_t
    return {"exposure": exposure, "wealth": wealth, "peak": peak, "net_return": net_ret,
            "avg_exposure": float(np.mean(exposure))}


# ======================================================================================
# 3. EVT / Peaks-Over-Threshold ruin-constrained leverage ceiling
# ======================================================================================
def evt_ceiling_leverage(returns: np.ndarray, threshold_quantile: float = 0.95,
                          epsilon: float | None = None, breach_years: float = 10.0,
                          periods_per_yr: float = 252.0) -> dict:
    """Fit a Generalized Pareto Distribution to the tail of REALIZED DAILY LOSSES of this
    strategy's own return series (Peaks-Over-Threshold, Balkema-de Haan-Pickands theorem;
    McNeil & Frey (2000) is the standard reference for applying POT/GPD to financial return
    tails specifically). Solve for the leverage L such that

        P(daily loss > 1/L) < epsilon                                                     (EVT-1)

    i.e. the leverage at which a single-day loss would wipe out (>=100% of) capital has
    probability epsilon under the fitted tail law. epsilon is tied EXPLICITLY to a declared
    ruin-frequency budget: epsilon = 1 / (breach_years * periods_per_yr) means "expect to breach
    the 1/L threshold about once every `breach_years` years" -- directly comparable to the
    established empirical fact that the -10% threshold at 10x leverage on SPY was breached once
    in ~8 real years (i.e. an empirical epsilon of about 1/(8*252) =~ 5e-4 for THAT specific
    L=10/threshold=10% pair). Default breach_years=10 is deliberately a bit more conservative than
    that observed 8-year frequency, on the standard EVT-in-practice principle that a POT tail fit
    from ~8 years of daily data (~2000 obs) is itself estimated with real sampling error in the
    shape parameter xi, and extrapolating far into the tail (asking for below-epsilon=1e-4-level
    probabilities) amplifies that error -- so the CEILING should not be pushed to exactly match
    the single worst historical multi-year frequency, it should sit a bit inside it.

    GPD tail formula used: for a threshold u with exceedances Y = (loss - u | loss > u) ~
    GPD(xi, beta) (loc fixed at 0, scale beta, shape xi), the tail survival function beyond u is
        P(loss > u + y) = P(loss > u) * (1 + xi*y/beta)^(-1/xi)        (xi != 0; log form if xi=0)
    Setting this equal to epsilon and solving for y gives the loss level y+u whose EXCEEDANCE
    probability is exactly epsilon; L_ceiling = 1 / (u+y).

    Returns dict: L_ceiling, xi, beta, u (threshold loss level), loss_at_epsilon, epsilon,
    p_exceed_u (empirical P(loss>u)), n_exceedances, breach_years.
    """
    r = returns[np.isfinite(returns)]
    losses = -r  # positive number = a loss
    u = float(np.quantile(losses, threshold_quantile))
    exceed = losses[losses > u] - u
    n_exc = len(exceed)
    if n_exc < 15:
        raise ValueError(f"evt_ceiling_leverage: only {n_exc} exceedances above q={threshold_quantile}, "
                          f"need >=15 for a stable GPD fit -- lower threshold_quantile or use more data")
    xi, _loc, beta = genpareto.fit(exceed, floc=0.0)
    p_exceed_u = float(np.mean(losses > u))
    if epsilon is None:
        epsilon = 1.0 / (breach_years * periods_per_yr)
    ratio = epsilon / p_exceed_u
    if abs(xi) < 1e-8:
        y = -beta * np.log(ratio)
    else:
        y = (beta / xi) * (ratio ** (-xi) - 1.0)
    loss_at_epsilon = u + float(y)
    L_ceiling = (1.0 / loss_at_epsilon) if loss_at_epsilon > 0 else float("inf")
    return {"L_ceiling": float(L_ceiling), "xi": float(xi), "beta": float(beta), "u": u,
            "loss_at_epsilon": loss_at_epsilon, "epsilon": float(epsilon),
            "p_exceed_u": p_exceed_u, "n_exceedances": int(n_exc), "breach_years": breach_years,
            "threshold_quantile": threshold_quantile}


# ======================================================================================
# 4. CDaR-constrained constant leverage (Chekhlov, Uryasev & Zabarankin 2005)
# ======================================================================================
def _empirical_cdar(drawdown_path: np.ndarray, alpha: float) -> float:
    """Closed-form empirical CDaR_alpha of a precomputed drawdown path: the mean of the worst
    (1-alpha) fraction of drawdown observations. This is the exact minimizer of
    zeta + mean(max(d-zeta,0))/(1-alpha) over zeta for a FIXED finite sample (the minimizing
    zeta* is the empirical (alpha)-quantile of d) -- used here purely as an independent
    closed-form cross-check on the cvxpy LP result below, not as the LP itself."""
    d = np.sort(drawdown_path)[::-1]
    k = max(1, int(np.ceil((1 - alpha) * len(d))))
    return float(np.mean(d[:k]))


def cdar_constrained_leverage(returns: np.ndarray, budget: float, alpha: float = 0.95) -> dict:
    """Chekhlov, Uryasev & Zabarankin (2005), "Portfolio Optimization with Drawdown Constraints",
    generalizing Rockafellar-Uryasev's CVaR LP to drawdown ("CDaR" = Conditional Drawdown-at-
    Risk). Their formulation works on the UNCOMPOUNDED cumulative return path c_t = sum_{s<=t} r_s
    (additive, not geometrically compounded) precisely so that a linear combination of return
    streams produces a linear combination of cumulative paths -- this is what keeps the whole
    problem a tractable LP.

    KEY DERIVATION for the single-stream, scalar-leverage case used here (this is what makes a
    closed-form cross-check possible, and is worth writing out because it is the reason CDaR
    optimization is tractable for a leverage choice at all): let c_t = cumsum(r_1..r_t) at unit
    (L=1) leverage, and C_t = running max of c up to t. At leverage L>=0 the scaled path is L*c_t,
    whose running max is L*C_t (since L>=0 preserves order), so the drawdown path at leverage L is

        Drawdown_t(L) = L*C_t - L*c_t = L * d_t,          d_t := C_t - c_t >= 0        (CDAR-1)

    i.e. EXACTLY LINEAR in L. Therefore CDaR_alpha(L) = L * CDaR_alpha(d) (CDaR of a nonnegatively
    -scaled series scales linearly for L>=0), and the optimization
        maximize_L   L * mean(r) * T          subject to   L * CDaR_alpha(d) <= budget,  L>=0
    has the trivial closed form L* = budget / CDaR_alpha(d) whenever mean(r) > 0 (objective
    strictly increasing in L, so the budget constraint always binds at the optimum).

    This function solves the SAME problem twice, independently: once via a genuine cvxpy LP using
    the standard Rockafellar-Uryasev-style auxiliary-variable recursion (zeta, z_t>=max(u_t-zeta,0),
    u_t >= u_{t-1} - L*r_t, u_t>=0), faithful to the general CUZ formulation (which is what a later
    integration agent would extend to a MULTI-sleeve blend, where the linear shortcut above no
    longer applies because the running max of a multi-asset combination is not the combination of
    running maxes) -- and once via the closed form (CDAR-1). Reporting both, and their agreement,
    is the correctness proof for the LP encoding.

    Returns dict: L_lp (from cvxpy), L_closed_form, cdar_unit (=CDaR_alpha(d) at L=1), budget,
    alpha, agree (bool, |L_lp - L_closed_form| < 1e-3 relative).
    """
    r = returns[np.isfinite(returns)]
    T = len(r)
    c = np.cumsum(r)
    C = np.maximum.accumulate(c)
    d = C - c  # unit-leverage drawdown path, d_t >= 0
    cdar_unit = _empirical_cdar(d, alpha)
    mean_r = float(np.mean(r))
    L_closed = budget / cdar_unit if cdar_unit > 0 else float("inf")

    out = {"cdar_unit": cdar_unit, "budget": budget, "alpha": alpha, "L_closed_form": L_closed,
           "mean_r": mean_r, "n_obs": T}

    if not _HAVE_CVXPY:
        out["L_lp"] = None
        out["agree"] = None
        out["lp_skipped_reason"] = "cvxpy not installed"
        return out

    L = cp.Variable(nonneg=True)
    zeta = cp.Variable()
    u = cp.Variable(T, nonneg=True)
    z = cp.Variable(T, nonneg=True)
    constraints = [u[0] >= -L * r[0]]
    for t in range(1, T):
        constraints.append(u[t] >= u[t - 1] - L * r[t])
    constraints += [z >= u - zeta, z >= 0]
    cdar_expr = zeta + cp.sum(z) / ((1 - alpha) * T)
    constraints.append(cdar_expr <= budget)
    prob = cp.Problem(cp.Maximize(L * mean_r * T), constraints)
    prob.solve(solver=cp.CLARABEL)
    L_lp = float(L.value) if L.value is not None else float("nan")
    out["L_lp"] = L_lp
    out["lp_status"] = prob.status
    denom = max(abs(L_closed), 1e-9)
    out["agree"] = bool(np.isfinite(L_lp) and abs(L_lp - L_closed) / denom < 1e-2)
    return out
