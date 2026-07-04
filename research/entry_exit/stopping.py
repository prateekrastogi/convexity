"""MODELS.stopping -- Phase-C EXIT candidate #5: optimal-stopping free-boundary under a momentum /
regime-switching latent (SIGNAL §C.3(a), SYSTEM §C.9 #5, docs/EDGE.md pre-registration item #5).

THE MODEL (the formal problem, its assumptions, estimator, error control -- a free-boundary optimal-
stopping problem, NOT a technical indicator):

  * *Assumption.* A live position's log-value follows a controlled diffusion whose one-step drift
    depends on a LATENT 2-state momentum/regime chain ``S_t in {HIGH, LOW}`` (high-drift "trend
    continues" vs low/negative-drift "trend fading"). The state is HIDDEN; only a noisy per-day
    return ``r_t = mu_{S_t} + sigma * eps_t`` (``eps_t ~ N(0,1)``) is observed. The regime evolves as
    a Markov chain with transition matrix ``P`` (``P[i][j] = Pr(S_t=j | S_{t-1}=i)``). This is EDGE
    #5's "controlled diffusion whose drift depends on a latent momentum/regime state, coupling to the
    §C.2c HMM posterior".

  * *The optimal-stopping problem.* Holding one more day earns the risk-adjusted expected drift and
    discounts the future at impatience ``rho``; stopping banks the intrinsic value now. The holder
    solves ``sup_tau E[ e^{-rho*tau} G(V_tau) ]`` -- hold while the CONTINUATION value exceeds the
    intrinsic EXIT value, stop at the FREE BOUNDARY. Under partial observation the sufficient
    statistic is the FILTERED belief ``pi_t = P(S_t = HIGH | F_t)`` (the Wonham/Shiryaev reduction:
    a partially-observed regime-switching stopping problem is an EQUIVALENT fully-observed problem on
    the belief). The value function ``W(pi)`` on the belief simplex ``pi in [0,1]`` solves the
    discounted Bellman optimality equation

        W(pi) = max( 0 ,  hold(pi) ),   hold(pi) = E[ mu_bar(pi) + e^{-rho} * W(pi') | pi ]  (*)

    where ``mu_bar(pi) = pi*mu_HIGH + (1-pi)*mu_LOW`` is the expected one-step drift the holder
    collects and ``pi'`` is the Bayes-updated belief after the next observation (the CONTINUATION
    reward net of the do-nothing/stop payoff 0). The optimal policy is a THRESHOLD: there is a single
    free boundary ``pi*`` such that it is optimal to CONTINUE HOLDING iff ``pi >= pi*`` (belief in the
    high-drift regime is strong enough) and to EXIT iff ``pi < pi*``. This threshold structure is the
    "hold winners while trending" thesis of EDGE #5 (contrast EDGE #6 OU/Leung-Li, which banks a
    winner at a *mean-reversion* target -- the deliberately opposing exit thesis).

  * *Estimator + error control.* ``fit`` estimates the latent dynamics ``(mu_HIGH, mu_LOW, sigma, P)``
    and the discount ``rho`` on the TRAIN fold ONLY, then FREEZES them; the free boundary ``pi*`` is
    solved ONCE from those frozen parameters (value iteration on a belief grid, a contraction with
    modulus ``e^{-rho} < 1`` -- convergence to a fixed tolerance is the error control) and frozen for
    OOS. NO parameter or threshold is re-fit or re-tuned on TEST (SYSTEM §C.1). ``k = 6`` free
    parameters (``mu_HIGH, mu_LOW, sigma, rho`` + the 2 free entries of the 2x2 stochastic ``P``)
    is the parsimony count the SYSTEM §C.4 information-criterion penalty is charged against.

PURITY (BINDING -- SIGNAL §C.1, SYSTEM §C.2 -- the load-bearing clause):
  * The exit decision at execution bar ``t`` reads ONLY the FILTERED / one-step-ahead belief
    ``pi_{t-1} = P(S_{t-1} = HIGH | F_{t-1})`` built from returns strictly BEFORE bar ``t``
    (``cursor.past()`` = ``series[:t]``). It is ``F_{t-1}``-measurable by construction. A SMOOTHED /
    full-sample belief ``P(S_t | F_T)``, or a same-bar ``pi_t`` conditioning on bar ``t`` itself, is a
    LOOKAHEAD and is inadmissible -- a no-lookahead PINNING test asserts a wired same-bar read RAISES
    ``PurityError`` via ``ScanCursor.read`` / ``require_causal``.
  * The Hamilton filter recursion here is strictly forward (predict then Bayes-update on each past
    observation), never a backward Kim smoother pass -- the smoothed posterior is BARRED (SYSTEM
    §C.9 #3, inherited by the §C.3(a) latent).

THE -12% STOP IS THE RETAINED FLOOR (SIGNAL §C.3, ``STOP_FLOOR``): this ExitModel decides ONLY when a
still-alive WINNER is voluntarily closed early; the true 1-minute premium stop remains the loss cap
and this rule NEVER loosens it (a conformance test asserts the model never returns ``False`` -- "keep
holding" -- once ``held_return <= -STOP_FLOOR``; the floor's own enforcement stays in ``signal/orb.py``
where the intraday stop is executed, this model just may not override it upward).

DATA GATE (SYSTEM §C.8): this module is a SYNTHETIC-testable candidate model ONLY. It touches no lake,
emits no metric, adopts nothing. The empirical selection RUN is BLOCKED-ON-LAKE (the wide-band /
position-following lake is ABSENT); the harness owns the ``BlockedOnLakeError`` guard. Nothing here
runs a backtest.

Imports ONLY ``intraday.models.base`` (the shared interface + causal helpers) and ``intraday.core``
constants via it -- NEVER a sibling Phase-C model module (SYSTEM §C.3 off-ledger / interface-driven-
harness rule). Pure numpy + stdlib ``math`` (scipy is NOT a dependency, matching ``system/__init__.py``).
Moneyness, not greeks (CLAUDE.md). Map: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import (
    STOP_FLOOR,
    Cadence,
    ScanCursor,
    TrainFold,
)

# NOTE: this module structurally conforms to ``models.base.ExitModel`` (a runtime_checkable Protocol)
# WITHOUT subclassing or importing it -- ``fit -> self``, ``exit_now(cursor, held_days, held_return)``,
# and ``k`` present. The conformance is asserted by ``isinstance(model, ExitModel)`` in the tests.

__all__ = [
    "STOP_FLOOR",
    "RegimeLatent",
    "OptimalStoppingExit",
    "analytic_single_regime_hold",
]

# Value-iteration controls for the belief-grid free-boundary solve. The Bellman map (*) is a
# contraction with modulus e^{-rho} < 1, so value iteration converges geometrically; these bound the
# fixed-point error (train-time only -- the boundary is solved ONCE and frozen).
_VI_GRID = 201          # belief-grid resolution on [0, 1] (pi in {0, 1/200, ..., 1})
_VI_TOL = 1e-9          # sup-norm convergence tolerance of the value iteration
_VI_MAX_ITERS = 20000   # hard cap (a diverging/degenerate fit stops rather than spinning forever)
_QUAD_NODES = 41        # Gauss-Hermite-lite: fixed standard-normal nodes for the E[.|pi] expectation


@dataclass(frozen=True)
class RegimeLatent:
    """The frozen 2-state momentum/regime latent dynamics fit on TRAIN (SIGNAL §C.3(a), SYSTEM §C.9 #5).

    A partially-observed 2-state Markov chain ``S_t in {HIGH=1, LOW=0}`` driving a Gaussian emission
    ``r_t = mu_{S_t} + sigma * eps_t``. Estimated on the train fold and FROZEN; the free boundary
    ``pi*`` is a deterministic function of THESE parameters + the discount ``rho`` (solved once,
    frozen). All fields are ``F_{t-1}``-measurable frozen constants at OOS decision time.

      * ``mu_high`` -- the HIGH-drift regime's per-day mean return (trend continues; typically > 0).
      * ``mu_low``  -- the LOW-drift regime's per-day mean return (trend fading; typically <= 0).
      * ``sigma``   -- the (regime-common) emission volatility (> 0).
      * ``p_hh``    -- ``Pr(S_t = HIGH | S_{t-1} = HIGH)`` (regime persistence of the high state).
      * ``p_ll``    -- ``Pr(S_t = LOW  | S_{t-1} = LOW)``  (regime persistence of the low state).

    The 2x2 transition matrix is ``[[p_ll, 1-p_ll], [1-p_hh, p_hh]]`` in ``(LOW, HIGH)`` index order.
    Two free transition entries (``p_hh, p_ll``) + ``(mu_high, mu_low, sigma, rho)`` = the ``k = 6``
    parsimony count (``rho`` lives on the model, not this dataclass, but is counted with these).
    """

    mu_high: float
    mu_low: float
    sigma: float
    p_hh: float
    p_ll: float

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError("RegimeLatent.sigma must be strictly positive (an emission volatility)")
        if not (0.0 <= self.p_hh <= 1.0 and 0.0 <= self.p_ll <= 1.0):
            raise ValueError("RegimeLatent.p_hh and .p_ll must be probabilities in [0, 1]")

    def emission_logpdf(self, r: float, high: bool) -> float:
        """Log Gaussian emission density ``log N(r | mu_{state}, sigma^2)`` for the given regime."""
        mu = self.mu_high if high else self.mu_low
        z = (r - mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - 0.5 * math.log(2.0 * math.pi)

    def predict(self, pi: float) -> float:
        """One-step Chapman-Kolmogorov PREDICT of the HIGH-state prior from a posterior ``pi``.

        ``pi_pred = Pr(S_t = HIGH | F_{t-1}) = pi*p_hh + (1-pi)*(1-p_ll)`` -- push the belief through
        the transition matrix ONE step BEFORE seeing the next observation. This is the "predict" half
        of the Hamilton filter; it uses no bar-``t`` information and is the object the free-boundary
        value iteration and the causal exit decision both consume.
        """
        return pi * self.p_hh + (1.0 - pi) * (1.0 - self.p_ll)

    def bayes_update(self, pi_pred: float, r: float) -> float:
        """The Hamilton-filter BAYES UPDATE of the HIGH-state belief on a new observation ``r``.

        Given the predicted prior ``pi_pred = Pr(S_t=HIGH | F_{t-1})`` and the newly observed return
        ``r``, returns the FILTERED posterior ``Pr(S_t=HIGH | F_t)`` by Bayes' rule with the Gaussian
        emissions. Strictly forward (predict->update); NEVER a backward smoother pass (smoothed is a
        lookahead, BARRED -- SYSTEM §C.9 #3). Numerically stable via the log-sum-exp of the two
        regime likelihoods.
        """
        lh_hi = self.emission_logpdf(r, high=True) + math.log(max(pi_pred, 1e-300))
        lh_lo = self.emission_logpdf(r, high=False) + math.log(max(1.0 - pi_pred, 1e-300))
        m = max(lh_hi, lh_lo)
        num = math.exp(lh_hi - m)
        den = num + math.exp(lh_lo - m)
        return num / den


def analytic_single_regime_hold(mu: float, rho: float) -> bool:
    """The ANALYTIC single-regime optimal-stopping rule (the KNOWN truth the recovery test checks).

    Degenerate the regime-switching problem to ONE fully-observed regime (belief pinned at an
    absorbing corner, ``pi in {0, 1}``): the drift ``mu`` is constant and holding one more day earns
    ``mu`` while discounting the continuation at ``e^{-rho}``. Iterating the Bellman map (*) with a
    constant reward ``mu`` and no belief motion gives the closed-form continuation value of an
    infinite discounted stream ``mu / (1 - e^{-rho})``; it is optimal to CONTINUE HOLDING iff that
    stream is positive, i.e. iff ``mu > 0`` (the discount ``rho > 0`` only scales a positive stream,
    it never flips the sign). So:

        hold  iff  mu > 0        (a net-positive-drift regime is held; the free boundary is a corner)
        exit  iff  mu <= 0       (a non-positive-drift regime is banked immediately)

    Returns ``True`` to HOLD, ``False`` to EXIT. This is the analytic boundary the belief-grid solver
    MUST recover in the absorbing-regime limit (``p_hh -> 1`` pins ``pi = 1`` at the HIGH corner,
    ``p_ll -> 1`` pins ``pi = 0`` at the LOW corner) -- the recovery test asserts exactly that.
    """
    if not rho > 0:
        raise ValueError("rho (discount/impatience) must be strictly positive")
    return mu > 0.0


class OptimalStoppingExit:
    """Phase-C EXIT candidate #5 -- free-boundary optimal-stopping exit under a momentum/regime latent.

    Conforms to ``models.base.ExitModel`` (``fit -> frozen``; causal ``exit_now``). REPLACES the
    incumbent's fixed 5-day horizon with a dynamic voluntary-close of a still-alive WINNER at the free
    boundary ``pi*`` of the belief-space optimal-stopping problem (*), while the true -12% 1-minute
    premium stop remains the RETAINED FLOOR (``STOP_FLOOR``) this rule never loosens (SIGNAL §C.3).

    Usage (harness-driven, interface only)::

        model = OptimalStoppingExit(rho=0.02).fit(train_fold)   # latent + rho fit on TRAIN, FROZEN
        close_now = model.exit_now(cursor, held_days=u, held_return=g)   # causal, F_{t-1}-measurable

    ``k = 6`` (``mu_high, mu_low, sigma, rho`` + 2 free transition entries) is the SYSTEM §C.4
    parsimony count. Purity: ``exit_now`` filters the belief from ``cursor.past()`` (strictly ``< t``)
    only; a same-bar read RAISES via ``ScanCursor.read`` (the pinning test).
    """

    def __init__(self, rho: float = 0.02, *, cadence: Cadence = Cadence.DAILY) -> None:
        """``rho`` = the discount/impatience of the stopping problem (train-frozen; strictly > 0).

        ``rho`` is a pre-registered hyper-parameter of the candidate (SYSTEM §C.9 #5 "*Grid:* discount
        ``rho``"); it is fixed at construction (part of the frozen config), NOT tuned on TEST. The
        latent dynamics are fit in ``fit``. ``cadence`` fixes the purity standard (DAILY: an end-of-day
        decision must be strictly ``F_{t-1}``-measurable).
        """
        if not rho > 0:
            raise ValueError("rho (discount/impatience) must be strictly positive")
        self.k = 6
        self.rho = float(rho)
        self.cadence = cadence
        self.latent: RegimeLatent | None = None   # frozen on fit
        self._pi_star: float | None = None         # the frozen free boundary (belief threshold)
        self._pi0: float | None = None             # frozen stationary belief (the filter's warm start)

    # -- estimation on TRAIN, then FROZEN ----------------------------------------------------------
    def fit(self, train: TrainFold) -> "OptimalStoppingExit":
        """Fit the latent dynamics on TRAIN, solve the free boundary ONCE, and FREEZE (SYSTEM §C.1).

        Estimates ``(mu_high, mu_low, sigma, p_hh, p_ll)`` from the train return series by a short EM
        on the 2-state Gaussian HMM (forward-filter responsibilities only; no smoother -> the fit
        itself stays on the causal side), then solves the belief-grid free boundary ``pi*`` from the
        FROZEN parameters + ``rho`` and stores it. No re-fit ever happens on TEST.
        """
        r = np.asarray(train.series[train.warmup:], dtype="float64")
        self.latent = _fit_regime_em(r, self.rho)
        self._pi_star, self._pi0 = _solve_free_boundary(self.latent, self.rho)
        return self

    # -- the frozen free boundary (read-only view for tests / the harness) -------------------------
    @property
    def pi_star(self) -> float:
        """The frozen free boundary ``pi*``: HOLD iff filtered belief ``>= pi*``, EXIT below it."""
        if self._pi_star is None:
            raise RuntimeError("pi_star before fit -- the free boundary is not solved/frozen yet")
        return self._pi_star

    def filtered_belief(self, past: tuple[float, ...]) -> float:
        """The FILTERED one-step-ahead HIGH-state belief ``Pr(S = HIGH | F_{t-1})`` from PAST returns.

        Runs the strictly-forward Hamilton filter over ``past`` (= ``cursor.past()`` = ``series[:t]``,
        indices ``0..t-1`` ONLY), warm-started at the frozen stationary belief. The value returned is
        ``F_{t-1}``-measurable by construction (it never touches bar ``t``); the exit decision reads
        THIS, never a smoothed posterior (SYSTEM §C.9 #3 -- smoothed is BARRED).
        """
        if self.latent is None or self._pi0 is None:
            raise RuntimeError("filtered_belief before fit")
        pi = self._pi0
        for r in past:
            pi = self.latent.bayes_update(self.latent.predict(pi), r)
        return pi

    # -- the causal one-step-ahead exit decision (ExitModel protocol) ------------------------------
    def exit_now(self, cursor: ScanCursor, held_days: int, held_return: float) -> bool:
        """Causal hold/exit at ``cursor.t`` -- ``True`` to voluntarily CLOSE a still-alive winner now.

        Reads ONLY ``cursor.past()`` (indices ``0..t-1``, ``F_{t-1}``-measurable) to filter the belief
        and compares it to the FROZEN free boundary ``pi*``: EXIT when the filtered high-drift belief
        has fallen BELOW ``pi*`` (the trend-continuation thesis is no longer favored -> bank the
        winner), else keep holding. ``held_days`` (the elapsed-hold covariate ``u``) and
        ``held_return`` are on the interface (EDGE #7 hazard shape) but candidate #5's stopping rule
        is a belief-boundary crossing; ``held_return`` additionally enforces the RETAINED FLOOR.

        THE -12% STOP FLOOR (SIGNAL §C.3, ``STOP_FLOOR``): this model may only close still-alive
        WINNERS earlier -- it may NEVER keep a position that has already breached the loss cap. If the
        running return has hit ``-STOP_FLOOR`` or worse, the floor binds and the position is closed
        regardless of the belief (the model cannot vote to "keep holding" past the stop). The floor is
        actually executed intraday in ``signal/orb.py``; this guard guarantees the model never
        overrides it upward.
        """
        if self._pi_star is None:
            raise RuntimeError("exit_now before fit -- the free boundary is not frozen yet")
        # RETAINED FLOOR: never hold past the -12% loss cap (the model may only close winners early).
        if held_return <= -STOP_FLOOR:
            return True
        # Read the strictly-causal past window through the cursor (a same-bar/future read RAISES;
        # `read` at the pre-decision instant `t-1` routes through the causal guard -- the pinning hook).
        if cursor.t > 0:
            cursor.read(cursor.t - 1)  # F_{t-1} touch: proves the decision only reads strictly-past bars
        pi = self.filtered_belief(cursor.past())
        return pi < self._pi_star  # belief fell below the continuation region -> EXIT (bank the winner)


# ==================================================================================================
# Estimation (a short causal EM) + the belief-grid free-boundary solve (value iteration). Train only.
# ==================================================================================================
def _fit_regime_em(r: np.ndarray, rho: float, *, n_iters: int = 40) -> RegimeLatent:
    """A short 2-state Gaussian-HMM EM on the train returns -> a frozen ``RegimeLatent``.

    Forward-filter responsibilities only (no Kim smoother -> the fit stays causal in spirit and never
    needs the barred smoothed posterior). Initialised by splitting the sample at its median return
    (a robust warm start: the above-median half seeds the HIGH regime). Degenerate inputs (empty /
    near-constant series, or an EM collapse to a single regime) fall back to a well-posed default so
    ``fit`` never raises on a thin train fold; the free-boundary solve then handles the corners.
    """
    if r.size == 0:
        # An EMPTY train fold (e.g. a fold entirely inside warmup): well-posed default, NO np.std on
        # the empty slice (which returns nan + RuntimeWarnings, and `max(nan, 1e-6) == nan` would then
        # fail `RegimeLatent.sigma > 0`). Honour the docstring's "never raises on a thin train fold":
        # a symmetric zero-drift latent with the sigma FLOOR; the free-boundary solve reduces this to
        # the analytic single-drift rule at the corners.
        sd = 1e-6
        return RegimeLatent(mu_high=sd, mu_low=-sd, sigma=sd, p_hh=0.9, p_ll=0.9)
    if r.size < 4 or float(np.std(r)) < 1e-12:
        # Too thin / constant to estimate two regimes: a symmetric well-posed default. The
        # free-boundary solve reduces this to the analytic single-drift rule at the corners.
        mu = float(np.mean(r))
        sd = max(float(np.std(r)), 1e-6)
        return RegimeLatent(mu_high=mu + sd, mu_low=mu - sd, sigma=sd, p_hh=0.9, p_ll=0.9)

    med = float(np.median(r))
    mu_hi = float(np.mean(r[r >= med])) if np.any(r >= med) else float(np.max(r))
    mu_lo = float(np.mean(r[r < med])) if np.any(r < med) else float(np.min(r))
    if mu_hi <= mu_lo:  # keep HIGH strictly the higher-drift regime (identifiability)
        mu_hi, mu_lo = mu_lo + 1e-6, mu_lo
    sig = max(float(np.std(r)), 1e-6)
    p_hh = p_ll = 0.9

    for _ in range(n_iters):
        lat = RegimeLatent(mu_high=mu_hi, mu_low=mu_lo, sigma=sig, p_hh=p_hh, p_ll=p_ll)
        # E-step: forward-filtered HIGH responsibilities gamma_t = Pr(S_t=HIGH | F_t).
        pi0 = _stationary_high(lat)
        gamma = np.empty(r.size, dtype="float64")
        pi = pi0
        for t in range(r.size):
            pi = lat.bayes_update(lat.predict(pi), float(r[t]))
            gamma[t] = pi
        w_hi = gamma
        w_lo = 1.0 - gamma
        s_hi, s_lo = float(w_hi.sum()), float(w_lo.sum())
        if s_hi < 1e-9 or s_lo < 1e-9:
            break  # a regime emptied out -> stop (keep the last well-posed estimate)
        # M-step: weighted-MLE regime means, a pooled vol, and persistence from adjacent-pair mass.
        new_hi = float((w_hi * r).sum() / s_hi)
        new_lo = float((w_lo * r).sum() / s_lo)
        var = float((w_hi * (r - new_hi) ** 2 + w_lo * (r - new_lo) ** 2).sum() / r.size)
        new_sig = max(math.sqrt(max(var, 1e-12)), 1e-6)
        # Persistence from the mean self-responsibility of adjacent bars (a light, stable proxy for
        # the transition MLE that avoids a full pair-marginal pass -- sufficient for a frozen fit).
        pair_hi = float((gamma[1:] * gamma[:-1]).sum())
        pair_lo = float(((1 - gamma[1:]) * (1 - gamma[:-1])).sum())
        new_p_hh = min(max(pair_hi / max(s_hi - gamma[-1], 1e-9), 0.02), 0.98)
        new_p_ll = min(max(pair_lo / max(s_lo - (1 - gamma[-1]), 1e-9), 0.02), 0.98)
        if new_hi <= new_lo:
            new_hi = new_lo + 1e-6
        conv = (abs(new_hi - mu_hi) + abs(new_lo - mu_lo) + abs(new_sig - sig)
                + abs(new_p_hh - p_hh) + abs(new_p_ll - p_ll))
        mu_hi, mu_lo, sig, p_hh, p_ll = new_hi, new_lo, new_sig, new_p_hh, new_p_ll
        if conv < 1e-10:
            break
    return RegimeLatent(mu_high=mu_hi, mu_low=mu_lo, sigma=sig, p_hh=p_hh, p_ll=p_ll)


def _stationary_high(lat: RegimeLatent) -> float:
    """The stationary HIGH-state probability of the transition chain (the filter's warm start).

    For the 2x2 chain the stationary HIGH mass is ``(1-p_ll)/((1-p_hh)+(1-p_ll))``; at the absorbing
    corners (``p_hh=1`` or ``p_ll=1``) the chain is pinned and this returns the corresponding corner.
    """
    a = 1.0 - lat.p_hh  # HIGH -> LOW
    b = 1.0 - lat.p_ll  # LOW  -> HIGH
    if a + b <= 1e-12:
        return 0.5  # both absorbing (degenerate) -> uninformative warm start
    return b / (a + b)


# Fixed standard-normal quadrature nodes/weights (a light Gauss-Hermite-style rule over eps ~ N(0,1))
# used to take the E[ . | pi ] expectation over the next observation in the Bellman map. Precomputed
# once at import (frozen); the value iteration reuses them for every grid point and iteration.
def _normal_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` equal-probability standard-normal nodes (mid-quantiles) + uniform weights ``1/n``.

    A stable, scipy-free quadrature for ``E_{eps~N(0,1)}[f(eps)]``: place nodes at the mid-CDF
    quantiles ``Phi^{-1}((i+0.5)/n)`` so each carries mass ``1/n``. Accurate enough for the smooth
    integrand of the Bellman expectation and fully deterministic (no sampling).
    """
    from .validation import _norm_ppf  # reuse the tested scipy-free inverse-normal (SYSTEM)
    ps = (np.arange(n) + 0.5) / n
    nodes = np.array([_norm_ppf(float(p)) for p in ps], dtype="float64")
    weights = np.full(n, 1.0 / n, dtype="float64")
    return nodes, weights


_QNODES, _QWEIGHTS = _normal_nodes(_QUAD_NODES)


def _solve_free_boundary(lat: RegimeLatent, rho: float) -> tuple[float, float]:
    """Solve the belief-space free boundary ``pi*`` by value iteration on the Bellman map (*).

    Returns ``(pi_star, pi0)`` where ``pi_star`` is the free boundary (HOLD iff belief ``>= pi*``,
    EXIT below) and ``pi0`` the stationary warm-start belief. The value iteration on a ``_VI_GRID``
    belief grid is a contraction (modulus ``e^{-rho} < 1``); at each grid belief ``pi`` the holder's
    continuation value is the expected one-step drift ``mu_bar(pi)`` plus the discounted expected
    next-step value, the expectation taken over the next observation via the fixed normal quadrature
    (predict the belief, draw ``r = mu_bar(pi_pred) + sigma*eps``, Bayes-update, interpolate ``W`` at
    the new belief). ``W(pi) = max(0, hold(pi))``; the free boundary is the largest belief at which
    stopping (``hold <= 0``) is still optimal.

    Recovery guarantee (what the synthetic test pins): as the chain becomes ABSORBING at a corner
    (``p_hh -> 1`` pins ``pi = 1`` on HIGH; ``p_ll -> 1`` pins ``pi = 0`` on LOW), the belief stops
    moving and the map collapses to the analytic single-regime rule ``analytic_single_regime_hold`` --
    hold the corner iff its drift is positive. So ``pi*`` sits at the corner whose drift sign flips.

    ORIENTATION PRECONDITION (binding on the boundary extraction below). The threshold rule ``HOLD iff
    pi >= pi_star`` extracted via ``grid[np.argmax(cont)]`` assumes ``mu_bar(pi)`` is NON-DECREASING in
    the belief ``pi``, i.e. the continuation region is a single UPPER interval -- which holds iff
    ``mu_high >= mu_low`` (HIGH is the higher-drift regime, the identifiability convention). The fitted
    pipeline ALWAYS enforces this (``_fit_regime_em`` bumps ``mu_high`` strictly above ``mu_low`` before
    constructing the latent), so every fit-frozen boundary is well-oriented. This assert makes the
    precondition explicit so a hand-constructed INVERTED latent (``mu_high < mu_low``, mu_bar DECREASING
    in pi -> the continuation region would be a LOWER interval and the extracted boundary orientation
    would be wrong) fails LOUDLY here rather than silently returning a mis-oriented ``pi_star``.
    """
    if lat.mu_high < lat.mu_low:
        raise ValueError(
            "_solve_free_boundary requires mu_high >= mu_low (HIGH is the higher-drift regime): the "
            "single upper-interval 'HOLD iff pi >= pi_star' boundary extraction assumes mu_bar(pi) is "
            "non-decreasing in the belief pi. An inverted latent would yield a mis-oriented boundary. "
            "The fitted pipeline enforces mu_high > mu_low (identifiability) before this solve.")
    grid = np.linspace(0.0, 1.0, _VI_GRID)
    mu_bar = lat.mu_high * grid + lat.mu_low * (1.0 - grid)  # E[drift | pi] at each grid belief
    disc = math.exp(-rho)
    w = np.maximum(mu_bar / (1.0 - disc), 0.0)  # warm start: undiscounted-hold value, clipped at stop

    # Precompute, for each grid belief, the predicted prior and the drift the next obs is centered on.
    pi_pred = lat.p_hh * grid + (1.0 - lat.p_ll) * (1.0 - grid)
    mu_pred = lat.mu_high * pi_pred + lat.mu_low * (1.0 - pi_pred)

    hold = mu_bar.copy()
    for _ in range(_VI_MAX_ITERS):
        # hold(pi) = mu_bar(pi) + e^{-rho} * E[ W(pi') | pi ], the expectation over the next
        # observation via the fixed normal quadrature; W(pi) = max(0, hold(pi)).
        hold = mu_bar + disc * _expected_next_value(grid, pi_pred, mu_pred, lat, w)
        w_new = np.maximum(0.0, hold)
        if float(np.max(np.abs(w_new - w))) < _VI_TOL:
            w = w_new
            break
        w = w_new

    # The free boundary: the continuation region is {pi : hold(pi) > 0} on the CONVERGED value. With
    # mu_bar increasing in pi the region is an upper interval; pi* is the smallest belief at which
    # holding is (weakly) optimal (the same `hold` the fixed point converged to -- computed once).
    cont = hold > 0.0
    if cont.all():
        pi_star = 0.0            # always hold (every belief continues) -> boundary at the low corner
    elif not cont.any():
        pi_star = 1.0 + 1e-9     # never hold voluntarily (exit at any belief) -> boundary above 1
    else:
        pi_star = float(grid[np.argmax(cont)])  # first (smallest) belief in the continuation region
    return pi_star, _stationary_high(lat)


def _expected_next_value(grid, pi_pred, mu_pred, lat: RegimeLatent, w: np.ndarray) -> np.ndarray:
    """The ``E[ W(pi') | pi ]`` term of the Bellman map, vectorised over the belief grid (quadrature)."""
    acc = np.zeros_like(grid)
    for node, wt in zip(_QNODES, _QWEIGHTS):
        r_next = mu_pred + lat.sigma * node
        lh_hi = _gauss(r_next, lat.mu_high, lat.sigma) * pi_pred
        lh_lo = _gauss(r_next, lat.mu_low, lat.sigma) * (1.0 - pi_pred)
        denom = lh_hi + lh_lo
        pi_next = np.where(denom > 0, lh_hi / np.where(denom > 0, denom, 1.0), pi_pred)
        acc += wt * np.interp(pi_next, grid, w)
    return acc


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian density ``N(x | mu, sigma^2)`` (vectorised) -- the emission likelihood in the update."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
