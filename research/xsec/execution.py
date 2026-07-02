"""Weekly rebalance execution: Almgren-Chriss trajectory, cube-root no-trade band, implementation
shortfall (docs/specs/EXECUTION.md §2, §4, §6, §7, §11.1).

Given the Friday-close required trade ``x = w_target - w_carried`` (EXECUTION.md §2.1), this module:

  1. applies a per-name no-trade band (§4) so small drifts are skipped, never fully rebalanced;
  2. solves the discrete Almgren-Chriss trajectory (§2.4) for whatever trade survives the band;
  3. prices the realized fills against the Friday-close decision price as an implementation-shortfall
     report (§7.1), the feedback series OPS consumes (§7.2) and the CPCV/DSR re-run of §9 scores.

Every function below is a PURE function of its inputs (no hidden state, no incrementally-updated
running statistic) so a data restatement is handled by a full recompute (§13's data-revision edge
case), never an incremental patch that could silently diverge.

Causality (Invariant W, §13): everything a ``RebalancePlan`` contains is computed from
``F_{t_0}``-measurable inputs only; only the post-hoc ``ShortfallReport`` (computed strictly AFTER
fills occur) may depend on ``(t_0, t_0+Delta]`` data.

Upstream truths consumed, never re-derived (Invariant N-exec, §1):
  * ``eta_i`` / ``gamma_i`` — the LOCAL LINEARIZATION of COST's fitted ``SpreadImpactModel`` square-
    root impact law ``k_amihud * sqrt(Amihud_i * Notional)`` at the per-slice trade-size scale (§2.2),
    read byte-for-byte off ``model.impact_frac(...)`` in ``cost_facts_from_spread_impact`` — never an
    independently re-fit constant here.
  * ``theta_i`` — the LINEAR fee+half-spread round-trip component (§5.1's
    ``accounting.round_trip_friction`` family; the convex impact term is deliberately excluded from
    the band input, Invariant B-band, §4.4).
  * decision price — ``panel.close`` at rebalance row ``d``, the exact starting mark of
    ``accounting._asset_week_return``.
  * ``sigma_i`` — ``portfolio.realized_vol``'s PRICE-return statistic (§2.5); the band calibration's
    ``sigma_w,i`` is this module's own, genuinely different WEIGHT-drift estimator
    (``weight_drift_vol``, Invariant SIGMA-W-DISTINCT, §4.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from research.xsec.accounting import HALF_SPREAD_FRAC

if TYPE_CHECKING:
    from research.xsec.cost.spread_impact import SpreadImpactModel

# One rebalance period in hours — the operative scale of §2.1's upper regularity bound
# ``Delta << 7 days`` (the weekly cadence of research/xsec/engine.py, EXECUTION.md §2.1/§6.1).
WEEK_HOURS: float = 7.0 * 24.0

# Per-name reason codes carried on a RebalancePlan (EXECUTION.md §11.1 "binding-gate/reason codes").
REASON_NONE = "NONE"  # x_i == 0, nothing to do
REASON_TRADE = "TRADE"  # trades to the band edge (§4.3)
REASON_BAND_SKIP = "BAND-SKIP"  # |x_i| <= b_i, skipped (§4.3)
REASON_FLIP_FULL = "FLIP-FULL"  # quantile-membership flip, trades in FULL, band ignored (§13)
REASON_SHORTABLE_GATE = "SHORTABLE-GATE"  # unshortable short target, trade_i = 0 (§13, Inv SHORTABLE-GATE)
REASON_GROSS_INCREASE_BLOCKED = "GROSS-INCREASE-BLOCKED"  # drift path is risk-off only (§6.2)


@dataclass(frozen=True)
class TrajectoryResult:
    """One name's solved Almgren-Chriss trajectory over ``n`` sub-intervals of ``[t_0, t_0+Delta]``."""

    schedule: np.ndarray  # shape (n,), quantity x_{i,k} traded in each sub-interval, signed, sums to x_i
    kappa: float  # solved front-loading rate for this name this week (kappa=0 => TWAP)
    e_cost: float  # E[C] at the solved trajectory, in NAV-return units (§2.3)
    var_cost: float  # Var[C] at the solved trajectory, in NAV-return^2 units (§2.3)


def solve_trajectory(
    x_i: float,
    *,
    delta: float,
    n: int,
    sigma_i: float,
    eta_i: float,
    gamma_i: float,
    lambda_i: float,
) -> TrajectoryResult:
    """Solve the discrete Almgren-Chriss trajectory for one name's required trade ``x_i`` (§2.4).

    ``x_i`` — required trade for this name this rebalance, fraction of NAV, signed (§2.1).
    ``delta`` — execution window length in the SAME time unit as ``sigma_i`` is quoted in (hours; an
        AMBER venue-liquidity parameter, §2.1 — the caller resolves this from venue config, this
        function does not choose it).
    ``n`` — sub-interval count (AMBER, §2.1); the exact discrete recursion is solved, not a resampled
        continuous curve (§2.4's stated reason: ``n`` is O(10-50), too coarse for resampling). This
        is the ``n*`` of §2.1's lower bound (the required child-slice count).
    ``sigma_i`` — per-unit-time PRICE-return volatility over the execution window (TEAL,
        ``portfolio.realized_vol``'s statistic, NOT ``sigma_w,i`` — §2.5/§4.2 non-identity note).
    ``eta_i``, ``gamma_i`` — temporary / permanent linear impact coefficients, READ byte-identical
        from COST's fitted ``SpreadImpactModel`` (Invariant N-exec, §2.2, §10) — never re-fit here.
    ``lambda_i`` — risk-aversion parameter, RED, set from the stated risk budget ``beta`` via §2.5's
        closed-form ``kappa_i^2`` solve (``lambda_from_budget`` below), never fitted on realized P&L.

    Returns a ``TrajectoryResult`` whose ``schedule`` sums to ``x_i`` (``X_0 = x_i``, ``X_n = 0``
    boundary conditions, §2.1) and whose ``kappa`` is the value implied by
    ``kappa^2 = lambda_i * sigma_i^2 / eta_i`` (§2.4) — ``kappa = 0`` recovers pure TWAP exactly
    (§2.4's ``lambda -> 0`` limit), the expected output for the liquid core (§3).

    The schedule is the exact discrete Almgren-Chriss solution (2001, eq. 12-14): the discrete rate
    ``kappa_d`` solves ``cosh(kappa_d * tau) = 1 + (kappa * tau)^2 / 2`` on the ``n``-slice lattice
    and the remaining position is ``X_j = x_i * sinh(kappa_d * (Delta - j*tau)) / sinh(kappa_d * Delta)``.
    Permanent impact contributes the path-independent ``gamma * x^2 / 2`` to ``e_cost`` and drops out
    of the shape optimization (§2.4).

    Raises ``ValueError`` when ``delta`` fails §2.1's UPPER regularity bound (``delta`` must be a
    positive window strictly shorter than one rebalance period, ``WEEK_HOURS`` — the ``Delta << 7
    days`` bound checked against the weekly cadence). §2.1's LOWER bound
    ``delta >= n / lambda_hat_fill,i`` is a per-name VENUE-metadata condition, not an Almgren-Chriss
    quantity this closed-form solve can compute (the fill-rate forecast lives in the venue adapter);
    it is enforced by ``check_lower_delta_bound`` / ``_plan`` (using ``CostFacts.lambda_fill``)
    strictly UPSTREAM of this function, so "raises on EITHER bound" is honored end-to-end without a
    venue argument leaking into this math (§11.1's exact signature is respected).
    """
    if not math.isfinite(x_i):
        raise ValueError(f"x_i must be finite, got {x_i!r}")
    if not math.isfinite(delta) or delta <= 0.0 or delta >= WEEK_HOURS:
        raise ValueError(
            f"delta={delta!r} fails §2.1's UPPER regularity bound: need 0 < delta < {WEEK_HOURS} "
            "hours (Delta << 7 days, checked against the weekly rebalance cadence)"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if not math.isfinite(sigma_i) or sigma_i < 0.0:
        raise ValueError(f"sigma_i must be finite and >= 0, got {sigma_i!r}")
    if not math.isfinite(eta_i) or eta_i <= 0.0:
        raise ValueError(f"eta_i must be finite and > 0, got {eta_i!r} (missing COST fit? §13, P6)")
    if not math.isfinite(gamma_i) or gamma_i < 0.0:
        raise ValueError(f"gamma_i must be finite and >= 0, got {gamma_i!r}")
    if not math.isfinite(lambda_i) or lambda_i < 0.0:
        raise ValueError(f"lambda_i must be finite and >= 0, got {lambda_i!r}")

    tau = delta / n
    # kappa^2 = lambda*sigma^2/eta (§2.4/§2.5, notation table line 126); written sigma_i**2 to
    # match the spec's sigma-SQUARED expression byte-for-byte (Inv N-exec byte-for-byte kappa check).
    kappa = math.sqrt(lambda_i * sigma_i**2 / eta_i)

    if kappa * tau == 0.0:
        # lambda -> 0 risk-neutral limit: exact TWAP, uniform schedule (§2.4).
        schedule = np.full(n, x_i / n)
        remaining = x_i * (1.0 - np.arange(n + 1) / n)  # linear decay, X_0 = x_i, X_n = 0
    else:
        # exact discrete recursion: cosh(kappa_d*tau) = 1 + (kappa*tau)^2/2 (Almgren-Chriss eq. 12-14).
        kappa_d = math.acosh(1.0 + 0.5 * (kappa * tau) ** 2) / tau
        j = np.arange(n + 1)
        remaining = x_i * np.sinh(kappa_d * (delta - j * tau)) / math.sinh(kappa_d * delta)
        schedule = remaining[:-1] - remaining[1:]

    # E[C] = gamma*x^2/2 (path-independent) + (eta/tau) * sum x_k^2 (temporary, quadratic) — §2.4.
    e_cost = 0.5 * gamma_i * x_i * x_i + (eta_i / tau) * float(np.sum(schedule * schedule))
    # Var[C] = sigma^2 * sum tau * X_{k-1}^2 — exposure held during each sub-interval (§2.4's
    # sigma^2 * integral X(t)^2 dt, discretized at the interval-start position).
    var_cost = sigma_i * sigma_i * tau * float(np.sum(remaining[:-1] ** 2))
    return TrajectoryResult(schedule=schedule, kappa=kappa, e_cost=e_cost, var_cost=var_cost)


def check_lower_delta_bound(delta: float, n: int, lambda_fill_i: float) -> None:
    """Enforce §2.1's LOWER regularity bound ``delta >= n / lambda_hat_fill,i`` for one name.

    This is the second of §2.1's two bounding conditions on the execution window. Unlike the upper
    bound (``delta << 7 days``, a cadence fact ``solve_trajectory`` checks itself), the lower bound
    depends on ``lambda_hat_fill,i`` — this name's CAUSAL fill-rate forecast (AMBER, from venue
    order-book depth/turnover metadata, §2.1), a venue quantity that is NOT part of the closed-form
    Almgren-Chriss solve. It is therefore enforced HERE (and threaded through ``_plan`` from
    ``CostFacts.lambda_fill``), strictly upstream of ``solve_trajectory``, so the "raises if delta
    fails EITHER of §2.1's two bounds" contract of §11.1 is honored end-to-end WITHOUT leaking a
    venue argument into ``solve_trajectory``'s spec-exact signature.

    ``delta`` — execution window length (hours). ``n`` — required child-slice count (``n*``, §2.1).
    ``lambda_fill_i`` — this name's forecast fill rate in slices per unit of the SAME time as
        ``delta``. Must be finite and > 0 (a non-positive/non-finite forecast is itself a violation).

    Raises ``ValueError`` if the forecast fill rate cannot clear ``n`` child slices inside ``delta``
    (``delta < n / lambda_fill_i``) — a trajectory that cannot be filled in time is never silently
    solved (P6).
    """
    if not math.isfinite(lambda_fill_i) or lambda_fill_i <= 0.0:
        raise ValueError(
            f"lambda_fill_i must be finite and > 0 when supplied, got {lambda_fill_i!r}"
        )
    min_delta = n / lambda_fill_i
    if delta < min_delta:
        raise ValueError(
            f"delta={delta!r} fails §2.1's LOWER regularity bound: need delta >= n/lambda_fill_i "
            f"= {n}/{lambda_fill_i!r} = {min_delta!r} so the forecast fill rate can clear all "
            "n child slices inside the window (P6: not silently solved anyway)"
        )


def lambda_from_budget(
    x_i: float,
    w_target_i: float,
    *,
    delta: float,
    beta: float,
    eta_i: float,
    sigma_i: float,
) -> float:
    """Derive the Almgren-Chriss ``lambda_i`` from a STATED variance budget ``beta`` (§2.5).

    §2.5 forbids fitting ``lambda`` on realized rebalance P&L (COST's Invariant C-cfg). Instead
    ``lambda`` is recovered from a governance-level risk budget ``beta in (0, 1)`` — "execution-cost
    variance must not exceed a fraction ``beta`` of one week's target-book variance" — via §2.5's
    verified closed form (the ``17/45`` small-``kappa*Delta`` expansion coefficient of
    ``Var[C]*(kappa)``):

        kappa_i^2  >=  (45 / (17 * Delta^2)) * [ 1 - (3*beta/Delta) * (w_target_i / x_i)^2 ]
        lambda_i   =   kappa_i^2 * eta_i / sigma_i^2                         (§2.4's kappa inverted)

    ``kappa_i`` is taken at the MINIMUM-satisfying value (least front-loading, hence least ``E[C]``,
    consistent with ``beta`` — front-loading is not free, §2.3). This function realizes the
    ``beta -> kappa_i^2 -> lambda_i`` mapping §2.5 derives at length: a book that STATES a positive
    risk budget ``beta`` uses it to obtain a DERIVED ``lambda_`` and injects that via
    ``ExecutionDefaults(lambda_=…)``. A book that states no ``beta`` sits on the spec-anchored
    neutral default ``ExecutionDefaults.lambda_ = 0.0`` (§2.5's "shipped RED default", the TWAP
    limit). Either way ``lambda_`` is a spec-traceable governance number, never fitted on P&L.

    ``x_i`` — required trade for this name, fraction of NAV (§2.1); must be nonzero (a name with no
        required trade has no trajectory to size).
    ``w_target_i`` — this name's target weight (the book-variance term of §2.5's budget).
    ``delta`` — execution window in the SAME per-week-equivalent units ``sigma_i`` is quoted in
        (§2.5's dimensional-consistency note: ``Delta`` as a fraction of a week, e.g. 2h/168h).
    ``beta`` — stated risk budget in (0, 1), RED governance choice (§2.5, §10).
    ``eta_i`` — COST's fitted temporary-impact coefficient (Invariant N-exec), > 0.
    ``sigma_i`` — per-unit-time price-return vol (``portfolio.realized_vol``), > 0.

    Returns ``lambda_i >= 0``. The §2.5 EDGE CASE — when the trade is small enough relative to the
    target weight that even the ``kappa=0`` TWAP variance already satisfies the budget (the bracket
    ``[1 - (3*beta/Delta)(w_target_i/x_i)^2] <= 0``) — returns ``0.0`` (pure TWAP, no front-loading
    required), the EXPECTED non-degenerate liquid-core behavior (§3), not a failure.
    """
    if not math.isfinite(x_i) or x_i == 0.0:
        raise ValueError(f"x_i must be finite and nonzero, got {x_i!r}")
    if not math.isfinite(w_target_i):
        raise ValueError(f"w_target_i must be finite, got {w_target_i!r}")
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be finite and > 0, got {delta!r}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta!r}")
    if not math.isfinite(eta_i) or eta_i <= 0.0:
        raise ValueError(f"eta_i must be finite and > 0, got {eta_i!r}")
    if not math.isfinite(sigma_i) or sigma_i <= 0.0:
        raise ValueError(f"sigma_i must be finite and > 0, got {sigma_i!r}")

    # §2.5's verified kappa_i^2 solve at the constraint's minimum-satisfying value.
    bracket = 1.0 - (3.0 * beta / delta) * (w_target_i / x_i) ** 2
    if bracket <= 0.0:
        # EDGE CASE (§2.5): TWAP already satisfies the budget -> no front-loading, lambda_i = 0.
        return 0.0
    kappa_sq = (45.0 / (17.0 * delta * delta)) * bracket
    # lambda_i = kappa^2 * eta / sigma^2 (§2.4's kappa definition, inverted).
    return kappa_sq * eta_i / (sigma_i * sigma_i)


def band_width(theta_i: float, rho: float, *, c: float) -> float:
    """No-trade band half-width for one name: ``b_i = c * (theta_i / rho)^(1/3)`` (§4.2).

    ``theta_i`` — name i's proportional (LINEAR fee+half-spread only, excludes the convex impact term
        per §4.4's conservative-approximation argument) round-trip cost, TEAL, read from
        ``round_trip_friction``-style inputs (§5.1) at the per-name level.
    ``rho`` — CRRA-style local risk-aversion coefficient (RED, dimensionless,
        ``-w*V''(w)/V'(w)`` at the frictionless optimum) — NOT the same object as this pillar's
        Almgren-Chriss ``lambda`` (§4.2's explicit non-identity note; routing a ``lambda_i`` value
        here is the bug Invariant C-cfg-exec's per-instrument-config discipline catches upstream).
    ``c`` — RED dimensionless constant absorbing the terms the cube-root scaling argument alone does
        not pin down (fitted once, offline, to the book's own ``sigma_w`` — ``weight_drift_vol`` —
        never per-instrument on P&L). The spec commits no numeric value for ``c`` or ``rho``
        (§10: RED, governance-assigned); callers must supply them explicitly.

    Returns ``b_i >= 0``, the half-width of the no-trade region ``[w*_i - b_i, w*_i + b_i]``
    (Constantinides 1986; Davis & Norman 1990, Theorem 2.1 — the CUBE-root exponent, not the
    illustrative quartic-root heuristic §4.2 shows and explicitly rejects).
    """
    if not math.isfinite(theta_i) or theta_i < 0.0:
        raise ValueError(f"theta_i must be finite and >= 0, got {theta_i!r}")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be finite and > 0, got {rho!r}")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError(f"c must be finite and >= 0, got {c!r}")
    return c * (theta_i / rho) ** (1.0 / 3.0)


def weight_drift_vol(
    w_carried: np.ndarray,
    w_target_prior: np.ndarray,
    *,
    half_life_weeks: float,
) -> np.ndarray:
    """Per-name EWMA-std of realized weight drift ``w_carried,i,t - w_target,i,t-1`` (§4.2).

    ``w_carried`` — shape (T, N) history of carried (drifted) weights at each of the last T rebalances.
    ``w_target_prior`` — shape (T, N) history of the PRIOR week's target weights each ``w_carried``
        row drifted from (``w_target_prior[t]`` is the target set at rebalance ``t-1``).
    ``half_life_weeks`` — EWMA half-life in WEEKS (RED, the book's own decision clock — same EWMA
        discipline as ``portfolio.realized_vol``'s window convention, §4.2, but computing a genuinely
        DIFFERENT input series: weight drift, not price return).

    Returns shape (N,) causal EWMA-std of ``w_carried - w_target_prior`` using only the rows passed
    in — the caller passes weights already realized strictly before the current rebalance (P7). NaN
    where a name has fewer than 2 finite drift observations.

    Invariant SIGMA-W-DISTINCT (§10, §13): this is NOT ``portfolio.realized_vol``'s price-return
    statistic — weight drift is price return passed through the vol-scaled weighting's own convexity
    (§4.2), so the two differ in general; a test asserts they are different code paths and values.
    """
    w_carried = np.asarray(w_carried, dtype=float)
    w_target_prior = np.asarray(w_target_prior, dtype=float)
    if w_carried.shape != w_target_prior.shape or w_carried.ndim != 2:
        raise ValueError(
            f"expected matching (T, N) histories, got {w_carried.shape} vs {w_target_prior.shape}"
        )
    if not math.isfinite(half_life_weeks) or half_life_weeks <= 0.0:
        raise ValueError(f"half_life_weeks must be finite and > 0, got {half_life_weeks!r}")

    drift = w_carried - w_target_prior  # (T, N), row T-1 = most recent realized rebalance
    n_obs, n_sym = drift.shape
    decay = 0.5 ** (1.0 / half_life_weeks)
    raw = decay ** np.arange(n_obs - 1, -1, -1, dtype=float)  # heaviest weight on the latest row

    out = np.full(n_sym, np.nan)
    finite = np.isfinite(drift)
    for i in range(n_sym):
        mask = finite[:, i]
        if int(mask.sum()) < 2:
            continue
        u = raw[mask]
        u = u / u.sum()
        d = drift[mask, i]
        mean = float(np.sum(u * d))
        out[i] = math.sqrt(float(np.sum(u * (d - mean) ** 2)))
    return out


@dataclass(frozen=True)
class ShortfallReport:
    """One rebalance's decomposed implementation shortfall (Perold 1988, §7.1)."""

    is_total: float  # book-level IS, NAV-return units, positive = cost (paid more than decision price)
    explicit_cost: float  # fees component (COST §5.2)
    realized_spread: float  # realized half-spread paid (COST §5.3's realized analogue)
    realized_impact: float  # realized temporary + permanent impact (§2.2)
    delay_cost: float  # price drift between t_0 and the first child fill


def implementation_shortfall(
    decision_price: np.ndarray,
    fill_price: np.ndarray,
    fill_notional: np.ndarray,
    *,
    nav: float,
    fee_component: np.ndarray,
    arrival_price: np.ndarray | None = None,
    realized_half_spread: np.ndarray | None = None,
) -> ShortfallReport:
    """Decompose realized shortfall vs. the Friday-close decision price (Perold 1988, §7.1).

    ``decision_price`` — shape (N,) ``panel.close`` at rebalance row ``d`` (research/xsec/panel.py),
        the EXACT price ``accounting._asset_week_return`` uses as the week's starting mark.
    ``fill_price`` — shape (N,) notional-weighted average realized fill price per name across all
        child orders executed in ``[t_0, t_0+Delta]``.
    ``fill_notional`` — shape (N,) total executed notional per name (signed by trade direction).
    ``nav`` — book NAV at ``t_0``, normalizing dollar shortfall into a return-unit ``is_total``.
    ``fee_component`` — shape (N,) realized fee cost per name in dollars (COST §5.2, explicit line).
    ``arrival_price`` — shape (N,) OPTIONAL realized price at the FIRST child fill per name (the
        arrival price of Perold's decision/arrival/fill triptych). When supplied, the delay line is
        the REALIZED drift between the decision print and arrival, ``(arrival-decision)/decision``,
        and the impact line is the realized drift between arrival and the notional-weighted fill —
        the genuine Perold delay-vs-impact split, not a hardcoded 0. When ``None`` the delay is not
        identifiable from a single fill VWAP and is 0 (documented, see attribution note below).
    ``realized_half_spread`` — shape (N,) OPTIONAL realized per-name half-spread paid, in
        FRACTION-of-notional units (COST §5.3's realized analogue; e.g. taken from the venue's own
        touch at fill time). When supplied the spread line is the REALIZED number the fills imply;
        when ``None`` it falls back to ``accounting.HALF_SPREAD_FRAC`` (§5.1's conservative one-truth
        constant) on traded notional — the fallback is NOT invented, it is COST's own anchor.

    Returns a ``ShortfallReport`` whose four cost lines sum to ``is_total`` exactly (Perold 1988's
    TCA decomposition, §7.1) — the series OPS's ``on_obs`` (§7.2) and the §9 CPCV/DSR re-run consume.
    Sign convention: positive = cost (a buy filled ABOVE / a sell filled BELOW the decision price).
    Computed strictly AFTER fills occur — the ONLY function in this module allowed to read
    ``(t_0, t_0+Delta]`` data (Invariant W, §13).

    Attribution (§7.1's Perold decomposition, now genuinely realized when the inputs are supplied):
      * ``explicit_cost`` — fees, always the realized ``fee_component`` (COST §5.2).
      * ``delay_cost`` — decision->arrival drift ``(arrival-decision)/decision`` on signed notional
        when ``arrival_price`` is given; else 0 (a single fill VWAP cannot separate delay from
        impact — stated, not silently fabricated).
      * ``realized_spread`` — ``realized_half_spread`` on traded notional when given; else the
        ``accounting.HALF_SPREAD_FRAC`` fallback (COST's anchor, never a locally-invented number).
      * ``realized_impact`` — the RESIDUAL price shortfall (arrival->fill drift net of the spread
        line), so the four lines sum to ``is_total`` exactly. Any residual misattribution moves cost
        BETWEEN lines that all sum into ``is_total``; the headline ``is_total`` is exact given fills
        + fees regardless of which sub-lines the caller can identify (P6-conservative).
    """
    decision_price = np.asarray(decision_price, dtype=float)
    fill_price = np.asarray(fill_price, dtype=float)
    fill_notional = np.asarray(fill_notional, dtype=float)
    fee_component = np.asarray(fee_component, dtype=float)
    shapes = {decision_price.shape, fill_price.shape, fill_notional.shape, fee_component.shape}
    if len(shapes) != 1 or decision_price.ndim != 1:
        raise ValueError(f"all inputs must share one (N,) shape, got {sorted(shapes)}")
    if not math.isfinite(nav) or nav <= 0.0:
        raise ValueError(f"nav must be finite and > 0, got {nav!r}")
    n = decision_price.shape[0]
    if arrival_price is not None:
        arrival_price = np.asarray(arrival_price, dtype=float)
        if arrival_price.shape != (n,):
            raise ValueError(f"arrival_price must be (N,)={n}, got {arrival_price.shape}")
    if realized_half_spread is not None:
        realized_half_spread = np.asarray(realized_half_spread, dtype=float)
        if realized_half_spread.shape != (n,):
            raise ValueError(f"realized_half_spread must be (N,)={n}, got {realized_half_spread.shape}")

    traded = np.isfinite(fill_notional) & (fill_notional != 0.0)
    if traded.any() and not (
        np.all(np.isfinite(decision_price[traded]))
        and np.all(decision_price[traded] > 0.0)
        and np.all(np.isfinite(fill_price[traded]))
    ):
        raise ValueError("traded names must have finite positive decision prices and finite fills")
    if arrival_price is not None and traded.any() and not np.all(np.isfinite(arrival_price[traded])):
        raise ValueError("traded names must have finite arrival prices when arrival_price is given")

    # per-name price-based shortfall in dollars: (fill - decision)/decision * signed notional.
    # buy (+notional) filled above decision -> +cost; sell (-notional) filled below -> +cost.
    price_is = np.zeros_like(fill_notional)
    price_is[traded] = (
        (fill_price[traded] - decision_price[traded])
        / decision_price[traded]
        * fill_notional[traded]
    )
    price_is_total = float(np.sum(price_is)) / nav

    explicit_cost = float(np.sum(np.nan_to_num(fee_component, nan=0.0))) / nav

    # delay = realized decision->arrival drift on signed notional (Perold), 0 if arrival unknown.
    if arrival_price is None:
        delay_cost = 0.0
    else:
        delay = np.zeros_like(fill_notional)
        delay[traded] = (
            (arrival_price[traded] - decision_price[traded])
            / decision_price[traded]
            * fill_notional[traded]
        )
        delay_cost = float(np.sum(delay)) / nav

    # realized half-spread paid: the venue's realized number when supplied, else COST's anchor.
    if realized_half_spread is None:
        realized_spread = HALF_SPREAD_FRAC * float(np.sum(np.abs(fill_notional[traded]))) / nav
    else:
        hs = np.nan_to_num(realized_half_spread, nan=0.0)
        realized_spread = float(np.sum(hs[traded] * np.abs(fill_notional[traded]))) / nav

    # impact absorbs the residual so the four Perold lines sum to the exact price-based total.
    realized_impact = price_is_total - delay_cost - realized_spread
    is_total = explicit_cost + realized_spread + realized_impact + delay_cost
    return ShortfallReport(
        is_total=is_total,
        explicit_cost=explicit_cost,
        realized_spread=realized_spread,
        realized_impact=realized_impact,
        delay_cost=delay_cost,
    )


# ---------------------------------------------------------------------------------------------
# facts / config / plan containers (§11's facts+assemble layers, single-module scope per §11.1)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecTarget:
    """This rebalance's ``w_target`` plus the per-name shortable eligibility flag (§11.1, §13).

    PORTFOLIO/UNIVERSE-owned facts this pillar reads but never recomputes (P2). ``shortable`` is the
    universe layer's per-name, per-week boolean companion field (Invariant SHORTABLE-GATE).
    """

    w_target: np.ndarray  # (N,) dollar-neutral target weights (portfolio.quantile_weights output)
    shortable: np.ndarray  # (N,) bool: universe-layer shortability flag at t_0


@dataclass(frozen=True)
class CostFacts:
    """COST-owned inputs, read-only (Invariant N-exec) — the §11 ``facts/cost_adapter`` bundle.

    ``eta`` / ``gamma`` are COST's fitted temporary/permanent impact coefficients (byte-identical,
    never re-fit here); ``theta`` is the LINEAR fee+half-spread round-trip component per name (§4.2,
    §5.1 — excludes convex impact, Invariant B-band); ``sigma`` is ``portfolio.realized_vol``'s
    per-name PRICE-return vol (TEAL), routed through this facts bundle for the trajectory solve
    (§2.5 — NOT ``sigma_w,i``).
    """

    eta: np.ndarray  # (N,) temporary impact coeff, = COST's impact_frac at the slice scale (§2.2)
    gamma: np.ndarray  # (N,) permanent impact coeff, RED §2.2 split of the same fitted impact
    theta: np.ndarray  # (N,) linear round-trip cost (fee + half-spread only, §4.2/§4.4)
    sigma: np.ndarray  # (N,) price-return vol per unit of the SAME time unit as cfg.delta_hours
    lambda_fill: np.ndarray | None = None  # (N,) causal fill-rate forecast (§2.1 lower bound); AMBER
    # venue order-book depth/turnover metadata. When present, solve_trajectory enforces §2.1's LOWER
    # delta bound per name; None => the bound is not checkable here (the venue adapter owns it, §2.1).


def cost_facts_from_spread_impact(
    model: "SpreadImpactModel",
    amihud: np.ndarray,
    *,
    theta: np.ndarray,
    sigma: np.ndarray,
    slice_notional: np.ndarray,
    permanent_fraction: float = 0.0,
    lambda_fill: np.ndarray | None = None,
) -> CostFacts:
    """Derive the linear Almgren-Chriss ``eta``/``gamma`` from COST's fitted ``SpreadImpactModel``
    at the per-slice trade-size scale (Invariant N-exec, EXECUTION.md §2.2).

    COST's fitted impact law is the SQUARE-ROOT form ``impact_frac_i(Q) = k_amihud *
    sqrt(Amihud_i * Q)`` (``research/xsec/cost/spread_impact.py::SpreadImpactModel.impact_frac``,
    COST.md §5.3) — the model exposes ``k_amihud`` and the ``impact_frac`` method, NOT a pre-baked
    linear ``eta``/``gamma`` pair. Per EXECUTION.md §2.2, the LINEAR Almgren-Chriss ``eta`` used
    inside one sub-interval is the LOCAL LINEARIZATION of that same square-root curve around the
    sub-interval's own trade-size scale, ``read off COST's fitted k_amihud/Amihud_i at the notional
    implied by trading x_i over n slices`` — never an independently re-fit constant (Invariant
    N-exec keeps COST P1's one impact truth).

    Concretely, at the per-slice dollar notional ``Q_i = slice_notional[i]``:
      * ``impact_i = model.impact_frac(Amihud_i, Q_i)`` — COST's own number, byte-for-byte, no
        re-fit or rescale (this call routes through ``k_amihud`` + the class fallback exactly as
        ``settle_week``'s cost path does).
      * the linear temporary coefficient is ``eta_i = impact_i`` in fraction-of-notional-per-slice
        units: it is the impact COST prices for ONE slice's trade, so a trajectory trading a
        per-slice quantity reproducing ``Q_i`` pays exactly COST's ``impact_frac`` for that slice —
        the local-linear coefficient that makes the sub-interval's temporary cost equal COST's
        fitted square-root value at the slice scale, not a lighter EXECUTION-local guess.

    ``model`` — COST's fitted ``SpreadImpactModel`` (read-only; ``k_amihud``/``impact_frac`` are the
        underlying fit, consumed byte-identical, Invariant N-exec).
    ``amihud`` — shape (N,) per-name ``Amihud_i`` illiquidity ratios (COST §5.3's AMBER trailing
        statistic, ``research/xsec/cost/spread_impact.py::amihud_ratio``); passed to ``impact_frac``
        exactly as ``settle_week``'s cost path passes it.
    ``theta`` — shape (N,) LINEAR fee+half-spread round-trip component (§4.2/§4.4, Invariant B-band).
    ``sigma`` — shape (N,) ``portfolio.realized_vol`` price-return vol (§2.5, NOT ``sigma_w,i``).
    ``slice_notional`` — shape (N,) per-slice dollar notional ``nav*|x_i|/n`` (the trade-size scale
        the §2.2 linearization is taken around; the caller resolves it from the required trade and
        ``cfg.n_slices``, this function does not choose ``n`` or NAV).
    ``permanent_fraction`` — RED §2.2 split of COST's single fitted impact number into the
        path-independent PERMANENT term ``gamma`` vs. the shape-dependent TEMPORARY term ``eta``.
        COST fits ONE impact truth (no separate permanent coefficient), so any temporary/permanent
        split is a governance choice, not a fitted quantity — it defaults to ``0.0`` (the
        conservative Almgren-Chriss all-temporary baseline: no permanent price move assumed, §2.2),
        never a fabricated nonzero constant. When ``>0``, ``gamma_i`` absorbs that fraction of the
        fitted impact as a path-independent per-unit permanent coefficient.

    Returns a ``CostFacts`` whose ``eta``/``gamma`` are FUNCTIONS OF COST's own fitted ``k_amihud``
    (byte-for-byte via ``impact_frac``), never re-fit here. A test asserts ``eta_i`` equals
    ``model.impact_frac(Amihud_i, Q_i)`` to exact float equality (no tolerance).
    """
    amihud = np.asarray(amihud, dtype=float)
    slice_notional = np.asarray(slice_notional, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = amihud.shape[0]
    if not (slice_notional.shape == theta.shape == sigma.shape == (n,)):
        raise ValueError("amihud, slice_notional, theta and sigma must share one (N,) shape")
    if not (0.0 <= permanent_fraction <= 1.0):
        raise ValueError(f"permanent_fraction must be in [0, 1], got {permanent_fraction!r}")

    # COST's fitted impact fraction at the per-slice trade-size scale, byte-for-byte (Inv N-exec):
    # this is the SAME model.impact_frac(...) settle_week's cost path calls, no re-fit/rescale.
    impact = np.array(
        [model.impact_frac(float(amihud[i]), float(slice_notional[i])) for i in range(n)]
    )
    # §2.2 temporary/permanent split of COST's ONE fitted impact number; permanent_fraction=0
    # (default) => pure-temporary all-eta baseline, no fabricated permanent coefficient.
    gamma = permanent_fraction * impact
    eta = (1.0 - permanent_fraction) * impact
    lam_fill = None if lambda_fill is None else np.asarray(lambda_fill, dtype=float)
    if lam_fill is not None and lam_fill.shape != (n,):
        raise ValueError(f"lambda_fill must be (N,)={n}, got {lam_fill.shape}")
    return CostFacts(eta=eta, gamma=gamma, theta=theta, sigma=sigma, lambda_fill=lam_fill)


@dataclass(frozen=True)
class ExecutionDefaults:
    """Resolved execution config (§10's TEAL/AMBER/RED buckets, §11 ``config/``).

    Invariant C-cfg-exec: every field is a BOOK- or CLASS-level scalar — none is per-instrument.
    RED fields (``rho``, ``band_c``, ``lambda_``) are governance choices, never fitted on realized
    P&L. Two of them (``rho``, ``band_c``) commit NO numeric value in the spec (§4.2/§10 — ``c`` is
    "fitted once offline to sigma_w"), so they are REQUIRED here, not defaulted — refusing to invent
    a number is the contract. ``lambda_`` is DIFFERENT: EXECUTION.md §2.5 now anchors a shipped RED
    default of ``0.0`` — the risk-neutral / TWAP limit — as a *designed knob* with a stated
    rationale (it is §2.4's closed-form ``λ→0 ⇒ κ=0`` identity, it reproduces the ``λ=0``
    full-rebalance backtest in ``docs/EDGE.md``, and it declines to pay the front-loading ``E[C]``
    cost until a book states a positive risk budget ``β``). A book that HAS a ``β`` derives its
    ``λ`` from ``lambda_from_budget(β, …)`` and injects it via ``ExecutionDefaults(lambda_=…)`` —
    the injection path stays open, this default only chooses the neutral spec-cited value for a book
    that has not stated one.
    """

    rho: float  # RED §4.2: CRRA-style band curvature (dimensionless; NOT lambda_)
    band_c: float  # RED §4.2: cube-root band constant, fitted once offline to sigma_w
    delta_hours: float = 2.0  # AMBER §2.1: window length, midpoint of the stated 1-4h range (§2.5)
    n_slices: int = 10  # AMBER §2.1: child-slice count, n* = O(10)
    # RED §2.5: risk aversion. Shipped default = 0.0 = risk-neutral/TWAP limit -- a spec-anchored
    # designed knob (§2.5 "shipped RED default" para): the λ→0 closed-form identity, reproduces the
    # λ=0 backtest in docs/EDGE.md, and pays no front-loading E[C] until a book states β>0. Inject a
    # λ>0 from lambda_from_budget(β, …) for a book that has a stated budget.
    lambda_: float = 0.0


@dataclass(frozen=True)
class OpsObservation:
    """One data-contract violation raised to OPS's ``on_obs`` seam (§7.2, §13 — P2: report, never halt)."""

    code: str  # e.g. "SHORTABLE-GATE"
    index: int  # symbol column index in the panel/weight vectors
    detail: str


@dataclass(frozen=True)
class RebalancePlan:
    """Per-name child-order schedule for ``[t_0, t_0+Delta]`` (§11 ``assemble/rebalance_plan``).

    Everything here is ``F_{t_0}``-measurable (Invariant W): the plan never reads realized data from
    inside the execution window. ``trade`` is the post-band, post-gate trade vector; names with a
    nonzero trade carry a solved ``TrajectoryResult`` in ``trajectories`` (keyed by symbol index).
    """

    x: np.ndarray  # (N,) required trade w_target - w_carried (§2.1), pre-band, for audit
    band: np.ndarray  # (N,) no-trade band half-width b_i applied (§4.2-4.3)
    trade: np.ndarray  # (N,) trade actually planned (0 = skipped/gated)
    reason: tuple[str, ...]  # (N,) per-name reason code (REASON_* constants)
    trajectories: Mapping[int, TrajectoryResult] = field(default_factory=dict)
    ops_observations: Mapping[int, OpsObservation] = field(default_factory=dict)

    @property
    def turnover_live(self) -> float:
        """Planned live turnover ``0.5 * sum_i |trade_i|`` (§6.4)."""
        return 0.5 * float(np.sum(np.abs(self.trade)))

    @property
    def turnover_backtest(self) -> float:
        """The backtest's full-rebalance turnover ``0.5 * sum_i |x_i|`` (accounting.py, §6.4).

        Invariant E-monotone (P4): ``turnover_live <= turnover_backtest`` always — the band only
        SKIPS trades the backtest took, never adds one; report as a one-sided inequality.
        """
        return 0.5 * float(np.sum(np.abs(self.x)))


def _plan(
    w_target: np.ndarray,
    shortable: np.ndarray,
    w_carried: np.ndarray,
    cost: CostFacts,
    cfg: ExecutionDefaults,
    *,
    reduce_only: bool,
) -> RebalancePlan:
    """Shared band/gate/trajectory assembly for the calendar (§6.1) and drift (§6.2) paths."""
    w_target = np.asarray(w_target, dtype=float)
    w_carried = np.asarray(w_carried, dtype=float)
    shortable = np.asarray(shortable, dtype=bool)
    n = w_target.shape[0]
    if not (w_carried.shape == shortable.shape == (n,)):
        raise ValueError("w_target, w_carried and shortable must share one (N,) shape")

    x = w_target - w_carried
    band = np.array([band_width(float(t), cfg.rho, c=cfg.band_c) for t in cost.theta])

    trade = np.zeros(n)
    reason: list[str] = [REASON_NONE] * n
    obs: dict[int, OpsObservation] = {}
    for i in range(n):
        if w_target[i] < 0.0 and not shortable[i]:
            # composed quantile-flip/unshortable edge case (§13): never traded, never silent —
            # raised to OPS as a data-contract violation, target unreachable this rebalance.
            reason[i] = REASON_SHORTABLE_GATE
            obs[i] = OpsObservation(
                code="SHORTABLE-GATE",
                index=i,
                detail=(
                    f"non-zero short target {w_target[i]:+.6f} without shortable=True; "
                    "trade_i=0, data-contract violation (EXECUTION.md §13, Inv SHORTABLE-GATE)"
                ),
            )
            continue
        if x[i] == 0.0:
            continue
        flip = (w_target[i] == 0.0) != (w_carried[i] == 0.0) or w_target[i] * w_carried[i] < 0.0
        if flip:
            # quantile-membership flip: a discrete signal change, NEVER band-gated (§13).
            trade[i] = x[i]
            reason[i] = REASON_FLIP_FULL
        elif abs(x[i]) <= band[i]:
            reason[i] = REASON_BAND_SKIP
        else:
            # trade only to the band EDGE, never back to the exact target (§4.3).
            trade[i] = x[i] - math.copysign(band[i], x[i])
            reason[i] = REASON_TRADE
        if reduce_only and trade[i] != 0.0 and abs(w_carried[i] + trade[i]) > abs(w_carried[i]):
            # drift path is a risk-off valve only: never increases per-name gross (§6.2).
            trade[i] = 0.0
            reason[i] = REASON_GROSS_INCREASE_BLOCKED

    trajectories: dict[int, TrajectoryResult] = {}
    for i in np.flatnonzero(trade != 0.0):
        ii = int(i)
        # §2.1 LOWER bound (venue fill-rate condition), enforced upstream of the closed-form solve
        # so solve_trajectory keeps §11.1's exact signature (no venue arg leaks into the AC math).
        if cost.lambda_fill is not None:
            check_lower_delta_bound(cfg.delta_hours, cfg.n_slices, float(cost.lambda_fill[ii]))
        trajectories[ii] = solve_trajectory(
            float(trade[i]),
            delta=cfg.delta_hours,
            n=cfg.n_slices,
            sigma_i=float(cost.sigma[i]),
            eta_i=float(cost.eta[i]),
            gamma_i=float(cost.gamma[i]),
            lambda_i=cfg.lambda_,
        )

    return RebalancePlan(
        x=x,
        band=band,
        trade=trade,
        reason=tuple(reason),
        trajectories=MappingProxyType(trajectories),
        ops_observations=MappingProxyType(obs),
    )


class ExecutionEngine:
    """The one public method (§12); matches the shape of ``core/protocols.py``'s
    ``ExecutionEngine.on_target`` seam, generalized from a single order's tactic/price/qty to a
    per-name ``RebalancePlan``. Stateless — a pure function wrapped in the protocol's class shape."""

    def on_target(
        self,
        target: ExecTarget,
        carried: np.ndarray,
        cost: CostFacts,
        regime_defensive: bool,
        cfg: ExecutionDefaults,
    ) -> RebalancePlan:
        """Turn ``target.w_target`` and ``carried`` (w_carried) into a per-name child-order schedule.

        The signature is EXACTLY §11.1/§12's build contract
        ``(target, carried, cost, regime_defensive, cfg)`` — no realized-``(t_0, t_0+Delta]`` window
        argument. Invariant W (§13) is upheld STRUCTURALLY, not by a testability hook: the ONLY
        inputs this method accepts are the ``F_{t_0}``-measurable objects above, so there is no
        channel through which realized window data could enter the plan. A test
        (``test_invariants.py::test_planned_trajectory_ignores_realized_window_data``) asserts, via
        ``inspect.signature``, that these five parameters are the complete set — so any future edit
        that adds a realized-data parameter (the only way window data could leak in) FAILS the test.
        The firewall is the signature itself; the sole function permitted to read
        ``(t_0, t_0+Delta]`` data is the post-hoc ``implementation_shortfall``.

        ``target`` — this rebalance's ``w_target`` plus the per-name shortable eligibility flag
            (Invariant SHORTABLE-GATE); PORTFOLIO/UNIVERSE-owned facts read, never recomputed (P2).
        ``carried`` — shape (N,) ``w_carried``, the drifted prior book (accounting.py's convention).
        ``cost`` — COST's fitted ``eta``/``gamma`` (read-only, Invariant N-exec) + fee/spread facts.
        ``regime_defensive`` — REGIME's one consumed boolean (§8). The CALENDAR path here is the
            always-on weekly trigger (§6.1) and fires regardless; this flag only ARMS the separate
            intra-week ``drift_check`` path (§6.2, Invariant CAL-PRIMARY) and is accepted here so
            the seam carries it explicitly rather than through hidden state.
        ``cfg`` — resolved ``ExecutionDefaults`` (TEAL/AMBER/RED per §10, Invariant C-cfg-exec).

        Returns a ``RebalancePlan``: per-name ``trade_i`` after the no-trade band (§4.3's
        ``trade_i = 0 if |x_i| <= b_i else x_i - sign(x_i)*b_i`` rule), each survivor's
        ``TrajectoryResult`` (``solve_trajectory``), and binding-gate/reason codes (e.g.
        ``SHORTABLE-GATE`` violations, §13). Does not place orders itself — the Nautilus adapter
        (§1, §15) consumes the plan and emits ``OrderCommand``s.
        """
        del regime_defensive  # consumed by drift_check only (§6.2); the calendar path is always-on
        return _plan(
            target.w_target, target.shortable, carried, cost, cfg, reduce_only=False
        )


def drift_check(
    target: ExecTarget,
    carried: np.ndarray,
    cost: CostFacts,
    cfg: ExecutionDefaults,
    *,
    regime_defensive: bool,
) -> RebalancePlan | None:
    """Intra-week drift-triggered path (§6.2-6.3) — DEFENSIVE ADDITION, REGIME-gated only.

    Returns ``None`` whenever ``regime_defensive`` is False, REGARDLESS of drift magnitude
    (Invariant CAL-PRIMARY, P5: the calendar cadence is the only always-on trigger; a drift trade
    fired without an active REGIME defensive flag is the bug this gate forbids). When armed, drift
    is measured against the CURRENTLY-VALID ``w_target`` (last Friday's target, §6.3) through the
    SAME band rule, and the resulting plan is a risk-off valve only: any per-name trade that would
    INCREASE that name's gross exposure is zeroed (§6.2 — never opportunistic re-entry, reason code
    ``GROSS-INCREASE-BLOCKED``).
    """
    if not regime_defensive:
        return None
    return _plan(target.w_target, target.shortable, carried, cost, cfg, reduce_only=True)


# ---------------------------------------------------------------------------------------------
# §9 validation deliverable: the execution-adjusted weekly series through the SAME battery
# ---------------------------------------------------------------------------------------------


def execution_adjusted_series(
    net_frictionless: np.ndarray,
    realized_shortfall: np.ndarray,
) -> np.ndarray:
    """Fold the realized implementation-shortfall series into the frictionless weekly net series (§9).

    §0/§9: the backtest's ``accounting.settle_week`` marks each week's P&L as a frictionless
    INSTANTANEOUS Friday-close fill (the DECISION-price benchmark, never the fill price). This pillar
    closes that gap by pricing the realized shortfall (``implementation_shortfall``, §7.1) and
    FEEDING IT BACK so the honest net Sharpe is measured against realized fills. The
    execution-adjusted week ``t`` return is simply

        net_adj[t] = net_frictionless[t] - realized_shortfall[t]

    where ``realized_shortfall[t]`` is the week-``t`` ``ShortfallReport.is_total`` (NAV-return units,
    positive = cost, §7.1). Subtracting a POSITIVE shortfall lowers the realized return — the P6
    conservative direction (execution can only make the realized series WORSE than the frictionless
    proxy, never better; §0). This is a pure per-week transform, not a new statistic.

    ``net_frictionless`` — shape (T,) the engine's per-week net return series
        (``research.xsec.engine.run(...).net``), the decision-price benchmark.
    ``realized_shortfall`` — shape (T,) per-week ``is_total`` from ``implementation_shortfall``,
        aligned week-for-week to ``net_frictionless``.

    Returns shape (T,) the execution-adjusted net series — the SECOND INPUT SERIES that §9 runs
    through ``validate.py``/``cpcv.py`` byte-for-byte (never a new statistical test).
    """
    net = np.asarray(net_frictionless, dtype=float)
    shortfall = np.asarray(realized_shortfall, dtype=float)
    if net.shape != shortfall.shape or net.ndim != 1:
        raise ValueError(
            f"net_frictionless and realized_shortfall must share one (T,) shape, "
            f"got {net.shape} vs {shortfall.shape}"
        )
    return net - shortfall


@dataclass(frozen=True)
class ExecutionValidation:
    """The §9 verdict: the execution-adjusted weekly series through the SAME validation battery.

    Every field is the OUTPUT of ``convexity.system.validate.{walkforward,deflated,cpcv}`` (reused
    BYTE-FOR-BYTE via ``research.xsec.{validate,cpcv}``, Invariant N-exec) on the execution-adjusted
    series — never a new statistical test (§9). ``naive_*`` fields carry the frictionless baseline the
    adjusted series is compared against, so the haircut of realizing execution is auditable.
    """

    wf_oos_sharpe_ann: float  # purged walk-forward OOS Sharpe on the ADJUSTED selected series
    naive_full_sharpe_ann: float  # full-sample Sharpe of the ADJUSTED selected series
    cpcv_mean_path_sharpe: float  # CPCV mean-path Sharpe (adjusted); baseline ~0.906 (docs/EDGE.md)
    cpcv_frac_pos: float  # CPCV fraction of paths with OOS Sharpe > 0; baseline 0.85-1.00
    cpcv_haircut: float  # naive_full - cpcv_mean (adjusted)
    cpcv_pbo: float  # PBO over the lookback family (adjusted); baseline 0.357
    dsr_passed: bool  # DeflatedSharpe.deflate(...).passed on the adjusted series


def reconcile(
    family_series: Mapping[str, np.ndarray],
    selected_label: str,
    realized_shortfall_selected: np.ndarray,
    *,
    n_groups: int = 8,
    k_test: int = 2,
    embargo: int = 1,
    wf_folds: int = 3,
    seed: int = 0,
) -> ExecutionValidation:
    """Re-run §9's validation battery on the execution-adjusted weekly series (EXECUTION.md §9).

    This is §9's deliverable check: "does the realized-shortfall-adjusted weekly net-return series
    still pass the SAME validation battery ``docs/EDGE.md`` reports for the frictionless backtest
    series?" It does NOT define a new test — it routes the adjusted series through the EXISTING
    machinery byte-for-byte (Invariant N-exec):

      * purged walk-forward  — ``research.xsec.validate.purged_walkforward`` (reuses SYSTEM's
        ``make_walkforward_folds``);
      * CPCV                 — ``research.xsec.cpcv.run_cpcv`` (reuses SYSTEM's ``SystemCPCV`` +
        the LdP PBO estimator), N=8/k=2/28 paths/1-week embargo — the recorded baseline config;
      * DSR                  — ``research.xsec.validate.deflated_sharpe`` (reuses SYSTEM's
        ``DeflatedSharpe.deflate``) over the lookback-family trial Sharpes.

    Only the SELECTED config's series is shortfall-adjusted (the deployed book is one ``w_target``
    stream, §2.1a); the rest of the family is fed frictionless as the PBO trial family exactly as
    ``scripts/xsec_cpcv.py`` does — the adjustment is a property of the deployed book's realized
    fills, and PBO's question ("is picking the best lookback overfitting?") is answered on the same
    decision-price family the frictionless run used.

    ``family_series`` — {label: frictionless weekly net series} for the aligned lookback family
        (``research.xsec.engine.run(...).net`` per L, aligned to a common length — see
        ``scripts/xsec_cpcv.py``).
    ``selected_label`` — the deployed config's label (e.g. the L=14 fitted peak).
    ``realized_shortfall_selected`` — shape (T,) per-week ``ShortfallReport.is_total`` for the
        deployed book, aligned to ``family_series[selected_label]`` (§7.1). All-zeros recovers the
        frictionless baseline exactly (the ``Delta->0, lambda->0`` corner, §13).
    ``n_groups`` / ``k_test`` / ``embargo`` — CPCV split family (recorded baseline: 8 / 2 / 1).
    ``wf_folds`` — purged walk-forward fold count. ``seed`` — CPCV path seed.

    Returns an ``ExecutionValidation`` whose fields are directly comparable to ``docs/EDGE.md``'s
    frictionless baselines: a material ``cpcv_frac_pos`` drop or a DSR degradation from
    track-length-limited to ``SR_pp``-fail is the §9 signal that execution eats real edge (retuned
    only via the RED governance path, §10 — never a per-instrument tune on realized P&L).
    """
    from research.xsec.cpcv import run_cpcv
    from research.xsec.validate import deflated_sharpe, purged_walkforward

    if selected_label not in family_series:
        raise KeyError(f"selected_label {selected_label!r} not in family {tuple(family_series)}")

    # Adjust ONLY the deployed config's realized series; the family stays frictionless for PBO.
    adjusted = execution_adjusted_series(
        family_series[selected_label], realized_shortfall_selected
    )
    adj_family: dict[str, np.ndarray] = {
        lab: (adjusted if lab == selected_label else np.asarray(s, dtype=float))
        for lab, s in family_series.items()
    }

    wf = purged_walkforward(adjusted, n_folds=wf_folds, embargo_weeks=embargo)
    cpcv = run_cpcv(
        adj_family, selected_label, n_groups=n_groups, k_test=k_test, embargo=embargo, seed=seed
    )
    # DSR over the FULL lookback family's trial Sharpes (the multiplicity the headline is selected
    # from), with the selected trial's own Sharpe adjusted — the same trial set the frictionless run
    # deflated against, so the comparison is like-for-like (§9).
    from research.xsec.cpcv import _sharpe_ann  # the family's own annualized-Sharpe helper

    trial_sharpes = np.array([_sharpe_ann(adj_family[lab]) for lab in adj_family])
    dsr = deflated_sharpe(adjusted, trial_sharpes)

    return ExecutionValidation(
        wf_oos_sharpe_ann=wf.oos_sharpe_ann,
        naive_full_sharpe_ann=cpcv.path_summary.naive_full_sharpe,
        cpcv_mean_path_sharpe=cpcv.path_summary.mean,
        cpcv_frac_pos=cpcv.path_summary.frac_pos,
        cpcv_haircut=cpcv.path_summary.haircut,
        cpcv_pbo=cpcv.pbo.pbo,
        dsr_passed=bool(dsr.passed),
    )
