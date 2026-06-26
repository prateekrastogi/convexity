"""Carver dynamic optimisation — Advanced Futures Trading Strategies (2023), Strategy 25.

Greedy tracking-error minimisation that turns a large universe of *ideal* (forecast-scaled,
unrounded) positions into a small set of integer-contract positions that best replicate the
ideal portfolio under limited capital. The instruments that receive a non-zero optimised
position are the dynamically-optimised list of candidates to actually trade.

Exact maths (transcribed from the chapter's equation images):

  weight per contract      cᵢ  = notional_per_contractᵢ / capital
                               = (multiplierᵢ · priceᵢ · fx_rateᵢ) / capital
  optimal (target) weight  wᵢ  = Nᵢ · cᵢ                         (Nᵢ = ideal unrounded contracts)
  covariance               Σ   = σᵀ · ρ · σ        i.e. Σᵢⱼ = σᵢ · ρᵢⱼ · σⱼ   (σ annualised)
  tracking-error weights   e   = w_candidate − w_target
  tracking error (std)     TE  = √(eᵀ · Σ · e)
  cost in weight terms     wᶜᵢ = (Cᵢ / capital) / cᵢ            (Cᵢ = cost per contract, currency)
  trade as a weight        Δᵢ  = |w_prevᵢ − w_candidateᵢ|
  cost of all trades       δ   = Σ Δᵢ · wᶜᵢ
  objective (with costs)   TE  = √(eᵀ Σ e) + COST_MULTIPLIER · δ      (COST_MULTIPLIER = 50)
  tracking-error buffer    Bσ  = BUFFER_FRACTION · τ              (τ = risk target; 0.05·0.20 = 1%)
  adjustment factor        α   = max((T − Bσ) / T, 0)             (T = TE of current vs optimised)
  required trade           round(α · (N*ᵢ − Pᵢ))

Greedy algorithm: start from all-zero weights; each pass, for every instrument add ONE contract
in the *direction of the target's sign* (never against the forecast), evaluate the objective,
keep the single best one-contract move; adopt it if it lowers the objective; repeat until no
single move improves. Positions are integers by construction (no rounding needed).

>>> BOOK ERRATA (verified): the worked example in Tables 116–123 prints intermediate tracking
errors (zero→2.67%, buy-5y→3.43%, …) and a final position of [0, 1, 2] that do NOT follow from
the chapter's own stated formula together with the covariance in Table 115 (which this module
reproduces to the last decimal — see tests). The formula-faithful greedy optimum for that
example is [0, 1, 3] (≈ naive rounding of the optimal [0.4, 0.9, 3.1]); [0,1,3] has tracking
error 0.30% versus 0.78% for [0,1,2], so the optimiser correctly keeps the 3rd S&P contract.
This module implements the canonical formula (matching Carver's `pysystemtrade` reference),
not the book's mis-arithmetic. Tests assert both the exact Table-115 covariance and the
formula-faithful result. <<<
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COST_MULTIPLIER: float = 50.0  # δ weight added to the tracking error. Book uses 50 ("any number
# 10–100 gives similar results"); Carver's pysystemtrade reference defaults trade_shadow_cost=10.
BUFFER_FRACTION: float = 0.05  # 10% symmetric position buffer == 5% asymmetric tracking-error buffer
DEFAULT_RISK_TARGET: float = 0.20  # τ — annualised risk target used in the book
_NO_LIMIT: int = 10**9  # sentinel for an unbounded position limit
_MAX_ITERS: int = 100_000  # greedy is monotone & bounded; this is only a runaway guard


def _round_half_up(x):
    """Deterministic round-half-up (np.round uses banker's rounding: np.round(2.5)==2). Sign-aware."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# --------------------------------------------------------------------------- core maths


def covariance_matrix(sigma, corr) -> np.ndarray:
    """Σ = σᵀ·ρ·σ, i.e. Σᵢⱼ = σᵢ·ρᵢⱼ·σⱼ. `sigma` annualised stdevs (1-D), `corr` correlation (N×N)."""
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    return (sigma[:, None] * corr) * sigma[None, :]


def tracking_error_std(weights, optimal_weights, cov) -> float:
    """Standard deviation of the tracking-error portfolio, √(eᵀΣe), with e = weights − optimal."""
    e = np.asarray(weights, dtype=float) - np.asarray(optimal_weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return float(np.sqrt(max(float(e @ cov @ e), 0.0)))


def _cost_in_weight_terms(cost_per_contract, weight_per_contract, capital) -> np.ndarray:
    """wᶜᵢ = (Cᵢ / capital) / cᵢ — the cost of trading one contract expressed as a portfolio weight."""
    c = np.asarray(cost_per_contract, dtype=float)
    wpc = np.asarray(weight_per_contract, dtype=float)
    return (c / capital) / wpc


def _objective(
    candidate_w: np.ndarray,
    target_w: np.ndarray,
    cov: np.ndarray,
    *,
    prev_w: np.ndarray | None,
    cost_weight: np.ndarray | None,
    cost_multiplier: float,
) -> float:
    """√(eᵀΣe)  (+ cost_multiplier·δ if costs supplied)."""
    te = tracking_error_std(candidate_w, target_w, cov)
    if cost_weight is not None and prev_w is not None:
        delta = np.abs(prev_w - candidate_w)
        te += cost_multiplier * float(np.sum(delta * cost_weight))
    return te


# --------------------------------------------------------------------------- greedy optimiser


def greedy_optimise(
    optimal_weights,
    weight_per_contract,
    cov,
    *,
    prev_positions=None,
    cost_per_contract=None,
    capital: float | None = None,
    cost_multiplier: float = COST_MULTIPLIER,
    tradeable: np.ndarray | None = None,
    position_limits=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy tracking-error optimiser. Returns (optimised_positions_int, optimised_weights).

    Parameters
    ----------
    optimal_weights        : ideal unrounded portfolio weights wᵢ = Nᵢ·cᵢ.
    weight_per_contract    : cᵢ (one contract as a fraction of capital).
    cov                    : covariance matrix Σ (use `covariance_matrix`).
    prev_positions         : currently-held integer contracts Pᵢ (for the cost penalty). Default 0.
    cost_per_contract      : Cᵢ in capital currency; enables the cost penalty when given with capital.
    tradeable              : bool mask — instruments we may HOLD (you can still forecast on others
                             to transfer risk to correlated tradeable markets). Default: all True.
    position_limits        : optional list of (min_contracts, max_contracts) per instrument.
    """
    target_w = np.asarray(optimal_weights, dtype=float)
    wpc = np.asarray(weight_per_contract, dtype=float)
    n = target_w.size
    cov = np.asarray(cov, dtype=float)

    direction = np.sign(target_w)  # only ever move toward the forecast's sign (never hedge against it)
    prev_positions = np.zeros(n, dtype=int) if prev_positions is None else np.asarray(prev_positions, int)

    cost_weight = None
    prev_w = None
    if cost_per_contract is not None:
        if capital is None:
            raise ValueError("capital is required when cost_per_contract is supplied")
        cost_weight = _cost_in_weight_terms(cost_per_contract, wpc, capital)
        prev_w = prev_positions * wpc

    if tradeable is None:
        tradeable = np.ones(n, dtype=bool)
    else:
        tradeable = np.asarray(tradeable, dtype=bool)

    def within_limits(contracts: float, i: int) -> bool:
        if position_limits is None:
            return True
        lo, hi = position_limits[i]
        return lo <= int(_round_half_up(contracts)) <= hi

    x = np.zeros(n, dtype=float)  # candidate weights — Carver starts from zero
    best = _objective(x, target_w, cov, prev_w=prev_w, cost_weight=cost_weight, cost_multiplier=cost_multiplier)

    for _ in range(_MAX_ITERS):
        best_move_obj = best
        best_move_x: np.ndarray | None = None
        for i in range(n):
            if direction[i] == 0 or not tradeable[i]:
                continue
            cand = x.copy()
            cand[i] += direction[i] * wpc[i]
            if not within_limits(cand[i] / wpc[i], i):
                continue
            obj = _objective(
                cand, target_w, cov, prev_w=prev_w, cost_weight=cost_weight, cost_multiplier=cost_multiplier
            )
            if obj < best_move_obj - 1e-15:
                best_move_obj, best_move_x = obj, cand
        if best_move_x is None:  # no single one-contract move improves the objective → optimal
            break
        x, best = best_move_x, best_move_obj

    # x is an exact integer multiple of wpc by construction; _round_half_up just guards round-off.
    positions = _round_half_up(x / wpc).astype(int)
    return positions, x


# --------------------------------------------------------------------------- buffering


def apply_buffer(
    optimised_positions,
    prev_positions,
    weight_per_contract,
    cov,
    *,
    risk_target: float = DEFAULT_RISK_TARGET,
    buffer_fraction: float = BUFFER_FRACTION,
    position_limits=None,
) -> np.ndarray:
    """Tracking-error buffering on the whole portfolio. Returns the new integer positions to hold.

    No trade if the current portfolio is already within Bσ = buffer_fraction·risk_target of the
    optimised portfolio; otherwise trade a fraction α = max((T−Bσ)/T, 0) of the way there
    (round-half-up). The result is always clamped into `position_limits` (list of (lo, hi) per
    instrument) when given, so a tightened limit can never leave us holding outside it.
    """
    opt = np.asarray(optimised_positions, dtype=int)
    prev = np.asarray(prev_positions, dtype=int)
    wpc = np.asarray(weight_per_contract, dtype=float)
    cov = np.asarray(cov, dtype=float)

    ep = (opt * wpc) - (prev * wpc)  # tracking error of current vs optimised (no cost penalty here)
    tracking = float(np.sqrt(max(float(ep @ cov @ ep), 0.0)))
    buffer = buffer_fraction * risk_target

    if tracking <= buffer or tracking == 0.0:
        new = prev.copy()
    else:
        alpha = max((tracking - buffer) / tracking, 0.0)
        required = _round_half_up(alpha * (opt - prev)).astype(int)
        new = prev + required

    if position_limits is not None:
        for i, (lo, hi) in enumerate(position_limits):
            new[i] = min(max(int(new[i]), lo), hi)
    return new


# --------------------------------------------------------------------------- high-level helper


@dataclass(frozen=True)
class Instrument:
    """One row of the universe fed to the optimiser."""

    ticker: str
    optimal_position: float  # Nᵢ — ideal unrounded contracts (forecast × instr weight × IDM × capital / risk)
    weight_per_contract: float  # cᵢ — notional per contract / capital
    sigma: float  # annualised stdev of % returns
    cost_per_contract: float = 0.0  # Cᵢ — currency cost of trading one contract
    prev_position: int = 0  # Pᵢ — currently held contracts
    tradeable: bool = True  # may we hold this one? (forecast-only instruments set False)
    min_contracts: int | None = None  # position limit (risk control: VIX cap, holiday cut, force-roll)
    max_contracts: int | None = None


@dataclass(frozen=True)
class DynOptResult:
    tickers: list[str]
    optimised_positions: dict[str, int]  # before buffering
    buffered_positions: dict[str, int]  # after buffering vs prev (what to actually hold)
    candidates: list[str]  # tickers with a non-zero buffered position — the dynamic candidate list
    optimal_weights: dict[str, float]
    tracking_error: float  # √(eᵀΣe) of the optimised vs ideal portfolio (std only, no cost)


def optimise_portfolio(
    instruments: list[Instrument],
    corr,
    capital: float,
    *,
    use_costs: bool = True,
    cost_multiplier: float = COST_MULTIPLIER,
    apply_buffering: bool = True,
    risk_target: float = DEFAULT_RISK_TARGET,
    buffer_fraction: float = BUFFER_FRACTION,
) -> DynOptResult:
    """Run the full Strategy-25 pipeline on a universe and return the dynamic candidate list.

    `corr` is the N×N correlation matrix aligned with `instruments` order.
    """
    tickers = [ins.ticker for ins in instruments]
    wpc = np.array([ins.weight_per_contract for ins in instruments], dtype=float)
    sigma = np.array([ins.sigma for ins in instruments], dtype=float)
    target_w = np.array([ins.optimal_position * ins.weight_per_contract for ins in instruments], dtype=float)
    prev_pos = np.array([ins.prev_position for ins in instruments], dtype=int)
    tradeable = np.array([ins.tradeable for ins in instruments], dtype=bool)
    cov = covariance_matrix(sigma, corr)

    cost_per_contract = None
    if use_costs and any(ins.cost_per_contract for ins in instruments):
        cost_per_contract = np.array([ins.cost_per_contract for ins in instruments], dtype=float)

    has_limits = any(ins.min_contracts is not None or ins.max_contracts is not None for ins in instruments)
    position_limits = None
    if has_limits:
        position_limits = [
            (
                ins.min_contracts if ins.min_contracts is not None else -_NO_LIMIT,
                ins.max_contracts if ins.max_contracts is not None else _NO_LIMIT,
            )
            for ins in instruments
        ]

    opt_pos, _opt_w = greedy_optimise(
        target_w,
        wpc,
        cov,
        prev_positions=prev_pos,
        cost_per_contract=cost_per_contract,
        capital=capital,
        cost_multiplier=cost_multiplier,
        tradeable=tradeable,
        position_limits=position_limits,
    )

    final_pos = apply_buffer(
        opt_pos, prev_pos, wpc, cov, risk_target=risk_target, buffer_fraction=buffer_fraction,
        position_limits=position_limits,
    ) if apply_buffering else opt_pos

    e = (opt_pos * wpc) - target_w
    te = float(np.sqrt(max(float(e @ cov @ e), 0.0)))

    return DynOptResult(
        tickers=tickers,
        optimised_positions={t: int(p) for t, p in zip(tickers, opt_pos)},
        buffered_positions={t: int(p) for t, p in zip(tickers, final_pos)},
        candidates=[t for t, p in zip(tickers, final_pos) if p != 0],
        optimal_weights={t: float(w) for t, w in zip(tickers, target_w)},
        tracking_error=te,
    )


def select_candidates(instruments: list[Instrument], corr, capital: float, **kwargs) -> list[str]:
    """Convenience: return only the dynamically-optimised candidate tickers (non-zero positions)."""
    return optimise_portfolio(instruments, corr, capital, **kwargs).candidates
