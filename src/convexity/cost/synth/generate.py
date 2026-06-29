"""Synthetic adversarial cost scenarios (COST §13).

No external data is required to run COST's tests — every component and gate is
exercised by a seeded synthetic generator that builds ``CostMarketState`` (and the
funding-history arrays) under the §13 adversarial overlays: calm funding, a spike
beyond the cap, a regime flip, published-next divergence, m-sensitivity, future-
volume garbage, commodity proxy financing, the band-binding marginal trade, a thin
residual series, the interest-unit double-discount, carry time-scaling, spread
blowout, missing/stale inputs, roll gaps, MTF accrual, and a short earning funding.

The generator is deterministic given its seed/knobs (so fixtures are reproducible)
and returns plain ``CostMarketState`` objects the components consume directly.
"""

from __future__ import annotations

import numpy as np

from convexity.cost.types import CarryClass, CostMarketState, InterestRateUnit

_8H_MS = 8 * 60 * 60 * 1000


def _funding_times(n: int, end_ts: int = 0) -> np.ndarray:
    """``n`` ascending 8h-spaced stamp timestamps ending at ``end_ts`` (epoch-ms)."""
    return np.array([end_ts - (n - 1 - i) * _8H_MS for i in range(n)], dtype=np.int64)


def calm_market(
    *,
    notional_mid: float = 50000.0,
    n_funding: int = 30,
    funding_level: float = 0.00012,
    seed: int = 0,
) -> CostMarketState:
    """Small stable funding; fees/spread dominate the gate (§13 calm)."""
    rng = np.random.default_rng(seed)
    rates = funding_level + rng.normal(0.0, 1e-5, size=n_funding)
    return CostMarketState(
        mid=notional_mid,
        bid=notional_mid - 5.0,
        ask=notional_mid + 5.0,
        perp_mark=notional_mid + 10.0,
        index_price=notional_mid,
        funding_cap=0.0075,
        interest_quote=0.0000333,
        interest_base=0.0,
        interest_rate_unit="per_interval",
        premium_index_8h=0.0002,
        published_next_funding=0.00015,
        recent_funding_rates=rates,
        recent_funding_times=_funding_times(n_funding),
        n_funding_events=n_funding,
        sigma=0.004,
        v_hat_exit_window=200000.0,
        rolling_30d_volume=1_000_000.0,
        carry_class="crypto_perp",
        impact_k=0.10,
        impact_k_units="fraction",
        residual_e_q_ci_edge=0.00008,
        n_residual_events=30,
        spread_regime="calm",
    )


def worked_example_market() -> CostMarketState:
    """Reproduce the §5.7 worked-example market state bit-for-bit.

    Long crypto_perp, N=1000, taker/taker VIP-0, mid 50000, k_i=0.10, sigma=0.004,
    v_hat=200000, premium_8h=0.0002, interest_8h=0.0000333 (per_interval), lambda=0.30,
    cap=0.0075, published_next=0.00015, e_q=0.00008, r_ewma=0.00012 (one stamp at that
    level reproduces the recursion target exactly with the same level history).
    """
    # A flat funding history at 0.00012 gives r_ewma -> 0.00012 exactly (§5.7 input).
    rates = np.full(30, 0.00012, dtype=float)
    return CostMarketState(
        mid=50000.0,
        bid=49995.0,
        ask=50005.0,
        perp_mark=50010.0,
        index_price=50000.0,
        funding_cap=0.0075,
        interest_quote=0.0000333,
        interest_base=0.0,
        interest_rate_unit="per_interval",
        premium_index_8h=0.0002,
        published_next_funding=0.00015,
        recent_funding_rates=rates,
        recent_funding_times=_funding_times(30),
        n_funding_events=30,
        sigma=0.004,
        v_hat_exit_window=200000.0,
        rolling_30d_volume=1_000_000.0,
        carry_class="crypto_perp",
        impact_k=0.10,
        impact_k_units="fraction",
        residual_e_q_ci_edge=0.00008,
        n_residual_events=30,
        spread_regime="calm",
    )


def funding_spike_market(*, seed: int = 1) -> CostMarketState:
    """Funding jumps an order of magnitude; the EWMA/basis blend would exceed the cap
    and the clamp binds (§13 spike)."""
    m = calm_market(seed=seed)
    rates = m.recent_funding_rates.copy()
    rates[-3:] = 0.05  # an order of magnitude beyond the 0.0075 cap
    return _replace(m, recent_funding_rates=rates, premium_index_8h=0.06)


def regime_flip_market(*, seed: int = 2) -> CostMarketState:
    """Sign flip: longs paid -> longs paying (§13 regime flip)."""
    m = calm_market(seed=seed)
    n = len(m.recent_funding_rates)
    rates = m.recent_funding_rates.copy()
    rates[: n // 2] = -0.0003
    rates[n // 2 :] = +0.0003
    return _replace(m, recent_funding_rates=rates, published_next_funding=0.0003)


def published_divergence_market(*, seed: int = 3) -> CostMarketState:
    """Venue published-next diverges sharply from the EWMA (§13 published divergence)."""
    m = calm_market(seed=seed)
    return _replace(m, published_next_funding=0.0050)


def thin_residual_market(*, seed: int = 4) -> CostMarketState:
    """A 3-residual funding series -> the band must fall back to the class worst (§13 thin)."""
    m = calm_market(n_funding=3, seed=seed)
    return _replace(m, n_residual_events=3, residual_e_q_ci_edge=None)


def stale_market(*, drop: str = "spread", seed: int = 5) -> CostMarketState:
    """Drop bid/ask, funding history, or impact data; conservative fallback fires (§13 stale)."""
    m = calm_market(seed=seed)
    if drop == "spread":
        return _replace(m, bid=None, ask=None, stale_flags=("spread",))
    if drop == "funding":
        return _replace(
            m,
            recent_funding_rates=np.array([], dtype=float),
            recent_funding_times=np.array([], dtype=np.int64),
            n_funding_events=0,
            published_next_funding=None,
            stale_flags=("funding",),
        )
    if drop == "impact":
        return _replace(m, impact_k=None, stale_flags=("impact",))
    raise ValueError(f"unknown drop {drop!r}")


def commodity_market(
    *, financing_mode: str = "unlevered", basis_observable: bool = False, seed: int = 6
) -> CostMarketState:
    """Dukascopy NatGas/Crude proxy under unlevered vs MTF financing (§13 commodity)."""
    m = calm_market(seed=seed)
    return _replace(
        m,
        carry_class="commodity",
        storage_rate=0.50,  # USD per unit per YEAR
        price=3.0,
        commodity_financing_mode=financing_mode,  # type: ignore[arg-type]
        commodity_basis_observable=basis_observable,
        futures_basis=0.02 if basis_observable else None,
        tau_spread_days=30.0,
        mtf_rate=0.12 if financing_mode == "mtf" else None,
        borrowed_notional=500.0 if financing_mode == "mtf" else None,
    )


def dated_future_market(*, roll_in_hold: bool = False, seed: int = 7) -> CostMarketState:
    """Dated future with an inter-contract basis, optional roll-gap (§13 roll gap)."""
    m = calm_market(seed=seed)
    return _replace(
        m,
        carry_class="dated_future",
        futures_basis=0.01,  # (F_far - F_near)/F_near over tau_spread
        tau_spread_days=90.0,
        roll_gap_quote=0.25 if roll_in_hold else None,
    )


def equity_mtf_market(*, seed: int = 8) -> CostMarketState:
    """US equity on MTF financing; the borrowed notional carries simple interest (§13 MTF)."""
    m = calm_market(seed=seed)
    return _replace(
        m,
        carry_class="equity_mtf",
        mtf_rate=0.12,
        borrowed_notional=500.0,
        mtf_compounding=False,
    )


def interest_unit_pair(*, seed: int = 9) -> tuple[CostMarketState, CostMarketState]:
    """The SAME venue numbers fed once as per_interval and once as per_day (§13 double-discount)."""
    base = calm_market(seed=seed)
    per_interval = _replace(base, interest_rate_unit="per_interval", interest_quote=0.0000333)
    per_day = _replace(base, interest_rate_unit="per_day", interest_quote=0.0001)
    return per_interval, per_day


def with_interest_unit(m: CostMarketState, unit: InterestRateUnit) -> CostMarketState:
    return _replace(m, interest_rate_unit=unit)


def with_carry_class(m: CostMarketState, carry_class: CarryClass) -> CostMarketState:
    return _replace(m, carry_class=carry_class)


def _replace(m: CostMarketState, **changes) -> CostMarketState:
    """``dataclasses.replace`` helper kept local so callers import from this module."""
    import dataclasses

    return dataclasses.replace(m, **changes)
