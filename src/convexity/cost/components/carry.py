"""CarryModel — ONE interface, per-class body keyed by AMBER (COST §5.6, Inv X/P/Y-finance/H-time).

The funding/carry concept generalizes across asset classes through a single
``carry`` method whose class branch lives INSIDE the model, selected by the AMBER
``carry_class`` (Invariant X — no consumer ever forks on asset class). All carry is
SIGNED relative to the position (P7) and CAUSAL.

| Class         | Mechanism                | Signed carry (over hold)                                       |
|---------------|--------------------------|----------------------------------------------------------------|
| crypto_perp   | 8h funding               | side·N·Σ r_hat over m stamps (routes to FundingForecast, §5.5) |
| dated_future  | inter-contract basis     | side·N·(F_far-F_near)/F_near·(Δt/τ_spread) + roll-gap (§5.6-time)|
| equity_mtf    | MTF financing            | +|borrowed|·mtf_rate·(Δt_days/365)  POSITIVE = cost (P7)        |
| commodity     | storage(+basis/MTF)      | financing booked ONCE (§5.6-commodity, Invariant Y-finance)    |

Time conventions are pinned per §5.6-time: inter-contract basis scales by
``Δt/τ_spread``; MTF and storage use Actual/365 SIMPLE accrual (compounding only via
explicit AMBER ``mtf_compounding``); perp uses the stamp clock. MTF financing is
booked POSITIVE = cost so it TIGHTENS LEVERAGE's stop ceiling (the clamp lives in
LEVERAGE, ARCHITECTURE §5.5).
"""

from __future__ import annotations

from convexity.core.types import HoldHorizon

from convexity.cost.components.funding_forecast import FundingForecast
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostMarketState, Side

_MS_PER_DAY = 24 * 60 * 60 * 1000


class CarryModel:
    """Cross-asset carry. ``carry`` returns (signed_quote_ccy, diagnostics) (§5.6)."""

    def __init__(self, funding_forecast: FundingForecast | None = None) -> None:
        self._funding = funding_forecast if funding_forecast is not None else FundingForecast()

    def carry(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
        *,
        use_band: bool = True,
    ) -> tuple[float, dict]:
        """Signed forecast carry over ``horizon`` (§5.6). The class branch is internal (Inv X).

        For crypto_perp, ``use_band`` selects the side-selected conservative band (the
        gate/hold/carry-term truth, Invariant T) vs the point forecast (telemetry).
        """
        cls = market.carry_class
        if cls == "crypto_perp":
            return self._carry_perp(side, notional, horizon, market, cfg, use_band=use_band)
        if cls == "dated_future":
            return self._carry_dated_future(side, notional, horizon, market, cfg)
        if cls == "equity_mtf":
            return self._carry_equity_mtf(notional, horizon, market, cfg)
        if cls == "commodity":
            return self._carry_commodity(side, notional, horizon, market, cfg)
        raise ValueError(f"unknown carry_class {cls!r}")

    # ---------------------------------------------------------------- crypto perp (§5.5)

    def _carry_perp(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
        *,
        use_band: bool,
    ) -> tuple[float, dict]:
        m = horizon.n_funding_stamps_remaining  # the conservative m_max (Inv M-hold)
        res = self._funding.forecast(side, notional, m, market, cfg)
        value = res.funding_band_conservative if use_band else res.f_forecast
        diag = {
            "carry_class": "crypto_perp",
            "m": float(m),
            "funding_forecast_point": res.f_forecast,
            "funding_band_low": res.funding_band_low,
            "funding_band_high": res.funding_band_high,
            "funding_band_conservative": res.funding_band_conservative,
            "funding_sensitivity": res.funding_sensitivity,
            "reason": res.reason,
        }
        return value, diag

    # ---------------------------------------------------------------- dated future (§5.6-time)

    def _hold_days(self, horizon: HoldHorizon) -> float:
        return horizon.max_hold_quantile_ms / _MS_PER_DAY

    def _carry_dated_future(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> tuple[float, dict]:
        """``side·N·(F_far-F_near)/F_near·(Δt/τ_spread)`` + roll-gap (§5.6-time).

        The inter-contract basis is TIME-SCALED by ``Δt/τ_spread`` (NOT the raw spread):
        at ``Δt = τ_spread`` the full inter-contract carry is booked; ``Δt -> 0`` -> 0.
        Contango (positive basis) is a cost to a long. The roll-gap is a DISCRETE event
        booked when a roll occurs in-hold, not scaled by ``Δt``.
        """
        dt_days = self._hold_days(horizon)
        basis = market.futures_basis
        tau = market.tau_spread_days
        if basis is None or tau is None or tau <= 0.0:
            # P6: missing basis -> conservative zero-basis is NOT used; fall back to a
            # storage-style positive cost is not defined for a pure future, so we book a
            # zero carry but flag it for the conservative fallback to be visible.
            value = 0.0
            reason = "fallback_no_basis"
        else:
            value = side * notional * basis * (dt_days / tau)
            reason = "inter_contract_basis_time_scaled"
        roll_gap = market.roll_gap_quote if market.roll_gap_quote is not None else 0.0
        value += roll_gap  # discrete, not Δt-scaled
        diag = {
            "carry_class": "dated_future",
            "dt_days": dt_days,
            "tau_spread_days": tau if tau is not None else float("nan"),
            "basis_carry": value - roll_gap,
            "roll_gap": roll_gap,
            "reason": reason,
        }
        return value, diag

    # ---------------------------------------------------------------- equity MTF (§5.6-time)

    def _carry_equity_mtf(
        self,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> tuple[float, dict]:
        """``+|borrowed|·mtf_rate·(Δt_days/365)`` SIMPLE Actual/365, POSITIVE = cost (P7).

        Side-symmetric (``|borrowed_notional|`` whether long or short), so Invariant P's
        long/short antisymmetry does NOT apply here (the dedicated sign test instead
        asserts the charge is POSITIVE and tightens ``L_stop``). Compounding only via the
        explicit AMBER ``mtf_compounding`` flag (§5.6-time).
        """
        dt_days = self._hold_days(horizon)
        borrowed = abs(market.borrowed_notional) if market.borrowed_notional is not None else 0.0
        rate = market.mtf_rate if market.mtf_rate is not None else 0.0
        if market.mtf_compounding:
            value = borrowed * ((1.0 + rate / 365.0) ** dt_days - 1.0)
            reason = "mtf_compound"
        else:
            value = borrowed * rate * (dt_days / 365.0)  # SIMPLE
            reason = "mtf_simple_actual365"
        diag = {
            "carry_class": "equity_mtf",
            "dt_days": dt_days,
            "borrowed_notional": borrowed,
            "mtf_rate": rate,
            "reason": reason,
        }
        return value, diag  # POSITIVE = cost (P7)

    # ---------------------------------------------------------------- commodity (§5.6-commodity)

    def _carry_commodity(
        self,
        side: Side,
        notional: float,
        horizon: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> tuple[float, dict]:
        """Commodity carry — financing booked exactly ONCE (Invariant Y-finance, §5.6-commodity).

        Observed basis (when ``commodity_basis_observable``) embeds storage+financing
        −convenience by no-arbitrage -> the SAME time-scaled basis route as dated
        futures, NO separate convenience term. Otherwise storage-only (convenience = 0,
        ``fallback_no_convenience``). Financing is NEVER in the commodity formula: an
        ``unlevered`` proxy pays none; a ``mtf`` proxy pays it via the MTF path (added
        separately by the caller / cost book), and the commodity line carries ONLY storage.
        """
        dt_days = self._hold_days(horizon)
        if market.commodity_basis_observable and market.futures_basis is not None:
            tau = market.tau_spread_days
            if tau is None or tau <= 0.0:
                value, reason = 0.0, "fallback_no_basis"
            else:
                value = side * notional * market.futures_basis * (dt_days / tau)
                reason = "observed_basis_time_scaled"
            diag = {
                "carry_class": "commodity",
                "dt_days": dt_days,
                "storage_carry": 0.0,
                "mtf_financing": 0.0,
                "reason": reason,
            }
            return value, diag

        # Storage-only, convenience = 0 (P6). storage_rate is quote-ccy per unit per YEAR,
        # so (storage_rate/price) is an annual fraction; *(Δt_days/365) over the hold.
        storage_rate = market.storage_rate if market.storage_rate is not None else 0.0
        price = market.price if (market.price is not None and market.price > 0.0) else market.mid
        storage_carry = side * notional * (storage_rate / price) * (dt_days / 365.0)

        # Financing booked ONCE: zero for unlevered, via MTF for a financed proxy.
        mtf_financing = 0.0
        if market.commodity_financing_mode == "mtf":
            borrowed = (
                abs(market.borrowed_notional) if market.borrowed_notional is not None else 0.0
            )
            rate = market.mtf_rate if market.mtf_rate is not None else 0.0
            mtf_financing = borrowed * rate * (dt_days / 365.0)  # POSITIVE = cost (P7)

        value = storage_carry + mtf_financing
        diag = {
            "carry_class": "commodity",
            "dt_days": dt_days,
            "storage_carry": storage_carry,
            "mtf_financing": mtf_financing,
            "reason": "storage_only_fallback_no_convenience",
        }
        return value, diag
