"""FundingForecast — causal, no-ML remaining-hold funding forecast (COST §5.5, P4).

Over [now, t_exit] COST forecasts the funding it has not yet paid, using ONLY
``F_t``-measurable inputs (Invariant C/F): the recent realized funding path
(EWMA, §5.5a), the basis-implied anchor (§5.5b), and the venue-published next rate
as a ``j == 1`` OVERRIDE (§5.5). The forecast is transparent, monotone, decomposable
— never a trained model.

Per-stamp rate forecast (piecewise, NOT a sum of three terms — §5.5):

    r_hat(τ_j) = clamp(r_published_next, -cap, +cap)            if j == 1 and published F_t-known
               = clamp((1-λ)·r_ewma + λ·r_basis, -cap, +cap)    otherwise

The published rate REPLACES the blend for the first stamp; it is NOT added on top
(that would inflate every stamp — Invariant E). Every ``r_hat`` is clamped to the
venue cap (realized funding cannot exceed it).

The gate's binding funding line is the side-selected conservative comonotone BAND
``F_band_conservative`` (§5.5c) at its CI edge, NOT the point ``F_forecast``
(Invariant F-band-gate / P-band / G-est). Funding is priced over a RANGE of holds
to the conservative ``m_max`` (§5.5d); ``dF/dm`` is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import CostMarketState, Side


@dataclass(frozen=True)
class FundingForecastResult:
    """The full funding forecast at ``m`` stamps plus the band and sensitivity (§5.5)."""

    r_hat: tuple[float, ...]  # per-stamp rate forecast, length m
    f_forecast: float  # POINT blend Σ side·N·r_hat — TELEMETRY, not the gate summand
    funding_band_low: float  # raw -e_q comonotone sum over m stamps (§5.5c)
    funding_band_high: float  # raw +e_q comonotone sum over m stamps (§5.5c)
    funding_band_conservative: float  # side-selected, CI-edge band — the GATE summand
    funding_sensitivity: float  # dF/dm at m (§5.5d)
    r_ewma: float
    r_basis: float | None
    reason: str


def _clamp(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


class FundingForecast:
    """Causal piecewise funding forecast + comonotone band (§5.5)."""

    # ---------------------------------------------------------------- §5.5a EWMA

    def ewma(self, rates: np.ndarray, cfg: ResolvedCostConfig) -> float:
        """``r_ewma`` via the O(1) streaming recursion over realized stamps (§5.5a).

        ``r_ewma_k = α·r_k + (1-α)·r_ewma_{k-1}``, ``α = 1 - 2^(-1/h)``. The recursion
        runs over the funding-stamp clock (not calendar time). Seeded with the first rate.
        """
        if rates.size == 0:
            return 0.0
        alpha = cfg.defaults.funding_ewma_alpha
        r = float(rates[0])
        for k in range(1, rates.size):
            r = alpha * float(rates[k]) + (1.0 - alpha) * r
        return r

    # ---------------------------------------------------------------- §5.5b basis anchor

    def interest_8h(self, market: CostMarketState) -> float:
        """``interest_8h`` with the (8/24) scaling CONDITIONAL on the AMBER unit (§5.5b).

        ``per_day`` inputs are scaled by (8/24); ``per_interval`` (venue-native) are
        used as-is. No silent default — the unit is a required AMBER field, so a wrong
        scaling cannot creep in (the ~3× double-discount this closes).
        """
        diff = market.interest_quote - market.interest_base
        if market.interest_rate_unit == "per_day":
            return diff * (8.0 / 24.0)
        return diff  # per_interval: venue-native, NO rescale

    def basis_anchor(self, market: CostMarketState) -> tuple[float | None, str]:
        """``r_basis = clamp(interest_8h + premium_8h, -cap, +cap)`` (§5.5b).

        The premium is taken from the venue-published premium index where available
        (``premium_index_8h``); the instantaneous ``(perp_mark-index)/index`` snapshot
        is a flagged FALLBACK (``premium_from_snapshot``). If neither is available the
        basis is undefined and the blend degrades to EWMA-only (``fallback_no_basis``).
        """
        if market.premium_index_8h is not None:
            premium = market.premium_index_8h
            reason = "basis_published_premium"
        elif (
            market.perp_mark is not None
            and market.index_price is not None
            and market.index_price != 0.0
        ):
            premium = (market.perp_mark - market.index_price) / market.index_price
            reason = "premium_from_snapshot"
        else:
            return None, "fallback_no_basis"
        r_basis = _clamp(self.interest_8h(market) + premium, market.funding_cap)
        return r_basis, reason

    # ---------------------------------------------------------------- §5.5 per-stamp r_hat

    def per_stamp_rates(
        self, market: CostMarketState, m: int, cfg: ResolvedCostConfig
    ) -> tuple[tuple[float, ...], float, float | None, str]:
        """The piecewise ``r_hat(τ_j)`` for ``j = 1..m`` (§5.5).

        Returns (r_hat tuple, r_ewma, r_basis, reason). Applies the event-count floor
        (§5.5a): below ``n_funding_min`` the EWMA falls back in priority order —
        published-flat, then last-funding-flat, then class-worst (P6).
        """
        cap = market.funding_cap
        rates = np.asarray(market.recent_funding_rates, dtype=float)
        published = market.published_next_funding
        use_published_j1 = market.pre_publishes_funding and published is not None

        # --- event-count floor (§5.5a): the EWMA is trusted only above n_funding_min.
        if market.n_funding_events >= cfg.defaults.n_funding_min and rates.size > 0:
            r_ewma = self.ewma(rates, cfg)
            r_basis, basis_reason = self.basis_anchor(market)
            if r_basis is None:
                blend = r_ewma  # EWMA-only degrade (§5.5b)
                reason = "ewma_only_no_basis"
            else:
                lam = cfg.defaults.basis_shrinkage_lambda
                blend = (1.0 - lam) * r_ewma + lam * r_basis
                reason = f"ewma_basis_blend:{basis_reason}"
            blend_clamped = _clamp(blend, cap)
            r_hat = []
            for j in range(1, m + 1):
                if j == 1 and use_published_j1:
                    r_hat.append(_clamp(published, cap))  # OVERRIDE, not added on top
                else:
                    r_hat.append(blend_clamped)
            return tuple(r_hat), r_ewma, r_basis, reason

        # --- below the floor: fall back (P6) in priority order.
        if use_published_j1:
            flat = _clamp(published, cap)
            return tuple([flat] * m), flat, None, "fallback_published_flat"
        if rates.size > 0:
            flat = _clamp(float(rates[-1]), cap)
            return tuple([flat] * m), flat, None, "fallback_last_funding"
        flat = _clamp(cfg.preset.class_worst_funding_rate, cap)
        return tuple([flat] * m), flat, None, "fallback_class_worst"

    # ---------------------------------------------------------------- §5.5c comonotone band

    def _e_q(self, market: CostMarketState, cfg: ResolvedCostConfig) -> tuple[float, str]:
        """The CI-edge one-step residual quantile ``e_q`` (magnitude, §5.5c).

        Below ``n_residual_min`` residuals (or when no CI-edge value is supplied) the
        band FALLS BACK to the class-level worst residual (``fallback_band_thin``, P6).
        Otherwise the EXECUTION/validation-supplied CI-edge value is binding.
        """
        if (
            market.residual_e_q_ci_edge is None
            or market.n_residual_events < cfg.defaults.n_residual_min
        ):
            return abs(cfg.preset.class_worst_residual_e_q), "fallback_band_thin"
        return abs(market.residual_e_q_ci_edge), "band_ci_edge"

    def forecast(
        self,
        side: Side,
        notional: float,
        m: int,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> FundingForecastResult:
        """Full forecast at horizon ``m`` stamps (§5.5).

        Notional is held FLAT to ``m`` (causal worst-case for the position, §5.5d — no
        TREND reduce schedule on the gate path, Invariant W). The point blend, the raw
        comonotone band ends, the side-selected conservative band (the GATE summand),
        and ``dF/dm`` are all returned.
        """
        r_hat, r_ewma, r_basis, reason = self.per_stamp_rates(market, m, cfg)
        cap = market.funding_cap

        # POINT blend (telemetry, not the gate summand).
        f_forecast = sum(side * notional * r for r in r_hat)

        # Comonotone worst-case sums: the SAME shock e_q applied to every stamp, re-clamped.
        e_q, band_reason = self._e_q(market, cfg)
        band_high = sum(side * notional * _clamp(r + e_q, cap) for r in r_hat)  # +e_q tail
        band_low = sum(side * notional * _clamp(r - e_q, cap) for r in r_hat)  # -e_q tail

        # Side-selected conservative band: the endpoint MAXIMIZING the position's funding
        # COST (§5.5c, Invariant P-band). ``max`` over the two signed sums picks the
        # worse-for-position tail regardless of side (a long's +e_q tail, a short's -e_q).
        band_conservative = max(band_high, band_low)

        # dF/dm: marginal carry of one more stamp (§5.5d).
        df_dm = side * notional * r_hat[-1] if r_hat else 0.0

        full_reason = reason if band_reason == "band_ci_edge" else f"{reason};{band_reason}"
        return FundingForecastResult(
            r_hat=r_hat,
            f_forecast=f_forecast,
            funding_band_low=band_low,
            funding_band_high=band_high,
            funding_band_conservative=band_conservative,
            funding_sensitivity=df_dm,
            r_ewma=r_ewma,
            r_basis=r_basis,
            reason=full_reason,
        )
