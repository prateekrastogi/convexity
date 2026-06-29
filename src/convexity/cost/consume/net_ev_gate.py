"""NetEVGate — the net-of-cost expected-value gate handed to BREAKOUT (COST §8.1, P2).

Take-the-trade-only-if gate. Returns a ``NetEVDecision`` (numbers, never an action —
Invariant D):

    passes = expected_favorable > C_round_trip + edge_buffer

where ``C_round_trip`` prices funding at the side-selected CONSERVATIVE BAND
``F_band_conservative(m_max, q)`` — never the point ``F_forecast`` (Invariant
F-band-gate / P-band / G-est) — over the conservative ``m_max`` stamp count (§5.5d,
Invariant M-hold), with fees priced at the conservative tier near a VIP breakpoint
(§7, Invariant J-tier) and impact on the CAUSAL ``v_hat`` (§5.4, Invariant W).
``binding_funding == round_trip.funding_band_conservative`` (Gap 1). The decision
carries ``funding_sensitivity`` (dF/dm) and ``funding_m_sensitive``. BREAKOUT
supplies ``expected_favorable`` (an EDGE quantity owned upstream — COST never
estimates edge). This method NEVER opens a trade.
"""

from __future__ import annotations

import dataclasses

from convexity.core.types import HoldHorizon

from convexity.cost.assemble.cost_book import CostBook
from convexity.cost.config.resolve import ResolvedCostConfig
from convexity.cost.types import (
    CostMarketState,
    LiquidityFlag,
    NetEVDecision,
    Side,
)


class NetEVGate:
    """BREAKOUT entry gate (§8.1). One public method, thin transform over the CostBook."""

    def __init__(self, cost_book: CostBook | None = None) -> None:
        self.cost_book = cost_book if cost_book is not None else CostBook()

    def evaluate(
        self,
        side: Side,
        notional: float,
        expected_favorable: float,
        liquidity_flag_in: LiquidityFlag,
        liquidity_flag_out: LiquidityFlag,
        expected_hold: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
    ) -> NetEVDecision:
        """Evaluate the gate at the conservative ``m_max`` and the conservative band (§8.1)."""
        edge_buffer = cfg.defaults.edge_buffer

        rt = self.cost_book.round_trip(
            side,
            notional,
            liquidity_flag_in,
            liquidity_flag_out,
            expected_hold,
            market,
            cfg,
            expected_favorable=expected_favorable,
            gate_fee_tier=True,
        )
        binding_funding = rt.funding_band_conservative  # the band, not the point (Gap 1)
        total = rt.total_quote_ccy  # already sums the band-conservative funding (CostBook)
        margin = expected_favorable - total - edge_buffer
        passes = margin > 0.0

        # funding_m_sensitive: does the verdict flip between m=1 and m=m_max? (§5.5d)
        m_sensitive = self._m_sensitive(
            side, notional, expected_favorable, edge_buffer,
            liquidity_flag_in, liquidity_flag_out, expected_hold, market, cfg, passes,
        )

        reasons = [rt.components[-1].reason]
        if market.stale_flags:
            reasons.append("conservative_fallback")
        if not passes:
            reasons.append("net_ev_fail")

        return NetEVDecision(
            passes=passes,
            expected_favorable=expected_favorable,
            round_trip=rt,
            binding_funding=binding_funding,
            edge_buffer=edge_buffer,
            margin=margin,
            funding_sensitivity=self.cost_book.funding_sensitivity(
                side, notional, expected_hold, market, cfg
            ),
            funding_m_sensitive=m_sensitive,
            reason_codes=tuple(reasons),
        )

    def _m_sensitive(
        self,
        side: Side,
        notional: float,
        expected_favorable: float,
        edge_buffer: float,
        liquidity_flag_in: LiquidityFlag,
        liquidity_flag_out: LiquidityFlag,
        expected_hold: HoldHorizon,
        market: CostMarketState,
        cfg: ResolvedCostConfig,
        passes_at_mmax: bool,
    ) -> bool:
        """True if the verdict differs between ``m = 1`` and ``m = m_max`` (§5.5d)."""
        if expected_hold.m_max <= 1:
            return False
        h1 = dataclasses.replace(
            expected_hold,
            max_hold_quantile_ms=expected_hold.funding_interval_ms,
            m_max=1,
        )
        rt1 = self.cost_book.round_trip(
            side, notional, liquidity_flag_in, liquidity_flag_out, h1, market, cfg,
            gate_fee_tier=True,
        )
        passes_at_1 = (expected_favorable - rt1.total_quote_ccy - edge_buffer) > 0.0
        return passes_at_1 != passes_at_mmax
