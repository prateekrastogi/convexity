"""CostDefaults — the RED shared economic/modelling constants (COST §4).

RED = shared, global, never per-symbol, reachable ONLY through
``resolve_config(defaults=...)`` (Invariant C-cfg). Per-symbol tuning of any RED
constant on realized P&L is forbidden (§4). Each field below cites the spec section
that defines it. A finite BLUE grid for the funding half-life ``h`` and basis
shrinkage ``lambda`` is searched offline at the class level only (§4 BLUE) — those
grids are recorded in :data:`COST_PARAM_REGISTRY`, never as per-instrument fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry


@dataclass(frozen=True)
class CostDefaults:
    """The RED constants. Defaults sized for the ~$1000 retail Binance USDT-M book."""

    # §5.5a EWMA — half-life h in NUMBER OF 8h STAMPS; alpha = 1 - 2^(-1/h). ~3 days.
    funding_ewma_half_life_stamps: float = 9.0
    # §5.5a event-count floor below which the EWMA falls back (>= 2*h so ESS is filled).
    n_funding_min: int = 18
    # §5.5c residual-event floor below which the comonotone band falls back to class worst.
    n_residual_min: int = 18
    # §5.5b basis -> carry shrinkage lambda in [0,1] (BLUE-searchable, class-level only).
    basis_shrinkage_lambda: float = 0.30
    # §5.5c conservative band quantile q (the worse-for-the-position tail), e.g. 90th pct.
    band_quantile_q: float = 0.90
    # §5.5d max-hold quantile (as a fraction) that sets m_max for the gate — caller may
    # override per call via the HoldHorizon it passes; this is the class default.
    max_hold_quantile: float = 0.90
    # §6 cost-as-fraction-of-edge rejection threshold — REPORTS ONLY, never the gate.
    cost_fraction_reject_threshold: float = 1.0
    # §5.3 fallback wide-spread half-spread (bps) when bid/ask missing — never zero (P6).
    fallback_half_spread_bps: float = 5.0
    # §5.4 conservative impact coefficient (FRACTION-valued) used when k_i cannot be fitted.
    fallback_impact_k_fraction: float = 0.50
    # §8.1 minimum-edge buffer added to the gate (a margin of safety), quote ccy.
    edge_buffer: float = 2.0
    # §5.5d / §12 funding-bootstrap block length L_funding (~2*h stamps), shared with the band.
    l_funding: int = 18
    # §7 fee-tier boundary-proximity tolerance (fraction of the breakpoint) that triggers
    # conservative (worse) tier pricing in the gate (Invariant J-tier).
    tier_proximity: float = 0.05
    # §5.5a funding-forecast staleness window (ms) beyond which history is treated stale.
    funding_staleness_ms: int = 3 * 24 * 60 * 60 * 1000
    # §9 TCA unattributed-residual tolerance (quote ccy); plug above this fails the test (Inv R).
    tca_residual_tolerance: float = 1e-6

    @property
    def funding_ewma_alpha(self) -> float:
        """alpha = 1 - 2^(-1/h) (§5.5a)."""
        return 1.0 - 2.0 ** (-1.0 / self.funding_ewma_half_life_stamps)

    @property
    def funding_ewma_ess(self) -> float:
        """ESS = (2 - alpha) / alpha (§5.5a)."""
        a = self.funding_ewma_alpha
        return (2.0 - a) / a


# §4 — the declared governance buckets for COST's parameters. The Inv-C test asserts
# none of the RED/BLUE names below leaks into a per-instrument InstrumentCostSpec.
COST_PARAM_REGISTRY = StaticParamRegistry(
    specs=(
        ParamSpec("funding_ewma_half_life_stamps", Bucket.RED, "EWMA half-life in 8h stamps",
                  "COST §5.5a"),
        ParamSpec("n_funding_min", Bucket.RED, "EWMA event-count floor", "COST §5.5a"),
        ParamSpec("n_residual_min", Bucket.RED, "band residual event-count floor", "COST §5.5c"),
        ParamSpec("basis_shrinkage_lambda", Bucket.RED, "basis->carry shrinkage", "COST §5.5b"),
        ParamSpec("band_quantile_q", Bucket.RED, "conservative band quantile", "COST §5.5c"),
        ParamSpec("max_hold_quantile", Bucket.RED, "max-hold quantile -> m_max", "COST §5.5d"),
        ParamSpec("cost_fraction_reject_threshold", Bucket.RED, "report-only edge ratio",
                  "COST §6"),
        ParamSpec("fallback_half_spread_bps", Bucket.RED, "wide-spread fallback", "COST §5.3"),
        ParamSpec("fallback_impact_k_fraction", Bucket.RED, "impact fallback coeff", "COST §5.4"),
        ParamSpec("edge_buffer", Bucket.RED, "gate margin of safety", "COST §8.1"),
        ParamSpec("l_funding", Bucket.RED, "funding bootstrap block length", "COST §5.5d/§12"),
        ParamSpec("tier_proximity", Bucket.RED, "fee-tier proximity tolerance", "COST §7"),
        ParamSpec("funding_staleness_ms", Bucket.RED, "funding staleness window", "COST §5.5a"),
        ParamSpec("tca_residual_tolerance", Bucket.RED, "TCA plug tolerance", "COST §9"),
        # BLUE class-level grids (§4): searched offline under DSR/PSR, never per instrument.
        ParamSpec("h_grid", Bucket.BLUE, "funding EWMA half-life candidates", "COST §4",
                  blue_grid=(6.0, 9.0, 12.0)),
        ParamSpec("lambda_grid", Bucket.BLUE, "basis shrinkage candidates", "COST §4",
                  blue_grid=(0.0, 0.3, 0.6)),
    )
)
