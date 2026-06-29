"""PolicyDefaults — the RED shared economic/modelling constants (LEVERAGE §4).

RED = shared, global, never per-symbol, reachable ONLY through
``resolve_config(defaults=...)`` (Invariant C). Per-instrument tuning of any RED
constant on realized P&L is forbidden (§4). Each field cites the spec section that
defines it. The governor breakpoints and the vol-blend weights are the finite BLUE
grids searched offline at the class level only (§4 BLUE / §6 DDCAUSAL) — recorded in
:data:`LEVERAGE_PARAM_REGISTRY`, never as per-instrument fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry


@dataclass(frozen=True)
class PolicyDefaults:
    """The RED constants. Defaults sized for the ~$1000 retail Binance USDT-M book."""

    # §4 / §8.1 per-position committed-margin loss cap c, in [0.12, 0.15]. The L_stop
    # numerator. NOT a fraction of total equity (P1) — the equity ladder (Inv M) bounds it.
    capital_cap: float = 0.13

    # §5.2 Kelly fractions: base Half-Kelly and the stressed 1/3, switched on observables.
    kelly_phi_base: float = 0.5
    kelly_phi_stressed: float = 1.0 / 3.0
    # §5.2 the OBSERVABLE switch thresholds (RED): realized-vol percentile, correlation
    # instability, jump intensity above which φ drops to 1/3.
    kelly_vol_pct_hi: float = 0.90
    kelly_vol_pct_lo: float = 0.80  # hysteresis: return toward 0.5 only below this
    kelly_corr_instability_hi: float = 0.50
    kelly_corr_instability_lo: float = 0.40
    kelly_jump_intensity_hi: float = 2.0
    kelly_jump_intensity_lo: float = 1.5

    # §8.1 stop-distance vol multiple k (e.g. ~2σ multiday). d = max(k·σ_horizon, MAE_q).
    stop_vol_multiple: float = 2.0
    # §8.1 conditional-MAE event-count floor below which the bucket collapses to global.
    cond_mae_min_events: int = 30

    # §8.2 per-position annualized volatility target σ_target (e.g. 15-20%).
    sigma_target: float = 0.175

    # §8.3 Expected-Shortfall confidence level and budget (loss fraction of position margin).
    es_level: float = 0.975
    es_budget: float = 0.15
    # §8.3 gap-survival ruin threshold (fraction of committed capital, e.g. 30-40%).
    gap_ruin_threshold: float = 0.35
    # §8.3 typical-gap multiple used as the fallback worst gap when the worst-gap sample is thin.
    gap_typical_multiple: float = 3.0
    # §8.3 EVT event-count floor below which the GPD fit is rejected for the empirical fallback.
    tail_min_exceed: int = 30

    # §7 total portfolio-heat ceiling H_book (RED) — the book's MCTR-share budget cap;
    # scaled to g_t·H_book so REGIME's macro cut enters HERE only (P7).
    h_book: float = 0.20
    # §7 max MCTR share any single instrument/cluster may carry (concentration cap).
    mctr_share_cap: float = 0.40
    # §7 Ledoit-Wolf post-shrink condition-number gate (Inv COND-L); above this Σ is ill-conditioned.
    cov_max_condition: float = 1.0e4
    # §7 minimum eigenvalue gate (relative to max eigenvalue) for the same conditioning check.
    cov_min_eig_ratio: float = 1.0e-6
    # §7 correlation-cluster threshold for budget sharing (simple capped grouping, NOT HRP).
    cluster_corr_threshold: float = 0.6

    # §8.4 free-margin buffer: keep >= this fraction free beyond the immediate requirement.
    free_margin_buffer: float = 0.25
    # §8.4 correlation-stress multiplier applied to portfolio margin (P4 netted-margin spike).
    margin_stress_multiplier: float = 2.0

    # §8.5 liquidity: max participation rate over the exit window.
    max_participation_rate: float = 0.08
    # §8.5 max tolerated round-trip spread+impact cost (bps) before L_liquidity tightens.
    max_liquidity_cost_bps: float = 50.0
    # §8.5 target exit window (minutes) used to size the participation constraint.
    target_exit_minutes: float = 30.0

    # §6 graduated drawdown-governor breakpoints and multipliers (BLUE-searchable, class-level;
    # geometric de-risking ladder, OOS-paid in SYSTEM's multiplicity budget, Inv DDCAUSAL).
    governor_breakpoints: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    governor_multipliers: tuple[float, ...] = (1.00, 0.75, 0.50, 0.25)
    # §6 the pinned 40-43% LINEAR ramp 0.25 -> 0.00 and the flat-the-book level.
    governor_ramp_lo: float = 0.40
    governor_ramp_hi: float = 0.43

    # §8.6.6 absolute per-class hard guardrail caps L_class_max (RED, class-level only).
    # Keyed by foundation AssetClass; set conservatively ABOVE typical live leverage so the
    # cap binds only on a model fault (`class_cap_capped`), not in normal operation.
    class_cap_by_asset: Mapping[str, float] = field(
        default_factory=lambda: {
            "crypto_perp": 3.0,
            "crypto_spot": 3.0,
            "us_equity": 4.0,
            "precious_metal": 6.0,
            "industrial_metal": 6.0,
            "commodity_future": 6.0,
        }
    )
    # §13 default class cap when the asset class is unknown (conservative).
    default_class_cap: float = 3.0

    # §3 engine version string echoed on every LeverageDecision for telemetry.
    engine_version: str = "leverage-1.0.0"


# §4 — the declared governance buckets. The Inv-C test asserts none of the RED/BLUE
# names below leaks into a per-instrument InstrumentLeverageSpec.
LEVERAGE_PARAM_REGISTRY = StaticParamRegistry(
    specs=(
        ParamSpec("capital_cap", Bucket.RED, "per-position committed-margin loss cap c",
                  "LEVERAGE §8.1"),
        ParamSpec("kelly_phi_base", Bucket.RED, "Half-Kelly fraction", "LEVERAGE §5.2"),
        ParamSpec("kelly_phi_stressed", Bucket.RED, "stressed 1/3 Kelly fraction", "LEVERAGE §5.2"),
        ParamSpec("kelly_vol_pct_hi", Bucket.RED, "vol-pct fraction-switch", "LEVERAGE §5.2"),
        ParamSpec("kelly_vol_pct_lo", Bucket.RED, "vol-pct hysteresis", "LEVERAGE §5.2"),
        ParamSpec("kelly_corr_instability_hi", Bucket.RED, "corr-instability switch",
                  "LEVERAGE §5.2"),
        ParamSpec("kelly_corr_instability_lo", Bucket.RED, "corr hysteresis", "LEVERAGE §5.2"),
        ParamSpec("kelly_jump_intensity_hi", Bucket.RED, "jump-intensity switch", "LEVERAGE §5.2"),
        ParamSpec("kelly_jump_intensity_lo", Bucket.RED, "jump hysteresis", "LEVERAGE §5.2"),
        ParamSpec("stop_vol_multiple", Bucket.RED, "stop-distance vol multiple k", "LEVERAGE §8.1"),
        ParamSpec("cond_mae_min_events", Bucket.RED, "conditional-MAE event floor",
                  "LEVERAGE §8.1"),
        ParamSpec("sigma_target", Bucket.RED, "per-position vol target", "LEVERAGE §8.2"),
        ParamSpec("es_level", Bucket.RED, "Expected-Shortfall confidence level", "LEVERAGE §8.3"),
        ParamSpec("es_budget", Bucket.RED, "Expected-Shortfall budget", "LEVERAGE §8.3"),
        ParamSpec("gap_ruin_threshold", Bucket.RED, "gap-survival ruin threshold",
                  "LEVERAGE §8.3"),
        ParamSpec("gap_typical_multiple", Bucket.RED, "typical-gap fallback multiple",
                  "LEVERAGE §8.3"),
        ParamSpec("tail_min_exceed", Bucket.RED, "EVT exceedance-count floor", "LEVERAGE §8.3"),
        ParamSpec("h_book", Bucket.RED, "portfolio-heat ceiling H_book", "LEVERAGE §7"),
        ParamSpec("mctr_share_cap", Bucket.RED, "single-instrument MCTR share cap", "LEVERAGE §7"),
        ParamSpec("cov_max_condition", Bucket.RED, "covariance condition-number gate",
                  "LEVERAGE §7"),
        ParamSpec("cov_min_eig_ratio", Bucket.RED, "covariance min-eigenvalue gate", "LEVERAGE §7"),
        ParamSpec("cluster_corr_threshold", Bucket.RED, "cluster correlation threshold",
                  "LEVERAGE §7"),
        ParamSpec("free_margin_buffer", Bucket.RED, "free-margin buffer", "LEVERAGE §8.4"),
        ParamSpec("margin_stress_multiplier", Bucket.RED, "portfolio-margin stress mult",
                  "LEVERAGE §8.4"),
        ParamSpec("max_participation_rate", Bucket.RED, "max participation rate", "LEVERAGE §8.5"),
        ParamSpec("max_liquidity_cost_bps", Bucket.RED, "max liquidity cost", "LEVERAGE §8.5"),
        ParamSpec("target_exit_minutes", Bucket.RED, "target exit window", "LEVERAGE §8.5"),
        ParamSpec("governor_breakpoints", Bucket.BLUE, "governor ladder breakpoints",
                  "LEVERAGE §6", blue_grid=((0.10, 0.20, 0.30, 0.40),)),
        ParamSpec("governor_multipliers", Bucket.BLUE, "governor ladder multipliers",
                  "LEVERAGE §6", blue_grid=((1.00, 0.75, 0.50, 0.25),)),
        ParamSpec("governor_ramp_lo", Bucket.RED, "governor ramp lo (40%)", "LEVERAGE §6"),
        ParamSpec("governor_ramp_hi", Bucket.RED, "governor ramp hi (43%)", "LEVERAGE §6"),
        ParamSpec("class_cap_by_asset", Bucket.RED, "hard per-class leverage cap",
                  "LEVERAGE §8.6.6"),
        ParamSpec("default_class_cap", Bucket.RED, "default hard class cap", "LEVERAGE §13"),
    )
)
