"""PolicyDefaults — the RED shared economic / modelling constants for EXECUTION (§4).

RED = shared, global, never per-symbol, reachable ONLY through ``resolve_config(defaults=...)``
(Invariant C). Per-symbol tuning of any RED constant on realized P&L is forbidden (Invariant T): χ,
the timeouts, and the urgency thresholds are class-level constants or BLUE grids searched against
realized IS (a COST metric), never fitted per symbol on trade profit.

Every field cites the spec section that defines it. The chase fraction χ, the maker-floor gate
constants (``floor_RED``, ``z``, ``κ``, ``calib_RED``), the fill-prob event-count floors
(``N_bucket``, ``N_global``, ``N_red``), and the no-fill-split abort floor are all here, NEVER on a
per-instrument :class:`InstrumentSpec` (the bug Invariant C hunts).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry


@dataclass(frozen=True)
class PolicyDefaults:
    """The RED constants. Defaults sized for the ~$1000 retail Binance USDT-M book (§4 / §13)."""

    # §5.3 — max chase fraction χ of the entry->invalidation distance. RED, range [0.25, 0.40].
    chase_fraction_chi: float = 0.30
    # §5.2 — cancel/replace cadence: ticks per reprice step, per-step timeout, max steps.
    reprice_tick_step: int = 1
    timeout_step_s: float = 5.0
    max_chase_steps: int = 8
    # §7 — order retry / idempotency bounds.
    max_retries: int = 3
    # §6.3 maker-floor gate — keyed to the p_fill LOWER-confidence bound, not the point estimate.
    floor_red: float = 0.30  # floor_RED — base maker fill-prob floor below which posting stops.
    z_lcb: float = 1.645  # z — the LCB z-multiplier (one-sided 95%).
    kappa_calib: float = 2.0  # κ — calibration-error floor-widening coefficient (§6.3).
    calib_red: float = 0.05  # calib_RED — calibration-error threshold above which floor_eff widens.
    # §5.5 #2 — fill-prob event-count floors (bucket -> global -> conservative class prior).
    n_bucket: int = 200  # N_bucket — per-bucket passive-fill floor.
    n_global: int = 1000  # N_global — pooled-global passive-fill floor.
    n_red: int = 50  # N_red — cold-start floor; below this maker is GATED OFF by default.
    # §5.5 #2 — cold-start POST_ONLY exploration fraction (PATIENT/calm only).
    cold_start_explore_fraction: float = 0.05
    # §5.1.2 — no-fill-split abort floor: conservative thin-data default biases P(abort) low so the
    # maker branch never looks phantom-cheap (one-signed conservatism toward crossing).
    p_abort_floor: float = 0.05
    # §6 / §7 — through-tick count for MARKETABLE_LIMIT (so crossing slippage is bounded by build).
    marketable_through_ticks: int = 2
    # §5.1 — fill horizon (seconds) the cost model prices the post-or-cross over.
    fill_horizon_s: float = 30.0
    # §4 — square-root impact coefficient η (RED/class; ~0 at $1k by Invariant I).
    impact_eta: float = 0.5
    # §6 — rate-limit backoff base (seconds) and multiplier (exponential).
    backoff_base_s: float = 1.0
    backoff_multiplier: float = 2.0
    # §6.2 — feed-staleness threshold (seconds) past which the halt/outage gate fires for exits.
    stale_threshold_s: float = 60.0


# §4 — the declared governance buckets for EXECUTION. The Invariant-C test asserts none of the
# RED/BLUE names below leaks into a per-instrument :class:`InstrumentSpec`.
EXECUTION_PARAM_REGISTRY = StaticParamRegistry(
    specs=(
        ParamSpec("chase_fraction_chi", Bucket.RED, "max chase fraction χ", "EXECUTION §5.3"),
        ParamSpec("reprice_tick_step", Bucket.RED, "reprice tick step", "EXECUTION §5.2"),
        ParamSpec("timeout_step_s", Bucket.RED, "per-step chase timeout", "EXECUTION §5.2"),
        ParamSpec("max_chase_steps", Bucket.RED, "max chase steps", "EXECUTION §5.2"),
        ParamSpec("max_retries", Bucket.RED, "max order retries", "EXECUTION §7"),
        ParamSpec("floor_red", Bucket.RED, "maker fill-prob floor", "EXECUTION §6.3"),
        ParamSpec("z_lcb", Bucket.RED, "LCB z-multiplier", "EXECUTION §6.3"),
        ParamSpec("kappa_calib", Bucket.RED, "calib floor-widen coeff κ", "EXECUTION §6.3"),
        ParamSpec("calib_red", Bucket.RED, "calib-error threshold", "EXECUTION §6.3"),
        ParamSpec("n_bucket", Bucket.RED, "per-bucket fill-count floor", "EXECUTION §5.5"),
        ParamSpec("n_global", Bucket.RED, "global fill-count floor", "EXECUTION §5.5"),
        ParamSpec("n_red", Bucket.RED, "cold-start fill-count floor", "EXECUTION §5.5"),
        ParamSpec("cold_start_explore_fraction", Bucket.RED, "cold-start explore frac",
                  "EXECUTION §5.5"),
        ParamSpec("p_abort_floor", Bucket.RED, "no-fill-split abort floor", "EXECUTION §5.1.2"),
        ParamSpec("marketable_through_ticks", Bucket.RED, "marketable through-ticks",
                  "EXECUTION §7"),
        ParamSpec("fill_horizon_s", Bucket.RED, "fill/cost horizon", "EXECUTION §5.1"),
        ParamSpec("impact_eta", Bucket.RED, "sqrt-impact coefficient η", "EXECUTION §4"),
        ParamSpec("backoff_base_s", Bucket.RED, "rate-limit backoff base", "EXECUTION §6"),
        ParamSpec("backoff_multiplier", Bucket.RED, "rate-limit backoff mult", "EXECUTION §6"),
        ParamSpec("stale_threshold_s", Bucket.RED, "feed-staleness threshold", "EXECUTION §6.2"),
        # BLUE class-level grids (§4): searched offline under deflated-Sharpe, never per instrument.
        ParamSpec("chi_grid", Bucket.BLUE, "chase-fraction candidates", "EXECUTION §4",
                  blue_grid=(0.25, 0.30, 0.40)),
        ParamSpec("reprice_step_grid", Bucket.BLUE, "reprice-step candidates", "EXECUTION §4",
                  blue_grid=(1, 2)),
        ParamSpec("n_bucket_grid", Bucket.BLUE, "N_bucket candidates", "EXECUTION §5.5",
                  blue_grid=(100, 200, 400)),
    )
)
