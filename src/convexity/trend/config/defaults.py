"""PolicyDefaults — the RED shared economic/modelling values (TREND §4 RED).

RED values are shared broadly, set once globally, reachable ONLY through ``resolve_config(defaults=
...)`` (Inv C). They NEVER appear on ``InstrumentSpec``. The trail multiplier schedule, the
deterioration detector kind/ARL, the drift floor, the gap-survival multiple, and the reduce/exit
priority order are the canonical RED knobs (§4, §10).

The convex trail multiplier schedule (§10) is monotone in open profit measured in initial-risk R:

    profit_r <= 0.5         -> tight multiple
    0.5 < profit_r <= 2.0   -> normal multiple
    2.0 < profit_r <= 4.0   -> loose multiple
    profit_r > 4.0          -> tail-preservation multiple

The four breakpoints are the ARCHITECTURE-pinned 0.4 / 1.0 / 3.0 / 5.0 R schedule anchors; the spec
prose bands (0.5/2.0/4.0) are the band edges. We keep both: the multipliers are keyed at the band
edges and the schedule is strictly increasing so INV-CVX (non-increasing marginal giveback) holds.

The declared governance buckets live in :data:`TREND_PARAM_REGISTRY`; the Inv-C test asserts no
RED/BLUE name leaks into ``InstrumentSpec`` (via ``assert_no_red_in_amber``).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry

ENGINE_VERSION = "trend-v1"

# §1.5b/§13 — the canonical normalized drift constant of the worked example: a +10% favorable move
# accumulated under integrated path-vol 0.1 gives z = log(1.1)/0.1. Pinned for the regression test.
Z_DRIFT_WORKED = 0.9531017980432486  # log(1.1) / 0.1


@dataclass(frozen=True)
class PolicyDefaults:
    """The RED constants (§4). Defaults sized for 1-minute crypto-perp + the cross-asset universe."""

    # --- §0/§9 drift -------------------------------------------------------------------------
    # Economic drift floor: a normalized drift z below this is not economically favorable (used by
    # the SPRT/CUSUM favorable-vs-dead hypotheses and the reduce slope-flip rule).
    drift_floor_z: float = 0.0
    eps: float = 1e-12  # variance-floor epsilon for z_drift denominator (§1.3)

    # --- §9 volatility -----------------------------------------------------------------------
    vol_method: str = "hybrid"  # ewma|mad|yang_zhang|hybrid (§9, §20 default hybrid)
    ewma_lambda: float = 0.94  # RiskMetrics-style decay for the EWMA close-to-close branch
    vol_warmup: int = 20  # bars before the vol stack is trusted (else MAD fallback)
    n_vol_buckets: int = 3
    n_liq_buckets: int = 3
    gap_var_multiple: float = 1.0  # gap variance applied once = (gap log-return)^2 * this (§9)

    # --- §9 kalman ---------------------------------------------------------------------------
    # Default class-preset noise ratio (process/observation). Class preset OVERRIDES this; it is
    # NEVER fit per symbol (Inv E). See config/asset_class.py for the eight class presets.
    kalman_noise_ratio: float = 0.01
    kalman_class_preset: str = "crypto_major"

    # --- §9 sequential detector --------------------------------------------------------------
    detector_kind: str = "cusum"  # exactly one (Inv B); CUSUM default (§20)
    arl0: int = 500  # target average run length to false alarm (RED; BLUE grid in tuning)
    cusum_k: float = 0.5  # reference value (drift-decay slack) in std-return units
    cusum_h: float = 5.0  # decision threshold (calibrated to arl0 under empirical/Student-t null)
    sprt_alpha: float = 0.05  # SPRT type-I (used only when detector_kind == 'sprt')
    sprt_beta: float = 0.10  # SPRT type-II
    sprt_mu0: float = 0.10  # H0 favorable normalized drift
    sprt_mu1: float = -0.10  # H1 dead/unfavorable normalized drift

    # --- §9 jump -----------------------------------------------------------------------------
    # Robust standardized-exceedance is the DEFAULT (ARCHITECTURE resolution #7); Lee-Mykland
    # bipower branch is config-gated OFF by default.
    jump_use_bipower: bool = False
    jump_exceedance_z: float = 6.0  # robust standardized-return exceedance threshold (|z| >)

    # --- §9 hazard ---------------------------------------------------------------------------
    hazard_mode: str = "disabled"  # disabled by default (§9, §20) until events justify it

    # --- §10 convex trail --------------------------------------------------------------------
    # Multiplier schedule keyed at the §10 band edges. Strictly increasing -> INV-CVX holds.
    # ARCHITECTURE anchors the breakpoints at 0.4/1.0/3.0/5.0 R; the spec band edges are
    # 0.5/2.0/4.0. We expose both: the band edges drive the lookup, the anchors are exposed for the
    # worked-example regression.
    trail_breakpoints_r: tuple[float, float, float] = (0.5, 2.0, 4.0)
    trail_schedule_anchors_r: tuple[float, float, float, float] = (0.4, 1.0, 3.0, 5.0)
    trail_mult_tight: float = 1.5
    trail_mult_normal: float = 2.5
    trail_mult_loose: float = 4.0
    trail_mult_tail: float = 6.0
    jump_with_widen: float = 1.5  # extra trail widening factor after a jump_with (§10)
    min_stop_tick_mult: float = 2.0  # initial risk must exceed (tick + half-spread) * this (§10)

    # --- §10 gap survival --------------------------------------------------------------------
    gap_survival_sigmas: float = 3.0  # adverse-gap stress = this many gap-sigmas (§10)
    gap_survival_min_fraction: float = 0.25  # never reduce below this on gap stress alone

    # --- §1 horizon --------------------------------------------------------------------------
    active_clock: str = "calendar_days"  # default primary clock (crypto 24/7, §1.2)
    # tau_cap (H_op) is NOT a default here — it is SYSTEM-injected via resolve_config (Inv HC).
    cap_binding_frac: float = 0.8  # cap_fraction_used above this raises the cap-binding warning

    # --- §6b hold-horizon / residual-move ----------------------------------------------------
    hold_quantile: float = 0.9  # survival-quantile backing max_hold_quantile_ms (§6b)
    residual_move_floor_r: float = 0.0  # conservative low band for expected residual move (R units)

    engine_version: str = ENGINE_VERSION
    detector_version: str = ENGINE_VERSION


# §4 — declared governance buckets. The Inv-C test asserts none of the RED/BLUE names leak into the
# per-instrument InstrumentSpec via assert_no_red_in_amber.
TREND_PARAM_REGISTRY = StaticParamRegistry(
    specs=(
        ParamSpec("drift_floor_z", Bucket.RED, "economic drift floor", "TREND §0/§4"),
        ParamSpec("vol_method", Bucket.RED, "volatility method", "TREND §9/§20"),
        ParamSpec("ewma_lambda", Bucket.RED, "EWMA vol decay", "TREND §9"),
        ParamSpec("gap_var_multiple", Bucket.RED, "gap variance multiple", "TREND §9"),
        ParamSpec("kalman_noise_ratio", Bucket.RED, "Kalman process/obs ratio", "TREND §9/Inv E"),
        ParamSpec("detector_kind", Bucket.RED, "exactly one detector", "TREND §4/Inv B"),
        ParamSpec("arl0", Bucket.RED, "target ARL to false alarm", "TREND §9"),
        ParamSpec("cusum_k", Bucket.RED, "CUSUM reference value", "TREND §9"),
        ParamSpec("cusum_h", Bucket.RED, "CUSUM threshold", "TREND §9"),
        ParamSpec("sprt_alpha", Bucket.RED, "SPRT type-I", "TREND §9"),
        ParamSpec("sprt_beta", Bucket.RED, "SPRT type-II", "TREND §9"),
        ParamSpec("jump_exceedance_z", Bucket.RED, "robust jump exceedance z", "TREND §9"),
        ParamSpec("hazard_mode", Bucket.RED, "hazard rung mode", "TREND §9/§20"),
        ParamSpec("trail_mult_tight", Bucket.RED, "tight trail multiple", "TREND §10"),
        ParamSpec("trail_mult_normal", Bucket.RED, "normal trail multiple", "TREND §10"),
        ParamSpec("trail_mult_loose", Bucket.RED, "loose trail multiple", "TREND §10"),
        ParamSpec("trail_mult_tail", Bucket.RED, "tail-preservation multiple", "TREND §10"),
        ParamSpec("min_stop_tick_mult", Bucket.RED, "stop noise-floor multiple", "TREND §10"),
        ParamSpec("gap_survival_sigmas", Bucket.RED, "gap-stress sigma multiple", "TREND §10"),
        ParamSpec("hold_quantile", Bucket.RED, "hold-horizon survival quantile", "TREND §6b"),
        # BLUE class-level grids (§15): searched offline under DSR, never per instrument.
        ParamSpec("trail_preset_grid", Bucket.BLUE, "trail presets", "TREND §15",
                  blue_grid=("tight", "balanced", "tail_preserve")),
        ParamSpec("detector_kind_grid", Bucket.BLUE, "detector kinds", "TREND §15",
                  blue_grid=("cusum", "sprt")),
        ParamSpec("arl0_grid", Bucket.BLUE, "ARL targets", "TREND §15",
                  blue_grid=(250, 500, 1000, 2000)),
        ParamSpec("vol_method_grid", Bucket.BLUE, "vol methods", "TREND §15",
                  blue_grid=("ewma", "mad", "hybrid")),
        ParamSpec("kalman_preset_grid", Bucket.BLUE, "Kalman class presets", "TREND §15",
                  blue_grid=("equity_liquid", "equity_illiquid", "futures_macro",
                             "futures_commodity", "precious_metal", "industrial_metal",
                             "crypto_major", "crypto_alt")),
        ParamSpec("hazard_mode_grid", Bucket.BLUE, "hazard rungs", "TREND §15",
                  blue_grid=("disabled", "constant", "single", "state")),
    )
)
