"""DetectorDefaults — the RED shared modelling policies (BREAKOUT §2 RED).

RED values are modelling policies: shared across instruments, set once globally, reachable ONLY
through ``resolve_config(defaults=...)`` (Invariant D / §2). They NEVER appear in ``InstrumentSpec``.
Each field cites the spec section. The breakout-confirmation cross magnitude ``k`` and the lookback
``windows`` are the canonical RED/BLUE scalars of Invariant J — a single shared ``k`` runs on crypto
and equities, never fit per symbol.

The declared governance buckets live in :data:`BREAKOUT_PARAM_REGISTRY`; the Inv-D test asserts no
RED/BLUE name leaks into ``InstrumentSpec`` (via ``assert_no_red_in_amber``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry

# §8 evidence-combiner weight presets. Weights are RED/BLUE preset values, never fit freely per
# symbol. Keys are the channels of EvidenceFrame; w_false penalises false-break risk.
EvidencePreset = dict


def _preset(strict: bool = False, early: bool = False) -> dict[str, float]:
    """Return one of the three §8 presets as a weight map."""
    if strict:  # fewer entries, stronger regime-change evidence required
        return dict(
            w_glr=1.4, w_bocpd=1.2, w_range=0.8, w_compression=1.0,
            w_trend=0.8, w_flow=0.4, w_jump=0.6, w_context=0.3, w_false=1.6,
        )
    if early:  # earlier entries, higher false-break risk (must be validated)
        return dict(
            w_glr=0.9, w_bocpd=0.8, w_range=1.2, w_compression=1.0,
            w_trend=1.0, w_flow=0.5, w_jump=0.4, w_context=0.3, w_false=0.8,
        )
    return dict(  # balanced — the default
        w_glr=1.0, w_bocpd=1.0, w_range=1.0, w_compression=1.0,
        w_trend=0.8, w_flow=0.4, w_jump=0.5, w_context=0.3, w_false=1.2,
    )


PRESETS: dict[str, dict[str, float]] = {
    "strict": _preset(strict=True),
    "balanced": _preset(),
    "early": _preset(early=True),
}


@dataclass(frozen=True)
class DetectorDefaults:
    """The RED constants. Defaults sized for 1-minute crypto-perp + the cross-asset universe."""

    # §2/§Inv J — breakout-confirmation cross magnitude k, penetration in causal-sigma units.
    # Single shared scalar; BLUE-searchable, NEVER per-symbol.
    cross_k: float = 1.0
    # §2/§Inv J — candidate lookback windows for the Donchian barriers (BLUE grid lives in tuning).
    windows: tuple[int, ...] = (20, 40, 80)
    # §12.2 — rolling vol window (EWMA-style) used to form the predictive sigma[t].
    vol_window: int = 60
    # §7 — target alert policy: expected alerts per session (drives empirical-null threshold).
    target_alert_rate: float = 0.02
    # §2 — ARL0 target (avg run length to false alarm) for the change statistics.
    arl0: int = 500
    # detect/bocpd — BOCPD hazard (RED/BLUE, never per-symbol) + run-length cap (pruning).
    bocpd_hazard: float = 1.0 / 500.0
    bocpd_max_run_length: int = 300
    # §8 — evidence-combiner preset name (strict|balanced|early).
    preset: str = "balanced"
    # §8 — the combiner weights (resolved from preset; overridable per class via BLUE).
    weights: dict[str, float] = field(default_factory=lambda: dict(PRESETS["balanced"]))
    # detect/combine — final accept threshold on quality_score (RED; scale-arbitrary, Inv K).
    quality_threshold: float = 1.5
    # §9 — false-break cap: reject candidates whose false_break_risk exceeds this.
    false_break_cap: float = 0.85
    # §9 — stop-distance floor as a multiple of (tick + half-spread) noise floor.
    min_stop_tick_mult: float = 2.0
    # §9 — max stop distance as a multiple of the one-bar predictive sigma_t (the R-budget ceiling).
    # The stop spans the pre-break Donchian range, which is naturally many one-bar sigmas wide, so
    # this caps only pathologically wide stops where trend convexity is gone.
    max_stop_sigma_mult: float = 40.0
    # detect/dedupe — cooldown bars suppressing same-zone repeats.
    cooldown_bars: int = 30
    # §10 — evaluation horizons (bars) for labels/walk-forward.
    horizons: tuple[int, ...] = (15, 30, 60, 120)
    # §10 — R-multiple targets for first-touch labels.
    r_targets: tuple[float, ...] = (1.0, 2.0)
    # §10 — purge + embargo bars for walk-forward / CPCV (max horizon by default).
    embargo_bars: int = 120
    # §7 — minimum events per empirical-null bucket before it is trusted (else fall back).
    null_min_events: int = 50
    # §7 — number of vol/liq regime buckets (Invariant M local conditioning).
    n_vol_buckets: int = 3
    n_liq_buckets: int = 3
    # §10 — cross-sectional FDR level for the universe-scan promotion gate (Invariant L).
    fdr_alpha: float = 0.10
    detector_version: str = "breakout-v1"


# §2 — declared governance buckets. The Inv-D test asserts none of the RED/BLUE names leak into the
# per-instrument InstrumentSpec via assert_no_red_in_amber.
BREAKOUT_PARAM_REGISTRY = StaticParamRegistry(
    specs=(
        ParamSpec("cross_k", Bucket.RED, "vol-normalized cross magnitude k", "BREAKOUT §2/Inv J"),
        ParamSpec("vol_window", Bucket.RED, "rolling predictive-vol window", "BREAKOUT §12.2"),
        ParamSpec("target_alert_rate", Bucket.RED, "expected alerts per session", "BREAKOUT §7"),
        ParamSpec("arl0", Bucket.RED, "avg run length to false alarm", "BREAKOUT §2"),
        ParamSpec("bocpd_hazard", Bucket.RED, "BOCPD hazard rate", "BREAKOUT §detect/bocpd"),
        ParamSpec("quality_threshold", Bucket.RED, "final accept threshold", "BREAKOUT §detect/combine"),
        ParamSpec("false_break_cap", Bucket.RED, "false-break rejection cap", "BREAKOUT §9"),
        ParamSpec("min_stop_tick_mult", Bucket.RED, "stop-distance floor", "BREAKOUT §9"),
        ParamSpec("max_stop_sigma_mult", Bucket.RED, "stop-distance ceiling", "BREAKOUT §9"),
        ParamSpec("cooldown_bars", Bucket.RED, "dedupe cooldown", "BREAKOUT §detect/dedupe"),
        ParamSpec("embargo_bars", Bucket.RED, "purge/embargo", "BREAKOUT §10"),
        ParamSpec("fdr_alpha", Bucket.RED, "cross-sectional FDR level", "BREAKOUT §10/Inv L"),
        # BLUE class-level grids (§2 BLUE): searched offline under DSR/PSR, never per instrument.
        ParamSpec("windows", Bucket.BLUE, "candidate lookback windows", "BREAKOUT §2",
                  blue_grid=((20, 40, 80), (40, 80, 160))),
        ParamSpec("cross_k_grid", Bucket.BLUE, "candidate cross magnitudes", "BREAKOUT §2",
                  blue_grid=(0.5, 1.0, 1.5)),
        ParamSpec("arl0_grid", Bucket.BLUE, "candidate ARL0 targets", "BREAKOUT §2",
                  blue_grid=(250, 500, 1000, 2000)),
        ParamSpec("preset_grid", Bucket.BLUE, "evidence-combiner presets", "BREAKOUT §8",
                  blue_grid=("strict", "balanced", "early")),
    )
)
