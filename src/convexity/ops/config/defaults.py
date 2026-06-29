"""OpsDefaults — the RED SPC limits + the BLUE candidate grids (OPS §4).

Spec: OPS §4 (parameter governance). OPS holds NO model parameter — only SPC LIMITS and the
in-control reference. Every field here is RED (a shared SPC/modelling constant) or BLUE (a finite
validated grid), NEVER per-symbol, NEVER PnL-fit (Invariant LIM/OOS/C). The canonical defaults
reproduce the §10 worked example bit-for-bit (``cusum_k=0.5, cusum_h=4.77, cusum_h2=8.0,
M_conf=3``) and the §6 standard operating points (EWMA ``(0.10, 2.70)``; SPRT ``(0.05, 0.10)``;
``ARL0=370``).

The limits are *derived* by ``validate/derive_limits.py`` from a SYSTEM in-control report + an ARL0
target and then FROZEN into one of these; they are immutable at runtime (Invariant LIM). RED is
reachable only through ``resolve_config(defaults=...)`` (Invariant C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry


@dataclass(frozen=True)
class CusumLimits:
    """Two-sided CUSUM ``(k, h)`` + the HALT escalation ``h2`` (§6.1, §7.3). ARL0 is the MC two-sided
    value the limit is solved against (``validate/arl.py``); ``arl0`` is recorded for Invariant ARL."""

    k: float = 0.5            # slack = δ_shift/2 in σ units (detect a 1σ shift)
    h: float = 4.77           # decision interval → two-sided ARL0 ≈ 370 (one-sided ≈ 743, §6.6)
    h2: float = 8.0           # escalation interval (DEGRADE → HALT, §7.3/§10)
    arl0: float = 370.0       # MC two-sided in-control ARL0 (Invariant ARL)
    arl1_1sigma: float = 10.4 # MC out-of-control ARL1 at the 1σ design shift (§10)


@dataclass(frozen=True)
class PageHinkleyLimits:
    """Page-Hinkley ``δ`` (allowed drift, σ units) and ``λ`` (detection threshold) (§6.2)."""

    delta: float = 0.5
    lam: float = 4.77


@dataclass(frozen=True)
class EwmaLimits:
    """EWMA ``(λ, L)`` standard Lucas–Saccucci pair → ARL0 ≈ 370, ARL1@1σ ≈ 10 (§6.3)."""

    lam: float = 0.10
    L: float = 2.70
    arl0: float = 370.0


@dataclass(frozen=True)
class ShewhartLimits:
    """Shewhart ``L_s`` — the EMPIRICAL / fitted-t tail quantile (§6.7c), NOT Gaussian 3σ.

    The default ``3.0`` is the thin-tailed Gaussian reference (ARL0 ≈ 370); ``derive_limits.py``
    overrides it with the in-control ``tail_quantile`` so the *empirical* exceedance rate == 1/ARL0.
    """

    L_s: float = 3.0
    arl0: float = 370.0


@dataclass(frozen=True)
class SprtLimits:
    """SPRT ``(α, β)`` operating point + the CUSUM-SPRT edge limit ``h_edge`` (§6.5/§6.5a).

    ``(α=0.05, β=0.10)`` with ``(p0=0.55, p1=0.45)`` gives ASN_H0 ≈ 99, ASN_H1 ≈ 118 (§6.5).
    ``h_edge`` is the §6.5a CUSUM-of-LLR limit whose pinned ARL0 enters the panel budget.
    """

    alpha: float = 0.05
    beta: float = 0.10
    h_edge: float = 4.77
    arl0: float = 370.0


@dataclass(frozen=True)
class OpsDefaults:
    """The full RED limit surface (§4). One frozen object, the sole config the detectors read.

    Binary-trigger thresholds (heartbeat misses, staleness/clock-skew windows, fill-rate floor, the
    DD-kill level), the warm-up + dedup + re-arm + flatten-retry windows, and the panel paging cadence
    are all RED. The DD-kill ``dd_kill`` (45%) sits strictly ABOVE the governor flat (43%) on the
    SHARED equity/peak reference (§7.1a, Invariant BRAKE).
    """

    # --- SPC limits ---
    cusum: CusumLimits = CusumLimits()
    page_hinkley: PageHinkleyLimits = PageHinkleyLimits()
    ewma: EwmaLimits = EwmaLimits()
    shewhart: ShewhartLimits = ShewhartLimits()
    sprt: SprtLimits = SprtLimits()

    # --- detector warm-up (§6.8) ---
    n_warm_cusum: int = 5
    n_warm_page_hinkley: int = 5
    n_warm_ewma: int = 40        # so (1-(1-λ)^{2i}) is within tolerance of 1 for λ=0.10
    n_warm_shewhart: int = 1
    n_warm_sprt: int = 1

    # --- 2-of-2 AND-rule (§6.6a) ---
    m_conf: int = 3              # confirmation window (consecutive samples both breached)

    # --- binary triggers (§7.4/§7.5/§7.6) ---
    n_hb_miss: int = 3           # consecutive missed heartbeats → HALT
    staleness_window_ms: int = 120_000   # feed age beyond this → DEGRADE (session-aware via AMBER)
    clock_skew_tol_ms: int = 1_000       # clock-skew beyond this → HALT
    fillrate_floor: float = 0.5          # realized fill rate below this over n_fr orders → DEGRADE
    n_fr: int = 10                       # window of orders for the fill-rate floor

    # --- drawdown backstop (§7.1, §7.1a) ---
    dd_kill: float = 0.45        # 45% — strictly above the LEVERAGE governor flat (43%)
    dd_governor_flat: float = 0.43  # echoed for the Invariant BRAKE ordering assertion (read-only)

    # --- alarm / re-arm / flatten cadence (§8.2/§8.4/§9.1c) ---
    dedup_window: int = 50       # coalesce repeated breaches of one fingerprint (samples)
    m_rearm: int = 10            # detector below limit for this many samples before re-arm (§8.4)
    flatten_retry_interval: int = 1   # bars between flatten re-attempts on reject/partial (§9.1c)
    flatten_retry_max: int = 5        # max flatten re-attempts before staying latched + paging

    # --- panel paging cadence (§6.6/§8.3) ---
    panel_arl0_target: float = 370.0  # per-source ARL0 target (the budget is Σ_d(1/ARL0,d))

    # --- rollout (§17) ---
    n_paper: int = 30
    n_canary: int = 50
    n_scaled: int = 100

    def __post_init__(self) -> None:
        if not (self.dd_kill > self.dd_governor_flat):
            raise ValueError(
                f"DD_kill ({self.dd_kill}) must be strictly above the governor flat "
                f"({self.dd_governor_flat}) — Invariant BRAKE §7.1a"
            )


DEFAULTS = OpsDefaults()

# BLUE grids — the finite, class-level, validated candidate sets searched OFFLINE inside SYSTEM's
# CPCV framework, promoted by the OOS limit gate (§4 BLUE, §11.2). Never fitted continuously, never
# per-symbol, never on raw P&L.
BLUE_GRID: Mapping[str, tuple[object, ...]] = {
    "cusum_kh": ((0.5, 4.77), (0.5, 5.0), (0.25, 5.0)),
    "ewma_lambda_L": ((0.10, 2.70), (0.20, 2.86)),
    "sprt_alpha_beta": ((0.05, 0.10), (0.01, 0.05)),
    "arl0_target": (500.0, 1000.0, 2000.0),
}


def param_registry() -> StaticParamRegistry:
    """Declared governance buckets for the OPS parameter surface (Invariant C / OOS introspection)."""
    return StaticParamRegistry(
        specs=(
            ParamSpec("cusum_k", Bucket.RED, "CUSUM slack (δ/2 in σ)", "OPS §6.1"),
            ParamSpec("cusum_h", Bucket.RED, "CUSUM decision interval", "OPS §6.1"),
            ParamSpec("cusum_h2", Bucket.RED, "CUSUM HALT escalation", "OPS §7.3"),
            ParamSpec("ph_delta", Bucket.RED, "Page-Hinkley drift tolerance", "OPS §6.2"),
            ParamSpec("ph_lambda", Bucket.RED, "Page-Hinkley threshold", "OPS §6.2"),
            ParamSpec("ewma_lambda", Bucket.RED, "EWMA smoothing", "OPS §6.3"),
            ParamSpec("ewma_L", Bucket.RED, "EWMA limit width", "OPS §6.3"),
            ParamSpec("shewhart_L_s", Bucket.RED, "Shewhart empirical-tail quantile", "OPS §6.7c"),
            ParamSpec("sprt_alpha", Bucket.RED, "SPRT type-I rate", "OPS §6.5"),
            ParamSpec("sprt_beta", Bucket.RED, "SPRT type-II rate", "OPS §6.5"),
            ParamSpec("h_edge", Bucket.RED, "CUSUM-SPRT edge limit", "OPS §6.5a"),
            ParamSpec("m_conf", Bucket.RED, "2-of-2 confirmation window", "OPS §6.6a"),
            ParamSpec("dd_kill", Bucket.RED, "Max-drawdown HALT backstop", "OPS §7.1"),
            ParamSpec("n_hb_miss", Bucket.RED, "Heartbeat-miss count", "OPS §7.6"),
            ParamSpec("fillrate_floor", Bucket.RED, "Fill-rate collapse floor", "OPS §7.4"),
            ParamSpec("m_rearm", Bucket.RED, "Re-arm window", "OPS §8.4"),
            ParamSpec("flatten_retry_max", Bucket.RED, "Flatten retry max", "OPS §9.1c"),
            ParamSpec("cusum_kh", Bucket.BLUE, "CUSUM (k,h) grid", "OPS §4",
                      blue_grid=BLUE_GRID["cusum_kh"]),
            ParamSpec("ewma_lambda_L", Bucket.BLUE, "EWMA (λ,L) grid", "OPS §4",
                      blue_grid=BLUE_GRID["ewma_lambda_L"]),
            ParamSpec("sprt_alpha_beta", Bucket.BLUE, "SPRT (α,β) grid", "OPS §4",
                      blue_grid=BLUE_GRID["sprt_alpha_beta"]),
            ParamSpec("arl0_target", Bucket.BLUE, "ARL0 target grid", "OPS §4",
                      blue_grid=BLUE_GRID["arl0_target"]),
            # AMBER metadata (per-instrument; carries NO RED/BLUE field).
            ParamSpec("staleness_window_ms", Bucket.AMBER, "Feed-staleness window per venue",
                      "OPS §4"),
            ParamSpec("clock_skew_tol_ms", Bucket.AMBER, "Clock-skew tolerance per venue", "OPS §4"),
        )
    )


# The AMBER per-instrument field names — what an Invariant-C test passes to assert_no_red_in_amber.
AMBER_FIELD_NAMES: frozenset[str] = frozenset(
    {"staleness_window_ms", "clock_skew_tol_ms", "venue", "calendar_id", "funding_interval_ms"}
)

__all__ = [
    "OpsDefaults",
    "DEFAULTS",
    "CusumLimits",
    "PageHinkleyLimits",
    "EwmaLimits",
    "ShewhartLimits",
    "SprtLimits",
    "BLUE_GRID",
    "AMBER_FIELD_NAMES",
    "param_registry",
]
