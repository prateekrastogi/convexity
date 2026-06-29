"""RED/BLUE policy constants for the regime overlay (REGIME §3.1, §3.2, §3.6, §4, §4.1, §5).

Every number here is a SHARED modelling/economic constant (RED) or a finite validated grid (BLUE)
— never per-symbol, never PnL-fit (Invariant C / NOFIT). The Ledoit–Wolf shrinkage INTENSITIES are
explicitly absent: they are TEAL (closed-form, data-derived inside the measures) and do NOT enter
this RED/BLUE surface or ``K_regime`` (REGIME §2.2/§2.3/§4.1).

The canonical defaults reproduce the worked examples (REGIME §6) bit-for-bit; the BLUE grid is the
small bounded neighbourhood around them whose cardinality SYSTEM deflates against (capped at
``K_regime_max``, §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from convexity.regime.types import G_OF_STATE, RegimeStateName

# A per-measure hysteresis band = three (enter, exit) threshold pairs, one per state boundary
# CALM->NORMAL, NORMAL->STRESSED, STRESSED->CRISIS (REGIME §3.2). Units are the measure's native
# unit (percentile in [0,1] for vol/turb/liq; standardized displacement for corr's delta_AR).
BandPair = tuple[float, float]  # (enter, exit), enter is the tighter (escalate) mark


@dataclass(frozen=True)
class HysteresisBands:
    """The three (enter, exit) boundary pairs for one measure (REGIME §3.2).

    ``normal``/``stressed``/``crisis`` are the CALM->NORMAL, NORMAL->STRESSED, STRESSED->CRISIS
    boundaries. ``enter`` escalates (the high mark), ``exit`` de-escalates (the looser low mark);
    ``enter > exit`` is the non-zero hysteresis gap (Invariant HYST).
    """

    normal: BandPair
    stressed: BandPair
    crisis: BandPair

    def pairs(self) -> tuple[tuple[RegimeStateName, BandPair], ...]:
        """(target_state, (enter, exit)) for each boundary, ordered least->most stressed."""
        return (
            ("NORMAL", self.normal),
            ("STRESSED", self.stressed),
            ("CRISIS", self.crisis),
        )


# Percentile bands (vol_pct / turb_pct / liq_stress), RED canonical (REGIME §3.2): NORMAL@70/60,
# STRESSED@85/75, CRISIS@95/88. Stored in [0,1].
_PCT_BANDS = HysteresisBands(normal=(0.70, 0.60), stressed=(0.85, 0.75), crisis=(0.95, 0.88))

# delta_AR standardized-displacement bands, RED canonical (REGIME §3.2, §6.2): NORMAL@+0.5/+0.2,
# STRESSED@+1.0/+0.6, CRISIS@+2.0/+1.4.
_CORR_BANDS = HysteresisBands(normal=(0.5, 0.2), stressed=(1.0, 0.6), crisis=(2.0, 1.4))


@dataclass(frozen=True)
class PolicyDefaults:
    """Frozen RED/BLUE defaults for the whole pillar (REGIME §4). RED unless noted BLUE."""

    # --- state set + g(s) map (RED, fixed; mirrors the LEVERAGE governor ladder, §3.1) ---
    g_of_state: Mapping[RegimeStateName, float] = field(default_factory=lambda: dict(G_OF_STATE))
    g_floor: float = 0.25  # REGIME never flats the book (Invariant FLOOR)

    # --- absorption (REGIME §2.2) ---
    kappa: float = 0.20  # top-k fraction; k = ceil(kappa*n_t). BLUE {0.15, 0.20, 0.25}.
    w_ar: int = 60  # rolling correlation window W_AR (RED)
    w_ar_short: int = 15  # short trailing-mean window for delta_AR (RED, §2.2)
    w_ar_long: int = 250  # long trailing-mean + sigma_AR window (RED, §2.2)
    c_ar: float = 2.0  # observations-per-dimension floor: require W_AR >= c_AR*n_t (RED, ARFLOOR)
    frac_dn: float = 0.05  # fixed-n_t stratum band / freeze threshold (RED, ARDISPNORM)

    # --- turbulence (REGIME §2.3) ---
    w_turb: int = 60  # trailing mean/cov window W_turb (RED)
    c_cov: float = 2.0  # require W_turb >= c_cov*n_t else robust-diagonal fallback (RED, COND)
    w_turb_pct: int = 250  # trailing window for the causal percentile of d_norm (RED)

    # --- volatility (REGIME §2.1) ---
    w_pct: int = 250  # trailing window for the causal percentile of sigma_agg (RED)
    w_vov: int = 20  # vol-of-vol window (RED)
    w_rv: int = 20  # short realized-variance window per asset (RED)
    w_term_short: int = 20  # vol term-structure short window (RED)
    w_term_long: int = 120  # vol term-structure long window (RED)
    vov_confirm: float = 0.30  # vov above this TIGHTENS g^vol by one rung (confirming input, §2.1)
    term_confirm: float = 1.10  # term-structure slope above this TIGHTENS g^vol by one rung (§2.1)

    # --- numerical sanity (REGIME §2.3) ---
    cond_max: float = 1e4  # post-shrink condition-number gate for BOTH C_t and Sigma_t (RED, CONDNUM)

    # --- state machine (REGIME §3.3, §3.5) ---
    d_min: int = 3  # global minimum dwell, in decision bars. BLUE {2, 3, 4}.
    d_liq_min: int = 3  # minute-level dwell on the event-driven liq path (RED, §3.6 b')

    # --- SPC / ARL design targets (REGIME §3.6, RED, telemetry only — never gate g_t) ---
    alpha_normal: float = 0.30  # raw per-bar false-entry rate at the NORMAL band (single-pct)
    alpha_stressed: float = 0.15
    alpha_crisis: float = 0.05
    alpha_liq_min: float = 0.01  # per-MINUTE design target for the event-driven liq path (§3.6 b')

    # --- cost / rebalance band (REGIME §5.2, RED, fixed) ---
    rebal_min_frac: float = 0.10  # don't rebalance on a < 10% gross move (clause b)
    rebal_cost_cap: float = 0.50  # re-grossing UP must keep cost <= this fraction of delta_gross (c)

    # --- cadence (REGIME §3.5) ---
    tau_regime: str = "hourly"  # decision cadence. BLUE {"hourly", "session"}.

    # --- BLUE-grid cardinality cap (REGIME §4.1) ---
    k_regime_max: int = 64

    def bands_for(self, name: str) -> HysteresisBands:
        """The hysteresis bands for one measure: corr uses delta_AR units, the rest percentile."""
        return _CORR_BANDS if name == "corr" else _PCT_BANDS


# The finite BLUE grid (REGIME §4.1): the FIVE searched families, each a small bounded one-step
# neighbourhood around the RED canonical. Cardinality is registered with SYSTEM's trial ledger and
# capped at K_regime_max; the Cartesian product is NOT taken (it is a one-step grid per family).
BLUE_GRID: Mapping[str, tuple[object, ...]] = {
    "kappa": (0.15, 0.20, 0.25),  # Kritzman-Li ~1/5 of assets
    "vol_stressed_entry": (0.80, 0.85, 0.90),  # per-measure STRESSED-entry level
    "corr_stressed_entry": (0.8, 1.0, 1.2),
    "turb_stressed_entry": (0.80, 0.85, 0.90),
    "liq_stressed_entry": (0.80, 0.85, 0.90),
    "d_min": (2, 3, 4),  # run-length filter
    "tau_regime": ("hourly", "session"),  # ARL0-vs-ARL1 trade
}

DEFAULTS = PolicyDefaults()
