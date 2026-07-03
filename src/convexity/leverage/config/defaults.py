"""RED constants for the LEVERAGE pillar (LEVERAGE.md §3 parameter governance).

Every constant here is RED — a shared economic/modelling constant, unreachable from any
per-instrument config (Invariant C-cfg, §3). Each field carries its §3 rationale and the exact
spec anchor; none is a bare bucket label. Per-instrument AMBER fields (shortable, lot/multiplier)
and TEAL runtime estimates (σ̂_t, ADV_i) are NOT here — they enter via inputs, not this table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Weekly annualization cadence, shared with src/convexity/system/metrics.PERIODS_PER_YEAR (365.25/7).
# LEVERAGE works on the per-observation (weekly) series exactly like SR_LCB_t and σ̂_t (§2.6), so
# annualization only ever appears in the vol-floor fallback (§4.2) and the §7 worked example.
PERIODS_PER_YEAR: float = 365.25 / 7.0  # ≈ 52.18 weekly periods per year (§7)


@dataclass(frozen=True)
class LeverageDefaults:
    """The RED constant set of LEVERAGE.md §3 (Invariant C-cfg — unreachable from instrument config).

    Illustrative v1 values with explicit rationale, one bucket each (§3). ``resolve_config`` is the
    single merge point; if any of these appears as a per-instrument field, that is the bug (§3).
    """

    # ---- fractional-Kelly dial (§2.3 c(2−c) derivation; only {0.25, 0.50} defined in v1) ----
    c_start: float = 0.25          # quarter-Kelly cold start / newly promoted (§6.2, P7)
    c_promoted: float = 0.50       # half-Kelly after the §6.3 sequential test passes
    c_grid: tuple[float, ...] = (0.25, 0.50)  # the two derived operating points (§2.3); no others

    # ---- volatility EWMA (§4.1–§4.2) ----
    h_vol: int = 8                 # EWMA half-life in WEEKS (§3): tracks a within-CPCV-group regime
    n_vol_min: int = 16            # event-count floor = 2·h_vol (§3, §4.2 rule n_vol_min ≥ 2·h_vol)
    vol_prior_annual: float = 0.28  # class-level realized-vol prior, ~28% annualized (§4.2, EMPIRICAL)
    vol_cold_start_multiple: float = 2.0  # cold-start (zero history) widens the prior (§4.2 step 2)

    # ---- block-bootstrap CI for SR_LCB_t (§2.6, §3 — SHARED with SYSTEM §7.3–§7.4) ----
    bootstrap_block_length: int = 6   # L_boot: pinned to SYSTEM's b*≈6.36≈6 (Politis & White 2004)
    bootstrap_n: int = 2_000          # bootstrap resample count (§2.6)
    bootstrap_ci_level: float = 0.90  # ONE-SIDED 90% lower bound convention for a sizing floor (§3)
    bootstrap_seed: int = 12345       # fixed, reproducible seed (§2.6, Invariant K-CI)

    # ---- capacity ceiling (§5) ----
    kappa: float = 0.02            # κ: max fraction of ADV_i a position may occupy (§3, ~$3M anchor)
    liquidity_cap_multiple: float = 2.0  # C ∈ {2,3}: capacity dial, OFF by default (§5.2, CAP-DIAL)
    liquidity_cap_book_threshold: float = 3_000_000.0  # $ book-size trigger for C (§5.2, ~$3M)

    # ---- drawdown governor ladder (§6.4, anchored to ~25% max-DD at 100% gross) ----
    dd_ref: float = 0.25           # cited EMPIRICAL max-DD anchor; every breakpoint a multiple of it
    governor_ladder: tuple[tuple[float, float], ...] = field(
        # (dd_upper_exclusive, m_t) rows; monotone non-increasing in dd. Breakpoints = 0.5×/1.0×/1.5×
        # of dd_ref, terminal at 1.6× (§6.4 illustrative v1 ladder). Last row is the terminal kill.
        default=(
            (0.125, 1.00),   # dd < 0.5·0.25 = 0.125: no de-risking (inside half the reference DD)
            (0.25, 0.60),    # 0.125 ≤ dd < 0.25 (1.0·ref): de-risk
            (0.375, 0.35),   # 0.25 ≤ dd < 0.375 (1.5·ref): de-risk hard
            (0.40, 0.00),    # dd ≥ 0.40 (1.6·ref): terminal threshold — full flat, governor kill
        )
    )
    dd_terminal: float = 0.40      # RED terminal drawdown threshold: m_t → 0 (§6.4 full flat)

    # ---- CUSUM promotion ratchet (§6.3, Page 1954 / Hawkins & Olwell 1998) ----
    k_ref: float = 0.25            # CUSUM reference offset in standardized-surprise units (§3, §6.3)
    h_promote: float = 5.0         # delivers ARL_0≈142wk (MC-verified) above the ≥52wk floor (§3, §6.3)
    n_min_weeks: int = 52          # minimum elapsed live-time floor before promotion eligible (§3)
    arl_0_target: float = 52.0     # target in-control ARL: ~1 false promotion / yr (§3, §6.3)
    # demote_z_threshold = -k_ref: the CUSUM ratchet's DOWN-STEP (de-lever) threshold, now anchored
    # in §3/§6.2. It is the SYMMETRIC MIRROR of the promotion CUSUM's own reference offset in its
    # degenerate IMMEDIATE (h_demote=0) form — the lower CUSUM min(0, S+z_k+k_ref) alarms on a single
    # step when z_k < -k_ref = -0.25 (§3, §6.2, Hawkins & Olwell 1998). Not a fabricated number: it
    # equals -k_ref by derivation (ARL_0≈8wk de-lever vs ≈52wk promote — the "fast to demote" half of
    # §6.2's asymmetry). Overridable at the call site (injection path retained, §6.2).
    demote_z_threshold: float = -0.25

    # ---- absolute circuit breaker (§3, §6.1) ----
    g_max: float = 1.5             # G_max: catches a corrupted σ̂_t forcing an absurd G_t (§3)

    def __post_init__(self) -> None:
        # §4.2 rule: n_vol_min ≥ 2·h_vol so ESS is meaningfully filled (§3 derives it from h_vol).
        if self.n_vol_min < 2 * self.h_vol:
            raise ValueError(
                f"n_vol_min ({self.n_vol_min}) must be >= 2*h_vol ({2 * self.h_vol}) (LEVERAGE §4.2)"
            )
        if not 0.0 < self.bootstrap_ci_level < 1.0:
            raise ValueError("bootstrap_ci_level must be in (0, 1) (LEVERAGE §3)")
