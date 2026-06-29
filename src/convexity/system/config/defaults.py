"""SystemDefaults — every RED composition/validation constant (SYSTEM §5).

SYSTEM holds NO model parameters; these are all *composition policy* and *validation* constants,
RED and shared, unreachable from per-instrument config (Invariant C). They group into:

  * composition  — precedence order, cadence τ_sel, deadband δ, rotation hysteresis Δ_rank,
                   fixed-point r_max/η, the book-heat ceiling H_book (shared with LEVERAGE), the
                   dual-equity ladder;
  * horizon      — the contractual hard cap H_sys (Invariant HC) and the triple-barrier multiples;
  * embargo/CV   — warmup, the module label horizons (union embargo, Section 6.2), the CPCV knobs;
  * deflation    — the DSR gate confidence (→ Z_gate), N_sys guards (K_min, var_floor, rho_cl),
                   the single annualization cadence ppy (Section 7.2a), the MinTRL n_obs_floor;
  * re-promotion — τ_promote, τ_forward, the paired forward-gate block length + CI level;
  * overnight    — the K_max / distinct-module-config cap and T_night (Section 9.2);
  * exec-stub    — the conservatism multiples (Section 6.1b).

The PARAM_REGISTRY (bottom) declares the bucket of every name so the Invariant-C firewall test
(:func:`convexity.core.params.assert_no_red_in_amber`) can prove no RED/BLUE name leaks onto the
AMBER instrument metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry

# The fixed precedence order of the §2 pipeline. RED policy; the entire conflict-resolution logic.
PRECEDENCE_ORDER: tuple[str, ...] = (
    "governor_kill",       # (A)
    "open_positions",      # (B) trend exit > ceiling collapse
    "reconcile_selection",  # (C) hold-incumbent firewall
    "new_signal_admission",  # (D)
    "iterate_to_consistency",  # (E)
    "quantize_dual_cap",   # (F)
    "emit",                # (G)
)


@dataclass(frozen=True)
class SystemDefaults:
    """All RED composition + validation constants (SYSTEM §5). Defaults are the §7.3b/§9.2 RED
    values; every one is shared, never per-symbol."""

    # ---- composition (Section 2/3) ----
    precedence_order: tuple[str, ...] = PRECEDENCE_ORDER
    tau_sel: str = "daily"            # scheduled selection↔leverage cadence (Section 3.5)
    deadband_weight: float = 0.02     # δ in risk-weight terms; sub-δ target change emits no order
    deadband_lots: int = 1            # δ in lots; one-lot jitter is suppressed (Invariant CAD)
    delta_rank: float = 2.0           # Δ_rank rotation hysteresis (Section 4); larger ⇒ rarer rotation
    r_max: int = 5                    # damped continuous-relaxation step cap (Section 3.3)
    eta: float = 0.5                  # damping η ∈ (0,1] — firmly non-expansive averaged projection
    iterate_tol: float = 5e-3         # ‖w^(r)−w^(r-1)‖ tolerance for converged (Section 3.3)
    h_book: float = 0.20              # book-heat ceiling H_book (shared with LEVERAGE), sum MCTR cap
    equity_ladder: tuple[float, ...] = (0.02, 0.04, 0.06)  # dual-equity planned-loss ladder rungs

    # ---- horizon + triple-barrier (Section 6.1, Invariant HC) ----
    h_sys: int = 10_080               # contractual hard cap in bars (7 trading days @ 1-min ≈ 10080)
    b_up: float = 2.0                 # upper barrier multiple +b_up·σ_horizon
    b_dn: float = 1.0                 # lower barrier multiple -b_dn·σ_horizon

    # ---- embargo / CV (Section 6.2/6.4) ----
    max_warmup_bars: int = 1_440      # longest causal warmup across modules (1 trading day @ 1-min)
    module_label_horizons: tuple[int, ...] = (1_440, 5_040, 10_080)  # breakout/trend/system horizons
    n_groups: int = 6                 # CPCV partition group count
    k_test: int = 2                   # CPCV test groups per arrangement ⇒ C(6,2)=15
    max_paths: int = 8                # sampled CPCV paths (deterministic seed, recorded)
    cpcv_seed: int = 12345            # the recorded sampling seed (Section 6.4)

    # ---- deflation / PSR-DSR gate (Section 7) ----
    dsr_threshold: float = 0.95       # the DSR gate confidence (RED); Z_gate = Φ⁻¹(threshold)
    k_min: int = 8                    # minimum distinct de-correlated trials before PASS (guard 1)
    var_floor_enabled: bool = True    # apply the Lo-2002 per-observation variance floor (guard 1)
    rho_cl: float = 0.7               # clustering threshold (TIGHT ⇒ more clusters / larger N_sys)
    ppy: float = 36.0                 # periods_per_H_sys: the SINGLE annualization cadence (~36/yr)
    n_obs_floor: int = 24             # absolute minimum independent-observation count (guard 3)
    psr_denom_eps: float = 1e-6       # PSR denominator floor to avoid division blow-up

    # ---- re-promotion / forward gate (Section 7.5, Invariant RP) ----
    tau_promote: str = "monthly"      # live-config swap cadence (NOT every night)
    tau_forward_obs: int = 60         # forward-block length in non-overlapping observations
    forward_block_length: int = 4     # paired-difference block-bootstrap block length (Section 8.1)
    forward_ci_level: float = 0.95    # paired-difference CI level (must exclude zero)
    forward_n_boot: int = 2_000       # bootstrap resample count

    # ---- robustness (Section 8.1) ----
    drop_top_n: int = 5               # drop-top-N profitable / worst trades count
    bootstrap_block_length: int = 4   # dependence-preserving block length (no iid bootstrap)
    bootstrap_ci_level: float = 0.95
    bootstrap_n: int = 2_000

    # ---- overnight compute budget (Section 9.2, Invariant CB-BUDGET) ----
    k_max: int = 24                   # max joint configs the overnight job may score
    max_distinct_module_configs: int = 8  # ≤ this many change module evidence (cache misses)
    t_night_s: float = 28_800.0       # T_night wall-clock budget (8 h)
    t_mod_s: float = 90.0             # per-module evidence wall-time per config per path
    t_comp_s: float = 5.0             # composition+labeling+deflation wall-time per config per path

    # ---- execution stub conservatism (Section 6.1b, Invariant XC) ----
    impact_safety_multiple: float = 1.5  # scale-UP factor on the impact model
    slippage_extreme_fraction: float = 1.0  # 1.0 ⇒ worst-in-bar fill (full high/low excursion)

    @property
    def z_gate(self) -> float:
        """Z_gate = Φ⁻¹(dsr_threshold) — the gate confidence as a normal quantile (Section 7.3a)."""
        from scipy.stats import norm

        return float(norm.ppf(self.dsr_threshold))


def system_embargo_bars(defaults: SystemDefaults) -> int:
    """The union embargo width (Section 6.2, Invariants SC/HC).

    ``max(H_sys, max module label horizon) + max warmup`` — the union (max) over every module's own
    label horizon, with the system triple-barrier horizon bounded by the contractual cap ``H_sys``
    (never a mean/typical realized horizon, which would leak a long trend-ride across the fold).
    """
    horizon = max(defaults.h_sys, max(defaults.module_label_horizons))
    return horizon + defaults.max_warmup_bars


SYSTEM_PARAM_REGISTRY = StaticParamRegistry(
    (
        ParamSpec("precedence_order", Bucket.RED, "fixed §2 pipeline order", "SYSTEM §5"),
        ParamSpec("tau_sel", Bucket.RED, "selection↔leverage cadence", "SYSTEM §3.5"),
        ParamSpec("deadband_weight", Bucket.RED, "δ risk-weight deadband", "SYSTEM §3.5"),
        ParamSpec("deadband_lots", Bucket.RED, "δ lot deadband", "SYSTEM §3.5"),
        ParamSpec("delta_rank", Bucket.RED, "Δ_rank rotation hysteresis", "SYSTEM §4"),
        ParamSpec("r_max", Bucket.RED, "fixed-point step cap", "SYSTEM §3.3"),
        ParamSpec("eta", Bucket.RED, "damping η", "SYSTEM §3.3"),
        ParamSpec("h_book", Bucket.RED, "book-heat ceiling", "SYSTEM §3"),
        ParamSpec("h_sys", Bucket.RED, "contractual horizon cap", "SYSTEM §6.1"),
        ParamSpec("b_up", Bucket.RED, "upper barrier multiple", "SYSTEM §6.1"),
        ParamSpec("b_dn", Bucket.RED, "lower barrier multiple", "SYSTEM §6.1"),
        ParamSpec("n_groups", Bucket.RED, "CPCV group count", "SYSTEM §6.4"),
        ParamSpec("k_test", Bucket.RED, "CPCV test groups", "SYSTEM §6.4"),
        ParamSpec("max_paths", Bucket.RED, "CPCV sampled paths", "SYSTEM §6.4"),
        ParamSpec("dsr_threshold", Bucket.RED, "DSR gate confidence", "SYSTEM §7.3"),
        ParamSpec("k_min", Bucket.RED, "min distinct trials", "SYSTEM §7.3a"),
        ParamSpec("rho_cl", Bucket.RED, "clustering threshold", "SYSTEM §7.1"),
        ParamSpec("ppy", Bucket.RED, "annualization cadence", "SYSTEM §7.2a"),
        ParamSpec("n_obs_floor", Bucket.RED, "MinTRL obs floor", "SYSTEM §7.3a"),
        ParamSpec("tau_promote", Bucket.RED, "re-promotion cadence", "SYSTEM §7.5"),
        ParamSpec("forward_block_length", Bucket.RED, "paired forward block length", "SYSTEM §7.5"),
        ParamSpec("k_max", Bucket.RED, "overnight config cap", "SYSTEM §9.2"),
        ParamSpec("t_night_s", Bucket.RED, "overnight wall-clock budget", "SYSTEM §9.2"),
        ParamSpec("impact_safety_multiple", Bucket.RED, "exec-stub impact scale-up", "SYSTEM §6.1b"),
        # BLUE composition grids (enumerated in tuning.py; declared here for the firewall)
        ParamSpec(
            "tau_sel_grid", Bucket.BLUE, "cadence candidate set", "SYSTEM §5",
            blue_grid=("daily", "session"),
        ),
        ParamSpec(
            "deadband_grid", Bucket.BLUE, "deadband candidate set", "SYSTEM §5",
            blue_grid=(0.01, 0.02, 0.04),
        ),
        ParamSpec(
            "delta_rank_grid", Bucket.BLUE, "rotation hysteresis candidate set", "SYSTEM §5",
            blue_grid=(1.0, 2.0, 3.0),
        ),
    )
)
