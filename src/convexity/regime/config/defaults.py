"""RED governance constants for the REGIME detector (REGIME.md §3.4, §6).

Every field here is a RED "shared economic/modelling constant" (REGIME.md §6): it is reachable
ONLY through :func:`regime.config.resolve.resolve_config`, never through AMBER per-sleeve metadata
and never through a per-instrument axis (REGIME has no per-instrument axis at all — §3.4 Invariant
C-cfg, §0 P2). The values below are the spec's own RED candidate choices; the illustrative
``(mu_0, sigma)`` calibration pair of §3.5 is deliberately NOT stored here — it is a TEAL quantity
fitted from the frozen in-control training window at calibration time (§3.4, §6 TEAL bucket), and
lives in a calibration artifact, not in this RED defaults table.

``K_regime_max`` is net-new here: the spec (§6) records that it is named in ``docs/ARCHITECTURE.md``
prose and ``core/params.py``'s docstring but has NO field/constant/ParamSpec anywhere in the repo,
so REGIME.md is "the first place its VALUE and role are actually specified." It caps how many
``(Delta_target, ARL_0_target, dd_trigger, dd_reset)`` combinations one calibration epoch may search
(§6, §7.4), feeding the ONE SYSTEM trial ledger (Invariant LEDGER).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import Bucket, ParamSpec, StaticParamRegistry


@dataclass(frozen=True)
class RegimeDefaults:
    """The frozen RED constant set the resolved config exposes (REGIME.md §3.4, §6).

    All fields are book-level scalars (no per-instrument axis, §3.4 Invariant C-cfg). ``Delta_target``
    and ``ARL_0_target`` are the two governing RED candidates from which ``k`` and ``h`` are derived
    (§3.4); the remaining fields govern the WATCH sub-threshold (``h_warn_frac``), the re-arm
    hysteresis (``n_confirm``), the early-history breadth floor (``n_min_dispersion``, §8.1), the
    DD-stop ablation power floor (``n_weeks_min_cpcv``, §5.4), the bleed-detection tolerance
    (``bleed_return_tol``, §5.4 gate (b)), and the overnight calibration budget (``K_regime_max``).
    """

    # --- §3.4 / §6: the two governing RED candidates (BLUE-searchable, §6 BLUE grids) ---
    delta_target: float = 1.5          # standardized shift size to detect (§3.5: detect 1.5-sigma)
    arl_0_target: float = 104.0        # in-control ARL, weeks (~2yr between false alarms, §3.5)

    # --- §4.1 / §4.2: state-machine governance ---
    h_warn_frac: float = 0.6           # WATCH sub-threshold as a fraction of h (§4.1: h_warn = 0.6h)
    n_confirm: int = 2                 # re-arm hysteresis: consecutive calm weeks to de-escalate (§4.2)

    # --- §4.1 state -> g_t band floors (the conservative live value emitted per state, §4.1) ---
    g_healthy: float = 1.00
    g_watch: float = 0.75
    g_stressed: float = 0.40
    g_halt: float = 0.00

    # --- §8.1: early-history breadth floor ---
    n_min_dispersion: int = 15         # |U_tau| below this -> WATCH, D_t excluded from CUSUM (§8.1)

    # --- §3.5 Step 5 / Invariant ARL: empirical ARL check tolerance & bootstrap governance ---
    arl_tol_frac: float = 0.20         # empirical ARL_0 must land within +/- this fraction of target
    arl_boot_n: int = 2000             # moving-block-bootstrap resamples for the empirical ARL check
    arl_boot_block: int = 6            # block length (weeks); pinned near SYSTEM's L_boot=6 (§7.3-7.4)

    # --- §5.4 / Invariant STOP-XFER: DD-stop promotion gate governance ---
    n_weeks_min_cpcv: int = 156        # min live history (weeks) for a DD-stop frac_pos to count (§5.4)
    bleed_return_tol: float = 0.02     # §5.4 gate (b): stopped-below-raw window-return tolerance

    # --- §6 / §7.4: overnight calibration compute cap feeding the ONE SYSTEM ledger ---
    k_regime_max: int = 27             # max (Delta_target, ARL_0_target, dd_trigger, dd_reset) combos

    @property
    def h_warn(self) -> float:
        """WATCH sub-threshold as an absolute value once ``h`` is known (caller passes ``h``)."""
        raise NotImplementedError(
            "h_warn is h * h_warn_frac; compute at the call site where the per-statistic h is known "
            "(REGIME.md §4.1). This property exists only to document the relationship."
        )


# REGIME's param registry (REGIME.md §3.4 Invariant C-cfg / §6 governance buckets).
#
# §3.4/§6 classify Delta_target and ARL_0_target as RED — "shared economic/modelling constants" — and
# the resulting (k, h) as RED too. The FROZEN LIVE values (RegimeDefaults.delta_target=1.5,
# arl_0_target=104.0) are therefore registered RED here, NOT BLUE: registering the live constant
# itself as BLUE would bucket-mislabel the deployed RED value (§6). What §6 lists under BLUE is the
# finite SEARCH GRID — Delta_target in {1.0, 1.5, 2.0}, ARL_0_target in {52, 104, 208} — searched
# OFFLINE under SYSTEM's DSR/CPCV gate, never per-symbol. Those grids are registered as SEPARATE BLUE
# entries (delta_target_grid / arl_0_target_grid), exactly mirroring COST's established pattern
# (funding_ewma_half_life_stamps=RED live value + h_grid=BLUE search grid, cost/config/defaults.py).
# The DD-stop (dd_trigger, dd_reset) BLUE pairs live with the ablation (§5.4) and are not stored as
# RegimeDefaults fields because a DD-stop is a nice-to-have overlay, not a required detector param
# (§5.4 step 5). K_regime_max (RED) caps the product of the BLUE grid cardinalities one calibration
# epoch may search (§6, §7.4 Invariant LEDGER).
REGIME_PARAM_REGISTRY = StaticParamRegistry(
    (
        # RED live constants (the deployed frozen values; §3.4/§6).
        ParamSpec("delta_target", Bucket.RED, "standardized shift to detect (live)", "REGIME §3.4"),
        ParamSpec("arl_0_target", Bucket.RED, "in-control ARL, weeks (live)", "REGIME §3.4"),
        ParamSpec("h_warn_frac", Bucket.RED, "WATCH sub-threshold fraction of h", "REGIME"),
        ParamSpec("n_confirm", Bucket.RED, "re-arm hysteresis weeks", "REGIME"),
        ParamSpec("n_min_dispersion", Bucket.RED, "early-history breadth floor", "REGIME"),
        ParamSpec("arl_tol_frac", Bucket.RED, "empirical ARL tolerance fraction", "REGIME"),
        ParamSpec("n_weeks_min_cpcv", Bucket.RED, "DD-stop ablation min history", "REGIME"),
        ParamSpec("bleed_return_tol", Bucket.RED, "bleed-window return tolerance", "REGIME"),
        ParamSpec("k_regime_max", Bucket.RED, "overnight calibration combo cap", "REGIME"),
        # BLUE class-level search grids (§6): searched offline under DSR/CPCV, never per instrument.
        ParamSpec("delta_target_grid", Bucket.BLUE, "standardized-shift candidates", "REGIME §6",
                  blue_grid=(1.0, 1.5, 2.0)),
        ParamSpec("arl_0_target_grid", Bucket.BLUE, "in-control ARL candidates", "REGIME §6",
                  blue_grid=(52, 104, 208)),
    )
)
