"""Register REGIME's BLUE grid into SYSTEM's trial ledger (REGIME §4.1, §9, Invariants NOFIT/V).

REGIME's degrees of freedom must be PAID FOR in SYSTEM's joint deflated Sharpe (SYSTEM P6). This
module computes the registered BLUE-grid cardinality ``K_regime`` and asserts the hard cap
``K_regime ≤ K_regime_max`` (§4.1). Critically the grid is a SMALL BOUNDED NEIGHBOURHOOD — a
one-step grid on each of the five searched families, NOT a full Cartesian explosion — so
``K_regime`` is the SUM of the per-family candidate counts, not their product. The Ledoit–Wolf
intensities are NOT here (TEAL, closed-form — they do not enter ``K_regime``).

Firewalled (Invariant V): the live panel/measure/state layers never import this module, and no code
path selects a breakpoint by maximising in-sample PnL (Invariant NOFIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.regime.config.defaults import BLUE_GRID, PolicyDefaults

# The FIVE searched families (REGIME §4.1): kappa, the per-measure STRESSED-entry level (4 measures),
# D_min, tau_regime, and the percentile-band variants. The per-measure STRESSED-entry families are
# counted once each.
_SEARCHED_FAMILIES: tuple[str, ...] = (
    "kappa",
    "vol_stressed_entry",
    "corr_stressed_entry",
    "turb_stressed_entry",
    "liq_stressed_entry",
    "d_min",
    "tau_regime",
)


@dataclass(frozen=True)
class BlueGridRegistration:
    """The registered BLUE grid handed to SYSTEM's trial ledger (REGIME §4.1)."""

    k_regime: int  # one-step-neighbourhood cardinality (SUM, not Cartesian product)
    k_regime_max: int
    families: Mapping[str, int]  # family -> candidate count


def k_regime(grid: Mapping[str, tuple[object, ...]] = BLUE_GRID) -> int:
    """The registered ``K_regime`` — SUM of per-family candidate counts (REGIME §4.1, NOT a product).

    A one-step grid on each searched family, never a Cartesian explosion: a config differs from the
    RED canonical in at most one family, so the trial count is the union of one-step deviations, not
    the product of all grids. This is what SYSTEM's joint deflation counts.
    """
    return sum(len(grid[f]) for f in _SEARCHED_FAMILIES if f in grid)


def register_blue_grid(cfg: PolicyDefaults, grid: Mapping[str, tuple[object, ...]] = BLUE_GRID) -> BlueGridRegistration:
    """Build the BLUE-grid registration + assert the hard cap ``K_regime ≤ K_regime_max`` (§4.1).

    Raises if the registered cardinality exceeds the cap — REGIME cannot silently grow its degrees of
    freedom and dilute SYSTEM's deflation (Invariant NOFIT, extended).
    """
    k = k_regime(grid)
    if k > cfg.k_regime_max:
        raise AssertionError(
            f"K_regime={k} exceeds K_regime_max={cfg.k_regime_max} (REGIME §4.1, Invariant NOFIT)"
        )
    families = {f: len(grid[f]) for f in _SEARCHED_FAMILIES if f in grid}
    return BlueGridRegistration(k_regime=k, k_regime_max=cfg.k_regime_max, families=families)
