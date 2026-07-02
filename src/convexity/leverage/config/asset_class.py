"""Per-asset-class capacity presets (LEVERAGE.md §9 layout, §5 capacity ceiling, §3 RED).

``AssetClassCapacityPreset`` supplies validated CLASS-LEVEL fallbacks for the two capacity dials
``κ`` (per-position ADV fraction) and ``C`` (the liquidity-cap multiple) when a per-class override
of the RED default is warranted (§5). It is NOT a per-instrument config (Invariant C-cfg): the key
is a ``core.types.AssetClass`` bucket, never an instrument symbol, so it cannot smuggle a
per-symbol dial into the sizing chain (§3). ``resolve_config`` remains the single merge point; this
preset only ever narrows the RED default to a class, never below it.

By design v1 ships EMPTY overrides (the RED ``LeverageDefaults`` κ/C hold for every class); the
type exists so a future class-level tightening (e.g. thinner crypto-spot books) has a validated,
C-cfg-respecting home rather than leaking into instrument config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from convexity.core.types import AssetClass
from convexity.leverage.config.defaults import LeverageDefaults


@dataclass(frozen=True)
class AssetClassCapacityPreset:
    """A validated class-level κ/C fallback (LEVERAGE.md §5, §3 RED, Invariant C-cfg).

    ``kappa``/``liquidity_cap_multiple`` mirror the RED ``LeverageDefaults`` fields but scoped to a
    single ``AssetClass`` bucket. Both are validated on construction to the same ranges §3 asserts
    for the RED defaults (``0 < κ ≤ 1``, ``C ∈ {2, 3}``) so a preset can never widen capacity past
    what §5.2 permits. This is a RED-adjacent CLASS constant, never reachable per instrument.
    """

    asset_class: AssetClass
    kappa: float
    liquidity_cap_multiple: float

    def __post_init__(self) -> None:
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError(f"kappa {self.kappa} must be in (0, 1] (LEVERAGE §3/§5.1)")
        if self.liquidity_cap_multiple not in (2.0, 3.0):
            raise ValueError(
                f"liquidity_cap_multiple {self.liquidity_cap_multiple} must be in {{2, 3}} "
                "(LEVERAGE §5.2 — C ∈ {2, 3})"
            )


@dataclass(frozen=True)
class AssetClassCapacityTable:
    """Class → preset lookup, defaulting to the RED ``LeverageDefaults`` (LEVERAGE.md §3, §5).

    ``resolve(asset_class, defaults)`` returns the class preset when one is registered, else the
    RED default κ/C — so an unlisted class transparently uses the shared RED dial (§3). The table
    is the ONLY place a class-level κ/C narrowing may live; nothing here is keyed by symbol
    (Invariant C-cfg). v1 registers no overrides (``presets`` empty), so every class resolves to
    the RED default exactly.
    """

    presets: Mapping[AssetClass, AssetClassCapacityPreset] = field(default_factory=dict)

    def resolve(
        self, asset_class: AssetClass, defaults: LeverageDefaults
    ) -> tuple[float, float]:
        """Resolve ``(κ, C)`` for a class, falling back to the RED default (LEVERAGE.md §3, §5.2)."""
        preset = self.presets.get(asset_class)
        if preset is None:
            return defaults.kappa, defaults.liquidity_cap_multiple
        return preset.kappa, preset.liquidity_cap_multiple
