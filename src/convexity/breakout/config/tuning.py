"""TuningPolicy — the BLUE search space and promotion gates (BREAKOUT §2 BLUE / §10).

BLUE is a governed search over a small, predeclared candidate set (BREAKOUT §2). It selects ONE
global default, ONE asset-class override, or ONE liquidity-bucket override inside an asset class —
never arbitrary per-symbol RED values. The grids here mirror the §2 examples and feed
``tune/search_space.py``; their cardinality is the per-pillar trial count SYSTEM's deflation counts
(via ``BREAKOUT_PARAM_REGISTRY.blue_grid_cardinality()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TuningPolicy:
    """The finite BLUE candidate grids + the promotion gate levels (BREAKOUT §2/§10)."""

    # §2 BLUE candidate grids.
    windows_grid: tuple[tuple[int, ...], ...] = ((20, 40, 80), (40, 80, 160))
    cross_k_grid: tuple[float, ...] = (0.5, 1.0, 1.5)
    arl0_grid: tuple[int, ...] = (250, 500, 1000, 2000)
    compression_window_grid: tuple[int, ...] = (30, 60, 120)
    preset_grid: tuple[str, ...] = ("strict", "balanced", "early")
    horizons_grid: tuple[tuple[int, ...], ...] = ((15, 30, 60, 120),)

    # §10 promotion gates.
    min_oos_signals: int = 30  # insufficient OOS evidence => shadow not approved
    fdr_alpha: float = 0.10  # cross-sectional FDR level (Invariant L)
    require_beat_baselines: bool = True
    # the baselines the detector must beat on OOS convexity AND false-break rate.
    beaten_baselines: tuple[str, ...] = field(
        default_factory=lambda: ("donchian", "atr_keltner", "cusum", "random")
    )

    def blue_cardinality(self) -> int:
        """Product of the searched BLUE grid sizes — the per-pillar trial count fed to the ledger."""
        return (
            len(self.windows_grid)
            * len(self.cross_k_grid)
            * len(self.arl0_grid)
            * len(self.preset_grid)
        )
