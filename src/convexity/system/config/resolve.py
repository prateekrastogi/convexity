"""resolve_config — the single merge point where RED enters (SYSTEM §5, Invariant C).

The ONE place the RED :class:`SystemDefaults` are merged with the BLUE :class:`CompositionPoint`
and the AMBER :class:`SystemUniverse` into the frozen :class:`ResolvedConfig` the compose/validate
layers read. RED is reachable ONLY through ``defaults=``; the BLUE composition coordinate comes
through the SEPARATE ``composition=`` argument; the universe carries AMBER only. A RED/BLUE value on
the per-instrument metadata is the bug this firewall catches — ``assert_no_red_in_amber`` runs over
the instrument field names against :data:`SYSTEM_PARAM_REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.core.params import assert_no_red_in_amber

from convexity.system.config.defaults import SYSTEM_PARAM_REGISTRY, SystemDefaults, system_embargo_bars
from convexity.system.config.tuning import CompositionPoint
from convexity.system.config.universe import INSTRUMENT_META_FIELDS, SystemUniverse


@dataclass(frozen=True)
class ResolvedConfig:
    """The frozen, fully-merged config every compose/validate module reads (SYSTEM §5).

    RED constants live on ``defaults`` (untouched by per-instrument config); the BLUE composition
    coordinate and the AMBER universe ride alongside. Helper properties surface the few derived
    values (embargo width, the composition fields the orchestrator actually branches on) so callers
    never re-merge.
    """

    defaults: SystemDefaults
    composition: CompositionPoint
    universe: SystemUniverse

    # ---- composition fields (BLUE coordinate overrides the RED default for the search axis) ----
    @property
    def tau_sel(self) -> str:
        return self.composition.tau_sel

    @property
    def deadband_weight(self) -> float:
        return self.composition.deadband_weight

    @property
    def delta_rank(self) -> float:
        return self.composition.delta_rank

    @property
    def budget_priority(self) -> str:
        return self.composition.budget_priority

    # ---- derived RED quantities ----
    @property
    def h_sys(self) -> int:
        return self.defaults.h_sys

    @property
    def h_book(self) -> float:
        return self.defaults.h_book

    @property
    def embargo_bars(self) -> int:
        """The union embargo width keyed to ``H_sys`` (Section 6.2, Invariants SC/HC)."""
        return system_embargo_bars(self.defaults)

    def config_hash(self, module_versions) -> str:
        """The content-addressed hash for the EvidenceCache / trial ledger (Invariant CA/SG)."""
        return self.composition.config_hash(module_versions)


def resolve_config(
    universe: SystemUniverse,
    *,
    defaults: SystemDefaults | None = None,
    composition: CompositionPoint | None = None,
) -> ResolvedConfig:
    """Merge RED defaults + BLUE composition coordinate + AMBER universe into one ``ResolvedConfig``.

    RED enters ONLY through ``defaults`` (Invariant C). The BLUE composition coordinate is the
    SEPARATE validated channel; if omitted, the default (live, frozen) coordinate is used. Runs
    ``assert_no_red_in_amber`` so a RED/BLUE name leaking onto the per-instrument metadata raises
    immediately.
    """
    assert_no_red_in_amber(INSTRUMENT_META_FIELDS, SYSTEM_PARAM_REGISTRY)
    red = defaults if defaults is not None else SystemDefaults()
    blue = composition if composition is not None else CompositionPoint()
    return ResolvedConfig(defaults=red, composition=blue, universe=universe)
