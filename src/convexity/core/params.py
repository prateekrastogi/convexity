"""Governance buckets + a typed parameter registry shared by every pillar.

Every spec (BREAKOUT/TREND/LEVERAGE/EXECUTION/COST/SYSTEM/OPS/REGIME §2/§4) classifies each
parameter into one of four governance buckets. The bucket decides *where* a value may enter the
system and *who* may set it — this is the load-bearing anti-overfit discipline, enforced by the
``resolve_config(defaults=...)`` single-merge-point pattern every pillar implements (see each
spec's Invariant C: "RED unreachable from per-instrument config").

  RED   — shared economic/modelling constants. Set once, globally; never per-symbol. Reachable
          ONLY through ``resolve_config(defaults=...)``. Examples: Kelly fraction φ, governor
          breakpoints, CUSUM (k,h), the g(s) map, chase fraction χ, edge_buffer.
  AMBER — supplied per-instrument market metadata, NOT optimised by any pillar: tick/lot/
          multiplier, venue, calendar/session, fee schedule, funding interval/cap. May appear on
          an ``InstrumentSpec`` — but a RED/BLUE value appearing there is THE bug each Invariant C
          test hunts for.
  BLUE  — finite, pre-declared, validated candidate GRIDS, searched OFFLINE at the class level
          (never per-symbol on realized P&L) and promoted only through SYSTEM's deflated-Sharpe /
          purged-CPCV gate. Each BLUE grid's cardinality is registered into SYSTEM's persistent
          trial ledger so the multiple-testing deflation counts it (SYSTEM Inv SG/DSR).
  TEAL  — fitted per instrument from its OWN past data at runtime, strictly causal (F_{t-1}):
          rolling vol/σ, EVT tail, conditional-MAE, empirical-null buckets, Ledoit-Wolf shrinkage
          intensity. Never a hand knob, never full-sample, never P&L-fit.

This module gives those four buckets a typed home and a tiny registry/protocol so a pillar can
*declare* the bucket of every parameter it consumes. The registry is descriptive (it documents +
lets tests assert governance), not a runtime config store — each pillar still owns its own frozen
``Defaults``/``ResolvedConfig`` dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class Bucket(str, Enum):
    """The four governance buckets. ``str`` mixin so values serialize/round-trip cleanly."""

    RED = "red"
    AMBER = "amber"
    BLUE = "blue"
    TEAL = "teal"


@dataclass(frozen=True)
class ParamSpec:
    """One declared parameter: its name, governance bucket, and a one-line meaning.

    ``spec_ref`` cites the owning spec + section (e.g. "LEVERAGE §8.1") so a governance test can
    trace a value back to its authority. ``blue_grid`` records the finite candidate set for BLUE
    params (used by SYSTEM's trial-ledger cardinality count); ``None`` for non-BLUE.
    """

    name: str
    bucket: Bucket
    meaning: str
    spec_ref: str
    blue_grid: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if self.blue_grid is not None and self.bucket is not Bucket.BLUE:
            raise ValueError(f"{self.name}: blue_grid given for non-BLUE bucket {self.bucket}")


@runtime_checkable
class ParamRegistry(Protocol):
    """Each pillar exposes one of these so governance tests can introspect its parameter set.

    Implemented as a plain mapping wrapper (see ``StaticParamRegistry``); the Protocol exists so
    SYSTEM's trial-ledger builder and the per-pillar Invariant-C tests can depend on the shape, not
    a concrete class.
    """

    def params(self) -> tuple[ParamSpec, ...]:
        """Every parameter this pillar consumes, with its governance bucket."""
        ...


@dataclass(frozen=True)
class StaticParamRegistry:
    """Concrete ``ParamRegistry`` backed by a frozen tuple of ``ParamSpec``."""

    specs: tuple[ParamSpec, ...]

    def params(self) -> tuple[ParamSpec, ...]:
        return self.specs

    def of_bucket(self, bucket: Bucket) -> tuple[ParamSpec, ...]:
        """All params in one bucket — e.g. ``.of_bucket(Bucket.RED)`` for an Inv-C assertion."""
        return tuple(p for p in self.specs if p.bucket is bucket)

    def blue_grid_cardinality(self) -> int:
        """Product of BLUE grid sizes — the per-pillar trial count fed to SYSTEM's ledger.

        Returns 1 when the pillar has no BLUE params (a single fixed config). SYSTEM caps each
        pillar's cardinality (e.g. REGIME K_regime_max, SYSTEM K_max) and counts it in the joint
        deflation (SYSTEM Inv DSR/SG).
        """
        card = 1
        for p in self.specs:
            if p.bucket is Bucket.BLUE and p.blue_grid is not None:
                card *= max(len(p.blue_grid), 1)
        return card


def assert_no_red_in_amber(amber_field_names: frozenset[str], registry: ParamRegistry) -> None:
    """Invariant-C helper: raise if any RED/BLUE param name leaks into the AMBER field set.

    Pillars call this in a test (or at resolve time) passing the field names of their
    per-instrument ``InstrumentSpec`` — a RED/BLUE name appearing there is the overfit bug every
    spec's Invariant C forbids.
    """
    forbidden = {
        p.name for p in registry.params() if p.bucket in (Bucket.RED, Bucket.BLUE)
    }
    leaked = forbidden & set(amber_field_names)
    if leaked:
        raise AssertionError(
            "RED/BLUE parameters reachable from per-instrument config "
            f"(Invariant C): {sorted(leaked)}"
        )
