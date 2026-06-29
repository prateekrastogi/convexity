"""resolve_config — the SINGLE merge point where RED enters BREAKOUT (BREAKOUT §1 / Invariant D).

The ONE place ``DetectorDefaults`` (RED) is merged with the per-class ``AssetClassOverride`` and the
per-instrument AMBER ``InstrumentSpec`` into the frozen ``ResolvedConfig`` every BREAKOUT module
reads. RED is reachable ONLY through the ``defaults=`` argument; per-class overrides come through the
SEPARATE ``override=`` argument; the instrument spec carries AMBER only.

This module also OWNS the barrier-family disablement (BREAKOUT §2/§3 barrier-family policy): the
session-anchored opening-range / anchored-session variants are DISABLED for any instrument whose
``calendar_id`` has no session boundary (24/7 crypto), falling back deterministically to rolling
Donchian — so the same code path is deterministic across the universe (no silent undefined anchor).
The chosen family is recorded on ``ResolvedConfig`` and is subject to the same OOS promotion gate as
W and k (§10).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from convexity.core.params import assert_no_red_in_amber
from convexity.core.types import InstrumentSpec

from convexity.breakout.config.asset_class import (
    DEFAULT_OVERRIDES,
    SESSION_ANCHORED_FAMILIES,
    AssetClassOverride,
    BarrierFamily,
)
from convexity.breakout.config.defaults import (
    BREAKOUT_PARAM_REGISTRY,
    PRESETS,
    DetectorDefaults,
)
from convexity.breakout.config.instrument import INSTRUMENT_SPEC_FIELDS

# calendar_ids that DECLARE a session boundary (vs 24/7). Anything not here is treated 24/7.
_SESSION_CALENDARS: frozenset[str] = frozenset(
    {"us_equity_rth", "us_equity_eth", "cme_globex", "comex", "lme", "nyse", "nasdaq"}
)


def has_session_boundary(calendar_id: str) -> bool:
    """True iff the calendar declares sessions (so a session-anchored barrier is well-defined)."""
    return bool(calendar_id) and calendar_id in _SESSION_CALENDARS


@dataclass(frozen=True)
class ResolvedConfig:
    """The frozen, fully-merged config every BREAKOUT module reads (§1/§4).

    RED constants live on ``defaults`` (untouched by per-instrument config); the class override and
    the AMBER instrument spec ride alongside. ``barrier_family`` is the FINAL family after the §2/§3
    disablement; ``weights`` is the resolved §8 combiner preset.
    """

    defaults: DetectorDefaults
    override: AssetClassOverride
    instrument: InstrumentSpec
    barrier_family: BarrierFamily
    weights: dict[str, float]

    # --- resolved scalars (class override beats default, default otherwise) ---
    @property
    def cross_k(self) -> float:
        return self.override.cross_k if self.override.cross_k is not None else self.defaults.cross_k

    @property
    def windows(self) -> tuple[int, ...]:
        return self.override.windows if self.override.windows is not None else self.defaults.windows

    @property
    def preset(self) -> str:
        return self.override.preset if self.override.preset is not None else self.defaults.preset

    @property
    def session_anchored_available(self) -> bool:
        """Whether the session-anchored barrier variant survived the §2/§3 disablement."""
        return self.barrier_family in SESSION_ANCHORED_FAMILIES


def resolve_config(
    instrument: InstrumentSpec,
    *,
    defaults: DetectorDefaults | None = None,
    override: AssetClassOverride | None = None,
) -> ResolvedConfig:
    """Merge RED defaults + class override + AMBER instrument into one frozen config (§1).

    ``defaults`` is the SOLE RED entry point (Invariant D). ``override`` is the SEPARATE per-class
    channel; defaulted from the instrument's asset class. Raises if any RED/BLUE name leaks into the
    instrument's AMBER field set. Owns the barrier-family disablement (§2/§3).
    """
    # Invariant D: RED/BLUE must be unreachable from the per-instrument spec.
    assert_no_red_in_amber(INSTRUMENT_SPEC_FIELDS, BREAKOUT_PARAM_REGISTRY)

    defaults = defaults if defaults is not None else DetectorDefaults()
    override = override if override is not None else DEFAULT_OVERRIDES[instrument.asset_class]
    if override.asset_class != instrument.asset_class:
        raise ValueError(
            f"override class {override.asset_class!r} != instrument class {instrument.asset_class!r}"
        )

    # §2/§3 barrier-family disablement: a session-anchored family is DISABLED for an instrument with
    # no session boundary, falling back deterministically to rolling Donchian.
    family: BarrierFamily = override.barrier_family
    if family in SESSION_ANCHORED_FAMILIES and not has_session_boundary(instrument.calendar_id):
        family = "donchian"

    preset_name = override.preset if override.preset is not None else defaults.preset
    weights = dict(PRESETS.get(preset_name, defaults.weights))

    # keep override's resolved scalars internally consistent (defensive copy)
    override = replace(override, barrier_family=family)

    return ResolvedConfig(
        defaults=defaults,
        override=override,
        instrument=instrument,
        barrier_family=family,
        weights=weights,
    )
