"""Crisis / correlation-to-1 / margin-explosion / worst-gap stress tests (LEVERAGE §12).

VALIDATION-ONLY (Invariant V): never imported by the live layers. Each stress applies an
adversarial transform to a scenario's risk inputs and re-sizes, asserting the
catastrophic layer (tail / portfolio / stressed margin) de-risks correctly (§12).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from convexity.leverage.engine_api import size_position
from convexity.leverage.synth.generate import LeverageScenario


@dataclass(frozen=True)
class StressResult:
    name: str
    base_leverage: float
    stressed_leverage: float
    base_binding: str
    stressed_binding: str
    de_risked: bool  # stressed leverage strictly lower (catastrophic layer engaged)


def correlation_to_one(scenario: LeverageScenario) -> StressResult:
    """Push the covariance correlations to ~1 (diversification vanishes, P4) and re-size.

    Covariance is BookState-owned (ARCHITECTURE §5.9): the crisis cov is applied to the
    ``book`` (the canonical source the engine reads), and ``corr_instability`` — a
    market-state measure — on the risk view; the two stay one consistent picture.
    """
    base = size_position(scenario.signal, scenario.risk, scenario.cfg, scenario.book)
    cov = np.asarray(scenario.book.cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    crisis_corr = np.full(cov.shape, 0.99)
    np.fill_diagonal(crisis_corr, 1.0)
    crisis_cov = (d[:, None] * crisis_corr) * d[None, :]
    stressed_risk = replace(scenario.risk, corr_instability=0.9)
    stressed_book = replace(scenario.book, cov=crisis_cov)
    stressed = size_position(scenario.signal, stressed_risk, scenario.cfg, stressed_book)
    return StressResult(
        name=f"{scenario.name}:corr_to_1",
        base_leverage=base.leverage, stressed_leverage=stressed.leverage,
        base_binding=base.binding_ceiling, stressed_binding=stressed.binding_ceiling,
        de_risked=stressed.leverage <= base.leverage + 1e-9,
    )


def margin_spike(scenario: LeverageScenario, *, free_margin_shrink: float = 0.25) -> StressResult:
    """Shrink free margin (a margin hike / spike) and confirm L_margin tightens (§8.4)."""
    base = size_position(scenario.signal, scenario.risk, scenario.cfg, scenario.book)
    # free margin is BookState-owned (ARCHITECTURE §5.9): shrink it on the ``book``.
    stressed_book = replace(scenario.book, free_margin=scenario.book.free_margin * free_margin_shrink)
    stressed = size_position(scenario.signal, scenario.risk, scenario.cfg, stressed_book)
    return StressResult(
        name=f"{scenario.name}:margin_spike",
        base_leverage=base.leverage, stressed_leverage=stressed.leverage,
        base_binding=base.binding_ceiling, stressed_binding=stressed.binding_ceiling,
        de_risked=stressed.leverage <= base.leverage + 1e-9,
    )


def worst_gap_replay(scenario: LeverageScenario, *, gap_multiple: float = 3.0) -> StressResult:
    """Inflate the historical worst adverse gap and confirm the tail gap-survival
    sub-constraint cuts size (§8.3 — the hard stop does not survive gaps)."""
    base = size_position(scenario.signal, scenario.risk, scenario.cfg, scenario.book)
    g = scenario.risk.gap_hist
    worse = replace(g, worst_adverse_gap=g.worst_adverse_gap * gap_multiple,
                    class_stress_gap=g.class_stress_gap * gap_multiple)
    stressed_risk = replace(scenario.risk, gap_hist=worse)
    stressed = size_position(scenario.signal, stressed_risk, scenario.cfg, scenario.book)
    return StressResult(
        name=f"{scenario.name}:worst_gap",
        base_leverage=base.leverage, stressed_leverage=stressed.leverage,
        base_binding=base.binding_ceiling, stressed_binding=stressed.binding_ceiling,
        de_risked=stressed.leverage <= base.leverage + 1e-9,
    )
