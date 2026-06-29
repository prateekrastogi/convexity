"""size_position — the public LEVERAGE entry point the strategy/SYSTEM adapter calls.

LEVERAGE engine_api (§2, §9). Satisfies the ``convexity.core.protocols.LeverageEngine``
Protocol: ``size_position(signal, risk, cfg, book) -> LeverageDecision``. Pure-Python,
NautilusTrader-free (composed by ``strategy/convexity_strategy.py``). The regime cut
surfaces on the returned ``LeverageDecision`` as ``binding_ceiling='portfolio'`` with the
portfolio reason_code ``'regime_heat_cut'`` (P7/Inv R) — never a separate 'regime' literal.

``book`` is the SYSTEM-owned ``BookState`` (ARCHITECTURE §5.9); LEVERAGE consumes it via
the read-only Protocol shape and is driven in tests by ``BookStateFixture``. The
covariance / candidate-weights the MCTR ceiling needs are mirrored on ``RiskInputs`` so
the engine has one consistent risk view; SYSTEM keeps ``book`` and ``risk`` in sync.
"""

from __future__ import annotations

from convexity.leverage.combine.engine import LeverageEngine
from convexity.leverage.config.margin import MarginModel, ScenarioMarginModel
from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import (
    BookStateFixture,
    LeverageDecision,
    RiskInputs,
    SignalInput,
)


def size_position(
    signal: SignalInput,
    risk: RiskInputs,
    cfg: ResolvedConfig,
    book: BookStateFixture,
    *,
    margin_model: MarginModel | None = None,
    carry: float = 0.0,
    prev_phi: float | None = None,
) -> LeverageDecision:
    """Convert a qualified ``SignalInput`` + measurable ``RiskInputs`` into a safe
    ``LeverageDecision`` (§1 equation). ``carry`` is COST's SIGNED carry (fraction of
    position margin); the §8.1 stop clamp applies ``max(carry,0)`` (Inv CARRY).
    ``margin_model`` defaults to the SPAN-style scenario model (§8.4)."""
    engine = LeverageEngine(margin_model or ScenarioMarginModel())
    return engine.decide(signal, risk, cfg, book, carry=carry, prev_phi=prev_phi)


class LeverageEngineAPI:
    """Object form of the public surface for the ``LeverageEngine`` Protocol seam.

    SYSTEM/strategy compose against ``size_position(signal, risk, cfg, book)``; this
    wrapper holds the margin model + ceiling instances so they are constructed once.
    """

    def __init__(self, margin_model: MarginModel | None = None) -> None:
        self._engine = LeverageEngine(margin_model or ScenarioMarginModel())

    def size_position(
        self,
        signal: SignalInput,
        risk: RiskInputs,
        cfg: ResolvedConfig,
        book: BookStateFixture,
        *,
        carry: float = 0.0,
        prev_phi: float | None = None,
    ) -> LeverageDecision:
        return self._engine.decide(signal, risk, cfg, book, carry=carry, prev_phi=prev_phi)
