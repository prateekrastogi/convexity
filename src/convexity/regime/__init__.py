"""Global gross-exposure multiplier g_t (pure-Python pillar).

Spec: REGIME. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
See docs/ARCHITECTURE.md for the package layout and docs/BUILD_PLAN.md for the build order.

REGIME is the portfolio-level macro brake: it measures the GLOBAL cross-asset market state (vol,
correlation/absorption, turbulence, liquidity), and emits exactly one thing — a causal, gradual,
sticky, cost-gated global risk multiplier ``g_t ∈ {1.00, 0.75, 0.50, 0.25}`` (floored at 0.25) as a
:class:`convexity.core.types.RegimeState`. LEVERAGE consumes it via the single ``g_t·H_book``
channel; SELECTION reads ``s_t`` for the concurrent-position count. REGIME never sizes, never tapers
for equity drawdown, never halts, never picks names, never trades (Invariant SCOPE).
"""

from convexity.regime.config.resolve import ResolvedConfig, param_registry, resolve_config
from convexity.regime.engine_api import RegimeEngine

__all__ = ["RegimeEngine", "ResolvedConfig", "resolve_config", "param_registry"]
