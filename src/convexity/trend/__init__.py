"""Convex trend-life exit engine (pure-Python pillar).

Spec: TREND. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
TREND is a death detector + convex exit policy, NOT a trend-prediction model (§0). It consumes an
already-open ``Position`` from BREAKOUT (Inv O) and, each bar, decides continue/reduce/exit via the
fixed-priority ``DecisionPolicy`` (Inv A) under the dynamic stopping-time horizon (§1) capped at the
SYSTEM-injected ``H_sys`` (Inv HC).
"""

from convexity.trend.config.resolve import ResolvedConfig, resolve_config
from convexity.trend.engine import TrendLifeEngine
from convexity.trend.types import ActionDecision, BeliefState, ExitState

__all__ = [
    "ResolvedConfig",
    "resolve_config",
    "TrendLifeEngine",
    "ActionDecision",
    "BeliefState",
    "ExitState",
]
