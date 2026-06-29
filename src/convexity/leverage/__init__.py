"""Risk-budget sizing engine: min() of seven ceilings (pure-Python pillar).

Spec: LEVERAGE. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
See docs/ARCHITECTURE.md for the package layout and docs/BUILD_PLAN.md for the build order.
"""

# TODO(impl): implement per LEVERAGE. Shared boundary types live in convexity.core.types;
# governance buckets + the resolve_config single-merge-point pattern in convexity.core.params.
