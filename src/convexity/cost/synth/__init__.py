"""Synthetic adversarial paths with funding/basis/spread/roll overlays.

Spec: COST §13. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
See docs/ARCHITECTURE.md for the package layout and docs/BUILD_PLAN.md for the build order.
"""

# TODO(impl): implement per COST §13. Shared boundary types live in convexity.core.types;
# governance buckets + the resolve_config single-merge-point pattern in convexity.core.params.
