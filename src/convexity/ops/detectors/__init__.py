"""Detector layer: CUSUM/PH/EWMA/Shewhart/SPRT (isolated O(1) causal recursions).

Spec: OPS §6. Pure-Python, NautilusTrader-free (composed by strategy/convexity_strategy.py).
See docs/ARCHITECTURE.md for the package layout and docs/BUILD_PLAN.md for the build order.
"""

# TODO(impl): implement per OPS §6. Shared boundary types live in convexity.core.types;
# governance buckets + the resolve_config single-merge-point pattern in convexity.core.params.
