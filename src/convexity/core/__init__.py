"""Shared foundation — governance buckets, cross-pillar data types, composition protocols.

Imported by every pillar; NautilusTrader-free. See ``params`` (RED/AMBER/BLUE/TEAL governance),
``types`` (the boundary objects that cross pillar lines), and ``protocols`` (the composition seams
the strategy adapter composes against).
"""

from convexity.core import params, protocols, types

__all__ = ["params", "types", "protocols"]
