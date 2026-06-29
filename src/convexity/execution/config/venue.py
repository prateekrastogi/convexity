"""VenueSpec — order types, reduce-only support, rate limits, native stop semantics (§4 AMBER).

AMBER supplied metadata: which order types the venue accepts, whether it supports reduce-only and
native modify, the stop trigger reference (mark / index / last — §9), and the rate-limit budget the
§6 rate-limit gate reads. The order ISSUER (§7) and the protective stop (§9) read this to construct
venue-valid commands; no modelling knob lives here (those are RED, §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class VenueSpec:
    """Venue capability metadata (§4 AMBER / §9 native stop semantics)."""

    venue: str
    supports_post_only: bool = True
    supports_reduce_only: bool = True
    supports_native_modify: bool = True  # else amend = place-before-cancel (§9.amend)
    supports_native_stop: bool = True  # STOP_MARKET / STOP reduce-only on the venue (P6)
    supports_oco: bool = False  # contingent/OCO so a stop fill cancels working entries (§9)
    stop_trigger_reference: Literal["mark", "index", "last"] = "mark"  # avoid wick triggers (§9/§13)
    stop_market_safe: bool = True  # else use STOP_LIMIT (class preset override, §9)
    # §6.4 rate-limit gate: the order-rate budget headroom flag is read live off MicroState; these
    # are the venue's published ceilings used to interpret it.
    max_orders_per_10s: int = 50
    accepted_order_kinds: tuple[str, ...] = field(
        default_factory=lambda: ("LIMIT", "MARKET", "STOP_MARKET", "STOP_LIMIT")
    )


# §13 — Binance USDT-M perps: native STOP_MARKET reduce-only with a MARK-price trigger; rate limits
# matter so cancel/replace churn is capped (§6.4).
BINANCE_USDM_VENUE = VenueSpec(
    venue="BINANCE",
    supports_post_only=True,
    supports_reduce_only=True,
    supports_native_modify=True,
    supports_native_stop=True,
    supports_oco=True,
    stop_trigger_reference="mark",
    stop_market_safe=True,
    max_orders_per_10s=300,
)

_VENUES: dict[str, VenueSpec] = {BINANCE_USDM_VENUE.venue: BINANCE_USDM_VENUE}


def venue_spec(venue: str) -> VenueSpec:
    """Look up the AMBER venue spec, defaulting to a conservative native-stop venue (§4)."""
    return _VENUES.get(venue, VenueSpec(venue=venue))
