"""MarketDNA — slow monthly/quarterly priors (TREND §3 priors).

Slow-moving structural priors (monthly/quarterly) used to seed TEAL profiles for a thin-history
symbol — NOT live diagnostics and NOT a decision signal (§3). They update at most monthly and are
pooled by asset class + liquidity bucket when symbol history is thin (§14). Read-only at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class MarketDNA:
    """A slow structural prior bundle for one asset class (§3). Updated monthly/quarterly."""

    asset_class: str
    typical_gap_var: float
    typical_trend_life_days: float  # seed for the survival prior on a thin symbol
    liquidity_profile: Mapping[str, float] = field(default_factory=dict)
    asof_month: int = 0


def default_dna(asset_class: str) -> MarketDNA:
    """A conservative default prior per asset class (seeds TEAL fits, §3/§14)."""
    table = {
        "crypto_perp": (1e-4, 1.5),
        "crypto_spot": (1e-4, 1.5),
        "us_equity": (4e-4, 2.0),
        "commodity_future": (3e-4, 3.0),
        "precious_metal": (2e-4, 3.0),
        "industrial_metal": (2e-4, 3.0),
    }
    gap, life = table.get(asset_class, (1e-4, 2.0))
    return MarketDNA(asset_class=asset_class, typical_gap_var=gap, typical_trend_life_days=life)
