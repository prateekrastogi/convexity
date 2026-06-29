"""AMBER per-asset-class metadata for panel alignment + cadence (REGIME §4 AMBER).

Cross-asset behaviour flows through METADATA, never a forked code path (REGIME §2 panel hygiene,
§2.1 DVOL switch). This module carries the AMBER switches: which calendar a class trades on (24/7
crypto vs session-bound equities vs Dukascopy commodities), whether a published implied-vol index
(DVOL) is available for the class's vol measure, and the staleness window inside which a non-trading
stamp is forward-filled to a zero return before the name is dropped from ``n_t``.

These are supplied metadata, NOT modelling knobs: no RED/BLUE threshold appears here (Invariant C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from convexity.core.types import AssetClass


@dataclass(frozen=True)
class AssetClassConfig:
    """AMBER metadata for one asset class (REGIME §4 AMBER, §2.1).

    ``always_open`` is True for 24/7 crypto (no session-close drop). ``dvol_available`` is the
    implied-vol-vs-realized switch for ``g^{vol}`` (a metadata switch, not new logic, §2.1).
    ``staleness_stamps`` is how many cadence stamps a name may be forward-filled to a zero return
    before it is dropped, shrinking ``n_t`` (§2 panel hygiene).
    """

    asset_class: AssetClass
    always_open: bool
    dvol_available: bool
    staleness_stamps: int


# RED-free AMBER defaults per class. Crypto-perp is the only path traded live in this repo, but the
# full set is carried faithfully (REGIME scope: full cross-asset).
ASSET_CLASS_CONFIG: Mapping[AssetClass, AssetClassConfig] = {
    "crypto_perp": AssetClassConfig("crypto_perp", always_open=True, dvol_available=True, staleness_stamps=2),
    "crypto_spot": AssetClassConfig("crypto_spot", always_open=True, dvol_available=True, staleness_stamps=2),
    "us_equity": AssetClassConfig("us_equity", always_open=False, dvol_available=False, staleness_stamps=1),
    "commodity_future": AssetClassConfig("commodity_future", always_open=False, dvol_available=False, staleness_stamps=1),
    "precious_metal": AssetClassConfig("precious_metal", always_open=False, dvol_available=False, staleness_stamps=1),
    "industrial_metal": AssetClassConfig("industrial_metal", always_open=False, dvol_available=False, staleness_stamps=1),
}


def config_for(asset_class: AssetClass) -> AssetClassConfig:
    """The AMBER config for one class; defaults to the conservative session-bound profile."""
    return ASSET_CLASS_CONFIG.get(
        asset_class,
        AssetClassConfig(asset_class, always_open=False, dvol_available=False, staleness_stamps=1),
    )
