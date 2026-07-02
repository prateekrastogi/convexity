"""AMBER per-sleeve metadata for REGIME (REGIME.md §6 AMBER bucket).

AMBER is "supplied metadata": the BTC symbol used for ``V_t`` (§2.3, matching ``overlay.py``'s
``market_symbol="BTCUSDT"`` default), the ``CRYPTO_REGIMES`` window boundaries (reused, NOT reforked,
from ``research/xsec/universe.py`` — §6 AMBER: "already AMBER-equivalent constants ... reused not
reforked"), and the Friday-close weekly rebalance clock (matching the xsec engine). No RED constant
(Delta_target, ARL_0_target, k, h, n_confirm, DD-stop params) may appear here — that leak is exactly
what Invariant C-cfg (§3.4, §6) forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.xsec.universe import CRYPTO_REGIMES


@dataclass(frozen=True)
class RegimeAssetClassPreset:
    """AMBER per-sleeve metadata (REGIME.md §6). ``market_symbol`` is the asset whose realized vol
    feeds ``V_t`` (§2.3); ``regime_windows`` are the ex-post-labeled calibration windows reused from
    ``research/xsec/universe.py`` (§3.4 designates ``2023_recovery``/``2024_bull`` as the in-control
    training windows, NOT ``2022_bear`` which is itself a stress regime)."""

    market_symbol: str
    regime_windows: dict[str, tuple[str, str]]
    in_control_windows: tuple[str, ...]


# The crypto sleeve preset (REGIME.md scope: Binance USDT-M perps, §0). market_symbol matches
# overlay.py's default; regime_windows are the reused CRYPTO_REGIMES table (§6 AMBER). The in-control
# training windows are 2023_recovery + 2024_bull per §3.4 (calm-to-normal; 2022_bear excluded).
CRYPTO_REGIME_PRESET = RegimeAssetClassPreset(
    market_symbol="BTCUSDT",
    regime_windows=dict(CRYPTO_REGIMES),
    in_control_windows=("2023_recovery", "2024_bull"),
)

DEFAULT_PRESETS: dict[str, RegimeAssetClassPreset] = {"crypto": CRYPTO_REGIME_PRESET}
