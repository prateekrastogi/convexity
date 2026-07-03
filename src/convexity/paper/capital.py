"""Collateral waterfall (PAPER.md §3.1, USER MANDATE 2026-07-03): USDT first, then USDC — BTC NEVER.

The demo account holds USDT + USDC + BTC. The book's NAV for sizing draws USDT alone while USDT
can fund the required initial margin with headroom; USDC folds in ONLY when USDT alone cannot
(logged loudly — it marks the account outgrowing its primary collateral). BTC never enters NAV,
never serves as margin from our side, and the book's BTCUSDT PERP positions are derivatives that
do not touch the wallet BTC. Keep the futures account in SINGLE-ASSET margin mode so the exchange
cannot silently draft BTC as cross-collateral (runbook step, deploy/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass

WATERFALL: tuple[str, ...] = ("USDT", "USDC")  # priority order; anything else (BTC, ...) EXCLUDED
MARGIN_HEADROOM = 2.0  # require USDT >= 2x the est. initial margin before ignoring USDC


@dataclass(frozen=True)
class NavDecision:
    nav: float  # sizing NAV in quote (USDT-equivalent; USDC treated 1:1 for the demo)
    used_assets: tuple[str, ...]
    excluded: dict[str, float]  # balances deliberately NOT counted (e.g. BTC)
    note: str


def sizing_nav(assets: dict[str, float], *, est_initial_margin: float = 0.0) -> NavDecision:
    """Resolve the sizing NAV per the waterfall.

    ``assets`` — per-asset margin balances (gateway.account_assets or a dry-run stub).
    ``est_initial_margin`` — this rebalance's estimated initial margin (gross x nav / leverage);
    at quarter-Kelly on a small book this is far below the USDT balance, so v1 normally sizes off
    USDT alone. USDC joins ONLY if USDT < MARGIN_HEADROOM x the estimate.
    """
    usdt = max(0.0, float(assets.get("USDT", 0.0)))
    usdc = max(0.0, float(assets.get("USDC", 0.0)))
    excluded = {a: b for a, b in assets.items() if a not in WATERFALL}
    if usdt >= MARGIN_HEADROOM * max(est_initial_margin, 0.0) and usdt > 0.0:
        return NavDecision(nav=usdt, used_assets=("USDT",), excluded=excluded,
                           note=f"USDT alone funds the book (headroom {MARGIN_HEADROOM}x)")
    nav = usdt + usdc
    if nav <= 0.0:
        raise ValueError("no USDT/USDC collateral available (BTC is excluded by mandate)")
    return NavDecision(nav=nav, used_assets=("USDT", "USDC"), excluded=excluded,
                       note="USDT exhausted vs required margin -- USDC folded in (LOGGED; "
                            "BTC remains excluded)")
