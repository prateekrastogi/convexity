"""Asset-class-aware sleeve config for the cross-sectional engine: symbols, friction, regimes, breadth.

The cross-sectional momentum engine (``panel``/``signals``/``portfolio``/``engine``) is class-agnostic:
it ranks daily closes and forms a dollar-neutral book. What DIFFERS across the three sleeves
(crypto / equities / commodities) is (a) the symbol list, (b) the realistic per-leg FRICTION, (c)
whether funding applies at all, and (d) the macro-regime windows worth slicing. This module funnels
all four so the driver stays one path and the engine is never re-implemented per class.

Sleeve facts that drive the config:

  * CRYPTO — 50 Binance USDT-M perps, native 8h funding. Momentum-only here uses ZERO funding (the
    carry harvest is a SEPARATE signal), but the friction is the Binance taker truth (see ``accounting``).
  * EQUITIES — 30 Dukascopy underlyings (US-session/RTH-only, real overnight/weekend gaps),
    indicator-grade, SELECTED for modern liquidity (a survivorship caveat the report must surface).
    They are UNDERLYINGS, not perps: there is NO cross-sectional funding to harvest -> funding is 0.
    Retail equity friction is tight: ~1 bp fee + ~1 bp half-spread per leg on mega-caps.
  * COMMODITIES — 7 Dukascopy underlyings (near-continuous, session breaks). Also underlyings ->
    funding 0. Friction is wider than equities (futures/CFD spreads): ~2 bp fee + ~3 bp half-spread.
    SEVEN names is NOT a real cross-section: a tercile leg is 2-3 names, idiosyncratic-dominated; the
    report LABELS it thin, not a quantile effect.

FUNDING = 0 for equities AND commodities (they are underlyings, momentum-only). The panel builder
(``build_class_panel``) forces the funding array to zero for those sleeves and asserts it, so the
momentum signal is PURE trailing-return momentum and ``settle_week``'s funding cash flow is exactly 0.
Crypto momentum-only ALSO runs funding 0 here (funding carry is not the momentum signal's business).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.assetclass import Sleeve

# ------------------------------------------------------------------------------------------------
# Per-class realistic friction (FRACTIONS of notional, per leg). Round-trip charged on both legs by
# ``accounting.round_trip_friction`` = 2 * (taker + half_spread + impact). NO Binance-perp taker for
# equities/commodities — the real underlying/CFD retail friction applies instead.
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassFriction:
    """Realistic per-leg friction for one sleeve (fractions of notional). Momentum-only: no funding."""

    taker: float  # exchange/broker fee per leg
    half_spread: float  # half the quoted bid/ask spread per leg
    impact: float  # conservative market-impact per leg at small size

    @property
    def round_trip_bps(self) -> float:
        """Round-trip friction in bps = 2 legs * (taker + half_spread + impact) * 1e4 (for reporting)."""
        return 2.0 * (self.taker + self.half_spread + self.impact) * 1.0e4


# CRYPTO: the Binance USDT-M VIP0 taker truth (5 bp) + 1 bp half-spread + 0.5 bp impact (matches the
# committed crypto ``accounting`` defaults) -> 13 bp round-trip. Kept here so the driver reads ONE table.
CRYPTO_FRICTION = ClassFriction(taker=5.0e-4, half_spread=1.0e-4, impact=0.5e-4)

# EQUITIES: retail mega-cap underlyings. ~1 bp broker fee + ~1 bp half-spread + ~0.25 bp impact per leg
# -> ~4.5 bp round-trip. Mega-caps (AAPL/MSFT/…) quote sub-bp spreads; 1 bp half-spread is conservative.
EQUITY_FRICTION = ClassFriction(taker=1.0e-4, half_spread=1.0e-4, impact=0.25e-4)

# COMMODITIES: CFD/futures underlyings, wider than equities. ~2 bp fee + ~3 bp half-spread + ~0.5 bp
# impact per leg -> ~11 bp round-trip. Metals/energy CFD spreads are materially wider than mega-cap eq.
COMMODITY_FRICTION = ClassFriction(taker=2.0e-4, half_spread=3.0e-4, impact=0.5e-4)

FRICTION_BY_SLEEVE: dict[Sleeve, ClassFriction] = {
    "crypto": CRYPTO_FRICTION,
    "equity": EQUITY_FRICTION,
    "commodity": COMMODITY_FRICTION,
}

# ------------------------------------------------------------------------------------------------
# Per-class macro-regime windows (UTC calendar). The equity/commodity macro clock differs from crypto
# (rate cycle, 2022 inflation shock, 2020 COVID), so the regime slices are class-specific.
# ------------------------------------------------------------------------------------------------

CRYPTO_REGIMES: dict[str, tuple[str, str]] = {
    "2021_bull": ("2021-01-01", "2021-11-10"),
    "2022_bear": ("2021-11-11", "2022-12-31"),
    "2023_recovery": ("2023-01-01", "2023-12-31"),
    "2024_bull": ("2024-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-12-31"),
}

# Equities/commodities share a macro clock (rate hikes 2022, inflation, soft-landing 2023-24). The
# panels start ~2021-06 (Dukascopy stitch floor for the 5-yr window), so pre-2021 windows stay empty.
TRADFI_REGIMES: dict[str, tuple[str, str]] = {
    "2021_reflation": ("2021-01-01", "2021-12-31"),
    "2022_ratehike": ("2022-01-01", "2022-12-31"),  # inflation shock / hiking cycle
    "2023_softland": ("2023-01-01", "2023-12-31"),
    "2024_riskon": ("2024-01-01", "2024-12-31"),
    "2025_2026": ("2025-01-01", "2026-12-31"),
}

REGIMES_BY_SLEEVE: dict[Sleeve, dict[str, tuple[str, str]]] = {
    "crypto": CRYPTO_REGIMES,
    "equity": TRADFI_REGIMES,
    "commodity": TRADFI_REGIMES,
}


def sleeve_symbols(sleeve: Sleeve) -> tuple[str, ...]:
    """Point-in-time symbol list for one sleeve, from UNIVERSE.md (REUSE ``all_symbols``).

    ``sleeve`` is the coarse {crypto, equity, commodity} bucket; the UNIVERSE.md sleeve key is the
    plural (``equities``/``commodities``). Returns the full sleeve; the panel builder then applies the
    point-in-time listing mask (a symbol appears only over its own live span).
    """
    from research.universe import all_symbols

    key = {"crypto": "crypto", "equity": "equities", "commodity": "commodities"}[sleeve]
    return tuple(all_symbols(sleeves=(key,)))


def effective_breadth(contrib: np.ndarray) -> float:
    """Effective number of contributing names = inverse-HHI on per-name TOTAL absolute contribution.

    ``contrib`` — (n_weeks, n_symbols) per-name price(+funding) contribution to the book return (the
    ``XSecBook.contrib`` atom). We sum |contribution| per name over all weeks, normalize to shares
    ``w`` (sum 1), and return ``1 / sum(w**2)`` — the inverse Herfindahl. A book whose PnL rides one
    name has breadth ~1; a perfectly even book has breadth = n_symbols. SAME formula the crypto
    attribution script uses, factored here so every sleeve reports it identically. Returns 0.0 for an
    empty / all-zero book.
    """
    if contrib.size == 0:
        return 0.0
    abs_name = np.abs(contrib).sum(axis=0)
    tot = float(abs_name.sum())
    if tot <= 0.0:
        return 0.0
    w = abs_name / tot
    return float(1.0 / np.sum(w**2))
