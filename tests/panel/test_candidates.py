"""``panel.candidates`` — the universe loader and its real per-symbol exchange filters.

Covers the fix for the uniform tick_size=0.01/lot_size=0.001 placeholder that used to apply to
every symbol regardless of asset: crypto entries now carry REAL Binance USDT-M futures filters
(EXECUTION.md §16, Invariant LOT-Q)."""

from __future__ import annotations

import pytest

from convexity.panel.candidates import ExchangeFilters, all_symbols, crypto_filters, load_universe


def test_load_universe_has_three_sleeves():
    u = load_universe()
    assert set(u) >= {"meta", "crypto", "equities", "commodities"}
    assert len(u["crypto"]) > 0
    assert len(u["equities"]) > 0
    assert len(u["commodities"]) > 0


def test_all_symbols_matches_universe_counts():
    u = load_universe()
    assert len(all_symbols(("crypto",))) == len(u["crypto"])
    assert len(all_symbols(("crypto", "equities", "commodities"))) == (
        len(u["crypto"]) + len(u["equities"]) + len(u["commodities"])
    )


def test_crypto_filters_are_real_not_a_uniform_placeholder():
    # The old bug: every symbol got tick_size=0.01/lot_size=0.001 regardless of asset. These three
    # real Binance USDT-M filters must differ from each other and from that placeholder pair.
    btc = crypto_filters("BTCUSDT")
    xrp = crypto_filters("XRPUSDT")
    doge = crypto_filters("DOGEUSDT")
    assert isinstance(btc, ExchangeFilters)
    assert btc.tick_size == pytest.approx(0.1)
    assert btc.lot_size == pytest.approx(0.001)
    # XRP's real lot_size (0.1) is 100x the old placeholder (0.001) -- the whole point of the fix.
    assert xrp.lot_size == pytest.approx(0.1)
    assert xrp.tick_size == pytest.approx(0.0001)
    # DOGE trades in whole-coin lots on Binance -- the old placeholder (0.001) would have generated
    # an order for a fraction of a coin that the exchange rejects outright.
    assert doge.lot_size == pytest.approx(1.0)
    assert doge.min_qty == pytest.approx(1.0)
    # every crypto entry must carry a positive min_notional (the old code left it at the dataclass
    # default 0.0, silently disabling the dollar-floor check).
    for sym in ("BTCUSDT", "XRPUSDT", "DOGEUSDT"):
        f = crypto_filters(sym)
        assert f.min_notional > 0.0
        assert f.tick_size > 0.0
        assert f.lot_size > 0.0


def test_crypto_filters_covers_every_crypto_candidate():
    # No silent gaps: every symbol in the crypto sleeve must resolve real filters.
    for sym in all_symbols(("crypto",)):
        f = crypto_filters(sym)
        assert f.lot_size > 0.0 and f.tick_size > 0.0 and f.min_notional > 0.0


def test_crypto_filters_raises_for_non_crypto_symbol():
    # Equities/commodities are Dukascopy-proxied, not a genuine Binance order book (EDGE.md scope:
    # dead sleeve, no live cross-sectional edge) -- crypto_filters must never silently fabricate a
    # number for them.
    with pytest.raises(KeyError):
        crypto_filters("AAPLUSDT")
