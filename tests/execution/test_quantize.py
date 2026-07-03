"""Lot/tick quantization — the last step from a continuous NAV-fraction trade to an actual,
venue-legal order (EXECUTION.md §16, Invariant LOT-Q). Everything upstream (solve_trajectory,
RebalancePlan) works in continuous fraction-of-NAV units by design; these functions close the gap
to a real quantity/price, and are independently testable before the venue adapter exists to call
them."""

from __future__ import annotations

import numpy as np
import pytest

from convexity.execution.execution import child_order_qty, quantize_price, quantize_qty
from convexity.panel.candidates import crypto_filters


def test_quantize_qty_rounds_down_to_lot_grid():
    assert quantize_qty(0.0034, lot_size=0.001) == pytest.approx(0.003)
    assert quantize_qty(0.0039, lot_size=0.001) == pytest.approx(0.003)


def test_quantize_qty_exact_multiple_survives_float_round_trip():
    # 0.003 / 0.001 can evaluate to 2.999999999999... in float64; without the epsilon guard this
    # would silently under-round to 0.002, a real one-lot short-fall on every exact-multiple trade.
    assert quantize_qty(0.003, lot_size=0.001) == pytest.approx(0.003)
    assert quantize_qty(1.0, lot_size=0.1) == pytest.approx(1.0)


def test_quantize_qty_never_exceeds_raw_magnitude():
    # Invariant LOT-Q / P6-conservative: quantization only ever removes size, never adds it.
    rng = np.random.default_rng(0)
    for raw in rng.uniform(-10, 10, size=200):
        for lot in (0.001, 0.01, 0.1, 1.0):
            q = quantize_qty(float(raw), lot_size=lot)
            assert abs(q) <= abs(raw) + 1e-9


def test_quantize_qty_preserves_sign():
    assert quantize_qty(-0.0034, lot_size=0.001) == pytest.approx(-0.003)
    assert quantize_qty(0.0034, lot_size=0.001) > 0.0


def test_quantize_qty_drops_below_min_qty_instead_of_inflating():
    # A slice too small to clear the exchange minimum is DROPPED (0.0), never rounded UP to
    # min_qty -- inflating would trade MORE than the caller intended, the opposite of §16.
    assert quantize_qty(0.0005, lot_size=0.001, min_qty=0.001) == 0.0
    assert quantize_qty(0.0, lot_size=0.001, min_qty=0.001) == 0.0


def test_quantize_qty_result_is_always_a_lot_multiple():
    rng = np.random.default_rng(1)
    lot = 0.001
    for raw in rng.uniform(-5, 5, size=100):
        q = quantize_qty(float(raw), lot_size=lot)
        n_lots = q / lot
        assert abs(n_lots - round(n_lots)) < 1e-6


@pytest.mark.parametrize(
    "raw_qty,lot_size,min_qty",
    [
        (float("nan"), 0.001, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, -0.001, 0.0),
        (1.0, 0.001, -1.0),
    ],
)
def test_quantize_qty_rejects_invalid_inputs(raw_qty, lot_size, min_qty):
    with pytest.raises(ValueError):
        quantize_qty(raw_qty, lot_size=lot_size, min_qty=min_qty)


def test_quantize_price_buy_rounds_down_sell_rounds_up():
    # Conservative direction: a buy never rounds UP past the reference price (would overpay), a
    # sell never rounds DOWN past it (would undersell).
    assert quantize_price(65042.37, 0.10, side=1) == pytest.approx(65042.3)
    assert quantize_price(65042.37, 0.10, side=-1) == pytest.approx(65042.4)


def test_quantize_price_never_more_aggressive_than_reference():
    rng = np.random.default_rng(2)
    for price in rng.uniform(0.001, 100_000, size=200):
        for tick in (0.0001, 0.01, 0.1, 1.0):
            buy = quantize_price(float(price), tick, side=1)
            sell = quantize_price(float(price), tick, side=-1)
            assert buy <= price + 1e-9  # never bids above the reference
            assert sell >= price - 1e-9  # never offers below the reference


def test_quantize_price_exact_on_grid_is_unchanged():
    assert quantize_price(100.0, 0.1, side=1) == pytest.approx(100.0)
    assert quantize_price(100.0, 0.1, side=-1) == pytest.approx(100.0)


def test_quantize_price_rejects_invalid_side():
    with pytest.raises(ValueError):
        quantize_price(100.0, 0.1, side=0)


def test_child_order_qty_realistic_btc_slice():
    f = crypto_filters("BTCUSDT")
    # 1% of a $1,000,000 book at $65,000/BTC = $10,000 = 0.153846... BTC, lot-quantized to 0.153.
    qty = child_order_qty(
        0.01, nav=1_000_000, price=65_000, lot_size=f.lot_size, min_qty=f.min_qty,
        min_notional=f.min_notional,
    )
    assert qty == pytest.approx(0.153)


def test_child_order_qty_drops_slice_below_min_qty():
    f = crypto_filters("DOGEUSDT")  # lot_size=1: cannot trade a fraction of one DOGE
    qty = child_order_qty(
        0.0000001, nav=100_000, price=0.2, lot_size=f.lot_size, min_qty=f.min_qty,
        min_notional=f.min_notional,
    )
    assert qty == 0.0


def test_child_order_qty_drops_slice_below_min_notional_even_if_lot_clears():
    # A quantity that clears lot_size/min_qty but whose dollar value is still under min_notional
    # must also be dropped -- a second, independent exchange-legality floor (§16).
    qty = child_order_qty(
        0.0001, nav=1_000, price=100.0, lot_size=0.001, min_qty=0.001, min_notional=50.0,
    )
    # raw_qty = 0.0001*1000/100 = 0.001 -> quantizes to 0.001 (clears lot/min_qty) but
    # notional = 0.001*100 = $0.10 << $50 min_notional -> must drop to 0.
    assert qty == 0.0


def test_child_order_qty_zero_trade_fraction_is_zero():
    f = crypto_filters("BTCUSDT")
    assert child_order_qty(0.0, nav=1_000_000, price=65_000, lot_size=f.lot_size) == 0.0
