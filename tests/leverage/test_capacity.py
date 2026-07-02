"""Capacity ceiling G_t^cap and the liquidity-cap dial (LEVERAGE.md §5, §10)."""

from __future__ import annotations

import numpy as np

from convexity.leverage.ceilings.capacity import capacity_ceiling, resolve_within_leg_scheme
from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.types import CapacityInput


def test_g_cap_is_min_over_instruments():
    # §5.1: G_cap = min_i(κ·ADV_i / |w_i^ref|). Tightest name (smallest ADV/|w|) binds.
    d = LeverageDefaults()  # kappa = 0.02
    inputs = [
        CapacityInput("BTC", adv=100.0, w_ref=0.5),   # 0.02*100/0.5 = 4.0
        CapacityInput("ALT", adv=10.0, w_ref=0.5),    # 0.02*10/0.5  = 0.4  <- binds
        CapacityInput("ETH", adv=200.0, w_ref=0.25),  # 0.02*200/0.25 = 16
    ]
    cap = capacity_ceiling(inputs, d)
    assert np.isclose(cap.g_cap, 0.4)
    assert cap.binding_symbol == "ALT"


def test_cap_dial_unshortable_short_leg_zeroes_adv():
    # §5.1: an unshortable name on the SHORT side contributes ADV_i = 0 -> ceiling 0 -> book flat.
    d = LeverageDefaults()
    inputs = [
        CapacityInput("BTC", adv=100.0, w_ref=0.5),
        CapacityInput("XYZ", adv=50.0, w_ref=-0.5, shortable=False),  # short & unshortable
    ]
    cap = capacity_ceiling(inputs, d)
    assert cap.g_cap == 0.0
    assert cap.n_unshortable_zeroed == 1
    assert cap.binding_symbol == "XYZ"


def test_unshortable_on_long_side_does_not_zero():
    # §5.1: unshortable only bites the SHORT side; the same name held LONG uses its real ADV.
    d = LeverageDefaults()
    inputs = [CapacityInput("XYZ", adv=50.0, w_ref=0.5, shortable=False)]
    cap = capacity_ceiling(inputs, d)
    assert cap.g_cap > 0.0
    assert cap.n_unshortable_zeroed == 0


def test_zero_weight_name_imposes_no_ceiling():
    # a name carrying no reference weight is skipped (no division by zero).
    d = LeverageDefaults()
    inputs = [CapacityInput("BTC", adv=100.0, w_ref=0.0), CapacityInput("ETH", adv=100.0, w_ref=0.5)]
    cap = capacity_ceiling(inputs, d)
    assert np.isclose(cap.g_cap, 0.02 * 100.0 / 0.5)
    assert cap.binding_symbol == "ETH"


def test_empty_book_has_no_capacity_constraint():
    cap = capacity_ceiling([], LeverageDefaults())
    assert cap.g_cap == float("inf")


def test_cap_dial_off_by_default_below_threshold():
    # Invariant CAP-DIAL / CAP-DEFAULT: below the RED book-size trigger, scheme resolves 'volscaled'.
    d = LeverageDefaults()
    dec = resolve_within_leg_scheme(d.liquidity_cap_book_threshold * 0.5, d)
    assert dec.scheme == "volscaled"
    assert dec.c is None
    assert dec.reason == ""


def test_cap_dial_engages_only_above_threshold_with_reason_code():
    # §5.2: liqcap engages ONLY above the trigger, emitting the distinct reason code + tightness C.
    d = LeverageDefaults()
    dec = resolve_within_leg_scheme(d.liquidity_cap_book_threshold * 2.0, d)
    assert dec.scheme == "liqcap_volscaled"
    assert dec.c == d.liquidity_cap_multiple
    assert dec.reason == "liquidity_cap_engaged"
