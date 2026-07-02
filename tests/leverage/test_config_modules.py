"""§9-named config modules: AssetClassCapacityPreset + InstrumentSpec (LEVERAGE.md §9, §3, §5)."""

from __future__ import annotations

import pytest

from convexity.leverage.config.asset_class import (
    AssetClassCapacityPreset,
    AssetClassCapacityTable,
)
from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.config.instrument import InstrumentSpec


def test_asset_class_preset_validates_kappa_and_c_ranges():
    # §5: κ ∈ (0,1], C ∈ {2,3}; a preset can never widen capacity past §5.2's permitted dials.
    AssetClassCapacityPreset("crypto_perp", kappa=0.02, liquidity_cap_multiple=2.0)
    with pytest.raises(ValueError):
        AssetClassCapacityPreset("crypto_perp", kappa=0.0, liquidity_cap_multiple=2.0)
    with pytest.raises(ValueError):
        AssetClassCapacityPreset("crypto_perp", kappa=1.5, liquidity_cap_multiple=2.0)
    with pytest.raises(ValueError):
        AssetClassCapacityPreset("crypto_perp", kappa=0.02, liquidity_cap_multiple=4.0)  # C∉{2,3}


def test_asset_class_table_falls_back_to_red_default():
    # §3: an unlisted class transparently uses the shared RED default κ/C (never a per-symbol dial).
    d = LeverageDefaults()
    empty = AssetClassCapacityTable()
    assert empty.resolve("crypto_perp", d) == (d.kappa, d.liquidity_cap_multiple)


def test_asset_class_table_uses_registered_class_preset():
    d = LeverageDefaults()
    preset = AssetClassCapacityPreset("crypto_spot", kappa=0.01, liquidity_cap_multiple=3.0)
    table = AssetClassCapacityTable(presets={"crypto_spot": preset})
    assert table.resolve("crypto_spot", d) == (0.01, 3.0)
    assert table.resolve("crypto_perp", d) == (d.kappa, d.liquidity_cap_multiple)  # unlisted -> RED


def test_asset_class_preset_is_keyed_by_class_not_symbol_invariant_c_cfg():
    # Invariant C-cfg: the preset key is an AssetClass bucket, never an instrument symbol, so it
    # cannot smuggle a per-symbol dial into the sizing chain.
    preset = AssetClassCapacityPreset("crypto_perp", kappa=0.02, liquidity_cap_multiple=2.0)
    assert preset.asset_class == "crypto_perp"
    assert not hasattr(preset, "symbol")


def test_instrument_spec_carries_amber_fields():
    # §3 AMBER: shortable + lot/multiplier are supplied metadata, defaulting to the linear unit case.
    spec = InstrumentSpec("BTCUSDT")
    assert spec.shortable is True and spec.lot == 1.0 and spec.multiplier == 1.0
    spec2 = InstrumentSpec("XYZ", shortable=False, lot=10.0, multiplier=0.5)
    assert spec2.shortable is False and spec2.lot == 10.0 and spec2.multiplier == 0.5


def test_instrument_spec_rejects_nonpositive_contract_size():
    with pytest.raises(ValueError):
        InstrumentSpec("BTCUSDT", lot=0.0)
    with pytest.raises(ValueError):
        InstrumentSpec("BTCUSDT", multiplier=-1.0)
