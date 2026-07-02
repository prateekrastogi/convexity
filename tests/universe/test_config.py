"""Invariant C — RED constants are shared/class-level, never per-symbol (UNIVERSE.md §8, §11.2)."""

from __future__ import annotations

import dataclasses

import pytest

from convexity.universe.config.defaults import DEFAULT_UNIVERSE_CONFIG, UniverseConfig


def test_red_constants_are_not_per_symbol():
    """Inv C (§8): ``q_in``, ``q_out``, ``N``, ``W`` are RED (shared, class-level). A per-symbol config
    object carrying any of them would be the bug. Static/schema check — no fixture data needed.

    ``UniverseConfig`` is the ONE RED holder: it carries the class-level constants and is NOT keyed by
    symbol (it has no symbol/instrument field). We assert both that the RED constants live on the
    class-level config AND that the config has no per-instrument key.
    """
    fields = {f.name for f in dataclasses.fields(UniverseConfig)}
    for red in ("W", "q_in", "q_out", "N"):
        assert red in fields, f"{red} must be a RED class-level constant on UniverseConfig"
    for per_symbol_key in ("symbol", "instrument", "sym", "asset"):
        assert per_symbol_key not in fields, f"UniverseConfig must not be keyed per-symbol ({per_symbol_key})"


def test_w_warmup_is_pinned_to_W_not_independent():
    """§1.4: ``w_warmup`` is DEFINED to equal ``W`` (not an independent field). Changing W moves it."""
    fields = {f.name for f in dataclasses.fields(UniverseConfig)}
    assert "w_warmup" not in fields, "w_warmup must be a derived property, never a stored free field"
    assert DEFAULT_UNIVERSE_CONFIG.w_warmup == DEFAULT_UNIVERSE_CONFIG.W
    cfg45 = UniverseConfig(W=45)
    assert cfg45.w_warmup == 45, "a change to W propagates to w_warmup automatically (§1.4)"


def test_deployed_red_defaults_match_spec():
    """The deployed RED defaults match UNIVERSE.md §8 / §2.3: W=30, q_in=0.80, q_out=0.65, N=50."""
    c = DEFAULT_UNIVERSE_CONFIG
    assert (c.W, c.q_in, c.q_out, c.N) == (30, 0.80, 0.65, 50)


def test_config_rejects_inverted_band():
    """§2.2: q_in > q_out is required; an inverted band is rejected at construction."""
    with pytest.raises(ValueError):
        UniverseConfig(q_in=0.65, q_out=0.80)


def test_conservative_fallback():
    """Cross-cutting: the min-valid-day fraction (§2.4) is a RED class-level constant, not per-symbol,
    so the conservative-fallback threshold itself cannot be tuned per name."""
    fields = {f.name for f in dataclasses.fields(UniverseConfig)}
    assert "min_valid_frac" in fields, "the §2.4 valid-day fraction is a RED class-level constant"
    assert DEFAULT_UNIVERSE_CONFIG.min_valid_frac == 0.5, "default = ceil(W/2) rule (§2.4)"
