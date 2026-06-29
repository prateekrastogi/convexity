"""Governance / no-overfit acceptance (REGIME §13 No-overfit, §4.1; Invariants C, NOFIT, ML, V)."""

import importlib
import pkgutil

import pytest

from convexity.core.params import Bucket, assert_no_red_in_amber
from convexity.core.types import InstrumentSpec
from convexity.regime.config.defaults import DEFAULTS
from convexity.regime.config.resolve import param_registry, resolve_config
from convexity.regime.validate.backtest import k_regime, register_blue_grid


def test_inv_c_red_unreachable_from_instrument_spec():
    """No RED/BLUE regime param name appears on the per-instrument InstrumentSpec (Invariant C)."""
    amber_fields = frozenset(InstrumentSpec.__dataclass_fields__.keys())
    assert_no_red_in_amber(amber_fields, param_registry())  # raises if a RED/BLUE name leaked


def test_inv_c_resolve_rejects_unlisted_blue():
    with pytest.raises(ValueError):
        resolve_config(blue={"kappa": 0.99})  # 0.99 not in the registered grid (NOFIT)
    with pytest.raises(ValueError):
        resolve_config(blue={"not_a_family": 1})


def test_resolve_applies_listed_blue():
    cfg = resolve_config(blue={"kappa": 0.25, "d_min": 4, "tau_regime": "session"})
    assert cfg.policy.kappa == 0.25
    assert cfg.policy.d_min == 4
    assert cfg.policy.tau_regime == "session"


def test_k_regime_capped(  ):
    """Registered K_regime (one-step sum, NOT Cartesian) <= K_regime_max (REGIME §4.1, NOFIT)."""
    reg = register_blue_grid(DEFAULTS)
    assert reg.k_regime <= DEFAULTS.k_regime_max
    assert reg.k_regime == k_regime()  # the registered count


def test_lw_intensities_not_in_searched_grid():
    """The Ledoit-Wolf intensities are TEAL/closed-form, NOT BLUE searched DOF (REGIME §4.1)."""
    names = {p.name for p in param_registry().params()}
    assert "delta_AR" not in names and "delta_turb" not in names
    assert "lw_intensity" not in names


def test_no_blue_param_is_free_continuous():
    """Every BLUE param carries a finite enumerated grid (Invariant NOFIT)."""
    for p in param_registry().params():
        if p.bucket is Bucket.BLUE:
            assert p.blue_grid is not None and len(p.blue_grid) >= 1


def test_inv_v_live_layers_do_not_import_validate():
    """Nothing in panel/measures/state imports regime.validate (Invariant V)."""
    import convexity.regime as regime_pkg

    live_packages = ["convexity.regime.panel", "convexity.regime.measures", "convexity.regime.state"]
    for pkg_name in live_packages:
        pkg = importlib.import_module(pkg_name)
        for mod in pkgutil.iter_modules(pkg.__path__):
            full = f"{pkg_name}.{mod.name}"
            m = importlib.import_module(full)
            src = m.__file__
            with open(src) as fh:
                text = fh.read()
            assert "regime.validate" not in text, f"{full} imports validate (Invariant V)"
    _ = regime_pkg


def test_inv_ml_no_learned_classifier():
    """No ML framework / learned classifier in measures or state (Invariant ML)."""
    import convexity.regime.measures as measures_pkg
    import convexity.regime.state as state_pkg

    banned = ("sklearn", "torch", "tensorflow", "hmmlearn", "keras")
    for pkg in (measures_pkg, state_pkg):
        for mod in pkgutil.iter_modules(pkg.__path__):
            m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
            with open(m.__file__) as fh:
                text = fh.read()
            for b in banned:
                assert f"import {b}" not in text and f"from {b}" not in text
