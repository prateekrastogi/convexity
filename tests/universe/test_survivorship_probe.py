"""Survivorship-probe surface tests — the §6/§11 provider (UNIVERSE.md §1.3, §1.5, §6).

The auditor's gap 3: ``validate/survivorship_probe.py`` carried real delegated logic
(``roster_effective_breadth`` -> ``convexity.panel.sleeve.effective_breadth``) but had NO companion test,
so the reused-formula delegation was unverified. These tests pin the delegation to the proven engine and
confirm the honest ``delist_return_signs`` stub raises (per §1.5 (A1)'s MEASURED-vs-asserted line) rather
than fabricating a measurement this repo has not run.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.panel.sleeve import effective_breadth
from convexity.universe.validate.survivorship_probe import (
    delist_return_signs,
    roster_effective_breadth,
)


def test_roster_effective_breadth_delegates_to_engine():
    """§6/§3.3: ``roster_effective_breadth`` is a thin pass-through to the PROVEN engine formula, not a
    re-derivation. It must return EXACTLY ``convexity.panel.sleeve.effective_breadth`` on the same input
    (bit-identical), so no divergent breadth notion is introduced (REUSE doctrine)."""
    rng = np.random.default_rng(7)
    contrib = rng.standard_normal((52, 30))
    assert roster_effective_breadth(contrib) == effective_breadth(contrib)


def test_roster_effective_breadth_bounds():
    """§3.3: effective breadth (inverse-HHI on total |contribution|) is bounded by the roster size.

    One dominant name -> breadth near 1; equal contributions across n names -> breadth near n. This
    exercises the delegated formula's real behavior on hand-constructed extremes so a silent swap to a
    wrong statistic would be caught."""
    # Single dominant name: one column carries all the |contribution|.
    concentrated = np.zeros((10, 5))
    concentrated[:, 0] = 1.0
    assert roster_effective_breadth(concentrated) == pytest.approx(1.0, abs=1e-9)
    # Perfectly equal magnitudes across 5 names -> effective breadth == 5.
    equal = np.ones((10, 5))
    assert roster_effective_breadth(equal) == pytest.approx(5.0, abs=1e-9)


def test_delist_return_signs_is_honest_stub():
    """§1.5 (A1): the empirical half of §1.3's monotonicity assumption is UNVERIFIED in this repo; the
    probe's named home must RAISE rather than fabricate the measurement (MEASURED-vs-asserted discipline).
    """
    panel = object()  # never touched — the stub raises before reading anything
    with pytest.raises(NotImplementedError, match="not implemented"):
        delist_return_signs(panel, np.array([5, 10]), window=30)
