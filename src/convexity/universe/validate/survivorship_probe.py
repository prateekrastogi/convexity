"""Survivorship probe — a §6-style effective-breadth check on any candidate roster (UNIVERSE.md §6, §11).

§1.3 proves a backward "modern roster" construction (``U_t := R``, the terminal-date survivor set) is
survivorship-biased with a specific, derivable sign: names in the delisted set ``D`` almost mechanically
underperformed into delisting, so ``R`` is enriched for names whose realized history looks good, inflating
any in-sample statistic. §6 records the concrete instance this repo already flagged (the equities sleeve
docstring, "SELECTED for modern liquidity").

This module provides the probe §11's directory layout names but that did not exist: given a per-name
contribution matrix, report the effective breadth (inverse-Herfindahl, REUSES
``research/xsec/universe.py::effective_breadth`` — the SAME formula §3.3's crypto liquidity-cap finding
uses, factored here, not re-derived). §1.5 (A1): the empirical half of §1.3's monotonicity assumption —
that crypto delistings are preceded by price collapse, not a rally — is NOT verified anywhere in this
repo; ``delist_return_signs`` is the not-yet-runnable stub §1.5 (A1) names as the check that WOULD close
that gap (it raises rather than fabricating a measurement, honest to §1.5's MEASURED-vs-asserted line).
"""

from __future__ import annotations

import numpy as np

from research.xsec.universe import effective_breadth


def roster_effective_breadth(contrib: np.ndarray) -> float:
    """Effective number of contributing names for a roster (UNIVERSE.md §3.3, §6; REUSES universe.py).

    ``contrib`` -- (n_weeks, n_symbols) per-name price(+funding) contribution to the book return. Thin
    pass-through to ``research.xsec.universe.effective_breadth`` (inverse-HHI on total |contribution|);
    §6 uses this to characterize a roster, NOT to certify point-in-time-ness (that is ``pit_check``). A
    low breadth relative to the roster size is one symptom (never proof) of a concentrated/flattered set.
    """
    return effective_breadth(contrib)


def delist_return_signs(panel, tau_delist: np.ndarray, *, window: int) -> np.ndarray:
    """UNVERIFIED probe stub for §1.5 (A1)'s monotonicity assumption (UNIVERSE.md §1.5).

    §1.5 (A1) states plainly: the claim that crypto delistings are overwhelmingly preceded by price
    collapse (not a rally) is asserted from general market-structure knowledge, NOT from a persisted
    measurement in this repo. The check that WOULD close the gap — for every ``i`` with ``tau_delist_i <
    ∞``, compute ``g_i`` = mean return over the trailing ``window`` ending at ``tau_delist_i`` and report
    the sign distribution — is not yet built. This stub is its named home (§1.5, §11's
    ``validate/survivorship_probe.py``); it raises rather than fabricating the measurement, honest to
    §1.5's MEASURED-vs-asserted discipline.
    """
    raise NotImplementedError(
        "delist-return-sign measurement is not implemented (UNIVERSE.md §1.5 (A1)): the empirical half of "
        "the monotonicity assumption is an UNVERIFIED assertion in this repo, not a measured regularity; "
        "this stub is its named home, and it must not fabricate the measurement it has not run"
    )


__all__ = ["roster_effective_breadth", "delist_return_signs"]
