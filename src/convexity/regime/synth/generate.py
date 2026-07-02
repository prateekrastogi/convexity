"""Adversarial synthetic panels for REGIME's own tests (REGIME.md §7, §9 ``synth/`` tree).

This is the DETECTOR-scope minimal seed of the ``regime/synth/`` tree §9 reserves. The one generator
implemented here is the exact panel the Marchenko-Pastur Invariant MP (§2.4) names: an ``N x T`` matrix
of i.i.d. Gaussian returns with NO true cross-sectional structure (no common factor, no correlation),
so that any nonzero "structure" a covariance-based statistic reports is pure estimation noise. The
full adversarial suite (dispersion spikes/collapses, vol-regime shifts, persistent-bleed drawdowns for
Invariant STOP-XFER, §7.2 baselines) is the scoped-out ``synth/generate.py`` build contract (§9); this
seeds it with the single generator the in-scope MP demonstration needs.
"""

from __future__ import annotations

import numpy as np


def iid_no_structure_panel(n_symbols: int, n_obs: int, *, seed: int) -> np.ndarray:
    """An ``N x T`` (symbols x observations) matrix of i.i.d. standard-normal returns (REGIME.md §2.4).

    NO common factor, NO cross-sectional correlation, NO true low-variance directions: the population
    covariance is exactly the identity, so the TRUE absorption ratio (fraction of variance in the top
    principal component) is ``1/N``. Any *sample* covariance estimated from this panel when ``N``
    approaches ``T`` has eigenvalues spread by the Marchenko-Pastur law purely from estimation noise
    (§2.4) -- precisely the regime this generator exists to exhibit in the Invariant MP demonstration.

    Returns a ``(n_symbols, n_obs)`` array; the caller decides orientation for its covariance estimate.
    """
    if n_symbols < 2 or n_obs < 2:
        raise ValueError("need at least 2 symbols and 2 observations for a covariance estimate")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_symbols, n_obs))
