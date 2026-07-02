"""Mid-week forced-liquidation + weight renormalization (``docs/specs/OPS.md`` sec 3).

Module placement: sec 8's directory layout gives sec 3's renormalization NO named module (the tree
has no ``delisting/`` directory, and sec 3 is a numbered section without a module citation). Placed
here in ``ops/brake/renormalize.py`` because renormalization is book-state REPAIR invoked on a forced
exit — the same brake package that owns the forced-flatten sequence (sec 6, ``brake/kill_switch.py``)
and the degenerate-leg -> HALT escalation (sec 3.4). The degenerate-leg case defers to the brake's
flatten, so co-locating keeps the forced-exit surface in one package rather than a spec-unnamed dir.

An exogenous stopping time ``tau_delist`` (a delisting / trading halt / forced perp settlement) can
land STRICTLY between the scheduled Friday-close rebalances and force an unplanned exit that breaks
dollar-neutrality (``Sum w_long = -Sum w_short``, P6). This module detects the break and enforces the
renormalization that restores it EXACTLY, without waiting for the next scheduled rebalance (sec 3.3).

The renormalization rule (sec 3.3): after forcing the delisted name ``i`` to ``w_i -> 0``, scale up
every SURVIVING weight in the AFFECTED leg (only that leg) proportionally::

    w_j' = w_j * ( (G/2) / (G/2 - w_i) )        for every surviving j in the affected leg

This is a pure gross-restoration operation orthogonal to the sizing scheme: proportional scaling
preserves every survivor's RELATIVE weight (``w_j'/w_k' = w_j/w_k``), so it does not need to know
whether the book was equal-weighted or vol-scaled to be correct (sec 3.3 "why proportional, not
equal top-up"). An equal-dollar top-up would silently convert a vol-scaled book into a blend.

Degenerate leg (``|surviving leg| == 0``, sec 3.4): the denominator ``G/2 - w_i -> 0`` is a genuine
singularity (a leg cannot be topped up from zero survivors). OPS then emits a forced HALT rather
than renormalizing — signalled here by raising :class:`DegenerateLegError`, which the brake maps to
``HALT``.

All pure functions — offline / simulatable, no live venue dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DegenerateLegError(RuntimeError):
    """Raised when a forced exit would leave the affected leg with zero survivors (OPS.md sec 3.4).

    A single-name book cannot be dollar-neutral by construction and is out of scope for a
    proportional fix; the brake maps this to a forced ``HALT`` deferring to the next full rebalance.
    """


@dataclass(frozen=True)
class RenormResult:
    """The renormalized book after a forced mid-week exit (OPS.md sec 3.3).

    ``weights`` is the full weight vector with the delisted name zeroed and the affected leg
    rescaled; ``scale`` is the proportional factor ``(G/2)/(G/2 - w_i)`` applied to the affected
    leg's survivors (``> 1``); ``leg`` names which leg lost a name.
    """

    weights: np.ndarray
    scale: float
    leg: str  # "long" or "short"


def renormalize_after_delist(weights: np.ndarray, delisted_idx: int, *,
                             tol: float = 1e-12) -> RenormResult:
    """Zero the delisted name and proportionally rescale its leg to restore neutrality (OPS.md sec 3.3).

    Preconditions (asserted): ``weights`` is dollar-neutral immediately before the exit
    (``|Sum w| < 1e-9``, the state ``portfolio.py::quantile_weights`` produces). The delisted name
    is on the long leg (``w_i > 0``) or the short leg (``w_i < 0``); the mirror-image argument
    handles the short case with ``L``/``S`` swapped (sec 3.3 "the short-leg-delisting case").

    Returns the renormalized book with dollar-neutrality restored EXACTLY (Inv DNEUT). Raises
    :class:`DegenerateLegError` if the affected leg would have zero survivors (sec 3.4). The proof
    (sec 3.3) makes no assumption that ``weights`` are the ORIGINAL rebalance weights — only that
    they are dollar-neutral immediately before the exit — so sequential double-delisting is valid by
    induction (sec 3.4): call this once per event against the already-renormalized book.
    """
    w = np.asarray(weights, dtype=float).copy()
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    net_before = float(w.sum())
    if abs(net_before) > 1e-9:
        raise ValueError(
            f"book is not dollar-neutral before the exit (Sum w = {net_before:.3e}); "
            "renormalization's invariant-preservation argument (OPS.md sec 3.3) requires it"
        )
    w_i = float(w[delisted_idx])
    if abs(w_i) <= tol:
        # The delisted name carried no weight — zeroing it changes nothing; still neutral.
        w[delisted_idx] = 0.0
        return RenormResult(w, 1.0, "long" if w_i >= 0.0 else "short")

    if w_i > 0.0:  # long leg
        leg_mask = w > 0.0
        leg = "long"
        leg_gross = float(w[leg_mask].sum())  # == G/2
    else:  # short leg
        leg_mask = w < 0.0
        leg = "short"
        leg_gross = float(w[leg_mask].sum())  # == -G/2

    # Step 1 (sec 3.3): the exit itself breaks the identity — surviving-leg gross = leg_gross - w_i.
    surviving_leg_gross = leg_gross - w_i
    survivors = leg_mask.copy()
    survivors[delisted_idx] = False
    n_surviving = int(survivors.sum())
    if n_surviving == 0 or abs(surviving_leg_gross) <= tol:
        # Degenerate: the delisted name was the entire leg (sec 3.4) -> forced HALT, no fix.
        raise DegenerateLegError(
            f"forced exit of idx {delisted_idx} leaves the {leg} leg with {n_surviving} survivors "
            f"(surviving gross {surviving_leg_gross:.3e}); OPS.md sec 3.4 forces HALT"
        )

    # Step 2 (sec 3.3): scale surviving-leg weights by (G/2)/(G/2 - w_i) = leg_gross/surviving_gross.
    scale = leg_gross / surviving_leg_gross
    w[delisted_idx] = 0.0
    w[survivors] = w[survivors] * scale
    return RenormResult(w, scale, leg)
