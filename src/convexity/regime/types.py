"""REGIME-internal data objects (REGIME §8).

The cross-pillar handoff ``RegimeState`` lives in :mod:`convexity.core.types` (it crosses into
LEVERAGE/SELECTION). This module holds the shapes that stay *inside* the regime pillar: the causal
return ``Panel``, a single measure's ``SubState`` (its value + per-sub-state hysteretic band + the
staleness/diagnostics the spec pins), and the neighbour-read inputs (``ExecState``/``CostState``/
``BookState``) the engine consumes from EXECUTION/COST/LEVERAGE without re-modelling them
(Invariant NREBUILD).

Everything here is pure-Python + numpy-typed, NautilusTrader-free, frozen, and causal (P6): no
field is ever populated from a future bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np

# REGIME's four ordered global states + its four sub-measures. RegimeStateName is the shared
# alias (core.types); SubName is regime-private.
RegimeStateName = Literal["CALM", "NORMAL", "STRESSED", "CRISIS"]
SubName = Literal["vol", "corr", "turb", "liq"]

# The ordered state ladder and its g(s) map (REGIME §3.1, RED). Index = severity rank.
STATE_ORDER: tuple[RegimeStateName, ...] = ("CALM", "NORMAL", "STRESSED", "CRISIS")
G_OF_STATE: Mapping[RegimeStateName, float] = {
    "CALM": 1.00,
    "NORMAL": 0.75,
    "STRESSED": 0.50,
    "CRISIS": 0.25,
}


def state_rank(state: RegimeStateName) -> int:
    """Severity rank 0..3 (CALM=0 … CRISIS=3). The worst sub-state is the max-rank one (§3.1)."""
    return STATE_ORDER.index(state)


def worst_state(a: RegimeStateName, b: RegimeStateName) -> RegimeStateName:
    """The more-stressed (higher-rank, lower-g) of two states — the discrete ``min()`` (P3)."""
    return a if state_rank(a) >= state_rank(b) else b


# --------------------------------------------------------------------------- causal return panel


@dataclass(frozen=True)
class Panel:
    """Causal cross-asset return panel at the regime decision bar (REGIME §2, §8).

    ``returns`` is ``(W, n_t)`` log-returns at the regime cadence, causally de-meaned (trailing
    mean only, P6). ``instruments`` is the live, post-staleness-drop universe (length ``n_t``).
    ``asof_bar`` is the ``F_t`` index — no row is sourced from a future bar (Invariant CAUSAL).
    """

    returns: np.ndarray  # (W, n_t)
    instruments: tuple[str, ...]
    asof_bar: int

    @property
    def n_t(self) -> int:
        return int(self.returns.shape[1])

    @property
    def window(self) -> int:
        return int(self.returns.shape[0])


# --------------------------------------------------------------------------- neighbour-read inputs


@dataclass(frozen=True)
class ExecState:
    """EXECUTION's aggregate spread/depth read for ``g^{liq}`` (REGIME §2.4, Invariant NREBUILD).

    REGIME does NOT re-model microstructure: ``spread_pct`` is the per-instrument causal percentile
    of each live name's OWN trailing spread (EXECUTION already produces it; stale/halt names are
    forced high by EXECUTION). REGIME only aggregates these scalars.
    """

    spread_pct: Mapping[str, float]  # symbol -> causal percentile in [0, 1] of its own spread


@dataclass(frozen=True)
class CostState:
    """COST's aggregate funding/carry read for ``g^{liq}`` (REGIME §2.4, Invariant NREBUILD).

    ``funding_pct`` is the per-instrument causal percentile of each live name's OWN trailing
    signed-magnitude funding (COST owns the funding model; REGIME only aggregates).
    """

    funding_pct: Mapping[str, float]  # symbol -> causal percentile in [0, 1] of |funding|


@dataclass(frozen=True)
class BookState:
    """LEVERAGE/SYSTEM-owned book snapshot the cost gate reads (REGIME §5.2, Invariant NREBUILD).

    ``book_gross_at_full`` is the time-varying ``g=1`` gross LEVERAGE permits the CURRENT book given
    its seven per-position ceilings (REGIME does NOT re-derive it — it reads it). ``current_gross``
    is the gross the book is ACTUALLY running now; it drifts away from ``g_prev·book_gross_at_full``
    as positions close (the §6.3 deadband case). Both are ``F_t``-measurable (P6).
    """

    book_gross_at_full: float
    current_gross: float


# --------------------------------------------------------------------------- per-measure sub-state


@dataclass(frozen=True)
class SubState:
    """One measure's value + its PER-SUB-STATE hysteretic band classification (REGIME §3.2, §8).

    ``value`` is in the measure's native unit (``vol_pct``/``delta_AR``/``turb_pct`` of ``d_norm``/
    ``liq_stress``). ``state`` is its band classification under THIS measure's own remembered
    entry/exit band (§3.2). ``g_sub = g(state)``.

    ``asof_bar`` is when this measure was last RECOMPUTED (== current bar for liq; the last
    scheduled-cadence bar for vol/corr/turb), and ``fresh`` is True only when recomputed this bar —
    together they make the staleness auditable (Invariant STALEATTR / ARFLOOR). ``diag`` carries the
    decomposition the spec pins (AR_t, sigma_AR raw-level std, d_norm, n_t, cond_number, lw_intensity,
    and the freeze/breach flags ar_window_deficient / ar_nt_changed / cond_breach_*).
    """

    name: SubName
    value: float
    state: RegimeStateName
    g_sub: float
    asof_bar: int
    fresh: bool
    diag: Mapping[str, float] = field(default_factory=dict)
