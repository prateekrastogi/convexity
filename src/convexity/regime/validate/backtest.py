"""REGIME's BLUE-grid backtest -> the ONE SYSTEM trial ledger (REGIME.md §7.4 / Invariant LEDGER).

This is the DETECTOR-scope minimal seed of the ``regime/validate/backtest.py`` §9 reserves. §7.4 says
the full REGIME-gated book's weekly net-return series is fed to ``system/validate/deflated.py``'s
``DeflatedSharpe`` exactly as any other candidate config, entering the SAME persistent trial ledger
SYSTEM's Invariant SG requires: every ``(Delta_target, ARL_0_target, dd_trigger, dd_reset)`` combination
REGIME's BLUE grid searches is ONE more ledger config. REGIME does NOT run a private, uncounted DSR
gate. The full net-return backtest per tuple is out of scope for the detector core; what IS binding
here -- and what Invariant LEDGER's equality test needs -- is the bookkeeping guarantee: the count of
distinct tuples this module SCORES equals the count of REGIME-tagged entries it APPENDS to the shared
``TrialLedger`` (a tuple scored but not appended, or appended under a private ledger, is the bug).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np

from convexity.regime.config.defaults import REGIME_PARAM_REGISTRY
from convexity.system.types import TrialRecord
from convexity.system.validate.trials import TrialLedger

# The module-version tag stamped on every REGIME-contributed ledger row, so the REGIME-tagged subset of
# the ONE shared ledger is identifiable for Invariant LEDGER's equality count (§7.4).
REGIME_LEDGER_TAG = "regime"


def blue_grid_tuples() -> tuple[tuple[float, float], ...]:
    """The ``(Delta_target, ARL_0_target)`` BLUE grid REGIME searches (REGIME.md §6, defaults.py).

    Read from the SAME ``REGIME_PARAM_REGISTRY`` the resolved config and the K_regime_max cardinality
    bound (§6, test_config) read -- one source of truth, so the ledger count and the compute-cap check
    can never silently diverge. The DD-stop ``(dd_trigger, dd_reset)`` axis (§5.4) is a nice-to-have
    overlay, not a registered detector param, so the v1 detector-scope grid is the two governing axes.
    """
    grids = {p.name: p.blue_grid for p in REGIME_PARAM_REGISTRY.params() if p.blue_grid is not None}
    deltas = grids["delta_target_grid"]
    arls = grids["arl_0_target_grid"]
    return tuple((float(d), float(a)) for d in deltas for a in arls)


def _config_hash(delta_target: float, arl_0_target: float) -> str:
    """Content address for one REGIME BLUE tuple (deterministic, REGIME-tagged; §7.4)."""
    payload = f"{REGIME_LEDGER_TAG}|delta_target={delta_target!r}|arl_0_target={arl_0_target!r}"
    return hashlib.blake2s(payload.encode(), digest_size=16).hexdigest()


def score_blue_grid_into_ledger(
    ledger: TrialLedger,
    *,
    score_fn: Callable[[float, float], tuple[float, np.ndarray]],
    tuples: Iterable[tuple[float, float]] | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Score each BLUE tuple and APPEND one REGIME-tagged record to the SHARED ledger (Invariant LEDGER).

    ``score_fn(delta_target, arl_0_target) -> (oos_sharpe_annual, oos_returns)`` is the (out-of-scope)
    backtest that produces the REGIME-gated book's OOS Sharpe + weekly net-return path for one tuple;
    it is injected so this module carries only the LEDGER bookkeeping, not the backtest engine. For
    EVERY tuple scored, exactly one record is appended to the passed-in ``ledger`` (the ONE SYSTEM
    ledger, never a private one). Returns ``(n_scored, config_hashes)`` so a caller / the Invariant
    LEDGER test can assert the count of tuples scored equals the count of REGIME-tagged ledger entries.
    """
    grid: Sequence[tuple[float, float]] = tuple(tuples) if tuples is not None else blue_grid_tuples()
    hashes: list[str] = []
    for delta_target, arl_0_target in grid:
        oos_sharpe, oos_returns = score_fn(delta_target, arl_0_target)
        ch = _config_hash(delta_target, arl_0_target)
        ledger.append(
            TrialRecord(
                config_hash=ch,
                module_versions={REGIME_LEDGER_TAG: "v1"},
                blue_point={"delta_target": delta_target, "arl_0_target": arl_0_target},
                oos_sharpe=float(oos_sharpe),
                oos_returns_ref=ch,
                oos_returns=np.asarray(oos_returns, dtype=float),
            )
        )
        hashes.append(ch)
    return len(grid), tuple(hashes)


def regime_tagged_records(ledger: TrialLedger) -> Iterator[TrialRecord]:
    """The REGIME-contributed subset of the shared ledger (rows stamped ``REGIME_LEDGER_TAG``; §7.4)."""
    return (r for r in ledger.records if REGIME_LEDGER_TAG in r.module_versions)
