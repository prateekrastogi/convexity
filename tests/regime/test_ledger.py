"""Invariant LEDGER equality (REGIME.md §7.4): tuples SCORED == REGIME-tagged entries in the ONE ledger.

The in-scope cardinality half -- "REGIME's BLUE grid product <= K_regime_max" -- is pinned by
``test_config.py::test_k_regime_max_bounds_the_blue_grid_cardinality_fed_to_the_system_ledger``. This
file adds Invariant LEDGER's STATED equality test: "the number of distinct (Delta_target, ARL_0_target,
dd_trigger, dd_reset) tuples REGIME's backtest.py has ever scored equals the count of REGIME-tagged
entries in the SYSTEM ledger." It is binding because ``score_blue_grid_into_ledger`` appends to the
SHARED ``TrialLedger`` (never a private one), and the test would fail if a tuple were scored without a
matching ledger row -- the "four-green-modules trap" the invariant exists to catch, applied to REGIME.
"""

from __future__ import annotations

import numpy as np

from convexity.regime.validate import backtest as bt
from convexity.system.validate.trials import TrialLedger


def _score_fn(seed_offset: int = 0):
    """A stand-in per-tuple backtest producing (annual OOS Sharpe, weekly OOS return path)."""
    rng = np.random.default_rng(seed_offset)

    def score(delta_target: float, arl_0_target: float) -> tuple[float, np.ndarray]:
        return float(rng.normal(0.5, 0.2)), rng.normal(0.0, 0.02, size=52)

    return score


def test_ledger_equality_scored_tuples_equal_regime_tagged_entries():
    """§7.4 Invariant LEDGER: every scored tuple appears exactly once as a REGIME-tagged ledger row."""
    ledger = TrialLedger()
    n_scored, hashes = bt.score_blue_grid_into_ledger(ledger, score_fn=_score_fn())

    tagged = list(bt.regime_tagged_records(ledger))
    assert n_scored == len(tagged)          # scored == REGIME-tagged entries (the stated equality)
    assert n_scored == ledger.k             # and they are the only rows in this fresh shared ledger
    assert len(set(hashes)) == n_scored     # distinct config_hashes: no silent tuple collision


def test_ledger_feeds_the_shared_ledger_not_a_private_one():
    """§7.4: REGIME contributes into the SAME ledger other modules use; it never keeps a private one."""
    shared = TrialLedger()
    # A pre-existing NON-regime row (another module's contribution) must survive untouched.
    from convexity.system.types import TrialRecord

    shared.append(
        TrialRecord(
            config_hash="system-abc",
            module_versions={"system": "v1"},
            blue_point={"foo": 1},
            oos_sharpe=0.3,
            oos_returns_ref="system-abc",
            oos_returns=np.zeros(52),
        )
    )
    n_scored, _ = bt.score_blue_grid_into_ledger(shared, score_fn=_score_fn(1))

    tagged = list(bt.regime_tagged_records(shared))
    assert len(tagged) == n_scored                       # REGIME-tagged count == tuples scored
    assert shared.k == n_scored + 1                      # the shared ledger accumulated REGIME + system
    assert any(r.config_hash == "system-abc" for r in shared.records)  # foreign row untouched


def test_scored_but_not_appended_would_break_equality():
    """§7.4: the invariant is BINDING -- scoring a tuple without appending breaks the equality count.

    Simulates the bug the invariant catches: only a strict SUBSET of the grid reaches the ledger, so a
    claim of the full grid cardinality would not equal the REGIME-tagged ledger count."""
    ledger = TrialLedger()
    full_grid = bt.blue_grid_tuples()
    subset = full_grid[: len(full_grid) - 2]  # "forget" to score/append two tuples
    n_scored, _ = bt.score_blue_grid_into_ledger(ledger, score_fn=_score_fn(2), tuples=subset)

    tagged = list(bt.regime_tagged_records(ledger))
    assert n_scored == len(tagged) == len(subset)        # bookkeeping stays honest for what WAS scored
    assert len(tagged) != len(full_grid)                 # but does NOT match a full-grid claim (the bug)


def test_blue_grid_tuples_match_the_registry_cardinality():
    """§6: the tuples scored come from the SAME registry the K_regime_max cardinality bound reads."""
    from convexity.regime.config.defaults import REGIME_PARAM_REGISTRY

    assert len(bt.blue_grid_tuples()) == REGIME_PARAM_REGISTRY.blue_grid_cardinality()
