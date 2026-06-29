"""EvidenceCache — memoize per-bar module evidence (SYSTEM §9, Invariant CA).

The expensive work is each module's per-bar evidence (breakout proposals, vol forecasts, ceilings,
trend beliefs) on each fold. Memoize it keyed by ``(module, module_version, fold_id, config_hash,
bar_range)``. A SYSTEM config that changes ONLY a COMBINER parameter (precedence, cadence,
hysteresis, budget rule) has the SAME keys for all module evidence and re-scores from cache for FREE
— only the cheap composition + labeling + deflation re-runs (P7). The combiner-only re-score
re-running ZERO module computation is Invariant CA.

In-memory + content-addressed here (a real deployment is on-disk, regime-bucketed); the contract is
the key shape and the hit/miss accounting, which the Invariant-CA test asserts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EvidenceCache:
    """Content-addressed memo of module evidence (Section 9). Tracks hits/misses so the Invariant-CA
    test can assert a combiner-only config change yields a 100% module-evidence hit rate."""

    _store: dict[tuple, object] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def key(module: str, module_version: str, fold_id: str, config_hash: str, bar_range: tuple) -> tuple:
        """The §9 cache key. ``config_hash`` is the combiner coordinate hash; a combiner-only change
        does NOT alter ``module``/``module_version``/``fold_id``/``bar_range``, so module evidence
        hits even when ``config_hash`` differs — keys for evidence omit the combiner hash."""
        return (module, module_version, fold_id, bar_range)

    def get_or_compute(
        self,
        module: str,
        module_version: str,
        fold_id: str,
        config_hash: str,
        bar_range: tuple,
        compute_fn: Callable[[], object],
    ) -> object:
        """Memoize a module's per-bar evidence (Section 9). Keyed so a combiner-only config re-scores
        from cache for free (Invariant CA): the key deliberately excludes ``config_hash`` because
        module evidence does not depend on the combiner coordinate — only on the module version + fold
        + bar range. The cheap composition layer is what re-runs with the new ``config_hash``."""
        k = self.key(module, module_version, fold_id, config_hash, bar_range)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        value = compute_fn()
        self._store[k] = value
        return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
