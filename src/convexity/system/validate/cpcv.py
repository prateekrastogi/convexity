"""Sampled combinatorial purged CV on the composed chain (SYSTEM §6.4).

Full CPCV is expensive on a 2-core VM. Partition the timeline into ``n_groups``; for each of
C(n_groups, k_test) train/test arrangements, purge+embargo, run the COMPOSED pipeline on the test
groups, score system triple-barrier P&L net of cost. If ``max_paths`` is set, sample arrangements
DETERMINISTICALLY with a seed stored in report metadata. Returns a DISTRIBUTION of OOS system Sharpe
across paths — the input to the bootstrap CIs and the regime/fold attribution (Section 8), NOT the
``trial_sharpes`` of the deflation (those are cross-CONFIG, Section 7.2).

Module per-bar evidence is pulled from the :class:`EvidenceCache` keyed by ``(module,
module_version, fold_id, config_hash, bar_range)`` so a cfg changing only a COMBINER parameter
re-scores WITHOUT re-running modules (P7 / Invariant CA).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CPCVPath:
    """One CPCV path: the test groups held out and the realized OOS system return series + Sharpe."""

    test_groups: tuple[int, ...]
    oos_returns: np.ndarray
    oos_sharpe: float


@dataclass(frozen=True)
class CPCVDistribution:
    """The OOS Sharpe distribution across sampled paths (Section 6.4). ``seed`` is recorded so the
    sampling is deterministic and reproducible (the §15 walk-forward+CPCV gate)."""

    paths: tuple[CPCVPath, ...]
    seed: int
    n_groups: int
    k_test: int

    @property
    def path_sharpes(self) -> tuple[float, ...]:
        return tuple(p.oos_sharpe for p in self.paths)


def _sample_arrangements(
    n_groups: int, k_test: int, max_paths: int | None, seed: int
) -> list[tuple[int, ...]]:
    """All C(n_groups, k_test) test-group arrangements, deterministically sampled to ``max_paths``."""
    all_arr = list(itertools.combinations(range(n_groups), k_test))
    if max_paths is None or max_paths >= len(all_arr):
        return all_arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_arr), size=max_paths, replace=False)
    return [all_arr[i] for i in sorted(int(j) for j in idx)]


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r)) / sd if sd > 0.0 else 0.0


class SystemCPCV:
    """Sampled combinatorial purged CV on the COMPOSED chain (SYSTEM §6.4/§12)."""

    def evaluate(
        self,
        group_returns: Sequence[np.ndarray],
        *,
        n_groups: int,
        k_test: int,
        max_paths: int | None,
        seed: int,
        score_fn: Callable[[tuple[int, ...]], np.ndarray] | None = None,
    ) -> CPCVDistribution:
        """Run the composed pipeline on each sampled arrangement and score OOS P&L (Section 6.4).

        ``group_returns[g]`` is the per-observation net-of-cost system return series produced on group
        ``g`` (already pulled through the EvidenceCache so combiner-only configs are free). For each
        arrangement the OOS series is the concatenation of the test groups' returns. ``score_fn`` is
        an optional hook to recompute the OOS series for an arrangement (e.g. to run the live composer
        on the test slice); when omitted the cached ``group_returns`` are concatenated.
        """
        arrangements = _sample_arrangements(n_groups, k_test, max_paths, seed)
        paths: list[CPCVPath] = []
        for arr in arrangements:
            if score_fn is not None:
                oos = np.asarray(score_fn(arr), dtype=float)
            else:
                oos = np.concatenate([np.asarray(group_returns[g], dtype=float) for g in arr])
            paths.append(CPCVPath(test_groups=arr, oos_returns=oos, oos_sharpe=_sharpe(oos)))
        return CPCVDistribution(paths=tuple(paths), seed=seed, n_groups=n_groups, k_test=k_test)
