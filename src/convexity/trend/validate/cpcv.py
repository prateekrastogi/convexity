"""SampledCPCV — sampled combinatorial purged CV for finalists (TREND §12/§14 cpcv).

Full CPCV is expensive; v1 uses SAMPLED CPCV for finalists only (§14). This builds a sampled set of
combinatorial train/test groupings of the blocks, each purged + embargoed, so the multiple-testing
penalty (stats.py) counts the searched configs honestly. Validation-only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


class SampledCPCV:
    """Sampled combinatorial purged cross-validation (§12/§14)."""

    def __init__(self, n_blocks: int = 6, n_test_blocks: int = 2, max_samples: int = 10) -> None:
        self.n_blocks = n_blocks
        self.n_test_blocks = n_test_blocks
        self.max_samples = max_samples

    def splits(self, n: int, embargo: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return up to ``max_samples`` purged (train_idx, test_idx) combinatorial splits (§12)."""
        edges = np.linspace(0, n, self.n_blocks + 1, dtype=int)
        blocks = [np.arange(edges[i], edges[i + 1]) for i in range(self.n_blocks)]
        all_combos = list(combinations(range(self.n_blocks), self.n_test_blocks))
        rng = np.random.default_rng(seed)
        if len(all_combos) > self.max_samples:
            picks = rng.choice(len(all_combos), size=self.max_samples, replace=False)
            combos = [all_combos[i] for i in picks]
        else:
            combos = all_combos

        out = []
        for combo in combos:
            test_idx = np.concatenate([blocks[i] for i in combo])
            test_set = set(test_idx.tolist())
            # purge + embargo: drop train bars within `embargo` of any test block.
            train = []
            for i in range(self.n_blocks):
                if i in combo:
                    continue
                for j in blocks[i]:
                    near = any(abs(int(j) - int(t)) <= embargo for t in (test_idx[0], test_idx[-1]))
                    if int(j) not in test_set and not near:
                        train.append(int(j))
            out.append((np.array(sorted(train)), np.array(sorted(test_idx))))
        return out
