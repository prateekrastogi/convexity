"""TrialLedger — the persistent trial ledger → effective N_sys (SYSTEM §7.1/§7.5, Inv SG/RP).

Every composed config SYSTEM ever scored against any validation fold — across ALL overnight runs to
date — appears here, and ``K``/``Var[SR_trials]``/``N_sys`` are DERIVED from it, not asserted
(Invariant SG). A config promoted "because it looked best" without its siblings being counted, or a
fresh nightly search that does not accumulate, is the overfit-by-iteration trap (Invariant RP): the
re-promotion policy appends every night's configs to the SAME ledger, so ``N_sys`` accumulates and
the consistency law (``Var[SR_trials]`` and ``N_sys`` over the same K configs) continues to hold.

The product of module BLUE-grid cardinalities is recorded as ``n_sys_nominal`` telemetry only; it is
NEVER fed to ``SR*₀`` (Section 7.1 consistency law).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from convexity.system.types import TrialRecord
from convexity.system.validate.deflated import effective_trials


@dataclass
class TrialLedger:
    """The persistent, accumulating trial ledger (Invariants SG/RP). Deduplicated by ``config_hash``:
    the ledger length equals the number of DISTINCT configs scored against any fold."""

    _records: dict[str, TrialRecord] = field(default_factory=dict)

    def append(self, record: TrialRecord) -> None:
        """Append a scored config (idempotent on ``config_hash``). Every overnight search appends its
        configs here so trials accumulate across calendar time (Invariant RP)."""
        self._records[record.config_hash] = record

    def extend(self, records) -> None:
        for r in records:
            self.append(r)

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return tuple(self._records[h] for h in sorted(self._records))

    @property
    def k(self) -> int:
        """K = the number of DISTINCT configs scored (the ledger length, Invariant SG)."""
        return len(self._records)

    def trial_sharpes(self) -> np.ndarray:
        """{ SR_1, …, SR_K } — the per-LEDGER-CONFIG OOS (annualized) Sharpes (Section 7.1)."""
        return np.array([r.oos_sharpe for r in self.records], dtype=float)

    def trial_returns(self) -> list[np.ndarray]:
        """The per-config OOS path-return series for the cross-config correlation matrix (Section 7.1)."""
        return [np.asarray(r.oos_returns, dtype=float) for r in self.records if r.oos_returns is not None]

    def n_sys(self, rho_cl: float) -> tuple[float, int]:
        """Effective N_sys + k_distinct over the ACCUMULATED ledger (Section 7.1). ``Var[SR_trials]``
        and ``N_sys`` reference the SAME ledger config set (the consistency law, Invariant DSR)."""
        returns = self.trial_returns()
        if len(returns) < 2:
            return 1.0, self.k
        return effective_trials(returns, rho_cl)
