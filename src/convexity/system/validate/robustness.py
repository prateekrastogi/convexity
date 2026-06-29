"""Drop-top-N + block-bootstrap CIs + regime/fold attribution (SYSTEM §8.1).

  1. DROP-TOP-N: recompute system Sharpe after removing the N most profitable trades (and the N
     worst), and after removing the single best instrument / fold / regime. A system whose edge
     evaporates when the top-N trades are dropped is a fat-tail/luck artifact, not a process — flag
     it (the convexity-vs-luck question from the convexity-repo backtests).
  2. BLOCK-BOOTSTRAP CIs on system Sharpe and per-pillar attribution, with block length ≥ the
     autocorrelation / vol-clustering scale so dependence is preserved (iid bootstrap is FORBIDDEN).
  3. REGIME / FOLD ATTRIBUTION: decompose by regime bucket (vol regime, trend/chop, asset class) and
     by fold; report whether one regime / instrument / fold drives the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r)) / sd if sd > 0.0 else 0.0


def drop_top_n(trade_pnls: np.ndarray, n: int) -> dict[str, float]:
    """Recompute the Sharpe-like ratio after dropping the N best and the N worst trades (Section 8.1).

    Returns {full, drop_best_n, drop_worst_n}. If the edge evaporates under ``drop_best_n``, the
    result is a fat-tail/luck artifact (the caller flags ``luck_artifact``)."""
    p = np.sort(np.asarray(trade_pnls, dtype=float))
    full = _sharpe(p)
    drop_best = _sharpe(p[: max(len(p) - n, 0)]) if len(p) > n else 0.0
    drop_worst = _sharpe(p[min(n, len(p)) :]) if len(p) > n else 0.0
    return {"full": full, "drop_best_n": drop_best, "drop_worst_n": drop_worst}


def block_bootstrap_ci(
    returns: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    ci_level: float,
    seed: int,
    statistic=_sharpe,
) -> tuple[float, float]:
    """Moving-block bootstrap CI on a statistic of a DEPENDENT return series (Section 8.1).

    Resamples overlapping blocks of length ``block_length`` (≥ the vol-clustering scale) so serial
    dependence is preserved — iid bootstrap is FORBIDDEN for dependent returns (SYSTEM §17). Returns
    the (lower, upper) CI bounds at ``ci_level``. Deterministic given ``seed``."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < block_length or n < 2:
        s = statistic(r)
        return (s, s)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block_length)  # ceil
    starts_pool = n - block_length + 1
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, starts_pool, size=n_blocks)
        sample = np.concatenate([r[s : s + block_length] for s in starts])[:n]
        stats[b] = statistic(sample)
    alpha = 1.0 - ci_level
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return (lo, hi)


@dataclass(frozen=True)
class RobustnessReport:
    """Output of :meth:`RobustnessSuite.run` (Section 8.1). ``luck_artifact`` is True when dropping
    the top-N trades collapses the edge; ``regime_attribution`` / ``fold_attribution`` localize it."""

    drop_top_n: Mapping[str, float]
    luck_artifact: bool
    sharpe_ci: tuple[float, float]
    regime_attribution: Mapping[str, float] = field(default_factory=dict)
    fold_attribution: Mapping[str, float] = field(default_factory=dict)
    best_instrument: str | None = None
    best_fold: int | None = None


class RobustnessSuite:
    """Drop-top-N + block-bootstrap + regime/fold attribution (SYSTEM §8.1/§12)."""

    def run(
        self,
        path_returns: np.ndarray,
        trade_pnls: np.ndarray,
        regimes: Mapping[str, np.ndarray] | None,
        cfg,
        *,
        fold_returns: Sequence[np.ndarray] | None = None,
        instrument_pnls: Mapping[str, float] | None = None,
        seed: int = 0,
    ) -> RobustnessReport:
        """Run the robustness suite (Section 8.1). ``regimes`` maps a regime-bucket label → its
        per-observation returns; ``fold_returns`` is the per-fold OOS series; ``instrument_pnls``
        the per-instrument total P&L (to find the single best instrument)."""
        defaults = cfg.defaults
        dtn = drop_top_n(trade_pnls, defaults.drop_top_n)
        luck = dtn["full"] > 0.0 and dtn["drop_best_n"] <= 0.0
        ci = block_bootstrap_ci(
            path_returns,
            block_length=defaults.bootstrap_block_length,
            n_boot=defaults.bootstrap_n,
            ci_level=defaults.bootstrap_ci_level,
            seed=seed,
        )
        regime_attr = {k: _sharpe(v) for k, v in (regimes or {}).items()}
        fold_attr = {f"fold_{i}": _sharpe(r) for i, r in enumerate(fold_returns or [])}
        best_inst = max(instrument_pnls, key=instrument_pnls.get) if instrument_pnls else None
        best_fold = (
            max(range(len(fold_returns)), key=lambda i: _sharpe(fold_returns[i]))
            if fold_returns
            else None
        )
        return RobustnessReport(
            drop_top_n=dtn,
            luck_artifact=luck,
            sharpe_ci=ci,
            regime_attribution=regime_attr,
            fold_attribution=fold_attr,
            best_instrument=best_inst,
            best_fold=best_fold,
        )
