"""Statistical controls: CIs, DSR/PSR-style deflation, FDR (BREAKOUT §10 / §7.1).

Validation-only statistics (Invariant C: under ``validate/``). Provides the report aggregations
(convexity ratio, hit rates), block-bootstrap confidence intervals for signal outcomes, the
deflated/probabilistic-Sharpe-style multiple-test correction for the WITHIN-symbol BLUE grid, and
the Benjamini-Hochberg cross-sectional FDR for the BETWEEN-symbol universe scan (Invariant L). scipy
is permitted in validation/research code (docs/ARCHITECTURE.md §5.8).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps

from convexity.breakout.types import EntryLabel, Report

_EPS = 1e-9


def convexity_ratio(labels: list[EntryLabel]) -> float:
    """median MFE_R / max(|median MAE_R|, eps) (BREAKOUT §4 convexity_ratio formula)."""
    if not labels:
        return 0.0
    mfe = np.median([lb.mfe_r for lb in labels])
    mae = np.median([lb.mae_r for lb in labels])
    return float(mfe / max(abs(mae), _EPS))


def aggregate_report(
    labels: list[EntryLabel], *, scores: np.ndarray | None = None, n_folds: int = 1,
    baseline_table=None, ablation_table=None, by_fold=None, by_regime=None,
) -> Report:
    """Aggregate per-signal labels into the headline :class:`Report` (identical shape, Inv F).

    ``scores`` is the per-signal BREAKOUT-internal ``quality_score`` array aligned 1:1 with ``labels``
    (the §10 headline "quality-score decile monotonicity"). It is threaded as a SEPARATE score-keyed
    array, never as an :class:`EntryLabel`/downstream field (Invariant K): the score is opaque to
    composition but is BREAKOUT's own promotion-gate diagnostic. When omitted, the decile table is
    empty (a baseline that exposes no internal score still returns the identical Report shape, Inv F).
    """
    n = len(labels)
    if n == 0:
        return Report(0, n_folds, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], baseline_table or [],
                      ablation_table or [], by_fold or [], by_regime or [])
    net = np.array([lb.net_return_to_horizon_bps for lb in labels])
    return Report(
        n_signals=n,
        n_folds=n_folds,
        net_return_mean_bps=float(np.mean(net)),
        hit_plus_1r_rate=float(np.mean([lb.hit_plus_1r_before_stop for lb in labels])),
        hit_stop_first_rate=float(np.mean([lb.hit_stop_first for lb in labels])),
        median_mfe_r=float(np.median([lb.mfe_r for lb in labels])),
        median_mae_r=float(np.median([lb.mae_r for lb in labels])),
        convexity_ratio=convexity_ratio(labels),
        false_breakout_rate=float(np.mean([lb.false_breakout for lb in labels])),
        score_decile_table=score_decile_table(labels, scores=scores),
        calibration_table=[],
        baseline_table=baseline_table or [],
        ablation_table=ablation_table or [],
        by_fold=by_fold or [],
        by_regime=by_regime or [],
    )


def score_decile_table(
    labels: list[EntryLabel], *, scores: np.ndarray | None = None, n_deciles: int = 5
) -> list[dict]:
    """Net-return-by-quality-score-decile table (BREAKOUT §10 headline + promotion gate).

    Deciles signals by their per-signal BREAKOUT-internal ``quality_score`` (passed in as ``scores``,
    aligned 1:1 with ``labels``) and reports the mean net return per decile. The §10 promotion gate
    "quality-score top decile must outperform lower deciles" reads ``top_decile_outperforms`` /
    ``net_bps`` off this table. When no ``scores`` are supplied (e.g. a parameter-free baseline that
    exposes no internal score) the table is empty — the decile monotonicity gate then simply does not
    fire for that report, preserving the identical Report shape (Inv F).
    """
    if not labels or scores is None:
        return []
    score = np.asarray(scores, dtype=float)
    net = np.array([lb.net_return_to_horizon_bps for lb in labels])
    if score.size != net.size:
        raise ValueError(f"scores ({score.size}) must align 1:1 with labels ({net.size})")
    order = np.argsort(score, kind="stable")  # ascending: decile 0 = lowest score, last = highest
    out: list[dict] = []
    net_by_decile: list[float] = []
    for d in range(n_deciles):
        lo = d * len(order) // n_deciles
        hi = (d + 1) * len(order) // n_deciles
        idx = order[lo:hi]
        if idx.size:
            mean_net = float(np.mean(net[idx]))
            net_by_decile.append(mean_net)
            out.append({
                "decile": d,
                "n": int(idx.size),
                "score_lo": float(score[idx].min()),
                "score_hi": float(score[idx].max()),
                "net_bps": mean_net,
            })
    # promotion-gate flag (§10): the TOP score decile out-net-returns the BOTTOM.
    if out:
        top_outperforms = net_by_decile[-1] > net_by_decile[0]
        out[-1]["top_decile_outperforms_bottom"] = bool(top_outperforms)
    return out


def block_bootstrap_ci(
    values: np.ndarray, *, block: int = 10, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Block-bootstrap (1-min autocorrelation aware) CI for the mean of ``values`` (§10)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(n - block, 1) + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def deflated_sharpe(
    sharpe: float, *, n_obs: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Deflated-Sharpe-style probability the true Sharpe > 0, accounting for ``n_trials`` BLUE-grid
    multiplicity (BREAKOUT §10 within-symbol deflation; Bailey-Lopez de Prado)."""
    if n_obs < 2:
        return 0.0
    # expected max of n_trials independent N(0,1) Sharpe estimates (the deflation threshold).
    emc = 0.5772156649
    z = _sps.norm.ppf(1 - 1.0 / max(n_trials, 1)) if n_trials > 1 else 0.0
    z += _sps.norm.ppf(1 - 1.0 / (max(n_trials, 1) * np.e)) * emc if n_trials > 1 else 0.0
    sr0 = z / np.sqrt(n_obs)
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / np.sqrt(n_obs - 1)
    if denom <= 0:
        return 0.0
    return float(_sps.norm.cdf((sharpe - sr0) / denom))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: return a boolean accept mask controlling the FDR at ``alpha`` (Inv L)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    accept = np.zeros(m, dtype=bool)
    if k > 0:
        accept[order[:k]] = True
    return accept
