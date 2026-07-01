"""Combinatorial Purged CV over the xsec momentum LOOKBACK FAMILY (REUSE SYSTEM ``SystemCPCV``).

The xsec engine (``research.xsec.engine.run``) already produces, for one config, the WEEKLY net-return
series with the point-in-time universe + no-lookahead baked in (per-day ``listed`` mask + per-signal
trailing-window warmup + Friday-close cadence). One observation = one NON-overlapping week (label
horizon 0 — a week is realized inside itself). So CPCV here operates directly on those weekly series:
partition the ~259-week timeline into ``n_groups`` contiguous blocks, hold out every C(n_groups,
k_test) combination of blocks as the TEST path, and score OOS Sharpe on the concatenated test blocks.
Purging is trivial (horizon 0); the ``embargo`` (>= 1 rebalance = 1 week) drops the week on EACH side
of every held-out test block from the complement so no train week touches a test-block boundary — the
only leakage channel once the universe/lookahead are already handled inside ``run``.

Two products, both keyed to the SAME C(n_groups, k_test) split family:

  * PATH-SHARPE DISTRIBUTION for the SELECTED config — REUSE ``convexity.system.validate.cpcv.SystemCPCV``.
    We feed it the selected config's per-GROUP weekly returns; it concatenates the test groups per
    arrangement and returns the OOS Sharpe per path. We then summarize (mean/median/p05/min/%>0/%>0.5).
  * PBO (probability of backtest overfitting) over the LOOKBACK FAMILY — the López de Prado
    logit-of-relative-rank estimator. For each split we pick the config that is BEST IN-SAMPLE (highest
    Sharpe on the TRAIN complement, embargo-purged), find its rank on the TEST path among the family,
    map the relative rank to a logit, and PBO = fraction of splits whose logit < 0 (best-IS config lands
    at or below the median OOS). This is the honest test of "is picking the best lookback overfitting?".

The CPCV mean path Sharpe (of the selected config) and its HAIRCUT vs the naive full-sample Sharpe fall
straight out of the path distribution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from convexity.system.validate.cpcv import CPCVDistribution, SystemCPCV

from research.xsec.metrics import PERIODS_PER_YEAR


def _sharpe_ann(returns: np.ndarray) -> float:
    """Annualized Sharpe of a weekly return series (52.18 periods/yr); 0 on degenerate input."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    return float(np.mean(r) / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0.0 else 0.0


def _group_bounds(n_weeks: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal [lo, hi) week blocks tiling the timeline (remainder to the last blocks)."""
    edges = np.linspace(0, n_weeks, n_groups + 1).astype(int)
    return [(int(edges[g]), int(edges[g + 1])) for g in range(n_groups)]


def _split_weeks(
    bounds: list[tuple[int, int]], test_groups: tuple[int, ...], *, embargo: int, n_weeks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Test week-indices (the held-out blocks) and the embargo-purged TRAIN week-indices (the rest).

    Test = union of the test blocks. Train = all other weeks MINUS an ``embargo``-week collar on each
    side of every test block (so no train week sits within ``embargo`` of a test-block boundary). With
    horizon-0 weekly labels this collar is the whole purge — it closes the only train/test leak once the
    point-in-time universe + no-lookahead are already enforced inside the engine that made the series.
    """
    test_mask = np.zeros(n_weeks, dtype=bool)
    embargo_mask = np.zeros(n_weeks, dtype=bool)
    for g in test_groups:
        lo, hi = bounds[g]
        test_mask[lo:hi] = True
        embargo_mask[max(0, lo - embargo) : min(n_weeks, hi + embargo)] = True
    train_mask = ~embargo_mask  # complement minus the collar (collar ⊇ the test blocks themselves)
    return np.flatnonzero(test_mask), np.flatnonzero(train_mask)


@dataclass(frozen=True)
class PBOResult:
    """López de Prado PBO over the lookback family across the CPCV split set."""

    pbo: float  # fraction of splits where best-IS config ranks at/below OOS median (logit <= 0)
    n_splits: int
    n_configs: int
    logits: tuple[float, ...]  # per-split logit of the best-IS config's OOS relative rank
    best_is_labels: tuple[str, ...]  # the config picked best-in-sample on each split's train
    oos_ranks: tuple[float, ...]  # its OOS relative rank in (0,1) on each split's test


@dataclass(frozen=True)
class PathSharpeSummary:
    """Summary of the selected config's OOS path-Sharpe distribution + the CPCV haircut."""

    n_paths: int
    mean: float
    median: float
    p05: float
    minimum: float
    frac_pos: float  # fraction of paths with OOS Sharpe > 0
    frac_gt_half: float  # fraction with OOS Sharpe > 0.5
    naive_full_sharpe: float  # annualized Sharpe on the full-sample selected series
    haircut: float  # naive_full_sharpe - mean (positive => CPCV mean is below the naive number)


@dataclass(frozen=True)
class CPCVReport:
    """The whole CPCV verdict for the xsec momentum family."""

    n_groups: int
    k_test: int
    embargo_weeks: int
    n_weeks: int
    n_paths: int
    selected_label: str
    family_labels: tuple[str, ...]
    path_summary: PathSharpeSummary
    pbo: PBOResult
    distribution: CPCVDistribution  # the raw SystemCPCV distribution for the selected config


def _relative_rank(oos_sharpe: float, family_oos: np.ndarray) -> float:
    """Relative rank of ``oos_sharpe`` within the family's OOS Sharpes, in the open interval (0, 1).

    Rank = (# configs it beats + 0.5 for itself) / N, so the median config maps to ~0.5. The +0.5
    (average-rank convention) keeps the logit finite even at the extremes without a separate clamp.
    """
    n = family_oos.size
    beats = float(np.sum(family_oos < oos_sharpe))
    return (beats + 0.5) / n


def compute_pbo(
    family_series: dict[str, np.ndarray],
    *,
    n_groups: int,
    k_test: int,
    embargo: int,
) -> PBOResult:
    """PBO over the lookback family via the CPCV split set (LdP logit-of-relative-rank).

    ``family_series`` maps each config label to its full-sample weekly net-return series (all the SAME
    length — the aligned family). For each C(n_groups, k_test) split: score every config IN-SAMPLE on the
    embargo-purged TRAIN weeks, pick the argmax-Sharpe config, then take THAT config's OOS relative rank
    on the TEST weeks and its logit. PBO = P(logit <= 0) = fraction of splits where the best-IS config is
    at/below the median out-of-sample.
    """
    labels = tuple(family_series.keys())
    mat = np.vstack([np.asarray(family_series[lab], dtype=float) for lab in labels])
    n_configs, n_weeks = mat.shape
    bounds = _group_bounds(n_weeks, n_groups)

    logits: list[float] = []
    picked: list[str] = []
    oos_ranks: list[float] = []
    for test_groups in itertools.combinations(range(n_groups), k_test):
        test_idx, train_idx = _split_weeks(bounds, test_groups, embargo=embargo, n_weeks=n_weeks)
        if train_idx.size < 2 or test_idx.size < 2:
            continue
        is_sharpes = np.array([_sharpe_ann(mat[c, train_idx]) for c in range(n_configs)])
        best = int(np.argmax(is_sharpes))
        oos_sharpes = np.array([_sharpe_ann(mat[c, test_idx]) for c in range(n_configs)])
        w = _relative_rank(oos_sharpes[best], oos_sharpes)
        logit = float(np.log(w / (1.0 - w)))
        logits.append(logit)
        picked.append(labels[best])
        oos_ranks.append(float(w))

    n_splits = len(logits)
    pbo = (sum(1 for x in logits if x <= 0.0) / n_splits) if n_splits else 0.0
    return PBOResult(
        pbo=pbo,
        n_splits=n_splits,
        n_configs=n_configs,
        logits=tuple(logits),
        best_is_labels=tuple(picked),
        oos_ranks=tuple(oos_ranks),
    )


def path_distribution(
    selected_series: np.ndarray,
    *,
    n_groups: int,
    k_test: int,
    embargo: int,
    seed: int = 0,
    max_paths: int | None = None,
) -> tuple[CPCVDistribution, PathSharpeSummary]:
    """OOS path-Sharpe distribution for the SELECTED config (REUSE SYSTEM ``SystemCPCV``).

    We chop ``selected_series`` into the same ``n_groups`` blocks and hand SystemCPCV the per-group
    return arrays; for each C(n_groups, k_test) arrangement it concatenates the test groups and scores
    OOS Sharpe. The ``embargo`` week on each side of every test block is DROPPED from that group's
    contribution BEFORE handing groups to SystemCPCV, so the OOS series never straddles a boundary — the
    boundary weeks belong to the train complement, exactly as in ``compute_pbo``.

    Note SystemCPCV scores per-observation Sharpe (mean/std); we re-express each path Sharpe in
    ANNUALIZED terms (x sqrt 52.18) in the summary so it is comparable to the naive full-sample Sharpe.
    """
    r = np.asarray(selected_series, dtype=float)
    n_weeks = r.size
    bounds = _group_bounds(n_weeks, n_groups)
    group_returns: list[np.ndarray] = []
    for lo, hi in bounds:
        # Interior of the block, embargo-trimmed at both edges so the held-out test path is boundary-free.
        g_lo = min(hi, lo + embargo)
        g_hi = max(lo, hi - embargo)
        group_returns.append(r[g_lo:g_hi] if g_hi > g_lo else r[lo:hi])

    dist = SystemCPCV().evaluate(
        group_returns,
        n_groups=n_groups,
        k_test=k_test,
        max_paths=max_paths,
        seed=seed,
    )
    # SystemCPCV's oos_sharpe is per-observation; annualize each path for a like-for-like haircut.
    ann = np.array([_sharpe_ann(p.oos_returns) for p in dist.paths])
    naive = _sharpe_ann(r)
    summary = PathSharpeSummary(
        n_paths=int(ann.size),
        mean=float(np.mean(ann)) if ann.size else 0.0,
        median=float(np.median(ann)) if ann.size else 0.0,
        p05=float(np.percentile(ann, 5)) if ann.size else 0.0,
        minimum=float(np.min(ann)) if ann.size else 0.0,
        frac_pos=float(np.mean(ann > 0.0)) if ann.size else 0.0,
        frac_gt_half=float(np.mean(ann > 0.5)) if ann.size else 0.0,
        naive_full_sharpe=naive,
        haircut=naive - (float(np.mean(ann)) if ann.size else 0.0),
    )
    return dist, summary


def run_cpcv(
    family_series: dict[str, np.ndarray],
    selected_label: str,
    *,
    n_groups: int = 8,
    k_test: int = 2,
    embargo: int = 1,
    seed: int = 0,
    max_paths: int | None = None,
) -> CPCVReport:
    """End-to-end CPCV for the xsec momentum family: path distribution (selected) + PBO (family).

    ``family_series`` — {label: full-sample weekly net-return series} for the WHOLE lookback family
    (aligned, same length). ``selected_label`` — the config whose OOS path distribution we report (the
    deployable pick, e.g. the L=14 fitted peak). ``n_groups`` / ``k_test`` give C(n_groups, k_test)
    paths; ``embargo`` is in WEEKS (the rebalance horizon). Both products key on the SAME split family.
    """
    if selected_label not in family_series:
        raise KeyError(f"selected_label {selected_label!r} not in family {tuple(family_series)}")
    lengths = {len(v) for v in family_series.values()}
    if len(lengths) != 1:
        raise ValueError(f"family series must be aligned to one length; got {sorted(lengths)}")
    n_weeks = lengths.pop()

    dist, summary = path_distribution(
        family_series[selected_label],
        n_groups=n_groups,
        k_test=k_test,
        embargo=embargo,
        seed=seed,
        max_paths=max_paths,
    )
    pbo = compute_pbo(family_series, n_groups=n_groups, k_test=k_test, embargo=embargo)
    return CPCVReport(
        n_groups=n_groups,
        k_test=k_test,
        embargo_weeks=embargo,
        n_weeks=n_weeks,
        n_paths=summary.n_paths,
        selected_label=selected_label,
        family_labels=tuple(family_series.keys()),
        path_summary=summary,
        pbo=pbo,
        distribution=dist,
    )
