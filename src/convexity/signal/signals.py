"""Cross-sectional signals: XS momentum (lookback x skip), funding carry, combined — all no-lookahead.

Every signal at rebalance day D is computed STRICTLY from panel data at rows <= D (information known
at D's close). The output is a cross-sectional RANK in [0, 1] per eligible symbol (higher rank ->
stronger LONG conviction), NaN for symbols not eligible at D. The portfolio layer turns ranks into
dollar-neutral weights; nothing here touches future rows.

Signals:

  * XS MOMENTUM — trailing simple return over lookback ``L`` days ending at day ``D - skip``:
        mom = close[D - skip] / close[D - skip - L] - 1
    The optional ``skip`` (0/3/7 d) drops the most-recent window to dodge short-term reversal (the
    well-documented 1-week mean-reversion in crypto). HIGHER trailing return -> HIGHER rank -> LONG.
    Eligibility needs ``L + skip`` days of trailing history AND both endpoint closes present.

  * FUNDING CARRY — trailing MEAN daily funding rate over the past ``carry_lookback`` days (default 7),
    ending at day D. Sign convention (COST §5.5): a perp with HIGH positive funding pays its longs, so
    we SHORT it; a perp with LOW/negative funding pays its shorts, so we LONG it. Therefore the carry
    SIGNAL that we rank for LONG conviction is the NEGATED mean funding: ``carry_signal = -mean_funding``
    -> low/negative funding gets a HIGH rank (long), high funding gets a LOW rank (short). This lines
    the carry rank up with the momentum rank (both: high rank -> long), so the combined average is
    coherent.

  * COMBINED — equal-weight average of the momentum rank and the carry rank, re-ranked cross-sectionally.

  * ENSEMBLE (the DEPLOYABLE core) — equal-weight average of the momentum ranks across the
    validated lookback BAND {12, 14, 16}, re-ranked. Carry is NOT blended into the deployable
    rank (w_carry = 0; SIGNAL.md §6.2) — it stays a separately-monitored satellite.

RANKING — a cross-sectional percentile rank over the ELIGIBLE symbols at D (ties averaged), scaled to
[0, 1]. A single eligible symbol ranks 0.5 (neutral). Symbols ineligible at D are NaN and excluded.
"""

from __future__ import annotations

import numpy as np

from convexity.panel.panel import Panel


def _pct_rank(values: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in [0, 1] over the non-NaN entries (ties averaged).

    NaN in -> NaN out. With ``m`` finite values the ranks are (average-rank - 0.5) / m so a single
    value maps to 0.5 (neutral) and the extremes sit symmetrically inside (0, 1).
    """
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    m = int(finite.sum())
    if m == 0:
        return out
    if m == 1:
        out[finite] = 0.5
        return out
    v = values[finite]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(m)
    # average-rank for ties
    sv = v[order]
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sv[j + 1] == sv[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    out[finite] = (ranks - 0.5) / m
    return out


def momentum_signal(panel: Panel, d: int, *, lookback: int, skip: int) -> np.ndarray:
    """Cross-sectional momentum RANK at rebalance day ``d`` (higher rank -> long). No lookahead.

    Return over ``[d - skip - lookback, d - skip]``. A symbol is eligible iff both endpoint closes
    are present AND ``listed`` is True at both ENDPOINTS — endpoint-only by design: the 2-point
    return uses only the endpoint closes, so interior non-listed days inside the window are
    tolerated (they occur on the real lake — shared data gaps — so the old "live span is contiguous
    per symbol" claim was false; audit P-1). All indices are <= d.
    """
    end = d - skip
    start = d - skip - lookback
    n_sym = panel.n_symbols
    raw = np.full(n_sym, np.nan)
    if start < 0:
        return raw  # not enough trailing history for anyone yet
    c_end = panel.close[end]
    c_start = panel.close[start]
    live = panel.listed[end] & panel.listed[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(live & (c_start > 0), c_end / c_start - 1.0, np.nan)
    raw[:] = r
    return _pct_rank(raw)


def carry_signal(panel: Panel, d: int, *, lookback: int = 7) -> np.ndarray:
    """Funding-carry RANK at rebalance day ``d`` (higher rank -> long). No lookahead.

    Trailing MEAN daily funding over ``[d - lookback + 1, d]`` (inclusive of d's accrued funding, which
    is known at d's close), then NEGATED so LOW/negative funding -> HIGH rank (long, collects the
    credit) and HIGH positive funding -> LOW rank (short, collects the credit). Eligible iff listed at
    d with a full trailing window of funding observations.
    """
    n_sym = panel.n_symbols
    raw = np.full(n_sym, np.nan)
    start = d - lookback + 1
    if start < 0:
        return raw
    win = panel.funding[start : d + 1]  # (lookback, n_sym)
    live_all = panel.listed[start : d + 1].all(axis=0)
    mean_funding = np.full(n_sym, np.nan)
    if live_all.any():
        # nanmean only over the columns live across the whole window (avoids all-NaN empty-slice warn).
        mean_funding[live_all] = np.nanmean(win[:, live_all], axis=0)
    raw[:] = -mean_funding  # negate: low funding -> long
    return _pct_rank(raw)


def combined_signal(
    panel: Panel, d: int, *, lookback: int, skip: int, carry_lookback: int = 7
) -> np.ndarray:
    """Equal-weight rank-average of momentum + carry, re-ranked cross-sectionally. No lookahead.

    Only symbols eligible for BOTH sub-signals get a combined rank (an intersection eligibility — a
    coin must have enough momentum history AND a full carry window). The two ranks are averaged then
    re-ranked so the combined output is itself a clean [0, 1] cross-sectional percentile.
    """
    mom = momentum_signal(panel, d, lookback=lookback, skip=skip)
    car = carry_signal(panel, d, lookback=carry_lookback)
    both = np.isfinite(mom) & np.isfinite(car)
    avg = np.full(panel.n_symbols, np.nan)
    avg[both] = 0.5 * (mom[both] + car[both])
    return _pct_rank(avg)


def ensemble_momentum_signal(
    panel: Panel,
    d: int,
    *,
    lookbacks: tuple[int, ...] = (12, 14, 16),
    skip: int = 0,
    weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Rank-average of ``momentum_signal`` across the lookback BAND, re-ranked. No lookahead.

    THE DEPLOYABLE CORE SIGNAL (SIGNAL.md §4.3/§6.2): the signal PORTFOLIO consumes for the
    live/paper book is this ensemble with ``w_carry = 0`` — funding is never read here (Invariant
    CARRY-ZERO); ``carry_signal`` stays a separately-monitored satellite and ``combined_signal``
    stays research/diagnostic-only. The default band ``{12, 14, 16}`` (trading days), ``skip = 0``
    and equal weights ``v_L = 1/3`` are RED constants (SIGNAL.md §7): fixed BEFORE any realized
    Sharpe was observed (Invariant NO-FIT — never a regressed/IC-fitted combination), and never
    per-symbol. Deploying a single fitted lookback instead of the band is Invariant BAND's bug
    (PBO = 0.31-0.47 across CPCV partitions says the argmax-L is partition-unstable, SIGNAL.md
    §4.1-4.2); a single-member call is still well-defined for research/diagnostics and reduces
    exactly to that member's ``momentum_signal`` (SIGNAL.md §9 step 3).

    Mechanics — SIGNAL.md §4.4's general rank-ensemble operator (``combined_signal``'s pattern
    generalized from m=2 sub-signals to m=len(lookbacks) lookbacks):

      1. ``rank^(s) = momentum_signal(panel, d, lookback=L_s, skip=skip)`` per band member;
      2. ``avg_i = sum_s v_s * rank^(s)_i`` only where ``i`` is finite in EVERY component —
         intersection eligibility (SIGNAL.md §8 case 5): a symbol not yet eligible for the longest
         lookback is excluded entirely, never scored on a partial band;
      3. ``_pct_rank(avg)`` — the final re-rank (an average of uniform ranks is not uniform,
         SIGNAL.md §8 case 7), restoring the flat [0, 1] percentile contract.

    Causality (Invariant NLA) and point-in-time ranking (Invariant RANK) are inherited from
    ``momentum_signal``/``_pct_rank``; nothing here reads a panel row ``> d``. ``weights`` must
    match ``lookbacks`` in length and be finite, non-negative, with a positive sum (a negative
    weight is nonsensical for a rank-average — SIGNAL.md §8 case 9); they are normalized to sum
    to 1 (the re-rank is invariant to positive rescaling, so normalization is cosmetic but keeps
    the convex-combination contract of §4.4 explicit).
    """
    m = len(lookbacks)
    if m == 0:
        raise ValueError("lookbacks must be a non-empty tuple (SIGNAL.md §4.3)")
    v = np.full(m, 1.0 / m) if weights is None else np.asarray(weights, dtype=float)
    if v.shape != (m,):
        raise ValueError(f"weights length {v.size} != lookbacks length {m}")
    if not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0.0:
        raise ValueError("weights must be finite, non-negative, with a positive sum (§8 case 9)")
    v = v / v.sum()
    ranks = np.vstack([momentum_signal(panel, d, lookback=lb, skip=skip) for lb in lookbacks])
    band_eligible = np.isfinite(ranks).all(axis=0)  # intersection across the WHOLE band
    avg = np.full(panel.n_symbols, np.nan)
    avg[band_eligible] = v @ ranks[:, band_eligible]
    return _pct_rank(avg)
