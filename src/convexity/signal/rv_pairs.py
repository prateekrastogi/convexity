"""Commodity RV pair mean-reversion signal (research/entry_exit/study RV_PREREG.md, frozen
2026-07-05) — the pre-named deployable, as pure functions over the daily Panel.

Construction (EXACTLY the frozen prereg; any change re-opens the study, never lands silently):
per pre-named pair, spread = log-price ratio at WEEKLY Friday closes; signal = the BAND-ENSEMBLE
mean of rolling z over {26, 52, 104} weeks; continuous sizing ``-clip(z, ±2)/2``; a trailing AR(1)
HALF-LIFE GATE [1, 26] wk (a spread that doesn't mean-revert within ~2 quarters is not an RV trade
— it de-activates structurally broken pairs, PL-PA style); PAIR RISK PARITY on trailing 26-wk pair
P&L vol (bounded [0.25, 4]); unit book gross; then a 10% annualized VOL TARGET on the trailing
26-wk book net (cap [0.5, 2]).

Causality: the weight at Friday row ``d`` reads closes/spreads/vols through ``d`` ONLY and is
scored on the d -> d+1wk return — verified by the study's lookahead canary (the off-by-one flips
the book to Sharpe -3.8) and re-pinned here by tests/signal/test_rv_pairs.py. Missing weekly
closes are forward-filled per symbol (the backtest's flat-price convention for a missing week).

Everything is recomputed from the panel every session (OPS pure-recompute doctrine): the pair-RP
and vol-target trailing windows come from the SAME construction walked over the full weekly
history, so a session needs no journal state to size itself.

Known reproducibility caveat (accepted, documented): the pair-RP normalization divides by the
median of the inverse-vol scale over ALL weeks <= d (the study's own convention). That is causal
— nothing past d is read — but it means week-t weights in the recomputed history shift slightly
as d advances, so the trailing series is not bit-frozen across sessions. The effect is bounded by
the [0.25, 4] clip and vanishes in the [0.5, 2] vol-target cap; the CAUSALITY of the final target
is what tests pin, not cross-session bit-equality of the history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from convexity.panel.panel import Panel

# ---- frozen prereg constants (RV_PREREG.md; the grid IS the multiplicity ledger) ----
PAIRS: tuple[tuple[str, str], ...] = (
    ("XAUUSDT", "XAGUSDT"),    # GC-SI
    ("XAUUSDT", "XPTUSDT"),    # GC-PL
    ("XAGUSDT", "XPTUSDT"),    # SI-PL
    ("XPTUSDT", "XPDUSDT"),    # PL-PA
    ("CLUSDT", "NATGASUSDT"),  # CL-NG
)
ZWINS: tuple[int, ...] = (26, 52, 104)
ZCLIP: float = 2.0
HL_WIN: int = 104
HL_LO: float = 1.0
HL_HI: float = 26.0
RP_WIN: int = 26
RP_CLIP: tuple[float, float] = (0.25, 4.0)
VT_TARGET: float = 0.10         # annualized book vol target
VT_WIN: int = 26
VT_CLIP: tuple[float, float] = (0.5, 2.0)
RT_BP: float = 12.0             # the prereg's primary cost, used ONLY inside the vol-target series
WK_PER_YR: float = 52.0


def friday_rows(dates: pd.DatetimeIndex, d: int) -> np.ndarray:
    """Panel row indices of Fridays up to and including row ``d`` (``d`` must be a Friday row)."""
    wd = dates.weekday.to_numpy()
    rows = np.flatnonzero(wd[: d + 1] == 4)
    if rows.size == 0 or rows[-1] != d:
        raise ValueError(f"rebalance row {d} ({dates[d].date()}) is not a Friday panel row")
    return rows


def weekly_log_close(panel: Panel, rows: np.ndarray) -> np.ndarray:
    """(n_weeks, n_syms) log Friday closes, per-symbol forward-filled over interior gaps.

    Forward-fill = the backtest's missing-week-is-flat convention (its return series used
    ``nan_to_num(wret)``); leading NaNs (pre-listing) are preserved so early weeks stay inactive.
    """
    close_w = pd.DataFrame(panel.close[rows]).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(close_w > 0, np.log(close_w), np.nan)


def roll_z(s: np.ndarray, window: int) -> np.ndarray:
    """Rolling z of ``s`` over ``window`` points through t inclusive (known at t's close)."""
    ser = pd.Series(s)
    mu = ser.rolling(window, min_periods=window).mean()
    sd = ser.rolling(window, min_periods=window).std()
    return ((ser - mu) / sd).to_numpy()


def half_life(s: np.ndarray, window: int = HL_WIN) -> np.ndarray:
    """Trailing AR(1) half-life of the spread: ds_t = a + phi*s_{t-1}; hl = -ln2/ln(1+phi).

    NaN where history is short (<80% finite), the fit is degenerate, or phi is outside (-1, 0)
    (no mean reversion) — the caller gates those weeks to zero.
    """
    n = len(s)
    hl = np.full(n, np.nan)
    for t in range(window, n):
        seg = s[t - window : t + 1]
        x, y = seg[:-1], np.diff(seg)
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() < int(0.8 * window):
            continue
        xv, yv = x[fin], y[fin]
        xd = xv - xv.mean()
        den = float((xd**2).sum())
        if den <= 0.0:
            continue
        phi = float((xd * (yv - yv.mean())).sum() / den)
        if -1.0 < phi < 0.0:
            hl[t] = -np.log(2.0) / np.log(1.0 + phi)
    return hl


def pair_weight_series(
    lp: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_weeks,) continuous pair position + its half-life gate mask for one pair.

    Position = -clip(band-ensemble z, ±2)/2 in [-1, 1]; the HL gate zeroes weeks whose trailing
    AR(1) half-life is outside [HL_LO, HL_HI]. NaN-safe: any week with an unknowable z is flat.
    """
    s = lp[:, idx_a] - lp[:, idx_b]
    with np.errstate(all="ignore"):
        zs = np.column_stack([roll_z(s, w) for w in ZWINS])
        z = np.nanmean(zs, axis=1)
    pw = -np.clip(z, -ZCLIP, ZCLIP) / ZCLIP
    hl = half_life(s)
    gate = np.isfinite(hl) & (hl >= HL_LO) & (hl <= HL_HI)
    return np.nan_to_num(pw) * gate, gate


@dataclass(frozen=True)
class RVBook:
    """The full weekly weight history (for RP/VT trailing windows) + the rebalance-row target."""

    w_target: np.ndarray            # (n_syms,) final target at the last Friday row (vol-targeted)
    w_unit: np.ndarray              # (n_weeks, n_syms) unit-gross book history (pre vol-target)
    vt_scale: float                 # the vol-target factor applied at the rebalance row
    friday_idx: np.ndarray          # panel row index of each weekly grid point
    diag: dict = field(default_factory=dict)


def book_weights(panel: Panel, d: int) -> RVBook:
    """The RV book's target weights at Friday row ``d`` (aligned to ``panel.symbols``).

    Walks the WHOLE weekly history so pair risk parity and the vol target come from the exact
    backtest construction (pure recompute, no journal state). Symbols outside the pre-named PAIRS
    always get weight 0; a pair with a leg missing from the panel is inactive and reported.
    """
    rows = friday_rows(panel.dates, d)
    n_w = rows.size
    n_sym = panel.n_symbols
    lp = weekly_log_close(panel, rows)
    sym_ix = {s: i for i, s in enumerate(panel.symbols)}

    per_pair: list[np.ndarray] = []
    diag: dict = {"pairs": {}, "inactive_pairs": []}
    for a, b in PAIRS:
        if a not in sym_ix or b not in sym_ix:
            diag["inactive_pairs"].append(f"{a}-{b} (leg missing from panel)")
            continue
        ia, ib = sym_ix[a], sym_ix[b]
        pw, gate = pair_weight_series(lp, ia, ib)
        # a leg must be LISTED at the week to hold it (proxy history keeps this true throughout).
        live = panel.listed[rows][:, ia] & panel.listed[rows][:, ib]
        pw = pw * live
        Wp = np.zeros((n_w, n_sym))
        Wp[:, ia] = 0.5 * pw
        Wp[:, ib] = -0.5 * pw
        per_pair.append(Wp)
        diag["pairs"][f"{a}-{b}"] = {
            "pw": round(float(pw[-1]), 4),
            "gate_open": bool(gate[-1]),
            "active_frac": round(float((np.abs(pw) > 0).mean()), 2),
        }

    if not per_pair:
        return RVBook(w_target=np.zeros(n_sym), w_unit=np.zeros((n_w, n_sym)),
                      vt_scale=1.0, friday_idx=rows, diag=diag)

    # weekly forward returns on the weekly grid (for RP/VT trailing vols; last row NaN, unused
    # because both estimators shift by one week).
    close_w = pd.DataFrame(panel.close[rows]).ffill().to_numpy()
    wr = np.full((n_w, n_sym), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr[:-1] = close_w[1:] / close_w[:-1] - 1.0

    # ---- pair risk parity (trailing RP_WIN pair-P&L vol, causal shift 1, bounded, summed) ----
    comb = np.zeros((n_w, n_sym))
    for Wp in per_pair:
        r = np.nansum(Wp * np.nan_to_num(wr), axis=1)
        tv = pd.Series(r).rolling(RP_WIN).std().shift(1).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(np.isfinite(tv) & (tv > 0), 1.0 / tv, 0.0)
        med = np.nanmedian(sc[sc > 0]) if (sc > 0).any() else 1.0
        sc = np.clip(sc / max(med, 1e-12), RP_CLIP[0], RP_CLIP[1])
        comb += Wp * sc[:, None]

    # ---- unit gross per week ----
    g = np.abs(comb).sum(axis=1, keepdims=True)
    w_unit = np.divide(comb, g, out=np.zeros_like(comb), where=g > 0)

    # ---- vol target on the trailing book net (the backtest's exact series: 12bp on gross
    # turnover), causal shift 1, capped ----
    gross_ret = np.nansum(w_unit * np.nan_to_num(wr), axis=1)
    dW = np.abs(np.diff(w_unit, axis=0, prepend=np.zeros((1, n_sym)))).sum(axis=1)
    net = gross_ret - (RT_BP / 1e4) * dW
    tv = pd.Series(net).rolling(VT_WIN).std().shift(1).to_numpy() * np.sqrt(WK_PER_YR)
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = np.clip(VT_TARGET / tv, VT_CLIP[0], VT_CLIP[1])
    vt_scale = float(vt[-1]) if np.isfinite(vt[-1]) else 1.0

    w_target = w_unit[-1] * vt_scale
    diag["gross_unit"] = round(float(np.abs(w_unit[-1]).sum()), 6)
    diag["vt_scale"] = round(vt_scale, 4)
    diag["n_weeks"] = int(n_w)
    return RVBook(w_target=w_target, w_unit=w_unit, vt_scale=vt_scale,
                  friday_idx=rows, diag=diag)
