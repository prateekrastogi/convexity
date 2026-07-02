"""Point-in-time daily panels: date x instrument DAILY CLOSE + date x instrument FUNDING RATE.

The #1 correctness requirement for any cross-sectional backtest is a POINT-IN-TIME UNIVERSE. A coin
that listed in 2025 must NOT appear in a 2022 cross-section (it did not exist — ranking it is
lookahead), and a coin that delisted must be closed at its last live bar (holding into a delist gap
fabricates a return). We enforce both here, at the panel layer, so every downstream module (signals,
portfolio, accounting) sees only legitimately-tradeable cells.

Construction (pure pandas/numpy; REUSES ``convexity.panel.data.load_bars`` / ``load_funding``):

  * DAILY CLOSE PANEL — resample each symbol's 1-min lake to a DAILY close (the last 1-min close of
    each UTC day). Row index = calendar days (UTC), columns = symbols. A cell is NaN before the
    symbol's first valid day and after its last valid day — that NaN band IS the point-in-time mask.
  * DAILY FUNDING PANEL — the funding lake is stamped every 8h (three stamps/day). We aggregate to a
    DAILY funding rate = the SUM of that day's 8h stamps (the total funding a position accrues over
    the calendar day). Aligned to the same day index; NaN outside the symbol's live span.
  * LIVE MASK — ``listed[date, sym]`` is True iff ``date`` is within [first_valid_day, last_valid_day]
    for ``sym`` (close-inclusive; the last live day can still hold, the day AFTER cannot).

No-lookahead: the daily close at day D is the 1-min close at ~23:59 UTC of day D, i.e. information
known only AT day D's close. A signal computed from closes up to and including day D is knowable at
D's close; the rebalance that consumes it is timestamped at that same close (see ``signals`` /
``engine``). The funding for day D is the funding that ACCRUED during day D (its three 8h stamps),
also known by D's close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexity.panel.data import load_bars, load_funding

MS_PER_DAY: int = 86_400_000


@dataclass(frozen=True)
class Panel:
    """The point-in-time daily panels for a crypto universe.

    ``dates`` — ascending UTC calendar days (``pd.DatetimeIndex``, tz-naive at 00:00 UTC), one row per
        day the union of symbols was live for at least one member.
    ``symbols`` — column order (tuple), stable across all three arrays.
    ``close`` — (n_days, n_syms) daily close; NaN outside a symbol's live span.
    ``funding`` — (n_days, n_syms) daily funding rate (SUM of the day's 8h stamps); NaN outside span,
        0.0 for a live day with no recorded stamp (a real gap in the funding lake is a 0-accrual day).
    ``listed`` — (n_days, n_syms) bool point-in-time mask: True iff the symbol was live on that day.
    ``first_day`` / ``last_day`` — per-symbol integer row index of first/last live day (for reporting).

    A cell is TRADEABLE on day D iff ``listed[D, sym]`` AND the signal has enough trailing history
    (checked in ``signals``); ``close``/``funding`` are NaN exactly where ``listed`` is False.
    """

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    close: np.ndarray
    funding: np.ndarray
    listed: np.ndarray
    first_day: dict[str, int]
    last_day: dict[str, int]

    @property
    def n_days(self) -> int:
        return int(self.close.shape[0])

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)


def _daily_close(bars_ts_ms: np.ndarray, close: np.ndarray) -> pd.Series:
    """Last 1-min close of each UTC calendar day -> a day-indexed close series (no lookahead).

    ``bars_ts_ms`` is the bar CLOSE epoch-ms (``convexity.panel.data`` sets ts = open_time + 60_000), so the
    day bucket uses ``(ts - 1) // MS_PER_DAY`` — the close stamp of 00:00:00.000 the next day belongs
    to the day that just ENDED, not the day that is starting. The last close in each bucket is that
    day's daily close.
    """
    day = (bars_ts_ms.astype(np.int64) - 1) // MS_PER_DAY
    s = pd.Series(close, index=day)
    last = s.groupby(level=0).last()
    idx = pd.to_datetime(last.index.to_numpy() * MS_PER_DAY, unit="ms", utc=True).tz_convert(None)
    return pd.Series(last.to_numpy(float), index=idx)


def _daily_funding(fund_ts_ms: np.ndarray, rate: np.ndarray) -> pd.Series:
    """Sum each UTC calendar day's 8h funding stamps -> a day-indexed daily funding series.

    Funding stamps are wall-clock (``calc_time`` in the lake); the funding that accrues DURING day D is
    the sum of D's stamps, knowable by D's close. Empty -> empty series.
    """
    if fund_ts_ms.size == 0:
        return pd.Series(dtype=float)
    day = fund_ts_ms.astype(np.int64) // MS_PER_DAY
    s = pd.Series(rate, index=day)
    daily = s.groupby(level=0).sum()
    idx = pd.to_datetime(daily.index.to_numpy() * MS_PER_DAY, unit="ms", utc=True).tz_convert(None)
    return pd.Series(daily.to_numpy(float), index=idx)


def build_panel(
    symbols: tuple[str, ...],
    *,
    lake_dir: str = "data/stitched",
    funding_dir: str = "data/funding",
) -> Panel:
    """Build the point-in-time daily close + funding panels for ``symbols``.

    Each symbol is loaded independently, resampled to daily, then outer-joined on the UTC day index so
    a coin that listed late simply has NaN rows before its first day. The ``listed`` mask is derived
    from each symbol's own [first_valid_day, last_valid_day] span — NEVER from the union — so a coin is
    marked live only on days it actually traded (no survivorship backfill, no delist hold-through).
    """
    close_cols: dict[str, pd.Series] = {}
    fund_cols: dict[str, pd.Series] = {}
    kept: list[str] = []
    for sym in symbols:
        bars = load_bars(sym, lake_dir=lake_dir)
        if len(bars) == 0:
            continue
        dc = _daily_close(bars.ts, bars.c)
        if dc.empty:
            continue
        fsched = load_funding(sym, funding_dir=funding_dir, bars=bars)
        df_ = _daily_funding(fsched.ts, fsched.rate)
        close_cols[sym] = dc
        fund_cols[sym] = df_
        kept.append(sym)

    if not kept:
        empty_idx = pd.DatetimeIndex([])
        return Panel(empty_idx, (), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0), bool), {}, {})

    close_df = pd.DataFrame(close_cols).sort_index()
    all_days = close_df.index
    # funding aligned to the SAME day index; a live day with no stamp -> 0 accrual (fill later).
    fund_df = pd.DataFrame(index=all_days)
    for sym in kept:
        fund_df[sym] = fund_cols[sym].reindex(all_days)
    close_df = close_df[kept]
    fund_df = fund_df[kept]

    close = close_df.to_numpy(float)
    funding = fund_df.to_numpy(float)

    # point-in-time live mask from each symbol's OWN valid span (no union backfill).
    listed = ~np.isnan(close)
    first_day: dict[str, int] = {}
    last_day: dict[str, int] = {}
    for j, sym in enumerate(kept):
        live = np.flatnonzero(listed[:, j])
        first_day[sym] = int(live[0])
        last_day[sym] = int(live[-1])

    # funding is 0 on a live day with no recorded stamp; NaN off-span (where not listed).
    funding = np.where(listed, np.nan_to_num(funding, nan=0.0), np.nan)

    return Panel(
        dates=all_days,
        symbols=tuple(kept),
        close=close,
        funding=funding,
        listed=listed,
        first_day=first_day,
        last_day=last_day,
    )


def build_class_panel(
    sleeve: str,
    *,
    lake_dir: str = "data/stitched",
    funding_dir: str = "data/funding",
    zero_funding: bool = False,
) -> Panel:
    """Point-in-time daily panel for ONE asset-class sleeve (crypto / equity / commodity).

    Thin sleeve-aware wrapper over ``build_panel`` (REUSES its resampler + point-in-time mask):

      1. resolve the sleeve's symbol list from UNIVERSE.md (``convexity.panel.sleeve.sleeve_symbols``),
      2. build the standard daily close + listed panels,
      3. when ``zero_funding`` (the MOMENTUM-ONLY, underlying case for equities/commodities — and crypto
         run momentum-only), overwrite the funding array with 0.0 on every LIVE cell (NaN off-span) so
         ``settle_week``'s signed funding cash flow is EXACTLY 0. We assert the post-condition so a stray
         non-zero funding cell can never silently leak carry into a momentum-only book.

    For equities the daily close is the last 1-min close of each UTC day = the RTH close (~20:00 UTC);
    a weekly return spanning an overnight/weekend gap therefore uses the ACTUAL gapped closes (Fri RTH
    close -> next-Fri RTH close), never a fabricated fill. Commodities are session-aware the same way
    (last live 1-min close of the UTC day). Both inherit the point-in-time listing mask unchanged.
    """
    from convexity.panel.sleeve import sleeve_symbols

    syms = sleeve_symbols(sleeve)  # type: ignore[arg-type]
    panel = build_panel(syms, lake_dir=lake_dir, funding_dir=funding_dir)
    if not zero_funding:
        return panel
    # momentum-only: force funding to 0 on live cells, NaN off-span; assert it stuck.
    funding = np.where(panel.listed, 0.0, np.nan)
    live = panel.listed
    assert np.all(funding[live] == 0.0), "momentum-only panel must have zero funding on live cells"
    assert np.all(np.isnan(funding[~live])), "off-span funding cells must be NaN"
    return Panel(
        dates=panel.dates,
        symbols=panel.symbols,
        close=panel.close,
        funding=funding,
        listed=panel.listed,
        first_day=panel.first_day,
        last_day=panel.last_day,
    )


def _daily_dollar_volume(bars_ts_ms: np.ndarray, close: np.ndarray, vol: np.ndarray) -> pd.Series:
    """Sum each UTC calendar day's 1-min notional (close*volume) -> a day-indexed dollar-volume series.

    Same day-bucketing as ``_daily_close`` (the bar-CLOSE stamp of 00:00 the next day belongs to the
    day that just ended). Volume is the base-asset volume from the lake; close*volume is the per-minute
    quote-notional traded, summed to a daily dollar-volume — a point-in-time liquidity proxy known at
    that day's close (no lookahead).
    """
    day = (bars_ts_ms.astype(np.int64) - 1) // MS_PER_DAY
    notional = close.astype(float) * vol.astype(float)
    s = pd.Series(notional, index=day)
    daily = s.groupby(level=0).sum()
    idx = pd.to_datetime(daily.index.to_numpy() * MS_PER_DAY, unit="ms", utc=True).tz_convert(None)
    return pd.Series(daily.to_numpy(float), index=idx)


def build_dollar_volume(
    panel: Panel,
    *,
    lake_dir: str = "data/stitched",
) -> np.ndarray:
    """Daily dollar-volume array aligned to ``panel`` (same dates x symbols order); NaN off-span.

    Re-reads each symbol's 1-min lake ONCE, resamples close*volume to a daily notional, and reindexes
    onto the panel's day index and symbol column order. A cell is NaN wherever the symbol is not listed
    (so the liquidity rank sees only live names) and NaN where the day has no bar. This is a SEPARATE
    array (not stored on the frozen ``Panel``) so the panel API and its tests are unchanged; the
    universe-size sweep uses it to pick the top-N most-liquid live names as-of each rebalance.

    No-lookahead: the value at day D is that day's traded notional, known at D's close — the same
    information altitude as the close/funding panels.
    """
    cols: dict[str, pd.Series] = {}
    for sym in panel.symbols:
        bars = load_bars(sym, lake_dir=lake_dir)
        if len(bars) == 0:
            continue
        cols[sym] = _daily_dollar_volume(bars.ts, bars.c, bars.v)
    dv = pd.DataFrame(index=panel.dates)
    for sym in panel.symbols:
        dv[sym] = cols[sym].reindex(panel.dates) if sym in cols else np.nan
    dv = dv[list(panel.symbols)]
    arr = dv.to_numpy(float)
    # mask to the point-in-time listing span (a name is only rankable on days it traded).
    return np.where(panel.listed, arr, np.nan)


def trailing_dollar_volume(dvol: np.ndarray, listed: np.ndarray, d: int, *, window: int = 30) -> np.ndarray:
    """Per-symbol mean daily dollar-volume over ``[d - window + 1, d]`` (no lookahead), NaN if not live at d.

    A rolling ~30d average traded notional as-of rebalance day ``d``. Only names LISTED at ``d`` are
    eligible (NaN otherwise); the mean ignores NaN days within the window so a name with a partial
    window still gets a (noisier) liquidity estimate. Used to rank the live universe by liquidity and
    keep the top-N as-of each rebalance.
    """
    n_sym = dvol.shape[1]
    out = np.full(n_sym, np.nan)
    start = max(0, d - window + 1)
    block = dvol[start : d + 1]  # (<=window, n_sym)
    live_at_d = listed[d]
    if block.shape[0] == 0:
        return out
    finite = np.isfinite(block)
    has_any = finite.any(axis=0)  # columns with >=1 finite day (avoid all-NaN-slice warning)
    mean_dv = np.full(n_sym, np.nan)
    if has_any.any():
        with np.errstate(invalid="ignore"):
            mean_dv[has_any] = np.nanmean(
                np.where(finite[:, has_any], block[:, has_any], np.nan), axis=0
            )
    out = np.where(live_at_d & np.isfinite(mean_dv), mean_dv, np.nan)
    return out


def universe_size_by_year(panel: Panel) -> dict[int, int]:
    """Peak point-in-time universe size (count of live symbols) observed in each calendar year.

    Reports the MAX live-symbol count per year — a compact way to see the universe grow from a handful
    of majors in the early years to the full ~50 later, which is the tell that the point-in-time mask
    is real (a survivorship-biased panel would show ~50 from day one).
    """
    if panel.n_days == 0:
        return {}
    live_per_day = panel.listed.sum(axis=1)
    years = panel.dates.year.to_numpy()
    out: dict[int, int] = {}
    for y in np.unique(years):
        out[int(y)] = int(live_per_day[years == y].max())
    return out
