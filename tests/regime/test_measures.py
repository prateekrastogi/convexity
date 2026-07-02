"""Cross-sectional dispersion D_t and BTC realized vol V_t tests (REGIME.md §2.2/§2.3, §8.1).

Pins the RMS-weekly aggregation of D_t (§2.2, NOT arithmetic mean), the eligible-universe scope, the
§8.1 breadth floor (D_tau excluded, not zero-filled, below n_min), and the annualized V_t (§2.3).
Also the panel adapter Invariant N (strict re-key, not recompute) and the MP-guard (no unshrunk
covariance in the v1 detector — §2.4 / Invariant MP).
"""

import ast
import pathlib

import numpy as np
import pandas as pd

from convexity.regime.overlay import _daily_returns
from convexity.panel.panel import Panel
from convexity.regime.measures.absorption import daily_dispersion, weekly_dispersion
from convexity.regime.measures.vol_regime import weekly_btc_vol
from convexity.regime.panel.build import build_regime_inputs


def _panel(n_days=10, n_sym=3):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, size=(n_days, n_sym)), axis=0)
    funding = np.zeros((n_days, n_sym))
    listed = np.ones((n_days, n_sym), dtype=bool)
    return Panel(
        dates=dates, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        close=close, funding=funding, listed=listed,
        first_day={s: 0 for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")},
        last_day={s: n_days - 1 for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")},
    )


# --------------------------------------------------------------------------- §2.2 D_t

def test_daily_dispersion_is_cross_sectional_std():
    """§2.2: D_tau = std over eligible names of that day's daily returns."""
    dr = np.array([[np.nan, np.nan, np.nan], [0.01, 0.03, -0.02]])
    listed = np.ones((2, 3), dtype=bool)
    d_tau, n = daily_dispersion(dr, listed, 1, n_min=1)
    assert n == 3
    assert d_tau == np.std([0.01, 0.03, -0.02])


def test_weekly_dispersion_is_rms_not_mean():
    """§2.2: D_t is the RMS (not arithmetic mean) of daily cross-sectional std over the week."""
    dr = np.array([[0.01, 0.03], [0.05, -0.05]])  # 2 days, 2 names
    listed = np.ones((2, 2), dtype=bool)
    d0 = np.std([0.01, 0.03])
    d1 = np.std([0.05, -0.05])
    d_t, used = weekly_dispersion(dr, listed, np.array([0, 1]), n_min=1)
    assert used == 2
    assert d_t == np.sqrt(np.mean([d0**2, d1**2]))  # RMS
    assert d_t != np.mean([d0, d1])                  # NOT the arithmetic mean


def test_breadth_floor_excludes_not_zero_fills():
    """§8.1: a day with eligible breadth < n_min is EXCLUDED from the RMS (NaN), not fed as 0."""
    dr = np.array([[0.01, 0.03, -0.02], [0.10, np.nan, np.nan]])  # day 1 has only 1 eligible name
    listed = np.array([[True, True, True], [True, True, True]])
    # day 1: only 1 finite return but n_min=2 -> excluded
    d_tau1, n1 = daily_dispersion(dr, listed, 1, n_min=2)
    assert np.isnan(d_tau1) and n1 == 1
    # weekly D_t uses only day 0 (day 1 excluded), so equals day-0 dispersion exactly
    d_t, used = weekly_dispersion(dr, listed, np.array([0, 1]), n_min=2)
    assert used == 1
    assert d_t == np.std([0.01, 0.03, -0.02])


def test_all_days_below_floor_gives_nan_week():
    """§8.1/§0 P6: when NO day clears the breadth floor, D_t is NaN -> caller reports WATCH."""
    dr = np.array([[0.01, np.nan, np.nan]])
    listed = np.ones((1, 3), dtype=bool)
    d_t, used = weekly_dispersion(dr, listed, np.array([0]), n_min=2)
    assert np.isnan(d_t) and used == 0


def test_majority_of_days_below_floor_gives_nan_week_even_when_some_clear():
    """§8.1 MAJORITY rule (the P6 escalation, NOT just all-days-below): when a MAJORITY of the week's
    days are below n_min — but a MINORITY still clear it — the WHOLE week's D_t is NaN so the caller
    reports WATCH/insufficient_breadth and excludes D_t from the CUSUM. Computing D_t from the surviving
    minority (the pre-fix behaviour) is the more-permissive HEALTHY-eligible outcome P6 forbids.

    5-day week, n_min=3: days 0,1 have 3 eligible names (clear); days 2,3,4 have <3 (below). 3 of 5
    below -> majority -> NaN week, even though days 0 and 1 individually produced finite dispersion."""
    dr = np.array([
        [0.01, 0.03, -0.02],       # day 0: 3 eligible -> clears
        [0.02, -0.01, 0.04],       # day 1: 3 eligible -> clears
        [0.05, np.nan, np.nan],    # day 2: 1 eligible -> below
        [-0.03, np.nan, np.nan],   # day 3: 1 eligible -> below
        [0.02, 0.01, np.nan],      # day 4: 2 eligible -> below
    ])
    listed = np.ones((5, 3), dtype=bool)
    d_t, used = weekly_dispersion(dr, listed, np.arange(5), n_min=3)
    assert np.isnan(d_t)   # majority (3/5) below floor -> whole week undefined -> WATCH
    assert used == 2       # the two clearing days are still counted for telemetry


def test_minority_below_floor_keeps_week_and_excludes_only_those_days():
    """§8.1 per-day clause (the non-majority case): when a MINORITY of days are below the floor, the
    week is KEPT (HEALTHY-eligible) and only the sub-floor days are excluded from the RMS — never
    zero-filled. Boundary: exactly half below is NOT a majority, so the week survives.

    4-day week, n_min=3: days 0,1 clear (3 eligible), days 2,3 below (1 eligible). 2 of 4 below is
    NOT strictly-more-than-half, so the week stands on days 0,1's RMS."""
    dr = np.array([
        [0.01, 0.03, -0.02],       # clears
        [0.02, -0.01, 0.04],       # clears
        [0.05, np.nan, np.nan],    # below
        [-0.03, np.nan, np.nan],   # below
    ])
    listed = np.ones((4, 3), dtype=bool)
    d0 = np.std([0.01, 0.03, -0.02])
    d1 = np.std([0.02, -0.01, 0.04])
    d_t, used = weekly_dispersion(dr, listed, np.arange(4), n_min=3)
    assert used == 2
    assert d_t == np.sqrt(np.mean([d0**2, d1**2]))   # RMS of the two surviving days only


# --------------------------------------------------------------------------- §2.3 V_t

def test_weekly_btc_vol_annualized():
    """§2.3: V_t = std of the week's daily BTC returns * sqrt(365.25)."""
    dr = np.zeros((5, 2))
    dr[:, 0] = [0.01, -0.01, 0.02, -0.02, 0.0]
    v = weekly_btc_vol(dr, 0, np.array([0, 1, 2, 3, 4]))
    assert v == float(np.std(dr[:, 0], ddof=1) * np.sqrt(365.25))


def test_weekly_btc_vol_nan_on_insufficient_data():
    """§2.3/§0 P6: fewer than two finite BTC returns -> NaN (undefined -> conservative WATCH)."""
    dr = np.array([[np.nan], [0.01]])
    assert np.isnan(weekly_btc_vol(dr, 0, np.array([0, 1])))


# --------------------------------------------------------------------------- §1 Invariant N

def test_panel_adapter_is_strict_rekey_not_recompute():
    """Invariant N: build_regime_inputs re-keys Panel fields; daily_returns uses the SHARED
    overlay._daily_returns (one convention), listed/symbols are passed through by reference."""
    p = _panel()
    inp = build_regime_inputs(p, market_symbol="BTCUSDT")
    np.testing.assert_array_equal(inp.daily_returns, _daily_returns(p))  # same convention, byte-identical
    assert inp.listed is p.listed        # passed through, not recomputed
    assert inp.symbols == p.symbols
    assert inp.btc_col == 0


def test_panel_adapter_raises_when_market_symbol_absent():
    """§0 P6: a missing market column is a hard error, never a silent HEALTHY default."""
    p = _panel()
    import pytest

    with pytest.raises(ValueError):
        build_regime_inputs(p, market_symbol="NOTALISTEDCOIN")


# --------------------------------------------------------------------------- §2.4 Invariant MP

def test_v1_detector_uses_no_unshrunk_covariance():
    """Invariant MP (§2.4): the v1 live detector (measures/absorption.py, vol_regime.py) uses ONLY
    univariate statistics — NO np.cov/np.corrcoef feeding a Mahalanobis/absorption ratio. A raw
    sample-covariance path in these modules is the Marchenko-Pastur bug this invariant blocks."""
    from convexity.regime.measures import absorption, vol_regime

    for mod in (absorption, vol_regime):
        src = pathlib.Path(mod.__file__).read_text()
        tree = ast.parse(src)
        calls = {
            f"{getattr(n.func.value, 'id', '')}.{n.func.attr}"
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "np.cov" not in calls, f"{mod.__name__} uses unshrunk np.cov (Invariant MP)"
        assert "np.corrcoef" not in calls, f"{mod.__name__} uses unshrunk np.corrcoef (Invariant MP)"
