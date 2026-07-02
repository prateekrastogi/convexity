"""Invariant CS-REJECT standing regression guard (COST.md Section 5.4.3).

Corwin-Schultz was tried as the impact proxy on 1-minute-aggregated daily OHLC and produced a
systematically, wrongly large estimate (~135bp round-trip, ~40%/yr drag vs the measured ~13bp).

Two layers of guard, both tested here:
  1. RE-RUN THE ESTIMATOR (the literal spec clause): a runnable CS estimator is fed a
     1-minute-aggregated daily OHLC panel whose intraday volatility swamps the (near-zero) true
     spread, and MUST reproduce the documented order-of-magnitude-too-large garbage (>= 10x 13bp).
  2. STRUCTURAL SCAN: no code-level Corwin-Schultz identifier/import may appear in the cost path.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from research.xsec.cost.validate.corwin_schultz_reject import (
    CS_REJECT_ORDER_OF_MAGNITUDE,
    CS_REJECTED_DRAG_PCT_PER_YEAR,
    CS_REJECTED_ROUND_TRIP_BP,
    MEASURED_ROUND_TRIP_BP,
    aggregate_minute_to_daily_ohlc,
    corwin_schultz_offenders,
    corwin_schultz_round_trip_bp,
    corwin_schultz_spread_bp,
)


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            raise RuntimeError("repo root with pyproject.toml not found")
        root = root.parent
    return root


def _minute_aggregated_daily_ohlc_panel(
    *, n_days: int = 120, mins_per_day: int = 1440, minute_sigma: float = 0.0008,
    true_half_spread: float = 0.0001, seed: int = 0,
):
    """A 1-minute OHLC panel with a TINY true spread but heavy intraday volatility.

    This is the exact input COST.md Section 5.4.3 describes: a daily high/low assembled from
    1-minute bars, where the daily range is dominated by realized intraday volatility (per-minute
    sigma ~8bp) rather than by the near-zero true quoted spread (~2bp). Corwin-Schultz's
    separation identity assumes a sampling frequency where volatility scales with sqrt(dt) while the
    spread does not; a daily range built from 1-minute bars violates that, so the estimator
    misattributes the volatility to spread and blows up. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    day_index: list[int] = []
    highs: list[float] = []
    lows: list[float] = []
    mid = 100.0
    for d in range(n_days):
        for _ in range(mins_per_day):
            mid *= 1.0 + minute_sigma * rng.standard_normal()
            bar_range = abs(minute_sigma * rng.standard_normal()) * mid
            highs.append(mid + 0.5 * bar_range + true_half_spread * mid)
            lows.append(mid - 0.5 * bar_range - true_half_spread * mid)
            day_index.append(d)
    return (
        np.array(day_index),
        np.array(highs, dtype=float),
        np.array(lows, dtype=float),
        2.0 * true_half_spread * 1e4,  # true quoted spread in bp (~2bp)
    )


# --------------------------------------------------------------------------------------------------
# LAYER 1 — the literal Invariant CS-REJECT clause: RE-RUN the estimator, REPRODUCE the garbage.
# --------------------------------------------------------------------------------------------------


def test_rerun_cs_on_minute_aggregated_daily_ohlc_reproduces_garbage():
    # The load-bearing "reproduce the result" guard (COST.md Section 5.4.3): running the runnable CS
    # estimator on a 1-minute-aggregated daily OHLC panel whose TRUE spread is ~2bp reproduces an
    # order-of-magnitude-too-large round-trip (>= 10x the 13bp anchor) — the documented ~135bp dead
    # end. If a future refactor ever "fixes" the estimator into a viable spread source, this fails.
    day_index, high, low, true_spread_bp = _minute_aggregated_daily_ohlc_panel()
    rt_bp = corwin_schultz_round_trip_bp(day_index, high, low)
    assert math.isfinite(rt_bp)
    # non-vacuous: the true quoted spread this panel was built with is tiny (~2bp), a real ~13bp-anchor
    # cost, NOT itself a garbage number — the blowup is the estimator's, not the input's.
    assert true_spread_bp < MEASURED_ROUND_TRIP_BP
    # the reproduced estimate is order-of-magnitude wrong, exactly why CS was rejected.
    assert rt_bp >= CS_REJECT_ORDER_OF_MAGNITUDE * MEASURED_ROUND_TRIP_BP
    # and it lands in the documented ~135bp ballpark (same order of magnitude), not merely "large".
    assert 100.0 <= rt_bp <= 300.0


def test_cs_spread_estimate_swamps_the_true_spread_by_an_order_of_magnitude():
    # The mechanism check: the CS spread estimate on the aggregated daily range is >= 10x the true
    # ~2bp quoted spread the panel was built with — volatility misattributed to spread (Section 5.4.3).
    day_index, high, low, true_spread_bp = _minute_aggregated_daily_ohlc_panel()
    dh, dl = aggregate_minute_to_daily_ohlc(day_index, high, low)
    s_bp = corwin_schultz_spread_bp(dh, dl)
    assert math.isfinite(s_bp)
    assert s_bp >= CS_REJECT_ORDER_OF_MAGNITUDE * true_spread_bp


def test_daily_ohlc_aggregation_is_max_high_min_low_per_day():
    # aggregate_minute_to_daily_ohlc: daily high = max of the day's 1-min highs, daily low = min.
    day_index = np.array([0, 0, 0, 1, 1])
    high = np.array([10.0, 12.0, 11.0, 9.0, 9.5])
    low = np.array([9.0, 9.5, 8.0, 7.0, 7.5])
    dh, dl = aggregate_minute_to_daily_ohlc(day_index, high, low)
    assert list(dh) == [12.0, 9.5]
    assert list(dl) == [8.0, 7.0]


def test_cs_estimator_on_a_zero_range_panel_is_zero_spread():
    # Sanity floor: if there is NO intraday range at all (H == L every bar, no volatility to conflate),
    # the estimator returns ~0 spread — proving the blowup above is driven by the range, not a bug that
    # always returns a large number regardless of input.
    day_index = np.repeat(np.arange(10), 100)
    price = np.full(day_index.shape[0], 100.0)
    rt_bp = corwin_schultz_round_trip_bp(day_index, price, price)
    # only the taker legs remain (2 * 5bp), the CS spread term is ~0 on a range-free panel.
    assert rt_bp == 10.0


def test_documented_negative_result_is_order_of_magnitude_wrong():
    # COST.md Section 5.4.3: the documented rejected estimate is bounded to the same order of
    # magnitude — >= 10x the measured 13bp round-trip anchor — which is exactly why it was rejected.
    assert CS_REJECTED_ROUND_TRIP_BP / MEASURED_ROUND_TRIP_BP >= CS_REJECT_ORDER_OF_MAGNITUDE
    assert CS_REJECTED_DRAG_PCT_PER_YEAR > 0.0
    assert MEASURED_ROUND_TRIP_BP == 13.0  # 2*(5 + 1 + 0.5)bp, the accounting.py default


# --------------------------------------------------------------------------------------------------
# LAYER 2 — structural anti-reintroduction scan: no CS identifier/import in the cost path.
# --------------------------------------------------------------------------------------------------


def test_no_corwin_schultz_anywhere_in_the_cost_path():
    # No code-level Corwin-Schultz identifier or import in the cost path (the reject module — which
    # DOES host the estimator so the result is reproducible — is exempt by filename).
    assert corwin_schultz_offenders(_repo_root()) == []


def test_guard_catches_a_reintroduction(tmp_path):
    # Prove the guard would fail loudly: plant a synthetic offender and scan it.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "bad_spread.py").write_text(
        "from corwin_schultz import estimate_spread\n\n"
        "def half_spread(high, low):\n    return estimate_spread(high, low)\n",
        encoding="utf-8",
    )
    offenders = corwin_schultz_offenders(tmp_path)
    assert offenders == ["research/xsec/bad_spread.py::corwin_schultz"]


def test_guard_sees_identifiers_not_docstrings(tmp_path):
    # Documentation mentions are allowed (this very invariant must be citable in docstrings);
    # only real identifiers/imports trip the guard.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "notes.py").write_text(
        '"""Corwin-Schultz was rejected here — see COST.md Section 5.4.3."""\n'
        "# corwin schultz is a documented dead end\nX = 1\n",
        encoding="utf-8",
    )
    assert corwin_schultz_offenders(tmp_path) == []


def test_guard_fails_loud_on_unparseable_file(tmp_path):
    # A file the AST cannot parse is reported, never silently skipped: the guard must fail closed.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "broken.py").write_text("def (:\n", encoding="utf-8")
    assert corwin_schultz_offenders(tmp_path) == ["research/xsec/broken.py::<unparseable>"]


def test_guard_exempts_legitimate_imports_of_its_own_api(tmp_path):
    # Review gap 4: a cost-path file that legitimately imports FROM the guard module (e.g. a
    # promotion runner or baselines consuming corwin_schultz_offenders) must NOT trip the guard.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "runner.py").write_text(
        "from research.xsec.cost.validate.corwin_schultz_reject import (\n"
        "    corwin_schultz_offenders,\n"
        "    MEASURED_ROUND_TRIP_BP,\n"
        ")\n\n"
        "def check(root):\n    return corwin_schultz_offenders(root)\n",
        encoding="utf-8",
    )
    assert corwin_schultz_offenders(tmp_path) == []


def test_guard_exempts_module_by_name_import(tmp_path):
    # `import ...corwin_schultz_reject as guard` (importing the module by dotted path) is exempt too.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "runner2.py").write_text(
        "import research.xsec.cost.validate.corwin_schultz_reject as guard\n\n"
        "USE = guard.MEASURED_ROUND_TRIP_BP\n",
        encoding="utf-8",
    )
    assert corwin_schultz_offenders(tmp_path) == []


def test_guard_still_fires_on_a_real_corwin_schultz_estimator_import(tmp_path):
    # The exemption is NARROW: importing a DIFFERENT module that actually reintroduces the estimator
    # (module named corwin_schultz, not the guard's corwin_schultz_reject) still fails loudly.
    pkg = tmp_path / "research" / "xsec"
    pkg.mkdir(parents=True)
    (pkg / "bad.py").write_text(
        "from corwin_schultz import estimate_spread\n\n"
        "def hs(h, low):\n    return estimate_spread(h, low)\n",
        encoding="utf-8",
    )
    offenders = corwin_schultz_offenders(tmp_path)
    assert offenders == ["research/xsec/bad.py::corwin_schultz"]
