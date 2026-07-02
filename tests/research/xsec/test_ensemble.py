"""Ensemble-averaging driver (PORTFOLIO.md §6.2 / §13): E-align, dollar-neutrality under the
average, Invariant B-band, and the E5 (warmup-max) / E8 (misalignment must raise) edge cases."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import convexity.portfolio.engine as eng
from convexity.portfolio.engine import (
    LOOKBACK_BAND,
    XSecConfig,
    band_configs,
    run,
    run_book,
    run_ensemble,
)
from convexity.panel.panel import Panel
from tests.research.xsec.conftest import make_panel

# "machine-epsilon tolerance" (§13): leg sums are a handful of terms around 0.5, so a few ulps.
EPS = 1e-14


# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------


@pytest.fixture
def aligned_panel() -> Panel:
    """10 symbols on deterministic monotone geometric trends: EVERY lookback ranks identically,
    so all three band members select the same legs — the fixture proving the averaged book keeps
    legs at exactly +gross/2 / -gross/2 (§13 gate iii, 'at the same gross')."""
    n_days, n_sym = 150, 10
    symbols = tuple(f"S{i:02d}" for i in range(n_sym))
    t = np.arange(n_days)[:, None]
    drift = np.linspace(-0.004, 0.005, n_sym)[None, :]
    close = 100.0 * (1.0 + drift) ** t
    funding = np.full((n_days, n_sym), 1.0e-4)
    listed = np.ones((n_days, n_sym), dtype=bool)
    return make_panel(close, funding, listed, symbols)


@pytest.fixture
def crossing_panel() -> Panel:
    """12 symbols, 200 days, seeded oscillating paths so the L=12/14/16 rank orders DISAGREE —
    the fixture where the ensemble average is materially different from any single member."""
    rng = np.random.default_rng(7)
    n_days, n_sym = 200, 12
    symbols = tuple(f"C{i:02d}" for i in range(n_sym))
    t = np.arange(n_days)[:, None]
    drift = np.linspace(-0.002, 0.002, n_sym)[None, :]
    phase = rng.uniform(0.0, 2.0 * np.pi, n_sym)[None, :]
    wiggle = 0.05 * np.sin(2.0 * np.pi * t / 17.0 + phase)
    walk = 0.004 * rng.normal(0.0, 1.0, (n_days, n_sym)).cumsum(axis=0)
    close = 100.0 * np.exp(drift * t + wiggle + walk)
    funding = np.full((n_days, n_sym), 5.0e-5)
    listed = np.ones((n_days, n_sym), dtype=bool)
    return make_panel(close, funding, listed, symbols)


def _leg_sums(w: np.ndarray) -> tuple[float, float]:
    """(long-leg sum, short-leg sum) of one weight row."""
    return float(w[w > 0.0].sum()), float(w[w < 0.0].sum())


# ------------------------------------------------------------------------------------------
# Invariant B-band (§6.2): the deployed set is exactly {12, 14, 16}
# ------------------------------------------------------------------------------------------


def test_band_configs_builds_the_canonical_deployed_ensemble():
    cfgs = band_configs()
    assert tuple(c.lookback for c in cfgs) == LOOKBACK_BAND == (12, 14, 16)
    for c in cfgs:
        assert c.signal == "momentum"
        assert c.quantile == "quintile"
        assert c.weighting == "volscaled"
        assert c.rebalance_days == 7
    # membership is order-insensitive: same band, any order, is fine.
    assert [c.lookback for c in band_configs((16, 12, 14))] == [12, 14, 16]


@pytest.mark.parametrize(
    "bad",
    [
        (12, 13, 14, 15, 16),  # dense fill of [12, 16]: L=13/L=15 were never backtested (§6.2)
        (12, 14),  # dropped member
        (14,),  # single fitted L — the exact P6 violation B-band forbids
        (12, 14, 18),  # substituted member
        (12, 14, 16, 18),  # widened band
        (12, 12, 14, 16),  # duplicated canonical member: must NOT silently de-dup to the band
        (12, 14, 14, 16),  # duplicate in the middle
    ],
)
def test_band_configs_rejects_any_other_lookback_set(bad):
    with pytest.raises(ValueError, match="B-band"):
        band_configs(bad)


def test_band_configs_rejects_duplicated_canonical_band():
    # regression for the silent-dedup edge: (12,12,14,16) de-dups to {12,14,16} but is a 4-member
    # multiset — accepting it would double L=12's vote in the §6.2 equal-weight average.
    with pytest.raises(ValueError, match="B-band"):
        band_configs((12, 12, 14, 16))
    # the canonical band still builds cleanly (guard against an over-tight multiset check).
    assert [c.lookback for c in band_configs((12, 14, 16))] == [12, 14, 16]


def test_band_configs_rejects_non_momentum_base():
    # §11 E7: the band's PBO evidence is momentum-specific; carry gets no band without its own CPCV.
    with pytest.raises(ValueError, match="momentum-only"):
        band_configs(base=XSecConfig(signal="carry"))


# ------------------------------------------------------------------------------------------
# §13 gate (i) — Invariant E-align: one Panel object, passed by reference to every member
# ------------------------------------------------------------------------------------------


def test_same_panel_object_passed_to_every_member(crossing_panel, monkeypatch):
    seen: list[Panel] = []
    real = eng.run_book

    def spy(panel: Panel, cfg: XSecConfig, **kw):
        seen.append(panel)
        return real(panel, cfg, **kw)

    monkeypatch.setattr(eng, "run_book", spy)
    run_ensemble(crossing_panel, band_configs())
    assert len(seen) == 3
    # the SAME object, by reference — run_ensemble constructs no Panel internally.
    assert all(p is crossing_panel for p in seen)
    # and, per the spec's frozen-dataclass phrasing, symbols/dates are byte-identical too.
    assert all(p.symbols == crossing_panel.symbols for p in seen)
    assert all(p.dates.equals(crossing_panel.dates) for p in seen)


# ------------------------------------------------------------------------------------------
# §13 gates (ii) + (iii) — dollar-neutrality: per member AND independently on the average
# ------------------------------------------------------------------------------------------


def test_each_member_weight_path_is_dollar_neutral(crossing_panel):
    cfgs = band_configs()
    for cfg in cfgs:
        _, book = run_book(crossing_panel, cfg)
        assert len(book) > 0
        for w in book.weights:
            long_sum, short_sum = _leg_sums(w)
            assert abs(long_sum - cfg.gross / 2.0) < EPS
            assert abs(short_sum + cfg.gross / 2.0) < EPS


def test_averaged_path_is_dollar_neutral_asserted_independently(crossing_panel):
    # §13 (iii): assert neutrality ON THE AVERAGED OUTPUT itself, not as a corollary of (ii).
    cfgs = band_configs()
    books = [run_book(crossing_panel, cfg)[1] for cfg in cfgs]
    w_avg = np.mean(np.stack([b.weights for b in books], axis=0), axis=0)
    signs = np.stack([np.sign(b.weights) for b in books], axis=0)
    for w in w_avg:
        long_sum, short_sum = _leg_sums(w)
        assert abs(long_sum + short_sum) < EPS  # zero-sum: long leg exactly offsets short leg
        assert float(np.abs(w).sum()) <= cfgs[0].gross + EPS  # gross never exceeds the dial
    # the fixture is meaningful: members DISAGREE on leg membership somewhere, so the average is
    # a genuine blend (and realized gross dips below G where a long under one L cancels a short).
    assert bool((signs.min(axis=0) != signs.max(axis=0)).any())


def test_averaged_legs_keep_gross_over_two_when_members_agree(aligned_panel):
    # monotone trends -> identical ranks for L=12/14/16 -> identical legs -> the averaged book
    # keeps legs at exactly +G/2 / -G/2 (§6.2: 'dollar-neutral at gross G, no re-normalization').
    cfgs = band_configs()
    books = [run_book(aligned_panel, cfg)[1] for cfg in cfgs]
    w_avg = np.mean(np.stack([b.weights for b in books], axis=0), axis=0)
    assert w_avg.shape[0] > 0
    for w in w_avg:
        long_sum, short_sum = _leg_sums(w)
        assert abs(long_sum - 0.5) < EPS
        assert abs(short_sum + 0.5) < EPS


# ------------------------------------------------------------------------------------------
# §13 gate (iv) — E8: corrupted symbol ordering must raise, never mis-add by position
# ------------------------------------------------------------------------------------------


def test_corrupted_symbol_ordering_raises(crossing_panel, monkeypatch):
    real = eng.run_book
    n_calls = {"n": 0}

    def corrupting(panel: Panel, cfg: XSecConfig, **kw):
        res, book = real(panel, cfg, **kw)
        n_calls["n"] += 1
        if n_calls["n"] == 2:  # the second member comes back with a permuted symbol ordering
            perm = np.arange(len(book.symbols))[::-1]
            book = dataclasses.replace(
                book,
                symbols=tuple(book.symbols[i] for i in perm),
                weights=book.weights[:, perm],
            )
        return res, book

    monkeypatch.setattr(eng, "run_book", corrupting)
    with pytest.raises(ValueError, match="symbol ordering"):
        run_ensemble(crossing_panel, band_configs())


# ------------------------------------------------------------------------------------------
# §13 gate (v) — a length-1 ensemble reproduces run() EXACTLY (no parallel reimplementation)
# ------------------------------------------------------------------------------------------


def test_single_config_ensemble_reproduces_run_exactly(crossing_panel):
    cfg = XSecConfig(signal="momentum", lookback=14, quantile="quintile", weighting="volscaled")
    res_single = run(crossing_panel, cfg)
    res_ens = run_ensemble(crossing_panel, [cfg])
    assert res_ens.dates.equals(res_single.dates)
    # exact float equality, not allclose: the driver is a strict generalization of run().
    assert np.array_equal(res_ens.net, res_single.net)
    assert np.array_equal(res_ens.gross_price, res_single.gross_price)
    assert np.array_equal(res_ens.funding, res_single.funding)
    assert np.array_equal(res_ens.turnover, res_single.turnover)
    assert np.array_equal(res_ens.cost, res_single.cost)
    assert np.array_equal(res_ens.universe_size, res_single.universe_size)


# ------------------------------------------------------------------------------------------
# E5 — warmup-max across the band; misaligned per-L outputs must RAISE, never truncate
# ------------------------------------------------------------------------------------------


def test_ensemble_starts_at_the_band_max_warmup(crossing_panel):
    cfgs = band_configs()
    res_ens = run_ensemble(crossing_panel, cfgs)
    member_starts = [run(crossing_panel, cfg).dates[0] for cfg in cfgs]
    members = [run(crossing_panel, cfg) for cfg in cfgs]
    # §13 LITERAL raise-clause reconciled with §11 E5: the bare §13 docstring says the driver
    # raises unless every member's `dates` are identical. For the CANONICAL band {12,14,16} the
    # warmup is vol_window-dominated, so all three members warm on the SAME first Friday and
    # their full date sequences ARE identical — the §13 literal 'dates identical' path holds and
    # the E5 front-trim is a verified no-op (0 rows trimmed). §13 raises only on a RESIDUAL
    # post-trim mismatch (E8), never on the deployed band.
    assert all(m.dates.equals(members[0].dates) for m in members[1:]), (
        "canonical band must warm on identical Fridays so §13's literal dates-identical "
        "contract holds and the E5 trim is a no-op"
    )
    assert res_ens.dates.equals(members[0].dates)  # trim removed zero rows for the band
    # the ensemble is only live from the first rebalance at which ALL members are warm (E5).
    assert res_ens.dates[0] == max(member_starts)
    assert all(res_ens.dates[0] >= s for s in member_starts)
    # weekly Friday-close cadence is preserved through the driver.
    assert all(d.weekday() == 4 for d in res_ens.dates)
    gaps = np.diff(res_ens.dates.values).astype("timedelta64[D]").astype(int)
    assert set(gaps.tolist()) == {7}


def test_ensemble_front_trims_to_max_warmup_when_members_differ(crossing_panel):
    # §11 E5: L=60 warms up on a LATER Friday than L=12, but both share the panel + weekly Friday
    # cadence, so L=60's rebalance grid is a strict suffix of L=12's. The ensemble is front-trimmed
    # to the max-warmup start (L=60's) and SUCCEEDS — it does NOT raise. This is the "trims a few
    # extra days off the front" behavior E5 names, which the {12,14,16} band never exercises
    # (all three snap to the same first Friday).
    short_cfg = XSecConfig(signal="momentum", lookback=12, quantile="quintile", weighting="volscaled")
    long_cfg = XSecConfig(signal="momentum", lookback=60, quantile="quintile", weighting="volscaled")
    r_short = run(crossing_panel, short_cfg)
    r_long = run(crossing_panel, long_cfg)
    # precondition: the two members genuinely warm up at different first Fridays (else this test
    # would not exercise the trim at all).
    assert r_long.dates[0] > r_short.dates[0]

    res_ens = run_ensemble(crossing_panel, [short_cfg, long_cfg])
    # the ensemble starts at the LATER (max-warmup) member's first rebalance ...
    assert res_ens.dates[0] == max(r_short.dates[0], r_long.dates[0])
    assert res_ens.dates[0] == r_long.dates[0]
    # ... and runs over exactly the common suffix (== the later member's full length here, since
    # its grid is a suffix of the earlier's), preserving the weekly Friday cadence.
    assert res_ens.dates.equals(r_long.dates)
    assert all(d.weekday() == 4 for d in res_ens.dates)
    gaps = np.diff(res_ens.dates.values).astype("timedelta64[D]").astype(int)
    assert set(gaps.tolist()) == {7}
    # the decomposition identity holds on the trimmed, re-settled series, and w_prev started flat
    # at the trimmed front (no phantom turnover carried over the trim boundary).
    assert np.allclose(res_ens.net, res_ens.gross_price + res_ens.funding - res_ens.cost)
    # §11 E5: universe_size over the trimmed common suffix is the element-wise MIN across
    # members (min of the FRONT-TRIMMED short-member series and the long member), pinning the
    # aggregation choice on the trim path too — not just the aligned/length-1 paths.
    short_trim = r_short.universe_size[r_short.dates.isin(res_ens.dates)]
    expected_min = np.minimum(short_trim, r_long.universe_size)
    assert np.array_equal(res_ens.universe_size, expected_min)


def test_genuine_cadence_mismatch_raises_after_trim(crossing_panel, monkeypatch):
    # §11 E8: trimming repairs only a warmup OFFSET on the shared Friday grid. A member whose
    # rebalance ROWS are off-grid (not a suffix of the others) cannot be aligned by trimming, so
    # after the front-trim the residual row disagreement must RAISE, never mis-add by position.
    real = eng.run_book
    n_calls = {"n": 0}

    def offgrid(panel: Panel, cfg: XSecConfig, **kw):
        res, book = real(panel, cfg, **kw)
        n_calls["n"] += 1
        if n_calls["n"] == 2:  # shift the 2nd member's rebalance rows off the shared grid by +1
            book = dataclasses.replace(book, d_index=book.d_index + 1)
        return res, book

    monkeypatch.setattr(eng, "run_book", offgrid)
    with pytest.raises(ValueError, match="rows"):
        run_ensemble(crossing_panel, band_configs())


def test_settle_parameter_mismatch_raises(crossing_panel):
    cfgs = [
        XSecConfig(signal="momentum", lookback=12, quantile="quintile", weighting="volscaled"),
        XSecConfig(
            signal="momentum", lookback=14, quantile="quintile", weighting="volscaled", gross=2.0
        ),
    ]
    with pytest.raises(ValueError, match="settle parameters"):
        run_ensemble(crossing_panel, cfgs)


def test_empty_config_list_raises(crossing_panel):
    with pytest.raises(ValueError, match="at least one config"):
        run_ensemble(crossing_panel, [])


# ------------------------------------------------------------------------------------------
# §13 docstring — re-settle at the averaged weight, NOT an average of per-member nets
# ------------------------------------------------------------------------------------------


def test_ensemble_settles_the_averaged_book_not_the_average_of_nets(crossing_panel):
    cfgs = band_configs()
    res_ens = run_ensemble(crossing_panel, cfgs)
    members = [run(crossing_panel, cfg) for cfg in cfgs]
    mean_gp = np.mean(np.stack([m.gross_price for m in members]), axis=0)
    mean_fnd = np.mean(np.stack([m.funding for m in members]), axis=0)
    mean_trn = np.mean(np.stack([m.turnover for m in members]), axis=0)
    mean_cst = np.mean(np.stack([m.cost for m in members]), axis=0)
    mean_net = np.mean(np.stack([m.net for m in members]), axis=0)
    # price P&L and funding are LINEAR in the weight path -> averaging commutes with settling.
    assert np.allclose(res_ens.gross_price, mean_gp, atol=1e-12)
    assert np.allclose(res_ens.funding, mean_fnd, atol=1e-12)
    # turnover/cost are NOT: |mean dw| <= mean |dw| (triangle inequality), strictly less
    # wherever members disagree — so the honest ensemble net is NOT the mean of member nets.
    assert bool(np.all(res_ens.turnover <= mean_trn + 1e-12))
    assert bool(np.all(res_ens.cost <= mean_cst + 1e-12))
    assert float(np.sum(mean_cst - res_ens.cost)) > 1e-9
    assert not np.allclose(res_ens.net, mean_net, atol=1e-12)
    # and the decomposition identity holds on the re-settled series.
    assert np.allclose(res_ens.net, res_ens.gross_price + res_ens.funding - res_ens.cost)


# ------------------------------------------------------------------------------------------
# §11 E5 — universe_size aggregation is pinned to the MIN across members (conservative
# "only live where EVERY member is warm/eligible"), so the choice cannot drift silently.
# ------------------------------------------------------------------------------------------


@pytest.fixture
def staggered_eligibility_panel() -> Panel:
    """crossing_panel with ONE name (idx 3) that lists on day 18, deep inside the L=16 trailing
    window but AFTER the L=12 windows start, at the bands shared first rebalance (row 32, Fri
    2022-02-04). Momentum eligibility needs ``listed[d-skip-L]`` (signals.py): at d=32 that is
    day 16 for L=16 (still unlisted -> name 3 INELIGIBLE) but day 20 for L=12/14 (listed ->
    ELIGIBLE). So at week 0 the members genuinely DISAGREE on the eligible count (L=12/14 -> 12,
    L=16 -> 11). All three share the same first Friday (warmup is vol_window-dominated), so no
    trim occurs and the min is over the fully-aligned series."""
    rng = np.random.default_rng(7)
    n_days, n_sym = 200, 12
    symbols = tuple(f"C{i:02d}" for i in range(n_sym))
    t = np.arange(n_days)[:, None]
    drift = np.linspace(-0.002, 0.002, n_sym)[None, :]
    phase = rng.uniform(0.0, 2.0 * np.pi, n_sym)[None, :]
    wiggle = 0.05 * np.sin(2.0 * np.pi * t / 17.0 + phase)
    walk = 0.004 * rng.normal(0.0, 1.0, (n_days, n_sym)).cumsum(axis=0)
    close = 100.0 * np.exp(drift * t + wiggle + walk)
    funding = np.full((n_days, n_sym), 5.0e-5)
    listed = np.ones((n_days, n_sym), dtype=bool)
    list_day = 18
    listed[:list_day, 3] = False
    close[:list_day, 3] = np.nan
    return make_panel(close, funding, listed, symbols)


def test_ensemble_universe_size_is_the_min_across_members(staggered_eligibility_panel):
    # §11 E5: the ensemble is only live where EVERY member is warm/eligible, so its reported
    # per-week universe_size is the element-wise MIN across the band members over the common
    # suffix — NOT the max, first, last, or mean. This test PINS that aggregation choice: it
    # would fail if run_ensemble drifted to any other reducer.
    panel = staggered_eligibility_panel
    cfgs = band_configs()
    members = [run(panel, cfg) for cfg in cfgs]
    res_ens = run_ensemble(panel, cfgs)

    # precondition: the members genuinely disagree on the eligible count somewhere (else the min
    # would coincide with max/first/last and the test would be vacuous).
    stacked = np.stack([m.universe_size for m in members])
    assert bool((stacked.min(axis=0) != stacked.max(axis=0)).any()), (
        "fixture must make the band members disagree on universe_size for the pin to bite"
    )
    # week 0 concretely: L=12/14 count 12 eligible, L=16 counts 11 (name 3 lists on day 18).
    assert int(members[0].universe_size[0]) == 12
    assert int(members[-1].universe_size[0]) == 11

    # all three share the first Friday here -> no trim; the pin is the element-wise min.
    assert all(m.dates.equals(res_ens.dates) for m in members)
    expected_min = stacked.min(axis=0)
    assert np.array_equal(res_ens.universe_size, expected_min)
    assert int(res_ens.universe_size[0]) == 11
    # explicitly NOT the max/mean (guards against a silent reducer swap).
    assert not np.array_equal(res_ens.universe_size, stacked.max(axis=0))
    assert res_ens.universe_size.dtype.kind == "i"
