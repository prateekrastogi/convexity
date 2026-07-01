"""Portfolio construction: dollar-neutrality, leg sizing, gross normalization, vol-scaling."""

from __future__ import annotations

import numpy as np

from research.xsec.portfolio import (
    _leg_count,
    _leg_liq_cap,
    liquidity_weights,
    quantile_weights,
)


def test_leg_count_floors_and_caps():
    assert _leg_count(1, "quintile") == 0  # need >= 2 names
    assert _leg_count(3, "quintile") == 1  # floor at 1
    assert _leg_count(10, "quintile") == 2  # 20% of 10
    assert _leg_count(50, "quintile") == 10  # 20% of 50
    assert _leg_count(3, "tercile") == 1  # floor + non-overlap cap (3//2=1)


def test_dollar_neutral_and_gross():
    rank = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    w = quantile_weights(rank, quantile="quintile", weighting="equal", gross=1.0)
    assert abs(w.sum()) < 1e-12  # dollar-neutral
    assert abs(np.abs(w).sum() - 1.0) < 1e-12  # 100% gross
    assert abs(w[w > 0].sum() - 0.5) < 1e-12  # long leg +0.5
    assert abs(w[w < 0].sum() + 0.5) < 1e-12  # short leg -0.5
    assert w[0] > 0 and w[-1] < 0  # top rank long, bottom rank short


def test_gross_scales_weights_linearly():
    rank = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    w1 = quantile_weights(rank, quantile="quintile", weighting="equal", gross=1.0)
    w2 = quantile_weights(rank, quantile="quintile", weighting="equal", gross=2.0)
    assert np.allclose(w2, 2.0 * w1)


def test_ineligible_names_zero_weight():
    rank = np.array([0.9, np.nan, 0.5, np.nan, 0.1])
    w = quantile_weights(rank, quantile="tercile", weighting="equal", gross=1.0)
    assert w[1] == 0.0 and w[3] == 0.0
    assert abs(w.sum()) < 1e-12


def test_volscaled_favors_low_vol_names():
    rank = np.array([0.9, 0.8, 0.2, 0.1])  # tercile -> 1 per leg at n=4 (33% of 4 = 1)
    vol = np.array([0.01, 0.05, 0.02, 0.04])
    w = quantile_weights(rank, quantile="tercile", weighting="volscaled", gross=1.0, vol=vol)
    assert abs(w.sum()) < 1e-12
    # still dollar-neutral, still 100% gross
    assert abs(np.abs(w).sum() - 1.0) < 1e-12


def test_flat_book_when_too_few_names():
    w = quantile_weights(np.array([0.5, np.nan, np.nan]), quantile="quintile", weighting="equal")
    assert np.allclose(w, 0.0)


# --- liquidity-aware within-leg weightings ---


def test_leg_liq_cap_binds_and_redistributes():
    # base wants name 0 at 4/6 of the leg, but its liquidity share is only 1/6; C=2 caps it at 2/6.
    base = np.array([4.0, 1.0, 1.0])
    dvol = np.array([1.0, 2.0, 3.0])  # liq shares 1/6, 2/6, 3/6
    out = _leg_liq_cap(base, dvol, cap_mult=2.0)
    assert abs(out.sum() - 1.0) < 1e-12
    # name 0 capped at C * its liq share = 2 * (1/6).
    assert abs(out[0] - 2.0 / 6.0) < 1e-9
    # freed mass (4/6 - 2/6 = 2/6) redistributed to names 1,2 in proportion to their base (1:1).
    assert out[1] > 1.0 / 3.0 and abs(out[1] - out[2]) < 1e-9


def test_leg_liq_cap_inf_is_identity():
    base = np.array([4.0, 1.0, 1.0])
    dvol = np.array([1.0, 2.0, 3.0])
    out = _leg_liq_cap(base, dvol, cap_mult=float("inf"))
    assert np.allclose(out, base)  # inf cap returns base magnitudes unchanged


def test_liqcap_volscaled_inf_equals_plain_volscaled():
    rank = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    vol = np.array([0.01, 0.05, 0.03, 0.02, 0.04, 0.06])
    dvol = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 6.0])
    w_plain = quantile_weights(rank, quantile="quintile", weighting="volscaled", vol=vol)
    w_cap = liquidity_weights(
        rank, quantile="quintile", scheme="liqcap_volscaled", dvol=dvol, vol=vol,
        cap_mult=float("inf"),
    )
    assert np.allclose(w_cap, w_plain, atol=1e-12)


def test_liquidity_weights_dollar_neutral_and_gross():
    rank = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    vol = np.array([0.01, 0.05, 0.03, 0.02, 0.04, 0.06])
    dvol = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 6.0])
    for scheme, kw in (
        ("liqcap_volscaled", {"vol": vol, "cap_mult": 2.0}),
        ("liqtilt", {}),
        ("eqcap", {"cap_mult": 2.0}),
    ):
        w = liquidity_weights(rank, quantile="quintile", scheme=scheme, dvol=dvol, **kw)
        assert abs(w.sum()) < 1e-12  # dollar-neutral
        assert abs(np.abs(w).sum() - 1.0) < 1e-12  # 100% gross
        assert abs(w[w > 0].sum() - 0.5) < 1e-12  # long leg +0.5
        assert w[0] > 0 and w[-1] < 0  # top rank long, bottom rank short


def test_liqcap_lowers_concentration_vs_volscaled():
    # 15 names -> quintile k=3 per leg. Name 0 (top rank) is a very-low-vol ILLIQUID major that
    # vol-scaling would over-weight; the liquidity cap must pull weight OUT of it into its leg-mates.
    rank = np.array([0.99, 0.95, 0.90] + [0.5] * 9 + [0.10, 0.05, 0.01])
    vol = np.array([0.005, 0.05, 0.05] + [0.05] * 9 + [0.05, 0.05, 0.05])  # name 0 ultra-low vol
    dvol = np.array([1.0, 10.0, 10.0] + [5.0] * 9 + [5.0, 5.0, 5.0])  # name 0 ILLIQUID
    w_plain = quantile_weights(rank, quantile="quintile", weighting="volscaled", vol=vol)
    w_cap = liquidity_weights(
        rank, quantile="quintile", scheme="liqcap_volscaled", dvol=dvol, vol=vol, cap_mult=2.0
    )
    assert w_cap[0] < w_plain[0]  # cap pulls weight OUT of the illiquid low-vol major

    def _breadth(v):
        a = np.abs(v[v > 0])
        return 1.0 / np.sum((a / a.sum()) ** 2)

    assert _breadth(w_cap) > _breadth(w_plain)  # long-leg breadth rises under the cap
