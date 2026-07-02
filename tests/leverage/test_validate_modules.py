"""§9-named validate + synth modules: labels firewall, reports pass-through, synth series."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from convexity.leverage.ceilings.governor import governor_state
from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.validate.labels import (
    align_decisions_to_labels,
    future_weekly_labels,
)
from convexity.leverage.validate.reports import render_decision
from convexity.leverage.synth.generate import (
    calm_series,
    drawdown_series,
    regime_break_series,
)
from convexity.leverage.validate.backtest import reproduce_worked_example


def test_labels_are_strictly_future_only():
    # §10 firewall: label[k] = net[k+1] (the return AFTER decision k); the last week is unlabelled.
    net = np.array([0.01, -0.02, 0.03, 0.04])
    labels = future_weekly_labels(net)
    assert labels.tolist() == [-0.02, 0.03, 0.04]  # net[1:], strictly forward
    assert labels.size == net.size - 1  # last decision dropped, never back-filled


def test_labels_reject_series_too_short_to_form_a_forward_label():
    with pytest.raises(ValueError):
        future_weekly_labels(np.array([0.01]))


def test_align_decisions_pairs_each_with_its_strictly_future_label():
    g = np.array([1.0, 0.9, 0.8, 0.7])
    net = np.array([0.01, -0.02, 0.03, 0.04])
    g_aligned, labels = align_decisions_to_labels(g, net)
    assert g_aligned.tolist() == [1.0, 0.9, 0.8]  # decision at k scored on net[k+1]
    assert labels.tolist() == [-0.02, 0.03, 0.04]
    assert g_aligned.size == labels.size


def test_align_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        align_decisions_to_labels(np.array([1.0, 0.9]), np.array([0.01, 0.02, 0.03]))


def test_validate_modules_never_imported_by_live_layers_invariant_v():
    # Invariant V (§10): no live layer imports leverage/validate (labels/reports/backtest/arl).
    import convexity.leverage as lev

    root = pathlib.Path(lev.__file__).parent
    live_dirs = {"vol", "edge", "ceilings", "ratchet", "combine"}
    for path in root.rglob("*.py"):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in live_dirs:
            src = path.read_text()
            assert "leverage.validate" not in src and "from ..validate" not in src


def test_reports_is_pure_pass_through_no_re_derivation():
    # §9: reports.py contains NO statistical machinery (Invariant K-CI — no duplicate bootstrap/DSR).
    res = reproduce_worked_example()
    report = render_decision(res.base)
    assert report.g_t == res.base.g_t
    assert report.binding_ceiling == res.base.binding_ceiling
    assert report.kelly_term == res.base.kelly_term
    # AST scan (not a substring scan of prose): reports.py must define no statistic and import no
    # estimator — it may only NAME them in its docstring explaining why it re-derives nothing.
    import ast

    src = pathlib.Path(render_decision.__code__.co_filename).read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("robustness" in m or "random" in m for m in imported), (
        "reports.py must not import a bootstrap/RNG estimator (Inv K-CI)"
    )
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    for banned in ("block_bootstrap_ci", "deflate", "deflated_sharpe", "_sharpe"):
        assert banned not in called, f"reports.py must not call the estimator {banned} (Inv K-CI)"


def test_synth_series_are_deterministic_and_shaped():
    assert calm_series(n=100, seed=0).shape == (100,)
    assert np.array_equal(calm_series(n=50, seed=3), calm_series(n=50, seed=3))  # deterministic


def test_drawdown_series_actually_produces_a_governor_derisk():
    # §6.4: the drawdown fixture walks the governor down its ladder (m_t < 1 by the end).
    d = LeverageDefaults()
    series = drawdown_series(n=120, dd_depth=0.30, dd_weeks=20, seed=1)
    gov = governor_state(series, d)
    assert gov.dd_t > d.dd_ref * 0.5  # crossed at least the first ladder breakpoint
    assert gov.m_t < 1.0  # the governor actually de-risked on this synthetic path


def test_regime_break_series_shifts_vol_midway():
    # §4.1/§6.3: the second half has materially higher realized vol than the first.
    series = regime_break_series(n=200, sigma1=0.02, sigma2=0.08, break_at=0.5, seed=2)
    first_vol = series[:100].std()
    second_vol = series[100:].std()
    assert second_vol > 2.0 * first_vol
