"""§10 acceptance-gate: validate/backtest.py reproduces the §7 worked example (LEVERAGE.md §7, §10)."""

from __future__ import annotations

import pathlib

from convexity.leverage.validate.backtest import (
    WorkedExampleInputs,
    reproduce_worked_example,
)


def test_worked_example_reproduces_bit_for_bit():
    # §10 gate: the harness reproduces ALL of §7's tabulated steps from the tabulated inputs.
    res = reproduce_worked_example()
    assert res.matches
    # Step 1/2: kelly binds at 0.585.
    assert round(res.base.g_t, 3) == 0.585
    assert res.base.binding_ceiling == "kelly"
    # Step 3: growth-capture identity c(2−c) at c=0.25.
    assert round(res.growth_capture, 4) == 0.4375
    # Step 4: governor binds at 0.525 once dd=0.30 pushes m_t to 0.35.
    assert round(res.governor.g_t, 3) == 0.525
    assert res.governor.binding_ceiling == "governor"


def test_worked_example_would_fail_if_composition_broke():
    # The gate has TEETH: perturbing an input off its §7 row breaks the reproduction (so a real
    # live-layer regression in the composition would flip matches to False, not pass silently).
    bad = WorkedExampleInputs(sigma_hat=0.05)  # off the §7 3.89% row -> kelly term != 0.585
    res = reproduce_worked_example(bad)
    assert not res.matches


def test_backtest_reproduces_by_executing_live_modules_not_restating_arithmetic():
    # §10 / Invariant K-CI spirit: the harness must DRIVE the live composition, so it imports the
    # live kelly + combine modules (one source of truth), never hard-code 0.585/0.525 as the result.
    src = pathlib.Path(
        reproduce_worked_example.__code__.co_filename
    ).read_text()
    assert "from convexity.leverage.ceilings.kelly_vol_target import kelly_vol_target_term" in src
    assert "from convexity.leverage.combine.engine import compose_g_t" in src


def test_backtest_is_firewalled_from_live_layers():
    # Invariant V (§10): validate/backtest.py may import live layers, but no live layer imports it.
    import convexity.leverage as lev

    root = pathlib.Path(lev.__file__).parent
    live_dirs = {"vol", "edge", "ceilings", "ratchet", "combine"}
    for path in root.rglob("*.py"):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in live_dirs:
            src = path.read_text()
            assert "validate.backtest" not in src and "validate import backtest" not in src
