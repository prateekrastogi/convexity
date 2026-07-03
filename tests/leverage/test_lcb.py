"""SR_LCB_t — DSR-gated block-bootstrap CI lower bound (LEVERAGE.md §2.5–§2.6, §10)."""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from convexity.system.validate.robustness import block_bootstrap_ci

from convexity.leverage.config.defaults import LeverageDefaults
from convexity.leverage.edge.lcb import sr_lcb_t


def _passing_series(n=400, seed=7):
    """A long, strongly-positive weekly series engineered to PASS the weekly DSR gate.

    Long enough to clear the MinTRL/n_obs floor and with a high enough per-obs Sharpe that DSR>=0.95
    over a small trial family — verified empirically in the assertions below (we assert dsr_passed).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.02, 0.02, n)  # per-obs Sharpe ~1.0, n large -> passes


def test_k_ci_bit_for_bit_matches_block_bootstrap_ci_at_fixed_seed():
    # Invariant K-CI: SR_LCB_t is a PASS-THROUGH of block_bootstrap_ci on the SAME series, no drift.
    d = LeverageDefaults()
    r = _passing_series()
    trials = np.array([1.0, 0.9, 0.8, 1.1, 0.95])  # >= k_min distinct trials
    edge = sr_lcb_t(r, trials, d)
    assert edge.dsr_passed  # fixture must actually pass the gate for this branch to be exercised
    ci_low, ci_high = block_bootstrap_ci(
        r,
        block_length=d.bootstrap_block_length,
        n_boot=d.bootstrap_n,
        # mirror lcb.py's one-sided->two-sided conversion (LEVERAGE.md §3: one-sided 90% lower
        # bound = 10th percentile = two-sided 80% lower bound), so bit-for-bit still holds.
        ci_level=2.0 * d.bootstrap_ci_level - 1.0,
        seed=d.bootstrap_seed,
    )
    assert edge.ci_low == ci_low  # bit-for-bit (same seed, same params)
    assert edge.ci_high == ci_high
    assert edge.sr_lcb == max(0.0, ci_low)


def test_k_lcb_returns_zero_when_dsr_gate_fails():
    # Invariant K-LCB: DSR failing -> SR_LCB_t = 0 unconditionally, regardless of raw Sharpe size.
    d = LeverageDefaults()
    # a SHORT, tiny-sample series fails the MinTRL/track-length floor -> .passed is False.
    r = np.array([0.03, 0.02, 0.04, 0.01, 0.05])
    trials = np.array([1.0, 0.9, 0.8])
    edge = sr_lcb_t(r, trials, d)
    assert not edge.dsr_passed
    assert edge.sr_lcb == 0.0
    assert edge.gated_to_zero


def test_widening_ci_can_only_lower_sr_lcb():
    # Invariant K-LCB: more estimation uncertainty (wider CI, lower ci_low) can only LOWER SR_LCB_t.
    d_tight = LeverageDefaults()
    # a wider one-sided level pushes ci_low DOWN (more of the lower tail included).
    d_wide = LeverageDefaults(bootstrap_ci_level=0.99)
    r = _passing_series()
    trials = np.array([1.0, 0.9, 0.8, 1.1, 0.95])
    tight = sr_lcb_t(r, trials, d_tight)
    wide = sr_lcb_t(r, trials, d_wide)
    assert wide.ci_low <= tight.ci_low
    assert wide.sr_lcb <= tight.sr_lcb


def test_sr_lcb_never_negative():
    # max(0, ci_low): even a barely-positive/borderline series never yields a negative multiplier.
    d = LeverageDefaults()
    r = _passing_series(seed=99)
    edge = sr_lcb_t(r, np.array([0.1, 0.2, 0.15, 0.05, 0.12]), d)
    assert edge.sr_lcb >= 0.0


def test_edge_module_defines_no_own_bootstrap_and_no_pathsharpesummary():
    # Invariant K-CI static check: edge/lcb.py must not import PathSharpeSummary nor define a
    # duplicate bootstrap/stats module anywhere under leverage/.
    import convexity.leverage as lev

    root = pathlib.Path(lev.__file__).parent
    for path in root.rglob("*.py"):
        if path.name.startswith("."):
            continue  # skip editor/transfer artifacts (e.g. macOS ._ AppleDouble files)
        assert path.name != "stats.py", f"{path} is a forbidden duplicate stats module (K-CI)"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # no IMPORT of PathSharpeSummary anywhere (docstring prose that explains why we DON'T
            # use it is fine; an actual import is not — Invariant K-CI).
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                mod = getattr(node, "module", "") or ""
                assert "PathSharpeSummary" not in names, (
                    f"{path} imports PathSharpeSummary (K-CI)"
                )
                assert "cpcv" not in mod.split("."), (
                    f"{path} imports from cpcv for the CI (K-CI: use robustness.block_bootstrap_ci)"
                )
            # no locally-defined bootstrap CI function (must delegate to system.validate.robustness).
            if isinstance(node, ast.FunctionDef):
                assert "block_bootstrap" not in node.name, (
                    f"{path}:{node.name} re-defines a bootstrap CI (K-CI forbids re-derivation)"
                )


def test_sr_lcb_bypass_with_raw_scalar_is_impossible():
    # Invariant K-LCB: you cannot feed a bare scalar Sharpe; the API takes a return SERIES that must
    # flow through the CI/DSR objects. A scalar raises inside numpy statistics rather than sizing.
    d = LeverageDefaults()
    with pytest.raises((ValueError, TypeError, IndexError)):
        sr_lcb_t(0.98, np.array([1.0, 0.9, 0.8]), d)  # a raw Sharpe, not a series


def test_ci_lower_bound_is_the_tenth_percentile():
    """LEVERAGE.md §3: the SR_LCB CI is a ONE-SIDED 90% lower bound = the 10th percentile of the
    bootstrap Sharpe distribution (same tail mass as a two-sided 80% CI) — NOT the 5th percentile
    a raw two-sided 90% call would produce (the audit-found L-1 defect this test pins)."""
    d = LeverageDefaults()
    r = _passing_series()
    trials = np.array([1.0, 0.9, 0.8, 1.1, 0.95])
    edge = sr_lcb_t(r, trials, d)
    assert edge.dsr_passed

    # reconstruct the bootstrap Sharpe distribution exactly as robustness.block_bootstrap_ci does
    rng = np.random.default_rng(d.bootstrap_seed)
    n = r.size
    bl = d.bootstrap_block_length
    n_blocks = -(-n // bl)
    starts_pool = n - bl + 1
    stats = np.empty(d.bootstrap_n)
    for b in range(d.bootstrap_n):
        starts = rng.integers(0, starts_pool, size=n_blocks)
        sample = np.concatenate([r[s : s + bl] for s in starts])[:n]
        stats[b] = sample.mean() / sample.std(ddof=1)
    tenth = float(np.quantile(stats, 0.10))
    fifth = float(np.quantile(stats, 0.05))
    assert edge.ci_low == tenth  # the spec'd one-sided 90% bound
    assert edge.ci_low != fifth  # the L-1 defect (raw two-sided 90%) would land here
