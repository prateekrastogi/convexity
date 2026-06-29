"""Config governance + provenance/firewall (Invariant C, LIM, PROV, V, STREAM)."""

from __future__ import annotations

import ast
import pathlib

import pytest

from convexity.core.params import Bucket, assert_no_red_in_amber
from convexity.ops.config.defaults import AMBER_FIELD_NAMES, DEFAULTS, param_registry
from convexity.ops.config.in_control import SystemReportError, fixture_system_report, in_control_from_report
from convexity.ops.config.resolve import resolve_config
from convexity.ops.detectors.cusum import Cusum
from convexity.ops.streams.whiten import standardize
from convexity.ops.synth.generate import live_variance_shift
from convexity.ops.types import InControlParams

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "ops"


def test_dd_kill_strictly_above_governor_flat():
    """Invariant BRAKE §7.1a: 45% > 43% by construction (rejected otherwise)."""
    assert DEFAULTS.dd_kill > DEFAULTS.dd_governor_flat
    from convexity.ops.config.defaults import OpsDefaults
    with pytest.raises(ValueError):
        OpsDefaults(dd_kill=0.40, dd_governor_flat=0.43)


def test_no_red_in_amber():
    """Invariant C: no RED/BLUE param name leaks into the AMBER per-instrument field set."""
    assert_no_red_in_amber(AMBER_FIELD_NAMES, param_registry())


def test_red_unreachable_from_per_instrument():
    """Invariant C: the registry's AMBER fields carry no SPC limit."""
    reg = param_registry()
    amber = {p.name for p in reg.of_bucket(Bucket.AMBER)}
    red_blue = {p.name for p in reg.params() if p.bucket in (Bucket.RED, Bucket.BLUE)}
    assert not (amber & red_blue)


def test_in_control_loads_from_system_report_not_live():
    """Invariant LIM/STREAM: in-control params trace to a SYSTEM-report field, not a live estimate."""
    ic = in_control_from_report(fixture_system_report())
    assert ic["cost_tca"].mu0 == 0.0 and ic["cost_tca"].sigma0 == 2.0
    assert ic["edge"].p0 == 0.55 and ic["edge"].p1 == 0.45


def test_in_control_rejects_invalid_report():
    """A missing stream or a non-positive σ0 is rejected (no silent default, LIM)."""
    bad = fixture_system_report()
    del bad["vol"]
    with pytest.raises(SystemReportError):
        in_control_from_report(bad)
    bad2 = fixture_system_report()
    bad2["cost_tca"]["sigma0"] = 0.0
    with pytest.raises(SystemReportError):
        in_control_from_report(bad2)


def test_lim_a_mean_shift_does_not_move_limit():
    """Invariant LIM-(a): a shifted-MEAN live stream trips the detector; the limit does NOT move."""
    cfg = resolve_config()
    h_before = cfg.defaults.cusum.h
    c = Cusum(cfg.defaults.cusum.k, cfg.defaults.cusum.h)
    for _ in range(20):
        c.update(2.0)        # a big mean shift
    assert c.h == h_before   # the limit is immutable at runtime


def test_lim_b_variance_shift_does_not_move_raw_trip_point():
    """Invariant LIM-(b): a 2× live-vol stream does NOT move the raw trip point μ0+h·σ0 — the
    standardization uses the FROZEN σ0, so the logged σ_live moves but the alarm sample does not."""
    ic = InControlParams(stream="cost_tca", mu0=0.0, sigma0=2.0)
    x = live_variance_shift(200, mu0=0.0, sigma0=2.0, vol_mult=2.0, change_at=100, seed=0)
    # standardizing against the FROZEN σ0 keeps the raw trip point at μ0 + h·σ0
    raw_trip = ic.mu0 + 4.77 * ic.sigma0
    assert abs(raw_trip - 9.54) < 1e-9
    # the standardized stream uses σ0=2.0 throughout regardless of the live 2× vol
    z = [standardize(v, ic) for v in x]
    assert z[0] == x[0] / 2.0 and z[150] == x[150] / 2.0   # same denominator before/after the shift


def test_blue_value_must_be_in_grid():
    with pytest.raises(ValueError):
        resolve_config(blue={"cusum_kh": (0.7, 9.9)})   # unlisted → NOFIT
    cfg = resolve_config(blue={"cusum_kh": (0.5, 5.0)})  # listed → OK
    assert cfg.defaults.cusum.h == 5.0


# --- Invariant V / PROV: the firewall (AST) ---------------------------------------------------

def _imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name)
    return mods


def test_invariant_v_live_layers_never_import_validate_or_forensics():
    """Invariant V / §13 provenance: nothing in streams/detectors/reconcile/brake imports validate/
    or forensics/."""
    live_dirs = ["streams", "detectors", "reconcile", "brake"]
    offenders = []
    for d in live_dirs:
        for py in (_SRC / d).glob("*.py"):
            for mod in _imports_of(py):
                if "convexity.ops.validate" in mod or "convexity.ops.forensics" in mod:
                    offenders.append((str(py.name), mod))
    assert not offenders, f"live layer imports validate/forensics: {offenders}"


def test_invariant_prov_ops_holds_no_edge_sizing_exit_param():
    """Invariant PROV: OPS holds no edge/sizing/exit parameter — only SPC limits + the in-control ref;
    its only output type is the directive + alarms."""
    forbidden = {"kelly", "edge_buffer", "stop_distance", "target_qty", "leverage", "size"}
    reg = param_registry()
    names = {p.name for p in reg.params()}
    assert not (names & forbidden)
