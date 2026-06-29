"""Forensics replay + the import-isolated nautilus adapter/flatten (Invariant AUDIT, PROV, SAFE §9.1)."""

from __future__ import annotations

import ast
import pathlib

from convexity.ops.forensics.log import ForensicLog
from convexity.ops.forensics.replay import replay_halts
from convexity.ops.nautilus.adapter import OpsAdapter
from convexity.ops.nautilus.flatten import FlattenCommand, flatten_command_for
from convexity.ops.ops_api import build_engine
from convexity.ops.types import AlarmRecord, EquityRef, KillVerdict, Severity

_OPS = pathlib.Path(__file__).resolve().parents[2] / "src" / "convexity" / "ops"


def test_forensic_log_append_only_and_replayable():
    log = ForensicLog()
    log.append_kill(KillVerdict(state="halt", flatten=True, cancel_entries=True,
                                triggers=("cost_divergence_escalated",), reason_codes=()))
    log.append_alarm(AlarmRecord("cost_tca", "cusum", "cost_divergence_escalated", Severity.HALT,
                                 8.0, 8.0, 18, None, "C+ crossed h2"))
    recs = replay_halts(log)
    assert len(recs) == 1
    s = recs[0].summary()
    assert "cost_divergence_escalated" in s and "8.0" in s


def test_flatten_command_race_free_ordering():
    """§9.1b: the flatten command always cancels entries first, then issues the reduce-only flatten."""
    cmd = flatten_command_for(True)
    assert isinstance(cmd, FlattenCommand)
    assert cmd.cancel_entries_first and cmd.reduce_only
    assert flatten_command_for(False) is None


def test_adapter_on_bar_returns_directive_and_flatten():
    eng = build_engine()
    adapter = OpsAdapter(engine=eng)
    directive, flatten = adapter.on_bar(EquityRef(equity=540.0, peak=1000.0))  # 46% DD → HALT
    assert directive.state == "halt" and directive.flatten
    assert flatten is not None and flatten.reduce_only


def test_adapter_rejected_flatten_stays_unconfirmed():
    """§9.1c through the adapter: a rejected flatten keeps the state latched HALT, never spurious flat."""
    eng = build_engine()
    adapter = OpsAdapter(engine=eng)
    d1, f1 = adapter.on_bar(recon_mismatch=True)       # first HALT issues flatten
    assert d1.state == "halt" and f1 is not None
    adapter.report_flatten_outcome(rejected=True)
    directive, _ = adapter.on_bar(recon_mismatch=True)
    assert directive.state == "halt"
    assert directive.flatten                           # retried on the RED cadence (§9.1c)
    assert not eng.kill.flatten_confirmed              # never spuriously 'flat'


def _imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    return mods


def test_nautilus_layer_is_import_isolated():
    """The ops/nautilus/ files do NOT import nautilus (pure-python, import-isolated per the build
    directive)."""
    offenders = []
    for py in (_OPS / "nautilus").glob("*.py"):
        for mod in _imports_of(py):
            if mod == "nautilus_trader" or mod.startswith("nautilus_trader."):
                offenders.append((py.name, mod))
    assert not offenders, f"ops/nautilus imports nautilus: {offenders}"
