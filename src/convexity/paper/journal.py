"""Append-only session journal (PAPER.md §5) — the persisted history pure-recompute rebuilds from.

One JSONL file under ``data/paper/`` (gitignored with the lake). Every record carries ``kind``,
``ts`` (epoch-ms, wall clock) and ``asof`` (the rebalance/observation day it describes). Kinds:
``equity_mark`` {equity}, ``flow`` {amount, equity_before}, ``targets`` {d_iso, weights, g_t,
mode}, ``orders`` {intents: [...]}, ``settle`` {week_end_iso, r_twr, equity_start, equity_end,
n_flows}, ``recon`` {n_breach, detail}, ``directive`` {action, reason}. Append-only by contract —
a correction is a NEW record, never an edit (OPS provenance discipline).
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

DEFAULT_PATH = pathlib.Path("data/paper/journal.jsonl")


class Journal:
    def __init__(self, path: pathlib.Path | str = DEFAULT_PATH) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, asof: str, **payload: Any) -> dict:
        rec = {"kind": kind, "ts": int(time.time() * 1000), "asof": asof, **payload}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        return rec

    def read(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-append leaves a torn FINAL line (append-only contract). Tolerate
                # exactly that -- skip it LOUDLY so replay recovers (D1v2 failsafe-restart,
                # PAPER.md §12.1). A torn line anywhere ELSE means real
                # corruption: fail fast rather than silently dropping history.
                if i == len(lines) - 1:
                    print(f"[journal] WARNING: torn final line skipped ({self.path}) -- "
                          f"crash-mid-append recovery; the interrupted record was never "
                          f"acknowledged and is treated as never written")
                    continue
                raise
            if kind is None or rec.get("kind") == kind:
                out.append(rec)
        return out

    def last(self, kind: str) -> dict | None:
        recs = self.read(kind)
        return recs[-1] if recs else None
