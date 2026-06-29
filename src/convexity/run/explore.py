"""MODE 1 — EXPLORATION: lean pandas/numpy first-pass backtest on the historical lake (no engine).

DELIBERATELY a SEPARATE, ENGINE-FREE path — NOT a peer of backtest/paper/live and NOT a consumer of
``run/session.build_session`` (which stands up a Nautilus node). This is the fast research loop that
PRECEDES the parity backtest: a vectorised pandas/numpy sweep over ``data/stitched`` to triage range
definitions / stops / holds / allocations before paying for a full event-driven run. Net-of-cost;
report drop-top-N + OOS robustness; match comparisons across pairs/variants. See ``run/README.md`` for
where this sits relative to the three engine modes, and ``conf/explore.yaml`` for the knob surface.

UNIMPLEMENTED SCAFFOLD — owner to build (CLAUDE.md). Importing it must not pretend a research loop
exists; :func:`run_explore` raises until the lean sweep is written.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_explore"]


def run_explore(*_args: Any, **_kwargs: Any) -> Any:
    """The MODE 1 exploration sweep — NOT YET IMPLEMENTED.

    This is intentionally engine-free (it does not call ``build_session`` / stand up a Nautilus node);
    it is the lean pandas first-pass over ``data/stitched``. Raises until the sweep is written so a
    caller never silently gets a no-op research result.
    """
    raise NotImplementedError(
        "run/explore.py (MODE 1, lean pandas first-pass) is an unimplemented scaffold. It is a "
        "SEPARATE engine-free research path (not a backtest/paper/live peer — see run/README.md). "
        "Build the vectorised sweep over data/stitched (range/stop/hold/allocation knobs in "
        "conf/explore.yaml) before using it."
    )
