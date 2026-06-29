"""Detector protocol + the WarmupMixin (alarms suppressed until the recursion settles).

Spec: OPS §6 (every detector is an O(1) causal recursion), §6.8 (warm-up). A detector is a streaming
recursion over one standardized stream: one ``update(z)`` per observation reads only ``z`` and bounded
prior state (Invariant CAUSAL), and returns a :class:`DetectorState`. ``reset()`` returns it to its
cold start. No detector reads another detector's output (Invariant V / architectural isolation).

The §6.8 warm-up is folded in here as a mixin: during the first ``N_warm`` samples the recursion
**still updates** (so state is correct the moment warm-up ends — no discontinuity) but the alarm is
**suppressed** (reported as ``warming_up=True``, ``alarm=False``). Warm-up suppresses only the
*statistical* (SPC) alarms; the binary recon/heartbeat/clock-skew/DD triggers are not detectors and
do not warm up (§6.8(a)).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from convexity.ops.types import DetectorName, DetectorState


@runtime_checkable
class Detector(Protocol):
    """The §6 detector seam: an O(1) causal recursion over one standardized stream.

    ``update(z)`` consumes one standardized observation and returns the post-update state; ``reset()``
    returns the recursion to its cold start. ``name`` identifies the family for the forensic chain.
    """

    name: DetectorName

    def update(self, z: float) -> DetectorState:
        """Advance the recursion by one standardized observation; return the post-update state."""
        ...

    def reset(self) -> None:
        """Return the recursion to its cold start (used on a gated re-arm, §8.4)."""
        ...


class WarmupMixin:
    """Shared warm-up bookkeeping for the SPC detectors (§6.8).

    A detector mixes this in, calls :meth:`_init_warmup` in ``__init__``, and calls
    :meth:`_tick_warmup` once per ``update``; the helper increments the sample counter and returns
    whether the detector is still inside ``N_warm``. The detector then builds its
    :class:`DetectorState` via :func:`make_state` so the recursion keeps updating while the *alarm* is
    suppressed to LOG during warm-up. :meth:`_reset_warmup` zeroes the counter so a gated re-arm
    restarts the warm-up (a freshly promoted config does not page on its first bars, §6.8(b)/§17.1).
    """

    n_warm: int = 0
    _n_seen: int = 0

    def _init_warmup(self, n_warm: int) -> None:
        self.n_warm = int(n_warm)
        self._n_seen = 0

    def _tick_warmup(self) -> bool:
        """Advance the warm-up counter by one sample; return True while still warming up."""
        self._n_seen += 1
        return self._n_seen <= self.n_warm

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def _reset_warmup(self) -> None:
        self._n_seen = 0


def make_state(
    *,
    detector: DetectorName,
    statistic: float,
    limit: float,
    breach: bool,
    sample_idx: int,
    warming_up: bool,
    diagnostics: dict[str, float] | None = None,
) -> DetectorState:
    """Build a :class:`DetectorState` applying the §6.8 warm-up suppression in one place.

    ``alarm`` is the raw ``breach`` gated by warm-up (suppressed to LOG while ``warming_up``);
    ``in_control`` is the inverse of the *un-suppressed* breach (the chart is genuinely out of control
    iff it breached, regardless of warm-up — warm-up suppresses the *response*, not the verdict, so
    the rollout gate can read ``in_control`` honestly, §17.1).
    """
    alarm = breach and not warming_up
    return DetectorState(
        detector=detector,
        in_control=not breach,
        statistic=statistic,
        limit=limit,
        alarm=alarm,
        sample_idx=sample_idx,
        warming_up=warming_up,
        diagnostics=diagnostics or {},
    )
