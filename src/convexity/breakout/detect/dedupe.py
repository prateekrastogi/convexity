"""SignalDeduper — cooldown / active-zone duplicate suppression (BREAKOUT §7 detect/dedupe).

Suppresses repeated signals from the same active breakout zone within a cooldown window. Symmetric
for long and short signals (BREAKOUT §detect/dedupe).

CAUSALITY (Invariant E batch/live parity): the rule is FORWARD-ONLY — the FIRST qualifying signal in
a cooldown window on a given side is kept and every later same-side signal within ``cooldown_bars`` of
it is suppressed. A "keep the highest-quality in the window" rule would be non-causal (it could prefer
a LATER signal, so the online path — which cannot see the future — would disagree with batch). Keeping
the earliest is the convex-breakout-correct choice anyway: the first cross is the actionable entry.
The quality magnitude stays internal (Invariant K firewall) and is not used to pick between dupes.
"""

from __future__ import annotations

from convexity.core.types import EntrySignal


class SignalDeduper:
    """Cooldown / active-zone dedupe (BREAKOUT §7). One public method, ``apply``."""

    def apply(self, signals: list[EntrySignal], cooldown_bars: int) -> list[EntrySignal]:
        """Suppress same-zone repeats within ``cooldown_bars``, keeping the EARLIEST (causal, Inv E).

        Symmetric in sign: long and short are deduped on their OWN side independently (a long and a
        short can fire close together — they are different zones).
        """
        if not signals:
            return []
        ordered = sorted(signals, key=lambda s: s.signal_idx)
        kept: list[EntrySignal] = []
        last_idx_by_side: dict[int, int] = {}  # side -> signal_idx of the last kept signal
        for sig in ordered:
            side = sig.direction
            prev = last_idx_by_side.get(side)
            if prev is not None and sig.signal_idx - prev <= cooldown_bars:
                continue  # within cooldown of the earlier kept signal on this side -> suppress
            kept.append(sig)
            last_idx_by_side[side] = sig.signal_idx
        return kept
