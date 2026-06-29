"""ActionTimer — decision_idx / action_idx conventions (TREND §8 data/timing, Inv J).

Every decision must carry ``decision_idx`` (the bar the decision becomes known), ``action_idx``
(the first bar the action can be evaluated/executed), and ``action_price_kind``. Default policy: a
decision known after bar close acts at the NEXT valid bar (``action_idx = decision_idx + 1``,
``next_open``). A same-bar policy must be EXPLICITLY marked executable (``close_if_live_executable``,
a validation/backtest convention reconciled to the venue stop in live mode, §6a). The timer skips
forward over invalid/halted bars so the action lands on a tradable bar.
"""

from __future__ import annotations

from convexity.core.types import Bars

ActionPriceKind = str  # Literal["next_open","next_mid","close_if_live_executable"]


class ActionTimer:
    """Map a decision bar to its actionable bar + price-kind (§8 / Inv J)."""

    def __init__(self, same_bar_executable: bool = False) -> None:
        self.same_bar_executable = same_bar_executable

    def action_index(
        self, decision_idx: int, bars: Bars, policy: str = "next_open"
    ) -> tuple[int, ActionPriceKind]:
        """Return ``(action_idx, action_price_kind)`` (§8).

        Default: act at the next valid bar. ``policy='close_if_live_executable'`` is allowed only
        when ``same_bar_executable`` was set (Inv J — no implicit same-bar exits).
        """
        n = len(bars)
        if policy == "close_if_live_executable":
            if not self.same_bar_executable:
                raise ValueError(
                    "same-bar exit not permitted unless explicitly executable (Inv J)"
                )
            return decision_idx, "close_if_live_executable"

        # Next valid bar (skip invalid/halted bars so the action is on a tradable bar).
        idx = decision_idx + 1
        valid = bars.is_valid
        if valid is not None:
            while idx < n and not bool(valid[idx]):
                idx += 1
        idx = min(idx, n - 1)
        return idx, "next_open"
