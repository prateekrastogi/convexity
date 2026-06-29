"""EntryLabeler — triple-barrier + MFE/MAE labels, EVALUATION ONLY (BREAKOUT §7 / Invariant C).

FUTURE DATA lives ONLY here (Invariant C / §12.6). This module is imported by ``validate/`` code
only; ``detect/``, ``features/``, ``standardize/``, ``online/`` and ``pipeline.py`` must NEVER import
it (enforced by the AST label-firewall test, §15). Labels start from ``entry_idx`` (the first
actionable bar, Invariant G) and the stop barrier is ``signal.invalidation`` / ``signal.stop_distance``
known at entry time. MFE, MAE, first-touch of stop/profit barriers, time-to-continuation, and the
false-breakout flag are validation outputs only.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, CostModel, EntrySignal

from convexity.breakout.types import EntryLabel


class EntryLabeler:
    """Future-looking evaluation labeler (BREAKOUT §7). One public method, ``label``."""

    def label(
        self,
        bars: Bars,
        signals: list[EntrySignal],
        cost: CostModel,
        horizons: tuple[int, ...],
        r_targets: tuple[float, ...],
    ) -> list[EntryLabel]:
        """Create future labels for evaluation only (§7). Uses ``entry_idx`` as the start (Inv G)."""
        out: list[EntryLabel] = []
        n = len(bars)
        c = bars.c
        h = bars.h
        low = bars.l
        max_h = max(horizons)
        cost_bps = cost.spread_bps + cost.slippage_bps + cost.fees_bps
        r_plus = r_targets[0] if r_targets else 1.0
        r_plus2 = r_targets[1] if len(r_targets) > 1 else 2.0

        for sig in signals:
            e = sig.entry_idx
            if e >= n:
                continue
            entry = c[e]
            d = sig.direction
            stop_dist = sig.stop_distance
            if stop_dist <= 0 or d == 0:
                continue
            end = min(e + max_h, n)
            fwd_h = h[e:end]
            fwd_l = low[e:end]

            # excursions in R (stop-distance) units, signed by direction.
            if d > 0:
                mfe = (np.max(fwd_h) - entry) / stop_dist
                mae = (entry - np.min(fwd_l)) / stop_dist
                stop_level = entry - stop_dist
                hit_stop = np.where(fwd_l <= stop_level)[0]
                hit_1r = np.where(fwd_h >= entry + r_plus * stop_dist)[0]
                hit_2r = np.where(fwd_h >= entry + r_plus2 * stop_dist)[0]
            else:
                mfe = (entry - np.min(fwd_l)) / stop_dist
                mae = (np.max(fwd_h) - entry) / stop_dist
                stop_level = entry + stop_dist
                hit_stop = np.where(fwd_h >= stop_level)[0]
                hit_1r = np.where(fwd_l <= entry - r_plus * stop_dist)[0]
                hit_2r = np.where(fwd_l <= entry - r_plus2 * stop_dist)[0]

            t_stop = int(hit_stop[0]) if hit_stop.size else None
            t_1r = int(hit_1r[0]) if hit_1r.size else None
            t_2r = int(hit_2r[0]) if hit_2r.size else None

            hit_1r_before_stop = t_1r is not None and (t_stop is None or t_1r < t_stop)
            hit_2r_before_stop = t_2r is not None and (t_stop is None or t_2r < t_stop)
            hit_stop_first = t_stop is not None and (t_1r is None or t_stop < t_1r)

            # net forward return to the primary horizon, cost-debited (Inv F identical contract).
            hbar = min(e + horizons[0], n - 1)
            gross_bps = d * (np.log(c[hbar] / entry)) * 1e4
            net_bps = float(gross_bps - 2.0 * cost_bps)  # round-trip cost

            # false breakout: stopped out before reaching +1R within the first horizon window.
            false_break = bool(hit_stop_first and not hit_1r_before_stop)

            out.append(
                EntryLabel(
                    signal_idx=sig.signal_idx,
                    entry_idx=e,
                    horizon=horizons[0],
                    mfe_r=float(mfe),
                    mae_r=float(mae),
                    hit_plus_1r_before_stop=bool(hit_1r_before_stop),
                    hit_plus_2r_before_stop=bool(hit_2r_before_stop),
                    hit_stop_first=bool(hit_stop_first),
                    time_to_plus_1r=t_1r,
                    false_breakout=false_break,
                    net_return_to_horizon_bps=net_bps,
                )
            )
        return out
