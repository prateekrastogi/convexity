"""JumpDeterioration — jump-against / jump-with diagnostics (TREND §9 jump).

Treated as a SPARSE event detector, not an all-hours volatility alarm (§21). DEFAULT branch
(ARCHITECTURE resolution #7, CLAUDE.md-compatible for crypto): robust standardized-return exceedance
with empirical calibration — flag a bar whose |standardized log return| exceeds ``jump_exceedance_z``
relative to a robust LOCAL scale (MAD of the trailing window, excluding the current bar so the scale
is past-only).

The bipower/Lee-Mykland branch (spec §9: "use bipower/Lee-Mykland-style jump logic when the data
quality supports it") is config-gated behind ``cfg.defaults.jump_use_bipower`` and is taken ONLY when
the causal vol quality flag reports clean OHLC (``vol.quality_flag == "ok"``); otherwise it degrades
to the robust-exceedance default even when the flag is on (so a poor-quality bar can never be priced
by the jump-sensitive bipower scale). The Lee-Mykland statistic compares ``|r_i|`` to a local
bipower-variation scale (jump-robust: products of ADJACENT |returns| in the trailing window, excluding
the current bar so the scale is past-only) and flags when it exceeds the Gumbel-law critical level for
the window size. This stays the documented CLAUDE.md exception — robust-exceedance remains the default.

A jump AGAINST the position is a candidate hard exit (D5). A jump WITH the position widens the trail
and does not exit by itself (§10). Roll-window stitch and split artifacts are suppressed upstream
(data guards), so this never reads a corporate-action discontinuity as a jump.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import Bars, Side
from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.data.rolls import FuturesRollGuard
from convexity.trend.types import VolState


class JumpDeterioration:
    """Robust standardized-exceedance jump detector (§9, default branch)."""

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def update(
        self,
        bars: Bars,
        idx: int,
        side: Side,
        vol: VolState,
        cfg: ResolvedConfig,
    ) -> tuple[bool, bool]:
        """Return ``(jump_against, jump_with)`` at ``idx`` (§9). Causal, robust-exceedance default."""
        if idx < 1:
            return False, False
        # Suppress jumps inside a flagged roll window (stitch artifact, §8).
        if FuturesRollGuard.is_roll_protected(bars, idx):
            return False, False
        # Suppress on invalid/halted bars (corporate-action/halt artifacts, §8).
        if bars.is_valid is not None and not bool(bars.is_valid[idx]):
            return False, False

        c = bars.c
        if c[idx] <= 0 or c[idx - 1] <= 0:
            return False, False
        r = float(np.log(c[idx] / c[idx - 1]))

        # Trailing log-returns EXCLUDING the current bar (past-only scale, §9 causal).
        lo = max(1, idx - self.window)
        with np.errstate(divide="ignore", invalid="ignore"):
            window_ret = np.diff(np.log(np.where(c[lo:idx] > 0, c[lo:idx], np.nan)))
        window_ret = window_ret[np.isfinite(window_ret)]
        if window_ret.size < 5:
            return False, False

        # §9 branch select: the bipower/Lee-Mykland statistic is config-gated AND only taken when the
        # causal OHLC/vol quality is clean; otherwise the robust-exceedance default (CLAUDE.md).
        use_bipower = cfg.defaults.jump_use_bipower and vol.quality_flag == "ok"
        if use_bipower:
            exceeded, signed = self._lee_mykland(r, window_ret, vol)
        else:
            exceeded, signed = self._robust_exceedance(r, window_ret, vol, cfg)
        if not exceeded:
            return False, False

        # Sign relative to the position: side*r > 0 is favorable (jump_with), else against.
        signed *= float(side)
        return (bool(signed < 0.0), bool(signed > 0.0))

    @staticmethod
    def _robust_exceedance(
        r: float, window_ret: np.ndarray, vol: VolState, cfg: ResolvedConfig
    ) -> tuple[bool, float]:
        """Default branch: |standardized return| > z vs a MAD scale (jump_exceedance_z)."""
        mad = float(np.median(np.abs(window_ret - np.median(window_ret))))
        scale = 1.4826 * mad if mad > 0 else max(vol.sigma_log, 1e-8)
        z = r / max(scale, 1e-12)
        return (abs(z) > cfg.defaults.jump_exceedance_z, z)

    @staticmethod
    def _lee_mykland(
        r: float, window_ret: np.ndarray, vol: VolState
    ) -> tuple[bool, float]:
        """Lee-Mykland (2008) branch: |r_i| / sqrt(local bipower variation), flagged at the Gumbel-law
        critical level for the window. Bipower variation is jump-robust — it uses products of ADJACENT
        absolute returns, so a single past jump cannot inflate the scale and mask a true jump."""
        absr = np.abs(window_ret)
        # bipower variation per-bar instantaneous variance: (pi/2) * mean(|r_t| |r_{t-1}|).
        bv = (np.pi / 2.0) * float(np.mean(absr[1:] * absr[:-1])) if absr.size >= 2 else 0.0
        sigma = np.sqrt(bv) if bv > 0 else max(vol.sigma_log, 1e-8)
        ell = r / max(sigma, 1e-12)  # signed Lee-Mykland statistic; |ell| ~ |N(0,1)| under no jump
        # Gumbel limiting law for max|ell| over the window of size n (Lee-Mykland Thm 1):
        #   (max|ell| - C_n) / S_n -> Gumbel; flag when |ell| exceeds the C_n + beta*S_n level.
        n = float(window_ret.size)
        c_const = np.sqrt(2.0 * np.log(n))
        c_n = c_const - (np.log(np.pi) + np.log(np.log(n))) / (2.0 * c_const)
        s_n = 1.0 / c_const
        # beta* = -log(-log(1-alpha)); alpha=0.01 single-step significance under the Gumbel law.
        beta_star = -np.log(-np.log(1.0 - 0.01))
        threshold = c_n + beta_star * s_n
        return (abs(ell) > threshold, ell)
