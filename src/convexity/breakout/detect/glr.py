"""RobustJointGLR — robust joint mean/variance change statistics (BREAKOUT §7 detect/glr).

A generalized-likelihood-ratio statistic for a joint change in mean and variance of the standardized
returns, computed causally: the statistic at t uses ``x[:t+1]`` ONLY (no future bars, Inv B). For
each window W it compares the most-recent W-bar segment ending at t against the preceding W-bar
reference segment under a robust (Student-t / self-normalized) loss, and the cross-window statistic
is a predeclared MAX rule (BREAKOUT §detect/glr: "combined across windows by a predeclared
max/mixture rule").
"""

from __future__ import annotations

import numpy as np

from convexity.breakout.types import StdSeries


class RobustJointGLR:
    """Robust joint mean/variance GLR (BREAKOUT §7). One public method, ``statistic``."""

    def statistic(self, s: StdSeries, windows: tuple[int, ...]) -> np.ndarray:
        """Return one evidence array per bar: max over windows of the robust joint GLR (uses x[:t+1]).

        For a window W, with reference segment ``x[t-2W .. t-W-1]`` and test segment
        ``x[t-W .. t-1]`` (both strictly before t so the statistic is comparable to the cross at t),
        the statistic is the self-normalized squared mean-shift plus a log-variance-ratio term.
        """
        x = s.x
        n = x.size
        out = np.zeros(n)
        for w in windows:
            stat_w = self._single_window(x, w)
            out = np.maximum(out, stat_w)
        # squash to a bounded evidence in [0, 1].
        return np.clip(np.tanh(out / 4.0), 0.0, 1.0)

    def _single_window(self, x: np.ndarray, w: int) -> np.ndarray:
        n = x.size
        out = np.zeros(n)
        for t in range(2 * w + 1, n):
            ref = x[t - 2 * w : t - w]
            test = x[t - w : t]  # ends at t-1
            mu_r, mu_t = np.mean(ref), np.mean(test)
            v_r = np.var(ref) + 1e-9
            v_t = np.var(test) + 1e-9
            # self-normalized mean-shift (robust: scaled by pooled variance).
            pooled = 0.5 * (v_r + v_t)
            mean_term = w * (mu_t - mu_r) ** 2 / (2.0 * pooled)
            # variance-change term: a symmetric log-ratio (Kullback-style).
            var_term = 0.5 * w * (v_t / v_r - 1.0 - np.log(v_t / v_r))
            out[t] = float(mean_term + var_term)
        return out
