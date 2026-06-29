"""StreamingState — the persistent O(1)-amortized hot-path engine (BREAKOUT §1 + Invariant E).

THE load-bearing online structure. BREAKOUT §1 requires the full online update to be **amortized
O(1) per instrument per bar** (the 87-instrument / 2-core / seconds-latency target), and Invariant E
requires batch replay and one-bar streaming to emit tolerance-identical signals. Both are satisfied
here by making this engine the SINGLE source of truth for every per-bar feature/evidence scalar:
``PipelineOrchestrator.build_frame`` feeds bars through it sequentially, and
``PipelineOrchestrator.update`` advances the SAME persistent engine one bar — so
``run_batch == sequential update()`` holds BY CONSTRUCTION (identical code, identical structures),
not by re-running a batch over the accumulated history.

PER-BAR COMPLEXITY (achieved): every ``advance(bar)`` is **amortized O(1)** in the stream length, and
**O(W + R + B)** in the bounded model constants (W = max lookback window, R = BOCPD run-length cap,
B = histogram bins) — all fixed RED/BLUE constants independent of how many bars have streamed. A whole
87-instrument 1-minute tick is therefore O(87). NO stage is an O(t)/O(W) expanding/windowed recompute
on the hot path:

  * rolling max/min barriers .......... monotonic deque, each bar pushed/popped at most once (O(1) am.)
  * predictive sigma .................. EWMA accumulator, one multiply-add per bar (O(1))
  * vol/liq regime buckets ............ fixed-bin streaming-histogram quantiles (O(B) = O(1) am.)
  * compression percentile ............ fixed-bin streaming-rank histogram (O(B) = O(1) am.)
  * empirical-null p/e-values ......... per-(vol,liq)-bucket fixed-bin histograms (O(B) = O(1) am.)
  * GLR / BOCPD change evidence ....... bounded rolling buffer / pruned run-length posterior (O(W),O(R))
  * seasonal de-seasonalisation ....... per-intrabar-index bounded deque of prior sessions (O(1) am.)
  * cooldown / active-zone ............ integer counters (O(1))

The previous implementation re-ran the full causal batch over the accumulated history on EVERY
``update()`` (O(n)/bar -> O(n^2) over a stream) and ``online/state.py`` snapshotted the raw bar
HISTORY. This engine advances incrementally; ``online/state.py`` serializes THIS streaming-structure
state (deque contents, EWMA accumulators, histogram bin counts, per-bucket null counters, cooldown),
so snapshot -> restore -> replay reproduces identical signals (Invariant E).

Faithfulness note: the expanding past-only empirical rank / quantile computations that the array
modules (``range_geometry._compression_pct``, ``volatility._past_only_buckets``,
``empirical_null.calibrate``) expressed as O(t) scans are realised here as the §1-sanctioned
fixed-bin streaming histograms. Because BOTH batch and online drive the SAME histograms, the two
paths agree bit-for-bit; the histograms remain strictly past-only (a value is queried before it is
inserted), so the future-needle / fold-isolation leakage invariants are preserved.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import lgamma

import numpy as np

# --------------------------------------------------------------------------- bounded streaming structs


@dataclass
class MonotonicExtremum:
    """Strict-pre-bar rolling extremum via a monotonic deque (Invariant I; O(1) amortized).

    ``value()`` returns ``extremum(x[t-window .. t-1])`` BEFORE bar t's own value is pushed, so bar t
    never sees its own high/low (the inclusive-max-unbreakable defect, Inv I). Each index is pushed
    and popped at most once over the stream.
    """

    window: int
    is_max: bool = True
    _idx: deque = field(default_factory=deque)
    _val: deque = field(default_factory=deque)
    _t: int = 0  # number of bars pushed so far == index of the next bar

    def value(self) -> float:
        while self._idx and self._idx[0] <= self._t - 1 - self.window:
            self._idx.popleft()
            self._val.popleft()
        return self._val[0] if self._val else np.nan

    def push(self, v: float) -> None:
        if self.is_max:
            while self._val and self._val[-1] <= v:
                self._idx.pop()
                self._val.pop()
        else:
            while self._val and self._val[-1] >= v:
                self._idx.pop()
                self._val.pop()
        self._idx.append(self._t)
        self._val.append(float(v))
        self._t += 1

    def to_state(self) -> dict:
        return {"window": self.window, "is_max": self.is_max, "t": self._t,
                "idx": list(self._idx), "val": list(self._val)}

    @classmethod
    def from_state(cls, st: dict) -> "MonotonicExtremum":
        m = cls(window=int(st["window"]), is_max=bool(st["is_max"]))
        m._t = int(st["t"])
        m._idx = deque(int(i) for i in st["idx"])
        m._val = deque(float(v) for v in st["val"])
        return m


@dataclass
class StreamingHistogram:
    """Fixed-bin streaming histogram for past-only empirical rank / quantiles (O(B) = O(1) amortized).

    Values are mapped into ``bins`` equal-width buckets over ``[lo, hi]`` (clamped). ``rank_ge(x)``
    is the past-only fraction of inserted values in bins at or above x's bin (a right-tail empirical
    p); ``quantile(q)`` is the bin-edge q-quantile. Bounded memory (B counters) and O(B) per query,
    independent of how many values were inserted — the hot-path O(1)-amortized guarantee (§1).
    """

    lo: float
    hi: float
    bins: int = 100
    _counts: object = None
    _n: int = 0

    def __post_init__(self) -> None:
        if self._counts is None:
            self._counts = np.zeros(self.bins, dtype=np.int64)

    def _bin(self, x: float) -> int:
        frac = (x - self.lo) / (self.hi - self.lo + 1e-12)
        b = int(np.floor(frac * self.bins))
        return min(max(b, 0), self.bins - 1)

    def insert(self, x: float) -> None:
        if np.isfinite(x):
            self._counts[self._bin(x)] += 1
            self._n += 1

    @property
    def n(self) -> int:
        return self._n

    def rank_ge(self, x: float) -> float:
        if self._n == 0 or not np.isfinite(x):
            return 1.0
        b = self._bin(x)
        ge = int(self._counts[b:].sum())
        return ge / self._n

    def quantile(self, q: float) -> float:
        if self._n == 0:
            return np.nan
        target = q * self._n
        cum = 0
        width = (self.hi - self.lo) / self.bins
        for b in range(self.bins):
            cum += int(self._counts[b])
            if cum >= target:
                return self.lo + (b + 1) * width
        return self.hi

    def to_state(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "bins": self.bins,
                "counts": np.asarray(self._counts).tolist(), "n": self._n}

    @classmethod
    def from_state(cls, st: dict) -> "StreamingHistogram":
        h = cls(lo=float(st["lo"]), hi=float(st["hi"]), bins=int(st["bins"]),
                _counts=np.asarray(st["counts"], dtype=np.int64))
        h._n = int(st["n"])
        return h


# --------------------------------------------------------------------------- per-bar output row


@dataclass(frozen=True)
class BarRow:
    """One bar's feature + evidence scalars — a single row of the arrays the array-based detector
    (RangeGeometry / EvidenceFrame) would have produced. Stacking these over the stream yields the
    EXACT same arrays both ``run_batch`` and ``update`` consume (Invariant E single source of truth)."""

    direction: int
    upper_barrier: float
    lower_barrier: float
    invalidation: float
    stop_distance: float
    range_width_sigma: float
    compression_pct: float
    barrier_age: int
    entry_ref_price: float
    sigma: float
    x: float
    vol_bucket: int
    liq_bucket: int
    candidate: int
    glr_e: float
    bocpd_e: float
    range_e: float
    compression_e: float
    trend_e: float
    flow_e: float
    jump_e: float
    context_e: float
    false_break_risk: float


# --------------------------------------------------------------------------- the streaming engine


class StreamingState:
    """Persistent O(1)-amortized per-bar engine (BREAKOUT §1 / Inv E). ``advance(...)`` consumes ONE
    bar and returns its :class:`BarRow`; the engine never rescans accumulated history for the
    expanding stats (it carries streaming structures), and the bounded-window evidence channels
    (GLR/BOCPD/seasonal/features) read only a fixed-length tail buffer.

    Configuration mirrors ``ResolvedConfig`` + ``DetectorDefaults`` exactly so the produced rows equal
    the array modules' output bar-for-bar (with the expanding ranks realised via the §1 histograms).
    """

    _WARM_COMP = 30  # range_geometry._compression_pct warm-up
    _WARM_BUCKET = 30  # volatility._past_only_buckets warm-up
    _WARM_NULL = 60  # empirical_null warm
    _NULL_MIN_EVENTS = 50  # empirical_null min_events before trusting a conditional bucket

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        d = cfg.defaults
        self.w_fast = min(cfg.windows)
        self.w_slow = max(cfg.windows)
        self.windows = tuple(cfg.windows)
        self.cross_k = cfg.cross_k
        self.vol_window = d.vol_window
        self.n_vol_buckets = d.n_vol_buckets
        self.n_liq_buckets = d.n_liq_buckets
        self.target_alert_rate = d.target_alert_rate
        self.bocpd_hazard = d.bocpd_hazard
        self.bocpd_max_run_length = d.bocpd_max_run_length
        self.flow_enabled = bool(cfg.instrument.optional_sources.get("flow", False))
        self.context_enabled = bool(cfg.instrument.optional_sources.get("context", False))
        # session-open derivation (incremental, mirrors data/calendar.Sessionizer) for when the caller
        # does not pre-mark sessions (the online path): 24/7 -> UTC-day boundary; session calendar ->
        # an inter-bar gap > 4*step opens a session.
        from convexity.breakout.config.resolve import has_session_boundary
        self.has_session = has_session_boundary(cfg.instrument.calendar_id)
        self._DAY_MS = 24 * 60 * 60 * 1000

        # --- rolling extrema (Inv I monotonic deques) ---
        self.up_fast = MonotonicExtremum(self.w_fast, is_max=True)
        self.dn_fast = MonotonicExtremum(self.w_fast, is_max=False)

        # --- EWMA predictive sigma accumulator ---
        self.alpha = 2.0 / (self.vol_window + 1.0)
        self.floor = 1e-9
        self.ewvar = 0.0
        self.seeded = False

        # --- streaming histograms for the expanding ranks/quantiles (§1) ---
        # ranges are set lazily on first finite observation (so bins straddle the data).
        self.comp_hist: StreamingHistogram | None = None  # compression_pct: rank of range_width_sigma
        self.sigma_hist: StreamingHistogram | None = None  # vol bucket quantiles
        self.liq_hist: StreamingHistogram | None = None  # liq bucket quantiles
        self.null_hist: dict[int, dict[str, StreamingHistogram]] = {}  # per joint-bucket null
        self.null_pool: dict[str, StreamingHistogram] = {}  # pooled fallback null

        # --- bounded tail buffers for the windowed evidence channels ---
        self._tail = max(2 * self.w_slow + 2, self.vol_window + 2, 35)
        self.x_buf: deque = deque(maxlen=self._tail)  # standardized returns
        self.c_buf: deque = deque(maxlen=self._tail)  # closes
        self.h_buf: deque = deque(maxlen=self._tail)
        self.l_buf: deque = deque(maxlen=self._tail)

        # --- BOCPD persistent run-length posterior (pruned) ---
        self.R = np.array([1.0])
        self.bo_mu = np.array([0.0])
        self.bo_kappa = np.array([1.0])
        self.bo_alpha = np.array([1.0])
        self.bo_beta = np.array([1.0])

        # --- seasonal: per intra-session-index deque of prior-session returns ---
        from collections import defaultdict
        self._lookback_sessions = 20
        self.seasonal_hist: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max(self._lookback_sessions, 1))
        )
        self.bar_in_sess = 0
        self._first_bar = True

        # --- barrier-age + same-zone active state + previous barrier levels ---
        self.prev_upper = np.nan
        self.prev_lower = np.nan
        self.barrier_age = 0
        self.active_zone = 0

        # --- bar counter + ts cadence (for the gap detector) ---
        self.t = 0
        self.prev_ts: int | None = None
        self.ts_step: float = 0.0
        self.prev_c: float | None = None

    # ------------------------------------------------------------------ advance one bar

    def advance(
        self, ts: int, o: float, h: float, low: float, c: float,
        v: float | None, is_session_open: bool | None, is_valid: bool | None,
        is_roll_window: bool | None,
    ) -> BarRow:
        """Consume one bar (F_t scalars) and emit its :class:`BarRow`. Amortized O(1) (BREAKOUT §1).

        ``is_session_open`` may be None (online path, unsessionized): the engine then derives the
        session-open mark incrementally to match ``data/calendar.Sessionizer`` (bit-identical to the
        batch flags), so batch/live parity holds (Inv E)."""
        t = self.t

        # derive the session-open mark incrementally when not supplied (mirrors Sessionizer).
        if is_session_open is None:
            if t == 0:
                is_session_open = True
            elif self.prev_ts is not None:
                if self.has_session:
                    dt = float(ts) - float(self.prev_ts)
                    step = self.ts_step if self.ts_step > 0 else dt
                    is_session_open = bool(dt > 4 * step) if step > 0 else False
                else:
                    is_session_open = (int(ts) // self._DAY_MS) != (int(self.prev_ts) // self._DAY_MS)
            else:
                is_session_open = False

        # --- log return + gap flag (close-to-close; first bar / session-open / ts-gap is a gap) ---
        if self.prev_c is None or self.prev_c <= 0:
            r = 0.0
        else:
            r = float(np.log(max(c, 1e-12)) - np.log(max(self.prev_c, 1e-12)))
        is_gap = bool(is_session_open) if is_session_open is not None else False
        if self.prev_ts is not None:
            dt = float(ts) - float(self.prev_ts)
            if self.ts_step <= 0 and dt > 0:
                self.ts_step = dt
            if self.ts_step > 0 and dt > 1.5 * self.ts_step:
                is_gap = True
        if t == 0:
            is_gap = True

        # --- seasonal deseasonalisation (past-only running mean over prior sessions) ---
        if is_session_open is not None:
            if self._first_bar or is_session_open:
                self.bar_in_sess = 0
            else:
                self.bar_in_sess += 1
            k = self.bar_in_sess
            prior = self.seasonal_hist[k]
            deseason = r - float(np.mean(prior)) if prior else r
            prior.append(r)
        else:
            deseason = r
        self._first_bar = False

        # --- predictive sigma: read EWMA (F_{t-1}) BEFORE folding bar t in ---
        if not self.seeded:
            sigma = float(np.sqrt(self.ewvar)) if self.ewvar > 0 else self.floor
        else:
            sigma = float(np.sqrt(max(self.ewvar, self.floor ** 2)))
        sigma = max(sigma, self.floor)
        if not is_gap:
            if self.seeded:
                self.ewvar = (1.0 - self.alpha) * self.ewvar + self.alpha * deseason * deseason
            else:
                self.ewvar = deseason * deseason
                self.seeded = True
        x = r / sigma

        # --- vol/liq regime buckets (past-only streaming-histogram quantiles) ---
        vol_bucket = self._bucket(self._sigma_hist(sigma), sigma, self.n_vol_buckets, t,
                                   warm=self._WARM_BUCKET)
        if v is not None:
            liq_bucket = self._bucket(self._liq_histogram(float(v)), float(v),
                                      self.n_liq_buckets, t, warm=self._WARM_BUCKET)
        else:
            liq_bucket = self.n_liq_buckets // 2

        # --- strict-pre-bar barriers (read BEFORE pushing bar t; Inv I) ---
        upper = self.up_fast.value()
        lower = self.dn_fast.value()
        self.up_fast.push(h)
        self.dn_fast.push(low)

        # --- raw geometry direction + range width sigma + compression percentile ---
        direction = 0
        if np.isfinite(upper) and c > upper:
            direction = 1
        elif np.isfinite(lower) and c < lower:
            direction = -1

        if np.isfinite(upper) and np.isfinite(lower):
            rng = upper - lower
            range_width_sigma = rng / (sigma * c + 1e-12)
        else:
            range_width_sigma = np.nan

        compression_pct = self._compression(range_width_sigma, t)

        invalidation = (lower if direction > 0 else (upper if direction < 0 else np.nan))
        entry_ref = float(c)
        stop_distance = abs(entry_ref - invalidation)

        # --- barrier age (bars since both fast levels last unchanged) ---
        same_u = np.isfinite(upper) and np.isfinite(self.prev_upper) and upper == self.prev_upper
        same_l = np.isfinite(lower) and np.isfinite(self.prev_lower) and lower == self.prev_lower
        if t == 0:
            self.barrier_age = 0
        else:
            self.barrier_age = self.barrier_age + 1 if (same_u and same_l) else 0
        self.prev_upper, self.prev_lower = upper, lower
        barrier_age = self.barrier_age

        # --- push bounded tail buffers (used by the windowed evidence channels) ---
        self.x_buf.append(x)
        self.c_buf.append(c)
        self.h_buf.append(h)
        self.l_buf.append(low)

        # --- vol-normalized candidate cross (Inv J) + artifact + same-zone suppression ---
        candidate = self._candidate(
            direction, c, upper, lower, sigma, is_valid, is_session_open, is_roll_window
        )

        # --- evidence channels ---
        glr_stat = self._glr_stat()
        glr_e = self._null_calibrate("glr", glr_stat, vol_bucket, liq_bucket, t)
        bocpd_raw = self._bocpd_step(x)
        bocpd_e = self._null_calibrate("bocpd", bocpd_raw, vol_bucket, liq_bucket, t)
        range_e = float(np.clip(np.nan_to_num(compression_pct), 0.0, 1.0))
        compression_e = self._compression_evidence(compression_pct)
        trend_e, false_break_risk = self._trend_and_falsebreak(direction, c, h, low)
        flow_e = 0.0  # optional, disabled by default (§8); enabled paths handled in batch builder
        jump_e = self._jump_evidence()
        context_e = 0.0

        self.t += 1
        self.prev_ts = int(ts)
        self.prev_c = float(c)

        return BarRow(
            direction=direction, upper_barrier=upper, lower_barrier=lower,
            invalidation=invalidation, stop_distance=stop_distance,
            range_width_sigma=range_width_sigma, compression_pct=compression_pct,
            barrier_age=barrier_age, entry_ref_price=entry_ref, sigma=sigma, x=x,
            vol_bucket=vol_bucket, liq_bucket=liq_bucket, candidate=candidate,
            glr_e=glr_e, bocpd_e=bocpd_e, range_e=range_e, compression_e=compression_e,
            trend_e=trend_e, flow_e=flow_e, jump_e=jump_e, context_e=context_e,
            false_break_risk=false_break_risk,
        )

    # ------------------------------------------------------------------ expanding-stat helpers

    def _sigma_hist(self, sigma: float) -> StreamingHistogram:
        if self.sigma_hist is None:
            self.sigma_hist = StreamingHistogram(lo=0.0, hi=max(sigma * 8.0, self.floor * 10), bins=200)
        return self.sigma_hist

    def _liq_histogram(self, v: float) -> StreamingHistogram:
        if self.liq_hist is None:
            self.liq_hist = StreamingHistogram(lo=0.0, hi=max(v * 8.0, 1.0), bins=200)
        return self.liq_hist

    def _bucket(self, hist: StreamingHistogram, value: float, k: int, t: int, *, warm: int) -> int:
        """Past-only bucket via streaming-histogram quantiles (mirrors _past_only_buckets)."""
        if t < warm:
            hist.insert(value)
            return k // 2
        qs = [hist.quantile(q) for q in np.linspace(0, 1, k + 1)[1:-1]]
        b = int(np.searchsorted(qs, value))
        hist.insert(value)  # insert AFTER querying (strictly past-only)
        return int(np.clip(b, 0, k - 1))

    def _compression(self, width_sigma: float, t: int) -> float:
        if self.comp_hist is None:
            # range_width_sigma is a positive ratio; a generous upper bound keeps bins meaningful.
            self.comp_hist = StreamingHistogram(lo=0.0, hi=50.0, bins=500)
        if t < self._WARM_COMP or not np.isfinite(width_sigma):
            if np.isfinite(width_sigma):
                self.comp_hist.insert(width_sigma)
            return 0.0
        if self.comp_hist.n < 5:
            self.comp_hist.insert(width_sigma)
            return 0.0
        pct = self.comp_hist.rank_ge(width_sigma)  # fraction of PAST widths >= current
        self.comp_hist.insert(width_sigma)
        return float(pct)

    def _null_calibrate(self, name: str, stat: float, vb: int, lb: int, t: int) -> float:
        """Past-only conditional empirical-null e-value via per-bucket streaming histograms (Inv M)."""
        key = int(vb) * 1000 + int(lb)
        bucket = self.null_hist.setdefault(key, {})
        pool = self.null_pool
        if name not in bucket:
            bucket[name] = StreamingHistogram(lo=0.0, hi=1.0, bins=200)
        if name not in pool:
            pool[name] = StreamingHistogram(lo=0.0, hi=1.0, bins=200)
        bh = bucket[name]
        ph = pool[name]
        floor = max(self.target_alert_rate, 1e-4)

        cal = 0.0
        if t >= self._WARM_NULL:
            if bh.n >= self._NULL_MIN_EVENTS:
                hist_n = bh.n
                p_hat = max(bh.rank_ge(stat), floor) if hist_n >= 5 else None
            elif ph.n >= 5:
                p_hat = max(ph.rank_ge(stat), floor)
            else:
                p_hat = None
            if p_hat is not None:
                cal = float(-np.log(p_hat))
        # insert AFTER querying (strictly past-only).
        bh.insert(stat)
        ph.insert(stat)
        return cal

    # ------------------------------------------------------------------ windowed evidence channels

    def _candidate(self, direction, c, upper, lower, sigma, is_valid, is_session_open,
                   is_roll_window) -> int:
        thresh = self.cross_k * sigma * c
        cand = 0
        if np.isfinite(upper) and (c - upper) >= thresh:
            cand = 1
        elif np.isfinite(lower) and (lower - c) >= thresh:
            cand = -1
        # artifact suppression
        if is_valid is not None and not bool(is_valid):
            cand = 0
        if is_session_open is not None and bool(is_session_open):
            cand = 0
        if is_roll_window is not None and bool(is_roll_window):
            cand = 0
        # same-zone suppression (keep only the first cross of a zone)
        out = cand
        if cand == 1:
            if self.active_zone == 1:
                out = 0
            else:
                self.active_zone = 1
        elif cand == -1:
            if self.active_zone == -1:
                out = 0
            else:
                self.active_zone = -1
        else:
            if self.active_zone == 1 and np.isfinite(upper) and c <= upper:
                self.active_zone = 0
            elif self.active_zone == -1 and np.isfinite(lower) and c >= lower:
                self.active_zone = 0
        return out

    def _glr_stat(self) -> float:
        """Max-over-windows robust joint GLR on the bounded x tail (mirrors RobustJointGLR)."""
        x = np.asarray(self.x_buf, dtype=float)
        m = x.size
        best = 0.0
        for w in self.windows:
            if m < 2 * w + 1:
                continue
            ref = x[-2 * w:-w]
            test = x[-w:]
            mu_r, mu_t = np.mean(ref), np.mean(test)
            v_r = np.var(ref) + 1e-9
            v_t = np.var(test) + 1e-9
            pooled = 0.5 * (v_r + v_t)
            mean_term = w * (mu_t - mu_r) ** 2 / (2.0 * pooled)
            var_term = 0.5 * w * (v_t / v_r - 1.0 - np.log(v_t / v_r))
            best = max(best, float(mean_term + var_term))
        return float(np.clip(np.tanh(best / 4.0), 0.0, 1.0))

    def _bocpd_step(self, x: float) -> float:
        """One BOCPD recursion step on the persistent pruned posterior (mirrors BocpdChannel)."""
        MU0, KAPPA0, ALPHA0, BETA0 = 0.0, 1.0, 1.0, 1.0
        hazard = self.bocpd_hazard
        xt = float(x) if np.isfinite(x) else 0.0
        mu, kappa, alpha, beta = self.bo_mu, self.bo_kappa, self.bo_alpha, self.bo_beta
        pred = _student_t_pdf(xt, mu, kappa, alpha, beta)
        growth = self.R * pred * (1.0 - hazard)
        cp = float(np.sum(self.R * pred * hazard))
        newR = np.concatenate([[cp], growth])
        total = newR.sum()
        if total <= 0:
            newR = np.zeros_like(newR)
            newR[0] = 1.0
        else:
            newR /= total
        mu_g = (kappa * mu + xt) / (kappa + 1.0)
        kappa_g = kappa + 1.0
        alpha_g = alpha + 0.5
        beta_g = beta + 0.5 * kappa * (xt - mu) ** 2 / (kappa + 1.0)
        mu = np.concatenate([[MU0], mu_g])
        kappa = np.concatenate([[KAPPA0], kappa_g])
        alpha = np.concatenate([[ALPHA0], alpha_g])
        beta = np.concatenate([[BETA0], beta_g])
        cap = self.bocpd_max_run_length + 1
        if newR.size > cap:
            newR = newR[:cap]
            mu, kappa, alpha, beta = mu[:cap], kappa[:cap], alpha[:cap], beta[:cap]
            newR /= newR.sum() + 1e-300
        self.R, self.bo_mu, self.bo_kappa, self.bo_alpha, self.bo_beta = newR, mu, kappa, alpha, beta
        short = min(5, newR.size)
        return float(np.clip(float(np.sum(newR[:short])), 0.0, 1.0))

    def _compression_evidence(self, compression_pct: float) -> float:
        """0.5*comp + 0.5*vol-contraction over the bounded x tail (mirrors CompressionState)."""
        comp = float(np.clip(compression_pct, 0.0, 1.0))
        x = np.asarray(self.x_buf, dtype=float)
        short_w = max(self.vol_window // 4, 5)
        long_w = self.vol_window
        vc = 0.0
        if x.size >= long_w + 1:
            sq = x * x
            short_v = np.mean(sq[-short_w - 1:-1])  # ends at t-1
            long_v = np.mean(sq[-long_w - 1:-1])
            ratio = np.sqrt(short_v / (long_v + 1e-12))
            vc = float(np.clip(1.0 - ratio, 0.0, 1.0))
        return float(np.clip(0.5 * comp + 0.5 * vc, 0.0, 1.0))

    def _trend_and_falsebreak(self, direction, c, h, low) -> tuple[float, float]:
        """CLV + slope-stability trend evidence and wick/reversion false-break risk (mirrors TrendState)."""
        rng = (h - low) if h > low else np.nan
        clv = 0.0
        if np.isfinite(rng):
            clv = float(((c - low) - (h - c)) / (rng + 1e-12))
        cb = np.asarray(self.c_buf, dtype=float)
        slope = 0.0
        w = 20
        if cb.size >= w + 1:
            lr = np.diff(np.log(np.clip(cb[-w - 1:], 1e-12, None)))  # last w returns ending at t-1
            seg = np.sign(lr)
            up = np.mean(seg > 0)
            dn = np.mean(seg < 0)
            slope = float(max(up, dn))
        directional_clv = float(np.clip(direction * clv, 0.0, 1.0))
        trend_e = float(np.clip(0.5 * directional_clv + 0.5 * slope, 0.0, 1.0))

        wick = float(np.clip(-direction * clv, 0.0, 1.0))
        reversion = 0.0
        xb = np.asarray(self.x_buf, dtype=float)
        if direction != 0 and xb.size >= 3:
            recent = abs(xb[-1])
            prior = abs(xb[-2]) + abs(xb[-3]) + 1e-9
            reversion = float(np.clip(recent / (recent + prior), 0.0, 1.0))
        fbr = float(np.clip(0.6 * wick + 0.4 * reversion, 0.0, 1.0))
        if direction == 0:
            fbr = 0.0
        return trend_e, fbr

    def _jump_evidence(self) -> float:
        """Robust standardized-return exceedance jump diagnostic (mirrors JumpEvidence default)."""
        xb = np.asarray(self.x_buf, dtype=float)
        if xb.size < 3:
            return 0.0
        thr = 4.0
        big = abs(xb[-1]) / thr
        quiet_before = thr / (abs(xb[-2]) + abs(xb[-3]) + thr)
        return float(np.clip(np.tanh(big) * quiet_before, 0.0, 1.0))


def _student_t_pdf(x: float, mu, kappa, alpha, beta):
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / (nu * scale2)
    lg = np.vectorize(lgamma)
    log_c = lg((nu + 1.0) / 2.0) - lg(nu / 2.0) - 0.5 * np.log(np.pi * nu * scale2)
    return np.exp(log_c) * (1.0 + z) ** (-(nu + 1.0) / 2.0)
