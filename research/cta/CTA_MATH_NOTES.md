# CTA Stage-A Math Notes: Unified Cross-Axis Documentation

**Program:** convexity equity-CTA research (worktree `/home/speculation/convexity-cta`, branch
`equity-cta`). **Scope:** this document is the single point of entry for the six Stage-A axis
research modules built in parallel — Axis A (signal extraction), Axis B (entry/exit control),
Axis C (regime modeling), Axis D (volatility forecasting), Axis E (leverage policy), Axis F
(portfolio construction). It (1) reproduces each axis's own frozen mathematical derivation and
validation record verbatim, with cross-axis notational clashes called out and resolved where they
would otherwise cause confusion; (2) states precisely how the six axes' code interfaces with the
other five so a later Integration/Sweep-phase agent can wire them together without re-deriving
anything; (3) declares the frozen Stage-A exploratory grid — the exact, final `N_cells` used as
the multiplicity denominator for Stage B's DSR/PBO correction.

**How to read this document.** Sections 1–6 are the six axes' own math, each ending with a "module
reference" subsection giving exact function signatures, injection-test results, and packages
installed — this content is authoritative and, other than heading-level normalization and the
explicit notation-clash callouts inserted inline, is reproduced as originally authored and
committed by each axis's own agent. Section 7 is new: it is the cross-axis notation
reconciliation table. Section 8 is new: it states the end-to-end pipeline architecture (which
axis's output feeds which axis's input) and lists every interface mismatch found, with the exact
fix required in each affected module. Section 9 is new: it declares the frozen Stage-A grid design
and the final `N_cells`. Section 10 records the test-suite run and commit/push confirming this
document and every axis module are on `equity-cta`.

---
## Axis A — Signal Extraction: Causal Filtering for Trend Estimation

### A.0 The problem with the incumbent construction

The exploratory CTA scans (`_classic_cta_universe_scan.py`) build every signal from a fixed dual-SMA
crossover: `position = sign(SMA_fast − SMA_slow)` gated by a 200-day regime filter, swept over a
small discrete grid of `(fast, slow)` pairs `{(5,20), (10,20), (10,50), (20,50), (20,100),
(50,200)}` and reported per-cell. Established Fact (5) is the honest verdict on this construction:
across the 93-name universe, MA(50,200) wins 3 of 4 scheme/direction cuts, but MA(10,50) wins the
fourth — an inconsistency, not noise, because a fixed-window SMA crossover is a single point
estimate of "the" trend speed, and different regimes/subsamples genuinely have different dominant
speeds. Treating "pick the best window" as the modeling problem is mis-specified: the right
question is not *which* speed is correct, but how to *combine* a spectrum of speeds so that no
single (possibly wrong, possibly regime-dependent) choice can dominate the result. This axis
replaces the single-timeframe SMA/argmax construction with (1) a family of causal filters that are
each individually better-founded than a raw SMA, and (2) an explicit, IC/breadth-motivated
combination rule across them — never an argmax.

Throughout, `p_t` denotes the log price at close of trading day `t`, and every construction is
applied to the point-in-time, source=='duka' daily close series (`daily_close_series_duka`
convention, i.e. last real trade of each calendar day) already established in `i1_pipeline.py` and
reused verbatim by the CTA universe scan.

---

### A.1 The EWMA continuum and the exact crossover ⇔ momentum identity

#### A.1.1 EWMA as a causal filter

An exponentially-weighted moving average with decay `λ ∈ (0,1)` is the causal linear filter

  E_λ[x]_t = (1−λ) Σ_{k=0}^{∞} λ^k x_{t−k}  ,  equivalently  E_t = λ E_{t−1} + (1−λ) x_t.

Parameterizing by **halflife** `h` (the lag at which the weight has decayed to one half) gives
`λ = 2^{−1/h}`, which is exactly pandas' `.ewm(halflife=h, adjust=False)` convention used
throughout `signal_extraction.py` (`halflife_to_decay`). The **center of mass** (mean age) of this
kernel, `c = λ/(1−λ)`, is exactly pandas' `com` parameter — this quantity reappears below as the
literal coefficient connecting a price-EWMA crossover to return-EWMA momentum, not as a
coincidence of notation but as the same mathematical object viewed two ways.

An **SMA** is a crude, equal-weight, hard-truncated special case of this family (`λ ≈ 1 −
2/(window+1)` approximately matches an EWMA's variance-reduction to an SMA of that window, but the
SMA's box-car kernel has a discontinuous cutoff at the window edge, producing a "phantom" jump in
the signal every time an old extreme observation drops out of the window — an artifact an EWMA
does not have). This is one first-order reason to prefer EWMAs to SMAs as the atomic building
block, independent of anything below.

#### A.1.2 The exact identity: a price-EWMA crossover is a weighted difference of two return-EWMA momentum signals

This is the rigorous version of the applied intuition in Levine & Pedersen (2016), "Which Trend Is
Your Friend?" (Financial Analysts Journal), that a moving-average crossover and a
trailing-return-based momentum signal are close cousins. Rather than restate their empirical
correlation findings (which are calibrated to their specific universe/sample and would be
mis-cited if reproduced here as if load-bearing), the exact algebraic identity connecting the two
representations is derived from first principles below — a stronger and sample-independent
statement.

**Derivation.** Write the EWMA recursion as `E_t = λE_{t−1} + (1−λ)p_t`. Subtract `p_t` from both
sides and telescope:

  E_t − p_t = λ(E_{t−1} − p_t) = λ(E_{t−1} − p_{t−1}) − λ r_t ,  r_t := p_t − p_{t−1}.

So the "EWMA gap" `e_t := E_t − p_t` obeys its own first-order recursion driven by the return
series: `e_t = λ e_{t−1} − λ r_t`. Expanding this recursion out to `t=0` (with the natural seed
`E_0 = p_0 ⇒ e_0 = 0`, matching pandas' `adjust=False` behavior on a series with no leading NaNs):

  e_t = −λ Σ_{j=0}^{t−1} λ^j r_{t−j} = −(λ/(1−λ)) · (1−λ) Σ_{j=0}^{t−1} λ^j r_{t−j} = −c · M_λ,t ,

where `M_λ,t := (1−λ) Σ_{j=0}^{t−1} λ^j r_{t−j}` is precisely the single-speed EWMA-of-RETURNS
momentum signal (Moskowitz–Ooi–Pedersen / Baltas–Kosowski style trailing "smoothed average past
return"), seeded at `M_λ,0 = 0` (i.e. `r_0 := 0`), and `c = λ/(1−λ)` is the center of mass defined
above. Hence

  **E_λ[p]_t = p_t − c(λ)·M_λ,t .**            (★)

Now take the crossover of two speeds `λ_f < λ_s` (fast/slow, using the `λ` convention where a
*larger* `λ` is a *slower*/longer-memory filter):

  D_t := E_{λ_f}[p]_t − E_{λ_s}[p]_t = [p_t − c_f M_{f,t}] − [p_t − c_s M_{s,t}]
       = **c_s · M_{s,t} − c_f · M_{f,t} .**       (★★)

This is exact — the price cancels — and it is the precise algebraic content behind "a dual-EWMA
crossover is a momentum signal in disguise": **a crossover of two price-EWMAs is, term for term, a
center-of-mass-weighted difference of two single-speed trailing-return momentum signals.** The
practical payoff is not merely conceptual: `M_{f,t}` and `M_{s,t}` are directly interpretable,
unit-comparable momentum scores in *return* units (not the joint, speed-dependent price-level
units of the raw crossover `D_t`), so they slot directly into the same combination machinery
(Section A.4) as every other component signal in this module, whereas the raw price-level
crossover does not (its scale is a function of both speeds jointly and is not comparable across
different `(λ_f, λ_s)` pairs without extra normalization). **This is why the final deliverable
(Section A.5) builds its EWMA leg from single-speed return-EWMA momentum signals `M_λ` directly,
rather than from crossovers of price-EWMAs** — (★★) shows this loses no generality (any crossover
is recoverable as a particular linear combination of two such momentum legs) while gaining
interpretability and uniform normalizability across horizons.

`signal_extraction.py`'s `ewma_price_crossover` and `crossover_as_return_ewma_difference`
implement the two sides of (★★) independently and are verified to agree to float precision on real
SPY/QQQ/MSFT/JPM/COST data (`validate_signal_extraction.py`, Part 1) — this is a nontrivial check,
because it requires the *finite-sample boundary seeding* on both sides of the identity to match
exactly (not merely asymptotically as history grows): naively seeding the return-EWMA at its own
first observation (pandas' default `ignore_na` behavior, `M_{λ,1}=r_1`) rather than at `M_{λ,0}=0`
introduces a boundary term that decays geometrically at rate `λ^t` but is not exactly zero at
finite `t` (confirmed empirically: up to ~0.25 log-price-units of discrepancy on early-sample SPY
data before correcting the seeding) — `ewma_return_momentum`'s `.diff().fillna(0.0)` seeding
convention is what makes (★★) exact rather than asymptotic.

#### A.1.3 Multi-speed combination is not a new primitive

Because (★★) writes *any* crossover as a linear combination of single-speed momentum legs, a bank
of crossovers spanning many `(λ_f,λ_s)` pairs is itself just a (larger, redundant) linear
combination of the same underlying `{M_λ}` basis. There is therefore no loss of generality — and a
real gain in estimability, since the `{M_λ}` basis is far smaller than the space of all crossover
pairs built from it — in building the multi-horizon ensemble directly on a handful of
well-separated `M_λ` speeds and combining THOSE (Section A.4), rather than separately combining a
larger, mutually-redundant set of crossovers.

---

### A.2 Local-linear-trend Kalman filter, and its exact equivalence to Holt's linear smoothing

#### A.2.1 The state-space model

  level_t = level_{t−1} + slope_{t−1} + w1_t ,  w1 ~ N(0, q_level)
  slope_t = slope_{t−1}               + w2_t ,  w2 ~ N(0, q_slope)
  y_t     = level_t                   + v_t  ,  v  ~ N(0, r)

i.e. transition `F=[[1,1],[0,1]]`, process noise `Q=diag(q_level,q_slope)`, observation `H=[1,0]`.
This is the textbook "local linear trend" / "integrated random walk plus noise" model (Harvey
(1989), *Forecasting, Structural Time Series Models and the Kalman Filter*, §2.3, §3.2). The
standard Kalman **filter** (not smoother — critical for causality, see below) recursion is:

  Predict:  x̂_{t|t−1} = F x̂_{t−1|t−1},  P_{t|t−1} = F P_{t−1|t−1} F' + Q
  Update:   K_t = P_{t|t−1} H' (H P_{t|t−1} H' + r)^{−1},  x̂_{t|t} = x̂_{t|t−1} + K_t(y_t − Hx̂_{t|t−1}),
            P_{t|t} = (I − K_t H) P_{t|t−1}

The filtered estimate `x̂_{t|t}` uses `y_1,…,y_t` only — it is causal by construction, unlike the
Kalman **smoother** (`x̂_{t|T}` for `T>t`) which is a common, easy-to-reach-for mistake in a
backtest (a smoother uses the entire sample, future included, and is a source of subtle lookahead
if applied naively). `signal_extraction.py`'s `kalman_local_linear_trend` runs the filter, never
the smoother.

#### A.2.2 Steady-state gain

As `t→∞`, `P_{t|t−1}` converges to the unique stabilizing fixed point `P̄` of the discrete algebraic
Riccati recursion `P ← F(P − PH'(HPH'+r)^{−1}HP)F' + Q`. For this 2-state system, fully eliminating
the 3 free entries of the symmetric `P̄` produces a genuine quartic with no illuminating closed
form in the general case — so, exactly as textbook treatments and production software (e.g.
statsmodels' `UnobservedComponents`) do, `steady_state_kalman_gain` finds `P̄` **numerically**, by
iterating the Riccati map from `P_0=Q` to convergence (guaranteed for this detectable/stabilizable
system — Anderson & Moore (1979), *Optimal Filtering*, Ch. 4). This converges to machine precision
in well under 100 iterations for the parameter regimes used here (verified: `n_iter`≈15–40,
`converged=True`).

**Special case check (closed form available and derived here as a sanity anchor):** setting
`q_slope=0` collapses the model to the local-LEVEL model `level_t=level_{t−1}+w1_t`,
`y_t=level_t+v_t`. Its steady-state Riccati fixed point solves `P = Pr/(P+r) + q_level`, i.e.
`P² − q_level·P − q_level·r = 0`, giving the closed form

  P̄ = [q_level + √(q_level² + 4·q_level·r)] / 2 ,  K̄ = P̄/(P̄+r) .

Substituting back, the steady-state update is `x̂_t = x̂_{t−1} + K̄(y_t − x̂_{t−1})` — **exactly** a
single EWMA with smoothing parameter `α=K̄`. This recovers, from first principles, the classical
result (Muth (1960); Chatfield & Yar (1988)) that a single EWMA is the steady-state Kalman filter
/ minimum-MSE predictor for a local level (ARIMA(0,1,1)) process.

#### A.2.3 Exact correspondence to Holt's linear (double) exponential smoothing

At steady state, writing the innovation `e_t := y_t − level_{t−1} − slope_{t−1}`, the filter update is

  level_t = level_{t−1} + slope_{t−1} + K_level · e_t
  slope_t = slope_{t−1}               + K_slope · e_t .

Holt's linear exponential smoothing (`α,β`) is

  level_t = level_{t−1}+slope_{t−1} + α·e_t
  slope_t = β(level_t−level_{t−1}) + (1−β)slope_{t−1}.

Expanding Holt's slope update using `level_t − level_{t−1} = slope_{t−1} + α e_t`:

  slope_t = β(slope_{t−1} + α e_t) + (1−β)slope_{t−1} = slope_{t−1} + (αβ)·e_t .

Matching level equations gives `α = K_level`; matching slope equations gives `αβ = K_slope`, hence

  **α = K_level ,  β = K_slope / K_level .**

This is the derivation (not merely a citation) of the well-known equivalence class between the
steady-state local-linear-trend Kalman filter and Holt's linear smoothing (ETS(A,A,N) in
Hyndman–Koehler–Ord–Snyder (2008); Harvey (1989) §2.5.4). `holt_parameters_from_kalman_gain`
implements this map, verified via a manual side-by-side recursion test.

**Why use the Kalman parameterization** rather than tuning `(α,β)` directly: `(q_level, q_slope, r)`
are *variances* — an interpretable signal-to-noise decomposition — and the filter also reports the
genuine pre-steady-state transient (first ~20–50 obs), which a fixed-`(α,β)` implementation cannot
represent.

---

### A.3 Causal rolling l1 trend filtering (Kim–Koh–Boyd, adapted)

#### A.3.1 The original (offline) method

Kim, Koh, Boyd & Boyd (2009), "l1 Trend Filtering" (SIAM Review 51(2)), solve, over an entire
FIXED, fully-observed series `y`,

  minimize_x  (1/2)‖y−x‖₂² + λ‖Dx‖₁ ,   (D x)_i = x_i − 2x_{i+1} + x_{i+2}.

The `ℓ1` penalty on second differences drives most to exactly zero, producing a piecewise-linear
fit with a sparse set of kinks — solved globally/offline; every fitted point, including the first,
uses the entire sample, future included.

#### A.3.2 The causal rolling adaptation, and what it costs

`l1_trend_filter_rolling` re-solves the SAME problem independently on each strictly causal window
`y[t−W+1:t+1]` and reports one point of that fit as "the trend at t". Approximation costs, stated
explicitly: (a) **information loss** — each window forgets everything before its start, so
globally-optimal kink placement is replaced by disagreeing local placements; (b) **boundary-effect
instability** — a window's last point has no future neighbor to two-sidedly regularize it, so the
raw edge estimate (`edge_offset=0`) is noisier than an interior fit. `edge_offset=k` reads a
better-anchored interior point at the cost of `k` days of lag, **remaining fully causal for any k**
since it only changes which position of an already-causal window is read, never which data is used.

#### A.3.3 Solving each window: ADMM with a cached, reusable factorization

Standard scaled ADMM (Boyd et al. 2011; Ramdas & Tibshirani 2016 apply exactly this to trend
filtering): `x ← (I+ρD'D)^{−1}(y+ρD'(z−u))`, `z ← soft_threshold(Dx+u, λ/ρ)`, `u ← u+Dx−z`.
`(I+ρD'D)` depends only on window length and `ρ`, so its Cholesky factor is cached once
(`lru_cache`) and reused across every day/name that shares that `(window,ρ)` — the key to making
this tractable across a multi-thousand-day, 93-name panel.

**A real bug found and fixed during validation:** seeding `z₀=Dy` (with `x₀=y`) is a degenerate
fixed point — the first x-update algebraically reproduces `x₁=y` exactly, so the convergence check
fires after one iteration before `z/u` move at all, silently returning the unfiltered input for
every `λ` tested. Re-seeding `z,u=0` fixes this (verified: monotone bias–variance tradeoff in `λ`
now holds, converging to a straight line only in the correct large-`λ` limit).

#### A.3.4 Calibrating λ: absolute thresholds do not generalize across assets

`λ` is an absolute threshold in `y`'s units, so one fixed value does not transfer across
assets/vol regimes. `λ=3.0` (plausible-looking) turned out to force near-maximum regularization
(a single straight line) on real ~1-2%/day-vol daily log-price data. Fix: `lam=None` computes,
inside each causal window, `λ := lam_mult · 1.4826 · MAD(diff(window))` (Donoho–Johnstone-style
robust threshold calibration); `lam_mult` is the actual portable free parameter, verified scale-
invariant under an arbitrary affine rescaling of price level.

---

### A.4 Multi-horizon combination: Grinold–Kahn-motivated IC weighting, not argmax

#### A.4.1 Why not argmax, and why not equal-weight either

Established Fact (5) demonstrates that argmax over a noisy regime-dependent statistic is fragile.
Grinold & Kahn's Fundamental Law of Active Management (`IR ≈ IC·√BR`) is the standard framework
for combining several noisy forecasts rather than picking one.

#### A.4.2 Derivation: optimal linear combination under a covariance structure

For `K` normalized component signals with `IC_i = Corr(s_i, r_{t+1})` and signal correlation
matrix `Σ`, the combination `c=Σw_i s_i` maximizing `Corr(c,r_{t+1})` s.t. `Var(c)=1` satisfies
the Lagrangian stationarity condition `IC=2κΣw`, i.e. **w ∝ Σ^{−1} IC**. When `Σ=I` this collapses
to `w_i∝IC_i` and `IC_combined=√(Σ IC_i²)` — the Fundamental Law's `√BR` scaling IS this formula
in the equal-IC, mutually-uncorrelated special case: breadth enters the weight as the mechanical
consequence of summing squared near-independent ICs inside `Σ^{−1}IC`, not as a separate bonus.

#### A.4.3 From the theoretical optimum to an estimable rule

Full `Σ^{−1}IC` is not used operationally: the component speeds are heavily mutually correlated
(near-singular `Σ`), so `combination_weights_from_ic` instead (1) estimates each component's own
trailing IC on a rolling causal window, (2) floors IC at zero (a stable negative IC is far less
plausible than estimation noise for a momentum-family signal at this breadth), (3) weights
proportional to IC (the `Σ≈I` diagonal approximation), then (4) shrinks toward equal-weight by a
Ledoit–Wolf-style convex blend (`shrinkage=0.5` default) — trading a controlled amount of bias for
a large reduction in the raw-IC estimator's sampling variance.

#### A.4.4 Causal IC estimation

`rolling_information_coefficient` applies the decide-at-close(t−1)/act-at(t) discipline one level
up (to a diagnostic about a signal): `raw_ic.loc[t]` needs `fwd_return.loc[t]=log_price_{t+1}-
log_price_t`, unknowable at close `t`, so the function returns `raw_ic.shift(1)`.

---

### A.5 The final deliverable and what was rejected

**FINAL VARIANT IMPLEMENTED** (`multi_horizon_trend_signal`): 3 EWMA return-momentum legs
(halflives 8/32/128 trading days) + 1 Kalman local-linear-trend slope
(`q_level=1e-5,q_slope=1e-7,r=1e-4`) + 1 causal rolling l1-trend-filter slope
(`window=120,lam_mult=1.0,edge_offset=0`), each normalized (Baltas–Kosowski bounded response),
combined via `combination_weights_from_ic` at `shrinkage=0.5, ic_window=252`, `.shift(1)`'d into
an exposure-ready score.

**Grid points contributed to the Stage-A sweep** (exact enumeration):
1. EWMA-only ensemble: halflife triples `{(8,32,128),(5,20,80),(10,50,200)}` × `shrinkage∈{0,0.5,1}` = **9**
2. Kalman-only: `(q_level,q_slope,r)∈{(1e-6,1e-8,1e-4),(1e-5,1e-7,1e-4),(1e-4,1e-6,1e-4)}` = **3**
3. l1-trend-filter-only: `window∈{60,120,252}` × `lam_mult∈{0.5,1.0,2.0}` = **9**
4. Full multi-method combination: `shrinkage∈{0,0.25,0.5,0.75,1}` × `ic_window∈{126,252}` = **10**

**Total Axis A grid points: 31.**

**Rejected: causal wavelet decomposition.** Standard DWT/MODWT use two-sided kernels; a causal
version either destroys orthogonality (minimum-phase basis) or incurs latency that doubles per
dyadic scale — precisely wrong for a signal meant to describe the current trend at low latency.
The existing causal EWMA/Kalman/l1TF machinery already performs the same multi-scale decomposition
job exactly causally with a documented combination recipe, so a wavelet leg would add the latency
and rolling-boundary-edge-effect problems without revealing materially different structure. No
wavelet code is implemented.

---

### A.6 Axis A module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/signal_extraction.py`
  (validation driver at `/home/speculation/convexity-cta/research/cta/lib/validate_signal_extraction.py`;
  pytest suite at `/home/speculation/convexity-cta/research/cta/tests/test_signal_extraction.py`,
  run via `pytest research/cta/tests -q`, not part of the main `tests/` gate).
- **Grid points declared:** 4 buckets — (1) EWMA-only ensemble: 3 halflife-triples
  `{(8,32,128),(5,20,80),(10,50,200)}` × 3 shrinkage values `{0,0.5,1.0}` = 9 points. (2)
  Kalman-only: 3 `(q_level,q_slope,r)` triples = 3 points. (3) l1-trend-filter-only: 3 window
  values `{60,120,252}` × 3 `lam_mult` values `{0.5,1.0,2.0}` = 9 points. (4) Full multi-method
  combination: 5 shrinkage values `{0,0.25,0.5,0.75,1.0}` × 2 `ic_window` values `{126,252}` = 10
  points. **Total: 31.**
- **Function signatures:**
```
halflife_to_decay(halflife: float) -> float
center_of_mass(halflife: float) -> float
ewma_level(log_price: pd.Series, halflife: float) -> pd.Series
ewma_return_momentum(log_price: pd.Series, halflife: float) -> pd.Series
ewma_price_crossover(log_price: pd.Series, fast_halflife: float, slow_halflife: float) -> pd.Series
crossover_as_return_ewma_difference(log_price: pd.Series, fast_halflife: float, slow_halflife: float) -> pd.Series
normalize_signal(raw: pd.Series, vol_window: int = 20, min_periods: int|None = None, response: str = "tanh_gaussian") -> pd.Series
steady_state_kalman_gain(q_level: float, q_slope: float, r: float, max_iter: int = 2000, tol: float = 1e-14) -> dict
holt_parameters_from_kalman_gain(K_level: float, K_slope: float) -> dict
kalman_local_linear_trend(log_price: pd.Series, q_level: float, q_slope: float, r: float) -> pd.DataFrame
l1_trend_filter_window(y: np.ndarray, lam: float, rho: float = 1.0, n_iter: int = 150, tol: float = 1e-6) -> np.ndarray
l1_trend_filter_rolling(log_price: pd.Series, window: int = 120, lam: float|None = None, lam_mult: float = 1.0, rho: float = 1.0, edge_offset: int = 0, n_iter: int = 150) -> pd.Series
l1_trend_filter_slope_signal(log_price: pd.Series, window: int = 120, lam: float|None = None, lam_mult: float = 1.0, rho: float = 1.0, edge_offset: int = 0, slope_lookback: int = 5) -> pd.Series
rolling_information_coefficient(signal: pd.Series, log_price: pd.Series, window: int) -> pd.Series
combination_weights_from_ic(trailing_ic: pd.DataFrame, shrinkage: float = 0.5) -> pd.DataFrame
build_component_signals(log_price: pd.Series, ewma_halflives=DEFAULT_EWMA_HALFLIVES, kalman_params=DEFAULT_KALMAN_PARAMS, l1tf_params=DEFAULT_L1TF_PARAMS, vol_window: int = 20) -> pd.DataFrame
multi_horizon_trend_signal(log_price: pd.Series, ewma_halflives=DEFAULT_EWMA_HALFLIVES, kalman_params=DEFAULT_KALMAN_PARAMS, l1tf_params=DEFAULT_L1TF_PARAMS, vol_window: int = 20, ic_window: int = 252, shrinkage: float = 0.5) -> pd.DataFrame  # TOP-LEVEL DELIVERABLE; 'combined' column is exposure-ready, bounded in (-1,1)
synthetic_outlier_injection_test(log_price: pd.Series, shock_index: int|None = None, shock_size: float = 0.5) -> dict
```
- **Injection test result:** Ran on real SPY and QQQ daily log-price series (source=='duka'
  filtered, `daily_close_series_duka` convention). Injected a permanent +50% (SPY) and -40% (QQQ)
  log-price shock at the series midpoint (index 996). Every raw component first reacts exactly AT
  the shock index (996), never before (`raw_causal_ok=True`); the exposure-ready `combined`
  ensemble output first reacts at 997 = shock_index+1, confirming the additional
  decide-at-close(t-1)/act-at(t) shift is correctly applied (`exposure_causal_ok=True`). Overall
  verdict both names: **PASS**. Also verified: the crossover⇔return-EWMA-momentum identity holds
  to float precision (~6e-15) on real SPY/QQQ/MSFT/JPM/COST data; Kalman filter's own last-day
  empirical gain matches the closed-form steady-state gain to within 1e-4; Holt (α,β)
  correspondence verified via manual side-by-side recursion; l1 trend filter produces 1874-1918 of
  ~2000 valid daily estimates per name with the expected monotone bias-variance tradeoff after two
  implementation bugs were found and fixed (degenerate ADMM z-seeding; non-portable absolute
  lambda threshold). Full `pytest research/cta/tests`: 16 passed. Full repo `pytest tests -q`: 654
  passed, 1 skipped (baseline unchanged).
- **Packages installed:** None. All functionality uses packages already in the shared venv (numpy
  2.5.0, pandas 3.0.3, scipy 1.18.0 — `scipy.linalg.cho_factor`/`cho_solve` and
  `scipy.sparse.diags` for the l1 trend filter's ADMM solve).
## Axis B: Entry/Exit as Optimal Control Under Transaction Costs

### B.1 The problem this replaces

Every trend-following overlay reduces, at the exposure-decision layer, to a rule for turning a
raw trend estimate into a position. The naive version used throughout the closed E1/E2/E3/I1
programs and the exploratory CTA scans is a fixed hysteresis band (e.g. "go long when price >
1.02*SMA, flat when price < 0.98*SMA, hold inside the band") — a 2% width chosen by inspection,
with no connection to the venue's actual cost structure or the signal's own statistical
properties. This is the object Axis B replaces with a **derived** control law. The two questions
a hysteresis band conflates are actually separable, and the transaction-cost-control literature
treats them as two faces of one problem:

1. **How fast should the position move toward the target once it is worth moving at all** — a
   *partial-adjustment* / aim-portfolio question (Garleanu & Pedersen 2013).
2. **How far can the position be allowed to drift from the target before it is worth paying to
   move it at all** — a *no-trade band* / singular-control question (Davis & Norman 1990; Shreve
   & Soner 1994).

Both are governed by the same primitive: the curvature of the cost of *not* holding the optimal
position (a risk-aversion/tracking-error term) versus the cost of *trading* to fix that. Section
B.4 makes the connection between the two explicit and shows they are literally the same
constrained optimization viewed through two different friction assumptions (quadratic trading
cost vs. proportional trading cost).

### B.2 Garleanu & Pedersen (2013): the aim portfolio and partial adjustment

#### B.2.1 Setup

GP2013 posits an investor with mean-variance preferences trading an asset (or vector of assets)
whose expected excess return is driven by a predictive factor $f_t$ that itself mean-reverts.
Trading is costly: the cost of moving the position by $\Delta x_t = x_t - x_{t-1}$ in one period
is *quadratic*, $\frac{1}{2}\Delta x_t' \Lambda \Delta x_t$, the standard reduced-form for a
linear-price-impact market (Kyle-style: impact per unit traded rises linearly with the trade,
so cost — impact integrated over the trade — is quadratic in size).

Scalar reduction (single instrument, the case relevant here): signal $f_t$ follows an AR(1),

$$ f_t = \rho\, f_{t-1} + \varepsilon_t, \qquad \rho = 1-\phi,\ \phi\in(0,1) \text{ the mean-reversion speed}, $$

the investor's one-period objective is

$$ \max_{x_t}\ \mathbb E\Big[\sum_{t} \beta^t\Big( x_{t-1} f_t - \tfrac{\gamma}{2}\sigma^2 x_{t-1}^2 - \tfrac{\lambda}{2}(x_t-x_{t-1})^2\Big)\Big], $$

where $\gamma$ is risk aversion, $\sigma^2$ the return variance, $\lambda$ the quadratic
trading-cost coefficient, $\beta$ the discount factor. The frictionless (Markowitz, $\lambda=0$)
optimum is static: $x_t^\* = f_t/(\gamma\sigma^2) \equiv b f_t$ — trade instantly to the current
signal every period.

#### B.2.2 The Euler equation (discrete time, general form)

Differentiating the objective with respect to $x_t$ (holding $x_{t-1}$ and the continuation value
fixed) gives GP's Euler equation — the exact discrete-time first-order condition their paper
states (their eq. 7, specialized to the scalar case):

$$ \lambda\,\Delta x_t \;=\; -\gamma\sigma^2 x_t + f_t \;+\; \beta\,\mathbb E_t[\lambda\,\Delta x_{t+1}]. $$

In words: *the marginal cost of trading today equals today's marginal mistracking benefit plus
the discounted expected marginal trading-cost saved tomorrow by having moved today.* This is a
genuine Euler equation (a stochastic difference equation linking today's control to tomorrow's
expected marginal continuation value) — it is *not* solved by a static rule; solving it in
general requires a matrix Riccati recursion (or, for the multi-signal case GP actually treat, an
infinite-dimensional "aim portfolio" that is a duration-weighted average of the whole future path
of frictionless targets).

#### B.2.3 A tractable closed form: continuous-time HJB solve (worked from scratch)

Rather than iterate GP's discrete Riccati numerically (which has no closed form once $\phi$ is
folded in), I solve the continuous-time limit directly — this *is* the standard GP small-$\Delta t$
reduction, and it yields an explicit, auditable closed form for the trading speed, which is what
an implementation needs. State the continuous-time analogue of the above: control $u_t$ is the
trading *rate*, $dx_t = u_t\,dt$; $f_t$ is an OU process, $df_t = -\phi f_t\,dt + \sigma_f\,dZ_t$;
instantaneous flow payoff $x_t f_t - \frac{\gamma}{2}\sigma^2 x_t^2 - \frac{\lambda}{2}u_t^2$.

**HJB equation:**
$$ \beta V(x,f) = \max_u\Big[\, x f - \tfrac{\gamma}{2}\sigma^2 x^2 - \tfrac{\lambda}{2}u^2 + V_x\,u - \phi f\,V_f + \tfrac{\sigma_f^2}{2}V_{ff}\,\Big]. $$

**FOC** (the actual first-order condition asked for): $\partial/\partial u = 0 \Rightarrow -\lambda u + V_x = 0 \Rightarrow u^\* = V_x/\lambda.$

Guess a quadratic value function $V(x,f) = -\tfrac{A}{2}x^2 + Bxf + \tfrac{C}{2}f^2 + D$, so
$V_x = -Ax+Bf$. Substituting $u^\*=V_x/\lambda$ back in gives $\tfrac{1}{2\lambda}V_x^2$ (the
standard LQ simplification $-\tfrac{\lambda}{2}u^{\*2}+V_x u^\* = \tfrac{1}{2\lambda}V_x^2$), and
matching coefficients of $x^2$, $xf$, $f^2$ on both sides of the HJB gives three algebraic
equations. The $x^2$-coefficient equation is the one that pins down the trading speed and is
self-contained:

$$ -\tfrac{\beta A}{2} = -\tfrac{\gamma\sigma^2}{2} + \tfrac{A^2}{2\lambda} \quad\Longrightarrow\quad A^2 + \beta\lambda A - \gamma\sigma^2\lambda = 0. $$

Solving the quadratic in $A$ (positive root, required for $V$ concave in $x$) and defining the
trading speed $\kappa \equiv A/\lambda$ (from $u^\*=V_x/\lambda=-\kappa x+\tfrac{B}{\lambda}f$, i.e.
$dx_t = \kappa\,(x^\*_t - x_t)\,dt$ with $x^\*_t=(B/A)f_t$ the *aim portfolio*):

$$ \boxed{\ \kappa = \dfrac{\sqrt{\beta^2 + 4\gamma\sigma^2/\lambda}\; -\; \beta}{2}\ } $$

At the horizons relevant here (daily rebalancing decisions; $\beta$ is an *annualized* discount
rate on the order of a few percent, i.e. utterly negligible against daily dynamics), the
$\beta\to0$ limit is the practically relevant one and collapses to the clean, literature-standard
form (the same expression appears, e.g., in practitioner cost-scaling treatments such as
Frazzini–Israel–Moskowitz's trading-cost appendices):

$$ \kappa \;\approx\; \sqrt{\dfrac{\gamma\sigma^2}{\lambda}} \;=\; \sqrt{\dfrac{k}{\lambda}}, \qquad k \equiv \gamma\sigma^2. $$

**Interpretation:** the optimal speed of adjustment is the *geometric mean* (as a ratio) of the
risk-cost curvature $k$ (how expensive it is, per unit time, to sit away from target) and the
trading-cost curvature $\lambda$ (how expensive it is to close the gap). Sanity check: as
$\lambda\to0$ (frictionless), $\kappa\to\infty$ — snap instantly to target, recovering the static
Markowitz solution, as required. The signal's own mean-reversion rate $\phi$ affects only the
*aim portfolio's* construction ($B/A$, which folds in the expected future path of $f$) and does
not appear in $\kappa$ once $x^\*_t$ is defined as *the already-decayed, mean-reversion-aware
target* — which is exactly the interface Axis B is handed (a scalar $x^\*_t\in[-1,1]$ from Axis A,
already representing "the target position given everything currently known about the trend",
not a raw signal). This is a deliberate and load-bearing modeling choice discussed in B.5.

Discretizing $dx=\kappa(x^\*-x)dt$ exactly over one trading day (holding $x^\*$ fixed over the
interval, the standard OU-update): $x_t = x_{t-1} + \kappa_d\,(x^\*_t - x_{t-1})$, with
$\kappa_d = 1-e^{-\kappa}$.

### B.3 Davis & Norman (1990) / Shreve & Soner (1994): the no-trade band under proportional costs

#### B.3.1 Why GP's framework is the wrong friction model for *this* venue

The GP machinery above assumes **quadratic** trading costs (price impact, cost rises with trade
size). This venue's actual, declared cost model is **linear/proportional**: `13bp * |Δexposure|`
per name-day (`apply_costs` in `research/indices/i1_pipeline.py`, `RT_PRIMARY_BP=13.0`) — the
marginal cost of trading is *constant*, not increasing in size. Under a genuinely proportional
cost, the classical result (Constantinides 1986; made rigorous as a singular-stochastic-control
free-boundary problem by Davis & Norman 1990 for the Merton consumption-investment problem, and
generalized via viscosity-solution methods by Shreve & Soner 1994) is qualitatively different
from GP's smooth partial adjustment: the optimal policy is **bang-bang at a free boundary** — do
nothing while the state is inside a *no-trade region*, and the instant it exits, trade the
*minimum amount* needed to return to the boundary (not all the way back to the frictionless
target). A proportional-cost problem literally cannot have an interior smooth-adjustment solution
because the marginal cost of the first infinitesimal unit of trading is already the full rate
$\theta$, not zero — so it is never worth trading an infinitesimal amount, only worth trading once
the accumulated benefit of moving clears the fixed-per-unit hurdle $\theta$. This is the
mathematically correct replacement for an arbitrary "2% band" here, *not* GP's smooth rule taken
on its own.

#### B.3.2 Derivation (worked from scratch via Harrison–Taksar smooth pasting)

Let $z_t = x_t - x^\*_t$ be the tracking error. Treat the target as locally slow relative to the
control (absorb its drift into a recentering, standard in this literature) so that the
*uncontrolled* dynamics of $z$ are driftless Brownian motion with instantaneous variance
$\sigma_z^2$ (in practice: the day-to-day noise of the aim-portfolio signal itself, since that is
the dominant source of "surprise" mistracking between trading opportunities). We hold an
instantaneous quadratic tracking-cost $\tfrac{k}{2}z^2$ ($k=\gamma\sigma_r^2$ — *the same
curvature primitive as GP's*, using the underlying's realized return variance $\sigma_r^2$) and
pay proportional cost $\theta$ per unit of cumulative trading effort $|dL_t|$ (a nondecreasing,
possibly two-sided control — Harrison & Taksar's (1983) "instantaneous control of Brownian
motion" is exactly this problem). Minimize the long-run average cost rate $g$
(the $\beta\to0$/undiscounted formulation, the standard simplification in this literature and the
one that yields a clean closed form):

$$ g = \lim_{T\to\infty}\tfrac1T\,\mathbb E\!\int_0^T\Big[\tfrac{k}{2}z_t^2\,dt + \theta\,d|L_t|\Big]. $$

In the continuation (no-trade) region the relative value function $W(z)$ satisfies

$$ g = \tfrac{k}{2}z^2 + \tfrac{\sigma_z^2}{2}W''(z), $$

with **smooth-pasting** boundary conditions at the (symmetric, by problem symmetry) free
boundary $z=\pm\Delta$: value-matching $W'(\pm\Delta)=\pm\theta$ (marginal value of relaxing the
constraint by one unit equals the marginal cost of trading) *and* second-order smooth-fit
$W''(\Delta)=0$ (the extra condition, standard for optimally-chosen singular-control boundaries,
that pins down $\Delta$ itself rather than leaving it free), plus $W'(0)=0$ by symmetry.

Integrate $W''(z) = \tfrac{2}{\sigma_z^2}\big(g - \tfrac{k}{2}z^2\big)$ twice:

$$ W'(z) = \tfrac{2g}{\sigma_z^2} z - \tfrac{k}{3\sigma_z^2} z^3 \qquad (\text{using } W'(0)=0). $$

Apply the two boundary conditions at $z=\Delta$:

$$ W''(\Delta)=0:\quad \tfrac{2g}{\sigma_z^2} - \tfrac{k}{\sigma_z^2}\Delta^2 = 0 \ \Longrightarrow\ g = \tfrac{k}{2}\Delta^2. $$
$$ W'(\Delta)=\theta:\quad \tfrac{2g}{\sigma_z^2}\Delta - \tfrac{k}{3\sigma_z^2}\Delta^3 = \theta. $$

Substitute $g=\tfrac{k}{2}\Delta^2$ into the second equation:
$\tfrac{k\Delta^3}{\sigma_z^2} - \tfrac{k\Delta^3}{3\sigma_z^2} = \theta \Rightarrow
\tfrac{2}{3}\tfrac{k\Delta^3}{\sigma_z^2} = \theta$, giving the **closed-form no-trade half-width**:

$$ \boxed{\ \Delta^\* = \left(\dfrac{3\,\theta\,\sigma_z^2}{2\,k}\right)^{1/3}\ } $$

This is exact for this idealized (Brownian-driven, quadratic-holding-cost, proportional-control-cost)
reduction, and it reproduces the well-known **cube-root scaling law** $\Delta^\*\propto\theta^{1/3}$
that Constantinides (1986) derived as the leading-order asymptotic for the *general* (power-utility
Merton) problem, later made fully rigorous via viscosity solutions by Shreve & Soner (1994) — the
constant $3/2$ here is specific to the quadratic-holding-cost reduction used, not a universal
constant of the general Merton problem, and this distinction is stated explicitly rather than
implied. **This is the actual derivation the task requested** — a real, closed-form,
non-guessed number for the no-trade band width, replacing the arbitrary fixed 2%, as a function
of the venue's real cost rate $\theta$ and the signal's own volatility characteristics
$\sigma_z,k$.

### B.4 The optimal-stopping / free-boundary connection (why these are one problem)

Singular stochastic control problems of exactly this shape (Harrison–Taksar 1983; Davis–Norman
1990) are solved via the **smooth-pasting principle**, which is the *same* device used to solve
optimal-stopping (free-boundary/obstacle) problems — this is not a coincidence but a documented
duality (El Karoui & Karatzas; Bank & El Karoui's representation of singular controls via a family
of stopping times; presented pedagogically in Øksendal's and Shreve–Soner's treatments of singular
control). Concretely: the marginal value function $W'(z)$ derived above is *itself* the value of
an auxiliary optimal-stopping problem — "given one marginal extra unit of mistracking exposure at
state $z$, is it worth continuing to hold it, or is the moment right to 'exercise' (pay $\theta$
and trade back)?" The free boundary $z=\Delta$ where $W'(\Delta)=\theta$ *is exactly the exercise
boundary of that stopping problem*: exit (trade) is optimal once the marginal expected future cost
of continuing to hold the mistracking exposure first equals the fixed cost of paying to remove it.
The **posterior-confidence framing** falls out immediately: the aim-portfolio signal's *own*
uncertainty ($\sigma_z$, entering $\Delta$ directly) IS the thing that widens or narrows the
exercise boundary — a noisier, less-confident trend estimate (large $\sigma_z$) widens the
no-trade band (cube-root in $\sigma_z^2$), meaning the strategy tolerates a larger deviation
before conceding it is worth paying to act on a possibly-spurious signal move; a cleaner signal
narrows it, i.e. the same posterior-confidence intuition is already embedded in $\Delta^\*$
via $\sigma_z$, not a separate mechanism bolted on.

The two frameworks are literally two specializations of the *same* underlying control problem —
minimize $\mathbb E\int e^{-\beta t}\big[\tfrac{k}{2}z_t^2\,dt + \text{(cost of moving } z\text{)}\big]$
— that differ only in whether the cost-of-moving term is **quadratic in the rate of movement**
(GP, giving a smooth interior control $u^\*=\kappa z$, no free boundary, trade a little every
instant) or **linear in the magnitude of movement** (Davis–Norman, giving a hard free boundary
and a "do nothing, then jump to the boundary" policy). Formally, GP's smooth solution is the
**$\theta\to0$ limit** of the Davis–Norman family with a residual (nonzero) $\lambda$: as
$\theta\to0$, $\Delta^\*\to0$ and the no-trade region collapses to a point, degenerating exactly
into GP's continuously-adjusting rule.

### B.5 From two closed forms to one implementable rule

Reality here is *primarily* proportional cost (the venue's cost model is linear in
`|Δexposure|`, no documented quadratic/impact term) — so Davis–Norman is the mathematically
correct *primary* framework and supplies the no-trade band width $\Delta^\*$. But a pure
bang-bang-to-boundary policy is fragile in practice: it snaps discontinuously the instant the
(noisily estimated) band is crossed, amplifying signal-estimation error and turning a
soft/graded trend-confidence signal into an artificially discrete on/off decision at the boundary.
GP's smooth partial-adjustment control is the natural robustness device for the region *outside*
the band: once trading is triggered, move only *partway* toward the target at the GP-derived
speed $\kappa$ rather than jumping fully to $x^\*_t$ (or even just to the boundary). This
"no-trade band, then partial-adjust" combination is a documented practitioner pattern (e.g.
Isichenko, *Quantitative Portfolio Management*, ch. on transaction costs, combines a hysteresis
band with a proportional rebalancing rate rather than instant snap-back) and is the specific
hybrid this module implements, not an arbitrary blend — it directly reuses **both** closed forms
above with no additional free parameters, by tying GP's required (but here unobservable) quadratic
coefficient $\lambda_{\text{eff}}$ to the *already-derived* $\Delta^\*$ via **marginal-cost
continuity at the free boundary**: the Davis–Norman branch's marginal cost at $z=\Delta^\*$ is
exactly $\theta$ (by the value-matching condition); a quadratic-cost model's marginal cost at
trade size $\Delta^\*$ is $\lambda_{\text{eff}}\Delta^\*$; setting these equal,

$$ \lambda_{\text{eff}} = \theta / \Delta^\*, \qquad \kappa = \sqrt{k/\lambda_{\text{eff}}} = \sqrt{k\,\Delta^\*/\theta}. $$

**Final rule** (implemented in `partial_adjustment_control`): at each $t$ (all quantities below
computed from data through $t-1$ only — see the causality proof in B.7),

$$ z_t = x^\*_t - x_{t-1}; \qquad
x_t = \begin{cases} x_{t-1} & |z_t|\le \Delta_t \\[4pt] x_{t-1} + \kappa_{d,t}\, z_t & |z_t| > \Delta_t \end{cases} $$

with $\Delta_t$ from B.3.2 (using trailing realized $\sigma_{z,t}$ of the signal and
$k_t=\gamma\sigma_{r,t}^2$ of the underlying) and $\kappa_{d,t}=1-e^{-\kappa_t}$ from B.5's tied
$\kappa_t$.

#### Alternatives considered and rejected

- **Pure Davis–Norman (snap to boundary, no partial adjustment outside it):** rejected as the
  sole rule because it reintroduces exactly the discontinuous-snap fragility a smoothing layer is
  meant to avoid, and because this venue's real cost, while dominantly linear, is not verified to
  be *purely* linear at all trade sizes (a residual convex component from realistic slippage at
  size is plausible even if not separately estimated) — the hybrid is the more defensible choice
  under that model uncertainty.
- **Pure GP (no no-trade band, always partially adjust):** rejected as the sole rule because it is
  provably suboptimal under genuinely proportional costs (Section B.3.1) — it would trade a
  little every single day even when the marginal benefit of doing so is below the fixed
  per-unit-trade cost, guaranteed to overpay on turnover relative to the derived band.
- **Full discrete-time GP Riccati (solving the actual multi-period Euler equation numerically,
  keeping $\phi$ explicit rather than folding it into the aim portfolio):** rejected for this
  module specifically because (a) it requires numerically iterating a matrix Riccati recursion
  with no closed form, which is far less auditable/causality-provable than the closed-form
  control below, and (b) the interface contract with Axis A already hands this module a
  *post-mean-reversion-decay* aim portfolio ($x^\*_t\in[-1,1]$), making $\phi$ Axis A's concern,
  not Axis B's — re-deriving it here would double-count the signal decay.
- **Asymptotic (small-$\theta$) Constantinides band instead of the exact HJB solve:** rejected;
  since the exact closed form for this specific LQ/proportional-cost reduction was derivable
  directly (B.3.2), there was no need to fall back to a leading-order approximation.

### B.6 Grid points contributed to the Stage-A sweep

Two knobs are exposed for the sweep, both dimensionless multipliers on the closed forms above (so
every sweep point remains anchored to the derivation, never an arbitrary re-guess):

- **`cost_sensitivity` ($\zeta$):** multiplies $\theta$ *only inside this module's own
  calibration* of $\Delta^\*$ and $\kappa$ (never the real P&L cost the Sweep phase's
  `apply_costs` charges) — a robustness/aggressiveness knob. $\zeta>1$ acts as if trading were
  more expensive than the real 13bp (wider band, slower speed, conservative); $\zeta<1$ acts as
  if cheaper (aggressive).
- **`band_scale` ($\eta$):** a direct multiplier on the derived $\Delta^\*$ itself, testing
  over/under-trading relative to the closed-form optimum (independent of $\zeta$, since $\eta$
  does not also change $\kappa$'s $\lambda_{\text{eff}}$ tie-back the way $\zeta$ does).

Declared grid — **exactly 4 points**:

| name | `cost_sensitivity` ($\zeta$) | `band_scale` ($\eta$) | interpretation |
|---|---|---|---|
| `conservative` | 2.0 | 1.5 | acts as if costs are 2x real + 50% wider band than derived — slow, infrequent trading |
| `derived_baseline` | 1.0 | 1.0 | pure closed-form output, no adjustment — the "as-derived" null-hypothesis point |
| `moderate_aggressive` | 0.5 | 0.75 | acts as if costs are half real, band 25% tighter than derived |
| `aggressive` | 0.25 | 0.5 | near-frictionless, narrow band, fast tracking |

Empirically (real SPY data, `gamma=1.0`, `cost_bps=13.0`, 20-day vol windows), these four points
span median no-trade half-widths of **0.033 (aggressive) to 0.195 (conservative)** in signal
units and annualized round-trip-equivalent turnover of **0.51/yr to 0.79/yr** — see B.8. The
`derived_baseline` point is the one with no free parameters at all: every number in it traces
directly to the venue's real 13bp cost rate and the empirical $\sigma_z,\sigma_r$ of the
name/signal in question.

### B.7 Module, interface, and causality proof

Implemented in `research/cta/lib/entry_exit_control.py`. Main entry point:

```
partial_adjustment_control(target: pd.Series, price: pd.Series, cost_bps: float = 13.0,
    gamma: float = 1.0, cost_sensitivity: float = 1.0, band_scale: float = 1.0,
    vol_window: int = 20, min_periods: int = 20, band_floor: float = 0.0,
    band_cap: float = 0.5, kappa_cap: float = 1.0) -> pd.DataFrame
```

`target` is Axis A's aim-portfolio trend-strength signal in $[-1,1]$, **already** decide-at-close/
act-next-day lagged (this module does not re-lag it — see the interface-contract docstring in the
module for the exact hand-off assumption); `price` is the daily close series (e.g.
`i1_pipeline.daily_close_series_duka`). Returns a DataFrame with `exposure`, `band`,
`kappa_daily`, `z`, `traded` columns.

Supporting functions: `realized_vol` / `realized_return_vol` (causal trailing vol estimators,
`shift(1)`-lagged), `realized_vol_naive_UNLAGGED` (the deliberately-wrong control used only to
prove the injection test has teeth), `no_trade_band_width` (B.3.2's $\Delta^\*$),
`gp_adjustment_speed` (B.5's tied $\kappa$), `kappa_to_daily_fraction` (the $1-e^{-\kappa}$
discretization), and `DEFAULT_GRID` (the four points in B.6, as importable dicts for the Sweep
phase).

**IMPORTANT INTERFACE NOTE (see the unifying Section 9 of this document for the required fix):**
`partial_adjustment_control` currently computes its own internal $\sigma_z,\sigma_r$ via
`realized_return_vol(price, vol_window)` — a plain trailing realized-vol estimator with the SAME
functional form as Axis D's `trailing_realized_vol_forecast`, which Axis D's own head-to-head
QLIKE/MSE evaluation (§D.8) found to be one of the *worst*-performing vol estimators on this data.
This module was built in parallel with, and without access to, Axis D's validated deliverable.

#### Causality (injection) test — `research/cta/lib/validate_entry_exit_control.py`

Methodology: inject a large, obviously-fake +50% one-day price spike (and, separately, a +0.9
spike directly into a signal-shaped series) at a single index into a copy of real data; confirm
every derived quantity at the injected index is **byte-identical** to the un-injected run (since
it must use only data through the prior index), while the quantity **one index later** does
change (proving the lag is exactly one period, not more, i.e. the estimator isn't simply frozen
or broken). A **naive, deliberately unlagged** version of the same estimator
(`realized_vol_naive_UNLAGGED`) is run through the identical injection to confirm it *does* react
at the injected index itself — this is what proves the test has real discriminating power rather
than being a tautology that would "pass" against a broken estimator too.

Run on real daily closes for **SPY, QQQ, AAPL, MSFT, NVDA** (all 5 from the 93-name base
universe; SPY/QQQ from the program's headline benchmark, AAPL/MSFT/NVDA as long-history liquid
mega-caps). Result: **all checks pass on all 5 names** —
`realized_return_vol` unchanged at the injection day and changed the day after on every name;
the naive unlagged control changes at the injection day itself on every name (teeth confirmed);
the same unchanged/changed-after pattern holds for the signal-vol path
(`realized_vol`/`realized_vol_naive_UNLAGGED` on a directly-injected signal series, isolated from
the proxy signal's own lag structure so the test targets this module's estimators specifically,
not a confound from the test signal's construction); and the full pipeline's `band`/`kappa_daily`
columns inherit the same one-period-lag property end-to-end. Full boolean detail per name is in
the module's own run log (`validate_entry_exit_control.py`, section 3 output).

### B.8 Real-data diagnostics (sanity, not a P&L claim — that is the Sweep phase's job)

Run with a clearly-labeled **synthetic-but-realistic proxy** trend signal (a causal 50/200-day
MA-crossover distance, vol-normalized and clipped to $[-1,1]$) standing in for Axis A's real
output, since Axis A's module was not available to import at the time this module was built
(explicitly *not* claimed to be Axis A's real signal — see the module's own docstring flagging
this — see Section 9 of this document for how the real hand-off is resolved). `derived_baseline`
grid point, `cost_bps=13.0`, `gamma=1.0`:

| symbol | span (yrs) | % valid days traded | median band $\Delta$ | median $\kappa_d$ | turnover (rt-equiv/yr) |
|---|---|---|---|---|---|
| SPY | 7.93 | 63.4% | 0.1034 | 0.0715 | 0.637 |
| QQQ | 7.92 | 54.0% | 0.1134 | 0.0973 | 0.651 |
| AAPL | 7.93 | 57.3% | 0.1024 | 0.1246 | 1.038 |
| MSFT | 7.96 | 55.4% | 0.1005 | 0.1094 | 0.747 |
| NVDA | 7.90 | 55.2% | 0.0783 | 0.1714 | 0.846 |

These numbers are proxy-signal-dependent (a different Axis A signal will shift both the band and
observed turnover) and are reported purely to confirm the module produces sane, boundedly-behaved
output on real data (bands in single-digit-to-teens percent of the $[-1,1]$ signal range, turnover
of well under 1 round-trip-equivalent per year at the derived baseline) — not as a claim about
strategy performance, which is the Sweep/Synthesis phase's responsibility once Axis A's real
signal is wired in.

### B.9 Axis B module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/entry_exit_control.py`
  (validation script alongside it at
  `/home/speculation/convexity-cta/research/cta/lib/validate_entry_exit_control.py`).
- **Grid points declared:** Exactly 4 `(cost_sensitivity ζ, band_scale η)` points — see B.6 table.
- **Function signatures:**
```
realized_vol(series: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series
realized_vol_naive_UNLAGGED(series: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series   # deliberately-wrong control, test-only
realized_return_vol(price: pd.Series, window: int = 20, min_periods: int = 20, ddof: int = 0) -> pd.Series
no_trade_band_width(theta: float | pd.Series, k_curvature: float | pd.Series, sigma_z: float | pd.Series, floor: float = 1e-10) -> float | pd.Series
gp_adjustment_speed(k_curvature: float | pd.Series, theta: float | pd.Series, band_width: float | pd.Series, floor: float = 1e-12) -> float | pd.Series
kappa_to_daily_fraction(kappa: float | np.ndarray, cap: float = 1.0) -> float | np.ndarray
partial_adjustment_control(target: pd.Series, price: pd.Series, cost_bps: float = 13.0, gamma: float = 1.0, cost_sensitivity: float = 1.0, band_scale: float = 1.0, vol_window: int = 20, min_periods: int = 20, band_floor: float = 0.0, band_cap: float = 0.5, kappa_cap: float = 1.0) -> pd.DataFrame
DEFAULT_GRID: list[dict]
```
- **Injection test result:** Ran on real daily closes for SPY, QQQ, AAPL, MSFT, NVDA. Injected a
  fake +50% one-day price spike (and, separately, a +0.9 spike directly into a signal-shaped
  series) at a single mid-sample index into a copy of real data. Result on ALL 5 names:
  `realized_return_vol` unchanged at the injection index (byte-identical vs. un-injected run) and
  changed at index+1; the same unchanged-then-changed pattern confirmed for `realized_vol` applied
  directly to a signal-shaped series (isolating this from the proxy trend-signal's own separate
  1-day lag); and the full `partial_adjustment_control()` pipeline's `band`/`kappa_daily` columns
  inherit the identical one-period-lag property end to end. All PASS=True on SPYUSDT, QQQUSDT,
  AAPLUSDT, MSFTUSDT, NVDAUSDT.
- **Packages installed:** None. Module uses only numpy and pandas, both already present in
  `/home/speculation/convexity/.venv`.
## Axis C — Regime Modeling: a strictly-causal Gaussian Hidden Markov Model regime gate

### C.1 Motivation, tied to Established Fact (4)

Established Fact (4) documents a concrete failure mode of the risk-management primitive used
throughout this program's prior axes: trailing realized volatility is **reactive**. It "protected
well during a gradual vol ramp (COVID, March 2020) but failed completely ahead of a sudden
reversal (2025-04-09 tariff-pause squeeze, trailing 20-day realized vol was an unremarkable ~2%
that very morning)." A trailing-vol estimator only accumulates evidence of stress *after* large
moves have already occurred within its lookback window; it has no notion of a discrete regime
*change* and therefore cannot distinguish "calm because nothing has happened yet" from "calm
because the underlying process is genuinely low-variance."

A **regime-switching model** attacks this from a different angle: rather than smoothing squared
returns, it posits a small number of latent, persistent states, each with its own return
distribution, and asks a probabilistic question every day — "given everything observed through
yesterday's close, how likely is today to belong to each state?" Because state persistence
(captured by a transition matrix) is itself estimated from history, the model can react to a
single unusual day's *return level*, not just its magnitude, and — critically — it degrades
*smoothly* near an ambiguous regime boundary instead of flipping a hard binary switch.

### C.2 The model: Gaussian Hidden Markov Model (Hamilton 1989 as conceptual root)

Let $r_t$ be a daily (log) return. Posit a latent state process $S_t \in \{1, \dots, K_C\}$, first-order
Markov, with transition matrix $A$, $A_{ij} = P(S_t = j \mid S_{t-1} = i)$, and initial distribution
$\pi = P(S_1 = \cdot)$. Conditional on the state, returns are drawn i.i.d. Gaussian:

$$ r_t \mid S_t = k \;\sim\; \mathcal{N}(\mu_k, \sigma_k^2), \qquad t=1,\dots,T $$

(Here $K_C$ denotes the number of latent states, 2 or 3 in this axis's grid — subscripted to
distinguish it from Axis A's unrelated use of $K$ for the count of component trend signals; see
the unifying document's notation-reconciliation table.)

This is the direct descendant of **Hamilton (1989)**, "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle" (*Econometrica* 57(2)), which introduced
regime-switching for macro/financial time series with an autoregressive conditional mean per
regime; here the emission is simplified to a state-conditional i.i.d. Gaussian (no AR term) — see
§C.6 for why that simplification was made deliberately, not out of ignorance of the richer model.
The broader estimation machinery (forward-backward, EM) is the Baum-Welch algorithm from the
speech-recognition HMM literature (Baum et al. 1970; Rabiner 1989's tutorial is the standard
citable reference for the forward/backward/Baum-Welch recursions used verbatim below), and
Hamilton (1994, *Time Series Analysis*, ch. 22) is the standard reference for the **filtered vs.
smoothed inference** distinction that is the crux of this axis's causality design (§C.4).
Regime-switching models have an extensive dedicated finance literature — Ang & Bekaert (2002,
"Regime Switches in Interest Rates," *J. Business & Economic Statistics*), Ang & Timmermann (2012,
"Regime Changes and Financial Markets," *Annual Review of Financial Economics*) survey
applications, and Kritzman, Page & Turkington (2012, "Regime Shifts: Implications for Dynamic
Strategies," *Financial Analysts Journal*) is the most directly relevant precedent for this axis's
specific deliverable: turning a fitted regime probability into a **continuous** dynamic-allocation
signal rather than a discretized regime label, exactly the "smooth gate" idea developed below.

### C.3 Estimation: EM / Baum-Welch (E-step = forward-backward, M-step = closed-form)

Given a training window of returns $r_{1:T}$ and current parameter guess
$\theta = (\pi, A, \{\mu_k,\sigma_k^2\})$, the **forward variables**
$\alpha_t(k) = P(S_t=k, r_{1:t} \mid \theta)$ and **backward variables**
$\beta_t(k) = P(r_{t+1:T} \mid S_t=k,\theta)$ satisfy the standard recursions

$$\alpha_1(k) = \pi_k\, b_k(r_1), \qquad \alpha_t(k) = b_k(r_t)\sum_i \alpha_{t-1}(i)\,A_{ik}$$
$$\beta_T(k) = 1, \qquad \beta_t(k) = \sum_j A_{kj}\,b_j(r_{t+1})\,\beta_{t+1}(j)$$

where $b_k(x) = \mathcal N(x;\mu_k,\sigma_k^2)$. The E-step computes the **smoothed** posteriors
$\gamma_t(k) \propto \alpha_t(k)\beta_t(k)$ and pairwise posteriors $\xi_t(i,j) \propto
\alpha_t(i)A_{ij}b_j(r_{t+1})\beta_{t+1}(j)$; the M-step updates $\mu_k, \sigma_k^2$ as
$\gamma$-weighted moments of $r_t$ and $A_{ij}$ as normalized expected transition counts from
$\xi$. This is implemented via `hmmlearn.hmm.GaussianHMM.fit` (tested against **hmmlearn 0.3.3**,
installed into the shared venv along with its dependency `scikit-learn 1.9.0`) — **used only as
the numerical EM optimizer on an already-frozen, already-causal training window**; see §C.4 for
why the resulting fitted object's own `.predict_proba` is never used downstream. EM only finds a
**local** optimum and Gaussian-HMM likelihood surfaces are well known to have multiple local
optima and a label-switching symmetry (permuting state indices leaves the likelihood invariant);
the module (`fit_hmm_on_window`) runs several random restarts (default 5) and keeps the
highest-log-likelihood fit, then labels states *post hoc* by sorted fitted mean: the **bull**
state is the state with the highest $\hat\mu_k$, **bear** the lowest.

### C.4 The causality problem: filtering vs. smoothing, and why it matters here

This is the single most important design decision on this axis, and it is exactly the shape of
bug the program's Independent Verify phase exists to catch. The $\gamma_t(k)$ posteriors used
inside EM are **smoothed**: $\gamma_t$ conditions on the *entire* training window $r_{1:T}$,
including observations strictly after $t$. If one were to fit an HMM once on the full historical
sample and then read off $\gamma_t$ at every $t$ as "the regime probability on day $t$," that
value would be computed in part from information the strategy could not have known at the time —
a textbook lookahead bug, mathematically identical in spirit to using `close[t]` instead of
`close[t-1]` in a trend filter, just less visually obvious because it is hidden inside a
forward-backward recursion instead of a `.shift()` call.

The correct causal quantity is the **filtered, one-step-ahead predictive distribution**:

$$ P(S_t = k \mid r_{1:t-1}, \theta) \;=\; \sum_i P(S_{t-1}=i \mid r_{1:t-1},\theta)\, A_{ik} $$

i.e. take the forward-filtered belief as of the *end of the previous day* and propagate it one
step through the transition matrix — using *no* information from day $t$ itself. This is exactly
the causal analogue of "SMA computed through close[t-1], compared against close[t-1]" used
elsewhere in this program (`s1_sma_band_state` in `research/indices/i1_pipeline.py`): both decide
at close of $t-1$, act at $t$.

**Investigation of hmmlearn's own API**: hmmlearn's `predict_proba`/`score_samples` compute
smoothed $\gamma_t$ via forward-backward over whatever array is passed in. One subtlety worth
recording: because $\beta_T(k) \equiv 1$ by definition, the smoothed posterior at the **last** row
of a passed array coincides exactly with the pure forward-filtered posterior at that row — so in
principle, calling `predict_proba` on a repeatedly-truncated growing prefix and reading only the
final row would also be causally correct. This module deliberately does **not** rely on that
API-internals coincidence: it is $O(T^2)$ if computed per-day rather than in closed form, it
depends on hmmlearn's internal smoothing implementation never changing evaluation order, and — the
decisive reason — it is far less legible to an independent verifier than an explicit forward
recursion written in the module itself. The module therefore implements the forward algorithm
directly (`_forward_filter`, ~25 lines, pure NumPy), and uses hmmlearn **exclusively** for the
EM parameter-fitting step. This directly answers the task brief's investigation question: yes,
hmmlearn's API *can* be coaxed into a causal computation via the terminal-row coincidence, but a
from-scratch forward algorithm was chosen instead for auditability.

### C.5 Refit schedule: strictly causal, expanding window, two frequencies swept

Parameters are re-estimated periodically, never continuously, both for computational tractability
and because daily re-estimation of a 2-6 parameter HMM on near-identical data offers essentially
no informational benefit over the persistence horizon implied by realistic transition
probabilities. Concretely (`causal_hmm_regime_posterior`):

- **Anchors**: refit at trading-day indices $a \in \{$`min_train`$,\,$`min_train`$+$`refit_every`$,\dots\}$
  (`min_train = 252`, i.e. one trading year of history required before the model produces any
  signal at all — a point-in-time gate exactly parallel to the sma200 gate: a name does not enter
  the regime-gated book until this is satisfied).
- **Training window**: `window_type="expanding"` (default) uses $r_{0:a}$ (all history through the
  anchor, growing every refit — the online analogue of Hamilton's full-sample MLE spirit, and the
  natural choice for a stationary-parameter Markov-switching model where more data strictly helps
  the transition-matrix estimate, historically the noisiest part of an HMM fit at short samples).
  A `window_type="rolling"` alternative (fixed trailing width, default 1260 obs ≈ 5y) is
  implemented but not the default — see §C.6 for the considered-and-rejected reasoning.
- **Live segment**: for each segment $[a, b)$ between consecutive anchors, the model computes a
  fresh forward pass *over the training window itself* (using the just-fit parameters) to obtain
  $\text{filtered}_{a-1}$ — this is legitimate information (entirely history through $a-1$) and
  seeds the live forward recursion for days $a, a+1, \dots, b-1$, each of which uses only
  parameters fit on data through $a-1$ and returns observed through $t-1$.
- **Refit frequency swept**: $\{63, 252\}$ trading days (quarterly vs. annual). Quarterly adapts
  faster to genuine drift in regime dynamics (e.g. the volatility-of-volatility regime shift
  post-2020); annual is more stable and less prone to over-fitting a short recent window's noise
  into the transition matrix. Both settings are fully causal by construction — the choice is a
  bias/variance tradeoff on the parameters themselves, not a causality question.

### C.6 Tradeoffs, and alternatives considered and rejected

**Why 2-state and 3-state, not more.** The motivating problem (Fact 4) is fundamentally
bull-vs.-bear: a 2-state model is the minimal object that can express it. A 3rd state tests
whether an intermediate "high-vol chop/crisis" state (distinguishable from a persistent bear
trend) improves the fit around known crisis windows without over-parameterizing. States beyond 3
were not attempted for Stage A: many of the 93-name universe's members have short real histories
(some <2 years per `research/equity/DATA.md`'s manifest), and a $K_C$-state Gaussian HMM has
$O(K_C^2)$ transition parameters plus $2K_C$ emission parameters — with `min_train=252`
observations, a 3-state fit is already at the edge of reliable estimation for short-history names;
a practical caveat for the Sweep phase is that **3-state HMM should be restricted to long-history
names** (SPY/QQQ/mega-cap tech-scale histories), falling back to 2-state or the SMA baseline for
young listings (see Section 9 of this document for how this per-name fallback threads through the
frozen Stage-A grid without changing its cell count). The number of states was fixed
*exogenously* (2 or 3), not selected in-sample via AIC/BIC, specifically to avoid a second layer
of look-ahead-adjacent overfitting (choosing $K_C$ based on which $K_C$ happens to produce the
best-looking in-sample fit is itself a form of data-snooping this program's discipline is designed
to avoid).

**Why Gaussian emission, not Markov-switching AR or MS-GARCH.** Hamilton's original model and its
many descendants often include an autoregressive term in the state-conditional mean, and a large
separate literature (Markov-Switching GARCH, e.g. Gray 1996's "collapsing" approximate filter,
since MS-GARCH's conditional variance depends on the entire unobserved state history and has no
closed-form filter) lets the *variance* process itself be regime-dependent AND
volatility-clustered within a regime. Both were considered and rejected for Stage A: they
substantially complicate the E-step (no closed-form EM update; approximate/particle filtering
required) and — more importantly — would confound the specific effect this axis is chartered to
isolate, namely *whether a smooth causal regime probability beats a binary SMA filter as an
exposure-scaling signal*, not *what is the single best possible model of equity return dynamics*.
Adding autoregressive or GARCH structure is a legitimate follow-on refinement for Stage B if the
plain Gaussian HMM proves promising in the Sweep phase, not a Stage-A scope item.

**Why expanding window by default, not rolling.** An expanding window uses strictly more data at
every refit, which for a 2-6 parameter model with a notoriously noisy transition-matrix estimate
(financial regimes are persistent, so transition probabilities are estimated from very few
effective "switch" events in any finite sample) is generally variance-reducing. The rejected
alternative (rolling window) trades that stability for faster adaptation to a genuine
structural change in regime dynamics (e.g., if post-2022 rate regime fundamentally altered
equity-return regime persistence) — plausible, but a second-order effect relative to getting the
causal *mechanics* right first; both are implemented so the Sweep phase can test the tradeoff
directly rather than take this axis's default on faith.

**HMM vs. SMA — the honest cost side.** An HMM is not a free upgrade over a binary filter. It
requires (a) periodic EM re-optimization (5 random restarts × up to 200 EM iterations, ~5-16
seconds per name per refit-window combination observed on real data below — nontrivial at
93-name × multiple grid-cell scale), (b) careful post-hoc state labeling to avoid label-switching
artifacts across refits, (c) a materially harder-to-audit implementation than a `.rolling().mean()`
call, and (d) EM local-optima/convergence noise (`hmmlearn`'s internal "Model is not converging"
warnings were observed on essentially every refit in real-data testing — informational, not fatal:
`n_iter=200, tol=1e-4` reliably finds a fit whose likelihood has stabilized to a few parts in
$10^4$, and this axis's own injection test confirms the resulting posteriors behave exactly as a
causal filter should regardless of this warning noise, but a later integration agent should not be
alarmed by it appearing in logs). Its payoff, if any, is the *smoothness* and the fact that it can
in principle react to a level-and-variance regime break using one or two extreme observations
rather than needing a slow-moving window to accumulate evidence — the empirical question of
whether that payoff survives costs and turns into better CAGR-at-a-DD-budget is explicitly left to
the Sweep phase; this axis's mandate stops at delivering a validated, causal, comparable-to-the-
null signal.

### C.7 Empirical validation on real data

All fits below use `covariance_type="diag"` (equivalent to a plain scalar variance per state for
this univariate case), `n_restarts=5`, `random_state=0`, on **daily log returns built from
`source=='duka'`-filtered closes via the `daily_close_series_duka` convention** (bug pattern 1
respected).

**Causality proof (synthetic-outlier injection, on real SPY log returns, $T=1992$).** A shock of
$+0.5$ (50 log-points) was injected at index 1982 (10 trading days before the end, safely inside
the final refit segment so the fitted parameters are bit-identical between the base and shocked
runs — isolating pure forward-filter propagation as the only possible source of divergence):

| Config | first divergence (causal) | first divergence (naive same-day control) | max \|diff\| at/before injection (causal) | PASS |
|---|---|---|---|---|
| 2-state, refit=63 | index **1983** (= inj+1) | index **1982** (= inj, same day) | **0.0** (bit-identical) | **True** |
| 3-state, refit=252 | index **1983** (= inj+1) | index **1982** (= inj, same day) | **0.0** (bit-identical) | **True** |

The causal predictive signal is provably unaffected by the shock at or before the injection day,
and first reacts on exactly the correct day (the first day whose predictive distribution
conditions on the shocked observation). The deliberately-wrong same-day control
(`_naive_same_day_pbull`, which reads off the *filtered* rather than *predictive* array — the
literal off-by-one lookahead bug) reacts on the injection day itself, proving the test is not a
tautology that any implementation, broken or not, would pass.

**Real-name fits (2-state, refit=63, min_train=252), 5 names from the 93-name universe:**

| Symbol | # returns | fit time | valid p_bull rows | mean p_bull |
|---|---|---|---|---|
| SPYUSDT | 1992 | 16.2s | 1740 | 0.909 |
| QQQUSDT | 1991 | 13.5s | 1739 | 0.751 |
| AAPLUSDT | 1992 | 6.6s | 1740 | 0.935 |
| MSFTUSDT | 2001 | 6.4s | 1749 | 0.821 |
| NVDAUSDT | 1986 | 5.3s | 1734 | 0.673 |

Mean p_bull well above 0.5 for every name is consistent with Established Fact (6): the 2018-2026
sample is one exceptional secular bull regime, so a bull-labeled state dominating the posterior
most of the time is the historically correct answer, not a model failure.

**Sanity check vs. the 200d-SMA binary baseline (SPY, both signals decide-at-close/act-next-day):**

| Window | n days | HMM p_bull mean (min) | SMA gate mean | SMA flips |
|---|---|---|---|---|
| 2020-03 COVID crash (2020-02-15 to 2020-04-15) | 41 | 0.622 (0.492) | 0.268 | 3 |
| 2022 drawdown (2022-01-01 to 2022-10-15) | 198 | 0.763 (0.114) | 0.212 | 3 |

The SMA gate spends most of both windows flat (mean 0.21-0.27) with several discrete flips
(whipsaw risk each one carries the full round-trip transaction cost per Established Fact/cost
convention); the HMM's p_bull dips meaningfully (down to 0.11-0.49) without ever fully
binarizing, i.e. it expresses graded uncertainty rather than an all-or-nothing switch — exactly
the qualitative behavior this axis was chartered to produce. A supplementary (non-required, but
directly tied to Fact (4)'s own named example) look at 2025-04-09 on SPY shows the mechanism
concretely: after the 2025-04-03 (-4.89%) and 2025-04-04 (-6.13%) tariff-shock returns, p_bull
collapsed from 0.968 to 0.048 within a single refit-free forward-filter step — a sharp,
level-driven regime reassessment triggered by two extreme return observations, in contrast to a
20-day trailing-vol estimator that (per Fact 4) remained "an unremarkable ~2%" that same morning.
This is a sanity check only, not a claim of profitability; the pre-existing SMA state was already
flat throughout this particular window so no like-for-like speed comparison is possible in this
specific sub-episode, and turning this qualitative observation into vindicated P&L is exactly the
Sweep phase's job.

### C.8 Chosen variants for the Stage-A sweep

Precisely parameterized, five grid points contributed by this axis:

1. **2-state Gaussian HMM, refit every 63 trading days** (expanding window, min_train=252,
   n_restarts=5) — fast-adapting variant. **(Recommended baseline for the Stage-A OAT design in
   Section 9/10 of this document.)**
2. **2-state Gaussian HMM, refit every 252 trading days** (otherwise identical) — slow/stable
   variant.
3. **3-state Gaussian HMM, refit every 63 trading days** (otherwise identical) — fast-adapting,
   with an explicit neutral/chop state; restrict to long-history names per §C.6.
4. **3-state Gaussian HMM, refit every 252 trading days** (otherwise identical) — slow/stable,
   3-state.
5. **200-day-SMA ±2% hysteresis binary gate** (`sma_binary_gate`, decide-at-close/act-next-day) —
   the explicit null/baseline this axis is chartered to beat; a refit-frequency dimension does not
   apply to this fixed-parameter rule (there is nothing to re-estimate), so it contributes exactly
   one grid point, not two.

Chosen over (and why rejected, in addition to §C.6's discussion): a single arbitrarily-picked
refit frequency (rejected — the brief explicitly asks for 1-2 settings and a single choice would
under-explore an axis this program has already flagged, via Fact (5), as prone to
single-timeframe fragility); a rolling-window default (rejected as the *default*, per §C.6, but
left implemented and available); MS-GARCH / Markov-switching-AR (rejected as out of Stage-A scope,
§C.6); state-count selection via in-sample AIC/BIC (rejected as a second data-snooping vector).

### C.9 Axis C module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/regime_hmm.py` (committed
  `ac8bb0b` on branch `equity-cta`). Validation scripts:
  `research/cta/lib/_validate_regime_hmm.py` (writes
  `research/cta/lib/validate_regime_hmm_results.json`) and
  `research/cta/lib/_check_20250409.py`.
- **Grid points declared:** exactly 5 — `{2-state,refit=63}`, `{2-state,refit=252}`,
  `{3-state,refit=63}`, `{3-state,refit=252}`, `{200d-SMA ±2% binary baseline}` (fixed-parameter,
  counted once). All HMM cells: expanding-window, `min_train=252`, `n_restarts=5`,
  `random_state=0`, `covariance_type=diag`.
- **Function signatures:**
```
def fit_hmm_on_window(train_returns: np.ndarray, n_states: int, n_restarts: int = 5, random_state: int = 0) -> HMMParams
def causal_hmm_regime_posterior(returns: pd.Series, n_states: int = 2, refit_every: int = 63, min_train: int = 252, window_type: str = "expanding", rolling_window: int = 1260, n_restarts: int = 5, random_state: int = 0) -> pd.DataFrame  # columns: p_state_0..p_state_{K-1}, p_bull, p_bear, state_argmax, refit_date
def sma_binary_gate(close: pd.Series, window: int = 200, band: float = 0.02) -> pd.Series
def exposure_scale_from_posterior(p_bull: pd.Series, floor: float = 0.0, cap: float = 1.0) -> pd.Series
def injection_test(returns: pd.Series, injection_idx: int | None = None, shock: float = 0.5, **kwargs) -> dict
```
- **Injection test result:** 2-state (refit=63) and 3-state (refit=252) both PASS on real SPY log
  returns (T=1992, shock=+0.5 injected at index 1982): causal predictive p_bull is bit-identical
  (max abs diff = 0.0) between base and shocked runs for every index <= 1982, and first diverges
  at exactly index 1983 (injection_idx + 1) in both configs.
- **Packages installed:** `hmmlearn==0.3.3` (plus its pulled dependencies:
  `scikit-learn==1.9.0`, `joblib==1.5.3`, `narwhals==2.23.0`, `threadpoolctl==3.6.0`), installed
  into the shared venv via `/home/speculation/convexity/.venv/bin/python -m pip install hmmlearn`.
  numpy 2.5.0 and scipy 1.18.0 were already present and satisfied hmmlearn's requirements without
  upgrade.
## Axis D — Volatility Forecasting for Position Sizing

### D.0 Motivation and evaluation philosophy

Every construction in this program sizes exposure as (roughly) `target_vol / forecast_vol`,
capped. The forecast_vol input therefore acts multiplicatively on every trade this system will
ever place — a biased or noisy vol forecast degrades sizing quality regardless of how good the
underlying entry/exit signal (Axes A–C) or portfolio construction (Axis F) is. Trailing realized
volatility (a rolling standard deviation of past returns) is the default in most of this
program's precedent work (I1, the exploratory CTA scans) and is a defensible null, but it is a
*backward-looking, equal-weighted, close-to-close-only* estimator: it discards intraday range
information, weights a return from 19 days ago identically to yesterday's, and — critically, per
established fact (4) above — is reactive rather than predictive, failing exactly at regime
transitions.

This axis asks whether four more structured alternatives — Yang-Zhang range estimation, EWMA
exponential smoothing, GARCH(1,1) conditional heteroskedasticity, and HAR-RV multi-horizon
realized-variance regression — improve on trailing RV by a **direct, portfolio-agnostic**
criterion before ever touching a backtest P&L: does the forecast made at the close of day `t-1`
actually predict day `t`'s realized (squared) return better? This ordering matters. A vol
estimator that "helps Sharpe" in a backtest is confounded with everything else in that backtest
(entry timing, cost model, universe composition); a vol estimator that reduces QLIKE/MSE against
the realized-return target is validated on its own terms, in isolation, exactly as the volatility
forecasting literature evaluates it (Patton (2011), Hansen & Lunde (2005)). Only after this
direct evaluation does the module hand off a sizing-ready output for later portfolio-level testing
in the Sweep phase.

### D.1 Data: daily OHLC and intraday realized variance from 1-minute duka bars

The stitched lake carries only 1-minute bars; every quantity below is derived. Per the DATA
CONTRACT, `df[df.source=='duka']` is applied before anything else (bug pattern (a)), and daily
bars are aggregated per real calendar day (`ts.dt.floor("D")`, identical grouping to
`i1_pipeline.daily_close_series_duka`):

```
open  = first duka bar's open that day
high  = max(high) across all duka bars that day
low   = min(low) across all duka bars that day
close = last duka bar's close that day
```

**Verification** (see `aggregate_daily_ohlc_from_duka` / validation run): on SPYUSDT, QQQUSDT, and
AAPLUSDT, the resulting `close` column matches `i1_pipeline.daily_close_series_duka`'s
independently-derived close series to floating-point exactness (max abs diff `0.0000000000` across
1,969–1,993 real trading days per name), and the resulting price levels are visually sane (SPY
$221–$695, QQQ $143–$637, AAPL $107–$507 over 2018-05 to 2026-04, consistent with real historical
ranges). This is the same daily-close construction already trusted by I1 and the CTA scans, now
independently re-derived from OHLC and cross-checked byte-for-byte — no daily-aggregation bug.

For HAR-RV, a genuine **intraday realized variance** is also built (not merely squared daily
returns): `RV_t = sum` of squared log returns of 5-minute-subsampled duka closes, strictly within
day `t` (day boundaries never crossed, so the overnight gap never contaminates this measure — it
is exactly what Yang-Zhang's separate overnight term is for instead). 5-minute subsampling is the
literature-standard choice (Andersen, Bollerslev, Diebold & Labys (2001); Corsi (2009) uses 5-
minute FX returns) to damp bid-ask-bounce/microstructure noise that a raw 1-minute sum would
otherwise inflate; 1-minute sampling is offered as an alternate but not used in the frozen grid
below (kept as a documented, not-yet-swept option).

### D.2 Causality convention

Every `*_forecast` function returns a series where `forecast[t]` is usable for trading on day `t`
and is computed **strictly from information through the close of day t-1** — the same
decide-at-close/act-next-day discipline as bug pattern (c). Concretely this means: compute the
estimator's own as-of-close-of-day-`t` value from the raw series, then apply a one-step
`.shift(1)` (or an equivalent explicit recursion, for GARCH/HAR-RV where refitting makes a naive
`.shift` insufficient) before returning. This is proven, not assumed, by the injection-test suite
in §D.7.

### D.3 Yang-Zhang (2000) OHLC range-based estimator

**The bias problem it solves.** Close-to-close realized vol ignores the overnight gap and all
intraday range information — informationally wasteful (Parkinson (1980) showed a pure high-low
range estimator is ~5x more statistically efficient per observation than close-to-close for a
pure diffusion). But naive range estimators (Parkinson; Garman & Klass (1980)) assume the price is
a continuous driftless diffusion with `O_t = C_{t-1}` — false whenever there is an overnight gap
(routine for equities) or genuine drift, and both biases matter here since every name in the
universe is a real equity with real overnight jumps.

**Rogers-Satchell (1991)** fixes the *drift* bias (not the gap bias) via a construction that is
provably drift-independent for the continuous open-to-close variance:

```
rs_t = ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t)
```

Under a driftless GBM, `E[rs_t] = sigma^2 * dt` exactly; the drift-independence follows because
each product term is a difference of two log-ranges from the SAME endpoint (`O_t` or `C_t`), which
cancels the drift's linear contribution to each range's expectation while preserving the diffusive
quadratic-variation contribution (see Rogers & Satchell (1991) for the full proof via a
reflection-principle argument on Brownian bridges). Rogers-Satchell alone, however, still ignores
the overnight gap.

**Yang-Zhang (2000)** explicitly separates the overnight and open-to-close variance and combines
them with the (already-unbiased) Rogers-Satchell open-to-close term, which is the key structural
idea: it does NOT try to make one range statistic do both jobs.

Define, for day `t`, using `O_t, H_t, L_t, C_t` and the prior close `C_{t-1}`:
```
o_t = ln(O_t / C_{t-1})     (overnight/close-to-open return)
c_t = ln(C_t / O_t)         (open-to-close return)
```
Over a trailing window of `n` days:
```
V_o  = (1/(n-1)) * sum (o_i - obar)^2          [overnight variance, sample var of o]
V_c  = (1/(n-1)) * sum (c_i - cbar)^2          [open-to-close variance, sample var of c]
V_rs = (1/n)     * sum rs_i                     [Rogers-Satchell mean, already unbiased]

V_YZ = V_o + k_yz * V_c + (1-k_yz) * V_rs
```
(the Yang-Zhang blend weight is written `k_yz` here to avoid collision with Axis B's
curvature `k = γσ²` and Axis E's variance-vs-cost curvature — same clash-avoidance
convention as the rest of this document) with the closed-form minimum-variance weight
```
k_yz = 0.34 / (1.34 + (n+1)/(n-1))     [-> 0.34/1.34 ~ 0.2537 as n -> infinity]
```
(Yang & Zhang derive `k_yz` by minimizing `Var(V_YZ)` under an assumed model where overnight and
open-to-close returns are independent with variance ratio governed by a free parameter `alpha`,
and their headline choice `alpha=1.34` gives this specific closed form — reused verbatim here
rather than re-derived, as re-deriving the optimal-`alpha` calibration from scratch is a research
program in itself and the headline `k_yz` is the value the original paper recommends and the one
consistently used in subsequent literature and practitioner implementations.) `V_YZ` is unbiased
under BOTH nonzero drift (inherited from Rogers-Satchell) and opening jumps (inherited from
explicitly modeling `V_o`) — the two failure modes that make close-to-close and naive range
estimators each separately wrong for real equity data with real overnight gaps.

**Tradeoff considered and accepted**: Yang-Zhang needs OHLC, unavailable pre-aggregation here and
requiring the verification step above; its main advantage over EWMA/trailing-RV is using strictly
more information PER DAY (4 prices vs. 1), which should reduce estimation noise at a fixed
window length — this is exactly the efficiency gain the empirical test in §D.6 checks for
directly, and (spoiler) does not clearly materialize net of estimation noise at daily granularity
on this data.

### D.4 EWMA / RiskMetrics exponential smoothing

The RiskMetrics (J.P. Morgan (1996)) variance recursion:
```
sigma2_t = lambda_ewma * sigma2_{t-1} + (1-lambda_ewma) * r_{t-1}^2
```
(`lambda_ewma` here is Axis D's own EWMA decay for the *variance* recursion — a distinct
calibration from Axis A's `λ` used for the *level/momentum* EWMA family; see this document's
notation-reconciliation table) is an IGARCH(1,1) special case (persistence forced to exactly 1,
zero constant `omega`) — it can be derived as the population EWMA of squared returns,
`sigma2_t = (1-lambda_ewma) * sum_{k>=1} lambda_ewma^{k-1} r_{t-k}^2`, an infinite-memory but
exponentially-decaying-weight estimator. Unlike a rolling-window trailing RV (which weights the
last `n` days equally and the `n+1`-th day zero), EWMA weights recent returns more and old returns
smoothly less, with an effective memory `~1/(1-lambda_ewma)` trading days (`lambda_ewma=0.94`
implies ~17 days; `lambda_ewma=0.97` implies ~33 days) — the two grid points bracket a
fast-adapting vs. slow-adapting design, directly analogous to the 20d/60d trailing-RV windows.

**Why it might beat both trailing-RV and GARCH here**: EWMA responds to a vol shock immediately
(the very next observation moves `sigma2` a nontrivial amount, unlike a rolling window which only
responds once the window has enough new high-vol days to move the average, and doesn't suffer
trailing-RV's *sharp dropout* problem — a huge return from `n` days ago disappearing from a
rolling window abruptly deflates trailing RV even though nothing informative happened today).
Against GARCH, EWMA has zero estimation risk (no MLE, no possible convergence failure, no
overfitting a mean-reversion parameter `omega/(1-alpha-beta)` on a finite, possibly
regime-non-stationary sample) at the cost of not modeling mean-reverting vol persistence
explicitly.

### D.5 GARCH(1,1) — fit strictly causally

Bollerslev (1986)'s GARCH(1,1), generalizing Engle (1982)'s ARCH, models conditional variance as
```
sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1},    omega>0, alpha,beta>=0, alpha+beta<1
```
i.e. EWMA plus a constant (`omega`) that pulls the forecast toward a long-run unconditional
variance `omega/(1-alpha-beta)` — the mean-reversion EWMA lacks. This should, in principle, be a
strict improvement over EWMA when vol genuinely mean-reverts (rather than following a pure
random-walk-in-variance process, which is what un-mean-reverting EWMA implicitly assumes).

**Causal fitting discipline (the part most naive GARCH backtests get wrong)**: refitting the
`(omega, alpha, beta)` MLE via `arch.arch_model` every single day, across a 93-name universe over
an 8-year sample, is both needlessly expensive and not obviously any more correct than a periodic
refit for a ONE-STEP-AHEAD forecast — so this module refits only every `refit_freq` trading days
(using only returns through the prior day's close, `window` trailing days if given, else the full
expanding history), and between refits advances the one-step forecast day-by-day using the FIXED
last-fitted parameters via the exact recursion above, fed with the REAL, already-realized daily
return (known by day `t`, hence still fully causal). This is standard practice for rolling GARCH
forecast evaluation and is exactly as causal as a daily refit for the one-step-ahead object, while
being computationally tractable at universe scale.

Two grid points bracket adaptation speed/estimation-window tradeoffs: `refit_freq=21` (~monthly)
with a 756-day (~3y) rolling window (fast-adapting to structural change, but noisier MLE from
less data), vs. `refit_freq=63` (~quarterly) with an expanding window (slower-adapting, but a
more data-efficient, lower-variance MLE).

**Tradeoff accepted**: GARCH is a genuinely more sophisticated model of the vol *process*
(explicit mean reversion, explicit persistence parameter) but introduces real estimation risk —
MLE convergence, parameter instability on a young/short-history name, misspecification if the true
process isn't GARCH(1,1) — that EWMA and trailing-RV are structurally immune to. Whether this
sophistication pays off is exactly what §D.6 tests directly, without any P&L confound.

### D.6 HAR-RV (Corsi 2009) — heterogeneous-horizon realized-variance regression

Corsi (2009) observes that realized volatility exhibits apparent long memory (slowly-decaying
autocorrelation) that a short-memory model like GARCH(1,1) cannot easily capture, without going to
a full (harder to estimate, harder to keep causal) fractionally-integrated ARFIMA model. HAR-RV's
parsimonious fix: regress tomorrow's realized variance on THREE realized-variance components at
different horizons —
```
RV_{t+1} = c + b_d * RV_t^(d) + b_w * RV_t^(w) + b_m * RV_t^(m) + eps_{t+1}
RV_t^(d) = RV_t                                  (today)
RV_t^(w) = mean(RV_{t-4},...,RV_t)               (trailing 5 trading days, incl. today)
RV_t^(m) = mean(RV_{t-21},...,RV_t)              (trailing 22 trading days, incl. today)
```
The heterogeneity of the three components (daily/weekly/monthly agents with different horizons,
in Corsi's original economic interpretation) mimics a long-memory-like autocorrelation decay from
a simple, easily-estimated linear-in-parameters model — a genuinely elegant piece of applied
econometrics, and one that is only usable here because the underlying stitched lake provides
genuine 1-minute intraday data to build `RV_t` from (§D.1) rather than only close-to-close daily
data, which would degenerate `RV_t` to a single squared return and defeat the entire point of the
model.

**A real implementation bug found and fixed during this axis's own validation (reported plainly,
per house discipline).** A first, textbook level-OLS implementation of HAR-RV was built and
tested on real SPY/QQQ/AAPL/MSFT/JPM data; it produced QLIKE values in the **hundreds of
millions** on a nontrivial share of days. Diagnosis: plain OLS on RV levels has no constraint
keeping the fitted hyperplane's output positive; during quiet, low-variance stretches (small,
near-collinear `d/w/m` inputs) the regression occasionally predicts a near-zero or slightly
negative RV, which — after the mechanically necessary floor at a small epsilon to keep `sqrt`
defined — produces a variance forecast many orders of magnitude below the realized proxy. QLIKE's
`rv/sigma2` term is *unbounded* as `sigma2 -> 0`, so a handful of such days dominate the mean loss
and mask everything else. **Fix**: regress in log space — `log(RV_{t+1})` on
`(log(RV_t^(d)), log(RV_t^(w)), log(RV_t^(m)))` — and exponentiate the prediction
(`forecast = exp(predicted_log_RV)`), which is strictly positive by construction and needs no
artificial floor. This is not an invented workaround: log-RV is closer to conditionally Gaussian
than heavily right-skewed raw RV, and log-space HAR regressions are a standard, literature-
documented refinement for exactly this reason. Post-fix, HAR-RV's QLIKE on real data falls from
the hundred-million range into the same `~1.5-6` range as every other estimator (see §D.7) — a
genuine correctness fix, verified on real data, not a tuning choice.

**Causality subtlety (the second easy-to-get-wrong step, also verified by injection test)**: when
about to forecast `RV_i` using `feat_{i-1}` (features through day `i-1`), the *training set* used
to fit `(c, b_d, b_w, b_m)` must consist only of pairs `(feat_t, RV_{t+1})` with `t+1 <= i-1`, i.e.
`train_feat = feat.iloc[:i-1]`, NOT `feat.iloc[:i]`. Using `feat.iloc[:i]` silently leaks `RV_i`
(the very value about to be predicted) into its own training set via the pair `(feat_{i-1}, RV_i)`
— a one-off-by-one lookahead that produces no obvious symptom (the model still "runs"; it is just
quietly cheating) and exists precisely to be the kind of bug caught by a synthetic-injection test
rather than eyeballing output. `_har_rv_NAIVE_UNLAGGED` implements the buggy version deliberately,
for the injection-test harness (§D.7) to prove the test discriminates it.

Two grid points: `refit_freq=21` (~monthly, matching GARCH's fast setting) and `refit_freq=63`
(~quarterly), both at 5-minute intraday RV sampling.

### D.7 Injection-test validation (causality proof + naive-control-has-teeth proof)

Methodology: inject a large synthetic shock (8x multiplicative) into a COPY of real SPYUSDT daily
data at a single mid-sample day index `j` (OHLC close for trailing-RV/Yang-Zhang/EWMA; the return
series directly for GARCH; the realized-variance series directly for HAR-RV), then confirm:
(a) the CAUSAL forecast at index `j` is byte-identical before/after the injection (no lookahead);
(b) the CAUSAL forecast at index `j+1` DOES change (the estimator reacts, exactly one day late);
(c) a deliberately-broken NAIVE/unlagged control version of the SAME estimator DOES react at
index `j` itself — proving the test has genuine discriminating power and is not a tautology that
would "pass" any implementation.

Result, on real SPYUSDT daily data (`run_injection_test_ohlc_based`, `run_injection_test_garch`,
`run_injection_test_har_rv`):

| estimator | forecast[j] unchanged (causal) | forecast[j+1] changed (causal) | naive control reacts at [j] | PASS |
|---|---|---|---|---|
| trailing_rv_20 | True | True | True | **True** |
| yang_zhang_20 | True | True | True | **True** |
| ewma_0.94 | True | True | True | **True** |
| garch11 (rf=21,w=756) | True | True | n/a (no naive-GARCH control defined; see module docstring) | **True** |
| har_rv (rf=21) | True | True | True | **True** |

All five estimator families pass; all four naive controls that were implemented (trailing-RV,
Yang-Zhang, EWMA, HAR-RV) confirm real discriminating power. (GARCH has no naive-unlagged control
implemented — "unlagged GARCH" is not a standard well-defined object the way "unshifted rolling
std" is — but the other four controls already establish the test methodology has teeth, and
GARCH's own causal test independently confirms `forecast[j]` unchanged / `forecast[j+1]` changed.)

### D.8 Forecast-quality evaluation: QLIKE / MSE vs. realized next-day squared return

Per-day loss, `sigma2_t = forecast_vol[t]^2`, `rv_t = realized_return[t]^2` (both aligned at day
`t`, `forecast_vol[t]` already causal per §D.2):
```
QLIKE_t = rv_t/sigma2_t - ln(rv_t/sigma2_t) - 1     (Patton (2011)'s "robust" loss under a noisy
                                                       vol proxy; minimized at rv_t == sigma2_t;
                                                       asymmetrically penalizes UNDER-forecasting
                                                       more than MSE does in relative terms —
                                                       the economically relevant failure mode for
                                                       a risk-sizing input, since understating vol
                                                       is what produces an oversized, under-margined
                                                       position)
MSE_t  = (rv_t - sigma2_t)^2
```
Evaluated on real SPYUSDT, QQQUSDT, AAPLUSDT, MSFTUSDT, JPMUSDT daily data (2018-05 through
2026-04, ~1,930-2,020 valid obs per name per estimator after each estimator's own warm-up), mean
across the 5 names, unweighted:

| estimator | mean QLIKE | mean MSE |
|---|---|---|
| trailing_rv_20 | 2.1283 | 3.570e-04 |
| trailing_rv_60 | 2.0870 | 3.531e-04 |
| yang_zhang_20 | 2.0992 | 3.582e-04 |
| yang_zhang_60 | 2.1158 | 3.534e-04 |
| **ewma_0.94** | **1.9342** | 3.473e-04 |
| **ewma_0.97** | 1.9546 | **3.421e-04** |
| garch11 (rf=21, w=756) | 2.2230 | 4.140e-04 |
| garch11 (rf=63, w=None) | 2.3206 | 4.128e-04 |
| har_rv (rf=21) | 2.5906 | 3.471e-04 |
| har_rv (rf=63) | 2.6106 | 3.471e-04 |

**Winner by both metrics: EWMA.** `ewma_0.94` wins mean QLIKE; `ewma_0.97` wins mean MSE; both
outperform trailing realized vol (the established baseline/null), Yang-Zhang, GARCH(1,1), and
HAR-RV on this direct, portfolio-agnostic evaluation. This is a genuine, somewhat humbling
finding, reported plainly per house discipline: neither GARCH's explicit mean-reversion modeling
nor HAR-RV's multi-horizon structure nor Yang-Zhang's richer per-day information beats simple
exponential smoothing of squared returns on *this* data, at daily granularity, on real equities.

A more granular look tempers a purely one-line "EWMA wins" reading — per-name QLIKE is not
uniform. GARCH(1,1) is in fact the single best estimator on SPYUSDT (1.4802 vs EWMA's 1.5545) and
QQQUSDT (1.4578 vs 1.5317) — its explicit mean-reversion helps most on the two broad, highly
diversified index-like names, where vol genuinely mean-reverts cleanly around a stable long-run
level. But GARCH is markedly WORSE than every other estimator on AAPLUSDT specifically (QLIKE
5.01-5.51 vs. EWMA's 3.36-3.40) — a single-name idiosyncratic-vol process with sharper, less
mean-reverting jumps (earnings, product events) that a GARCH(1,1) fit on daily returns
misspecifies badly enough to drag its AGGREGATE mean above every simpler alternative. Yang-Zhang
is a consistent, small improvement over trailing-RV at the SAME window (20d: 2.0992 vs. 2.1283;
though NOT at 60d, where it is roughly a wash) — the extra per-day OHLC information helps a little
but not enough to close the gap with EWMA's superior recency-weighting. HAR-RV, even after the
log-space correctness fix, is the weakest of the five families on QLIKE specifically (though
competitive on MSE) — plausibly because a genuinely 3-day-effective HAR structure estimated from
only ~60-90 obs per refit window is itself a noisier fit than a 2-parameter EWMA on this sample
size, even though the underlying long-memory hypothesis HAR-RV encodes is not wrong in principle.

**Implication for sizing**: this is exactly the kind of result the brief's explicit "evaluate
forecast quality FIRST" instruction is designed to surface — a portfolio backtest confound
(GARCH "helping" specifically on SPY/QQQ-heavy portfolio constructions, "hurting" on
single-name-heavy ones) would have been very easy to mistake for a signal-interaction effect
rather than what it actually is: a vol-forecast-quality effect visible directly in QLIKE, before a
single trade is ever simulated.

### D.9 Sizing interface

`vol_target_exposure_magnitude(forecast_vol, target_daily_vol, leverage_target, cap_mult)`
implements the exact functional form already established and validated in this program's prior
work (`i1_pipeline.vol_target_magnitude` / `_classic_cta_exploration.calibrate_target_daily_vol`):
```
mag[t] = min(cap_mult * leverage_target,  leverage_target * target_daily_vol / forecast_vol[t])
```
`forecast_vol` is any of the five causal estimators above (a drop-in replacement for the
established `rv_lagged` input); `target_daily_vol` itself should be calibrated with the
harmonic-mean correction already validated upstream (an arithmetic-mean calibration systematically
overshoots `leverage_target` by Jensen's inequality on the convex map `vol -> 1/vol`) — that
calibration primitive is deliberately NOT reimplemented here (to avoid duplicating an
already-validated piece of shared infrastructure) and is left to the Integration/Sweep phase,
which already has access to it.

### D.10 Chosen variants for the Stage-A sweep, and rejected alternatives

**Implementing all five estimator families, 2 settings each (10 grid points total)** rather than
pre-selecting only the QLIKE/MSE winner, for three reasons: (1) the brief explicitly asks for the
grid to include the established baseline for comparison, not just a winner; (2) §D.8's per-name
breakdown shows the "best" estimator is itself construction-dependent (GARCH wins on
broad/index-like names) — a single Stage-A axis should not foreclose that interaction for the
later portfolio-level Sweep phase, which will test these against a REAL 93-name universe rather
than 5 hand-picked liquid names; (3) this is Stage A of a two-stage program — the adoption
decision belongs to a later, portfolio-outcome-aware gate, not to this axis alone.

**Alternatives considered and explicitly rejected**: (a) a full ARFIMA-RV model for HAR-RV's
long-memory hypothesis — rejected as a materially harder, slower-to-refit-causally estimation
problem for a benefit HAR-RV already captures approximately, and out of scope for a Stage-A axis
module; (b) GJR-GARCH / EGARCH asymmetric-leverage-effect variants — a genuinely interesting
extension (equities exhibit a well-known vol-asymmetry to negative returns) but adds a second
estimation-risk axis on top of the plain-GARCH(1,1) risk already documented above, and the
symmetric GARCH(1,1) already underperforms EWMA here without that extra complexity, so the
asymmetric extension was not prioritized for this Stage-A grid; (c) daily GARCH refitting (no
periodic-refit approximation) — rejected purely on computational-cost grounds at 93-name x 8-year
x multiple grid-point scale, with the periodic-refit-plus-fixed-parameter-recursion design proven
exactly as causal via the injection test; (d) 1-minute (unsampled) intraday RV for HAR-RV —
offered as a documented option in `realized_variance_from_intraday` but not sweep-frozen, in favor
of the literature-standard 5-minute subsampling to control microstructure noise.

**Grid points contributed to the Stage-A sweep ledger (5 estimators x 2 settings = 10 points)**:
1. `trailing_rv_20` — trailing realized vol, 20-day window (established baseline/null)
2. `trailing_rv_60` — trailing realized vol, 60-day window (established baseline/null, slow)
3. `yang_zhang_20` — Yang-Zhang OHLC estimator, 20-day window
4. `yang_zhang_60` — Yang-Zhang OHLC estimator, 60-day window
5. `ewma_0.94` — EWMA/RiskMetrics, lambda_ewma=0.94 (**wins mean QLIKE**)
6. `ewma_0.97` — EWMA/RiskMetrics, lambda_ewma=0.97 (**wins mean MSE**)
7. `garch11_rf21_w756` — GARCH(1,1), refit every 21 trading days, 756-day rolling window
8. `garch11_rf63_wNone` — GARCH(1,1), refit every 63 trading days, expanding window
9. `har_rv_rf21` — HAR-RV (log-space), refit every 21 trading days, 5-min intraday RV
10. `har_rv_rf63` — HAR-RV (log-space), refit every 63 trading days, 5-min intraday RV

**Recommended default for the Integration phase, pending the portfolio-level Sweep**: `ewma_0.94`
— the aggregate QLIKE/MSE winner on this direct evaluation, structurally simplest (no MLE
convergence risk, trivially fast to compute at 93-name scale), and with no observed catastrophic
per-name failure mode (unlike GARCH's AAPL degradation). All ten points remain in the ledger for
the Sweep phase to test against real portfolio outcomes, per this axis's own Stage-A mandate.

### References
- Yang, D. and Zhang, Q. (2000), "Drift-Independent Volatility Estimation Based on High, Low,
  Open, and Close Prices", *Journal of Business*, 73(3).
- Rogers, L.C.G. and Satchell, S.E. (1991), "Estimating Variance From High, Low and Closing
  Prices", *Annals of Applied Probability*, 1(4).
- Parkinson, M. (1980), "The Extreme Value Method for Estimating the Variance of the Rate of
  Return", *Journal of Business*, 53(1).
- Garman, M.B. and Klass, M.J. (1980), "On the Estimation of Security Price Volatilities from
  Historical Data", *Journal of Business*, 53(1).
- J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*, 4th ed. (EWMA/lambda=0.94
  convention).
- Engle, R.F. (1982), "Autoregressive Conditional Heteroscedasticity with Estimates of the
  Variance of United Kingdom Inflation", *Econometrica*, 50(4).
- Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of
  Econometrics*, 31(3).
- Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized Volatility", *Journal of
  Financial Econometrics*, 7(2).
- Andersen, T.G., Bollerslev, T., Diebold, F.X. and Labys, P. (2001), "The Distribution of
  Realized Exchange Rate Volatility", *Journal of the American Statistical Association*, 96(453).
- Patton, A.J. (2011), "Volatility Forecast Comparison Using Imperfect Volatility Proxies",
  *Journal of Econometrics*, 160(1). (QLIKE loss function.)
- Hansen, P.R. and Lunde, A. (2005), "A Forecast Comparison of Volatility Models: Does Anything
  Beat a GARCH(1,1)?", *Journal of Applied Econometrics*, 20(7). (Motivates evaluating GARCH
  against simple alternatives directly, as done here.)

### D.11 Axis D module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/vol_forecast.py`.
- **Grid points declared:** 5 estimator families x 2 parameter settings each = 10 grid points (see
  D.10 for the exact enumeration).
- **Function signatures:**
```
aggregate_daily_ohlc_from_duka(df_duka: pd.DataFrame) -> pd.DataFrame
daily_log_returns(daily_close: pd.Series) -> pd.Series
realized_variance_from_intraday(df_duka: pd.DataFrame, sampling_minutes: int = 5) -> pd.Series
trailing_realized_vol_forecast(daily_close: pd.Series, window: int = 20) -> pd.Series
yang_zhang_variance_asof(daily_ohlc: pd.DataFrame, window: int = 20) -> pd.Series
yang_zhang_vol_forecast(daily_ohlc: pd.DataFrame, window: int = 20) -> pd.Series
ewma_vol_forecast(daily_close: pd.Series, lam: float = 0.94) -> pd.Series
garch11_vol_forecast(daily_close: pd.Series, refit_freq: int = 21, window: int|None = 756, min_obs: int = 250, dist: str = "normal") -> pd.Series
har_rv_features(rv: pd.Series) -> pd.DataFrame
har_rv_vol_forecast(rv: pd.Series, refit_freq: int = 21, min_obs: int = 60) -> pd.Series
evaluate_forecast_quality(forecast_vol: pd.Series, realized_return: pd.Series) -> dict
vol_target_exposure_magnitude(forecast_vol: pd.Series, target_daily_vol: float, leverage_target: float = 1.0, cap_mult: float = 2.0) -> pd.Series
run_injection_test_ohlc_based(daily_ohlc: pd.DataFrame, j: int|None = None) -> dict
run_injection_test_garch(daily_close: pd.Series, j: int|None = None, **kwargs) -> dict
run_injection_test_har_rv(rv: pd.Series, j: int|None = None, **kwargs) -> dict
```
- **Injection test result:** All 5 estimator families PASS on real SPYUSDT daily data (8x
  synthetic price/return/RV shock injected at a mid-sample day index j): causal forecast[j]
  byte-identical before/after injection, causal forecast[j+1] changes exactly one day late —
  confirmed for trailing_rv_20, yang_zhang_20, ewma_0.94, garch11 (refit_freq=21, window=756), and
  har_rv (refit_freq=21). Full JSON results captured in `/tmp/validate_vol_forecast2.log` on
  arb-vm.
- **Packages installed:** `arch==8.0.0` (pulled in `statsmodels==0.14.6` and `patsy==1.0.2` as
  dependencies). numpy 2.5.0 and pandas 3.0.3 already present, unchanged.
## Axis E — Leverage as a Growth-Optimal Control Problem Under Fat Tails and Drawdown Constraints

### E.0 The question this axis actually answers

Leverage choice is not a free parameter to be swept for the highest Sharpe. Established fact (1)
of this program proves Sharpe is exactly leverage-invariant under this cost structure (both the
trading-fee and funding terms are linear in exposure), so **Sharpe cannot discriminate between
leverage levels at all** — every constant-leverage rescaling of a return series has identical
Sharpe by construction. The only economically meaningful leverage objective is therefore a
**growth-rate objective under an explicit risk (ruin/drawdown) constraint**: maximize the
time-average compounded growth rate of wealth, subject to a declared maximum-drawdown or
ruin-probability budget. This is exactly the classical Kelly/Kelly-with-constraint literature,
not a heuristic vol-target overlay, and this section derives it properly against this program's
own real, fat-tailed daily P&L rather than a Gaussian approximation.

Four complementary mathematical constructions are derived and implemented below, in increasing
order of how directly they attack the drawdown/ruin constraint:

1. **Discrete-time empirical Kelly** — the unconstrained growth-optimal leverage $L$, solved
   numerically against the actual historical return distribution (no Gaussian assumption).
2. **Fractional Kelly** — Kelly shrunk by a constant fraction, justified by parameter/estimation
   error and non-stationarity (Thorp; MacLean, Ziemba & Thorp).
3. **Grossman & Zhou (1993)** — the literature's actual derived *state-dependent* growth-optimal
   policy under a hard running-drawdown constraint; exposure is scaled down continuously as
   wealth approaches its own floor, rather than cut only after realized volatility has already
   risen.
4. **EVT/CDaR ceilings** — two complementary, non-parametric-in-the-tail (EVT) and
   convex-optimization (CDaR) constructions that bound the *ruin* probability and the
   *conditional drawdown* directly, used both standalone and as caps composed on top of (1)-(3).

Throughout this section, $L$ (and $L^\*$, $L_{\text{ceiling}}$, $L_{\text{closed\_form}}$) denotes
**leverage**, a scalar multiplier on a return stream, consistently — this is the one symbol in
this document that is used identically and without any cross-axis clash in every place it appears.

### E.1 Discrete-time Kelly against the empirical distribution

**Objective.** For a strategy with i.i.d.-approximated per-period simple returns $r_t$, wealth
under constant leverage $L$ compounds as $W_T = W_0\prod_t (1+Lr_t)$. Maximizing the long-run
growth rate $\frac{1}{T}\log(W_T/W_0)\to E[\log(1+Lr)]$ (by the strong law of large numbers) is
the Kelly criterion (Kelly, 1956; the finance formulation and "fortune's formula" framing is
Thorp, 1971, 2006). This is **not** an approximation of Sharpe maximization — it is a genuinely
different objective, the only one that is not leverage-invariant, because $\log$ is strictly
concave and therefore penalizes the variance/tail-risk of large $L$ nonlinearly, unlike the
linear-in-$L$ Sharpe ratio.

**Why empirical, not Gaussian.** A Gaussian closed form ($L^\*=\mu/\sigma^2$) throws away exactly
the information this program's own established facts say matters most: the real single worst
days (SPY −10.5%, QQQ −10.7%, the 93-name benchmark −12.1%, all on 2020-03-16) are far outside
any Gaussian fit to the bulk of daily returns and are the entire reason a Kelly-style analysis is
being done here in the first place — Grossman & Zhou's own motivation, and this program's fact
(4), is that vol clusters and jumps are exactly what generic risk statistics under-react to.
`kelly_optimal_leverage()` therefore maximizes

$$g(L) = \frac{1}{T}\sum_{t=1}^T \log(1+Lr_t)$$

directly over the **realized historical sample** $\{r_t\}$ (or a bootstrap resample of it),
numerically, via a bounded scalar search. $g$ is concave in $L$ wherever finite (log of an affine
function of $L$), and is $-\infty$ for any $L$ at which some historical day would produce total
ruin ($1+Lr_t\le 0$) — this is the exact, correct mechanism by which a single historical crash
day (a −10.5% day, say) hard-constrains the feasible leverage region without needing a separate
ruin check: Kelly's own objective already refuses to go there. A bounded 1-D concave search
(`scipy.optimize.minimize_scalar`, `method="bounded"`) is therefore both sufficient and exact to
numerical tolerance; no general nonlinear solver or moment-matching is needed.

**Estimation error and fractional Kelly.** Full Kelly is maximally aggressive exactly where
capital is most exposed to estimation error in the input mean, because $g''(L^\*)$ (the curvature
of the growth curve at its own peak) is small — the growth curve is locally flat near $L^\*$, so
small leverage cuts cost little expected growth, while the same-size *overshoot* past $L^\*$ costs
increasingly much (the curve is asymmetric: it is bounded above by $g(L^\*)$ but unbounded below
toward $-\infty$ at the ruin boundary). This is the textbook argument for fractional Kelly
(Thorp's own live-trading practice; MacLean, Ziemba & Thorp (2010) is the standard collected
reference — "half-Kelly" specifically is their headline practical recommendation, trading roughly
25% of the growth rate for a much larger cut in variance and drawdown). It is not merely
prudence — it is the correct response to the fact that $\mu$ (and the tail shape) are themselves
estimated with real sampling error from a finite, non-stationary historical sample, and Kelly's
optimum is a discontinuous function of that error near the ruin boundary. `n_bootstrap`
resampling in `kelly_optimal_leverage()` makes this estimation error explicit and reportable (see
§E.6) rather than asserted.

`fractional_kelly_leverage(L_star, fraction)` implements the shrinkage trivially; it is kept as
an explicit, named function (not inlined) purely so the fraction used anywhere in the sweep is
self-documenting.

### E.2 Grossman & Zhou (1993): the literature-correct realization of "drawdown control"

Grossman & Zhou, *"Optimal Investment Strategies for Controlling Drawdowns"*, Mathematical
Finance 3(3), 1993, solve for the growth-optimal continuous-time portfolio policy under a **hard
constraint that wealth never fall below a fraction $\alpha_{GZ}$ of its own running maximum**:
$W_t \ge \alpha_{GZ} M_t$ for all $t$, where $M_t = \sup_{s\le t}W_s$ is the running peak of
**wealth** (kept as $M_t$, the dominant convention in the Grossman-Zhou literature — this is a
distinct object from Axis A's unrelated $\mathrm{Mom}_\lambda(t)$ notation for a trailing-return
momentum score; see this document's notation-reconciliation table). Under geometric Brownian
dynamics for the risky asset and a CRRA/log objective, their key result is that the fraction of
**current wealth** invested in the risky asset is

$$\pi_t = \lambda^\*_{\text{Merton}}\Big(1-\alpha_{GZ}\,\frac{M_t}{W_t}\Big)_{+}\tag{GZ-1}$$

where $\lambda^\*_{\text{Merton}}=(\mu-r)/\sigma^2$ is the ordinary unconstrained Merton/Kelly
fraction (renamed from GZ's own paper notation $\lambda^\*$ to $\lambda^\*_{\text{Merton}}$ here
purely to avoid collision with Axis A's EWMA decay parameter $\lambda$ and Axis B's quadratic
trading-cost coefficient $\lambda$; see notation table) and $(\cdot)_+$ denotes the ramp function
(no shorting the cushion). Two consequences fall directly out of (GZ-1) and are the entire
economic content of the result:

- **At a fresh equity high** ($W_t=M_t$), $\pi_t=\lambda^\*_{\text{Merton}}(1-\alpha_{GZ}) <
  \lambda^\*_{\text{Merton}}$: the policy is *already* de-risked relative to unconstrained Kelly
  at the single best possible moment, by exactly the drawdown budget $(1-\alpha_{GZ})$. Drawdown
  insurance is paid for continuously, not just reactively during a decline.
- As wealth drifts down against a fixed prior peak, $M_t/W_t$ rises above 1 and $\pi_t$ falls
  *continuously*, reaching **exactly zero** as $W_t\to\alpha_{GZ} M_t$. In continuous, no-jump
  dynamics the constraint literally cannot be breached, because exposure has already vanished at
  the boundary. This is the precise sense in which (GZ-1) is *the* mathematically correct
  realization of "drawdown control" — a state-dependent, peak-anchored policy that is a
  *derived optimum*, not an ad hoc vol-target overlay that only reduces exposure *after* realized
  volatility has already risen (which fact (4) of this program shows can fail completely ahead of
  a sudden reversal, e.g. 2025-04-09).

**Discrete-daily adaptation and its explicit limitation.** This desk trades once a day with real
overnight gaps; GZ's continuous-time no-breach guarantee does not literally survive that (a large
enough single-day gap — this program's own real −10.5%/−10.7% crash days — can cross the floor
within one step even under (GZ-1)). `grossman_zhou_exposure()` implements the discrete-daily
analogue exactly as stated in (GZ-1), decide-at-close/act-next-day: it recursively tracks this
*policy's own* realized wealth $W_t$ and running peak $M_t$ (not the base strategy's own
unlevered wealth — the state variable is inherently path-dependent on the policy itself, so this
is a sequential loop, not a vectorizable transform), and sets

$$L_t = l_{\text{base}}\cdot\max\!\Big(0,\;1-\alpha_{\text{floor}}\cdot\frac{M_{t-1}}{W_{t-1}}\Big)$$

applied to day $t$'s return, using **only** $W_{t-1}, M_{t-1}$ — quantities fully determined by
information through the close of day $t-1$. This is the one genuinely time-varying, causally
load-bearing signal in this module and is the one proven causal by the synthetic-injection test
(§E.6). Explicitly stated limitation: the module composes this with the EVT ceiling (§E.3)
specifically because GZ's continuous-time proof does not, by itself, bound the residual
single-day gap-through-the-floor risk in a once-a-day-rebalanced world.

### E.3 EVT / Peaks-Over-Threshold ruin-constrained leverage ceiling

Grossman-Zhou answers "how should exposure *scale* with distance from the peak"; it does not by
itself answer "what is the largest constant multiplier this book can tolerate given the realized
tail of its own daily P&L". For that, `evt_ceiling_leverage()` fits a Generalized Pareto
Distribution to the tail of realized daily *losses*, following the Peaks-Over-Threshold / GPD
methodology standard in quantitative risk management (Balkema–de Haan–Pickands theorem; McNeil &
Frey (2000) is the standard reference for applying POT/GPD specifically to financial return
tails). For a threshold $u$ (here the empirical 95th percentile of daily losses) with
exceedances $Y=(\text{loss}-u\mid \text{loss}>u)$ modeled as $\text{GPD}(\xi,\beta)$, the tail
survival function beyond $u$ is

$$P(\text{loss}>u+y) = P(\text{loss}>u)\cdot\Big(1+\frac{\xi y}{\beta}\Big)^{-1/\xi}\qquad(\xi\ne0)$$

Setting this equal to a declared ruin-probability budget $\epsilon$ and solving for $y$ gives the
loss level whose exceedance probability is exactly $\epsilon$; the ruin-constrained leverage
ceiling is then $L_{\text{ceiling}}=1/(u+y)$ — the leverage at which that loss level would exactly
wipe out capital. $\epsilon$ is tied explicitly to a declared **ruin-frequency budget**:
$\epsilon = 1/(\text{breach\_years}\times \text{periods\_per\_yr})$ means "expect the $1/L$
threshold to be breached about once every `breach_years` years" — directly comparable to the
established empirical fact that the −10% threshold at 10x leverage on SPY was breached once in
~8 real years (an empirical $\epsilon\approx 1/(8\times252)\approx 5\times10^{-4}$ for that
specific pair). The default `breach_years=10` is deliberately a little more conservative than
that observed 8-year frequency: a GPD tail fit from ~8 years of daily data (~2,000 observations,
~450 exceedances at a 95th-percentile threshold in practice here) is itself estimated with real
sampling error in $\xi$, and extrapolating deep into the tail amplifies that error — the ceiling
should sit a bit inside the single worst observed multi-year frequency, not exactly match it.

### E.4 CDaR-constrained leverage (Chekhlov, Uryasev & Zabarankin, 2005) and why the scalar case is closed-form

Chekhlov, Uryasev & Zabarankin (2005), *"Portfolio Optimization with Drawdown Constraints"*,
generalize Rockafellar–Uryasev's CVaR linear program to drawdown ("Conditional Drawdown-at-Risk").
Their formulation deliberately works on the **uncompounded, additive cumulative return path**
$c_t=\sum_{s\le t}r_s$ (not the compounded equity curve) precisely so that a linear combination
of return streams produces a linear combination of cumulative paths, which is what keeps the
whole optimization a tractable LP.

**Key derivation for this axis's scalar-leverage case** (worth deriving explicitly, because it is
*why* CDaR optimization is tractable here at all, and gives a closed-form cross-check on the LP):
let $c_t=\text{cumsum}(r_1,\dots,r_t)$ at unit ($L=1$) leverage and $C_t=\max_{s\le t}c_s$ its
running max. At leverage $L\ge0$ the scaled path is $Lc_t$, whose running max is $LC_t$ (order is
preserved for $L\ge0$), so the drawdown path at leverage $L$ is

$$\text{Drawdown}_t(L) = LC_t - Lc_t = L\cdot d_t,\qquad d_t := C_t-c_t\ge0 \tag{CDAR-1}$$

— **exactly linear in $L$**. Since CDaR of a nonnegatively-scaled series scales linearly for
$L\ge0$, $\text{CDaR}_\alpha(L)=L\cdot\text{CDaR}_\alpha(d)$ (here $\alpha$ is CDaR's own
confidence-level parameter, e.g. $\alpha=0.95$ — a different, independent use of $\alpha$ from
both Grossman-Zhou's floor fraction $\alpha_{GZ}$ above and Axis F's correlation-percentile
parameters; each is disambiguated by subscript/context in this document), and the program
"maximize $L\cdot\text{mean}(r)$ subject to $L\cdot\text{CDaR}_\alpha(d)\le\text{budget}$,
$L\ge0$" has the trivial closed form $L^\*=\text{budget}/\text{CDaR}_\alpha(d)$ whenever
$\text{mean}(r)>0$ (the objective is strictly increasing in $L$, so the budget constraint always
binds). `cdar_constrained_leverage()` solves this **twice, independently** — once via a genuine
`cvxpy` LP using the standard Rockafellar–Uryasev-style auxiliary-variable recursion
($\zeta$, $z_t\ge\max(u_t-\zeta,0)$, $u_t\ge u_{t-1}-Lr_t$, $u_t\ge0$), faithful to the general
CUZ formulation that a later integration agent would need if extending this to a genuine
multi-sleeve *blend* (where the linear shortcut above no longer applies, because the running max
of a multi-asset combination is not the combination of running maxes) — and once via the closed
form (CDAR-1). Reporting both, and their numerical agreement, is the correctness proof for the LP
encoding; empirically the two agree to within the LP solver's own tolerance (`agree=True` in
every scenario tested, §E.6). **Tradeoff worth stating plainly**: CDaR-on-the-uncompounded path
is systematically *more conservative* than a compounded-equity drawdown measure, because it never
credits compounding gains for reducing the effective future distance-to-floor the way a real
equity curve does — this is the price of convexity/tractability, not a free lunch, and is why the
CDaR-constrained leverage in every test below comes out well below the Kelly-family numbers.

### E.5 Alternatives considered and rejected

- **Continuous-time HJB solution of the drawdown-constrained problem from scratch** — rejected as
  the primary construction in favor of directly implementing Grossman-Zhou's already-published,
  already-verified closed-form policy (GZ-1); re-deriving the HJB PDE from first principles would
  not change the deployed policy and would spend effort re-proving a 1993 result.
- **Full continuous-time CDaR/CVaR portfolio optimization treating the running max as a stochastic
  control state** — rejected in favor of the discrete-additive-cumulative-return linearization
  (CDAR-1), which is the standard, literature-sanctioned tractable approximation for a
  once-daily-rebalanced desk (matches how CUZ (2005) itself frames the LP).
- **Gaussian-closed-form Kelly ($L^\*=\mu/\sigma^2$)** — rejected outright per the axis brief; the
  empirical/bootstrap numerical optimizer in §E.1 is used everywhere instead, specifically so the
  real crash days bind the answer.
- **A single "blended" leverage number reported without parameterization** — rejected; the
  deliverable is a **policy family** (kelly-fraction, GZ-alpha, EVT-breach-years, CDaR-budget are
  all first-class, named parameters of reusable functions), because the correct leverage number
  is a property of whatever base signal the other five axes' eventual integration produces, not a
  fixed constant this axis can responsibly hardcode in isolation (see §E.6's honest finding).

### E.6 Validation: real data, causality proof, and where each policy actually lands

All functions were tested against real daily data for SPY, QQQ, AMD, JPM, and COST (`source==
'duka'` filtered, `daily_close_series_duka` aggregation, both reused verbatim from
`research/indices/i1_pipeline.py`), driving a simple, honest 200-day-SMA long/flat base strategy
(13bp round-trip cost on flips) built specifically to exercise this module end-to-end — not to be
a candidate trading signal (that is the other axes' job).

**IMPORTANT INTERFACE NOTE for the Integration phase (see Section 9 of this document):** the
`base_returns` fed to every function in this module below is this toy 200d-SMA proxy, chosen
because Axes A/B/C/D/F's real modules were not available when this axis's own validation ran.
Before Stage-B integration, every leverage recipe in §E.7 must be **recomputed fresh** against the
REAL Axis A→B→C→D→F combined book return series (Section 9.2 of this document) — the specific
numeric leverage values quoted just below are illustrative of the *methodology*, not the
production leverage the Sweep phase should actually use.

**Data-quality finding (reported, not silently worked around).** AAPL was dropped from this set
after discovering its `data/stitched/AAPLUSDT.parquet` daily-close series carries an
**unadjusted 2020-08-31 4-for-1 stock split** — the raw duka-sourced close drops from 499.97 to
129.40 (a −74.12% "single day"), corrupting exactly the kind of fat-tail statistic this axis's
Kelly/EVT/CDaR machinery is built to take seriously. AMD, JPM, COST, SPY, and QQQ were
individually spot-checked clean (worst days all correspond to real, already-known events, chiefly
2020-03-16). This is a genuine bug in the shared `data/stitched` lake (distinct from, and
apparently not fixed by, the newer split-detection pipeline documented for the separate databento
equity lake in `research/equity/DATA.md`) and has been flagged out-of-scope for a dedicated fix.

**Causality proof (the required deliverable).** Only `grossman_zhou_exposure()` produces a
genuinely time-varying, path-dependent exposure signal; every other function in this module
returns a single scalar calibrated once from a historical sample and held constant thereafter
(a constant carries no $t$-dependence, so the exposure-timing-alignment bug class does not apply
to it by construction). A synthetic-outlier-injection test was run: a fake −20% return was
injected at index 896 of a 1,793-day SPY base-return series (far outside any real observed day),
and `grossman_zhou_exposure()` was compared, index by index, on the original vs. injected series:

| check | result |
|---|---|
| exposure reacts at the injection day $t$ itself (should be **False**) | **False** |
| exposure reacts at day $t+1$ (should be **True**) | **True** |
| exposure reacts at any day **before** $t$ (should be **False**) | **False** |

A deliberately-wrong **naive/lookahead control**
(`grossman_zhou_exposure_naive_lookahead_control()`, which sizes day $t$'s exposure using
wealth/peak computed *after* applying day $t$'s own return) was run through the identical test and
**does** react at the injection day itself — proving the test has real discriminating power, not
just a tautological pass. **Causality test: PASS. Naive control has teeth: confirmed.**

**Where the policies land (honest report, both toy-base and higher-Sharpe proxy).** On the
5-name pooled toy base (weak trend Sharpe by construction — three of the five names carry no
diversification benefit against each other), fresh full empirical Kelly is $L^\*=2.77$
(bootstrap 95% CI $[1.47,\,3.95]$ over 200 resamples), already at the low end of the requested
2x–5x band; every drawdown-aware construction (half-Kelly $1.39$, GZ$(\alpha_{GZ}{=}0.75)$
average $0.48$, GZ$(\alpha_{GZ}{=}0.50)$ average $1.04$, CDaR-25%-budget $0.41$) sits below it —
an honest, expected finding: **drawdown protection is not free**, and this toy base's realized
Sharpe is well below what a properly-constructed, vol-targeted, multi-horizon, ~93-name
diversified trend ensemble (what the other five axes are building) should realize. Repeating the
same policies on the SPY+QQQ pooled base (higher per-name Sharpe, no weak-single-name drag, and
empirically the closest available proxy in this dataset for that eventual integrated base) gives
fresh full Kelly $L^\*=5.75$, **half-Kelly $=2.87$ (inside the 2x–5x band)**, quarter-Kelly
$=1.44$, GZ$(\alpha_{GZ}{=}0.75)$ average $=1.00$, **GZ$(\alpha_{GZ}{=}0.50)$ average $=2.18$
(inside the 2x–5x band)**, EVT 10-year-breach ceiling $=17.2$, CDaR-25%-budget $=1.21$. Full
numeric detail in `research/cta/validate_leverage_policy.out`.

### E.7 Final deliverable: the specific, precisely-parameterized variants for the Stage-A sweep

Per-axis grid points (exact count: **6**), each a *recipe* (a fresh call against whatever base
return series the integration agent supplies, never a hardcoded numeral, matching this program's
"derived, not hardcoded" annualization discipline):

1. **Constant fractional-Kelly, half-Kelly** — `L = fractional_kelly_leverage(kelly_optimal_leverage(base_returns).L_star, 0.5)`.
2. **Constant fractional-Kelly, quarter-Kelly** — same, `fraction=0.25`.
3. **Grossman-Zhou, tight ($\alpha_{\text{floor}}=0.75$, 25% drawdown budget)** — `grossman_zhou_exposure(base_returns, l_base=kelly_optimal_leverage(base_returns).L_star, alpha_floor=0.75, rt_bp=13.0)`.
4. **Grossman-Zhou, loose ($\alpha_{\text{floor}}=0.50$, 50% drawdown budget)** — same, `alpha_floor=0.50`; this is the parameterization that lands inside the requested 2x–5x band on the SPY+QQQ proxy base (§E.6) and is the recommended flagship construction for this axis.
5. **EVT-ceiling-constrained constant leverage** — `evt_ceiling_leverage(base_returns, threshold_quantile=0.95, breach_years=10.0).L_ceiling`, used both standalone and as the optional `cap=` argument composed onto (3)/(4) to bound discrete-time gap risk that GZ's continuous-time proof does not cover.
6. **CDaR-constrained constant leverage (25% CDaR budget, 95% confidence)** — `cdar_constrained_leverage(base_returns, budget=0.25, alpha=0.95).L_closed_form` (cross-checked against the cvxpy LP solution, `agree=True` in every case tested).

Chosen over the rejected alternatives in §E.5 specifically because each is (a) derivable from the
named literature with an explicit closed-form or convex-optimization justification, (b) causal by
construction or provably causal by injection test where genuinely time-varying, and (c) a
parameterized *recipe* rather than a fixed number, so its landing zone is honestly reported as a
property of whatever base Sharpe the integrated system ultimately realizes — not asserted in
isolation by this axis.

### E.8 Axis E module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/leverage_policy.py`
  (committed to branch `equity-cta` at commit `ac8bb0bf5e876c065e3fda9df98a34d58ab264c9`; note this
  commit's message is titled "Axis C -- regime modeling..." because a concurrent parallel axis
  agent's commit swept up this axis's already-staged files in the same shared worktree — content
  is correct and fully committed/pushed, only the commit-message attribution is shared/mixed).
  Validation script: `research/cta/validate_leverage_policy.py`, output log:
  `research/cta/validate_leverage_policy.out`.
- **Grid points declared:** exactly 6 — see E.7.
- **Function signatures:**
```
empirical_log_growth(returns: np.ndarray, L: float) -> float
kelly_optimal_leverage(returns: np.ndarray, l_max: float = 25.0, n_bootstrap: int = 0, bootstrap_seed: int = 0) -> dict
fractional_kelly_leverage(l_star: float, fraction: float) -> float
grossman_zhou_exposure(base_returns: np.ndarray, l_base: float, alpha_floor: float, rt_bp: float = 0.0, cap: float | None = None) -> dict
grossman_zhou_exposure_naive_lookahead_control(base_returns: np.ndarray, l_base: float, alpha_floor: float, rt_bp: float = 0.0, cap: float | None = None) -> dict
evt_ceiling_leverage(returns: np.ndarray, threshold_quantile: float = 0.95, epsilon: float | None = None, breach_years: float = 10.0, periods_per_yr: float = 252.0) -> dict
cdar_constrained_leverage(returns: np.ndarray, budget: float, alpha: float = 0.95) -> dict
_empirical_cdar(drawdown_path: np.ndarray, alpha: float) -> float
```
- **Injection test result:** PASS. Synthetic -20% outlier injected at index 896 of a real
  1,793-day SPY-based 200d-SMA daily net-return series. `grossman_zhou_exposure()` (the only
  genuinely time-varying, causally-load-bearing function): exposure at the injection day $t$ is
  unchanged, exposure at day $t+1$ changes, and no day before $t$ changes — full causal-lag proof.
  All other functions produce a single scalar calibrated once and held constant, so causality does
  not apply to them by construction.
- **Packages installed:** `cvxpy 1.9.2` (plus transitive dependencies: `clarabel 0.11.1`,
  `osqp 1.1.3`, `scs 3.2.11`, `qdldl 0.1.9.post1`, `highspy 1.15.1`, `sparsediffpy 0.3.0`,
  `jinja2 3.1.6`, `MarkupSafe 3.0.3`). scipy 1.18.0 already present (used for
  `scipy.optimize.minimize_scalar` and `scipy.stats.genpareto`, no install needed).
## Axis F — Portfolio Construction Across the 93-Name Universe

> **Disclosed git-history note (carried forward from the original axis deliverable, resolved by
> disclosure rather than by a risky history rewrite):** while committing this axis's deliverable
> to the shared `equity-cta` worktree, a concurrent `git add`/commit race with another parallel
> axis agent caused commit `dee7423` ("research(cta): Axis F -- portfolio construction across the
> 93-name universe") to accidentally also include two files that are NOT part of Axis F:
> `research/cta/lib/entry_exit_control.py` and `research/cta/lib/validate_entry_exit_control.py`
> (Axis B's deliverable). The Axis F author attempted a non-destructive `git rm --cached` fix,
> which was blocked by the permission system as an unauthorized modification of a shared resource,
> and stopped rather than working around the denial, per house rules. **Verified during this
> unifying-document pass:** both files are real, correct, and present on disk at their proper
> paths (confirmed via `git log` and `find` on the `equity-cta` worktree — see Section 9.5 of this
> document); this is a git-attribution/commit-message hygiene issue only, not a content-
> correctness bug in either axis, and is left as-is (no destructive history rewrite was attempted,
> consistent with the host safety rules governing this program).

### F.1 The problem: what does "93 names" actually buy you?

Established Fact (3) states the mechanism plainly: all 93 names in this universe tend to crash *together* on the same systematic-factor days (worst single days for SPY, QQQ, and the equal-weight benchmark all land on 2020-03-16). This is the textbook signature of a market with one dominant common factor. Naive risk-parity — size every name inversely to its own volatility and call the result "diversified" — implicitly treats the *N* names as *N* separate bets. If they are not separate bets, sizing them as if they were mis-states the book's true risk, and mis-states it worst exactly when it matters most: during a systematic-factor event, when every "diversifying" position moves the same direction at once.

The right question is not "how many names are in the book" but "how many **independent** bets does the book actually contain." This section derives that number, measures it on the real 93-name lake, shows it collapsing exactly where Established Fact (3) predicts it should (2020-03, 2022), and then builds the portfolio-construction layer that (a) sizes each name against its true contribution to *book* risk, not its own isolated risk, and (b) explicitly de-risks the aggregate book when the correlation structure signals that today's names are not being genuinely diversifying.

### F.2 Effective number of independent bets: closed-form derivation

**Equal-weight, equal-variance case.** Let there be $N$ assets, each with variance $\sigma^2$ and constant pairwise correlation $\bar\rho$ (the average of all $\binom{N}{2}$ off-diagonal correlations — a distinct, portfolio-level object from Axis B's unused within-axis AR(1) parameter $\rho$, which never appears in any of Axis B's final formulas or code; no real cross-axis clash). The variance of the equal-weight portfolio is the standard two-term decomposition:

$$
\mathrm{Var}_{EW} = \frac{1}{N}\sigma^2 + \frac{N-1}{N}\,\bar\rho\,\sigma^2 .
$$

Define $N_{\text{eff}}$ as the number of *genuinely independent*, equal-variance assets whose own equal-weight portfolio would carry the identical variance — i.e. solve $\sigma^2/N_{\text{eff}} = \mathrm{Var}_{EW}$:

$$
\frac{1}{N_{\text{eff}}} = \frac{1}{N} + \frac{N-1}{N}\bar\rho = \frac{1+(N-1)\bar\rho}{N}
\quad\Longrightarrow\quad
\boxed{N_{\text{eff}} = \frac{N}{1+(N-1)\bar\rho}} .
$$

Sanity checks: $\bar\rho=0 \Rightarrow N_{\text{eff}}=N$ (full diversification, every name an independent bet); $\bar\rho=1 \Rightarrow N_{\text{eff}}=1$ (one bet, no matter how many tickers are in the book). This is the standard result used throughout the risk-parity and diversification literature (see e.g. Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing*, ch. 2, and any standard "diversification ratio" treatment). It is exact only under the equal-variance, constant-correlation assumption; the 93-name universe obviously violates both exactly, which is why a second, more general estimator is also computed (below).

A subtlety surfaced only by testing this formula on real data (not evident from the algebra alone): a valid $N$-asset correlation matrix requires $\bar\rho \ge -\tfrac{1}{N-1}$ (a PSD constraint — for $N=2$ the pair can be at most $\rho=-1$; for $N=3$ the three pairwise correlations cannot all simultaneously sit below $-0.5$). Early in the sample, when only a handful of names have listed, a noisy rolling *sample* correlation estimate can land marginally below this theoretical floor purely from small-sample noise, and because the denominator $1+(N-1)\bar\rho \to 0^+$ exactly at that floor, $N_{\text{eff}}$ explodes toward $+\infty$ — a pure numerical artifact of applying a population formula to a noisy small-$N$ estimate, never a real diversification signal (confirmed: the real, positive, regime-varying $\bar\rho$ observed everywhere in this dataset is nowhere close to this pathological floor). The implementation clips $\bar\rho(t)$ to its own per-day valid floor $-1/(N(t)-1)+\epsilon$ and additionally caps $N_{\text{eff}}$ at $N$ (a name cannot contribute more than one independent bet to its own book) — a numerical guard, not a modeling choice, but one worth recording since it was found empirically, not assumed.

**General case — Meucci's eigenvalue-entropy decomposition.** The closed-form above assumes a single number ($\bar\rho$) fully describes the correlation structure. Meucci (2010, "Managing Diversification," *Risk* magazine) gives a basis-invariant generalization for an arbitrary covariance matrix $\Sigma$: eigendecompose $\Sigma = V\,\mathrm{diag}(\lambda_1,\dots,\lambda_N)\,V^\top$ (these $\lambda_k$ are eigenvalues of the covariance matrix — a third, unrelated use of the symbol $\lambda$ in this document, alongside Axis A's EWMA decay and Axis B's quadratic trading-cost coefficient; disambiguated in this document's notation table as $\lambda_k^{\Sigma}$ where needed), form the **diversification distribution** $p_k = \lambda_k / \sum_j \lambda_j$ (the fraction of total variance attributable to principal component $k$), and define the effective number of bets as the exponential of its Shannon entropy:

$$
N_{\text{ent}} = \exp\!\Big(-\sum_k p_k \log p_k\Big).
$$

If variance is spread uniformly over all $N$ principal components, $p_k = 1/N$ for all $k$ and $N_{\text{ent}} = N$; if a single component (a common factor) dominates, $p_k \to \mathbf{1}[k=1]$ and $N_{\text{ent}} \to 1$ — exactly Established Fact (3)'s claim about crash days. Unlike counting "how many names carry weight above some threshold," this measure is invariant to the choice of basis (it is a property of $\Sigma$ itself, not of any particular portfolio's weights on top of it), which makes it the more rigorous of the two and the one reported as the finer-grained diagnostic (`effective_n_pca_entropy`, evaluated on the same rolling shrunk covariance used for sizing — see F.4).

### F.3 What the real 93-name lake says (standing diagnostic, not a one-off number)

Computed on the full reconciled 93-name base universe (see F.6 for the reconciliation), rolling 60-trading-day average pairwise correlation of daily `source=='duka'` returns, 2018-05-01 through 2026-07-01 (2,053 trading days):

| Window | avg pairwise ρ (mean) | avg pairwise ρ (max) | $N_{\text{eff}}$ (mean) | $N_{\text{eff}}$ (min) | avg names active |
|---|---|---|---|---|---|
| Full sample | 0.336 | 0.724 | 4.98 | 0.00* | 77.8 |
| Pre-2020 calm (2019-01 → 2020-01) | 0.392 | 0.577 | 2.64 | 1.71 | 65.4 |
| **COVID crash (2020-02-15 → 2020-04-15)** | **0.575** | **0.724** | **1.84** | **1.37** | 68.9 |
| Calm 2021 | 0.261 | 0.396 | 3.85 | 2.48 | 78.1 |
| **2022 selloff** | **0.453** | **0.548** | **2.26** | **1.81** | 84.6 |
| Recent (2024 → 2026-07) | 0.267 | 0.596 | 4.02 | 1.67 | 84.5 |

\* the full-sample minimum of 0.00 is the trivial very first calendar day of the panel (2018-05-01), where every name's `pct_change()` is NaN by construction (no prior close exists yet) — an expected boundary artifact, not a diversification finding.

**Reading this exactly as Established Fact (3) predicts**: a "93-name universe" is, on an ordinary day, worth roughly **3–5 independent bets**, not 93 — and during the two windows this universe's own worst-day history already flags as systematic-factor events (COVID crash, 2022 selloff), the *already-thin* effective breadth compresses further, to **~1.4–2.3 independent bets at the minimum**. This is the standing diagnostic requested: it is reported as a full time series (`_axisF_effective_n_timeseries.csv`, 2,053 daily rows: `avg_pairwise_corr`, `n_active`, `n_eff_closed_form`) precisely so a later integration/sweep agent can plot it and confirm its behavior in any other window of interest, not just the two anchors above. The mechanical implication for sizing: a scheme that ignores this (equal-risk-weight, treating all 93 as independent) is running an unacknowledged concentration bet on the common factor at all times, and running it at its *most* concentrated exactly when a "93-name" label most invites false comfort.

### F.4 Correlation-adjusted risk parity vs. equal-risk-weight

**Baseline — floored cross-sectional inverse-vol ("equal-risk-weight").** This is the incumbent convention already validated and reused verbatim across `e2_pipeline.py`/`e3_pipeline.py` and the exploratory `_classic_cta_universe_scan.py`: for active names on day $t$,

$$
w_i^{\text{raw}}(t) = \frac{1}{\max\!\big(\widehat{\sigma}_i(t{-}1),\; \mathrm{floor}(t)\big)},
\qquad
\mathrm{floor}(t) = 0.5 \times \mathrm{median}_j\big\{\widehat{\sigma}_j(t{-}1)\big\}_{j \in \text{active}(t)},
$$

with $\widehat{\sigma}_i(t{-}1)$ the trailing 20-day realized vol through $t{-}1$'s close. This targets *equal risk contribution under the (false, per F.3) assumption that names are uncorrelated* — a name gets less weight only because it is individually more volatile, never because it is redundant with the rest of the book.

**IMPORTANT INTERFACE NOTE for the Integration phase (see Section 9 of this document):** this
$\widehat{\sigma}_i(t-1)$ is, as inherited from e2/e3, a plain trailing-20d realized-vol estimator
— the SAME family Axis D's own head-to-head QLIKE/MSE study (§D.8) found to be one of the
*worst*-performing estimators available. The choice was deliberate for isolating the
portfolio-layer effect (see F.8), but the Sweep phase should substitute Axis D's
`ewma_vol_forecast(lam=0.94)` here once the cross-axis comparison in F.8 has been run once with
the legacy estimator for continuity with prior program history.

**Correlation-adjusted risk parity (true ERC).** The alternative solves for weights that equalize each name's *marginal contribution to total book risk*, which is the economically correct diversification target. For a covariance matrix $\Sigma$, the risk contribution of asset $i$ at weight $w$ is $RC_i(w) = w_i (\Sigma w)_i$, and the equal-risk-contribution (ERC) condition is $RC_i(w) = RC_j(w)\ \forall i,j$, $w>0$. Directly solving this fixed-point system is not a convex problem in $w$, but Spinu (2013, "An Algorithm for Computing Risk Parity Weights") shows it is equivalent to the convex program

$$
y^\star = \arg\min_{y>0}\ \tfrac12\, y^\top \Sigma y - \sum_i \log y_i,
\qquad w = y^\star / \textstyle\sum_i y_i^\star,
$$

which has a unique, strictly positive solution whenever $\Sigma \succ 0$ (the log barrier drives the objective to $+\infty$ as any $y_i \to 0^+$, and the quadratic term drives it to $+\infty$ as $\|y\|\to\infty$, so a unique interior minimum exists by strict convexity). The implementation solves this via the standard multiplicative risk-budgeting iteration (Roncalli, *Introduction to Risk Parity and Budgeting*, 2013, §2.2; closely related to Maillard, Roncalli & Teiletche 2010, "The Properties of Equally Weighted Risk Contribution Portfolios"): $y_i \leftarrow y_i\sqrt{\,(1/k)\,/\,RC_i(y)\,}$ (here $k$ is simply the asset count in the risk-budgeting loop, an unrelated, purely local use of the symbol from Axis B's curvature $k=\gamma\sigma^2$ or Axis C's state-count $K_C$ — a standard collision in the risk-parity literature's own conventional notation, harmless because it never leaves this one iteration formula), renormalized to $\sum y_i = 1$ each sweep, which converges rapidly (empirically <100 iterations to `1e-10` tolerance) for the well-conditioned shrunk covariance matrices this module constructs.

**Covariance estimation, honestly simplified.** A raw rolling-window sample correlation matrix across up to 93 names is exactly the kind of high-dimensional, noisy estimate that makes naive ERC dangerous in practice (with a 60-day window and ~80 active names, the sample covariance is ill-conditioned and its smallest eigenvalues are pure noise). Rather than a full DCC-GARCH estimation (judged not clearly worth the added complexity and estimation risk for a Stage-A module, per the task's own explicit invitation to make and justify this simplification), this module uses **Ledoit–Wolf-style shrinkage toward the constant-correlation target** (Ledoit & Wolf, 2003/2004, "Honey, I Shrunk the Sample Covariance Matrix," *J. Portfolio Management*): the shrinkage target sets every off-diagonal correlation to the sample's own average pairwise correlation $\bar\rho$ (diagonal = 1) — a single-parameter, always-well-conditioned target that is a *better-motivated* prior for this specific universe than "shrink toward the identity," precisely because Established Fact (3) says the dominant real structure here *is* a single common factor, which the constant-correlation model captures directly. The blend

$$
\Sigma_{\text{shrunk}} = (1-\delta)\,\Sigma_{\text{sample}} + \delta\,\Sigma_{\text{const-corr}}
$$

uses a **fixed, disclosed shrinkage intensity** $\delta$ (swept at $\delta=0.3$ in this module's validation; exposed as a caller parameter) rather than the full analytic Ledoit-Wolf optimal-$\delta$ estimator (which requires estimating fourth-moment cross-terms over the same rolling window — a real additional engineering cost this Stage-A module elects not to pay, disclosed rather than hidden). A small numerical ridge ($10^{-6}\times$ mean variance) is added to the diagonal before ERC solving, purely for invertibility.

**Only names with a full, complete rolling window enter the correlation estimate**; a day with fewer than `MIN_ACTIVE_FOR_CORR = 10` simultaneously-active, fully-windowed names falls back explicitly to the equal-risk-weight scheme for that day (`scheme_used` diagnostic column records this per-date — a disclosed fallback, never a silent one). This matters mechanically at the very start of the sample (2018-2019), when only a handful of the 93 names have listed.

### F.5 The correlation-spike de-risking overlay

Ang & Chen (2002, "Asymmetric Correlations of Equity Portfolios," *J. Financial Economics*; see also Longin & Solnik, 2001) document that equity correlations are not stationary — they spike specifically during market stress and drawdowns, precisely the regime in which "diversification across many names" is weakest (this is the *same* phenomenon F.3 measures directly on this universe's own history: $\bar\rho$ nearly doubles from its calm-2021 level in both the COVID crash and the 2022 selloff). A portfolio-construction scheme that only re-estimates correlation for *sizing* (F.4) but never *scales down aggregate exposure* when correlation itself is signaling a regime change is running its normal risk budget straight into the one environment where its realized effective breadth has already collapsed toward $N_{\text{eff}}\approx 1$–2.

The overlay is deliberately **relative to the series' own trailing history**, not an absolute threshold — a universe whose typical $\bar\rho$ sits at 0.35 needs a different trigger than one whose typical $\bar\rho$ sits at 0.10, and anchoring to a hand-tuned absolute constant would not transfer across regimes or universes:

$$
\text{scale}(t) =
\begin{cases}
1 & \bar\rho(t) \le p_{90}(t) \\[2pt]
1 - \dfrac{\bar\rho(t)-p_{90}(t)}{p_{99}(t)-p_{90}(t)}\,(1-\text{floor}) & p_{90}(t) < \bar\rho(t) < p_{99}(t) \\[6pt]
\text{floor} & \bar\rho(t) \ge p_{99}(t)
\end{cases}
$$

where $p_{90}(t), p_{99}(t)$ are the 90th/99th percentiles of $\bar\rho$ over the trailing `lookback = 756` trading days (~3 years), evaluated using **only observations strictly before $t$** (`shift(1)` before the rolling quantile — see causality proof, F.7), and `floor = 0.4` (disclosed constant; a full de-lever to zero was judged too punitive for a *diagnostic* de-risking layer whose job is to shrink the book's aggregate risk budget, not shut it down entirely — a harder floor is a one-line change for a later integration agent to tune). The first `lookback` days (no trailing distribution yet) get scale 1.0, fail-open, matching `overlay.py::vol_target_scale`'s own warm-up convention. Both weighting schemes (F.4) are renormalized to unit gross ($\sum_i |w_i| = 1$ across active names, per the unit-gross bug-fix already discovered and documented in the exploratory `_classic_cta_universe_scan.py`) *before* this overlay is applied multiplicatively — confirmed in the module's own grid smoke test: with the overlay off, realized gross is 1.0 to float precision on every active day; with the overlay on, realized gross tracks down to as low as the 0.4 floor exactly when the correlation-window setting shows the overlay engaging.

### F.6 Universe reconciliation (fresh, not assumed)

Loaded fresh via `convexity.panel.candidates.load_universe()` in this session: 110 total equities entries in `candidates.md` → 102 with a real `dukascopy` field → minus 3 short-history exclusions (`STXXUSDT`, `CBRSUSDT`, `DRAMUSDT`) → minus 6 leveraged/inverse/decay-ETF exclusions (`UVXYUSDT`, `SQQQUSDT`, `TQQQUSDT`, `SOXLUSDT`, `KORUUSDT`, `MVLLUSDT`) → **93 names, exact match to the declared base universe, no discrepancy to investigate**. All 93 have a stitched parquet on disk; zero dropped for missing data.

### F.7 Causality and validation summary

Every function that produces a time-varying sizing/gating value is provably causal: `rolling_avg_pairwise_correlation`, `rolling_shrunk_covariance`, and `correlation_spike_scale` all compute their rolling statistic over a window ending at $t{-}1$ (`values[t-window:t, :]`, i.e. rows strictly before $t$) before it is used to size or gate the exposure applied *at* $t$; the percentile lookback inside the overlay additionally applies its own `shift(1)` on top of an already-causal input series. Verified two ways:

1. **Synthetic injection** (12 synthetic names, 800 days, one fabricated +40% simultaneous move injected at a single date): the causal `rolling_avg_pairwise_correlation` is bit-identical at the injection date to the un-injected baseline, changes at the very next date (proving the function is not simply inert), and a deliberately-wrong **naive control** (`_naive_uncausal_avg_corr`, an including-day-$t$ rolling window) *does* react same-day — confirming the test methodology has real discriminating power, not a tautological pass. Result: **PASS** on all three legs.
2. **Real-data injection**, run twice: (a) directly on `rolling_avg_pairwise_correlation` using real SPY/QQQ/AAPL/MSFT/NVDA daily returns with a synthetic +50% one-day shock spliced into a copy of the real series; (b) end-to-end through the full `aggregate_portfolio()` pipeline (real returns, a placeholder 200-day-SMA trend signal, real trailing vol) confirming the *final portfolio weights themselves* — not just the intermediate correlation diagnostic — are unchanged at the injection date and change the day after. Both **PASS**, both include the same naive-control check confirming teeth.

The full validation run (universe reconciliation, both injection tests, the full-universe effective-N diagnostic in F.3, and an 8-cell grid smoke test of `aggregate_portfolio()` proving correct unit-gross renormalization pre-overlay and correct de-risking post-overlay) is captured in `research/cta/lib/_axisF_validation_summary.json`, with the full daily effective-N/avg-corr/n-active time series in `research/cta/lib/_axisF_effective_n_timeseries.csv`. The 20-name, full-history, placeholder-signal grid smoke test is mechanics-only (there is no real Axis A–E signal yet at Stage A) — it exists solely to prove `aggregate_portfolio()` executes correctly end to end on real data before a later integration agent wires in the real upstream signal; it is explicitly **not** a performance claim. Full repo pytest suite: **654 passed, 1 skipped** — unchanged from the established baseline.

### F.8 Chosen variants for the Stage-A sweep, and what was rejected

**Implementing, precisely parameterized:**

- **Scheme A — equal-risk-weight baseline**: floored cross-sectional inverse-vol, `FLOOR_MULT = 0.5`, 20-day trailing vol (unchanged from the already-validated e2/e3 convention — kept identical so any Sharpe/CAGR delta against Scheme B is attributable to the portfolio-layer choice alone, not a confound from also changing the vol estimator).
- **Scheme B — correlation-adjusted risk parity**: Spinu-convex ERC solve on a Ledoit–Wolf-constant-correlation-shrunk covariance, shrinkage intensity $\delta=0.3$ (fixed, disclosed), `MIN_ACTIVE_FOR_CORR = 10` fallback threshold.
- **Correlation-window grid**: $\{60, 120\}$ trading days — 60d as the fast, more regime-reactive setting, 120d as a slower, more stable setting less prone to whipsaw from a single volatile week — deliberately not a wider sweep (e.g. adding 20d or 250d) to keep the axis's contribution to the overall Stage-A grid a modest, clearly-bounded 4/8-cell block rather than a combinatorial explosion the Synthesis phase then has to absorb across all six axes.
- **Correlation-spike de-risking overlay**: on/off, `lookback=756` (~3y), trigger/full-delever percentiles `(90, 99)`, `scale_floor=0.4` — all fixed and disclosed rather than swept, since the overlay's *presence* (does de-risking on correlation spikes help at all) is the Stage-A question this axis is answerable on; its own internal hyperparameters are left to a later robustness pass once a real upstream signal exists to measure Sharpe/CAGR/maxDD sensitivity against.

**Grid points contributed to the Stage-A sweep ledger**: $\{$equal_risk_weight, corr_risk_parity$\} \times \{$60d, 120d correlation window$\} \times \{$with, without spike overlay$\}$ = **2 × 2 × 2 = 8 grid points**.

**Rejected alternatives, and why:**

- **Full DCC-GARCH covariance estimation** — theoretically the more rigorous choice for time-varying correlation, but a materially larger engineering and estimation-risk lift for a Stage-A module (multivariate GARCH parameter estimation across up to 93 series, with its own convergence and stability failure modes) relative to the marginal benefit over a well-shrunk rolling-window estimate for this specific use (sizing + a binary-ish de-risking trigger, not a precise volatility forecast in its own right). Rolling-window + constant-correlation shrinkage is judged the honestly-adequate simplification the task explicitly permits.
- **Analytic (data-driven) Ledoit-Wolf optimal shrinkage intensity** — rejected in favor of a fixed, disclosed $\delta$ for the same reason: the analytic estimator requires additional fourth-moment cross-term estimation over the same rolling window, a real cost not justified when the module's main empirical claim (F.3's effective-N collapse) is already robust to the exact shrinkage intensity (the *target itself*, i.e. $\bar\rho(t)$, still moves with the regime even at fixed $\delta$).
- **A fixed absolute correlation threshold for the de-risking overlay** (e.g. "de-lever whenever $\bar\rho > 0.5$") — rejected in favor of the trailing-percentile design specifically because F.3 shows this universe's *typical* $\bar\rho$ already sits at 0.26–0.39 even in calm regimes; a fixed absolute threshold picked by eye on this sample would not generalize and is exactly the kind of hand-tuned constant the percentile design avoids.
- **A full-delever-to-zero overlay floor** — rejected in favor of `scale_floor=0.4`: this axis's brief is a *diagnostic de-risking overlay* on top of a portfolio-construction scheme, not a regime-timing kill-switch (which is arguably Axis D/E's territory); zeroing out entirely would conflate two different mechanisms the Synthesis phase should be able to attribute separately.
- **Widening the correlation-window grid beyond {60, 120}** (e.g. adding 20d/250d) — rejected to keep this axis's contribution to the combined six-axis Stage-A grid a bounded, disclosed 8 cells; a later robustness pass can widen this once a real signal exists to make the extra cells informative rather than merely more numerous.

### F.9 Axis F module reference

- **Module path:** `/home/speculation/convexity-cta/research/cta/lib/portfolio_construction.py`
  (validation driver: `/home/speculation/convexity-cta/research/cta/lib/validate_axisF.py`;
  validation artifacts: `_axisF_validation_summary.json` and `_axisF_effective_n_timeseries.csv`,
  all committed on branch `equity-cta` at commit `dee7423`).
- **Grid points declared:** `{equal_risk_weight, corr_risk_parity}` scheme x `{60d, 120d}` rolling
  correlation window x `{with, without}` the correlation-spike de-risking overlay = 2 x 2 x 2 = 8,
  fully enumerated: (equal_risk_weight,60d,overlay-on), (equal_risk_weight,60d,overlay-off),
  (equal_risk_weight,120d,overlay-on), (equal_risk_weight,120d,overlay-off),
  (corr_risk_parity,60d,overlay-on), (corr_risk_parity,60d,overlay-off),
  (corr_risk_parity,120d,overlay-on), (corr_risk_parity,120d,overlay-off).
- **Function signatures:**
```
load_duka_daily_returns_panel(symbols: list[str], data_stitched_dir) -> pd.DataFrame
effective_n_closed_form(avg_pairwise_corr: pd.Series, n_active: pd.Series) -> pd.Series
effective_n_pca_entropy(cov: np.ndarray) -> float
rolling_avg_pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series
rolling_shrunk_covariance(returns: pd.DataFrame, t: int, window: int = 60, shrink_intensity: float = 0.3) -> tuple[np.ndarray, list[str]] | None
correlation_spike_scale(avg_pairwise_corr: pd.Series, lookback: int = 756, spike_percentile: float = 90.0, full_deleverage_percentile: float = 99.0, scale_floor: float = 0.4) -> pd.Series
floored_inverse_vol_weights(rv_lagged_row: pd.Series, active_mask: pd.Series) -> pd.Series
risk_parity_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray
class PortfolioResult (dataclass: weights, avg_pairwise_corr, n_active, n_eff_closed_form, n_eff_pca, corr_spike_scale, scheme_used)
aggregate_portfolio(target_exposure: pd.DataFrame, returns: pd.DataFrame, rv_lagged: pd.DataFrame, scheme: str = "corr_risk_parity", corr_window: int = 60, shrink_intensity: float = 0.3, apply_spike_overlay: bool = True, spike_lookback: int = 756, spike_percentile: float = 90.0, full_deleverage_percentile: float = 99.0, scale_floor: float = 0.4) -> PortfolioResult
run_causality_injection_test(window: int = 60, inject_at_frac: float = 0.7, seed: int = 0) -> dict
_naive_uncausal_avg_corr(returns: pd.DataFrame, window: int = 60) -> pd.Series
```
- **Injection test result:** Three independent injection tests, all PASS — (1) synthetic (12
  names, 800 days, +40% simultaneous move); (2) real data on `rolling_avg_pairwise_correlation`
  (SPY/QQQ/AAPL/MSFT/NVDA, +50% one-day shock spliced in); (3) real data end-to-end through
  `aggregate_portfolio()` — final portfolio weights unchanged at the injection date, changed the
  day after.
- **Packages installed:** None new. numpy 2.5.0, pandas 3.0.3, scipy 1.18.0 already present;
  `risk_parity_weights` uses a dependency-light hand-rolled multiplicative fixed-point iteration
  rather than `scipy.optimize`.
## Section 7 — Cross-Axis Notation Reconciliation

Each axis's own mathematical section (1–6 above) is reproduced with its author's original,
internally-self-consistent notation, since rewriting six independent derivations symbol-by-symbol
carries real transcription risk. Six symbols are genuinely **overloaded across axes** — used for
different objects in different sections — and are listed here explicitly so no reader carries a
meaning from one axis into another by accident. Inline callouts have also been inserted at the
point of clash in the axis text above (Axis C §C.2/§C.6 for $K$, Axis D §D.3/§D.4 for the
Yang-Zhang/EWMA decay symbols, Axis E §E.2/§E.4 for $\lambda$/$\pi_t$/$M_t$, Axis F §F.2/§F.4 for
$\lambda_k$/$k$). Symbols not listed here (e.g. $L$ for leverage, used only in Axis E; $z_t$ for
tracking error, used only in Axis B; $\bar\rho$ for average pairwise correlation, used only in
Axis F) are unique to one axis and carry no cross-axis ambiguity.

| Symbol | Axis A meaning | Axis B meaning | Axis C meaning | Axis D meaning | Axis E meaning | Axis F meaning | Resolution used in this document |
|---|---|---|---|---|---|---|---|
| $\lambda$ | EWMA decay, $\lambda\in(0,1)$, parameterized by halflife | GP(2013) quadratic trading-cost coefficient $\lambda$ (and $\lambda_{\text{eff}}$, tied to the Davis-Norman band) | — | EWMA-of-**variance** decay (RiskMetrics), $\lambda\in\{0.94,0.97\}$ — same *role* (exponential decay) as Axis A's but a materially different calibrated object (variance recursion, not price/return level) | Grossman-Zhou's Merton fraction $\lambda^\*=(\mu-r)/\sigma^2$ | eigenvalues $\lambda_k$ of the covariance matrix $\Sigma$ (Meucci decomposition) | Kept native to each axis's own text; Axis D's is written $\lambda_{\text{ewma}}$/`lam`, Axis E's Merton fraction renamed $\lambda^\*_{\text{Merton}}$, Axis F's eigenvalues written $\lambda_k^\Sigma$ where disambiguation is needed. Axis B's $\lambda$/$\lambda_{\text{eff}}$ is the one genuinely load-bearing quadratic-cost coefficient in this document and is left unqualified. |
| $K$ | number of component trend signals combined (5 in the final variant: 3 EWMA + Kalman + l1TF) | — | number of latent HMM states (2 or 3) | — | — | — | Axis A's count written $K_A$, Axis C's state count written $K_C$ in this document's cross-axis text (each axis's own internal formulas are unaffected). |
| $\pi$ / $\pi_t$ | — | — | HMM initial state distribution $\pi=P(S_1=\cdot)$ | — | Grossman-Zhou's fraction-of-wealth-invested policy $\pi_t$ (GZ-1) | — | Axis C's $\pi$ (a static distribution) and Axis E's $\pi_t$ (a time-varying policy) never appear in the same formula; Axis E's own module code does not use the symbol `pi` at all (implemented as the `exposure` array), so no code-level clash exists — only a notation-table clash, resolved by context. |
| $M$ / $M_t$ | $M_{\lambda,t}$, the single-speed EWMA-of-returns momentum signal (Section A.1.2) | — | — | — | $M_t=\sup_{s\le t}W_s$, the running peak of Grossman-Zhou wealth | — | Axis A's momentum score is written $\mathrm{Mom}_\lambda(t)$ in this document's cross-axis text; Axis E keeps $M_t$ for the wealth peak, the dominant convention in the GZ literature. |
| $\alpha$ | — | — | — | — | $\alpha_{\text{floor}}$ (Grossman-Zhou drawdown-floor fraction) **and**, separately, $\alpha$ (EVT threshold quantile / CDaR confidence level, e.g. 0.95) — already two distinct uses *within* Axis E itself, disambiguated there by argument name (`alpha_floor` vs `alpha`) | correlation-spike overlay's percentile parameters ($p_{90},p_{99}$, not literally named $\alpha$, no clash) | Written $\alpha_{GZ}$ (drawdown floor) and $\alpha_{\text{conf}}$ (EVT/CDaR confidence) in this document's cross-axis text where both appear near each other (Section E.4). |
| $k$ | — | curvature $k=\gamma\sigma_r^2$ (GP/Davis-Norman risk-aversion-weighted return variance) | state index $k=1,\dots,K_C$ | — | — | risk-budgeting iteration's local loop variable (asset count, `1/k` inside `risk_parity_weights`'s update — a harmless, purely-local collision with the risk-parity literature's own conventional notation, never visible outside that one iteration formula); also Yang-Zhang's blend weight, written $k_{yz}$ in Axis D's own text specifically to avoid this collision | — | Disambiguated at point of use; Axis B's curvature $k$ is the one load-bearing cross-axis object (it is the same $\gamma\sigma_r^2$ construction referenced by Axis D's forecast_vol interface fix in Section 8 below). |
| $\sigma$ / $\sigma^2$ | (not a headline symbol; `vol_window` used only for internal signal normalization) | $\sigma_r^2$ (underlying return variance, in curvature $k$), $\sigma_z^2$ (tracking-error/signal vol, in the no-trade band $\Delta^\*$) | (not used; HMM emission variance is $\sigma_k^2$, a per-state parameter, no clash) | **canonical deliverable**: `forecast_vol`, i.e. $\hat\sigma_t$, this axis's causal one-step-ahead vol forecast | $\sigma^2$ (return variance, rejected Gaussian-Kelly closed form only; the adopted empirical Kelly does not use $\sigma^2$ directly) | (not used directly; covariance matrix $\Sigma$ is the object, not a scalar $\sigma$) | **This is the most consequential entry in this table** — see Section 8.4 below: Axis D's $\hat\sigma_t$ is designated the single canonical vol estimator every other axis's own internal $\sigma$ should consume, and two concrete code changes are required to make that true (Axis B's `entry_exit_control.py`, Axis F's `rv_lagged` wiring). |

**Symbols confirmed to have NO cross-axis clash** (checked explicitly, listed for completeness so
a future reader does not have to re-derive this): $L$ (leverage, Axis E only), $z_t$ (tracking
error, Axis B only), $\theta$ (proportional trading-cost rate, Axis B only), $\Delta^\*$ (no-trade
band half-width, Axis B only), $\kappa$ (GP adjustment speed, Axis B only), $x_t/x^\*_t$ (position
/ target, Axis B only), $\bar\rho(t)$ (average pairwise correlation, Axis F only — Axis B's
unrelated, and in Axis B's own text never actually used in any final formula or code, AR(1)
parameter $\rho=1-\phi$ was checked and confirmed absent from `entry_exit_control.py`'s function
signatures), $N$/$N_{\text{eff}}$ (name count / effective bet count, Axis F only), $\xi,\beta$
(GPD tail parameters, Axis E only), $V(x,f)$ vs. $W(z)$ (two distinct value functions used for two
distinct problems *within* Axis B's own section, already disambiguated there by the author).
## Section 8 — Interface Architecture: How the Six Axes Compose

### 8.1 The three-layer pipeline

Reading the six modules' own interface contracts together (Axis B's docstring: "`target` must
already be Axis A's causal ... signal in $[-1,1]$"; Axis F's signature:
`aggregate_portfolio(target_exposure: pd.DataFrame, returns, rv_lagged, ...)`; Axis E's own
disclosed proxy-base caveat in §E.6) yields one coherent three-layer pipeline, per name and then
across the book:

```
PER-NAME SIGNAL/SIZING LAYER (per name i, per day t)
  Axis A:  raw_dir_i(t)   = multi_horizon_trend_signal(log_price_i)['combined'][t]     in (-1,1), already .shift(1)'d
  Axis C:  regime_gate_i(t) = exposure_scale_from_posterior(p_bull_i(t))               in [floor,cap] subset [0,1]
  Axis D:  vol_scale_i(t) = vol_target_exposure_magnitude(forecast_vol_i(t), ...)      >= 0, capped at cap_mult*leverage_target
  ---------------------------------------------------------------------------------------------------------
  gated_target_i(t) = clip( raw_dir_i(t) * regime_gate_i(t), 0, None ) * vol_scale_i(t)   -- see 8.2 on the clip
  Axis B:  exposure_i(t) = partial_adjustment_control(gated_target_i, price_i, ...)['exposure'][t]

PORTFOLIO AGGREGATION LAYER (across all i=1..93, per day t)
  Axis F:  PortfolioResult = aggregate_portfolio(target_exposure=exposure_{1..93}, returns, rv_lagged, scheme, ...)
  book_return(t) = sum_i PortfolioResult.weights[i, t] * returns[i, t]      (unit-gross by construction)

LEVERAGE / CAPITAL-ALLOCATION LAYER (on the aggregate book_return series)
  Axis E:  L(t) = one of {half-Kelly, quarter-Kelly, GZ-tight, GZ-loose, EVT-ceiling, CDaR}(book_return)
  final_net_return(t) = L(t) * book_return(t)
```

Every arrow above is causal by each axis's own injection-test proof (Sections 1–6), and the
elementwise product `raw_dir_i(t) * regime_gate_i(t) * vol_scale_i(t)` of three independently
`.shift(1)`-lagged series is itself still exactly decide-at-close(t-1)/act-at(t) (a product of
functions of $\{r_{1:t-1}\}$ is still a function of $\{r_{1:t-1}\}$) — no additional lag is
introduced or lost by the composition, and no re-lagging is needed or correct at this step.

### 8.2 Interface fix #1 (required): long-only clipping, per Established Fact (2)

Axis A's `multi_horizon_trend_signal` is a genuinely **symmetric** signal — its normalized
components use a `tanh_gaussian` response bounded in $(-1,1)$ and its IC-weighted combination is a
convex (nonnegative, sum-to-one) blend of them, so `combined` can and does go negative (verified
directly against the module source: `combination_weights_from_ic` floors IC at zero but does
**not** floor the resulting signal itself). None of Axis A/B/C/D/E/F's own text reconciles this
against **Established Fact (2)**: "SHORTING equities is a proven net negative on this universe
... DEFAULT CONSTRUCTION: long/flat only." Axis F's two weighting schemes
(`floored_inverse_vol_weights`, `risk_parity_weights`) both implicitly assume the exposures they
are allocating across names are being sized, not directionally decided from scratch inside the
portfolio layer — i.e. the long/flat decision must already be made before Axis F sees the
exposure.

**Required fix, stated explicitly so it does not have to be rediscovered at Integration time:**
the Sweep phase must clip the combined `raw_dir_i(t) * regime_gate_i(t)` product to
$[0,\infty)$ — i.e. `max(0, ·)` — as the DEFAULT Stage-A construction, before multiplying by
Axis D's `vol_scale_i(t)` and before handing the result to Axis B. This is a one-line change at
the Integration-layer call site (not inside any of the six axis modules themselves, which are each
correctly agnostic to this house-level directional-construction rule); no axis module needs to be
edited for this specific fix. A short-sleeve variant, if ever tested, must be a separately-
ledgered Stage-A branch per the house rules already established for Established Fact (2), never a
silent symmetric change to this default pipeline.

### 8.3 Interface fix #2 (required): Axis C's gate multiplies Axis A's direction, it does not replace it

Axis C's `exposure_scale_from_posterior(p_bull, floor, cap)` produces a **magnitude-only gate** in
$[\text{floor},\text{cap}]\subset[0,1]$, not a directional signal — it is Axis A's job to say
*which way*, and Axis C's job to say *how much of full sizing is warranted given the regime*. This
is stated here explicitly because Axis B's own interface contract only names Axis A ("`target`
must already be Axis A's ... signal") and does not mention Axis C at all — a reader integrating
the two could plausibly (and incorrectly) wire Axis C's `p_bull`-derived gate in as if it were an
alternative *directional* signal to Axis A's, rather than a multiplicative overlay on top of it.
The correct composition, stated in 8.1, is `raw_dir_i(t) * regime_gate_i(t)`; because
$|\text{regime\_gate}_i(t)|\le 1$, this product cannot push the combined signal outside Axis A's
own native $(-1,1)$ range, so Axis B's stated interface contract ("target already in $[-1,1]$")
continues to hold for the gated product without modification.

### 8.4 Interface fix #3 (required): a single canonical vol estimator, not three independently-reinvented ones

Three of the six axes each maintain their **own internal trailing-realized-vol estimator**,
despite Axis D's entire mandate being to determine, once, which vol estimator is actually best on
this data:

| Where | What it currently does | Axis D's finding that applies |
|---|---|---|
| Axis B `entry_exit_control.py`, `partial_adjustment_control` | Computes $\sigma_z,\sigma_r$ internally via `realized_return_vol(price, vol_window=20)` — a plain trailing-20d realized-vol estimator | Axis D's own head-to-head QLIKE/MSE study (§D.8) found `trailing_rv_20` to be one of the two *worst*-performing families (mean QLIKE 2.1283 vs. EWMA's 1.9342) |
| Axis F `portfolio_construction.py`, `floored_inverse_vol_weights` (fed by the `rv_lagged` argument to `aggregate_portfolio`) | Also documented as "trailing 20-day realized vol," reused verbatim from the e2/e3 convention, deliberately unchanged so Scheme A vs. B is attributable to the portfolio layer alone (F.4's own disclosed rationale) | Same finding — Axis F's own weighting is running on the same demonstrably-inferior estimator family |
| Axis E leverage functions | Do **not** need an external vol estimate (Kelly/GZ/EVT/CDaR all work directly off the realized historical return distribution) | N/A — no fix needed here |

**Required fixes:**
1. `entry_exit_control.py`'s `partial_adjustment_control` should be given an optional
   `external_vol: pd.Series | None = None` parameter that, when supplied, is used in place of the
   internal `realized_return_vol(price, vol_window)` call for both $\sigma_r$ (curvature $k$) and
   $\sigma_z$ (tracking-error vol). This is a small, additive, backward-compatible signature
   change (default `None` preserves today's behavior exactly, so Axis B's own already-passing
   injection test and B.8 diagnostics remain valid unchanged).
2. At the Integration/Sweep-phase call site, both this new `external_vol` argument (for Axis B)
   and the `rv_lagged` argument of `aggregate_portfolio` (for Axis F) should be populated with
   **the same series**: Axis D's own recommended default, `vol_forecast.ewma_vol_forecast(daily_close,
   lam=0.94)`, per name. This both (a) removes the redundant/inferior duplicate vol estimation
   currently embedded in two other modules, and (b) ensures the risk-management stack (Axis B's
   trading-cost curvature, Axis F's cross-sectional weighting) is driven by the single
   estimator Axis D's own direct evaluation actually validated as best — rather than three modules
   silently disagreeing about "the" volatility of the same name on the same day.
3. This fix is **deliberately not applied inside Axis D's own grid** (Axis D's 10 sweep points
   remain a pure, isolated comparison of estimator families against realized returns, per its own
   Stage-A mandate) — it is an Integration-layer wiring change, made once, after Axis D's own
   winner is picked (or, for full Stage-A generality per Section 9 below, made once per Axis-D
   grid point, since the Stage-A sweep tests all 10 Axis D variants, not just the recommended
   default).

### 8.5 Interface note (disclosed, no fix required): Axis E must be recalibrated against the real integrated book, not its Stage-A validation proxy

Axis E's own §E.6 explicitly discloses that its reported landing-zone leverage numbers (half-Kelly
$=2.87$, GZ-loose average $=2.18$, etc.) were computed against a "simple, honest 200-day-SMA
long/flat base strategy," not the real Axis A→B→C→D→F combined book, because the other five axes'
modules were not available when Axis E's own validation ran. This is not a bug to fix in
`leverage_policy.py` — the module's functions are correctly generic (`kelly_optimal_leverage`,
`grossman_zhou_exposure`, etc. all take an arbitrary `base_returns`/`returns` array) — it is simply
a fact that must be re-run, not re-derived, once Section 8.1's `book_return(t)` series exists from
a real Sweep-phase run. No code change is required; this is stated here so the Sweep phase does
not mistake Axis E's illustrative §E.6 numbers for the numbers it should actually use.

### 8.6 Interface note (disclosed, no fix required): Axis C's per-name 3-state fallback

Axis C's §C.6 caveat — "3-state HMM should be restricted to long-history names ... falling back to
2-state or the SMA baseline for young listings" — is a per-name implementation detail inside the
signal-generation layer (8.1's Axis C step), not a grid-dimension change: whichever of the 5 Axis
C grid points is selected for a given Stage-A cell, a young name simply receives the 2-state (or
SMA) construction transparently in place of an unreliable 3-state fit, with no separate grid cell
or `N_cells` consequence. This is noted here so the Sweep-phase implementation of the frozen grid
(Section 9) does not need to special-case it at the grid-design level — it is purely a per-name,
per-cell runtime branch inside the Axis-C-3-state cells.

### 8.7 Verification that every axis module is present and committed

All six axis modules, plus every one of their own validation/diagnostic artifacts named in
Sections 1–6's "module reference" subsections, were confirmed present on disk in the
`equity-cta` worktree (`/home/speculation/convexity-cta`) at the exact paths declared, via a fresh
`find research/cta -type f` immediately before this document was assembled — nothing was found
missing, and nothing had to be re-fetched or rewritten:

```
research/cta/lib/signal_extraction.py            (Axis A)
research/cta/lib/validate_signal_extraction.py   (Axis A)
research/cta/tests/test_signal_extraction.py     (Axis A)
research/cta/lib/entry_exit_control.py           (Axis B)
research/cta/lib/validate_entry_exit_control.py  (Axis B)
research/cta/lib/regime_hmm.py                   (Axis C)
research/cta/lib/_validate_regime_hmm.py         (Axis C)
research/cta/lib/_check_20250409.py              (Axis C)
research/cta/lib/validate_regime_hmm_results.json (Axis C)
research/cta/lib/vol_forecast.py                 (Axis D)
research/cta/lib/leverage_policy.py              (Axis E)
research/cta/validate_leverage_policy.py         (Axis E)
research/cta/validate_leverage_policy.out        (Axis E)
research/cta/lib/portfolio_construction.py       (Axis F)
research/cta/lib/validate_axisF.py               (Axis F)
research/cta/lib/_axisF_validation_summary.json  (Axis F)
research/cta/lib/_axisF_effective_n_timeseries.csv (Axis F)
```

All six modules are already committed on `equity-cta` (commits `961be81` Axis D, `dee7423` Axis F
+ Axis B's two files per the disclosed attribution note in Section 6's preamble, `ac8bb0b` Axis C
+ Axis E, `e64b320` Axis A — see `git log` in Section 10 below) and pushed to `origin/equity-cta`
prior to this document being written. No axis module required re-fetching, rewriting, or
re-committing.
## Section 9 — The Frozen Stage-A Exploratory Grid

### 9.1 Per-axis grid point counts (recap, each independently frozen by its own axis)

| Axis | What it varies | Grid points |
|---|---|---|
| A — Signal Extraction | trend-filter family, halflives/params, IC-shrinkage | **31** |
| B — Entry/Exit Control | `(cost_sensitivity, band_scale)` | **4** |
| C — Regime Modeling | `(n_states, refit_every)` + SMA baseline | **5** |
| D — Vol Forecasting | estimator family x setting | **10** |
| E — Leverage Policy | Kelly fraction / GZ floor / EVT / CDaR recipe | **6** |
| F — Portfolio Construction | scheme x corr-window x spike-overlay | **8** |

**Naive full cross-product**: $31\times4\times5\times10\times6\times8 = \mathbf{297{,}600}$ cells.

### 9.2 Why the naive full cross-product is rejected

297,600 clearly exceeds the "a few thousand cells" threshold that triggers this program's own
requirement to propose a justified reduction rather than default to full enumeration. Two
independent reasons, not just one, argue against running it as-is:

1. **Computational cost is real, even though each individual cell is cheap.** Every one of Axis
   A's 31, Axis C's 5, and Axis D's 10 signal-generation configurations is computed **once per
   name**, cached, and reused — none of the combinatorics of B/E/F requires re-fitting an HMM or
   re-estimating a GARCH model, so the 297,600 figure does *not* imply 297,600 independent
   HMM/GARCH fits. What it *does* imply is 297,600 independent full 93-name, ~2,000-trading-day
   equity-curve simulations (elementwise signal combination → Axis B's O(T) recursion → Axis F's
   cross-sectional aggregation → Axis E's leverage calc), each a few tenths of a second to a
   couple of seconds of vectorized pandas/numpy work. At even a conservative 1 second/cell this is
   ~83 CPU-hours — trivially parallelizable across cores or names, so not infeasible in the literal
   sense, but a large amount of throwaway compute and storage (297,600 equity curves, weight
   matrices, and diagnostics) for what Section 9.3 argues is not a proportionally large amount of
   *additional information* over a smarter design.
2. **Scientific redundancy, exactly per this program's own explicit hint.** Fully crossing every
   vol-forecast variant (D) against every regime-model variant (C) against every leverage-policy
   variant (E) tests $5\times10\times6=300$ combinations of three axes that all act on
   *approximately the same channel* — the risk/sizing scale applied to an already-decided
   direction — rather than three axes hypothesized to interact richly with each other beyond their
   own well-understood, largely-separable, near-multiplicative individual effects (a regime gate
   that scales exposure by $\hat p_{\text{bull}}$, a vol forecast that scales exposure by
   $1/\hat\sigma$, and a leverage policy that scales the whole book by a scalar or slowly-varying
   $L(t)$ are three different multiplicative dials on the same knob, not three interacting
   structural choices about *what to trade*). Crossing all three of these fully with each other
   AND with every one of Axis A's 31 signal variants and every one of Axis F's 8 portfolio schemes
   is where essentially all of the 297,600-vs-a-few-thousand blowup comes from.

### 9.3 The proposed design: full factorial on {A, B, F}, replicated one-at-a-time on {C, D, E}

**Core (fully crossed): Axis A (signal construction) x Axis B (entry/exit control) x Axis F
(portfolio construction).** These three are treated as the axes most likely to have first-order
interactions worth resolving individually and exhaustively: A and B jointly determine the actual
shape of the per-name traded exposure path (A decides *what*, B decides *how fast/when*, and
Section 8.1 shows they compose directly with no intermediary), and F (per Established Fact 3 and
its own F.3 headline finding) is arguably the single most consequential unresolved methodological
question in this entire program — how much genuine diversification 93 names actually buys, and
whether correlation-aware weighting versus naive inverse-vol weighting changes the answer. Every
one of Axis A's 31 points is tested against every one of Axis B's 4 and every one of Axis F's 8:

$$ |A|\times|B|\times|F| = 31\times4\times8 = 992 \text{ core cells.} $$

**Replicated one-at-a-time (OAT) block: Axis C x Axis D x Axis E, anchored at a disclosed
baseline, re-run at EVERY core cell (not just once).** For each of the 992 core cells, Axis C,
Axis D, and Axis E are set to their own baseline point, EXCEPT one axis at a time, which is swept
across its full remaining grid while the other two stay at baseline:

- $C_{\text{baseline}}$ = 2-state HMM, refit every 63 trading days (Axis C's own §C.8 first-listed,
  fastest-adapting, most-tested HMM configuration).
- $D_{\text{baseline}}$ = `ewma_0.94` (Axis D's own explicitly declared "Recommended default for
  the Integration phase," §D.10).
- $E_{\text{baseline}}$ = Grossman-Zhou loose, $\alpha_{GZ}=0.50$ (Axis E's own explicitly declared
  "recommended flagship construction," §E.7 point 4).

For each core cell, the OAT block contributes:

$$ 1\ (\text{all three at baseline}) \;+\; (|C|-1)\;+\;(|D|-1)\;+\;(|E|-1) \;=\; 1+4+9+5 \;=\; 19 \text{ cells.} $$

(The "$-1$" in each term avoids double-counting the shared baseline-baseline-baseline cell, which
appears exactly once, in the leading "1".) Crucially, this OAT block is **replicated across every
one of the 992 core cells**, not anchored to a single arbitrarily-chosen "best" core cell — so
each of Axis C's 4 non-baseline points, each of Axis D's 9 non-baseline points, and each of Axis
E's 5 non-baseline points is actually tested against **992 different (A,B,F) combinations**, not
just one. This is the choice that keeps the design aligned with the owner's explicit preference
for "all permutations and combinations captured" over a single-anchor curated design: a
single-anchor OAT design would have produced only $992 + 18 = 1{,}010$ total cells (each C/D/E
non-baseline point tested against exactly one fixed core cell), which is far more compact but
throws away the ability to see whether a C/D/E choice's effect is stable across different
signal/control/portfolio constructions — exactly the kind of single-point fragility Established
Fact (5) warns against in a different context. The replicated design costs ~19x more cells than
the single-anchor version but is far more exhaustive-in-spirit, per the owner's explicit steer.

### 9.4 Final formula and count

$$ N_{\text{cells}} \;=\; \big(|A|\times|B|\times|F|\big)\times\Big[1+(|C|-1)+(|D|-1)+(|E|-1)\Big] \;=\; 992\times19 \;=\; \boxed{18{,}848} $$

This is a **15.8x reduction** from the naive full cross-product ($297{,}600 / 18{,}848 \approx
15.79$), while still (a) fully enumerating every pairwise and higher-order interaction among the
three axes judged most likely to matter (signal construction, its own entry/exit control, and
portfolio construction across the 93-name book), (b) testing **every single one of the 64
individual axis grid points declared across all six axes** (nothing from any axis's own frozen
Stage-A grid is silently dropped), and (c) testing each of the three "risk-stack" axes' (C, D, E)
own full within-axis variation against 992 different upstream signal/control/portfolio
combinations rather than just one arbitrary anchor.

**What was explicitly cut, stated plainly per the task's requirement:** every combination in which
*two or more* of {C, D, E} are simultaneously away from their disclosed baseline (e.g. a
3-state-refit-252 HMM tested jointly with `har_rv_rf63` AND CDaR-constrained leverage,
simultaneously) is NOT tested. This removes $5\times10\times6 - 19 = 281$ of the $300$ possible
{C,D,E} triples per core cell (all but the 19 in the OAT block), for a total of $992\times281 =
278{,}752$ cells cut — accounting for the full 297,600 → 18,848 reduction. The justification is
Section 9.2's point 2: these three axes are hypothesized, on inspection of what each one
mechanically does (a multiplicative regime gate, a multiplicative vol-target scale, and a
book-level leverage multiplier), to interact with each other only weakly and near-multiplicatively
relative to their own individual (marginal) effects — an assumption the replicated-OAT design
itself lets a later analyst partially check (if, say, Axis D's ranking of estimators turns out to
differ meaningfully across the 992 different A×B×F contexts it is tested in, that is directly
visible in the 992-cell replication and would be a signal that a fuller C×D×E cross is warranted
in a Stage-B follow-up — this design does not foreclose that, it just does not pay for it
up front in Stage A).

### 9.5 Computational feasibility estimate

At the same conservative ~1 second/cell estimate used in Section 9.2 (each cell: elementwise
combination of cached per-name signals, an $O(T)$ Axis-B recursion per name, Axis-F cross-sectional
aggregation, an Axis-E leverage calc — all vectorized, none requiring a fresh model fit), 18,848
cells is approximately **5.2 CPU-hours**, trivially parallelizable across the 93 names or across
grid cells (e.g. ~15-20 minutes wall-clock on a 16-core machine). The genuinely expensive,
non-parallelizable-per-cell cost — the 46 total per-name signal-generation runs (31 Axis-A + 5
Axis-C + 10 Axis-D configurations, each run once per name and cached) — is a **fixed cost
independent of $N_{\text{cells}}$**: Axis C's own §C.6 measurement of ~5-16 seconds per name per
HMM refit-window configuration implies at most $5\times93\times16\text{s}\approx 2.1$ hours for
every HMM cell across all 93 names (a generous upper bound; most names/configs measured faster),
and Axis D's GARCH/HAR-RV refits are of comparable or lesser order. This fixed signal-generation
cost is paid exactly once regardless of whether $N_{\text{cells}}$ is 1,010, 18,848, or 297,600 —
it is not a reason to prefer a smaller design, and is called out here only so a later Sweep-phase
agent budgets for it correctly as a separate, one-time cost.

### 9.6 The DD-budget dimension is a reporting cut, NOT a grid dimension — explicit clarification

Per the frozen objective (Established Fact (1): "maximize CAGR ... subject to maxDD $\le$ a
declared budget"), every one of the 18,848 cells above is simulated **once**, producing one full
daily equity curve (and hence one realized CAGR and one realized maxDD) per cell. The three
declared DD budgets — $\{20\%, 30\%, 41\%\}$ (the last being Established Fact (6)'s own
buy-and-hold benchmark maxDD, included so every cell can be honestly compared against the
benchmark on its own risk terms) — are applied **after** each cell's equity curve already exists,
as a pure filter/segmentation of the *same* 18,848 realized (CAGR, maxDD) pairs:

$$ \text{Cell } c \text{ "satisfies budget } b\text{"} \iff \text{realized\_maxDD}(c) \le b, \qquad b\in\{20\%,30\%,41\%\}. $$

For each budget $b$, the Stage-B report will show "the best CAGR among cells with
realized\_maxDD $\le b$" (and, where informative, the full distribution of CAGR across all
qualifying cells, DSR/PBO-corrected using the single shared $N_{\text{cells}}=18{,}848$
denominator regardless of which budget cut is being reported). **This does not multiply
$N_{\text{cells}}$ by 3** — there are not $18{,}848\times3=56{,}544$ simulations; there is exactly
one simulation per cell and three different, cheap, post-hoc lenses applied to the same 18,848
results. Conflating the DD-budget dimension with a genuine grid axis would have been a subtle but
consequential multiplicity-accounting error carried into Stage B's DSR/PBO correction, which is
exactly why this distinction is stated explicitly and separately here.

### 9.7 Final statement

$$ \boxed{N_{\text{cells}} = 18{,}848} $$

This is the exact, final integer to be carried forward as the multiplicity denominator for Stage
B's informational DSR (Deflated Sharpe Ratio) and PBO (Probability of Backtest Overfitting)
reporting. It reflects a full factorial across Axis A (signal extraction, 31 points), Axis B
(entry/exit control, 4 points), and Axis F (portfolio construction, 8 points) — $992$ core cells —
crossed with a disclosed, baseline-anchored, fully-replicated one-at-a-time sweep across Axis C
(regime modeling, 5 points), Axis D (vol forecasting, 10 points), and Axis E (leverage policy, 6
points) — $19$ cells per core cell — for $992\times19=18{,}848$ total cells, a 15.8x reduction
from the naive $297{,}600$-cell full factorial, with the exact cut (281 of 300 possible
{C,D,E} triples per core cell) stated and justified in Section 9.4 above.
## Section 9-REDESIGN — Making Stage A Tractable on a 2-vCPU VM (post-hoc revision)

**Why this section exists.** Section 9 above (18,848 cells/universe, 56,544 total rows across the
93-name book + SPY-alone + QQQ-alone) is preserved verbatim above, unedited, for the historical
record. It turned out to be computationally intractable on the actual hardware available to run
it (a 2-vCPU VM): a 3-name smoke test measured ~220s of per-name Stage-A signal-generation cost
alone, implying roughly **10-12 hours** wall-clock for the full run — not practical for an
exploratory Stage-A sweep. This section documents a genuine, disclosed redesign: what was
investigated, what was actually found (which differs in an important way from the working
hypothesis that motivated the investigation), what was fixed, what was cut, and the new frozen
grid and cell count that the committed `ctaA_sweep.py` actually implements and that Stage B's
DSR/PBO multiplicity correction must use going forward.

### 9-REDESIGN.1 Investigation: is Axis C (HMM) or Axis D (GARCH) refitting every single day?

The working hypothesis, going in, was that Axis C's HMM and/or Axis D's GARCH modules might be
refitting their models on every single trading day (a "fully rolling" pattern) rather than on a
realistic periodic schedule — daily re-estimation of a maximum-likelihood model at 93-name x
multi-year x multi-grid-point scale would indeed be a classic, fixable performance bug.

**Finding: this hypothesis is FALSE for both axes.** Direct inspection of
`research/cta/lib/regime_hmm.py`'s `causal_hmm_regime_posterior` and
`research/cta/lib/vol_forecast.py`'s `garch11_vol_forecast`/`har_rv_vol_forecast` confirms both
were ALREADY correctly implemented with a periodic-refit-plus-fast-forward-propagation design
exactly analogous to what the task brief was looking for: HMM refits every 63 or 252 trading days
(expanding window, `causal_hmm_regime_posterior`'s `anchors = range(min_train, T, refit_every)`),
with a cheap O(1)/day forward-filter recursion between refits; GARCH(1,1) refits every 21 or 63
trading days (`garch11_vol_forecast`'s `need_refit = (i - min_obs) % refit_freq == 0`), with a
cheap fixed-parameter GARCH recursion (`sigma2 = omega + alpha*r^2 + beta*sigma2`) between refits.
Both are already exactly as causal as a daily refit would be (proven by each module's own
injection test, Sections C.7/D.7) at a small fraction of the cost.

**Direct profiling on real SPYUSDT data (research/cta's own ad-hoc profiling script, not
estimates) confirms these axes are cheap in practice**, contrary to the working hypothesis:

| component | measured cost (SPYUSDT, all configs in the ORIGINAL grid) | share of ~225s total |
|---|---|---|
| Axis A: `build_component_signals` (default ewma/kalman/l1tf) | 12.0s | 5% |
| Axis A: 8 non-default L1TF configs | **188.5s** | **84%** |
| Axis A: ewma/kalman/full-combo buckets | 0.2s | <1% |
| Axis C: all 4 HMM configs (2-state and 3-state, refit=63/252) | 22.0s | 10% |
| Axis D: all 10 vol-forecast configs (incl. both GARCH, both HAR-RV) | 2.8s | 1% |

**The actual bottleneck — confirmed by direct measurement, not assumption — is Axis A's causal
rolling l1-trend-filter (L1TF), NOT Axis C or Axis D.** Unlike the HMM/GARCH modules, `signal_
extraction.py`'s `l1_trend_filter_rolling` (Section A.3.2/A.3.3) genuinely re-solves a fresh ADMM
optimization at **every single trading day** within its rolling window (`for t in range(window-1,
n): ... l1_trend_filter_window(w, ...)`) — this is not a bug, it is the axis's own explicitly
disclosed design (a windowed adaptation of an offline global QP, re-solved per window because
there is no analogue of HMM/GARCH's "periodic refit + fixed-parameter fast propagation" for this
estimator family), but it is nonetheless the dominant real cost, worst at the largest window/
lambda-multiplier settings (`l1tf_w252_lm2.0`: 61.4s alone, on one name, for one grid point).

This is reported plainly because it means the task brief's specific hypothesis (fix C/D's refit
cadence) does not apply here — investigating and reporting a negative finding honestly, per house
discipline, rather than "fixing" something that was not broken.

### 9-REDESIGN.2 A legitimate performance fix: ADMM warm-starting in `l1_trend_filter_rolling`

Since the l1-trend-filter's per-window objective, $\min_x \tfrac12\|y-x\|_2^2 + \lambda\|Dx\|_1$,
is **strictly convex in $x$** (the quadratic term is positive-definite everywhere), ADMM converges
to the same unique global optimum regardless of how its $(z,u)$ auxiliary/dual variables are
initialized — initialization affects only how many iterations are needed to reach the `tol`
stopping criterion, never the converged answer. Consecutive rolling windows share $n-1$ of their
$n$ points, so the previous day's converged $(z,u)$ — shifted by one index to re-align local
second-difference positions with the new window (oldest interior second-difference dropped,
newest seeded at the previous edge value) — is a good warm start for the next day's solve, exactly
the same "reuse the immediately-preceding, highly similar problem's solution" idea that makes
HMM/GARCH's periodic-refit-plus-propagation design cheap.

**Implementation** (`research/cta/lib/signal_extraction.py`): `l1_trend_filter_window` gained
optional `z_init`/`u_init`/`return_dual` parameters (all default to the exact original zero-seeded,
x-only-return behavior — every pre-existing caller and unit test is byte-identical); `l1_trend_
filter_rolling` now threads the previous window's converged, shifted $(z,u)$ through as the warm
start for the next window, resetting to a cold start after any NaN gap.

**Verification, on real SPYUSDT data:**
- The module's own `synthetic_outlier_injection_test` (shock=+0.5, on log-price) still returns
  `overall_pass: True` after the change — causality is unaffected (warm-starting only ever reuses
  STRICTLY PAST solver state, never future information).
- `research/cta/tests` (16 tests, signal-extraction unit + causality suite): **16 passed**,
  unchanged.
- Numerical equivalence check (warm-start vs. a cold-start reference re-implementation, same
  `n_iter=150`/`tol=1e-6`), on 3 representative (window, lam_mult) configs: max abs. difference in
  the fitted log-price level ranged **2.4e-5 to 4.4e-4** — small (several orders of magnitude below
  realistic day-to-day log-price moves) but not bit-identical, because neither the cold- nor
  warm-started iteration necessarily fully converges to machine precision within `n_iter=150`, so
  the two trajectories can land at very slightly different points that both satisfy the same `tol`
  criterion. This is disclosed explicitly rather than claimed as an exact reproduction.
- **Speedup**: the 8 non-default L1TF configs' combined cost on SPYUSDT fell from **188.5s to
  92.1s (~2.0x)** — a large, legitimate speedup from reduced ADMM iteration counts, not from
  changing the objective, the grid, or any causality property.

### 9-REDESIGN.3 Lever 1: a stratified 22-name Stage-A pilot universe

Per the task's explicit standard pilot/confirm pattern: Stage A now runs on a **22-name subset**
of the full 93-name reconciled base universe; Stage B's later FROZEN confirmatory pipeline will
run the one selected construction on the full 93. Selection rule (deterministic, no randomness,
reproducible from `research/cta/_universe_manifest_93.csv` via `research/cta/_build_universe22.py`,
both committed alongside this document):

0. **Known-bad-data exclusion, found and fixed during this redesign's own feasibility testing**:
   `AAPLUSDT` is dropped from the candidate pool BEFORE stratification. Its real 2020-08-31
   4-for-1 stock split was never detected/adjusted by the shared data lake's split pipeline —
   `data/stitched/AAPLUSDT.parquet` shows a fake **-74.12%** single-day return on that date, and
   `research/equity/DATA.md`'s own per-name manifest independently confirms **zero** splits
   detected for AAPLUSDT despite this being a real, undisputed corporate action (Axis E's own
   §E.6 had already flagged this exact finding in isolation but left it "out of scope for a
   dedicated fix" — this redesign independently re-confirmed it directly against the current
   parquet, since AAPL had been selected into the Stage-A pilot list before this check was run,
   and treats it as load-bearing for THIS universe choice specifically). This is a known
   main-branch data-lake bug, out of scope to fix from this worktree per the host's explicit
   prohibition on touching anything upstream — the correct, safe response is exclusion from the
   candidate pool, not a silent ride-through that would have corrupted every downstream
   Kelly/EVT/CDaR leverage calc, HMM regime fit, and GARCH vol forecast touching this name (and,
   because Axis E's leverage functions operate on the AGGREGATE 22-name book return, would have
   propagated a single fake -74% day into every one of the 9,920 Stage-A cells' leverage sizing,
   not just AAPL's own signal). Several OTHER large single-day moves elsewhere in the final list
   were individually spot-checked (comparing price levels for several days before/after each
   flagged date) and found to be genuine, PERSISTING moves — price holds at the new level
   afterward, unlike a snap-back data artifact — consistent with real small-/micro-cap
   event-driven volatility, and are correctly RETAINED: `BMNRUSDT` +665.5% on 2025-06-30 (real
   bitcoin-treasury-strategy rally, price continued 33→45→57→136 over the following days),
   `BEUSDT` -42.4% on 2019-08-13 (real secondary-offering-style drop, held near the new level),
   `AAOIUSDT` +66.8% on 2023-08-04 and `ASTSUSDT` +72.0% on 2024-05-29 (both real news-driven
   rallies that persisted). This distinction — one confirmed data bug excluded, several genuine
   extreme-but-real events retained — is stated explicitly so it is not mistaken for either an
   uninvestigated blanket exclusion or a missed data-quality issue.
1. Force-include `SPYUSDT` and `QQQUSDT` (already re-run as separate single-instrument books
   elsewhere in this sweep, and this program's own established benchmark reference names).
2. Rank the remaining 90 candidate names (91 minus the AAPLUSDT exclusion) by trailing
   dollar-volume (`mean(close*volume)` over each name's full real `source=='duka'` history) into 3
   equal-sized terciles (low/med/high liquidity).
3. Cross each liquidity tercile with 2 history-length bins (short/long, split at the
   cross-sectional median `span_days` ≈ 2,911 real duka days) → 6 strata.
4. Draw a fixed quota from each stratum (base $20\div6=3$, remainder 2 assigned to the 2
   largest-population strata, deterministic) summing to 20; within each stratum, greedily prefer a
   sector not yet represented anywhere in the running selection (tie-break: alphabetical symbol),
   maximizing sector diversity for a fixed total name count.

**Result — 22 names spanning 21 of the universe's 47 raw sector labels**: `AAOIUSDT` (optical),
`ADBEUSDT` (software), `AMATUSDT` (semicap), `AMDUSDT` (semis), `AMZNUSDT`
(consumer-disc), `ASTSUSDT` (space), `BEUSDT` (clean-energy), `BMNRUSDT` (bitcoin-proxy),
`BRKBUSDT` (financials), `BXUSDT` (financials), `CATUSDT` (industrials), `CIENUSDT` (networking),
`COINUSDT` (crypto-exchange), `CRCLUSDT` (stablecoin), `CRWDUSDT` (software-security), `DELLUSDT`
(hardware), `DISUSDT` (media), `EWJUSDT` (country-etf-japan), `EWTUSDT` (country-etf-taiwan),
`EWYUSDT` (country-etf-korea), `QQQUSDT` (index-largetech), `SPYUSDT` (index-sp500). This
deliberately includes both very-high liquidity (SPY/QQQ/AMZN, dollar-vol proxy in the millions)
and low-liquidity names (BMNRUSDT/CIENUSDT/EWTUSDT, proxy in the low hundred-thousands), and both
long-history (12 names with ≥~2,840 real duka days) and short-history names (BMNRUSDT ~370d,
CRCLUSDT ~249d, COINUSDT ~1,762d, ASTSUSDT ~1,881d, DELLUSDT ~2,713d, CRWDUSDT ~2,550d).
Removing `AAPLUSDT` from the 91-name candidate pool shifts every remaining name's dollar-volume
RANK slightly, which shifted the tercile boundaries enough that the net change versus the
originally-computed (pre-exclusion) list is **not** a clean 1-for-1 swap: the exact set difference
is `{AAPLUSDT, DKNGUSDT}` removed, `{CRWDUSDT, DELLUSDT}` added (verified directly by diffing the
two selection runs) — stated precisely here rather than glossed over as "AAPL's replacement."
Both newly-added names were independently spot-checked clean (worst days: CRWD -16.5% on
2020-03-16, DELL -19.0% on 2025-04-03 — both real, well-known macro-shock dates, no split
signature); `DKNGUSDT`'s removal is a harmless consequence of the same deterministic rule, not a
data-quality finding about DKNGUSDT itself.

A fresh, DERIVED (not hardcoded — bug pattern (b) discipline) equal-weight, unlevered,
daily-rebalanced buy-and-hold benchmark is also computed on exactly this 22-name subset
(`BENCHMARK_CAGR_22NAME` in `ctaA_sweep.py`) as an apples-to-apples complement to the established
93-name `BENCHMARK_CAGR=0.468` (Established Fact 6) — a 22-name subset's own passive benchmark
CAGR need not equal the full 93-name figure, and results below report the delta against BOTH.

### 9-REDESIGN.4 Lever 3: trimming Axis C and Axis D's OWN grids using their own stated preference

**Axis C (regime) trimmed from 5 → 3 points**: `hmm2_rf63` (baseline), `hmm2_rf252`, `sma200`.
Both 3-state HMM variants (`hmm3_rf63`, `hmm3_rf252`) are dropped, for two reasons stated together,
not just one: (a) Section C.6 already disclosed that a 3-state fit is "already at the edge of
reliable estimation for short-history names" and should be "restricted to long-history names" —
several names in the new 22-name Stage-A universe are DELIBERATELY short-history (chosen for
diversity in 9-REDESIGN.3), so 3-state would silently fall back to 2-state for a material fraction
of this specific universe regardless; (b) 3-state is also the single most expensive Axis-C
sub-variant (measured ~10.1s vs. 2-state's ~7.75s on SPY at refit=63, ~2.25x per-EM-iteration cost
from $O(K^2)$ scaling). The two retained HMM refit frequencies plus the SMA baseline still span the
full fast/slow-adaptation tradeoff Axis C's own §C.5 was chartered to test.

**Axis D (vol forecast) trimmed from 10 → 3 points**: `ewma_0.94` (baseline — Axis D's own §D.8
outright aggregate QLIKE winner), `ewma_0.97` (§D.8's outright aggregate MSE winner), and
`garch11_rf21_w756`. The third point is a **deliberate, disclosed deviation** from pure
aggregate-QLIKE ranking (which would instead pick `trailing_rv_60`, the 3rd-best aggregate
performer): §D.8 explicitly documents GARCH(1,1) as the single BEST individual estimator
specifically on SPY and QQQ (broad, cleanly mean-reverting-vol index-like names) — and this exact
sweep tests SPY and QQQ as two of its three "universes" — so dropping GARCH entirely would blind
the redesigned Stage-A grid to a documented, axis-flagged interaction on precisely the two names
most likely to exhibit it. This substitution costs nothing computationally (GARCH measured at
~0.4–1.8s/name, negligible relative to the axis's own already-small ~2.8s total), so there is no
tradeoff being made here beyond deviating from a purely-mechanical top-3 cut.

**Axis E (leverage) and Axis F (portfolio construction) are UNCHANGED** at 6 and 8 points
respectively — Axis E's leverage calc operates on the already-assembled book-level return series
(a cheap, $O(T)$ scalar calibration, not a per-name cost driver), and Axis F is part of the fully-
crossed CORE grid (not the OAT block) and was independently judged, per Section 9.3, "arguably the
single most consequential unresolved methodological question in this entire program." **Axis A
(31 points) and Axis B (4 points) are also UNCHANGED** — per 9-REDESIGN.1/2, Axis A was in fact the
real bottleneck, but it was fixed at the source (the ADMM warm-start, 9-REDESIGN.2) rather than by
shrinking its grid, precisely so its and Axis B's full, already-declared Stage-A grids ("the heart
of the mandate") survive intact into the sweep.

### 9-REDESIGN.5 New frozen grid and cell count

$$ \text{OAT block} = 1 + (|C_{\text{new}}|-1) + (|D_{\text{new}}|-1) + (|E|-1) = 1+2+2+5 = 10
\quad(\text{down from }19) $$
$$ N_{\text{cells}} = |A|\times|B|\times|F|\times\text{OAT block} = 31\times4\times8\times10 =
\boxed{9{,}920}\ \text{per universe} $$
$$ \text{Total rows} = 9{,}920 \times 3\ \text{universes (STAGE\_A\_22, SPY-alone, QQQ-alone)} =
\boxed{29{,}760} $$

This supersedes Section 9.7's $N_{\text{cells}}=18{,}848$ (per universe) /
$56{,}544$ (total rows) as the multiplicity denominator Stage B's informational DSR/PBO reporting
must use for anything derived from this Stage-A run. **What changed and why, stated explicitly per
the task's own requirement (this is a computationally-necessitated revision, not a silent
simplification):**

| lever | before | after | mechanism |
|---|---|---|---|
| Universe | 93 names | 22 names (stratified pilot) | cuts per-name signal-generation cost and per-cell Axis-B/F cost roughly proportionally; Stage B re-runs the full 93 |
| Axis A L1TF cost | 188.5s/name (cold ADMM) | 92.1s/name (warm-started ADMM) | genuine ~2x speedup, same math, verified by injection test + numerical closeness |
| Axis C grid | 5 points | 3 points | drop both 3-state variants (C.6's own disclosed short-history caveat + genuinely more expensive) |
| Axis D grid | 10 points | 3 points | Axis D's own §D.8 aggregate QLIKE/MSE ranking, +1 deliberate SPY/QQQ-specific GARCH retention |
| OAT block | 19 | 10 | direct consequence of the C/D trims above |
| Axis A, B, E, F grids | unchanged | unchanged | not the bottleneck (A/B: fixed at the source instead of shrunk; E/F: already cheap / already judged highest-value) |
| $N_{\text{cells}}$/universe | 18,848 | 9,920 | $992\times10$ vs. $992\times19$ |
| Total rows | 56,544 | 29,760 | $9{,}920\times3$ |

Every one of Axis A/B/E/F's originally-declared grid points is still tested in full; the cut is
confined to Axis C (3-state HMM, 2 of 5 points) and Axis D (7 of 10 points, chosen by the axis's
own stated ranking with one disclosed exception). Nothing was cut silently.

### 9-REDESIGN.5b Two more no-math performance fixes found during feasibility testing

1. **BLAS thread pinning.** `ctaA_sweep.py` parallelizes via `multiprocessing.Pool` (process-level
   parallelism across names / (A,B)-pairs). Left at their defaults, numpy/scipy's own internal
   BLAS thread pools additionally try to use every available core INSIDE each worker process,
   oversubscribing this VM's 2 vCPUs. `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
   `NUMEXPR_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS` are now pinned to `1` at the very top of the
   script (before numpy/scipy are imported) — a standard, math-neutral practice for
   multiprocessing + BLAS workloads.
2. **Phase-3 (SPY-alone / QQQ-alone) parallelization.** The original script computed SPY's and
   QQQ's full 9,920-cell single-instrument sweeps SEQUENTIALLY in a plain `for` loop — two fully
   independent computations, needlessly serialized onto one core while the other sat idle. This is
   now dispatched through the same fork-based `Pool` used elsewhere in the file (no math changed,
   `process_single_instrument`'s body is untouched — only how the two calls are scheduled).

### 9-REDESIGN.5c A second, more consequential bottleneck found live: aggregate_portfolio() call volume

After all of the above levers were implemented, a live feasibility launch (per 9-REDESIGN.6 below)
surfaced a SECOND, larger bottleneck that the original per-cell cost estimate
("portfolio aggregation ~4-8s/call") badly undersold: `aggregate_portfolio()`'s own per-day loop
(`for t in range(n)`, T≈2,053 real trading days, Python-level object construction dominating the
actual `risk_parity_weights` numerical solve, which itself converges in well under 100 fast
numpy-only iterations) costs roughly 2s/call on real data — and `process_ab_pair`'s F-loop calls
it **8 times** per OAT slot (once per `AXIS_F_GRID` point). A live launch measured 4 (A,B) pairs
in 345s and printed an ETA of **~173 minutes** for phase 2 alone — wildly over budget, discovered
and killed (by exact PID) before any further compute was wasted.

**Exact (not approximate) fix, requiring no change to `portfolio_construction.py`.** Two
observations, both directly verifiable from `aggregate_portfolio`'s own source:
1. Its "with spike overlay" output at every date $t$ is exactly
   $\text{final}_{\text{spike}}(t) = \text{final}_{\text{no-spike}}(t) \times \text{spike\_scale}(t)$
   — spike scale multiplies the already unit-gross-renormalized weights; with
   `apply_spike_overlay=False`, spike_scale(t) ≡ 1.0 identically. So the "on" and "off" variant of
   any (scheme, corr_window) pair never need two separate calls — compute once with
   `apply_spike_overlay=False` and multiply by the (already-cached) spike-scale series to get the
   "on" variant.
2. For `scheme="equal_risk_weight"`, the per-day weight computation does not depend on
   `corr_window` AT ALL (`corr_window` only feeds `avg_corr`, itself only consumed by the spike
   overlay) — so all 4 `equal_risk_weight` grid points (2 corr_windows × 2 spike settings) are
   derivable from a SINGLE base call. `scheme="corr_risk_parity"` genuinely needs 2 base calls
   (corr_window changes the covariance, hence the ERC weights) — each itself spike-derivable per
   point 1.

This cuts **8 calls down to 3 genuinely distinct calls per OAT slot** (implemented in
`_compute_f_variant_weights()`), verified byte-identical to the naive 8-call path on real 22-name
data: **max abs weight diff = 0.0 across all 8 `AXIS_F_GRID` points**, both before and after
applying this fix. Measured on real data: ONE full (A,B) pair (10 OAT slots × 8 derived F-variants
= 80 rows) took **66.0s** with the fix (vs. an implied ~172s/pair naive baseline from the killed
launch) — a **~2.6x** reduction, consistent with the $8/3\approx2.67$x call-count reduction.

**A third-tier refinement for the SPY-alone/QQQ-alone single-instrument books specifically**: with
only 1 possible name, `aggregate_portfolio`'s own `len(usable) >= 2` guard for entering the
`corr_risk_parity` code path can never be satisfied — scheme="corr_risk_parity" ALWAYS falls back
to the identical "equal_risk_weight_fallback" computation regardless of `corr_window`, for any
1-name book. All 8 `AXIS_F_GRID` points therefore collapse to exactly **1** real computation for a
1-name book (further verified exact by the same `len(usable)` guard `aggregate_portfolio` itself
already enforces — not a new approximation introduced here).

**Revised phase-2 estimate**: $124$ (A,B) pairs × ~66s ÷ 2 worker processes ≈ **68 minutes** for
the STAGE_A_22 book alone — still over the "well under 60 minutes" aspiration even after this
fix, on top of phase 1's own measured ~23 minutes and phase 3's (now-parallelized, further-reduced
via the single-name refinement above) SPY/QQQ sweeps. This is reported plainly rather than
papered over: see the structured output / chat report for the actual total wall-clock time the
real launch took, and Section 9-REDESIGN.6 immediately below for the go/no-go reasoning applied at
this point (the task's own explicit tolerance — "if it runs meaningfully over an hour ... let it
continue rather than killing it" — was invoked here rather than a further, riskier redesign
iteration under time pressure).

### 9-REDESIGN.6 Feasibility check before launch

Per the task's explicit requirement, the redesigned grid was estimated to fit the sub-60-minute
target BEFORE launching the real run, using a combination of the direct profiling above and a live
smoke test of the actual reduced pipeline (real 22-name phase-1 signal cache build, plus real
timed `process_ab_pair` calls) on the same 2-vCPU VM the real run executes on — see the structured
output / chat report for the exact measured numbers and the go/no-go decision that followed.

### 9-REDESIGN.7 Actual completed run — real numbers, not estimates

The redesigned grid launched via `nohup` at 2026-07-07 13:02:21 UTC and was polled to completion
(spaced `ssh` checks, never restarted or thrashed) rather than killed when it ran over the
sub-60-minute aspiration — exactly the tolerance the task itself specified ("if it runs
meaningfully over an hour ... let it continue rather than killing it"). **Real, measured phase
breakdown** (from the run's own progress log, not re-estimated):

| phase | what | wall-clock |
|---|---|---|
| 1 | per-name signal caching, 22 names (incl. the L1TF warm-start + BLAS-pinning fixes) | 1,400.0s (23.3 min) |
| 2 | STAGE_A_22 book, 124 (A,B) pairs × 80 rows = 9,920 rows | 4,507.0s (75.1 min) |
| 3 | SPY-alone + QQQ-alone, 9,920 rows each, run concurrently on the 2 workers | 5,267.0s (87.8 min) |
| **total** | | **11,185s = 186.4 min ≈ 3.1 hours** |

**Result row count: 29,760 — exactly matching the declared N_cells×3-universes figure
(9,920×3) from Section 9-REDESIGN.5**, verified directly from the written artifact
(`results_ctaA_full_grid.parquet`'s own row count and its `universe` value-counts: `STAGE_A_22`
9,920 / `SPY` 9,920 / `QQQ` 9,920).

**Honest disclosure: the sub-60-minute aspiration was NOT met** (186 minutes actual vs. the
Section 9-REDESIGN.5c estimate of ~91 minutes total, itself already flagged there as over
budget). The single biggest reason the actual run exceeded even that revised estimate: **Phase 3
(87.8 min) took LONGER than Phase 2 (75.1 min) despite each Phase-3 OAT slot needing only 1 real
`aggregate_portfolio`-equivalent call vs. Phase 2's 3** (the `_compute_f_variant_weights`
single-name collapse, Section 9-REDESIGN.5c, IS implemented and verified correct — this is a
real, additional finding about ITS COST, not a bug in whether it fires). This is consistent with,
not contradictory to, 9-REDESIGN.5c's own root-cause diagnosis that `aggregate_portfolio`'s
per-day loop cost is dominated by **Python-level object construction repeated for every one of
T≈2,053 trading days, not by the numerical ERC solve itself** — collapsing 8 calls to 1 removes
7 solves' worth of arithmetic but each surviving call still pays the same O(T) object-construction
floor, so reducing call COUNT saves less wall-clock than the naive "8x fewer real calls implies
8x faster" arithmetic suggests. This is reported plainly as a genuine finding for any future
Stage-A/Stage-B re-run to account for (a further, real speedup would require optimizing
`aggregate_portfolio`'s per-day loop body itself in `portfolio_construction.py`, out of scope for
this already-completed redesign pass), not papered over as a rounding error.

**Artifact format note**: the repo's own root `.gitignore` blanket-excludes `*.parquet` anywhere
in the tree (a deliberate data-hygiene guardrail, not specific to `data/`). Per this task's own
explicit allowance ("`results_ctaA_full_grid.parquet` (or `.csv.gz` if parquet is awkward)"),
the version committed to `equity-cta` is `research/cta/results_ctaA_full_grid.csv.gz`
(gzip-compressed CSV, 29,760 rows × 33 columns, ~1.46 MB, byte-for-byte derived from the real
`.parquet` run artifact via `pd.read_parquet(...).to_csv(..., compression="gzip")` — no rows or
columns altered) rather than fighting the repo-wide ignore rule with a forced add. Load it with
`pd.read_csv("research/cta/results_ctaA_full_grid.csv.gz")`.

**Headline result (STAGE_A_22 book, cost=13bp primary, funding=0)**: best cell by
CAGR-at-30%-DD-budget = **18.49% CAGR** (`kalman_1e-06_1e-08_0.0001` × `aggressive` ×
`hmm2_rf63` × `ewma_0.94` × `cdar` × `corr_risk_parity_w120_spike`, maxDD 28.10%, Sharpe 1.03),
a **CAGR delta of −28.3 percentage points vs. the established 46.8% 93-name benchmark** (Fact 6)
— consistent with, not an exception to, the program's standing finding that every trend-timing
construction tested so far (I1, the four prior exploratory CTA scans, and now this redesigned
Stage-A grid) underperforms passive buy-and-hold on this sample. 1,984 of 9,920 STAGE_A_22 cells
(20%) satisfy the ≤30% DD budget at all; the worst cells (several `evt_ceiling`-leverage,
`aggressive`-entry-control combinations) hit outright ruin (`cagr=-1.0`, `ruin_flag=True`,
realized average leverage ~9.4–10.0x) — a genuine, separate risk-policy finding (EVT-ceiling
leverage sizing is not uniformly safe across Axis-A/B combinations) reported here rather than
filtered out of the sample. Full best-10/worst-10/SPY/QQQ tables are in the chat report (this
document's own numerical claims here are the same numbers, reproduced for the permanent record).

## Section 10 — Test Suite and Commit Record

**Full repo test suite**, run before this document and every axis module were committed, per this
program's METHODOLOGY requirement:

```
cd /home/speculation/convexity-cta && PYTHONPATH=/home/speculation/convexity-cta/src:/home/speculation/convexity-cta \
  /home/speculation/convexity/.venv/bin/python -m pytest tests -q
```

Result: **654 passed, 1 skipped, 9 warnings** — unchanged from the established baseline carried
through E1/E2/E3/I1 and every prior CTA axis commit (`ac8bb0b`, `dee7423`, `961be81`, `e64b320`).
The 9 warnings are pre-existing `RuntimeWarning: Mean of empty slice` occurrences in
`tests/signal/test_rv_pairs.py` / `src/convexity/signal/rv_pairs.py`, unrelated to any code
touched by this document or the six CTA axis modules.

**Module presence verification** (Section 8.7 above): all six axis modules and their declared
validation/diagnostic artifacts were confirmed present on disk at their reported paths in the
`equity-cta` worktree before this document was written. Nothing was found missing; nothing needed
re-fetching or rewriting.

**This document** (`research/cta/CTA_MATH_NOTES.md`) was committed to branch `equity-cta` and
pushed to `origin/equity-cta`. Commit hash and push confirmation follow in the structured output
accompanying this report.
