# REGIME.md — build spec for the market-state detector gating the cross-sectional book

This file is the authoritative build spec. Build to the contracts and invariants
below. Where this file and your own instinct disagree, this file wins. Ask
before deviating from any **Invariant**.

**This spec supersedes the pre-pivot REGIME design.** The prior draft (referenced
in `docs/ARCHITECTURE.md` §4/§5.2 as a single-instrument directional-breakout
gate emitting `g_t ∈ {1, .75, .5, .25}` from absorption/turbulence/corr-spike
measures on ONE book) belonged to the directional-breakout system, which is
**PROVEN DEAD** (gross directional R≈0 across timeframes, exits, filters,
leverage, three asset classes, even inverted — see `research/xsec/` README and
the VolArb backtest finding; background only, not re-litigated here). The live
edge is the cross-sectional momentum book (`research/xsec/{panel,signals,
portfolio,accounting,engine,overlay,cpcv,metrics,validate}.py`). **The risk unit
this spec gates is no longer "one trade's R-multiple" — it is next week's
portfolio return** of a dollar-neutral, ~50-name, weekly-rebalanced book. Every
formula, invariant, and worked example below is rebuilt on that unit. The
portable math salvaged from the old draft is exactly: the CUSUM/ARL
change-detection machinery (Page 1954), the single-scalar-channel discipline
(never a second `min()` term, never a multiplicative combination with LEVERAGE),
and the ban on continuously-refit ML classifiers. Everything else — the
per-instrument absorption/turbulence measures, the `g_t∈{1,.75,.5,.25}` ladder
tuned for a single-instrument breakout book, the §6.1 corr-spike worked example
— is discarded as inapplicable to a cross-sectional weekly book.

REGIME is the **defensive-overlay gate**: it does not generate the momentum
signal, does not size positions, and does not pick the universe. It answers one
question, once per week, at the Friday-close rebalance clock the whole xsec
engine runs on:

> Given the market state observable at `F_t` (this Friday's close), should NEXT
> WEEK's book run at full risk, reduced risk, or flat — and if flat, was this
> triggered by an isolated, plausibly-transient dispersion spike, or by a
> persistent regime-bleed that a naive stop would only make worse?

REGIME composes with the sibling modules without importing their internals:
- The **xsec engine** (`src/convexity/portfolio/engine.py`) computes the raw weekly
  dollar-neutral net return from the momentum signal — REGIME never touches
  signal computation.
- `src/convexity/regime/overlay.py` **already implements the overlay MECHANICS**
  (`vol_target_scale`, `beta_hedge_returns`, `drawdown_stop_scale`) as pure
  causal transforms of a return series. REGIME does **not** duplicate these. It
  is the missing **detector** that decides *when* and *how hard* to apply them,
  replacing the ad-hoc fixed-threshold calibration those functions currently
  take as keyword defaults (`dd_trigger=0.15`, `target_vol_ann`, …) with a
  statistically calibrated trigger (Section 3) and — critically — with the
  honest finding of Section 5 that a DD-stop can be net-harmful on a
  persistent-bleed regime and must not be applied uncritically to momentum
  just because it was tested on carry.
- `src/convexity/system/validate/{deflated,cpcv,walkforward}.py` are REUSED
  (not reimplemented) for REGIME's own promotion gate (Section 7), exactly as
  `src/convexity/system/cpcv.py` already reuses `SystemCPCV` for the momentum family.

REGIME never opens, holds, reduces, or routes anything. It emits one scalar
multiplier `g_t ∈ [0, 1]` (never a vector, never a per-instrument value) that
scales the book's target gross exposure for the *next* rebalance, plus a
labeled state (`HEALTHY`, `WATCH`, `STRESSED`, `HALT`) and a reason chain. The
consumer (the xsec engine's weekly rebalancer, or — once composed — SYSTEM) is
the decider; REGIME is a provider, mirroring COST's P2 (`COST.md` §0).

Scope: the crypto cross-sectional momentum sleeve (Binance USDT-M perps,
weekly Friday-close rebalance, dollar-neutral, ~50-name point-in-time universe).
The equity/commodity sleeves (`src/convexity/panel/sleeve.py` `ClassFriction`
table) share the same detector machinery through the same `Panel` abstraction
but are out of scope for the worked numbers below, which are crypto-specific.
Compute target: closed-form, `O(n_days · n_symbols)` per week, on the existing
2-core/11GB ARM VM — no retraining loop, ever (see P4 below).

**Notation note (cross-spec index alignment).** This spec indexes the weekly
rebalance grid by `t` (`D_t`, `V_t`, `S_t`, `g_t`), matching SIGNAL.md (`F_t`,
`U_t`, rank at rebalance day `t`) and LEVERAGE.md (`R_t`, `G_t`) exactly —
these three specs share one panel and one rebalance clock, so `t` means the
same Friday close everywhere across them. SYSTEM.md's own promotion-gate
math instead indexes the same weekly grid with `k` (`R_k`, path Sharpe over
`k=1..n_weeks`), because SYSTEM's CPCV/DSR derivations are stated generically
over "the n observations of a track record" and reserve `t` for other uses
internal to that derivation. **The two symbols denote the identical weekly
observation grid** (SYSTEM.md §1: "one observation = one calendar week"); a
reader moving between REGIME/SIGNAL/LEVERAGE's `t`-indexed formulas and
SYSTEM's `k`-indexed ones is not crossing a real definitional boundary, only
a per-file symbol choice — flagged here once so it is never re-derived or
mistaken for two different clocks.

---

## 0. Governing principles (read before anything else)

**P1 — The risk unit is next week's portfolio return, not one trade's
R-multiple.** Every statistic REGIME watches, and every threshold it is
calibrated against, is expressed on the **weekly, non-overlapping** observation
grid the xsec engine already uses (`src/convexity/system/metrics.py`
`PERIODS_PER_YEAR = 365.25/7 ≈ 52.18`). A regime "event" is a week or a
short run of weeks, not a single trade's excursion past a stop. Any borrowed
formula from `COST.md`/`SYSTEM.md` that was expressed in bar-count or
trade-count units is re-derived here on the weekly grid before use (e.g.
`H_sys`-style horizon caps become **week counts**, not bar counts).

**P2 — One scalar channel, never a second `min()` term (inherited, corrected).**
`g_t` is a single multiplier on the book's target gross exposure. It is
**consumed exactly once**, multiplicatively, by whichever downstream sizing
layer sets gross (today: the xsec engine's weekly `gross` parameter fed to
`portfolio.quantile_weights(..., gross=g_t · gross_target)`; once LEVERAGE is
built: `LEVERAGE.md`'s single gross-exposure scalar `G_t` — i.e. the
live channel is `gross_live = g_t · G_t`, per the resolved cross-spec mismatch
in `docs/ARCHITECTURE.md` §5.2, "build to LEVERAGE.md — single channel," now
current under LEVERAGE.md's OWN post-pivot notation, `G_t`, not the old
draft's `H_book`/`L_portfolio` naming that `docs/ARCHITECTURE.md` §5.2 still
uses verbatim from the pre-pivot resolution note). REGIME never emits a
second independent ceiling, never emits a per-instrument adjustment, and
never appears as an additive term. This is the one piece of the old draft's
architecture that is genuinely portable and is kept unchanged.

**P3 — Closed-form, no-ML, calibrated once offline (hard constraint, not a
style preference).** Regime detection uses ONLY: CUSUM sequential
change-detection (Page 1954) with ARL calibration, EWMA, method-of-moments,
Ledoit-Wolf-shrunk covariance (Ledoit & Wolf 2004) with Mahalanobis distance
where a multivariate dispersion statistic is used, moving-block bootstrap
(Künsch 1989; block length per Politis & White 2004) for threshold CIs, and
rank/order statistics. **Forbidden absolutely:** neural networks,
gradient-boosted trees, any learned/fitted combination of regime features, any
hyperparameter grid search beyond a small enumerated BLUE grid, and — the
specific temptation this pillar invites — a **continuously-refit Hidden Markov
Model**. Hamilton (1989) regime-switching is cited **only as the textbook
reference point this system deliberately does NOT implement**: an EM-fitted HMM
re-estimates its transition matrix and emission parameters as new data arrives,
which is exactly the "expensive/overfit-prone retraining" this whole repo's
`pyproject.toml` doctrine ("scipy: closed-form (NOT ML)") forbids. CUSUM with a
fixed, offline-calibrated `(k, h)` pair is the discrete-state alternative that
needs no retraining loop and ships a provable false-alarm rate (Section 3).

**P4 — The detector is causal; the calibration is offline and frozen.** The
live statistic `S_t` at week `t` uses only weekly closes/dispersion up to and
including week `t`'s Friday close (the same `F_t` the xsec engine's signals
are computed at, `src/convexity/panel/panel.py`: "known only AT day D's close"). The
CUSUM parameters `(mu_0, k, h)` are fit ONCE, offline, on the pre-pivot history
using the class-level regime windows already defined in
`src/convexity/panel/sleeve.py` (`CRYPTO_REGIMES`: `2021_bull, 2022_bear,
2023_recovery, 2024_bull, 2025_2026`) and then FROZEN for live use — no online
re-estimation of `(mu_0, k, h)` from the live stream (that would be exactly the
forbidden continuous refit of P3, just relabeled as "calibration"). Re-
calibration is a discrete, versioned, offline event (BLUE-grid governed,
Section 6), never a per-week update.

**P5 — Honest asymmetry: a stop that helps a tail event can hurt a persistent
bleed, and this must be TESTED, not assumed.** Our own tested result on the
funding-carry book (`src/convexity/regime/overlay.py` `drawdown_stop_scale`,
digest below) is that a causal DD-stop **caps drawdown at ~24% but collapses
Sharpe from 0.57 to 0.10 and funding income from ~12.9%/yr to ~0.6%/yr**,
because that book's ~70-75% drawdown is a **persistent regime-bleed** (the
2023-24 low-funding stretch), not a dodgeable isolated tail. A stop-and-flatten
rule that is correct for a tail event is actively harmful applied to a bleed: it
locks in the underperformance during the bleed and then, because it re-arms
slowly (or not causally optimally), misses the recovery. REGIME formalizes
the FIRST-PASSAGE / optimal-stopping condition (Section 5) under which a DD-stop
is net-positive vs net-negative, and explicitly REFUSES to assume the momentum
book's DD-stop behavior transfers from carry's — Section 5.4 specifies the
exact validation (regime-conditional ablation) that must pass before a DD-stop
is promoted for momentum, exactly mirroring how COST/SYSTEM forbid promoting a
RED constant without validation (`COST.md` §4 Invariant C-cfg).

**P6 — Conservative under uncertainty (inherited from COST P6).** A missing or
stale input (a panel gap, a `Panel.listed` universe collapse, a funding-feed
outage) makes REGIME report the **more defensive** state, never the more
permissive one. If the CUSUM statistic cannot be computed (insufficient
trailing history), REGIME emits `WATCH` with `g_t` at the WATCH-band value
(Section 4), never `HEALTHY`'s `g_t=1.0`, and never a crash/exception that
silently leaves the caller at its last-known (possibly `HEALTHY`) `g_t`.

**P7 — Decomposable and attributable (inherited from SYSTEM P8).** Every `g_t`
change carries a reason chain: which detector fired (dispersion-CUSUM,
BTC-vol-CUSUM, drawdown state machine), at what statistic value, against what
threshold. **Verified as real code, not just named in prose:** `src/convexity/
core/types.py` DOES already contain a frozen `RegimeState` dataclass and
`src/convexity/core/protocols.py` DOES already contain a `RegimeEngine`
Protocol with a `regime_state(...) -> RegimeState` method — these are not
aspirational. **But the EXISTING `RegimeState`/`RegimeStateName` in
`core/types.py` is the OLD pre-pivot shape** (`g_t` implied `∈{1,.75,.5,.25}`
floored `.25`, `RegimeStateName = Literal["CALM","NORMAL","STRESSED","CRISIS"]`,
`binding_sub: Literal["vol","corr","turb","liq"]`) and does **not** match this
spec's new `HEALTHY/WATCH/STRESSED/HALT` states, continuous `g_t∈[0,1]`
floored at exactly `0`, or the dispersion/BTC-vol-CUSUM reason-chain vocabulary
of Section 4. **This is an open cross-file migration this spec flags but does
not itself resolve** (a code change, out of scope for a spec-only pass):
`core/types.RegimeState` and `core/protocols.RegimeEngine` must be updated to
the Section 4/Section 8 vocabulary before REGIME's implementation can honestly
satisfy this Invariant — until then, "the `core/types.py` boundary object" is
a real dataclass with the WRONG shape, not a nonexistent one, and a builder
must not assume today's `RegimeState` needs no changes.

Forbidden in v1: any ML / learned regime classifier (P3); a continuously-refit
HMM (P3, explicitly — Hamilton 1989 is cited as the road NOT taken); `g_t`
entering as a second additive/multiplicative ceiling alongside LEVERAGE's
ceilings rather than through the single `g_t · G_t` channel (P2); online
re-estimation of CUSUM `(mu_0, k, h)` from the live stream (P4); a DD-stop
promoted for momentum by assuming the carry-book calibration transfers (P5);
`g_t` silently defaulting to `1.0` (permissive) on a data outage (P6); a
per-instrument `g_t` (P2 — REGIME never adjusts one leg of the book
differently from another); reusing the old draft's `g_t∈{1,.75,.5,.25}`
per-instrument absorption/turbulence bands without re-deriving them from a
cross-sectional weekly statistic (this entire spec is that re-derivation).

---

## 1. What REGIME builds ON TOP OF the existing panel/overlay machinery

REGIME does not rebuild the panel, the signal, the portfolio construction, or
the overlay mechanics. It consumes:

| Existing module (do NOT rebuild)                             | REGIME consumes it as                                   |
|----------------------------------------------------------------|----------------------------------------------------------|
| `src/convexity/panel/panel.py` `Panel.close/listed/funding`          | the raw daily/weekly data REGIME's statistics are built from |
| `src/convexity/panel/panel.py` `build_dollar_volume`, `trailing_dollar_volume` | a liquidity/turbulence input if a multivariate measure is used (Section 2.4) |
| `src/convexity/portfolio/engine.py` (weekly net-return series)            | the book-level return series the drawdown state machine (Section 5) watches |
| `src/convexity/regime/overlay.py` `vol_target_scale`, `beta_hedge_returns`, `drawdown_stop_scale` | the MECHANICS REGIME's detector gates — REGIME decides trigger levels, these functions execute them |
| `src/convexity/system/validate/deflated.py` `DeflatedSharpe`   | REGIME's own promotion gate reuses this, exactly as `src/convexity/system/cpcv.py` reuses `SystemCPCV` |
| `src/convexity/system/validate/cpcv.py` `SystemCPCV`           | REGIME's threshold-stability validation (Section 7) |

> **Invariant N — no re-implementation of panel/overlay/validate.** REGIME's
> `panel/build.py` (per the existing `docs/ARCHITECTURE.md` §2 tree) is a thin
> adapter that reads `Panel` fields; it does not recompute daily closes,
> funding sums, or listed masks. A test asserts `panel/build.py`'s outputs are
> a strict re-key of `research.xsec.panel.Panel`, not an independent
> recomputation.

---

## 2. The market-state statistic — cross-sectional dispersion, not per-instrument absorption

### 2.1 Why dispersion, not the old per-instrument absorption/turbulence pair

The pre-pivot draft watched **per-instrument** absorption ratios and a
Mahalanobis-style turbulence measure across a small directional-breakout
universe, because that book's risk was concentrated in a handful of live
single-instrument trades. The cross-sectional momentum book's risk is
structurally different: it is **dollar-neutral** (Section 0, P1) and its P&L is
driven by the **cross-sectional spread of returns** (momentum needs winners to
keep winning and losers to keep losing) and by the **realized volatility of the
funding/basis-driving asset** (BTC, whose regime historically leads altcoin
regime shifts). Two closed-form, no-ML statistics, both derived from the
`Panel` the xsec engine already builds:

1. **Cross-sectional return dispersion** `D_t` — the statistic a
   quintile-spread strategy's edge and risk both scale with.
2. **BTC realized volatility** `V_t` — a market-wide stress proxy that is
   available even in weeks with a thin cross-section (early history, per
   `universe_size_by_year`).

Both are **univariate weekly scalars** (never a covariance matrix fitted
per-week — see Section 2.4 for why a full covariance-based turbulence measure
is explicitly NOT the default), so each feeds its own single-statistic CUSUM
(Section 3) rather than a fitted multivariate model.

### 2.2 Cross-sectional dispersion `D_t` (worked derivation)

**Definition — reusing SIGNAL.md's own eligible-universe notation, not a new
symbol for the same thing.** Let `U_τ ⊆ {1,...,N}` be the point-in-time
eligible cross-section at day `τ` (SIGNAL.md §2.1's `U_t`, `research/xsec/
panel.py` `Panel.listed[τ]`) — exactly the same eligibility mask
`src/convexity/signal/signals.py` uses for the momentum rank, so `D_t` and the
momentum signal are computed over an IDENTICAL peer set at every date, by
construction (there is no separate "REGIME universe"). Define the
cross-sectional dispersion as the **cross-sectional standard deviation of
daily returns**, aggregated to the week:

```
r_{i,τ} = close[τ, i] / close[τ-1, i] − 1                    (daily simple return, symbol i, day τ)
D_τ     = std_{i ∈ U_τ} ( r_{i,τ} )                           (cross-sectional std across ELIGIBLE i ∈ U_τ, one day τ)
D_t     = sqrt( (1/5) · Σ_{τ ∈ week t} D_τ²  )                (RMS of daily cross-sectional std over rebalance week t)
```

(`t` here indexes the WEEKLY/rebalance grid — one value of `t` per Friday
close, per P1 — while `τ` indexes the daily grid within week `t`; this
two-index convention, daily `τ` aggregated up to weekly `t`, is used
consistently through Sections 2-4 exactly as introduced here.)

**Derivation of the RMS aggregation (why not a simple mean of `D_τ`).** `D_τ` is
itself a standard deviation (a second-moment object); averaging standard
deviations directly is not variance-additive and biases low relative to the
week's actual dispersion of returns. If daily cross-sectional returns within
the week are treated as approximately uncorrelated draws from a
(day-varying-scale) distribution, the week's aggregate variance is the average
of the daily *variances*, `E[D_t²] ≈ (1/5)Σ_τ D_τ²`, so the week's dispersion
scale is the **root-mean-square**, not the arithmetic mean, of the daily
values — the same variance-additivity argument that justifies aggregating
daily vol to weekly vol as `σ_week = σ_day·√5` under i.i.d. daily returns. This
is a first-order (independence) approximation; day-to-day dispersion is itself
autocorrelated (dispersion clusters, exactly like volatility), which is why the
CUSUM detector (Section 3), not the raw `D_t` level, is what triggers a
regime call — CUSUM accumulates evidence of a **level shift** and is far less
sensitive to this approximation error than a single-day threshold would be.

**Why dispersion (not just BTC vol) is the right cross-sectional stress proxy.**
A weekly quintile-spread book's realized return variance is, to first order,
proportional to the **spread** between the long-quintile and short-quintile
mean returns, which itself scales with cross-sectional dispersion: wider
dispersion means both larger potential edge AND larger potential adverse
spread compression risk (a dispersion COLLAPSE, not just a rise, historically
coincides with momentum crowding/unwind — see the edge-case in Section 8).
`D_t` is therefore the natural univariate proxy the CUSUM watches for **both**
directions of anomaly (Section 3.3 uses a two-sided scheme for this reason).

### 2.3 BTC realized volatility `V_t`

```
V_t = std_{τ ∈ week t}( r_{BTC,τ} ) · sqrt(365.25)        (annualized weekly RV of BTC daily returns)
```

Simple annualized realized vol of BTC's own daily returns over the week,
computed directly from `Panel.close[:, btc_col]` (no separate data source).
`V_t` proxies market-wide stress independent of the cross-section's breadth,
so it remains informative even in the sparse early-history weeks where `D_t`
is estimated over very few names (`universe_size_by_year`, `research/xsec/
panel.py`).

### 2.4 Why NOT a fitted covariance-based turbulence/absorption measure by default

The old draft's absorption ratio (Kritzman & Li 2010-style: fraction of total
variance explained by the top-k principal components of a returns covariance
matrix) is a legitimate closed-form (non-ML) statistic, but it requires
estimating an `n_symbols × n_symbols` sample covariance matrix from ~50-75
symbols on a weekly cadence with only ~250-260 historical weekly observations
total (`src/convexity/system/cpcv.py`: "~259-week timeline"). This is exactly the
regime the **Marchenko–Pastur / random-matrix-theory eigenvalue-spreading
argument** warns against: when the number of assets `N` approaches the number
of observations `T` (here `N/T ≈ 50/50 ≈ 1` in-sample, or worse in any rolling
window shorter than the full history), the sample covariance matrix's smallest
eigenvalues are dominated by estimation noise, not true low-variance
directions, and its largest eigenvalues are inflated by the same noise
(Marchenko & Pastur 1967; Laloux, Cizeau, Bouchaud & Potters 1999 applied this
to financial correlation matrices). An absorption ratio computed on such a
matrix is not a stable turbulence signal; it is close to a noise statistic
dressed as one, and a **fitted mean-variance optimizer over that same
noisy matrix is exactly what P3 already forbids for closed-form portfolio
construction** — the same argument applies to a covariance-derived regime
detector. REGIME therefore does NOT use a raw sample-covariance absorption
ratio as its primary statistic (Section 2.2/2.3's univariate `D_t`/`V_t` avoid
the `N×N` estimation problem entirely — each is a single cross-sectional or
single-asset scalar per week, with sampling error that scales with `N`, not
`N²`).

**If a covariance-flavored measure is used at all** (e.g. as a BLUE-gridded
research variant, never the default live detector), it MUST be **Ledoit-Wolf
shrunk** (Ledoit & Wolf 2004: shrink the sample covariance `Σ_sample` toward a
structured target `F` — e.g. the constant-correlation or single-factor target
— with the closed-form optimal shrinkage intensity `δ* = min(1, max(0, (Σ_τ
‖Σ̂_τ − Σ̄‖²_F ) / (n·‖Σ̂ − F‖²_F)))`, the analytic estimator of Ledoit & Wolf's
Theorem 1, computed once per fit — no numerical optimization loop, so this
remains closed-form and NOT an "ML fit" in the forbidden sense), and any
distance measure over it MUST be the **Mahalanobis distance**
`M_t = sqrt((r_t − μ̂)ᵀ Σ̂_LW⁻¹ (r_t − μ̂))` against the shrunk covariance, never
a raw-sample-covariance Mahalanobis distance (which inherits the same
ill-conditioning) and never an EM-fitted Hidden Markov Model's state posterior
(explicitly rejected, P3). This is recorded as the accepted **fallback
construction** should a future BLUE-grid variant want a multivariate
statistic; it is NOT part of the v1 live detector, which uses only `D_t`
and `V_t`.

> **Invariant MP — no unshrunk covariance in any regime statistic.** Any
> regime-detection code path that estimates a covariance/correlation matrix
> over more than a handful of series MUST apply Ledoit-Wolf shrinkage before
> any inversion or eigendecomposition; a raw `np.cov`/`np.corrcoef` feeding
> directly into a Mahalanobis distance or an absorption ratio is the bug (the
> Marchenko-Pastur argument of Section 2.4). Test (two halves, BOTH binding in
> the detector scope): (i) the STRUCTURAL half — the v1 live detector
> (`measures/absorption.py`, `measures/vol_regime.py`) contains NO `np.cov`/
> `np.corrcoef` feeding an absorption ratio / Mahalanobis distance
> (`tests/regime/test_measures.py::test_v1_detector_uses_no_unshrunk_covariance`);
> (ii) the EMPIRICAL DEMONSTRATION — on a synthetic `N=50, T=50` i.i.d. panel
> (no true structure; `regime/synth/generate.py::iid_no_structure_panel`), the
> UNSHRUNK sample absorption ratio (`regime/measures/turbulence.py`) fires above
> a nominal threshold (3×`1/N`, i.e. 3× the population absorption ratio) on
> essentially every draw, while the Ledoit-Wolf-SHRUNK estimate fires on almost
> none, at the same threshold — this is the empirical demonstration that
> motivates defaulting to Section 2.2/2.3's univariate statistics instead
> (`tests/regime/test_mp_demonstration.py`). The full §7 validation report
> (`regime/validate/threshold_stability.py` etc.) remains the scoped-out
> reporting layer; the invariant's DEMONSTRATION itself is now binding in the
> detector scope via `turbulence.py` + `synth/generate.py` (§9).

---

## 3. CUSUM change-detection with ARL calibration (the core detector)

### 3.1 The standard one-sided CUSUM statistic (derivation, per Page 1954)

Let `x_t` be the weekly statistic under surveillance (either `D_t` or `V_t`;
**`sigma` throughout this Section 3 is bound to whichever series is under
surveillance — `sigma = std(D_t)` when watching dispersion, `sigma = std(V_t)`
when watching BTC vol, both fit exactly as Section 3.4 specifies: the sample
std over the frozen in-control training window. There is no third, generic
"Gaussian-stream sigma" — every appearance of `sigma` in this section IS one
of these two fitted quantities**), and let the **in-control** (healthy-regime)
mean be `mu_0` (bound the same way: `mu_0 = mean(D_t)` or `mean(V_t)` over the
same window) and a **reference value** `k` (the "allowable slack",
conventionally half the shift-to-detect, Section 3.2). Page's (1954) one-sided
upper CUSUM for detecting an **upward** shift (dispersion/vol spiking — the
STRESSED direction) is:

```
S_0 = 0
S_t = max( 0,  S_{t-1} + (x_t − mu_0 − k) )
```

**Derivation of why this is the right statistic (sequential probability ratio
motivation, not asserted).** CUSUM is the sequential analogue of the
Neyman-Pearson likelihood-ratio test for a mean shift from `mu_0` to
`mu_1 = mu_0 + 2k` in a Gaussian stream with the SAME known variance `sigma^2`
just bound above: the log-likelihood ratio for one observation is
`ℓ_t = (mu_1 − mu_0)/sigma^2 · (x_t − (mu_0+mu_1)/2) = (2k/sigma^2)·(x_t − mu_0 − k)`.
Accumulating `ℓ_t` and resetting at zero whenever the running sum goes negative
(discarding evidence against a shift, since a change could have started at any
point) gives exactly `S_t` up to the positive scale factor `2k/sigma^2`, which
does not affect the crossing time. So the CUSUM recursion above is a
constant-rescaling of the cumulative log-likelihood-ratio SPRT statistic for a
one-sided mean shift, restarted at its own running minimum. **Two distinct
claims follow, from two distinct sources — kept separate here, unlike this
spec's earlier draft, which ran them together:**
(i) Page's (1954) original construction and its stopping rule (`S_t≥h`, the
alarm rule below) is exactly the SPRT-restarted-at-its-minimum statistic just
derived — this is Page's own contribution, needing no further citation beyond
the derivation above; (ii) that this specific construction is
ASYMPTOTICALLY MINIMAX-OPTIMAL for expected detection delay among ALL
stopping rules with a given false-alarm rate is a SEPARATE, later theorem, due
to Lorden (1971), proved under a worst-case-changepoint criterion Page's 1954
paper does not itself establish. Montgomery's SPC textbook is cited only as a
practitioner-level restatement of Page's recursion (no SPRT derivation, no
optimality proof), never as a source for either (i) or (ii).

**The alarm rule.** REGIME signals an upward-shift alarm the first time
`S_t ≥ h` (the decision interval / threshold), and RESETS `S_t → 0` upon
alarm (a fresh accumulation starts after each flagged regime change,
consistent with Page's original stopping-rule construction).

### 3.2 Average Run Length (ARL) and the calibration of `(mu_0, k, h)`

**ARL definition and the two operating points that matter.** `ARL(mu)` is the
expected number of observations (here: WEEKS, per P1's risk-unit
re-derivation) until `S_t` first crosses `h`, as a function of the true
underlying mean `mu`. Two values matter:

- `ARL_0 = ARL(mu_0)` — the **in-control** ARL: how many weeks, on average, a
  HEALTHY regime runs before a false alarm. This is the quantity P3/P4 require
  REGIME to state explicitly as a **low annual false-alarm probability**.
- `ARL_1 = ARL(mu_1)` — the **out-of-control** ARL at the shift size the
  detector is tuned to catch: how many weeks, on average, it takes to detect a
  genuine shift of size `mu_1 − mu_0 = 2k`.

**Closed-form (Wald-approximation) ARL formula, derived.** The random walk
under surveillance is the CUSUM increment itself, `w_t = x_t − mu_0 − k`
(sigma-standardize as `theta = (mu − mu_0)/sigma − k/sigma`, i.e. `theta` is
the standardized DRIFT of the random walk `S_t` when the true mean is `mu` —
**not** the standardized distance of `mu` from `mu_0` alone; the `−k/sigma`
term is the random walk's own reference-value drag and must stay inside
`theta`, never be re-substituted afterward). For a one-sided CUSUM on i.i.d.
Gaussian observations, the Wald / Siegmund (1985) approximation to the ARL —
obtained by treating `S_t` as a random walk with drift `theta` reflected at 0
and absorbed at `h/sigma`, applying Wald's identity for a random walk's
first-passage time to this one-sided-reflecting/one-sided-absorbing boundary
problem, with Siegmund's `1.166·sigma` correction for the discreteness of the
reflecting boundary at 0 — is:

```
ARL(mu) ≈ ( exp(−2·theta·b) − 1 + 2·theta·b ) / ( 2·theta² )     for theta ≠ 0,   b := h/sigma + 1.166
```

**The in-control special case is a Δ→0 LIMIT of this SAME formula taken in
`mu`, not a substitution of `theta = −k/sigma` into it — these are different
operations and give different numbers; conflating them is a derivation bug
this spec previously committed and now corrects.** Write `theta` as a function
of the shift-from-`mu_0`, `Δ := (mu − mu_0)/sigma`, so `theta(Δ) = Δ − k/sigma`.
`ARL(mu)` above is smooth in `theta`, hence in `Δ`, everywhere except the
removable singularity at `theta = 0`, i.e. at `Δ = k/sigma` (the point where
the random walk `S_t` has exactly zero drift — **not** `Δ = 0`, i.e. **not**
`mu = mu_0`). Two consequences, both checked numerically below (`k=0.0045`,
`sigma=0.0060`, `h=0.05419`, so `k/sigma = 0.75`, `b = 10.198`):

1. **The textbook `b²` formula is the EXACT limit at `mu = mu_0 + k`** (where
   `theta = 0`), by L'Hopital / second-order Taylor expansion of the numerator
   in `theta`: `exp(−2θb) − 1 + 2θb = 2θ²b² − (4/3)θ³b³ + O(θ⁴)`, so
   `ARL → b²` as `theta → 0`. Substituting `h=0.05419, sigma=0.0060` gives
   `b² = 10.198² = 103.99`, confirmed by direct numerical evaluation of the
   general formula at `theta = 1e-6` (`ARL ≈ 103.9917`, converging to `103.992`
   as `theta → 0`).
2. **Evaluating the general formula at `theta = −k/sigma = −0.75` (i.e. at
   `mu = mu_0`, the ACTUAL in-control mean) instead gives `ARL ≈ 3.9×10⁶`,
   not `104`** — because at `mu_0` the random walk has a large negative drift
   (`theta = −0.75`, strongly pulled toward the reflecting boundary at 0), so
   the true in-control ARL under this Gaussian/i.i.d. model is astronomically
   longer than 104 weeks, not shorter. The two calculations answer different
   questions and are not interchangeable.

**Why the delivered "`ARL_0`" constant is `b²` at `mu_0+k`, not the (much
larger) true value at `mu_0`, and why this is the CORRECT convention, not a
sleight of hand.** `k` is chosen small relative to the shift being detected
(Section 3.4: `k = Δ_target·sigma/2`, half the target shift), specifically so
that `mu_0 + k` — the CUSUM's own zero-drift reference point — sits close to
`mu_0` itself. Practitioner references (Montgomery; Hawkins & Olwell 1998,
*Cumulative Sum Charts and Charting for Quality Improvement*, §2.3) report
`ARL(mu_0+k) = b²` AS the "in-control ARL" precisely because `mu_0+k ≈ mu_0`
for the conventional half-shift `k`, and because it is the analytically
tractable zero-drift point, not because `mu_0+k = mu_0` exactly (it does not).
**This is an approximation of convenience whose error grows with `k/sigma`;**
REGIME does not lean on it for anything beyond an initialization value (next
paragraph), which is why Section 3.5 Step 5 mandates the empirical
block-bootstrap ARL_0 as the number that actually ships (Invariant ARL) — the
closed form's role is strictly to seed a starting `h`, not to certify the
final false-alarm rate.

This is Siegmund's (1985) continuity-corrected Wald approximation
(the standard closed-form used in Montgomery's *Introduction to Statistical
Process Control* for CUSUM ARL, cited here as the secondary/practitioner
reference per the pillar brief). It is used ONLY to **initialize** the
`(k, h)` search (Section 3.4); the FINAL calibrated `h` for the live detector
is validated against the **empirical**, block-bootstrapped ARL on REGIME's
own weekly `D_t`/`V_t` series (Section 3.5), because `D_t`/`V_t` are neither
i.i.d. nor exactly Gaussian and the closed-form is a first approximation, not
the delivered guarantee.

### 3.3 Two-sided scheme (REGIME needs both directions)

Section 2.2 noted dispersion COLLAPSE (momentum crowding/unwind) is also a
risk signal, distinct from dispersion/vol SPIKE (market stress). REGIME runs
**two one-sided CUSUMs per statistic, run off the SAME underlying series** —
an upper CUSUM `S_t^+` (as in 3.1) and a lower CUSUM for a downward shift,
`S_t^- = max(0, S_{t-1}^- + (mu_0 − k − x_t))` — each with its own `(k, h)`
pair, with the ARL calibration of Section 3.2 applied **per side, using each
side's own marginal distribution of increments** (next paragraph explains why
this is the correct scope of "separately," and where it stops).

**Assumption flagged explicitly (a gap in this spec's earlier draft): `S_t^+`
and `S_t^-` are NOT statistically independent processes, because they are
both deterministic functions of the SAME weekly draw `x_t`.** The textbook
two-sided CUSUM (Page 1954) is usually motivated by two independent
one-sided alternatives (a distinct upward-shift hypothesis and a distinct
downward-shift hypothesis about two conceptually different changepoints);
here, by construction, a single high draw of `x_t` simultaneously pushes
`S_t^+` up AND (weakly) pushes `S_t^-` toward its floor at 0 — the two
accumulators are mechanically ANTI-correlated through their shared input,
not independent. **What this does and does NOT break:** each side's
MARGINAL in-control ARL calibration (Section 3.2, applied to `S_t^+` alone or
`S_t^-` alone against the marginal distribution of `x_t`) is unaffected — the
Wald/Siegmund approximation only uses the marginal mean/variance of the
increment feeding that one random walk, which is correctly specified whether
or not the other side's accumulator is independent. What IS affected is any
claim about the JOINT false-alarm rate of "`S_t^+` and `S_t^-` alarm
together" (used by `HALT`'s two-or-more-statistics rule, Section 4.1): because
`S_t^+` and `S_t^-` cannot both be elevated from the SAME statistic's noise
(one accumulates positive excursions, the other negative), a simultaneous
`S_D^+`/`S_D^-` alarm on `D_t` ALONE is a near-impossible event under the
in-control model (reassuring — it makes that particular joint alarm an even
stronger signal than a naive independence calculation would suggest), whereas
a simultaneous alarm across DIFFERENT statistics (`S_D^+` and `S_V^+`,
dispersion and BTC-vol) has no such shared-input anti-correlation and is the
combination `HALT` (Section 4.1) is actually designed around. This is why
Section 4.1's `HALT` trigger is stated over `{S_D^+, S_D^-, S_V^+}` as a set —
the practically meaningful two-or-more co-occurrence is cross-statistic
(`D` and `V`), not `S_D^+` with `S_D^-`. This is the standard two-sided CUSUM
construction in its PER-SIDE marginal-calibration sense (Page 1954 treats the
two-sided case as two one-sided schemes run in parallel — a single signed
CUSUM cannot simultaneously have low in-control ARL for both directions with
one threshold), with the joint-alarm caveat above made explicit rather than
silently assumed. REGIME therefore tracks four running CUSUMs live: `S_D^+,
S_D^-` (dispersion up/down) and `S_V^+, S_V^-` (BTC-vol up/down; a "down"
BTC-vol shift is not a stress signal per se and is retained mainly for
symmetry/diagnostics, not as a `g_t`-affecting trigger — Section 4 only gates
on `S_D^+`, `S_D^-`, `S_V^+`).

### 3.4 Parameter governance for `(mu_0, k, h)`

- `mu_0`, `sigma` — **TEAL**, fit once per calibration epoch (Section 6) as the
  sample mean/std of `D_t` (resp. `V_t`) over the designated in-control
  training window (the `2023_recovery`/`2024_bull` `CRYPTO_REGIMES` windows —
  chosen because they are the ex-post-labeled "calm-to-normal" stretches, NOT
  `2022_bear`, which is itself a stress regime and would contaminate `mu_0`
  upward). Frozen for live use per P4; re-estimated only at the next discrete
  calibration epoch.
- `k` (reference/slack) — **RED**, chosen per Section 3.2's standard
  half-shift convention: `k = Δ_target·sigma/2` where `Δ_target` (RED, a BLUE-
  searchable candidate, Section 6) is the standardized shift size REGIME
  commits to detecting (e.g. "detect a 1.5-sigma sustained shift").
- `h` (decision interval) — **RED**, solved from the target `ARL_0` (RED, e.g.
  ARL_0 ≥ 100 weeks ⇒ roughly a false alarm at most once per ~2 years in a
  genuinely healthy regime, stated explicitly per P4/P3) via Section 3.2's
  closed form, then verified empirically (Section 3.5).
- `ARL_0_target` — **RED**, the single governing constant translating directly
  to the "stated low annual false-alarm probability" P4/P3 require: annual
  false-alarm probability `≈ 52.18/ARL_0_target` for `ARL_0_target` in weeks
  (a crude but standard renewal-process approximation, valid when
  `ARL_0_target` is not tiny relative to a year).

> **Invariant C-cfg (inherited).** RED enters only via `resolve_config`. `k`,
> `h`, `ARL_0_target`, `Δ_target` never appear as per-instrument fields — REGIME
> has no per-instrument axis at all (P2: it is a single book-level scalar).

### 3.5 Worked example — calibrating `h` for `D_t`, reproduce bit-for-bit

> **NOT YET EMPIRICALLY GROUNDED — read before trusting any number below.**
> `mu_0=0.0420` and `sigma=0.0060` are NOT fitted from `src/convexity/panel/panel.py`
> data and are NOT in the EVIDENCE digest at the end of this spec; they are
> invented, self-consistent placeholder numbers whose sole job is to let a
> builder reproduce the ARITHMETIC of Sections 3.5/4.3 bit-for-bit while
> implementing. `Δ_target=1.5` and `ARL_0_target=104` ARE real RED governance
> choices (Section 3.4, Section 6's BLUE grid) but are not yet validated
> against the live `D_t` series either. **The one EVIDENCE-grounded numbers
> this worked example does NOT need to invent** are the momentum book's own
> risk profile it is gating — max drawdown ≈25% at 100% gross (≈28%
> annualized vol), weekly win rate ≈60% (Evidence digest, end of file) — which
> is why `HEALTHY`'s `g_t=1.0` (Section 4.1) is calibrated to tolerate exactly
> that unconditional risk, not the invented `(mu_0, sigma)` pair. **A real
> calibration MUST refit `mu_0`, `sigma` from the frozen in-control training
> window (Section 3.4) before any `(k, h)` pair from this worked example is
> promoted to live** — Section 7.1's threshold-stability check and Section
> 3.5 Step 5's empirical-ARL check both gate on the REFITTED values, never on
> these placeholders.

**Inputs** (illustrative calibration-epoch numbers, per the callout above; the
ARITHMETIC below is exact given these inputs, so a builder can assert against
it, but the INPUTS themselves are not yet real):

| Input | Value |
|---|---|
| `mu_0` (mean weekly `D_t` in-control) — **illustrative, not fitted** | `0.0420` (4.20% cross-sectional daily-return std, RMS-weekly) |
| `sigma` (std of weekly `D_t` in-control) — **illustrative, not fitted** | `0.0060` |
| `Δ_target` (standardized shift to detect) — real RED candidate | `1.5` (detect a sustained 1.5-sigma dispersion rise) |
| `ARL_0_target` — real RED candidate | `104` weeks (≈2 years between false alarms in a healthy regime) |

**Step 1 — reference value `k`:**
```
k = Δ_target · sigma / 2 = 1.5 · 0.0060 / 2 = 0.0045
```

**Step 2 — solve `h` from the in-control ARL closed form (Section 3.2).**
Using the `theta → 0` limiting case, `ARL(mu_0+k) ≈ b² = (h/sigma + 1.166)²`
(the conventional in-control approximation, valid because `mu_0+k` is close
to `mu_0` for the half-shift `k` chosen in Step 1 — Section 3.2's derivation,
NOT a substitution of `theta = −k/sigma` into the general formula, which would
give a materially different, wrong number, ≈3.9×10⁶ — see Section 3.2):
```
ARL_0 ≈ (h/sigma + 1.166)²
104 ≈ (h/0.0060 + 1.166)²
sqrt(104) ≈ h/0.0060 + 1.166
10.198     ≈ h/0.0060 + 1.166
h/0.0060   ≈ 9.032
h          ≈ 0.05419
```

**Step 3 — the resulting standardized decision interval** `h/sigma ≈ 9.032`,
which is in the typical CUSUM operating range (`h/sigma` of 4–10 for `k/sigma`
of 0.5–1.0 — cf. Montgomery's tabulated ARL curves), consistent with the
closed-form approximation being reasonably applicable here (the approximation
degrades for very small `h/sigma`, which is not the regime we are in).

**Step 4 — implied out-of-control ARL at the target shift** (`mu = mu_0 +
2k = 0.0510`, so `theta = (mu − mu_0)/sigma − k/sigma = (0.0510−0.0420)/0.0060
− 0.75 = 1.500 − 0.750 = 0.75`; note this `theta=0.75` is the SAME magnitude
as, but opposite sign from, the in-control point's `theta=−0.75` at `mu_0` —
consistent with `mu_0+2k` and `mu_0` being symmetric about the zero-drift
reference `mu_0+k` used in Step 2):
```
ARL_1 ≈ ( exp(−2·0.75·(9.032+1.166)) − 1 + 2·0.75·(9.032+1.166) ) / (2·0.75²)
      = ( exp(−15.297) − 1 + 15.297 ) / 1.125
      ≈ ( 0 − 1 + 15.297 ) / 1.125
      ≈ 14.297 / 1.125
      ≈ 12.71 weeks
```
So the calibrated detector waits, on average, ~104 weeks for a false alarm in
a healthy regime, but detects a genuine sustained 1.5-sigma dispersion rise in
~12.7 weeks — a >8x speed-up, which is the point of CUSUM over a static
threshold (a static threshold has no such asymmetry; it either fires
immediately on noise or never fires on a genuine slow drift, whichever the
single level happens to favor).

**Step 5 — mandatory empirical check (not skippable).** Because `D_t` is
autocorrelated (dispersion clusters) and not exactly Gaussian, Step 2-4's
closed form is a starting point, not the delivered number. The FINAL `h`
used live is the value (searched on a small RED/BLUE grid around the
closed-form solution, Section 6) whose **empirical** ARL_0 — measured by a
moving-block bootstrap (Künsch 1989, block length per Politis & White 2004's
automatic block-length selection, applied to the frozen in-control training
window) simulating many resampled `D_t` paths and averaging the observed
run-length to first crossing — matches the target `ARL_0_target` within a
stated tolerance. This is analogous to COST's `e_q` block-bootstrap
discipline (`COST.md` §5.5c) applied here to a run-length rather than a
quantile.

> **Invariant ARL — the delivered ARL_0 is empirically verified, not just
> closed-form asserted.** No calibrated `(k, h)` ships without a
> block-bootstrap empirical ARL_0 estimate on the frozen training window
> landing within the RED tolerance band of `ARL_0_target`; a closed-form-only
> calibration with no empirical check is the bug. Test: perturbing the
> in-control window's autocorrelation (a synthetic AR(1)-injected variant of
> the training data) changes the empirical ARL_0 estimate but the reported
> number always reflects the EMPIRICAL, not the closed-form, value.

---

## 4. From CUSUM state to `g_t` — the state machine

### 4.1 States and bands

Four states, ordered by severity, each mapping to a **band** for `g_t`, not a
single hardwired point value (an explicit correction of the old draft's
`g_t∈{1,.75,.5,.25}` fixed ladder, which was tuned for a single-instrument
book's discrete leverage steps — the cross-sectional book's exposure is
continuous, so REGIME emits the band's FLOOR as the conservative live value,
per P6, while recording the band for research/BLUE-grid purposes):

| State | Trigger | `g_t` |
|---|---|---|
| `HEALTHY` | no CUSUM has alarmed since last reset | `1.00` |
| `WATCH` | `S_D^+` OR `S_D^-` has crossed a WARN sub-threshold `h_warn < h` (RED, e.g. `h_warn = 0.6h`) but not yet `h`; OR data is stale/insufficient (P6) | `0.75` |
| `STRESSED` | `S_D^+`, `S_D^-`, or `S_V^+` has crossed its full `h` (a CUSUM alarm has fired and not yet reset by the re-arm rule, Section 4.2) | `0.40` |
| `HALT` | TWO OR MORE of `{S_D^+, S_D^-, S_V^+}` are simultaneously in alarm, OR the drawdown-stop of Section 5 has independently triggered | `0.00` |

`g_t` is **floored at the state's value, never interpolated below it and never
exceeding it while the state holds** — this mirrors LEVERAGE's governor
convention (`LEVERAGE.md`, referenced via `docs/ARCHITECTURE.md` "g_t∈{1,.75,
.5,.25} floored .25" for the OLD system) with the update that `HALT` here is
allowed to floor at exactly `0` (full flat), unlike the old draft's `.25`
floor, because a `HALT`-triggering **simultaneous multi-statistic alarm** is
evidence of a joint stress event the weekly-rebalanced book cannot safely
hold any exposure through until the next scheduled recompose.

### 4.2 Re-arm (recovery) rule

A `STRESSED` or `HALT` state does not clear the instant the triggering CUSUM
dips below `h`. Mirroring `overlay.py`'s `drawdown_stop_scale` re-arm
discipline (return to `armed=True` only once equity recovers to within
`dd_reset` of the peak, never immediately on one good week), REGIME re-arms
one severity level at a time:

```
STRESSED -> WATCH   : requires the alarmed CUSUM to reset to S_t = 0 (Section 3.1's
                       own reset rule) AND stay at S_t < h_warn for >= n_confirm
                       consecutive weeks (RED, e.g. n_confirm = 2)
WATCH    -> HEALTHY : requires S_t < h_warn for >= n_confirm consecutive weeks
                       with no re-alarm
```

This is a **causal, deterministic hysteresis**, not a fitted transition
matrix (the Hamilton-1989 alternative this spec rejects, P3) — `n_confirm` is
a fixed RED integer, not a probability re-estimated from data.

### 4.3 Worked example continuing Section 3.5

Suppose, at week `t`, `S_D^+` has just crossed `h = 0.05419` (a dispersion
spike alarm) while `S_D^-` and `S_V^+` remain at 0. Only one statistic is in
alarm, so by Section 4.1's table the state is `STRESSED`, `g_t = 0.40`,
reason chain `{"detector": "dispersion_cusum_upper", "S_t": 0.0544,
"h": 0.05419, "state": "STRESSED"}`. If in the following week `S_V^+` ALSO
crosses its own `h_V`, two statistics are now simultaneously in alarm, the
state escalates to `HALT`, `g_t = 0.00`, reason chain lists both triggering
detectors. This is the concrete instance of Section 0's `g_t ∈ [0,1]` scalar
and P7's mandatory reason chain.

---

## 5. The drawdown-stop — first-passage framing, and the honest carry-vs-momentum asymmetry

### 5.1 Formal framing (Grossman & Zhou 1993)

A drawdown-constrained trading rule is formally a portfolio-choice problem
under the constraint that wealth `W_t` never falls below a fraction
`(1 − D_max)` of its running maximum `M_t = sup_{s≤t} W_s`. Grossman & Zhou
(1993), "Optimal Investment Strategies for Controlling Drawdowns," derive the
optimal dynamic asset allocation under exactly this constraint for a
continuous-time diffusion-return asset, showing the optimal risky-asset
weight is a function of the CURRENT drawdown-to-peak ratio `W_t/M_t`: the
allocation shrinks continuously as `W_t/M_t` falls toward `1 − D_max`, rather
than jumping discretely to zero at a fixed trigger. This is cited here as the
**formal justification for why a drawdown constraint is a legitimate
portfolio-choice object** (not an ad hoc risk-management add-on) and as the
**benchmark REGIME's discrete stop is a coarse approximation of**: a
continuous, wealth-ratio-proportional de-risking is the theoretically optimal
rule; REGIME's Section 4.1 discrete `WATCH/STRESSED/HALT` bands and
`overlay.py`'s discrete `drawdown_stop_scale` are computationally cheap,
closed-form, no-ML STEP approximations to that continuous optimal policy,
which is the right trade-off for a 2-core VM and a weekly (not continuous)
decision clock, but the approximation error is explicitly acknowledged rather
than asserted away.

### 5.2 First-passage-time framing of "when does a stop even bind"

Let `W_t` be the book's cumulative equity (product of `1 + net_weekly returns`,
`src/convexity/system/metrics.py` `compute()`'s `equity` construction) and `M_t` its
running peak. The drawdown-stop of `overlay.py` fires at the **first-passage
time** `T_trigger = inf{t : (M_t − W_t)/M_t ≥ dd_trigger}`. Whether a stop
triggered at `T_trigger` HELPS or HURTS subsequent Sharpe depends entirely on
the process's behavior AFTER `T_trigger`:

```
E[ realized-P&L WITH stop ]  >  E[ realized-P&L WITHOUT stop ]
   <=>   E[ cumulative return over (T_trigger, T_recover) | drawdown continues past dd_trigger ]  <  0
```

i.e. the stop is a good bet exactly when the CONDITIONAL expected return
during the period the stop would keep the book flat is negative — a
**momentum/persistence** condition on the drawdown process itself. This is
the formal statement of "isolated tail event vs. persistent regime bleed":

- **Isolated tail event** — the drawdown is driven by a short-lived shock
  (e.g. a single violent, mean-reverting week) whose conditional forward
  return, once the shock has passed, is roughly the strategy's unconditional
  positive expectation. A stop that flattens through the shock and re-arms
  promptly captures nearly all of the avoided loss and gives back little of
  the recovery. **The stop helps.**
- **Persistent regime bleed** — the drawdown is driven by a sustained,
  slow-moving structural shift (funding compresses to near-zero for many
  months; momentum's cross-sectional dispersion collapses for many months)
  whose conditional forward return, CONTINGENT on remaining in the flattened
  state, includes most of the eventual recovery. Because `overlay.py`'s
  `drawdown_stop_scale` only re-arms once equity has recovered to within
  `dd_reset` of the OLD peak — but a persistent bleed means the book earns
  ~0 while flat, so equity never approaches that peak, so the stop STAYS
  OFF through the entire eventual recovery, converting a drawdown into a
  permanent flatline. **The stop hurts, structurally, not by bad luck.**

### 5.3 The tested empirical instance — carry, exactly as specified

Our own tested result on the funding-carry book (Section 0 P5, restated with
its numbers): the raw carry book earns **+12.9%/yr** funding income at a
**~70-75% maxDD**; applying a causal `drawdown_stop_scale` caps maxDD at **~24%**
but the strategy's Sharpe **collapses from 0.57 (with the beta-hedge overlay,
Section 5.5) to 0.10**, and funding income collapses to **~0.6%/yr**. By
Section 5.2's condition, this happened because the carry drawdown IS the
persistent-bleed case: the 2023-24 low-funding-rate stretch is a multi-quarter
structural regime (funding rates compressed market-wide, not a single
name/week shock), so the conditional forward return during the flattened
period was **not** negative — it was close to zero-to-mildly-positive once
funding normalized, meaning the stop paid the full cost of missing the
eventual recovery for a drawdown it could not usefully have avoided (the
bleed happens whether or not the stop is engaged; the stop only prevents
PARTICIPATING once the bleed ends). This is why P5 states the finding as
**"a naive stop-and-flatten rule can destroy more edge than it protects"** —
it is not a coding bug or a badly-chosen `dd_trigger`, it is Section 5.2's
formal condition being violated by carry's actual drawdown process.

### 5.4 The required validation before a DD-stop is promoted for MOMENTUM

Section 5.3's finding is **carry-specific and must NOT be assumed to transfer
to momentum** (explicit instruction in the pillar brief). Momentum's
drawdown process differs structurally from carry's: momentum's known bad
regime is a sharp, sudden **reversal/crash week** (a v-shaped or short-lived
cross-sectional reversal), which is closer to Section 5.2's isolated-tail-event
case than carry's slow funding-regime bleed. But "closer to" is not
"confirmed," so REGIME requires the following **regime-conditional ablation**
(reusing `src/convexity/system/cpcv.py`'s CPCV machinery exactly as-is, no new
validation code) before any DD-stop parameter is promoted to live for the
momentum book:

```
1. Compute the raw (no-stop) momentum weekly net-return series (src/convexity/portfolio/engine.py).
2. Apply drawdown_stop_scale at each RED-candidate (dd_trigger, dd_reset) pair (BLUE grid, Section 6).
3. For EACH candidate, compute per-REGIME-WINDOW (using the EXISTING src/convexity/panel/sleeve.py
   CRYPTO_REGIMES labels: 2021_bull, 2022_bear, 2023_recovery, 2024_bull, 2025_2026) the
   Sharpe/CAGR/maxDD delta vs the no-stop series, exactly as overlay.py's docstring compares
   carry's overlaid vs raw series.
4. PROMOTE a candidate only if:
     (a) frac_pos of CPCV OOS paths with stop-Sharpe >= no-stop-Sharpe is itself the majority
         (mirrors the digest's stated preference for frac_pos over a fragile single percentile,
         applied here to the STOP-VS-NO-STOP comparison rather than the raw strategy Sharpe), AND
     (b) no single CRYPTO_REGIMES window shows the carry-like signature of Section 5.2's bleed
         condition (a window where the STOPPED series' total window return is materially BELOW
         the RAW series' window return by more than a RED tolerance, indicating the stop sat out
         a genuine recovery rather than avoiding a genuine loss).
   A candidate failing (b) in even one historical window is REJECTED regardless of its aggregate
   Sharpe improvement — a single bleed-like window is the carry lesson recurring, and aggregate
   Sharpe can hide it (an aggregate improvement can be entirely one lucky-avoided crash offsetting
   one costly missed recovery).
5. If NO candidate passes both gates, REGIME ships WITHOUT a promoted DD-stop for momentum (the
   Section 4 CUSUM states remain the only defensive gate) — this is an acceptable, EXPECTED outcome,
   not a spec failure; a DD-stop is a nice-to-have overlay, not a hard requirement (unlike Section 3's
   CUSUM detector, which IS required).
```

**Edge case: the ablation's own return series is shorter than the CPCV
split family it must reuse.** Step 4(a)'s `frac_pos` reuses `research/xsec/
cpcv.py`'s `N=8` groups / `k=2` test groups / 28 paths / 1-week-embargo
configuration verbatim (Section 7, "REGIME does not reimplement CPCV"),
which the EVIDENCE digest validated on the **full ~259-week crypto history**.
If momentum's LIVE track record at ablation time is materially shorter (e.g.
a recently-live sleeve with 60-100 weeks, not 259), splitting it into `N=8`
groups still runs mechanically, but each of the 28 paths' train/test blocks
is shorter, so `frac_pos` itself is estimated with materially less power —
the same partition-fragility problem Section 5.4(a) invokes `frac_pos` to
AVOID (relative to `p05`) is not eliminated, only pushed one level up (into
`frac_pos`'s own sampling noise at small `n_weeks`). **Rule:** if
`n_weeks_live < n_weeks_min_cpcv` (RED, e.g. `156` weeks / 3 years — chosen so
each of the 28 groups' held-out blocks still spans multiple regime-relevant
weeks), the ablation of Step 4 is **run and reported for research visibility
only**; its `frac_pos` result CANNOT by itself satisfy gate (a) — a DD-stop
may not be promoted to live on a short-history `frac_pos` alone. This does
not relax gate (b) (the regime-conditional bleed check), which is a
per-window sign check, not a small-sample statistic, and remains fully
binding regardless of `n_weeks_live`.

> **Invariant STOP-XFER — no DD-stop calibration transfers across strategies
> without its own regime-conditional ablation.** A `(dd_trigger, dd_reset)`
> pair validated on the carry book's return series (or any other sleeve) may
> NOT be applied to the momentum book's live config without independently
> passing Section 5.4's four-step gate on momentum's OWN return series, run at
> full statistical power (`n_weeks_live ≥ n_weeks_min_cpcv`) — a short-history
> ablation may inform research but does not by itself satisfy gate (a) (prior
> paragraph). Test: the promotion pipeline raises/rejects if a candidate's
> ablation record is keyed to a different strategy's return series than the
> one being deployed, OR if a promotion claims gate (a) is satisfied from an
> ablation run with `n_weeks_live < n_weeks_min_cpcv`.

### 5.5 Beta-hedge — honest, modest, quantified

`overlay.py`'s `beta_hedge_returns` neutralizes the book's residual BTC-beta
each week via a trailing OLS beta (Section 0 P3's closed-form-only
constraint — `trailing_beta` in `overlay.py` is exactly the sample
covariance/variance ratio estimator, no shrinkage currently applied; a
Ledoit-Wolf-shrunk multivariate beta is a valid BLUE-grid research extension
per Section 2.4's general shrinkage discipline, but the univariate
single-factor OLS beta here is a `1×1` "covariance matrix" and is not
subject to the `N≈T` ill-conditioning Section 2.4 warns about, so shrinkage is
not required for THIS estimator). On carry, this overlay is **tested and
quantified, not asserted**: it lifted Sharpe from **0.48 to 0.57** while
dropping maxDD from **74.9% to 69.9%** — cite this exactly, as a **real but
modest** gain (a ~19% relative Sharpe improvement, a ~7% relative maxDD
improvement), not a regime-fixing intervention. REGIME's role here is limited
to GATING whether the beta-hedge overlay is even engaged (it is a Section 4
`g_t`-independent, always-on overlay for carry-like sleeves per its own
promotion record) — REGIME does not re-derive the beta-hedge math, only
reports its tested honest numbers when a caller asks whether it is
appropriate to compose with the momentum book (Section 5.4's same
regime-conditional ablation applies before beta-hedging is promoted for
momentum, for the same non-transfer reason as the DD-stop).

---

## 6. Parameter governance

Every REGIME parameter is exactly one bucket.

- **TEAL — fitted from REGIME's own past data, at calibration time, CAUSAL.**
  `mu_0`, `sigma` for `D_t` and `V_t` (Section 3.4), fit on the frozen
  in-control training window and refrozen only at the next discrete
  calibration epoch (never online).
- **AMBER — supplied metadata.** The BTC symbol used for `V_t`
  (`market_symbol="BTCUSDT"`, matching `overlay.py`'s default), the
  `CRYPTO_REGIMES` window boundaries (already AMBER-equivalent constants in
  `src/convexity/panel/sleeve.py`, reused not reforked), the weekly rebalance
  clock (Friday close, matching the xsec engine).
- **RED — shared economic/modelling constants.** `Δ_target`, `ARL_0_target`,
  the resulting `(k, h)` per statistic, `h_warn` (the WATCH sub-threshold
  fraction of `h`), `n_confirm` (re-arm hysteresis weeks), `rho_cl`-style
  ablation tolerances for Section 5.4's bleed-detection tolerance,
  `n_weeks_min_cpcv` (Section 5.4's minimum live history for a DD-stop
  ablation's `frac_pos` to count toward promotion, e.g. `156` weeks), and an
  overnight compute cap bounding how many `(Δ_target, ARL_0_target,
  dd_trigger, dd_reset)` combinations may be searched in one calibration
  epoch — named `K_regime_max` in `docs/ARCHITECTURE.md` §3's prose
  ("caps like REGIME `K_regime_max`") and in `core/params.py`'s module
  docstring. **Verified status: `K_regime_max` is named in TWO places in
  prose/docstrings but has NO corresponding field, constant, or `ParamSpec`
  entry anywhere in the repo today** (confirmed by grep across
  `src/convexity/`) — unlike `resolve_config` (a real, already-implemented
  pattern: `src/convexity/system/config/resolve.py` and
  `src/convexity/cost/config/resolve.py` both exist and this spec expects
  REGIME to add its own `regime/config/resolve.py` following the identical
  pattern) and unlike `core/types.RegimeState`/`core/protocols.RegimeEngine`
  (real, if wrong-shaped, code — previous paragraph), `K_regime_max` is
  purely an architecture-doc name with no implementation anywhere yet. This
  spec is the first place its VALUE and role are actually specified; treat
  it as newly defined here, not as an existing constant being referenced.
- **BLUE — finite validated grids, class-level only.** Small enumerated
  candidate sets for `Δ_target` (e.g. `{1.0, 1.5, 2.0}` sigma), `ARL_0_target`
  (e.g. `{52, 104, 208}` weeks), and the DD-stop `(dd_trigger, dd_reset)` pairs
  of Section 5.4 — each promoted only under Section 7's validation gate, never
  fitted continuously and never per-instrument (REGIME has no per-instrument
  axis to tune against, P2).

> **Invariant C-cfg (inherited).** RED enters only via `resolve_config`. No
> `Δ_target`, `ARL_0_target`, `k`, `h`, `n_confirm`, or DD-stop parameter
> appears in AMBER metadata; per-symbol tuning of any RED constant on realized
> P&L is forbidden (there is no per-symbol axis for REGIME to tune against in
> the first place — this invariant exists mainly to block a future
> misguided attempt to add one).

---

## 7. Validation — threshold stability, ablation, and the promotion gate

REGIME's own directory tree already anticipates this
(`docs/ARCHITECTURE.md` §2: `regime/validate/{threshold_stability,baselines,
ablation,backtest}.py`). The validation reuses SYSTEM's machinery exactly as
`src/convexity/system/cpcv.py`/`validate.py` already do for the momentum family —
REGIME does not reimplement CPCV, DSR, or block-bootstrap CIs.

### 7.1 Threshold stability (`threshold_stability.py`)

The ARL calibration of Section 3.5 must be shown STABLE across the
`CRYPTO_REGIMES` windows used for training vs held-out windows: refit
`(mu_0, sigma)` on each of `{2023_recovery, 2024_bull}` independently and
confirm the resulting `h` values (Section 3.2's closed form) do not differ by
more than a RED tolerance — an `h` that swings wildly between two nominally
"calm" windows indicates `D_t`/`V_t` themselves are regime-dependent in a way
that undermines a single frozen `(mu_0, sigma, h)` triple, and would be
grounds to widen the training window or add a slow, discretely-recalibrated
"epoch" axis (never an online one, P4).

### 7.2 Baselines (`baselines.py`)

Mandatory comparison baselines, mirroring `COST.md`/`SYSTEM.md`'s baseline
discipline:
- **`always_healthy`** — `g_t ≡ 1.0` always (the no-detector null).
- **`always_half`** — `g_t ≡ 0.5` always (a naive constant de-risk, to show
  the detector earns its keep vs blind de-leverage).
- **`realized-vol-only`** — a naive fixed-threshold rule on `V_t` alone (no
  CUSUM, no dispersion), to isolate CUSUM's sequential-detection value-add
  from a single-statistic static threshold.
- **`perfect-foresight`** — `g_t` set with full hindsight of the actual
  forward drawdown (an upper bound, never achievable live, reported only as a
  ceiling).

### 7.3 Ablation (`ablation.py`)

Turn off each REGIME component independently (dispersion CUSUM only,
BTC-vol CUSUM only, DD-stop only, beta-hedge only) and report the attributed
Sharpe/maxDD delta of each, exactly mirroring SYSTEM §8's Shapley-style
per-pillar attribution, applied here to REGIME's own sub-components.

### 7.4 Backtest / promotion gate (`backtest.py`)

The full REGIME-gated momentum book's weekly net-return series is fed to
`src/convexity/system/validate/deflated.py`'s `DeflatedSharpe` exactly as any
other candidate config, entering the SAME persistent trial ledger SYSTEM's
Invariant SG requires (`SYSTEM.md` §5, "Invariant SG"): every `(Δ_target,
ARL_0_target, dd_trigger, dd_reset)` combination REGIME's BLUE grid searches
is one more ledger config, and `K_regime_max` (Section 6) bounds this
contribution to the overnight compute budget SYSTEM's own `K_max` governs
(`SYSTEM.md` §5 RED list, "the overnight compute cap (`K_max`, ...)"). REGIME
does NOT run its own separate, uncounted DSR gate — it contributes trials to
the one SYSTEM ledger, per SYSTEM's own Invariant SG ("every composed config
... across ALL overnight runs ... appears in the persistent trial ledger").

> **Invariant LEDGER — REGIME's BLUE grid feeds the ONE system trial ledger,
> never a private one.** A REGIME calibration promoted without its
> `config_hash` appearing in SYSTEM's persistent ledger (Invariant SG) is the
> bug SYSTEM's four-green-modules trap exists to catch, applied to REGIME
> specifically. Test (two halves, BOTH binding in the detector scope): (i) the
> CARDINALITY half — REGIME's BLUE-grid product (`(Δ_target, ARL_0_target)` =
> `3×3 = 9` detector tuples) never exceeds `K_regime_max`, the overnight compute
> cap (`tests/regime/test_config.py::test_k_regime_max_bounds_the_blue_grid_cardinality_fed_to_the_system_ledger`);
> (ii) the EQUALITY half — the number of distinct `(Δ_target, ARL_0_target[,
> dd_trigger, dd_reset])` tuples `regime/validate/backtest.py` scores equals the
> count of REGIME-tagged entries it appends to the ONE shared
> `system/validate/trials.TrialLedger` (never a private one), and a tuple scored
> without a matching ledger row breaks the equality
> (`tests/regime/test_ledger.py`). `backtest.py`'s per-tuple net-return
> BACKTEST is injected (`score_fn`) and remains the scoped-out reporting layer
> (§9); the LEDGER BOOKKEEPING equality is now binding in the detector scope.

---

## 8. Edge cases and invariants (consolidated)

1. **Insufficient history at the start of the panel (early-history sparse
   universe).** `universe_size_by_year` shows the early crypto universe is
   much smaller than 50-75 names. `D_t` computed over too few names is a
   noisy estimate of cross-sectional dispersion. **Rule:** if
   `|U_τ| < n_min_dispersion` for a majority of days `τ` in week `t` (RED,
   e.g. `n_min_dispersion = 15` — chosen well below the ~50-75 deployable
   range so the floor binds only in genuinely early history), REGIME reports
   `WATCH` (never `HEALTHY`, per P6) with reason `insufficient_breadth`, and
   `D_t` is excluded from that week's CUSUM update entirely (not fed as a
   degenerate/zero value, which would artificially suppress `S_D^+`/`S_D^-`
   and could mask a genuine early regime).

2. **Dispersion COLLAPSE, not just spike (the momentum-crowding edge case,
   Section 2.2).** A sustained DOWNWARD shift in `D_t` (crowding/unwind, low
   idiosyncratic spread) is caught by `S_D^-`'s independent CUSUM (Section
   3.3) — this is precisely why REGIME runs the two-sided scheme rather than
   only watching for a spike; a spike-only detector would completely miss
   this documented risk to a cross-sectional strategy.

3. **Simultaneous alarms across statistics vs. one persistent alarm.** Two
   statistics alarming AT THE SAME TIME (`HALT`, Section 4.1) is treated more
   severely than one statistic staying in prolonged alarm (`STRESSED`
   indefinitely, re-arm gated by Section 4.2's hysteresis, never
   auto-escalating to `HALT` on duration alone) — REGIME does NOT
   escalate severity purely as a function of how long a single alarm has
   persisted, because Section 5's persistent-bleed lesson says a long,
   ISOLATED alarm on one axis (e.g. dispersion staying moderately elevated
   for months in a genuine new-normal regime) is not necessarily worse than a
   short one; only the CO-OCCURRENCE of independent stress signals is used as
   the stronger evidence of joint, book-threatening stress.

4. **A calibration epoch boundary falling inside a live drawdown.**
   Recalibrating `(mu_0, sigma)` (Section 3.4) is a discrete, versioned event
   (Section 6, Invariant C-cfg) — it MUST NOT be triggered as an ad hoc
   reaction to an ongoing alarm (that would be P4's forbidden online
   re-estimation in disguise: refitting `mu_0` upward mid-stress to make an
   alarm clear is exactly the quiet-shrink hazard SYSTEM's Invariant DSR-MONO
   warns against in a different context). Recalibration only occurs at
   scheduled epoch boundaries defined and versioned ahead of time.

5. **`g_t` interacting with the xsec engine's own gross parameter.** Because
   `portfolio.quantile_weights`/`liquidity_weights` take `gross` as an
   explicit argument (Section 1), `g_t` composes as
   `gross_live = g_t · gross_target` at the SAME call site that already sets
   `gross` — REGIME never reaches into the portfolio construction internals,
   it only multiplies the scalar passed in, preserving Invariant N.

6. **Universe delisting during a `HALT`.** If a symbol the book holds
   delists while `g_t = 0` (`HALT`), REGIME's own detector is unaffected (it
   operates on the eligible cross-section, not on individual open positions);
   the position-level unwind is the engine/accounting layer's concern
   (`src/convexity/cost/accounting.py`), not REGIME's — REGIME's firewall (P2)
   means it never reaches into position management.

---

## 9. Directory layout (as already anticipated in `docs/ARCHITECTURE.md`)

```text
regime/
+-- config/
|   +-- defaults.py          # RED: Delta_target, ARL_0_target, k, h, h_warn, n_confirm,
|   |                        #   n_weeks_min_cpcv, K_regime_max (net-new here; see Section 6)
|   +-- asset_class.py       # AMBER: per-sleeve market_symbol, regime windows (reuse universe.py tables)
|   +-- resolve.py           # single merge point (RED unreachable from AMBER)
+-- panel/
|   +-- build.py             # thin adapter over research/xsec/panel.Panel (Invariant N)
+-- measures/
|   +-- vol_regime.py        # V_t: BTC realized vol (Section 2.3)
|   +-- absorption.py        # cross-sectional dispersion D_t (Section 2.2) [kept filename from
|   |                        #   the pre-pivot tree per docs/ARCHITECTURE.md; content is the NEW D_t,
|   |                        #   not the old per-instrument absorption ratio]
|   +-- turbulence.py        # optional Ledoit-Wolf + Mahalanobis fallback (Section 2.4), BLUE-only
|   +-- liquidity.py         # optional dollar-volume-based breadth diagnostic (reuses
|                             #   research/xsec/panel.build_dollar_volume)
+-- state/
|   +-- bands.py             # Section 4.1 state -> g_t band table
|   +-- machine.py           # Section 4.2 re-arm hysteresis state machine
|   +-- cost_gate.py         # optional: feed g_t/state into COST's round-trip estimate for
|   |                        #   a regime-aware conservative cost telemetry line (diagnostic only,
|   |                        #   never a second decision channel — P2)
|   +-- emit.py              # constructs the frozen RegimeState record (P7 reason chain)
+-- validate/
|   +-- threshold_stability.py   # Section 7.1
|   +-- baselines.py             # Section 7.2
|   +-- ablation.py              # Section 7.3
|   +-- backtest.py              # Section 7.4 (feeds SYSTEM's DeflatedSharpe ledger)
+-- synth/
|   +-- generate.py          # adversarial dispersion spikes/collapses, vol regime shifts,
|                             #   persistent-bleed synthetic drawdowns (to test Invariant STOP-XFER)
+-- types.py                 # RegimeState and the four-CUSUM internal state, frozen dataclasses
+-- engine_api.py             # public entry point: RegimeEngine.regime_state(panel, t) -> RegimeState
```

`measures/vol_regime.py`, `measures/absorption.py` (repurposed name, new
content — see the bracket note above), `state/{bands,machine,emit}.py`, and
`validate/{threshold_stability,baselines,ablation,backtest}.py` are new files
against the tree `docs/ARCHITECTURE.md` §2 already reserves for this pillar;
nothing under this tree currently exists as code (the pillar brief: "DOES NOT
EXIST YET"), so this is the from-scratch build target, not a migration.

**Detector-scope status of the `validate/`+`synth/` subtrees (what is built vs.
deferred).** The DETECTOR is 10/10 for its declared scope: the two named
invariants that reference these subtrees — **MP** (§2.4) and **LEDGER** (§7.4)
— each now have a BINDING test in the detector scope, backed by the minimal
seeds of these subtrees:

- `synth/generate.py::iid_no_structure_panel` — the single `N×T` i.i.d.
  no-structure generator Invariant MP's empirical demonstration needs (the full
  adversarial suite — dispersion spikes/collapses, vol-regime shifts,
  persistent-bleed drawdowns for STOP-XFER, §7.2 baselines — is the explicit
  FUTURE build-contract: each is a labelled generator added to this same file,
  none required for the detector's own correctness).
- `measures/turbulence.py` — the closed-form Ledoit-Wolf (2004) shrunk
  covariance + Kritzman-Li absorption ratio, the "accepted fallback
  construction" §2.4 reserves this file for. It is NOT a v1 live detector path
  (Invariant MP still forbids any unshrunk covariance in a LIVE statistic, and
  `absorption.py`/`vol_regime.py` remain covariance-free); it exists to make
  the MP DEMONSTRATION binding and as the BLUE-only research fallback.
- `validate/backtest.py` — the LEDGER-bookkeeping half of §7.4: it enumerates
  the BLUE-grid tuples and appends one REGIME-tagged `TrialRecord` per scored
  tuple to the ONE shared `system/validate/trials.TrialLedger`, with the
  per-tuple net-return backtest INJECTED (`score_fn`). The equality Invariant
  LEDGER states is now binding; the per-tuple net-return backtest ENGINE, plus
  `validate/{threshold_stability,baselines,ablation}.py`, are the explicit
  FUTURE build-contract (the §7 reporting layer), out of the detector scope.

`measures/vol_regime.py`, `measures/absorption.py` (repurposed name, new
content), `state/{bands,machine,emit}.py`, `config/*`, `cusum.py`,
`engine_api.py`, plus the seeds above are BUILT; the remaining `validate/`
reporting modules and the extended `synth/` adversarial suite are the deferred
build-contract described above, out of the detector's declared scope.

---

## 10. What REGIME is NOT (non-goals)

- **Not a signal.** REGIME never proposes a trade, a side, or an instrument.
  It scales exposure the momentum signal (`src/convexity/signal/signals.py`) already
  decided on.
- **Not a sizing engine.** `g_t` is a single multiplicative scalar on gross,
  not a per-instrument weight adjustment (P2). Within-leg weighting
  (`equal`/`volscaled`/liquidity-aware) is entirely `research/xsec/
  portfolio.py`'s concern.
- **Not a covariance-optimized regime classifier.** Section 2.4 explicitly
  rejects a fitted absorption-ratio/HMM approach as the default, on
  Marchenko-Pastur grounds.
- **Not a per-strategy-transferable stop calibration.** Section 5.4's
  Invariant STOP-XFER is the load-bearing non-goal: REGIME does not assume
  carry's DD-stop numbers, or any other sleeve's, apply to momentum without
  its own ablation.
- **Not the SYSTEM orchestrator.** Once SYSTEM is built, REGIME is one more
  Protocol (`RegimeEngine.regime_state`, per `docs/ARCHITECTURE.md` §3
  `core/protocols.py`) SYSTEM composes; REGIME does not itself arbitrate
  between pillars.

---

## References

- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2),
  100–115. The CUSUM recursion, its SPRT-restarted-at-minimum construction,
  and its `S_t ≥ h` stopping rule (Section 3.1) — REGIME's `S_t` recursion IS
  this construction, not an approximation of it.
- Lorden, G. (1971). "Procedures for Reacting to a Change in Distribution."
  *Annals of Mathematical Statistics*, 42(6), 1897–1908. A LATER, SEPARATE
  result from Page's: proves CUSUM is asymptotically minimax-optimal for
  expected detection delay (worst-case changepoint) among all rules with a
  given false-alarm rate. Cited only for this specific optimality claim
  (Section 3.1's closing paragraph explicitly separates "Page's construction"
  from "Lorden's optimality theorem" — do not attribute the optimality result
  to Page 1954, which does not contain it).
- Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence Intervals*.
  Springer. The Wald-approximation closed-form ARL with the `1.166·sigma`
  continuity correction (Section 3.2).
- Montgomery, D. C. *Introduction to Statistical Process Control* (any recent
  edition). Secondary/practitioner framing of the CUSUM ARL formulas and
  typical `(k/sigma, h/sigma)` operating ranges (Section 3.2, 3.5), cited per
  the pillar brief's acceptable-secondary-cite instruction.
- Hawkins, D. M., & Olwell, D. H. (1998). *Cumulative Sum Charts and Charting
  for Quality Improvement*. Springer. Secondary/practitioner source (alongside
  Montgomery) for the convention of reporting `ARL(mu_0+k) = (h/sigma+1.166)^2`
  as "the in-control ARL" (Section 3.2's corrected derivation of why this is a
  distinct zero-drift limiting case, not a substitution instance of the
  general two-sided formula).
- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*,
  88(2), 365–411. The shrinkage estimator and closed-form optimal intensity
  (Section 2.4) required for any covariance-flavored regime statistic.
- Marchenko, V. A., & Pastur, L. A. (1967). "Distribution of Eigenvalues for
  Some Sets of Random Matrices." *Mathematics of the USSR-Sbornik*, 1(4),
  457–483. The random-matrix eigenvalue-spreading result justifying REGIME's
  refusal to use an unshrunk sample covariance / absorption ratio when
  `N` approaches `T` (Section 2.4).
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). "Noise
  Dressing of Financial Correlation Matrices." *Physical Review Letters*,
  83(7), 1467–1470. Applies Marchenko-Pastur to financial correlation
  matrices; the applied justification cited alongside the original RMT result
  (Section 2.4).
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2),
  357–384. Cited ONLY as the discrete regime-switching reference point this
  system deliberately does NOT implement (Section 0 P3): an EM-fitted HMM's
  continuous re-estimation is the forbidden retraining loop.
- Grossman, S. J., & Zhou, Z. (1993). "Optimal Investment Strategies for
  Controlling Drawdowns." *Mathematical Finance*, 3(3), 241–276. The formal
  drawdown-constrained portfolio-choice result motivating Section 5's
  first-passage/optimal-stopping framing of the DD-stop as a discrete
  approximation to a continuous optimal policy.
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." *Annals of Statistics*, 17(3), 1217–1241. The
  moving-block bootstrap used for the empirical ARL check (Section 3.5) and
  any threshold CI.
- Politis, D. N., & White, H. (2004). "Automatic Block-Length Selection for
  the Dependent Bootstrap." *Econometric Reviews*, 23(1), 53–70. The
  block-length selection rule for the Section 3.5 bootstrap.
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5), 94–107; and López de Prado, M.
  (2018), *Advances in Financial Machine Learning* (CPCV, purging, embargo).
  Reused, not rederived, by REGIME's Section 7.4 promotion gate via
  `src/convexity/system/validate/{deflated,cpcv}.py`.
- Bailey, D. H., & López de Prado, M. (2014). "The probability of backtest
  overfitting" — the logit-of-relative-rank PBO construction reused
  unmodified from `src/convexity/system/cpcv.py`'s existing `compute_pbo`, cited
  here because Section 5.4's ablation gate borrows the same CPCV split
  family for the stop-vs-no-stop comparison.

---

## Evidence digest cited in this spec (repeated here for a single audit trail)

- PBO = 0.31–0.47 across CPCV partitions for the L=14 fitted single-lookback
  pick; best-in-sample was L=12 on 21/28 CPCV splits, never a clean pass below
  0.2 — cited in Section 0 as the reason the momentum signal itself deploys a
  lookback BAND (12–16 days), a fact REGIME's own Section 5.4 ablation gate
  structurally mirrors (never trust one best-in-sample calibration without
  the same OOS-rank discipline).
- Honest net Sharpe ≈0.8–1.0/lookback (real Amihud-impact cost); light-cost
  proxy Sharpe ≈1.6–1.9 is cited ONLY as an optimistic upper bound, never a
  planning number (Section 0 framing carried through this whole spec's
  numeric grounding).
- Max drawdown ≈25% at 100% gross (≈28% annualized vol); weekly win rate
  ≈60% — the un-gated momentum book's own risk profile REGIME's `HEALTHY`
  state (Section 4.1) is calibrated to tolerate without intervention.
- CPCV (N=8 groups, k=2 test groups, 28 paths, 1-week embargo): mean OOS path
  Sharpe ≈0.91 vs naive single-path 0.98 (haircut ≈0.07, no collapse);
  frac_pos = 0.85–1.00 is the robust statistic, p05 is partition-fragile
  (+0.42 to −0.59) — this is why Section 5.4's promotion gate for a DD-stop
  explicitly weights `frac_pos` of the stop-vs-no-stop comparison over any
  single percentile, following the same robust-statistic discipline the
  digest already established for the raw strategy.
- Funding carry: +12.9%/yr income, ~70-75% maxDD, DSR fails (0.10–0.45);
  beta-hedge alone: Sharpe 0.48→0.57, maxDD 74.9%→69.9%; DD-stop alone:
  Sharpe 0.57→0.10, income 12.9%→0.6%/yr, maxDD capped ~24% — the exact
  numbers Section 5.3/5.5 cite verbatim as the honest carry precedent REGIME
  must NOT assume transfers to momentum (Invariant STOP-XFER, Section 5.4).
