# PORTFOLIO.md — build spec for rank-to-position construction (cross-sectional momentum book)

This file is the authoritative build spec for the PORTFOLIO pillar. Build to the
contracts and invariants below. Where this file and your own instinct disagree,
this file wins. Ask before deviating from any **Invariant**.

**Context shift from the old directional-breakout specs (COST.md, SYSTEM.md).**
Those specs were written for a single-instrument, single-trade system whose risk
unit was **one trade's R-multiple** (entry → stop/target → exit). That system —
directional breakout/trend, tested across timeframes, exits, filters, leverage,
three asset classes, even inverted — is **PROVEN DEAD**: gross directional R ≈ 0
everywhere (`docs/EDGE.md` §(a)). It is retained here only as background; it is
not the subject of this spec.

The live edge is **crypto cross-sectional momentum**: a weekly, dollar-neutral,
quintile-bucketed, inverse-vol-weighted book over a point-in-time universe of
liquid USDT-M perps (`docs/EDGE.md` §(b)). The risk unit is now **next week's
portfolio return** — a vector of simultaneous positions held over a fixed
horizon, not a single trade's excursion to a stop or target. PORTFOLIO is the
pillar that turns a cross-sectional **signal rank** (already computed upstream —
`research/xsec/signals.py`) into a **dollar-neutral weight vector**, and it is the
one pillar in the new system with **no directional predecessor**: nothing in the
old BREAKOUT/TREND/LEVERAGE stack answers "how many names, which quantile, how
much weight per name" for a simultaneous multi-asset book. This spec formalizes
that construction from first principles.

Scope of this document: rank → quantile bucketing → within-leg weighting →
dollar-neutral aggregation → turnover, at weekly (Friday-close) cadence. It does
**not** re-derive the signal (SIGNAL, upstream of this pillar, already computed
as `research/xsec/signals.py::momentum_signal` / `carry_signal` /
`combined_signal`), and it does **not** re-derive round-trip friction pricing
(COST.md owns fee/spread/impact structure; this spec only connects PORTFOLIO's
turnover definition to COST's per-unit friction, salvaged verbatim below).

**A note on evidence citation.** Every numeric constant in this document is
either derived from first principles (§2.3, §4.2) or **cited, never
invented**, from the underlying empirical record. Two granularities of source
exist and are cited by name, not conflated: (1) `docs/EDGE.md`, the top-line
decision memo, which carries only the headline numbers (net Sharpe, CAGR,
maxDD, win-rate, the blended PBO point estimate, the qualitative
frac_pos-vs-p05 guidance) — cited here as **`docs/EDGE.md` §(x)** wherever a
number appears there verbatim; (2) the finer-grained statistics that do
**not** appear in `docs/EDGE.md` itself (the per-partition PBO spread, the
`frac_pos` numeric range, the CPCV path-Sharpe-vs-naive haircut, the p05
partition-fragility swing, the `N=100` shortability/drag breakdown, and the
overlay collapse figures) — these are cited here as **EVIDENCE (task digest,
cross-referenced in `docs/specs/{SIGNAL,UNIVERSE,LEVERAGE,REGIME}.md`)**,
because that is where a reader can actually verify them against matching
figures in this repo's own sibling specs, rather than as "`docs/EDGE.md`" or
an unqualified "EVIDENCE digest," which would incorrectly imply they are
checkable in that one top-line file.

---

## 0. Governing principles (read before anything else)

**P1 — The risk unit is the WEEK, not the trade.** Every quantity in this spec is
indexed by a discrete weekly rebalance clock `t ∈ {0, 1, 2, …}` (Friday close to
Friday close). There is no stop, no target, no per-trade R. The portfolio's
P&L over `[t, t+1)` is the sum of `N_t` simultaneous position returns, not one
position's path to a barrier. Any invariant, formula, or acceptance gate written
in terms of "the trade" is out of scope for PORTFOLIO; the analogous quantity
here is "the week's realized book return" (COST.md's `WeekResult.net`,
`research/xsec/accounting.py`).

**P2 — Rank in, dollar-neutral weight out; PORTFOLIO is a provider, never a
forecaster.** PORTFOLIO consumes a cross-sectional rank vector
`rank_t ∈ [0,1]^{|U_t|} ∪ {NaN}` (SIGNAL's output) and the point-in-time eligible
universe `U_t` (UNIVERSE's output) and emits a weight vector `w_t`. It never
computes momentum, carry, or any predictive quantity itself, and it never
decides whether to trade a given week — that a `w_t` is dollar-neutral and
gross-normalized is a **structural** property of this pillar's output, not a
policy choice made per-instrument.

**P3 — Dollar-neutrality and gross-normalization are invariants, not defaults.**
For every `t`: `sum_{i: w_i>0} w_i = +G/2`, `sum_{i: w_i<0} w_i = -G/2`, hence
`sum_i w_i = 0` and `sum_i |w_i| = G` (gross exposure `G`, reference `G=1.0` =
100% gross = 50% long + 50% short). Because Sharpe of a linear book scales
invariantly under a uniform rescale of `G` (Lemma 1, §2), we report at `G=1.0`
and treat larger/smaller `G` as a pure leverage dial (**quarter-Kelly** sizing —
`docs/EDGE.md` §(e) — is applied by scaling `G`, not by re-deriving weights).

**P4 — Closed-form, diagonal weighting only; no fitted covariance optimizer.**
Within-leg weights are `w_i ∝ 1/σ̂_i` (inverse trailing realized vol), a
diagonal, closed-form estimator. A full Markowitz mean-variance optimizer over
the sample covariance matrix of the `|U_t| ≈ 50` names against `T ≈ 258` weekly
observations is explicitly **forbidden** — not because it is unimplementable,
but because the sample covariance is provably ill-conditioned at this
names-to-observations ratio (derived in §4.2, citing Marchenko & Pastur 1967 and
Ledoit & Wolf 2004). This mirrors the repo-wide NO-ML / closed-form-only
doctrine already declared in `pyproject.toml` ("scipy: closed-form (NOT ML)").

**P5 — Liquidity-capped weighting is a TESTED AND REJECTED default; capacity dial
only.** A liquidity-share cap on within-leg weight was implemented
(`research/xsec/portfolio.py::liquidity_weights`, scheme `liqcap_volscaled`) and
evaluated. It does not beat plain inverse-vol weighting net of cost, and it
provably **reduces** effective breadth in this specific market structure (§5).
It is retained in code as a capacity dial, gated on book size, never as the
default within-leg scheme (Invariant L).

**P6 — Deploy the lookback BAND, not a fitted point.** The momentum lookback `L`
is a family `L ∈ {12, 14, 16}` trading days, not a single number. CPCV
over the family shows the single best-in-sample `L` is partition-fragile
(PBO = 0.36 blended — `docs/EDGE.md` §(c) — with per-partition logit-relative-rank
spanning 0.31–0.47 and the best-IS pick landing on `L=12` in 21/28 splits yet
never failing clean `<0.2` — EVIDENCE, task digest, cross-referenced in
`docs/specs/SIGNAL.md` §4.1). PORTFOLIO's public contract
therefore takes an **ensemble of configs** and averages their independently
computed weight vectors, not one config's weights (§6). The band is the
sparse, tested set `{12, 14, 16}`, not every integer in `[12, 16]` — see §6.2
for why `L=13`/`L=15` are excluded.

**P7 — Turnover is priced by COST, not re-derived here.** PORTFOLIO computes
turnover as a pure geometric quantity — `0.5 * sum_i |w_new_i − w_carried_i|` —
and hands it to COST's `round_trip_friction()` for pricing. PORTFOLIO owns the
turnover **definition** (including the weight-drift mechanics, salvaged from
`research/xsec/accounting.py::settle_week`, §7); COST owns the friction **rate**.

Forbidden in v1: any full covariance-matrix (Markowitz/Σ⁻¹μ) optimizer fitted at
runtime (P4); liquidity-capped weighting as a default scheme (P5); deploying a
single fitted lookback `L` instead of the `{12,14,16}` band (P6); any neural network,
gradient-boosted tree, or other learned/black-box combiner for the rank or the
weight (repo-wide NO-ML doctrine); re-deriving friction rates inside this pillar
(P7 — COST.md is the one cost truth); sizing decisions (gross `G`, Kelly
fraction) made per-instrument rather than as a single book-level dial; treating
the p05 CPCV path-Sharpe as the primary robustness statistic instead of
`frac_pos` (§9, EVIDENCE per the citation convention above); using a static universe list instead of the
point-in-time eligible set `U_t` (UNIVERSE's responsibility, consumed here as
given); deploying `L=13` or `L=15` as ensemble members (§6.2 — no tested
backtest configuration exists for either value anywhere in the evidence base).

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its
first point of use in prose; this table is a lookup, not a substitute for
those definitions), mirroring `LEVERAGE.md` §0.1 / `SYSTEM.md` §0.1's format.
Week index `t` runs over Friday-close rebalances, the same clock
SIGNAL/LEVERAGE/SYSTEM use (`research/xsec/engine.py`).

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `F_t` | Information set (filtration) at rebalance `t` — no lookahead | — | 2.1 |
| `U_t`, `m_t` | Point-in-time eligible universe (UNIVERSE-owned); its size `m_t = \|U_t\|` | UNIVERSE-owned | 2.1, 4.1 |
| `rank_{i,t}`, `rank_t` | SIGNAL's cross-sectional percentile rank in `[0,1]`, NaN if `i ∉ U_t` | SIGNAL-owned | 2.1 |
| `w_t`, `w_{i,t}` | PORTFOLIO's output dollar-neutral weight vector / per-name weight — THE output of this module | — | 0, 2.1, 4 |
| `r_{i,t}`, `f_{i,t}` | Name `i`'s simple price return / summed funding accrual over `[t, t+1)` | data | 2.1 |
| `R_t` | Book's net weekly return (price P&L + signed funding − turnover friction) | — | 2.2 |
| `G` | Reference gross exposure (`G=1.0` = 100% gross = 50% long + 50% short) | RED (sizing dial, LEVERAGE-scaled) | 0, 2.3, 4.3 |
| `Φ_t(G)` | Funding cash-flow term at gross `G` (capital phi — never confused with filtration `F_t`) | derived | 2.3 |
| `q` | Quantile fraction (`q=0.2`, quintile) | RED | 3, 4.1 |
| `k_t` | Per-leg name count at `t`, `k_t = min(⌊q·m_t⌋ ∨ 1, ⌊m_t/2⌋)` | derived | 4.1 |
| `S_t`, `L_t` | Short leg / long leg index sets at `t` (bottom-`k_t` / top-`k_t` by rank) | derived | 4.1 |
| `σ̂_{i,t}` | Trailing realized vol estimator (flat 28-day window) for name `i` at `t` | TEAL | 4.2 |
| `Σ`, `Σ̂`, `D` | True / sample covariance matrix; `D = diag(Σ)` (the diagonal PORTFOLIO actually uses) | — | 4.2 |
| `γ = k/n` | Names-to-observations ratio in the Marchenko–Pastur eigenvalue-spreading argument | — | 4.2 |
| `C` | Liquidity-cap multiple (capacity dial, off by default) | RED (LEVERAGE-shared, §5.4/Invariant L) | 3, 5.1, 5.4 |
| `cap_i`, `dvol_i` | Name `i`'s liquidity-capped weight ceiling / trailing dollar volume | derived, TEAL | 5.1 |
| `breadth` | Effective breadth (inverse-HHI on per-name P&L contribution) | derived | 5.3 |
| `𝓛` | Deployed lookback band, `𝓛 = {12,14,16}` trading days (ensemble, not a point; sparse — anchored only on tested `L`s, §6.2) | BLUE | 3, 6.1–6.2 |
| `L` | One lookback in `𝓛`; `w_t^{(L)}` its independently-constructed weight vector | BLUE-member | 6.2 |
| `PBO` | Probability of backtest overfitting over the `𝓛` family (SYSTEM-owned machinery) | SYSTEM-owned | 6.1 |
| `Turnover_t` | `0.5·Σ_i \|w_{new,i,t} − w_{carried,i,t}\|` — pure geometric quantity | derived | 7.2 |
| `w_{carried,i,t}` | Prior target weight drifted forward by the realized week's return | derived | 7.2 |
| `c_rt` | COST's per-unit round-trip friction rate, `2·(taker+half_spread+impact)` | COST-owned | 7.3 |
| `Cost_t` | `Turnover_t · c_rt` — turnover priced by COST's rate | derived | 7.3 |
| `frac_pos`, `p05` | Fraction of CPCV paths with positive OOS Sharpe; 5th-pct path Sharpe (SYSTEM-owned) | SYSTEM-owned | 9 |
| `DSR`, `PSR`, `MinTRL` | Deflated/Probabilistic Sharpe, minimum track-record length (SYSTEM-owned) | SYSTEM-owned | 9 |
| `N` | Target universe size before the point-in-time listing mask (UNIVERSE's dial) | UNIVERSE-owned | 10 |

---

## 1. What PORTFOLIO builds on top of SIGNAL / UNIVERSE / COST (and what it must NOT rebuild)

PORTFOLIO sits between three sibling modules and does not reimplement any of
them:

| Sibling owns (do NOT rebuild)                                  | PORTFOLIO consumes it as                       |
|------------------------------------------------------------------|-------------------------------------------------|
| Point-in-time eligible universe `U_t`, listing/delisting mask     | `Panel.listed` (`research/xsec/panel.py`)       |
| Cross-sectional signal rank `rank_t ∈ [0,1]`                      | `momentum_signal` / `carry_signal` / `combined_signal` (`research/xsec/signals.py`) |
| Round-trip friction rate (fee + half-spread + impact, per leg)    | `round_trip_friction()` (`research/xsec/accounting.py`, backed by COST's `BINANCE_USDM_FEE_SCHEDULE`) |
| Signed funding cash-flow accrual                                  | `_week_funding` (`research/xsec/accounting.py`) |
| Realized-vol estimator infrastructure (EWMA / trailing-std)       | `realized_vol` (`research/xsec/portfolio.py`) — PORTFOLIO's own estimator, but the closed-form doctrine is shared with COST/SYSTEM |
| DSR / CPCV / purged walk-forward machinery                        | `convexity.system.validate.{deflated,cpcv,walkforward}` reused by `research/xsec/{validate,cpcv}.py` |

> **Invariant N-port — no re-derivation of sibling math.** PORTFOLIO does not
> compute a momentum or carry signal (that is SIGNAL's job, already implemented
> in `research/xsec/signals.py`), does not compute a friction rate in bps (that
> is COST's job), and does not implement its own DSR/CPCV estimator (that is
> SYSTEM's job, reused via `research/xsec/{validate,cpcv}.py`). A test asserts
> PORTFOLIO's turnover-cost number equals `turnover * round_trip_friction()`
> called with COST's own constants — not a locally reinvented fee/spread/impact
> triple.

---

## 2. The risk unit: from "one trade's R" to "next week's portfolio return"

### 2.1 Notation and information set

Fix a weekly rebalance grid `t = 0, 1, 2, …, T-1` (Friday UTC close to Friday UTC
close, `research/xsec/engine.py::_first_friday_index`). Let `F_t` be the
filtration (information set) generated by all daily closes, funding stamps, and
listing/delisting events up to and including day `t`'s close. Every quantity
PORTFOLIO computes at `t` — the rank `rank_t`, the vol estimate `σ̂_{i,t}`, the
weight `w_t` — must be `F_t`-measurable: a function of information known **at**
`t`'s close, never of any row after `t`. This is enforced structurally in
`research/xsec/panel.py` (the `listed` point-in-time mask) and
`research/xsec/signals.py` (every signal window ends at `d - skip ≤ d`).

Let `U_t ⊆ {1, …, N}` be the point-in-time eligible universe at `t` (names live
and with enough trailing history for the signal), `|U_t| = m_t`. Let
`rank_{i,t} ∈ [0,1]` be the SIGNAL pillar's cross-sectional percentile rank for
`i ∈ U_t` (NaN for `i ∉ U_t`). Let `w_t ∈ ℝ^N` be PORTFOLIO's output weight
vector at `t` (zero outside the selected legs). Let `r_{i,t}` be name `i`'s
simple price return over `[t, t+1)` and `f_{i,t}` its summed funding accrual
over the same window (both realized only as of `t+1`, hence not in `F_t`).

### 2.2 The portfolio return as the new risk unit

The book's **net weekly return** — the quantity every downstream statistic
(Sharpe, CAGR, maxDD, DSR, CPCV) is computed on — is

```
R_t = sum_i w_{i,t} * r_{i,t}            (price P&L, dollar-neutral)
    + sum_i (-w_{i,t} * f_{i,t})         (signed funding cash flow, COST §5.5 sign convention)
    - Turnover_t * c_rt                  (turnover friction, §7)
```

(`research/xsec/accounting.py::settle_week`, `WeekResult.net`.) This is the
direct replacement for the old system's per-trade R: **one observation is one
week's simultaneous multi-name outcome**, not one instrument's excursion to a
stop/target. `R_t` is realized at `t+1`'s close and is therefore `F_{t+1}`-
measurable, not `F_t`-measurable — exactly analogous to a trade's R being
unknown until the trade closes, but here the "trade" is the whole book and the
"close" is the next Friday.

### 2.3 Lemma 1 — Sharpe is gross-invariant, CAGR/maxDD are not

**Notation guard.** `F_t` (§2.1) denotes the filtration only, everywhere in
this spec. The funding cash-flow term introduced below is written `Φ_t(G)`
(capital phi), never `F_t(G)` — the two are unrelated objects (one a
sigma-algebra, the other a real-valued cash flow) and must not share a symbol.

**Claim.** For a fixed weight *shape* `w̃_t` (i.e., `w_t = G · w̃_t` for a scalar
gross `G` applied uniformly across `t`), the annualized Sharpe ratio of
`{R_t}_{t=0}^{T-1}` is invariant to `G`, while CAGR and maxDD scale
monotonically with `G`.

**Derivation — Sharpe invariance.** Write
`R_t(G) = G · X_t + Φ_t(G) - G · Turnover_t · c_rt` where
`X_t = sum_i w̃_{i,t} r_{i,t}` is the gross-normalized price P&L and, to first
order, funding cash flow `Φ_t(G) = G · sum_i (-w̃_{i,t} f_{i,t})` also scales
linearly in `G` (funding is charged on the same linearly-scaled notional).
Turnover `Turnover_t = 0.5 sum_i |w_{new,i,t} - w_{carried,i,t}|` scales exactly
linearly in `G` too, because both `w_new` and `w_carried` scale linearly in `G`
(the carry-forward drift in §7 is a linear operation on `w`). Hence
`R_t(G) = G · Y_t` for a `G`-independent random variable
`Y_t := X_t + sum_i(-w̃_{i,t} f_{i,t}) - Turnover_t · c_rt`. Then

```
Sharpe(G) = E[R_t(G)] / Std[R_t(G)] * sqrt(52.18) = (G·E[Y_t]) / (G·Std[Y_t]) * sqrt(52.18) = Sharpe(1)
```

for any `G > 0` — Sharpe is scale-invariant.

**Derivation — CAGR is concave in `G` (worked, not asserted).** Define
`Equity_t(G) = prod_{s=0}^{t-1} (1 + G·Y_s)`, `Equity_0(G) = 1`, and
`CAGR(G) := Equity_T(G)^{1/T} - 1` (`T` = number of weekly periods,
annualization by `52.18/T` suppressed here for clarity — it is a constant
positive rescale of the exponent and does not affect concavity in `G`). Take
logs: `log(1 + CAGR(G)) = (1/T) sum_{t=0}^{T-1} log(1 + G·Y_t)`. Expand each
term in `G` around `G=0` via the standard second-order Taylor expansion of
`log(1+x)` at `x=0`, `log(1+x) = x - x^2/2 + O(x^3)`, with `x = G·Y_t`:

```
log(1 + G·Y_t) = G·Y_t - (1/2)·G^2·Y_t^2 + O(G^3)
```

Averaging over `t` and using the law-of-large-numbers approximation
`(1/T) sum_t Y_t ≈ E[Y_t]` and `(1/T) sum_t Y_t^2 ≈ E[Y_t^2] = Var[Y_t] + E[Y_t]^2`
(both exact in expectation, approximate path-wise for finite `T`):

```
log(1 + CAGR(G)) ≈ G·E[Y_t] - (1/2)·G^2·(Var[Y_t] + E[Y_t]^2) + O(G^3)
```

Differentiate in `G` and evaluate at `G=0`:
`d(CAGR)/dG |_{G=0} = d(log(1+CAGR))/dG |_{G=0} = E[Y_t]` (chain rule,
`d(log(1+CAGR))/dCAGR |_{CAGR=0} = 1`), confirming the linear term. The
second-order (curvature) term in the expansion of `log(1+CAGR(G))` is
`-(1/2)·G^2·(Var[Y_t] + E[Y_t]^2)`, which is **strictly negative** for any
`G > 0` whenever `Var[Y_t] > 0` (true whenever the book has any return
dispersion at all — the degenerate `Var[Y_t]=0` case is excluded by
construction since `Y_t` sums `m_t ≥ 2` risky legs). Because
`log(1+CAGR(G))` is concave in `G` to second order and `CAGR` is a monotone
increasing transform of `log(1+CAGR)`, `CAGR(G)` itself is concave in `G` over
the range where the Taylor truncation is valid (regularity condition below):
it rises approximately linearly at `E[Y_t]` for small `G`, then bends over and
falls once the `-(1/2)G^2 Var[Y_t]` variance-drag term dominates the linear
mean term — the textbook geometric-vs-arithmetic-return wedge (Kelly-criterion
variance drag), here derived explicitly rather than quoted.

**Derivation — maxDD is monotone increasing (worse) in `G`.** Fix any two
gross levels `0 < G_1 < G_2`. Consider the log-equity path
`ℓ_t(G) := log Equity_t(G) = sum_{s<t} log(1+G·Y_s)`, and let
`d(G) := max_{0≤u≤v≤T} [ℓ_u(G) - ℓ_v(G)]` be the maximum log-drawdown (maxDD in
log space; the level-space maxDD `1 - exp(-d(G))` is a monotone increasing
function of `d(G)`, so it suffices to show `d(G)` is monotone in `G`). Because
`log(1+x)` is concave and increasing, and because the sample paths `{Y_t}` are
held fixed across the two `G` values (we are asking how the *same realized
history* of gross-normalized returns compounds at different leverage, exactly
Lemma 1's fixed-shape setup), the per-period log-increment
`log(1+G·Y_t)` is a **concave function of `G`** for each fixed `Y_t`
(second derivative `-Y_t^2/(1+G Y_t)^2 < 0`, valid on the domain `G Y_t > -1`).
For a concave function `h(G)` with `h(0)=0`, the ratio `h(G)/G` is
non-increasing in `G` (a standard concavity property: the secant slope from
the origin can only shrink as `G` grows). Apply this per-period with
`h(G) := log(1+G Y_s)`: for `Y_s < 0` (a losing day/week), `h(G) < 0` for all
`G>0`, and `h(G)/G` non-increasing means `h(G)` becomes more negative than a
linear extrapolation from small `G` would predict — i.e. losses compound
**super-linearly** worse in `G`, precisely the well-known leveraged-decay
mechanism. Summing this per-period inequality over any interval `[u,v]` where
the path is realized *losing* (`sum_{s=u}^{v-1} Y_s < 0`, a drawdown segment)
gives `sum_{s=u}^{v-1} log(1+G_2 Y_s) ≤ (G_2/G_1) · sum_{s=u}^{v-1} log(1+G_1 Y_s) < sum_{s=u}^{v-1} log(1+G_1 Y_s)`
for `G_2 > G_1 > 0` (the last inequality because the segment sum is negative
and `G_2/G_1>1`), i.e. the log-drawdown magnitude on that fixed segment is
strictly larger (worse) at the higher gross level. Taking the max over all
segments `[u,v]` preserves this monotonicity (`d(G)` is a max of functions
each non-decreasing in `G`, hence itself non-decreasing in `G`).
**Consequence:** reporting Sharpe at the reference `G=1.0` (100% gross) is a
legitimate, leverage-independent statistic; CAGR and maxDD are only
meaningful *together with* the stated `G` — which is why `docs/EDGE.md` always
quotes maxDD "at 100% gross" (~25%) and why quarter-Kelly sizing is applied as
a post-hoc scale on `G`, never baked into the weight *shape* `w̃_t`. This is
the formal justification for **P3**.

> **Invariant SHARPE-INV — Lemma 1's gross-invariance is a testable,
> code-level contract, not only a derivation.** A regression fixture must
> replay one committed weekly-return path `{Y_t}` (§2.3's fixed-shape
> object) at two distinct gross levels `G_1 ≠ G_2` (e.g. `G_1=1.0`,
> `G_2=1.5`) through `research/xsec/metrics.py`'s Sharpe computation and
> assert the two annualized Sharpe ratios agree to within floating-point
> tolerance (`abs(Sharpe(G_1) - Sharpe(G_2)) < 1e-9`, the tolerance
> reflecting pure floating-point rounding in the `G_t·Y_t` scaling, not a
> approximation of the underlying identity, which is exact). The same
> fixture must assert CAGR and maxDD **differ** between `G_1` and `G_2`
> (monotonically, per the concavity/monotonicity derivations above) — a
> test that found CAGR/maxDD ALSO gross-invariant would indicate the fixture
> is not actually scaling `G` (e.g. accidentally re-normalizing weights back
> to unit gross before settling the week), the inverse failure mode this
> invariant is meant to catch alongside the Sharpe-invariance check itself.

**Assumptions/regularity.** (i) The linear-scaling argument for funding and
turnover assumes friction and funding rates are exogenous to the book's own
size (true for a book far below the venue's own liquidity — the capacity
discussion in §5 is precisely about where this assumption breaks). (ii) The
Sharpe invariance is exact only for a *fixed shape*; it does NOT hold if
increasing `G` forces a *different* weighting scheme (e.g., invoking the
liquidity cap of §5 at larger book size), which is exactly why §5 frames
capping as a capacity dial that trades Sharpe for capacity, not a scale-free
operation. (iii) The CAGR Taylor expansion is a local (small-`G`) result; it
is valid as long as `G·Y_t > -1` for every realized `t` (equity never hits
zero — always true at `G ≤ 1` given the dollar-neutral, diversified `Y_t`
construction, but the second-order truncation quietly assumes `G` is not so
large that higher-order terms in the expansion dominate — a formal statement
of exactly the "variance drag dominates the mean" regime named in the
Consequence above). (iv) The maxDD monotonicity derivation assumes the
realized path `{Y_t}` is held fixed across the compared `G` values (the same
fixed-shape assumption as Sharpe invariance) — it is a pathwise, not a
distributional, statement: it says a *given* realized drawdown segment gets
worse at higher `G`, not that the *distribution* of maxDD shifts by any
specific parametric amount, which is why `docs/EDGE.md`'s "~25% at 100%
gross... scales ~linearly with gross" is reported as an empirical
near-linear approximation over the tested `G` range, not re-derived here as
an exact closed form (the exact scaling is path-dependent per the derivation
above, and only approximately linear for `G` well inside the region where the
Taylor truncation in (iii) holds).

---

## 3. Parameter governance

Every free parameter this pillar introduces is exactly one bucket of the
repo-wide RED/AMBER/BLUE/TEAL discipline (`src/convexity/core/params.py`,
`pyproject.toml`'s closed-form-only doctrine), the same taxonomy
`LEVERAGE.md` §3, `SYSTEM.md` §10, `COST.md` §4, and `UNIVERSE.md` §8 each
restate for their own parameters. This section states PORTFOLIO's own bucket
ownership explicitly — the prose in §4.2 (the parenthetical "more free
(RED-bucket) parameters to misconfigure") and §9 ("BLUE-grid discipline does
not help a track-length gate") refers back to this table, not to a bucket
assignment left implicit.

- **RED — shared economic/modelling constants, set once globally, reachable
  only through `resolve_config(defaults=...)`.**
  - The quantile fraction `q = 0.2` (quintile, §4.1) — a single global choice
    applied to every rebalance, never per-instrument or per-leg; see §4.1's
    worked derivation below for why `q=0.2` rather than a finer/coarser
    fraction.
  - The realized-vol window `window = 28` days (§4.2,
    `research/xsec/portfolio.py::realized_vol`'s default) — one estimator
    window for every name's `σ̂_{i,t}`, not tuned per-instrument.
  - The liquidity-cap tightness dial `C ∈ {2, 3}` (§5.1) — **ownership note**:
    `C` is the same symbol LEVERAGE's own capacity ceiling uses for its
    parallel liquidity-cap multiple (`LEVERAGE.md` §5.2). To avoid the
    ambiguity the reviewer flagged, the ownership split is: PORTFOLIO's `C`
    governs the **within-leg weighting scheme** (`liquidity_weights`'s
    `cap_i = C · dvol_i/Σdvol_j`, §5.1) — a capacity dial on *how a leg's
    weights are shaped*; LEVERAGE's `κ`/`C` (its §5.1's `κ=0.02` ceiling and
    its own capacity-dial multiple) govern *how large the book's gross `G`
    may grow* given the shaped weights PORTFOLIO already produced. They are
    RED in both pillars, but PORTFOLIO's `C` is upstream of, and structurally
    prior to, LEVERAGE's — LEVERAGE's capacity ceiling is computed against
    the reference-gross weights PORTFOLIO's `quantile_weights`/
    `liquidity_weights` already emitted, never the reverse. A promotion
    memo that cites "the capacity dial `C`" without saying which pillar's
    `C` it means is the ambiguity this note exists to foreclose.
  - Reference gross `G = 1.0` (§2.3, §4.3) — the sizing dial LEVERAGE scales;
    PORTFOLIO fixes it at 1.0 for every headline number in this spec (P3).
- **BLUE — finite, pre-declared candidate grids searched offline, promoted
  only through SYSTEM's deflated-Sharpe/CPCV gate.**
  - The lookback band `𝓛 = {12, 14, 16}` trading days (§6.1–§6.2) —
    the canonical BLUE grid this pillar deploys as an ensemble rather than a
    fitted point (P6, Invariant B-band); its cardinality (`|𝓛|=3`) is counted
    into SYSTEM's trial ledger exactly as `ARCHITECTURE.md` §3 requires for
    every BLUE grid in the system.
  - The quantile granularity choice itself, `quantile ∈ {"quintile",
    "tercile"}` (`research/xsec/portfolio.py::Quantile`) is the finite,
    pre-declared alternative set `q=0.2` was validated against (§4.1) — a
    two-point BLUE grid, not a continuously-tunable dial; no other quantile
    granularity is implemented or v1-deployable.
- **TEAL — fitted per-name from strictly causal (`F_{t-1}`/`F_t`) past data
  at runtime, never a hand knob, never full-sample, never P&L-fit.**
  - The trailing realized vol `σ̂_{i,t}` itself (§4.2) — fitted fresh each
    rebalance from the trailing `window=28`-day close series, never a static
    per-instrument constant.
  - The trailing dollar-volume `dvol_i` feeding the liquidity cap (§5.1) —
    same causal, per-rebalance-refreshed discipline.
- **AMBER — supplied per-instrument market metadata PORTFOLIO consumes but
  never optimizes.** Whether a name is shortable (binding on `S_t`'s
  membership via UNIVERSE's upstream filter, §10, Invariant U-short) and its
  listing/delisting timestamps (`Panel.listed`) are AMBER, owned by UNIVERSE
  — PORTFOLIO reads them, never sets them.

> **Invariant C-cfg-port — RED unreachable from per-instrument config**
> (inherited verbatim from COST/SYSTEM/LEVERAGE's own Invariant C). `q`,
> `window`, `C`, and reference `G` enter `quantile_weights`/
> `liquidity_weights` only via `resolve_config(defaults=...)` or an explicit
> keyword default on the function signature itself (`research/xsec/portfolio.py`)
> — never as a per-symbol override threaded through `Panel` or `InstrumentSpec`.
> If any of these four appears keyed by instrument, that is the bug.

---

## 4. Rank-to-position construction

### 4.1 Quantile bucketing via order statistics

At rebalance `t`, let `m_t = |U_t|` and let `rank_{(1),t} ≤ rank_{(2),t} ≤ … ≤
rank_{(m_t),t}` be the order statistics of `{rank_{i,t} : i ∈ U_t}`. Fix a
quantile fraction `q = 0.2` (quintile — validated choice vs. tercile,
`research/xsec/portfolio.py` docstring; `docs/EDGE.md` §(b) confirms quintile
over tercile net of cost). Define the per-leg name count

```
k_t = min( floor(q * m_t) ∨ 1,  floor(m_t / 2) )
```

(`research/xsec/portfolio.py::_leg_count`) — floored at 1 so a small early
universe still forms a book, and capped at `floor(m_t/2)` so the two legs never
overlap. The **short leg** is `S_t = { i ∈ U_t : rank_{i,t} ≤ rank_{(k_t),t} }`
(bottom-`k_t` by rank) and the **long leg** is
`L_t = { i ∈ U_t : rank_{i,t} ≥ rank_{(m_t - k_t + 1),t} }` (top-`k_t` by rank).
Names strictly between the two cutoffs are flat (`w_{i,t} = 0`). If `m_t < 2` no
book can form (`k_t = 0`, all-flat) — see Invariant E1 (§11).

**Why order statistics, not a fixed absolute rank threshold.** The rank
`rank_{i,t}` is already a percentile (`_pct_rank`, `research/xsec/signals.py`),
so the `q`-quantile cutoff via order statistics is automatically adaptive to
`m_t`: at `m_t = 50` this is 10 names/leg, at `m_t = 60` it is 12 names/leg — the
selection rule scales with the live universe without any re-calibration, which
is exactly what a point-in-time universe (names entering/leaving over the
5-year sample) requires.

> **Invariant Q-ADAPT — leg size is a function of `m_t`, never a hardcoded
> constant.** `_leg_count(n_eligible, quantile)` must return `k_t =
> min(⌊q·m_t⌋∨1, ⌊m_t/2⌋)` recomputed fresh from the CURRENT rebalance's
> `m_t`, never a cached or hardcoded name count carried over from a prior
> rebalance or a design-time universe size. A test asserts `_leg_count`
> called with two different `n_eligible` values in the same test process
> returns two different `k_t` values whenever `q·m_t` crosses an integer
> boundary between them (e.g. `_leg_count(50, "quintile") == 10` and
> `_leg_count(60, "quintile") == 12` in the same assertion, not two separate
> tests that could each pass against a stale hardcoded 10 or 12). This is
> the code-level contract behind the "automatically adaptive to `m_t`" claim
> above — a claim about behavior, promoted here to a specific,
> two-input-values regression check.

**Derivation — why a quintile is the plausible a priori tradeoff point,
before consulting the CPCV result.** Fixing `m_t`, the choice of `q` is a
bias-variance tradeoff on the leg, in the same spirit as LEVERAGE §2.3's
`c(2−c)` derivation of why a *quarter*- or *half*-Kelly fraction is the right
range rather than merely citing "quarter/half-Kelly are standard." A larger
`q` (e.g. tercile, `q=1/3`) admits a larger per-leg name count `k_t =
min(⌊q·m_t⌋∨1, ⌊m_t/2⌋)` (§4.1's formula); a smaller `q` (e.g. decile,
`q=0.1`) admits a smaller `k_t`. Two opposing effects move with `k_t`:

1. **Estimation-noise dilution (favors a LARGER `k_t`, i.e. a coarser
   quantile like a tercile).** A leg's realized return is the (inverse-vol
   weighted) average of `k_t` names, so it inherits each name's
   idiosyncratic noise attenuated roughly by the `1/√k_t` averaging-down
   identity (the same estimation-precision logic that motivates preferring
   many individually-estimated `σ̂_i` terms over few jointly-estimated
   covariance terms in §4.2's Marchenko–Pastur argument, applied here to the
   *count* of names rather than the *covariance structure* among them). A
   leg of `k_t≈17` (tercile at `m_t≈50`) averages down idiosyncratic noise
   more than a leg of `k_t≈5` (decile at the same `m_t`) — this effect
   favors a LARGER `k_t`, i.e. a coarser quantile (tercile over decile).
2. **Signal dilution (favors a SMALLER `k_t`, i.e. a finer quantile like a
   decile).** The expected cross-sectional rank spread captured by a leg is
   `E[|rank_{(j),t} − 0.5|]` averaged over the `k_t` order statistics
   actually included. Because order statistics of a near-uniform `[0,1]`
   rank distribution are denser near the median than near the tails (the
   `j`-th order statistic out of `m_t` has a Beta`(j, m_t−j+1)` density,
   which concentrates around `j/m_t` — points near the middle of the sorted
   list are, in expectation, closer to 0.5 than points near either end),
   *widening* the leg from a decile to a tercile (i.e., admitting MORE
   names, moving `k_t` toward `m_t/2`) necessarily admits marginal names
   whose expected `|rank − 0.5|` is smaller than the names already in the
   tighter (decile-sized) leg — each added name dilutes the leg's average
   distance from the median, hence its expected exposure to the
   cross-sectional spread the momentum signal is trying to monetize. This
   effect favors a SMALLER `k_t`, i.e. a finer quantile (decile over
   tercile).

**Where quintile plausibly sits.** These two effects move in opposite
directions as `q` varies, so the optimal `q` is an interior point, not a
boundary — a purely mechanical reason to expect the answer to land strictly
between "as fine as possible" (a single pair of extreme names, `k_t=1`:
maximal signal concentration, maximal per-name idiosyncratic noise, no
noise-averaging at all) and "as coarse as possible" (`k_t = ⌊m_t/2⌋`, the
largest leg the construction allows: maximal noise-averaging, but the leg
is by then just "top half vs. bottom half," no longer a "conviction" leg —
the median-adjacent names it admits carry close to zero expected rank
spread). A decile (`q=0.1`) sits closer to the concentrated end of this
tradeoff, a tercile (`q=1/3`) closer to the diluted end, and a quintile
(`q=0.2`) is the interior point roughly equidistant between them on a log
scale (`log(0.1), log(0.2), log(1/3)` are approximately evenly spaced) —
consistent with, but not a proof of, why the tested optimum (§4.1's opening
citation: quintile beats tercile net of cost, `research/xsec/portfolio.py`
docstring, `docs/EDGE.md` §(b)) lands where it does rather than at the
tested tercile boundary. **This derivation motivates why a quintile is a
priori a reasonable interior candidate, not a guarantee that `q=0.2`
exactly maximizes the tradeoff** — the actual optimum among the two GRID
POINTS this repo tests (quintile, tercile; §3's BLUE grid: no decile point
is implemented or tested, `research/xsec/portfolio.py::Quantile`) is, and
remains, the cited empirical CPCV/net-of-cost result, not a closed-form
solution to the tradeoff above — the two effects are not quantified into a
single closed-form objective in this spec (that would require assuming a
specific parametric shape for the rank-spread-vs-`k_t` and noise-vs-`k_t`
curves, which this repo's
closed-form-but-not-invented-numbers discipline declines to do without a
citation), so the CITED comparison remains the deciding evidence, exactly as
stated before this derivation, and the derivation's role is only to explain
*why* an interior quantile was worth testing against the boundary-adjacent
tercile in the first place, not to replace that empirical test.

### 4.2 Within-leg weighting: inverse realized vol, and why not full Markowitz

**The weighting rule.** Within leg `L_t` (symmetrically for `S_t`), weight
name `i` proportional to its inverse trailing realized vol:

```
w_{i,t} = sign(leg) * (G/2) * (1/σ̂_{i,t}) / sum_{j ∈ leg} (1/σ̂_{j,t})
```

where `σ̂_{i,t}` is the trailing sample standard deviation of daily simple
returns over a `[t-28, t]`-day window (`research/xsec/portfolio.py::realized_vol`,
default `window=28`), an unweighted (flat) trailing estimator — a special case
of an EWMA with weight concentrated uniformly over the last 28 observations.
Names with no finite `σ̂` (insufficient trailing history) fall back to the
leg's mean inverse-vol so they still participate (documented fallback in
`quantile_weights`). This is the "risk-parity-within-leg" convention: a
low-vol name receives a *larger* weight so that, to first order, each name
contributes comparable **risk**, not comparable **notional**, to the leg.

**Derivation — why diagonal (inverse-vol) weighting is preferred to a full
mean-variance optimum at this sample size.**

The textbook minimum-variance (or mean-variance) weight for a leg of `k`
names with return covariance matrix `Σ ∈ ℝ^{k×k}` is `w* ∝ Σ^{-1} 1` (or
`Σ^{-1}μ` if also optimizing a signal). Inverse-vol weighting is the special
case `w ∝ D^{-1} 1` where `D = diag(Σ)` — i.e., it discards every off-diagonal
(correlation) term and uses only the individual variances. The question is
whether the off-diagonal information in a *sample* `Σ̂` is worth using at
`k ≈ 10–15` per leg (`m_t/2 · q`, i.e. `k_t` from §4.1) estimated from
`T ≈ 258` weekly observations (`docs/EDGE.md` §(d), ~5yr sample) — or, if
`Σ̂` is instead estimated from *daily* returns to get more observations, at
`k ≈ 50–75` (`m_t`, the full universe, since a leg's true covariance
structure is a submatrix of the universe-wide one) against a daily sample of
comparable order to `k` once you account for the effective independence
horizon.

The Marchenko–Pastur theorem (Marchenko & Pastur, 1967) answers this
directly. For a `k × k` sample covariance matrix estimated from `n` i.i.d.
observations of a distribution with true covariance `σ²I` (the null of no
correlation structure), as `k, n → ∞` with `k/n → γ ∈ (0,1)`, the empirical
spectral distribution of the eigenvalues of `Σ̂` converges to the
Marchenko–Pastur law supported on

```
[ σ²(1 − sqrt(γ))² ,  σ²(1 + sqrt(γ))² ]
```

— i.e., even when the **true** covariance is a multiple of the identity (no
real correlation structure at all), the sample eigenvalues spread out over a
band of width `∝ sqrt(γ)` purely from estimation noise. The **smallest**
sample eigenvalue is pushed down toward `σ²(1-sqrt(γ))²` (potentially near
zero as `γ → 1`) even though every true eigenvalue is exactly `σ²`. A
minimum-variance optimizer `w* ∝ Σ̂^{-1}1` places its **largest** weights
exactly on the directions of the **smallest** sample eigenvalues — by
construction it seeks "cheap" (low apparent variance) combinations — so it
systematically overweights the eigenvectors that are most contaminated by
estimation noise. This is not a small-sample nuisance; it is the defining
failure mode of naive mean-variance optimization (formalized further by
Ledoit & Wolf 2004, who show the sample covariance's largest eigenvalues are
biased **upward** and its smallest **downward**, and propose linear shrinkage
toward a structured target as the fix precisely because computing a
well-conditioned inverse from `Σ̂` directly is unsound at these `k/n` ratios).

Plugging in this repo's numbers: at the leg level, `k_t ≈ 10` names against
`T ≈ 258` **weekly** observations gives `γ = k/n ≈ 0.04` — survivable in
isolation — but the weekly sample is exactly what SIGNAL/PORTFOLIO have to
work with (daily granularity does not help: adjacent-day crypto returns are
strongly autocorrelated through the same weekly rebalance information set, so
the *effective* sample size for a covariance estimate that must be stable at
the weekly holding horizon is closer to `T ≈ 258` than to `~1800` raw daily
rows). At the **universe** level — which is what a true joint optimizer would
need, since the two legs' relative sizing is itself a portfolio decision —
`k = m_t ≈ 50–75` against `T ≈ 258` gives `γ ≈ 0.19–0.29`, deep in the regime
where Marchenko–Pastur predicts severe eigenvalue spreading and Ledoit-Wolf
shrinkage would need a shrinkage intensity `δ` well above zero to be usable at
all — at which point the "optimizer" is mostly reproducing a **shrunk,
diagonal-dominated** estimate, i.e. converging back toward inverse-vol
weighting, just with more machinery and more free (RED-bucket) parameters to
misconfigure. Since **P4** (this repo's closed-form-only, no-fitted-black-box
doctrine) already forbids a full fitted optimizer, and the derivation above
shows the fitted optimum would be numerically dominated by noise at this
`k/n` ratio anyway, **inverse-vol (diagonal-only) weighting is not a
simplification made under duress — it is the estimator that is actually
better-conditioned at this sample size**: individual variances `σ̂_i²`
(one parameter each, `k` total) are estimated far more reliably from `T ≈ 258`
weekly (or even the 28-day trailing daily) observations than the
`k(k-1)/2 ≈ 45–2775` off-diagonal covariance terms a full `Σ̂` would require.

**Assumptions/regularity.** This argument assumes the *cross-sectional*
correlation structure among the `k` names in a leg is not so strong and
persistent that ignoring it costs more than the noise saved — an assumption
validated empirically, not just theoretically: `docs/EDGE.md` §(c) records
that plain vol-scaled weighting **beats** liquidity-capped weighting net of
cost in the actual backtest (the liquidity-cap experiment is the closest this
repo comes to testing a non-diagonal weighting scheme, and it lost, §5). It
also assumes `σ̂_i` itself is a reasonably stable estimator over the 28-day
window; a CUSUM-style regime-break detector (Page, 1954) on `σ̂_i` is a
natural closed-form extension if vol regimes are found to shift faster than
28 days, but is **not** implemented in v1 (§14, Non-goals).

### 4.3 Dollar-neutrality constraint (restated formally)

For every rebalance `t`:

```
sum_{i ∈ L_t} w_{i,t} = +G/2
sum_{i ∈ S_t} w_{i,t} = -G/2
sum_i w_{i,t} = 0                    (dollar-neutral)
sum_i |w_{i,t}| = G                  (gross exposure exactly G)
```

This holds by construction: `_leg()` in `research/xsec/portfolio.py` normalizes
each leg's inverse-vol magnitudes to sum to exactly `±G/2` before assignment,
so the constraint is an algebraic identity of the implementation, not a
post-hoc check. Reference gross is `G = 1.0` (100% gross = 50% long + 50%
short); Sharpe at any other `G` is identical by Lemma 1 (§2.3), so CAGR/maxDD
figures must always be quoted alongside the `G` used to produce them
(`docs/EDGE.md`: "maxDD ~25% at 100% gross exposure (~28% annualized vol) and
scales ~linearly with gross/leverage" — EVIDENCE digest).

---

## 5. Liquidity-capped weighting: formalized negative result

### 5.1 The scheme that was tested

`research/xsec/portfolio.py::liquidity_weights` (scheme `liqcap_volscaled`)
starts from the same inverse-vol base magnitude as §4.2, then caps each name's
share of its leg at `C` times its **dollar-volume share of the leg**:

```
cap_i = C * ( dvol_i / sum_{j ∈ leg} dvol_j )
```

via an iterative water-filling procedure (`_leg_liq_cap`): any name whose
inverse-vol share exceeds `cap_i` is clamped to `cap_i`, and the freed mass is
redistributed proportionally among the remaining uncapped names (subject to
their own caps), iterated to a fixed point. `C = ∞` (no binding cap) reduces
**exactly** to plain vol-scaled weighting — the two schemes are nested, so the
comparison in §4.2 is a clean ablation of one mechanism, not a comparison of
unrelated designs.

### 5.2 Why it binds tightest on majors, not small names — the mechanism

**Claim.** In the crypto USDT-M perp universe, liquidity (dollar volume) and
realized volatility are *negatively* correlated across the market-cap
spectrum: majors (BTC, ETH, and the next tier) are simultaneously the
**highest-liquidity** and the **lowest-volatility** names, while small-cap
alts are simultaneously **lower-liquidity** and **higher-volatility**. This is
the standard cross-sectional pattern in crypto (large-cap coins have deeper
order books and lower idiosyncratic vol; small-caps are thin and violently
volatile) and it is exactly what the base weighting scheme already
overweights: inverse-vol weighting (§4.2) gives a major its **largest** base
weight (low `σ̂` → large `1/σ̂`) purely because it is a major.

Now apply the liquidity cap. Because a major's dollar-volume share of the leg
`dvol_i / sum dvol_j` is *also* the largest in the leg (it is simultaneously
the most liquid name), its cap `cap_i = C · (\text{its own largest liquidity
share})` is the **loosest** cap in the leg — it can absorb the largest
inverse-vol weight before the constraint even binds. Conversely, a small
alt's liquidity share is small, so its cap is the **tightest** — it is
clamped down even though (being high-vol) its base inverse-vol weight was
already small to begin with, and it is now clamped further, with the freed
mass redistributed disproportionately back toward... the already-loose-capped
majors. **The mechanism therefore concentrates the book onto majors rather
than diversifying it away from them** — the opposite of a capacity-motivated
liquidity constraint's naive intent.

### 5.3 The empirical measurement: effective breadth via inverse-HHI

Effective breadth is measured as the inverse Herfindahl-Hirschman index on
per-name total absolute contribution to book P&L
(`research/assetclass.py::effective_breadth`):

```
breadth = 1 / sum_i w̄_i²,     w̄_i = |contrib_i| / sum_j |contrib_j|
```

A book whose P&L is carried entirely by one name has `breadth → 1`; a
perfectly even `n`-name book has `breadth = n`. Under liquidity-capped
weighting, empirically, **breadth FELL from ~40 to 28–34** relative to plain
vol-scaled weighting (EVIDENCE, task digest, cross-referenced in
`docs/specs/UNIVERSE.md` §3.3 and `docs/specs/LEVERAGE.md` §5.2) — the capped
scheme is *less* diversified, exactly as §5.2 predicts, not more.

### 5.4 What capping DOES buy, and the resulting rule

The cap is not worthless — it buys **capacity**. Median deployable book size
was measured at **~$3M uncapped**, rising to **~$12–23M at cap tightness
`C = 2–3`** (EVIDENCE, task digest, cross-referenced in `docs/specs/UNIVERSE.md`
§3.3 and `docs/specs/LEVERAGE.md` §5.2), at a cost of **~0.4–0.6 Sharpe** relative
to plain vol-scaled. This is a genuine capacity-vs-Sharpe tradeoff, not a free
lunch and not a pure loss either.

> **Invariant L — liquidity capping is a capacity dial, never a default.**
> `weighting="volscaled"` (plain inverse-vol, §4.2) is the default and the one
> used for every headline number in this spec. `liqcap_volscaled` /
> `liqtilt` / `eqcap` (`research/xsec/portfolio.py::Weighting`,
> `liquidity_weights`) are gated behind an explicit book-size threshold
> (deploy only once AUM approaches the uncapped ~$3M capacity ceiling) and
> must always be reported alongside the Sharpe cost they buy. A promotion
> decision that silently switches to a capped scheme without a stated AUM
> justification is the bug this invariant exists to catch.

---

## 6. The lookback ensemble (deploying a band, not a point)

### 6.1 Why a single fitted `L` is partly overfit

The momentum signal (`research/xsec/signals.py::momentum_signal`) is
parameterized by a lookback `L` (trading days). A grid search over `L` finds a
sharp peak at `L = 14`. CPCV (López de Prado combinatorial purged
cross-validation, `N=8` groups, `k=2` test-groups per split, `C(8,2) = 28`
paths, 1-week embargo, `research/xsec/cpcv.py::run_cpcv`) evaluates whether
that peak is a genuine signal or a partition-dependent artifact via the
probability-of-backtest-overfitting (PBO) statistic (Bailey, Borwein, Salehipour
& López de Prado, 2014 — logit-of-relative-rank estimator,
`research/xsec/cpcv.py::compute_pbo`): for each of the 28 splits, pick the
config with the best **in-sample** Sharpe on the embargo-purged train
complement, find *that* config's **out-of-sample** relative rank among the
family on the held-out test blocks, map the rank to a logit
`logit(w) = log(w/(1-w))`, and `PBO = P(logit ≤ 0)` (best-IS config lands
at/below the OOS median — no better than a coin flip at picking the true best
config).

**The measured result**: PBO = **0.31–0.47** across partitions (EVIDENCE, task
digest, cross-referenced in `docs/specs/SIGNAL.md` §4.1; the blended figure
"PBO = 0.36" is the single number `docs/EDGE.md` §(c) reports — the wider
0.31–0.47 per-partition spread is not in `docs/EDGE.md` itself), with the
best-in-sample pick landing on `L=12` in **21 of 28** CPCV
splits, and PBO never falling cleanly below 0.2 in any partition scheme
tried. A PBO in this range is neither the reassuring near-zero of a robust
signal nor the damning near-one of a purely fitted one — it says **picking a
single best `L` is partly overfitting the sample**, so the deployment
decision cannot rely on the point estimate at `L=14` alone.

### 6.2 The ensemble construction

**Rule.** Deploy the lookback **band** `𝓛 = {12, 14, 16}` (trading
days) as an ensemble, not a single `L`. At each rebalance `t`, PORTFOLIO
computes one independent rank vector `rank_t^{(L)}` and one independent
weight vector `w_t^{(L)}` (§4, full quantile + inverse-vol construction) **for
each `L ∈ 𝓛` separately**, then averages the resulting weight vectors:

```
w_t = (1/|𝓛|) * sum_{L ∈ 𝓛} w_t^{(L)}
```

**Why `{12, 14, 16}` (m=3), not every integer `{12, 13, 14, 15, 16}` (m=5).**
The band is a **sparse set anchored on tested points**, not a dense fill of
the interval `[12, 16]`. The evidence base for this pillar is the peak-width
refinement grid `L ∈ {7, 10, 12, 14, 16, 18, 21, 28, 35}` and the CPCV/PBO
validation grid `L ∈ {10, 12, 14, 16, 18, 21, 28}` (§6.1) — every `L` in `𝓛`
must be a member of one of these two grids, because those are the only
lookbacks with an explicit, cited, tested Sharpe figure anywhere in this
pillar's evidence (e.g. the peak-width refinement's named per-`L` Sharpe at
`N=50` — "L14: 1.59, L16: 1.70" — and the breadth-peak figures — "L14 1.09,
L16 1.07"). `L=13` and `L=15` are **not** members of either grid: no backtest
was ever run at those lookbacks, so no Sharpe, PBO contribution, or any other
statistic exists for them at any point in this investigation's history. A
band of `{12, 13, 14, 15, 16}` would silently interpolate two untested
configurations into the deployed ensemble — exactly the kind of unevidenced
number the citation discipline elsewhere in this spec (§9's `[EDGE]`/`[TASK]`
tagging, the References preamble's "never invented" rule) exists to forbid.
`{12, 14, 16}` is therefore the correct deployable band: three lookbacks,
evenly spaced across the same `L=12`–`16` neighborhood the CPCV run in §6.1
covers, every one of them individually evidenced.

**Why average weights, not average ranks.** Averaging the *rank* vectors
before bucketing would collapse the ensemble back into a single quantile
decision with one blended signal — exactly the single-config selection
problem PBO flags as fragile, just relabeled. Averaging the realized
*weight* vectors instead means each `L`'s quintile decision is made
independently (a name can be in the long leg under `L=12` and flat or in the
short leg under `L=16`), and the final book is the equal-weight blend of
three independently-constructed dollar-neutral books. Each `w_t^{(L)}`
individually satisfies §4.3's dollar-neutrality identity, and
dollar-neutrality is preserved under the average (a convex combination of
zero-sum vectors is zero-sum: `sum_i w_{i,t} = (1/|𝓛|) sum_L sum_i
w_{i,t}^{(L)} = (1/|𝓛|) sum_L 0 = 0`, with `|𝓛|=3` here), so the ensembled
book is itself dollar-neutral at gross `G` with no separate re-normalization
needed.

**Robustness interpretation.** This is a variance-reduction argument, not a
signal-strengthening one: if the three `L`'s each carry a genuine but noisy
momentum signal, averaging their independently-realized weights reduces the
book's exposure to any single `L`'s partition-specific overfitting (the
mechanism PBO measures) without requiring the ensemble members to be
decorrelated by design — they are highly overlapping lookback windows and
*will* be correlated, so this is closer to a shrinkage-toward-the-family-mean
than a true diversification argument, consistent with treating `frac_pos`
(the fraction of CPCV paths with positive Sharpe across the family, §9)
rather than any single path's Sharpe as the trustworthy statistic. A
three-member ensemble shrinks less aggressively toward the family mean than
a five-member one would, but the two members it omits (`L=13`, `L=15`) carry
no evidence to shrink *with* — including them would not add tested
robustness, only interpolated arithmetic.

> **Invariant B-band — the deployable config is always the `𝓛`-ensemble.**
> A promotion, paper, or live-sizing decision that reports or sizes off a
> single-`L` weight vector (e.g. `L=14` alone) rather than the three-member
> average is a policy violation of P6, not a valid alternative
> parameterization — regardless of how favorable that single `L`'s backtest
> looks, because §6.1 has already shown that favorability is partition-
> selected. Conversely, widening `𝓛` to include any `L` without a tested
> Sharpe figure in the §6.1 evidence grids (e.g. `L=13`, `L=15`) is equally a
> policy violation — of P6's evidence-grounding, not its band-not-point
> requirement.

---

## 7. Rebalance cadence and turnover (salvaged from COST.md / `accounting.py`, reframed)

### 7.1 Weekly cadence, Friday close

Rebalance day `t` snaps to the first Friday at/after the signal's warmup
window, then steps 7 calendar days (`research/xsec/engine.py::_first_friday_index`,
`run`). Every hold period is Friday-close to Friday-close. This cadence is a
direct, load-bearing input to the friction economics: paying the round-trip
cost **52 times a year** (not thousands, as the old 1-minute directional
system did) is the structural reason the ~13bp round-trip friction barely
bites the weekly book (`docs/EDGE.md` §(b): "Weekly rebalance means the 13bp
friction is paid ~52×/yr, not thousands of times — so it barely bites.").

> **Invariant CADENCE-52 — the rebalance step is exactly 7 calendar days,
> never a variable or data-dependent step.** `_first_friday_index` sets the
> FIRST rebalance; every subsequent rebalance is exactly `+7` calendar days
> from the previous one (`research/xsec/engine.py::run`), regardless of
> whether the intervening week contained a delisting, a data gap, or a
> holiday-adjacent low-volume period. A test asserts the full `dates`
> array of a multi-year `XSecResult` has a constant `7`-calendar-day
> `diff()` throughout (excluding only the single first-rebalance offset from
> warmup), and a second test asserts the annualization factor used by
> `metrics.py` is the fixed constant `PERIODS_PER_YEAR = 365.25/7 ≈ 52.18`
> — never a period count inferred from the realized number of rebalances in
> a given sample, which would silently understate/overstate annualized
> Sharpe on a truncated or gap-containing sample. This is the code-level
> contract behind the "~52×/yr" friction claim: the claim is only true
> because the cadence is a fixed 7-day step, not an approximation that could
> drift if a future change made the step data-dependent.

### 7.2 Turnover definition (geometric quantity, PORTFOLIO's to own)

```
Turnover_t = 0.5 * sum_i | w_{new,i,t} - w_{carried,i,t} |
```

`w_{new,i,t}` is this rebalance's target weight (§4–§6). `w_{carried,i,t}` is
**not** simply last period's target weight — it is that weight **drifted
forward by the realized return over the just-elapsed week**
(`research/xsec/accounting.py::settle_week`):

```
w_{carried,i,t} = w_{new,i,t-1} * (1 + r_{i,t-1})     if i was live at both t-1 and t
                = 0                                     if i delisted over [t-1, t)  (position closed at last live day)
```

The `0.5` factor is the standard convention so a full flip of one unit of
gross from name A to name B counts as one unit of turnover (`|+1| + |-1| = 2`,
halved to `1`), not two.

**Why drift, not the raw prior target.** A name that ran up over the
elapsed week arrives at the next rebalance already carrying a *larger* weight
than it was originally assigned — even if PORTFOLIO's quintile decision keeps
it in exactly the same leg with exactly the same target weight next period,
rebalancing back down to that target is a real trade with real turnover. Using
the un-drifted prior target would silently **understate** turnover (and hence
cost) for any name that moved, which is precisely the kind of optimistic
accounting error the honest-cost discipline in this repo is built to catch
(cf. COST.md P6: conservative under uncertainty). A delisted name is treated
as closed at its last live day (drift → 0), not held into a return-undefined
gap.

### 7.3 Handing turnover to COST (the seam)

PORTFOLIO computes `Turnover_t` (a pure number in `[0, G]`) and passes it to
COST's per-unit round-trip friction rate:

```
Cost_t = Turnover_t * c_rt,       c_rt = round_trip_friction(taker, half_spread, impact)
                                        = 2 * (taker + half_spread + impact)
```

(`research/xsec/accounting.py::round_trip_friction`, backed by COST's
`BINANCE_USDM_FEE_SCHEDULE` for the taker leg). The factor of 2 is because a
unit of turnover is a **round trip** (enter one side, and the accounting
convention already nets the exit of the other) — priced on both legs. The
**salvaged constant**: taker = 5bp (Binance USDT-M VIP0, the dominant term),
half-spread = 1bp (crypto-major research fallback, matches
`research.config`), impact = 0.5bp (conservative fallback at low size) →
`c_rt = 2*(5+1+0.5)bp = 13.0bp` round-trip — the same number the COST pillar's
own worked example uses (COST.md §5.7), reused here verbatim, not
re-derived. Per EVIDENCE, the **real** cost model replaces the flat
half-spread/impact fallback with a size-aware Amihud illiquidity estimator
(§7.4) for the honest Sharpe figure; the flat 13bp is the "light-cost" number
that produces the optimistic headline (§7.1).

> **Invariant T-drift — turnover is priced on the drifted, not the static,
> prior weight.** A cost calculation that uses `|w_{new} - w_{prior\_target}|`
> instead of `|w_{new} - w_{carried}|` is undercounting cost and must not be
> used for any reported Sharpe/CAGR figure (§7.2).

### 7.4 Real cost: the Amihud impact estimator (cautionary note salvaged from COST)

The **honest** net Sharpe figures in this spec (§9) price impact with a real
Amihud (2002)-style illiquidity estimator anchored to dollar volume — a
`~1/sqrt(ADV)` curve calibrated so the median liquid name pays ~1bp — **not**
a Corwin–Schultz high/low spread estimator applied to 1-minute-aggregated
daily bars. The latter was tried and rejected: it conflates realized
**volatility** with **spread** (both inflate the daily high/low range), and
on this data it produced a spurious **~135bp / ~40%-annualized-drag** cost
estimate — an order of magnitude too pessimistic and a documented dead end
(`docs/specs/COST.md` §5.4.3, Invariant CS-REJECT — the authoritative source
for this figure; not in `docs/EDGE.md`). PORTFOLIO does not implement either
estimator (that is COST's component layer, `components/impact.py` in the COST.md directory
layout); this note exists here only so a future contributor tuning the
turnover-to-cost seam does not resurrect the Corwin–Schultz path by mistake.

---

## 8. Worked numeric example (reproduce this bit-for-bit)

All numbers below are **[ILLUSTRATIVE]** — pure arithmetic applied to the
formulas of §4–§7 on a small, hand-picked illustrative universe, exactly
the [ILLUSTRATIVE] tagging convention `LEVERAGE.md` §7 uses for its own
worked sizing example (contrast with the `[EDGE]`/`[TASK]` tags used
throughout this spec's §9 table, which classify a number's SOURCE — this
section instead classifies a number as arithmetic illustration rather than
a real measurement; the two taggings answer different questions and are not
interchangeable). This section walks §4's quantile-and-weighting
construction and §7's turnover/cost seam end-to-end on concrete numbers,
which the citation-heavy §9 table does not do.

**Setup.** An illustrative universe of `m_t = 10` eligible names at
rebalance `t`. Per §4.1: `q = 0.2`, so
`k_t = min(⌊0.2·10⌋∨1, ⌊10/2⌋) = min(2, 5) = 2` names per leg. Reference
gross `G = 1.0`, so each leg targets `±G/2 = ±0.5`.

**Step 1 — quantile bucketing (§4.1).** Suppose the rank order statistics
place names `L1, L2` in the top-2 (long leg `L_t`) and `S1, S2` in the
bottom-2 (short leg `S_t`); the remaining 6 names are flat.

**Step 2 — within-leg inverse-vol weights (§4.2).** Illustrative trailing
28-day realized vols: `σ̂_{L1}=0.05`, `σ̂_{L2}=0.10`, `σ̂_{S1}=0.08`,
`σ̂_{S2}=0.04` (daily, unannualized — the raw units `realized_vol` returns).
Inverse vols: `1/σ̂_{L1}=20`, `1/σ̂_{L2}=10` (long leg sum `=30`);
`1/σ̂_{S1}=12.5`, `1/σ̂_{S2}=25` (short leg sum `=37.5`). Per §4.2's formula
`w_{i,t} = sign(leg)·(G/2)·(1/σ̂_i)/Σ(1/σ̂_j)`:

```
w_L1 = +0.5 · (20/30)   = +0.5 · 0.66667 = +0.33333
w_L2 = +0.5 · (10/30)   = +0.5 · 0.33333 = +0.16667
w_S1 = -0.5 · (12.5/37.5) = -0.5 · 0.33333 = -0.16667
w_S2 = -0.5 · (25/37.5)   = -0.5 · 0.66667 = -0.33333
```

**Step 3 — dollar-neutrality check (§4.3).**

```
sum(long)  = 0.33333 + 0.16667 = 0.50000   =  +G/2   ✓
sum(short) = -0.16667 + -0.33333 = -0.50000 =  -G/2   ✓
sum(all)   = 0.50000 + (-0.50000) = 0.00000            ✓ dollar-neutral
sum(|all|) = 0.33333+0.16667+0.16667+0.33333 = 1.00000 =  G       ✓
```

This is the §4.3 identity reproduced bit-for-bit on concrete numbers, not
merely asserted algebraically.

**Step 4 — one week elapses; drift the weights forward (§7.2).**
Illustrative realized weekly returns over `[t, t+1)`: `r_{L1}=+10%`,
`r_{L2}=-2%`, `r_{S1}=+5%`, `r_{S2}=-3%`. Per §7.2's drift rule
`w_{carried,i,t+1} = w_{new,i,t}·(1+r_{i,t})`:

```
w_carried_L1 = 0.33333 · 1.10 = 0.36667
w_carried_L2 = 0.16667 · 0.98 = 0.16333
w_carried_S1 = -0.16667 · 1.05 = -0.17500
w_carried_S2 = -0.33333 · 0.97 = -0.32333
```

**Step 5 — the next rebalance's NEW target weights (§4.1–§4.3 again, new
`σ̂`).** Suppose the same four names remain in the same legs (illustrative
— a real rebalance could reshuffle leg membership entirely; this example
holds membership fixed to isolate the vol-reweighting and drift effects)
with updated trailing vols `σ̂_{L1}=0.06`, `σ̂_{L2}=0.09`, `σ̂_{S1}=0.07`,
`σ̂_{S2}=0.05`. Inverse vols: `16.667, 11.111` (long sum `27.778`);
`14.286, 20.000` (short sum `34.286`):

```
w_new_L1 = 0.5 · (16.667/27.778) = 0.5 · 0.60000 = 0.30000
w_new_L2 = 0.5 · (11.111/27.778) = 0.5 · 0.40000 = 0.20000
w_new_S1 = -0.5 · (14.286/34.286) = -0.5 · 0.41667 = -0.20833
w_new_S2 = -0.5 · (20.000/34.286) = -0.5 · 0.58333 = -0.29167
```

(Dollar-neutrality check: `0.30000+0.20000=0.50000`;
`-0.20833+-0.29167=-0.50000`; net `= 0.00000` — holds again, as §6.2's
convex-combination argument guarantees for any single `w_t^{(L)}`.)

**Step 6 — turnover (§7.2's formula, on the DRIFTED prior weight, per
Invariant T-drift).**

```
Turnover_t = 0.5 · ( |0.30000-0.36667| + |0.20000-0.16333|
                    + |-0.20833-(-0.17500)| + |-0.29167-(-0.32333)| )
           = 0.5 · ( 0.06667 + 0.03667 + 0.03333 + 0.03167 )
           = 0.5 · 0.16833
           = 0.08417
```

**Contrast — what Invariant T-drift forbids.** Using the UN-drifted prior
target (`0.33333, 0.16667, -0.16667, -0.33333`) instead of the carried
weight would give
`Turnover_t' = 0.5·(|0.30000-0.33333|+|0.20000-0.16667|+|-0.20833-(-0.16667)|+|-0.29167-(-0.33333)|)
= 0.5·(0.03333+0.03333+0.04167+0.04167) = 0.5·0.15000 = 0.07500` — a
**~11% smaller** (understated) turnover than the correct `0.08417`, exactly
the optimistic-accounting failure mode Invariant T-drift (§7.2) exists to
catch: the un-drifted calculation silently ignores that `L1` and `S2` moved
during the week and so already carry weight away from their own prior
target before the new target is even computed.

**Step 7 — pricing turnover through COST's seam (§7.3).**

```
Cost_t = Turnover_t · c_rt = 0.08417 · 0.0013 = 0.0001094   (≈ 1.09bp of gross)
```

using the salvaged `c_rt = 13.0bp` round-trip figure (§7.3). This
`Cost_t` is exactly the third term of §2.2's `R_t` decomposition for this
illustrative week.

**Reading this example.** Every number above is either an input
declared as illustrative (the `σ̂` and `r` values) or a direct arithmetic
consequence of §4–§7's formulas applied to those inputs — no cited
empirical figure is used or implied by this section, and this section does
not (and cannot) stand in for the cited CPCV/DSR validation numbers in §9,
which remain the actual evidence for whether this construction is a real
edge.

---

## 9. Validation numbers this pillar's construction must be judged against

PORTFOLIO's job is rank→weight construction; its correctness is judged by the
weekly net-return series it produces once COST's friction and SIGNAL's rank
are held fixed. The following are the **cited, not invented**, benchmark
numbers. Per the citation convention in the preamble: rows marked `[EDGE]` are
verbatim in `docs/EDGE.md`; rows marked `[TASK]` are the finer-grained EVIDENCE
from the task digest, cross-referenced in the named sibling spec — do not
restate different numbers in either category without a new measurement:

| Statistic | Honest (real Amihud cost) | Light-cost / single-peak headline | Source |
|---|---|---|---|
| Net Sharpe (annualized) | **~0.8–1.0** per lookback in the band | ~1.6–1.9 (cite only as "optimistic upper bound, not deployable") | `[EDGE]` §(b) |
| CAGR @ 100% gross | ~20–25% | ~65% | `[EDGE]` §(b) |
| maxDD @ 100% gross | **~25%** (annualized vol ~28%; scales ~linearly with gross) | — | `[EDGE]` §(b)/(d) |
| Weekly win rate | **~60%** | — | `[EDGE]` §(b) |
| CPCV mean OOS path Sharpe (`N=8`, `k=2`, 28 paths, 1-wk embargo) | **~0.91** | naive single-path Sharpe ~0.98 (haircut ~0.07 — **no OOS collapse**) | `[TASK]`, cross-ref `UNIVERSE.md` §7, `LEVERAGE.md` §7, `EXECUTION.md` §11 |
| `frac_pos` (fraction of CPCV paths with Sharpe > 0) | **0.85–1.00** — the ROBUST statistic | p05 path Sharpe swings **+0.42 to −0.59** across `N`/`k` choices — PARTITION-FRAGILE, do not lead with it | `[TASK]`, cross-ref `UNIVERSE.md` §7, `LEVERAGE.md` §2.6/§7 |
| PBO (lookback-family overfitting) | **0.36** blended (0.31–0.47 per-partition); best-IS pick = `L=12` on 21/28 splits | — | `0.36` is `[EDGE]` §(c); `0.31–0.47`/`21/28` is `[TASK]`, cross-ref `SIGNAL.md` §4.1 |
| DSR | **fails on track-length only** — PSR ≈ 0.994 (high), `n_obs ≈ 258` weekly obs falls short of the multiplicity-inflated MinTRL | — | track-length framing is `[EDGE]` §(d); PSR ≈ 0.994 point figure is `[TASK]`, cross-ref `SIGNAL.md` §7 |

**Reading this table.** The CPCV haircut (0.98 → 0.91, only ~0.07) is
qualitatively different from the old directional-breakout system's CPCV
result, which flagged `luck_artifact=True` with OOS paths straddling zero
(the old system's own SYSTEM.md §7 worked example) — the cross-sectional
book's OOS Sharpe does **not** collapse, which is the central piece of
evidence that this edge is real and not a backtest artifact. The DSR failure
is a **track-record-length** problem (not enough weekly observations yet to
clear the multiple-testing-adjusted MinTRL, even though the point PSR is
high) — a genuinely different failure mode from a low PSR, and PORTFOLIO's
rollout plan (§14, sizing) treats it accordingly: paper/live re-confirmation
lengthens `n_obs` over time, which is the only lever that fixes a MinTRL
failure (deflating further via BLUE-grid discipline does not help a
track-length gate).

> **Invariant R-stat — `frac_pos`, not p05, is the primary CPCV robustness
> read.** Any promotion memo or dashboard that leads with the p05 path Sharpe
> as the headline robustness number, rather than `frac_pos`, has the wrong
> headline: EVIDENCE (task digest, cross-referenced in `docs/specs/UNIVERSE.md`
> §7 and `docs/specs/LEVERAGE.md` §2.6/§7) shows p05 swings by more than a
> full Sharpe unit (+0.42 to −0.59) purely from the `(N, k)` partition choice,
> while `frac_pos` (0.85–1.00) is stable across the same partition choices.

---

## 10. Universe size and the shortability constraint (connects PORTFOLIO to UNIVERSE)

PORTFOLIO's leg construction (§4.1) operates on whatever `U_t` UNIVERSE
supplies; it does not choose `U_t` itself, but the choice of `N = |U_t|`
(target universe size, before the point-in-time listing mask thins it on any
given day) materially changes the quintile leg composition and must be stated
here because it interacts directly with the dollar-neutrality mechanics
(§4.3: the short leg requires shortable names).

**The measured tradeoff**: net Sharpe **peaks around `N ≈ 60–75`** (this
peak-location claim is `docs/EDGE.md` §(c): "net edge peaks around 60-75
names") **and declines toward `N = 100`**, driven by two real effects, not
just dilution — these two mechanisms and their magnitudes are EVIDENCE (task
digest, cross-referenced in `docs/specs/UNIVERSE.md` §3, not itself in
`docs/EDGE.md`): (i) the added illiquid tail costs an incremental **~+2.9%/yr
drag** once priced with the real Amihud impact estimator (§7.4), and (ii) of
the ~50 names added going from `N=50` to `N=100`, roughly **15 are effectively
unshortable** on the venue — meaning part of the apparent gain from expanding
the universe is not realizable, because a name that cannot be shorted cannot
occupy the short leg `S_t` (§4.1) even if its rank qualifies it. **The gain of
`N ≈ 60–75` over `N = 50` is inside the block-bootstrap confidence interval**
(Kunsch 1989 moving-block bootstrap, block length per Politis & White 2004
automatic block-length selection) — i.e. not statistically distinguishable
from noise. **`N ≈ 50` is therefore the defensible simple choice**: smaller
than the empirical peak, but the peak's excess over `N=50` is not a real,
tested improvement, and `N=50` avoids the shortability drag entirely.

> **Invariant U-short — a name failing the venue's shortability check is
> excluded from `U_t` upstream (UNIVERSE), not silently zero-weighted here.**
> PORTFOLIO's leg construction (§4.1) assumes every `i ∈ U_t` is tradeable on
> both sides. If UNIVERSE ever passes through an unshortable name, the
> correct fix is to remove it from `U_t`, not to special-case it inside
> `quantile_weights` — a special case here would silently break the "order
> statistics over `U_t`" cleanliness of §4.1.

---

## 11. Edge cases and invariants (structural; enforce, do not special-case around)

**E1 — degenerate universe (`m_t < 2`).** If fewer than 2 names are eligible
at `t`, no dollar-neutral book is possible; `_leg_count` returns `k_t = 0` and
`quantile_weights`/`liquidity_weights` return the all-zero vector (flat book,
zero turnover from the prior book, only exit cost if a name delisted out of a
prior position). This is a real early-universe condition (the very first few
weeks of the panel, before `m_t` has grown past the signal's warmup), not an
error state.

**E2 — a name with no finite `σ̂_i` at rebalance time.** Falls back to the
leg's mean inverse-vol (`np.nanmean(iv)` in `quantile_weights::_leg`), so it
still participates at a neutral (average) weight rather than being silently
dropped or crashing the normalization. This differs from E1: the name IS
eligible for the *rank* (it has enough momentum history) but not yet for the
*vol estimator* (28-day trailing window not yet full) — the two eligibility
checks are independent and this is the documented resolution when they
diverge.

**E3 — a name delists mid-week (after a rebalance, before the next).** Its
week return `r_{i,t}` is undefined (no close at `t+1`); `_asset_week_return`
returns NaN, `settle_week` treats it as closed at its last live day (weight
drifts to exactly 0 for the *next* period's turnover calculation), and its
contribution to that week's `gross_price` is excluded (not imputed as zero
return, which would be a fabricated observation — the position is simply
marked closed at the last true price). No holding into a delist gap (this is
the same discipline `panel.py`'s point-in-time mask enforces going into a
listing).

**E4 — all names in a leg share the same rank (a tie at the quantile
boundary).** `_pct_rank`'s average-rank convention (`research/xsec/signals.py`)
assigns tied raw scores the same percentile, so a tie exactly at the `q`-
quantile boundary is broken by `argsort`'s stable (mergesort) tie-order,
which is deterministic but not economically meaningful at the margin — this
is accepted as a negligible-probability edge case for continuous-valued
momentum/carry signals and is not separately mitigated (a genuine near-zero-
probability event for real-valued trailing returns).

**E5 — `L` band member has insufficient warmup at the very start of the
panel.** Each `w_t^{(L)}` in the ensemble (§6.2) has its own warmup
(`lookback + skip + rebalance_days`, `research/xsec/engine.py::run`). Since
`𝓛 = {12, 14, 16}` are close in magnitude, all three warm up within days of
each other; the ensemble average (§6.2) is only computed from the first
rebalance at which **all** three members are warm (the max over the band),
which trims a few extra days off the front of the deployable series relative
to any single `L` — an acceptable, small cost of ensembling.

**E6 — `G` scaled up into the capacity-constrained regime.** Lemma 1 (§2.3)
assumes friction and funding rates are exogenous to book size. Once `G` (or
AUM) grows toward the ~$3M uncapped capacity ceiling (§5.4), this assumption
degrades — realized impact rises with the book's own footprint, which is
exactly the trigger for switching the within-leg scheme to a liquidity-capped
variant (Invariant L) rather than continuing to scale `G` on the plain
vol-scaled shape.

**E7 — funding-carry signal used as the sole rank (not combined/momentum).**
§4–§6 are written for the momentum band, but the quantile-bucketing and
inverse-vol weighting machinery (§4) is signal-agnostic — it operates on
whatever `rank_t` it is handed. If `carry_signal` or `combined_signal`
(`research/xsec/signals.py`) is substituted, §4's construction is unchanged,
but §6's PBO/band argument is specific to the momentum lookback family and
does **not** transfer to a carry lookback without its own CPCV re-run; and per
`docs/EDGE.md` §(a)/(d) ("~70% structural drawdown... income overlay at
most"), funding carry is a satellite income signal at most — structural
~70-75% max drawdown standalone is the more precise figure, EVIDENCE (task
digest, cross-referenced in `docs/specs/REGIME.md` §7/§11) — PORTFOLIO
must not default to sizing a book off `carry_signal` alone as if it had the
momentum book's risk profile.

**E8 — ensemble members disagree on `U_t` or produce misaligned-length
outputs.** §6.2's ensemble average `w_t = (1/|𝓛|) sum_L w_t^{(L)}` is only a
valid convex combination if every `w_t^{(L)}` is defined over the **same**
index set `{1,…,N}` at the same `t` — i.e. all three `L`-configs must be run
against the identical panel and the identical point-in-time universe `U_t`
(UNIVERSE's output, §1), differing only in the signal's lookback window, never
in which names are eligible. The per-`L` driver that exists today,
`research/xsec/engine.py::run(panel, cfg)`, already makes this safe **by
construction for a single `L`**: it takes one `Panel` object (one `U_t`
sequence, one `n_symbols`) as a parameter rather than building it internally,
so the natural way to run the ensemble is to construct one `Panel` once and
call `run(panel, cfg)` three times (once per `cfg.lookback ∈ 𝓛`) against that
same object — there is no code path inside `run` that could give two calls a
different universe if they are handed the same `panel`. **What is NOT yet
implemented, and is exactly the gap this edge case flags:** the §6.2
ensemble-averaging driver itself (the loop that calls `run`/`run_book` three
times and averages the resulting `w_t^{(L)}` vectors) does not exist yet in
`research/xsec/` as of this spec — §6.2 is a construction PORTFOLIO's
contract requires, not yet a built and tested code path (§13's acceptance
gate for this exact driver states the precise build contract — signature,
file path, and test — a future implementer must satisfy). **Invariant
E-align** (build obligation for that future driver): it must construct a
single `Panel` once and pass that same object (not three independently-built
panels, even if built from the same underlying data) to every `L` in the
band, and its test suite must assert byte-identical `panel.symbols`/`panel.dates`
identity across all three calls before allowing the weight-average in §6.2 to
run — silently averaging `w_t^{(L)}` vectors built from independently
re-constructed (and potentially subtly different, e.g. differently-ordered)
panels would misalign-and-sum incorrectly (numpy would either broadcast-error
or, worse, silently mis-add by position if shapes coincidentally matched).
This is distinct from E5 (which is about warmup **timing** — a difference in
**when** the ensemble average starts once all members share one universe) and
from E1-E4 (which are all about a single `L`'s construction in isolation) —
E8 is the one edge case that is specific to what can go wrong only once
multiple independently-parameterized `w_t^{(L)}` vectors must be combined.

---

## 12. Directory layout (implemented; this spec documents, does not restructure)

```text
research/
  xsec/
    panel.py          # UNIVERSE: point-in-time daily close/funding panel + listed mask
    signals.py         # SIGNAL: momentum / carry / combined cross-sectional rank
    portfolio.py        # PORTFOLIO (this spec): quantile_weights, liquidity_weights, realized_vol
    accounting.py       # turnover + friction + signed funding -> WeekResult.net (§7, seam to COST)
    engine.py           # driver: panel -> signal -> portfolio -> accounting, per config, per week
    overlay.py           # beta-hedge / drawdown-stop overlays (post-PORTFOLIO; not this spec's scope)
    cpcv.py              # CPCV path distribution + PBO over the lookback family (§6, §9)
    metrics.py            # weekly Sharpe/CAGR/maxDD/turnover-drag/funding-income summary
    validate.py            # purged walk-forward + DeflatedSharpe reuse (SYSTEM pillar, §9)
    universe.py             # (thin) UNIVERSE.md loader
  assetclass.py                # sleeve config (friction/regimes) + effective_breadth (§5.3)
  data.py                       # bar/funding loaders panel.py builds on
  universe.py                    # UNIVERSE.md parser
src/convexity/
  cost/                           # COST pillar: fee schedule PORTFOLIO's accounting.py imports (§7.3)
  system/validate/{deflated,cpcv,walkforward}.py   # SYSTEM pillar: DSR/CPCV/walk-forward reused by research/xsec/{validate,cpcv}.py
```

PORTFOLIO's own logic is `research/xsec/portfolio.py` (quantile bucketing +
within-leg weighting, §4–§5) plus the turnover half of `accounting.py` (§7);
the remaining files above are the sibling pillars it composes with, listed
for orientation, not because this spec governs them.

**Whole-system placement.** This tree is the `research/xsec/` half of the
xsec-momentum system; how it composes with the three surviving
`src/convexity/{core,cost,system}/` pillars and the other new pillar specs
(SIGNAL, UNIVERSE, LEVERAGE, REGIME, EXECUTION, OPS) end-to-end is mapped in
`docs/ARCHITECTURE.md` §1–§2 ("The nine pillars, the weekly rebalance, and
the code they map to" / "Weekly data-flow (the composition graph)"), added
alongside this rewrite specifically so a reader is not left to reconstruct
the composition graph from eight separate spec files.

---

## 13. Module acceptance gates (CI)

Mirroring `LEVERAGE.md` §10's per-file "Pass when: ..." contract. The first
two gates cover code that **already exists** in `research/xsec/portfolio.py`
(§4–§5 of this spec); the third gate is a **specification artifact** for the
§6.2 ensemble-averaging driver that, per Invariant E-align (§11, E8), **does
not exist yet** — its signature, file path, and test are a build contract a
future implementer must satisfy, not a description of running code.

### Quantile + within-leg weighting (`research/xsec/portfolio.py::quantile_weights`) — EXISTING
Pass when: `quantile_weights(rank, *, quantile, weighting, gross=1.0,
vol=None)` returns a vector satisfying the dollar-neutrality identity
exactly (§4.3: long leg sums to `+gross/2`, short leg to `-gross/2`, `sum_i
w_i = 0`) as a unit-test assertion at machine-epsilon tolerance, not an
approximate check; `_leg_count(n_eligible, quantile)` returns `0` for
`n_eligible < 2` and never divides within a leg of size `0` (Invariant E1,
§11); `weighting="volscaled"` is the resolved default absent an explicit
override (Invariant L, §5.4) — a test asserts the function's own keyword
default, not merely a caller convention; a name with non-finite `vol` falls
back to the leg's mean inverse-vol (E2, §11), verified by a fixture with at
least one `NaN`-vol name per leg.

### Liquidity-capped weighting (`research/xsec/portfolio.py::liquidity_weights`) — EXISTING
Pass when: `C = ∞` (or the caller's sentinel for "no cap") reproduces
`quantile_weights`'s `volscaled` output bit-for-bit (§5.1's nesting claim,
a regression fixture, not just a prose assertion); the measured effective-
breadth regression (§5.3: ~40 → 28–34 under the cap) is reproduced within a
stated tolerance by a fixture replaying the same historical panel through
both schemes; `liquidity_weights` is never the `resolve_config(defaults=...)`
default (Invariant L) — a static check asserts no call site sets
`weighting="liqcap_volscaled"` without an explicit, logged AUM-threshold
justification string.

### Ensemble-averaging driver (`research/xsec/engine.py::run_ensemble`) — SPECIFICATION, NOT YET BUILT
**Status: this function does not exist in `research/xsec/` as of this spec
(Invariant E-align, §11, E8).** The signature, file path, and test below are
PORTFOLIO's build contract for a future implementer — they are written with
the same type-annotation style as the real, existing `run(panel, cfg) ->
XSecResult` in `research/xsec/engine.py` (verified against that file's
current contents) so the contract is drop-in compatible with the module it
extends, not a hypothetical API invented independently of the real code.

```python
# research/xsec/engine.py  (proposed addition — does not exist yet)

def run_ensemble(panel: Panel, cfgs: list[XSecConfig]) -> XSecResult:
    """Average the dollar-neutral weight vectors of |cfgs| independently-run configs (§6.2).

    Each ``cfg`` in ``cfgs`` (typically one per lookback L in the band 𝓛 = {12, 14, 16}, §6.1)
    is run against the SAME ``panel`` object via ``run(panel, cfg)`` — never a separately
    constructed Panel per cfg, even from identical underlying data (Invariant E-align).
    The returned ``XSecResult`` is the equal-weight average of the per-cfg weight paths,
    re-settled through ``accounting.settle_week`` at the averaged weight (not an average of
    the per-cfg net returns, which would NOT equal the return of the averaged book whenever
    turnover/cost is nonlinear in the weight path).

    Raises ``ValueError`` if any two ``cfgs`` yield ``XSecResult`` objects whose ``dates``
    are not identical (fails loudly rather than silently misaligning).
    """
```

Pass when (the exact test a future implementer must satisfy, in
`tests/research/xsec/test_ensemble.py` — this file also does not exist yet):
(i) `run_ensemble` constructs no `Panel` internally and accepts exactly one
`Panel` instance, passed by reference to every per-`cfg` call — a test
asserts `id(panel)` (or, since `Panel` is a frozen dataclass, an equality
check on `panel.symbols` and `panel.dates`) is byte-identical across all
`len(cfgs)` internal calls to `run`, per Invariant E-align; (ii) each
per-`cfg` weight path individually satisfies §4.3's dollar-neutrality
identity (long leg `+gross/2`, short leg `-gross/2`) at machine-epsilon
tolerance; (iii) the ensembled (averaged) weight path is ALSO exactly
dollar-neutral at the same gross (a convex combination of zero-sum vectors
is zero-sum, §6.2's derivation) — the test must not merely assume this
follows from (ii), it must assert it independently on the averaged output;
(iv) on a synthetic fixture where two `cfgs` are deliberately constructed
to disagree on `panel.symbols` ordering (a corrupted-input regression case),
`run_ensemble` raises `ValueError` rather than silently broadcasting or
mis-summing by position — this is the specific failure mode E8 names as the
reason the driver's build contract requires this test; (v) `run_ensemble(panel,
[cfg])` (a length-1 list) reproduces `run(panel, cfg)` exactly — the
ensemble driver is a strict generalization of the single-config path, never
a parallel reimplementation with its own drift risk.

---

## 14. Non-goals (belong to sibling modules)

- **The rank itself** — momentum/carry/combined signal computation is
  SIGNAL's responsibility (`research/xsec/signals.py`); PORTFOLIO consumes
  `rank_t` as given (P2, Invariant N-port).
- **The point-in-time eligible universe `U_t`** — listing/delisting,
  shortability, and the target universe size `N` are UNIVERSE's
  responsibility (`research/xsec/panel.py`, `research/universe.py`); PORTFOLIO
  consumes `U_t` as given (§10, Invariant U-short).
- **Friction rate in bps** — fee tier, half-spread, and impact estimation
  (including the Amihud vs. Corwin-Schultz choice, §7.4) are COST's
  responsibility; PORTFOLIO only supplies the turnover **quantity** the rate
  is charged against (P7, §7.3).
- **DSR / CPCV / purged walk-forward machinery** — implemented once in
  `convexity.system.validate` and reused, never reimplemented, by
  `research/xsec/{validate,cpcv}.py` (Invariant N-port).
- **Post-construction overlays** (beta-hedging residual market exposure,
  causal drawdown-stops) — `research/xsec/overlay.py`. EVIDENCE (task digest,
  cross-referenced in `docs/specs/REGIME.md` §5.3 (the tested carry instance;
  Sharpe 0.57→0.10, income →0.6%/yr) and §5.5 (beta-hedge, 0.48→0.57) — not in
  `docs/EDGE.md`,
  which only states the qualitative "~70% structural drawdown, income-only"
  verdict for carry) records that overlays do not fix funding carry's
  structural drawdown (beta-hedge lifts Sharpe to only ~0.57 at ~70% DD; a
  drawdown-stop caps DD at ~24% but collapses Sharpe to ~0.10 and funding
  income to ~0.6%/yr) — this is cited
  here only as context for why PORTFOLIO does not attempt to solve carry's
  drawdown problem via its own weighting scheme; overlay policy is a separate
  pillar's decision.
- **Live/paper execution** — the live rebalancer against NautilusTrader is
  explicitly not yet built (`docs/EDGE.md` §(e)); this spec governs the
  research-layer weight construction that a future live strategy slot will
  call.
- **Sizing policy (quarter-Kelly, scale-up-on-confirmation)** — PORTFOLIO
  exposes the gross dial `G` (§2.3, §4.3) that a sizing policy scales; it does
  not itself decide the Kelly fraction or the promotion criteria for scaling
  up (`docs/EDGE.md` §(d)/(e)).

---

## 15. References

- Marchenko, V. A. & Pastur, L. A. (1967), "Distribution of eigenvalues for
  some sets of random matrices," *Mathematics of the USSR-Sbornik* — the
  random-matrix eigenvalue-spreading result that shows a `k×k` sample
  covariance matrix's smallest eigenvalues are severely biased downward when
  `k/n` is not small; the formal justification for restricting §4.2 to a
  diagonal (inverse-vol) estimator rather than a fitted full covariance
  optimizer.
- Ledoit, O. & Wolf, M. (2004), "Honey, I Shrunk the Sample Covariance
  Matrix," *Journal of Portfolio Management* (also *Journal of Multivariate
  Analysis*, 2004) — linear shrinkage of the sample covariance toward a
  structured target; cited in §4.2 both for the eigenvalue-bias
  characterization and as the closed-form (non-fitted-optimizer) alternative
  this repo's doctrine already names (`pyproject.toml` scipy comment).
- Bailey, D. H., Borwein, J. M., Salehipour, A. & López de Prado, M. (2014),
  "Probability of Backtest Overfitting" (published as López de Prado &
  co-authors' PBO work) and López de Prado, M. (2018), *Advances in
  Financial Machine Learning* (combinatorial purged cross-validation,
  purging, embargo) — the CPCV/PBO machinery in `research/xsec/cpcv.py` (§6,
  §9) and the logit-of-relative-rank PBO estimator.
- Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality,"
  *Journal of Portfolio Management* — the PSR/DSR and MinTRL machinery reused
  from `convexity.system.validate.deflated` (§9), including the
  track-length-vs-deflation distinction invoked for the DSR row of §9's
  table.
- Kunsch, H. R. (1989), "The Jackknife and the Bootstrap for General
  Stationary Observations," *Annals of Statistics* — the moving-block
  bootstrap used for the universe-size confidence interval in §10 (block
  bootstrap respects the weekly return series' serial dependence; an i.i.d.
  bootstrap would understate the CI width).
- Politis, D. N. & White, H. (2004), "Automatic Block-Length Selection for
  the Dependent Bootstrap," *Econometric Reviews* — the block-length
  calibration rule for the Kunsch moving-block bootstrap used in §10.
- Page, E. S. (1954), "Continuous Inspection Schemes," *Biometrika* — CUSUM
  sequential change-detection with average-run-length calibration; named in
  §4.2 as the closed-form regime-break-detection extension for the vol
  estimator that is **not** implemented in v1 (a documented possible
  extension, not a current dependency).
- Amihud, Y. (2002), "Illiquidity and Stock Returns: Cross-Section and
  Time-Series Effects," *Journal of Financial Markets* — the illiquidity
  ratio underlying the real (size-aware) impact estimator cited in §7.4 as
  the honest-cost model, in contrast to the rejected Corwin-Schultz
  high/low-range approach.
- Corwin, S. A. & Schultz, P. (2012), "A Simple Way to Estimate Bid-Ask
  Spreads from Daily High and Low Prices," *Journal of Finance* — the
  estimator explicitly **rejected** in §7.4 as a documented dead end for this
  data (conflates volatility with spread on 1-minute-aggregated daily
  high/low, producing a spurious ~135bp / ~40%-drag estimate).
- `docs/EDGE.md` (this repo) — the top-line decision memo recording the
  headline empirical results (net Sharpe, CAGR, maxDD, win-rate, the blended
  PBO point estimate, the qualitative frac_pos-vs-p05 guidance, the carry
  income/drawdown verdict). **Scope note:** `docs/EDGE.md` carries only these
  headline numbers, not the finer-grained per-partition statistics this spec
  also cites (PBO per-partition spread, `frac_pos` numeric range, CPCV
  path-Sharpe-vs-naive haircut, p05 partition-fragility swing, the `N=100`
  shortability/drag breakdown, the overlay collapse figures) — those are
  EVIDENCE from the task's empirical digest and are cross-referenced at their
  point of use in this spec to the sibling spec that also carries them
  (`docs/specs/SIGNAL.md`, `docs/specs/UNIVERSE.md`, `docs/specs/LEVERAGE.md`,
  `docs/specs/REGIME.md`), per the citation convention stated in this
  document's preamble. Every numeric constant in this document is either
  derived above (§2.3, §4.2) or drawn from one of these two, explicitly
  named, sources — never invented.
- `docs/specs/SIGNAL.md` (this repo) — the per-partition PBO spread
  (0.31-0.47, its §4.1) and the PSR point estimate (its §7) cited in this
  spec's §6.1 and §9's table; SIGNAL owns the lookback-family CPCV/PBO
  computation this spec's band-deployment rule (P6, Invariant B-band)
  depends on.
- `docs/specs/UNIVERSE.md` (this repo) — the CPCV path-Sharpe-vs-naive
  figures (its §7), the `frac_pos`/p05 numeric ranges (its §7), and the
  liquidity-cap breadth/capacity figures (its §4) cited in this spec's
  §5.3-5.4, §9, and Invariant R-stat; UNIVERSE owns the universe-size and
  shortability measurements this spec's §10 depends on.
- `docs/specs/LEVERAGE.md` (this repo) — an independent cross-reference for
  the same CPCV path-Sharpe, `frac_pos`, and liquidity-cap breadth/capacity
  figures (its §2.6, §5.2, §7), confirming these are consistent, repo-wide
  cited figures rather than single-spec assertions.
- `docs/specs/REGIME.md` (this repo) — the beta-hedge/drawdown-stop overlay
  collapse figures for funding carry (Sharpe 0.57→0.10, income →0.6%/yr, its
  §5.5/§7/§11) cited in this spec's §14 Non-goals as context for why
  PORTFOLIO does not attempt to fix carry's structural drawdown itself.
- `docs/specs/COST.md` (this repo) — the single cost/funding truth; §7 of
  this spec salvages its round-trip friction decomposition
  (`c_rt = 2*(taker+half_spread+impact)`) and worked 13bp example verbatim,
  reframed from a per-trade to a per-turnover-unit charge; its §5.4.3 is the
  authoritative source for the rejected Corwin-Schultz ~135bp/~40%-drag
  figure cited in this spec's §7.4.
- `docs/specs/SYSTEM.md` (this repo) — the DSR/CPCV centerpiece math (its
  §7) that `research/xsec/{validate,cpcv}.py` reuse without reimplementation
  (Invariant N-port); this spec's §9 table cites SYSTEM's PSR/DSR/MinTRL
  vocabulary directly.
- `docs/ARCHITECTURE.md` (this repo, §1–§2) — "The nine pillars, the weekly
  rebalance, and the code they map to" (§1) and "Weekly data-flow (the
  composition graph)" (§2), added alongside this rewrite, connecting
  PORTFOLIO's directory layout (§12) to the whole-system composition graph.
