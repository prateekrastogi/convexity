# LEVERAGE.md — build spec for portfolio-level gross-exposure sizing (cross-sectional-momentum book)

This file is the authoritative build spec. Build to the contracts and
invariants below. Where this file and your own instinct disagree, this file
wins. Ask before deviating from any **Invariant**.

> **This supersedes the prior directional-breakout LEVERAGE.md.** The old spec
> sized a single trade's stop-to-target risk unit (`min()` of seven per-trade
> ceilings: stop distance, vol target, Kelly-on-a-trade, tail/EVT, portfolio
> MCTR, margin, liquidity). The pivot (see `EMPIRICAL BASELINE` below) retired
> the entire directional-breakout book — gross directional R ≈ 0 across
> timeframes/exits/filters/leverage/three asset classes, even inverted. The
> surviving, live edge is **crypto cross-sectional momentum**: a weekly,
> dollar-neutral, quintile-sorted book with no single-trade stop and no
> single-instrument entry/exit. **There is no "stop distance" and no
> "per-trade R" in this system.** The risk unit this module sizes is
> **next week's portfolio return**, not one trade's outcome. Every
> per-trade ceiling from the old spec (`L_stop`, `L_tail` on a single
> position's MAE, per-trade EVT) is retired; portfolio-level ceilings
> (vol-target, Kelly-on-the-book, capacity/ADV, and a drawdown governor) are
> kept and re-derived on the new risk unit. Where old math is genuinely
> portable (fractional-Kelly growth algebra, the drawdown-governor ladder
> shape, capacity-vs-ADV capping) it is salvaged explicitly below and cited.

This module answers exactly one question, at exactly one cadence:

> Given the book's forecast weekly volatility and a **statistically honest,
> confidence-adjusted** estimate of its edge, what fraction of capital
> (`G_t`, gross exposure as a multiple of equity) should the cross-sectional
> book run into next week's rebalance — and how does that fraction change as
> the live track record accumulates?

LEVERAGE is a **sizer of the whole book**, not of a position. SELECTION/the
signal layer (`research/xsec/signals.py`, `research/xsec/portfolio.py`)
already decides *which* ~50 names are long/short and in what *relative*
proportion (quintile membership, inverse-realized-vol weighting within each
leg). LEVERAGE decides the **one scalar** that scales that entire relative
book up or down: `G_t`. It never touches which names are in, never touches
relative within-book weights, and never predicts direction.

**Explicit hand-off from PORTFOLIO.md.** `docs/specs/PORTFOLIO.md` (P3, §2.3,
§4.3) defines the reference gross `G` at which its own dollar-neutral,
inverse-vol weight vector `w_t` is constructed (`sum_i |w_{i,t}| = G`,
reference `G = 1.0`) and explicitly declines to decide *what* `G` should be
at deploy time, naming this LEVERAGE's job instead (PORTFOLIO.md §14: "PORTFOLIO
exposes the gross dial `G`... a sizing policy scales; it does not itself
decide the Kelly fraction"). **LEVERAGE.md's `G_t` IS that same symbol,
time-indexed and code-realized**: at each Friday-close rebalance, LEVERAGE
computes `G_t` (Section 6.1) and the live channel scales PORTFOLIO's unit-gross
weights by it, `w_{i,t}^{live} = G_t · w_{i,t}` (equivalently, PORTFOLIO's own
`gross=` parameter to `quantile_weights`/`liquidity_weights` is called with
`gross=G_t`, not the reference `gross=1.0`, per §5.2's discussion of the
one-week-lagged capacity-trigger exception). This is the SAME channel
`docs/specs/REGIME.md` (§0) composes its own multiplier onto
(`gross_live = g_t · G_t`) — REGIME, LEVERAGE, and PORTFOLIO share one gross
symbol at one call site, never three independent ones.

Scope: Binance USDT-M perpetual futures, weekly (Friday-close) rebalanced,
dollar-neutral cross-sectional momentum, point-in-time universe of ~50 liquid
USDT-M perps, retail book. Compute budget: a 2-core / 11 GB ARM VM; this
module runs once per week, so there is no latency constraint worth
engineering for.

---

## 0. Governing principles (read before anything else)

**P1 — The risk unit is the NEXT WEEKLY BOOK RETURN, not a trade's R-multiple.**
Every formula in this file is stated on the book's per-week net-of-cost return
series `R_t` (one observation per Friday-close rebalance), never on a single
instrument's stop-to-target distance. There is no `d` (stop distance in this
system) to divide by. A ceiling here bounds `G_t`, the scalar multiplier
applied uniformly to the dollar-neutral book that `research/xsec/portfolio.py`
already built.

**P2 — Kelly sizing is on the BOOK's return distribution, using a
CONFIDENCE-ADJUSTED edge, never the raw point-estimate Sharpe.** Plug-in Kelly
computed from a backtest's raw sample mean overstates the safe fraction under
parameter uncertainty (Section 2.3). This system's own DSR fails on track-record
length (PSR≈0.994 is high, but `n_obs=258` weekly observations over ~5yr is
short relative to the multi-testing-inflated MinTRL — a track-length problem,
not a deflation rejection; **SYSTEM §5.3–§5.5**, gate branch 2b of **SYSTEM
§6.1**). LEVERAGE therefore sizes off a **lower-confidence-bound Sharpe pulled
from SYSTEM's own DSR/bootstrap machinery** (`convexity.system.validate
.deflated.DeflatedSharpe` for the pass/fail gate; `convexity.system.validate
.robustness.block_bootstrap_ci` for the CI itself — Section 2.6 pins the exact
call path), never the raw backtest Sharpe. (Invariant K-LCB.)

**P3 — Vol targeting and Kelly are two views of the SAME closed-form
optimizer, never a fitted or learned combiner.** `G_t = σ_target / σ̂_t` (vol
targeting) and `f* = μ/σ²` (Kelly) are both special cases of the same
mean-variance-optimal-growth first-order condition (Section 2.1 derives this).
LEVERAGE runs the fractional-Kelly form as the PRIMARY sizing rule and reports
vol-targeting as a diagnostic re-expression of the same number — never two
independently-tuned dials that could disagree. (Invariant SIZE-ONE.)

**P4 — No `Σ⁻¹μ`, no fitted mean-variance optimizer over the cross-section.**
This module sizes ONE scalar (`G_t`) applied uniformly to an already-built
relative book. It never inverts a covariance matrix to reweight *within* the
book — that risk-weighting is SELECTION's inverse-realized-vol job
(`research/xsec/portfolio.py::quantile_weights`), a closed-form, diagonal,
per-name computation, not a joint fitted optimization. Justification: with
`N≈50` names and `n_obs≈258` weekly observations, the sample cross-sectional
covariance matrix has an eigenvalue spectrum spread governed by the
Marchenko–Pastur ratio `N/n_obs ≈ 0.19` — not degenerate, but comfortably
inside the regime where the smallest sample eigenvalues are pure noise
(Marchenko & Pastur, 1967; Laloux et al., 1999) and a fitted `Σ⁻¹μ`
allocator would be dominated by estimation error on the very eigenvectors a
mean-variance optimizer weights most. LEVERAGE's covariance use (Section 4.3)
is restricted to a **shrunk, diagonal-anchored, closed-form** volatility
estimate of the ONE aggregate book return series — never an inverted
cross-sectional matrix. (Invariant NO-MV.)

**P5 — NO ML.** Every estimator in this module is closed-form classical
statistics: EWMA, method-of-moments, a block bootstrap for confidence
intervals (Kunsch 1989; Politis & White 2004 for block length), a CUSUM
change-detector for the drawdown/ratchet gate (Page 1954), and Ledoit-Wolf
shrinkage if a covariance term is ever needed (Ledoit & Wolf, 2004). No neural
net, no gradient-boosted tree, no learned combiner, no grid search retrained
on live P&L, no continuously-refit hidden-Markov regime model. This mirrors
the repo's `pyproject.toml` doctrine ("scipy: closed-form (NOT ML)").

**P6 — Capacity is a hard ceiling, never a soft preference.** `G_t` is capped
so that no single position implied by the scaled book exceeds
`κ · ADV_i` for that instrument (Section 5). This is a per-instrument liquidity
FACT, not a forecast, and binds via `min()`, never a weighted blend.

**P7 — The ratchet from quarter-Kelly toward half-Kelly is a SEQUENTIAL
STATISTICAL TEST on live data, never a vibes-based judgment call.** Scaling up
the Kelly fraction `c` requires a stated, pre-registered, falsifiable
criterion evaluated on live (not backtest) weekly returns (Section 6).

**P8 — Provider of ONE scalar, never a decider on which names trade.**
LEVERAGE emits `G_t` (and its full ceiling decomposition + binding reason).
It does not choose instruments, does not choose sides, does not choose
rebalance timing. Those are SELECTION's/the signal layer's job. (Invariant
PROV, inherited naming convention from COST P2/SYSTEM P1.)

Forbidden in v1: any ML/learned sizing model (P5); sizing off the raw
backtest point-estimate Sharpe instead of a DSR/bootstrap lower confidence
bound (P2, Invariant K-LCB); inverting a fitted cross-sectional covariance
matrix to reweight within the book (P4, Invariant NO-MV); a second,
independently-tuned vol-target dial that can disagree with the Kelly fraction
(P3, Invariant SIZE-ONE); raising `c` toward half-Kelly on a discretionary call
rather than the sequential test of Section 6 (P7); ignoring the capacity
ceiling above a stated book size (P6); per-symbol tuning of any RED constant
in this file on realized P&L (Section 3).

### 0.1 Consolidated notation table

Every symbol used below, collected in one place (each is re-defined at its
first point of use in prose; this table is a lookup, not a substitute for
those definitions). Week index `t` runs over Friday-close rebalances, the
same clock SYSTEM/SELECTION use (`research/xsec/engine.py`).

| Symbol | Meaning | Bucket | Owning section |
|---|---|---|---|
| `F_t` | Information set at rebalance `t` (no lookahead) | — | 2.1 |
| `R_{t+1}` | Book's 100%-gross reference weekly net return, week `t→t+1` | data | 2.1 |
| `μ, σ, σ²` | True (unknown) weekly mean/vol/variance of `R_{t+1}` | — | 2.1 |
| `μ̂, σ̂_t` | Estimators of `μ, σ` — `σ̂_t` is the causal EWMA (Section 4) | TEAL | 2.1, 4.1 |
| `f, f*` | Leverage fraction; Kelly-optimal `f* = μ/σ²` | — | 2.1–2.2 |
| `g(f)` | Expected log-growth rate at leverage `f` | — | 2.2 |
| `SR ≡ μ/σ` | Per-observation (weekly) Sharpe ratio | — | 2.2 |
| `c, c_t` | Fractional-Kelly dial, `c ∈ {0.25, 0.50}` | RED | 2.3, 6.2 |
| `SR_LCB_t` | Deflated, lower-confidence-bound per-observation Sharpe | derived (TEAL-sourced) | 2.5–2.6 |
| `G_t` | Gross exposure multiplier — THE output of this module | — | 0, 6.1 |
| `G_t^{cap}` | Capacity ceiling on `G_t` | derived | 5.1 |
| `G_max` | Absolute circuit-breaker ceiling on `G_t` | RED | 3, 6.1 |
| `κ` | Capacity tightness: max fraction of `ADV_i` a position may occupy | RED | 5.1 |
| `C` | Liquidity-cap multiple (capacity dial, off by default) | RED | 5.2 |
| `ADV_i`, `w_i^{ref}` | Instrument `i`'s trailing dollar-volume; its reference-gross (100%) dollar weight from `research/xsec/portfolio.py`'s book | TEAL, derived | 5.1 |
| `h_vol`, `n_vol_min` | EWMA half-life (weeks) and event-count floor for `σ̂_t` | RED | 3, 4.1–4.2 |
| `L_boot`, CI level | Block-bootstrap block length and confidence level for `SR_LCB_t` | RED (shared w/ SYSTEM) | 2.6, 3 |
| `dd_t`, `m_t` | Running peak-to-trough book-equity drawdown; governor ladder multiplier | derived, RED-ladder | 6.4 |
| `k_ref`, `h_promote`, `ARL_0` | CUSUM reference offset, decision threshold, target average run length | RED | 6.3 |
| `N_min_weeks` | Minimum elapsed live-time floor before promotion is even eligible | RED | 6.3 |
| `S_k`, `z_k` | CUSUM statistic and standardized weekly Sharpe surprise | derived | 6.3 |
| `DSR`, `PSR`, `MinTRL` | Deflated/Probabilistic Sharpe, minimum track-record length (SYSTEM §5.3–§5.5) | SYSTEM-owned | 2.5 |
| `frac_pos`, `p05` | Fraction of CPCV paths with positive OOS Sharpe; 5th-pct path Sharpe (SYSTEM §4.3) | SYSTEM-owned | 2.5 |
| `PBO` | Probability of backtest overfitting (SYSTEM §4.4) | SYSTEM-owned | 11 |
| `n_obs` | Count of non-overlapping weekly observations | data | 2.5 |

---

## 1. What changed vs. the old (per-trade) LEVERAGE.md — mapping table

| Old concept (per-trade stop-to-target) | New concept (portfolio-level, this file) |
|---|---|
| Risk unit = one trade's R-multiple (stop distance `d`) | Risk unit = next week's book return `R_t` |
| `L_stop = (c − max(carry,0)) / d` | retired — no per-trade stop; carry is netted directly into the weekly return series by `research/xsec/accounting.py::settle_week` (signed funding cash flow), not clamped into a sizing ceiling |
| `L_kelly` = Kelly on one trade's win/loss distribution | `f*` = Kelly on the BOOK's weekly return distribution (Section 2) |
| `L_tail` = per-position EVT on MAE | retired — no single position's adverse excursion; book-level tail risk shows up in the realized weekly-return distribution's skew/kurtosis, already consumed by SYSTEM's DSR (γ3, γ4) |
| `L_portfolio` (MCTR ceiling across open positions) | retired as a *ceiling*; MCTR-style risk decomposition still exists but lives in SELECTION's within-book weighting (`research/xsec/portfolio.py`), not as a LEVERAGE min()-term |
| `L_margin`, `L_liquidity` (per-position) | capacity ceiling `κ·ADV_i` (Section 5), evaluated on the SCALED book's per-name notional, same spirit, new unit |
| Drawdown governor (ladder on equity peak-to-trough) | KEPT, re-derived on book equity (Section 6.4) — same shape, same citation lineage, new trigger reference (book-level DD, not per-trade) |
| `min()` of seven ceilings | `min()` of THREE ceilings: Kelly/vol-target sizing (Section 2), capacity (Section 5), drawdown governor (Section 6.4) — fewer terms because the per-trade machinery (stop, per-trade tail, per-trade margin) has no referent in a weekly rebalanced book |

The reduction from seven ceilings to three is not a simplification for its
own sake — it is a direct consequence of P1: most of the old ceilings were
answers to "how big can *this trade's stop* be," and there is no trade stop
in a weekly cross-sectional rebalance.

---

## 2. Core sizing math: vol-targeting is a special case of fractional Kelly

### 2.1 Setup and notation

Let `F_t` be the information set available at the Friday-close rebalance
decision `t` (all quantities below are `F_t`-measurable; no future weekly
return enters `G_t`). Let `R_{t+1}` be the **book's dollar-neutral, 100%-gross
reference weekly net return** for the week starting at rebalance `t` — i.e.
the `net` series `research/xsec/engine.py::XSecResult.net` (price P&L + signed
funding − turnover cost, as decomposed in `research/xsec/accounting.py`) at
`gross=1.0`. Because the book is dollar-neutral (`research/xsec/portfolio.py`
long leg `+gross/2`, short leg `−gross/2`), Sharpe is leverage-invariant:
scaling gross by `G` scales every weekly return by `G`, so `Sharpe(G·R) =
Sharpe(R)` for `G>0`, while CAGR and maxDD scale with `G` (documented in
`portfolio.py`'s module docstring). This invariance is what makes it
meaningful to fit `μ, σ` once at the reference gross and then choose `G_t`
as a pure scalar multiplier — the shape of the return distribution does not
change with `G_t`, only its scale.

Let `μ = E[R_{t+1} | F_t]` and `σ² = Var[R_{t+1} | F_t]` be the (unknown,
estimated) weekly mean and variance of the reference book return. Leverage
`f` (a fraction of equity staked per unit of the reference book, i.e.
`f ≡ G_t`) turns the reference return into the levered weekly return
`f·R_{t+1}`, and equity compounds multiplicatively: `E_{t+1} = E_t·(1 + f·R_{t+1})`.

**Stationarity assumption (regularity condition, stated explicitly).** The
derivation in 2.2–2.3 treats `μ, σ²` as constants estimated from a rolling
history and assumes the estimation-window return distribution is
approximately stationary (its first two moments do not drift materially
within the window used to fit `μ̂, σ̂_t`). This is a real assumption, not a
formality: crypto cross-sectional momentum returns are NOT stationary over
multi-year horizons (regime shifts are the documented reason `frac_pos` and
`p05` diverge across CPCV partitions, SYSTEM §4.3). LEVERAGE does not assume
away non-stationarity — it manages the consequence with THREE distinct,
already-specified mechanisms rather than a stronger IID assumption: (i) the
EWMA in Section 4 is deliberately RECENCY-WEIGHTED (a short `h_vol` window
means `σ̂_t` tracks a regime change within `~h_vol` weeks rather than
averaging over the whole non-stationary history); (ii) `SR_LCB_t` (Section
2.5–2.6) is a LOWER bound sourced from an out-of-sample path distribution,
which is already penalized for the same regime variability that would
violate stationarity; and (iii) the drawdown governor (Section 6.4) and CUSUM
demotion (Section 6.2) are explicit, causal, non-stationarity circuit
breakers that act on live data if the estimated moments turn out to have
been wrong going forward. The Taylor/Kelly algebra itself is exact only
"locally in time" (over one estimation window) — global non-stationarity is
handled by re-estimating every week, not by a stronger assumption on `R`.

### 2.2 Worked derivation: Kelly-optimal `f*` and the growth rate `g(f)`

**The Kelly criterion.** The Kelly-optimal fraction maximizes the expected
long-run compound growth rate of equity, `E[log(1 + f·R)]` (Kelly, 1956;
Thorp, 2006). Expand `log(1+x)` in a second-order Taylor series around `x=0`
(valid for `|f·R|` small relative to 1, the standard approximation for
continuous/near-Gaussian return streams — Thorp 2006 §3):

```
log(1 + f·R)  ≈  f·R − (1/2)·f²·R²  + O(f³R³)
```

Take the expectation over the (unknown, and — per the stationarity
regularity condition stated in Section 2.1 — assumed approximately constant
over the estimation window) distribution of `R`:

```
g(f) ≡ E[log(1 + f·R)]  ≈  f·μ − (1/2)·f²·(σ² + μ²)  ≈  f·μ − (1/2)·f²·σ²
```

The last step drops `μ²` relative to `σ²` — for a weekly return stream with
`μ` on the order of tens of bp and `σ` on the order of a few percent (this
book's realized weekly vol is consistent with **~28% annualized vol ⇒
σ_weekly ≈ 28%/√52.18 ≈ 3.9%**, see Section 3), `μ² ≪ σ²` by roughly two
orders of magnitude, so the approximation is excellent here specifically
(this is stated as an explicit regularity condition, not asserted for all
return streams — see Section 8 edge cases). This is the standard **Kelly
second-order / Gaussian approximation** (Thorp, 2006; MacLean, Thorp & Ziemba,
2011).

**First-order condition.** Maximize `g(f) = f·μ − (1/2)·f²·σ²` over `f`:

```
dg/df = μ − f·σ²  =  0     ⟹     f* = μ / σ²
```

This is the closed-form Kelly-optimal leverage — identical in form to the
classical Kelly fraction `edge/odds` and to the unconstrained mean-variance
tangency weight on a single risky asset. Substituting back:

```
g(f*) = f*·μ − (1/2)·(f*)²·σ²
      = (μ/σ²)·μ − (1/2)·(μ/σ²)²·σ²
      = μ²/σ² − (1/2)·μ²/σ²
      = μ²/(2σ²)
```

So the maximal achievable (second-order-approximate) growth rate is
`g(f*) = μ²/(2σ²) = SR²/2` where `SR ≡ μ/σ` is the per-observation Sharpe
ratio. This is the well-known result that full-Kelly growth is proportional
to the SQUARE of the Sharpe ratio — a critical fact for Section 2.3, because
it means growth is much more sensitive to `μ`-estimation error than a naive
linear intuition would suggest.

### 2.3 Fractional Kelly: the relative-growth-capture identity

Full Kelly `f*` is well known to be dangerously aggressive under realistic
parameter uncertainty and fat tails (MacLean, Thorp & Ziemba, 2011) — the
standard mitigation is **fractional Kelly**, sizing at `f = c·f*` for some
`c ∈ (0,1)`. Derive the growth captured at fraction `c`:

```
g(c·f*) = (c·f*)·μ − (1/2)·(c·f*)²·σ²
        = c·f*·μ − (1/2)·c²·(f*)²·σ²
```

Substitute `f* = μ/σ²`:

```
g(c·f*) = c·(μ²/σ²) − (1/2)·c²·(μ²/σ²)
        = (μ²/σ²)·[c − (1/2)·c²]
        = (μ²/σ²)·(1/2)·[2c − c²]
        = g(f*)·c·(2 − c)
```

using `g(f*) = μ²/(2σ²)` from 2.2. So:

```
g(c·f*) / g(f*)  =  c·(2 − c)
```

**This is the exact closed-form relative-growth-capture identity** (not an
approximation beyond the second-order Taylor step already taken in 2.2) —
it holds for ANY `c`, derived purely from the quadratic form of `g(f)`, and
is the standard fractional-Kelly result reproduced here from first principles
rather than asserted (cf. Thorp, 2006 §12; MacLean, Thorp & Ziemba, 2011).
Evaluate at the two conventional operating points:

```
c = 0.25 (quarter-Kelly):  c·(2−c) = 0.25 · 1.75 = 0.4375   →  ~44% of full-Kelly growth
c = 0.50 (half-Kelly):     c·(2−c) = 0.50 · 1.50 = 0.75     →  75% of full-Kelly growth
```

**Why quarter-to-half is the standard conservative range, precisely.** The
growth-capture curve `c(2−c)` is concave and steep near `c=0`: quarter-Kelly
already captures 44% of the growth for a QUARTER of the stake, while full
Kelly's last 25% of stake (`c: 0.75→1.0`) buys only the last
`1·(2−1) − 0.75·(2−0.75) = 1.0 − 0.9375 = 6.25` percentage points of growth
capture. Meanwhile, realized variance of terminal wealth and the depth of
drawdown scale MUCH worse than linearly in `c` — for the Gaussian/continuous-
time Kelly analogue, the probability of a drawdown of depth `d` before
reaching a new equity high scales as `d^{(2/c − 1)}` in the diffusion limit
(a standard result in the fractional-Kelly literature, e.g. MacLean, Thorp &
Ziemba 2011 ch. on drawdown), so DOUBLING `c` from 0.25 to 0.5 roughly
squares the tail-drawdown-probability exponent's sensitivity, while only
buying `0.75/0.4375 ≈ 1.71×` the growth. This asymmetry (large growth capture
at small `c`, disproportionate tail-risk growth at large `c`) is exactly why
practitioners standardize on quarter-to-half Kelly (Thorp, 2006, "The Kelly
Criterion in Blackjack, Sports Betting, and the Stock Market"; MacLean, Thorp
& Ziemba, 2011). LEVERAGE adopts this range as RED policy (Section 3),
grounded in this derivation, not asserted by convention alone.

### 2.4 Vol-targeting as the SAME rule with `σ` in place of `σ²`-scaled `μ`

Vol targeting sets `G_t = σ_target / σ̂_t` directly. This is not a second,
independent sizing philosophy — substitute `f* = μ/σ²` into vol-target form:
if the operator picks `σ_target = c·f*·σ̂_t = c·(μ/σ̂_t)`, the two rules
coincide exactly at `G_t = c·f*`. LEVERAGE therefore does **not** carry two
independently-calibrated dials (a `σ_target` chosen by feel AND a `c` chosen
by feel) — it carries exactly one degree of freedom, `c` (Section 3), and
reports the implied `σ_target = c·μ̂/σ̂_t · σ̂_t = c·μ̂` purely as a diagnostic
re-expression, satisfying P3/Invariant SIZE-ONE.

### 2.5 The Kelly-under-parameter-uncertainty correction (the load-bearing fix)

Section 2.2–2.4 treats `μ` and `σ` as known. They are not: `μ̂` is a sample
mean over `n_obs` weekly observations and carries real estimation error — this
system's own Sharpe estimate carries a wide block-bootstrap CI (**SYSTEM
§7.4/Invariant BLOCK-LEN** for the block-length derivation, **§8.1** for the
robustness suite that runs it) and the DSR gate is known to fail on
track-record length even with a high point-estimate PSR (**SYSTEM
§5.3–§5.5**; **PSR≈0.994** cited in the EMPIRICAL BASELINE, yet `n_obs≈258`
weekly observations sits below the multi-testing-inflated `MinTRL` — SYSTEM's
own worked example at §5.5 computes `MinTRL ≈ 824` weeks against `n_obs=258`,
i.e. `n_obs < MinTRL` by roughly 3×). Plug-in Kelly computed from `μ̂` alone
is **provably biased toward over-betting** under parameter uncertainty:
because `g(f)` in 2.2 is concave in `f` with maximum at `f*=μ/σ²`, and the
estimator `μ̂` has sampling variance `Var[μ̂] = σ²/n_obs`, a second-order
Taylor expansion of `g` around the true `f*` makes this precise rather than
gestural. Write `f̂* = μ̂/σ²` (treating `σ²` as known for this step, since
`Var[μ̂] ≫` the vol-estimation error at this order) and expand `g` around
`f*`:

```
g(f̂*)  =  g(f* + (f̂*−f*))
       ≈  g(f*) + g'(f*)·(f̂*−f*) + (1/2)·g''(f*)·(f̂*−f*)²
```

`g'(f*) = 0` by the first-order condition (Section 2.2), and `g''(f) = −σ²`
(the second derivative of the quadratic `g(f) = f·μ − (1/2)f²σ²`), so:

```
E[g(f̂*)]  ≈  g(f*) − (1/2)·σ²·E[(f̂*−f*)²]  =  g(f*) − (1/2)·σ²·Var[f̂*]
```

since `f̂*−f* = (μ̂−μ)/σ²` is unbiased for `f*`, so `E[(f̂*−f*)²] = Var[f̂*] =
Var[μ̂]/σ⁴ = 1/(n_obs·σ²)`. Substituting:

```
E[g(f̂*)]  ≈  g(f*) − 1/(2·n_obs)
```

**This is the closed-form bias bound**: plug-in Kelly's REALIZED expected
growth is *lower* than the growth the point estimate `f̂*` promises, by a
term of order `1/n_obs`, purely because `g` is concave (Jensen's inequality
applied to the concave `g` around the noisy `f̂*` gives exactly this sign —
`E[g(f̂*)] ≤ g(E[f̂*]) = g(f*)`). At `n_obs=258`, this bias term is
`1/(2·258) ≈ 0.00194` in log-growth units per week — small in isolation, but
it is a STRICTLY NEGATIVE correction on top of a `g(f*) = SR²/2` that is
itself already small (Section 2.2), and — more importantly for sizing policy
— the bias is only the EXPECTED shortfall; the same concavity means the
*variance* of realized growth under plug-in sizing is asymmetric, with
downside adverse draws (`f̂*` too large because `μ̂` overshot) costing more
curvature than symmetric upside draws (`f̂*` too small) gain. This is the
textbook Kelly-shrinkage result (Chopra & Ziemba, 1993, on the sensitivity
of mean-variance-optimal weights to estimation error in `μ` vs. `σ`; Baker &
McHale, 2013, "Optimal Betting Under Parameter Uncertainty," for the
specific Kelly-fraction shrinkage result under an estimated edge) — sizing
off the point estimate `μ̂/σ²` is not merely noisy, it is a biased-high
estimate of the safe fraction, and the correction below is the standard
prescription: replace the point estimate with a defensible lower bound.

**The correction: replace `μ̂/σ̂` with a lower-confidence-bound Sharpe.**
LEVERAGE's sizing rule is:

```
G_t  =  c · (SR_LCB_t / σ̂_t)
```

where:
- `c ∈ [0.25, 0.50]` is the fractional-Kelly dial of Section 2.3 (RED,
  ratcheted per Section 6);
- `σ̂_t` is the causal EWMA-estimated weekly realized book volatility
  (Section 4);
- `SR_LCB_t` is a **deflated, lower-confidence-bound per-observation Sharpe**,
  NOT the raw sample Sharpe `μ̂_t/σ̂_t`. Section 2.6 pins its exact
  construction, gated so that `SR_LCB_t = 0` (i.e. `G_t` floors at the
  drawdown-governor's minimum multiplier, never negative or undefined)
  whenever the DSR gate is failing (`DeflatedSharpe.deflate(...).passed ==
  False`, SYSTEM §5.3–§5.5 / §6.1 Gate 2). This is the Kelly-under-
  estimation-risk correction: **`f* = μ/σ²` is only ever evaluated at the
  point estimate in Section 2.2's textbook derivation; the deployable rule
  replaces the point-estimate numerator with a statistically defensible
  LOWER bound on it**, which the bias derivation above shows is directionally
  necessary (plug-in Kelly overshoots in expectation), not merely
  conservative by convention.

> **Invariant K-LCB — Kelly is never sized off the raw point-estimate
> Sharpe.** `SR_LCB_t` MUST be sourced from a confidence-interval or
> deflated-Sharpe object (SYSTEM's `DeflatedSharpe`/bootstrap CI output),
> never from a bare `mean(R)/std(R)` computed inside this module. A test
> asserts that feeding LEVERAGE's sizing function the raw sample Sharpe
> directly (bypassing the CI/DSR object) raises, and that widening the
> bootstrap CI (more estimation uncertainty) can only LOWER `G_t`, never
> raise it, holding `σ̂_t` fixed.

**Numeric grounding (cited, not invented).** Using the EMPIRICAL BASELINE's
own numbers: honest net Sharpe (real Amihud-impact cost) is **~0.8–1.0 per
lookback**, CPCV mean OOS path Sharpe **≈0.91** vs naive single-path **0.98**
(haircut only **≈0.07** — no OOS collapse), and `frac_pos` (fraction of CPCV
paths with positive Sharpe) is **0.85–1.00**, the ROBUST statistic, while the
5th-percentile path Sharpe is **partition-fragile** (swinging **+0.42 to
−0.59** depending on `(n_groups, k_test)` — SYSTEM §4.3's Bernoulli-vs-order-
statistic argument for why). This is also why the lookback band itself is
deployed as an ensemble rather than the single fitted peak: SYSTEM §4.4's
**PBO = 0.357** (range **[0.31, 0.47]** under alternative partitionings),
with `L=12` winning in-sample on **21 of 28** CPCV splits, is direct evidence
that the single-`L` peak is partly a partition-dependent artifact of the
search, not a cleanly-validated point estimate — exactly the reason `μ̂` for
THIS book should not be trusted as a sharp point value even before the
generic Kelly-shrinkage argument above is invoked. LEVERAGE's `SR_LCB_t`
(Section 2.6) is built from the block-bootstrap CI on the deployed BAND's
realized path returns — NOT the raw 0.98 naive figure and NOT a
single-lookback point estimate — consistent with the EVIDENCE digest's
explicit finding (cross-referenced in `docs/specs/SIGNAL.md` §4.1 and
`docs/specs/UNIVERSE.md` §7) that promotion criteria must weight `frac_pos` over the
partition-fragile `p05`, and consistent with SYSTEM's own Gate 4
deployment-mode consequence (PBO in `[0.2, 0.5)` FORCES band, not point,
deployment — SYSTEM §4.4/§6.1).

### 2.6 `SR_LCB_t` construction (closed-form, no new estimator invented)

**This is the exact, verified call path — the single most load-bearing
pointer in this file, because the entire Kelly sizing rule (`G_t^{Kelly}`)
is a function of `SR_LCB_t`.** `SR_LCB_t` is NOT read off
`PathSharpeSummary` (`research/xsec/cpcv.py`) — `PathSharpeSummary` has no
bootstrap-CI field (its fields are `n_paths, mean, median, p05, minimum,
frac_pos, frac_gt_half, naive_full_sharpe, haircut`; confirmed against the
dataclass directly). The actual confidence interval is a SEPARATE
computation, using a DIFFERENT, already-existing function:

```
SR_LCB_t  =  0                                          if DeflatedSharpe.deflate(...).passed is False
          =  max( 0 ,  ci_low )                         otherwise

where (ci_low, ci_high) = convexity.system.validate.robustness.block_bootstrap_ci(
                              path_returns  = <promoted config's per-week net-of-cost return series>,
                              block_length  = defaults.bootstrap_block_length,   # RED, shared w/ SYSTEM
                              n_boot        = defaults.bootstrap_n,              # RED, shared w/ SYSTEM
                              ci_level      = defaults.bootstrap_ci_level,       # RED, shared w/ SYSTEM
                              seed          = <fixed, reproducible>,
                              statistic     = <per-observation Sharpe>,
                          )
```

Concretely, `block_bootstrap_ci` is the real function defined in
`src/convexity/system/validate/robustness.py` (NOT a `stats.py` — no such
module exists under `src/convexity/system/validate/`; a same-named
`stats.py` exists only under the unrelated `src/convexity/cost/validate/`
pillar and must not be confused with this one). It is already wired into
production via `RobustnessSuite.run(...)` (same file), which returns a
`RobustnessReport` whose `.sharpe_ci: tuple[float, float]` field is exactly
`(ci_low, ci_high)`; LEVERAGE calls `block_bootstrap_ci` directly (or reuses
`RobustnessSuite.run(...).sharpe_ci`, either is acceptable since they are
the same code path) on the SAME `path_returns` series `SYSTEM
§5.2`/`research/xsec/validate.py::deflated_sharpe` already uses as its
`net_weekly` input — i.e. `SR_LCB_t` and the DSR gate are computed from
IDENTICAL input data, just two different statistics of it (a bootstrap CI
lower bound vs. a deflated point test), so there is no risk of the two
numbers silently referencing different underlying series.

`block_bootstrap_ci`'s default `statistic` callback is a per-observation
Sharpe (`mean/std`, `ddof=1`) — this already matches the per-observation
convention `SR_LCB_t` needs (no extra annualization step; `SR_LCB_t` is
per-observation exactly like `σ̂_t` is a per-observation/weekly vol, so their
ratio `SR_LCB_t/σ̂_t` is dimensionally a fraction-of-equity leverage number
with no annualization mismatch — the same `Invariant DSR-ANN` discipline
SYSTEM enforces for the DSR gate, Section 5.2, applies here by construction
because both quantities are computed on the same non-annualized weekly
series).

The gate itself reads `DeflatedSharpe.deflate(...).passed` — the boolean
field on `DSRResult` (`src/convexity/system/types.py`), which is `False`
exactly when Gate 2 of SYSTEM §6.1 fails (either branch 2a — clean
`DSR ≥ 0.95` — or branch 2b — `PSR(0) ≥ 0.95` AND the shortfall is diagnosed
as `insufficient_track_record`, i.e. `n_obs < MinTRL`). Full formula:

```
SR_LCB_t  =  0                                         if DSR(config) fails the SYSTEM §6.1 Gate-2 conjunction
          =  max( 0 ,  ci_low )                        otherwise
```

The `max(0, ·)` floor is P6-style conservatism (never a negative or NaN
sizing multiplier): a book whose lower confidence bound on edge is at or
below zero sizes to `G_t = 0` via this term alone (the drawdown governor and
capacity ceiling still apply as a `min()`, but they cannot ever exceed the
sizing-implied zero once the CI floor bites).

> **Invariant K-CI — the CI edge is read, never recomputed ad hoc.**
> `SR_LCB_t` is a pass-through of `convexity.system.validate.robustness
> .block_bootstrap_ci` (directly, or via `RobustnessSuite.run(...)
> .sharpe_ci[0]`) applied to the LIVE promoted config's weekly return series,
> not a second, independently-implemented confidence interval inside
> `leverage/`. A test asserts LEVERAGE's `SR_LCB_t` for a fixture config
> equals `block_bootstrap_ci`'s lower bound on the SAME fixture series
> bit-for-bit at a fixed seed (no re-derivation drift between the two
> modules), and separately asserts the DSR gate check calls
> `DeflatedSharpe.deflate(...).passed` rather than re-implementing any part
> of the SYSTEM §5.3–§5.5 test.

---

## 3. Parameter governance

Every parameter is exactly one bucket (same four-bucket discipline as
COST/SYSTEM).

- **TEAL — fitted per-book from its own past data, at runtime, CAUSAL.** The
  EWMA-estimated weekly book volatility `σ̂_t` (Section 4); the realized
  weekly-return skew/kurtosis feeding the same DSR object LEVERAGE reads
  `SR_LCB_t` from (owned by SYSTEM, consumed here); the trailing dollar-volume
  `ADV_i` per instrument used by the capacity ceiling (Section 5).
- **AMBER — supplied market metadata.** Per-instrument tick/lot/contract
  multiplier; the venue's minimum notional; which instruments are
  `shortable` (binding on the capacity ceiling per instrument — a name that
  cannot be shorted contributes 0 to the short leg regardless of its ADV,
  consistent with the EMPIRICAL BASELINE's ~15-of-50-added-names
  unshortable finding at N=100).
- **RED — shared economic/modelling constants, unreachable from
  per-instrument config (Invariant C, inherited).** Every constant below has
  an illustrative v1 value and an explicit rationale — none is a bare
  bucket label (contrast with the gap this closes: earlier drafts left these
  as symbols with no anchor, unlike COST.md's concrete `fee=5bp` style).
  - The fractional-Kelly dial `c ∈ {0.25, 0.50}` (Section 2.3's derivation —
    `c(2−c)` growth-capture identity — is the derivation for these two exact
    values, not a convention; no other `c` grid point is defined in v1);
  - the EWMA half-life `h_vol = 8` weeks (in WEEKS — the book's own decision
    clock, analogous to COST's funding-stamp-clock discipline). Rationale:
    the CPCV embargo (SYSTEM §4.2) is 1 week and the CPCV group size at
    `n_groups=8` over `n_obs≈258` weeks is `≈32` weeks/group — `h_vol=8`
    weeks (`ESS=(2−α)/α≈15` observations at this half-life) is short enough
    to track a within-CPCV-group regime shift (Section 2.1's non-stationarity
    discussion) yet long enough that `ESS` exceeds `n_vol_min` well inside
    one CPCV group, so the vol estimate does not itself become the
    bottleneck on usable history;
  - the vol-EWMA event-count floor `n_vol_min = 16` weeks (`= 2·h_vol`,
    Section 4.2's own stated rule `n_vol_min ≥ 2·h_vol`, so this is DERIVED
    from `h_vol` once `h_vol` is fixed, not an independent free choice);
  - the bootstrap block length `L_boot = 6` weeks and CI level `= 0.90`
    (one-sided lower bound). `L_boot` is NOT independently chosen — it is
    SHARED with, and takes its value from, SYSTEM §7.3–§7.4's derivation
    (`b* ~ c·n^{1/3}`, Politis & White 2004, reconciled at `n≈258` to
    `b*≈6.36≈6` weeks) — Invariant K-CI forbids LEVERAGE from picking its
    own block length independently of that derivation. The CI level `0.90`
    (rather than SYSTEM's `0.95` used for its two-sided robustness-suite CI)
    is a ONE-SIDED lower-confidence-bound convention appropriate to a
    sizing floor (only the lower tail matters for `SR_LCB_t`; a one-sided
    90% lower bound corresponds to the same tail mass as a two-sided 80% CI
    and is deliberately looser than SYSTEM's promotion-gate CI so that
    `SR_LCB_t` is not systematically starved to near-zero on top of the
    already-strict DSR gate — the DSR gate (Section 2.6) is what enforces
    "does this edge exist at all," while `SR_LCB_t`'s CI level only tunes
    "how much of a PASSING edge to size against");
  - the capacity tightness `κ = 0.02` (2% of trailing `ADV_i` per position)
    and the liquidity-cap multiple `C ∈ {2, 3}` (only engaged above the
    stated capacity threshold, never by default — Section 5.2). `κ=0.02` is
    anchored to the EMPIRICAL BASELINE's own measured **~$3M median
    uncapped-deployable book size**: at `κ=0.02`, a single position capped
    at 2% of a liquid major's typical multi-hundred-million-dollar daily
    volume does not bind until the BOOK itself approaches the measured
    capacity figure, consistent with observed behavior rather than an
    independently invented tightness;
  - the drawdown-governor ladder breakpoints, anchored to the EMPIRICAL
    BASELINE's own **~25% max-DD-at-100%-gross** figure (Section 6.4 states
    the full illustrative ladder and its derivation from this anchor);
  - the CUSUM/sequential-test parameters: reference offset `k_ref = 0.25`
    (in standardized-surprise units — Page 1954's convention is to set the
    reference offset to roughly half the standardized shift you want to
    detect quickly; Section 6.3 targets detecting a sustained
    half-Kelly-worthy outperformance, i.e. `z_k` running consistently
    `~0.5` above `SR_ref` in standardized units, so `k_ref=0.25` is the
    "half the target shift" convention applied to that target), decision
    threshold `h_promote = 5.0` (chosen, per the standard CUSUM ARL relation
    `ARL_0 ≈ exp(2·k_ref·h_promote)/(2·k_ref²)` for a one-sided CUSUM on
    standardized Gaussian increments — Page 1954; Hawkins & Olwell 1998 give
    this closed-form ARL approximation for the one-sided Gaussian CUSUM —
    to target `ARL_0 ≈ 52` weeks, i.e. an expected ONE FALSE PROMOTION PER
    YEAR of in-control operation: solving `52 ≈ exp(2·0.25·h)/(2·0.25²) =
    8·exp(0.5h)` gives `exp(0.5h) ≈ 6.5`, `h ≈ 2·ln(6.5) ≈ 3.75`; `h_promote`
    is set slightly above this closed-form solution, at `5.0`, as a
    deliberate conservative margin given the Gaussian-increment assumption
    is only approximate for weekly crypto returns — RED, BLUE-searchable
    offline only against this formula, never against realized promotions);
  - `N_min_weeks = 52` (minimum elapsed live-time floor before promotion is
    even eligible, regardless of CUSUM state — anchored to requiring at
    least one full annual cycle of live weekly observations, which is also
    the same order of magnitude as the `ARL_0≈52`-week false-promotion
    budget above, so the floor and the sequential test are calibrated on a
    consistent time unit);
  - the CUSUM ratchet's **down-step (de-lever) threshold**
    `demote_z_threshold = -k_ref = -0.25`. This is NOT a new free number: it
    is the SYMMETRIC MIRROR of the promotion CUSUM's own reference offset,
    derived from the identical Hawkins & Olwell (1998) one-sided-Gaussian-CUSUM
    framework used for `h_promote`. The promotion side is an UPPER CUSUM
    `S_k^+ = max(0, S_{k-1}^+ + z_k − k_ref)`; the de-lever side is the
    corresponding LOWER CUSUM `S_k^- = min(0, S_{k-1}^- + z_k + k_ref)` that
    alarms at `S_k^- ≤ −h_demote`. The §6.2 / §8 design mandate is that
    demotion be IMMEDIATE and asymmetric (fast to demote, slow to promote) —
    NOT gated by an accumulation — which is exactly the degenerate lower-CUSUM
    case `h_demote = 0`: with no accumulation buffer the lower CUSUM alarms on
    a single step whenever `z_k + k_ref < 0`, i.e. `z_k < −k_ref = −0.25`. So
    the same `k_ref = 0.25` that sets "half the target shift" on the promotion
    side pins the demotion trip at `−0.25` on the de-lever side; the value is
    the reference offset, not an independent quantity. The closed-form ARL of
    this `h_demote = 0` lower CUSUM is `ARL_0 ≈ exp(2·k_ref·0)/(2·k_ref²) =
    1/(2·0.25²) = 8` weeks — an order of magnitude FASTER than the promotion
    side's `ARL_0 ≈ 52`, which is precisely the "fast to demote, slow to
    promote" asymmetry §6.2 requires, now quantified rather than asserted
    (RED, BLUE-searchable offline only against this same formula, never against
    realized demotions). The injection path is retained: an OPS governance
    layer MAY override this default (e.g. `0.0` to demote on ANY negative
    surprise, a stricter user preference) — but the spec-cited default is the
    framework-derived `−0.25`, so it is no longer an unassigned constant;
  - `G_max = 1.5` (absolute gross-exposure ceiling independent of the Kelly
    computation — a circuit breaker for a data glitch, not a sizing
    target). Rationale for `1.5` specifically: it sits ABOVE `c=0.50 ·
    f*_LCB` at the EMPIRICAL BASELINE's own honest-Sharpe magnitude (Section
    7's worked example computes `c_t·f*_LCB ≈ 0.585` at quarter-Kelly using
    the CITED `SR_LCB_t`/`σ̂_t` magnitudes; at half-Kelly this doubles to
    `≈1.17`), so `G_max=1.5` does not bind under any INTENDED operating
    point implied by the cited evidence — it exists purely to catch a
    pathological input (e.g. a corrupted `σ̂_t` near zero) forcing an
    absurd `G_t`, never to cap a legitimately-computed half-Kelly sizing.
- **BLUE — finite validated grids (optional, class-level only).** A small
  candidate set for `h_vol` (e.g. `{4, 8, 13}` weeks) and for the CUSUM
  `(k_ref, h_promote)` pair, searched offline under DSR/PBO control,
  promoted the same way SYSTEM's own BLUE grids are (never per-symbol,
  never on raw live P&L). `L_boot` is explicitly NOT a BLUE grid — it is
  pinned to SYSTEM's own derivation (above), so there is exactly one
  block-length choice in the whole system, not two independently tuned
  ones.

> **Invariant C-cfg — RED unreachable from per-instrument config** (inherited
> verbatim from COST/SYSTEM). `c`, `h_vol`, `κ`, `C`, the governor ladder, and
> the CUSUM ratchet parameters enter only via `resolve_config(defaults=...)`.
> If any of these appears as a per-instrument field, that is the bug.

---

## 4. Volatility estimation: `σ̂_t`, the EWMA, and its event-count floor

### 4.1 The estimator

`σ̂_t` is a causal EWMA of the book's own realized weekly (100%-gross
reference) return, on the WEEKLY clock (the book's native decision cadence,
mirroring COST's discipline of putting the funding EWMA on the funding-stamp
clock rather than wall-clock time, COST §5.5a):

```
m_k        = EWMA of the weekly return level:  m_k = α·R_k + (1−α)·m_{k−1}
v_k        = EWMA of the squared deviation:     v_k = α·(R_k − m_{k−1})² + (1−α)·v_{k−1}
σ̂_t        = sqrt(v_k)   at the most recent completed week k = t
α          = 1 − 2^(−1/h_vol)          (half-life h_vol in WEEKS, RED)
ESS        = (2 − α) / α                (effective sample size of the EWMA)
```

This is exactly the COST §5.5a closed-form recursion, re-derived on the
weekly-return clock instead of the funding-stamp clock — same estimator
family (RiskMetrics-style EWMA; J.P. Morgan RiskMetrics, 1996), same
closed-form update, same `O(1)` streaming cost, same half-life-to-`α` mapping.
No parameter is shared numerically with COST's funding EWMA (different clock,
different half-life) — only the FORM is reused (Invariant NO-DUP-EWMA below).

### 4.2 Event-count floor and fallback (mirrors COST §5.5a, TREND's own floors)

`σ̂_t` is trusted only once the realized weekly-return history has at least
`n_vol_min` observations (RED, e.g. `≥ 2·h_vol` so `ESS` is meaningfully
filled — identical discipline to COST's `n_funding_min` and LEVERAGE's own
`n_residual_min`-style floors elsewhere in the repo). Below the floor, `σ̂_t`
falls back, in priority order (P6-style — conservative, i.e. HIGHER assumed
vol, never lower):
1. a class-level realized-vol prior anchored to the EMPIRICAL BASELINE's own
   figure — **~28% annualized vol at 100% gross** (⇒ `σ̂_weekly ≈
   28%/√52.18 ≈ 3.9%`), used as the fallback level, not a hand-picked guess;
2. if even that prior is unavailable (a cold-start with zero history), a
   deliberately wide multiple of it (RED), so a cold book starts
   UNDER-levered, never over-levered.

> **Invariant VOL-FLOOR — thin history widens, never narrows, the assumed
> vol.** A test asserts that a synthetic history shorter than `n_vol_min`
> produces `σ̂_t ≥` the class-level 28%-annualized prior, and that `G_t` at
> that fallback is strictly smaller than `G_t` computed from a full, calm
> live history at the same `SR_LCB_t`.

### 4.3 Why this is NOT a fitted cross-sectional covariance model (Invariant NO-MV)

`σ̂_t` is a univariate EWMA on ONE scalar series (the aggregate book return).
It is not, and must never become, an estimate of the `N×N` cross-sectional
covariance matrix of the ~50 underlying names — that matrix is exactly the
object the Marchenko–Pastur argument in P4 says is too noisy to invert at
`N≈50, n_obs≈258` (`N/n_obs ≈ 0.19`; Marchenko & Pastur, 1967; empirical
finance treatment in Laloux, Cizeau, Bouchaud & Potters, 1999). If a
covariance term is ever needed for a future extension (e.g. blending crypto
momentum with the equities/commodities sleeves `research/xsec/universe.py`
already scaffolds), it MUST be Ledoit-Wolf shrunk toward a structured target
(Ledoit & Wolf, 2004), never the raw sample covariance, and is out of scope
for the crypto-only v1 sizing rule in this file.

---

## 5. Capacity ceiling: `κ·ADV_i` and the liquidity-capping dial

### 5.1 The ceiling

For each instrument `i` in the scaled book, the position's dollar notional
implied by `G_t` must satisfy:

```
|position_i(G_t)|  ≤  κ · ADV_i
```

where `ADV_i` is the TEAL trailing dollar-volume estimate for instrument `i`
(causal, no lookahead — the same style of trailing-volume forecast COST's
impact model uses for its own causal `v_hat`, COST §5.4/Invariant W) and `κ`
is a RED tightness constant. This is evaluated on the FULLY SCALED book
(after `G_t` is applied), so it is a genuine `min()`-term on `G_t`: the
largest `G_t` consistent with EVERY instrument's capacity ceiling holding
simultaneously is:

```
G_t^{cap}  =  min_i  ( κ · ADV_i  /  |w_i^{ref}| )
```

where `w_i^{ref}` is instrument `i`'s reference-gross (100%) dollar weight
from `research/xsec/portfolio.py`'s already-built book. A name with
`shortable = False` (AMBER) contributes `ADV_i = 0` on the short side of this
computation regardless of its measured liquidity, so an unshortable name
cannot silently inflate capacity via the long-only side of a two-sided
liquidity estimate.

### 5.2 The liquidity-capping dial — a CAPACITY tool, never a default

The EMPIRICAL BASELINE is explicit and must be encoded literally: **liquidity-
capped within-leg weighting (`research/xsec/portfolio.py`'s `liqcap_volscaled`
scheme) was tested and LOSES vs plain inverse-vol weighting** — because
crypto majors are simultaneously low-vol AND high-liquidity, a liquidity-
share cap paradoxically gives majors the LARGEST weight ceiling, so effective
breadth (inverse-HHI, `research/xsec/universe.py::effective_breadth`) FALLS
under capping (**~40 → 28–34**), not rises. Capping only buys CAPACITY:
**median deployable book ~$3M uncapped → ~$12–23M at cap tightness `C=2–3`**,
at a cost of **~0.4–0.6 Sharpe**. Therefore:

```
INVARIANT CAP-DEFAULT: the within-leg weighting scheme is 'volscaled'
(plain inverse-realized-vol) BY DEFAULT. 'liqcap_volscaled' at cap
tightness C is engaged ONLY when the target book size (equity × G_t
× reference gross) exceeds a stated capacity threshold (RED, anchored
at the measured ~$3M uncapped-median-deployable figure) — i.e. C is a
capacity DIAL turned on above a book-size trigger, never a default
sizing choice. Turning it on is logged as a reason code
('liquidity_cap_engaged', C) distinct from any LEVERAGE ceiling.
```

> **Invariant CAP-DIAL — liquidity capping is capacity, not a sizing
> ceiling, and is off by default.** A test asserts that at the default
> book size (below the RED capacity threshold), the within-leg weighting
> scheme resolved by config is `'volscaled'`, and that engaging
> `'liqcap_volscaled'` strictly reduces realized Sharpe in a backtest
> fixture relative to `'volscaled'` at the same `C` grid point used in the
> EMPIRICAL BASELINE sweep (consistent with the measured ~0.4–0.6 Sharpe
> cost), so a future change cannot silently flip the default without
> failing this regression.

This is a boundary case worth being precise about: LEVERAGE (this file) owns
the CEILING math (`G_t^{cap}` above); WHICH within-leg weighting scheme is
active is technically a SELECTION/portfolio-construction choice
(`research/xsec/portfolio.py`), but because the choice is capacity-driven and
must be gated on the SAME book-size number LEVERAGE computes (`equity ×
G_t`), LEVERAGE is the module that owns the trigger threshold and emits the
`liquidity_cap_engaged` directive; `portfolio.py` consumes it as a config
flag. This avoids a circular fit (SELECTION would otherwise need to know
`G_t` before LEVERAGE has computed it) — the resolution is that
`quantile_weights`/`liquidity_weights` are evaluated at a REFERENCE gross of
1.0 exactly as today, and the capacity trigger is checked against the
PROJECTED book size using the CURRENT week's `G_t`, applied to next week's
construction (a one-week lag, consistent with the weekly rebalance cadence
already in the system — no same-week circularity).

---

## 6. The composed sizing rule, the drawdown governor, and the promotion ratchet

### 6.1 The `min()` composition

```
G_t  =  min( c_t · (SR_LCB_t / σ̂_t) ,     # Section 2.5 — fractional-Kelly / vol-target sizing
             G_t^{cap} ,                   # Section 5 — capacity ceiling
             m_t · G_max ,                 # Section 6.4 — drawdown governor × hard cap
             G_max )                       # absolute circuit breaker (RED)
```

`min()` only, never a weighted blend (mirroring COST P3's "ceiling, never a
score" discipline and SYSTEM's own P2 "no learned combiner"). The binding
term is recorded on every decision (`binding_ceiling ∈ {'kelly', 'capacity',
'governor', 'hard_cap'}`), exactly analogous to `LeverageDecision
.binding_ceiling` in the pre-pivot system's boundary type.

### 6.2 `c_t`: which fractional-Kelly dial is currently active

`c_t ∈ {0.25, 0.50}` (RED, Section 2.3), starting at `c_t = 0.25`
(quarter-Kelly) on cold start and for every newly-promoted config, per P7.
`c_t` steps to `0.50` (half-Kelly) ONLY after the sequential test of Section
6.3 passes, and steps back DOWN to `0.25` immediately if the latest
standardized weekly surprise `z_k` falls below the demotion threshold
`demote_z_threshold = −k_ref = −0.25` (asymmetric: fast to demote, slow to
promote — a standard risk-control asymmetry, not a special statistical
claim). The demotion threshold is NOT a free constant: it is the symmetric
mirror of the SAME Section 6.3 CUSUM framework, taken in its degenerate
IMMEDIATE (`h_demote = 0`) form so that no accumulation is required — the
lower-CUSUM step `S_k^- = min(0, S_{k-1}^- + z_k + k_ref)` alarms on a single
observation exactly when `z_k + k_ref < 0`, i.e. `z_k < −k_ref = −0.25`.
Section 3 states the RED value and its full derivation (including the
`ARL_0 ≈ 8`-week de-lever run length this `h_demote = 0` case implies, an
order of magnitude below the promotion side's `ARL_0 ≈ 52`, quantifying the
"fast to demote" half of the asymmetry). The value is overridable by an OPS
governance layer (the injection path is kept), but is no longer unassigned.

### 6.3 The ratchet as a sequential test (CUSUM), not a vibes-based judgment call

Per P7, promotion from quarter- to half-Kelly is formalized as a one-sided
CUSUM sequential test (Page, 1954) on the LIVE weekly (not backtest) net
return series, run in per-observation Sharpe-standardized units:

```
Let z_k = (SR_week,k − SR_ref) / SE(SR_week,k)      (standardized weekly Sharpe surprise,
                                                       SR_ref = the deflated SR_LCB_t at
                                                       promotion time, held fixed as the reference)
S_0 = 0
S_k = max(0, S_k-1 + z_k − k_ref)                    (upper CUSUM, reference offset k_ref, RED)
PROMOTE when S_k ≥ h_promote                          (RED decision threshold)
```

**The generating-process assumption, stated explicitly (this was previously
left implicit).** The CUSUM statistic `S_k` and its `ARL_0` calibration are
defined and calibrated under the assumption that, IN CONTROL (i.e. the book
truly performing at exactly `SR_LCB_t`, no better and no worse), the
standardized weekly surprises `z_k` are approximately **i.i.d. standard
Gaussian** (`z_k ~ N(0,1)`, independent across weeks). This is the SAME
"per-observation, standardized" convention `SR_LCB_t`/DSR's own z-test uses
(SYSTEM §5.2's `SR_pp` per-observation contract), applied here to a rolling
SURPRISE rather than a single point estimate. This assumption is an
approximation, not exactly true, for two identified reasons: (i) weekly
cross-sectional momentum returns have non-zero skew/kurtosis (SYSTEM §5.3's
`γ3,γ4` moments feed the DSR z-test precisely because normality is not
exact), and (ii) `SE(SR_week,k)` — the standardizing denominator — is
ITSELF an estimate with sampling error, and is assumed constant/known for
the purpose of the CUSUM recursion (a "known-variance CUSUM," the classical
Page 1954 setting) rather than re-estimated adaptively; if the book's
realized-vol regime shifts materially, the OFF-SPEC risk is a distorted ARL,
which is exactly why Section 4's EWMA governs `σ̂_t` on a SHORT half-life
(`h_vol=8` weeks, Section 3) so `SE(SR_week,k)` tracks regime change rather
than being fixed to a stale estimate over the multi-year CUSUM accumulation
window.

`k_ref = 0.25` and `h_promote = 5.0` (RED, BLUE-searchable offline only,
never per-symbol; Section 3 states the illustrative v1 values and their
derivation) are chosen to target a stated in-control average run length
`ARL_0 ≈ 52` weeks (the expected number of weeks before a FALSE promotion
under a book truly performing at exactly `SR_LCB_t`, no better) via the
standard closed-form one-sided-Gaussian-CUSUM ARL approximation (Page, 1954;
Hawkins & Olwell, 1998, *Cumulative Sum Charts and Charting for Quality
Improvement*, ch. 4, for the explicit closed-form `ARL_0 ≈
exp(2·k_ref·h_promote)/(2·k_ref²)` under the i.i.d.-Gaussian in-control model
stated above) — the same average-run-length calibration discipline OPS's own
CUSUM detectors use elsewhere in this repo, and (per Section 3) a genuine
closed-form calculation from the stated model, not a Monte Carlo fit to
historical P&L. A Monte-Carlo verification against synthetic `N(0,1)`
surprise sequences (never against realized live P&L, which would violate
P5/P7) is the acceptance-gate check (Section 10) that the closed-form
approximation and the actual recursion agree within stated tolerance. A
**minimum elapsed live-time floor** `N_min_weeks = 52` (RED, Section 3) is
enforced as an additional AND-condition, because CUSUM alone can fire early
on a short lucky streak — the floor is the operational embodiment of "N
months of live weekly Sharpe holding above a threshold," a governance
requirement this spec adopts (RED, Section 3) rather than a figure sourced
from EVIDENCE, made precise as `elapsed_weeks ≥ N_min_weeks AND S_k ≥ h_promote`.

> **Invariant RATCHET — promotion is a stopping-time, not a snapshot
> comparison.** `c_t` may only move from 0.25 to 0.50 when the CUSUM
> statistic `S_k` (accumulated over the ENTIRE live track record since
> promotion eligibility began, not a rolling window that forgets prior bad
> weeks) crosses `h_promote` AND the minimum-elapsed-time floor is satisfied.
> A test asserts that a live series with one lucky week embedded in an
> otherwise flat/negative record does NOT promote (the accumulated `S_k`
> from the surrounding weeks pulls it back below `h_promote`), while a
> genuinely sustained outperformance eventually crosses within the
> ARL-calibrated expected time.

### 6.4 Drawdown governor (kept from the old spec, re-derived on book equity)

The drawdown-governor LADDER SHAPE is genuinely portable math from the
pre-pivot LEVERAGE.md and is salvaged here verbatim in form, re-triggered on
the NEW risk unit (book equity peak-to-trough, computed from the SAME weekly
`net` series everything else in this file uses, not from any per-trade P&L):

```
dd_t  =  1 − E_t / peak(E_{≤t})                      (running peak-to-trough drawdown on book equity)
m_t   =  ladder(dd_t)                                 (RED breakpoints; a monotone NON-INCREASING
                                                        step or ramp in dd_t, m_t ∈ [0, 1])
m_t → 0  at a RED terminal drawdown threshold          (full flat — governor kill)
```

The EMPIRICAL BASELINE's own max-drawdown figure anchors the ladder's design
range: **max drawdown ~25% at 100% gross exposure (~28% annualized vol)**.

**The linear-in-gross scaling claim, made precise (regularity condition
stated explicitly — this was previously asserted by hand-wave).** The claim
"drawdown scales ~linearly with gross" is NOT a generic property of the
drawdown functional — it requires a stated condition. Let `E_t(G) = E_0 ·
∏_{s≤t}(1 + G·R_s)` be the equity path at constant gross `G`, and let
`DD(G) = 1 − min_t[E_t(G)/peak_{s≤t}E_s(G)]` be its running max-drawdown
functional. `DD(G)` is linear in `G` ONLY in the SMALL-`G·R` regime where
`log(1+G·R_s) ≈ G·R_s` (the identical first-order Taylor approximation
already invoked in Section 2.2, `|f·R|` small) — under that approximation,
`log E_t(G) ≈ G·Σ_{s≤t} R_s`, a linear (not multiplicative-compounding)
functional of `G`, and the running-max-drawdown of a linearly-G-scaled LOG
path is EXACTLY `G`-proportional in log-drawdown terms (`1 − e^{G·(peak−E_t)
in log space}`), which is approximately proportional in the ORIGINAL
(non-log) drawdown units only for small drawdowns. **This is the same
regularity condition as Section 2.2's, applied to a different functional of
the same linearized return series** — it holds well at the book's own
`G_t` operating range (Section 7's worked example never exceeds `G_t≈1.2`,
comfortably inside the small-`G·R` regime given weekly `σ̂_t≈3.9%`), and
degrades as `G` grows large enough that compounding effects and the
nonlinearity of `1−e^x` versus `x` diverge — i.e. the linear-scaling
approximation is reliable PRECISELY IN THE REGIME this module's own Kelly
sizing (Section 2) keeps `G_t` inside, which is a consistency property of
the design, not a coincidence, but it is NOT claimed to hold for an
arbitrarily large hypothetical `G_t` (e.g. `G_t=5`) where the approximation
would need re-derivation.

**Illustrative v1 ladder (RED, anchored to the 25% figure above).** A
monotone-non-increasing step ladder in `dd_t`, expressed as fractions of the
25% reference anchor:

```
dd_t < 0.25 · 0.5 = 0.125            : m_t = 1.00   (no de-risking; inside half the reference DD)
0.125 ≤ dd_t < 0.25                  : m_t = 0.60   (b/w half and one reference DD — de-risk)
0.25  ≤ dd_t < 0.25 · 1.5 = 0.375    : m_t = 0.35   (past the reference anchor — de-risk hard)
dd_t ≥ 0.40                          : m_t = 0.00   (terminal threshold — full flat, governor kill)
```

This ladder is illustrative (RED, BLUE-searchable offline only) but is not
unanchored: EVERY breakpoint is a stated multiple of the cited 25% figure
(`0.5×`, `1.0×`, `1.5×`, `1.6×`), not an arbitrary round number, so a future
recalibration of the empirical max-DD figure propagates through the whole
ladder rather than requiring four independently-re-justified breakpoints.

**Cautionary edge case — an over-aggressive ladder can reproduce a DD-stop's
Sharpe collapse (evidence-grounded, previously missing from this spec).**
The EMPIRICAL BASELINE's OWN funding-carry sleeve was tested with a CAUSAL
drawdown-stop (structurally the closest analogue in the evidence base to
this governor): capping the carry sleeve's max DD at ~24% collapsed its
Sharpe from ~0.57 (beta-hedged) to ~0.10 and its income from ~12.9%/yr to
~0.6%/yr. This is NOT a reason to avoid a governor — this book's DD profile
(25% max at 100% gross, no structural regime-bleed) is categorically
different from carry's (70-75% structural DD, dies in the 2023-24 low-
funding stretch) — but it IS direct evidence that a drawdown-based
risk-control mechanism CAN destroy most of a strategy's Sharpe if tuned too
tightly relative to the strategy's OWN natural drawdown scale, independent
of whether the underlying edge is real. The governor ladder above is
deliberately anchored at multiples of the measured 25% max-DD (not an
arbitrary tighter number, e.g. not de-risking at `dd_t=0.05`) precisely to
avoid reproducing the carry sleeve's failure mode: de-risking should engage
around and somewhat past the historically observed drawdown scale, not
inside its normal operating range. See Section 8 for this as a standing
edge case with an explicit test.

The governor multiplies the min() chain (`m_t · G_max` term above) exactly as
the old spec's governor did — this is the one ceiling where "kept and
re-derived on the new risk unit" is a near-literal port, because equity
drawdown is already a portfolio-level (not per-trade) concept and required no
risk-unit translation, only a change of the SERIES it reads (book weekly
equity, not per-trade realized P&L).

---

## 7. Worked numeric example (reproduce this bit-for-bit)

All numbers below are either DERIVED (Section 2 formulas) or CITED from the
EMPIRICAL BASELINE by name — no invented constants.

**Inputs (all `F_t`-measurable at a hypothetical rebalance `t`):**

| Input | Value | Source |
|---|---|---|
| `σ̂_t` (weekly, EWMA) | `0.0389` (3.89%) | derived from cited **~28% annualized vol** ÷ `√52.18` (`research/xsec/metrics.py::PERIODS_PER_YEAR = 365.25/7 ≈ 52.18`) |
| `SR_LCB_t` (per-observation, deflated lower bound) | `0.091` | illustrative CI-low draw consistent with CPCV mean OOS path Sharpe **≈0.91** annualized ÷ `√52.18` ≈ 0.126 per-obs, shaded down for the bootstrap CI margin per Section 2.6 — the ACTUAL number at deploy time is `max(0, ci_low)` from `convexity.system.validate.robustness.block_bootstrap_ci(path_returns, ...)`, gated by `DeflatedSharpe.deflate(...).passed` (Section 2.6); this row is a worked-example placeholder consistent with the cited magnitude, not a real bootstrap draw |
| `c_t` | `0.25` (quarter-Kelly, cold start) | Section 6.2 |
| `dd_t` | `0.0` (no drawdown at deploy) | — |
| `m_t` | `1.0` | governor ladder at zero drawdown |
| `G_max` | `1.5` | RED circuit breaker (illustrative) |
| `G_t^{cap}` | `2.0` | illustrative capacity ceiling, not binding here |

**Step 1 — Kelly/vol-target sizing term:**

```
f*_LCB = SR_LCB_t / σ̂_t = 0.091 / 0.0389 = 2.339
c_t · f*_LCB = 0.25 × 2.339 = 0.585
```

**Step 2 — `min()` composition:**

```
G_t = min( 0.585 ,        # Kelly/vol-target
           2.0 ,           # capacity
           1.0 × 1.5 ,     # governor × hard cap = 1.5
           1.5 )           # absolute circuit breaker
    = 0.585
binding_ceiling = 'kelly'
```

**Step 3 — growth-capture sanity check (Section 2.3 identity):**

```
c·(2−c) at c=0.25:  0.25 × 1.75 = 0.4375   (~44% of full-Kelly growth captured)
```

This is a pure arithmetic identity of Section 2.3's `c·(2−c)` growth-capture
formula at `c=0.25` (quarter-Kelly), not an externally-sourced figure — it
confirms the derivation in Section 2.3 is applied consistently end-to-end from
the first-order condition through to the deployed `G_t`.

**Step 4 — a drawdown-governor-binding contrast.** If instead `dd_t = 0.15`
and the RED ladder maps `dd_t=0.15 → m_t=0.60`:

```
governor term = 0.60 × 1.5 = 0.90
G_t = min(0.585, 2.0, 0.90, 1.5) = 0.585     # Kelly term still binds; governor has slack
```

whereas at `dd_t = 0.30` with `m_t = 0.35` (illustrative ladder point past the
cited **~25% empirical max-DD anchor**):

```
governor term = 0.35 × 1.5 = 0.525
G_t = min(0.585, 2.0, 0.525, 1.5) = 0.525
binding_ceiling = 'governor'
```

showing the ceiling switch a builder should reproduce: the SAME Kelly term
(`0.585`) is now NOT binding once realized drawdown pushes past the point
where the empirically-anchored ladder bites.

---

## 8. Edge cases and invariants (structural; ask before breaking)

- **`σ̂_t = 0` or non-finite.** Falls back per Section 4.2's priority list
  (class-level 28%-annualized prior, then a wider multiple); `G_t` NEVER
  divides by a zero/NaN `σ̂_t` un-guarded (`Invariant VOL-FLOOR`).
- **`SR_LCB_t ≤ 0`.** `G_t^{Kelly-term} = 0` via the `max(0, ·)` floor in
  Section 2.6 — the book runs FLAT (subject still to `min()` with the other
  terms, all of which are ≥0, so the composed `G_t = 0`), never a negative
  or "short-the-book" leverage (LEVERAGE never inverts the book's side; a
  negative edge estimate means "don't lever it," not "run it inverted" — the
  EMPIRICAL BASELINE's own note that inverting the dead directional-breakout
  book did NOT produce an edge is the cautionary precedent for why a
  negative-edge diagnosis is never translated into an inverted position).
- **DSR gate failing at deploy time.** `SR_LCB_t = 0` unconditionally
  (Section 2.6) regardless of how large the raw sample Sharpe looks — this is
  the direct, load-bearing consequence of Invariant K-LCB and is exactly the
  regime the EMPIRICAL BASELINE describes today (PSR≈0.994 but `n_obs≈258 <
  MinTRL≈824`, SYSTEM §5.5's own worked number — currently a Gate-2-branch-2b
  CONDITIONAL pass per SYSTEM §6.2, not a clean pass): LEVERAGE does not wait
  for a future spec revision to apply this guard; it is live from week one.
  Concretely, TODAY's deploy-time gate state (per SYSTEM §6.2) is branch 2b
  (PSR high, track too short) rather than branch 2a (clean DSR≥0.95) — under
  Section 2.6's rule this STILL yields `DeflatedSharpe.deflate(...).passed ==
  True` (branch 2b is a PASS, just a qualified one), so `SR_LCB_t` is sourced
  from the bootstrap CI as normal; it is Gate 1 (`n_true_distinct >= k_min`,
  `k_min=3` per SYSTEM §6.1 — NOT the fabricated `>=30` floor an earlier
  SYSTEM.md draft stated and has since retracted; the correct citation is
  `research/xsec/validate.py::weekly_defaults(k_min=3)`) and Gate 6
  (cost-stress) being OPEN (SYSTEM §6.2) — not Gate 2 — that would
  currently block a live promotion if either resolves to a fail once run.
- **Drawdown governor ladder tuned too aggressively collapses realized
  Sharpe the same way the EMPIRICAL BASELINE's carry-sleeve DD-stop did.**
  The carry sleeve's causal DD-stop (cap at ~24% DD) cut its Sharpe from
  ~0.57 to ~0.10 and its income to ~0.6%/yr — proof that an aggressively
  tuned drawdown de-risking mechanism can gut a strategy's realized Sharpe
  independent of whether the edge is genuine. Section 6.4's ladder is
  deliberately anchored at multiples (`0.5×, 1.0×, 1.5×, 1.6×`) of the
  measured ~25% max-DD-at-100%-gross figure rather than an arbitrary tighter
  threshold specifically to avoid this failure mode — de-risking begins only
  AT/BEYOND the historically observed drawdown scale, not inside it. A test
  (Section 10) asserts that a backtest fixture replaying the book's own
  historical drawdown path through the governor at the v1 ladder does NOT
  reduce full-history Sharpe by more than a stated tolerance relative to an
  un-governed (`m_t≡1`) replay — an empirical regression guard against
  silently re-tightening the ladder into carry's failure regime.
- **CUSUM in-control model (i.i.d. Gaussian surprises, Section 6.3) is
  violated by a real regime shift.** If the book's true weekly Sharpe or vol
  regime shifts abruptly, the calibrated `ARL_0≈52` weeks no longer holds
  exactly (the closed-form ARL assumes the in-control model applies
  throughout the accumulation window) — this is a known, stated
  approximation (Section 6.3), not silently ignored. The mitigation is
  structural, not a fix to the CUSUM math: `σ̂_t`'s short `h_vol=8`-week
  half-life re-estimates the standardizing `SE(SR_week,k)` quickly, and a
  genuine adverse regime shift shows up independently in the drawdown
  governor (Section 6.4) and in `c_t` demotion (Section 6.2), both of which
  act on realized returns directly rather than through the CUSUM's
  parametric assumption — the CUSUM is a PROMOTION gate (asymmetric, slow to
  fire), not the system's only defense against regime change.
- **Cold start, zero live history.** `c_t` starts at the quarter-Kelly floor
  (Section 6.2); the CUSUM ratchet (Section 6.3) cannot have accumulated
  enough evidence to promote, so `c_t = 0.25` is enforced structurally, not
  just by "starting there" — the minimum-elapsed-time floor makes early
  promotion impossible even if a lucky short streak produces a large `S_k`.
- **Capacity ceiling binds below the Kelly-implied `G_t`.** The book simply
  runs smaller than the edge would otherwise justify; this is an intentional,
  hard `min()`, and the `liquidity_cap_engaged` capacity dial (Section 5.2)
  is the ONLY lever that relaxes it — never a relaxation of `κ` on
  discretion.
- **A newly-listed instrument with `< n_vol_min` (or no) trailing `ADV_i`
  history.** Treated as `ADV_i = 0` for capacity purposes until enough
  history accrues (P6 conservative fallback) — it cannot contribute
  unbounded capacity by virtue of being unmeasured.
- **Governor and Kelly-term disagree in DIRECTION as drawdown evolves.** By
  construction (`min()`, monotone-non-increasing ladder in `dd_t`, Section
  6.4) the governor term can only ever PULL `G_t` down relative to the
  Kelly/capacity terms, never push it up — there is no code path where the
  governor increases `G_t` above what Kelly/capacity alone would allow.
- **`c_t` demotion.** Demotion from half- back to quarter-Kelly is immediate
  on a stated live-Sharpe breach (Section 6.2) and is NOT itself gated by a
  symmetric CUSUM accumulation — the asymmetry (slow to promote, fast to
  demote) is a deliberate risk-control design choice, not an oversight, and
  is called out explicitly so a reviewer does not "fix" it into a symmetric
  test.
- **Reference-gross vs. deployed-gross bookkeeping.** All of Sections 2, 4,
  and the worked example in Section 7 operate at the 100%-gross REFERENCE
  book; `G_t` is the multiplier applied once, at the very end, to translate
  reference-book weights into deployed dollar positions. No intermediate
  computation in this file re-multiplies by `G_t` twice (a `G_t²` bug is
  structurally impossible because `μ, σ, SR_LCB_t` are all estimated at the
  fixed reference gross of 1.0, per Section 2.1's leverage-invariance
  argument, and `G_t` is applied exactly once at the position-translation
  boundary, mirroring `research/xsec/portfolio.py`'s own "report at 100%
  gross and note the scaling" convention).

> **Invariant NO-DUP-EWMA — one EWMA form, two independent instantiations,
> never a shared mutable half-life.** LEVERAGE's `σ̂_t` EWMA (Section 4) and
> COST's funding-forecast EWMA (COST §5.5a) share the closed-form
> `α = 1 − 2^(−1/h)` update RULE but are separately parameterized
> (`h_vol` in weeks vs. COST's `h` in 8h funding stamps) and separately
> instantiated objects. A test asserts changing COST's funding half-life
> config leaves LEVERAGE's `σ̂_t` byte-identical, and vice versa.

> **Invariant SIZE-ONE — exactly one free sizing dial (`c`).** A test
> asserts that the reported diagnostic `σ_target = c·μ̂` (Section 2.4) is
> ALWAYS internally consistent with the deployed `G_t`'s Kelly term
> (`G_t^{Kelly} · σ̂_t == c·SR_LCB_t`, to floating-point tolerance) — there is
> no code path that sets a `σ_target` independently of `c`.

---

## 9. Directory layout

Reuses the `leverage/` package location `docs/ARCHITECTURE.md` already
scaffolds, but with the internal module set reduced to match the three-
ceiling composition of Section 6 (the old `ceilings/{stop,tail,portfolio,
margin,liquidity}.py` five-file set is retired; `vol_target.py` and
`kelly.py` merge into one module because Section 2 shows they are the same
rule, per Invariant SIZE-ONE):

```text
leverage/
+-- types.py               # LeverageDecision, GTarget, CapacityInput, GovernorState (frozen dataclasses)
+-- engine_api.py           # LeverageEngine.size_book(...) -> LeverageDecision (ONE public method)
+-- config/
|   +-- defaults.py         # RED: c grid, h_vol, n_vol_min, L_boot/CI level, kappa, C, ladder, CUSUM params, G_max
|   +-- asset_class.py      # AssetClassCapacityPreset (validated per-class kappa/C fallbacks)
|   +-- instrument.py       # InstrumentSpec fields this module reads: shortable, lot/multiplier (AMBER only)
|   +-- resolve.py          # single merge point (RED unreachable from instrument, Invariant C-cfg)
+-- vol/
|   +-- forecast.py         # sigma_hat_t: the weekly EWMA (Section 4) + event-count floor/fallback
+-- edge/
|   +-- lcb.py              # SR_LCB_t: DeflatedSharpe.deflate(...).passed gate (fail->0) + max(0, block_bootstrap_ci(...)[0])
|   |                        # from convexity.system.validate.robustness (Section 2.6, Invariant K-CI) -- NOT PathSharpeSummary
+-- ceilings/
|   +-- kelly_vol_target.py # the ONE Kelly/vol-target term (Section 2.5); c_t * SR_LCB_t / sigma_hat_t
|   +-- capacity.py         # G_t^cap = min_i(kappa*ADV_i / |w_i_ref|); liquidity_cap_engaged trigger (5.2)
|   +-- governor.py         # drawdown ladder m_t (Section 6.4); the kept/re-derived piece
+-- ratchet/
|   +-- cusum.py            # the CUSUM promotion sequential test (Section 6.3, Page 1954)
|   +-- state.py            # c_t state machine: promote/demote, minimum-elapsed-time floor
+-- combine/
|   +-- engine.py           # min() composition (Section 6.1) + binding_ceiling reason code
+-- validate/
|   +-- labels.py           # realized weekly book return labels (firewalled, future-only)
|   +-- backtest.py         # reproduces Section 7's worked example against a fixture
|   +-- arl.py               # CUSUM average-run-length Monte-Carlo verification against the Hawkins & Olwell (1998)
|   |                        # closed-form target (Section 6.3) on SYNTHETIC N(0,1) surprise sequences only -- never on
|   |                        # realized live P&L (that would violate P5/P7)
|   +-- reports.py           # NOTE: LEVERAGE does NOT implement its own block-bootstrap CI module -- SR_LCB_t (edge/lcb.py)
|                             # calls convexity.system.validate.robustness.block_bootstrap_ci directly; a duplicate
|                             # stats.py/bootstrap module here would violate Invariant K-CI (no re-derivation)
+-- synth/
    +-- generate.py          # synthetic weekly return series (calm/drawdown/regime-break) for tests
```

Stack: Python 3.11+, numpy, scipy (for the normal quantiles inside any CI
construction and CUSUM ARL calibration — same scipy allowance ARCHITECTURE.md
§5 item 8 already grants COST/SYSTEM/OPS), pandas, pytest. No ML frameworks.

---

## 10. Module acceptance gates (CI)

### Volatility forecast (`vol/forecast.py`)
Pass when: `σ̂_t` is the closed-form EWMA of Section 4.1 on the WEEKLY clock;
`α = 1 − 2^(−1/h_vol)` with reported `ESS`; below `n_vol_min` observations
falls back to the **28%-annualized** class prior (Invariant VOL-FLOOR), never
to zero or an unguarded division.

### Confidence-adjusted edge (`edge/lcb.py`)
Pass when: `SR_LCB_t` calls `convexity.system.validate.robustness
.block_bootstrap_ci` (directly, or via `RobustnessSuite.run(...)
.sharpe_ci[0]`) on the promoted config's weekly return series and takes
`max(0, ci_low)`, matching a fixture bit-for-bit at a fixed seed (Invariant
K-CI); returns exactly `0` when `DeflatedSharpe.deflate(...).passed is
False` (Invariant K-LCB); a static-analysis/import check asserts `edge/`
never imports `research.xsec.cpcv.PathSharpeSummary` for this purpose and
never defines its own bootstrap routine (no `stats.py`/duplicate estimator
anywhere under `leverage/`).

### Kelly/vol-target ceiling (`ceilings/kelly_vol_target.py`)
Pass when: implements `c_t · SR_LCB_t / σ̂_t` exactly; the growth-capture
identity `c(2−c)` reproduces `0.4375` at `c=0.25` and `0.75` at `c=0.5`
bit-for-bit in a unit test (Section 2.3); no second independent `σ_target`
input exists anywhere in config (Invariant SIZE-ONE).

### Capacity ceiling (`ceilings/capacity.py`)
Pass when: `G_t^{cap} = min_i(κ·ADV_i/|w_i^{ref}|)`; unshortable names
contribute `ADV_i=0` on the short side; `liquidity_cap_engaged` fires only
above the RED book-size threshold and never by default (Invariant CAP-DIAL);
default within-leg scheme resolves to `'volscaled'`.

### Governor (`ceilings/governor.py`)
Pass when: `m_t` is a monotone non-increasing step/ramp in running
peak-to-trough book-equity drawdown `dd_t`; anchored so the ladder's design
range reflects the cited **~25% max-DD at 100% gross** (breakpoints at
stated multiples `0.5×/1.0×/1.5×/1.6×` of that anchor, Section 6.4); `m_t →
0` at the RED terminal threshold (full flat); reads BOOK equity (the same
weekly `net` series everything else in this file uses), never a per-trade
P&L series; a regression fixture replays the book's own historical drawdown
path through the ladder and asserts the governed Sharpe does not fall more
than a stated tolerance below the un-governed (`m_t≡1`) replay (the
carry-sleeve-DD-stop cautionary edge case, Section 8).

### Ratchet (`ratchet/cusum.py`, `ratchet/state.py`)
Pass when: promotion from `c=0.25→0.50` requires BOTH the CUSUM statistic
`S_k` crossing `h_promote` (calibrated to a stated `ARL_0≈52` weeks via the
Hawkins & Olwell 1998 closed form, Page 1954 for the CUSUM itself) AND the
minimum-elapsed-live-time floor `N_min_weeks=52` (Invariant RATCHET);
demotion is immediate/asymmetric (Section 6.2); a single lucky week cannot
promote (regression fixture required); the in-control model is documented
as i.i.d. `N(0,1)` standardized weekly surprises (Section 6.3) and the ARL
Monte-Carlo check (`validate/arl.py`) runs ONLY on synthetic draws from that
model, never on realized live P&L.

### Composition (`combine/engine.py`)
Pass when: `G_t = min()` of exactly the four terms in Section 6.1, never a
weighted blend; `binding_ceiling` reason code always populated and matches
the numerically smallest term; `G_t ≥ 0` always (never negative, never an
inverted book — edge case list, Section 8).

### Validation harness
Pass when: `validate/backtest.py` reproduces the Section 7 worked example
bit-for-bit from its tabulated inputs; `edge/lcb.py`'s block-bootstrap call
uses `L_boot=6` weeks (pinned to SYSTEM §7.3–§7.4's derivation, never an
independently-tuned duplicate); CUSUM ARL is verified by Monte Carlo against
the target `ARL_0≈52` within a stated tolerance on synthetic `N(0,1)`
sequences only; no live layer (`vol/`, `edge/`, `ceilings/`, `ratchet/`,
`combine/`) imports `validate/` (Invariant V, inherited firewall discipline).

---

## 11. Non-goals (belong to sibling modules)

- **Which instruments are long/short and their within-book relative
  weights** — SELECTION / `research/xsec/signals.py` +
  `research/xsec/portfolio.py`. LEVERAGE consumes the reference-gross book
  as given and only scales it.
- **Signal construction (the momentum lookback band, quintile vs tercile,
  the universe size N)** — the signal/universe layer
  (`research/xsec/signals.py`, `research/xsec/universe.py`). LEVERAGE does
  not choose or validate the lookback band `L ∈ [12,16]` days or the
  universe size `N≈50` — those are cited inputs from the EMPIRICAL BASELINE,
  not decisions this file makes. In particular, the decision to deploy the
  BAND rather than the single in-sample-fitted peak `L=12` is a SYSTEM/
  SELECTION decision driven by **PBO = 0.357 (range [0.31, 0.47] under
  alternative CPCV partitionings; `L=12` won in-sample on 21/28 splits)** —
  SYSTEM §4.4's three-zone PBO convention, not a LEVERAGE computation.
  LEVERAGE's ONLY interaction with this fact is indirect: `SR_LCB_t`
  (Section 2.5–2.6) is built from the bootstrap CI on the deployed BAND's
  own realized path returns, so a partition-fragile signal automatically
  produces a wider/lower CI and therefore a smaller `G_t` — the PBO evidence
  is consumed only through that channel, never re-validated here.
- **Cost/funding modelling** — COST (`src/convexity/cost/`). LEVERAGE reads
  the ALREADY-NETTED weekly return series (`research/xsec/accounting.py`'s
  `settle_week`, which nets signed funding and turnover friction into `net`)
  — it does not re-derive fee tiers, spread, impact, or funding forecasts.
- **The DSR/PSR/bootstrap machinery itself** — SYSTEM
  (`src/convexity/system/validate/deflated.py`, `.../cpcv.py`). LEVERAGE is
  a CONSUMER of `SR_LCB_t` (Invariant K-CI), never a re-implementer of the
  deflation math.
- **Execution / order placement / fill tactics** — a future EXECUTION
  pillar (out of scope here; this book rebalances weekly, so tactic
  selection matters far less than in the retired 1-minute directional
  system, but the boundary is the same: LEVERAGE emits a target `G_t`
  and per-instrument target notional, never an order).
- **Funding-carry as an independent alpha overlay.** Per the EMPIRICAL
  BASELINE, funding carry alone is a small income SATELLITE with structural
  ~70–75% drawdown and fails DSR (0.10–0.45) — it is NEVER a sizing input to
  this module's `SR_LCB_t` beyond however much it is already netted into the
  momentum book's own `net` return series via `settle_week`'s signed funding
  term. LEVERAGE does not size a separate carry sleeve.

---

## 12. References

- **Kelly criterion and fractional Kelly:** Kelly, J. L. (1956), "A New
  Interpretation of Information Rate," *Bell System Technical Journal* — the
  original growth-optimal criterion; Thorp, E. O. (2006), "The Kelly
  Criterion in Blackjack, Sports Betting, and the Stock Market" — the
  second-order Gaussian approximation and the fractional-Kelly practice this
  file's Section 2.2–2.3 derives from first principles; MacLean, L. C.,
  Thorp, E. O. & Ziemba, W. T. (2011), *The Kelly Capital Growth Investment
  Criterion* — fractional-Kelly drawdown-severity scaling and the standard
  quarter-to-half-Kelly conservative range.
- **Kelly under parameter uncertainty / plug-in bias:** Chopra, V. K. &
  Ziemba, W. T. (1993), "The Effect of Errors in Means, Variances, and
  Covariances on Optimal Portfolio Choice," *Journal of Portfolio
  Management* — the foundational result that mean-variance/growth-optimal
  weights are far more sensitive to estimation error in `μ` than in `σ²`,
  underlying Section 2.5's `Var[μ̂]=σ²/n_obs`-driven bias derivation; Baker,
  R. D. & McHale, I. G. (2013), "Optimal Betting Under Parameter
  Uncertainty: Improving the Kelly Criterion," *Decision Analysis* — the
  specific closed-form result that plug-in Kelly sized off an estimated edge
  is biased toward over-betting and that sizing off a shrunk/lower-bound
  edge estimate corrects it, the direct precedent for Section 2.5's
  `E[g(f̂*)] ≈ g(f*) − 1/(2·n_obs)` bias bound and the `SR_LCB_t`
  substitution it motivates.
- **Deflated / Probabilistic Sharpe Ratio and the track-length floor:**
  Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio";
  Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014),
  "Pseudo-Mathematics and Financial Charlatanism" (Minimum Track-Record
  Length) — the source of `SYSTEM`'s `DeflatedSharpe`/MinTRL machinery
  (SYSTEM §5.3–§5.5) that `SR_LCB_t` (Section 2.5–2.6, Invariant K-LCB/K-CI)
  is a direct consumer of.
- **Combinatorial Purged CV / PBO:** López de Prado, M., *Advances in
  Financial Machine Learning* (2018); Bailey, D. H., Borwein, J., López de
  Prado, M. & Zhu, Q. J. (2014), "The Probability of Backtest Overfitting,"
  *Journal of Computational Finance* — the CPCV path-Sharpe distribution
  (`research/xsec/cpcv.py`, SYSTEM §4) and the PBO logit-of-relative-rank
  estimator cited for the lookback-band overfitting evidence (**PBO = 0.357,
  range [0.31, 0.47] under alternative partitionings, `L=12` winning 21/28
  splits — SYSTEM §4.4**) this file's Section 2.5 numeric grounding draws on
  explicitly, at its point of use, not only here in the bibliography.
- **Sequential-test average-run-length closed form:** Hawkins, D. M. &
  Olwell, D. H. (1998), *Cumulative Sum Charts and Charting for Quality
  Improvement*, Springer, ch. 4 — the closed-form one-sided-Gaussian-CUSUM
  ARL approximation `ARL_0 ≈ exp(2·k_ref·h_promote)/(2·k_ref²)` used in
  Section 6.3 to derive `h_promote` from a target `ARL_0`.
- **Random matrix theory / Marchenko–Pastur:** Marchenko, V. A. & Pastur, L.
  A. (1967), "Distribution of eigenvalues for some sets of random matrices,"
  *Sbornik: Mathematics*; Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters,
  M. (1999), "Noise Dressing of Financial Correlation Matrices," *Physical
  Review Letters* — the eigenvalue-spreading argument (Section 4.3, P4,
  Invariant NO-MV) for refusing a fitted `Σ⁻¹μ` allocator at `N≈50,
  n_obs≈258`.
- **Covariance shrinkage:** Ledoit, O. & Wolf, M. (2004), "A well-conditioned
  estimator for large-dimensional covariance matrices," *Journal of
  Multivariate Analysis* — the shrinkage target any FUTURE multi-sleeve
  covariance extension of this module must use (Section 4.3), never a raw
  sample covariance.
- **EWMA / RiskMetrics:** J.P. Morgan/Reuters, *RiskMetrics Technical
  Document* (1996) — the exponentially-weighted volatility estimator form
  reused (with a book-specific, weekly-clock half-life) from COST §5.5a's
  funding-EWMA precedent (Section 4.1, Invariant NO-DUP-EWMA).
- **Sequential change detection / CUSUM and average run length:** Page, E.
  S. (1954), "Continuous Inspection Schemes," *Biometrika* — the sequential
  test underlying the promotion ratchet (Section 6.3), calibrated by
  average-run-length (closed form per Hawkins & Olwell 1998, above) exactly
  as OPS's own CUSUM detectors are calibrated elsewhere in this repo.
- **Moving-block bootstrap:** Künsch, H. R. (1989), "The Jackknife and the
  Bootstrap for General Stationary Observations," *Annals of Statistics*;
  Politis, D. N. & White, H. (2004), "Automatic Block-Length Selection for
  the Dependent Bootstrap," *Econometric Reviews* — the block-bootstrap CI
  construction `SR_LCB_t` is read from (Section 2.6, via
  `convexity.system.validate.robustness.block_bootstrap_ci`), sharing
  block-length discipline with SYSTEM's own derivation of the same length
  (SYSTEM §7.2–§7.4, `b* ~ c·n^{1/3}` reconciled to `b≈6` weeks at `n≈258`).
- **The empirical grounding for every numeric constant in this file** is the
  EMPIRICAL BASELINE digest supplied with this spec (crypto cross-sectional
  momentum backtest/validation results): honest net Sharpe ~0.8–1.0,
  CPCV mean OOS path Sharpe ≈0.91 vs naive 0.98 (haircut ≈0.07), `frac_pos`
  0.85–1.00, 5th-percentile path Sharpe fragility (+0.42 to −0.59), PSR≈0.994
  with `n_obs≈258` weeks short of MinTRL, max drawdown ~25% at 100% gross
  (~28% annualized vol), weekly win rate ~60%, capacity ~$3M uncapped /
  ~$12–23M at liquidity-cap tightness `C=2–3` with a ~0.4–0.6 Sharpe cost and
  a breadth fall from ~40 to 28–34 — each cited by name at its point of use
  above, never invented.
