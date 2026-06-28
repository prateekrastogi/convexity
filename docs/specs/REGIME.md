# REGIME.md — build spec for the global regime / risk-gate overlay

This file is the authoritative build spec. Build to the contracts and invariants
below. Where this file and your own instinct disagree, this file wins. Ask before
deviating from any **Invariant**.

This module is the **portfolio-level macro brake**. It sits ABOVE every
per-instrument pillar and measures the GLOBAL market state — how volatile, how
correlated, how turbulent, how liquid the whole cross-asset book's environment
is right now — and emits exactly one thing: a **causal global risk multiplier
`g_t ∈ [0,1]`** (equivalently, a small discrete regime state mapped to a
multiplier). That scalar throttles the WHOLE book's gross exposure / leverage
cap / concurrent-position count. It is the independent answer to one question,
re-evaluated on a coarse cadence:

> Is the global cross-asset environment one in which the book's diversification
> premise and survival assumptions still hold — and if not, by how much should
> the entire book's gross exposure be cut, gradually and without whipsawing?

Nothing else. REGIME does **not** size a position, does **not** taper for an
individual position's drawdown, does **not** halt on an operational fault, does
**not** pick which instruments to run, and does **not** generate or exit any
trade. It emits a number (and a state), and the sizing/selection pillars consume
it. The boundary with each neighbour is pinned in Section 10 and is load-bearing:
**REGIME multiplies (gradual, global, market-state); LEVERAGE governs per-position
(drawdown taper); OPS halts (binary, operational/statistical safety).** No
double-counting of drawdown or risk across the three.

It composes with the six pillars and **NautilusTrader** without importing their
internals:
- LEVERAGE.md consumes `g_t` as **one more term in its `min()` ceiling** and as a
  scalar on the book-heat ceiling `H_book`.
- The SELECTION module consumes `g_t` (or the discrete state) to decide **how many
  Carver-optimised candidates** to run concurrently.
- EXECUTION.md supplies the aggregate spread/depth-blowout signal REGIME reads for
  its liquidity sub-state (REGIME does not model microstructure).
- COST.md supplies the cross-book funding-extreme signal and prices the rebalance
  a regime change forces (REGIME does not model funding).
- SYSTEM.md owns the firewalled out-of-sample validation that promotes REGIME's
  thresholds; REGIME's BLUE grid feeds SYSTEM's joint deflation count.

NautilusTrader already owns order plumbing, position/account/portfolio state,
restart reconciliation, data transport, backtest↔live parity, pre-trade risk
checks, and funding ingestion. REGIME builds the **regime-state POLICY** on top of
the bars Nautilus already delivers; it never rebuilds any of that.

Scope: full cross-asset (Binance USDT-M perpetuals 24/7, Dukascopy-proxied
commodities/metals, US/Indian equities and index futures), ~50–90 assets, 1-minute
bars aggregated to a coarse regime cadence, ~$1000 retail book, target compute = a
2-core / 11 GB ARM VM. Retail latency: seconds, bar-level — never microseconds.
Every live number is `F_t`-measurable; future data appears only inside SYSTEM's
firewalled validation.

---

## 0. Governing principles (read before anything else)

**P1 — REGIME emits a scalar/state; it does not re-implement sizing, the
per-position drawdown governor, or selection.** The entire public output is
`g_t ∈ [0,1]` plus the discrete state `s_t` and a reason chain. LEVERAGE puts
`g_t` into its `min()`; SELECTION reads `s_t`/`g_t` to set the concurrent-position
count. REGIME never computes a position size, never tapers for one position's
own drawdown, never ranks an instrument, never opens/closes/reduces anything. Any
sizing math, per-position drawdown math, or selection ranking living in `regime/`
that is not pure global-state measurement is the bug. (See Invariant SCOPE.)

**P2 — REGIME measures global RISK STATE without ML, causally and cheaply.** The
four measures (Section 2) are economically-motivated statistics of the rolling
cross-asset return panel: realized-vol percentile and vol-of-vol; the **absorption
ratio** (Kritzman–Li, fraction of variance in the top-k eigenvectors of the
rolling correlation matrix); **turbulence** (Kritzman–Mancini Mahalanobis distance
of the current return vector); and an aggregate liquidity/funding-stress flag from
EXECUTION/COST. Each is a transparent, decomposable number with a stated per-bar
cost (Section 2.6). No trained model, no learned regime classifier, no hidden
Markov fit. The classic regime-overfit trap — fitting breakpoints to maximise
backtest PnL — is forbidden (P5). (See Invariant ML, Invariant CHEAP.)

**P3 — The four measures combine by a `min()`/firewall, never a product of
scores.** Each measure maps to its own sub-multiplier `g^{vol}, g^{corr},
g^{turb}, g^{liq} ∈ [0,1]`, each guarding a distinct global failure mode. The
global multiplier is the **minimum** of the four (after the state machine, Section
3): `g_t = min(g^{vol}, g^{corr}, g^{turb}, g^{liq})`. A product
(`g^{vol}·g^{corr}·…`) compounds correlated signals into an unvalidatable
over-cut and is forbidden; the `min()` is conservative-by-construction — the
binding sub-state is always the nearest global failure mode, and is reported.
(See Invariant MIN.) This mirrors LEVERAGE's `min()`-not-product (its P6) and
SYSTEM's `min()`/precedence composition (its P2).

**P4 — The multiplier is GRADUAL and STICKY, never a whipsaw.** REGIME steps the
multiplier through a small set of states with **hysteresis** (entry and exit at
different thresholds) and a **minimum dwell time** (a state, once entered, is held
for at least `D_min` decision bars before it may change). This is the regime
analogue of Carver's position buffering and EXECUTION's no-churn discipline: the
brake does not flap on a 1% wiggle in a statistic. A regime measure that crosses a
threshold for one noisy bar and reverts must NOT move `g_t`. (See Invariant HYST,
Invariant DWELL.)

**P5 — No overfitting: thresholds are economically-motivated or quantile-based,
governed RED/BLUE, validated OUT-OF-SAMPLE via SYSTEM, NEVER fit to maximise
backtest/live PnL.** REGIME thresholds are either (a) standardized-shift triggers
(a z-score / percentile of the measure against its own rolling causal history), or
(b) a small finite BLUE grid of standardized levels, promoted only through
SYSTEM's deflated-Sharpe gate over purged walk-forward / CPCV. Choosing the
breakpoint that maximised in-sample PnL is the single most common regime fraud and
is structurally forbidden: REGIME's thresholds enter SYSTEM's joint multiplicity
count (P6 there) and are paid for in the deflation. Few parameters, each justified.
(See Invariant NOFIT.)

**P6 — Causal / no-lookahead.** Every statistic uses only data at or before bar
`t`: rolling windows, causal percentiles (the percentile of today's value within
the trailing window, never the full-sample distribution), an expanding/rolling
mean and covariance for turbulence, an eigendecomposition of the trailing
correlation matrix for absorption. No future bar, no full-sample standardization,
no centred window. Future data appears ONLY inside SYSTEM's firewalled validation
labels. (See Invariant CAUSAL.)

**P7b — The brake IS a change-point detector; its thresholds carry an explicit
false-alarm / run-length budget (SPC/ARL).** A regime brake is a multivariate
statistical-process-control (SPC) change-point detector, and a threshold without a
stated false-alarm rate is unjustifiable on its own — the BLUE grid search RANKS
candidate thresholds out-of-sample but never PINS how often the brake fires on
noise. REGIME therefore states, for the canonical thresholds: (a) a per-decision-bar
**false-escalation rate** `alpha` the thresholds are chosen to hit under a calm
(in-control) null panel; (b) the in-control **Average Run Length** `ARL0` (expected
bars between false escalations) that the threshold + dwell + hysteresis together
imply — the dwell and the one-step-at-a-time hysteresis act as a run-length filter,
the discrete analogue of a CUSUM/EWMA chart's headstart and decision interval; and
(c) the out-of-control detection ARL1 (how fast a TRUE shift is caught). An ablation
reports the REALIZED escalation frequency against this `ARL0` budget. "How often
does this brake fire on noise?" must be answerable from the spec, not discovered in
production. (See Section 3.6, Invariant ARL.)

**P7 — A regime change costs money; it must clear a cost/rebalance gate.** Moving
`g_t` forces the book toward a new gross target, which means trades, which means
spread + impact + funding (COST.md). So `g_t` only changes when the move clears a
**rebalance-threshold band** (don't rebalance the whole book on a 1% regime
wiggle), and the implied book delta is priced against COST before it propagates.
The hysteresis + dwell (P4) and this cost gate together make regime changes rare,
deliberate, and paid-for. (See Invariant COSTGATE.)

Forbidden in v1: any ML / learned regime classifier or HMM (P2); a multiplicative
"regime score" instead of the `min()` (P3); a multiplier that can flap bar-to-bar
without hysteresis/dwell (P4); fitting regime breakpoints to maximise backtest or
live PnL (P5); any non-causal / full-sample standardization or percentile (P6);
re-implementing per-position sizing, the per-position drawdown governor, or
selection ranking inside REGIME (P1, Section 10); a `g_t` that can **exceed** 1.0
or otherwise INCREASE book exposure above the pillars' own ceilings (Invariant
CAP); per-symbol tuning of the RED/BLUE regime constants on realized P&L
(Section 4); rebuilding Nautilus state/reconciliation or EXECUTION microstructure
or COST funding (Section 1).

---

## 1. What REGIME builds ON TOP OF the pillars and Nautilus (and must NOT rebuild)

| Owned elsewhere (do NOT rebuild)                              | REGIME consumes it as                                          |
|--------------------------------------------------------------|----------------------------------------------------------------|
| Nautilus: bars, position/account state, reconciliation       | the cross-asset return panel; the live book gross it scales    |
| Nautilus: funding ingestion                                  | (via COST) cross-book funding extremes for `g^{liq}`           |
| EXECUTION.md: realized spread/impact, stale/halt flags        | aggregate spread/depth-blowout signal for `g^{liq}`            |
| COST.md: funding/carry forecast, funding band                | cross-book funding-stress signal for `g^{liq}`; rebalance cost |
| LEVERAGE.md: the seven ceilings, book heat `H_book`, governor | the consumer of `g_t` (one more `min()` term + heat scalar)    |
| SELECTION (Carver Ch.25): instrument ranking                 | the consumer of `s_t`/`g_t` (concurrent-position count)        |
| SYSTEM.md: purged WF / CPCV, deflated Sharpe, BLUE accounting | the validation that promotes REGIME thresholds; trial ledger   |

REGIME builds the **state layer**: the cross-asset return panel, the four
measures, their standardized triggers, the hysteretic dwell-bounded state machine,
the cost-gated multiplier emission, and the diagnostics. It is a PROVIDER of a
scalar/state — like COST it never emits a trading action.

> **Invariant NREBUILD — no neighbour re-implementation.** REGIME does not compute
> commissions, does not ingest funding from the venue, does not model
> microstructure execution, does not estimate per-instrument vol forecasts for
> sizing (it reads the panel; LEVERAGE owns the sizing vol forecast), does not
> rank instruments, and does not size positions. Tests assert REGIME's liquidity
> sub-state is a pure function of EXECUTION/COST outputs (not a re-modelled
> spread), and that nothing in `regime/` imports a sizing or selection internal.

---

## 2. REGIME measures global state without ML (the mathematical centerpiece)

The input is the **cross-asset return panel** `R_t ∈ ℝ^{W×n}`: `n` assets
(~50–90), `W` trailing bars of (causal) returns at the regime cadence, aligned and
de-meaned causally. Each of the four measures below is a transparent statistic of
that panel, each mapped to a sub-multiplier in `[0,1]` by a standardized trigger
with hysteresis (Section 3). The per-bar cost of every measure is stated and is
trivially within the 2-core budget (Section 2.6).

> **Panel hygiene (shared by all measures).** Returns are log-returns at the
> regime cadence (Section 3.5). Assets that do not trade at a given stamp (closed
> session, halt) are forward-filled to a zero return for that stamp ONLY within a
> stated staleness window, else dropped from the panel for that bar (and the panel
> dimension `n_t` shrinks — every measure is computed on the live `n_t`). 24/7
> crypto, session-bound equities, and Dukascopy-proxied commodities are aligned by
> AMBER calendar, never by forking the math. De-meaning uses the trailing causal
> mean only (P6).

### 2.1 Volatility regime — realized-vol percentile, vol-of-vol, vol term-structure

The aggregate realized volatility of the book's environment, its instability, and
its slope:

```
sigma_agg_t   = cross-asset aggregate realized vol at t
              = sqrt( mean_i  RV_i_t )           (equal-weight mean of per-asset realized variance)
              where RV_i_t is asset i's causal realized variance over a short window
              (the SAME Yang-Zhang/EWMA family LEVERAGE uses, read not re-fit; Inv. NREBUILD)

vol_pct_t     = causal percentile of sigma_agg_t within its trailing window [t-W_pct, t]
              (the percentile of TODAY's value in its OWN past, never full-sample; P6)

vov_t         = realized vol OF sigma_agg over a short window
              = std( sigma_agg over [t-W_vov, t] ) / mean( sigma_agg over same )   (a coefficient of variation)

term_t        = sigma_agg(short window) / sigma_agg(long window)                   (vol term-structure slope)
              term_t > 1  => short-term vol elevated vs baseline (stress onset)
              term_t < 1  => calm relative to baseline
```

Crypto implied-vol analog: where a venue publishes an implied-vol index (e.g.
Deribit **DVOL** for BTC/ETH), `sigma_agg` for the crypto sub-panel MAY use the
published implied vol instead of realized — AMBER-selected per asset, never a
forked code path. If DVOL is unavailable, fall back to realized (P6-causal,
conservative). The implied-vol path is a metadata switch, not new logic.

The volatility sub-multiplier `g^{vol}` is driven by `vol_pct_t` (primary), with
`vov_t` and `term_t` as confirming inputs that can only TIGHTEN it (an elevated
percentile with rising vol-of-vol and a steep term-structure is a stronger
stress signal than the percentile alone). Mapping in Section 3.

### 2.2 Correlation / systemic regime — the absorption ratio (Kritzman–Li)

The centerpiece systemic measure. When assets co-move, a few principal components
capture most of the cross-asset variance, diversification (the Carver
dynamic-optimisation premise SELECTION and LEVERAGE rely on) collapses, and a
shock propagates through the whole book at once. The **absorption ratio** (AR)
measures exactly this:

```
0. SHRINK C_t (mandatory, before the eigendecomposition; see "Conditioning C_t" below):
   C_t <- Ledoit-Wolf constant-correlation shrink of the raw rolling correlation matrix.
1. C_t = the SHRUNK n_t x n_t rolling CORRELATION matrix of the panel R_t over window W_AR (causal).
2. Eigendecompose C_t:  eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_{n_t} >= 0.
3. AR_t = ( sum_{j=1..k} lambda_j ) / ( sum_{j=1..n_t} lambda_j )
        = fraction of total cross-asset variance captured by the top-k eigenvectors.
   k = ceil(kappa * n_t),  kappa RED (Kritzman-Li use ~1/5 of assets; default kappa = 0.20).
```

**Conditioning `C_t` before the eigendecomposition (mandatory — the marginal-conditioning
fix, route (a) PINNED).** The ARFLOOR floor below (`W_AR ≥ c_AR·n_t`) guards the
rank-DEFICIENT case (`W_AR < n_t`, where `C_t` is singular). It does NOT cure the
**marginal-conditioning** regime it admits: at `n_t ≈ 87` and `c_AR = 2`, `T/N ≈ 2`,
where the sample-correlation eigenvalue spectrum suffers severe **Marchenko–Pastur
dispersion** — the top eigenvalues are inflated and the bottom depressed even on pure
i.i.d. noise. The top-`k` share `AR_t` is therefore biased HIGH on noise even when the
floor is satisfied, which would fire `g^{corr}` spuriously. The fix (pin ONE; route (a)):

> **Eigendecompose the SHRUNK `C_t`, never the raw sample correlation.** Apply
> **Ledoit–Wolf shrinkage toward the constant-correlation target** to the rolling
> correlation matrix BEFORE the eigendecomposition — the SAME closed-form, no-ML
> estimator family turbulence uses for `Sigma_t` (Section 2.3, Invariant COND), here
> applied to the correlation matrix so the trace identity (denominator `= n_t`) is
> preserved (the constant-correlation target is itself a correlation matrix — unit
> diagonal — so the shrink `δ·F + (1−δ)·S` keeps a unit diagonal and `trace = n_t`
> exactly, TRACE survives). Shrinkage pulls the inflated top eigenvalues down and lifts
> the depressed bottom ones toward the MP-debiased spectrum, so `AR_t` is no longer
> biased high on noise at marginal `T/N`. Routes (b) `c_AR ≥ 5` and (c) a
> sector/asset-class-block reduction are deliberately-rejected alternatives (b raises
> the data floor and starves the window; c discards within-block co-movement that the
> absorption ratio exists to catch); route (a) is canonical because it is the cheapest
> and is symmetric to COND.
>
> **Shrinkage-intensity governance (NOT a free parameter — it does not enter the
> overfit count).** The Ledoit–Wolf intensity `δ_AR ∈ [0,1]` is the closed-form,
> data-derived OPTIMAL shrinkage (the analytic estimator of Ledoit–Wolf 2004, the ratio
> of estimated dispersion to the expected squared error of the sample matrix), computed
> CAUSALLY from the trailing panel `R_t` over `W_AR` only — trailing-only,
> `F_t`-measurable (P6), never fit on PnL, never per-symbol. It is therefore TEAL (a
> statistic of the panel), NOT a RED/BLUE knob: it is NOT enumerated in the BLUE grid
> and does NOT contribute to `K_regime` (Section 4) or to SYSTEM's deflation count. The
> Ledoit–Wolf TARGET (constant-correlation) is the only RED choice and is pinned, not
> searched.
>
> **Cost (CHEAP).** The LW intensity is `O(n²·W)` (it reuses the same sample
> moments already formed to build `C_t`); the eigendecomposition of the shrunk `C_t`
> is the SAME `O(n³)` as the raw one (Section 2.6). No new order of work: at `n = 90`,
> still sub-millisecond on one core of the 2-core box. (See Invariant ARCOND.)

**The dispersion bias is NON-STATIONARY in `n_t` (the un-fixed TURBNORM analog —
mandatory).** Even after the shrink, a residual MP dispersion bias on `AR_t` depends
on the observations-per-dimension ratio `W_AR/n_t`, and `n_t` VARIES bar-to-bar
(staleness drops, session closes) exactly as it does for turbulence. The standardized
shift `delta_AR` (below) cancels a STATIONARY level bias (short-minus-long mean over the
raw-level `sigma_AR`), but it does NOT cancel a bias that MOVES when `n_t` moves: when
`n_t` drops mid-window, the dispersion bias on `AR_t` changes, injecting a SPURIOUS
`delta_AR` shift — the precise pathology TURBNORM fixes for turbulence (`d_t/n_t`) but
which raw absorption leaves unhandled. Pin a fix symmetric to TURBNORM:

> **Freeze `delta_AR` across any bar where `n_t` changes by more than a threshold (RED
> `frac_dn`, e.g. ±5% of `n_t`); otherwise compute `delta_AR` only over a stretch of
> CONSTANT `n_t`.** Concretely: the trailing AR-level series feeding `AR_bar_short`,
> `AR_bar_long`, and `sigma_AR` is taken over the fixed-`n_t` stratum = the set of
> ALL trailing bars whose `n_t` lies within `frac_dn` of the current bar's `n_t`
> (a current-bar-anchored band — the SAME `frac_dn` criterion that gates the
> freeze, so the pooling rule and the freeze threshold are one and the same test,
> not two), POOLED across non-contiguous sessions/days (NOT a single
> contiguous run). On a mixed crypto+equity book `n_t`
> structurally alternates (equity sub-panel present in-session vs absent overnight,
> §3.5), so each recurring panel size has its OWN persistent stratum accumulating
> across calendar days; when `n_t` steps outside the `frac_dn` band the absorption
> sub-state HOLDS its last `delta_AR`-derived `g^{corr}` (frozen, flagged
> `ar_nt_changed`, conservative — same family as the ARFLOOR freeze) until the
> stratum MATCHING the new `n_t` has enough bars to recompute `delta_AR` — and
> because strata PERSIST across days, a recurring panel size RE-USES its existing
> history on re-entry, so the freeze is TRANSIENT (a single cadence stamp at a
> genuinely new size), never a permanent freeze on a book whose `n_t` alternates
> every session. This makes
> a panel-size change unable to manufacture a `delta_AR` shift, exactly as `d_t/n_t`
> makes it unable to manufacture a `turb_pct` shift. (See Invariant ARDISPNORM.)

`AR_t ∈ [k/n_t, 1]`. High AR = assets co-move = fragile, undiversified market =
cut gross. Low AR = idiosyncratic, diversified = the book's premise holds.

**Denominator is pinned, not recomputed (the `/80.0` in Section 6).** For a
**correlation** matrix the diagonal is all ones, so `sum_{j=1..n_t} lambda_j =
trace(C_t) = n_t` EXACTLY. Therefore

```
AR_t = ( sum_{j=1..k} lambda_j ) / n_t            (the denominator is n_t, not a re-summed eigenvalue total)
```

The worked example's `Σλ_{1..80} = 80.0` is not an input to be checked against the
eigensolver — it is `n_t = 80` by identity. An implementer must use `n_t` as the
denominator and must NOT recompute `sum(all lambda)` numerically (round-off makes
the eigenvalue sum `80.0 ± ε`, which would make `AR_t` jitter at the last digit and
break the bit-for-bit fixture). (See Invariant TRACE.) This identity holds ONLY for
the correlation matrix; if a covariance matrix were ever used the denominator is
`trace(Σ) ≠ n_t` and the identity is void — REGIME uses the correlation matrix
precisely so the denominator is pinned.

**Degenerate / boundary cases (they occur on a 50–90 asset retail panel):**

- **Tie / cluster at the cutoff.** `k = ceil(kappa·n_t)` can land inside a
  degenerate eigenvalue cluster (`lambda_k = lambda_{k+1}`), where the individual
  eigenvectors are not uniquely defined. `AR_t` itself is still well-defined because
  it is a SUM of the top-`k` eigenvalues (a sum over a cluster is basis-invariant
  even when the individual vectors are not), so no tie-break on eigenvectors is
  needed for `AR_t`. Pin the eigenvalue ordering as **descending with NaN/negative
  clamped to 0** (a symmetric PSD correlation matrix has `lambda_j ≥ 0` in exact
  arithmetic; tiny negative eigenvalues from round-off are clamped to 0 before the
  top-`k` sum). Any per-eigenvector diagnostic (e.g. a named "dominant factor") is
  log-only and is explicitly marked unstable inside a cluster; it never feeds
  `AR_t`. (Invariant TRACE.)
- **Rank-deficient window (`W_AR < n_t`).** AR has the SAME observations-per-dimension
  problem turbulence guards with `c_cov`: if the causal correlation window has fewer
  bars than assets, `C_t` is rank-deficient, at most `W_AR` eigenvalues are nonzero,
  and the top-`k` share is biased HIGH (a spurious "everything co-moves" reading
  that would fire `g^{corr}` on thin data, not on real co-movement). Section 2.2
  therefore imposes a floor symmetric to COND's: require `W_AR ≥ c_AR · n_t` (RED,
  `c_AR ≥ 2`, same spirit as `c_cov`); if the live `n_t` rises so that
  `W_AR < c_AR · n_t` (e.g. the universe grows or the window is shortened),
  **fall back to ABSORPTION-SUSPENDED**: hold `g^{corr}` at its last validly-computed
  value (frozen, not 1.0 — freezing is conservative; resetting to 1.0 would release
  the brake on a data shortfall) and flag `ar_window_deficient`. A rank-deficient
  raw top-`k` share is never emitted as a live trigger. (See Invariant ARFLOOR.)

**The trigger is a standardized SHIFT, not a raw level** (P5, P6). Raw AR drifts
slowly with the universe; what signals stress is a sharp RISE relative to its own
recent history. Use the Kritzman–Li standardized shift:

```
AR_bar_short_t = trailing mean of AR over a SHORT window (W_ARshort, e.g. ~15 bars)
AR_bar_long_t  = trailing mean of AR over a LONG  window (W_ARlong,  e.g. ~250 bars)
sigma_AR_t     = trailing std of the RAW AR LEVEL series over the LONG window
                 (the std of single-bar AR values, NOT the standard error of the
                 short-minus-long mean difference — see the labelling note below)
delta_AR_t     = ( AR_bar_short_t - AR_bar_long_t ) / sigma_AR_t
                 (the Kritzman-Li STANDARDIZED DISPLACEMENT — NOT a z-score of a mean difference)
```

> **Labelling note (read before "fixing" `delta_AR`).** `delta_AR_t` is the
> Kritzman–Li standardized displacement, by construction scaled by `sigma_AR_t`,
> the std of the **raw single-bar AR level** over the long window. It is **NOT** a
> z-score of the difference of two means. A true z of `(AR_bar_short − AR_bar_long)`
> would divide by the standard error of that difference,
> `SE ≈ sigma_AR · sqrt(1/W_ARshort + 1/W_ARlong)` (with `W_ARshort≈15, W_ARlong≈250`,
> `SE ≈ sigma_AR · sqrt(1/15 + 1/250) ≈ sigma_AR · 0.267`), which is `~3.7×` smaller
> than `sigma_AR`. Under that (wrong) definition the worked example's `+2.20` would
> read as `~+8.2` "standard errors" — a different number on a different scale. The
> raw-level scaling is the **intended Kritzman–Li convention** (a displacement
> measured in units of normal day-to-day AR variation, which is what an
> economically-meaningful regime shift should be measured in), so the MATH here is
> correct. An implementer who "corrects" the denominator into the mean-difference SE
> would silently shrink the denominator `~3.7×`, inflate every `delta_AR` by the
> same factor, and thereby retune every correlation threshold in Sections 3.2/6 —
> which is forbidden (it is a per-PnL retune of a RED constant by stealth). Keep the
> raw-level std. (See Invariant DISP.)

`delta_AR_t` crossing a RED/BLUE standardized threshold (e.g. `> +1.0`) is the
correlation-regime trigger. Pin the window, `k` (via `kappa`), and the threshold;
all are RED or BLUE (Section 4), never per-symbol, never PnL-fit. The
correlation sub-multiplier `g^{corr}` steps down as `delta_AR_t` rises through the
state bands (Section 3). This is the measure that most directly defends the
diversification assumption that SELECTION's Carver optimisation and LEVERAGE's
MCTR portfolio ceiling both depend on (LEVERAGE P4: in a crisis correlations go to
1 and netted portfolio margin explodes).

### 2.3 Turbulence — Kritzman–Mancini Mahalanobis distance

Absorption flags a STRUCTURAL shift (the co-movement geometry changed).
Turbulence flags a STATISTICALLY UNUSUAL DAY: a current cross-asset return vector
that is far from its historical mean given the historical covariance — large
moves, in unusual combinations, relative to normal.

```
y_t          = the current cross-asset return vector (1 x n_t) at bar t
mu_t         = trailing causal mean return vector over window W_turb
Sigma_t      = trailing causal covariance matrix over window W_turb (the SAME panel covariance family
               LEVERAGE uses for MCTR; read/aligned, regularized as below)
d_t          = ( y_t - mu_t ) Sigma_t^{-1} ( y_t - mu_t )^T          (squared Mahalanobis distance)
d_norm_t     = d_t / n_t                                             (per-dimension turbulence; dof-normalized)
turb_pct_t   = causal percentile of d_norm_t within its trailing window   (P6)
```

**Why `d_t` is normalized by `n_t` before the percentile (mandatory).** Under
multivariate normality `d_t ~ chi^2_{n_t}`, whose MEAN is `n_t` and whose spread
grows with `n_t`. But `n_t` VARIES bar-to-bar: a session close drops the equity
sub-panel, a halt drops an instrument, the universe is rebalanced — and each such
change shifts the scale of `d_t` (a chi-square with changing degrees of freedom).
If the raw `d_t` were fed to a trailing percentile, the percentile would become
**non-stationary precisely when the universe composition shifts** — a turbulence
escalation could be an artifact of `n_t` dropping (fewer assets ⇒ smaller expected
`d_t`, so a normal vector looks unusually small; or the reverse on a re-expansion)
rather than a genuinely unusual return vector. Dividing by `n_t` gives
`E[d_norm_t] ≈ 1` regardless of `n_t` (the per-dimension Mahalanobis distance),
which is approximately scale-stable across panel-size changes, so the trailing
causal percentile compares like with like. (`d_t/n_t` is the cheapest stationarizer;
a stricter alternative — computing the percentile within fixed-`n_t` strata — is a
BLUE option but `d_t/n_t` is the RED default.) A turbulence trigger must therefore
key off `turb_pct_t` of `d_norm_t`, NEVER of the raw `d_t`. (See Invariant TURBNORM.)

`d_t` is large on days that are both big AND unusually-combined (e.g. equities and
gold both crashing together when they normally diverge). Under multivariate
normality `d_t ~ chi^2_{n_t}`, giving an economically-motivated reference, but we
trigger on the **causal percentile** of `d_t` (distribution-free, robust to fat
tails) rather than the chi-square tail, because cross-asset returns are not
Gaussian. The turbulence sub-multiplier `g^{turb}` steps down as `turb_pct_t`
exceeds its bands (Section 3).

**Covariance conditioning (mandatory, since we invert `Sigma_t`).** With `n_t` up
to ~90 and a finite window, `Sigma_t` can be ill-conditioned or singular. Apply
**Ledoit–Wolf shrinkage** toward a scaled-identity / constant-correlation target
before inversion (a closed-form, no-ML, no-tuning shrinkage), and require
`W_turb >= c_cov * n_t` (RED, e.g. `c_cov >= 2`) so the window has enough
observations per dimension; if not, drop to a robust diagonal (per-asset variance)
fallback and flag it. A raw unshrunk inverse on a near-singular matrix produces a
nonsense turbulence spike and is forbidden. (See Invariant COND.)

**Condition-number / min-eigenvalue sanity gate (mandatory, on BOTH matrices, AFTER
shrinkage).** Shrinkage USUALLY conditions the matrix, but does not GUARANTEE it: a
pathological panel (a near-duplicated pair of assets, a degenerate window) can leave
the shrunk matrix near-singular. So both the shrunk turbulence `Sigma_t` and the
shrunk absorption `C_t` pass a sanity gate AFTER shrinkage, computed from the
eigenvalues already available:

```
cond_t = lambda_max / lambda_min   (or for Sigma_t, computed from the same decomposition)
GATE: cond_t <= cond_max           (RED, e.g. cond_max = 1e4; equivalently lambda_min >= lambda_max / cond_max)
```

If the gate is BREACHED after shrinkage:
- **Turbulence `Sigma_t`:** do NOT invert — fall back to the robust diagonal
  (per-asset variance) inverse, flag `cond_breach_turb`. (Inverting a `cond > 1e4`
  matrix amplifies noise into a nonsense `d_t` spike — the COND failure mode.)
- **Absorption `C_t`:** do NOT emit a fresh `g^{corr}` trigger — FREEZE `g^{corr}` at
  its last valid value (same conservative freeze as ARFLOOR/ARDISPNORM), flag
  `cond_breach_corr`. (A near-singular `C_t` has an MP-degenerate spectrum that biases
  `AR_t` regardless of the shrink.)

`cond_number` is recorded in `SubState.diag` for both matrices (it was already a diag
field — Section 8, `SubState.diag` — this operationalizes it). The threshold `cond_max` is RED. (See
Invariant CONDNUM.)

### 2.4 Liquidity / funding-stress regime — aggregated from EXECUTION and COST

The fourth measure is not computed from the price panel; it is **aggregated from
the neighbour pillars** (P2, Invariant NREBUILD). Two inputs:

```
spread_blowout_t = aggregate over the live book of EXECUTION's realized spread / depth
                   signal, expressed as a causal percentile of each instrument's own
                   trailing spread, then aggregated (e.g. fraction of book whose
                   spread percentile exceeds a high level, OR equal-weight mean spread
                   z-score). EXECUTION already flags stale-book / halt; those force
                   the instrument's contribution to the blowout high.
funding_extreme_t= aggregate over the live book of COST's funding/carry forecast
                   magnitude, as a causal percentile of each instrument's own trailing
                   funding (signed-magnitude; an extreme funding regime — perp funding
                   spiking, contango blowing out — is a stress signal regardless of side).
liq_stress_t     = max( spread_blowout_t , funding_extreme_t )      (the worse of the two; min()-style)
```

> **Null distribution of `liq_stress` (pins the 3.6 alpha).** Because `liq_stress` is
> the MAX of two ~uniform causal percentiles, it is NOT itself uniform under the calm
> null: `P[liq_stress > p] = 1 − p²` under independence (larger under positive
> dependence). Its per-bar false-alarm `alpha` is therefore derived from the
> `1 − p²` form (Section 3.6a), NOT from `1 − p`. **BLUE alternative:** define
> `liq_stress` as a SINGLE causal percentile of a combined spread+funding stress
> signal (e.g. the percentile of `max(spread_z, funding_z)` as ONE series, or of a
> weighted sum), which restores a uniform null and the simpler `1 − p` alpha — a
> finite BLUE choice, not a per-symbol knob. RED default keeps the max-of-two.

The liquidity sub-multiplier `g^{liq}` steps down as `liq_stress_t` exceeds its
bands. This is the regime-level read of the same microstructure/funding stress
that EXECUTION's rate-limit and COST's conservative-fallback handle at the
per-trade level — REGIME aggregates it to a book-wide brake. REGIME does NOT
re-model spread or funding; it reads the neighbours' numbers (Invariant NREBUILD).

### 2.5 Combining the four — `min()`, never a product (P3)

After each measure is mapped through the state machine (Section 3) to its
sub-multiplier, the global multiplier is the minimum:

```
g_raw_t = min( g^{vol}_t , g^{corr}_t , g^{turb}_t , g^{liq}_t )
```

`g_raw_t` then passes through the cost gate (Section 5) to produce the emitted
`g_t`. The `min()` means the single nearest global failure mode binds, and the
**binding sub-state** (`vol | corr | turb | liq`) is always reported in the reason
chain. A product is forbidden (Invariant MIN): two moderately-elevated measures
must not multiply into an over-cut; if a measure is not at a stress level its
sub-multiplier is 1.0 and it does not bind.

### 2.6 Per-bar compute cost (CHEAP, on a 2-core / 11 GB ARM box)

All costs are per **regime decision bar** (Section 3.5: coarse, e.g. hourly or
session, not per 1-minute bar), on `n ≈ 50–90` assets, `W` a few hundred bars.

| Measure        | Dominant op                                   | Cost                | Notes |
|----------------|-----------------------------------------------|---------------------|-------|
| Volatility     | mean of `n` per-asset RVs + rolling percentile| `O(n·W)`            | RVs read from LEVERAGE's forecast, not re-fit |
| Absorption     | `n×n` correlation + symmetric eigendecomp     | `O(n²·W)` build, `O(n³)` eig | `n=90 ⇒ n³ ≈ 7.3e5` flops, sub-millisecond |
| Turbulence     | shrink + invert `n×n` cov, one quadratic form | `O(n³)` invert      | Ledoit–Wolf shrink + one solve; `n=90` trivial |
| Liquidity      | aggregate `n` neighbour scalars               | `O(n)`              | pure read of EXECUTION/COST outputs |

Total per decision bar is dominated by the `O(n³)` eigendecomposition and matrix
inverse: at `n=90` that is order `10^6` flops — **microseconds to a millisecond**
on one core. On an hourly cadence this is utterly negligible. The expensive part
is not the linear algebra; it is keeping the panel and the trailing windows
updated, which is `O(n·W)` per bar and equally trivial. There is no per-1-minute
matrix work: the panel updates cheaply and the eigendecomposition runs only on the
coarse regime cadence. (See Invariant CHEAP.)

> **Invariant CHEAP — regime measurement is `O(n³)` per coarse bar and never
> per-minute.** The eigendecomposition and covariance inverse run only on the
> regime decision cadence (Section 3.5), at `O(n³)` on `n ≤ ~90`. A test asserts
> no eigendecomposition / matrix inverse runs on the 1-minute path, and that one
> full regime recompute at `n=90` completes well under the retail latency budget
> on the target box.

---

## 3. The state machine — hysteresis + minimum dwell time

The four measures could each, in principle, emit a continuous sub-multiplier. We
deliberately discretize into a small **ordered set of global states** with
hysteresis and dwell, because a continuous multiplier driven by noisy statistics
WHIPSAWS, and every change costs money (P7). Discrete states with sticky
transitions are the regime analogue of Carver's position buffering and
EXECUTION's no-churn discipline (P4).

### 3.1 The state set and the multiplier map (RED)

A small, ordered, economically-named state set. Default four states:

```
state s      g(s)   meaning
-----------  -----  --------------------------------------------------------------
CALM         1.00   diversified, normal vol, normal liquidity — full gross
NORMAL       0.75   mild elevation in one measure — modest brake
STRESSED     0.50   clear systemic/vol/turbulence/liquidity stress — half gross
CRISIS       0.25   severe co-movement / turbulence / liquidity blowout — quarter gross
```

The state set and the `g(s)` map are **RED** (shared, not per-symbol). The values
deliberately mirror LEVERAGE's drawdown-governor ladder (`1.00/0.75/0.50/0.25`)
for legibility, but they are a SEPARATE control on a SEPARATE axis (market state,
not equity drawdown) and are not double-counted (Section 10, Invariant NODOUBLE).
REGIME never emits a state below `CRISIS` / `g < 0.25`: taking the book fully flat
is OPS's binary kill-switch job (Section 10), not REGIME's gradual brake. (See
Invariant FLOOR.)

Each sub-state (`vol/corr/turb/liq`) is independently classified into the same
ordered set via its measure's bands; the **global state is the WORST (lowest-`g`)
sub-state** — the discrete form of the `min()` (P3, Section 2.5). The binding
sub-state is reported.

> **Two clocks, pinned (the per-sub-state band logic vs the global dwell — read
> before implementing the state machine).** REGIME runs hysteresis at TWO levels
> and they must not be conflated:
> 1. **Hysteresis is PER-SUB-STATE.** Each of the four measures has its OWN
>    entry/exit memory: `s^{vol}, s^{corr}, s^{turb}, s^{liq}` are four independent
>    hysteretic classifications (Section 3.2 bands applied to that measure's own
>    value, with that measure's own remembered band). A measure dithering across
>    its own band does not move its own sub-state — and the other three measures'
>    memories are untouched.
> 2. **Dwell is GLOBAL, on `s_t = min over sub-states`.** There is exactly ONE
>    dwell clock, on the GLOBAL state `s_t` (the worst sub-state), NOT four
>    per-measure dwell clocks. The dwell clock counts decision bars since the
>    global `s_t` last changed IDENTITY.
> The remaining ambiguities are pinned in 3.3 (when does the global clock reset)
> and exercised by the worked multi-measure handoff in Section 6.2.

### 3.2 Hysteresis — entry and exit at different thresholds (PER sub-state)

Each measure has, per state boundary, an **entry threshold** (to escalate to a
more-stressed state) and a **looser exit threshold** (to de-escalate back). The
gap between them is the hysteresis band: you enter STRESSED when the measure
crosses the high mark, but you do not leave STRESSED until it falls back below a
distinctly lower mark. This prevents flapping at a single threshold.

Pinned defaults (standardized; RED for the canonical set, BLUE for the validated
grid around them — Section 4). Using the absorption-ratio standardized shift
`delta_AR` (Section 2.2) as the worked example; volatility uses `vol_pct`,
turbulence uses `turb_pct`, liquidity uses `liq_stress` percentile — same band
structure, different units:

```
boundary               ENTER (escalate) when      EXIT (de-escalate) when      hysteresis gap
---------------------  -------------------------  ---------------------------  --------------
CALM    -> NORMAL       delta_AR >= +0.5           delta_AR <  +0.2             0.3
NORMAL  -> STRESSED     delta_AR >= +1.0           delta_AR <  +0.6             0.4
STRESSED-> CRISIS       delta_AR >= +2.0           delta_AR <  +1.4             0.6
```

(For `vol_pct`/`turb_pct`/`liq_stress`, the analogous entry/exit pairs are in
percentile units, e.g. NORMAL@`70/60`, STRESSED@`85/75`, CRISIS@`95/88` — RED
canonical, BLUE grid around them.) Escalation may jump multiple states in one bar
(a true crisis prints `delta_AR > +2.0` directly → CRISIS); **de-escalation is
one step at a time**, each step gated by THIS measure's own exit threshold ALONE
(the per-sub-state band — NOT the global dwell clock). The global dwell clock of
3.3 is a SEPARATE, second gate that applies only to de-escalation of the GLOBAL
state `g(s_t)` (3.3 and the "two clocks" box above), never to these per-sub-state
band moves — so a sub-state may step down on its own exit threshold while the
global state is still dwell-locked (exactly the t2 `corr` handoff in worked
example 6.2). Asymmetry is deliberate: enter stress fast,
leave it slowly. (See Invariant HYST.)

### 3.3 Minimum dwell time (GLOBAL clock, on `s_t`)

A state, once entered, is held for at least `D_min` decision bars (RED, e.g. 3
decision bars) before any transition — escalation OR de-escalation — is permitted,
EXCEPT escalation to a MORE-stressed global state is always allowed immediately
(you may always brake harder without waiting). De-escalation is blocked until both
(a) the global dwell clock `>= D_min` AND (b) the binding measure's exit threshold
is satisfied. This guarantees the book is never repeatedly re-grossed-and-cut on a
measure oscillating around a threshold.

**The three reset questions, pinned (without these the machine is non-deterministic
at exactly the multi-measure transitions it exists to handle, and Invariant DET
cannot be tested):**

- **(i) Hysteresis is applied per-sub-state; dwell is applied to the global `min`.**
  Each measure keeps its own entry/exit band memory (3.2). The single dwell clock
  lives on the global `s_t` only. There are NOT four dwell clocks.
- **(ii) The global dwell clock resets IFF the global state `g(s_t)` CHANGES VALUE —
  NOT when the binding sub-state changes IDENTITY at unchanged `g(s_t)`.** The clock
  is keyed on the multiplier level `g(s_t)`, not on which measure is binding. So if
  `corr` de-escalates out of CRISIS into STRESSED while `vol` simultaneously
  escalates into STRESSED — the global `min` stays at STRESSED (`g = 0.50`
  throughout) but the binding identity hands off `corr → vol` — the dwell clock
  **CONTINUES, does not reset.** (Rationale: the book's gross target did not move,
  so no re-gross/re-cut happened, so there is nothing for the dwell to protect
  against; resetting here would needlessly delay a later legitimate de-escalation.)
  The binding-identity handoff IS recorded in the reason chain (`binding` changes,
  transition logged as `hold` with a `binding_handoff` reason code), but `dwell_bars`
  keeps counting. The clock resets ONLY when `g(s_t)` itself steps to a new level.
- **(iii) A non-binding sub-state's oscillation can NEVER reset the global clock.**
  Because the clock is keyed on `g(s_t) = min over sub-states`, a sub-state that is
  not the current minimum can dither across its own bands freely (escalate NORMAL↔
  STRESSED, say) WITHOUT touching `g(s_t)` as long as it stays above the binding
  sub-state's level — and therefore without resetting the dwell clock or forcing any
  rebalance. Only a move that changes the `min` (a new lower sub-state, or the
  current binding sub-state moving) can change `g(s_t)`, and only a change in
  `g(s_t)` resets the clock. This is what makes a multi-measure panel deterministic:
  the clock sees only the scalar `g(s_t)`, never the four-way identity.

In short: **escalation of the global `min` is immediate and resets the clock;
de-escalation requires `dwell ≥ D_min` AND the binding measure's exit threshold,
one step at a time; a binding-identity handoff at constant `g(s_t)` neither resets
the clock nor counts as a transition; a non-binding measure's wiggle is invisible to
the clock.** (See Invariant DWELL, Invariant CLOCK.)

```python
class RegimeStateMachine:
    def step(self, sub_states: dict[str, "SubState"], prev: "RegimeState",
             cfg: "ResolvedConfig") -> tuple["RegimeState", dict]:
        """sub_states maps {'vol','corr','turb','liq'} -> SubState (current measure
        value + its OWN per-sub-state hysteretic band classification, 3.2).
        Global state s_t = worst (lowest-g) sub-state. ONE global dwell clock on
        g(s_t):
        - escalation of the global min to a more-stressed state: ALWAYS allowed
          (brake harder now); resets the dwell clock.
        - de-escalation: only if dwell_bars >= D_min AND the binding measure's exit
          threshold met, ONE step at a time; resets the dwell clock.
        - binding-identity handoff at UNCHANGED g(s_t) (e.g. corr->vol both at
          STRESSED): transition='hold', reason 'binding_handoff', dwell_bars
          CONTINUES (NOT reset) -- 3.3(ii).
        - a non-binding sub-state oscillating below the min: invisible to g(s_t),
          no clock reset -- 3.3(iii).
        Returns (new RegimeState carrying g(s_t), binding sub-state, dwell_bars,
        transition reason) and diagnostics. Pure function of inputs + prev; no
        hidden state, deterministic, no future data (P6)."""
```

### 3.4 What `g_t` feeds (the output contract restated)

```
g_t  ->  LEVERAGE.md :  scales the book-heat ceiling H_book -> g_t * H_book
                        (and thereby enters the per-position min() through the
                        H_book-derived portfolio ceiling). THIS IS THE GROSS CHANNEL.
g_t / s_t -> SELECTION : caps the concurrent-position COUNT N_concurrent(s_t)
                        (e.g. CALM=full N, NORMAL=0.75N, STRESSED=0.50N, CRISIS=0.25N,
                        floored at the incumbents already held — REGIME never force-closes;
                        it throttles NEW admissions, P4-incumbent-respect inherited from SYSTEM).
                        THIS IS A STRUCTURAL/DIVERSIFICATION CHANNEL, NOT a second gross multiplier.
```

> **Only ONE channel carries the gross multiplier (the `g_t² over-cut` trap,
> pinned).** `g_t` reduces book gross through the LEVERAGE `H_book` path ONLY. The
> SELECTION count cap `N_concurrent(s_t)` is a **structural** control (how many
> independent bets the book is allowed to hold, defending diversification), NOT a
> second multiplier on gross. The two MUST NOT compound: if both naively multiplied
> gross, CRISIS would cut heat by `g_t = 0.25` AND count by `~0.25`, for a realized
> gross of `~g_t² = 0.0625` — a `16×` cut where the spec promises a `4×` cut. That is
> a NODOUBLE-style problem WITHIN REGIME's own two output channels and is forbidden.
> The contract is:
> - **Gross is set by the LEVERAGE `H_book` scaling alone.** `target_gross = g_t ·
>   book_gross_at_full`. This is the single, auditable gross response and the
>   number the cost gate (Section 5) and the worked example (Section 6) price.
> - **The count cap re-DISTRIBUTES that same gross across fewer names; it does not
>   re-cut the total.** At CRISIS the book runs `0.25·N` names at the
>   `g_t`-reduced gross, so each surviving name carries a LARGER share of a SMALLER
>   book — the per-name notional need not fall by `g_t²`. The count cap binds on
>   admissions/concentration, the heat scaling binds on total gross; they act on
>   orthogonal quantities (count vs gross), exactly as REGIME-vs-governor act on
>   orthogonal axes (Section 10).
> - **Edge case — count cap forces gross below `g_t·full`.** If incumbents already
>   held exceed `N_concurrent(s_t)`, REGIME does NOT force-close (P4 incumbent
>   respect); the book simply admits no new names until attrition. If the surviving
>   `N_concurrent` names cannot absorb the full `g_t·book_gross_at_full` without
>   each breaching its own LEVERAGE per-position ceiling, the realized gross falls
>   BELOW `g_t·full` — but that shortfall is driven by the per-position ceilings,
>   not by a second `g_t` factor, and it is reported by LEVERAGE's binding-ceiling
>   telemetry, not modelled as a `g_t²` in REGIME. The combined gross response is
>   therefore **bounded as `realized_gross ≤ g_t · book_gross_at_full`, with the
>   count cap able only to LOWER it further via per-position ceilings, never via a
>   second multiplicative `g_t`.** (See Invariant NODOUBLE2.)

**REGIME emits; the pillars consume.** REGIME does not itself multiply any size or
drop any candidate — it publishes `g_t` and `s_t`, and LEVERAGE/SELECTION read
them. The exact consumption is pinned in their specs; REGIME's contract is only to
emit a valid, in-range, sticky scalar/state. (See Invariant SCOPE, Invariant CAP.)

### 3.5 Regime cadence (coarse, by design)

Regime is a SLOW variable. Recomputing it every 1-minute bar both wastes compute
and invites whipsaw. The decision cadence is governed:

- **Scheduled recompute (cadence `tau_regime`, RED):** the full four-measure +
  state-machine evaluation runs on a coarse cadence — e.g. hourly for 24/7 crypto,
  or session-boundary + a few intraday stamps for session-bound assets. Pinned in
  AMBER calendar terms, never per-minute.
- **Event-driven escalation (always allowed, but LIQ-ONLY, and minute-dwell-gated):**
  a sharp intraday EXECUTION spread-blowout / halt cascade or a COST funding spike MAY
  trigger an immediate out-of-cadence ESCALATION (never a de-escalation — de-escalation
  always waits for the scheduled cadence + dwell). To keep this fast path from
  false-firing on a single-minute blip (it evaluates every 1-minute bar, `~60×` the
  decision-bar opportunities — Section 3.6 b′), it must clear a **minute-level dwell**:
  `liq_stress` stays above the entry mark for `D_liq_min` consecutive 1-minute bars
  (RED) before it moves `g_t`. This minute run-length filter, plus the separate
  per-minute false-alarm budget `alpha^liq_min`/`ARL0^liq_min` (Section 3.6 b′), is
  what lets the brake engage fast on a SUSTAINED flash event without firing on noise or
  re-grossing fast.
- **Measurement asymmetry between recomputes (pinned).** Only the liquidity
  sub-state is cheap enough to recompute out-of-cadence: `g^{liq}` is `O(n)` over
  EXECUTION/COST scalars and is FRESH every minute. The price-panel measures
  (`g^{vol}, g^{corr}, g^{turb}`) carry their LAST scheduled-cadence value — they
  are STALE between recomputes (the `O(n²·W)`/`O(n³)` work is cadence-gated,
  Invariant CHEAP). Two consequences are load-bearing and must be enforced:
  - (a) an out-of-cadence event escalation is computed from the **fresh `g^{liq}`
    and the FROZEN last `g^{vol/corr/turb}`** — the min() still runs, but only the
    liq term can have moved since the last scheduled bar;
  - (b) therefore an out-of-cadence escalation can ONLY be reported with
    `binding = liq`. It is FORBIDDEN to attribute a flash escalation to `corr`,
    `vol`, or `turb` off a stale value — those measures were not recomputed, so the
    reason chain would be falsely attributing the escalation to a measure that did
    not move. If the min() at an out-of-cadence bar appears to be bound by a stale
    `corr/vol/turb` value, that only means a *prior scheduled* bar already
    classified that sub-state lower; the out-of-cadence step itself can only LOWER
    the state via `liq`, and the transition reason names `liq`. The frozen measures'
    last values and their `asof_bar` are carried in `SubState.diag` so the staleness
    is auditable. (See Invariant STALEATTR.)
- The 1-minute path only UPDATES the panel cheaply (`O(n·W)`) and recomputes
  `g^{liq}` (`O(n)`); the `O(n³)` work is cadence-gated (Invariant CHEAP).

### 3.6 SPC / ARL — the false-alarm and run-length budget the thresholds are chosen to hit (P7b)

The state machine of 3.1–3.3 IS a multivariate change-point detector: four
standardized statistics, each with an escalation threshold (the entry mark), a
de-escalation threshold (the looser exit mark — the hysteresis gap), and a global
dwell `D_min`. Statistical-process-control gives the language to say what those
levels MEAN on noise. This subsection pins the budget; the BLUE grid (Section 4) is
searched OOS *within* this budget, never to maximise PnL (P5).

**(a) Per-decision-bar false-escalation rate `alpha` (the design target).** Define
the in-control (calm, null) panel as one drawn from REGIME's own trailing
distribution with no true regime change. The canonical thresholds are chosen so
that, per decision bar, the probability that ANY of the four sub-states escalates
its band is a small target `alpha` (RED design target, e.g. `alpha ≈ 0.01` per
decision bar for the first escalation rung CALM→NORMAL — roughly a "1-in-100 calm
bars" false trip before dwell/hysteresis filtering). The threshold levels are read
DIRECTLY off this target, because each measure's trigger is a causal percentile or
standardized displacement against its OWN trailing null:

```
For a SINGLE-PERCENTILE measure (vol_pct, turb_pct):
  a band entry at the p-th causal percentile fires on a calm bar with prob (1 - p/100)
  BY CONSTRUCTION (the percentile is uniform on [0,1] under the null).
  => CALM->NORMAL entry at the 70th pct  => per-measure false-entry rate alpha_1 = 0.30 ... (raw)
     NORMAL->STRESSED at 85th            => alpha_2 = 0.15  (raw)
     STRESSED->CRISIS at 95th            => alpha_3 = 0.05  (raw)
  These RAW per-bar rates are deliberately loose; the dwell + hysteresis (b) below
  collapse them into the actual run length. (Per-measure; the union over 4 measures
  multiplies the calm-bar trip probability by <=4 -- a Bonferroni-style ceiling,
  bounded because under the calm null the four measures are only weakly dependent.)

For liq_stress = max(spread_blowout_pct, funding_extreme_pct) (Section 2.4): the MAX
  of two ~uniform percentiles is NOT uniform, so its band-entry false-alarm rate is
  NOT (1 - p/100). Under the null with the two sub-percentiles INDEPENDENT,
        P[max > p/100] = 1 - (p/100)^2,
  and under POSITIVE dependence (spread and funding stress tend to co-occur) it is
  LARGER (bounded above by 1 - p/100, the comonotone limit where the max degenerates
  to a single percentile). So liq's per-bar raw false-entry rate at the p-th band is:
     CALM->NORMAL at p=70  => alpha_1^liq = 1 - 0.70^2 = 0.51  (independent null)
     NORMAL->STRESSED p=85 => alpha_2^liq = 1 - 0.85^2 = 0.2775
     STRESSED->CRISIS p=95 => alpha_3^liq = 1 - 0.95^2 = 0.0975
  i.e. ROUGHLY DOUBLE the single-percentile rate at high p (since 1-p^2 = (1-p)(1+p)
  and (1+p) -> ~1.95 at p=0.95). The independence form is the DESIGN assumption stated
  here; the realized dependence is audited in Section 9 (positive dependence only makes
  the realized rate higher than 1-p^2, which the audit will catch). The dwell +
  hysteresis (b) collapse THIS raw rate into the liq run length using alpha^liq, NOT the
  single-percentile alpha. (BLUE alternative, pinned in Section 2.4: define liq_stress as
  a single causal percentile of a COMBINED stress signal, restoring the uniform null and
  the 1-p/100 form; the RED default keeps the max-of-two with the 1-p^2 null.)

For the DISPLACEMENT measure (delta_AR): under a calm null, delta_AR is
  approximately mean-zero with unit-ish scale in sigma_AR units (it is a
  standardized displacement, Section 2.2 labelling note). A +1.0 entry is ~1 raw
  std => one-sided normal tail ~0.16 raw per-bar; +2.0 => ~0.023 raw per-bar.
```

**(b) The implied in-control ARL0 (threshold + dwell + hysteresis = a run-length
filter).** A single-bar false trip does NOT move `g_t`: escalation moves the global
state immediately, but the *cost* of a false escalation is the unnecessary gross cut
held until de-escalation, which is gated by `D_min` AND the looser exit threshold.
The hysteresis gap + one-step-at-a-time de-escalation is the discrete analogue of a
CUSUM decision interval: a transient spike escalates, then must persist or it
de-escalates after `D_min` bars once the measure falls back below the exit mark. The
in-control Average Run Length — expected decision bars between FALSE escalations
into a given state — is bounded below by the geometric run length of the raw rate,
floored by the dwell:

```
ARL0(state k)  ~  1 / alpha_k_effective       (expected bars between false entries to state k)

where alpha_k_effective is the per-bar false-entry rate AFTER the hysteresis gap is
applied: a false entry requires the measure to cross the ENTRY mark, and a false
SUSTAINED escalation (the costly event) requires it to STAY above the EXIT mark for
the dwell. Treating bars near the null as ~independent, the probability a single
false entry survives D_min bars above the (looser) exit mark is ~ (P[> exit])^(D_min-1),
so:

  alpha_k_effective  ~  P[cross entry] * (P[stay > exit])^(D_min - 1)
  ARL0(state k)      ~  1 / alpha_k_effective   (in DECISION bars; multiply by tau_regime for wall-clock)
```

Worked ARL0 for CALM→NORMAL on a percentile measure, entry@70 / exit@60, `D_min=3`:
`P[cross entry] = 0.30`, `P[stay > exit] = 0.40`, so `alpha_eff ≈ 0.30·0.40² = 0.048`
per decision bar ⇒ `ARL0 ≈ 1/0.048 ≈ 21` decision bars between sustained false
NORMAL trips. For NORMAL→STRESSED (entry@85 / exit@75): `0.15·0.25² ≈ 0.0094` ⇒
`ARL0 ≈ 106` decision bars. For STRESSED→CRISIS (entry@95 / exit@88):
`0.05·0.12² ≈ 7.2e-4` ⇒ `ARL0 ≈ 1390` decision bars. On an hourly cadence
(`tau_regime = 1h`), a false CRISIS is expected roughly once per `~1390 h ≈ 58`
trading days of pure calm — the brake does not cry crisis on noise. The dwell and
hysteresis are what turn raw per-bar rates of 30%/15%/5% into ARL0s of ~21/~106/~1390
bars; without them the raw 5% CRISIS rate would false-trip every ~20 bars, which is
why a bare threshold is unjustifiable (P7b).

Worked ARL0 for the LIQ channel on the DECISION cadence (using the `1 − p²` raw rates
above, entry@70/exit@60, `D_min=3`): `P[cross entry] = 1 − 0.70² = 0.51`,
`P[stay > exit] = 1 − 0.60² = 0.64`, so `alpha_eff^liq ≈ 0.51·0.64² ≈ 0.209` per
decision bar ⇒ `ARL0^liq ≈ 4.8` decision bars for sustained false NORMAL — markedly
SHORTER than the 21 of a single-percentile measure, because the max-of-two roughly
doubles each raw rate. For NORMAL→STRESSED (entry@85/exit@75):
`(1−0.85²)·(1−0.75²)² ≈ 0.2775·0.4375² ≈ 0.0531` ⇒ `ARL0^liq ≈ 19` decision bars; for
STRESSED→CRISIS (entry@95/exit@88): `(1−0.95²)·(1−0.88²)² ≈ 0.0975·0.2256² ≈ 0.00496`
⇒ `ARL0^liq ≈ 202` decision bars. These are the liq channel's CORRECT decision-bar
budgets and they REPLACE the single-percentile numbers for liq in the audit; the
single-percentile (`1−p`) numbers apply ONLY to `vol_pct`/`turb_pct`. (The BLUE
single-percentile `liq_stress`, Section 2.4, would restore the 21/106/1390 figures for
liq too.)

**(b′) The LIQ channel ALSO escalates OUT-OF-CADENCE every minute — it needs a
SEPARATE per-MINUTE budget (mandatory).** The decision-bar ARL0s above assume one
evaluation per `tau_regime`. But the event-driven escalation path (Section 3.5)
evaluates the liq threshold every 1-minute bar, giving liq `~ tau_regime/1min` MORE
false-escalation OPPORTUNITIES than the decision-bar budget assumes (`~60×` on an
hourly cadence). The liq channel's in-control WALL-CLOCK ARL0 on the event-driven path
is therefore `~60×` SHORTER than the decision-bar `ARL0^liq` implies, so an
out-of-cadence liq brake can fire on noise far more often than the decision-bar budget
claims. Two things are pinned to close this:
- **A separate per-MINUTE design target `alpha^liq_min` and the wall-clock
  `ARL0^liq_min` it implies** (RED). The event-driven path is held to
  `ARL0^liq_min` in MINUTES, computed from the same `1−p²` raw rates but at the
  1-minute evaluation frequency.
- **A minute-level RUN-LENGTH (dwell) filter on the event-driven path:** an intraday
  liq escalation must clear a minute-bar PERSISTENCE — `liq_stress` must stay above the
  entry mark for `D_liq_min` CONSECUTIVE 1-minute bars (RED, e.g. `D_liq_min` minutes)
  before it moves `g_t` out of cadence. This is the minute-level analogue of the
  decision-bar dwell and it is what restores the wall-clock `ARL0^liq_min` to the
  intended budget (a single-minute spread blip cannot trip the book; a SUSTAINED
  minute-level blowout can). De-escalation still waits for the scheduled cadence
  (Section 3.5 — escalation only, out of cadence).
- **Section 9 audits the realized INTRADAY (per-minute) liq-escalation rate against
  `ARL0^liq_min`, SEPARATELY from the decision-bar audit.** The decision-bar ARL audit
  does NOT silently cover the per-minute liq path. (See Invariant ARL, extended for the
  liq minute path; Section 3.5; Invariant STALEATTR.)

**(c) Out-of-control detection ARL1 (how fast a TRUE shift is caught).** Escalation
is IMMEDIATE and may jump multiple states (Section 3.2), so the detection delay for
a true shift large enough to clear the entry mark in one bar is `ARL1 ≈ 1` decision
bar (plus the event-driven path for a liq flash, Section 3.5, which can fire
intra-cadence). The ARL0/ARL1 asymmetry — `ARL0` in the hundreds-to-thousands of
bars, `ARL1 ≈ 1` — is the whole point: slow to false-alarm, fast to detect, exactly
the enter-fast/leave-slow asymmetry of 3.2 restated in run-length units.

**Cadence raises the per-bar cost of a false alarm — so the budget is in DECISION
bars, not minutes.** Because the `O(n³)` measures recompute only on `tau_regime`
(3.5), `alpha` is a per-DECISION-bar rate and the ARLs above are in decision bars;
multiply by `tau_regime` for wall-clock. A shorter cadence buys faster detection
(`ARL1` in shorter wall-clock) at the price of more frequent false-alarm
opportunities (`ARL0` in shorter wall-clock) — the `tau_regime ∈ {hourly, session}`
BLUE choice is exactly this ARL0-vs-ARL1 trade, resolved OOS through SYSTEM, never
on PnL.

**(d) Ablation reports realized escalation frequency vs the ARL0 budget
(mandatory).** The validation layer (Section 9) computes, per state and per measure,
the REALIZED count of escalations on held-out CALM stretches and compares it to the
`1/alpha_k_effective` budget above. A realized false-escalation rate materially
ABOVE budget means a threshold is too tight (the brake over-fires, taxing turnover);
materially BELOW means it is too loose (the brake will be late in a real crisis).
This is the answer to "how often does this brake fire on noise" and it is a
first-class acceptance number, not a footnote. (See Invariant ARL.)

> **Invariant ARL — every escalation threshold has a stated `alpha` and an implied
> `ARL0`, and the realized rate is audited against it.** The canonical entry/exit
> thresholds (3.2) are accompanied by their per-decision-bar false-escalation rate
> `alpha_k` and the dwell/hysteresis-implied `ARL0(state k)` (3.6b). Tests:
> (1) on a calm synthetic null panel the realized per-bar escalation rate into each
> state matches `alpha_k_effective` to within bootstrap CI; (2) `ARL0` increases
> monotonically with `D_min` and with the hysteresis gap (tighter filter ⇒ longer
> run length); (3) `ARL1 ≈ 1` decision bar for a step shift clearing the entry mark;
> (4) the ablation report (Section 9) emits realized-vs-budget escalation frequency
> per state; (5) the LIQ channel's `alpha` is derived from the max-of-two-percentiles
> null (`1 − p²`, Section 3.6a), NOT the single-percentile `1 − p`, and its
> decision-bar `ARL0^liq` (~4.8/19/202) matches the realized calm-panel rate; (6) the
> EVENT-DRIVEN liq path carries a SEPARATE per-MINUTE budget `alpha^liq_min`/
> `ARL0^liq_min` and a minute-level dwell `D_liq_min`, and Section 9 audits the
> realized INTRADAY liq-escalation rate against it SEPARATELY from the decision-bar
> audit (Section 3.6 b′; the decision-bar audit must not silently cover the per-minute
> path). A threshold with no stated `alpha`/`ARL0` is the bug.

---

## 4. Parameter governance

Every REGIME parameter is exactly one bucket. REGIME holds **no per-instrument
fitted state for sizing** — its TEAL surface is the rolling causal statistics of
the panel, fold-local and causal; its policy is RED/BLUE.

- **TEAL — fitted from data at runtime, causal, fold-local.** The rolling causal
  windows themselves: `sigma_agg` and its trailing percentile, the AR long/short
  trailing means and `sigma_AR`, the turbulence `mu_t`/`Sigma_t` (Ledoit–Wolf
  shrunk), the per-instrument spread/funding trailing percentiles aggregated into
  `liq_stress`. These are STATISTICS of the panel, recomputed each bar from
  permitted history — not tuned, not per-symbol-optimised.
- **AMBER — supplied metadata.** Universe membership and `n`; per-asset calendar /
  session (for panel alignment + cadence); the implied-vol-vs-realized switch per
  asset (DVOL availability); the regime cadence `tau_regime` in calendar terms;
  asset-class tags. Cross-asset behaviour flows through metadata, never forked
  logic.
- **RED — shared modelling/economic constants, NOT per-symbol.** The state set and
  the `g(s)` map (`1.00/0.75/0.50/0.25`); the `min()`-not-product combination; the
  absorption `kappa` (top-k fraction) and windows `W_AR`, `W_ARshort`, `W_ARlong`;
  the absorption window-floor `c_AR` (observations-per-dimension floor for `C_t`,
  Section 2.2); the turbulence window `W_turb`, the `c_cov` observations-per-dimension
  floor, the `d_t/n_t` dof-normalization (Section 2.3), the Ledoit–Wolf target; the
  volatility windows `W_pct`, `W_vov`, term-structure windows; the canonical
  hysteresis entry/exit threshold pairs (Section 3.2); the minimum dwell `D_min`;
  the per-decision-bar false-escalation design target `alpha_k` and the implied
  `ARL0` budget per state (Section 3.6); the per-MINUTE liq design target
  `alpha^liq_min`/`ARL0^liq_min` and the minute-level dwell `D_liq_min` for the
  event-driven liq path (Section 3.6 b′, 3.5); the post-shrink condition-number gate
  `cond_max` for both `C_t` and `Sigma_t` (Section 2.3); the absorption fixed-`n_t`
  freeze band `frac_dn` (Section 2.2); the Ledoit–Wolf TARGETS (constant-correlation
  for `C_t`/`Sigma_t`) — the LW INTENSITIES are TEAL, not RED (Sections 2.2/2.3); the
  BLUE-grid cardinality cap `K_regime_max` (Section 4.1); the cost/rebalance band
  (Section 5); the floor (`g >= 0.25`, REGIME never flats the book). RED is shared and **unreachable
  from per-instrument config** (Invariant C, inherited from the sibling specs).
- **BLUE — finite validated grids, class-level only, NEVER per-symbol-on-PnL.** A
  small enumerated candidate set around the RED canonical thresholds: e.g.
  `delta_AR` STRESSED-entry ∈ {+0.8, +1.0, +1.2}, `vol_pct` STRESSED-entry ∈
  {80, 85, 90}, `kappa` ∈ {0.15, 0.20, 0.25}, `D_min` ∈ {2, 3, 4}, `tau_regime` ∈
  {hourly, session}. Each is finite, searched OFFLINE only, and **promoted solely
  through SYSTEM's deflated-Sharpe gate over purged walk-forward / CPCV** — never
  fitted continuously, never per symbol, never on raw PnL without the
  multiple-test tax. REGIME's BLUE grid is one of the module grids whose union
  SYSTEM deflates against (SYSTEM P6).

### 4.1 Free-parameter count and the hard cap on BLUE cardinality (Probe-4, mandatory)

So the regime model itself cannot be silently overfit, the free parameters are
enumerated and the BLUE-grid cardinality is hard-capped.

**The free parameters (each RED or BLUE — none TEAL-on-PnL):**

| Parameter | Bucket | Motivation (quantile / economic) |
|---|---|---|
| `g(s)` map `{1.00,0.75,0.50,0.25}` | RED (fixed) | mirrors LEVERAGE governor ladder; not searched |
| `kappa` (top-k fraction) | BLUE {0.15, 0.20, 0.25} | Kritzman–Li ~1/5 of assets |
| hysteresis entry/exit pairs (3 boundaries × 2 marks, per measure) | RED canonical + BLUE around the STRESSED entry per measure | causal percentile / standardized-displacement levels |
| `D_min` (global dwell) | BLUE {2, 3, 4} | run-length filter (Section 3.6) |
| `tau_regime` (cadence) | BLUE {hourly, session} | ARL0-vs-ARL1 trade |
| `W_AR, W_ARshort, W_ARlong, W_turb, W_pct, W_vov, term windows` | RED (fixed) | trailing-window lengths, economically set |
| `c_AR, c_cov` (obs-per-dim floors) | RED (fixed) | MP/rank-deficiency floors |
| `cond_max` (condition-number gate) | RED (fixed) | numerical-sanity threshold |
| `rebal_min_frac, rebal_cost_cap` (cost band) | RED (fixed) | "don't rebalance on a wiggle" band |
| `alpha_k, alpha^liq_min` (false-alarm targets) | RED (fixed) | SPC design targets, Section 3.6 |
| **Ledoit–Wolf intensities `δ_AR, δ_turb`, LW targets** | **TEAL (δ closed-form data-derived) / RED (targets)** | NOT searched — do NOT enter the count (Section 2.2/2.3) |

The shrinkage intensities are explicitly EXCLUDED from the searched DOF (they are
closed-form, causal, data-derived — Section 2.2). The cost band, windows, floors, and
SPC targets are RED-fixed, not searched. **Only five families are BLUE-searched:**
`kappa` (3), the per-measure STRESSED-entry level (≤3 each over 4 measures),
`D_min` (3), `tau_regime` (2), and the percentile-band variants.

> **Hard cap `K_regime` on the BLUE-grid cardinality.** The total number of distinct
> BLUE configurations REGIME exposes to SYSTEM's joint trial ledger is capped:
> `K_regime ≤ K_regime_max` (RED, e.g. `K_regime_max = 64`). The CANONICAL RED set is
> the default; BLUE is a SMALL BOUNDED NEIGHBOURHOOD around it (a one-step grid on each
> of the five searched families, never a full Cartesian explosion). REGIME contributes
> exactly its registered `K_regime` to SYSTEM's joint `N_sys` deflation count
> (SYSTEM P6) — it cannot silently grow its degrees of freedom and dilute the
> deflation. A test asserts the registered grid cardinality `K_regime ≤ K_regime_max`
> and that every BLUE parameter is quantile/economically motivated (not a free
> continuous fit, not on PnL). (See Invariant NOFIT, extended.)

> **Invariant C — RED unreachable from per-instrument config** (inherited). RED
> enters only via `resolve_config(defaults=...)`. If a `g(s)` value, a hysteresis
> threshold, `kappa`, `D_min`, or the cost band appears as a per-instrument field,
> that is the bug.

> **Invariant NOFIT — thresholds are standardized/quantile-based and validated
> out-of-sample, never PnL-fit (P5).** Every threshold is either a standardized
> shift (z-score / causal percentile of a measure against its own history) or a
> BLUE candidate promoted only by SYSTEM's deflated-Sharpe gate. Tests: (1) no
> code path selects a regime breakpoint by maximising in-sample/backtest PnL;
> (2) REGIME's BLUE grid is registered in SYSTEM's persistent trial ledger so its
> degrees of freedom are paid for in the joint deflation (SYSTEM Invariant SG);
> (3) a regime config tuned to maximise a fold's PnL FAILS the accumulated-ledger
> DSR — the overfit is caught, not rewarded; (4) the registered BLUE-grid cardinality
> `K_regime ≤ K_regime_max` (Section 4.1) and the Ledoit–Wolf intensities `δ_AR/δ_turb`
> are NOT in the searched grid (they are closed-form/TEAL, not DOF), so the regime
> layer cannot silently grow its degrees of freedom and dilute SYSTEM's deflation.

---

## 5. Cost interaction — the rebalance-threshold band (P7)

A change in `g_t` moves the book's gross target, which forces trades, which cost
money (COST.md). Two safeguards, beyond the hysteresis + dwell of Section 3,
ensure REGIME does not rebalance the whole book on a 1% regime wiggle:

### 5.1 The multiplier deadband

`g_t` is emitted in the discrete steps `{1.00, 0.75, 0.50, 0.25}` (Section 3.1),
so by construction it cannot move by 1% — the smallest possible move is a full
0.25 step, and that move is itself gated by hysteresis + dwell. There is **no
continuous `g_t`** in v1 precisely to avoid micro-rebalances. (A continuous
multiplier is a deliberately-rejected design: it re-grosses the book on every
noisy tick of a statistic.)

### 5.2 The cost / rebalance gate

**`book_gross_at_full` — definition and causality (pinned; the load-bearing gross
anchor).** `book_gross_at_full` is the gross notional the book WOULD run at `g = 1`
GIVEN the CURRENT per-position LEVERAGE ceilings — i.e. the sum over the live book of
each position's `g`-removed target gross (the gross LEVERAGE's seven-ceiling `min()`
permits each name when the regime multiplier is set to 1.0). It is:

- **NOT a constant.** It is TIME-VARYING: it moves as LEVERAGE's per-position
  ceilings move (vol-target changes, margin/heat changes, names enter/leave). It is
  the `g`-FACTORED-OUT gross, not a static book size.
- **Computed by LEVERAGE, READ by REGIME (Invariant NREBUILD).** REGIME does not
  re-derive position sizing; it reads the `g=1` gross LEVERAGE reports for the current
  book. `target_gross_new = g_new · book_gross_at_full` is therefore "set the regime
  multiplier to `g_new` and let LEVERAGE re-evaluate its ceilings."
- **`F_t`-measurable / causal (P6).** It is built from current ceilings and current
  book state at bar `t`; no future data. It is a snapshot input to the cost gate, not
  a forecast.

`current_gross` (below) is the gross the book is ACTUALLY running right now — which can
DRIFT away from `g_prev · book_gross_at_full` when positions close, names are dropped, or
ceilings tighten between regime decisions. That drift is exactly what makes clause (b)
live (see the worked dwell example 6.3).

Even a legitimate full-step change must clear a cost gate before it propagates:

```
On a proposed state change s_prev -> s_new with multiplier g_prev -> g_new:

  target_gross_new   = g_new * book_gross_at_full          # the gross the new state implies
  delta_gross        = | target_gross_new - current_gross |
  rebalance_cost_hat = COST.round_trip_estimate(delta_gross)   # spread + impact + funding to move the book

  Apply the move ONLY if:
    (a) the state change passed hysteresis + dwell (Section 3), AND
    (b) delta_gross  >=  rebal_min_frac * current_gross      # rebalance-threshold band, RED
        (don't rebalance for a sub-threshold gross change), AND
    (c) for a DE-escalation (re-grossing UP), the implied rebalance cost is small
        relative to the gross being added: rebalance_cost_hat <= rebal_cost_cap
        (RED, a fraction of delta_gross). Escalation (cutting gross DOWN for safety)
        is NEVER blocked by cost — safety beats cost on the downside.
  Else: hold g_t at g_prev, log reason `regime_change_cost_gated`.
```

Asymmetry is deliberate and mirrors P7 + LEVERAGE/OPS safety-first: **cutting gross
for safety is never blocked by cost; re-adding gross must pay its way.** The
`rebal_min_frac` band (RED, e.g. 0.10 — a state change implying less than a 10%
gross move is deferred) is the "don't rebalance on a 1% wiggle" control made
concrete. (See Invariant COSTGATE.)

### 5.3 Who actually trades

REGIME does not place the rebalance orders. It emits the changed `g_t`; LEVERAGE
re-sizes (its `min()` now includes the new `g_t`) and SYSTEM/EXECUTION execute the
resulting book delta on the normal cadence with the normal no-churn deadband
(SYSTEM `delta` deadband, EXECUTION tactic ladder). REGIME's cost gate is a
PRE-CHECK that the regime change is worth propagating; the actual fills are priced
and executed downstream. REGIME never emits an order (Invariant SCOPE).

---

## 6. Worked examples (reproduce bit-for-bit)

### 6.1 A correlation spike cuts the book

One fully-specified regime decision carried through the absorption-ratio measure,
the standardized shift, the hysteretic state machine, the `min()` combination, the
cost gate, and the downstream LEVERAGE consumption — every number reproducible
from the inputs with no hidden step. This is the headline scenario: a correlation
spike (absorption ratio crossing its standardized trigger) steps `g_t` down with
hysteresis, LEVERAGE consumes it in its `min()`, the book gross is cut, and the
cut's cost is priced.

**State at the regime decision bar (all `F_t`-measurable, coarse cadence):**

| Input | Value |
|---|---|
| universe size `n` | `80` assets |
| absorption `kappa`, `k` | `kappa = 0.20` → `k = ceil(0.20·80) = 16` |
| top-16 eigenvalue sum / total | `Σλ_{1..16} = 60.0`, `Σλ_{1..80} = 80.0` → `AR_t = 60.0/80.0 = 0.750` |
| AR long-window mean / std | `AR_bar_long = 0.600`, `sigma_AR = 0.050` |
| AR short-window mean | `AR_bar_short = 0.710` |
| standardized shift | `delta_AR = (0.710 − 0.600)/0.050 = +2.20` |
| other sub-measures | `vol_pct = 72` (NORMAL band), `turb_pct = 80` (NORMAL band), `liq_stress_pct = 55` (CALM band) |
| previous state | `NORMAL` (`g_prev = 0.75`), dwell_bars = 5 (≥ `D_min = 3`) |
| book gross at full | `book_gross_at_full = 1000.00` USDT notional; current_gross = 750.00 (= 0.75·1000) |
| rebalance band | `rebal_min_frac = 0.10` |

**Step 1 — classify each sub-state (Section 3.2 thresholds).**

```
corr (delta_AR = +2.20):  >= +2.0  -> CRISIS entry threshold crossed       -> sub-state = CRISIS  (g^corr = 0.25)
vol  (vol_pct  = 72):     in [70,85) NORMAL band                            -> sub-state = NORMAL  (g^vol  = 0.75)
turb (turb_pct = 80):     in [75,88) NORMAL band                            -> sub-state = NORMAL  (g^turb = 0.75)
liq  (liq_stress_pct=55): < 60 CALM                                         -> sub-state = CALM    (g^liq  = 1.00)
```

**Step 2 — global state = worst sub-state = `min()` (P3, Section 2.5).**

```
g_raw = min(g^vol, g^corr, g^turb, g^liq) = min(0.75, 0.25, 0.75, 1.00) = 0.25
binding sub-state = corr (the absorption-ratio crisis is the nearest failure mode)
proposed state = CRISIS
```

**Step 3 — hysteresis + dwell (Section 3.3).** This is an ESCALATION
(NORMAL→CRISIS, more stressed). Escalation is ALWAYS allowed immediately, may jump
multiple states, and is not blocked by dwell. So the proposed transition stands:
`s_new = CRISIS`, `g_new = 0.25`. (Had this been a de-escalation, it would require
`dwell_bars ≥ 3` AND the looser exit threshold, one step at a time.)

**Step 4 — cost / rebalance gate (Section 5.2).** Escalation cuts gross DOWN; cost
NEVER blocks a safety cut (clause c applies only to re-grossing up). Still check
the deadband (clause b):

```
target_gross_new = 0.25 · 1000.00 = 250.00 USDT
current_gross    = 750.00 USDT
delta_gross      = |250.00 − 750.00| = 500.00 USDT
rebal_min_frac · current_gross = 0.10 · 750.00 = 75.00 USDT
delta_gross (500.00)  >=  75.00  -> deadband cleared
escalation -> cost clause (c) does NOT apply -> APPLY the move.
```

**Step 5 — emit and feed LEVERAGE (Section 3.4).**

```
g_t = 0.25,  s_t = CRISIS,  binding = corr,  delta_AR = +2.20,  reason = regime_escalate_corr_crisis
```

LEVERAGE consumes `g_t` through the `H_book` scaling — the SINGLE gross channel
(Section 3.4). For a representative position whose seven-ceiling pre-regime minimum
was `L_pre = 4.0x` and whose drawdown governor multiplier was `1.00` (no equity
drawdown — a SEPARATE axis, Section 10):

```
LEVERAGE final (schematic, the regime term entering the min/scaling):
  L_after_regime = min( L_pre , ... ) with g_t scaling the book-heat ceiling H_book:
       H_book_effective = g_t · H_book = 0.25 · H_book
  The book-heat ceiling now binds at a quarter of normal, so the portfolio ceiling
  L_portfolio drops proportionally; the book's TOTAL gross target falls from
  750.00 -> 250.00 USDT (the 0.25 multiplier on the full-gross 1000.00).

  The SELECTION count cap also tightens (CRISIS => N_concurrent = 0.25N), but it is
  a STRUCTURAL channel, NOT a second multiplier on gross (Section 3.4, Invariant
  NODOUBLE2): it re-distributes the SAME 250.00 USDT across fewer names; it does NOT
  cut gross a second time to ~g_t^2 = 0.0625*1000 = 62.50. The realized gross is
  250.00 (= g_t * full), NOT 62.50. If the surviving names cannot hold 250.00
  without breaching their own per-position ceilings, LEVERAGE's per-position min()
  lowers it further and reports WHICH ceiling bound -- there is no second g_t factor.
```

**Step 6 — the cost of the cut (priced by COST, Section 5.2).** Cutting gross by
`delta_gross = 500.00` USDT one-way (reducing positions; reduce-only, taker-side
conservative), using COST's round-trip half of a one-way reduce at, say, an
all-in `~20 bps` one-way reduce cost on the moved notional:

```
rebalance_cost_hat ≈ 20 bps · 500.00 USDT = 0.0020 · 500.00 = 1.00 USDT
```

The book is cut from 750 → 250 USDT gross at a one-way de-risking cost of ~1.00
USDT — paid because a correlation crisis (`delta_AR = +2.20`, AR jumped from a
0.60 baseline to 0.75) means diversification has collapsed and the netted
portfolio-margin/co-movement risk LEVERAGE warns about (its P4) is live. This is
the macro brake doing its single job: a global co-movement spike → absorption
trigger → `g_t` step-down → LEVERAGE `min()`/`H_book` scaling → gross cut → priced.

**The hysteresis tail (why the book does not snap back).** Suppose next bar
`delta_AR` falls to `+1.5`. De-escalation CRISIS→STRESSED requires the looser exit
threshold (`delta_AR < +1.4`) AND `dwell_bars ≥ 3`. `+1.5` is NOT below `+1.4`, and
dwell has just reset to 0, so the book STAYS at `g_t = 0.25`. Only when `delta_AR`
falls below `+1.4` AND at least 3 decision bars have elapsed does it step UP one
level to STRESSED (`g_t = 0.50`) — and that re-grossing-up move must then clear the
cost gate's clause (c). Enter the brake fast, release it slowly and only when it
pays — no whipsaw.

### 6.2 A binding-sub-state HANDOFF at unchanged `g(s_t)` (the multi-measure case the clock must get right)

This is the transition the two-clock design (Sections 3.2–3.3) exists to handle, and
it is the case where a naive single-clock implementation goes non-deterministic. It
exercises 3.3(ii) — the dwell clock must NOT reset on a binding-identity handoff at
constant `g(s_t)` — and 3.3(iii) — a non-binding measure's wiggle is invisible.

**Three consecutive decision bars (hourly cadence, `D_min = 3`). Percentile bands:
NORMAL@70/60, STRESSED@85/75, CRISIS@95/88. `delta_AR` bands: NORMAL@+0.5/+0.2,
STRESSED@+1.0/+0.6, CRISIS@+2.0/+1.4.**

| bar | `delta_AR` (corr) | `turb_pct` | `vol_pct` | `liq` | per-sub-state | global `s_t` | `g(s_t)` | dwell after | transition | binding |
|---|---|---|---|---|---|---|---|---|---|---|
| `t0` | `+1.3` → STRESSED | 50 → CALM | 50 → CALM | CALM | corr=STRESSED, rest CALM | STRESSED | 0.50 | 1 (just entered) | escalate (from NORMAL) | corr |
| `t1` | `+0.9` (still > exit +0.6) → STRESSED | 50 → CALM | 88 → STRESSED (entry 85) | CALM | corr=STRESSED, vol=STRESSED | STRESSED | 0.50 | 2 | hold | corr |
| `t2` | `+0.5` (< exit +0.6) → de-esc to NORMAL | 50 → CALM | 90 → STRESSED | CALM | corr=NORMAL, vol=STRESSED | STRESSED | 0.50 | 3 | **hold (binding_handoff corr→vol)** | **vol** |

Reading the three reset questions on bar `t2`:

- `corr` de-escalates STRESSED→NORMAL (its own value `+0.5` fell below its own exit
  mark `+0.6`) — **per-sub-state hysteresis**, 3.3(i).
- `vol` is now the worst sub-state (STRESSED), so the **global `min` stays at
  STRESSED, `g(s_t) = 0.50` UNCHANGED**. The binding identity hands off `corr → vol`.
- Therefore the **dwell clock CONTINUES (1→2→3), it does NOT reset** — 3.3(ii). The
  book's gross target never moved (`0.50·full` throughout `t0..t2`), so there was no
  re-gross/re-cut to protect against; resetting would have needlessly delayed the
  next legitimate de-escalation.
- The transition is logged `hold` with reason `binding_handoff`, and
  `binding` flips `corr → vol` in the reason chain — auditable, but NOT a state
  change and NOT a clock reset.
- Note `vol` escalating CALM→STRESSED across `t0→t1` while it was BELOW the binding
  `corr` level was a **non-binding oscillation invisible to the clock** (3.3(iii)):
  it did not change `g(s_t)` (corr was already holding STRESSED), so it neither reset
  the clock nor forced a rebalance. It only became visible when it BECAME the min on
  `t2`.

Now bar `t3`: suppose `delta_AR = +0.3` (NORMAL), `vol_pct = 70` (de-esc vol toward
NORMAL, exit mark 75 — `70 < 75` so vol exits STRESSED→NORMAL), all else CALM. The
worst sub-state is now NORMAL. This IS a global de-escalation STRESSED→NORMAL
(`g(s_t): 0.50 → 0.75`). It is permitted because `dwell_bars = 3 ≥ D_min = 3` AND
the binding (vol) exit threshold is met — and ONLY because the dwell kept counting
through the handoff instead of resetting on `t2`. Had a naive single clock reset on
the `t2` handoff, this legitimate de-escalation would have been wrongly blocked
until `t5`, needlessly holding the book at half gross for two extra bars. THIS is
why 3.3(ii) is pinned. `g(s_t)` now steps to `0.75`, dwell resets to 1, transition
`deescalate`, binding `vol`, and the re-grossing-up move clears the cost gate clause
(c) before propagating.

### 6.3 The cost-gate deadband DEFERS a state change after `current_gross` has drifted (clause (b) live)

The discrete `g(s)` ladder (5.1) makes every legal state change imply a `≥ 0.25`
step, so `delta_gross = |g_new − g_prev|·book_gross_at_full ≥ 0.25·book_gross_at_full`
WHENEVER `current_gross == g_prev·book_gross_at_full` — which always exceeds
`rebal_min_frac·current_gross` for any sane `rebal_min_frac`, making clause (b) appear
DEAD. It is NOT dead: clause (b) compares against `current_gross`, and `current_gross`
DRIFTS below `g_prev·book_gross_at_full` whenever positions have closed or the book has
shrunk since the last regime decision. This example shows clause (b) actually deferring.

The key is that clause (b) compares `delta_gross` against `current_gross`, and the
target is `g_new·book_gross_at_full` — so the EFFECTIVE move is small whenever
`current_gross` has already drifted CLOSE to where the new state would put it. Set up
that drift: a STRESSED-era book attritioned (stops hit, names closed, no new
admissions under the count cap), so both `current_gross` AND the per-position ceilings
(hence `book_gross_at_full`) shrank, leaving the book already sitting near `g_new`'s
target.

**State at the regime decision bar (book has DRIFTED since the last regime decision):**

| Input | Value |
|---|---|
| `book_gross_at_full` (also shrunk as ceilings tightened) | `160.00` USDT (the `g=1` gross LEVERAGE permits the now-smaller book) |
| previous state | `STRESSED` (`g_prev = 0.50`), dwell_bars = 4 (≥ `D_min = 3`) |
| `g_prev · book_gross_at_full` | `0.50 · 160.00 = 80.00` (what the STRESSED state nominally implies now) |
| `current_gross` (DRIFTED) | `120.00` USDT — attrition + no re-gross left the live gross ABOVE `g_prev·full`, near where NORMAL would sit |
| proposed change | de-escalate STRESSED→NORMAL: `g_new = 0.75` (hysteresis + dwell PASSED) |
| `rebal_min_frac` | `0.10` |

**Step 1 — the EFFECTIVE move is small because `current_gross` has drifted.**

```
target_gross_new = g_new · book_gross_at_full = 0.75 · 160.00 = 120.00 USDT
current_gross    = 120.00 USDT                    (book already sitting there post-drift)
delta_gross      = |120.00 − 120.00| = 0.00 USDT  (NOT the 0.25-step the undrifted ladder would imply)
rebal_min_frac · current_gross = 0.10 · 120.00 = 12.00 USDT
delta_gross (0.00)  <  12.00  -> clause (b) FAILS the deadband
```

The discrete ladder would imply a `≥ 0.25·book_gross_at_full = 40.00` move if
`current_gross` equalled `g_prev·book_gross_at_full = 80.00` (clause (b) would then
pass: `40.00 ≥ 0.10·80.00 = 8.00`). But the DRIFT to `120.00` makes the effective move
`0.00`, so the ladder's `≥ 0.25-step` guarantee no longer holds and clause (b) bites.

**Step 2 — clause (b) DEFERS the change.**

```
Hysteresis + dwell PASSED (clause a), but delta_gross (0.00) < rebal_min_frac·current_gross (12.00)
-> clause (b) NOT satisfied -> HOLD g_t at g_prev = 0.50, log reason `regime_change_cost_gated`.
```

Even though the state machine wants to step `g(s_t)` STRESSED→NORMAL, the book is
ALREADY sitting at the gross the new state implies (it attritioned there on its own),
so re-stamping the regime state would force a whole-book rebalance to achieve a ZERO
net gross change — pure churn. The deadband correctly DEFERS: `g(s_t)` stays at
STRESSED on the books, no rebalance fires, and the change is reconsidered next bar when
the book has re-grown or `book_gross_at_full` has moved enough to make the step worth
its cost. This is the Probe-5 case the 6.1 escalation and 6.2 handoff examples do not
show: clause (b) preventing a whole-book rebalance on a small EFFECTIVE `g(s_t)` move
once `current_gross` has DRIFTED away from `g_prev·book_gross_at_full`. (See Invariant
COSTGATE; Invariant WE.)

> **Invariant WE — the worked examples are regression fixtures.** A test reproduces
> every number in Section 6.1 to floating tolerance from the tabulated inputs:
> `AR_t = 0.750` (denominator `n_t = 80` by the trace identity, Section 2.2),
> `delta_AR = +2.20`, the four sub-states, `g_raw = 0.25` with `binding = corr`,
> escalation-applies-immediately, the deadband arithmetic (`delta_gross = 500.00 ≥
> 75.00`), `g_t = 0.25`, `H_book_effective = 0.25·H_book`, the gross cut `750 →
> 250` (the SINGLE gross channel — realized gross is `250`, NOT `g_t²·full = 62.50`,
> Invariant NODOUBLE2), the `~1.00 USDT` rebalance cost, and the hysteresis tail
> (de-escalation blocked at `delta_AR = +1.5`). A second test reproduces Section
> 6.2: the `t0..t3` table, the `binding_handoff corr→vol` at unchanged `g(s_t) =
> 0.50` with the dwell clock CONTINUING (not resetting), the non-binding vol
> oscillation being invisible to the clock, and the legitimate `t3` de-escalation
> being PERMITTED (which a naive resetting clock would wrongly block). If any RED
> constant changes, these fixtures change with it — they are the anchors pinning the
> absorption trigger, the `min()`, the hysteresis asymmetry, the cost-gate
> asymmetry, the single-gross-channel, and the two-clock determinism. A third test
> reproduces Section 6.3: with `current_gross` DRIFTED to `120.00` and
> `book_gross_at_full` shrunk to `160.00`, a hysteresis+dwell-PASSED STRESSED→NORMAL
> de-escalation is DEFERRED because `delta_gross = 0.00 < rebal_min_frac·current_gross
> = 12.00` (clause (b) live), logged `regime_change_cost_gated` — proving the deadband
> is not dead once the book has drifted.

---

## 7. Architectural split and directory layout

Four layers, kept independent for testability (mirrors LEVERAGE/COST):

1. **Panel layer:** builds the causal cross-asset return panel `R_t` from
   Nautilus bars, aligned by AMBER calendar, de-meaned causally. No modelling
   beyond alignment + causal de-mean.
2. **Measure layer:** four independent estimators — `VolRegime`, `AbsorptionRatio`,
   `Turbulence`, `LiquidityStress` — each emitting one measure + diagnostics, each
   testable in isolation. `LiquidityStress` is a pure read of EXECUTION/COST
   (Invariant NREBUILD).
3. **State layer:** `RegimeStateMachine` (hysteresis + dwell + `min()` over
   sub-states) and `CostGate` (the rebalance-threshold band). Emits `RegimeState`
   (`g_t`, `s_t`, binding sub-state, reason chain).
4. **Validation layer:** backtests, threshold-stability tests, ablations; never
   imported by the live panel/measure/state layers (Invariant V).

```text
regime/
+-- config/
|   +-- defaults.py          # PolicyDefaults: RED states, g(s) map, thresholds, kappa, windows, D_min, cost band
|   +-- asset_class.py       # AMBER calendar, DVOL-vs-realized switch, cadence per class
|   +-- resolve.py           # single merge point (RED unreachable from per-instrument)
+-- panel/
|   +-- build.py             # causal cross-asset return panel; calendar alignment; causal de-mean; staleness drops
+-- measures/                # MEASURE LAYER — each emits one measure, never an action
|   +-- vol_regime.py        # sigma_agg, vol_pct, vov, term-structure (reads LEVERAGE vol forecasts)
|   +-- absorption.py        # correlation eigendecomp, AR_t, standardized shift delta_AR (Kritzman-Li)
|   +-- turbulence.py        # Ledoit-Wolf shrink + Mahalanobis d_t, turb_pct (Kritzman-Mancini)
|   +-- liquidity.py         # aggregate EXECUTION spread-blowout + COST funding-extreme -> liq_stress
+-- state/
|   +-- bands.py             # map each measure -> ordered sub-state via entry/exit thresholds
|   +-- machine.py           # RegimeStateMachine: worst-sub-state min(), hysteresis, dwell
|   +-- cost_gate.py         # rebalance-threshold band; escalation never cost-blocked
|   +-- emit.py              # assemble RegimeState (g_t, s_t, binding, reason chain)
+-- validate/
|   +-- backtest.py          # purged WF / CPCV hooks into SYSTEM; OOS only
|   +-- threshold_stability.py # are the standardized triggers stable across folds? (no PnL-fit)
|   +-- baselines.py         # always-on (g=1), single-measure-only, continuous-multiplier (rejected) baselines
|   +-- ablation.py          # which measure actually binds; ARL audit (realized vs alpha_k/ARL0 budget, incl. liq 1-p^2 null
|   |                        #   and SEPARATE per-minute alpha^liq_min/ARL0^liq_min for the event-driven path); does each earn its place
|   +-- stats.py             # block-bootstrap CIs; multiple-test control (feeds SYSTEM deflation)
|   +-- reports.py
+-- synth/
|   +-- generate.py          # adversarial panels: correlation-to-1, vol spike, turbulence day, liquidity blowout, oscillation,
|   |                        #   calm-null (ARL audit), panel-shrink (TURBNORM), corr-structure-fixed panel-shrink (ARDISPNORM),
|   |                        #   rank-deficient window (ARFLOOR), marginal-T/N noise (ARCOND), condition-breach (CONDNUM),
|   |                        #   liq-flash-on-stale-corr (STALEATTR), per-minute liq-flash null (ARL b'), drifted-gross deadband (6.3),
|   |                        #   multi-measure binding handoff (CLOCK, Section 6.2)
+-- types.py
+-- engine_api.py            # public entry point: regime_state(panel, exec_state, cost_state, book, cfg) -> RegimeState
tests/
+-- ...                      # unit, causal, hysteresis/dwell, cost-gate, worked-example fixture, ablation, parity
```

Stack: Python 3.11+, numpy, scipy (eig/cholesky/Ledoit–Wolf), pandas, pytest. No
ML frameworks.

---

## 8. Shared data objects (`types.py`)

```python
from dataclasses import dataclass
from typing import Literal, Mapping
import numpy as np

RegimeStateName = Literal["CALM", "NORMAL", "STRESSED", "CRISIS"]
SubName = Literal["vol", "corr", "turb", "liq"]

@dataclass(frozen=True)
class Panel:                       # causal cross-asset return panel at the regime bar
    returns: "np.ndarray"          # shape (W, n_t), log-returns, causally de-meaned
    instruments: tuple[str, ...]   # length n_t (live, post staleness-drop)
    asof_bar: int                  # F_t index; no future data (P6)

@dataclass(frozen=True)
class SubState:                    # one measure's PER-SUB-STATE value + hysteretic band classification
    name: SubName
    value: float                   # vol_pct / delta_AR / turb_pct(of d_norm) / liq_stress (its native unit)
    state: RegimeStateName         # band classification under THIS measure's own hysteresis (3.2)
    g_sub: float                   # the sub-multiplier g(state)
    asof_bar: int                  # when this measure was last RECOMPUTED (== current bar for liq;
                                   # last scheduled cadence bar for vol/corr/turb -- staleness audit, Inv STALEATTR)
    fresh: bool                    # True if recomputed this bar; False if carried/frozen (Inv STALEATTR/ARFLOOR)
    diag: Mapping[str, float]      # e.g. AR_t, AR_bar_short, sigma_AR (raw-level std, Inv DISP),
                                   # eig_topk_share, n_t, d_norm,
                                   # cond_number (post-shrink, BOTH C_t and Sigma_t; gated by cond_max, Inv CONDNUM),
                                   # lw_intensity (delta_AR/delta_turb, TEAL closed-form, NOT a knob),
                                   # ar_window_deficient (ARFLOOR), ar_nt_changed (ARDISPNORM freeze),
                                   # cond_breach_turb/cond_breach_corr (CONDNUM)

@dataclass(frozen=True)
class RegimeState:                 # THE OUTPUT
    g_t: float                     # global risk multiplier in {1.00,0.75,0.50,0.25} (>=0.25, P/Inv FLOOR)
    state: RegimeStateName
    binding_sub: SubName           # which sub-state bound the min() (P3)
    dwell_bars: int                # bars in current state
    transition: Literal["escalate","deescalate","hold","cost_gated"]
    sub_states: Mapping[SubName, SubState]
    cost_gate_passed: bool
    rebalance_cost_hat: float      # COST estimate of the implied book delta (telemetry)
    reason_codes: tuple[str, ...]  # e.g. ('regime_escalate_corr_crisis',)
    asof_bar: int
    engine_version: str
```

```python
class RegimeEngine:
    def regime_state(self, panel: "Panel", exec_state, cost_state,
                     book, prev: "RegimeState", cfg: "ResolvedConfig"
                     ) -> "RegimeState":
        """The public entry point. Compute the four measures from `panel`
        (+ exec_state/cost_state for liquidity, READ not re-modelled), classify
        each into an ordered sub-state under hysteresis, take the worst (min(),
        P3), apply the state machine (hysteresis + dwell, P4), pass the implied
        gross change through the cost gate (P7), and emit RegimeState. Pure
        function of (panel, exec_state, cost_state, book, prev, cfg); causal (P6);
        deterministic; emits NO order and NO size (P1, Invariant SCOPE)."""
```

---

## 9. Validation, stress, and ablation (OUT-OF-SAMPLE via SYSTEM)

Validation is part of the module, and it is the place P5 (no-overfit) is enforced.
REGIME is validated as a SUB-OBJECT of the composed chain through SYSTEM's purged
walk-forward / sampled CPCV, with REGIME's BLUE grid registered in SYSTEM's
persistent trial ledger so its degrees of freedom are paid for in the joint
deflated Sharpe (SYSTEM P6, Invariant SG). REGIME never self-scores a threshold on
PnL.

Required validations:
- **Threshold stability across folds.** Are the standardized triggers (`delta_AR`,
  `vol_pct`, `turb_pct`, `liq_stress`) stable out-of-sample, or did a fold's
  best-PnL breakpoint move? A threshold that must be re-tuned per fold is overfit
  and is rejected (P5, Invariant NOFIT).
- **Ablation — which measure binds.** Report how often each of the four
  sub-states is the binding one. A measure that NEVER binds is redundant or
  mis-scaled; a measure that ALWAYS binds is the only real signal (simplify).
  Mirrors LEVERAGE's "which ceiling binds" ablation.
- **ARL audit — realized escalation frequency vs the false-alarm budget (P7b,
  Section 3.6d).** On held-out CALM stretches, count realized escalations into each
  state, per measure and for the global `min`, and compare to the
  `1/alpha_k_effective` budget (`ARL0 ≈ 21 / 106 / 1390` decision bars for
  NORMAL/STRESSED/CRISIS at the canonical thresholds + `D_min=3`). A realized rate
  materially above budget ⇒ a threshold too tight (over-firing, turnover tax);
  materially below ⇒ too loose (late in a real crisis). The liq channel is audited on
  TWO budgets: its decision-bar `ARL0^liq` (off the `1−p²` null, Section 3.6a) and,
  SEPARATELY, the event-driven path's per-MINUTE `ARL0^liq_min` (Section 3.6 b′) — the
  decision-bar audit does NOT cover the per-minute path. Also report the realized
  detection delay (`ARL1`) on the crisis-replay set. This makes "how often does the
  brake fire on noise" a first-class acceptance number (Invariant ARL).
- **Baselines.** Compare against (a) always-on (`g_t ≡ 1`, no regime overlay);
  (b) each single measure alone; (c) the REJECTED continuous-multiplier design (to
  demonstrate it whipsaws / over-trades vs the discrete sticky machine). The
  regime overlay must reduce realized tail risk / drawdown without destroying
  net-of-cost return relative to always-on; if it does not, it is not earning its
  turnover.
- **Whipsaw / turnover audit.** Count regime-driven rebalances and their COST.
  Confirm hysteresis + dwell + the cost gate keep regime-driven turnover low; a
  high regime-rebalance count means the thresholds are too tight or the dwell too
  short.
- **Crisis replay.** Replay historical co-movement crises (2008-style, COVID-style,
  crypto-deleveraging cascades) on the panel and confirm the absorption +
  turbulence measures escalate to STRESSED/CRISIS BEFORE the worst of the
  drawdown, and that `g_t` cuts gross in time.

Required statistics: block-bootstrap CIs (respect vol clustering and
cross-asset correlation); report whether one regime episode / one asset cluster /
one fold drives the result; feed the multiple-test control into SYSTEM's
deflation. Validation is **never imported by the live layers** (Invariant V).

---

## 10. Boundaries — REGIME vs LEVERAGE governor vs OPS kill-switch (no double-counting)

The three brakes act on **different axes** and must never double-count drawdown or
risk. This is the most important boundary in the spec.

| Brake | Axis it measures | Granularity | Shape | Action |
|---|---|---|---|---|
| **REGIME (this)** | GLOBAL market state (vol, correlation, turbulence, liquidity) | whole book, one scalar | gradual, sticky, `1.00→0.25` | MULTIPLIES gross / heat / N_concurrent |
| **LEVERAGE drawdown governor** | per-position / book EQUITY drawdown (peak-to-current) | book-wide multiplier on each position | graduated, `1.00→0.00` at 43% | TAPERS per-position size; flats at 43% |
| **OPS kill-switch** | OPERATIONAL / STATISTICAL safety fault (data outage, reconciliation break, statistical-anomaly halt) | binary, whole system | step, `1 / 0` | HALTS — binary safety stop |

- **REGIME multiplies; LEVERAGE governs per-position; OPS halts.** REGIME reads
  MARKET STATE (is the environment co-moving / turbulent / illiquid?). The
  LEVERAGE governor reads EQUITY DRAWDOWN (how much money has the book already
  lost from its peak?). These are orthogonal: a calm-market book can be in deep
  drawdown (governor cuts, regime does not), and a full-equity book can hit a
  correlation crisis (regime cuts, governor does not). They compose **multiplicatively
  only because they are independent axes** — and critically, NEITHER consumes the
  other's input: REGIME never reads equity drawdown (that would double-count the
  governor); the governor never reads the absorption ratio (that would double-count
  REGIME). (See Invariant NODOUBLE.)
- **REGIME is gradual; OPS is binary.** REGIME's floor is `g = 0.25` — it never
  takes the book fully flat. Taking the book to zero is OPS's job, on an
  operational or statistical SAFETY trigger (a kill-switch), or the LEVERAGE
  governor's job at ≥43% equity drawdown. REGIME is the macro BRAKE, not the
  kill-switch (Invariant FLOOR).
- **REGIME emits a scalar/state; it does not re-implement what it feeds.** REGIME
  does not size (LEVERAGE does), does not taper for per-position drawdown (the
  governor does), does not pick instruments (SELECTION does), does not halt (OPS
  does). It publishes `g_t`/`s_t` and the consumers act (Invariant SCOPE).

How `g_t` composes downstream, made explicit (so the no-double-counting is
auditable):

```
LEVERAGE final per position:
    L_final = min( seven ceilings, evaluated with H_book scaled to g_t · H_book )
              · drawdown_governor(equity_drawdown)         # SEPARATE axis, SEPARATE input
                                                           # governor reads equity, NOT regime
OPS:
    if kill_switch_active: FLAT_ALL, halt (binary)         # SEPARATE, binary, safety
```

`g_t` enters LEVERAGE's `min()`/heat scaling (one term among many — `min()`-not-product,
LEVERAGE P6); the governor is a SEPARATE post-`min()` scalar on a SEPARATE
(equity-drawdown) input; OPS is a SEPARATE binary halt. No quantity appears in two
of the three.

> **Invariant NODOUBLE — no axis is counted twice.** REGIME's inputs are
> market-state statistics ONLY (the four measures); it never reads equity
> drawdown or an operational fault flag. The LEVERAGE drawdown governor reads
> equity drawdown ONLY; it never reads a regime measure. OPS reads operational /
> statistical safety state ONLY. Tests: (1) feeding arbitrary equity-drawdown
> values to REGIME leaves `g_t` unchanged; (2) feeding arbitrary regime measures
> to the governor leaves its multiplier unchanged; (3) the three brakes' input
> sets are provably disjoint.

### 10.1 The VOL and CORRELATION boundary with LEVERAGE (NODOUBLE extended beyond drawdown)

NODOUBLE above pins the DRAWDOWN axis disjoint. But LEVERAGE acts on vol and
correlation too, and REGIME's `g^{vol}`/`g^{corr}` must not silently re-cut what
LEVERAGE already cut. Two overlaps to reconcile explicitly:

- **Vol.** LEVERAGE vol-targets each instrument: `L_vol = sigma_target / forecast_vol`
  uses the PER-INSTRUMENT forecast vol. LEVERAGE's adaptive Kelly fraction also cuts on
  the per-instrument realized-vol percentile. REGIME's `g^{vol}` must therefore key
  ONLY off quantities LEVERAGE does NOT capture: **CROSS-asset AGGREGATE vol**
  (`sigma_agg`, the equal-weight mean over the book — already cross-asset, Section 2.1),
  **vol-of-vol** (`vov`), and the **vol TERM-STRUCTURE** (`term`). REGIME NEVER reads the
  per-instrument `forecast_vol` vector that `L_vol` consumes. The two are on disjoint
  quantities: LEVERAGE divides EACH position by ITS OWN forecast vol; REGIME brakes the
  WHOLE book on the AGGREGATE/vol-of-vol/term-structure geometry that no single
  `L_vol` term sees.
- **Correlation (the load-bearing reconciliation).** LEVERAGE's adaptive Kelly fraction
  (LEVERAGE 5.2) already cuts on "correlation-matrix instability — a large change in the
  leading eigenvalue / average pairwise correlation," which is conceptually REGIME's
  absorption ratio. They MUST NOT both independently cut on the same systemic-correlation
  event. The boundary is pinned by SCOPE of the correlation measure, not by which fires
  first:
  - **LEVERAGE's Kelly-fraction correlation term is LOCAL/per-position-Kelly:** it
    governs how aggressively a SINGLE position's Kelly fraction is taken, keyed to that
    position's correlation to its current book neighbours (a per-instrument, local
    cut applied inside each position's sizing).
  - **REGIME's `g^{corr}` is BOOK-WIDE SYSTEMIC:** it is the absorption ratio of the
    ENTIRE cross-asset panel's correlation matrix (`delta_AR` on `C_t` over all `n_t`
    assets) — a single scalar describing the whole book's co-movement geometry, NOT any
    one position's neighbour-correlation.
  - These are computed on DISJOINT quantities (one per-instrument-local Kelly haircut,
    one book-wide eigenstructure scalar) and act through DISJOINT channels (LEVERAGE's
    per-position Kelly fraction vs REGIME's `H_book` gross scaling). **REGIME owns the
    book-wide systemic-correlation cut; LEVERAGE's Kelly term owns the per-position-local
    correlation haircut.** They are non-redundant because no single quantity feeds both.
  (See Invariant NODOUBLEVC.)

> **Invariant SCOPE — REGIME emits only a scalar/state, never an action or a
> size.** The entire output is `RegimeState`. No `place_order`, no size, no
> per-position drawdown taper, no instrument rank, no halt originates in
> `regime/`. Test: the engine API returns a `RegimeState` and nothing in
> `regime/` calls a sizing, selection, execution, or order primitive.

> **Invariant FLOOR — REGIME never flats the book.** `g_t ≥ 0.25` always
> (the `min()` over `g(s)` with `min g(s) = 0.25`). Taking the book to zero is the
> LEVERAGE governor (≥43% drawdown) or OPS (binary safety). Test: no input,
> however extreme, makes the engine emit `g_t < 0.25`.

---

## 11. Invariants (enforce structurally; ask before breaking)

**Invariant SCOPE — REGIME emits only `g_t`/`s_t`, never an action/size.** (Section 10.)

**Invariant MIN — multiplier is `min()` over the four sub-multipliers, never a
product.** `g_raw = min(g^{vol}, g^{corr}, g^{turb}, g^{liq})`. No
`g^{vol}·g^{corr}·…`, no weighted multiplicative regime score. The binding
sub-state is reported. Test: a measure not at a stress level has `g_sub = 1.0` and
does not affect `g_t`; two moderately-elevated measures do not compound below the
worse single one.

**Invariant HYST — hysteresis: entry and exit thresholds differ.** Escalation uses
the high (entry) threshold; de-escalation uses the lower (exit) threshold; the gap
is non-zero. Test: a measure oscillating between the entry and exit marks does NOT
change state.

**Invariant DWELL — minimum dwell holds.** A state is held ≥ `D_min` decision bars
before any de-escalation; escalation to a more-stressed state is always immediate.
Test: a de-escalation that satisfies the exit threshold but with `dwell_bars <
D_min` is blocked; an escalation is never blocked by dwell.

**Invariant CLOCK — two clocks: per-sub-state hysteresis, one GLOBAL dwell keyed on
`g(s_t)`.** Hysteresis memory is per measure (four independent band classifications);
there is exactly one dwell clock, on the global `g(s_t)`. It resets IFF `g(s_t)`
changes VALUE — NOT on a binding-identity handoff at unchanged `g(s_t)`, and NEVER
on a non-binding sub-state's oscillation (Section 3.3 i–iii). Test: the Section 6.2
fixture — a `corr→vol` handoff at constant `g(s_t)=0.50` keeps `dwell_bars` counting
(transition `hold`, reason `binding_handoff`), a non-binding measure's wiggle leaves
the clock untouched, and the legitimate next-bar de-escalation is permitted (a
resetting clock would wrongly block it).

**Invariant ARL — every escalation threshold has a stated `alpha` and an implied
`ARL0`, audited against realized.** (Section 3.6.)

**Invariant DISP — `delta_AR` is the Kritzman–Li standardized DISPLACEMENT, scaled
by the raw AR-level std, NOT a z-score of a mean difference.** `sigma_AR` is the std
of the single-bar AR level over the long window, not the standard error
`sigma_AR·sqrt(1/W_ARshort+1/W_ARlong)` of the short-minus-long mean. Test: the
denominator equals the raw-level std (a unit test asserts `sigma_AR` is `std(AR
level over W_ARlong)`, not the mean-difference SE); a code comment/guard forbids the
SE substitution that would inflate every `delta_AR` `~3.7×` and silently retune the
correlation thresholds (Section 2.2 labelling note).

**Invariant TRACE — the AR denominator is `n_t` by the correlation-matrix trace
identity, not a re-summed eigenvalue total.** `AR_t = (Σ_{j≤k} λ_j)/n_t`; the code
uses `n_t`, does not numerically re-sum all eigenvalues (round-off would jitter the
fixture), clamps tiny negative eigenvalues to 0, and needs no eigenvector tie-break
because `AR_t` is a cluster-invariant eigenvalue sum. Test: `Σ all λ = n_t` to
tolerance on a correlation matrix; `AR_t` is stable under a degenerate eigenvalue
cluster at the `k` cutoff.

**Invariant ARFLOOR — absorption has an observations-per-dimension floor symmetric
to COND.** Require `W_AR ≥ c_AR·n_t`; below it `C_t` is rank-deficient and the
top-`k` share is biased high, so `g^{corr}` is held FROZEN at its last valid value
(not reset to 1.0) and `ar_window_deficient` is flagged. Test: a panel with
`W_AR < c_AR·n_t` does not emit a live `g^{corr}` trigger; the freeze/flag path is
exercised.

**Invariant ARCOND — the absorption correlation matrix `C_t` is Ledoit–Wolf shrunk
BEFORE the eigendecomposition (the marginal-conditioning fix).** At marginal `T/N`
(`W_AR ≈ c_AR·n_t`, `c_AR = 2`) the raw sample-correlation spectrum is MP-dispersed and
`AR_t` is biased HIGH on noise even when ARFLOOR is satisfied. The eigendecomposition
runs on the constant-correlation Ledoit–Wolf shrunk `C_t` (same family as COND, unit
diagonal preserved so TRACE survives, denominator stays `n_t`). The LW intensity `δ_AR`
is closed-form, causal, data-derived — TEAL, NOT a BLUE/RED knob, and does NOT enter
`K_regime` (Section 2.2, Section 4.1). Cost is the same `O(n³)` (Section 2.6). Test:
on a pure-noise panel at `T/N ≈ 2`, `AR_t` on the SHRUNK `C_t` is materially less
biased-high than on the raw `C_t`; the eigendecomposition is fed the shrunk matrix; the
trace identity (`Σλ = n_t`) still holds post-shrink.

**Invariant ARDISPNORM — `delta_AR` is computed within a fixed-`n_t` stratum / frozen
across a panel-size change (the absorption analog of TURBNORM).** The MP dispersion
bias on `AR_t` depends on `W_AR/n_t`, and `n_t` varies bar-to-bar; a panel-size change
would inject a spurious `delta_AR` shift. The AR-level series feeding `AR_bar_short`,
`AR_bar_long`, `sigma_AR` is taken over the fixed-`n_t` stratum = all trailing bars whose
`n_t` lies within `frac_dn` of the current `n_t` (a current-bar-anchored band — the SAME
`frac_dn` criterion that gates the freeze, so the pooling rule and the freeze threshold are
one and the same test), pooled across non-contiguous sessions/days; when `n_t` changes by more than `frac_dn`, `g^{corr}` is
FROZEN (flagged `ar_nt_changed`) until the matching band-stratum has enough bars (a recurring
panel size re-uses its persistent cross-day history, so the freeze is transient). Test (mirrors
the TURBNORM panel-shrink test): on a panel whose CORRELATION STRUCTURE is unchanged but `n_t`
is dropped (session close / halts), `delta_AR` does NOT move (raw absorption WOULD have); a
genuine co-movement shift still moves it; and a trailing bar within `frac_dn` of the current
`n_t` is INCLUDED in the current stratum while one beyond `frac_dn` is EXCLUDED (the pooling
band EQUALS the freeze band).

**Invariant CONDNUM — both shrunk matrices pass a condition-number/min-eigenvalue
sanity gate, with a defined breach action.** After shrinkage, `cond_t = λ_max/λ_min ≤
cond_max` (RED) is checked for both `Sigma_t` (turbulence) and `C_t` (absorption). On
breach: turbulence falls back to the robust diagonal inverse (flag `cond_breach_turb`);
absorption FREEZES `g^{corr}` (flag `cond_breach_corr`). `cond_number` is in
`SubState.diag` for both. Test: (1) on the near-singular and rank-deficient synth
panels the POST-shrink `cond_t` stays below `cond_max` for the normal path; (2) a
pathological panel that breaches even post-shrink exercises the breach/fallback/freeze
path (turbulence does not emit a nonsense `d_t` spike; absorption does not emit a fresh
`g^{corr}`). (Section 2.3.)

**Invariant NODOUBLEVC — REGIME's vol/correlation signal is disjoint from what
LEVERAGE already cuts (NODOUBLE extended beyond drawdown).** REGIME's `g^{vol}` keys
ONLY off CROSS-asset AGGREGATE vol / vol-of-vol / term-structure, NEVER the
per-instrument `forecast_vol` vector that LEVERAGE's `L_vol` uses. REGIME's `g^{corr}`
is the BOOK-WIDE absorption ratio (whole-panel `C_t` eigenstructure); LEVERAGE's
Kelly-fraction correlation term is PER-POSITION-LOCAL (a position's neighbour-correlation
haircut). REGIME owns the book-wide systemic-correlation cut; LEVERAGE owns the
per-position-local one; they act on disjoint quantities through disjoint channels. Test:
(1) feeding REGIME the per-instrument forecast-vol vector leaves `g_t` unchanged
(REGIME does not read it); (2) the two correlation cuts are computed on provably
disjoint quantities (book-wide `AR_t` scalar vs per-position neighbour-correlation) and
do not both fire on the same input. (Section 10.1.)

**Invariant TURBNORM — turbulence percentile is taken on the dof-normalized
`d_t/n_t`, not raw `d_t`.** Because `n_t` varies bar-to-bar (staleness drops, session
closes), the raw `d_t ~ chi²_{n_t}` changes scale with `n_t`; the trailing percentile
is taken on `d_norm_t = d_t/n_t` (`E≈1` regardless of `n_t`) so a turbulence
escalation cannot be an artifact of the panel shrinking. Test: dropping assets
(lowering `n_t`) on an otherwise-unchanged-distribution panel does NOT shift
`turb_pct`; a genuine unusual vector does.

**Invariant STALEATTR — out-of-cadence escalations are LIQ-only and attributed to
`liq`.** Between scheduled recomputes only `g^{liq}` is fresh; `g^{vol/corr/turb}`
are frozen at their last cadence value (with `asof_bar` in `diag`). An event-driven
escalation is computed from fresh `g^{liq}` + frozen others and MUST report
`binding = liq`; it is forbidden to attribute a flash escalation to a stale
`corr/vol/turb`. Test: an intraday liq spike with stale (lower) corr cannot emit a
`binding = corr` escalation; the frozen measures' `asof_bar` lag the current bar.

**Invariant NODOUBLE2 — only ONE channel carries the gross multiplier (no `g_t²`
over-cut).** `g_t` reduces gross via LEVERAGE's `H_book` scaling alone
(`target_gross = g_t·book_gross_at_full`); the SELECTION count cap re-distributes
that gross across fewer names and is NOT a second multiplier. Realized gross is
bounded `≤ g_t·book_gross_at_full`, lowered further only by per-position LEVERAGE
ceilings (reported by LEVERAGE), never by a second `g_t` factor. Test: at CRISIS the
modelled realized gross is `0.25·full`, NOT `0.25²·full`; the count cap changes the
NUMBER of names, not the total-gross multiplier.

**Invariant CAP — `g_t ∈ [0.25, 1.0]`; it can only REDUCE exposure, never
increase it above the pillars' own ceilings.** `g_t ≤ 1.0` always, and it enters
LEVERAGE as one more `min()` term / a `≤1` scalar on `H_book`, so it cannot lift
leverage above the seven-ceiling `min()`. Test: at `g_t = 1.0` the downstream book
is identical to no-regime-overlay; `g_t` never produces a larger book than the
pillars alone.

**Invariant FLOOR — `g_t ≥ 0.25`; REGIME never flats the book.** (Section 10.)

**Invariant NODOUBLE — disjoint input sets across REGIME / governor / OPS; no
double-counting of drawdown or risk.** (Section 10.)

**Invariant ML — no ML / no learned regime classifier.** Every measure is a
transparent statistic; the state machine is rule-based hysteresis + dwell. Test:
no trained model, HMM, or learned classifier in `measures/` or `state/`.

**Invariant NOFIT — thresholds standardized/quantile-based, validated OOS via
SYSTEM, never PnL-fit.** (Section 4.)

**Invariant CAUSAL — everything is `F_t`-measurable.** Rolling causal windows,
causal percentiles (today's value within its OWN trailing window), trailing
mean/cov for turbulence, trailing correlation for absorption. No full-sample
standardization, no centred/future window. Future data lives ONLY in SYSTEM's
firewalled validation. Test: shifting any future bar leaves `g_t` at `t`
unchanged; a deliberately-injected future leak is caught.

**Invariant COND — covariance is conditioned before inversion.** Turbulence shrinks
`Sigma_t` (Ledoit–Wolf) and requires `W_turb ≥ c_cov·n_t`; below that it falls back
to a robust diagonal and flags it. The shrunk `Sigma_t` ALSO passes the post-shrink
condition-number gate (Invariant CONDNUM). Test: a near-singular synthetic covariance
does not produce a nonsense turbulence spike; the shrink/fallback path is exercised.

**Invariant CHEAP — `O(n³)` per coarse bar, never per-minute.** (Section 2.6.)

**Invariant COSTGATE — regime changes clear a cost/rebalance band; escalation is
never cost-blocked.** A state change implying less than `rebal_min_frac` gross
change is deferred; re-grossing UP must clear the cost cap; cutting gross DOWN for
safety is never blocked by cost. `book_gross_at_full` is the time-varying `g=1` gross
READ from LEVERAGE (Section 5.2), and clause (b) is LIVE when `current_gross` has
DRIFTED away from `g_prev·book_gross_at_full` (Section 6.3). Test: a sub-threshold gross
change is `regime_change_cost_gated` (including the drifted-gross 6.3 fixture); an
escalation is applied even with a high implied cost.

**Invariant NREBUILD — no neighbour re-implementation.** REGIME reads
EXECUTION/COST/LEVERAGE outputs; it does not re-model spread, funding, or the
sizing vol forecast, and does not rebuild Nautilus state/reconciliation. (Section 1.)

**Invariant C — RED unreachable from per-instrument config** (inherited). (Section 4.)

**Invariant V — validation never imported by live layers.** Nothing in `panel/`,
`measures/`, or `state/` imports from `validate/`.

**Invariant WE — the worked example (Section 6) is a regression fixture.** (Section 6.)

**Invariant DET — deterministic and decomposable.** Given identical
`(panel, exec_state, cost_state, book, prev, cfg)`, the engine emits identical
`RegimeState`; every emission carries the binding sub-state and a reason chain.
Test: diagnostics never alter `g_t` except through their defined sub-state band.

---

## 12. Synthetic data requirements (`synth/generate.py`)

Generate adversarial cross-asset panels so the module is testable with no external
data:
- **Calm, diversified panel** (low AR, low turbulence, normal vol) → `g_t = 1.00`,
  no sub-state binds.
- **Correlation-to-1 spike** (absorption ratio jumps; `delta_AR` crosses +2.0) →
  `g^{corr}` drives CRISIS; this is the Section 6 worked-example scenario.
- **Volatility spike** (aggregate realized vol percentile + vol-of-vol + steep
  term-structure) → `g^{vol}` binds.
- **Turbulence day** (a single statistically-unusual cross-asset return vector;
  large Mahalanobis distance) without a sustained correlation shift → `g^{turb}`
  binds, then reverts (tests dwell on the way back up).
- **Liquidity / funding blowout** (EXECUTION spread cascade + COST funding spike)
  → `g^{liq}` binds.
- **Oscillating measure** (a statistic dithering across a single threshold) →
  hysteresis + dwell hold the state; NO whipsaw, NO repeated rebalance.
- **Near-singular covariance** (`n_t` near `W_turb`, highly collinear assets) →
  Ledoit–Wolf shrink / diagonal fallback engages; no nonsense turbulence spike
  (Invariant COND).
- **Rank-deficient correlation window** (`W_AR < c_AR·n_t`) → absorption freezes
  `g^{corr}` and flags `ar_window_deficient`; no spurious top-`k`-share trigger
  (Invariant ARFLOOR).
- **Marginal-conditioning noise panel** (`T/N ≈ 2`, pure i.i.d. noise, NO true
  co-movement) → `AR_t` on the SHRUNK `C_t` is materially LESS biased-high than on the
  raw `C_t`, so `g^{corr}` does NOT fire on noise; the trace identity survives the
  shrink (Invariant ARCOND).
- **Correlation-structure-fixed panel-shrink** (same correlation STRUCTURE, `n_t`
  dropped by session close / halts) → `delta_AR` does NOT move (raw absorption WOULD
  have); `g^{corr}` freezes/flags `ar_nt_changed` across the `n_t` step (Invariant
  ARDISPNORM — the absorption analog of the TURBNORM panel-shrink panel).
- **Condition-breach panel** (a near-duplicated asset pair that leaves the shrunk
  matrix `cond > cond_max`) → turbulence falls back to the diagonal inverse
  (`cond_breach_turb`, no nonsense `d_t` spike) and absorption freezes `g^{corr}`
  (`cond_breach_corr`); the breach path is exercised (Invariant CONDNUM).
- **Calm null panel for the ARL audit** — a long pure-calm stretch with no true
  regime change, used to measure the realized per-bar false-escalation rate against
  the `alpha_k`/`ARL0` budget (Invariants ARL; Section 3.6d).
- **Panel-shrink panel** — same return distribution but `n_t` dropped (session close
  / halts) → `turb_pct` (on `d_norm = d_t/n_t`) does NOT move; the raw-`d_t`
  percentile WOULD have, proving the normalization (Invariant TURBNORM).
- **Intraday liq flash on stale corr** — a spread/funding spike between scheduled
  recomputes while the frozen `corr` sits lower → escalation fires with
  `binding = liq`, never `binding = corr` (Invariant STALEATTR).
- **Multi-measure binding handoff** — the Section 6.2 `t0..t3` panel: `corr→vol`
  handoff at constant `g(s_t)`, non-binding oscillation, dwell continuity (Invariant
  CLOCK).
- **Sub-threshold gross wiggle** → the cost gate defers the change
  (`regime_change_cost_gated`).
- **Drifted-gross deadband panel** — `current_gross` has DRIFTED below
  `g_prev·book_gross_at_full` (positions closed, book shrank) so a hysteresis+dwell-PASSED
  state change implies a sub-`rebal_min_frac` `delta_gross` → clause (b) DEFERS the
  change (`regime_change_cost_gated`), the Section 6.3 fixture (Invariant COSTGATE/WE).
- **Per-minute liq-flash null** — a long calm minute-bar stretch with intermittent
  single-minute spread/funding blips → the minute-level dwell `D_liq_min` suppresses
  them and the realized INTRADAY liq-escalation rate matches `ARL0^liq_min`, NOT the
  decision-bar budget (Invariant ARL (6), Section 3.6 b′).

Behavioral assertions: `g_t ∈ [0.25, 1.0]` always; `g_t` is the `min()` over
sub-multipliers and never their product; realized gross at CRISIS is `g_t·full`, not
`g_t²·full` (Invariant NODOUBLE2); escalation is immediate, de-escalation is
dwell-and-hysteresis-gated and one step at a time; a binding handoff at constant
`g(s_t)` does not reset the dwell clock (Invariant CLOCK); the realized calm-panel
escalation rate matches the `alpha_k`/`ARL0` budget within CI (Invariant ARL); the
absorption crisis cuts gross before the drawdown in crisis-replay (`ARL1≈1`); an
injected future bar never changes a past `g_t`; all three worked examples (6.1, 6.2, 6.3)
reproduce bit-for-bit.

---

## 13. Module acceptance gates (CI)

### Panel
Pass when: the cross-asset return panel is causal, calendar-aligned across 24/7
crypto + session-bound equities + Dukascopy commodities, causally de-meaned, with
staleness drops shrinking `n_t`; no NaN/inf; no future bar enters (Invariant CAUSAL).

### Volatility regime
Pass when: `sigma_agg`, `vol_pct` (causal percentile in its own trailing window),
`vov`, and term-structure compute correctly; DVOL-vs-realized is an AMBER switch
not a forked path; the sub-state classifies under the hysteresis bands.

### Absorption ratio
Pass when: `AR_t` = top-k eigenvalue share of the causal correlation matrix with
`k = ceil(kappa·n_t)` and denominator `n_t` by the trace identity (Invariant TRACE);
the standardized shift `delta_AR` matches the worked example (`0.750`, `+2.20`) and
its denominator is the raw AR-level std, NOT the mean-difference SE (Invariant DISP);
the `W_AR ≥ c_AR·n_t` floor freezes `g^{corr}` on a rank-deficient window (Invariant
ARFLOOR); `C_t` is Ledoit–Wolf (constant-correlation) shrunk BEFORE the
eigendecomposition so `AR_t` is not MP-biased-high at marginal `T/N` (Invariant
ARCOND), with the LW intensity TEAL/closed-form (not in `K_regime`); the shrunk `C_t`
passes the post-shrink condition gate `cond ≤ cond_max` and freezes `g^{corr}` on
breach (Invariant CONDNUM); `delta_AR` is computed on a fixed-`n_t` stratum / frozen
across an `n_t` step so a panel-size change cannot manufacture a `delta_AR` shift
(Invariant ARDISPNORM); the eigendecomposition is `O(n³)` and runs only on the regime
cadence (Invariant CHEAP).

### Turbulence
Pass when: `d_t` is the Mahalanobis distance against the trailing causal mean/cov;
`Sigma_t` is Ledoit–Wolf shrunk and `W_turb ≥ c_cov·n_t` is enforced with a
diagonal fallback below it (Invariant COND); the shrunk `Sigma_t` passes the
post-shrink condition gate `cond ≤ cond_max` and falls back to the diagonal inverse on
breach (Invariant CONDNUM); the trailing causal percentile is taken on the
dof-normalized `d_norm = d_t/n_t` so a panel-size change cannot manufacture a
turbulence spike (Invariant TURBNORM).

### Liquidity stress
Pass when: `liq_stress = max(spread_blowout, funding_extreme)` is a pure aggregation
of EXECUTION/COST outputs (Invariant NREBUILD), not a re-modelled spread or funding;
its false-alarm `alpha` is derived from the max-of-two-percentiles null (`1 − p²`, NOT
`1 − p`, Section 3.6a) with the corrected decision-bar `ARL0^liq`; the event-driven
liq path is held to a SEPARATE per-minute budget `alpha^liq_min`/`ARL0^liq_min` behind
a minute-level dwell `D_liq_min` (Section 3.6 b′, 3.5; Invariant ARL).

### State machine
Pass when: global state = worst sub-state = `min()` (Invariant MIN); hysteresis is
PER-sub-state and an oscillating measure does not change its own state (Invariant
HYST); the single GLOBAL dwell clock keys on `g(s_t)` — resets on a `g(s_t)` change,
CONTINUES through a binding-identity handoff at constant `g(s_t)`, ignores a
non-binding wiggle (Invariant CLOCK, the Section 6.2 fixture); minimum dwell holds
and escalation is immediate while de-escalation is gated one step at a time
(Invariant DWELL); out-of-cadence escalations are LIQ-only/LIQ-attributed (Invariant
STALEATTR); the escalation thresholds carry their `alpha_k`/`ARL0` budget and the
realized calm-panel rate is audited against it (Invariant ARL); only ONE channel
carries the gross multiplier (Invariant NODOUBLE2); `g_t ∈ [0.25, 1.0]` (Invariants
CAP, FLOOR); deterministic and decomposable (Invariant DET).

### Cost gate
Pass when: a sub-threshold gross change is deferred (`regime_change_cost_gated`),
INCLUDING the drifted-`current_gross` case where a hysteresis+dwell-PASSED step implies
a sub-`rebal_min_frac` `delta_gross` (Section 6.3); `book_gross_at_full` is the
time-varying `g=1` gross read from LEVERAGE, `F_t`-measurable (Section 5.2);
re-grossing up must clear the cost cap; cutting gross down for safety is never
blocked (Invariant COSTGATE).

### Boundary / composition
Pass when: REGIME emits only a `RegimeState` (Invariant SCOPE); REGIME's inputs are
disjoint from the LEVERAGE governor's and OPS's (Invariant NODOUBLE); REGIME's
`g^{vol}` keys only off cross-asset aggregate vol/vov/term (never the per-instrument
forecast vol) and `g^{corr}` is the book-wide absorption ratio disjoint from LEVERAGE's
per-position-local Kelly correlation term (Invariant NODOUBLEVC); `g_t` enters
LEVERAGE as one more `min()` term / `≤1` heat scalar and SELECTION as a
concurrent-position cap; at `g_t = 1.0` the downstream book equals no-overlay
(Invariant CAP).

### No-overfit / validation
Pass when: thresholds are standardized/quantile-based and REGIME's BLUE grid is
registered in SYSTEM's persistent trial ledger so its DOF are deflated (Invariants
NOFIT, ML); the registered BLUE-grid cardinality `K_regime ≤ K_regime_max` and the
Ledoit–Wolf intensities are TEAL/closed-form (not searched DOF) so the regime layer
cannot silently grow its degrees of freedom (Section 4.1, Invariant NOFIT); no code
path selects a breakpoint by maximising in-sample PnL; the worked examples (6.1/6.2/6.3)
reproduce bit-for-bit (Invariant WE).

---

## 14. Build order

1. `types.py` and `synth/generate.py` (adversarial panels of Section 12).
2. `config/` with TEAL/AMBER/RED/BLUE; test Invariant C.
3. `panel/build.py` (causal, calendar-aligned, de-meaned; Invariant CAUSAL).
4. `measures/vol_regime.py` (reads LEVERAGE vol forecasts; DVOL AMBER switch).
5. `measures/absorption.py` (Ledoit–Wolf-shrunk eigendecomp, `delta_AR`; the
   worked-example anchor; trace-identity denominator + `W_AR≥c_AR·n_t` floor +
   post-shrink condition gate + fixed-`n_t` stratum; Invariants
   TRACE/DISP/ARFLOOR/ARCOND/ARDISPNORM/CONDNUM).
6. `measures/turbulence.py` (Ledoit–Wolf shrink + post-shrink condition gate +
   Mahalanobis on `d_t/n_t`; Invariants COND/CONDNUM/TURBNORM).
7. `measures/liquidity.py` (aggregate EXECUTION/COST; Invariant NREBUILD).
8. `state/bands.py` (measure → ordered sub-state under PER-sub-state entry/exit
   thresholds; carries the `alpha_k`/`ARL0` budget metadata incl. the liq `1−p²` null
   and the per-minute `alpha^liq_min`/`D_liq_min`; Section 3.6).
9. `state/machine.py` (worst-sub-state `min()`, per-sub-state hysteresis, ONE global
   dwell clock on `g(s_t)`, the minute-level liq dwell on the event path, single gross
   channel; Invariants MIN/HYST/DWELL/CLOCK/STALEATTR/NODOUBLE2/NODOUBLEVC/CAP/FLOOR).
10. `state/cost_gate.py` (rebalance band; escalation-never-blocked; Invariant COSTGATE).
11. `state/emit.py` and `engine_api.py` (assemble `RegimeState`; Invariant SCOPE/DET).
12. `validate/threshold_stability.py`, `validate/baselines.py` (incl. the rejected
    continuous-multiplier baseline), `validate/ablation.py` (which measure binds +
    the ARL audit: realized calm-panel escalation rate vs the `alpha_k`/`ARL0`
    budget, and crisis-replay detection delay `ARL1`; Invariant ARL).
13. `validate/backtest.py` (hooks into SYSTEM purged WF / CPCV; register BLUE grid
    in the trial ledger), `validate/stats.py`, reports.

Each step complete only when its acceptance gate (Section 13) is green. A module
that fails its gate does not advance.

Stop after step 13. The remaining questions — which of the four measures actually
binds on your data, whether absorption or turbulence leads in your crises, whether
the dwell is long enough to suppress whipsaw without missing a real escalation —
are answered by RUNNING the validation and crisis-replay harness through SYSTEM on
real data, not by adding measures.

---

## 15. References and rationale

- **Absorption ratio:** Kritzman, Li, Page & Rigobon (2011), *Principal Components
  as a Measure of Systemic Risk* — the fraction of total variance captured by the
  top-k eigenvectors of the asset return correlation matrix; rises when assets
  co-move and diversification fails. The standardized-shift trigger (short vs long
  trailing mean over the long-window std) is from the same work.
- **Turbulence / financial turbulence:** Kritzman & Li (2010), *Skulls, Financial
  Turbulence, and Risk Management* — the Mahalanobis distance of the current return
  vector from its historical mean given its covariance flags statistically unusual
  cross-asset days; persistent and forward-informative about risk.
- **Covariance shrinkage:** Ledoit & Wolf (2004), *A well-conditioned estimator for
  large-dimensional covariance matrices* — closed-form, no-tuning shrinkage toward a
  structured target; required because turbulence inverts an `n×n` covariance on a
  finite window (Invariant COND).
- **Vol regime / vol-of-vol / term-structure:** realized-vol percentile and
  vol-of-vol as regime state; the implied-vol analog (Deribit DVOL for crypto)
  where published, else realized (P6-conservative).
- **Hysteresis / buffering:** Carver, *Systematic Trading* / *Advanced Futures
  Trading Strategies* — position buffering to suppress turnover on noise; the
  regime state machine applies the same discipline to the global brake (P4), as
  EXECUTION's no-churn cadence does at the order level.
- **Diversification premise:** the absorption ratio defends the exact assumption
  SELECTION's Carver dynamic optimisation and LEVERAGE's MCTR portfolio ceiling
  rely on — that cross-asset correlations are stable enough for diversification to
  reduce risk (LEVERAGE P4: in a crisis correlations → 1 and netted portfolio
  margin explodes).
- **Statistical process control / run length (P7b, Section 3.6):** Page (1954,
  CUSUM); Roberts (1959, EWMA control charts); Lorden (1971) and the
  Average-Run-Length framework — a control chart is characterized by its in-control
  ARL0 (false-alarm rate) and out-of-control ARL1 (detection delay), not by a bare
  threshold. REGIME's threshold + dwell + one-step hysteresis is the discrete
  multivariate analogue (the dwell is the decision interval / headstart that turns a
  loose per-bar rate into a long run length); the brake is stated with an explicit
  `alpha`/`ARL0` budget rather than a level asserted without a false-alarm rate.
  Multivariate change-point context: Hotelling `T²` / Mahalanobis monitoring (the
  turbulence measure is exactly a `T²` statistic) and absorption-ratio shifts as a
  systemic change-point signal (Kritzman–Li).
- **Out-of-sample discipline / deflation:** Bailey & López de Prado — the deflated
  / probabilistic Sharpe ratio; REGIME's thresholds are paid for in SYSTEM's joint
  multiple-testing tax (SYSTEM P6), which is why fitting breakpoints to backtest
  PnL is forbidden (P5). The BLUE grid is searched OOS *within* the Section 3.6 ARL
  budget — the ARL pins the false-alarm rate, the DSR gate pins that the ranking
  among candidates is not overfit; the two are complementary (the grid search ranks,
  it never pins a false-alarm rate — P7b).

---

## 16. Non-goals (belong to sibling modules)

Position sizing and the seven ceilings (LEVERAGE.md); the per-position / book
EQUITY-drawdown governor (LEVERAGE.md §6 — a SEPARATE axis, Section 10); the binary
operational / statistical kill-switch (OPS — a SEPARATE binary halt); instrument
selection and ranking (the Carver Ch.25 SELECTION module — REGIME caps the COUNT,
SELECTION picks the NAMES); entry detection (BREAKOUT.md); exit / trend-life
management (TREND.md); order tactics and microstructure (EXECUTION.md); fee /
spread / funding modelling (COST.md); the composed-chain validation and joint
deflation framework (SYSTEM.md — REGIME plugs INTO it). REGIME measures the global
market state and emits one gradual, sticky, cost-gated multiplier. It never sizes,
never tapers for equity drawdown, never halts, never picks names, and never trades.
