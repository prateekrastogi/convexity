# LEVERAGE.md — build spec for an independent leverage / position-sizing engine

This file is the authoritative build spec. Build to the contracts and
invariants below. Where this file and your own instinct disagree, this file
wins. Ask before deviating from any **Invariant**.

This module is INDEPENDENT and composes with two sibling modules without
importing their internals:
- BREAKOUT.md decides trade / no-trade (entry gate).
- TREND.md decides hold / reduce / exit on an open position (trend-life + exit).
- **LEVERAGE.md (this) decides HOW MUCH** — it converts a qualified signal into
  an instrument-specific leverage / position size, subject to survival
  constraints.

It answers exactly one question, on entry and on each re-evaluation:

> Given a risk budget, volatility, stop distance, correlation, margin, and
> liquidity — what is the maximum SAFE leverage for this instrument?

Nothing else. It does not generate entries, predict trends, or manage the
profitable exit. Scope: full cross-asset (commodity futures, metals, US
equities, crypto, FX, and index futures), under margin or portfolio-margin
regimes with **per-instrument leverage**.

---

## 0. Governing principles (read before anything else)

**P1 — Leverage is DERIVED, never chosen.** For a leveraged position,
`capital_loss% ≈ L × adverse_underlying_move% + carry`. Fixing the per-position
capital-loss cap and a volatility-appropriate stop distance uniquely determines
the maximum leverage. **The 12–15% cap in this spec is a loss cap on the
position's committed margin / position risk capital, NOT on total portfolio
equity.** Portfolio equity loss is controlled separately by portfolio heat,
cluster caps, margin stress, and the drawdown governor. Leverage is the
dependent variable.

**P2 — The edge/risk firewall (the most important rule in this file).**
Leverage is governed ONLY by statistically MEASURABLE RISK quantities:
volatility, correlation, drawdown, gap/tail statistics, liquidity, margin.
Leverage is governed by ZERO EDGE quantities: expected return, trend-strength
confidence, expected trend length, breakout score, expected favorable
excursion. Those are unmeasurable at the intraday-to-1-week horizon
(t = SR·√T) — the exact conclusion BREAKOUT.md and TREND.md reached — and they
belong to the ENTRY decision, not to sizing. **Leverage rises when risk is low,
never because edge looks high.** Any edge-derived term entering the leverage
math is the bug. (See Invariant A.)

**P3 — Kelly outputs a RISK BUDGET, not leverage.** Kelly is used as a long-run
capital-growth UPPER BOUND under parameter uncertainty. Production sizing uses
**Half Kelly** because edge-estimation uncertainty dominates at
intraday-to-weekly horizons. Full Kelly is forbidden: it expects drawdowns in
the 40–50% region as a matter of course, which under a 43% portfolio governor
means repeated full-book liquidation — maximizing liquidation, not growth.

**P4 — Portfolio margin makes portfolio heat the PRIMARY constraint, not an
optimization perk.** Portfolio margin nets offsetting risks, so it can offer
more leverage — but it ASSUMES correlations are stable. In a crisis correlations
go to 1, netted margin requirements EXPLODE while P&L collapses simultaneously.
Therefore size as if diversification can vanish: the correlation-aware portfolio
ceiling is load-bearing precisely BECAUSE margin is portfolio-level.

**P5 — Separate normal risk from catastrophic risk.** Two risk layers, validated
and reasoned about separately:
- NORMAL layer (binds on most trades): volatility targeting, conditional MAE
  stop, Half Kelly budget.
- CATASTROPHIC layer (binds only in stress): EVT / Expected Shortfall, historical
  gap analysis, portfolio-margin stress test.
Most trades are governed by the normal layer; the catastrophic layer truncates
the tail.

**P6 — Final leverage is a min() of independent ceilings, NOT a product of
scores.** Each ceiling corresponds to a distinct failure mode. A product of
weighted "scores" (vol × confidence × regime × …) is an unvalidatable
degrees-of-freedom explosion and is forbidden. The min() is conservative by
construction: the binding constraint is always the nearest failure mode.

**P7 — The regime macro-multiplier `g_t` enters through exactly ONE channel and
only de-risks.** REGIME.md emits a global market-state multiplier `g_t ∈ [0.25,
1.0]` (its Invariant CAP) derived from market-state geometry ONLY — cross-asset
aggregate vol / vol-of-vol / term-structure, book-wide systemic correlation
(absorption ratio), turbulence, liquidity — and NEVER from equity drawdown or an
operational fault (REGIME Invariant NODOUBLE). LEVERAGE consumes `g_t` by scaling
the **book-heat ceiling input** `H_book → g_t · H_book` inside `PortfolioCeiling`
(Section 7), so the regime cut surfaces through `L_portfolio` as one more `min()`
term — NOT as a separate post-min scalar, and NOT applied to `L_vol`, the Kelly
fraction, or any other ceiling. This is the GROSS channel of REGIME's contract
(its Section 3.4). The drawdown governor (P8) is a wholly separate axis. Because
`g_t ≤ 1`, it can only lower `H_book` and therefore only de-risk (Invariant R).

**P8 — The drawdown governor and the regime multiplier compose multiplicatively
ONLY because their input axes are provably disjoint.** The drawdown governor
(Section 6) is a per-book EQUITY-drawdown taper; it reads peak-to-current equity
ONLY and NEVER reads a regime measure (vol, correlation, turbulence, liquidity).
`g_t` reads market-state ONLY and NEVER reads equity drawdown. A calm-market book
in deep drawdown is cut by the governor and not by `g_t`; a full-equity book in a
correlation crisis is cut by `g_t` and not by the governor. They multiply
(`L_portfolio` carries `g_t` via `H_book`; the governor is a post-min scalar)
WITHOUT double-counting precisely because no quantity feeds both. This boundary
is pinned verbatim against REGIME Section 10 / Invariant NODOUBLE; see Invariant R.

Forbidden in v1: any edge/confidence/expected-length term in sizing (P2); Full
Kelly (P3); a multiplicative score formula (P6); return-optimized portfolio
weights via Σ⁻¹μ (the μ vector is unmeasurable and matrix inversion amplifies
its error — use risk-based allocation instead, Section 7); HRP / hierarchical
clustering machinery (overfit for a small retail book — see Section 7); ML of
any kind; per-instrument optimization of the RED constants on historical P&L.

---

## 1. The seven ceilings (the whole engine in one equation)

For instrument i, the leverage is the minimum of seven independent failure-mode
ceilings, then clamped by a hard per-class guardrail `L_class_max` (§8.6.6):

```
L_i_pre_governor =
    min( L_stop , L_vol , L_kelly , L_tail ,
         L_portfolio , L_margin , L_liquidity ,
         L_class_max )   # hard per-class guardrail (§8.6.6), not a modelled failure-mode ceiling

L_i_final = L_i_pre_governor * drawdown_governor(equity_drawdown)
```

where `L_portfolio` is computed with the **regime-scaled book-heat ceiling**
`g_t · H_book` (Section 7, P7). The regime multiplier `g_t ∈ [0.25, 1.0]` from
REGIME.md therefore enters the `min()` THROUGH `L_portfolio` — it is **not** a
separate eighth `min()` term and **not** a post-min scalar (only the drawdown
governor is post-min). When the regime is the active cut, the binding ceiling
reports as `portfolio` with a reason code distinguishing the regime-heat cut from
a correlation-MCTR cut (Section 9, Invariant R). `g_t` and the drawdown governor
compose multiplicatively on disjoint input axes (P8). The regime cut is applied to
the `H_book` channel ONLY: it does NOT also scale `L_vol` or the Kelly fraction
(P7, Invariant R; mirrors REGIME Invariant NODOUBLE2/NODOUBLEVC).

| Ceiling      | Failure mode it guards        | Layer        | Driven by (MEASURABLE only)                  |
|--------------|-------------------------------|--------------|----------------------------------------------|
| L_stop       | normal single-trade loss      | normal       | horizon volatility, conditional MAE, cap c   |
| L_vol        | volatility-regime change      | normal       | vol-target / forecast vol (YZ+EWMA+HAR-RV)   |
| L_kelly      | long-run growth / over-bet    | normal       | Half-Kelly RISK BUDGET (adaptive fraction)   |
| L_tail       | rare extreme / gap-through    | catastrophic | EVT / Expected Shortfall, historical gaps    |
| L_portfolio  | correlation concentration + regime cut | catastrophic | covariance, MCTR, regime-scaled `g_t·H_book` |
| L_margin     | operational / margin call     | both         | broker margin, financing, free-margin buffer |
| L_liquidity  | cannot exit without impact    | both         | ADV/depth/spread/impact/participation limits |

The **adaptive Kelly fraction** (Section 5) is consumed once inside
`L_kelly`. It is NOT multiplied again in the final engine. The only post-min
scaler is the **drawdown governor** (Section 6), which can reduce exposure but
cannot increase it. The **regime multiplier** `g_t` is consumed once inside
`L_portfolio` (via `g_t·H_book`); it is NOT applied anywhere else (P7).

No ceiling may take an edge quantity as input (Invariant A).

---

## 2. Architectural split

Four layers, kept independent for testability:

1. **Inputs layer:** receives the qualified signal (side, entry, instrument,
   per-position capital cap c) from upstream, the market-state risk inputs
   (volatility, covariance, gap history, margin, liquidity), and the REGIME
   macro-multiplier `g_t ∈ [0.25,1.0]` (consumed ONLY to scale `H_book`, P7).
   Receives NO edge score and NO equity-drawdown-derived regime input (Invariant
   R: `g_t` is market-state-only; equity drawdown reaches the engine only via the
   governor's own `portfolio_peak`/`portfolio_equity`).
2. **Ceiling layer:** seven independent estimators, each emitting one ceiling
   `L_*` plus diagnostics. Each is testable in isolation.
3. **Combine layer:** computes the adaptive Kelly fraction, passes it into
   `KellyCeiling`, scales `H_book → g_t·H_book` for `PortfolioCeiling` (P7), takes
   the min() (including the class-cap clamp), applies the drawdown governor exactly
   once on the separate equity-drawdown axis (P8), then performs lot/contract
   rounding. It emits final leverage / size and the binding-ceiling or rejection
   reason code (a regime cut surfaces as `portfolio`/`regime_heat_cut`, Invariant R).
4. **Validation layer:** backtests, stress tests, ablations; never imported by
   the live ceiling/combine layers.

Why: a bad tail model can be isolated from a bad stop model; each ceiling
ablated independently; the binding ceiling is always explicit.

---

## 3. Directory layout

```text
leverage/
+-- config/
|   +-- defaults.py          # PolicyDefaults: RED shared values (c, fractions, thresholds)
|   +-- asset_class.py       # AssetClassPreset: validated per-class caps & params
|   +-- instrument.py        # InstrumentSpec: AMBER metadata only
|   +-- margin.py            # broker margin + financing model (portfolio-margin aware)
|   +-- resolve.py           # single merge point (RED unreachable from instrument)
+-- ceilings/                # CEILING LAYER — each emits one L_*, never an action
|   +-- stop.py              # L_stop: capital cap / horizon-vol conditional-MAE stop
|   +-- vol_target.py        # L_vol: volatility targeting on forecast vol
|   +-- kelly.py             # L_kelly: Half-Kelly risk budget -> ceiling (adaptive frac)
|   +-- tail.py              # L_tail: EVT / Expected Shortfall + gap model (CATASTROPHIC)
|   +-- portfolio.py         # L_portfolio: covariance + MCTR + portfolio heat (CATASTROPHIC)
|   +-- margin.py            # L_margin: broker margin, financing, free-margin buffer
|   +-- liquidity.py         # L_liquidity: impact, spread, participation, exit-time ceiling
+-- vol/
|   +-- forecast.py          # YangZhang + EWMA + HAR-RV blend, noise-robust (bipower)
+-- combine/
|   +-- governor.py          # graduated drawdown risk multiplier
|   +-- kelly_fraction.py    # adaptive 1/2 -> 1/3 fraction, RULE-BASED on observables
|   +-- lot_sizer.py         # contract/lot/share rounding; reject if min unit breaks risk
|   +-- engine.py            # LeverageEngine: min() of ceilings, governor, fraction
+-- allocate/
|   +-- risk_budget.py       # ERC / MCTR split of risk across simultaneous signals
|   +-- clusters.py          # correlation clustering (simple, capped) for budget sharing
+-- priors/
|   +-- market_dna.py        # slow monthly/quarterly priors (long-run vol, gap baseline)
+-- validate/
|   +-- labels.py            # realized loss vs planned loss, gap-through events
|   +-- stress.py            # crisis / correlation-to-1 / margin-explosion stress tests
|   +-- baselines.py         # fixed-fraction, single-ceiling, full-Kelly (as a cautionary baseline)
|   +-- ablation.py          # which ceilings actually bind / contribute
|   +-- walkforward.py       # purged walk-forward of realized vs planned loss
|   +-- stats.py             # bootstrap CIs, multiple-test control
|   +-- reports.py
+-- synth/
|   +-- generate.py          # adversarial paths: gaps, vol spikes, correlation breaks
+-- types.py
+-- engine_api.py            # public entry point used by the trading layer
tests/
+-- ...                      # unit, firewall, stress, parity, acceptance, ablation
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest. No ML frameworks.

---

## 4. Parameter governance

Every parameter is exactly one bucket.

- **TEAL — fitted per instrument from its own past data, at runtime.** Examples:
  forecast-vol scale, conditional-MAE distributions per regime bucket, EVT tail
  parameters (GPD shape/scale) fitted to the instrument's loss tail, historical
  gap distribution by session boundary, covariance/correlation estimates.
- **AMBER — supplied market metadata.** Asset class, venue, timezone, session
  calendar, tick/lot/multiplier, margin regime, financing rate, bid/ask
  availability, roll/expiry metadata.
- **RED — shared economic/modelling choices.** Per-position committed-margin
  loss cap c (0.12–0.15 of position margin / committed risk capital, not total
  portfolio equity), base Kelly fraction (0.5) and stressed fraction (≈0.33), the
  fraction-switch thresholds, the drawdown-governor breakpoints, the
  Expected-Shortfall confidence level and budget, the portfolio-heat ceiling,
  the gap-survival ruin threshold, the free-margin buffer, σ_target, max
  participation rate, max spread/impact tolerance, and lot-rounding policy.
- **BLUE — finite validated grids** (optional, class-level only): small
  candidate sets for things like the vol-blend weights, searched offline with
  promotion gates, NEVER per instrument on P&L.

> **Invariant C — RED unreachable from per-instrument config.** RED enters only
> via `resolve_config(defaults=...)`. Per-class overrides go through a separate
> `AssetClassPreset` argument and only when validated. If c, a Kelly fraction, a
> governor breakpoint, or an ES level appears as a per-instrument field, that is
> the bug.

---

## 5. The Kelly ceiling and the adaptive fraction

### 5.1 Kelly outputs a risk budget (P3)

Full-Kelly leverage is `f* = μ/σ²` (or portfolio `Σ⁻¹μ`). This is NOT used
directly. Instead:

1. Compute a Half-Kelly RISK BUDGET — the maximum fraction of capital to put at
   risk — using the fractional Kelly fraction φ (Section 5.2).
2. Convert that risk budget into a leverage ceiling `L_kelly` given the stop
   distance (so the budget and the stop are consistent).
3. `L_kelly` enters the min() as one ceiling among seven. If the Kelly ceiling
   exceeds the stop/tail/portfolio ceilings, it does NOT bind — and that is the
   intended behavior, because the Kelly estimate is the least trustworthy input.
4. φ is applied exactly once: inside `KellyCeiling`. The combine engine must not
   multiply the final leverage by φ again.

**The explicit budget→leverage map (pinned, reproducible).** All quantities are
RETURNS-based and HORIZON-scaled to the same holding horizon `T` as the stop
(Invariant S), so the map is dimensionally consistent with `L_stop=(c−carry)/d`
and `L_vol=σ_target/forecast_vol`:

```
μ_hat        = forecast excess return over the holding horizon  (Kelly-only μ, Inv. A)
σ_hat        = forecast return volatility over the SAME horizon  (= forecast_vol·√(T)
               scaled consistently with the stop's σ·√T, Invariant S)
f_full       = μ_hat / σ_hat²          # full-Kelly leverage (the classic μ/σ²)
b            = φ · f_full              # Half-Kelly RISK BUDGET as a leverage  (φ ≤ 0.5)
d            = stop_distance           # the SAME fractional adverse move as L_stop (Section 8.1)
L_kelly      = b                       # leverage IS the budget here: f_full is already a
                                       # leverage (return/variance, both returns-based),
                                       # so the fractional-Kelly budget b is L_kelly directly.
```

Equivalently, "budget at risk = L·d": at leverage `L_kelly` the budgeted loss at
the stop is `L_kelly · d = b · d`, which is the fraction of capital the Half-Kelly
bet puts at risk through this stop. This is the SAME `L·d` accounting `L_stop`
uses, so when `L_kelly > L_stop` the stop binds and the Kelly budget is slack — the
intended behaviour, because the Kelly estimate is the least trustworthy input.

**Leverage-invariance (pinned, with a test).** `μ_hat`, `σ_hat`, and `d` are all
RETURNS-based (fractions of price), so scaling the instrument's notional / contract
size / point value leaves `f_full`, `b`, and therefore `L_kelly` UNCHANGED — a pure
leverage. (Section 15 Kelly-ceiling gate tests this: doubling contract multiplier
leaves `L_kelly` invariant.) The Kelly-only `μ_hat` is isolated to this ceiling
(Invariant A) and never leaks to `L_stop`, `L_vol`, `L_tail`, `L_portfolio`,
`L_margin`, or `L_liquidity`.

**Robust Kelly.** Because μ, σ, jump frequency, and slippage are uncertain, the
Kelly fraction is the uncertainty-shrunk (worst-case) one. In practice this
reduces to fractional Kelly with the fraction set by parameter uncertainty —
which is WHY the production fraction is ≤ ½, not a separate nested min-max
solver. Do NOT build an elaborate robust-optimization engine; implement the
shrinkage as the fraction φ.

```python
class KellyCeiling:
    def ceiling(self, mu_hat: float, sigma_hat: float, stop_distance: float,
                phi: float, cfg: "ResolvedConfig") -> tuple[float, dict]:
        """Half-Kelly RISK BUDGET converted to a leverage ceiling.
        L_kelly = phi * (mu_hat / sigma_hat**2), where mu_hat and sigma_hat are
        BOTH horizon-scaled to the same holding horizon T as the stop (Invariant
        S): sigma_hat = forecast_vol * sqrt(T). phi is the adaptive fraction
        (Section 5.2), phi <= 0.5 ALWAYS. All inputs are returns-based, so L_kelly
        is invariant to the instrument's notional/contract scale (Section 5.1).
        mu_hat is used ONLY here and ONLY as the Kelly input; it must not leak
        into any other ceiling (Invariant A). Returns (L_kelly, diagnostics)."""
```

### 5.2 Adaptive fraction — RULE-BASED on observables only

The Kelly fraction is time-varying but its trigger is a MEASURABLE risk
quantity, never discretion:

- φ = 0.5 (Half Kelly) under normal conditions.
- φ = 1/3 when parameter/market uncertainty is unusually high, where "high" is
  defined by OBSERVABLES: realized-vol percentile above a RED threshold, OR
  correlation-matrix instability above a threshold (e.g. large change in the
  leading eigenvalue / average pairwise correlation), OR recent jump-intensity
  elevated.
- Return toward 0.5 only after the observable subsides (with hysteresis to avoid
  flapping).

```python
class AdaptiveKellyFraction:
    def fraction(self, vol_percentile: float, corr_instability: float,
                 jump_intensity: float, cfg: "ResolvedConfig") -> tuple[float, dict]:
        """Return (phi, reason). phi in {0.5, 1/3} (or a smooth ramp between),
        switched by MEASURABLE risk observables only. A discretionary 'feels
        uncertain' input is forbidden — the trigger must be a logged number."""
```

> **Invariant K — Kelly fraction never exceeds 0.5, and its adaptation is
> driven only by logged observables.** No code path produces φ > 0.5. No
> discretionary or edge-derived input changes φ.

> **Invariant F — the Kelly fraction is applied exactly once.** φ enters only
> `KellyCeiling`. The final combine step applies the min() and the drawdown
> governor, but it does not multiply by φ again.

---

## 6. The drawdown governor (graduated, on the whole book)

The 43% portfolio limit is implemented as a GRADUATED risk multiplier on the
final leverage of every position, not a single kill-switch. Drawdown is measured
peak-to-current on portfolio equity, where **the peak is a TRAILING running max**
(`portfolio_peak = max over [t0, t] of equity`, `F_t`-measurable, no look-ahead to
a future peak — Invariant CAUSAL-L). `drawdown = 1 − equity/peak`.

```
portfolio drawdown    risk multiplier
   0% – 10%               1.00
  10% – 20%               0.75
  20% – 30%               0.50
  30% – 40%               0.25
  40% – 43%               LINEAR RAMP 0.25 -> 0.00   (see below)
     >= 43%               0.00  (flat the book; stop opening; research)
```

**The 40–43% band is a pinned LINEAR ramp** from 0.25 at 40% to 0.00 at 43%:
`mult(dd) = 0.25 · (0.43 − dd) / (0.43 − 0.40)` for `dd ∈ [0.40, 0.43)`, clamped to
0.00 at `dd ≥ 0.43`. This makes the band reproducible and the Section 15
"steps exactly at breakpoints" test well-defined (steps at 10/20/30/40%, then a
deterministic linear ramp to 0 across 40–43%). This de-risks smoothly as equity
falls — itself a drawdown-control property — and makes 43% a level the book is
ENGINEERED AROUND, reached only after progressive throttling, rather than a cliff
hit at full size.

**Breakpoints are RED, economically-derived (geometric de-risking), and OOS-paid.**
The ladder is a fixed geometric de-risking schedule, not fitted to in-sample P&L.
Where the breakpoints are treated as a finite BLUE grid (alternative ladders
considered), they are promoted ONLY through SYSTEM's deflated-Sharpe / purged-CPCV
OOS gate and counted in SYSTEM's joint multiplicity budget — the SAME deflation
accounting REGIME routes its mirror `1.00/0.75/0.50/0.25` ladder through (REGIME
Section 4 / its `g(s)` map). The governor breakpoints are therefore PAID FOR in the
shared multiplicity count, never silently free. (Invariant DDCAUSAL.)

```python
class DrawdownGovernor:
    def multiplier(self, portfolio_peak: float, portfolio_equity: float,
                   cfg: "ResolvedConfig") -> tuple[float, dict]:
        """Return (risk_multiplier in [0,1], reason). Applied to final leverage
        of all positions. >= 43% drawdown returns 0.0 (flat, halt-open)."""
```

> **Invariant G — the governor only reduces, never increases, exposure**, and it
> applies to the whole book uniformly. It cannot lift leverage above the min()
> of ceilings.

---

## 7. Portfolio ceiling and allocation across simultaneous signals

Under portfolio margin, this is the load-bearing constraint (P4). Two related
jobs, both RISK-based (no μ, no Σ⁻¹μ return optimization):

### 7.1 Portfolio ceiling via marginal contribution to risk

```python
class PortfolioCeiling:
    def ceiling(self, instrument: str, candidate_weights: dict,
                cov: "np.ndarray", g_t: float, cfg: "ResolvedConfig"
                ) -> tuple[float, dict]:
        """L_portfolio from the instrument's MARGINAL CONTRIBUTION TO RISK
        (MCTR) given the current/candidate book and covariance, against the
        REGIME-SCALED book-heat ceiling H_book_eff = g_t * H_book (P7). Correlated
        positions (e.g. NIFTY/BANKNIFTY/bank stocks) get a LOWER ceiling because
        their marginal risk contribution is high; diversifying positions
        (e.g. Gold/USDINR/NatGas/NIFTY) get a higher one. Enforces a total
        portfolio-heat ceiling H_book (RED), scaled to g_t*H_book so REGIME's
        macro cut enters HERE (one channel only, P7/Invariant R) — NOT on L_vol
        or the Kelly fraction. g_t in [0.25, 1.0] (REGIME Invariant CAP) so it can
        only LOWER H_book_eff and only de-risk. When the regime scaling is the
        active cut, set diagnostic reason_code='regime_heat_cut' (vs
        'mctr_concentration_cut' for a correlation-driven cut) so telemetry can
        attribute a gross cut to REGIME vs to MCTR (REGIME NODOUBLE audit test 2).
        cov MUST be the conditioned (Ledoit-Wolf-shrunk, condition-gated)
        covariance (Invariant CAUSAL-L / COND-L) — a near-singular Sigma makes
        MCTR shares and therefore L_portfolio nonsense. Uses covariance for RISK
        only — never return-optimized Sigma^-1 mu (mu is unmeasurable and
        inversion amplifies its error)."""
```

MCTR for instrument i: `MCTR_i = w_i · (Σw)_i / sqrt(wᵀΣw)`; the sum of MCTRs is
total portfolio volatility. The ceiling caps each instrument's MCTR share so no
correlated cluster dominates book risk. **Allocation mode = MODERATE partial
netting:** correlated instruments receive partially-netted budgets via MCTR (not
one shared budget, not naive independence).

**Covariance conditioning (mandatory, before any MCTR / inversion).** The `cov`
passed to `PortfolioCeiling` and `RiskBudgetAllocator` MUST be **Ledoit–Wolf
shrunk** (closed-form, data-derived shrinkage toward a constant-correlation /
scaled-identity target — no fit-on-P&L, no ML) and then pass a **post-shrink
condition-number / minimum-eigenvalue gate** before it is used. A near-singular or
ill-conditioned Σ makes `(Σw)_i`, the MCTR shares, and therefore `L_portfolio`
nonsense (the same MCTR sum-decomposition that turbulence and absorption rely on).
If the gate is breached the ceiling falls back to a robust diagonal / conservative
class-level heat budget and flags `cov_cond_breach`. This is the SAME panel
covariance family REGIME conditions for its turbulence / absorption matrices
(REGIME Invariants COND / CONDNUM / ARCOND) — LEVERAGE shares the panel and must
condition it identically. (Invariant COND-L.)

### 7.2 Risk-budget allocation across simultaneous signals

```python
class RiskBudgetAllocator:
    def allocate(self, signals: list, cov: "np.ndarray",
                 total_budget: float, cfg: "ResolvedConfig") -> dict:
        """Split the total risk budget across simultaneously-qualified signals by
        EQUAL RISK CONTRIBUTION (ERC) within correlation clusters. Correlated
        cluster shares a budget partially-netted by MCTR. This is a SECTION, not
        a separate module: for a small retail book (typically <= ~6 positions),
        ERC/MCTR is sufficient. Do NOT implement Hierarchical Risk Parity or
        hierarchical clustering machinery — it is built for large universes and
        overfits a small book."""
```

Correlation clustering (`allocate/clusters.py`) is a SIMPLE, capped grouping
(e.g. threshold on pairwise correlation) used only to decide which positions
share a budget — not a full HRP dendrogram.

> **Invariant P — covariance is used for RISK only.** No expected-return vector
> enters allocation or the portfolio ceiling. No `Σ⁻¹μ`. Allocation is
> risk-parity / MCTR based.

---

## 8. The other ceilings (contracts)

### 8.1 Stop ceiling (normal layer)

```python
class StopCeiling:
    def ceiling(self, side: int, entry_price: float, c: float, carry: float,
                forecast_vol: float, regime_bucket: str,
                cond_mae: "ConditionalMAE", cfg: "ResolvedConfig"
                ) -> tuple[float, dict]:
        """L_stop = (c - max(carry, 0)) / d, where d is the stop distance.
        The carry contribution to the SURVIVAL cap is CLAMPED from above: a
        positive carry (a financing/funding COST, P7 in COST.md) tightens L_stop
        exactly as (c - carry)/d; a negative carry (a CREDIT, e.g. a funding-earning
        short) is clamped to 0 and does NOT add survival headroom. A credit is an
        UNREALIZED forecast that can flip on a funding-regime change; letting it
        lift L_stop above the pure capital-loss cap c would make the adverse-move
        loss-at-stop exceed c*position_margin on the strength of a forecast — an
        optimistic/lookahead leak into a HARD survival constraint, contradicting
        P1. The earned-carry BENEFIT is booked only in the net-EV gate
        (BREAKOUT/COST), never as extra leverage room here. (Invariant CARRY.)
        d = max( k * forecast_vol_HORIZON , conditional_MAE_quantile(regime) ).
        forecast_vol is scaled to the HOLDING HORIZON (sigma * sqrt(T)) — the
        sqrt(T) is non-negotiable; a daily-sigma stop on a multiday hold is a
        tight stop masquerading as a wide one and causes premature stop-outs.
        Conditional MAE uses the regime bucket's winner-MAE distribution with an
        EVENT-COUNT FLOOR: if the bucket has too few events, fall back to global
        MAE. c in [0.12, 0.15]. Choose d to target a low P(noise hits stop)."""
```

`ConditionalMAE` buckets are a SMALL, robust set (e.g. vol-regime high/low,
optionally trend/chop). Each bucket needs enough winners to estimate its
80th-percentile MAE stably or it collapses to global (same event-count
discipline as the hazard ladder in TREND.md).

### 8.2 Volatility-target ceiling (normal layer)

```python
class VolTargetCeiling:
    def ceiling(self, forecast_vol: float, sigma_target: float
                ) -> tuple[float, dict]:
        """L_vol = sigma_target / forecast_vol. Rises in calm regimes, falls in
        volatile ones. sigma_target is RED (per-position target, e.g. 15-20%
        annualized)."""
```

### 8.3 Tail ceiling (CATASTROPHIC layer)

```python
class TailCeiling:
    def ceiling(self, loss_tail: "GPDFit", gap_hist: "GapModel",
                es_budget: float, es_level: float, cfg: "ResolvedConfig"
                ) -> tuple[float, dict]:
        """L_tail keeps Expected Shortfall within budget, using EVT
        (peaks-over-threshold / Generalized Pareto Distribution fit to the loss
        tail) rather than variance. ALSO enforces gap survival:
        L * plausible_adverse_gap (from gap_hist, e.g. worst overnight/weekend
        gap of the past year, or a multiple of typical gap) <= a ruin threshold
        (RED, e.g. 30-40% of committed capital). The hard per-trade stop does
        NOT survive gaps; only this ceiling does. es_level e.g. 97.5%."""
```

EVT is optional and event-count gated. If the loss-tail sample is too small,
unstable, or fails diagnostics, `TailCeiling` must fall back to conservative
empirical/stress estimates:
- worst adverse gap over a configured lookback;
- class-level stress gap;
- rolling empirical ES / high quantile with wide safety multiplier.

No GPD fit may output a precise ceiling without recording exceedance count,
threshold, shape/scale, fit diagnostics, and a fallback decision.

### 8.4 Margin ceiling (both layers)

```python
class MarginCeiling:
    def ceiling(self, instrument: str, margin_model: "MarginModel",
                free_margin: float, financing_rate: float, horizon: float,
                cfg: "ResolvedConfig") -> tuple[float, dict]:
        """L_margin from broker margin and a FREE-MARGIN BUFFER (keep >= ~20-30%
        free beyond immediate requirement so a margin call never forces
        liquidation before the stop does). Subtract financing/carry (MTF
        interest for stocks; basis carry for futures) over the horizon — this
        reduces the effective L_stop too (the carry term in 8.1).
        PORTFOLIO-MARGIN CAVEAT: model the margin requirement under a
        correlation-stress scenario, because netted portfolio margin SPIKES when
        correlations break (P4). Size against the stressed requirement, not the
        calm one.

        BROKER-REAL, NOT THEORETICAL: margin must come from the actual
        broker/exchange margin model or a broker API snapshot where available —
        NOT a fixed leverage constant. Futures margin is SCENARIO-BASED (CME SPAN
        and SPAN-style systems evaluate portfolio loss across a grid of price/vol
        scenarios), so the margin model must evaluate the position's worst-case
        scenario loss, and must account for the broker applying margin CHANGES
        (hikes around events/expiry). `MarginModel` ingests the broker/exchange
        margin file or API and exposes both the current requirement and a
        stressed (scenario / hiked) requirement; the ceiling sizes against the
        stressed one. A hardcoded notional/margin ratio is forbidden except as a
        clearly-labeled offline placeholder before the broker feed is wired."""
```

### 8.5 Liquidity ceiling (both layers)

```python
class LiquidityCeiling:
    def ceiling(self, liquidity: "LiquidityState", target_exit_minutes: float,
                cfg: "ResolvedConfig") -> tuple[float, dict]:
        """L_liquidity caps size so the position can be reduced or exited without
        unacceptable spread/impact.

        Inputs are measurable only: bid/ask spread, ADV or rolling notional
        volume, top-of-book/depth if available, realized impact, expected
        participation rate, and target exit time. The ceiling enforces:
        - max participation rate (e.g. <= 5-10% of expected volume over exit window);
        - max spread cost and slippage budget;
        - max impact cost under a simple square-root or conservative linear model;
        - stale-book / outage / halt flags that can force L_liquidity to 0.

        This is mandatory for US equities, crypto alts, thin commodity contracts,
        and stressed metals sessions. If reliable depth/ADV is unavailable, use
        a conservative class-level fallback, not infinity."""
```

### 8.6 Lot / contract quantization (full sizing-to-tradable-units layer)

This is a first-class part of position sizing, not an afterthought. The ceilings
and governor produce a *continuous* target (a leverage and an implied notional);
real instruments trade in *discrete* units (shares, lots, contracts), and the
gap between the two is where a carefully-derived risk constraint silently leaks.
This layer converts the continuous target into a tradable size under a strict
rule: **quantization may only reduce risk or reject the trade — never increase
it** (Invariant U).

#### 8.6.1 The quantization contract

```python
class Quantizer:
    def quantize(self, raw_leverage: float, raw_notional: float,
                 instrument: "InstrumentSpec", position_margin: float,
                 account_equity: float, book: "BookState",
                 risk_inputs: "RiskInputs", cfg: "ResolvedConfig"
                 ) -> "QuantizedSize":
        """Convert continuous (raw_leverage, raw_notional) into whole tradable
        units, then RE-VALIDATE every binding constraint against the rounded
        size. Returns a QuantizedSize carrying the final integer units, the
        residual risk after rounding, and explicit divergence/rejection flags.
        This is a gate, not an optimizer: it never searches for a size that
        maximizes anything; it rounds down and checks."""
```

#### 8.6.2 The rounding rule (direction is the whole point)

1. Compute the continuous target unit count:
   `raw_units = raw_notional / (instrument.contract_multiplier * price)`
   (for futures: notional per lot = lot_size * price * point_value; for shares:
   one unit = one share).
2. **Round DOWN to the nearest whole tradable unit:** `units = floor(raw_units)`
   adjusted to the instrument's lot step (e.g. NIFTY lot = 65, BANKNIFTY = 30,
   so tradable units come in multiples of the lot). Rounding down guarantees the
   realized position is ≤ the continuous target, so no constraint is breached by
   the rounding itself. **Never round up "for convenience" — rounding up
   converts a 13% cap into a 15%+ cap and is the single most common silent
   risk-limit leak.** (Invariant U.)
3. **Zero-rounding is an explicit, reported outcome, not a silent drop.** If
   `raw_units < 1` minimum tradable unit, the desired position is smaller than
   one contract. Return `units = 0` with flag `below_min_unit`. This happens
   routinely (a) on small accounts, and (b) **after the drawdown governor
   throttles size** — a position that was tradable at full size may round to
   zero at 25% risk. The system must SURFACE this ("governor throttled this trade
   below one lot"), because otherwise the book silently stops trading after a
   drawdown without anyone deciding that.
4. **Upper cap at the exchange/broker MAXIMUM (hard, ceiling-independent).** Cap
   the rounded count at `min(ceiling_implied_lots, exchange_max_lots)`, where
   `exchange_max_lots` comes from `InstrumentSpec` (AMBER): exchange position
   limit, single-order max-lot, or broker per-instrument cap. This is INDEPENDENT
   of the seven soft ceilings — a large account can pass every ceiling and still
   exceed an exchange position limit. When this cap binds, flag
   `exchange_max_capped` so the trader knows the size was limited by a venue rule,
   not a risk model. (Direction-preserving: this only ever LOWERS units, so
   Invariant U still holds.)

#### 8.6.3 Re-validation after rounding (the checks the continuous math passed may now differ)

Rounding *down* cannot breach a ceiling, but it changes the realized numbers, so
recompute and report:
- **Realized capital loss at stop on the rounded size**, as a % of BOTH position
  margin and total equity (Invariant M dual basis). Rounding down makes this ≤
  target — confirm and log the residual.
- **Realized SPAN/exchange margin for the integer lot count**, and the resulting
  **free-margin buffer** after this position is added to the existing book.
  Because margin is scenario-based and lumpy (one more lot can be a large margin
  jump), the post-rounding margin must be re-checked against the free-margin
  buffer (Section 8.4) — a position can pass the continuous margin ceiling and
  fail after rounding up to the next whole lot's margin. If it fails, drop one
  lot and recheck; if zero lots, reject with `margin_after_rounding`.
- **Realized liquidity / participation on the rounded size** (Section 8.5).
- **Portfolio heat / MCTR with the rounded size** included in the book — the
  discrete addition changes the covariance contribution slightly; re-check the
  total-heat ceiling.

#### 8.6.4 Multi-position interaction (quantization is book-aware) — ownership boundary with SYSTEM

Quantization is not per-trade-isolated. When several signals fire together and
share a netted risk budget (Section 7), rounding each independently can overshoot
the *aggregate* budget (many small round-downs can still sum past the cluster
budget if a few round up to the lot step).

**Ownership boundary (pinned, ONE quantizer, non-overlapping).** The book-level
aggregate fill is **OWNED by SYSTEM's single monotone greedy lot-fill** (SYSTEM
Section 3.3/3.4), not duplicated here. LEVERAGE's `Quantizer` is the **per-instrument
leaf** that SYSTEM's greedy fill invokes: it rounds ONE instrument DOWN to its lot
step and re-validates that instrument's own ceilings (8.6.2/8.6.3). The
book-aware aggregate trim/admission across simultaneous signals is the greedy fill's
job. The two procedures are NOT independently specified: there is exactly one
aggregate quantizer (SYSTEM's), and LEVERAGE supplies the per-leaf round-down +
revalidate it calls.

When LEVERAGE is run stand-alone (no SYSTEM orchestrator, e.g. in unit tests or a
single-account batch), its aggregate trim MUST use SYSTEM's priority order
**verbatim** so the two never diverge:

```
1. Highest binding-ceiling SLACK (room before the nearest failure mode) first.
2. Highest MCTR-efficiency = (risk-weight delivered)/(book heat consumed).
3. Tie-break: instrument id (deterministic).
```

Breakout score, expected return, expected favorable excursion, and trend
confidence are FORBIDDEN as priority inputs (Invariant A). **Incumbents are FIXED
(SYSTEM P4):** the aggregate trim re-quantizes only NEW-capital positions; an
open, trend-alive incumbent is never re-quantized or trimmed by this step (its
reserved heat is subtracted from the budget first). The aggregate re-check covers
the cluster heat and the total free-margin buffer; a breach trims the
lowest-priority NON-incumbent position by a lot (`cluster_trimmed`).

#### 8.6.5 Output object

```python
@dataclass(frozen=True)
class QuantizedSize:
    units: int                      # final whole tradable units (lots/shares/contracts)
    notional: float                 # realized notional at these units
    realized_leverage: float        # notional / position_margin (<= raw_leverage)
    realized_loss_pct_margin: float # at stop, % of position margin (<= c)
    realized_loss_pct_equity: float # at stop, % of total account equity
    realized_span_margin: float     # exchange/broker margin for these units
    free_margin_after: float        # free-margin buffer after adding this position
    divergence: float               # (raw_units - units) / raw_units  (rounding gap)
    flags: tuple[str, ...]          # e.g. ('below_min_unit',), ('margin_after_rounding',),
                                    #      ('rounded_down',), ('cluster_trimmed',),
                                    #      ('exchange_max_capped',), ('class_cap_capped',)
    rejected: bool                  # True if no tradable size satisfies constraints
```

The `divergence` field is mandatory telemetry: a chronically high rounding gap
(desired size persistently far from tradable size) means the account is too small
for the instrument's lot size at the chosen risk, and the trader should know that
rather than discover it through silently-skipped or silently-oversized trades.

#### 8.6.6 Absolute per-asset-class leverage cap (hard floor of last resort)

The per-class numbers in Section 13 (NIFTY ~4–5x, crypto 1–2x, …) are **starting
points for the soft ceilings**, but a per-class **absolute hard cap** `L_class_max`
(RED, validated class-level only — never per-instrument on P&L) is enforced as a
HARD guardrail so a mis-fit TEAL forecast vol or a stale/ill-conditioned covariance
can never emit an absurd leverage that only the soft ceilings restrain. It is
implemented as an extra clamp applied to the pre-governor leverage and to the
quantizer's `ceiling_implied_lots`:

```
L_i_pre_governor = min( L_stop, L_vol, L_kelly, L_tail,
                        L_portfolio, L_margin, L_liquidity, L_class_max )
```

`L_class_max` is a HARD guardrail, NOT a modelled failure-mode ceiling, so it does
NOT appear in the seven-ceiling table or take TEAL inputs; when it binds, flag
`class_cap_capped`. It is the last line of defence against estimator pathology, set
conservatively above the typical live leverage so it binds only on a model fault,
not in normal operation (if it binds routinely a ceiling is broken — investigate).
(Invariant CLASSCAP.)

---

## 9. Shared data objects (`types.py`)

```python
from dataclasses import dataclass
from typing import Literal, Mapping
import numpy as np

Side = Literal[-1, 1]

@dataclass(frozen=True)
class SignalInput:           # from upstream; carries NO edge score (Invariant A)
    instrument: str
    side: Side
    entry_price: float
    position_margin: float   # committed margin / position risk capital
    capital_cap: float       # c, 0.12-0.15 of position_margin, RED-sourced
    # NOTE: deliberately NO trend_confidence / expected_length / breakout_score.

@dataclass(frozen=True)
class LiquidityState:
    spread_bps: float
    expected_volume_exit_window: float
    adv_notional: float
    top_of_book_notional: float | None
    impact_bps: float
    stale_or_halted: bool

@dataclass(frozen=True)
class RiskInputs:            # all MEASURABLE
    forecast_vol: float      # horizon-scaled, from vol/forecast.py
    vol_percentile: float
    cov: "np.ndarray"
    corr_instability: float
    jump_intensity: float
    gap_hist: "GapModel"
    loss_tail: "GPDFit"
    free_margin: float
    financing_rate: float
    liquidity: LiquidityState
    portfolio_peak: float
    portfolio_equity: float
    account_equity: float
    g_t: float               # REGIME macro multiplier in [0.25,1.0]; scales H_book ONLY
                             # (P7). Market-state-derived; NEVER equity drawdown (Inv. R).

@dataclass(frozen=True)
class Ceilings:              # the seven, each with diagnostics
    L_stop: float
    L_vol: float
    L_kelly: float
    L_tail: float
    L_portfolio: float       # computed against g_t*H_book; carries the regime cut (P7)
    L_margin: float
    L_liquidity: float
    g_t: float               # echoed for telemetry / the NODOUBLE audit (Inv. R)
    diag: Mapping[str, float]  # incl. portfolio reason_code:
                             #   'regime_heat_cut' vs 'mctr_concentration_cut'

@dataclass(frozen=True)
class LeverageDecision:      # OUTPUT
    instrument: str
    leverage: float          # final, post-min, post-fraction, post-governor
    size: float              # notional / units implied
    rounded_size: float      # valid lot/share/contract size after LotSizer
    binding_ceiling: Literal["stop","vol","kelly","tail","portfolio","margin","liquidity","governor","unit"]
    # NOTE: a regime cut surfaces as binding_ceiling="portfolio" (it enters via
    # g_t*H_book, P7), NOT a separate "regime" literal. The portfolio reason_code
    # (below) MUST distinguish 'regime_heat_cut' from 'mctr_concentration_cut' so
    # the NODOUBLE audit can attribute a gross cut to REGIME vs to MCTR (Inv. R).
    kelly_fraction: float    # phi actually used
    governor_multiplier: float
    g_t: float               # regime multiplier actually applied to H_book (Inv. R)
    risk_budget: float
    planned_loss_on_margin: float
    planned_loss_on_equity: float
    reason_codes: tuple[str, ...]
    engine_version: str
```

---

## 10. Invariants (enforce structurally; ask before breaking)

**Invariant A — the edge/risk firewall.** No ceiling and no combine step takes
an edge quantity (expected return except as the Kelly-only μ input, trend
confidence, expected trend length, breakout score, expected favorable excursion)
as input to sizing. `SignalInput` carries no edge score by construction.
Required test: replacing any hypothetical edge field with arbitrary values
leaves every ceiling and the final leverage unchanged. The ONLY permitted μ is
inside `KellyCeiling`, and it must not leak to any other ceiling.

**Invariant B — leverage is min() of ceilings, never a product of scores.** The
final pre-governor leverage equals `min(L_stop, L_vol, L_kelly, L_tail,
L_portfolio, L_margin, L_liquidity, L_class_max)`, where `L_portfolio` already
carries the regime cut via `g_t·H_book` (P7) and `L_class_max` is the hard
guardrail clamp (8.6.6, Invariant CLASSCAP). No weighted multiplicative combination
exists; `g_t` enters the `min()` through `L_portfolio` only, not as a multiplier on
the result (P7/Invariant R).

**Invariant C — RED unreachable from per-instrument config.** (Section 4.)

**Invariant G — governor only de-risks.** (Section 6.)

**Invariant K — Kelly fraction ≤ 0.5, adaptation driven only by logged
observables.** (Section 5.)

**Invariant F — Kelly fraction applied exactly once.** (Section 5.)

**Invariant P — covariance used for risk only; no Σ⁻¹μ, no return-optimized
weights.** (Section 7.)

**Invariant H — diagnostics are log-only.** Per-ceiling diagnostics exist for
monitoring; none may alter the leverage except through its defined ceiling
value. Test: final leverage is invariant to diagnostic contents.

**Invariant V — validation never imported by live layers.** Nothing in
`ceilings/`, `combine/`, `allocate/`, `vol/` imports from `validate/`.

**Invariant S — horizon-scaled volatility in the stop.** The stop distance uses
horizon volatility (σ·√T), never a single-bar/daily σ on a multiday hold. Test
that doubling the holding horizon widens the stop by √2.

**Invariant M — dual capital cap: position-margin basis AND total-equity bound.**
The 12–15% cap `c` is applied to `SignalInput.position_margin`, not to total
portfolio equity — this is the per-position loss budget you committed to (put X
on a trade, stop at X·(1−c)). HOWEVER, that per-position cap must ALSO be bounded
as a fraction of total account equity, because across a multi-position book the
sum of position-margin losses can be large relative to the whole account. The
final per-position size is constrained by BOTH, with the binding one being
whichever implies LESS risk:
- **position-margin cap:** realized loss at stop ≤ c · position_margin (c = 0.12–0.15);
- **total-equity ladder** (RED, the absolute bound regardless of c):
  - planned (normal) loss at stop: ≤ ~1–3% of total equity;
  - stressed-gap loss (gap through the stop, from the tail/gap model): ≤ ~5–8% of total equity;
  - absolute catastrophic per-position loss: ≤ ~10–15% of total equity.

So a position whose 12–15%-of-margin stop would cost more than ~1–3% of TOTAL
equity is sized DOWN until it fits the equity ladder. On a small book with one
position the two bases can nearly coincide; on a multi-position book the
total-equity ladder typically binds first and prevents the aggregate of
per-position caps from endangering the account. Reports must show planned loss as
a percent of position margin AND as a percent of total equity. A position can
pass both cap tests and still be rejected by portfolio heat, liquidity, margin,
or the quantization gate.

**Invariant U — quantization cannot increase risk.** Rounding to shares,
contracts, lots, or exchange minimums may only reduce size or reject the trade
(round DOWN, never up). If the minimum tradable unit violates planned loss,
margin, liquidity, or aggregate-cluster limits, the final decision has
`units=0` and an explicit flag (`below_min_unit`, `margin_after_rounding`, or
`cluster_trimmed`). Zero-rounding — including when the drawdown governor throttles
a position below one lot — is a REPORTED outcome, never a silent drop. The
exchange-max cap (`exchange_max_capped`) and class cap (`class_cap_capped`) only
LOWER units, consistent with this invariant.

**Invariant R — the regime multiplier enters exactly ONE channel and only
de-risks.** `g_t ∈ [0.25, 1.0]` (REGIME Invariant CAP) enters LEVERAGE ONLY by
scaling the book-heat ceiling input `H_book → g_t·H_book` inside `PortfolioCeiling`
(P7). It is NOT a separate `min()` term, NOT a post-min scalar, and is NOT applied
to `L_vol`, the Kelly fraction, or any other ceiling (REGIME Invariants NODOUBLE2
/ NODOUBLEVC: REGIME owns cross-asset aggregate vol and book-wide systemic
correlation; LEVERAGE's `L_vol` uses per-instrument forecast vol and the Kelly
fraction uses per-position-local correlation — the two never re-cut the same axis).
`g_t` and the drawdown governor compose multiplicatively ONLY because their input
axes are provably disjoint (P8): `g_t` reads market-state ONLY and never equity
drawdown; the governor reads equity drawdown ONLY and never a regime measure
(REGIME Invariant NODOUBLE). Because `g_t ≤ 1`, it can only lower `H_book_eff` and
can never lift leverage above the seven-ceiling `min()`. Tests: (1) feeding
arbitrary equity-drawdown values to the `g_t` source leaves the regime cut
unchanged, and feeding arbitrary regime measures to the governor leaves its
multiplier unchanged (input sets provably disjoint); (2) at `g_t = 1.0` the book
is identical to no-regime overlay; (3) a regime-driven cut surfaces as
`binding_ceiling="portfolio"` with `reason_code='regime_heat_cut'`, distinguishable
in telemetry from `'mctr_concentration_cut'` (so REGIME's NODOUBLE audit test (2)
is verifiable from LEVERAGE's side); (4) `g_t` is applied in exactly one place —
removing the `g_t·H_book` scaling is the only code change that alters its effect.

**Invariant CARRY — a carry credit never relaxes the survival stop.** The carry
contribution to `L_stop` is clamped from above: `L_stop = (c − max(carry,0))/d`. A
positive carry (financing/funding COST, COST.md P7) tightens `L_stop`; a negative
carry (a CREDIT) is clamped to 0 and leaves `L_stop` at its no-carry value, never
above it. The earned-carry benefit is booked only in the net-EV gate
(BREAKOUT/COST), never as survival headroom. Test (mirrors COST's stop-ceiling
sign test): a positive carry tightens `L_stop`; a negative (credit) carry leaves
`L_stop` at its no-carry value, never above it.

**Invariant COND-L — every covariance LEVERAGE inverts/uses is shrunk and
conditioned.** The MCTR / portfolio covariance (and the allocator's) is
Ledoit–Wolf-shrunk (closed-form, no P&L fit) and passes a post-shrink
condition-number / minimum-eigenvalue gate before use; on breach it falls back to a
robust diagonal / class heat budget and flags `cov_cond_breach`. This is the SAME
panel covariance family and the SAME conditioning REGIME pins (COND/CONDNUM/ARCOND).
Test: a near-singular synthetic Σ does not produce a nonsense MCTR / `L_portfolio`;
the shrink + fallback path is exercised.

**Invariant CAUSAL-L — every estimator in LEVERAGE is `F_t`-measurable
(trailing-only).** Every vol, covariance, correlation, MAE, EVT/GPD, gap, and
vol/correlation-instability estimate, and the drawdown governor's equity peak, uses
ONLY trailing windows and causal percentiles (today's value within its OWN trailing
window). No centred window, no full-sample standardization, no look-ahead to a
future peak. Future data lives ONLY in SYSTEM's firewalled validation. This mirrors
REGIME Invariant CAUSAL. Test: shifting any future bar leaves every ceiling, the
governor multiplier, and the final leverage at `t` unchanged; an injected future
leak is caught.

**Invariant CLASSCAP — the absolute per-class leverage cap is a hard guardrail.**
`L_class_max` (RED, class-level, never per-instrument-on-P&L, no TEAL inputs) clamps
the pre-governor leverage as a last line of defence against estimator pathology
(mis-fit vol, stale Σ). It is set above typical live leverage so it binds only on a
model fault (`class_cap_capped`); routine binding means a ceiling is broken. Test:
an artificially absurd `L_vol`/`L_kelly` is clamped to `L_class_max`; under normal
inputs the cap is slack.

**Invariant DDCAUSAL — the governor ladder is causal and not P&L-fit.** Breakpoints
are economically-derived (geometric de-risking) or a finite BLUE grid promoted ONLY
through SYSTEM's deflated-Sharpe / purged-CPCV OOS gate and counted in SYSTEM's
joint multiplicity budget (never fit to in-sample P&L). The equity peak is a
trailing running max (Invariant CAUSAL-L). Test: the 40–43% linear ramp is
reproducible; the breakpoints appear in SYSTEM's multiplicity accounting.

---

## 11. Normal vs catastrophic risk separation (P5)

The validation and the reporting must label which layer bound each decision:

- **Normal layer** ceilings: L_stop, L_vol, L_kelly. These bind on the large
  majority of trades and govern routine sizing.
- **Catastrophic layer** ceilings: L_tail, L_portfolio (and the stressed branch
  of L_margin). These bind only in stress — high vol, fat tails, correlation
  concentration, margin-spike scenarios. The REGIME macro cut surfaces HERE,
  through `L_portfolio` (= `g_t·H_book`), and is attributed by the portfolio
  reason_code (`regime_heat_cut` vs `mctr_concentration_cut`, Invariant R) so a
  regime-driven gross cut is distinguishable from a correlation-driven one in the
  layer-attribution report.
- **Operational layer** ceilings/gates: L_margin, L_liquidity, and LotSizer.
  These bind when the broker, market depth, spread, outage/halt state, or
  minimum tradable unit makes the theoretical size unsafe or impossible.

Report the fraction of decisions bound by each layer. If the catastrophic layer
binds frequently in normal markets, a parameter is miscalibrated (too tight); if
it never binds even in stress backtests, it is too loose to protect you.

---

## 12. Validation, stress tests, and ablation

Validation is part of the module. Score on REALIZED vs PLANNED loss, net of
costs, purged walk-forward.

Required validations:
- **Planned-vs-realized loss:** does the 12–15% cap on position margin /
  committed position risk capital hold in practice? Also report the same loss as
  a percent of total account equity. Systematic exceedance ⇒ gaps are breaking
  stops ⇒ tail/gap/liquidity ceiling too loose or leverage too high.
- **Stress tests (catastrophic layer):** correlation-to-1 scenario (does
  portfolio heat hold when diversification vanishes?); margin-explosion scenario
  under portfolio margin (does L_margin's stressed branch prevent a forced
  liquidation?); historical worst-gap replay per asset class.
- **Drawdown-governor backtest:** confirm the graduated multiplier keeps
  realized max portfolio drawdown below 43% across adversarial paths, and that
  full-book flat triggers correctly.
- **Half-Kelly vs Full-Kelly baseline:** include Full Kelly as a CAUTIONARY
  baseline to demonstrate it repeatedly trips the 43% governor; Half Kelly must
  show materially lower max drawdown for comparable growth.
- **Ablation — which ceilings bind:** report how often each ceiling is the
  binding one. A ceiling that NEVER binds is either redundant or miscalibrated;
  a ceiling that ALWAYS binds may be the only real constraint (simplify).
- **Lot-size rejection report:** count trades rejected because the minimum
  tradable unit exceeds margin-cap, portfolio-heat, margin, or liquidity limits.
- **Regime-vs-MCTR attribution report:** of the decisions bound by `L_portfolio`,
  report the split between `regime_heat_cut` (REGIME `g_t`) and
  `mctr_concentration_cut` (correlation), so the cross-pillar NODOUBLE audit
  (REGIME Invariant NODOUBLE test 2) is verifiable from LEVERAGE's telemetry; and
  count `class_cap_capped` / `cov_cond_breach` events (routine occurrence ⇒ a
  ceiling or the covariance pipeline is miscalibrated).

Required statistics: block-bootstrap CIs (respect vol clustering and
correlation), report tail metrics (ES, worst gap-through loss), and whether one
instrument / one regime / one fold drives the result.

---

## 13. Asset-class caps and parameters (cross-asset)

Per-class presets (validated, class-level only — never per instrument). Leverage
ceilings fall as volatility and gap/tail risk rise; these are CEILINGS, not
targets, and the min() / governor / tail ceiling routinely produce lower live
leverage:

- **Indian index futures (NIFTY/BANKNIFTY):** moderate vol; NIFTY ~4-5x,
  BANKNIFTY ~3-4x typical L_stop at a 2-sigma multiday stop and 12-15% cap;
  exchange margin permits ~8.7-8.8x but do NOT use it all. Overnight global-cue
  gaps; event gaps (RBI, budget, expiry) → reduce or flatten before catalysts.
- **Indian single stocks (MTF):** idiosyncratic earnings gaps 5-15% overnight →
  much lower caps; avoid leveraged holds through earnings; MTF financing ~9-18%
  p.a. directly reduces ceilings and compounds over week-long holds.
- **Commodities / metals (MCX):** session gaps, geopolitical sensitivity
  (crude); high-vol, moderate caps.
- **Crypto:** 24/7, very high vol (realized vol multiples of equities), violent
  gaps even without a close → very low caps (1-2x); the wide-vol stop and the
  gap/tail ceiling enforce this automatically.
- **FX majors:** lower vol but weekend/news gaps; modest caps, watch carry.

Asset-class differences also flow through TEAL fits (each instrument's own vol,
MAE, EVT tail, gap distribution) — the presets are caps and starting points, not
the live numbers.

---

## 14. Synthetic data requirements (`synth/generate.py`)

Generate adversarial scenarios so the module is testable with no external data:
- Calm regime (vol-target ceiling should bind, leverage high but capped).
- Volatility spike (vol/stop ceilings tighten; fraction may drop to 1/3).
- Fat-tailed loss regime (tail ceiling binds via EVT).
- Adverse overnight/weekend gap (gap-survival sub-test of tail ceiling).
- Correlation-to-1 crisis (portfolio ceiling + governor + margin-spike all
  engage; book de-risks).
- Regime cut at full equity (g_t = 0.25 with NO equity drawdown: L_portfolio
  binds via g_t·H_book, binding_ceiling='portfolio'/'regime_heat_cut', governor
  stays 1.00 — proves the two axes are disjoint, Invariant R/P8); and the dual at
  g_t = 1.0 reproducing the no-regime book.
- Estimator-pathology guardrail (mis-fit forecast vol or near-singular Σ would
  imply an absurd L; the class cap clamps it, `class_cap_capped`; the conditioning
  gate trips `cov_cond_breach`).
- Progressive drawdown 0→43% (governor steps 1.00→0.75→0.50→0.25→0.00).
- Multiple simultaneous correlated signals (ERC/MCTR shares one partially-netted
  budget) vs diversifying signals (each gets more).
- Margin-explosion under portfolio margin (stressed L_margin prevents forced
  liquidation).
- Thin-liquidity / wide-spread session (L_liquidity binds or rejects).
- Minimum-lot-too-large account (LotSizer rejects instead of over-risking).

Behavioral assertions: leverage never exceeds the min() of ceilings; correlated
clusters get less than the sum of standalone leverages; governor de-risks
monotonically with drawdown and flats at 43%; Full-Kelly baseline trips the
governor far more often than Half Kelly; planned per-position loss holds except
where gaps are explicitly modeled (and there the tail ceiling should have
reduced size); rounded size never increases risk; `g_t` enters only via
`L_portfolio` (at `g_t=1.0` the book is identical to no-regime overlay) and only
de-risks; a carry credit never lifts `L_stop` above its no-carry value; the class
cap and conditioning gate clamp/flag estimator pathology rather than emitting an
absurd L.

---

## 15. Module acceptance gates (CI)

### Vol forecast
Pass when: YZ+EWMA+HAR-RV blend is causal, noise-robust (bipower-guarded at
1-min), horizon-scaled; no NaN/inf on adversarial data.

### Stop ceiling
Pass when: L_stop = (c − max(carry,0))/d exactly; a positive carry (cost) tightens
L_stop and a negative (credit) carry leaves L_stop at its no-carry value, never
above it (Invariant CARRY, mirrors COST's stop-ceiling sign test); d uses horizon σ
(√T test, Invariant S); conditional-MAE event-count floor falls back to global when
thin; c stays in [0.12,0.15].

### Vol-target ceiling
Pass when: L_vol = σ_target/forecast_vol; rises/falls correctly with vol.

### Kelly ceiling
Pass when: `L_kelly = φ·(μ_hat/σ_hat²)` exactly, with μ_hat, σ_hat horizon-scaled
consistently with the stop (Invariant S); φ ≤ 0.5 always (Invariant K); μ used only
here and does not leak (Invariant A); L_kelly is INVARIANT to the instrument's
notional/contract scale (doubling the contract multiplier leaves L_kelly unchanged
— returns-based, Section 5.1); Kelly outputs a budget→ceiling, and when it exceeds
other ceilings it does not bind.

### Adaptive fraction
Pass when: φ switches 0.5→1/3 only on logged observables with hysteresis; no
discretionary/edge input changes it.

### Tail ceiling
Pass when: EVT/GPD is used only when event-count and fit diagnostics pass;
otherwise empirical/stress fallback is used; ES held within budget;
gap-survival sub-constraint caps L so a modeled worst gap stays within the ruin
threshold.

### Portfolio ceiling + allocation
Pass when: MCTR computed correctly; correlated cluster gets a partially-netted
(MODERATE) budget, strictly less than naive-independent sum; total portfolio heat
respected against the regime-scaled `g_t·H_book` (P7); a regime cut surfaces as
`reason_code='regime_heat_cut'` distinct from `'mctr_concentration_cut'` (Invariant
R); the covariance is Ledoit–Wolf-shrunk and passes the condition-number gate before
MCTR, with a near-singular Σ exercising the fallback (Invariant COND-L); NO Σ⁻¹μ
anywhere (Invariant P); no HRP/clustering-dendrogram machinery.

### Margin ceiling
Pass when: free-margin buffer enforced; financing subtracted over horizon;
stressed (correlation-break) margin branch prevents forced liquidation in the
margin-explosion synthetic.

### Liquidity ceiling
Pass when: max participation, spread, impact, stale-book, and outage/halt rules
cap leverage correctly; thin books and high spread lower L_liquidity; missing
liquidity data falls back to conservative class-level ceilings, not infinity.

### Quantization (sizing to tradable units)
Pass when:
- rounding is always DOWN to the lot/share/contract step; a test confirms it
  NEVER increases realized leverage or realized loss-at-stop above the continuous
  target (Invariant U);
- desired size below one tradable unit returns `units=0` with `below_min_unit`,
  and the governor-throttles-below-one-lot case is detected and flagged (not a
  silent drop);
- realized capital loss is recomputed on the rounded size on BOTH bases
  (position margin and total equity, Invariant M) and is ≤ the binding cap;
- margin is re-checked AFTER rounding (one more lot can breach the free-margin
  buffer); failure drops a lot or rejects with `margin_after_rounding`;
- multi-signal aggregate test: independently-rounded simultaneous positions are
  re-checked against the cluster budget and total free margin, and the
  lowest-priority position is trimmed by a lot if the aggregate breaches
  (`cluster_trimmed`); the aggregate trim uses SYSTEM's priority order VERBATIM
  (ceiling-slack → MCTR-efficiency → instrument id, Section 8.6.4) and NEVER an
  edge input (Invariant A); incumbents are FIXED and never re-quantized/trimmed
  (SYSTEM P4); when orchestrated, the aggregate fill is OWNED by SYSTEM's greedy
  lot-fill and LEVERAGE supplies only the per-instrument round-down+revalidate leaf;
- the per-instrument round-down is also capped at the exchange/broker maximum
  (`exchange_max_capped`) and the absolute class cap (`class_cap_capped`), each
  only ever LOWERING units (Invariant U/CLASSCAP);
- the `divergence` (rounding-gap) telemetry is emitted, and a chronically high
  divergence on an instrument is surfaced (account too small for the lot size at
  the chosen risk);
- futures lots (NIFTY 65, BANKNIFTY 30, etc.), equity share lots, and crypto
  minimum order sizes are each tested.

### Drawdown governor
Pass when: multiplier steps exactly at the 10/20/30/40% breakpoints then follows
the pinned LINEAR ramp 0.25→0.00 across 40–43% (reproducible, Section 6); only
de-risks (Invariant G); flats the book at ≥43%; the equity peak is a trailing
running max with no look-ahead (Invariant CAUSAL-L/DDCAUSAL); the ladder appears in
SYSTEM's multiplicity accounting (Invariant DDCAUSAL).

### Causality
Pass when: shifting any future bar leaves every ceiling, the governor multiplier,
and the final leverage at t unchanged; every vol/cov/corr/MAE/EVT/gap estimate and
the equity peak is trailing-only (Invariant CAUSAL-L); an injected future leak is
caught.

### Engine / combine
Pass when: pre-governor leverage = min() of the seven ceilings AND the class cap
(`min(L_stop,L_vol,L_kelly,L_tail,L_portfolio,L_margin,L_liquidity,L_class_max)`,
Invariant B/CLASSCAP), with `g_t` entering ONLY through `L_portfolio`'s `g_t·H_book`
input and applied in exactly one place (Invariant R: at `g_t=1.0` the book equals
no-regime overlay; feeding equity drawdown to the `g_t` source / regime measures to
the governor changes neither); Kelly fraction applied only inside L_kelly; final
leverage = pre-governor leverage × drawdown_governor(equity_drawdown) (Invariant
B/F/R); `g_t` and `governor_multiplier` echoed in LeverageDecision telemetry;
LotSizer then rounds down or rejects; binding_ceiling reported correctly (regime
cut → `portfolio`/`regime_heat_cut`); diagnostics don't affect output (Invariant H);
deterministic given inputs.

---

## 16. Build order

1. `types.py` and `synth/generate.py`.
2. `config/` with TEAL/AMBER/RED/BLUE; test Invariant C.
3. `vol/forecast.py` (YZ+EWMA+HAR-RV, bipower-guarded, horizon-scaled).
4. `ceilings/stop.py` (Invariant S √T test; conditional-MAE event-count floor).
5. `ceilings/vol_target.py`.
6. `ceilings/margin.py` (free-margin buffer; stressed portfolio-margin branch).
7. `ceilings/liquidity.py` (participation/spread/impact/stale-book ceiling).
8. `ceilings/tail.py` (EVT/GPD + fallback empirical gap survival) — catastrophic layer.
9. `allocate/clusters.py` and `allocate/risk_budget.py` (ERC/MCTR; NO HRP;
    Ledoit–Wolf-shrunk + condition-gated cov, Invariant COND-L).
10. `ceilings/portfolio.py` (MCTR against `g_t·H_book`, Invariant P/R/COND-L;
    `regime_heat_cut` vs `mctr_concentration_cut` reason code) — catastrophic layer.
11. `combine/kelly_fraction.py` (adaptive, rule-based; Invariant K).
12. `ceilings/kelly.py` (Half-Kelly budget→ceiling; Invariant A/F μ-isolation).
13. `combine/governor.py` (graduated; Invariant G).
14. `combine/lot_sizer.py` (round down or reject; exchange-max + class-cap clamp;
    SYSTEM-priority aggregate trim, incumbents fixed; Invariant U/CLASSCAP).
15. `combine/engine.py` (min() of seven ceilings + class cap, `g_t·H_book` into
    L_portfolio as the sole regime channel, governor on the disjoint equity axis,
    lot gate; Invariant B/F/U/R/CLASSCAP/CAUSAL-L) and `engine_api.py`.
16. `validate/labels.py`, `validate/stress.py` (correlation-to-1, margin
    explosion, worst-gap replay).
17. `validate/baselines.py` (incl. Full-Kelly cautionary baseline),
    `validate/ablation.py` (which ceilings bind).
18. `validate/walkforward.py`, `validate/stats.py`, reports.
19. `priors/market_dna.py` last.

Each step complete only when its acceptance gate (Section 15) is green. A module
that fails its gate does not advance.

Stop after step 19. The remaining questions — which ceilings actually bind on
your data, whether the conditional-MAE buckets earn their split, whether the
catastrophic layer is calibrated right — are answered by RUNNING the validation
and stress harness on real data, not by adding ceilings.

---

## 17. References and rationale

- Leverage-stop identity: capital_loss ≈ L·d + carry; invert for L_max.
- Kelly (1956); Thorp; MacLean, Thorp & Ziemba (Kelly applications and
  full-Kelly drawdown severity). Full Kelly's drawdown distribution makes a 43%
  governor a near-certain repeated trip → Half Kelly in production.
- Risk-constrained / drawdown-constrained Kelly: Busseti, Ryu & Boyd
  (risk-constrained Kelly gambling); MacLean, Ziemba & Blazenko (capital growth
  with drawdown constraints) — the formal basis for "fractional Kelly from a
  drawdown limit."
- Volatility targeting / scaling: Moskowitz, Ooi & Pedersen (Time Series
  Momentum); Barroso & Santa-Clara (Momentum has its moments); Moreira & Muir
  (Volatility-managed portfolios). Most of the risk control is the vol-scaling
  step.
- Yang-Zhang (2000) OHLC volatility; Corsi (2009) HAR-RV; bipower variation
  (Barndorff-Nielsen & Shephard) for noise-robust 1-min vol.
- t = SR·√T (drift power wall): μ unmeasurable at short horizons → edge cannot
  drive leverage (the firewall, P2); reinforces BREAKOUT.md / TREND.md.
- EVT / Expected Shortfall: peaks-over-threshold + Generalized Pareto for the
  loss tail; size to ES, not variance, because the tail (gaps, crises) is the
  real leveraged risk.
- Marginal contribution to risk / Equal Risk Contribution: standard risk-parity
  decomposition `MCTR_i = w_i (Σw)_i / sqrt(wᵀΣw)`; risk-based allocation, NOT
  return-optimized Σ⁻¹μ (unstable under estimation error).
- MAE/MFE (Sweeney, Campaign Trading): conditional adverse-excursion for stop
  calibration.
- Fung & Hsieh (trend-as-lookback-straddle): the convex payoff the per-trade
  stop + trailing exit produce; left-tail truncation is what this module's stop
  ceiling enforces.
- Ledoit & Wolf (2004) constant-correlation / scaled-identity shrinkage: the
  closed-form, data-derived covariance conditioning used before MCTR (Invariant
  COND-L), shared with REGIME's turbulence/absorption matrices.
- Cross-pillar composition contracts: REGIME.md (g_t macro multiplier — its
  Section 3.4 output contract, Section 10 boundary, Invariants NODOUBLE /
  NODOUBLE2 / NODOUBLEVC / CAP / COND / CONDNUM / CAUSAL; consumed here via
  `g_t·H_book`, P7/Invariant R); COST.md (signed carry term, its P7 / CarryTerm §8.3
  contract — reconciled: COST keeps carry SIGNED for the EV/hold path and states the survival
  cap consumes the `max(carry,0)` clamp LEVERAGE applies here, Invariant CARRY);
  SYSTEM.md (selection↔leverage iterate-to-consistency and the single greedy
  lot-fill that owns book-level quantization, its Section 3.3/3.4 priority order
  and P4 incumbent-fixity — LEVERAGE supplies the per-instrument leaf, Section 8.6.4).

---

## 18. Non-goals (belong to sibling modules)

Entry detection (BREAKOUT.md), trend-life / profitable-exit management
(TREND.md), order routing and execution microstructure (a later EXECUTION
layer), and any edge/alpha estimation. This module receives a qualified signal
and a risk budget and emits a safe leverage. It never predicts whether the trade
will work — only how much capital can be exposed to it without violating
survival constraints.
