# SYSTEM.md — build spec for the composition + integrated-validation layer

This file is the authoritative build spec for the SYSTEM module. Build to the
contracts and invariants below. Where this file and your own instinct disagree,
this file wins. Ask before deviating from any **Invariant**.

The SYSTEM module is the orchestrator. It owns no edge logic, no sizing math, no
exit math of its own. It **composes** four already-built policy modules into one
auditable decision per bar, and it **validates the composed chain as a single
object** — which is a different and harder problem than validating each module
alone. Its two jobs:

1. **Composition algebra.** Given the per-bar outputs of selection, breakout,
   leverage, and trend, produce the book's target portfolio (which instruments,
   which side, how many lots each) by a *deterministic precedence* and a
   *bounded iterate-to-consistency* between selection and leverage. No discretion,
   no learned combiner, no multiplicative score blob.

2. **Integrated validation.** Validate the WHOLE composed pipeline
   (selection → breakout → leverage → trend → execution-stub) under purged
   walk-forward and sampled combinatorial purged CV, with system-level
   triple-barrier labels, and deflate the system Sharpe by the **union of
   degrees-of-freedom across every module's BLUE grid**. This is the centerpiece:
   it is the math that stops the *four-green-modules-but-no-edge* trap.

The composed chain sits on top of **NautilusTrader**, which already owns order
plumbing, position/account/portfolio state, restart reconciliation, data
transport, backtest↔live parity, pre-trade risk checks, and funding ingestion.
SYSTEM builds POLICY on that engine. It does **not** rebuild any of it.

Scope: full cross-asset (Binance USDT-M perpetuals 24/7, Dukascopy-proxied
commodities/metals, US/Indian equities and index futures), 1-minute bars,
~$1000 retail book, target compute = a 2-core / 11 GB ARM VM. Retail latency:
seconds, bar-level — never microseconds.

---

## 0. Governing principles (read before anything else)

**P1 — SYSTEM composes; it never reimplements.** The four modules are imported
as black boxes behind their public contracts (`select()`, `propose()`,
`size()` / engine `decide()`, `update()`). SYSTEM may read their outputs and
their reason codes; it may not reach inside them, duplicate their estimators, or
override their internal thresholds. Any sizing math, any exit math, any entry
math living in `system/` that is not pure orchestration is the bug.

**P2 — Deterministic precedence, not a score.** The composed decision is a fixed
pipeline with a fixed conflict-resolution order (Section 2). Given identical
module outputs and identical book state, SYSTEM emits identical actions. There is
**no weighted blend** of module outputs, **no** `score = w1·selection +
w2·breakout + …`, and **no** tunable combiner coefficients fitted on P&L. The
composition is an algebra of `min()`, set operations, and a contraction map — all
auditable, all monotone. (See Invariant CB.)

**P3 — The risk budget is the shared scarce resource, and it is allocated by a
firewall, not an auction on edge.** Selection proposes *which* instruments;
leverage's seven ceilings + the book heat ceiling + the dual equity cap decide
*how much* each gets; when proposals exceed the budget the contest is resolved by
**measurable-risk priority** (binding-ceiling slack, MCTR efficiency), **never**
by breakout score or expected return. The selection↔leverage interaction is the
only place two modules genuinely couple, and it is resolved as a **bounded
fixed-point iteration** (Section 3), not a one-shot.

**P4 — Selection-wants-to-drop must lose to trend-wants-to-ride, by default.**
An open position is governed by TREND until TREND says exit. Selection operates
on the *candidate set for new capital*, not on open positions. Rotating out of a
live, trend-alive winner because a fresh instrument scored higher is a turnover
tax that destroys the convex right tail the whole stack is built to capture.
SYSTEM enforces a **hold-incumbent bias**: an open, trend-alive position is only
closed by (a) TREND exit, (b) a leverage ceiling collapse (governor, margin,
liquidity, tail), or (c) selection demotion *with* a hysteresis + replacement-edge
margin that must clear costs. (See Invariant HI.)

**P5 — Causality is a system property, not just a module property.** Each module
is individually causal (`F_t`-measurable live values). SYSTEM adds a stronger
joint requirement: **no module's test fold may leak into another module's fit.**
The folds are shared and synchronized across the whole chain; the embargo is the
**union** of every module's label horizon; TEAL fits in every module use only
that fold's permitted history. Validation labels (system triple-barrier) live in
a firewalled layer unreachable from any live module. (See Invariants SC, FW.)

**P6 — The system multiplicity is the JOINT search, counted as EFFECTIVE trials —
never the max, never a fabricated product.** Four modules each searched a small
BLUE grid; the composed live config is one point in the joint space, selected
(implicitly or explicitly) against the alternatives, so the deflation must pay the
*joint* multiple-testing tax — not `1`, not any single module's count. But the
benchmark `SR*₀` requires the trial count and `Var[SR_trials]` to be estimated
over the **same** scored trials (the BLdP consistency law); you may not feed a
never-sampled Cartesian *product* `N` into a variance measured from the tens of
configs you actually composed. Therefore `N_sys` is the **effective number of
independent trials among the configs actually scored** (`1 ≤ N_sys ≤ K`, Section
7.1), with `Var[SR_trials]` over those same `K` configs; the product is recorded
only as `n_sys_nominal` telemetry. Reporting an undeflated, per-module-only, or
product-inflated-but-inconsistent Sharpe is the single most common way a
four-green-modules stack ships with no edge. (See Invariant DSR.)

**P7 — Cheap by caching per-module evidence.** The expensive work is each
module's per-bar evidence (breakout proposals, vol forecasts, ceilings, trend
beliefs) on each fold. Cache it keyed by `(module_version, fold_id, config_hash,
bar_range)`. A SYSTEM config that changes only a *composition/combiner* parameter
(precedence order, hysteresis, rebalance cadence) MUST re-score from cache for
free — never re-run the modules. Sampled CPCV, regime-bucketed caches, heavy
validation runs scheduled overnight. No per-bar optimization; numba only after
correctness tests pass. (See Invariant CA.)

**P8 — Decomposable and attributable.** Every emitted action carries a reason
chain naming which module drove it and which ceiling/precedence rule bound it.
Every validation result is decomposed into **per-pillar P&L attribution** via
sequential ablation (Shapley-style over the four pillars; Section 8). A green
system whose P&L cannot be attributed to any pillar is not green — it is luck
that has not yet been measured.

**Forbidden in v1:** any learned/black-box combiner (P1); a multiplicative score
that blends module outputs (P2); allocating the risk budget by edge / breakout
score / expected return (P3); rotating out of trend-alive winners on selection
alone (P4); per-module embargo when a joint embargo is required (P5);
undeflated or per-module-only deflated Sharpe at the system level (P6);
re-running modules when only a combiner parameter changed (P7); per-symbol tuning
of any SYSTEM parameter on realized P&L (governance, Section 5); ML of any kind.

---

## 1. What SYSTEM is and is not (the module map)

```text
SELECTION  (Carver Ch.25)  -> proposes the integer portfolio: which N instruments,
                              long/short eligibility, target risk-weights. RISK-based
                              instrument diversification + a continuation/quality rank
                              that NEVER drives sizing.
BREAKOUT.md                -> per qualified instrument: trade / no-trade, side,
                              entry, invalidation, stop-distance, evidence decomp.
LEVERAGE.md                -> per qualified signal: safe leverage = min() of 7
                              ceilings x drawdown governor; book heat + dual equity cap.
TREND.md                   -> per OPEN position: continue / reduce / exit + convex trail.
EXECUTION.md (later)       -> order tactics; SYSTEM uses only a thin execution STUB
                              (cost + fill model) for validation parity.
```

SYSTEM is the layer that turns those four streams into **one target book per
bar** and proves the composed thing has an edge net of costs and net of the
multiple-testing tax. It is the only module that sees all four at once.

SYSTEM is NOT: a fifth alpha source, a sizing engine, an exit engine, an
execution algorithm, or the Nautilus engine. It holds no model parameters that
could be tuned for return — only composition policy (precedence, hysteresis,
cadence, budget split rule) governed as RED/BLUE.

---

## 2. Composition algebra — deterministic precedence

On each **decision bar** (Section 3.5 sets the cadence; not every 1-minute bar
triggers a recompose), SYSTEM evaluates the pipeline below. Each step is a pure
function of `F_t` and the current `BookState`. The order is fixed and is itself
the conflict-resolution rule.

```text
INPUTS at bar t:
  book        = NautilusTrader portfolio/position/account snapshot (F_t)
  sel         = SelectionModule.select(universe, evidence_cache, cfg)   -> ProposedPortfolio
  open_pos    = { instrument -> TrendModule.update(position) }          -> TrendAction per open
  candidates  = { instrument in sel.eligible AND not open ->
                    BreakoutModule.propose(instrument) }                -> Signal or None

PIPELINE (deterministic precedence, top wins):

  (A) GOVERNOR / KILL.  Read LeverageEngine drawdown governor on the whole book.
      If multiplier == 0 (>= 43% drawdown): emit FLAT_ALL, halt new opens, return.
      Else carry the multiplier m in [0,1] forward; it scales every size.

  (B) OPEN POSITIONS FIRST.  For each open position, apply TrendAction:
        exit   -> schedule close (overrides any selection opinion). reason=trend_exit
        reduce -> schedule partial reduce to TREND's target fraction. reason=trend_reduce
        continue-> keep; it is an INCUMBENT and reserves its current risk budget.
      A LeverageEngine re-evaluation runs on every continue/reduce position:
      if any ceiling (governor/margin/liquidity/tail) now implies LESS size than
      held, the ceiling wins and the position is reduced. reason=ceiling_collapse.
      Leverage may only REDUCE an incumbent here, never add (adds go through (D)).

  (C) RECONCILE SELECTION vs INCUMBENTS.  Selection's ProposedPortfolio is a wish
      list for *capital deployment*, evaluated against the post-(B) book:
        - incumbent in sel.keep         -> hold (no action).
        - incumbent NOT in sel.keep     -> candidate for DEMOTION, but subject to
                                           the hold-incumbent firewall (Section 4):
                                           only demote if trend not-alive OR
                                           replacement clears hysteresis+cost margin.
        - sel proposes instrument not open -> becomes a new-capital candidate (D).

  (D) NEW-SIGNAL ADMISSION.  For each (selection-eligible AND breakout-qualified
      AND not-open) candidate, request size from LeverageEngine. New signals and
      surviving demotion-replacements compete for the RESIDUAL risk budget
      (book heat ceiling minus incumbent reserved heat) via the firewall
      priority of Section 3.4. NEVER by breakout score.

  (E) ITERATE-TO-CONSISTENCY.  Steps (C)-(D) propose an integer portfolio; (D)'s
      sizes consume MCTR budget that changes the covariance contribution of every
      other position, which changes each ceiling, which can change which
      candidates fit. Run the fixed-point map of Section 3 until the integer
      portfolio is stable (bounded iterations) or fall back to the last feasible.

  (F) QUANTIZE + DUAL-CAP RECHECK.  LeverageEngine.Quantizer rounds every target
      DOWN to tradable lots and re-checks the dual equity cap and book heat on the
      ROUNDED book (Invariant U from LEVERAGE.md). Aggregate cluster trims apply.

  (G) EMIT.  Produce a SystemDecision: the target book delta (opens, closes,
      reduces, holds), each with its full reason chain and binding pillar/ceiling.
      NautilusTrader executes; SYSTEM does not place raw orders itself.
```

The precedence is **TREND exit > ceiling collapse > governor scaling > hold
incumbent (trend-alive) > selection demotion-with-margin > new-signal
admission**. This ordering is RED policy and is the entire conflict-resolution
logic. There is no place where a higher breakout score promotes a trade ahead of
a risk constraint.

> **The firewall restated.** SYSTEM's only multi-input combination is `min()`
> over ceilings (delegated to LeverageEngine), set membership (selection),
> and precedence (this section). No product of scores, anywhere. A composed
> decision that depends on the *magnitude* of a breakout score (rather than its
> boolean qualification + side) is the bug. (Invariant CB.)

---

## 3. The selection ↔ leverage interaction as iterate-to-consistency

This is the only genuine two-way coupling in the stack and the mathematical heart
of the composition.

### 3.1 Why a one-shot is wrong

Selection proposes an integer portfolio `S = {i_1, …, i_k}` with target
risk-weights `w^sel`. LeverageEngine sizes each position; but `L_portfolio`
(MCTR), the book-heat ceiling, and the dual equity cap all depend on the **whole
book's covariance contribution**. Adding position `j` raises every other
position's marginal contribution to risk, lowering their ceilings, which can make
some proposed positions infeasible, which changes the covariance again. A single
pass either over-commits (sizes each as if alone) or arbitrarily under-commits
(orders matter). The consistent book is a **fixed point** of:

```
w*  =  Φ(w*)          where Φ takes a candidate weight vector to the
                      LeverageEngine-feasible weight vector under the SHARED
                      book-heat budget and the dual equity cap.
```

### 3.2 The contraction map Φ (explicit)

Let the active set be incumbents (post-(B)) plus admitted candidates. Let
`Σ` be the LeverageEngine covariance estimate (RISK only; **no** `Σ⁻¹μ`,
Invariant P inherited from LEVERAGE.md). Iteration index `r`:

```
1. Given current integer book b^(r) and notional weights w^(r):
2. For each position i, LeverageEngine.size(i | book=b^(r), Σ) returns
   L_i^(r) = min(seven ceilings) · m   (governor multiplier m from step A).
   The portfolio ceiling uses MCTR_i(w^(r)) = w_i (Σ w)_i / sqrt(wᵀΣw).
3. Convert L_i^(r) -> target notional -> w_tilde^(r+1).
4. Enforce the SHARED constraints jointly:
     sum_i MCTR_i  <=  H_book           (book heat ceiling, RED)
     sum_i planned_loss_i(equity basis) <= dual-equity ladder (LEVERAGE Inv. M)
   If violated, scale the NON-incumbent (new-capital) weights down proportionally
   to MCTR-efficiency until satisfied (incumbents are reserved first, P4).
5. Quantize candidates to lots (round DOWN); recompute w^(r+1) from integer lots.
6. Stop when the INTEGER book is unchanged: b^(r+1) == b^(r), OR r == r_max.
```

### 3.3 Convergence, determinism, and the fallback

We do **not** claim the integer iterate converges to a fixed point. Lot rounding
DOWN is a discontinuous map, and a naive loop can oscillate between two
equal-heat integer books: A drops a lot → B now fits → B's MCTR lifts A's
portfolio ceiling → A fits again → B no longer fits → … This is a real failure
mode, so we engineer a **provably terminating** loop and define "converged"
honestly, rather than assert a contraction that the integer projection breaks.

**The continuous relaxation does converge.** Drop the integer projection. On the
continuous pass, `w_tilde^(r+1) = Π_F( (1-η) w^(r) + η · LeverageFeasible(w^(r)) )`
where `Π_F` is Euclidean projection onto the feasible set
`F = { w : Σ_i MCTR_i(w) ≤ H_book, equity-ladder(w) ≤ ladder, 0 ≤ w_i ≤ ceiling_i(w) }`.
`F` with the ceilings evaluated at the *current* iterate is a convex box ∩
budget simplex; damping with `η ∈ (0,1]` (RED) makes the averaged-projection map
firmly non-expansive, so the continuous `w_tilde^(r)` converges (Krasnoselskii–
Mann). This gives a unique continuous target `w_∞`. We do not iterate the
continuous map to numerical convergence either — `r_max` damped steps land close
enough; the continuous limit is only used to define the rounding target.

**The integer book is then resolved by a single monotone descent, not a loop.**
Given the continuous target `w_tilde` from the last damped step, quantize ONCE by
the deterministic, strictly-decreasing procedure:

```
1. Rank candidates by budget priority (Section 3.4: ceiling-slack, then
   MCTR-efficiency, then instrument id). Incumbents are fixed (P4).
2. Greedily admit lots in priority order, each time re-checking H_book and the
   equity ladder against the partially-built integer book. Admit a lot only if
   BOTH constraints still hold after adding it.
3. Stop when no further lot can be admitted without breaching a shared
   constraint. Each step strictly increases admitted heat and never removes an
   already-admitted lot, so the procedure terminates in ≤ (Σ_i ceiling-lots)
   steps and is a deterministic function of (w_tilde, priority order).
```

Because step 2 is monotone-additive (heat only goes up, admitted lots are never
revoked), there is **no oscillation**: the A-drops/B-fits cycle cannot occur,
because a lot once admitted is never dropped to make room for another. This is a
fixed greedy fill of a knapsack-like budget, not an iterated map, so termination
is by construction, not by a contraction argument.

**Definition of `converged`.** The damped continuous loop runs ≤ `r_max` (RED,
e.g. 4–6) steps; `converged = True` iff the integer book produced by the greedy
fill is **identical on two consecutive decision bars given identical inputs**
(book stable). The oscillation case is therefore impossible *within* a bar (no
loop), and across bars a flapping target is suppressed by the deadband `δ`
(Section 3.5). If the continuous relaxation has not settled within `r_max`
(`‖w^(r) − w^(r-1)‖ > tol`), set `converged = False`, emit the greedy-filled
integer book from the last step (which is feasible by construction), reason
`iterate_fallback`, and log the residual `‖w^(r) − w^(r-1)‖`. We never emit an
infeasible book and never loop unbounded. (Invariant IT.)

- **Determinism:** ties (equal MCTR-efficiency, equal priority) broken by a fixed
  total order (instrument id), seeded nowhere. Same inputs ⇒ same integer book
  (the greedy fill is order-deterministic). Same inputs across two bars ⇒
  `converged = True`.

### 3.4 New-signal vs open-position competition for the shared budget

When residual budget `H_book − Σ_incumbent MCTR` cannot admit all qualified new
candidates at minimum tradable size, admit by **measurable-risk priority**, in
order:

```
1. Highest binding-ceiling SLACK (room before the nearest failure mode) —
   admit positions that are furthest from any ceiling first.
2. Highest MCTR-efficiency = (risk-weight delivered) / (book heat consumed) —
   i.e. the most diversifying instrument per unit of shared budget.
3. Tie-break: instrument id (deterministic).
```

Breakout score, expected return, expected favorable excursion, and trend
confidence are **forbidden** as priority inputs (firewall, inherited from
LEVERAGE.md Invariant A). A position that cannot be admitted this bar is logged
`budget_deferred`, not silently dropped, and is reconsidered next decision bar.

### 3.5 Rebalance cadence

Recomposing every 1-minute bar churns the book and pays spread/impact on noise.
The decision bar is governed:

- **Event-driven recompose (always):** a TREND exit/reduce, a ceiling collapse,
  a governor step change, or a new breakout qualification triggers an immediate
  recompose of the *affected* slice (incumbents are not re-quantized unless their
  ceiling moved).
- **Scheduled selection recompose (cadence τ_sel, RED):** the full
  selection↔leverage fixed point runs on a coarse cadence (e.g. daily, or on
  session boundaries), NOT per minute. Carver-style instrument selection is slow
  by design; turnover is a cost.
- **No-trade band / hysteresis:** a recompose only changes a position if the new
  target differs from the held size by more than a deadband `δ` (RED, in lots or
  in risk-weight), so floating-point and one-lot jitter do not generate orders.
  (Invariant CAD.)

---

## 4. Conflict resolution: selection-drop vs trend-ride (the hold-incumbent firewall)

The most consequential conflict. Resolved by an explicit, monotone rule — never
by comparing a selection score to a trend score on the same axis (they measure
different things; selection ranks *risk diversification + quality of the
candidate set*, trend measures *is this open move still alive*).

```
For an open position i NOT in sel.keep (selection wants it out):

  if TrendModule says EXIT                      -> close. reason=trend_exit
  elif any LeverageEngine ceiling collapsed     -> reduce/close. reason=ceiling_collapse
  elif TrendModule says trend NOT alive (reduce/death) -> allow demotion. reason=sel_demote_dead
  elif TrendModule says CONTINUE (trend alive):
        # incumbent is winning and alive; selection alone may not evict it.
        replacement = best selection-eligible, breakout-qualified, not-open candidate
        if replacement exists AND
           replacement_rank - incumbent_rank  >  Δ_rank       (hysteresis, RED) AND
           expected turnover cost (round-trip spread+impact+funding) is cleared by
           the diversification/heat improvement of the swap (a RISK improvement,
           NOT an edge improvement)                            -> rotate. reason=sel_rotate
        else                                                   -> HOLD. reason=hold_incumbent
```

Key points:
- The rotation test is a **risk/cost** test (does the swap improve book
  diversification per unit heat by more than the round-trip cost?), not an edge
  test. We never sell a live winner to buy a higher-"alpha" name.
- `Δ_rank` hysteresis and the cost margin make rotation rare and deliberate.
- A reduce from TREND and a demote from selection on the same position **compose**:
  TREND's target fraction is the cap; selection demotion cannot *increase* it.

> **Invariant HI — incumbents are protected.** An open, trend-alive position is
> never closed by selection alone. Closing requires TREND exit, a ceiling
> collapse, trend-not-alive, or a rotation that clears both the rank hysteresis
> and the round-trip cost on a RISK basis. Test: a selection demotion of a
> trend-alive, ceiling-slack position with no qualifying replacement leaves the
> position open and emits `hold_incumbent`.

---

## 5. Parameter governance

Every SYSTEM parameter is exactly one bucket. SYSTEM holds **no model
parameters** — only composition policy. The dangerous bucket here is BLUE,
because SYSTEM's BLUE grid is what feeds the deflation count (P6).

- **TEAL — fitted per instrument from its own past data, causal.** SYSTEM itself
  fits almost nothing (composition is policy). The TEAL surface SYSTEM *consumes*
  comes from the modules (their vol, MAE, EVT, covariance, hazards). The one
  SYSTEM-level TEAL-ish object is the **per-fold cost realization** used in
  attribution, and it is fold-local and causal.
- **AMBER — supplied metadata.** Universe membership, per-instrument calendar /
  session, tick/lot/multiplier, funding schedule (ingested by Nautilus), venue,
  asset class. Drives cross-asset behavior via metadata, never forked logic.
- **RED — shared composition/modelling constants.** Precedence order; rebalance
  cadence `τ_sel`; event-recompose triggers; deadband `δ`; rotation hysteresis
  `Δ_rank`; iteration cap `r_max` and damping `η`; book heat ceiling `H_book`
  (shared with LEVERAGE); the dual-equity ladder; the **contractual horizon cap
  `H_sys`** (Invariant HC); the embargo construction; the triple-barrier
  multiples; the CPCV `(n_groups, k_test, max_paths)`; the deflation/PSR
  confidence threshold (the DSR gate confidence, from which `Z_gate =
  Φ⁻¹(threshold)`); the **DSR singularity guards** (`K_min` minimum distinct
  trials, `var_floor`, the `N_sys=1` convention; **the MinTRL track-length floor
  `n_obs_floor` and `Z_gate`**, Section 7.3a); the **annualization cadence
  `ppy = periods_per_H_sys`** (Section 7.2a — the single factor for every Sharpe
  and `var_floor`); the effective-trials clustering threshold `rho_cl` (Section
  7.1, **set TIGHT toward more clusters / larger `N_sys`**, default ≈0.7); the
  drop-top-N count; the bootstrap block length and CI level; the **re-promotion
  cadence `τ_promote`, forward-gate window `τ_forward`, and the paired-difference
  forward-gate block length + CI level** (Section 7.5); the **overnight compute cap**
  (`K_max`, distinct-module-config cap, `T_night`; Section 9.2); the
  **execution-stub conservatism multiples** (Section 6.1b). RED is shared and
  **unreachable from per-instrument config** (Invariant C, inherited).
- **BLUE — finite validated grids, class-level only, NEVER per-symbol-on-PnL.**
  SYSTEM's own BLUE grid is the small set of *composition* choices that are
  legitimately uncertain: precedence variants (e.g. governor-before vs
  after-trend), cadence `τ_sel ∈ {daily, session}`, deadband `δ` candidates,
  rotation hysteresis `Δ_rank` candidates, budget-priority rule variants. Each is
  a finite, enumerated candidate set searched offline under the validation gates,
  promoted by the deflated-Sharpe rule — never fitted continuously, never per
  symbol, never on raw P&L without the multiple-test tax.

> **Invariant SG — SYSTEM trial accounting is exhaustive and persistent.** Every
> composed config SYSTEM ever scored against any validation fold — across ALL
> overnight runs to date (Section 7.5) — appears in the persistent trial ledger,
> and `K`/`Var[SR_trials]`/`N_sys` are derived from it (Section 7.1), not
> asserted. A config promoted "because it looked best" without its siblings being
> counted, or a fresh nightly search that does not accumulate, is the four-green
> (and overfit-by-iteration) trap. Test: the persistent ledger length equals the
> number of distinct `config_hash` values scored against any validation fold over
> all runs; `N_sys` and `Var[SR_trials]` reference the same ledger config set.

> **Invariant C — RED unreachable from per-instrument config** (inherited from
> the sibling specs). RED enters only via `resolve_config(defaults=...)`. No
> cadence, hysteresis, embargo, or deflation constant appears as a per-instrument
> field.

---

## 6. Integrated validation — purged walk-forward + sampled CPCV on the composed chain

Validation is **part of the model**, not a report bolted on. SYSTEM validates the
*composed* pipeline as one object, under the strictest causal discipline in the
whole repo.

### 6.1 System-level triple-barrier labels (firewalled)

The label for a system decision at bar `t` (an open, a hold-resize, a rotate) is
the realized net-of-cost path outcome of the *book delta* it produced, resolved
by a triple barrier:

```
upper barrier : +b_up · σ_horizon   (profit target, RED multiple)
lower barrier : -b_dn · σ_horizon   (stop / invalidation, RED multiple)
vertical      : H_sys (the CONTRACTUAL system max-hold cap, in bars, RED)
label         : which barrier touched first, signed, net of the execution-stub cost
```

`σ_horizon` is the position's horizon volatility from the LEVERAGE/TREND vol
forecast. Labels use **future bars** and therefore live ONLY in
`validate/labels.py`, structurally unreachable from any live module (Invariant
FW). Because labels span future horizons and overlap in time across positions,
purge + embargo are mandatory.

**The endogenous-horizon trap (and the fix).** TREND exits at an *adaptive*
stopping time `tau_live = min(tau_death, tau_cap)` (TREND.md §1.4). The realized
holding period is path-dependent: a multi-week trend-ride can run far longer than
any *typical* horizon. If the vertical barrier — and therefore the embargo — were
keyed to a typical or mean horizon, a long realized hold would leak across the
fold boundary into the next fold's fit. A fixed `embargo = max(label_horizon_m)`
cannot purge a label whose *realized* horizon ran longer than the assumed max.

The fix is a **contractual hard cap `H_sys`**, a RED constant in bars, with the
invariant that TREND's operational cap satisfies `tau_cap ≤ H_sys` for every
position the SYSTEM composes (TREND.md §1.6 role 2: the operational maximum-hold
cap). SYSTEM passes `H_sys` into the TREND adapter as the operational cap; a
position still alive at `H_sys` is force-marked `time_cap` (not death). Therefore
**no realized holding period can exceed `H_sys`**, so `H_sys` is a true upper
bound on every label's horizon and is the correct quantity to feed the embargo.
This converts a path-dependent, unbounded horizon into a bounded one by contract.
(Invariant HC.)

> **Invariant HC — contractual horizon cap bounds every label.** The system
> triple-barrier vertical equals `H_sys` (RED, bars), and the TREND adapter is
> configured so `tau_cap = H_sys`; no realized holding period exceeds `H_sys`.
> Test: across an adversarial persistent-trend synthetic, every labeled position
> closes at or before `H_sys` bars (force-`time_cap` at the boundary), and the
> embargo width equals `H_sys + max warmup` — never the mean or typical horizon.

### 6.1b The execution stub must be a CONSERVATIVE upper bound on cost

The net-of-cost labels above — and therefore the entire DSR gate — are only as
honest as `exec_stub` (`nautilus/exec_stub.py`). A stub that under-estimates
fills/slippage/funding biases every Sharpe **up** and lets a no-edge system pass.
The contract is therefore not "model cost accurately" but "**over**-estimate
cost", so the DSR is biased **low** and a config that passes the gate passes
*despite* a pessimistic cost assumption:

```
The exec stub MUST, for every fill used in a label:
  - spread     : charge the FULL quoted spread (cross the book), never mid or
                 half-spread, on entry AND exit.
  - slippage   : assume the WORST plausible fill within the bar (adverse extreme:
                 buys fill at/above the bar high region, sells at/below the low
                 region), parameterized by a conservative impact model.
  - impact     : square-root (or conservative linear) market impact at the
                 modeled participation rate, scaled UP by a safety multiple (RED).
  - funding    : full funding/financing/carry over the realized hold (perp
                 funding ingested by Nautilus; MTF interest; futures basis), with
                 NO netting of favorable funding.
  - fees       : full taker fees on every fill (never assume maker rebates).
All cost components are charged at their pessimistic end; favorable realizations
are NOT credited. The stub's net P&L is a LOWER bound on achievable net P&L.
```

> **Invariant XC — execution stub is a conservative cost upper bound.** For any
> fill, `exec_stub` cost ≥ a realistic cost model's cost (full spread + adverse
> slippage + scaled impact + full funding + taker fees), so labeled net-of-cost
> P&L is biased low and DSR is biased low. Tests: (1) on synthetic fills the stub
> cost is ≥ a mid-price/half-spread reference for every fill; (2) **calibration
> against realized fills once live** — when live fills accumulate, the stub's
> charged cost must remain ≥ the realized cost on a rolling basis, else the stub
> is re-tightened (made more pessimistic) and the gate re-run. A stub that
> under-charges versus realized fills is a failing test, not a tuning knob.

### 6.2 Purge and embargo are the UNION across modules

```
embargo_bars_sys = max(  H_sys,                                  # system label cap (Inv. HC)
                         max over modules of label_horizon_m )   # each module's own label horizon
                   + max over modules of warmup_m                # longest causal warmup
purge            : drop training labels whose [start,end] overlaps any test label
                   after applying the system embargo on BOTH sides of each test fold
```

The embargo is the **union** (the max horizon) because a breakout label, a trend
survival label, and a system triple-barrier label have different horizons; the
joint pipeline must respect the longest. The system triple-barrier horizon is
bounded by the contractual cap `H_sys` (Invariant HC), so `H_sys` — not a mean or
typical realized horizon — enters the union; this is what closes the endogenous,
path-dependent-horizon lookahead. A per-module embargo, or an embargo keyed to a
typical horizon, would leak a short-horizon module's (or a long trend-ride's)
test data into another module's fit. (Invariants SC, HC.)

### 6.3 Purged walk-forward (the primary protocol)

Anchored / rolling expanding windows. In each fold:
- every module's TEAL fit, every threshold, every selection rank, every
  covariance estimate is trained ONLY on that fold's permitted history;
- the composed pipeline runs forward on the test slice;
- system triple-barrier labels score the realized book deltas net of cost;
- folds are time-ordered, embargoed per 6.2.

### 6.4 Sampled combinatorial purged CV (the distribution protocol)

Full CPCV is expensive on a 2-core VM. Sample paths deterministically:

```python
class SystemCPCV:
    def evaluate(self, bars, universe, cfg, cost, n_groups: int, k_test: int,
                 embargo_bars: int, max_paths: int | None,
                 evidence_cache: "EvidenceCache") -> "SystemReport":
        """Partition the timeline into n_groups; for each of C(n_groups, k_test)
        train/test arrangements, purge+embargo, run the COMPOSED pipeline on the
        test groups, score system triple-barrier P&L net of cost. If max_paths is
        set, sample arrangements deterministically with a seed stored in report
        metadata. Returns a DISTRIBUTION of out-of-sample system Sharpe / outcome
        metrics across paths — the input to the bootstrap CI and to the deflated
        Sharpe of Section 7. Module per-bar evidence is pulled from evidence_cache
        keyed by (module_version, fold_id, config_hash, bar_range) so that a cfg
        changing only a COMBINER parameter re-scores without re-running modules
        (P7)."""
```

CPCV gives a *distribution* of OOS performance (one number per path), which is
exactly what the deflated/probabilistic Sharpe needs and what a single
walk-forward number cannot give.

---

## 7. The centerpiece — deflated / probabilistic Sharpe over the UNION of degrees of freedom

This section is why SYSTEM exists. Four modules can each pass their own gate and
the composition can still have **no edge**, because the live config was selected
(explicitly via BLUE search, or implicitly by the human picking the green
variant) from a large joint search space. The Sharpe must be deflated by that
*joint* multiplicity.

### 7.1 Counting the joint trials `N_sys` — and the consistency law

The DSR luck-benchmark `SR*₀` (Section 7.3) requires `N` trials drawn from the
**same distribution** whose variance `Var[SR_trials]` we estimate. This is the
defining assumption of Bailey–López de Prado: the `N` configs and the variance
must be computed over **one and the same set of scored trials**. Feeding a
fabricated, never-sampled product `N` into a variance estimated from a *different*
(smaller) set of configs is mathematically incoherent — it is the single most
dangerous error in this whole spec, so we forbid it as an invariant.

**The naive product is compute-infeasible and inconsistent.** The Cartesian
product `N_selection · N_breakout · N_leverage · N_trend · N_system` (e.g.
8·8·8·8·... ≈ thousands) counts joint grid points that the SYSTEM **never
actually composes and scores** — the overnight job only ever runs the COMPOSED
system on the configs in the ledger (tens, Section 9). You cannot estimate
`Var[SR_trials]` from tens of scored configs and then pretend `N` is the
thousands-strong product: the variance and the count would describe different
populations. (a) Scoring the full product is compute-infeasible on the target VM
(Section 9). (b) Therefore we do **not** use the raw product.

**What we use instead: the BLdP effective-trials count, derived ONLY from scored
configs.** Let the trial ledger (Invariant SG) hold the `K` distinct composed
configs actually scored against any validation fold, each with its OOS path-return
series `r^(k)` and its OOS Sharpe `SR_k`. Then:

```
trial_sharpes  = { SR_1, …, SR_K }          (one per LEDGER config, K = ledger length)
Var[SR_trials] = sample variance of trial_sharpes        (over the SAME K configs)
N_sys          = effective number of independent trials among those K configs
```

`N_sys` is derived from the configs we measured, under the explicit **BLdP
clustering assumption**: configs whose OOS return series are highly correlated do
not count as independent trials. We compute it as:

```
C            = K×K correlation matrix of the ledger configs' OOS path returns r^(k)
N_eff        = K / ( 1 + (K - 1) · rho_bar )            (Kish-style effective count)
   where rho_bar = mean off-diagonal correlation in C (CLIPPED AT 0 below: a
   negative mean off-diagonal is treated as 0, so anti-correlated configs give
   rho_bar = 0 => N_eff = K, i.e. fully independent trials — see the pin below).
   Equivalently, cluster the K configs at a RED correlation threshold rho_cl and
   set N_clusters = number of clusters (two configs share a cluster iff their OOS
   return correlation >= rho_cl); report both and use the LARGER (more
   conservative).
N_sys        = max( N_eff , N_clusters , 1 )             (never < 1; see 7.3 guard)
```

**Monotonicity-in-threshold (the penalty cannot be quietly shrunk).** The
cluster route depends on the RED correlation threshold `rho_cl`. TIGHTENING the
threshold (raising `rho_cl` toward 1, so it takes a *higher* correlation for two
configs to be treated as the "same trial") splits clusters apart and yields MORE
clusters ⇒ a LARGER `N_clusters` ⇒ a LARGER `N_sys` ⇒ a LARGER multiple-testing
penalty. The hazard is the reverse: a LOOSE (low) `rho_cl` silently merges
distinct configs into a few clusters, shrinking `N_sys` and the deflation, which
is the wrong direction for a conservative gate. We pin the direction:

```
INVARIANT DSR-MONO (N_sys cluster route is monotone non-decreasing as rho_cl tightens):
  for rho_cl_loose < rho_cl_tight,  N_clusters(rho_cl_loose) <= N_clusters(rho_cl_tight)
  (single-linkage clustering on |corr| is nested: a finer cut never merges what a
   coarser cut split, so the cluster count is non-decreasing in rho_cl). Therefore
  loosening the threshold can only SHRINK the penalty, never grow it — forbidden
  as the quiet-shrink path.

CONSERVATIVE DEFAULT (RED): rho_cl is set TIGHT (biased toward MORE clusters /
  LARGER N_sys), e.g. rho_cl_default = 0.7. The default errs toward over-counting
  trials (paying more multiple-testing tax), never under-counting.
```

**Anti-correlated configs ⇒ `N_sys = K` (the de-correlated count is a floor).**
Because `rho_bar` is clipped at 0, a ledger of mutually anti-correlated (or
zero-correlated) configs gives `rho_bar = 0` ⇒ `N_eff = K`, so the Kish route
already reports `K` independent trials. The `max(N_eff, N_clusters, 1)` then
**cannot fall below** the de-correlated count `K`: anti-correlation never lets the
penalty drop below the all-independent count. (This was implied but is now pinned
and tested.)

This makes `N_sys` and `Var[SR_trials]` **consistent by construction**: both are
functions of the same `K` scored ledger configs. The product of module
cardinalities is recorded as `n_sys_nominal` for transparency (it is the size of
the space we *could* have searched) but is **never** fed to `SR*₀` — that would
violate the consistency law.

> **The BLdP assumption, stated.** We assume the `K` scored configs are a
> representative sample of the search the human + BLUE grids actually conducted,
> and that the *independent* degrees of freedom exercised equal the number of
> de-correlated config clusters among them. If the human privately eyeballed
> configs that were never scored against a fold, those trials are invisible and
> `N_sys` understates the true multiplicity — which is why Invariant SG forces
> every evaluated config into the ledger, and why §15.5 forbids promotion of any
> config absent from the ledger. The re-promotion policy (Section 7.5) extends
> this across calendar time: every night's re-search adds its configs to the same
> ledger, so `N_sys` accumulates and the consistency law continues to hold.

> Effective trials are therefore **bounded**: `1 ≤ N_sys ≤ K`. They can never
> exceed the number of configs actually scored (you cannot pay a multiple-test
> tax for trials you never ran) and never drop below 1 (Section 7.3 guard). The
> conservative direction is *more* trials; clustering only reduces `N_sys` toward
> the count of genuinely distinct trials, and the reduction is data-driven from
> the CPCV path-return correlation matrix and recorded.

### 7.2 Which series feeds which input (the equation block)

PSR and DSR take inputs that differ by **orders of magnitude** depending on which
series you feed; getting this wrong swings PSR from ~0.6 to ~0.99. We pin every
input here, once, as a contract. `DeflatedSharpe.deflate(path_returns,
trial_sharpes, n_obs, …)` is fed exactly:

```
LET the PROMOTED config be config c* (the one being gated).

path_returns   := the PROMOTED config's per-observation net-of-cost system
                  return series, where one observation = one NON-OVERLAPPING
                  system return bucket. We use NON-overlapping returns so n_obs
                  is an honest independent-sample count, NOT the inflated
                  1-minute bar count.
SR_pp          := the PER-OBSERVATION (per-bucket) Sharpe of path_returns,
                  i.e. SR_pp = SR_annual / sqrt(ppy). *** THIS, not the
                  annualized Sharpe, is what feeds PSR/DSR. *** The
                  Bailey–López de Prado z-statistic (§7.2b) is derived for a
                  PER-OBSERVATION Sharpe: its non-normality denominator terms
                  (γ3·SR, ((γ4−1)/4)·SR²) are only correct at annualization
                  factor f = 1. Feeding an ANNUALIZED Sharpe inflates the
                  numerator by ~sqrt(ppy) with no matching denominator scaling
                  and produces a spurious near-1 DSR. Annualization is a
                  REPORTING-ONLY convention applied AFTER the gate (§7.2a),
                  never inside PSR/DSR.
n_obs          := number of NON-OVERLAPPING system return observations in
                  path_returns.  *** NOT bar-count, NOT path-count,
                  NOT trade-count. ***  Concretely: aggregate the net-of-cost
                  book return to a non-overlapping cadence at least as long as
                  H_sys (the label horizon) so observations do not overlap; n_obs
                  is the count of those buckets. (At 1-min bars over 1 yr,
                  ~525k bars collapse to n_obs ≈ years·(periods per H_sys) —
                  e.g. H_sys = 7 trading days ⇒ ~36 non-overlapping obs/yr.)
γ3, γ4         := skew and kurtosis of path_returns (the SAME promoted series).
                  These are PER-OBSERVATION moments already (shape statistics
                  of the per-bucket return distribution) and need no rescaling.
trial_sharpes  := { SR_1, …, SR_K }, the per-LEDGER-CONFIG OOS Sharpes
                  (Section 7.1), expressed in the SAME PER-OBSERVATION units as
                  SR_pp (each annualized trial Sharpe divided by sqrt(ppy)).
                  *** CROSS-CONFIG, never cross-CPCV-PATH. ***
Var[SR_trials] := sample variance of the PER-OBSERVATION trial_sharpes (over
                  the K ledger configs). Per-observation so it is commensurable
                  with SR_pp and var_floor inside SR*₀.
n_trials_eff   := N_sys from Section 7.1 (effective independent configs).
```

Two errors are explicitly forbidden and tested (Invariant SR-SRC):

- **`Var[SR_trials]` must be the cross-CONFIG ledger variance** (variance of the
  Sharpes of the different promoted-candidate configs), **NOT** the cross-CPCV-
  PATH variance (the spread of one config's Sharpe across CPCV paths). The
  cross-path spread measures sampling noise of one strategy; the cross-config
  spread measures the multiple-testing search dispersion that `SR*₀` deflates
  against. Using the former collapses `SR*₀` and defeats the gate.
- **`n_obs` must be the non-overlapping system-observation count**, not the
  bar-count (orders of magnitude too large ⇒ PSR → ~1 spuriously) and not the
  path-count or trade-count.

The CPCV path distribution (Section 6.4) is used for the bootstrap CIs and the
regime/fold attribution (Section 8), **not** as `trial_sharpes`.

#### 7.2a Per-observation units inside the gate; annualization is reporting-only

**The genuinely invariant practice is to run the ENTIRE PSR/DSR computation in
PER-OBSERVATION units, so NO annualization factor enters the z-test at all.**
The Bailey–López de Prado z-statistic
`z = (SR − SR*₀)·sqrt(n_obs−1)/sqrt(1 − γ3·SR + ((γ4−1)/4)·SR²)` is derived for a
per-observation Sharpe: only at annualization factor `f = 1` do the denominator's
non-normality terms (`γ3·SR`, `((γ4−1)/4)·SR²`) match the numerator's `SR`.
Feeding an **annualized** Sharpe scales the numerator's `SR` and the
`(SR − SR*₀)` gap by `sqrt(ppy)` while the denominator's non-normality terms are
NOT correspondingly rescaled, so `z` is inflated by roughly the annualization
factor with no offset — pushing DSR spuriously toward 1. (This is not a wash:
for a single fixed strategy `z` rises monotonically with the factor — e.g. by a
factor of ~`f` as `f` goes 1 → 6 → 12 — it does NOT cancel.)

We therefore pin one cadence and convert every Sharpe to per-observation units
**before** the gate:

```
ppy := periods_per_H_sys = (periods per year) / (H_sys in those periods)
       — the number of NON-OVERLAPPING H_sys-length buckets per year (~36/yr at
       H_sys = 7 trading days). This is the SINGLE cadence; n_obs is counted in
       these same buckets.

ENFORCED (Invariant DSR-ANN):
  (i)  PSR/DSR is fed the PER-OBSERVATION Sharpe SR_pp = SR_annual / sqrt(ppy),
       and EVERY trial_sharpe is likewise put in per-observation units
       (SR_trial_annual / sqrt(ppy)) before Var[SR_trials] is taken;
  (ii) var_floor (Section 7.3a, Lo-2002) is `(1 + SR_pp²/2)/n_obs` — the sampling
       variance of a PER-OBSERVATION Sharpe — so var_floor and Var[SR_trials]
       are both per-observation and directly commensurable inside SR*₀;
  (iii) annualization (multiply by sqrt(ppy)) is applied ONLY to the
        reported/headline Sharpe AFTER the gate decision, NEVER inside PSR/DSR;
  (iv) an internal consistency check rejects the report (`annualization_mismatch`)
        if any Sharpe entering the z-test is not in per-observation units, or if
        the `ppy` used to de-annualize differs from the bucketing that produced
        `n_obs`.
```

Because SR_pp, every per-observation trial Sharpe, and var_floor share one
per-observation time unit, `Var[SR_trials]`, `var_floor`, `SR*₀`, and the z-test
are all commensurable, and `DSR` carries no annualization artifact. Annualization
remains an arbitrary reporting convention — but the correct way to make `DSR`
immune to it is to keep it OUT of the z-test entirely (run per-observation), not
to assert that `sqrt(ppy)` "cancels" inside the test (it does not).

### 7.2b Probabilistic Sharpe Ratio (PSR)

For the promoted config's **per-observation** Sharpe `SR_pp` over `n_obs`
observations with skew `γ3` and kurtosis `γ4` of `path_returns`, the probability
that the true Sharpe exceeds a benchmark `SR*` (also in per-observation units):

```
PSR(SR*) = Φ(  (SR_pp − SR*) · sqrt(n_obs − 1)
               / sqrt( 1 − γ3·SR_pp + ((γ4 − 1)/4)·SR_pp² )  )
```

where `Φ` is the standard normal CDF and `γ4` is **non-excess (raw) kurtosis**
(normal => 3, so the normal denominator is `1 + 0.5·SR_pp²`); an implementer
using *excess* kurtosis would silently use the wrong denominator. **Both `SR_pp`
and `SR*` MUST be per-observation Sharpes (Section 7.2a): the denominator's
non-normality terms are only correct at annualization factor f = 1.** The
standard error is the Mertens / Bailey–López de Prado adjusted-Sharpe SE:
`SE = sqrt(denom / (n_obs − 1))` with the `(n_obs − 1)` term **outside** the sqrt
of the variance term `denom = 1 − γ3·SR_pp + ((γ4−1)/4)·SR_pp²`, so equivalently
`z = (SR_pp − SR*)·sqrt(n_obs − 1) / sqrt(denom)` — the `(n_obs − 1)` is never
placed inside `denom`. PSR corrects naïve Sharpe for non-normality
(fat tails, skew) and finite sample — both severe at 1-minute, leveraged,
cross-asset returns. The denominator is floored at a small positive `eps` to
avoid division blow-up when `1 − γ3·SR_pp + ((γ4−1)/4)·SR_pp² ≤ 0` (recorded as
a `psr_denom_floored` flag).

### 7.3 Deflated Sharpe Ratio (DSR) — the gate

The benchmark `SR*` is **not zero**. Under `N_sys` independent trials, the
expected maximum Sharpe from luck alone is positive and grows with `N_sys`:

```
SR*₀ = sqrt(Var[SR_trials]) · [ (1−ɣ)·Φ⁻¹(1 − 1/N_sys)
                                +  ɣ·Φ⁻¹(1 − 1/(N_sys·e)) ]
```

where `Var[SR_trials]` is the cross-CONFIG ledger variance **in per-observation
units** (Section 7.2/7.2a), `ɣ` is the Euler–Mascheroni constant (≈0.5772), `e`
is Euler's number, and `Φ⁻¹` the normal quantile. Because `Var[SR_trials]` is
per-observation, `SR*₀` comes out in per-observation units and is directly
comparable to `SR_pp` inside the z-test. Then:

```
DSR = PSR(SR*₀)        — the probability the strategy's true Sharpe beats the
                         luck-maximum expected from N_sys trials.
```

**The system gate:** `DSR ≥ threshold` (RED, e.g. 0.95). A four-green-modules
stack with strong per-module reports but `DSR < threshold` is **rejected at the
system level** regardless of the module gates. This is the trap-breaker.

### 7.3a Singularity guards (degenerate variance and `N_sys = 1`)

`SR*₀` has two singularities that, left unguarded, make the gate trivially
passable — defeating the entire trap. Both are invariants with tests
(Invariant DSR-GUARD):

1. **Degenerate variance.** If all `K` ledger configs are near-identical,
   `Var[SR_trials] → 0` and `SR*₀ → 0`, so DSR collapses to `PSR(0)` (a system
   beats luck "easily" only because there was no search dispersion to deflate
   against). Guard:
   - **Minimum distinct trials:** require `K_distinct ≥ K_min` (RED, e.g. 8)
     genuinely de-correlated configs (correlation below the §7.1 clustering
     threshold) before the DSR gate may PASS. With fewer, the report is marked
     `insufficient_trials` and the config is **not promotable** — you have not
     searched enough to know whether the result is luck.
   - **Variance floor:** `Var[SR_trials] ← max(Var[SR_trials], var_floor)` where
     `var_floor` (RED) is a conservative lower bound on cross-config Sharpe
     dispersion (the sampling variance of a single PER-OBSERVATION Sharpe
     estimate, `(1 + SR_pp²/2)/n_obs`, which is the irreducible variance even of
     identical strategies). Because both `Var[SR_trials]` and `var_floor` are
     per-observation (Section 7.2a), they are commensurable. This keeps `SR*₀`
     strictly positive so the deflation always bites. (At thin `n_obs` the floor
     can DOMINATE the cross-config trial variance — see §7.3b — which correctly
     raises the deflated benchmark when the search has not produced much
     dispersion relative to single-estimate noise.)

2. **`N_sys = 1` ⇒ `Φ⁻¹(1 − 1/1) = Φ⁻¹(0) = −∞`.** The BLdP convention for a
   single trial is that there is **no** maximum-of-N inflation to deflate (the
   expected max of one draw is the draw itself), so:
   - **BLdP convention:** for `N_sys = 1`, set `SR*₀ = 0` and `DSR = PSR(0)` (the
     plain probabilistic Sharpe against a zero benchmark). This is the correct
     `N = 1` limit, not `−∞`.
   - `Invariant DSR` uses `N_sys = 1` as a test case precisely to exercise this
     branch; the implementation must return `PSR(0)`, never evaluate `Φ⁻¹(0)`.
   - The `N_sys`-derivation also floors at 1 (Section 7.1), so `N_sys ∈ [1, K]`
     always and `Φ⁻¹(1 − 1/(N_sys·e))` for `N_sys = 1` is `Φ⁻¹(1 − 1/e)` — finite
     and used only when `N_sys = 1` is reached via the floor with `K > 1`; the
     `SR*₀ = 0` convention above takes precedence and short-circuits this.

3. **Thin track record — the `N_sys` guard is the trial-COUNT axis; this is the
   orthogonal track-LENGTH axis.** The `K_min` guard in (1) protects against too
   few *trials*; it says nothing about whether the *promoted config's own return
   series is long enough* to estimate its Sharpe at all. With non-overlapping
   `H_sys` bucketing, a 3-year backtest collapses to `n_obs ≈ 108` observations
   (Section 7.2). A DSR computed off only 108 obs is **not** more certain than
   108 fat-tailed observations can support — the only place `n_obs` enters the
   DSR is `sqrt(n_obs − 1) = sqrt(107) ≈ 10.34×`, and a DSR reported without its
   `n_obs` invites reading a thin sample as a clean PASS. The
   Bailey–Borwein–López de Prado–Zhu **Minimum Track-Record Length** (the
   "Pseudo-Mathematics" minimum backtest length, cited in §17) is operationalized
   here as a hard RED floor:
   - **MinTRL floor.** Compute the smallest `n` at which the observed
     per-observation Sharpe `SR_pp` could plausibly clear the gate benchmark
     `SR*₀` (also per-observation) at the gate confidence, given the same
     non-normality terms PSR uses:

     ```
     MinTRL = 1 + ( 1 − γ3·SR_pp + ((γ4 − 1)/4)·SR_pp² )
                  · ( Z_gate / (SR_pp − SR*₀) )²

       where Z_gate = Φ⁻¹(DSR_threshold)   (the gate confidence, RED, e.g. 0.95
       ⇒ Z_gate ≈ 1.645), γ4 is RAW (non-excess) kurtosis (normal ⇒ 3), and
       SR_pp, SR*₀ are both in PER-OBSERVATION units (Section 7.2a) — the SAME
       units the z-test runs in. If SR_pp ≤ SR*₀ the config cannot clear the
       benchmark at any length — MinTRL is +∞ and the gate fails on edge, not on
       length.
     ```

     This is the algebraic inverse of the PSR/DSR z-test: it is exactly the `n`
     at which `(SR_pp − SR*₀)·sqrt(n − 1)/sqrt(denom) = Z_gate`. It uses `SR*₀`
     (not `0`) as the benchmark so the floor reflects the *deflated* gate the
     config actually faces. Note that MinTRL grows sharply as the per-observation
     edge `(SR_pp − SR*₀)` shrinks: a modest per-observation Sharpe over a
     deflated benchmark can require many hundreds of independent buckets, which
     a thin-sample candidate will not have (see §7.3b).
   - **The gate.** Require `n_obs ≥ max(MinTRL, n_obs_floor)` where `n_obs_floor`
     (RED, e.g. 24) is an absolute minimum independent-observation count below
     which no Sharpe is trusted regardless of MinTRL. If `n_obs` is below this,
     the report is marked `insufficient_track_record`, `passed = False`, and the
     config is **NOT promotable** — mirroring the `K_min` `insufficient_trials`
     guard but on the track-LENGTH axis (orthogonal to the trial-COUNT axis). A
     config may **not** PASS the DSR gate with `n_obs` below the floor *even when
     its point-DSR ≥ threshold*: a high point-DSR off a thin sample is precisely
     the failure mode this guard catches.
   - **Headline confidence qualifier.** The reported headline carries the DSR
     **together with `n_obs`, `MinTRL`, and the boolean `n_obs ≥ MinTRL`**, so a
     DSR off a thin `n_obs` is never read as more certain than that many
     observations support. The DSR alone is never the headline; `(DSR | n_obs vs
     MinTRL)` is.

### 7.3b Worked numeric example (so the inputs are unambiguous)

**Everything inside PSR/DSR is in PER-OBSERVATION units (Section 7.2a).** The
annualized Sharpes are an input convention only; we divide each by `sqrt(ppy)`
before it touches the gate. Here `ppy ≈ 36` (non-overlapping H_sys ≈ 7 trading
days), so `sqrt(ppy) = 6`.

```
Ledger: K = 12 distinct composed configs scored on the CPCV folds.
  Their OOS Sharpes (annualized, the reporting convention):
    [0.92, 0.61, 0.40, 0.85, 0.30, 0.71, 0.55, 0.20, 0.78, 0.48, 0.66, 0.35]
  PER-OBSERVATION trial_sharpes = that vector / sqrt(36) = / 6:
    [0.1533, 0.1017, 0.0667, 0.1417, 0.0500, 0.1183, 0.0917, 0.0333,
     0.1300, 0.0800, 0.1100, 0.0583]
  Var[SR_trials] (per-observation, ddof=1) ≈ 0.0014389  (sd ≈ 0.03793).
    (The annualized variance 0.0518 = 0.0014389 · 36, as expected.)
  Mean off-diagonal OOS-return correlation rho_bar ≈ 0.35
    -> N_eff = 12 / (1 + 11·0.35) ≈ 12 / 4.85 ≈ 2.47
    -> clustering at threshold gives ~4 de-correlated clusters
    -> N_sys = max(2.47, 4) = 4   (use the more conservative larger count).
  K_distinct = 12 ≥ K_min (8): gate is allowed to evaluate.

Promoted config c* :
  path_returns aggregated to non-overlapping H_sys-length buckets.
  Over ~3 yr with H_sys ≈ 7 trading days  ->  n_obs ≈ 108 buckets.
  SR_annual = 0.92  ->  SR_pp = 0.92 / 6 = 0.153333  (per-observation).
  γ3 = -0.4,  γ4 = 6.0  (per-observation shape moments; no rescaling).

Variance floor (per-observation Lo-2002):
  var_floor = (1 + SR_pp²/2)/n_obs = (1 + 0.153333²/2)/108 = 0.0093681.
  var_used  = max(Var[SR_trials]=0.0014389, var_floor=0.0093681) = 0.0093681.
  >>> The FLOOR DOMINATES: at n_obs=108 the irreducible single-estimate noise
      exceeds the cross-config dispersion, so SR*0 is set by the floor (sd 0.09679).
      This is correct, conservative behavior (Section 7.3a guard 1).

SR*0 = sqrt(0.0093681) · [ (1-0.5772)·Φ⁻¹(1 - 1/4) + 0.5772·Φ⁻¹(1 - 1/(4e)) ]
     = 0.096789 · [ 0.4228·Φ⁻¹(0.75) + 0.5772·Φ⁻¹(0.90802) ]
     = 0.096789 · [ 0.4228·0.674490 + 0.5772·1.328722 ]
     = 0.096789 · [ 0.285215 + 0.766938 ]
     = 0.096789 · 1.052153 ≈ 0.101834   (PER-OBSERVATION SR*0)

DSR = PSR(SR*0=0.101834) with SR_pp=0.153333, n_obs=108, γ3=-0.4, γ4=6.0
      (γ4 = non-excess / raw kurtosis; normal => 3)
    = Φ( (0.153333 - 0.101834)·sqrt(107)
         / sqrt(1 - (-0.4)·0.153333 + ((6.0-1)/4)·0.153333²) )
    = Φ( 0.051500·10.34408 / sqrt(1 + 0.061333 + 0.029389) )
    = Φ( 0.532721 / sqrt(1.090722) ) = Φ( 0.532721 / 1.044376 )
    = Φ(0.510079) ≈ 0.695002.

GATE: DSR = 0.6950  <  0.95 (RED threshold)  ==>  *** FAIL. NOT promotable. ***

MinTRL floor check (Section 7.3a guard 3; Z_gate = Φ⁻¹(0.95) ≈ 1.644854):
  denom = 1 - (-0.4)·0.153333 + ((6.0-1)/4)·0.153333² = 1.090722 (= the DSR denom)
  MinTRL = 1 + 1.090722·( 1.644854 / (0.153333 - 0.101834) )²
         = 1 + 1.090722·( 1.644854 / 0.051500 )²
         = 1 + 1.090722·(31.939)² = 1 + 1.090722·1020.1 ≈ 1114 obs.
  n_obs = 108  <  max(MinTRL ≈ 1114, n_obs_floor = 24) = 1114.
  ==> insufficient_track_record, passed = False — the same FAIL on the
      track-LENGTH axis. To clear the *deflated* benchmark at 95% confidence with
      this thin per-observation edge (SR_pp - SR*0 = 0.0515) the config would need
      ~1114 independent buckets (~30 years at ppy=36), not 108 (~3 years).

  HEADLINE qualifier: DSR = 0.6950 @ n_obs = 108 (< MinTRL ≈ 1114).
```

**This config FAILS — and that is the correct, honest verdict.** A 0.92
annualized Sharpe sounds strong, but in the per-observation units the z-test
actually lives in, `SR_pp = 0.153` is only a thin edge over the deflated
luck-maximum `SR*0 = 0.102`, and `sqrt(n_obs−1) = sqrt(107) ≈ 10.3` is too small
to make that gap significant at 95%. The gate correctly refuses to promote it on
**both** axes: the point-DSR (0.695) is below threshold, and the MinTRL floor
(~1114 obs needed vs 108 available) flags the track as far too short. This
directly reinforces the minimum-track-record discipline (Section 7.3a guard 3):
a thin-sample, modest-per-observation-edge candidate SHOULD fail, and feeding the
annualized 0.92 into the z-test instead — the now-corrected defect — would have
inflated `z` from 0.51 to ≈2.05 and spuriously reported DSR ≈ 0.98 (PASS),
flipping the gate purely on the annualization convention.

```
CONTRAST — a genuinely sufficient candidate (PASS):
  Same ledger / N_sys = 4 / γ3 = -0.4 / γ4 = 6.0, but a stronger, longer config:
    SR_annual = 1.80  ->  SR_pp = 1.80 / 6 = 0.300000  (per-observation).
    n_obs = 400 buckets (~11 yr at ppy=36, or a faster H_sys cadence).
  var_floor = (1 + 0.300²/2)/400 = 1.045/400 = 0.0026125; cross-config Var ≈ 0.0014389
    -> var_used = max(.,.) = 0.0026125 (floor still binds), sd = 0.051113.
  SR*0 = 0.051113 · 1.052153 ≈ 0.053777  (per-observation).
  denom = 1 - (-0.4)·0.300 + ((6.0-1)/4)·0.300² = 1 + 0.120 + 0.1125 = 1.2325.
  z = (0.300 - 0.053777)·sqrt(399)/sqrt(1.2325)
    = 0.246223·19.97498 / 1.110180 = 4.91836 / 1.110180 ≈ 4.4302.
  DSR = Φ(4.4302) ≈ 0.999995  ==> >= 0.95  *** PASS ***.
  MinTRL = 1 + 1.2325·(1.644854/0.246223)² = 1 + 1.2325·44.63 ≈ 56.0 obs.
  n_obs = 400 >= max(MinTRL ≈ 56, n_obs_floor = 24) = 56.  PASS (sufficient track).
  HEADLINE: DSR = 0.999996 @ n_obs = 400 (>= MinTRL ≈ 56).
```

The contrast makes the discipline explicit: a PASS requires a per-observation
edge over the deflated benchmark AND enough independent buckets to make it
significant — a higher `SR_pp` (here 0.300 vs 0.153) and a longer track (400 vs
108) together. Note how sensitive the realistic case is to the input contract:
feeding the BAR-count (n_obs ≈ 1.5M) would push the z-score absurdly high
regardless of edge; feeding the cross-PATH variance for `Var[SR_trials]` (often
an order of magnitude smaller) would shrink `SR*0` and let weak configs through;
and feeding the ANNUALIZED Sharpe into the z-test (the corrected defect) inflates
`z` by ≈sqrt(ppy) with no matching denominator term and manufactures a spurious
PASS. The source/units rules in 7.2 and 7.2a are what keep these numbers honest.

```python
class DeflatedSharpe:
    def deflate(self, path_returns: "np.ndarray", trial_sharpes: "np.ndarray",
                n_obs: int, skew: float, kurt: float,
                n_trials_effective: float, cfg) -> "DSRResult":
        """Compute PSR and DSR for the COMPOSED system. INPUT CONTRACT (Section
        7.2): path_returns = the PROMOTED config's NON-OVERLAPPING net-of-cost
        system return series; n_obs = COUNT of those non-overlapping observations
        (NOT bar/path/trade count); trial_sharpes = the per-LEDGER-CONFIG OOS
        Sharpes (cross-CONFIG, NEVER cross-CPCV-path; Invariant SG); skew/kurt are
        of path_returns (kurt = non-excess/RAW kurtosis, normal => 3, NOT excess);
        n_trials_effective = N_sys, the effective independent
        config count derived from the SAME ledger configs (Section 7.1) so it is
        consistent with Var[SR_trials]=var(trial_sharpes). Applies the singularity
        guards (Section 7.3a): floors Var[SR_trials] at var_floor, requires
        K_distinct >= K_min or returns passed=False with 'insufficient_trials',
        uses the BLdP N_sys=1 convention (SR*0=0 => DSR=PSR(0)) instead of
        evaluating Phi^-1(0), AND enforces the MinTRL track-length floor (guard 3):
        computes MinTRL from (SR_pp, SR*0, skew, kurt, Z_gate) and requires
        n_obs >= max(MinTRL, n_obs_floor) or returns passed=False with
        'insufficient_track_record' (a high point-DSR off a thin n_obs does NOT
        pass). PSR/DSR is fed PER-OBSERVATION Sharpes (Section 7.2a): SR_pp =
        SR_annual/sqrt(ppy) and each trial_sharpe likewise, so NO annualization
        factor enters the z-test; annualization is reporting-only, applied after
        the gate. Rejects 'annualization_mismatch' if any Sharpe entering the
        z-test is not per-observation or ppy is not derived from the H_sys
        bucketing that produced n_obs. Returns (PSR, DSR, SR*0,
        var_trials, n_sys, n_obs, min_trl, passed). The HEADLINE is always
        (DSR | n_obs vs MinTRL), never the DSR alone and never a vanilla Sharpe.
        Records every input so the deflation is fully auditable."""
```

### 7.4 Why joint, not per-module (the trap, stated once more)

If each module deflates only by its own `N_m`, and SYSTEM reports the raw
composed Sharpe, then four modules at PSR 0.8 each can combine into a system that
*looks* like PSR 0.99 but is actually below the joint luck-max. The
union-of-DOF deflation is the only correct accounting: you searched the joint
space, you must pay the joint multiple-testing tax.

But "joint" does **not** mean the Cartesian *product* fed to `SR*₀`. The product
counts grid points never composed and scored, and is inconsistent with the
variance we can actually estimate (Section 7.1, the consistency law). The correct
quantity is the **effective number of independent trials among the configs
actually scored**, `N_sys ∈ [1, K]`, with `Var[SR_trials]` over the same `K`
configs. The product survives only as `n_sys_nominal` telemetry — the size of the
space we declined to exhaustively search because it is compute-infeasible
(Section 9). The trap is broken by counting the real search honestly and floored
by `K_min` distinct trials, not by inventing a large unmeasured `N`.

### 7.5 Re-promotion, staleness, and overfit-by-iteration

Freezing a config for live trading is not the end of the multiple-testing story.
Every night the overnight job may re-search the joint BLUE grid, and **each
night's pick is another trial**. If those repeated searches are not accumulated
into the ledger, `N_sys` silently understates the true multiplicity over calendar
time, and the system overfits-by-iteration — the exact trap it claims to break,
just spread across days instead of within one night. The policy that closes this:

```
RE-PROMOTION CADENCE (RED):
  - The live config is frozen between swaps. The overnight job may propose a swap
    only on a fixed cadence tau_promote (e.g. monthly), NOT every night.
  - More frequent overnight re-scoring is allowed for MONITORING, but a
    monitoring run that does not propose a swap still appends its scored configs
    to the SAME persistent ledger (Invariant SG): trials accumulate across
    calendar time, so N_sys grows with every search ever run, not just tonight's.

FORWARD GATE before any swap (held-out, never used in the search):
  - A candidate that wins the overnight DSR gate (Section 7.3) must ALSO clear a
    forward gate on a held-out, strictly-later, never-searched window of length
    tau_forward (RED). The gate is a PAIRED test on the SAME forward block, NOT a
    bare point margin on the Sharpe difference. At ~36 obs/yr a forward block is a
    handful of non-overlapping buckets, where a single fat-tail trade can flip the
    SIGN of a point Sharpe difference — so a point margin is not admissible.
  - PAIRED-DIFFERENCE BLOCK-BOOTSTRAP (respecified):
      1. Run BOTH candidate and incumbent on the IDENTICAL forward bars (same
         universe, costs, folds, embargo, execution stub — Invariant BL).
      2. Form the per-bucket return DIFFERENCE series d_t = r_candidate,t -
         r_incumbent,t on the SAME non-overlapping H_sys buckets (a paired series,
         not two independent samples).
      3. BLOCK-bootstrap d_t (the Section 8.1 dependence-preserving block length;
         iid bootstrap is FORBIDDEN) to get a CI on the Sharpe/mean of the paired
         difference.
      4. REQUIRE the CI to EXCLUDE ZERO and its LOWER bound to clear round-trip
         cost (spread+impact+funding) before any swap. A point difference that is
         positive but whose paired-difference CI includes zero does NOT swap.
  - The candidate's forward DSR (deflated against the ACCUMULATED ledger N_sys)
    must ALSO still pass.
  - tau_forward must yield n_obs >= the §7.3a MinTRL floor (`max(MinTRL,
    n_obs_floor)`); otherwise the forward block is too thin to judge and the swap
    is DEFERRED with `insufficient_forward_track` (not swapped, reconsidered next
    tau_promote).
  - The forward window is embargoed from the search window by H_sys (Invariant
    HC) so the gate is causal.

ACCUMULATION (the anti-overfit-by-iteration rule):
  - N_sys is computed over the persistent ledger spanning ALL searches to date,
    not just the latest night. A strategy re-searched 200 nights has paid for 200
    nights of trials; its SR*0 benchmark rises accordingly. A config that only
    ever wins by being re-drawn many times fails the accumulated DSR gate.
```

> **Invariant RP — repeated searches accumulate; swaps pass a paired forward
> gate.** Every overnight search appends its scored configs to the persistent
> trial ledger; `N_sys` is derived from the full accumulated ledger. A live-config
> swap occurs only on the `tau_promote` cadence, only for a candidate that clears
> the held-out forward gate AND the accumulated-ledger DSR. The forward gate is a
> **paired block-bootstrap test on the per-bucket candidate-minus-incumbent return
> DIFFERENCE over the identical forward block** (Section 7.5): the bootstrap CI on
> the paired difference's Sharpe/mean must EXCLUDE ZERO and its lower bound clear
> round-trip cost — **never** a bare point margin — using the §8.1
> dependence-preserving block length (iid forbidden); and `tau_forward` must yield
> `n_obs ≥ max(MinTRL, n_obs_floor)` or the swap is deferred
> (`insufficient_forward_track`). Tests: (1) simulating `n` nightly re-searches
> grows the ledger by the configs scored each night and monotonically raises
> `SR*₀`; a config that wins only via repeated re-draws fails the accumulated
> gate. (2) a candidate that beats the incumbent on a POINT margin but whose
> paired-difference block-bootstrap CI includes zero does NOT swap.

---

## 8. Robustness, attribution, and required baselines

### 8.1 Drop-top-N + block-bootstrap CIs

```python
class RobustnessSuite:
    def run(self, path_returns, trades, regimes, cfg) -> "RobustnessReport":
        """1. DROP-TOP-N: recompute system Sharpe/DSR after removing the N most
              profitable trades (and the N worst), and after removing the single
              best instrument, best fold, best regime. A system whose edge
              evaporates when the top-N trades are dropped is a fat-tail/luck
              artifact, not a process (flag it; this is the convexity-vs-luck
              question from the convexity-repo backtests).
           2. BLOCK-BOOTSTRAP CIs on system Sharpe and on per-pillar attribution,
              with block length >= the autocorrelation/vol-clustering scale so
              dependence is preserved (iid bootstrap is forbidden here).
           3. REGIME / FOLD ATTRIBUTION: decompose the result by regime bucket
              (vol regime, trend/chop, asset class) and by fold; report whether
              one regime, one instrument, or one fold drives the edge."""
```

### 8.2 P&L attribution to each pillar — Shapley requires the full coalition lattice

A Shapley value over 4 pillars is the average marginal contribution over all 4!
orderings, which is computed from the OOS P&L of all **2⁴ = 16 coalition
subsets** `S ⊆ {selection, breakout, leverage, trend}`. The Shapley value of
pillar `p` is

```
ϕ_p = Σ_{S ⊆ N\{p}}  ( |S|! · (|N|−|S|−1)! / |N|! ) · ( v(S ∪ {p}) − v(S) )
```

where `v(S)` is the OOS net-of-cost P&L (under the identical contract) of the
system with **only** the pillars in `S` switched ON and the rest switched OFF.
Order-independence is a *theorem only if every `v(S)` is provisioned* — it is
**not** recoverable from 5 single-pillar ablations, and the cross-pillar
interactions are otherwise undefined. So either we provision all 16, or we do not
claim "Shapley" and "order-independent". Two honest modes:

**Mode A — full Shapley (only if the 16 coalition baselines are provisioned).**
The 16 coalitions and their precise **off-semantics** (what each pillar does when
OFF) are:

```
PILLAR-OFF SEMANTICS (each pillar, when OFF, is replaced by a NULL policy that
keeps the pipeline runnable under the identical contract — Invariant BL):
  selection OFF : no selection filter. Candidate set = the FULL eligible
                  universe; risk-weights = equal-risk; rank = constant (no
                  rotation, no demotion — incumbents never demoted by selection).
  breakout  OFF : no entry edge. Entry = RANDOM matched-frequency admission
                  (same trade count + holding-time distribution as full system,
                  random side/timing) — the breakout null of Section 8.3.
  leverage  OFF : no ceilings/governor. Sizing = FIXED per-position notional
                  (RED constant fraction of equity), still lot-quantized DOWN,
                  no min()-of-ceilings, no drawdown governor.
  trend     OFF : no adaptive exit. Exit = FIXED vertical barrier at H_sys (and
                  the fixed ±b·σ horizontal barriers of the label), no convex
                  trail, no reduce.

The 16 subsets S (pillars ON listed; the rest OFF by the above):
  {}                                  v(∅)  = all four OFF (random entry, full
                                              universe, fixed size, fixed exit)
  {sel} {brk} {lev} {trn}             4 singletons
  {sel,brk} {sel,lev} {sel,trn}
  {brk,lev} {brk,trn} {lev,trn}       6 pairs
  {sel,brk,lev} {sel,brk,trn}
  {sel,lev,trn} {brk,lev,trn}         4 triples
  {sel,brk,lev,trn}                   the full system
```

Every `v(S)` runs under the same bars, costs, folds, embargo, label function, and
execution stub (Invariant BL). The combiner-evidence cache (Section 9) makes the
16 runs cheap *only* for pillars that are pure combiner toggles; pillars whose OFF
state changes module evidence (breakout-off random entry, leverage-off fixed
size) require their own cached evidence keyed by the OFF policy.

**Mode B — sequential ablation (the DEFAULT; honest, compute-cheap, NOT
order-independent).** Provision a **single fixed knock-in order** (RED, e.g.
`∅ → +leverage → +trend → +selection → +breakout`) and report each pillar's
marginal contribution *in that order only*. This is 5 runs, not 16. We **do not**
call this a Shapley value and **do not** claim order-independence; the report
states the order explicitly and notes the contributions are order-dependent.

The compute budget (Section 9) decides the mode: full Shapley's 16 coalitions are
provisioned only if the overnight wall-clock (Section 9.2) admits them for the
chosen `(n_groups, k_test, max_paths)`; otherwise the system uses Mode B and the
report is labeled `attribution=sequential_fixed_order`. Claiming Shapley while
only 5 ablations were run is forbidden (Invariant ATTR).

```python
class PillarAttribution:
    def attribute(self, coalition_pnl: dict, mode: str, fixed_order: tuple, cfg
                  ) -> "AttributionReport":
        """mode='shapley' REQUIRES coalition_pnl to contain all 2^4=16 coalition
        baselines (keys = frozensets of ON pillars), each scored under the
        identical contract with the Section-8.2 off-semantics; computes the exact
        Shapley value phi_p and labels attribution='shapley'. mode='sequential'
        REQUIRES only the 5 knock-in runs along fixed_order, computes
        order-dependent marginal contributions, and labels
        attribution='sequential_fixed_order'. It is an ERROR to pass mode='shapley'
        with fewer than 16 coalitions (Invariant ATTR). A pillar with ~zero
        contribution (either mode) is not earning its complexity and its inclusion
        must be re-justified (mirror of TREND.md's Final Rule)."""
```

### 8.3 Required baselines and single-module ablations

Every baseline runs under the **identical** label contract, costs, folds,
embargo, and execution stub as the full system (Invariant BL).

```
Trivial baselines (the system must beat these on DSR, not raw Sharpe):
  - BUY & HOLD per instrument and equal-weight book.
  - RANDOM matched-frequency entries (same number of trades, same holding
    distribution, random side/timing) — the null that a real edge must clear.

Single-module ablations (turn one pillar OFF, keep the rest):
  - BREAKOUT-ONLY      : no selection (full universe), no trend exit (fixed
                         barrier), governor-off; isolates raw entry edge.
  - NO-SELECTION       : trade every breakout-qualified instrument; isolates
                         selection's diversification/turnover value.
  - NO-TREND           : fixed-horizon / fixed-barrier exit instead of TREND;
                         isolates the convex-exit contribution.
  - GOVERNOR-OFF       : LeverageEngine drawdown governor disabled; must show
                         WORSE max drawdown (the governor's job), confirming it
                         earns its place.
  - NO-LEVERAGE-CEILINGS (cautionary): fixed leverage; should show fatter tails /
                         worse ES, demonstrating the ceilings bind for a reason.

Full Shapley coalition baselines (ONLY in attribution Mode A, Section 8.2):
  the 2^4 = 16 coalition subsets S of {selection, breakout, leverage, trend},
  each with the precise pillar-OFF semantics defined in Section 8.2 (selection
  OFF = full universe + equal-risk + constant rank; breakout OFF = random
  matched-frequency entry; leverage OFF = fixed-fraction size, lot-quantized,
  no governor; trend OFF = fixed H_sys vertical + fixed horizontal barriers).
  The five single-pillar ablations above are a SUBSET of these 16; they suffice
  ONLY for sequential (Mode B) attribution, NOT for a Shapley value (Invariant
  ATTR). Provision all 16 only if the overnight budget (Section 9.2) admits them.
```

> If the full system does not beat random-matched-frequency entries on DSR, there
> is no edge — stop and research, do not ship. If an ablation that removes a
> pillar does NOT degrade the result, that pillar is not contributing on this
> data and its complexity is unjustified (mirror of TREND.md's Final Rule).

---

## 9. Caching, compute, and the overnight schedule (P7)

```python
class EvidenceCache:
    def get_or_compute(self, module: str, module_version: str, fold_id: str,
                       config_hash: str, bar_range: tuple, compute_fn) -> object:
        """Memoize a module's per-bar evidence (breakout proposals, vol forecasts,
        ceilings, trend beliefs) keyed by (module, module_version, fold_id,
        config_hash, bar_range). A SYSTEM config that changes ONLY a combiner
        parameter (precedence, cadence, hysteresis, budget rule) has the SAME keys
        for all module evidence and re-scores from cache for FREE — only the
        cheap composition + labeling + deflation re-runs (Invariant CA). On-disk,
        regime-bucketed, content-addressed by config_hash."""
```

### 9.1 Compute discipline (commodity hardware)
- **O(n) streaming composition.** The per-bar pipeline (Section 2) is O(active
  positions), not O(universe); the fixed point is ≤ `r_max` small iterations.
- **Sampled CPCV** with `max_paths`; full CPCV only if a fold count is tiny.
- **Heavy validation overnight:** the full ledger scoring, CPCV path
  distribution, bootstrap, and deflation run as a batch job; live composition
  reads only the frozen promoted config.
- **numba only after correctness tests** pass on the pure-Python path; no per-bar
  optimization.

### 9.2 The overnight compute budget — explicit cost model and hard cap

The overnight job is **not** allowed to be open-ended. On a 2-core / 11 GB ARM
VM, a naive plan (full Cartesian grid × full CPCV × 16 coalitions × multi-year
1-min bars) is **days, not one night**. We bound it with a per-stage cost model,
choose the CPCV knobs to fit, and cap the number of joint configs scored.

```
NOTATION
  B      = 1-min bars per instrument per year ≈ 525,600;  Y years;  U instruments.
  P      = number of CPCV paths actually scored = min( C(n_groups, k_test),
                                                       max_paths )   (RED knobs).
  K      = number of distinct COMPOSED configs scored (the ledger length).
  t_mod  = wall-time to compute ONE module's per-bar evidence over the full
           timeline for ONE config ON ONE path (the expensive, cached part).
  t_comp = wall-time to run composition + labeling + deflation over one path
           (the cheap, never-cached part) — O(active positions), << t_mod.

PER-STAGE COST (with the EvidenceCache, P7)
  module evidence (paid ONCE per distinct module config per path):
      cost_evidence  =  (# distinct module configs across the ledger)  ·  P  ·  t_mod
      *** A config that changes only a COMBINER param adds ZERO here (cache hit). ***
  composition+labeling+deflation (paid per (config × path), always):
      cost_compose   =  K  ·  P  ·  t_comp
  attribution:
      Mode B (sequential): + 5 · P · (t_mod_off + t_comp)   for the 5 knock-ins
      Mode A (Shapley):    + up to 16 · P · (t_mod_off + t_comp)   — 16 coalitions
  robustness (drop-top-N, block-bootstrap): O(K · P) cheap recomputations on the
      already-scored return series (no module re-run) — bounded by cost_compose.

  TOTAL ≈ cost_evidence + cost_compose + cost_attribution
       ≈ P · [ (#distinct module configs)·t_mod  +  K·t_comp  +  attrib·t_mod_off ].

THE HARD CAP (RED, must fit one night ≈ T_night, e.g. 8 h wall-clock):
  Choose (n_groups, k_test, max_paths) and K so that
      P · [ (#distinct module configs)·t_mod + K·t_comp + attrib·t_mod_off ]  ≤  T_night.
  Because t_mod dominates and scales with B·Y·U, the binding term is almost always
  cost_evidence. So the cap is driven by DISTINCT MODULE configs × paths, NOT by K
  (combiner-only configs are nearly free — that is the whole point of P7).
```

**Concrete target-VM budget (RED defaults, to be re-measured on the real VM):**

```
  n_groups = 6,  k_test = 2   ->  C(6,2) = 15 arrangements;  max_paths = 8
      => P = 8 sampled paths (deterministic seed, recorded).
  K (max joint configs the overnight job may score) = 24
      of which AT MOST 8 change module evidence (distinct module configs);
      the other 16 are combiner-only variants that re-score from cache for free.
  Attribution: Mode B (5 knock-ins) by default; Mode A (16 coalitions) ONLY when
      a lighter (n_groups, k_test, max_paths) is chosen so the night still fits.
  Instruments per overnight run: U ≤ 8 (the small retail book).

  Wall-clock estimate (illustrative, t_mod ≈ 90 s/config/path on 2 cores for
  ~3 yr of 1-min bars across U=8 with numba-warmed module evidence,
  t_comp ≈ 5 s/config/path):
      cost_evidence ≈ 8 · 8 · 90 s  ≈ 5,760 s  ≈ 1.6 h
      cost_compose  ≈ 24 · 8 · 5 s  ≈   960 s  ≈ 0.27 h
      cost_attrib(B)≈ 5 · 8 · 95 s  ≈ 3,800 s  ≈ 1.06 h
      TOTAL         ≈ 2.9 h  < T_night (8 h).  FITS.
  Switching attribution to Mode A (16 coalitions) at the same knobs adds
  ≈ 11 · 8 · 95 s ≈ 2.3 h -> TOTAL ≈ 5.2 h, still < 8 h but tighter; if it would
  exceed T_night, reduce max_paths or U, OR fall back to Mode B (Section 8.2).
```

> **Invariant CB-BUDGET — the overnight job has a hard, pre-declared cap.** The
> RED config fixes `(n_groups, k_test, max_paths)` and the **maximum number of
> joint configs `K_max`** (and of distinct module configs) the overnight job may
> score, chosen so the per-stage cost model above is `≤ T_night` on the target
> VM. The job refuses to enqueue more than `K_max` configs and records the
> measured wall-clock; if a run exceeds `T_night`, it aborts cleanly to the last
> promoted config rather than bleeding into the trading day. Test: a synthetic
> ledger of `K_max + 1` configs is rejected; the cost estimate is logged and
> compared to actual wall-clock each run.

---

## 10. Directory layout

```text
system/
+-- config/
|   +-- defaults.py          # SystemDefaults: RED composition constants (precedence,
|   |                        #   cadence, deadband, hysteresis, r_max, eta, H_book share,
|   |                        #   embargo construction, triple-barrier multiples, DSR conf)
|   +-- universe.py          # AMBER: universe membership, per-instrument metadata
|   +-- tuning.py            # BLUE: finite composition grids + promotion gates (DSR)
|   +-- resolve.py           # single merge point (RED unreachable from instrument)
+-- modules/                 # THIN ADAPTERS to the four built modules (import, don't reimpl)
|   +-- selection.py         # wraps the Carver Ch.25 selection module's select()
|   +-- breakout_adapter.py  # wraps BreakoutModule.propose()
|   +-- leverage_adapter.py  # wraps LeverageEngine.decide()/size()
|   +-- trend_adapter.py     # wraps TrendModule.update()
+-- compose/                 # THE ALGEBRA (Sections 2-4)
|   +-- precedence.py        # deterministic pipeline A..G
|   +-- iterate.py           # selection<->leverage fixed point Phi (Section 3)
|   +-- budget.py            # residual-budget priority, MCTR-efficiency (no edge)
|   +-- incumbent.py         # hold-incumbent firewall + rotation test (Section 4)
|   +-- cadence.py           # decision-bar / deadband / event-recompose logic
|   +-- orchestrator.py      # SystemOrchestrator.decide(): one public method
+-- nautilus/                # GLUE to NautilusTrader (no engine reimplementation)
|   +-- adapter.py           # read book/account/position; emit SystemDecision deltas
|   +-- exec_stub.py         # CONSERVATIVE cost+fill UPPER bound, validation only (6.1b, Inv XC)
+-- validate/                # PART OF THE MODEL; never imported by compose/ or modules/
|   +-- labels.py            # system triple-barrier labels; vertical=H_sys (firewalled, future bars)
|   +-- walkforward.py       # purged walk-forward; embargo keyed to H_sys (Inv HC)
|   +-- cpcv.py              # sampled combinatorial purged CV (Section 6.4)
|   +-- deflated.py          # PSR / DSR over EFFECTIVE N_sys + singularity guards (var/K_min/N=1/
|   |                        #   MinTRL track-length floor) + one-ppy annualization (Sec 7) — centerpiece
|   +-- robustness.py        # drop-top-N + block-bootstrap CIs + regime/fold attrib
|   +-- attribution.py       # Shapley (16 coalitions) OR sequential (5 knock-ins) (Section 8.2)
|   +-- baselines.py         # buy&hold, random-matched, ablations, 16 coalition off-semantics (8.3)
|   +-- trials.py            # the PERSISTENT trial LEDGER -> effective N_sys (Inv SG); accumulates (7.5)
|   +-- promotion.py         # re-promotion cadence + held-out PAIRED block-bootstrap forward gate
|   |                        #   (candidate-minus-incumbent diff CI excludes 0) (Section 7.5, Inv RP)
|   +-- budget.py            # overnight per-stage cost model + K_max/T_night cap (9.2, Inv CB-BUDGET)
|   +-- cache.py             # EvidenceCache (Section 9)
|   +-- reports.py
+-- synth/
|   +-- generate.py          # adversarial COMPOSED scenarios (Section 13)
+-- types.py
+-- system_api.py            # public entry point: one decide() for live, one for batch
tests/
+-- ...                      # composition, firewall, parity, causality, acceptance, ablation
```

Stack: Python 3.11+, numpy, scipy, pandas, pytest, NautilusTrader (engine). No ML
frameworks. uv for env/deps.

---

## 11. Shared data objects (`types.py`)

```python
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
import numpy as np

Side = Literal[-1, 1]
PillarName = Literal["selection", "breakout", "leverage", "trend"]
ActionKind = Literal["open", "close", "reduce", "hold", "flat_all"]

@dataclass(frozen=True)
class ProposedPortfolio:        # FROM the Carver Ch.25 selection module
    keep: tuple[str, ...]       # instruments selection wants capital in
    eligible: tuple[str, ...]   # instruments eligible for NEW entries this bar
    risk_weights: Mapping[str, float]   # target RISK weights (NOT sizes; no edge)
    rank: Mapping[str, float]   # diversification/quality rank (NEVER sizes)
    reason_codes: tuple[str, ...]

@dataclass(frozen=True)
class BookState:                # READ from NautilusTrader (F_t)
    equity: float
    peak_equity: float
    free_margin: float
    positions: Mapping[str, "OpenPosition"]
    cov: "np.ndarray"           # LeverageEngine covariance (RISK only)
    governor_multiplier: float  # m in [0,1] from LeverageEngine drawdown governor

@dataclass(frozen=True)
class OpenPosition:
    instrument: str
    side: Side
    units: int
    entry_price: float
    held_risk_weight: float     # reserved MCTR budget (P4)
    trend_state: str            # "alive" / "reduce" / "death" reason from TREND

@dataclass(frozen=True)
class SystemAction:             # one line of the book delta
    instrument: str
    kind: ActionKind
    target_units: int           # post-quantize, round-DOWN (LEVERAGE Invariant U)
    side: Side | None
    driven_by: PillarName | Literal["governor", "iterate", "system"]
    binding: str                # binding ceiling / precedence rule / reason
    reason_chain: tuple[str, ...]   # full audit trail across modules

@dataclass(frozen=True)
class SystemDecision:           # OUTPUT of one decide()
    bar_ts: int
    actions: tuple[SystemAction, ...]
    iterations: int             # fixed-point iterations used (<= r_max)
    converged: bool
    book_heat_used: float       # sum MCTR after the delta
    equity_ladder_used: Mapping[str, float]
    deferred: tuple[str, ...]   # budget_deferred candidates
    reason_codes: tuple[str, ...]
    system_version: str

@dataclass(frozen=True)
class TrialRecord:              # one row of the ledger -> N_sys
    config_hash: str
    module_versions: Mapping[str, str]
    blue_point: Mapping[str, object]   # the joint BLUE grid coordinate
    oos_sharpe: float
    oos_returns_ref: str        # pointer to the path-return series (for corr -> effective trials)

@dataclass(frozen=True)
class SystemReport:             # OUTPUT of validation
    psr: float
    dsr: float
    sr_star0: float             # luck-max benchmark from N_sys (0 when N_sys==1)
    n_sys: float                # EFFECTIVE independent trials, 1 <= n_sys <= K (Sec 7.1)
    n_sys_nominal: int          # Cartesian product, TELEMETRY ONLY (never fed to SR*0)
    k_ledger: int               # K = distinct configs scored (ledger length)
    k_distinct: int             # de-correlated trial count (vs K_min guard)
    var_trials: float           # cross-CONFIG ledger variance, POST var_floor (Sec 7.2/7.3a)
    var_floored: bool           # whether var_floor bound (degenerate-variance guard)
    n_obs: int                  # NON-OVERLAPPING system observations (NOT bars/paths/trades)
    ppy: float                  # periods_per_H_sys: the SINGLE annualization cadence (Sec 7.2a)
    min_trl: float              # MinTRL track-length floor (Sec 7.3a guard 3); +inf if SR_pp<=SR*0
    n_obs_sufficient: bool      # n_obs >= max(MinTRL, n_obs_floor) (Invariant DSR-MINTRL)
    rho_cl: float               # RED clustering threshold used for N_clusters (Sec 7.1, Inv DSR-MONO)
    walkforward: Mapping[str, float]
    cpcv_path_sharpes: tuple[float, ...]   # CPCV distribution -> bootstrap/attrib, NOT trial_sharpes
    drop_top_n: Mapping[str, float]
    bootstrap_ci: Mapping[str, tuple[float, float]]
    attribution_mode: Literal["shapley", "sequential_fixed_order"]  # Invariant ATTR
    pillar_contribution: Mapping[PillarName, float]  # Shapley phi_p OR fixed-order marginals
    baselines: Mapping[str, float]
    regime_attribution: Mapping[str, float]
    forward_gate: Mapping[str, float] | None  # held-out PAIRED forward-gate result on a swap
                                # (Sec 7.5): carries the candidate-minus-incumbent paired-difference
                                # block-bootstrap CI bounds (ci_low, ci_high), the difference Sharpe,
                                # the block length used, and forward n_obs vs MinTRL
    overnight_wall_clock_s: float             # measured vs T_night (Invariant CB-BUDGET)
    passed: bool                # DSR gate AND beats random-matched AND K_distinct>=K_min
                                # AND n_obs >= max(MinTRL, n_obs_floor) (Invariant DSR-MINTRL)
    reason_codes: tuple[str, ...]   # e.g. 'insufficient_trials', 'insufficient_track_record',
                                # 'annualization_mismatch', 'insufficient_forward_track',
                                # 'psr_denom_floored'
```

---

## 12. One-public-method module contracts (signatures are contracts)

```python
class SystemOrchestrator:
    def decide(self, book: BookState, sel: ProposedPortfolio,
               open_actions: Mapping[str, "TrendAction"],
               candidates: Mapping[str, "Signal"],
               cfg: "ResolvedConfig") -> SystemDecision:
        """Run the deterministic precedence pipeline (Section 2) and the
        selection<->leverage fixed point (Section 3) to produce ONE target book
        delta. Pure function of F_t and BookState. Imports the four modules only
        through modules/ adapters; never reaches inside them (P1). Emits a full
        reason chain per action. Deterministic: identical inputs -> identical
        SystemDecision. Never places raw orders — returns deltas for the Nautilus
        adapter to execute."""

class SelectionAdapter:
    def select(self, universe: Sequence[str], evidence: "EvidenceCache",
               cfg: "ResolvedConfig") -> ProposedPortfolio:
        """Thin wrapper over the Carver Ch.25 selection module. Returns a
        RISK-weighted ProposedPortfolio + a diversification/quality rank. The rank
        NEVER becomes a size and NEVER enters the budget-priority firewall
        (Invariant A inherited). Cross-asset via AMBER metadata only."""

class FixedPointComposer:
    def solve(self, incumbents: Mapping[str, OpenPosition],
              candidates: Mapping[str, "Signal"], book: BookState,
              cfg: "ResolvedConfig") -> tuple[Mapping[str, int], int, bool]:
        """Damped projection fixed point Phi (Section 3). Returns
        (integer_book, iterations, converged). Incumbents reserved first (P4);
        new-capital weights scaled DOWN only; lot rounding DOWN only. On
        non-convergence returns the last FEASIBLE integer book with converged=False
        (Invariant IT). Deterministic tie-break by instrument id."""

class DeflatedSharpe:
    def deflate(self, path_returns: "np.ndarray", trial_sharpes: "np.ndarray",
                n_obs: int, skew: float, kurt: float,
                n_trials_effective: float, cfg) -> "DSRResult":
        """PSR/DSR over N_sys (Section 7). The reported headline performance
        number for the SYSTEM is the DSR, never the vanilla Sharpe (Invariant
        DSR)."""

class SystemCPCV:
    def evaluate(self, bars, universe, cfg, cost, n_groups, k_test, embargo_bars,
                 max_paths, evidence_cache) -> SystemReport:
        """Sampled combinatorial purged CV on the COMPOSED chain (Section 6.4).
        Pulls module evidence from cache so combiner-only configs re-score for
        free (P7)."""

class PillarAttribution:
    def attribute(self, coalition_pnl: dict, mode: str, fixed_order: tuple, cfg
                  ) -> "AttributionReport":
        """Pillar P&L decomposition (Section 8.2). mode='shapley' REQUIRES all
        2^4=16 coalition baselines with the defined off-semantics; mode='sequential'
        uses the 5 knock-ins along fixed_order and is order-dependent. Claiming
        Shapley from <16 coalitions is an error (Invariant ATTR)."""
```

Each module is testable in isolation against synthetic inputs; the orchestrator
is testable against scripted module outputs without running the real modules.

---

## 13. Synthetic adversarial data (`synth/generate.py`)

Composed scenarios so the SYSTEM is fully testable with no external data. Each
asserts a *composition* behavior, not a single module's:

- **Budget contention:** more qualified breakouts than the book-heat budget
  admits → priority firewall admits by ceiling-slack/MCTR-efficiency, defers the
  rest with `budget_deferred`, NEVER admits by score.
- **Selection-drop vs trend-ride:** an open, trend-alive, ceiling-slack position
  is demoted by selection with no qualifying replacement → `hold_incumbent`
  (Invariant HI); with a risk-improving, cost-clearing replacement → `sel_rotate`.
- **Correlation-to-1 contention:** many correlated qualified signals → the fixed
  point converges to far less aggregate exposure than the sum of standalone sizes
  (MCTR shared budget), and the dual equity cap binds on the rounded book.
- **Governor cascade:** progressive drawdown 0→43% scales every action, then
  flats the book and halts opens at 43% (`flat_all`); positions that round below
  one lot under the throttle are reported, not silently dropped.
- **Cadence/deadband:** sub-deadband target changes generate NO orders; an event
  (trend exit, ceiling collapse, new qualification) forces an immediate recompose
  of the affected slice only.
- **Non-convergence / oscillation:** an adversarial covariance that would make a
  naive integer loop oscillate (A-drops/B-fits/B-lifts-A's-ceiling cycle, Section
  3.3) → the monotone greedy fill terminates without looping and yields ONE
  deterministic feasible book; an unsettled continuous relaxation → last-feasible
  fallback with `iterate_fallback` and `converged=False`.
- **Endogenous-horizon leak:** a persistent multi-week trend-ride whose realized
  hold would exceed a *typical* horizon → with `tau_cap = H_sys` it force-closes
  `time_cap` at `H_sys` and the embargo (keyed to `H_sys`) purges it; a variant
  with an embargo keyed to the *mean* horizon must show inflated walk-forward vs
  CPCV and be caught (Invariant HC).
- **Validation traps:** a "strategy" that is pure luck (random matched-frequency
  entries dressed as a system) must FAIL the DSR gate; a system whose entire edge
  is the top-N trades must be flagged by drop-top-N; a system with a leaking
  embargo (per-module instead of union) must show inflated walk-forward vs CPCV
  and be caught.
- **DSR input-provenance traps:** feeding the CROSS-CPCV-PATH variance as
  `Var[SR_trials]` (instead of cross-config) must be rejected/flagged; feeding the
  BAR count as `n_obs` (instead of non-overlapping observations) must be
  rejected/flagged (Invariant SR-SRC). A ledger of `K` near-identical configs
  (`Var[SR_trials]→0`) must NOT pass trivially — it returns `insufficient_trials`
  or a floored-variance `SR*₀>0` (Invariant DSR-GUARD). An `N_sys=1` config must
  evaluate `SR*₀=0, DSR=PSR(0)`, never `Φ⁻¹(0)`.
- **Thin-track-record trap (track-LENGTH axis):** a config with a high point-DSR
  (e.g. ≥0.999) but `n_obs < max(MinTRL, n_obs_floor)` — e.g. a ~4-month backtest
  giving `n_obs ≈ 12` — must return `insufficient_track_record`, `passed=False`,
  and be NOT promotable; the headline must carry `(DSR | n_obs, MinTRL)` so the
  thin sample is visible (Invariant DSR-MINTRL). The §7.3b realistic worked
  example must reproduce `n_obs=108` vs `MinTRL≈1114` (a FAIL on both DSR and
  track length); the PASS contrast reproduces `n_obs=400` vs `MinTRL≈56`.
- **N_sys threshold-monotonicity trap:** sweeping the clustering threshold
  `rho_cl` from loose→tight must produce a monotonically NON-DECREASING `N_sys`
  (the penalty cannot be shrunk by loosening the threshold); an all-anti-correlated
  (or zero-correlated) ledger must yield `N_sys = K` (Invariant DSR-MONO).
- **Per-observation-units trap:** PSR/DSR must be fed the PER-OBSERVATION Sharpe
  `SR_pp = SR_annual/sqrt(ppy)` (and every trial_sharpe likewise), with `var_floor
  = (1 + SR_pp²/2)/n_obs` and `SR*₀` also per-observation — so NO annualization
  factor enters the z-test. Feeding the ANNUALIZED Sharpe instead inflates `z` by
  ≈`sqrt(ppy)` (numerator scales, the non-normality denominator does not) and
  manufactures a spurious near-1 DSR; this must be rejected as
  `annualization_mismatch`, as must a `ppy` not derived from the `n_obs` bucketing
  (e.g. `sqrt(252)` on daily while `n_obs` is on the ~36/yr cadence) (Invariant
  DSR-ANN). DSR is NOT invariant to feeding different annualizations into the
  test; the invariant practice is to keep annualization OUT of the test.
- **Paired-forward-gate trap:** a swap candidate that beats the incumbent on a
  POINT Sharpe-difference margin but whose per-bucket paired-difference
  block-bootstrap CI (candidate minus incumbent on the identical forward block,
  §8.1 block length, never iid) INCLUDES zero must NOT swap; a forward block too
  thin for `n_obs ≥ max(MinTRL, n_obs_floor)` must defer with
  `insufficient_forward_track` (Invariant RP).
- **Overfit-by-iteration trap:** a config that wins only by being re-searched many
  nights → accumulating each night's configs into the persistent ledger raises
  `SR*₀` until it FAILS the accumulated-ledger DSR (Invariant RP).
- **Cost-stub under-charge trap:** an execution stub that charges less than full
  spread + adverse slippage + funding inflates net P&L → the conservatism test
  (stub cost ≥ mid/half-spread reference per fill) catches it (Invariant XC).

Behavioral assertions (must hold on synthetic data): composition is
deterministic; the integer book never oscillates; incumbents are protected;
budget never exceeds `H_book`; equity ladder holds on the rounded book; governor
only de-risks; no realized hold exceeds `H_sys`; `Var[SR_trials]` and `N_sys` come
from the same ledger; the DSR gate rejects the luck strategies, the random
baseline, the degenerate-variance ledger, the thin-track-record (sub-MinTRL)
config, and the under-charging cost stub; `N_sys` is non-decreasing as the
clustering threshold tightens and equals `K` for an anti-correlated ledger;
PSR/DSR runs in per-observation units (annualization stays OUT of the z-test) and
feeding an annualized Sharpe is rejected; a swap requires the
paired-difference forward CI to exclude zero (never a point margin);
combiner-only config changes re-score from cache (zero module recompute).

---

## 14. Invariants (enforce structurally; ask before breaking)

**Invariant CB — composition is `min()`/precedence/sets, never a score blob.**
No weighted multiplicative or additive blend of module outputs exists. The
composed decision depends on breakout's *qualification + side*, never on its
score magnitude. Test: scaling every breakout score by an arbitrary positive
constant leaves every SystemDecision unchanged.

**Invariant HI — incumbents protected (Section 4).** A trend-alive,
ceiling-slack open position is never closed by selection alone.

**Invariant IT — bounded, feasible, oscillation-free integer book.** The
continuous damped relaxation runs ≤ `r_max` steps; the integer book is then
resolved by the **single monotone greedy fill** of Section 3.3 (lots only added,
never revoked), which terminates by construction and **cannot oscillate** between
equal-heat configurations. `converged` is defined honestly as *integer book
stable on two consecutive decision bars given identical inputs* — NOT as a
fixed-point of a discontinuous integer map. The emitted book is always feasible
(book heat + dual equity cap satisfied). Tests: (1) the A-drops/B-fits
oscillation synthetic terminates without looping and yields one deterministic
book; (2) a continuous relaxation that has not settled within `r_max` emits the
greedy-filled feasible book with `converged=False` and `iterate_fallback`, never
an infeasible book and never an unbounded loop.

**Invariant SC — system causality / no cross-module fold leakage.** No module's
test fold leaks into another module's fit; the embargo is the union (max) of all
module label horizons; every TEAL fit in every module is fold-local. Test: a
fold's training set, after purge+embargo, contains no bar whose label window
overlaps any test bar of any module.

**Invariant FW — validation labels firewalled.** Nothing in `modules/`,
`compose/`, or `nautilus/` imports from `validate/`. System triple-barrier labels
(future bars) are structurally unreachable from live composition. Test: import
graph has no edge from live packages into `validate/`.

**Invariant DSR — system Sharpe is deflated over the effective joint trials, and
`N_sys` is consistent with `Var[SR_trials]`.** The reported headline performance
is the DSR computed against `N_sys` (Section 7.1: the BLdP **effective number of
independent trials among the configs actually scored**, `1 ≤ N_sys ≤ K`,
**NOT** the fabricated Cartesian product), and `Var[SR_trials]` is the variance
over the **same** `K` ledger configs (the consistency law). The reported headline
is never a vanilla or per-module-only Sharpe. Tests: (1) the product is recorded
only as `n_sys_nominal` and never passed to `SR*₀`; (2) `N_sys` and
`Var[SR_trials]` are computed from the same ledger config set (a config in one
but not the other fails the audit); (3) the `N_sys = 1` test case returns
`SR*₀ = 0, DSR = PSR(0)` (BLdP convention), never `Φ⁻¹(0)`.

**Invariant DSR-GUARD — singularity guards hold (Section 7.3a).** `Var[SR_trials]`
is floored at `var_floor`; the gate cannot PASS with fewer than `K_min` distinct
de-correlated trials (`insufficient_trials`); `N_sys = 1` uses the BLdP zero-
benchmark convention. Test: a ledger of near-identical configs (`Var→0`) does NOT
pass the gate trivially — it returns `insufficient_trials` or a floored-variance
`SR*₀ > 0` that still bites.

**Invariant DSR-MINTRL — the track-LENGTH floor gates thin samples (Section
7.3a guard 3).** The promoted config may not PASS the DSR gate unless
`n_obs ≥ max(MinTRL, n_obs_floor)`, where `MinTRL = 1 + (1 − γ3·SR_pp +
((γ4−1)/4)·SR_pp²)·(Z_gate/(SR_pp − SR*₀))²` (per-observation units, Section
7.2a) is the Bailey–Borwein minimum track-record length at the gate confidence
(RED `Z_gate`, `n_obs_floor`). This is orthogonal to the trial-COUNT axis
(`K_min`, Invariant DSR-GUARD): a config may have plenty of trials yet too short
an own track. Below the floor the report is `insufficient_track_record`,
`passed = False`, NOT promotable. The headline carries
`(DSR | n_obs, MinTRL, n_obs ≥ MinTRL)` so a high point-DSR off a thin `n_obs` is
never read as more certain than the sample supports. Tests: (1) a config with
`n_obs < MinTRL` returns `insufficient_track_record` and does NOT pass; (2) the
§7.3b realistic worked example states `n_obs = 108` vs `MinTRL ≈ 1114` (a FAIL on
both DSR and track length), while the PASS contrast states `n_obs = 400` vs
`MinTRL ≈ 56`.

**Invariant DSR-MONO — the `N_sys` cluster count is monotone non-decreasing as
the correlation threshold tightens (Section 7.1).** For
`rho_cl_loose < rho_cl_tight`, `N_clusters(rho_cl_loose) ≤ N_clusters(rho_cl_tight)`,
so loosening the threshold can only SHRINK the penalty — forbidden as the
quiet-shrink path. The RED default `rho_cl` is TIGHT (biased toward more clusters
/ larger `N_sys`). An all-anti-correlated (or zero-correlated) ledger gives
`rho_bar` clipped at 0 ⇒ `N_eff = K` ⇒ `N_sys = K`. Tests: (1) sweeping `rho_cl`
loose→tight produces a monotonically non-decreasing `N_sys`; (2) an
all-anti-correlated ledger yields `N_sys = K`.

**Invariant DSR-ANN — PSR/DSR runs in PER-OBSERVATION units; annualization is
reporting-only (Section 7.2a).** The z-test is fed the per-observation Sharpe
`SR_pp = SR_annual / sqrt(ppy)`; EVERY `trial_sharpe` is put in per-observation
units before `Var[SR_trials]` is taken; and `var_floor = (1 + SR_pp²/2)/n_obs` is
the per-observation Lo-2002 variance, so `Var[SR_trials]`, `var_floor`, `SR*₀`,
and `SR_pp` are all commensurable. NO annualization factor enters the z-test — the
BLdP non-normality denominator terms are only correct at `f = 1`, so feeding an
annualized Sharpe inflates `z` by ≈`sqrt(ppy)` with no offset and produces a
spurious near-1 DSR. (DSR is NOT invariant to feeding different annualizations
INTO the test: for one fixed strategy `z` rises ~linearly with the factor; the
correct invariant practice is to keep annualization OUT of the test entirely by
running per-observation.) Tests: (1) feeding the ANNUALIZED Sharpe into the z-test
instead of `SR_pp` inflates `z` by ≈`sqrt(ppy)` and is rejected
(`annualization_mismatch`); (2) any Sharpe entering the z-test that is not in
per-observation units, or a `ppy` not derived from the `n_obs` bucketing, is
rejected by the internal consistency check.

**Invariant SG — exhaustive trial ledger.** Every BLUE candidate scored against
any validation fold appears in the trial ledger; `N_sys` is derived from the
ledger, not asserted. Test: ledger length == number of distinct `config_hash`
scored.

**Invariant BL — baselines share the identical evaluation contract.** Every
baseline and ablation uses the same bars, costs, folds, embargo, label function,
and execution stub as the full system. Test: report shapes and split objects are
identical across baseline and full-system runs.

**Invariant CA — combiner-only configs re-score from cache.** A SystemConfig
that changes only a composition/combiner parameter produces the same module
evidence cache keys and re-runs zero module computation. Test: cache hit-rate is
100% on module evidence when only a combiner field changed.

**Invariant CAD — cadence/deadband suppresses jitter.** A target change smaller
than the deadband `δ` generates no order; recomposes are event-driven or on the
selection cadence `τ_sel`, never per-1-minute-bar by default. Test: identical
inputs across consecutive bars produce no actions.

**Invariant PAR — batch/live parity.** The composed pipeline produces identical
SystemDecisions in batch replay and one-bar streaming mode (tolerance-equal),
relying on Nautilus's engine-level parity. Test: replay vs streaming on the same
bars match.

**Invariant P1 — no reimplementation.** `compose/` and `modules/` contain no
sizing, exit, or entry math — only orchestration, set ops, and `min()`/priority.
Test: a grep/AST check that the module internals are imported, not duplicated.

**Invariant C — RED unreachable from per-instrument config** (inherited).

**Invariant A — edge/risk firewall** (inherited from LEVERAGE.md): no edge
quantity enters sizing or the budget-priority contest. The selection RANK is a
diversification/quality measure for the candidate set, not an edge input to size
or priority.

**Invariant HC — contractual horizon cap bounds every label** (Section 6.1). The
triple-barrier vertical = `H_sys`; TREND's `tau_cap = H_sys`; no realized hold
exceeds `H_sys`; the embargo uses `H_sys`, never a mean/typical horizon.

**Invariant XC — execution stub is a conservative cost upper bound** (Section
6.1b). The stub over-charges cost (full spread, adverse slippage, scaled impact,
full funding, taker fees) so labeled net P&L and DSR are biased LOW; it is
calibrated against realized fills once live and re-tightened if it ever
under-charges.

**Invariant SR-SRC — DSR inputs come from the pinned series** (Section 7.2).
`Var[SR_trials]` is the cross-CONFIG ledger variance (never cross-CPCV-path);
`n_obs` is the non-overlapping system-observation count (never bar/path/trade
count). Test: feeding the cross-path variance or the bar count is rejected by an
input-provenance check, and the two errors are exercised in the synthetic suite.

**Invariant RP — repeated searches accumulate; swaps pass a forward gate**
(Section 7.5). Every overnight search appends its configs to the persistent
ledger; `N_sys` is over the full accumulated ledger; a live swap occurs only on
`τ_promote`, only for a candidate clearing the held-out forward gate AND the
accumulated-ledger DSR.

**Invariant ATTR — attribution mode is honest** (Section 8.2). `mode='shapley'`
requires all 2⁴=16 coalition baselines with the defined off-semantics; with fewer
the system uses `mode='sequential'` (fixed order, order-dependent) and labels the
report accordingly. Claiming Shapley / order-independence from <16 coalitions is
an error.

**Invariant CB-BUDGET — the overnight job has a pre-declared hard cap** (Section
9.2). RED fixes `(n_groups, k_test, max_paths)`, `K_max`, and `T_night` so the
per-stage cost model is `≤ T_night` on the target VM; the job refuses `>K_max`
configs and aborts to the last promoted config rather than overrunning into the
trading day.

---

## 15. Module acceptance gates (CI)

### Composition / precedence
Pass when: the pipeline A→G is deterministic; precedence order matches RED;
TREND exit overrides selection; ceiling collapse reduces incumbents; governor
scales all; `hold_incumbent` fires for a protected position (Invariant HI);
breakout score scaling leaves decisions unchanged (Invariant CB).

### Fixed point (selection↔leverage)
Pass when: Φ converges on benign inputs within `r_max`; incumbents reserved
first; new-capital weights only scale down; lot rounding only down; book heat +
dual equity cap always satisfied on the emitted book; non-convergence yields a
feasible fallback (Invariant IT); deterministic tie-break.

### Budget priority
Pass when: admission order is ceiling-slack → MCTR-efficiency → id; no
edge/score/expected-return input changes admission (Invariant A); deferred
candidates flagged, not dropped.

### Causality / embargo
Pass when: the embargo is `max(H_sys, max module label horizon) + max warmup`
(Invariant HC) — union, keyed to the contractual cap, never a mean/typical
horizon; TREND's `tau_cap = H_sys` so no realized hold exceeds `H_sys`; purge
removes overlapping training labels on both sides; no cross-module fold leakage
(Invariant SC); live packages do not import `validate/` (Invariant FW).

### Labels + execution stub
Pass when: system triple-barrier labels resolve first-touch correctly, vertical =
`H_sys`, net of the execution-stub cost, using only future bars inside
`validate/labels.py`; the execution stub is a conservative cost UPPER bound (full
spread + adverse slippage + scaled impact + full funding + taker fees), so labels
and DSR are biased low (Invariant XC); the stub cost ≥ a mid/half-spread reference
on every synthetic fill, and the realized-fill calibration test is wired.

### Walk-forward + CPCV
Pass when: every fit/threshold/rank/covariance is fold-local; sampled CPCV is
deterministic given a seed (recorded in metadata); the path-Sharpe distribution
is produced; combiner-only configs re-score from cache (Invariant CA); the
overnight job respects the `K_max` / `T_night` hard cap and records measured
wall-clock vs the §9.2 cost estimate (Invariant CB-BUDGET).

### Deflated Sharpe (centerpiece)
Pass when: `N_sys` is the BLdP **effective** independent-trial count among scored
ledger configs (`1 ≤ N_sys ≤ K`), **never** the Cartesian product (which appears
only as `n_sys_nominal` telemetry); `Var[SR_trials]` is the cross-CONFIG ledger
variance over the SAME `K` configs (consistency law) and is floored at
`var_floor`; `n_obs` is the non-overlapping system-observation count (NOT
bar/path/trade), tested against the swing-failure cases (Invariants DSR, DSR-GUARD,
SR-SRC); the `N_sys=1` case returns `SR*₀=0, DSR=PSR(0)`; a ledger with
`< K_min` distinct trials returns `insufficient_trials` and does NOT pass; a config
with `n_obs < max(MinTRL, n_obs_floor)` returns `insufficient_track_record` and
does NOT pass even at high point-DSR (Invariant DSR-MINTRL); `N_sys` is monotone
non-decreasing as `rho_cl` tightens and equals `K` for an anti-correlated ledger
(Invariant DSR-MONO); PSR/DSR is fed PER-OBSERVATION Sharpes (`SR_pp = SR_annual /
sqrt(ppy)`) so no annualization factor enters the z-test, and feeding an
annualized Sharpe or a `ppy`-mismatched factor is rejected (`annualization_mismatch`,
Invariant DSR-ANN); PSR uses skew/kurtosis with a floored denominator; the
headline is `(DSR | n_obs vs MinTRL)`; a luck/random strategy and an undeflated
config FAIL the gate; the worked example of §7.3b reproduces (the realistic case
FAILS: `DSR≈0.695`, `n_obs=108` vs `MinTRL≈1114`; the PASS contrast: `DSR≈0.999995`,
`n_obs=400` vs `MinTRL≈56`).

### Re-promotion / staleness
Pass when: every overnight search appends scored configs to the persistent ledger
and `N_sys` is computed over the accumulated ledger; a live-config swap occurs
only on `τ_promote`, only after a candidate clears the held-out forward gate and
the accumulated-ledger DSR (Invariant RP); the forward gate is a PAIRED
block-bootstrap test on the per-bucket candidate-minus-incumbent return difference
over the identical forward block (CI must EXCLUDE ZERO and the lower bound clear
round-trip cost; §8.1 dependence-preserving block length, never iid; never a bare
point margin); a candidate winning on a point margin but with a paired-difference
CI that includes zero does NOT swap; a forward block too thin for `n_obs ≥
max(MinTRL, n_obs_floor)` defers with `insufficient_forward_track`; simulating
`n` nightly re-searches monotonically raises `SR*₀`.

### Robustness + attribution
Pass when: drop-top-N and drop-best-instrument/fold/regime are reported;
block-bootstrap CIs use a dependence-preserving block length (no iid bootstrap);
regime/fold attribution is produced; attribution is honest (Invariant ATTR) —
`mode='shapley'` only when all 16 coalition baselines (with the §8.2 off-semantics)
are provisioned, else `mode='sequential_fixed_order'` with the order stated and
order-dependence acknowledged; a zero-contribution pillar is flagged.

### Baselines + ablations
Pass when: buy&hold, random-matched-frequency, and the single-pillar ablations
(breakout-only, no-selection, no-trend, governor-off, no-leverage-ceilings) — and,
for Shapley, all 16 coalition baselines — run under the identical contract
(Invariant BL) with the defined pillar-off semantics (§8.2); the full system beats
random-matched on DSR; governor-off shows worse drawdown; no-leverage-ceilings
shows fatter ES.

### Nautilus glue + parity
Pass when: SYSTEM reads book/account/position from Nautilus and emits deltas
(never raw orders); batch replay and streaming produce tolerance-equal
SystemDecisions (Invariant PAR); the execution stub is used only in validation.

---

## 16. Build order

Each step is complete only when its acceptance gate (Section 15) is green. A step
that fails its gate does not advance.

1. `types.py` and `synth/generate.py` (scripted module outputs + adversarial
   composed scenarios).
2. `config/` with TEAL/AMBER/RED/BLUE and `resolve.py`; test Invariant C + SG
   scaffolding (the trial ledger shape).
3. `modules/` thin adapters (import the four real modules behind their public
   contracts; no reimplementation — Invariant P1). Mockable for tests.
4. `compose/cadence.py` (decision-bar, deadband, event-recompose; Invariant CAD).
5. `compose/incumbent.py` (hold-incumbent firewall + rotation test; Invariant HI).
6. `compose/budget.py` (residual-budget priority; Invariant A).
7. `compose/iterate.py` (the damped fixed point Φ; Invariant IT).
8. `compose/precedence.py` + `compose/orchestrator.py` (pipeline A→G;
   Invariants CB, P1) and `system_api.py`.
9. `nautilus/adapter.py` + `nautilus/exec_stub.py` (CONSERVATIVE cost upper bound;
   Invariant XC); test batch/live parity (Invariant PAR).
10. `validate/cache.py` (EvidenceCache; Invariant CA) and `validate/trials.py`
    (the PERSISTENT ledger → effective `N_sys`; Invariants SG, RP).
11. `validate/labels.py` (system triple-barrier, vertical = `H_sys`; Invariants
    FW, HC).
12. `validate/walkforward.py` (purged, union embargo keyed to `H_sys`;
    Invariants SC, HC).
13. `validate/cpcv.py` (sampled CPCV; path-Sharpe distribution).
14. `validate/deflated.py` (PSR/DSR over EFFECTIVE `N_sys` + singularity guards +
    the MinTRL track-length floor + one-`ppy` annualization consistency +
    threshold-monotone `N_sys`; Invariants DSR, DSR-GUARD, DSR-MINTRL, DSR-MONO,
    DSR-ANN, SR-SRC) — the centerpiece.
15. `validate/baselines.py` (buy&hold, random-matched, single-pillar ablations,
    and the 16 coalition off-semantics; Invariant BL).
16. `validate/robustness.py` (drop-top-N, block-bootstrap CIs, regime/fold attrib).
17. `validate/attribution.py` (Shapley over 16 coalitions OR sequential 5
    knock-ins; Invariant ATTR).
18. `validate/budget.py` (overnight cost model + `K_max`/`T_night` cap;
    Invariant CB-BUDGET) and `validate/promotion.py` (re-promotion cadence +
    held-out forward gate; Invariant RP).
19. `validate/reports.py`.

Stop after step 19. The remaining questions — does the composed chain beat
random-matched on DSR, which pillar actually carries the edge, whether the edge
survives drop-top-N, whether one regime drives it — are answered by RUNNING the
integrated harness on real data overnight, not by adding composition machinery.

---

## 17. References and rationale

The implementation must be self-contained and test-driven. These references
motivate the design:

- **Probabilistic & Deflated Sharpe:** Bailey & López de Prado, *The Sharpe Ratio
  Efficient Frontier* (PSR) and *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality* — the `SR*₀` luck-max
  benchmark, the `Var[SR_trials]`/`N` **consistency requirement** (both estimated
  over the same scored trials, Section 7.1), the **effective-trials** count, and
  the **`N=1` ⇒ no-deflation** convention (Section 7.3a) are taken from here. This
  is the centerpiece math (Section 7).
- **Effective number of trials / correlated-trial reduction:** the Kish-style
  effective sample size `N_eff = K / (1 + (K−1)·ρ̄)` and the de-correlated-cluster
  count are how Section 7.1 turns `K` scored, correlated configs into an
  *independent*-trial count `N_sys ≤ K` consistent with the estimated variance —
  never the fabricated Cartesian product.
- **Purged / embargoed CV and Combinatorial Purged CV:** López de Prado,
  *Advances in Financial Machine Learning* (purging, embargo, CPCV, triple-barrier
  labeling). The union embargo and system-level triple barrier (Section 6) follow
  from labels that overlap in time across positions.
- **Backtest overfitting & the multiple-testing tax:** Harvey, Liu & Zhu,
  *…and the Cross-Section of Expected Returns* (deflating t-stats for the number
  of trials); Bailey, Borwein, López de Prado & Zhu, *Pseudo-Mathematics and
  Financial Charlatanism* (the minimum backtest length / minimum track-record).
  The union-of-DOF count `N_sys` (P6) is the composed-system analogue; the
  Minimum Track-Record Length is **operationalized** as the `MinTRL` track-length
  floor (Section 7.3a guard 3, Invariant DSR-MINTRL) — `n_obs ≥ max(MinTRL,
  n_obs_floor)` is a hard gate on the orthogonal track-LENGTH axis, so a high
  point-DSR off a thin `n_obs` is not read as a clean PASS without its confidence
  qualifier (in the §7.3b realistic case the thin `n_obs=108` falls far below
  `MinTRL≈1114`, so the candidate fails the gate outright).
- **Shapley attribution:** Shapley value over the four pillars gives an
  order-independent marginal contribution; sequential ablation is the
  fixed-order special case. (Section 8.2.)
- **Block bootstrap:** Künsch / Politis & Romano (moving/stationary block
  bootstrap) — preserves volatility clustering and serial dependence so the CIs
  are honest; iid bootstrap is forbidden for dependent returns.
- **Marginal contribution to risk / risk parity:** standard decomposition
  `MCTR_i = w_i(Σw)_i / sqrt(wᵀΣw)` (inherited from LEVERAGE.md, Section 7), used
  here only to allocate the shared budget — never `Σ⁻¹μ`.
- **Instrument selection / diversification:** Carver, *Systematic Trading* /
  *Advanced Futures Trading Strategies* (Ch. 25 instrument-selection logic) — the
  selection module ranks for risk diversification and quality, and that rank never
  drives sizing (Invariant A).
- **Trend convexity & the cost of turnover:** Fung & Hsieh (trend as a
  lookback-straddle); the hold-incumbent firewall (P4, Section 4) exists because
  rotating out of a live winner clips the convex right tail and pays round-trip
  cost.
- **Fixed-point / damped projection:** the selection↔leverage loop is a damped
  projection onto a convex feasible set (budget simplex ∩ box ceilings); damping
  guarantees the contraction (Section 3.3).

---

## 18. Non-goals (belong to other modules / the engine)

- **Module internals.** Entry detection (BREAKOUT.md), sizing/ceilings
  (LEVERAGE.md), exit/trend-life (TREND.md), and instrument selection (the
  Carver Ch.25 module) are COMPOSED, never reimplemented here.
- **Execution tactics.** Order slicing, routing, smart order types, and
  microstructure live in EXECUTION.md. SYSTEM uses only a thin cost+fill stub for
  validation parity.
- **The engine.** Order plumbing, position/account/portfolio state, restart
  reconciliation, data transport, backtest↔live parity, pre-trade risk checks,
  and funding ingestion are NautilusTrader's. SYSTEM builds policy on top and
  reads/writes only through the Nautilus adapter.
- **Any edge/alpha estimation, any learned combiner.** SYSTEM holds no model
  parameters tunable for return — only composition policy and the validation that
  proves the composed chain has an edge net of cost and net of the joint
  multiple-testing tax.
