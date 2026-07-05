# PREREGISTRATION — carried verbatim from intraday@800d209 (2026-07-04)

> The three frozen Phase-C sections below (EDGE pre-registration, SIGNAL candidate menu,
> SYSTEM selection protocol) are the anti-overfit contract this package was built against.
> They were REGISTERED FOR NIFTY WEEKLY OPTIONS. Per their own §C.7 rule, the planned
> COMMODITY/METAL-futures study is a NEW study and MUST freeze a NEW dated pre-registration
> (instruments, candidate ledger, metric, adoption threshold) BEFORE any run — this file is
> the template and the discipline, not a license to run.

---

## Phase C pre-registration (2026-07-04, BEFORE any run)

**Pre-registered:** 2026-07-04, ahead of any Phase-C model-selection run. This section timestamps
the exact candidate set, the selection protocol, and the adoption threshold **before any data is
touched**, so the anti-snoop discipline is auditable: the trials committed here are the only ones
whose out-performance can later be claimed, and the multiplicity penalty is counted against this
frozen set (fewer pre-registered trials ⇒ smaller multiplicity tax — the methodologically stronger
choice, so the candidate set below is deliberately a **focused POC subset** of the strongest formal
representative per category, with the long tail listed as explicit **deferrals**, not built).

**Incumbent to beat.** The benchmark every candidate must out-perform OUT-OF-SAMPLE is the validated
record above: ORB strong-close entry (SIGNAL §3) + fixed 5-trading-day hold + −12% true 1-min
premium stop (SIGNAL §5, `moneyness=0.014` ≈ δ0.65, `hold=5`, `stop=0.12`, `spread=0.01`). A
candidate that does not beat this incumbent on the held-out slice is **not adopted**; the negative is
recorded here and the incumbent stays. This is a candidate, not a confirmed edge (no vehicle survives
drop-top-3 on a realistic window — see above); Phase C is a search for a *better* entry/hold model,
not a re-opening of the settled long-vol question.

### The pre-registered candidate set (frozen)

PhD-level FORMAL models only (NOT technical indicators). One strongest formal representative per
category is admitted; siblings/variants are deferred. Every candidate is scored ONLY in its causal
**FILTERED / one-step-ahead** form (see PURITY below).

**ENTRY (structural-break / detection / regime — replaces or gates the ORB strong-close trigger):**
1. **Change-point / structural-break** — the online/sequential representative admitted: **Bayesian
   Online Change-Point Detection (BOCPD)** on the intraday return/price stream (run-length posterior).
   **Same-bar purity (binding, closes the one-bar-lookahead hole).** The BOCPD run-length posterior
   `P(r_t | F_t)` is a *filtered* estimate that **conditions on bar `t`**; a decision that reads it
   and *transacts on bar `t`* is a one-bar look-ahead. Phase-C therefore fixes the **decision-to-
   execution offset explicitly**: the changepoint posterior available at the close of bar `t`
   (`F_t`-measurable) may only trigger an entry **executed on bar `t+1`** — i.e. the decision that
   *fires* the trade is `F_{t-1}`-measurable relative to its own execution bar, exactly the SIGNAL
   §7/§8 and SYSTEM §C.2 no-lookahead standard. Equivalently: the signal used to enter at bar `s` is
   `P(r_{s-1} | F_{s-1})`, never `P(r_s | F_s)`. A pinning test (SYSTEM §C.2) must FAIL any BOCPD
   wiring whose entry at `s` reads a posterior conditioned on bar `s` or later. Batch retrospective
   methods (Bai-Perron, e-divisive) and offline CUSUM are **deferred** (retrospective ⇒ lookahead
   unless run strictly expanding-window; keep the search small).
2. **Sequential detection** — **Wald SPRT / GLR** change detector on the drift sign (first-passage of
   the log-likelihood-ratio boundary = entry). One detector, pre-set boundaries from train.
3. **Regime-switching** — a **2-state Gaussian HMM** used ONLY on its **filtered** posterior
   `P(state_t | F_t)` (Hamilton filter, one-step-ahead) — never the smoothed posterior (smoothed =
   lookahead). Entry when the filtered high-drift-state probability crosses a train-frozen threshold.
4. **Lee-Mykland jump test** (RE-ADMITTED, directional jump-continuation) — enter in the sign of a
   detected jump; the test statistic is computed on the trailing window only (causal).

**EXIT / DYNAMIC HOLD (replaces the fixed 5-day horizon; the −12% stop is retained as the floor):**
5. **Optimal-stopping free-boundary** — the momentum / regime-switching-latent optimal-stopping exit
   (continuation-region boundary solved on train, applied one-step-ahead).
6. **OU / Leung-Li mean-reversion optimal exit** (RE-ADMITTED) — the Leung-Li closed-form optimal
   liquidation level under an OU latent, as a *dynamic hold* boundary (exit when the filtered state
   crosses the free boundary).
7. **Survival / hazard duration** — dynamic hold = reversal-hazard threshold crossing (a discrete-time
   hazard model of the reversal event; exit when the one-step-ahead hazard exceeds a train-frozen
   level).

Deferred exit-side siblings (listed as clean **deferrals**, NOT scored in the POC, NOT sub-models of
any admitted candidate): the **regime-switching reversal exit** and the **Bayesian change-point of
the reversal series**. To keep the trial ledger minimal they are held back entirely — they are **not
"folded into" candidate (3) or (1)**: candidates (3)/(1) carry **only** their own stated parameters
(their `k` and their `N_eff` contribution below reflect the entry/exit model as written, with **no
hidden reversal-exit parameters** smuggled in under an existing candidate's name). A deferred exit
sibling is **re-openable ONLY under a NEW dated pre-registration** (SYSTEM §C.7) — never admitted
mid-study "if a first pass motivates it" (that would be a **first-pass-snooped trial**, exactly what
pre-registration forbids). The prior first-pass result cannot license an off-ledger add; it can only
motivate a *fresh* pre-registration that pays its own multiplicity tax.

**VOL INPUT (a covariate / vol-scaling / adaptive-threshold input to the above — NOT a standalone
signal):**
8. **HAR-RV (Corsi)** (RE-ADMITTED, DIRECTIONAL use only) — realized-vol cascade as a vol-scaling /
   adaptive-threshold / covariate input to the entry & stop widths. Signed-jump / semivariance /
   HARQ variants are **deferred** (one HAR representative in the POC); the refuted **RV-IV spread
   signal stays BARRED** and is NOT a Phase-C candidate.

**Pre-registered building blocks: 4 entry + 3 exit/hold + 1 vol covariate. But the multiplicity `N`
is NOT the additive 8 — it is the COMPOSED scored-config space.** The additive "8" understates the
luck bar and must not be fed to DSR. Two composition facts drive this:
- **HAR-RV (candidate 8) is a covariate/toggle, not a standalone trial.** It is defined (below) as a
  vol-scaling / adaptive-threshold / covariate **applied to** the entry and stop widths of the other
  candidates. A covariate that composes with the other models **multiplies** the scored-config space
  (each entry/exit config is scored with HAR **on** and **off**), it does not add 1. Counting it as
  "+1" is category error.
- **Entry and exit compose (per SIGNAL §C: a candidate "swaps the ENTRY and/or the EXIT and/or adds
  the VOL covariate").** The protocol admits combinations, not one-axis-at-a-time only. So the true
  effective space is up to **~(4 entry) × (3 exit/hold) × {HAR on/off} ≈ 24 composed configs**, plus
  each config's own within-model parameter search — NOT 8.

**What the DSR `N_eff` MUST be fed.** `n_trials_eff` (SYSTEM §3.3, §C.3) counts the **composed,
correlation-deflated** space: the entry×exit×{HAR} configuration grid AND each config's within-model
parameter search, **deflated** for the correlation among nearby settings and among families that
share machinery (use the effective-independent count, not the raw ~24×grid cardinality). It is
**never** the additive 8 — feeding the additive count would understate `SR*_0` and is precisely the
FAIL→PASS flip the gate exists to prevent (SYSTEM §C.3). Because the incumbent is scored one-axis-at-
a-time against each candidate (matched comparison, see the selection protocol and adoption threshold
below), the *reported* comparisons are pairwise, but the *multiplicity tax* is charged on the full
composed space that was searchable, not on the subset that happened to be reported. `N_eff ≥ 2`
always here (the shipped `N==1` base case of SYSTEM §3.3 does NOT apply to a multi-family study).

### The selection protocol (from SYSTEM — frozen)

- **PURITY (admissibility filter).** Only **causal FILTERED / one-step-ahead `F_{t-1}`-measurable**
  estimates are admissible; any smoothed / full-sample / retrospective estimate is a lookahead and is
  inadmissible. Parameters are **fit on train and FROZEN for OOS** — no re-fit on TEST, no threshold
  re-tuning on TEST.
- **Split + embargo + walk-forward (SYSTEM §2.1, §3.1, §C.2).** The single pre-registered held-out
  boundary **`SPLIT = 2025-06-01`** (frozen, never tuned) is the headline gate; partition by trade
  `exit_date`; **embargo = `cfg.hold` = 5 lake session days** on the train side of *every* train↔OOS
  boundary so train/test stay information-disjoint. Selection reads the **held-out OOS slice ONLY**.
  For a multi-family *selection* study (unlike the incumbent-only report), SYSTEM §C.2 re-admits an
  **anchored walk-forward** overlay — an expanding train window, forward OOS window, at the **smallest
  fold count** the 3-year sample bears (few wide folds, not many thin ones) — to show the selection is
  not an artifact of one boundary date; parameters are re-fit and frozen per fold. CPCV (many-fold
  combinatorial-purged CV) stays deferred; the CSCV combinations used by PBO (below) are formed over
  the *candidate performance matrix*, not by proliferating time-folds.
- **Sign/scale discipline (SYSTEM §4).** Every multiplicity / bootstrap test consumes the
  **per-observation** Sharpe `SR_pp = a.mean()/a.std()` (ddof=0) with `n_obs = len(trades)`; the
  annualized display Sharpe is NEVER fed into DSR/PSR/MinTRL/bootstrap (the annualization-inflation
  lesson — feeding the annualized figure flips a marginal edge past the gate).
- **Multiplicity gate (SYSTEM §3.3, §C.3, §8).** `deflated_sharpe(returns, n_trials_eff)` —
  PSR/DSR/MinTRL (Bailey-López de Prado 2014), scipy-free, `V>0` and `SR_pp>SR*_0` degeneracy-guarded.
  **DSR pass threshold `0.95`**, **`Z_gate = Φ⁻¹(0.95) ≈ 1.6449`**, **`n_obs` floor `= 24`**.
  `n_trials_eff` counts the **COMPOSED, correlation-deflated scored-config space** — the
  entry×exit×{HAR on/off} configuration grid (~24, per the composition note above) **AND** each
  config's within-model parameter search — deflated for correlation among nearby settings and among
  families that share machinery. It is **NOT** the additive candidate count (8) and **NOT** a raw grid
  cardinality: the effective-independent count. `N_eff ≥ 2` always (the SYSTEM §3.3 `N==1` base case
  is inapplicable to this multi-family study). **PBO / CSCV** — for this multi-family study the
  candidate ledger is large enough to form CSCV combinations, so PBO is computed over the pooled
  candidate performance matrix and reported alongside DSR (SYSTEM §C.3); a candidate whose in-sample
  rank does not survive the combinatorial train/test swap is rejected regardless of its point OOS
  number. (This is the one place Phase-C re-admits the SYSTEM §6 PBO deferral, which held only for the
  incumbent-only single-family report.)
- **Winner-robustness (SIGNAL §11, SYSTEM §2.2).** Drop-top-N (top-1 / top-3) EV and top-N PnL share
  reported on every candidate; a candidate whose OOS out-performance evaporates on drop-top-3 is not
  robust. This is the PRIMARY robustness statistic for this convex, fat-tailed family.
- **Bootstrap CI (SYSTEM §3.2, §8).** Paired trade-level bootstrap, **`B = 20000`, `block = 1`**,
  seeded/reproducible, on matched ORB trade dates vs the incumbent — the Sharpe-difference CI must
  clear zero, not merely have a favorable point estimate.
- **Cross-variant consistency (the anti-snooping mechanism).** A filter/model that helps one variant
  but not its siblings is a snooping trap (the consec-loss trap this check already caught). A
  candidate must out-perform **consistently across its variants / instruments**, never on a single
  cherry-picked cut.
- **Parsimony / information-criterion penalty (ENCODED in the objective, not just a tie-break).** The
  OOS objective is **IC-penalized** (AIC/BIC-style, SYSTEM §C.4): each candidate carries an explicit
  complexity count `k` (free parameters fit on train + latent-state dimension), and out-performance is
  measured on the **penalized** OOS objective (OOS Sharpe / EV **net of the `k`-penalty**), never on
  raw OOS fit alone. A richer model must beat the incumbent by a margin **exceeding** its added-
  complexity penalty — a candidate that only matches (or beats by less than its penalty) LOSES to the
  parsimonious one. The incumbent (fixed 5-day hold + −12% stop, `k` minimal) is the parsimony floor
  and holds on a tie. This penalty is **part of adoption clause 1 below**, not merely a tie-break.

### The adoption threshold (frozen)

A candidate is **ADOPTED** over the incumbent **iff ALL** of the following hold on the held-out OOS
slice, else the **negative is recorded here and the incumbent stays**:
1. **OOS out-performance, NET OF THE PARSIMONY/IC PENALTY** — strictly beats the incumbent on the
   held-out slice on the **IC-penalized** objective: net-of-cost EV/trade and per-trade Sharpe
   **penalized by the candidate's complexity count `k`** (AIC/BIC-style, SYSTEM §C.4), on **matched**
   trades / hold / instrument. A candidate that beats raw OOS Sharpe/EV by less than its added-
   complexity penalty **FAILS this clause** — the penalty is inside the gate, not just a downstream
   tie-break. Ties (equal OOS net of penalty) resolve to the incumbent (the parsimony floor).
2. **DSR / PBO pass** — `DSR ≥ 0.95` with `n_trials_eff` = the **composed, correlation-deflated
   scored-config space** (entry×exit×{HAR} grid + each config's within-model search, per §C.3 above —
   NOT the additive 8), and `n_obs ≥ 24`; **AND** PBO below its pre-registered ceiling (computed via
   CSCV over the pooled candidate matrix — this multi-family ledger is large enough to form the
   combinations, unlike the incumbent-only report where PBO was deferred).
3. **Drop-top-N robust** — the out-performance survives drop-top-1 AND drop-top-3 (does not rest on a
   one-to-three-trade lottery).
4. **Cross-variant consistency** — the out-performance holds across the candidate's variants /
   instruments, not on a single cut.
5. **Paired-bootstrap CI clears zero** — the Sharpe-difference vs the incumbent has a
   `B=20000, block=1` 90% CI that excludes zero.

Miss ANY one ⇒ **not adopted**; log the negative with the failing gate named. The incumbent is the
default and survives every tie.

### Prior EDGE negatives these re-admitted tools carry (CONTEXT — the study must re-confirm them)

The three re-admitted tools were previously refuted **in a different use** and are re-admitted ONLY
as DIRECTIONAL Phase-C candidates (owner un-ban 2026-07-04, recorded in CLAUDE.md's forbidden list as
a scoped exception). Their prior negatives are **context, not a bar** — and the Phase-C study **must
be able to RE-CONFIRM these negatives** (a candidate that reproduces the prior failure is the
expected default outcome, recorded as a negative, incumbent kept):
- **Lee-Mykland jump test** — directional **jump-continuation is dead** in the prior record (buying
  vol *after* a move pays inflated IV; the move is already priced in). Re-admitted here as a
  directional *entry* trigger; the study must show it either beats the incumbent OOS or re-confirms
  the dead-continuation negative.
- **OU / Leung-Li mean-reversion optimal exit** — **refuted in the long-vol context** (the OU /
  Leung-Li exit bled). Re-admitted here as a *dynamic-hold / exit boundary* on the directional book;
  must beat the fixed 5-day hold OOS or re-confirm the refutation.
- **HAR-RV (Corsi)** — **reactive in the RV-IV spread** (the vol-forecast-minus-implied signal was a
  dead end). Re-admitted here ONLY as a DIRECTIONAL vol-scaling / adaptive-threshold / covariate
  input (the **RV-IV spread signal stays BARRED**); must earn its place as a covariate OOS or
  re-confirm that it adds nothing.

STILL BARRED (NOT Phase-C candidates, unchanged): long-vol **straddle / strangle / ratio-backspread
STRUCTURES** and the **RV-IV spread signal**. The scoped un-ban touches only the three tools above,
only in their directional use.

### Empirical selection status: BLOCKED-ON-LAKE

The empirical selection RUN is **BLOCKED-ON-LAKE** and cannot proceed until a **wide-band /
position-following re-pull** exists. The 3-yr Breeze lake is currently ABSENT (deleted in the 2026-07
cleanup — see § Data assets above), and any re-pull MUST be wide-band so drifting winners' marks are
never delta-fabricated (the known lake bias / biggest open data task). While the lake is absent,
Phase C delivers ONLY (a) this pre-registration + (b) the candidate-model code + the SYSTEM selection
harness, synthetic-tested (known-generating-process recovery). **No backtest is fabricated, nothing
is adopted, no metric on this candidate set is reported** until the wide-band lake is in hand; the RUN
is a terminal BLOCKED-ON-LAKE state whose prerequisite is the re-pull. This section is the frozen
pre-registration against which that future run will be judged.


---

## Phase C — pre-registered candidate menu (2026-07-04)

**PRE-REGISTRATION ONLY — no model code exists yet** (that is a later step; the empirical selection
RUN is BLOCKED-ON-LAKE, `docs/EDGE.md` §Data assets). This section fixes, *before any data is
touched*, the FOCUSED family of PhD-level formal models that Phase C is permitted to try, each with
its generating-process assumptions, the parameter(s) that will be fit on the **train fold only**,
and the causal/filtered purity requirement. Nothing here mutates §1–§11 — the shipped ORB signal is
untouched. This is a menu of *candidate replacements/overlays* for two of the incumbent's three
moving parts (the ENTRY trigger and the EXIT/HOLD rule), plus one covariate input (vol).

**Why a small menu is the methodologically correct choice.** Every pre-registered candidate is a
genuinely-distinct trial that enters SYSTEM's `n_trials_eff` (§11 ↔ SYSTEM §3.3, §4) and raises the
DSR luck bar `SR*_0`. Fewer pre-registered trials ⇒ smaller multiplicity penalty ⇒ a lower bar for
any true edge to clear. The proportionate POC choice is therefore the *single strongest formal
representative per category*, with the long tail listed as explicit **deferrals** (§C.5), not a
detector zoo. This mirrors the BUILD_PLAN calibration guardrail (smallest correct contract) and
convexity's honest-deferrals pattern.

### C.0 The INCUMBENT (baseline — every candidate must BEAT this OOS)

The validated §3/§5 rule, expressed as the thing to beat:
- **ENTRY** = ORB range-break **+ strong close** (finish in the top/bottom 30% of the day's range),
  sign-antisymmetric, history-free (features from the signal day only).
- **HOLD/EXIT** = **fixed 5 trading-day** horizon (`hold=5`), true **−12% 1-minute premium stop**
  (`stop=0.12`, gap-through aware), winner floored to conservative intrinsic on a missing mark.
- **VOL** = none (no vol input enters the incumbent; it is price-structure only).

A candidate is **adopted only if it beats this incumbent on the held-out OOS under the full SYSTEM
protocol** (SYSTEM §C — anchored walk-forward + embargo, DSR/PBO multiplicity, parsimony, drop-top-N,
cross-variant consistency; enumerated as SIGNAL's adoption bar in §C.6). Absent that, the negative is
recorded in `docs/EDGE.md` and the incumbent stays.
Candidates compose with the incumbent's *unchanged* structure (§4 strike selection, §6 cost, §11
metrics) — a candidate swaps the ENTRY trigger and/or the EXIT rule and/or adds the VOL covariate;
it never re-prices, re-nets a cost, or introduces greeks/IV (forbidden, CLAUDE.md).

### C.1 Purity requirement (binding on EVERY candidate — the no-lookahead extension of §7)

- **Causal / one-step-ahead only.** Every statistic feeding an entry-or-exit *decision at day `t`*
  must be **`F_{t-1}`-measurable** (or, for a within-day trigger, measurable from information up to
  the decision instant), i.e. a **FILTERED** estimate `E[·|F_{t}]` or a one-step-ahead forecast
  `E[·|F_{t-1}]` — **never a SMOOTHED estimate** `E[·|F_T]` that conditions on the whole sample.
  Smoothed regime probabilities, full-sample Viterbi paths, two-sided HAR fits, and full-sample
  change-point segmentations are **lookahead by construction** and are inadmissible as live signals
  (they may be used only for descriptive/diagnostic plots, clearly labelled, never for a decision).
- **Train-fit, OOS-frozen.** Any parameter a candidate needs (thresholds, priors, transition
  matrices, HAR coefficients, free-boundary levels) is estimated on the **train fold only** and
  **frozen** for the held-out TEST slice (SYSTEM §2.1 `SPLIT = 2025-06-01`; §8). No parameter is
  tuned on TEST; no threshold grid-search (CLAUDE.md, `docs/EDGE.md` no-snoop discipline).
- **Pinning.** Each candidate, when it is later coded, MUST carry a test that FAILS under a 1-bar
  leak — the same standard as §7/§8 (a mutation that lets a decision at `t` read a bar `≥ t`, or
  that swaps a filtered estimate for a smoothed one, must break a test).
- **Warm-up honesty.** A candidate that needs history (HMM filter burn-in, HAR's 22-day lag, a
  change-point run-length prior) starts its scan only after its own warm-up is satisfied *within the
  train fold* — never borrowing pre-sample or cross-fold information.

### C.2 ENTRY candidates (formal — replace/gate the strong-close trigger)

Each takes the incumbent's per-day price path (and, where noted, the HAR-RV covariate §C.4) and
emits a **causal directional signal** in the same `{+1, 0, −1}` sign convention as §3, feeding the
*unchanged* §4 strike selection. The generating-process assumption is stated so the synthetic
recovery test (later step) can be built against a *known* DGP.

- **(a) Bayesian Online Change-Point Detection (Adams–MacKay 2007) on the drift.**
  *Assumption:* the intraday (or daily-close-to-close within the session) log-return has a
  piecewise-constant **drift** `μ` with i.i.d. Gaussian innovations; regime durations follow a
  hazard `H(r)` (constant-hazard `1/λ` prior). *Signal:* the **online** run-length posterior
  `p(r_t | x_{1:t})` collapses (mass jumps to `r=0`) when the drift breaks; enter in the **sign of
  the post-break filtered drift** `E[μ | most-probable run]`. Strictly forward-filtered — the BOCPD
  recursion is causal by construction (never the retrospective MAP segmentation). *Train-fit,
  frozen:* the hazard rate `λ` (expected regime length) and the Normal-inverse-Gamma prior
  `(μ_0, κ_0, α_0, β_0)`; a fixed run-length-posterior-mass entry threshold. *Rejects:* offline/batch
  change-point (that is the e-divisive/Bai–Perron deferral, §C.5).

- **(b) Wald SPRT / GLR sequential detection on the drift sign.**
  *(FROZEN-LEDGER NOTE — reconciled 2026-07-04. This slot is the SINGLE sequential-detection trial.
  It is named IDENTICALLY here and in the frozen Phase-C pre-registration `docs/EDGE.md` §Phase C
  pre-registration ("2. Sequential detection — Wald SPRT / GLR") — the two artifacts name the SAME
  representative, closing the anti-snoop hole that a divergent CUSUM-vs-SPRT naming would open, where
  whichever detector scored better could be claimed post-hoc as "the pre-registered one." CUSUM is
  the DEFERRED sequential sibling of this slot, §C.5 — deferred because one sequential representative
  suffices for the POC, NOT because CUSUM is retrospective: CUSUM is a fully **online, recursive,
  causal** detector, so the §C.5 deferral rationale is "one sequential representative is enough," not
  a lookahead concern.)*
  *Assumption:* a sequential test of `H_0: μ = 0` (no directional drift) vs the composite alternative
  `H_1: μ = ±δ` (a drift shift of unknown sign). The Wald **SPRT** accumulates the log-likelihood
  ratio `Λ^{±}_t = Λ^{±}_{t-1} + log[ f_{±δ}(x_t) / f_0(x_t) ]` for each drift-sign hypothesis; the
  **GLR** generalization replaces the fixed `±δ` with the running MLE of the post-change drift
  (Lorden/Siegmund window-limited GLR), which does not require pre-specifying the shift magnitude.
  *Signal:* the first arm whose statistic crosses its upper boundary `A` triggers entry in that arm's
  drift **sign** (`+1` for the `+δ` arm, `−1` for the `−δ` arm); a lower-boundary crossing resets
  to the no-signal continue region. *Train-fit, frozen:* the minimum-detectable drift `δ` (or, for
  GLR, the window length `m`) and the decision boundaries `(A, B)` set from the Wald approximations
  `A ≈ log((1−β)/α)`, `B ≈ log(β/(1−α))` to a **pre-registered false-alarm / miss pair `(α, β)`**
  (equivalently a target in-control Average Run Length) — set a priori from the error-probability
  design, NOT grid-searched for P&L (the no-snoop way to fix the boundaries). Purely recursive ⇒
  causal (each update reads only `x_t` and the prior accumulator). *Contrast with (a):* SPRT/GLR is a
  frequentist likelihood-ratio sequential test with fixed error-probability boundaries; BOCPD is a
  Bayesian run-length posterior — the two are distinct formal representatives, deliberately kept as
  the two canonical change-detection schools (one Bayesian-online, one sequential-frequentist).

- **(c) 2-state Hamilton (1989) HMM on the FILTERED trend-posterior.**
  *Assumption:* returns are drawn from a 2-state Markov-switching mean model (bull drift `μ_1>0` /
  bear drift `μ_2<0`, state-dependent variance), with transition matrix `P`. *Signal:* run the
  **Hamilton filter** to get the one-step **filtered** state probability `ξ_{t|t} = P(S_t | F_t)` (or
  the predicted `ξ_{t+1|t}`); enter `+1` when the filtered bull-probability crosses a pre-registered
  level, `−1` for bear. **FILTERED, never SMOOTHED** — the Kim smoother `P(S_t|F_T)` is explicitly
  barred as lookahead (this is the load-bearing purity clause for the HMM). *Train-fit, frozen:* the
  EM-estimated `(μ_1, μ_2, σ_1, σ_2, P)` and the probability entry threshold. *Deferred variant:*
  ≥3 states / duration-dependent (semi-Markov) HMM (§C.5).

- **(d) Lee–Mykland (2008) jump test — RE-ADMITTED as directional jump-continuation.**
  *Assumption:* the price has a continuous component plus rare jumps; the standardized statistic
  `L_t = |r_t| / \hatσ_t` (with `\hatσ_t` a **causal, backward-looking** bipower/realized-vol
  estimate on the *prior* window) exceeds a Gumbel-calibrated critical value on a jump. *Signal:*
  on a detected **significant jump**, enter in the **direction of the jump** (jump-continuation, the
  directional re-admission — NOT the refuted long-vol/straddle use). *Train-fit, frozen:* the
  local-vol window length and the jump significance level (Gumbel critical value at a pre-registered
  `α`). *Purity:* `\hatσ_t` uses only bars strictly before `t`; the jump decision is `F_{t-1}`- (or
  intra-bar-causal) measurable. *History note:* Lee–Mykland previously bled as a **long-vol** signal
  (`docs/EDGE.md`); the un-ban (2026-07-04) is scoped to the **directional continuation** use, and is
  protocol-gated — the prior negative is context, not a bar (CLAUDE.md amendment records the scope).

### C.3 EXIT / DYNAMIC-HOLD candidates (formal — replace the fixed 5-day horizon)

Each replaces the incumbent's fixed `hold=5` horizon with a **causal dynamic exit** while keeping
the true −12% 1-minute stop **unchanged** (the stop *is* the edge, `docs/EDGE.md` — a candidate may
only change *when a still-alive winner is voluntarily closed*, never loosen the loss cap). The
recorded-exit-date simplification (§5) is inherited; a dynamic exit records its *actual* decided
exit day (an explicit refinement over the incumbent's scheduled-horizon bucketing, to be stated in
SYSTEM/REGIME when coded).

- **(a) Optimal-stopping free-boundary under a regime-switching / momentum latent.**
  *Assumption:* the position's log-value follows a controlled diffusion whose drift depends on a
  latent momentum/regime state (couples to the §C.2c HMM or §C.2a BOCPD posterior); the exit is the
  solution of the **optimal-stopping** problem `sup_τ E[e^{-ρτ} G(V_τ)]`, i.e. hold while the
  continuation value exceeds the intrinsic exit value, stop at the **free boundary**. *Signal:*
  exit when the *filtered* state value crosses the pre-computed continuation/stopping boundary.
  *Train-fit, frozen:* the discount/impatience `ρ` and the latent-state dynamics (inherited from the
  entry model's train fit); the free boundary is solved once on train parameters and frozen. *Purity:*
  the boundary is a function of the **filtered** posterior only.

- **(b) OU / Leung–Li mean-reversion optimal exit — RE-ADMITTED (F/G free-boundary `b*`).**
  *Assumption:* the exit-relevant quantity (e.g. the option-value deviation, or a mean-reverting
  component of the underlying) follows an **Ornstein–Uhlenbeck** process
  `dX = θ(m − X)dt + σ dW`. *Signal:* the Leung–Li optimal *liquidation* level `b*` — the value that
  maximizes the expected discounted exit payoff, obtained from the smooth-fit `F`/`G` value-function
  ODE solution — sets the take-profit boundary; exit on the first `F_{t-1}`-measurable crossing of
  `b*`. *Train-fit, frozen:* the OU parameters `(θ, m, σ)` (estimated on train) and the discount rate;
  `b*` is then a deterministic frozen function of those. *History note:* the OU/Leung–Li exit
  previously bled (`docs/EDGE.md` dead-ends); re-admitted (2026-07-04) as a **directional exit**
  candidate, protocol-gated (CLAUDE.md amendment). *Contrast with (a):* (a) exits on a *momentum
  /trend-continuation* latent (hold winners while trending); (b) exits on a *mean-reversion* target
  (bank a winner at a reversion level) — deliberately kept as the two opposing exit theses so the
  cross-variant check (§C.6) can adjudicate which, if either, survives.

- **(c) Survival / hazard duration — dynamic hold = reversal-hazard threshold crossing.**
  *Assumption:* the trade's *time-to-adverse-reversal* is a duration with a **hazard function**
  `h(u | covariates)` (Cox proportional-hazards or a parametric Weibull/accelerated-failure-time
  form), covariates being `F_{t-1}`-measurable (elapsed holding time, filtered trend strength, the
  HAR-RV level §C.4). *Signal:* hold while the one-step-ahead **reversal hazard** is below a
  pre-registered threshold; exit when the hazard crosses it (the winner's continuation is no longer
  favored). *Train-fit, frozen:* the hazard-model coefficients (fit on train trade durations) and the
  exit hazard threshold. *Purity:* the hazard at holding-time `u` conditions only on information up to
  the current bar — a *predictive* hazard, never a full-sample fitted survival curve.

### C.4 VOL INPUT candidate (formal — a covariate, NOT a standalone signal)

- **HAR-RV (Corsi 2009) realized-volatility, one-step-ahead `F_{t-1}`-measurable — RE-ADMITTED.**
  *Assumption:* realized variance has long memory well-approximated by the **heterogeneous
  autoregressive** cascade `RV_{t} = c + β_d RV^{(d)}_{t-1} + β_w RV^{(w)}_{t-1} + β_m RV^{(m)}_{t-1}
  + ε_t` (daily/weekly/monthly averages of *past* RV). *Use — three admissible roles, all as an
  INPUT to the above entry/exit candidates, never as the trade signal itself:* (i) **vol-scaling**
  of position size or the equity curve; (ii) an **adaptive threshold** — normalize a SPRT/GLR (§C.2b)
  / BOCPD (§C.2a) / HMM (§C.2c) statistic by the HAR one-step-ahead `\hat{RV}_{t|t-1}` so the
  detector self-scales to the current
  vol regime; (iii) a **covariate** in the survival hazard (§C.3c) or the jump local-vol (§C.2d).
  *Train-fit, frozen:* the HAR coefficients `(c, β_d, β_w, β_m)` (OLS on train RV). *Purity:* the
  forecast `\hat{RV}_{t|t-1}` uses only lags `≤ t−1` — a genuine one-step-ahead forecast; the
  two-sided/full-sample HAR fit is barred (lookahead). *History note:* HAR-RV previously bled as a
  **vol-forecast trade signal** and, critically, as the **RV−IV spread** (`docs/EDGE.md`, CLAUDE.md).
  The un-ban (2026-07-04) is **strictly** as a directional vol-scaling / adaptive-threshold /
  covariate input. **The RV−IV spread signal STAYS BARRED** — it is not on this menu in any form.

### C.5 Explicit DEFERRALS (named formal models NOT implemented in the first study)

Stated decisions, not omissions (BUILD_PLAN calibration; convexity honest-deferrals pattern). Each
is a legitimate formal model deliberately held back to keep `n_trials_eff` small; admit one only by
amending this pre-registration *before* it is run:
- **Bai–Perron multiple structural breaks** (offline multiple-break least-squares) — superseded for
  the POC by the online BOCPD (§C.2a) and the Wald SPRT/GLR sequential detector (§C.2b); offline
  segmentation is also a lookahead hazard if misused as a signal (this is the genuinely retrospective
  concern — distinct from CUSUM, which is online/causal, deferred above only for parsimony).
- **e-divisive / ECP nonparametric multiple change-point** — offline, distribution-free; deferred
  for the same reason.
- **CUSUM (Page 1954), ARL-calibrated** — the sequential-frequentist sibling of the admitted Wald
  SPRT/GLR (§C.2b): the two-sided cumulative sum `S^{±}_t = max(0, S^{±}_{t-1} ± (x_t − k))` alarms
  when `S ≥ h`, with `k` = half the minimum-detectable drift and `h` calibrated to a pre-registered
  in-control ARL. It is a fully **online, recursive, causal** detector (not a retrospective/offline
  method — that distinction belongs to the Bai–Perron / e-divisive deferrals above); it is deferred
  ONLY because **one sequential representative (the SPRT/GLR of §C.2b) is enough for the POC** and a
  second sequential detector would add a near-duplicate trial to `n_trials_eff` for little
  independent information. Admit it only by amending this pre-registration before any run.
- **Bayesian change-point of the REVERSAL** (BOCPD applied to the exit side) — an exit-side analog of
  §C.2a; deferred in favor of the optimal-stopping/OU/hazard exit trio (§C.3).
- **Signed-jump / semivariance / HARQ variants** — realized-semivariance decomposition, signed-jump
  covariates, and the measurement-error-corrected HARQ; deferred vol-input elaborations of the plain
  HAR-RV (§C.4).
- (Also implicitly deferred, per §C.2/§C.3 notes: ≥3-state / semi-Markov HMM; regime-switching
  *reversal* as a standalone exit.)

### C.6 Adoption bar (the SYSTEM protocol — no candidate is adopted without it)

Pre-registered here so the decision rule is fixed before any run. **This bar IS the SYSTEM Phase-C
protocol (SYSTEM §C) — SIGNAL does not define an alternative; it enumerates the same conjunction the
SYSTEM selection RUN obeys, so the two artifacts describe one selection procedure, not two.** The
resampling scheme is therefore **SYSTEM §C.2's anchored walk-forward + embargo** (an expanding train
window, a forward OOS window, stepped forward; parameters re-fit and frozen per fold): the single
fixed held-out boundary `SPLIT = 2025-06-01` (SYSTEM §2.1) is the **headline** OOS gate, and the
walk-forward folds are the **robustness overlay** that shows the result is not an artifact of one
boundary date. (This supersedes an earlier reading of this section as single-split-only; it is
reconciled to SYSTEM §C.2, which mandates anchored walk-forward for *model selection* — as distinct
from the incumbent-only report, where multi-fold WF is a SYSTEM §6 deferral.) A candidate replaces
the incumbent **only if ALL hold on the HELD-OUT OOS**
(SYSTEM §C.2/§C.6, §2.1 `SPLIT`, §2.3, §3, §5; `docs/EDGE.md` standards):
1. **OOS out-performance** — the candidate's held-out TEST metrics (§11) beat the incumbent's on the
   *same* held-out slice (the `SPLIT` headline, SYSTEM §2.1) **and hold up across the anchored
   walk-forward folds** (SYSTEM §C.2), matched instrument/hold/cost (matched-comparison discipline,
   SYSTEM §2.3 / §C.5). *(The matched-comparison discipline is SYSTEM §2.3, not §2.1 — §2.1 is the
   held-out split; an earlier draft mis-cited §2.1 for the matched comparison.)*
2. **DSR / PBO multiplicity** — deflated Sharpe with **`n_trials_eff` counting every candidate on
   this menu AND each candidate's own within-model parameter search AND every entry×exit PAIRING that
   is actually scored** as genuinely-distinct trials (SYSTEM §3.3, §C.3). *Entry×exit cross-product:*
   because a dynamic-exit candidate (§C.3) couples to an entry latent (§C.3a to the §C.2c HMM or the
   §C.2a BOCPD posterior; §C.3c hazard to the filtered trend strength + HAR-RV §C.4), the SAME exit
   candidate scored against 2 entry latents is **2** distinct scored trials, not 1 — each such
   pairing enters `n_trials_eff` (deflated for the machinery the pairings share, per SYSTEM §C.3's
   correlation-deflation rule), so the entry×exit combinatorial cross-product is taxed, not free.
   Under-counting it would understate `SR*_0` and could flip a DSR FAIL to a spurious PASS — the exact
   multiplicity leak this gate exists to prevent. The per-observation Sharpe, never the annualized
   display figure (§11 ↔ SYSTEM §4). PBO/CSCV when the enumerated ledger is large enough (SYSTEM §C.3).
3. **Drop-top-N robustness** — the candidate's edge must survive drop-top-1/top-3 on a realistic
   window (§11; the fat-tail tell — *no vehicle in the record survives this yet*, `docs/EDGE.md`).
4. **Cross-variant consistency** — a candidate that helps but whose sibling variants (the other menu
   members, its own parameter neighbours, **and its own walk-forward folds**, SYSTEM §C.2/§C.5) do
   NOT is a **snooping trap**, not an edge (the consec-loss-trap lesson, `docs/EDGE.md` / SYSTEM §2.3).
   Consistency across siblings AND across folds is the anti-snooping check, never a P&L threshold grid.

If any gate fails, the **negative is recorded in `docs/EDGE.md`** and the **incumbent stays**
(§C.0). The empirical selection RUN is **BLOCKED-ON-LAKE** until the wide-band / position-following
re-pull exists (§10.4; `docs/EDGE.md` §Data assets) — this menu is the pre-registration that must be
frozen *before* that data is touched.


---

## Phase C — model-selection protocol (2026-07-04)

**PRE-REGISTRATION ONLY.** This section fixes — *before any candidate is scored* — the exact
protocol, the **enumerated 8-trial candidate ledger (§C.9)**, effective-trial accounting, penalties,
and single adoption conjunction by which a Phase-C entry/hold model may replace the incumbent. It adds
NO model code and reports NO result; it is the binding contract the (future, lake-gated) selection RUN
must obey. It sits ATOP §1–§8: it reuses `split`/`bootstrap_ci`/`deflated_sharpe` (§3) and the
annualization-inflation invariant (§4) verbatim, and it inherits the SIGNAL `(exit_date, net_return)`
unit unchanged (§1, §7). The scope is a POC: a **focused set of the strongest formal representatives
per candidate category** (enumerated in §C.9, identical to `docs/EDGE.md`'s frozen set and SIGNAL
§C.2–§C.4), with the long tail listed as explicit **deferrals (§C.7)** — the fewer pre-registered
trials, the smaller the multiplicity penalty, which is also the methodologically honest choice.

### C.1 The incumbent and what "select" means

- **INCUMBENT** = the validated candidate frozen in `docs/EDGE.md` / SIGNAL: strong-close ORB entry
  + fixed **5-day hold** + **−12% true 1-min premium stop**, on the deep-ish ITM (~δ0.65) option.
  This is the reigning model; it is NOT re-selected and NOT re-tuned. Every Phase-C candidate is
  measured **against it**, on the **same paired ORB trade dates** wherever the candidate only swaps
  the exit/hold or the entry timing (matched comparison, §C.5).
- **"Select" is decided on the HELD-OUT OOS SLICE ONLY.** No candidate is ranked, tuned, or adopted
  on any statistic computed on the training folds. Training folds exist solely to FIT each model's
  parameters; those parameters are then **frozen** and the candidate is scored once on OOS. Any
  procedure that consults an OOS number to pick a parameter, a fold, or a candidate collapses the
  held-out guarantee and is a protocol violation (a pinning test must catch a candidate that peeks).

### C.2 Walk-forward + embargo (the resampling scheme)

- The selection RUN uses **anchored walk-forward** over the lake calendar: an expanding train window,
  a forward OOS window, stepped forward; parameters are re-fit on each train window and frozen before
  its OOS window is scored. This *generalizes* the single frozen `SPLIT` (§2.1, §3.1) — the single
  held-out split remains the headline gate; walk-forward is the robustness overlay that shows the
  selection is not an artifact of one boundary date. (Rolling/anchored WF was a §6 deferral for the
  incumbent-only report; Phase-C model *selection* is exactly the case that re-admits it, at the
  smallest fold count the 3-year sample can bear — few wide folds, not many thin ones.)
- **Embargo** (§3.1) is applied at **every** train↔OOS boundary: drop from each train window any
  trade whose scheduled `exit_date` lands within `embargo_days = cfg.hold` **lake session days**
  before that fold's boundary, so a trade open across the boundary cannot leak train↔OOS. Embargo
  width is counted in session days, never calendar days (§3.1). Because SIGNAL records the *scheduled*
  horizon as `exit_date` (§2.1, SIGNAL §5), the embargo uses that same convention — stated, not
  silently re-defined.
- **Purity — filtered/one-step-ahead only (two admissible cases, precisely distinguished — SIGNAL §7,
  §C.4).** Every candidate's state estimate at decision time must be causal FILTERED / one-step-ahead,
  and the measurability standard is **fixed by the decision cadence** — do NOT conflate the two:
  - **A daily / end-of-day decision at day `t` must be `F_{t-1}`-measurable** — it reads only bars
    strictly *before* `t`. In particular a **same-bar `RV_t` / same-bar filtered estimate `E[·|F_t]`
    is BARRED** for the daily-decision candidates (a classic same-bar lookahead): the HAR-RV covariate
    (§C.4 / EDGE ledger #8) enters a daily decision only as the one-step-ahead forecast
    `\hat{RV}_{t|t-1}` built from lags `<= t-1` (matching SIGNAL §C.4's "HAR uses lags `<= t-1`"), never
    as `RV_t`.
  - **A within-day trigger** (a candidate that decides *at* an intra-session instant, mirroring the
    incumbent's transact-at-the-close convention, SIGNAL §7) may use information **up to that decision
    instant** — i.e. it is measurable w.r.t. the sigma-field at the decision instant, which for a
    same-bar within-day trigger legitimately includes the partial bar observed up to that instant, but
    **never any bar dated after the decision instant.**
  - **Smoothed** (full-sample, two-sided) estimates are LOOKAHEAD and are inadmissible in BOTH cases —
    an HMM/BOCPD posterior must be the *filtered* `E[·|F_t]`/one-step `E[·|F_{t-1}]` posterior, never
    the Kim-smoothed / full-sample `E[·|F_T]`; an OU/HAR fit uses only information available at the
    decision cadence's cutoff (`<= t-1` for a daily decision), never the whole sample.
  - **Pinning (matches the cadence — closes the same-bar hole).** For a **daily-decision** candidate a
    test must FAIL any decision at `t` that reads a bar `>= t` (INCLUDING the same bar `t` — a same-bar
    RV/filtered read is a leak here). For a **within-day-trigger** candidate a test must FAIL any
    decision that reads a bar dated *after* the decision instant. Either way, swapping a filtered
    estimate for a smoothed one must break a test (mirrors SIGNAL §7/§8 no-lookahead).

### C.3 The multiplicity tax — effective trials count EVERYTHING searched

The DSR/PBO gate (§3.3) is fed an **effective trial count `N_eff`** that counts, jointly:

1. **the parameter search WITHIN each candidate model** — every distinct hyper-parameter setting
   (e.g. each BOCPD hazard/threshold node, each SPRT/GLR boundary, each HMM probability threshold, each
   OU-boundary grid node, each HAR lag window, each hazard-threshold — the per-candidate grids
   enumerated in §C.9) that is *scored*, deflated by the correlation among those settings (nearby
   settings are not independent; use the effective-independent count, not the raw grid cardinality),
   **AND**
2. **every candidate family** in the pre-registered set (entry families + exit/hold families + vol-
   input families) — the cross-family multiplicity, likewise deflated for correlation between
   families that share machinery.

`N_eff` is therefore `>= 2` for the Phase-C study (the incumbent's `N==1` base case of §3.3 does NOT
apply here — that base case is only for the shipped single pre-registered family). The luck benchmark
`SR*_0` grows with `N_eff` (§3.3), so under-counting `N_eff` **understates** the bar and can flip a
FAIL to a spurious PASS — the exact failure the gate exists to prevent. **The enumerated candidate
ledger (§C.9) IS the trial ledger:** `N_eff` is computed from the 8 numbered candidates + their frozen
per-candidate search grids (§C.9), and no candidate scored off that ledger may be adopted (an
off-ledger winner is, by definition, a snooped trial). §C.9 is the *same* frozen set as `docs/EDGE.md`
"The pre-registered candidate set (frozen)" and SIGNAL §C.2–§C.4 — one identical list across the three
specs, so the count feeding `SR*_0` and the off-ledger rule is unambiguous. The
per-observation-Sharpe / never-annualized rule (§4) applies to every `SR_pp` fed into DSR here.

- **PBO (Probability of Backtest Overfitting, CSCV).** Once the candidate ledger is large enough to
  form CSCV combinations (it is, for a multi-family study — unlike the incumbent-only report where PBO
  was a §6 deferral), PBO is computed over the pooled candidate performance matrix and reported
  alongside DSR. A candidate with high PBO (its in-sample rank does not survive the combinatorial
  train/test swap) is rejected regardless of its point OOS number.
- **Walk-forward folds do NOT enter the adopt statistic (closing the residual WF snoop path).** The
  anchored walk-forward of §C.2 is a **consistency-only overlay** (§C.1's single `SPLIT` is the
  headline gate); its per-fold refits and repeated per-fold OOS scores are themselves multiple looks
  at held-out data. To keep that surface off `N_eff` and off the adopt decision: the DSR/PBO adopt
  statistic and the §C.6 out-performance/CI numbers are computed on **the single pre-registered
  held-out slice ONLY** (§C.1, `SPLIT = 2025-06-01`). The WF folds feed **only** the cross-variant
  consistency check (§C.5, adopt-clause 4) — a pass/fail *robustness* read, never a scored trial that
  raises `SR*_0` and never the source of the adopt point estimate. Thus `N_eff` counts only (1)
  within-model hyper-parameter search and (2) cross-family multiplicity (above); the WF fold count is
  deliberately excluded, and no candidate may be adopted on a WF fold that beats the single slice.

### C.4 Parsimony / information-criterion penalty (prefer the simpler model at equal OOS)

At **equal OOS performance**, the **simpler model wins.** Formalized as an information-criterion
tie-break: each candidate carries an explicit parameter/complexity count `k` (free parameters fit on
train + latent-state dimension), and candidates are compared on an **IC-penalized OOS objective**
(AIC/BIC-style: OOS log-likelihood or OOS Sharpe penalized by `k`), never on raw OOS fit alone. A
richer model must beat the incumbent (and simpler candidates) by a margin **exceeding** its added-
complexity penalty to be preferred — a candidate that only matches OOS with more parameters LOSES to
the parsimonious one. The incumbent (fixed 5-day hold + −12% stop) is the parsimony floor: `k` is
minimal, so a candidate must earn its extra parameters on OOS, net of penalty.

### C.5 Winner-robustness (drop-top-N) and matched comparisons

- **Drop-top-N is a first-class gate, not a report.** The edge is convex and fat-tailed (§2.2,
  `docs/EDGE.md`): a candidate's OOS out-performance must **survive dropping the top-1 and top-3 OOS
  winners** — `ev_drop1`/`ev_drop3` (SIGNAL §11) recomputed on the candidate's OOS trades — else the
  "edge" is one or three trades of luck and the candidate is rejected. This is applied to the *paired
  difference* against the incumbent too: drop the trades that most drive the candidate's margin and
  the margin must persist.
- **Matched comparisons (SYSTEM §2.3, CLAUDE.md).** A candidate that only changes the exit/hold is
  compared to the incumbent on the **same entries / same trade dates** (equal hold horizon where the
  comparison is about a different axis; same instrument at equal expiry). A candidate that changes the
  entry is compared with the exit/hold held fixed at the incumbent's. No candidate is compared across
  a confound (different hold *and* different moneyness at once) — that is the matched-hold discipline
  the record was built on.
- **Cross-variant consistency (the anti-snooping mechanism).** A candidate must out-perform
  **consistently across its own sibling variants** (e.g. across the instruments/expiries of the
  pooled basket, across the walk-forward folds), not on one lucky slice. A filter that helps one
  variant but not its siblings is the snooping trap the record already caught (the consec-loss trap,
  §2.3) — such a candidate is rejected as an artifact.

### C.6 ADOPTION RULE (the single conjunction — replace the incumbent ONLY IF all hold)

A Phase-C candidate **replaces the incumbent IF AND ONLY IF, on the HELD-OUT OOS slice, ALL of:**

1. it **BEATS the incumbent OOS** — a positive OOS out-performance (matched, paired where applicable,
   §C.5), net of the parsimony/IC penalty (§C.4); **AND**
2. it **PASSES the multiplicity gate** — DSR ≥ the pre-registered pass threshold (§8, `0.95`) computed
   on the per-observation Sharpe with `N_eff` counting the full within-model + cross-family search
   (§C.3), **AND** its PBO is below the pre-registered ceiling; **AND**
3. it **SURVIVES drop-top-N** — its OOS out-performance persists after removing the top-1 and top-3
   OOS winners (§C.5); **AND**
4. it is **CROSS-VARIANT CONSISTENT** — out-performs across its sibling variants / walk-forward folds,
   not on a single slice (§C.5); **AND**
5. its **PAIRED-BOOTSTRAP CI CLEARS ZERO** — the Sharpe-difference vs the incumbent, from
   `bootstrap_ci` (§3.2) at the pre-registered **`B = 20000`, `block = 1`** (§8) on the *same* matched
   ORB trade dates, has a **90% CI that EXCLUDES zero** (a favorable point estimate is not enough).
   This clause is load-bearing, not cosmetic: it is precisely the test that in the futures-vs-options
   study **STRADDLED zero** ([−0.27, +0.42], P(option > future) ≈ 0.66) and killed the NIFTY
   option-vs-future claim (`docs/EDGE.md`); SYSTEM OWNS `bootstrap_ci` (§3.2) and must apply it in its
   own adopt gate, so dropping it would leave the SYSTEM conjunction weaker than EDGE's. This matches
   `docs/EDGE.md` "The adoption threshold (frozen)" clause 5 — the SYSTEM and EDGE conjunctions are the
   **same five clauses**.

**ELSE the negative is RECORDED and the INCUMBENT is KEPT.** Every scored candidate — winner or loser
— is written to `docs/EDGE.md` as a dated, pre-registered result (so nothing is silently re-tested;
CLAUDE.md append-never-rewrite). Failing ANY one clause is a KEEP-INCUMBENT + record-the-negative
outcome; there is no partial-credit adoption and no re-scoring a loser off a fresh grid. Ties (equal
OOS net of penalty) resolve to the incumbent (the parsimony floor) — adoption requires a strict,
gate-clearing win.

### C.7 Deferrals (the long tail, explicit — not omissions)

The pre-registered set is a **focused** subset of the strongest formal representatives per category
(entry: change-point / sequential-detection / regime-switching / Lee-Mykland; exit-hold: optimal-
stopping free-boundary / OU-Leung-Li / survival-hazard / regime-switching reversal; vol-input:
HAR-RV directional). The full method zoo — every additional detector variant, every finer grid node,
every alternative latent structure — is **deferred**, precisely to keep `N_eff` small (§C.3): each
un-pre-registered method NOT added is a trial NOT paid for. A deferred method is a stated decision to
re-open only under a new pre-registration, never a silent later add to the same study.

### C.8 DATA GATE — the empirical RUN is BLOCKED-ON-LAKE

- **The harness is BUILT and SYNTHETIC-TESTED; the empirical RUN is BLOCKED.** The selection RUN
  requires the **wide-band / position-following lake** (SIGNAL §10.4, `docs/EDGE.md` data-assets) so
  drifting winners' marks are real, never delta-reconstructed. **That lake is ABSENT.** The Phase-C
  deliverables are therefore ONLY: (a) this pre-registration protocol, (b) the candidate-model code +
  the SYSTEM selection harness, **synthetic-tested via known-generating-process recovery** (a fitted
  candidate must recover the parameters of a simulated process it is the correct model for; no real
  data touched).
- **Guard behavior (non-negotiable, mirrors the repo guardrail).** While the wide-band lake is
  absent, the selection RUN **must RAISE** (a guard error naming the missing wide-band data), **never
  fabricate a backtest**, never emit a metric, never adopt a candidate, and never loop. A synthetic
  fixture is admissible ONLY for the known-generating-process recovery tests and MUST be
  unmistakably tagged synthetic — it may never be passed off as a selection result. This is the
  data-lake-absent guardrail (BUILD_PLAN §Guardrails) made a hard precondition of Phase-C's RUN:
  BLOCKED-ON-LAKE is an acceptable terminal state; a fabricated selection is not.

### C.9 THE ENUMERATED TRIAL LEDGER (pinned here — the one artifact `N_eff` is charged against)

`deflated_sharpe` and the off-ledger rule (§C.3) both depend on **one** pinned, enumerable candidate
set. SYSTEM owns `deflated_sharpe`, so the ledger is enumerated **here**, with a fixed trial count, and
it is the **identical** list frozen in `docs/EDGE.md` "The pre-registered candidate set (frozen)" and
SIGNAL §C.2–§C.4. No candidate outside this list may be scored or adopted; an off-ledger winner is a
snooped trial by definition (§C.3). Each candidate is admitted ONLY in its causal FILTERED /
one-step-ahead form (§C.2 purity); every within-candidate parameter *searched* on train adds to
`N_eff` (§C.3, clause 1) — the per-candidate grid is not free.

**ENTRY (structural-break / detection / regime — replaces or gates the ORB strong-close trigger):**
1. **Bayesian Online Change-Point Detection (BOCPD, Adams–MacKay)** on the intraday drift — enter in
   the sign of the post-break *filtered* drift; the online run-length posterior only (never the
   retrospective MAP segmentation). *Grid (train-fit, frozen):* hazard `λ`, NIG prior
   `(μ_0,κ_0,α_0,β_0)`, run-length-mass entry threshold.
2. **Sequential detection — Wald SPRT / GLR** change detector on the drift sign — enter on first-
   passage of the log-likelihood-ratio boundary. *Grid:* the SPRT boundaries `(A,B)` / GLR threshold,
   pre-set from train (ARL-style, not P&L-tuned).
3. **2-state Gaussian HMM on the FILTERED posterior** (Hamilton filter, one-step-ahead
   `P(state_t|F_t)`) — enter when the filtered high-drift-state probability crosses a train-frozen
   level; the Kim-smoothed posterior is BARRED (lookahead). *Grid:* EM-estimated `(μ_1,μ_2,σ_1,σ_2,P)`,
   probability threshold.
4. **Lee–Mykland (2008) jump test** (RE-ADMITTED, directional jump-continuation) — enter in the sign
   of a detected jump; the standardized statistic uses a causal trailing-window vol only. *Grid:*
   local-vol window length, Gumbel `α`.

**EXIT / DYNAMIC HOLD (replaces the fixed 5-day horizon; the −12% 1-min stop is RETAINED as the floor,
never loosened):**
5. **Optimal-stopping free-boundary** under a momentum / regime-switching latent — exit when the
   *filtered* state value crosses the continuation/stopping boundary solved on train. *Grid:* discount
   `ρ`, latent dynamics (inherited from the entry fit); boundary solved once and frozen.
6. **OU / Leung–Li mean-reversion optimal exit** (RE-ADMITTED) — the closed-form optimal liquidation
   level `b*` under an OU latent, as a dynamic-hold boundary; exit on the first `F_{t-1}`-measurable
   crossing. *Grid:* OU `(θ,m,σ)`, discount rate; `b*` a frozen deterministic function of those.
7. **Survival / hazard duration** — dynamic hold = reversal-hazard threshold crossing (Cox / Weibull
   AFT on `F_{t-1}`-measurable covariates); exit when the one-step-ahead reversal hazard exceeds a
   train-frozen level. *Grid:* hazard coefficients, exit-hazard threshold.

**VOL INPUT (a covariate / vol-scaling / adaptive-threshold input to the above — NOT a standalone
signal):**
8. **HAR-RV (Corsi 2009)** (RE-ADMITTED, DIRECTIONAL use only) — realized-vol cascade as vol-scaling /
   adaptive-threshold / covariate input; consumed ONLY as the one-step-ahead forecast
   `\hat{RV}_{t|t-1}` from lags `<= t-1` (§C.2 purity, SIGNAL §C.4). *Grid:* OLS HAR coefficients
   `(c,β_d,β_w,β_m)`. The **RV−IV spread signal STAYS BARRED** — not on this ledger in any form.

**Total pre-registered candidate trials: 8 (= 4 entry + 3 exit/hold + 1 vol input).** This integer is
the cross-family term of §C.3 clause 2; `N_eff >= 8` once each candidate's own scored parameter grid
(§C.3 clause 1) is added and deflated for within/between-family correlation. `N_eff` is `>= 2` for the
study, so the §3.3 `N==1` base case never applies (as §C.3 already states).

**Deferrals that pin the ledger (the entries most prone to cross-spec drift — must be DEFERRED here,
not scored):** offline/batch **CUSUM**, **Bai–Perron**, and **e-divisive/ECP** change-point (offline
segmentation ⇒ lookahead unless run strictly expanding-window; the online BOCPD #1 is the admitted
change-point representative and the SPRT/GLR #2 is the admitted sequential representative); the
exit-side **Bayesian change-point of the reversal** and standalone **regime-switching reversal** exit
(folded into #1/#3); **≥3-state / semi-Markov HMM**; **signed-jump / semivariance / HARQ** vol variants.
The full long tail is §C.7.

**Cross-spec consistency note (RECONCILED at integration 2026-07-04 — the ledger IS one identical list).**
This §C.9 ledger is authoritative-consistent with `docs/EDGE.md` "The pre-registered candidate set
(frozen)" (which also states "Total: 8"). Entry candidate **#2 is Wald SPRT / GLR** in all three specs:
EDGE §Phase C pre-registration item 2, this §C.9 item 2, AND SIGNAL §C.2`(b)` — SIGNAL scores SPRT/GLR
and DEFERS CUSUM to its §C.5 (with the explicit "one sequential representative suffices; CUSUM is
online/causal, deferred for parsimony, NOT for lookahead" rationale). CUSUM, Bai–Perron, and
e-divisive/ECP are the deferred change-point siblings in all three. Because the ledger drives `N_eff`
and the off-ledger rule, this identity is load-bearing: the three specs name the *same* single
sequential-detection representative, so exactly ONE sequential trial enters the count and the total
stays 8 (picking the same one in all three, never both, which would raise `N_eff`). *(Historical note:
an earlier draft of this note flagged a SIGNAL-vs-EDGE disagreement where SIGNAL scored CUSUM as #2;
that was corrected in SIGNAL §C.2/§C.5 before this commit — all three specs now enumerate the identical
8-trial set with SPRT/GLR as #2.)*
