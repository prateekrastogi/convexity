# P5 PRE-REGISTRATION — RV BREADTH + DRAWDOWN SHAPING (Program 5)

**Frozen before any run. Committed 2026-07-12. Owner objective: lift the commodity book's
net CAGR:maxDD ratio from ~0.5:1 (deployed RV: 8.9%/18.5%) toward ~1:1, building ON the
deployed RV pair core — the construction that survived Program 4.**

Binding context from prior certified results (not re-litigated here):
- Program 3 (DURABLE): directional TSMOM and carry sleeves on this universe are DEAD sub-noise
  (drop-top-5 negative everywhere). P5 does NOT include directional/carry return engines.
- Program 4 (RV): the deployed core — band z {26,52,104}wk on log price ratio, β=1, HL-gate
  [1,26]wk, pair risk parity, weekly Friday, net Sharpe 0.732 @ 12bp — is the anchor. Its
  construction parameters are FROZEN throughout P5; P5 changes only universe breadth and adds
  drawdown-shaping overlays, then calibrates sizing.

## The two pre-named levers (in fixed order; each ONE shot)

### Step A — breadth: the HG (copper) extension "U1"
COPPERUSDT is now venue-listed; HG futures curves cover 2010-06→now in the lake. Add EXACTLY
these three pairs, chosen for economic priors, not mined:
- **HG–CL**: industrial-demand complex (global growth beta on both legs).
- **HG–PL**: industrial metals pair.
- **GC–HG**: the classic copper/gold macro ratio (KNOWN RISK, stated up front: this ratio
  trends with macro cycles and may not revert — the GC–SI precedent. The HL gate is expected
  to de-activate it in trend regimes; if it scores ~0 that is the honest signature, not a bug).
U1 book = 8 pairs (5 deployed + 3 above), all other construction identical.
**Accept U1 iff**: net Sharpe(U1) > net Sharpe(S0 repro) AND drop-top-5(U1) > 0 AND no new
pair's solo maxDD > 70%. Else the book stays 5-pair and Step B runs on S0.

### Step B — drawdown waterfall (the D1 equity method, verbatim discipline)
Candidates in FROZEN order, evaluated ONE SHOT each on the Step-A winner, composing accepted
ones (a candidate is evaluated on the composition of previously accepted candidates):
1. **B1 GZ-brake@25%**: m_t = clip(1 − DD_{t−1}/0.25, 0, 1) off the book's own equity curve,
   weekly, applied to gross.
2. **B2 active-fraction gross**: book gross ∝ (open HL-gates / n_pairs), NO renormalization —
   the equity book's breadth-proportional analog.
3. **B3 per-pair stop-reset**: a pair de-activates when its own running drawdown (pair equity,
   from inception of current activation) exceeds 25%; re-arms when its z re-crosses 0.
**Accept a candidate iff**: relative maxDD reduction ≥ 15% AND relative CAGR reduction ≤ half
the relative DD reduction AND CPCV frac_pos not reduced by more than 0.05. No re-ordering, no
retries, no parameter tweaks after seeing results.

### Step C — sizing calibration (report-only; NO gate, NO selection)
On the final pre-named net series: the vol-scalar frontier k ∈ {0.5..3.0 step 0.25}: CAGR(k),
maxDD(k), the k* where CAGR≈maxDD (the owner's 1:1 point), and quarter/half-Kelly (drawdown
distribution reported, method identical to the equity book's reference derivation). The
deployment recommendation, if any, is stated at a maxDD budget ≤ 25%.

## Costs, panel, mechanics (all inherited frozen from Program 4)
Weekly Friday panel 2010-06→now on databento front-month settles (roll-aware, instrument_id;
negative-price invalidation; ±40% winsorization); 12bp RT primary, {6, 24}bp scenarios and
funding ±5bp/wk; HG legs planned at the 24bp number like the other thin legs. Positions from
~2012 (104wk warmup).

## Adoption gates (FINAL construction only; frozen)
| gate | threshold |
|---|---|
| net Sharpe | ≥ 0.75 (must beat the deployed 0.732 — else the program did not improve the book) |
| CPCV (8 groups, k=2, embargo 8wk) | mean path > 0 AND frac_pos ≥ 0.90 |
| per-obs DSR | ≥ 0.90 at N_eff = 30 (Program 4's 20 trials + P5's ≤10 cells, charged cumulatively) |
| PBO (CSCV) over the P5 family (≤12 configs) | ≤ 0.25 |
| drop-top-5-weeks Sharpe | > 0 |
| 24bp scenario Sharpe | ≥ 0.50 |
| ratio at recommended sizing | CAGR:maxDD ≥ 0.8 |

**Verdict rules**: all gates pass → recommend deploy of the final PRE-NAMED construction (owner
decides venue/sizing separately, per standing practice). Any gate fails → record the negative;
the deployed 5-pair book stands unchanged; no partial adoption of individual overlays without
the gates. The 1:1 objective is NOT itself a reason to lever past a failed gate.

## Cell count (DSR/PBO honesty)
S0 repro (1) + U1 (1) + new-pair solos (3, diagnostics only) + B1/B2/B3 (3) + accepted
compositions (≤2) + final (1) ≈ 8–10 evaluated cells; family for PBO = {5p, 8p} × {no overlay,
B1, B2, B3, accepted-composition, all-three} = 12 configs.
