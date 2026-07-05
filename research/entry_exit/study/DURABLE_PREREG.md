# DURABLE COMMODITY BOOK — PRE-REGISTRATION (Program 3; FROZEN 2026-07-05, before any run)

Owner-directed durable-edge study. Isolation: runs in the `commodity` worktree
(/home/speculation/convexity-commodity); reads the shared on-disk lake at
/home/speculation/convexity/data (databento curves + COT); NEVER touches the live paper system
(main @ 31d3238) or src/convexity. Executed by durable_pipeline.py; outcome in DURABLE_REPORT.md.

## Thesis (accepted prior)
Durability and Sharpe are inversely coupled through limits-to-arbitrage: a non-decaying edge
survives because it periodically hurts enough that fast capital leaves, which caps its Sharpe. The
only families passing the three durability tests (structurally-compelled payers; periodically
painful to hold; real capacity) are **slow time-series trend** and **carry-on-storables**. Honest
target: **0.5–0.8 net combined**; 1.0 = good-regime outcome; **>1.2 = alarm/bug-hunt**; 18–24-month
flat stretches are the admission price, not a defect.

## Prior-kill reconciliation (why this is a NEW hypothesis, not a re-run)
- Dead #1 (convexity sweep): FAST breakout-with-stops, all timeframes/exits — gross R ≈ 0.
- Dead #2 (swing xsec, Program 2): CROSS-SECTIONAL ranks at breadth 7 — xsec structurally starved
  (2-vs-2 book; 4/7 degenerate-carry metals).
- **Neither covers slow per-name TIME-SERIES momentum**, which needs time-series *length* (we have
  16y), not cross-sectional *breadth*. That is the primary tested hypothesis here. If it also fails,
  the negative is recorded plainly.

## Universe / data / calendar
7 roots: CL, NG, GC, SI, HG, PL, PA (no zinc/aluminum — Binance caps at 7; MCX names in the owner
research are unavailable). Price basis = databento front-month daily settles, roll-aware returns via
instrument_id (roll-day return = c0_t/c1_{t-1}), negative-price invalidation + ±40% winsorization
(reused verbatim from swing_pipeline). Weekly **Friday-close UTC** grid, 2010-06→2026-07 (~839 wk,
breadth 7). COT joined at usable_from ≤ Friday. Clusters (for risk parity): {GC,SI,PL,PA} precious,
{HG}, {CL}, {NG} → effective breadth ≈ 4.

## Sleeves (frozen; grids ARE the multiplicity ledger)
- **S1 TSMOM (all 7).** Per-name position = sign of trailing L-week return, L∈{13,26,39,52} deployed
  as the **BAND ENSEMBLE** (equal-weight mean of the 4 signed books — NEVER argmax). Positions
  vol-scaled 1/σ20d. Time-series directional (net exposure floats; vol-target handles it).
- **S2 CARRY-ON-STORABLES ({CL,NG,HG} only).** Per-name position = sign of the annualized front
  slope (rank0→rank1, annualized by data-derived per-root roll spacing), vol-scaled 1/σ20d.
  Time-series, NOT xsec (breadth 3 makes ranks meaningless). Preciouses excluded (degenerate carry:
  basis = funding − lease, no convenience yield).
- **S3 INTERACTION.** Carry-gated trend on storables: (a) GATE variant — take the storable's trend
  position only where sign(carry)==sign(trend), else 0; (b) VETO variant — trend position unless
  carry opposes, then 0. Preciouses keep standalone S1 trend in both.
- **S4 CLUSTER RISK PARITY + VOL TARGET.** Equal risk contribution across the 4 clusters (scale each
  cluster's sub-book to equal trailing-vol), then book vol-target 10% annualized (26-wk trailing σ,
  scale capped [0.5, 2]).
- **S5 KRT overlay (≤20% risk share, declared).** Small tilt toward last-week reversal on names with
  a COT |Δmm_net|/OI flow shock (top-half), usable_from ≤ Friday. Opportunistic, not load-bearing.

## Combined book (the pre-named deployable)
S1 (preciouses standalone trend) + S2 + S3-gate (storables) → cluster-risk-parity (S4) + 10%
vol-target overlay + S5 overlay. This exact construction is what gets a deploy recommendation IF it
passes the gates — never the best single cell.

## Costs & validation
Net of 12bp RT primary per name-week traded ({6,24}bp + funding ±5bp/wk scenarios reported). CPCV(8
groups, k=2, embargo 8 wk) mean-path + frac_pos; per-observation DSR at N_eff = every grid point
charged (~15: S1×4 L + S1-ensemble + S2 + S3×2 + S5 + combined + cost/robustness variants); PBO over
the family; drop-top-5-weeks; per-year Sharpes.

## Regime sanity signature (MANDATORY pre-belief check)
The **2011–2019 trend drought MUST show flat-to-weak** in S1/combined equity; **2020–2022 must carry
the P&L**. A smooth equity line through 2013–2019 = overfit/lookahead/bug → investigate BEFORE
believing any headline number. This is the single most important honesty gate for a trend book.

## Adoption gate (frozen)
Deploy the pre-named combined book ONLY if net Sharpe positive AND DSR pass AND PBO ≤ 0.25 AND
drop-top-5 positive AND the regime signature holds (drought weak, 2020-22 strong). Else record the
negative and recommend NOT deploying. Leverage (quarter/full-Kelly + DD-vs-leverage) computed ONLY
if the combined edge is positive.
