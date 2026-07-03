# Robustness study — PRE-REGISTRATION (phase ensemble / real cost / embargo-3)

**Pre-registered before any run** (commit timestamp = the record). Three items, none touching the
core (weekly dollar-neutral quintile 12-16d band momentum on price+funding only). R1 targets
timing-luck VARIANCE (Hoffstein-Faber-Braun rebalance-timing-luck, ~1/N tranching reduction); R2
and R3 are MEASUREMENT restatements (realism/hygiene), not alpha ideas.

## R1 — rebalance-phase ensemble (the adoption candidate)

The recorded book rebalances only at Friday closes — one arbitrary phase of 7 (crypto is 24/7).
Construction: run the IDENTICAL band ensemble on each weekday phase p in {Mon..Sun} (each phase's
signals computed FRESH at its own rebalance — zero staleness, unlike the rejected stale-tranche
variant), each settled by the FROZEN weekly settle_week on its own grid; the phase-ensemble series
is the week-index-aligned equal-weight average of the 7 phase books' weekly returns (the standard
tranched-portfolio construction; cross-phase compounding/boundary effects are second-order
~r^2/7 ~ 1e-4/wk and no cost-netting benefit is claimed — both conservative, documented).
Constants: N_phases = 7 (the number of weekdays — canonical, not searched).
Report: per-phase naive Sharpe/CAGR/maxDD (their DISPERSION = the measured RTL), the phase-mean
series' CPCV(8,2) mean/frac_pos vs the Friday book, and PBO impact.
DECISION RULE (fixed ex ante): the theory says E[r] is phase-invariant, so the ensemble is
PREFERRED UNCONDITIONALLY unless it fails sanity — adopt as the deployable construction UNLESS
(frac_pos drops > 0.10) OR (maxDD > 25%) on the aligned grid. If the phase-mean sits materially
below the Friday book, that is Friday-LUCK REVEALED and adopting the ensemble is the honest
restatement, not a degradation.

## R2 — per-name real cost in the settle (measurement restatement)

Wire the already-built, already-calibrated per-name cost (dollar-volume sqrt spread proxy + Amihud
sqrt impact, cost/friction/spread_impact.py) into the ensemble settle via settle_week's existing
per-symbol friction-array path, replacing the flat 13bp scalar. Constants (all pre-existing/
canonical): trailing window 30d (panel.py's own); anchor at the panel median ADV/Amihud (the
model's committed anchoring); book notional = $3,000,000 (LEVERAGE's own capacity-dial anchor —
the repo's one stated book-size number); per-name one-way notional = |dw_i| x NAV. Shortability
premium stays OUT (blocked on UNIVERSE per-symbol is_shortable; documented limitation, unchanged).
No adoption question: record the restated numbers alongside the flat-cost record; flag loudly if
the restated net Sharpe band drops below the 0.5 floor.

## R3 — embargo = 3 weeks CPCV re-baseline (validation hygiene)

Re-run the committed CPCV/PBO battery at embargo = ceil(L_max/7) = 3 weeks (LdP: embargo >=
feature horizon; the SY-5 gap). No adoption question: record alongside the embargo-1 numbers;
recommend the official record move to embargo-3 going forward if the deltas are small.

## Protocol
Same machinery as always (run_cpcv/compute_pbo byte-identical reuse); all runs checkpointed to
_ckpt/robust/results.json per stage; every series produced by the frozen settle_week.
Reusable harness: `research/xsec_robust.py` (run with `PYTHONPATH=. .venv/bin/python
research/xsec_robust.py` from the repo root; per-stage checkpoint resume).

## RESULTS (run 2026-07-03, full 100-crypto panel; raw detail in _ckpt/robust/results.json)

### R1 — MAJOR HONESTY FINDING: the edge is PHASE-CONCENTRATED at the Friday close

Per-phase band-ensemble Sharpe (identical strategy, only the weekly rebalance weekday changed):

| Mon | Tue | Wed | Thu | **Fri (recorded)** | Sat | Sun |
|---|---|---|---|---|---|---|
| −0.01 | +0.24 | +0.43 | +0.68 | **+1.12** | +0.26 | −0.03 |

RTL dispersion: Sharpe std 0.41 across phases; the 7-phase ensemble scores Sharpe 0.50 / CPCV
mean 0.38 / frac_pos 0.75 vs the Friday book's 1.06 / 0.96. **Per the pre-registered decision
rule the ensemble FAILS its sanity gate (frac_pos drop 0.21 > 0.10) and is NOT adopted — the
Friday-close construction stays deployed.** The honest reading, stated both ways: (a) the profile
is MONOTONE-rising into Friday then collapsing over the weekend — consistent with STRUCTURE, not
exchangeable luck (a Friday-close signal carries the full trading week's information into the
weekend regime; signals formed at weekend/Monday closes are polluted by thin weekend prints), and
Friday was the a-priori convention, never fitted; (b) with ~5yr a luck component cannot be
excluded — under exchangeability the pre-chosen phase landing on the max is a 1-in-7 event, and
the honest expectation would sit nearer the phase-mean (~0.5). Recorded as the book's
second-biggest caveat after track length (EDGE.md §(d)); paper trading at the Friday phase is the
out-of-sample discriminator.

### R2 — per-name real cost: the flat-cost record HOLDS (restatement immaterial)

Per-name dollar-volume spread + Amihud impact at NAV = $3M, one-way notional |dw_i|·NAV: cost
3.80 → 4.13 %/yr, naive Sharpe 1.124 → 1.109, CPCV mean 1.061 → 1.045 (same aligned grid). The
flat-13bp record is robust to the fuller cost model at this book size (~0.02 Sharpe haircut).
Shortability premium remains out (documented, blocked on UNIVERSE per-symbol flags).

### R3 — embargo-3 re-baseline: the edge is NOT an embargo-leak artifact — ADOPTED as official

Committed 7-lookback family, L=14 selected, embargo 1wk → 3wk: CPCV mean 0.914 → 0.950 (rose),
frac_pos 1.00 → 0.93, PBO (8,2)/(6,2)/(10,3): 0.393/0.467/0.308 → 0.429/0.400/0.425 (range
~[0.40, 0.43], same moderate zone). Embargo = 3 weeks (= ceil(L_max/7), LdP's embargo ≥ feature
horizon) is now the OFFICIAL validation configuration; embargo-1 numbers remain the historical
record.
