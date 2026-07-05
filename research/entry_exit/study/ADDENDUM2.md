# PREREG ADDENDUM 2 — stage-3 compute-feasibility amendments (frozen 2026-07-04, BEFORE any candidate run)

Declared BEFORE any candidate result exists (only baseline + anchor-diagnostic numbers have been
seen). Motivation: BOCPD's causal `signal_at` is O(t²) per decision on this 2-core VM; the main
prereg's 28-split CPCV with train-side grid SELECTION for fitted candidates is computationally
infeasible (weeks). Amendments:

1. **No train-side grid selection.** Every declared grid point runs OOS and is REPORTED as its own
   family member; `N_eff` charges ALL of them (this removes a selection step, strengthening — not
   weakening — the anti-overfit accounting; nothing is picked on train, nothing tuned).
2. **Resampling for FITTED candidates = expanding walk-forward over 5 equal blocks, 10-day
   embargo** — the LAST FOUR blocks are scored OOS (the first is train-only seed); each fold fits
   on [0, block_start−embargo) and scores its contiguous block with params FROZEN.
   The fixed-parameter baseline keeps its 28-split CPCV (already run). The paired candidate-vs-
   baseline comparison uses the SAME OOS days union.
3. **BOCPD grid trimmed** from hazard 1/{50,100,200}×p_break{0.5,0.7} (6) to
   **1/{100,200} × {0.5,0.7} (4)**. All other grids unchanged: SPRT/GLR (α,β)∈{(.05,.2),(.01,.1)}
   ×m∈{20,60} (4), HMM thresh {0.6,0.75} (2), LM-jumps K∈{16,32} (2); exits X0 fixed-5d,
   X1 optimal-stopping (ρ=0.02), X2 OU/Leung-Li (r=0.05), X3 hazard quantile {0.5,0.7};
   HAR toggle {off,on}.
4. **Family size** = entry-points (4+4+2+2=12) × exit-variants (X0, X1, X2, X3@0.5, X3@0.7 = 5)
   × HAR {off,on} = **120 configs/symbol**; `N_eff` for DSR = 120 (+ the 6 baseline cells + 42
   addendum-1 cells; correlation-deflation per the main prereg is reported alongside the raw count).
5. **Hold cap** 10 days for dynamic exits (per main prereg); a hold reaching a fold boundary exits
   at the boundary close (declared edge rule). Entry executes at the decision day's close with the
   model reading strictly `past()` (F_{t-1}) — one day MORE conservative than the baseline's
   same-day-intraday signal, stated and accepted (candidates carry no intraday information).
6. **HAR-on stop**: distance = s × 1.6 × HAR-forecast vol × price (1.6 ≈ the E[TR]/σ equivalence
   constant making it scale-comparable to the ATR20 stop; declared, not tuned). HAR fits on daily
   squared log-returns.

Interpretation/adoption rules of the main prereg §6 are unchanged.
