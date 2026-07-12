# P7 PRE-REGISTRATION — INSTITUTIONAL RISK STACK FOR THE RV BOOK (Program 7)

**Frozen before any run, 2026-07-12 (after P5/P6 negatives; candidates chosen from the
owner-supplied institutional technique list, minus what is already deployed, already falsified,
or unvalidatable here — triage recorded in the report). Canonical parameters, no grids.**

Base: deployed Program-4 construction (S0 must repro 0.732/18.5). Owner objective unchanged:
reduce drawdown per unit CAGR.

## Candidates — FROZEN order, ONE shot each, composed (P5/P6 discipline)
1. **K1 Kalman dynamic hedge ratio**: replace β=1 with Kalman-filtered β_t (state: random-walk
   β; δ = 1e-4 canonical, obs noise via EW residual var λ=0.98). Spread s_t = a_t − β_t b_t;
   z/HL machinery unchanged on the Kalman spread. (P4 tested rolling-OLS β; Kalman is the
   institutional standard — smoother, no window cliff.)
2. **K2 CUSUM change-point on spread INNOVATIONS** (the honest version of #1; distinct from
   P6's falsified z-LEVEL stop): innovations e_t from the trailing AR(1) (the HL machinery's
   fit), standardized by trailing σ_e (26wk); two-sided CUSUM with k=0.5, h=5 (canonical).
   On alarm: pair de-arms 8 weeks AND until CUSUM resets below h/2. Targets distribution
   breaks (mean/variance shifts in dynamics), which need not coincide with stretch.
3. **K3 HMM regime gate**: 2-state Gaussian HMM (own EM) on the BOOK's weekly returns,
   expanding refit every 52wk, filtered (causal) state probabilities; gross × 0.5 while
   P(high-vol state) > 0.7.
4. **K4 Expected-shortfall sizing**: replace the vol-target overlay with ES-targeting:
   scale_t = clip(ES* / ES_t, 0.5, 2.0), ES_t = −mean of the worst 10% of trailing 52wk book
   returns (causal shift 1), ES* = expanding median of ES_t (≥104wk burn-in).
5. **K5 ensemble vol target**: the deployed vol target with σ_t = mean(26wk SD, EWMA λ=0.94)
   — two-estimator ensemble, same 10% target, same [0.5,2] clamp.
6. **K6 PCA-1 neutralization**: project the combined weight vector orthogonal to the first
   principal component of trailing 104wk asset-return covariance (causal shift 1), per week —
   removes the common commodity-factor loading RV books accumulate silently.
7. **K7 Amihud liquidity de-gross**: per-leg trailing 13wk Amihud (|daily ret|/dollar volume,
   from the daily curve data) ; a pair de-grosses ×0.5 while EITHER leg's Amihud exceeds its
   own expanding 80th percentile (≥104wk burn-in) — de-risk into evaporating liquidity.

**Acceptance per candidate**: the P5/P6 ½-rule (relDD ≥ 15%, relCAGR ≤ ½·relDD, CPCV frac_pos
held within 0.05) **OR** (pre-registered second branch, needed for construction-quality
candidates K1/K5/K6): ΔSharpe ≥ +0.05 AND maxDD not worse. Stated openly: the OR-branch admits
Sharpe-improving changes that the DD-rule alone would miss; it still requires no DD degradation.

## Gates (final composed construction; N_eff = 50 cumulative: P4 20 + P5 10 + P6 8 + P7 ≤12)
Sharpe ≥ 0.70 · ratio at 25%-DD sizing ≥ 0.60 · CPCV frac_pos ≥ 0.90 · drop-top-5 > 0 ·
24bp Sharpe ≥ 0.45 · DSR ≥ 0.80 (reported at N_eff=50) · PBO ≤ 0.35 over the P7 family.
Verdict rules identical to P6: pass → recommend for PAPER first; fail → record negative,
deployed book stands, no piecemeal adoption.

## Explicitly out of scope (with reasons, so the record is complete)
Tail overlays (#18: no options/VIX instruments on venue) · RL sizing (#16: unvalidatable at
839 weekly obs) · full convex optimization (#12: degenerate at breadth 5; CVaR essence in K4) ·
already-deployed items (#5, #7, #15, #20) · already-falsified items (P5 B1–B3, P6 L1–L5).
