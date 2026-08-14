"""Fleet configuration — every owner-tunable value in ONE place (PAPER.md §12.1).

Values here are OWNER DECISIONS, changeable without code edits elsewhere. Provenance is noted per
value; nothing in this module is derived at runtime.
"""

from __future__ import annotations

# --- Cross-book capital allocation (per 1000 quantized units; owner decision, D1v2) -----------
ALLOC_UNITS: dict[str, int] = {"crypto": 400, "commodity": 300, "equity": 300}
ALLOC_TOTAL = 1000


def alloc_fraction(book: str) -> float:
    return ALLOC_UNITS[book] / ALLOC_TOTAL


# --- Equity book: the D1-validated drawdown-controlled half-Kelly policy ----------------------
# FROZEN half-Kelly constant, derived once from the validated construction's own empirical
# fat-tailed return distribution (D1 program; reference derivation pinned at
# tests/fixtures/equity_book/reference_leverage_policy.py, re-asserted by the parity test —
# deliberately NOT re-derived at boot, which would drift with new data).
EQUITY_L_HALF = 2.325628533609451
# GZ drawdown-brake ceiling (owner's drawdown budget; the D1 waterfall's Step-A policy).
EQUITY_BRAKE_CEILING = 0.50
# Point-in-time signal warmup: all three ensemble rules computable (~252d rule + buffer).
EQUITY_WARMUP_DAYS = 260
# Leveraged/inverse/decay ETF exclusions (validated finding: structural daily-rebalancing decay
# contaminates trend/beta signals — E3/CTA programs; UVXY alone was 20% of an unguarded book's
# gross P&L).
EQUITY_ETF_EXCLUSIONS: frozenset[str] = frozenset({
    "UVXYUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "KORUUSDT", "MVLLUSDT",
})
# Split/corporate-action watch: |1-day move| above this flags a name for human review and
# excludes it from rebalances until cleared (the AAPL-2020 unadjusted-split failure mode).
EQUITY_SPLIT_WATCH_THRESHOLD = 0.25
EQUITY_FLAGS_PATH = "data/universe/equity_flags.json"

# --- Partial-book renormalization (PAPER.md §12.8; owner switch, default OFF) -----------------
# Under the venue's min-notional/lot floor a fraction of names cannot hold a position at all, so
# a small book runs BELOW its target gross: measured 2026-08-13 the equity book held 1.122 gross
# against a 1.386 target, and that EXPOSURE gap — not composition — is the dominant tracking
# error vs the validated construction (7.98%/yr, falling to 4.23%/yr when the achievable names
# are scaled back up; correlation is 0.992 either way because ~20 names already capture all the
# diversification this universe offers).
# Enabling it means deliberately holding survivors ABOVE their per-name target weight, which
# reverses the standing "never trade more than intended" convention, and it restores risk the
# book is currently not taking. Owner decision — hence a switch, defaulted off.
# ENABLED for all three books by owner decision 2026-08-13: the fleet is small enough that
# the floor materially under-deploys every book, and matching the validated construction's
# GROSS is worth more than matching its per-name composition (TE 7.98%/yr -> 4.23%/yr on
# equity at nav 449; strict no-op once capital is large enough to hold every name).
RENORM_PARTIAL_BOOK: dict[str, bool] = {"crypto": True, "commodity": True, "equity": True}

# --- Integer exchange leverage dial (per-symbol, set ONCE at book boot; never per-trade) -------
# Binance USDT-M leverage is an integer per symbol. The dial is a CAP + margin divisor; effective
# book leverage is fractional via notional sizing. equity: 3 covers the 2.33x nominal ceiling and
# caps accidental over-leverage at 3x NAV. crypto/commodity: not set by this program (venue
# default stands; observed dial journaled at equity boot for fleet visibility — normalizing them
# is an owner cutover option, not an autonomous change).
LEVERAGE_DIAL: dict[str, int] = {"equity": 3, "crypto": 5, "commodity": 5}
# crypto/commodity normalized from the 20x venue default (owner decision 2026-07-11):
# the dial is a hard venue-side cap, defense-in-depth over software sizing; 5x >> any
# realistic book gross (~2x) and RAISES per-tier max position (the -2027 class shrinks).

# --- Cross-margin software guardrails (PAPER.md §12.2) --------------------------
# Account maintenance-margin-ratio thresholds. Justification: equity book worst historical day
# −10.4% at ~1.9x ≈ −20% book NAV ≈ −6% of account at 300/1000; DERISK at 60% still leaves >3x
# that shock before liquidation territory.
MARGIN_ALERT_RATIO = 0.40
MARGIN_DERISK_RATIO = 0.60
# De-risk priority (levered, directional book flattens FIRST; market-neutral books last).
DERISK_PRIORITY: tuple[str, ...] = ("equity", "commodity", "crypto")
# Liquidation-distance assertion: distance must exceed this multiple of the worst historical
# one-day move (12.1%, the 93-name benchmark's 2020-03-16).
LIQ_DISTANCE_MIN_MULTIPLE = 3.0
WORST_HISTORICAL_DAY = 0.121

# --- Account lock (PAPER.md §12.2) ----------------------------------------------
# lock path is venue-resolved (venue.state_root()/account.lock) — see lock.py, PAPER.md §12.6
ACCOUNT_LOCK_TIMEOUT_S = 300.0

# --- Shared API pacing (PAPER.md §12.2) ------------------------------------------
AUTHED_CALL_MIN_INTERVAL_S = 0.12
PUBLIC_CALL_MIN_INTERVAL_S = 0.06
# ORDER placements have their OWN venue cap, discovered live at prod cutover 2026-07-11:
# -1015 "Too many new orders; current limit is 20 orders per TEN_SECONDS" (account tier).
# 0.55s floor => <=18 orders / 10s, inside the cap with margin; demo tolerates it trivially.
ORDER_CALL_MIN_INTERVAL_S = 0.55

# NO-TRADE BAND CONSTANT (EXECUTION.md §4.2). Band half-width is
# b = BAND_C * (theta/rho)^(1/3) with theta = 2*(taker + half_spread) — so this ONE constant sets
# the band for all three books (theta differs per book via cli.PLAN_HALF_SPREAD).
#
# FITTED OFFLINE 2026-08-14 (research/bandcal/), which is what §4.2 demands: c is "fitted once
# offline to sigma_w", and ExecutionDefaults deliberately REFUSES to default it ("refusing to
# invent a number is the contract"). policy.build_plan had been supplying 0.5 as a Python default
# — inventing exactly the number the config layer refused to invent. At 0.5 the band is 5.85% of
# NAV (equity/commodity) and 5.31% (crypto), LARGER THAN THE ENTIRE TARGET POSITION of 100% of
# names in both the equity and crypto books. Consequence: every resize of a held name was frozen
# and only REASON_FLIP_FULL trades (open-from-flat / close-to-flat / sign change) reached the
# venue. That is the mechanism behind the stable ~640 USDT demo-crypto position_drift that recon
# logged daily for a week without ever correcting.
#
# Calibration (research/bandcal/band_calibration.py). FAITHFULNESS GATE: at c=0 the harness
# reproduces the D1-validated reference BIT-EXACTLY (max diff 0.00e+00), so the rows are trusted:
#     c=0.50 (was shipped)  CAGR 15.21%  Sharpe 1.015  turnover  1.39x/yr  95.5% decisions skipped
#     c=0.03 (this)         CAGR 29.71%  Sharpe 1.204  turnover  6.06x/yr  53.0% skipped
#     c=0.00 (no band)      CAGR 28.26%  Sharpe 1.168  turnover 10.15x/yr  = validated construction
# A SMALL band genuinely beats no band — it saves ~4x/yr of turnover at 13bp, which is exactly
# what §4's band is for. 0.5 was simply 15-35x too large.
#
# ROBUSTNESS (research/bandcal/band_robust.py): the 0.02-0.05 plateau wins in BOTH history halves
# (H1 argmax 0.03, H2 0.05) and beats 0.5 in 7/9 years. The choice WITHIN 0.02-0.05 is noise;
# moving off 0.5 is not. Note 0.5 did post a LOWER maxDD (25.75% vs 30.47%) — a frozen book
# de-risks passively, which is why it won 2022 — but it costs 13 CAGR points to buy that.
#
# FITTED ON THE EQUITY BOOK ONLY — the one book with a reproducible validated backtest. Crypto and
# commodity inherit it; their bands land at 0.32%/0.35% of NAV = ~23%/~5% of a median position,
# which is sane, but they are NOT independently calibrated. Recalibrate per book if either grows
# its own validated backtest.
BAND_C = 0.03