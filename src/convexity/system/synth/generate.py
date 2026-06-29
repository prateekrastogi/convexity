"""Synthetic adversarial COMPOSED scenarios + scripted module outputs (SYSTEM §13).

Builders that produce the SYSTEM boundary objects (``BookState``, ``ProposedPortfolio``, scripted
``TrendAction`` / ``CandidateSignal`` maps) and the validation fixtures (the §7.3b DSR ledgers, the
anti-correlated ledger, the persistent-trend label scenario). Each scenario asserts a COMPOSITION
behavior — budget contention, selection-drop vs trend-ride, correlation-to-1, governor cascade,
cadence/deadband, oscillation, endogenous-horizon leak, the DSR/track-record/monotonicity/
annualization/forward-gate traps. Deterministic (seeded); no external data.
"""

from __future__ import annotations

import numpy as np

from convexity.core.types import EntrySignal
from convexity.system.compose.orchestrator import CandidateSignal
from convexity.system.modules.trend_adapter import TrendAction
from convexity.system.types import BookState, OpenPosition, ProposedPortfolio, TrialRecord


# --------------------------------------------------------------------------- composition fixtures


def make_book(
    positions: dict[str, OpenPosition] | None = None,
    *,
    equity: float = 1000.0,
    peak_equity: float = 1000.0,
    free_margin: float = 800.0,
    governor_multiplier: float = 1.0,
    cov: np.ndarray | None = None,
) -> BookState:
    """A :class:`BookState` for composition tests. ``governor_multiplier=0`` exercises the FLAT_ALL
    branch (governor cascade, §13); a low ``equity/peak_equity`` exercises the drawdown axis."""
    positions = positions or {}
    n = max(len(positions), 1)
    cov = cov if cov is not None else np.eye(n)
    return BookState(
        equity=equity,
        peak_equity=peak_equity,
        free_margin=free_margin,
        positions=positions,
        cov=cov,
        governor_multiplier=governor_multiplier,
    )


def make_open_position(
    instrument: str,
    *,
    side: int = 1,
    units: int = 2,
    entry_price: float = 100.0,
    held_risk_weight: float = 0.05,
    trend_state: str = "alive",
) -> OpenPosition:
    return OpenPosition(
        instrument=instrument,
        side=side,  # type: ignore[arg-type]
        units=units,
        entry_price=entry_price,
        held_risk_weight=held_risk_weight,
        trend_state=trend_state,
    )


def make_proposed(
    keep: tuple[str, ...],
    eligible: tuple[str, ...] | None = None,
    *,
    risk_weights: dict[str, float] | None = None,
    rank: dict[str, float] | None = None,
) -> ProposedPortfolio:
    eligible = eligible if eligible is not None else keep
    return ProposedPortfolio(
        keep=keep,
        eligible=eligible,
        risk_weights=risk_weights or {s: 0.05 for s in eligible},
        rank=rank or {s: 0.0 for s in eligible},
        reason_codes=("synthetic",),
    )


def make_entry_signal(instrument: str, *, direction: int = 1, quality_score: float = 1.0) -> EntrySignal:
    """A scripted breakout entry. ``quality_score`` is carried for audit but is FORBIDDEN as a
    composition input (Invariant CB); the score-scaling test multiplies it freely."""
    return EntrySignal(
        symbol=instrument,
        signal_idx=0,
        entry_idx=0,
        direction=direction,  # type: ignore[arg-type]
        status="entry",
        quality_score=quality_score,
        threshold=0.5,
        entry_ref_price=100.0,
        invalidation=98.0,
        stop_distance=2.0,
        barrier_price=104.0,
        range_width_sigma=1.0,
        compression_pct=0.3,
        false_break_risk=0.1,
        urgency="urgent",
        evidence={},
        reason_codes=("synthetic",),
        detector_version="synth",
    )


def make_candidate(
    instrument: str,
    *,
    direction: int = 1,
    quality_score: float = 1.0,
    ceiling_weight: float = 0.06,
    wpc: float = 0.03,
    ceiling_slack: float = 0.5,
    equity_loss_per_lot: float = 5.0,
) -> CandidateSignal:
    """A scripted new-capital candidate (breakout entry + RISK metadata) for the orchestrator."""
    return CandidateSignal(
        signal=make_entry_signal(instrument, direction=direction, quality_score=quality_score),
        ceiling_weight=ceiling_weight,
        wpc=wpc,
        ceiling_slack=ceiling_slack,
        equity_loss_per_lot=equity_loss_per_lot,
    )


def make_trend_action(kind: str = "continue", *, time_cap: bool = False) -> TrendAction:
    alive = kind == "continue"
    return TrendAction(
        kind=kind,  # type: ignore[arg-type]
        alive=alive,
        target_fraction=1.0 if alive else 0.5 if kind == "reduce" else 0.0,
        time_cap=time_cap,
        reason_codes=(f"trend_{kind}",),
    )


# --------------------------------------------------------------------------- validation fixtures


# The §7.3b ledger of K=12 annualized OOS Sharpes (the DSR FAIL/PASS worked example).
DSR_LEDGER_SHARPES_ANNUAL = np.array(
    [0.92, 0.61, 0.40, 0.85, 0.30, 0.71, 0.55, 0.20, 0.78, 0.48, 0.66, 0.35]
)


def make_correlated_ledger(k: int, rho: float, *, n: int = 200, seed: int = 7) -> list[np.ndarray]:
    """K OOS path-return series with a target mean pairwise correlation ``rho`` (Section 7.1).

    A common factor + idiosyncratic noise yields a controllable rho_bar. ``rho < 0`` (via a sign-
    alternated factor loading) builds the anti-correlated ledger that must give N_sys = K
    (Invariant DSR-MONO). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    out: list[np.ndarray] = []
    for i in range(k):
        load = math_sign(rho, i)
        w = abs(rho) ** 0.5
        series = w * load * factor + (1.0 - abs(rho)) ** 0.5 * rng.normal(size=n)
        out.append(series)
    return out


def math_sign(rho: float, i: int) -> float:
    """Factor loading sign: +1 for a correlated ledger; alternating for an anti-correlated one."""
    if rho >= 0.0:
        return 1.0
    return 1.0 if i % 2 == 0 else -1.0


def make_trial_records(sharpes: np.ndarray, returns: list[np.ndarray]) -> list[TrialRecord]:
    """Build :class:`TrialRecord` rows for the persistent ledger from scored Sharpes + OOS series."""
    records: list[TrialRecord] = []
    for i, (sr, r) in enumerate(zip(sharpes, returns)):
        records.append(
            TrialRecord(
                config_hash=f"cfg{i:03d}",
                module_versions={"breakout": "1", "leverage": "1", "trend": "1", "selection": "1"},
                blue_point={"variant": i},
                oos_sharpe=float(sr),
                oos_returns_ref=f"ref{i:03d}",
                oos_returns=np.asarray(r, dtype=float),
            )
        )
    return records


def persistent_trend_prices(n: int = 20_000, *, drift: float = 0.0002, seed: int = 11) -> np.ndarray:
    """A persistent multi-week up-trend whose realized hold would exceed a typical horizon — the
    endogenous-horizon leak scenario (§13). With ``tau_cap = H_sys`` the label force-closes at
    ``H_sys`` (Invariant HC). Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    steps = drift + 0.001 * rng.normal(size=n)
    return 100.0 * np.exp(np.cumsum(steps))
