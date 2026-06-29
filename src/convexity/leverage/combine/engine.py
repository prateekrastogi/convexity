"""LeverageEngine — min() of seven ceilings + class cap, governor once, lot gate (LEVERAGE §1).

The combine layer (§2 layer 3). The whole engine in one equation (§1):

    L_i_pre_governor = min( L_stop, L_vol, L_kelly, L_tail,
                            L_portfolio, L_margin, L_liquidity, L_class_max )
    L_i_final        = L_i_pre_governor · drawdown_governor(equity_drawdown)

It is a min() of independent failure-mode ceilings, NEVER a product of scores (Inv B/P6),
clamped by the HARD per-class guardrail ``L_class_max`` (§8.6.6, Inv CLASSCAP), THEN
scaled by the drawdown governor EXACTLY ONCE on the disjoint equity axis (Inv G, P8),
then lot-quantized (Inv U — rounding can only REDUCE risk).

REGIME's ``g_t`` enters ONLY through ``L_portfolio``'s ``g_t·H_book`` input (P7/Inv R):
it is NOT a separate min() term and NOT a post-min scalar (only the governor is
post-min). A regime cut surfaces as ``binding_ceiling='portfolio'`` with the portfolio
reason_code ``'regime_heat_cut'``. The Kelly fraction φ is consumed ONCE inside
``L_kelly`` (Inv F) — the engine never multiplies by φ again.

BookState is the CANONICAL equity/peak/cov source (ARCHITECTURE §5.9): the governor's
equity-drawdown axis (``book.peak_equity``/``book.equity``), the portfolio ceiling's
covariance / candidate-weights (``book.cov``/``book.cov_index``/``book.candidate_weights``),
the account-equity basis and the free-margin / existing-book-margin the quantizer
re-checks all read from ``book`` — NOT from divergent ``risk.*`` mirrors. ``RiskInputs``
supplies only the MEASURABLE market-state view that is NOT book-owned (forecast vol, the
Kelly-only μ, gap/tail fits, financing, liquidity, the regime ``g_t``). One book, one
risk view — no two divergent risk pictures.

Diagnostics are LOG-ONLY (Inv H): the leverage is invariant to diagnostic contents — it
depends only on the ceiling VALUES, the governor multiplier, and the quantizer.
"""

from __future__ import annotations

from convexity.leverage.ceilings.kelly import KellyCeiling
from convexity.leverage.ceilings.liquidity import LiquidityCeiling
from convexity.leverage.ceilings.margin import MarginCeiling
from convexity.leverage.ceilings.portfolio import PortfolioCeiling
from convexity.leverage.ceilings.stop import StopCeiling
from convexity.leverage.ceilings.tail import TailCeiling
from convexity.leverage.ceilings.vol_target import VolTargetCeiling
from convexity.leverage.combine.governor import DrawdownGovernor
from convexity.leverage.combine.kelly_fraction import AdaptiveKellyFraction
from convexity.leverage.combine.lot_sizer import Quantizer
from convexity.leverage.config.margin import MarginModel
from convexity.leverage.config.resolve import ResolvedConfig
from convexity.leverage.types import (
    CEILING_NAMES,
    BookStateFixture,
    Ceilings,
    LeverageDecision,
    RiskInputs,
    SignalInput,
)

_EPS: float = 1e-12


class LeverageEngine:
    """Composes the seven ceilings into one ``LeverageDecision`` (§1, §2 layer 3)."""

    def __init__(
        self,
        margin_model: MarginModel,
        *,
        stop: StopCeiling | None = None,
        vol: VolTargetCeiling | None = None,
        kelly: KellyCeiling | None = None,
        tail: TailCeiling | None = None,
        portfolio: PortfolioCeiling | None = None,
        margin: MarginCeiling | None = None,
        liquidity: LiquidityCeiling | None = None,
        fraction: AdaptiveKellyFraction | None = None,
        governor: DrawdownGovernor | None = None,
        quantizer: Quantizer | None = None,
    ) -> None:
        self.margin_model = margin_model
        self.stop = stop or StopCeiling()
        self.vol = vol or VolTargetCeiling()
        self.kelly = kelly or KellyCeiling()
        self.tail = tail or TailCeiling()
        self.portfolio = portfolio or PortfolioCeiling()
        self.margin = margin or MarginCeiling()
        self.liquidity = liquidity or LiquidityCeiling()
        self.fraction = fraction or AdaptiveKellyFraction()
        self.governor = governor or DrawdownGovernor()
        self.quantizer = quantizer or Quantizer()

    # ----------------------------------------------------------------- ceiling assembly
    def compute_ceilings(
        self,
        signal: SignalInput,
        risk: RiskInputs,
        cfg: ResolvedConfig,
        book: BookStateFixture,
        *,
        carry: float = 0.0,
        prev_phi: float | None = None,
    ) -> tuple[Ceilings, float, dict]:
        """Compute the seven ceilings and the adaptive Kelly fraction φ (consumed ONCE
        inside L_kelly, Inv F). Returns (Ceilings, phi, combine_diag). ``carry`` is COST's
        SIGNED carry as a fraction of position margin (the §8.1 stop clamp consumes it).

        BookState is the canonical equity/cov source (ARCHITECTURE §5.9): the covariance,
        candidate weights, account-equity basis and free margin are read from ``book``, not
        from ``risk.*`` mirrors — one risk view, never two divergent ones.
        """
        fv = risk.forecast_vol  # horizon-scaled σ·√T already (Inv S)

        # canonical book-owned quantities (ARCHITECTURE §5.9): equity / cov / weights /
        # free margin. The portfolio ceiling and margin ceiling read THESE, not risk mirrors.
        account_equity = float(book.equity)
        free_margin = float(book.free_margin)

        # adaptive fraction first (rule-based, observables only — Inv K)
        phi, phi_reason = self.fraction.fraction(
            risk.vol_percentile, risk.corr_instability, risk.jump_intensity, cfg, prev_phi
        )

        l_stop, d_stop = self.stop.ceiling(
            signal.side, signal.entry_price, signal.capital_cap, carry, fv,
            risk.regime_bucket, risk.cond_mae, cfg,
        )
        stop_distance = float(d_stop["d"])

        l_vol, d_vol = self.vol.ceiling(fv, cfg.sigma_target)

        sigma_hat = fv  # already σ·√T on the holding horizon (Inv S, consistent with stop)
        l_kelly, d_kelly = self.kelly.ceiling(risk.mu_hat, sigma_hat, stop_distance, phi, cfg)

        l_tail, d_tail = self.tail.ceiling(
            risk.loss_tail, risk.gap_hist, cfg.defaults.es_budget, cfg.defaults.es_level, cfg
        )

        l_port, d_port = self.portfolio.ceiling(
            signal.instrument, book.candidate_weights, book.cov, book.cov_index, risk.g_t,
            signal.position_margin, account_equity, cfg,
        )

        horizon_years = max(risk.holding_horizon_periods, 0.0) / 252.0  # periods -> years proxy
        l_margin, d_marg = self.margin.ceiling(
            signal.instrument, self.margin_model, free_margin, risk.financing_rate,
            horizon_years, signal.position_margin, signal.entry_price, cfg,
        )

        l_liq, d_liq = self.liquidity.ceiling(
            risk.liquidity, cfg.defaults.target_exit_minutes, signal.position_margin, cfg
        )

        diag: dict[str, object] = {
            "stop": d_stop, "vol": d_vol, "kelly": d_kelly, "tail": d_tail,
            "portfolio": d_port, "margin": d_marg, "liquidity": d_liq,
            "phi_reason": phi_reason, "stop_distance": stop_distance,
            "portfolio_reason_code": d_port.get("reason_code"),
        }
        ceilings = Ceilings(
            L_stop=l_stop, L_vol=l_vol, L_kelly=l_kelly, L_tail=l_tail,
            L_portfolio=l_port, L_margin=l_margin, L_liquidity=l_liq,
            g_t=float(risk.g_t), diag=diag,
        )
        return ceilings, phi, diag

    # ----------------------------------------------------------------- the equation
    def decide(
        self,
        signal: SignalInput,
        risk: RiskInputs,
        cfg: ResolvedConfig,
        book: BookStateFixture,
        *,
        carry: float = 0.0,
        prev_phi: float | None = None,
    ) -> LeverageDecision:
        """The §1 equation end-to-end. Deterministic given inputs (Inv H/B/F/U/R/G)."""
        ceilings, phi, diag = self.compute_ceilings(
            signal, risk, cfg, book, carry=carry, prev_phi=prev_phi
        )

        # min() of the seven ceilings + the HARD class-cap clamp (Inv B/CLASSCAP)
        soft = ceilings.as_tuple()
        l_class_max = cfg.l_class_max
        clamped = tuple(min(s, l_class_max) for s in soft)
        l_pre = min(clamped)

        # which ceiling binds (the seven names, then class cap)
        binding_idx = min(range(len(clamped)), key=lambda i: clamped[i])
        binding = CEILING_NAMES[binding_idx]
        reason_codes: list[str] = []
        # the class cap binds when it is strictly below the tightest soft ceiling (§8.6.6)
        if l_class_max < min(soft) - _EPS:
            reason_codes.append("class_cap_capped")
        if binding == "portfolio":
            rc = diag["portfolio"].get("reason_code")
            if isinstance(rc, str):
                reason_codes.append(rc)

        # drawdown governor — applied EXACTLY ONCE on the disjoint equity axis (Inv G/P8).
        # Reads the CANONICAL book equity/peak (ARCHITECTURE §5.9), not a risk.* mirror.
        gov_mult, gov_reason = self.governor.multiplier(
            book.peak_equity, book.equity, cfg
        )
        l_final = l_pre * gov_mult
        if gov_mult < 1.0 - _EPS:
            reason_codes.append("governor_throttled")
        if gov_mult <= _EPS:
            binding = "governor"

        # continuous notional implied, then lot quantization (Inv U)
        raw_notional = l_final * signal.position_margin
        # the leverage ceiling itself, in continuous units, caps the quantizer
        unit_notional = max(
            float(cfg.instrument.contract_multiplier) * float(signal.entry_price), _EPS
        )
        ceiling_units = (l_pre * signal.position_margin) / unit_notional
        class_cap_units = (l_class_max * signal.position_margin) / unit_notional

        # Margin ALREADY consumed by the OPEN book (§8.6.3): for a margin account the used
        # margin is equity − free_margin. Incumbents are FIXED (SYSTEM P4); the quantizer
        # subtracts this off the top so the post-rounding free-margin buffer is computed on
        # the equity AFTER the existing book, never on gross free margin. All book-owned.
        existing_margin_used = max(float(book.equity) - float(book.free_margin), 0.0)

        qsize = self.quantizer.quantize(
            raw_leverage=l_final,
            raw_notional=raw_notional,
            instrument=cfg.instrument,
            position_margin=signal.position_margin,
            account_equity=float(book.equity),
            entry_price=signal.entry_price,
            stop_distance=diag["stop_distance"],
            capital_cap=signal.capital_cap,
            margin_model=self.margin_model,
            free_margin=float(book.free_margin),
            existing_margin_used=existing_margin_used,
            cfg=cfg,
            ceiling_implied_lots=ceiling_units,
            class_cap_lots=class_cap_units,
        )
        reason_codes.extend(qsize.flags)
        if qsize.units == 0:
            binding = "unit"

        return LeverageDecision(
            instrument=signal.instrument,
            leverage=float(l_final),
            size=float(raw_notional),
            rounded_size=float(qsize.notional),
            binding_ceiling=binding,  # type: ignore[arg-type]
            kelly_fraction=float(phi),
            governor_multiplier=float(gov_mult),
            g_t=float(risk.g_t),
            risk_budget=float(ceilings.L_kelly),
            planned_loss_on_margin=float(qsize.realized_loss_pct_margin),
            planned_loss_on_equity=float(qsize.realized_loss_pct_equity),
            reason_codes=tuple(dict.fromkeys(reason_codes)),  # dedupe, order-preserving
            engine_version=cfg.defaults.engine_version,
        )
