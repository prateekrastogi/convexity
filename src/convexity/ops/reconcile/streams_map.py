"""Bind monitored streams to detectors with the §6 limits (the reconciliation layer, §2.3).

Spec: OPS §2 (layer 3), §5 (which detectors police which stream), §6 (the limits). This module is the
explicit binding: each statistical stream → its detector(s) constructed from the FROZEN
:class:`ResolvedOpsConfig` limits + its in-control reference + a prewhitener (§6.7a). No detector reads
another's output (architectural isolation); the binding is the only place a stream meets a detector.

Per §5:
  fills    : CUSUM + EWMA on e_fill; Shewhart on single catastrophic fills.
  cost_tca : CUSUM on e_cost confirmed by Page-Hinkley (2-of-2, §6.6a); CUSUM on e_funding.
  vol      : EWMA + CUSUM on e_vol.
  edge     : CUSUM-SPRT on the hit-rate LLR (§6.5a v1 default) + CUSUM on e_edge + Shewhart on freq.
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.config.resolve import ResolvedOpsConfig
from convexity.ops.detectors.cusum import Cusum
from convexity.ops.detectors.ewma import Ewma
from convexity.ops.detectors.page_hinkley import PageHinkley
from convexity.ops.detectors.shewhart import Shewhart
from convexity.ops.detectors.sprt import CusumSprt
from convexity.ops.streams.whiten import CausalPrewhitener
from convexity.ops.types import InControlParams


@dataclass
class StreamBinding:
    """The detectors + the causal prewhitener bound to one stream (§2.3/§5).

    ``prewhiten`` is the §6.7a innovation filter (empty ``φ`` degrades to plain standardization). The
    detectors are the §5 charts for this stream; the brake's triggers read their states.
    """

    prewhiten: CausalPrewhitener
    detectors: dict[str, object]      # detector-role -> Detector instance
    # The cost_tca stream's e_funding arm has its OWN prewhitener (§5.2/§6.7a): the funding residual is
    # a distinct signed-bps distribution and MUST be standardized against its own frozen
    # (mu0_fund, sigma0_fund) before the §7.8 CUSUM — routing it through the same prewhitener machinery
    # as every other stream so the funding CUSUM sees unit-variance input (Invariant STREAM). None on
    # streams without a funding arm.
    funding_prewhiten: CausalPrewhitener | None = None

    def standardize(self, residual: float) -> float:
        """Causally prewhiten a raw residual to a standardized innovation (§6.7a)."""
        return self.prewhiten.update(residual)

    def standardize_funding(self, residual: float) -> float:
        """Standardize the e_funding residual against its OWN frozen reference (§5.2/Invariant STREAM).

        Routes through the funding prewhitener (empty φ ⇒ plain ``(x-mu0_fund)/sigma0_fund``) so the
        §7.8 CUSUM sees a unit-variance input, exactly like e_cost/e_vol/e_fill/e_edge.
        """
        if self.funding_prewhiten is None:  # defensive: cost_tca always has one (build_bindings)
            return self.prewhiten.update(residual)
        return self.funding_prewhiten.update(residual)


def _ic(cfg: ResolvedOpsConfig, stream: str) -> InControlParams:
    return cfg.in_control[stream]  # type: ignore[index]


def _funding_ic(ic_cost: InControlParams) -> InControlParams:
    """The e_funding arm's OWN in-control reference (§5.2/Invariant STREAM).

    The funding residual is a distinct signed-bps distribution from the round-trip cost residual, so it
    standardizes against its own ``(mu0_fund, sigma0_fund)``. We build a plain (φ=()) prewhitener on
    that reference so the §7.8 CUSUM sees unit-variance input — never the raw bps it saw before. Falls
    back to the cost ``(mu0, sigma0)`` only if the report omitted the funding fields (defensive; the
    loader requires them for cost_tca)."""
    mu0_fund = ic_cost.mu0_fund if ic_cost.mu0_fund is not None else ic_cost.mu0
    sigma0_fund = ic_cost.sigma0_fund if ic_cost.sigma0_fund is not None else ic_cost.sigma0
    return InControlParams(stream="cost_tca", mu0=mu0_fund, sigma0=sigma0_fund)


def build_bindings(cfg: ResolvedOpsConfig) -> dict[str, StreamBinding]:
    """Construct the per-stream detector bindings from the frozen config (§5/§6).

    Limits come from ``cfg.defaults`` (RED, frozen); the in-control reference + ``φ`` from
    ``cfg.in_control`` (the SYSTEM report). Returns a mapping the engine iterates over per bar.
    """
    d = cfg.defaults
    bindings: dict[str, StreamBinding] = {}

    # --- fills (§5.1): CUSUM + EWMA on e_fill; Shewhart on catastrophic single fills ---
    ic_fills = _ic(cfg, "fills")
    bindings["fills"] = StreamBinding(
        prewhiten=CausalPrewhitener(ic_fills),
        detectors={
            "cusum": Cusum(d.cusum.k, d.cusum.h, n_warm=d.n_warm_cusum,
                           escalation_h=d.cusum.h2),
            "ewma": Ewma(d.ewma.lam, d.ewma.L, n_warm=d.n_warm_ewma),
            "shewhart": Shewhart(_shewhart_ls(d, ic_fills), n_warm=d.n_warm_shewhart),
        },
    )

    # --- cost_tca (§5.2): CUSUM on e_cost (centerpiece) + Page-Hinkley confirmation (2-of-2) ---
    ic_cost = _ic(cfg, "cost_tca")
    bindings["cost_tca"] = StreamBinding(
        prewhiten=CausalPrewhitener(ic_cost),
        funding_prewhiten=CausalPrewhitener(_funding_ic(ic_cost)),
        detectors={
            "cusum": Cusum(d.cusum.k, d.cusum.h, n_warm=d.n_warm_cusum,
                           escalation_h=d.cusum.h2),
            "page_hinkley": PageHinkley(d.page_hinkley.delta, d.page_hinkley.lam,
                                        n_warm=d.n_warm_page_hinkley),
            "funding_cusum": Cusum(d.cusum.k, d.cusum.h, n_warm=d.n_warm_cusum),
        },
    )

    # --- vol (§5.3): EWMA (the right smoother for autocorr) + CUSUM ---
    ic_vol = _ic(cfg, "vol")
    bindings["vol"] = StreamBinding(
        prewhiten=CausalPrewhitener(ic_vol),
        detectors={
            "ewma": Ewma(d.ewma.lam, d.ewma.L, n_warm=d.n_warm_ewma),
            "cusum": Cusum(d.cusum.k, d.cusum.h, n_warm=d.n_warm_cusum),
        },
    )

    # --- edge (§5.4): CUSUM-SPRT on hit-rate LLR (v1 default) + CUSUM on e_edge + Shewhart on freq ---
    ic_edge = _ic(cfg, "edge")
    p0 = ic_edge.p0 if ic_edge.p0 is not None else 0.55
    p1 = ic_edge.p1 if ic_edge.p1 is not None else 0.45
    bindings["edge"] = StreamBinding(
        prewhiten=CausalPrewhitener(ic_edge),
        detectors={
            "cusum_sprt": CusumSprt(p0, p1, d.sprt.h_edge, n_warm=d.n_warm_sprt),
            "edge_cusum": Cusum(d.cusum.k, d.cusum.h, n_warm=d.n_warm_cusum),
            "freq_shewhart": Shewhart(_shewhart_ls(d, ic_edge), n_warm=d.n_warm_shewhart),
        },
    )

    return bindings


def _shewhart_ls(defaults, ic: InControlParams) -> float:
    """The empirical-tail Shewhart ``L_s`` (§6.7c): the in-control ``tail_quantile`` when present,
    else the Gaussian-3σ reference. NOT the Gaussian limit on a fat-tailed stream."""
    if ic.tail_quantile is not None:
        return ic.tail_quantile
    return defaults.shewhart.L_s


__all__ = ["StreamBinding", "build_bindings"]
