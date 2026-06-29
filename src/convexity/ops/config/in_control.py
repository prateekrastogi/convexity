"""The FROZEN validated in-control distribution params, loaded from a SYSTEM-report FIXTURE.

Spec: OPS §4 (the RED in-control reference), §5 (per-stream ``(μ0,σ0,φ,ρ,dof,quantile)``), §11.1
(derived once per validated config from SYSTEM's validation report, frozen, never live). This module
loads the per-stream :class:`InControlParams` *from a SYSTEM validation report* — a plain mapping
(JSON-shaped) the SYSTEM pillar emits at promotion time. **It does NOT import the real ``system``
module** so OPS stays independently testable (the build directive): the contract is the report
*shape*, validated here, not a live SYSTEM object.

Invariant LIM/STREAM: every field traces to a report field, never a live rolling estimate (except the
diagnostic-only ``σ_live`` of §4 TEAL, which lives in ``streams/`` and never enters a limit). A report
missing a stream, or carrying a non-positive ``σ0``, is rejected here — a silent default would smuggle
an un-validated limit into the panel.
"""

from __future__ import annotations

from typing import Any, Mapping

from convexity.ops.types import InControlParams, StreamName

# The statistical streams that require an in-control reference (the binary streams recon/heartbeat and
# the equity read carry no σ0 — they are not standardized, §5.5/§5.6/§7.1).
STATISTICAL_STREAMS: tuple[StreamName, ...] = ("fills", "cost_tca", "vol", "edge")


class SystemReportError(ValueError):
    """A SYSTEM in-control report is missing a required stream or carries an invalid parameter."""


def _require(d: Mapping[str, Any], key: str, stream: str) -> Any:
    if key not in d:
        raise SystemReportError(f"in-control report for stream {stream!r} missing field {key!r}")
    return d[key]


def in_control_from_report(report: Mapping[str, Mapping[str, Any]]) -> dict[StreamName, InControlParams]:
    """Build the frozen per-stream :class:`InControlParams` from a SYSTEM validation report (§11.1).

    ``report`` is ``{stream_name: {mu0, sigma0, phi, rho, tail_dof, tail_quantile, p0, p1}}`` — the
    SYSTEM-report fixture shape. The ``edge`` stream carries the validated hit-rate ``(p0, p1)`` in
    place of ``(μ0, σ0)`` for its SPRT arm but ALSO carries ``(μ0, σ0)`` for the CUSUM-on-e_edge arm.
    The ``cost_tca`` stream ALSO carries ``(mu0_fund, sigma0_fund)`` — the e_funding arm's own frozen
    reference, so the §7.8 funding CUSUM standardizes against unit-variance input like every other
    stream (Invariant STREAM), not raw bps. Validates presence + positivity; rejects an un-validated
    stream rather than defaulting (LIM).
    """
    out: dict[StreamName, InControlParams] = {}
    for stream in STATISTICAL_STREAMS:
        if stream not in report:
            raise SystemReportError(f"in-control report missing required stream {stream!r}")
        d = report[stream]
        mu0 = float(_require(d, "mu0", stream))
        sigma0 = float(_require(d, "sigma0", stream))
        if sigma0 <= 0.0:
            raise SystemReportError(f"stream {stream!r}: sigma0 must be > 0 (got {sigma0})")
        phi = tuple(float(x) for x in d.get("phi", ()))
        rho = float(d.get("rho", 0.0))
        if not (-1.0 < rho < 1.0):
            raise SystemReportError(f"stream {stream!r}: rho must be in (-1,1) (got {rho})")
        # The cost_tca report carries a SECOND in-control reference for its e_funding arm (§5.2): the
        # funding CUSUM standardizes against (mu0_fund, sigma0_fund), NOT the cost (mu0, sigma0), so its
        # ARL0 is the pinned one (Invariant STREAM). Required for cost_tca; rejected if σ0_fund ≤ 0.
        mu0_fund: float | None = None
        sigma0_fund: float | None = None
        if stream == "cost_tca":
            mu0_fund = float(_require(d, "mu0_fund", stream))
            sigma0_fund = float(_require(d, "sigma0_fund", stream))
            if sigma0_fund <= 0.0:
                raise SystemReportError(
                    f"stream {stream!r}: sigma0_fund must be > 0 (got {sigma0_fund})"
                )
        out[stream] = InControlParams(
            stream=stream,
            mu0=mu0,
            sigma0=sigma0,
            phi=phi,
            rho=rho,
            tail_dof=(None if d.get("tail_dof") is None else float(d["tail_dof"])),
            tail_quantile=(None if d.get("tail_quantile") is None else float(d["tail_quantile"])),
            p0=(None if d.get("p0") is None else float(d["p0"])),
            p1=(None if d.get("p1") is None else float(d["p1"])),
            mu0_fund=mu0_fund,
            sigma0_fund=sigma0_fund,
        )
    return out


def fixture_system_report() -> dict[StreamName, dict[str, Any]]:
    """A canonical SYSTEM-report FIXTURE for OPS unit tests (the §10 worked-example values).

    ``cost_tca`` carries the §10 ``μ0=0.0, σ0=2.0`` so the worked-example fixture reproduces bit-for-
    bit. ``ρ`` and ``tail_dof`` are the §6.7 dependence/fat-tail params the AR(1)-Student-t synthetic
    is generated against. This stands in for SYSTEM's real report so OPS is testable in isolation
    (the build directive: do NOT import the real system module).
    """
    return {
        "cost_tca": {
            "mu0": 0.0, "sigma0": 2.0, "phi": (0.4,), "rho": 0.4,
            "tail_dof": 5.0, "tail_quantile": 3.0,
            # e_funding's own in-control reference (§5.2): signed-bps, distinct from the cost residual.
            "mu0_fund": 0.0, "sigma0_fund": 1.0,
        },
        "fills": {
            "mu0": 0.0, "sigma0": 1.5, "phi": (0.3,), "rho": 0.3,
            "tail_dof": 6.0, "tail_quantile": 3.0,
        },
        "vol": {
            "mu0": 0.0, "sigma0": 0.25, "phi": (0.5,), "rho": 0.5,
            "tail_dof": 5.0, "tail_quantile": 3.0,
        },
        "edge": {
            "mu0": 0.0, "sigma0": 1.0, "phi": (), "rho": 0.0,
            "tail_dof": 4.0, "tail_quantile": 3.2, "p0": 0.55, "p1": 0.45,
        },
    }


__all__ = [
    "STATISTICAL_STREAMS",
    "SystemReportError",
    "in_control_from_report",
    "fixture_system_report",
]
