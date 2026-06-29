"""Limit-derivation + ARL/false-alarm reports (§11). Part of the MODEL — never imported by live layers.

Spec: OPS §11, §13 (ARL/limits acceptance gate). Assembles the per-stream limit table with its pinned
ARL0/ARL1 (Invariant ARL records both + the panel false-alarm contribution), the analytic-vs-MC
cross-check, and the source-counted panel budget — the artefact the OOS gate (§11.2) promotes against.
``scipy``/MC live here, never in the engine closure (Invariant V).
"""

from __future__ import annotations

from dataclasses import dataclass

from convexity.ops.brake.alarms import expected_samples_between_pages, panel_false_page_rate


@dataclass(frozen=True)
class StreamLimitRow:
    """One stream's pinned limit + its ARL0/ARL1 + its panel-source ARL0 (Invariant ARL)."""

    stream: str
    detector: str
    arl0: float
    arl1_at_design: float
    panel_source_arl0: float   # the ARL0 this source contributes to Σ_d(1/ARL0,d) (§6.6)


@dataclass(frozen=True)
class PanelReport:
    """The full panel budget report (§6.6/§8.3). The paging cadence is set from this source-counted
    rate, NOT the raw stream/detector count."""

    rows: tuple[StreamLimitRow, ...]
    panel_false_page_rate: float
    samples_between_pages: float

    def n_independent_sources(self) -> int:
        """The number of INDEPENDENT alarm sources (AND-pairs counted once, §6.6) — i.e. the rows."""
        return len(self.rows)


def build_panel_report(rows: list[StreamLimitRow]) -> PanelReport:
    """Assemble the panel report + the source-counted false-page budget ``Σ_d(1/ARL0,d)`` (§6.6/§8.3).

    Each row is ONE independent source (a single detector, or a 2-of-2 AND-pair at its joint ARL0, or
    the edge stream at its §6.5a ARL0-equivalent). The panel false-page rate is the sum of ``1/ARL0``
    over the rows; the expected samples between any false page is its reciprocal.
    """
    source_arl0s = [r.panel_source_arl0 for r in rows]
    return PanelReport(
        rows=tuple(rows),
        panel_false_page_rate=panel_false_page_rate(source_arl0s),
        samples_between_pages=expected_samples_between_pages(source_arl0s),
    )


__all__ = ["StreamLimitRow", "PanelReport", "build_panel_report"]
