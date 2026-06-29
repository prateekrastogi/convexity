"""PromotionRegistry — config hashes / windows / selected presets (TREND §15 registry).

Every trial must be recorded with a config hash + data window (§15) so the multiple-testing
deflation (validate/stats.py / SYSTEM DSR) counts every searched config. This is an in-memory record
store; persistence is a SYSTEM concern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from convexity.trend.config.resolve import ResolvedConfig
from convexity.trend.tune.promotion import PromotionDecision


def config_hash(cfg: ResolvedConfig) -> str:
    """Stable hash of the resolved RED/preset config (the trial identity, §15)."""
    payload = {
        "defaults": asdict(cfg.defaults),
        "preset": asdict(cfg.preset),
        "tau_cap": cfg.tau_cap,
        "h_obs": cfg.h_obs,
        "active_clock": cfg.active_clock,
        "trail_multipliers": list(cfg.trail_multipliers),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PromotionRegistry:
    """In-memory trial/promotion record store (§15)."""

    def __init__(self) -> None:
        self._records: list[PromotionDecision] = []

    def record(self, decision: PromotionDecision) -> None:
        self._records.append(decision)

    def n_trials(self) -> int:
        """The trial count fed into the multiple-testing penalty."""
        return len(self._records)

    def approved(self) -> list[PromotionDecision]:
        return [r for r in self._records if r.status == "approved"]
