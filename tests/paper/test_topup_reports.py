"""Venue metadata fetching (topup) and the COST §12 validation reports.

topup's HTTP functions are faked at the httpx boundary so the real response parsing runs —
including the precision clamp that caused a live -1111 rejection. The report functions are pure
and were at 0%; every one of them is asked for a summary on an EMPTY input somewhere in the
validation pipeline, so degrading instead of raising is the property that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexity.cost.validate.reports import (
    forecast_error,
    gate_calibration,
    tca_residual_distribution,
)
from convexity.paper import topup
from convexity.paper.topup import effective_filters, last_completed_day_ms

MS_PER_DAY = 86_400_000


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Stands in for httpx.Client as a context manager."""

    def __init__(self, payload):
        self._payload = payload
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, *a, **kw):
        self.urls.append(url)
        return FakeResponse(self._payload)


def install(monkeypatch, payload) -> FakeClient:
    holder = {}

    def factory(*a, **kw):
        c = FakeClient(payload)
        holder["client"] = c
        return c

    monkeypatch.setattr(topup.httpx, "Client", factory)
    return holder


# ---------------------------------------------------------------- precision clamp
def test_effective_filters_is_a_noop_when_precision_is_coarser_than_the_grid():
    assert effective_filters(0.01, 0.001, 2, 3) == (0.01, 0.001)


def test_effective_filters_takes_the_COARSER_of_grid_and_precision():
    """Binance enforces BOTH the grid and the decimal cap; the effective step is the coarser."""
    tick, lot = effective_filters(0.0001, 0.0001, 2, 1)
    assert tick == pytest.approx(0.01)
    assert lot == pytest.approx(0.1)


def test_effective_filters_pins_the_SPCX_whole_share_case():
    """REGRESSION: demo SPCXUSDT has quantityPrecision=0 (whole shares) while the stale filter
    source implied a fractional lot — a fractional qty drew -1111 in the equity book's first
    supervised session."""
    _tick, lot = effective_filters(0.01, 0.001, 2, 0)
    assert lot == pytest.approx(1.0)


def test_effective_filters_leaves_the_grid_alone_when_precision_is_absent():
    assert effective_filters(0.05, 0.5, None, None) == (0.05, 0.5)


# ---------------------------------------------------------------- exchangeInfo parsing
def test_live_filters_parse_and_clamp(monkeypatch):
    install(monkeypatch, {"symbols": [
        {"symbol": "BTCUSDT", "pricePrecision": 2, "quantityPrecision": 3,
         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]},
    ]})
    out = topup.fetch_live_filters()
    assert out["BTCUSDT"] == (pytest.approx(0.10), pytest.approx(0.001),
                              pytest.approx(0.001), pytest.approx(5.0))


def test_live_filters_default_min_notional_when_the_venue_omits_it(monkeypatch):
    install(monkeypatch, {"symbols": [
        {"symbol": "XUSDT", "pricePrecision": 2, "quantityPrecision": 2,
         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                     {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"}]},
    ]})
    assert topup.fetch_live_filters()["XUSDT"][3] == pytest.approx(5.0)


def test_live_filters_skip_a_symbol_missing_a_required_filter(monkeypatch):
    """One malformed symbol must not take down the whole filter map for a live session."""
    install(monkeypatch, {"symbols": [
        {"symbol": "GOODUSDT", "pricePrecision": 2, "quantityPrecision": 2,
         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                     {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"}]},
        {"symbol": "BADUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}]},
    ]})
    out = topup.fetch_live_filters()
    assert "GOODUSDT" in out and "BADUSDT" not in out


def test_ticker_prices_and_book_tickers_parse(monkeypatch):
    install(monkeypatch, [{"symbol": "BTCUSDT", "price": "60000.5"}])
    assert topup.fetch_ticker_prices() == {"BTCUSDT": pytest.approx(60000.5)}

    install(monkeypatch, [{"symbol": "BTCUSDT", "bidPrice": "59999", "askPrice": "60001"}])
    assert topup.fetch_book_tickers()["BTCUSDT"] == (pytest.approx(59999.0), pytest.approx(60001.0))


def test_tradeable_gate_admits_only_status_TRADING(monkeypatch):
    """The gate that stops the fleet ordering a delisted or halted symbol."""
    install(monkeypatch, {"symbols": [
        {"symbol": "LIVEUSDT", "status": "TRADING"},
        {"symbol": "DEADUSDT", "status": "SETTLING"},
        {"symbol": "GONEUSDT", "status": "BREAK"},
    ]})
    tr = topup.fetch_execution_venue_tradeable()
    assert "LIVEUSDT" in tr
    assert "DEADUSDT" not in tr and "GONEUSDT" not in tr


# ---------------------------------------------------------------- day boundary
def test_last_completed_day_is_never_the_running_day():
    now = 3 * MS_PER_DAY + 12 * 3_600_000       # midday on day 3
    assert last_completed_day_ms(now) == 2 * MS_PER_DAY


def test_last_completed_day_at_exact_midnight_steps_back_one_full_day():
    assert last_completed_day_ms(5 * MS_PER_DAY) == 4 * MS_PER_DAY


# ---------------------------------------------------------------- §12 reports
def test_forecast_error_reports_signed_bias_and_rmse():
    rep = forecast_error(np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 2.0]))
    assert rep.n == 3
    assert rep.bias == pytest.approx(2.0 / 3.0)          # over-forecasting
    assert rep.rmse == pytest.approx(np.sqrt(2.0 / 3.0))


def test_forecast_error_of_a_perfect_forecast_is_zero():
    rep = forecast_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert rep.bias == 0.0 and rep.rmse == 0.0


def test_forecast_error_on_empty_input_degrades():
    rep = forecast_error(np.array([]), np.array([]))
    assert (rep.bias, rep.rmse, rep.n) == (0.0, 0.0, 0)


def test_gate_calibration_splits_admitted_from_rejected():
    admitted = np.array([True, True, False, False])
    net_pos = np.array([True, False, True, True])
    out = gate_calibration(admitted, net_pos)
    assert out["admitted_net_positive_rate"] == pytest.approx(0.5)
    assert out["rejected_net_positive_rate"] == pytest.approx(1.0)


def test_gate_calibration_is_nan_not_a_crash_when_a_side_is_empty():
    out = gate_calibration(np.array([False, False]), np.array([True, False]))
    assert np.isnan(out["admitted_net_positive_rate"])
    assert not np.isnan(out["rejected_net_positive_rate"])


def test_tca_residuals_summarise_magnitude_and_tolerance_breaches():
    out = tca_residual_distribution(np.array([0.1, -0.5, 2.0]), tolerance=1.0)
    assert out["max_abs_residual"] == pytest.approx(2.0)
    assert out["mean_abs_residual"] == pytest.approx((0.1 + 0.5 + 2.0) / 3.0)
    assert out["frac_over_tolerance"] == pytest.approx(1 / 3)


def test_tca_residuals_on_empty_input_degrade():
    out = tca_residual_distribution(np.array([]), tolerance=1.0)
    assert out == {"max_abs_residual": 0.0, "mean_abs_residual": 0.0, "frac_over_tolerance": 0.0}
