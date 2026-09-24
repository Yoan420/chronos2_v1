"""Retained cache-helper contracts and their exclusion from the paper report."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
import math

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import model_storm_rolling as rolling


@pytest.fixture(autouse=True)
def small_windows_and_no_storage_solver(monkeypatch):
    monkeypatch.setattr(rolling, "WINDOWS", (1,))
    monkeypatch.setattr(rolling, "optimize_storage_schedule", lambda forecast: None)


def payload(day="2026-09-19", days=1, timezone="Europe/Paris"):
    rows = []
    for offset in reversed(range(days)):
        current = (date.fromisoformat(day) - timedelta(days=offset)).isoformat()
        for index, stamp in enumerate(rolling._hours(current, timezone)):
            rows.append({"timestamp_utc": stamp.isoformat(), "observed": 100.0 + index,
                         "storm": 102.0 + index, "storm_dashboard_cache": 103.0 + index,
                         "model": 96.0 + index})
    return {"delivery_day": day, "zones": [{"zone": "FR", "name": "France", "timezone": timezone,
            "rolling_history": {"rows": rows, "sources": {"storm_dashboard_cache": {
                "status": "complete", "exact_cache_hour_mask_verified": True,
                "official_dashboard_formula_verified": False}}}}]}


def dashboard(data):
    return rolling._dashboard_zone(data["zones"][0], data["delivery_day"])


def table(data, frequency="60min", window="1"):
    return dashboard(data)["windows"][window]["frequencies"][frequency]


def by_provider(result, provider):
    return next(row for row in result["providers"] if row["key"] == provider)


def test_dashboard_denominator_includes_missing_forecast_and_observation_slots():
    result = rolling.score_dashboard_forecast([0, 10, 20, np.nan], [1, np.nan, 26, 0])
    assert result["samples"] == 2
    assert result["denominator_samples"] == 4
    assert result["mae"] == 3.5
    assert result["bias"] == 3.5
    assert result["rmse"] == pytest.approx(math.sqrt(37 / 2))
    assert result["hit_rate"] == .25
    # The finite observation 10 contributes to SST despite its missing forecast.
    assert result["r2"] == pytest.approx(1 - 37 / 200)


def test_hourly_prices_are_rounded_to_cents_before_scoring():
    result = rolling.score_dashboard_forecast([0, 10, 20, 30], [5.004, 15.006, 14.996, 24.994])
    assert result["samples"] == result["denominator_samples"] == 4
    assert result["hit_rate"] == .5
    assert result["mae"] == pytest.approx(5.005)
    assert result["rmse"] == pytest.approx(math.sqrt((5.00 ** 2 + 5.01 ** 2) / 2))
    internal = rolling.score_forecast([0, 10, 20, 30], [5.004, 15.006, 14.996, 24.994])
    assert internal["hit_rate"] == 0


def test_r_squared_reference_includes_all_finite_observations():
    result = rolling.score_dashboard_forecast([0, 10, 100], [1, 11, np.nan])
    observed = np.array([0, 10, 100], dtype=float)
    expected = 1 - 2 / np.sum((observed - observed.mean()) ** 2)
    assert result["r2"] == pytest.approx(expected)
    assert result["r2"] != pytest.approx(rolling.score_forecast([0, 10], [1, 11])["r2"])
    assert result["hit_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("actual,predicted", [([0, 0], [1, np.nan]), ([.1] * 24, [.2] * 24)])
def test_constant_observed_reference_has_no_r_squared(actual, predicted):
    result = rolling.score_dashboard_forecast(actual, predicted)
    assert result["r2"] is None
    assert result["samples"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("actual,predicted", [([], []), ([0, 10], [np.nan, np.nan]),
                                             ([np.nan, np.nan], [0, 10]),
                                             ([0, np.nan], [np.nan, 10])])
def test_no_valid_samples_are_missing_not_fabricated_zero(actual, predicted):
    result = rolling.score_dashboard_forecast(actual, predicted)
    assert result["samples"] == 0
    assert result["denominator_samples"] == len(actual)
    for metric in ("mae", "bias", "rmse", "hit_rate", "r2"):
        assert result[metric] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("actual,predicted", [([0], []), ([[0]], [[1]]),
                                             ([np.inf], [1]), ([0], [-np.inf])])
def test_score_rejects_malformed_shapes_and_infinite_prices(actual, predicted):
    with pytest.raises(ValueError):
        rolling.score_dashboard_forecast(actual, predicted)


@pytest.mark.parametrize("source", [{}, {"status": "complete"},
    {"status": "invalid", "exact_cache_hour_mask_verified": True},
    {"status": "complete", "exact_cache_hour_mask_verified": False},
    {"status": "complete", "exact_cache_hour_mask_verified": 1}])
def test_invalid_cache_provenance_fails_closed_even_when_rows_have_values(source):
    data = payload()
    data["zones"][0]["rolling_history"]["sources"]["storm_dashboard_cache"] = source
    result = dashboard(data)
    assert result["history_status"] == "cache_provenance_unavailable"
    section = result["windows"]["1"]
    assert section["storm_hours"] == section["model_hours"] == 0
    assert section["cache_missing_hours"] == section["expected_hours"] == 24
    for frequency in ("60min", "day"):
        for provider in section["frequencies"][frequency]["providers"]:
            assert provider["samples"] == 0
            assert provider["mae"] is provider["daily_pnl"] is None
    assert all(row["storm"] is not None for row in data["zones"][0]["rolling_history"]["rows"])


def test_missing_model_hour_does_not_change_storm_numbers_and_disables_comparison():
    data = payload()
    baseline = by_provider(table(data), "storm")
    data["zones"][0]["rolling_history"]["rows"][0]["model"] = None
    result = table(data)
    storm = by_provider(result, "storm")
    for field in ("samples", "denominator_samples", "mae", "bias", "rmse", "hit_rate", "r2"):
        assert storm[field] == baseline[field]
    assert result["storm_samples"] == 24 and result["model_samples"] == 23
    assert result["same_support"] is False
    assert all(provider["comparison_eligible"] is False for provider in result["providers"])


def test_completely_missing_model_still_keeps_storm_metrics():
    data = payload()
    for row in data["zones"][0]["rolling_history"]["rows"]:
        row["model"] = None
    result = table(data)
    assert by_provider(result, "storm")["samples"] == 24
    assert by_provider(result, "storm")["mae"] == 3
    assert by_provider(result, "model")["samples"] == 0
    assert by_provider(result, "model")["mae"] is None


def test_cache_missing_hours_exclude_model_but_never_use_merged_fallback():
    data = payload()
    rows = data["zones"][0]["rolling_history"]["rows"]
    rows[0]["storm_dashboard_cache"] = None
    rows[0]["storm"] = 99999
    rows[0]["model"] = -99999
    result = table(data)
    assert result["same_support"] is True
    assert result["storm_samples"] == result["model_samples"] == 23
    assert by_provider(result, "storm")["mae"] == 3
    assert by_provider(result, "model")["mae"] == 4
    assert by_provider(result, "storm")["hit_rate"] == pytest.approx(23 / 24)


def test_daily_frequency_uses_mean_available_prices_and_rejects_partial_model_support():
    data = payload()
    rows = data["zones"][0]["rolling_history"]["rows"]
    for row in rows[-4:]:
        row["storm_dashboard_cache"] = None
    # Observed mean=111.5; Storm mean over first 20 hours=112.5.
    # Model mean on that same cache support=105.5.
    result = table(data, "day")
    assert result["same_support"] is True
    assert result["denominator_samples"] == 1
    assert by_provider(result, "storm")["bias"] == 1
    assert by_provider(result, "model")["bias"] == -6
    rows[0]["model"] = None
    partial = table(data, "day")
    assert by_provider(partial, "storm")["bias"] == 1
    assert by_provider(partial, "storm")["samples"] == 1
    assert by_provider(partial, "model")["samples"] == 0
    assert by_provider(partial, "model")["mae"] is None
    assert partial["same_support"] is False
    assert all(not provider["comparison_eligible"] for provider in partial["providers"])


def test_daily_means_preserve_subcent_precision_after_hourly_rounding():
    data = payload()
    for index, row in enumerate(data["zones"][0]["rolling_history"]["rows"]):
        row["observed"] = 100.004
        row["storm_dashboard_cache"] = 101.004 if index < 12 else 101.014
    result = by_provider(table(data, "day"), "storm")
    assert result["bias"] == pytest.approx(1.005)
    assert result["mae"] == pytest.approx(1.005)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dashboard_grid_preserves_distinct_physical_dst_hours(day, hours):
    data = payload(day)
    rows = data["zones"][0]["rolling_history"]["rows"]
    assert len({row["timestamp_utc"] for row in rows}) == hours
    if hours == 25:
        repeated = [row for row in rows if pd.Timestamp(row["timestamp_utc"]).tz_convert("Europe/Paris").hour == 2]
        assert len(repeated) == 2
        repeated[0]["storm_dashboard_cache"] = None
    result = dashboard(data)["windows"]["1"]
    valid = hours - (1 if hours == 25 else 0)
    assert result["expected_hours"] == hours
    assert result["storm_hours"] == valid
    assert result["frequencies"]["60min"]["denominator_samples"] == hours
    assert by_provider(result["frequencies"]["60min"], "storm")["hit_rate"] == pytest.approx(valid / hours)


def test_dashboard_anchors_report_day_not_last_complete_observed_day():
    data = payload(days=2)
    data["zones"][0]["rolling_history"]["rows"][-1]["observed"] = None
    legacy = dashboard(data)
    assert legacy["anchor_day"] == "2026-09-19"
    assert legacy["windows"]["1"]["storm_hours"] == 23
    result = rolling.build_rolling_performance(data)
    for strategy in result["strategies"]:
        completed = result["strategy_zones"][strategy][0]
        assert completed["anchor_day"] == "2026-09-18"
        assert completed["windows"]["1"]["paired_hours"] == 24


def test_completed_scope_and_input_payload_are_preserved_for_both_strategies():
    data = payload()
    before = deepcopy(data)
    internal = rolling._build_zone(data["zones"][0], data["delivery_day"])
    result = rolling.build_rolling_performance(data)
    assert data == before
    assert result["schema_version"] == 3
    assert result["source_scope"] == "internal_completed"
    assert "scopes" not in result and "dashboard_methodology" not in result
    assert result["zones"] == [internal]
    assert set(result["strategy_zones"]) == {"quantile_based", "unlimited_bid"}
    for strategy in result["strategies"]:
        completed = result["strategy_zones"][strategy][0]["windows"]["1"]["frequencies"]["60min"]
        assert by_provider(completed, "storm")["mae"] == 2
    json.dumps(result, allow_nan=False)


def test_payload_without_cache_metadata_still_has_both_completed_history_strategies():
    data = payload()
    del data["zones"][0]["rolling_history"]["sources"]
    result = rolling.build_rolling_performance(data)
    assert result["schema_version"] == 3
    assert "default_scope" not in result and "internal_zones" not in result
    assert result["source_scope"] == "internal_completed"
    assert set(result["strategy_zones"]) == {"quantile_based", "unlimited_bid"}
    assert by_provider(result["zones"][0]["windows"]["1"]["frequencies"]["60min"], "storm")["mae"] == 2


def test_annual_scope_is_explicit_extension_without_official_pnl(monkeypatch):
    monkeypatch.setattr(rolling, "WINDOWS", (7, 30, 60, 90, 365))
    windows = dashboard(payload())["windows"]
    for days in (7, 30, 60, 90):
        assert windows[str(days)]["external_window_available"] is True
    annual = windows["365"]
    assert annual["external_window_available"] is False
    assert annual["start_day"] == "2025-09-20"
    assert annual["end_day"] == "2026-09-19"
    assert annual["expected_hours"] == 365 * 24
    assert annual["frequencies"]["60min"]["denominator_samples"] == 365 * 24
    assert annual["frequencies"]["day"]["denominator_samples"] == 365
    assert annual["official_pnl_verified"] is False
    assert annual["pnl_days"] == 0
    for frequency in ("60min", "day"):
        assert all(provider["daily_pnl"] is None for provider in annual["frequencies"][frequency]["providers"])


def test_partial_but_verified_cache_is_accepted_with_full_grid_denominator():
    data = payload()
    history = data["zones"][0]["rolling_history"]
    history["sources"]["storm_dashboard_cache"]["status"] = "partial"
    for row in history["rows"][:12]:
        row["storm_dashboard_cache"] = None
    result = dashboard(data)
    assert result["history_status"] == "available"
    hourly = result["windows"]["1"]["frequencies"]["60min"]
    assert hourly["denominator_samples"] == 24
    assert hourly["storm_samples"] == hourly["model_samples"] == 12
    assert by_provider(hourly, "storm")["hit_rate"] == .5


def test_mixed_zone_payload_does_not_grant_cache_provenance_to_unverified_zone():
    data = payload()
    other = deepcopy(data["zones"][0])
    other.update(zone="BE", name="Belgium", timezone="Europe/Brussels")
    del other["rolling_history"]["sources"]["storm_dashboard_cache"]
    data["zones"].append(other)
    legacy = rolling._dashboard_zone(other, data["delivery_day"])
    assert legacy["history_status"] == "cache_provenance_unavailable"
    hourly = legacy["windows"]["1"]["frequencies"]["60min"]
    assert all(provider["samples"] == 0 and provider["mae"] is None for provider in hourly["providers"])
    result = rolling.build_rolling_performance(data)
    assert result["source_scope"] == "internal_completed"
    for item in result["zones"]:
        assert item["history_status"] == "available"
        assert item["windows"]["1"]["paired_hours"] == 24


def test_report_never_calls_legacy_storage_or_cache_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The paper report must not invoke the legacy MILP/cache path")
    monkeypatch.setattr(rolling, "optimize_storage_schedule", forbidden)
    monkeypatch.setattr(rolling, "_dashboard_zone", forbidden)
    result = rolling.build_rolling_performance(payload())
    for strategy in result["strategies"]:
        for frequency in ("60min", "day"):
            table = result["strategy_zones"][strategy][0]["windows"]["1"]["frequencies"][frequency]
            assert all(provider["pnl_kind"] == "paper_simulation" for provider in table["providers"])
            assert all(provider["daily_pnl"] is None and provider["pnl_days"] == 0 for provider in table["providers"])


def test_duplicate_cache_hour_is_removed_instead_of_choosing_an_arbitrary_revision():
    data = payload()
    rows = data["zones"][0]["rolling_history"]["rows"]
    rows.append(dict(rows[0], storm_dashboard_cache=5000))
    result = dashboard(data)
    assert result["duplicate_hours"] == 1
    section = result["windows"]["1"]
    assert section["expected_hours"] == 24
    assert section["storm_hours"] == section["model_hours"] == 23
    assert by_provider(section["frequencies"]["60min"], "storm")["mae"] == 3


def test_valid_future_rows_cannot_shift_dashboard_grid_or_metrics():
    data = payload()
    baseline = dashboard(data)
    data["zones"][0]["rolling_history"]["rows"].extend(payload("2026-09-20")["zones"][0]["rolling_history"]["rows"])
    assert dashboard(data) == baseline
