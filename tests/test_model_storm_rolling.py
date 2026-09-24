from __future__ import annotations

from datetime import date, timedelta
import json
import math

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import model_storm_rolling as rolling


def payload(day="2026-09-18", days=1):
    rows = []
    for offset in reversed(range(days)):
        current = (date.fromisoformat(day) - timedelta(days=offset)).isoformat()
        for index, stamp in enumerate(rolling._hours(current, "Europe/Paris")):
            rows.append({"timestamp_utc": stamp.isoformat(), "observed": float(index),
                         "storm": float(index) + 2, "model": float(index) - 4,
                         "model_p10": float(index) - 14, "model_p90": float(index) + 6})
    return {"delivery_day": day, "zones": [{"zone": "FR", "name": "France", "timezone": "Europe/Paris",
                                            "rolling_history": {"rows": rows}}]}


def zone_result(data):
    return rolling.build_rolling_performance(data)["zones"][0]


def table(data, window="7", frequency="60min"):
    return zone_result(data)["windows"][window]["frequencies"][frequency]


def test_score_equations_and_inclusive_hit_threshold():
    result = rolling.score_forecast([0, 10, 20, 30], [5, 5, 25.00001, 30])
    assert result["mae"] == pytest.approx(15.00001 / 4)
    assert result["bias"] == pytest.approx(5.00001 / 4)
    assert result["rmse"] == pytest.approx(math.sqrt((25 + 25 + 5.00001 ** 2) / 4))
    assert result["hit_rate"] == .75
    assert result["r2"] == pytest.approx(1 - (50 + 5.00001 ** 2) / 500)


@pytest.mark.parametrize("actual,forecast", [([], []), ([0, 0], [0, 0]), ([-5, -5], [-4, -4]),
                                             ([.1] * 24, [.2] * 24)])
def test_constant_or_empty_observation_has_no_r_squared(actual, forecast):
    assert rolling.score_forecast(actual, forecast)["r2"] is None


@pytest.mark.parametrize("actual,forecast", [([0], []), ([[0]], [[1]]), ([float("nan")], [1]), ([0], [float("inf")])])
def test_score_rejects_unpaired_nonfinite_arrays(actual, forecast):
    with pytest.raises(ValueError):
        rolling.score_forecast(actual, forecast)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-09-18", 24), ("2026-10-25", 25)])
def test_daily_scores_use_every_physical_dst_hour(day, hours):
    result = zone_result(payload(day))
    section = result["windows"]["7"]
    assert result["anchor_day"] == day
    assert section["paired_hours"] == hours
    assert section["complete_paired_days"] == 1
    assert section["pnl_days"] == 0  # One day cannot calibrate Storm's quantiles.
    assert section["frequencies"]["day"]["providers"][0]["mae"] == pytest.approx(2)
    assert section["frequencies"]["day"]["providers"][1]["bias"] == pytest.approx(-4)
    assert section["frequencies"]["day"]["providers"][0]["daily_pnl"] == section["frequencies"]["60min"]["providers"][0]["daily_pnl"]


@pytest.mark.parametrize("key", ["storm", "model"])
@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), True])
def test_missing_forecast_pairs_identical_hourly_samples_and_drops_day(key, missing):
    data = payload()
    data["zones"][0]["rolling_history"]["rows"][5][key] = missing
    result = zone_result(data)["windows"]["7"]
    assert result["paired_hours"] == 23
    assert result["complete_paired_days"] == 0
    assert result["pnl_days"] == 0
    assert all(provider["samples"] == 23 for provider in result["frequencies"]["60min"]["providers"])
    assert all(provider["mae"] is None for provider in result["frequencies"]["day"]["providers"])


def test_anchor_is_latest_complete_observed_day_not_latest_forecast_day():
    data = payload(days=3)
    rows = data["zones"][0]["rolling_history"]["rows"]
    rows[-1]["observed"] = None
    rows[-30]["model"] = None
    result = zone_result(data)
    assert result["anchor_day"] == "2026-09-17"
    assert result["windows"]["7"]["paired_hours"] == 47


def test_future_rows_never_influence_scores_or_anchor():
    data = payload(days=2)
    baseline = zone_result(data)
    data["zones"][0]["rolling_history"]["rows"] += payload("2026-09-19")["zones"][0]["rolling_history"]["rows"]
    assert zone_result(data) == baseline


def test_duplicate_instants_fail_closed_instead_of_selecting_a_value():
    data = payload(days=2)
    rows = data["zones"][0]["rolling_history"]["rows"]
    rows.append(dict(rows[-1], observed=99999))
    result = zone_result(data)
    assert result["duplicate_hours"] == 1
    assert result["anchor_day"] == "2026-09-17"


@pytest.mark.parametrize("stamp", ["not-a-date", "2026-09-18T12:00:00", "2026-09-18T12:15:00Z", None])
def test_invalid_timestamp_rows_are_not_scored(stamp):
    data = payload()
    data["zones"][0]["rolling_history"]["rows"].append({"timestamp_utc": stamp, "model": 0, "storm": 0, "observed": 0})
    result = zone_result(data)
    assert result["invalid_timestamp_rows"] == 1
    assert result["windows"]["7"]["paired_hours"] == 24


def test_all_unavailable_or_no_history_is_json_safe_and_never_fabricates_zero():
    data = payload()
    for row in data["zones"][0]["rolling_history"]["rows"]:
        row.update(observed=None, storm=None, model=None)
    result = rolling.build_rolling_performance(data)
    json.dumps(result, allow_nan=False)
    assert result["zones"][0]["anchor_day"] is None
    assert result["zones"][0]["windows"]["90"]["paired_hours"] == 0
    assert all(provider["mae"] is None and provider["daily_pnl"] is None
               for provider in result["zones"][0]["windows"]["7"]["frequencies"]["60min"]["providers"])
    del data["zones"][0]["rolling_history"]
    assert zone_result(data)["history_status"] == "no_complete_observed_day"


def test_true_zero_scores_and_zero_storage_profit_remain_numbers():
    data = payload(days=61)
    for row in data["zones"][0]["rolling_history"]["rows"]:
        row.update(observed=0, storm=0, model=0, model_p10=0, model_p90=0)
    result = table(data)
    for provider in result["providers"]:
        assert provider["mae"] == provider["rmse"] == provider["bias"] == provider["daily_pnl"] == 0
        assert provider["hit_rate"] == 1


def test_day_frequency_scores_daily_mean_not_mean_hourly_absolute_error():
    data = payload()
    for index, row in enumerate(data["zones"][0]["rolling_history"]["rows"]):
        row.update(observed=10, model=(5 if index % 2 else 15), storm=11)
    assert table(data)["providers"][1]["mae"] == 5
    assert table(data, frequency="day")["providers"][1]["mae"] == 0


def test_paper_strategies_schedule_eligible_days_once_and_reuse_all_windows(monkeypatch):
    from chronos2_hourly import model_storm_bess as bess
    calls = []
    original = bess.simulate_bess_day
    def spy(forecast, observed, **kwargs):
        calls.append((tuple(forecast), kwargs["strategy"]))
        return original(forecast, observed, **kwargs)
    monkeypatch.setattr(bess, "simulate_bess_day", spy)
    monkeypatch.setattr(rolling, "optimize_storage_schedule",
                        lambda _: pytest.fail("The legacy MILP must not run in the paper report"))
    result = zone_result(payload(days=366))
    assert len(calls) == 306 * 2 * 2  # 60 prior days for calibration, 2 providers, 2 strategies.
    for window in rolling.WINDOWS:
        assert result["windows"][str(window)]["complete_paired_days"] == window
        assert result["windows"][str(window)]["pnl_days"] == min(window, 306)


def test_invalid_native_interval_removes_same_day_for_both_provider_pnl_only():
    data = payload(days=67)
    data["zones"][0]["rolling_history"]["rows"][-1]["model_p90"] = -1000.
    result = rolling.build_rolling_performance(data)
    for strategy in result["strategies"]:
        window = result["strategy_zones"][strategy][0]["windows"]["7"]
        assert window["paired_hours"] == 7 * 24
        assert window["pnl_excluded_days"] == 1
        assert window["pnl_days"] == 6
        assert all(provider["pnl_days"] == 6 for provider in window["frequencies"]["60min"]["providers"])


@pytest.mark.parametrize("forecast", [[0.] * 24, [50.] * 24, [-50.] * 24, [-40.] * 5 + [150.] * 19,
                                       list(np.linspace(-80, 160, 23)), list(np.linspace(300, -100, 25))])
def test_storage_physical_constraints_even_with_negative_prices(forecast):
    plan = rolling.optimize_storage_schedule(forecast)
    assert plan is not None
    charge, discharge, soc = [np.asarray(plan[key]) for key in ("charge_mw", "discharge_mw", "soc_mwh")]
    assert (charge >= -1e-6).all() and (charge <= 1 + 1e-6).all()
    assert (discharge >= -1e-6).all() and (discharge <= 1 + 1e-6).all()
    assert not ((charge > 1e-6) & (discharge > 1e-6)).any()
    assert (soc >= -1e-6).all() and (soc <= 4 + 1e-6).all()
    assert soc[0] == pytest.approx(0) and soc[-1] == pytest.approx(0)
    eta = math.sqrt(.85)
    np.testing.assert_allclose(np.diff(soc), eta * charge - discharge / eta, atol=1e-6)
    assert plan["forecast_pnl"] >= -1e-6


def test_paper_unlimited_pair_cannot_use_same_day_observed_prices_and_reprices_fixed_pair():
    data = payload(days=61)
    rows = data["zones"][0]["rolling_history"]["rows"]
    for index, row in enumerate(rows):
        price = 0. if index % 24 < 12 else 100.
        row.update(storm=price, model=price, observed=price, model_p10=price - 10, model_p90=price + 10)
    original = zone_result(data)
    before = original["pnl_audit"][-1]["providers"]["storm"]
    for row in rows[-24:]:
        row["observed"] = -row["observed"]
    changed = zone_result(data)["pnl_audit"][-1]["providers"]["storm"]
    assert before["pnl_eur"] > 0
    assert changed["operating_cost_eur"] == before["operating_cost_eur"] == 25.
    assert changed["pnl_eur"] == pytest.approx(-before["pnl_eur"] - 2 * before["operating_cost_eur"])
    for field in ("buy_index", "sell_index", "predicted_profit_eur"):
        assert changed[field] == before[field]


def test_storage_returns_copy_not_mutable_cached_state():
    forecast = [10.] * 12 + [100.] * 12
    plan = rolling.optimize_storage_schedule(forecast)
    plan["charge_mw"][0] = 99
    assert rolling.optimize_storage_schedule(forecast)["charge_mw"][0] <= 1


@pytest.mark.parametrize("forecast", [[0.] * 22, [0.] * 26, [0.] * 23 + [float("nan")], [True] * 24])
def test_storage_rejects_non_hourly_or_nonfinite_forecasts(forecast):
    with pytest.raises(ValueError):
        rolling.optimize_storage_schedule(forecast)


def test_metadata_discloses_paper_simulation_and_common_calibrated_support():
    result = rolling.build_rolling_performance(payload())
    assert result["frequencies"] == ["60min", "day"]
    assert "15min" not in str(result)
    assert "Paper-based battery simulation" in result["methodology"]["pnl"]
    assert "not live trading or published Storm VPS" in result["methodology"]["pnl"]
    assert "Native NYX P10/P90 and causal Storm P10/P90" in result["methodology"]["pnl_support"]
