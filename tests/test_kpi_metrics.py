import json

import numpy as np
import pandas as pd
import pytest

from kpi_report.metrics import KPIError, compute_kpis


def panel(day="2026-09-14", *, models=("a", "b"), zones=("FR",), days=1):
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = start + pd.DateOffset(days=days)
    hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    return pd.DataFrame([
        {"model_id": model, "zone": zone, "timestamp_utc": stamp,
         "forecast": 12. if model == "a" else 15., "actual": 10., "storm": 14.}
        for zone in zones for model in models for stamp in hours
    ])


def get(result, model="a", zone="FR"):
    return next(row for row in result["rows"] if row["model_id"] == model and row["zone"] == zone)


def test_known_metrics_and_three_win_rates():
    result = compute_kpis(panel(), end_day="2026-09-14")
    row = get(result)
    assert row["n_hours"] == 24
    assert row["n_days"] == 1
    assert row["mae_eur_mwh"] == row["rmse_eur_mwh"] == 2.
    assert row["mean_price_eur_mwh"] == 12.
    assert row["observed_mean_price_eur_mwh"] == 10.
    assert row["storm_mean_price_eur_mwh"] == 14.
    for key in ("win_rate_hour_pct", "win_rate_day_mae_pct", "win_rate_day_mean_price_pct"):
        assert row[key] == 100.
        assert get(result, "b")[key] == 0.
    assert row["wins_hour"] == 24
    assert row["wins_day_mae"] == row["wins_day_mean_price"] == 1
    assert get(result, "b")["losses_hour"] == 24
    assert get(result, "__storm__")["win_rate_hour_pct"] is None
    assert get(result, "__storm__")["mae_eur_mwh"] == 4.
    json.dumps(result, allow_nan=False)


def test_ties_are_not_wins_and_remain_in_denominator():
    frame = panel(models=("a",))
    frame.loc[:7, "forecast"] = 14.
    frame.loc[8:15, "forecast"] = 14.-0.5e-9
    row = get(compute_kpis(frame, end_day="2026-09-14"))
    assert row["ties_hour"] == 16
    assert row["wins_hour"] == 8
    assert row["win_rate_hour_pct"] == pytest.approx(100./3.)
    frame["forecast"] = 14.
    row = get(compute_kpis(frame, end_day="2026-09-14"))
    assert row["win_rate_day_mae_pct"] == row["win_rate_day_mean_price_pct"] == 0.
    assert row["ties_day_mae"] == row["ties_day_mean_price"] == 1


def test_daily_mae_and_mean_price_wins_are_different():
    frame = panel(models=("a",))
    frame.loc[:11, "forecast"] = 4.
    frame.loc[12:, "forecast"] = 16.
    row = get(compute_kpis(frame, end_day="2026-09-14"))
    assert row["mae_eur_mwh"] == 6.
    assert row["mae_day_mean_price_eur_mwh"] == 0.
    assert row["win_rate_day_mae_pct"] == 0.
    assert row["win_rate_day_mean_price_pct"] == 100.


def test_rmse_is_not_mae():
    frame = panel(models=("a",))
    frame["forecast"] = 10.
    frame.loc[0, "forecast"] = 34.
    row = get(compute_kpis(frame, end_day="2026-09-14"))
    assert row["mae_eur_mwh"] == 1.
    assert row["rmse_eur_mwh"] == pytest.approx(np.sqrt(24.))


def test_missing_model_gives_no_comparable_support():
    result = compute_kpis(panel(), end_day="2026-09-14", models=["a", "absent"])
    row = get(result)
    assert row["n_hours"] == 0
    assert row["status"] == "no_common_support"
    assert row["mae_eur_mwh"] is None
    assert result["coverage"][0]["missing_models"] == ["absent"]


def test_intersection_not_individual_support_and_incomplete_day_excluded():
    frame = panel()
    frame.loc[frame.model_id.eq("b") & frame.timestamp_utc.eq(frame.timestamp_utc.min()), "forecast"] = np.nan
    result = compute_kpis(frame, end_day="2026-09-14")
    for model in ("a", "b", "__storm__"):
        row = get(result, model)
        assert row["n_hours"] == 23
        assert row["n_days"] == 0
        assert row["win_rate_day_mae_pct"] is None
    assert not result["daily_rows"]
    result_a = compute_kpis(frame, end_day="2026-09-14", models=["a"])
    assert get(result_a)["n_hours"] == 24


@pytest.mark.parametrize("day,expected", [("2026-03-29", 23), ("2025-10-26", 25), ("2026-09-14", 24)])
def test_complete_dst_civil_days(day, expected):
    result = compute_kpis(panel(day), end_day=day, days=1)
    assert get(result)["n_hours"] == expected
    assert get(result)["n_days"] == 1
    assert result["period"]["expected_hours_per_zone"] == expected
    assert result["daily_rows"][0]["n_hours"] == expected


def test_missing_repeated_fall_hour_is_not_a_complete_day():
    frame = panel("2025-10-26", models=("a",))
    frame = frame.loc[~frame.timestamp_utc.eq(pd.Timestamp("2025-10-26T01:00Z"))]
    result = compute_kpis(frame, end_day="2025-10-26", days=1)
    assert get(result)["n_hours"] == 24
    assert get(result)["n_days"] == 0


def test_365_day_boundaries_exclude_future_and_previous_day():
    frame = panel("2025-09-14", models=("a",), days=367)
    result = compute_kpis(frame, end_day="2026-09-14")
    assert result["period"]["start_day"] == "2025-09-15"
    assert get(result)["n_days"] == 365
    assert get(result)["n_hours"] == 8760
    assert min(record["delivery_day"] for record in result["daily_rows"]) == "2025-09-15"
    assert max(record["delivery_day"] for record in result["daily_rows"]) == "2026-09-14"


def test_zero_negative_and_missing_reference_prices():
    frame = panel(models=("a",))
    frame["forecast"] = 0.
    frame["actual"] = -10.
    frame["storm"] = 0.
    row = get(compute_kpis(frame, end_day="2026-09-14"))
    assert row["n_hours"] == 24
    assert row["mean_price_eur_mwh"] == 0.
    assert row["mae_eur_mwh"] == 10.
    frame["actual"] = np.nan
    result = compute_kpis(frame, end_day="2026-09-14")
    assert get(result)["n_hours"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reference", ["actual", "storm"])
def test_conflicting_references_are_rejected(reference):
    frame = panel()
    frame.loc[frame.model_id.eq("b"), reference] += 1.
    with pytest.raises(KPIError, match=f"Conflicting shared {reference}"):
        compute_kpis(frame, end_day="2026-09-14")


def test_shared_references_can_fill_another_models_missing_reference():
    frame = panel()
    frame.loc[frame.model_id.eq("a"), ["actual", "storm"]] = np.nan
    assert get(compute_kpis(frame, end_day="2026-09-14"))["n_hours"] == 24


def test_duplicate_identical_is_deduplicated_conflicting_is_rejected():
    frame = panel()
    duplicate = frame.iloc[[0]].copy()
    assert get(compute_kpis(pd.concat([frame, duplicate]), end_day="2026-09-14"))["n_hours"] == 24
    duplicate["forecast"] += 1.
    with pytest.raises(KPIError, match="Conflicting duplicate"):
        compute_kpis(pd.concat([frame, duplicate]), end_day="2026-09-14")


@pytest.mark.parametrize("value", ["2026-09-14 00:00", pd.NaT, "not a timestamp"])
def test_naive_or_missing_timestamp_refused(value):
    frame = panel(models=("a",))
    frame["timestamp_utc"] = frame.timestamp_utc.astype(object)
    frame.loc[0, "timestamp_utc"] = value
    with pytest.raises(KPIError, match="timestamp_utc"):
        compute_kpis(frame, end_day="2026-09-14")


def test_timezone_aware_non_utc_inputs_are_normalized():
    frame = panel()
    frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_convert("Europe/Paris")
    assert get(compute_kpis(frame, end_day="2026-09-14"))["n_hours"] == 24


def test_non_hourly_physical_stamp_is_rejected():
    frame = panel()
    frame["timestamp_utc"] += pd.Timedelta(minutes=15)
    with pytest.raises(KPIError, match="whole physical hour"):
        compute_kpis(frame, end_day="2026-09-14")


def test_all_uses_hour_and_country_day_weights_not_country_averages():
    frame = panel(models=("a",), zones=("FR", "DE"), days=2)
    de_day_two = frame.zone.eq("DE") & frame.timestamp_utc.ge(pd.Timestamp("2026-09-15", tz="Europe/Paris"))
    frame = frame.loc[~de_day_two].copy()
    frame.loc[frame.zone.eq("DE"), "forecast"] = 18.
    result = compute_kpis(frame, end_day="2026-09-15")
    row = get(result, zone="ALL")
    assert row["n_hours"] == 72
    assert row["n_days"] == 3
    assert row["n_calendar_days"] == 2
    assert row["mae_eur_mwh"] == 4.  # (48*2 + 24*8)/72, not (2+8)/2
    assert row["rmse_eur_mwh"] == pytest.approx(np.sqrt(24.))
    assert row["win_rate_day_mae_pct"] == pytest.approx(200./3.)


def test_daily_mean_prices_do_not_hour_weight_dst_days():
    frame = panel("2026-03-28", models=("a",), days=2)
    second = frame.timestamp_utc.ge(pd.Timestamp("2026-03-29", tz="Europe/Paris"))
    frame.loc[second, "forecast"] = 20.
    result = compute_kpis(frame, end_day="2026-03-29", days=2)
    row = get(result)
    assert row["mean_daily_price_eur_mwh"] == 16.
    assert row["mean_price_eur_mwh"] == pytest.approx((24*12+23*20)/47)


def test_empty_frame_and_absent_zone_are_json_safe():
    empty = panel().iloc[:0]
    result = compute_kpis(empty, end_day="2026-09-14", models=["a"], zones=["FR"])
    assert get(result)["n_hours"] == 0
    assert result["coverage"][0]["missing_models"] == ["a"]
    json.dumps(result, allow_nan=False)


def test_input_frame_unchanged():
    frame = panel()
    before = frame.copy(deep=True)
    compute_kpis(frame, end_day="2026-09-14")
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("days", [0, -1, 2.5, True])
def test_invalid_period_is_rejected(days):
    with pytest.raises(KPIError, match="positive integer"):
        compute_kpis(panel(), end_day="2026-09-14", days=days)


@pytest.mark.parametrize("day", ["20260914", "2026-09-14T00:00:00", "no"])
def test_non_iso_civil_day_is_rejected(day):
    with pytest.raises(KPIError, match="ISO civil date"):
        compute_kpis(panel(), end_day=day)


def test_duplicate_selections_are_rejected():
    with pytest.raises(KPIError, match="duplicate selection"):
        compute_kpis(panel(), end_day="2026-09-14", models=["a", "a"])


@pytest.mark.parametrize("name", ["actual", "storm", "zone", "delivery_day", "__storm__"])
def test_internal_identifier_collisions_fail_explicitly(name):
    frame = panel(models=(name,))
    with pytest.raises(KPIError, match="Reserved"):
        compute_kpis(frame, end_day="2026-09-14")


def test_missing_zone_only_blocks_that_zones_support():
    frame = panel(zones=("FR", "DE"))
    frame = frame.loc[~(frame.zone.eq("DE") & frame.model_id.eq("b"))]
    result = compute_kpis(frame, end_day="2026-09-14")
    assert get(result, zone="FR")["n_hours"] == 24
    assert get(result, zone="DE")["n_hours"] == 0
    assert get(result, zone="ALL")["n_hours"] == 24


def test_day_with_all_hours_but_one_missing_storm_is_not_complete():
    frame = panel()
    first = frame.timestamp_utc.eq(frame.timestamp_utc.min())
    frame.loc[first, "storm"] = np.nan
    result = compute_kpis(frame, end_day="2026-09-14")
    assert get(result)["n_hours"] == 23
    assert get(result)["n_days"] == 0


def test_close_shared_references_are_tolerated_but_not_large_revisions():
    frame = panel()
    frame.loc[frame.model_id.eq("b"), "actual"] += 0.5e-9
    assert get(compute_kpis(frame, end_day="2026-09-14"))["n_hours"] == 24


@pytest.mark.parametrize("column", ["forecast", "actual", "storm"])
def test_infinity_is_rejected(column):
    frame = panel()
    frame.loc[0, column] = np.inf
    with pytest.raises(KPIError, match="infinity"):
        compute_kpis(frame, end_day="2026-09-14")
