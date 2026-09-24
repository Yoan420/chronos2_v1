from __future__ import annotations

import math

import pandas as pd
import pytest

from auxiliary_lab.weather import (
    HOURLY_VARIABLES,
    WeatherMaterializationError,
    WeatherPoint,
    build_request_parameters,
    delivery_utc_index,
    issue_times_for_delivery,
    parse_zone_day_payload,
    validate_causality,
)


def _response(
    index: pd.DatetimeIndex,
    *,
    point_number: int,
    remove_position: int | None = None,
    null_variable: str | None = None,
) -> dict[str, object]:
    times = [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in index]
    values: dict[str, list[float | None]] = {}
    for variable_number, variable in enumerate(HOURLY_VARIABLES):
        if variable == "wind_direction_100m":
            base = 350.0 if point_number == 0 else 10.0
        else:
            base = 10.0 + 4.0 * point_number + variable_number
        values[variable] = [base] * len(times)
    if null_variable is not None:
        values[null_variable][3] = None
    if remove_position is not None:
        times.pop(remove_position)
        for variable_values in values.values():
            variable_values.pop(remove_position)
    hourly: dict[str, object] = {"time": times, **values}
    return {
        "latitude": 48.0 + point_number,
        "longitude": 2.0 + point_number,
        "elevation": 100.0 + point_number,
        "hourly_units": {
            variable: "degree" if variable == "wind_direction_100m" else "unit"
            for variable in HOURLY_VARIABLES
        },
        "hourly": hourly,
    }


@pytest.mark.parametrize(
    ("day", "expected_hours"),
    [
        ("2026-03-28", 24),
        ("2026-03-29", 23),
        ("2026-10-25", 25),
        ("2026-10-26", 24),
    ],
)
def test_delivery_utc_index_preserves_physical_dst_hours(
    day: str,
    expected_hours: int,
) -> None:
    index = delivery_utc_index(day)

    assert len(index) == expected_hours
    assert str(index.tz) == "UTC"
    assert index.is_monotonic_increasing
    assert not index.has_duplicates


def test_issue_policy_is_d_minus_two_18z_before_civil_cutoff() -> None:
    run, cutoff = issue_times_for_delivery("2026-03-29")

    assert run == pd.Timestamp("2026-03-27T18:00:00Z")
    assert cutoff == pd.Timestamp("2026-03-28T07:00:00Z")
    assert run < cutoff


def test_validate_causality_rejects_equal_or_later_run() -> None:
    cutoff = pd.Timestamp("2026-08-27T06:00:00Z")

    with pytest.raises(WeatherMaterializationError, match="Run non causal"):
        validate_causality(cutoff, cutoff)
    with pytest.raises(WeatherMaterializationError, match="Run non causal"):
        validate_causality(cutoff + pd.Timedelta(hours=1), cutoff)


def test_parse_zone_day_aggregates_without_direction_wraparound_error() -> None:
    day = "2026-08-28"
    expected = delivery_utc_index(day)
    run, cutoff = issue_times_for_delivery(day)
    points = (
        WeatherPoint("west", 48.0, 2.0),
        WeatherPoint("east", 49.0, 3.0),
    )

    parsed = parse_zone_day_payload(
        [_response(expected, point_number=0), _response(expected, point_number=1)],
        zone="FR",
        points=points,
        delivery_day=day,
        run_init_utc=run,
        cutoff_utc=cutoff,
    )

    frame = parsed.frame
    assert len(frame) == 24
    assert frame["delivery_start_utc"].tolist() == expected.tolist()
    assert frame["run_init_utc"].nunique() == 1
    assert frame["cutoff_utc"].nunique() == 1
    assert frame["lead_hours"].tolist() == list(range(28, 52))
    assert frame["fr_temperature_2m_mean"].eq(12.0).all()
    assert frame["fr_temperature_2m_std"].eq(2.0).all()
    direction = float(frame["fr_wind_direction_100m_mean"].iloc[0])
    assert min(abs(direction), abs(direction - 360.0)) < 1e-9
    assert 9.0 < float(frame["fr_wind_direction_100m_std"].iloc[0]) < 11.0
    assert len(parsed.returned_grid_points) == 2


def test_parse_zone_day_rejects_missing_hour_instead_of_filling() -> None:
    day = "2026-03-29"
    expected = delivery_utc_index(day)
    run, cutoff = issue_times_for_delivery(day)
    point = WeatherPoint("only", 48.0, 2.0)

    with pytest.raises(WeatherMaterializationError, match="heure.*absente"):
        parse_zone_day_payload(
            _response(expected, point_number=0, remove_position=5),
            zone="FR",
            points=(point,),
            delivery_day=day,
            run_init_utc=run,
            cutoff_utc=cutoff,
        )


def test_parse_zone_day_rejects_null_instead_of_interpolating() -> None:
    day = "2026-10-25"
    expected = delivery_utc_index(day)
    run, cutoff = issue_times_for_delivery(day)
    point = WeatherPoint("only", 48.0, 2.0)

    with pytest.raises(WeatherMaterializationError, match="sans fill autorise"):
        parse_zone_day_payload(
            _response(
                expected,
                point_number=0,
                null_variable="temperature_2m",
            ),
            zone="FR",
            points=(point,),
            delivery_day=day,
            run_init_utc=run,
            cutoff_utc=cutoff,
        )


def test_request_is_single_ifs_run_and_does_not_contain_api_key() -> None:
    run = pd.Timestamp("2026-08-26T18:00:00Z")
    params = build_request_parameters(
        points=(
            WeatherPoint("a", 48.0, 2.0),
            WeatherPoint("b", 49.0, 3.0),
        ),
        run_init_utc=run,
    )

    assert params["models"] == "ecmwf_ifs"
    assert params["run"] == "2026-08-26T18:00"
    assert params["forecast_hours"] == 72
    assert params["timezone"] == "GMT"
    assert params["latitude"] == "48.000000,49.000000"
    assert "apikey" not in params
    assert not any(
        isinstance(value, float) and math.isnan(value) for value in params.values()
    )
