import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from materialize_openmeteo_previous_runs import (
    FRANCE_PANEL,
    WEATHER_VARIABLES,
    WeatherMaterialisationError,
    _api_variable,
    materialize,
    parse_weather_payload,
    physical_delivery_index,
    timing_contract,
)


def _location_payload(index: pd.DatetimeIndex, offset: float, location_id: int) -> dict:
    values = np.arange(len(index), dtype=float) + offset
    hourly = {"time": (index.asi8 // 1_000_000_000).tolist()}
    for variable_number, variable in enumerate(WEATHER_VARIABLES):
        hourly[_api_variable(variable, 2)] = (values + 100 * variable_number).tolist()
    return {
        "location_id": location_id,
        "latitude": 48.0 + location_id,
        "longitude": 2.0 + location_id,
        "elevation": 50.0,
        "timezone": "GMT",
        "utc_offset_seconds": 0,
        "hourly": hourly,
    }


def test_physical_delivery_index_keeps_23_and_25_hour_days() -> None:
    spring = physical_delivery_index("2025-03-30", "2025-03-30")
    autumn = physical_delivery_index("2024-10-27", "2024-10-27")
    assert len(spring) == 23
    assert len(autumn) == 25
    assert spring.tz is not None and str(spring.tz) == "UTC"
    assert autumn.is_unique
    assert (autumn[1:] - autumn[:-1] == pd.Timedelta(hours=1)).all()


def test_previous_day2_is_strictly_before_civil_cutoff_across_dst() -> None:
    index = physical_delivery_index("2024-10-26", "2024-10-28")
    timing = timing_contract(index, lead_days=2)
    assert (
        timing["weather_fixed_lead_reference_time_utc"]
        < timing["weather_cutoff_time_utc"]
    ).all()
    local_cutoffs = timing["weather_cutoff_time_utc"].dt.tz_convert("Europe/Paris")
    assert set(local_cutoffs.dt.hour) == {8}


def test_previous_day1_is_rejected_for_eight_oclock_cutoff() -> None:
    index = physical_delivery_index("2025-01-15", "2025-01-15")
    with pytest.raises(WeatherMaterialisationError, match="not strictly earlier"):
        timing_contract(index, lead_days=1)


def test_parser_keeps_exact_hours_and_builds_unweighted_panel_aggregates() -> None:
    sites = FRANCE_PANEL[:2]
    expected = physical_delivery_index("2025-03-30", "2025-03-30")
    # The API query is UTC-date based and can legitimately contain extra hours.
    api_index = pd.date_range(
        expected.min().floor("D"), expected.max().ceil("D"), freq="h"
    )
    payload = [
        _location_payload(api_index, 0.0, 0),
        _location_payload(api_index, 10.0, 1),
    ]
    frame, grids = parse_weather_payload(
        payload,
        sites=sites,
        expected_index=expected,
        lead_days=2,
    )
    assert frame.index.equals(expected)
    assert len(frame) == 23
    first = WEATHER_VARIABLES[0].output_stem
    expected_mean = (
        frame[f"{first}__{sites[0].slug}"]
        + frame[f"{first}__{sites[1].slug}"]
    ) / 2.0
    pd.testing.assert_series_equal(
        frame[f"{first}__panel_mean"], expected_mean, check_names=False
    )
    assert [grid["site_slug"] for grid in grids] == [site.slug for site in sites]


def test_parser_fails_on_missing_hour_without_interpolation() -> None:
    sites = FRANCE_PANEL[:1]
    expected = physical_delivery_index("2025-01-15", "2025-01-15")
    payload = _location_payload(expected.delete(12), 0.0, 0)
    with pytest.raises(WeatherMaterialisationError, match="exact UTC hour.*missing"):
        parse_weather_payload(
            payload,
            sites=sites,
            expected_index=expected,
            lead_days=2,
        )


def test_parser_fails_on_null_value_without_fill() -> None:
    sites = FRANCE_PANEL[:1]
    expected = physical_delivery_index("2025-01-15", "2025-01-15")
    payload = _location_payload(expected, 0.0, 0)
    payload["hourly"][_api_variable(WEATHER_VARIABLES[1], 2)][5] = None
    with pytest.raises(WeatherMaterialisationError, match="non-finite"):
        parse_weather_payload(
            payload,
            sites=sites,
            expected_index=expected,
            lead_days=2,
        )


def test_materializer_writes_output_provenance_and_manifest_sha(
    tmp_path, monkeypatch
) -> None:
    sites = FRANCE_PANEL[:2]

    def fake_fetch(api_url, params, *, timeout_seconds, retries):
        del api_url, timeout_seconds, retries
        utc_index = pd.date_range(
            pd.Timestamp(params["start_date"], tz="UTC"),
            pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23),
            freq="h",
        )
        payload = [
            _location_payload(utc_index, 0.0, 0),
            _location_payload(utc_index, 10.0, 1),
        ]
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    monkeypatch.setattr(
        "materialize_openmeteo_previous_runs.fetch_response_bytes", fake_fetch
    )
    output = tmp_path / "weather.parquet"
    manifest_path = tmp_path / "weather.manifest.json"
    sha_path = tmp_path / "weather.manifest.json.sha256"
    raw_dir = tmp_path / "raw"
    result = materialize(
        start_day="2025-03-29",
        end_day="2025-03-31",
        output=output,
        manifest_path=manifest_path,
        manifest_sha_path=sha_path,
        raw_dir=raw_dir,
        sites=sites,
        chunk_days=2,
    )
    frame = pd.read_parquet(output)
    assert len(frame) == 71  # 24 + 23 + 24 physical hours
    assert frame.columns[0] == "value_time_utc"
    assert frame["value_time_utc"].is_unique
    assert result["causality_contract"]["storm_used"] is False
    assert result["causality_contract"]["actual_weather_used"] is False
    assert result["causality_contract"][
        "minimum_reference_to_cutoff_margin_hours"
    ] > 0
    assert result["causality"] == {
        "classification": "strict_fixed_lead_forecast",
        "strict_pit_eligible": True,
        "available_by_d_minus_1_08_europe_paris": True,
        "cutoff_violations": 0,
        "target_or_price_used": False,
        "storm_used": False,
    }
    assert result["output_sha256"] == result["output"]["sha256"]
    assert result["timeline_contract"]["interpolation"] == "none"
    assert result["raw_responses_preserved"] is True
    assert len(list(raw_dir.glob("*.json"))) == 2
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert sha_path.read_text(encoding="ascii").split()[0] == manifest_sha
    assert result["manifest_sha256"] == manifest_sha
