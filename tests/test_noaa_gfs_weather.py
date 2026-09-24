"""Causality, physical-hour and bounded-download tests for public GFS inputs."""
import json
import threading
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from auxiliary_lab.noaa_gfs import (
    DecodedField, GfsClient, GfsError, hourly_radiation, issue_times, parse_index,
)
from auxiliary_lab.weather import delivery_utc_index
from auxiliary_lab import noaa_gfs


RUN = pd.Timestamp("2023-09-07T00:00:00Z")
CUTOFF = pd.Timestamp("2023-09-07T06:00:00Z")
INDEX = "\n".join([
    "1:0:d=2023090700:TMP:2 m above ground:24 hour fcst:",
    "2:24:d=2023090700:UGRD:100 m above ground:24 hour fcst:",
    "3:48:d=2023090700:VGRD:100 m above ground:24 hour fcst:",
    "4:72:d=2023090700:DSWRF:surface:18-24 hour ave fcst:",
    "5:96:d=2023090700:PRES:surface:24 hour fcst:",
])


@pytest.mark.parametrize("day,hours,origin", [
    ("2023-09-08", 24, "2023-09-07T06:00:00Z"),
    ("2024-03-31", 23, "2024-03-30T07:00:00Z"),
    ("2024-10-27", 25, "2024-10-26T06:00:00Z"),
    ("2024-01-08", 24, "2024-01-07T07:00:00Z"),
])
def test_origin_is_civil_eight_and_complete_hour_intervals(day, hours, origin):
    run, cutoff = issue_times(day)
    index = delivery_utc_index(day)
    assert cutoff == pd.Timestamp(origin)
    assert run.hour == 0 and run < cutoff < index[0]
    assert len(index) == hours
    endpoints = pd.date_range(index[0], index[-1] + pd.Timedelta(hours=1), freq="h")
    assert len(endpoints) == hours + 1
    assert all(np.diff(endpoints.asi8) == pd.Timedelta(hours=1).value)


def radiation(start, end, values):
    return DecodedField(np.array(values, dtype=float), start, end,
        {"grid_points": [{"lat": 50.0, "lon": float(i)} for i in range(len(values))]})


def test_solar_energy_difference_and_accumulation_reset():
    values, clipped = hourly_radiation(radiation(18, 23, [100]), radiation(18, 24, [110]))
    np.testing.assert_allclose(values, [160])
    assert clipped == 0
    reset, clipped = hourly_radiation(radiation(18, 24, [110]), radiation(24, 25, [75]))
    np.testing.assert_allclose(reset, [75])
    assert clipped == 0


def test_solar_packing_noise_is_counted_but_real_negative_is_rejected():
    values, clipped = hourly_radiation(radiation(0, 1, [0.3, 5]), radiation(0, 2, [0.1, 8]))
    np.testing.assert_allclose(values, [0, 11])
    assert clipped == 1
    with pytest.raises(GfsError, match="physical bounds"):
        hourly_radiation(radiation(0, 1, [100]), radiation(0, 2, [0]))


@pytest.mark.parametrize("previous,current", [
    (radiation(0, 1, [1]), radiation(0, 3, [2])),
    (radiation(0, 3, [1]), radiation(2, 4, [2])),
    (radiation(0, 1, [1]), radiation(0, 2, [float("nan")])),
    (radiation(0, 1, [1]), radiation(0, 2, [1, 2])),
])
def test_solar_rejects_unreconstructable_hours(previous, current):
    with pytest.raises(GfsError):
        hourly_radiation(previous, current)


def test_index_selects_exact_level_and_interval():
    slices = parse_index(INDEX, RUN, 24, 120)
    assert (slices["solar"].start, slices["solar"].end) == (72, 95)
    assert (slices["solar"].start_step, slices["solar"].end_step) == (18, 24)
    assert set(slices) == {"temperature_2m", "u100", "v100", "solar"}


@pytest.mark.parametrize("bad", [
    INDEX.replace("d=2023090700", "d=2023090706"),
    INDEX.replace("100 m above ground", "10 m above ground"),
    INDEX.replace("18-24 hour ave fcst", "0-24 hour acc fcst"),
    INDEX.replace("2:24:", "2:0:"),
    INDEX.replace("18-24 hour ave fcst", "18-25 hour ave fcst"),
])
def test_index_rejects_wrong_cycle_level_offsets_or_time(bad):
    with pytest.raises(GfsError):
        parse_index(bad, RUN, 24, 120)


def transport(*, modified="Thu, 07 Sep 2023 03:45:00 GMT", bad_status=False, bad_etag=False):
    calls = []
    def handler(request):
        calls.append(request)
        headers = {"last-modified": modified, "etag": '"version1"', "accept-ranges": "bytes"}
        if str(request.url).endswith(".idx"):
            return httpx.Response(200, headers=headers, content=INDEX.encode())
        if request.method == "HEAD":
            return httpx.Response(200, headers={**headers, "content-length": "120"})
        assert request.headers["if-match"] == '"version1"'
        span = request.headers["range"].removeprefix("bytes=")
        a, b = (int(part) for part in span.split("-"))
        if bad_etag:
            headers["etag"] = '"version2"'
        return httpx.Response(200 if bad_status else 206,
            headers={**headers, "content-range": f"bytes {a}-{b}/120"},
            content=bytes(range(a, b + 1)))
    return httpx.MockTransport(handler), calls


def test_raw_cache_reuse_revalidates_hashes_without_network(tmp_path):
    mock, calls = transport()
    with httpx.Client(transport=mock) as http:
        client = GfsClient(tmp_path, client=http, retries=0)
        data, meta = client.fetch_endpoint(RUN, CUTOFF, 24)
        assert len(calls) == 6 and not meta["from_cache"]
        again, cached = client.fetch_endpoint(RUN, CUTOFF, 24)
        assert again == data and cached["from_cache"] and len(calls) == 6
        raw = tmp_path / "2023090700" / "f024" / "solar.grib2"
        raw.write_bytes(b"x" * 24)
        with pytest.raises(GfsError, match="SHA"):
            client.fetch_endpoint(RUN, CUTOFF, 24)
        assert len(calls) == 6


@pytest.mark.parametrize("kwargs", [
    {"modified": "Thu, 07 Sep 2023 06:00:01 GMT"},
    {"bad_status": True},
    {"bad_etag": True},
])
def test_publication_or_object_revision_rejected(tmp_path, kwargs):
    mock, _ = transport(**kwargs)
    with httpx.Client(transport=mock) as http:
        client = GfsClient(tmp_path, client=http, retries=0)
        with pytest.raises(GfsError):
            client.fetch_endpoint(RUN, CUTOFF, 24)
    assert not list(tmp_path.rglob("source.json"))


def test_cached_publication_checked_against_requested_cutoff(tmp_path):
    mock, _ = transport()
    with httpx.Client(transport=mock) as http:
        client = GfsClient(tmp_path, client=http, retries=0)
        client.fetch_endpoint(RUN, CUTOFF, 24)
        with pytest.raises(GfsError, match="cutoff"):
            client.fetch_endpoint(RUN, RUN + pd.Timedelta(hours=3), 24)


@pytest.fixture
def synthetic_weather(monkeypatch):
    """Network-free complete day; native decoding must stay on coordinator."""
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def fetch_endpoint(self, run, cutoff, hour, fields):
            messages = {}
            for field in fields:
                start = ((hour - 1) // 6) * 6 if field == "solar" else hour
                messages[field] = {"start_step": start, "end_step": hour, "bytes": 1}
            return {field: b"x" for field in fields}, {
                "publication_max_utc": (run + pd.Timedelta(hours=4)).isoformat(),
                "from_cache": False, "messages": messages,
            }

    def decoder(raw, name, run, hour, points, radiation_tolerance=0.5):
        assert threading.current_thread() is threading.main_thread(), "Native decoder ran concurrently"
        value = {"temperature_2m": 280, "u100": 3, "v100": 4, "solar": 100}[name]
        start = ((hour - 1) // 6) * 6 if name == "solar" else hour
        metadata = {"grid_points": [{"lat": p.latitude, "lon": p.longitude} for p in points]}
        return DecodedField(np.full(len(points), value, dtype=float), start, hour, metadata)

    monkeypatch.setattr(noaa_gfs, "GfsClient", FakeClient)
    monkeypatch.setattr(noaa_gfs, "decode_message", decoder)


@pytest.mark.parametrize("day,hours", [("2023-09-08", 24), ("2024-03-31", 23), ("2024-10-27", 25)])
def test_complete_materializer_roundtrip_and_chronos_input(day, hours, tmp_path, synthetic_weather):
    from chronos2_exogenous.feature_bank import ParquetFeatureSource, build_exogenous_bank

    output = tmp_path / "weather.parquet"
    result = noaa_gfs.materialize_noaa_gfs_weather(
        start_day=day, end_day=day, output_path=output, workers=2,
    )
    frame = pd.read_parquet(output)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert result["row_count"] == hours
    assert manifest["forecast_endpoint_count"] == hours + 1
    assert manifest["production_pit_evidence"] is False
    assert manifest["promotion_eligible"] is False
    assert noaa_gfs.sha256(output.read_bytes()) == manifest["dataset_sha256"]
    for column in ("delivery_start_utc", "run_init_utc", "cutoff_utc", "publication_max_utc"):
        assert str(frame[column].dtype) == "datetime64[ns, UTC]"
    np.testing.assert_allclose(frame["fr_gfs_temperature_2m_c"], 6.85)
    np.testing.assert_allclose(frame["fr_gfs_wind_speed_100m_ms"], 5)
    np.testing.assert_allclose(frame["fr_gfs_shortwave_radiation_wm2"], 100)
    for zone in ("fr", "de", "be", "nl"):
        columns = [column for column in frame if column.startswith(zone + "_gfs_")]
        source = ParquetFeatureSource(
            name=f"noaa_{zone}", family="weather", path=output,
            audit_path=Path(result["manifest_path"]), value_columns={c: c for c in columns},
            timestamp_column="delivery_start_utc", cutoff_column="cutoff_utc",
            information_time_columns=("run_init_utc", "publication_max_utc"),
            age_column="publication_max_utc",
        )
        bank = build_exogenous_bank((source,), start_day=day, end_day=day, require_complete=True)
        assert bank.audit["complete"] is True
        assert bank.audit["production_ready"] is False
        assert bank.audit["sources"][f"noaa_{zone}"]["causality_violations"] == 0
        assert set(columns).issubset(bank.columns_for("chronos"))


def test_materializer_preserves_existing_output(tmp_path, synthetic_weather):
    output = tmp_path / "weather.parquet"
    output.write_bytes(b"existing results")
    with pytest.raises(GfsError):
        noaa_gfs.materialize_noaa_gfs_weather(
            start_day="2023-09-08", end_day="2023-09-08", output_path=output,
        )
    assert output.read_bytes() == b"existing results"


def test_failed_manifest_publish_leaves_no_completed_dataset(tmp_path, synthetic_weather, monkeypatch):
    output = tmp_path / "weather.parquet"
    manifest = output.with_suffix(".manifest.json")
    original_link = noaa_gfs.os.link

    def fail_manifest(source, destination):
        if Path(destination) == manifest:
            raise OSError("simulated final manifest publication failure")
        return original_link(source, destination)

    monkeypatch.setattr(noaa_gfs.os, "link", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        noaa_gfs.materialize_noaa_gfs_weather(
            start_day="2023-09-08", end_day="2023-09-08", output_path=output,
        )
    assert not output.exists() and not manifest.exists()
    assert not list(tmp_path.glob("*.lock"))


def test_concurrent_destination_reservation_is_not_stolen(tmp_path, synthetic_weather):
    output = (tmp_path / "weather.parquet").resolve()
    manifest = output.with_suffix(".manifest.json")
    with noaa_gfs._output_reservations(output, manifest):
        locks = {path: path.read_bytes() for path in tmp_path.glob("*.lock")}
        assert len(locks) == 2
        with pytest.raises(GfsError, match="reserved"):
            noaa_gfs.materialize_noaa_gfs_weather(
                start_day="2023-09-08", end_day="2023-09-08", output_path=output,
            )
        assert {path: path.read_bytes() for path in locks} == locks
    assert not output.exists() and not manifest.exists()
    assert not list(tmp_path.glob("*.lock"))


def test_incomplete_previous_output_is_preserved(tmp_path, synthetic_weather):
    output = tmp_path / "weather.parquet"
    manifest = output.with_suffix(".manifest.json")
    manifest.write_bytes(b"incomplete previous publication")
    with pytest.raises(GfsError, match="already exist"):
        noaa_gfs.materialize_noaa_gfs_weather(
            start_day="2023-09-08", end_day="2023-09-08", output_path=output,
        )
    assert manifest.read_bytes() == b"incomplete previous publication"
    assert not output.exists() and not list(tmp_path.glob("*.lock"))


def test_competing_final_manifest_is_never_overwritten_or_removed(tmp_path, synthetic_weather, monkeypatch):
    output = tmp_path / "weather.parquet"
    manifest = output.with_suffix(".manifest.json")
    original_link = noaa_gfs.os.link

    def racing_manifest(source, destination):
        if Path(destination) == manifest:
            manifest.write_bytes(b"another publisher")
        return original_link(source, destination)

    monkeypatch.setattr(noaa_gfs.os, "link", racing_manifest)
    with pytest.raises(FileExistsError):
        noaa_gfs.materialize_noaa_gfs_weather(
            start_day="2023-09-08", end_day="2023-09-08", output_path=output,
        )
    assert not output.exists()
    assert manifest.read_bytes() == b"another publisher"
    assert not list(tmp_path.glob("*.lock"))
