import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import heatwave_sources as s


def _frame(first, last, value=22.):
    index = s._physical(first, last)
    days = index.tz_convert(s.TIMEZONE).tz_localize(None).normalize()
    cutoffs = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(s.TIMEZONE).tz_convert("UTC")
    return pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": cutoffs, "revision_time_utc": cutoffs,
                         "value": value, "downloaded_at_utc": pd.Timestamp("2026-09-14", tz="UTC")})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    original = tmp_path / "data/pit/kalman_weather"
    original.mkdir(parents=True)
    for alias, spec in s.TEMPERATURE_SOURCES.items():
        path = original / (alias + ".parquet")
        frame = _frame("2025-10-25", "2025-10-26")
        frame.to_parquet(path, index=False)
        metadata = s._metadata(frame, spec, sha256=s._sha(path))
        s._write_json(path.with_name(path.name + ".audit.json"), metadata)
    calls = []
    def fetch(spec, day):
        calls.append((spec["alias"], str(day.date())))
        return _frame(day, day, -4.)
    monkeypatch.setattr(s, "_fetch_day", fetch)
    return tmp_path / "data/pit/heatwave/test", original, calls, fetch


def test_five_forecast_daily_indices_are_reused_without_incumbent_writes(setup):
    output, original, calls, _ = setup
    before = {p: s._sha(p) for p in original.iterdir()}
    result = s.materialize_temperature_sources(output, "2025-10-24", "2025-10-27", 2)
    assert list(result) == list(s.TEMPERATURE_SOURCES)
    assert len(calls) == 10
    for alias, source in result.items():
        assert source["audit"]["ready"]
        assert source["audit"]["covered_hours"] == 97
        assert source["audit"]["seed_hours_reused"] == 49
        assert source["audit"]["unit"] == "degC"
        assert source["audit"]["hourly_temperature_information"] is False
        assert source["audit"]["production_pit_evidence"] is False
        values = pd.read_parquet(source["path"]).value
        assert values.eq(-4).sum() == 48  # Cold forecasts are valid, never clamped to zero.
    assert {p: s._sha(p) for p in original.iterdir()} == before


def test_complete_bundle_reuses_frozen_bytes_without_network_or_seed_refresh(setup, monkeypatch):
    output, original, _, _ = setup
    result = s.materialize_temperature_sources(output, "2025-10-25", "2025-10-27", 1)
    for p in original.iterdir():
        p.write_bytes(b"upstream was updated")
    monkeypatch.setattr(s, "_fetch_day", lambda *a: pytest.fail("No network"))
    assert s.materialize_temperature_sources(output, "2025-10-25", "2025-10-27", 1) == result


def test_failure_preserves_daily_checkpoints_then_resumes_missing_only(setup, monkeypatch):
    output, _, calls, fetch = setup
    def unavailable(spec, day):
        if str(day.date()) == "2025-10-28":
            raise RuntimeError("missing forecast at cutoff")
        return fetch(spec, day)
    monkeypatch.setattr(s, "_fetch_day", unavailable)
    with pytest.raises(s.HeatwaveSourceError, match="missing forecast at cutoff"):
        s.materialize_temperature_sources(output, "2025-10-25", "2025-10-28", 2)
    for alias in s.TEMPERATURE_SOURCES:
        assert (output / alias / "days/2025-10-27").is_dir()
        assert not (output / alias / "bundles/2025-10-25_2025-10-28").exists()
    calls.clear()
    monkeypatch.setattr(s, "_fetch_day", fetch)
    s.materialize_temperature_sources(output, "2025-10-25", "2025-10-28", 2)
    assert len(calls) == 5 and all(day == "2025-10-28" for _, day in calls)


@pytest.mark.parametrize("field,value", [("series", "meteo.actual.fr"), ("daily_broadcast", False),
    ("causal_contract", "latest"), ("snapshot_time_semantics", "latest"),
    ("provider_revision_timestamp_available", True), ("cutoff_time", "10:30"),
    ("value_scale", 1000), ("fill_or_interpolation", "interpolated"),
    ("unit", "GW"), ("unit", "K"), ("actual_weather_used", True)])
def test_seed_semantics_fail_closed(setup, field, value):
    output, original, _, _ = setup
    sidecar = original / "fr_temperature_fcst.parquet.audit.json"
    metadata = json.loads(sidecar.read_text())
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(s.HeatwaveSourceError):
        s.materialize_temperature_sources(output, "2025-10-25", "2025-10-26", 1)


@pytest.mark.parametrize("mutation", ["sha", "naive", "cutoff", "download", "missing", "duplicate", "unordered", "varies", "kelvin", "bool", "nan"])
def test_bad_source_bytes_or_causality_rejected(setup, mutation):
    output, original, _, _ = setup
    path = original / "fr_temperature_fcst.parquet"
    frame = pd.read_parquet(path)
    if mutation == "sha":
        frame.loc[0, "value"] += 1
    elif mutation == "naive":
        frame["snapshot_time_utc"] = frame.snapshot_time_utc.dt.tz_localize(None)
    elif mutation == "cutoff":
        frame.loc[0, "revision_time_utc"] += pd.Timedelta(hours=1)
    elif mutation == "download":
        frame.loc[0, "downloaded_at_utc"] = pd.Timestamp("2020-01-01", tz="UTC")
    elif mutation == "missing":
        frame = frame.iloc[1:]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "unordered":
        frame = frame.iloc[::-1]
    elif mutation == "varies":
        frame.loc[0, "value"] = 24
    elif mutation == "kelvin":
        frame["value"] = 300.
    elif mutation == "bool":
        frame["value"] = False
    else:
        frame.loc[0, "value"] = np.nan
    frame.to_parquet(path, index=False)
    if mutation != "sha":
        sidecar = path.with_name(path.name + ".audit.json")
        metadata = json.loads(sidecar.read_text())
        metadata["sha256"] = s._sha(path)
        sidecar.write_text(json.dumps(metadata))
    with pytest.raises(s.HeatwaveSourceError):
        s.materialize_temperature_sources(output, "2025-10-25", "2025-10-26", 1)


def test_spring_dst_uses_23_hours_and_following_day_civil_cutoff(setup):
    output, _, _, _ = setup
    result = s.materialize_temperature_sources(output, "2026-03-29", "2026-03-30", 1)
    for source in result.values():
        frame = pd.read_parquet(source["path"])
        assert len(frame) == 47
        assert frame.snapshot_time_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(8).all()


def test_mutated_bundle_is_never_overwritten(setup):
    output, _, _, _ = setup
    result = s.materialize_temperature_sources(output, "2025-10-25", "2025-10-26", 1)
    path = Path(result["fr_temperature_fcst"]["path"])
    path.write_bytes(b"corrupt")
    with pytest.raises(s.HeatwaveSourceError):
        s.materialize_temperature_sources(output, "2025-10-25", "2025-10-26", 1)
    assert path.read_bytes() == b"corrupt"


@pytest.mark.parametrize("workers", [0, 5, True, 1.5])
def test_parallelism_is_bounded(setup, workers):
    with pytest.raises(s.HeatwaveSourceError):
        s.materialize_temperature_sources(setup[0], "2025-10-25", "2025-10-26", workers)


@pytest.mark.parametrize("first,last", [("2025-10-26", "2025-10-25"), ("2025-10-25T08:00", "2025-10-26"),
    ("2025-10-25T00:00Z", "2025-10-26"), ("2000-01-01", "2026-01-01")])
def test_ranges_are_bounded_civil_dates(setup, first, last):
    with pytest.raises(s.HeatwaveSourceError):
        s.materialize_temperature_sources(setup[0], first, last)


def test_existing_pipeline_output_roots_are_forbidden(setup):
    with pytest.raises(s.HeatwaveSourceError, match="data/pit/heatwave"):
        s.materialize_temperature_sources(setup[1], "2025-10-25", "2025-10-26")


def test_directory_publish_retries_transient_windows_scanner_without_replacement(tmp_path, monkeypatch):
    stage, destination = tmp_path / "stage", tmp_path / "final"
    stage.mkdir()
    real_rename, calls = Path.rename, []
    def flaky(path, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError("Windows antivirus transient lock")
        return real_rename(path, target)
    monkeypatch.setattr(Path, "rename", flaky)
    monkeypatch.setattr(s.time, "sleep", lambda _: None)
    s._rename_new_directory(stage, destination)
    assert destination.is_dir() and len(calls) == 2
    with pytest.raises(s.HeatwaveSourceError, match="already exists"):
        s._rename_new_directory(stage, destination)
