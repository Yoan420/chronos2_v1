"""Daily NOAA backfill recovery and complete-timeline publication, offline."""
import json
from pathlib import Path
import ssl
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

import run_noaa_gfs_backfill as backfill
from auxiliary_lab import noaa_gfs as gfs
from auxiliary_lab.weather import delivery_utc_index


def write_partition(output, day, zones=gfs.DEFAULT_ZONES, tolerance=0.5):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    contract = backfill._contract(day, day, zones, tolerance)
    index = delivery_utc_index(day)
    run, cutoff = gfs.issue_times(day)
    publication = run + pd.Timedelta(hours=4)
    frame = pd.DataFrame({"delivery_start_utc": index, "delivery_day": day,
                          "run_init_utc": run, "cutoff_utc": cutoff,
                          "publication_max_utc": publication})
    for column in backfill.TIME_COLUMNS:
        frame[column] = frame[column].astype("datetime64[ns, UTC]")
    for zone in zones:
        for variable, value in zip(backfill.VARIABLES, [10.0, 5.0, 100.0]):
            frame[f"{zone.lower()}_gfs_{variable}"] = value
    frame.to_parquet(output, index=False)
    digest = backfill.file_sha256(output)
    first = int((index[0] - run) / pd.Timedelta(hours=1))
    sources = [{"forecast_hour": hour, "url": gfs.object_url(run, hour),
                "run_init_utc": run.isoformat(), "publication_max_utc": publication.isoformat()}
               for hour in range(first, first + len(index) + 1)]
    audit = {**contract, "schema_version": gfs.SCHEMA_VERSION,
             "output_path": str(output), "dataset_sha256": digest, "output_sha256": digest,
             "day_count": 1, "row_count": len(index), "forecast_endpoint_count": len(sources),
             "sources": sources, "evidence_kind": "historical_archive_publication",
             "local_prospective_capture": False, "production_pit_evidence": False,
             "production_pipeline_evidence": False, "promotion_eligible": False}
    output.with_suffix(".manifest.json").write_text(json.dumps(audit), encoding="utf-8")
    return frame


@pytest.fixture
def offline(monkeypatch, tmp_path):
    calls = []

    def materialize(**kwargs):
        calls.append(kwargs)
        assert kwargs["start_day"] == kwargs["end_day"]
        assert kwargs["workers"] == 2
        write_partition(kwargs["output_path"], kwargs["start_day"], kwargs["zones"], kwargs["radiation_tolerance"])

    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", materialize)
    monkeypatch.setattr(backfill.shutil, "disk_usage", lambda path: SimpleNamespace(free=500 * 1024**3))
    options = {"start_day": "2024-03-30", "end_day": "2024-04-01",
               "output_root": tmp_path / "run", "cache_dir": tmp_path / "cache"}
    return options, calls


def test_daily_pilot_resumes_and_aggregate_includes_all_physical_hours(offline):
    options, calls = offline
    partial = backfill.run_backfill(**options, max_new_days=1)
    root = options["output_root"]
    assert partial["status"] == "partial" and partial["verified_days"] == 1
    assert not (root / "aggregate").exists()
    first_sha = backfill.file_sha256(root / "partitions" / "2024-03-30.parquet")
    result = backfill.run_backfill(**options)
    assert result["status"] == "complete" and result["row_count"] == 71
    assert len(calls) == 3
    assert backfill.file_sha256(root / "partitions" / "2024-03-30.parquet") == first_sha
    assert backfill.validate_aggregate(root) == result
    assert backfill.run_backfill(**options) == result
    assert len(calls) == 3
    audit = json.loads(Path(result["manifest_path"]).read_text())
    assert audit["production_pit_evidence"] is False and audit["promotion_eligible"] is False
    assert audit["cutoff_time"] == "08:00" and audit["cutoff_timezone"] == "Europe/Paris"
    assert len(audit["partitions"]) == 3
    events = [json.loads(line)["event"] for line in (root / "journal.jsonl").read_text().splitlines()]
    assert events.count("day_completed") == 3 and "paused_at_daily_limit" in events


def test_verified_seed_is_copied_without_modifying_source(offline, tmp_path):
    options, calls = offline
    seed = tmp_path / "seeds"
    seed_output = seed / "2024-03-30.parquet"
    write_partition(seed_output, "2024-03-30")
    before = {p.name: backfill.file_sha256(p) for p in seed.iterdir()}
    partial = backfill.run_backfill(**options, seed_directory=seed, max_new_days=0)
    assert partial["verified_days"] == 1 and not calls
    assert {p.name: backfill.file_sha256(p) for p in seed.iterdir()} == before
    copied = options["output_root"] / "partitions" / seed_output.name
    assert backfill.file_sha256(copied) == before[seed_output.name]
    audit = json.loads(copied.with_suffix(".manifest.json").read_text())
    assert audit["seed_provenance"]["manifest_sha256"] == before["2024-03-30.manifest.json"]
    assert Path(audit["output_path"]) == copied


@pytest.mark.parametrize("change", ["parquet", "manifest", "source_version"])
def test_resume_refuses_changed_registered_partition_or_source(offline, change):
    options, calls = offline
    backfill.run_backfill(**options, max_new_days=1)
    root = options["output_root"]
    if change == "parquet":
        with (root / "partitions" / "2024-03-30.parquet").open("ab") as target:
            target.write(b"changed")
    elif change == "manifest":
        manifest = root / "partitions" / "2024-03-30.manifest.json"
        audit = json.loads(manifest.read_text())
        audit["row_count"] += 1
        manifest.write_text(json.dumps(audit))
    else:
        path = root / "contract.json"
        contract = json.loads(path.read_text())
        contract["materializer_source_sha256"] = "0" * 64
        path.write_text(json.dumps(contract))
    with pytest.raises(gfs.GfsError, match="SHA|contract changed"):
        backfill.run_backfill(**options)
    assert len(calls) == 1 and not (root / "aggregate").exists()


def test_incomplete_partition_is_not_silently_rebuilt(offline):
    options, calls = offline
    partition = options["output_root"] / "partitions" / "2024-03-30.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"interrupted write")
    with pytest.raises(gfs.GfsError, match="Incomplete daily partition"):
        backfill.run_backfill(**options)
    assert not calls and partition.read_bytes() == b"interrupted write"


def test_disk_preflight_and_existing_run_lock_prevent_download(offline, monkeypatch):
    options, calls = offline
    monkeypatch.setattr(backfill.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**3))
    with pytest.raises(gfs.GfsError, match="Insufficient free disk"):
        backfill.run_backfill(**options)
    assert not calls
    with backfill._run_lock(options["output_root"]):
        with pytest.raises(gfs.GfsError, match="already reserved"):
            backfill.run_backfill(**options)
    assert not calls


def test_aggregate_values_must_equal_registered_daily_partitions(offline):
    options, _ = offline
    result = backfill.run_backfill(**options)
    output = Path(result["output_path"])
    frame = pd.read_parquet(output)
    frame.loc[0, "fr_gfs_temperature_2m_c"] += 1
    frame.to_parquet(output, index=False)
    manifest_path = Path(result["manifest_path"])
    audit = json.loads(manifest_path.read_text())
    audit["dataset_sha256"] = audit["output_sha256"] = backfill.file_sha256(output)
    manifest_path.write_text(json.dumps(audit))
    with pytest.raises(gfs.GfsError, match="values differ"):
        backfill.validate_aggregate(options["output_root"])


def test_long_context_window_dry_plan_is_network_free(offline):
    options, calls = offline
    options.update(start_day="2023-06-08", end_day="2026-09-02")
    result = backfill.run_backfill(**options, max_new_days=7, dry_run=True)
    assert result["total_days"] == 1183 and result["new_days_this_run"] == 7
    assert result["disk"]["estimated_raw_bytes"] == 630_000_000
    assert not calls and not (options["output_root"] / "aggregate").exists()


def test_daily_protocol_retry_reuses_same_cache_and_journals_bounded_waits(offline, monkeypatch):
    options, calls = offline
    materialize = gfs.materialize_noaa_gfs_weather
    attempts, waits = [], []
    cache_marker = options["cache_dir"] / "already_verified_message.grib2"
    cache_marker.parent.mkdir()
    cache_marker.write_bytes(b"preserved raw response")

    def interrupted(**kwargs):
        attempts.append(kwargs)
        assert cache_marker.read_bytes() == b"preserved raw response"
        if len(attempts) <= 2:
            raise httpx.RemoteProtocolError("peer closed response before its declared length")
        return materialize(**kwargs)

    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", interrupted)
    monkeypatch.setattr(backfill.time, "sleep", waits.append)
    result = backfill.run_backfill(**options, max_new_days=1, day_retries=2)
    assert result["verified_days"] == 1 and len(calls) == 1
    assert len(attempts) == 3 and attempts[0] == attempts[1] == attempts[2]
    assert waits == [1, 2]
    assert cache_marker.read_bytes() == b"preserved raw response"
    events = [json.loads(line) for line in (options["output_root"] / "journal.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "started" and events[0]["day_retries"] == 2
    retries = [event for event in events if event["event"] == "day_retry_scheduled"]
    assert [event["next_attempt"] for event in retries] == [2, 3]
    assert [event["wait_seconds"] for event in retries] == waits
    assert all(event["maximum_attempts"] == 3 for event in retries)


@pytest.mark.parametrize("day_retries", [0, 2, 4])
def test_daily_protocol_retry_stops_at_configured_limit(offline, monkeypatch, day_retries):
    options, _ = offline
    attempts, waits = [], []

    def interrupted(**kwargs):
        attempts.append(kwargs)
        raise httpx.RemoteProtocolError("truncated response")

    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", interrupted)
    monkeypatch.setattr(backfill.time, "sleep", waits.append)
    with pytest.raises(httpx.RemoteProtocolError):
        backfill.run_backfill(**options, day_retries=day_retries)
    assert len(attempts) == day_retries + 1
    assert waits == [1, 2, 4, 8][:day_retries]
    assert not (options["output_root"] / "aggregate").exists()
    events = [json.loads(line) for line in (options["output_root"] / "journal.jsonl").read_text().splitlines()]
    assert events[-2]["event"] == "day_retries_exhausted"
    assert events[-2]["failed_attempt"] == day_retries + 1
    assert events[-1]["event"] == "failed"


@pytest.mark.parametrize("kind", ["gfs_contract", "wrapped_protocol", "certificate", "protocol_certificate", "http_permanent", "direct_timeout"])
def test_daily_retry_never_relaxes_contract_tls_or_other_failure_policy(offline, monkeypatch, kind):
    options, _ = offline
    attempts, waits = [], []

    def failure(**kwargs):
        attempts.append(kwargs)
        if kind == "gfs_contract":
            raise gfs.GfsError("publication after cutoff")
        if kind == "wrapped_protocol":
            raise gfs.GfsError("materializer exhausted its own retries") from httpx.RemoteProtocolError("interrupted")
        if kind == "certificate":
            raise ssl.SSLCertVerificationError("certificate verify failed")
        if kind == "protocol_certificate":
            raise httpx.RemoteProtocolError("TLS peer failed") from ssl.SSLCertVerificationError("certificate verify failed")
        if kind == "http_permanent":
            request = httpx.Request("GET", "https://noaa-gfs-bdp-pds.s3.amazonaws.com/missing")
            raise httpx.HTTPStatusError("not found", request=request, response=httpx.Response(404, request=request))
        raise httpx.ReadTimeout("direct timeout remains governed by the materializer")

    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", failure)
    monkeypatch.setattr(backfill.time, "sleep", waits.append)
    with pytest.raises((gfs.GfsError, ssl.SSLCertVerificationError, httpx.HTTPError)):
        backfill.run_backfill(**options, day_retries=4)
    assert len(attempts) == 1 and not waits
    events = [json.loads(line)["event"] for line in (options["output_root"] / "journal.jsonl").read_text().splitlines()]
    assert "day_retry_scheduled" not in events


@pytest.mark.parametrize("artifact", ["output", "manifest", "lock", "staging"])
def test_daily_retry_refuses_started_publication_without_deleting_it(offline, monkeypatch, artifact):
    options, _ = offline
    attempts, waits, created = [], [], []

    def interrupted(**kwargs):
        attempts.append(kwargs)
        output = Path(kwargs["output_path"])
        manifest = output.with_suffix(".manifest.json")
        path = {"output": output, "manifest": manifest,
                "lock": manifest.with_name(manifest.name + ".publish.lock"),
                "staging": output.with_name(output.name + ".owned-by-writer.tmp")}[artifact]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"publication state must be preserved")
        created.append(path)
        raise httpx.RemoteProtocolError("interrupted after writer state became visible")

    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", interrupted)
    monkeypatch.setattr(backfill.time, "sleep", waits.append)
    with pytest.raises(gfs.GfsError, match="refusing retry"):
        backfill.run_backfill(**options, day_retries=2)
    assert len(attempts) == 1 and not waits
    assert created[0].read_bytes() == b"publication state must be preserved"
    events = [json.loads(line)["event"] for line in (options["output_root"] / "journal.jsonl").read_text().splitlines()]
    assert "day_retry_refused" in events and "day_retry_scheduled" not in events


def test_daily_retry_rechecks_frozen_materializer_sha_before_next_attempt(offline, monkeypatch):
    options, _ = offline
    attempts = []
    original_sha = backfill.file_sha256

    def checked_sha(path):
        if attempts and Path(path) == Path(gfs.__file__):
            return "0" * 64
        return original_sha(path)

    def interrupted(**kwargs):
        attempts.append(kwargs)
        raise httpx.RemoteProtocolError("truncated response")

    monkeypatch.setattr(backfill, "file_sha256", checked_sha)
    monkeypatch.setattr(gfs, "materialize_noaa_gfs_weather", interrupted)
    monkeypatch.setattr(backfill.time, "sleep", lambda seconds: None)
    with pytest.raises(gfs.GfsError, match="source changed before a daily attempt"):
        backfill.run_backfill(**options, day_retries=2)
    assert len(attempts) == 1


@pytest.mark.parametrize("invalid", [-1, 5, 1.5, True])
def test_daily_retry_bound_is_validated_before_any_work(offline, invalid):
    options, calls = offline
    with pytest.raises(gfs.GfsError, match="day_retries"):
        backfill.run_backfill(**options, day_retries=invalid)
    assert not calls and not options["output_root"].exists()
