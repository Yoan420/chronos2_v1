"""Real refresh snapshots and provenance validators; only provider I/O is fake."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import export_nyx_annual_comparisons as export
from chronos2_hourly import nuclear_reporting_refresh as refresh
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.reporting_observations import epex_reporting_identity
from chronos2_hourly.storm_dashboard import storm_dashboard_series


FIRST, STOP = "2026-09-21", "2026-09-24"


def provider_sources(monkeypatch, *, fail_zone=None):
    calls = []
    failed = []
    def actual_fetch(client, *, spec, expected_index, extracted_at_utc):
        calls.append(("actual", spec.zone))
        if spec.zone == fail_zone and not failed:
            failed.append(True)
            raise ConnectionError("provider unavailable once")
        _, current = refresh._support(spec.delivery_day, spec.timezone)
        values = pd.Series(50. + np.sin(np.arange(len(expected_index)) / 23), index=expected_index)
        values.loc[current] = np.nan
        return values, {**epex_reporting_identity(spec.zone, spec.timezone),
                        "extracted_at_utc": str(extracted_at_utc)}
    def storm_fetch(client, *, zone, expected_index, extracted_at_utc):
        calls.append(("storm", zone))
        timezone = export.TIMEZONES[zone]
        _, current = refresh._support(STOP, timezone)
        values = pd.Series(52. + np.sin(np.arange(len(expected_index)) / 23), index=expected_index)
        values.loc[current] = np.nan
        civil = expected_index.tz_convert(timezone).tz_localize(None)
        folds = expected_index[civil.tz_localize(timezone, ambiguous=False).tz_convert("UTC") != expected_index]
        values.loc[folds.difference(current)] = np.nan
        series = storm_dashboard_series(zone)
        return values, {"kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback",
                        "series": series, "primary_series": series, "zone": zone,
                        "fallback_series": f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm",
                        "extracted_at_utc": str(extracted_at_utc), "used_for_prediction": False}
    monkeypatch.setattr(refresh, "create_saturn_client", lambda *a, **k: object())
    monkeypatch.setattr(refresh, "_fetch_epex_observed", actual_fetch)
    monkeypatch.setattr(refresh, "fetch_native_dashboard_snapshot", storm_fetch)
    return calls


def test_real_four_country_snapshots_are_verified_portable_and_read_only_on_resume(tmp_path, monkeypatch):
    calls = provider_sources(monkeypatch)
    source = tmp_path / "personal_pc"
    receipt = export.export(source, FIRST, STOP)
    assert len(calls) == 8
    assert set(receipt["countries"]) == {"FR", "DE", "BE", "NL"}
    for zone, record in receipt["countries"].items():
        snapshot = source / record["snapshot"]
        loaded = _load_verified_snapshot(snapshot, zone=zone, timezone=export.TIMEZONES[zone])
        assert loaded is not None
        actual, storm = export.read_snapshot(snapshot, zone, STOP)
        assert np.isfinite(actual.dropna()).all()
        assert storm.dropna().size > 8700
    destination = tmp_path / "work_pc"
    shutil.copytree(source, destination)
    source.rename(tmp_path / "old_pc_no_longer_at_original_path")
    # Portability rebases the observation path only in memory, preserving all
    # original byte hashes and extraction evidence in the copied archive.
    before = {p.relative_to(destination): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in destination.rglob("*") if p.is_file()}
    assert export.validate_comparisons(destination, FIRST, STOP) == receipt
    assert export.export(destination, FIRST, STOP) == receipt
    assert len(calls) == 8
    assert before == {p.relative_to(destination): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in destination.rglob("*") if p.is_file()}


def test_partial_country_export_resumes_pinned_sources_without_refetch(tmp_path, monkeypatch):
    calls = provider_sources(monkeypatch, fail_zone="DE")
    root = tmp_path / "partial"
    with pytest.raises(ConnectionError, match="unavailable"):
        export.export(root, FIRST, STOP)
    assert (root / "source_checkpoints/FR.json").is_file()
    original = (root / "FR.parquet").read_bytes()
    receipt = export.export(root, FIRST, STOP)
    assert calls.count(("actual", "FR")) == 1
    assert calls.count(("actual", "DE")) == 2
    assert (root / "FR.parquet").read_bytes() == original
    assert export.validate_comparisons(root, FIRST, STOP) == receipt


def test_interruption_after_source_checkpoint_before_values_is_recoverable(tmp_path, monkeypatch):
    calls = provider_sources(monkeypatch)
    original_writer = export._parquet
    failure = []
    def interrupted(path, frame):
        if path.name == "FR.parquet" and not failure:
            failure.append(True)
            raise OSError("write interrupted")
        return original_writer(path, frame)
    monkeypatch.setattr(export, "_parquet", interrupted)
    with pytest.raises(OSError, match="write interrupted"):
        export.export(tmp_path, FIRST, STOP)
    assert (tmp_path / "source_checkpoints/FR.json").is_file()
    assert not (tmp_path / "FR.parquet").exists()
    export.export(tmp_path, FIRST, STOP)
    assert calls.count(("actual", "FR")) == 1


@pytest.mark.parametrize("artifact", ["output", "source", "checkpoint"])
def test_modified_output_source_or_resume_receipt_is_rejected(tmp_path, monkeypatch, artifact):
    provider_sources(monkeypatch)
    receipt = export.export(tmp_path, FIRST, STOP)
    record = receipt["countries"]["FR"]
    path = {"output": tmp_path / "FR.parquet",
            "source": tmp_path / record["snapshot"] / "inputs/observed_latest.parquet",
            "checkpoint": tmp_path / "source_checkpoints/FR.json"}[artifact]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="modified|changed"):
        export.validate_comparisons(tmp_path, FIRST, STOP)


def test_legacy_entsoe_snapshot_passes_generic_verifier_but_not_annual_epex_contract(tmp_path, monkeypatch):
    provider_sources(monkeypatch)
    receipt = export.export(tmp_path, FIRST, STOP)
    directory = tmp_path / receipt["countries"]["FR"]["snapshot"]
    audit_path = directory / export.ARTIFACTS[0]
    audit = json.loads(audit_path.read_text())
    audit["schema_version"] = 1
    audit.pop("observation_policy")
    series = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
    source = audit["observed"]["source"]
    source.update(kind="saturn_target_latest_extraction", series=series)
    source.pop("policy")
    source.pop("actual_reference")
    audit["observed"]["series"] = series
    audit["canonical_actuals"]["source"] = deepcopy(source)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    raw = pd.read_parquet(directory / export.ARTIFACTS[1])
    actual = pd.Series(raw.actual.to_numpy(), index=pd.DatetimeIndex(raw.timestamp))
    check = refresh.verify_refreshed_observations(actual, audit, zone="FR", timezone="Europe/Paris", delivery_day=STOP)
    assert check["actual_reference"] == "ENTSO-E"
    with pytest.raises(ValueError, match="EPEX-only"):
        export.read_snapshot(directory, "FR", STOP)


def test_nonofficial_storm_and_outside_snapshot_paths_are_rejected(tmp_path, monkeypatch):
    provider_sources(monkeypatch)
    receipt = export.export(tmp_path, FIRST, STOP)
    directory = tmp_path / receipt["countries"]["FR"]["snapshot"]
    path = directory / export.ARTIFACTS[0]
    audit = json.loads(path.read_text())
    audit["storm_dashboard"]["source"]["series"] = "unapproved_storm_variant"
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Storm officiel"):
        export.read_snapshot(directory, "FR", STOP)
    for bad in ("../outside", "C:/outside", "C:relative", "/outside"):
        with pytest.raises(ValueError, match="path"):
            export._inside(tmp_path.resolve(), bad)
