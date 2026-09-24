from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly import model_storm_data as loader
from chronos2_hourly.model_storm_data import load_model_storm_payload
from chronos2_hourly.nuclear_run_archive import save_nuclear_result_bundle
from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources
from test_nuclear_run_archive import _fixture
from test_nuclear_reporting_refresh import CONFIG, TZ, sources


DAY = "2026-09-10"


@pytest.fixture
def tmp_path(tmp_path_factory):
    # The production snapshot layout is deep; keep Windows fixture paths short.
    return tmp_path_factory.mktemp("ms")


def workspace(project: Path, zone="FR", day=DAY):
    return project / "runs/experiments/nuclear_forecast_v1" / day / zone.lower() / "civil_pit_v2"


def frozen(project: Path, zone="FR", day=DAY):
    _, result = _fixture(project / "fixture", day)
    result.audit["zone"] = zone
    work = workspace(project, zone, day)
    (work / "snapshot").mkdir(parents=True)
    source = work / "snapshot/target.csv"
    source.write_text("frozen target", encoding="utf-8")
    config = work / "resolved_config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "input_snapshot.json").write_text(json.dumps({
        "identity": {"zone": zone, "delivery_day": day},
        "files": [{"snapshot": str(source), "sha256": sha(source)}],
        "resolved_config_sha256": sha(config),
    }), encoding="utf-8")
    return save_nuclear_result_bundle(result, workdir=work)


def refreshed(project, monkeypatch, *, day=DAY, actual_count=24, storm_count=24):
    sources(monkeypatch, day=day, actual_count=actual_count, storm_count=storm_count)
    return refresh_nuclear_reporting_sources(
        CONFIG, "FR", TZ, day, workspace(project, day=day) / "report_only/sources", client=object())


def zone(payload, key="FR"):
    return next(item for item in payload["zones"] if item["zone"] == key)


def test_real_frozen_bundle_and_audited_sources_load_exclusive_model_without_mutation(tmp_path, monkeypatch):
    frozen(tmp_path)
    refreshed(tmp_path, monkeypatch)
    fingerprint = lambda: {str(path): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                           for path in tmp_path.rglob("*") if path.is_file()}
    before = fingerprint()
    payload = load_model_storm_payload(tmp_path, DAY)
    fr = zone(payload)
    assert payload["has_data"] is True
    assert [item["zone"] for item in payload["zones"]] == ["BE", "DE", "FR", "NL"]
    assert fr["status"] == "complete"
    assert fr["coverage"] == {"model": 24, "storm": 24, "observed": 24}
    assert all(row["model"] == 51.5 and row["storm"] == 125 and row["observed"] == 120 for row in fr["rows"])
    assert all(row["model_p10"] == 41.5 and row["model_p90"] == 61.5 for row in fr["rows"])
    assert set(fr["sources"]) == {"model", "storm", "observed"}
    assert fr["sources"]["model"]["column"] == "residual_kalman__q50"
    assert fr["sources"]["model"]["quantile_columns"] == {
        "p10": "residual_kalman__q10", "p50": "residual_kalman__q50", "p90": "residual_kalman__q90"}
    assert len(fr["sources"]["storm"]["artifact_sha256"]) == 64
    assert len(fr["sources"]["observed"]["artifact_sha256"]) == 64
    assert zone(payload, "DE")["status"] == "unavailable"
    assert all(row["model"] is None and row["model_p10"] is None and row["model_p90"] is None
               for row in zone(payload, "DE")["rows"])
    assert fingerprint() == before
    json.dumps(payload, allow_nan=False)


def test_previous_delivery_and_regular_kalman_are_never_model_fallbacks(tmp_path):
    frozen(tmp_path)
    ordinary = tmp_path / "runs/exports/2026-09-11/fr_kalman.csv"
    ordinary.parent.mkdir(parents=True)
    ordinary.write_text("timestamp,q50\n2026-09-10T22:00:00+00:00,999\n")
    result = load_model_storm_payload(tmp_path, "2026-09-11")
    assert result["has_data"] is False
    assert all(item["status"] == "unavailable" for item in result["zones"])


def test_de_nuclear_bundle_becomes_available_automatically(tmp_path):
    frozen(tmp_path, "DE")
    de = zone(load_model_storm_payload(tmp_path, DAY), "DE")
    assert de["coverage"]["model"] == 24
    assert all(row["model"] == 51.5 for row in de["rows"])
    assert de["timezone"] == "Europe/Berlin"


def test_sources_remain_available_when_model_missing_or_rejected(tmp_path, monkeypatch):
    directory = frozen(tmp_path)
    refreshed(tmp_path, monkeypatch)
    (directory / "kalman_forecast.parquet").write_bytes(b"corrupt")
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["status"] == "partial"
    assert fr["sources"]["model"]["status"] == "invalid"
    assert fr["coverage"] == {"model": 0, "storm": 24, "observed": 24}
    assert all(row["model"] is None and row["model_p10"] is None and row["model_p90"] is None for row in fr["rows"])


def test_independent_storm_or_observed_corruption_is_visible_without_losing_other_source(tmp_path, monkeypatch):
    _, directory, _ = refreshed(tmp_path, monkeypatch)
    (directory / "inputs/observed_latest.parquet").write_bytes(b"corrupt")
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["sources"]["observed"]["status"] == "invalid"
    assert fr["coverage"]["storm"] == 24 and fr["coverage"]["observed"] == 0


def test_newest_invalid_publication_does_not_fall_back_to_valid_older_snapshot(tmp_path, monkeypatch):
    refreshed(tmp_path, monkeypatch)
    _, newest, _ = refreshed(tmp_path, monkeypatch)
    (newest / "inputs/storm_dashboard_official_statistics.parquet").write_bytes(b"corrupt")
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["sources"]["storm"]["status"] == "invalid"
    assert fr["coverage"]["storm"] == 0
    assert fr["coverage"]["observed"] == 24


def test_partial_observation_day_remains_empty_and_partial_storm_remains_explicit(tmp_path, monkeypatch):
    refreshed(tmp_path, monkeypatch, actual_count=12, storm_count=12)
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["coverage"] == {"model": 0, "storm": 12, "observed": 0}
    assert all(row["observed"] is None for row in fr["rows"])
    assert fr["sources"]["storm"]["status"] == "partial"
    assert sum(row["storm"] is None for row in fr["rows"]) == 12


@pytest.mark.parametrize("day,hours", [(DAY, 24), ("2026-03-29", 23), ("2026-10-25", 25)])
def test_physical_dst_days_and_repeated_local_hour_keep_distinct_utc_keys(tmp_path, monkeypatch, day, hours):
    frozen(tmp_path, day=day)
    refreshed(tmp_path, monkeypatch, day=day, actual_count=hours, storm_count=hours)
    result = load_model_storm_payload(tmp_path, day)
    assert all(item["expected_hours"] == hours and len(item["rows"]) == hours for item in result["zones"])
    fr = zone(result)
    assert len({row["timestamp_utc"] for row in fr["rows"]}) == hours
    assert fr["coverage"]["model"] == hours
    assert all(row["model_p10"] == 41.5 and row["model"] == 51.5 and row["model_p90"] == 61.5
               for row in fr["rows"])
    assert set(fr["coverage"]) == {"model", "storm", "observed"}
    if hours == 25:
        assert [row["local_label"] for row in fr["rows"]].count("02:00") == 2


def test_snapshot_identity_must_match_zone_and_requested_day(tmp_path, monkeypatch):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    audit["delivery_day_local"] = "2026-09-09"
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["status"] == "invalid"
    assert fr["coverage"] == {"model": 0, "storm": 0, "observed": 0}


def test_malformed_observed_metadata_is_reported_without_crashing_other_series(tmp_path, monkeypatch):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    audit["observed"] = []
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["sources"]["observed"]["status"] == "invalid"
    assert fr["coverage"]["storm"] == 24


@pytest.mark.parametrize("day", ["2026-09-10T12:00:00", "2026-09-10T00:00:00Z", "not-a-day", "NaT"])
def test_invalid_delivery_date_is_rejected(tmp_path, day):
    with pytest.raises(ValueError):
        load_model_storm_payload(tmp_path, day)


def test_relative_nuclear_root_resolves_under_project(tmp_path):
    result = load_model_storm_payload(tmp_path, DAY, Path("custom/nuclear"))
    assert result["has_data"] is False
    assert len(result["zones"]) == 4


@pytest.mark.parametrize("case", ["missing_p10", "missing_p90", "nan_p10", "infinite_p90",
                                 "crossed_lower", "crossed_upper", "missing_hour", "extra_hour",
                                 "duplicate_hour"])
def test_model_envelope_rejects_invalid_quantiles_or_incomplete_day(tmp_path, monkeypatch, case):
    # Independently test the report boundary even if a future archive reader is relaxed.
    frozen(tmp_path)
    _, result = _fixture(tmp_path / "invalid_fixture", DAY)
    frame = result.kalman_view.forecast
    if case.startswith("missing_p"):
        frame.drop(columns="residual_kalman__q" + case.removeprefix("missing_p"), inplace=True)
    elif case == "nan_p10":
        frame.loc[0, "residual_kalman__q10"] = float("nan")
    elif case == "infinite_p90":
        frame.loc[0, "residual_kalman__q90"] = float("inf")
    elif case == "crossed_lower":
        frame.loc[0, "residual_kalman__q10"] = 52
    elif case == "crossed_upper":
        frame.loc[0, "residual_kalman__q90"] = 51
    elif case == "missing_hour":
        result.kalman_view.forecast = frame.iloc[1:].copy()
    elif case == "extra_hour":
        extra = frame.iloc[[-1]].copy()
        extra["delivery_start_utc"] += pd.Timedelta(hours=1)
        result.kalman_view.forecast = pd.concat([frame, extra], ignore_index=True)
    else:
        result.kalman_view.forecast = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    monkeypatch.setattr(loader, "load_nuclear_result_bundle", lambda **kwargs: result)
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["sources"]["model"]["status"] == "invalid"
    assert fr["coverage"]["model"] == 0
    assert all(row["model"] is None and row["model_p10"] is None and row["model_p90"] is None
               for row in fr["rows"])
    json.dumps(fr, allow_nan=False)


def test_equal_and_negative_quantiles_are_preserved_without_clipping(tmp_path, monkeypatch):
    frozen(tmp_path)
    _, result = _fixture(tmp_path / "negative_fixture", DAY)
    result.kalman_view.forecast["residual_kalman__q10"] = -25.0
    result.kalman_view.forecast["residual_kalman__q50"] = -25.0
    result.kalman_view.forecast["residual_kalman__q90"] = -10.0
    monkeypatch.setattr(loader, "load_nuclear_result_bundle", lambda **kwargs: result)
    fr = zone(load_model_storm_payload(tmp_path, DAY))
    assert fr["coverage"]["model"] == 24
    assert all((row["model_p10"], row["model"], row["model_p90"]) == (-25.0, -25.0, -10.0)
               for row in fr["rows"])
