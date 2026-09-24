from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly.nuclear_sources import (
    NUCLEAR_ALIAS, NUCLEAR_SERIES, ROOT, audit_nuclear_store,
    build_materialize_command,
)


def _fixture(
    tmp_path: Path, start: str, end: str | None = None,
) -> tuple[Path, pd.DataFrame, dict]:
    frames = []
    days = pd.date_range(start, end or start, freq="D")
    for day in days:
        lo = day.tz_localize("Europe/Paris")
        hi = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
        physical = pd.date_range(lo.tz_convert("UTC"), hi.tz_convert("UTC"), freq="h", inclusive="left")
        cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        frames.append(pd.DataFrame({
            "value_time_utc": physical, "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff, "value": 45.0,
            "downloaded_at_utc": hi.tz_convert("UTC"),
        }))
    frame = pd.concat(frames, ignore_index=True)
    payload = {
        "schema_version": 1, "alias": NUCLEAR_ALIAS, "series": NUCLEAR_SERIES,
        "timezone": "Europe/Paris", "naive_timezone": "Europe/Paris",
        "cutoff_timezone": "Europe/Paris", "cutoff_time": "08:00",
        "daily_broadcast": False, "value_scale": 1.0,
        "incomplete_dst_policy": "duplicate",
        "fill_or_interpolation": "none_except_duplicate_missing_autumn_fold_if_source_is_civil_naive",
        "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
        "snapshot_time_semantics": "query_asof_cutoff",
        "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp is not returned by Client.get(revision_date=...)",
        "provider_revision_timestamp_available": False,
        "start_day": start, "end_day": end or start, "days": len(days),
    }
    path = tmp_path / "nuclear.parquet"
    _write(path, frame, payload)
    return path, frame, payload


def _write(path: Path, frame: pd.DataFrame, payload: dict) -> None:
    frame.to_parquet(path, index=False)
    payload["rows"] = len(frame)
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".audit.json").write_text(json.dumps(payload), encoding="utf-8")


def test_command_is_isolated_and_explicit(tmp_path: Path) -> None:
    command = build_materialize_command("2024-09-08", "2026-09-07", tmp_path / "with spaces.parquet", "python with spaces.exe", 4)
    assert command[0] == "python with spaces.exe"
    arguments = dict(zip(command[2::2], command[3::2]))
    assert arguments["--series"] == NUCLEAR_SERIES
    assert arguments["--alias"] == NUCLEAR_ALIAS
    assert arguments["--naive-timezone"] == "Europe/Paris"
    assert arguments["--cutoff-timezone"] == "Europe/Paris"
    assert arguments["--cutoff-time"] == "08:00"
    assert arguments["--value-scale"] == "1"
    assert arguments["--incomplete-dst-policy"] == "duplicate"
    assert arguments["--end-day"] == "2026-09-07"
    assert "--allow-incomplete-days" not in command
    assert "--merge-existing" not in command
    with pytest.raises(ValueError, match="legacy"):
        build_materialize_command("2026-01-01", "2026-01-01", ROOT / "data/pit/vintages/fr_nuclear_generation_fcst_long.parquet", "python", 1)


@pytest.mark.parametrize("day,hours", [("2025-01-15", 24), ("2025-03-30", 23), ("2025-10-26", 25)])
def test_complete_physical_grids_with_explicit_evidence_limit(tmp_path: Path, day: str, hours: int) -> None:
    path, _, _ = _fixture(tmp_path, day)
    audit = audit_nuclear_store(path, day, day)
    assert audit["complete"], audit["blockers"]
    assert audit["expected_hours"] == audit["covered_hours"] == hours
    assert audit["production_pit_evidence"] is False
    assert audit["provider_revision_timestamp_available"] is False
    assert audit["synthetic_dst_disclosure"]["potential_added_fold_hours"] == (1 if hours == 25 else 0)
    assert audit["synthetic_dst_disclosure"]["exact_repair_count_available"] is False
    assert audit["synthetic_dst_disclosure"]["independent_second_fold_evidence"] is False
    json.dumps(audit, allow_nan=False)


def test_missing_second_fold_is_not_invented_by_auditor(tmp_path: Path) -> None:
    path, frame, payload = _fixture(tmp_path, "2025-10-26")
    absent = pd.Timestamp("2025-10-26T01:00:00Z")
    _write(path, frame.loc[frame.value_time_utc.ne(absent)], payload)
    audit = audit_nuclear_store(path, "2025-10-26", "2025-10-26")
    assert not audit["complete"]
    assert audit["missing_hours"] == [absent.isoformat()]
    assert audit["missing_days"] == ["2025-10-26"]


@pytest.mark.parametrize("day", ["2025-03-31", "2025-10-27"])
def test_cutoff_is_calendar_correct_on_day_after_dst(tmp_path: Path, day: str) -> None:
    path, frame, payload = _fixture(tmp_path, day)
    assert audit_nuclear_store(path, day, day)["complete"]
    frame.loc[0, "snapshot_time_utc"] += pd.Timedelta(minutes=1)
    frame.loc[0, "revision_time_utc"] += pd.Timedelta(minutes=1)
    _write(path, frame, payload)
    audit = audit_nuclear_store(path, day, day)
    assert not audit["complete"]
    assert audit["post_cutoff_rows_excluded"] == 1
    assert audit["missing_hour_count"] == 1


@pytest.mark.parametrize("field,value", [
    ("series", "power.fr.generation.nuclear.remit.mw.fcst"),
    ("naive_timezone", "UTC"), ("value_scale", 0.001),
    ("provider_revision_timestamp_available", True),
    ("snapshot_time_semantics", "provider_publication"),
    ("fill_or_interpolation", "forward_fill"), ("unit", "MW"),
])
def test_source_provenance_is_fail_closed(tmp_path: Path, field: str, value) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    payload[field] = value
    _write(path, frame, payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    assert not audit["source_provenance_valid"]
    assert audit["covered_hours"] == 24


def test_nonfinite_latest_vintage_cannot_use_older_finite_value(tmp_path: Path) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    older = frame.iloc[[0]].copy()
    older["snapshot_time_utc"] -= pd.Timedelta(hours=1)
    older["revision_time_utc"] -= pd.Timedelta(hours=1)
    frame.loc[0, "value"] = float("inf")
    _write(path, pd.concat([frame, older], ignore_index=True), payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    assert audit["missing_hour_count"] == 1
    json.dumps(audit, allow_nan=False)


def test_checksum_mismatch_missing_audit_and_naive_rows(tmp_path: Path) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    frame.loc[0, "value"] = 50.0
    frame.to_parquet(path, index=False)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    assert any("SHA256" in message for message in audit["blockers"])
    frame["value_time_utc"] = frame["value_time_utc"].dt.tz_localize(None)
    _write(path, frame, payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["timing_valid"]
    path.with_name(path.name + ".audit.json").unlink()
    assert not audit_nuclear_store(path, "2026-01-01", "2026-01-01")["complete"]


def test_missing_artifact_is_actionable_and_strict_dst_can_be_requested(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    audit = audit_nuclear_store(path, "2025-10-26", "2025-10-26")
    assert not audit["complete"]
    assert audit["missing_hour_count"] == 25
    command = build_materialize_command("2025-10-26", "2025-10-26", path, "python", 1, "raise")
    assert command[command.index("--incomplete-dst-policy") + 1] == "raise"


def test_context_prefix_can_be_audited_as_a_subset(tmp_path: Path) -> None:
    path, _, _ = _fixture(tmp_path, "2026-01-01", "2026-01-03")
    audit = audit_nuclear_store(path, "2026-01-02", "2026-01-02")
    assert audit["complete"], audit["blockers"]
    assert audit["covered_hours"] == 24


def test_post_cutoff_updates_do_not_replace_the_last_admissible_forecast(tmp_path: Path) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    late = frame.iloc[[0]].copy()
    late["snapshot_time_utc"] += pd.Timedelta(minutes=1)
    late["revision_time_utc"] += pd.Timedelta(minutes=1)
    late["value"] = float("nan")
    _write(path, pd.concat([frame, late], ignore_index=True), payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert audit["complete"], audit["blockers"]
    assert audit["post_cutoff_rows_excluded"] == 1


def test_ambiguous_vintage_identity_and_sidecar_span_are_rejected(tmp_path: Path) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    _write(path, pd.concat([frame, frame.iloc[[0]]], ignore_index=True), payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    assert not audit["timing_valid"]
    payload["days"] = 2
    _write(path, frame, payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    assert not audit["source_provenance_valid"]


@pytest.mark.parametrize("malformed_policy", [["duplicate"], float("nan")])
def test_malformed_sidecar_policy_returns_json_safe_blockers(tmp_path: Path, malformed_policy) -> None:
    path, frame, payload = _fixture(tmp_path, "2026-01-01")
    payload["incomplete_dst_policy"] = malformed_policy
    _write(path, frame, payload)
    audit = audit_nuclear_store(path, "2026-01-01", "2026-01-01")
    assert not audit["complete"]
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("start,end,workers", [
    ("2026-01-02", "2026-01-01", 1),
    ("2026-01-01T01:00", "2026-01-02", 1),
    ("2026-01-01T00:00Z", "2026-01-02", 1),
    ("2026-01-01", "2026-01-02", 0),
])
def test_command_rejects_invalid_bounds(tmp_path: Path, start: str, end: str, workers: int) -> None:
    with pytest.raises(ValueError):
        build_materialize_command(start, end, tmp_path / "n.parquet", "python", workers)
