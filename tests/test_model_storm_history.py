"""Historical CWE extraction from synthetic verified archives; no model or network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import model_storm_data as loader
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources
from chronos2_hourly.nuclear_run_archive import save_nuclear_result_bundle
from test_model_storm_data import workspace, zone
from test_nuclear_reporting_refresh import CONFIG, TZ, sources
from test_nuclear_run_archive import _fixture


SOURCE_DAY = "2026-09-15"


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Keep the real archive/snapshot layout within Windows path-length limits.
    return tmp_path_factory.mktemp("msh")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(root):
    return {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, sha(path))
            for path in root.rglob("*") if path.is_file()}


def archive(project, source_day=SOURCE_DAY, target_day="2026-09-14"):
    _, result = _fixture(project / "fixture", source_day)
    target_index = local_delivery_day_index(target_day, timezone=TZ)
    frame = result.kalman_view.backtest
    positions = frame["delivery_start_utc"].isin(target_index)
    if positions.sum() == len(target_index):
        median = np.arange(len(target_index), dtype=float) - 20
        for column, offset in (("q10", -8), ("q50", 0), ("q90", 12)):
            frame.loc[positions, "residual_kalman__" + column] = median + offset
    work = workspace(project, day=source_day)
    (work / "snapshot").mkdir(parents=True)
    frozen_source = work / "snapshot/target.csv"
    frozen_source.write_text("explicit immutable historical fixture", encoding="utf-8")
    config = work / "resolved_config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    (work / "input_snapshot.json").write_text(json.dumps({
        "identity": {"zone": "FR", "delivery_day": source_day},
        "files": [{"snapshot": str(frozen_source), "sha256": sha(frozen_source)}],
        "resolved_config_sha256": sha(config),
    }), encoding="utf-8")
    directory = save_nuclear_result_bundle(result, workdir=work)
    return SimpleNamespace(work=work, directory=directory, result=result, target_index=target_index)


def audited_sources(project, monkeypatch, source_day, target_day):
    _, _, observed, storm, calls = sources(monkeypatch, day=source_day, actual_count=24,
                                          storm_count=24, dst_hole=False)
    target_index = local_delivery_day_index(target_day, timezone=TZ)
    observed.loc[target_index] = np.arange(len(target_index)) + 1000.0
    storm.loc[target_index] = np.arange(len(target_index)) + 2000.0
    _, directory, audit = refresh_nuclear_reporting_sources(
        CONFIG, "FR", TZ, source_day, workspace(project, day=source_day) / "report_only/sources",
        client=object())
    assert len(calls) == 2  # Both fetch functions above are local synthetic fixtures.
    return directory, audit, calls


def assert_no_model(fr):
    assert fr["coverage"]["model"] == 0
    assert all(row["model"] is None and row["model_p10"] is None and row["model_p90"] is None
               for row in fr["rows"])
    json.dumps(fr, allow_nan=False)


@pytest.mark.parametrize("target_day,source_day,hours", [
    ("2026-09-13", SOURCE_DAY, 24), ("2026-09-14", SOURCE_DAY, 24),
    ("2026-03-29", "2026-03-30", 23), ("2026-10-25", "2026-10-26", 25),
])
def test_exact_historical_slice_quantiles_and_three_audited_series_without_mutation(
        tmp_path, monkeypatch, target_day, source_day, hours):
    bundle = archive(tmp_path, source_day, target_day)
    _, _, calls = audited_sources(tmp_path, monkeypatch, source_day, target_day)
    before = fingerprint(tmp_path)
    payload = loader.load_model_storm_payload(tmp_path, target_day, history_from_delivery=source_day)
    fr = zone(payload)
    assert payload["delivery_day"] == target_day
    assert payload["history_from_delivery"] == source_day
    assert payload["has_data"] is True
    assert fr["status"] == "complete"
    assert fr["expected_hours"] == hours
    assert fr["coverage"] == {"model": hours, "storm": hours, "observed": hours}
    assert [row["timestamp_utc"] for row in fr["rows"]] == [stamp.isoformat() for stamp in bundle.target_index]
    for number, row in enumerate(fr["rows"]):
        assert (row["model_p10"], row["model"], row["model_p90"]) == (number - 28, number - 20, number - 8)
        assert row["observed"] == number + 1000
        assert row["storm"] == number + 2000
    for name in ("model", "storm", "observed"):
        source = fr["sources"][name]
        assert source["source_delivery_day"] == source_day
        assert source["delivery_day"] == target_day
        assert len(source["artifact_sha256"]) == 64
    model = fr["sources"]["model"]
    assert model["kind"] == "nuclear_kalman_historical_replay"
    assert Path(model["artifact_path"]) == bundle.directory / "kalman_backtest.parquet"
    assert model["artifact_sha256"] == sha(bundle.directory / "kalman_backtest.parquet")
    assert all(item["expected_hours"] == hours for item in payload["zones"])
    if hours == 25:
        assert [row["local_label"] for row in fr["rows"]].count("02:00") == 2
        assert len({row["timestamp_utc"] for row in fr["rows"]}) == 25
    assert len(calls) == 2, "Loading existing sources must never refresh them"
    assert fingerprint(tmp_path) == before
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("target_day", [SOURCE_DAY, "2026-09-16"])
def test_history_source_must_be_strictly_later_than_target_before_archive_read(tmp_path, monkeypatch, target_day):
    monkeypatch.setattr(loader, "load_nuclear_result_bundle",
                        lambda **kwargs: pytest.fail("Invalid date ordering must be rejected before loading"))
    before = fingerprint(tmp_path)
    with pytest.raises(ValueError):
        loader.load_model_storm_payload(tmp_path, target_day, history_from_delivery=SOURCE_DAY)
    assert fingerprint(tmp_path) == before


def test_default_mode_never_discovers_future_archive_as_automatic_fallback(tmp_path, monkeypatch):
    archive(tmp_path)
    audited_sources(tmp_path, monkeypatch, SOURCE_DAY, "2026-09-14")
    before = fingerprint(tmp_path)
    payload = loader.load_model_storm_payload(tmp_path, "2026-09-14")
    assert not payload["has_data"]
    assert payload.get("history_from_delivery") is None
    for item in payload["zones"]:
        assert item["coverage"] == {"model": 0, "storm": 0, "observed": 0}
        assert_no_model(item)
    assert fingerprint(tmp_path) == before


def test_target_outside_365_day_backtest_cannot_use_raw_730_day_history(tmp_path):
    target_day = "2025-09-14"  # Present in raw_history, absent from the Kalman replay.
    archive(tmp_path, target_day=target_day)
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, target_day, history_from_delivery=SOURCE_DAY))
    assert_no_model(fr)
    assert fr["sources"]["model"]["status"] in {"invalid", "unavailable"}
    assert fingerprint(tmp_path) == before


def test_corrupt_backtest_is_rejected_without_losing_verified_comparators(tmp_path, monkeypatch):
    bundle = archive(tmp_path)
    audited_sources(tmp_path, monkeypatch, SOURCE_DAY, "2026-09-14")
    (bundle.directory / "kalman_backtest.parquet").write_bytes(b"explicit corrupted archive fixture")
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, "2026-09-14", history_from_delivery=SOURCE_DAY))
    assert_no_model(fr)
    assert fr["sources"]["model"]["status"] == "invalid"
    assert fr["coverage"]["storm"] == fr["coverage"]["observed"] == 24
    assert fingerprint(tmp_path) == before


@pytest.mark.parametrize("fault", ["missing_hour", "duplicate_hour", "nan_p10", "crossed_p90"])
def test_historical_report_boundary_refuses_incomplete_or_invalid_target_quantiles(tmp_path, monkeypatch, fault):
    # Keep a genuine archive on disk, then exercise the report's independent
    # validation in case a future archive reader relaxes its own shape checks.
    bundle = archive(tmp_path)
    frame = bundle.result.kalman_view.backtest
    position = frame.index[frame.delivery_start_utc.eq(bundle.target_index[4])][0]
    if fault == "missing_hour":
        bundle.result.kalman_view.backtest = frame.drop(index=position)
    elif fault == "duplicate_hour":
        bundle.result.kalman_view.backtest = pd.concat([frame, frame.loc[[position]]]).sort_values("delivery_start_utc")
    elif fault == "nan_p10":
        frame.loc[position, "residual_kalman__q10"] = np.nan
    else:
        frame.loc[position, "residual_kalman__q90"] = -1000
    monkeypatch.setattr(loader, "load_nuclear_result_bundle", lambda **kwargs: bundle.result)
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, "2026-09-14", history_from_delivery=SOURCE_DAY))
    assert_no_model(fr)
    assert fr["sources"]["model"]["status"] == "invalid"
    assert fingerprint(tmp_path) == before


def test_comparator_audit_must_identify_source_delivery_not_requested_historical_day(tmp_path, monkeypatch):
    archive(tmp_path)
    directory, audit, _ = audited_sources(tmp_path, monkeypatch, SOURCE_DAY, "2026-09-14")
    audit["delivery_day_local"] = "2026-09-14"
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, "2026-09-14", history_from_delivery=SOURCE_DAY))
    assert fr["coverage"] == {"model": 24, "storm": 0, "observed": 0}
    assert fr["sources"]["storm"]["status"] == fr["sources"]["observed"]["status"] == "invalid"
    assert fingerprint(tmp_path) == before


def test_report_cli_forwards_explicit_history_source_without_invoking_science(tmp_path, monkeypatch):
    import run_model_storm_report as runner
    from chronos2_hourly import model_storm_report
    calls = []
    payload = {"has_data": True, "delivery_day": "2026-09-14", "history_from_delivery": SOURCE_DAY, "zones": []}
    def load(project, day, nuclear_root=None, *, history_from_delivery=None):
        calls.append((project, day, nuclear_root, history_from_delivery))
        return payload
    def render(data, output):
        assert data == payload
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html>explicit historical report fixture</html>", encoding="utf-8")
        return output
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(loader, "load_model_storm_payload", load)
    monkeypatch.setattr(model_storm_report, "render_model_storm_report", render)
    assert runner.main(["--delivery-day", "2026-09-14", "--history-from-delivery", SOURCE_DAY,
                        "--skip-vps-sync"]) == 0
    assert calls == [(tmp_path, "2026-09-14", None, SOURCE_DAY)]
    assert [path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()] == [
        "runs/reports/model_storm/CWE_Model_Storm_2026-09-14.html"]


@pytest.mark.parametrize("source_day", [SOURCE_DAY, '<script>alert("historical-source")</script>'])
def test_historical_source_note_is_visible_and_escaped_in_standalone_report(tmp_path, monkeypatch, source_day):
    from html import escape
    from chronos2_hourly import model_storm_report as report
    from test_model_storm_report import _payload
    monkeypatch.setattr(report, "get_plotlyjs", lambda: "/* EXPLICIT HISTORICAL PLOTLY FIXTURE */")
    payload = _payload("2026-09-14")
    payload["history_from_delivery"] = source_day
    output = tmp_path / "history.html"
    report.render_model_storm_report(payload, output)
    document = output.read_text(encoding="utf-8")
    assert escape(source_day, quote=True) in document
    assert '<script>alert("historical-source")</script>' not in document
    assert "replay" in document.lower() or "histor" in document.lower()
    assert "2026-09-14" in document
    assert "/* EXPLICIT HISTORICAL PLOTLY FIXTURE */" in document
