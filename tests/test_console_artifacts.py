"""Small synthetic fixtures for read-only historical adapters; no model runs."""
import gzip
import json
from pathlib import Path

from experiment_console.artifacts import compare_scopes, discover_runs, inspect_run, read_forecast


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_run(path: Path):
    write_json(path / "run_manifest.json", {
        "model_id": "fixture_only", "zone": "FR", "target_column": "spot_price",
        "target_contract": "hourly_utc_no_interpolation", "delivery_horizon": "dynamic_23_24_25",
        "config": "missing_original.yaml",
    })
    write_json(path / "metrics_hourly.json", {"metrics": [{"model": "fixture", "mae": 2.5, "n_scored": 24, "n_expected": 24, "prediction_coverage": 1, "score_coverage": 1}], "training_diagnostics": {"metric_scope": "fixture_window"}})
    write_json(path / "evaluation_summary.json", {"start_utc": "2026-09-01T00:00:00Z", "end_utc": "2026-09-01T23:00:00Z", "candidate_mae": 2.5})
    return path


def test_discovery_stable_identity_and_no_historical_mutation(tmp_path):
    folder = fixture_run(tmp_path / "historical")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    first, second = discover_runs(tmp_path), discover_runs(tmp_path)
    assert len(first["candidates"]) == 1
    assert first["candidates"][0]["source_key"] == second["candidates"][0]["source_key"]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    candidate = first["candidates"][0]
    assert candidate["metrics"]["fixture.mae"] == 2.5
    assert candidate["status"] == "unknown"
    assert candidate["started_at"] is None
    assert candidate["config"] == {}
    assert candidate["scope"]["frequency"] == "1h"


def test_incomplete_and_unknown_do_not_abort_import(tmp_path):
    fixture_run(tmp_path / "good")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "run_manifest.json").write_text('{"model_id":', encoding="utf-8")
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "odd.txt").write_text("unrecognized", encoding="utf-8")
    result = discover_runs(tmp_path)
    assert len(result["candidates"]) == 2
    assert any("incomplet" in message for run in result["candidates"] for message in run["warnings"])
    assert any("non reconnu" in message for message in result["warnings"])
    assert inspect_run(tmp_path / "absent")["status"] == "unknown"


def test_status_alias_does_not_duplicate_and_external_is_not_assumed_alive(tmp_path):
    status = {"status": "running", "run_id": "original", "started_at_utc": "2026-09-01T10:00:00Z", "steps": [{"name": "sources", "status": "complete"}, {"name": "nuclear_kalman", "zone": "DE", "status": "running"}]}
    write_json(tmp_path / "day" / "original" / "status.json", status)
    write_json(tmp_path / "day" / "latest_status.json", dict(status, status_file=str(tmp_path / "day" / "original" / "status.json")))
    result = discover_runs(tmp_path)
    assert len(result["candidates"]) == 1
    run = result["candidates"][0]
    assert run["source"] == "external"
    assert run["status"] == "unknown"
    assert run["reported_status"] == "running"
    assert run["activity"] == "nuclear_kalman DE"


def test_explicit_completion_and_duration(tmp_path):
    write_json(tmp_path / "status.json", {"status": "complete", "started_at_utc": "2026-09-01T10:00:00Z", "finished_at_utc": "2026-09-01T10:00:04Z", "returncode": 0})
    run = inspect_run(tmp_path)
    assert run["status"] == "succeeded"
    assert run["duration_seconds"] == 4
    assert run["returncode"] == 0
    assert run["return_code"] == 0


def test_scope_difference_and_missing_metadata_warn(tmp_path):
    left = inspect_run(fixture_run(tmp_path / "a"))
    right = inspect_run(fixture_run(tmp_path / "b"))
    assert compare_scopes([left, right])["comparable"]
    right["scope"]["zone"] = "BE"
    result = compare_scopes([left, right])
    assert not result["comparable"]
    assert result["dimensions"]["zone"]["status"] == "different"
    right["scope"]["horizon"] = None
    assert compare_scopes([left, right])["dimensions"]["horizon"]["status"] == "unknown"
    assert not compare_scopes([left])["comparable"]


def test_equal_partial_coverage_is_not_proof_of_same_scored_hours(tmp_path):
    left = inspect_run(fixture_run(tmp_path / "a"))
    left["scope"]["coverage"][0].update(n_scored=12, score_coverage=0.5)
    result = compare_scopes([left, left])
    assert not result["comparable"]
    assert any("heures évaluées" in warning for warning in result["warnings"])


def test_forecast_export_price_alias_is_not_observation(tmp_path):
    path = tmp_path / "forecast_hourly_fr.csv"
    path.write_text("delivery_start_utc,q50,price_eur_mwh\n2026-09-01T00:00:00Z,50,50\n", encoding="utf-8")
    result = read_forecast(path)
    assert result["points"] == [{"timestamp": "2026-09-01T00:00:00Z", "model": "q50", "predicted": 50, "observed": None, "error": None}]
    assert result["warnings"]


def test_backtest_gzip_pairs_errors_and_missing_values(tmp_path):
    path = tmp_path / "backtest_hourly_oof.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("delivery_start_utc,actual,chronos2__q50\n2026-09-01T00:00:00Z,10,12\n2026-09-01T01:00:00Z,,NaN\n2026-09-01T02:00:00Z,13,15\n")
    result = read_forecast(path, limit=1)
    assert len(result["points"]) == 1
    assert result["points"][0]["error"] == 2
    assert result["truncated"]
    assert read_forecast(tmp_path / "missing.csv")["warnings"]


def test_metrics_reject_nonfinite_and_arbitrary_numeric_metadata(tmp_path):
    write_json(tmp_path / "metrics_hourly.json", {"metrics": [{"model": "m", "mae": float("nan"), "rmse": 3, "n_scored": 0}], "seed": 123})
    metrics = inspect_run(tmp_path)["metrics"]
    assert metrics == {"m.rmse": 3, "m.n_scored": 0}


def test_metadata_is_strict_json_safe_and_stale_alias_warns(tmp_path):
    write_json(tmp_path / "effective_config.json", {"value": float("inf")})
    (tmp_path / "run_manifest.json").write_text('{"effective_config": {"large": 1e9999}}', encoding="utf-8")
    run = inspect_run(tmp_path)
    json.dumps(run, allow_nan=False)
    write_json(tmp_path / "old" / "latest_status.json", {"status_file": "missing_status.json", "status": "running"})
    assert any("canonique" in warning for warning in discover_runs(tmp_path)["warnings"])


def test_backtest_preview_respects_explicit_utc_evaluation_period(tmp_path):
    path = tmp_path / "backtest_hourly_oof.csv"
    path.write_text("delivery_start_utc,actual,m__q50\n2025-09-01T00:00:00Z,10,12\n2026-09-01T00:00:00Z,20,23\n", encoding="utf-8")
    scope = {"period": {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T23:00:00Z", "basis": "utc"}}
    result = read_forecast(path, scope=scope)
    assert result["evaluation_period_applied"]
    assert len(result["points"]) == 1
    assert result["points"][0]["observed"] == 20


def test_nuclear_progress_is_discovered_and_phase_refreshed(tmp_path):
    folder = tmp_path / "nuclear"
    status = {"zone": "DE", "stage": "run", "phase": "forecast", "status": "running", "started_at_utc": "2026-09-01T10:00:00Z", "updated_at_utc": "2026-09-01T10:01:00Z"}
    write_json(folder / "run_status.json", status)
    result = discover_runs(tmp_path)
    assert len(result["candidates"]) == 1
    run = result["candidates"][0]
    assert run["source"] == "external"
    assert run["activity"] == "DE · run · forecast"
    assert run["status"] == "unknown"
    status.update(phase="publish_exports", status="complete", updated_at_utc="2026-09-01T10:03:00Z")
    write_json(folder / "run_status.json", status)
    current = inspect_run(folder, include_artifacts=False)
    assert current["activity"] == "DE · run · publish_exports"
    assert current["reported_status"] == "complete"
    assert current["status"] == "unknown"
    assert current["finished_at"] is None
    assert current["status_updated_at"] == "2026-09-01T10:03:00Z"


def test_nuclear_raw_inputs_are_not_artifacts_and_light_inspection_skips_listing(tmp_path, monkeypatch):
    from experiment_console import artifacts
    write_json(tmp_path / "run_status.json", {"phase": "forecast", "status": "running"})
    for directory in ("prepared", "snapshot", "sources", "selected_pit"):
        write_json(tmp_path / directory / "sensitive_input.json", {"data": 1})
    report = tmp_path / "reports" / "report.html"
    report.parent.mkdir()
    report.write_text("<p>Real fixture report</p>", encoding="utf-8")
    visible = inspect_run(tmp_path)["artifacts"]
    assert {item["path"] for item in visible} == {"run_status.json", "reports/report.html"}
    monkeypatch.setattr(artifacts, "list_artifacts", lambda _path: (_ for _ in ()).throw(AssertionError("No traversal during lightweight refresh")))
    assert inspect_run(tmp_path, include_artifacts=False)["artifacts"] == []
