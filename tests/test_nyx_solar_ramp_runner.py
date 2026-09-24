"""Small mocked lifecycle tests; no providers, real fits, or production writes."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nyx_solar_ramp import runner


ROOT = Path(__file__).resolve().parents[1]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _fixture_panel():
    times = pd.date_range("2026-09-17", periods=4, freq="12h", tz="Europe/Paris").tz_convert("UTC")
    days = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.DataFrame({"zone": "FR", "timestamp_utc": times, "forecast_origin_utc": origin,
        "forecast": 100., "q10": 70., "q90": 130., "actual": 105.,
        "benchmark_forecast": 110., "model": "nuclear_kalman",
        "sample": np.where(days == pd.Timestamp("2026-09-18"), "live", "evaluation")})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = runner.load_config(ROOT / "config/nyx_solar_ramp.yaml")
    baseline = _fixture_panel()
    source = tmp_path / config["source_snapshot"]
    source.mkdir(parents=True)
    historical = baseline.iloc[:2][["zone", "timestamp_utc"]].copy()
    historical["feature_fixture"] = 1.
    historical["label_eligible"] = True
    historical.to_parquet(source / "panel.parquet", index=False)
    _json(source / "config.json", {"fixture": True})
    _json(source / "data_audit.json", {"fixture": True})
    _json(source / "manifest.json", {"input_files": {
        name: runner.digest(source / name) for name in ("config.json", "data_audit.json", "panel.parquet")}})
    _json(tmp_path / "config/nyx_solar_ramp_literature.json", {"references": [{"title": "Fixture only"}]})
    (tmp_path / "Forecast.ps1").write_text("PROTECTED FIXTURE", encoding="utf-8")
    code = tmp_path / "nyx_solar_ramp/policy.py"
    code.parent.mkdir()
    code.write_text("# frozen fixture code", encoding="utf-8")
    monkeypatch.setattr(runner, "exclusive_process_lock", lambda path: nullcontext())
    monkeypatch.setattr(runner, "version", lambda package: "fixture")
    audit = {"evaluation_start_day": "2025-09-18", "evaluation_end_day": "2026-09-17", "evaluation_days": 365}
    monkeypatch.setattr(runner, "load_report_panel", lambda *args, **kwargs: (baseline.copy(deep=True), audit))
    monkeypatch.setattr(runner, "build_features", lambda panel, **kwargs: (
        panel.assign(solarx_eligible=panel.feature_fixture.notna()), {"controls": ["solarx_fixture"]}, {"fixture": True}))
    snapshot = runner.prepare(config, root=tmp_path)
    return snapshot, config, tmp_path


@pytest.fixture
def pipeline_mocks(monkeypatch):
    from nyx_solar_ramp import analysis, evaluation, reporting

    events = []

    def fit(panel, groups, config):
        events.append("fit")
        result = panel.copy(deep=True)
        result["variant"] = "governed"
        result["candidate_forecast"], result["candidate_q10"], result["candidate_q90"] = result.forecast, result.q10, result.q90
        result["risk_probability"], result["alert"] = np.nan, False
        result["gate_reason"] = "fixture_identity"
        result["evaluation_phase"] = np.where(result["sample"].eq("live"), "live_historical", "final_diagnostic")
        return result, [{"fixture": True}], [{"weight": 0.}], {}

    def evaluate(predictions, config):
        events.append("evaluate")
        return {"decision": {"retain_baseline": True, "promotion_allowed": False}}

    def analyse(panel, predictions, config):
        events.append("analyse")
        return {"fixture": True}

    def render(directory, **kwargs):
        events.append("report")
        return directory / "index.html"

    monkeypatch.setattr(runner, "run_policy", fit)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(analysis, "analyse", analyse)
    monkeypatch.setattr(reporting, "render_report", render)
    return events


def test_prepare_seals_literature_and_preserves_missing_suffix(prepared):
    snapshot, config, root = prepared
    _, manifest, saved = runner.read_snapshot(snapshot, root=root)
    assert saved == config
    assert "literature.json" in manifest["input_files"]
    assert "config/nyx_solar_ramp_literature.json" in manifest["code_sha256"]
    panel = pd.read_parquet(snapshot / "panel.parquet")
    assert len(panel) == 4
    assert panel.solarx_eligible.tolist() == [True, True, False, False]
    assert panel.forecast.eq(100).all()
    assert (root / "Forecast.ps1").read_text() == "PROTECTED FIXTURE"
    assert runner.resolve_latest(config, root=root) == snapshot


@pytest.mark.parametrize("path", [".", "runs/exports/new", "runs/live/new", "config/new",
    "runs/experiments/other", "runs/experiments/nyx_solar_ramp_v1/../other"])
def test_path_guard_rejects_other_namespaces(tmp_path, path):
    with pytest.raises(ValueError, match="Solar outputs"):
        runner.safe(tmp_path, path)


def test_path_guard_rejects_symlink_alias(tmp_path):
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    link = tmp_path / runner.NAMESPACE
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(ValueError):
        runner.safe(tmp_path, link / "snapshots/run")


@pytest.mark.parametrize("key,value", [
    ("mode", "production"), ("cutoff_time", "12:00"), ("mae_tolerance", .1),
    ("enabled", 1), ("zones", ["FR"]), ("training_window_days", 366),
    ("minimum_training_days", 89), ("calibration_days", 120), ("threads", 5),
    ("bootstrap_samples", 10001), ("block_days", 0), ("business_spike", float("nan")),
    ("correction_clip", -1.), ("learning_rate", float("inf")), ("seed", -1),
    ("candidate_weights", [0., .5, .5]), ("candidate_weights", []),
    ("candidate_weights", [False, .5]), ("candidate_weights", [0., float("nan")]),
    ("unknown_option", True), ("final_days", 300),
])
def test_config_rejects_unsafe_or_invalid_protocol(tmp_path, key, value):
    config = runner.load_config(ROOT / "config/nyx_solar_ramp.yaml")
    config[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_config(path)


@pytest.mark.parametrize("name", ["config.json", "panel.parquet", "literature.json"])
def test_changed_snapshot_input_is_rejected(prepared, name):
    snapshot, _, root = prepared
    with (snapshot / name).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        runner.read_snapshot(snapshot, root=root)


def test_incomplete_input_checksum_manifest_is_rejected(prepared):
    snapshot, _, root = prepared
    record = json.loads((snapshot / "manifest.json").read_text())
    del record["input_files"]["literature.json"]
    _json(snapshot / "manifest.json", record)
    with pytest.raises(ValueError):
        runner.read_snapshot(snapshot, root=root)


def test_completed_backtest_is_reused_and_report_never_fits(prepared, pipeline_mocks, monkeypatch):
    snapshot, _, root = prepared
    result = runner.backtest(snapshot, root=root)
    assert pipeline_mocks == ["fit", "evaluate", "analyse", "report"]
    prediction_hash = runner.digest(snapshot / "predictions.parquet")
    manifest = json.loads((snapshot / "results_manifest.json").read_text())
    assert {"training_manifest.json", "models.joblib", "metrics.json"}.issubset(manifest["result_files"])
    monkeypatch.setattr(runner, "run_policy", lambda *a, **k: pytest.fail("Never refit a completed snapshot"))
    from nyx_solar_ramp import evaluation
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: pytest.fail("Never reevaluate a sealed report"))
    assert runner.backtest(snapshot, root=root) == result
    assert runner.report(snapshot, root=root) == result
    assert pipeline_mocks == ["fit", "evaluate", "analyse", "report", "report", "report"]
    assert runner.digest(snapshot / "predictions.parquet") == prediction_hash
    assert (root / "Forecast.ps1").read_text() == "PROTECTED FIXTURE"


def test_incomplete_snapshot_refuses_changed_code(prepared, pipeline_mocks):
    snapshot, _, root = prepared
    (root / "nyx_solar_ramp/policy.py").write_text("# changed code", encoding="utf-8")
    with pytest.raises(ValueError, match="Code changed"):
        runner.backtest(snapshot, root=root)
    assert pipeline_mocks == []


def test_report_of_unfinished_snapshot_does_not_start_training(prepared, pipeline_mocks):
    snapshot, _, root = prepared
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.report(snapshot, root=root)
    assert pipeline_mocks == []


def test_report_rejects_modified_result_before_render_or_fit(prepared, pipeline_mocks):
    snapshot, _, root = prepared
    runner.backtest(snapshot, root=root)
    _json(snapshot / "metrics.json", {"tampered": True})
    before = list(pipeline_mocks)
    with pytest.raises(ValueError, match="checksum"):
        runner.report(snapshot, root=root)
    assert pipeline_mocks == before


def test_incomplete_result_checksum_manifest_is_not_completed(prepared, pipeline_mocks):
    snapshot, _, root = prepared
    runner.backtest(snapshot, root=root)
    record = json.loads((snapshot / "results_manifest.json").read_text())
    record["result_files"] = {}
    _json(snapshot / "results_manifest.json", record)
    with pytest.raises(ValueError):
        runner.verify_results(snapshot)


def _fail_evaluation(monkeypatch):
    from nyx_solar_ramp import evaluation

    def fail(*args, **kwargs):
        raise RuntimeError("fixture evaluation failure")

    monkeypatch.setattr(evaluation, "evaluate", fail)


def test_failed_evaluation_resumes_completed_training_without_refit(prepared, pipeline_mocks, monkeypatch):
    snapshot, _, root = prepared
    _fail_evaluation(monkeypatch)
    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        runner.backtest(snapshot, root=root)
    assert pipeline_mocks == ["fit"]
    assert (snapshot / "training_manifest.json").is_file()
    assert not (snapshot / "results_manifest.json").exists()
    assert json.loads((snapshot / "status.json").read_text())["status"] == "failed"
    frozen_prediction = runner.digest(snapshot / "predictions.parquet")
    monkeypatch.setattr(runner, "run_policy", lambda *a, **k: pytest.fail("Checkpoint must avoid refit"))
    from nyx_solar_ramp import evaluation
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"decision": {"retain_baseline": True}})
    runner.backtest(snapshot, root=root)
    assert pipeline_mocks == ["fit", "analyse", "report"]
    assert runner.digest(snapshot / "predictions.parquet") == frozen_prediction
    runner.verify_results(snapshot)


@pytest.mark.parametrize("corruption", ["prediction", "snapshot_hash", "missing_file_entry"])
def test_training_checkpoint_corruption_is_rejected_before_reuse(prepared, pipeline_mocks, monkeypatch, corruption):
    snapshot, _, root = prepared
    _fail_evaluation(monkeypatch)
    with pytest.raises(RuntimeError):
        runner.backtest(snapshot, root=root)
    path = snapshot / "training_manifest.json"
    record = json.loads(path.read_text())
    if corruption == "prediction":
        with (snapshot / "predictions.parquet").open("ab") as stream:
            stream.write(b"corruption")
    elif corruption == "snapshot_hash":
        record["manifest_sha256"] = "another-snapshot"
        _json(path, record)
    else:
        del record["files"]["predictions.parquet"]
        _json(path, record)
    monkeypatch.setattr(runner, "run_policy", lambda *a, **k: pytest.fail("Corrupt checkpoint must not trigger refit"))
    with pytest.raises(ValueError):
        runner.backtest(snapshot, root=root)
    assert pipeline_mocks == ["fit"]
    assert json.loads((snapshot / "status.json").read_text())["status"] == "failed"


def test_source_snapshot_tampering_stops_prepare(prepared, monkeypatch):
    _, config, root = prepared
    source = root / config["source_snapshot"]
    _json(source / "data_audit.json", {"tampered": True})
    monkeypatch.setattr(runner, "load_report_panel", lambda *a, **k: pytest.fail("Do not read baseline after bad source checksum"))
    with pytest.raises(ValueError, match="Frozen source checksum mismatch"):
        runner.prepare(config, root=root)
