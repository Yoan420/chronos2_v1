"""Offline namespace, provenance and resume checks; no production/API activity."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import uuid

import numpy as np
import pandas as pd
import pytest

from nyx_congestion_calibration import runner

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return runner.load_config(ROOT/"config/nyx_congestion_calibration.yaml")


@pytest.mark.parametrize("field,value", [
    ("output_root", "runs/exports"), ("output_root", "runs/experiments/nyx_congestion_v1"),
    ("production_modified", True), ("activation_performed", True), ("diagnostic_only", False),
    ("options", {"threads":True}), ("options", {"threads":3}),
    ("options", {"threads":2, "calibration_days":28}), ("schema_version", True), ("unknown", 1)])
def test_configuration_is_fixed_and_research_only(field, value):
    config = configuration(); config[field] = value
    with pytest.raises(ValueError): runner.validate_config(config)


@pytest.mark.parametrize("path", ["runs/exports/a", "../outside", "runs/experiments/nyx_congestion_v1/new",
    "runs/experiments/nyx_congestion_calibration_v1/../outside"])
def test_namespace_escape_is_rejected(tmp_path, path):
    with pytest.raises(ValueError): runner.safe_path(tmp_path, path)


def test_checkpoint_is_verified_before_deserializing(tmp_path, monkeypatch):
    directory = tmp_path/runner.NAMESPACE/"snapshots/test/residual"
    cache = runner.FitCache(directory, "sealed-snapshot", root=tmp_path)
    state = {"fit_day":"2026-09-14", "model":"synthetic"}
    cache.save("2026-09-14", state)
    assert cache.load("2026-09-14") == state
    with pytest.raises(ValueError): cache.save("2026-09-14", state)
    with pytest.raises(ValueError): cache.load("../escape")
    with pytest.raises(ValueError): runner.FitCache(directory, "other-snapshot", root=tmp_path).load("2026-09-14")
    raw = directory/"fits/2026-09-14.joblib"
    raw.write_bytes(raw.read_bytes()+b"tampered")
    monkeypatch.setattr(runner.joblib, "load", lambda path: pytest.fail("Unverified pickle must never load"))
    with pytest.raises(ValueError): cache.load("2026-09-14")


def test_forecast_and_baseline_support_are_preserved():
    panel = pd.DataFrame({"forecast":[100., 50.], "actual":[120., 55.]})
    prediction = panel.copy()
    for name in runner.MODELS:
        prediction[name] = panel.forecast
        prediction[name+"_q10"] = panel.forecast-10
        prediction[name+"_q90"] = panel.forecast+10
    runner.validate_predictions(panel, prediction)
    for name, value in [("actual",0.), (runner.MODELS[0],np.nan), (runner.MODELS[0]+"_q10",999.)]:
        bad = prediction.copy(); bad.loc[0,name] = value
        with pytest.raises((ValueError, AssertionError)): runner.validate_predictions(panel, bad)
    bad = prediction.drop(columns=[runner.MODELS[0]+"_q90"])
    with pytest.raises(ValueError): runner.validate_predictions(panel, bad)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Keep Windows synthetic report paths below MAX_PATH; still under pytest's
    # isolated temporary root, never a real experiment or production directory.
    tmp_path = tmp_path.parent/("cc_"+uuid.uuid4().hex[:8])
    tmp_path.mkdir()
    config = configuration()
    source = tmp_path/config["source_suite"]
    source.mkdir(parents=True)
    for name in runner.SOURCE_FILES:
        (source/name).write_text("fixture", encoding="utf8")
    panel = pd.DataFrame({"zone":["FR","DE"], "forecast":[100.,200.], "actual":[101.,201.]})
    panel.to_parquet(source/"panel.parquet", index=False)
    pd.DataFrame({"network_eligible":[True,True], "feature_network_x":[1.,2.]}).to_parquet(source/"network_features.parquet", index=False)
    pd.DataFrame({"congestion_ready":[True,True], "feature_congestion_x":[3.,4.]}).to_parquet(source/"signals.parquet", index=False)
    source_manifest = {"settings":{"calibration_days":28}}
    (source/"manifest.json").write_text(json.dumps(source_manifest))
    activation = {"suite_manifest_sha256":runner.digest(source/"manifest.json"),
                  "files":{n:runner.digest(source/n) for n in runner.source_runner.ACTIVATION_FILES}}
    (source/"activation_manifest.json").write_text(json.dumps(activation))
    results = {"status":"completed", "suite_manifest_sha256":runner.digest(source/"manifest.json"),
        "activation_manifest_sha256":runner.digest(source/"activation_manifest.json"),
        "result_files":{n:runner.digest(source/n) for n in runner.source_runner.RESULTS}}
    (source/"results_manifest.json").write_text(json.dumps(results))
    monkeypatch.setattr(runner.source_runner, "read_snapshot", lambda path, **kwargs:(path.resolve(), {}, source_manifest))
    monkeypatch.setattr(runner.protected_runner, "protected_state", lambda root:{"Forecast.ps1":"unchanged"})
    monkeypatch.setattr(runner, "training_code", lambda root:{"new_policy.py":"fixed"})
    monkeypatch.setattr(runner, "runtime_identity", lambda:{"python":"test"})
    original = {n:runner.digest(source/n) for n in runner.SOURCE_FILES}
    directory = runner.prepare(config, root=tmp_path)
    return tmp_path, config, source, directory, original


def test_prepare_copies_inputs_byte_exact_and_never_changes_old_snapshot(prepared):
    root, config, source, directory, original = prepared
    for name in runner.COPIED_INPUTS:
        assert (directory/name).read_bytes() == (source/name).read_bytes()
    _, frozen, manifest = runner.read_snapshot(directory, root=root)
    assert frozen == config and manifest["source_files"] == original
    assert set(manifest["source_files"]) == runner.SOURCE_FILES
    assert manifest["stage1_retrained"] is False
    assert {n:runner.digest(source/n) for n in runner.SOURCE_FILES} == original
    assert runner.resolve_snapshot(config, root=root) == directory


@pytest.mark.parametrize("target", ["input", "source_signal", "source_result", "runtime", "code", "settings", "pointer"])
def test_snapshot_rejects_changed_inputs_sources_and_environment(prepared, monkeypatch, target):
    root, config, source, directory, _ = prepared
    if target == "input": (directory/"signals.parquet").write_bytes(b"tampered")
    elif target == "source_signal": (source/"signals.parquet").write_bytes(b"tampered")
    elif target == "source_result": (source/"predictions.parquet").write_bytes(b"tampered")
    elif target == "runtime": monkeypatch.setattr(runner, "runtime_identity", lambda:{"python":"different"})
    elif target == "code": monkeypatch.setattr(runner, "training_code", lambda root:{"new_policy.py":"different"})
    elif target == "settings":
        manifest = json.loads((directory/"manifest.json").read_text()); manifest["settings"]["calibration_days"] = 90
        (directory/"manifest.json").write_text(json.dumps(manifest))
    else:
        pointer = root/runner.NAMESPACE/"latest_prepared.json"
        content = json.loads(pointer.read_text()); content["manifest_sha256"] = "invalid"
        pointer.write_text(json.dumps(content))
    with pytest.raises(ValueError): runner.resolve_snapshot(config, root=root)


def fake_policy(monkeypatch, callback):
    module = types.ModuleType("nyx_congestion_calibration.policy")
    module.run_replay = callback
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_run_only_calibrates_and_completed_replay_is_reused(prepared, monkeypatch):
    root, _, source, directory, original = prepared
    calls = []
    def run(panel, network, signals, settings, **kwargs):
        calls.append(True)
        pd.testing.assert_frame_equal(signals, pd.read_parquet(source/"signals.parquet"))
        assert settings == {"calibration_days":28}  # The policy owns its fixed 90-day override.
        assert kwargs["load_fit"]("2026-09-14") is None
        kwargs["save_fit"]("2026-09-14", {"fit_day":"2026-09-14", "fake":"calibration"})
        result = panel.copy()
        for name in runner.MODELS: result[name] = panel.forecast
        return {"predictions":result, "folds":pd.DataFrame({"fit_day":["2026-09-14"]}),
                "governance":pd.DataFrame({"weight":[0.]}), "audit":{"calibration_days":90}}
    fake_policy(monkeypatch, run)
    # Any accidental stage-one execution must fail this test.
    monkeypatch.setattr(runner.source_runner, "evaluate", lambda *a,**k:pytest.fail("Old replay must never run"))
    assert runner.evaluate(directory, root=root) == directory
    assert runner.evaluate(directory, root=root) == directory
    assert calls == [True]
    _, _, manifest = runner.read_snapshot(directory, root=root)
    runner.verify_result(directory, manifest)
    assert runner.status(directory, root=root)["results_verified"]
    assert {n:runner.digest(source/n) for n in runner.SOURCE_FILES} == original
    audit = json.loads((directory/"model_audit.json").read_text())
    assert audit["stage1_retrained"] is False and audit["fits_saved_this_run"] == 1


def test_concurrent_source_change_prevents_result_publication(prepared, monkeypatch):
    root, _, source, directory, _ = prepared
    def run(panel, network, signals, settings, **kwargs):
        kwargs["save_fit"]("2026-09-14", {"fit_day":"2026-09-14"})
        (source/"signals.parquet").write_bytes(b"concurrent modification")
        result = panel.copy()
        for name in runner.MODELS: result[name] = panel.forecast
        return {"predictions":result, "folds":pd.DataFrame(), "governance":pd.DataFrame(), "audit":{}}
    fake_policy(monkeypatch, run)
    with pytest.raises(ValueError): runner.evaluate(directory, root=root)
    assert not (directory/"results_manifest.json").exists()
    assert json.loads((directory/"status.json").read_text())["status"] == "failed"


def test_reports_are_versioned_sealed_and_source_remains_unchanged(prepared, monkeypatch):
    root, _, source, directory, original = prepared
    # Synthetic numerical results, sealed exactly as evaluate would publish.
    for name in runner.RESULTS:
        if name == "model_audit.json": (directory/name).write_text('{}')
        elif name == "predictions.parquet": pd.read_parquet(directory/"panel.parquet").to_parquet(directory/name, index=False)
        else: (directory/name).write_text("synthetic")
    manifest = json.loads((directory/"manifest.json").read_text())
    result = dict(status="completed", suite_manifest_sha256=runner.digest(directory/"manifest.json"),
        source_activation_manifest_sha256=manifest["source_files"]["activation_manifest.json"],
        result_files={n:runner.digest(directory/n) for n in runner.RESULTS})
    (directory/"results_manifest.json").write_text(json.dumps(result))
    for name in ["nyx_congestion_calibration/report.py", "nyx_congestion/report.py",
                 "nyx_physical_p50/report.py", "config/economic_value.yaml"]:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("report fixture")
    module = types.ModuleType("nyx_congestion_calibration.report")
    def build(predictions, source_directory, destination, *, root, audit):
        assert source_directory == source
        assert audit["snapshot"] == str(directory)
        destination.mkdir(parents=True)
        (destination/"index.html").write_text("<html>isolated report</html>")
        return {"index":str(destination/"index.html")}
    module.build_report = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    first = runner.report(directory, root=root)
    second = runner.report(directory, root=root)
    assert first != second and first.parent == second.parent == directory/"reports"
    assert runner.status(directory, root=root)["report_verified"]
    assert {n:runner.digest(source/n) for n in runner.SOURCE_FILES} == original
    (second/"index.html").write_text("tampered")
    with pytest.raises(ValueError): runner.status(directory, root=root)


def test_dryrun_has_no_writes_or_source_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_source", lambda *a,**k:pytest.fail("DryRun must not inspect/evaluate source"))
    result = runner.dry_run(configuration(), root=tmp_path)
    assert result["writes"] is False and result["api_calls"] is False and not list(tmp_path.iterdir())


def test_cli_dryrun_has_no_writes(monkeypatch, tmp_path):
    import run_nyx_congestion_calibration as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--action","dryrun","--config",str(ROOT/"config/nyx_congestion_calibration.yaml")]) == 0
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit): cli.main(["--action","collect"])


def test_powershell_dryrun_displays_argv_without_execution():
    executable = shutil_which("pwsh") or shutil_which("powershell")
    if executable is None: pytest.skip("PowerShell not installed")
    result = subprocess.run([executable,"-NoProfile","-File",str(ROOT/"NyxCongestionCalibration.ps1"),
        "-Action","Run","-DryRun","-PythonExecutable",sys.executable],capture_output=True,text=True,check=False)
    assert result.returncode == 0, result.stderr
    assert "run_nyx_congestion_calibration.py" in result.stdout and "aucun fichier" in result.stdout


def shutil_which(name):
    import shutil
    return shutil.which(name)
