"""Frozen fundamental experiment lifecycle; every model fit is mocked."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_fundamental_stress import runner
from nyx_scarcity import runner as base
from nyx_scarcity.policy import PolicyResult, _parameters
from nyx_scarcity_zonal import runner as zonal
from test_nyx_scarcity_runner import sample, fake_policy
from test_nyx_scarcity_zonal_policy import saved_result


ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return runner.load_config(ROOT/"config/nyx_fundamental_stress.yaml")


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 2}, {"unknown": 1},
    {"variants": ["fundamental"]}, {"primary_variant": "calendar"}, {"fixed_alpha": .5},
    {"max_parallel": True}, {"max_parallel": 3}, {"diagnostic_only": False},
    {"production_modified": True}, {"activation_performed": True}, {"source_suite": ""}, {"output_root": None},
])
def test_recipe_rejects_changed_experiment_or_production_flags(change):
    config = recipe()
    config.update(change)
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize("path", ["runs/exports/new", "runs/live/new", "data/pit/new", ".",
    "runs/experiments/nyx_scarcity_v1/zonal/new", "runs/experiments/nyx_scarcity_v1/variants/new",
    "runs/experiments/nyx_scarcity_v1/fundamental/../snapshots/new"])
def test_output_cannot_escape_fundamental_namespace(tmp_path, path):
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    baseline, panel, data_audit = sample()
    panel = panel.iloc[:48].copy()
    settings = _parameters(base.model_settings(baseline, data_audit))
    prediction = fake_policy(panel, settings).predictions
    prediction["bounded_correction"] = 0.
    prediction["probability_gate"] = .6
    prediction["proposal_reason"] = "physical_context_below_threshold"
    source_config = zonal.load_config(ROOT/"config/nyx_scarcity_zonal.yaml")
    directory = tmp_path/zonal.NAMESPACE/"snapshots/source"
    directory.mkdir(parents=True)
    audit = {"source_comparison": {"source_settings": settings}, "source_data_audit": data_audit,
             "source_config": baseline, "diagnostic_only": True}
    for name, value in (("config.json", source_config), ("base_config.json", baseline),
                        ("source_audit.json", audit), ("regional_audit.json", {"config": settings})):
        base._json(directory/name, value)
    for name, frame in (("panel.parquet", panel), ("hgb_predictions.parquet", prediction),
                        ("regional_predictions.parquet", prediction), ("regional_25_predictions.parquet", prediction)):
        base._parquet(directory/name, frame)
    manifest = {"config": source_config, "settings": settings,
                "input_files": {name: base.digest(directory/name) for name in zonal.INPUTS}}
    base._json(directory/"manifest.json", manifest)
    for name in source_config["variants"]:
        at = directory/name
        at.mkdir()
        for artifact in ("predictions.parquet", "strict_predictions.parquet"):
            base._parquet(at/artifact, prediction)
        for artifact in ("folds.parquet", "governance.parquet"):
            base._parquet(at/artifact, pd.DataFrame({"status": ["archived_mock"]}))
        base._json(at/"model_audit.json", {"diagnostic_only": True})
        (at/"latest_model.joblib").write_bytes(b"archived mock; never deserialized")
        base._json(at/"results_manifest.json", {"status": "completed", "variant": name,
            "suite_manifest_sha256": base.digest(directory/"manifest.json"),
            "result_files": {key: base.digest(at/key) for key in zonal.RESULTS}})
    base._json(directory/"comparison.json", {"model_labels": {}})
    base._json(directory/"comparison_manifest.json", {"status": "completed",
        "suite_manifest_sha256": base.digest(directory/"manifest.json"),
        "comparison_sha256": base.digest(directory/"comparison.json"),
        "variant_manifests": {name: base.digest(directory/name/"results_manifest.json") for name in source_config["variants"]}})
    monkeypatch.setattr(runner, "protected_state", lambda *args: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner, "code_seals", lambda *args: {"fundamental.py": "frozen"})
    monkeypatch.setattr(runner, "runtime_seals", lambda *args: {"xgboost": "private_mock"})
    config = recipe()
    config["source_suite"] = str(directory)
    return directory, config, panel, tmp_path


@pytest.fixture
def prepared(source):
    directory, config, panel, root = source
    return runner.prepare(config, root=root), config, panel, root, directory


def _mock_fit(monkeypatch):
    def fit(panel, settings, name, **kwargs):
        kwargs["on_last_fit"]({"mock_model": name}, settings)
        result = fake_policy(panel, settings)
        result.predictions["bounded_correction"] = 0.
        result.predictions["probability_gate"] = .6
        result.predictions["proposal_reason"] = "physical_context_below_threshold"
        return PolicyResult(result.predictions, pd.DataFrame({"audit": [{}]}), result.governance,
                            {"config": settings, "variant": name, "diagnostic_only": True, "changed_forecast_rows": 0})
    monkeypatch.setattr(runner, "run_fundamental_policy", fit)


def test_prepared_snapshot_preserves_all_inputs_and_source(prepared):
    directory, config, panel, root, source = prepared
    before = base.digest(source/"panel.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(directory/"panel.parquet"), panel)
    assert runner.read_suite(directory, root=root)[1] == config
    assert runner.resolve_latest(config, root=root, completed=False) == directory
    assert before == base.digest(source/"panel.parquet")


@pytest.mark.parametrize("name", sorted(runner.INPUTS))
def test_every_frozen_input_is_checked(prepared, name):
    directory, _, _, root, _ = prepared
    (directory/name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_suite(directory, root=root)


@pytest.mark.parametrize("field", ["settings", "config"])
def test_manifest_recipe_cannot_diverge_from_sealed_inputs(prepared, field):
    directory, _, _, root, _ = prepared
    manifest = json.loads((directory/"manifest.json").read_text())
    if field == "settings":
        manifest[field]["probability_gate"] = .51
    else:
        manifest[field]["max_parallel"] = 1
    base._json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.read_suite(directory, root=root)


@pytest.mark.parametrize("kind", ["comparison", "variant_manifest", "variant_artifact"])
def test_prepare_rejects_corrupt_completed_source_suite(source, kind):
    directory, config, _, root = source
    target = {"comparison": directory/"comparison.json",
              "variant_manifest": directory/"zonal_hiercal/results_manifest.json",
              "variant_artifact": directory/"zonal_hiercal/model_audit.json"}[kind]
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.prepare(config, root=root)


def test_fixed25_audit_counts_its_own_interventions_and_has_own_interval_metadata():
    strict = saved_result(days=5)
    strict.predictions["proposal_reason"] = np.where(strict.predictions.bounded_correction.gt(0),
        "physical_gate_and_positive_signed_residual", "physical_context_below_threshold")
    strict = replace(strict, audit={**strict.audit, "variant": "fundamental", "changed_forecast_rows": 0})
    output = runner.fixed_diagnostic(strict)
    active = output.predictions.applied_correction.gt(0)
    assert active.any()
    assert output.audit["changed_forecast_rows"] == int(active.sum())
    assert output.audit.get("interval_protocol") or output.audit.get("intervals")
    assert output.audit["decision_policy"] == "fixed25_diagnostic"
    assert output.audit["strict_governor_enforced"] is False
    assert output.audit["annual_non_regression_guaranteed"] is False
    assert output.predictions.loc[~active, "gate_reason"].equals(strict.predictions.loc[~active, "proposal_reason"])
    np.testing.assert_allclose(output.predictions.candidate_forecast, strict.predictions.forecast+.25*strict.predictions.bounded_correction)
    assert strict.predictions.candidate_forecast.eq(100.).all()


def test_completed_worker_reuses_governed_and_fixed25_without_fitting(prepared, monkeypatch):
    directory, _, panel, root, _ = prepared
    _mock_fit(monkeypatch)
    result = runner.run_worker(directory, "fundamental", root=root)
    assert result["status"] == "completed"
    for artifact in ("predictions.parquet", "proposals_25.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(directory/"fundamental"/artifact)[panel.columns], panel)
    checksums = {name: base.digest(directory/"fundamental"/name) for name in runner.RESULTS}
    model = runner.load_model(directory, "fundamental", root=root)
    assert model["state"]["mock_model"] == "fundamental"
    assert model["suite_manifest_sha256"] == base.digest(directory/"manifest.json")
    monkeypatch.setattr(runner, "run_fundamental_policy", lambda *a, **k: pytest.fail("Refit completed model"))
    assert runner.run_worker(directory, "fundamental", root=root)["status"] == "reused"
    assert checksums == {name: base.digest(directory/"fundamental"/name) for name in runner.RESULTS}


@pytest.mark.parametrize("name", sorted(runner.RESULTS))
def test_every_completed_artifact_checked_before_reuse(prepared, monkeypatch, name):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    runner.run_worker(directory, "calendar", root=root)
    (directory/"calendar"/name).write_bytes(b"corrupt")
    monkeypatch.setattr(runner, "run_fundamental_policy", lambda *a, **k: pytest.fail("Corrupt result silently refitted"))
    with pytest.raises(ValueError, match="checksum"):
        runner.run_worker(directory, "calendar", root=root)


def test_code_drift_blocks_unfinished_fit_and_model_deserialization(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    runner.run_worker(directory, "fundamental", root=root)
    monkeypatch.setattr(runner, "code_seals", lambda *a: {"changed": "code"})
    monkeypatch.setattr(runner.joblib, "load", lambda *a: pytest.fail("Unverified model loaded"))
    with pytest.raises(ValueError, match="Code/runtime changed"):
        runner.load_model(directory, "fundamental", root=root)
    with pytest.raises(ValueError, match="Code/runtime changed"):
        runner.run_worker(directory, "calendar", root=root)


def test_concurrent_manifest_change_prevents_publication(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    original = runner.run_fundamental_policy
    def fit(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest["created_at_utc"] = "mutated during fit"
        base._json(directory/"manifest.json", manifest)
        return result
    monkeypatch.setattr(runner, "run_fundamental_policy", fit)
    with pytest.raises(ValueError, match="Concurrent"):
        runner.run_worker(directory, "fundamental", root=root)
    assert not (directory/"fundamental/results_manifest.json").exists()


def test_registered_names_cannot_be_replaced_by_path_traversal(prepared):
    directory, _, _, root, _ = prepared
    with pytest.raises(ValueError, match="Unregistered"):
        runner.run_worker(directory, "../../exports/escape", root=root)
    assert not (root/"runs/exports").exists()


def test_reports_reuse_completed_predictions_and_frozen_fixed25_diagnostics(prepared, monkeypatch):
    from nyx_scarcity import variant_reporting
    from nyx_fundamental_stress import reporting
    directory, config, _, root, _ = prepared
    _mock_fit(monkeypatch)
    for name in config["variants"]:
        runner.run_worker(directory, name, root=root)
    monkeypatch.setattr(runner, "run_fundamental_policy", lambda *a, **k: pytest.fail("Completed model refitted"))
    monkeypatch.setattr(runner, "fixed_diagnostic", lambda *a, **k: pytest.fail("Frozen fixed25 recalculated"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Worker respawned"))
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: {"model_labels": {}})
    def render(predictions, metrics, audit, path):
        path.write_text("<html><main>frozen diagnostic</main></html>", encoding="utf-8")
    def reports(predictions, **kwargs):
        at = kwargs["output_directory"]
        at.mkdir(parents=True, exist_ok=True)
        index = at/"index.html"
        index.write_text("<html>"+kwargs["decision_policy"]+"</html>", encoding="utf-8")
        return {"index": index}
    monkeypatch.setattr(variant_reporting, "render_comparison", render)
    monkeypatch.setattr(reporting, "render_fundamental_reports", reports)
    path = runner.run_suite(directory, root=root)
    assert path.is_file() and (directory/"reports_fixed25/index.html").is_file()
    assert runner.resolve_latest(config, root=root, completed=True) == directory
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: pytest.fail("Metrics recalculated"))
    assert runner.run_suite(directory, root=root) == path
    (directory/"comparison.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.report(directory, root=root)


def test_cli_conflicting_arguments_rejected_without_work():
    from run_nyx_fundamental_stress import main
    assert main(["--action", "run", "--run-directory", "existing"]) == 2
    assert main(["--action", "prepare", "--run-directory", "existing"]) == 2
    assert main(["--action", "report", "--variant", "fundamental"]) == 2


def test_powershell_dryrun_from_unrelated_directory_creates_nothing(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([shell, "-NoProfile", "-File", str(ROOT/"FundamentalStress.ps1"), "-Action", "Run", "-DryRun"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "run_nyx_fundamental_stress.py" in result.stdout and "shell=False" in result.stdout
    assert not list(tmp_path.iterdir())


def test_reporting_failure_is_recorded_without_losing_predictions_or_touching_production(prepared, monkeypatch):
    directory, config, _, root, _ = prepared
    _mock_fit(monkeypatch)
    for name in config["variants"]:
        runner.run_worker(directory, name, root=root)
    for relative in ("Forecast.ps1", "runs/exports/operational.html", "data/pit/source.parquet"):
        sentinel = root/relative
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"unchanged operational sentinel")
    before = {p.relative_to(root).as_posix(): base.digest(p) for p in root.rglob("*") if p.is_file()}
    error = RuntimeError("synthetic HTML rendering failure")
    calls = []
    def failed_report(value, *, root):
        # This executes only after the public wrapper has validated the sealed suite.
        calls.append(value)
        assert runner.read_suite(value, root=root)[1] == config
        base._json(value/"status.json", {"status": "running", "stage": "reporting"})
        raise error
    monkeypatch.setattr(runner, "_report", failed_report)
    with pytest.raises(RuntimeError, match="synthetic HTML rendering failure") as caught:
        runner.report(directory, root=root)
    assert caught.value is error and calls == [directory]
    status = json.loads((directory/"status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "reporting"
    assert status["predictions_retained"] is True and status["production_modified"] is False
    assert status["snapshot"] == str(directory) and status["error"] == str(error)
    after = {p.relative_to(root).as_posix(): base.digest(p) for p in root.rglob("*") if p.is_file()}
    status_key = (directory/"status.json").relative_to(root).as_posix()
    assert {k: v for k, v in before.items() if k != status_key} == {
        k: v for k, v in after.items() if k != status_key}
    for name in config["variants"]:
        runner.verify_result(directory, name, runner.read_suite(directory, root=root)[2])
