"""Isolated launcher/snapshot lifecycle, with model fitting fully mocked."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from nyx_scarcity import runner as base
from nyx_scarcity.policy import PolicyResult, _parameters
from nyx_scarcity_zonal import runner as zonal
from test_nyx_scarcity_runner import sample, fake_policy


ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return zonal.load_config(ROOT/"config/nyx_scarcity_zonal.yaml")


@pytest.mark.parametrize("change", [
    {"unexpected": 1}, {"schema_version": 2}, {"variants": ["zonal_hiercal"]},
    {"primary_variant": "regional_hiercal"}, {"fixed_alpha": .5},
    {"offset_penalty": True}, {"offset_penalty": 2.}, {"offset_bound": 4.},
    {"max_parallel": 3}, {"max_parallel": True}, {"max_parallel": 1.0},
    {"diagnostic_only": False}, {"production_modified": True}, {"activation_performed": True},
    {"source_suite": ""}, {"output_root": None},
])
def test_invalid_or_unregistered_recipes_refused(change):
    config = recipe()
    config.update(change)
    with pytest.raises(ValueError):
        zonal.validate_config(config)


@pytest.mark.parametrize("path", ["runs/exports/zonal", "runs/live/zonal", "data/pit/zonal", ".",
    "runs/experiments/nyx_scarcity_v1/variants/new", "runs/experiments/nyx_scarcity_v1/snapshots/new",
    "runs/experiments/nyx_scarcity_v1/zonal/../snapshots/new"])
def test_outputs_confined_to_new_zonal_namespace(tmp_path, path):
    with pytest.raises(ValueError):
        zonal.safe_path(tmp_path, path)


def test_output_junction_or_symlink_cannot_redirect_to_production(tmp_path):
    outside = tmp_path/"runs/exports"
    outside.mkdir(parents=True)
    link = tmp_path/zonal.NAMESPACE
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted on this host")
    with pytest.raises(ValueError):
        zonal.safe_path(tmp_path, link/"snapshots/new")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import run_nyx_scarcity_adjustments as adjustment
    baseline, panel, data_audit = sample()
    panel = panel.iloc[:48].copy()
    settings = _parameters(base.model_settings(baseline, data_audit))
    frozen = fake_policy(panel, settings)
    frozen.predictions["bounded_correction"] = 0.
    frozen.predictions["probability_gate"] = .6
    regional_audit = {"config": settings, "diagnostic_only": True}
    source = tmp_path/zonal.old.NAMESPACE/"snapshots/source"
    source.mkdir(parents=True)
    (source/"xgb_unweighted_fixed").mkdir()
    base._parquet(source/"panel.parquet", panel)
    for name, value in (("base_config.json", baseline), ("data_audit.json", data_audit),
                        ("xgb_unweighted_fixed/model_audit.json", regional_audit)):
        base._json(source/name, value)
    source_audit = {"source_settings": settings, "source_hashes": {"fake_source": "immutable"}}
    mapping = {"hgb_v1": frozen.predictions.copy(), "xgb_unweighted_fixed": frozen.predictions.copy()}
    monkeypatch.setattr(adjustment, "read_source", lambda *a, **k: (deepcopy(mapping), deepcopy(source_audit)))
    monkeypatch.setattr(zonal.old, "read_suite", lambda *a, **k: (source, {}, {"settings": settings}))
    monkeypatch.setattr(zonal, "protected_state", lambda *a: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(zonal, "code_seals", lambda *a: {"zonal.py": "frozen"})
    monkeypatch.setattr(zonal, "runtime_seals", lambda *a: {"xgboost": "private_mock"})
    config = recipe()
    config["source_suite"] = str(source)
    directory = zonal.prepare(config, root=tmp_path)
    return directory, config, panel, tmp_path, source


def _mock_fit(monkeypatch):
    def run(panel, settings, name, **kwargs):
        kwargs["on_last_fit"]({"mock_model": name}, settings)
        result = fake_policy(panel, settings)
        result.predictions["bounded_correction"] = 0.
        result.predictions["probability_gate"] = .6
        return PolicyResult(result.predictions, pd.DataFrame({"thresholds": [{}]}), result.governance,
                            {"config": settings, "diagnostic_only": True})
    monkeypatch.setattr(zonal, "run_zonal_policy", run)
    monkeypatch.setattr(zonal, "fixed_conservative_forecast", lambda value, **kwargs: value)


def test_prepare_copies_source_without_mutation_and_binds_recipe(prepared):
    directory, config, panel, root, source = prepared
    original = (source/"panel.parquet").read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(directory/"panel.parquet"), panel)
    read, stored, manifest = zonal.read_suite(directory, root=root)
    assert read == directory and stored == config
    assert manifest["settings"]["feature_columns"] == ["feature_x"]
    assert (source/"panel.parquet").read_bytes() == original
    assert zonal.resolve_latest(config, root=root, completed=False) == directory


@pytest.mark.parametrize("name", sorted(zonal.INPUTS))
def test_all_frozen_inputs_are_hash_verified(prepared, name):
    directory, _, _, root, _ = prepared
    (directory/name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        zonal.read_suite(directory, root=root)


def test_manifest_settings_must_match_sealed_source_settings(prepared):
    directory, _, _, root, _ = prepared
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["settings"]["probability_gate"] = .51
    base._json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError, match="settings"):
        zonal.read_suite(directory, root=root)


def test_manifest_config_must_match_hashed_config(prepared):
    directory, _, _, root, _ = prepared
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["config"]["max_parallel"] = 1
    base._json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        zonal.read_suite(directory, root=root)


def test_unfinished_code_drift_refused_before_any_model_fit(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    monkeypatch.setattr(zonal, "code_seals", lambda *a: {"changed": "code"})
    monkeypatch.setattr(zonal, "run_zonal_policy", lambda *a, **k: pytest.fail("Unsealed fit"))
    with pytest.raises(ValueError, match="Code/runtime changed"):
        zonal.run_worker(directory, "zonal_hiercal", root=root)


def test_completed_worker_reuses_hash_verified_predictions_and_model(prepared, monkeypatch):
    directory, _, panel, root, _ = prepared
    _mock_fit(monkeypatch)
    result = zonal.run_worker(directory, "zonal_hiercal", root=root)
    assert result["status"] == "completed"
    destination = directory/"zonal_hiercal"
    before = {name: base.digest(destination/name) for name in zonal.RESULTS}
    pd.testing.assert_frame_equal(pd.read_parquet(destination/"predictions.parquet")[panel.columns], panel)
    model = zonal.load_model(directory, "zonal_hiercal", root=root)
    assert model["state"]["mock_model"] == "zonal_hiercal"
    assert model["suite_manifest_sha256"] == base.digest(directory/"manifest.json")
    monkeypatch.setattr(zonal, "run_zonal_policy", lambda *a, **k: pytest.fail("Completed model retrained"))
    assert zonal.run_worker(directory, "zonal_hiercal", root=root)["status"] == "reused"
    assert {name: base.digest(destination/name) for name in zonal.RESULTS} == before


@pytest.mark.parametrize("name", sorted(zonal.RESULTS))
def test_completed_artifact_corruption_fails_closed_without_refit(prepared, monkeypatch, name):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    zonal.run_worker(directory, "zonal_context", root=root)
    (directory/"zonal_context"/name).write_bytes(b"corrupt")
    monkeypatch.setattr(zonal, "run_zonal_policy", lambda *a, **k: pytest.fail("Corrupt completed snapshot silently refitted"))
    with pytest.raises(ValueError, match="checksum"):
        zonal.run_worker(directory, "zonal_context", root=root)


def test_trained_result_cannot_be_transplanted_to_another_suite_manifest(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    zonal.run_worker(directory, "zonal_context", root=root)
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["created_at_utc"] = "different experiment identity"
    base._json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError, match="not bound"):
        zonal.run_worker(directory, "zonal_context", root=root)


def test_model_deserialization_refuses_hash_or_runtime_drift(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    zonal.run_worker(directory, "regional_hiercal", root=root)
    monkeypatch.setattr(zonal, "code_seals", lambda *a: {"changed": "code"})
    monkeypatch.setattr(zonal.joblib, "load", lambda *a: pytest.fail("Unverified pickle loaded"))
    with pytest.raises(ValueError, match="Code/runtime changed"):
        zonal.load_model(directory, "regional_hiercal", root=root)


def test_concurrent_manifest_change_prevents_result_publication(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    _mock_fit(monkeypatch)
    fit = zonal.run_zonal_policy
    def changed(*args, **kwargs):
        result = fit(*args, **kwargs)
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest["created_at_utc"] = "changed during fit"
        base._json(directory/"manifest.json", manifest)
        return result
    monkeypatch.setattr(zonal, "run_zonal_policy", changed)
    with pytest.raises(ValueError, match="Concurrent"):
        zonal.run_worker(directory, "zonal_hiercal", root=root)
    assert not (directory/"zonal_hiercal/results_manifest.json").exists()
    assert json.loads((directory/"zonal_hiercal/status.json").read_text())["status"] == "failed"


def test_unregistered_worker_never_creates_outside_path(prepared):
    directory, _, _, root, _ = prepared
    with pytest.raises(ValueError, match="Unregistered"):
        zonal.run_worker(directory, "../../../../exports/escape", root=root)
    assert not (root/"runs/exports").exists()


def test_completed_comparison_reuses_frozen_regional_overlay_without_recalculation(prepared, monkeypatch):
    from nyx_scarcity import variant_reporting
    from nyx_scarcity_zonal import reporting
    directory, config, _, root, _ = prepared
    _mock_fit(monkeypatch)
    for name in config["variants"]:
        zonal.run_worker(directory, name, root=root)
    monkeypatch.setattr(zonal, "fixed_conservative_forecast", lambda *a, **k: pytest.fail("Frozen regional comparator recalculated"))
    monkeypatch.setattr(zonal, "run_zonal_policy", lambda *a, **k: pytest.fail("Completed suite retrained"))
    monkeypatch.setattr(zonal.subprocess, "Popen", lambda *a, **k: pytest.fail("Completed worker respawned"))
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: {"model_labels": {}})
    def comparison(predictions, metrics, audit, path):
        path.write_text("<html>frozen comparison</html>", encoding="utf-8")
    def production_like(predictions, **kwargs):
        destination = kwargs["output_directory"]
        destination.mkdir(parents=True, exist_ok=True)
        index = destination/"index.html"
        index.write_text("<html>frozen zonal reports</html>", encoding="utf-8")
        return {"index": index}
    monkeypatch.setattr(variant_reporting, "render_comparison", comparison)
    monkeypatch.setattr(reporting, "render_zonal_reports", production_like)
    report = zonal.run_suite(directory, root=root)
    assert report.is_file()
    assert zonal.resolve_latest(config, root=root, completed=True) == directory
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: pytest.fail("Completed comparison rebuilt"))
    assert zonal.run_suite(directory, root=root) == report
    (directory/"comparison.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        zonal.report(directory, root=root)


def test_cli_rejects_conflicting_resume_and_worker_options_before_work():
    from run_nyx_scarcity_zonal import main
    assert main(["--action", "run", "--run-directory", "existing"]) == 2
    assert main(["--action", "prepare", "--run-directory", "existing"]) == 2
    assert main(["--action", "report", "--variant", "zonal_context"]) == 2


def test_launcher_dryrun_from_another_working_directory_has_no_mutation(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([shell, "-NoProfile", "-File", str(ROOT/"ScarcityZonal.ps1"),
                             "-Action", "Run", "-DryRun"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "run_nyx_scarcity_zonal.py" in result.stdout and "shell=False" in result.stdout
    assert not list(tmp_path.iterdir())


def test_launcher_invalid_run_directory_combination_is_rejected(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([shell, "-NoProfile", "-File", str(ROOT/"ScarcityZonal.ps1"),
                             "-Action", "Run", "-RunDirectory", "existing", "-DryRun"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())
