"""CoherentP50 lifecycle tests; fitting/reporting are mocked, no network calls."""
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from nyx_coherent_p50 import runner
from nyx_scarcity import runner as base
from nyx_scarcity.policy import PolicyResult, _parameters
from nyx_fundamental_stress import runner as fundamental
from test_nyx_scarcity_runner import sample, fake_policy

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return runner.load_config(ROOT/"config/nyx_coherent_p50.yaml")


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 2}, {"unknown": 1},
    {"variants": ["forest"]}, {"variants": ["empirical", "forest"]}, {"primary_variant": "empirical"},
    {"max_parallel": True}, {"max_parallel": 3}, {"diagnostic_only": False},
    {"production_modified": True}, {"activation_performed": True}, {"source_suite": ""}, {"output_root": None},
    {"fixed_alpha": .25},
])
def test_recipe_rejects_drift_or_production_flags(change):
    config = recipe()
    config.update(change)
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize("path", ["runs/exports/new", "runs/live/new", "data/pit/new", ".",
    "runs/experiments/nyx_scarcity_v1/fundamental/new", "runs/experiments/nyx_scarcity_v1/variants/new",
    "runs/experiments/nyx_scarcity_v1/coherent_p50/../snapshots/new"])
def test_output_scope_rejects_other_namespaces(tmp_path, path):
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, path)


def test_output_symlink_cannot_escape_namespace(tmp_path):
    elsewhere = tmp_path/"data/pit"
    elsewhere.mkdir(parents=True)
    link = tmp_path/runner.NAMESPACE
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink permission unavailable")
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, link/"new")


@pytest.fixture
def source(tmp_path, monkeypatch):
    baseline, panel, data_audit = sample()
    panel = panel.iloc[:48].copy()
    settings = _parameters(base.model_settings(baseline, data_audit))
    prediction = fake_policy(panel, settings).predictions
    source_config = fundamental.load_config(ROOT/"config/nyx_fundamental_stress.yaml")
    directory = tmp_path/fundamental.NAMESPACE/"snapshots/source"
    directory.mkdir(parents=True)
    audit = {"source_settings": settings, "source_data_audit": data_audit, "source_config": baseline,
             "source_suite_path": "original_frozen_storm_archive", "diagnostic_only": True}
    for name, value in (("config.json", source_config), ("base_config.json", baseline), ("source_audit.json", audit)):
        base._json(directory/name, value)
    for name, frame in (("panel.parquet", panel), ("hgb_predictions.parquet", prediction), ("regional_25_predictions.parquet", prediction)):
        base._parquet(directory/name, frame)
    manifest = {"config": source_config, "settings": settings,
                "input_files": {name: base.digest(directory/name) for name in fundamental.INPUTS}}
    base._json(directory/"manifest.json", manifest)
    for name in source_config["variants"]:
        at = directory/name
        at.mkdir()
        for artifact in ("predictions.parquet", "proposals_25.parquet"):
            base._parquet(at/artifact, prediction)
        base._parquet(at/"folds.parquet", pd.DataFrame({"status": ["trained"], "audit": ['{"core":"past-only"}']}))
        base._parquet(at/"governance.parquet", pd.DataFrame({"status": ["archived"]}))
        base._json(at/"model_audit.json", {"diagnostic_only": True})
        (at/"latest_model.joblib").write_bytes(b"archived mock; never deserialised")
        base._json(at/"results_manifest.json", {"status": "completed", "variant": name,
            "suite_manifest_sha256": base.digest(directory/"manifest.json"),
            "result_files": {key: base.digest(at/key) for key in fundamental.RESULTS}})
    base._json(directory/"comparison.json", {"model_labels": {}})
    base._json(directory/"comparison_manifest.json", {"status": "completed",
        "suite_manifest_sha256": base.digest(directory/"manifest.json"),
        "comparison_sha256": base.digest(directory/"comparison.json"),
        "variant_manifests": {name: base.digest(directory/name/"results_manifest.json") for name in source_config["variants"]}})
    monkeypatch.setattr(runner, "protected_state", lambda *args: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner, "code_seals", lambda *args: {"coherent.py": "frozen"})
    monkeypatch.setattr(runner, "runtime_identity", lambda *args: {"private_runtime": "frozen"})
    config = recipe()
    config["source_suite"] = str(directory)
    return directory, config, panel, tmp_path


@pytest.fixture
def prepared(source):
    directory, config, panel, root = source
    return runner.prepare(config, root=root), config, panel, root, directory


def mock_fit(monkeypatch):
    def fit(panel, source_predictions, source_folds, settings, kind, **kwargs):
        pd.testing.assert_frame_equal(source_predictions[panel.columns], panel)
        assert source_folds.status.tolist() == ["trained"]
        kwargs["on_last_fit"]({"mock_cdf_model": kind}, settings)
        result = fake_policy(panel, settings)
        direct = result.predictions.copy()
        direct["candidate_forecast"] = panel.forecast+2.
        direct["applied_correction"] = 2.
        direct["selected_weight"] = 1.
        direct["expert_ready"] = True
        folds = pd.DataFrame({"status": ["trained"], "audit": [{"core_only": True}]})
        return {"direct": PolicyResult(direct, folds, result.governance, {"decision_policy": "direct", "kind": kind}),
                "governed": PolicyResult(result.predictions, folds, result.governance, {"decision_policy": "governed", "kind": kind})}
    monkeypatch.setattr(runner, "run_coherent_policy", fit)


def test_snapshot_keeps_detector_folds_and_original_provenance(prepared):
    directory, config, panel, root, source = prepared
    source_files = {p.relative_to(source).as_posix(): base.digest(p) for p in source.rglob("*") if p.is_file()}
    pd.testing.assert_frame_equal(pd.read_parquet(directory/"panel.parquet"), panel)
    for target, origin in (("source_predictions", "fundamental/predictions"), ("source_folds", "fundamental/folds"),
                           ("old_fundamental_25_predictions", "fundamental/proposals_25")):
        pd.testing.assert_frame_equal(pd.read_parquet(directory/f"{target}.parquet"), pd.read_parquet(source/f"{origin}.parquet"))
    audit = json.loads((directory/"source_audit.json").read_text())
    assert audit["source_suite_path"] == "original_frozen_storm_archive"
    assert audit["source_data_audit"] and audit["source_config"]
    assert audit["decision_policy"] == "direct" and not audit["classifier_refitted"]
    assert runner.read_suite(directory, root=root)[1] == config
    assert runner.resolve_latest(config, root=root, completed=False) == directory
    assert source_files == {p.relative_to(source).as_posix(): base.digest(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", sorted(runner.INPUTS))
def test_every_input_checksum_verified(prepared, name):
    directory, _, _, root, _ = prepared
    (directory/name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_suite(directory, root=root)


@pytest.mark.parametrize("field", ["config", "settings"])
def test_manifest_cannot_diverge_from_sealed_inputs(prepared, field):
    directory, _, _, root, _ = prepared
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest[field]["max_parallel" if field == "config" else "threads"] = 17
    base._json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.read_suite(directory, root=root)


@pytest.mark.parametrize("target", ["comparison.json", "fundamental/results_manifest.json", "fundamental/predictions.parquet",
                                    "fundamental/folds.parquet", "calendar/model_audit.json"])
def test_source_comparison_and_every_source_variant_are_verified(source, target):
    directory, config, _, root = source
    (directory/target).write_bytes(b"corrupt")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        runner.prepare(config, root=root)


def test_worker_publishes_direct_separately_and_reuses_without_fit(prepared, monkeypatch):
    directory, _, panel, root, _ = prepared
    mock_fit(monkeypatch)
    assert runner.run_worker(directory, "forest", root=root)["status"] == "completed"
    direct = pd.read_parquet(directory/"forest/predictions.parquet")
    governed = pd.read_parquet(directory/"forest/governed_predictions.parquet")
    pd.testing.assert_frame_equal(direct[panel.columns], panel)
    assert direct.candidate_forecast.eq(panel.forecast+2).all()
    assert governed.candidate_forecast.eq(panel.forecast).all()
    model = runner.load_model(directory, "forest", root=root)
    assert model["state"]["mock_cdf_model"] == "forest"
    assert model["suite_manifest_sha256"] == base.digest(directory/"manifest.json")
    hashes = {name: base.digest(directory/"forest"/name) for name in runner.RESULTS}
    monkeypatch.setattr(runner, "run_coherent_policy", lambda *a, **k: pytest.fail("Refit cached CDF"))
    assert runner.run_worker(directory, "forest", root=root)["status"] == "reused"
    assert hashes == {name: base.digest(directory/"forest"/name) for name in runner.RESULTS}


@pytest.mark.parametrize("name", sorted(runner.RESULTS))
def test_all_completed_artifacts_verified_before_reuse(prepared, monkeypatch, name):
    directory, _, _, root, _ = prepared
    mock_fit(monkeypatch)
    runner.run_worker(directory, "forest", root=root)
    (directory/"forest"/name).write_bytes(b"corrupt")
    monkeypatch.setattr(runner, "run_coherent_policy", lambda *a, **k: pytest.fail("Refit corrupt result"))
    with pytest.raises(ValueError, match="checksum"):
        runner.run_worker(directory, "forest", root=root)


def test_drift_blocks_unfinished_fit_and_untrusted_deserialization(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    mock_fit(monkeypatch)
    runner.run_worker(directory, "forest", root=root)
    monkeypatch.setattr(runner, "code_seals", lambda *a: {"changed": "code"})
    monkeypatch.setattr(runner.joblib, "load", lambda *a: pytest.fail("Unverified pickle loaded"))
    with pytest.raises(ValueError, match="Code/runtime changed"):
        runner.load_model(directory, "forest", root=root)
    with pytest.raises(ValueError, match="Code/runtime changed"):
        runner.run_worker(directory, "empirical", root=root)


def test_concurrent_input_change_prevents_publication(prepared, monkeypatch):
    directory, _, _, root, _ = prepared
    mock_fit(monkeypatch)
    original = runner.run_coherent_policy
    def fit(*args, **kwargs):
        result = original(*args, **kwargs)
        (directory/"source_folds.parquet").write_bytes(b"changed during fit")
        return result
    monkeypatch.setattr(runner, "run_coherent_policy", fit)
    with pytest.raises(ValueError, match="checksum"):
        runner.run_worker(directory, "forest", root=root)
    assert not (directory/"forest/results_manifest.json").exists()


def test_unregistered_worker_cannot_write_path_traversal(prepared):
    directory, _, _, root, _ = prepared
    with pytest.raises(ValueError, match="Unregistered"):
        runner.run_worker(directory, "../../exports/escape", root=root)
    assert not (root/"runs/exports").exists()


def test_report_no_fit_and_direct_primary_with_frozen_comparators(prepared, monkeypatch):
    from nyx_scarcity import variant_reporting
    from nyx_coherent_p50 import reporting
    directory, config, _, root, _ = prepared
    mock_fit(monkeypatch)
    for name in config["variants"]:
        runner.run_worker(directory, name, root=root)
    frames, _ = runner.collect(directory, config, runner.read_suite(directory, root=root)[2])
    assert set(frames) == {"hgb_v1", "regional_25", "fundamental_old", "fundamental_old_25", "forest", "forest_governed", "empirical", "empirical_governed"}
    monkeypatch.setattr(runner, "run_coherent_policy", lambda *a, **k: pytest.fail("Cached model refitted"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Worker respawned"))
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: {"model_labels": {}})
    def render(predictions, comparison, audit, path):
        assert comparison["primary_variant"] == "forest"
        assert not comparison["strict_governor_enforced_in_primary"]
        path.write_text("<html><main>frozen</main></html>", encoding="utf-8")
    policies = []
    def render_reports(predictions, **kwargs):
        policies.append((kwargs["decision_policy"], kwargs["model_name"]))
        at = kwargs["output_directory"]
        at.mkdir(parents=True, exist_ok=True)
        index = at/"index.html"
        index.write_text("<html>"+kwargs["decision_policy"]+"</html>", encoding="utf-8")
        return {"index": index}
    monkeypatch.setattr(variant_reporting, "render_comparison", render)
    monkeypatch.setattr(reporting, "render_p50_reports", render_reports)
    path = runner.run_suite(directory, root=root)
    assert path == directory/"reports/index.html" and (directory/"reports_governed/index.html").is_file()
    assert policies == [("direct", "coherent_p50"), ("governed", "coherent_p50_governed")]
    assert runner.resolve_latest(config, root=root, completed=True) == directory
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: pytest.fail("Frozen comparison recalculated"))
    assert runner.run_suite(directory, root=root) == path
    (directory/"comparison.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        runner.report(directory, root=root)


def test_reporting_failure_keeps_results_and_source_unchanged(prepared, monkeypatch):
    directory, config, _, root, source = prepared
    mock_fit(monkeypatch)
    for name in config["variants"]:
        runner.run_worker(directory, name, root=root)
    before = {p.relative_to(root).as_posix(): base.digest(p) for p in root.rglob("*") if p.is_file()}
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic report error")
    monkeypatch.setattr(runner, "_report", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.report(directory, root=root)
    status = json.loads((directory/"status.json").read_text())
    assert status["stage"] == "reporting" and status["predictions_retained"]
    status_key = (directory/"status.json").relative_to(root).as_posix()
    after = {p.relative_to(root).as_posix(): base.digest(p) for p in root.rglob("*") if p.is_file()}
    assert {k:v for k,v in before.items() if k != status_key} == {k:v for k,v in after.items() if k != status_key}


@pytest.mark.parametrize("args", [["--action", "run", "--run-directory", "existing"],
    ["--action", "prepare", "--run-directory", "existing"], ["--action", "report", "--variant", "forest"],
    ["--action", "worker"], ["--action", "worker", "--variant", "forest"]])
def test_cli_conflicting_arguments_fail_without_work(args):
    from run_nyx_coherent_p50 import main
    assert main(args) == 2


def test_powershell_dryrun_is_portable_and_creates_nothing(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([shell, "-NoProfile", "-File", str(ROOT/"CoherentP50.ps1"), "-Action", "Run", "-DryRun"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "run_nyx_coherent_p50.py" in result.stdout and "shell=False" in result.stdout
    assert not list(tmp_path.iterdir())
