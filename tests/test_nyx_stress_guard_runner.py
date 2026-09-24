"""Isolated launcher/lifecycle tests; all replay/model/network work is stubbed."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from nyx_stress_guard import runner

ROOT = Path(__file__).resolve().parents[1]


def config():
    return runner.load_config(ROOT/"config/nyx_stress_guard.yaml")


@pytest.fixture
def source(tmp_path, monkeypatch):
    cfg = config()
    cfg["source_suite"] = "runs/experiments/nyx_scarcity_v1/coherent_p50/snapshots/source"
    directory = tmp_path/cfg["source_suite"]
    directory.mkdir(parents=True)
    p = {"threads": 1, "correction_clip_eur_mwh": 400., "feature_columns": ["feature_fundamental_x"]}
    manifest = {"settings": p, "identity": "first"}
    panel = pd.DataFrame({"zone": ["FR"], "timestamp_utc": [pd.Timestamp("2026-09-15T00:00:00Z")],
        "forecast_origin_utc": [pd.Timestamp("2026-09-14T06:00:00Z")], "forecast": [100.],
        "q10": [80.], "q90": [120.], "actual": [102.], "feature_fundamental_x": [1.]})
    for name, value in {"manifest": manifest, "base_config": {}, "source_audit": {"source_settings": p},
                        "status": {"status": "completed"}}.items():
        runner.base._json(directory/f"{name}.json", value)
    panel.to_parquet(directory/"panel.parquet", index=False)
    panel.to_parquet(directory/"hgb_predictions.parquet", index=False)
    for kind in ("forest", "empirical"):
        (directory/kind).mkdir()
        panel.assign(candidate_forecast=101.).to_parquet(directory/kind/"predictions.parquet", index=False)
        runner.base._json(directory/kind/"results_manifest.json", {"kind": kind})
    monkeypatch.setattr(runner.previous, "read_suite", lambda path, **kw: (directory, {}, deepcopy(manifest)))
    monkeypatch.setattr(runner.previous, "verify_result", lambda *args: None)
    monkeypatch.setattr(runner.previous, "protected_state", lambda root: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner.previous, "runtime_identity", lambda root: {"python": "test"})
    monkeypatch.setattr(runner, "code_seals", lambda root: {"new.py": "sealed"})
    return cfg, directory, panel, manifest


@pytest.fixture
def prepared(source, tmp_path):
    cfg, directory, panel, manifest = source
    return runner.prepare(cfg, root=tmp_path), cfg, panel


@pytest.mark.parametrize("field,value", [("unknown", 1), ("schema_version", True),
    ("production_modified", True), ("activation_performed", True), ("diagnostic_only", False),
    ("primary_variant", "physics_direct"), ("event_probability_gate", .6), ("event_probability_gate", True),
    ("amplitude_kind", "forest"), ("intervals", {"lookback_days": 10}), ("source_suite", ""),
    ("prospective", {"issue_policy": "late", "deadline_local": "11:45"})])
def test_fixed_recipe_rejects_mutations(field, value):
    cfg = config(); cfg[field] = value
    with pytest.raises(ValueError):
        runner.validate_config(cfg)


def test_strict_08_grade_is_explicitly_supported():
    cfg = config(); cfg["prospective"]["issue_policy"] = "strict_08_issue"
    runner.validate_config(cfg)


@pytest.mark.parametrize("value", ["runs/exports/x", "runs/live/x", "data/pit/x", ".",
    "runs/experiments/nyx_scarcity_v1/stress_guard", "runs/experiments/nyx_scarcity_v1/coherent_p50/x",
    "runs/experiments/nyx_scarcity_v1/stress_guard/../x"])
def test_output_scope_is_private(tmp_path, value):
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, value)


def test_prepare_preserves_inputs_and_links_settings_to_audit(prepared, tmp_path):
    snapshot, cfg, panel = prepared
    _, _, manifest = runner.read_suite(snapshot, root=tmp_path)
    assert set(manifest["input_files"]) == runner.INPUTS
    pd.testing.assert_frame_equal(pd.read_parquet(snapshot/"panel.parquet"), panel, check_exact=True)
    assert manifest["production_modified"] is False
    assert runner.resolve_latest(cfg, root=tmp_path) == snapshot
    assert manifest["code_sha256"] == {"new.py": "sealed"}


@pytest.mark.parametrize("name,value", [("settings", {"threads": 999}), ("source_identity", {}),
    ("production_modified", True), ("activation_performed", True), ("schema_version", True),
    ("runtime", {}), ("code_sha256", {}), ("protected_files", {})])
def test_manifest_drift_refused_even_if_input_files_unchanged(prepared, tmp_path, name, value):
    snapshot, _, _ = prepared
    manifest = json.loads((snapshot/"manifest.json").read_text())
    manifest[name] = value
    runner.base._json(snapshot/"manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.read_suite(snapshot, root=tmp_path)


def test_input_corruption_is_rejected(prepared, tmp_path):
    snapshot, _, _ = prepared
    (snapshot/"source_audit.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_suite(snapshot, root=tmp_path)


def test_prepare_rejects_unlinked_source_settings(source, tmp_path):
    cfg, source_dir, _, _ = source
    runner.base._json(source_dir/"source_audit.json", {"source_settings": {}})
    with pytest.raises(ValueError, match="settings"):
        runner.prepare(cfg, root=tmp_path)
    assert not (tmp_path/runner.NAMESPACE).exists()


def test_prepare_rechecks_source_identity_after_copy_reads(source, tmp_path, monkeypatch):
    cfg, source_dir, _, manifest = source
    calls = []
    def changed(path, **kwargs):
        calls.append(path)
        result = deepcopy(manifest)
        if len(calls) == 2:
            result["identity"] = "second"
        return source_dir, {}, result
    monkeypatch.setattr(runner.previous, "read_suite", changed)
    with pytest.raises(ValueError, match="identity changed"):
        runner.prepare(cfg, root=tmp_path)
    assert not (tmp_path/runner.NAMESPACE).exists()


def install_replay_stubs(monkeypatch, *, trained=True, failure=False):
    from nyx_stress_guard import policy
    def replay(panel, settings, **kwargs):
        if failure:
            raise RuntimeError("synthetic replay failure")
        if trained:
            kwargs["on_last_fit"]({"fit_day": "2026-09-15"}, settings)
        result = SimpleNamespace(predictions=panel.assign(candidate_forecast=101.),
            folds=pd.DataFrame({"fit_day": ["2026-09-15"], "status": ["trained"]}),
            governance=pd.DataFrame({"selected_weight": [0.]}), audit={"diagnostic_only": True})
        return {"direct": result, "governed": result}
    monkeypatch.setattr(policy, "run_stress_policy", replay)
    monkeypatch.setattr(policy, "recalibrate_previous", lambda panel, **kw: (panel, {}))
    monkeypatch.setattr(runner, "report", lambda directory, **kw: directory/"test_report.html")


def test_completed_replay_reuses_frozen_artifacts_without_fitting(prepared, tmp_path, monkeypatch):
    from nyx_stress_guard import policy
    snapshot, _, panel = prepared
    install_replay_stubs(monkeypatch)
    report = runner.run_suite(snapshot, root=tmp_path)
    hashes = {n: runner.base.digest(snapshot/n) for n in runner.INPUTS|runner.OUTPUTS}
    monkeypatch.setattr(policy, "run_stress_policy", lambda *a, **kw: pytest.fail("No repeated fit"))
    assert runner.run_suite(snapshot, root=tmp_path) == report
    assert hashes == {n: runner.base.digest(snapshot/n) for n in runner.INPUTS|runner.OUTPUTS}
    assert runner.load_model(snapshot, root=tmp_path)["state"]["fit_day"] == "2026-09-15"


@pytest.mark.parametrize("trained,failure", [(False, False), (True, True)])
def test_replay_failure_is_recorded_without_sealing_result(prepared, tmp_path, monkeypatch, trained, failure):
    snapshot, _, _ = prepared
    before = {n: runner.base.digest(snapshot/n) for n in runner.INPUTS}
    install_replay_stubs(monkeypatch, trained=trained, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run_suite(snapshot, root=tmp_path)
    assert json.loads((snapshot/"status.json").read_text())["status"] == "failed"
    assert not (snapshot/"results_manifest.json").exists()
    assert before == {n: runner.base.digest(snapshot/n) for n in runner.INPUTS}


def test_tampered_model_is_never_deserialised(prepared, tmp_path, monkeypatch):
    snapshot, _, _ = prepared
    install_replay_stubs(monkeypatch)
    runner.run_suite(snapshot, root=tmp_path)
    (snapshot/"latest_model.joblib").write_bytes(b"changed")
    monkeypatch.setattr(runner.joblib, "load", lambda *a: pytest.fail("Unverified pickle must never load"))
    with pytest.raises(ValueError, match="checksum"):
        runner.load_model(snapshot, root=tmp_path)


def test_code_seals_include_old_dependencies_and_all_new_modules(tmp_path, monkeypatch):
    (tmp_path/"nyx_stress_guard").mkdir()
    for name in ("a.py", "b.py"):
        (tmp_path/"nyx_stress_guard"/name).write_text(name)
    for name in ("run_nyx_stress_guard.py", "StressGuard.ps1"):
        (tmp_path/name).write_text(name)
    monkeypatch.setattr(runner.previous, "code_seals", lambda root: {"old.py": "old_sha"})
    seals = runner.code_seals(tmp_path)
    assert set(seals) == {"old.py", "nyx_stress_guard/a.py", "nyx_stress_guard/b.py", "run_nyx_stress_guard.py", "StressGuard.ps1"}


def test_powershell_dryrun_is_networkless_and_does_not_execute_python(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell unavailable")
    sentinel = tmp_path/"not actually python.exe"
    sentinel.write_text("This is intentionally not executable.")
    done = subprocess.run([shell, "-NoProfile", "-File", str(ROOT/"StressGuard.ps1"),
        "-Action", "Issue", "-PythonExecutable", str(sentinel), "-Config", "a folder/config.yaml",
        "-LedgerDirectory", "runs/experiments/nyx_scarcity_v1/stress_guard/my ledger",
        "-DeliveryDay", "2026-09-16", "-DryRun"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout+done.stderr
    argv = json.loads(next(line.split(": ", 1)[1] for line in done.stdout.splitlines() if line.startswith("Commande")))
    assert argv[0] == str(sentinel)
    assert argv[argv.index("--config")+1] == str(ROOT/"a folder/config.yaml")
    assert argv[argv.index("--action")+1] == "issue"
    assert "aucun calcul, appel API ou fichier cree" in done.stdout
    assert list(tmp_path.iterdir()) == [sentinel]
