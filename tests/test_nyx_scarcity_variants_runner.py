from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from nyx_scarcity import runner as base
from nyx_scarcity import variants_runner as suite
from test_nyx_scarcity_runner import sample, fake_policy

ROOT = Path(__file__).resolve().parents[1]


def test_dwt_empty_threshold_audit_exports_real_parquet(tmp_path):
    original = pd.DataFrame({"thresholds_eur_mwh": [{}, {}, None], "days": [90, 97, 104],
                             "metadata": [{"zone": "FR"}, {}, None]})
    result = suite._audit_parquet_frame(original)
    path = tmp_path / "folds.parquet"
    base._parquet(path, result)
    loaded = pd.read_parquet(path)
    assert loaded.thresholds_eur_mwh.tolist() == ["{}", "{}", None]
    assert json.loads(loaded.metadata[0]) == {"zone": "FR"}
    assert loaded.days.tolist() == [90, 97, 104]
    assert original.thresholds_eur_mwh[0] == {}


def recipe():
    return suite.load_config(ROOT / "config/nyx_scarcity_variants.yaml")


@pytest.mark.parametrize("changes", [
    {"max_parallel": 3}, {"max_parallel": True}, {"production_modified": True},
    {"activation_performed": True}, {"diagnostic_only": False}, {"other": 1},
    {"variants": []}, {"variants": [{"id": "../../exports", "weighted": True, "threshold_kind": "dwt"}]},
    {"policy_overrides": {"feature_columns": ["actual"]}},
    {"explanations": {"sample_days_last": 7, "historical_stride_days": 0, "historical_hour": 19}},
])
def test_invalid_recipes(changes):
    config = recipe()
    config.update(changes)
    with pytest.raises(ValueError):
        suite.validate_config(config)


@pytest.mark.parametrize("target", ["runs/exports/x", "runs/experiments/nyx_scarcity_v1/snapshots/a",
                                    "runs/experiments/nyx_scarcity_v1/variants/../snapshots/a", "."])
def test_variant_writes_cannot_touch_prior_snapshots_or_production(tmp_path, target):
    with pytest.raises(ValueError):
        suite.safe_path(tmp_path, target)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    from nyx_scarcity import policy
    cfg, panel, audit = sample()
    audit.update(evaluation_start_day="2025-09-15", evaluation_end_day="2026-09-14", delivery_day="2026-09-15")
    monkeypatch.setattr(base, "audit_inputs", lambda *a, **k: (panel.copy(), audit))
    monkeypatch.setattr(policy, "run_policy", fake_policy)
    source = base.prepare(cfg, root=tmp_path)
    base.evaluate(source, root=tmp_path)
    config = recipe()
    config["source_snapshot"] = str(source)
    config["variants"] = config["variants"][:1]
    monkeypatch.setattr(suite, "runtime_seals", lambda: {"version": "test"})
    return config, source, panel, tmp_path


def test_prepare_copies_exact_control_and_inputs_without_mutating_source(frozen):
    config, source, panel, root = frozen
    before = {name: base.digest(source / name) for name in base.INPUTS | base.RESULTS}
    directory = suite.prepare(config, root=root)
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "panel.parquet"), panel)
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "control_predictions.parquet"), pd.read_parquet(source / "predictions.parquet"))
    assert {name: base.digest(source / name) for name in before} == before
    assert suite.read_suite(directory, root=root)[1] == config
    assert suite.resolve_latest(config, root=root, completed=False) == directory


def test_prepare_rejects_modified_control(frozen):
    config, source, _, root = frozen
    (source / "predictions.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        suite.prepare(config, root=root)


def test_frozen_suite_rejects_tampered_input(frozen):
    config, _, _, root = frozen
    directory = suite.prepare(config, root=root)
    (directory / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        suite.read_suite(directory, root=root)


def test_oversubscribed_cpu_rejected(frozen):
    config, _, _, root = frozen
    config["max_parallel"] = 2
    config["policy_overrides"]["threads"] = 3
    with pytest.raises(ValueError, match="four CPU"):
        suite.prepare(config, root=root)


def test_code_drift_rejects_unfinished_suite(frozen, monkeypatch):
    config, _, _, root = frozen
    directory = suite.prepare(config, root=root)
    monkeypatch.setattr(suite, "code_seals", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="Code/runtime changed"):
        suite.run_suite(directory, root=root)


def test_single_worker_thread_limit_rejected(frozen):
    config, _, _, root = frozen
    config["max_parallel"] = 1
    config["policy_overrides"]["threads"] = 3
    with pytest.raises(ValueError, match="two CPU"):
        suite.prepare(config, root=root)


def test_manifest_settings_must_match_sealed_recipe(frozen):
    config, _, _, root = frozen
    directory = suite.prepare(config, root=root)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["settings"]["threads"] = 1
    base._json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="settings differ"):
        suite.read_suite(directory, root=root)


def test_manifest_change_during_worker_blocks_publication(frozen, monkeypatch):
    from nyx_scarcity import variant_policy
    config, _, _, root = frozen
    directory = suite.prepare(config, root=root)
    name = config["variants"][0]["id"]
    def train(panel, settings, variant, **kwargs):
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["created_at_utc"] = "modified while fitting"
        base._json(directory / "manifest.json", manifest)
        return fake_policy(panel, settings)
    monkeypatch.setattr(variant_policy, "run_variant_policy", train)
    with pytest.raises(ValueError, match="manifest changed during"):
        suite.run_worker(directory, name, root=root)
    assert not (directory / name / "results_manifest.json").exists()


def test_launcher_from_any_directory_and_input_validation(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([shell, "-NoProfile", "-File", str(ROOT / "ScarcityVariants.ps1"),
                             "-Action", "Run", "-DryRun"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_nyx_scarcity_variants.py" in result.stdout
    assert not list(tmp_path.iterdir())


def test_cli_frozen_override_rejected():
    from run_nyx_scarcity_variants import main
    assert main(["--action", "backtest", "--source-snapshot", "other"]) == 2
    assert main(["--action", "run", "--run-directory", "other"]) == 2
    assert main(["--action", "run", "--variant", "xgb_weighted_dwt"]) == 2


def test_worker_cache_and_whole_suite_resume_without_retraining(frozen, monkeypatch):
    from nyx_scarcity import variant_policy, variant_reporting
    config, _, panel, root = frozen
    directory = suite.prepare(config, root=root)
    name = config["variants"][0]["id"]
    monkeypatch.setattr(variant_policy, "run_variant_policy", lambda panel, settings, variant, **kw: fake_policy(panel, settings))
    monkeypatch.setattr(suite, "version", lambda name: "test")
    assert suite.run_worker(directory, name, root=root)["status"] == "completed"
    at = directory / name / "predictions.parquet"
    checksum = base.digest(at)
    pd.testing.assert_frame_equal(pd.read_parquet(at)[panel.columns], panel)
    monkeypatch.setattr(variant_policy, "run_variant_policy", lambda *a, **kw: pytest.fail("Completed variant refitted"))
    assert suite.run_worker(directory, name, root=root)["status"] == "reused"
    monkeypatch.setattr(suite.subprocess, "Popen", lambda *a, **kw: pytest.fail("Completed worker spawned"))
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda predictions: {"variants": list(predictions)})
    def render(predictions, summary, audit, path, explanations=None):
        path.write_text("<html>test report</html>", encoding="utf-8")
        return path
    monkeypatch.setattr(variant_reporting, "render_comparison", render)
    report = suite.run_suite(directory, root=root)
    assert report.is_file() and base.digest(at) == checksum
    monkeypatch.setattr(variant_reporting, "build_comparison", lambda *a: pytest.fail("Completed metrics recomputed"))
    assert suite.run_suite(directory, root=root) == report
    (directory / "comparison.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        suite.report(directory, root=root)


def test_worker_tampering_rejected_and_not_silently_refitted(frozen, monkeypatch):
    from nyx_scarcity import variant_policy
    config, _, _, root = frozen
    directory = suite.prepare(config, root=root)
    name = config["variants"][0]["id"]
    monkeypatch.setattr(variant_policy, "run_variant_policy", lambda panel, settings, variant, **kw: fake_policy(panel, settings))
    monkeypatch.setattr(suite, "version", lambda name: "test")
    suite.run_worker(directory, name, root=root)
    (directory / name / "shap_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        suite.run_worker(directory, name, root=root)
