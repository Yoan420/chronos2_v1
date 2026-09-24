from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_rmse import runner

ROOT = Path(__file__).resolve().parents[1]


def test_private_namespace_and_traversal(tmp_path):
    allowed = tmp_path/runner.NAMESPACE/"snapshots/test"
    assert runner.safe_path(tmp_path, allowed) == allowed
    for path in (tmp_path, tmp_path/"runs/exports/test", tmp_path/runner.NAMESPACE/"../other"):
        with pytest.raises(ValueError):
            runner.safe_path(tmp_path, path)


def test_junction_or_symlink_rejected(tmp_path):
    inside = tmp_path/runner.NAMESPACE
    inside.mkdir(parents=True)
    outside = tmp_path/"outside"
    outside.mkdir()
    try:
        (inside/"alias").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink permission unavailable")
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, inside/"alias/test")


def test_fit_cache_bound_and_corruption_fails_closed(tmp_path, monkeypatch):
    directory = tmp_path/runner.NAMESPACE/"snapshots/test"
    cache = runner.FitCache(directory, "suite-a", root=tmp_path)
    day = "2026-01-02"
    assert cache.load(day) is None
    state = {"fit_day": day, "model": [1, 2]}
    cache.save(day, state)
    assert cache.load(day) == state
    assert (cache.saved, cache.reused) == (1, 1)
    with pytest.raises(ValueError):
        cache.save(day, state)
    monkeypatch.setattr(runner.joblib, "load", lambda _: pytest.fail("Unverified pickle loaded"))
    other = runner.FitCache(directory, "suite-b", root=tmp_path)
    with pytest.raises(ValueError, match="Checkpoint invalide"):
        other.load(day)
    (cache.directory/(day+".joblib")).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checkpoint invalide"):
        cache.load(day)


def test_uncommitted_fit_is_not_loaded_and_can_resume(tmp_path):
    cache = runner.FitCache(tmp_path/runner.NAMESPACE/"snapshots/test", "id", root=tmp_path)
    day = "2026-01-02"
    (cache.directory/(day+".joblib")).write_bytes(b"incomplete")
    assert cache.load(day) is None
    cache.save(day, {"fit_day": day})
    assert cache.load(day) == {"fit_day": day}
    for bad in ("../../escape", "2026-99-99", 1, "2026-1-2"):
        with pytest.raises(ValueError):
            cache.load(bad)


def test_mean_is_not_constrained_by_baseline_quantiles():
    panel = pd.DataFrame({"forecast": [100., np.nan], "actual": [120., np.nan],
                          "q10": [80., np.nan], "q90": [110., np.nan]})
    pred = panel.copy()
    for name in runner.MODELS:
        pred[name] = [200., np.nan]
    runner.validate_predictions(panel, pred)
    pred.loc[1, "nyx_rmse"] = 0.
    with pytest.raises(ValueError, match="Support"):
        runner.validate_predictions(panel, pred)
    pred.loc[1, "nyx_rmse"] = np.nan
    pred["candidate_q90"] = pred.q90
    with pytest.raises(ValueError, match="distribution"):
        runner.validate_predictions(panel, pred)


def test_no_reference_revision_allowed():
    panel = pd.DataFrame({"forecast": [100.], "actual": [120.]})
    pred = panel.assign(**{name: [101.] for name in runner.MODELS})
    pred.actual = 121.
    with pytest.raises(AssertionError):
        runner.validate_predictions(panel, pred)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    config = runner.load_config(ROOT/"config/nyx_rmse.yaml")
    source = tmp_path/config["source_suite"]
    source.mkdir(parents=True)
    settings = {"training_window_days": 365}
    runner._json(source/"manifest.json", {"settings": settings})
    for name in runner.INPUTS - {"config.json"}:
        pd.DataFrame({"value": [1, 2]}).to_parquet(source/name)
    source_hashes = {p.name: runner.digest(p) for p in source.iterdir()}
    monkeypatch.setattr(runner, "inspect_source", lambda *a, **k: (source, {"settings": settings}, source_hashes))
    monkeypatch.setattr(runner.source_runner, "protected_state", lambda _: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner, "training_code", lambda _: {"code": "same"})
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"runtime": "same"})
    directory = runner.prepare(config, root=tmp_path)
    return tmp_path, directory, config, source


def test_prepare_inputs_frozen_and_pointer_config_guard(frozen):
    root, directory, config, source = frozen
    checked, loaded, manifest = runner.read_snapshot(directory, root=root)
    assert loaded == config
    assert checked == directory
    assert runner.resolve_snapshot(config, root=root) == directory
    revised = deepcopy(config)
    revised["options"]["learner"]["max_iter"] += 1
    with pytest.raises(ValueError, match="Configuration modifiee"):
        runner.resolve_snapshot(revised, root=root)
    with pytest.raises(ValueError, match="Configuration modifiee"):
        runner.resolve_snapshot(revised, root=root, value=str(directory))
    assert manifest["input_files"]["panel.parquet"] == runner.digest(source/"panel.parquet")
    (source/"panel.parquet").write_bytes(b"revision")
    with pytest.raises(ValueError, match="SHA divergent"):
        runner.read_snapshot(directory, root=root)


def test_modified_runtime_or_code_refuses_resume(frozen, monkeypatch):
    root, directory, _, _ = frozen
    monkeypatch.setattr(runner, "training_code", lambda _: {"code": "different"})
    with pytest.raises(ValueError, match="Code/runtime"):
        runner.read_snapshot(directory, root=root)


def test_result_reuse_without_fitting(frozen, monkeypatch):
    root, directory, _, _ = frozen
    from nyx_rmse import policy
    monkeypatch.setattr(policy, "run_replay", lambda *a, **k: pytest.fail("Unexpected refit"))
    for name in runner.RESULTS:
        (directory/name).write_bytes(b"sealed result")
    runner._json(directory/"results_manifest.json", {"status": "completed",
        "suite_manifest_sha256": runner.digest(directory/"manifest.json"),
        "result_files": {n: runner.digest(directory/n) for n in runner.RESULTS}})
    assert runner.evaluate(directory, root=root) == directory
    (directory/"predictions.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA divergent"):
        runner.evaluate(directory, root=root)


def test_unknown_config_and_activation_rejected():
    config = runner.load_config(ROOT/"config/nyx_rmse.yaml")
    for key, value in (("activation_performed", True), ("diagnostic_only", False), ("production_modified", True), ("unknown", 1)):
        candidate = {**config, key: value}
        with pytest.raises(ValueError):
            runner.validate_config(candidate)


def test_report_status_bound_to_snapshot_results(frozen):
    root, directory, _, _ = frozen
    for name in runner.RESULTS:
        (directory/name).write_bytes(b"result")
    runner._json(directory/"results_manifest.json", {"status": "completed",
        "suite_manifest_sha256": runner.digest(directory/"manifest.json"),
        "result_files": {n: runner.digest(directory/n) for n in runner.RESULTS}})
    target = directory/"reports/test"
    target.mkdir(parents=True)
    (target/"test.html").write_text("<html>ok</html>")
    report = {"status": "completed", "suite_manifest_sha256": runner.digest(directory/"manifest.json"),
              "results_manifest_sha256": runner.digest(directory/"results_manifest.json"),
              "files": {"test.html": runner.digest(target/"test.html")}}
    def publish(value):
        runner._json(target/"report_manifest.json", value)
        runner._json(directory/"latest_report.json", {"snapshot": str(directory), "report_directory": str(target),
            "manifest_sha256": runner.digest(directory/"manifest.json"),
            "report_manifest_sha256": runner.digest(target/"report_manifest.json")})
    publish(report)
    assert runner.status(directory, root=root)["report_verified"]
    publish({**report, "results_manifest_sha256": "another-valid-run"})
    with pytest.raises(ValueError, match="Rapport non lie"):
        runner.status(directory, root=root)
