"""Private namespace, artifact contracts and safe resume; no network or fit."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_physical_p50 import runner

ROOT = Path(__file__).resolve().parents[1]


def test_namespace_production_and_traversal_refused(tmp_path):
    inside = tmp_path/runner.NAMESPACE/"snapshots/example"
    assert runner.safe_path(tmp_path, inside) == inside
    for target in (tmp_path, tmp_path/"runs/exports/fr", tmp_path/"runs/live/fr",
                   tmp_path/runner.NAMESPACE/"../other", tmp_path/"Forecast.ps1"):
        with pytest.raises(ValueError):
            runner.safe_path(tmp_path, target)


def test_namespace_alias_refused(tmp_path):
    inside, outside = tmp_path/runner.NAMESPACE, tmp_path/"outside"
    inside.mkdir(parents=True)
    outside.mkdir()
    try:
        (inside/"alias").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink permission unavailable")
    with pytest.raises(ValueError):
        runner.safe_path(tmp_path, inside/"alias/report.html")


def test_both_models_in_one_fold_checkpoint(tmp_path):
    cache = runner.FitCache(tmp_path/runner.NAMESPACE/"snapshots/test", "snapshot-a", root=tmp_path)
    day = "2026-09-14"
    state = {"fit_day": day, "models": {"fuel_transport": {"value": 1}, "network_fuel": None}}
    assert cache.load(day) is None
    cache.save(day, state)
    assert cache.load(day) == state
    assert cache.load(day) == state
    assert (cache.saved, cache.reused) == (1, 2)
    with pytest.raises(ValueError):
        cache.save(day, state)


def test_checkpoint_seal_precedes_pickle_deserialization(tmp_path, monkeypatch):
    directory = tmp_path/runner.NAMESPACE/"snapshots/test"
    cache = runner.FitCache(directory, "a", root=tmp_path)
    day = "2026-09-14"
    cache.save(day, {"fit_day": day, "models": {}})
    monkeypatch.setattr(runner.joblib, "load", lambda _: pytest.fail("Unverified checkpoint deserialized"))
    with pytest.raises(ValueError, match="Checkpoint invalide"):
        runner.FitCache(directory, "b", root=tmp_path).load(day)
    (cache.directory/(day+".joblib")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="Checkpoint invalide"):
        cache.load(day)


def test_uncommitted_checkpoint_is_ignored_and_recoverable(tmp_path):
    cache = runner.FitCache(tmp_path/runner.NAMESPACE/"snapshots/test", "a", root=tmp_path)
    day = "2026-09-14"
    (cache.directory/(day+".joblib")).write_bytes(b"uncommitted")
    assert cache.load(day) is None
    cache.save(day, {"fit_day": day, "models": {}})
    assert cache.load(day)["fit_day"] == day
    for bad in ("../../escape", "2026-99-14", "2026-9-14", 1, None):
        with pytest.raises(ValueError):
            cache.load(bad)


def predictions():
    panel = pd.DataFrame({"forecast": [100., 150.], "actual": [120., 180.],
                          "q10": [80., 130.], "q90": [110., 170.]})
    result = panel.copy()
    for name in runner.MODELS:
        result[name] = [140., 200.]
        result[name+"_q10"] = [90., 140.]
        result[name+"_q90"] = [180., 230.]
    return panel, result


def test_genuine_quantiles_can_leave_old_baseline_interval():
    panel, result = predictions()
    runner.validate_predictions(panel, result)
    assert result.nyx_physical_p50.iloc[0] > panel.q90.iloc[0]


@pytest.mark.parametrize("change", ["crossing_lower", "crossing_upper", "missing_bound", "nonfinite", "missing_prediction"])
def test_invalid_point_or_distribution_refused(change):
    panel, result = predictions()
    if change == "crossing_lower":
        result.loc[0, "nyx_physical_p50_q10"] = 200.
    elif change == "crossing_upper":
        result.loc[0, "nyx_physical_p50_q90"] = 100.
    elif change == "missing_bound":
        result = result.drop(columns="nyx_physical_p50_q90")
    elif change == "nonfinite":
        result.loc[0, "nyx_physical_p50"] = np.inf
    else:
        result.loc[0, "nyx_physical_p50"] = np.nan
    with pytest.raises(ValueError):
        runner.validate_predictions(panel, result)


def test_original_forecast_observation_and_quantiles_are_immutable():
    for column in ("forecast", "actual", "q10", "q90"):
        panel, result = predictions()
        result.loc[0, column] += 1.
        with pytest.raises(AssertionError):
            runner.validate_predictions(panel, result)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    config = runner.load_config(ROOT/"config/nyx_physical_p50.yaml")
    source = tmp_path/config["source_suite"]
    source.mkdir(parents=True)
    settings = {"training_window_days": 365}
    runner._json(source/"manifest.json", {"settings": settings})
    for name in ("panel.parquet", "source_predictions.parquet", "source_folds.parquet"):
        pd.DataFrame({"value": [1, 2]}).to_parquet(source/name)
    identities = {p.name: runner.digest(p) for p in source.iterdir()}
    monkeypatch.setattr(runner, "inspect_source", lambda *a, **k: (source, {"settings": settings}, identities))
    monkeypatch.setattr(runner.source_runner, "protected_state", lambda _: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner, "training_code", lambda _: {"code": "same"})
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"runtime": "same"})
    monkeypatch.setattr(runner, "_network_inputs", lambda panel, **k:
        (pd.DataFrame({"network_eligible": [False, False], "feature_network_x": [np.nan, np.nan]}),
         {"source_files": {}, "network_missing_policy": "abstain"}))
    directory = runner.prepare(config, root=tmp_path)
    return tmp_path, directory, config, source


def test_prepare_snapshots_derived_network_and_source_inputs(frozen):
    root, directory, config, source = frozen
    checked, frozen_config, manifest = runner.read_snapshot(directory, root=root)
    assert checked == directory and frozen_config == config
    assert set(manifest["input_files"]) == runner.INPUTS
    assert manifest["input_files"]["panel.parquet"] == runner.digest(source/"panel.parquet")
    assert runner.resolve_snapshot(config, root=root) == directory
    assert (directory/"network_audit.json").is_file()
    assert not pd.read_parquet(directory/"network_features.parquet").network_eligible.any()


@pytest.mark.parametrize("input_name", ["panel.parquet", "network_features.parquet", "network_audit.json"])
def test_changed_frozen_input_refuses_resume(frozen, input_name):
    root, directory, _, _ = frozen
    (directory/input_name).write_bytes(b"revision")
    with pytest.raises(ValueError, match="SHA divergent"):
        runner.read_snapshot(directory, root=root)


def test_changed_source_refuses_resume(frozen):
    root, directory, _, source = frozen
    (source/"panel.parquet").write_bytes(b"new observation revision")
    with pytest.raises(ValueError, match="SHA divergent"):
        runner.read_snapshot(directory, root=root)


@pytest.mark.parametrize("changed", ["code", "runtime"])
def test_changed_code_or_runtime_refuses_resume(frozen, monkeypatch, changed):
    root, directory, _, _ = frozen
    if changed == "code":
        monkeypatch.setattr(runner, "training_code", lambda _: {"code": "new"})
    else:
        monkeypatch.setattr(runner, "runtime_identity", lambda: {"runtime": "new"})
    with pytest.raises(ValueError, match="Code/runtime"):
        runner.read_snapshot(directory, root=root)


def test_changed_config_requires_new_snapshot(frozen):
    root, directory, config, _ = frozen
    revised = deepcopy(config)
    revised["options"]["threads"] = 1
    with pytest.raises(ValueError, match="Configuration"):
        runner.resolve_snapshot(revised, root=root)
    with pytest.raises(ValueError, match="Configuration"):
        runner.resolve_snapshot(revised, root=root, value=str(directory))


def seal_results(directory):
    for name in runner.RESULTS:
        (directory/name).write_bytes(b"sealed result")
    runner._json(directory/"results_manifest.json", {"status": "completed",
        "suite_manifest_sha256": runner.digest(directory/"manifest.json"),
        "result_files": {n: runner.digest(directory/n) for n in runner.RESULTS}})


def test_completed_replay_reused_without_model_fit(frozen, monkeypatch):
    from nyx_physical_p50 import policy
    root, directory, _, _ = frozen
    monkeypatch.setattr(policy, "run_replay", lambda *a, **k: pytest.fail("Unexpected fit"))
    seal_results(directory)
    assert runner.evaluate(directory, root=root) == directory
    (directory/"predictions.parquet").write_bytes(b"corrupt result")
    with pytest.raises(ValueError, match="SHA divergent"):
        runner.evaluate(directory, root=root)


def test_report_status_is_bound_to_snapshot_and_result(frozen):
    root, directory, _, _ = frozen
    seal_results(directory)
    target = directory/"reports/example"
    target.mkdir(parents=True)
    (target/"report.html").write_text("<html>ok</html>", encoding="utf-8")
    sealed = {"status": "completed", "suite_manifest_sha256": runner.digest(directory/"manifest.json"),
              "results_manifest_sha256": runner.digest(directory/"results_manifest.json"),
              "files": {"report.html": runner.digest(target/"report.html")}}
    def publish(record):
        runner._json(target/"report_manifest.json", record)
        runner._json(directory/"latest_report.json", {"snapshot": str(directory), "report_directory": str(target),
            "manifest_sha256": runner.digest(directory/"manifest.json"),
            "report_manifest_sha256": runner.digest(target/"report_manifest.json")})
    publish(sealed)
    assert runner.status(directory, root=root)["report_verified"]
    publish({**sealed, "results_manifest_sha256": "different-result"})
    with pytest.raises(ValueError, match="Rapport non lie"):
        runner.status(directory, root=root)


@pytest.mark.parametrize("key,value", [("activation_performed", True), ("production_modified", True),
    ("diagnostic_only", False), ("schema_version", True), ("unknown", 1),
    ("options", {"threads": True}), ("options", {"threads": 4}), ("options", {"threads": 2, "boost_today": 2})])
def test_unknown_parameters_and_production_activation_refused(key, value):
    config = runner.load_config(ROOT/"config/nyx_physical_p50.yaml")
    with pytest.raises(ValueError):
        runner.validate_config({**config, key: value})


@pytest.mark.parametrize("start,end", [(None, "2026-09-15"), ("2026-9-1", "2026-09-15"),
    ("2026-09-31", "2026-10-01"), ("2026-09-15", "2026-09-01"), ("2026-08-01", "2026-09-15")])
def test_collection_range_bounded_before_external_calls(tmp_path, start, end):
    config = runner.load_config(ROOT/"config/nyx_physical_p50.yaml")
    with pytest.raises(ValueError):
        runner.collect(config, root=tmp_path, start_day=start, end_day=end)


def test_prepare_detects_concurrent_production_change(tmp_path, monkeypatch):
    config = runner.load_config(ROOT/"config/nyx_physical_p50.yaml")
    source = tmp_path/config["source_suite"]
    source.mkdir(parents=True)
    runner._json(source/"manifest.json", {"settings": {}})
    for name in ("panel.parquet", "source_predictions.parquet", "source_folds.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(source/name)
    identity = {p.name: runner.digest(p) for p in source.iterdir()}
    monkeypatch.setattr(runner, "inspect_source", lambda *a, **k: (source, {"settings": {}}, identity))
    calls = iter(({"Forecast.ps1": "before"}, {"Forecast.ps1": "after"}))
    monkeypatch.setattr(runner.source_runner, "protected_state", lambda _: next(calls))
    monkeypatch.setattr(runner, "_network_inputs", lambda p, **k: (pd.DataFrame({"x": [1]}), {}))
    with pytest.raises(ValueError, match="production a change"):
        runner.prepare(config, root=tmp_path)
    assert not (tmp_path/config["output_root"]/"latest_prepared.json").exists()
