from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import runner

ROOT = Path(__file__).resolve().parents[1]


def sample():
    config = runner.load_config(ROOT / "config/nyx_scarcity.yaml")
    config["zones"] = ["FR"]
    times = pd.date_range("2025-09-15", "2026-09-16", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    days = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    available = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    panel = pd.DataFrame({"zone": "FR", "timestamp_utc": times, "forecast_origin_utc": origin,
                          "forecast": 100., "q10": 80., "q90": 120., "actual": 102.,
                          "benchmark_forecast": 105., "label_available_at_utc": available,
                          "feature_x": 1., "feature_eligible": True, "forecast_eligible": True,
                          "label_eligible": True, "sample": np.where(days >= pd.Timestamp("2026-09-15"), "live", "evaluation")})
    panel.loc[panel["sample"].eq("live"), "actual"] = np.nan
    return config, panel, {"feature_columns": ["feature_x"], "required_feature_columns": ["feature_x"],
                           "evaluation_days": 365, "diagnostic_only": True}


def fake_policy(panel, settings):
    out = panel.copy()
    for name, value in {"candidate_forecast": panel.forecast, "candidate_q10": panel.q10,
                        "candidate_q90": panel.q90, "applied_correction": 0., "raw_correction": 0.,
                        "selected_weight": 0., "expert_ready": False, "spike_probability": np.nan,
                        "gate_reason": "warmup", "threshold_eur_mwh": np.nan,
                        "interval_status": "baseline"}.items():
        out[name] = value
    return SimpleNamespace(predictions=out, folds=pd.DataFrame({"status": ["fallback"]}),
                           governance=pd.DataFrame({"selected_weight": [0.]}), audit={"diagnostic_only": True})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config, panel, audit = sample()
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **kw: (panel.copy(), audit))
    (tmp_path / "Forecast.ps1").write_text("do not change", encoding="utf-8")
    return runner.prepare(config, root=tmp_path), config, panel


def test_lifecycle_reuses_completed_training_and_keeps_operational_files(prepared, tmp_path, monkeypatch):
    from nyx_scarcity import policy
    snapshot, config, panel = prepared
    monkeypatch.setattr(policy, "run_policy", fake_policy)
    report = runner.evaluate(snapshot, root=tmp_path)
    assert report.is_file()
    assert (tmp_path / "Forecast.ps1").read_text() == "do not change"
    predictions = pd.read_parquet(snapshot / "predictions.parquet")
    pd.testing.assert_frame_equal(predictions[panel.columns], panel)
    assert predictions.candidate_forecast.equals(panel.forecast)
    assert predictions.loc[predictions["sample"].eq("live"), "actual"].isna().all()
    checksum = runner.digest(snapshot / "predictions.parquet")
    monkeypatch.setattr(policy, "run_policy", lambda *a: pytest.fail("Completed snapshot must not refit"))
    assert runner.evaluate(snapshot, root=tmp_path) == report
    assert runner.digest(snapshot / "predictions.parquet") == checksum
    assert runner.resolve_snapshot(tmp_path, config, None) == snapshot
    assert runner.resolve_snapshot(tmp_path, config, None, prepared=True) == snapshot


@pytest.mark.parametrize("path", ["runs/exports/new", "runs/live/new", "data/pit/new", "runs/experiments/another", ".", "runs/experiments/nyx_scarcity_v1/../another"])
def test_rejects_output_outside_isolated_namespace(tmp_path, path):
    with pytest.raises(ValueError):
        runner.safe_output(tmp_path, path)


def test_rejects_output_symlink(tmp_path):
    outside = tmp_path / "data" / "pit"
    outside.mkdir(parents=True)
    link = tmp_path / runner.NAMESPACE
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not available")
    with pytest.raises(ValueError):
        runner.safe_output(tmp_path, link / "snapshot")


def test_input_tampering_is_rejected(prepared, tmp_path):
    snapshot, _, _ = prepared
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_snapshot(snapshot, root=tmp_path)


def test_changed_code_rejects_incomplete_snapshot(prepared, tmp_path, monkeypatch):
    snapshot, _, _ = prepared
    monkeypatch.setattr(runner, "code_seals", lambda: {"changed": "checksum"})
    with pytest.raises(ValueError, match="Code changed"):
        runner.evaluate(snapshot, root=tmp_path)


def test_completed_result_tampering_is_rejected(prepared, tmp_path, monkeypatch):
    from nyx_scarcity import policy
    snapshot, _, _ = prepared
    monkeypatch.setattr(policy, "run_policy", fake_policy)
    runner.evaluate(snapshot, root=tmp_path)
    (snapshot / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.report(snapshot, root=tmp_path)


@pytest.mark.parametrize("field,value", [("production_modified", True), ("activation_performed", True),
                                         ("diagnostic_only", False), ("cutoff_time", "10:30"),
                                         ("evaluation_days", 180), ("candidate_model", "kalman"),
                                         ("unknown_option", True), ("zones", ["FR", "FR"])])
def test_bad_recipe_rejected(field, value):
    config, _, _ = sample()
    config[field] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_features_are_locked_to_data_audit():
    config, _, audit = sample()
    config["policy"]["feature_columns"] = ["actual"]
    with pytest.raises(ValueError, match="allowlists"):
        runner.validate_config(config)


@pytest.mark.parametrize("mutation", [lambda p: p.iloc[1:], lambda p: p.assign(actual=999.),
                                       lambda p: p.assign(forecast=999.), lambda p: p.assign(candidate_forecast=999.),
                                       lambda p: p.assign(selected_weight=.8), lambda p: p.assign(candidate_q90=0.),
                                       lambda p: p.assign(spike_probability=1.1)])
def test_candidate_cannot_change_baseline_or_contract(mutation):
    config, panel, _ = sample()
    candidate = mutation(fake_policy(panel, {}).predictions)
    with pytest.raises(ValueError):
        runner.validate_predictions(panel, candidate, config)


def test_launcher_from_another_directory_and_spaces(tmp_path):
    shell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell unavailable")
    completed = subprocess.run([shell, "-NoProfile", "-File", str(ROOT / "Scarcity.ps1"),
                                "-Action", "Run", "-Countries", "FR,DE", "-DeliveryDay", "2026-09-15", "-DryRun"],
                               cwd=tmp_path, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "run_nyx_scarcity.py" in completed.stdout and "aucun calcul" in completed.stdout
    assert not list(tmp_path.iterdir())


def test_cli_blocks_overrides_of_frozen_snapshot():
    from run_nyx_scarcity import main
    assert main(["--action", "backtest", "--zones", "DE"]) == 2


@pytest.mark.parametrize("stage", ["physical", "fuel", "complete"])
def test_cli_refresh_chain_never_prepares_partial_inputs(stage, monkeypatch, tmp_path):
    from run_nyx_scarcity import main
    from nyx_scarcity import refresh, fuel_refresh
    config = runner.load_config(ROOT / "config/nyx_scarcity.yaml")
    config["data"]["source_overrides"] = {"ttf": "private-fuel-marker"}
    events = []
    def physical(current, *, root):
        events.append("physical")
        return {"config": config, "required_sources_complete": stage != "physical", "audit_path": "physical.json"}
    def fuel(current, *, root):
        assert current == config
        events.append("fuel")
        return {"config": config, "required_sources_complete": stage == "complete", "audit_path": "fuel.json"}
    def prepare(current, *, root):
        assert current == config
        events.append("prepare")
        return tmp_path
    monkeypatch.setattr(refresh, "refresh_sources", physical)
    monkeypatch.setattr(fuel_refresh, "refresh_fuels", fuel)
    monkeypatch.setattr(runner, "prepare", prepare)
    assert main(["--action", "prepare", "--refresh-sources"]) == (0 if stage == "complete" else 2)
    assert events == {"physical": ["physical"], "fuel": ["physical", "fuel"],
                      "complete": ["physical", "fuel", "prepare"]}[stage]
