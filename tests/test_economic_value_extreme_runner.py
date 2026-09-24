from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from economic_value import extreme_runner as runner
from economic_value import runner as base

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = runner.load_config(ROOT / "config/economic_extreme_policy.yaml")
    config["zones"] = ["FR"]
    original = base.load_config(ROOT / "config/economic_value.yaml")
    original.update(zones=["FR"], models=["nuclear_kalman"])
    times = pd.date_range("2025-09-11", "2026-09-11", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    days = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (days-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    panel = pd.DataFrame({"timestamp_utc": times, "zone": "FR", "model": "nuclear_kalman", "forecast_origin_utc": origin,
                          "forecast": 80., "q10": 60., "q90": 100., "actual": 90., "reference_price": 50.,
                          "benchmark_forecast": 70., "reference_available_at_utc": origin-pd.Timedelta(hours=14),
                          "duration_hours": 1., "forecast_eligible": True, "reference_eligible": True,
                          "sample": "evaluation", "feature_test": 1., "feature_eligible": True})
    audit = {**base._validate_panel(panel, original), "feature_columns": ["feature_test"], "required_core_feature_columns": ["feature_test"]}
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **kw: (panel.copy(), panel.copy(), audit, original))
    (tmp_path / "Forecast.ps1").write_text("operational script unchanged", encoding="utf-8")
    return runner.prepare(config, root=tmp_path), config, panel


def fake_train(history, panel, config):
    assert config["training_days"] == 365
    assert config["feature_columns"] == ["feature_test"]
    d = panel[["timestamp_utc", "zone"]].copy()
    d["baseline_position_fraction"] = 1.
    d["policy_position_fraction"] = .5
    d["policy_available_at_utc"] = panel.forecast_origin_utc
    d["governance_weight"] = 1.
    d["policy_reason"] = "test_policy"
    d["extreme_probability_up"] = .8
    d["extreme_probability_down"] = .1
    d["extreme_expected_edge"] = 40.
    return SimpleNamespace(decisions=d, folds=pd.DataFrame({"status": ["fitted"]}), audit={"causal": True}, governance=pd.DataFrame())


def test_full_snapshot_evaluation_reuse_and_manifest_integrity(prepared, tmp_path, monkeypatch):
    from economic_value import extreme_policy
    snapshot, config, panel = prepared
    monkeypatch.setattr(extreme_policy, "run_extreme_policy", fake_train)
    report = runner.evaluate(snapshot, root=tmp_path)
    assert report.is_file()
    rows = pd.read_parquet(snapshot / "rows.parquet")
    model = rows.loc[rows.strategy.eq("model")]
    assert model.forecast.eq(80).all()
    assert model.loc[model.model.eq(config["candidate_model"]), "position_mw"].eq(50).all()
    audit = json.loads((snapshot / "results_manifest.json").read_text())
    assert audit["extreme_policy"]["baseline_comparison_paired"] is True
    assert audit["forecast_values_unchanged"] is True
    before = base.digest(snapshot / "rows.parquet")
    monkeypatch.setattr(extreme_policy, "run_extreme_policy", lambda *a, **kw: pytest.fail("Should not train completed snapshot"))
    runner.evaluate(snapshot, root=tmp_path)
    assert base.digest(snapshot / "rows.parquet") == before
    assert (tmp_path / "Forecast.ps1").read_text() == "operational script unchanged"
    audit["snapshot_files"]["config.json"] = "another_run"
    base._json(snapshot / "results_manifest.json", audit)
    with pytest.raises(ValueError, match="manifest mismatch"):
        runner.report(snapshot, root=tmp_path)


def test_partial_policy_cannot_shorten_comparison(prepared, tmp_path, monkeypatch):
    from economic_value import extreme_policy
    snapshot, _, _ = prepared
    def incomplete(*args):
        result = fake_train(*args)
        result.decisions = result.decisions.iloc[1:]
        return result
    monkeypatch.setattr(extreme_policy, "run_extreme_policy", incomplete)
    with pytest.raises(ValueError, match="retain every"):
        runner.evaluate(snapshot, root=tmp_path)
    assert json.loads((snapshot / "status.json").read_text())["status"] == "failed"
    assert not (snapshot.parent.parent / "latest.json").exists()


def test_read_inputs_checksum_and_code_seal(prepared, tmp_path, monkeypatch):
    snapshot, _, _ = prepared
    monkeypatch.setattr(runner, "_seals", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="Code changed"):
        runner.evaluate(snapshot, root=tmp_path)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.read_snapshot(snapshot, root=tmp_path)


@pytest.mark.parametrize("key,value", [("training_days", 364), ("production_modified", True), ("zones", ["FR", "FR"]), ("order_execution_enabled", True)])
def test_invalid_recipe(key, value):
    config = runner.load_config(ROOT / "config/economic_extreme_policy.yaml")
    config[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_cannot_move_costs_into_expert_configuration():
    config = runner.load_config(ROOT / "config/economic_extreme_policy.yaml")
    original = base.load_config(ROOT / "config/economic_value.yaml")
    config["expert"]["transaction_cost_eur_mwh"] = 0
    with pytest.raises(ValueError, match="cannot be overridden"):
        runner.policy_config(config, original)


def test_powershell_dryrun_from_another_directory(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell unavailable")
    call = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "EconomicExpert.ps1"), "-Action", "Run", "-DryRun"], cwd=tmp_path, capture_output=True, text=True)
    assert call.returncode == 0, call.stdout + call.stderr
    assert "run_economic_extreme_policy.py" in call.stdout
    assert "aucun calcul" in call.stdout
    assert not list(tmp_path.iterdir())
