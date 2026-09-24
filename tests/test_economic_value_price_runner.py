from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from economic_value import price_runner as runner
from economic_value import runner as base

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = runner.load_config(ROOT / "config/nuclear_kalman_extreme.yaml")
    config["zones"] = ["FR"]
    original = base.load_config(ROOT / "config/economic_value.yaml")
    original.update(zones=["FR"], models=["nuclear_kalman"])
    times = pd.date_range("2025-09-11", "2026-09-11", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    days = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (days-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    panel = pd.DataFrame({"timestamp_utc": times, "zone": "FR", "model": "nuclear_kalman", "forecast_origin_utc": origin,
                          "forecast": 40., "q10": 20., "q90": 60., "actual": 70., "reference_price": 50.,
                          "benchmark_forecast": 75., "reference_available_at_utc": origin-pd.Timedelta(hours=14),
                          "duration_hours": 1., "forecast_eligible": True, "reference_eligible": True,
                          "sample": "evaluation", "feature_test": 1., "feature_eligible": True})
    audit = {**base._validate_panel(panel, original), "feature_columns": ["feature_test"], "required_core_feature_columns": ["feature_test"]}
    return config, original, panel, audit


def fake_train(panel, config):
    assert config["training_window_days"] == 365 and config["minimum_training_days"] == 90
    d = panel[["timestamp_utc", "zone", "forecast_origin_utc"]].copy()
    d["baseline_forecast"] = panel.forecast
    d["candidate_forecast"] = panel.forecast+40.
    d["raw_residual_prediction"] = 80.
    d["applied_correction"] = 40.
    d["selected_weight"] = .5
    d["expert_ready"] = True
    d["expert_available_at_utc"] = panel.forecast_origin_utc
    d["reason"] = "test_correction"
    return SimpleNamespace(decisions=d, folds=pd.DataFrame({"status": ["fitted"]}), governance=pd.DataFrame(), audit={"diagnostic": True})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config, original, panel, audit = inputs()
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **kw: (panel.copy(), audit, original))
    (tmp_path / "Forecast.ps1").write_text("operational file", encoding="utf-8")
    return runner.prepare(config, root=tmp_path), config, panel


def test_full_lifecycle_new_price_recomputed_signals_and_no_retraining(prepared, tmp_path, monkeypatch):
    from economic_value import price_policy
    snapshot, config, panel = prepared
    monkeypatch.setattr(price_policy, "run_price_policy", fake_train)
    path = runner.evaluate(snapshot, root=tmp_path)
    assert path.is_file()
    rows = pd.read_parquet(snapshot / "rows.parquet")
    a = rows.loc[rows.strategy.eq("model") & rows.model.eq("nuclear_kalman")]
    b = rows.loc[rows.strategy.eq("model") & rows.model.eq("nuclear_kalman_extreme")]
    assert a.forecast.eq(40).all() and a.position_mw.eq(-100).all()
    assert b.forecast.eq(80).all() and b.position_mw.eq(100).all()
    assert b.q10.isna().all() and b.q90.isna().all()
    assert a.q10.eq(20).all() and a.q90.eq(60).all()
    assert b.trading_cost_eur.eq(100).all()
    assert not any(c.startswith("policy_position") for c in rows)
    audit = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    assert audit["fixed_economic_decision_rule"] is True
    assert audit["price_expert"]["summary"][0]["annual_mae_gain_eur_mwh"] == 20
    before = base.digest(snapshot / "rows.parquet")
    monkeypatch.setattr(price_policy, "run_price_policy", lambda *a: pytest.fail("No retraining completed snapshot"))
    runner.evaluate(snapshot, root=tmp_path)
    assert base.digest(snapshot / "rows.parquet") == before
    assert (tmp_path / "Forecast.ps1").read_text() == "operational file"


def test_candidate_alignment_and_quantiles_only_removed_when_price_changes():
    config, original, panel, audit = inputs()
    decisions = fake_train(panel, runner.model_config(config, original, audit)).decisions
    decisions.loc[0, ["selected_weight", "applied_correction"]] = 0.
    decisions.loc[0, "candidate_forecast"] = panel.loc[0, "forecast"]
    candidate = runner.candidate_panel(panel, decisions.iloc[::-1], config)
    assert candidate.loc[0, "q10"] == panel.loc[0, "q10"]
    assert candidate.loc[1:, "q10"].isna().all()
    pd.testing.assert_series_equal(candidate.actual, panel.actual)


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.iloc[1:].copy(), "every baseline interval"),
    (lambda d: d.assign(baseline_forecast=90.), "frozen baseline"),
    (lambda d: d.assign(candidate_forecast=500.), "baseline plus"),
    (lambda d: d.assign(expert_available_at_utc=d.forecast_origin_utc+pd.Timedelta(seconds=1)), "08:00"),
    (lambda d: d.assign(selected_weight=.75), "candidate bank"),
    (lambda d: d.assign(policy_position_fraction=.5), "Position-only"),
    (lambda d: d.assign(expert_ready=False), "Unavailable expert"),
    (lambda d: d.assign(selected_weight=0.), "trigger, clip and selected weight"),
    (lambda d: d.assign(raw_residual_prediction=0.), "trigger, clip and selected weight"),
    (lambda d: d.assign(raw_residual_prediction=np.nan), "finite residual"),
    (lambda d: d.assign(expert_ready="False"), "explicit booleans"),
    (lambda d: d.assign(expert_available_at_utc=d.forecast_origin_utc.dt.tz_localize(None)), "UTC-aware"),
])
def test_invalid_candidate_rejected(mutation, message):
    config, original, panel, audit = inputs()
    decisions = fake_train(panel, runner.model_config(config, original, audit)).decisions
    with pytest.raises(ValueError, match=message):
        runner.candidate_panel(panel, mutation(decisions), config)


def test_price_accuracy_does_not_require_reference_and_is_paired_to_storm():
    config, original, panel, audit = inputs()
    panel.loc[0, "reference_price"] = np.nan
    panel.loc[1, "benchmark_forecast"] = np.nan
    candidate = runner.candidate_panel(panel, fake_train(panel, runner.model_config(config, original, audit)).decisions, config)
    metrics, daily = runner.forecast_statistics(panel, candidate, config, base.engine_config(original))
    assert metrics.loc[metrics.subset.eq("all"), "hours"].eq(len(panel)-1).all()
    assert len(daily) == 365*3*2


def test_seals_and_result_tampering(prepared, tmp_path, monkeypatch):
    snapshot, _, _ = prepared
    monkeypatch.setattr(runner, "_seals", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="Code changed"):
        runner.evaluate(snapshot, root=tmp_path)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.read_snapshot(snapshot, root=tmp_path)


@pytest.mark.parametrize("key, value", [("minimum_training_days", 89), ("minimum_training_days", True),
                                       ("training_window_days", 366), ("production_modified", True),
                                       ("order_execution_enabled", True), ("candidate_model", "autonomous")])
def test_invalid_configuration(key, value):
    config = inputs()[0]
    config[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_protected_economic_costs_and_input_features():
    config, original, _, audit = inputs()
    config["expert"]["transaction_cost_eur_mwh"] = 0.
    with pytest.raises(ValueError, match="cannot be overridden"):
        runner.model_config(config, original, audit)


def test_launcher_dry_run_from_other_directory(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell unavailable")
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "PriceExpert.ps1"),
                             "-Action", "Run", "-DryRun"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "run_price_expert.py" in result.stdout and "aucun calcul" in result.stdout
    assert not list(tmp_path.iterdir())


def test_cannot_override_expert_recipe_through_governance_section():
    config, original, _, audit = inputs()
    config["governance"]["candidate_weights"] = [0., 1.]
    with pytest.raises(ValueError, match="collisions"):
        runner.model_config(config, original, audit)
    config["governance"].pop("candidate_weights")
    config["governance"]["unknown_model_setting"] = 1
    with pytest.raises(ValueError, match="dedicated sections"):
        runner.model_config(config, original, audit)
