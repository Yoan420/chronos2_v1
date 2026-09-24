from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_value import extreme_runner, price_data, runner as base


ROOT = Path(__file__).resolve().parents[1]


def _reseal(snapshot):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["snapshot_files"] = {name: base.digest(snapshot / name) for name in extreme_runner.INPUTS}
    base._json(snapshot / "manifest.json", manifest)
    results = json.loads((snapshot / "results_manifest.json").read_text())
    results.update({name: manifest[name] for name in price_data._BINDING_KEYS})
    results["result_files"] = {name: base.digest(snapshot / name) for name in extreme_runner.RESULTS}
    base._json(snapshot / "results_manifest.json", results)


@pytest.fixture
def source(tmp_path):
    snapshot = tmp_path / "runs/experiments/source/snapshots/frozen"
    snapshot.mkdir(parents=True)
    baseline = base.load_config(ROOT / "config/economic_value.yaml")
    baseline.update(zones=["FR", "DE"], models=["nuclear_kalman"])
    config = extreme_runner.load_config(ROOT / "config/economic_extreme_policy.yaml")
    config["zones"] = baseline["zones"]
    times = pd.date_range("2025-09-11", "2026-09-11", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    days = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (days-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    label_time = (days-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    one = pd.DataFrame({
        "timestamp_utc": times, "model": "nuclear_kalman", "forecast_origin_utc": origin,
        "label_available_at_utc": label_time, "label_publication_time_assumed": True,
        "forecast": 80., "q10": 60., "q90": 100., "actual": 90., "reference_price": 50.,
        "benchmark_forecast": 70., "reference_available_at_utc": origin-pd.Timedelta(hours=14),
        "duration_hours": 1., "forecast_eligible": True, "reference_eligible": True,
        "sample": "evaluation", "feature_eligible": True, "feature_pit_certified": False,
    })
    for name in sorted(price_data.ALLOWED_FEATURES):
        one[name] = 0. if name.endswith("_missing") else 1.
    panel = pd.concat([one.assign(zone=zone) for zone in baseline["zones"]], ignore_index=True)
    metrics = pd.DataFrame([
        {"zone": zone, "model": "nuclear_kalman", "strategy": strategy,
         "pnl_net_eur": 1000., "pnl_gross_eur": 1100., "trading_cost_eur": 100.,
         "absolute_energy_mwh": 100., "eligible_hours": 8760., "active_hours": 100., "max_drawdown_eur": 10.}
        for zone in [*baseline["zones"], "PORTFOLIO"] for strategy in ("no_forecast", "benchmark", "model")
    ])
    feature_audit = {
        "feature_columns": sorted(price_data.ALLOWED_FEATURES),
        "required_core_feature_columns": sorted(price_data.REQUIRED_CORE),
        "storm_used_as_feature": False, "selected_cutoff_violations": 0,
        "source_baseline_metrics": json.loads(metrics.to_json(orient="records")),
        "source_hashes": [{"path": "original-pit-source", "sha256": "original-pit-checksum"}],
    }
    for name, value in (("config", config), ("baseline_config", baseline), ("feature_audit", feature_audit)):
        base._json(snapshot / f"{name}.json", value)
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    # Deliberately no baseline history: the loader must hash but never read it.
    pd.DataFrame({"fundamental_only": [1.]}).to_parquet(snapshot / "history.parquet", index=False)
    panel.assign(strategy="model").to_parquet(snapshot / "rows.parquet", index=False)
    metrics.to_parquet(snapshot / "metrics.parquet", index=False)
    for name in ("decisions", "folds", "governance", "daily", "breakdowns"):
        pd.DataFrame({"fixture": [1.]}).to_parquet(snapshot / f"{name}.parquet", index=False)
    base._json(snapshot / "policy_audit.json", {"diagnostic_only": True})
    manifest = {
        "kind": "extreme_economic_policy", **base._validate_panel(panel, baseline),
        "zones": config["zones"], "models": [config["baseline_model"], config["candidate_model"]],
        "source_code_sha256": {"old_source.py": "historical-code-not-current"},
    }
    base._json(snapshot / "manifest.json", manifest)
    base._json(snapshot / "results_manifest.json", {"status": "completed_hypothetical_diagnostic", "forecast_values_unchanged": True})
    _reseal(snapshot)
    settings = {
        "source_expert_snapshot": str(snapshot), "zones": baseline["zones"].copy(),
        "baseline_model": "nuclear_kalman", "candidate_model": "nuclear_kalman_extreme",
        "training_window_days": 365, "minimum_training_days": 90,
    }
    return tmp_path, snapshot, settings, panel, baseline


def test_exact_frozen_baseline_and_progressive_warmup(source, monkeypatch):
    root, snapshot, settings, original, original_config = source
    checksums = {p.name: base.digest(p) for p in snapshot.iterdir()}
    read = pd.read_parquet
    def no_history(path, *args, **kwargs):
        assert Path(path).name != "history.parquet", "No nuclear Kalman baseline exists in the older fundamental history."
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", no_history)
    panel, audit, baseline = price_data.load_price_inputs(root, settings)
    pd.testing.assert_frame_equal(panel, original)
    assert baseline == original_config
    assert baseline is not original_config
    assert audit["available_baseline_days"] == 365
    assert audit["pre_evaluation_baseline_days"] == 0
    assert audit["warmup_baseline_days"] == 90
    assert audit["first_possible_fit_day"] == "2025-12-10"
    assert audit["potential_post_warmup_days"] == 275
    assert audit["strict_365_training_available_in_evaluation"] is False
    assert audit["baseline_neural_oof_certified"] is False
    assert not set(audit["feature_columns"]) & price_data._FEATURE_METADATA
    assert audit["source_feature_audit"]["source_hashes"][0]["path"] == "original-pit-source"
    assert {p.name: base.digest(p) for p in snapshot.iterdir()} == checksums


def test_strict_365_keeps_all_baseline_days_without_invented_training(source):
    root, _, settings, original, _ = source
    settings["minimum_training_days"] = 365
    panel, audit, _ = price_data.load_price_inputs(root, settings)
    assert audit["potential_post_warmup_days"] == 0
    assert audit["first_possible_fit_day"] == "2026-09-11"
    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize("key,value", [
    ("zones", ["FR"]), ("zones", ["FR", "FR"]),
    ("minimum_training_days", 89), ("minimum_training_days", 366),
    ("minimum_training_days", True), ("training_window_days", 364),
    ("baseline_model", "nuclear_autonomous"), ("candidate_model", "nuclear_kalman_extreme_governed"),
])
def test_invalid_settings_or_country_reallocation(source, key, value):
    root, _, settings, _, _ = source
    settings[key] = value
    with pytest.raises(ValueError):
        price_data.load_price_inputs(root, settings)


def test_changed_input_checksum_is_refused(source):
    root, snapshot, settings, _, _ = source
    (snapshot / "feature_audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        price_data.load_price_inputs(root, settings)


def test_result_manifest_cannot_be_transplanted(source):
    root, snapshot, settings, _, _ = source
    result = json.loads((snapshot / "results_manifest.json").read_text())
    result["snapshot_files"]["panel.parquet"] = "different-input"
    base._json(snapshot / "results_manifest.json", result)
    with pytest.raises(ValueError, match="manifest mismatch"):
        price_data.load_price_inputs(root, settings)


def test_missing_result_file_is_refused(source):
    root, snapshot, settings, _, _ = source
    result = json.loads((snapshot / "results_manifest.json").read_text())
    result["result_files"].pop("governance.parquet")
    base._json(snapshot / "results_manifest.json", result)
    with pytest.raises(ValueError, match="Incomplete"):
        price_data.load_price_inputs(root, settings)


def test_resealed_results_must_reproduce_baseline_input_fields(source):
    root, snapshot, settings, _, _ = source
    rows = pd.read_parquet(snapshot / "rows.parquet")
    rows.loc[0, "q90"] = 1000.
    rows.to_parquet(snapshot / "rows.parquet", index=False)
    _reseal(snapshot)
    with pytest.raises(ValueError, match="baseline fields differ"):
        price_data.load_price_inputs(root, settings)


def test_resealed_result_metrics_must_match_original_eva(source):
    root, snapshot, settings, _, _ = source
    metrics = pd.read_parquet(snapshot / "metrics.parquet")
    metrics.loc[0, "pnl_net_eur"] = 1.e9
    metrics.to_parquet(snapshot / "metrics.parquet", index=False)
    _reseal(snapshot)
    with pytest.raises(ValueError, match="metric differs"):
        price_data.load_price_inputs(root, settings)


@pytest.mark.parametrize("name", ["feature_storm", "feature_actual", "feature_eligible"])
def test_source_audit_cannot_authorize_leaking_predictors(source, name):
    root, snapshot, settings, _, _ = source
    audit = json.loads((snapshot / "feature_audit.json").read_text())
    audit["feature_columns"].append(name)
    base._json(snapshot / "feature_audit.json", audit)
    _reseal(snapshot)
    with pytest.raises(ValueError, match="forbidden predictors"):
        price_data.load_price_inputs(root, settings)


@pytest.mark.parametrize("change,match", [
    ("missing_hour", "shortened"), ("missing_forecast", "Missing baseline"),
    ("label_time", "Label availability"), ("missing_flag", "missingness flag"),
    ("undeclared_feature", "explicit source allowlist"),
])
def test_invalid_panel_support_or_provenance(source, change, match):
    root, snapshot, settings, panel, _ = source
    panel = panel.copy()
    if change == "missing_hour":
        panel = panel.iloc[1:]
    elif change == "missing_forecast":
        panel.loc[0, "forecast"] = np.nan
    elif change == "label_time":
        panel.loc[0, "label_available_at_utc"] -= pd.Timedelta(days=1)
    elif change == "missing_flag":
        panel.loc[0, "feature_hour_sin_missing"] = 1.
    elif change == "undeclared_feature":
        panel["feature_unknown"] = 1.
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    _reseal(snapshot)
    with pytest.raises(ValueError, match=match):
        price_data.load_price_inputs(root, settings)


def test_optional_nan_is_preserved_without_invalidating_core(source):
    root, snapshot, settings, panel, _ = source
    panel = panel.copy()
    panel.loc[0, "feature_hour_sin"] = np.nan
    panel.loc[0, "feature_hour_sin_missing"] = 1.
    panel.to_parquet(snapshot / "panel.parquet", index=False)
    panel.assign(strategy="model").to_parquet(snapshot / "rows.parquet", index=False)
    _reseal(snapshot)
    result, _, _ = price_data.load_price_inputs(root, settings)
    assert np.isnan(result.loc[0, "feature_hour_sin"])
    assert result.loc[0, "feature_eligible"]


def test_concurrent_source_change_after_read_is_refused(source, monkeypatch):
    root, snapshot, settings, _, _ = source
    read = pd.read_parquet
    def changed(path, *args, **kwargs):
        result = read(path, *args, **kwargs)
        if Path(path).name == "metrics.parquet":
            with (snapshot / "feature_audit.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
        return result
    monkeypatch.setattr(pd, "read_parquet", changed)
    with pytest.raises(ValueError, match="changed during loading"):
        price_data.load_price_inputs(root, settings)
