"""Isolated 730/365-day runner tests with synthetic, frozen data sources."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from marginal_cost_expert import runner


def _physical_config():
    return {"demand_basis": "residual", "parameters": {
        "ccgt_efficiency": 0.5, "ocgt_efficiency": 0.25,
        "ccgt_vom_eur_mwh": 0.0, "ocgt_vom_eur_mwh": 0.0,
        "must_run_bid_eur_mwh": -20.0, "scarcity_price_eur_mwh": 4000.0,
    }, "scenarios": [
        {"id": "central", "hypothesis": "Synthetic physical baseline."},
        {"id": "premium", "hypothesis": "Predeclared modest bid premium.",
         "thermal_bid_premium_eur_mwh": 10.0},
    ]}


def _features(index, zone="FR"):
    return pd.DataFrame({"delivery_start_utc": index, "zone": zone, "demand_mw": 100.0,
                         "must_run_mw": 0.0, "ccgt_available_mw": 200.0, "ocgt_available_mw": 100.0,
                         "ttf_eur_mwh_th": 20.0, "eua_eur_tco2": 0.0, "demand_basis": "residual"})


def _experiment(tmp_path, monkeypatch, *, zones=("FR",), missing=()):
    root = tmp_path / "independent project"
    root.mkdir()
    config = {
        "schema_version": 1, "activate": False,
        "evaluation": {"start_day": "2025-09-10", "end_day": "2026-09-09",
                       "training_days": 365, "timezone": "Europe/Paris"},
        "output_root": "runs/experiments/marginal_test",
        "sources_config": "config/sources.yaml", "expert": _physical_config(),
        "risk": {"tight_capacity_margin_fraction": 3.0, "surplus_curtailment_mw": 1000.0},
        "guard": {"timezone": "Europe/Paris"},
        "targets": {zone: f"inputs/{zone}_target.csv" for zone in zones},
        "references": {zone: {"path": f"inputs/{zone}_reference.html", "label": "Fixed incumbent"} for zone in zones},
    }
    (root / "config").mkdir()
    (root / "config/sources.yaml").write_text("schema_version: 1\ndemand_basis: residual\n", encoding="utf-8")
    (root / "inputs").mkdir()
    for zone in zones:
        (root / config["targets"][zone]).write_text("Synthetic target source identity", encoding="utf-8")
        (root / config["references"][zone]["path"]).write_text("Synthetic frozen reference identity", encoding="utf-8")
    # Mimic existing source/config/model cache identities without touching the real project.
    preserved = [root / "Forecast.ps1", root / "run_complete_forecast.py", root / "config/kalman_operational.yaml",
                 root / "runs/live/existing/forecast.csv", root / "runs/cache/kalman/immutable_key.json"]
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing production bytes; never mutate")
    data_state = {"actual": 43.0, "reference_missing_hour": False, "target_missing_hour": False}
    support = runner.physical_index("2024-09-10", "2026-09-09", "Europe/Paris")
    evaluation = runner.physical_index("2025-09-10", "2026-09-09", "Europe/Paris")

    def inputs(source_config, *, project_root, start_day, end_day, zones):
        assert project_root == root.resolve()
        assert (start_day, end_day) == ("2024-09-10", "2026-09-09")
        zone = zones[0]
        if zone in missing:
            raise runner.MarginalCostDataError("Explicit synthetic missing expert source")
        return _features(support, zone), {"production_modified": False, "demand_basis": "residual"}

    def target(path, zone):
        frame = pd.DataFrame({"timestamp": support, "zone": zone, "actual": data_state["actual"]})
        return frame.iloc[1:] if data_state["target_missing_hour"] else frame

    def comparator(path, *, zone, label, timezone):
        frame = pd.DataFrame({"timestamp": evaluation, "zone": zone, "actual": 43.0, "base": 70.0, "storm": 45.0})
        if data_state["reference_missing_hour"]:
            frame = frame.iloc[1:]
        return frame, {"path": str(path), "sha256": runner.digest_file(path), "label": label}

    import marginal_cost_expert.reporting as reporting

    def render(frame, metrics, paired_metrics, daily, audit, output):
        output.write_text("<!doctype html><title>Synthetic isolated report</title>", encoding="utf-8")
        return output

    monkeypatch.setattr(runner, "load_zonal_inputs", inputs)
    monkeypatch.setattr(runner, "read_target", target)
    monkeypatch.setattr(runner, "read_report_comparator", comparator)
    monkeypatch.setattr(reporting, "render_report", render)
    return root, config, data_state, preserved, support, evaluation


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_end_to_end_730_day_physical_bank_365_day_evaluation_and_missing_expert_preserve_production(tmp_path, monkeypatch):
    root, config, state, protected, support, evaluation = _experiment(tmp_path, monkeypatch, zones=("FR", "DE"), missing=("DE",))
    original = {path: path.read_bytes() for path in protected}
    source_files = {path: path.read_bytes() for path in (root / "inputs").iterdir()}
    snapshot = runner.prepare(config, project_root=root)
    prepared = json.loads((snapshot / "audit.json").read_text(encoding="utf-8"))
    assert prepared["evaluation_days"] == prepared["training_window_days"] == 365
    assert "DE" in prepared["unavailable_experts"]
    assert len(pd.read_parquet(snapshot / "features.parquet")) == len(support)
    assert len(pd.read_parquet(snapshot / "references.parquet")) == 2 * len(evaluation)
    assert len(pd.Index(support.tz_convert("Europe/Paris").date).unique()) == 730

    assert runner.backtest(snapshot, project_root=root) == snapshot
    predictions = pd.read_parquet(snapshot / "predictions.parquet")
    assert len(predictions) == 2 * len(evaluation)
    assert predictions.groupby("zone").size().to_dict() == {"DE": len(evaluation), "FR": len(evaluation)}
    unavailable = predictions.loc[predictions.zone.eq("DE")]
    assert unavailable.expert.isna().all()
    assert unavailable.weight.eq(0.0).all()
    np.testing.assert_array_equal(unavailable.guarded, unavailable.base)
    assert unavailable["mode"].eq("unavailable").all()

    available = predictions.loc[predictions.zone.eq("FR")].sort_values("timestamp")
    days = available.timestamp.dt.tz_convert("Europe/Paris").dt.date
    warmup_end = (pd.Timestamp("2025-09-10") + pd.Timedelta(days=60)).date()
    assert available.loc[days < warmup_end, "weight"].eq(0.0).all()
    assert available.weight.gt(0.0).any()
    assert available.expert.eq(40.0).all()
    assert available["mode"].eq("zonal_degraded").all()
    assert not available.observed_marginal_unit_identified.any()
    selections = pd.read_parquet(snapshot / "rolling_selection.parquet")
    assert len(selections) == 365
    assert selections.training_days.eq(365).all()
    assert (pd.to_datetime(selections.training_end) < pd.to_datetime(selections.day)).all()
    assert not selections.evaluation_label_used.any()

    metrics = pd.read_parquet(snapshot / "metrics.parquet")
    assert set(metrics.zone) == {"FR", "DE"}
    assert {path: path.read_bytes() for path in protected} == original
    assert {path: path.read_bytes() for path in source_files} == source_files
    audit = json.loads((snapshot / "result_audit.json").read_text(encoding="utf-8"))
    assert audit["activation_performed"] is False
    assert audit["production_modified"] is False
    assert audit["production_pit_evidence"] is False
    assert audit["annual_non_regression_guaranteed"] is False

    # Report is a frozen-result reader, not a hidden training/replay operation.
    monkeypatch.setattr(runner, "physical_candidates", lambda *args, **kwargs: pytest.fail("Report must not call physical engine"))
    monkeypatch.setattr(runner, "walkforward_guard", lambda *args, **kwargs: pytest.fail("Report must not recalibrate policy"))
    monkeypatch.setattr(runner, "load_zonal_inputs", lambda *args, **kwargs: pytest.fail("Report must not load source data"))
    before = {name: _sha(snapshot / name) for name in audit["result_files"]}
    assert runner.report(snapshot, project_root=root).is_file()
    assert {name: _sha(snapshot / name) for name in before} == before
    assert {path: path.read_bytes() for path in protected} == original
    with pytest.raises(ValueError, match="already completed"):
        runner.backtest(snapshot, project_root=root)


@pytest.mark.parametrize("missing", ["target_missing_hour", "reference_missing_hour"])
def test_prepare_refuses_incomplete_730_labels_or_365_fixed_reference_before_publication(tmp_path, monkeypatch, missing):
    root, config, state, _, _, _ = _experiment(tmp_path, monkeypatch)
    state[missing] = True
    with pytest.raises(ValueError, match="cover"):
        runner.prepare(config, project_root=root)
    assert not (root / config["output_root"] / "snapshots").exists()


@pytest.mark.parametrize("change", ["training_days", "evaluation_length", "activate"])
def test_prepare_direct_api_enforces_strict_protocol_before_source_calls(tmp_path, monkeypatch, change):
    root, config, _, _, _, _ = _experiment(tmp_path, monkeypatch)
    if change == "training_days":
        config["evaluation"]["training_days"] = 364
    elif change == "evaluation_length":
        config["evaluation"]["start_day"] = "2025-09-11"
    else:
        config["activate"] = True
    monkeypatch.setattr(runner, "load_zonal_inputs", lambda *args, **kwargs: pytest.fail("Invalid protocol must not inspect sources"))
    with pytest.raises(ValueError):
        runner.prepare(config, project_root=root)


def _fake_completed(root: Path):
    output = root / "runs/experiments/test/snapshots/example"
    output.mkdir(parents=True)
    names = ("predictions.parquet", "metrics.parquet", "paired_metrics.parquet", "daily_metrics.parquet", "policies.json")
    for name in names:
        if name.endswith("parquet"):
            pd.DataFrame({"placeholder": [1]}).to_parquet(output / name, index=False)
        else:
            (output / name).write_text("{}", encoding="utf-8")
    audit = {"result_files": {name: _sha(output / name) for name in names}}
    (output / "result_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return output, audit


@pytest.mark.parametrize("change", ["payload", "missing_checksum", "path_escape"])
def test_report_refuses_corrupted_or_incomplete_result_seal_before_rendering(tmp_path, monkeypatch, change):
    import marginal_cost_expert.reporting as reporting
    root = tmp_path / "project"
    snapshot, audit = _fake_completed(root)
    if change == "payload":
        pd.DataFrame({"placeholder": [2]}).to_parquet(snapshot / "predictions.parquet", index=False)
    elif change == "missing_checksum":
        del audit["result_files"]["predictions.parquet"]
    else:
        audit["result_files"]["../escaped.parquet"] = "0" * 64
    (snapshot / "result_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(reporting, "render_report", lambda *args, **kwargs: pytest.fail("Never render an unsealed result"))
    with pytest.raises(ValueError):
        runner.report(snapshot, project_root=root)


def test_snapshot_input_checksum_is_checked_before_any_physical_engine(tmp_path, monkeypatch):
    root, config, _, _, _, _ = _experiment(tmp_path, monkeypatch)
    snapshot = runner.prepare(config, project_root=root)
    features = pd.read_parquet(snapshot / "features.parquet")
    features.loc[0, "ttf_eur_mwh_th"] += 10.0
    features.to_parquet(snapshot / "features.parquet", index=False)
    monkeypatch.setattr(runner, "physical_candidates", lambda *args, **kwargs: pytest.fail("Invalid source seal"))
    with pytest.raises(ValueError, match="checksum"):
        runner.backtest(snapshot, project_root=root)


def test_all_experts_missing_still_reports_every_reference_hour_without_engine(tmp_path, monkeypatch):
    root, config, _, protected, _, evaluation = _experiment(tmp_path, monkeypatch, zones=("FR",), missing=("FR",))
    before = {path: path.read_bytes() for path in protected}
    snapshot = runner.prepare(config, project_root=root)
    monkeypatch.setattr(runner, "physical_candidates", lambda *args, **kwargs: pytest.fail("No engine with missing expert inputs"))
    runner.backtest(snapshot, project_root=root)
    predictions = pd.read_parquet(snapshot / "predictions.parquet")
    assert len(predictions) == len(evaluation)
    assert predictions.expert.isna().all()
    assert predictions.weight.eq(0.0).all()
    np.testing.assert_array_equal(predictions.guarded, predictions.base)
    paired = pd.read_parquet(snapshot / "paired_metrics.parquet")
    assert set(paired.model) == {"base", "guarded", "storm"}
    assert paired.hours.eq(len(evaluation)).all()
    audit = json.loads((snapshot / "result_audit.json").read_text(encoding="utf-8"))
    assert audit["intervention_diagnostics"]["FR"]["zero_intervention_is_not_predictive_gain"] is True
    assert {path: path.read_bytes() for path in protected} == before


def test_physical_cache_reuses_label_independent_months_and_invalidates_only_changed_inputs(tmp_path, monkeypatch):
    config = _physical_config()
    panel = _features(pd.to_datetime(["2026-01-15T00:00Z", "2026-02-15T00:00Z"]))
    calls = []
    actual_engine = runner.MarginalCostExpert

    class SpyEngine(actual_engine):
        def predict_candidates(self, features, network=None, **kwargs):
            calls.append(features.delivery_start_utc.dt.month.unique().tolist())
            return super().predict_candidates(features, network=network, **kwargs)

    monkeypatch.setattr(runner, "MarginalCostExpert", SpyEngine)
    cache = tmp_path / "isolated_cache"
    first = runner.physical_candidates(panel, config, cache=cache)
    assert calls == [[1], [2]]
    initial_keys = {path.name: path.read_bytes() for path in cache.glob("*.json")}
    labels = pd.DataFrame({"actual": [50.0, 60.0]})
    labels["actual"] += 1000.0  # Deliberately absent from the physical API/key.
    second = runner.physical_candidates(panel.copy(), deepcopy(config), cache=cache)
    assert calls == [[1], [2]]
    pd.testing.assert_frame_equal(first, second)
    assert {path.name: path.read_bytes() for path in cache.glob("*.json")} == initial_keys
    altered = panel.copy()
    altered.loc[altered.delivery_start_utc.dt.month.eq(2), "ttf_eur_mwh_th"] += 1.0
    third = runner.physical_candidates(altered, config, cache=cache)
    assert calls == [[1], [2], [2]]
    assert third.loc[third.delivery_start_utc.dt.month.eq(1), "price_eur_mwh"].to_list() == first.loc[
        first.delivery_start_utc.dt.month.eq(1), "price_eur_mwh"].to_list()
    changed_recipe = deepcopy(config)
    changed_recipe["parameters"]["ccgt_efficiency"] = 0.55
    runner.physical_candidates(panel, changed_recipe, cache=cache)
    assert calls == [[1], [2], [2], [1], [2]]


def test_physical_cache_rejects_mismatched_demand_basis_instead_of_dropping_contract(tmp_path):
    panel = _features(pd.to_datetime(["2026-01-15T00:00Z"]))
    config = _physical_config()
    config["demand_basis"] = "gross"
    with pytest.raises(ValueError, match="basis"):
        runner.physical_candidates(panel, config, cache=tmp_path / "cache")


def test_physical_cache_recomputes_corrupted_payload_and_invalidates_runtime_versions(tmp_path, monkeypatch):
    import importlib.metadata
    config = _physical_config()
    panel = _features(pd.to_datetime(["2026-01-15T00:00Z"]))
    calls = []
    actual_engine = runner.MarginalCostExpert

    class SpyEngine(actual_engine):
        def predict_candidates(self, features, network=None, **kwargs):
            calls.append(True)
            return super().predict_candidates(features, network=network, **kwargs)

    monkeypatch.setattr(runner, "MarginalCostExpert", SpyEngine)
    cache = tmp_path / "cache"
    before = runner.physical_candidates(panel, config, cache=cache)
    payload = next(cache.glob("*.parquet"))
    modified = pd.read_parquet(payload)
    modified["price_eur_mwh"] += 999.0
    modified.to_parquet(payload, index=False)
    after = runner.physical_candidates(panel, config, cache=cache)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(before, after)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: original_version(name) + ".changed")
    runner.physical_candidates(panel, config, cache=cache)
    assert len(calls) == 3
