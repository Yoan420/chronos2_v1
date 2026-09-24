from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from chronos2_exogenous.lora_finetune import (
    EVALUATION_COLUMNS, load_config, publish_evaluation_evidence,
    sha256_directory, train_lora, validate_panel,
)
from chronos2_exogenous.oof_residual import (
    _assert_final_bundle_matches_recipe, _fold_manifest_core, _fold_name,
    _json_sha256, _range, build_oof_plan,
)
from chronos2_exogenous.production import fit_oof_residual_corrector
from chronos2_exogenous.research_oof_evaluation import (
    DAILY_NAME, MANIFEST_NAME, OOF_COLUMNS, ResearchOofEvaluationError,
    _same_observations, evaluate_research_oof,
)
from test_chronos2_exogenous_lora_finetune import (
    _RecordingLoader, _fake_model_snapshot, _persist_panel, _write_config,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _physical_panel() -> pd.DataFrame:
    pieces = []
    for day in pd.date_range("2023-09-04", "2026-09-02", freq="D"):
        start = day.tz_localize("Europe/Paris")
        end = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
        origin = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
        timestamps = pd.date_range(start.tz_convert("UTC") - pd.Timedelta(hours=24),
                                   end.tz_convert("UTC"), inclusive="left", freq="h")
        pieces.append(pd.DataFrame({"timestamp": timestamps, "origin_timestamp": origin.tz_convert("UTC"),
                                    "item_id": "FR", "feature_available_at_utc": origin.tz_convert("UTC"),
                                    "target": 12., "wind_fcst": 20., "load_fcst": 50., "price_lag": 10.}))
    return pd.concat(pieces, ignore_index=True)


def _canonical_audit(config: object, panel: pd.DataFrame, target: Path) -> None:
    _persist_panel(config, panel)
    audit = json.loads(config.panel_audit_path.read_text())
    audit["target_sources"]["FR"] = {"source_path": str(target), "source_sha256": _sha(target)}
    audit["target_contracts"]["FR"] = {"series": "power.price.da.fr.canonical", "cache_path": str(target)}
    audit["target_contracts"]["FR"]["naive_timezone"] = "UTC"
    _write(config.panel_audit_path, audit)


@pytest.fixture
def experiment(tmp_path: Path) -> dict:
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["evaluation_role"] = "diagnostic_only"
    payload["data"].update(training_window_days=365, validation_days=30, evaluation_days=365)
    payload["model"]["model_id"] = str(_fake_model_snapshot(tmp_path))
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(config_path)
    panel = _physical_panel()
    target = tmp_path / "canonical_target.csv"
    panel[["timestamp", "target"]].drop_duplicates("timestamp").rename(
        columns={"target": "value"}).to_csv(target, index=False)
    origins = pd.DatetimeIndex(panel.origin_timestamp.drop_duplicates()).sort_values()
    _canonical_audit(config, panel.loc[panel.origin_timestamp.isin(origins[-730:])], target)
    run = train_lora(config, pipeline_loader=_RecordingLoader())
    manifest = json.loads((run / "experiment_manifest.json").read_text())
    cal_config = replace(config, panel_path=tmp_path / "inputs/calibration.parquet",
                         panel_audit_path=tmp_path / "inputs/calibration.parquet.audit.json",
                         training_window_days=730)
    _canonical_audit(cal_config, panel, target)
    validated, _, panel_audit = validate_panel(pd.read_parquet(cal_config.panel_path), cal_config)
    plan = build_oof_plan(origins, timezone_name=config.timezone, cutoff_local_time="08:00", validation_days=30)
    _, _, deployment_sha, recipe = _assert_final_bundle_matches_recipe(config, run, manifest)
    recipe_sha = _json_sha256(recipe)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    pieces = []
    declarations = []
    for fold in plan.folds:
        directory = calibration / "folds" / _fold_name(fold, config.timezone)
        checkpoint = directory / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter.safetensors").write_bytes(f"fold-{fold.index}".encode())
        rows = []
        for origin in fold.prediction_origins:
            day = origin.tz_convert(config.timezone).date() + timedelta(days=1)
            hours = pd.date_range(pd.Timestamp(day, tz=config.timezone).tz_convert("UTC"),
                                  pd.Timestamp(day + timedelta(days=1), tz=config.timezone).tz_convert("UTC"),
                                  inclusive="left", freq="h")
            rows.append(pd.DataFrame({"delivery_start_utc": hours, "forecast_origin_utc": origin,
                                      "actual": 12., "candidate_q10": 5., "candidate_q50": 10., "candidate_q90": 15.}))
        predictions = pd.concat(rows, ignore_index=True).loc[:, OOF_COLUMNS]
        prediction_path = directory / "oof_predictions.csv.gz"
        predictions.to_csv(prediction_path, index=False, compression="gzip")
        core = _fold_manifest_core(fold, recipe=recipe, recipe_sha256=recipe_sha,
                                   checkpoint_sha256=sha256_directory(checkpoint))
        core["predictions_sha256"] = _sha(prediction_path)
        _write(directory / "fold_manifest.json", core)
        declarations.append({k: core[k] for k in ("fold_index", "fit_origins", "train_origins", "validation_origins",
                                                   "prediction_origins", "checkpoint_sha256", "predictions_sha256",
                                                   "refit_uses_only_strictly_prior_days")})
        pieces.append(predictions)
    oof_path = calibration / "oof_predictions_365.csv.gz"
    pd.concat(pieces, ignore_index=True).to_csv(oof_path, index=False, compression="gzip")
    checkpoint_set = _json_sha256([{k: row[k] for k in ("fold_index", "checkpoint_sha256")} for row in declarations])
    audit = {"schema_version": 2, "purpose": "chronos2_exogenous_blocked_prequential_oof",
             "fit_protocol": "blocked_prequential_oof_rolling365", "candidate_model": "chronos2_exogenous",
             "candidate_checkpoint_sha256": deployment_sha,
             "candidate_checkpoint_role": "deployment_identity_anchor_not_oof_predictor",
             "deployment_checkpoint_used_for_oof": False, "fold_checkpoints_are_origin_specific": True,
             "fold_checkpoint_set_sha256": checkpoint_set, "fold_candidate_recipe_sha256": recipe_sha,
             "fold_count": len(plan.folds), "folds": declarations, "fold_lookback_days": 365,
             "training_days": 365, "training_start_day": "2024-09-03", "training_end_day": "2025-09-02",
             "holdout_start_day": "2025-09-03", "predictions_sha256": _sha(oof_path),
             "refit_uses_only_strictly_prior_days": True, "same_day_actual_excluded_from_fit": True,
             "future_actuals_used_as_features": False, "holdout_used_for_fit": False,
             "holdout_targets_used_for_fit": False, "selection_frozen_before_oof": True,
             "panel_sha256": _sha(cal_config.panel_path), "panel_audit_sha256": _sha(cal_config.panel_audit_path),
             "production_pit_evidence": False, "promotion_eligible": False}
    audit_path = calibration / "oof_predictions_365.csv.gz.audit.json"
    _write(audit_path, audit)
    corrector_path = calibration / "residual_corrector.json"
    fit_oof_residual_corrector(oof_predictions_path=oof_path, oof_audit_path=audit_path,
                               holdout_start_day="2025-09-03", output_path=corrector_path)
    complete = {"schema_version": 1, "purpose": "chronos2_exogenous_lora_residual_calibration",
                "status": "complete", "item_id": "FR", "target_column": "target", "block_days": 30,
                "training_days": 365, "oof_days": 365, "holdout_days": 365, "required_origins": 1095,
                "fold_count": len(plan.folds), "recipe_sha256": recipe_sha,
                "deployment_checkpoint_sha256": deployment_sha,
                "panel_sha256": _sha(cal_config.panel_path), "panel_audit_sha256": _sha(cal_config.panel_audit_path),
                "calibration_origins": _range(plan.calibration_origins), "holdout_origins": _range(plan.holdout_origins),
                "production_pit_evidence": False, "holdout_targets_used_for_fit": False, "promotion_eligible": False,
                "predictions_relative_path": oof_path.name, "predictions_sha256": _sha(oof_path),
                "oof_audit_relative_path": audit_path.name, "oof_audit_sha256": _sha(audit_path),
                "corrector_relative_path": corrector_path.name, "corrector_sha256": _sha(corrector_path)}
    _write(calibration / "calibration_manifest.json", complete)
    rows = []
    for origin in plan.holdout_origins:
        day = origin.tz_convert(config.timezone).date() + timedelta(days=1)
        hours = pd.date_range(pd.Timestamp(day, tz=config.timezone).tz_convert("UTC"),
                              pd.Timestamp(day + timedelta(days=1), tz=config.timezone).tz_convert("UTC"),
                              inclusive="left", freq="h")
        rows.append(pd.DataFrame({"delivery_start_utc": hours, "forecast_origin_utc": origin,
                                  "actual": 12., "baseline_q10": 5., "baseline_q50": 9., "baseline_q90": 15.,
                                  "candidate_q10": 5., "candidate_q50": 10., "candidate_q90": 15.}))
    publish_evaluation_evidence(run, pd.concat(rows, ignore_index=True).loc[:, EVALUATION_COLUMNS])
    return {"config_path": config_path, "run_directory": run,
            "calibration_directory": calibration, "calibration_panel_path": cal_config.panel_path,
            "calibration_panel_audit_path": cal_config.panel_audit_path,
            "output_directory": tmp_path / "research-result", "zone": "FR"}


def test_diagnostic_false_pit_gets_real_oof_report_without_mutating_inputs(experiment: dict) -> None:
    root = experiment["run_directory"].parent.parent
    hashes = {path: _sha(path) for path in root.rglob("*") if path.is_file()}
    result = evaluate_research_oof(**experiment)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_role"] == "diagnostic_only"
    assert manifest["genuine_oof_days"] == 365 and manifest["fold_count"] == 13
    assert manifest["production_pit_evidence"] is False
    assert manifest["production_pipeline_evidence"] is False
    assert manifest["promotion_eligible"] is False and manifest["shadow_eligible"] is False
    assert result.metrics["physical_hours"] == 8760
    assert result.metrics["models"]["LoRA brut"]["mae_eur_mwh"] == 2.
    assert result.metrics["models"]["LoRA brut"]["daily_mean_mae_eur_mwh"] == 2.
    assert result.metrics["models"]["LoRA + correcteur OOF"]["mae_eur_mwh"] < .001
    assert "Mode nuit / jour" in result.report_path.read_text(encoding="utf-8")
    assert "Statistics — prix moyens" in result.report_path.read_text(encoding="utf-8")
    daily = pd.read_csv(result.directory / DAILY_NAME)
    assert len(daily) == 365
    assert set(daily.hours) == {23, 24, 25}
    assert daily.baseline_daily_mean_abs_error_eur_mwh.eq(2.).all()
    assert hashes == {path: _sha(path) for path in hashes}
    for name, reference in manifest["outputs"].items():
        assert _sha(result.directory / name) == reference["sha256"]
    with pytest.raises(ResearchOofEvaluationError, match="existante"):
        evaluate_research_oof(**experiment)


@pytest.mark.parametrize("target", ["checkpoint", "fold_predictions", "fold_dates", "corrector"])
def test_tampered_oof_chain_is_rejected_before_publication(experiment: dict, target: str) -> None:
    calibration = experiment["calibration_directory"]
    fold = sorted((calibration / "folds").iterdir())[0]
    if target == "checkpoint":
        (fold / "checkpoint/adapter.safetensors").write_bytes(b"tampered")
    elif target == "fold_predictions":
        frame = pd.read_csv(fold / "oof_predictions.csv.gz")
        frame.loc[0, "actual"] += 1
        frame.to_csv(fold / "oof_predictions.csv.gz", index=False, compression="gzip")
    elif target == "fold_dates":
        path = fold / "fold_manifest.json"
        value = json.loads(path.read_text())
        value["fit_origins"]["last_utc"] = value["prediction_origins"]["first_utc"]
        _write(path, value)
    else:
        path = calibration / "residual_corrector.json"
        value = json.loads(path.read_text())
        value["coefficients"][0] = 5.
        _write(path, value)
        complete = calibration / "calibration_manifest.json"
        value = json.loads(complete.read_text())
        value["corrector_sha256"] = _sha(path)
        _write(complete, value)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_research_oof(**experiment)
    assert not experiment["output_directory"].exists()


def test_observation_roundtrip_is_absolute_only_with_exact_dates() -> None:
    left = pd.DataFrame({"delivery_start_utc": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                         "forecast_origin_utc": pd.to_datetime(["2025-12-31T07:00:00Z"]),
                         "actual": [12.]})
    right = left.copy()
    right.loc[0, "actual"] += 5e-13
    _same_observations(left, right, label="roundtrip")
    right.loc[0, "actual"] = 12.000001
    with pytest.raises(ResearchOofEvaluationError, match="prix observes"):
        _same_observations(left, right, label="difference")
    left.loc[0, "actual"], right.loc[0, "actual"] = 1e6, 1e6 + 1e-7
    with pytest.raises(ResearchOofEvaluationError, match="prix observes"):
        _same_observations(left, right, label="pas de rtol proportionnel au prix")
    right = left.copy()
    right.loc[0, "forecast_origin_utc"] += pd.Timedelta(nanoseconds=1)
    with pytest.raises(ResearchOofEvaluationError, match="forecast_origin_utc"):
        _same_observations(left, right, label="origines exactes")


def test_research_references_require_exact_observations(experiment: dict, tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    shutil.copytree(experiment["run_directory"], reference)
    result = evaluate_research_oof(**experiment, reference_runs={"Ancien LoRA rang 16": reference})
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["references"][0]["input_contracts_declared_identical"] is False
    assert "Ancien LoRA rang 16" in result.metrics["models"]
    experiment["output_directory"] = tmp_path / "bad-reference-result"
    predictions = reference / "evaluation_predictions.csv.gz"
    frame = pd.read_csv(predictions)
    frame.loc[0, "actual"] += 1
    frame.to_csv(predictions, index=False, compression="gzip")
    path = reference / "experiment_manifest.json"
    value = json.loads(path.read_text())
    value["evaluation_evidence"]["sha256"] = _sha(predictions)
    _write(path, value)
    with pytest.raises(ResearchOofEvaluationError, match="prix observes"):
        evaluate_research_oof(**experiment, reference_runs={"Ancien LoRA": reference})
    assert not experiment["output_directory"].exists()
    source_hashes = {p: _sha(p) for p in reference.rglob("*") if p.is_file()}
    result = evaluate_research_oof(**experiment, reference_runs={"Ancien LoRA": reference},
                                   reference_actual_policy="canonical_recompute")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    audit = manifest["references"][0]
    assert audit["actual_policy"] == "canonical_recompute"
    assert audit["revised_actual_hours"] == 1
    assert audit["maximum_absolute_actual_revision_eur_mwh"] == 1.
    assert audit["original_observation_metrics"]["mae_eur_mwh"] > audit["common_observation_metrics"]["mae_eur_mwh"]
    predictions = pd.read_csv(result.predictions_path)
    assert predictions.loc[0, "actual"] == 12.
    assert predictions.loc[0, "reference_1_original_actual"] == 13.
    assert source_hashes == {p: _sha(p) for p in source_hashes}
    assert "recalcul explicite sur la cible canonique commune" in result.report_path.read_text(encoding="utf-8")
    value["target_contracts"]["FR"]["series"] = "different.series"
    _write(path, value)
    experiment["output_directory"] = tmp_path / "bad-contract-result"
    with pytest.raises(ResearchOofEvaluationError, match="identite cible canonique"):
        evaluate_research_oof(**experiment, reference_runs={"Ancien LoRA": reference},
                              reference_actual_policy="canonical_recompute")
