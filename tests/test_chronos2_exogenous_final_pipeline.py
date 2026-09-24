from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.final_pipeline import (
    ACTUAL_PAIRING_ATOL_EUR_MWH,
    FINAL_AUDIT_NAME,
    FINAL_DIRECTORY_NAME,
    FINAL_EVIDENCE_NAME,
    FINAL_MANIFEST_NAME,
    FINAL_METRICS_NAME,
    FINAL_REPORT_NAME,
    FinalPipelineEvaluationError,
    run_final_pipeline_evaluation,
)
from chronos2_exogenous.lora_finetune import (
    EVALUATION_COLUMNS,
    publish_evaluation_evidence,
    sha256_directory,
)
from chronos2_exogenous.production import (
    BASE_MODEL,
    OUTPUT_MODEL,
    _validate_final_pipeline_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timeline(
    start_day: str = "2025-01-01",
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    deliveries: list[pd.Timestamp] = []
    origins: list[pd.Timestamp] = []
    for day in pd.date_range(start_day, periods=365, freq="D"):
        local_start = day.tz_localize("Europe/Paris")
        local_end = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
        horizon = pd.date_range(
            local_start.tz_convert("UTC"),
            local_end.tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        origin = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(
            "Europe/Paris"
        ).tz_convert("UTC")
        deliveries.extend(horizon)
        origins.extend([origin] * len(horizon))
    return pd.DatetimeIndex(deliveries), pd.DatetimeIndex(origins)


def _make_bundle(
    tmp_path: Path, *, start_day: str = "2025-01-01"
) -> tuple[Path, pd.DataFrame]:
    run = tmp_path / "lora-run"
    checkpoint = run / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    schema = {
        "format_version": 1,
        "timestamp_column": "timestamp",
        "origin_column": "forecast_origin_utc",
        "item_column": "item_id",
        "feature_available_at_column": "feature_available_at_utc",
        "target_columns": ["target"],
        "known_future_covariates": [],
        "past_only_covariates": [],
        "timezone": "Europe/Paris",
        "cutoff_local_time": "08:00",
        "frequency": "h",
        "context_length": 2048,
        "prediction_length": 24,
    }
    schema_path = run / "schema.json"
    _write_json(schema_path, schema)
    delivery, origins = _timeline(start_day)
    distinct_origins = origins.unique().sort_values()
    manifest = {
        "format_version": 1,
        "experiment_id": "lora-final-test-v1",
        "evaluation_role": "primary_predeclared",
        "model_id": "amazon/chronos-2",
        "finetune_mode": "lora",
        "training_window_days": 365,
        "evaluation_days": 365,
        "cutoff_local_time": "08:00",
        "candidate_frozen_before_evaluation": True,
        "feature_selection_frozen_before_evaluation": True,
        "actual_future_used_as_input": False,
        "storm_used_for_input": False,
        "storm_used_for_selection": False,
        "mkonline_used_for_input": False,
        "pit_audit_passed": True,
        "production_pit_evidence": False,
        "production_pipeline_evidence": False,
        "checkpoint_sha256": sha256_directory(checkpoint),
        "schema_sha256": _sha256(schema_path),
        "splits": {
            "evaluation_holdout": {
                "count": 365,
                "first_utc": distinct_origins[0].isoformat(),
                "last_utc": distinct_origins[-1].isoformat(),
            }
        },
    }
    _write_json(run / "experiment_manifest.json", manifest)
    phase = np.arange(len(delivery), dtype=float)
    actual = 50.0 + 12.0 * np.sin(phase * 2.0 * np.pi / (24.0 * 30.0))
    raw = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origins,
            "actual": actual,
            "baseline_q10": actual - 7.0,
            "baseline_q50": actual + 3.0,
            "baseline_q90": actual + 13.0,
            "candidate_q10": actual - 6.0,
            "candidate_q50": actual + 2.0,
            "candidate_q90": actual + 10.0,
        },
        columns=EVALUATION_COLUMNS,
    )
    publish_evaluation_evidence(run, raw)
    return run, raw


def _make_corrector(
    tmp_path: Path, run: Path
) -> tuple[Path, Path]:
    experiment = json.loads((run / "experiment_manifest.json").read_text())
    raw = pd.read_csv(run / "evaluation_predictions.csv.gz")
    first_delivery = pd.to_datetime(
        raw["delivery_start_utc"], utc=True
    ).dt.tz_convert("Europe/Paris").dt.date.min()
    training_end = first_delivery - pd.Timedelta(days=1)
    training_start = first_delivery - pd.Timedelta(days=365)
    holdout_start = first_delivery.isoformat()
    predictions_sha = "a" * 64
    audit = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": BASE_MODEL,
        "candidate_checkpoint_sha256": experiment["checkpoint_sha256"],
        "training_days": 365,
        "training_start_day": training_start.isoformat(),
        "training_end_day": training_end.isoformat(),
        "holdout_start_day": holdout_start,
        "predictions_sha256": predictions_sha,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
    }
    audit_path = tmp_path / "oof_audit.json"
    _write_json(audit_path, audit)
    corrector = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": BASE_MODEL,
        "output_model": OUTPUT_MODEL,
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "training_start_day": training_start.isoformat(),
        "training_end_day": training_end.isoformat(),
        "holdout_start_day": holdout_start,
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "oof_training_predictions_sha256": predictions_sha,
        "oof_training_audit_sha256": _sha256(audit_path),
        "oof_sidecar_schema_version": 1,
        "candidate_model": BASE_MODEL,
        "candidate_checkpoint_sha256": experiment["checkpoint_sha256"],
        "feature_columns": ["intercept"],
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "coefficients": [-1.0],
        "ridge_alpha": 1.0,
        "maximum_absolute_shift_eur_mwh": 20.0,
    }
    corrector_path = tmp_path / "residual_corrector.json"
    _write_json(corrector_path, corrector)
    return corrector_path, audit_path


def _make_incumbent(
    tmp_path: Path,
    raw: pd.DataFrame,
    *,
    actual_delta: float = 0.0,
    extra_days: bool = True,
) -> Path:
    actual = raw["actual"].to_numpy(float) + actual_delta
    incumbent = pd.DataFrame(
        {
            "delivery_start_utc": raw["delivery_start_utc"],
            "forecast_origin_utc": raw["forecast_origin_utc"],
            "actual": actual,
            "residual_corrected__q10": actual - 6.5,
            "residual_corrected__q50": actual + 1.5,
            "residual_corrected__q90": actual + 9.5,
            "unrelated": 1,
        }
    )
    if extra_days:
        prefix_delivery = pd.date_range(
            "2024-12-30T23:00:00Z", periods=24, freq="h"
        )
        prefix = pd.DataFrame(
            {
                "delivery_start_utc": prefix_delivery,
                "forecast_origin_utc": pd.Timestamp("2024-12-30T07:00:00Z"),
                "actual": 10.0,
                "residual_corrected__q10": 5.0,
                "residual_corrected__q50": 10.0,
                "residual_corrected__q90": 15.0,
                "unrelated": 2,
            }
        )
        incumbent = pd.concat([prefix, incumbent], ignore_index=True)
    path = tmp_path / "statistics_history_hourly.csv.gz"
    incumbent.to_csv(path, index=False, compression="gzip")
    return path


def _inputs(
    tmp_path: Path, *, start_day: str = "2025-01-01"
) -> tuple[Path, Path, Path, pd.DataFrame]:
    run, raw = _make_bundle(tmp_path, start_day=start_day)
    corrector, audit = _make_corrector(tmp_path, run)
    incumbent = _make_incumbent(tmp_path, raw)
    return run, corrector, audit, raw


def test_final_pipeline_publishes_complete_paired_evidence_atomically(
    tmp_path: Path,
) -> None:
    run, corrector, audit, raw = _inputs(tmp_path)
    incumbent = tmp_path / "statistics_history_hourly.csv.gz"
    raw_sha = _sha256(run / "evaluation_predictions.csv.gz")

    result = run_final_pipeline_evaluation(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="fr",
    )

    assert result.output_directory == run / FINAL_DIRECTORY_NAME
    assert {path.name for path in result.output_directory.iterdir()} == {
        FINAL_EVIDENCE_NAME,
        FINAL_METRICS_NAME,
        FINAL_REPORT_NAME,
        FINAL_AUDIT_NAME,
        FINAL_MANIFEST_NAME,
    }
    assert _sha256(run / "evaluation_predictions.csv.gz") == raw_sha
    evidence = pd.read_csv(result.evidence_path)
    assert tuple(evidence.columns) == EVALUATION_COLUMNS
    assert len(evidence) == 8760
    assert result.metrics["baseline_mae_eur_mwh"] == pytest.approx(1.5)
    assert result.metrics["candidate_mae_eur_mwh"] == pytest.approx(1.0)
    assert result.metrics["mae_gain_eur_mwh"] == pytest.approx(0.5)
    assert "MAE LoRA corrig" in result.report_path.read_text(
        encoding="utf-8"
    )

    experiment = json.loads(result.experiment_manifest_path.read_text())
    assert experiment["production_pit_evidence"] is False
    assert experiment["production_pipeline_evidence"] is True
    assert experiment["candidate_output_stage"] == OUTPUT_MODEL
    assert experiment["raw_evaluation_evidence"]["sha256"] == raw_sha
    assert experiment["evaluation_evidence"]["sha256"] == _sha256(
        result.evidence_path
    )
    assert experiment["residual_corrector_sha256"] == _sha256(corrector)
    assert experiment["oof_training_audit_sha256"] == _sha256(audit)
    _validate_final_pipeline_evidence(experiment, rolling_path=result.evidence_path)
    final_audit = json.loads(result.audit_path.read_text())
    assert final_audit["incumbent_statistics"]["source_rows"] == 8784
    assert final_audit["incumbent_statistics"]["paired_rows"] == 8760
    assert final_audit["checks"]["production_pit_evidence_unchanged"] is True
    assert len(raw) == 8760


@pytest.mark.parametrize(
    ("start_day", "expected_hours"),
    (("2021-03-28", 8759), ("2021-10-31", 8761)),
)
def test_final_pipeline_accepts_dst_shifted_365_day_physical_windows(
    tmp_path: Path,
    start_day: str,
    expected_hours: int,
) -> None:
    # Depending on its exact civil boundaries, 365 local days may contain two
    # spring changes or two autumn changes and therefore 8,759/8,761 hours.
    run, corrector, audit, raw = _inputs(tmp_path, start_day=start_day)
    incumbent = _make_incumbent(tmp_path, raw, extra_days=False)
    assert len(raw) == expected_hours

    result = run_final_pipeline_evaluation(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="FR",
    )

    assert result.metrics["physical_hours"] == expected_hours
    final_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["window"]["physical_hours"] == expected_hours
    final_audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert final_audit["checks"][
        "exact_physical_hours_for_365_local_days"
    ] is True
    assert final_audit["checks"]["expected_physical_hours"] == expected_hours
    assert final_audit["checks"]["exact_8760_physical_hours"] is False


def test_actual_csv_rounding_tolerance_is_audited(tmp_path: Path) -> None:
    run, corrector, audit, raw = _inputs(tmp_path)
    incumbent = _make_incumbent(
        tmp_path,
        raw,
        actual_delta=ACTUAL_PAIRING_ATOL_EUR_MWH * 0.5,
        extra_days=False,
    )

    result = run_final_pipeline_evaluation(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="FR",
    )

    audit_payload = json.loads(result.audit_path.read_text())
    assert audit_payload["checks"]["actual_pairing_atol_eur_mwh"] == pytest.approx(
        ACTUAL_PAIRING_ATOL_EUR_MWH
    )
    assert audit_payload["checks"][
        "actual_pairing_max_delta_eur_mwh"
    ] == pytest.approx(ACTUAL_PAIRING_ATOL_EUR_MWH * 0.5)


def test_final_pipeline_refuses_unresolved_two_phase_labels_without_resolution(
    tmp_path: Path,
) -> None:
    run, corrector, audit, _raw = _inputs(tmp_path)
    manifest_path = run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["panel_sha256"] = "1" * 64
    manifest["panel_audit_sha256"] = "2" * 64
    manifest["evaluation_label_binding"] = {
        "schema_version": 1,
        "policy": "last_evaluation_physical_horizon_only",
        "allow_unresolved_final_evaluation_day": True,
        "unresolved_groups": [{"placeholder": True}],
        "unresolved_cells": 24,
        "labels_used_for_fit": False,
        "train_validation_labels_all_finite": True,
        "resolution_required_before_evaluation": True,
        "input_contract_sha256": "3" * 64,
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(FinalPipelineEvaluationError, match="panel resolu"):
        run_final_pipeline_evaluation(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
            zone="FR",
        )

    assert not (run / FINAL_DIRECTORY_NAME).exists()


def test_actual_divergence_fails_before_any_publication(tmp_path: Path) -> None:
    run, corrector, audit, raw = _inputs(tmp_path)
    incumbent = _make_incumbent(
        tmp_path,
        raw,
        actual_delta=ACTUAL_PAIRING_ATOL_EUR_MWH * 2.0,
        extra_days=False,
    )
    original_manifest = (run / "experiment_manifest.json").read_bytes()

    with pytest.raises(FinalPipelineEvaluationError, match="actuals"):
        run_final_pipeline_evaluation(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=incumbent,
            zone="FR",
        )

    assert not (run / FINAL_DIRECTORY_NAME).exists()
    assert (run / "experiment_manifest.json").read_bytes() == original_manifest


def test_overwrite_is_explicit_and_keeps_raw_reference(tmp_path: Path) -> None:
    run, corrector, audit, _raw = _inputs(tmp_path)
    incumbent = tmp_path / "statistics_history_hourly.csv.gz"
    first = run_final_pipeline_evaluation(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="FR",
    )
    raw_reference = json.loads((run / "experiment_manifest.json").read_text())[
        "raw_evaluation_evidence"
    ]

    with pytest.raises(FinalPipelineEvaluationError, match="overwrite explicite"):
        run_final_pipeline_evaluation(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=incumbent,
            zone="FR",
        )

    second = run_final_pipeline_evaluation(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="FR",
        overwrite=True,
    )
    assert first.output_directory == second.output_directory
    experiment = json.loads((run / "experiment_manifest.json").read_text())
    assert experiment["raw_evaluation_evidence"] == raw_reference
    assert experiment["evaluation_evidence"]["sha256"] == _sha256(
        second.evidence_path
    )


def test_oof_audit_must_be_for_same_checkpoint_and_holdout(tmp_path: Path) -> None:
    run, corrector, audit, _raw = _inputs(tmp_path)
    incumbent = tmp_path / "statistics_history_hourly.csv.gz"
    payload = json.loads(audit.read_text())
    payload["holdout_start_day"] = "2025-01-02"
    _write_json(audit, payload)
    # Rebind the corrector to the modified sidecar so the failure exercises
    # the temporal contract rather than only the checksum contract.
    corrector_payload = json.loads(corrector.read_text())
    corrector_payload["oof_training_audit_sha256"] = _sha256(audit)
    _write_json(corrector, corrector_payload)

    with pytest.raises(FinalPipelineEvaluationError, match="sidecar OOF invalide"):
        run_final_pipeline_evaluation(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=incumbent,
            zone="FR",
        )


def test_manifest_commit_failure_rolls_back_every_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, audit, _raw = _inputs(tmp_path)
    incumbent = tmp_path / "statistics_history_hourly.csv.gz"
    experiment_path = run / "experiment_manifest.json"
    original_manifest = experiment_path.read_bytes()
    real_replace = os.replace

    def fail_manifest_commit(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == experiment_path and ".tmp-" in Path(source).name:
            raise PermissionError("simulated manifest lock")
        real_replace(source, destination)

    monkeypatch.setattr("chronos2_exogenous.final_pipeline.os.replace", fail_manifest_commit)
    with pytest.raises(PermissionError, match="simulated manifest lock"):
        run_final_pipeline_evaluation(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=incumbent,
            zone="FR",
        )

    assert experiment_path.read_bytes() == original_manifest
    assert not (run / FINAL_DIRECTORY_NAME).exists()
    assert not list(run.glob(".final_pipeline.*"))
