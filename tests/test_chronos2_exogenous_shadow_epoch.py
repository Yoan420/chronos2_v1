from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

import chronos2_exogenous.shadow_epoch as shadow_epoch
from chronos2_exogenous.lora_finetune import sha256_directory
from chronos2_exogenous.shadow_epoch import (
    ShadowEpochError,
    assess_shadow_epoch,
    earliest_new_epoch_plan,
    freeze_shadow_epoch,
    validate_finalisation_against_epoch,
    validate_shadow_delivery_against_epoch,
    verify_shadow_epoch,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(
    tmp_path: Path,
    *,
    production_pit: bool = True,
    unresolved_labels: bool = False,
) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    checkpoint = run / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    schema = {
        "format_version": 1,
        "timestamp_column": "timestamp",
        "origin_column": "origin_timestamp",
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
    _write_json(run / "schema.json", schema)
    checkpoint_sha = sha256_directory(checkpoint)
    oof = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": "chronos2_exogenous",
        "candidate_checkpoint_sha256": checkpoint_sha,
        "training_days": 365,
        "training_start_day": "2024-09-03",
        "training_end_day": "2025-09-02",
        "holdout_start_day": "2025-09-03",
        "predictions_sha256": "7" * 64,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
        "production_pit_evidence": production_pit,
    }
    label_binding = {
        "schema_version": 1,
        "policy": "last_evaluation_physical_horizon_only",
        "allow_unresolved_final_evaluation_day": True,
        "unresolved_groups": [
            {
                "origin_utc": "2026-09-01T06:00:00+00:00",
                "item_id": "FR",
                "delivery_day": "2026-09-02",
                "target_columns": ["target"],
                "hours": 24,
                "first_delivery_utc": "2026-09-01T22:00:00+00:00",
                "last_delivery_utc": "2026-09-02T21:00:00+00:00",
            }
        ],
        "unresolved_cells": 24,
        "labels_used_for_fit": False,
        "train_validation_labels_all_finite": True,
        "resolution_required_before_evaluation": True,
        "input_contract_sha256": "6" * 64,
    }
    if unresolved_labels:
        oof["holdout_label_binding"] = label_binding
        oof["holdout_targets_used_for_fit"] = False
    oof_path = tmp_path / "oof.json"
    _write_json(oof_path, oof)
    corrector = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": "chronos2_exogenous",
        "output_model": "exogenous_residual_corrected",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "training_start_day": "2024-09-03",
        "training_end_day": "2025-09-02",
        "holdout_start_day": "2025-09-03",
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "oof_training_predictions_sha256": oof["predictions_sha256"],
        "oof_training_audit_sha256": _sha(oof_path),
        "oof_sidecar_schema_version": 1,
        "candidate_model": "chronos2_exogenous",
        "candidate_checkpoint_sha256": checkpoint_sha,
        "feature_columns": ["intercept"],
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "coefficients": [0.0],
        "ridge_alpha": 1.0,
        "maximum_absolute_shift_eur_mwh": 20.0,
    }
    corrector_path = tmp_path / "corrector.json"
    _write_json(corrector_path, corrector)
    manifest = {
        "format_version": 1,
        "model_id": "amazon/chronos-2",
        "experiment_id": "future-r16",
        "evaluation_role": "primary_predeclared",
        "finetune_mode": "lora",
        "zone": "FR",
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
        "production_pit_evidence": production_pit,
        "production_pipeline_evidence": False,
        "checkpoint_sha256": checkpoint_sha,
        "schema_sha256": _sha(run / "schema.json"),
        "splits": {
            "evaluation_holdout": {
                "count": 365,
                "first_utc": "2025-09-02T06:00:00+00:00",
                "last_utc": "2026-09-01T06:00:00+00:00",
            }
        },
    }
    if unresolved_labels:
        manifest["evaluation_label_binding"] = label_binding
    _write_json(run / "experiment_manifest.json", manifest)
    return run, corrector_path, oof_path


def test_current_window_is_irrecoverably_missed(tmp_path: Path) -> None:
    run, corrector, oof = _bundle(tmp_path)
    result = assess_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        assessed_at_utc="2026-09-04T17:00:00Z",
    )
    assert result.holdout_end_day == "2026-09-02"
    assert result.first_shadow_day == "2026-09-03"
    assert result.freeze_window_open is False
    assert result.ready_to_freeze is False
    assert any("fenetre de gel manquee" in blocker for blocker in result.blockers)


def test_next_new_epoch_dates_after_september_4_evening() -> None:
    result = earliest_new_epoch_plan(
        timezone_name="Europe/Paris",
        assessed_at_utc="2026-09-04T17:00:00Z",
    )
    assert result["holdout_start_day"] == "2025-09-06"
    assert result["holdout_end_day"] == "2026-09-05"
    assert result["freeze_must_complete_before_utc"] == "2026-09-05T06:00:00+00:00"
    assert result["first_shadow_day"] == "2026-09-06"
    assert result["minimum_shadow_end_day"] == "2026-10-05"
    assert result["earliest_governance_day"] == "2026-10-06"
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize(
    ("assessed_at_utc", "expected_deadline_utc", "expected_first_shadow"),
    (
        ("2026-03-29T05:00:00Z", "2026-03-29T06:00:00+00:00", "2026-03-30"),
        ("2026-10-25T05:00:00Z", "2026-10-25T07:00:00+00:00", "2026-10-26"),
    ),
)
def test_epoch_plan_uses_wall_clock_cutoff_across_dst_transitions(
    assessed_at_utc: str,
    expected_deadline_utc: str,
    expected_first_shadow: str,
) -> None:
    result = earliest_new_epoch_plan(
        timezone_name="Europe/Paris",
        assessed_at_utc=assessed_at_utc,
    )

    assert result["freeze_must_complete_before_utc"] == expected_deadline_utc
    assert result["first_shadow_day"] == expected_first_shadow


def test_two_phase_freeze_is_self_contained_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    monkeypatch.setattr(
        shadow_epoch, "_utc_now", lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z")
    )
    output = tmp_path / "epoch"
    manifest_path = freeze_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        output_directory=output,
    )
    assert manifest_path == output / "shadow_epoch_freeze.json"
    payload = verify_shadow_epoch(output, run_directory=run)
    assert payload["shadow"]["first_day"] == "2026-09-03"
    assert payload["final_backtest_status_at_freeze"] == "metrics_finalisation_pending"
    assert payload["phase_contract"]["rolling_metrics_may_be_finalized_after_first_origin"] is True
    assert payload["promotion_eligible"] is False

    (output / "residual_corrector.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ShadowEpochError, match="SHA de residual_corrector divergent"):
        verify_shadow_epoch(output, run_directory=run)


def test_freeze_rechecks_the_clock_after_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    clocks = iter(
        (
            pd.Timestamp("2026-09-02T05:00:00Z"),
            pd.Timestamp("2026-09-02T05:59:59Z"),
            pd.Timestamp("2026-09-02T06:00:00Z"),
        )
    )
    monkeypatch.setattr(shadow_epoch, "_utc_now", lambda: next(clocks))
    output = tmp_path / "late-epoch"

    with pytest.raises(ShadowEpochError, match="commit atomique.*depasse"):
        freeze_shadow_epoch(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=oof,
            zone="FR",
            output_directory=output,
        )

    assert not output.exists()


def test_legacy_fully_observed_bundle_accepts_new_empty_oof_binding(
    tmp_path: Path,
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    oof_payload = json.loads(oof.read_text(encoding="utf-8"))
    oof_payload["holdout_label_binding"] = {
        "schema_version": 1,
        "policy": "last_evaluation_physical_horizon_only",
        "allow_unresolved_final_evaluation_day": False,
        "unresolved_groups": [],
        "unresolved_cells": 0,
        "labels_used_for_fit": False,
        "train_validation_labels_all_finite": True,
        "resolution_required_before_evaluation": False,
        "input_contract_sha256": "6" * 64,
    }
    oof_payload["holdout_targets_used_for_fit"] = False
    _write_json(oof, oof_payload)
    corrector_payload = json.loads(corrector.read_text(encoding="utf-8"))
    corrector_payload["oof_training_audit_sha256"] = _sha(oof)
    _write_json(corrector, corrector_payload)

    result = assess_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        assessed_at_utc="2026-09-01T05:00:00Z",
    )
    assert result.ready_to_freeze is True


def test_production_freeze_refuses_research_pit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, corrector, oof = _bundle(tmp_path, production_pit=False)
    monkeypatch.setattr(
        shadow_epoch, "_utc_now", lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z")
    )
    assessment = assess_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
    )
    assert assessment.production_pit_ready is False
    with pytest.raises(ShadowEpochError, match="production_pit_evidence=false"):
        freeze_shadow_epoch(
            run_directory=run,
            residual_corrector_path=corrector,
            oof_audit_path=oof,
            zone="FR",
            output_directory=tmp_path / "epoch",
        )


def test_phase_a_freezes_with_predeclared_missing_actual_and_no_future_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path, unresolved_labels=True)
    monkeypatch.setattr(
        shadow_epoch,
        "_utc_now",
        lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z"),
    )
    output = tmp_path / "epoch"
    freeze_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        output_directory=output,
    )
    payload = verify_shadow_epoch(output, run_directory=run)
    assert payload["evaluation_label_binding"]["unresolved_cells"] == 24
    assert payload["final_backtest_status_at_freeze"] == "metrics_finalisation_pending"
    assert payload["promotion_eligible"] is False

    manifest_path = run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["production_pipeline_evidence"] = True
    manifest["candidate_output_stage"] = "exogenous_residual_corrected"
    _write_json(manifest_path, manifest)
    with pytest.raises(ShadowEpochError, match="labels holdout.*panel resolu"):
        validate_finalisation_against_epoch(output, run_directory=run)


def test_freeze_refuses_a_requested_gap(tmp_path: Path) -> None:
    run, corrector, oof = _bundle(tmp_path)
    result = assess_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        requested_first_shadow_day="2026-09-06",
        assessed_at_utc="2026-09-01T05:00:00Z",
    )
    assert result.continuity_ready is False
    assert any("lendemain exact" in blocker for blocker in result.blockers)


def test_shadow_delivery_preflight_refuses_a_missing_first_or_next_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    monkeypatch.setattr(
        shadow_epoch,
        "_utc_now",
        lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z"),
    )
    output = tmp_path / "epoch"
    freeze_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        output_directory=output,
    )
    journal = tmp_path / "shadow.csv.gz"
    first = validate_shadow_delivery_against_epoch(
        output,
        run_directory=run,
        journal_path=journal,
        delivery_day="2026-09-03",
    )
    assert first["status"] == "first_forecast"
    with pytest.raises(ShadowEpochError, match="Premier shadow"):
        validate_shadow_delivery_against_epoch(
            output,
            run_directory=run,
            journal_path=journal,
            delivery_day="2026-09-04",
        )

    start = pd.Timestamp("2026-09-03", tz="Europe/Paris")
    end = pd.Timestamp("2026-09-04", tz="Europe/Paris")
    deliveries = pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
    )
    checkpoint_sha = json.loads(
        (run / "experiment_manifest.json").read_text(encoding="utf-8")
    )["checkpoint_sha256"]
    pd.DataFrame(
        {
            "delivery_start_utc": deliveries,
            "forecast_origin_utc": pd.Timestamp("2026-09-02T06:00:00Z"),
            "record_kind": "forecast",
            "checkpoint_sha256": checkpoint_sha,
        }
    ).to_csv(journal, index=False, compression="gzip")
    next_day = validate_shadow_delivery_against_epoch(
        output,
        run_directory=run,
        journal_path=journal,
        delivery_day="2026-09-04",
    )
    assert next_day["status"] == "next_forecast"
    with pytest.raises(ShadowEpochError, match="non continue"):
        validate_shadow_delivery_against_epoch(
            output,
            run_directory=run,
            journal_path=journal,
            delivery_day="2026-09-05",
        )


def test_phase_b_refuses_to_infer_finalbacktest_from_the_phase_a_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    monkeypatch.setattr(
        shadow_epoch, "_utc_now", lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z")
    )
    output = tmp_path / "epoch"
    freeze_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        output_directory=output,
    )
    with pytest.raises(ShadowEpochError, match="FinalBacktest.*non scelle"):
        validate_finalisation_against_epoch(output, run_directory=run)


def test_phase_b_accepts_only_the_predeclared_window_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, corrector, oof = _bundle(tmp_path)
    monkeypatch.setattr(
        shadow_epoch, "_utc_now", lambda: shadow_epoch.pd.Timestamp("2026-09-01T05:00:00Z")
    )
    output = tmp_path / "epoch"
    freeze_shadow_epoch(
        run_directory=run,
        residual_corrector_path=corrector,
        oof_audit_path=oof,
        zone="FR",
        output_directory=output,
    )

    final_dir = run / "final_pipeline"
    final_dir.mkdir()
    local_start = pd.Timestamp("2025-09-03", tz="Europe/Paris")
    local_end = pd.Timestamp("2026-09-03", tz="Europe/Paris")
    deliveries = pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    origins = [
        pd.Timestamp(
            datetime.combine(
                timestamp.tz_convert("Europe/Paris").date() - timedelta(days=1),
                time(8),
            ),
            tz="Europe/Paris",
        ).tz_convert("UTC")
        for timestamp in deliveries
    ]
    evidence_path = final_dir / "final_pipeline_predictions.csv.gz"
    pd.DataFrame(
        {
            "delivery_start_utc": deliveries,
            "forecast_origin_utc": origins,
            "actual": 50.0,
            "baseline_q10": 40.0,
            "baseline_q50": 50.0,
            "baseline_q90": 60.0,
            "candidate_q10": 41.0,
            "candidate_q50": 51.0,
            "candidate_q90": 61.0,
        }
    ).to_csv(evidence_path, index=False, compression="gzip")
    evidence_sha = _sha(evidence_path)
    final_manifest_path = final_dir / "final_pipeline_manifest.json"
    _write_json(
        final_manifest_path,
        {
            "schema_version": 1,
            "kind": "chronos2_exogenous_final_pipeline_evaluation",
            "zone": "FR",
            "experiment_id": "future-r16",
            "production_pipeline_evidence": True,
            "window": {
                "physical_days": 365,
                "physical_hours": len(deliveries),
                "first_delivery_utc": deliveries[0].isoformat(),
                "last_delivery_utc": deliveries[-1].isoformat(),
            },
            "artifacts": {
                "predictions": {
                    "relative_path": "final_pipeline_predictions.csv.gz",
                    "sha256": evidence_sha,
                }
            },
        },
    )
    manifest_path = run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "production_pipeline_evidence": True,
            "candidate_output_stage": "exogenous_residual_corrected",
            "production_pipeline_evidence_detail": {
                "comparison_scope": "paired_operational_final_pipelines",
                "baseline_output_stage": "residual_corrected",
                "candidate_output_stage": "exogenous_residual_corrected",
                "paired_same_input_contract": True,
                "paired_same_evaluation_window": True,
                "baseline_residual_corrector_applied": True,
                "candidate_residual_corrector_applied": True,
                "rolling_evaluation_days": 365,
                "promotion_eligible": True,
            },
            "residual_corrector_sha256": _sha(corrector),
            "oof_training_audit_sha256": _sha(oof),
            "evaluation_evidence": {
                "relative_path": "final_pipeline/final_pipeline_predictions.csv.gz",
                "sha256": evidence_sha,
                "rows": 8760,
                "physical_days": 365,
                "first_delivery_utc": deliveries[0].isoformat(),
                "last_delivery_utc": deliveries[-1].isoformat(),
            },
            "final_pipeline_evaluation": {
                "relative_path": "final_pipeline/final_pipeline_manifest.json",
                "sha256": _sha(final_manifest_path),
            },
        }
    )
    _write_json(manifest_path, manifest)

    result = validate_finalisation_against_epoch(output, run_directory=run)
    assert result["ready_for_final_shadow_scoring"] is True
    assert result["holdout_end_day"] == "2026-09-02"
    assert result["first_shadow_day"] == "2026-09-03"
    assert result["promotion_eligible"] is False

    forged = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    forged["kind"] = "lookalike-finalbacktest"
    _write_json(final_manifest_path, forged)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_pipeline_evaluation"]["sha256"] = _sha(final_manifest_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ShadowEpochError, match="contenu du manifeste FinalBacktest"):
        validate_finalisation_against_epoch(output, run_directory=run)
