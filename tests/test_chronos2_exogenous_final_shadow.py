from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import chronos2_exogenous.production as production
from chronos2_exogenous.evaluation import (
    SHADOW_JOURNAL_COLUMNS,
    _canonical_record_hash,
)
from chronos2_exogenous.governance import (
    ExogenousGovernanceError,
    validate_final_shadow_manifest,
)
from chronos2_exogenous.lora_finetune import EVALUATION_COLUMNS, sha256_directory
from chronos2_exogenous.production import (
    BASE_MODEL,
    OUTPUT_MODEL,
    ExogenousProductionError,
    PromotedBundle,
    load_promoted_shadow_history,
)
from chronos2_exogenous.shadow_final import (
    FINAL_SHADOW_EVIDENCE_NAME,
    FINAL_SHADOW_ISSUED_NAME,
    FINAL_SHADOW_MANIFEST_NAME,
    FinalShadowError,
    finalize_shadow_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _timeline() -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    delivery_day = pd.Timestamp("2026-02-01")
    start = delivery_day.tz_localize("Europe/Paris")
    end = (delivery_day + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    delivery = pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    origin = pd.Timestamp("2026-01-31 08:00", tz="Europe/Paris").tz_convert(
        "UTC"
    )
    return delivery, origin


def _panel_evidence(
    *,
    origin: pd.Timestamp,
    checkpoint_sha: str,
    zone: str = "FR",
    delivery_day: str = "2026-02-01",
) -> dict[str, Any]:
    del checkpoint_sha  # The adapter identity is carried by the shadow manifest.
    panel: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": "1" * 64,
        "panel_audit_sha256": "2" * 64,
        "panel_created_at_utc": (origin + pd.Timedelta(minutes=30)).isoformat(),
        "zone": zone,
        "delivery_day": delivery_day,
        "forecast_origin_utc": origin.isoformat(),
        "forecast_origin_timezone": "Europe/Paris",
        "delivery_timezone": "Europe/Paris",
        "pack": "residual_only",
        "production_pit_evidence": True,
        "source_hashes": {
            "deterministic_calendar": "3" * 64,
            "residual_load": "4" * 64,
        },
        "source_audit_hashes": {"residual_load": "5" * 64},
        "source_cutoff_timezones": {
            "deterministic_calendar": "Europe/Paris",
            "residual_load": "Europe/Paris",
        },
        "target_source_sha256": "6" * 64,
        "horizon_actuals_present": False,
    }
    panel["panel_contract_sha256"] = _canonical_sha(panel)
    return panel


def _write_bundle_and_corrector(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    run = tmp_path / "lora-run"
    checkpoint = run / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter.safetensors").write_bytes(b"sealed-adapter")
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
    checkpoint_sha = sha256_directory(checkpoint)

    oof_audit = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": BASE_MODEL,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "training_days": 365,
        "training_start_day": "2024-01-01",
        "training_end_day": "2024-12-30",
        "holdout_start_day": "2025-01-01",
        "predictions_sha256": "7" * 64,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
    }
    oof_path = tmp_path / "oof_audit.json"
    _write_json(oof_path, oof_audit)
    corrector = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": BASE_MODEL,
        "output_model": OUTPUT_MODEL,
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "training_start_day": "2024-01-01",
        "training_end_day": "2024-12-30",
        "holdout_start_day": "2025-01-01",
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "oof_training_predictions_sha256": oof_audit["predictions_sha256"],
        "oof_training_audit_sha256": _sha256(oof_path),
        "oof_sidecar_schema_version": 1,
        "candidate_model": BASE_MODEL,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "feature_columns": ["intercept"],
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "coefficients": [-1.0],
        "ridge_alpha": 1.0,
        "maximum_absolute_shift_eur_mwh": 20.0,
    }
    corrector_path = tmp_path / "residual_corrector.json"
    _write_json(corrector_path, corrector)

    rolling_evidence = run / "final_pipeline" / "final_pipeline_predictions.csv.gz"
    rolling_evidence.parent.mkdir()
    pd.DataFrame({"proof": [1]}).to_csv(
        rolling_evidence, index=False, compression="gzip"
    )
    rolling_manifest = rolling_evidence.parent / "final_pipeline_manifest.json"
    _write_json(
        rolling_manifest,
        {
            "schema_version": 1,
            "purpose": "chronos2_exogenous_final_pipeline_rolling365",
            "evidence_sha256": _sha256(rolling_evidence),
        },
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_id": "amazon/chronos-2",
        "experiment_id": "chronos2-exogenous-final-shadow-test-v1",
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
        "production_pit_evidence": True,
        "production_pipeline_evidence": True,
        "candidate_output_stage": OUTPUT_MODEL,
        "checkpoint_sha256": checkpoint_sha,
        "schema_sha256": _sha256(schema_path),
        "residual_corrector_sha256": _sha256(corrector_path),
        "oof_training_audit_sha256": _sha256(oof_path),
        "evaluation_evidence": {
            "relative_path": "final_pipeline/final_pipeline_predictions.csv.gz",
            "sha256": _sha256(rolling_evidence),
            "rows": 8760,
            "physical_days": 365,
            "baseline_output_stage": "residual_corrected",
            "candidate_output_stage": OUTPUT_MODEL,
        },
        "final_pipeline_evaluation": {
            "relative_path": "final_pipeline/final_pipeline_manifest.json",
            "sha256": _sha256(rolling_manifest),
        },
        "production_pipeline_evidence_detail": {
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_output_stage": "residual_corrected",
            "candidate_output_stage": OUTPUT_MODEL,
            "paired_same_input_contract": True,
            "paired_same_evaluation_window": True,
            "baseline_residual_corrector_applied": True,
            "candidate_residual_corrector_applied": True,
            "rolling_evaluation_days": 365,
            "promotion_eligible": True,
        },
    }
    _write_json(run / "experiment_manifest.json", manifest)
    return run, corrector_path, oof_path, manifest


def _write_raw_shadow(
    tmp_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, pd.DataFrame]:
    delivery, origin = _timeline()
    actual = 50.0 + np.arange(len(delivery), dtype=float)
    raw = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origin,
            "actual": actual,
            # Deliberately very poor: the final baseline must never reuse this.
            "baseline_q10": actual + 90.0,
            "baseline_q50": actual + 100.0,
            "baseline_q90": actual + 110.0,
            "candidate_q10": actual - 6.0,
            "candidate_q50": actual + 2.0,
            "candidate_q90": actual + 10.0,
        },
        columns=EVALUATION_COLUMNS,
    )
    evidence_path = tmp_path / "shadow_observed_evidence.csv.gz"
    raw.to_csv(evidence_path, index=False, compression="gzip")

    panel = _panel_evidence(
        origin=origin,
        checkpoint_sha=str(manifest["checkpoint_sha256"]),
    )
    forecast_created = origin + pd.Timedelta(hours=1)
    actual_attached = delivery[-1] + pd.Timedelta(hours=12)
    temporal = [
        {
            "delivery_start_utc": timestamp.isoformat(),
            "forecast_origin_utc": origin.isoformat(),
            "forecast_deadline_utc": (origin + pd.Timedelta(hours=4)).isoformat(),
            "forecast_created_at_utc": forecast_created.isoformat(),
            "actual_attached_at_utc": actual_attached.isoformat(),
            "forecast_record_sha256": hashlib.sha256(
                f"forecast-{timestamp.isoformat()}".encode()
            ).hexdigest(),
            "resolution_record_sha256": hashlib.sha256(
                f"resolution-{timestamp.isoformat()}".encode()
            ).hexdigest(),
            "panel_contract_sha256": panel["panel_contract_sha256"],
        }
        for timestamp in delivery
    ]
    raw_manifest = {
        "format_version": 3,
        "kind": "chronos2_exogenous_append_only_shadow",
        "candidate_output_stage": "chronos2_exogenous",
        "residual_corrector_applied": False,
        "candidate_model": manifest["experiment_id"],
        "zone": "FR",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "schema_sha256": manifest["schema_sha256"],
        "candidate_frozen_before_shadow": True,
        "actuals_attached_after_forecast_freeze": True,
        "prospective_capture_deadline_enforced": True,
        "prospective_capture_deadline_hours": 4,
        "forecast_artifact_checksums_valid": True,
        "storm_used_for_prediction": False,
        "mkonline_used_for_prediction": False,
        "predictions_sha256": _sha256(evidence_path),
        "forecast_created_at_utc": forecast_created.isoformat(),
        "actual_attached_at_utc": actual_attached.isoformat(),
        "all_actuals_attached_after_corresponding_forecast": True,
        "shadow_panel_production_ready": True,
        "shadow_panel_evidence": [panel],
        "temporal_attachment_audit": temporal,
        "observed_governance_evidence": {
            "relative_path": evidence_path.name,
            "sha256": _sha256(evidence_path),
            "rows": len(raw),
        },
    }
    provenance_json = json.dumps(
        panel, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    journal_records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for record_kind in ("forecast", "actual_resolution"):
        for index, timestamp in enumerate(delivery):
            row = raw.iloc[index]
            record: dict[str, Any] = {
                "delivery_start_utc": timestamp,
                "forecast_origin_utc": origin,
                "item_id": "FR",
                "target_column": "target",
                "actual": (
                    np.nan if record_kind == "forecast" else float(row["actual"])
                ),
                "baseline_q10": float(row["baseline_q10"]),
                "baseline_q50": float(row["baseline_q50"]),
                "baseline_q90": float(row["baseline_q90"]),
                "candidate_q10": float(row["candidate_q10"]),
                "candidate_q50": float(row["candidate_q50"]),
                "candidate_q90": float(row["candidate_q90"]),
                "input_contract_sha256": "8" * 64,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "panel_sha256": panel["panel_sha256"],
                "panel_audit_sha256": panel["panel_audit_sha256"],
                "panel_contract_sha256": panel["panel_contract_sha256"],
                "panel_created_at_utc": pd.Timestamp(panel["panel_created_at_utc"]),
                "panel_provenance_json": provenance_json,
                "captured_at_utc": (
                    forecast_created
                    if record_kind == "forecast"
                    else actual_attached
                ),
                "revision": 1 if record_kind == "forecast" else 2,
                "record_kind": record_kind,
                "previous_record_sha256": previous_hash,
                "record_sha256": "",
            }
            record["record_sha256"] = _canonical_record_hash(record)
            previous_hash = str(record["record_sha256"])
            journal_records.append(record)

    # The last issued shadow day deliberately has no observed price yet.  It
    # models the first production activation at D-1 08:00, where the scoring
    # evidence naturally stops at D-2 but the exact FINAL365 history needs the
    # already frozen forecast for D-1.
    pending_day = pd.Timestamp("2026-02-02")
    pending_start = pending_day.tz_localize("Europe/Paris")
    pending_end = (pending_day + pd.Timedelta(days=1)).tz_localize(
        "Europe/Paris"
    )
    pending_delivery = pd.date_range(
        pending_start.tz_convert("UTC"),
        pending_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    pending_origin = pd.Timestamp(
        "2026-02-01 08:00", tz="Europe/Paris"
    ).tz_convert("UTC")
    pending_panel = _panel_evidence(
        origin=pending_origin,
        checkpoint_sha=str(manifest["checkpoint_sha256"]),
        delivery_day="2026-02-02",
    )
    pending_provenance_json = json.dumps(
        pending_panel,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    for timestamp in pending_delivery:
        record = {
            "delivery_start_utc": timestamp,
            "forecast_origin_utc": pending_origin,
            "item_id": "FR",
            "target_column": "target",
            "actual": np.nan,
            "baseline_q10": 45.0,
            "baseline_q50": 50.0,
            "baseline_q90": 55.0,
            "candidate_q10": 40.0,
            "candidate_q50": 50.0,
            "candidate_q90": 60.0,
            "input_contract_sha256": "9" * 64,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "panel_sha256": pending_panel["panel_sha256"],
            "panel_audit_sha256": pending_panel["panel_audit_sha256"],
            "panel_contract_sha256": pending_panel["panel_contract_sha256"],
            "panel_created_at_utc": pd.Timestamp(
                pending_panel["panel_created_at_utc"]
            ),
            "panel_provenance_json": pending_provenance_json,
            "captured_at_utc": pending_origin + pd.Timedelta(hours=1),
            "revision": 1,
            "record_kind": "forecast",
            "previous_record_sha256": previous_hash,
            "record_sha256": "",
        }
        record["record_sha256"] = _canonical_record_hash(record)
        previous_hash = str(record["record_sha256"])
        journal_records.append(record)
    journal_path = tmp_path / "shadow_predictions.csv.gz"
    pd.DataFrame(journal_records, columns=SHADOW_JOURNAL_COLUMNS).to_csv(
        journal_path, index=False, compression="gzip"
    )
    raw_manifest["journal"] = {
        "relative_path": journal_path.name,
        "sha256": _sha256(journal_path),
        "records": len(journal_records),
        "unique_hours": len(raw) + len(pending_delivery),
        "last_record_sha256": previous_hash,
    }
    manifest_path = tmp_path / "shadow_manifest.json"
    _write_json(manifest_path, raw_manifest)
    return evidence_path, manifest_path, raw


def _write_incumbent(
    tmp_path: Path,
    raw: pd.DataFrame,
    *,
    actual_delta: float = 0.0,
    origin_delta: pd.Timedelta = pd.Timedelta(0),
) -> Path:
    actual = raw["actual"].to_numpy(float) + actual_delta
    incumbent = pd.DataFrame(
        {
            "delivery_start_utc": raw["delivery_start_utc"],
            "forecast_origin_utc": pd.to_datetime(
                raw["forecast_origin_utc"], utc=True
            )
            + origin_delta,
            "actual": actual,
            "residual_corrected__q10": actual - 5.0,
            "residual_corrected__q50": actual + 1.5,
            "residual_corrected__q90": actual + 8.0,
        }
    )
    path = tmp_path / "statistics_history_hourly.csv.gz"
    incumbent.to_csv(path, index=False, compression="gzip")
    return path


def _inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, pd.DataFrame, dict[str, Any]]:
    run, corrector, audit, manifest = _write_bundle_and_corrector(tmp_path)
    raw_path, raw_manifest_path, raw = _write_raw_shadow(tmp_path, manifest)
    incumbent = _write_incumbent(tmp_path, raw)
    return run, raw_path, raw_manifest_path, corrector, audit, raw, manifest


def _referenced_sha(payload: object, filename: str) -> str | None:
    if isinstance(payload, Mapping):
        relative = payload.get("relative_path")
        digest = payload.get("sha256")
        if (
            isinstance(relative, str)
            and Path(relative).name == filename
            and isinstance(digest, str)
        ):
            return digest
        for value in payload.values():
            found = _referenced_sha(value, filename)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _referenced_sha(value, filename)
            if found is not None:
                return found
    return None


def _reseal_final_prediction_digest(
    manifest_path: Path, *, evidence_path: Path
) -> None:
    """Make a tampered output internally SHA-consistent, like an adversary."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(evidence_path)
    payload["predictions_sha256"] = digest
    payload["observed_governance_evidence"]["sha256"] = digest
    payload["derivation_lineage"]["final_shadow_predictions_sha256"] = digest
    payload["derivation_sha256"] = _canonical_sha(payload["derivation_lineage"])
    _write_json(manifest_path, payload)


def test_final_shadow_materialises_only_the_paired_corrected_pipeline(
    tmp_path: Path,
) -> None:
    run, raw_path, raw_manifest_path, corrector, audit, raw, experiment = _inputs(
        tmp_path
    )
    incumbent = tmp_path / "statistics_history_hourly.csv.gz"
    output = tmp_path / "final-shadow"

    result = finalize_shadow_evidence(
        run_directory=run,
        raw_observed_evidence_path=raw_path,
        raw_shadow_manifest_path=raw_manifest_path,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=incumbent,
        zone="fr",
        output_directory=output,
    )

    assert result.pending is False
    assert result.output_directory == output.resolve()
    assert result.evidence_path == output.resolve() / FINAL_SHADOW_EVIDENCE_NAME
    assert result.manifest_path == output.resolve() / FINAL_SHADOW_MANIFEST_NAME
    final = pd.read_csv(result.evidence_path)
    assert tuple(final.columns) == EVALUATION_COLUMNS
    np.testing.assert_allclose(final["baseline_q50"], raw["actual"] + 1.5)
    np.testing.assert_allclose(final["candidate_q50"], raw["actual"] + 1.0)
    assert not np.allclose(final["baseline_q50"], raw["baseline_q50"])

    published_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert published_manifest["format_version"] == 4
    assert published_manifest["kind"] == "chronos2_exogenous_final_pipeline_shadow"
    assert published_manifest["comparison_scope"] == (
        "paired_operational_final_pipelines"
    )
    assert published_manifest["baseline_output_stage"] == "residual_corrected"
    assert published_manifest["candidate_output_stage"] == OUTPUT_MODEL
    assert published_manifest["residual_corrector_applied"] is True
    assert published_manifest["predictions_sha256"] == _sha256(result.evidence_path)
    assert published_manifest["residual_corrector_sha256"] == _sha256(corrector)
    assert published_manifest["oof_training_audit_sha256"] == _sha256(audit)
    assert _referenced_sha(
        published_manifest, "raw_shadow_observed_evidence.csv.gz"
    ) == _sha256(raw_path)
    assert _referenced_sha(published_manifest, "raw_shadow_manifest.json") == _sha256(
        raw_manifest_path
    )
    assert _referenced_sha(
        published_manifest, "shadow_final_incumbent.csv.gz"
    ) is not None
    assert {
        FINAL_SHADOW_EVIDENCE_NAME,
        FINAL_SHADOW_ISSUED_NAME,
        FINAL_SHADOW_MANIFEST_NAME,
        "raw_shadow_observed_evidence.csv.gz",
        "raw_shadow_manifest.json",
        "raw_shadow_journal.csv.gz",
        "shadow_final_incumbent.csv.gz",
        "residual_corrector.json",
        "oof_audit.json",
    }.issubset({path.name for path in output.iterdir()})
    issued = pd.read_csv(output / FINAL_SHADOW_ISSUED_NAME)
    assert len(issued) == 48
    assert issued["actual"].notna().sum() == 24
    assert issued["actual"].tail(24).isna().all()
    # The frozen corrector has a -1 EUR/MWh intercept and must also be applied
    # once to the latest, not-yet-observed forecast needed at first activation.
    np.testing.assert_allclose(issued["candidate_q50"].tail(24), 49.0)

    validated = validate_final_shadow_manifest(
        result.manifest_path,
        experiment_manifest=experiment,
        zone="FR",
        expected_rows=len(final),
    )
    assert validated["predictions_sha256"] == _sha256(result.evidence_path)


@pytest.mark.parametrize(
    ("actual_delta", "origin_delta", "message"),
    [
        (1e-3, pd.Timedelta(0), "actual"),
        (0.0, pd.Timedelta(hours=1), "origine|appari"),
    ],
)
def test_final_shadow_rejects_unpaired_incumbent_before_publication(
    tmp_path: Path,
    actual_delta: float,
    origin_delta: pd.Timedelta,
    message: str,
) -> None:
    run, raw_path, raw_manifest_path, corrector, audit, raw, _experiment = _inputs(
        tmp_path
    )
    incumbent = _write_incumbent(
        tmp_path,
        raw,
        actual_delta=actual_delta,
        origin_delta=origin_delta,
    )
    output = tmp_path / "refused"

    with pytest.raises(
        (FinalShadowError, ExogenousProductionError, ExogenousGovernanceError),
        match=message,
    ):
        finalize_shadow_evidence(
            run_directory=run,
            raw_observed_evidence_path=raw_path,
            raw_shadow_manifest_path=raw_manifest_path,
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=incumbent,
            zone="FR",
            output_directory=output,
        )

    assert not output.exists()


def test_final_shadow_manifest_rejects_tampered_copied_lineage(tmp_path: Path) -> None:
    run, raw_path, raw_manifest_path, corrector, audit, _raw, experiment = _inputs(
        tmp_path
    )
    output = tmp_path / "final-shadow"
    result = finalize_shadow_evidence(
        run_directory=run,
        raw_observed_evidence_path=raw_path,
        raw_shadow_manifest_path=raw_manifest_path,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
        zone="FR",
        output_directory=output,
    )
    copied = output / "raw_shadow_observed_evidence.csv.gz"
    copied.write_bytes(copied.read_bytes() + b"tampered")

    with pytest.raises(ExogenousGovernanceError, match="SHA|sha256|empreinte"):
        validate_final_shadow_manifest(
            result.manifest_path,
            experiment_manifest=experiment,
            zone="FR",
            expected_rows=24,
        )


def test_final_shadow_validator_recomputes_candidate_despite_consistent_hashes(
    tmp_path: Path,
) -> None:
    run, raw_path, raw_manifest_path, corrector, audit, _raw, experiment = _inputs(
        tmp_path
    )
    output = tmp_path / "final-shadow"
    result = finalize_shadow_evidence(
        run_directory=run,
        raw_observed_evidence_path=raw_path,
        raw_shadow_manifest_path=raw_manifest_path,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
        zone="FR",
        output_directory=output,
    )
    assert result.evidence_path is not None
    assert result.manifest_path is not None

    tampered = pd.read_csv(result.evidence_path)
    tampered["candidate_q50"] += 0.25
    tampered.to_csv(result.evidence_path, index=False, compression="gzip")
    _reseal_final_prediction_digest(
        result.manifest_path, evidence_path=result.evidence_path
    )

    with pytest.raises(ExogenousGovernanceError, match="candidat different"):
        validate_final_shadow_manifest(
            result.manifest_path,
            experiment_manifest=experiment,
            zone="FR",
            expected_rows=24,
        )


def test_final_shadow_validator_refuses_broadened_actual_tolerance(
    tmp_path: Path,
) -> None:
    run, raw_path, raw_manifest_path, corrector, audit, _raw, experiment = _inputs(
        tmp_path
    )
    output = tmp_path / "final-shadow"
    result = finalize_shadow_evidence(
        run_directory=run,
        raw_observed_evidence_path=raw_path,
        raw_shadow_manifest_path=raw_manifest_path,
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
        zone="FR",
        output_directory=output,
    )
    assert result.manifest_path is not None
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["actual_pairing_atol_eur_mwh"] = 1_000_000.0
    _write_json(result.manifest_path, payload)

    with pytest.raises(ExogenousGovernanceError, match="tolerance/ecart"):
        validate_final_shadow_manifest(
            result.manifest_path,
            experiment_manifest=experiment,
            zone="FR",
            expected_rows=24,
        )


def test_raw_v3_shadow_cannot_satisfy_final_pipeline_validator(
    tmp_path: Path,
) -> None:
    _run, _raw_path, raw_manifest_path, _corrector, _audit, raw, experiment = _inputs(
        tmp_path
    )

    with pytest.raises(ExogenousGovernanceError, match="format_version|pipeline final"):
        validate_final_shadow_manifest(
            raw_manifest_path,
            experiment_manifest=experiment,
            zone="FR",
            expected_rows=len(raw),
        )


def test_final_shadow_can_report_pending_without_publishing_empty_evidence(
    tmp_path: Path,
) -> None:
    run, _raw_path, _raw_manifest_path, corrector, audit, _raw, _experiment = _inputs(
        tmp_path
    )
    output = tmp_path / "pending"

    result = finalize_shadow_evidence(
        run_directory=run,
        raw_observed_evidence_path=tmp_path / "missing-observed.csv.gz",
        raw_shadow_manifest_path=tmp_path / "missing-shadow-manifest.json",
        residual_corrector_path=corrector,
        oof_audit_path=audit,
        incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
        zone="FR",
        output_directory=output,
        allow_no_observed=True,
    )

    assert result.pending is True
    assert result.output_directory == output.resolve()
    assert result.evidence_path is None
    assert result.manifest_path is None
    assert not output.exists()


def test_pending_shadow_still_requires_a_frozen_final_backtest(
    tmp_path: Path,
) -> None:
    run, _raw_path, _raw_manifest_path, corrector, audit, _raw, experiment = _inputs(
        tmp_path
    )
    experiment["production_pipeline_evidence"] = False
    _write_json(run / "experiment_manifest.json", experiment)

    with pytest.raises(FinalShadowError, match="FinalBacktest"):
        finalize_shadow_evidence(
            run_directory=run,
            raw_observed_evidence_path=tmp_path / "missing-observed.csv.gz",
            raw_shadow_manifest_path=tmp_path / "missing-shadow-manifest.json",
            residual_corrector_path=corrector,
            oof_audit_path=audit,
            incumbent_statistics_path=tmp_path / "statistics_history_hourly.csv.gz",
            zone="FR",
            output_directory=tmp_path / "pending-refused",
            allow_no_observed=True,
        )
