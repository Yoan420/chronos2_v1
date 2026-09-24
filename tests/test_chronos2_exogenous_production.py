from __future__ import annotations

from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import chronos2_exogenous.production as production
from chronos2_exogenous.production import (
    ExogenousProductionError,
    PromotedBundle,
    fit_oof_residual_corrector,
    load_promoted_shadow_history,
    load_registered_bundle,
    register_promoted_bundle,
    run_registered_candidate,
    validate_registered_candidate_run,
    validate_promoted_bundle,
)
from chronos2_exogenous.lora_finetune import sha256_directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_promoted_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "candidate-fr"
    base_snapshot = tmp_path / "chronos2-base"
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (base_snapshot / "model.safetensors").write_bytes(b"frozen-base")
    checkpoint = bundle / "artifacts" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_snapshot.resolve())}),
        encoding="utf-8",
    )
    checkpoint_sha256 = sha256_directory(checkpoint)
    schema = {
        "format_version": 1,
        "timestamp_column": "timestamp",
        "origin_column": "origin_timestamp",
        "item_column": "item_id",
        "feature_available_at_column": "feature_available_at_utc",
        "target_columns": ["target"],
        "known_future_covariates": ["temperature"],
        "past_only_covariates": [],
        "timezone": "Europe/Paris",
        "cutoff_local_time": "08:00",
        "context_length": 4,
    }
    _write_json(bundle / "artifacts" / "schema" / "schema.json", schema)
    oof_audit = {
        "schema_version": 1,
        "purpose": "chronos2_exogenous_blocked_prequential_oof",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "candidate_model": "chronos2_exogenous",
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "training_days": 365,
        "training_start_day": "2025-01-01",
        "training_end_day": "2025-12-31",
        "holdout_start_day": "2026-01-01",
        "predictions_sha256": "a" * 64,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "future_actuals_used_as_features": False,
        "holdout_used_for_fit": False,
        "selection_frozen_before_oof": True,
    }
    oof_audit_path = bundle / "artifacts" / "oof_audit" / "oof_audit.json"
    _write_json(oof_audit_path, oof_audit)
    corrector = {
        "schema_version": 1,
        "model_kind": "linear_shift_v1",
        "base_model": "chronos2_exogenous",
        "output_model": "exogenous_residual_corrected",
        "fit_protocol": "blocked_prequential_oof_rolling365",
        "training_days": 365,
        "selection_frozen_before_holdout": True,
        "holdout_used_for_fit": False,
        "future_actuals_used_as_features": False,
        "oof_audit_required": True,
        "refit_uses_only_strictly_prior_days": True,
        "same_day_actual_excluded_from_fit": True,
        "oof_training_predictions_sha256": "a" * 64,
        "oof_training_audit_sha256": _sha256(oof_audit_path),
        "candidate_model": "chronos2_exogenous",
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "feature_columns": ["intercept"],
        "coefficients": [1.0],
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "maximum_absolute_shift_eur_mwh": 20.0,
    }
    corrector_path = (
        bundle / "artifacts" / "residual_corrector" / "residual_corrector.json"
    )
    _write_json(corrector_path, corrector)
    rolling = pd.DataFrame(
        {
            "delivery_start_utc": ["2026-01-01T00:00:00Z"],
            "forecast_origin_utc": ["2025-12-31T07:00:00Z"],
            "actual": [10.0],
            "baseline_q10": [7.0],
            "baseline_q50": [9.0],
            "baseline_q90": [11.0],
            "candidate_q10": [8.0],
            "candidate_q50": [10.0],
            "candidate_q90": [12.0],
        }
    )
    rolling_path = (
        bundle / "evidence" / "rolling365_predictions" / "evaluation.csv.gz"
    )
    rolling_path.parent.mkdir(parents=True)
    rolling.to_csv(rolling_path, index=False)
    experiment = {
        "experiment_id": "chronos2_exogenous_lora_poc_v1",
        "evaluation_role": "primary_predeclared",
        "production_pit_evidence": True,
        "production_pipeline_evidence": True,
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
        "base_model_snapshot_sha256": sha256_directory(base_snapshot),
        "checkpoint_sha256": checkpoint_sha256,
        "candidate_output_stage": "exogenous_residual_corrected",
        "residual_corrector_sha256": _sha256(corrector_path),
        "oof_training_audit_sha256": _sha256(oof_audit_path),
        "evaluation_evidence": {"sha256": _sha256(rolling_path)},
        "source_hashes": {
            "FR": {
                "deterministic_calendar": "b" * 64,
                "residual_load": "c" * 64,
            }
        },
        "source_audit_hashes": {"FR": {"residual_load": "d" * 64}},
        "source_cutoff_timezones": {
            "FR": {
                "deterministic_calendar": "Europe/Paris",
                "residual_load": "Europe/Paris",
            }
        },
        "panel_pack": "residual_only",
        "target_sources": {
            "FR": {"source_path": str((tmp_path / "canonical_target.csv").resolve())}
        },
        "target_contracts": {
            "FR": {
                "series": "power.price.da.fr.canonical",
                "cache_path": str((tmp_path / "canonical_target.csv").resolve()),
            }
        },
        "production_runtime": {
            "schema_version": 1,
            "layout": "per_zone",
            "cross_learning": False,
            "target": "target",
        },
    }
    _write_json(
        bundle / "evidence" / "experiment_manifest" / "experiment_manifest.json",
        experiment,
    )
    decision = {
        "decision": "promote",
        "production_pit_evidence": True,
        "production_pit_gate_passes": True,
        "production_pipeline_evidence": True,
        "production_pipeline_gate_passes": True,
        "shadow_evidence_verified": True,
        "rolling365_gate": {"passes": True},
        "live_shadow_gate": {"passes": True},
        "production_activation_performed": False,
    }
    _write_json(bundle / "promotion_decision.json", decision)
    _write_json(bundle / "artifact_checksums.json", {"algorithm": "sha256"})
    manifest = {
        "schema_version": 1,
        "candidate_id": bundle.name,
        "candidate_model": "chronos2_exogenous_lora_poc_v1",
        "zone": "FR",
        "decision": "promote",
    }
    _write_json(bundle / "bundle_manifest.json", manifest)
    return bundle, manifest


def _promoted_bundle_with_shadow(
    tmp_path: Path,
) -> tuple[PromotedBundle, dict[str, object]]:
    bundle_path, bundle_manifest = _fake_promoted_bundle(tmp_path)
    timezone_name = "Europe/Paris"
    delivery_day = date(2026, 2, 1)
    start = pd.Timestamp(delivery_day, tz=timezone_name)
    end = pd.Timestamp(delivery_day + timedelta(days=1), tz=timezone_name)
    delivery = pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    origin = pd.Timestamp(
        datetime.combine(delivery_day - timedelta(days=1), time(8)),
        tz=timezone_name,
    ).tz_convert("UTC")
    shadow = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origin,
            "actual": np.full(len(delivery), 10.0),
            "baseline_q10": np.full(len(delivery), 6.0),
            "baseline_q50": np.full(len(delivery), 8.0),
            "baseline_q90": np.full(len(delivery), 10.0),
            "candidate_q10": np.full(len(delivery), 7.0),
            "candidate_q50": np.full(len(delivery), 9.0),
            "candidate_q90": np.full(len(delivery), 11.0),
        }
    )
    shadow_path = (
        bundle_path
        / "evidence"
        / "shadow_predictions"
        / "shadow_final_evidence.csv.gz"
    )
    shadow_path.parent.mkdir(parents=True)
    shadow.to_csv(shadow_path, index=False)
    pending_day = delivery_day + timedelta(days=1)
    pending_start = pd.Timestamp(pending_day, tz=timezone_name)
    pending_end = pd.Timestamp(pending_day + timedelta(days=1), tz=timezone_name)
    pending_delivery = pd.date_range(
        pending_start.tz_convert("UTC"),
        pending_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    pending_origin = pd.Timestamp(
        datetime.combine(pending_day - timedelta(days=1), time(8)),
        tz=timezone_name,
    ).tz_convert("UTC")
    issued_delivery = delivery.append(pending_delivery)
    issued_origins = pd.DatetimeIndex(
        [origin] * len(delivery) + [pending_origin] * len(pending_delivery)
    )
    experiment = json.loads(
        (
            bundle_path
            / "evidence"
            / "experiment_manifest"
            / "experiment_manifest.json"
        ).read_text(encoding="utf-8")
    )
    issued = pd.DataFrame(
        {
            "delivery_start_utc": issued_delivery,
            "forecast_origin_utc": issued_origins,
            "actual": [10.0] * len(delivery) + [np.nan] * len(pending_delivery),
            "candidate_q10": [7.0] * len(delivery) + [10.0] * len(pending_delivery),
            "candidate_q50": [9.0] * len(delivery) + [12.0] * len(pending_delivery),
            "candidate_q90": [11.0] * len(delivery) + [14.0] * len(pending_delivery),
            "item_id": "FR",
            "target_column": "target",
            "checkpoint_sha256": experiment["checkpoint_sha256"],
            "input_contract_sha256": "1" * 64,
            "panel_contract_sha256": "2" * 64,
            "forecast_created_at_utc": issued_origins + pd.Timedelta(hours=1),
            "forecast_record_sha256": [
                hashlib.sha256(f"issued-{timestamp.isoformat()}".encode()).hexdigest()
                for timestamp in issued_delivery
            ],
        }
    )
    issued_path = (
        bundle_path
        / "evidence"
        / "shadow_lineage"
        / "shadow_final_issued_history.csv.gz"
    )
    issued_path.parent.mkdir(parents=True)
    issued.to_csv(issued_path, index=False, compression="gzip")
    temporal = [
        {
            "delivery_start_utc": timestamp.isoformat(),
            "forecast_origin_utc": origin.isoformat(),
        }
        for timestamp in delivery
    ]
    shadow_manifest = {
        "format_version": 4,
        "kind": "chronos2_exogenous_final_pipeline_shadow",
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
        "residual_corrector_applied": True,
        "predictions_sha256": _sha256(shadow_path),
        "observed_governance_evidence": {
            "relative_path": shadow_path.name,
            "sha256": _sha256(shadow_path),
            "rows": len(shadow),
        },
        "issued_shadow_history": {
            "relative_path": issued_path.name,
            "sha256": _sha256(issued_path),
            "rows": len(issued),
            "first_delivery_utc": issued_delivery[0].isoformat(),
            "last_delivery_utc": issued_delivery[-1].isoformat(),
            "delivery_days": [delivery_day.isoformat(), pending_day.isoformat()],
            "candidate_output_stage": "exogenous_residual_corrected",
            "actual_nullable": True,
        },
        "temporal_attachment_audit": temporal,
    }
    shadow_manifest_path = (
        bundle_path
        / "evidence"
        / "shadow_manifest"
        / "shadow_final_manifest.json"
    )
    shadow_manifest["residual_corrector"] = {
        "relative_path": "residual_corrector.json",
        "sha256": _sha256(
            bundle_path
            / "artifacts"
            / "residual_corrector"
            / "residual_corrector.json"
        ),
    }
    _write_json(shadow_manifest_path, shadow_manifest)
    lineage = bundle_path / "evidence" / "shadow_lineage"
    lineage.mkdir(exist_ok=True)
    (lineage / shadow_path.name).write_bytes(shadow_path.read_bytes())
    (lineage / shadow_manifest_path.name).write_bytes(
        shadow_manifest_path.read_bytes()
    )
    promoted = PromotedBundle(
        alias="fr-v1",
        bundle_path=bundle_path,
        candidate_id=str(bundle_manifest["candidate_id"]),
        candidate_model=str(bundle_manifest["candidate_model"]),
        zone="FR",
        bundle_manifest_sha256=_sha256(bundle_path / "bundle_manifest.json"),
        artifact_checksums_sha256=_sha256(
            bundle_path / "artifact_checksums.json"
        ),
        checkpoint_path=bundle_path / "artifacts" / "checkpoint",
        schema_path=bundle_path / "artifacts" / "schema" / "schema.json",
        experiment_manifest_path=(
            bundle_path
            / "evidence"
            / "experiment_manifest"
            / "experiment_manifest.json"
        ),
        residual_corrector_path=(
            bundle_path
            / "artifacts"
            / "residual_corrector"
            / "residual_corrector.json"
        ),
        oof_audit_path=(
            bundle_path / "artifacts" / "oof_audit" / "oof_audit.json"
        ),
        rolling_predictions_path=(
            bundle_path
            / "evidence"
            / "rolling365_predictions"
            / "evaluation.csv.gz"
        ),
    )
    return promoted, shadow_manifest


def _live_panel(
    tmp_path: Path,
    *,
    delivery_day: date = date(2026, 9, 3),
) -> tuple[Path, Path]:
    timezone = ZoneInfo("Europe/Paris")
    start = pd.Timestamp(datetime.combine(delivery_day, time()), tz=timezone).tz_convert(
        "UTC"
    )
    end = pd.Timestamp(
        datetime.combine(delivery_day + timedelta(days=1), time()), tz=timezone
    ).tz_convert("UTC")
    horizon = pd.date_range(start, end, inclusive="left", freq="h")
    context = pd.date_range(start - pd.Timedelta(hours=4), periods=4, freq="h")
    timestamps = context.append(horizon)
    origin = pd.Timestamp(
        datetime.combine(delivery_day - timedelta(days=1), time(8)), tz=timezone
    ).tz_convert("UTC")
    panel = pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": origin,
            "item_id": "FR",
            "feature_available_at_utc": origin,
            "phase": ["context"] * 4 + ["horizon"] * 24,
            "delivery_day": delivery_day.isoformat(),
            "target": [1.0, 2.0, 3.0, 4.0] + [np.nan] * 24,
            "temperature": np.linspace(5.0, 10.0, 28),
        }
    )
    panel_path = tmp_path / "live_panel.csv.gz"
    panel.to_csv(panel_path, index=False)
    target_path = tmp_path / "canonical_target.csv"
    pd.DataFrame(
        {"timestamp": context, "value": [1.0, 2.0, 3.0, 4.0]}
    ).to_csv(target_path, index=False)
    audit_path = tmp_path / "live_panel.audit.json"
    _write_json(
        audit_path,
        {
            "panel_sha256": _sha256(panel_path),
            "purpose": "prospective_shadow",
            "layout": "per_zone",
            "zones": ["FR"],
            "created_at_utc": (origin + pd.Timedelta(hours=1)).isoformat(),
            "forecast_origin_utc": origin.isoformat(),
            "forecast_origin_timezone": "Europe/Paris",
            "delivery_day": delivery_day.isoformat(),
            "delivery_timezones": {"FR": "Europe/Paris"},
            "pack": "residual_only",
            "horizon_actuals_present": False,
            "first_publication_with_actuals_must_be_rejected": True,
            "production_ready": True,
            "production_pit_evidence": {"FR": True},
            "exogenous_banks": {
                "FR": {
                    "production_ready": True,
                    "forecast_origin_timezone": "Europe/Paris",
                    "source_hashes": {
                        "deterministic_calendar": "b" * 64,
                        "residual_load": "c" * 64,
                    },
                    "source_audit_hashes": {"residual_load": "d" * 64},
                    "production_pit_evidence": {
                        "deterministic_calendar": True,
                        "residual_load": True,
                    },
                    "source_cutoff_timezones": {
                        "deterministic_calendar": "Europe/Paris",
                        "residual_load": "Europe/Paris",
                    },
                }
            },
            "target_sources": {
                "FR": {
                    "source_path": str(target_path.resolve()),
                    "source_sha256": _sha256(target_path),
                    "first_timestamp": context.min().isoformat(),
                    "last_timestamp": context.max().isoformat(),
                }
            },
            "canonical_target_contracts_verified": True,
            "target_contracts": {
                "FR": {
                    "series": "power.price.da.fr.canonical",
                    "cache_path": str(target_path.resolve()),
                }
            },
        },
    )
    return panel_path, audit_path


def _reseal_live_panel(panel_path: Path, audit_path: Path, panel: pd.DataFrame) -> None:
    panel.to_csv(panel_path, index=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["panel_sha256"] = _sha256(panel_path)
    _write_json(audit_path, audit)


class _FakePipeline:
    def predict(self, *_: object, **__: object) -> object:
        raise AssertionError("Le runtime ne doit pas interpreter des samples comme quantiles.")

    def predict_quantiles(
        self,
        inputs: object,
        *,
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs: object,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert len(inputs) == 1
        assert quantile_levels == [0.1, 0.5, 0.9]
        assert kwargs["cross_learning"] is False
        assert kwargs["limit_prediction_length"] is False
        values = np.empty((1, prediction_length, 3), dtype=float)
        values[:, :, 0] = 9.0
        values[:, :, 1] = 10.0
        values[:, :, 2] = 11.0
        return [values], [values[:, :, 1]]


def test_shadow_bundle_is_never_registerable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    rejected = dict(manifest)
    rejected["decision"] = "shadow"
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: rejected)

    with pytest.raises(ExogenousProductionError, match="non promu"):
        validate_promoted_bundle(bundle, alias="fr-v1")


def test_promoted_shadow_history_uses_final_candidate_without_double_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, shadow_manifest = _promoted_bundle_with_shadow(tmp_path)
    monkeypatch.setattr(
        production,
        "verify_promotion_bundle",
        lambda _path: {
            "candidate_id": bundle.candidate_id,
            "candidate_model": bundle.candidate_model,
            "zone": bundle.zone,
            "decision": "promote",
        },
    )
    monkeypatch.setattr(
        production,
        "validate_final_shadow_manifest",
        lambda manifest, **_kwargs: json.loads(
            Path(manifest).read_text(encoding="utf-8")
        ),
    )

    result = load_promoted_shadow_history(bundle)

    assert result.delivery_days == ("2026-02-01", "2026-02-02")
    assert result.predictions_sha256 == shadow_manifest["predictions_sha256"]
    assert result.issued_history_sha256 == shadow_manifest[
        "issued_shadow_history"
    ]["sha256"]
    assert "chronos2_exogenous__q50" not in result.predictions
    local_days = result.predictions["delivery_start_utc"].dt.tz_convert(
        "Europe/Paris"
    ).dt.date
    assert result.predictions.loc[
        local_days.eq(date(2026, 2, 1)),
        "exogenous_residual_corrected__q50",
    ].eq(9.0).all()
    assert result.predictions.loc[
        local_days.eq(date(2026, 2, 2)),
        "exogenous_residual_corrected__q50",
    ].eq(12.0).all()


def test_promoted_shadow_history_refuses_a_prediction_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _shadow_manifest = _promoted_bundle_with_shadow(tmp_path)
    monkeypatch.setattr(
        production,
        "verify_promotion_bundle",
        lambda _path: {
            "candidate_id": bundle.candidate_id,
            "candidate_model": bundle.candidate_model,
            "zone": bundle.zone,
            "decision": "promote",
        },
    )
    manifest_path = (
        bundle.bundle_path
        / "evidence"
        / "shadow_manifest"
        / "shadow_final_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    _write_json(
        bundle.bundle_path
        / "evidence"
        / "shadow_lineage"
        / "shadow_final_manifest.json",
        manifest,
    )
    monkeypatch.setattr(
        production,
        "validate_final_shadow_manifest",
        lambda raw, **_kwargs: json.loads(
            Path(raw).read_text(encoding="utf-8")
        ),
    )

    with pytest.raises(ExogenousProductionError, match="SHA des predictions shadow"):
        load_promoted_shadow_history(bundle)


def test_promoted_shadow_history_refuses_raw_v3_adapter_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _shadow_manifest = _promoted_bundle_with_shadow(tmp_path)
    monkeypatch.setattr(
        production,
        "verify_promotion_bundle",
        lambda _path: {
            "candidate_id": bundle.candidate_id,
            "candidate_model": bundle.candidate_model,
            "zone": bundle.zone,
            "decision": "promote",
        },
    )
    manifest_path = (
        bundle.bundle_path
        / "evidence"
        / "shadow_manifest"
        / "shadow_final_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "format_version": 3,
            "kind": "chronos2_exogenous_append_only_shadow",
            "candidate_output_stage": "chronos2_exogenous",
            "residual_corrector_applied": False,
            "kind": "chronos2_exogenous_append_only_shadow",
            "candidate_output_stage": "chronos2_exogenous",
            "residual_corrector_applied": False,
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        bundle.bundle_path
        / "evidence"
        / "shadow_lineage"
        / "shadow_final_manifest.json",
        manifest,
    )
    monkeypatch.setattr(
        production,
        "validate_final_shadow_manifest",
        lambda raw, **_kwargs: json.loads(
            Path(raw).read_text(encoding="utf-8")
        ),
    )

    with pytest.raises(ExogenousProductionError, match="pipeline final"):
        load_promoted_shadow_history(bundle)


def test_raw_pipeline_bundle_is_never_registerable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    decision_path = bundle / "promotion_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["production_pipeline_evidence"] = False
    decision["production_pipeline_gate_passes"] = False
    _write_json(decision_path, decision)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)

    with pytest.raises(ExogenousProductionError, match="production_pipeline"):
        validate_promoted_bundle(bundle, alias="fr-v1")


def test_raw_comparison_cannot_be_relabelled_as_final_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    experiment_path = (
        bundle / "evidence" / "experiment_manifest" / "experiment_manifest.json"
    )
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["production_pipeline_evidence_detail"][
        "baseline_output_stage"
    ] = "chronos2_base_raw"
    _write_json(experiment_path, experiment)

    with pytest.raises(ExogenousProductionError, match="pipeline final invalide"):
        validate_promoted_bundle(bundle, alias="fr-v1")


def test_diagnostic_evaluation_cannot_be_used_for_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    experiment_path = (
        bundle / "evidence" / "experiment_manifest" / "experiment_manifest.json"
    )
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["evaluation_role"] = "diagnostic_only"
    _write_json(experiment_path, experiment)

    with pytest.raises(ExogenousProductionError, match="primary_predeclared"):
        validate_promoted_bundle(bundle, alias="fr-v1")


def test_promoted_bundle_rechecks_exact_base_snapshot_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    adapter = json.loads(
        (bundle / "artifacts" / "checkpoint" / "adapter_config.json").read_text(
            encoding="utf-8"
        )
    )
    base = Path(adapter["base_model_name_or_path"])
    (base / "model.safetensors").write_bytes(b"tampered-base")

    with pytest.raises(ExogenousProductionError, match="differe de celui utilise"):
        validate_promoted_bundle(bundle, alias="fr-v1")


def test_register_is_inactive_and_rechecks_registry_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"

    registered = register_promoted_bundle(
        bundle, registry_path=registry, alias="fr-v1", expected_zone="FR"
    )
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert registered.zone == "FR"
    assert payload["entries"]["fr-v1"]["enabled_by_default"] is False
    assert payload["entries"]["fr-v1"]["incumbent_modified"] is False
    assert load_registered_bundle(registry_path=registry, alias="fr-v1").candidate_id == bundle.name

    (bundle / "bundle_manifest.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ExogenousProductionError, match="a change"):
        load_registered_bundle(registry_path=registry, alias="fr-v1")


def test_explicit_run_publishes_consumable_corrected_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)

    loader_kwargs: dict[str, object] = {}

    def load_pipeline(_path: Path, kwargs: dict[str, object]) -> _FakePipeline:
        loader_kwargs.update(kwargs)
        return _FakePipeline()

    result = run_registered_candidate(
        registry_path=registry,
        alias="fr-v1",
        live_panel_path=panel,
        live_panel_audit_path=audit,
        delivery_day="2026-09-03",
        output_directory=tmp_path / "output",
        pipeline_loader=load_pipeline,
    )

    forecast = pd.read_csv(result.forecast_path)
    assert len(forecast) == 24
    assert forecast["chronos2_exogenous__q50"].eq(10.0).all()
    assert forecast["exogenous_residual_corrected__q50"].eq(11.0).all()
    assert forecast["residual_shift_eur_mwh"].eq(1.0).all()
    assert result.backtest_path.is_file()
    assert (result.output_directory / "artifact_checksums.json").is_file()
    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["explicit_opt_in_required"] is True
    assert run_manifest["launcher_mode_agnostic"] is True
    assert "included_in_mode_all" not in run_manifest
    assert run_manifest["incumbent_modified"] is False
    assert loader_kwargs["import_allowlist"] == ["chronos.chronos2.model"]


def test_sealed_daily_candidate_can_be_reused_but_never_silently_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    result = run_registered_candidate(
        registry_path=registry,
        alias="fr-v1",
        live_panel_path=panel,
        live_panel_audit_path=audit,
        delivery_day="2026-09-03",
        output_directory=tmp_path / "output",
        pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
    )

    reused = validate_registered_candidate_run(
        registry_path=registry,
        alias="fr-v1",
        output_directory=result.output_directory,
        delivery_day="2026-09-03",
        expected_zone="FR",
    )
    assert reused.forecast_path == result.forecast_path
    assert (reused.output_directory / "live_panel.csv.gz").is_file()
    assert (reused.output_directory / "live_panel.audit.json").is_file()

    forecast = pd.read_csv(reused.forecast_path)
    forecast.loc[0, "exogenous_residual_corrected__q50"] += 1.0
    forecast.to_csv(reused.forecast_path, index=False)
    with pytest.raises(ExogenousProductionError, match="modifie"):
        validate_registered_candidate_run(
            registry_path=registry,
            alias="fr-v1",
            output_directory=result.output_directory,
            delivery_day="2026-09-03",
            expected_zone="FR",
        )


def test_live_panel_with_future_actuals_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel_path, original_audit = _live_panel(tmp_path)
    panel = pd.read_csv(panel_path)
    panel.loc[panel["phase"].eq("horizon"), "target"] = 99.0
    panel.to_csv(panel_path, index=False)
    audit = tmp_path / "new.audit.json"
    audit_payload = json.loads(original_audit.read_text(encoding="utf-8"))
    audit_payload["panel_sha256"] = _sha256(panel_path)
    _write_json(audit, audit_payload)

    with pytest.raises(ExogenousProductionError, match="cibles futures"):
        run_registered_candidate(
            registry_path=registry,
            alias="fr-v1",
            live_panel_path=panel_path,
            live_panel_audit_path=audit,
            delivery_day="2026-09-03",
            output_directory=tmp_path / "forbidden",
            pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
        )


def test_live_panel_without_every_source_sidecar_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["exogenous_banks"]["FR"]["source_audit_hashes"] = {}
    _write_json(audit, audit_payload)

    with pytest.raises(ExogenousProductionError, match="source_audit_hashes"):
        run_registered_candidate(
            registry_path=registry,
            alias="fr-v1",
            live_panel_path=panel,
            live_panel_audit_path=audit,
            delivery_day="2026-09-03",
            output_directory=tmp_path / "missing-source-audit",
            pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
        )


def test_live_panel_with_mutated_canonical_target_cache_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    target_path = tmp_path / "canonical_target.csv"
    target_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ExogenousProductionError, match="SHA du panel"):
        run_registered_candidate(
            registry_path=registry,
            alias="fr-v1",
            live_panel_path=panel,
            live_panel_audit_path=audit,
            delivery_day="2026-09-03",
            output_directory=tmp_path / "mutated-target",
            pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
        )


def test_live_runtime_accepts_append_only_canonical_target_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    target_path = tmp_path / "canonical_target.csv"
    current = pd.read_csv(target_path)
    following = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(current["timestamp"], utc=True).max()
                + pd.Timedelta(hours=1)
            ],
            "value": [999.0],
        }
    )
    pd.concat([current, following], ignore_index=True).to_csv(target_path, index=False)

    result = run_registered_candidate(
        registry_path=registry,
        alias="fr-v1",
        live_panel_path=panel,
        live_panel_audit_path=audit,
        delivery_day="2026-09-03",
        output_directory=tmp_path / "append-only-output",
        pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
    )

    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    verification = run_manifest["target_cache_verification"]
    assert verification["mode"] == "panel_target_equivalence"
    assert verification["verified_timestamps"] == 4
    assert verification["max_abs_delta"] == 0.0


def test_live_runtime_rejects_revision_of_sealed_target_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    target_path = tmp_path / "canonical_target.csv"
    current = pd.read_csv(target_path)
    current.loc[0, "value"] = float(current.loc[0, "value"]) + 1.0
    current.to_csv(target_path, index=False)

    with pytest.raises(ExogenousProductionError, match="valeurs utilisees"):
        run_registered_candidate(
            registry_path=registry,
            alias="fr-v1",
            live_panel_path=panel,
            live_panel_audit_path=audit,
            delivery_day="2026-09-03",
            output_directory=tmp_path / "revised-output",
            pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
        )


def test_live_runtime_accepts_market_cent_equivalent_target_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _fake_promoted_bundle(tmp_path)
    monkeypatch.setattr(production, "verify_promotion_bundle", lambda _: manifest)
    registry = tmp_path / "registry.json"
    register_promoted_bundle(bundle, registry_path=registry, alias="fr-v1")
    panel, audit = _live_panel(tmp_path)
    target_path = tmp_path / "canonical_target.csv"
    current = pd.read_csv(target_path)
    current.loc[0, "value"] = float(current.loc[0, "value"]) + 0.0025
    current.to_csv(target_path, index=False)

    result = run_registered_candidate(
        registry_path=registry,
        alias="fr-v1",
        live_panel_path=panel,
        live_panel_audit_path=audit,
        delivery_day="2026-09-03",
        output_directory=tmp_path / "rounding-output",
        pipeline_loader=lambda _path, _kwargs: _FakePipeline(),
    )

    run_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    verification = run_manifest["target_cache_verification"]
    assert verification["rounding_equivalent_timestamps"] == 1
    assert verification["max_abs_delta"] == pytest.approx(0.0025)


def _oof_year(tmp_path: Path) -> Path:
    timezone = ZoneInfo("Europe/Paris")
    parts: list[pd.DataFrame] = []
    first = date(2025, 1, 1)
    for offset in range(365):
        day = first + timedelta(days=offset)
        start = pd.Timestamp(day, tz=timezone).tz_convert("UTC")
        end = pd.Timestamp(day + timedelta(days=1), tz=timezone).tz_convert("UTC")
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        origin_day = day - timedelta(days=1)
        origin = pd.Timestamp(f"{origin_day:%Y-%m-%d} 08:00", tz=timezone)
        parts.append(pd.DataFrame({
            "delivery_start_utc": hours,
            "forecast_origin_utc": origin.tz_convert("UTC"),
            "actual": 12.0,
            "candidate_q50": 10.0,
        }))
    path = tmp_path / "oof.csv.gz"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return path


def _oof_audit(source: Path) -> Path:
    audit = source.with_suffix(source.suffix + ".audit.json")
    _write_json(
        audit,
        {
            "schema_version": 1,
            "purpose": "chronos2_exogenous_blocked_prequential_oof",
            "fit_protocol": "blocked_prequential_oof_rolling365",
            "candidate_model": "chronos2_exogenous",
            "candidate_checkpoint_sha256": "f" * 64,
            "training_days": 365,
            "training_start_day": "2025-01-01",
            "training_end_day": "2025-12-31",
            "holdout_start_day": "2026-01-01",
            "predictions_sha256": _sha256(source),
            "refit_uses_only_strictly_prior_days": True,
            "same_day_actual_excluded_from_fit": True,
            "future_actuals_used_as_features": False,
            "holdout_used_for_fit": False,
            "selection_frozen_before_oof": True,
        },
    )
    return audit


def test_fit_corrector_uses_exact_pre_holdout_oof_year(tmp_path: Path) -> None:
    source = _oof_year(tmp_path)
    output = fit_oof_residual_corrector(
        oof_predictions_path=source,
        oof_audit_path=_oof_audit(source),
        holdout_start_day="2026-01-01",
        output_path=tmp_path / "corrector.json",
        feature_columns=["intercept"],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_days"] == 365
    assert payload["training_end_day"] == "2025-12-31"
    assert payload["feature_means"] == [0.0]
    assert payload["feature_scales"] == [1.0]
    assert payload["coefficients"][0] == pytest.approx(2.0)
    assert payload["fit_mae_after"] == pytest.approx(0.0)


def test_fit_corrector_rejects_holdout_overlap(tmp_path: Path) -> None:
    source = _oof_year(tmp_path)
    with pytest.raises(ExogenousProductionError, match="chevauche"):
        fit_oof_residual_corrector(
            oof_predictions_path=source,
            oof_audit_path=_oof_audit(source),
            holdout_start_day="2025-12-31",
            output_path=tmp_path / "forbidden.json",
            feature_columns=["intercept"],
        )


def test_fit_corrector_rejects_label_as_feature_and_unbound_audit(
    tmp_path: Path,
) -> None:
    source = _oof_year(tmp_path)
    audit = _oof_audit(source)
    with pytest.raises(ExogenousProductionError, match="Features label/prediction"):
        fit_oof_residual_corrector(
            oof_predictions_path=source,
            oof_audit_path=audit,
            holdout_start_day="2026-01-01",
            output_path=tmp_path / "leaky.json",
            feature_columns=["actual"],
        )

    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["predictions_sha256"] = "0" * 64
    _write_json(audit, payload)
    with pytest.raises(ExogenousProductionError, match="pas lie"):
        fit_oof_residual_corrector(
            oof_predictions_path=source,
            oof_audit_path=audit,
            holdout_start_day="2026-01-01",
            output_path=tmp_path / "unbound.json",
            feature_columns=["intercept"],
        )
