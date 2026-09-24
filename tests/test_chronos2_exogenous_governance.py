from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.governance import (
    ExogenousGovernanceError,
    GovernancePolicy,
    evaluate_promotion,
    seal_promotion_bundle,
    validate_experiment_manifest,
    validate_shadow_manifest,
    verify_promotion_bundle,
)


def _policy(**updates: object) -> GovernancePolicy:
    return replace(
        GovernancePolicy(bootstrap_samples=300, bootstrap_seed=7), **updates
    ).validate()


def _manifest(
    zone: str = "FR",
    *,
    production_pit_evidence: bool = True,
    production_pipeline_evidence: bool = True,
) -> dict[str, object]:
    return {
        "model_id": "chronos2_exogenous_lora_v1",
        "experiment_id": "chronos2_exogenous_lora_v1",
        "evaluation_role": "primary_predeclared",
        "zone": zone,
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
        "production_pit_evidence": production_pit_evidence,
        "production_pipeline_evidence": production_pipeline_evidence,
        "candidate_output_stage": "exogenous_residual_corrected",
        "residual_corrector_sha256": "d" * 64,
        "oof_training_audit_sha256": "e" * 64,
        "evaluation_evidence": {"sha256": "7" * 64},
        "final_pipeline_evaluation": {"sha256": "8" * 64},
        "production_pipeline_evidence_detail": {
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_output_stage": "residual_corrected",
            "candidate_output_stage": "exogenous_residual_corrected",
            "baseline_residual_corrector_applied": True,
            "candidate_residual_corrector_applied": True,
            "rolling_evaluation_days": 365,
            "promotion_eligible": True,
        },
        "checkpoint_sha256": "a" * 64,
        "schema_sha256": "b" * 64,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _shadow_manifest(
    experiment: dict[str, object],
    predictions: pd.DataFrame,
    zone: str = "FR",
) -> dict[str, object]:
    temporal = []
    panels: dict[str, dict[str, object]] = {}
    for index, row in predictions.reset_index(drop=True).iterrows():
        origin = pd.Timestamp(row["forecast_origin_utc"])
        created = origin + pd.Timedelta(hours=1)
        attached = origin + pd.Timedelta(hours=5)
        delivery_day = pd.Timestamp(row["delivery_start_utc"]).tz_convert(
            "Europe/Paris"
        ).date().isoformat()
        panel_payload: dict[str, object] = {
            "schema_version": 1,
            "purpose": "prospective_shadow",
            "panel_sha256": hashlib.sha256(
                f"panel-{origin.isoformat()}".encode()
            ).hexdigest(),
            "panel_audit_sha256": hashlib.sha256(
                f"audit-{origin.isoformat()}".encode()
            ).hexdigest(),
            "panel_created_at_utc": (
                origin + pd.Timedelta(minutes=30)
            ).isoformat(),
            "zone": zone,
            "delivery_day": delivery_day,
            "forecast_origin_utc": origin.isoformat(),
            "forecast_origin_timezone": "Europe/Paris",
            "delivery_timezone": "Europe/Paris",
            "pack": "residual_only",
            "production_pit_evidence": bool(
                experiment["production_pit_evidence"]
            ),
            "source_hashes": {
                "deterministic_calendar": "d" * 64,
                "residual_load": "e" * 64,
            },
            "source_audit_hashes": {"residual_load": "f" * 64},
            "source_cutoff_timezones": {
                "deterministic_calendar": "Europe/Paris",
                "residual_load": "Europe/Paris",
            },
            "target_source_sha256": "1" * 64,
            "horizon_actuals_present": False,
        }
        panel_contract = hashlib.sha256(
            json.dumps(
                panel_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        panel_payload["panel_contract_sha256"] = panel_contract
        panels[panel_contract] = panel_payload
        temporal.append(
            {
                "delivery_start_utc": pd.Timestamp(
                    row["delivery_start_utc"]
                ).isoformat(),
                "forecast_origin_utc": origin.isoformat(),
                "forecast_deadline_utc": (
                    origin + pd.Timedelta(hours=4)
                ).isoformat(),
                "forecast_created_at_utc": created.isoformat(),
                "actual_attached_at_utc": attached.isoformat(),
                "forecast_record_sha256": hashlib.sha256(
                    f"forecast-{index}".encode()
                ).hexdigest(),
                "resolution_record_sha256": hashlib.sha256(
                    f"actual-{index}".encode()
                ).hexdigest(),
                "panel_contract_sha256": panel_contract,
            }
        )
    return {
        "format_version": 3,
        "kind": "chronos2_exogenous_append_only_shadow",
        "candidate_model": experiment["experiment_id"],
        "candidate_output_stage": "chronos2_exogenous",
        "residual_corrector_applied": False,
        "zone": zone,
        "checkpoint_sha256": experiment["checkpoint_sha256"],
        "schema_sha256": experiment["schema_sha256"],
        "candidate_frozen_before_shadow": True,
        "actuals_attached_after_forecast_freeze": True,
        "prospective_capture_deadline_enforced": True,
        "prospective_capture_deadline_hours": 4,
        "forecast_artifact_checksums_valid": True,
        "storm_used_for_prediction": False,
        "mkonline_used_for_prediction": False,
        "predictions_sha256": "c" * 64,
        "forecast_created_at_utc": min(
            record["forecast_created_at_utc"] for record in temporal
        ),
        "actual_attached_at_utc": max(
            record["actual_attached_at_utc"] for record in temporal
        ),
        "shadow_panel_production_ready": bool(
            experiment["production_pit_evidence"]
        ),
        "shadow_panel_evidence": [panels[key] for key in sorted(panels)],
        "temporal_attachment_audit": temporal,
    }


def _final_shadow_manifest(
    experiment: dict[str, object],
    predictions: pd.DataFrame,
    zone: str = "FR",
) -> dict[str, object]:
    raw = _shadow_manifest(experiment, predictions, zone)
    rows = len(predictions)
    hashes = {
        "predictions": "c" * 64,
        "raw_evidence": "9" * 64,
        "raw_manifest": "0" * 64,
        "raw_journal": "1" * 64,
        "incumbent": "2" * 64,
        "issued": "3" * 64,
        "experiment": "4" * 64,
    }
    first_delivery = pd.Timestamp(predictions["delivery_start_utc"].iloc[0])
    last_delivery = pd.Timestamp(predictions["delivery_start_utc"].iloc[-1])
    days = list(
        dict.fromkeys(
            pd.DatetimeIndex(predictions["delivery_start_utc"])
            .tz_convert("Europe/Paris")
            .date.astype(str)
        )
    )
    lineage = {
        "raw_shadow_evidence_sha256": hashes["raw_evidence"],
        "raw_shadow_manifest_sha256": hashes["raw_manifest"],
        "raw_shadow_journal_sha256": hashes["raw_journal"],
        "paired_incumbent_sha256": hashes["incumbent"],
        "issued_shadow_history_sha256": hashes["issued"],
        "residual_corrector_sha256": experiment["residual_corrector_sha256"],
        "oof_training_audit_sha256": experiment["oof_training_audit_sha256"],
        "rolling365_final_evidence_sha256": experiment["evaluation_evidence"][
            "sha256"
        ],
        "rolling365_final_manifest_sha256": experiment[
            "final_pipeline_evaluation"
        ]["sha256"],
        "checkpoint_sha256": experiment["checkpoint_sha256"],
        "schema_sha256": experiment["schema_sha256"],
        "final_shadow_predictions_sha256": hashes["predictions"],
        "rows": rows,
        "issued_rows": rows,
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
    }
    return {
        **raw,
        "format_version": 4,
        "kind": "chronos2_exogenous_final_pipeline_shadow",
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": "exogenous_residual_corrected",
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "residual_corrector_applied": True,
        "paired_same_delivery_hours": True,
        "paired_same_forecast_origins": True,
        "paired_same_observed_actuals": True,
        "actual_pairing_atol_eur_mwh": 5e-5,
        "actual_pairing_max_delta_eur_mwh": 0.0,
        "predictions_sha256": hashes["predictions"],
        "source_experiment_manifest_sha256": hashes["experiment"],
        "observed_governance_evidence": {
            "relative_path": "shadow_final_evidence.csv.gz",
            "sha256": hashes["predictions"],
            "rows": rows,
        },
        "raw_shadow_evidence": {
            "relative_path": "raw_shadow_observed_evidence.csv.gz",
            "sha256": hashes["raw_evidence"],
            "rows": rows,
        },
        "raw_shadow_manifest": {
            "relative_path": "raw_shadow_manifest.json",
            "sha256": hashes["raw_manifest"],
            "format_version": 3,
        },
        "raw_shadow_journal": {
            "relative_path": "raw_shadow_journal.csv.gz",
            "sha256": hashes["raw_journal"],
            "records": rows,
            "last_record_sha256": "5" * 64,
        },
        "issued_shadow_history": {
            "relative_path": "shadow_final_issued_history.csv.gz",
            "sha256": hashes["issued"],
            "rows": rows,
            "first_delivery_utc": first_delivery.isoformat(),
            "last_delivery_utc": last_delivery.isoformat(),
            "delivery_days": days,
            "candidate_output_stage": "exogenous_residual_corrected",
            "actual_nullable": True,
        },
        "paired_incumbent_evidence": {
            "relative_path": "shadow_final_incumbent.csv.gz",
            "sha256": hashes["incumbent"],
            "rows": rows,
            "output_stage": "residual_corrected",
        },
        "residual_corrector": {
            "relative_path": "residual_corrector.json",
            "sha256": experiment["residual_corrector_sha256"],
        },
        "oof_training_audit": {
            "relative_path": "oof_audit.json",
            "sha256": experiment["oof_training_audit_sha256"],
        },
        "schema": {
            "relative_path": "schema.json",
            "sha256": experiment["schema_sha256"],
        },
        "source_experiment_manifest": {
            "relative_path": "experiment_manifest.json",
            "sha256": hashes["experiment"],
        },
        "rolling365_final_pipeline_evidence": {
            "sha256": experiment["evaluation_evidence"]["sha256"],
            "manifest_sha256": experiment["final_pipeline_evaluation"]["sha256"],
        },
        "residual_corrector_sha256": experiment["residual_corrector_sha256"],
        "oof_training_audit_sha256": experiment["oof_training_audit_sha256"],
        "transformation": {
            "algorithm": "sealed_residual_corrector_linear_shift_v1",
            "raw_candidate_output_stage": "chronos2_exogenous",
            "output_stage": "exogenous_residual_corrected",
            "mean_shift_eur_mwh": 0.0,
            "maximum_absolute_shift_eur_mwh": 0.0,
            "issued_mean_shift_eur_mwh": 0.0,
            "issued_maximum_absolute_shift_eur_mwh": 0.0,
        },
        "derivation_lineage": lineage,
        "derivation_sha256": _canonical_sha(lineage),
    }


def _evidence(
    *,
    end_day: date,
    days: int,
    candidate_offset: float = 1.0,
    baseline_offset: float = 2.0,
) -> pd.DataFrame:
    timezone = ZoneInfo("Europe/Paris")
    indexes: list[pd.DatetimeIndex] = []
    origins: list[pd.Timestamp] = []
    for offset in range(days):
        day_local = end_day - timedelta(days=days - 1 - offset)
        start = pd.Timestamp(datetime.combine(day_local, time()), tz=timezone)
        end = pd.Timestamp(
            datetime.combine(day_local + timedelta(days=1), time()), tz=timezone
        )
        index = pd.date_range(
            start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
        )
        indexes.append(index)
        cutoff = pd.Timestamp(
            datetime.combine(day_local - timedelta(days=1), time(8)), tz=timezone
        ).tz_convert("UTC")
        origins.extend([cutoff] * len(index))
    delivery = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
    hour = delivery.tz_convert(timezone).hour.to_numpy(dtype=float)
    actual = 70.0 + 15.0 * np.sin(2.0 * np.pi * hour / 24.0)
    baseline = actual + baseline_offset
    candidate = actual + candidate_offset
    return pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origins,
            "actual": actual,
            "baseline_q10": baseline - 5.0,
            "baseline_q50": baseline,
            "baseline_q90": baseline + 5.0,
            "candidate_q10": candidate - 5.0,
            "candidate_q50": candidate,
            "candidate_q90": candidate + 5.0,
        }
    )


def test_rolling365_pass_only_authorises_shadow() -> None:
    evidence = _evidence(end_day=date(2026, 8, 31), days=365)
    decision = evaluate_promotion(
        rolling_predictions=evidence,
        experiment_manifest=_manifest(),
        zone="FR",
        policy=_policy(),
    )

    assert decision.decision == "shadow"
    assert decision.rolling365.days == 365
    assert decision.rolling365.hours == 8760
    assert decision.rolling365.mae_gain_eur_mwh == pytest.approx(1.0)
    assert decision.rolling365_gate.passes is True
    assert decision.live_shadow is None
    assert decision.production_activation_performed is False


def test_two_phase_governance_requires_canonical_epoch_and_manifest(
    tmp_path: Path,
) -> None:
    evidence = _evidence(end_day=date(2026, 8, 31), days=365)
    manifest = _manifest()
    manifest["evaluation_label_binding"] = {
        "allow_unresolved_final_evaluation_day": True
    }

    with pytest.raises(ExogenousGovernanceError, match="shadow_epoch_directory"):
        evaluate_promotion(
            rolling_predictions=evidence,
            experiment_manifest=manifest,
            zone="FR",
            policy=_policy(),
        )
    with pytest.raises(ExogenousGovernanceError, match="chemin canonique"):
        evaluate_promotion(
            rolling_predictions=evidence,
            experiment_manifest=manifest,
            zone="FR",
            policy=_policy(),
            shadow_epoch_directory=tmp_path / "epoch",
        )


def test_live_shadow_passes_and_bundle_is_self_contained(tmp_path: Path) -> None:
    rolling = _evidence(end_day=date(2026, 8, 31), days=365)
    shadow = _evidence(end_day=date(2026, 9, 30), days=30)
    rolling_path = tmp_path / "rolling.csv.gz"
    rolling.to_csv(rolling_path, index=False)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"sealed-lora")
    schema = tmp_path / "schema.json"
    schema.write_text('{"features":["wind","ram"]}\n', encoding="utf-8")
    experiment = _manifest()
    experiment["checkpoint_sha256"] = _checkpoint_sha256(checkpoint)
    experiment["schema_sha256"] = _sha256(schema)
    experiment["evaluation_evidence"] = {"sha256": _sha256(rolling_path)}
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(experiment), encoding="utf-8")

    decision = evaluate_promotion(
        rolling_predictions=rolling,
        shadow_predictions=shadow,
        experiment_manifest=manifest_path,
        shadow_manifest=_final_shadow_manifest(experiment, shadow),
        zone="FR",
        policy=_policy(),
    )
    assert decision.decision == "promote"
    assert decision.live_shadow_gate is not None
    assert decision.live_shadow_gate.passes is True

    bundle_policy = _policy(shadow_required_for_promotion=False)
    bundle_decision = evaluate_promotion(
        rolling_predictions=rolling,
        experiment_manifest=manifest_path,
        zone="FR",
        policy=bundle_policy,
    )
    bundle = seal_promotion_bundle(
        output_root=tmp_path / "promotion",
        rolling_predictions_path=rolling_path,
        experiment_manifest_path=manifest_path,
        decision=bundle_decision,
        policy=bundle_policy,
        candidate_artifacts={"checkpoint": checkpoint, "schema": schema},
    )
    verified = verify_promotion_bundle(bundle)
    assert verified["decision"] == "promote"
    assert (bundle / "artifacts" / "checkpoint" / "adapter_model.safetensors").is_file()
    assert (bundle / "artifacts" / "schema" / "schema.json").is_file()
    assert verified["deployment"]["activation_performed"] is False

    (bundle / "rogue.txt").write_text("unsealed", encoding="utf-8")
    with pytest.raises(ExogenousGovernanceError, match="non scelles"):
        verify_promotion_bundle(bundle)


def test_worse_candidate_is_rejected() -> None:
    evidence = _evidence(
        end_day=date(2026, 8, 31),
        days=365,
        candidate_offset=3.0,
        baseline_offset=2.0,
    )
    decision = evaluate_promotion(
        rolling_predictions=evidence,
        experiment_manifest=_manifest(),
        zone="FR",
        policy=_policy(),
    )
    assert decision.decision == "reject"
    assert decision.rolling365_gate.passes is False
    assert "mae_gain" in decision.rolling365_gate.checks


def test_dst_hour_missing_fails_closed() -> None:
    evidence = _evidence(end_day=date(2026, 8, 31), days=365)
    local = evidence["delivery_start_utc"].dt.tz_convert("Europe/Paris")
    spring_day = local.dt.date.eq(date(2026, 3, 29))
    evidence = evidence.drop(evidence.index[spring_day][0])
    with pytest.raises(ExogenousGovernanceError, match="grille DST incomplete"):
        evaluate_promotion(
            rolling_predictions=evidence,
            experiment_manifest=_manifest(),
            zone="FR",
            policy=_policy(),
        )


def test_post_cutoff_origin_and_storm_selection_are_rejected() -> None:
    evidence = _evidence(end_day=date(2026, 8, 31), days=365)
    evidence.loc[0, "forecast_origin_utc"] = (
        pd.Timestamp(evidence.loc[0, "forecast_origin_utc"]) + pd.Timedelta(minutes=1)
    )
    with pytest.raises(ExogenousGovernanceError, match="posterieure"):
        evaluate_promotion(
            rolling_predictions=evidence,
            experiment_manifest=_manifest(),
            zone="FR",
            policy=_policy(),
        )

    leaky = _manifest()
    leaky["storm_used_for_selection"] = True
    with pytest.raises(ExogenousGovernanceError, match="storm_used_for_selection"):
        validate_experiment_manifest(leaky, zone="FR")


def test_diagnostic_ablation_cannot_enter_governance() -> None:
    diagnostic = _manifest()
    diagnostic["evaluation_role"] = "diagnostic_only"
    with pytest.raises(ExogenousGovernanceError, match="diagnostic_only"):
        evaluate_promotion(
            rolling_predictions=_evidence(
                end_day=date(2026, 8, 31), days=365
            ),
            experiment_manifest=diagnostic,
            zone="FR",
            policy=_policy(),
        )


def test_current_non_production_jao_evidence_blocks_promotion() -> None:
    rolling = _evidence(end_day=date(2026, 8, 31), days=365)
    shadow = _evidence(end_day=date(2026, 9, 30), days=30)
    experiment = _manifest(production_pit_evidence=False)
    decision = evaluate_promotion(
        rolling_predictions=rolling,
        shadow_predictions=shadow,
        shadow_manifest=_final_shadow_manifest(experiment, shadow),
        experiment_manifest=experiment,
        zone="FR",
        policy=_policy(),
    )

    assert decision.rolling365_gate.passes is True
    assert decision.live_shadow_gate is not None
    assert decision.live_shadow_gate.passes is True
    assert decision.production_pit_evidence is False
    assert decision.production_pit_gate_passes is False
    assert decision.decision == "shadow"
    assert any("interdit la promotion" in reason for reason in decision.reasons)


def test_shadow_must_start_the_day_after_the_final_rolling_window() -> None:
    rolling = _evidence(end_day=date(2026, 8, 31), days=365)
    # Starts on 2 September, leaving 1 September unaccounted for.
    shadow = _evidence(end_day=date(2026, 10, 1), days=30)
    experiment = _manifest()

    with pytest.raises(ExogenousGovernanceError, match="Continuite rolling365/shadow"):
        evaluate_promotion(
            rolling_predictions=rolling,
            shadow_predictions=shadow,
            shadow_manifest=_final_shadow_manifest(experiment, shadow),
            experiment_manifest=experiment,
            zone="FR",
            policy=_policy(),
        )


def test_raw_lora_pipeline_evidence_blocks_promotion() -> None:
    rolling = _evidence(end_day=date(2026, 8, 31), days=365)
    experiment = _manifest(production_pipeline_evidence=False)
    decision = evaluate_promotion(
        rolling_predictions=rolling,
        experiment_manifest=experiment,
        zone="FR",
        policy=_policy(),
    )

    assert decision.rolling365_gate.passes is True
    assert decision.live_shadow_gate is None
    assert decision.production_pipeline_evidence is False
    assert decision.production_pipeline_gate_passes is False
    assert decision.decision == "shadow"
    assert any("pipeline final" in reason for reason in decision.reasons)


def test_shadow_governance_recomputes_panel_and_source_audit_identity() -> None:
    experiment = _manifest()
    shadow = _evidence(end_day=date(2026, 9, 2), days=2)
    manifest = _shadow_manifest(experiment, shadow)
    manifest["shadow_panel_evidence"][0]["source_audit_hashes"][
        "residual_load"
    ] = "0" * 64

    with pytest.raises(ExogenousGovernanceError, match="panel_contract_sha256 divergent"):
        validate_shadow_manifest(
            manifest,
            experiment_manifest=experiment,
            zone="FR",
            expected_rows=len(shadow),
        )
