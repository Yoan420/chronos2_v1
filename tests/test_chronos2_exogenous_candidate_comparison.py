from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.candidate_comparison import (
    COMPARISON_DAILY_NAME,
    COMPARISON_JSON_NAME,
    COMPARISON_REPORT_NAME,
    CandidateComparisonError,
    compare_final_candidates,
)
from chronos2_exogenous.evaluation import compute_metrics
from chronos2_exogenous.final_pipeline import (
    FINAL_AUDIT_NAME,
    FINAL_EVIDENCE_NAME,
    FINAL_MANIFEST_NAME,
    FINAL_METRICS_NAME,
    FINAL_REPORT_NAME,
)
from chronos2_exogenous.lora_finetune import EVALUATION_COLUMNS
from chronos2_exogenous.production import BASE_MODEL, OUTPUT_MODEL, RUNTIME_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = PROJECT_ROOT / "config" / "chronos2_exogenous_promotion_v1.yaml"


def _fast_policy(tmp_path: Path) -> Path:
    path = tmp_path / "comparison-policy.yaml"
    if not path.exists():
        path.write_text(
            POLICY.read_text(encoding="utf-8").replace(
                "bootstrap_samples: 20000", "bootstrap_samples: 200"
            ),
            encoding="utf-8",
        )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timeline(start: str = "2025-01-01") -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    deliveries: list[pd.Timestamp] = []
    origins: list[pd.Timestamp] = []
    for day in pd.date_range(start, periods=365, freq="D"):
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


def _make_final_proof(
    root: Path,
    *,
    rank: int,
    candidate_error: float,
    zone: str = "FR",
    start: str = "2025-01-01",
    actual_delta: float = 0.0,
    origin_delta_hours: int = 0,
    production_pit: bool = True,
    canonical_audit: bool = True,
) -> Path:
    run = root / f"rank{rank}"
    final = run / "final_pipeline"
    final.mkdir(parents=True)
    delivery, origins = _timeline(start)
    origins = origins + pd.Timedelta(hours=origin_delta_hours)
    phase = np.arange(len(delivery), dtype=float)
    actual = (
        50.0
        + 12.0 * np.sin(phase * 2.0 * np.pi / (24.0 * 30.0))
        + actual_delta
    )
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origins,
            "actual": actual,
            "baseline_q10": actual - 6.0,
            "baseline_q50": actual + 3.0,
            "baseline_q90": actual + 10.0,
            "candidate_q10": actual - 5.0,
            "candidate_q50": actual + candidate_error,
            "candidate_q90": actual + 5.0,
        },
        columns=EVALUATION_COLUMNS,
    )
    predictions = final / FINAL_EVIDENCE_NAME
    frame.to_csv(predictions, index=False, compression="gzip")
    metrics, _ = compute_metrics(frame, timezone_name="Europe/Paris")
    metrics.update(
        {
            "comparison_scope": "paired_operational_final_pipelines",
            "baseline_label": "incumbent",
            "candidate_label": "LoRA corrected",
        }
    )
    metrics_path = final / FINAL_METRICS_NAME
    _write_json(metrics_path, metrics)
    report = final / FINAL_REPORT_NAME
    report.write_text("<html>final</html>", encoding="utf-8")
    experiment_id = f"candidate-rank-{rank}"
    checkpoint_sha = ("8" if rank == 8 else "f") * 64
    raw_predictions = run / "evaluation_predictions.csv.gz"
    frame.to_csv(raw_predictions, index=False, compression="gzip")
    incumbent = run / "incumbent_statistics.csv.gz"
    pd.DataFrame(
        {
            "delivery_start_utc": frame["delivery_start_utc"],
            "forecast_origin_utc": frame["forecast_origin_utc"],
            "actual": frame["actual"],
            "residual_corrected__q10": frame["baseline_q10"],
            "residual_corrected__q50": frame["baseline_q50"],
            "residual_corrected__q90": frame["baseline_q90"],
        }
    ).to_csv(incumbent, index=False, compression="gzip")
    holdout_start = pd.Timestamp(start).date()
    training_end = holdout_start - pd.Timedelta(days=1)
    training_start = training_end - pd.Timedelta(days=364)
    oof_predictions_sha = "d" * 64
    oof_audit = run / "oof_predictions_365.csv.gz.audit.json"
    _write_json(
        oof_audit,
        {
            "schema_version": 1,
            "purpose": "chronos2_exogenous_blocked_prequential_oof",
            "fit_protocol": "blocked_prequential_oof_rolling365",
            "candidate_model": BASE_MODEL,
            "candidate_checkpoint_sha256": checkpoint_sha,
            "training_days": 365,
            "training_start_day": training_start.isoformat(),
            "training_end_day": training_end.isoformat(),
            "holdout_start_day": holdout_start.isoformat(),
            "predictions_sha256": oof_predictions_sha,
            "refit_uses_only_strictly_prior_days": True,
            "same_day_actual_excluded_from_fit": True,
            "future_actuals_used_as_features": False,
            "holdout_used_for_fit": False,
            "selection_frozen_before_oof": True,
        },
    )
    corrector = run / "residual_corrector.json"
    _write_json(
        corrector,
        {
            "schema_version": 1,
            "model_kind": "linear_shift_v1",
            "base_model": BASE_MODEL,
            "output_model": OUTPUT_MODEL,
            "fit_protocol": "blocked_prequential_oof_rolling365",
            "training_days": 365,
            "training_start_day": training_start.isoformat(),
            "training_end_day": training_end.isoformat(),
            "holdout_start_day": holdout_start.isoformat(),
            "selection_frozen_before_holdout": True,
            "holdout_used_for_fit": False,
            "future_actuals_used_as_features": False,
            "oof_audit_required": True,
            "refit_uses_only_strictly_prior_days": True,
            "same_day_actual_excluded_from_fit": True,
            "oof_training_predictions_sha256": oof_predictions_sha,
            "oof_training_audit_sha256": _sha256(oof_audit),
            "oof_sidecar_schema_version": 1,
            "candidate_model": BASE_MODEL,
            "candidate_checkpoint_sha256": checkpoint_sha,
            "feature_columns": ["intercept"],
            "feature_means": [0.0],
            "feature_scales": [1.0],
            "coefficients": [0.0],
            "maximum_absolute_shift_eur_mwh": 20.0,
        },
    )
    schema = run / "schema.json"
    _write_json(
        schema,
        {
            "format_version": 1,
            "timestamp_column": "timestamp",
            "target_columns": ["target"],
            "timezone": "Europe/Paris",
        },
    )
    detail = {
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": OUTPUT_MODEL,
        "paired_same_input_contract": True,
        "paired_same_evaluation_window": True,
        "baseline_residual_corrector_applied": True,
        "candidate_residual_corrector_applied": True,
        "rolling_evaluation_days": 365,
        "promotion_eligible": True,
        "paired_delivery_hours": len(frame),
        "paired_forecast_origins": True,
        "paired_actuals": True,
    }
    audit = final / FINAL_AUDIT_NAME
    if canonical_audit:
        _write_json(
            audit,
            {
                "schema_version": 1,
                "purpose": "chronos2_exogenous_final_pipeline_rolling365",
                "zone": zone,
                "raw_candidate_evidence": {
                    "relative_path": raw_predictions.name,
                    "sha256": _sha256(raw_predictions),
                },
                "incumbent_statistics": {
                    "path": str(incumbent),
                    "sha256": _sha256(incumbent),
                    "model": "residual_corrected",
                    "source_rows": len(frame),
                    "paired_rows": len(frame),
                },
                "residual_corrector": {
                    "path": str(corrector),
                    "sha256": _sha256(corrector),
                    "fit_protocol": "blocked_prequential_oof_rolling365",
                    "holdout_used_for_fit": False,
                },
                "oof_training_audit": {
                    "path": str(oof_audit),
                    "sha256": _sha256(oof_audit),
                },
                "checks": {
                    "bundle_verified": True,
                    "exact_365_physical_days": True,
                    "exact_8760_physical_hours": True,
                    "continuous_hourly_utc": True,
                    "dst_days_verified": True,
                    "same_delivery_timeline": True,
                    "same_forecast_origins": True,
                    "same_actuals": True,
                    "actual_pairing_atol_eur_mwh": 5e-5,
                    "actual_pairing_max_delta_eur_mwh": 0.0,
                    "corrector_strictly_pre_holdout_oof": True,
                    "production_pit_evidence_unchanged": True,
                },
                "production_pit_evidence_value": production_pit,
                "production_pipeline_evidence_detail": detail,
                "final_evidence_sha256": _sha256(predictions),
                "metrics": {
                    key: metrics[key]
                    for key in (
                        "baseline_mae_eur_mwh",
                        "candidate_mae_eur_mwh",
                        "mae_gain_eur_mwh",
                    )
                },
            },
        )
    else:
        _write_json(audit, {"rank": rank})
    final_manifest = {
        "schema_version": 1,
        "kind": "chronos2_exogenous_final_pipeline_evaluation",
        "zone": zone,
        "experiment_id": experiment_id,
        "comparison_scope": "paired_operational_final_pipelines",
        "baseline_output_stage": "residual_corrected",
        "candidate_output_stage": OUTPUT_MODEL,
        "production_pipeline_evidence": True,
        "production_pipeline_evidence_detail": detail,
        "window": {
            "physical_days": 365,
            "physical_hours": len(frame),
            "first_delivery_utc": delivery[0].isoformat(),
            "last_delivery_utc": delivery[-1].isoformat(),
        },
        "artifacts": {
            "predictions": {
                "relative_path": predictions.name,
                "sha256": _sha256(predictions),
            },
            "metrics": {
                "relative_path": metrics_path.name,
                "sha256": _sha256(metrics_path),
            },
            "report": {
                "relative_path": report.name,
                "sha256": _sha256(report),
            },
            "audit": {
                "relative_path": audit.name,
                "sha256": _sha256(audit),
            },
        },
    }
    final_manifest_path = final / FINAL_MANIFEST_NAME
    _write_json(final_manifest_path, final_manifest)
    experiment = {
        "experiment_id": experiment_id,
        "zone": zone,
        "model_id": "amazon/chronos-2",
        "evaluation_role": "primary_predeclared",
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
        "checkpoint_sha256": checkpoint_sha,
        "schema_sha256": _sha256(schema),
        "candidate_output_stage": OUTPUT_MODEL,
        "production_pit_evidence": production_pit,
        "production_pipeline_evidence": True,
        "production_pipeline_evidence_detail": detail,
        "production_runtime": {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "layout": "per_zone",
            "cross_learning": False,
            "target": "target",
        },
        "training": {"lora_config": {"r": rank}},
        "raw_evaluation_evidence": {
            "relative_path": raw_predictions.name,
            "sha256": _sha256(raw_predictions),
        },
        "evaluation_evidence": {
            "relative_path": f"final_pipeline/{FINAL_EVIDENCE_NAME}",
            "sha256": _sha256(predictions),
        },
        "residual_corrector_sha256": _sha256(corrector),
        "oof_training_audit_sha256": _sha256(oof_audit),
        "final_pipeline_evaluation": {
            "relative_path": f"final_pipeline/{FINAL_MANIFEST_NAME}",
            "sha256": _sha256(final_manifest_path),
            "audit_relative_path": f"final_pipeline/{FINAL_AUDIT_NAME}",
            "audit_sha256": _sha256(audit),
            "metrics_relative_path": f"final_pipeline/{FINAL_METRICS_NAME}",
            "metrics_sha256": _sha256(metrics_path),
            "incumbent_statistics_sha256": _sha256(incumbent),
        },
    }
    _write_json(run / "experiment_manifest.json", experiment)
    return run


def _make_raw_proof(
    root: Path,
    *,
    rank: int,
    candidate_error: float,
    input_contract: str = "a" * 64,
) -> Path:
    run = root / f"raw-rank{rank}"
    run.mkdir(parents=True)
    delivery, origins = _timeline()
    phase = np.arange(len(delivery), dtype=float)
    actual = 50.0 + 12.0 * np.sin(phase * 2.0 * np.pi / (24.0 * 30.0))
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": origins,
            "actual": actual,
            "baseline_q10": actual - 6.0,
            "baseline_q50": actual + 3.0,
            "baseline_q90": actual + 10.0,
            "candidate_q10": actual - 5.0,
            "candidate_q50": actual + candidate_error,
            "candidate_q90": actual + 5.0,
        },
        columns=EVALUATION_COLUMNS,
    )
    predictions = run / "evaluation_predictions.csv.gz"
    frame.to_csv(predictions, index=False, compression="gzip")
    metrics, daily = compute_metrics(frame, timezone_name="Europe/Paris")
    metrics_path = run / "evaluation_metrics.json"
    _write_json(metrics_path, metrics)
    daily_path = run / "evaluation_daily.csv.gz"
    daily.to_csv(daily_path, index=False, compression="gzip")
    report = run / "evaluation_report.html"
    report.write_text("<html>raw</html>", encoding="utf-8")
    experiment_id = f"raw-candidate-rank-{rank}"
    experiment = {
        "experiment_id": experiment_id,
        "production_pipeline_evidence": False,
        "training": {"lora_config": {"r": rank}},
        "evaluation_evidence": {
            "relative_path": predictions.name,
            "sha256": _sha256(predictions),
        },
    }
    experiment_path = run / "experiment_manifest.json"
    _write_json(experiment_path, experiment)
    manifest = {
        "format_version": 1,
        "kind": "chronos2_exogenous_lora_rolling365_evaluation",
        "experiment_id": experiment_id,
        "evaluation_role": "primary_predeclared",
        "item_id": "FR",
        "input_contract_sha256": input_contract,
        "comparison": {
            "same_inputs": True,
            "cross_learning": False,
            "residual_corrector_applied": False,
        },
        "window": {
            "physical_days": 365,
            "physical_hours": len(frame),
            "first_delivery_utc": delivery[0].isoformat(),
            "last_delivery_utc": delivery[-1].isoformat(),
        },
        "artifacts": {
            "evidence": {
                "relative_path": predictions.name,
                "sha256": _sha256(predictions),
            },
            "metrics": {
                "relative_path": metrics_path.name,
                "sha256": _sha256(metrics_path),
            },
            "daily": {
                "relative_path": daily_path.name,
                "sha256": _sha256(daily_path),
            },
            "report": {
                "relative_path": report.name,
                "sha256": _sha256(report),
            },
        },
        "bundle_manifest_sha256": _sha256(experiment_path),
    }
    _write_json(run / "evaluation_manifest.json", manifest)
    return run


def test_compare_selects_rank16_only_when_full_governance_gate_passes(
    tmp_path: Path,
) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)
    rank8_hash = _sha256(rank8 / "final_pipeline" / FINAL_EVIDENCE_NAME)
    rank16_hash = _sha256(rank16 / "final_pipeline" / FINAL_EVIDENCE_NAME)

    result = compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16 / "final_pipeline" / FINAL_MANIFEST_NAME,
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "comparison",
        expected_zone="FR",
    )

    assert result.winner == "rank16"
    assert result.decision == "select_rank16"
    assert result.comparison_path.name == COMPARISON_JSON_NAME
    assert result.report_path.name == COMPARISON_REPORT_NAME
    assert result.daily_path.name == COMPARISON_DAILY_NAME
    payload = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    assert payload["governance"]["gate"]["passes"] is True
    assert payload["candidate_selection_eligible"] is True
    assert payload["selection_readiness"] == {"passes": True, "blockers": []}
    assert payload["promotion_performed"] is False
    assert payload["activation_performed"] is False
    assert payload["paired_metrics"]["baseline_mae_eur_mwh"] == pytest.approx(2.0)
    assert payload["paired_metrics"]["candidate_mae_eur_mwh"] == pytest.approx(1.0)
    assert len(pd.read_csv(result.daily_path)) == 365
    assert "Rang 16 retenu" in result.report_path.read_text(encoding="utf-8")
    assert _sha256(rank8 / "final_pipeline" / FINAL_EVIDENCE_NAME) == rank8_hash
    assert _sha256(rank16 / "final_pipeline" / FINAL_EVIDENCE_NAME) == rank16_hash


def test_compare_retains_rank8_when_rank16_gain_is_below_policy_threshold(
    tmp_path: Path,
) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.99)

    result = compare_final_candidates(
        rank8_source=rank8 / "final_pipeline" / FINAL_METRICS_NAME,
        rank16_source=rank16 / "final_pipeline" / FINAL_EVIDENCE_NAME,
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "comparison",
    )

    assert result.winner == "rank8"
    assert result.decision == "retain_rank8"
    gate = result.comparison["governance"]["gate"]
    assert gate["passes"] is False
    assert gate["checks"]["mae_gain"]["passes"] is False
    assert result.comparison["candidate_selection_eligible"] is True


def test_final_comparison_with_pit_false_is_research_only(tmp_path: Path) -> None:
    rank8 = _make_final_proof(
        tmp_path, rank=8, candidate_error=2.0, production_pit=False
    )
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)

    result = compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16,
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "comparison",
    )

    payload = result.comparison
    assert payload["evidence_stage"] == "final_corrected"
    assert payload["winner"] == "rank16"
    assert payload["decision"] == "research_preference_rank16"
    assert payload["candidate_selection_eligible"] is False
    assert payload["promotion_eligible"] is False
    assert payload["selection_readiness"]["passes"] is False
    blockers = " | ".join(payload["selection_readiness"]["blockers"])
    assert "rang 8: production_pit_evidence n'est pas true" in blockers
    assert "rang 8: audit final: preuve PIT production absente" in blockers
    assert "Rang 16 retenu" not in result.report_path.read_text(encoding="utf-8")
    assert "Avantage exploratoire au rang 16" in result.report_path.read_text(
        encoding="utf-8"
    )


def test_final_comparison_with_fake_audits_cannot_select(tmp_path: Path) -> None:
    rank8 = _make_final_proof(
        tmp_path, rank=8, candidate_error=2.0, canonical_audit=False
    )
    rank16 = _make_final_proof(
        tmp_path, rank=16, candidate_error=1.0, canonical_audit=False
    )

    result = compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16,
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "comparison",
    )

    assert result.decision == "research_preference_rank16"
    assert result.comparison["candidate_selection_eligible"] is False
    blockers = " | ".join(result.comparison["selection_readiness"]["blockers"])
    assert "rang 8: audit final: identite/purpose non canonique" in blockers
    assert "rang 16: audit final: checks absents" in blockers


def test_mutated_corrector_downgrades_final_comparison_to_research(
    tmp_path: Path,
) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)
    (rank16 / "residual_corrector.json").write_text("{}\n", encoding="utf-8")

    result = compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16,
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "comparison",
    )

    assert result.decision == "research_preference_rank16"
    assert result.comparison["candidate_selection_eligible"] is False
    blockers = " | ".join(result.comparison["selection_readiness"]["blockers"])
    assert "rang 16: audit final: residual_corrector absent ou modifie" in blockers
    assert "rang 16: chaine correcteur/OOF non reproductible" in blockers


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"actual_delta": 0.01}, "actual differe"),
        ({"origin_delta_hours": -1}, "forecast_origin_utc differe"),
        ({"start": "2025-01-02"}, "delivery_start_utc differe"),
        ({"zone": "DE"}, "Zones divergentes"),
    ],
)
def test_compare_refuses_unpaired_proofs(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(
        tmp_path, rank=16, candidate_error=1.0, **change
    )

    with pytest.raises(CandidateComparisonError, match=message):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=tmp_path / "comparison",
        )
    assert not (tmp_path / "comparison").exists()


def test_compare_refuses_swapped_rank_metadata(tmp_path: Path) -> None:
    alleged_rank8 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)
    alleged_rank16 = _make_final_proof(
        tmp_path / "other", rank=8, candidate_error=2.0
    )

    with pytest.raises(CandidateComparisonError, match="annonce rang 8"):
        compare_final_candidates(
            rank8_source=alleged_rank8,
            rank16_source=alleged_rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=tmp_path / "comparison",
        )


def test_compare_refuses_tampered_final_artifact(tmp_path: Path) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)
    (rank16 / "final_pipeline" / FINAL_METRICS_NAME).write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(CandidateComparisonError, match="SHA-256 divergent"):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=tmp_path / "comparison",
        )


def test_compare_output_is_atomic_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)
    output = tmp_path / "comparison"
    compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16,
        policy_path=_fast_policy(tmp_path),
        output_directory=output,
    )
    marker = output / "user-marker.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(CandidateComparisonError, match="overwrite explicite"):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=output,
        )
    assert marker.read_text(encoding="utf-8") == "preserve"

    replaced = compare_final_candidates(
        rank8_source=rank8,
        rank16_source=rank16,
        policy_path=_fast_policy(tmp_path),
        output_directory=output,
        overwrite=True,
    )
    assert replaced.comparison_path.is_file()
    assert not marker.exists()


def test_compare_refuses_output_inside_a_candidate_bundle(tmp_path: Path) -> None:
    rank8 = _make_final_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)

    with pytest.raises(CandidateComparisonError, match="distinct"):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=rank16 / "comparison",
        )


def test_compare_accepts_two_raw_proofs_but_marks_result_non_promotable(
    tmp_path: Path,
) -> None:
    rank8 = _make_raw_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_raw_proof(tmp_path, rank=16, candidate_error=1.0)

    result = compare_final_candidates(
        rank8_source=rank8 / "evaluation_manifest.json",
        rank16_source=rank16 / "evaluation_predictions.csv.gz",
        policy_path=_fast_policy(tmp_path),
        output_directory=tmp_path / "raw-comparison",
    )

    assert result.winner == "rank16"
    assert result.decision == "research_preference_rank16"
    payload = result.comparison
    assert payload["evidence_stage"] == "raw_lora"
    assert payload["candidate_selection_eligible"] is False
    assert payload["promotion_eligible"] is False
    assert payload["promotion_performed"] is False
    assert payload["activation_performed"] is False
    assert len(payload["limitations"]) == 3
    report = result.report_path.read_text(encoding="utf-8")
    assert "résultat exploratoire" in report
    assert "Sélection opérationnelle interdite" in report
    assert "deux preuves finales corrigées" in report


def test_compare_raw_requires_same_input_contract(tmp_path: Path) -> None:
    rank8 = _make_raw_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_raw_proof(
        tmp_path, rank=16, candidate_error=1.0, input_contract="b" * 64
    )

    with pytest.raises(CandidateComparisonError, match="input_contract_sha256"):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=tmp_path / "raw-comparison",
        )


def test_compare_refuses_mixing_raw_and_final_proofs(tmp_path: Path) -> None:
    rank8 = _make_raw_proof(tmp_path, rank=8, candidate_error=2.0)
    rank16 = _make_final_proof(tmp_path, rank=16, candidate_error=1.0)

    with pytest.raises(CandidateComparisonError, match="Stages incompatibles"):
        compare_final_candidates(
            rank8_source=rank8,
            rank16_source=rank16,
            policy_path=_fast_policy(tmp_path),
            output_directory=tmp_path / "comparison",
        )
