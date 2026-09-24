from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.lora_finetune import (
    EVALUATION_COLUMNS,
    ExogenousFineTuneConfig,
)
from chronos2_exogenous.production import (
    ExogenousProductionError,
    _validate_corrector,
)
from chronos2_exogenous import research_validation_corrector as research


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class _FakePipeline:
    def predict_quantiles(self, inputs, *, prediction_length, **_kwargs):
        outputs = []
        for payload in inputs:
            targets = np.asarray(payload["target"]).shape[0]
            values = np.empty((targets, prediction_length, 3), dtype=float)
            values[:, :, 0] = 45.0
            values[:, :, 1] = 50.0
            values[:, :, 2] = 55.0
            outputs.append(values)
        return outputs, None


@pytest.fixture(autouse=True)
def _verified_checkpoint_test_double(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests CPU-only while exercising the non-injectable publish API."""

    monkeypatch.setattr(
        research,
        "load_checkpoint",
        lambda *_args, **_kwargs: _FakePipeline(),
    )


def _fixture(tmp_path: Path) -> tuple[ExogenousFineTuneConfig, Path, dict[str, object]]:
    run = tmp_path / "candidate"
    run.mkdir()
    schema = run / "schema.json"
    audit = tmp_path / "panel.parquet.audit.json"
    audit.write_text("{}\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    validation_origins: list[pd.Timestamp] = []
    for day in pd.date_range("2024-01-01", periods=30, freq="D"):
        origin = (day + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
        validation_origins.append(origin.tz_convert("UTC"))
        delivery_day = day + pd.Timedelta(days=1)
        delivery = pd.date_range(
            delivery_day.tz_localize("Europe/Paris").tz_convert("UTC"),
            (delivery_day + pd.Timedelta(days=1))
            .tz_localize("Europe/Paris")
            .tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        context = pd.date_range(
            end=delivery[0] - pd.Timedelta(hours=1), periods=2, freq="h"
        )
        for timestamp in context.append(delivery):
            rows.append(
                {
                    "timestamp": timestamp,
                    "origin_timestamp": origin,
                    "item_id": "FR",
                    "feature_available_at_utc": origin,
                    "target": 55.0 if timestamp in delivery else 40.0,
                }
            )
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(panel, index=False)
    first_holdout = pd.Timestamp("2024-01-31 08:00", tz="Europe/Paris")
    last_holdout = pd.Timestamp("2025-01-29 08:00", tz="Europe/Paris")
    manifest: dict[str, object] = {
        "format_version": 1,
        "experiment_id": "rank-test",
        "evaluation_role": "diagnostic_only",
        "finetune_mode": "lora",
        "model_id": "fake",
        "model_revision": None,
        "cutoff_local_time": "08:00",
        "training_window_days": 365,
        "evaluation_days": 365,
        "production_pit_evidence": False,
        "candidate_frozen_before_evaluation": True,
        "feature_selection_frozen_before_evaluation": True,
        "production_pipeline_evidence": False,
        "panel_sha256": _sha(panel),
        "panel_audit_sha256": _sha(audit),
        "schema_sha256": "0" * 64,
        "checkpoint_sha256": "c" * 64,
        "zone": "FR",
        "training": {
            "learning_rate": 1e-5,
            "num_steps": 1,
            "batch_size": 4,
            "seed": 42,
            "lora_config": {"r": 8},
        },
        "splits": {
            "validation": {
                "count": 30,
                "first_utc": validation_origins[0].isoformat(),
                "last_utc": validation_origins[-1].isoformat(),
            },
            "evaluation_holdout": {
                "count": 365,
                "first_utc": first_holdout.isoformat(),
                "last_utc": last_holdout.isoformat(),
            },
        },
    }
    config = ExogenousFineTuneConfig(
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        experiment_id="rank-test",
        evaluation_role="diagnostic_only",
        panel_path=panel,
        panel_audit_path=audit,
        output_directory=run,
        timestamp_column="timestamp",
        origin_column="origin_timestamp",
        item_column="item_id",
        feature_available_at_column="feature_available_at_utc",
        target_columns=("target",),
        known_future_covariates=(),
        past_only_covariates=(),
        timezone="Europe/Paris",
        cutoff_local_time="08:00",
        frequency="h",
        context_length=2,
        prediction_length=24,
        training_window_days=365,
        validation_days=30,
        evaluation_days=365,
        require_consecutive_origins=True,
        require_complete_known_future=True,
        production_pit_evidence=False,
        model_id="fake",
        model_revision=None,
        local_files_only=True,
        device_map="cpu",
        learning_rate=1e-5,
        num_steps=1,
        batch_size=4,
        seed=42,
        lora_config={"r": 8},
    )
    _write_json(schema, research._schema_contract(config))
    manifest["schema_sha256"] = _sha(schema)
    _write_json(run / "experiment_manifest.json", manifest)
    return config, run, manifest


def _holdout() -> pd.DataFrame:
    deliveries: list[pd.Timestamp] = []
    origins: list[pd.Timestamp] = []
    for day in pd.date_range("2024-02-01", periods=365, freq="D"):
        start = day.tz_localize("Europe/Paris")
        end = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
        horizon = pd.date_range(
            start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
        )
        origin = (
            day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
        ).tz_localize("Europe/Paris").tz_convert("UTC")
        deliveries.extend(horizon)
        origins.extend([origin] * len(horizon))
    count = len(deliveries)
    return pd.DataFrame(
        {
            "delivery_start_utc": deliveries,
            "forecast_origin_utc": origins,
            "actual": np.full(count, 55.0),
            "baseline_q10": np.full(count, 44.0),
            "baseline_q50": np.full(count, 49.0),
            "baseline_q90": np.full(count, 54.0),
            "candidate_q10": np.full(count, 45.0),
            "candidate_q50": np.full(count, 50.0),
            "candidate_q90": np.full(count, 55.0),
        },
        columns=EVALUATION_COLUMNS,
    )


def _publish_holdout(
    run: Path,
    manifest: dict[str, object],
    raw: pd.DataFrame,
    config: ExogenousFineTuneConfig,
) -> None:
    raw_path = run / "evaluation_predictions.csv.gz"
    raw.to_csv(raw_path, index=False, compression="gzip")
    metrics, daily = research.compute_metrics(raw, timezone_name=config.timezone)
    metrics_path = run / "evaluation_metrics.json"
    _write_json(metrics_path, metrics)
    daily_path = run / "evaluation_daily.csv.gz"
    daily.to_csv(daily_path, index=False, compression="gzip")
    report_path = run / "evaluation_report.html"
    report_path.write_text("<!doctype html><title>raw</title>\n", encoding="utf-8")
    manifest["evaluation_evidence"] = {
        "relative_path": raw_path.name,
        "sha256": _sha(raw_path),
        "rows": len(raw),
        "physical_days": 365,
        "first_delivery_utc": metrics["first_delivery_utc"],
        "last_delivery_utc": metrics["last_delivery_utc"],
        "dst_days": metrics["dst_days"],
    }
    _write_json(run / "experiment_manifest.json", manifest)
    evaluation_manifest = {
        "format_version": 1,
        "created_at_utc": "2025-02-01T00:00:00+00:00",
        "kind": "chronos2_exogenous_lora_rolling365_evaluation",
        "experiment_id": manifest["experiment_id"],
        "evaluation_role": manifest["evaluation_role"],
        "item_id": "FR",
        "target_column": "target",
        "comparison": {
            "baseline": research.BASELINE_LABEL,
            "candidate": research.CANDIDATE_LABEL,
            "same_inputs": True,
            "cross_learning": False,
            "residual_corrector_applied": False,
        },
        "window": {
            "physical_days": 365,
            "physical_hours": len(raw),
            "first_delivery_utc": metrics["first_delivery_utc"],
            "last_delivery_utc": metrics["last_delivery_utc"],
            "dst_days": metrics["dst_days"],
        },
        "input_contract_sha256": "d" * 64,
        "evaluation_label_resolution": manifest.get("evaluation_label_resolution"),
        "bundle_manifest_sha256": _sha(run / "experiment_manifest.json"),
        "artifacts": {
            "evidence": {
                "relative_path": raw_path.name,
                "sha256": _sha(raw_path),
            },
            "metrics": {
                "relative_path": metrics_path.name,
                "sha256": _sha(metrics_path),
            },
            "daily": {
                "relative_path": daily_path.name,
                "sha256": _sha(daily_path),
            },
            "report": {
                "relative_path": report_path.name,
                "sha256": _sha(report_path),
            },
        },
    }
    _write_json(run / "evaluation_manifest.json", evaluation_manifest)


def test_fit_reads_validation_only_and_seals_non_promotable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    # A poison holdout artifact may exist, but the fit phase must never decode it.
    (run / "evaluation_predictions.csv.gz").write_bytes(b"must-not-be-opened")
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail("fit opened holdout evidence"),
    )
    result = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-fit",
        item_id="FR",
        batch_size=4,
        inference_chunk_size=30,
    )
    payload = json.loads(result.corrector_path.read_text(encoding="utf-8"))
    fit_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["calibration_days"] == 30
    assert payload["calibration_predictions_are_not_oof"] is True
    assert payload["validation_used_for_checkpoint_monitoring_and_selection"] is True
    assert payload["holdout_used_for_fit"] is False
    assert payload["holdout_actuals_used_for_fit"] is False
    assert payload["returned_holdout_rows"] is False
    assert payload["promotion_eligible"] is False
    assert payload["fit_mae_before_eur_mwh"] == pytest.approx(5.0)
    assert payload["fit_mae_after_eur_mwh"] == pytest.approx(0.0, abs=1e-9)
    assert (
        fit_manifest["raw_holdout_evaluation_artifacts_opened_by_fit_process"]
        is False
    )
    assert fit_manifest["parquet_page_level_decode_scope_attested"] is False
    assert not (result.output_directory / "residual_corrector.json").exists()
    assert not (result.output_directory / "oof_predictions_365.csv.gz.audit.json").exists()
    assert {path.name for path in result.output_directory.iterdir()} == set(
        research.FIT_ARTIFACT_NAMES
    )
    assert not (result.output_directory / "cache").exists()


def test_fit_records_possible_physical_holdout_row_group_decode_without_using_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    panel = pd.read_parquet(config.panel_path)
    poison = panel.iloc[[0]].copy()
    poison[config.origin_column] = pd.Timestamp(
        manifest["splits"]["evaluation_holdout"]["first_utc"]
    )
    poison[config.timestamp_column] = pd.Timestamp("2024-02-01T00:00:00Z")
    poison[config.feature_available_at_column] = poison[config.origin_column]
    poison["target"] = 999999.0
    pd.concat([panel, poison], ignore_index=True).to_parquet(
        config.panel_path, index=False, row_group_size=len(panel) + len(poison)
    )
    manifest["panel_sha256"] = _sha(config.panel_path)
    _write_json(run / "experiment_manifest.json", manifest)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    result = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-overlap",
        item_id="FR",
    )
    fit_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(result.predictions_path)
    assert fit_manifest["parquet_engine_may_decode_holdout_rows"] is True
    assert fit_manifest["parquet_page_level_decode_scope_attested"] is False
    assert fit_manifest["returned_holdout_rows"] is False
    assert fit_manifest["parquet_filter_audit"]["returned_origin_count"] == 30
    assert 999999.0 not in predictions["actual"].to_numpy(float)


def test_fit_rejects_posthoc_config_semantic_change_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    bad = replace(config, timezone="UTC")
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="diverge du schema"
    ):
        research.fit_research_validation_corrector(
            bad,
            run_directory=run,
            output_directory=tmp_path / "bad-config",
            item_id="FR",
        )
    assert not (tmp_path / "bad-config").exists()


def test_fit_rejects_source_mutation_during_prediction_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    original = research._predict_validation

    def mutating_predict(*args, **kwargs):
        result = original(*args, **kwargs)
        config.panel_path.write_bytes(config.panel_path.read_bytes() + b"mutated")
        return result

    monkeypatch.setattr(research, "_predict_validation", mutating_predict)
    destination = tmp_path / "source-mutated"
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="Identite source divergente"
    ):
        research.fit_research_validation_corrector(
            config,
            run_directory=run,
            output_directory=destination,
            item_id="FR",
        )
    assert not destination.exists()


def test_fit_rejects_output_nested_in_another_lora_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    other = tmp_path / "other-candidate"
    (other / "checkpoint").mkdir(parents=True)
    (other / "experiment_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="autre bundle LoRA"
    ):
        research.fit_research_validation_corrector(
            config,
            run_directory=run,
            output_directory=other / "research",
            item_id="FR",
        )


def test_evaluation_verifies_seal_then_applies_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    fit = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-fit",
        item_id="FR",
        batch_size=4,
    )
    corrector_sha_before = _sha(fit.corrector_path)
    raw = _holdout()
    # Canonical backtest publication appends this mutable reference after the
    # corrector may already have been sealed.  The immutable candidate
    # contract must remain identical.
    _publish_holdout(run, manifest, raw, config)
    result = research.evaluate_research_validation_corrector(
        config,
        run_directory=run,
        research_directory=fit.output_directory,
        item_id="FR",
    )
    assert _sha(fit.corrector_path) == corrector_sha_before
    assert result.metrics["baseline_mae_eur_mwh"] == pytest.approx(5.0)
    assert result.metrics["candidate_mae_eur_mwh"] == pytest.approx(0.0, abs=1e-9)
    published = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert published["corrector_verified_before_holdout_read"] is True
    assert published["holdout_refit_performed"] is False
    assert published["promotion_eligible"] is False
    assert published["corrector_verified_at_utc"] <= published[
        "holdout_read_started_at_utc"
    ]
    assert {path.name for path in result.output_directory.iterdir()} == set(
        research.EVALUATION_ARTIFACT_NAMES
    )


def _fitted_and_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ExogenousFineTuneConfig,
    Path,
    dict[str, object],
    research.ResearchFitArtifacts,
]:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    fit = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-fit",
        item_id="FR",
    )
    _publish_holdout(run, manifest, _holdout(), config)
    return config, run, manifest, fit


def test_evaluation_rejects_noncanonical_raw_comparison_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    evaluation_path = run / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["comparison"]["same_inputs"] = False
    _write_json(evaluation_path, evaluation)
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="preuve holdout brute n'a pas le contrat attendu",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )
    assert not (fit.output_directory / research.EVALUATION_DIRECTORY_NAME).exists()


def test_evaluation_rejects_replacement_of_holdout_that_existed_at_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    original = _holdout()
    _publish_holdout(run, manifest, original, config)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    fit = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-fit",
        item_id="FR",
    )
    replacement = original.copy()
    replacement["actual"] = replacement["actual"] + 1.0
    _publish_holdout(run, manifest, replacement, config)
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="holdout brut deja present au fit a ete remplace",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_evaluation_rejects_raw_metrics_even_if_attacker_updates_artifact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    metrics_path = run / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["candidate_mae_eur_mwh"] = 0.0
    _write_json(metrics_path, metrics)
    evaluation_path = run / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["artifacts"]["metrics"]["sha256"] = _sha(metrics_path)
    _write_json(evaluation_path, evaluation)
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="Metrique holdout brute divergente",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_evaluation_rejects_corrector_mutation_during_holdout_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    original = research._apply_corrector

    def mutate_corrector(*args, **kwargs):
        result = original(*args, **kwargs)
        fit.corrector_path.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(research, "_apply_corrector", mutate_corrector)
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="SHA research divergent"
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )
    assert not (fit.output_directory / research.EVALUATION_DIRECTORY_NAME).exists()


def test_evaluation_rejects_raw_evidence_mutation_between_read_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    original = research._apply_corrector
    raw_path = run / "evaluation_predictions.csv.gz"

    def mutate_raw(*args, **kwargs):
        result = original(*args, **kwargs)
        raw_path.write_bytes(raw_path.read_bytes() + b"mutated")
        return result

    monkeypatch.setattr(research, "_apply_corrector", mutate_raw)
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="artefact holdout brut evidence a change",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )
    assert not (fit.output_directory / research.EVALUATION_DIRECTORY_NAME).exists()


def test_evaluation_recomputes_validation_input_contract_before_holdout_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    replacement = "e" * 64
    corrector = json.loads(fit.corrector_path.read_text(encoding="utf-8"))
    corrector["input_contract_sha256"] = replacement
    _write_json(fit.corrector_path, corrector)
    fit_manifest = json.loads(fit.manifest_path.read_text(encoding="utf-8"))
    fit_manifest["input_contract_sha256"] = replacement
    fit_manifest["artifacts"]["corrector"]["sha256"] = _sha(fit.corrector_path)
    _write_json(fit.manifest_path, fit_manifest)
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "holdout read before input-contract recomputation"
        ),
    )

    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="contrat d'entree validation n'est pas reproductible",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_evaluation_recomputes_parquet_audit_from_source_before_holdout_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    fit_manifest = json.loads(fit.manifest_path.read_text(encoding="utf-8"))
    audit = fit_manifest["parquet_filter_audit"]
    record = audit["selected_row_groups"][0]
    minimum = pd.Timestamp(record["minimum_origin_utc"])
    record["minimum_origin_utc"] = (minimum + pd.Timedelta(hours=1)).isoformat()
    audit["audit_sha256"] = research._digest_without_field(audit, "audit_sha256")
    corrector = json.loads(fit.corrector_path.read_text(encoding="utf-8"))
    corrector["parquet_filter_audit_sha256"] = audit["audit_sha256"]
    _write_json(fit.corrector_path, corrector)
    fit_manifest["artifacts"]["corrector"]["sha256"] = _sha(fit.corrector_path)
    _write_json(fit.manifest_path, fit_manifest)
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "holdout read before parquet-audit recomputation"
        ),
    )

    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="filtre ou le contrat d'entree validation n'est pas reproductible",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_evaluation_rejects_production_named_extra_file_before_holdout_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    (fit.output_directory / "residual_corrector.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail("holdout read before closure check"),
    )
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="nom production interdit",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_fit_seal_recomputes_coefficients_instead_of_trusting_self_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    corrector = json.loads(fit.corrector_path.read_text(encoding="utf-8"))
    corrector["coefficients"][0] += 1.0
    _write_json(fit.corrector_path, corrector)
    fit_manifest = json.loads(fit.manifest_path.read_text(encoding="utf-8"))
    fit_manifest["artifacts"]["corrector"]["sha256"] = _sha(fit.corrector_path)
    _write_json(fit.manifest_path, fit_manifest)
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail("holdout read before fit recomputation"),
    )
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="Coefficients/normalisation.*non reproductibles",
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_evaluation_rejects_hardlinked_fit_artifact_before_holdout_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, _manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    try:
        (tmp_path / "external-corrector-link.json").hardlink_to(fit.corrector_path)
    except OSError:
        pytest.skip("hardlinks indisponibles sur ce volume")
    monkeypatch.setattr(
        research,
        "_verified_holdout_evidence",
        lambda *_args, **_kwargs: pytest.fail("holdout read before hardlink check"),
    )
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="hardlink interdit"
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )


def test_tampered_research_seal_is_rejected_before_holdout_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest = _fixture(tmp_path)
    monkeypatch.setattr(research, "verify_bundle", lambda _run: manifest)
    fit = research.fit_research_validation_corrector(
        config,
        run_directory=run,
        output_directory=tmp_path / "research-fit",
        item_id="FR",
        batch_size=4,
    )
    fit.corrector_path.write_text("{}\n", encoding="utf-8")
    reads = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("holdout opened before seal verification")

    monkeypatch.setattr(research.pd, "read_csv", forbidden_read)
    with pytest.raises(
        research.ResearchValidationCorrectorError, match="SHA research divergent"
    ):
        research.evaluate_research_validation_corrector(
            config,
            run_directory=run,
            research_directory=fit.output_directory,
            item_id="FR",
        )
    assert reads == 0


def test_evaluation_publication_recomputes_metrics_after_hash_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, run, manifest, fit = _fitted_and_published(tmp_path, monkeypatch)
    result = research.evaluate_research_validation_corrector(
        config,
        run_directory=run,
        research_directory=fit.output_directory,
        item_id="FR",
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    metrics["candidate_mae_eur_mwh"] += 1.0
    _write_json(result.metrics_path, metrics)
    evaluation_manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )
    evaluation_manifest["artifacts"]["metrics"]["sha256"] = _sha(
        result.metrics_path
    )
    _write_json(result.manifest_path, evaluation_manifest)

    seal = research._verify_fit_seal(fit.output_directory)
    identity = research._source_identity(config, run, manifest, item_id="FR")
    raw = research._verified_holdout_evidence(
        run,
        config,
        manifest,
        item_id="FR",
        target_column="target",
    )
    with pytest.raises(
        research.ResearchValidationCorrectorError,
        match="Metrique evaluation research non reproductible",
    ):
        research._verify_evaluation_publication(
            result.output_directory,
            seal=seal,
            raw=raw,
            source_identity=identity,
            item_id="FR",
            target_column="target",
        )


def test_production_explicitly_rejects_research_corrector(
    tmp_path: Path,
) -> None:
    path = tmp_path / research.CORRECTOR_NAME
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": research.CORRECTOR_KIND,
            "research_only": True,
            "fit_protocol": research.FIT_PROTOCOL,
        },
    )
    with pytest.raises(ExogenousProductionError, match="research-only interdit"):
        _validate_corrector(
            path,
            experiment={
                "residual_corrector_sha256": _sha(path),
                "candidate_output_stage": "exogenous_residual_corrected",
            },
        )
