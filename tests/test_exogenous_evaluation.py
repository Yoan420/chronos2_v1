from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.evaluation import (
    EVALUATION_COLUMNS,
    SHADOW_INPUT_COLUMNS,
    ExogenousEvaluationError,
    append_shadow_predictions,
    build_inference_input,
    compute_metrics,
    daily_inference,
    evaluate_holdout,
    render_report,
    run_evaluation,
    run_shadow,
)
from chronos2_exogenous.lora_finetune import (
    ExogenousFineTuneConfig,
    OriginSplit,
    sha256_directory,
    validate_panel,
)
from chronos2_exogenous.governance import validate_shadow_manifest


class FakePipeline:
    def __init__(self, median: float, *, fail: bool = False) -> None:
        self.median = float(median)
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def predict_quantiles(self, inputs: list[dict[str, Any]], **kwargs: Any):
        if self.fail:
            raise AssertionError("pipeline should not be called when cache is valid")
        assert kwargs["cross_learning"] is False
        horizon = int(kwargs["prediction_length"])
        self.calls.append({"count": len(inputs), **kwargs})
        output = []
        for value in inputs:
            target = np.asarray(value["target"])
            assert target.ndim == 2
            assert target.shape[-1] == 2
            assert all(len(x) == 2 for x in value["past_covariates"].values())
            assert all(len(x) == horizon for x in value["future_covariates"].values())
            quantiles = np.empty((target.shape[0], horizon, 3), dtype=float)
            quantiles[:, :, 0] = self.median - 10.0
            quantiles[:, :, 1] = self.median
            quantiles[:, :, 2] = self.median + 10.0
            output.append(quantiles)
        return output, [item[:, :, 1] for item in output]


def _config(tmp_path: Path, *, evaluation_days: int = 365) -> ExogenousFineTuneConfig:
    return ExogenousFineTuneConfig(
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        experiment_id="unit",
        evaluation_role="primary_predeclared",
        panel_path=tmp_path / "panel.parquet",
        panel_audit_path=tmp_path / "panel.parquet.audit.json",
        output_directory=tmp_path / "bundle",
        timestamp_column="timestamp",
        origin_column="origin_timestamp",
        item_column="item_id",
        feature_available_at_column="feature_available_at_utc",
        target_columns=("target",),
        known_future_covariates=("weather",),
        past_only_covariates=("fuel",),
        timezone="Europe/Paris",
        cutoff_local_time="08:00",
        frequency="h",
        context_length=2,
        prediction_length=24,
        training_window_days=365,
        validation_days=30,
        evaluation_days=evaluation_days,
        require_consecutive_origins=True,
        require_complete_known_future=True,
        production_pit_evidence=False,
        model_id="amazon/chronos-2",
        model_revision=None,
        local_files_only=True,
        device_map="cpu",
        learning_rate=1e-5,
        num_steps=1,
        batch_size=4,
        seed=7,
        lora_config={"r": 2},
    )


def _day_frame(day: pd.Timestamp, *, actual: float | None = 100.0) -> pd.DataFrame:
    timezone_name = "Europe/Paris"
    local_day = pd.Timestamp(day).normalize()
    start = local_day.tz_localize(timezone_name)
    end = (local_day + pd.Timedelta(days=1)).tz_localize(timezone_name)
    horizon = pd.date_range(
        start=start.tz_convert("UTC"),
        end=end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    context = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1), periods=2, freq="h")
    origin_local = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(
        timezone_name
    )
    origin = origin_local.tz_convert("UTC")
    timestamps = context.append(horizon)
    target = np.full(len(timestamps), 90.0)
    target[-len(horizon) :] = np.nan if actual is None else float(actual)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "origin_timestamp": origin,
            "item_id": "FR",
            "feature_available_at_utc": origin,
            "target": target,
            "weather": 3.0,
            "fuel": 2.0,
        }
    )


def test_build_inference_input_preserves_dst_and_masks_actual(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spring = _day_frame(pd.Timestamp("2026-03-29"))
    payload, horizon, actual = build_inference_input(spring, config)
    assert len(horizon) == 23
    assert payload["target"].shape == (1, 2)
    assert payload["past_covariates"]["weather"].shape == (2,)
    assert payload["future_covariates"]["weather"].shape == (23,)
    assert np.all(actual == 100.0)

    autumn = _day_frame(pd.Timestamp("2025-10-26"))
    _, horizon, _ = build_inference_input(autumn, config)
    assert len(horizon) == 25


def test_missing_shadow_actual_allowed_but_partial_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = _day_frame(pd.Timestamp("2026-09-04"), actual=None)
    _, _, actual = build_inference_input(frame, config, allow_missing_actual=True)
    assert np.isnan(actual).all()
    frame.loc[frame.index[-1], "target"] = 42.0
    with pytest.raises(ExogenousEvaluationError, match="entièrement"):
        build_inference_input(frame, config, allow_missing_actual=True)


def test_evaluate_exact_365_days_batches_dst_and_reuses_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    days = pd.date_range("2025-09-04", periods=365, freq="D")
    panel = pd.concat([_day_frame(day) for day in days], ignore_index=True)
    origins = tuple(
        pd.DatetimeIndex(panel["origin_timestamp"].drop_duplicates()).sort_values()
    )
    split = OriginSplit(train=(), validation=(), evaluation=origins)
    base = FakePipeline(105.0)
    candidate = FakePipeline(102.0)
    cache = tmp_path / "cache"
    evidence, audit = evaluate_holdout(
        panel,
        split,
        config,
        baseline_pipeline=base,
        candidate_pipeline=candidate,
        item_id="FR",
        inference_chunk_size=500,
        cache_directory=cache,
        baseline_identity="base-v1",
        candidate_identity="adapter-v1",
    )
    assert evidence.columns.tolist() == list(EVALUATION_COLUMNS)
    assert evidence["delivery_start_utc"].nunique() == len(evidence)
    assert len(evidence) == 365 * 24  # one 23h and one 25h day cancel out
    assert audit["evaluation_days"] == 365
    assert audit["cross_learning"] is False
    assert sorted(call["prediction_length"] for call in base.calls) == [23, 24, 25]

    # Fully cached replay must not load/use model outputs again.
    replay, _ = evaluate_holdout(
        panel,
        split,
        config,
        baseline_pipeline=FakePipeline(0, fail=True),
        candidate_pipeline=FakePipeline(0, fail=True),
        item_id="FR",
        inference_chunk_size=500,
        cache_directory=cache,
        baseline_identity="base-v1",
        candidate_identity="adapter-v1",
    )
    pd.testing.assert_frame_equal(evidence, replay)

    metrics, daily = compute_metrics(evidence, timezone_name=config.timezone)
    assert metrics["physical_days"] == 365
    assert metrics["candidate_mae_eur_mwh"] < metrics["baseline_mae_eur_mwh"]
    assert {23, 25}.issubset(set(daily["hours"]))
    report = render_report(metrics, daily, audit)
    assert "Mode nuit" in report
    assert "Aucun correcteur résiduel" in report
    assert "365 journées" in report


def _fake_bundle(path: Path) -> dict[str, Any]:
    checkpoint = path / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    schema = path / "schema.json"
    schema.write_text("{}\n", encoding="utf-8")
    import hashlib

    schema_digest = hashlib.sha256(schema.read_bytes()).hexdigest()
    manifest = {
        "model_id": "amazon/chronos-2",
        "experiment_id": "unit",
        "finetune_mode": "lora",
        "checkpoint_sha256": sha256_directory(checkpoint),
        "schema_sha256": schema_digest,
    }
    (path / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _shadow_input(checkpoint_sha256: str, actual: float | None) -> pd.DataFrame:
    origin = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    delivery_day = (
        origin.tz_convert("Europe/Paris") + pd.DateOffset(days=1)
    ).date()
    timestamp = pd.Timestamp(delivery_day, tz="Europe/Paris").tz_convert("UTC")
    value = np.nan if actual is None else actual
    provenance = _panel_evidence(origin, horizon_actuals_present=False)
    return pd.DataFrame(
        [
            {
                "delivery_start_utc": timestamp,
                "forecast_origin_utc": origin,
                "item_id": "FR",
                "target_column": "target",
                "actual": value,
                "baseline_q10": 80.0,
                "baseline_q50": 90.0,
                "baseline_q90": 100.0,
                "candidate_q10": 82.0,
                "candidate_q50": 92.0,
                "candidate_q90": 102.0,
                "input_contract_sha256": "a" * 64,
                "checkpoint_sha256": checkpoint_sha256,
                "panel_sha256": provenance["panel_sha256"],
                "panel_audit_sha256": provenance["panel_audit_sha256"],
                "panel_contract_sha256": provenance["panel_contract_sha256"],
                "panel_created_at_utc": provenance["panel_created_at_utc"],
                "panel_provenance_json": json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ],
        columns=SHADOW_INPUT_COLUMNS,
    )


def _panel_evidence(
    origin: pd.Timestamp, *, horizon_actuals_present: bool
) -> dict[str, Any]:
    delivery_day = (
        pd.Timestamp(origin).tz_convert("Europe/Paris") + pd.DateOffset(days=1)
    ).date().isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": "c" * 64,
        "panel_audit_sha256": "d" * 64,
        "panel_created_at_utc": (
            pd.Timestamp(origin) + pd.Timedelta(minutes=30)
        ).isoformat(),
        "zone": "FR",
        "delivery_day": delivery_day,
        "forecast_origin_utc": pd.Timestamp(origin).isoformat(),
        "forecast_origin_timezone": "Europe/Paris",
        "delivery_timezone": "Europe/Paris",
        "pack": "residual_only",
        "production_pit_evidence": False,
        "source_hashes": {
            "deterministic_calendar": "e" * 64,
            "residual_load": "f" * 64,
        },
        "source_audit_hashes": {"residual_load": "1" * 64},
        "source_cutoff_timezones": {
            "deterministic_calendar": "Europe/Paris",
            "residual_load": "Europe/Paris",
        },
        "target_source_sha256": "2" * 64,
        "horizon_actuals_present": bool(horizon_actuals_present),
    }
    payload["panel_contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _write_shadow_panel_sidecar(
    panel_path: Path,
    frame: pd.DataFrame,
    *,
    created_at: pd.Timestamp,
    horizon_actuals_present: bool,
) -> Path:
    origin = pd.Timestamp(frame["origin_timestamp"].iloc[0])
    delivery_day = (
        origin.tz_convert("Europe/Paris") + pd.DateOffset(days=1)
    ).date().isoformat()
    digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    audit = {
        "schema_version": 1,
        "purpose": "prospective_shadow",
        "panel_sha256": digest,
        "created_at_utc": pd.Timestamp(created_at).isoformat(),
        "delivery_day": delivery_day,
        "forecast_origin_utc": origin.isoformat(),
        "forecast_origin_timezone": "Europe/Paris",
        "delivery_timezones": {"FR": "Europe/Paris"},
        "pack": "residual_only",
        "layout": "per_zone",
        "zones": ["FR"],
        "horizon_actuals_present": bool(horizon_actuals_present),
        "production_pit_evidence": {"FR": False},
        "target_sources": {"FR": {"source_sha256": "2" * 64}},
        "exogenous_banks": {
            "FR": {
                "production_ready": False,
                "source_hashes": {
                    "deterministic_calendar": "e" * 64,
                    "residual_load": "f" * 64,
                },
                "source_audit_hashes": {"residual_load": "1" * 64},
                "source_cutoff_timezones": {
                    "deterministic_calendar": "Europe/Paris",
                    "residual_load": "Europe/Paris",
                },
            }
        },
    }
    audit_path = panel_path.with_suffix(panel_path.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path


def test_shadow_journal_is_idempotent_and_appends_actual_resolution(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _fake_bundle(bundle)
    first = append_shadow_predictions(
        bundle, _shadow_input(manifest["checkpoint_sha256"], None)
    )
    assert first.appended_rows == 1
    assert first.observed_evidence_path is None
    assert first.manifest_path is None

    duplicate = append_shadow_predictions(
        bundle, _shadow_input(manifest["checkpoint_sha256"], None)
    )
    assert duplicate.appended_rows == 0

    resolved = append_shadow_predictions(
        bundle, _shadow_input(manifest["checkpoint_sha256"], 97.0)
    )
    assert resolved.appended_rows == 1
    assert resolved.observed_evidence_path is not None
    assert resolved.manifest_path is not None
    journal = pd.read_csv(resolved.journal_path)
    assert journal["record_kind"].tolist() == ["forecast", "actual_resolution"]
    evidence = pd.read_csv(resolved.observed_evidence_path)
    assert evidence.columns.tolist() == list(EVALUATION_COLUMNS)
    assert evidence["actual"].tolist() == [97.0]
    shadow_payload = validate_shadow_manifest(
        resolved.manifest_path,
        experiment_manifest=manifest,
        zone="FR",
    )
    import hashlib

    assert shadow_payload["predictions_sha256"] == hashlib.sha256(
        resolved.observed_evidence_path.read_bytes()
    ).hexdigest()

    second_duplicate = append_shadow_predictions(
        bundle, _shadow_input(manifest["checkpoint_sha256"], 97.0)
    )
    assert second_duplicate.appended_rows == 0


def test_shadow_refuses_first_emission_after_actual_publication(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = _fake_bundle(bundle)
    with pytest.raises(ExogenousEvaluationError, match="rétrospective interdite"):
        append_shadow_predictions(
            bundle, _shadow_input(manifest["checkpoint_sha256"], 97.0)
        )
    assert not (bundle / "shadow_predictions.csv.gz").exists()


def test_run_shadow_attaches_actual_without_reinvoking_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronos2_exogenous.evaluation as evaluation_module

    config = _config(tmp_path)
    bundle = config.output_directory
    bundle.mkdir()
    manifest = _fake_bundle(bundle)
    day = pd.Timestamp("2026-09-04")
    forecast_panel = _day_frame(day, actual=None)
    origin = pd.Timestamp(forecast_panel["origin_timestamp"].iloc[0])
    captures = iter(
        [
            origin + pd.Timedelta(hours=1),
            origin + pd.Timedelta(hours=5),
            origin + pd.Timedelta(hours=6),
        ]
    )
    monkeypatch.setattr(evaluation_module, "_utc_now", lambda: next(captures))
    forecast = daily_inference(
        forecast_panel,
        config,
        baseline_pipeline=FakePipeline(90.0),
        candidate_pipeline=FakePipeline(92.0),
        origins=[origin],
        item_id="FR",
        checkpoint_sha256=manifest["checkpoint_sha256"],
        panel_evidence=_panel_evidence(
            origin, horizon_actuals_present=False
        ),
    )
    append_shadow_predictions(bundle, forecast)

    observed_panel = _day_frame(day, actual=97.0)
    observed_path = tmp_path / "observed.parquet"
    observed_panel.to_parquet(observed_path, index=False)
    observed_audit = _write_shadow_panel_sidecar(
        observed_path,
        observed_panel,
        created_at=origin + pd.Timedelta(hours=5),
        horizon_actuals_present=True,
    )
    result = run_shadow(
        config,
        panel_path=observed_path,
        panel_audit_path=observed_audit,
        origins=[origin],
        item_id="FR",
        baseline_pipeline=FakePipeline(0, fail=True),
        candidate_pipeline=FakePipeline(0, fail=True),
    )
    assert result.appended_rows == len(forecast)
    assert result.observed_evidence_path is not None


def test_run_shadow_requires_sidecar_bound_to_exact_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronos2_exogenous.evaluation as evaluation_module

    config = _config(tmp_path)
    bundle = config.output_directory
    bundle.mkdir()
    _fake_bundle(bundle)
    day = pd.Timestamp.now(tz="Europe/Paris").normalize() + pd.Timedelta(days=1)
    panel = _day_frame(day.tz_localize(None), actual=None)
    panel_path = tmp_path / "shadow.parquet"
    panel.to_parquet(panel_path, index=False)
    with pytest.raises(ExogenousEvaluationError, match="sidecar audit explicite"):
        run_shadow(
            config,
            panel_path=panel_path,
            item_id="FR",
            baseline_pipeline=FakePipeline(90.0),
            candidate_pipeline=FakePipeline(92.0),
        )

    origin = pd.Timestamp(panel["origin_timestamp"].iloc[0])
    monkeypatch.setattr(
        evaluation_module,
        "_utc_now",
        lambda: origin + pd.Timedelta(hours=1),
    )
    audit_path = _write_shadow_panel_sidecar(
        panel_path,
        panel,
        created_at=origin + pd.Timedelta(minutes=30),
        horizon_actuals_present=False,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["panel_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExogenousEvaluationError, match="octets du panel"):
        run_shadow(
            config,
            panel_path=panel_path,
            panel_audit_path=audit_path,
            item_id="FR",
            baseline_pipeline=FakePipeline(90.0),
            candidate_pipeline=FakePipeline(92.0),
        )


def test_backtest_refuses_panel_changed_after_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle = config.output_directory
    bundle.mkdir()
    manifest = _fake_bundle(bundle)
    config.panel_path.write_bytes(b"panel-modified-after-training")
    manifest["panel_sha256"] = "0" * 64
    (bundle / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ExogenousEvaluationError, match="diffère du panel scellé"):
        run_evaluation(
            config,
            baseline_pipeline=FakePipeline(100.0),
            candidate_pipeline=FakePipeline(100.0),
        )


def _write_training_panel_audit(
    config: ExogenousFineTuneConfig,
    *,
    target_source: Path,
) -> None:
    panel_sha = hashlib.sha256(config.panel_path.read_bytes()).hexdigest()
    target_sha = hashlib.sha256(target_source.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "layout": "per_zone",
        "zones": ["FR"],
        "pack": "residual_only",
        "panel_path": str(config.panel_path),
        "panel_sha256": panel_sha,
        "production_ready": False,
        "production_pit_evidence": {"FR": False},
        "allow_unresolved_final_evaluation_day": True,
        "canonical_target_contracts_verified": True,
        "target_sources": {
            "FR": {
                "source_path": str(target_source),
                "source_sha256": target_sha,
            }
        },
        "target_contracts": {
            "FR": {
                "series": "power.price.da.fr.canonical",
                "cache_path": str(target_source),
            }
        },
        "exogenous_banks": {
            "FR": {
                "production_ready": False,
                "production_blockers": ["unit_test"],
                "source_hashes": {"synthetic": "a" * 64},
                "source_audit_hashes": {"synthetic": "b" * 64},
                "source_cutoff_timezones": {"synthetic": "Europe/Paris"},
            }
        },
    }
    config.panel_audit_path.write_text(json.dumps(payload), encoding="utf-8")


def test_backtest_binds_late_actuals_without_changing_any_input(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path, evaluation_days=365),
        training_window_days=2,
        validation_days=1,
        allow_unresolved_final_evaluation_day=True,
    )
    # Two warm-up days followed by the exact 365-day governance holdout.
    # This span also crosses both DST transitions, so the late-bound final
    # horizon remains a physical 23/24/25-hour delivery day by construction.
    days = pd.date_range("2025-09-05", periods=367, freq="D")
    frozen_panel = pd.concat(
        [
            _day_frame(day, actual=None if number == len(days) - 1 else 100.0)
            for number, day in enumerate(days)
        ],
        ignore_index=True,
    )
    config.panel_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_panel.to_parquet(config.panel_path, index=False)
    _write_training_panel_audit(config, target_source=config.panel_path)
    validated, split, audit = validate_panel(frozen_panel, config)
    assert audit["evaluation_label_binding"]["unresolved_cells"] == 24

    bundle = config.output_directory
    bundle.mkdir()
    checkpoint = bundle / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    schema_payload = {
        "format_version": 1,
        "timestamp_column": config.timestamp_column,
        "origin_column": config.origin_column,
        "item_column": config.item_column,
        "feature_available_at_column": config.feature_available_at_column,
        "target_columns": list(config.target_columns),
        "known_future_covariates": list(config.known_future_covariates),
        "past_only_covariates": list(config.past_only_covariates),
        "timezone": config.timezone,
        "cutoff_local_time": config.cutoff_local_time,
        "frequency": config.frequency,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
    }
    schema_path = bundle / "schema.json"
    schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")
    range_payload = lambda values: {
        "count": len(values),
        "first_utc": values[0].isoformat(),
        "last_utc": values[-1].isoformat(),
    }
    manifest = {
        "model_id": "amazon/chronos-2",
        "experiment_id": "unit-prospective",
        "evaluation_role": "primary_predeclared",
        "finetune_mode": "lora",
        "checkpoint_sha256": sha256_directory(checkpoint),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "panel_sha256": hashlib.sha256(config.panel_path.read_bytes()).hexdigest(),
        "panel_audit_sha256": hashlib.sha256(
            config.panel_audit_path.read_bytes()
        ).hexdigest(),
        "target_contracts": audit["upstream_panel_audit"]["target_contracts"],
                "evaluation_days": 365,
        "production_pipeline_evidence": False,
        "evaluation_label_binding": audit["evaluation_label_binding"],
        "splits": {
            "train": range_payload(split.train),
            "validation": range_payload(split.validation),
            "evaluation_holdout": range_payload(split.evaluation),
        },
    }
    (bundle / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ExogenousEvaluationError, match="Backtest differe requis"):
        run_evaluation(
            config,
            baseline_pipeline=FakePipeline(105.0),
            candidate_pipeline=FakePipeline(102.0),
        )

    resolved_config = replace(
        config,
        panel_path=tmp_path / "panel_resolved.parquet",
        panel_audit_path=tmp_path / "panel_resolved.parquet.audit.json",
    )
    resolved_panel = pd.concat([_day_frame(day, actual=100.0) for day in days])
    resolved_panel.to_parquet(resolved_config.panel_path, index=False)
    _write_training_panel_audit(resolved_config, target_source=config.panel_path)
    result = run_evaluation(
        config,
        panel_path=resolved_config.panel_path,
        panel_audit_path=resolved_config.panel_audit_path,
        baseline_pipeline=FakePipeline(105.0),
        candidate_pipeline=FakePipeline(102.0),
    )
    assert result.metrics["physical_days"] == 365
    committed = json.loads((bundle / "experiment_manifest.json").read_text())
    resolution = committed["evaluation_label_resolution"]
    assert resolution["resolved_cells"] == 24
    assert resolution["all_other_values_identical"] is True
    assert resolution["frozen_panel_sha256"] == manifest["panel_sha256"]
