from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from chronos2_exogenous.lora_finetune import (
    EVALUATION_COLUMNS,
    ExogenousFineTuneError,
    build_fit_inputs,
    bind_resolved_evaluation_panel,
    load_checkpoint,
    load_config,
    publish_evaluation_evidence,
    resolve_local_model_source,
    train_lora,
    validate_panel,
    verify_bundle,
)


def _write_config(root: Path, *, output: str = "runs/artifact") -> Path:
    payload = {
        "format_version": 1,
        "experiment_id": "test_exogenous_lora",
        "evaluation_role": "primary_predeclared",
        "project_root": ".",
        "data": {
            "panel_path": "inputs/panel.parquet",
            "timestamp_column": "timestamp",
            "origin_column": "origin_timestamp",
            "item_column": "item_id",
            "feature_available_at_column": "feature_available_at_utc",
            "target_columns": ["target"],
            "known_future_covariates": ["wind_fcst", "load_fcst"],
            "past_only_covariates": ["price_lag"],
            "timezone": "Europe/Paris",
            "cutoff_local_time": "08:00",
            "frequency": "h",
            "context_length": 24,
            "prediction_length": 24,
            "training_window_days": 4,
            "validation_days": 1,
            "evaluation_days": 2,
            "require_consecutive_origins": True,
            "require_complete_known_future": True,
            "production_pit_evidence": False,
        },
        "model": {
            "model_id": "amazon/chronos-2",
            "revision": "test-revision",
            "local_files_only": True,
            "device_map": "cpu",
        },
        "training": {
            "finetune_mode": "lora",
            "learning_rate": 1e-5,
            "num_steps": 2,
            "batch_size": 8,
            "seed": 7,
            "lora_config": {"r": 2, "lora_alpha": 4, "target_modules": ["q"]},
        },
        "output": {"directory": output},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_evaluation_role_is_explicit_and_strict(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    assert load_config(path).evaluation_role == "primary_predeclared"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["evaluation_role"] = "choose_best_after_holdout"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExogenousFineTuneError, match="evaluation_role"):
        load_config(path)


def _panel(
    *,
    start_day: str = "2026-01-01",
    days: int = 6,
    irregular_origin_index: int | None = None,
    irregular_horizon: int = 23,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day_index in range(days):
        origin = pd.Timestamp(start_day, tz="Europe/Paris") + pd.DateOffset(
            days=day_index, hours=8
        )
        delivery_start = origin.normalize() + pd.DateOffset(days=1)
        horizon = irregular_horizon if day_index == irregular_origin_index else 24
        timestamps = pd.date_range(
            delivery_start - pd.Timedelta(hours=24),
            periods=24 + horizon,
            freq="h",
        )
        for row_index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp.tz_convert("UTC"),
                    "origin_timestamp": origin.tz_convert("UTC"),
                    "item_id": "FR",
                    "feature_available_at_utc": (
                        origin - pd.Timedelta(minutes=5)
                    ).tz_convert("UTC"),
                    "target": float(100 + day_index + row_index),
                    "wind_fcst": float(20 + row_index),
                    "load_fcst": float(50 + row_index),
                    "price_lag": float(80 + row_index),
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _persist_panel(
    config: object,
    frame: pd.DataFrame,
    *,
    production_ready: bool = False,
) -> pd.DataFrame:
    config.panel_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.panel_path, index=False)
    audit = {
        "schema_version": 1,
        "zones": ["FR"],
        "pack": "residual_only",
        "panel_path": str(config.panel_path),
        "panel_sha256": _sha256(config.panel_path),
        "production_ready": production_ready,
        "allow_unresolved_final_evaluation_day": bool(
            config.allow_unresolved_final_evaluation_day
        ),
        "production_pit_evidence": {"FR": production_ready},
        "exogenous_banks": {
            "FR": {
                "production_ready": production_ready,
                "production_blockers": [] if production_ready else ["research_only"],
                "source_hashes": {"synthetic": "a" * 64},
                "source_audit_hashes": {"synthetic": "b" * 64},
                "source_cutoff_timezones": {"synthetic": "Europe/Paris"},
            }
        },
        "target_sources": {
            "FR": {
                "source_path": str(config.panel_path),
                "source_sha256": _sha256(config.panel_path),
            }
        },
        "canonical_target_contracts_verified": True,
        "target_contracts": {
            "FR": {
                "series": "power.price.da.fr.canonical",
                "cache_path": str(config.panel_path),
            }
        },
    }
    config.panel_audit_path.write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return pd.read_parquet(config.panel_path)


def _fake_model_snapshot(root: Path) -> Path:
    snapshot = root / "fake-chronos2-snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "config.json").write_text('{"model":"chronos2"}\n', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"frozen-base-model")
    return snapshot


class _FakePipeline:
    def __init__(self) -> None:
        self.fit_kwargs: dict[str, object] | None = None

    def fit(self, **kwargs: object) -> "_FakePipeline":
        self.fit_kwargs = kwargs
        checkpoint = Path(str(kwargs["output_dir"])) / str(
            kwargs["finetuned_ckpt_name"]
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "amazon/chronos-2"}),
            encoding="utf-8",
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fake-lora")
        return self


class _RecordingLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.pipeline = _FakePipeline()

    def __call__(self, source: str | Path, kwargs: object) -> _FakePipeline:
        self.calls.append((str(source), dict(kwargs)))
        return self.pipeline


class _TrainerStatePipeline(_FakePipeline):
    def __init__(
        self,
        *,
        declared_max_steps: int = 2,
        retained_best_only: bool = False,
    ) -> None:
        super().__init__()
        self.declared_max_steps = declared_max_steps
        self.retained_best_only = retained_best_only

    def fit(self, **kwargs: object) -> "_TrainerStatePipeline":
        super().fit(**kwargs)
        output = Path(str(kwargs["output_dir"]))
        best_checkpoint = output / "checkpoint-1"
        final_checkpoint = output / "checkpoint-2"
        best_checkpoint.mkdir()
        state_checkpoint = best_checkpoint
        if not self.retained_best_only:
            final_checkpoint.mkdir()
            state_checkpoint = final_checkpoint
        state = {
            "global_step": 1 if self.retained_best_only else 2,
            "max_steps": self.declared_max_steps,
            "best_global_step": 1,
            "best_metric": 0.125,
            "best_model_checkpoint": str(best_checkpoint),
            "log_history": [{"step": 1, "eval_loss": 0.125}]
            + ([] if self.retained_best_only else [{"step": 2, "eval_loss": 0.25}]),
            "stateful_callbacks": {"TrainerControl": {}},
        }
        (state_checkpoint / "trainer_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        return self


def test_exact_origin_inputs_mask_past_only_future(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel())
    validated, split, audit = validate_panel(source, config)

    assert len(split.train) == 3
    assert len(split.validation) == 1
    assert len(split.evaluation) == 2
    assert audit["physical_evaluation_days"] == 2
    inputs = build_fit_inputs(validated, split.train, config)

    assert len(inputs) == 3
    assert inputs[0]["target"].shape == (1, 48)
    assert np.isnan(inputs[0]["past_covariates"]["price_lag"][-24:]).all()
    np.testing.assert_array_equal(
        inputs[0]["future_covariates"]["wind_fcst"],
        inputs[0]["past_covariates"]["wind_fcst"][-24:],
    )


def _prospective_config(root: Path) -> object:
    path = _write_config(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["allow_unresolved_final_evaluation_day"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_config(path)


def _blank_last_horizon(frame: pd.DataFrame, *, hours: int = 24) -> pd.DataFrame:
    result = frame.copy()
    last_origin = pd.Timestamp(result["origin_timestamp"].max())
    indices = (
        result.loc[result["origin_timestamp"].eq(last_origin)]
        .sort_values("timestamp")
        .index[-hours:]
    )
    result.loc[indices, "target"] = np.nan
    return result


def test_default_contract_still_rejects_unresolved_holdout(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _blank_last_horizon(_panel()))
    with pytest.raises(ExogenousFineTuneError, match="labels target"):
        validate_panel(source, config)


def test_prospective_contract_allows_only_last_complete_horizon(
    tmp_path: Path,
) -> None:
    config = _prospective_config(tmp_path)
    source = _persist_panel(config, _blank_last_horizon(_panel()))
    validated, split, audit = validate_panel(source, config)

    binding = audit["evaluation_label_binding"]
    assert binding["unresolved_cells"] == 24
    assert binding["resolution_required_before_evaluation"] is True
    assert binding["labels_used_for_fit"] is False
    assert len(build_fit_inputs(validated, split.train, config)) == 3
    assert len(build_fit_inputs(validated, split.validation, config)) == 1

    earlier = _panel()
    evaluation_origins = sorted(earlier["origin_timestamp"].unique())[-2:]
    earlier_origin = pd.Timestamp(evaluation_origins[0])
    earlier_indices = (
        earlier.loc[earlier["origin_timestamp"].eq(earlier_origin)]
        .sort_values("timestamp")
        .index[-24:]
    )
    earlier.loc[earlier_indices, "target"] = np.nan
    earlier_source = _persist_panel(config, earlier)
    with pytest.raises(ExogenousFineTuneError, match="toute derniere origine"):
        validate_panel(earlier_source, config)


def test_prospective_contract_accepts_23_hour_final_dst_horizon(
    tmp_path: Path,
) -> None:
    config = _prospective_config(tmp_path)
    frame = _panel(start_day="2026-03-23", irregular_origin_index=5)
    source = _persist_panel(config, _blank_last_horizon(frame, hours=23))
    _, _, audit = validate_panel(source, config)
    group = audit["evaluation_label_binding"]["unresolved_groups"][0]
    assert group["hours"] == 23
    assert audit["evaluation_label_binding"]["unresolved_cells"] == 23


def test_prospective_contract_accepts_25_hour_final_dst_horizon(
    tmp_path: Path,
) -> None:
    config = _prospective_config(tmp_path)
    frame = _panel(
        start_day="2026-10-19",
        irregular_origin_index=5,
        irregular_horizon=25,
    )
    source = _persist_panel(config, _blank_last_horizon(frame, hours=25))
    _, _, audit = validate_panel(source, config)
    group = audit["evaluation_label_binding"]["unresolved_groups"][0]
    assert group["hours"] == 25
    assert audit["evaluation_label_binding"]["unresolved_cells"] == 25


def test_prospective_contract_rejects_partial_or_context_missing_target(
    tmp_path: Path,
) -> None:
    config = _prospective_config(tmp_path)
    partial = _panel()
    last_origin = pd.Timestamp(partial["origin_timestamp"].max())
    group_indices = (
        partial.loc[partial["origin_timestamp"].eq(last_origin)]
        .sort_values("timestamp")
        .index
    )
    partial.loc[group_indices[-1], "target"] = np.nan
    partial_source = _persist_panel(config, partial)
    with pytest.raises(ExogenousFineTuneError, match="partiellement resolue"):
        validate_panel(partial_source, config)

    context = _panel()
    context.loc[
        context.loc[context["origin_timestamp"].eq(last_origin)]
        .sort_values("timestamp")
        .index[0],
        "target",
    ] = np.nan
    context_source = _persist_panel(config, context)
    with pytest.raises(ExogenousFineTuneError, match="target contexte"):
        validate_panel(context_source, config)


def test_late_resolution_changes_only_predeclared_nan_cells(tmp_path: Path) -> None:
    frozen_root = tmp_path / "frozen"
    frozen_root.mkdir()
    config = _prospective_config(frozen_root)
    unresolved_source = _persist_panel(config, _blank_last_horizon(_panel()))
    _, _, frozen_audit = validate_panel(unresolved_source, config)
    manifest = {
        "panel_sha256": _sha256(config.panel_path),
        "panel_audit_sha256": _sha256(config.panel_audit_path),
        "target_contracts": frozen_audit["upstream_panel_audit"]["target_contracts"],
        "evaluation_label_binding": frozen_audit["evaluation_label_binding"],
    }

    resolved_root = tmp_path / "resolved"
    resolved_root.mkdir()
    resolved_config = replace(
        config,
        panel_path=resolved_root / "panel.parquet",
        panel_audit_path=resolved_root / "panel.parquet.audit.json",
    )
    resolved_source = _persist_panel(resolved_config, _panel())
    resolved_sidecar = json.loads(
        resolved_config.panel_audit_path.read_text(encoding="utf-8")
    )
    frozen_sidecar = json.loads(config.panel_audit_path.read_text(encoding="utf-8"))
    resolved_sidecar["target_sources"] = frozen_sidecar["target_sources"]
    resolved_sidecar["target_contracts"] = frozen_sidecar["target_contracts"]
    resolved_config.panel_audit_path.write_text(
        json.dumps(resolved_sidecar, indent=2), encoding="utf-8"
    )

    _, _, _, resolution = bind_resolved_evaluation_panel(
        frozen_frame=unresolved_source,
        resolved_frame=resolved_source,
        frozen_config=config,
        resolved_config=resolved_config,
        experiment_manifest=manifest,
    )
    assert resolution["resolved_cells"] == 24
    assert resolution["all_other_values_identical"] is True

    changed_covariate = resolved_source.copy()
    changed_covariate.loc[0, "wind_fcst"] += 0.25
    with pytest.raises(ExogenousFineTuneError, match="entree|different"):
        bind_resolved_evaluation_panel(
            frozen_frame=unresolved_source,
            resolved_frame=changed_covariate,
            frozen_config=config,
            resolved_config=resolved_config,
            experiment_manifest=manifest,
        )

    changed_earlier_target = resolved_source.copy()
    changed_earlier_target.loc[0, "target"] += 0.25
    with pytest.raises(ExogenousFineTuneError, match="entree|different"):
        bind_resolved_evaluation_panel(
            frozen_frame=unresolved_source,
            resolved_frame=changed_earlier_target,
            frozen_config=config,
            resolved_config=resolved_config,
            experiment_manifest=manifest,
        )


def test_training_and_oof_panels_share_the_same_holdout_label_binding(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    training_config = _prospective_config(training_root)
    training_panel = _blank_last_horizon(
        _panel(start_day="2026-01-05", days=6)
    )
    training_source = _persist_panel(training_config, training_panel)
    _, _, training_audit = validate_panel(training_source, training_config)

    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    calibration_config = replace(
        training_config,
        project_root=calibration_root,
        panel_path=calibration_root / "panel.parquet",
        panel_audit_path=calibration_root / "panel.parquet.audit.json",
        training_window_days=8,
    )
    calibration_panel = pd.concat(
        [_panel(start_day="2026-01-01", days=4), training_panel],
        ignore_index=True,
    )
    calibration_source = _persist_panel(calibration_config, calibration_panel)
    _, _, calibration_audit = validate_panel(
        calibration_source, calibration_config
    )

    assert (
        training_audit["evaluation_label_binding"]
        == calibration_audit["evaluation_label_binding"]
    )


def test_late_feature_is_rejected(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    frame = _panel()
    frame.loc[0, "feature_available_at_utc"] = (
        pd.Timestamp(frame.loc[0, "origin_timestamp"]) + pd.Timedelta(minutes=1)
    )

    source = _persist_panel(config, frame)
    with pytest.raises(ExogenousFineTuneError, match="Fuite PIT"):
        validate_panel(source, config)


def test_dst_day_is_audited_and_excluded_only_from_fit(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    # Origin 1 is in the training portion. The synthetic 23-hour suffix remains
    # a physical split day but cannot be sampled by fixed-24 Chronos training.
    source = _persist_panel(config, _panel(irregular_origin_index=1))
    validated, split, audit = validate_panel(source, config)

    assert len(split.train) == 3
    assert len(audit["irregular_fit_origins"]) == 1
    assert len(build_fit_inputs(validated, split.train, config)) == 2


def test_train_publishes_verifiable_lora_and_can_reload(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    loader = _RecordingLoader()
    _persist_panel(config, _panel())

    model_snapshot = _fake_model_snapshot(tmp_path)
    output = train_lora(
        config,
        pipeline_loader=loader,
        model_source_override=model_snapshot,
    )
    manifest = verify_bundle(output)

    assert manifest["training_window_days"] == 4
    assert manifest["evaluation_days"] == 2
    assert manifest["candidate_frozen_before_evaluation"] is True
    assert manifest["evaluation_role"] == "primary_predeclared"
    assert manifest["actual_future_used_as_input"] is False
    assert manifest["pit_audit_passed"] is True
    assert manifest["production_pit_evidence"] is False
    assert manifest["production_pipeline_evidence"] is False
    assert manifest["production_pipeline_evidence_detail"]["comparison_scope"] == (
        "raw_chronos2_base_vs_raw_lora"
    )
    assert manifest["production_pipeline_evidence_detail"]["promotion_eligible"] is False
    assert len(manifest["panel_audit_sha256"]) == 64
    assert manifest["source_hashes"]["FR"]["synthetic"] == "a" * 64
    assert manifest["source_audit_hashes"]["FR"]["synthetic"] == "b" * 64
    assert len(manifest["base_model_snapshot_sha256"]) == 64
    assert loader.pipeline.fit_kwargs is not None
    assert loader.pipeline.fit_kwargs["finetune_mode"] == "lora"
    assert loader.pipeline.fit_kwargs["min_past"] == 24
    assert len(loader.pipeline.fit_kwargs["inputs"]) == 3
    assert len(loader.pipeline.fit_kwargs["validation_inputs"]) == 1
    model_selection = manifest["training"]["model_selection"]
    assert model_selection == {
        "validation_used": True,
        "early_stopping_used": False,
        "strategy": "best_eval_loss_after_fixed_steps",
        "selection_metric": "eval_loss",
        "trainer_state_available": False,
        "trainer_state_relative_path": None,
        "trainer_state_sha256": None,
        "trainer_state_observed_step": None,
        "best_step": None,
        "best_eval_loss": None,
        "checkpoint_source": None,
        "max_steps_completed": None,
        "max_steps_expected": 2,
        "max_steps_completion_evidence": None,
    }

    reload_loader = _RecordingLoader()
    loaded = load_checkpoint(output, pipeline_loader=reload_loader, device_map="cpu")
    assert loaded is reload_loader.pipeline
    assert reload_loader.calls[0][0].endswith("checkpoint")
    assert reload_loader.calls[0][1]["local_files_only"] is True
    assert reload_loader.calls[0][1]["import_allowlist"] == [
        "chronos.chronos2.model"
    ]

    (model_snapshot / "model.safetensors").write_bytes(b"tampered-base-model")
    blocked_loader = _RecordingLoader()
    with pytest.raises(ExogenousFineTuneError, match="checksum divergent"):
        load_checkpoint(output, pipeline_loader=blocked_loader, device_map="cpu")
    assert blocked_loader.calls == []


def test_train_seals_best_validation_checkpoint_provenance(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    _persist_panel(config, _panel())
    loader = _RecordingLoader()
    loader.pipeline = _TrainerStatePipeline()

    output = train_lora(
        config,
        pipeline_loader=loader,
        model_source_override=_fake_model_snapshot(tmp_path),
    )

    manifest = verify_bundle(output)
    selection = manifest["training"]["model_selection"]
    assert selection["validation_used"] is True
    assert selection["early_stopping_used"] is False
    assert selection["strategy"] == "best_eval_loss_after_fixed_steps"
    assert selection["trainer_state_available"] is True
    assert selection["trainer_state_relative_path"] == (
        "checkpoint-2/trainer_state.json"
    )
    assert len(selection["trainer_state_sha256"]) == 64
    assert selection["trainer_state_observed_step"] == 2
    assert selection["best_step"] == 1
    assert selection["best_eval_loss"] == pytest.approx(0.125)
    assert selection["checkpoint_source"] == "checkpoint-1"
    assert selection["max_steps_completed"] == 2
    assert selection["max_steps_expected"] == 2
    assert selection["max_steps_completion_evidence"] == (
        "fit_returned_after_fixed_steps_with_matching_trainer_state"
    )

    state_path = output / selection["trainer_state_relative_path"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["best_metric"] = 0.5
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ExogenousFineTuneError, match="Provenance LoRA"):
        verify_bundle(output)


def test_train_rejects_incoherent_present_trainer_state(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    _persist_panel(config, _panel())
    loader = _RecordingLoader()
    loader.pipeline = _TrainerStatePipeline(declared_max_steps=3)

    with pytest.raises(ExogenousFineTuneError, match="max_steps divergent"):
        train_lora(
            config,
            pipeline_loader=loader,
            model_source_override=_fake_model_snapshot(tmp_path),
        )
    assert not config.output_directory.exists()


def test_retained_best_state_distinguishes_selection_step_from_fixed_budget(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    _persist_panel(config, _panel())
    loader = _RecordingLoader()
    loader.pipeline = _TrainerStatePipeline(retained_best_only=True)

    output = train_lora(
        config,
        pipeline_loader=loader,
        model_source_override=_fake_model_snapshot(tmp_path),
    )

    selection = verify_bundle(output)["training"]["model_selection"]
    assert selection["trainer_state_relative_path"] == (
        "checkpoint-1/trainer_state.json"
    )
    assert selection["trainer_state_observed_step"] == 1
    assert selection["best_step"] == 1
    assert selection["max_steps_completed"] == 2
    assert selection["max_steps_expected"] == 2


def test_publish_evaluation_evidence_uses_exact_physical_holdout(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    loader = _RecordingLoader()
    source = _persist_panel(config, _panel())
    output = train_lora(
        config,
        pipeline_loader=loader,
        model_source_override=_fake_model_snapshot(tmp_path),
    )
    validated, split, _ = validate_panel(source, config)
    holdout = validated[validated[config.origin_column].isin(split.evaluation)]
    rows: list[dict[str, object]] = []
    for origin, group in holdout.groupby(config.origin_column, sort=True):
        ordered = group.sort_values(config.timestamp_column).iloc[-24:]
        for row in ordered.itertuples(index=False):
            actual = float(row.target)
            rows.append(
                {
                    "delivery_start_utc": row.timestamp,
                    "forecast_origin_utc": origin,
                    "actual": actual,
                    "baseline_q10": actual - 5,
                    "baseline_q50": actual - 1,
                    "baseline_q90": actual + 5,
                    "candidate_q10": actual - 4,
                    "candidate_q50": actual,
                    "candidate_q90": actual + 4,
                }
            )
    evidence = pd.DataFrame(rows, columns=EVALUATION_COLUMNS)

    path = publish_evaluation_evidence(output, evidence)

    assert path.name == "evaluation_predictions.csv.gz"
    manifest = json.loads(
        (output / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evaluation_evidence"]["physical_days"] == 2
    assert manifest["evaluation_evidence"]["rows"] == 48
    assert manifest["evaluation_evidence"]["dst_days"] == []

    manifest["production_pipeline_evidence"] = True
    manifest["candidate_output_stage"] = "exogenous_residual_corrected"
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ExogenousFineTuneError, match="downgrade|preuve du pipeline final"):
        publish_evaluation_evidence(output, evidence, overwrite=True)


def test_local_model_id_resolves_pinned_snapshot_without_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(_write_config(tmp_path))
    cache = tmp_path / "hf-cache"
    snapshot = (
        cache
        / "models--amazon--chronos-2"
        / "snapshots"
        / "test-revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))

    assert resolve_local_model_source(config) == snapshot.resolve()


def test_missing_peft_refuses_silent_full_finetune(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    _persist_panel(config, _panel())
    with patch("chronos2_exogenous.lora_finetune.importlib.util.find_spec", return_value=None):
        with pytest.raises(ExogenousFineTuneError, match="PEFT n'est pas installe"):
            train_lora(config)


def test_storm_or_realized_future_cannot_enter_schema(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["known_future_covariates"] = ["storm_price"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExogenousFineTuneError, match="interdits"):
        load_config(path)

    payload["data"]["known_future_covariates"] = ["wind_actual"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExogenousFineTuneError, match="realisees"):
        load_config(path)


def test_yaml_cannot_upgrade_research_audit_to_production(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["production_pit_evidence"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    source = _persist_panel(config, _panel(), production_ready=False)

    with pytest.raises(ExogenousFineTuneError, match="n'est pas production-ready"):
        validate_panel(source, config)


def test_effective_production_evidence_is_derived_from_sidecar(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel(), production_ready=True)

    _, _, audit = validate_panel(source, config)

    assert config.production_pit_evidence is False
    assert audit["production_pit_evidence"] is True


def test_panel_audit_hash_must_match_exact_panel_bytes(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel())
    config.panel_path.write_bytes(config.panel_path.read_bytes() + b"tampered")

    with pytest.raises(ExogenousFineTuneError, match="panel_sha256"):
        validate_panel(source, config)


def _bind_separate_canonical_target_cache(
    config: object,
    source: pd.DataFrame,
) -> Path:
    timestamps = pd.to_datetime(source["timestamp"], utc=True)
    first = timestamps.min()
    source["target"] = (
        (timestamps - first) / pd.Timedelta(hours=1)
    ).astype(float)
    source.to_parquet(config.panel_path, index=False)
    target = (
        source.loc[:, ["timestamp", "target"]]
        .drop_duplicates("timestamp")
        .rename(columns={"target": "value"})
        .sort_values("timestamp")
    )
    target_path = config.panel_path.parent / "canonical_target.csv.gz"
    target.to_csv(target_path, index=False, compression="gzip")
    audit = json.loads(config.panel_audit_path.read_text(encoding="utf-8"))
    audit["panel_sha256"] = _sha256(config.panel_path)
    audit["target_sources"]["FR"] = {
        "source_path": str(target_path),
        "source_sha256": _sha256(target_path),
    }
    audit["target_contracts"]["FR"]["cache_path"] = str(target_path)
    config.panel_audit_path.write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return target_path


def test_append_only_target_cache_is_verified_against_sealed_panel(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel())
    target_path = _bind_separate_canonical_target_cache(config, source)
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
    pd.concat([current, following], ignore_index=True).to_csv(
        target_path, index=False, compression="gzip"
    )

    _, _, audit = validate_panel(pd.read_parquet(config.panel_path), config)

    verification = audit["upstream_panel_audit"]["target_cache_verification"]["FR"]
    assert verification["mode"] == "panel_target_equivalence"
    assert verification["max_abs_delta"] == 0.0


def test_quarter_cent_hourly_aggregation_is_market_rounding_equivalent(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel())
    target_path = _bind_separate_canonical_target_cache(config, source)
    current = pd.read_csv(target_path)
    current.loc[0, "value"] = float(current.loc[0, "value"]) + 0.0025
    current.to_csv(target_path, index=False, compression="gzip")

    _, _, audit = validate_panel(pd.read_parquet(config.panel_path), config)

    verification = audit["upstream_panel_audit"]["target_cache_verification"]["FR"]
    assert verification["rounding_equivalent_timestamps"] == 1
    assert verification["max_abs_delta"] == pytest.approx(0.0025)


def test_target_cache_revision_on_panel_scope_is_rejected(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    source = _persist_panel(config, _panel())
    target_path = _bind_separate_canonical_target_cache(config, source)
    current = pd.read_csv(target_path)
    current.loc[0, "value"] = float(current.loc[0, "value"]) + 1.0
    current.to_csv(target_path, index=False, compression="gzip")

    with pytest.raises(ExogenousFineTuneError, match="valeurs utilisees"):
        validate_panel(pd.read_parquet(config.panel_path), config)
