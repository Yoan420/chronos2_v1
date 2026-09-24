from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

import chronos2_exogenous.training_snapshot_recovery as recovery
from chronos2_exogenous.governance import validate_experiment_manifest
from chronos2_exogenous.lora_finetune import sha256_directory, verify_bundle
from chronos2_exogenous.training_snapshot_recovery import (
    RECOVERY_CONFIG_NAME,
    RECOVERY_MANIFEST_NAME,
    RECOVERY_REFERENCE_KEY,
    TrainingSnapshotRecoveryError,
    recover_training_snapshot_and_prepare_zones,
    verify_recovered_training_snapshot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _evaluated_bundle(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "experiment" / "artifact"
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "C:/models/revision-pinned",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["self_attention.q"],
    }
    _write_json(checkpoint / "adapter_config.json", adapter_config)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"rank-8-adapter")
    retained = source / "checkpoint-400"
    retained.mkdir()
    _write_json(retained / "adapter_config.json", adapter_config)
    (retained / "adapter_model.safetensors").write_bytes(b"rank-8-adapter")
    _write_json(
        retained / "trainer_state.json",
        {
            "global_step": 400,
            "max_steps": 500,
            "best_global_step": 400,
            "best_metric": 0.05,
            "best_model_checkpoint": str(retained),
            "log_history": [{"step": 400, "eval_loss": 0.05}],
            "stateful_callbacks": {},
        },
    )

    schema = {
        "format_version": 1,
        "timestamp_column": "timestamp",
        "origin_column": "origin_timestamp",
        "item_column": "item_id",
        "feature_available_at_column": "feature_available_at_utc",
        "target_columns": ["target"],
        "known_future_covariates": ["weather"],
        "past_only_covariates": [],
        "timezone": "Europe/Paris",
        "cutoff_local_time": "08:00",
        "frequency": "h",
        "context_length": 48,
        "prediction_length": 24,
    }
    _write_json(source / "schema.json", schema)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    panel = inputs / "panel.parquet"
    panel.write_bytes(b"immutable-panel")
    panel_sha = _sha256(panel)
    audit = inputs / "panel.parquet.audit.json"
    _write_json(
        audit,
        {
            "schema_version": 1,
            "layout": "per_zone",
            "zones": ["FR", "DE", "BE", "NL"],
            "panel_sha256": panel_sha,
        },
    )

    manifest = {
        "format_version": 1,
        "created_at_utc": "2026-09-03T18:00:00+00:00",
        "experiment_id": "rank8-reference",
        "evaluation_role": "primary_predeclared",
        "model_id": "amazon/chronos-2",
        "model_revision": "revision-pinned",
        "base_model_resolved_locally": True,
        "base_model_snapshot_sha256": "b" * 64,
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
        "production_pit_evidence_detail": {"derived_from_panel_audit": False},
        "production_pipeline_evidence": False,
        "production_pipeline_evidence_detail": {"promotion_eligible": False},
        "checkpoint_relative_path": "checkpoint",
        "checkpoint_sha256": sha256_directory(checkpoint),
        "schema_relative_path": "schema.json",
        "schema_sha256": _sha256(source / "schema.json"),
        "panel_sha256": panel_sha,
        "panel_audit_sha256": _sha256(audit),
        "source_hashes": {"FR": {"weather": "c" * 64}},
        "source_audit_hashes": {"FR": {"weather": "d" * 64}},
        "source_cutoff_timezones": {"FR": {"weather": "Europe/Paris"}},
        "panel_pack": "full",
        "target_sources": {"FR": {"source_sha256": "e" * 64}},
        "target_contracts": {"FR": {"series": "canonical-target"}},
        "panel_audit_summary": {
            "audit_path": str(audit.resolve()),
            "zones": ["FR", "DE", "BE", "NL"],
        },
        "splits": {
            "train": {"count": 335},
            "validation": {"count": 30},
            "evaluation_holdout": {"count": 365},
        },
        "pit_audit": {
            "items": ["BE", "DE", "FR", "NL"],
            "production_pit_evidence": False,
        },
        "training": {
            "learning_rate": 1e-5,
            "num_steps": 500,
            "batch_size": 64,
            "seed": 42,
            "lora_config": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["self_attention.q"],
            },
            "train_samples": 100,
            "validation_samples": 10,
        },
    }

    derived = {
        "evaluation_predictions.csv.gz": b"evidence",
        "evaluation_daily.csv.gz": b"daily",
        "evaluation_metrics.json": b"metrics",
        "evaluation_report.html": b"report",
    }
    for name, content in derived.items():
        (source / name).write_bytes(content)
    cache = source / "evaluation_cache"
    cache.mkdir()
    (cache / "baseline.part").write_bytes(b"cache")
    manifest["evaluation_evidence"] = {
        "relative_path": "evaluation_predictions.csv.gz",
        "sha256": _sha256(source / "evaluation_predictions.csv.gz"),
        "rows": 8760,
        "physical_days": 365,
    }
    _write_json(source / "experiment_manifest.json", manifest)
    evaluation_manifest = {
        "format_version": 1,
        "kind": "chronos2_exogenous_lora_rolling365_evaluation",
        "experiment_id": manifest["experiment_id"],
        "item_id": "FR",
        "evaluation_label_resolution": None,
        "artifacts": {
            role: {
                "relative_path": name,
                "sha256": _sha256(source / name),
            }
            for role, name in {
                "evidence": "evaluation_predictions.csv.gz",
                "daily": "evaluation_daily.csv.gz",
                "metrics": "evaluation_metrics.json",
                "report": "evaluation_report.html",
            }.items()
        },
        "bundle_manifest_sha256": _sha256(source / "experiment_manifest.json"),
    }
    _write_json(source / "evaluation_manifest.json", evaluation_manifest)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "rank8.yaml"
    config = {
        "format_version": 1,
        "experiment_id": manifest["experiment_id"],
        "evaluation_role": "primary_predeclared",
        "project_root": "..",
        "data": {
            "panel_path": "inputs/panel.parquet",
            "panel_audit_path": "inputs/panel.parquet.audit.json",
            "timestamp_column": "timestamp",
            "origin_column": "origin_timestamp",
            "item_column": "item_id",
            "feature_available_at_column": "feature_available_at_utc",
            "target_columns": ["target"],
            "known_future_covariates": ["weather"],
            "past_only_covariates": [],
            "timezone": "Europe/Paris",
            "cutoff_local_time": "08:00",
            "frequency": "h",
            "context_length": 48,
            "prediction_length": 24,
            "training_window_days": 365,
            "validation_days": 30,
            "evaluation_days": 365,
            "require_consecutive_origins": True,
            "require_complete_known_future": True,
            "production_pit_evidence": False,
        },
        "model": {
            "model_id": "amazon/chronos-2",
            "revision": "revision-pinned",
            "local_files_only": True,
            "device_map": "auto",
        },
        "training": {
            "finetune_mode": "lora",
            "learning_rate": 1e-5,
            "num_steps": 500,
            "batch_size": 64,
            "seed": 42,
            "lora_config": manifest["training"]["lora_config"],
        },
        "output": {"directory": "experiment/artifact"},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    verify_bundle(source)
    return source, config_path


def test_recovery_authenticates_raw_evaluation_and_creates_verifiable_zone_clones(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "recovered"

    result = recover_training_snapshot_and_prepare_zones(
        source,
        config_reference=config,
        zones=["FR", "DE", "BE", "NL"],
        output_root=output,
    )

    assert result.created is True
    assert {item.zone for item in result.artifacts} == {"FR", "DE", "BE", "NL"}
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before
    sidecar = verify_recovered_training_snapshot(result.snapshot_directory)
    assert sidecar["source_was_modified"] is False
    assert sidecar["evaluation_outputs_copied"] is False
    assert sidecar["configuration_reference"]["claimed_as_original_training_bytes"] is False
    assert {record["path"] for record in sidecar["excluded_derived_files"]} >= {
        "evaluation_predictions.csv.gz",
        "evaluation_manifest.json",
        "evaluation_cache/baseline.part",
    }
    snapshot_manifest = verify_bundle(result.snapshot_directory)
    assert "evaluation_evidence" not in snapshot_manifest
    assert RECOVERY_REFERENCE_KEY in snapshot_manifest
    assert not (result.snapshot_directory / "evaluation_cache").exists()
    assert (result.snapshot_directory / RECOVERY_CONFIG_NAME).is_file()
    assert (result.snapshot_directory / RECOVERY_MANIFEST_NAME).is_file()
    for artifact in result.artifacts:
        clone = verify_bundle(artifact.path)
        assert clone["zone"] == artifact.zone
        validate_experiment_manifest(clone, zone=artifact.zone)


def test_recovery_is_idempotent_but_refuses_tampered_existing_snapshot(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    output = tmp_path / "recovered"
    first = recover_training_snapshot_and_prepare_zones(
        source, config_reference=config, zones=["FR"], output_root=output
    )
    second = recover_training_snapshot_and_prepare_zones(
        source, config_reference=config, zones=["FR"], output_root=output
    )
    assert first.created is True
    assert second.created is False
    assert second.artifacts[0].created is False

    (second.snapshot_directory / "checkpoint" / "adapter_model.safetensors").write_bytes(
        b"tampered"
    )
    with pytest.raises(Exception, match="checksum mismatch|divergent"):
        recover_training_snapshot_and_prepare_zones(
            source, config_reference=config, zones=["FR"], output_root=output
        )


def test_recovery_idempotence_is_bound_to_exact_source_and_config_paths(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    output = tmp_path / "recovered"
    result = recover_training_snapshot_and_prepare_zones(
        source, config_reference=config, zones=["FR"], output_root=output
    )
    sidecar_path = result.snapshot_directory / RECOVERY_MANIFEST_NAME
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["source_artifact_path"] = str(tmp_path / "same-bytes-other-source")
    _write_json(sidecar_path, sidecar)
    manifest_path = result.snapshot_directory / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[RECOVERY_REFERENCE_KEY]["sha256"] = _sha256(sidecar_path)
    _write_json(manifest_path, manifest)
    verify_recovered_training_snapshot(result.snapshot_directory)

    with pytest.raises(TrainingSnapshotRecoveryError, match="autre source/configuration"):
        recover_training_snapshot_and_prepare_zones(
            source, config_reference=config, zones=["FR"], output_root=output
        )


def test_recovery_rejects_tampered_evaluation_and_unknown_root_entry(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    (source / "evaluation_metrics.json").write_bytes(b"tampered")
    with pytest.raises(TrainingSnapshotRecoveryError, match="metrics.*divergent"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out-a",
        )

    source, config = _evaluated_bundle(tmp_path / "second")
    (source / "unclassified.bin").write_bytes(b"must-not-be-silently-dropped")
    with pytest.raises(TrainingSnapshotRecoveryError, match="non classee"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out-b",
        )


@pytest.mark.parametrize("mutation", ["remove", "replace_sha"])
def test_recovery_cannot_be_enabled_by_removing_or_repointing_evaluation_evidence(
    tmp_path: Path, mutation: str
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    manifest_path = source / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "remove":
        manifest.pop("evaluation_evidence")
    else:
        manifest["evaluation_evidence"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    evaluation_path = source / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["bundle_manifest_sha256"] = _sha256(manifest_path)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(
        TrainingSnapshotRecoveryError,
        match="evaluation_evidence.*absente|SHA de evaluation_predictions",
    ):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "value", ["../evaluation_predictions.csv.gz", "C:evaluation_predictions.csv.gz"]
)
def test_recovery_rejects_path_traversal_and_drive_relative_paths(value: str) -> None:
    with pytest.raises(TrainingSnapshotRecoveryError, match="non confine"):
        recovery._safe_relative(value, label="adversarial path")


def test_recovery_rejects_semantically_different_config(tmp_path: Path) -> None:
    source, config = _evaluated_bundle(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["training"]["lora_config"]["r"] = 16
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingSnapshotRecoveryError, match="training.lora_config"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "validation_days"), 31, "validation_days"),
        (
            ("data", "require_consecutive_origins"),
            False,
            "require_consecutive_origins",
        ),
        (
            ("data", "require_complete_known_future"),
            False,
            "require_complete_known_future",
        ),
        (("data", "production_pit_evidence"), True, "production_pit_evidence"),
        (
            ("data", "allow_unresolved_final_evaluation_day"),
            True,
            "allow_unresolved_final_evaluation_day",
        ),
        (("model", "local_files_only"), False, "local_files_only"),
    ],
)
def test_recovery_rejects_post_hoc_config_semantic_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TrainingSnapshotRecoveryError, match=message):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "relative",
    [
        "checkpoint/adapter_model.safetensors",
        "checkpoint-400/adapter_config.json",
        "checkpoint-400/adapter_model.safetensors",
        "checkpoint-400/trainer_state.json",
    ],
)
def test_recovery_rejects_incomplete_final_or_retained_checkpoint(
    tmp_path: Path, relative: str
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    (source / relative).unlink()
    if relative.startswith("checkpoint/"):
        manifest_path = source / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_sha256"] = sha256_directory(
            source / "checkpoint"
        )
        _write_json(manifest_path, manifest)
        evaluation_path = source / "evaluation_manifest.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["bundle_manifest_sha256"] = _sha256(manifest_path)
        _write_json(evaluation_path, evaluation)

    with pytest.raises(TrainingSnapshotRecoveryError, match="absents|incomplet"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


def test_recovery_rejects_unrecognised_file_hidden_in_checkpoint(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    hidden = source / "checkpoint" / "evaluation_predictions.csv.gz"
    hidden.write_bytes(b"derived-output-disguised-as-training")
    manifest_path = source / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = sha256_directory(source / "checkpoint")
    _write_json(manifest_path, manifest)
    evaluation_path = source / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["bundle_manifest_sha256"] = _sha256(manifest_path)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(TrainingSnapshotRecoveryError, match="non reconnu"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


def test_recovery_rejects_manifest_config_rank_not_matching_adapter(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    adapter_path = source / "checkpoint" / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter["r"] = 16
    _write_json(adapter_path, adapter)
    manifest_path = source / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = sha256_directory(source / "checkpoint")
    _write_json(manifest_path, manifest)
    evaluation_path = source / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["bundle_manifest_sha256"] = _sha256(manifest_path)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(TrainingSnapshotRecoveryError, match="checkpoint.r"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


def test_recovery_rejects_final_weights_not_matching_retained_best(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    (source / "checkpoint" / "adapter_model.safetensors").write_bytes(
        b"different-final-adapter"
    )
    manifest_path = source / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = sha256_directory(source / "checkpoint")
    _write_json(manifest_path, manifest)
    evaluation_path = source / "evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["bundle_manifest_sha256"] = _sha256(manifest_path)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(TrainingSnapshotRecoveryError, match="meilleur checkpoint"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


def test_recovery_detects_config_change_during_copy_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    real_copy = recovery._copy_included_training_entries

    def mutate_config_after_training_copy(source_path: Path, staging: Path) -> None:
        real_copy(source_path, staging)
        config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    monkeypatch.setattr(
        recovery,
        "_copy_included_training_entries",
        mutate_config_after_training_copy,
    )
    output = tmp_path / "out"
    with pytest.raises(TrainingSnapshotRecoveryError, match="configuration.*change"):
        recover_training_snapshot_and_prepare_zones(
            source, config_reference=config, zones=["FR"], output_root=output
        )
    assert not (output / "training_snapshot" / "artifact").exists()
    assert not (output / "zones" / "FR" / "artifact").exists()


def test_recovery_verifier_rejects_extra_empty_directory(tmp_path: Path) -> None:
    source, config = _evaluated_bundle(tmp_path)
    result = recover_training_snapshot_and_prepare_zones(
        source,
        config_reference=config,
        zones=["FR"],
        output_root=tmp_path / "out",
    )
    (result.snapshot_directory / "evaluation_cache").mkdir()
    with pytest.raises(TrainingSnapshotRecoveryError, match="repertoires.*extra"):
        verify_recovered_training_snapshot(result.snapshot_directory)


def test_recovery_rejects_hardlinked_checkpoint_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    real_copy = recovery._copy_included_training_entries

    def introduce_hardlink(source_path: Path, staging: Path) -> None:
        real_copy(source_path, staging)
        relative = Path("checkpoint/adapter_model.safetensors")
        copied = staging / relative
        copied.unlink()
        copied.hardlink_to(source_path / relative)

    monkeypatch.setattr(recovery, "_copy_included_training_entries", introduce_hardlink)
    with pytest.raises(TrainingSnapshotRecoveryError, match="Hardlink"):
        recover_training_snapshot_and_prepare_zones(
            source,
            config_reference=config,
            zones=["FR"],
            output_root=tmp_path / "out",
        )


def test_recovery_rejects_directory_symlink_source_when_supported(
    tmp_path: Path,
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    link = tmp_path / "artifact-link"
    junction = False
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy can forbid symlinks.
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {exc}")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode != 0:
            pytest.skip(f"directory links unavailable: {created.stderr}")
        junction = True

    try:
        with pytest.raises(TrainingSnapshotRecoveryError, match="lien.*reparse"):
            recover_training_snapshot_and_prepare_zones(
                link,
                config_reference=config,
                zones=["FR"],
                output_root=tmp_path / "out",
            )
        with pytest.raises(Exception, match="lien.*reparse|junction"):
            verify_bundle(link)
    finally:
        if junction:
            os.rmdir(link)
        else:
            link.unlink(missing_ok=True)


def test_recovery_detects_source_change_during_copy_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config = _evaluated_bundle(tmp_path)
    real_copy = recovery._copy_included_training_entries

    def mutate_after_copy(source_path: Path, staging: Path) -> None:
        real_copy(source_path, staging)
        (source_path / "checkpoint-400" / "trainer_state.json").write_text(
            '{"global_step": 401}\n', encoding="utf-8"
        )

    monkeypatch.setattr(recovery, "_copy_included_training_entries", mutate_after_copy)
    output = tmp_path / "out"
    with pytest.raises(TrainingSnapshotRecoveryError, match="change pendant la copie"):
        recover_training_snapshot_and_prepare_zones(
            source, config_reference=config, zones=["FR"], output_root=output
        )
    assert not (output / "training_snapshot" / "artifact").exists()
    assert not (output / "zones" / "FR" / "artifact").exists()


def test_recovery_rolls_back_new_snapshot_if_prepare_zones_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config = _evaluated_bundle(tmp_path)

    def fail_prepare(*args: object, **kwargs: object) -> object:
        raise recovery.ZoneArtifactPreparationError("simulated PrepareZones failure")

    monkeypatch.setattr(recovery, "prepare_zone_artifacts", fail_prepare)
    output = tmp_path / "out"
    with pytest.raises(Exception, match="simulated PrepareZones failure"):
        recover_training_snapshot_and_prepare_zones(
            source, config_reference=config, zones=["FR"], output_root=output
        )
    assert not (output / "training_snapshot" / "artifact").exists()
    assert not (output / "zones" / "FR" / "artifact").exists()
