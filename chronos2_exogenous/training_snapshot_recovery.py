"""Recover an immutable training-only LoRA snapshot from a raw evaluation run.

The rolling evaluator appends evidence to ``experiment_manifest.json`` and
writes derived files next to the trained adapter.  That is intentional, but it
means a late request for per-zone forks cannot use ``PrepareZones``: the
original directory is no longer pristine.

Recovery is deliberately narrower than a generic copy/filter operation.  It
accepts exactly one completed *raw* evaluation, independently authenticates
all of its evidence, checks a semantically equivalent configuration reference,
then reconstructs the pre-evaluation manifest.  Every included and excluded
file is SHA-256 inventoried in a lineage sidecar.  The evaluated source is
snapshotted before and after copying and is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .lora_finetune import (
    ExogenousFineTuneError,
    ExogenousFineTuneConfig,
    _model_selection_provenance,
    config_summary,
    load_config,
    verify_bundle,
)
from .zone_artifacts import (
    PreparedZoneArtifact,
    ZoneArtifactPreparationError,
    _is_link_or_reparse,
    _tree_snapshot,
    _write_json_atomic,
    prepare_zone_artifacts,
)


RECOVERY_SCHEMA_VERSION = 1
RECOVERY_MANIFEST_NAME = "training_snapshot_recovery_manifest.json"
RECOVERY_CONFIG_NAME = "training_config_reference.yaml"
RECOVERY_REFERENCE_KEY = "training_snapshot_recovery"

_RAW_EVALUATION_ROOTS = {
    "evaluation_predictions.csv.gz",
    "evaluation_daily.csv.gz",
    "evaluation_metrics.json",
    "evaluation_report.html",
    "evaluation_manifest.json",
    "evaluation_cache",
}
_RAW_EVALUATION_ARTIFACTS = {
    "evidence": "evaluation_predictions.csv.gz",
    "daily": "evaluation_daily.csv.gz",
    "metrics": "evaluation_metrics.json",
    "report": "evaluation_report.html",
}
_MANIFEST_EVALUATION_KEYS = {
    "evaluation_evidence",
    "evaluation_label_resolution",
}
_TRAINING_DIRECTORY_PATTERN = re.compile(r"checkpoint(?:-\d+)?")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ALLOWED_CHECKPOINT_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scaler.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}

# These fields are immutable training identity.  Later raw/final evaluation is
# allowed to append or update its own evidence, but may never alter this
# projection without invalidating the recovered snapshot reference.
_IMMUTABLE_TRAINING_KEYS = (
    "format_version",
    "created_at_utc",
    "experiment_id",
    "evaluation_role",
    "model_id",
    "model_revision",
    "base_model_resolved_locally",
    "base_model_snapshot_sha256",
    "finetune_mode",
    "training_window_days",
    "evaluation_days",
    "cutoff_local_time",
    "candidate_frozen_before_evaluation",
    "feature_selection_frozen_before_evaluation",
    "actual_future_used_as_input",
    "storm_used_for_input",
    "storm_used_for_selection",
    "mkonline_used_for_input",
    "pit_audit_passed",
    "production_pit_evidence",
    "production_pit_evidence_detail",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "schema_relative_path",
    "schema_sha256",
    "panel_sha256",
    "panel_audit_sha256",
    "source_hashes",
    "source_audit_hashes",
    "source_cutoff_timezones",
    "panel_pack",
    "target_sources",
    "target_contracts",
    "panel_audit_summary",
    "splits",
    "pit_audit",
    "training",
)


class TrainingSnapshotRecoveryError(RuntimeError):
    """Raised when evaluated bytes cannot prove a pristine training snapshot."""


@dataclass(frozen=True)
class RecoveredTrainingSnapshot:
    source_directory: Path
    snapshot_directory: Path
    output_root: Path
    recovery_manifest_path: Path
    source_tree_sha256: str
    created: bool
    artifacts: tuple[PreparedZoneArtifact, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise TrainingSnapshotRecoveryError(f"{label} absent ou non regulier: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingSnapshotRecoveryError(f"{label} illisible: {path}.") from exc
    if not isinstance(value, dict):
        raise TrainingSnapshotRecoveryError(f"{label} doit etre un objet JSON.")
    return value


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingSnapshotRecoveryError(f"{label}: chemin relatif absent.")
    relative = Path(value)
    if (
        relative.is_absolute()
        or bool(relative.drive)
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise TrainingSnapshotRecoveryError(f"{label}: chemin non confine: {value!r}.")
    return value


def _resolve_without_links(value: str | Path, *, label: str) -> Path:
    """Resolve a path only after rejecting symlink/junction components."""

    absolute = Path(value).expanduser().absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise TrainingSnapshotRecoveryError(
                f"{label}: lien symbolique/junction/reparse interdit: {component}."
            )
    return absolute.resolve()


def _file_record(path: Path, *, root: Path, role: str) -> dict[str, object]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise TrainingSnapshotRecoveryError(f"Fichier non regulier refuse: {path}.")
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _immutable_training_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _IMMUTABLE_TRAINING_KEYS if key not in manifest]
    if missing:
        raise TrainingSnapshotRecoveryError(
            "Manifeste d'entrainement incomplet pour la recuperation: "
            + ", ".join(missing)
            + "."
        )
    return {key: manifest[key] for key in _IMMUTABLE_TRAINING_KEYS}


def _schema_expected(config: ExogenousFineTuneConfig) -> dict[str, Any]:
    return {
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


def _adapter_semantic_mismatches(
    source: Path, config: ExogenousFineTuneConfig
) -> list[str]:
    """Cross-check YAML/manifest LoRA claims against every saved adapter."""

    mismatches: list[str] = []
    checkpoints = sorted(
        entry
        for entry in source.iterdir()
        if entry.is_dir() and _TRAINING_DIRECTORY_PATTERN.fullmatch(entry.name)
    )
    for checkpoint in checkpoints:
        adapter_path = checkpoint / "adapter_config.json"
        if not adapter_path.is_file():
            # The exact closure check reports the more useful missing-file
            # error later; avoid turning it into an unhandled read failure.
            continue
        adapter = _read_json(
            adapter_path, label=f"adapter config {checkpoint.name}"
        )
        if str(adapter.get("peft_type", "")).upper() != "LORA":
            mismatches.append(f"{checkpoint.name}.peft_type")
        for key, expected in config.lora_config.items():
            observed = adapter.get(key)
            if key == "target_modules":
                if (
                    not isinstance(expected, Sequence)
                    or isinstance(expected, (str, bytes))
                    or any(not isinstance(value, str) for value in expected)
                    or len(expected) != len(set(expected))
                    or not isinstance(observed, list)
                    or any(not isinstance(value, str) for value in observed)
                    or len(observed) != len(set(observed))
                    or set(observed) != set(expected)
                ):
                    mismatches.append(f"{checkpoint.name}.target_modules")
            elif observed != expected:
                mismatches.append(f"{checkpoint.name}.{key}")
        base_value = adapter.get("base_model_name_or_path")
        if (
            config.model_revision
            and (
                not isinstance(base_value, str)
                or Path(base_value).name != config.model_revision
            )
        ):
            mismatches.append(f"{checkpoint.name}.base_model_name_or_path")
    return mismatches


def _validate_config_reference(
    config: ExogenousFineTuneConfig,
    *,
    source: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if config.output_directory.resolve() != source:
        raise TrainingSnapshotRecoveryError(
            "La configuration de reference ne designe pas exactement l'artefact source."
        )
    manifest_pairs = {
        "experiment_id": config.experiment_id,
        "evaluation_role": config.evaluation_role,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "finetune_mode": "lora",
        "training_window_days": config.training_window_days,
        "evaluation_days": config.evaluation_days,
        "cutoff_local_time": config.cutoff_local_time,
    }
    mismatches = [
        key for key, expected in manifest_pairs.items() if manifest.get(key) != expected
    ]
    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise TrainingSnapshotRecoveryError("experiment_manifest.training absent.")
    training_pairs = {
        "learning_rate": config.learning_rate,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "lora_config": dict(config.lora_config),
    }
    mismatches += [
        f"training.{key}"
        for key, expected in training_pairs.items()
        if training.get(key) != expected
    ]
    mismatches += _adapter_semantic_mismatches(source, config)
    schema = _read_json(source / "schema.json", label="schema LoRA")
    if schema != _schema_expected(config):
        mismatches.append("schema.json")

    # The original YAML was not archived by the first version of the trainer,
    # so this is explicitly a post-hoc replay reference.  Only semantics that
    # are independently recoverable from the sealed training manifest may be
    # claimed as equivalent.  In particular, do not silently accept a
    # different validation split or weaker causal/PIT validation flags.
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        mismatches.append("splits")
    else:
        train_split = splits.get("train")
        validation_split = splits.get("validation")
        evaluation_split = splits.get("evaluation_holdout")
        if not isinstance(train_split, Mapping) or train_split.get("count") != (
            config.training_window_days - config.validation_days
        ):
            mismatches.append("data.validation_days/train split")
        if (
            not isinstance(validation_split, Mapping)
            or validation_split.get("count") != config.validation_days
        ):
            mismatches.append("data.validation_days/validation split")
        if (
            not isinstance(evaluation_split, Mapping)
            or evaluation_split.get("count") != config.evaluation_days
        ):
            mismatches.append("data.evaluation_days/evaluation split")

    if config.require_consecutive_origins is not True:
        mismatches.append("data.require_consecutive_origins")
    if config.require_complete_known_future is not True:
        mismatches.append("data.require_complete_known_future")
    if config.local_files_only is not True or manifest.get(
        "base_model_resolved_locally"
    ) is not True:
        mismatches.append("model.local_files_only/base_model_resolved_locally")
    if config.production_pit_evidence is not bool(
        manifest.get("production_pit_evidence")
    ):
        mismatches.append("data.production_pit_evidence")
    binding = manifest.get("evaluation_label_binding")
    if binding is None:
        # Legacy rank-8 artefacts predate the two-phase unresolved-label
        # contract.  Absence therefore means the historical strict default.
        unresolved_allowed = False
    elif isinstance(binding, Mapping):
        unresolved_allowed = binding.get(
            "allow_unresolved_final_evaluation_day"
        ) is True
    else:
        mismatches.append("evaluation_label_binding")
        unresolved_allowed = False
    if config.allow_unresolved_final_evaluation_day is not unresolved_allowed:
        mismatches.append("data.allow_unresolved_final_evaluation_day")

    panel_path = config.panel_path.resolve()
    panel_audit_path = config.panel_audit_path.resolve()
    if (
        not panel_path.is_file()
        or _is_link_or_reparse(panel_path)
        or _sha256_file(panel_path) != manifest.get("panel_sha256")
    ):
        mismatches.append("data.panel_path/panel_sha256")
    if (
        not panel_audit_path.is_file()
        or _is_link_or_reparse(panel_audit_path)
        or _sha256_file(panel_audit_path) != manifest.get("panel_audit_sha256")
    ):
        mismatches.append("data.panel_audit_path/panel_audit_sha256")
    summary = manifest.get("panel_audit_summary")
    if not isinstance(summary, Mapping) or Path(
        str(summary.get("audit_path", ""))
    ).expanduser().resolve() != panel_audit_path:
        mismatches.append("panel_audit_summary.audit_path")
    if mismatches:
        raise TrainingSnapshotRecoveryError(
            "Configuration de reference divergente du training scelle: "
            + ", ".join(mismatches)
            + "."
        )
    return config_summary(config)


def _validate_raw_evaluation(
    source: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("production_pipeline_evidence") is not False:
        raise TrainingSnapshotRecoveryError(
            "Seul un backtest brut, avant pipeline final, peut etre recupere."
        )
    if manifest.get("candidate_output_stage") not in (None, ""):
        raise TrainingSnapshotRecoveryError(
            "candidate_output_stage derive interdit pour cette recuperation."
        )
    forbidden = sorted(
        key
        for key in ("raw_evaluation_evidence", "final_pipeline_evaluation")
        if key in manifest
    )
    if forbidden:
        raise TrainingSnapshotRecoveryError(
            "Preuves post-correcteur interdites: " + ", ".join(forbidden) + "."
        )
    evidence = manifest.get("evaluation_evidence")
    if not isinstance(evidence, Mapping):
        raise TrainingSnapshotRecoveryError(
            "evaluation_evidence brute absente du manifeste source."
        )
    evidence_path_value = _safe_relative(
        evidence.get("relative_path"), label="evaluation_evidence.relative_path"
    )
    if evidence_path_value != "evaluation_predictions.csv.gz":
        raise TrainingSnapshotRecoveryError(
            "evaluation_evidence doit pointer vers evaluation_predictions.csv.gz."
        )
    evidence_path = source / evidence_path_value
    evidence_sha = evidence.get("sha256")
    if (
        not isinstance(evidence_sha, str)
        or _SHA256_PATTERN.fullmatch(evidence_sha) is None
        or not evidence_path.is_file()
        or _sha256_file(evidence_path) != evidence_sha
    ):
        raise TrainingSnapshotRecoveryError(
            "SHA de evaluation_predictions.csv.gz divergent."
        )

    evaluation = _read_json(
        source / "evaluation_manifest.json", label="manifest d'evaluation brute"
    )
    source_manifest_sha = _sha256_file(source / "experiment_manifest.json")
    if evaluation.get("bundle_manifest_sha256") != source_manifest_sha:
        raise TrainingSnapshotRecoveryError(
            "evaluation_manifest ne lie pas le manifeste source courant."
        )
    if evaluation.get("experiment_id") != manifest.get("experiment_id"):
        raise TrainingSnapshotRecoveryError(
            "experiment_id divergent entre training et evaluation."
        )
    artifacts = evaluation.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        _RAW_EVALUATION_ARTIFACTS
    ):
        raise TrainingSnapshotRecoveryError(
            "Inventaire des artefacts d'evaluation brut non exact."
        )
    for role, expected_name in _RAW_EVALUATION_ARTIFACTS.items():
        reference = artifacts.get(role)
        if not isinstance(reference, Mapping):
            raise TrainingSnapshotRecoveryError(
                f"Reference evaluation {role} invalide."
            )
        relative = _safe_relative(
            reference.get("relative_path"), label=f"evaluation.artifacts.{role}"
        )
        digest = reference.get("sha256")
        path = source / relative
        if (
            relative != expected_name
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise TrainingSnapshotRecoveryError(
                f"Artefact d'evaluation {role} absent ou divergent."
            )
    if artifacts["evidence"].get("sha256") != evidence_sha:
        raise TrainingSnapshotRecoveryError(
            "La preuve evaluation differe entre les deux manifestes."
        )
    resolution = manifest.get("evaluation_label_resolution")
    if evaluation.get("evaluation_label_resolution") != resolution:
        raise TrainingSnapshotRecoveryError(
            "Resolution des labels divergente entre les manifestes."
        )
    return evaluation


def _classify_source_files(
    source: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        if _is_link_or_reparse(entry):
            raise TrainingSnapshotRecoveryError(
                f"Lien/reparse interdit dans la source: {entry}."
            )
        name = entry.name
        if name == "experiment_manifest.json":
            continue
        if name == "schema.json" or (
            entry.is_dir() and _TRAINING_DIRECTORY_PATTERN.fullmatch(name)
        ):
            role = "schema" if name == "schema.json" else "training_checkpoint"
            paths = [entry] if entry.is_file() else sorted(entry.rglob("*"))
            if entry.is_dir():
                nested_directories = [path for path in paths if path.is_dir()]
                unexpected_files = [
                    path
                    for path in paths
                    if path.is_file() and path.name not in _ALLOWED_CHECKPOINT_FILES
                ]
                if nested_directories or unexpected_files:
                    unexpected = [*nested_directories, *unexpected_files]
                    raise TrainingSnapshotRecoveryError(
                        f"Contenu checkpoint non reconnu dans {entry.name}: "
                        + ", ".join(
                            path.relative_to(source).as_posix() for path in unexpected
                        )
                        + "."
                    )
            for path in paths:
                if path.is_file():
                    included.append(_file_record(path, root=source, role=role))
            continue
        if name in _RAW_EVALUATION_ROOTS:
            paths = [entry] if entry.is_file() else sorted(entry.rglob("*"))
            for path in paths:
                if path.is_file():
                    record = _file_record(path, root=source, role="derived_evaluation")
                    record["exclusion_reason"] = "raw_evaluation_output"
                    excluded.append(record)
            continue
        raise TrainingSnapshotRecoveryError(
            f"Entree racine non classee refusee: {entry}."
        )
    required = {
        "schema.json",
        "checkpoint/adapter_config.json",
        "checkpoint/adapter_model.safetensors",
    }
    observed = {str(record["path"]) for record in included}
    missing = sorted(required - observed)
    if missing:
        raise TrainingSnapshotRecoveryError(
            "Fichiers de training requis absents: " + ", ".join(missing) + "."
        )
    retained = sorted(
        entry
        for entry in source.iterdir()
        if entry.is_dir() and re.fullmatch(r"checkpoint-\d+", entry.name)
    )
    for checkpoint in retained:
        checkpoint_required = {
            f"{checkpoint.name}/adapter_config.json",
            f"{checkpoint.name}/adapter_model.safetensors",
            f"{checkpoint.name}/trainer_state.json",
        }
        checkpoint_missing = sorted(checkpoint_required - observed)
        if checkpoint_missing:
            raise TrainingSnapshotRecoveryError(
                f"Checkpoint retenu incomplet {checkpoint.name}: "
                + ", ".join(checkpoint_missing)
                + "."
            )
    return included, excluded


def _validate_retained_model_selection(
    source: Path, config: ExogenousFineTuneConfig
) -> dict[str, Any]:
    """Validate the legacy retained best-checkpoint evidence fail-closed."""

    try:
        selection = _model_selection_provenance(
            source, expected_max_steps=config.num_steps
        )
    except ExogenousFineTuneError as exc:
        raise TrainingSnapshotRecoveryError(
            f"Provenance du checkpoint retenu invalide: {exc}"
        ) from exc
    if selection.get("trainer_state_available") is not True:
        raise TrainingSnapshotRecoveryError(
            "Provenance du checkpoint retenu absente: trainer_state.json requis."
        )
    checkpoint_name = selection.get("checkpoint_source")
    if not isinstance(checkpoint_name, str):
        raise TrainingSnapshotRecoveryError(
            "Provenance du meilleur checkpoint sans checkpoint_source."
        )
    final_weights = source / "checkpoint" / "adapter_model.safetensors"
    selected_weights = source / checkpoint_name / "adapter_model.safetensors"
    if (
        not selected_weights.is_file()
        or _sha256_file(final_weights) != _sha256_file(selected_weights)
    ):
        raise TrainingSnapshotRecoveryError(
            "Le checkpoint final ne correspond pas aux poids du meilleur "
            "checkpoint retenu."
        )
    return selection


def _copy_included_training_entries(source: Path, staging: Path) -> None:
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        if entry.name == "schema.json" or (
            entry.is_dir() and _TRAINING_DIRECTORY_PATTERN.fullmatch(entry.name)
        ):
            destination = staging / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination, copy_function=shutil.copy2)
            else:
                shutil.copy2(entry, destination)


def _assert_independent_files(
    source: Path, destination: Path, records: Sequence[Mapping[str, object]]
) -> None:
    for record in records:
        relative = Path(str(record["path"]))
        copied = destination / relative
        original = source / relative
        if (
            not copied.is_file()
            or copied.stat().st_size != record["size_bytes"]
            or _sha256_file(copied) != record["sha256"]
        ):
            raise TrainingSnapshotRecoveryError(
                f"Copie de training divergente: {copied}."
            )
        try:
            if os.path.samefile(original, copied):
                raise TrainingSnapshotRecoveryError(
                    f"Hardlink avec la source interdit: {copied}."
                )
        except OSError as exc:
            raise TrainingSnapshotRecoveryError(
                f"Independance physique invérifiable: {copied}."
            ) from exc


def verify_recovery_reference(
    run_directory: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the optional recovery reference embedded in a bundle manifest.

    This routine intentionally does not call :func:`verify_bundle`, allowing
    ``verify_bundle`` itself to invoke it without recursion.
    """

    run_dir = _resolve_without_links(
        run_directory, label="Artefact avec reference de recuperation"
    )
    reference = manifest.get(RECOVERY_REFERENCE_KEY)
    if not isinstance(reference, Mapping) or set(reference) != {
        "schema_version",
        "kind",
        "relative_path",
        "sha256",
    }:
        raise TrainingSnapshotRecoveryError("Reference de recuperation invalide.")
    if (
        reference.get("schema_version") != RECOVERY_SCHEMA_VERSION
        or reference.get("kind") != "chronos2_exogenous_training_snapshot_recovery"
        or reference.get("relative_path") != RECOVERY_MANIFEST_NAME
    ):
        raise TrainingSnapshotRecoveryError("Contrat de reference de recuperation invalide.")
    expected_sha = reference.get("sha256")
    sidecar_path = run_dir / RECOVERY_MANIFEST_NAME
    if (
        not isinstance(expected_sha, str)
        or _SHA256_PATTERN.fullmatch(expected_sha) is None
        or not sidecar_path.is_file()
        or _is_link_or_reparse(sidecar_path)
        or _sha256_file(sidecar_path) != expected_sha
    ):
        raise TrainingSnapshotRecoveryError("SHA du manifeste de recuperation divergent.")
    sidecar = _read_json(sidecar_path, label="manifest de recuperation")
    if (
        sidecar.get("schema_version") != RECOVERY_SCHEMA_VERSION
        or sidecar.get("kind")
        != "chronos2_exogenous_training_snapshot_recovery_lineage"
    ):
        raise TrainingSnapshotRecoveryError("Schema du manifeste de recuperation invalide.")

    contract = _immutable_training_contract(manifest)
    if (
        sidecar.get("immutable_training_contract") != contract
        or sidecar.get("immutable_training_contract_sha256")
        != _json_digest(contract)
    ):
        raise TrainingSnapshotRecoveryError(
            "Contrat immutable du training divergent de sa filiation."
        )
    records = sidecar.get("included_training_files")
    if not isinstance(records, list) or not records:
        raise TrainingSnapshotRecoveryError("Inventaire des fichiers inclus absent.")
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TrainingSnapshotRecoveryError("Enregistrement de fichier inclus invalide.")
        relative_text = _safe_relative(raw.get("path"), label="included_training_files")
        if relative_text in seen:
            raise TrainingSnapshotRecoveryError(f"Fichier inclus duplique: {relative_text}.")
        seen.add(relative_text)
        path = run_dir / relative_text
        digest = raw.get("sha256")
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not path.is_file()
            or _is_link_or_reparse(path)
            or path.stat().st_size != raw.get("size_bytes")
            or _sha256_file(path) != digest
        ):
            raise TrainingSnapshotRecoveryError(
                f"Fichier inclus absent ou divergent: {relative_text}."
            )
    config_reference = sidecar.get("configuration_reference")
    if not isinstance(config_reference, Mapping):
        raise TrainingSnapshotRecoveryError("Configuration de reference absente.")
    config_relative = _safe_relative(
        config_reference.get("snapshot_relative_path"),
        label="configuration_reference.snapshot_relative_path",
    )
    config_path = run_dir / config_relative
    if (
        config_relative != RECOVERY_CONFIG_NAME
        or not config_path.is_file()
        or _is_link_or_reparse(config_path)
        or _sha256_file(config_path) != config_reference.get("sha256")
        or config_reference.get("provenance_class")
        != "post_hoc_semantically_verified_reference"
        or config_reference.get("claimed_as_original_training_bytes") is not False
        or config_reference.get("resolved_semantics_sha256")
        != _json_digest(config_reference.get("resolved_semantics"))
    ):
        raise TrainingSnapshotRecoveryError("Configuration de reference divergente.")
    return sidecar


def verify_recovered_training_snapshot(
    snapshot_directory: str | Path,
) -> dict[str, Any]:
    """Verify a pristine recovered source, including its exact file closure."""

    snapshot = _resolve_without_links(
        snapshot_directory, label="Snapshot de training recupere"
    )
    manifest = verify_bundle(snapshot)
    sidecar = verify_recovery_reference(snapshot, manifest)
    if manifest.get("zone") is not None or any(
        key in manifest for key in _MANIFEST_EVALUATION_KEYS
    ):
        raise TrainingSnapshotRecoveryError(
            "Le snapshot recupere doit rester vierge et multi-zone."
        )
    core = dict(manifest)
    core.pop(RECOVERY_REFERENCE_KEY, None)
    if sidecar.get("reconstructed_training_manifest_core_sha256") != _json_digest(core):
        raise TrainingSnapshotRecoveryError(
            "Le manifeste de training reconstruit diverge de la filiation."
        )
    expected_files = {
        str(record["path"]) for record in sidecar["included_training_files"]
    } | {
        "experiment_manifest.json",
        RECOVERY_MANIFEST_NAME,
        RECOVERY_CONFIG_NAME,
    }
    for path in snapshot.rglob("*"):
        if _is_link_or_reparse(path):
            raise TrainingSnapshotRecoveryError(
                f"Lien/junction/reparse interdit dans le snapshot: {path}."
            )
    actual_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise TrainingSnapshotRecoveryError(
            "Closure du snapshot recupere divergente: "
            f"extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}."
        )
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise TrainingSnapshotRecoveryError(
            "Closure des repertoires du snapshot divergente: "
            f"extra={sorted(actual_directories - expected_directories)}, "
            f"missing={sorted(expected_directories - actual_directories)}."
        )
    before = sidecar.get("source_snapshot_before_copy")
    after = sidecar.get("source_snapshot_after_copy")
    if not isinstance(before, Mapping) or before != after:
        raise TrainingSnapshotRecoveryError(
            "La source a change pendant la recuperation selon la filiation."
        )
    return sidecar


def _publish_recovered_source(
    *,
    source: Path,
    destination: Path,
    config: ExogenousFineTuneConfig,
    config_path: Path,
    config_sha256: str,
    source_manifest: Mapping[str, Any],
    source_snapshot: object,
    included: Sequence[Mapping[str, object]],
    excluded: Sequence[Mapping[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.recover-{uuid4().hex}"
    published = False
    try:
        staging.mkdir()
        if _sha256_file(config_path) != config_sha256:
            raise TrainingSnapshotRecoveryError(
                "La configuration de reference a change avant sa copie."
            )
        _copy_included_training_entries(source, staging)
        shutil.copy2(config_path, staging / RECOVERY_CONFIG_NAME)
        if (
            _sha256_file(config_path) != config_sha256
            or _sha256_file(staging / RECOVERY_CONFIG_NAME) != config_sha256
        ):
            raise TrainingSnapshotRecoveryError(
                "La configuration de reference a change pendant sa copie."
            )
        try:
            if os.path.samefile(config_path, staging / RECOVERY_CONFIG_NAME):
                raise TrainingSnapshotRecoveryError(
                    "Hardlink avec la configuration source interdit."
                )
        except OSError as exc:
            raise TrainingSnapshotRecoveryError(
                "Independance physique de la configuration invérifiable."
            ) from exc
        after_copy = _tree_snapshot(source)
        if after_copy != source_snapshot:
            raise TrainingSnapshotRecoveryError(
                "La source evaluee a change pendant la copie; publication refusee."
            )

        core_manifest = dict(source_manifest)
        removed_fields: list[dict[str, str]] = []
        for key in sorted(_MANIFEST_EVALUATION_KEYS):
            if key in core_manifest:
                removed_fields.append(
                    {"key": key, "value_sha256": _json_digest(core_manifest[key])}
                )
                core_manifest.pop(key)
        contract = _immutable_training_contract(core_manifest)
        resolved_config = config_summary(config)
        sidecar = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "kind": "chronos2_exogenous_training_snapshot_recovery_lineage",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifact_path": str(source),
            "source_snapshot_before_copy": {
                "tree_sha256": source_snapshot.sha256,
                "entries": source_snapshot.entries,
                "experiment_manifest_sha256": _sha256_file(
                    source / "experiment_manifest.json"
                ),
            },
            "source_snapshot_after_copy": {
                "tree_sha256": after_copy.sha256,
                "entries": after_copy.entries,
                "experiment_manifest_sha256": _sha256_file(
                    source / "experiment_manifest.json"
                ),
            },
            "manifest_transformation": {
                "mode": "remove_authenticated_raw_evaluation_fields_only",
                "removed_fields": removed_fields,
                "added_to_recovered_manifest": [RECOVERY_REFERENCE_KEY],
            },
            "reconstructed_training_manifest_core_sha256": _json_digest(core_manifest),
            "immutable_training_contract": contract,
            "immutable_training_contract_sha256": _json_digest(contract),
            "included_training_files": list(included),
            "excluded_derived_files": list(excluded),
            "excluded_derived_roots": sorted(
                name for name in _RAW_EVALUATION_ROOTS if (source / name).exists()
            ),
            "configuration_reference": {
                "source_path": str(config_path),
                "snapshot_relative_path": RECOVERY_CONFIG_NAME,
                "sha256": config_sha256,
                "provenance_class": "post_hoc_semantically_verified_reference",
                "claimed_as_original_training_bytes": False,
                "resolved_semantics": resolved_config,
                "resolved_semantics_sha256": _json_digest(resolved_config),
            },
            "source_was_modified": False,
            "evaluation_outputs_copied": False,
            "hardlinks_with_source_allowed": False,
        }
        _write_json_atomic(staging / RECOVERY_MANIFEST_NAME, sidecar)
        recovered_manifest = dict(core_manifest)
        recovered_manifest[RECOVERY_REFERENCE_KEY] = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "kind": "chronos2_exogenous_training_snapshot_recovery",
            "relative_path": RECOVERY_MANIFEST_NAME,
            "sha256": _sha256_file(staging / RECOVERY_MANIFEST_NAME),
        }
        _write_json_atomic(staging / "experiment_manifest.json", recovered_manifest)
        verify_recovered_training_snapshot(staging)
        _assert_independent_files(source, staging, included)
        if destination.exists():
            raise TrainingSnapshotRecoveryError(
                f"Destination apparue pendant la recuperation: {destination}."
            )
        os.replace(staging, destination)
        published = True
        verify_recovered_training_snapshot(destination)
        _assert_independent_files(source, destination, included)
        try:
            if os.path.samefile(
                config_path, destination / RECOVERY_CONFIG_NAME
            ):
                raise TrainingSnapshotRecoveryError(
                    "Hardlink publie avec la configuration source interdit."
                )
        except OSError as exc:
            raise TrainingSnapshotRecoveryError(
                "Independance physique publiee de la configuration invérifiable."
            ) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if published and destination.exists():
            shutil.rmtree(destination)
        raise


def recover_training_snapshot_and_prepare_zones(
    source_run_directory: str | Path,
    *,
    config_reference: str | Path,
    zones: Sequence[str],
    output_root: str | Path | None = None,
) -> RecoveredTrainingSnapshot:
    """Recover one training snapshot and transactionally prepare zone clones."""

    source = _resolve_without_links(
        source_run_directory, label="Artefact source"
    )
    if not source.is_dir() or _is_link_or_reparse(source):
        raise TrainingSnapshotRecoveryError(
            f"Artefact source absent, lien ou reparse interdit: {source}."
        )
    config_path = _resolve_without_links(
        config_reference, label="Configuration de reference"
    )
    if not config_path.is_file() or _is_link_or_reparse(config_path):
        raise TrainingSnapshotRecoveryError(
            f"Configuration de reference absente ou non reguliere: {config_path}."
        )
    config_sha256 = _sha256_file(config_path)
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (source.parent / "recovered_training_snapshot").resolve()
    )
    destination = root / "training_snapshot" / "artifact"
    if destination == source or source in destination.parents:
        raise TrainingSnapshotRecoveryError(
            "La destination de recuperation ne peut pas etre dans la source."
        )

    source_snapshot = _tree_snapshot(source)
    source_manifest = verify_bundle(source)
    if RECOVERY_REFERENCE_KEY in source_manifest:
        raise TrainingSnapshotRecoveryError(
            "Une recuperation ne peut pas servir de source a une nouvelle recuperation."
        )
    _validate_raw_evaluation(source, source_manifest)
    config = load_config(config_path)
    _validate_config_reference(config, source=source, manifest=source_manifest)
    if _sha256_file(config_path) != config_sha256:
        raise TrainingSnapshotRecoveryError(
            "La configuration de reference a change pendant sa validation."
        )
    included, excluded = _classify_source_files(source)
    _validate_retained_model_selection(source, config)
    if _tree_snapshot(source) != source_snapshot:
        raise TrainingSnapshotRecoveryError(
            "La source evaluee a change pendant sa validation."
        )

    created = False
    if destination.exists():
        sidecar = verify_recovered_training_snapshot(destination)
        before = sidecar["source_snapshot_before_copy"]
        config_info = sidecar["configuration_reference"]
        if (
            before.get("tree_sha256") != source_snapshot.sha256
            or before.get("entries") != source_snapshot.entries
            or sidecar.get("source_artifact_path") != str(source)
            or config_info.get("source_path") != str(config_path)
            or config_info.get("sha256") != config_sha256
        ):
            raise TrainingSnapshotRecoveryError(
                "Snapshot recupere existant lie a une autre source/configuration; "
                "overwrite interdit."
            )
    else:
        _publish_recovered_source(
            source=source,
            destination=destination,
            config=config,
            config_path=config_path,
            source_manifest=source_manifest,
            source_snapshot=source_snapshot,
            included=included,
            excluded=excluded,
            config_sha256=config_sha256,
        )
        created = True

    prepared = None
    try:
        prepared = prepare_zone_artifacts(
            destination,
            zones=zones,
            output_root=root,
        )
        if _tree_snapshot(source) != source_snapshot:
            raise TrainingSnapshotRecoveryError(
                "La source evaluee a change avant la fin; copies nouvelles retirees."
            )
        if _sha256_file(config_path) != config_sha256:
            raise TrainingSnapshotRecoveryError(
                "La configuration de reference a change avant la fin; "
                "copies nouvelles retirees."
            )
    except Exception as exc:
        if prepared is not None:
            for artifact in prepared.artifacts:
                if artifact.created and artifact.path.exists():
                    shutil.rmtree(artifact.path)
        if created and destination.exists():
            shutil.rmtree(destination)
        if isinstance(exc, (TrainingSnapshotRecoveryError, ZoneArtifactPreparationError)):
            raise
        raise TrainingSnapshotRecoveryError(
            f"Preparation des zones depuis le snapshot recupere echouee: {exc}"
        ) from exc

    return RecoveredTrainingSnapshot(
        source_directory=source,
        snapshot_directory=destination,
        output_root=root,
        recovery_manifest_path=destination / RECOVERY_MANIFEST_NAME,
        source_tree_sha256=source_snapshot.sha256,
        created=created,
        artifacts=prepared.artifacts,
    )


__all__ = [
    "RECOVERY_CONFIG_NAME",
    "RECOVERY_MANIFEST_NAME",
    "RECOVERY_REFERENCE_KEY",
    "RECOVERY_SCHEMA_VERSION",
    "RecoveredTrainingSnapshot",
    "TrainingSnapshotRecoveryError",
    "recover_training_snapshot_and_prepare_zones",
    "verify_recovered_training_snapshot",
    "verify_recovery_reference",
]
