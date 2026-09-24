"""Research-only residual calibration on the frozen 30-day validation split.

This module is deliberately separated from the production OOF residual path.
The fit phase returns and uses only the validation slice of the training panel,
then seals the corrector before opening the canonical rolling-365 evaluation
artifacts.  A monolithic Parquet row group can physically overlap validation
and holdout; that possible engine-level decode scope is audited and never
misrepresented as row-level isolation.  Artifacts use distinct names and an
incompatible contract so they cannot be consumed by FinalBacktest, Shadow,
Govern or Promote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import pandas as pd

from .evaluation import (
    BASELINE_LABEL,
    CANDIDATE_LABEL,
    _input_sha256,
    _predict_batches_with_cache,
    build_inference_input,
    compute_metrics,
)
from .lora_finetune import (
    EVALUATION_COLUMNS,
    ExogenousFineTuneConfig,
    load_checkpoint,
    load_config,
    verify_bundle,
)


FIT_KIND = "chronos2_exogenous_research_validation_corrector_fit"
EVALUATION_KIND = "chronos2_exogenous_research_validation_corrector_evaluation"
CORRECTOR_KIND = "research_validation_corrector"
FIT_PROTOCOL = "post_training_validation30_in_sample_not_oof"
CORRECTOR_NAME = "research_validation_corrector.sealed.json"
FIT_MANIFEST_NAME = "research_validation_fit_manifest.json"
VALIDATION_PREDICTIONS_NAME = "research_validation_predictions.csv.gz"
EVALUATION_DIRECTORY_NAME = "holdout_evaluation_research_only"
EVALUATION_MANIFEST_NAME = "research_validation_evaluation_manifest.json"
EVALUATION_PREDICTIONS_NAME = "research_holdout_predictions.csv.gz"
EVALUATION_METRICS_NAME = "research_holdout_metrics.json"
EVALUATION_DAILY_NAME = "research_holdout_daily.csv.gz"
EVALUATION_REPORT_NAME = "research_holdout_report.html"
VALIDATION_COLUMNS = (
    "delivery_start_utc",
    "forecast_origin_utc",
    "actual",
    "candidate_q10",
    "candidate_q50",
    "candidate_q90",
)
RAW_EVALUATION_ARTIFACTS = {
    "evidence": "evaluation_predictions.csv.gz",
    "daily": "evaluation_daily.csv.gz",
    "metrics": "evaluation_metrics.json",
    "report": "evaluation_report.html",
}
FIT_ARTIFACT_NAMES = frozenset(
    {CORRECTOR_NAME, FIT_MANIFEST_NAME, VALIDATION_PREDICTIONS_NAME}
)
EVALUATION_ARTIFACT_NAMES = frozenset(
    {
        EVALUATION_MANIFEST_NAME,
        EVALUATION_PREDICTIONS_NAME,
        EVALUATION_METRICS_NAME,
        EVALUATION_DAILY_NAME,
        EVALUATION_REPORT_NAME,
    }
)
FORBIDDEN_RESEARCH_ARTIFACT_NAMES = frozenset(
    {
        "residual_corrector.json",
        "oof_predictions_365.csv.gz",
        "oof_predictions_365.csv.gz.audit.json",
        "evaluation_manifest.json",
        "final_pipeline_manifest.json",
        "shadow_manifest.json",
        "shadow_final_manifest.json",
        "bundle_manifest.json",
    }
)

# Fixed before any research fit.  There is intentionally no CLI grid or
# hyperparameter override: selecting these values on the holdout would create
# another layer of leakage.
FEATURE_COLUMNS = (
    "intercept",
    "local_hour_sin",
    "local_hour_cos",
    "local_dow_sin",
    "local_dow_cos",
)
RIDGE_ALPHA = 1.0
MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH = 20.0
EXPECTED_VALIDATION_DAYS = 30
EXPECTED_HOLDOUT_DAYS = 365
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ResearchValidationCorrectorError(RuntimeError):
    """Raised when the research-only causal boundary cannot be proved."""


@dataclass(frozen=True)
class ResearchFitArtifacts:
    output_directory: Path
    corrector_path: Path
    predictions_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ResearchEvaluationArtifacts:
    output_directory: Path
    predictions_path: Path
    metrics_path: Path
    daily_path: Path
    report_path: Path
    manifest_path: Path
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class _FitSeal:
    manifest: Mapping[str, Any]
    corrector: Mapping[str, Any]
    manifest_sha256: str
    corrector_sha256: str
    validation_predictions_sha256: str


@dataclass(frozen=True)
class _RawHoldout:
    frame: pd.DataFrame
    manifest: Mapping[str, Any]
    manifest_sha256: str
    evidence_sha256: str
    artifact_hashes: Mapping[str, str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _resolve_without_links(
    value: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve a path only after rejecting symlink/junction components."""

    absolute = Path(value).expanduser().absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise ResearchValidationCorrectorError(
                f"{label}: lien symbolique/junction/reparse interdit: {component}."
            )
    return absolute.resolve()


def _regular_private_file(path: Path, *, label: str) -> None:
    if not path.is_file() or _is_link_or_reparse(path):
        raise ResearchValidationCorrectorError(
            f"{label} absent, non regulier ou lien/reparse: {path}."
        )
    try:
        links = int(path.stat(follow_symlinks=False).st_nlink)
    except (AttributeError, OSError):
        links = 1
    if links > 1:
        raise ResearchValidationCorrectorError(
            f"{label}: hardlink interdit pour un artefact research scelle: {path}."
        )


def _read_bytes_stable(
    path: Path,
    *,
    label: str,
    private: bool = False,
) -> bytes:
    if private:
        _regular_private_file(path, label=label)
    elif not path.is_file() or _is_link_or_reparse(path):
        raise ResearchValidationCorrectorError(
            f"{label} absent, non regulier ou lien/reparse: {path}."
        )
    before = path.stat(follow_symlinks=False)
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    fingerprint_before = (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ctime_ns", None),
        getattr(before, "st_ino", None),
        getattr(before, "st_dev", None),
    )
    fingerprint_after = (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ctime_ns", None),
        getattr(after, "st_ino", None),
        getattr(after, "st_dev", None),
    )
    if fingerprint_before != fingerprint_after or len(payload) != before.st_size:
        raise ResearchValidationCorrectorError(
            f"{label} a change pendant sa lecture."
        )
    return payload


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ResearchValidationCorrectorError(
                    f"{label}: cle JSON dupliquee {key!r}."
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ResearchValidationCorrectorError(
            f"{label}: constante JSON non finie interdite ({value})."
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except ResearchValidationCorrectorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchValidationCorrectorError(f"{label} illisible.") from exc
    if not isinstance(value, dict):
        raise ResearchValidationCorrectorError(f"{label} doit etre un objet JSON.")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _same_datetime_index(
    left: pd.DatetimeIndex,
    right: pd.DatetimeIndex,
) -> bool:
    """Compare instants independently of Parquet/CSV timestamp resolution."""

    try:
        return left.as_unit("ns").equals(right.as_unit("ns"))
    except (AttributeError, ValueError, OverflowError):
        return np.array_equal(
            left.to_numpy(dtype="datetime64[ns]"),
            right.to_numpy(dtype="datetime64[ns]"),
        )


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _strict_json_bytes(
        _read_bytes_stable(path, label=label),
        label=f"{label} ({path})",
    )


def _read_json_snapshot(
    path: Path,
    *,
    label: str,
    private: bool = False,
) -> tuple[dict[str, Any], str]:
    payload = _read_bytes_stable(path, label=label, private=private)
    return (
        _strict_json_bytes(payload, label=f"{label} ({path})"),
        hashlib.sha256(payload).hexdigest(),
    )


def _safe_artifact(root: Path, relative: object, *, expected_name: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ResearchValidationCorrectorError("Chemin relatif d'artefact absent.")
    value = Path(relative)
    if value.is_absolute() or value != Path(expected_name):
        raise ResearchValidationCorrectorError(
            f"Chemin d'artefact research invalide: {relative!r}."
        )
    resolved = (root / value).resolve()
    if root.resolve() not in resolved.parents:
        raise ResearchValidationCorrectorError("Artefact research hors repertoire.")
    return resolved


def _schema_contract(config: ExogenousFineTuneConfig) -> dict[str, Any]:
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


def _config_semantic_contract(config: ExogenousFineTuneConfig) -> dict[str, Any]:
    return {
        "schema": _schema_contract(config),
        "experiment_id": config.experiment_id,
        "evaluation_role": config.evaluation_role,
        "training_window_days": config.training_window_days,
        "validation_days": config.validation_days,
        "evaluation_days": config.evaluation_days,
        "require_consecutive_origins": config.require_consecutive_origins,
        "require_complete_known_future": config.require_complete_known_future,
        "production_pit_evidence": config.production_pit_evidence,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "finetune_mode": "lora",
        "learning_rate": config.learning_rate,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "lora_config": dict(config.lora_config),
    }


def _verify_config_bundle_contract(
    config: ExogenousFineTuneConfig,
    run_directory: Path,
    manifest: Mapping[str, Any],
    *,
    item_id: str,
) -> str:
    """Refuse a post-hoc config that changes the sealed inference semantics."""

    schema = _read_json(run_directory / "schema.json", label="schema LoRA")
    expected_schema = _schema_contract(config)
    if schema != expected_schema:
        raise ResearchValidationCorrectorError(
            "La configuration courante diverge du schema LoRA scelle."
        )
    expected_manifest: Mapping[str, object] = {
        "format_version": 1,
        "experiment_id": config.experiment_id,
        "evaluation_role": config.evaluation_role,
        "training_window_days": config.training_window_days,
        "evaluation_days": config.evaluation_days,
        "cutoff_local_time": config.cutoff_local_time,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "finetune_mode": "lora",
        "production_pit_evidence": config.production_pit_evidence,
        "production_pipeline_evidence": False,
    }
    failures = [
        key
        for key, expected in expected_manifest.items()
        if type(manifest.get(key)) is not type(expected)
        or manifest.get(key) != expected
    ]
    if failures:
        raise ResearchValidationCorrectorError(
            "La configuration courante diverge du manifeste LoRA scelle: "
            + ", ".join(failures)
            + "."
        )
    declared_zone = manifest.get("zone")
    if declared_zone is not None and str(declared_zone).strip().upper() != str(
        item_id
    ).strip().upper():
        raise ResearchValidationCorrectorError(
            "La zone demandee diverge de la projection PrepareZones."
        )
    training = manifest.get("training")
    expected_training: Mapping[str, object] = {
        "learning_rate": config.learning_rate,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "lora_config": dict(config.lora_config),
    }
    if not isinstance(training, Mapping) or any(
        type(training.get(key)) is not type(expected)
        or training.get(key) != expected
        for key, expected in expected_training.items()
    ):
        raise ResearchValidationCorrectorError(
            "Les hyperparametres de la configuration divergent du training scelle."
        )
    payload = json.dumps(
        _config_semantic_contract(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_outside_candidate_bundle(
    destination: Path,
    *,
    run_directory: Path,
    label: str,
) -> None:
    if (
        destination == run_directory
        or run_directory in destination.parents
        or destination in run_directory.parents
    ):
        raise ResearchValidationCorrectorError(
            f"{label} doit rester strictement hors du bundle candidat."
        )
    for parent in (destination, *destination.parents):
        if (parent / "experiment_manifest.json").is_file() and (
            parent / "checkpoint"
        ).is_dir():
            raise ResearchValidationCorrectorError(
                f"{label} ne peut pas etre imbriquee dans un autre bundle LoRA."
            )


def _assert_research_closure(root: Path) -> None:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ResearchValidationCorrectorError(
            f"Repertoire research absent, non regulier ou lien/reparse: {root}."
        )
    names: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in [*directory_names, *file_names]:
            child = parent / name
            if _is_link_or_reparse(child):
                raise ResearchValidationCorrectorError(
                    f"Lien/junction/reparse interdit dans l'artefact research: {child}."
                )
            if name in FORBIDDEN_RESEARCH_ARTIFACT_NAMES:
                raise ResearchValidationCorrectorError(
                    f"Artefact au nom production interdit dans la sortie research: {name}."
                )
        if parent == root:
            names.update(directory_names)
            names.update(file_names)
    allowed = set(FIT_ARTIFACT_NAMES) | {EVALUATION_DIRECTORY_NAME}
    if not set(FIT_ARTIFACT_NAMES).issubset(names) or not names.issubset(allowed):
        raise ResearchValidationCorrectorError(
            "Closure du repertoire research invalide: "
            f"attendu={sorted(FIT_ARTIFACT_NAMES)}, observe={sorted(names)}."
        )


def _assert_evaluation_closure(root: Path) -> None:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ResearchValidationCorrectorError(
            f"Repertoire d'evaluation research invalide: {root}."
        )
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != set(EVALUATION_ARTIFACT_NAMES):
        raise ResearchValidationCorrectorError(
            "Closure de l'evaluation research invalide."
        )
    for entry in entries:
        _regular_private_file(entry, label=f"artefact evaluation {entry.name}")


def _hyperparameter_contract() -> dict[str, Any]:
    core = {
        "model_family": "ridge_linear_common_quantile_shift_v1",
        "feature_columns": list(FEATURE_COLUMNS),
        "ridge_alpha": RIDGE_ALPHA,
        "maximum_absolute_shift_eur_mwh": MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**core, "contract_sha256": digest}


def default_research_directory(
    config: ExogenousFineTuneConfig,
    manifest: Mapping[str, Any],
    *,
    item_id: str,
) -> Path:
    checkpoint = manifest.get("checkpoint_sha256")
    if not _is_sha256(checkpoint):
        raise ResearchValidationCorrectorError("SHA du checkpoint candidat invalide.")
    experiment = str(manifest.get("experiment_id", "candidate")).strip()
    safe_experiment = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment) or "candidate"
    return _resolve_without_links(
        (
        config.project_root
        / "runs"
        / "experiments"
        / "chronos2_exogenous"
        / "research_validation_correctors"
        / f"{safe_experiment}_{checkpoint[:12]}"
        / str(item_id).strip().lower()
        ),
        label="sortie research par defaut",
    )


def _source_identity(
    config: ExogenousFineTuneConfig,
    run_directory: Path,
    manifest: Mapping[str, Any],
    *,
    item_id: str,
) -> dict[str, str]:
    current_manifest, current_manifest_sha = _read_json_snapshot(
        run_directory / "experiment_manifest.json",
        label="manifeste d'experience LoRA",
    )
    if current_manifest != dict(manifest):
        raise ResearchValidationCorrectorError(
            "Le manifeste LoRA a change depuis la verification du bundle."
        )
    config_semantic_sha = _verify_config_bundle_contract(
        config, run_directory, manifest, item_id=item_id
    )
    expected: dict[str, tuple[Path, object]] = {
        "panel_sha256": (config.panel_path, manifest.get("panel_sha256")),
        "panel_audit_sha256": (
            config.panel_audit_path,
            manifest.get("panel_audit_sha256"),
        ),
        "schema_sha256": (run_directory / "schema.json", manifest.get("schema_sha256")),
    }
    result: dict[str, str] = {}
    for name, (path, declared) in expected.items():
        safe_path = _resolve_without_links(path, label=f"source {name}")
        if not _is_sha256(declared) or not safe_path.is_file():
            raise ResearchValidationCorrectorError(
                f"Identite source absente/invalide: {name}."
            )
        actual = _sha256(safe_path)
        if actual != declared:
            raise ResearchValidationCorrectorError(
                f"Identite source divergente: {name}."
            )
        result[name] = actual
    checkpoint = manifest.get("checkpoint_sha256")
    if not _is_sha256(checkpoint):
        raise ResearchValidationCorrectorError("checkpoint_sha256 absent/invalide.")
    result["checkpoint_sha256"] = str(checkpoint)
    # Backtest publication is allowed to append only its evidence reference
    # after the research fit.  Bind the immutable candidate contract
    # separately so fit-before-backtest remains possible without accepting a
    # checkpoint, recipe, source or split change.
    immutable_manifest = dict(manifest)
    immutable_manifest.pop("evaluation_evidence", None)
    immutable_manifest.pop("evaluation_label_resolution", None)
    result["candidate_contract_sha256"] = hashlib.sha256(
        json.dumps(
            immutable_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    evaluation_reference = manifest.get("evaluation_evidence")
    if evaluation_reference is None:
        evaluation_sha_at_observation = "absent"
    elif isinstance(evaluation_reference, Mapping) and _is_sha256(
        evaluation_reference.get("sha256")
    ):
        evaluation_sha_at_observation = str(evaluation_reference["sha256"])
    else:
        raise ResearchValidationCorrectorError(
            "evaluation_evidence presente mais son SHA est invalide."
        )
    result.update(
        {
            "config_semantic_sha256": config_semantic_sha,
            "experiment_manifest_sha256_at_observation": current_manifest_sha,
            "evaluation_evidence_sha256_at_observation": (
                evaluation_sha_at_observation
            ),
            "implementation_research_corrector_sha256": _sha256(Path(__file__)),
            "implementation_evaluation_sha256": _sha256(
                Path(__file__).with_name("evaluation.py")
            ),
            "implementation_lora_finetune_sha256": _sha256(
                Path(__file__).with_name("lora_finetune.py")
            ),
        }
    )
    return result


def _declared_split(
    manifest: Mapping[str, Any],
    name: str,
    *,
    expected_count: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    splits = manifest.get("splits")
    value = splits.get(name) if isinstance(splits, Mapping) else None
    if not isinstance(value, Mapping) or value.get("count") != expected_count:
        raise ResearchValidationCorrectorError(
            f"Split {name}: exactement {expected_count} origines sont requises."
        )
    first = pd.to_datetime(value.get("first_utc"), utc=True, errors="coerce")
    last = pd.to_datetime(value.get("last_utc"), utc=True, errors="coerce")
    if pd.isna(first) or pd.isna(last) or pd.Timestamp(first) > pd.Timestamp(last):
        raise ResearchValidationCorrectorError(f"Split {name}: bornes invalides.")
    return pd.Timestamp(first), pd.Timestamp(last)


def _expected_origins(
    first: pd.Timestamp,
    last: pd.Timestamp,
    *,
    count: int,
    config: ExogenousFineTuneConfig,
) -> pd.DatetimeIndex:
    hour, minute = (int(part) for part in config.cutoff_local_time.split(":"))
    first_local = first.tz_convert(config.timezone)
    last_local = last.tz_convert(config.timezone)
    local_days = pd.date_range(
        first_local.normalize(), last_local.normalize(), freq="D"
    )
    values = pd.DatetimeIndex(
        [
            pd.Timestamp(
                f"{day:%Y-%m-%d} {hour:02d}:{minute:02d}",
                tz=config.timezone,
            )
            for day in local_days
        ]
    ).tz_convert("UTC")
    if len(values) != count or values[0] != first or values[-1] != last:
        raise ResearchValidationCorrectorError(
            "Split validation non consecutif ou cutoff local incoherent."
        )
    return values


def _parquet_filter_audit(
    parquet_file: Any,
    *,
    origin_column: str,
    validation_first: pd.Timestamp,
    validation_last: pd.Timestamp,
    holdout_first: pd.Timestamp,
) -> dict[str, Any]:
    """Describe the physical row-group scope without claiming page isolation."""

    schema = parquet_file.schema_arrow
    origin_index = schema.get_field_index(origin_column)
    if origin_index < 0:
        raise ResearchValidationCorrectorError(
            "La colonne d'origine est absente du schema Parquet."
        )
    selected: list[dict[str, Any]] = []
    all_selected_isolated = True
    possible_holdout_decode = False
    for index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(index)
        statistics = row_group.column(origin_index).statistics
        if (
            statistics is None
            or not statistics.has_min_max
            or statistics.min is None
            or statistics.max is None
        ):
            minimum = None
            maximum = None
            intersects = True
            isolated = False
            may_touch_holdout = True
        else:
            minimum = pd.Timestamp(statistics.min)
            maximum = pd.Timestamp(statistics.max)
            if minimum.tzinfo is None:
                minimum = minimum.tz_localize("UTC")
            else:
                minimum = minimum.tz_convert("UTC")
            if maximum.tzinfo is None:
                maximum = maximum.tz_localize("UTC")
            else:
                maximum = maximum.tz_convert("UTC")
            intersects = maximum >= validation_first and minimum <= validation_last
            isolated = minimum >= validation_first and maximum <= validation_last
            may_touch_holdout = maximum >= holdout_first
        if not intersects:
            continue
        selected.append(
            {
                "row_group": index,
                "rows": int(row_group.num_rows),
                "minimum_origin_utc": minimum.isoformat() if minimum is not None else None,
                "maximum_origin_utc": maximum.isoformat() if maximum is not None else None,
                "wholly_within_validation": isolated,
                "may_include_holdout_rows": may_touch_holdout,
            }
        )
        all_selected_isolated = all_selected_isolated and isolated
        possible_holdout_decode = possible_holdout_decode or may_touch_holdout
    if not selected:
        raise ResearchValidationCorrectorError(
            "Aucun row group Parquet ne peut couvrir la validation declaree."
        )
    return {
        "schema_version": 1,
        "reader": "pandas.read_parquet_pyarrow_filtered",
        "logical_filter_on_origin_and_item": True,
        "validation_first_origin_utc": validation_first.isoformat(),
        "validation_last_origin_utc": validation_last.isoformat(),
        "holdout_first_origin_utc": holdout_first.isoformat(),
        "selected_row_groups": selected,
        "selected_row_groups_wholly_within_validation": all_selected_isolated,
        "parquet_engine_may_decode_nonvalidation_rows": not all_selected_isolated,
        "parquet_engine_may_decode_holdout_rows": possible_holdout_decode,
        "parquet_page_level_decode_scope_attested": False,
        "returned_origins_exactly_validation": True,
        "returned_holdout_rows": False,
        "holdout_labels_used_for_fit": False,
    }


def _read_validation_slice(
    config: ExogenousFineTuneConfig,
    manifest: Mapping[str, Any],
    *,
    item_id: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    """Return only validation rows and audit possible physical row-group overlap."""

    if config.validation_days != EXPECTED_VALIDATION_DAYS:
        raise ResearchValidationCorrectorError(
            "Ce protocole v1 exige exactement 30 jours de validation."
        )
    if config.evaluation_days != EXPECTED_HOLDOUT_DAYS:
        raise ResearchValidationCorrectorError(
            "Ce protocole v1 exige un holdout gele de 365 jours."
        )
    validation_first, validation_last = _declared_split(
        manifest, "validation", expected_count=EXPECTED_VALIDATION_DAYS
    )
    holdout_first, _ = _declared_split(
        manifest, "evaluation_holdout", expected_count=EXPECTED_HOLDOUT_DAYS
    )
    if validation_last >= holdout_first:
        raise ResearchValidationCorrectorError(
            "La validation chevauche le holdout; calibration refusee."
        )
    if (
        validation_last.tz_convert(config.timezone).date()
        + pd.Timedelta(days=1)
        != holdout_first.tz_convert(config.timezone).date()
    ):
        raise ResearchValidationCorrectorError(
            "La validation ne finit pas exactement la veille du holdout."
        )
    expected_origins = _expected_origins(
        validation_first,
        validation_last,
        count=EXPECTED_VALIDATION_DAYS,
        config=config,
    )
    columns = list(
        dict.fromkeys(
            (
                config.timestamp_column,
                config.origin_column,
                config.item_column,
                config.feature_available_at_column,
                *config.target_columns,
                *config.covariate_columns,
            )
        )
    )
    if "".join(config.panel_path.suffixes).lower() != ".parquet":
        raise ResearchValidationCorrectorError(
            "Le fit sans lecture du holdout exige un panel Parquet filtrable."
        )
    try:
        # Arrow requires the predicate timezone to match the physical Parquet
        # field timezone.  Inspecting the schema decodes no data/holdout label.
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(config.panel_path)
        origin_type = parquet_file.schema_arrow.field(config.origin_column).type
        if not pa.types.is_timestamp(origin_type):
            raise ResearchValidationCorrectorError(
                "La colonne d'origine Parquet n'est pas un timestamp."
            )
        storage_timezone = origin_type.tz
        if storage_timezone:
            filter_first = validation_first.tz_convert(storage_timezone)
            filter_last = validation_last.tz_convert(storage_timezone)
        else:
            filter_first = validation_first.tz_convert("UTC").tz_localize(None)
            filter_last = validation_last.tz_convert("UTC").tz_localize(None)
        filter_audit = _parquet_filter_audit(
            parquet_file,
            origin_column=config.origin_column,
            validation_first=validation_first,
            validation_last=validation_last,
            holdout_first=holdout_first,
        )
        frame = pd.read_parquet(
            config.panel_path,
            columns=columns,
            filters=[
                (config.origin_column, ">=", filter_first),
                (config.origin_column, "<=", filter_last),
                (config.item_column, "==", str(item_id)),
            ],
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Lecture filtree de la seule validation impossible."
        ) from exc
    if frame.empty:
        raise ResearchValidationCorrectorError(
            f"Aucune ligne de validation pour {item_id}."
        )
    for column in (
        config.timestamp_column,
        config.origin_column,
        config.feature_available_at_column,
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise ResearchValidationCorrectorError(
                f"Timestamp validation invalide: {column}."
            )
    frame[config.item_column] = frame[config.item_column].astype(str)
    if set(frame[config.item_column]) != {str(item_id)}:
        raise ResearchValidationCorrectorError("Le filtre item validation a derive.")
    present = pd.DatetimeIndex(
        frame[config.origin_column].drop_duplicates()
    ).sort_values()
    if not _same_datetime_index(present, expected_origins):
        raise ResearchValidationCorrectorError(
            "Le panel ne couvre pas exactement les 30 origines validation."
        )
    if bool(
        (frame[config.feature_available_at_column] > frame[config.origin_column]).any()
    ):
        raise ResearchValidationCorrectorError(
            "Fuite PIT dans la tranche validation."
        )
    for column in (*config.target_columns, *config.covariate_columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for origin in expected_origins:
        group = frame.loc[frame[config.origin_column].eq(origin)]
        _payload, horizon, _actual = build_inference_input(group, config)
        ordered = group.sort_values(config.timestamp_column, kind="stable")
        timestamps = pd.DatetimeIndex(ordered[config.timestamp_column])
        if timestamps.duplicated().any():
            raise ResearchValidationCorrectorError(
                f"Validation {origin.isoformat()}: timestamp duplique."
            )
        expected_horizon = pd.date_range(
            pd.Timestamp(horizon[0]).tz_convert(config.timezone).normalize().tz_convert("UTC"),
            (
                pd.Timestamp(horizon[0]).tz_convert(config.timezone).normalize()
                + pd.DateOffset(days=1)
            ).tz_convert("UTC"),
            inclusive="left",
            freq="h",
        )
        expected_context = pd.date_range(
            end=expected_horizon[0] - pd.Timedelta(hours=1),
            periods=config.context_length,
            freq="h",
        )
        if not _same_datetime_index(horizon, expected_horizon) or not _same_datetime_index(
            timestamps, expected_context.append(expected_horizon)
        ):
            raise ResearchValidationCorrectorError(
                f"Validation {origin.isoformat()}: timeline/DST invalide."
            )
    filter_audit["returned_rows"] = int(len(frame))
    filter_audit["returned_origin_count"] = int(len(present))
    filter_audit["returned_item_id"] = str(item_id)
    filter_audit["audit_sha256"] = hashlib.sha256(
        json.dumps(
            filter_audit,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return frame, expected_origins, filter_audit


def _predict_validation(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    config: ExogenousFineTuneConfig,
    *,
    candidate_pipeline: Any,
    item_id: str,
    target_column: str,
    checkpoint_sha256: str,
    batch_size: int,
    inference_chunk_size: int,
    cache_directory: Path,
    progress: Callable[[int, int, pd.Timestamp], None] | None,
) -> tuple[pd.DataFrame, str]:
    if target_column not in config.target_columns:
        raise ResearchValidationCorrectorError(
            f"Target de validation inconnue: {target_column}."
        )
    target_index = config.target_columns.index(target_column)
    by_horizon: dict[int, list[dict[str, Any]]] = {23: [], 24: [], 25: []}
    input_hashes: dict[pd.Timestamp, str] = {}
    for origin in origins:
        group = frame.loc[frame[config.origin_column].eq(origin)]
        payload, horizon, actual = build_inference_input(group, config)
        digest = _input_sha256(payload)
        by_horizon[len(horizon)].append(
            {
                "origin": pd.Timestamp(origin),
                "payload": payload,
                "horizon": horizon,
                "actual": actual,
                "input_sha256": digest,
            }
        )
        input_hashes[pd.Timestamp(origin)] = digest
    predicted: dict[pd.Timestamp, np.ndarray] = {}
    for hours, samples in by_horizon.items():
        if not samples:
            continue
        values = _predict_batches_with_cache(
            candidate_pipeline,
            [sample["payload"] for sample in samples],
            prediction_length=hours,
            context_length=config.context_length,
            batch_size=batch_size,
            chunk_size=inference_chunk_size,
            cache_directory=cache_directory,
            cache_label="rv",
            model_identity=checkpoint_sha256,
        )
        for sample, output in zip(samples, values, strict=True):
            if _input_sha256(sample["payload"]) != sample["input_sha256"]:
                raise ResearchValidationCorrectorError(
                    "Le pipeline a mute une entree validation."
                )
            predicted[sample["origin"]] = output
    rows: list[pd.DataFrame] = []
    for number, origin in enumerate(origins, start=1):
        matches = [
            sample
            for samples in by_horizon.values()
            for sample in samples
            if sample["origin"] == pd.Timestamp(origin)
        ]
        if len(matches) != 1:
            raise ResearchValidationCorrectorError("Origine validation ambigue.")
        sample = matches[0]
        output = predicted[pd.Timestamp(origin)]
        if output.shape[0] != len(config.target_columns):
            raise ResearchValidationCorrectorError(
                "Nombre de targets predit incoherent."
            )
        rows.append(
            pd.DataFrame(
                {
                    "delivery_start_utc": sample["horizon"],
                    "forecast_origin_utc": pd.DatetimeIndex(
                        [origin] * len(sample["horizon"])
                    ),
                    "actual": sample["actual"][target_index],
                    "candidate_q10": output[target_index, :, 0],
                    "candidate_q50": output[target_index, :, 1],
                    "candidate_q90": output[target_index, :, 2],
                }
            )
        )
        if progress is not None:
            progress(number, len(origins), pd.Timestamp(origin))
    result = pd.concat(rows, ignore_index=True)
    digest = hashlib.sha256(
        "\n".join(input_hashes[pd.Timestamp(origin)] for origin in origins).encode(
            "ascii"
        )
    ).hexdigest()
    return result, digest


def _validation_input_contract(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    config: ExogenousFineTuneConfig,
) -> str:
    """Rebuild the ordered inference-input identity without model inference."""

    hashes: list[str] = []
    for origin in origins:
        group = frame.loc[frame[config.origin_column].eq(origin)]
        payload, _horizon, _actual = build_inference_input(group, config)
        hashes.append(_input_sha256(payload))
    return hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()


def _design(delivery: pd.Series | pd.DatetimeIndex, *, timezone_name: str) -> np.ndarray:
    timestamps = pd.DatetimeIndex(pd.to_datetime(delivery, utc=True, errors="raise"))
    local = timestamps.tz_convert(timezone_name)
    return np.column_stack(
        (
            np.ones(len(local)),
            np.sin(2 * np.pi * local.hour / 24.0),
            np.cos(2 * np.pi * local.hour / 24.0),
            np.sin(2 * np.pi * local.dayofweek / 7.0),
            np.cos(2 * np.pi * local.dayofweek / 7.0),
        )
    )


def _fit_corrector(
    predictions: pd.DataFrame,
    *,
    timezone_name: str,
    checkpoint_sha256: str,
    panel_sha256: str,
    panel_audit_sha256: str,
    predictions_sha256: str,
    input_contract_sha256: str,
    item_id: str,
    target_column: str,
    validation_first_origin: pd.Timestamp,
    validation_last_origin: pd.Timestamp,
    parquet_filter_audit: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = predictions[["actual", "candidate_q50"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ResearchValidationCorrectorError(
            "Actuals/predictions validation non finis."
        )
    design = _design(predictions["delivery_start_utc"], timezone_name=timezone_name)
    means = design.mean(axis=0)
    scales = design.std(axis=0)
    means[0], scales[0] = 0.0, 1.0
    scales[scales <= 1e-12] = 1.0
    normalised = (design - means) / scales
    residual = (
        numeric["actual"].to_numpy(float)
        - numeric["candidate_q50"].to_numpy(float)
    )
    penalty = np.eye(normalised.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.lstsq(
        normalised.T @ normalised + penalty,
        normalised.T @ residual,
        rcond=None,
    )[0]
    fitted = np.clip(
        normalised @ coefficients,
        -MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
        MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
    )
    return {
        "schema_version": 1,
        "kind": CORRECTOR_KIND,
        "model_kind": "linear_shift_v1_research_validation_only",
        "base_model": "chronos2_exogenous",
        "output_model": "chronos2_exogenous_research_validation_corrected",
        "fit_protocol": FIT_PROTOCOL,
        "calibration_days": EXPECTED_VALIDATION_DAYS,
        "calibration_start_origin_utc": validation_first_origin.isoformat(),
        "calibration_end_origin_utc": validation_last_origin.isoformat(),
        "validation_used_for_checkpoint_monitoring_and_selection": True,
        "calibration_predictions_are_not_oof": True,
        "candidate_checkpoint_frozen_before_holdout_evaluation": True,
        "holdout_used_for_fit": False,
        "holdout_logically_used_for_fit": False,
        "holdout_actuals_used_for_fit": False,
        "returned_holdout_rows": False,
        "parquet_engine_may_decode_holdout_rows": bool(
            parquet_filter_audit["parquet_engine_may_decode_holdout_rows"]
        ),
        "parquet_page_level_decode_scope_attested": False,
        "parquet_filter_audit_sha256": str(parquet_filter_audit["audit_sha256"]),
        "full_panel_bytes_hashed_for_identity": True,
        "future_actuals_used_as_features": False,
        "hyperparameters_fixed_without_holdout_grid": True,
        "historical_holdout_may_have_been_previously_inspected": True,
        "research_only": True,
        "candidate_selection_eligible": False,
        "final_backtest_eligible": False,
        "shadow_eligible": False,
        "governance_eligible": False,
        "promotion_eligible": False,
        "forbidden_consumers": ["FinalBacktest", "Shadow", "Govern", "Promote"],
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "panel_sha256": panel_sha256,
        "panel_audit_sha256": panel_audit_sha256,
        "validation_predictions_sha256": predictions_sha256,
        "input_contract_sha256": input_contract_sha256,
        "item_id": str(item_id),
        "target_column": str(target_column),
        "timezone": timezone_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": RIDGE_ALPHA,
        "maximum_absolute_shift_eur_mwh": MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
        "hyperparameter_contract_sha256": _hyperparameter_contract()[
            "contract_sha256"
        ],
        "fit_rows": int(len(predictions)),
        "fit_mae_before_eur_mwh": float(np.mean(np.abs(residual))),
        "fit_mae_after_eur_mwh": float(np.mean(np.abs(residual - fitted))),
        "sealed_at_utc": _now(),
    }


def _apply_corrector(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    delivery: pd.Series | pd.DatetimeIndex,
    corrector: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design = _design(delivery, timezone_name=str(corrector["timezone"]))
    means = np.asarray(corrector["feature_means"], dtype=float)
    scales = np.asarray(corrector["feature_scales"], dtype=float)
    coefficients = np.asarray(corrector["coefficients"], dtype=float)
    if (
        means.shape != (len(FEATURE_COLUMNS),)
        or scales.shape != means.shape
        or coefficients.shape != means.shape
        or not np.isfinite(np.concatenate((means, scales, coefficients))).all()
        or bool((scales <= 0).any())
    ):
        raise ResearchValidationCorrectorError(
            "Parametres du correcteur research invalides."
        )
    shift = ((design - means) / scales) @ coefficients
    shift = np.clip(
        shift,
        -float(corrector["maximum_absolute_shift_eur_mwh"]),
        float(corrector["maximum_absolute_shift_eur_mwh"]),
    )
    return q10 + shift, q50 + shift, q90 + shift, shift


def fit_research_validation_corrector(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    run_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    item_id: str,
    batch_size: int = 64,
    inference_chunk_size: int = 30,
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> ResearchFitArtifacts:
    """Fit and seal from validation only, without opening holdout evidence."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    run_dir = _resolve_without_links(
        run_directory or config.output_directory,
        label="bundle candidat",
    )
    manifest = verify_bundle(run_dir)
    if manifest.get("candidate_frozen_before_evaluation") is not True or manifest.get(
        "feature_selection_frozen_before_evaluation"
    ) is not True:
        raise ResearchValidationCorrectorError(
            "Le candidat et ses features doivent etre geles avant le holdout."
        )
    if manifest.get("production_pipeline_evidence") is not False:
        raise ResearchValidationCorrectorError(
            "Ce mode research part uniquement d'un candidat LoRA brut."
        )
    # The primary target is part of the sealed schema order.  A post-training
    # CLI/API target choice would create another, unregistered selection axis.
    selected_target = config.target_columns[0]
    if batch_size <= 0 or inference_chunk_size <= 0:
        raise ResearchValidationCorrectorError(
            "batch_size et inference_chunk_size doivent etre positifs."
        )
    identity = _source_identity(
        config, run_dir, manifest, item_id=str(item_id)
    )
    destination = (
        _resolve_without_links(output_directory, label="sortie fit research")
        if output_directory is not None
        else default_research_directory(config, manifest, item_id=item_id)
    )
    _assert_outside_candidate_bundle(
        destination,
        run_directory=run_dir,
        label="La sortie fit research",
    )
    if destination.exists():
        raise ResearchValidationCorrectorError(
            f"Sortie research deja existante et immutable: {destination}."
        )
    validation, origins, parquet_filter_audit = _read_validation_slice(
        config, manifest, item_id=str(item_id)
    )
    # Deliberately no injectable pipeline/loader on this publishing API: the
    # predictions must come from the checkpoint whose bytes verify_bundle and
    # source_identity have pinned.  Unit tests patch this symbol internally.
    model = load_checkpoint(run_dir, device_map=config.device_map)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep cache temporary paths below the legacy Windows MAX_PATH threshold.
    staging = destination.parent / f".rv-fit-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        predictions, input_contract = _predict_validation(
            validation,
            origins,
            config,
            candidate_pipeline=model,
            item_id=str(item_id),
            target_column=selected_target,
            checkpoint_sha256=identity["checkpoint_sha256"],
            batch_size=int(batch_size),
            inference_chunk_size=int(inference_chunk_size),
            cache_directory=staging / "cache",
            progress=progress,
        )
        current_manifest = verify_bundle(run_dir)
        current_identity = _source_identity(
            config, run_dir, current_manifest, item_id=str(item_id)
        )
        if current_manifest != manifest or current_identity != identity:
            raise ResearchValidationCorrectorError(
                "Le bundle, le panel ou l'implementation a change pendant le fit."
            )
        predictions_path = staging / VALIDATION_PREDICTIONS_NAME
        predictions.to_csv(predictions_path, index=False, compression="gzip")
        predictions_sha = _sha256(predictions_path)
        corrector = _fit_corrector(
            predictions,
            timezone_name=config.timezone,
            checkpoint_sha256=identity["checkpoint_sha256"],
            panel_sha256=identity["panel_sha256"],
            panel_audit_sha256=identity["panel_audit_sha256"],
            predictions_sha256=predictions_sha,
            input_contract_sha256=input_contract,
            item_id=str(item_id),
            target_column=selected_target,
            validation_first_origin=origins[0],
            validation_last_origin=origins[-1],
            parquet_filter_audit=parquet_filter_audit,
        )
        corrector_path = staging / CORRECTOR_NAME
        corrector_path.write_text(_json_text(corrector), encoding="utf-8")
        manifest_payload = {
            "schema_version": 1,
            "kind": FIT_KIND,
            "fit_protocol": FIT_PROTOCOL,
            "created_at_utc": _now(),
            "research_only": True,
            "validation_used_for_checkpoint_monitoring_and_selection": True,
            "calibration_predictions_are_not_oof": True,
            "holdout_used_for_fit": False,
            "holdout_logically_used_for_fit": False,
            "holdout_actuals_used_for_fit": False,
            "raw_holdout_evaluation_artifacts_opened_by_fit_process": False,
            "returned_holdout_rows": False,
            "parquet_engine_may_decode_holdout_rows": bool(
                parquet_filter_audit["parquet_engine_may_decode_holdout_rows"]
            ),
            "parquet_page_level_decode_scope_attested": False,
            "full_panel_bytes_hashed_for_identity": True,
            "historical_holdout_may_have_been_previously_inspected": True,
            "hyperparameters_fixed_without_holdout_grid": True,
            "candidate_selection_eligible": False,
            "final_backtest_eligible": False,
            "shadow_eligible": False,
            "governance_eligible": False,
            "promotion_eligible": False,
            "forbidden_consumers": [
                "FinalBacktest",
                "Shadow",
                "Govern",
                "Promote",
            ],
            "source_run_directory": str(run_dir),
            "source_identity": identity,
            "item_id": str(item_id),
            "target_column": selected_target,
            "validation_window": {
                "origins": EXPECTED_VALIDATION_DAYS,
                "first_origin_utc": origins[0].isoformat(),
                "last_origin_utc": origins[-1].isoformat(),
                "rows": int(len(predictions)),
                "panel_rows": int(len(validation)),
            },
            "holdout_window": {
                "origins": EXPECTED_HOLDOUT_DAYS,
                "first_origin_utc": manifest["splits"]["evaluation_holdout"][
                    "first_utc"
                ],
                "last_origin_utc": manifest["splits"]["evaluation_holdout"][
                    "last_utc"
                ],
            },
            "parquet_filter_audit": parquet_filter_audit,
            "hyperparameters": _hyperparameter_contract(),
            "input_contract_sha256": input_contract,
            "artifacts": {
                "corrector": {
                    "relative_path": CORRECTOR_NAME,
                    "sha256": _sha256(corrector_path),
                },
                "validation_predictions": {
                    "relative_path": VALIDATION_PREDICTIONS_NAME,
                    "sha256": predictions_sha,
                },
            },
        }
        manifest_path = staging / FIT_MANIFEST_NAME
        manifest_path.write_text(_json_text(manifest_payload), encoding="utf-8")
        cache = staging / "cache"
        if cache.exists():
            shutil.rmtree(cache)
        _verify_fit_seal(staging)
        precommit_manifest = verify_bundle(run_dir)
        precommit_identity = _source_identity(
            config,
            run_dir,
            precommit_manifest,
            item_id=str(item_id),
        )
        if precommit_manifest != manifest or precommit_identity != identity:
            raise ResearchValidationCorrectorError(
                "Le bundle, le panel ou l'implementation a change avant "
                "la publication du fit research."
            )
        if destination.exists():
            raise ResearchValidationCorrectorError(
                f"Sortie research creee pendant le fit: {destination}."
            )
        os.replace(staging, destination)
        _verify_fit_seal(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ResearchFitArtifacts(
        output_directory=destination,
        corrector_path=destination / CORRECTOR_NAME,
        predictions_path=destination / VALIDATION_PREDICTIONS_NAME,
        manifest_path=destination / FIT_MANIFEST_NAME,
    )


def _digest_without_field(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_utc(value: object, *, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ResearchValidationCorrectorError(f"Timestamp invalide: {label}.")
    return pd.Timestamp(parsed)


def _verify_parquet_filter_audit(
    audit: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    corrector: Mapping[str, Any],
    predictions: pd.DataFrame,
) -> None:
    """Verify logical filter evidence and keep physical-decode claims modest."""

    expected_fields = {
        "schema_version",
        "reader",
        "logical_filter_on_origin_and_item",
        "validation_first_origin_utc",
        "validation_last_origin_utc",
        "holdout_first_origin_utc",
        "selected_row_groups",
        "selected_row_groups_wholly_within_validation",
        "parquet_engine_may_decode_nonvalidation_rows",
        "parquet_engine_may_decode_holdout_rows",
        "parquet_page_level_decode_scope_attested",
        "returned_origins_exactly_validation",
        "returned_holdout_rows",
        "holdout_labels_used_for_fit",
        "returned_rows",
        "returned_origin_count",
        "returned_item_id",
        "audit_sha256",
    }
    if set(audit) != expected_fields:
        raise ResearchValidationCorrectorError(
            "Schema exact de l'audit du filtre Parquet invalide."
        )
    fixed: Mapping[str, object] = {
        "schema_version": 1,
        "reader": "pandas.read_parquet_pyarrow_filtered",
        "logical_filter_on_origin_and_item": True,
        "parquet_page_level_decode_scope_attested": False,
        "returned_origins_exactly_validation": True,
        "returned_holdout_rows": False,
        "holdout_labels_used_for_fit": False,
        "returned_origin_count": EXPECTED_VALIDATION_DAYS,
        "returned_item_id": manifest.get("item_id"),
    }
    if any(
        type(audit.get(key)) is not type(value) or audit.get(key) != value
        for key, value in fixed.items()
    ):
        raise ResearchValidationCorrectorError(
            "Audit logique du filtre Parquet invalide."
        )
    if (
        not _is_sha256(audit.get("audit_sha256"))
        or audit.get("audit_sha256")
        != _digest_without_field(audit, "audit_sha256")
        or corrector.get("parquet_filter_audit_sha256")
        != audit.get("audit_sha256")
    ):
        raise ResearchValidationCorrectorError(
            "Empreinte de l'audit du filtre Parquet invalide."
        )

    origins = pd.DatetimeIndex(
        predictions["forecast_origin_utc"].drop_duplicates()
    )
    validation_first = pd.Timestamp(origins[0])
    validation_last = pd.Timestamp(origins[-1])
    holdout_first = _parse_utc(
        audit.get("holdout_first_origin_utc"),
        label="holdout_first_origin_utc Parquet",
    )
    validation_window = manifest.get("validation_window")
    holdout_window = manifest.get("holdout_window")
    declared_holdout_first = (
        _parse_utc(
            holdout_window.get("first_origin_utc"),
            label="first_origin_utc holdout research",
        )
        if isinstance(holdout_window, Mapping)
        else None
    )
    if (
        not isinstance(validation_window, Mapping)
        or type(validation_window.get("panel_rows")) is not int
        or validation_window.get("panel_rows") != audit.get("returned_rows")
        or not isinstance(holdout_window, Mapping)
        or set(holdout_window) != {
            "origins",
            "first_origin_utc",
            "last_origin_utc",
        }
        or holdout_window.get("origins") != EXPECTED_HOLDOUT_DAYS
        or declared_holdout_first != holdout_first
        or audit.get("validation_first_origin_utc")
        != validation_first.isoformat()
        or audit.get("validation_last_origin_utc")
        != validation_last.isoformat()
        or type(audit.get("returned_rows")) is not int
        or int(audit["returned_rows"]) <= len(predictions)
    ):
        raise ResearchValidationCorrectorError(
            "Bornes/volumetrie de l'audit du filtre Parquet incoherentes."
        )
    holdout_last = _parse_utc(
        holdout_window.get("last_origin_utc"),
        label="last_origin_utc holdout research",
    )
    timezone_name = str(corrector.get("timezone", ""))
    try:
        expected_first = validation_last.tz_convert(timezone_name) + pd.DateOffset(
            days=1
        )
        expected_last = holdout_first.tz_convert(timezone_name) + pd.DateOffset(
            days=EXPECTED_HOLDOUT_DAYS - 1
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Timezone de l'audit Parquet invalide."
        ) from exc
    if (
        holdout_first <= validation_last
        or holdout_first.tz_convert(timezone_name) != expected_first
        or holdout_last.tz_convert(timezone_name) != expected_last
    ):
        raise ResearchValidationCorrectorError(
            "Le holdout de l'audit Parquet n'est pas la suite civile exacte."
        )

    selected = audit.get("selected_row_groups")
    if not isinstance(selected, list) or not selected:
        raise ResearchValidationCorrectorError(
            "Row groups selectionnes absents de l'audit Parquet."
        )
    row_group_fields = {
        "row_group",
        "rows",
        "minimum_origin_utc",
        "maximum_origin_utc",
        "wholly_within_validation",
        "may_include_holdout_rows",
    }
    indexes: list[int] = []
    isolated_values: list[bool] = []
    holdout_values: list[bool] = []
    for record in selected:
        if not isinstance(record, Mapping) or set(record) != row_group_fields:
            raise ResearchValidationCorrectorError(
                "Enregistrement de row group Parquet invalide."
            )
        index = record.get("row_group")
        rows = record.get("rows")
        isolated = record.get("wholly_within_validation")
        may_holdout = record.get("may_include_holdout_rows")
        if (
            type(index) is not int
            or index < 0
            or type(rows) is not int
            or rows <= 0
            or type(isolated) is not bool
            or type(may_holdout) is not bool
        ):
            raise ResearchValidationCorrectorError(
                "Types du row group Parquet invalides."
            )
        minimum_raw = record.get("minimum_origin_utc")
        maximum_raw = record.get("maximum_origin_utc")
        if minimum_raw is None or maximum_raw is None:
            if minimum_raw is not None or maximum_raw is not None:
                raise ResearchValidationCorrectorError(
                    "Statistiques min/max partielles dans l'audit Parquet."
                )
            expected_isolated = False
            expected_may_holdout = True
        else:
            minimum = _parse_utc(
                minimum_raw, label=f"row_group[{index}].minimum_origin_utc"
            )
            maximum = _parse_utc(
                maximum_raw, label=f"row_group[{index}].maximum_origin_utc"
            )
            if minimum > maximum or not (
                maximum >= validation_first and minimum <= validation_last
            ):
                raise ResearchValidationCorrectorError(
                    "Row group declare hors du filtre validation."
                )
            expected_isolated = (
                minimum >= validation_first and maximum <= validation_last
            )
            expected_may_holdout = maximum >= holdout_first
        if isolated != expected_isolated or may_holdout != expected_may_holdout:
            raise ResearchValidationCorrectorError(
                "Portee physique declaree du row group Parquet incoherente."
            )
        indexes.append(index)
        isolated_values.append(isolated)
        holdout_values.append(may_holdout)
    if indexes != sorted(set(indexes)):
        raise ResearchValidationCorrectorError(
            "Index des row groups Parquet duplique ou non ordonne."
        )
    all_isolated = all(isolated_values)
    may_decode_holdout = any(holdout_values)
    booleans: Mapping[str, bool] = {
        "selected_row_groups_wholly_within_validation": all_isolated,
        "parquet_engine_may_decode_nonvalidation_rows": not all_isolated,
        "parquet_engine_may_decode_holdout_rows": may_decode_holdout,
    }
    if any(
        type(audit.get(key)) is not bool or audit.get(key) is not value
        for key, value in booleans.items()
    ) or (
        type(corrector.get("parquet_engine_may_decode_holdout_rows")) is not bool
        or corrector.get("parquet_engine_may_decode_holdout_rows")
        is not may_decode_holdout
        or type(manifest.get("parquet_engine_may_decode_holdout_rows")) is not bool
        or manifest.get("parquet_engine_may_decode_holdout_rows")
        is not may_decode_holdout
    ):
        raise ResearchValidationCorrectorError(
            "Agregats de portee physique Parquet incoherents."
        )


def _verify_validation_predictions(
    payload: bytes,
    *,
    manifest: Mapping[str, Any],
    corrector: Mapping[str, Any],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(payload), compression="gzip")
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Predictions validation research illisibles."
        ) from exc
    if tuple(frame.columns) != VALIDATION_COLUMNS:
        raise ResearchValidationCorrectorError(
            "Schema exact des predictions validation invalide."
        )
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise ResearchValidationCorrectorError(
                f"Timestamp validation invalide: {column}."
            )
    numeric_columns = list(VALIDATION_COLUMNS[2:])
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(frame[numeric_columns].to_numpy(float)).all():
        raise ResearchValidationCorrectorError(
            "Actuals/predictions validation non finis."
        )
    if bool(
        (frame["candidate_q10"] > frame["candidate_q50"]).any()
        or (frame["candidate_q50"] > frame["candidate_q90"]).any()
    ):
        raise ResearchValidationCorrectorError(
            "Croisement de quantiles dans la validation research."
        )
    delivery = pd.DatetimeIndex(frame["delivery_start_utc"])
    if not delivery.is_monotonic_increasing or delivery.duplicated().any():
        raise ResearchValidationCorrectorError(
            "Timeline validation non triee ou dupliquee."
        )
    origins = pd.DatetimeIndex(frame["forecast_origin_utc"].drop_duplicates())
    if len(origins) != EXPECTED_VALIDATION_DAYS:
        raise ResearchValidationCorrectorError(
            "Les predictions ne couvrent pas exactement validation30."
        )
    timezone_name = str(corrector.get("timezone", ""))
    try:
        local_origins = origins.tz_convert(timezone_name)
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Timezone du correcteur research invalide."
        ) from exc
    if any(
        local_origins[index].date()
        != local_origins[0].date() + pd.Timedelta(days=index)
        or (local_origins[index].hour, local_origins[index].minute)
        != (local_origins[0].hour, local_origins[0].minute)
        for index in range(len(local_origins))
    ):
        raise ResearchValidationCorrectorError(
            "Origines validation non consecutives au meme cutoff civil."
        )
    for origin, origin_local in zip(origins, local_origins, strict=True):
        mask = frame["forecast_origin_utc"].eq(origin).to_numpy()
        actual_delivery = pd.DatetimeIndex(frame.loc[mask, "delivery_start_utc"])
        delivery_day = origin_local.date() + pd.Timedelta(days=1)
        start = pd.Timestamp(delivery_day, tz=timezone_name)
        end = pd.Timestamp(delivery_day + pd.Timedelta(days=1), tz=timezone_name)
        expected_delivery = pd.date_range(
            start.tz_convert("UTC"),
            end.tz_convert("UTC"),
            inclusive="left",
            freq="h",
        )
        if not _same_datetime_index(actual_delivery, expected_delivery):
            raise ResearchValidationCorrectorError(
                f"Timeline validation/DST invalide pour {origin.isoformat()}."
            )
    window = manifest.get("validation_window")
    if not isinstance(window, Mapping) or (
        window.get("origins") != EXPECTED_VALIDATION_DAYS
        or window.get("rows") != len(frame)
        or window.get("first_origin_utc") != origins[0].isoformat()
        or window.get("last_origin_utc") != origins[-1].isoformat()
    ):
        raise ResearchValidationCorrectorError(
            "Fenetre validation du manifeste research divergente."
        )
    if (
        corrector.get("calibration_start_origin_utc") != origins[0].isoformat()
        or corrector.get("calibration_end_origin_utc") != origins[-1].isoformat()
    ):
        raise ResearchValidationCorrectorError(
            "Bornes du correcteur et predictions validation divergentes."
        )
    return frame


def _verify_fit_seal(research_directory: Path) -> _FitSeal:
    research_directory = _resolve_without_links(
        research_directory, label="repertoire fit research"
    )
    _assert_research_closure(research_directory)
    manifest, manifest_sha = _read_json_snapshot(
        research_directory / FIT_MANIFEST_NAME,
        label="manifeste fit research",
        private=True,
    )
    expected: Mapping[str, object] = {
        "schema_version": 1,
        "kind": FIT_KIND,
        "fit_protocol": FIT_PROTOCOL,
        "research_only": True,
        "validation_used_for_checkpoint_monitoring_and_selection": True,
        "calibration_predictions_are_not_oof": True,
        "holdout_used_for_fit": False,
        "holdout_logically_used_for_fit": False,
        "holdout_actuals_used_for_fit": False,
        "raw_holdout_evaluation_artifacts_opened_by_fit_process": False,
        "returned_holdout_rows": False,
        "parquet_page_level_decode_scope_attested": False,
        "full_panel_bytes_hashed_for_identity": True,
        "hyperparameters_fixed_without_holdout_grid": True,
        "candidate_selection_eligible": False,
        "final_backtest_eligible": False,
        "shadow_eligible": False,
        "governance_eligible": False,
        "promotion_eligible": False,
    }
    failures = [
        key
        for key, value in expected.items()
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value
    ]
    if failures:
        raise ResearchValidationCorrectorError(
            "Contrat research invalide: " + ", ".join(failures) + "."
        )
    if manifest.get("hyperparameters") != _hyperparameter_contract():
        raise ResearchValidationCorrectorError(
            "Les hyperparametres research divergent du protocole fige."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "corrector",
        "validation_predictions",
    }:
        raise ResearchValidationCorrectorError("Artefacts research absents.")
    corrector_ref = artifacts.get("corrector")
    predictions_ref = artifacts.get("validation_predictions")
    if not isinstance(corrector_ref, Mapping) or not isinstance(
        predictions_ref, Mapping
    ):
        raise ResearchValidationCorrectorError("References research invalides.")
    corrector_path = _safe_artifact(
        research_directory,
        corrector_ref.get("relative_path"),
        expected_name=CORRECTOR_NAME,
    )
    predictions_path = _safe_artifact(
        research_directory,
        predictions_ref.get("relative_path"),
        expected_name=VALIDATION_PREDICTIONS_NAME,
    )
    artifact_payloads: dict[str, bytes] = {}
    for name, path, reference in (
        ("corrector", corrector_path, corrector_ref),
        ("validation_predictions", predictions_path, predictions_ref),
    ):
        declared = reference.get("sha256")
        payload = _read_bytes_stable(
            path, label=f"artefact research {name}", private=True
        )
        actual = hashlib.sha256(payload).hexdigest()
        if not _is_sha256(declared) or actual != declared:
            raise ResearchValidationCorrectorError(
                f"SHA research divergent: {path.name}."
            )
        artifact_payloads[name] = payload
    corrector = _strict_json_bytes(
        artifact_payloads["corrector"], label="correcteur research"
    )
    corrector_expected: Mapping[str, object] = {
        "schema_version": 1,
        "kind": CORRECTOR_KIND,
        "model_kind": "linear_shift_v1_research_validation_only",
        "fit_protocol": FIT_PROTOCOL,
        "calibration_days": EXPECTED_VALIDATION_DAYS,
        "calibration_predictions_are_not_oof": True,
        "validation_used_for_checkpoint_monitoring_and_selection": True,
        "holdout_used_for_fit": False,
        "holdout_logically_used_for_fit": False,
        "holdout_actuals_used_for_fit": False,
        "returned_holdout_rows": False,
        "parquet_page_level_decode_scope_attested": False,
        "full_panel_bytes_hashed_for_identity": True,
        "future_actuals_used_as_features": False,
        "hyperparameters_fixed_without_holdout_grid": True,
        "research_only": True,
        "candidate_selection_eligible": False,
        "final_backtest_eligible": False,
        "shadow_eligible": False,
        "governance_eligible": False,
        "promotion_eligible": False,
        "base_model": "chronos2_exogenous",
        "output_model": "chronos2_exogenous_research_validation_corrected",
        "candidate_checkpoint_frozen_before_holdout_evaluation": True,
    }
    failures = [
        key
        for key, value in corrector_expected.items()
        if type(corrector.get(key)) is not type(value) or corrector.get(key) != value
    ]
    if failures:
        raise ResearchValidationCorrectorError(
            "Correcteur research invalide: " + ", ".join(failures) + "."
        )
    predictions_sha = hashlib.sha256(
        artifact_payloads["validation_predictions"]
    ).hexdigest()
    if corrector.get("validation_predictions_sha256") != predictions_sha:
        raise ResearchValidationCorrectorError(
            "Le correcteur n'est pas lie aux predictions validation."
        )
    input_contract_sha = manifest.get("input_contract_sha256")
    if (
        not _is_sha256(input_contract_sha)
        or corrector.get("input_contract_sha256") != input_contract_sha
    ):
        raise ResearchValidationCorrectorError(
            "Le correcteur n'est pas lie au contrat d'entree validation."
        )
    if corrector.get("feature_columns") != list(FEATURE_COLUMNS) or corrector.get(
        "hyperparameter_contract_sha256"
    ) != _hyperparameter_contract()["contract_sha256"]:
        raise ResearchValidationCorrectorError(
            "Famille/hyperparametres du correcteur research invalides."
        )
    if (
        type(corrector.get("ridge_alpha")) is not float
        or corrector.get("ridge_alpha") != RIDGE_ALPHA
        or type(corrector.get("maximum_absolute_shift_eur_mwh")) is not float
        or corrector.get("maximum_absolute_shift_eur_mwh")
        != MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH
        or corrector.get("forbidden_consumers")
        != ["FinalBacktest", "Shadow", "Govern", "Promote"]
        or manifest.get("forbidden_consumers")
        != ["FinalBacktest", "Shadow", "Govern", "Promote"]
    ):
        raise ResearchValidationCorrectorError(
            "Parametres/consommateurs interdits du correcteur research invalides."
        )
    source_identity = manifest.get("source_identity")
    required_identity_sha = {
        "panel_sha256",
        "panel_audit_sha256",
        "schema_sha256",
        "checkpoint_sha256",
        "candidate_contract_sha256",
        "config_semantic_sha256",
        "experiment_manifest_sha256_at_observation",
        "implementation_research_corrector_sha256",
        "implementation_evaluation_sha256",
        "implementation_lora_finetune_sha256",
    }
    if not isinstance(source_identity, Mapping) or any(
        not _is_sha256(source_identity.get(key)) for key in required_identity_sha
    ):
        raise ResearchValidationCorrectorError(
            "Identite source du fit research incomplete."
        )
    observed_evidence = source_identity.get(
        "evaluation_evidence_sha256_at_observation"
    )
    if observed_evidence != "absent" and not _is_sha256(observed_evidence):
        raise ResearchValidationCorrectorError(
            "Etat evaluation_evidence au fit invalide."
        )
    for corrector_key, identity_key in (
        ("candidate_checkpoint_sha256", "checkpoint_sha256"),
        ("panel_sha256", "panel_sha256"),
        ("panel_audit_sha256", "panel_audit_sha256"),
    ):
        if corrector.get(corrector_key) != source_identity.get(identity_key):
            raise ResearchValidationCorrectorError(
                "Le correcteur n'est pas lie a l'identite source du fit."
            )
    if (
        corrector.get("item_id") != manifest.get("item_id")
        or corrector.get("target_column") != manifest.get("target_column")
    ):
        raise ResearchValidationCorrectorError(
            "Item/target du correcteur et du manifeste divergent."
        )
    parquet_audit = manifest.get("parquet_filter_audit")
    if not isinstance(parquet_audit, Mapping):
        raise ResearchValidationCorrectorError(
            "Audit du filtre Parquet research absent."
        )
    predictions = _verify_validation_predictions(
        artifact_payloads["validation_predictions"],
        manifest=manifest,
        corrector=corrector,
    )
    _verify_parquet_filter_audit(
        parquet_audit,
        manifest=manifest,
        corrector=corrector,
        predictions=predictions,
    )
    design = _design(
        predictions["delivery_start_utc"], timezone_name=str(corrector["timezone"])
    )
    means = design.mean(axis=0)
    scales = design.std(axis=0)
    means[0], scales[0] = 0.0, 1.0
    scales[scales <= 1e-12] = 1.0
    normalised = (design - means) / scales
    residual = (
        predictions["actual"].to_numpy(float)
        - predictions["candidate_q50"].to_numpy(float)
    )
    penalty = np.eye(normalised.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.lstsq(
        normalised.T @ normalised + penalty,
        normalised.T @ residual,
        rcond=None,
    )[0]
    fitted = np.clip(
        normalised @ coefficients,
        -MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
        MAXIMUM_ABSOLUTE_SHIFT_EUR_MWH,
    )
    declared_arrays = (
        np.asarray(corrector.get("feature_means"), dtype=float),
        np.asarray(corrector.get("feature_scales"), dtype=float),
        np.asarray(corrector.get("coefficients"), dtype=float),
    )
    expected_arrays = (means, scales, coefficients)
    if any(
        declared.shape != expected.shape
        or not np.isfinite(declared).all()
        or not np.allclose(declared, expected, rtol=0.0, atol=1e-12)
        for declared, expected in zip(declared_arrays, expected_arrays, strict=True)
    ) or not bool((declared_arrays[1] > 0.0).all()):
        raise ResearchValidationCorrectorError(
            "Coefficients/normalisation du correcteur non reproductibles."
        )
    if (
        corrector.get("fit_rows") != len(predictions)
        or not np.isclose(
            float(corrector.get("fit_mae_before_eur_mwh", np.nan)),
            float(np.mean(np.abs(residual))),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(corrector.get("fit_mae_after_eur_mwh", np.nan)),
            float(np.mean(np.abs(residual - fitted))),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ResearchValidationCorrectorError(
            "Metriques de fit research non reproductibles."
        )
    sealed = _parse_utc(corrector.get("sealed_at_utc"), label="sealed_at_utc")
    created = _parse_utc(manifest.get("created_at_utc"), label="created_at_utc")
    if sealed > created:
        raise ResearchValidationCorrectorError(
            "Le manifeste fit est anterieur au scellement du correcteur."
        )
    return _FitSeal(
        manifest=manifest,
        corrector=corrector,
        manifest_sha256=manifest_sha,
        corrector_sha256=hashlib.sha256(
            artifact_payloads["corrector"]
        ).hexdigest(),
        validation_predictions_sha256=predictions_sha,
    )


def _verified_holdout_evidence(
    run_directory: Path,
    config: ExogenousFineTuneConfig,
    experiment: Mapping[str, Any],
    *,
    item_id: str,
    target_column: str,
) -> _RawHoldout:
    evaluation_manifest_path = run_directory / "evaluation_manifest.json"
    evaluation, evaluation_manifest_sha = _read_json_snapshot(
        evaluation_manifest_path,
        label="manifeste holdout LoRA brut",
    )
    comparison = evaluation.get("comparison")
    window = evaluation.get("window")
    if (
        evaluation.get("format_version") != 1
        or evaluation.get("kind")
        != "chronos2_exogenous_lora_rolling365_evaluation"
        or evaluation.get("experiment_id") != experiment.get("experiment_id")
        or evaluation.get("evaluation_role") != experiment.get("evaluation_role")
        or evaluation.get("item_id") != str(item_id)
        or evaluation.get("target_column") != target_column
        or not isinstance(comparison, Mapping)
        or comparison.get("baseline") != BASELINE_LABEL
        or comparison.get("candidate") != CANDIDATE_LABEL
        or comparison.get("same_inputs") is not True
        or comparison.get("cross_learning") is not False
        or comparison.get("residual_corrector_applied") is not False
        or not isinstance(window, Mapping)
        or window.get("physical_days") != EXPECTED_HOLDOUT_DAYS
        or not _is_sha256(evaluation.get("input_contract_sha256"))
        or evaluation.get("evaluation_label_resolution")
        != experiment.get("evaluation_label_resolution")
    ):
        raise ResearchValidationCorrectorError(
            "La preuve holdout brute n'a pas le contrat attendu."
        )
    _parse_utc(evaluation.get("created_at_utc"), label="created_at_utc holdout")
    current_manifest, current_manifest_sha = _read_json_snapshot(
        run_directory / "experiment_manifest.json",
        label="manifeste LoRA courant",
    )
    if current_manifest != dict(experiment):
        raise ResearchValidationCorrectorError(
            "Le manifeste LoRA a change avant la lecture du holdout."
        )
    if evaluation.get("bundle_manifest_sha256") != current_manifest_sha:
        raise ResearchValidationCorrectorError(
            "Le manifeste holdout n'est pas lie au bundle courant."
        )
    artifacts = evaluation.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        RAW_EVALUATION_ARTIFACTS
    ):
        raise ResearchValidationCorrectorError(
            "Closure des artefacts holdout bruts invalide."
        )
    artifact_hashes: dict[str, str] = {}
    artifact_payloads: dict[str, bytes] = {}
    for name, expected_name in RAW_EVALUATION_ARTIFACTS.items():
        reference = artifacts.get(name)
        if not isinstance(reference, Mapping) or set(reference) != {
            "relative_path",
            "sha256",
        }:
            raise ResearchValidationCorrectorError(
                f"Reference holdout brute invalide: {name}."
            )
        path = _safe_artifact(
            run_directory,
            reference.get("relative_path"),
            expected_name=expected_name,
        )
        payload = _read_bytes_stable(path, label=f"artefact holdout brut {name}")
        actual = hashlib.sha256(payload).hexdigest()
        if not _is_sha256(reference.get("sha256")) or reference.get(
            "sha256"
        ) != actual:
            raise ResearchValidationCorrectorError(
                f"SHA holdout brut divergent: {name}."
            )
        artifact_hashes[name] = actual
        artifact_payloads[name] = payload
    try:
        frame = pd.read_csv(
            io.BytesIO(artifact_payloads["evidence"]), compression="gzip"
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Preuve holdout brute illisible."
        ) from exc
    if tuple(frame.columns) != EVALUATION_COLUMNS:
        raise ResearchValidationCorrectorError(
            "Schema exact de la preuve holdout requis."
        )
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    numeric_columns = list(EVALUATION_COLUMNS[2:])
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    _validate_holdout_frame(frame, config, experiment)
    recomputed_metrics, recomputed_daily = compute_metrics(
        frame, timezone_name=config.timezone
    )
    declared_metrics = _strict_json_bytes(
        artifact_payloads["metrics"], label="metriques holdout brutes"
    )
    for key, expected in recomputed_metrics.items():
        observed = declared_metrics.get(key)
        if isinstance(expected, float):
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise ResearchValidationCorrectorError(
                    f"Metrique holdout brute invalide: {key}."
                )
            if not np.isclose(
                float(observed), expected, rtol=0.0, atol=1e-12
            ):
                raise ResearchValidationCorrectorError(
                    f"Metrique holdout brute divergente: {key}."
                )
        elif observed != expected:
            raise ResearchValidationCorrectorError(
                f"Metrique holdout brute divergente: {key}."
            )
    try:
        declared_daily = pd.read_csv(
            io.BytesIO(artifact_payloads["daily"]), compression="gzip"
        )
        pd.testing.assert_frame_equal(
            declared_daily,
            recomputed_daily,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Detail journalier holdout brut divergent."
        ) from exc
    if (
        window.get("physical_hours") != len(frame)
        or window.get("first_delivery_utc")
        != recomputed_metrics["first_delivery_utc"]
        or window.get("last_delivery_utc")
        != recomputed_metrics["last_delivery_utc"]
        or window.get("dst_days") != recomputed_metrics["dst_days"]
    ):
        raise ResearchValidationCorrectorError(
            "Fenetre declaree du holdout brut divergente."
        )
    experiment_reference = experiment.get("evaluation_evidence")
    if not isinstance(experiment_reference, Mapping) or (
        experiment_reference.get("relative_path")
        != RAW_EVALUATION_ARTIFACTS["evidence"]
        or experiment_reference.get("sha256") != artifact_hashes["evidence"]
        or experiment_reference.get("rows") != len(frame)
        or experiment_reference.get("physical_days") != EXPECTED_HOLDOUT_DAYS
        or experiment_reference.get("first_delivery_utc")
        != recomputed_metrics["first_delivery_utc"]
        or experiment_reference.get("last_delivery_utc")
        != recomputed_metrics["last_delivery_utc"]
    ):
        raise ResearchValidationCorrectorError(
            "evaluation_evidence du bundle ne lie pas le holdout brut canonique."
        )
    return _RawHoldout(
        frame=frame,
        manifest=evaluation,
        manifest_sha256=evaluation_manifest_sha,
        evidence_sha256=artifact_hashes["evidence"],
        artifact_hashes=artifact_hashes,
    )


def _validate_holdout_frame(
    frame: pd.DataFrame,
    config: ExogenousFineTuneConfig,
    experiment: Mapping[str, Any],
) -> None:
    if tuple(frame.columns) != EVALUATION_COLUMNS:
        raise ResearchValidationCorrectorError(
            "Schema exact de la preuve holdout requis."
        )
    delivery = pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="coerce")
    origins = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    if delivery.isna().any() or origins.isna().any():
        raise ResearchValidationCorrectorError("Timestamps holdout invalides.")
    delivery_index = pd.DatetimeIndex(delivery)
    origin_index = pd.DatetimeIndex(origins)
    if not delivery_index.is_monotonic_increasing or delivery_index.duplicated().any():
        raise ResearchValidationCorrectorError(
            "Timeline holdout non triee ou dupliquee."
        )
    numeric = frame[list(EVALUATION_COLUMNS[2:])].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ResearchValidationCorrectorError("Valeurs holdout non finies.")
    for prefix in ("baseline", "candidate"):
        if bool(
            (numeric[f"{prefix}_q10"] > numeric[f"{prefix}_q50"]).any()
            or (numeric[f"{prefix}_q50"] > numeric[f"{prefix}_q90"]).any()
        ):
            raise ResearchValidationCorrectorError("Croisement de quantiles holdout.")
    local = delivery_index.tz_convert(config.timezone)
    days = tuple(dict.fromkeys(local.date))
    if len(days) != EXPECTED_HOLDOUT_DAYS or any(
        days[index] != days[0] + pd.Timedelta(days=index)
        for index in range(len(days))
    ):
        raise ResearchValidationCorrectorError(
            "Le holdout ne couvre pas 365 jours civils consecutifs."
        )
    hour, minute = (int(part) for part in config.cutoff_local_time.split(":"))
    for day in days:
        start = pd.Timestamp(day, tz=config.timezone)
        end = pd.Timestamp(day + pd.Timedelta(days=1), tz=config.timezone)
        expected = pd.date_range(
            start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left"
        )
        mask = np.asarray(local.date == day)
        if not _same_datetime_index(delivery_index[mask], expected):
            raise ResearchValidationCorrectorError(
                f"Jour holdout incomplet/DST invalide: {day}."
            )
        actual_origins = origin_index[mask].unique()
        expected_origin = pd.Timestamp(
            f"{day - pd.Timedelta(days=1):%Y-%m-%d} {hour:02d}:{minute:02d}",
            tz=config.timezone,
        ).tz_convert("UTC")
        if len(actual_origins) != 1 or actual_origins[0] != expected_origin:
            raise ResearchValidationCorrectorError(
                f"Origine D-1 invalide pour {day}."
            )
    first, last = _declared_split(
        experiment, "evaluation_holdout", expected_count=EXPECTED_HOLDOUT_DAYS
    )
    distinct = origin_index.unique().sort_values()
    if len(distinct) != EXPECTED_HOLDOUT_DAYS or distinct[0] != first or distinct[-1] != last:
        raise ResearchValidationCorrectorError(
            "Les origines holdout divergent du split gele."
        )


def _assert_raw_holdout_unchanged(
    run_directory: Path,
    snapshot: _RawHoldout,
) -> None:
    _payload, manifest_sha = _read_json_snapshot(
        run_directory / "evaluation_manifest.json",
        label="manifeste holdout LoRA brut",
    )
    if manifest_sha != snapshot.manifest_sha256:
        raise ResearchValidationCorrectorError(
            "Le manifeste holdout brut a change pendant l'evaluation research."
        )
    artifacts = snapshot.manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ResearchValidationCorrectorError(
            "Artefacts du holdout brut absents au recontrole."
        )
    for name, expected_name in RAW_EVALUATION_ARTIFACTS.items():
        reference = artifacts.get(name)
        if not isinstance(reference, Mapping):
            raise ResearchValidationCorrectorError(
                f"Reference holdout brute absente au recontrole: {name}."
            )
        path = _safe_artifact(
            run_directory,
            reference.get("relative_path"),
            expected_name=expected_name,
        )
        payload = _read_bytes_stable(
            path, label=f"artefact holdout brut {name} au recontrole"
        )
        if hashlib.sha256(payload).hexdigest() != snapshot.artifact_hashes.get(name):
            raise ResearchValidationCorrectorError(
                f"L'artefact holdout brut {name} a change pendant l'evaluation."
            )


def _verify_evaluation_publication(
    root: Path,
    *,
    seal: _FitSeal,
    raw: _RawHoldout,
    source_identity: Mapping[str, str],
    item_id: str,
    target_column: str,
) -> None:
    _assert_evaluation_closure(root)
    manifest, _manifest_sha = _read_json_snapshot(
        root / EVALUATION_MANIFEST_NAME,
        label="manifeste evaluation research",
        private=True,
    )
    expected: Mapping[str, object] = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
        "fit_protocol": FIT_PROTOCOL,
        "research_only": True,
        "calibration_predictions_are_not_oof": True,
        "validation_used_for_checkpoint_monitoring_and_selection": True,
        "holdout_used_for_fit": False,
        "holdout_logically_used_for_fit": False,
        "holdout_refit_performed": False,
        "holdout_actuals_used_for_fit": False,
        "returned_holdout_rows_during_fit": False,
        "parquet_page_level_decode_scope_attested": False,
        "corrector_verified_before_holdout_read": True,
        "historical_holdout_may_have_been_previously_inspected": True,
        "candidate_selection_eligible": False,
        "final_backtest_eligible": False,
        "shadow_eligible": False,
        "governance_eligible": False,
        "promotion_eligible": False,
        "item_id": item_id,
        "target_column": target_column,
        "research_fit_manifest_sha256": seal.manifest_sha256,
        "research_corrector_sha256": seal.corrector_sha256,
        "research_validation_predictions_sha256": (
            seal.validation_predictions_sha256
        ),
        "raw_holdout_manifest_sha256": raw.manifest_sha256,
        "raw_holdout_predictions_sha256": raw.evidence_sha256,
    }
    failures = [
        key
        for key, value in expected.items()
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value
    ]
    if failures:
        raise ResearchValidationCorrectorError(
            "Contrat de publication evaluation research invalide: "
            + ", ".join(failures)
            + "."
        )
    if (
        manifest.get("forbidden_consumers")
        != ["FinalBacktest", "Shadow", "Govern", "Promote"]
        or manifest.get("source_identity") != dict(source_identity)
        or manifest.get("raw_holdout_artifact_hashes")
        != dict(raw.artifact_hashes)
        or manifest.get("raw_holdout_input_contract_sha256")
        != raw.manifest.get("input_contract_sha256")
    ):
        raise ResearchValidationCorrectorError(
            "Filiation de l'evaluation research invalide."
        )
    verified_at = _parse_utc(
        manifest.get("corrector_verified_at_utc"),
        label="corrector_verified_at_utc",
    )
    read_started = _parse_utc(
        manifest.get("holdout_read_started_at_utc"),
        label="holdout_read_started_at_utc",
    )
    created = _parse_utc(
        manifest.get("created_at_utc"), label="created_at_utc evaluation"
    )
    if verified_at > read_started or read_started > created:
        raise ResearchValidationCorrectorError(
            "Ordre temporel du scellement/holdout/publication invalide."
        )
    artifacts = manifest.get("artifacts")
    expected_names = {
        "predictions": EVALUATION_PREDICTIONS_NAME,
        "metrics": EVALUATION_METRICS_NAME,
        "daily": EVALUATION_DAILY_NAME,
        "report": EVALUATION_REPORT_NAME,
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_names):
        raise ResearchValidationCorrectorError(
            "References d'artefacts evaluation research invalides."
        )
    artifact_payloads: dict[str, bytes] = {}
    for key, expected_name in expected_names.items():
        reference = artifacts.get(key)
        if not isinstance(reference, Mapping) or set(reference) != {
            "relative_path",
            "sha256",
        }:
            raise ResearchValidationCorrectorError(
                f"Reference evaluation research invalide: {key}."
            )
        path = _safe_artifact(
            root, reference.get("relative_path"), expected_name=expected_name
        )
        payload = _read_bytes_stable(
            path, label=f"artefact evaluation research {key}", private=True
        )
        if (
            not _is_sha256(reference.get("sha256"))
            or hashlib.sha256(payload).hexdigest() != reference.get("sha256")
        ):
            raise ResearchValidationCorrectorError(
                f"SHA evaluation research divergent: {key}."
            )
        artifact_payloads[key] = payload

    try:
        predictions = pd.read_csv(
            io.BytesIO(artifact_payloads["predictions"]), compression="gzip"
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Predictions evaluation research illisibles."
        ) from exc
    if tuple(predictions.columns) != EVALUATION_COLUMNS:
        raise ResearchValidationCorrectorError(
            "Schema exact des predictions evaluation research invalide."
        )
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        predictions[column] = pd.to_datetime(
            predictions[column], utc=True, errors="coerce"
        )
        if predictions[column].isna().any():
            raise ResearchValidationCorrectorError(
                f"Timestamp evaluation research invalide: {column}."
            )
    numeric_columns = list(EVALUATION_COLUMNS[2:])
    predictions[numeric_columns] = predictions[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(predictions[numeric_columns].to_numpy(float)).all():
        raise ResearchValidationCorrectorError(
            "Valeurs evaluation research non finies."
        )
    q10, q50, q90, shift = _apply_corrector(
        raw.frame["candidate_q10"].to_numpy(float),
        raw.frame["candidate_q50"].to_numpy(float),
        raw.frame["candidate_q90"].to_numpy(float),
        raw.frame["delivery_start_utc"],
        seal.corrector,
    )
    expected_predictions = pd.DataFrame(
        {
            "delivery_start_utc": raw.frame["delivery_start_utc"],
            "forecast_origin_utc": raw.frame["forecast_origin_utc"],
            "actual": raw.frame["actual"],
            "baseline_q10": raw.frame["candidate_q10"],
            "baseline_q50": raw.frame["candidate_q50"],
            "baseline_q90": raw.frame["candidate_q90"],
            "candidate_q10": q10,
            "candidate_q50": q50,
            "candidate_q90": q90,
        },
        columns=EVALUATION_COLUMNS,
    )
    try:
        pd.testing.assert_frame_equal(
            predictions,
            expected_predictions,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ResearchValidationCorrectorError(
            "Predictions evaluation research non reproductibles depuis le sceau."
        ) from exc

    expected_metrics, expected_daily = compute_metrics(
        predictions, timezone_name=str(seal.corrector["timezone"])
    )
    expected_metrics.update(
        {
            "comparison_scope": (
                "research_raw_lora_vs_validation30_corrected_lora"
            ),
            "baseline_label": "Chronos-2 + LoRA brut",
            "candidate_label": (
                "Chronos-2 + LoRA + correcteur validation30 research"
            ),
            "research_only": True,
            "calibration_predictions_are_not_oof": True,
            "candidate_selection_eligible": False,
            "final_backtest_eligible": False,
            "shadow_eligible": False,
            "governance_eligible": False,
            "promotion_eligible": False,
            "residual_shift_mean_eur_mwh": float(np.mean(shift)),
            "residual_shift_mean_absolute_eur_mwh": float(np.mean(np.abs(shift))),
            "residual_shift_max_absolute_eur_mwh": float(np.max(np.abs(shift))),
        }
    )
    declared_metrics = _strict_json_bytes(
        artifact_payloads["metrics"], label="metriques evaluation research"
    )
    if set(declared_metrics) != set(expected_metrics):
        raise ResearchValidationCorrectorError(
            "Schema exact des metriques evaluation research invalide."
        )
    for key, expected_value in expected_metrics.items():
        observed_value = declared_metrics.get(key)
        if isinstance(expected_value, float):
            valid = (
                not isinstance(observed_value, bool)
                and isinstance(observed_value, (int, float))
                and np.isclose(
                    float(observed_value), expected_value, rtol=0.0, atol=1e-12
                )
            )
        else:
            valid = (
                type(observed_value) is type(expected_value)
                and observed_value == expected_value
            )
        if not valid:
            raise ResearchValidationCorrectorError(
                f"Metrique evaluation research non reproductible: {key}."
            )
    try:
        declared_daily = pd.read_csv(
            io.BytesIO(artifact_payloads["daily"]), compression="gzip"
        )
        pd.testing.assert_frame_equal(
            declared_daily,
            expected_daily,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except Exception as exc:
        raise ResearchValidationCorrectorError(
            "Detail journalier evaluation research non reproductible."
        ) from exc
    expected_report = _render_report(
        declared_metrics,
        expected_daily,
        generated_at_utc=str(manifest["created_at_utc"]),
    ).encode("utf-8")
    if artifact_payloads["report"] != expected_report:
        raise ResearchValidationCorrectorError(
            "Rapport HTML evaluation research non reproductible."
        )


def _render_report(
    metrics: Mapping[str, Any],
    daily: pd.DataFrame,
    *,
    generated_at_utc: str,
) -> str:
    gain = float(metrics["mae_gain_eur_mwh"])
    colour = "good" if gain > 0 else "bad" if gain < 0 else "muted"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.delivery_day))}</td>"
        f"<td>{int(row.hours)}</td>"
        f"<td>{row.observed_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.candidate_mean_price_eur_mwh:.2f}</td>"
        f"<td>{row.baseline_hourly_mae_eur_mwh:.2f}</td>"
        f"<td>{row.candidate_hourly_mae_eur_mwh:.2f}</td>"
        "</tr>"
        for row in daily.itertuples(index=False)
    )
    return f"""<!doctype html><html lang="fr" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Correcteur validation — research-only</title><style>
:root{{--bg:#f4f7fb;--card:#fff;--text:#172033;--muted:#65738a;--line:#dbe3ef;--warn:#9b4d00;--good:#087a4b;--bad:#c43d48}}
[data-theme=dark]{{--bg:#0d1421;--card:#172131;--text:#edf3fb;--muted:#a9b6ca;--line:#304057;--warn:#ffbd66;--good:#61d79b;--bad:#ff7e87}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}}main{{max-width:1160px;margin:auto;padding:26px}}header{{display:flex;justify-content:space-between;gap:16px}}button,.card,.warning,details{{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:11px}}button{{height:38px;padding:8px 12px}}.warning{{padding:14px;border-left:5px solid var(--warn);margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{padding:15px}}.value{{font-size:23px;font-weight:700}}.muted{{color:var(--muted)}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}details{{padding:12px;margin-top:18px}}.table{{max-height:520px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--card)}}</style></head>
<body><main><header><div><h1>Correcteur résiduel sur validation — recherche uniquement</h1><div class="muted">LoRA brut contre le même LoRA après décalage résiduel figé.</div></div><button id="theme">🌙 Mode nuit</button></header>
<div class="warning"><strong>Non OOF, non promotionnable.</strong> Le correcteur a été ajusté sur les 30 jours de validation qui ont aussi servi au suivi/choix éventuel du checkpoint. Aucune ligne ni aucun label du holdout n'a été retourné ou utilisé logiquement par le fit; un row group Parquet chevauchant peut néanmoins avoir été décodé par le moteur. Ce résultat reste une analyse post-training et ne peut alimenter FinalBacktest, Shadow, Govern ou Promote.</div>
<section class="cards"><div class="card"><div class="muted">MAE LoRA brut</div><div class="value">{float(metrics['baseline_mae_eur_mwh']):.3f}</div></div><div class="card"><div class="muted">MAE LoRA corrigé</div><div class="value">{float(metrics['candidate_mae_eur_mwh']):.3f}</div></div><div class="card"><div class="muted">Gain exploratoire</div><div class="value {colour}">{gain:.3f}</div><div>EUR/MWh</div></div><div class="card"><div class="muted">Jours améliorés</div><div class="value">{int(metrics['candidate_better_hourly_mae_days'])} / 365</div></div></section>
<details><summary>Détail des 365 jours</summary><div class="table"><table><thead><tr><th>Jour</th><th>Heures</th><th>Observé</th><th>LoRA brut</th><th>LoRA corrigé</th><th>MAE brut</th><th>MAE corrigé</th></tr></thead><tbody>{rows}</tbody></table></div></details>
<p class="muted">Généré le {html.escape(generated_at_utc)} · journées DST conservées à 23/25 heures.</p></main><script>const r=document.documentElement,b=document.getElementById('theme');function a(t){{r.dataset.theme=t;b.textContent=t==='dark'?'☀️ Mode jour':'🌙 Mode nuit'}}a(localStorage.getItem('chronos2-research-validation-theme')||'light');b.onclick=()=>{{const t=r.dataset.theme==='dark'?'light':'dark';localStorage.setItem('chronos2-research-validation-theme',t);a(t)}};</script></body></html>"""


def evaluate_research_validation_corrector(
    config_or_path: ExogenousFineTuneConfig | str | Path,
    *,
    run_directory: str | Path | None = None,
    research_directory: str | Path | None = None,
    item_id: str,
) -> ResearchEvaluationArtifacts:
    """Verify the seal first, then and only then open the frozen holdout."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExogenousFineTuneConfig)
        else load_config(config_or_path)
    )
    run_dir = _resolve_without_links(
        run_directory or config.output_directory,
        label="bundle candidat",
    )
    experiment = verify_bundle(run_dir)
    selected_target = config.target_columns[0]
    research = (
        _resolve_without_links(
            research_directory, label="repertoire correcteur research"
        )
        if research_directory is not None
        else default_research_directory(config, experiment, item_id=item_id)
    )
    _assert_outside_candidate_bundle(
        research,
        run_directory=run_dir,
        label="Le repertoire correcteur research",
    )
    destination = research / EVALUATION_DIRECTORY_NAME
    _assert_outside_candidate_bundle(
        destination,
        run_directory=run_dir,
        label="La sortie evaluation research",
    )
    if destination.exists():
        raise ResearchValidationCorrectorError(
            f"Evaluation research deja existante et immutable: {destination}."
        )
    seal = _verify_fit_seal(research)
    corrector = seal.corrector
    identity = _source_identity(
        config, run_dir, experiment, item_id=str(item_id)
    )
    fit_identity = seal.manifest.get("source_identity")
    immutable_identity_keys = {
        "panel_sha256",
        "panel_audit_sha256",
        "schema_sha256",
        "checkpoint_sha256",
        "candidate_contract_sha256",
        "config_semantic_sha256",
        "implementation_research_corrector_sha256",
        "implementation_evaluation_sha256",
        "implementation_lora_finetune_sha256",
    }
    if (
        seal.manifest.get("source_run_directory") != str(run_dir)
        or not isinstance(fit_identity, Mapping)
        or {
            key: fit_identity.get(key) for key in immutable_identity_keys
        }
        != {key: identity.get(key) for key in immutable_identity_keys}
    ):
        raise ResearchValidationCorrectorError(
            "Le bundle courant differe de celui ayant produit le correcteur."
        )
    observed_evidence_sha = fit_identity.get(
        "evaluation_evidence_sha256_at_observation"
    )
    current_evidence = experiment.get("evaluation_evidence")
    current_evidence_sha = (
        current_evidence.get("sha256")
        if isinstance(current_evidence, Mapping)
        else None
    )
    if observed_evidence_sha != "absent" and observed_evidence_sha != current_evidence_sha:
        raise ResearchValidationCorrectorError(
            "Le holdout brut deja present au fit a ete remplace."
        )
    if corrector.get("candidate_checkpoint_sha256") != identity[
        "checkpoint_sha256"
    ] or corrector.get("item_id") != str(item_id) or corrector.get(
        "target_column"
    ) != selected_target:
        raise ResearchValidationCorrectorError(
            "Le correcteur research ne correspond pas au candidat/item/target."
        )
    if (
        corrector.get("timezone") != config.timezone
        or corrector.get("panel_sha256") != identity["panel_sha256"]
        or corrector.get("panel_audit_sha256")
        != identity["panel_audit_sha256"]
        or seal.manifest.get("input_contract_sha256") is None
        or not _is_sha256(seal.manifest.get("input_contract_sha256"))
    ):
        raise ResearchValidationCorrectorError(
            "Le correcteur research diverge du panel/schema/input contract."
        )
    validation, validation_origins, current_filter_audit = _read_validation_slice(
        config,
        experiment,
        item_id=str(item_id),
    )
    current_input_contract = _validation_input_contract(
        validation,
        validation_origins,
        config,
    )
    if (
        current_filter_audit != seal.manifest.get("parquet_filter_audit")
        or current_input_contract != seal.manifest.get("input_contract_sha256")
        or current_input_contract != corrector.get("input_contract_sha256")
    ):
        raise ResearchValidationCorrectorError(
            "Le filtre ou le contrat d'entree validation n'est pas reproductible."
        )
    # Every fit seal, config and immutable candidate check above completes
    # before the first canonical rolling-365 evaluation artifact byte is read.
    corrector_verified_at = _now()
    holdout_read_started_at = _now()
    raw_snapshot = _verified_holdout_evidence(
        run_dir,
        config,
        experiment,
        item_id=str(item_id),
        target_column=selected_target,
    )
    raw = raw_snapshot.frame
    q10, q50, q90, shift = _apply_corrector(
        raw["candidate_q10"].to_numpy(float),
        raw["candidate_q50"].to_numpy(float),
        raw["candidate_q90"].to_numpy(float),
        raw["delivery_start_utc"],
        corrector,
    )
    evidence = pd.DataFrame(
        {
            "delivery_start_utc": raw["delivery_start_utc"],
            "forecast_origin_utc": raw["forecast_origin_utc"],
            "actual": raw["actual"],
            "baseline_q10": raw["candidate_q10"],
            "baseline_q50": raw["candidate_q50"],
            "baseline_q90": raw["candidate_q90"],
            "candidate_q10": q10,
            "candidate_q50": q50,
            "candidate_q90": q90,
        },
        columns=EVALUATION_COLUMNS,
    )
    _validate_holdout_frame(evidence, config, experiment)
    metrics, daily = compute_metrics(evidence, timezone_name=config.timezone)
    metrics.update(
        {
            "comparison_scope": "research_raw_lora_vs_validation30_corrected_lora",
            "baseline_label": "Chronos-2 + LoRA brut",
            "candidate_label": "Chronos-2 + LoRA + correcteur validation30 research",
            "research_only": True,
            "calibration_predictions_are_not_oof": True,
            "candidate_selection_eligible": False,
            "final_backtest_eligible": False,
            "shadow_eligible": False,
            "governance_eligible": False,
            "promotion_eligible": False,
            "residual_shift_mean_eur_mwh": float(np.mean(shift)),
            "residual_shift_mean_absolute_eur_mwh": float(np.mean(np.abs(shift))),
            "residual_shift_max_absolute_eur_mwh": float(np.max(np.abs(shift))),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the transaction outside the sealed fit directory so its exact
    # closure remains verifiable while the evaluation is being assembled.
    staging = research.parent / f".rv-eval-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        evaluation_created_at = _now()
        predictions_path = staging / EVALUATION_PREDICTIONS_NAME
        evidence.to_csv(predictions_path, index=False, compression="gzip")
        metrics_path = staging / EVALUATION_METRICS_NAME
        metrics_path.write_text(_json_text(metrics), encoding="utf-8")
        daily_path = staging / EVALUATION_DAILY_NAME
        daily.to_csv(daily_path, index=False, compression="gzip")
        report_path = staging / EVALUATION_REPORT_NAME
        report_path.write_text(
            _render_report(
                metrics,
                daily,
                generated_at_utc=evaluation_created_at,
            ),
            encoding="utf-8",
            newline="\n",
        )
        manifest_payload = {
            "schema_version": 1,
            "kind": EVALUATION_KIND,
            "created_at_utc": evaluation_created_at,
            "fit_protocol": FIT_PROTOCOL,
            "research_only": True,
            "calibration_predictions_are_not_oof": True,
            "validation_used_for_checkpoint_monitoring_and_selection": True,
            "holdout_used_for_fit": False,
            "holdout_logically_used_for_fit": False,
            "holdout_refit_performed": False,
            "holdout_actuals_used_for_fit": False,
            "returned_holdout_rows_during_fit": False,
            "parquet_engine_may_decode_holdout_rows_during_fit": bool(
                seal.manifest["parquet_engine_may_decode_holdout_rows"]
            ),
            "parquet_page_level_decode_scope_attested": False,
            "corrector_verified_before_holdout_read": True,
            "corrector_verified_at_utc": corrector_verified_at,
            "holdout_read_started_at_utc": holdout_read_started_at,
            "historical_holdout_may_have_been_previously_inspected": True,
            "candidate_selection_eligible": False,
            "final_backtest_eligible": False,
            "shadow_eligible": False,
            "governance_eligible": False,
            "promotion_eligible": False,
            "forbidden_consumers": [
                "FinalBacktest",
                "Shadow",
                "Govern",
                "Promote",
            ],
            "item_id": str(item_id),
            "target_column": selected_target,
            "source_identity": identity,
            "research_fit_manifest_sha256": seal.manifest_sha256,
            "research_corrector_sha256": seal.corrector_sha256,
            "research_validation_predictions_sha256": (
                seal.validation_predictions_sha256
            ),
            "raw_holdout_manifest_sha256": raw_snapshot.manifest_sha256,
            "raw_holdout_predictions_sha256": raw_snapshot.evidence_sha256,
            "raw_holdout_artifact_hashes": dict(raw_snapshot.artifact_hashes),
            "raw_holdout_input_contract_sha256": raw_snapshot.manifest.get(
                "input_contract_sha256"
            ),
            "artifacts": {
                "predictions": {
                    "relative_path": EVALUATION_PREDICTIONS_NAME,
                    "sha256": _sha256(predictions_path),
                },
                "metrics": {
                    "relative_path": EVALUATION_METRICS_NAME,
                    "sha256": _sha256(metrics_path),
                },
                "daily": {
                    "relative_path": EVALUATION_DAILY_NAME,
                    "sha256": _sha256(daily_path),
                },
                "report": {
                    "relative_path": EVALUATION_REPORT_NAME,
                    "sha256": _sha256(report_path),
                },
            },
        }
        manifest_path = staging / EVALUATION_MANIFEST_NAME
        manifest_path.write_text(_json_text(manifest_payload), encoding="utf-8")
        _verify_evaluation_publication(
            staging,
            seal=seal,
            raw=raw_snapshot,
            source_identity=identity,
            item_id=str(item_id),
            target_column=selected_target,
        )
        precommit_seal = _verify_fit_seal(research)
        precommit_experiment = verify_bundle(run_dir)
        precommit_identity = _source_identity(
            config,
            run_dir,
            precommit_experiment,
            item_id=str(item_id),
        )
        if (
            precommit_seal.manifest_sha256 != seal.manifest_sha256
            or precommit_seal.corrector_sha256 != seal.corrector_sha256
            or precommit_seal.validation_predictions_sha256
            != seal.validation_predictions_sha256
            or precommit_experiment != experiment
            or precommit_identity != identity
        ):
            raise ResearchValidationCorrectorError(
                "Le fit, le bundle, le panel ou l'implementation a change "
                "avant la publication de l'evaluation research."
            )
        _assert_raw_holdout_unchanged(run_dir, raw_snapshot)
        if destination.exists():
            raise ResearchValidationCorrectorError(
                f"Sortie evaluation creee pendant le calcul: {destination}."
            )
        os.replace(staging, destination)
        _verify_evaluation_publication(
            destination,
            seal=seal,
            raw=raw_snapshot,
            source_identity=identity,
            item_id=str(item_id),
            target_column=selected_target,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ResearchEvaluationArtifacts(
        output_directory=destination,
        predictions_path=destination / EVALUATION_PREDICTIONS_NAME,
        metrics_path=destination / EVALUATION_METRICS_NAME,
        daily_path=destination / EVALUATION_DAILY_NAME,
        report_path=destination / EVALUATION_REPORT_NAME,
        manifest_path=destination / EVALUATION_MANIFEST_NAME,
        metrics=metrics,
    )


__all__ = [
    "CORRECTOR_KIND",
    "CORRECTOR_NAME",
    "EVALUATION_DIRECTORY_NAME",
    "EVALUATION_KIND",
    "FIT_KIND",
    "FIT_MANIFEST_NAME",
    "FIT_PROTOCOL",
    "ResearchEvaluationArtifacts",
    "ResearchFitArtifacts",
    "ResearchValidationCorrectorError",
    "default_research_directory",
    "evaluate_research_validation_corrector",
    "fit_research_validation_corrector",
]
