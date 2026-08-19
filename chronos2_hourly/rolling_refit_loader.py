"""Strict filesystem boundary for causal rolling-refit blocks.

The loader intentionally does not infer evidence from legacy run folders.
Every accepted source is an immutable ``rolling-refit-block/v1`` bundle whose
checksum manifest digest is supplied by the caller as an external trust
anchor.  The bundle contains one self-contained row artifact with raw Chronos
quantiles, realised targets, declared PIT/deterministic features and per-hour
causal timestamp evidence.

Legacy live archives currently expose only aggregate revision ranges.  Those
ranges cannot prove which vintage was used for each delivery hour, so their
absence is a hard error rather than a reason to broadcast an invented
timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.rolling_refit import (
    FORBIDDEN_TOKENS,
    QUANTILES,
    RollingRefitBlock,
    RollingRefitContractError,
    RollingRefitPolicy,
    RollingRefitSelection,
    select_rolling_refit_window,
)


BLOCK_SCHEMA_VERSION = "rolling-refit-block/v1"
BLOCK_MANIFEST_NAME = "rolling_refit_block_manifest.json"
ROWS_FILE_NAME = "rolling_refit_training_rows.csv.gz"
CHECKSUM_MANIFEST_NAME = "artifact_checksums.json"
BLOCK_MANIFEST_ROLE = "rolling_refit_block_manifest"
ROWS_ROLE = "rolling_refit_training_rows"
FEATURE_PROVENANCE_KINDS = frozenset(
    {"pit_asof", "deterministic_calendar"}
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_ROW_COLUMNS = (
    "delivery_start_utc",
    "actual",
    "q10",
    "q50",
    "q90",
    "forecast_origin_utc",
    "maximum_snapshot_time_utc",
    "maximum_revision_time_utc",
    "pit_inputs_present",
)
_RUN_CONTRACTS = {
    "sealed_oof": ("sealed_oof", "sealed_oof"),
    "pit_replay": ("pit_replay", "pit_reconstruction"),
    "issued_live": ("live_day_ahead", "issued_live"),
}


class RollingRefitFilesystemError(RollingRefitContractError):
    """Raised when an on-disk block cannot prove the strict contract."""


@dataclass(frozen=True)
class FilesystemBlockSpec:
    """One trusted immutable block location.

    ``artifact_checksums_sha256`` is deliberately supplied outside the block.
    It anchors the manifest which, in turn, hashes the proof and every file
    consumed by this loader.
    """

    directory: str | Path
    source_kind: str
    artifact_checksums_sha256: str

    def __post_init__(self) -> None:
        directory = Path(self.directory)
        if not str(directory):
            raise RollingRefitFilesystemError("block directory must be explicit")
        object.__setattr__(self, "directory", directory)
        kind = str(self.source_kind).strip()
        if kind not in _RUN_CONTRACTS:
            raise RollingRefitFilesystemError(
                f"unsupported filesystem source_kind={self.source_kind!r}"
            )
        object.__setattr__(self, "source_kind", kind)
        digest = _normalise_sha256(
            self.artifact_checksums_sha256,
            name="artifact_checksums_sha256",
        )
        object.__setattr__(self, "artifact_checksums_sha256", digest)


@dataclass(frozen=True)
class FilesystemRollingRefitResult:
    """Loaded blocks, exact core selection and filesystem audit."""

    blocks: tuple[RollingRefitBlock, ...]
    selection: RollingRefitSelection
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _LoadedBlock:
    block: RollingRefitBlock
    source_kind: str
    source_id: str
    bootstrap_replay: bool
    audit: Mapping[str, Any]


def _normalise_sha256(value: Any, *, name: str) -> str:
    result = str(value).strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise RollingRefitFilesystemError(
            f"{name} must be one explicit SHA-256 digest"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RollingRefitFilesystemError(
            f"cannot read valid {label} JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RollingRefitFilesystemError(f"{label} must be one JSON object")
    return value


def _inside(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RollingRefitFilesystemError(
            f"{label} must resolve to a regular file inside block directory"
        ) from exc
    if not resolved.is_file():
        raise RollingRefitFilesystemError(f"{label} is not a regular file")
    return resolved


def _require_exact_bool(manifest: Mapping[str, Any], field: str, expected: bool) -> None:
    value = manifest.get(field)
    if not isinstance(value, bool) or value is not expected:
        raise RollingRefitFilesystemError(
            f"{field} must be the explicit JSON boolean {str(expected).lower()}"
        )


def _require_text(manifest: Mapping[str, Any], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RollingRefitFilesystemError(f"{field} must be explicit text")
    return value.strip()


def _contains_forbidden(value: Any) -> bool:
    folded = str(value).casefold()
    return any(token in folded for token in FORBIDDEN_TOKENS)


def _parse_aware_timestamp_column(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for value in values:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            if not allow_missing:
                raise RollingRefitFilesystemError(
                    f"{name} contains missing timestamp evidence"
                )
            converted.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise RollingRefitFilesystemError(
                f"{name} contains an invalid timestamp"
            ) from exc
        if timestamp.tzinfo is None:
            raise RollingRefitFilesystemError(
                f"{name} timestamps must carry an explicit timezone"
            )
        converted.append(timestamp.tz_convert("UTC"))
    return pd.Series(converted, index=values.index, name=name)


def _parse_explicit_bool_column(values: pd.Series, *, name: str) -> pd.Series:
    converted: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
            continue
        if isinstance(value, str) and value in {"True", "False", "true", "false"}:
            converted.append(value.casefold() == "true")
            continue
        raise RollingRefitFilesystemError(
            f"{name} must contain only explicit true/false values"
        )
    return pd.Series(converted, index=values.index, name=name, dtype=bool)


def _civil_date(
    value: date | str | pd.Timestamp,
    *,
    timezone: str,
    name: str,
) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RollingRefitFilesystemError(f"{name} must be one civil date") from exc
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise RollingRefitFilesystemError(f"{name} must be one civil date")
    return timestamp.date()


def _checksum_entries(
    root: Path,
    checksum_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if checksum_manifest.get("algorithm") != "sha256":
        raise RollingRefitFilesystemError(
            "artifact_checksums.json algorithm must be exactly sha256"
        )
    raw_entries = checksum_manifest.get("artifacts")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RollingRefitFilesystemError(
            "artifact_checksums.json must contain a non-empty artifacts list"
        )
    result: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise RollingRefitFilesystemError(
                f"artifact checksum entry {position} must be an object"
            )
        role = raw.get("role")
        if not isinstance(role, str) or not role.strip():
            raise RollingRefitFilesystemError(
                f"artifact checksum entry {position} has no explicit role"
            )
        role = role.strip()
        if role in result:
            raise RollingRefitFilesystemError(
                f"artifact checksum role must be unique: {role}"
            )
        path = _inside(root, raw.get("path"), label=f"artifact role {role}")
        expected_sha = _normalise_sha256(
            raw.get("sha256"),
            name=f"artifact role {role} sha256",
        )
        expected_size = raw.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise RollingRefitFilesystemError(
                f"artifact role {role} size_bytes must be an integer"
            )
        result[role] = {
            "path": path,
            "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    return result


def _verify_artifact(entry: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    path = Path(entry["path"])
    observed_size = path.stat().st_size
    if observed_size != int(entry["size_bytes"]):
        raise RollingRefitFilesystemError(
            f"artifact role {role} size mismatch: expected={entry['size_bytes']}, "
            f"observed={observed_size}"
        )
    observed_sha = _sha256_file(path)
    if observed_sha != str(entry["sha256"]):
        raise RollingRefitFilesystemError(
            f"artifact role {role} checksum mismatch"
        )
    return {
        "role": role,
        "path": str(path),
        "size_bytes": observed_size,
        "sha256": observed_sha,
    }


def _validate_manifest_metadata(
    manifest: Mapping[str, Any],
    *,
    spec: FilesystemBlockSpec,
    zone: str,
    delivery_timezone: str,
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
) -> tuple[str, bool, list[str], dict[str, str], str]:
    if manifest.get("schema_version") != BLOCK_SCHEMA_VERSION:
        raise RollingRefitFilesystemError(
            f"schema_version must be exactly {BLOCK_SCHEMA_VERSION}"
        )
    if manifest.get("status") != "sealed":
        raise RollingRefitFilesystemError("block status must be exactly sealed")
    if manifest.get("source_kind") != spec.source_kind:
        raise RollingRefitFilesystemError("source_kind disagrees with trusted spec")
    expected_run_type, expected_status = _RUN_CONTRACTS[spec.source_kind]
    if manifest.get("run_type") != expected_run_type:
        raise RollingRefitFilesystemError(
            f"{spec.source_kind} run_type must be {expected_run_type}"
        )
    if manifest.get("forecast_status") != expected_status:
        raise RollingRefitFilesystemError(
            f"{spec.source_kind} forecast_status must be {expected_status}"
        )
    if str(manifest.get("zone", "")).upper() != zone:
        raise RollingRefitFilesystemError("block zone does not match requested zone")
    if manifest.get("timezone") != delivery_timezone:
        raise RollingRefitFilesystemError(
            "block timezone does not match requested delivery timezone"
        )
    if _normalise_sha256(
        manifest.get("config_sha256"), name="config_sha256"
    ) != expected_config_sha256:
        raise RollingRefitFilesystemError("block config_sha256 mismatch")
    if _normalise_sha256(
        manifest.get("base_bundle_sha256"), name="base_bundle_sha256"
    ) != expected_base_bundle_sha256:
        raise RollingRefitFilesystemError("block base_bundle_sha256 mismatch")
    _require_exact_bool(manifest, "storm_used_as_feature", False)
    _require_exact_bool(manifest, "mkonline_used_as_feature", False)
    _require_exact_bool(manifest, "raw_chronos_only", True)
    if manifest.get("chronos_artifact_kind") != "raw_pre_residual_quantiles":
        raise RollingRefitFilesystemError(
            "chronos_artifact_kind must be raw_pre_residual_quantiles"
        )
    if manifest.get("prediction_mode") != "autonomous_only":
        raise RollingRefitFilesystemError(
            "prediction_mode must be exactly autonomous_only"
        )
    source_id = _require_text(manifest, "source_id")
    if _contains_forbidden(source_id) or _contains_forbidden(spec.directory):
        raise RollingRefitFilesystemError(
            "source path/id contains forbidden Storm or MKOnline token"
        )
    bootstrap = manifest.get("bootstrap_replay")
    if not isinstance(bootstrap, bool):
        raise RollingRefitFilesystemError(
            "bootstrap_replay must be an explicit JSON boolean"
        )
    if spec.source_kind != "pit_replay" and bootstrap:
        raise RollingRefitFilesystemError(
            "bootstrap_replay may be true only for pit_replay blocks"
        )

    raw_features = manifest.get("feature_columns")
    if (
        not isinstance(raw_features, list)
        or not raw_features
        or any(not isinstance(value, str) or not value.strip() for value in raw_features)
    ):
        raise RollingRefitFilesystemError(
            "feature_columns must be a non-empty list of explicit names"
        )
    feature_columns = [value.strip() for value in raw_features]
    if len(feature_columns) != len(set(feature_columns)):
        raise RollingRefitFilesystemError("feature_columns contains duplicates")
    forbidden = [value for value in feature_columns if _contains_forbidden(value)]
    if forbidden:
        raise RollingRefitFilesystemError(
            f"feature_columns contains forbidden forecasts: {forbidden}"
        )
    raw_provenance = manifest.get("feature_provenance")
    if not isinstance(raw_provenance, dict) or set(raw_provenance) != set(
        feature_columns
    ):
        raise RollingRefitFilesystemError(
            "feature_provenance must map every and only declared feature"
        )
    provenance = {str(key): str(value) for key, value in raw_provenance.items()}
    invalid = {
        key: value
        for key, value in provenance.items()
        if value not in FEATURE_PROVENANCE_KINDS
    }
    if invalid:
        raise RollingRefitFilesystemError(
            f"unsupported feature provenance: {invalid}"
        )
    if not any(value == "pit_asof" for value in provenance.values()):
        raise RollingRefitFilesystemError(
            "at least one feature must carry pit_asof provenance"
        )

    evidence = manifest.get("timestamp_evidence")
    expected_evidence = {
        "granularity": "per_delivery_hour",
        "forecast_origin_column": "forecast_origin_utc",
        "maximum_snapshot_time_column": "maximum_snapshot_time_utc",
        "maximum_revision_time_column": "maximum_revision_time_utc",
        "pit_inputs_present_column": "pit_inputs_present",
        "source": "materialized_pit_selection",
    }
    if evidence != expected_evidence:
        raise RollingRefitFilesystemError(
            "timestamp_evidence must declare exact per-hour materialized PIT proof"
        )
    if manifest.get("rows_role") != ROWS_ROLE:
        raise RollingRefitFilesystemError(f"rows_role must be exactly {ROWS_ROLE}")
    causal_timestamp_scope = manifest.get("causal_timestamp_scope")
    if causal_timestamp_scope != "delivery_day_all_pit_inputs":
        raise RollingRefitFilesystemError(
            "causal_timestamp_scope must be exactly "
            "delivery_day_all_pit_inputs for v1"
        )
    return (
        source_id,
        bootstrap,
        feature_columns,
        provenance,
        causal_timestamp_scope,
    )


def _load_rows(
    path: Path,
    *,
    source_kind: str,
    manifest: Mapping[str, Any],
    feature_columns: Sequence[str],
    feature_provenance: Mapping[str, str],
    delivery_timezone: str,
    causal_timestamp_scope: str,
) -> RollingRefitBlock:
    try:
        rows = pd.read_csv(path)
    except Exception as exc:
        raise RollingRefitFilesystemError(
            f"cannot read rolling-refit rows: {path}"
        ) from exc
    expected_columns = list(_REQUIRED_ROW_COLUMNS) + list(feature_columns)
    if list(rows.columns) != expected_columns:
        raise RollingRefitFilesystemError(
            "rolling-refit rows must contain the exact ordered schema: "
            f"{expected_columns}"
        )
    if rows.empty:
        raise RollingRefitFilesystemError("rolling-refit rows must not be empty")
    expected_rows = manifest.get("n_rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise RollingRefitFilesystemError("n_rows must be an integer")
    if len(rows) != expected_rows:
        raise RollingRefitFilesystemError(
            f"n_rows mismatch: manifest={expected_rows}, file={len(rows)}"
        )

    delivery = _parse_aware_timestamp_column(
        rows["delivery_start_utc"],
        name="delivery_start_utc",
        allow_missing=False,
    )
    index = pd.DatetimeIndex(delivery, name="delivery_start_utc")
    origin = _parse_aware_timestamp_column(
        rows["forecast_origin_utc"],
        name="forecast_origin_utc",
        allow_missing=False,
    )
    origin.index = index
    pit_inputs_present = _parse_explicit_bool_column(
        rows["pit_inputs_present"], name="pit_inputs_present"
    )
    pit_inputs_present.index = index
    snapshot = _parse_aware_timestamp_column(
        rows["maximum_snapshot_time_utc"],
        name="maximum_snapshot_time_utc",
        allow_missing=True,
    )
    snapshot.index = index
    revision = _parse_aware_timestamp_column(
        rows["maximum_revision_time_utc"],
        name="maximum_revision_time_utc",
        allow_missing=True,
    )
    revision.index = index

    features = rows[list(feature_columns)].copy()
    converted = features.apply(pd.to_numeric, errors="coerce")
    introduced_missing = converted.isna() & ~features.isna()
    if bool(introduced_missing.to_numpy().any()):
        raise RollingRefitFilesystemError("features contain non-numeric values")
    if bool(np.isinf(converted.to_numpy(dtype=float)).any()):
        raise RollingRefitFilesystemError("features contain infinite values")
    features = converted
    features.index = index
    pit_columns = [
        name for name in feature_columns if feature_provenance[name] == "pit_asof"
    ]
    deterministic_columns = [
        name
        for name in feature_columns
        if feature_provenance[name] == "deterministic_calendar"
    ]
    if deterministic_columns and bool(features[deterministic_columns].isna().any().any()):
        raise RollingRefitFilesystemError(
            "deterministic_calendar features must never be missing"
        )
    declared_pit_presence = pit_inputs_present.to_numpy(dtype=bool)
    timestamp_presence = snapshot.notna().to_numpy(dtype=bool)
    revision_presence = revision.notna().to_numpy(dtype=bool)
    if not np.array_equal(timestamp_presence, declared_pit_presence) or not np.array_equal(
        revision_presence, declared_pit_presence
    ):
        raise RollingRefitFilesystemError(
            "PIT max timestamps are required iff a PIT input is present; "
            "never broadcast or invent origin-epsilon"
        )

    local_days = pd.Index(index.tz_convert(delivery_timezone).date)
    first_day = _require_text(manifest, "delivery_start_day_local")
    last_day = _require_text(manifest, "delivery_end_day_local")
    try:
        first_expected = date.fromisoformat(first_day)
        last_expected = date.fromisoformat(last_day)
    except ValueError as exc:
        raise RollingRefitFilesystemError(
            "delivery day bounds must use ISO civil dates"
        ) from exc
    if local_days.min() != first_expected or local_days.max() != last_expected:
        raise RollingRefitFilesystemError(
            "row delivery-day bounds disagree with block manifest"
        )
    if source_kind != "sealed_oof" and first_expected != last_expected:
        raise RollingRefitFilesystemError(
            "pit_replay/issued_live blocks must contain exactly one local day"
        )
    if causal_timestamp_scope != "delivery_day_all_pit_inputs":
        raise RollingRefitFilesystemError(
            "unsupported causal timestamp scope"
        )
    for day_value in pd.Index(local_days).unique():
        day_selector = np.asarray(local_days == day_value, dtype=bool)
        if not bool(declared_pit_presence[day_selector].all()):
            raise RollingRefitFilesystemError(
                "delivery_day_all_pit_inputs requires pit_inputs_present=true "
                f"on every physical hour of {day_value}"
            )
        if not bool(features.loc[day_selector, pit_columns].notna().any().any()):
            raise RollingRefitFilesystemError(
                f"{day_value} has no real selected PIT input to timestamp"
            )
        if snapshot.loc[day_selector].nunique(dropna=True) != 1:
            raise RollingRefitFilesystemError(
                "maximum_snapshot_time_utc must be the constant real daily "
                f"maximum for {day_value}"
            )
        if revision.loc[day_selector].nunique(dropna=True) != 1:
            raise RollingRefitFilesystemError(
                "maximum_revision_time_utc must be the constant real daily "
                f"maximum for {day_value}"
            )

    target = pd.to_numeric(rows["actual"], errors="coerce").astype(float)
    target.index = index
    target.name = "actual"
    quantiles = rows[list(QUANTILES)].apply(pd.to_numeric, errors="coerce").astype(float)
    quantiles.index = index
    return RollingRefitBlock(
        source_kind=source_kind,
        source_id=_require_text(manifest, "source_id"),
        source_sha256="",  # replaced after anchored checksum verification
        features=features,
        target=target,
        chronos_quantiles=quantiles,
        forecast_origin_utc=origin,
        pit_inputs_present=pit_inputs_present,
        maximum_snapshot_time_utc=snapshot,
        maximum_revision_time_utc=revision,
    )


def _load_one(
    spec: FilesystemBlockSpec,
    *,
    zone: str,
    delivery_timezone: str,
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
) -> _LoadedBlock:
    try:
        root = Path(spec.directory).resolve(strict=True)
    except OSError as exc:
        raise RollingRefitFilesystemError(
            f"block directory does not exist: {spec.directory}"
        ) from exc
    if not root.is_dir():
        raise RollingRefitFilesystemError(f"block path is not a directory: {root}")
    checksum_path = root / CHECKSUM_MANIFEST_NAME
    if not checksum_path.is_file():
        raise RollingRefitFilesystemError(
            f"missing {CHECKSUM_MANIFEST_NAME} in block directory"
        )
    observed_checksum_digest = _sha256_file(checksum_path)
    if observed_checksum_digest != spec.artifact_checksums_sha256:
        raise RollingRefitFilesystemError(
            "artifact_checksums.json does not match its external trust anchor"
        )
    checksum_manifest = _read_json_object(
        checksum_path, label="artifact checksum manifest"
    )
    entries = _checksum_entries(root, checksum_manifest)
    missing_roles = [
        role for role in (BLOCK_MANIFEST_ROLE, ROWS_ROLE) if role not in entries
    ]
    if missing_roles:
        raise RollingRefitFilesystemError(
            "strict per-hour rolling proof is absent; missing artifact roles "
            f"{missing_roles}. Aggregate run_manifest/input_diagnostics ranges "
            "cannot be expanded into per-hour timestamps."
        )
    proof_audit = _verify_artifact(
        entries[BLOCK_MANIFEST_ROLE], role=BLOCK_MANIFEST_ROLE
    )
    manifest = _read_json_object(
        Path(entries[BLOCK_MANIFEST_ROLE]["path"]), label="rolling block manifest"
    )
    (
        source_id,
        bootstrap,
        feature_columns,
        feature_provenance,
        causal_timestamp_scope,
    ) = (
        _validate_manifest_metadata(
            manifest,
            spec=spec,
            zone=zone,
            delivery_timezone=delivery_timezone,
            expected_config_sha256=expected_config_sha256,
            expected_base_bundle_sha256=expected_base_bundle_sha256,
        )
    )
    rows_audit = _verify_artifact(entries[ROWS_ROLE], role=ROWS_ROLE)
    partial = _load_rows(
        Path(entries[ROWS_ROLE]["path"]),
        source_kind=spec.source_kind,
        manifest=manifest,
        feature_columns=feature_columns,
        feature_provenance=feature_provenance,
        delivery_timezone=delivery_timezone,
        causal_timestamp_scope=causal_timestamp_scope,
    )
    block = RollingRefitBlock(
        source_kind=partial.source_kind,
        source_id=partial.source_id,
        source_sha256=spec.artifact_checksums_sha256,
        features=partial.features,
        target=partial.target,
        chronos_quantiles=partial.chronos_quantiles,
        forecast_origin_utc=partial.forecast_origin_utc,
        pit_inputs_present=partial.pit_inputs_present,
        maximum_snapshot_time_utc=partial.maximum_snapshot_time_utc,
        maximum_revision_time_utc=partial.maximum_revision_time_utc,
    )
    audit = {
        "source_kind": spec.source_kind,
        "source_id": source_id,
        "directory": str(root),
        "artifact_checksums_path": str(checksum_path),
        "artifact_checksums_sha256": observed_checksum_digest,
        "bootstrap_replay": bootstrap,
        "verified_artifacts": [proof_audit, rows_audit],
        "feature_provenance": dict(feature_provenance),
        "causal_timestamp_scope": causal_timestamp_scope,
        "rows": int(len(block.features)),
        "pit_input_present_hours": int(block.pit_inputs_present.sum()),
        "pit_input_missing_hours": int((~block.pit_inputs_present).sum()),
    }
    return _LoadedBlock(
        block=block,
        source_kind=spec.source_kind,
        source_id=source_id,
        bootstrap_replay=bootstrap,
        audit=audit,
    )


def load_rolling_refit_filesystem(
    sources: Sequence[FilesystemBlockSpec],
    *,
    zone: str,
    forecast_delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
    rolling_policy: RollingRefitPolicy | None = None,
    max_bootstrap_replay_days: int = 0,
    bootstrap_replay_end_day: date | str | pd.Timestamp | None = None,
) -> FilesystemRollingRefitResult:
    """Load anchored v1 blocks and select the exact causal rolling window."""

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise RollingRefitFilesystemError("sources must be a non-empty sequence")
    if not sources:
        raise RollingRefitFilesystemError("sources must be a non-empty sequence")
    if not all(isinstance(value, FilesystemBlockSpec) for value in sources):
        raise RollingRefitFilesystemError(
            "every source must be a FilesystemBlockSpec"
        )
    requested_zone = str(zone).strip().upper()
    if not requested_zone or _contains_forbidden(requested_zone):
        raise RollingRefitFilesystemError("zone must be explicit and safe")
    if not str(delivery_timezone).strip():
        raise RollingRefitFilesystemError("delivery_timezone must be explicit")
    config_sha = _normalise_sha256(
        expected_config_sha256, name="expected_config_sha256"
    )
    bundle_sha = _normalise_sha256(
        expected_base_bundle_sha256,
        name="expected_base_bundle_sha256",
    )
    if isinstance(max_bootstrap_replay_days, bool) or not isinstance(
        max_bootstrap_replay_days, int
    ):
        raise RollingRefitFilesystemError(
            "max_bootstrap_replay_days must be an integer"
        )
    if max_bootstrap_replay_days < 0:
        raise RollingRefitFilesystemError(
            "max_bootstrap_replay_days must be non-negative"
        )
    forecast_day = _civil_date(
        forecast_delivery_day,
        timezone=delivery_timezone,
        name="forecast_delivery_day",
    )
    bootstrap_end = (
        None
        if bootstrap_replay_end_day is None
        else _civil_date(
            bootstrap_replay_end_day,
            timezone=delivery_timezone,
            name="bootstrap_replay_end_day",
        )
    )
    if bootstrap_end is not None and bootstrap_end >= forecast_day:
        raise RollingRefitFilesystemError(
            "bootstrap_replay_end_day must be strictly before forecast_delivery_day"
        )

    loaded = [
        _load_one(
            spec,
            zone=requested_zone,
            delivery_timezone=delivery_timezone,
            expected_config_sha256=config_sha,
            expected_base_bundle_sha256=bundle_sha,
        )
        for spec in sources
    ]
    source_ids = [item.source_id for item in loaded]
    if len(source_ids) != len(set(source_ids)):
        raise RollingRefitFilesystemError("source_id must be unique across blocks")
    all_bootstrap_days: list[date] = []
    for item in loaded:
        if not item.bootstrap_replay:
            continue
        all_bootstrap_days.extend(
            pd.Index(
                item.block.features.index.tz_convert(delivery_timezone).date
            ).unique()
        )
    if any(day >= forecast_day for day in all_bootstrap_days):
        raise RollingRefitFilesystemError(
            "every bootstrap replay block must be strictly before forecast day"
        )
    if bootstrap_end is None and all_bootstrap_days:
        raise RollingRefitFilesystemError(
            "bootstrap replay blocks are forbidden when bootstrap_replay_end_day "
            "is None"
        )
    if bootstrap_end is not None and any(
        day > bootstrap_end for day in all_bootstrap_days
    ):
        raise RollingRefitFilesystemError(
            "bootstrap replay block exceeds explicit bootstrap suffix end day"
        )
    if len(all_bootstrap_days) != len(set(all_bootstrap_days)):
        raise RollingRefitFilesystemError(
            "bootstrap replay delivery days must be unique across source blocks"
        )
    if bootstrap_end is not None and all_bootstrap_days:
        if max_bootstrap_replay_days == 0:
            raise RollingRefitFilesystemError(
                "bootstrap replay blocks are forbidden when the day bound is zero"
            )
        earliest_allowed = (
            pd.Timestamp(bootstrap_end)
            - pd.Timedelta(days=max_bootstrap_replay_days - 1)
        ).date()
        if any(day < earliest_allowed for day in all_bootstrap_days):
            raise RollingRefitFilesystemError(
                "bootstrap replay block lies outside the explicit bounded suffix: "
                f"allowed={earliest_allowed}..{bootstrap_end}"
            )
    selection = select_rolling_refit_window(
        [item.block for item in loaded],
        forecast_delivery_day=forecast_delivery_day,
        delivery_timezone=delivery_timezone,
        policy=rolling_policy,
    )

    bootstrap_source_ids = {
        item.source_id for item in loaded if item.bootstrap_replay
    }
    local_dates = pd.Index(
        selection.provenance.index.tz_convert(delivery_timezone).date
    )
    bootstrap_selector = selection.provenance["source_id"].isin(
        bootstrap_source_ids
    ).to_numpy()
    bootstrap_days = sorted(set(local_dates[bootstrap_selector]))
    if len(bootstrap_days) > max_bootstrap_replay_days:
        raise RollingRefitFilesystemError(
            "bootstrap PIT replay bound exceeded: "
            f"selected={len(bootstrap_days)}, "
            f"allowed={max_bootstrap_replay_days}"
        )
    if bootstrap_days:
        assert bootstrap_end is not None
        expected_bootstrap_days = [
            (pd.Timestamp(bootstrap_end) - pd.Timedelta(days=offset)).date()
            for offset in range(len(bootstrap_days) - 1, -1, -1)
        ]
        if bootstrap_days != expected_bootstrap_days:
            raise RollingRefitFilesystemError(
                "selected bootstrap replay days must form one contiguous suffix "
                f"ending {bootstrap_end}: observed="
                f"{[str(value) for value in bootstrap_days]}"
            )
    filesystem_digest = hashlib.sha256()
    for digest in sorted(spec.artifact_checksums_sha256 for spec in sources):
        filesystem_digest.update(digest.encode("ascii"))
    audit: dict[str, Any] = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "zone": requested_zone,
        "delivery_timezone": delivery_timezone,
        "forecast_delivery_day": str(forecast_day),
        "expected_config_sha256": config_sha,
        "expected_base_bundle_sha256": bundle_sha,
        "external_checksum_anchors_verified": True,
        "consumed_files_rehashed": True,
        "raw_chronos_only": True,
        "storm_used_as_feature": False,
        "mkonline_used_as_feature": False,
        "bootstrap_replay_days": [str(value) for value in bootstrap_days],
        "bootstrap_replay_days_count": len(bootstrap_days),
        "max_bootstrap_replay_days": max_bootstrap_replay_days,
        "bootstrap_replay_end_day": (
            None if bootstrap_end is None else str(bootstrap_end)
        ),
        "blocks": [dict(item.audit) for item in loaded],
        "filesystem_sources_sha256": filesystem_digest.hexdigest(),
        "selection_training_corpus_sha256": selection.audit[
            "training_corpus_sha256"
        ],
    }
    return FilesystemRollingRefitResult(
        blocks=tuple(item.block for item in loaded),
        selection=selection,
        audit=audit,
    )


def load_blocks(
    sources: Sequence[FilesystemBlockSpec],
    *,
    zone: str,
    forecast_delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
    rolling_policy: RollingRefitPolicy | None = None,
    max_bootstrap_replay_days: int = 0,
    bootstrap_replay_end_day: date | str | pd.Timestamp | None = None,
) -> tuple[RollingRefitBlock, ...]:
    """Compatibility wrapper returning only validated in-memory blocks."""

    return load_rolling_refit_filesystem(
        sources,
        zone=zone,
        forecast_delivery_day=forecast_delivery_day,
        delivery_timezone=delivery_timezone,
        expected_config_sha256=expected_config_sha256,
        expected_base_bundle_sha256=expected_base_bundle_sha256,
        rolling_policy=rolling_policy,
        max_bootstrap_replay_days=max_bootstrap_replay_days,
        bootstrap_replay_end_day=bootstrap_replay_end_day,
    ).blocks


__all__ = [
    "BLOCK_MANIFEST_NAME",
    "BLOCK_MANIFEST_ROLE",
    "BLOCK_SCHEMA_VERSION",
    "CHECKSUM_MANIFEST_NAME",
    "FilesystemBlockSpec",
    "FilesystemRollingRefitResult",
    "ROWS_FILE_NAME",
    "ROWS_ROLE",
    "RollingRefitFilesystemError",
    "load_blocks",
    "load_rolling_refit_filesystem",
]
