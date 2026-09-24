#!/usr/bin/env python
"""Prospective paired A/B comparison for residual-load forecast providers.

The comparison is deliberately prospective: it only scores delivery days for
which both immutable live archives already exist and all realized prices are
available.  The downstream price model/corrector is treated as frozen.  This
isolates the operational effect of replacing Saturn residual-load forecasts
with Chronos-2 forecasts, but it remains a distribution-shift experiment and
must not be interpreted as a retrained model benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from chronos2_hourly.app_service import (
    CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR,
    ExistingForecastArchiveError,
    ZoneStatus,
    load_best_statistics_history,
    validate_existing_forecast_archive,
)
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.residual_load_comparison_report import (
    render_residual_load_comparison_report,
)
from chronos2_hourly.zone_live import canonical_zone, load_zone_registry
from chronos2_modular.common import load_yaml


SUPPORTED_ZONES: tuple[str, ...] = ("FR", "DE", "BE", "NL", "ES")
DEFAULT_REGISTRY = "chronos2_hourly_live_zones.yaml"
SCHEMA_VERSION = 1
FORECAST_Q50_COLUMNS: tuple[str, ...] = ("q50", "candidate_model__q50")
DOWNSTREAM_IDENTITY_KEYS: tuple[str, ...] = (
    "script_version",
    "config",
    "model_id",
    "target_contract",
    "delivery_horizon",
    "n_training_hours",
    "n_chronos_oof_hours",
    "active_features",
    "source_run",
    "residual_recipe",
    "residual_schema",
    "prediction_mode",
    "candidate_model",
    "native_model",
    "baseline_model",
    "recipe_mode",
    "autonomous_weight",
    "mkonline_weight",
    "forecast_origin_timezone",
    "forecast_origin_local_time",
    "forecast_cutoff_utc",
    "frozen_autonomous_manifest_sha256",
    "sealed_benchmark_manifest_sha256",
)
TREATMENT_SOURCE_ROLES = frozenset(
    {
        "residual_load_bundle_manifest",
        "shadow_reporting_source",
    }
)
SOURCE_ROLE_ALIASES = {
    "frozen_recipe": "recipe_manifest",
}
REQUIRED_DOWNSTREAM_SOURCE_ROLES = frozenset(
    {
        "live_config",
        "base_config",
        "recipe_manifest",
        "frozen_autonomous_checksum_manifest",
        "sealed_benchmark_checksum_manifest",
        "source_code",
    }
)
RESIDUAL_LOAD_TREATMENT_FEATURES: tuple[str, ...] = (
    "fr_residual_load_fcst",
    "de_residual_load_fcst",
    "be_residual_load_fcst",
    "nl_residual_load_fcst",
    "es_residual_load_fcst",
)
RESIDUAL_LOAD_TREATMENT_COVARIATES: tuple[str, ...] = (
    *RESIDUAL_LOAD_TREATMENT_FEATURES,
    *(
        f"known_{feature}_oracle"
        for feature in RESIDUAL_LOAD_TREATMENT_FEATURES
    ),
)
MKONLINE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "value",
)


class ComparisonError(ValueError):
    """Raised when a prospective pair cannot be audited safely."""


@dataclass(frozen=True)
class ZoneComparisonSpec:
    code: str
    timezone: str
    project_root: Path
    live_config: Path
    output_root: Path
    forecast_filename: str
    status: ZoneStatus


@dataclass(frozen=True)
class ActualSeries:
    values: pd.Series
    source: str
    path: Path | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ComparisonBuild:
    paired_hourly: pd.DataFrame
    metrics: pd.DataFrame
    manifest: dict[str, Any]


ActualLoader = Callable[[ZoneComparisonSpec], ActualSeries | pd.Series | pd.DataFrame]
ArchiveValidator = Callable[[ZoneComparisonSpec, date, str], Path | None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"JSON absent ou illisible: {path}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"Objet JSON attendu: {path}")
    return payload


def normalize_zones(values: Sequence[str]) -> tuple[str, ...]:
    flattened = [
        item
        for raw in values
        for item in (part.strip() for part in str(raw).split(","))
        if item
    ]
    if not flattened:
        raise ComparisonError("Au moins une zone est obligatoire.")
    result: list[str] = []
    for value in flattened:
        code = canonical_zone(value)
        if code not in SUPPORTED_ZONES:
            raise ComparisonError(
                f"{code}: comparaison disponible uniquement pour "
                f"{', '.join(SUPPORTED_ZONES)}."
            )
        if code in result:
            raise ComparisonError(f"Zone dupliquée: {code}")
        result.append(code)
    return tuple(result)


def _parse_day(value: str | date, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ComparisonError(f"{name} doit respecter YYYY-MM-DD.") from exc


def load_zone_specs(
    registry_path: str | Path,
    *,
    zones: Sequence[str],
) -> tuple[ZoneComparisonSpec, ...]:
    registry_file = Path(registry_path).expanduser().resolve()
    registry, project_root = load_zone_registry(registry_file)
    raw_zones = registry.get("zones")
    if not isinstance(raw_zones, Mapping):
        raise ComparisonError(f"{registry_file}: mapping zones absent.")
    specs: list[ZoneComparisonSpec] = []
    for code in normalize_zones(zones):
        raw = raw_zones.get(code)
        if not isinstance(raw, Mapping):
            raise ComparisonError(f"{code}: zone absente du registre.")
        raw_live_config = raw.get("live_config")
        if raw_live_config in (None, ""):
            raise ComparisonError(f"{code}: live_config absent du registre.")
        live_config = Path(str(raw_live_config)).expanduser()
        if not live_config.is_absolute():
            live_config = project_root / live_config
        live_config = live_config.resolve()
        try:
            live_config.relative_to(project_root)
        except ValueError as exc:
            raise ComparisonError(f"{code}: live_config hors projet.") from exc
        payload = load_yaml(live_config)
        live = payload.get("live") if isinstance(payload, Mapping) else None
        if not isinstance(live, Mapping):
            raise ComparisonError(f"{code}: mapping live absent de {live_config}.")
        raw_output_root = live.get("output_root")
        if raw_output_root in (None, ""):
            raise ComparisonError(f"{code}: live.output_root absent.")
        output_root = Path(str(raw_output_root)).expanduser()
        if not output_root.is_absolute():
            output_root = live_config.parent / output_root
        output_root = output_root.resolve()
        try:
            output_root.relative_to(project_root)
        except ValueError as exc:
            raise ComparisonError(f"{code}: output_root hors projet.") from exc
        forecast_filename = str(
            live.get("forecast_filename")
            or f"forecast_hourly_{code.lower()}.csv"
        ).strip()
        if not forecast_filename or Path(forecast_filename).name != forecast_filename:
            raise ComparisonError(f"{code}: forecast_filename invalide.")
        timezone_name = str(raw.get("delivery_timezone") or "").strip()
        if not timezone_name:
            raise ComparisonError(f"{code}: delivery_timezone absente.")
        status = ZoneStatus(
            code=code,
            timezone=timezone_name,
            enabled=bool(raw.get("enabled", False)),
            production_ready=bool(raw.get("production_ready", False)),
            ready=True,
            runner=None,
            live_config=live_config,
            checks=(),
            blockers=(),
        )
        specs.append(
            ZoneComparisonSpec(
                code=code,
                timezone=timezone_name,
                project_root=project_root,
                live_config=live_config,
                output_root=output_root,
                forecast_filename=forecast_filename,
                status=status,
            )
        )
    return tuple(specs)


def _scan_archive_days(
    spec: ZoneComparisonSpec,
    *,
    start: date,
    end: date,
) -> tuple[dict[date, Path], dict[date, Path]]:
    production: dict[date, Path] = {}
    challenger: dict[date, Path] = {}
    if not spec.output_root.is_dir():
        return production, challenger
    prefix = re.escape(spec.code.lower())
    production_pattern = re.compile(
        rf"^{prefix}_day_ahead_(\d{{4}}-\d{{2}}-\d{{2}})$"
    )
    challenger_pattern = re.compile(
        rf"^{prefix}_day_ahead_(\d{{4}}-\d{{2}}-\d{{2}})"
        r"_residual_load_chronos2$"
    )
    challenger_root = spec.output_root / CHRONOS2_RESIDUAL_ARCHIVE_SUBDIR
    scan_roots = (
        (spec.output_root, production_pattern, production),
        (challenger_root, challenger_pattern, challenger),
    )
    canonical_output_root = spec.output_root.resolve()
    for archive_root, pattern, destination in scan_roots:
        if not archive_root.is_dir():
            continue
        resolved_root = archive_root.resolve()
        try:
            resolved_root.relative_to(canonical_output_root)
        except ValueError as exc:
            raise ComparisonError(
                f"{spec.code}: racine d'archive hors output_root: {archive_root}"
            ) from exc
        for raw_path in archive_root.iterdir():
            if not raw_path.is_dir():
                continue
            resolved = raw_path.resolve()
            if resolved.parent != resolved_root:
                raise ComparisonError(
                    f"{spec.code}: archive hors racine canonique: {raw_path}"
                )
            match = pattern.fullmatch(raw_path.name)
            if match is None:
                continue
            try:
                delivery_day = date.fromisoformat(match.group(1))
            except ValueError as exc:
                raise ComparisonError(f"Date d'archive invalide: {raw_path}") from exc
            if start <= delivery_day <= end:
                destination[delivery_day] = resolved
    return production, challenger


def _default_archive_validator(
    spec: ZoneComparisonSpec,
    delivery_day: date,
    source: str,
) -> Path | None:
    try:
        return validate_existing_forecast_archive(
            spec.status,
            project_root=spec.project_root,
            delivery_day=delivery_day,
            residual_load_source=source,
        )
    except ExistingForecastArchiveError as exc:
        raise ComparisonError(str(exc)) from exc


def _require_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    spec: ZoneComparisonSpec,
    delivery_day: date,
    source: str,
    path: Path,
) -> None:
    expected_common = {
        "zone": spec.code,
        "timezone": spec.timezone,
        "delivery_day_local": delivery_day.isoformat(),
    }
    for key, expected in expected_common.items():
        if manifest.get(key) != expected:
            raise ComparisonError(
                f"{path}: {key}={manifest.get(key)!r}, attendu {expected!r}."
            )
    observed_source = str(manifest.get("residual_load_source", "saturn")).lower()
    if source == "saturn":
        if observed_source != "saturn":
            raise ComparisonError(
                f"{path}: la production doit déclarer Saturn ou omettre la source."
            )
        expected_run_type = "live_day_ahead"
        expected_status = "issued_live"
    else:
        if observed_source != "chronos2":
            raise ComparisonError(
                f"{path}: le challenger doit déclarer residual_load_source=chronos2."
            )
        if manifest.get("production_eligible") is not False:
            raise ComparisonError(
                f"{path}: le challenger shadow doit déclarer production_eligible=false."
            )
        expected_run_type = "shadow_live_day_ahead"
        expected_status = "shadow_challenger"
    if manifest.get("run_type") != expected_run_type:
        raise ComparisonError(
            f"{path}: run_type={manifest.get('run_type')!r}, "
            f"attendu {expected_run_type!r}."
        )
    if manifest.get("forecast_status") != expected_status:
        raise ComparisonError(
            f"{path}: forecast_status={manifest.get('forecast_status')!r}, "
            f"attendu {expected_status!r}."
        )


def _require_frozen_downstream(
    production: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    production_path: Path,
    challenger_path: Path,
) -> None:
    divergent: list[str] = []
    for key in DOWNSTREAM_IDENTITY_KEYS:
        if key in production or key in challenger:
            if production.get(key) != challenger.get(key):
                divergent.append(key)
    if divergent:
        raise ComparisonError(
            "Downstream non gelé entre la production et le challenger "
            f"({production_path}, {challenger_path}); champs divergents: {divergent}."
        )


def _downstream_source_identity(archive: Path) -> list[dict[str, Any]]:
    """Load the checksum-sealed, non-treatment forecast source identity."""

    checksum_path = archive / "artifact_checksums.json"
    payload = _read_json_object(checksum_path)
    if payload.get("algorithm") != "sha256":
        raise ComparisonError(
            f"{checksum_path}: algorithm doit être sha256."
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ComparisonError(
            f"{checksum_path}: artifacts doit être une liste."
        )
    identity: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise ComparisonError(
                f"{checksum_path}: déclaration checksum invalide."
            )
        raw_role = str(raw.get("role") or "").strip()
        if (
            raw_role == "run_artifact"
            or raw_role in TREATMENT_SOURCE_ROLES
            or raw_role.startswith("residual_load_")
        ):
            continue
        if not raw_role:
            raise ComparisonError(
                f"{checksum_path}: rôle source absent."
            )
        role = SOURCE_ROLE_ALIASES.get(raw_role, raw_role)
        source_path = str(raw.get("path") or "").strip().replace("\\", "/")
        if not source_path:
            raise ComparisonError(
                f"{checksum_path}: chemin source absent pour {raw_role}."
            )
        sha256 = str(raw.get("sha256") or "").strip().lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ComparisonError(
                f"{checksum_path}: SHA-256 invalide pour {raw_role}."
            )
        try:
            size_bytes = int(raw.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ComparisonError(
                f"{checksum_path}: taille invalide pour {raw_role}."
            ) from exc
        if size_bytes < 0:
            raise ComparisonError(
                f"{checksum_path}: taille négative pour {raw_role}."
            )
        key = (role, source_path)
        if key in identity:
            raise ComparisonError(
                f"{checksum_path}: source dupliquée pour {role}: {source_path}."
            )
        identity[key] = {
            "role": role,
            "path": source_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
    observed_roles = {item["role"] for item in identity.values()}
    missing = sorted(REQUIRED_DOWNSTREAM_SOURCE_ROLES - observed_roles)
    if missing:
        raise ComparisonError(
            f"{checksum_path}: sources downstream obligatoires absentes: {missing}."
        )
    return sorted(
        identity.values(),
        key=lambda item: (str(item["role"]), str(item["path"])),
    )


def _require_frozen_downstream_sources(
    production_archive: Path,
    challenger_archive: Path,
) -> dict[str, Any]:
    production = _downstream_source_identity(production_archive)
    challenger = _downstream_source_identity(challenger_archive)
    if production != challenger:
        production_map = {
            (str(item["role"]), str(item["path"])): item
            for item in production
        }
        challenger_map = {
            (str(item["role"]), str(item["path"])): item
            for item in challenger
        }
        production_keys = set(production_map)
        challenger_keys = set(challenger_map)
        missing_in_challenger = sorted(production_keys - challenger_keys)
        extra_in_challenger = sorted(challenger_keys - production_keys)
        divergent_sha = sorted(
            key
            for key in production_keys & challenger_keys
            if production_map[key] != challenger_map[key]
        )
        raise ComparisonError(
            "Sources downstream non gelées entre production et challenger; "
            f"absentes_challenger={missing_in_challenger}, "
            f"supplémentaires_challenger={extra_in_challenger}, "
            f"SHA_ou_taille_divergents={divergent_sha}."
        )
    canonical = json.dumps(
        production,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "n_sources": len(production),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _read_timestamped_input_csv(
    archive: Path,
    filename: str,
    *,
    required_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    path = archive / "inputs" / filename
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        raise ComparisonError(f"{path}: inputs alignés absents ou illisibles.") from exc
    timestamp_column = next(
        (
            column
            for column in ("timestamp", "delivery_start_utc")
            if column in frame
        ),
        None,
    )
    if timestamp_column is None:
        raise ComparisonError(f"{path}: timeline des inputs absente.")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{path}: timeline des inputs invalide.") from exc
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ComparisonError(f"{path}: timeline des inputs dupliquée ou non triée.")
    missing_features = sorted(set(required_features).difference(frame.columns))
    if missing_features:
        raise ComparisonError(
            f"{path}: features obligatoires absentes: {missing_features}."
        )
    normalized = frame.copy()
    normalized[timestamp_column] = index
    return normalized, index


def _require_paired_input_parity(
    production_archive: Path,
    challenger_archive: Path,
    *,
    delivery_index: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Prove that only the five delivery-day residual-load cells may differ."""

    production_aligned, production_aligned_index = _read_timestamped_input_csv(
        production_archive,
        "aligned_inputs.csv.gz",
        required_features=RESIDUAL_LOAD_TREATMENT_FEATURES,
    )
    challenger_aligned, challenger_aligned_index = _read_timestamped_input_csv(
        challenger_archive,
        "aligned_inputs.csv.gz",
        required_features=RESIDUAL_LOAD_TREATMENT_FEATURES,
    )
    if list(production_aligned.columns) != list(challenger_aligned.columns):
        raise ComparisonError(
            "Features downstream différentes entre production et challenger: "
            f"production={list(production_aligned.columns)}, "
            f"challenger={list(challenger_aligned.columns)}."
        )
    if not production_aligned_index.equals(challenger_aligned_index):
        raise ComparisonError(
            "Timelines des inputs différentes entre production et challenger."
        )
    if not production_aligned.equals(challenger_aligned):
        divergent_aligned = [
            column
            for column in production_aligned.columns
            if not production_aligned[column].equals(challenger_aligned[column])
        ]
        raise ComparisonError(
            "Inputs hors traitement non identiques entre production et "
            f"challenger: {divergent_aligned}."
        )

    production, production_index = _read_timestamped_input_csv(
        production_archive,
        "model_covariates_with_future.csv.gz",
        required_features=RESIDUAL_LOAD_TREATMENT_COVARIATES,
    )
    challenger, challenger_index = _read_timestamped_input_csv(
        challenger_archive,
        "model_covariates_with_future.csv.gz",
        required_features=RESIDUAL_LOAD_TREATMENT_COVARIATES,
    )
    if list(production.columns) != list(challenger.columns):
        raise ComparisonError(
            "Features futures différentes entre production et challenger: "
            f"production={list(production.columns)}, "
            f"challenger={list(challenger.columns)}."
        )
    if not production_index.equals(challenger_index):
        raise ComparisonError(
            "Timelines des covariates futures différentes entre les deux runs."
        )
    delivery_mask = production_index.isin(delivery_index)
    if int(delivery_mask.sum()) != len(delivery_index):
        raise ComparisonError(
            "Les inputs alignés ne couvrent pas exactement les heures J+1."
        )
    historical_mask = production_index < delivery_index[0]
    if not bool(historical_mask.any()):
        raise ComparisonError("Le contexte historique des inputs est vide.")
    divergent_columns: list[str] = []
    for column in production.columns:
        if column in RESIDUAL_LOAD_TREATMENT_COVARIATES:
            left = production.loc[historical_mask, column].reset_index(drop=True)
            right = challenger.loc[historical_mask, column].reset_index(drop=True)
        else:
            left = production[column].reset_index(drop=True)
            right = challenger[column].reset_index(drop=True)
        if not left.equals(right):
            divergent_columns.append(column)
    if divergent_columns:
        raise ComparisonError(
            "Inputs hors traitement non identiques entre production et "
            f"challenger: {divergent_columns}."
        )
    for label, frame in (("production", production), ("challenger", challenger)):
        delivery_values = frame.loc[
            delivery_mask,
            list(RESIDUAL_LOAD_TREATMENT_COVARIATES),
        ].apply(pd.to_numeric, errors="coerce")
        if not bool(np.isfinite(delivery_values.to_numpy(float)).all()):
            raise ComparisonError(
                f"{label}: residual_load J+1 incomplet ou non fini."
            )

    protected_covariates = production.copy()
    protected_covariates.loc[
        delivery_mask,
        list(RESIDUAL_LOAD_TREATMENT_COVARIATES),
    ] = np.nan
    protected_bytes = (
        production_aligned.to_csv(index=False)
        + "\n--MODEL-COVARIATES--\n"
        + protected_covariates.to_csv(index=False)
    ).encode("utf-8")

    production_primary = (
        production_archive / "inputs" / "mkonline_primary_live.parquet"
    )
    challenger_primary = (
        challenger_archive / "inputs" / "mkonline_primary_live.parquet"
    )
    if production_primary.is_file() != challenger_primary.is_file():
        raise ComparisonError(
            "MKOnline primary n'est pas présent des deux côtés de la paire."
        )
    primary_sha: str | None = None
    if production_primary.is_file():
        try:
            production_primary_frame = pd.read_parquet(production_primary)
            challenger_primary_frame = pd.read_parquet(challenger_primary)
            missing_primary = sorted(
                set(MKONLINE_PREDICTION_COLUMNS).difference(
                    production_primary_frame.columns
                )
                | set(MKONLINE_PREDICTION_COLUMNS).difference(
                    challenger_primary_frame.columns
                )
            )
            if missing_primary:
                raise ValueError(
                    f"colonnes MKOnline absentes: {missing_primary}"
                )
            production_primary_frame = production_primary_frame.loc[
                :, MKONLINE_PREDICTION_COLUMNS
            ].copy()
            challenger_primary_frame = challenger_primary_frame.loc[
                :, MKONLINE_PREDICTION_COLUMNS
            ].copy()
            pd.testing.assert_frame_equal(
                production_primary_frame,
                challenger_primary_frame,
                check_dtype=False,
                check_like=False,
            )
        except (AssertionError, OSError, ValueError) as exc:
            raise ComparisonError(
                "MKOnline primary diffère entre production et challenger."
            ) from exc
        primary_sha = hashlib.sha256(
            production_primary_frame.to_csv(index=False).encode("utf-8")
        ).hexdigest()
    return {
        "n_aligned_input_rows": len(production_aligned),
        "n_future_covariate_rows": len(production),
        "n_historical_input_rows": int(historical_mask.sum()),
        "protected_input_identity_sha256": hashlib.sha256(
            protected_bytes
        ).hexdigest(),
        "mkonline_primary_present": production_primary.is_file(),
        "mkonline_primary_semantic_sha256": primary_sha,
    }


def _forecast_series(
    path: Path,
    *,
    spec: ZoneComparisonSpec,
    delivery_day: date,
) -> tuple[pd.Series, str]:
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        raise ComparisonError(f"Forecast absent ou illisible: {path}") from exc
    if "delivery_start_utc" not in frame.columns:
        raise ComparisonError(f"{path}: delivery_start_utc absente.")
    q50_column = next(
        (column for column in FORECAST_Q50_COLUMNS if column in frame.columns),
        None,
    )
    if q50_column is None:
        raise ComparisonError(
            f"{path}: aucune courbe finale parmi {FORECAST_Q50_COLUMNS}."
        )
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{path}: timeline UTC invalide.") from exc
    if index.has_duplicates:
        raise ComparisonError(f"{path}: timeline dupliquée.")
    values = pd.to_numeric(frame[q50_column], errors="coerce").to_numpy(float)
    if not bool(np.isfinite(values).all()):
        raise ComparisonError(f"{path}: {q50_column} contient des valeurs non finies.")
    series = pd.Series(values, index=index, name=q50_column).sort_index()
    expected = local_delivery_day_index(delivery_day, timezone=spec.timezone)
    if len(series) not in {23, 24, 25} or not series.index.equals(expected):
        raise ComparisonError(
            f"{path}: timeline différente de la journée locale canonique "
            f"{delivery_day} ({len(expected)} h)."
        )
    return series, q50_column


def load_audited_statistics_actuals(spec: ZoneComparisonSpec) -> ActualSeries:
    """Load canonical prices from the freshest checksum-sealed Statistics file."""

    try:
        artifact, dataset = load_best_statistics_history(
            spec.status,
            project_root=spec.project_root,
            variant="production",
        )
    except (ExistingForecastArchiveError, OSError, ValueError) as exc:
        raise ComparisonError(
            f"{spec.code}: actuals Statistics audités indisponibles: {exc}"
        ) from exc
    frame = dataset.frame
    if not {"timestamp", "actual"}.issubset(frame.columns):
        raise ComparisonError(
            f"{spec.code}: colonnes timestamp/actual absentes des Statistics."
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True, errors="raise"))
    values = pd.Series(
        pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float),
        index=index,
        name="actual",
    )
    return ActualSeries(
        values=values,
        source="audited_statistics_history",
        path=artifact.statistics_path,
        sha256=artifact.statistics_sha256,
    )


def _normalize_actuals(
    raw: ActualSeries | pd.Series | pd.DataFrame,
    *,
    zone: str,
) -> ActualSeries:
    if isinstance(raw, ActualSeries):
        actuals = raw
    elif isinstance(raw, pd.Series):
        actuals = ActualSeries(raw, source="injected_actual_loader")
    elif isinstance(raw, pd.DataFrame):
        timestamp_column = next(
            (name for name in ("delivery_start_utc", "timestamp") if name in raw),
            None,
        )
        if timestamp_column is None or "actual" not in raw:
            raise ComparisonError(
                f"{zone}: actual_loader DataFrame exige timestamp et actual."
            )
        index = pd.DatetimeIndex(
            pd.to_datetime(raw[timestamp_column], utc=True, errors="raise")
        )
        actuals = ActualSeries(
            pd.Series(raw["actual"].to_numpy(), index=index, name="actual"),
            source="injected_actual_loader",
        )
    else:
        raise ComparisonError(f"{zone}: type d'actuals non supporté: {type(raw)!r}.")
    series = actuals.values
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ComparisonError(f"{zone}: actuals doit être une Series temporelle.")
    if series.index.tz is None:
        raise ComparisonError(f"{zone}: timeline actuals doit être timezone-aware.")
    if series.index.has_duplicates:
        raise ComparisonError(f"{zone}: timeline actuals dupliquée.")
    normalized = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(float),
        index=series.index.tz_convert("UTC"),
        name="actual",
    ).sort_index()
    return ActualSeries(
        values=normalized,
        source=str(actuals.source),
        path=actuals.path,
        sha256=actuals.sha256,
    )


def _metric_row(frame: pd.DataFrame, *, zone: str, scope: str) -> dict[str, Any]:
    n_hours = int(len(frame))
    n_days = int(frame["delivery_day_local"].nunique()) if n_hours else 0
    base: dict[str, Any] = {
        "scope": scope,
        "zone": zone,
        "n_days": n_days,
        "n_hours": n_hours,
    }
    if not n_hours:
        base.update(
            {
                name: np.nan
                for name in (
                    "production_mae",
                    "challenger_mae",
                    "mae_delta",
                    "production_rmse",
                    "challenger_rmse",
                    "rmse_delta",
                    "production_bias",
                    "challenger_bias",
                    "absolute_bias_delta",
                    "production_smape_pct",
                    "challenger_smape_pct",
                    "smape_delta_pct",
                    "challenger_win_rate",
                    "tie_rate",
                    "challenger_day_win_rate",
                )
            }
        )
        return base
    actual = frame["actual"].to_numpy(float)
    production = frame["production_q50"].to_numpy(float)
    challenger = frame["challenger_q50"].to_numpy(float)
    production_error = production - actual
    challenger_error = challenger - actual

    def smape(forecast: np.ndarray) -> float:
        denominator = np.abs(forecast) + np.abs(actual)
        ratio = np.divide(
            2.0 * np.abs(forecast - actual),
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 1e-12,
        )
        return 100.0 * float(np.mean(ratio))

    production_abs = np.abs(production_error)
    challenger_abs = np.abs(challenger_error)
    daily = frame.assign(
        _production_abs=production_abs,
        _challenger_abs=challenger_abs,
    ).groupby(["zone", "delivery_day_local"], sort=False)[
        ["_production_abs", "_challenger_abs"]
    ].mean()
    production_bias = float(np.mean(production_error))
    challenger_bias = float(np.mean(challenger_error))
    production_mae = float(np.mean(production_abs))
    challenger_mae = float(np.mean(challenger_abs))
    production_rmse = float(np.sqrt(np.mean(np.square(production_error))))
    challenger_rmse = float(np.sqrt(np.mean(np.square(challenger_error))))
    production_smape = smape(production)
    challenger_smape = smape(challenger)
    base.update(
        {
            "production_mae": production_mae,
            "challenger_mae": challenger_mae,
            "mae_delta": challenger_mae - production_mae,
            "production_rmse": production_rmse,
            "challenger_rmse": challenger_rmse,
            "rmse_delta": challenger_rmse - production_rmse,
            "production_bias": production_bias,
            "challenger_bias": challenger_bias,
            "absolute_bias_delta": abs(challenger_bias) - abs(production_bias),
            "production_smape_pct": production_smape,
            "challenger_smape_pct": challenger_smape,
            "smape_delta_pct": challenger_smape - production_smape,
            "challenger_win_rate": float(np.mean(challenger_abs < production_abs)),
            "tie_rate": float(np.mean(challenger_abs == production_abs)),
            "challenger_day_win_rate": float(
                np.mean(daily["_challenger_abs"] < daily["_production_abs"])
            ),
        }
    )
    return base


def _metrics(paired: pd.DataFrame, zones: Sequence[str]) -> pd.DataFrame:
    rows = [
        _metric_row(
            paired.loc[paired["zone"] == zone],
            zone=zone,
            scope="zone",
        )
        for zone in zones
    ]
    rows.append(_metric_row(paired, zone="ALL", scope="aggregate"))
    return pd.DataFrame(rows)


def build_comparison(
    specs: Sequence[ZoneComparisonSpec],
    *,
    start: str | date,
    end: str | date,
    actual_loader: ActualLoader = load_audited_statistics_actuals,
    archive_validator: ArchiveValidator = _default_archive_validator,
    registry_path: str | Path | None = None,
) -> ComparisonBuild:
    start_day = _parse_day(start, name="start")
    end_day = _parse_day(end, name="end")
    if start_day > end_day:
        raise ComparisonError("start doit être antérieur ou égal à end.")
    zones = tuple(spec.code for spec in specs)
    if len(zones) != len(set(zones)):
        raise ComparisonError("Les zones de comparaison doivent être uniques.")
    paired_parts: list[pd.DataFrame] = []
    pair_audit: list[dict[str, Any]] = []
    unpaired: list[dict[str, str]] = []
    actual_audit: dict[str, dict[str, Any]] = {}

    for spec in specs:
        production, challenger = _scan_archive_days(
            spec,
            start=start_day,
            end=end_day,
        )
        for day in sorted(set(production).difference(challenger)):
            unpaired.append(
                {"zone": spec.code, "delivery_day_local": day.isoformat(), "missing": "challenger"}
            )
        for day in sorted(set(challenger).difference(production)):
            unpaired.append(
                {"zone": spec.code, "delivery_day_local": day.isoformat(), "missing": "production"}
            )
        paired_days = sorted(set(production).intersection(challenger))
        if not paired_days:
            actual_audit[spec.code] = {
                "source": "not_loaded_no_archive_pair",
                "path": None,
                "sha256": None,
                "n_available_hours": 0,
            }
            continue
        raw_actuals = actual_loader(spec)
        actuals = _normalize_actuals(raw_actuals, zone=spec.code)
        actual_audit[spec.code] = {
            "source": actuals.source,
            "path": str(actuals.path) if actuals.path is not None else None,
            "sha256": actuals.sha256,
            "n_available_hours": int(len(actuals.values)),
        }
        for delivery_day in paired_days:
            expected_production = production[delivery_day]
            expected_challenger = challenger[delivery_day]
            validated_production = archive_validator(spec, delivery_day, "saturn")
            validated_challenger = archive_validator(spec, delivery_day, "chronos2")
            if (
                validated_production is None
                or validated_production.resolve() != expected_production
            ):
                raise ComparisonError(
                    f"{expected_production}: archive production non validée comme immuable."
                )
            if (
                validated_challenger is None
                or validated_challenger.resolve() != expected_challenger
            ):
                raise ComparisonError(
                    f"{expected_challenger}: archive challenger non validée comme immuable."
                )
            production_manifest_path = expected_production / "run_manifest.json"
            challenger_manifest_path = expected_challenger / "run_manifest.json"
            production_manifest = _read_json_object(production_manifest_path)
            challenger_manifest = _read_json_object(challenger_manifest_path)
            _require_manifest_identity(
                production_manifest,
                spec=spec,
                delivery_day=delivery_day,
                source="saturn",
                path=production_manifest_path,
            )
            _require_manifest_identity(
                challenger_manifest,
                spec=spec,
                delivery_day=delivery_day,
                source="chronos2",
                path=challenger_manifest_path,
            )
            _require_frozen_downstream(
                production_manifest,
                challenger_manifest,
                production_path=production_manifest_path,
                challenger_path=challenger_manifest_path,
            )
            downstream_source_identity = _require_frozen_downstream_sources(
                expected_production,
                expected_challenger,
            )
            paired_input_identity = _require_paired_input_parity(
                expected_production,
                expected_challenger,
                delivery_index=local_delivery_day_index(
                    delivery_day,
                    timezone=spec.timezone,
                ),
            )
            production_forecast_path = expected_production / spec.forecast_filename
            challenger_forecast_path = expected_challenger / spec.forecast_filename
            production_q50, production_column = _forecast_series(
                production_forecast_path,
                spec=spec,
                delivery_day=delivery_day,
            )
            challenger_q50, challenger_column = _forecast_series(
                challenger_forecast_path,
                spec=spec,
                delivery_day=delivery_day,
            )
            if not production_q50.index.equals(challenger_q50.index):
                raise ComparisonError(
                    f"{spec.code} {delivery_day}: timelines A/B différentes."
                )
            expected_hours = len(production_q50)
            realized = actuals.values.reindex(production_q50.index)
            realized_complete = (
                len(realized) == expected_hours
                and bool(np.isfinite(realized.to_numpy(float)).all())
            )
            pair_record = {
                "zone": spec.code,
                "delivery_day_local": delivery_day.isoformat(),
                "hours_in_local_day": expected_hours,
                "production_archive": str(expected_production),
                "challenger_archive": str(expected_challenger),
                "production_manifest_sha256": _sha256(production_manifest_path),
                "challenger_manifest_sha256": _sha256(challenger_manifest_path),
                "production_artifact_checksums_sha256": _sha256(
                    expected_production / "artifact_checksums.json"
                ),
                "challenger_artifact_checksums_sha256": _sha256(
                    expected_challenger / "artifact_checksums.json"
                ),
                "downstream_source_identity_sha256": (
                    downstream_source_identity["sha256"]
                ),
                "n_verified_downstream_sources": (
                    downstream_source_identity["n_sources"]
                ),
                **paired_input_identity,
                "production_forecast_sha256": _sha256(production_forecast_path),
                "challenger_forecast_sha256": _sha256(challenger_forecast_path),
                "production_q50_column": production_column,
                "challenger_q50_column": challenger_column,
                "realized_complete": realized_complete,
                "scored": realized_complete,
            }
            pair_audit.append(pair_record)
            if not realized_complete:
                continue
            frame = pd.DataFrame(
                {
                    "zone": spec.code,
                    "timezone": spec.timezone,
                    "delivery_day_local": delivery_day.isoformat(),
                    "delivery_start_utc": production_q50.index,
                    "hours_in_local_day": expected_hours,
                    "production_q50": production_q50.to_numpy(float),
                    "challenger_q50": challenger_q50.to_numpy(float),
                    "actual": realized.to_numpy(float),
                    "production_q50_column": production_column,
                    "challenger_q50_column": challenger_column,
                    "production_archive": expected_production.name,
                    "challenger_archive": expected_challenger.name,
                }
            )
            frame["production_error"] = frame["production_q50"] - frame["actual"]
            frame["challenger_error"] = frame["challenger_q50"] - frame["actual"]
            frame["production_abs_error"] = frame["production_error"].abs()
            frame["challenger_abs_error"] = frame["challenger_error"].abs()
            frame["challenger_wins"] = (
                frame["challenger_abs_error"] < frame["production_abs_error"]
            )
            frame["tie"] = (
                frame["challenger_abs_error"] == frame["production_abs_error"]
            )
            paired_parts.append(frame)

    columns = [
        "zone",
        "timezone",
        "delivery_day_local",
        "delivery_start_utc",
        "hours_in_local_day",
        "production_q50",
        "challenger_q50",
        "actual",
        "production_q50_column",
        "challenger_q50_column",
        "production_archive",
        "challenger_archive",
        "production_error",
        "challenger_error",
        "production_abs_error",
        "challenger_abs_error",
        "challenger_wins",
        "tie",
    ]
    if paired_parts:
        paired = pd.concat(paired_parts, ignore_index=True).loc[:, columns]
        paired = paired.sort_values(
            ["zone", "delivery_start_utc"], kind="stable"
        ).reset_index(drop=True)
    else:
        paired = pd.DataFrame(columns=columns)
    metrics = _metrics(paired, zones)
    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "comparison_type": "prospective_paired_ab",
        "treatment": "residual_load_source_chronos2_vs_saturn",
        "downstream_policy": "frozen_current_production_downstream",
        "downstream_verification": (
            "identical checksum-sealed configs/recipes/model manifests; "
            "identical aligned history and non-treatment future covariates; "
            "only the five residual-load J+1 features and their oracle mirrors "
            "may differ"
        ),
        "interpretation_warning": (
            "Prospective paired comparison under input distribution shift; "
            "this is not a symmetrically retrained downstream benchmark."
        ),
        "zones": list(zones),
        "start_delivery_day_local": start_day.isoformat(),
        "end_delivery_day_local": end_day.isoformat(),
        "registry_path": str(registry_file) if registry_file is not None else None,
        "registry_sha256": (
            _sha256(registry_file)
            if registry_file is not None and registry_file.is_file()
            else None
        ),
        "n_discovered_pairs": len(pair_audit),
        "n_scored_pairs": int(sum(bool(item["scored"]) for item in pair_audit)),
        "n_scored_hours": int(len(paired)),
        "pairs": pair_audit,
        "unpaired_archives": unpaired,
        "actuals": actual_audit,
        "metric_contract": {
            "error": "forecast_minus_actual",
            "smape_pct": "100 * mean(2*abs(forecast-actual)/(abs(forecast)+abs(actual)))",
            "challenger_win_rate": "strict hourly abs-error wins divided by all paired hours",
            "challenger_day_win_rate": "strict daily-MAE wins divided by all paired days",
            "ties": "not counted as wins",
            "complete_day_hours": [23, 24, 25],
        },
    }
    return ComparisonBuild(paired_hourly=paired, metrics=metrics, manifest=manifest)


def publish_comparison(
    build: ComparisonBuild,
    *,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ComparisonError(
            f"Publication immuable refusée: output-dir existe déjà: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if staging.exists():
        raise ComparisonError(f"Dossier temporaire inattendu: {staging}")
    staging.mkdir()
    try:
        paired_path = staging / "paired_hourly.csv.gz"
        metrics_path = staging / "metrics.csv"
        report_path = staging / "comparison_report.html"
        manifest_path = staging / "comparison_manifest.json"
        paired = build.paired_hourly.copy()
        if "delivery_start_utc" in paired:
            paired["delivery_start_utc"] = paired["delivery_start_utc"].astype(str)
        paired.to_csv(
            paired_path,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        build.metrics.to_csv(metrics_path, index=False)
        manifest = dict(build.manifest)
        manifest["published_at_utc"] = datetime.now(timezone.utc).isoformat()
        paired_artifact = {
            "path": paired_path.name,
            "sha256": _sha256(paired_path),
            "size_bytes": paired_path.stat().st_size,
        }
        metrics_artifact = {
            "path": metrics_path.name,
            "sha256": _sha256(metrics_path),
            "size_bytes": metrics_path.stat().st_size,
        }
        report_path.write_text(
            render_residual_load_comparison_report(
                build.paired_hourly,
                build.metrics,
                manifest,
                paired_sha256=paired_artifact["sha256"],
                metrics_sha256=metrics_artifact["sha256"],
            ),
            encoding="utf-8",
        )
        manifest["artifacts"] = [
            paired_artifact,
            metrics_artifact,
            {
                "path": report_path.name,
                "sha256": _sha256(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        ]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return output


def run_comparison(
    *,
    registry_path: str | Path,
    zones: Sequence[str],
    start: str | date,
    end: str | date,
    output_dir: str | Path,
    actual_loader: ActualLoader = load_audited_statistics_actuals,
    archive_validator: ArchiveValidator = _default_archive_validator,
) -> Path:
    registry_file = Path(registry_path).expanduser().resolve()
    specs = load_zone_specs(registry_file, zones=zones)
    build = build_comparison(
        specs,
        start=start,
        end=end,
        actual_loader=actual_loader,
        archive_validator=archive_validator,
        registry_path=registry_file,
    )
    return publish_comparison(build, output_dir=output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prospectivement les runs appariés Saturn/Chronos-2 avec "
            "le downstream courant gelé."
        )
    )
    parser.add_argument("--zones", nargs="+", default=list(SUPPORTED_ZONES))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD inclus")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD inclus")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_comparison(
        registry_path=args.registry,
        zones=args.zones,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "published",
                "comparison_type": "prospective_paired_ab",
                "output_dir": str(output),
                "html_report": str(output / "comparison_report.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
