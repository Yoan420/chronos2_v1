"""Strict operational configuration and sidecar inputs for Kalman exports.

This module is intentionally independent from the sealed Chronos-2 live
contract.  It only enriches the disposable reporting view used to derive the
``residual_kalman`` export.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import yaml

from .kalman_covariates import (
    BASE_RESIDUAL_LOAD_COVARIATES,
    KalmanCovariateConfig,
    KalmanCovariateError,
)
from .kalman_residual import KalmanResidualConfig, KalmanResidualError


KALMAN_OPERATIONAL_CONFIG_VERSION = 1
_SAFE_SOURCE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SUPPORTED_SOURCE_SUFFIXES = (".parquet", ".csv", ".csv.gz")
KALMAN_WEATHER_ZONES: Mapping[str, str] = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
_WEATHER_CANDIDATES = frozenset(
    {"linear_weather", "linear_renewables", "linear_fundamental"}
)


class KalmanConfigurationError(ValueError):
    """Raised when an operational Kalman sidecar is unsafe or ambiguous."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML safe loader that refuses shadowed duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise KalmanConfigurationError(f"Cle YAML dupliquee: {key!r}.")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class KalmanAdditionalSource:
    name: str
    path: Path
    timestamp_column: str
    # Output alias -> source-file column.  Aliases must be raw covariates from
    # the explicit Kalman contract; derived columns are computed downstream.
    columns: Mapping[str, str]
    information_type: str
    cutoff_policy: str
    origin_column: str
    revision_column: str | None
    cutoff_column: str | None
    cutoff_time: str
    sha256: str
    size_bytes: int

    def audit_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "timestamp_column": self.timestamp_column,
            "columns": dict(self.columns),
            "information_type": self.information_type,
            "cutoff_policy": self.cutoff_policy,
            "origin_column": self.origin_column,
            "revision_column": self.revision_column,
            "cutoff_column": self.cutoff_column,
            "cutoff_time": self.cutoff_time,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }


@dataclass(frozen=True)
class KalmanUpstreamHistory:
    path: Path
    sha256: str
    audit_path: Path
    audit_sha256: str
    upstream_model: str
    protocol: str

    def audit_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "audit_path": str(self.audit_path),
            "audit_sha256": self.audit_sha256,
            "upstream_model": self.upstream_model,
            "protocol": self.protocol,
        }


@dataclass(frozen=True)
class KalmanOperationalConfiguration:
    path: Path
    sha256: str
    filter_config: KalmanResidualConfig
    covariate_config: KalmanCovariateConfig
    training_lookback_days: int | None
    rolling_refit_workers: int
    upstream_history: KalmanUpstreamHistory | None
    additional_sources: tuple[KalmanAdditionalSource, ...]
    provenance: Mapping[str, Any]

    def contract_dict(self) -> dict[str, Any]:
        return {
            "version": KALMAN_OPERATIONAL_CONFIG_VERSION,
            "training_lookback_days": self.training_lookback_days,
            "rolling_refit_workers": int(self.rolling_refit_workers),
            "upstream_history": (
                self.upstream_history.audit_dict()
                if self.upstream_history is not None
                else None
            ),
            "filter_parameters": {
                field.name: _json_value(getattr(self.filter_config, field.name))
                for field in fields(KalmanResidualConfig)
            },
            "covariates": self.covariate_config.to_dict(),
            "additional_sources": [
                source.audit_dict() for source in self.additional_sources
            ],
            "provenance": _json_value(self.provenance),
        }

    def audit_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "contract": self.contract_dict(),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise KalmanConfigurationError(
        f"Valeur de provenance non serialisable: {type(value).__name__}."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KalmanConfigurationError(f"{name} doit etre un objet YAML.")
    return value


def _strict_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KalmanConfigurationError(f"{name} doit etre numerique.")
    result = float(value)
    if not np.isfinite(result):
        raise KalmanConfigurationError(f"{name} doit etre fini.")
    return result


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KalmanConfigurationError(f"{name} doit etre un entier.")
    return int(value)


def _parse_filter_parameters(raw: Any) -> KalmanResidualConfig:
    mapping = _strict_mapping(raw, name="filter_parameters")
    defaults = KalmanResidualConfig()
    allowed = {field.name for field in fields(KalmanResidualConfig)}
    unknown = sorted(set(map(str, mapping)).difference(allowed))
    if unknown:
        raise KalmanConfigurationError(
            f"Champs filter_parameters inconnus: {unknown}."
        )
    values: dict[str, Any] = {}
    integer_fields = {
        "governance_lookback_days",
        "governance_minimum_days",
        "governance_confirmation_days",
    }
    for field in fields(KalmanResidualConfig):
        value = mapping.get(field.name, getattr(defaults, field.name))
        if field.name == "candidate_kinds":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise KalmanConfigurationError(
                    "filter_parameters.candidate_kinds doit etre une liste."
                )
            candidates = tuple(str(item).strip() for item in value)
            if any(not item for item in candidates):
                raise KalmanConfigurationError(
                    "filter_parameters.candidate_kinds contient une valeur vide."
                )
            values[field.name] = candidates
        elif field.name in integer_fields:
            values[field.name] = _strict_int(
                value, name=f"filter_parameters.{field.name}"
            )
        else:
            values[field.name] = _strict_float(
                value, name=f"filter_parameters.{field.name}"
            )
    try:
        config = KalmanResidualConfig(**values)
        config.validate()
    except (KalmanResidualError, TypeError, ValueError) as exc:
        raise KalmanConfigurationError(str(exc)) from exc
    return config


def _resolve_source_path(
    value: Any,
    *,
    project_root: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise KalmanConfigurationError("additional_sources.path doit etre non vide.")
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    if not path.is_file():
        raise KalmanConfigurationError(f"Source Kalman introuvable: {path}.")
    lowered = path.name.casefold()
    if not any(lowered.endswith(suffix) for suffix in _SUPPORTED_SOURCE_SUFFIXES):
        raise KalmanConfigurationError(
            f"Format de source Kalman non supporte: {path.name}."
        )
    return path


def _parse_additional_sources(
    raw: Any,
    *,
    project_root: Path,
    covariates: KalmanCovariateConfig,
) -> tuple[KalmanAdditionalSource, ...]:
    if raw is None:
        return ()
    mapping = _strict_mapping(raw, name="additional_sources")
    output: list[KalmanAdditionalSource] = []
    aliases_seen: set[str] = set()
    for raw_name, value in mapping.items():
        name = str(raw_name)
        if not _SAFE_SOURCE_NAME.fullmatch(name):
            raise KalmanConfigurationError(
                f"Nom de source Kalman invalide: {name!r}."
            )
        source = _strict_mapping(value, name=f"additional_sources.{name}")
        allowed = {
            "path",
            "timestamp_column",
            "columns",
            "information_type",
            "cutoff_policy",
            "origin_column",
            "revision_column",
            "cutoff_column",
            "cutoff_time",
            "sha256",
        }
        unknown = sorted(set(map(str, source)).difference(allowed))
        if unknown:
            raise KalmanConfigurationError(
                f"additional_sources.{name}: champs inconnus: {unknown}."
            )
        path = _resolve_source_path(source.get("path"), project_root=project_root)
        timestamp_column = str(
            source.get("timestamp_column", "delivery_start_utc")
        ).strip()
        if not timestamp_column:
            raise KalmanConfigurationError(
                f"additional_sources.{name}.timestamp_column est vide."
            )
        column_mapping = _strict_mapping(
            source.get("columns"), name=f"additional_sources.{name}.columns"
        )
        columns = {
            str(alias).strip(): str(source_column).strip()
            for alias, source_column in column_mapping.items()
        }
        if not columns or any(not alias or not column for alias, column in columns.items()):
            raise KalmanConfigurationError(
                f"additional_sources.{name}.columns doit etre non vide."
            )
        forbidden_source_columns = sorted(
            column
            for column in columns.values()
            if any(
                token in column.casefold()
                for token in (
                    "actual",
                    "observed",
                    "realized",
                    "realised",
                    "target",
                    "oracle",
                    "storm",
                    "mkonline",
                )
            )
        )
        if forbidden_source_columns:
            raise KalmanConfigurationError(
                f"additional_sources.{name}: colonnes non causales interdites: "
                f"{forbidden_source_columns}."
            )
        duplicated = sorted(aliases_seen.intersection(columns))
        if duplicated:
            raise KalmanConfigurationError(
                f"Aliases fournis par plusieurs sources: {duplicated}."
            )
        invalid_aliases = sorted(set(columns).difference(covariates.input_columns))
        if invalid_aliases:
            raise KalmanConfigurationError(
                f"additional_sources.{name}: aliases hors input_columns: "
                f"{invalid_aliases}."
            )
        if len(set(columns.values())) != len(columns):
            raise KalmanConfigurationError(
                f"additional_sources.{name}: colonnes source dupliquees."
            )
        actual_sha = _sha256(path)
        expected_sha = source.get("sha256")
        if expected_sha is None:
            raise KalmanConfigurationError(
                f"additional_sources.{name}.sha256 est obligatoire."
            )
        expected_sha = str(expected_sha).strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise KalmanConfigurationError(
                f"additional_sources.{name}.sha256 doit avoir 64 hexadecimaux."
            )
        if actual_sha != expected_sha:
            raise KalmanConfigurationError(
                f"Checksum source Kalman invalide pour {name}: "
                f"attendu={expected_sha}, obtenu={actual_sha}."
            )
        information_type = str(source.get("information_type", "")).strip()
        if information_type not in {
            "forecast",
            "day_ahead_forecast",
            "last_known_market_prices",
        }:
            raise KalmanConfigurationError(
                f"additional_sources.{name}.information_type doit declarer "
                "forecast, day_ahead_forecast ou last_known_market_prices."
            )
        cutoff_policy = str(source.get("cutoff_policy", "")).strip()
        if not cutoff_policy:
            raise KalmanConfigurationError(
                f"additional_sources.{name}.cutoff_policy est obligatoire."
            )
        if not any(
            marker in cutoff_policy.casefold()
            for marker in ("before", "as-of", "asof", "cutoff", "d-1", "previous_day")
        ):
            raise KalmanConfigurationError(
                f"additional_sources.{name}.cutoff_policy doit decrire "
                "explicitement une selection causale avant le cutoff."
            )
        origin_column = str(source.get("origin_column", "")).strip()
        revision_column = (
            str(source["revision_column"]).strip()
            if source.get("revision_column") is not None
            else None
        )
        cutoff_column = (
            str(source["cutoff_column"]).strip()
            if source.get("cutoff_column") is not None
            else None
        )
        cutoff_time = str(source.get("cutoff_time", "08:00")).strip()
        if not origin_column:
            raise KalmanConfigurationError(
                f"additional_sources.{name}.origin_column est obligatoire."
            )
        if revision_column == "" or cutoff_column == "" or not cutoff_time:
            raise KalmanConfigurationError(
                f"additional_sources.{name}: provenance PIT incomplete."
            )
        aliases_seen.update(columns)
        output.append(
            KalmanAdditionalSource(
                name=name,
                path=path,
                timestamp_column=timestamp_column,
                columns=columns,
                information_type=information_type,
                cutoff_policy=cutoff_policy,
                origin_column=origin_column,
                revision_column=revision_column,
                cutoff_column=cutoff_column,
                cutoff_time=cutoff_time,
                sha256=actual_sha,
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(output)


def _verified_sha256(
    raw: Any,
    *,
    path: Path,
    name: str,
) -> str:
    if raw is None:
        raise KalmanConfigurationError(f"{name}.sha256 est obligatoire.")
    expected = str(raw).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise KalmanConfigurationError(
            f"{name}.sha256 doit avoir 64 hexadecimaux."
        )
    actual = _sha256(path)
    if actual != expected:
        raise KalmanConfigurationError(
            f"Checksum invalide pour {name}: attendu={expected}, obtenu={actual}."
        )
    return actual


def _parse_upstream_history(
    raw: Any,
    *,
    project_root: Path,
) -> KalmanUpstreamHistory | None:
    if raw is None:
        return None
    source = _strict_mapping(raw, name="upstream_history")
    allowed = {
        "path",
        "sha256",
        "audit_path",
        "audit_sha256",
        "upstream_model",
        "protocol",
    }
    unknown = sorted(set(map(str, source)).difference(allowed))
    if unknown:
        raise KalmanConfigurationError(
            f"upstream_history: champs inconnus: {unknown}."
        )
    path = _resolve_source_path(source.get("path"), project_root=project_root)
    raw_audit_path = source.get("audit_path")
    if not isinstance(raw_audit_path, str) or not raw_audit_path.strip():
        raise KalmanConfigurationError("upstream_history.audit_path est obligatoire.")
    audit_candidate = Path(raw_audit_path).expanduser()
    audit_path = (
        audit_candidate
        if audit_candidate.is_absolute()
        else project_root / audit_candidate
    ).resolve()
    if not audit_path.is_file() or audit_path.suffix.casefold() != ".json":
        raise KalmanConfigurationError(
            f"Audit upstream Kalman introuvable ou invalide: {audit_path}."
        )
    sha256 = _verified_sha256(
        source.get("sha256"), path=path, name="upstream_history"
    )
    audit_sha256 = _verified_sha256(
        source.get("audit_sha256"),
        path=audit_path,
        name="upstream_history.audit",
    )
    upstream_model = str(source.get("upstream_model", "")).strip()
    protocol = str(source.get("protocol", "")).strip()
    expected_protocol = (
        "blocked_prequential_residual_then_issued_evaluation_overlay"
    )
    if upstream_model != "residual_corrected" or protocol != expected_protocol:
        raise KalmanConfigurationError(
            "upstream_history doit declarer residual_corrected et le protocole "
            "prequentiel causal attendu."
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KalmanConfigurationError(
            f"Audit upstream Kalman illisible: {audit_path}."
        ) from exc
    if not isinstance(audit, Mapping):
        raise KalmanConfigurationError("Audit upstream Kalman invalide.")
    required_audit = {
        "protocol": expected_protocol,
        "selected_recipe_is_frozen": True,
        "fit_label_rule": "local_delivery_day < block_start_day",
        "causality_violations": 0,
        "published_origin_causality_violations": 0,
        "published_origin_contract_mismatches": 0,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise KalmanConfigurationError(
                f"Audit upstream Kalman {key}={audit.get(key)!r}, "
                f"attendu={expected!r}."
            )
    return KalmanUpstreamHistory(
        path=path,
        sha256=sha256,
        audit_path=audit_path,
        audit_sha256=audit_sha256,
        upstream_model=upstream_model,
        protocol=protocol,
    )


def load_kalman_operational_configuration(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    zone: str | None = None,
    upstream_model: str = "residual_corrected",
) -> KalmanOperationalConfiguration:
    """Validate one sidecar and select only the requested country's upstream.

    A country-indexed incumbent prefix is deliberately excluded when Kalman
    consumes another upstream model, including a promoted LoRA pipeline.
    """

    candidate = Path(path).expanduser()
    base = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not resolved.is_file():
        raise KalmanConfigurationError(
            f"Configuration Kalman operationnelle introuvable: {resolved}."
        )
    try:
        raw = yaml.load(
            resolved.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise KalmanConfigurationError(
            f"Impossible de lire la configuration Kalman {resolved}: {exc}"
        ) from exc
    mapping = _strict_mapping(raw, name="configuration Kalman")
    allowed = {
        "version",
        "training_lookback_days",
        "rolling_refit_workers",
        "upstream_history",
        "upstream_history_by_zone",
        "filter_parameters",
        "covariates",
        "additional_sources",
        "provenance",
    }
    unknown = sorted(set(map(str, mapping)).difference(allowed))
    if unknown:
        raise KalmanConfigurationError(
            f"Champs racine Kalman inconnus: {unknown}."
        )
    if "upstream_history" in mapping and "upstream_history_by_zone" in mapping:
        raise KalmanConfigurationError(
            "upstream_history et upstream_history_by_zone sont mutuellement exclusifs."
        )
    if not isinstance(upstream_model, str) or not upstream_model.strip():
        raise KalmanConfigurationError("upstream_model doit etre un nom non vide.")
    selected_model = upstream_model.strip()
    selected_zone = None
    if zone is not None:
        if not isinstance(zone, str) or zone.strip().upper() not in KALMAN_WEATHER_ZONES:
            raise KalmanConfigurationError("zone doit etre FR, DE, BE, NL ou ES.")
        selected_zone = zone.strip().upper()
    missing_required = sorted(
        {"filter_parameters", "covariates"}.difference(mapping)
    )
    if missing_required:
        raise KalmanConfigurationError(
            f"Champs racine Kalman obligatoires absents: {missing_required}."
        )
    version = mapping.get("version", KALMAN_OPERATIONAL_CONFIG_VERSION)
    if isinstance(version, bool) or version != KALMAN_OPERATIONAL_CONFIG_VERSION:
        raise KalmanConfigurationError(
            f"version Kalman attendue={KALMAN_OPERATIONAL_CONFIG_VERSION}, "
            f"obtenue={version!r}."
        )
    try:
        covariates = KalmanCovariateConfig.from_mapping(mapping.get("covariates"))
    except KalmanCovariateError as exc:
        raise KalmanConfigurationError(str(exc)) from exc
    filter_config = _parse_filter_parameters(mapping.get("filter_parameters", {}))
    training_lookback_raw = mapping.get("training_lookback_days")
    if training_lookback_raw is None:
        training_lookback_days = None
    else:
        training_lookback_days = _strict_int(
            training_lookback_raw,
            name="training_lookback_days",
        )
        if training_lookback_days < 1:
            raise KalmanConfigurationError(
                "training_lookback_days doit etre strictement positif."
            )
    rolling_refit_workers = _strict_int(
        mapping.get("rolling_refit_workers", 1),
        name="rolling_refit_workers",
    )
    if not 1 <= rolling_refit_workers <= 8:
        raise KalmanConfigurationError(
            "rolling_refit_workers doit etre compris entre 1 et 8."
        )
    provenance_raw = mapping.get("provenance", {})
    provenance = _strict_mapping(provenance_raw, name="provenance")
    # Eagerly validate that provenance can be written with allow_nan=False.
    json.dumps(_json_value(provenance), allow_nan=False)
    sources = _parse_additional_sources(
        mapping.get("additional_sources"),
        project_root=base,
        covariates=covariates,
    )
    provenance = dict(provenance)
    if "upstream_history_by_zone" in mapping:
        by_zone = _strict_mapping(
            mapping["upstream_history_by_zone"], name="upstream_history_by_zone"
        )
        if not by_zone:
            raise KalmanConfigurationError("upstream_history_by_zone ne peut pas etre vide.")
        indexed_sources = {}
        for raw_zone, source in by_zone.items():
            if not isinstance(raw_zone, str) or raw_zone.strip().upper() not in KALMAN_WEATHER_ZONES:
                raise KalmanConfigurationError(
                    f"Pays upstream_history_by_zone invalide: {raw_zone!r}."
                )
            code = raw_zone.strip().upper()
            if code in indexed_sources:
                raise KalmanConfigurationError(
                    f"Pays upstream_history_by_zone duplique: {code}."
                )
            indexed_sources[code] = _strict_mapping(
                source, name=f"upstream_history_by_zone.{code}"
            )
        if selected_model != "residual_corrected":
            # Do not read/hash incumbent country files for a LoRA branch.
            # Their predictions must never be mixed with another upstream.
            upstream_history = None
            selection_status = "excluded_incompatible_upstream_model"
        else:
            if selected_zone is None:
                raise KalmanConfigurationError(
                    "zone est obligatoire pour upstream_history_by_zone."
                )
            if selected_zone not in indexed_sources:
                raise KalmanConfigurationError(
                    f"upstream_history_by_zone: aucun prefixe pour {selected_zone}."
                )
            upstream_history = _parse_upstream_history(
                indexed_sources[selected_zone], project_root=base
            )
            selection_status = "selected"
        provenance["upstream_history_selection"] = {
            "source": "upstream_history_by_zone",
            "status": selection_status,
            "zone": selected_zone,
            "upstream_model": selected_model,
            "incumbent_history_attached": upstream_history is not None,
        }
    else:
        configured_upstream = mapping.get("upstream_history")
        if configured_upstream is not None and selected_model != "residual_corrected":
            raise KalmanConfigurationError(
                "upstream_history residual_corrected incompatible avec "
                f"upstream_model={selected_model!r}; aucun melange de modeles autorise."
            )
        upstream_history = _parse_upstream_history(
            configured_upstream, project_root=base
        )
    return KalmanOperationalConfiguration(
        path=resolved,
        sha256=_sha256(resolved),
        filter_config=filter_config,
        covariate_config=covariates,
        training_lookback_days=training_lookback_days,
        rolling_refit_workers=rolling_refit_workers,
        upstream_history=upstream_history,
        additional_sources=sources,
        provenance=dict(provenance),
    )


def _expand_weather_tokens(value: Any, *, tokens: Mapping[str, str]) -> Any:
    """Expand a small, explicit token set in YAML keys and scalar strings."""

    if isinstance(value, Mapping):
        output: dict[Any, Any] = {}
        for raw_key, raw_value in value.items():
            key = _expand_weather_tokens(raw_key, tokens=tokens)
            if key in output:
                raise KalmanConfigurationError(
                    f"Cle dupliquee apres expansion du template meteo: {key!r}."
                )
            output[key] = _expand_weather_tokens(raw_value, tokens=tokens)
        return output
    if isinstance(value, list):
        return [_expand_weather_tokens(item, tokens=tokens) for item in value]
    if isinstance(value, str):
        expanded = value
        for token, replacement in tokens.items():
            expanded = expanded.replace(token, replacement)
        return expanded
    return value


def render_kalman_weather_operational_configuration(
    template_path: str | Path,
    *,
    zone: str,
    delivery_day: str,
    output_path: str | Path,
    project_root: str | Path | None = None,
    runtime_source_root: str | Path | None = None,
) -> KalmanOperationalConfiguration:
    """Render one immutable, checksum-pinned weather sidecar for a delivery.

    The editable template contains zone tokens but deliberately no durable
    source checksum.  Checksums are injected only after the PIT weather files
    have been refreshed, then the ordinary strict loader validates the exact
    runtime artifact.  Existing Kalman configuration files are never changed.
    """

    canonical_zone = str(zone).strip().upper()
    if canonical_zone not in KALMAN_WEATHER_ZONES:
        raise KalmanConfigurationError(
            f"Zone Kalman meteo non supportee: {zone!r}."
        )
    try:
        delivery = pd.Timestamp(delivery_day).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise KalmanConfigurationError(
            f"Jour de livraison Kalman meteo invalide: {delivery_day!r}."
        ) from exc
    base = Path(project_root).expanduser().resolve() if project_root else Path.cwd()
    raw_template = Path(template_path).expanduser()
    resolved_template = (
        raw_template if raw_template.is_absolute() else base / raw_template
    ).resolve()
    if not resolved_template.is_file():
        raise KalmanConfigurationError(
            f"Template Kalman meteo introuvable: {resolved_template}."
        )
    try:
        raw = yaml.load(
            resolved_template.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise KalmanConfigurationError(
            f"Impossible de lire le template Kalman meteo {resolved_template}: {exc}"
        ) from exc
    template = _strict_mapping(raw, name="template Kalman meteo")
    lower = canonical_zone.casefold()
    resolved_runtime_source_root = (
        Path(runtime_source_root).expanduser().resolve()
        if runtime_source_root is not None
        else None
    )
    runtime_source_manifest: Path | None = None
    if resolved_runtime_source_root is not None:
        runtime_source_manifest = (
            resolved_runtime_source_root / "source_bundle_manifest.json"
        )
        if not runtime_source_manifest.is_file():
            raise KalmanConfigurationError(
                "Manifest du bundle de sources runtime Kalman absent: "
                f"{runtime_source_manifest}."
            )
    rendered = _expand_weather_tokens(
        template,
        tokens={
            "${zone}": canonical_zone,
            "${zone_lower}": lower,
            "${timezone}": KALMAN_WEATHER_ZONES[canonical_zone],
            "${delivery_day}": delivery,
            "${runtime_source_root}": (
                str(resolved_runtime_source_root)
                if resolved_runtime_source_root is not None
                else "${runtime_source_root}"
            ),
        },
    )
    if not isinstance(rendered, dict):  # pragma: no cover - guarded above.
        raise KalmanConfigurationError("Template Kalman meteo invalide.")
    if rendered.get("training_lookback_days") != 365:
        raise KalmanConfigurationError(
            "Le template Kalman meteo operationnel exige "
            "training_lookback_days: 365."
        )
    sources = _strict_mapping(
        rendered.get("additional_sources"),
        name="additional_sources du template Kalman meteo",
    )
    if not sources:
        raise KalmanConfigurationError(
            "Le template Kalman meteo doit declarer des sources PIT."
        )
    source_aliases: set[str] = set()
    for name, raw_source in sources.items():
        source = _strict_mapping(
            raw_source,
            name=f"additional_sources.{name}",
        )
        path = _resolve_source_path(source.get("path"), project_root=base)
        source["sha256"] = _sha256(path)
        columns = _strict_mapping(
            source.get("columns"),
            name=f"additional_sources.{name}.columns",
        )
        source_aliases.update(str(alias) for alias in columns)
    upstream = rendered.get("upstream_history")
    if upstream is not None:
        upstream_mapping = _strict_mapping(
            upstream, name="upstream_history du template Kalman meteo"
        )
        upstream_path = _resolve_source_path(
            upstream_mapping.get("path"), project_root=base
        )
        raw_audit_path = upstream_mapping.get("audit_path")
        if not isinstance(raw_audit_path, str) or not raw_audit_path.strip():
            raise KalmanConfigurationError(
                "upstream_history.audit_path est obligatoire."
            )
        audit_candidate = Path(raw_audit_path).expanduser()
        audit_path = (
            audit_candidate if audit_candidate.is_absolute() else base / audit_candidate
        ).resolve()
        if not audit_path.is_file():
            raise KalmanConfigurationError(
                f"Audit upstream Kalman introuvable: {audit_path}."
            )
        upstream_mapping["sha256"] = _sha256(upstream_path)
        upstream_mapping["audit_sha256"] = _sha256(audit_path)
    covariates = _strict_mapping(
        rendered.get("covariates"),
        name="covariates du template Kalman meteo",
    )
    inputs = covariates.get("input_columns")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise KalmanConfigurationError(
            "Le template Kalman meteo doit declarer input_columns."
        )
    input_aliases = {str(value) for value in inputs}
    sourced_native_aliases = source_aliases.intersection(
        BASE_RESIDUAL_LOAD_COVARIATES
    )
    expected_external_aliases = input_aliases.difference(
        BASE_RESIDUAL_LOAD_COVARIATES
    )
    if (
        source_aliases.difference(BASE_RESIDUAL_LOAD_COVARIATES)
        != expected_external_aliases
        or sourced_native_aliases
        not in (set(), set(BASE_RESIDUAL_LOAD_COVARIATES))
        or any(
            not alias.startswith(f"{lower}_")
            for alias in source_aliases.difference(
                BASE_RESIDUAL_LOAD_COVARIATES
            )
        )
    ):
        raise KalmanConfigurationError(
            f"Le template Kalman meteo {canonical_zone} doit utiliser "
            "exactement ses propres aliases de sources PIT; les cinq charges "
            "residuelles doivent rester toutes natives ou etre toutes fournies "
            "par une source PIT auditee."
        )
    filter_parameters = _strict_mapping(
        rendered.get("filter_parameters"),
        name="filter_parameters du template Kalman meteo",
    )
    candidates = filter_parameters.get("candidate_kinds")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise KalmanConfigurationError(
            "Le template Kalman meteo doit declarer candidate_kinds."
        )
    if not _WEATHER_CANDIDATES.intersection(map(str, candidates)):
        raise KalmanConfigurationError(
            "Le template Kalman meteo ne contient aucun candidat meteo."
        )
    provenance = rendered.get("provenance")
    if provenance is None:
        provenance = {}
        rendered["provenance"] = provenance
    if not isinstance(provenance, dict):
        raise KalmanConfigurationError("provenance doit etre un objet YAML.")
    provenance["runtime_sidecar"] = {
        "zone": canonical_zone,
        "delivery_day": delivery,
        "template_path": str(resolved_template),
        "template_sha256": _sha256(resolved_template),
        "checksums_injected_after_all_source_refresh": True,
        "runtime_source_root": (
            str(resolved_runtime_source_root)
            if resolved_runtime_source_root is not None
            else None
        ),
        "runtime_source_manifest": (
            str(runtime_source_manifest)
            if runtime_source_manifest is not None
            else None
        ),
        "runtime_source_manifest_sha256": (
            _sha256(runtime_source_manifest)
            if runtime_source_manifest is not None
            else None
        ),
    }
    output = Path(output_path).expanduser()
    output = (output if output.is_absolute() else base / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(rendered, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_kalman_operational_configuration(output, project_root=base)


def attach_kalman_upstream_history(
    statistics: pd.DataFrame,
    configuration: KalmanOperationalConfiguration,
    *,
    timezone: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Prepend the audited causal residual history needed by rolling D-365."""

    source = configuration.upstream_history
    if source is None:
        return statistics.copy(), {"status": "not_configured"}
    if _sha256(source.path) != source.sha256 or _sha256(source.audit_path) != source.audit_sha256:
        raise KalmanConfigurationError(
            "Le prefixe upstream Kalman a change apres validation."
        )
    if "delivery_start_utc" not in statistics:
        raise KalmanConfigurationError(
            "Statistics ne contient pas delivery_start_utc."
        )
    issued = statistics.copy()
    issued_index = _aware_utc_index(
        issued["delivery_start_utc"], name="Statistics upstream Kalman"
    )
    issued = issued.drop(columns=["delivery_start_utc"])
    issued.index = issued_index
    prefix = (
        pd.read_parquet(source.path)
        if source.path.name.casefold().endswith(".parquet")
        else pd.read_csv(source.path)
    )
    if "delivery_start_utc" not in prefix:
        raise KalmanConfigurationError(
            "Prefixe upstream Kalman sans delivery_start_utc."
        )
    prefix_index = _aware_utc_index(
        prefix["delivery_start_utc"], name="prefixe upstream Kalman"
    )
    prefix = prefix.drop(columns=["delivery_start_utc"])
    prefix.index = prefix_index
    required = {
        "actual",
        "residual_corrected__q10",
        "residual_corrected__q50",
        "residual_corrected__q90",
        "chronos2__q50",
        "residual_correction",
    }
    missing = sorted(required.difference(prefix.columns))
    if missing:
        raise KalmanConfigurationError(
            f"Prefixe upstream Kalman incomplet: {missing}."
        )
    numeric = prefix.loc[:, sorted(required)].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise KalmanConfigurationError(
            "Prefixe upstream Kalman contient des valeurs non finies."
        )
    if bool(
        (
            numeric["residual_corrected__q10"]
            > numeric["residual_corrected__q50"]
        ).any()
        or (
            numeric["residual_corrected__q50"]
            > numeric["residual_corrected__q90"]
        ).any()
    ):
        raise KalmanConfigurationError(
            "Prefixe upstream Kalman contient des quantiles croises."
        )
    overlap = prefix.index.intersection(issued.index)
    actual_updates = 0
    actual_maximum_difference = 0.0
    if len(overlap):
        prediction_columns = sorted(required.difference({"actual"}))
        left = prefix.reindex(overlap).loc[:, prediction_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        right = issued.reindex(overlap).reindex(columns=prediction_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        populated = np.isfinite(right.to_numpy(dtype=float))
        differences = np.abs(
            left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
        )
        conflict = populated & (differences > 1e-9)
        if bool(conflict.any()):
            raise KalmanConfigurationError(
                "Les predictions du prefixe prequentiel divergent de "
                "l'historique emis sur leur chevauchement."
            )
        # Realised day-ahead prices may be corrected by the canonical provider
        # after the prequential artifact was built.  They are labels, not
        # issued predictions: the refreshed Statistics value deliberately
        # wins in combine_first below, while every prediction remains exact.
        prefix_actual = pd.to_numeric(
            prefix.reindex(overlap)["actual"], errors="coerce"
        ).to_numpy(dtype=float)
        issued_actual = pd.to_numeric(
            issued.reindex(overlap)["actual"], errors="coerce"
        ).to_numpy(dtype=float)
        actual_populated = np.isfinite(prefix_actual) & np.isfinite(issued_actual)
        actual_differences = np.abs(prefix_actual - issued_actual)
        actual_changed = actual_populated & (actual_differences > 1e-5)
        actual_updates = int(actual_changed.sum())
        if bool(actual_populated.any()):
            actual_maximum_difference = float(
                actual_differences[actual_populated].max()
            )
    columns = list(dict.fromkeys([*issued.columns, *prefix.columns]))
    combined_index = prefix.index.union(issued.index).sort_values()
    combined = issued.reindex(index=combined_index, columns=columns).combine_first(
        prefix.reindex(index=combined_index, columns=columns)
    )
    local_days = pd.Index(combined.index.tz_convert(timezone).date)
    valid = combined.loc[:, sorted(required)].apply(
        pd.to_numeric, errors="coerce"
    ).notna().all(axis=1)
    valid_index = combined.index[valid]
    if len(valid_index) > 1 and not valid_index.to_series().diff().iloc[1:].eq(
        pd.Timedelta(hours=1)
    ).all():
        raise KalmanConfigurationError(
            "Le prefixe upstream et Statistics ne forment pas une timeline contigue."
        )
    combined.insert(0, "delivery_start_utc", combined.index)
    combined.index = pd.RangeIndex(len(combined))
    return combined, {
        "status": "complete",
        **source.audit_dict(),
        "prefix_rows": int(len(prefix)),
        "prefix_first_timestamp_utc": prefix_index[0].isoformat(),
        "prefix_last_timestamp_utc": prefix_index[-1].isoformat(),
        "issued_rows": int(len(issued)),
        "combined_rows": int(len(combined)),
        "combined_local_days": int(local_days.nunique()),
        "overlap_rows": int(len(overlap)),
        "overlap_conflicts": 0,
        "overlap_prediction_conflicts": 0,
        "overlap_actual_updates": actual_updates,
        "overlap_actual_maximum_difference_eur_mwh": (
            actual_maximum_difference
        ),
        "causality_violations": 0,
    }


def _read_source(source: KalmanAdditionalSource) -> pd.DataFrame:
    current_sha = _sha256(source.path)
    if current_sha != source.sha256:
        raise KalmanConfigurationError(
            f"La source Kalman {source.name} a change apres validation."
        )
    if source.path.name.casefold().endswith(".parquet"):
        frame = pd.read_parquet(source.path)
    else:
        frame = pd.read_csv(source.path)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise KalmanConfigurationError(
            f"La source Kalman {source.name} est vide."
        )
    required = {
        source.timestamp_column,
        source.origin_column,
        *source.columns.values(),
        *(item for item in (source.revision_column, source.cutoff_column) if item),
    }
    missing = sorted(required.difference(map(str, frame.columns)))
    if missing:
        raise KalmanConfigurationError(
            f"Source Kalman {source.name}: colonnes absentes: {missing}."
        )
    return frame


def _aware_utc_index(
    values: pd.Series,
    *,
    name: str,
    require_unique: bool = True,
) -> pd.DatetimeIndex:
    # ``utc=True`` alone silently accepts naive wall times.  Verify awareness
    # first so DST folds can never be guessed by pandas.
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        pass
    elif pd.api.types.is_datetime64_dtype(values.dtype):
        raise KalmanConfigurationError(
            f"{name}: timestamps naifs interdits; un offset explicite est requis."
        )
    else:
        for position, value in enumerate(values):
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise KalmanConfigurationError(
                    f"{name}: timestamp invalide a la ligne {position}."
                ) from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise KalmanConfigurationError(
                    f"{name}: timestamp sans offset a la ligne {position}."
                )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise KalmanConfigurationError(f"{name}: timeline invalide.") from exc
    if (
        index.hasnans
        or (require_unique and index.has_duplicates)
        or not index.is_monotonic_increasing
    ):
        raise KalmanConfigurationError(
            f"{name}: timeline UTC attendue unique, complete et croissante."
        )
    return index


def _expected_civil_cutoffs(
    delivery_index: pd.DatetimeIndex,
    *,
    timezone: str,
    cutoff_time: str,
) -> pd.DatetimeIndex:
    try:
        clock = pd.Timedelta(
            cutoff_time + ":00" if cutoff_time.count(":") == 1 else cutoff_time
        )
    except (TypeError, ValueError) as exc:
        raise KalmanConfigurationError(
            f"cutoff_time Kalman invalide: {cutoff_time!r}."
        ) from exc
    local_days = delivery_index.tz_convert(timezone).normalize().tz_localize(None)
    return pd.DatetimeIndex(
        [
            (day - pd.Timedelta(days=1) + clock)
            .tz_localize(timezone)
            .tz_convert("UTC")
            for day in local_days
        ]
    )


def _validate_source_provenance(
    raw: pd.DataFrame,
    source: KalmanAdditionalSource,
    *,
    delivery_index: pd.DatetimeIndex,
    timezone: str,
) -> Mapping[str, Any]:
    expected = _expected_civil_cutoffs(
        delivery_index,
        timezone=timezone,
        cutoff_time=source.cutoff_time,
    )
    origin = _aware_utc_index(
        raw[source.origin_column],
        name=f"source Kalman {source.name}.{source.origin_column}",
        require_unique=False,
    )
    violations = np.asarray(origin > expected)
    latest_revision: str | None = None
    if source.revision_column:
        revision = _aware_utc_index(
            raw[source.revision_column],
            name=f"source Kalman {source.name}.{source.revision_column}",
            require_unique=False,
        )
        violations |= np.asarray(revision > expected)
        latest_revision = revision.max().isoformat()
    if source.cutoff_column:
        declared = _aware_utc_index(
            raw[source.cutoff_column],
            name=f"source Kalman {source.name}.{source.cutoff_column}",
            require_unique=False,
        )
        mismatch = np.asarray(declared != expected)
        if bool(mismatch.any()):
            first = int(np.flatnonzero(mismatch)[0])
            raise KalmanConfigurationError(
                f"Source Kalman {source.name}: cutoff declare different du "
                f"cutoff civil a la ligne {first}."
            )
    count = int(violations.sum())
    if count:
        first = int(np.flatnonzero(violations)[0])
        raise KalmanConfigurationError(
            f"Source Kalman {source.name}: {count} violation(s) PIT; "
            f"origine/revision posterieure au cutoff (ligne {first})."
        )
    return {
        "latest_origin_utc": origin.max().isoformat(),
        "latest_revision_utc": latest_revision,
        "cutoff_time": source.cutoff_time,
        "causality_violations": 0,
    }


def attach_additional_kalman_sources(
    base_covariates: pd.DataFrame,
    configuration: KalmanOperationalConfiguration,
    *,
    timestamp_column: str = "timestamp",
    required_future_index: pd.DatetimeIndex | None = None,
    timezone: str = "Europe/Paris",
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Join configured files onto a copied operational covariate frame.

    No archive is mutated.  Historical holes remain visible for the explicit
    missing-data policy, while the requested future horizon fails closed.
    """

    if not isinstance(base_covariates, pd.DataFrame) or base_covariates.empty:
        raise KalmanConfigurationError("Les covariables operationnelles sont vides.")
    if timestamp_column not in base_covariates:
        raise KalmanConfigurationError(
            f"Colonne temporelle operationnelle absente: {timestamp_column}."
        )
    output = base_covariates.copy()
    base_index = _aware_utc_index(
        output[timestamp_column], name="covariates_operationnelles"
    )
    output.index = base_index
    future: pd.DatetimeIndex | None = None
    if required_future_index is not None:
        future = pd.DatetimeIndex(required_future_index)
        if future.tz is None or future.has_duplicates:
            raise KalmanConfigurationError("Horizon futur Kalman invalide.")
        future = future.tz_convert("UTC")
        missing_timestamps = future.difference(base_index)
        if len(missing_timestamps):
            raise KalmanConfigurationError(
                "L'horizon futur Kalman n'est pas couvert par la timeline de base."
            )
    source_audits: list[dict[str, Any]] = []
    for source in configuration.additional_sources:
        collisions = sorted(set(source.columns).intersection(output.columns))
        source_aliases = set(source.columns)
        residual_aliases = set(BASE_RESIDUAL_LOAD_COVARIATES)
        authoritative_residual_source = source_aliases == residual_aliases
        if collisions and not authoritative_residual_source:
            raise KalmanConfigurationError(
                f"Source Kalman {source.name}: collision de colonnes: "
                f"{collisions}. Seule une source PIT fournissant exactement "
                "les cinq charges residuelles peut remplacer des colonnes "
                "deja presentes."
            )
        raw = _read_source(source)
        source_index = _aware_utc_index(
            raw[source.timestamp_column], name=f"source Kalman {source.name}"
        )
        provenance_audit = _validate_source_provenance(
            raw,
            source,
            delivery_index=source_index,
            timezone=timezone,
        )
        selected = raw.loc[:, list(source.columns.values())].copy()
        selected.columns = list(source.columns.keys())
        selected.index = source_index
        selected = selected.apply(pd.to_numeric, errors="coerce").astype(float)
        if bool(np.isinf(selected.to_numpy(dtype=float)).any()):
            raise KalmanConfigurationError(
                f"Source Kalman {source.name}: valeurs infinies interdites."
            )
        aligned = selected.reindex(base_index)
        replacement_policy = "append_non_colliding_source"
        overlap_count: dict[str, int] = {}
        mismatch_count: dict[str, int] = {}
        max_abs_difference: dict[str, float | None] = {}
        replacement_count: dict[str, int] = {}
        comparison_absolute_tolerance = 1e-9
        if authoritative_residual_source:
            if future is not None:
                missing_source_future = future.difference(source_index)
                if len(missing_source_future):
                    raise KalmanConfigurationError(
                        "Covariables Kalman futures incompletes: la source PIT "
                        f"autoritative {source.name} ne couvre pas "
                        f"{len(missing_source_future)} heure(s) requise(s)."
                    )
            replacement_policy = (
                "authoritative_complete_residual_load_pit_on_source_timestamps"
            )
            # A complete residual-load source is deliberately authoritative:
            # the live archive may contain another issuance convention (or
            # values rounded by CSV serialisation).  Its five audited PIT
            # series therefore replace the native block together, while the
            # differences remain fully visible in the sidecar audit.  Missing
            # values are not hidden by the archive and are left for the
            # configured history/future completeness policies to reject.
            covered = np.asarray(base_index.isin(source_index), dtype=bool)
            for alias in BASE_RESIDUAL_LOAD_COVARIATES:
                incoming = aligned[alias].to_numpy(dtype=float)
                existing = (
                    pd.to_numeric(output[alias], errors="coerce").to_numpy(
                        dtype=float
                    )
                    if alias in output
                    else np.full(len(output), np.nan, dtype=float)
                )
                paired = covered & np.isfinite(existing) & np.isfinite(incoming)
                differences = np.abs(existing[paired] - incoming[paired])
                overlap_count[alias] = int(paired.sum())
                mismatch_count[alias] = int(
                    np.count_nonzero(
                        differences > comparison_absolute_tolerance
                    )
                )
                max_abs_difference[alias] = (
                    float(differences.max()) if differences.size else None
                )
                replacement_count[alias] = int(covered.sum())
                replacement = existing.copy()
                replacement[covered] = incoming[covered]
                output[alias] = replacement
        else:
            for alias in selected.columns:
                incoming = aligned[alias].to_numpy(dtype=float)
                # Collisions were rejected above, so every ordinary source
                # only adds a new explicit covariate.
                output[alias] = incoming
        source_audits.append(
            {
                **source.audit_dict(),
                "rows": int(len(selected)),
                "first_timestamp_utc": source_index[0].isoformat(),
                "last_timestamp_utc": source_index[-1].isoformat(),
                "matched_rows": int(aligned.notna().all(axis=1).sum()),
                "missing_rows": int(aligned.isna().any(axis=1).sum()),
                "replacement_policy": replacement_policy,
                "comparison_absolute_tolerance": (
                    comparison_absolute_tolerance
                    if authoritative_residual_source
                    else None
                ),
                "overlap_count": overlap_count,
                "mismatch_count": mismatch_count,
                "max_abs_difference": max_abs_difference,
                "replacement_count": replacement_count,
                **provenance_audit,
            }
        )
    missing_inputs = sorted(
        set(configuration.covariate_config.input_columns).difference(output.columns)
    )
    if missing_inputs:
        raise KalmanConfigurationError(
            f"Covariables Kalman brutes absentes apres jointure: {missing_inputs}."
        )
    if future is not None:
        future_values = output.loc[
            future, list(configuration.covariate_config.input_columns)
        ].apply(pd.to_numeric, errors="coerce")
        if configuration.covariate_config.require_future_complete and not np.isfinite(
            future_values.to_numpy(dtype=float)
        ).all():
            missing_columns = future_values.columns[
                future_values.isna().any(axis=0)
            ].astype(str).tolist()
            raise KalmanConfigurationError(
                "Covariables Kalman futures incompletes: "
                f"{missing_columns}."
            )
    output.index = pd.RangeIndex(len(output))
    audit = {
        **configuration.audit_dict(),
        "base_rows": int(len(output)),
        "base_first_timestamp_utc": base_index[0].isoformat(),
        "base_last_timestamp_utc": base_index[-1].isoformat(),
        "sources": source_audits,
    }
    return output, audit


__all__ = [
    "KALMAN_OPERATIONAL_CONFIG_VERSION",
    "KALMAN_WEATHER_ZONES",
    "KalmanAdditionalSource",
    "KalmanConfigurationError",
    "KalmanOperationalConfiguration",
    "KalmanUpstreamHistory",
    "attach_kalman_upstream_history",
    "attach_additional_kalman_sources",
    "load_kalman_operational_configuration",
    "render_kalman_weather_operational_configuration",
]
