"""Safe, in-memory overlays for point-in-time input experiments.

The production YAML files are read-only inputs to this module.  Resolving an
experiment returns a deep-copied configuration whose output is confined to
``runs/experiments``; it never writes a configuration, an artefact, or live
history to disk.

Only local hourly numeric PIT parquet inputs are accepted.  There is no hook
for Python callables, expressions, transforms, Storm, or MKOnline forecasts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


SCHEMA_VERSION = 1
SUPPORTED_ZONES = ("FR", "DE", "BE", "NL", "ES")
SUPPORTED_KINDS = frozenset({"hourly_numeric_pit"})
FORBIDDEN_INPUT_TOKENS = ("storm", "mkonline")

_ROOT_KEYS = {
    "schema_version",
    "experiment_id",
    "production_configs",
    "output_directory",
    "series",
}
_SERIES_KEYS = {
    "enabled",
    "zones",
    "kind",
    "timezone",
    "series",
    "pit_file",
    "columns",
    "minimum_coverage",
}
_COLUMN_KEYS = {"delivery", "value", "availability", "revision"}
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExperimentContractError(ValueError):
    """Raised when an input experiment is unsafe or structurally invalid."""


@dataclass(frozen=True)
class ExperimentInput:
    """One local, numeric, hourly point-in-time covariate declaration."""

    alias: str
    enabled: bool
    zones: tuple[str, ...]
    kind: str
    timezone: str
    series: str
    pit_file: Path
    delivery_column: str
    value_column: str
    availability_column: str
    revision_column: str
    minimum_coverage: float

    def applies_to(self, zone: str) -> bool:
        """Return whether this input is enabled for ``zone``."""

        return self.enabled and str(zone).strip().upper() in self.zones


@dataclass(frozen=True)
class ExperimentContract:
    """Validated experiment declaration with all paths resolved."""

    schema_version: int
    experiment_id: str
    source_path: Path
    project_root: Path
    production_configs: Mapping[str, Path]
    output_directory: Path
    inputs: tuple[ExperimentInput, ...]

    @property
    def enabled_aliases(self) -> tuple[str, ...]:
        return tuple(item.alias for item in self.inputs if item.enabled)

    def inputs_for_zone(self, zone: str) -> tuple[ExperimentInput, ...]:
        code = _zone(zone)
        return tuple(item for item in self.inputs if item.applies_to(code))


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """An isolated production configuration overlaid only in memory."""

    experiment_id: str
    zone: str
    production_config: Path
    production_config_directory: Path
    output_directory: Path
    enabled_aliases: tuple[str, ...]
    config: dict[str, Any]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentContractError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentContractError(f"{name} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ExperimentContractError(f"{name} has an invalid schema: {', '.join(details)}")


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{name} must be a non-empty string")
    return value.strip()


def _forbid_competing_forecasts(value: str, *, name: str) -> None:
    normalized = value.casefold()
    found = [token for token in FORBIDDEN_INPUT_TOKENS if token in normalized]
    if found:
        raise ExperimentContractError(
            f"{name} contains forbidden competing-forecast input token(s): {found}"
        )


def _alias(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    if not _ALIAS_RE.fullmatch(result):
        raise ExperimentContractError(
            f"{name} must be a lower-case snake_case alias"
        )
    if result == "target":
        raise ExperimentContractError(f"{name} cannot replace the target")
    _forbid_competing_forecasts(result, name=name)
    return result


def _zone(value: Any) -> str:
    if not isinstance(value, str):
        raise ExperimentContractError("zone must be a string")
    code = value.strip().upper()
    if code not in SUPPORTED_ZONES:
        raise ExperimentContractError(
            f"unsupported zone={value!r}; allowed={list(SUPPORTED_ZONES)}"
        )
    return code


def _timezone(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    try:
        ZoneInfo(result)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ExperimentContractError(f"{name} is not a valid IANA timezone") from exc
    # PIT parquet timestamps are normalized to UTC by the existing loader.
    # Requiring this declaration prevents a local-naive DST interpretation.
    if result != "UTC":
        raise ExperimentContractError(
            f"{name} must be UTC for hourly_numeric_pit inputs"
        )
    return result


def _resolve_inside(
    value: Any,
    *,
    base: Path,
    root: Path,
    name: str,
    allow_root: bool = False,
) -> Path:
    raw = _text(value, name=name)
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else base / path).resolve()
    allowed_root = root.resolve()
    if resolved == allowed_root:
        if allow_root:
            return resolved
        raise ExperimentContractError(f"{name} must be below {allowed_root}")
    if not resolved.is_relative_to(allowed_root):
        raise ExperimentContractError(f"{name} must stay below {allowed_root}")
    return resolved


def _load_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentContractError(f"{name} is not readable YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ExperimentContractError(f"{name} must contain a YAML mapping: {path}")
    return payload


def _parse_input(
    raw_alias: Any,
    raw_value: Any,
    *,
    project_root: Path,
) -> ExperimentInput:
    alias = _alias(raw_alias, name="series alias")
    item = _mapping(raw_value, name=f"series.{alias}")
    _exact_keys(item, _SERIES_KEYS, name=f"series.{alias}")

    enabled = item["enabled"]
    if not isinstance(enabled, bool):
        raise ExperimentContractError(f"series.{alias}.enabled must be true or false")

    raw_zones = _sequence(item["zones"], name=f"series.{alias}.zones")
    zones = tuple(_zone(value) for value in raw_zones)
    if not zones:
        raise ExperimentContractError(f"series.{alias}.zones must not be empty")
    if len(zones) != len(set(zones)):
        raise ExperimentContractError(f"series.{alias}.zones contains duplicates")

    kind = _text(item["kind"], name=f"series.{alias}.kind")
    if kind not in SUPPORTED_KINDS:
        raise ExperimentContractError(
            f"series.{alias}.kind={kind!r} is unsupported; allowed={sorted(SUPPORTED_KINDS)}"
        )
    timezone = _timezone(item["timezone"], name=f"series.{alias}.timezone")
    series_name = _text(item["series"], name=f"series.{alias}.series")
    _forbid_competing_forecasts(series_name, name=f"series.{alias}.series")

    pit_root = (project_root / "data" / "pit" / "vintages").resolve()
    pit_file = _resolve_inside(
        item["pit_file"],
        base=project_root,
        root=pit_root,
        name=f"series.{alias}.pit_file",
    )
    _forbid_competing_forecasts(str(pit_file), name=f"series.{alias}.pit_file")
    if pit_file.suffix.casefold() not in {".parquet", ".pq"}:
        raise ExperimentContractError(
            f"series.{alias}.pit_file must be a parquet file"
        )
    if enabled and not pit_file.is_file():
        raise ExperimentContractError(
            f"series.{alias}.pit_file is missing: {pit_file}"
        )

    columns = _mapping(item["columns"], name=f"series.{alias}.columns")
    _exact_keys(columns, _COLUMN_KEYS, name=f"series.{alias}.columns")
    parsed_columns: dict[str, str] = {}
    for role in sorted(_COLUMN_KEYS):
        column = _text(columns[role], name=f"series.{alias}.columns.{role}")
        if not _COLUMN_RE.fullmatch(column):
            raise ExperimentContractError(
                f"series.{alias}.columns.{role} is not a plain column name"
            )
        _forbid_competing_forecasts(
            column,
            name=f"series.{alias}.columns.{role}",
        )
        parsed_columns[role] = column
    if len(set(parsed_columns.values())) != len(parsed_columns):
        raise ExperimentContractError(f"series.{alias}.columns must be distinct")

    coverage = item["minimum_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ExperimentContractError(
            f"series.{alias}.minimum_coverage must be numeric"
        )
    minimum_coverage = float(coverage)
    if not 0.0 < minimum_coverage <= 1.0:
        raise ExperimentContractError(
            f"series.{alias}.minimum_coverage must be in ]0, 1]"
        )

    return ExperimentInput(
        alias=alias,
        enabled=enabled,
        zones=zones,
        kind=kind,
        timezone=timezone,
        series=series_name,
        pit_file=pit_file,
        delivery_column=parsed_columns["delivery"],
        value_column=parsed_columns["value"],
        availability_column=parsed_columns["availability"],
        revision_column=parsed_columns["revision"],
        minimum_coverage=minimum_coverage,
    )


def load_experiment_contract(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentContract:
    """Load and validate the strict YAML declaration without writing files."""

    source_path = Path(path).expanduser().resolve()
    payload = _load_yaml_mapping(source_path, name="experiment contract")
    _exact_keys(payload, _ROOT_KEYS, name="experiment contract")

    version = payload["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ExperimentContractError(
            f"schema_version={version!r}; expected={SCHEMA_VERSION}"
        )

    experiment_id = _alias(payload["experiment_id"], name="experiment_id")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else source_path.parent.parent.resolve()
    )

    raw_production = _mapping(payload["production_configs"], name="production_configs")
    normalized_keys = {str(key).strip().upper() for key in raw_production}
    if normalized_keys != set(SUPPORTED_ZONES) or len(raw_production) != len(SUPPORTED_ZONES):
        raise ExperimentContractError(
            "production_configs must declare exactly FR, DE, BE, NL and ES"
        )
    production_configs: dict[str, Path] = {}
    for zone in SUPPORTED_ZONES:
        raw_key = next(key for key in raw_production if str(key).strip().upper() == zone)
        config_path = _resolve_inside(
            raw_production[raw_key],
            base=root,
            root=root,
            name=f"production_configs.{zone}",
        )
        if not config_path.is_file():
            raise ExperimentContractError(
                f"production_configs.{zone} is missing: {config_path}"
            )
        if config_path.suffix.casefold() not in {".yaml", ".yml"}:
            raise ExperimentContractError(
                f"production_configs.{zone} must be YAML"
            )
        production_configs[zone] = config_path

    experiments_root = (root / "runs" / "experiments").resolve()
    output_directory = _resolve_inside(
        payload["output_directory"],
        base=root,
        root=experiments_root,
        name="output_directory",
    )
    if output_directory.name != experiment_id:
        raise ExperimentContractError(
            "output_directory must end with experiment_id"
        )

    raw_series = _mapping(payload["series"], name="series")
    if not raw_series:
        raise ExperimentContractError("series must declare at least one input")
    inputs = tuple(
        _parse_input(alias, item, project_root=root)
        for alias, item in raw_series.items()
    )
    aliases = [item.alias for item in inputs]
    if len(aliases) != len(set(aliases)):
        raise ExperimentContractError("series aliases must be unique")

    return ExperimentContract(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        source_path=source_path,
        project_root=root,
        production_configs=production_configs,
        output_directory=output_directory,
        inputs=inputs,
    )


def validate_pit_input_file(item: ExperimentInput) -> None:
    """Verify the declared local parquet schema without network access."""

    if not item.enabled:
        return
    try:
        import pandas as pd
        from pandas.api.types import is_numeric_dtype
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ExperimentContractError("pandas is required to inspect PIT inputs") from exc

    requested = [
        item.delivery_column,
        item.value_column,
        item.availability_column,
        item.revision_column,
    ]
    try:
        frame = pd.read_parquet(item.pit_file, columns=requested)
    except Exception as exc:
        raise ExperimentContractError(
            f"{item.alias}: PIT parquet is unreadable or misses declared columns"
        ) from exc
    if frame.empty:
        raise ExperimentContractError(f"{item.alias}: PIT parquet is empty")
    if not is_numeric_dtype(frame[item.value_column].dtype):
        raise ExperimentContractError(
            f"{item.alias}.{item.value_column} must have a numeric dtype"
        )
    if frame[item.value_column].notna().sum() == 0:
        raise ExperimentContractError(
            f"{item.alias}.{item.value_column} contains no numeric value"
        )

    for role, column in (
        ("delivery", item.delivery_column),
        ("availability", item.availability_column),
        ("revision", item.revision_column),
    ):
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if parsed.notna().sum() == 0:
            raise ExperimentContractError(
                f"{item.alias}.{role} contains no valid timestamp"
            )
        if role == "delivery":
            valid = parsed.dropna()
            off_hour = (
                (valid.dt.minute != 0)
                | (valid.dt.second != 0)
                | (valid.dt.microsecond != 0)
            )
            if bool(off_hour.any()):
                raise ExperimentContractError(
                    f"{item.alias}.delivery must contain hourly timestamps"
                )


def validate_experiment_inputs(contract: ExperimentContract) -> None:
    """Inspect every enabled PIT file declared by ``contract``."""

    for item in contract.inputs:
        validate_pit_input_file(item)


def _zone_mapping(config: Mapping[str, Any], zone: str) -> Mapping[str, Any]:
    zones = config.get("zones")
    if not isinstance(zones, Mapping):
        raise ExperimentContractError("production config must define zones")
    raw = zones.get(zone)
    if not isinstance(raw, Mapping):
        raise ExperimentContractError(
            f"production config does not define zone {zone}"
        )
    return raw


def overlay_experiment_inputs(
    contract: ExperimentContract,
    *,
    zone: str,
    production_config: Mapping[str, Any],
    production_config_path: str | Path | None = None,
) -> ResolvedExperimentConfig:
    """Return an isolated deep copy with enabled inputs and safe output paths."""

    code = _zone(zone)
    original_zone = _zone_mapping(production_config, code)
    expected_timezone = {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }[code]
    if str(original_zone.get("timezone", "")).strip() != expected_timezone:
        raise ExperimentContractError(
            f"production config zone {code} must use timezone {expected_timezone}"
        )

    resolved = copy.deepcopy(dict(production_config))
    zone_config = resolved["zones"][code]
    covariates = zone_config.setdefault("covariates", {})
    if not isinstance(covariates, dict):
        raise ExperimentContractError(
            f"production config zones.{code}.covariates must be a mapping"
        )

    enabled = contract.inputs_for_zone(code)
    for item in enabled:
        if item.alias in covariates:
            raise ExperimentContractError(
                f"{code}/{item.alias}: experiment alias already exists in production config"
            )
        covariates[item.alias] = {
            "enabled": True,
            "source": "pit_parquet",
            "series": item.series,
            "pit_file": str(item.pit_file),
            "timestamp_col": item.delivery_column,
            "value_col": item.value_column,
            "availability_col": item.availability_column,
            "revision_col": item.revision_column,
            "fill_method": "none",
            "fill_limit": 0,
            "minimum_coverage": item.minimum_coverage,
            "include_base_context": True,
            "future": {"known_future": True, "strategies": ["oracle"]},
        }

    data = resolved.setdefault("data", {})
    if not isinstance(data, dict):
        raise ExperimentContractError("production config data must be a mapping")
    pit_files = data.setdefault("pit_files", {})
    if not isinstance(pit_files, dict):
        raise ExperimentContractError("production config data.pit_files must be a mapping")
    for item in enabled:
        if item.alias in pit_files:
            raise ExperimentContractError(
                f"{code}/{item.alias}: experiment PIT alias already exists in production config"
            )
        pit_files[item.alias] = str(item.pit_file)

    output_directory = (contract.output_directory / code.lower()).resolve()
    if not output_directory.is_relative_to(contract.output_directory):
        raise ExperimentContractError("resolved experiment output escaped its root")
    output = resolved.setdefault("output", {})
    if not isinstance(output, dict):
        raise ExperimentContractError("production config output must be a mapping")
    output["directory"] = str(output_directory)

    source_path = (
        Path(production_config_path).expanduser().resolve()
        if production_config_path is not None
        else contract.production_configs[code]
    )
    return ResolvedExperimentConfig(
        experiment_id=contract.experiment_id,
        zone=code,
        production_config=source_path,
        production_config_directory=source_path.parent,
        output_directory=output_directory,
        enabled_aliases=tuple(item.alias for item in enabled),
        config=resolved,
    )


def resolve_experiment_config(
    contract: ExperimentContract,
    *,
    zone: str,
    validate_inputs: bool = True,
) -> ResolvedExperimentConfig:
    """Read one production YAML and resolve its experiment only in memory."""

    code = _zone(zone)
    if validate_inputs:
        for item in contract.inputs_for_zone(code):
            validate_pit_input_file(item)
    source_path = contract.production_configs[code]
    production = _load_yaml_mapping(source_path, name=f"production config {code}")
    return overlay_experiment_inputs(
        contract,
        zone=code,
        production_config=production,
        production_config_path=source_path,
    )


__all__ = [
    "ExperimentContract",
    "ExperimentContractError",
    "ExperimentInput",
    "ResolvedExperimentConfig",
    "SCHEMA_VERSION",
    "SUPPORTED_KINDS",
    "SUPPORTED_ZONES",
    "load_experiment_contract",
    "overlay_experiment_inputs",
    "resolve_experiment_config",
    "validate_experiment_inputs",
    "validate_pit_input_file",
]
