"""Strict, immutable contract for a zone-specific operational model bundle.

This module is deliberately independent from the sealed France runner.  It is
the configuration boundary for future multi-zone runners: all market identity,
model artefacts, hashes and blend weights must be declared explicitly in both
the live configuration and the zone registry where applicable.  Nothing is
inferred from France and there is no permissive fallback mode.

Expected ``live`` keys in a new zone configuration::

    contract_schema_version: 1
    zone: BE
    delivery_timezone: Europe/Brussels
    forecast_origin_timezone: Europe/Paris
    forecast_origin_local_time: "08:00"
    target_series: power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh
    prediction_mode: autonomous_only
    mkonline_enabled: false
    primary_series: null
    storm_dashboard_series: power.price.be.euromwh.h.fcst.3mv.storm  # or null
    storm_dashboard_primary_series: 41378_native                    # or null
    storm_dashboard_naive_timezone: Europe/Brussels                 # or null
    storm_strict_08_series: power.price.be.euromwh.h.fcst.3mv.storm.da.basecase
    forecast_filename: forecast_hourly_be.csv
    required_covariates: [fr_residual_load_fcst, ...]
    base_config: ...
    frozen_autonomous_run: ...
    sealed_benchmark_run: ...
    recipe_manifest: ...
    dependency_manifest: null
    output_root: ...
    expected_hashes:
      base_config_sha256: <64 lowercase/uppercase hexadecimal characters>
      frozen_autonomous_checksum_manifest_sha256: <sha256>
      sealed_benchmark_checksum_manifest_sha256: <sha256>
      recipe_manifest_sha256: <sha256>
      dependency_manifest_sha256: null
    weights:
      autonomous: 1.0
      mkonline_primary: 0.0

The registry must explicitly repeat the identity fields, required covariates
and nullable ``storm_dashboard_series``.  This duplication is intentional: a
cross-zone copy/paste error fails before any target, model or Saturn series is
read by a prediction path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE,
    STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE,
    STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE,
    storm_dashboard_series,
)
from chronos2_hourly.zone_live import (
    MARKET_ZONES,
    ZoneBundleError,
    canonical_zone,
    load_zone_registry,
)
from chronos2_modular.common import load_yaml


CONTRACT_SCHEMA_VERSION = 1

# Only the frozen algorithmic implementation executed by
# ``multizone_live._train_and_predict_autonomous`` belongs here.  In
# particular, application, reporting, history and Storm-comparison code are
# deliberately excluded: those components run after the candidate has been
# frozen and may evolve without changing the model forecast.  Every path must
# also be present as ``role=source_code`` in the sealed extended-run checksum
# manifest.
ALGORITHM_SOURCE_ROOT = Path(__file__).resolve().parents[1]
LIVE_REFIT_SOURCE_CODE_PATHS: tuple[str, ...] = (
    "run_extended_residual_hourly.py",
    "chronos2_hourly/features.py",
    "chronos2_hourly/hourly_contract.py",
    "chronos2_hourly/models/__init__.py",
    "chronos2_hourly/models/base.py",
    "chronos2_hourly/models/blended_residual_corrector.py",
    "chronos2_hourly/models/calibration.py",
    "chronos2_hourly/models/catboost_hourly.py",
    "chronos2_hourly/models/ensemble.py",
    "chronos2_hourly/models/lear.py",
    "chronos2_hourly/models/residual_corrector.py",
)

HASH_KEYS = (
    "base_config_sha256",
    "frozen_autonomous_checksum_manifest_sha256",
    "sealed_benchmark_checksum_manifest_sha256",
    "recipe_manifest_sha256",
    "dependency_manifest_sha256",
)

# These are fingerprints of the already sealed France bundle.  Renaming a
# French directory is not sufficient to make it a valid foreign-zone model.
FR_SEALED_ARTIFACT_HASHES = frozenset(
    {
        "f1127f26bcdd8f5c62dd714fa319c024068327fa1284b3d183fa7872bdeaeea2",
        "5de35822e7b2a40fe9f7a45a906268b38f29fce81f04d1049e084de4618ee915",
        "9fa7f86a4550b64dd490e48297abd8a7eabd360925f39c75cbdb4654c84f8543",
        "2dae7a579aee1c1f8bb2e96b70e6e0224a7e8b133e5a9e5f836dee5a646f53cc",
        "e08eacdabf7d3c9d03e061706b20839ee75689138c682a22880e388de7b466bc",
        "e5af979b842ffcdb717be6e0d5eed190a12166c5d00456c4c6f1543c672f24dd",
    }
)
FR_PRIMARY_SERIES = "41551_native"
FR_TARGET_SERIES = "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh"
FR_DASHBOARD_SERIES = "power.price.fr.euromwh.h.fcst.3mv.storm"
FR_BLEND_WEIGHTS = (0.4977609282924196, 0.5022390717075804)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FR_TOKEN_RE = re.compile(r"(?:^|[_.-])fr(?:$|[_.-])", re.IGNORECASE)


class ZoneModelContractError(ValueError):
    """Raised when a zone bundle is incomplete, inconsistent or cross-wired."""


@dataclass(frozen=True)
class ZoneModelPaths:
    """Resolved paths consumed or produced by one operational zone bundle."""

    live_config: Path
    registry: Path
    base_config: Path
    frozen_autonomous_run: Path
    sealed_benchmark_run: Path
    recipe_manifest: Path
    dependency_manifest: Path | None
    output_root: Path


@dataclass(frozen=True)
class ZoneArtifactHashes:
    """Expected SHA-256 fingerprints for every frozen model boundary."""

    base_config_sha256: str
    frozen_autonomous_checksum_manifest_sha256: str
    sealed_benchmark_checksum_manifest_sha256: str
    recipe_manifest_sha256: str
    dependency_manifest_sha256: str | None


@dataclass(frozen=True)
class ZoneBlendWeights:
    """Frozen convex blend weights selected independently for one zone."""

    autonomous: float
    mkonline_primary: float


@dataclass(frozen=True)
class ZoneModelContract:
    """Fully verified, immutable description of one market-zone model."""

    schema_version: int
    zone: str
    delivery_timezone: str
    forecast_origin_timezone: str
    forecast_origin_local_time: str
    target_series: str
    primary_series: str | None
    storm_dashboard_series: str | None
    storm_dashboard_primary_series: str | None
    storm_dashboard_naive_timezone: str | None
    storm_strict_08_series: str | None
    forecast_filename: str
    required_covariates: tuple[str, ...]
    paths: ZoneModelPaths
    checksum_hashes: ZoneArtifactHashes
    weights: ZoneBlendWeights
    prediction_mode: str = "mkonline_blend"
    mkonline_enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit representation."""

        result = asdict(self)
        result["paths"] = {
            key: (str(value) if value is not None else None)
            for key, value in result["paths"].items()
        }
        return result


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ZoneModelContractError(f"{name} must be an explicit mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, name: str) -> Any:
    if key not in mapping:
        raise ZoneModelContractError(f"{name}.{key} must be declared explicitly")
    return mapping[key]


def _text(value: Any, *, name: str) -> str:
    if value is None or not str(value).strip():
        raise ZoneModelContractError(f"{name} must be a non-empty string")
    return str(value).strip()


def _nullable_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ZoneModelContractError(f"{name} must be an explicit list")
    result = tuple(_text(item, name=f"{name}[]").lower() for item in value)
    if not result:
        raise ZoneModelContractError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ZoneModelContractError(f"{name} contains duplicate aliases")
    if any("storm" in alias.casefold() for alias in result):
        raise ZoneModelContractError(
            f"{name} contains Storm; Storm is evaluation-only"
        )
    return result


def _resolve(value: Any, *, base: Path, name: str) -> Path:
    raw = _text(value, name=name)
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ZoneModelContractError(f"{name} is missing: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZoneModelContractError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ZoneModelContractError(f"{name} must contain a JSON object: {path}")
    return result


def _expected_hashes(
    value: Any,
    *,
    dependency_required: bool = True,
) -> ZoneArtifactHashes:
    raw = _mapping(value, name="live.expected_hashes")
    normalized: dict[str, str] = {}
    for key in HASH_KEYS:
        raw_value = _required(raw, key, name="live.expected_hashes")
        if key == "dependency_manifest_sha256" and not dependency_required:
            if raw_value is not None:
                raise ZoneModelContractError(
                    "live.expected_hashes.dependency_manifest_sha256 must be null "
                    "for autonomous_only"
                )
            normalized[key] = None  # type: ignore[assignment]
            continue
        candidate = _text(
            raw_value,
            name=f"live.expected_hashes.{key}",
        )
        if not _SHA256_RE.fullmatch(candidate):
            raise ZoneModelContractError(
                f"live.expected_hashes.{key} must be a SHA-256 hexadecimal digest"
            )
        normalized[key] = candidate.lower()
    return ZoneArtifactHashes(**normalized)  # type: ignore[arg-type]


def _weights(value: Any) -> ZoneBlendWeights:
    raw = _mapping(value, name="live.weights")
    values: dict[str, float] = {}
    for key in ("autonomous", "mkonline_primary"):
        candidate = _required(raw, key, name="live.weights")
        try:
            number = float(candidate)
        except (TypeError, ValueError) as exc:
            raise ZoneModelContractError(
                f"live.weights.{key} must be numeric"
            ) from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ZoneModelContractError(
                f"live.weights.{key} must be finite and between 0 and 1"
            )
        values[key] = number
    if not math.isclose(
        values["autonomous"] + values["mkonline_primary"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ZoneModelContractError("live.weights must sum exactly to one")
    return ZoneBlendWeights(**values)


def _assert_equal(observed: Any, expected: Any, *, name: str) -> None:
    if observed != expected:
        raise ZoneModelContractError(
            f"{name} mismatch: {observed!r} != {expected!r}"
        )


def _validate_timezone(value: str, *, name: str) -> None:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ZoneModelContractError(f"{name} is not a valid IANA timezone") from exc


def _validate_origin_time(value: str) -> None:
    try:
        hour, minute = value.split(":", maxsplit=1)
        valid = (
            len(hour) == 2
            and len(minute) == 2
            and 0 <= int(hour) <= 23
            and 0 <= int(minute) <= 59
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ZoneModelContractError(
            "forecast_origin_local_time must use the explicit HH:MM format"
        )


def _relative_artifact(item: Mapping[str, Any]) -> str | None:
    raw = str(item.get("path", "")).replace("\\", "/")
    candidate = Path(raw)
    if (
        not raw
        or candidate.is_absolute()
        or any(part == ".." for part in candidate.parts)
    ):
        return None
    return candidate.as_posix()


def _verified_run_manifest(
    run_dir: Path,
    *,
    expected_checksum_sha256: str,
    zone: str,
    timezone: str,
    label: str,
    forecast_filename: str,
    required_live_inputs: Sequence[str] = (),
    required_source_code: Sequence[str] = (),
) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ZoneModelContractError(f"{label} is missing: {run_dir}")
    checksum_path = run_dir / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise ZoneModelContractError(
            f"{label} checksum manifest is missing: {checksum_path}"
        )
    observed_checksum = _sha256(checksum_path)
    _assert_equal(
        observed_checksum,
        expected_checksum_sha256,
        name=f"{label} checksum manifest SHA-256",
    )
    checksum = _json_object(checksum_path, name=f"{label} checksum manifest")
    if str(checksum.get("algorithm", "")).lower() != "sha256":
        raise ZoneModelContractError(f"{label} checksum algorithm must be sha256")
    artifacts = checksum.get("artifacts")
    if not isinstance(artifacts, list):
        raise ZoneModelContractError(f"{label} checksum artifacts must be a list")
    declarations: dict[str, Mapping[str, Any]] = {}
    source_code_declarations: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        if role not in {
            "run_artifact",
            "materialized_input",
            "source_code",
        }:
            continue
        relative = _relative_artifact(item)
        if relative is None:
            raise ZoneModelContractError(
                f"{label} has unsafe checksummed artifact path: "
                f"{item.get('path')!r}"
            )
        target = (
            source_code_declarations if role == "source_code" else declarations
        )
        if relative in target:
            raise ZoneModelContractError(
                f"{label} has duplicate checksum declaration for {relative}"
            )
        target[relative] = item
    required_roles = {
        "run_manifest.json": "run_artifact",
        "backtest_hourly_oof.csv.gz": "run_artifact",
        forecast_filename: "run_artifact",
        **{
            relative: (
                "materialized_input"
                if relative.startswith("inputs/")
                else "run_artifact"
            )
            for relative in required_live_inputs
        },
    }
    for relative, required_role in required_roles.items():
        item = declarations.get(relative)
        if item is None:
            raise ZoneModelContractError(
                f"{label} does not checksum required {required_role} {relative}"
            )
        if item.get("role") != required_role:
            raise ZoneModelContractError(
                f"{label} checksum role mismatch for {relative}: "
                f"{item.get('role')!r} != {required_role!r}"
            )
        artifact = run_dir / relative
        if not artifact.is_file():
            raise ZoneModelContractError(f"{label} artifact is missing: {artifact}")
        expected = str(item.get("sha256", "")).lower()
        if not _SHA256_RE.fullmatch(expected) or _sha256(artifact) != expected:
            raise ZoneModelContractError(
                f"{label} artifact checksum mismatch: {relative}"
            )
    source_root = ALGORITHM_SOURCE_ROOT.resolve()
    for raw_relative in required_source_code:
        relative = _relative_artifact({"path": raw_relative})
        if relative is None or relative != raw_relative.replace("\\", "/"):
            raise ZoneModelContractError(
                f"{label} requires an unsafe source_code path: {raw_relative!r}"
            )
        item = source_code_declarations.get(relative)
        if item is None:
            raise ZoneModelContractError(
                f"{label} does not checksum required source_code {relative}"
            )
        source_path = (source_root / relative).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError as exc:
            raise ZoneModelContractError(
                f"{label} source_code escapes the project root: {relative}"
            ) from exc
        if not source_path.is_file():
            raise ZoneModelContractError(
                f"{label} source_code is missing: {relative}"
            )
        expected = str(item.get("sha256", "")).lower()
        if not _SHA256_RE.fullmatch(expected) or _sha256(source_path) != expected:
            raise ZoneModelContractError(
                f"{label} source_code checksum mismatch: {relative}"
            )
    manifest = _json_object(run_dir / "run_manifest.json", name=f"{label} manifest")
    _assert_equal(manifest.get("zone"), zone, name=f"{label} manifest.zone")
    _assert_equal(
        manifest.get("timezone"),
        timezone,
        name=f"{label} manifest.timezone",
    )
    if manifest.get("storm_used_as_feature") is not False:
        raise ZoneModelContractError(
            f"{label} manifest.storm_used_as_feature must be explicitly false"
        )
    prediction_inputs = manifest.get("prediction_inputs", [])
    if not isinstance(prediction_inputs, Sequence) or isinstance(
        prediction_inputs, (str, bytes)
    ):
        raise ZoneModelContractError(
            f"{label} manifest.prediction_inputs must be an explicit list"
        )
    if any("storm" in str(value).casefold() for value in prediction_inputs):
        raise ZoneModelContractError(
            f"{label} manifest.prediction_inputs contains Storm"
        )
    return manifest


def _validate_base_config(
    path: Path,
    *,
    zone: str,
    timezone: str,
    target_series: str,
    required_covariates: tuple[str, ...],
) -> None:
    if not path.is_file():
        raise ZoneModelContractError(f"base_config is missing: {path}")
    config = load_yaml(path)
    zones = _mapping(config.get("zones"), name="base_config.zones")
    if set(str(key).upper() for key in zones) != {zone}:
        raise ZoneModelContractError(
            f"base_config must contain exactly zones.{zone}; no zone fallback is allowed"
        )
    raw_zone = _mapping(zones.get(zone), name=f"base_config.zones.{zone}")
    _assert_equal(
        raw_zone.get("timezone"),
        timezone,
        name=f"base_config.zones.{zone}.timezone",
    )
    target = _mapping(raw_zone.get("target"), name=f"base_config.zones.{zone}.target")
    _assert_equal(
        target.get("series"),
        target_series,
        name=f"base_config.zones.{zone}.target.series",
    )
    covariates = _mapping(
        raw_zone.get("covariates"),
        name=f"base_config.zones.{zone}.covariates",
    )
    active_covariates: dict[str, Mapping[str, Any]] = {}
    for raw_alias, raw_definition in covariates.items():
        alias = str(raw_alias).strip().lower()
        if not isinstance(raw_definition, Mapping):
            raise ZoneModelContractError(
                f"base_config covariate {raw_alias!r} must be a mapping"
            )
        if bool(raw_definition.get("enabled", True)):
            active_covariates[alias] = raw_definition
    if set(active_covariates) != set(required_covariates):
        raise ZoneModelContractError(
            "base_config active covariates must exactly match "
            f"required_covariates: {sorted(active_covariates)} != "
            f"{sorted(required_covariates)}"
        )
    for alias in required_covariates:
        definition = active_covariates.get(alias)
        if definition is None:
            raise ZoneModelContractError(
                f"required covariate is missing or disabled: {alias}"
            )
        if "storm" in str(definition.get("series", "")).casefold():
            raise ZoneModelContractError(
                f"required covariate {alias} refers to Storm; Storm is evaluation-only"
            )


def _validate_recipe_and_dependency(
    *,
    recipe_path: Path,
    dependency_path: Path | None,
    zone: str,
    timezone: str,
    primary_series: str | None,
    frozen_run: Path,
    hashes: ZoneArtifactHashes,
    weights: ZoneBlendWeights,
    prediction_mode: str,
    mkonline_enabled: bool,
    base: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = _json_object(recipe_path, name="recipe_manifest")
    dependency = (
        _json_object(dependency_path, name="dependency_manifest")
        if dependency_path is not None
        else {}
    )
    _assert_equal(recipe.get("zone"), zone, name="recipe.zone")
    _assert_equal(recipe.get("timezone"), timezone, name="recipe.timezone")
    recipe_mode = recipe.get("prediction_mode", recipe.get("recipe_mode"))
    _assert_equal(
        recipe_mode,
        prediction_mode,
        name="recipe.prediction_mode/recipe_mode",
    )
    _assert_equal(
        recipe.get("mkonline_enabled"),
        mkonline_enabled,
        name="recipe.mkonline_enabled",
    )
    source_run = _resolve(
        _required(recipe, "source_autonomous_run", name="recipe"),
        base=base,
        name="recipe.source_autonomous_run",
    )
    _assert_equal(source_run, frozen_run, name="recipe.source_autonomous_run")
    _assert_equal(
        str(recipe.get("source_autonomous_checksum_manifest_sha256", "")).lower(),
        hashes.frozen_autonomous_checksum_manifest_sha256,
        name="recipe.source_autonomous_checksum_manifest_sha256",
    )
    recipe_weights = _mapping(recipe.get("weights"), name="recipe.weights")
    for key, expected in (
        ("autonomous", weights.autonomous),
        ("mkonline_primary", weights.mkonline_primary),
    ):
        try:
            observed = float(recipe_weights.get(key))
        except (TypeError, ValueError) as exc:
            raise ZoneModelContractError(f"recipe.weights.{key} is invalid") from exc
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ZoneModelContractError(
                f"recipe.weights.{key} mismatch: {observed!r} != {expected!r}"
            )
    if prediction_mode == "autonomous_only":
        validation = recipe.get("autonomous_validation")
        validation_reason = (
            str(validation.get("reason", "")).strip()
            if isinstance(validation, Mapping)
            else ""
        )
        if not (
            str(recipe.get("autonomous_only_reason", "")).strip()
            or validation_reason
        ):
            raise ZoneModelContractError(
                "recipe autonomous-only reason is required"
            )
        if primary_series is not None:
            raise ZoneModelContractError(
                "autonomous_only primary_series must be null"
            )
        if dependency_path is not None or hashes.dependency_manifest_sha256 is not None:
            raise ZoneModelContractError(
                "autonomous_only must not declare a dependency manifest"
            )
        if recipe.get("external_expert") not in (None, {}):
            raise ZoneModelContractError(
                "autonomous_only must not declare an external expert"
            )
        return recipe, dependency

    if primary_series is None or dependency_path is None:
        raise ZoneModelContractError(
            "mkonline_blend requires primary_series and dependency_manifest"
        )
    _assert_equal(dependency.get("zone"), zone, name="dependency.zone")
    _assert_equal(
        dependency.get("timezone"), timezone, name="dependency.timezone"
    )
    external = _mapping(recipe.get("external_expert"), name="recipe.external_expert")
    _assert_equal(
        external.get("series"), primary_series, name="recipe.external_expert.series"
    )
    if external.get("storm_used_as_feature") is not False:
        raise ZoneModelContractError(
            "recipe.external_expert.storm_used_as_feature must be false"
        )
    _assert_equal(
        str(external.get("dependency_manifest_sha256", "")).lower(),
        hashes.dependency_manifest_sha256,
        name="recipe.external_expert.dependency_manifest_sha256",
    )
    declared_dependency = _resolve(
        _required(external, "dependency_manifest", name="recipe.external_expert"),
        base=base,
        name="recipe.external_expert.dependency_manifest",
    )
    _assert_equal(
        declared_dependency,
        dependency_path,
        name="recipe.external_expert.dependency_manifest",
    )
    _assert_equal(
        dependency.get("terminal_series"),
        primary_series,
        name="dependency.terminal_series",
    )
    if dependency.get("terminal_type") != "primary":
        raise ZoneModelContractError("dependency.terminal_type must be 'primary'")
    if dependency.get("terminal_formula") is not None:
        raise ZoneModelContractError("dependency terminal must not be a formula")
    if dependency.get("storm_token_found") is not False:
        raise ZoneModelContractError("dependency is not certified Storm-free")
    if dependency.get("dependency_gate_passed") is not True:
        raise ZoneModelContractError("dependency gate has not passed")
    return recipe, dependency


def _reject_fr_reuse_outside_fr(
    *,
    zone: str,
    target_series: str,
    primary_series: str | None,
    dashboard_series: str | None,
    dashboard_primary_series: str | None,
    dashboard_naive_timezone: str | None,
    strict_08_series: str | None,
    paths: ZoneModelPaths,
    hashes: ZoneArtifactHashes,
    weights: ZoneBlendWeights,
    recipe: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> None:
    if zone == "FR":
        return
    if target_series == FR_TARGET_SERIES or ".fr." in target_series.casefold():
        raise ZoneModelContractError("a France target cannot be reused outside FR")
    if primary_series == FR_PRIMARY_SERIES:
        raise ZoneModelContractError("the France primary cannot be reused outside FR")
    if dashboard_series == FR_DASHBOARD_SERIES:
        raise ZoneModelContractError(
            "the France Storm dashboard series cannot be reused outside FR"
        )
    if dashboard_primary_series == STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE["FR"]:
        raise ZoneModelContractError(
            "the France Storm dashboard primary cannot be reused outside FR"
        )
    if strict_08_series and ".fr." in strict_08_series.casefold():
        raise ZoneModelContractError(
            "the France strict-08 Storm proxy cannot be reused outside FR"
        )
    if (
        dashboard_series is not None
        and dashboard_naive_timezone
        != STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE.get(zone)
    ):
        raise ZoneModelContractError(
            "the Storm dashboard naive timezone is cross-wired for the zone"
        )
    observed_weights = (weights.autonomous, weights.mkonline_primary)
    if all(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)
        for observed, expected in zip(observed_weights, FR_BLEND_WEIGHTS)
    ):
        raise ZoneModelContractError(
            "the sealed France blend weights cannot be reused outside FR"
        )
    if set(asdict(hashes).values()) & FR_SEALED_ARTIFACT_HASHES:
        raise ZoneModelContractError(
            "a sealed France artifact hash cannot be reused outside FR"
        )
    model_paths = (
        paths.base_config,
        paths.frozen_autonomous_run,
        paths.sealed_benchmark_run,
        paths.recipe_manifest,
        paths.dependency_manifest,
    )
    french_names = [
        path.name
        for path in model_paths
        if path is not None and _FR_TOKEN_RE.search(path.name)
    ]
    if french_names:
        raise ZoneModelContractError(
            "France-labelled model artifacts cannot be reused outside FR: "
            + ", ".join(french_names)
        )
    external = recipe.get("external_expert")
    wrapper = (
        str(external.get("wrapper_series", ""))
        if isinstance(external, Mapping)
        else ""
    )
    dependency_wrapper = str(dependency.get("wrapper_series", ""))
    metadata = dependency.get("terminal_metadata")
    country = (
        str(metadata.get("mercure:country", ""))
        if isinstance(metadata, Mapping)
        else ""
    )
    if ".fr." in wrapper.casefold() or ".fr." in dependency_wrapper.casefold():
        raise ZoneModelContractError(
            "a France primary wrapper cannot be reused outside FR"
        )
    if country.strip().upper() in {"FR", "FRANCE"}:
        raise ZoneModelContractError(
            "a France terminal primary cannot be reused outside FR"
        )


def load_zone_model_contract(
    live_config_path: str | Path,
    registry_path: str | Path,
    *,
    strict: bool = True,
) -> ZoneModelContract:
    """Load and verify one explicit zone bundle without any fallback.

    ``strict=False`` is intentionally unsupported.  The keyword exists to
    make the security posture visible at call sites and to prevent a future
    caller from assuming that an omitted check implies a valid live bundle.
    """

    if strict is not True:
        raise ZoneModelContractError(
            "non-strict zone model contracts are unsupported; no fallback is allowed"
        )
    live_path = Path(live_config_path).expanduser().resolve()
    registry_file = Path(registry_path).expanduser().resolve()
    if not live_path.is_file():
        raise ZoneModelContractError(f"live configuration is missing: {live_path}")
    config = load_yaml(live_path)
    live = _mapping(config.get("live"), name="live")
    raw_schema_version = _required(live, "contract_schema_version", name="live")
    if isinstance(raw_schema_version, bool) or not isinstance(
        raw_schema_version, int
    ):
        raise ZoneModelContractError(
            "live.contract_schema_version must be an explicit integer"
        )
    schema_version = raw_schema_version
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ZoneModelContractError(
            f"live.contract_schema_version={schema_version}, expected "
            f"{CONTRACT_SCHEMA_VERSION}"
        )

    raw_zone = _text(_required(live, "zone", name="live"), name="live.zone").upper()
    try:
        zone = canonical_zone(raw_zone)
    except ZoneBundleError as exc:
        raise ZoneModelContractError(str(exc)) from exc
    if zone != raw_zone:
        raise ZoneModelContractError(
            f"live.zone must use canonical code {zone!r}, not alias {raw_zone!r}"
        )
    try:
        registry, registry_dir = load_zone_registry(registry_file)
    except (OSError, ValueError, ZoneBundleError) as exc:
        raise ZoneModelContractError(
            f"zone registry is invalid: {registry_file}: {exc}"
        ) from exc
    registry_zones = _mapping(registry.get("zones"), name="registry.zones")
    registry_zone = _mapping(
        registry_zones.get(zone), name=f"registry.zones.{zone}"
    )
    if registry_zone.get("enabled") is not True:
        raise ZoneModelContractError(f"registry.zones.{zone}.enabled must be true")
    if registry_zone.get("production_ready") is not True:
        raise ZoneModelContractError(
            f"registry.zones.{zone}.production_ready must be true"
        )
    declared_blockers = registry_zone.get("blockers", [])
    if not isinstance(declared_blockers, Sequence) or isinstance(
        declared_blockers, (str, bytes)
    ):
        raise ZoneModelContractError(
            f"registry.zones.{zone}.blockers must be a list when declared"
        )
    remaining_blockers = [str(value).strip() for value in declared_blockers if str(value).strip()]
    if remaining_blockers:
        raise ZoneModelContractError(
            f"registry.zones.{zone} still declares blockers: "
            + "; ".join(remaining_blockers)
        )
    registry_live_config = _resolve(
        _required(
            registry_zone,
            "live_config",
            name=f"registry.zones.{zone}",
        ),
        base=registry_dir,
        name=f"registry.zones.{zone}.live_config",
    )
    _assert_equal(
        registry_live_config,
        live_path,
        name=f"registry.zones.{zone}.live_config",
    )

    delivery_timezone = _text(
        _required(live, "delivery_timezone", name="live"),
        name="live.delivery_timezone",
    )
    origin_timezone = _text(
        _required(live, "forecast_origin_timezone", name="live"),
        name="live.forecast_origin_timezone",
    )
    origin_time = _text(
        _required(live, "forecast_origin_local_time", name="live"),
        name="live.forecast_origin_local_time",
    )
    _validate_timezone(delivery_timezone, name="live.delivery_timezone")
    _validate_timezone(origin_timezone, name="live.forecast_origin_timezone")
    _validate_origin_time(origin_time)
    _assert_equal(
        delivery_timezone,
        MARKET_ZONES[zone].timezone,
        name="live.delivery_timezone versus market contract",
    )
    for key, observed in (
        ("delivery_timezone", delivery_timezone),
        ("forecast_origin_timezone", origin_timezone),
        ("forecast_origin_local_time", origin_time),
    ):
        _assert_equal(
            registry_zone.get(key),
            observed,
            name=f"registry.zones.{zone}.{key}",
        )

    target_series = _text(
        _required(live, "target_series", name="live"), name="live.target_series"
    )
    primary_series = _nullable_text(
        _required(live, "primary_series", name="live"), name="live.primary_series"
    )
    _assert_equal(
        registry_zone.get("target_series"),
        target_series,
        name=f"registry.zones.{zone}.target_series",
    )
    _assert_equal(
        registry_zone.get("primary_series"),
        primary_series,
        name=f"registry.zones.{zone}.primary_series",
    )
    if registry_zone.get("target_status") != "audited_dst_strict":
        raise ZoneModelContractError(
            f"registry.zones.{zone}.target_status must be audited_dst_strict"
        )
    expected_primary_status = (
        "audited_primary"
        if primary_series is not None
        else "not_used_autonomous_only"
    )
    if registry_zone.get("primary_status") != expected_primary_status:
        raise ZoneModelContractError(
            f"registry.zones.{zone}.primary_status must be {expected_primary_status}"
        )
    if registry_zone.get("price_unit") != "EUR/MWh":
        raise ZoneModelContractError(
            f"registry.zones.{zone}.price_unit must be EUR/MWh"
        )

    dashboard = _nullable_text(
        _required(live, "storm_dashboard_series", name="live"),
        name="live.storm_dashboard_series",
    )
    registry_dashboard = _nullable_text(
        _required(
            registry_zone,
            "storm_series",
            name=f"registry.zones.{zone}",
        ),
        name=f"registry.zones.{zone}.storm_series",
    )
    _assert_equal(
        registry_dashboard, dashboard, name=f"registry.zones.{zone}.storm_series"
    )
    dashboard_primary = _nullable_text(
        _required(live, "storm_dashboard_primary_series", name="live"),
        name="live.storm_dashboard_primary_series",
    )
    registry_dashboard_primary = _nullable_text(
        _required(
            registry_zone,
            "storm_primary_series",
            name=f"registry.zones.{zone}",
        ),
        name=f"registry.zones.{zone}.storm_primary_series",
    )
    _assert_equal(
        registry_dashboard_primary,
        dashboard_primary,
        name=f"registry.zones.{zone}.storm_primary_series",
    )
    dashboard_naive_timezone = _nullable_text(
        _required(live, "storm_dashboard_naive_timezone", name="live"),
        name="live.storm_dashboard_naive_timezone",
    )
    registry_dashboard_timezone = _nullable_text(
        _required(
            registry_zone,
            "storm_naive_timezone",
            name=f"registry.zones.{zone}",
        ),
        name=f"registry.zones.{zone}.storm_naive_timezone",
    )
    _assert_equal(
        registry_dashboard_timezone,
        dashboard_naive_timezone,
        name=f"registry.zones.{zone}.storm_naive_timezone",
    )
    strict_08_series = _nullable_text(
        _required(live, "storm_strict_08_series", name="live"),
        name="live.storm_strict_08_series",
    )
    registry_strict_08 = _nullable_text(
        _required(
            registry_zone,
            "storm_strict_08_series",
            name=f"registry.zones.{zone}",
        ),
        name=f"registry.zones.{zone}.storm_strict_08_series",
    )
    _assert_equal(
        registry_strict_08,
        strict_08_series,
        name=f"registry.zones.{zone}.storm_strict_08_series",
    )
    if strict_08_series is not None and zone in {"FR", "BE", "DE", "NL", "ES"}:
        _assert_equal(
            strict_08_series,
            f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.basecase",
            name="live.storm_strict_08_series versus zone strict-08 proxy",
        )
    if dashboard is not None:
        _assert_equal(
            registry_zone.get("storm_status"),
            "audited_native_dashboard",
            name=f"registry.zones.{zone}.storm_status",
        )
        try:
            verified_dashboard = storm_dashboard_series(zone)
        except ValueError as exc:
            raise ZoneModelContractError(
                f"{zone}: native Storm dashboard series has not been verified"
            ) from exc
        _assert_equal(
            dashboard,
            verified_dashboard,
            name="live.storm_dashboard_series versus verified native series",
        )
        _assert_equal(
            dashboard_primary,
            STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[zone],
            name="live.storm_dashboard_primary_series versus verified native primary",
        )
        _assert_equal(
            dashboard_naive_timezone,
            STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[zone],
            name="live.storm_dashboard_naive_timezone versus verified native timezone",
        )
    else:
        if zone in STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE:
            raise ZoneModelContractError(
                f"{zone}: verified native Storm dashboard declarations are mandatory"
            )
        _assert_equal(
            dashboard_primary,
            None,
            name="live.storm_dashboard_primary_series when dashboard is unavailable",
        )
        _assert_equal(
            dashboard_naive_timezone,
            None,
            name="live.storm_dashboard_naive_timezone when dashboard is unavailable",
        )
        _assert_equal(
            registry_zone.get("storm_status"),
            "native_dashboard_unavailable",
            name=f"registry.zones.{zone}.storm_status",
        )

    required_covariates = _sequence(
        _required(live, "required_covariates", name="live"),
        name="live.required_covariates",
    )
    registry_covariates = _sequence(
        _required(
            registry_zone,
            "required_covariates",
            name=f"registry.zones.{zone}",
        ),
        name=f"registry.zones.{zone}.required_covariates",
    )
    if set(required_covariates) != set(registry_covariates):
        raise ZoneModelContractError(
            f"required covariates mismatch for {zone}: "
            f"{sorted(required_covariates)} != {sorted(registry_covariates)}"
        )

    forecast_filename = _text(
        _required(live, "forecast_filename", name="live"),
        name="live.forecast_filename",
    )
    expected_forecast_filename = f"forecast_hourly_{zone.lower()}.csv"
    if Path(forecast_filename).name != forecast_filename:
        raise ZoneModelContractError("live.forecast_filename must be a plain filename")
    _assert_equal(
        forecast_filename,
        expected_forecast_filename,
        name="live.forecast_filename",
    )

    base = live_path.parent
    paths = ZoneModelPaths(
        live_config=live_path,
        registry=registry_file,
        base_config=_resolve(
            _required(live, "base_config", name="live"),
            base=base,
            name="live.base_config",
        ),
        frozen_autonomous_run=_resolve(
            _required(live, "frozen_autonomous_run", name="live"),
            base=base,
            name="live.frozen_autonomous_run",
        ),
        sealed_benchmark_run=_resolve(
            _required(live, "sealed_benchmark_run", name="live"),
            base=base,
            name="live.sealed_benchmark_run",
        ),
        recipe_manifest=_resolve(
            _required(live, "recipe_manifest", name="live"),
            base=base,
            name="live.recipe_manifest",
        ),
        dependency_manifest=(
            None
            if _required(live, "dependency_manifest", name="live") is None
            else _resolve(
                live["dependency_manifest"],
                base=base,
                name="live.dependency_manifest",
            )
        ),
        output_root=_resolve(
            _required(live, "output_root", name="live"),
            base=base,
            name="live.output_root",
        ),
    )
    model_paths = {
        paths.base_config,
        paths.frozen_autonomous_run,
        paths.sealed_benchmark_run,
        paths.recipe_manifest,
        paths.dependency_manifest,
    }
    expected_distinct = 5 if paths.dependency_manifest is not None else 4
    model_paths.discard(None)
    if len(model_paths) != expected_distinct:
        raise ZoneModelContractError("zone model input paths must be distinct")
    if not paths.output_root.name:
        raise ZoneModelContractError("live.output_root must name a zone output directory")
    output_marker = paths.output_root.name.casefold()
    if not (
        _FR_TOKEN_RE.search(paths.output_root.name)
        if zone == "FR"
        else re.search(
            rf"(?:^|[_.-]){re.escape(zone.lower())}(?:$|[_.-])",
            output_marker,
        )
    ):
        raise ZoneModelContractError(
            f"live.output_root must be explicitly labelled for zone {zone}"
        )
    if any(
        paths.output_root == path
        or paths.output_root in path.parents
        or path in paths.output_root.parents
        for path in model_paths
    ):
        raise ZoneModelContractError(
            "live.output_root must not overlap any frozen model input"
        )

    weights = _weights(_required(live, "weights", name="live"))
    prediction_mode = _text(
        _required(live, "prediction_mode", name="live"),
        name="live.prediction_mode",
    )
    if prediction_mode not in {"mkonline_blend", "autonomous_only"}:
        raise ZoneModelContractError(
            "live.prediction_mode must be mkonline_blend or autonomous_only"
        )
    raw_mkonline_enabled = _required(live, "mkonline_enabled", name="live")
    if not isinstance(raw_mkonline_enabled, bool):
        raise ZoneModelContractError("live.mkonline_enabled must be boolean")
    mkonline_enabled = raw_mkonline_enabled
    hashes = _expected_hashes(
        _required(live, "expected_hashes", name="live"),
        dependency_required=mkonline_enabled,
    )
    for key, observed in (
        ("prediction_mode", prediction_mode),
        ("mkonline_enabled", mkonline_enabled),
    ):
        _assert_equal(
            registry_zone.get(key),
            observed,
            name=f"registry.zones.{zone}.{key}",
        )
    if prediction_mode == "mkonline_blend":
        if primary_series is None or paths.dependency_manifest is None:
            raise ZoneModelContractError(
                "mkonline_blend requires primary_series and dependency_manifest"
            )
        if not mkonline_enabled or not (
            0.0 < weights.autonomous < 1.0
            and 0.0 < weights.mkonline_primary < 1.0
        ):
            raise ZoneModelContractError(
                "mkonline_blend requires enabled=true and positive weights"
            )
    else:
        if primary_series is not None or paths.dependency_manifest is not None:
            raise ZoneModelContractError(
                "autonomous_only requires null primary_series and dependency_manifest"
            )
        if (
            mkonline_enabled
            or weights.autonomous != 1.0
            or weights.mkonline_primary != 0.0
        ):
            raise ZoneModelContractError(
                "autonomous_only requires enabled=false and weights 1.0/0.0"
            )
    file_hash_checks = [
        (paths.base_config, hashes.base_config_sha256, "base_config"),
        (paths.recipe_manifest, hashes.recipe_manifest_sha256, "recipe_manifest"),
    ]
    if paths.dependency_manifest is not None:
        file_hash_checks.append((
            paths.dependency_manifest,
            hashes.dependency_manifest_sha256,
            "dependency_manifest",
        ))
    for path, expected_hash, label in file_hash_checks:
        if not path.is_file():
            raise ZoneModelContractError(f"{label} is missing: {path}")
        _assert_equal(_sha256(path), expected_hash, name=f"{label} SHA-256")

    _validate_base_config(
        paths.base_config,
        zone=zone,
        timezone=delivery_timezone,
        target_series=target_series,
        required_covariates=required_covariates,
    )
    frozen_manifest = _verified_run_manifest(
        paths.frozen_autonomous_run,
        expected_checksum_sha256=(
            hashes.frozen_autonomous_checksum_manifest_sha256
        ),
        zone=zone,
        timezone=delivery_timezone,
        label="frozen_autonomous_run",
        forecast_filename=forecast_filename,
        required_live_inputs=(
            "feature_manifest.csv",
            "chronos_oof_hourly.csv.gz",
            "chronos_live_hourly.csv",
            "inputs/chronos_oof_extended.csv.gz",
            "inputs/aligned_inputs.csv.gz",
            "inputs/model_covariates_with_future.csv.gz",
        ),
        required_source_code=LIVE_REFIT_SOURCE_CODE_PATHS,
    )
    benchmark_manifest = _verified_run_manifest(
        paths.sealed_benchmark_run,
        expected_checksum_sha256=(
            hashes.sealed_benchmark_checksum_manifest_sha256
        ),
        zone=zone,
        timezone=delivery_timezone,
        label="sealed_benchmark_run",
        forecast_filename=forecast_filename,
    )
    for label, manifest in (
        ("frozen_autonomous_run", frozen_manifest),
        ("sealed_benchmark_run", benchmark_manifest),
    ):
        _assert_equal(
            manifest.get("target_series"),
            target_series,
            name=f"{label} manifest.target_series",
        )
        comparators = manifest.get("evaluation_only_comparators", [])
        if not isinstance(comparators, Sequence) or isinstance(
            comparators, (str, bytes)
        ):
            raise ZoneModelContractError(
                f"{label} manifest.evaluation_only_comparators must be a list"
            )
        foreign_dashboard = sorted(
            set(str(value) for value in comparators)
            & (
                set(STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE.values())
                - ({dashboard} if dashboard is not None else set())
            )
        )
        if foreign_dashboard:
            raise ZoneModelContractError(
                f"{label} contains a foreign-zone Storm comparator: "
                + ", ".join(foreign_dashboard)
            )
    _assert_equal(
        benchmark_manifest.get("recipe_mode"),
        prediction_mode,
        name="sealed_benchmark_run manifest.recipe_mode",
    )
    benchmark_inputs = benchmark_manifest.get("prediction_inputs")
    if prediction_mode == "autonomous_only" and list(benchmark_inputs) != [
        "autonomous_extended_residual"
    ]:
        raise ZoneModelContractError(
            "sealed autonomous benchmark prediction_inputs must contain only "
            "autonomous_extended_residual"
        )
    recipe, dependency = _validate_recipe_and_dependency(
        recipe_path=paths.recipe_manifest,
        dependency_path=paths.dependency_manifest,
        zone=zone,
        timezone=delivery_timezone,
        primary_series=primary_series,
        frozen_run=paths.frozen_autonomous_run,
        hashes=hashes,
        weights=weights,
        prediction_mode=prediction_mode,
        mkonline_enabled=mkonline_enabled,
        base=base,
    )
    _reject_fr_reuse_outside_fr(
        zone=zone,
        target_series=target_series,
        primary_series=primary_series,
        dashboard_series=dashboard,
        dashboard_primary_series=dashboard_primary,
        dashboard_naive_timezone=dashboard_naive_timezone,
        strict_08_series=strict_08_series,
        paths=paths,
        hashes=hashes,
        weights=weights,
        recipe=recipe,
        dependency=dependency,
    )

    # ``registry_dir`` is intentionally evaluated even when the registry path
    # is absolute: it prevents a future API change from silently resolving the
    # registry against the live YAML directory.
    _assert_equal(registry_dir, registry_file.parent, name="registry directory")
    return ZoneModelContract(
        schema_version=schema_version,
        zone=zone,
        delivery_timezone=delivery_timezone,
        forecast_origin_timezone=origin_timezone,
        forecast_origin_local_time=origin_time,
        target_series=target_series,
        primary_series=primary_series,
        storm_dashboard_series=dashboard,
        storm_dashboard_primary_series=dashboard_primary,
        storm_dashboard_naive_timezone=dashboard_naive_timezone,
        storm_strict_08_series=strict_08_series,
        forecast_filename=forecast_filename,
        required_covariates=required_covariates,
        paths=paths,
        checksum_hashes=hashes,
        weights=weights,
        prediction_mode=prediction_mode,
        mkonline_enabled=mkonline_enabled,
    )


__all__ = (
    "CONTRACT_SCHEMA_VERSION",
    "FR_SEALED_ARTIFACT_HASHES",
    "ZoneArtifactHashes",
    "ZoneBlendWeights",
    "ZoneModelContract",
    "ZoneModelContractError",
    "ZoneModelPaths",
    "load_zone_model_contract",
)
