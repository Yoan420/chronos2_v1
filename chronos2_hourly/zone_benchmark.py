"""Offline, fail-closed publication of sealed multi-zone benchmarks.

This module is the pre-production counterpart of ``multizone_contract``.  It
creates the sealed benchmark directory later consumed by the live contract,
therefore it cannot load ``ZoneModelContract`` itself (that contract already
requires the sealed directory to exist).  Market and Storm identities are
nevertheless taken from the same audited registry and native-series maps.

Two recipes are supported and must be selected explicitly in the frozen JSON
manifest:

``mkonline_blend``
    A convex common-shift blend with a pre-materialised primary MKOnline
    series.  Both B1 and B2 selection artefacts must have passed before final
    publication.

``autonomous_only``
    The frozen autonomous model is published unchanged.  A positive,
    pre-final validation decision is mandatory; a failed or missing MKOnline
    dependency can never silently select this mode.

The publisher performs no Saturn call.  Storm is attached only after the
candidate forecast has been frozen.  The exact native dashboard series is
accepted for DE/BE/NL (and FR for completeness), while ES is deliberately
published without an ``official dashboard`` label because no such native
series has been verified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.multizone_contract import CONTRACT_SCHEMA_VERSION
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_ARTIFACT,
    STORM_DASHBOARD_COLUMN,
    STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE,
    STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE,
    STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE,
    StormDashboardComparator,
    normalize_native_dashboard_series,
)
from chronos2_hourly.zone_live import canonical_zone, load_zone_registry
from chronos2_modular.common import load_yaml
from evaluate_hourly_backtest import evaluate


QUANTILES = ("q10", "q50", "q90")
RECIPE_MODES = frozenset(("mkonline_blend", "autonomous_only"))
DEFAULT_FINAL_START = "2025-08-12"
DEFAULT_FINAL_END = "2026-08-11"
DEFAULT_LIVE_DAY = "2026-08-12"
_SHA256_LENGTH = 64


class BenchmarkContractError(ValueError):
    """A sealed benchmark declaration or artefact failed validation."""


@dataclass(frozen=True)
class BenchmarkIdentity:
    zone: str
    timezone: str
    target_series: str
    primary_series: str | None
    storm_dashboard_series: str | None
    storm_dashboard_primary_series: str | None
    storm_dashboard_naive_timezone: str | None


@dataclass(frozen=True)
class BenchmarkBuildSettings:
    config_path: Path
    registry_path: Path
    source_run: Path
    recipe_path: Path
    expected_recipe_sha256: str
    dependency_path: Path | None
    expected_dependency_sha256: str | None
    output_dir: Path
    identity: BenchmarkIdentity
    final_start: str
    final_end: str
    live_day: str
    primary_final_path: Path | None
    primary_live_path: Path | None
    storm_native_path: Path | None
    storm_native_sha256: str | None
    storm_timestamp_column: str
    storm_value_column: str
    report_filename: str
    report_title: str
    report_history_hours: int
    extreme_threshold: float
    bootstrap_samples: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    return len(raw) == _SHA256_LENGTH and all(c in "0123456789abcdef" for c in raw)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{name} must be an explicit mapping")
    return value


def _resolve(value: Any, *, base: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise BenchmarkContractError(f"{name} must be an explicit path")
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _resolve_optional(value: Any, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkContractError(f"{name} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"{name} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError(f"{name} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def complete_local_range(
    start_day: str,
    end_day: str,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    days = pd.date_range(start_day, end_day, freq="D")
    if days.empty:
        raise BenchmarkContractError("benchmark period is empty")
    pieces = [
        local_delivery_day_index(day.date(), timezone=timezone) for day in days
    ]
    result = pieces[0].append(pieces[1:])
    result.name = "delivery_start_utc"
    return result


def day_histogram(index: pd.DatetimeIndex, *, timezone: str) -> dict[int, int]:
    values = pd.DatetimeIndex(index)
    if values.tz is None:
        raise BenchmarkContractError("delivery index must be timezone-aware")
    local_days = pd.Index(values.tz_convert(timezone).date)
    counts = pd.Series(1, index=local_days).groupby(level=0).sum()
    return {
        int(hours): int(number)
        for hours, number in counts.value_counts().items()
    }


def expected_cutoff(
    index: pd.DatetimeIndex,
    *,
    delivery_timezone: str,
    origin_timezone: str = "Europe/Paris",
    origin_local_time: str = "08:00",
) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(index)
    if values.tz is None:
        raise BenchmarkContractError("delivery index must be timezone-aware")
    try:
        hour, minute = (int(token) for token in origin_local_time.split(":"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("origin_local_time must use HH:MM") from exc
    local_days = pd.DatetimeIndex(values.tz_convert(delivery_timezone).date)
    civil = (
        local_days
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )
    return civil.tz_localize(
        origin_timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")


def load_benchmark_identity(
    registry_path: str | Path,
    *,
    zone: str,
) -> BenchmarkIdentity:
    registry, _ = load_zone_registry(registry_path)
    code = canonical_zone(zone)
    raw = _mapping(registry.get("zones", {}).get(code), name=f"zones.{code}")
    timezone = str(raw.get("delivery_timezone", "")).strip()
    target = str(raw.get("target_series", "")).strip()
    if not timezone or not target:
        raise BenchmarkContractError(f"{code}: incomplete market identity")
    if raw.get("target_status") != "audited_dst_strict":
        raise BenchmarkContractError(f"{code}: target is not DST-strict audited")
    if raw.get("price_unit") != "EUR/MWh":
        raise BenchmarkContractError(f"{code}: target unit must be EUR/MWh")

    primary = raw.get("primary_series")
    primary = str(primary).strip() if primary not in (None, "") else None
    dashboard = raw.get("storm_series")
    dashboard = str(dashboard).strip() if dashboard not in (None, "") else None
    dashboard_primary = raw.get("storm_primary_series")
    dashboard_primary = (
        str(dashboard_primary).strip()
        if dashboard_primary not in (None, "")
        else None
    )
    dashboard_timezone = raw.get("storm_naive_timezone")
    dashboard_timezone = (
        str(dashboard_timezone).strip()
        if dashboard_timezone not in (None, "")
        else None
    )
    verified = STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE.get(code)
    if verified is None:
        if any(value is not None for value in (dashboard, dashboard_primary, dashboard_timezone)):
            raise BenchmarkContractError(
                f"{code}: unverified native Storm dashboard identity must remain null"
            )
        if raw.get("storm_status") != "native_dashboard_unavailable":
            raise BenchmarkContractError(
                f"{code}: missing explicit native_dashboard_unavailable status"
            )
    else:
        expected = (
            verified,
            STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[code],
            STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[code],
        )
        if (dashboard, dashboard_primary, dashboard_timezone) != expected:
            raise BenchmarkContractError(
                f"{code}: native Storm identity differs from audited contract"
            )
        if raw.get("storm_status") != "audited_native_dashboard":
            raise BenchmarkContractError(
                f"{code}: native Storm series is not marked audited"
            )
    return BenchmarkIdentity(
        zone=code,
        timezone=timezone,
        target_series=target,
        primary_series=primary,
        storm_dashboard_series=dashboard,
        storm_dashboard_primary_series=dashboard_primary,
        storm_dashboard_naive_timezone=dashboard_timezone,
    )


def _artifact_gate(
    artifact_path: Path,
    *,
    expected_sha256: str,
    block: str,
    expected_weight: float,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise BenchmarkContractError(f"{block} gate SHA-256 is invalid")
    if sha256_file(artifact_path) != expected_sha256.lower():
        raise BenchmarkContractError(f"{block} gate checksum mismatch")
    payload = _read_json(artifact_path, name=f"{block} gate")
    result = _mapping(payload.get(block), name=f"{block} gate.{block}")
    if result.get("passes") is not True:
        raise BenchmarkContractError(f"{block} gate did not pass")
    protocol = _mapping(payload.get("protocol"), name=f"{block} gate.protocol")
    if protocol.get("final_loaded") is not False:
        raise BenchmarkContractError(f"{block} gate accessed final data")
    observed = (
        payload.get("learned_mkonline_weight")
        if block == "B1"
        else protocol.get("frozen_mkonline_weight")
    )
    try:
        observed_weight = float(observed)
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(f"{block} gate weight is invalid") from exc
    if not math.isclose(observed_weight, expected_weight, rel_tol=0.0, abs_tol=1e-15):
        raise BenchmarkContractError(f"{block} gate weight differs from recipe")
    return payload


def validate_recipe_manifest(
    recipe: Mapping[str, Any],
    *,
    zone: str,
    timezone: str,
    source_run: str | Path,
    source_checksum_sha256: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate a recipe without reading any final target or prediction."""

    code = canonical_zone(zone)
    if recipe.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise BenchmarkContractError("recipe.schema_version must be 1")
    mode = str(recipe.get("recipe_mode", "")).strip()
    if mode not in RECIPE_MODES:
        raise BenchmarkContractError(
            "recipe.recipe_mode must explicitly be mkonline_blend or autonomous_only"
        )
    if recipe.get("status") not in {
        "frozen_before_final_opening",
        "sealed_before_final_opening",
    }:
        raise BenchmarkContractError("recipe was not frozen before final opening")
    if str(recipe.get("zone", "")).upper() != code:
        raise BenchmarkContractError("recipe zone mismatch")
    if recipe.get("timezone") != timezone:
        raise BenchmarkContractError("recipe timezone mismatch")
    source = Path(source_run).expanduser().resolve()
    declared_source = _resolve(
        recipe.get("source_autonomous_run"),
        base=Path(project_root).resolve(),
        name="recipe.source_autonomous_run",
    )
    if declared_source != source:
        raise BenchmarkContractError("recipe autonomous source mismatch")
    declared_checksum = str(
        recipe.get("source_autonomous_checksum_manifest_sha256", "")
    ).lower()
    if declared_checksum != source_checksum_sha256.lower():
        raise BenchmarkContractError("recipe autonomous checksum mismatch")
    if not _is_sha256(declared_checksum):
        raise BenchmarkContractError("recipe autonomous checksum is invalid")
    if recipe.get("final_target_used_for_weight_or_hyperparameters") is not False:
        raise BenchmarkContractError(
            "recipe must explicitly declare no final target used for selection"
        )

    weights = _mapping(recipe.get("weights"), name="recipe.weights")
    try:
        autonomous_weight = float(weights.get("autonomous"))
        primary_weight = float(weights.get("mkonline_primary"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("recipe weights must be numeric") from exc
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (autonomous_weight, primary_weight)
    ) or not math.isclose(
        autonomous_weight + primary_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise BenchmarkContractError("recipe weights must be finite and sum to one")

    validated = dict(recipe)
    validated["recipe_mode"] = mode
    # Operational live contracts repeat this state explicitly.  Keeping both
    # keys in the frozen recipe lets the offline publisher and live runner
    # consume the same immutable JSON without translating modes later.
    declared_prediction_mode = recipe.get("prediction_mode", mode)
    if declared_prediction_mode != mode:
        raise BenchmarkContractError("recipe prediction_mode differs from recipe_mode")
    expected_mkonline_enabled = mode == "mkonline_blend"
    declared_mkonline_enabled = recipe.get(
        "mkonline_enabled", expected_mkonline_enabled
    )
    if declared_mkonline_enabled is not expected_mkonline_enabled:
        raise BenchmarkContractError(
            "recipe mkonline_enabled differs from recipe_mode"
        )
    validated["prediction_mode"] = mode
    validated["mkonline_enabled"] = expected_mkonline_enabled
    validated["_validated_weights"] = {
        "autonomous": autonomous_weight,
        "mkonline_primary": primary_weight,
    }
    if mode == "autonomous_only":
        if not (
            math.isclose(autonomous_weight, 1.0, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(primary_weight, 0.0, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise BenchmarkContractError("autonomous_only requires weights 1/0")
        if recipe.get("external_expert") not in (None, {}):
            raise BenchmarkContractError(
                "autonomous_only must not declare an external expert"
            )
        decision = _mapping(
            recipe.get("autonomous_validation"),
            name="recipe.autonomous_validation",
        )
        if decision.get("approved") is not True:
            raise BenchmarkContractError("autonomous_only was not explicitly approved")
        if decision.get("final_loaded") is not False:
            raise BenchmarkContractError(
                "autonomous_only validation accessed final data"
            )
        if not str(decision.get("reason", "")).strip():
            raise BenchmarkContractError(
                "autonomous_only validation requires an explicit reason"
            )
        return validated

    if not (0.0 < primary_weight < 1.0):
        raise BenchmarkContractError("mkonline_blend requires a non-zero convex weight")
    external = _mapping(recipe.get("external_expert"), name="recipe.external_expert")
    if external.get("storm_used_as_feature") is not False:
        raise BenchmarkContractError("recipe external expert is not Storm-free")
    if external.get("interpolation_allowed") is not False:
        raise BenchmarkContractError("primary interpolation must be forbidden")
    selection = _mapping(recipe.get("selection_protocol"), name="selection_protocol")
    if selection.get("final_loaded") is not False:
        raise BenchmarkContractError("selection protocol accessed final data")
    root = Path(project_root).resolve()
    gate_payloads: dict[str, Any] = {}
    for block, path_key, hash_key in (
        ("B1", "b1_gate_artifact", "b1_gate_artifact_sha256"),
        ("B2", "b2_veto_artifact", "b2_veto_artifact_sha256"),
    ):
        path = _resolve(selection.get(path_key), base=root, name=path_key)
        gate_payloads[block] = _artifact_gate(
            path,
            expected_sha256=str(selection.get(hash_key, "")),
            block=block,
            expected_weight=primary_weight,
        )
    validated["_validated_gate_payloads"] = gate_payloads
    return validated


def validate_dependency_manifest(
    dependency: Mapping[str, Any],
    *,
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    """Validate the direct, formula-free MKOnline dependency declaration."""

    if dependency.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise BenchmarkContractError("dependency.schema_version must be 1")
    if str(dependency.get("zone", "")).upper() != identity.zone:
        raise BenchmarkContractError("dependency zone mismatch")
    if dependency.get("timezone") != identity.timezone:
        raise BenchmarkContractError("dependency timezone mismatch")
    if dependency.get("terminal_series") != identity.primary_series:
        raise BenchmarkContractError("dependency primary series mismatch")
    if dependency.get("terminal_type") != "primary":
        raise BenchmarkContractError("dependency terminal must be a primary series")
    if dependency.get("terminal_formula") is not None:
        raise BenchmarkContractError("dependency terminal must not be a formula")
    if dependency.get("storm_token_found") is not False:
        raise BenchmarkContractError("dependency is not certified Storm-free")
    if dependency.get("dependency_gate_passed") is not True:
        raise BenchmarkContractError("dependency gate did not pass")
    if identity.zone == "ES" and dependency.get("status") == "experimental_monthly_fallback":
        raise BenchmarkContractError(
            "ES monthly MKOnline fallback cannot be promoted as a validated blend"
        )
    metadata = _mapping(
        dependency.get("terminal_metadata"), name="dependency.terminal_metadata"
    )
    if str(metadata.get("mercure:provider", "")).upper() != "MKONLINE":
        raise BenchmarkContractError("dependency provider must be MKONLINE")
    if str(metadata.get("mercure:source", "")).upper() != "WATTSIGHT":
        raise BenchmarkContractError("dependency source must be WATTSIGHT")
    return dict(dependency)


def verify_source_run(
    source_run: str | Path,
    *,
    zone: str,
    timezone: str,
    forecast_filename: str,
    expected_manifest_sha256: str,
    allowed_external_price_series: str | None = None,
) -> dict[str, Any]:
    """Verify immutable source artefacts without inspecting final values."""

    source = Path(source_run).expanduser().resolve()
    checksum_path = source / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise BenchmarkContractError(f"source checksum manifest missing: {checksum_path}")
    if sha256_file(checksum_path) != expected_manifest_sha256.lower():
        raise BenchmarkContractError("source checksum manifest mismatch")
    checksum = _read_json(checksum_path, name="source checksum manifest")
    if str(checksum.get("algorithm", "")).lower() != "sha256":
        raise BenchmarkContractError("source checksum algorithm must be sha256")
    artifacts = checksum.get("artifacts")
    if not isinstance(artifacts, list):
        raise BenchmarkContractError("source checksum artifacts must be a list")
    declared: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or item.get("role") != "run_artifact":
            continue
        raw = str(item.get("path", "")).replace("\\", "/")
        if raw and not Path(raw).is_absolute() and not raw.startswith("../"):
            declared[Path(raw).as_posix()] = item
    for relative in ("run_manifest.json", "backtest_hourly_oof.csv.gz", forecast_filename):
        artifact = source / relative
        item = declared.get(relative)
        if item is None or not artifact.is_file():
            raise BenchmarkContractError(f"source artefact is not sealed: {relative}")
        expected = str(item.get("sha256", "")).lower()
        if not _is_sha256(expected) or sha256_file(artifact) != expected:
            raise BenchmarkContractError(f"source artefact checksum mismatch: {relative}")
    manifest = _read_json(source / "run_manifest.json", name="source run manifest")
    if str(manifest.get("zone", "")).upper() != canonical_zone(zone):
        raise BenchmarkContractError("source run zone mismatch")
    if manifest.get("timezone") != timezone:
        raise BenchmarkContractError("source run timezone mismatch")
    if manifest.get("uses_legacy_price_forecast") is not False:
        raise BenchmarkContractError("source run uses a forbidden legacy forecast")
    prediction_inputs = manifest.get("prediction_inputs", [])
    if not isinstance(prediction_inputs, Sequence) or isinstance(
        prediction_inputs, (str, bytes)
    ):
        raise BenchmarkContractError(
            "source manifest prediction_inputs must be a list"
        )
    external_prices = manifest.get("external_price_forecasts_loaded", [])
    if not isinstance(external_prices, Sequence) or isinstance(
        external_prices, (str, bytes)
    ):
        raise BenchmarkContractError(
            "source manifest external_price_forecasts_loaded must be a list"
        )
    values: list[Any] = [*prediction_inputs, *external_prices]
    for key, candidate in (
        ("prediction_inputs", prediction_inputs),
        ("external_price_forecasts_loaded", external_prices),
    ):
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            raise BenchmarkContractError(f"source manifest {key} must be a list")
    if any("storm" in str(value).casefold() for value in values):
        raise BenchmarkContractError("source run uses Storm as a prediction input")
    if manifest.get("storm_used_as_feature") not in (None, False):
        raise BenchmarkContractError("source run does not certify Storm-free inputs")
    declared_prices = [str(value) for value in external_prices]
    if allowed_external_price_series is None and declared_prices:
        raise BenchmarkContractError(
            "autonomous source unexpectedly loads an external price forecast"
        )
    if allowed_external_price_series is not None and any(
        value != allowed_external_price_series for value in declared_prices
    ):
        raise BenchmarkContractError(
            "source run loads a foreign external price forecast"
        )
    for filename in ("backtest_hourly_oof.csv.gz", forecast_filename):
        columns = pd.read_csv(source / filename, nrows=0).columns
        forbidden = [name for name in columns if "storm" in str(name).casefold()]
        if forbidden:
            raise BenchmarkContractError(
                f"source artefact contains Storm columns: {filename}: {forbidden}"
            )
    return manifest


def _indexed_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    result = frame.copy()
    if "delivery_start_utc" in result:
        index = pd.DatetimeIndex(
            pd.to_datetime(result.pop("delivery_start_utc"), utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        result.index = index
    else:
        index = pd.DatetimeIndex(result.index)
        if index.tz is None:
            raise BenchmarkContractError(f"{name} index must be UTC-aware")
        result.index = index.tz_convert("UTC")
        result.index.name = "delivery_start_utc"
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise BenchmarkContractError(f"{name} timeline is duplicate or unordered")
    return result


def _quantiles(frame: pd.DataFrame, *, model: str, name: str) -> pd.DataFrame:
    columns = {quantile: f"{model}__{quantile}" for quantile in QUANTILES}
    if not all(column in frame for column in columns.values()):
        if all(quantile in frame for quantile in QUANTILES):
            columns = {quantile: quantile for quantile in QUANTILES}
        else:
            raise BenchmarkContractError(f"{name}: autonomous quantiles are missing")
    result = frame.loc[:, list(columns.values())].rename(
        columns={value: key for key, value in columns.items()}
    )
    result = result.apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(float)
    if not np.isfinite(values).all():
        raise BenchmarkContractError(f"{name}: non-finite autonomous quantiles")
    if not ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all():
        raise BenchmarkContractError(f"{name}: crossed autonomous quantiles")
    return result


def blend_quantiles(
    autonomous: pd.DataFrame,
    primary_q50: pd.Series,
    *,
    autonomous_weight: float,
    primary_weight: float,
) -> pd.DataFrame:
    if not math.isclose(
        float(autonomous_weight) + float(primary_weight),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise BenchmarkContractError("blend weights do not sum to one")
    auto = _quantiles(autonomous, model="unused", name="autonomous blend input")
    primary = pd.to_numeric(primary_q50, errors="coerce").reindex(auto.index)
    if not np.isfinite(primary.to_numpy(float)).all():
        raise BenchmarkContractError("primary series is incomplete")
    q50 = autonomous_weight * auto["q50"] + primary_weight * primary
    shift = q50 - auto["q50"]
    result = pd.DataFrame(index=auto.index)
    for quantile in QUANTILES:
        result[quantile] = auto[quantile] + shift
    result["shift"] = shift
    result["mkonline_q50"] = primary
    if not np.allclose(
        result["q90"] - result["q10"],
        auto["q90"] - auto["q10"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("common shift changed interval width")
    return result


def load_primary_materialization(
    path: str | Path,
    *,
    expected_index: pd.DatetimeIndex,
    identity: BenchmarkIdentity,
    origin_timezone: str = "Europe/Paris",
    origin_local_time: str = "08:00",
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    materialized = Path(path).expanduser().resolve()
    frame = pd.read_parquet(materialized)
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BenchmarkContractError(f"{materialized}: missing PIT columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected = pd.DatetimeIndex(expected_index)
    if not delivery.equals(expected):
        raise BenchmarkContractError(f"{materialized}: exact delivery timeline mismatch")
    cutoff = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    required_cutoff = expected_cutoff(
        delivery,
        delivery_timezone=identity.timezone,
        origin_timezone=origin_timezone,
        origin_local_time=origin_local_time,
    )
    if not cutoff.equals(required_cutoff) or not revision.equals(required_cutoff):
        raise BenchmarkContractError(
            f"{materialized}: PIT marker differs from civil D-1 {origin_local_time}"
        )
    values = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(float),
        index=delivery,
        name="mkonline_primary__q50",
    )
    if not np.isfinite(values.to_numpy(float)).all():
        raise BenchmarkContractError(f"{materialized}: primary contains non-finite values")
    return values, pd.Series(cutoff, index=delivery), {
        "series": identity.primary_series,
        "path": str(materialized),
        "sha256": sha256_file(materialized),
        "hours": int(len(values)),
        "local_days": int(len(pd.Index(delivery.tz_convert(identity.timezone).date).unique())),
        "local_day_hour_histogram": {
            str(key): value for key, value in day_histogram(delivery, timezone=identity.timezone).items()
        },
        "coverage": 1.0,
        "cutoff": f"civil D-1 {origin_local_time} {origin_timezone}",
        "cutoff_violations": 0,
        "interpolation": False,
    }


def build_frozen_candidate(
    source_backtest: pd.DataFrame,
    source_forecast: pd.DataFrame,
    recipe: Mapping[str, Any],
    final_index: pd.DatetimeIndex,
    live_index: pd.DatetimeIndex,
    primary_final: pd.Series | None = None,
    primary_live: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Freeze candidate values before any Storm comparator is accepted."""

    mode = str(recipe.get("recipe_mode", ""))
    if mode not in RECIPE_MODES:
        raise BenchmarkContractError("recipe mode was not validated")
    weights = _mapping(recipe.get("_validated_weights", recipe.get("weights")), name="weights")
    auto_weight = float(weights.get("autonomous"))
    primary_weight = float(weights.get("mkonline_primary"))
    source_model = str(recipe.get("source_autonomous_model", "residual_corrected"))
    backtest = _indexed_frame(source_backtest, name="source backtest")
    forecast = _indexed_frame(source_forecast, name="source forecast")
    final = pd.DatetimeIndex(final_index)
    live = pd.DatetimeIndex(live_index)
    if not final.isin(backtest.index).all():
        raise BenchmarkContractError("source backtest does not cover final index")
    if not forecast.index.equals(live):
        raise BenchmarkContractError("source forecast does not equal live delivery day")
    for frame, label in ((backtest, "backtest"), (forecast, "forecast")):
        forbidden = [column for column in frame if "storm" in str(column).casefold()]
        if forbidden:
            raise BenchmarkContractError(f"source {label} contains Storm columns: {forbidden}")
    actual = pd.to_numeric(backtest.loc[final, "actual"], errors="coerce")
    if not np.isfinite(actual.to_numpy(float)).all():
        raise BenchmarkContractError("final canonical actual is incomplete")

    if mode == "mkonline_blend":
        if primary_final is None or primary_live is None:
            raise BenchmarkContractError(
                "mkonline_blend requires both explicit primary materialisations"
            )
        auto_final = _quantiles(backtest.loc[final], model=source_model, name="final")
        auto_live = _quantiles(forecast, model=source_model, name="live")
        blended_final = blend_quantiles(
            auto_final,
            primary_final,
            autonomous_weight=auto_weight,
            primary_weight=primary_weight,
        )
        blended_live = blend_quantiles(
            auto_live,
            primary_live,
            autonomous_weight=auto_weight,
            primary_weight=primary_weight,
        )
        for quantile in QUANTILES:
            column = f"mkonline_blend__{quantile}"
            backtest[column] = np.nan
            backtest.loc[final, column] = blended_final[quantile].to_numpy(float)
            forecast[column] = blended_live[quantile].to_numpy(float)
            forecast[quantile] = blended_live[quantile].to_numpy(float)
        backtest["mkonline_primary__q50"] = np.nan
        backtest.loc[final, "mkonline_primary__q50"] = pd.Series(primary_final).reindex(final).to_numpy(float)
        backtest["mkonline_blend_shift"] = np.nan
        backtest.loc[final, "mkonline_blend_shift"] = blended_final["shift"].to_numpy(float)
        forecast["mkonline_primary__q50"] = pd.Series(primary_live).reindex(live).to_numpy(float)
        forecast["mkonline_blend_shift"] = blended_live["shift"].to_numpy(float)
        forecast["price_eur_mwh"] = forecast["q50"]
        native_model = "mkonline_blend"
    else:
        if primary_final is not None or primary_live is not None:
            raise BenchmarkContractError(
                "autonomous_only forbids primary inputs; no fallback is allowed"
            )
        auto_live = _quantiles(forecast, model=source_model, name="live")
        for quantile in QUANTILES:
            forecast[quantile] = auto_live[quantile].to_numpy(float)
        forecast["price_eur_mwh"] = forecast["q50"]
        native_model = source_model

    audit = {
        "recipe_mode": mode,
        "native_model": native_model,
        "source_autonomous_model": source_model,
        "weights": {
            "autonomous": auto_weight,
            "mkonline_primary": primary_weight,
        },
        "final_hours": int(len(final)),
        "live_hours": int(len(live)),
        "candidate_frozen_before_storm_attachment": True,
        "storm_used_for_prediction": False,
        "fallback_used": False,
    }
    return backtest, forecast, audit


def _native_raw_series(
    path: Path,
    *,
    timestamp_column: str,
    value_column: str,
) -> pd.Series:
    frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if timestamp_column not in frame or value_column not in frame:
        raise BenchmarkContractError(
            f"{path}: native Storm columns {timestamp_column!r}/{value_column!r} missing"
        )
    parsed = pd.to_datetime(frame[timestamp_column], errors="raise")
    index = pd.DatetimeIndex(parsed)
    if index.tz is not None:
        raise BenchmarkContractError(
            "official native Storm artefact must retain its timezone-naive civil index"
        )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise BenchmarkContractError("native Storm timeline is duplicate or unordered")
    return pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float),
        index=index,
        name="value",
    )


def attach_native_storm_for_statistics(
    statistics: pd.DataFrame,
    native_values: pd.Series,
    *,
    zone: str,
    timezone: str,
    source: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], StormDashboardComparator]:
    """Attach exact native Storm after candidate freeze, never for ES."""

    code = canonical_zone(zone)
    if code not in STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE:
        raise BenchmarkContractError(
            f"{code}: no verified native Storm dashboard series; official label forbidden"
        )
    frame = _indexed_frame(statistics, name="Statistics")
    if "actual" not in frame:
        raise BenchmarkContractError("Statistics actual is missing")
    delivery = pd.DatetimeIndex(frame.index)
    actual = pd.Series(
        pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float),
        index=delivery,
        name="actual",
    )
    provenance = dict(source or {})
    provenance.update(
        {
            "series": STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE[code],
            "primary_series": STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[code],
            "zone": code,
            "naive_timezone": STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[code],
            "used_for_prediction": False,
        }
    )
    try:
        comparator = normalize_native_dashboard_series(
            native_values,
            zone=code,
            expected_index=delivery,
            actual=actual,
            source=provenance,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            f"{code}: invalid exact native Storm comparator: {exc}"
        ) from exc
    frame[STORM_DASHBOARD_COLUMN] = comparator.values.to_numpy(float)
    audit = {
        "mode": "sealed_benchmark_statistics",
        "zone": code,
        "timezone": timezone,
        "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
        "source": dict(comparator.audit.get("source", provenance)),
        "storm_dashboard": dict(comparator.audit),
        "used_for_prediction": False,
        "storm_used_for_prediction": False,
        "candidate_frozen_before_comparator_attachment": True,
        "historical_forecasts_rewritten": False,
        "sealed_benchmark_rewritten": False,
        "report_scope_note": (
            "Storm officiel dashboard utilise uniquement l'extraction figée "
            "de la série native exacte; il est attaché après gel du candidat."
        ),
    }
    return frame, audit, comparator


def _metric_rows(backtest: pd.DataFrame, final_index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    frame = backtest.loc[final_index]
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if not str(column).endswith("__q50"):
            continue
        prediction = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(prediction)
        if not valid.any():
            continue
        rows.append(
            {
                "model": str(column).removesuffix("__q50"),
                "mae": float(np.mean(np.abs(actual[valid] - prediction[valid]))),
                "n_scored": int(valid.sum()),
                "n_expected": int(len(frame)),
                "prediction_coverage": float(np.isfinite(prediction).mean()),
                "score_coverage": float(valid.mean()),
            }
        )
    return rows


def _evaluate_comparator_pairs(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    timezone: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate an exact native comparator on its finite physical timestamps.

    The native Storm curve represents the autumn fold once.  Requiring every
    civil day to be complete would either fabricate the absent fold or discard
    a real market day.  This paired evaluator keeps the explicit finite mask
    and reports its coverage instead.
    """

    required = {"actual", baseline, candidate}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BenchmarkContractError(f"comparator evaluation columns missing: {missing}")
    work = frame.loc[:, ["actual", baseline, candidate]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(work.to_numpy(float)).all(axis=1)
    paired = work.loc[finite].copy()
    if paired.empty:
        raise BenchmarkContractError("native Storm paired evaluation is empty")
    local = paired.index.tz_convert(timezone)
    paired["local_date"] = local.date
    paired["local_month"] = local.strftime("%Y-%m")
    paired["local_hour"] = local.hour
    paired["baseline_abs_error"] = (paired["actual"] - paired[baseline]).abs()
    paired["candidate_abs_error"] = (paired["actual"] - paired[candidate]).abs()

    def grouped(column: str) -> pd.DataFrame:
        result = paired.groupby(column, sort=True).agg(
            n_hours=("actual", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            candidate_mae=("candidate_abs_error", "mean"),
        )
        result["delta_mae"] = result["candidate_mae"] - result["baseline_mae"]
        return result.reset_index()

    daily = grouped("local_date")
    baseline_mae = float(paired["baseline_abs_error"].mean())
    candidate_mae = float(paired["candidate_abs_error"].mean())
    delta = candidate_mae - baseline_mae
    summary = {
        "baseline": baseline,
        "candidate": candidate,
        "n_hours": int(len(paired)),
        "n_expected_hours": int(len(frame)),
        "paired_coverage": float(len(paired) / len(frame)),
        "n_local_days": int(len(pd.Index(paired["local_date"]).unique())),
        "start_utc": str(paired.index[0]),
        "end_utc": str(paired.index[-1]),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "delta_mae": delta,
        "relative_improvement": float(-delta / baseline_mae),
        "native_dst_policy": "finite exact timestamps; no fold interpolation",
    }
    return summary, daily, grouped("local_month"), grouped("local_hour")


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    previous: Path | None = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite")
        previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        output.replace(previous)
    try:
        staging.replace(output)
    except Exception:
        if previous is not None and previous.exists() and not output.exists():
            previous.replace(output)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def _write_checksums(
    staging: Path,
    *,
    output: Path,
    settings: BenchmarkBuildSettings,
    project_root: Path,
) -> None:
    checksum_path = staging / "artifact_checksums.json"
    entries: list[dict[str, Any]] = []
    for role, path in (
        ("source_config", settings.config_path),
        ("frozen_recipe", settings.recipe_path),
    ):
        entries.append(
            {
                "path": str(path),
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if settings.dependency_path is not None:
        entries.append(
            {
                "path": str(settings.dependency_path),
                "role": "dependency_manifest",
                "size_bytes": settings.dependency_path.stat().st_size,
                "sha256": sha256_file(settings.dependency_path),
            }
        )
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "role": "run_artifact",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for relative in (
        "run_zone_benchmark_hourly.py",
        "chronos2_hourly/zone_benchmark.py",
        "chronos2_hourly/multizone_contract.py",
        "chronos2_hourly/storm_dashboard.py",
        "chronos2_hourly/reporting.py",
        "evaluate_hourly_backtest.py",
    ):
        path = project_root / relative
        entries.append(
            {
                "path": relative,
                "role": "source_code",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "output_directory": str(output),
            "artifacts": entries,
        },
    )


def load_build_settings(config_path: str | Path) -> BenchmarkBuildSettings:
    path = Path(config_path).expanduser().resolve()
    config = load_yaml(path)
    raw = _mapping(config.get("zone_benchmark"), name="zone_benchmark")
    allowed = {
        "schema_version",
        "zone",
        "timezone",
        "registry",
        "source_run",
        "recipe_manifest",
        "recipe_manifest_sha256",
        "dependency_manifest",
        "dependency_manifest_sha256",
        "final_start",
        "final_end",
        "live_day",
        "primary_final_file",
        "primary_live_file",
        "storm_dashboard",
        "bootstrap_samples",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise BenchmarkContractError(
            "unknown zone_benchmark keys (implicit materialisation commands are "
            f"forbidden): {unknown}"
        )
    if raw.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise BenchmarkContractError("zone_benchmark.schema_version must be 1")
    base = path.parent
    zone = canonical_zone(str(raw.get("zone", "")))
    registry_path = _resolve(
        raw.get("registry", "chronos2_hourly_live_zones.yaml"),
        base=base,
        name="zone_benchmark.registry",
    )
    identity = load_benchmark_identity(registry_path, zone=zone)
    if raw.get("timezone") != identity.timezone:
        raise BenchmarkContractError("benchmark timezone differs from registry")
    source_run = _resolve(raw.get("source_run"), base=base, name="source_run")
    recipe_path = _resolve(raw.get("recipe_manifest"), base=base, name="recipe_manifest")
    expected_recipe_sha256 = str(raw.get("recipe_manifest_sha256", "")).lower()
    if not _is_sha256(expected_recipe_sha256):
        raise BenchmarkContractError(
            "zone_benchmark.recipe_manifest_sha256 must be explicit"
        )
    dependency_path = _resolve_optional(raw.get("dependency_manifest"), base=base)
    dependency_sha_raw = raw.get("dependency_manifest_sha256")
    expected_dependency_sha256 = (
        str(dependency_sha_raw).lower()
        if dependency_sha_raw not in (None, "")
        else None
    )
    if (dependency_path is None) != (expected_dependency_sha256 is None):
        raise BenchmarkContractError(
            "dependency_manifest and dependency_manifest_sha256 must be declared together"
        )
    if expected_dependency_sha256 is not None and not _is_sha256(
        expected_dependency_sha256
    ):
        raise BenchmarkContractError("dependency manifest SHA-256 is invalid")
    output = _mapping(config.get("output"), name="output")
    report = _mapping(config.get("report"), name="report")
    report_filename = str(
        report.get("filename", f"chronos2_hourly_{zone.lower()}_benchmark.html")
    )
    if Path(report_filename).name != report_filename or not report_filename.lower().endswith(
        ".html"
    ):
        raise BenchmarkContractError("report.filename must be a plain HTML filename")
    storm = raw.get("storm_dashboard")
    storm_path: Path | None = None
    storm_sha: str | None = None
    timestamp_column = "timestamp"
    value_column = "value"
    if storm is not None:
        storm_raw = _mapping(storm, name="zone_benchmark.storm_dashboard")
        if identity.storm_dashboard_series is None:
            raise BenchmarkContractError(
                f"{zone}: storm_dashboard must be null; official native series is unavailable"
            )
        if storm_raw.get("kind") != "native_exact_naive":
            raise BenchmarkContractError("Storm dashboard kind must be native_exact_naive")
        if storm_raw.get("series") != identity.storm_dashboard_series:
            raise BenchmarkContractError("Storm dashboard series mismatch")
        if storm_raw.get("primary_series") != identity.storm_dashboard_primary_series:
            raise BenchmarkContractError("Storm dashboard primary mismatch")
        storm_path = _resolve(storm_raw.get("path"), base=base, name="storm_dashboard.path")
        storm_sha = str(storm_raw.get("sha256", "")).lower()
        if not _is_sha256(storm_sha):
            raise BenchmarkContractError("Storm dashboard SHA-256 is invalid")
        timestamp_column = str(storm_raw.get("timestamp_column", "timestamp"))
        value_column = str(storm_raw.get("value_column", "value"))
    elif identity.storm_dashboard_series is not None:
        raise BenchmarkContractError(
            f"{zone}: exact native Storm materialisation is mandatory for Statistics"
        )
    return BenchmarkBuildSettings(
        config_path=path,
        registry_path=registry_path,
        source_run=source_run,
        recipe_path=recipe_path,
        expected_recipe_sha256=expected_recipe_sha256,
        dependency_path=dependency_path,
        expected_dependency_sha256=expected_dependency_sha256,
        output_dir=_resolve(output.get("directory"), base=base, name="output.directory"),
        identity=identity,
        final_start=str(raw.get("final_start", DEFAULT_FINAL_START)),
        final_end=str(raw.get("final_end", DEFAULT_FINAL_END)),
        live_day=str(raw.get("live_day", DEFAULT_LIVE_DAY)),
        primary_final_path=_resolve_optional(raw.get("primary_final_file"), base=base),
        primary_live_path=_resolve_optional(raw.get("primary_live_file"), base=base),
        storm_native_path=storm_path,
        storm_native_sha256=storm_sha,
        storm_timestamp_column=timestamp_column,
        storm_value_column=value_column,
        report_filename=report_filename,
        report_title=str(report.get("title", f"Chronos-2 horaire {zone} - benchmark scellé")),
        report_history_hours=int(report.get("forecast_history_hours", 168)),
        extreme_threshold=float(report.get("extreme_threshold", 150.0)),
        bootstrap_samples=int(raw.get("bootstrap_samples", 20_000)),
    )


def publish_zone_benchmark(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Publish one offline sealed benchmark from already materialised inputs."""

    settings = load_build_settings(config_path)
    if output_dir is not None:
        settings = BenchmarkBuildSettings(
            **{
                **settings.__dict__,
                "output_dir": Path(output_dir).expanduser().resolve(),
            }
        )
    identity = settings.identity
    forecast_filename = f"forecast_hourly_{identity.zone.lower()}.csv"
    if sha256_file(settings.recipe_path) != settings.expected_recipe_sha256:
        raise BenchmarkContractError("frozen recipe manifest checksum mismatch")
    recipe_raw = _read_json(settings.recipe_path, name="recipe manifest")
    declared_source_checksum = str(
        recipe_raw.get("source_autonomous_checksum_manifest_sha256", "")
    ).lower()
    if not _is_sha256(declared_source_checksum):
        raise BenchmarkContractError("recipe autonomous checksum is invalid")
    recipe = validate_recipe_manifest(
        recipe_raw,
        zone=identity.zone,
        timezone=identity.timezone,
        source_run=settings.source_run,
        source_checksum_sha256=declared_source_checksum,
        project_root=settings.config_path.parent,
    )
    source_manifest = verify_source_run(
        settings.source_run,
        zone=identity.zone,
        timezone=identity.timezone,
        forecast_filename=forecast_filename,
        expected_manifest_sha256=declared_source_checksum,
        allowed_external_price_series=None,
    )
    sealed_final = recipe_raw.get("sealed_final_protocol")
    if sealed_final is not None:
        final_contract = _mapping(
            sealed_final, name="recipe.sealed_final_protocol"
        )
        declared = (
            str(final_contract.get("start_local_day", "")),
            str(final_contract.get("end_local_day", "")),
        )
        if declared != (settings.final_start, settings.final_end):
            raise BenchmarkContractError(
                "configured final period differs from frozen recipe"
            )

    mode = recipe["recipe_mode"]
    if mode == "mkonline_blend":
        if identity.primary_series is None:
            raise BenchmarkContractError(f"{identity.zone}: no audited primary series")
        external = _mapping(recipe.get("external_expert"), name="external_expert")
        if external.get("series") != identity.primary_series:
            raise BenchmarkContractError("recipe primary differs from registry")
        if settings.primary_final_path is None or settings.primary_live_path is None:
            raise BenchmarkContractError("mkonline_blend materialisations are missing")
        if (
            settings.dependency_path is None
            or settings.expected_dependency_sha256 is None
        ):
            raise BenchmarkContractError(
                "mkonline_blend requires a checksummed dependency manifest"
            )
        if (
            not settings.dependency_path.is_file()
            or sha256_file(settings.dependency_path)
            != settings.expected_dependency_sha256
        ):
            raise BenchmarkContractError("dependency manifest checksum mismatch")
        dependency = validate_dependency_manifest(
            _read_json(settings.dependency_path, name="dependency manifest"),
            identity=identity,
        )
        declared_dependency = _resolve(
            external.get("dependency_manifest"),
            base=settings.config_path.parent,
            name="recipe.external_expert.dependency_manifest",
        )
        if declared_dependency != settings.dependency_path:
            raise BenchmarkContractError("recipe dependency path mismatch")
        if str(external.get("dependency_manifest_sha256", "")).lower() != (
            settings.expected_dependency_sha256
        ):
            raise BenchmarkContractError("recipe dependency checksum mismatch")
    else:
        dependency = None
        if settings.primary_final_path is not None or settings.primary_live_path is not None:
            raise BenchmarkContractError(
                "autonomous_only config must not declare primary files"
            )
        if settings.dependency_path is not None:
            raise BenchmarkContractError(
                "autonomous_only config must not declare a dependency manifest"
            )

    final_index = complete_local_range(
        settings.final_start,
        settings.final_end,
        timezone=identity.timezone,
    )
    live_index = complete_local_range(
        settings.live_day,
        settings.live_day,
        timezone=identity.timezone,
    )
    source_backtest = pd.read_csv(settings.source_run / "backtest_hourly_oof.csv.gz")
    source_forecast = pd.read_csv(settings.source_run / forecast_filename)

    primary_final: pd.Series | None = None
    primary_live: pd.Series | None = None
    primary_audit: dict[str, Any] | None = None
    if mode == "mkonline_blend":
        assert settings.primary_final_path is not None
        assert settings.primary_live_path is not None
        primary_final, cutoff_final, final_audit = load_primary_materialization(
            settings.primary_final_path,
            expected_index=final_index,
            identity=identity,
        )
        primary_live, cutoff_live, live_audit = load_primary_materialization(
            settings.primary_live_path,
            expected_index=live_index,
            identity=identity,
        )
        primary_audit = {"final": final_audit, "live": live_audit}
    else:
        cutoff_final = cutoff_live = None

    backtest, forecast, candidate_audit = build_frozen_candidate(
        source_backtest,
        source_forecast,
        recipe,
        final_index,
        live_index,
        primary_final=primary_final,
        primary_live=primary_live,
    )
    native_model = str(candidate_audit["native_model"])
    source_model = str(candidate_audit["source_autonomous_model"])
    if mode == "mkonline_blend":
        origin_column = f"{source_model}_forecast_origin_utc"
        source_origin = (
            backtest.loc[final_index, origin_column]
            if origin_column in backtest
            else backtest.loc[final_index, "forecast_origin_utc"]
        )
        auto_origin = pd.DatetimeIndex(pd.to_datetime(source_origin, utc=True, errors="raise"))
        candidate_origin = pd.DatetimeIndex(
            np.maximum(auto_origin.asi8, pd.DatetimeIndex(cutoff_final).asi8),
            tz="UTC",
        )
        if not bool((candidate_origin < final_index).all()):
            raise BenchmarkContractError("benchmark origin is not causal")
        backtest["mkonline_blend_forecast_origin_utc"] = pd.NaT
        backtest.loc[final_index, "mkonline_blend_forecast_origin_utc"] = candidate_origin.astype(str)

        live_origin_column = f"{source_model}_forecast_origin_utc"
        if live_origin_column in forecast:
            live_source_origin = forecast[live_origin_column]
        elif "forecast_origin_utc" in forecast:
            live_source_origin = forecast["forecast_origin_utc"]
        else:
            live_source_origin = pd.Series(
                expected_cutoff(
                    live_index,
                    delivery_timezone=identity.timezone,
                ),
                index=forecast.index,
            )
        auto_live_origin = pd.DatetimeIndex(
            pd.to_datetime(live_source_origin, utc=True, errors="raise")
        )
        candidate_live_origin = pd.DatetimeIndex(
            np.maximum(auto_live_origin.asi8, pd.DatetimeIndex(cutoff_live).asi8),
            tz="UTC",
        )
        if not bool((candidate_live_origin < live_index).all()):
            raise BenchmarkContractError("live benchmark origin is not causal")
        forecast["mkonline_blend_forecast_origin_utc"] = candidate_live_origin.astype(str)

    output = settings.output_dir
    if output == settings.source_run:
        raise BenchmarkContractError("output must differ from source run")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(settings.source_run, staging, dirs_exist_ok=True)
        for html in staging.glob("*.html"):
            html.unlink()
        backtest.reset_index().to_csv(
            staging / "backtest_hourly_oof.csv.gz",
            index=False,
            compression="gzip",
        )
        forecast.reset_index().to_csv(staging / forecast_filename, index=False)

        statistics = backtest.loc[final_index].copy()
        statistics_audit: dict[str, Any] = {
            "mode": "sealed_benchmark_statistics",
            "zone": identity.zone,
            "timezone": identity.timezone,
            "evaluation_start_local_date": settings.final_start,
            "evaluation_end_local_date": settings.final_end,
            "n_total_statistics_hours": int(len(statistics)),
            "candidate_model": native_model,
            "recipe_mode": mode,
            "storm_used_for_prediction": False,
            "candidate_frozen_before_comparator_attachment": True,
        }
        comparator: StormDashboardComparator | None = None
        if settings.storm_native_path is not None:
            if sha256_file(settings.storm_native_path) != settings.storm_native_sha256:
                raise BenchmarkContractError("native Storm materialisation checksum mismatch")
            raw_native = _native_raw_series(
                settings.storm_native_path,
                timestamp_column=settings.storm_timestamp_column,
                value_column=settings.storm_value_column,
            )
            statistics, native_audit, comparator = attach_native_storm_for_statistics(
                statistics,
                raw_native,
                zone=identity.zone,
                timezone=identity.timezone,
                source={
                    "kind": "sealed_offline_native_exact",
                    "path": str(settings.storm_native_path),
                    "sha256": settings.storm_native_sha256,
                },
            )
            statistics_audit.update(native_audit)
        else:
            statistics_audit["storm_primary_report_benchmark"] = None
            statistics_audit["storm_dashboard"] = {
                "available": False,
                "official_dashboard_metric": False,
                "reason": (
                    "No verified native Storm dashboard series for this zone; "
                    "no official label or proxy is emitted."
                ),
            }
            statistics_audit["report_scope_note"] = (
                f"{identity.zone}: aucun benchmark Storm dashboard natif officiel "
                "vérifié; aucune série proxy n'est présentée comme officielle."
            )
        statistics_path = staging / "statistics_history_hourly.csv.gz"
        statistics.reset_index().to_csv(statistics_path, index=False, compression="gzip")
        statistics_audit["statistics_history_path"] = statistics_path.name
        statistics_audit["statistics_history_sha256"] = sha256_file(statistics_path)
        _write_json(staging / "statistics_history_audit.json", statistics_audit)
        if comparator is not None:
            storm_path = staging / STORM_DASHBOARD_ARTIFACT
            storm_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "delivery_start_utc": final_index,
                    STORM_DASHBOARD_COLUMN: comparator.values.to_numpy(float),
                }
            ).to_parquet(storm_path, index=False)

        metric_rows = _metric_rows(backtest, final_index)
        pd.DataFrame(metric_rows).to_csv(staging / "metrics_hourly.csv", index=False)
        source_metrics_path = settings.source_run / "metrics_hourly.json"
        source_metrics = _read_json(source_metrics_path, name="source metrics")
        diagnostics = dict(source_metrics.get("training_diagnostics", {}))
        diagnostics.update(
            {
                "evaluation_start_local_date": settings.final_start,
                "evaluation_end_local_date": settings.final_end,
                "zone_benchmark": candidate_audit,
                "storm_evaluation_only": (
                    dict(comparator.audit) if comparator is not None else None
                ),
            }
        )
        _write_json(
            staging / "metrics_hourly.json",
            {
                "metrics": metric_rows,
                "ensemble_weights": source_metrics.get("ensemble_weights", {}),
                "training_diagnostics": diagnostics,
                "forecast_diagnostics": {
                    "n_forecast_hours": int(len(forecast)),
                    "forecast_start_utc": str(live_index[0]),
                    "forecast_end_utc": str(live_index[-1]),
                    "model": native_model,
                    "recipe_mode": mode,
                },
            },
        )
        baseline_model = source_model if mode == "mkonline_blend" else (
            "ensemble" if "ensemble__q50" in backtest else "chronos2"
        )
        summary, daily, monthly, hourly = evaluate(
            backtest.loc[final_index].reset_index(),
            baseline=f"{baseline_model}__q50",
            candidate=f"{native_model}__q50",
            actual="actual",
            timezone=identity.timezone,
            bootstrap_samples=settings.bootstrap_samples,
            seed=42,
        )
        _write_json(staging / "evaluation_summary.json", summary)
        daily.to_csv(staging / "evaluation_by_day.csv", index=False)
        monthly.to_csv(staging / "evaluation_by_month.csv", index=False)
        hourly.to_csv(staging / "evaluation_by_hour.csv", index=False)
        if comparator is not None:
            storm_frame = backtest.loc[final_index].copy()
            storm_frame[STORM_DASHBOARD_COLUMN] = comparator.values.reindex(
                final_index
            ).to_numpy(float)
            storm_summary, storm_daily, storm_monthly, storm_hourly = (
                _evaluate_comparator_pairs(
                storm_frame,
                baseline=STORM_DASHBOARD_COLUMN,
                candidate=f"{native_model}__q50",
                timezone=identity.timezone,
                )
            )
            _write_json(staging / "evaluation_vs_storm_summary.json", storm_summary)
            storm_daily.to_csv(staging / "evaluation_vs_storm_by_day.csv", index=False)
            storm_monthly.to_csv(staging / "evaluation_vs_storm_by_month.csv", index=False)
            storm_hourly.to_csv(staging / "evaluation_vs_storm_by_hour.csv", index=False)

        shutil.copy2(settings.recipe_path, staging / "zone_benchmark_recipe.json")
        if settings.dependency_path is not None:
            shutil.copy2(
                settings.dependency_path,
                staging / "mkonline_primary_dependency.json",
            )
        if primary_audit is not None:
            _write_json(staging / "mkonline_pit_audit.json", primary_audit)
        manifest = dict(source_manifest)
        manifest.update(
            {
                "script_version": "1.0.0-zone-benchmark-sealed-offline",
                "run_type": "sealed_zone_benchmark",
                "config": str(settings.config_path),
                "source_run": str(settings.source_run),
                "zone": identity.zone,
                "timezone": identity.timezone,
                "target_series": identity.target_series,
                "model_id": f"chronos2_{identity.zone.lower()}_{mode}",
                "native_model": native_model,
                "baseline_model": baseline_model,
                "recipe_mode": mode,
                "prediction_inputs": (
                    ["autonomous_extended_residual", identity.primary_series]
                    if mode == "mkonline_blend"
                    else ["autonomous_extended_residual"]
                ),
                "external_price_forecasts_loaded": (
                    [identity.primary_series] if mode == "mkonline_blend" else []
                ),
                "evaluation_only_comparators": (
                    [identity.storm_dashboard_series]
                    if identity.storm_dashboard_series is not None
                    else []
                ),
                "storm_evaluation_only_loaded_after_candidate_frozen": (
                    comparator is not None
                ),
                "storm_used_as_feature": False,
                "uses_legacy_price_forecast": False,
                "fallback_used": False,
                "n_evaluation_hours": int(len(final_index)),
                "n_evaluation_days": int(
                    len(pd.Index(final_index.tz_convert(identity.timezone).date).unique())
                ),
                "evaluation_start_local_date": settings.final_start,
                "evaluation_end_local_date": settings.final_end,
                "sha256_manifest": "artifact_checksums.json",
            }
        )
        _write_json(staging / "run_manifest.json", manifest)
        # Persist the normalized operational recipe as a distinct immutable
        # artifact.  The original selection JSON above is retained verbatim.
        operational_recipe = dict(recipe)
        operational_recipe["prediction_mode"] = mode
        operational_recipe["mkonline_enabled"] = mode == "mkonline_blend"
        if mode == "autonomous_only" and not str(
            operational_recipe.get("autonomous_only_reason", "")
        ).strip():
            validation = operational_recipe.get("autonomous_validation", {})
            if isinstance(validation, Mapping):
                operational_recipe["autonomous_only_reason"] = str(
                    validation.get("reason", "")
                ).strip()
        _write_json(staging / "zone_live_recipe.json", operational_recipe)

        report_path = staging / settings.report_filename
        write_hourly_html_report(
            staging,
            output_path=report_path,
            title=settings.report_title,
            native_model=native_model,
            baseline_model=baseline_model,
            zone=identity.zone,
            timezone=identity.timezone,
            extreme_threshold=settings.extreme_threshold,
            history_hours=settings.report_history_hours,
        )
        _write_checksums(
            staging,
            output=output,
            settings=settings,
            project_root=settings.config_path.parent,
        )
        _publish(staging, output, overwrite=overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


__all__ = (
    "BenchmarkBuildSettings",
    "BenchmarkContractError",
    "BenchmarkIdentity",
    "RECIPE_MODES",
    "attach_native_storm_for_statistics",
    "blend_quantiles",
    "build_frozen_candidate",
    "complete_local_range",
    "day_histogram",
    "expected_cutoff",
    "load_benchmark_identity",
    "load_build_settings",
    "load_primary_materialization",
    "publish_zone_benchmark",
    "sha256_file",
    "validate_dependency_manifest",
    "validate_recipe_manifest",
    "verify_source_run",
)
