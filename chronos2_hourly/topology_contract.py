"""Strict contract for the PriceFM-inspired topology experiment.

This module is deliberately read-only.  It validates a YAML declaration,
checksum-sealed autonomous sources, the immutable 365-day protocol, and safe
experiment output paths.  Storm is represented only by a post-freeze policy:
the contract contains no Storm path and this module has no Storm loader.

The topology layer has two trained radii (0=self-only, 1=self+direct
neighbours).  Both are distinct from the identity fallback, which applies no
layer.  MKOnline is accepted only in a separate, post-autonomous blend for FR
and NL; it is forbidden from every topology prediction input.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


SCHEMA_VERSION = 1
SUPPORTED_ZONES = ("FR", "DE", "BE", "NL", "ES")
SUPPORTED_VARIANTS = ("autonomous", "mkonline_blend")
BLEND_ZONES = ("FR", "NL")
RADIUS_CANDIDATES = (0, 1)
FORBIDDEN_TOPOLOGY_INPUT_TOKENS = ("storm", "mkonline")

ZONE_TIMEZONES: Mapping[str, str] = MappingProxyType(
    {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }
)

# Direct links in the five-zone induced subgraph from PriceFM Table V.
EXPECTED_ADJACENCY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "FR": ("BE", "DE", "ES"),
        "DE": ("BE", "FR", "NL"),
        "BE": ("DE", "FR", "NL"),
        "NL": ("BE", "DE"),
        "ES": ("FR",),
    }
)

EXPECTED_RESIDUAL_LOAD_COLUMNS: Mapping[str, str] = MappingProxyType(
    {zone: f"{zone.lower()}_residual_load_fcst" for zone in SUPPORTED_ZONES}
)

TOPOLOGY_CONTEXT_COLUMNS = tuple(
    f"topology__{family}__{suffix}"
    for family in ("residual_load", "price_da_lag24h")
    for suffix in (
        "local",
        "pool_mean",
        "pool_count",
        "pool_coverage",
        "local_minus_pool",
    )
) + (
    "calendar_local_hour",
    "calendar_weekday",
    "calendar_is_weekend",
    "calendar_hour_sin",
    "calendar_hour_cos",
    "calendar_weekday_sin",
    "calendar_weekday_cos",
    "calendar_dayofyear_sin",
    "calendar_dayofyear_cos",
    "calendar_dst_fold",
    "calendar_is_dst",
    "calendar_utc_offset_hours",
)

# The first formal gate was replaced after it had been inspected.  The exact
# replacement protocol below is a governance invariant, not a default that a
# caller may override.
SPLIT_DAY_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "seed": 120,
        "A": 65,
        "development": 60,
        "B1": 30,
        "B2": 30,
        "final": 60,
    }
)
EXPECTED_TOTAL_DAYS = 365
EXPECTED_MINIMUM_PIT_COVERAGE = 0.90
FORMAL_GATE_HALF_DAYS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"B1": (15, 15), "B2": (15, 15), "final": (30, 30)}
)
EXPECTED_MODEL_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "backend": "HistGradientBoostingRegressor",
        "loss": "absolute_error",
        "learning_rate": 0.03,
        "max_iter": 100,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 240,
        "l2_regularization": 50.0,
        "early_stopping": False,
        "random_state": 120,
        "correction_clip_eur_mwh": 10.0,
        "min_training_rows": 168,
        "same_hyperparameters_for_radii": True,
    }
)
EXPECTED_CORRECTION_SCALE_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

_ROOT_KEYS = {
    "schema_version",
    "experiment_id",
    "output_directory",
    "sources",
    "topology",
    "model",
    "protocol",
    "variants",
}
_SOURCE_KEYS = {
    "timezone",
    "autonomous_run",
    "checksum_manifest_sha256",
    "target_cache",
    "target_cache_sha256",
}
_TOPOLOGY_KEYS = {
    "radii",
    "adjacency",
    "residual_load",
    "price",
    "minimum_pit_coverage",
    "missing_policy",
    "fallback",
}
_RESIDUAL_LOAD_KEYS = {"columns", "source", "interpolation"}
_PRICE_KEYS = {
    "column",
    "lag_hours",
    "lag_basis",
    "source_local_day_strictly_before_delivery",
}
_MODEL_KEYS = set(EXPECTED_MODEL_VALUES) | {"correction_scale_grid"}
_PROTOCOL_KEYS = {
    "start_local_day",
    "splits",
    "selection_min_gain_eur_mwh",
    "training",
    "gates",
    "storm",
}
_TRAINING_KEYS = {
    "selection_fit_blocks",
    "selection_score_block",
    "selection_includes_identity",
    "selection_radius_candidates",
    "selection_correction_scale_grid",
    "development_score_block",
    "development_used_for_formal_gate",
    "formal_gate_periods_unopened_before_freeze",
    "recipe_and_config_frozen_before_formal_gates",
    "gate_fit_blocks",
    "gate_score_blocks",
    "refit_between_gate_blocks",
    "b2_refit_add_blocks",
    "final_fit_blocks",
    "final_score_block",
    "single_final_opening",
}
_GATE_KEYS = {
    "minimum_mae_gain_eur_mwh",
    "require_positive_halves",
    "bootstrap_samples",
    "bootstrap_seed",
    "bootstrap_confidence",
    "require_bootstrap_lower_bound_positive",
}
_STORM_KEYS = {
    "evaluation_only",
    "load_after_candidate_freeze",
    "used_for_selection_or_tuning",
}
_AUTONOMOUS_VARIANT_KEYS = {
    "zones",
    "native_model",
    "baseline_model",
    "prediction_inputs",
}
_BLEND_VARIANT_KEYS = {
    "zones",
    "native_model",
    "baseline_model",
    "recalibrate_after_autonomous_gates",
    "weight_fit_block",
    "weight_fit_method",
    "weight_grid_step",
    "veto_blocks",
    "final_targets_used_for_weight_or_hyperparameters",
    "cutoff_timezone",
    "dependencies",
}
_BLEND_DEPENDENCY_KEYS = {
    "recipe_manifest",
    "recipe_manifest_sha256",
    "dependency_manifest",
    "dependency_manifest_sha256",
    "forecast_file",
    "forecast_file_sha256",
    "primary_series",
    "current_weight_autonomous",
    "current_weight_mkonline",
}

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class TopologyContractError(ValueError):
    """Raised when the topology experiment is unsafe or non-reproducible."""


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start_local_day: date
    end_local_day: date
    days: int
    expected_physical_hours: int


@dataclass(frozen=True)
class TopologyModelSpec:
    backend: str
    loss: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    early_stopping: bool
    random_state: int
    correction_clip_eur_mwh: float
    min_training_rows: int
    correction_scale_grid: tuple[float, ...]
    same_hyperparameters_for_radii: bool
    sha256: str


@dataclass(frozen=True)
class GatePolicy:
    minimum_mae_gain_eur_mwh: float
    require_positive_halves: bool
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_confidence: float
    require_bootstrap_lower_bound_positive: bool


@dataclass(frozen=True)
class SealedAutonomousSource:
    zone: str
    timezone: str
    run_directory: Path
    checksum_manifest: Path
    checksum_manifest_sha256: str
    backtest_file: Path
    backtest_sha256: str
    aligned_inputs_file: Path
    aligned_inputs_sha256: str
    run_manifest_file: Path
    run_manifest_sha256: str
    feature_manifest_file: Path
    feature_manifest_sha256: str
    recipe_file: Path
    recipe_sha256: str
    metrics_file: Path
    metrics_sha256: str
    target_cache: Path
    target_cache_sha256: str


@dataclass(frozen=True)
class BlendDependency:
    zone: str
    recipe_manifest: Path
    recipe_manifest_sha256: str
    dependency_manifest: Path
    dependency_manifest_sha256: str
    forecast_file: Path
    forecast_file_sha256: str
    primary_series: str
    current_weight_autonomous: float
    current_weight_mkonline: float


@dataclass(frozen=True)
class TopologyExperimentContract:
    schema_version: int
    experiment_id: str
    source_path: Path
    source_sha256: str
    project_root: Path
    output_directory: Path
    sources: Mapping[str, SealedAutonomousSource]
    adjacency: Mapping[str, tuple[str, ...]]
    residual_load_columns: Mapping[str, str]
    radii: tuple[int, ...]
    minimum_pit_coverage: float
    model: TopologyModelSpec
    start_local_day: date
    selection_min_gain_eur_mwh: float
    gates: GatePolicy
    autonomous_prediction_inputs: tuple[str, ...]
    blend_dependencies: Mapping[str, BlendDependency]
    blend_weight_grid_step: float
    feature_schema_sha256: str

    def split_windows(self, zone: str) -> tuple[SplitWindow, ...]:
        """Return exact civil-day windows and DST-aware physical hours."""

        code = _zone(zone)
        tz = ZoneInfo(ZONE_TIMEZONES[code])
        cursor = self.start_local_day
        result: list[SplitWindow] = []
        for name, days in SPLIT_DAY_COUNTS.items():
            end = cursor + timedelta(days=days - 1)
            start_dt = datetime.combine(cursor, time.min, tzinfo=tz)
            stop_dt = datetime.combine(end + timedelta(days=1), time.min, tzinfo=tz)
            hours = int(
                (stop_dt.astimezone(timezone.utc) - start_dt.astimezone(timezone.utc))
                .total_seconds()
                // 3600
            )
            result.append(SplitWindow(name, cursor, end, days, hours))
            cursor = end + timedelta(days=1)
        return tuple(result)

    def zones_within_radius(self, zone: str, radius: int) -> tuple[str, ...]:
        """Return the deterministic PriceFM mask for radius 0 or 1."""

        code = _zone(zone)
        if isinstance(radius, bool) or radius not in self.radii:
            raise TopologyContractError(
                f"radius={radius!r}; allowed={list(self.radii)}"
            )
        if radius == 0:
            return (code,)
        return (code, *self.adjacency[code])

    def output_directory_for(self, zone: str, variant: str) -> Path:
        """Resolve a safe variant output; reject unsupported blends first."""

        code = _zone(zone)
        name = str(variant).strip().lower()
        if name not in SUPPORTED_VARIANTS:
            raise TopologyContractError(f"unsupported variant={variant!r}")
        if name == "mkonline_blend" and code not in BLEND_ZONES:
            raise TopologyContractError(
                f"{code}: MKOnline blend is forbidden; allowed={list(BLEND_ZONES)}"
            )
        resolved = (self.output_directory / code.lower() / name).resolve()
        if not resolved.is_relative_to(self.output_directory):
            raise TopologyContractError("variant output escaped experiment root")
        return resolved

    def blend_dependency_for(self, zone: str) -> BlendDependency:
        """Return a blend dependency after the zone eligibility guard."""

        code = _zone(zone)
        if code not in BLEND_ZONES:
            raise TopologyContractError(
                f"{code}: MKOnline dependency must not be read"
            )
        return self.blend_dependencies[code]


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TopologyContractError(f"unreadable sealed file: {path}") from exc
    return digest.hexdigest()


def _verify_declared_file(path: Path, expected_sha256: str, *, name: str) -> None:
    """Fail closed on a missing/tampered declaration before parsing its content."""

    if not path.is_file():
        raise TopologyContractError(f"missing sealed {name}: {path}")
    if _sha256_file(path) != expected_sha256:
        raise TopologyContractError(f"{name} hash mismatch: {path}")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TopologyContractError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TopologyContractError(f"{name} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise TopologyContractError(
            f"{name} invalid schema: missing={missing}, unknown={unknown}"
        )


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopologyContractError(f"{name} must be a non-empty string")
    return value.strip()


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TopologyContractError(f"{name} must be true or false")
    return value


def _integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TopologyContractError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TopologyContractError(f"{name} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise TopologyContractError(f"{name} must be >= {minimum}")
    return result


def _sha(value: Any, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if not _SHA_RE.fullmatch(result):
        raise TopologyContractError(f"{name} must be a lowercase SHA-256")
    return result


def _zone(value: Any) -> str:
    if not isinstance(value, str):
        raise TopologyContractError("zone must be a string")
    code = value.strip().upper()
    if code not in SUPPORTED_ZONES:
        raise TopologyContractError(f"unsupported zone={value!r}")
    return code


def _resolve_inside(value: Any, *, base: Path, root: Path, name: str) -> Path:
    raw = _text(value, name=name)
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else base / path).resolve()
    allowed = root.resolve()
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise TopologyContractError(f"{name} must stay below {allowed}")
    return resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TopologyContractError(f"unreadable topology YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise TopologyContractError("topology YAML must contain a mapping")
    return payload


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyContractError(f"{name} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TopologyContractError(f"{name} must contain a JSON object")
    return payload


def _forbid_prediction_tokens(values: Sequence[Any], *, name: str) -> None:
    offenders = sorted(
        {
            str(value)
            for value in values
            if any(
                token in str(value).casefold()
                for token in FORBIDDEN_TOPOLOGY_INPUT_TOKENS
            )
        }
    )
    if offenders:
        raise TopologyContractError(
            f"{name} contains forbidden prediction input(s): {offenders}"
        )


def validate_prediction_input_names(values: Sequence[Any]) -> tuple[str, ...]:
    names = tuple(_text(value, name="prediction input") for value in values)
    if not names or len(names) != len(set(names)):
        raise TopologyContractError("prediction inputs must be unique and non-empty")
    _forbid_prediction_tokens(names, name="prediction inputs")
    return names


def _artifact_member(
    run_dir: Path,
    manifest: Mapping[str, Any],
    relative: str,
    *,
    role: str,
    verify_hashes: bool,
) -> tuple[Path, str]:
    normalized = relative.replace("\\", "/")
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping)
        and str(item.get("path", "")).replace("\\", "/") == normalized
        and item.get("role") == role
    ]
    if len(matches) != 1:
        raise TopologyContractError(
            f"{run_dir}: expected one sealed {role} record for {relative}"
        )
    expected = _sha(matches[0].get("sha256"), name=f"{relative} checksum")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
        raise TopologyContractError(f"missing sealed artifact: {path}")
    if verify_hashes and _sha256_file(path) != expected:
        raise TopologyContractError(f"sealed artifact checksum mismatch: {path}")
    return path, expected


def _parse_topology(raw: Any) -> tuple[Mapping[str, tuple[str, ...]], float]:
    item = _mapping(raw, name="topology")
    _exact_keys(item, _TOPOLOGY_KEYS, name="topology")
    radii = tuple(item["radii"])
    if radii != RADIUS_CANDIDATES:
        raise TopologyContractError("topology.radii must be exactly [0, 1]")

    adjacency_raw = _mapping(item["adjacency"], name="topology.adjacency")
    if {_zone(key) for key in adjacency_raw} != set(SUPPORTED_ZONES):
        raise TopologyContractError("topology.adjacency must define all five zones")
    adjacency: dict[str, tuple[str, ...]] = {}
    for zone in SUPPORTED_ZONES:
        key = next(key for key in adjacency_raw if _zone(key) == zone)
        neighbours = tuple(_zone(value) for value in _sequence(
            adjacency_raw[key], name=f"topology.adjacency.{zone}"
        ))
        if zone in neighbours or len(neighbours) != len(set(neighbours)):
            raise TopologyContractError(f"invalid adjacency for {zone}")
        if set(neighbours) != set(EXPECTED_ADJACENCY[zone]):
            raise TopologyContractError(
                f"topology.adjacency.{zone} differs from PriceFM five-zone graph"
            )
        adjacency[zone] = neighbours
    for zone, neighbours in adjacency.items():
        for neighbour in neighbours:
            if zone not in adjacency[neighbour]:
                raise TopologyContractError("topology adjacency must be symmetric")

    residual = _mapping(item["residual_load"], name="topology.residual_load")
    _exact_keys(residual, _RESIDUAL_LOAD_KEYS, name="topology.residual_load")
    if residual["source"] != "sealed_pit_materialization":
        raise TopologyContractError("residual load source must be sealed PIT")
    if residual["interpolation"] != "none":
        raise TopologyContractError("topology inputs cannot be interpolated")
    columns = _mapping(residual["columns"], name="topology.residual_load.columns")
    parsed_columns = {_zone(key): str(value) for key, value in columns.items()}
    if parsed_columns != dict(EXPECTED_RESIDUAL_LOAD_COLUMNS):
        raise TopologyContractError("residual-load columns must be the five PIT aliases")
    _forbid_prediction_tokens(tuple(parsed_columns.values()), name="residual-load columns")

    price = _mapping(item["price"], name="topology.price")
    _exact_keys(price, _PRICE_KEYS, name="topology.price")
    expected_price = {
        "column": "actual",
        "lag_hours": 24,
        "lag_basis": "physical_utc_rows",
        "source_local_day_strictly_before_delivery": True,
    }
    if dict(price) != expected_price:
        raise TopologyContractError(
            "price input must be physical lag-24 with a strict prior-local-day guard"
        )
    coverage = _number(
        item["minimum_pit_coverage"],
        name="topology.minimum_pit_coverage",
        minimum=0.0,
    )
    if coverage > 1.0:
        raise TopologyContractError("minimum PIT coverage must be <= 1")
    if coverage != EXPECTED_MINIMUM_PIT_COVERAGE:
        raise TopologyContractError(
            "minimum_pit_coverage is frozen at 0.90 for the topology signal"
        )
    if item["missing_policy"] != "preserve_nan":
        raise TopologyContractError("missing_policy must preserve NaN")
    if item["fallback"] != "identity":
        raise TopologyContractError("fallback must be identity (no layer)")
    return MappingProxyType(adjacency), coverage


def _parse_model(raw: Any) -> TopologyModelSpec:
    item = _mapping(raw, name="model")
    _exact_keys(item, _MODEL_KEYS, name="model")
    normalized: dict[str, Any] = {}
    for key, expected in EXPECTED_MODEL_VALUES.items():
        observed = item[key]
        if isinstance(expected, bool):
            observed = _bool(observed, name=f"model.{key}")
        elif isinstance(expected, int):
            observed = _integer(observed, name=f"model.{key}", minimum=0)
        elif isinstance(expected, float):
            observed = _number(observed, name=f"model.{key}")
        else:
            observed = _text(observed, name=f"model.{key}")
        if observed != expected:
            raise TopologyContractError(
                f"model.{key}={observed!r}; frozen value={expected!r}"
            )
        normalized[key] = observed
    grid = tuple(float(value) for value in _sequence(
        item["correction_scale_grid"], name="model.correction_scale_grid"
    ))
    if grid != EXPECTED_CORRECTION_SCALE_GRID:
        raise TopologyContractError(
            f"correction_scale_grid must be {list(EXPECTED_CORRECTION_SCALE_GRID)}"
        )
    normalized["correction_scale_grid"] = list(grid)
    return TopologyModelSpec(
        backend=str(normalized["backend"]),
        loss=str(normalized["loss"]),
        learning_rate=float(normalized["learning_rate"]),
        max_iter=int(normalized["max_iter"]),
        max_leaf_nodes=int(normalized["max_leaf_nodes"]),
        min_samples_leaf=int(normalized["min_samples_leaf"]),
        l2_regularization=float(normalized["l2_regularization"]),
        early_stopping=bool(normalized["early_stopping"]),
        random_state=int(normalized["random_state"]),
        correction_clip_eur_mwh=float(normalized["correction_clip_eur_mwh"]),
        min_training_rows=int(normalized["min_training_rows"]),
        correction_scale_grid=grid,
        same_hyperparameters_for_radii=bool(
            normalized["same_hyperparameters_for_radii"]
        ),
        sha256=canonical_sha256(normalized),
    )


def _parse_protocol(raw: Any) -> tuple[date, float, GatePolicy]:
    item = _mapping(raw, name="protocol")
    _exact_keys(item, _PROTOCOL_KEYS, name="protocol")
    try:
        start = date.fromisoformat(_text(item["start_local_day"], name="start_local_day"))
    except ValueError as exc:
        raise TopologyContractError("start_local_day must be YYYY-MM-DD") from exc
    splits = _mapping(item["splits"], name="protocol.splits")
    if dict(splits) != dict(SPLIT_DAY_COUNTS):
        raise TopologyContractError(
            "protocol.splits must be seed120/A65/development60/B1-30/B2-30/final60"
        )
    if sum(int(value) for value in splits.values()) != EXPECTED_TOTAL_DAYS:
        raise TopologyContractError("protocol must total 365 days")
    selection_min = _number(
        item["selection_min_gain_eur_mwh"],
        name="protocol.selection_min_gain_eur_mwh",
        minimum=0.0,
    )

    training = _mapping(item["training"], name="protocol.training")
    _exact_keys(training, _TRAINING_KEYS, name="protocol.training")
    expected_training = {
        "selection_fit_blocks": ["seed"],
        "selection_score_block": "A",
        "selection_includes_identity": True,
        "selection_radius_candidates": [0, 1],
        "selection_correction_scale_grid": list(EXPECTED_CORRECTION_SCALE_GRID),
        "development_score_block": "development",
        "development_used_for_formal_gate": False,
        "formal_gate_periods_unopened_before_freeze": True,
        "recipe_and_config_frozen_before_formal_gates": True,
        "gate_fit_blocks": ["seed", "A", "development"],
        "gate_score_blocks": ["B1", "B2"],
        "refit_between_gate_blocks": True,
        "b2_refit_add_blocks": ["B1"],
        "final_fit_blocks": ["seed", "A", "development", "B1", "B2"],
        "final_score_block": "final",
        "single_final_opening": True,
    }
    if dict(training) != expected_training:
        raise TopologyContractError(
            "protocol.training violates the frozen development/B1/B2/final order"
        )

    gates = _mapping(item["gates"], name="protocol.gates")
    _exact_keys(gates, _GATE_KEYS, name="protocol.gates")
    gain = _number(
        gates["minimum_mae_gain_eur_mwh"],
        name="protocol.gates.minimum_mae_gain_eur_mwh",
        minimum=0.0,
    )
    if gain != 0.05:
        raise TopologyContractError("minimum_mae_gain_eur_mwh is frozen at 0.05")
    halves = _bool(
        gates["require_positive_halves"],
        name="protocol.gates.require_positive_halves",
    )
    samples = _integer(
        gates["bootstrap_samples"],
        name="protocol.gates.bootstrap_samples",
    )
    if samples != 20_000:
        raise TopologyContractError("bootstrap_samples must be 20000")
    seed = _integer(
        gates["bootstrap_seed"],
        name="protocol.gates.bootstrap_seed",
        minimum=0,
    )
    confidence = _number(
        gates["bootstrap_confidence"],
        name="protocol.gates.bootstrap_confidence",
    )
    if confidence != 0.95:
        raise TopologyContractError("bootstrap_confidence must be 0.95")
    lower = _bool(
        gates["require_bootstrap_lower_bound_positive"],
        name="protocol.gates.require_bootstrap_lower_bound_positive",
    )
    if not halves or not lower:
        raise TopologyContractError("positive halves and bootstrap lower bound are mandatory")

    storm = _mapping(item["storm"], name="protocol.storm")
    _exact_keys(storm, _STORM_KEYS, name="protocol.storm")
    if dict(storm) != {
        "evaluation_only": True,
        "load_after_candidate_freeze": True,
        "used_for_selection_or_tuning": False,
    }:
        raise TopologyContractError("Storm must be evaluation-only and post-freeze")
    return start, selection_min, GatePolicy(gain, halves, samples, seed, confidence, lower)


def _parse_source(
    zone: str,
    raw: Any,
    *,
    project_root: Path,
    verify_hashes: bool,
) -> SealedAutonomousSource:
    item = _mapping(raw, name=f"sources.{zone}")
    _exact_keys(item, _SOURCE_KEYS, name=f"sources.{zone}")
    tz = _text(item["timezone"], name=f"sources.{zone}.timezone")
    if tz != ZONE_TIMEZONES[zone]:
        raise TopologyContractError(f"{zone}: timezone must be {ZONE_TIMEZONES[zone]}")
    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - environment guard
        raise TopologyContractError(f"unknown timezone {tz}") from exc

    target_root = (project_root / "data" / "cache" / zone.lower()).resolve()
    target_cache = _resolve_inside(
        item["target_cache"],
        base=project_root,
        root=target_root,
        name=f"sources.{zone}.target_cache",
    )
    if (
        target_cache.parent != target_root
        or not target_cache.name.startswith("target__")
        or not target_cache.name.endswith(".csv.gz")
    ):
        raise TopologyContractError(
            f"{zone}: target_cache must be one exact target__*.csv.gz in {target_root}"
        )
    target_cache_sha = _sha(
        item["target_cache_sha256"], name=f"sources.{zone}.target_cache_sha256"
    )
    if not target_cache.is_file():
        raise TopologyContractError(f"{zone}: missing sealed target cache: {target_cache}")
    if verify_hashes and _sha256_file(target_cache) != target_cache_sha:
        raise TopologyContractError(f"{zone}: target cache hash mismatch")

    run_root = (project_root / "runs").resolve()
    run_dir = _resolve_inside(
        item["autonomous_run"],
        base=project_root,
        root=run_root,
        name=f"sources.{zone}.autonomous_run",
    )
    expected_name = f"chronos2_hourly_{zone.lower()}_residual_extended_v1"
    if run_dir.name != expected_name or any(
        run_dir.is_relative_to((run_root / part).resolve())
        for part in ("live", "tmp", "experiments")
    ):
        raise TopologyContractError(f"{zone}: source must be canonical autonomous run")
    checksum_manifest = run_dir / "artifact_checksums.json"
    configured_sha = _sha(
        item["checksum_manifest_sha256"],
        name=f"sources.{zone}.checksum_manifest_sha256",
    )
    if not checksum_manifest.is_file():
        raise TopologyContractError(f"missing checksum manifest: {checksum_manifest}")
    if verify_hashes and _sha256_file(checksum_manifest) != configured_sha:
        raise TopologyContractError(f"{zone}: checksum manifest hash mismatch")
    manifest = _load_json(checksum_manifest, name=f"{zone} checksum manifest")
    backtest, backtest_sha = _artifact_member(
        run_dir, manifest, "backtest_hourly_oof.csv.gz", role="run_artifact", verify_hashes=verify_hashes
    )
    aligned, aligned_sha = _artifact_member(
        run_dir, manifest, "inputs/aligned_inputs.csv.gz", role="materialized_input", verify_hashes=verify_hashes
    )
    run_manifest, run_manifest_sha = _artifact_member(
        run_dir, manifest, "run_manifest.json", role="run_artifact", verify_hashes=verify_hashes
    )
    feature_manifest, feature_manifest_sha = _artifact_member(
        run_dir, manifest, "feature_manifest.csv", role="run_artifact", verify_hashes=verify_hashes
    )
    recipe, recipe_sha = _artifact_member(
        run_dir, manifest, "extended_residual_recipe.json", role="run_artifact", verify_hashes=verify_hashes
    )
    metrics, metrics_sha = _artifact_member(
        run_dir, manifest, "metrics_hourly.json", role="run_artifact", verify_hashes=verify_hashes
    )

    run_payload = _load_json(run_manifest, name=f"{zone} run manifest")
    if str(run_payload.get("zone", "")).upper() != zone:
        raise TopologyContractError(f"{zone}: source run zone mismatch")
    if run_payload.get("timezone") != tz:
        raise TopologyContractError(f"{zone}: source run timezone mismatch")
    if run_payload.get("target_contract") != "hourly_utc_no_interpolation":
        raise TopologyContractError(f"{zone}: unsafe target contract")
    if run_payload.get("delivery_horizon") != "dynamic_23_24_25":
        raise TopologyContractError(f"{zone}: unsafe delivery horizon")
    if run_payload.get("external_price_forecasts_loaded", []) != []:
        raise TopologyContractError(f"{zone}: autonomous source loaded a price expert")
    if run_payload.get("storm_used_as_feature") is True:
        raise TopologyContractError(f"{zone}: Storm was used as a feature")
    _forbid_prediction_tokens(
        tuple(run_payload.get("prediction_inputs", []) or []),
        name=f"{zone} sealed prediction inputs",
    )
    _forbid_prediction_tokens(
        tuple(run_payload.get("active_features", []) or []),
        name=f"{zone} sealed active features",
    )
    with feature_manifest.open("r", encoding="utf-8", newline="") as handle:
        features = [row.get("feature", "") for row in csv.DictReader(handle)]
    _forbid_prediction_tokens(features, name=f"{zone} sealed feature manifest")
    recipe_payload = _load_json(recipe, name=f"{zone} residual recipe")
    if recipe_payload.get("external_price_forecasts_loaded", []) != []:
        raise TopologyContractError(f"{zone}: residual recipe loaded a price expert")
    protocol = recipe_payload.get("protocol", {})
    if not isinstance(protocol, Mapping) or protocol.get(
        "final_targets_used_for_evaluation_fit"
    ) is not False:
        raise TopologyContractError(f"{zone}: source final targets were not sealed")

    return SealedAutonomousSource(
        zone=zone,
        timezone=tz,
        run_directory=run_dir,
        checksum_manifest=checksum_manifest,
        checksum_manifest_sha256=configured_sha,
        backtest_file=backtest,
        backtest_sha256=backtest_sha,
        aligned_inputs_file=aligned,
        aligned_inputs_sha256=aligned_sha,
        run_manifest_file=run_manifest,
        run_manifest_sha256=run_manifest_sha,
        feature_manifest_file=feature_manifest,
        feature_manifest_sha256=feature_manifest_sha,
        recipe_file=recipe,
        recipe_sha256=recipe_sha,
        metrics_file=metrics,
        metrics_sha256=metrics_sha,
        target_cache=target_cache,
        target_cache_sha256=target_cache_sha,
    )


def _parse_blend_dependency(
    zone: str,
    raw: Any,
    *,
    project_root: Path,
) -> BlendDependency:
    """Parse sealed MKOnline declarations without touching any declared file."""

    item = _mapping(raw, name=f"blend.dependencies.{zone}")
    _exact_keys(item, _BLEND_DEPENDENCY_KEYS, name=f"blend.dependencies.{zone}")
    recipe = _resolve_inside(
        item["recipe_manifest"],
        base=project_root,
        root=project_root,
        name=f"blend.dependencies.{zone}.recipe_manifest",
    )
    expected_recipe_name = f"mkonline_{zone.lower()}_blend_recipe_v1.json"
    if recipe.parent != project_root or recipe.name != expected_recipe_name:
        raise TopologyContractError(
            f"{zone}: recipe_manifest must be the canonical root file "
            f"{expected_recipe_name}"
        )
    recipe_sha = _sha(
        item["recipe_manifest_sha256"], name="blend recipe manifest SHA"
    )
    manifest = _resolve_inside(
        item["dependency_manifest"], base=project_root, root=project_root,
        name=f"blend.dependencies.{zone}.dependency_manifest",
    )
    manifest_sha = _sha(item["dependency_manifest_sha256"], name="dependency manifest SHA")
    forecast = _resolve_inside(
        item["forecast_file"], base=project_root, root=project_root / "runs",
        name=f"blend.dependencies.{zone}.forecast_file",
    )
    forecast_sha = _sha(item["forecast_file_sha256"], name="blend forecast SHA")
    primary = _text(item["primary_series"], name="blend primary_series")
    weight_autonomous = _number(
        item["current_weight_autonomous"],
        name=f"blend.dependencies.{zone}.current_weight_autonomous",
        minimum=0.0,
    )
    weight_mkonline = _number(
        item["current_weight_mkonline"],
        name=f"blend.dependencies.{zone}.current_weight_mkonline",
        minimum=0.0,
    )
    if weight_autonomous > 1.0 or weight_mkonline > 1.0:
        raise TopologyContractError(f"{zone}: current blend weights must be <= 1")
    if abs((weight_autonomous + weight_mkonline) - 1.0) > 1e-12:
        raise TopologyContractError(f"{zone}: current blend weights must sum to 1")
    return BlendDependency(
        zone=zone,
        recipe_manifest=recipe,
        recipe_manifest_sha256=recipe_sha,
        dependency_manifest=manifest,
        dependency_manifest_sha256=manifest_sha,
        forecast_file=forecast,
        forecast_file_sha256=forecast_sha,
        primary_series=primary,
        current_weight_autonomous=weight_autonomous,
        current_weight_mkonline=weight_mkonline,
    )


def load_topology_contract(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> TopologyExperimentContract:
    """Load the strict YAML contract without writing or loading Storm."""

    source_path = Path(path).expanduser().resolve()
    payload = _load_yaml(source_path)
    _exact_keys(payload, _ROOT_KEYS, name="topology contract")
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise TopologyContractError(f"schema_version must be {SCHEMA_VERSION}")
    experiment_id = _text(payload["experiment_id"], name="experiment_id")
    if not _ID_RE.fullmatch(experiment_id):
        raise TopologyContractError("experiment_id must be lower snake_case")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else source_path.parent.parent.resolve()
    )
    output = _resolve_inside(
        payload["output_directory"], base=root, root=root / "runs" / "experiments",
        name="output_directory",
    )
    if output.name != experiment_id:
        raise TopologyContractError("output_directory must end with experiment_id")

    adjacency, minimum_coverage = _parse_topology(payload["topology"])
    model = _parse_model(payload["model"])
    start, selection_min, gates = _parse_protocol(payload["protocol"])

    raw_sources = _mapping(payload["sources"], name="sources")
    if {_zone(key) for key in raw_sources} != set(SUPPORTED_ZONES) or len(raw_sources) != 5:
        raise TopologyContractError("sources must define exactly FR, DE, BE, NL, ES")
    sources: dict[str, SealedAutonomousSource] = {}
    for zone in SUPPORTED_ZONES:
        key = next(key for key in raw_sources if _zone(key) == zone)
        sources[zone] = _parse_source(
            zone, raw_sources[key], project_root=root, verify_hashes=verify_hashes
        )

    variants = _mapping(payload["variants"], name="variants")
    if set(variants) != set(SUPPORTED_VARIANTS):
        raise TopologyContractError("variants must define autonomous and mkonline_blend")
    autonomous = _mapping(variants["autonomous"], name="variants.autonomous")
    _exact_keys(autonomous, _AUTONOMOUS_VARIANT_KEYS, name="variants.autonomous")
    auto_zones = tuple(_zone(value) for value in _sequence(autonomous["zones"], name="autonomous zones"))
    if auto_zones != SUPPORTED_ZONES:
        raise TopologyContractError("autonomous variant must use all zones in canonical order")
    if autonomous["native_model"] != "topology_autonomous" or autonomous["baseline_model"] != "residual_corrected":
        raise TopologyContractError("autonomous model names are frozen")
    prediction_inputs = validate_prediction_input_names(
        _sequence(autonomous["prediction_inputs"], name="autonomous prediction_inputs")
    )

    blend = _mapping(variants["mkonline_blend"], name="variants.mkonline_blend")
    _exact_keys(blend, _BLEND_VARIANT_KEYS, name="variants.mkonline_blend")
    blend_zones = tuple(_zone(value) for value in _sequence(blend["zones"], name="blend zones"))
    if blend_zones != BLEND_ZONES:
        raise TopologyContractError("blend zones must be exactly FR, NL")
    expected_blend_scalars = {
        "native_model": "topology_mkonline_blend",
        "baseline_model": "mkonline_blend",
        "recalibrate_after_autonomous_gates": True,
        "weight_fit_block": "A",
        "weight_fit_method": "constrained_l1_grid",
        "veto_blocks": ["B1", "B2"],
        "final_targets_used_for_weight_or_hyperparameters": False,
        "cutoff_timezone": "Europe/Paris",
    }
    for key, expected in expected_blend_scalars.items():
        if blend[key] != expected:
            raise TopologyContractError(f"variants.mkonline_blend.{key} must be {expected!r}")
    grid_step = _number(blend["weight_grid_step"], name="blend.weight_grid_step")
    if grid_step not in {0.025, 0.05}:
        raise TopologyContractError("weight_grid_step must be 0.025 or 0.05")
    raw_dependencies = _mapping(blend["dependencies"], name="blend.dependencies")
    if {_zone(key) for key in raw_dependencies} != set(BLEND_ZONES) or len(raw_dependencies) != 2:
        raise TopologyContractError("blend dependencies must define exactly FR and NL")
    dependencies: dict[str, BlendDependency] = {}
    for zone in BLEND_ZONES:
        key = next(key for key in raw_dependencies if _zone(key) == zone)
        dependencies[zone] = _parse_blend_dependency(
            zone, raw_dependencies[key], project_root=root
        )

    return TopologyExperimentContract(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        project_root=root,
        output_directory=output,
        sources=MappingProxyType(sources),
        adjacency=adjacency,
        residual_load_columns=EXPECTED_RESIDUAL_LOAD_COLUMNS,
        radii=RADIUS_CANDIDATES,
        minimum_pit_coverage=minimum_coverage,
        model=model,
        start_local_day=start,
        selection_min_gain_eur_mwh=selection_min,
        gates=gates,
        autonomous_prediction_inputs=prediction_inputs,
        blend_dependencies=MappingProxyType(dependencies),
        blend_weight_grid_step=grid_step,
        feature_schema_sha256=canonical_sha256(list(TOPOLOGY_CONTEXT_COLUMNS)),
    )


def _expected_index(contract: TopologyExperimentContract, zone: str):
    import pandas as pd

    tz = ZONE_TIMEZONES[zone]
    start = pd.Timestamp(contract.start_local_day, tz=tz).tz_convert("UTC")
    stop = pd.Timestamp(
        contract.start_local_day + timedelta(days=EXPECTED_TOTAL_DAYS), tz=tz
    ).tz_convert("UTC")
    return pd.date_range(start, stop, freq="h", inclusive="left", name="delivery_start_utc")


def audit_topology_sources(
    contract: TopologyExperimentContract,
) -> dict[str, dict[str, Any]]:
    """Validate the exact frozen 365-day OOF and sealed PIT materialisations."""

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise TopologyContractError("pandas and numpy are required for source audit") from exc

    audits: dict[str, dict[str, Any]] = {}
    common_index = None
    required = [
        "delivery_start_utc",
        "forecast_origin_utc",
        "actual",
        "residual_corrected__q10",
        "residual_corrected__q50",
        "residual_corrected__q90",
    ]
    for zone, source in contract.sources.items():
        # This cache is the sole reproducible provenance for price_da_lag24h.
        # Its bytes are authenticated before pandas is allowed to parse them.
        _verify_declared_file(
            source.target_cache,
            source.target_cache_sha256,
            name=f"{zone} target cache",
        )
        raw = pd.read_csv(source.backtest_file, usecols=required)
        index = pd.DatetimeIndex(
            pd.to_datetime(raw.pop("delivery_start_utc"), utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        raw.index = index
        expected = _expected_index(contract, zone)
        selected = raw.loc[(raw.index >= expected[0]) & (raw.index <= expected[-1])]
        if not selected.index.equals(expected):
            raise TopologyContractError(f"{zone}: sealed OOF 365-day timeline is not exact")
        if common_index is None:
            common_index = expected
        elif not expected.equals(common_index):
            raise TopologyContractError("zone OOF timelines are not simultaneous")
        numeric_columns = [column for column in required if column not in {"delivery_start_utc", "forecast_origin_utc"}]
        numeric = selected[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise TopologyContractError(f"{zone}: non-finite sealed target/prediction")
        if bool((numeric["residual_corrected__q10"] > numeric["residual_corrected__q50"]).any()) or bool(
            (numeric["residual_corrected__q50"] > numeric["residual_corrected__q90"]).any()
        ):
            raise TopologyContractError(f"{zone}: autonomous quantiles cross")
        origins = pd.DatetimeIndex(pd.to_datetime(selected["forecast_origin_utc"], utc=True, errors="raise"))
        expected_origins = pd.DatetimeIndex(
            [
                pd.Timestamp(day - timedelta(days=1), tz=source.timezone)
                .replace(hour=8)
                .tz_convert("UTC")
                for day in expected.tz_convert(source.timezone).date
            ]
        )
        origin_violations = int(np.count_nonzero(origins.asi8 != expected_origins.asi8))
        if origin_violations:
            raise TopologyContractError(f"{zone}: forecast-origin cutoff violations")

        target_raw = pd.read_csv(source.target_cache, usecols=["timestamp", "value"])
        target_index = pd.DatetimeIndex(
            pd.to_datetime(target_raw.pop("timestamp"), utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        if target_index.has_duplicates:
            raise TopologyContractError(f"{zone}: target cache has duplicate timestamps")
        target_values = pd.Series(
            pd.to_numeric(target_raw["value"], errors="coerce").to_numpy(float),
            index=target_index,
        ).sort_index()
        lag_index = expected - pd.Timedelta(hours=24)
        lagged_prices = target_values.reindex(lag_index).to_numpy(float)
        delivery_days = expected.tz_convert(source.timezone).date
        lag_days = lag_index.tz_convert(source.timezone).date
        strict_prior_day = np.asarray(
            [
                source_day < delivery_day
                for source_day, delivery_day in zip(lag_days, delivery_days)
            ],
            dtype=bool,
        )
        if any(source_day > delivery_day for source_day, delivery_day in zip(lag_days, delivery_days)):
            raise TopologyContractError(f"{zone}: price lag travels into a future local day")
        if not np.isfinite(lagged_prices[strict_prior_day]).all():
            raise TopologyContractError(f"{zone}: target cache cannot reproduce price_da_lag24h")
        # A 25-hour autumn day has one physical t-24 timestamp in the same
        # civil day.  The core masks it to NaN; this is a causal guard, not a
        # missing-source failure.
        masked_same_day_hours = int(np.count_nonzero(~strict_prior_day))
        contemporaneous = target_values.reindex(expected).to_numpy(float)
        if not np.isfinite(contemporaneous).all() or not np.allclose(
            contemporaneous,
            numeric["actual"].to_numpy(float),
            rtol=0.0,
            # OOF targets were serialized from float32 while cache values
            # retain their source decimals; this tolerance is far below the
            # market tick and only absorbs representation noise.
            atol=1e-4,
        ):
            raise TopologyContractError(f"{zone}: target cache differs from sealed actual prices")

        context = pd.read_csv(source.aligned_inputs_file)
        context_index = pd.DatetimeIndex(
            pd.to_datetime(context.pop("timestamp"), utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        context.index = context_index
        block = context.reindex(expected)
        missing_columns = sorted(set(contract.residual_load_columns.values()) - set(block.columns))
        if missing_columns:
            raise TopologyContractError(f"{zone}: missing PIT columns {missing_columns}")
        _forbid_prediction_tokens(tuple(block.columns), name=f"{zone} PIT schema")
        coverage = {
            alias: float(pd.to_numeric(block[column], errors="coerce").notna().mean())
            for alias, column in contract.residual_load_columns.items()
        }
        below = {key: value for key, value in coverage.items() if value < contract.minimum_pit_coverage}
        if below:
            raise TopologyContractError(f"{zone}: PIT coverage below threshold: {below}")

        metrics = _load_json(source.metrics_file, name=f"{zone} metrics")
        diagnostics = metrics.get("training_diagnostics", {})
        if not isinstance(diagnostics, Mapping) or diagnostics.get("metric_scope") != "sealed_final_365_delivery_days":
            raise TopologyContractError(f"{zone}: source metric scope is not sealed final 365")
        audits[zone] = {
            "checksum_manifest_sha256": source.checksum_manifest_sha256,
            "backtest_sha256": source.backtest_sha256,
            "aligned_inputs_sha256": source.aligned_inputs_sha256,
            "target_cache_sha256": source.target_cache_sha256,
            "start_local_day": str(contract.start_local_day),
            "end_local_day": str(contract.start_local_day + timedelta(days=364)),
            "n_days": EXPECTED_TOTAL_DAYS,
            "n_hours": int(len(expected)),
            "forecast_origin_violations": origin_violations,
            "pit_coverage": coverage,
            "price_da_lag24h_coverage": float(np.mean(strict_prior_day)),
            "price_da_lag24h_strict_prior_local_day": True,
            "price_lag_strict_prior_day_hours": int(np.count_nonzero(strict_prior_day)),
            "price_lag_masked_same_day_hours": masked_same_day_hours,
            "storm_used_as_prediction_input": False,
            "mkonline_used_as_topology_input": False,
        }
    return audits


def audit_blend_source(
    contract: TopologyExperimentContract,
    zone: str,
    *,
    autonomous_promoted: bool,
) -> dict[str, Any]:
    """Audit MKOnline only after zone eligibility and autonomous promotion.

    The loader intentionally performs no filesystem operation on any MKOnline
    declaration.  This function is the only I/O boundary: it first enforces
    the two guards, then authenticates all three sealed files, and only then
    parses recipe/dependency/forecast content.
    """

    code = _zone(zone)
    dependency = contract.blend_dependency_for(code)  # zone guard before I/O
    if autonomous_promoted is not True:
        raise TopologyContractError(
            f"{code}: autonomous candidate must be promoted before MKOnline I/O"
        )

    _verify_declared_file(
        dependency.recipe_manifest,
        dependency.recipe_manifest_sha256,
        name=f"{code} current blend recipe",
    )
    _verify_declared_file(
        dependency.dependency_manifest,
        dependency.dependency_manifest_sha256,
        name=f"{code} MKOnline dependency manifest",
    )
    _verify_declared_file(
        dependency.forecast_file,
        dependency.forecast_file_sha256,
        name=f"{code} MKOnline forecast",
    )

    recipe = _load_json(
        dependency.recipe_manifest, name=f"{code} current blend recipe"
    )
    if str(recipe.get("zone", "")).upper() != code:
        raise TopologyContractError(f"{code}: current blend recipe zone mismatch")
    if recipe.get("status") != "frozen_before_final_opening":
        raise TopologyContractError(f"{code}: current blend recipe is not sealed")
    if recipe.get("source_autonomous_model") != "residual_corrected":
        raise TopologyContractError(f"{code}: current blend baseline is not residual_corrected")
    weights = _mapping(recipe.get("weights"), name=f"{code} current blend weights")
    recipe_autonomous = _number(
        weights.get("autonomous"), name=f"{code} recipe autonomous weight"
    )
    recipe_mkonline = _number(
        weights.get("mkonline_primary"), name=f"{code} recipe MKOnline weight"
    )
    if (
        abs(recipe_autonomous - dependency.current_weight_autonomous) > 1e-12
        or abs(recipe_mkonline - dependency.current_weight_mkonline) > 1e-12
        or abs((recipe_autonomous + recipe_mkonline) - 1.0) > 1e-12
    ):
        raise TopologyContractError(f"{code}: configured production weights differ from sealed recipe")
    external = _mapping(recipe.get("external_expert"), name=f"{code} external expert")
    if str(external.get("series")) != dependency.primary_series:
        raise TopologyContractError(f"{code}: blend recipe terminal series mismatch")
    if external.get("dependency_manifest_sha256") != dependency.dependency_manifest_sha256:
        raise TopologyContractError(f"{code}: blend recipe dependency hash mismatch")
    if external.get("storm_used_as_feature") is not False:
        raise TopologyContractError(f"{code}: current blend recipe is not Storm-free")
    if external.get("interpolation_allowed") is not False:
        raise TopologyContractError(f"{code}: current blend recipe permits interpolation")
    selection = recipe.get("selection_protocol", {})
    if not isinstance(selection, Mapping):
        raise TopologyContractError(f"{code}: invalid current blend selection protocol")
    final_used = recipe.get(
        "final_target_used_for_weight_or_hyperparameters",
        selection.get("final_target_used_for_weight_or_hyperparameters"),
    )
    if final_used is not False:
        raise TopologyContractError(f"{code}: current blend used final targets for tuning")

    manifest = _load_json(
        dependency.dependency_manifest, name=f"{code} blend dependency"
    )
    if manifest.get("dependency_gate_passed") is not True:
        raise TopologyContractError(f"{code}: blend dependency gate failed")
    if manifest.get("storm_token_found") is not False:
        raise TopologyContractError(f"{code}: blend dependency is not Storm-free")
    if str(manifest.get("terminal_series")) != dependency.primary_series:
        raise TopologyContractError(f"{code}: blend terminal series mismatch")

    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise TopologyContractError("pandas and numpy are required for blend audit") from exc
    raw = pd.read_parquet(dependency.forecast_file)
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise TopologyContractError(f"{code}: blend forecast missing {missing}")
    delivery = pd.DatetimeIndex(pd.to_datetime(raw["value_time_utc"], utc=True, errors="raise"))
    expected = _expected_index(contract, code)
    if not delivery.equals(expected):
        raise TopologyContractError(f"{code}: blend forecast timeline is not exact")
    values = pd.to_numeric(raw["value"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise TopologyContractError(f"{code}: blend forecast contains non-finite values")
    local_days = delivery.tz_convert(ZONE_TIMEZONES[code]).date
    expected_cutoff = pd.DatetimeIndex(
        [
            pd.Timestamp(day - timedelta(days=1), tz="Europe/Paris")
            .replace(hour=8)
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    snapshot = pd.DatetimeIndex(pd.to_datetime(raw["snapshot_time_utc"], utc=True, errors="raise"))
    revision = pd.DatetimeIndex(pd.to_datetime(raw["revision_time_utc"], utc=True, errors="raise"))
    violations = int(np.count_nonzero(snapshot.asi8 != expected_cutoff.asi8) + np.count_nonzero(revision.asi8 != expected_cutoff.asi8))
    if violations:
        raise TopologyContractError(f"{code}: blend cutoff violations")
    return {
        "zone": code,
        "primary_series": dependency.primary_series,
        "recipe_manifest_sha256": dependency.recipe_manifest_sha256,
        "dependency_manifest_sha256": dependency.dependency_manifest_sha256,
        "forecast_file_sha256": dependency.forecast_file_sha256,
        "baseline_model": "mkonline_blend",
        "previous_production_weights": {
            "autonomous": dependency.current_weight_autonomous,
            "mkonline_primary": dependency.current_weight_mkonline,
        },
        "n_days": EXPECTED_TOTAL_DAYS,
        "n_hours": int(len(delivery)),
        "coverage": 1.0,
        "cutoff_violations": violations,
        "final_targets_used_for_weight_or_hyperparameters": False,
    }


__all__ = [
    "BLEND_ZONES",
    "EXPECTED_ADJACENCY",
    "EXPECTED_CORRECTION_SCALE_GRID",
    "EXPECTED_MODEL_VALUES",
    "EXPECTED_MINIMUM_PIT_COVERAGE",
    "EXPECTED_RESIDUAL_LOAD_COLUMNS",
    "FORBIDDEN_TOPOLOGY_INPUT_TOKENS",
    "FORMAL_GATE_HALF_DAYS",
    "GatePolicy",
    "RADIUS_CANDIDATES",
    "SCHEMA_VERSION",
    "SPLIT_DAY_COUNTS",
    "SUPPORTED_VARIANTS",
    "SUPPORTED_ZONES",
    "SealedAutonomousSource",
    "SplitWindow",
    "TOPOLOGY_CONTEXT_COLUMNS",
    "TopologyContractError",
    "TopologyExperimentContract",
    "TopologyModelSpec",
    "BlendDependency",
    "audit_blend_source",
    "audit_topology_sources",
    "canonical_sha256",
    "load_topology_contract",
    "validate_prediction_input_names",
]
