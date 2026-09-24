"""Immutable causal capture blocks for the future rolling-365 shadow model.

The existing live archives do not retain enough row-level point-in-time (PIT)
metadata to reconstruct a trustworthy rolling training corpus.  This module
therefore starts the evidence trail prospectively.  One successful issued-live
run can write a target-pending candidate block for delivery day ``D``.  A later
run copies that immutable candidate into a separate final block only after the
realised target for ``D`` is available.

No production forecast is accepted by this API.  Inputs are limited to the raw
Chronos quantiles, the autonomous feature matrix and the selected PIT source
files.  Columns or source aliases containing ``Storm`` or ``MKOnline`` are
rejected before anything is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.observation_precision import validate_observation_precision


LOGGER = logging.getLogger("rolling_capture")
SCHEMA_VERSION = "rolling-refit-block/v1"
QUANTILES = ("q10", "q50", "q90")
FORECAST_ORIGIN_TIMEZONE = "Europe/Paris"
FORBIDDEN_TOKENS = ("storm", "mkonline")
PENDING_DIRNAME = "pending"
FINAL_DIRNAME = "blocks"
CANDIDATE_FILENAME = "rolling_refit_candidate_v1.parquet"
BLOCK_FILENAME = "rolling_refit_training_rows.csv.gz"
MANIFEST_FILENAME = "rolling_refit_block_manifest.json"
CHECKSUM_FILENAME = "artifact_checksums.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DETERMINISTIC_CALENDAR_FEATURES = frozenset(
    {
        "known_hour_sin",
        "known_hour_cos",
        "known_dow_sin",
        "known_dow_cos",
        "known_doy_sin",
        "known_doy_cos",
        "known_is_weekend",
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
    }
)


class RollingCaptureError(ValueError):
    """Raised when a prospective rolling block cannot prove its identity."""


@dataclass(frozen=True)
class RollingCaptureResult:
    """Result of one non-production prospective capture attempt."""

    status: str
    zone: str
    delivery_day: date
    candidate_directory: Path | None
    finalized_previous_directory: Path | None
    audit: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path, date)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _normalise_sha256(value: str, *, name: str) -> str:
    digest = str(value).strip().lower()
    if _SHA256.fullmatch(digest) is None:
        raise RollingCaptureError(f"{name} must be one SHA-256 digest")
    return digest


def _forbidden(values: Sequence[Any]) -> list[str]:
    return [
        str(value)
        for value in values
        if any(token in str(value).casefold() for token in FORBIDDEN_TOKENS)
    ]


def _require_timezone_aware_values(values: Sequence[Any], *, name: str) -> None:
    for value in values:
        if pd.isna(value):
            raise RollingCaptureError(f"{name} contains a missing timestamp")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise RollingCaptureError(f"{name} contains an invalid timestamp") from exc
        if timestamp.tzinfo is None:
            raise RollingCaptureError(
                f"{name} timestamps must carry an explicit timezone"
            )


def _civil_day(value: date | str | pd.Timestamp, *, timezone: str) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise RollingCaptureError("delivery_day must be one civil date")
    return timestamp.date()


def _utc_index(frame: pd.DataFrame, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise RollingCaptureError(f"{name} needs a timezone-aware DatetimeIndex")
    index = pd.DatetimeIndex(frame.index.tz_convert("UTC"), name="delivery_start_utc")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RollingCaptureError(f"{name} index must be unique and sorted")
    return index


def _select_pit_source(
    *,
    alias: str,
    source: Mapping[str, Any],
    delivery_index: pd.DatetimeIndex,
    forecast_origin_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _forbidden([alias]):
        raise RollingCaptureError(f"forbidden PIT source alias: {alias}")
    path_value = source.get("path")
    if path_value in (None, ""):
        raise RollingCaptureError(f"{alias}: PIT source path is missing")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise RollingCaptureError(f"{alias}: PIT source is missing: {path}")
    declared_sha = _normalise_sha256(
        str(source.get("sha256", "")),
        name=f"{alias}.sha256",
    )
    observed_sha = _sha256(path)
    if observed_sha != declared_sha:
        raise RollingCaptureError(f"{alias}: PIT source checksum mismatch")
    feature_column = str(source.get("feature_column", "")).strip()
    if not feature_column or _forbidden([feature_column]):
        raise RollingCaptureError(
            f"{alias}: feature_column must explicitly bind one safe feature"
        )
    raw_tolerance = source.get("serialization_tolerance")
    if isinstance(raw_tolerance, bool) or raw_tolerance is None:
        raise RollingCaptureError(
            f"{alias}: serialization_tolerance must be explicitly declared"
        )
    try:
        tolerance = float(raw_tolerance)
    except (TypeError, ValueError) as exc:
        raise RollingCaptureError(
            f"{alias}: serialization_tolerance must be numeric"
        ) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise RollingCaptureError(
            f"{alias}: serialization_tolerance must be finite and non-negative"
        )
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    try:
        frame = pd.read_parquet(
            path,
            columns=sorted(required),
            filters=[
                (
                    "value_time_utc",
                    ">=",
                    delivery_index[0].to_pydatetime(),
                ),
                (
                    "value_time_utc",
                    "<=",
                    delivery_index[-1].to_pydatetime(),
                ),
            ],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=sorted(required))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RollingCaptureError(f"{alias}: PIT columns missing: {missing}")
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        _require_timezone_aware_values(frame[column].tolist(), name=f"{alias}.{column}")
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    selected = frame.loc[
        frame["value_time_utc"].between(
            delivery_index[0],
            delivery_index[-1],
            inclusive="both",
        )
        & frame["snapshot_time_utc"].le(forecast_origin_utc)
        & frame["revision_time_utc"].le(forecast_origin_utc)
    ].sort_values(
        ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
        kind="stable",
    )
    selected = selected.drop_duplicates("value_time_utc", keep="last")
    selected_index = pd.DatetimeIndex(
        selected["value_time_utc"],
        name="delivery_start_utc",
    )
    if not selected_index.equals(delivery_index):
        raise RollingCaptureError(
            f"{alias}: PIT source does not cover the exact delivery grid"
        )
    values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(float)
    if np.isinf(values).any():
        raise RollingCaptureError(f"{alias}: selected PIT values contain infinity")
    selected = selected.set_index("value_time_utc")
    selected.index = delivery_index
    return selected, {
        "alias": alias,
        "feature_column": feature_column,
        "serialization_tolerance": tolerance,
        "path": str(path),
        "sha256": observed_sha,
        "rows": int(len(selected)),
        "maximum_snapshot_time_utc": str(selected["snapshot_time_utc"].max()),
        "maximum_revision_time_utc": str(selected["revision_time_utc"].max()),
        "cutoff_violations": 0,
    }


def _write_checksum_manifest(
    directory: Path,
    files: Mapping[str, Path],
) -> str:
    entries: list[dict[str, Any]] = []
    for role, path in files.items():
        if not path.is_file() or path.parent != directory:
            raise RollingCaptureError(f"invalid rolling artifact: {path}")
        entries.append(
            {
                "path": path.name,
                "role": str(role),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    checksum_path = directory / CHECKSUM_FILENAME
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "schema_version": SCHEMA_VERSION,
            "artifacts": entries,
        },
    )
    return _sha256(checksum_path)


def _verify_checksum_manifest(directory: Path) -> tuple[dict[str, Any], str]:
    checksum_path = directory / CHECKSUM_FILENAME
    if not checksum_path.is_file():
        raise RollingCaptureError(f"checksum manifest missing: {checksum_path}")
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("algorithm") != "sha256":
        raise RollingCaptureError("invalid rolling checksum manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RollingCaptureError("rolling checksum manifest has no artifacts")
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise RollingCaptureError("invalid rolling checksum entry")
        relative = str(item.get("path", ""))
        if not relative or relative in seen or Path(relative).name != relative:
            raise RollingCaptureError("unsafe or duplicate rolling checksum path")
        seen.add(relative)
        path = directory / relative
        expected = _normalise_sha256(
            str(item.get("sha256", "")),
            name=f"checksum[{relative}]",
        )
        if not path.is_file() or _sha256(path) != expected:
            raise RollingCaptureError(f"rolling artifact checksum mismatch: {relative}")
    return dict(payload), _sha256(checksum_path)


def _verify_issued_live_archive(
    *,
    archive: str | Path,
    forecast_filename: str,
    zone: str,
    delivery_day: date,
    target_series: str,
) -> Mapping[str, Any]:
    """Re-hash a published official archive without exposing it as a feature."""

    directory = Path(archive).expanduser().resolve()
    if not directory.is_dir():
        raise RollingCaptureError(
            f"issued-live archive is not published: {directory}"
        )
    if not forecast_filename or Path(forecast_filename).name != forecast_filename:
        raise RollingCaptureError("issued-live forecast filename is unsafe")
    forecast_path = directory / forecast_filename
    run_manifest_path = directory / "run_manifest.json"
    checksum_path = directory / CHECKSUM_FILENAME
    for path in (forecast_path, run_manifest_path, checksum_path):
        if not path.is_file():
            raise RollingCaptureError(f"issued-live proof is missing: {path}")
    checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    artifacts = checksum_payload.get("artifacts") if isinstance(checksum_payload, Mapping) else None
    if checksum_payload.get("algorithm") != "sha256" or not isinstance(artifacts, list):
        raise RollingCaptureError("issued-live archive checksum manifest is invalid")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        relative = str(item.get("path", ""))
        if item.get("role") == "run_artifact" and relative:
            if relative in indexed:
                raise RollingCaptureError(
                    f"duplicate issued-live checksum entry: {relative}"
                )
            indexed[relative] = item
    observed: dict[str, str] = {}
    for relative, path in (
        (forecast_filename, forecast_path),
        ("run_manifest.json", run_manifest_path),
    ):
        item = indexed.get(relative)
        if item is None:
            raise RollingCaptureError(
                f"issued-live checksum entry is missing: {relative}"
            )
        declared = _normalise_sha256(
            str(item.get("sha256", "")),
            name=f"issued_live[{relative}]",
        )
        digest = _sha256(path)
        if digest != declared:
            raise RollingCaptureError(
                f"issued-live artifact checksum mismatch: {relative}"
            )
        observed[relative] = digest
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(run_manifest, Mapping):
        raise RollingCaptureError("issued-live run manifest must be an object")
    expected_identity = {
        "zone": zone,
        "delivery_day_local": delivery_day.isoformat(),
        "run_type": "live_day_ahead",
        "forecast_status": "issued_live",
        "target_series": target_series,
    }
    for key, expected in expected_identity.items():
        if run_manifest.get(key) != expected:
            raise RollingCaptureError(
                f"issued-live archive identity mismatch for {key}: "
                f"{run_manifest.get(key)!r} != {expected!r}"
            )
    return {
        "issued_live_archive_directory": str(directory),
        "issued_live_forecast_filename": forecast_filename,
        "issued_live_forecast_sha256": observed[forecast_filename],
        "issued_live_run_manifest_sha256": observed["run_manifest.json"],
        "issued_live_archive_checksums_sha256": _sha256(checksum_path),
        "issued_live_observed_at_utc": run_manifest.get("issued_at_utc")
        or run_manifest.get("execution_started_at_utc")
        or run_manifest.get("run_started_as_of_utc"),
    }


def _atomic_directory(parent: Path, final: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"immutable rolling block already exists: {final}")
    return Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=parent))


def prepare_supported_capture_inputs(
    *,
    fresh_features: pd.DataFrame,
    required_pit_aliases: Sequence[str],
    pit_freshness: Mapping[str, Mapping[str, Any]],
    serialization_tolerance: float = 1e-5,
) -> tuple[pd.DataFrame, Mapping[str, Mapping[str, Any]], Mapping[str, str], Mapping[str, Any]]:
    """Select only features whose row-level provenance can be proved today.

    Current residual feature frames also carry historical-price lag columns.
    Those are deliberately excluded because their source cache does not expose
    the per-row PIT timestamp contract required by the rolling loader.  The
    supported subset is an explicit allow-list: raw known-future oracle columns
    bound one-to-one to required PIT aliases, plus calendar columns generated
    deterministically from the delivery index.
    """

    if isinstance(required_pit_aliases, (str, bytes)) or not isinstance(
        required_pit_aliases, Sequence
    ):
        raise RollingCaptureError("required_pit_aliases must be a sequence")
    aliases = [str(alias).strip() for alias in required_pit_aliases]
    if not aliases or any(not alias for alias in aliases) or len(aliases) != len(set(aliases)):
        raise RollingCaptureError("required_pit_aliases must be unique and non-empty")
    if _forbidden(aliases):
        raise RollingCaptureError("required_pit_aliases contain a forbidden source")
    if isinstance(serialization_tolerance, bool):
        raise RollingCaptureError("serialization_tolerance must be numeric")
    tolerance = float(serialization_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise RollingCaptureError(
            "serialization_tolerance must be finite and non-negative"
        )
    selected_columns: list[str] = []
    sources: dict[str, Mapping[str, Any]] = {}
    provenance: dict[str, str] = {}
    for alias in aliases:
        feature_column = f"known_{alias}_oracle"
        if feature_column not in fresh_features:
            raise RollingCaptureError(
                f"{alias}: exact oracle feature is missing: {feature_column}"
            )
        source = pit_freshness.get(alias)
        if not isinstance(source, Mapping):
            raise RollingCaptureError(f"{alias}: row-level PIT audit is missing")
        selected_columns.append(feature_column)
        provenance[feature_column] = "pit_asof"
        sources[alias] = {
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "feature_column": feature_column,
            "serialization_tolerance": tolerance,
        }
    for column in fresh_features.columns:
        name = str(column)
        if name in DETERMINISTIC_CALENDAR_FEATURES and name not in provenance:
            selected_columns.append(name)
            provenance[name] = "deterministic_calendar"
    excluded = [
        str(column)
        for column in fresh_features.columns
        if str(column) not in provenance
    ]
    selected = fresh_features.loc[:, selected_columns].copy()
    return selected, sources, provenance, {
        "selected_feature_count": int(len(selected_columns)),
        "pit_feature_count": int(len(aliases)),
        "deterministic_calendar_feature_count": int(
            len(selected_columns) - len(aliases)
        ),
        "excluded_unproved_features": excluded,
        "historical_price_features_excluded": [
            name for name in excluded if name.startswith("price_")
        ],
        "serialization_tolerance": tolerance,
    }


def _schema_sha256(columns: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(column) for column in columns],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def prove_frozen_builder_subset_equivalence(
    *,
    full_features: pd.DataFrame,
    captured_features: pd.DataFrame,
    chronos_live: pd.DataFrame,
    delivery_timezone: str,
    primary_country: str,
) -> Mapping[str, Any]:
    """Prove the capture subset is identical under the frozen v1 builder."""

    from chronos2_hourly.models.residual_corrector import (
        build_residual_meta_features,
    )

    full_index = _utc_index(full_features, name="full_features")
    captured_index = _utc_index(captured_features, name="captured_features")
    chronos_index = _utc_index(chronos_live, name="chronos_live")
    if not full_index.equals(captured_index) or not full_index.equals(chronos_index):
        raise RollingCaptureError("subset proof inputs are not exactly aligned")
    full = full_features.copy()
    full.index = full_index
    captured = captured_features.copy()
    captured.index = captured_index
    experts = chronos_live.loc[:, list(QUANTILES)].copy().rename(
        columns={name: f"chronos2__{name}" for name in QUANTILES}
    )
    experts.index = chronos_index
    options = {
        "timezone": str(delivery_timezone),
        "include_calendar": True,
        "include_rich_calendar": True,
        "rich_calendar_countries": ("FR", "DE", "BE", "ES", "NL"),
        "rich_calendar_primary_country": str(primary_country).upper(),
        "include_daily_profiles": True,
        "include_fundamental_interactions": False,
        "include_missing_indicators": False,
        "exclude_historical_prices": True,
        "exclude_day_of_year": True,
    }
    full_meta = build_residual_meta_features(full, experts, **options)
    captured_meta = build_residual_meta_features(captured, experts, **options)
    meta_feature_column_order_normalized = False
    if list(full_meta.columns) != list(captured_meta.columns):
        missing = sorted(set(full_meta.columns).difference(captured_meta.columns))
        unexpected = sorted(set(captured_meta.columns).difference(full_meta.columns))
        if missing or unexpected:
            raise RollingCaptureError(
                "captured subset changes frozen residual meta-feature schema: "
                f"missing={missing}, unexpected={unexpected}"
            )
        # The frozen builder can preserve the order of its physical inputs.
        # A capture subset may therefore emit the exact same named features in
        # another order.  Align by the already-proved identical names, then keep
        # the strict value and hash checks below.
        captured_meta = captured_meta.loc[:, list(full_meta.columns)]
        meta_feature_column_order_normalized = True
    left = full_meta.to_numpy(dtype=float)
    right = captured_meta.to_numpy(dtype=float)
    if not np.allclose(
        left,
        right,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        finite = np.isfinite(left) & np.isfinite(right)
        maximum_delta = (
            float(np.max(np.abs(left[finite] - right[finite])))
            if bool(finite.any())
            else None
        )
        raise RollingCaptureError(
            "captured subset changes frozen residual meta-feature values: "
            f"maximum_delta={maximum_delta}"
        )
    full_hash = pd.util.hash_pandas_object(full_meta, index=True).to_numpy(
        dtype=np.uint64
    )
    captured_hash = pd.util.hash_pandas_object(
        captured_meta,
        index=True,
    ).to_numpy(dtype=np.uint64)
    if not np.array_equal(full_hash, captured_hash):
        raise RollingCaptureError("captured subset meta-feature hash differs")
    excluded = [
        str(column)
        for column in full.columns
        if str(column) not in captured.columns
    ]
    return {
        "excluded_by_frozen_builder": True,
        "frozen_builder_recipe": "blend_cat_hgb_w0.50/v1",
        "full_input_feature_schema_sha256": _schema_sha256(full.columns),
        "captured_feature_schema_sha256": _schema_sha256(captured.columns),
        "meta_feature_schema_sha256": _schema_sha256(full_meta.columns),
        "meta_feature_values_sha256": hashlib.sha256(
            full_hash.tobytes()
        ).hexdigest(),
        "full_input_feature_count": int(len(full.columns)),
        "captured_feature_count": int(len(captured.columns)),
        "meta_feature_count": int(len(full_meta.columns)),
        "excluded_features": excluded,
        "meta_feature_column_order_normalized": (
            meta_feature_column_order_normalized
        ),
        "maximum_meta_feature_difference": 0.0,
    }


def write_target_pending_candidate(
    *,
    capture_root: str | Path,
    zone: str,
    delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    full_features_for_equivalence: pd.DataFrame,
    fresh_features: pd.DataFrame,
    chronos_live: pd.DataFrame,
    pit_sources: Mapping[str, Mapping[str, Any]],
    feature_provenance: Mapping[str, str],
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
    target_series: str,
    target_source_path: str | Path,
    issued_live_archive: str | Path,
    issued_live_forecast_filename: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Write one immutable, target-pending issued-live training candidate."""

    zone_code = str(zone).strip().upper()
    if not zone_code or _forbidden([zone_code]):
        raise RollingCaptureError("zone must be explicit and non-forbidden")
    if not str(delivery_timezone).strip():
        raise RollingCaptureError("delivery_timezone must be explicit")
    day = _civil_day(delivery_day, timezone=delivery_timezone)
    expected_index = local_delivery_day_index(day, timezone=delivery_timezone)
    feature_index = _utc_index(fresh_features, name="fresh_features")
    if not feature_index.equals(expected_index):
        raise RollingCaptureError("fresh_features do not cover exact delivery day")
    feature_columns = [str(column) for column in fresh_features.columns]
    if len(feature_columns) != len(set(feature_columns)):
        raise RollingCaptureError("fresh_features contain duplicate columns")
    forbidden_features = _forbidden(feature_columns)
    if forbidden_features:
        raise RollingCaptureError(
            f"forbidden forecast features: {forbidden_features}"
        )
    if not isinstance(feature_provenance, Mapping):
        raise RollingCaptureError("feature_provenance must be an explicit mapping")
    provenance = {str(key): str(value) for key, value in feature_provenance.items()}
    if set(provenance) != set(feature_columns):
        missing = sorted(set(feature_columns).difference(provenance))
        extra = sorted(set(provenance).difference(feature_columns))
        raise RollingCaptureError(
            "feature_provenance must cover the exact feature schema: "
            f"missing={missing}, extra={extra}"
        )
    invalid_provenance = {
        key: value
        for key, value in provenance.items()
        if value not in {"pit_asof", "deterministic_calendar"}
    }
    if invalid_provenance:
        raise RollingCaptureError(
            f"unsupported feature provenance: {invalid_provenance}"
        )
    features = fresh_features.copy()
    features.index = feature_index
    numeric = features.apply(pd.to_numeric, errors="coerce")
    introduced = numeric.isna() & ~features.isna()
    if bool(introduced.to_numpy().any()) or bool(
        np.isinf(numeric.to_numpy(dtype=float)).any()
    ):
        raise RollingCaptureError("fresh_features are not finite numeric/NaN")

    chronos_index = _utc_index(chronos_live, name="chronos_live")
    if not chronos_index.equals(expected_index):
        raise RollingCaptureError("chronos_live does not cover exact delivery day")
    if _forbidden(list(chronos_live.columns)):
        raise RollingCaptureError("chronos_live contains forbidden forecast columns")
    missing_quantiles = sorted(set(QUANTILES).difference(chronos_live.columns))
    if missing_quantiles:
        raise RollingCaptureError(f"raw Chronos quantiles missing: {missing_quantiles}")
    quantiles = chronos_live.loc[:, list(QUANTILES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    quantiles.index = chronos_index
    quantile_values = quantiles.to_numpy(dtype=float)
    if not np.isfinite(quantile_values).all() or not bool(
        (
            (quantile_values[:, 0] <= quantile_values[:, 1])
            & (quantile_values[:, 1] <= quantile_values[:, 2])
        ).all()
    ):
        raise RollingCaptureError("raw Chronos quantiles are invalid or crossed")
    if "forecast_origin_utc" not in chronos_live:
        raise RollingCaptureError("raw Chronos origin is missing")
    _require_timezone_aware_values(
        chronos_live["forecast_origin_utc"].tolist(),
        name="chronos_live.forecast_origin_utc",
    )
    origins = pd.to_datetime(
        chronos_live["forecast_origin_utc"],
        utc=True,
        errors="raise",
    )
    origins = pd.Series(origins.to_numpy(), index=chronos_index)
    if origins.nunique() != 1:
        raise RollingCaptureError("raw Chronos origin is not constant for the day")
    forecast_origin = pd.Timestamp(origins.iloc[0]).tz_convert("UTC")
    expected_origin = (
        pd.Timestamp(day)
        .tz_localize(FORECAST_ORIGIN_TIMEZONE)
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=8)
    ).tz_convert("UTC")
    if forecast_origin != expected_origin:
        raise RollingCaptureError(
            "raw Chronos origin must equal delivery D-1 08:00 Europe/Paris"
        )
    if not bool((pd.DatetimeIndex(origins) < expected_index).all()):
        raise RollingCaptureError("raw Chronos origin is not strictly causal")

    if not isinstance(pit_sources, Mapping) or not pit_sources:
        raise RollingCaptureError("at least one checksummed PIT source is required")
    selected_sources: list[pd.DataFrame] = []
    source_audit: list[dict[str, Any]] = []
    bound_features: set[str] = set()
    for alias in sorted(pit_sources):
        selected, audit = _select_pit_source(
            alias=str(alias),
            source=pit_sources[alias],
            delivery_index=expected_index,
            forecast_origin_utc=forecast_origin,
        )
        feature_column = str(audit["feature_column"])
        if feature_column not in numeric:
            raise RollingCaptureError(
                f"{alias}: bound feature is absent: {feature_column}"
            )
        if feature_column in bound_features:
            raise RollingCaptureError(
                f"multiple PIT sources bind the same feature: {feature_column}"
            )
        if provenance[feature_column] != "pit_asof":
            raise RollingCaptureError(
                f"{alias}: bound feature must have pit_asof provenance"
            )
        feature_values = numeric[feature_column].to_numpy(dtype=float)
        source_values = pd.to_numeric(selected["value"], errors="coerce").to_numpy(
            dtype=float
        )
        tolerance = float(audit["serialization_tolerance"])
        if not np.allclose(
            feature_values,
            source_values,
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        ):
            maximum_delta = float(np.nanmax(np.abs(feature_values - source_values)))
            raise RollingCaptureError(
                f"{alias}: PIT values do not match bound feature {feature_column}; "
                f"maximum_delta={maximum_delta}, tolerance={tolerance}"
            )
        bound_features.add(feature_column)
        selected_sources.append(selected)
        source_audit.append(audit)
    declared_pit_features = {
        name for name, kind in provenance.items() if kind == "pit_asof"
    }
    if bound_features != declared_pit_features:
        missing_bindings = sorted(declared_pit_features.difference(bound_features))
        unexpected_bindings = sorted(bound_features.difference(declared_pit_features))
        raise RollingCaptureError(
            "every pit_asof feature must bind one checksummed PIT source: "
            f"missing={missing_bindings}, unexpected={unexpected_bindings}"
        )
    # The residual feature builder can derive delivery-day profiles, ramps and
    # daily statistics from the complete PIT curve.  Conservatively attach the
    # latest *real selected* timestamp across every contributing row and alias
    # to every training row.  This is stricter than a point-only timestamp and
    # proves the lineage of daily-broadcast derived features without inventing
    # a feature dependency graph.
    day_maximum_snapshot = max(
        pd.Timestamp(frame["snapshot_time_utc"].max())
        for frame in selected_sources
    )
    day_maximum_revision = max(
        pd.Timestamp(frame["revision_time_utc"].max())
        for frame in selected_sources
    )
    maximum_snapshot = pd.Series(
        day_maximum_snapshot,
        index=expected_index,
        name="maximum_snapshot_time_utc",
    )
    maximum_revision = pd.Series(
        day_maximum_revision,
        index=expected_index,
        name="maximum_revision_time_utc",
    )
    if bool((maximum_snapshot > forecast_origin).any()) or bool(
        (maximum_revision > forecast_origin).any()
    ):
        raise RollingCaptureError("selected PIT timestamp exceeds forecast origin")

    reserved = {
        "delivery_start_utc",
        *QUANTILES,
        "actual",
        "forecast_origin_utc",
        "maximum_snapshot_time_utc",
        "maximum_revision_time_utc",
        "pit_inputs_present",
    }
    collisions = sorted(reserved.intersection(feature_columns))
    if collisions:
        raise RollingCaptureError(f"fresh feature names collide with reserved fields: {collisions}")
    frame = numeric.copy()
    frame.insert(0, "delivery_start_utc", expected_index)
    for position, name in enumerate(QUANTILES, start=1):
        frame.insert(position, name, quantiles[name].to_numpy(dtype=float))
    frame.insert(4, "forecast_origin_utc", origins.to_numpy())
    frame.insert(5, "maximum_snapshot_time_utc", maximum_snapshot.to_numpy())
    frame.insert(6, "maximum_revision_time_utc", maximum_revision.to_numpy())
    frame.insert(7, "pit_inputs_present", True)

    config_sha = _normalise_sha256(
        expected_config_sha256,
        name="expected_config_sha256",
    )
    bundle_sha = _normalise_sha256(
        expected_base_bundle_sha256,
        name="expected_base_bundle_sha256",
    )
    issued_live_proof = _verify_issued_live_archive(
        archive=issued_live_archive,
        forecast_filename=issued_live_forecast_filename,
        zone=zone_code,
        delivery_day=day,
        target_series=target_series,
    )
    subset_proof = prove_frozen_builder_subset_equivalence(
        full_features=full_features_for_equivalence,
        captured_features=numeric,
        chronos_live=chronos_live,
        delivery_timezone=delivery_timezone,
        primary_country=zone_code,
    )
    if subset_proof.get("captured_feature_schema_sha256") != _schema_sha256(
        feature_columns
    ):
        raise RollingCaptureError("captured feature schema proof is inconsistent")
    root = Path(capture_root).expanduser().resolve() / zone_code.lower()
    pending_root = root / PENDING_DIRNAME
    final = pending_root / day.isoformat()
    staging = _atomic_directory(pending_root, final)
    try:
        parquet_path = staging / CANDIDATE_FILENAME
        frame.to_parquet(parquet_path, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "target_pending",
            "source_kind": "issued_live",
            "source_id": f"issued-live-{zone_code.lower()}-{day.isoformat()}",
            "run_type": "live_day_ahead",
            "forecast_status": "issued_live",
            "zone": zone_code,
            "target_series": str(target_series),
            "timezone": delivery_timezone,
            "delivery_start_day_local": day.isoformat(),
            "delivery_end_day_local": day.isoformat(),
            "hours": int(len(frame)),
            "n_rows": int(len(frame)),
            "prediction_mode": "autonomous_only",
            "causal_timestamp_scope": "delivery_day_all_pit_inputs",
            "feature_columns": feature_columns,
            "feature_provenance": provenance,
            "feature_subset_proof": subset_proof,
            "raw_chronos_columns": list(QUANTILES),
            "raw_chronos_only": True,
            "chronos_artifact_kind": "raw_pre_residual_quantiles",
            "forecast_origin_utc": str(forecast_origin),
            "config_sha256": config_sha,
            "base_bundle_sha256": bundle_sha,
            "pit_sources": source_audit,
            "maximum_snapshot_time_utc": str(maximum_snapshot.max()),
            "maximum_revision_time_utc": str(maximum_revision.max()),
            "target_available": False,
            "storm_used_as_feature": False,
            "mkonline_used_as_feature": False,
            "production_forecast_used_as_feature": False,
            "timestamp_evidence": {
                "granularity": "per_delivery_hour",
                "forecast_origin_column": "forecast_origin_utc",
                "maximum_snapshot_time_column": "maximum_snapshot_time_utc",
                "maximum_revision_time_column": "maximum_revision_time_utc",
                "pit_inputs_present_column": "pit_inputs_present",
                "source": "materialized_pit_selection",
            },
            "bootstrap_replay": False,
            "rows_role": "rolling_refit_training_rows",
            "candidate_filename": CANDIDATE_FILENAME,
            **issued_live_proof,
        }
        manifest_path = staging / MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        checksum_sha = _write_checksum_manifest(
            staging,
            {
                "rolling_refit_target_pending_rows": parquet_path,
                "rolling_refit_block_manifest": manifest_path,
            },
        )
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final, {
        "status": "target_pending",
        "directory": str(final),
        "artifact_checksums_sha256": checksum_sha,
        "hours": int(len(frame)),
        "pit_source_count": int(len(source_audit)),
        "storm_used_as_feature": False,
        "mkonline_used_as_feature": False,
    }


def finalize_target_pending_candidate(
    *,
    capture_root: str | Path,
    zone: str,
    delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    canonical_target: pd.Series,
    target_series: str,
    target_source_path: str | Path,
    target_observation_archive: str | Path,
    target_observation_forecast_filename: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Copy one verified pending candidate into an immutable realised block."""

    zone_code = str(zone).strip().upper()
    day = _civil_day(delivery_day, timezone=delivery_timezone)
    root = Path(capture_root).expanduser().resolve() / zone_code.lower()
    pending = root / PENDING_DIRNAME / day.isoformat()
    if not pending.is_dir():
        raise FileNotFoundError(f"pending rolling candidate is missing: {pending}")
    _checksum_payload, pending_checksum_sha = _verify_checksum_manifest(pending)
    manifest_path = pending / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RollingCaptureError("pending rolling manifest must be an object")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "status": "target_pending",
        "source_kind": "issued_live",
        "zone": zone_code,
        "timezone": delivery_timezone,
        "delivery_start_day_local": day.isoformat(),
        "delivery_end_day_local": day.isoformat(),
        "target_available": False,
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise RollingCaptureError(
                f"pending rolling identity mismatch for {key}: {manifest.get(key)!r}"
            )
    if manifest.get("target_series") != str(target_series):
        raise RollingCaptureError(
            "pending rolling target_series differs from finalization target"
        )
    observation_proof = _verify_issued_live_archive(
        archive=target_observation_archive,
        forecast_filename=target_observation_forecast_filename,
        zone=zone_code,
        delivery_day=(pd.Timestamp(day) + pd.Timedelta(days=1)).date(),
        target_series=str(target_series),
    )
    observation_manifest = json.loads(
        (
            Path(target_observation_archive).expanduser().resolve()
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    target_diagnostics = observation_manifest.get("input_diagnostics", {}).get(
        "target", {}
    )
    if not isinstance(target_diagnostics, Mapping):
        raise RollingCaptureError(
            "target observation archive lacks input_diagnostics.target"
        )
    declared_target_path = target_diagnostics.get("cache") or target_diagnostics.get(
        "input"
    )
    source_path = Path(target_source_path).expanduser().resolve()
    if declared_target_path in (None, "") or Path(str(declared_target_path)).expanduser().resolve() != source_path:
        raise RollingCaptureError(
            "target source path is not anchored by the D+1 issued-live manifest"
        )
    if not source_path.is_file():
        raise RollingCaptureError(f"target source is missing: {source_path}")
    if observation_manifest.get("target_source_path") != str(source_path):
        raise RollingCaptureError(
            "target source path is not sealed at top level of the D+1 run manifest"
        )
    declared_target_sha = _normalise_sha256(
        observation_manifest.get("target_source_sha256"),
        name="target_source_sha256",
    )
    observed_target_sha = _sha256(source_path)
    if observed_target_sha != declared_target_sha:
        raise RollingCaptureError(
            "target source checksum differs from the D+1 run manifest anchor"
        )
    if source_path.suffix.lower() in {".parquet", ".pq"}:
        target_frame = pd.read_parquet(source_path)
    else:
        target_frame = pd.read_csv(source_path)
    if not {"timestamp", "value"}.issubset(target_frame.columns):
        raise RollingCaptureError(
            "target proof must use standardized timestamp/value cache columns"
        )
    _require_timezone_aware_values(
        target_frame["timestamp"].tolist(),
        name="target_source.timestamp",
    )
    target_source_index = pd.DatetimeIndex(
        pd.to_datetime(target_frame["timestamp"], utc=True, errors="raise")
    )
    if target_source_index.has_duplicates:
        raise RollingCaptureError("target source contains duplicate timestamps")
    target_source_values = pd.Series(
        pd.to_numeric(target_frame["value"], errors="coerce").to_numpy(float),
        index=target_source_index,
    )
    candidate_path = pending / CANDIDATE_FILENAME
    candidate = pd.read_parquet(candidate_path)
    if "actual" in candidate:
        raise RollingCaptureError("pending candidate already contains an actual")
    if "delivery_start_utc" not in candidate:
        raise RollingCaptureError("pending candidate delivery index is missing")
    index = pd.DatetimeIndex(
        pd.to_datetime(candidate["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected_index = local_delivery_day_index(day, timezone=delivery_timezone)
    if not index.equals(expected_index):
        raise RollingCaptureError("pending candidate has an invalid delivery grid")
    if not isinstance(canonical_target.index, pd.DatetimeIndex) or canonical_target.index.tz is None:
        raise RollingCaptureError("canonical_target needs a timezone-aware index")
    target = canonical_target.copy()
    target.index = target.index.tz_convert("UTC")
    if target.index.has_duplicates:
        raise RollingCaptureError("canonical_target contains duplicate timestamps")
    values = pd.to_numeric(target.reindex(index), errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RollingCaptureError("realised target is unavailable or non-finite")
    source_values = target_source_values.reindex(index).to_numpy(dtype=float)
    # Data preparation retains model targets as float32, while the sealed
    # source cache retains float64 observations.  Prove that representation
    # change explicitly; a general absolute tolerance can reject legitimate
    # rounding while accepting a smaller but real price revision.
    try:
        target_precision = validate_observation_precision(
            values, source_values, name="rolling canonical target / sealed source"
        )
    except ValueError as exc:
        raise RollingCaptureError(
            f"canonical_target differs from the checksummed target source: {exc}"
        ) from exc
    candidate.insert(1, "actual", values)

    final_root = root / FINAL_DIRNAME
    final = final_root / day.isoformat()
    if final.exists():
        _verify_checksum_manifest(final)
        return final, {
            "status": "already_finalized",
            "directory": str(final),
            "pending_artifact_checksums_sha256": pending_checksum_sha,
        }
    staging = _atomic_directory(final_root, final)
    try:
        block_path = staging / BLOCK_FILENAME
        candidate.to_csv(
            block_path,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        final_manifest = {
            **dict(manifest),
            "status": "sealed",
            "target_available": True,
            "rows_filename": BLOCK_FILENAME,
            "candidate_filename": None,
            "pending_directory": str(pending),
            "pending_artifact_checksums_sha256": pending_checksum_sha,
            "pending_candidate_sha256": _sha256(candidate_path),
            "actual_rows": int(len(values)),
            "target_source_path": str(source_path),
            "target_source_sha256": observed_target_sha,
            "target_observation_precision": target_precision,
            "target_series": str(target_series),
            "target_observed_at_utc": observation_proof.get(
                "issued_live_observed_at_utc"
            ),
            "target_observation_archive": observation_proof.get(
                "issued_live_archive_directory"
            ),
            "target_observation_archive_checksums_sha256": observation_proof.get(
                "issued_live_archive_checksums_sha256"
            ),
        }
        final_manifest_path = staging / MANIFEST_FILENAME
        _write_json(final_manifest_path, final_manifest)
        final_checksum_sha = _write_checksum_manifest(
            staging,
            {
                "rolling_refit_training_rows": block_path,
                "rolling_refit_block_manifest": final_manifest_path,
            },
        )
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final, {
        "status": "complete",
        "directory": str(final),
        "artifact_checksums_sha256": final_checksum_sha,
        "pending_artifact_checksums_sha256": pending_checksum_sha,
        "hours": int(len(values)),
        "target_observation_precision": target_precision,
    }


def capture_issued_live_block_isolated(
    *,
    capture_root: str | Path,
    zone: str,
    delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    full_features_for_equivalence: pd.DataFrame,
    fresh_features: pd.DataFrame,
    chronos_live: pd.DataFrame,
    pit_sources: Mapping[str, Mapping[str, Any]],
    feature_provenance: Mapping[str, str],
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
    target_series: str,
    target_source_path: str | Path,
    issued_live_archive: str | Path,
    issued_live_forecast_filename: str,
    canonical_target: pd.Series,
) -> RollingCaptureResult:
    """Finalize ``D-1`` and capture ``D`` without propagating ordinary errors.

    The caller must invoke this only *after* the official archive has been
    published.  Every exception is converted to an audit result so a shadow
    instrumentation failure cannot invalidate that official forecast.
    """

    zone_code = str(zone).strip().upper()
    try:
        day = _civil_day(delivery_day, timezone=delivery_timezone)
    except Exception as exc:
        return RollingCaptureResult(
            status="failed",
            zone=zone_code,
            delivery_day=pd.Timestamp(delivery_day).date(),
            candidate_directory=None,
            finalized_previous_directory=None,
            audit={"error_type": type(exc).__name__, "error": str(exc)},
        )
    previous = day - pd.Timedelta(days=1)
    previous_directory: Path | None = None
    previous_audit: Mapping[str, Any]
    try:
        previous_directory, previous_audit = finalize_target_pending_candidate(
            capture_root=capture_root,
            zone=zone_code,
            delivery_day=previous,
            delivery_timezone=delivery_timezone,
            canonical_target=canonical_target,
            target_series=target_series,
            target_source_path=target_source_path,
            target_observation_archive=issued_live_archive,
            target_observation_forecast_filename=issued_live_forecast_filename,
        )
    except FileNotFoundError as exc:
        previous_audit = {
            "status": "not_available",
            "delivery_day_local": str(previous),
            "reason": str(exc),
        }
    except Exception as exc:
        previous_audit = {
            "status": "failed",
            "delivery_day_local": str(previous),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        LOGGER.warning("%s: rolling D-1 finalization failed: %s", zone_code, exc)
    try:
        candidate_directory, candidate_audit = write_target_pending_candidate(
            capture_root=capture_root,
            zone=zone_code,
            delivery_day=day,
            delivery_timezone=delivery_timezone,
            full_features_for_equivalence=full_features_for_equivalence,
            fresh_features=fresh_features,
            chronos_live=chronos_live,
            pit_sources=pit_sources,
            feature_provenance=feature_provenance,
            expected_config_sha256=expected_config_sha256,
            expected_base_bundle_sha256=expected_base_bundle_sha256,
            target_series=target_series,
            target_source_path=target_source_path,
            issued_live_archive=issued_live_archive,
            issued_live_forecast_filename=issued_live_forecast_filename,
        )
    except Exception as exc:
        LOGGER.warning("%s: rolling issued-live capture failed: %s", zone_code, exc)
        return RollingCaptureResult(
            status="failed",
            zone=zone_code,
            delivery_day=day,
            candidate_directory=None,
            finalized_previous_directory=previous_directory,
            audit={
                "candidate": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                "previous_day_finalization": previous_audit,
                "official_forecast_status": "unmodified_already_published",
                "storm_used_as_feature": False,
                "mkonline_used_as_feature": False,
            },
        )
    return RollingCaptureResult(
        status="target_pending",
        zone=zone_code,
        delivery_day=day,
        candidate_directory=candidate_directory,
        finalized_previous_directory=previous_directory,
        audit={
            "candidate": candidate_audit,
            "previous_day_finalization": previous_audit,
            "official_forecast_status": "unmodified_already_published",
            "storm_used_as_feature": False,
            "mkonline_used_as_feature": False,
        },
    )


def capture_supported_issued_live_block_isolated(
    *,
    capture_root: str | Path,
    zone: str,
    delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    fresh_features: pd.DataFrame,
    chronos_live: pd.DataFrame,
    required_pit_aliases: Sequence[str],
    pit_freshness: Mapping[str, Mapping[str, Any]],
    expected_config_sha256: str,
    expected_base_bundle_sha256: str,
    target_series: str,
    target_source_path: str | Path,
    issued_live_archive: str | Path,
    issued_live_forecast_filename: str,
    canonical_target: pd.Series,
    serialization_tolerance: float = 1e-5,
) -> RollingCaptureResult:
    """Strict allow-list adapter used by optional live-runner instrumentation."""

    zone_code = str(zone).strip().upper()
    day = pd.Timestamp(delivery_day).date()
    try:
        selected, sources, provenance, preparation_audit = (
            prepare_supported_capture_inputs(
                fresh_features=fresh_features,
                required_pit_aliases=required_pit_aliases,
                pit_freshness=pit_freshness,
                serialization_tolerance=serialization_tolerance,
            )
        )
    except Exception as exc:
        LOGGER.warning("%s: rolling capture preparation failed: %s", zone_code, exc)
        return RollingCaptureResult(
            status="failed",
            zone=zone_code,
            delivery_day=day,
            candidate_directory=None,
            finalized_previous_directory=None,
            audit={
                "preparation": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                "official_forecast_status": "unmodified_already_published",
                "storm_used_as_feature": False,
                "mkonline_used_as_feature": False,
            },
        )
    result = capture_issued_live_block_isolated(
        capture_root=capture_root,
        zone=zone_code,
        delivery_day=day,
        delivery_timezone=delivery_timezone,
        full_features_for_equivalence=fresh_features,
        fresh_features=selected,
        chronos_live=chronos_live,
        pit_sources=sources,
        feature_provenance=provenance,
        expected_config_sha256=expected_config_sha256,
        expected_base_bundle_sha256=expected_base_bundle_sha256,
        target_series=target_series,
        target_source_path=target_source_path,
        issued_live_archive=issued_live_archive,
        issued_live_forecast_filename=issued_live_forecast_filename,
        canonical_target=canonical_target,
    )
    return RollingCaptureResult(
        status=result.status,
        zone=result.zone,
        delivery_day=result.delivery_day,
        candidate_directory=result.candidate_directory,
        finalized_previous_directory=result.finalized_previous_directory,
        audit={"preparation": preparation_audit, **dict(result.audit)},
    )


__all__ = [
    "BLOCK_FILENAME",
    "CANDIDATE_FILENAME",
    "CHECKSUM_FILENAME",
    "DETERMINISTIC_CALENDAR_FEATURES",
    "FORECAST_ORIGIN_TIMEZONE",
    "FINAL_DIRNAME",
    "MANIFEST_FILENAME",
    "PENDING_DIRNAME",
    "QUANTILES",
    "RollingCaptureError",
    "RollingCaptureResult",
    "SCHEMA_VERSION",
    "capture_issued_live_block_isolated",
    "capture_supported_issued_live_block_isolated",
    "finalize_target_pending_candidate",
    "prepare_supported_capture_inputs",
    "prove_frozen_builder_subset_equivalence",
    "write_target_pending_candidate",
]
