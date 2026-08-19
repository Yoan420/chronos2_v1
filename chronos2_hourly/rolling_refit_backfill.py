"""Offline one-day PIT evidence reconstruction for rolling-refit backfills.

This module is deliberately read-only.  It reconstructs one delivery day at
a time from local checksummed vintage parquets, validates the selected values
against the already archived raw PIT feature columns, and returns conservative
daily timestamp maxima.  A separate writer may package the result into the
strict filesystem contract; this module never publishes or mutates a block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN = ("storm", "mkonline")
_PIT_COLUMNS = (
    "value_time_utc",
    "snapshot_time_utc",
    "revision_time_utc",
    "value",
)


class RollingRefitBackfillError(ValueError):
    """Raised when an offline day cannot be reconstructed exactly."""


@dataclass(frozen=True)
class PitBackfillSource:
    alias: str
    path: str | Path
    sha256: str
    archived_feature_column: str

    def __post_init__(self) -> None:
        alias = str(self.alias).strip()
        feature = str(self.archived_feature_column).strip()
        if not alias or not feature:
            raise RollingRefitBackfillError(
                "PIT alias and archived_feature_column must be explicit"
            )
        if _forbidden([alias, feature, self.path]):
            raise RollingRefitBackfillError(
                "PIT backfill source contains a forbidden forecast token"
            )
        path = Path(self.path).resolve()
        if not path.is_file():
            raise RollingRefitBackfillError(f"PIT parquet is missing: {path}")
        digest = str(self.sha256).strip().lower()
        if _SHA256.fullmatch(digest) is None:
            raise RollingRefitBackfillError("PIT source sha256 is invalid")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "archived_feature_column", feature)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True)
class PitBackfillDayResult:
    selected_values: pd.DataFrame
    timestamp_evidence: pd.DataFrame
    audit: Mapping[str, Any]


def _forbidden(values: Sequence[Any]) -> bool:
    return any(
        token in str(value).casefold()
        for value in values
        for token in _FORBIDDEN
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _civil_day(value: date | str | pd.Timestamp, *, timezone: str) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RollingRefitBackfillError("delivery_day must be a civil date") from exc
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone).tz_localize(None)
    if timestamp != timestamp.normalize():
        raise RollingRefitBackfillError("delivery_day must be a civil date")
    return timestamp.date()


def _read_day(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(
            path,
            columns=list(_PIT_COLUMNS),
            filters=[
                ("value_time_utc", ">=", index[0].to_pydatetime()),
                ("value_time_utc", "<=", index[-1].to_pydatetime()),
            ],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=list(_PIT_COLUMNS))
    missing = sorted(set(_PIT_COLUMNS).difference(frame.columns))
    if missing:
        raise RollingRefitBackfillError(f"PIT parquet columns missing: {missing}")
    for column in _PIT_COLUMNS[:3]:
        values: list[pd.Timestamp] = []
        for raw in frame[column]:
            if pd.isna(raw):
                raise RollingRefitBackfillError(
                    f"PIT {column} contains missing timestamp"
                )
            timestamp = pd.Timestamp(raw)
            if timestamp.tzinfo is None:
                raise RollingRefitBackfillError(
                    f"PIT {column} must carry an explicit timezone"
                )
            values.append(timestamp.tz_convert("UTC"))
        frame[column] = pd.to_datetime(
            pd.Series(values, index=frame.index),
            utc=True,
            errors="raise",
        )
    return frame.loc[
        frame["value_time_utc"].between(index[0], index[-1], inclusive="both")
    ].copy()


def reconstruct_pit_backfill_day(
    sources: Sequence[PitBackfillSource],
    *,
    delivery_day: date | str | pd.Timestamp,
    delivery_timezone: str,
    archived_features: pd.DataFrame,
    serialization_tolerance: float = 5e-6,
    origin_timezone: str = "Europe/Paris",
    origin_hour_local: int = 8,
) -> PitBackfillDayResult:
    """Reconstruct one day with stable double-cutoff PIT selection.

    The returned daily maxima are real maxima over every selected row from
    every declared source.  They are repeated across the canonical 23/24/25
    hour grid so they conservatively cover downstream profiles, ramps and
    daily statistics derived from the full PIT curve.
    """

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise RollingRefitBackfillError("sources must be a non-empty sequence")
    if not sources or not all(isinstance(value, PitBackfillSource) for value in sources):
        raise RollingRefitBackfillError(
            "sources must contain PitBackfillSource values"
        )
    aliases = [value.alias for value in sources]
    features = [value.archived_feature_column for value in sources]
    if len(aliases) != len(set(aliases)) or len(features) != len(set(features)):
        raise RollingRefitBackfillError(
            "PIT aliases and archived feature bindings must be unique"
        )
    if isinstance(serialization_tolerance, bool):
        raise RollingRefitBackfillError("serialization_tolerance must be numeric")
    tolerance = float(serialization_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise RollingRefitBackfillError(
            "serialization_tolerance must be finite and non-negative"
        )
    if isinstance(origin_hour_local, bool) or not isinstance(origin_hour_local, int):
        raise RollingRefitBackfillError("origin_hour_local must be an integer")
    if not 0 <= origin_hour_local <= 23:
        raise RollingRefitBackfillError("origin_hour_local must be between 0 and 23")
    day = _civil_day(delivery_day, timezone=delivery_timezone)
    expected = pd.DatetimeIndex(
        local_delivery_day_index(day, timezone=delivery_timezone).tz_convert("UTC"),
        name="delivery_start_utc",
    )
    if not isinstance(archived_features.index, pd.DatetimeIndex):
        raise RollingRefitBackfillError(
            "archived_features needs a DatetimeIndex"
        )
    if archived_features.index.tz is None:
        raise RollingRefitBackfillError(
            "archived_features index must be timezone-aware"
        )
    archived = archived_features.copy()
    archived.index = pd.DatetimeIndex(
        archived.index.tz_convert("UTC"), name="delivery_start_utc"
    )
    if archived.index.has_duplicates:
        raise RollingRefitBackfillError(
            "archived_features contains duplicate timestamps"
        )
    missing_columns = sorted(set(features).difference(archived.columns))
    if missing_columns:
        raise RollingRefitBackfillError(
            f"archived PIT feature columns missing: {missing_columns}"
        )
    archived = archived.reindex(expected)
    cutoff = (
        pd.Timestamp(day)
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=origin_hour_local)
    ).tz_localize(
        origin_timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")

    value_columns: dict[str, pd.Series] = {}
    selected_frames: list[pd.DataFrame] = []
    source_audit: list[dict[str, Any]] = []
    for source in sources:
        observed_sha = _sha256(Path(source.path))
        if observed_sha != source.sha256:
            raise RollingRefitBackfillError(
                f"{source.alias}: PIT source checksum mismatch"
            )
        frame = _read_day(Path(source.path), expected)
        eligible = frame.loc[
            frame["snapshot_time_utc"].le(cutoff)
            & frame["revision_time_utc"].le(cutoff)
        ].sort_values(
            ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
            kind="stable",
        )
        selected = eligible.drop_duplicates("value_time_utc", keep="last")
        numeric = pd.to_numeric(selected["value"], errors="coerce")
        if bool((numeric.notna() & ~np.isfinite(numeric)).any()):
            raise RollingRefitBackfillError(
                f"{source.alias}: selected PIT value is infinite"
            )
        selected = selected.assign(value=numeric).set_index("value_time_utc")
        selected.index = pd.DatetimeIndex(
            selected.index.tz_convert("UTC"), name="delivery_start_utc"
        )
        selected = selected.reindex(expected)
        raw_archived = archived[source.archived_feature_column]
        archived_value = pd.to_numeric(raw_archived, errors="coerce")
        introduced_missing = archived_value.isna() & ~raw_archived.isna()
        if bool(introduced_missing.any()):
            raise RollingRefitBackfillError(
                f"{source.alias}: archived feature is non-numeric"
            )
        selected_value = pd.to_numeric(selected["value"], errors="coerce")
        if not selected_value.isna().equals(archived_value.isna()):
            raise RollingRefitBackfillError(
                f"{source.alias}: selected/archived missing masks differ"
            )
        present = selected_value.notna()
        if present.any() and not np.allclose(
            selected_value.loc[present].to_numpy(dtype=float),
            archived_value.loc[present].to_numpy(dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            maximum_delta = float(
                np.max(
                    np.abs(
                        selected_value.loc[present].to_numpy(dtype=float)
                        - archived_value.loc[present].to_numpy(dtype=float)
                    )
                )
            )
            raise RollingRefitBackfillError(
                f"{source.alias}: PIT values disagree with archive; "
                f"maximum_delta={maximum_delta}, tolerance={tolerance}"
            )
        selected_frames.append(selected)
        value_columns[source.alias] = selected_value.rename(source.alias)
        source_audit.append(
            {
                "alias": source.alias,
                "path": str(source.path),
                "sha256": observed_sha,
                "archived_feature_column": source.archived_feature_column,
                "dependency_scope": "delivery_day",
                "selected_rows": int(present.sum()),
                "missing_physical_hours": int((~present).sum()),
                "maximum_snapshot_time_utc": (
                    None
                    if selected["snapshot_time_utc"].dropna().empty
                    else str(selected["snapshot_time_utc"].max())
                ),
                "maximum_revision_time_utc": (
                    None
                    if selected["revision_time_utc"].dropna().empty
                    else str(selected["revision_time_utc"].max())
                ),
            }
        )
    snapshots = pd.concat(
        [frame["snapshot_time_utc"] for frame in selected_frames], axis=1
    )
    revisions = pd.concat(
        [frame["revision_time_utc"] for frame in selected_frames], axis=1
    )
    maximum_snapshot = snapshots.max(axis=None, skipna=True)
    maximum_revision = revisions.max(axis=None, skipna=True)
    if pd.isna(maximum_snapshot) or pd.isna(maximum_revision):
        raise RollingRefitBackfillError(
            f"{day}: no real selected PIT timestamp exists"
        )
    maximum_snapshot = pd.Timestamp(maximum_snapshot).tz_convert("UTC")
    maximum_revision = pd.Timestamp(maximum_revision).tz_convert("UTC")
    if maximum_snapshot > cutoff or maximum_revision > cutoff:
        raise RollingRefitBackfillError("selected PIT evidence exceeds its origin")
    selected_values = pd.DataFrame(value_columns, index=expected)
    evidence = pd.DataFrame(
        {
            "maximum_snapshot_time_utc": [maximum_snapshot] * len(expected),
            "maximum_revision_time_utc": [maximum_revision] * len(expected),
            "pit_inputs_present": [True] * len(expected),
        },
        index=expected,
    )
    audit = {
        "status": "complete",
        "delivery_day_local": day.isoformat(),
        "delivery_timezone": delivery_timezone,
        "physical_hours": int(len(expected)),
        "forecast_origin_utc": str(cutoff),
        "origin_contract": (
            f"civil T-1 {origin_hour_local:02d}:00 {origin_timezone}"
        ),
        "causal_timestamp_scope": "delivery_day_all_pit_inputs",
        "serialization_tolerance": tolerance,
        "maximum_snapshot_time_utc": str(maximum_snapshot),
        "maximum_revision_time_utc": str(maximum_revision),
        "sources": source_audit,
        "storm_used_as_feature": False,
        "mkonline_used_as_feature": False,
        "network_access": False,
        "files_written": False,
    }
    return PitBackfillDayResult(
        selected_values=selected_values,
        timestamp_evidence=evidence,
        audit=audit,
    )


__all__ = [
    "PitBackfillDayResult",
    "PitBackfillSource",
    "RollingRefitBackfillError",
    "reconstruct_pit_backfill_day",
]
