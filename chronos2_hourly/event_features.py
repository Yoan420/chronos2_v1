"""Point-in-time event features for day-ahead electricity forecasting.

The first supported source is the official Météo-France archived Vigilance
``CDP_CARTE_EXTERNE`` JSON product.  The archive keeps dated snapshots and the
payload exposes diffusion/generation timestamps, making a D-1 08:00 cutoff
auditable without using a latest-only endpoint.

This module intentionally does not download data and does not promote any
feature into a model.  It parses previously fetched official snapshots and
materializes a bounded, numeric feature table.  Rendered PNG/PDF maps and free
text bulletins are never interpreted here: their structured risk information
is already present in the JSON product.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


METEOFRANCE_VIGILANCE_ARCHIVE_BASE = (
    "https://files.data.gouv.fr/meteofrance/data/vigilance/metropole"
)
METEOFRANCE_VIGILANCE_DATASET_URL = (
    "https://www.data.gouv.fr/datasets/vigilance-meteorologique-archivee"
)

HAZARD_NAMES: dict[str, str] = {
    "1": "wind",
    "2": "rain",
    "3": "thunderstorm",
    "4": "flood",
    "5": "snow_ice",
    "6": "heat",
    "7": "cold",
    "8": "avalanche",
    "9": "coastal_wave",
}

# These structured hazards have a plausible direct relationship with demand,
# renewable generation, grid stress or plant availability.  Avalanche and
# coastal-wave events remain in the audit records but are not expanded into
# default model columns without evidence of incremental value.
DEFAULT_FEATURE_HAZARDS: tuple[str, ...] = (
    "wind",
    "rain",
    "thunderstorm",
    "flood",
    "snow_ice",
    "heat",
    "cold",
)

_ARCHIVE_PATH_RE = re.compile(
    r"^/meteofrance/data/vigilance/metropole/"
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/"
    r"(?P<hms>\d{6})/(?:[^/]*_)?CDP_CARTE_EXTERNE\.json$",
    flags=re.IGNORECASE,
)
_DEPARTMENT_RE = re.compile(r"^(?:0[1-9]|[1-8][0-9]|9[0-5]|2A|2B)$")
_MAX_METADATA_SKEW = pd.Timedelta(minutes=30)


class EventFeatureContractError(ValueError):
    """Raised when event data cannot be used without causal ambiguity."""


@dataclass(frozen=True)
class EventRecord:
    """Provider-agnostic, revision-aware event record."""

    source: str
    source_event_id: str
    zone: str
    spatial_id: str
    category: str
    severity: int
    temporal_precision: Literal["interval", "period_max"]
    event_start_utc: pd.Timestamp
    event_end_utc: pd.Timestamp
    published_at_utc: pd.Timestamp
    revision_at_utc: pd.Timestamp
    available_at_utc: pd.Timestamp
    snapshot_id: str
    source_uri: str
    content_sha256: str


@dataclass(frozen=True)
class VigilanceSnapshot:
    """One full Météo-France Vigilance map revision."""

    snapshot_id: str
    published_at_utc: pd.Timestamp
    revision_at_utc: pd.Timestamp
    transmission_at_utc: pd.Timestamp
    available_at_utc: pd.Timestamp
    source_uri: str
    content_sha256: str
    valid_periods_utc: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    department_ids: tuple[str, ...]
    events: tuple[EventRecord, ...]


@dataclass(frozen=True)
class EventFeatureMaterialization:
    """Causal hourly features plus their selected raw-event audit trail."""

    features: pd.DataFrame
    events: pd.DataFrame
    provenance: dict[str, Any]


def day_ahead_cutoff_utc(
    delivery_day: str | date,
    *,
    timezone: str = "Europe/Paris",
    local_cutoff: time = time(8, 0),
) -> pd.Timestamp:
    """Return D-1 08:00 local in UTC, including DST transitions."""

    parsed_day = (
        delivery_day
        if isinstance(delivery_day, date)
        else date.fromisoformat(str(delivery_day))
    )
    local = datetime.combine(
        parsed_day - timedelta(days=1), local_cutoff, tzinfo=ZoneInfo(timezone)
    )
    return pd.Timestamp(local).tz_convert("UTC")


def _utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EventFeatureContractError(f"Invalid timestamp for {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EventFeatureContractError(f"{field} must be timezone-aware.")
    return parsed.tz_convert("UTC")


def _official_transmission_time(source_uri: str) -> pd.Timestamp:
    parsed = urlparse(source_uri)
    if parsed.scheme != "https" or parsed.netloc.lower() != "files.data.gouv.fr":
        raise EventFeatureContractError(
            "source_uri must reference the official files.data.gouv.fr Vigilance archive."
        )
    match = _ARCHIVE_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise EventFeatureContractError(
            "source_uri must identify a dated CDP_CARTE_EXTERNE.json snapshot."
        )
    parts = match.groupdict()
    hms = parts["hms"]
    return pd.Timestamp(
        f"{parts['year']}-{parts['month']}-{parts['day']}T"
        f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}Z"
    )


def _decode_payload(raw: bytes | str | Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if isinstance(raw, bytes):
        raw_bytes = raw
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventFeatureContractError("Vigilance payload is not valid UTF-8 JSON.") from exc
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EventFeatureContractError("Vigilance payload is not valid JSON.") from exc
    elif isinstance(raw, Mapping):
        payload = dict(raw)
        raw_bytes = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    else:
        raise EventFeatureContractError("Vigilance payload must be bytes, text or a mapping.")
    if not isinstance(payload, dict):
        raise EventFeatureContractError("Vigilance JSON root must be an object.")
    return payload, raw_bytes


def _severity(color_id: Any, *, field: str) -> int:
    try:
        value = int(color_id)
    except (TypeError, ValueError) as exc:
        raise EventFeatureContractError(f"{field} must be an integer from 1 to 4.") from exc
    if value not in (1, 2, 3, 4):
        raise EventFeatureContractError(f"{field} must be between 1 (green) and 4 (red).")
    return value - 1


def _is_metropolitan_department(value: str) -> bool:
    return _DEPARTMENT_RE.fullmatch(value.upper()) is not None


def _event_id(
    snapshot_id: str,
    spatial_id: str,
    category: str,
    precision: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    return ":".join(
        (
            snapshot_id,
            spatial_id,
            category,
            precision,
            start.isoformat(),
            end.isoformat(),
        )
    )


def parse_meteofrance_vigilance_snapshot(
    raw: bytes | str | Mapping[str, Any],
    *,
    source_uri: str,
) -> VigilanceSnapshot:
    """Parse and validate one official archived Vigilance map snapshot.

    ``available_at_utc`` is the conservative maximum of the payload diffusion,
    generation and archive-transmission timestamps.  This is the timestamp used
    by the point-in-time selector.
    """

    payload, raw_bytes = _decode_payload(raw)
    product = payload.get("product")
    if not isinstance(product, dict):
        raise EventFeatureContractError("Vigilance payload is missing object 'product'.")
    if str(product.get("warning_type", "")).lower() != "vigilance":
        raise EventFeatureContractError("Unexpected Météo-France warning_type.")
    if str(product.get("type_cdp", "")).lower() != "cdp_carte_externe":
        raise EventFeatureContractError("Only cdp_carte_externe snapshots are supported.")
    if str(product.get("domain_id", "")).upper() != "FRA":
        raise EventFeatureContractError("Only the France-metropole FRA product is supported.")

    meta = payload.get("meta", product.get("meta"))
    if not isinstance(meta, dict):
        raise EventFeatureContractError("Vigilance payload is missing object 'meta'.")
    snapshot_id = str(meta.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise EventFeatureContractError("Vigilance snapshot_id is required.")
    published = _utc_timestamp(product.get("update_time"), field="product.update_time")
    revision = _utc_timestamp(
        meta.get("generation_timestamp"), field="meta.generation_timestamp"
    )
    transmission = _official_transmission_time(source_uri)
    if abs(revision - published) > _MAX_METADATA_SKEW:
        raise EventFeatureContractError(
            "Vigilance update_time and generation_timestamp differ by more than 30 minutes."
        )
    if abs(transmission - revision) > _MAX_METADATA_SKEW:
        raise EventFeatureContractError(
            "Archive transmission directory and payload revision differ by more than 30 minutes."
        )
    available = max(published, revision, transmission)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    periods = product.get("periods")
    if not isinstance(periods, list) or not periods:
        raise EventFeatureContractError("Vigilance product.periods must be a non-empty list.")

    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    departments: set[str] = set()
    events: list[EventRecord] = []
    for period_index, period in enumerate(periods):
        if not isinstance(period, dict):
            raise EventFeatureContractError(f"periods[{period_index}] must be an object.")
        echeance = str(period.get("echeance", "")).upper()
        if echeance not in {"J", "J1"}:
            raise EventFeatureContractError(f"Unexpected Vigilance echeance: {echeance!r}.")
        period_start = _utc_timestamp(
            period.get("begin_validity_time"),
            field=f"periods[{period_index}].begin_validity_time",
        )
        period_end = _utc_timestamp(
            period.get("end_validity_time"),
            field=f"periods[{period_index}].end_validity_time",
        )
        if period_start >= period_end:
            raise EventFeatureContractError("Vigilance period must have positive duration.")
        valid_periods.append((period_start, period_end))

        timelaps = period.get("timelaps")
        if not isinstance(timelaps, dict):
            raise EventFeatureContractError("Each Vigilance period must contain object 'timelaps'.")
        domain_rows = timelaps.get("domain_ids")
        if not isinstance(domain_rows, list):
            raise EventFeatureContractError("timelaps.domain_ids must be a list.")
        for domain_index, domain in enumerate(domain_rows):
            if not isinstance(domain, dict):
                raise EventFeatureContractError("Each Vigilance domain must be an object.")
            spatial_id = str(domain.get("domain_id", "")).upper()
            if not _is_metropolitan_department(spatial_id):
                # National, defence-zone and coastal-domain rows are legitimate
                # but excluded to avoid double-counting geographic exposure.
                continue
            departments.add(spatial_id)
            phenomena = domain.get("phenomenon_items", [])
            if not isinstance(phenomena, list):
                raise EventFeatureContractError("phenomenon_items must be a list.")
            for phenomenon_index, phenomenon in enumerate(phenomena):
                if not isinstance(phenomenon, dict):
                    raise EventFeatureContractError("Each phenomenon must be an object.")
                hazard_id = str(phenomenon.get("phenomenon_id", ""))
                if hazard_id not in HAZARD_NAMES:
                    raise EventFeatureContractError(
                        f"Unknown Vigilance phenomenon_id: {hazard_id!r}."
                    )
                category = HAZARD_NAMES[hazard_id]
                period_severity = _severity(
                    phenomenon.get("phenomenon_max_color_id"),
                    field=(
                        f"periods[{period_index}].domain_ids[{domain_index}]"
                        f".phenomenon_items[{phenomenon_index}]"
                        ".phenomenon_max_color_id"
                    ),
                )
                events.append(
                    EventRecord(
                        source="meteo_france_vigilance",
                        source_event_id=_event_id(
                            snapshot_id,
                            spatial_id,
                            category,
                            "period_max",
                            period_start,
                            period_end,
                        ),
                        zone="FR",
                        spatial_id=spatial_id,
                        category=category,
                        severity=period_severity,
                        temporal_precision="period_max",
                        event_start_utc=period_start,
                        event_end_utc=period_end,
                        published_at_utc=published,
                        revision_at_utc=revision,
                        available_at_utc=available,
                        snapshot_id=snapshot_id,
                        source_uri=source_uri,
                        content_sha256=digest,
                    )
                )

                intervals = phenomenon.get("timelaps_items", [])
                if not isinstance(intervals, list):
                    raise EventFeatureContractError("timelaps_items must be a list.")
                parsed_intervals: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
                for interval_index, interval in enumerate(intervals):
                    if not isinstance(interval, dict):
                        raise EventFeatureContractError("Each timelaps item must be an object.")
                    start = _utc_timestamp(
                        interval.get("begin_time"),
                        field=f"timelaps_items[{interval_index}].begin_time",
                    )
                    end = _utc_timestamp(
                        interval.get("end_time"),
                        field=f"timelaps_items[{interval_index}].end_time",
                    )
                    if start >= end or start < period_start or end > period_end:
                        raise EventFeatureContractError(
                            "Vigilance interval must be positive and contained in its period."
                        )
                    severity = _severity(
                        interval.get("color_id"),
                        field=f"timelaps_items[{interval_index}].color_id",
                    )
                    parsed_intervals.append((start, end, severity))
                parsed_intervals.sort(key=lambda item: (item[0], item[1]))
                for previous, current in zip(parsed_intervals, parsed_intervals[1:]):
                    if current[0] < previous[1]:
                        raise EventFeatureContractError(
                            "Overlapping Vigilance intervals for one department/hazard."
                        )
                for start, end, severity in parsed_intervals:
                    events.append(
                        EventRecord(
                            source="meteo_france_vigilance",
                            source_event_id=_event_id(
                                snapshot_id,
                                spatial_id,
                                category,
                                "interval",
                                start,
                                end,
                            ),
                            zone="FR",
                            spatial_id=spatial_id,
                            category=category,
                            severity=severity,
                            temporal_precision="interval",
                            event_start_utc=start,
                            event_end_utc=end,
                            published_at_utc=published,
                            revision_at_utc=revision,
                            available_at_utc=available,
                            snapshot_id=snapshot_id,
                            source_uri=source_uri,
                            content_sha256=digest,
                        )
                    )

    return VigilanceSnapshot(
        snapshot_id=snapshot_id,
        published_at_utc=published,
        revision_at_utc=revision,
        transmission_at_utc=transmission,
        available_at_utc=available,
        source_uri=source_uri,
        content_sha256=digest,
        valid_periods_utc=tuple(valid_periods),
        department_ids=tuple(sorted(departments)),
        events=tuple(events),
    )


def load_meteofrance_vigilance_snapshot(
    path: str | Path,
    *,
    source_uri: str,
) -> VigilanceSnapshot:
    """Load one locally persisted official snapshot with byte-level hashing."""

    local_path = Path(path)
    try:
        raw = local_path.read_bytes()
    except OSError as exc:
        raise EventFeatureContractError(f"Cannot read Vigilance snapshot: {local_path}") from exc
    return parse_meteofrance_vigilance_snapshot(raw, source_uri=source_uri)


def select_latest_causal_snapshot(
    snapshots: Iterable[VigilanceSnapshot],
    *,
    cutoff_utc: Any,
) -> VigilanceSnapshot:
    """Select the last complete revision demonstrably available by cutoff."""

    cutoff = _utc_timestamp(cutoff_utc, field="cutoff_utc")
    eligible = [snapshot for snapshot in snapshots if snapshot.available_at_utc <= cutoff]
    if not eligible:
        raise EventFeatureContractError(
            f"No Météo-France Vigilance snapshot was available by {cutoff.isoformat()}."
        )
    eligible.sort(
        key=lambda item: (
            item.available_at_utc,
            item.revision_at_utc,
            item.published_at_utc,
            item.snapshot_id,
        )
    )
    latest = eligible[-1]
    same_revision = [
        item
        for item in eligible
        if item.available_at_utc == latest.available_at_utc
        and item.revision_at_utc == latest.revision_at_utc
    ]
    if len({item.content_sha256 for item in same_revision}) > 1:
        raise EventFeatureContractError(
            "Ambiguous Vigilance revision: different contents share the same availability time."
        )
    return latest


def _validate_delivery_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise EventFeatureContractError("delivery_index_utc must be a non-empty DatetimeIndex.")
    if index.tz is None:
        raise EventFeatureContractError("delivery_index_utc must be timezone-aware UTC.")
    normalized = index.tz_convert("UTC")
    if not normalized.equals(index) or not normalized.is_monotonic_increasing:
        raise EventFeatureContractError("delivery_index_utc must be sorted on the UTC timeline.")
    if normalized.has_duplicates:
        raise EventFeatureContractError("delivery_index_utc cannot contain duplicates.")
    if len(normalized) > 1 and not (normalized[1:] - normalized[:-1] == pd.Timedelta(hours=1)).all():
        raise EventFeatureContractError("delivery_index_utc must be a continuous hourly grid.")
    return normalized


def _records_frame(events: tuple[EventRecord, ...]) -> pd.DataFrame:
    columns = tuple(EventRecord.__dataclass_fields__)
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([{name: getattr(event, name) for name in columns} for event in events])


def _severity_by_department(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float)
    return events.groupby("spatial_id", sort=False)["severity"].max()


def _fill_exposure_features(
    row: dict[str, float],
    events: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    department_severity = _severity_by_department(events)
    row[f"{prefix}_max_level"] = (
        float(department_severity.max()) if not department_severity.empty else 0.0
    )
    row[f"{prefix}_departments_yellow_plus"] = float((department_severity >= 1).sum())
    row[f"{prefix}_departments_orange_plus"] = float((department_severity >= 2).sum())
    row[f"{prefix}_departments_red"] = float((department_severity >= 3).sum())


def materialize_meteofrance_vigilance_features(
    snapshots: Iterable[VigilanceSnapshot],
    *,
    delivery_index_utc: pd.DatetimeIndex,
    cutoff_utc: Any,
    minimum_departments: int = 90,
    feature_hazards: Iterable[str] = DEFAULT_FEATURE_HAZARDS,
) -> EventFeatureMaterialization:
    """Materialize causal hourly warning features from the latest eligible map.

    Interval fields use only explicit ``timelaps_items``.  Period maxima are
    emitted under separately named columns, so a warning with no chronology is
    never presented as active in every delivery hour.
    """

    index = _validate_delivery_index(delivery_index_utc)
    cutoff = _utc_timestamp(cutoff_utc, field="cutoff_utc")
    if cutoff >= index[0]:
        raise EventFeatureContractError("cutoff_utc must be strictly before delivery starts.")
    if minimum_departments < 1:
        raise EventFeatureContractError("minimum_departments must be positive.")
    hazards = tuple(str(value) for value in feature_hazards)
    if not hazards or len(set(hazards)) != len(hazards):
        raise EventFeatureContractError("feature_hazards must be unique and non-empty.")
    unknown = sorted(set(hazards).difference(HAZARD_NAMES.values()))
    if unknown:
        raise EventFeatureContractError(f"Unknown feature hazards: {unknown}")

    selected = select_latest_causal_snapshot(snapshots, cutoff_utc=cutoff)
    if len(selected.department_ids) < minimum_departments:
        raise EventFeatureContractError(
            "Vigilance snapshot department coverage is too low: "
            f"{len(selected.department_ids)} < {minimum_departments}."
        )
    uncovered = [
        timestamp
        for timestamp in index
        if not any(start <= timestamp < end for start, end in selected.valid_periods_utc)
    ]
    if uncovered:
        raise EventFeatureContractError(
            "Selected Vigilance snapshot does not cover every delivery hour; first missing: "
            f"{uncovered[0].isoformat()}."
        )

    events = _records_frame(selected.events)
    rows: list[dict[str, float]] = []
    for timestamp in index:
        active = events.loc[
            (events["temporal_precision"] == "interval")
            & (events["event_start_utc"] <= timestamp)
            & (events["event_end_utc"] > timestamp)
            & (events["severity"] > 0)
        ]
        period_max = events.loc[
            (events["temporal_precision"] == "period_max")
            & (events["event_start_utc"] <= timestamp)
            & (events["event_end_utc"] > timestamp)
            & (events["severity"] > 0)
        ]
        row: dict[str, float] = {"mf_vigilance_source_available": 1.0}
        _fill_exposure_features(row, active, prefix="mf_vigilance")
        _fill_exposure_features(row, period_max, prefix="mf_vigilance_period")
        active_by_department = _severity_by_department(active)
        row["mf_vigilance_weighted_department_score"] = float(active_by_department.sum())
        row["mf_vigilance_active_hazard_count"] = float(active["category"].nunique())
        for category in hazards:
            _fill_exposure_features(
                row,
                active.loc[active["category"] == category],
                prefix=f"mf_vigilance_{category}",
            )
            category_period = period_max.loc[period_max["category"] == category]
            row[f"mf_vigilance_{category}_period_max_level"] = (
                float(category_period["severity"].max())
                if not category_period.empty
                else 0.0
            )
        rows.append(row)

    features = pd.DataFrame(rows, index=index, dtype=float)
    features.index.name = "delivery_start_utc"
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise EventFeatureContractError("Materialized Vigilance features are not finite.")
    if not selected.events:
        raise EventFeatureContractError("Selected Vigilance snapshot contains no event records.")
    selected_events = events.sort_values(
        ["temporal_precision", "category", "spatial_id", "event_start_utc"],
        kind="stable",
    ).reset_index(drop=True)
    provenance: dict[str, Any] = {
        "source": "Météo-France Vigilance météorologique archivée",
        "source_dataset_url": METEOFRANCE_VIGILANCE_DATASET_URL,
        "source_uri": selected.source_uri,
        "content_sha256": selected.content_sha256,
        "snapshot_id": selected.snapshot_id,
        "published_at_utc": selected.published_at_utc.isoformat(),
        "revision_at_utc": selected.revision_at_utc.isoformat(),
        "transmission_at_utc": selected.transmission_at_utc.isoformat(),
        "available_at_utc": selected.available_at_utc.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "causal_rule": "max(update_time, generation_timestamp, archive_transmission) <= cutoff",
        "department_count": len(selected.department_ids),
        "event_record_count": len(selected.events),
        "feature_hazards": list(hazards),
        "free_text_used": False,
        "image_used": False,
        "period_max_is_not_hourly_activity": True,
    }
    return EventFeatureMaterialization(
        features=features,
        events=selected_events,
        provenance=provenance,
    )


__all__ = [
    "DEFAULT_FEATURE_HAZARDS",
    "EventFeatureContractError",
    "EventFeatureMaterialization",
    "EventRecord",
    "HAZARD_NAMES",
    "METEOFRANCE_VIGILANCE_ARCHIVE_BASE",
    "METEOFRANCE_VIGILANCE_DATASET_URL",
    "VigilanceSnapshot",
    "day_ahead_cutoff_utc",
    "load_meteofrance_vigilance_snapshot",
    "materialize_meteofrance_vigilance_features",
    "parse_meteofrance_vigilance_snapshot",
    "select_latest_causal_snapshot",
]
