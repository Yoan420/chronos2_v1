"""Report-only Storm comparator matching the official dashboard contract.

The operational model must never import Storm as a feature or prediction
input.  This module therefore only accepts an already frozen Statistics
history, aligns a separately materialized dashboard comparator to it and
publishes a *copy* of the report snapshot.  The original strict D-1 08:00
comparator is retained under its own explicit column and audit contract.

The dashboard contract is the current extraction of the exact native Saturn
series supplied by the dashboard owners. It is not a per-delivery-day PIT
selection. Each extraction is normalized, materialized and checksummed so a
report remains reproducible if Saturn later revises the native curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


TIMEZONE = "Europe/Paris"
STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE: dict[str, str] = {
    zone: f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"
    for zone in ("FR", "DE", "BE", "NL")
}
STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE: dict[str, str] = {
    "FR": "41377_native",
    "DE": "41376_native",
    "BE": "41378_native",
    "NL": "41379_native",
}
STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE: dict[str, str] = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
}


def storm_dashboard_series(zone: str = "FR") -> str:
    """Return the verified native dashboard identifier for one zone."""

    zone_key = str(zone).strip().upper()
    try:
        return STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE[zone_key]
    except KeyError as exc:
        raise ValueError(
            "No verified native Storm dashboard series for zone "
            f"{zone_key!r}; verified zones are "
            f"{sorted(STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE)}"
        ) from exc


def fetch_native_dashboard_snapshot(
    client: Any,
    *,
    zone: str,
    expected_index: pd.DatetimeIndex,
    extracted_at_utc: pd.Timestamp | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch one auditable latest snapshot of the exact native series.

    The raw timezone-naive civil index is preserved here. Its sole timezone
    conversion belongs to :func:`normalize_native_dashboard_series`, keeping
    the DST policy identical for live runs and report-only refreshes.
    """

    zone_key = str(zone).strip().upper()
    series = storm_dashboard_series(zone_key)
    timezone = STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[zone_key]
    expected = _utc_index(expected_index, name="expected_index")
    if expected.empty:
        raise ValueError("expected_index is empty")
    start_local = expected[0].tz_convert(timezone).tz_localize(None)
    end_local = expected[-1].tz_convert(timezone).tz_localize(None)
    extracted = (
        pd.Timestamp.now(tz="UTC")
        if extracted_at_utc is None
        else pd.Timestamp(extracted_at_utc)
    )
    if extracted.tzinfo is None:
        raise ValueError("extracted_at_utc must be timezone-aware")
    extracted = extracted.tz_convert("UTC")
    raw = client.get(
        series,
        from_value_date=start_local - pd.Timedelta(hours=2),
        to_value_date=end_local + pd.Timedelta(hours=2),
        _keep_nans=True,
    )
    if not isinstance(raw, pd.Series):
        raise TypeError("Saturn native Storm response must be a pandas Series")
    source = {
        "kind": "saturn_native_latest_extraction",
        "requested_series": series,
        "series": series,
        "primary_series": STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[zone_key],
        "zone": zone_key,
        "naive_timezone": timezone,
        "extracted_at_utc": str(extracted),
        "requested_from_local": str(start_local - pd.Timedelta(hours=2)),
        "requested_to_local": str(end_local + pd.Timedelta(hours=2)),
        "selection": "latest values returned by Saturn at extraction time",
        "used_for_prediction": False,
    }
    return raw, source


# Backward-compatible FR alias.  It is the native dashboard series, not the
# separate ``.da.basecase`` materialization used by the PIT proxy loader.
STORM_DASHBOARD_SERIES = storm_dashboard_series("FR")
STORM_DASHBOARD_COLUMN = "storm_dashboard_official__q50"
STORM_STRICT_08_COLUMN = "storm_strict_08__q50"
STORM_LEGACY_STRICT_COLUMN = "storm_evaluation_only__q50"
STORM_DASHBOARD_ARTIFACT = Path(
    "inputs/storm_dashboard_official_statistics.parquet"
)
STATISTICS_HISTORY_NAME = "statistics_history_hourly.csv.gz"
STATISTICS_AUDIT_NAME = "statistics_history_audit.json"
COMPARATOR_AUDIT_NAME = "storm_dashboard_report_only_audit.json"


@dataclass(frozen=True)
class StormDashboardComparator:
    """A dashboard comparator aligned to the complete Statistics timeline."""

    values: pd.Series
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _utc_index(values: Any, *, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    index = index.tz_convert("UTC")
    if index.empty:
        raise ValueError(f"{name} is empty")
    if index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted")
    return index


def _series_utc(values: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    index = pd.DatetimeIndex(values.index)
    if index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware")
    result = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        index=index.tz_convert("UTC"),
        name=name,
    ).sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    return result


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _materialized_series(
    path: Path,
    *,
    value_column: str | None,
) -> tuple[pd.Series, str, str]:
    frame = _read_frame(path)
    timestamp_column = next(
        (
            column
            for column in (
                "delivery_start_utc",
                "value_time_utc",
                "timestamp_utc",
                "timestamp",
            )
            if column in frame
        ),
        None,
    )
    if timestamp_column is None:
        raise ValueError(
            f"{path}: missing delivery_start_utc/value_time_utc timestamp"
        )
    candidates = (
        [value_column]
        if value_column
        else [STORM_DASHBOARD_COLUMN, "value", "q50"]
    )
    selected_value = next(
        (column for column in candidates if column and column in frame),
        None,
    )
    if selected_value is None:
        raise ValueError(
            f"{path}: missing comparator value column; tried {candidates}"
        )
    naive_timestamps = [
        value
        for value in frame[timestamp_column]
        if pd.Timestamp(value).tzinfo is None
    ]
    if naive_timestamps:
        raise ValueError(
            f"{path}: {timestamp_column} must contain explicit UTC offsets"
        )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame[timestamp_column], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    values = pd.Series(
        pd.to_numeric(frame[selected_value], errors="coerce").to_numpy(float),
        index=delivery,
        name=STORM_DASHBOARD_COLUMN,
    ).sort_index()
    if values.index.has_duplicates:
        raise ValueError(f"{path}: duplicate delivery timestamps")
    return values, timestamp_column, selected_value


def _day_histogram(index: pd.DatetimeIndex, *, timezone: str) -> dict[str, int]:
    if index.empty:
        return {}
    local_days = pd.Index(index.tz_convert(timezone).date)
    hours_per_day = pd.Series(1, index=local_days).groupby(level=0).sum()
    return {
        str(int(hours)): int(days)
        for hours, days in hours_per_day.value_counts().sort_index().items()
    }


def _metric_audit(actual: pd.Series, forecast: pd.Series) -> dict[str, Any]:
    paired = pd.concat(
        [
            pd.to_numeric(actual, errors="coerce").rename("actual"),
            pd.to_numeric(forecast, errors="coerce").rename("forecast"),
        ],
        axis=1,
    ).dropna()
    if paired.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    error = (
        paired["forecast"].to_numpy(float)
        - paired["actual"].to_numpy(float)
    )
    return {
        "n": int(len(paired)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def build_dashboard_comparator(
    values: pd.Series,
    *,
    expected_index: pd.DatetimeIndex,
    actual: pd.Series,
    source: Mapping[str, Any],
    timezone: str = TIMEZONE,
    minimum_coverage: float = 1.0,
    maximum_missing_hours: int = 0,
) -> StormDashboardComparator:
    """Align and audit a materialized dashboard comparator.

    Missing dashboard hours remain missing; they are never interpolated from
    the strict 08:00 curve.  Coverage and the absolute missing-hour allowance
    are explicit parts of the caller's contract.
    """

    expected = _utc_index(expected_index, name="expected_index")
    comparator = _series_utc(values, name=STORM_DASHBOARD_COLUMN)
    actual_utc = _series_utc(actual, name="actual").reindex(expected)
    if not np.isfinite(actual_utc.to_numpy(dtype=float)).all():
        raise ValueError("canonical actuals are incomplete on the Statistics period")
    aligned = comparator.reindex(expected)
    finite = np.isfinite(aligned.to_numpy(dtype=float))
    available_index = expected[finite]
    missing = expected[~finite]
    coverage = float(finite.mean())
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    if int(maximum_missing_hours) < 0:
        raise ValueError("maximum_missing_hours must be non-negative")
    if coverage < float(minimum_coverage):
        raise ValueError(
            "Storm dashboard coverage below contract: "
            f"{coverage:.6f} < {float(minimum_coverage):.6f}"
        )
    if len(missing) > int(maximum_missing_hours):
        raise ValueError(
            "Storm dashboard has too many missing hours: "
            f"{len(missing)} > {int(maximum_missing_hours)}"
        )

    expected_local_days = pd.Index(expected.tz_convert(timezone).date)
    available_local_days = pd.Index(available_index.tz_convert(timezone).date)
    expected_counts = pd.Series(1, index=expected_local_days).groupby(level=0).sum()
    available_counts = (
        pd.Series(1, index=available_local_days).groupby(level=0).sum()
        if len(available_local_days)
        else pd.Series(dtype=int)
    )
    transition_days: list[dict[str, Any]] = []
    for day, expected_hours in expected_counts.items():
        if int(expected_hours) == 24:
            continue
        observed_hours = int(available_counts.get(day, 0))
        transition_days.append(
            {
                "local_day": str(day),
                "expected_hours": int(expected_hours),
                "available_hours": observed_hours,
                "missing_hours": int(expected_hours) - observed_hours,
            }
        )

    missing_detail = []
    for timestamp in missing[:50]:
        local = timestamp.tz_convert(timezone)
        missing_detail.append(
            {
                "utc": str(timestamp),
                "local": str(local),
                "fold": int(getattr(local.to_pydatetime(), "fold", 0)),
            }
        )
    finite_actual = int(np.isfinite(actual_utc.to_numpy(dtype=float)).sum())
    source_extra = comparator.index.difference(expected)
    metrics = _metric_audit(actual_utc, aligned)
    audit: dict[str, Any] = {
        "role": "evaluation_only_dashboard_comparator",
        "contract": str(
            source.get(
                "contract",
                "explicit materialized Storm dashboard comparator",
            )
        ),
        "series": str(source.get("series", STORM_DASHBOARD_SERIES)),
        "column": STORM_DASHBOARD_COLUMN,
        "source": dict(source),
        "period": {
            "start_utc": str(expected[0]),
            "end_utc": str(expected[-1]),
            "start_local_day": str(expected[0].tz_convert(timezone).date()),
            "end_local_day": str(expected[-1].tz_convert(timezone).date()),
            "timezone": timezone,
        },
        "expected_hours": int(len(expected)),
        "source_rows": int(len(comparator)),
        "source_extra_hours_outside_period": int(len(source_extra)),
        "available_hours": int(finite.sum()),
        "missing_hours": int(len(missing)),
        "coverage": coverage,
        "minimum_coverage": float(minimum_coverage),
        "maximum_missing_hours": int(maximum_missing_hours),
        "missing_timestamps": missing_detail,
        "missing_timestamps_truncated": bool(len(missing) > len(missing_detail)),
        "dst": {
            "expected_local_day_hour_histogram": _day_histogram(
                expected, timezone=timezone
            ),
            "available_local_day_hour_histogram": _day_histogram(
                available_index, timezone=timezone
            ),
            "transition_days": transition_days,
            "interpolation": False,
            "strict_08_fallback": False,
        },
        "actual_available_hours": finite_actual,
        "metrics": metrics,
        "used_for_prediction": False,
        "used_for_live_forecast": False,
        "candidate_frozen_before_comparator_attachment": True,
    }
    aligned.name = STORM_DASHBOARD_COLUMN
    return StormDashboardComparator(values=aligned, audit=audit)


def _delivery_day_start_utc(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    delivery = pd.DatetimeIndex(index)
    if delivery.tz is None:
        raise ValueError("delivery index must be timezone-aware")
    local_days = pd.DatetimeIndex(delivery.tz_convert(timezone).date)
    return local_days.tz_localize(
        timezone,
        ambiguous="raise",
        nonexistent="raise",
    ).tz_convert("UTC")


def load_dashboard_from_basecase_vintages(
    path: str | Path,
    *,
    expected_index: pd.DatetimeIndex,
    actual: pd.Series,
    timezone: str = TIMEZONE,
    zone: str = "FR",
) -> StormDashboardComparator:
    """Build a legacy pre-midnight PIT proxy for diagnostics only.

    The official dashboard identifier is the zone's native ``...3mv.storm``
    series and must be fetched directly. This compatibility loader applies an
    older selection rule to a separately materialized ``.da.basecase`` PIT
    artifact: for every physical
    delivery hour in civil day D, select the last vintage whose *snapshot* and
    *revision* are both strictly earlier than the start of D in ``timezone``.
    A marker exactly at midnight is ineligible.  The artifact provenance keeps
    both identifiers explicit; it must not be labelled as the official
    dashboard metric.
    """

    source_path = Path(path).expanduser().resolve()
    native_series = storm_dashboard_series(zone)
    materialization_series = (
        f"power.price.{str(zone).strip().lower()}.euromwh.h.fcst."
        "3mv.storm.da.basecase"
    )
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    expected = _utc_index(expected_index, name="expected_index")
    try:
        frame = pd.read_parquet(
            source_path,
            columns=sorted(required),
            filters=[
                ("value_time_utc", ">=", expected[0].to_pydatetime()),
                ("value_time_utc", "<=", expected[-1].to_pydatetime()),
            ],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(source_path, columns=sorted(required))
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{source_path}: missing Storm PIT columns {missing_columns}"
        )
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame = frame.loc[
        frame["value_time_utc"].between(
            expected[0], expected[-1], inclusive="both"
        )
    ].copy()
    if frame.empty:
        raise ValueError(f"{source_path}: no Storm basecase vintages in period")
    rows_in_period = int(len(frame))

    exact_key = [
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
    ]
    duplicate_keys = frame.duplicated(exact_key, keep=False)
    if bool(duplicate_keys.any()):
        conflicting = (
            frame.loc[duplicate_keys]
            .groupby(exact_key, dropna=False)["value"]
            .nunique(dropna=False)
        )
        if bool((conflicting > 1).any()):
            raise ValueError(
                f"{source_path}: conflicting values for one Storm PIT vintage"
            )
        frame = frame.drop_duplicates(exact_key, keep="last")

    delivery = pd.DatetimeIndex(frame["value_time_utc"])
    cutoff_by_row = pd.Series(
        _delivery_day_start_utc(delivery, timezone=timezone),
        index=frame.index,
    )
    eligible_mask = (
        (frame["snapshot_time_utc"] < cutoff_by_row)
        & (frame["revision_time_utc"] < cutoff_by_row)
    )
    eligible = frame.loc[eligible_mask].sort_values(
        ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
        kind="stable",
    )
    selected = eligible.drop_duplicates("value_time_utc", keep="last").copy()
    selected_index = pd.DatetimeIndex(
        selected["value_time_utc"], name="delivery_start_utc"
    )
    if not selected_index.equals(expected):
        missing = expected.difference(selected_index)
        extra = selected_index.difference(expected)
        raise ValueError(
            "Storm dashboard basecase PIT does not cover the exact Statistics "
            f"timeline: missing={len(missing)}, extra={len(extra)}"
        )

    selected_cutoff = _delivery_day_start_utc(
        selected_index, timezone=timezone
    )
    selected_snapshot = pd.DatetimeIndex(selected["snapshot_time_utc"])
    selected_revision = pd.DatetimeIndex(selected["revision_time_utc"])
    if not bool((selected_snapshot < selected_cutoff).all()):
        raise ValueError("Storm dashboard snapshot midnight-cutoff violation")
    if not bool((selected_revision < selected_cutoff).all()):
        raise ValueError("Storm dashboard revision midnight-cutoff violation")
    latest_marker = pd.DatetimeIndex(
        np.maximum(selected_snapshot.asi8, selected_revision.asi8),
        tz="UTC",
    )
    lag_minutes = (
        selected_cutoff.asi8 - latest_marker.asi8
    ) / (60.0 * 1_000_000_000.0)
    values = pd.Series(
        pd.to_numeric(selected["value"], errors="coerce").to_numpy(float),
        index=selected_index,
        name=STORM_DASHBOARD_COLUMN,
    )
    return build_dashboard_comparator(
        values,
        expected_index=expected,
        actual=actual,
        source={
            "kind": "saturn_basecase_pit_proxy_for_native_dashboard",
            "contract": (
                "native Storm dashboard contract represented by the last "
                "basecase PIT vintage with snapshot_time_utc AND "
                "revision_time_utc strictly before civil delivery-day start"
            ),
            "series": native_series,
            "series_kind": "native_exact",
            "materialization_series": materialization_series,
            "materialization_kind": "separate_basecase_pit_artifact",
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "selection_rule": (
                "snapshot_time_utc < civil D 00:00 AND revision_time_utc "
                "< civil D 00:00 Europe/Paris"
            ),
            "comparison_operator": "strict_less_than",
            "rows_in_period": rows_in_period,
            "eligible_rows": int(len(eligible)),
            "selected_rows": int(len(selected)),
            "first_selected_snapshot_utc": str(selected_snapshot.min()),
            "last_selected_snapshot_utc": str(selected_snapshot.max()),
            "first_selected_revision_utc": str(selected_revision.min()),
            "last_selected_revision_utc": str(selected_revision.max()),
            "minimum_cutoff_lag_minutes": float(np.min(lag_minutes)),
            "maximum_cutoff_lag_minutes": float(np.max(lag_minutes)),
            "snapshot_cutoff_violations": 0,
            "revision_cutoff_violations": 0,
        },
        timezone=timezone,
        minimum_coverage=1.0,
        maximum_missing_hours=0,
    )


def load_materialized_dashboard_comparator(
    path: str | Path,
    *,
    expected_index: pd.DatetimeIndex,
    actual: pd.Series,
    value_column: str | None = None,
    timezone: str = TIMEZONE,
    minimum_coverage: float = 1.0,
    maximum_missing_hours: int = 0,
    zone: str = "FR",
) -> StormDashboardComparator:
    """Load an explicit CSV/Parquet dashboard materialization."""

    source_path = Path(path).expanduser().resolve()
    values, timestamp_column, selected_value = _materialized_series(
        source_path,
        value_column=value_column,
    )
    return build_dashboard_comparator(
        values,
        expected_index=expected_index,
        actual=actual,
        source={
            "kind": "explicit_materialized_artifact",
            "series": storm_dashboard_series(zone),
            "series_kind": "native_exact",
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "timestamp_column": timestamp_column,
            "value_column": selected_value,
        },
        timezone=timezone,
        minimum_coverage=minimum_coverage,
        maximum_missing_hours=maximum_missing_hours,
    )


def normalize_native_dashboard_series(
    values: pd.Series,
    *,
    zone: str = "FR",
    expected_index: pd.DatetimeIndex,
    actual: pd.Series,
    source: Mapping[str, Any] | None = None,
) -> StormDashboardComparator:
    """Normalize the native dashboard curve without fabricating DST hours.

    Saturn's exact ``...3mv.storm`` series is timezone-naive and follows the
    local civil clock.  On the autumn transition it contains a single 02:00;
    we map that point to standard time (``fold=1``), leave the other physical
    fold missing, and compare candidate/Storm on the same finite timestamps.
    No interpolation, duplication, or fallback to the 08:00 curve is allowed.
    """

    zone_key = str(zone).strip().upper()
    timezone = STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE.get(zone_key)
    if timezone is None:
        raise ValueError(f"No native Storm timezone contract for {zone_key!r}")
    if not isinstance(values, pd.Series):
        raise TypeError("native Storm values must be a pandas Series")
    expected_series = storm_dashboard_series(zone_key)
    expected_primary = STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE[zone_key]
    supplied_source = dict(source or {})
    for key in ("requested_series", "series"):
        supplied = supplied_source.get(key)
        if supplied is not None and str(supplied) != expected_series:
            raise ValueError(
                f"native Storm {key} mismatch: {supplied!r} != "
                f"{expected_series!r}"
            )
    supplied_primary = supplied_source.get("primary_series")
    if supplied_primary is not None and str(supplied_primary) != expected_primary:
        raise ValueError(
            "native Storm primary mismatch: "
            f"{supplied_primary!r} != {expected_primary!r}"
        )
    raw_index = pd.DatetimeIndex(values.index)
    if raw_index.tz is not None:
        raise ValueError("native Storm dashboard index must be timezone-naive")
    localized = raw_index.tz_localize(
        timezone,
        ambiguous=False,
        nonexistent="NaT",
    )
    valid = ~localized.isna()
    normalized = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)[valid],
        index=localized[valid].tz_convert("UTC"),
        name=STORM_DASHBOARD_COLUMN,
    )
    if normalized.index.has_duplicates:
        raise ValueError("native Storm normalization produced duplicates")
    expected = _utc_index(expected_index, name="expected_index")
    expected_local_naive = expected.tz_convert(timezone).tz_localize(None)
    canonical_from_naive = expected_local_naive.tz_localize(
        timezone,
        ambiguous=False,
        nonexistent="NaT",
    ).tz_convert("UTC")
    allowed_missing = expected[canonical_from_naive != expected]
    actual_missing = expected.difference(normalized.index)
    unexpected_missing = actual_missing.difference(allowed_missing)
    missing_allowed_but_present = allowed_missing.difference(actual_missing)
    if len(unexpected_missing):
        raise ValueError(
            "native Storm is missing non-DST Statistics hours: "
            + ", ".join(str(value) for value in unexpected_missing[:10])
        )
    if len(missing_allowed_but_present):
        raise ValueError(
            "native Storm DST representation changed; expected missing fold(s) "
            "are present and require a new explicit contract"
        )
    provenance = supplied_source
    provenance.update(
        {
            "kind": "saturn_native_dashboard_series",
            "series": expected_series,
            "primary_series": expected_primary,
            "series_kind": "native_exact",
            "raw_index_timezone": None,
            "naive_timezone": timezone,
            "ambiguous_policy": "standard_time_fold_only",
            "nonexistent_policy": "missing_no_fill",
            "raw_rows": int(len(values)),
            "normalized_rows": int(len(normalized)),
            "dropped_nonexistent_rows": int((~valid).sum()),
        }
    )
    comparator = build_dashboard_comparator(
        normalized,
        expected_index=expected,
        actual=actual,
        source=provenance,
        timezone=timezone,
        minimum_coverage=0.0,
        maximum_missing_hours=len(allowed_missing),
    )
    comparator.audit["dst"].update(
        {
            "native_allowed_missing_hours": int(len(allowed_missing)),
            "native_allowed_missing_utc": [
                str(value) for value in allowed_missing
            ],
            "native_actual_missing_matches_allowed": True,
        }
    )
    return comparator


def _statistics_frame(run_dir: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Series]:
    path = run_dir / STATISTICS_HISTORY_NAME
    frame = pd.read_csv(path)
    required = {"delivery_start_utc", "actual"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing Statistics columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError(f"{path}: invalid Statistics timeline")
    actual = pd.Series(
        pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float),
        index=delivery,
        name="actual",
    )
    return frame, delivery, actual


def statistics_contract(
    run_dir: str | Path,
) -> tuple[pd.DatetimeIndex, pd.Series]:
    """Return the exact report Statistics timeline and canonical actuals."""

    _, delivery, actual = _statistics_frame(Path(run_dir).expanduser().resolve())
    return delivery, actual


def _write_checksum_manifest(directory: Path, *, source_run: Path) -> None:
    checksum_path = directory / "artifact_checksums.json"
    artifacts: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != checksum_path:
            artifacts.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "role": "storm_dashboard_report_snapshot_artifact",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    source_manifest = source_run / "artifact_checksums.json"
    _write_json(
        checksum_path,
        {
            "algorithm": "sha256",
            "run_type": "storm_dashboard_report_snapshot",
            "source_run": str(source_run),
            "source_checksum_manifest_sha256": (
                _sha256(source_manifest) if source_manifest.is_file() else None
            ),
            "artifacts": artifacts,
        },
    )


def _assert_separate_paths(source: Path, output: Path) -> None:
    if source == output:
        raise ValueError("report-only output must differ from the source run")
    if source in output.parents:
        raise ValueError("report-only output cannot be inside the source run")
    if output in source.parents:
        raise ValueError("report-only output cannot contain the source run")
    if output.exists():
        raise FileExistsError(
            f"report-only snapshot already exists and remains immutable: {output}"
        )


def create_report_only_copy(
    *,
    source_run_dir: str | Path,
    output_dir: str | Path,
    comparator: StormDashboardComparator,
    regenerate_html: bool = True,
    report_title: str | None = None,
    native_model: str | None = None,
    baseline_model: str | None = None,
    zone: str = "FR",
    timezone: str = TIMEZONE,
    extreme_threshold: float = 150.0,
    history_hours: int = 168,
) -> tuple[Path, Path | None]:
    """Create an immutable report copy with dashboard and strict-08 Storm.

    The source run is only read.  Forecast and model artifacts in the copy are
    byte-for-byte copies and the dashboard comparator is attached afterwards.
    """

    source = Path(source_run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    _assert_separate_paths(source, output)
    source_history = source / STATISTICS_HISTORY_NAME
    source_history_sha = _sha256(source_history)
    source_forecast = source / f"forecast_hourly_{zone.lower()}.csv"
    if not source_forecast.is_file() and zone.upper() == "FR":
        source_forecast = source / "forecast_hourly_fr.csv"
    source_forecast_sha = (
        _sha256(source_forecast) if source_forecast.is_file() else None
    )
    # Native Storm has one local 02:00 at the autumn transition while the
    # physical market day has two folds.  The comparator builder has already
    # enforced its explicit coverage and absolute missing-hour guards.  Keep
    # that fold missing and pair candidate/Storm on finite timestamps only.
    if int(comparator.audit.get("available_hours", 0)) <= 0:
        raise ValueError("official Storm dashboard comparator is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    report_path: Path | None = None
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for old_html in staging.glob("*.html"):
            old_html.unlink()

        history, delivery, actual = _statistics_frame(staging)
        dashboard = comparator.values.reindex(delivery)
        if not dashboard.index.equals(delivery):
            raise ValueError("Storm dashboard timeline differs from Statistics")
        if STORM_LEGACY_STRICT_COLUMN in history:
            history[STORM_STRICT_08_COLUMN] = pd.to_numeric(
                history[STORM_LEGACY_STRICT_COLUMN], errors="coerce"
            )
        history[STORM_DASHBOARD_COLUMN] = dashboard.to_numpy(dtype=float)
        history_path = staging / STATISTICS_HISTORY_NAME
        history.to_csv(history_path, index=False, compression="gzip")

        materialized_path = staging / STORM_DASHBOARD_ARTIFACT
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "delivery_start_utc": delivery,
                STORM_DASHBOARD_COLUMN: dashboard.to_numpy(dtype=float),
            }
        ).to_parquet(materialized_path, index=False)

        statistics_audit_path = staging / STATISTICS_AUDIT_NAME
        statistics_audit = _read_json(statistics_audit_path)
        original_storm = statistics_audit.get("storm")
        strict_values = (
            pd.Series(
                pd.to_numeric(
                    history[STORM_STRICT_08_COLUMN], errors="coerce"
                ).to_numpy(float),
                index=delivery,
            )
            if STORM_STRICT_08_COLUMN in history
            else pd.Series(np.nan, index=delivery)
        )
        strict_audit = {
            "role": "evaluation_only_strict_08_comparator",
            "contract": (
                "last snapshot and revision <= civil D-1 08:00 Europe/Paris"
            ),
            "column": STORM_STRICT_08_COLUMN,
            "metrics": _metric_audit(actual, strict_values),
            "used_for_prediction": False,
            "available": bool(np.isfinite(strict_values.to_numpy(float)).any()),
        }
        if isinstance(original_storm, Mapping):
            strict_audit["original_audit"] = dict(original_storm)
        dashboard_audit = dict(comparator.audit)
        dashboard_audit["normalized_artifact_path"] = str(
            materialized_path.relative_to(staging).as_posix()
        )
        dashboard_audit["normalized_artifact_sha256"] = _sha256(
            materialized_path
        )
        strict_mae = strict_audit["metrics"].get("mae")
        dashboard_mae = dashboard_audit.get("metrics", {}).get("mae")
        mae_delta = (
            float(dashboard_mae) - float(strict_mae)
            if dashboard_mae is not None and strict_mae is not None
            else None
        )
        statistics_audit.update(
            {
                "statistics_history_sha256": _sha256(history_path),
                "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
                "storm_dashboard": dashboard_audit,
                "storm_strict_08": strict_audit,
                "storm_dashboard_mae_minus_strict_08_mae": mae_delta,
                "storm_used_for_prediction": False,
                "historical_forecasts_rewritten": False,
                "sealed_benchmark_rewritten": False,
            }
        )
        prior_note = str(statistics_audit.get("report_scope_note", "")).strip()
        explicit_note = (
            "Storm officiel dashboard (extraction courante de la série "
            "native exacte, figée et checksumée) est le benchmark primaire "
            "du rapport "
            f"({dashboard_audit['available_hours']}/"
            f"{dashboard_audit['expected_hours']} heures appariées). "
            "Le snapshot causal 08:00 reste séparé dans "
            f"{STORM_STRICT_08_COLUMN} à titre diagnostique. Storm est attaché "
            "après gel du candidat et n'est utilisé ni par le modèle ni par "
            "la prévision live."
        )
        statistics_audit["report_scope_note"] = (
            f"{prior_note} {explicit_note}".strip()
        )
        _write_json(statistics_audit_path, statistics_audit)

        run_manifest_path = staging / "run_manifest.json"
        run_manifest = _read_json(run_manifest_path)
        run_manifest.update(
            {
                "run_type": "storm_dashboard_report_snapshot",
                "source_run": str(source),
                "source_statistics_history_sha256": source_history_sha,
                "source_forecast_sha256": source_forecast_sha,
                "storm_dashboard": dashboard_audit,
                "storm_strict_08": strict_audit,
                "storm_used_for_prediction": False,
                "forecast_recomputed": False,
                "forecast_rewritten": False,
                "report_snapshot_created_at_utc": str(pd.Timestamp.now(tz="UTC")),
            }
        )
        _write_json(run_manifest_path, run_manifest)

        copy_audit = {
            "mode": "report_only_copy",
            "source_run": str(source),
            "output_dir": str(output),
            "source_statistics_history_sha256": source_history_sha,
            "source_forecast_sha256": source_forecast_sha,
            "dashboard_column": STORM_DASHBOARD_COLUMN,
            "strict_08_column": STORM_STRICT_08_COLUMN,
            "legacy_strict_column_preserved": (
                STORM_LEGACY_STRICT_COLUMN in history
            ),
            "dashboard": dashboard_audit,
            "strict_08": strict_audit,
            "used_for_prediction": False,
            "source_modified": False,
        }
        _write_json(staging / COMPARATOR_AUDIT_NAME, copy_audit)

        if regenerate_html:
            from chronos2_hourly.reporting import write_hourly_html_report

            report_path = staging / (
                f"chronos2_hourly_{zone.lower()}_storm_dashboard.html"
            )
            write_hourly_html_report(
                staging,
                output_path=report_path,
                title=report_title,
                native_model=native_model,
                baseline_model=baseline_model,
                zone=zone,
                timezone=timezone,
                extreme_threshold=extreme_threshold,
                history_hours=history_hours,
            )

        if _sha256(source_history) != source_history_sha:
            raise RuntimeError("source Statistics history changed during copy")
        if source_forecast_sha is not None and _sha256(source_forecast) != source_forecast_sha:
            raise RuntimeError("source forecast changed during copy")
        _write_checksum_manifest(staging, source_run=source)
        staging.replace(output)
        if report_path is not None:
            report_path = output / report_path.name
        return output, report_path
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = (
    "COMPARATOR_AUDIT_NAME",
    "STATISTICS_AUDIT_NAME",
    "STATISTICS_HISTORY_NAME",
    "STORM_DASHBOARD_COLUMN",
    "STORM_DASHBOARD_ARTIFACT",
    "STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE",
    "STORM_DASHBOARD_SERIES",
    "STORM_LEGACY_STRICT_COLUMN",
    "STORM_STRICT_08_COLUMN",
    "StormDashboardComparator",
    "build_dashboard_comparator",
    "create_report_only_copy",
    "fetch_native_dashboard_snapshot",
    "load_dashboard_from_basecase_vintages",
    "load_materialized_dashboard_comparator",
    "normalize_native_dashboard_series",
    "STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE",
    "STORM_DASHBOARD_NATIVE_PRIMARY_BY_ZONE",
    "statistics_contract",
    "storm_dashboard_series",
)
