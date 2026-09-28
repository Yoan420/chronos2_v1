"""Pure builders for lagged French hydro proxies from Energy-Charts public power.

These are revised historical actuals, NOT archived provider forecasts. A 48 h
physical lag is an explicit research assumption, not proof of historical
publication availability. No prices, labels or residual-load inputs are used.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd

PROTOCOL = "nyx_fr_public_hydro_lag48_v1"
SERIES = {
    "hydro_run_of_river": "ror_gw",
    "hydro_water_reservoir": "hydro_water_reservoir_gw",
    "hydro_pumped_storage": "hydro_pumped_storage_gw",
}
FEATURE_SPECS = {
    "fr_ror_gw_lag48h": ("ror_gw", 1),
    "fr_ror_gw_mean24h_lag48h": ("ror_gw", 24),
    "fr_ror_gw_mean168h_lag48h": ("ror_gw", 168),
    "fr_ror_gw_trend24h_minus168h_lag48h": ("ror_gw", "trend"),
    "fr_hydro_water_reservoir_gw_lag48h": ("hydro_water_reservoir_gw", 1),
    "fr_hydro_water_reservoir_gw_mean24h_lag48h": ("hydro_water_reservoir_gw", 24),
    "fr_hydro_pumped_storage_gw_lag48h": ("hydro_pumped_storage_gw", 1),
    "fr_hydro_pumped_storage_gw_mean24h_lag48h": ("hydro_pumped_storage_gw", 24),
}
FEATURES = tuple(FEATURE_SPECS)
TIMEZONE = "Europe/Paris"
LAG_HOURS = 48
HOURLY_COLUMNS = [*SERIES.values(), *(f"{v}__coverage" for v in SERIES.values()), "source_interval_end_utc"]
INTERVAL_POLICY = ("A payload declared 60 minutes may contain an effective 15-minute hour only when all "
                   "four starts 00/15/30/45 are present in that payload. Incomplete contradictory hours "
                   "are rejected. A payload declared 15 minutes is never promoted to hourly resolution.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
                          +json.dumps(list(frame.columns)).encode()).hexdigest()


def _utc_index(index, name, *, nonempty=True):
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and not index.hasnans and index.is_unique and index.is_monotonic_increasing
            and (len(index) > 0 or not nonempty) and index.equals(index.floor("h")),
            name+" must contain unique ordered physical UTC hours")


def _empty_hourly():
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp_utc")
    result = pd.DataFrame(index=index)
    for column in HOURLY_COLUMNS[:-1]: result[column] = pd.Series([], index=index, dtype=float)
    result["source_interval_end_utc"] = pd.Series([], index=index, dtype="datetime64[ns, UTC]")
    return result


def public_power_hourly(payloads: Sequence[dict]) -> pd.DataFrame:
    """Extract only 3 physical generation series; never subtract them from load.

    ISO timestamps denote interval starts. Within each UTC hour, accept exactly
    four distinct 15-minute intervals or one 60-minute interval. A missing row
    or value leaves that series' hourly mean NaN and records fractional coverage.
    A declared-60-minute payload containing subhours must have all four quarter
    starts within that physical hour; then that hour is explicitly reinterpreted
    as 15 minutes and recorded in attrs. Incomplete contradictory hours fail.
    Equal overlaps between monthly files are harmless; conflicting revisions or
    overlapping effective resolutions are rejected, never silently chosen.
    """
    records, corrections = {}, []
    for payload_index, payload in enumerate(payloads):
        require(payload.get("schema_version") == "2.0" and payload.get("endpoint") == "public_power"
                and str(payload.get("country", "")).lower() == "fr"
                and payload.get("timezone") == TIMEZONE and payload.get("unit") == "MW",
                "French v2 public_power in MW with explicit Europe/Paris metadata required")
        step = payload.get("interval_minutes")
        require(type(step) is int and step in (15, 60)
                and payload.get("resolution") in (("PT15M",) if step == 15 else ("PT1H", "PT60M")),
                "Only explicit 15-minute or 60-minute intervals are supported")
        declarations = {row["id"]: row for row in payload.get("series", [])}
        require(all(name in declarations and declarations[name].get("unit", "MW") == "MW" for name in SERIES),
                "All three physical hydro generation series in MW must be declared")
        # generated_at is deliberately ignored: a retrieval-time field cannot
        # establish the timestamp when these historical actuals were published.
        parsed, starts_by_hour = {}, {}
        for row in payload.get("data", []):
            stamp = pd.Timestamp(row["timestamp"])
            require(not pd.isna(stamp) and stamp.tzinfo is not None, "Explicit ISO timezone/offset required")
            stamp = stamp.tz_convert("UTC")
            require(stamp == stamp.floor("15min"), "Source interval is not on a supported physical grid")
            values = []
            for name in SERIES:
                value = row.get("values", {}).get(name)
                require(value is None or (isinstance(value, Real) and not isinstance(value, bool)
                        and not np.isinf(value)), "Hydro values must be finite numeric MW or missing")
                values.append(np.nan if value is None else float(value)/1000.)
            current_values = tuple(values)
            if stamp in parsed:
                require(np.array_equal(parsed[stamp], current_values, equal_nan=True),
                        "Overlapping source intervals disagree; historical revision choice is unspecified")
            else:
                parsed[stamp] = current_values
                starts_by_hour.setdefault(stamp.floor("h"), []).append(stamp)
        effective_steps = dict.fromkeys(parsed, step)
        if step == 60:
            for hour, starts in starts_by_hour.items():
                if any(stamp != hour for stamp in starts):
                    expected = pd.date_range(hour, periods=4, freq="15min")
                    require(set(starts) == set(expected), "Incomplete quarter-hour block contradicts declared 60-minute metadata")
                    for stamp in starts: effective_steps[stamp] = 15
                    corrections.append({"payload_index": payload_index, "hour_start_utc": hour.isoformat(),
                        "hour_end_utc": (hour+pd.Timedelta(hours=1)).isoformat(),
                        "declared_interval_minutes": 60, "declared_resolution": payload["resolution"],
                        "effective_interval_minutes": 15, "distinct_interval_starts": 4})
        for stamp, values in parsed.items():
            effective_step = effective_steps[stamp]
            current = (effective_step, values)
            if stamp in records:
                previous = records[stamp]
                require(previous[0] == effective_step and np.array_equal(previous[1], current[1], equal_nan=True),
                        "Overlapping source intervals disagree; historical revision choice is unspecified")
            else:
                records[stamp] = current
    if not records:
        result = _empty_hourly()
        result.attrs["interval_metadata"] = {"policy": INTERVAL_POLICY, "corrections": corrections,
                                               "effective_hour_counts_by_interval_minutes": {}}
        return result
    stamps = pd.DatetimeIndex(sorted(records))
    raw = pd.DataFrame([records[t][1] for t in stamps], index=stamps, columns=list(SERIES.values()))
    steps = pd.Series([records[t][0] for t in stamps], index=stamps)
    require((steps.resample("h").nunique() <= 1).all(), "Overlapping mixed resolutions within one hour")
    expected = 60/steps.resample("h").first()
    means, counts = raw.resample("h").mean(), raw.resample("h").count()
    coverage = counts.div(expected, axis=0).fillna(0.)
    result = means.where(coverage.eq(1.))
    result.index.name = "timestamp_utc"
    for column in SERIES.values(): result[column+"__coverage"] = coverage[column]
    result["source_interval_end_utc"] = result.index+pd.Timedelta(hours=1)
    result.attrs["interval_metadata"] = {"policy": INTERVAL_POLICY, "corrections": corrections,
        "effective_hour_counts_by_interval_minutes": {
            str(int(step)): int(count) for step, count in steps.resample("h").first().value_counts().items()}}
    return result[HOURLY_COLUMNS]


def _default_origins(target_index):
    local = target_index.tz_convert(TIMEZONE)
    # Recreate each civil date before subtraction; never subtract a fixed 24h
    # from a timezone-aware midnight across a daylight-saving transition.
    civil_previous = pd.DatetimeIndex(local.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)
    return civil_previous.tz_localize(TIMEZONE).tz_convert("UTC")


def build_fr_hydro_features(hourly: pd.DataFrame, target_index: pd.DatetimeIndex,
                           origins_utc: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, dict]:
    """Return exactly 8 features plus per-feature interval-end/coverage audit.

    Each value uses the hour starting target-48h, or the 24/168 consecutive
    physical hours ending with that hour. Mean windows require every hourly
    mean to have complete underlying interval coverage. No interpolation or
    carry-forward occurs. ``audit['rows']`` is a separate DataFrame suited for
    Parquet; all other audit fields are JSON-serializable metadata.
    """
    _utc_index(target_index, "Target index")
    _utc_index(hourly.index, "Hydro source index", nonempty=False)
    require(set(HOURLY_COLUMNS).issubset(hourly.columns), "Hourly hydro coverage and interval-end metadata required")
    declared_end = pd.DatetimeIndex(hourly.source_interval_end_utc)
    require(declared_end.equals(hourly.index+pd.Timedelta(hours=1)), "Hourly interval-end identity differs")
    for column in SERIES.values():
        coverage = hourly[column+"__coverage"].to_numpy(dtype=float)
        values = hourly[column].to_numpy(dtype=float)
        require(np.isfinite(coverage).all() and ((coverage >= 0)&(coverage <= 1)).all()
                and not np.isinf(values).any() and np.array_equal(np.isfinite(values), coverage == 1.),
                "Hourly means require complete underlying source coverage")
    deadline = _default_origins(target_index)
    origins = deadline if origins_utc is None else origins_utc
    require(isinstance(origins, pd.DatetimeIndex) and str(origins.tz) == "UTC" and len(origins) == len(target_index)
            and not origins.hasnans and (origins <= deadline).all(), "Origins must be no later than D-1 08:00 Europe/Paris")
    lagged_starts = target_index-pd.Timedelta(hours=LAG_HOURS)
    requested_ends = lagged_starts+pd.Timedelta(hours=1)
    require((requested_ends <= origins).all(), "Source interval would end after the forecast origin")
    source_hash = _frame_hash(hourly)
    interval_metadata_json = json.dumps(hourly.attrs.get("interval_metadata", {}), sort_keys=True, allow_nan=False)
    if len(hourly):
        dense_index = pd.date_range(hourly.index[0], hourly.index[-1], freq="h", tz="UTC")
        dense = hourly[list(SERIES.values())].reindex(dense_index)
    else:
        dense = pd.DataFrame(columns=list(SERIES.values()), index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    features = pd.DataFrame(index=target_index.copy())
    rows = pd.DataFrame({"origin_utc": origins, "requested_source_interval_end_utc": requested_ends}, index=target_index.copy())
    for feature, (source, window) in FEATURE_SPECS.items():
        effective_window = 168 if window == "trend" else window
        if window == 1:
            series = dense[source]
        elif window == "trend":
            series = dense[source].rolling(24, min_periods=24).mean()-dense[source].rolling(168, min_periods=168).mean()
        else:
            series = dense[source].rolling(window, min_periods=window).mean()
        values = series.reindex(lagged_starts).to_numpy(dtype=float)
        available = np.isfinite(values)
        features[feature] = values
        # Coverage describes complete hourly means in the requested window,
        # not a fabricated probability of data being available historically.
        coverage = (dense[source].notna().astype(float).rolling(effective_window, min_periods=1).sum()/effective_window)
        rows[feature+"__coverage"] = coverage.reindex(lagged_starts).fillna(0.).to_numpy(dtype=float)
        endpoints = pd.Series(requested_ends, index=target_index).where(available)
        rows[feature+"__latest_source_interval_end_utc"] = endpoints
        rows[feature+"__age_at_origin_hours"] = np.where(available, (origins-requested_ends).total_seconds()/3600., np.nan)
    require(tuple(features.columns) == FEATURES and _frame_hash(hourly) == source_hash, "Source mutated during pure feature construction")
    metadata = {
        "protocol": PROTOCOL, "country": "FR", "timezone": TIMEZONE, "units": "GW",
        "feature_names": list(FEATURES), "features_sha256": _frame_hash(features),
        "source_hourly_sha256": source_hash, "audit_rows_sha256": _frame_hash(rows),
        "source_interval_metadata": json.loads(interval_metadata_json),
        "source_interval_metadata_sha256": hashlib.sha256(interval_metadata_json.encode()).hexdigest(),
        "lag_hours_physical": LAG_HOURS, "source_timestamp_semantics": "interval start",
        "cutoff_rule": "latest used source interval end <= D-1 08:00 Europe/Paris",
        "cutoff_checked_for_all_rows": True, "rows": rows,
        "dataset_forecast": False, "revisable_historical_actuals": True,
        "provider_vintage_archive_available": False, "historical_publication_availability_proven": False,
        "assumed_delay_not_publication_timestamp": True, "generated_at_used_as_available_at": False,
        "operational_point_in_time_certified": False, "independent_validation": False,
        "residual_load_modified": False, "prices_or_labels_used": False,
        "missing_values_filled": False,
        "availability_note": "48-hour lag is a conservative research assumption. Retrieved historical actuals may include later revisions; no historical provider publication timestamp or archived forecast vintage is established.",
    }
    return features, metadata
