"""Five signed historical exchange descriptors; no forecasts or capacity proxies.

Source values are retrospectively returned observations/schedules. The fixed
48-hour physical lag and nominal H+1 deadline do not certify historical vintages.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Real

import numpy as np
import pandas as pd

PROTOCOL = "nyx_lagged_exchange_features_v1"
ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = dict(zip(ZONES, ("Europe/Paris", "Europe/Berlin", "Europe/Brussels", "Europe/Amsterdam")))
SOURCE_COLUMNS = ("de_fr_gw", "de_nl_gw", "de_be_gw", "de_net_gw", "fr_commercial_gw")
LEVELS = ("exchange__de_physical_fr_gw_lag48h", "exchange__de_physical_nl_gw_lag48h",
          "exchange__de_physical_be_gw_lag48h", "exchange__de_physical_net_gw_lag48h",
          "exchange__fr_commercial_net_gw_lag48h")
COLUMNS = tuple(c for level in LEVELS for c in (level, level+"__available"))
HOURLY_COLUMNS = (*SOURCE_COLUMNS, *(c+"__coverage" for c in SOURCE_COLUMNS), "source_interval_end_utc")
INTERVAL_POLICY = ("Declared 60-minute FR hours with subhours require all four 00/15/30/45 starts "
                   "within that payload, then explicitly become 15-minute hours. Incomplete "
                   "contradictions and overlapping effective resolutions fail. Declared 15-minute "
                   "missing slots/values produce NaN, never hourly promotion or filling.")
CONTRACT = {"protocol": PROTOCOL, "new_columns": list(COLUMNS), "source_columns": list(SOURCE_COLUMNS),
            "DE_fields": ["france", "netherlands", "belgium", "sum"],
            "FR_field": "cross_border_electricity_trading", "unit": "GW",
            "arithmetic": "float64 per observation; FR MW/1000 before complete-hour arithmetic mean; DE unchanged",
            "sign": "positive import, negative export; raw sign preserved",
            "FR_sign_metadata": "public_power attributes empty; commercial-product convention documented separately",
            "lag_hours_physical": 48, "nominal_publication_delay_hours": 1,
            "cutoff": "source interval end + nominal 1h <= previous civil day 08:00 in target country",
            "interval_policy": INTERVAL_POLICY, "common_regional_features": True,
            "missing_value": "NaN", "missing_flag": 0, "known_zero_retained": True,
            "imputation": False, "rolling_windows": False, "capacity_inference": False,
            "physical_flow_forecast": False, "historical_publication_vintages_certified": False,
            "provider_ingestion_delay_certified": False, "generated_at_is_publication_time": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
                          +json.dumps(list(frame.columns)).encode()).hexdigest()


def utc_grid(index, name, nonempty=True):
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and not index.hasnans and index.is_unique and index.is_monotonic_increasing
            and (len(index) > 0 or not nonempty) and index.equals(index.floor("h")),
            name+" must be unique ordered physical UTC hours")


def _hourly(payloads, country, fields, columns):
    records, corrections = {}, []
    unit, endpoint, divisor = ("GW", "cbpf", 1.) if country == "de" else ("MW", "public_power", 1000.)
    for payload_number, obj in enumerate(payloads):
        require(obj.get("schema_version") == "2.0" and obj.get("country") == country
                and obj.get("endpoint") == endpoint and obj.get("unit") == unit
                and obj.get("timezone") == TIMEZONES[country.upper()], "Source identity/unit/timezone differs")
        step = obj.get("interval_minutes")
        allowed = (15,) if country == "de" else (15, 60)
        require(type(step) is int and step in allowed
                and obj.get("resolution") in (("PT15M",) if step == 15 else ("PT1H", "PT60M")),
                "Unsupported interval metadata")
        if country == "de":
            require(obj.get("attributes", {}).get("sign_convention") == "positive = import, negative = export",
                    "Physical signed-flow convention differs")
        declarations = {row["id"]: row for row in obj.get("series", [])}
        require(len(declarations) == len(obj.get("series", [])) and all(
            name in declarations and declarations[name].get("unit", unit) == unit for name in fields),
            "Missing or duplicate signed exchange series")
        parsed, by_hour = {}, {}
        for row in obj.get("data", []):
            t = pd.Timestamp(row["timestamp"])
            require(not pd.isna(t) and t.tzinfo is not None, "Explicit observation timezone required")
            t = t.tz_convert("UTC")
            require(t == t.floor("15min"), "Off-grid exchange interval")
            values = []
            for field in fields:
                value = row.get("values", {}).get(field)
                require(value is None or (isinstance(value, Real) and not isinstance(value, bool)
                        and not np.isinf(value)), "Invalid signed exchange value")
                values.append(np.nan if value is None else float(value)/divisor)
            values = tuple(values)
            if t in parsed:
                require(np.array_equal(parsed[t], values, equal_nan=True), "Conflicting duplicate exchange interval")
            else:
                parsed[t] = values
                by_hour.setdefault(t.floor("h"), []).append(t)
        effective = dict.fromkeys(parsed, step)
        if step == 60:
            for hour, starts in by_hour.items():
                if any(t != hour for t in starts):
                    require(set(starts) == set(pd.date_range(hour, periods=4, freq="15min")),
                            "Incomplete subhour block contradicts declared 60-minute interval")
                    for t in starts:
                        effective[t] = 15
                    corrections.append({"payload_index": payload_number, "hour_start_utc": hour.isoformat(),
                                        "declared_minutes": 60, "effective_minutes": 15})
        for t, values in parsed.items():
            if t in records:
                require(records[t][0] == effective[t] and np.array_equal(records[t][1], values, equal_nan=True),
                        "Conflicting exchange revisions or effective resolutions")
            else:
                records[t] = (effective[t], values)
    idx = pd.DatetimeIndex(sorted(records), tz="UTC") if not records else pd.DatetimeIndex(sorted(records))
    if len(idx):
        raw = pd.DataFrame([records[t][1] for t in idx], index=idx, columns=columns, dtype=float)
        steps = pd.Series([records[t][0] for t in idx], index=idx)
        require((steps.resample("h").nunique() <= 1).all(), "Mixed overlapping effective resolutions")
        coverage = raw.resample("h").count().div(60/steps.resample("h").first(), axis=0).fillna(0.)
        result = raw.resample("h").mean().where(coverage.eq(1.))
        counts = {str(int(k)): int(v) for k, v in steps.resample("h").first().value_counts().items()}
    else:
        result = pd.DataFrame(index=idx, columns=columns, dtype=float)
        coverage, counts = result.copy(), {}
    for c in columns:
        result[c+"__coverage"] = coverage[c]
    result.attrs["interval_metadata"] = {"corrections": corrections, "effective_hour_counts": counts}
    return result


def exchange_hourly(de_payloads, fr_payloads):
    de = _hourly(de_payloads, "de", ("france", "netherlands", "belgium", "sum"), SOURCE_COLUMNS[:4])
    fr = _hourly(fr_payloads, "fr", ("cross_border_electricity_trading",), SOURCE_COLUMNS[4:])
    result = pd.concat([de, fr], axis=1).sort_index()
    for c in SOURCE_COLUMNS:
        result[c+"__coverage"] = result[c+"__coverage"].fillna(0.)
    result.index.name = "timestamp_utc"
    result["source_interval_end_utc"] = result.index+pd.Timedelta(hours=1)
    result.attrs["interval_metadata"] = {"policy": INTERVAL_POLICY,
        "DE": de.attrs["interval_metadata"], "FR": fr.attrs["interval_metadata"]}
    return result.loc[:, HOURLY_COLUMNS]


def build_features(hourly, index, origins_utc=None):
    utc_grid(index, "Target")
    utc_grid(hourly.index, "Source", nonempty=False)
    require(tuple(hourly.columns) == HOURLY_COLUMNS, "Hourly exchange schema differs")
    require(pd.DatetimeIndex(hourly.source_interval_end_utc).equals(hourly.index+pd.Timedelta(hours=1)),
            "Source interval-end identity differs")
    source_hash = frame_hash(hourly)
    for c in SOURCE_COLUMNS:
        coverage, values = hourly[c+"__coverage"].to_numpy(float), hourly[c].to_numpy(float)
        require(np.isfinite(coverage).all() and ((coverage >= 0)&(coverage <= 1)).all()
                and not np.isinf(values).any() and np.array_equal(np.isfinite(values), coverage == 1),
                "Hourly value/coverage mismatch")
    cutoffs = {}
    for z, tz in TIMEZONES.items():
        days = pd.DatetimeIndex(index.tz_convert(tz).date)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)
        cutoffs[z] = days.tz_localize(tz).tz_convert("UTC")
    origins = cutoffs["FR"] if origins_utc is None else origins_utc
    require(isinstance(origins, pd.DatetimeIndex) and str(origins.tz) == "UTC" and len(origins) == len(index)
            and not origins.hasnans and all((origins <= x).all() for x in cutoffs.values()),
            "Origin later than country civil D-1 08:00")
    starts = index-pd.Timedelta(hours=48)
    ends = starts+pd.Timedelta(hours=1)
    nominal_available = ends+pd.Timedelta(hours=1)
    require((nominal_available <= origins).all(), "Nominal publication would exceed origin")
    aligned = hourly.reindex(starts)
    features = pd.DataFrame(index=index.copy())
    rows = pd.DataFrame({"origin_utc": origins, "source_interval_start_utc": starts,
        "source_interval_end_utc": ends, "nominal_publication_utc": nominal_available,
        "nominal_margin_hours": (origins-nominal_available).total_seconds()/3600.}, index=index.copy())
    for source, level in zip(SOURCE_COLUMNS, LEVELS):
        values = aligned[source].to_numpy(float)
        features[level] = values
        features[level+"__available"] = np.isfinite(values).astype(float)
        rows[source+"__coverage"] = aligned[source+"__coverage"].fillna(0.).to_numpy(float)
    require(tuple(features.columns) == COLUMNS and frame_hash(hourly) == source_hash, "Pure source mutated")
    audit = {"protocol": PROTOCOL, "contract": CONTRACT, "source_hourly_sha256": source_hash,
             "features_sha256": frame_hash(features), "audit_rows_sha256": frame_hash(rows),
             "interval_metadata": hourly.attrs.get("interval_metadata", {}),
             "cutoff_checked_all_zones_and_rows": True, "minimum_nominal_margin_hours": float(rows.nominal_margin_hours.min()),
             "available_hours": {k: int(features[k+"__available"].sum()) for k in LEVELS},
             "prices_or_labels_used": False, "missing_values_filled": False,
             "historical_publication_vintages_certified": False, "rows": rows}
    return {z: features.copy(deep=True) for z in ZONES}, audit


def append_features(base, added):
    require(set(base) == set(added) == set(ZONES), "Exactly four country matrices required")
    result = {}
    for z in ZONES:
        require(base[z].columns.is_unique and not set(base[z].columns).intersection(COLUMNS)
                and base[z].index.equals(added[z].index) and tuple(added[z].columns) == COLUMNS,
                "Exchange append index/schema differs")
        result[z] = pd.concat([base[z], added[z]], axis=1)
        pd.testing.assert_frame_equal(result[z].loc[:, base[z].columns], base[z], check_exact=True)
    return result
