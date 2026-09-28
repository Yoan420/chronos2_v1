"""Additional causal features from current NYX forecasts and frozen fuel quotes.

No labels, interpolation, imputation, fitting, or I/O. Forecast target times may
be later than their origins; quote-source times and as-of metadata may not be
later than each delivery day's civil D-1 08:00 cutoff.
"""
from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd

ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
FUEL_COLUMNS = ("ttf_m1_eur_mwh_th", "eua_first_dec_eur_tco2", "ttf_change_1d", "ttf_change_5d",
                "eua_change_1d", "eua_change_5d", "fuel_volatility_20d", "ccgt_marginal_cost_eur_mwh")
FUEL_TIME_COLUMNS = ("value_time_utc", "snapshot_time_utc", "revision_time_utc",
                     "ttf_source_value_time_utc", "eua_source_value_time_utc", "market_source_value_time_utc")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(index, name):
    _require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC" and len(index)
             and not index.hasnans and index.is_unique and index.is_monotonic_increasing
             and index.equals(index.floor("h")), f"{name}: ordered unique UTC hourly index required")
    return index


def _times(series, name):
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    stamps = [pd.Timestamp(value) for value in series]
    _require(all(pd.isna(value) or value.tzinfo is not None for value in stamps),
             f"{name}: timestamps must have an explicit timezone")
    return pd.Series(pd.to_datetime(stamps, utc=True), index=series.index, name=series.name)


def cutoff_times(index, timezone):
    days = index.tz_convert(timezone).date
    cutoffs = {day: pd.Timestamp(f"{day-timedelta(days=1)} 08:00", tz=timezone).tz_convert("UTC") for day in set(days)}
    return pd.Series([cutoffs[day] for day in days], index=index, dtype="datetime64[ns, UTC]")


def _latest(*series):
    result = series[0].copy()
    for value in series[1:]:
        result = result.where(result >= value, value).where(result.notna() & value.notna())
    return result


def normalize_fuel(frame):
    """value_time is the delivery key, not a quote-observation timestamp."""
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
             and set(frame.columns) == set(FUEL_COLUMNS) | set(FUEL_TIME_COLUMNS), "Unexpected fuel columns")
    result = frame.copy()
    for name in FUEL_TIME_COLUMNS:
        result[name] = _times(result[name], name)
    delivery = pd.DatetimeIndex(result["value_time_utc"])
    _index(delivery, "fuel value_time_utc")
    if isinstance(result.index, pd.DatetimeIndex):
        _require(result.index.equals(delivery), "Fuel index and value_time_utc differ")
    else:
        _require(isinstance(result.index, pd.RangeIndex), "Use the explicit fuel value_time_utc key")
    result.index = delivery
    for name in FUEL_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    return result


def build_additional_features(*, zone, delivery_index, nyx_forecasts, forecast_origins, fuel):
    index = _index(delivery_index, "delivery_index")
    _require(zone in ZONES and set(nyx_forecasts) == set(forecast_origins) == set(ZONES),
             "Exactly four declared NYX countries and their origins are required")
    timezone = TIMEZONES[zone]
    cutoff = cutoff_times(index, timezone)
    predictions, origins, eligible, diagnostics = {}, {}, {}, {}
    for country in ZONES:
        supplied = nyx_forecasts[country]
        _require(isinstance(supplied, pd.DataFrame) and supplied.columns.is_unique, "NYX quantile frame required")
        _index(supplied.index, f"NYX {country}")
        if set(supplied.columns) == {f"nyx__{q}" for q in ("q10", "q50", "q90")}:
            supplied = supplied.rename(columns=lambda name: name[5:])
        _require(set(supplied.columns) == {"q10", "q50", "q90"},
                 "Only NYX quantiles are allowed; labels and Storm are forbidden")
        values = supplied[["q10", "q50", "q90"]].astype(float).reindex(index)
        origin = forecast_origins[country]
        _require(isinstance(origin, pd.Series), "Forecast origin Series required")
        _index(origin.index, f"NYX origin {country}")
        stamp = _times(origin, f"NYX origin {country}").reindex(index)
        known = stamp.notna() & (stamp <= cutoff)
        ordered = np.isfinite(values.to_numpy()).all(axis=1) & (np.diff(values.to_numpy(), axis=1) >= 0).all(axis=1)
        predictions[country], origins[country], eligible[country] = values, stamp, known & ordered
        diagnostics[country] = {"missing_origin_hours": int(stamp.isna().sum()),
                                "late_origin_hours": int((stamp > cutoff).sum()),
                                "invalid_or_missing_quantile_hours": int((~ordered).sum())}

    frame = pd.DataFrame(index=index)
    definitions, traces = {}, {}

    def add(name, values, valid, available_by, source_time, kind, **details):
        values = pd.Series(values, index=index, dtype=float)
        valid = pd.Series(valid, index=index).fillna(False).astype(bool) & np.isfinite(values)
        frame[name] = values.where(valid)
        flag = name + "__available"
        frame[flag] = valid.astype("int8")
        definitions[name] = {"availability_column": flag, "source_kind": kind, **details}
        traces[name] = {"valid": valid, "available_by": available_by.where(valid), "source_time": source_time.where(valid)}

    for country in ZONES:
        if country == zone:
            continue
        point = predictions[country].q50
        add(f"extra_neighbor_{country}_point", point, eligible[country], origins[country], origins[country],
            "current_nyx_forecast", country=country, delivery_alignment="same physical UTC hour")
        add(f"extra_neighbor_{country}_spread_vs_own", point-predictions[zone].q50,
            eligible[country] & eligible[zone], _latest(origins[country], origins[zone]),
            _latest(origins[country], origins[zone]), "current_nyx_forecast_difference", country=country, own_country=zone)
        add(f"extra_neighbor_{country}_width80", predictions[country].q90-predictions[country].q10,
            eligible[country], origins[country], origins[country], "current_nyx_interval_width", country=country,
            definition="NYX q90 minus q10; no new uncertainty interval is inferred")
    local = index.tz_convert(timezone)
    angle = 2*np.pi*(local.dayofyear.to_numpy()-1)/np.where(local.is_leap_year, 366., 365.)
    for name, values in (("extra_annual_sin", np.sin(angle)), ("extra_annual_cos", np.cos(angle))):
        add(name, values, True, cutoff, cutoff, "known_calendar",
            formula="2*pi*(local day_of_year-1)/(366 if leap year else 365)")

    normalized = normalize_fuel(fuel)
    current = normalized.reindex(index)  # deliberately no ffill, asof, or interpolation
    known = pd.Series(True, index=index)
    fuel_diagnostics = {"missing_delivery_rows": int((~index.isin(normalized.index)).sum())}
    for name in FUEL_TIME_COLUMNS[1:]:
        fuel_diagnostics[f"{name}_missing_hours"] = int(current[name].isna().sum())
        fuel_diagnostics[f"{name}_late_hours"] = int((current[name] > cutoff).sum())
        known &= current[name].notna() & (current[name] <= cutoff)
    available = _latest(*(current[name] for name in FUEL_TIME_COLUMNS[1:]))
    for name in FUEL_COLUMNS:
        source = "ttf_source_value_time_utc" if name.startswith("ttf_") else (
            "eua_source_value_time_utc" if name.startswith("eua_") else "market_source_value_time_utc")
        add("extra_fuel_"+name, current[name], known, available, current[source], "cached_asof_fuel",
            source_column=name, source_timestamp=source, delivery_alignment="exact value_time_utc only")
    for commodity in ("ttf", "eua"):
        source = current[f"{commodity}_source_value_time_utc"]
        age = (cutoff-source).dt.total_seconds()/3600.
        add(f"extra_fuel_{commodity}_quote_age_hours", age, known & (age >= 0), available, source,
            "cached_asof_fuel_quote_age", formula="(civil D-1 08 cutoff minus quote source timestamp)/1 hour")

    days = np.asarray(local.date)
    daily_traces = {name: pd.DataFrame({"available_hours": trace["valid"].groupby(days).sum(),
                    "latest_available_by_utc": trace["available_by"].groupby(days).max(),
                    "latest_source_event_utc": trace["source_time"].groupby(days).max()})
                    for name, trace in traces.items()}
    daily = []
    for day in dict.fromkeys(days):
        rows = days == day
        record = {"delivery_day": str(day), "forecast_cutoff_utc": cutoff.loc[rows].iloc[0].isoformat(),
                  "delivery_hours": int(rows.sum()), "features": {}}
        for name, table in daily_traces.items():
            item = table.loc[day]
            count = int(item["available_hours"])
            record["features"][name] = {"available_hours": count,
                "latest_available_by_utc": item["latest_available_by_utc"].isoformat() if count else None,
                "latest_source_event_utc": item["latest_source_event_utc"].isoformat() if count else None}
        daily.append(record)
    _require(len(frame.columns) == 42 and not np.isinf(frame.to_numpy(dtype=float)).any(), "Invalid extra feature matrix")
    audit = {"protocol": "nyx_neighbor_season_fuel_features_v1", "zone": zone, "timezone": timezone,
             "shape": list(frame.shape), "numeric_features": 21, "availability_features": 21,
             "first_delivery_utc": index[0].isoformat(), "last_delivery_utc": index[-1].isoformat(),
             "Storm_used_as_input": False, "current_neighbor_labels_used": False, "model_fits": 0,
             "imputation_performed": False, "interpolation_performed": False, "forward_fill_performed": False,
             "cutoff_rule": "each delivery day's own civil D-1 08:00; not weekly model origin",
             "forecast_origin_semantics": "logical replay origins checked; not newly certified provider vintages",
             "fuel_timestamp_semantics": "historical cache as-of query timestamps; provider insertion timestamps unavailable",
             "fuel_delivery_alignment": "value_time_utc is the future delivery key, not an observation availability time",
             "fuel_last_delivery_utc": normalized.index[-1].isoformat(), "nyx_eligibility": diagnostics,
             "fuel_eligibility": fuel_diagnostics, "feature_definitions": definitions, "daily_sources": daily}
    return frame, audit
