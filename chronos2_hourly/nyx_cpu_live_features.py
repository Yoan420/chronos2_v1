"""Causal price features for a new, CPU-only NYX recipe.

The source tables are supplied by the caller.  In production they are the
audited Saturn residual-load and nuclear banks used by the nuclear pipeline,
plus the four observed day-ahead price histories.  Nothing in this module
reads a research run, fetches a service, fits a model, or uses a future price.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

import numpy as np
import pandas as pd


ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
RESIDUAL_COLUMNS = tuple(f"{zone.lower()}_residual_load_fcst"
                         for zone in (*ZONES, "ES"))
PRICE_LAGS = (1, 2, 7)
HOUR_LAGS = (1, 7)
CALENDAR_COLUMNS = (
    "calendar_hour_sin", "calendar_hour_cos", "calendar_weekday_sin",
    "calendar_weekday_cos", "calendar_year_sin", "calendar_year_cos",
    "calendar_is_weekend", "calendar_dst_fold", "calendar_utc_offset_hours",
)


def _feature_columns() -> tuple[str, ...]:
    columns = []
    for source_zone in ZONES:
        prefix = f"price_{source_zone.lower()}"
        values = [*(f"{prefix}_d{lag}_mean" for lag in PRICE_LAGS),
                  f"{prefix}_d1_min", f"{prefix}_d1_max",
                  *(f"{prefix}_d{lag}_hour" for lag in HOUR_LAGS)]
        for name in values:
            columns.extend((name, name + "__available"))
    for name in (*RESIDUAL_COLUMNS, "fr_nuclear_generation_fcst_gw"):
        columns.extend((name, name + "__available"))
    columns.extend(CALENDAR_COLUMNS)
    return tuple(columns)


FEATURE_COLUMNS = _feature_columns()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_index(index: object, name: str) -> pd.DatetimeIndex:
    _require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC",
             f"{name}: UTC DatetimeIndex required")
    result = index
    _require(not result.empty and not result.hasnans and result.is_unique
             and result.is_monotonic_increasing and result.equals(result.floor("h")),
             f"{name}: unique ordered physical UTC hours required")
    return result


def _full_civil_days(index: pd.DatetimeIndex, timezone: str) -> tuple[object, object]:
    local = index.tz_convert(timezone)
    first, last = local[0].date(), local[-1].date()
    expected = pd.date_range(str(first), str(last + timedelta(days=1)),
                             tz=timezone, freq="h", inclusive="left").tz_convert("UTC")
    _require(index.equals(expected), "delivery_index: complete consecutive civil days required")
    return first, last


def _cutoffs(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    days = index.tz_convert(timezone).date
    distinct = {day: pd.Timestamp(f"{day - timedelta(days=1)} 08:00", tz=timezone)
                .tz_convert("UTC") for day in set(days)}
    return pd.DatetimeIndex([distinct[day] for day in days])


def _utc_column(frame: pd.DataFrame, column: str, source: str) -> pd.DatetimeIndex:
    _require(column in frame and isinstance(frame[column].dtype, pd.DatetimeTZDtype)
             and str(frame[column].dt.tz) == "UTC",
             f"{source}: {column} must be UTC")
    return pd.DatetimeIndex(frame[column])


def _saturn_rows(frame: pd.DataFrame, index: pd.DatetimeIndex,
                 cutoffs: pd.DatetimeIndex, *, source: str,
                 value_columns: tuple[str, ...], exact_cutoff: bool) -> pd.DataFrame:
    _require(isinstance(frame, pd.DataFrame) and not frame.empty
             and frame.columns.is_unique and set(value_columns).issubset(frame.columns),
             f"{source}: required Saturn columns missing")
    times = _utc_column(frame, "value_time_utc", source)
    _require(not times.hasnans and times.is_unique and times.is_monotonic_increasing
             and times.equals(times.floor("h")),
             f"{source}: source hours must be ordered, unique and physical")
    # Reindex the original typed columns so datetime and numeric dtypes survive.
    available = frame.copy()
    available.index = times
    selected = available.reindex(index)
    _require(selected["value_time_utc"].notna().all(),
             f"{source}: a required historical or future hour is absent")
    for column in ("snapshot_time_utc", "revision_time_utc"):
        stamps = _utc_column(selected, column, source)
        _require(not stamps.hasnans and (stamps <= cutoffs).all(),
                 f"{source}: missing or post-cutoff {column}")
    if exact_cutoff:
        stamps = _utc_column(selected, "cutoff_time_utc", source)
        _require(not stamps.hasnans and stamps.equals(cutoffs),
                 f"{source}: wrong civil D-1 08:00 snapshot cutoff")
    numeric = selected.loc[:, value_columns].apply(pd.to_numeric, errors="raise")
    _require(np.isfinite(numeric.to_numpy(dtype=float)).all(),
             f"{source}: a required forecast value is missing or nonfinite")
    return numeric.astype(float)


def _prices(prices: Mapping[str, pd.Series], zone: str,
            first_day: object, last_day: object) -> dict[str, pd.Series]:
    _require(isinstance(prices, Mapping) and set(prices) == set(ZONES),
             "prices: exactly FR, DE, BE and NL histories required")
    timezone = TIMEZONES[zone]
    first_needed = first_day - timedelta(days=max(PRICE_LAGS))
    last_needed = last_day - timedelta(days=1)
    required = pd.date_range(str(first_needed), str(last_needed + timedelta(days=1)),
                             tz=timezone, freq="h", inclusive="left").tz_convert("UTC")
    result = {}
    for country in ZONES:
        source = prices[country]
        _require(isinstance(source, pd.Series), f"prices[{country}]: Series required")
        original = _utc_index(source.index, f"prices[{country}]")
        numeric = pd.to_numeric(source, errors="raise")
        series = pd.Series(numeric.to_numpy(dtype=float), index=original)
        selected = series.reindex(required)
        _require(np.isfinite(selected.to_numpy(dtype=float)).all(),
                 f"prices[{country}]: incomplete D-1 through D-7 auction-price history")
        result[country] = selected
    return result


def _calendar(index: pd.DatetimeIndex, timezone: str) -> pd.DataFrame:
    local = index.tz_convert(timezone)
    year_days = np.where(local.is_leap_year, 366., 365.)
    hour = local.hour.to_numpy(dtype=float)
    weekday = local.dayofweek.to_numpy(dtype=float)
    year_phase = (local.dayofyear.to_numpy(dtype=float) - 1. + hour / 24.) / year_days
    stamps = local.to_pydatetime()
    data = {
        "calendar_hour_sin": np.sin(2 * np.pi * hour / 24.),
        "calendar_hour_cos": np.cos(2 * np.pi * hour / 24.),
        "calendar_weekday_sin": np.sin(2 * np.pi * weekday / 7.),
        "calendar_weekday_cos": np.cos(2 * np.pi * weekday / 7.),
        "calendar_year_sin": np.sin(2 * np.pi * year_phase),
        "calendar_year_cos": np.cos(2 * np.pi * year_phase),
        "calendar_is_weekend": (weekday >= 5).astype(float),
        "calendar_dst_fold": np.array([stamp.fold for stamp in stamps], dtype=float),
        "calendar_utc_offset_hours": np.array(
            [stamp.utcoffset().total_seconds() / 3600. for stamp in stamps], dtype=float),
    }
    return pd.DataFrame(data, index=index).loc[:, CALENDAR_COLUMNS]


def build_country_features(
    *, zone: str, delivery_index: pd.DatetimeIndex,
    prices: Mapping[str, pd.Series], residual_bank: pd.DataFrame,
    nuclear_store: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build one country matrix for training days and a later delivery day.

    Every Saturn hour must exist and have snapshot/revision timestamps no later
    than that *hour's delivery day* D-1 08:00 civil cutoff.  Every required past
    auction-price day must be complete.  The only allowed missing feature is a
    same-civil-hour lag when that hour did not exist on a spring DST day; its
    availability flag is zero and the value remains NaN for the estimator.
    """
    _require(zone in ZONES, "Unsupported country")
    index = _utc_index(delivery_index, "delivery_index")
    timezone = TIMEZONES[zone]
    first, last = _full_civil_days(index, timezone)
    cutoffs = _cutoffs(index, timezone)
    source_prices = _prices(prices, zone, first, last)
    residual = _saturn_rows(residual_bank, index, cutoffs,
                            source="residual_bank", value_columns=RESIDUAL_COLUMNS,
                            exact_cutoff=True)
    nuclear = _saturn_rows(nuclear_store, index, cutoffs,
                           source="nuclear_store", value_columns=("value",),
                           exact_cutoff=False)
    _require((nuclear["value"] >= 0).all(), "nuclear_store: negative generation forecast")
    local = index.tz_convert(timezone)
    days = local.date
    hours = local.hour
    result = pd.DataFrame(index=index)
    missing_dst = {}
    for country in ZONES:
        source = source_prices[country]
        source_local = source.index.tz_convert(timezone)
        daily = pd.DataFrame({"day": source_local.date, "hour": source_local.hour,
                              "value": source.to_numpy(dtype=float)})
        aggregate = daily.groupby("day", sort=True).value.agg(["mean", "min", "max"])
        hourly = daily.groupby(["day", "hour"], sort=True).value.mean()
        prefix = f"price_{country.lower()}"

        def add(name: str, values: np.ndarray) -> None:
            numbers = np.asarray(values, dtype=float)
            _require(not np.isinf(numbers).any(), f"{name}: infinite feature")
            result[name] = numbers
            result[name + "__available"] = np.isfinite(numbers).astype(float)

        for lag in PRICE_LAGS:
            keys = [day - timedelta(days=lag) for day in days]
            add(f"{prefix}_d{lag}_mean", aggregate["mean"].reindex(keys).to_numpy(dtype=float))
            if lag == 1:
                for stat in ("min", "max"):
                    add(f"{prefix}_d1_{stat}", aggregate[stat].reindex(keys).to_numpy(dtype=float))
        for lag in HOUR_LAGS:
            keys = pd.MultiIndex.from_arrays(
                [[day - timedelta(days=lag) for day in days], hours],
                names=["day", "hour"],
            )
            name = f"{prefix}_d{lag}_hour"
            values = hourly.reindex(keys).to_numpy(dtype=float)
            add(name, values)
            missing_dst[name] = int(np.isnan(values).sum())
    for name in RESIDUAL_COLUMNS:
        result[name] = residual[name].to_numpy(dtype=float)
        result[name + "__available"] = 1.
    result["fr_nuclear_generation_fcst_gw"] = nuclear["value"].to_numpy(dtype=float)
    result["fr_nuclear_generation_fcst_gw__available"] = 1.
    result = pd.concat([result, _calendar(index, timezone)], axis=1)
    _require(len(result.columns) == len(FEATURE_COLUMNS)
             and set(result.columns) == set(FEATURE_COLUMNS)
             and not np.isinf(result.to_numpy(float)).any(),
             "Deterministic CPU feature schema or numeric contract changed")
    result = result.loc[:, FEATURE_COLUMNS]
    flags = result.loc[:, [name for name in result if name.endswith("__available")]]
    _require(flags.isin((0., 1.)).all().all(), "Nonbinary feature availability")
    audit = {
        "protocol": "nyx_saturn_cpu_live_features_v1", "zone": zone,
        "feature_columns": list(FEATURE_COLUMNS), "rows": len(index),
        "first_delivery_day": str(first), "last_delivery_day": str(last),
        "first_cutoff_utc": cutoffs[0].isoformat(),
        "last_cutoff_utc": cutoffs[-1].isoformat(),
        "all_saturn_hours_present_and_asof_cutoff": True,
        "all_required_price_days_complete": True,
        "civil_hour_lag_missing_due_to_spring_dst": missing_dst,
        "target_day_prices_or_storm_used_as_features": False,
        "provider_publication_vintages_certified": False,
        "saturn_snapshot_semantics": "query as of civil D-1 08:00, not provider publication evidence",
    }
    return result, audit


__all__ = ["FEATURE_COLUMNS", "RESIDUAL_COLUMNS", "ZONES", "build_country_features"]
