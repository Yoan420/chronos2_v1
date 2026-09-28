"""Cached daily Pmax descriptors, with explicit missingness and vintage limits.

These are partial provider fleet capacities in GW, not generation or national
reserves. Collector query cutoffs are not evidence of provider publication time.
"""
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

ZONES = ("FR", "DE", "BE", "NL")
TECHNOLOGIES = ("ccgt", "gt", "coal", "lignite", "nuclear")
SOURCES = tuple(sorted(("fr_ccgt", "fr_gt", "de_ccgt", "de_gt", "de_coal", "de_lignite",
                        "be_ccgt", "be_gt", "be_nuclear", "nl_ccgt", "nl_gt", "nl_coal", "nl_nuclear")))
LEVELS = tuple("thermal__" + name + "_pmax_gw" for name in SOURCES) + tuple(
    "thermal__own_" + tech + "_pmax_gw" for tech in TECHNOLOGIES)
COLUMNS = tuple(c for level in LEVELS for c in (level, level + "__available"))
CONTRACT = {
    "sources": list(SOURCES), "new_columns": list(COLUMNS), "unit": "GW",
    "daily_broadcast": True, "join": "exact physical UTC hour",
    "missing_value": "NaN", "missing_flag": 0, "known_zero_retained": True,
    "imputation": False, "aggregated_gas_used": False, "capacity_totals_computed": False,
    "national_fleet_coverage_certified": False, "provider_publication_timestamps_certified": False,
    "snapshot_semantics": "collector query as-of civil D-1 08:00 Europe/Paris, not publication evidence",
    "excluded": {"fr_coal": "No evaluation-year coverage; historical values all zero"},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_source(frame, name):
    require(name in SOURCES, "Unexpected thermal source")
    require(isinstance(frame, pd.DataFrame) and not frame.empty and frame.columns.is_unique,
            "Nonempty unique source columns required")
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc"}
    require(set(frame.columns) == required, "Thermal source schema changed")
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        require(isinstance(frame[column].dtype, pd.DatetimeTZDtype)
                and str(frame[column].dt.tz) == "UTC", "Explicit UTC timestamp dtype required")
    index = pd.DatetimeIndex(pd.to_datetime(frame.value_time_utc, utc=True)).as_unit("ns")
    require(index.is_unique and index.is_monotonic_increasing and not index.hasnans
            and index.equals(index.floor("h")), "Thermal source must have unique ordered UTC hours")
    values = frame.value.to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid capacity GW")
    local_days = index.tz_convert("Europe/Paris").date
    for day in np.unique(local_days):
        mask = local_days == day
        expected = pd.date_range(str(day), str(day + timedelta(days=1)), tz="Europe/Paris",
                                 freq="h", inclusive="left").tz_convert("UTC")
        require(index[mask].equals(expected) and np.unique(values[mask]).size == 1,
                "Daily Pmax must cover every physical hour of its civil day")
    cutoffs = pd.DatetimeIndex([
        datetime.combine(day - timedelta(days=1), time(8)) for day in local_days
    ]).tz_localize("Europe/Paris").tz_convert("UTC").as_unit("ns")
    for name in ("snapshot_time_utc", "revision_time_utc"):
        recorded = pd.DatetimeIndex(pd.to_datetime(frame[name], utc=True)).as_unit("ns")
        require(recorded.equals(cutoffs), "Collector nominal civil D-1 08:00 cutoffs differ")
    return pd.Series(values, index=index, name="pmax_gw")


def build_features(sources, index):
    require(set(sources) == set(SOURCES), "Exactly thirteen technology caches required")
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
            and index.is_unique and index.is_monotonic_increasing and not index.hasnans
            and index.equals(index.floor("h")), "Ordered physical UTC output grid required")
    aligned = {name: validate_source(sources[name], name).reindex(index) for name in SOURCES}
    global_data = {}
    for name in SOURCES:
        column = "thermal__" + name + "_pmax_gw"
        global_data[column] = aligned[name].to_numpy()
        global_data[column + "__available"] = aligned[name].notna().to_numpy(dtype=float)
    outputs = {}
    for zone in ZONES:
        data = dict(global_data)
        for tech in TECHNOLOGIES:
            column, source = "thermal__own_" + tech + "_pmax_gw", zone.lower() + "_" + tech
            values = aligned[source].to_numpy() if source in aligned else np.full(len(index), np.nan)
            data[column] = values
            data[column + "__available"] = np.isfinite(values).astype(float)
        outputs[zone] = pd.DataFrame(data, index=index.copy()).loc[:, COLUMNS]
    return outputs


def append_features(base, added):
    require(set(base) == set(added) == set(ZONES), "Exactly four country matrices required")
    result = {}
    for zone in ZONES:
        previous, extra = base[zone], added[zone]
        require(previous.columns.is_unique and not set(previous.columns).intersection(COLUMNS)
                and previous.index.equals(extra.index) and tuple(extra.columns) == COLUMNS,
                "Thermal append schema/index mismatch")
        result[zone] = pd.concat([previous, extra], axis=1)
        pd.testing.assert_frame_equal(result[zone].loc[:, previous.columns], previous, check_exact=True)
    return result
