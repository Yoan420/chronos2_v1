"""Deterministic national calendars for a separate pooled CWE feature recipe.

This module reads no prices, labels, forecasts, or external files. It leaves the
frozen 334-column matrices untouched. Dates refer to local civil delivery days;
the output retains the supplied physical UTC index, including both DST folds.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json

import holidays
import numpy as np
import pandas as pd

PROTOCOL = "nyx_pooled_national_calendar_v1"
ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
COLUMNS = tuple(f"pooled_cal_holiday_{zone.lower()}" for zone in ZONES) + (
    "pooled_cal_own_holiday", "pooled_cal_own_business_day",
    "pooled_cal_own_before_holiday", "pooled_cal_own_after_holiday",
    "pooled_cal_own_bridge_day",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def build_pooled_calendar_features(*, zone, delivery_index):
    """Return nine float64 columns and a reproducible calendar audit.

    All four national holiday indicators are retained in every country's rows,
    so a neighbour's holiday can be distinguished from an own-country holiday.
    Only national public holidays are included (no subdivision or school dates).
    Adjacent-day flags use calendar dates rather than +/- 24 physical hours.
    """
    if zone not in ZONES:
        raise ValueError("A declared CWE country is required")
    index = delivery_index
    if (not isinstance(index, pd.DatetimeIndex) or not len(index) or str(index.tz) != "UTC"
            or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.floor("h"))):
        raise ValueError("A nonempty ordered unique physical UTC hourly index is required")
    local = index.tz_convert(TIMEZONES[zone])
    days = list(local.date)
    years = list(range(min(day.year for day in days)-1, max(day.year for day in days)+2))
    sets = {
        country: set(holidays.country_holidays(country, years=years, subdiv=None,
                    observed=True, expand=False, language="en_US").keys())
        for country in ZONES
    }
    own = sets[zone]
    previous = [day-timedelta(days=1) for day in days]
    following = [day+timedelta(days=1) for day in days]
    is_holiday = np.asarray([day in own for day in days], dtype=bool)
    previous_holiday = np.asarray([day in own for day in previous], dtype=bool)
    next_holiday = np.asarray([day in own for day in following], dtype=bool)
    weekday = np.asarray([day.weekday() < 5 for day in days], dtype=bool)
    previous_weekend = np.asarray([day.weekday() >= 5 for day in previous], dtype=bool)
    next_weekend = np.asarray([day.weekday() >= 5 for day in following], dtype=bool)
    values = {f"pooled_cal_holiday_{country.lower()}": [day in sets[country] for day in days]
              for country in ZONES}
    values.update({
        "pooled_cal_own_holiday": is_holiday,
        "pooled_cal_own_business_day": weekday & ~is_holiday,
        "pooled_cal_own_before_holiday": next_holiday,
        "pooled_cal_own_after_holiday": previous_holiday,
        "pooled_cal_own_bridge_day": weekday & ~is_holiday & (
            (previous_holiday & next_weekend) | (next_holiday & previous_weekend)),
    })
    frame = pd.DataFrame(values, index=index, columns=COLUMNS, dtype=np.float64)
    calendar = {
        "holidays_version": holidays.__version__, "countries": list(ZONES),
        "years": years, "subdiv": None, "observed": True,
        "categories": "PUBLIC (package default)", "language": "en_US",
        "national_dates": {country: sorted(day.isoformat() for day in sets[country])
                           for country in ZONES},
    }
    audit = {
        "protocol": PROTOCOL, "zone": zone, "timezone": TIMEZONES[zone],
        "columns": list(COLUMNS), "dtype": "float64", "hours": len(index),
        "first_timestamp_utc": index[0].isoformat(), "last_timestamp_utc": index[-1].isoformat(),
        "first_local_day": days[0].isoformat(), "last_local_day": days[-1].isoformat(),
        "calendar": calendar, "calendar_sha256": _digest(calendar),
        "feature_values_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
            + json.dumps(list(frame.columns)).encode()).hexdigest(),
        "only_calendar_inputs": True, "all_values_available_from_calendar": True,
        "Storm_used_as_input": False, "observation_inputs": False,
        "calendar_scope": "National public holidays; no regional or school holidays",
        "availability_basis": "Deterministic calendar rules; historical announcement vintages are not modeled",
        "bridge_definition": "Non-holiday weekday between a holiday and an adjacent weekend",
    }
    return frame, audit
