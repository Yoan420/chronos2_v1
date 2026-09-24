"""Past-error Storm intervals for the paper's quantile-based P&L diagnostic.

Storm only supplies a point forecast.  These are estimated empirical intervals,
not native Storm quantiles and not a claim of nominal coverage.  Subtracting the
historical median error preserves the exact Storm point used by both strategies.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "storm_past_error_centered_intervals_v1"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _physical_hours(day: date, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day).tz_localize(timezone)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def calibrate_storm_quantiles(
    rows: Mapping[pd.Timestamp, Mapping[str, Any]],
    *,
    timezone: str,
    lookback_days: int = 365,
    min_history_days: int = 60,
    min_hour_samples: int = 30,
) -> tuple[dict[pd.Timestamp, dict[str, float | None]], dict[str, Any]]:
    """Estimate q10/q90 per local hour using complete days strictly before D.

    Training is restricted to the calendar interval [D-lookback_days, D), not
    the last N available dates.  A training day needs every physical hourly
    timestamp and finite ``observed`` and ``storm`` values.  Repeated autumn
    hours are two distinct observations of the same local hour; spring days
    have 23 observations.  The current day's observed values are never needed.

    Returned values are ``storm + Qp(error) - Q50(error)`` for p=.1 and .9.
    Insufficient history or a missing Storm point yields None, never an invented
    degenerate interval.  Inputs are not mutated.  Timestamps must be aware and
    are normalized to UTC in the output.
    """
    for name, value in (
        ("lookback_days", lookback_days),
        ("min_history_days", min_history_days),
        ("min_hour_samples", min_hour_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # Validate the timezone even when rows is empty.
    pd.Timestamp("2000-01-01").tz_localize(timezone)

    normalized: dict[pd.Timestamp, Mapping[str, Any]] = {}
    by_day: dict[date, list[pd.Timestamp]] = defaultdict(list)
    local_hour: dict[pd.Timestamp, int] = {}
    for raw_timestamp, row in rows.items():
        timestamp = pd.Timestamp(raw_timestamp)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("Storm calibration timestamps must be timezone-aware")
        timestamp = timestamp.tz_convert("UTC")
        if timestamp in normalized:
            raise ValueError("Storm calibration has duplicate physical timestamps")
        if not isinstance(row, Mapping):
            raise ValueError("Storm calibration rows must be mappings")
        normalized[timestamp] = row
        local = timestamp.tz_convert(timezone)
        by_day[local.date()].append(timestamp)
        local_hour[timestamp] = local.hour
    for timestamps in by_day.values():
        timestamps.sort()

    complete_errors: dict[date, list[tuple[int, float]]] = {}
    incomplete_days: list[str] = []
    for day in sorted(by_day):
        timestamps = by_day[day]
        expected = _physical_hours(day, timezone)
        if timestamps != list(expected):
            incomplete_days.append(day.isoformat())
            continue
        errors: list[tuple[int, float]] = []
        for timestamp in timestamps:
            row = normalized[timestamp]
            actual = _number(row.get("observed"))
            point = _number(row.get("storm"))
            if actual is None or point is None:
                break
            error = actual - point
            if not math.isfinite(error):
                break
            errors.append((local_hour[timestamp], error))
        if len(errors) == len(expected):
            complete_errors[day] = errors
        else:
            incomplete_days.append(day.isoformat())

    updates: dict[pd.Timestamp, dict[str, float | None]] = {}
    daily_audit: list[dict[str, Any]] = []
    complete_dates = sorted(complete_errors)
    for day in sorted(by_day):
        lower = day - timedelta(days=int(lookback_days))
        history = [past for past in complete_dates if lower <= past < day]
        errors_by_hour: dict[int, list[float]] = defaultdict(list)
        for past in history:
            for hour, error in complete_errors[past]:
                errors_by_hour[hour].append(error)
        counts = {str(hour): len(errors_by_hour[hour]) for hour in range(24)}
        enough_days = len(history) >= min_history_days
        spreads: dict[int, tuple[float, float]] = {}
        if enough_days:
            for hour in range(24):
                if len(errors_by_hour[hour]) < min_hour_samples:
                    continue
                p10, median, p90 = np.quantile(
                    errors_by_hour[hour], [.1, .5, .9], method="linear"
                )
                spreads[hour] = (float(p10 - median), float(p90 - median))

        available = 0
        missing_points = 0
        missing_hour_history = 0
        for timestamp in by_day[day]:
            point = _number(normalized[timestamp].get("storm"))
            spread = spreads.get(local_hour[timestamp])
            result: dict[str, float | None] = {"storm_p10": None, "storm_p90": None}
            if point is None:
                missing_points += 1
            elif enough_days and spread is None:
                missing_hour_history += 1
            elif spread is not None:
                p10, p90 = point + spread[0], point + spread[1]
                if math.isfinite(p10) and math.isfinite(p90):
                    result = {"storm_p10": p10, "storm_p90": p90}
                    available += 1
            updates[timestamp] = result

        if not enough_days:
            status = "warmup"
        elif missing_hour_history:
            status = "insufficient_hour_samples"
        elif available != len(by_day[day]):
            status = "missing_storm_or_nonfinite_interval"
        else:
            status = "available"
        daily_audit.append({
            "delivery_day": day.isoformat(),
            "training_window_start": lower.isoformat(),
            "training_window_end_exclusive": day.isoformat(),
            "train_first_day": history[0].isoformat() if history else None,
            "train_last_day": history[-1].isoformat() if history else None,
            "history_days": len(history),
            "hour_samples": counts,
            "forecast_hours": len(by_day[day]),
            "available_hours": available,
            "missing_storm_hours": missing_points,
            "status": status,
        })

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "label": "Storm calibré",
        "method": "past complete-day errors per local hour; empirical intervals centered on Storm",
        "formula": "storm_pXX = storm + QXX(observed - storm) - Q50(observed - storm)",
        "native_storm_quantiles": False,
        "nominal_coverage_verified": False,
        "storm_point_unchanged": True,
        "timezone": timezone,
        "lookback_days": int(lookback_days),
        "min_history_days": int(min_history_days),
        "min_hour_samples": int(min_hour_samples),
        "quantile_method": "linear",
        "quantile_levels": [.1, .9],
        "training_day_rule": "complete physical hourly grid with finite observed and storm; D excluded",
        "incomplete_training_days": incomplete_days,
        "days": daily_audit,
    }
    return updates, audit
