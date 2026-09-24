"""Frozen VPS cashflows and an explicitly approximate, forecast-only NYX replica.

No network or forecast changes. This compatibility diagnostic consumes one MWh
of starting inventory daily; it must not be presented as self-financing profit.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from functools import lru_cache
import math
from numbers import Real

import numpy as np
import pandas as pd


CONTRACT = "saturn_vps_empirical_soc2to1_v1"
METHODOLOGY = {
    "contract": CONTRACT,
    "official_algorithm_verified": False,
    "storm": "Published Saturn hourly VPS cashflows summed / calendar days, including missing days in the denominator.",
    "model": "Forecast-only 1 MW / 4 MWh MILP; charge efficiency .85, discharge efficiency 1; initial SOC 2, final SOC 1 MWh; first hour sells 1 MWh.",
    "settlement": "Round each hourly net position times unrounded observed price to cents, then sum / calendar days.",
    "support": "Model shown only if every day containing a finite Storm cashflow has complete cashflows and an estimable Model schedule.",
    "limits": "First-hour forecast <= 0 is unvalidated and excluded. 23/25-hour DST days are a physical extension, not empirically reconciled. No fees or degradation. The daily inventory drawdown is not self-financing profit.",
    "empirical_check": "227 Storm days: mean absolute daily gap 2.71 EUR; mean absolute gap of 16 rolling cells 0.92 EUR/day, max 2.29 EUR/day. In-sample reconstruction, not an accuracy guarantee for NYX or future dates.",
}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


@lru_cache(maxsize=3)
def _problem(n):
    from scipy.optimize import Bounds, LinearConstraint

    if n not in (23, 24, 25):
        raise ValueError("VPS requires a complete physical civil day.")
    size, state, mode = 4 * n + 1, 2 * n, 3 * n + 1
    lower, upper = np.zeros(size), np.ones(size)
    upper[state:state + n + 1] = 4
    lower[state] = upper[state] = 2
    lower[state + n] = upper[state + n] = 1
    lower[0] = upper[0] = 0
    lower[n] = upper[n] = 1
    matrix = np.zeros((3 * n, size))
    lo, hi = np.full(3 * n, -np.inf), np.zeros(3 * n)
    for hour in range(n):
        matrix[hour, state + hour + 1] = 1
        matrix[hour, state + hour] = -1
        matrix[hour, hour] = -.85
        matrix[hour, n + hour] = 1
        lo[hour] = 0
        matrix[n + hour, hour] = 1
        matrix[n + hour, mode + hour] = -1
        matrix[2 * n + hour, n + hour] = 1
        matrix[2 * n + hour, mode + hour] = 1
        hi[2 * n + hour] = 1
    integrality = np.zeros(size)
    integrality[mode:] = 1
    return Bounds(lower, upper), LinearConstraint(matrix, lo, hi), integrality


@lru_cache(maxsize=2048)
def _schedule(prices):
    n = len(prices)
    try:
        from scipy.optimize import milp

        bounds, constraints, integrality = _problem(n)
        objective = np.zeros(4 * n + 1)
        objective[:n], objective[n:2 * n] = prices, -np.asarray(prices)
        solved = milp(objective, bounds=bounds, constraints=constraints, integrality=integrality,
                      options={"time_limit": 2.0, "mip_rel_gap": 0.0})
    except (ImportError, ValueError, RuntimeError):
        return None
    if not solved.success or solved.status != 0 or solved.x is None:
        return None
    c, d, s = solved.x[:n], solved.x[n:2*n], solved.x[2*n:3*n+1]
    tol = 2e-6
    if (not np.isfinite(solved.x).all() or np.min(c) < -tol or np.max(c) > 1 + tol
            or np.min(d) < -tol or np.max(d) > 1 + tol or np.min(s) < -tol or np.max(s) > 4 + tol
            or abs(s[0] - 2) > tol or abs(s[-1] - 1) > tol or abs(c[0]) > tol or abs(d[0] - 1) > tol
            or np.any((c > tol) & (d > tol)) or np.max(np.abs(np.diff(s) - .85*c + d)) > tol):
        return None
    return tuple(c), tuple(d), tuple(s)


def optimize_vps_schedule(forecast):
    """Fresh forecast-only plan; None for an unvalidated first-hour sign/failed solve."""
    values = list(forecast)
    if len(values) not in (23, 24, 25) or not all(_finite(v) for v in values):
        raise ValueError("VPS requires 23, 24 or 25 finite hourly forecasts.")
    if values[0] <= 0:
        return None
    prices = tuple(float(v) for v in values)
    plan = _schedule(prices)
    if plan is None:
        return None
    charge, discharge, state = plan
    return {"charge_mw": list(charge), "discharge_mw": list(discharge), "soc_mwh": list(state)}


def _stamp(value):
    stamp = pd.Timestamp(value)
    if (pd.isna(stamp) or stamp.tzinfo is None or stamp.minute or stamp.second
            or stamp.microsecond or stamp.nanosecond):
        raise ValueError("Explicit physical hourly timestamp required.")
    return stamp.tz_convert("UTC")


def _hours(day, timezone):
    return pd.date_range(pd.Timestamp(day, tz=timezone),
                         pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def build_vps_windows(zone, report_day, windows):
    """Keep published Storm independent of NYX; never compare different day sets."""
    end = date.fromisoformat(report_day)
    timezone = zone["timezone"]
    history = zone.get("vps_history", {})
    source = history.get("source", {})
    expected_series = f"power.vps.{zone['zone'].lower()}.euromwh.h.da.pnl.storm"
    verified = (source.get("status") in ("complete", "partial")
                and source.get("series") == expected_series and source.get("zone") == zone["zone"]
                and source.get("timezone") == timezone)
    cashflows, seen = {}, set()
    if verified:
        try:
            for row in history.get("rows", []):
                stamp = _stamp(row.get("timestamp_utc"))
                if stamp in seen:
                    raise ValueError("Duplicate VPS hour.")
                seen.add(stamp)
                value = row.get("pnl")
                if value is not None and not _finite(value):
                    raise ValueError("Invalid VPS cashflow.")
                day = stamp.tz_convert(timezone).date().isoformat()
                if day <= report_day and value is not None:
                    cashflows[stamp] = float(value)
        except (ValueError, TypeError, OverflowError):
            verified, cashflows = False, {}
    # Remove every occurrence of duplicate price hours, rather than keep an
    # arbitrary forecast. Missing Model data must not alter published Storm.
    prices, duplicates = {}, set()
    for row in zone.get("rolling_history", {}).get("rows", []):
        try:
            stamp = _stamp(row.get("timestamp_utc"))
        except (ValueError, TypeError, OverflowError):
            continue
        if stamp in prices or stamp in duplicates:
            prices.pop(stamp, None)
            duplicates.add(stamp)
        else:
            prices[stamp] = row
    source_days = sorted({stamp.tz_convert(timezone).date().isoformat() for stamp in cashflows})
    earliest = (end - timedelta(days=max(windows) - 1)).isoformat()
    daily, failures = {}, {}
    for day in source_days:
        if day < earliest:
            continue
        hours = _hours(day, timezone)
        if not all(stamp in cashflows for stamp in hours):
            failures[day] = "partial_storm_cashflows"
            continue
        if not all(all(_finite(prices.get(stamp, {}).get(key)) for key in ("observed", "model")) for stamp in hours):
            failures[day] = "missing_model_or_observed"
            continue
        forecast = [prices[stamp]["model"] for stamp in hours]
        if forecast[0] <= 0:
            failures[day] = "unvalidated_first_hour_sign"
            continue
        plan = optimize_vps_schedule(forecast)
        if plan is None:
            failures[day] = "solver_unavailable"
            continue
        actual = [prices[stamp]["observed"] for stamp in hours]
        daily[day] = float(np.round((np.asarray(plan["discharge_mw"]) - plan["charge_mw"]) * actual, 2).sum())
    result = {}
    for window in windows:
        start = (end - timedelta(days=window - 1)).isoformat()
        query_covers_window = True
        if source.get("window_start_utc") or source.get("window_end_utc"):
            try:
                query_covers_window = (_stamp(source["window_start_utc"]) <= _hours(start, timezone)[0]
                    and _stamp(source["window_end_utc"]) >= _hours(report_day, timezone)[-1])
            except (ValueError, TypeError, KeyError, OverflowError):
                query_covers_window = False
        dates = [day for day in source_days if start <= day <= report_day]
        selected = [value for stamp, value in cashflows.items()
                    if start <= stamp.tz_convert(timezone).date().isoformat() <= report_day]
        unavailable = {day: failures[day] for day in dates if day in failures}
        same_support = query_covers_window and bool(dates) and all(day in daily for day in dates)
        result[str(window)] = {
            "contract": CONTRACT, "source": source, "source_verified": verified,
            "query_covers_window": query_covers_window,
            "calendar_days": window, "published_days": len(dates), "published_hours": len(selected),
            "estimated_days": sum(day in daily for day in dates),
            "missing_publication_days": window - len(dates), "same_support": same_support,
            "excluded_days": unavailable, "exclusion_counts": dict(Counter(unavailable.values())),
            "dst_extension_days": sum(len(_hours(day, timezone)) != 24 for day in dates),
            "storm_daily_pnl": math.fsum(selected) / window if selected and query_covers_window else None,
            "model_daily_pnl": math.fsum(daily[day] for day in dates) / window if same_support else None,
            "official_algorithm_verified": False,
        }
    return result
