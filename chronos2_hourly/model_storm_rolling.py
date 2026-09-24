"""Paired CWE rolling scores and the paper\'s two one-cycle battery strategies.

This module reads the frozen report payload only. No network access, forecast
training, imputation or operational model changes take place here.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
import math
from numbers import Real

import numpy as np
import pandas as pd


WINDOWS = (7, 30, 60, 90, 365)
FREQUENCIES = ("60min", "day")
PROVIDERS = (("storm", "Storm"), ("model", "Model"))
STORAGE_POWER_MW = 1.0
STORAGE_CAPACITY_MWH = 4.0
STORAGE_ROUND_TRIP_EFFICIENCY = 0.85


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _hours(day: str, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=timezone)
    end = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=timezone)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def score_forecast(observed, forecast) -> dict:
    """Score finite paired arrays; tolerance hit rate is inclusive and fractional."""
    actual, predicted = np.asarray(observed, dtype=float), np.asarray(forecast, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Observed and forecast must be equally sized one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Score inputs must already be paired and finite.")
    result = dict(samples=int(len(actual)), mae=None, bias=None, rmse=None, hit_rate=None, r2=None)
    if not len(actual):
        return result
    error = predicted - actual
    residual_squares = float(np.dot(error, error))
    centered = actual - actual.mean()
    # A repeated decimal such as 0.1 can have a rounded mean differing by one
    # ulp. Detect an exactly constant observed vector before forming R².
    total_squares = 0.0 if np.all(actual == actual[0]) else float(np.dot(centered, centered))
    result.update(mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                  rmse=float(np.sqrt(np.mean(error ** 2))),
                  hit_rate=float(np.mean(np.abs(error) <= 5.0)),
                  r2=(1.0 - residual_squares / total_squares) if total_squares > 0 else None)
    return result


@lru_cache(maxsize=3)
def _storage_problem(n: int):
    """Matrices for charge, discharge, stored energy and binary operating mode."""
    from scipy.optimize import Bounds, LinearConstraint

    if n not in (23, 24, 25):
        raise ValueError("Storage is defined only on a complete 23, 24 or 25-hour civil day.")
    eta = math.sqrt(STORAGE_ROUND_TRIP_EFFICIENCY)
    size = 4 * n + 1
    charge, discharge, state, mode = 0, n, 2 * n, 3 * n + 1
    lower = np.zeros(size)
    upper = np.ones(size)
    upper[state:state + n + 1] = STORAGE_CAPACITY_MWH
    upper[state] = upper[state + n] = 0.0
    matrix = np.zeros((3 * n, size))
    limits_low = np.full(3 * n, -np.inf)
    limits_high = np.zeros(3 * n)
    for hour in range(n):
        matrix[hour, state + hour + 1] = 1
        matrix[hour, state + hour] = -1
        matrix[hour, charge + hour] = -eta
        matrix[hour, discharge + hour] = 1 / eta
        limits_low[hour] = 0
        matrix[n + hour, charge + hour] = 1
        matrix[n + hour, mode + hour] = -STORAGE_POWER_MW
        matrix[2 * n + hour, discharge + hour] = 1
        matrix[2 * n + hour, mode + hour] = STORAGE_POWER_MW
        limits_high[2 * n + hour] = STORAGE_POWER_MW
    integrality = np.zeros(size)
    integrality[mode:] = 1
    return Bounds(lower, upper), LinearConstraint(matrix, limits_low, limits_high), integrality


@lru_cache(maxsize=1024)
def _storage_schedule(prices: tuple[float, ...]):
    """Forecast-only MILP; a failed/time-limited solve is never reported as optimal."""
    n = len(prices)
    if n not in (23, 24, 25) or not all(math.isfinite(value) for value in prices):
        raise ValueError("Storage scheduling requires one finite forecast per physical hour.")
    # A flat nonnegative forecast offers no profitable cycle. Resolve zero-price
    # degeneracy explicitly, rather than inventing arbitrary schedules.
    if len(set(prices)) == 1 and prices[0] >= 0:
        return tuple([0.0] * n), tuple([0.0] * n), tuple([0.0] * (n + 1))
    try:
        from scipy.optimize import milp
        bounds, constraints, integrality = _storage_problem(n)
        objective = np.zeros(4 * n + 1)
        objective[:n], objective[n:2 * n] = prices, -np.asarray(prices)
        solved = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                      options={"time_limit": 2.0, "mip_rel_gap": 0.0})
    except (ImportError, RuntimeError, ValueError):
        return None
    if not solved.success or solved.status != 0 or solved.x is None:
        return None
    charge = np.asarray(solved.x[:n])
    discharge = np.asarray(solved.x[n:2 * n])
    state = np.asarray(solved.x[2 * n:3 * n + 1])
    eta, tolerance = math.sqrt(STORAGE_ROUND_TRIP_EFFICIENCY), 2e-6
    if (not np.isfinite(solved.x).all() or np.min(charge) < -tolerance
            or np.min(discharge) < -tolerance or np.max(charge) > 1 + tolerance
            or np.max(discharge) > 1 + tolerance or np.min(state) < -tolerance
            or np.max(state) > 4 + tolerance or abs(state[0]) > tolerance
            or abs(state[-1]) > tolerance
            or np.any((charge > tolerance) & (discharge > tolerance))
            or np.max(np.abs(np.diff(state) - eta * charge + discharge / eta)) > tolerance):
        return None
    return tuple(charge), tuple(discharge), tuple(state)


def optimize_storage_schedule(forecast):
    """Return a fresh forecast-only plan, or None when the optimizer is unavailable.

    Both powers are measured at the grid connection. Each physical interval is
    one hour; stored energy uses symmetric charge/discharge efficiency sqrt(.85).
    """
    if not all(_finite(value) for value in forecast):
        raise ValueError("Storage forecast values must be finite numbers.")
    prices = tuple(float(value) for value in forecast)
    plan = _storage_schedule(prices)
    if plan is None:
        return None
    charge, discharge, state = plan
    return {"charge_mw": list(charge), "discharge_mw": list(discharge), "soc_mwh": list(state),
            "forecast_pnl": float(np.dot(np.asarray(discharge) - charge, prices))}


def _settle_plan(plan, actual):
    return float(np.dot(np.asarray(plan["discharge_mw"]) - plan["charge_mw"], actual))


def _history(zone, report_day):
    """Reject malformed and duplicated instants, preserving no arbitrary duplicate."""
    timezone = zone["timezone"]
    rows, duplicates, invalid = {}, set(), 0
    for source in zone.get("rolling_history", {}).get("rows", []):
        try:
            timestamp = pd.Timestamp(source.get("timestamp_utc"))
            if (pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.minute
                    or timestamp.second or timestamp.microsecond or timestamp.nanosecond):
                raise ValueError("Explicit hourly timestamp required.")
            timestamp = timestamp.tz_convert("UTC")
            day = timestamp.tz_convert(timezone).date().isoformat()
        except (ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        if day > report_day:
            continue
        if timestamp in rows or timestamp in duplicates:
            rows.pop(timestamp, None)
            duplicates.add(timestamp)
            continue
        rows[timestamp] = {key: float(source[key]) if _finite(source.get(key)) else None
                           for key in ("observed", "storm", "model", "storm_dashboard_cache", "model_p10", "model_p90")}
    return rows, invalid, len(duplicates)


def _build_strategy_zones(zone, report_day):
    """Compute both strategies once, on identical days after causal calibration."""
    from .model_storm_bess import STRATEGIES, simulate_bess_day
    from .model_storm_storm_calibration import calibrate_storm_quantiles

    timezone = zone["timezone"]
    rows, invalid, duplicate = _history(zone, report_day)
    calibration, calibration_audit = calibrate_storm_quantiles(rows, timezone=timezone)
    for stamp, interval in calibration.items():
        rows[stamp].update(interval)
    candidate_days = sorted({stamp.tz_convert(timezone).date().isoformat() for stamp in rows})
    observed_days = {day for day in candidate_days
                     if all(rows.get(hour, {}).get("observed") is not None for hour in _hours(day, timezone))}
    anchor = max(observed_days) if observed_days else None
    base = {"zone": zone["zone"], "name": zone.get("name", zone["zone"]), "timezone": timezone,
            "anchor_day": anchor, "invalid_timestamp_rows": invalid, "duplicate_hours": duplicate,
            "history_status": "available" if anchor else "no_complete_observed_day",
            "source_scope": "internal_completed"}
    end = date.fromisoformat(anchor or report_day)
    paired, complete_days, intervals, pnl = {}, {}, {}, {}
    for offset in range(max(WINDOWS)) if anchor else []:
        day = (end - timedelta(days=offset)).isoformat()
        expected = _hours(day, timezone)
        available = [(hour, rows[hour]) for hour in expected if hour in rows
                     and all(rows[hour][key] is not None for key in ("observed", "storm", "model"))]
        paired[day] = available
        if len(available) != len(expected):
            continue
        complete_days[day] = {key: float(np.mean([row[key] for _, row in available]))
                              for key in ("observed", "storm", "model")}
        intervals[day] = {
            key: all(row.get(key + "_p10") is not None and row.get(key + "_p90") is not None
                     and row[key + "_p10"] <= row[key] <= row[key + "_p90"]
                     for _, row in available)
            for key, _ in PROVIDERS
        }
        # Identical support for QB and UB, both providers: calibration warmup
        # and missing native intervals are excluded from BOTH strategies' PnL.
        if not all(intervals[day].values()):
            continue
        actual = [row["observed"] for _, row in available]
        pnl[day] = {}
        for strategy in STRATEGIES:
            pnl[day][strategy] = {}
            for key, _ in PROVIDERS:
                trade = simulate_bess_day(
                    [row[key] for _, row in available], actual, strategy=strategy,
                    lower=[row[key + "_p10"] for _, row in available],
                    upper=[row[key + "_p90"] for _, row in available])
                trade.update(buy_timestamp_utc=expected[trade["buy_index"]].isoformat(),
                             sell_timestamp_utc=expected[trade["sell_index"]].isoformat())
                pnl[day][strategy][key] = trade
    output = {}
    for strategy in STRATEGIES:
        result = {**base, "strategy": strategy, "windows": {},
                  "pnl_audit": [{"delivery_day": day, "providers": pnl[day][strategy]}
                                for day in sorted(pnl)]}
        for window in WINDOWS:
            start = end - timedelta(days=window - 1)
            days = [(start + timedelta(days=offset)).isoformat() for offset in range(window)]
            hourly_rows = [row for day in days for _, row in paired.get(day, [])]
            daily_rows = [complete_days[day] for day in days if day in complete_days]
            pnl_days = [day for day in days if day in pnl]
            missing = {key: sum(day in intervals and not intervals[day][key] for day in days)
                       for key, _ in PROVIDERS}
            section = {"start_day": start.isoformat(), "end_day": end.isoformat(),
                       "expected_hours": sum(len(_hours(day, timezone)) for day in days),
                       "paired_hours": len(hourly_rows), "complete_paired_days": len(daily_rows),
                       "observed_complete_days": sum(day in observed_days for day in days),
                       "pnl_days": len(pnl_days), "pnl_failed_days": len(daily_rows) - len(pnl_days),
                       "pnl_excluded_days": len(daily_rows) - len(pnl_days),
                       "pnl_provider_days": {key: len(pnl_days) for key, _ in PROVIDERS},
                       "pnl_missing_quantile_days": missing,
                       "pnl_support_days": pnl_days, "frequencies": {}}
            for frequency, values in (("60min", hourly_rows), ("day", daily_rows)):
                providers = []
                for key, label in PROVIDERS:
                    trades = [pnl[day][strategy][key] for day in pnl_days]
                    total = float(sum(trade["pnl_eur"] for trade in trades)) if trades else None
                    metrics = score_forecast([row["observed"] for row in values], [row[key] for row in values])
                    metrics.update(key=key, label="Storm calibré" if key == "storm" and strategy == "quantile_based" else label,
                                   pnl_days=len(pnl_days), total_pnl=total,
                                   daily_pnl=total / len(pnl_days) if pnl_days else None,
                                   trade_days=sum(trade["executed"] for trade in trades),
                                   submitted_days=sum(trade["proposed"] for trade in trades),
                                   pnl_comparison_eligible=bool(pnl_days),
                                   pnl_unavailable_reason=None if pnl_days else
                                   "Historique commun insuffisant : 60 jours antérieurs complets pour calibrer Storm et P10/P90 NYX requis.",
                                   pnl_kind="paper_simulation")
                    providers.append(metrics)
                section["frequencies"][frequency] = {"samples": len(values), "providers": providers}
            result["windows"][str(window)] = section
        output[strategy] = result
    return output, calibration_audit


def _build_zone(zone, report_day):
    """The default view is Unlimited Bid on common completed-history support."""
    return _build_strategy_zones(zone, report_day)[0]["unlimited_bid"]


def score_dashboard_forecast(observed, forecast, *, round_prices=True) -> dict:
    """Compatibility convention numerically checked against the STP UI.

    Unlike the internal paired score, missing forecasts remain in the hit-rate
    denominator and the R2 reference population. Hourly prices are rounded to
    cents before scoring or DAY aggregation. Daily means must not be rounded a
    second time (round_prices=False).
    This is an observed convention, not access to the external server's code.
    """
    actual, predicted = np.asarray(observed, dtype=float), np.asarray(forecast, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Observed and forecast must be equally sized one-dimensional arrays.")
    if np.isinf(actual).any() or np.isinf(predicted).any():
        raise ValueError("Infinite prices are invalid.")
    if round_prices:
        actual, predicted = np.round(actual, 2), np.round(predicted, 2)
    paired = np.isfinite(actual) & np.isfinite(predicted)
    result = score_forecast(actual[paired], predicted[paired])
    result["denominator_samples"] = int(len(actual))
    if not paired.any():
        return result
    error = predicted[paired] - actual[paired]
    all_observed = actual[np.isfinite(actual)]
    centered = all_observed - all_observed.mean()
    sst = 0.0 if np.all(all_observed == all_observed[0]) else float(np.dot(centered, centered))
    result.update(hit_rate=float(np.count_nonzero(np.abs(error) <= 5.0 + 1e-9) / len(actual)),
                  r2=1 - float(np.dot(error, error)) / sst if sst > 0 else None)
    return result


def _dashboard_zone(zone, report_day):
    """Cache-only scores; never let a missing Model forecast change Storm's score."""
    from .model_storm_vps_metrics import build_vps_windows

    vps_windows = build_vps_windows(zone, report_day, WINDOWS) if "vps_history" in zone else {}
    timezone = zone["timezone"]
    rows, invalid, duplicates = _history(zone, report_day)
    source = zone.get("rolling_history", {}).get("sources", {}).get("storm_dashboard_cache", {})
    verified = source.get("exact_cache_hour_mask_verified") is True and source.get("status") in ("complete", "partial")
    result = {"zone": zone["zone"], "name": zone.get("name", zone["zone"]), "timezone": timezone,
              "anchor_day": report_day, "history_status": "available" if verified else "cache_provenance_unavailable",
              "invalid_timestamp_rows": invalid, "duplicate_hours": duplicates,
              "cache_source": source, "windows": {}}
    end = date.fromisoformat(report_day)
    days = [(end - timedelta(days=offset)).isoformat() for offset in reversed(range(max(WINDOWS)))]
    grid = pd.DatetimeIndex([stamp for day in days for stamp in _hours(day, timezone)])
    frame = pd.DataFrame([{key: rows.get(stamp, {}).get(key) for key in
                           ("observed", "storm_dashboard_cache", "model")} for stamp in grid], index=grid, dtype=float)
    if not verified:
        frame["storm_dashboard_cache"] = np.nan
    frame = frame.rename(columns={"storm_dashboard_cache": "storm"})
    # The external table reconciles at cent precision *before* aggregation.
    # Rounding just the final score does not reproduce DE's 30-day bias.
    frame = frame.round(2)
    # Model uses only cache-supported hours. Storm is independent of Model gaps.
    frame.loc[frame.storm.isna(), "model"] = np.nan
    frame["day"] = [stamp.tz_convert(timezone).date().isoformat() for stamp in grid]
    for window in WINDOWS:
        start = (end - timedelta(days=window - 1)).isoformat()
        selected = frame.loc[frame.day >= start]
        actual = selected.observed
        storm_pair = actual.notna() & selected.storm.notna()
        model_pair = storm_pair & selected.model.notna()
        groups = selected.groupby("day", sort=True)
        daily = groups[["observed", "storm", "model"]].mean()
        counts = groups[["observed", "storm", "model"]].count()
        # A partial Model day cannot be compared with Storm's daily mean.
        daily.loc[counts.model != counts.storm, "model"] = np.nan
        expected_per_day = groups.size()
        complete = (counts.observed == expected_per_day) & (counts.storm == expected_per_day) & (counts.model == expected_per_day)
        section = {"start_day": start, "end_day": report_day, "scope": "dashboard_cache",
                   "source_status": source.get("status", "unavailable"),
                   "source_extracted_at_utc": source.get("extracted_at_utc"),
                   "expected_hours": len(selected), "paired_hours": int(model_pair.sum()),
                   "storm_hours": int(storm_pair.sum()), "model_hours": int(model_pair.sum()),
                   "cache_missing_hours": int(selected.storm.isna().sum()),
                   "complete_paired_days": int(complete.sum()),
                   "observed_complete_days": int((counts.observed == expected_per_day).sum()),
                   "pnl_days": 0, "pnl_failed_days": 0, "official_pnl_verified": False,
                   "external_window_available": window in (7, 30, 60, 90), "frequencies": {}}
        vps = vps_windows.get(str(window))
        if vps is not None:
            section["vps"] = vps
            section["pnl_days"] = vps["estimated_days"]
            section["pnl_failed_days"] = len(vps["excluded_days"])
        for frequency, data in (("60min", selected), ("day", daily)):
            providers = []
            masks = {key: data.observed.notna() & data[key].notna() for key, _ in PROVIDERS}
            same_support = bool(masks["storm"].equals(masks["model"]))
            for key, label in PROVIDERS:
                metrics = score_dashboard_forecast(data.observed, data[key], round_prices=False)
                metrics.update(key=key, label=label, daily_pnl=None, pnl_days=0,
                               comparison_eligible=same_support, calculation_contract="stp_cache_compatibility_v1")
                if vps is not None:
                    metrics.update(daily_pnl=vps[f"{key}_daily_pnl"],
                                   pnl_days=vps["published_days" if key == "storm" else "estimated_days"],
                                   pnl_kind="published" if key == "storm" else "estimated",
                                   pnl_comparison_eligible=False)
                providers.append(metrics)
            section["frequencies"][frequency] = {"samples": int(masks["model"].sum()),
                "storm_samples": int(masks["storm"].sum()), "model_samples": int(masks["model"].sum()),
                "denominator_samples": len(data), "same_support": same_support, "providers": providers}
        result["windows"][str(window)] = section
    return result


def build_rolling_performance(payload: dict) -> dict:
    """Two battery strategies over completed history, never cache-only/VPS scores."""
    from .model_storm_bess import METHODOLOGY, STRATEGIES, QUANTILE_ALPHA

    report_day = date.fromisoformat(payload["delivery_day"]).isoformat()
    strategy_zones = {strategy: [] for strategy in STRATEGIES}
    calibration = {}
    for zone in payload.get("zones", []):
        variants, audit = _build_strategy_zones(zone, report_day)
        for strategy in STRATEGIES:
            strategy_zones[strategy].append(variants[strategy])
        calibration[zone["zone"]] = audit
    return {"schema_version": 3, "windows": list(WINDOWS), "frequencies": list(FREQUENCIES),
            "default_window": 90, "default_frequency": "60min", "report_delivery_day": report_day,
            "source_scope": "internal_completed", "strategies": list(STRATEGIES),
            "default_strategy": "unlimited_bid", "quantile_alpha": QUANTILE_ALPHA,
            "strategy_zones": strategy_zones, "zones": strategy_zones["unlimited_bid"],
            "storm_calibration": calibration,
            "methodology": {
                "anchor": "Latest complete observed civil day on or before report delivery day, independently per zone.",
                "pairing": "Identical finite observed, Model and Storm hourly samples; completed history only; no imputation.",
                "day": "Daily mean prices require all 23/24/25 physical hours for both forecasts and observed prices.",
                "bias": "Forecast minus observed (EUR/MWh); closest to zero is best.",
                "hit_rate": "Fraction of absolute errors <= 5 EUR/MWh (inclusive).",
                "r2": "1 - squared errors / observed sum squared deviations; undefined for constant observations.",
                "pnl": "Paper-based battery simulation, not live trading or published Storm VPS. Forecast-only pair selection, then realized-price settlement.",
                "pnl_unit": "EUR/day over identical eligible complete days for both providers AND strategies, including zeros on rejected/no-trade days.",
                "pnl_support": "Native NYX P10/P90 and causal Storm P10/P90 required; initial calibration and missing intervals excluded from both QB and UB.",
                "pnl_day_frequency": "DAY changes accuracy aggregation only; trading always uses physical hourly prices.",
                "storm_quantiles": "Historical errors by local hour, previous 365 civil days only; at least 60 complete days and 30 samples/hour. Empirical P10/P90 residuals centered on their P50 keep Storm's point forecast fixed. Not native Storm quantiles, no coverage guarantee.",
                "storage": dict(METHODOLOGY)}}
