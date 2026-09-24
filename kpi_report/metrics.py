"""Paired KPI comparisons on common physical hours, never filled observations.

All countries use the Europe/Paris delivery calendar.  Daily metrics only use
complete civil days (including the 23/25 physical hours of DST transitions).
The ``ALL`` result pools hours and country-days; it is not a mean of country KPIs.
"""
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd


TIMEZONE = "Europe/Paris"
TOLERANCE = 1e-9
STORM_ID = "__storm__"
ALL_ZONE = "ALL"
REQUIRED = ("model_id", "zone", "timestamp_utc", "forecast", "actual", "storm")
RESERVED_MODELS = {STORM_ID, "actual", "storm", "zone", "delivery_day"}


class KPIError(ValueError):
    """An input identity or a comparison contract is invalid."""


def _selection(values: list[str] | None, available: pd.Series, name: str) -> list[str]:
    result = sorted(available.unique().tolist()) if values is None else list(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise KPIError(f"{name}: nonempty string identifiers are required.")
    if len(result) != len(set(result)):
        raise KPIError(f"{name}: duplicate selection.")
    return result


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise KPIError(f"Missing columns: {sorted(missing)}")
    out = frame.loc[:, REQUIRED].copy(deep=True)
    for name in ("model_id", "zone"):
        if any(not isinstance(value, str) or not value.strip() for value in out[name]):
            raise KPIError(f"{name}: nonempty string identifiers are required.")
    if out.model_id.isin(RESERVED_MODELS).any() or out.zone.eq(ALL_ZONE).any():
        raise KPIError("Reserved model or zone identifier.")
    if isinstance(out.timestamp_utc.dtype, pd.DatetimeTZDtype):
        # Large Parquet panels are already typed: never parse hundreds of
        # thousands of physical timestamps individually for each UI period.
        out["timestamp_utc"] = out.timestamp_utc.dt.tz_convert("UTC")
    else:
        timestamps = []
        for value in out.timestamp_utc:
            try:
                stamp = pd.Timestamp(value)
            except (ValueError, TypeError) as exc:
                raise KPIError("Invalid timestamp_utc.") from exc
            if pd.isna(stamp) or stamp.tzinfo is None:
                raise KPIError("timestamp_utc must be timezone-aware and nonmissing.")
            timestamps.append(stamp.tz_convert("UTC"))
        out["timestamp_utc"] = pd.to_datetime(timestamps, utc=True)
    if out.timestamp_utc.isna().any():
        raise KPIError("timestamp_utc must be timezone-aware and nonmissing.")
    if not out.timestamp_utc.eq(out.timestamp_utc.dt.floor("h")).all():
        raise KPIError("timestamp_utc must identify a whole physical hour.")
    for name in ("forecast", "actual", "storm"):
        try:
            out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise KPIError(f"{name} must contain numeric values or NaN.") from exc
        if np.isinf(out[name]).any():
            raise KPIError(f"{name} contains infinity.")
    keys = ["model_id", "zone", "timestamp_utc"]
    duplicates = out.duplicated(keys, keep=False)
    for _, group in out.loc[duplicates].groupby(keys, sort=False):
        # Duplicate observations must really be identical, including missingness.
        for name in ("forecast", "actual", "storm"):
            if group[name].nunique(dropna=False) != 1:
                raise KPIError(f"Conflicting duplicate {keys}: {name}.")
    return out.drop_duplicates(keys).reset_index(drop=True)


def _references(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["zone", "timestamp_utc"]
    groups = frame.groupby(keys, sort=True, observed=True)
    for name in ("actual", "storm"):
        spreads = groups[name].max() - groups[name].min()
        bad = spreads.gt(TOLERANCE)
        if bad.any():
            identity = spreads.index[bad][0]
            raise KPIError(f"Conflicting shared {name} reference at {identity}.")
    # first() ignores NaNs: one finite verified reference is enough for this key.
    return groups[["actual", "storm"]].first()


def _empty_row(zone: str, model_id: str) -> dict:
    return {
        "zone": zone, "model_id": model_id, "status": "no_common_support",
        "n_hours": 0, "n_days": 0, "n_calendar_days": 0,
        "daily_status": "no_complete_common_days",
        "mae_eur_mwh": None, "rmse_eur_mwh": None,
        "win_rate_hour_pct": None, "win_rate_day_mae_pct": None,
        "win_rate_day_mean_price_pct": None,
        "wins_hour": 0, "ties_hour": 0, "losses_hour": 0,
        "wins_day_mae": 0, "ties_day_mae": 0, "losses_day_mae": 0,
        "wins_day_mean_price": 0, "ties_day_mean_price": 0,
        "losses_day_mean_price": 0,
        "mae_day_mean_price_eur_mwh": None,
        "mean_price_eur_mwh": None, "observed_mean_price_eur_mwh": None,
        "storm_mean_price_eur_mwh": None,
        "mean_daily_price_eur_mwh": None,
        "observed_mean_daily_price_eur_mwh": None,
        "storm_mean_daily_price_eur_mwh": None,
    }


def _counts(error: np.ndarray, reference_error: np.ndarray) -> tuple[int, int, int]:
    delta = error - reference_error
    return int((delta < -TOLERANCE).sum()), int((np.abs(delta) <= TOLERANCE).sum()), int((delta > TOLERANCE).sum())


def _summarize(zone: str, model_id: str, hours: pd.DataFrame, daily: pd.DataFrame) -> dict:
    row = _empty_row(zone, model_id)
    if hours.empty:
        return row
    prediction = hours.storm if model_id == STORM_ID else hours[model_id]
    error = (prediction - hours.actual).to_numpy(dtype=float)
    storm_error = (hours.storm - hours.actual).abs().to_numpy(dtype=float)
    row.update({
        "status": "ok", "n_hours": int(len(hours)),
        "mae_eur_mwh": float(np.abs(error).mean()),
        "rmse_eur_mwh": float(np.sqrt(np.square(error).mean())),
        "mean_price_eur_mwh": float(prediction.mean()),
        "observed_mean_price_eur_mwh": float(hours.actual.mean()),
        "storm_mean_price_eur_mwh": float(hours.storm.mean()),
    })
    if model_id != STORM_ID:
        wins, ties, losses = _counts(np.abs(error), storm_error)
        row.update(wins_hour=wins, ties_hour=ties, losses_hour=losses,
                   win_rate_hour_pct=100.0*wins/len(hours))
    if not daily.empty:
        row.update({
            "daily_status": "ok", "n_days": int(len(daily)),
            "n_calendar_days": int(daily.delivery_day.nunique()),
            "mae_day_mean_price_eur_mwh": float(daily.abs_mean_price_error_eur_mwh.mean()),
            "mean_daily_price_eur_mwh": float(daily.mean_price_eur_mwh.mean()),
            "observed_mean_daily_price_eur_mwh": float(daily.observed_mean_price_eur_mwh.mean()),
            "storm_mean_daily_price_eur_mwh": float(daily.storm_mean_price_eur_mwh.mean()),
        })
        if model_id != STORM_ID:
            for suffix in ("mae", "mean_price"):
                for label in ("wins", "ties", "losses"):
                    row[f"{label}_day_{suffix}"] = int(daily[f"{label}_day_{suffix}"].sum())
                row[f"win_rate_day_{suffix}_pct"] = 100.0*row[f"wins_day_{suffix}"]/len(daily)
    return row


def compute_kpis(
    frame: pd.DataFrame, *, end_day: str, days: int = 365,
    models: list[str] | None = None, zones: list[str] | None = None,
) -> dict:
    """Return strict paired KPI rows, coverage and complete-day diagnostics.

    ``end_day`` is inclusive in Europe/Paris. Every selected model must supply
    a finite forecast for an hour to enter any model's comparison in that zone.
    Zero and negative prices are valid. No missing value is interpolated.
    Wins use absolute error, with a 1e-9 EUR/MWh tie tolerance; ties stay in the
    denominator but are not wins. Storm rows have no self-comparison win rate.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise KPIError("days must be a positive integer.")
    try:
        end = date.fromisoformat(end_day)
    except (ValueError, TypeError) as exc:
        raise KPIError("end_day must be an ISO civil date YYYY-MM-DD.") from exc
    if end.isoformat() != end_day:
        raise KPIError("end_day must be an ISO civil date YYYY-MM-DD.")
    start = end-timedelta(days=days-1)
    lower = pd.Timestamp(start, tz=TIMEZONE).tz_convert("UTC")
    upper = pd.Timestamp(end+timedelta(days=1), tz=TIMEZONE).tz_convert("UTC")
    expected = pd.date_range(lower, upper, freq="h", inclusive="left")
    expected_counts = pd.Series(expected.tz_convert(TIMEZONE).date).value_counts()
    data = _validate(frame)
    selected_models = _selection(models, data.model_id, "models")
    selected_zones = _selection(zones, data.zone, "zones")
    if RESERVED_MODELS.intersection(selected_models) or ALL_ZONE in selected_zones:
        raise KPIError("Reserved model or zone identifier.")
    data = data.loc[data.model_id.isin(selected_models) & data.zone.isin(selected_zones)
                    & data.timestamp_utc.ge(lower) & data.timestamp_utc.lt(upper)]
    references = _references(data)
    result = {
        "schema_version": 1,
        "period": {"start_day": start.isoformat(), "end_day": end.isoformat(),
                   "days": days, "timezone": TIMEZONE,
                   "start_utc": lower.isoformat(), "end_exclusive_utc": upper.isoformat(),
                   "expected_hours_per_zone": int(len(expected))},
        "selection": {"models": selected_models, "zones": selected_zones,
                      "storm_model_id": STORM_ID, "aggregate_zone": ALL_ZONE},
        "methodology": {
            "support": "finite forecasts for all selected models and shared actual/Storm references, per zone",
            "daily_support": "complete common civil days only (23/24/25 physical hours)",
            "win_rule": "absolute error strictly smaller than Storm by more than 1e-9 EUR/MWh",
            "tie_denominator": "included; ties are not wins",
            "aggregation": "pooled physical hours for hourly KPIs; pooled country-days for daily KPIs",
            "mean_price": "hour-weighted on the common hourly support",
            "mean_daily_price": "equal-weight complete country-day means",
            "tolerance_eur_mwh": TOLERANCE,
        },
        "coverage": [], "rows": [], "daily_rows": [],
    }
    hour_frames = []
    day_records = []
    for zone in selected_zones:
        local = data.loc[data.zone.eq(zone)]
        wide = local.pivot(index="timestamp_utc", columns="model_id", values="forecast").reindex(columns=selected_models)
        refs = references.xs(zone, level="zone") if zone in references.index.get_level_values("zone") else pd.DataFrame(columns=["actual", "storm"], index=wide.index)
        wide = wide.join(refs, how="outer")
        paired = wide.dropna(subset=selected_models+["actual", "storm"]).copy() if selected_models else wide.iloc[:0].copy()
        paired["zone"] = zone
        paired["delivery_day"] = paired.index.tz_convert(TIMEZONE).date
        hour_frames.append(paired)
        counts = paired.groupby("delivery_day").size()
        complete_days = [day for day, count in counts.items() if int(count) == int(expected_counts[day])]
        available = {model: int(wide[model].notna().sum()) for model in selected_models}
        result["coverage"].append({
            "zone": zone, "n_expected_hours": int(len(expected)),
            "n_common_hours": int(len(paired)), "n_complete_days": len(complete_days),
            "n_incomplete_days": int(len(counts)-len(complete_days)),
            "available_hours_by_model": available,
            "missing_models": [model for model in selected_models if available[model] == 0],
            "reference_hours": int(wide[["actual", "storm"]].notna().all(axis=1).sum()),
            "first_common_hour_utc": paired.index.min().isoformat() if len(paired) else None,
            "last_common_hour_utc": paired.index.max().isoformat() if len(paired) else None,
        })
        for day in sorted(complete_days):
            subset = paired.loc[paired.delivery_day.eq(day)]
            actual_mean = float(subset.actual.mean())
            storm_mean = float(subset.storm.mean())
            storm_mae = float((subset.storm-subset.actual).abs().mean())
            storm_mean_error = abs(storm_mean-actual_mean)
            for model in selected_models+[STORM_ID]:
                prediction = subset.storm if model == STORM_ID else subset[model]
                error = prediction-subset.actual
                predicted_mean = float(prediction.mean())
                mae = float(error.abs().mean())
                mean_error = abs(predicted_mean-actual_mean)
                record = {
                    "zone": zone, "delivery_day": day.isoformat(), "model_id": model,
                    "n_hours": int(len(subset)), "mae_eur_mwh": mae,
                    "rmse_eur_mwh": float(np.sqrt(np.square(error).mean())),
                    "mean_price_eur_mwh": predicted_mean,
                    "observed_mean_price_eur_mwh": actual_mean,
                    "storm_mean_price_eur_mwh": storm_mean,
                    "abs_mean_price_error_eur_mwh": mean_error,
                    "storm_mae_eur_mwh": storm_mae,
                    "storm_abs_mean_price_error_eur_mwh": storm_mean_error,
                }
                for suffix, value, reference_value in (("mae", mae, storm_mae), ("mean_price", mean_error, storm_mean_error)):
                    count_values = _counts(np.array([value]), np.array([reference_value]))
                    for label, count in zip(("wins", "ties", "losses"), count_values):
                        record[f"{label}_day_{suffix}"] = None if model == STORM_ID else count
                day_records.append(record)
    daily = pd.DataFrame(day_records)
    for zone, hours in zip(selected_zones, hour_frames):
        for model in selected_models+[STORM_ID]:
            selected_daily = daily.loc[daily.zone.eq(zone) & daily.model_id.eq(model)] if len(daily) else daily
            result["rows"].append(_summarize(zone, model, hours, selected_daily))
    all_hours = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    for model in selected_models+[STORM_ID]:
        selected_daily = daily.loc[daily.model_id.eq(model)] if len(daily) else daily
        result["rows"].append(_summarize(ALL_ZONE, model, all_hours, selected_daily))
    result["daily_rows"] = day_records
    return result
