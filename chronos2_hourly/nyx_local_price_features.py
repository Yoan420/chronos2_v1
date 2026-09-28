"""Price features for each daily D-1 08:00 decision, with no I/O or fitting.

Delivery prices for complete day D-1 are assumed already auction-published.
No price or observed error from D is read into D's features. Forecast origins
are logical replay origins: checking them does NOT certify provider vintages.

Civil-hour mapping is deliberate: source-day duplicate hours are averaged;
both target folds receive that same value. A nonexistent spring hour, missing
physical source row or incomplete daily aggregate remains NaN. Each numeric
feature has an explicit availability flag. No interpolation/imputation occurs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
import re

import numpy as np
import pandas as pd


TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels",
             "NL": "Europe/Amsterdam", "ES": "Europe/Madrid"}
DAY_LAGS = (1, 2, 7)
_FORBIDDEN = re.compile(r"(^|_)(actual|target|observed|realized|realised|label)(_|$)", re.I)


def _index(index, name):
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or not len(index)
            or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.floor("h"))):
        raise ValueError(f"{name}: nonempty, unique, ordered UTC hourly index required")
    return index


def _no_storm(names):
    if any("storm" in str(name).lower() for name in names if name is not None):
        raise ValueError("Storm is forbidden in price-feature inputs")


def _series(value, name, *, forecast=False):
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1 or not value.columns.is_unique:
            raise ValueError(f"{name}: exactly one numeric column required")
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name}: Series or single-column DataFrame required")
    _index(value.index, name)
    _no_storm([value.name])
    if forecast and _FORBIDDEN.search(str(value.name)):
        raise ValueError(f"{name}: observed labels cannot be supplied as predictions")
    values = pd.to_numeric(value, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name}: supplied values must be finite; absent timestamps may stay absent")
    return values.copy()


def _cutoffs(index, timezone):
    dates = index.tz_convert(timezone).date
    # Add the clock hour before timezone localization: DST changes must not
    # turn 08:00 civil time into 07:00/09:00 on the transition day.
    by_day = {day: pd.Timestamp(f"{day - timedelta(days=1)} 08:00", tz=timezone).tz_convert("UTC")
              for day in set(dates)}
    return pd.Series([by_day[day] for day in dates], index=index, dtype="datetime64[ns, UTC]")


def _origins(value, indices, timezone):
    if not indices:
        return None
    if not isinstance(value, pd.Series):
        raise ValueError("forecast_origins is required for predictions or known covariates")
    _index(value.index, "forecast_origins")
    stamps = [pd.Timestamp(stamp) for stamp in value]
    if any(pd.isna(stamp) or stamp.tzinfo is None for stamp in stamps):
        raise ValueError("forecast_origins must contain finite timezone-aware timestamps")
    origins = pd.Series(pd.to_datetime(stamps, utc=True), index=value.index)
    for index in indices:
        selected = origins.reindex(index)
        if selected.isna().any() or (selected > _cutoffs(index, timezone)).any():
            raise ValueError("Missing forecast origin or origin later than civil D-1 08:00")
    return origins


def _civil_table(series, timezone, origins=None):
    local = series.index.tz_convert(timezone)
    grid = pd.date_range(pd.Timestamp(local[0].date(), tz=timezone),
                         pd.Timestamp(local[-1].date() + timedelta(days=1), tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")
    civil = grid.tz_convert(timezone)
    work = pd.DataFrame({"value": series.reindex(grid), "day": civil.date,
                         "hour": civil.hour, "source": grid}, index=grid)
    if origins is None:
        # Conservative availability bound under the auction-price contract;
        # this is explicitly not a measured publication timestamp.
        bounds = {day: pd.Timestamp(f"{day} 08:00", tz=timezone).tz_convert("UTC")
                  for day in set(civil.date)}
        work["available_by"] = [bounds[day] for day in civil.date]
    else:
        work["available_by"] = origins.reindex(grid)
    hourly = work.groupby(["day", "hour"], sort=True)
    daily = work.groupby("day", sort=True)

    def aggregate(groups):
        result = groups.agg(mean=("value", "mean"), minimum=("value", "min"),
                            maximum=("value", "max"), source=("source", "max"),
                            available_by=("available_by", "max"))
        result["std"] = groups["value"].std(ddof=0)
        complete = groups["value"].count().eq(groups.size())
        result.loc[~complete, ["mean", "minimum", "maximum", "std"]] = np.nan
        result.loc[~complete, ["source", "available_by"]] = pd.NaT
        return result

    return aggregate(hourly), aggregate(daily)


def _lag(table, index, timezone, days_back, statistic="hour"):
    local = index.tz_convert(timezone)
    dates = [day - timedelta(days=days_back) for day in local.date]
    if statistic == "hour":
        keys = pd.MultiIndex.from_arrays([dates, local.hour], names=["day", "hour"])
        selected = table[0].reindex(keys)
        column = "mean"
    else:
        selected = table[1].reindex(pd.Index(dates, name="day"))
        column = statistic
    return (pd.Series(selected[column].to_numpy(dtype=float), index=index),
            pd.Series(pd.to_datetime(selected.source.to_numpy(), utc=True), index=index),
            pd.Series(pd.to_datetime(selected.available_by.to_numpy(), utc=True), index=index))


def _latest(left, right):
    return left.where(left >= right, right).where(left.notna() & right.notna())


def build_price_features(
    prices: Mapping[str, pd.Series | pd.DataFrame], *, zone: str,
    delivery_index: pd.DatetimeIndex, nyx_predictions: pd.DataFrame | None = None,
    past_predictions: Mapping[str, pd.Series | pd.DataFrame] | None = None,
    known_covariates: pd.DataFrame | None = None, known_columns: Sequence[str] = (),
    known_delta_columns: Sequence[str] = (), forecast_origins: pd.Series | None = None,
):
    """Return numeric features plus a per-day, per-feature source audit.

    ``prices`` supplies observed day-ahead prices by country; a full historical
    cache may be passed because only D-1/D-2/D-7 are selected. Other supplied
    countries become neighbours. ``past_predictions`` optionally supplies their
    historical point forecasts; errors are formed on matching physical hours
    before civil-hour aggregation and then lagged.

    ``nyx_predictions`` accepts exactly q10/q50/q90 or nyx__q10/q50/q90, with
    no actual column. ``known_columns`` and ``known_delta_columns`` explicitly
    whitelist known forecast covariates and optional changes versus D-1/D-7.
    All forecast input rows require a shared ``forecast_origins`` Series whose
    logical origin is no later than that row's civil D-1 08:00. This cannot
    establish provider publication/vintage provenance or prequential training.

    A weekly frozen model may consume these new features daily: the audit uses
    each delivery day's own origin, never the start of its weekly model block.
    NaNs and availability flags must be handled by the downstream estimator;
    any learned imputer must be fitted on its training partition only.
    """
    index = _index(delivery_index, "delivery_index")
    if not isinstance(prices, Mapping) or zone not in prices or zone not in TIMEZONES:
        raise ValueError("Explicit supported target country and its prices required")
    _no_storm(prices)
    if not set(prices) <= set(TIMEZONES):
        raise ValueError("Only FR/DE/BE/NL/ES price countries are supported")
    timezone = TIMEZONES[zone]
    price_series = {key: _series(value, f"prices[{key}]") for key, value in sorted(prices.items())}
    past_predictions = {} if past_predictions is None else past_predictions
    if not isinstance(past_predictions, Mapping) or not set(past_predictions) <= set(prices):
        raise ValueError("Past predictions require matching country prices")
    past = {key: _series(value, f"past_predictions[{key}]", forecast=True)
            for key, value in sorted(past_predictions.items())}
    forecast_indices = [value.index for value in past.values()]
    nyx = None
    if nyx_predictions is not None:
        if not isinstance(nyx_predictions, pd.DataFrame):
            raise TypeError("nyx_predictions must be a DataFrame")
        _index(nyx_predictions.index, "nyx_predictions")
        _no_storm(nyx_predictions.columns)
        names = ["q10", "q50", "q90"]
        if set(nyx_predictions.columns) == {"nyx__" + q for q in names}:
            nyx_predictions = nyx_predictions.rename(columns=lambda c: c[5:])
        if not nyx_predictions.columns.is_unique or set(nyx_predictions.columns) != set(names):
            raise ValueError("nyx_predictions requires exactly three quantiles; labels are forbidden")
        nyx = nyx_predictions[names].astype(float)
        if not np.isfinite(nyx.to_numpy()).all() or (np.diff(nyx.to_numpy(), axis=1) < 0).any():
            raise ValueError("Finite ordered NYX quantiles required")
        forecast_indices.append(nyx.index)
    known_columns, known_delta_columns = tuple(known_columns), tuple(known_delta_columns)
    if (len(set(known_columns)) != len(known_columns) or len(set(known_delta_columns)) != len(known_delta_columns)
            or not set(known_delta_columns) <= set(known_columns)):
        raise ValueError("Unique known columns and a delta subset are required")
    known = None
    if known_covariates is not None:
        if not isinstance(known_covariates, pd.DataFrame) or not known_covariates.columns.is_unique:
            raise ValueError("Known covariates require unique DataFrame columns")
        _index(known_covariates.index, "known_covariates")
        _no_storm(known_covariates.columns)
        if any(_FORBIDDEN.search(str(c)) for c in known_covariates):
            raise ValueError("Observed labels cannot be packaged as known covariates")
        if not known_columns or not set(known_columns) <= set(known_covariates):
            raise ValueError("An explicit nonempty known_columns whitelist is required")
        if any(not isinstance(c, str) or not c for c in known_columns):
            raise ValueError("Known column names must be nonempty strings")
        known = known_covariates[list(known_columns)].astype(float)
        if not np.isfinite(known.to_numpy()).all():
            raise ValueError("Supplied known covariates must be finite")
        forecast_indices.append(known.index)
    elif known_columns or known_delta_columns:
        raise ValueError("Known column lists require known_covariates")
    origins = _origins(forecast_origins, forecast_indices, timezone)
    cutoff = _cutoffs(index, timezone)
    blocks, sources, definitions = {}, {}, {}

    def add(name, values, source, available_by, kind):
        values = pd.Series(values, index=index, dtype=float)
        valid = values.notna()
        if not np.isfinite(values[valid].to_numpy()).all():
            raise ValueError(f"Nonfinite derived feature: {name}")
        if (available_by[valid] > cutoff[valid]).any() or available_by[valid].isna().any():
            raise ValueError(f"Feature exceeds daily availability cutoff: {name}")
        blocks[name] = values
        blocks[name + "__available"] = valid
        sources[name] = (source.where(valid), available_by.where(valid))
        definitions[name] = {"source_kind": kind, "availability_column": name + "__available"}

    price_lags = {}
    for country, series in price_series.items():
        table = _civil_table(series, timezone)
        for lag in DAY_LAGS:
            for statistic in ("hour", "mean", "minimum", "maximum", "std"):
                values, source, available = _lag(table, index, timezone, lag, statistic)
                name = f"price_{country.lower()}_d{lag}_{statistic}"
                add(name, values, source, available, "past_day_ahead_price")
                if statistic == "hour":
                    price_lags[country, lag] = values, source, available
    for country in sorted(set(prices) - {zone}):
        for lag in DAY_LAGS:
            own, other = price_lags[zone, lag], price_lags[country, lag]
            add(f"spread_{zone.lower()}_{country.lower()}_d{lag}", own[0] - other[0],
                _latest(own[1], other[1]), _latest(own[2], other[2]), "past_day_ahead_price_spread")
    for country, prediction in past.items():
        error = price_series[country].subtract(prediction)
        table = _civil_table(error, timezone)
        for lag in DAY_LAGS:
            for statistic in ("hour", "mean"):
                values, source, available = _lag(table, index, timezone, lag, statistic)
                add(f"past_error_{country.lower()}_d{lag}_{statistic}", values, source,
                    available, "past_actual_minus_forecast")
    delivery_sources = pd.Series(index, index=index)
    if nyx is not None:
        current = nyx.reindex(index)
        for name, values in {"nyx_q50": current.q50, "nyx_width": current.q90 - current.q10,
                             "nyx_lower_width": current.q50 - current.q10,
                             "nyx_upper_width": current.q90 - current.q50}.items():
            add(name, values, delivery_sources, origins.reindex(index), "current_nyx_forecast")
    if known is not None:
        for column in known_columns:
            current = known[column].reindex(index)
            add("known__" + column, current, delivery_sources, origins.reindex(index), "known_forecast")
            if column in known_delta_columns:
                table = _civil_table(known[column], timezone, origins)
                for lag in (1, 7):
                    previous, source, available = _lag(table, index, timezone, lag)
                    add(f"known_delta_d{lag}__{column}", current - previous,
                        _latest(delivery_sources, source), _latest(origins.reindex(index), available),
                        "current_minus_past_known_forecast")
    local = index.tz_convert(timezone)
    calendar = {"calendar_hour_sin": np.sin(2*np.pi*local.hour/24),
                "calendar_hour_cos": np.cos(2*np.pi*local.hour/24),
                "calendar_weekday_sin": np.sin(2*np.pi*local.dayofweek/7),
                "calendar_weekday_cos": np.cos(2*np.pi*local.dayofweek/7),
                "calendar_is_weekend": (local.dayofweek >= 5).astype(float),
                "calendar_dst_fold": [stamp.fold for stamp in local.to_pydatetime()],
                "calendar_utc_offset_hours": [stamp.utcoffset().total_seconds()/3600
                                              for stamp in local.to_pydatetime()]}
    for name, values in calendar.items():
        add(name, values, delivery_sources, cutoff, "deterministic_calendar")
    frame = pd.DataFrame(blocks, index=index.copy())
    daily_audit = []
    dates = local.date
    # Reduce each hourly column once, rather than scanning the entire history
    # again for every delivery day and feature. sort=False preserves day order.
    daily_grid = cutoff.groupby(dates, sort=False).agg(["first", "size"])
    daily_source = pd.DataFrame({name: pair[0] for name, pair in sources.items()}).groupby(
        dates, sort=False).max()
    daily_available = pd.DataFrame({name: pair[1] for name, pair in sources.items()}).groupby(
        dates, sort=False).max()
    daily_counts = frame[[name + "__available" for name in sources]].groupby(
        dates, sort=False).sum()
    stamp_cache = {}

    def stamp_strings(stamp):
        if pd.isna(stamp):
            return None, None
        if stamp not in stamp_cache:
            stamp_cache[stamp] = stamp.isoformat(), str(stamp.tz_convert(timezone).date())
        return stamp_cache[stamp]

    for (day, daily_cutoff, hours), source_row, available_row, count_row in zip(
        daily_grid.itertuples(name=None), daily_source.itertuples(index=False, name=None),
        daily_available.itertuples(index=False, name=None), daily_counts.itertuples(index=False, name=None),
    ):
        record = {"delivery_day": str(day), "forecast_cutoff_utc": daily_cutoff.isoformat(),
                  "requested_hours": int(hours), "features": {}}
        for name, latest_source, latest_available, count in zip(sources, source_row, available_row, count_row):
            source_stamp, source_day = stamp_strings(latest_source)
            record["features"][name] = {
                "available_hours": int(count),
                "latest_source_delivery_utc": source_stamp,
                "latest_source_civil_day": source_day,
                "latest_available_by_utc": stamp_strings(latest_available)[0],
            }
        daily_audit.append(record)
    audit = {"protocol": "nyx_local_price_features_v1", "zone": zone, "timezone": timezone,
             "neighbors": sorted(set(prices) - {zone}), "price_day_lags": list(DAY_LAGS),
             "civil_hour_mapping": "mean_of_all_source_folds; broadcast_to_target_folds; absent_or_incomplete_source_hour_is_NaN",
             "price_availability_rule": "auction prices for source civil day are assumed known by source-day 08:00",
             "forecast_origins_are_logical": True, "provider_publication_vintages_verified": False,
             "past_predictions_prequential_training_verified": False,
             "daily_feature_origin_independent_of_weekly_fit_origin": True,
             "known_columns": list(known_columns), "known_delta_columns": list(known_delta_columns),
             "imputation_performed": False, "Storm_used_as_input": False,
             "feature_definitions": definitions, "daily_sources": daily_audit}
    return frame, audit


__all__ = ["build_price_features"]
