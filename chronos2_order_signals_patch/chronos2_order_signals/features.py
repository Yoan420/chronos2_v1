from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_modular.common import ZoneData

from .labels import LABEL_COLUMNS


METADATA_COLUMNS = (
    "timestamp",
    "delivery_day",
)


def _previous_local_day_index(
    index: pd.DatetimeIndex,
    days: int,
) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [timestamp - pd.DateOffset(days=days) for timestamp in index]
    )


def _price_lag(
    target: pd.Series,
    index: pd.DatetimeIndex,
    days: int,
) -> np.ndarray:
    lookup = _previous_local_day_index(index, days)
    return target.reindex(lookup).to_numpy(dtype=np.float64)


def _safe_stat(values: pd.Series, operation: str) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    if operation == "median":
        return float(values.median())
    if operation == "std":
        return float(values.std())
    if operation == "mean":
        return float(values.mean())
    if operation == "min":
        return float(values.min())
    if operation == "max":
        return float(values.max())
    raise ValueError(operation)


def _add_path_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    excluded_prefixes = (
        "known_hour_",
        "known_dow_",
        "known_doy_",
        "known_is_weekend",
    )
    base_columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and not column.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    for column in base_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        first_difference = values.diff().fillna(0.0)
        result[f"{column}__d1"] = first_difference
        result[f"{column}__d2"] = first_difference.diff().fillna(0.0)
        result[f"{column}__day_mean"] = float(values.mean())
        result[f"{column}__day_std"] = float(values.std())
        result[f"{column}__day_min"] = float(values.min())
        result[f"{column}__day_max"] = float(values.max())
        result[f"{column}__day_range"] = float(values.max() - values.min())

    return result


def _add_price_history_features(
    frame: pd.DataFrame,
    target: pd.Series,
    delivery_index: pd.DatetimeIndex,
    spike_threshold: float,
) -> pd.DataFrame:
    result = frame.copy()
    origin = delivery_index[0]
    history = target.loc[target.index < origin].dropna()

    result["price_lag24"] = _price_lag(target, delivery_index, 1)
    result["price_lag48"] = _price_lag(target, delivery_index, 2)
    result["price_lag168"] = _price_lag(target, delivery_index, 7)

    last_price = float(history.iloc[-1]) if not history.empty else np.nan
    short = history.iloc[-168:]
    long = history.iloc[-720:]
    previous_day = target.reindex(
        _previous_local_day_index(delivery_index, 1)
    ).dropna()

    short_median = _safe_stat(short, "median")
    long_median = _safe_stat(long, "median")

    scalar_features = {
        "origin_last_known_price": last_price,
        "regime_level_7d": short_median,
        "regime_volatility_7d": _safe_stat(short, "std"),
        "regime_negative_rate_30d": (
            float(long.lt(0.0).mean()) if not long.empty else np.nan
        ),
        "regime_spike_rate_30d": (
            float(long.abs().gt(spike_threshold).mean())
            if not long.empty
            else np.nan
        ),
        "regime_trend": (
            short_median - long_median
            if np.isfinite(short_median) and np.isfinite(long_median)
            else np.nan
        ),
        "previous_day_mean": _safe_stat(previous_day, "mean"),
        "previous_day_std": _safe_stat(previous_day, "std"),
        "previous_day_min": _safe_stat(previous_day, "min"),
        "previous_day_max": _safe_stat(previous_day, "max"),
        "previous_day_range": (
            float(previous_day.max() - previous_day.min())
            if not previous_day.empty
            else np.nan
        ),
        "previous_day_max_abs_ramp": (
            float(previous_day.diff().abs().max())
            if len(previous_day) > 1
            else np.nan
        ),
    }
    for name, value in scalar_features.items():
        result[name] = value

    return result


def _add_historical_signal_features(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    delivery_day: pd.Timestamp,
) -> pd.DataFrame:
    result = frame.copy()
    history = labels.loc[labels["delivery_day"] < delivery_day]
    if history.empty:
        for signal in LABEL_COLUMNS:
            result[f"history_{signal}_7d"] = np.nan
            result[f"history_{signal}_30d"] = np.nan
            result[f"history_{signal}_same_hour_30d"] = np.nan
        return result

    recent_7_start = delivery_day - pd.DateOffset(days=7)
    recent_30_start = delivery_day - pd.DateOffset(days=30)
    recent_7 = history.loc[history["delivery_day"] >= recent_7_start]
    recent_30 = history.loc[history["delivery_day"] >= recent_30_start]

    delivery_hours = result.index.hour
    for signal in LABEL_COLUMNS:
        result[f"history_{signal}_7d"] = (
            float(recent_7[signal].mean()) if not recent_7.empty else np.nan
        )
        result[f"history_{signal}_30d"] = (
            float(recent_30[signal].mean()) if not recent_30.empty else np.nan
        )
        by_hour = (
            recent_30.assign(_hour=recent_30.index.hour)
            .groupby("_hour")[signal]
            .mean()
        )
        result[f"history_{signal}_same_hour_30d"] = [
            float(by_hour.get(hour, np.nan)) for hour in delivery_hours
        ]

    return result


def _known_future_frame(
    data: ZoneData,
    delivery_index: pd.DatetimeIndex,
    excluded_signal_prefixes: Sequence[str],
) -> pd.DataFrame:
    columns = [
        column
        for column in data.known_future_columns
        if not any(prefix in column for prefix in excluded_signal_prefixes)
    ]
    missing = [
        timestamp
        for timestamp in delivery_index
        if timestamp not in data.model_context_covariates.index
    ]
    if missing:
        raise KeyError(
            "Les covariables futures ne couvrent pas la journée demandée. "
            f"Premiers timestamps absents : {missing[:3]}"
        )
    return data.model_context_covariates.loc[delivery_index, columns].copy()


def build_day_feature_frame(
    data: ZoneData,
    labels: pd.DataFrame,
    delivery_day: pd.Timestamp,
    config: Mapping[str, Any],
    excluded_signal_prefixes: Sequence[str] = ("order_",),
) -> pd.DataFrame:
    delivery_day = pd.Timestamp(delivery_day)
    if delivery_day.tzinfo is None:
        delivery_day = delivery_day.tz_localize(data.timezone)
    else:
        delivery_day = delivery_day.tz_convert(data.timezone)
    delivery_day = delivery_day.normalize()

    delivery_index = data.model_context_covariates.index[
        data.model_context_covariates.index.normalize() == delivery_day
    ]
    if len(delivery_index) != 24:
        raise ValueError(
            f"{delivery_day}: journée locale irrégulière ou incomplète "
            f"({len(delivery_index)} périodes au lieu de 24)."
        )

    frame = _known_future_frame(
        data,
        delivery_index,
        excluded_signal_prefixes,
    )
    frame["horizon_step"] = np.arange(1, 25, dtype=np.float32)
    frame["delivery_hour"] = delivery_index.hour.astype(np.float32)
    frame["delivery_month"] = delivery_index.month.astype(np.float32)

    spike_threshold = float(
        config.get("metrics", {}).get("extreme_threshold", 150.0)
    )
    frame = _add_path_features(frame)
    frame = _add_price_history_features(
        frame,
        data.target,
        delivery_index,
        spike_threshold,
    )
    frame = _add_historical_signal_features(
        frame,
        labels,
        delivery_day,
    )
    frame["timestamp"] = delivery_index
    frame["delivery_day"] = delivery_day
    return frame


def build_feature_table(
    data: ZoneData,
    labels: pd.DataFrame,
    config: Mapping[str, Any],
    include_live_day: bool = True,
) -> pd.DataFrame:
    historical_days = sorted(pd.unique(labels["delivery_day"]))
    days: list[pd.Timestamp] = [pd.Timestamp(day) for day in historical_days]

    if include_live_day:
        live_candidates = sorted(
            pd.unique(data.model_context_covariates.index.normalize())
        )
        last_target_day = data.target.index[-1].normalize()
        days.extend(
            pd.Timestamp(day)
            for day in live_candidates
            if pd.Timestamp(day) > last_target_day
        )

    frames: list[pd.DataFrame] = []
    for day in sorted(set(days)):
        try:
            frames.append(build_day_feature_frame(data, labels, day, config))
        except ValueError:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
