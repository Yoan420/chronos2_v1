from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .labels import LABEL_COLUMNS


OUTPUT_ALIASES: Mapping[str, str] = {
    "order_ramp_pressure": "fr_order_ramp_pressure_fcst",
    "order_plateau_pressure": "fr_order_plateau_pressure_fcst",
    "order_jump_probability": "fr_order_jump_probability_fcst",
    "order_block_pressure": "fr_order_block_pressure_fcst",
    "order_gradient_binding_probability": (
        "fr_order_gradient_binding_probability_fcst"
    ),
}


def snapshot_time_for_delivery(
    delivery_day: pd.Timestamp,
    timezone: str,
    origin_hour: int,
    origin_minute: int,
) -> pd.Timestamp:
    day = pd.Timestamp(delivery_day)
    if day.tzinfo is None:
        day = day.tz_localize(timezone)
    else:
        day = day.tz_convert(timezone)
    local_snapshot = (
        day.normalize()
        - pd.DateOffset(days=1)
        + pd.Timedelta(hours=origin_hour, minutes=origin_minute)
    )
    return local_snapshot.tz_convert("UTC")


def predictions_to_pit_rows(
    prediction_frame: pd.DataFrame,
    timezone: str,
    origin_hour: int,
    origin_minute: int,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    timestamps = pd.DatetimeIndex(prediction_frame["timestamp"])
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timezone)
    else:
        timestamps = timestamps.tz_convert(timezone)

    delivery_days = pd.DatetimeIndex(prediction_frame["delivery_day"])
    if delivery_days.tz is None:
        delivery_days = delivery_days.tz_localize(timezone)
    else:
        delivery_days = delivery_days.tz_convert(timezone)

    snapshot_times = [
        snapshot_time_for_delivery(
            day,
            timezone,
            origin_hour,
            origin_minute,
        )
        for day in delivery_days
    ]

    for signal in LABEL_COLUMNS:
        alias = OUTPUT_ALIASES[signal]
        frame = pd.DataFrame(
            {
                "value_time_utc": timestamps.tz_convert("UTC"),
                "snapshot_time_utc": pd.DatetimeIndex(snapshot_times),
                "revision_time_utc": pd.DatetimeIndex(snapshot_times),
                "value": prediction_frame[signal].to_numpy(dtype=float),
                "model_train_end_day": prediction_frame[
                    "model_train_end_day"
                ].astype(str).to_numpy(),
                "model_block_start_day": prediction_frame[
                    "model_block_start_day"
                ].astype(str).to_numpy(),
            }
        )
        result[alias] = frame.sort_values(
            ["value_time_utc", "snapshot_time_utc"]
        ).reset_index(drop=True)
    return result


def write_pit_vintages(
    frames: Mapping[str, pd.DataFrame],
    directory: Path,
    append: bool,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for alias, frame in frames.items():
        path = directory / f"{alias}.parquet"
        combined = frame
        if append and path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, frame], ignore_index=True)
        combined = (
            combined.sort_values(
                ["value_time_utc", "snapshot_time_utc", "revision_time_utc"]
            )
            .drop_duplicates(
                subset=["value_time_utc", "snapshot_time_utc"],
                keep="last",
            )
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False)
        paths[alias] = path
    return paths
