from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    CALENDAR_COLUMNS,
    DEFAULT_QUANTILES,
    KNOWN_FUTURE_COLUMN_PATTERN,
    LOGGER,
    ModelRuntime,
    ZoneData,
    configure_huggingface_ssl,
    deep_get,
    parse_future_lag_hours,
    resolve_device,
)
from .data import calendar_frame


def load_model(
    config: Mapping[str, Any],
    requested_device: str | None,
    local_files_only: bool,
) -> ModelRuntime:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            'Chronos-2 est requis : pip install "chronos-forecasting>=2.2.2,<3"'
        ) from exc
    model_id = str(deep_get(config, "model.model_id", "amazon/chronos-2"))
    device_name = requested_device or str(
        deep_get(config, "model.device", "auto")
    )
    device, dtype = resolve_device(device_name)
    kwargs: dict[str, Any] = {
        "device_map": device,
        "local_files_only": bool(
            local_files_only
            or deep_get(config, "model.local_files_only", False)
        ),
    }
    attn = str(deep_get(config, "model.attn_implementation", "auto"))
    if attn != "auto":
        kwargs["attn_implementation"] = attn
    configure_huggingface_ssl()
    LOGGER.info("Chargement %s sur %s...", model_id, device)
    started = time.perf_counter()
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **kwargs,
        )
    LOGGER.info(
        "Modèle chargé en %.2f secondes.",
        time.perf_counter() - started,
    )
    return ModelRuntime(
        pipeline=pipeline,
        model_id=model_id,
        device=device,
        dtype=dtype,
    )


def select_origins(
    target: pd.Series,
    context_length: int,
    horizon: int,
    windows: int,
    origin_hour: int | None,
    skip_irregular_days: bool,
) -> list[int]:
    day_counts = (
        pd.Series(1, index=target.index.normalize())
        .groupby(level=0)
        .sum()
        .to_dict()
    )
    candidates: list[int] = []
    for position in range(
        context_length,
        len(target) - horizon + 1,
    ):
        timestamp = target.index[position]
        if origin_hour is not None and timestamp.hour != origin_hour:
            continue
        if skip_irregular_days and origin_hour == 0:
            target_index = target.index[position : position + horizon]
            day = timestamp.normalize()
            if (
                int(day_counts.get(day, 0)) != horizon
                or len(target_index) != horizon
                or not (target_index.normalize() == day).all()
            ):
                continue
        candidates.append(position)
    if not candidates:
        raise ValueError("Aucune origine de backtest admissible.")
    return candidates[-windows:]


def batched(
    values: Sequence[int],
    batch_size: int,
) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def to_model_timestamp_index(
    index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(index)
    if values.tz is not None:
        values = values.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(
        values.to_numpy(dtype="datetime64[ns]")
    )


def prepare_chronos_frame(
    frame: pd.DataFrame | None,
    frame_name: str,
) -> pd.DataFrame | None:
    if frame is None:
        return None
    result = frame.copy()
    timestamps = pd.to_datetime(
        result["timestamp"],
        errors="raise",
        utc=True,
    )
    result["timestamp"] = (
        timestamps.dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    result = result.sort_values(
        ["item_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    array = result["timestamp"].to_numpy()
    if array.dtype != np.dtype("datetime64[ns]"):
        raise TypeError(
            f"{frame_name}: timestamp incompatible : {array.dtype}"
        )
    array.view("int64")
    return result


def restore_local_timestamp(
    frame: pd.DataFrame,
    timezone: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    ).dt.tz_convert(timezone)
    return result


def future_proxy_frame(
    data: ZoneData,
    future_index: pd.DatetimeIndex,
    origin_position: int,
) -> pd.DataFrame:
    frame = calendar_frame(future_index)
    for column in data.known_future_columns:
        if column in CALENDAR_COLUMNS:
            continue
        match = KNOWN_FUTURE_COLUMN_PATTERN.match(column)
        if not match:
            frame[column] = np.nan
            continue
        alias, strategy = match.groups()
        if alias not in data.covariates:
            frame[column] = np.nan
            continue
        series = data.covariates[alias]
        lag_hours = parse_future_lag_hours(strategy)
        if lag_hours is not None:
            frame[column] = series.reindex(
                future_index - pd.Timedelta(hours=lag_hours)
            ).to_numpy()
        elif strategy == "persistence":
            history = series.iloc[:origin_position].dropna()
            value = (
                float(history.iloc[-1])
                if not history.empty
                else math.nan
            )
            frame[column] = value
        elif strategy == "oracle":
            frame[column] = series.reindex(future_index).to_numpy()
    for column in data.known_future_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[data.known_future_columns].astype(np.float32)


def build_origin_frames(
    data: ZoneData,
    origin: int,
    context_length: int,
    horizon: int,
    item_id: str,
    with_covariates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    start = max(0, origin - context_length)
    context_index = data.target.index[start:origin]
    future_index = data.target.index[origin : origin + horizon]
    context = pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": to_model_timestamp_index(context_index),
            "target": data.target.loc[context_index].to_numpy(
                dtype=np.float32
            ),
        }
    )
    future: pd.DataFrame | None = None
    if with_covariates and not data.model_context_covariates.empty:
        context = pd.concat(
            [
                context.reset_index(drop=True),
                data.model_context_covariates.loc[
                    context_index
                ].reset_index(drop=True),
            ],
            axis=1,
        )
        if data.known_future_columns:
            future_values = future_proxy_frame(
                data,
                future_index,
                origin,
            )
            future = pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": to_model_timestamp_index(future_index),
                }
            )
            future = pd.concat(
                [
                    future.reset_index(drop=True),
                    future_values.reset_index(drop=True),
                ],
                axis=1,
            )
    metadata = pd.DataFrame(
        {
            "item_id": item_id,
            "zone": data.zone,
            "origin_position": origin,
            "origin_timestamp": data.target.index[origin],
            "timestamp": to_model_timestamp_index(future_index),
            "horizon_step": np.arange(1, len(future_index) + 1),
            "actual": data.target.iloc[
                origin : origin + horizon
            ].to_numpy(dtype=np.float32),
        }
    )
    return context, future, metadata


def build_live_frames(
    data: ZoneData,
    context_length: int,
    horizon: int,
    item_id: str,
    with_covariates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    origin = len(data.target)
    start = max(0, origin - context_length)
    context_index = data.target.index[start:origin]
    offset = pd.tseries.frequencies.to_offset(data.frequency)
    future_index = pd.date_range(
        data.target.index[-1] + offset,
        periods=horizon,
        freq=data.frequency,
        tz=data.target.index.tz,
    )
    context = pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": to_model_timestamp_index(context_index),
            "target": data.target.loc[context_index].to_numpy(
                dtype=np.float32
            ),
        }
    )
    future = None
    if with_covariates and not data.model_context_covariates.empty:
        context = pd.concat(
            [
                context.reset_index(drop=True),
                data.model_context_covariates.loc[
                    context_index
                ].reset_index(drop=True),
            ],
            axis=1,
        )
        if data.known_future_columns:
            future_values = future_proxy_frame(
                data,
                future_index,
                origin,
            )
            future = pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": to_model_timestamp_index(future_index),
                }
            )
            future = pd.concat(
                [
                    future.reset_index(drop=True),
                    future_values.reset_index(drop=True),
                ],
                axis=1,
            )
    return context, future


def normalize_prediction_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.loc[frame["target_name"].eq("target")].copy()
    result = result.rename(columns={"predictions": "point"})
    for level in DEFAULT_QUANTILES:
        result = result.rename(
            columns={str(level): f"q{int(level * 100):02d}"}
        )
    return result


def predict_df_batch(
    runtime: ModelRuntime,
    contexts: list[pd.DataFrame],
    futures: list[pd.DataFrame],
    metadata: list[pd.DataFrame],
    data: ZoneData,
    context_length: int,
    horizon: int,
    model_batch_size: int,
) -> pd.DataFrame:
    context_df = prepare_chronos_frame(
        pd.concat(contexts, ignore_index=True),
        "context_df",
    )
    future_df = (
        prepare_chronos_frame(
            pd.concat(futures, ignore_index=True),
            "future_df",
        )
        if futures
        else None
    )
    n_variates = len(
        [
            column
            for column in context_df.columns
            if column not in {"item_id", "timestamp"}
        ]
    )
    prediction = runtime.pipeline.predict_df(
        context_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=horizon,
        quantile_levels=list(DEFAULT_QUANTILES),
        batch_size=max(model_batch_size, n_variates),
        context_length=context_length,
        cross_learning=False,
        validate_inputs=False,
        freq=data.frequency,
    )
    prediction = normalize_prediction_columns(prediction)
    meta = pd.concat(metadata, ignore_index=True)
    merged = meta.merge(
        prediction.drop(columns=["target_name"]),
        on=["item_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if merged["q50"].isna().any():
        raise RuntimeError("Prévisions Chronos-2 incomplètes.")
    return restore_local_timestamp(merged, data.timezone)


def run_backtest_variant(
    data: ZoneData,
    runtime: ModelRuntime,
    origins: Sequence[int],
    context_length: int,
    horizon: int,
    origin_batch_size: int,
    model_batch_size: int,
    with_covariates: bool,
    variant: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch_number, origin_batch in enumerate(
        batched(origins, origin_batch_size),
        start=1,
    ):
        contexts: list[pd.DataFrame] = []
        futures: list[pd.DataFrame] = []
        metadata: list[pd.DataFrame] = []
        for origin in origin_batch:
            context, future, meta = build_origin_frames(
                data,
                origin,
                context_length,
                horizon,
                f"{data.zone}_{variant}_{origin}",
                with_covariates,
            )
            contexts.append(context)
            if future is not None:
                futures.append(future)
            metadata.append(meta)
        frame = predict_df_batch(
            runtime,
            contexts,
            futures,
            metadata,
            data,
            context_length,
            horizon,
            model_batch_size,
        )
        frame["variant"] = variant
        frame["model"] = runtime.model_id
        frames.append(frame)
        LOGGER.info(
            "[%s] %s : %d/%d origines.",
            data.zone,
            variant,
            min(batch_number * origin_batch_size, len(origins)),
            len(origins),
        )
    return pd.concat(frames, ignore_index=True)


def run_live_forecast_variant(
    data: ZoneData,
    runtime: ModelRuntime,
    context_length: int,
    horizon: int,
    model_batch_size: int,
    with_covariates: bool,
    variant: str,
) -> pd.DataFrame:
    context, future = build_live_frames(
        data,
        context_length,
        horizon,
        f"{data.zone}_{variant}_live",
        with_covariates,
    )
    context = prepare_chronos_frame(context, "live_context")
    future = prepare_chronos_frame(future, "live_future")
    n_variates = len(
        [
            column
            for column in context.columns
            if column not in {"item_id", "timestamp"}
        ]
    )
    prediction = runtime.pipeline.predict_df(
        context,
        future_df=future,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=horizon,
        quantile_levels=list(DEFAULT_QUANTILES),
        batch_size=max(model_batch_size, n_variates),
        context_length=context_length,
        cross_learning=False,
        validate_inputs=False,
        freq=data.frequency,
    )
    prediction = normalize_prediction_columns(prediction)
    prediction = restore_local_timestamp(
        prediction,
        data.timezone,
    )
    prediction["zone"] = data.zone
    prediction["variant"] = variant
    prediction["model"] = runtime.model_id
    return prediction
