from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data

from .features import build_feature_table
from .labels import LABEL_COLUMNS, build_ex_post_labels
from .modeling import (
    fit_multioutput_model,
    predict_scores,
    save_model_bundle,
)
from .pit import OUTPUT_ALIASES, predictions_to_pit_rows, write_pit_vintages

LOGGER = logging.getLogger("chronos2_order_signals")


def _origin_clock(config: Mapping[str, Any]) -> tuple[int, int]:
    raw = str(deep_get(config, "data.forecast_origin_local_time", "08:00"))
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw.strip())
    if not match:
        raise ValueError("forecast_origin_local_time doit être HH:MM.")
    return int(match.group(1)), int(match.group(2))


def _config_without_output_signals(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(dict(config))
    zones = cleaned.get("zones", {})
    if isinstance(zones, Mapping):
        for zone_payload in zones.values():
            if not isinstance(zone_payload, dict):
                continue
            covariates = zone_payload.get("covariates", {})
            if not isinstance(covariates, dict):
                continue
            for alias in OUTPUT_ALIASES.values():
                if alias in covariates and isinstance(covariates[alias], dict):
                    covariates[alias]["enabled"] = False
    pit_files = cleaned.get("data", {}).get("pit_files", {})
    if isinstance(pit_files, dict):
        for alias in OUTPUT_ALIASES.values():
            pit_files.pop(alias, None)
    return cleaned


def _join_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_values = labels[list(LABEL_COLUMNS)].copy()
    joined = features.join(label_values, how="left")
    return joined


def _rolling_oof_predictions(
    dataset: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
    start_day: pd.Timestamp | None,
    end_day: pd.Timestamp | None,
) -> pd.DataFrame:
    settings = config.get("order_signals", {}).get("backfill", {})
    min_train_days = int(settings.get("min_train_days", 365))
    train_window_days = int(settings.get("train_window_days", 730))
    retrain_every_days = int(settings.get("retrain_every_days", 7))
    label_lag_days = int(settings.get("label_availability_lag_days", 1))

    all_days = sorted(
        pd.Timestamp(day) for day in pd.unique(dataset["delivery_day"])
    )
    label_days = sorted(
        pd.Timestamp(day)
        for day in pd.unique(
            dataset.loc[
                dataset[list(LABEL_COLUMNS)].notna().all(axis=1),
                "delivery_day",
            ]
        )
    )
    if len(label_days) <= min_train_days:
        raise ValueError(
            f"Historique insuffisant : {len(label_days)} jours avec labels, "
            f"minimum demandé={min_train_days}."
        )

    earliest_prediction_day = label_days[min_train_days]
    if start_day is not None:
        earliest_prediction_day = max(earliest_prediction_day, start_day)
    latest_prediction_day = label_days[-1]
    if end_day is not None:
        latest_prediction_day = min(latest_prediction_day, end_day)

    prediction_days = [
        day
        for day in all_days
        if earliest_prediction_day <= day <= latest_prediction_day
        and day in set(label_days)
    ]
    if not prediction_days:
        raise ValueError("Aucune journée OOF dans la période demandée.")

    rows: list[pd.DataFrame] = []
    cursor = 0
    models_dir = output_dir / "models_oof"

    while cursor < len(prediction_days):
        block_days = prediction_days[cursor : cursor + retrain_every_days]
        block_start = block_days[0]
        train_end_day = block_start - pd.DateOffset(days=label_lag_days)
        train_start_day = (
            train_end_day - pd.DateOffset(days=train_window_days - 1)
            if train_window_days > 0
            else dataset["delivery_day"].min()
        )
        training = dataset.loc[
            (dataset["delivery_day"] >= train_start_day)
            & (dataset["delivery_day"] <= train_end_day)
            & dataset[list(LABEL_COLUMNS)].notna().all(axis=1)
        ].copy()
        if training["delivery_day"].nunique() < min_train_days:
            cursor += len(block_days)
            continue

        model, features, metadata = fit_multioutput_model(training, config)
        metadata.update(
            {
                "model_block_start_day": str(block_start),
                "model_train_start_day": str(training["delivery_day"].min()),
                "model_train_end_day": str(training["delivery_day"].max()),
                "strict_oof": True,
            }
        )
        stem = pd.Timestamp(block_start).strftime("model_%Y%m%d")
        save_model_bundle(model, metadata, models_dir, stem)

        prediction_input = dataset.loc[
            dataset["delivery_day"].isin(block_days)
        ].copy()
        predictions = predict_scores(model, prediction_input, features)
        prediction_output = prediction_input[
            ["timestamp", "delivery_day"]
        ].copy()
        for signal in LABEL_COLUMNS:
            prediction_output[signal] = predictions[signal].to_numpy()
        prediction_output["model_train_end_day"] = training[
            "delivery_day"
        ].max()
        prediction_output["model_block_start_day"] = block_start
        rows.append(prediction_output)
        cursor += len(block_days)

    if not rows:
        raise ValueError("Le backfill OOF n'a produit aucune prévision.")
    return pd.concat(rows, ignore_index=True).sort_values("timestamp")


def _fit_live_prediction(
    dataset: pd.DataFrame,
    config: Mapping[str, Any],
    delivery_day: pd.Timestamp,
    output_dir: Path,
) -> pd.DataFrame:
    settings = config.get("order_signals", {}).get("backfill", {})
    train_window_days = int(settings.get("train_window_days", 730))
    label_lag_days = int(settings.get("label_availability_lag_days", 1))
    train_end_day = delivery_day - pd.DateOffset(days=label_lag_days)
    train_start_day = (
        train_end_day - pd.DateOffset(days=train_window_days - 1)
        if train_window_days > 0
        else dataset["delivery_day"].min()
    )
    training = dataset.loc[
        (dataset["delivery_day"] >= train_start_day)
        & (dataset["delivery_day"] <= train_end_day)
        & dataset[list(LABEL_COLUMNS)].notna().all(axis=1)
    ].copy()
    prediction_input = dataset.loc[
        dataset["delivery_day"] == delivery_day
    ].copy()
    if len(prediction_input) != 24:
        raise ValueError(
            f"Features live absentes pour {delivery_day}: "
            f"{len(prediction_input)} lignes."
        )

    model, features, metadata = fit_multioutput_model(training, config)
    metadata.update(
        {
            "model_block_start_day": str(delivery_day),
            "model_train_start_day": str(training["delivery_day"].min()),
            "model_train_end_day": str(training["delivery_day"].max()),
            "strict_oof": True,
            "live": True,
        }
    )
    save_model_bundle(
        model,
        metadata,
        output_dir / "models_live",
        pd.Timestamp(delivery_day).strftime("live_%Y%m%d"),
    )
    prediction = predict_scores(model, prediction_input, features)
    result = prediction_input[["timestamp", "delivery_day"]].copy()
    for signal in LABEL_COLUMNS:
        result[signal] = prediction[signal].to_numpy()
    result["model_train_end_day"] = training["delivery_day"].max()
    result["model_block_start_day"] = delivery_day
    return result


def _metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    truth = labels[list(LABEL_COLUMNS)].copy()
    merged = predictions.set_index("timestamp").join(
        truth,
        how="inner",
        lsuffix="_pred",
        rsuffix="_actual",
    )
    rows = []
    for signal in LABEL_COLUMNS:
        predicted = merged[f"{signal}_pred"].astype(float)
        actual = merged[f"{signal}_actual"].astype(float)
        error = predicted - actual
        rows.append(
            {
                "signal": signal,
                "n": int(len(merged)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(error.mean()),
                "correlation": float(predicted.corr(actual)),
                "top_decile_actual_mean": float(
                    actual.loc[predicted >= predicted.quantile(0.90)].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(
    config_path: Path,
    mode: str,
    zone_name: str,
    refresh_data: bool,
    start_day: str | None,
    end_day: str | None,
    delivery_day: str | None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml(config_path)
    set_reproducibility(int(deep_get(config, "model.seed", 42)))

    cleaned_config = _config_without_output_signals(config)
    zone_configs = build_zone_configs(
        cleaned_config,
        [zone_name],
        include_covariates=None,
        exclude_covariates=list(OUTPUT_ALIASES.values()),
    )
    zone = zone_configs[0]
    project_root = resolve_path(
        deep_get(cleaned_config, "data.project_root", "."),
        config_path.parent,
    )
    output_dir = resolve_path(
        deep_get(
            cleaned_config,
            "order_signals.output_dir",
            "runs/order_signals",
        ),
        project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_zone_data(
        zone,
        cleaned_config,
        config_path.parent,
        refresh_data,
        output_dir / "prepared_inputs",
    )
    labels = build_ex_post_labels(data.target, cleaned_config)
    labels.reset_index(names="timestamp").to_parquet(
        output_dir / "ex_post_labels.parquet", index=False
    )

    features = build_feature_table(
        data,
        labels,
        cleaned_config,
        include_live_day=True,
    )
    if features.empty:
        raise RuntimeError("Aucune feature auxiliaire construite.")
    features.reset_index(drop=True).to_parquet(
        output_dir / "auxiliary_features.parquet", index=False
    )
    dataset = _join_labels(features, labels)

    origin_hour, origin_minute = _origin_clock(cleaned_config)
    pit_dir = resolve_path(
        deep_get(
            cleaned_config,
            "data.pit_vintage_dir",
            "data/pit/vintages",
        ),
        project_root,
    )
    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "pit_dir": str(pit_dir),
    }

    if mode in {"backfill", "backfill-live"}:
        timezone = zone.timezone
        parsed_start = (
            pd.Timestamp(start_day).tz_localize(timezone).normalize()
            if start_day
            else None
        )
        parsed_end = (
            pd.Timestamp(end_day).tz_localize(timezone).normalize()
            if end_day
            else None
        )
        oof = _rolling_oof_predictions(
            dataset,
            cleaned_config,
            output_dir,
            parsed_start,
            parsed_end,
        )
        oof.to_parquet(output_dir / "oof_signal_predictions.parquet", index=False)
        metric_frame = _metrics(oof, labels)
        metric_frame.to_csv(output_dir / "oof_signal_metrics.csv", index=False)
        pit_frames = predictions_to_pit_rows(
            oof,
            zone.timezone,
            origin_hour,
            origin_minute,
        )
        paths = write_pit_vintages(pit_frames, pit_dir, append=False)
        result["backfill_rows"] = int(len(oof))
        result["backfill_pit_files"] = {
            alias: str(path) for alias, path in paths.items()
        }

    if mode in {"live", "backfill-live"}:
        timezone = zone.timezone
        if delivery_day:
            live_day = pd.Timestamp(delivery_day)
            live_day = (
                live_day.tz_localize(timezone)
                if live_day.tzinfo is None
                else live_day.tz_convert(timezone)
            ).normalize()
        else:
            live_day = data.target.index[-1].normalize() + pd.DateOffset(days=1)
        live = _fit_live_prediction(
            dataset,
            cleaned_config,
            live_day,
            output_dir,
        )
        live.to_csv(output_dir / "live_signal_forecast.csv", index=False)
        pit_frames = predictions_to_pit_rows(
            live,
            zone.timezone,
            origin_hour,
            origin_minute,
        )
        paths = write_pit_vintages(pit_frames, pit_dir, append=True)
        result["live_delivery_day"] = str(live_day)
        result["live_pit_files"] = {
            alias: str(path) for alias, path in paths.items()
        }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return result
