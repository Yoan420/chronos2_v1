from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronos2_modular.common import LOGGER, ZoneConfig, ZoneData, deep_get


def residual_mode_enabled(config: Mapping[str, Any]) -> bool:
    return bool(
        deep_get(config, "structural_model.residual_mode.enabled", False)
    )


def _price_alias(config: Mapping[str, Any]) -> str:
    return str(
        deep_get(
            config,
            "structural_model.residual_price_alias",
            "milp_structural_price",
        )
    )


def make_prepare_zone_data_for_residual(
    base_prepare: Callable[..., ZoneData],
) -> Callable[..., ZoneData]:
    def wrapped(
        zone: ZoneConfig,
        config: Mapping[str, Any],
        config_dir: Path,
        refresh: bool,
        output_dir: Path,
    ) -> ZoneData:
        data = base_prepare(
            zone,
            config,
            config_dir,
            refresh,
            output_dir,
        )
        if not residual_mode_enabled(config):
            return data

        alias = _price_alias(config)
        if alias not in data.covariates:
            raise KeyError(
                f"Mode résiduel : covariable structurelle absente : {alias}"
            )
        if alias not in data.model_context_covariates:
            raise KeyError(
                f"Mode résiduel : colonne modèle absente : {alias}"
            )

        original_target = pd.to_numeric(data.target, errors="coerce")
        structural_history = pd.to_numeric(
            data.covariates[alias], errors="coerce"
        ).reindex(data.target.index)
        missing = int(structural_history.isna().sum())
        if missing:
            raise ValueError(
                f"Mode résiduel : {missing} prix structurels manquants "
                "sur l'historique de la cible."
            )

        data.price_target_original = original_target.astype(np.float32)
        data.structural_price_alias = alias
        data.structural_price_model = pd.to_numeric(
            data.model_context_covariates[alias], errors="coerce"
        ).astype(np.float32)
        data.target = (
            original_target - structural_history
        ).astype(np.float32)
        data.target.name = "target"

        data.diagnostics["structural_residual_mode"] = {
            "enabled": True,
            "structural_price_alias": alias,
            "residual_mean": float(data.target.mean()),
            "residual_std": float(data.target.std()),
            "original_price_mean": float(original_target.mean()),
            "structural_price_mean": float(structural_history.mean()),
        }

        pd.DataFrame(
            {
                "timestamp": data.target.index,
                "actual_price": data.price_target_original.to_numpy(),
                "structural_price": structural_history.to_numpy(),
                "residual_target": data.target.to_numpy(),
            }
        ).to_csv(
            output_dir / "structural_residual_target.csv.gz",
            index=False,
            compression="gzip",
        )
        LOGGER.info(
            "[%s] Mode résiduel activé : target = prix - %s.",
            data.zone,
            alias,
        )
        return data

    return wrapped


def _structural_values(
    data: ZoneData,
    timestamps: pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    if not hasattr(data, "structural_price_model"):
        return np.full(len(timestamps), np.nan, dtype=np.float32)
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    series = data.structural_price_model
    if index.tz is None and series.index.tz is not None:
        index = index.tz_localize(series.index.tz)
    elif index.tz is not None and series.index.tz is not None:
        index = index.tz_convert(series.index.tz)
    return series.reindex(index).to_numpy(dtype=np.float32)


def restore_final_price_frame(
    frame: pd.DataFrame,
    data: ZoneData,
) -> pd.DataFrame:
    if not hasattr(data, "structural_price_model"):
        return frame

    result = frame.copy()
    structural = _structural_values(data, result["timestamp"])
    if np.isnan(structural).any():
        missing = int(np.isnan(structural).sum())
        raise ValueError(
            f"Prix structurel manquant pour {missing} prévisions résiduelles."
        )

    result["structural_price"] = structural
    if "actual" in result:
        result["actual_residual"] = result["actual"]
        result["actual"] = result["actual_residual"] + structural

    prediction_columns = [
        column
        for column in result.columns
        if column == "point" or column.startswith("q")
    ]
    for column in prediction_columns:
        result[f"{column}_residual"] = result[column]
        result[column] = result[column] + structural
    return result


def make_run_backtest_variant_for_residual(
    base_run: Callable[..., pd.DataFrame],
) -> Callable[..., pd.DataFrame]:
    def wrapped(
        data: ZoneData,
        runtime: Any,
        origins: Sequence[int],
        context_length: int,
        horizon: int,
        origin_batch_size: int,
        model_batch_size: int,
        with_covariates: bool,
        variant: str,
    ) -> pd.DataFrame:
        frame = base_run(
            data,
            runtime,
            origins,
            context_length,
            horizon,
            origin_batch_size,
            model_batch_size,
            with_covariates,
            variant,
        )
        return restore_final_price_frame(frame, data)

    return wrapped


def make_run_live_forecast_variant_for_residual(
    base_run: Callable[..., pd.DataFrame],
) -> Callable[..., pd.DataFrame]:
    def wrapped(
        data: ZoneData,
        runtime: Any,
        context_length: int,
        horizon: int,
        model_batch_size: int,
        with_covariates: bool,
        variant: str,
    ) -> pd.DataFrame:
        frame = base_run(
            data,
            runtime,
            context_length,
            horizon,
            model_batch_size,
            with_covariates,
            variant,
        )
        return restore_final_price_frame(frame, data)

    return wrapped
