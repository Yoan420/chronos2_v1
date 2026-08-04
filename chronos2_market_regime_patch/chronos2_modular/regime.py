from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get
from .data import prepare_zone_data as prepare_zone_data_base
from .forecasting import future_proxy_frame as future_proxy_frame_base


REGIME_COLUMNS = (
    "regime_level_7d",
    "regime_volatility_7d",
    "regime_negative_rate_30d",
    "regime_spike_rate_30d",
    "regime_trend",
)


def _settings(
    config: Mapping[str, Any],
    zone_name: str | None = None,
) -> dict[str, Any]:
    global_settings = deep_get(config, "market_regime", {}) or {}
    zone_settings = (
        deep_get(config, f"zones.{zone_name}.market_regime", {}) or {}
        if zone_name
        else {}
    )
    return {**global_settings, **zone_settings}


def market_regime_enabled(
    config: Mapping[str, Any],
    zone: ZoneConfig,
) -> bool:
    return bool(_settings(config, zone.zone).get("enabled", False))


def build_market_regime_frame(
    target: pd.Series,
    config: Mapping[str, Any],
    zone_name: str | None = None,
) -> pd.DataFrame:
    """Construit des régimes causaux, disponibles au timestamp de chaque ligne.

    La valeur au timestamp t est exclusivement calculée à partir des prix
    strictement antérieurs à t grâce à target.shift(1).
    """
    settings = _settings(config, zone_name)

    short_window = int(settings.get("short_window_hours", 168))
    long_window = int(settings.get("long_window_hours", 720))
    short_min = int(settings.get("short_min_periods", 72))
    long_min = int(settings.get("long_min_periods", 168))
    negative_threshold = float(settings.get("negative_threshold", 0.0))
    spike_threshold = float(settings.get("spike_threshold", 150.0))

    if short_window <= 0 or long_window <= 0:
        raise ValueError("Les fenêtres de régime doivent être positives.")
    if not 0 < short_min <= short_window:
        raise ValueError(
            "short_min_periods doit être compris entre 1 et short_window_hours."
        )
    if not 0 < long_min <= long_window:
        raise ValueError(
            "long_min_periods doit être compris entre 1 et long_window_hours."
        )

    past = pd.to_numeric(target, errors="coerce").shift(1)

    short_median = past.rolling(
        short_window,
        min_periods=short_min,
    ).median()
    long_median = past.rolling(
        long_window,
        min_periods=long_min,
    ).median()

    return pd.DataFrame(
        {
            "regime_level_7d": short_median,
            "regime_volatility_7d": past.rolling(
                short_window,
                min_periods=short_min,
            ).std(),
            "regime_negative_rate_30d": (
                past.lt(negative_threshold)
                .rolling(long_window, min_periods=long_min)
                .mean()
            ),
            "regime_spike_rate_30d": (
                past.abs()
                .gt(spike_threshold)
                .rolling(long_window, min_periods=long_min)
                .mean()
            ),
            "regime_trend": short_median - long_median,
        },
        index=target.index,
        dtype=np.float32,
    )


def regime_at_origin(
    target_history: pd.Series,
    config: Mapping[str, Any],
    zone_name: str | None = None,
) -> dict[str, float]:
    """Calcule la valeur connue exactement à une origine de prévision.

    target_history doit s'arrêter au dernier prix connu. Contrairement aux
    lignes historiques, aucun shift supplémentaire n'est nécessaire : toute
    la série fournie appartient déjà au passé de l'origine.
    """
    settings = _settings(config, zone_name)

    short_window = int(settings.get("short_window_hours", 168))
    long_window = int(settings.get("long_window_hours", 720))
    short_min = int(settings.get("short_min_periods", 72))
    long_min = int(settings.get("long_min_periods", 168))
    negative_threshold = float(settings.get("negative_threshold", 0.0))
    spike_threshold = float(settings.get("spike_threshold", 150.0))

    history = pd.to_numeric(target_history, errors="coerce").dropna()
    short_history = history.iloc[-short_window:]
    long_history = history.iloc[-long_window:]

    short_median = (
        float(short_history.median())
        if len(short_history) >= short_min
        else np.nan
    )
    long_median = (
        float(long_history.median())
        if len(long_history) >= long_min
        else np.nan
    )

    return {
        "regime_level_7d": short_median,
        "regime_volatility_7d": (
            float(short_history.std())
            if len(short_history) >= short_min
            else np.nan
        ),
        "regime_negative_rate_30d": (
            float(long_history.lt(negative_threshold).mean())
            if len(long_history) >= long_min
            else np.nan
        ),
        "regime_spike_rate_30d": (
            float(long_history.abs().gt(spike_threshold).mean())
            if len(long_history) >= long_min
            else np.nan
        ),
        "regime_trend": (
            short_median - long_median
            if np.isfinite(short_median) and np.isfinite(long_median)
            else np.nan
        ),
    }


def prepare_zone_data_with_regime(
    zone: ZoneConfig,
    config: Mapping[str, Any],
    config_dir,
    refresh: bool,
    output_dir,
) -> ZoneData:
    """Appelle le pipeline existant puis ajoute les régimes à ZoneData."""
    data = prepare_zone_data_base(
        zone,
        config,
        config_dir,
        refresh,
        output_dir,
    )

    if not market_regime_enabled(config, zone):
        return data

    # ZoneData n'utilise pas slots=True : cette configuration peut être
    # attachée pour recalculer correctement le régime à chaque origine.
    data.market_regime_config = config

    regime = build_market_regime_frame(
        data.target,
        config,
        zone.zone,
    )
    future_index = data.model_context_covariates.index.difference(
        data.target.index
    )
    live_values = regime_at_origin(
        data.target,
        config,
        zone.zone,
    )

    for column in REGIME_COLUMNS:
        # Covariable passée : valeur à t calculée sur target[:t].
        data.covariates[column] = regime[column]
        data.model_context_covariates.loc[
            data.target.index, column
        ] = regime[column]
        data.model_context_covariates.loc[
            future_index, column
        ] = np.nan

        # Covariable future connue par construction : constante sur les
        # 24 pas et égale au régime disponible à l'origine.
        known_column = f"known_{column}_persistence"
        data.model_context_covariates.loc[
            data.target.index, known_column
        ] = regime[column]
        data.model_context_covariates.loc[
            future_index, known_column
        ] = live_values[column]
        data.known_future_columns.append(known_column)

    data.model_context_covariates = (
        data.model_context_covariates.astype(np.float32)
    )

    coverage = regime.notna().mean()
    coverage_additions = pd.DataFrame(
        [
            {
                "zone": data.zone,
                "alias": column,
                "series": "derived_from_target_shift_1",
                "coverage_exact": float(coverage[column]),
                "coverage_after_fill": float(coverage[column]),
                "missing_after_fill": int(regime[column].isna().sum()),
            }
            for column in REGIME_COLUMNS
        ]
    )
    data.coverage = pd.concat(
        [data.coverage, coverage_additions],
        ignore_index=True,
    )

    manifest_additions = pd.DataFrame(
        [
            {
                "alias": column,
                "role": "derived_market_regime",
                "series": "target.shift(1)",
                "description": (
                    "Régime de marché causal dérivé des prix passés"
                ),
                "future_strategies": "persistence_at_origin",
                "known_future": False,
                "source": "derived_target",
            }
            for column in REGIME_COLUMNS
        ]
    )
    data.input_manifest = pd.concat(
        [data.input_manifest, manifest_additions],
        ignore_index=True,
    )

    data.diagnostics["market_regime"] = {
        "enabled": True,
        "columns": list(REGIME_COLUMNS),
        "causal_shift_hours": 1,
        "future_strategy": "persistence_at_origin",
        "settings": _settings(config, zone.zone),
        "coverage": {
            column: float(coverage[column])
            for column in REGIME_COLUMNS
        },
    }

    # Réécrit les diagnostics produits par prepare_zone_data afin qu'ils
    # incluent aussi les nouvelles variables dérivées.
    output_dir.mkdir(parents=True, exist_ok=True)
    data.coverage.to_csv(
        output_dir / "input_coverage.csv",
        index=False,
    )
    data.input_manifest.to_csv(
        output_dir / "input_manifest.csv",
        index=False,
    )
    pd.concat(
        [data.target.rename("target"), data.covariates],
        axis=1,
    ).reset_index(names="timestamp").to_csv(
        output_dir / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )
    data.model_context_covariates.reset_index(
        names="timestamp"
    ).to_csv(
        output_dir / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )

    LOGGER.info(
        "[%s] Régimes de marché activés : %s",
        data.zone,
        ", ".join(REGIME_COLUMNS),
    )
    return data


def future_proxy_frame_with_regime(
    data: ZoneData,
    future_index: pd.DatetimeIndex,
    origin_position: int,
) -> pd.DataFrame:
    """Construit les covariables futures pour une origine de backtest/live."""
    frame = future_proxy_frame_base(
        data,
        future_index,
        origin_position,
    )

    regime_columns = [
        column
        for column in data.known_future_columns
        if column.startswith("known_regime_")
        and column.endswith("_persistence")
    ]
    if not regime_columns:
        return frame

    config = getattr(data, "market_regime_config", {})
    origin_values = regime_at_origin(
        data.target.iloc[:origin_position],
        config,
        data.zone,
    )

    for column in regime_columns:
        alias = (
            column.removeprefix("known_")
            .removesuffix("_persistence")
        )
        frame[column] = origin_values.get(alias, np.nan)

    return frame[data.known_future_columns].astype(np.float32)
