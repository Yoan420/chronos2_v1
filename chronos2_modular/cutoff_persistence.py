from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get


def _settings(config: Mapping[str, Any], zone: str) -> dict[str, Any]:
    global_settings = dict(
        deep_get(config, "exogenous_extensions.cutoff_persistence", {}) or {}
    )
    zone_settings = (
        deep_get(
            config,
            f"zones.{zone}.exogenous_extensions.cutoff_persistence",
            {},
        )
        or {}
    )
    return {**global_settings, **zone_settings}


def parse_clock(raw: str) -> tuple[int, int]:
    parts = str(raw).strip().split(":")
    if len(parts) != 2:
        raise ValueError("cutoff_local_time doit être au format HH:MM.")
    hour, minute = int(parts[0]), int(parts[1])
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("cutoff_local_time doit être une heure valide.")
    return hour, minute


def daily_cutoff_index(
    index: pd.DatetimeIndex,
    *,
    hour: int,
    minute: int,
) -> pd.DatetimeIndex:
    local = pd.DatetimeIndex(index)
    return (
        local.normalize()
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=hour, minutes=minute)
    )


def asof_values(
    series: pd.Series,
    cutoffs: pd.DatetimeIndex,
    *,
    max_age_hours: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.index.duplicated().any():
        clean = clean.groupby(level=0).last()

    values = np.full(len(cutoffs), np.nan, dtype=np.float32)
    ages = np.full(len(cutoffs), np.nan, dtype=np.float32)
    if clean.empty:
        return values, ages

    source_index = pd.DatetimeIndex(clean.index)
    if source_index.tz is None:
        raise ValueError(
            "La série cutoff_persistence doit avoir un index timezone-aware."
        )

    cutoff_index = pd.DatetimeIndex(cutoffs).tz_convert(source_index.tz)
    source_ns = source_index.asi8
    cutoff_ns = cutoff_index.asi8
    positions = np.searchsorted(source_ns, cutoff_ns, side="right") - 1
    valid = positions >= 0

    if valid.any():
        selected_positions = positions[valid]
        values[valid] = clean.to_numpy(dtype=np.float32)[selected_positions]
        age_hours = (
            cutoff_ns[valid] - source_ns[selected_positions]
        ) / 3_600_000_000_000.0
        ages[valid] = age_hours.astype(np.float32)

        if max_age_hours is not None:
            too_old = np.zeros(len(cutoffs), dtype=bool)
            too_old[valid] = age_hours > float(max_age_hours)
            values[too_old] = np.nan
            ages[too_old] = np.nan

    return values, ages


def cutoff_persistence_frame(
    series: pd.Series,
    index: pd.DatetimeIndex,
    *,
    cutoff_local_time: str,
    max_age_hours: float | None,
) -> pd.DataFrame:
    hour, minute = parse_clock(cutoff_local_time)
    cutoffs = daily_cutoff_index(index, hour=hour, minute=minute)
    values, ages = asof_values(
        series,
        cutoffs,
        max_age_hours=max_age_hours,
    )
    return pd.DataFrame(
        {"value": values, "age_hours": ages},
        index=index,
    )


def add_cutoff_persistence_features(
    data: ZoneData,
    config: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    settings = _settings(config, data.zone)
    if not settings.get("enabled", False):
        return []

    aliases_raw = settings.get("aliases", [])
    if isinstance(aliases_raw, str):
        aliases = [aliases_raw]
    elif isinstance(aliases_raw, Mapping):
        aliases = [
            str(alias)
            for alias, enabled in aliases_raw.items()
            if bool(enabled)
        ]
    else:
        aliases = [str(alias) for alias in aliases_raw]

    cutoff_time = str(
        settings.get(
            "cutoff_local_time",
            deep_get(config, "data.forecast_origin_local_time", "08:00"),
        )
    )
    max_age_raw = settings.get("max_age_hours", 72)
    max_age_hours = (
        None if max_age_raw in (None, "", "none") else float(max_age_raw)
    )
    include_age = bool(settings.get("include_age_hours", True))

    added: list[str] = []
    attached: dict[str, dict[str, Any]] = {}

    for alias in aliases:
        if alias not in data.covariates:
            LOGGER.warning(
                "[%s] cutoff_persistence ignorée pour %s : source absente.",
                data.zone,
                alias,
            )
            continue

        source = data.covariates[alias]
        frame = cutoff_persistence_frame(
            source,
            data.model_context_covariates.index,
            cutoff_local_time=cutoff_time,
            max_age_hours=max_age_hours,
        )

        value_column = f"known_{alias}_cutoff_persistence"
        data.model_context_covariates[value_column] = frame["value"]
        if value_column not in data.known_future_columns:
            data.known_future_columns.append(value_column)
        added.append(value_column)

        age_column = None
        if include_age:
            age_column = f"known_{alias}_cutoff_age_hours"
            data.model_context_covariates[age_column] = frame["age_hours"]
            if age_column not in data.known_future_columns:
                data.known_future_columns.append(age_column)
            added.append(age_column)

        attached[alias] = {
            "cutoff_local_time": cutoff_time,
            "max_age_hours": max_age_hours,
            "value_column": value_column,
            "age_column": age_column,
        }

        coverage = float(frame["value"].notna().mean())
        data.coverage = pd.concat(
            [
                data.coverage,
                pd.DataFrame(
                    [
                        {
                            "zone": data.zone,
                            "alias": value_column,
                            "series": alias,
                            "coverage_exact": coverage,
                            "coverage_after_fill": coverage,
                            "missing_after_fill": int(frame["value"].isna().sum()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        manifest_rows = [
            {
                "alias": value_column,
                "role": "known_future_covariate",
                "series": alias,
                "description": (
                    "Dernière valeur horodatée disponible au cutoff "
                    f"D-1 {cutoff_time}, constante sur D."
                ),
                "future_strategies": "cutoff_persistence",
                "known_future": True,
                "source": "derived_from_past_covariate",
            }
        ]
        if age_column:
            manifest_rows.append(
                {
                    "alias": age_column,
                    "role": "known_future_covariate",
                    "series": alias,
                    "description": (
                        "Âge en heures de la valeur retenue au cutoff "
                        f"D-1 {cutoff_time}."
                    ),
                    "future_strategies": "cutoff_age_hours",
                    "known_future": True,
                    "source": "derived_from_past_covariate",
                }
            )
        data.input_manifest = pd.concat(
            [data.input_manifest, pd.DataFrame(manifest_rows)],
            ignore_index=True,
        )

    if attached:
        data.cutoff_persistence_specs = attached
        data.model_context_covariates = data.model_context_covariates.astype(
            np.float32
        )
        data.diagnostics["cutoff_persistence"] = {
            "enabled": True,
            "aliases": attached,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        data.coverage.to_csv(output_dir / "input_coverage.csv", index=False)
        data.input_manifest.to_csv(
            output_dir / "input_manifest.csv", index=False
        )
        data.model_context_covariates.reset_index(names="timestamp").to_csv(
            output_dir / "model_covariates_with_future.csv.gz",
            index=False,
            compression="gzip",
        )
        LOGGER.info(
            "[%s] cutoff_persistence activée : %s",
            data.zone,
            ", ".join(added),
        )

    return added


def make_prepare_zone_data_with_cutoff_persistence(
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
        add_cutoff_persistence_features(data, config, output_dir)
        return data

    return wrapped


def make_future_proxy_frame_with_cutoff_persistence(
    base_future_proxy: Callable[
        [ZoneData, pd.DatetimeIndex, int], pd.DataFrame
    ],
) -> Callable[[ZoneData, pd.DatetimeIndex, int], pd.DataFrame]:
    def wrapped(
        data: ZoneData,
        future_index: pd.DatetimeIndex,
        origin_position: int,
    ) -> pd.DataFrame:
        frame = base_future_proxy(data, future_index, origin_position)
        specs = getattr(data, "cutoff_persistence_specs", {})
        if not specs:
            return frame

        for alias, settings in specs.items():
            if alias not in data.covariates:
                continue
            computed = cutoff_persistence_frame(
                data.covariates[alias],
                future_index,
                cutoff_local_time=settings["cutoff_local_time"],
                max_age_hours=settings["max_age_hours"],
            )
            frame[settings["value_column"]] = computed["value"].to_numpy(
                dtype=np.float32
            )
            age_column = settings.get("age_column")
            if age_column:
                frame[age_column] = computed["age_hours"].to_numpy(
                    dtype=np.float32
                )

        for column in data.known_future_columns:
            if column not in frame:
                frame[column] = np.nan
        return frame[data.known_future_columns].astype(np.float32)

    return wrapped
