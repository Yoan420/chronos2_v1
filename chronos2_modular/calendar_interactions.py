from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get


DEFAULT_NEIGHBOUR_COUNTRIES = ("DE", "BE", "NL", "ES")


def _settings(
    config: Mapping[str, Any],
    zone_name: str,
) -> dict[str, Any]:
    global_settings = (
        deep_get(
            config,
            "exogenous_extensions.calendar_interactions",
            {},
        )
        or {}
    )
    zone_settings = (
        deep_get(
            config,
            (
                f"zones.{zone_name}.exogenous_extensions."
                "calendar_interactions"
            ),
            {},
        )
        or {}
    )
    return {**global_settings, **zone_settings}


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _required_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(
            f"Colonne requise absente pour les interactions : {column}"
        )
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).astype(np.float32)


def _oracle_column(alias: str) -> str:
    return f"known_{alias}_oracle"


def _neighbour_holiday_count(
    frame: pd.DataFrame,
    countries: Sequence[str],
    primary_country: str,
) -> pd.Series:
    primary = primary_country.upper()
    columns = [
        f"known_cal_holiday_{str(country).lower()}_oracle"
        for country in countries
        if str(country).upper() != primary
    ]
    missing = [
        column for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            "Colonnes calendaires voisines absentes : "
            + ", ".join(missing)
        )

    if not columns:
        return pd.Series(
            0.0,
            index=frame.index,
            dtype=np.float32,
        )

    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, min_count=1)
        .astype(np.float32)
    )


def _coverage_row(
    zone: str,
    column: str,
    values: pd.Series,
    future_index: pd.DatetimeIndex,
) -> dict[str, Any]:
    historical_values = values.loc[
        ~values.index.isin(future_index)
    ]
    future_values = values.reindex(future_index)
    return {
        "zone": zone,
        "column": column,
        "historical_non_missing": int(
            historical_values.notna().sum()
        ),
        "historical_expected": int(len(historical_values)),
        "historical_coverage": (
            float(historical_values.notna().mean())
            if len(historical_values)
            else np.nan
        ),
        "future_non_missing": int(future_values.notna().sum()),
        "future_expected": int(len(future_values)),
        "future_coverage": (
            float(future_values.notna().mean())
            if len(future_values)
            else np.nan
        ),
    }


def add_calendar_interaction_features(
    data: ZoneData,
    config: Mapping[str, Any],
    output_dir: Path | None = None,
) -> list[str]:
    settings = _settings(config, data.zone)
    if not bool(settings.get("enabled", False)):
        return []

    residual_alias = str(
        settings.get(
            "residual_load_alias",
            "fr_residual_load_fcst",
        )
    )
    nuclear_alias = str(
        settings.get(
            "nuclear_alias",
            "fr_nuclear_generation_fcst",
        )
    )
    primary_country = str(
        settings.get("primary_country", "FR")
    ).upper()
    neighbour_countries = [
        str(country).upper()
        for country in settings.get(
            "neighbour_countries",
            DEFAULT_NEIGHBOUR_COUNTRIES,
        )
    ]

    frame = data.model_context_covariates
    residual = _required_column(
        frame,
        _oracle_column(residual_alias),
    )
    nuclear = _required_column(
        frame,
        _oracle_column(nuclear_alias),
    )

    morning_peak = _required_column(
        frame,
        "known_cal_morning_peak_oracle",
    )
    evening_peak = _required_column(
        frame,
        "known_cal_evening_peak_oracle",
    )
    public_holiday = _required_column(
        frame,
        (
            f"known_cal_holiday_"
            f"{primary_country.lower()}_oracle"
        ),
    )
    weekend = _required_column(
        frame,
        "known_is_weekend",
    )
    bridge_day = _required_column(
        frame,
        (
            f"known_cal_bridge_day_"
            f"{primary_country.lower()}_oracle"
        ),
    )
    neighbour_holiday_count = _neighbour_holiday_count(
        frame,
        neighbour_countries,
        primary_country,
    )

    residual_after_nuclear = residual - nuclear

    features: dict[str, tuple[pd.Series, str]] = {
        "known_interaction_residual_load_morning_peak_oracle": (
            residual * morning_peak,
            "Charge résiduelle PIT × pointe du matin.",
        ),
        "known_interaction_residual_load_evening_peak_oracle": (
            residual * evening_peak,
            "Charge résiduelle PIT × pointe du soir.",
        ),
        "known_interaction_residual_load_public_holiday_oracle": (
            residual * public_holiday,
            "Charge résiduelle PIT × jour férié français.",
        ),
        "known_interaction_residual_load_weekend_oracle": (
            residual * weekend,
            "Charge résiduelle PIT × week-end.",
        ),
        "known_interaction_residual_load_bridge_day_oracle": (
            residual * bridge_day,
            "Charge résiduelle PIT × jour de pont français.",
        ),
        (
            "known_interaction_residual_load_"
            "neighbour_holiday_count_oracle"
        ): (
            residual * neighbour_holiday_count,
            (
                "Charge résiduelle PIT × nombre de pays voisins "
                "en jour férié."
            ),
        ),
        "known_residual_load_after_nuclear_oracle": (
            residual_after_nuclear,
            (
                "Charge résiduelle PIT moins génération nucléaire "
                "prévue PIT."
            ),
        ),
        (
            "known_interaction_residual_after_nuclear_"
            "morning_peak_oracle"
        ): (
            residual_after_nuclear * morning_peak,
            (
                "Charge résiduelle après nucléaire × "
                "pointe du matin."
            ),
        ),
        (
            "known_interaction_residual_after_nuclear_"
            "evening_peak_oracle"
        ): (
            residual_after_nuclear * evening_peak,
            (
                "Charge résiduelle après nucléaire × "
                "pointe du soir."
            ),
        ),
        (
            "known_interaction_residual_after_nuclear_"
            "public_holiday_oracle"
        ): (
            residual_after_nuclear * public_holiday,
            (
                "Charge résiduelle après nucléaire × "
                "jour férié français."
            ),
        ),
    }

    future_index = frame.index.difference(data.target.index)
    coverage_rows: list[dict[str, Any]] = []

    for column, (values, _) in features.items():
        values = pd.to_numeric(
            values.reindex(frame.index),
            errors="coerce",
        ).astype(np.float32)
        frame[column] = values
        _append_unique(data.known_future_columns, column)
        coverage_rows.append(
            _coverage_row(
                data.zone,
                column,
                values,
                future_index,
            )
        )

    manifest_rows = [
        {
            "alias": column,
            "role": "known_future_covariate",
            "series": "derived_pit_calendar_interaction",
            "description": description,
            "future_strategies": "oracle",
            "known_future": True,
            "source": "derived_pit_interaction",
        }
        for column, (_, description) in features.items()
    ]
    data.input_manifest = pd.concat(
        [
            data.input_manifest,
            pd.DataFrame(manifest_rows),
        ],
        ignore_index=True,
    )

    coverage = pd.DataFrame(coverage_rows)
    require_complete = bool(
        settings.get("require_complete_future", True)
    )
    incomplete = coverage.loc[
        coverage["future_non_missing"]
        < coverage["future_expected"]
    ]
    if require_complete and not incomplete.empty:
        details = ", ".join(
            (
                f"{row.column}="
                f"{int(row.future_non_missing)}/"
                f"{int(row.future_expected)}"
            )
            for row in incomplete.itertuples()
        )
        raise ValueError(
            "Interactions calendaires incomplètes sur l'horizon "
            f"live : {details}"
        )

    data.model_context_covariates = frame.astype(np.float32)
    data.diagnostics["calendar_interactions"] = {
        "enabled": True,
        "residual_load_source": _oracle_column(residual_alias),
        "nuclear_source": _oracle_column(nuclear_alias),
        "primary_country": primary_country,
        "neighbour_countries": neighbour_countries,
        "columns": list(features),
        "point_in_time": True,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(
            output_dir / "calendar_interaction_coverage.csv",
            index=False,
        )
        data.input_manifest.to_csv(
            output_dir / "input_manifest.csv",
            index=False,
        )
        data.model_context_covariates.reset_index(
            names="timestamp"
        ).to_csv(
            output_dir / "model_covariates_with_future.csv.gz",
            index=False,
            compression="gzip",
        )

    LOGGER.info(
        "[%s] Interactions calendaires PIT : %d colonnes ajoutées.",
        data.zone,
        len(features),
    )
    return list(features)


def make_prepare_zone_data_with_calendar_interactions(
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
        add_calendar_interaction_features(
            data,
            config,
            output_dir,
        )
        return data

    return wrapped
