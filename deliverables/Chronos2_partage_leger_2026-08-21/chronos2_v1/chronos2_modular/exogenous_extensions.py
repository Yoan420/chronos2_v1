from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get


def _settings(config: Mapping[str, Any], zone: str) -> dict[str, Any]:
    base = dict(deep_get(config, "exogenous_extensions", {}) or {})
    override = deep_get(config, f"zones.{zone}.exogenous_extensions", {}) or {}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _add_derived_with_lags(
    data: ZoneData,
    alias: str,
    values: pd.Series,
    description: str,
    lags: tuple[int, ...] = (24, 168),
) -> None:
    history = pd.to_numeric(
        values.reindex(data.target.index), errors="coerce"
    ).astype(np.float32)
    data.covariates[alias] = history
    data.model_context_covariates.loc[data.target.index, alias] = history
    future_index = data.model_context_covariates.index.difference(
        data.target.index
    )
    data.model_context_covariates.loc[future_index, alias] = np.nan

    expanded = history.reindex(data.model_context_covariates.index)
    for lag in lags:
        column = f"known_{alias}_lag{lag}"
        data.model_context_covariates[column] = expanded.shift(lag)
        _append_unique(data.known_future_columns, column)

    coverage = float(history.notna().mean())
    data.coverage = pd.concat(
        [
            data.coverage,
            pd.DataFrame(
                [{
                    "zone": data.zone,
                    "alias": alias,
                    "series": "derived",
                    "coverage_exact": coverage,
                    "coverage_after_fill": coverage,
                    "missing_after_fill": int(history.isna().sum()),
                }]
            ),
        ],
        ignore_index=True,
    )
    data.input_manifest = pd.concat(
        [
            data.input_manifest,
            pd.DataFrame(
                [{
                    "alias": alias,
                    "role": "derived_past_covariate",
                    "series": "derived",
                    "description": description,
                    "future_strategies": ", ".join(
                        f"lag{lag}" for lag in lags
                    ),
                    "known_future": False,
                    "source": "derived",
                }]
            ),
        ],
        ignore_index=True,
    )


def add_neighbour_price_features(
    data: ZoneData,
    config: Mapping[str, Any],
) -> list[str]:
    settings = _settings(config, data.zone).get("neighbour_prices", {}) or {}
    if not settings.get("enabled", False):
        return []

    configured = settings.get("aliases", {}) or {}
    aliases = {
        str(country).upper(): str(alias)
        for country, alias in configured.items()
        if str(alias) in data.covariates
    }
    if not aliases:
        LOGGER.warning("[%s] Aucun prix voisin disponible.", data.zone)
        return []

    target = pd.to_numeric(data.target, errors="coerce")
    neighbours = pd.DataFrame(
        {
            country: pd.to_numeric(
                data.covariates[alias], errors="coerce"
            )
            for country, alias in aliases.items()
        },
        index=data.target.index,
    )

    added: list[str] = []
    for country in aliases:
        alias = f"fr_{country.lower()}_price_spread"
        _add_derived_with_lags(
            data,
            alias,
            target - neighbours[country],
            f"Spread Day-Ahead FR moins {country}.",
        )
        added.append(alias)

    if settings.get("include_aggregates", True):
        mean = neighbours.mean(axis=1)
        aggregate = {
            "neighbour_price_mean": mean,
            "neighbour_price_max": neighbours.max(axis=1),
            "neighbour_price_min": neighbours.min(axis=1),
            "neighbour_price_dispersion": neighbours.std(axis=1),
            "fr_neighbour_spread_mean": target - mean,
            "fr_neighbour_spread_tightest": target - neighbours.max(axis=1),
            "fr_neighbour_spread_widest": target - neighbours.min(axis=1),
        }
        for alias, values in aggregate.items():
            _add_derived_with_lags(
                data,
                alias,
                values,
                "Facteur agrégé causal des prix Day-Ahead voisins.",
            )
            added.append(alias)
    return added


def _holiday_set(country: str, years: list[int]) -> set:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "Installe le package holidays : python -m pip install holidays"
        ) from exc
    return set(holidays.country_holidays(country, years=years).keys())


def rich_calendar_frame(
    index: pd.DatetimeIndex,
    countries: list[str],
    primary_country: str = "FR",
) -> pd.DataFrame:
    index = pd.DatetimeIndex(index)
    local_naive = index.tz_localize(None) if index.tz is not None else index
    dates = pd.Series(local_naive.date, index=index)
    years = sorted(
        set(index.year.tolist())
        | {int(index.min().year) - 1, int(index.max().year) + 1}
    )
    masks: dict[str, pd.Series] = {}
    holiday_sets: dict[str, set] = {}
    for country in countries:
        code = country.upper()
        holiday_sets[code] = _holiday_set(code, years)
        masks[code] = dates.isin(holiday_sets[code]).astype(np.float32)

    result = pd.DataFrame(index=index)
    for country, mask in masks.items():
        result[f"known_cal_holiday_{country.lower()}_oracle"] = mask

    primary = primary_country.upper()
    primary_holidays = holiday_sets.get(primary, set())
    previous_date = dates.map(lambda d: d - pd.Timedelta(days=1))
    next_date = dates.map(lambda d: d + pd.Timedelta(days=1))
    is_holiday = dates.isin(primary_holidays).to_numpy()
    previous_holiday = previous_date.isin(primary_holidays).to_numpy()
    next_holiday = next_date.isin(primary_holidays).to_numpy()
    dow = index.dayofweek
    is_weekday = dow < 5
    previous_weekend = np.isin((dow - 1) % 7, [5, 6])
    next_weekend = np.isin((dow + 1) % 7, [5, 6])
    bridge = (
        is_weekday
        & ~is_holiday
        & (
            (previous_holiday & next_weekend)
            | (next_holiday & previous_weekend)
        )
    )

    holiday_count = np.sum(
        [mask.to_numpy() for mask in masks.values()], axis=0
    ).astype(np.float32)
    primary_mask = masks.get(primary, pd.Series(0.0, index=index)).to_numpy()
    offsets = np.array(
        [ts.utcoffset().total_seconds() / 3600.0 for ts in index],
        dtype=np.float32,
    )
    daily_offsets = pd.Series(offsets, index=index).groupby(
        index.normalize()
    ).first()
    days = pd.DatetimeIndex(index.normalize())
    prev_offsets = np.array(
        [
            daily_offsets.get(day - pd.Timedelta(days=1), np.nan)
            for day in days
        ]
    )
    next_offsets = np.array(
        [
            daily_offsets.get(day + pd.Timedelta(days=1), np.nan)
            for day in days
        ]
    )

    result["known_cal_holiday_count_oracle"] = holiday_count
    result["known_cal_any_neighbour_holiday_oracle"] = (
        holiday_count - primary_mask > 0
    ).astype(np.float32)
    result["known_cal_bridge_day_fr_oracle"] = bridge.astype(np.float32)
    result["known_cal_day_before_holiday_fr_oracle"] = (
        next_holiday.astype(np.float32)
    )
    result["known_cal_day_after_holiday_fr_oracle"] = (
        previous_holiday.astype(np.float32)
    )
    result["known_cal_is_business_day_fr_oracle"] = (
        is_weekday & ~is_holiday
    ).astype(np.float32)
    result["known_cal_morning_peak_oracle"] = np.isin(
        index.hour, [7, 8, 9, 10]
    ).astype(np.float32)
    result["known_cal_evening_peak_oracle"] = np.isin(
        index.hour, [17, 18, 19, 20]
    ).astype(np.float32)
    result["known_cal_night_offpeak_oracle"] = (
        (index.hour <= 5) | (index.hour >= 23)
    ).astype(np.float32)
    result["known_cal_month_end_oracle"] = index.is_month_end.astype(
        np.float32
    )
    result["known_cal_quarter_end_oracle"] = index.is_quarter_end.astype(
        np.float32
    )
    result["known_cal_dst_oracle"] = (offsets > 1.0).astype(np.float32)
    result["known_cal_dst_transition_day_oracle"] = (
        (offsets != prev_offsets) | (offsets != next_offsets)
    ).astype(np.float32)
    return result.astype(np.float32)


def add_rich_calendar_features(
    data: ZoneData,
    config: Mapping[str, Any],
) -> list[str]:
    settings = _settings(config, data.zone).get("rich_calendar", {}) or {}
    if not settings.get("enabled", False):
        return []
    frame = rich_calendar_frame(
        data.model_context_covariates.index,
        [str(x).upper() for x in settings.get(
            "countries", ["FR", "DE", "BE", "NL", "ES"]
        )],
        str(settings.get("primary_country", "FR")).upper(),
    )
    for column in frame:
        data.model_context_covariates[column] = frame[column]
        _append_unique(data.known_future_columns, column)
    data.input_manifest = pd.concat(
        [
            data.input_manifest,
            pd.DataFrame(
                [{
                    "alias": column,
                    "role": "known_future_covariate",
                    "series": "deterministic_calendar",
                    "description": "Calendrier déterministe connu à l'avance.",
                    "future_strategies": "oracle",
                    "known_future": True,
                    "source": "derived_calendar",
                } for column in frame.columns]
            ),
        ],
        ignore_index=True,
    )
    return list(frame.columns)


def _rewrite_outputs(data: ZoneData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.coverage.to_csv(output_dir / "input_coverage.csv", index=False)
    data.input_manifest.to_csv(
        output_dir / "input_manifest.csv", index=False
    )
    pd.concat(
        [data.target.rename("target"), data.covariates], axis=1
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


def augment_zone_data(
    data: ZoneData,
    zone: ZoneConfig,
    config: Mapping[str, Any],
    output_dir: Path,
) -> ZoneData:
    if not _settings(config, zone.zone).get("enabled", False):
        return data
    neighbour = add_neighbour_price_features(data, config)
    calendar = add_rich_calendar_features(data, config)
    data.model_context_covariates = (
        data.model_context_covariates.astype(np.float32)
    )
    data.diagnostics["exogenous_extensions"] = {
        "neighbour_price_columns": neighbour,
        "rich_calendar_columns": calendar,
    }
    _rewrite_outputs(data, output_dir)
    LOGGER.info(
        "[%s] Extensions exogènes : %d prix/spreads et %d calendaires.",
        data.zone, len(neighbour), len(calendar)
    )
    return data


def make_prepare_zone_data_with_extensions(
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
            zone, config, config_dir, refresh, output_dir
        )
        return augment_zone_data(
            data, zone, config, output_dir
        )
    return wrapped
